//! Late fusion: Process each modality separately, combine at decision level
//!
//! Each modality has its own dedicated SNN. Outputs are combined via
//! voting, averaging, or learned weights.

use super::{FusionNetwork, FusionConfig, Modality, average_spikes};
use crate::{SpikeTensor, SpikingLinear, SNNResult, SNNError, NeuronParams};
use crate::layers::SpikingLayer;
use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Fusion strategy for combining modality-specific outputs
#[derive(Debug, Clone, Copy)]
pub enum LateFusionStrategy {
    /// Simple averaging
    Average,
    /// Weighted average (learned weights)
    WeightedAverage,
    /// Max voting
    MaxVoting,
}

/// Late fusion network with separate processing per modality
pub struct LateFusionSNN {
    config: FusionConfig,
    /// One network per modality
    modality_networks: HashMap<Modality, Vec<SpikingLinear>>,
    /// Fusion layer combines modality outputs
    fusion_layer: Option<SpikingLinear>,
    /// Weights for weighted average (if applicable)
    modality_weights: HashMap<Modality, f32>,
    fusion_strategy: LateFusionStrategy,
    neuron_params: NeuronParams,
}

impl LateFusionSNN {
    /// Create a new late fusion network
    pub fn new(config: FusionConfig, strategy: LateFusionStrategy) -> SNNResult<Self> {
        let neuron_params = NeuronParams::default();
        let dt = 1.0;
        let adaptive = false;
        let mut modality_networks = HashMap::new();

        // Create separate network for each modality
        for modality in &config.modalities {
            let input_size = modality.default_feature_dim();
            let mut layers = Vec::new();
            let mut prev_size = input_size;

            for i in 0..config.num_layers {
                let next_size = if i == config.num_layers - 1 {
                    config.output_size
                } else {
                    config.hidden_size
                };

                layers.push(SpikingLinear::new(
                    prev_size,
                    next_size,
                    true,
                    neuron_params.clone(),
                    dt,
                    adaptive,
                ));

                prev_size = next_size;
            }

            modality_networks.insert(*modality, layers);
        }

        // Initialize equal weights
        let weight = 1.0 / config.modalities.len() as f32;
        let modality_weights: HashMap<_, _> = config.modalities.iter()
            .map(|m| (*m, weight))
            .collect();

        // Optional fusion layer for learned combination
        let fusion_layer = match strategy {
            LateFusionStrategy::WeightedAverage => {
                let num_modalities = config.modalities.len();
                Some(SpikingLinear::new(
                    num_modalities * config.output_size,
                    config.output_size,
                    true,
                    neuron_params.clone(),
                    dt,
                    adaptive,
                ))
            }
            _ => None,
        };

        Ok(Self {
            config,
            modality_networks,
            fusion_layer,
            modality_weights,
            fusion_strategy: strategy,
            neuron_params,
        })
    }

    /// Set modality weight (for weighted average)
    pub fn set_modality_weight(&mut self, modality: Modality, weight: f32) {
        self.modality_weights.insert(modality, weight);
    }

    /// Get modality networks
    pub fn modality_networks(&self) -> &HashMap<Modality, Vec<SpikingLinear>> {
        &self.modality_networks
    }
}

impl FusionNetwork for LateFusionSNN {
    fn name(&self) -> &str {
        "LateFusionSNN"
    }

    fn modalities(&self) -> Vec<Modality> {
        self.config.modalities.clone()
    }

    fn forward(&mut self, inputs: &HashMap<Modality, SpikeTensor>) -> SNNResult<SpikeTensor> {
        let mut modality_outputs = Vec::new();

        // Process each modality independently
        for modality in &self.config.modalities {
            if let Some(input) = inputs.get(modality) {
                let mut x = input.clone();

                // Forward through modality-specific network
                if let Some(layers) = self.modality_networks.get_mut(modality) {
                    for layer in layers {
                        x = layer.forward(&x)?;
                    }
                    modality_outputs.push((modality, x));
                }
            }
        }

        if modality_outputs.is_empty() {
            return Err(SNNError::InvalidConfig("No modality inputs provided".to_string()));
        }

        // Combine modality outputs based on strategy
        match self.fusion_strategy {
            LateFusionStrategy::Average => {
                let refs: Vec<&SpikeTensor> = modality_outputs.iter()
                    .map(|(_, t)| t)
                    .collect();
                average_spikes(&refs)
            }
            LateFusionStrategy::WeightedAverage => {
                // Weighted average
                let first = &modality_outputs[0].1;
                let shape = match &first.data {
                    crate::SpikeRepresentation::Dense(arr) => arr.dim(),
                    _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
                };

                let mut weighted_sum = Array3::<f32>::zeros(shape);

                for (modality, output) in &modality_outputs {
                    let weight = self.modality_weights.get(modality).unwrap_or(&1.0);
                    match &output.data {
                        crate::SpikeRepresentation::Dense(arr) => {
                            weighted_sum = weighted_sum + arr * *weight;
                        }
                        _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
                    }
                }

                Ok(SpikeTensor::from_dense(weighted_sum, first.requires_grad))
            }
            LateFusionStrategy::MaxVoting => {
                // Element-wise maximum
                let first = &modality_outputs[0].1;
                let mut max_output = match &first.data {
                    crate::SpikeRepresentation::Dense(arr) => arr.clone(),
                    _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
                };

                for (_, output) in &modality_outputs[1..] {
                    match &output.data {
                        crate::SpikeRepresentation::Dense(arr) => {
                            max_output = ndarray::Zip::from(&max_output)
                                .and(arr)
                                .map_collect(|&a, &b| a.max(b));
                        }
                        _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
                    }
                }

                Ok(SpikeTensor::from_dense(max_output, first.requires_grad))
            }
        }
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        for layers in self.modality_networks.values() {
            total += layers.iter()
                .map(|layer| {
                    layer.parameters().iter()
                        .map(|p| p.len())
                        .sum::<usize>()
                })
                .sum::<usize>();
        }

        if let Some(fusion) = &self.fusion_layer {
            total += fusion.parameters().iter()
                .map(|p| p.len())
                .sum::<usize>();
        }

        total
    }

    fn reset_state(&mut self) {
        for layers in self.modality_networks.values_mut() {
            for layer in layers {
                layer.reset_state();
            }
        }

        if let Some(fusion) = &mut self.fusion_layer {
            fusion.reset_state();
        }
    }

    fn handles_missing_modalities(&self) -> bool {
        true  // Late fusion can handle missing modalities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_late_fusion_creation() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 128,
            num_layers: 2,
            output_size: 10,
            ..Default::default()
        };

        let network = LateFusionSNN::new(config, LateFusionStrategy::Average).unwrap();
        assert_eq!(network.name(), "LateFusionSNN");
        assert_eq!(network.modality_networks.len(), 2);
    }

    #[test]
    fn test_late_fusion_forward_average() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = LateFusionSNN::new(config, LateFusionStrategy::Average).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );
        inputs.insert(
            Modality::Hand,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );

        let output = network.forward(&inputs).unwrap();

        match output.data {
            crate::SpikeRepresentation::Dense(arr) => {
                assert_eq!(arr.dim().2, 5);
            }
            _ => panic!("Expected dense output"),
        }
    }

    #[test]
    fn test_late_fusion_missing_modality_ok() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = LateFusionSNN::new(config, LateFusionStrategy::Average).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );
        // Missing Hand - should still work

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_late_fusion_weighted() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = LateFusionSNN::new(config, LateFusionStrategy::WeightedAverage).unwrap();
        network.set_modality_weight(Modality::Contact, 0.7);
        network.set_modality_weight(Modality::Hand, 0.3);

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::ones((1, 10, 128)), false),
        );
        inputs.insert(
            Modality::Hand,
            SpikeTensor::from_dense(Array3::<f32>::ones((1, 10, 128)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_late_fusion_max_voting() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = LateFusionSNN::new(config, LateFusionStrategy::MaxVoting).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );
        inputs.insert(
            Modality::Hand,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }
}
