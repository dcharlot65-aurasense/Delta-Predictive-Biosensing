//! Gated fusion: Adaptive modality selection based on input quality
//!
//! Uses gating mechanisms to dynamically weight modalities based on
//! their reliability and informativeness for the current input.

use super::{FusionNetwork, FusionConfig, Modality};
use crate::layers::SpikingLayer;
use crate::{SpikeTensor, SpikingLinear, SNNResult, SNNError, NeuronParams};
use ndarray::{Array2, Array3, Axis};
use std::collections::HashMap;

/// Gated fusion network with adaptive modality weighting
pub struct GatedFusionSNN {
    config: FusionConfig,
    /// Feature extractors per modality
    feature_extractors: HashMap<Modality, SpikingLinear>,
    /// Gate networks (compute modality importance)
    gate_networks: HashMap<Modality, SpikingLinear>,
    /// Fusion layers
    fusion_layers: Vec<SpikingLinear>,
    neuron_params: NeuronParams,
}

impl GatedFusionSNN {
    /// Create a new gated fusion network
    pub fn new(config: FusionConfig) -> SNNResult<Self> {
        let neuron_params = NeuronParams::default();
        let dt = 1.0;
        let adaptive = false;

        // Create feature extractor for each modality
        let mut feature_extractors = HashMap::new();
        let mut gate_networks = HashMap::new();

        for modality in &config.modalities {
            let input_size = modality.default_feature_dim();

            // Feature extractor
            feature_extractors.insert(
                *modality,
                SpikingLinear::new(
                    input_size,
                    config.hidden_size,
                    true,
                    neuron_params.clone(),
                    dt,
                    adaptive,
                ),
            );

            // Gate network (outputs single value per timestep)
            gate_networks.insert(
                *modality,
                SpikingLinear::new(
                    input_size,
                    1,  // Gate value
                    true,
                    neuron_params.clone(),
                    dt,
                    adaptive,
                ),
            );
        }

        // Fusion layers
        let mut fusion_layers = Vec::new();
        let mut prev_size = config.hidden_size;  // After gating, we sum across modalities

        for i in 0..config.num_layers {
            let next_size = if i == config.num_layers - 1 {
                config.output_size
            } else {
                config.hidden_size
            };

            fusion_layers.push(SpikingLinear::new(prev_size, next_size, true, neuron_params.clone(), dt, adaptive));

            prev_size = next_size;
        }

        Ok(Self {
            config,
            feature_extractors,
            gate_networks,
            fusion_layers,
            neuron_params,
        })
    }

    /// Compute gate values for each modality
    fn compute_gates(
        &mut self,
        inputs: &HashMap<Modality, SpikeTensor>,
    ) -> SNNResult<HashMap<Modality, Array3<f32>>> {
        let mut gates = HashMap::new();

        for (modality, input) in inputs {
            if let Some(gate_net) = self.gate_networks.get_mut(modality) {
                let gate_output = gate_net.forward(input)?;

                match gate_output.data {
                    crate::SpikeRepresentation::Dense(gate_arr) => {
                        // Apply sigmoid-like activation (spike rate as proxy)
                        // In a full implementation, we'd use proper sigmoid surrogate
                        let normalized = gate_arr.mapv(|x| x.clamp(0.0, 1.0));
                        gates.insert(*modality, normalized);
                    }
                    _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
                }
            }
        }

        Ok(gates)
    }

    /// Get feature extractors for testing
    pub fn feature_extractors(&self) -> &HashMap<Modality, SpikingLinear> {
        &self.feature_extractors
    }

    /// Get gate networks for testing
    pub fn gate_networks(&self) -> &HashMap<Modality, SpikingLinear> {
        &self.gate_networks
    }
}

impl FusionNetwork for GatedFusionSNN {
    fn name(&self) -> &str {
        "GatedFusionSNN"
    }

    fn modalities(&self) -> Vec<Modality> {
        self.config.modalities.clone()
    }

    fn forward(&mut self, inputs: &HashMap<Modality, SpikeTensor>) -> SNNResult<SpikeTensor> {
        if inputs.is_empty() {
            return Err(SNNError::InvalidConfig("No inputs provided".to_string()));
        }

        // Compute gate values for each modality
        let gates = self.compute_gates(inputs)?;

        // Extract features and apply gating
        let mut gated_features = Vec::new();

        for modality in &self.config.modalities {
            if let Some(input) = inputs.get(modality)
                && let (Some(extractor), Some(gate)) = (
                    self.feature_extractors.get_mut(modality),
                    gates.get(modality),
                ) {
                    let features = extractor.forward(input)?;

                    // Apply gate (element-wise multiplication with broadcasting)
                    match features.data {
                        crate::SpikeRepresentation::Dense(feat_arr) => {
                            // Broadcast gate across features
                            let gated = &feat_arr * gate;
                            gated_features.push(SpikeTensor::from_dense(
                                gated,
                                features.requires_grad,
                            ));
                        }
                        _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
                    }
                }
        }

        if gated_features.is_empty() {
            return Err(SNNError::InvalidConfig("No gated features produced".to_string()));
        }

        // Sum gated features (weighted combination)
        let first = &gated_features[0];
        let mut fused = match &first.data {
            crate::SpikeRepresentation::Dense(arr) => arr.clone(),
            _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
        };

        for gated in &gated_features[1..] {
            match &gated.data {
                crate::SpikeRepresentation::Dense(arr) => {
                    fused += arr;
                }
                _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
            }
        }

        let mut x = SpikeTensor::from_dense(fused, first.requires_grad);

        // Apply fusion layers
        for layer in &mut self.fusion_layers {
            x = layer.forward(&x)?;
        }

        Ok(x)
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Feature extractor parameters
        for extractor in self.feature_extractors.values() {
            total += extractor.parameters().iter()
                .map(|p| p.len())
                .sum::<usize>();
        }

        // Gate network parameters
        for gate in self.gate_networks.values() {
            total += gate.parameters().iter()
                .map(|p| p.len())
                .sum::<usize>();
        }

        // Fusion layer parameters
        for layer in &self.fusion_layers {
            total += layer.parameters().iter()
                .map(|p| p.len())
                .sum::<usize>();
        }

        total
    }

    fn reset_state(&mut self) {
        for extractor in self.feature_extractors.values_mut() {
            extractor.reset_state();
        }

        for gate in self.gate_networks.values_mut() {
            gate.reset_state();
        }

        for layer in &mut self.fusion_layers {
            layer.reset_state();
        }
    }

    fn handles_missing_modalities(&self) -> bool {
        true  // Gated fusion naturally handles missing modalities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_gated_fusion_creation() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 128,
            num_layers: 3,
            output_size: 10,
            ..Default::default()
        };

        let network = GatedFusionSNN::new(config).unwrap();
        assert_eq!(network.name(), "GatedFusionSNN");
        assert_eq!(network.feature_extractors().len(), 2);
        assert_eq!(network.gate_networks().len(), 2);
    }

    #[test]
    fn test_gated_fusion_forward() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = GatedFusionSNN::new(config).unwrap();

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
    fn test_gated_fusion_single_modality() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = GatedFusionSNN::new(config).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::ones((1, 10, 128)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gated_fusion_parameters() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact],
            hidden_size: 32,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let network = GatedFusionSNN::new(config).unwrap();
        let num_params = network.num_parameters();
        assert!(num_params > 0);
    }

    #[test]
    fn test_gated_fusion_reset() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 64,
            num_layers: 3,
            output_size: 10,
            ..Default::default()
        };

        let mut network = GatedFusionSNN::new(config).unwrap();
        network.reset_state();
        // Should not panic
    }

    #[test]
    fn test_gated_fusion_all_modalities() {
        let config = FusionConfig {
            modalities: Modality::all(),
            hidden_size: 128,
            num_layers: 3,
            output_size: 20,
            ..Default::default()
        };

        let mut network = GatedFusionSNN::new(config).unwrap();

        let mut inputs = HashMap::new();
        for modality in Modality::all() {
            inputs.insert(
                modality,
                SpikeTensor::from_dense(
                    Array3::<f32>::zeros((1, 10, modality.default_feature_dim())),
                    false,
                ),
            );
        }

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }
}
