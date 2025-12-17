//! Early fusion: Concatenate spike trains at feature level
//!
//! This is the simplest fusion approach where spike trains from all modalities
//! are concatenated and processed by a single shared SNN.

use super::{FusionNetwork, FusionConfig, Modality, concatenate_spikes};
use crate::{SpikeTensor, SpikingLinear, SNNResult, SNNError, NeuronParams};
use crate::layers::SpikingLayer;
use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// Early fusion network that concatenates all modality spike trains
pub struct EarlyFusionSNN {
    config: FusionConfig,
    layers: Vec<SpikingLinear>,
    neuron_params: NeuronParams,
}

impl EarlyFusionSNN {
    /// Create a new early fusion network
    pub fn new(config: FusionConfig) -> SNNResult<Self> {
        let neuron_params = NeuronParams::default();
        let dt = 1.0;  // Default time step
        let adaptive = false;  // Non-adaptive neurons by default

        // Calculate input size (sum of all modality dimensions)
        let input_size: usize = config.modalities.iter()
            .map(|m| m.default_feature_dim())
            .sum();

        // Build layers
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
                true,  // use_bias
                neuron_params.clone(),
                dt,
                adaptive,
            ));

            prev_size = next_size;
        }

        Ok(Self {
            config,
            layers,
            neuron_params,
        })
    }

    /// Get layer reference for testing
    pub fn layers(&self) -> &[SpikingLinear] {
        &self.layers
    }
}

impl FusionNetwork for EarlyFusionSNN {
    fn name(&self) -> &str {
        "EarlyFusionSNN"
    }

    fn modalities(&self) -> Vec<Modality> {
        self.config.modalities.clone()
    }

    fn forward(&mut self, inputs: &HashMap<Modality, SpikeTensor>) -> SNNResult<SpikeTensor> {
        // Validate all modalities are present
        for modality in &self.config.modalities {
            if !inputs.contains_key(modality) {
                return Err(SNNError::InvalidConfig(
                    format!("Missing modality: {:?}", modality)
                ));
            }
        }

        // Concatenate all spike trains
        let spike_refs: Vec<&SpikeTensor> = self.config.modalities.iter()
            .filter_map(|m| inputs.get(m))
            .collect();

        let mut x = concatenate_spikes(&spike_refs)?;

        // Forward through layers
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }

        Ok(x)
    }

    fn num_parameters(&self) -> usize {
        self.layers.iter()
            .map(|layer| {
                layer.parameters().iter()
                    .map(|p| p.len())
                    .sum::<usize>()
            })
            .sum()
    }

    fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }

    fn handles_missing_modalities(&self) -> bool {
        false  // Early fusion requires all modalities
    }

    fn required_modalities(&self) -> Vec<Modality> {
        self.config.modalities.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_early_fusion_creation() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 128,
            num_layers: 2,
            output_size: 10,
            ..Default::default()
        };

        let network = EarlyFusionSNN::new(config).unwrap();
        assert_eq!(network.name(), "EarlyFusionSNN");
        assert_eq!(network.layers().len(), 2);
    }

    #[test]
    fn test_early_fusion_forward() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = EarlyFusionSNN::new(config).unwrap();

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
                assert_eq!(arr.dim().0, 1);  // batch
                assert_eq!(arr.dim().1, 10); // time
                assert_eq!(arr.dim().2, 5);  // output features
            }
            _ => panic!("Expected dense output"),
        }
    }

    #[test]
    fn test_early_fusion_missing_modality() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = EarlyFusionSNN::new(config).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );
        // Missing Hand modality

        let result = network.forward(&inputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_early_fusion_parameters() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact],
            hidden_size: 32,
            num_layers: 1,
            output_size: 5,
            ..Default::default()
        };

        let network = EarlyFusionSNN::new(config).unwrap();
        let num_params = network.num_parameters();
        assert!(num_params > 0);
    }

    #[test]
    fn test_early_fusion_reset() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact],
            hidden_size: 32,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = EarlyFusionSNN::new(config).unwrap();
        network.reset_state();
        // Should not panic
    }
}
