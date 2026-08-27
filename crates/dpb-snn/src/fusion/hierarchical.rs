//! Hierarchical fusion: Multi-level integration of modalities
//!
//! Features are fused at multiple levels:
//! - Low-level: Raw spike features
//! - Mid-level: Modality-specific concepts
//! - High-level: Decision fusion
use crate::layers::SpikingLayer;

use super::{FusionNetwork, FusionConfig, Modality, concatenate_spikes};
use crate::{SpikeTensor, SpikingLinear, SNNResult, SNNError, NeuronParams};
use ndarray::Array3;
use std::collections::HashMap;

/// Hierarchical fusion with multi-level integration
// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
pub struct HierarchicalFusionSNN {
    config: FusionConfig,
    /// Low-level processors (per modality)
    low_level: HashMap<Modality, SpikingLinear>,
    /// Mid-level fusion layer
    mid_level: SpikingLinear,
    /// High-level decision layers
    high_level: Vec<SpikingLinear>,
    neuron_params: NeuronParams,
}

impl HierarchicalFusionSNN {
    /// Create a new hierarchical fusion network
    pub fn new(config: FusionConfig) -> SNNResult<Self> {
        let neuron_params = NeuronParams::default();
        let dt = 1.0;
        let adaptive = false;

        // Low-level: Extract features from each modality
        let mut low_level = HashMap::new();
        for modality in &config.modalities {
            let input_size = modality.default_feature_dim();
            low_level.insert(
                *modality,
                SpikingLinear::new(
                    input_size,
                    config.hidden_size / 2,
                    true,
                    neuron_params.clone(),
                    dt,
                    adaptive,
                ),
            );
        }

        // Mid-level: Fuse low-level features
        let num_modalities = config.modalities.len();
        let mid_input_size = num_modalities * (config.hidden_size / 2);
        let mid_level = SpikingLinear::new(
            mid_input_size,
            config.hidden_size,
            true,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        // High-level: Decision layers
        let mut high_level = Vec::new();
        let mut prev_size = config.hidden_size;

        for i in 0..(config.num_layers - 1) {
            let next_size = if i == config.num_layers - 2 {
                config.output_size
            } else {
                config.hidden_size
            };

            high_level.push(SpikingLinear::new(prev_size, next_size, true, neuron_params.clone(), dt, adaptive));

            prev_size = next_size;
        }

        Ok(Self {
            config,
            low_level,
            mid_level,
            high_level,
            neuron_params,
        })
    }

    /// Get low-level processors for testing
    pub fn low_level(&self) -> &HashMap<Modality, SpikingLinear> {
        &self.low_level
    }

    /// Get high-level layers for testing
    pub fn high_level(&self) -> &[SpikingLinear] {
        &self.high_level
    }
}

impl FusionNetwork for HierarchicalFusionSNN {
    fn name(&self) -> &str {
        "HierarchicalFusionSNN"
    }

    fn modalities(&self) -> Vec<Modality> {
        self.config.modalities.clone()
    }

    fn forward(&mut self, inputs: &HashMap<Modality, SpikeTensor>) -> SNNResult<SpikeTensor> {
        // Low-level: Process each modality
        let mut low_features = Vec::new();
        for modality in &self.config.modalities {
            if let Some(input) = inputs.get(modality) {
                if let Some(processor) = self.low_level.get_mut(modality) {
                    let features = processor.forward(input)?;
                    low_features.push(features);
                }
            } else {
                // Handle missing modality with zeros
                let shape = match inputs.values().next() {
                    Some(t) => match &t.data {
                        crate::SpikeRepresentation::Dense(arr) => {
                            (arr.dim().0, arr.dim().1, self.config.hidden_size / 2)
                        }
                        _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
                    },
                    None => return Err(SNNError::InvalidConfig("No inputs".to_string())),
                };
                low_features.push(SpikeTensor::from_dense(
                    Array3::<f32>::zeros(shape),
                    false,
                ));
            }
        }

        // Mid-level: Concatenate and fuse
        let low_refs: Vec<&SpikeTensor> = low_features.iter().collect();
        let mid_input = concatenate_spikes(&low_refs)?;
        let mut x = self.mid_level.forward(&mid_input)?;

        // High-level: Final decision layers
        for layer in &mut self.high_level {
            x = layer.forward(&x)?;
        }

        Ok(x)
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Low-level parameters
        for processor in self.low_level.values() {
            total += processor.parameters().iter()
                .map(|p| p.len())
                .sum::<usize>();
        }

        // Mid-level parameters
        total += self.mid_level.parameters().iter()
            .map(|p| p.len())
            .sum::<usize>();

        // High-level parameters
        for layer in &self.high_level {
            total += layer.parameters().iter()
                .map(|p| p.len())
                .sum::<usize>();
        }

        total
    }

    fn reset_state(&mut self) {
        for processor in self.low_level.values_mut() {
            processor.reset_state();
        }

        self.mid_level.reset_state();

        for layer in &mut self.high_level {
            layer.reset_state();
        }
    }

    fn handles_missing_modalities(&self) -> bool {
        true  // Hierarchical fusion handles missing modalities with zeros
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_hierarchical_fusion_creation() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 128,
            num_layers: 3,
            output_size: 10,
            ..Default::default()
        };

        let network = HierarchicalFusionSNN::new(config).unwrap();
        assert_eq!(network.name(), "HierarchicalFusionSNN");
        assert_eq!(network.low_level().len(), 2);
        assert!(!network.high_level().is_empty());
    }

    #[test]
    fn test_hierarchical_fusion_forward() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 3,
            output_size: 5,
            ..Default::default()
        };

        let mut network = HierarchicalFusionSNN::new(config).unwrap();

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
    fn test_hierarchical_fusion_missing_modality() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 3,
            output_size: 5,
            ..Default::default()
        };

        let mut network = HierarchicalFusionSNN::new(config).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );
        // Missing Hand modality - should fill with zeros

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hierarchical_fusion_parameters() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact],
            hidden_size: 32,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let network = HierarchicalFusionSNN::new(config).unwrap();
        let num_params = network.num_parameters();
        assert!(num_params > 0);
    }

    #[test]
    fn test_hierarchical_fusion_reset() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 64,
            num_layers: 3,
            output_size: 10,
            ..Default::default()
        };

        let mut network = HierarchicalFusionSNN::new(config).unwrap();
        network.reset_state();
        // Should not panic
    }

    #[test]
    fn test_hierarchical_fusion_all_modalities() {
        let config = FusionConfig {
            modalities: Modality::all(),
            hidden_size: 128,
            num_layers: 4,
            output_size: 20,
            ..Default::default()
        };

        let mut network = HierarchicalFusionSNN::new(config).unwrap();

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
