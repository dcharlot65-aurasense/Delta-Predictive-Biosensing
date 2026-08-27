//! Temporal alignment fusion: Synchronize spike trains across modalities
//!
//! Handles different sampling rates and aligns events across time
//! before fusion.

use super::{FusionConfig, FusionNetwork, Modality, concatenate_spikes};
use crate::layers::SpikingLayer;
use crate::{NeuronParams, SNNError, SNNResult, SpikeTensor, SpikingLinear, SpikingRNN};
use ndarray::{Array3, s};
use std::collections::HashMap;

/// Temporal alignment strategy
#[derive(Debug, Clone, Copy)]
pub enum AlignmentStrategy {
    /// Interpolate to common timesteps
    Interpolation,
    /// Use recurrent layers to capture temporal dynamics
    RecurrentAlignment,
    /// Window-based aggregation
    WindowAggregation,
}

/// Temporal alignment fusion network
// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
pub struct TemporalAlignmentSNN {
    config: FusionConfig,
    /// Temporal processors per modality
    temporal_processors: HashMap<Modality, SpikingRNN>,
    /// Alignment layers
    alignment_layers: Vec<SpikingLinear>,
    /// Final fusion layer
    fusion_layer: SpikingLinear,
    alignment_strategy: AlignmentStrategy,
    neuron_params: NeuronParams,
}

impl TemporalAlignmentSNN {
    /// Create a new temporal alignment fusion network
    pub fn new(config: FusionConfig, strategy: AlignmentStrategy) -> SNNResult<Self> {
        let neuron_params = NeuronParams::default();
        let dt = 1.0;
        let adaptive = false;

        // Create temporal processor for each modality
        let mut temporal_processors = HashMap::new();
        for modality in &config.modalities {
            let input_size = modality.default_feature_dim();
            temporal_processors.insert(
                *modality,
                SpikingRNN::new(
                    input_size,
                    config.hidden_size,
                    true,
                    neuron_params.clone(),
                    dt,
                    adaptive,
                ),
            );
        }

        // Alignment layers
        let mut alignment_layers = Vec::new();
        let num_modalities = config.modalities.len();
        let aligned_size = num_modalities * config.hidden_size;

        alignment_layers.push(SpikingLinear::new(
            aligned_size,
            config.hidden_size,
            true,
            neuron_params.clone(),
            dt,
            adaptive,
        ));

        // Fusion layer
        let fusion_layer = SpikingLinear::new(
            config.hidden_size,
            config.output_size,
            true,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        Ok(Self {
            config,
            temporal_processors,
            alignment_layers,
            fusion_layer,
            alignment_strategy: strategy,
            neuron_params,
        })
    }

    /// Align spike tensors to common temporal resolution
    fn align_temporal(
        &self,
        tensors: &HashMap<Modality, SpikeTensor>,
    ) -> SNNResult<HashMap<Modality, SpikeTensor>> {
        // For simplicity, we'll ensure all tensors have the same time dimension
        // In a full implementation, this would do proper interpolation/resampling

        let mut aligned = HashMap::new();

        // Find maximum time steps
        let max_time = tensors
            .values()
            .filter_map(|t| match &t.data {
                crate::SpikeRepresentation::Dense(arr) => Some(arr.dim().1),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        for (modality, tensor) in tensors {
            match &tensor.data {
                crate::SpikeRepresentation::Dense(arr) => {
                    let (batch, time, features) = arr.dim();

                    if time == max_time {
                        aligned.insert(*modality, tensor.clone());
                    } else if time < max_time {
                        // Pad with zeros
                        let mut padded = Array3::<f32>::zeros((batch, max_time, features));
                        padded.slice_mut(s![.., 0..time, ..]).assign(arr);
                        aligned.insert(
                            *modality,
                            SpikeTensor::from_dense(padded, tensor.requires_grad),
                        );
                    } else {
                        // Truncate
                        let truncated = arr.slice(s![.., 0..max_time, ..]).to_owned();
                        aligned.insert(
                            *modality,
                            SpikeTensor::from_dense(truncated, tensor.requires_grad),
                        );
                    }
                }
                _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
            }
        }

        Ok(aligned)
    }

    /// Get temporal processors for testing
    pub fn temporal_processors(&self) -> &HashMap<Modality, SpikingRNN> {
        &self.temporal_processors
    }
}

impl FusionNetwork for TemporalAlignmentSNN {
    fn name(&self) -> &str {
        "TemporalAlignmentSNN"
    }

    fn modalities(&self) -> Vec<Modality> {
        self.config.modalities.clone()
    }

    fn forward(&mut self, inputs: &HashMap<Modality, SpikeTensor>) -> SNNResult<SpikeTensor> {
        if inputs.is_empty() {
            return Err(SNNError::InvalidConfig("No inputs provided".to_string()));
        }

        // Step 1: Align temporal dimensions
        let aligned = self.align_temporal(inputs)?;

        // Step 2: Process each modality with temporal processor
        let mut processed = Vec::new();
        for modality in &self.config.modalities {
            if let Some(input) = aligned.get(modality) {
                if let Some(processor) = self.temporal_processors.get_mut(modality) {
                    let output = processor.forward(input)?;
                    processed.push(output);
                }
            } else {
                // Handle missing modality
                let shape = match aligned.values().next() {
                    Some(t) => match &t.data {
                        crate::SpikeRepresentation::Dense(arr) => {
                            (arr.dim().0, arr.dim().1, self.config.hidden_size)
                        }
                        _ => {
                            return Err(SNNError::InvalidConfig(
                                "Sparse not supported".to_string(),
                            ));
                        }
                    },
                    None => return Err(SNNError::InvalidConfig("No aligned inputs".to_string())),
                };
                processed.push(SpikeTensor::from_dense(Array3::<f32>::zeros(shape), false));
            }
        }

        // Step 3: Concatenate processed features
        let processed_refs: Vec<&SpikeTensor> = processed.iter().collect();
        let mut x = concatenate_spikes(&processed_refs)?;

        // Step 4: Apply alignment layers
        for layer in &mut self.alignment_layers {
            x = layer.forward(&x)?;
        }

        // Step 5: Final fusion
        x = self.fusion_layer.forward(&x)?;

        Ok(x)
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Temporal processor parameters
        for processor in self.temporal_processors.values() {
            total += processor
                .parameters()
                .iter()
                .map(|p| p.len())
                .sum::<usize>();
        }

        // Alignment layer parameters
        for layer in &self.alignment_layers {
            total += layer.parameters().iter().map(|p| p.len()).sum::<usize>();
        }

        // Fusion layer parameters
        total += self
            .fusion_layer
            .parameters()
            .iter()
            .map(|p| p.len())
            .sum::<usize>();

        total
    }

    fn reset_state(&mut self) {
        for processor in self.temporal_processors.values_mut() {
            processor.reset_state();
        }

        for layer in &mut self.alignment_layers {
            layer.reset_state();
        }

        self.fusion_layer.reset_state();
    }

    fn handles_missing_modalities(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_temporal_alignment_creation() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 128,
            num_layers: 3,
            output_size: 10,
            ..Default::default()
        };

        let network =
            TemporalAlignmentSNN::new(config, AlignmentStrategy::RecurrentAlignment).unwrap();
        assert_eq!(network.name(), "TemporalAlignmentSNN");
        assert_eq!(network.temporal_processors().len(), 2);
    }

    #[test]
    fn test_temporal_alignment_forward() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network =
            TemporalAlignmentSNN::new(config, AlignmentStrategy::RecurrentAlignment).unwrap();

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
    fn test_temporal_alignment_different_lengths() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network =
            TemporalAlignmentSNN::new(config, AlignmentStrategy::RecurrentAlignment).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );
        inputs.insert(
            Modality::Hand,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 15, 128)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_temporal_alignment_missing_modality() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network =
            TemporalAlignmentSNN::new(config, AlignmentStrategy::RecurrentAlignment).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_temporal_alignment_parameters() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact],
            hidden_size: 32,
            num_layers: 2,
            output_size: 5,
            ..Default::default()
        };

        let network =
            TemporalAlignmentSNN::new(config, AlignmentStrategy::RecurrentAlignment).unwrap();
        let num_params = network.num_parameters();
        assert!(num_params > 0);
    }
}
