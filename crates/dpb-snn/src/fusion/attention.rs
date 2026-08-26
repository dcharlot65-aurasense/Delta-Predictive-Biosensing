//! Cross-modal attention fusion: Each modality attends to others
//!
//! Uses spiking attention mechanisms to learn which modalities are
//! relevant for the current input context.

use super::{FusionNetwork, FusionConfig, Modality, concatenate_spikes};
use crate::{SpikeTensor, SpikingLinear, SpikingAttention, SNNResult, SNNError, NeuronParams};
use crate::layers::SpikingLayer;
use ndarray::{Array2, Array3, Array4};
use std::collections::HashMap;

/// Cross-modal attention fusion network
pub struct CrossModalAttentionSNN {
    config: FusionConfig,
    /// Encode each modality to common representation
    encoders: HashMap<Modality, SpikingLinear>,
    /// Cross-attention between modalities
    attention_layers: Vec<SpikingAttention>,
    /// Final projection to output
    output_projection: SpikingLinear,
    neuron_params: NeuronParams,
}

impl CrossModalAttentionSNN {
    /// Create a new cross-modal attention fusion network
    pub fn new(config: FusionConfig) -> SNNResult<Self> {
        let neuron_params = NeuronParams::default();
        let dt = 1.0;
        let adaptive = false;

        // Create encoder for each modality
        let mut encoders = HashMap::new();
        for modality in &config.modalities {
            let input_size = modality.default_feature_dim();
            encoders.insert(
                *modality,
                SpikingLinear::new(input_size, config.hidden_size, true, neuron_params.clone(), dt, adaptive),
            );
        }

        // Create attention layers
        let mut attention_layers = Vec::new();
        for _ in 0..config.num_layers {
            attention_layers.push(SpikingAttention::new(
                config.hidden_size,
                config.attention_heads,
                neuron_params.clone(),
                dt,
                adaptive,
            ));
        }

        // Output projection
        let output_projection = SpikingLinear::new(
            config.hidden_size,
            config.output_size,
            true,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        Ok(Self {
            config,
            encoders,
            attention_layers,
            output_projection,
            neuron_params,
        })
    }

    /// Get encoders for testing
    pub fn encoders(&self) -> &HashMap<Modality, SpikingLinear> {
        &self.encoders
    }

    /// Get attention layers for testing
    pub fn attention_layers(&self) -> &[SpikingAttention] {
        &self.attention_layers
    }
}

impl FusionNetwork for CrossModalAttentionSNN {
    fn name(&self) -> &str {
        "CrossModalAttentionSNN"
    }

    fn modalities(&self) -> Vec<Modality> {
        self.config.modalities.clone()
    }

    fn forward(&mut self, inputs: &HashMap<Modality, SpikeTensor>) -> SNNResult<SpikeTensor> {
        if inputs.is_empty() {
            return Err(SNNError::InvalidConfig("No inputs provided".to_string()));
        }

        // Encode each modality to common hidden space
        let mut encoded = Vec::new();
        for modality in &self.config.modalities {
            if let Some(input) = inputs.get(modality)
                && let Some(encoder) = self.encoders.get_mut(modality) {
                    let enc = encoder.forward(input)?;
                    encoded.push(enc);
                }
        }

        if encoded.is_empty() {
            return Err(SNNError::InvalidConfig("No valid modality encodings".to_string()));
        }

        // Stack encodings: average pool for now (simplification)
        // In a full implementation, we'd use proper cross-attention
        let mut x = encoded[0].clone();
        for enc in &encoded[1..] {
            match (&x.data, &enc.data) {
                (crate::SpikeRepresentation::Dense(arr1),
                 crate::SpikeRepresentation::Dense(arr2)) => {
                    x = SpikeTensor::from_dense(arr1 + arr2, x.requires_grad);
                }
                _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
            }
        }

        // Normalize by number of modalities
        match &x.data {
            crate::SpikeRepresentation::Dense(arr) => {
                x = SpikeTensor::from_dense(
                    arr / (encoded.len() as f32),
                    x.requires_grad,
                );
            }
            _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
        }

        // Apply attention layers
        for attention in &mut self.attention_layers {
            x = attention.forward(&x)?;
        }

        // Project to output
        x = self.output_projection.forward(&x)?;

        Ok(x)
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Encoder parameters
        for encoder in self.encoders.values() {
            total += encoder.parameters().iter()
                .map(|p| p.len())
                .sum::<usize>();
        }

        // Attention parameters
        for attention in &self.attention_layers {
            total += attention.parameters().iter()
                .map(|p| p.len())
                .sum::<usize>();
        }

        // Output projection parameters
        total += self.output_projection.parameters().iter()
            .map(|p| p.len())
            .sum::<usize>();

        total
    }

    fn reset_state(&mut self) {
        for encoder in self.encoders.values_mut() {
            encoder.reset_state();
        }

        for attention in &mut self.attention_layers {
            attention.reset_state();
        }

        self.output_projection.reset_state();
    }

    fn handles_missing_modalities(&self) -> bool {
        true  // Attention can handle variable number of modalities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_attention_fusion_creation() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 128,
            num_layers: 2,
            attention_heads: 4,
            output_size: 10,
            ..Default::default()
        };

        let network = CrossModalAttentionSNN::new(config).unwrap();
        assert_eq!(network.name(), "CrossModalAttentionSNN");
        assert_eq!(network.encoders().len(), 2);
        assert_eq!(network.attention_layers().len(), 2);
    }

    #[test]
    fn test_attention_fusion_forward() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 1,
            attention_heads: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = CrossModalAttentionSNN::new(config).unwrap();

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
    fn test_attention_fusion_single_modality() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Hand],
            hidden_size: 64,
            num_layers: 1,
            attention_heads: 2,
            output_size: 5,
            ..Default::default()
        };

        let mut network = CrossModalAttentionSNN::new(config).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_attention_fusion_parameters() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact],
            hidden_size: 32,
            num_layers: 1,
            attention_heads: 2,
            output_size: 5,
            ..Default::default()
        };

        let network = CrossModalAttentionSNN::new(config).unwrap();
        let num_params = network.num_parameters();
        assert!(num_params > 0);
    }

    #[test]
    fn test_attention_fusion_reset() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 64,
            num_layers: 2,
            attention_heads: 4,
            output_size: 10,
            ..Default::default()
        };

        let mut network = CrossModalAttentionSNN::new(config).unwrap();
        network.reset_state();
        // Should not panic
    }
}
