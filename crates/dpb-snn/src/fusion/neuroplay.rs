//! NeuroPlay: Full multi-modal integration for Parkinson's assessment
//!
//! Combines all 5 modalities (Contact, Pose, Hand, Eye, Voice) using
//! hierarchical processing to predict clinical scores (UPDRS total).

use super::{FusionConfig, FusionNetwork, Modality, concatenate_spikes};
use crate::layers::SpikingLayer;
use crate::{
    NeuronParams, SNNError, SNNResult, SpikeTensor, SpikingAttention, SpikingLinear, SpikingRNN,
};
use ndarray::Array3;
use std::collections::HashMap;

/// Modality group for hierarchical processing
#[derive(Debug, Clone, Copy)]
pub enum ModalityGroup {
    /// Motor symptoms: Contact + Pose + Hand
    Motor,
    /// Cognitive/behavioral: Eye + Voice
    Cognitive,
}

/// NeuroPlay full multi-modal integration network
// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
pub struct NeuroPlaySNN {
    config: FusionConfig,
    /// Low-level: Per-modality feature extraction
    modality_encoders: HashMap<Modality, Vec<SpikingLinear>>,
    /// Mid-level: Temporal dynamics per modality
    temporal_processors: HashMap<Modality, SpikingRNN>,
    /// High-level: Cross-modal attention
    motor_attention: SpikingAttention,
    cognitive_attention: SpikingAttention,
    /// Group fusion layers
    motor_fusion: SpikingLinear,
    cognitive_fusion: SpikingLinear,
    /// Final integration
    final_fusion: Vec<SpikingLinear>,
    /// UPDRS score regression head
    updrs_head: SpikingLinear,
    neuron_params: NeuronParams,
}

impl NeuroPlaySNN {
    /// Create a new NeuroPlay network
    pub fn new(config: FusionConfig) -> SNNResult<Self> {
        let neuron_params = NeuronParams::default();
        let dt = 1.0;
        let adaptive = false;

        // Ensure all 5 modalities are configured
        let all_modalities = Modality::all();
        let config = FusionConfig {
            modalities: all_modalities.clone(),
            ..config
        };

        // Low-level encoders: 2 layers per modality
        let mut modality_encoders = HashMap::new();
        for modality in &all_modalities {
            let input_size = modality.default_feature_dim();
            let layers = vec![
                SpikingLinear::new(
                    input_size,
                    config.hidden_size / 2,
                    true,
                    neuron_params.clone(),
                    dt,
                    adaptive,
                ),
                SpikingLinear::new(
                    config.hidden_size / 2,
                    config.hidden_size,
                    true,
                    neuron_params.clone(),
                    dt,
                    adaptive,
                ),
            ];

            modality_encoders.insert(*modality, layers);
        }

        // Temporal processors
        let mut temporal_processors = HashMap::new();
        for modality in &all_modalities {
            temporal_processors.insert(
                *modality,
                SpikingRNN::new(
                    config.hidden_size,
                    config.hidden_size,
                    true,
                    neuron_params.clone(),
                    dt,
                    adaptive,
                ),
            );
        }

        // Cross-modal attention for each group
        let motor_attention = SpikingAttention::new(
            config.hidden_size,
            config.attention_heads,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        let cognitive_attention = SpikingAttention::new(
            config.hidden_size,
            config.attention_heads,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        // Group fusion layers
        // Motor: Contact + Pose + Hand = 3 modalities
        let motor_fusion = SpikingLinear::new(
            3 * config.hidden_size,
            config.hidden_size,
            true,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        // Cognitive: Eye + Voice = 2 modalities
        let cognitive_fusion = SpikingLinear::new(
            2 * config.hidden_size,
            config.hidden_size,
            true,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        // Final integration layers
        let final_fusion = vec![
            // Motor + Cognitive
            SpikingLinear::new(
                2 * config.hidden_size,
                config.hidden_size,
                true,
                neuron_params.clone(),
                dt,
                adaptive,
            ),
            SpikingLinear::new(
                config.hidden_size,
                config.hidden_size / 2,
                true,
                neuron_params.clone(),
                dt,
                adaptive,
            ),
        ];

        // UPDRS score regression (typically 0-132 for total UPDRS)
        let updrs_head = SpikingLinear::new(
            config.hidden_size / 2,
            config.output_size, // Default: 1 for total score
            true,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        Ok(Self {
            config,
            modality_encoders,
            temporal_processors,
            motor_attention,
            cognitive_attention,
            motor_fusion,
            cognitive_fusion,
            final_fusion,
            updrs_head,
            neuron_params,
        })
    }

    /// Process motor modality group
    fn process_motor_group(
        &mut self,
        contact: Option<&SpikeTensor>,
        pose: Option<&SpikeTensor>,
        hand: Option<&SpikeTensor>,
    ) -> SNNResult<SpikeTensor> {
        let mut motor_features = Vec::new();

        // Get reference shape for filling missing modalities
        let reference = contact
            .or(pose)
            .or(hand)
            .ok_or_else(|| SNNError::InvalidConfig("No motor modalities available".to_string()))?;

        let ref_shape = match &reference.data {
            crate::SpikeRepresentation::Dense(arr) => {
                (arr.dim().0, arr.dim().1, self.config.hidden_size)
            }
            _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
        };

        for (modality, input) in [
            (Modality::Contact, contact),
            (Modality::Pose, pose),
            (Modality::Hand, hand),
        ] {
            let feature = if let Some(inp) = input {
                // Encode and process
                let mut x = inp.clone();
                if let Some(layers) = self.modality_encoders.get_mut(&modality) {
                    for layer in layers {
                        x = layer.forward(&x)?;
                    }
                }
                if let Some(rnn) = self.temporal_processors.get_mut(&modality) {
                    x = rnn.forward(&x)?;
                }
                x
            } else {
                // Missing modality: use zeros
                SpikeTensor::from_dense(Array3::<f32>::zeros(ref_shape), false)
            };

            motor_features.push(feature);
        }

        // Concatenate and fuse
        let motor_refs: Vec<&SpikeTensor> = motor_features.iter().collect();
        let mut motor_concat = concatenate_spikes(&motor_refs)?;

        // Fuse first (reduce dimensionality)
        motor_concat = self.motor_fusion.forward(&motor_concat)?;

        // Apply attention (now on correct dimensions)
        motor_concat = self.motor_attention.forward(&motor_concat)?;

        Ok(motor_concat)
    }

    /// Process cognitive modality group
    fn process_cognitive_group(
        &mut self,
        eye: Option<&SpikeTensor>,
        voice: Option<&SpikeTensor>,
    ) -> SNNResult<SpikeTensor> {
        let mut cognitive_features = Vec::new();

        // Get reference shape
        let reference = eye.or(voice).ok_or_else(|| {
            SNNError::InvalidConfig("No cognitive modalities available".to_string())
        })?;

        let ref_shape = match &reference.data {
            crate::SpikeRepresentation::Dense(arr) => {
                (arr.dim().0, arr.dim().1, self.config.hidden_size)
            }
            _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
        };

        for (modality, input) in [(Modality::Eye, eye), (Modality::Voice, voice)] {
            let feature = if let Some(inp) = input {
                // Encode and process
                let mut x = inp.clone();
                if let Some(layers) = self.modality_encoders.get_mut(&modality) {
                    for layer in layers {
                        x = layer.forward(&x)?;
                    }
                }
                if let Some(rnn) = self.temporal_processors.get_mut(&modality) {
                    x = rnn.forward(&x)?;
                }
                x
            } else {
                // Missing modality: use zeros
                SpikeTensor::from_dense(Array3::<f32>::zeros(ref_shape), false)
            };

            cognitive_features.push(feature);
        }

        // Concatenate and fuse
        let cognitive_refs: Vec<&SpikeTensor> = cognitive_features.iter().collect();
        let mut cognitive_concat = concatenate_spikes(&cognitive_refs)?;

        // Fuse first (reduce dimensionality)
        cognitive_concat = self.cognitive_fusion.forward(&cognitive_concat)?;

        // Apply attention (now on correct dimensions)
        cognitive_concat = self.cognitive_attention.forward(&cognitive_concat)?;

        Ok(cognitive_concat)
    }

    /// Get modality encoders for testing
    pub fn modality_encoders(&self) -> &HashMap<Modality, Vec<SpikingLinear>> {
        &self.modality_encoders
    }
}

impl FusionNetwork for NeuroPlaySNN {
    fn name(&self) -> &str {
        "NeuroPlaySNN"
    }

    fn modalities(&self) -> Vec<Modality> {
        Modality::all()
    }

    fn forward(&mut self, inputs: &HashMap<Modality, SpikeTensor>) -> SNNResult<SpikeTensor> {
        if inputs.is_empty() {
            return Err(SNNError::InvalidConfig("No inputs provided".to_string()));
        }

        // Extract modalities
        let contact = inputs.get(&Modality::Contact);
        let pose = inputs.get(&Modality::Pose);
        let hand = inputs.get(&Modality::Hand);
        let eye = inputs.get(&Modality::Eye);
        let voice = inputs.get(&Modality::Voice);

        // Process motor group (Contact, Pose, Hand)
        let motor_output = if contact.is_some() || pose.is_some() || hand.is_some() {
            Some(self.process_motor_group(contact, pose, hand)?)
        } else {
            None
        };

        // Process cognitive group (Eye, Voice)
        let cognitive_output = if eye.is_some() || voice.is_some() {
            Some(self.process_cognitive_group(eye, voice)?)
        } else {
            None
        };

        // Combine groups
        let combined = match (motor_output, cognitive_output) {
            (Some(motor), Some(cognitive)) => {
                let refs = vec![&motor, &cognitive];
                concatenate_spikes(&refs)?
            }
            (Some(motor), None) => {
                // Only motor available - pad with zeros for cognitive
                let shape = match &motor.data {
                    crate::SpikeRepresentation::Dense(arr) => arr.dim(),
                    _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
                };
                let zeros = SpikeTensor::from_dense(Array3::<f32>::zeros(shape), false);
                let refs = vec![&motor, &zeros];
                concatenate_spikes(&refs)?
            }
            (None, Some(cognitive)) => {
                // Only cognitive available - pad with zeros for motor
                let shape = match &cognitive.data {
                    crate::SpikeRepresentation::Dense(arr) => arr.dim(),
                    _ => return Err(SNNError::InvalidConfig("Sparse not supported".to_string())),
                };
                let zeros = SpikeTensor::from_dense(Array3::<f32>::zeros(shape), false);
                let refs = vec![&zeros, &cognitive];
                concatenate_spikes(&refs)?
            }
            (None, None) => {
                return Err(SNNError::InvalidConfig(
                    "No modality groups available".to_string(),
                ));
            }
        };

        // Final integration
        let mut x = combined;
        for layer in &mut self.final_fusion {
            x = layer.forward(&x)?;
        }

        // UPDRS score prediction
        x = self.updrs_head.forward(&x)?;

        Ok(x)
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Modality encoder parameters
        for layers in self.modality_encoders.values() {
            for layer in layers {
                total += layer.parameters().iter().map(|p| p.len()).sum::<usize>();
            }
        }

        // Temporal processor parameters
        for rnn in self.temporal_processors.values() {
            total += rnn.parameters().iter().map(|p| p.len()).sum::<usize>();
        }

        // Attention parameters
        total += self
            .motor_attention
            .parameters()
            .iter()
            .map(|p| p.len())
            .sum::<usize>();
        total += self
            .cognitive_attention
            .parameters()
            .iter()
            .map(|p| p.len())
            .sum::<usize>();

        // Fusion parameters
        total += self
            .motor_fusion
            .parameters()
            .iter()
            .map(|p| p.len())
            .sum::<usize>();
        total += self
            .cognitive_fusion
            .parameters()
            .iter()
            .map(|p| p.len())
            .sum::<usize>();

        // Final fusion parameters
        for layer in &self.final_fusion {
            total += layer.parameters().iter().map(|p| p.len()).sum::<usize>();
        }

        // UPDRS head parameters
        total += self
            .updrs_head
            .parameters()
            .iter()
            .map(|p| p.len())
            .sum::<usize>();

        total
    }

    fn reset_state(&mut self) {
        for layers in self.modality_encoders.values_mut() {
            for layer in layers {
                layer.reset_state();
            }
        }

        for rnn in self.temporal_processors.values_mut() {
            rnn.reset_state();
        }

        self.motor_attention.reset_state();
        self.cognitive_attention.reset_state();
        self.motor_fusion.reset_state();
        self.cognitive_fusion.reset_state();

        for layer in &mut self.final_fusion {
            layer.reset_state();
        }

        self.updrs_head.reset_state();
    }

    fn handles_missing_modalities(&self) -> bool {
        true // NeuroPlay is designed to handle missing modalities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_neuroplay_creation() {
        let config = FusionConfig::default();
        let network = NeuroPlaySNN::new(config).unwrap();
        assert_eq!(network.name(), "NeuroPlaySNN");
        assert_eq!(network.modalities().len(), 5);
        assert_eq!(network.modality_encoders().len(), 5);
    }

    #[test]
    fn test_neuroplay_all_modalities() {
        let config = FusionConfig {
            hidden_size: 64,
            num_layers: 3,
            attention_heads: 4,
            output_size: 1,
            ..Default::default()
        };

        let mut network = NeuroPlaySNN::new(config).unwrap();

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

        let output = network.forward(&inputs).unwrap();

        match output.data {
            crate::SpikeRepresentation::Dense(arr) => {
                assert_eq!(arr.dim().2, 1); // UPDRS score
            }
            _ => panic!("Expected dense output"),
        }
    }

    #[test]
    fn test_neuroplay_motor_only() {
        let config = FusionConfig::default();
        let mut network = NeuroPlaySNN::new(config).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );
        inputs.insert(
            Modality::Pose,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 256)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_neuroplay_cognitive_only() {
        let config = FusionConfig::default();
        let mut network = NeuroPlaySNN::new(config).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Eye,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 64)), false),
        );
        inputs.insert(
            Modality::Voice,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_neuroplay_single_modality() {
        let config = FusionConfig::default();
        let mut network = NeuroPlaySNN::new(config).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_neuroplay_parameters() {
        let config = FusionConfig::default();
        let network = NeuroPlaySNN::new(config).unwrap();
        let num_params = network.num_parameters();
        assert!(num_params > 0);
        // Should be quite large due to all the layers
        assert!(num_params > 100000);
    }

    #[test]
    fn test_neuroplay_reset() {
        let config = FusionConfig::default();
        let mut network = NeuroPlaySNN::new(config).unwrap();
        network.reset_state();
        // Should not panic
    }

    #[test]
    fn test_neuroplay_mixed_modalities() {
        let config = FusionConfig::default();
        let mut network = NeuroPlaySNN::new(config).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );
        inputs.insert(
            Modality::Eye,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 64)), false),
        );
        inputs.insert(
            Modality::Hand,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 128)), false),
        );

        let result = network.forward(&inputs);
        assert!(result.is_ok());
    }
}
