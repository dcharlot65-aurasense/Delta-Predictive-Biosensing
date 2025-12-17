//! Multi-modal fusion architectures for combining spike trains from different modalities
//!
//! This module provides various fusion strategies for integrating spike trains
//! from multiple biosensing modalities (contact sensors, pose, hand, eye, voice).

pub mod early;
pub mod late;
pub mod attention;
pub mod hierarchical;
pub mod temporal;
pub mod gated;
pub mod neuroplay;

pub use early::EarlyFusionSNN;
pub use late::LateFusionSNN;
pub use attention::CrossModalAttentionSNN;
pub use hierarchical::HierarchicalFusionSNN;
pub use temporal::TemporalAlignmentSNN;
pub use gated::GatedFusionSNN;
pub use neuroplay::NeuroPlaySNN;

use crate::{SpikeTensor, SNNResult, SNNError};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Modality types for multi-modal fusion
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Contact sensors: ECG, EDA, tremor sensors
    Contact,
    /// Full body pose and gait tracking
    Pose,
    /// Hand landmarks and fine motor control
    Hand,
    /// Eye tracking: gaze, pupil, saccades
    Eye,
    /// Voice and speech features
    Voice,
}

impl Modality {
    /// Get typical feature dimension for each modality
    pub fn default_feature_dim(&self) -> usize {
        match self {
            Modality::Contact => 128,  // ECG, EDA, tremor combined
            Modality::Pose => 256,     // Full body keypoints
            Modality::Hand => 128,     // Hand landmarks
            Modality::Eye => 64,       // Gaze, pupil metrics
            Modality::Voice => 128,    // Speech features
        }
    }

    /// Get all modalities
    pub fn all() -> Vec<Modality> {
        vec![
            Modality::Contact,
            Modality::Pose,
            Modality::Hand,
            Modality::Eye,
            Modality::Voice,
        ]
    }
}

/// Configuration for fusion networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Modalities to fuse
    pub modalities: Vec<Modality>,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Number of layers in each modality-specific network
    pub num_layers: usize,
    /// Number of attention heads (for attention-based fusion)
    pub attention_heads: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Temporal window size in milliseconds
    pub temporal_window_ms: f32,
    /// Output dimension
    pub output_size: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            modalities: Modality::all(),
            hidden_size: 256,
            num_layers: 3,
            attention_heads: 8,
            dropout: 0.1,
            temporal_window_ms: 100.0,
            output_size: 1,  // Default: single clinical score
        }
    }
}

/// Core trait for all fusion networks
pub trait FusionNetwork: Send + Sync {
    /// Get the network name
    fn name(&self) -> &str;

    /// Get the modalities this network handles
    fn modalities(&self) -> Vec<Modality>;

    /// Forward pass through the fusion network
    ///
    /// # Arguments
    /// * `inputs` - Map from modality to spike tensor
    ///
    /// # Returns
    /// Fused spike tensor output
    fn forward(&mut self, inputs: &HashMap<Modality, SpikeTensor>) -> SNNResult<SpikeTensor>;

    /// Get total number of trainable parameters
    fn num_parameters(&self) -> usize;

    /// Reset internal state
    fn reset_state(&mut self);

    /// Check if network can handle missing modalities
    fn handles_missing_modalities(&self) -> bool {
        true
    }

    /// Get required modalities (if any)
    fn required_modalities(&self) -> Vec<Modality> {
        Vec::new()
    }
}

/// Spike train type alias for cleaner code
pub type SpikeTrain = SpikeTensor;

/// Utility: Concatenate spike tensors along feature dimension
pub fn concatenate_spikes(tensors: &[&SpikeTensor]) -> SNNResult<SpikeTensor> {
    if tensors.is_empty() {
        return Err(SNNError::InvalidConfig("Cannot concatenate empty tensor list".to_string()));
    }

    let arrays: Vec<&Array3<f32>> = tensors
        .iter()
        .map(|t| match &t.data {
            crate::SpikeRepresentation::Dense(arr) => arr,
            crate::SpikeRepresentation::Sparse(_) => {
                panic!("Sparse spike tensors not yet supported in fusion")
            }
        })
        .collect();

    // Get dimensions: (batch, time, features)
    let (batch, time, _) = arrays[0].dim();

    // Check all have same batch and time dims
    for arr in &arrays {
        let (b, t, _) = arr.dim();
        if b != batch || t != time {
            return Err(SNNError::DimensionMismatch {
                expected: format!("({}, {}, _)", batch, time),
                actual: format!("({}, {}, _)", b, t),
            });
        }
    }

    // Calculate total features
    let total_features: usize = arrays.iter().map(|a| a.dim().2).sum();

    // Allocate output
    let mut output = Array3::<f32>::zeros((batch, time, total_features));

    // Concatenate along feature dimension
    let mut offset = 0;
    for arr in arrays {
        let features = arr.dim().2;
        output.slice_mut(ndarray::s![.., .., offset..offset + features])
            .assign(arr);
        offset += features;
    }

    Ok(SpikeTensor::from_dense(output, tensors[0].requires_grad))
}

/// Utility: Average spike tensors
pub fn average_spikes(tensors: &[&SpikeTensor]) -> SNNResult<SpikeTensor> {
    if tensors.is_empty() {
        return Err(SNNError::InvalidConfig("Cannot average empty tensor list".to_string()));
    }

    let first = match &tensors[0].data {
        crate::SpikeRepresentation::Dense(arr) => arr,
        crate::SpikeRepresentation::Sparse(_) => {
            return Err(SNNError::InvalidConfig("Sparse tensors not supported".to_string()));
        }
    };

    let mut sum = first.clone();

    for tensor in &tensors[1..] {
        match &tensor.data {
            crate::SpikeRepresentation::Dense(arr) => {
                sum = sum + arr;
            }
            crate::SpikeRepresentation::Sparse(_) => {
                return Err(SNNError::InvalidConfig("Sparse tensors not supported".to_string()));
            }
        }
    }

    let avg = sum / (tensors.len() as f32);
    Ok(SpikeTensor::from_dense(avg, tensors[0].requires_grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_modality_default_dims() {
        assert_eq!(Modality::Contact.default_feature_dim(), 128);
        assert_eq!(Modality::Pose.default_feature_dim(), 256);
        assert_eq!(Modality::Hand.default_feature_dim(), 128);
        assert_eq!(Modality::Eye.default_feature_dim(), 64);
        assert_eq!(Modality::Voice.default_feature_dim(), 128);
    }

    #[test]
    fn test_modality_all() {
        let all = Modality::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&Modality::Contact));
        assert!(all.contains(&Modality::Voice));
    }

    #[test]
    fn test_concatenate_spikes() {
        let spike1 = SpikeTensor::from_dense(Array3::<f32>::zeros((2, 10, 32)), false);
        let spike2 = SpikeTensor::from_dense(Array3::<f32>::zeros((2, 10, 64)), false);

        let result = concatenate_spikes(&[&spike1, &spike2]).unwrap();

        match result.data {
            crate::SpikeRepresentation::Dense(arr) => {
                assert_eq!(arr.dim(), (2, 10, 96));
            }
            _ => panic!("Expected dense representation"),
        }
    }

    #[test]
    fn test_average_spikes() {
        let spike1 = SpikeTensor::from_dense(Array3::<f32>::ones((2, 10, 32)), false);
        let spike2 = SpikeTensor::from_dense(Array3::<f32>::ones((2, 10, 32)) * 3.0, false);

        let result = average_spikes(&[&spike1, &spike2]).unwrap();

        match result.data {
            crate::SpikeRepresentation::Dense(arr) => {
                assert_eq!(arr.dim(), (2, 10, 32));
                assert!((arr[[0, 0, 0]] - 2.0).abs() < 1e-5);
            }
            _ => panic!("Expected dense representation"),
        }
    }

    #[test]
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        assert_eq!(config.modalities.len(), 5);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 3);
    }
}
