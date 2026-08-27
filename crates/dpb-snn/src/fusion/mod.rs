//! Multi-modal fusion architectures for combining spike trains from different modalities
//!
//! This module provides various fusion strategies for integrating spike trains
//! from multiple biosensing modalities (contact sensors, pose, hand, eye, voice).

pub mod attention;
pub mod cognitive_motor;
pub mod early;
pub mod gated;
pub mod hierarchical;
pub mod late;
pub mod neuroplay;
pub mod temporal;

pub use attention::CrossModalAttentionSNN;
pub use cognitive_motor::{
    ChangeDirection, ChangeMetrics, CognitiveMotorFusion, CognitiveMotorFusionConfig,
    CognitiveMotorFusionSNN, CognitiveProfile, DissociationPattern, DissociationType,
    DomainZScores, IntegratedAssessment, MotorProfile, RiskCategory,
};
pub use early::EarlyFusionSNN;
pub use gated::GatedFusionSNN;
pub use hierarchical::HierarchicalFusionSNN;
pub use late::LateFusionSNN;
pub use neuroplay::NeuroPlaySNN;
pub use temporal::TemporalAlignmentSNN;

use crate::{SNNError, SNNResult, SpikeTensor};
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
            Modality::Contact => 128, // ECG, EDA, tremor combined
            Modality::Pose => 256,    // Full body keypoints
            Modality::Hand => 128,    // Hand landmarks
            Modality::Eye => 64,      // Gaze, pupil metrics
            Modality::Voice => 128,   // Speech features
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
            output_size: 1, // Default: single clinical score
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

/// Helper: Determine if a tensor should use sparse representation
///
/// Returns true if the tensor has < 10% sparsity (i.e., < 10% non-zero values)
/// and the values are binary (suitable for sparse spike representation)
fn should_use_sparse(tensor: &SpikeTensor) -> bool {
    match &tensor.data {
        crate::SpikeRepresentation::Dense(arr) => {
            let total_elements = arr.len();

            // Count non-zero values and check if they're binary
            let mut non_zero_count = 0;
            let mut is_binary = true;
            for &val in arr.iter() {
                if val.abs() > 1e-6 {
                    non_zero_count += 1;
                    // Check if value is close to 0 or 1 (binary spike)
                    if (val - 1.0).abs() > 1e-6 && val.abs() > 1e-6 {
                        is_binary = false;
                    }
                }
            }

            let sparsity = non_zero_count as f32 / total_elements as f32;
            sparsity < 0.1 && is_binary
        }
        crate::SpikeRepresentation::Sparse(sparse) => {
            let total_elements = sparse.batch_size * sparse.num_steps * sparse.num_neurons;
            let sparsity = sparse.num_spikes() as f32 / total_elements as f32;
            sparsity < 0.1
        }
    }
}

/// Utility: Concatenate spike tensors along feature dimension
///
/// Supports both dense and sparse tensors. If any input is sparse, it will be
/// converted to dense for concatenation. The output will be sparse if the
/// result is sufficiently sparse (< 10% sparsity).
pub fn concatenate_spikes(tensors: &[&SpikeTensor]) -> SNNResult<SpikeTensor> {
    if tensors.is_empty() {
        return Err(SNNError::InvalidConfig(
            "Cannot concatenate empty tensor list".to_string(),
        ));
    }

    // Track if any input was sparse
    let has_sparse = tensors
        .iter()
        .any(|t| matches!(&t.data, crate::SpikeRepresentation::Sparse(_)));

    // Convert all tensors to dense arrays for concatenation
    let arrays: Vec<Array3<f32>> = tensors.iter().map(|t| t.to_dense()).collect();

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
    for arr in &arrays {
        let features = arr.dim().2;
        output
            .slice_mut(ndarray::s![.., .., offset..offset + features])
            .assign(arr);
        offset += features;
    }

    // If input had sparse tensors, convert output back to sparse if beneficial
    let result_tensor = SpikeTensor::from_dense(output, tensors[0].requires_grad);

    if has_sparse && should_use_sparse(&result_tensor) {
        Ok(SpikeTensor::from_sparse(
            result_tensor.to_sparse(),
            tensors[0].requires_grad,
        ))
    } else {
        Ok(result_tensor)
    }
}

/// Utility: Average spike tensors
///
/// Supports both dense and sparse tensors. If any input is sparse, it will be
/// converted to dense for averaging. The output will be sparse if the result
/// is sufficiently sparse (< 10% sparsity) and any input was sparse.
pub fn average_spikes(tensors: &[&SpikeTensor]) -> SNNResult<SpikeTensor> {
    if tensors.is_empty() {
        return Err(SNNError::InvalidConfig(
            "Cannot average empty tensor list".to_string(),
        ));
    }

    // Track if any input was sparse
    let has_sparse = tensors
        .iter()
        .any(|t| matches!(&t.data, crate::SpikeRepresentation::Sparse(_)));

    // Convert first tensor to dense
    let mut sum = tensors[0].to_dense();

    // Add remaining tensors
    for tensor in &tensors[1..] {
        sum += &tensor.to_dense();
    }

    let avg = sum / (tensors.len() as f32);
    let result_tensor = SpikeTensor::from_dense(avg, tensors[0].requires_grad);

    // If input had sparse tensors, convert output back to sparse if beneficial
    if has_sparse && should_use_sparse(&result_tensor) {
        Ok(SpikeTensor::from_sparse(
            result_tensor.to_sparse(),
            tensors[0].requires_grad,
        ))
    } else {
        Ok(result_tensor)
    }
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

    #[test]
    fn test_concatenate_sparse_spikes() {
        // Create sparse tensors with low sparsity
        let mut sparse1 = crate::tensor::SparseSpikes::new(2, 10, 32);
        sparse1.add_spike(0, 1, 5).unwrap();
        sparse1.add_spike(0, 2, 10).unwrap();
        sparse1.add_spike(1, 3, 15).unwrap();

        let mut sparse2 = crate::tensor::SparseSpikes::new(2, 10, 64);
        sparse2.add_spike(0, 1, 20).unwrap();
        sparse2.add_spike(1, 5, 30).unwrap();

        let spike1 = SpikeTensor::from_sparse(sparse1, false);
        let spike2 = SpikeTensor::from_sparse(sparse2, false);

        let result = concatenate_spikes(&[&spike1, &spike2]).unwrap();

        // Result should be sparse (low sparsity maintained)
        match &result.data {
            crate::SpikeRepresentation::Sparse(sparse) => {
                assert_eq!(sparse.batch_size, 2);
                assert_eq!(sparse.num_steps, 10);
                assert_eq!(sparse.num_neurons, 96); // 32 + 64
                assert_eq!(sparse.num_spikes(), 5); // Total spikes preserved
            }
            _ => panic!("Expected sparse representation"),
        }
    }

    #[test]
    fn test_concatenate_mixed_spikes() {
        // Mix dense and sparse tensors
        let dense = SpikeTensor::from_dense(Array3::<f32>::zeros((2, 10, 32)), false);

        let mut sparse_data = crate::tensor::SparseSpikes::new(2, 10, 64);
        sparse_data.add_spike(0, 1, 20).unwrap();
        let sparse = SpikeTensor::from_sparse(sparse_data, false);

        let result = concatenate_spikes(&[&dense, &sparse]).unwrap();

        // Verify shape
        let (batch, time, features) = result.shape();
        assert_eq!((batch, time, features), (2, 10, 96));
    }

    #[test]
    fn test_average_sparse_spikes() {
        // Create two sparse tensors
        let mut sparse1 = crate::tensor::SparseSpikes::new(2, 10, 32);
        sparse1.add_spike(0, 1, 5).unwrap();
        sparse1.add_spike(0, 2, 10).unwrap();

        let mut sparse2 = crate::tensor::SparseSpikes::new(2, 10, 32);
        sparse2.add_spike(0, 1, 5).unwrap(); // Same spike
        sparse2.add_spike(1, 3, 15).unwrap(); // Different spike

        let spike1 = SpikeTensor::from_sparse(sparse1, false);
        let spike2 = SpikeTensor::from_sparse(sparse2, false);

        let result = average_spikes(&[&spike1, &spike2]).unwrap();

        // Result should maintain correct dimensions
        let (batch, time, features) = result.shape();
        assert_eq!((batch, time, features), (2, 10, 32));

        // Verify averaging works correctly
        let dense = result.to_dense();
        assert!((dense[[0, 1, 5]] - 1.0).abs() < 1e-5); // (1 + 1) / 2 = 1.0
        assert!((dense[[0, 2, 10]] - 0.5).abs() < 1e-5); // (1 + 0) / 2 = 0.5
        assert!((dense[[1, 3, 15]] - 0.5).abs() < 1e-5); // (0 + 1) / 2 = 0.5
    }

    #[test]
    fn test_sparse_dense_dimensions_match() {
        // Ensure sparse and dense with same dimensions can be concatenated
        let dense = SpikeTensor::from_dense(Array3::<f32>::ones((1, 5, 10)), false);

        let mut sparse_data = crate::tensor::SparseSpikes::new(1, 5, 20);
        sparse_data.add_spike(0, 2, 5).unwrap();
        let sparse = SpikeTensor::from_sparse(sparse_data, false);

        let result = concatenate_spikes(&[&dense, &sparse]).unwrap();
        assert_eq!(result.shape(), (1, 5, 30));
    }

    #[test]
    fn test_should_use_sparse_helper() {
        // Very sparse tensor (< 10% sparsity)
        let mut sparse_data = crate::tensor::SparseSpikes::new(1, 100, 100);
        for i in 0..50 {
            sparse_data.add_spike(0, i, i).unwrap(); // 50 spikes in 10000 elements = 0.5%
        }
        let sparse_tensor = SpikeTensor::from_sparse(sparse_data, false);
        assert!(should_use_sparse(&sparse_tensor));

        // Dense tensor (100% sparsity)
        let dense_tensor = SpikeTensor::from_dense(Array3::<f32>::ones((1, 10, 10)), false);
        assert!(!should_use_sparse(&dense_tensor));
    }
}
