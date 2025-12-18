//! Knowledge Distillation for SNN Compression
//!
//! This module implements knowledge distillation techniques for compressing
//! Spiking Neural Networks (SNNs) while preserving performance. Knowledge
//! distillation transfers knowledge from a large teacher network to a smaller
//! student network through various mechanisms.
//!
//! ## Features
//!
//! - **Teacher-Student Framework**: Coordinate training between teacher and student
//! - **Multiple Distillation Modes**: Response-based, feature-based, and relation-based
//! - **SNN-Specific Distillation**: Temporal spike patterns, firing rates, membrane potentials
//! - **Compression Utilities**: Architecture search, pruning, quantization
//! - **Self-Distillation**: Progressive model compression
//!
//! ## Example: Basic Knowledge Distillation
//!
//! ```rust
//! use dpb_snn::distillation::*;
//! use dpb_snn::*;
//!
//! # fn example() -> SNNResult<()> {
//! // Create teacher (large model)
//! let teacher = FeedforwardSNN::new(vec![128, 256, 128, 10], SNNConfig::default());
//!
//! // Create student (smaller model)
//! let student = FeedforwardSNN::new(vec![128, 64, 10], SNNConfig::default());
//!
//! // Configure distillation
//! let config = DistillationConfig {
//!     temperature: 4.0,
//!     alpha: 0.7,  // Weight for distillation loss
//!     beta: 0.3,   // Weight for student loss
//!     mode: DistillationMode::ResponseBased,
//! };
//!
//! // Create distillation framework
//! let mut framework = TeacherStudentFramework::new(teacher, student, config)?;
//!
//! // Train with distillation
//! // framework.train(train_data, epochs)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: SNN-Specific Distillation
//!
//! ```rust
//! use dpb_snn::distillation::*;
//!
//! # fn example() -> dpb_snn::SNNResult<()> {
//! // Configure spike-aware distillation
//! let config = SpikeDistillationConfig {
//!     match_spike_patterns: true,
//!     match_firing_rates: true,
//!     match_membrane_potentials: false,
//!     temporal_alignment: true,
//!     pattern_weight: 0.4,
//!     rate_weight: 0.3,
//!     membrane_weight: 0.0,
//! };
//!
//! let distiller = SpikePatternDistillation::new(config);
//! # Ok(())
//! # }
//! ```

pub mod teacher_student;
pub mod losses;
pub mod spike_distillation;
pub mod compression;
pub mod self_distillation;

// Re-export main types
pub use teacher_student::{
    TeacherStudentFramework, TeacherModel, StudentModel,
    DistillationConfig, DistillationMode, KnowledgeTransfer,
};
pub use losses::{
    DistillationLoss, KLDivergenceLoss, MSELoss, CosineSimLoss,
    HintLoss, AttentionTransferLoss, CombinedDistillationLoss,
    LossWeights,
};
pub use spike_distillation::{
    SpikePatternDistillation, SpikeRateDistillation,
    MembranePotentialDistillation, SynapticWeightTransfer,
    TemporalCreditAssignment, SpikeDistillationConfig,
};
pub use compression::{
    ArchitectureSearch, LayerMerging, ChannelPruningGuided,
    QuantizationAwareDistillation, CompressionMetrics,
    CompressionConfig, SearchStrategy,
};
pub use self_distillation::{
    SelfDistillation, BornAgainNetworks, ProgressiveDistillation,
    SelfDistillationConfig, ProgressiveConfig,
};

use crate::{SNNError, SNNResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all main types are exported
        let _ = DistillationMode::ResponseBased;
        let config = DistillationConfig::default();
        assert_eq!(config.temperature, 4.0);
    }
}
