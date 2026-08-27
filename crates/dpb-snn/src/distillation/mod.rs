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
//! let teacher = FeedforwardSNN::new(vec![128, 256, 128, 10], SNNConfig::default(), true)?;
//!
//! // Create student (smaller model)
//! let student = FeedforwardSNN::new(vec![128, 64, 10], SNNConfig::default(), true)?;
//!
//! // Configure distillation
//! let config = DistillationConfig {
//!     temperature: 4.0,
//!     alpha: 0.7,  // Weight for distillation loss
//!     beta: 0.3,   // Weight for student loss
//!     mode: DistillationMode::ResponseBased,
//!     ..DistillationConfig::default()
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

pub mod compression;
pub mod losses;
pub mod self_distillation;
pub mod spike_distillation;
pub mod teacher_student;

// Re-export main types
pub use compression::{
    ArchitectureSearch, ChannelPruningGuided, CompressionConfig, CompressionMetrics, LayerMerging,
    QuantizationAwareDistillation, SearchStrategy,
};
pub use losses::{
    AttentionTransferLoss, CombinedDistillationLoss, CosineSimLoss, DistillationLoss, HintLoss,
    KLDivergenceLoss, LossWeights, MSELoss,
};
pub use self_distillation::{
    BornAgainNetworks, ProgressiveConfig, ProgressiveDistillation, SelfDistillation,
    SelfDistillationConfig,
};
pub use spike_distillation::{
    MembranePotentialDistillation, SpikeDistillationConfig, SpikePatternDistillation,
    SpikeRateDistillation, SynapticWeightTransfer, TemporalCreditAssignment,
};
pub use teacher_student::{
    DistillationConfig, DistillationMode, KnowledgeTransfer, StudentModel, TeacherModel,
    TeacherStudentFramework,
};

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
