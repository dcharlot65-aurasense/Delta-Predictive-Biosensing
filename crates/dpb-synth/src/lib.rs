//! Synthetic Data Generation for DPB Framework
//!
//! This crate provides 200+ synthetic biosignal generators for validation and testing
//! of the Delta-Predictive Biosensing Framework. All generators support:
//! - Seeded RNG for reproducibility
//! - Ground truth output
//! - Parameter sweeps
//! - Clinical validity
//!
//! ## Module Organization
//!
//! ### Core Signal Modules
//! - [`contact`] - Contact-based biosignals (ECG, EMG, EDA, etc.)
//! - [`neural`] - Neural signals (EEG, ERP, sleep microstructure)
//! - [`pose`] - Pose and movement signals
//! - [`hand`] - Hand tracking and gestures
//! - [`eye`] - Eye tracking and gaze
//!
//! ### Biomechanical Modules (New)
//! - [`force`] - Force dynamics (GRF, grip strength, RFD)
//! - [`balance`] - Balance and COP (posturography, perturbation)
//! - [`vestibular`] - Vestibular signals (VOR, nystagmus, caloric)
//!
//! ### Clinical Protocol Modules (New)
//! - [`pain`] - Pain and sensory testing (QST, temporal summation)
//! - [`cardiopulmonary`] - Cardiorespiratory signals (HRV, respiratory)
//! - [`cognitive`] - Cognitive task responses (RT, accuracy, d-prime)

pub mod traits;
pub mod streaming;
pub mod contact;
pub mod pose;
pub mod hand;
pub mod eye;
pub mod voice;
pub mod neural;
pub mod multimodal;
pub mod level3;
pub mod media;

// New biomechanical and clinical modules
pub mod force;
pub mod balance;
pub mod vestibular;
pub mod pain;
pub mod cardiopulmonary;
pub mod cognitive;

pub use traits::{SyntheticGenerator, GroundTruth, ParameterSpace};
pub use streaming::{
    StreamingGenerator, FrameStreamingGenerator,
    RingBuffer, AtomicRingBuffer, MultiChannelBuffer,
    StreamingConfig, StreamingStats,
    // Contact biosignal streaming
    StreamingEcg, StreamingEcgState, StreamingEcgParams,
    StreamingTremor, StreamingTremorState, StreamingTremorParams,
    StreamingPpg, StreamingPpgState, StreamingPpgParams,
    StreamingEmg, StreamingEmgState, StreamingEmgParams,
    StreamingEda, StreamingEdaState, StreamingEdaParams,
    StreamingRespiratory, StreamingRespiratoryState, StreamingRespiratoryParams,
    StreamingThermal, StreamingThermalState, StreamingThermalParams,
    // Eye tracking streaming
    StreamingGaze, StreamingGazeState, StreamingGazeParams, GazeSample,
    // Frame-based streaming (pose, hand)
    StreamingPose, StreamingPoseState, StreamingPoseParams, PoseFrame,
    StreamingHand, StreamingHandState, StreamingHandParams, HandFrame, HandMotionType,
    // rPPG (remote photoplethysmography) streaming
    StreamingRppg, StreamingRppgState, StreamingRppgParams, RppgFrame,
    // Level 3 audio streaming
    AudioSample,
    StreamingVowel, StreamingVowelState, StreamingVowelParams,
    StreamingDdk, StreamingDdkState, StreamingDdkParams, DdkEvent,
    // Level 3 clinical pose/hand streaming
    ClinicalGaitType, ClinicalPoseFrame,
    StreamingClinicalPose, StreamingClinicalPoseState, StreamingClinicalPoseParams,
    ClinicalHandTask, ClinicalHandFrame,
    StreamingClinicalHand, StreamingClinicalHandState, StreamingClinicalHandParams,
    // Multi-modal streaming
    MultiModalStreaming, MultiModalState, MultiModalParams, MultiModalSample,
};

/// Common result type for generators
pub type Result<T> = std::result::Result<T, GeneratorError>;

/// Generator errors
#[derive(Debug, thiserror::Error)]
pub enum GeneratorError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Smoke test to ensure modules compile
        assert!(true);
    }
}
