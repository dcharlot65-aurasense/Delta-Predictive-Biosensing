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
//!
//! ### Disease Pathology Modules (New)
//! - [`pathology`] - Disease-specific models (ALS, MS, stroke)
//!
//! ### Augmentation and Cohort Modules
//! - [`augmentation`] - Signal augmentation (noise, temporal, spectral)
//! - [`cohort`] - Virtual patient cohort generation

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
pub mod pathology;

// Augmentation and cohort modules
pub mod augmentation;
pub mod cohort;

pub use traits::{SyntheticGenerator, GroundTruth, ParameterSpace};
pub use augmentation::{
    SignalAugmentation, AugmentationPipeline,
    noise::{GaussianNoise, PinkNoise, BaselineWander, PowerlineNoise, MotionArtifact},
    temporal::{TimeWarp, TimeShift, WindowCrop, Resample, RandomDropout},
    spectral::{MagnitudeScale, FrequencyMask, TimeMask},
};
pub use cohort::{Sex, Demographics, VirtualPatient, CohortGenerator};
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
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_module_structure() {
        // Smoke test to ensure modules compile
        assert!(true);
    }

    #[test]
    fn test_augmentation_pipeline() {
        use std::f64::consts::PI;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal: Vec<f64> = (0..1000).map(|i| (2.0 * PI * i as f64 / 50.0).sin()).collect();

        let pipeline = AugmentationPipeline::new()
            .add(GaussianNoise::new(20.0), 1.0)
            .add(MagnitudeScale::new((0.9, 1.1)), 1.0);

        let augmented = pipeline.apply(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
        assert_ne!(augmented, signal);  // Should be different due to augmentation
    }

    #[test]
    fn test_cohort_generation() {
        let generator = CohortGenerator::new(10)
            .with_disease("Hypertension", 0.3)
            .with_age_distribution(50.0, 15.0);

        let cohort = generator.generate(42);

        assert_eq!(cohort.len(), 10);

        for patient in &cohort {
            assert!(patient.demographics.age_years > 0.0);
            assert!(patient.baseline_hr > 0.0);
            assert!(patient.baseline_hrv >= 0.0);
        }
    }

    #[test]
    fn test_augmentation_types() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test noise augmentations
        let gaussian = GaussianNoise::new(10.0);
        assert_eq!(gaussian.name(), "GaussianNoise");
        let _ = gaussian.augment(&signal, &mut rng);

        let pink = PinkNoise::new(15.0);
        assert_eq!(pink.name(), "PinkNoise");
        let _ = pink.augment(&signal, &mut rng);

        // Test temporal augmentations
        let time_warp = TimeWarp::new(0.2, 3);
        assert_eq!(time_warp.name(), "TimeWarp");
        let _ = time_warp.augment(&signal, &mut rng);

        let time_shift = TimeShift::new(2);
        assert_eq!(time_shift.name(), "TimeShift");
        let _ = time_shift.augment(&signal, &mut rng);

        // Test spectral augmentations
        let mag_scale = MagnitudeScale::new((0.8, 1.2));
        assert_eq!(mag_scale.name(), "MagnitudeScale");
        let scaled = mag_scale.augment(&signal, &mut rng);
        assert_eq!(scaled.len(), signal.len());
    }

    #[test]
    fn test_virtual_patient() {
        let demographics = Demographics::new(45.0, Sex::Male, 25.0, None);
        let patient = VirtualPatient::new(
            "VP001".to_string(),
            demographics,
            70.0,
            40.0,
            vec!["Hypertension".to_string()],
            vec!["ACE Inhibitor".to_string()],
        );

        assert_eq!(patient.id, "VP001");
        assert_eq!(patient.age_category(), "Middle Aged");
        assert_eq!(patient.bmi_category(), "Normal");
        assert!(patient.has_condition("Hypertension"));
        assert!(patient.takes_medication("ACE Inhibitor"));
        assert!(!patient.has_condition("Diabetes"));
    }
}
