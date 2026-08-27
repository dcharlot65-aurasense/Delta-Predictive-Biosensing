//! # dpb-synth - Synthetic Biosignal Generation
//!
//! Generates realistic synthetic biosignals for testing and training.
//!
//! ## Features
//!
//! - **Signal Generation**: ECG, EEG, EMG, PPG, respiratory, gaze, pose, and more
//! - **Pathology Models**: ALS, MS, stroke disease signatures
//! - **Augmentation**: 13+ signal augmentation techniques
//! - **Cohort Generation**: Virtual patient populations
//! - **Streaming**: Real-time signal generation for online systems
//! - **Ground Truth**: All generators provide ground truth labels
//!
//! ## Quick Start: Signal Generation
//!
//! ```rust
//! use dpb_synth::contact::ecg::*;
//! use dpb_synth::traits::SyntheticGenerator;
//!
//! # fn example() -> dpb_synth::Result<()> {
//! // Generators are unit structs; their parameters carry the configuration,
//! // and each generator supplies a physiologically sensible starting set.
//! let generator = EcgMorphologyGenerator;
//! let params = EcgMorphologyParams {
//!     heart_rate: 70.0,  // bpm
//!     ..EcgMorphologyGenerator::default_params()
//! };
//!
//! // The seed makes the output reproducible.
//! let output = generator.generate(&params, 42)?;
//!
//! println!("Generated {} samples", output.signal.len());
//! println!("Ground-truth events: {}", output.ground_truth.events.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Signal Augmentation
//!
//! ```rust
//! use dpb_synth::augmentation::*;
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//!
//! # fn example() {
//! let mut rng = ChaCha8Rng::seed_from_u64(42);
//! let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Build augmentation pipeline
//! let pipeline = AugmentationPipeline::new()
//!     .add(GaussianNoise::new(20.0), 0.8)  // 80% probability
//!     .add(TimeWarp::new(0.2, 4), 0.5)     // 50% probability
//!     .add(MagnitudeScale::new((0.9, 1.1)), 1.0);
//!
//! let augmented = pipeline.apply(&signal, &mut rng);
//! # }
//! ```
//!
//! ## Example: Virtual Patient Cohort
//!
//! ```rust
//! use dpb_synth::cohort::*;
//!
//! # fn example() {
//! // Generate virtual patient cohort
//! let generator = CohortGenerator::new(100)  // 100 patients
//!     .with_disease("Hypertension", 0.3)     // 30% prevalence
//!     .with_disease("Diabetes", 0.15)
//!     .with_age_distribution(50.0, 15.0);    // mean=50, std=15
//!
//! let cohort = generator.generate(42);  // seed=42
//!
//! for patient in &cohort {
//!     println!("Patient {}: age={}, HR={}",
//!         patient.id,
//!         patient.demographics.age_years,
//!         patient.baseline_hr
//!     );
//! }
//! # }
//! ```
//!
//! ## Example: Streaming Generation
//!
//! ```rust
//! use dpb_synth::streaming::*;
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//!
//! # fn example() -> dpb_synth::Result<()> {
//! let mut rng = ChaCha8Rng::seed_from_u64(42);
//!
//! // Create streaming ECG generator
//! let _config = StreamingConfig {
//!     sample_rate: 250.0,
//!     buffer_size: 1024,
//!     ..StreamingConfig::default()
//! };
//! let params = StreamingEcgParams {
//!     heart_rate: 75.0,
//!     ..StreamingEcgParams::default()
//! };
//!
//! // The generator is a unit struct; the STATE it hands back is what advances.
//! let generator = StreamingEcg;
//! let mut state = generator.init_state(&params, 42);
//!
//! // Generate samples in real-time
//! for _ in 0..1000 {
//!     let _sample = generator.next_sample(&mut state);
//!     // Process sample...
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`contact`] | Contact biosignals (ECG, EMG, EDA, PPG) |
//! | [`neural`] | Neural signals (EEG, ERP, sleep) |
//! | [`pose`] | Pose and movement signals |
//! | [`hand`] | Hand tracking and gestures |
//! | [`eye`] | Eye tracking and gaze |
//! | [`voice`] | Voice and speech signals |
//! | [`force`] | Force dynamics and grip |
//! | [`balance`] | Balance and posturography |
//! | [`vestibular`] | Vestibular system signals |
//! | [`pain`] | Pain and sensory testing |
//! | [`cardiopulmonary`] | Cardiopulmonary signals |
//! | [`cognitive`] | Cognitive task responses |
//! | [`pathology`] | Disease pathology models |
//! | [`augmentation`] | Signal augmentation |
//! | [`cohort`] | Virtual patient cohorts |
//! | [`streaming`] | Real-time signal generation |
//!
//! ## Augmentation Techniques
//!
//! ### Noise Augmentations
//! - [`GaussianNoise`]: White noise addition
//! - [`PinkNoise`]: 1/f noise addition
//! - [`BaselineWander`]: Low-frequency drift
//! - [`PowerlineNoise`]: 50/60 Hz interference
//! - [`MotionArtifact`]: Movement artifacts
//!
//! ### Temporal Augmentations
//! - [`TimeWarp`]: Non-linear time warping
//! - [`TimeShift`]: Circular time shift
//! - [`WindowCrop`]: Random cropping
//! - [`Resample`]: Sample rate changes
//! - [`RandomDropout`]: Sample dropout
//!
//! ### Spectral Augmentations
//! - [`MagnitudeScale`]: Amplitude scaling
//! - [`FrequencyMask`]: Frequency masking
//! - [`TimeMask`]: Time masking

pub mod contact;
pub mod eye;
pub mod hand;
pub mod level3;
pub mod media;
pub mod multimodal;
pub mod neural;
pub mod pose;
pub mod streaming;
pub mod traits;
pub mod voice;

// New biomechanical and clinical modules
pub mod balance;
pub mod cardiopulmonary;
pub mod cognitive;
pub mod force;
pub mod pain;
pub mod pathology;
pub mod vestibular;

// Augmentation and cohort modules
pub mod augmentation;
pub mod cohort;

pub use augmentation::{
    AugmentationPipeline, SignalAugmentation,
    noise::{BaselineWander, GaussianNoise, MotionArtifact, PinkNoise, PowerlineNoise},
    spectral::{FrequencyMask, MagnitudeScale, TimeMask},
    temporal::{RandomDropout, Resample, TimeShift, TimeWarp, WindowCrop},
};
pub use cohort::{CohortGenerator, Demographics, Sex, VirtualPatient};
pub use streaming::{
    AtomicRingBuffer,
    // Level 3 audio streaming
    AudioSample,
    // Level 3 clinical pose/hand streaming
    ClinicalGaitType,
    ClinicalHandFrame,
    ClinicalHandTask,
    ClinicalPoseFrame,
    DdkEvent,
    FrameStreamingGenerator,
    GazeSample,
    HandFrame,
    HandMotionType,
    MultiChannelBuffer,
    MultiModalParams,
    MultiModalSample,
    MultiModalState,
    // Multi-modal streaming
    MultiModalStreaming,
    PoseFrame,
    RingBuffer,
    RppgFrame,
    StreamingClinicalHand,
    StreamingClinicalHandParams,
    StreamingClinicalHandState,
    StreamingClinicalPose,
    StreamingClinicalPoseParams,
    StreamingClinicalPoseState,
    StreamingConfig,
    StreamingDdk,
    StreamingDdkParams,
    StreamingDdkState,
    // Contact biosignal streaming
    StreamingEcg,
    StreamingEcgParams,
    StreamingEcgState,
    StreamingEda,
    StreamingEdaParams,
    StreamingEdaState,
    StreamingEmg,
    StreamingEmgParams,
    StreamingEmgState,
    // Eye tracking streaming
    StreamingGaze,
    StreamingGazeParams,
    StreamingGazeState,
    StreamingGenerator,
    StreamingHand,
    StreamingHandParams,
    StreamingHandState,
    // Frame-based streaming (pose, hand)
    StreamingPose,
    StreamingPoseParams,
    StreamingPoseState,
    StreamingPpg,
    StreamingPpgParams,
    StreamingPpgState,
    StreamingRespiratory,
    StreamingRespiratoryParams,
    StreamingRespiratoryState,
    // rPPG (remote photoplethysmography) streaming
    StreamingRppg,
    StreamingRppgParams,
    StreamingRppgState,
    StreamingStats,
    StreamingThermal,
    StreamingThermalParams,
    StreamingThermalState,
    StreamingTremor,
    StreamingTremorParams,
    StreamingTremorState,
    StreamingVowel,
    StreamingVowelParams,
    StreamingVowelState,
};
pub use traits::{GroundTruth, ParameterSpace, SyntheticGenerator};

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
        // Compiling this test is the check: it names the module's items
        // through the imports above. An assert!(true) added nothing.
    }

    #[test]
    fn test_augmentation_pipeline() {
        use std::f64::consts::PI;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * i as f64 / 50.0).sin())
            .collect();

        let pipeline = AugmentationPipeline::new()
            .add(GaussianNoise::new(20.0), 1.0)
            .add(MagnitudeScale::new((0.9, 1.1)), 1.0);

        let augmented = pipeline.apply(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
        assert_ne!(augmented, signal); // Should be different due to augmentation
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
        // BMI 22.0 is unambiguously normal. The previous value of 25.0 sits
        // exactly ON the WHO cut-point, where "overweight" begins (normal is
        // 18.5-24.9), so the classifier was right to call it Overweight.
        let demographics = Demographics::new(45.0, Sex::Male, 22.0, None);
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
    /// WHO BMI cut-points, asserted exactly ON each boundary.
    ///
    /// Boundary values are where a classifier is worth testing, and picking one
    /// by accident is how the neighbouring test came to expect the wrong side
    /// of it.
    #[test]
    fn test_bmi_category_boundaries() {
        let categorize = |bmi: f64| {
            VirtualPatient::new(
                "VP".to_string(),
                Demographics::new(45.0, Sex::Male, bmi, None),
                70.0,
                40.0,
                vec![],
                vec![],
            )
            .bmi_category()
            .to_string()
        };

        for (bmi, expected) in [
            (16.0, "Underweight"),
            (18.4, "Underweight"),
            (18.5, "Normal"),
            (24.9, "Normal"),
            (25.0, "Overweight"),
            (29.9, "Overweight"),
            (30.0, "Obese"),
            (45.0, "Obese"),
        ] {
            assert_eq!(categorize(bmi), expected, "BMI {bmi}");
        }
    }
}
