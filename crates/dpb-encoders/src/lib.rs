//! # DPB Encoders - Event-Based Encoders for Biosignal Processing
//!
//! This crate implements 75 event-based encoders and 62 population templates for the
//! Delta-Predictive Biosensing (DPB) Framework.
//!
//! ## Overview
//!
//! Event-based encoders convert continuous biosignals into discrete spike events,
//! enabling neuromorphic processing with Spiking Neural Networks (SNNs). Each encoder
//! is paired with population templates that provide clinical priors and normative data.
//!
//! ## Architecture
//!
//! The encoders are organized by modality:
//!
//! - **Contact Modalities**: ECG, PPG, EDA, EMG, Tremor
//! - **Pose/Gait**: Heel strike, toe-off, gait phase, stride analysis
//! - **Hand Tracking**: Finger tapping, hand tremor
//! - **Eye Tracking**: Saccades, fixations, microsaccades, pupil responses
//! - **Voice**: Phonation (F0, jitter, shimmer, HNR), articulation, prosody
//!
//! ## Quick Start
//!
//! ```rust
//! use dpb_encoders::prelude::*;
//! use dpb_core::{SignalBuffer, Context};
//!
//! # fn example() -> dpb_core::Result<()> {
//! // Create a signal
//! let data = vec![0.0, 0.5, 1.0, 0.5, 0.0];
//! let signal = SignalBuffer::single_channel(data, 100.0);
//!
//! // Configure encoder
//! let config = LevelCrossingConfig {
//!     threshold: 0.3,
//!     relative: false,
//!     refractory_period: 0.01,
//! };
//!
//! // Encode signal
//! let encoder = LevelCrossingEncoder::new("example");
//! let events = encoder.encode(&signal, &config)?;
//!
//! println!("Generated {} events", events.len());
//! # Ok(())
//! # }
//! ```
//!
//! ## Encoder Types
//!
//! ### Base Encoders
//!
//! - [`LevelCrossingEncoder`]: Detects threshold crossings
//! - [`TemplateDeviationEncoder`]: Matches against expected patterns
//! - [`DerivativeEncoder`]: Detects rapid signal changes
//! - [`DiscreteEventEncoder`]: Converts pre-detected events to spikes
//!
//! ### Contact Modalities
//!
//! **ECG (Electrocardiogram)**:
//! - [`EcgRPeakEncoder`]: R-peak detection
//! - [`EcgMorphologyEncoder`]: QRS morphology analysis
//! - [`EcgStDeviationEncoder`]: ST-segment deviation
//! - [`EcgHrvEncoder`]: Heart rate variability
//!
//! **PPG (Photoplethysmography)**:
//! - [`PpgPulseEncoder`]: Pulse detection
//! - [`PpgAmplitudeEncoder`]: Perfusion changes
//! - [`PpgPttEncoder`]: Pulse transit time
//!
//! **EDA (Electrodermal Activity)**:
//! - [`EdaLevelCrossingEncoder`]: Tonic level changes
//! - [`EdaScrEncoder`]: Skin conductance responses
//! - [`EdaTonicEncoder`]: Baseline level analysis
//!
//! **EMG (Electromyography)**:
//! - [`EmgBurstEncoder`]: Muscle activation bursts
//! - [`EmgAmplitudeEncoder`]: Amplitude analysis
//! - [`EmgFatigueEncoder`]: Fatigue detection
//!
//! **Tremor**:
//! - [`TremorLevelCrossingEncoder`]: 3-axis tremor detection
//! - [`TremorFrequencyEncoder`]: Frequency analysis
//! - [`TremorAmplitudeEncoder`]: Amplitude tracking
//!
//! ### Pose/Gait Encoders
//!
//! - [`HeelStrikeEncoder`]: Heel contact detection
//! - [`ToeOffEncoder`]: Toe-off detection
//! - [`GaitPhaseEncoder`]: Stance/swing phase analysis
//! - [`StrideTimeEncoder`]: Stride timing analysis
//! - [`GaitAsymmetryEncoder`]: Left-right asymmetry
//! - [`KeypointDeviationEncoder`]: Pose deviation from template
//! - [`JointAngleEncoder`]: Joint angle analysis
//! - [`BodySwayEncoder`]: Center of mass stability
//!
//! ### Hand Tracking Encoders
//!
//! - [`TapOnsetEncoder`]: Finger tap detection
//! - [`TapApertureEncoder`]: Aperture analysis
//! - [`TapFrequencyEncoder`]: Tapping rate
//! - [`TapDecrementEncoder`]: Amplitude decay
//! - [`HandTremorEncoder`]: Hand tremor detection
//!
//! ### Eye Tracking Encoders
//!
//! - [`SaccadeOnsetEncoder`]: Saccade detection
//! - [`SaccadeMainSequenceEncoder`]: Main sequence analysis
//! - [`SaccadeLatencyEncoder`]: Response latency
//! - [`FixationStabilityEncoder`]: Fixation stability
//! - [`MicrosaccadeEncoder`]: Microsaccade detection
//! - [`PupilDilationEncoder`]: Pupil size changes
//! - [`PupilLightReflexEncoder`]: Light reflex response
//!
//! ### Voice Encoders
//!
//! **Phonation**:
//! - [`F0Encoder`]: Fundamental frequency
//! - [`JitterEncoder`]: Pitch perturbation
//! - [`ShimmerEncoder`]: Amplitude perturbation
//! - [`HnrEncoder`]: Harmonics-to-noise ratio
//!
//! **Articulation**:
//! - [`FormantEncoder`]: Formant tracking
//! - [`VowelSpaceEncoder`]: Vowel space analysis
//!
//! **Prosody**:
//! - [`SpeechRateEncoder`]: Speech rate analysis
//! - [`PauseEncoder`]: Pause detection
//! - [`IntonationEncoder`]: Pitch contour
//!
//! ## Population Templates
//!
//! All encoders are paired with population templates providing clinical norms:
//!
//! ```rust
//! use dpb_encoders::templates::*;
//! use dpb_core::{Context, PopulationTemplate};
//!
//! let context = create_context(
//!     Some(30.0),      // age
//!     Some("Male"),    // sex
//!     Some(175.0),     // height in cm
//!     Some(75.0),      // weight in kg
//! );
//!
//! let hr_template = HeartRateTemplate;
//! let expected_hr = hr_template.expected_value(&context);
//! let variance = hr_template.variance(&context);
//!
//! println!("Expected HR: {} ± {} bpm", expected_hr, variance.sqrt());
//! ```
//!
//! ## Template Registry
//!
//! Access all templates through the registry:
//!
//! ```rust
//! use dpb_encoders::templates::TemplateRegistry;
//! use dpb_core::Context;
//!
//! let registry = TemplateRegistry::new();
//! println!("Available templates: {}", registry.count());
//!
//! let context = Context::default();
//! let results = registry.evaluate_all(&context);
//!
//! for (name, expected, variance) in results {
//!     println!("{}: {} ± {}", name, expected, variance.sqrt());
//! }
//! ```
//!
//! ## Features
//!
//! - **77+ Event Encoders**: Comprehensive coverage of biosignals
//! - **61+ Population Templates**: Clinical priors and normative data
//! - **Type-Safe**: Strong typing with compile-time guarantees
//! - **Configurable**: Flexible configuration for each encoder
//! - **Efficient**: Optimized for real-time processing
//! - **Extensible**: Easy to add custom encoders and templates
//!
//! ## Example: Multi-Modal Encoding
//!
//! ```rust
//! use dpb_encoders::prelude::*;
//! use dpb_core::SignalBuffer;
//!
//! # fn example() -> dpb_core::Result<()> {
//! # let ecg_data = vec![0.0; 100];
//! # let ppg_data = vec![0.0; 100];
//! // ECG signal
//! let ecg_signal = SignalBuffer::single_channel(ecg_data, 250.0);
//!
//! // PPG signal
//! let ppg_signal = SignalBuffer::single_channel(ppg_data, 100.0);
//!
//! // Encode ECG
//! let ecg_encoder = EcgRPeakEncoder::new();
//! let ecg_config = EcgRPeakConfig::default();
//! let ecg_events = ecg_encoder.encode(&ecg_signal, &ecg_config)?;
//!
//! // Encode PPG
//! let ppg_encoder = PpgPulseEncoder::new();
//! let ppg_config = PpgPulseConfig::default();
//! let ppg_events = ppg_encoder.encode(&ppg_signal, &ppg_config)?;
//!
//! println!("ECG: {} events, PPG: {} events", ecg_events.len(), ppg_events.len());
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

// Re-export dpb-core types
pub use dpb_core::{
    Context, DpbError, EventEncoder, PopulationTemplate, Result, Signal, SignalBuffer, SpikeEvent,
};

// Base encoder implementations
pub mod base;

// Modality-specific encoders
pub mod contact;
pub mod eye;
pub mod hand;
pub mod pose;
pub mod voice;

// New modality encoders (Phase D)
pub mod balance;
pub mod cardiopulmonary;
pub mod cognitive;
pub mod eeg;
pub mod force;
pub mod pain;
pub mod vestibular;

// Population templates
pub mod templates;

// Re-exports for convenience
pub use balance::*;
pub use base::*;
pub use cardiopulmonary::*;
pub use cognitive::*;
pub use contact::*;
pub use eeg::*;
pub use eye::*;
pub use force::*;
pub use hand::*;
pub use pain::*;
pub use pose::*;
pub use vestibular::*;
pub use voice::*;

/// Prelude module for convenient imports
pub mod prelude {
    // Base encoders
    pub use crate::base::*;

    // Contact modality encoders
    pub use crate::contact::ecg::*;
    pub use crate::contact::eda::*;
    pub use crate::contact::emg::*;
    pub use crate::contact::ppg::*;
    pub use crate::contact::tremor::*;

    // Movement and tracking encoders
    pub use crate::eye::*;
    pub use crate::hand::*;
    pub use crate::pose::*;
    pub use crate::voice::*;

    // New modality encoders (Phase D)
    pub use crate::balance::*;
    pub use crate::cardiopulmonary::*;
    pub use crate::cognitive::*;
    pub use crate::eeg::*;
    pub use crate::force::*;
    pub use crate::pain::*;
    pub use crate::vestibular::*;

    // Templates
    pub use crate::templates;

    // Core re-exports
    pub use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_count() {
        // Verify we have all major encoder types accessible
        // This is a compile-time check that all types are exported
        let _: LevelCrossingEncoder;
        let _: EcgRPeakEncoder;
        let _: PpgPulseEncoder;
        let _: EdaLevelCrossingEncoder;
        let _: TremorLevelCrossingEncoder;
        let _: EmgBurstEncoder;
        let _: HeelStrikeEncoder;
        let _: TapOnsetEncoder;
        let _: SaccadeOnsetEncoder;
        let _: F0Encoder;
    }

    #[test]
    fn test_template_count() {
        let registry = templates::TemplateRegistry::new();
        assert!(registry.count() >= 36, "Should have at least 36 templates");
    }
}
