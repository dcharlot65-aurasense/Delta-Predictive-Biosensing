//! Disease-specific pathology models for biosignal synthesis.
//!
//! This module provides comprehensive disease models that affect multiple
//! biosignal modalities, including:
//! - ALS (Amyotrophic Lateral Sclerosis)
//! - MS (Multiple Sclerosis)
//! - Stroke (Cerebrovascular Accident)
//!
//! Each disease model captures the characteristic signatures across:
//! - Motor function (EMG, force, gait)
//! - Autonomic function (HRV, EDA)
//! - Cognitive markers (EEG, reaction time)
//! - Respiratory patterns

pub mod als;
pub mod ms;
pub mod stroke;
pub mod progression;

pub use als::{AlsModel, AlsStage, AlsFunctionalRating, AlsEmgSignature, AlsRespiratoryStatus};
pub use ms::{MsModel, MsType, MsRelapse, MsSymptomProfile, EdssScore};
pub use stroke::{StrokeModel, StrokeType, StrokeLocation, StrokeSeverity, RecoveryPhase};
pub use progression::{DiseaseProgression, ProgressionRate, ProgressionPattern, TimePoint};

use serde::{Deserialize, Serialize};

/// Generic disease stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiseaseStage {
    /// Prodromal - subtle early changes
    Prodromal,
    /// Early - diagnosis made, mild symptoms
    Early,
    /// Moderate - functional impairment
    Moderate,
    /// Advanced - significant disability
    Advanced,
    /// End-stage - severe impairment
    EndStage,
}

impl DiseaseStage {
    /// Get numeric severity (0-4)
    pub fn severity(&self) -> u8 {
        match self {
            DiseaseStage::Prodromal => 0,
            DiseaseStage::Early => 1,
            DiseaseStage::Moderate => 2,
            DiseaseStage::Advanced => 3,
            DiseaseStage::EndStage => 4,
        }
    }
}

/// Affected body region
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodyRegion {
    /// Upper limbs (arms, hands)
    UpperLimb,
    /// Lower limbs (legs, feet)
    LowerLimb,
    /// Bulbar (speech, swallowing)
    Bulbar,
    /// Trunk (core, respiratory)
    Trunk,
    /// Cognitive (brain function)
    Cognitive,
    /// Sensory (vision, proprioception)
    Sensory,
}

/// Laterality of symptoms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Laterality {
    /// Symptoms on left side
    Left,
    /// Symptoms on right side
    Right,
    /// Symptoms on both sides
    Bilateral,
    /// Symptoms predominantly on one side
    Predominant(bool), // true = right predominant
}

/// Signal modulation factors from disease
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalModulation {
    /// Amplitude scaling factor
    pub amplitude_factor: f64,
    /// Frequency shift in Hz
    pub frequency_shift: f64,
    /// Added noise level (std dev)
    pub noise_level: f64,
    /// Tremor component (frequency, amplitude)
    pub tremor: Option<(f64, f64)>,
    /// Delay/latency increase (seconds)
    pub latency_increase: f64,
    /// Variability scaling
    pub variability_factor: f64,
}

impl Default for SignalModulation {
    fn default() -> Self {
        Self {
            amplitude_factor: 1.0,
            frequency_shift: 0.0,
            noise_level: 0.0,
            tremor: None,
            latency_increase: 0.0,
            variability_factor: 1.0,
        }
    }
}

impl SignalModulation {
    /// Create modulation for weakness
    pub fn weakness(severity: f64) -> Self {
        Self {
            amplitude_factor: 1.0 - severity * 0.7,
            variability_factor: 1.0 + severity * 0.5,
            ..Default::default()
        }
    }

    /// Create modulation for spasticity
    pub fn spasticity(severity: f64) -> Self {
        Self {
            frequency_shift: severity * 10.0, // increased firing rate
            variability_factor: 1.0 - severity * 0.3, // more stereotyped
            ..Default::default()
        }
    }

    /// Create modulation for tremor
    pub fn with_tremor(mut self, frequency: f64, amplitude: f64) -> Self {
        self.tremor = Some((frequency, amplitude));
        self
    }

    /// Create modulation for slowing
    pub fn slowing(severity: f64) -> Self {
        Self {
            latency_increase: severity * 0.3,
            variability_factor: 1.0 + severity * 0.4,
            ..Default::default()
        }
    }
}

/// Multi-modal disease signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiseaseSignature {
    /// Disease identifier
    pub disease: String,
    /// Current stage
    pub stage: DiseaseStage,
    /// Primary affected regions
    pub affected_regions: Vec<BodyRegion>,
    /// Laterality
    pub laterality: Laterality,
    /// EMG modulation
    pub emg_modulation: SignalModulation,
    /// Force modulation
    pub force_modulation: SignalModulation,
    /// Gait modulation
    pub gait_modulation: SignalModulation,
    /// HRV modulation
    pub hrv_modulation: SignalModulation,
    /// EEG modulation
    pub eeg_modulation: SignalModulation,
    /// Voice modulation
    pub voice_modulation: SignalModulation,
    /// Eye movement modulation
    pub eye_modulation: SignalModulation,
}

impl DiseaseSignature {
    /// Create signature for healthy individual
    pub fn healthy() -> Self {
        Self {
            disease: "healthy".to_string(),
            stage: DiseaseStage::Prodromal, // Not applicable
            affected_regions: vec![],
            laterality: Laterality::Bilateral,
            emg_modulation: SignalModulation::default(),
            force_modulation: SignalModulation::default(),
            gait_modulation: SignalModulation::default(),
            hrv_modulation: SignalModulation::default(),
            eeg_modulation: SignalModulation::default(),
            voice_modulation: SignalModulation::default(),
            eye_modulation: SignalModulation::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disease_stage_severity() {
        assert_eq!(DiseaseStage::Prodromal.severity(), 0);
        assert_eq!(DiseaseStage::EndStage.severity(), 4);
    }

    #[test]
    fn test_signal_modulation_weakness() {
        let mod_mild = SignalModulation::weakness(0.3);
        let mod_severe = SignalModulation::weakness(0.8);

        assert!(mod_mild.amplitude_factor > mod_severe.amplitude_factor);
    }

    #[test]
    fn test_healthy_signature() {
        let sig = DiseaseSignature::healthy();
        assert_eq!(sig.disease, "healthy");
        assert!(sig.affected_regions.is_empty());
    }
}
