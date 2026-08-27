//! Amyotrophic Lateral Sclerosis (ALS) disease model.
//!
//! ALS is a progressive motor neuron disease characterized by:
//! - Upper and lower motor neuron degeneration
//! - Progressive weakness and muscle atrophy
//! - Fasciculations and fibrillations
//! - Bulbar dysfunction (dysarthria, dysphagia)
//! - Respiratory compromise
//!
//! This module models EMG signatures, force decline, respiratory patterns,
//! and functional progression in ALS.

use super::{BodyRegion, DiseaseSignature, DiseaseStage, Laterality, SignalModulation};
use serde::{Deserialize, Serialize};

/// ALS disease stages (King's staging system)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlsStage {
    /// Stage 1: Symptom onset, one region
    Stage1,
    /// Stage 2a: Diagnosis, second region
    Stage2a,
    /// Stage 2b: Two regions with functional decline
    Stage2b,
    /// Stage 3: Three regions
    Stage3,
    /// Stage 4a: Need for gastrostomy
    Stage4a,
    /// Stage 4b: Need for respiratory support
    Stage4b,
}

impl AlsStage {
    /// Convert to generic disease stage
    pub fn to_disease_stage(self) -> DiseaseStage {
        match self {
            AlsStage::Stage1 => DiseaseStage::Early,
            AlsStage::Stage2a | AlsStage::Stage2b => DiseaseStage::Moderate,
            AlsStage::Stage3 => DiseaseStage::Advanced,
            AlsStage::Stage4a | AlsStage::Stage4b => DiseaseStage::EndStage,
        }
    }

    /// Get typical ALSFRS-R score range
    pub fn alsfrs_range(&self) -> (u8, u8) {
        match self {
            AlsStage::Stage1 => (40, 48),
            AlsStage::Stage2a => (35, 42),
            AlsStage::Stage2b => (28, 38),
            AlsStage::Stage3 => (20, 32),
            AlsStage::Stage4a => (12, 24),
            AlsStage::Stage4b => (0, 18),
        }
    }
}

/// ALS Functional Rating Scale - Revised (ALSFRS-R)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlsFunctionalRating {
    /// Speech (0-4)
    pub speech: u8,
    /// Salivation (0-4)
    pub salivation: u8,
    /// Swallowing (0-4)
    pub swallowing: u8,
    /// Handwriting (0-4)
    pub handwriting: u8,
    /// Cutting food / handling utensils (0-4)
    pub cutting_food: u8,
    /// Dressing and hygiene (0-4)
    pub dressing: u8,
    /// Turning in bed (0-4)
    pub turning_bed: u8,
    /// Walking (0-4)
    pub walking: u8,
    /// Climbing stairs (0-4)
    pub climbing_stairs: u8,
    /// Dyspnea (0-4)
    pub dyspnea: u8,
    /// Orthopnea (0-4)
    pub orthopnea: u8,
    /// Respiratory insufficiency (0-4)
    pub respiratory_insufficiency: u8,
}

impl AlsFunctionalRating {
    /// Calculate total ALSFRS-R score (0-48)
    pub fn total_score(&self) -> u8 {
        self.speech
            + self.salivation
            + self.swallowing
            + self.handwriting
            + self.cutting_food
            + self.dressing
            + self.turning_bed
            + self.walking
            + self.climbing_stairs
            + self.dyspnea
            + self.orthopnea
            + self.respiratory_insufficiency
    }

    /// Create healthy (normal) rating
    pub fn healthy() -> Self {
        Self {
            speech: 4,
            salivation: 4,
            swallowing: 4,
            handwriting: 4,
            cutting_food: 4,
            dressing: 4,
            turning_bed: 4,
            walking: 4,
            climbing_stairs: 4,
            dyspnea: 4,
            orthopnea: 4,
            respiratory_insufficiency: 4,
        }
    }

    /// Calculate bulbar subscore (0-12)
    pub fn bulbar_score(&self) -> u8 {
        self.speech + self.salivation + self.swallowing
    }

    /// Calculate fine motor subscore (0-12)
    pub fn fine_motor_score(&self) -> u8 {
        self.handwriting + self.cutting_food + self.dressing
    }

    /// Calculate gross motor subscore (0-12)
    pub fn gross_motor_score(&self) -> u8 {
        self.turning_bed + self.walking + self.climbing_stairs
    }

    /// Calculate respiratory subscore (0-12)
    pub fn respiratory_score(&self) -> u8 {
        self.dyspnea + self.orthopnea + self.respiratory_insufficiency
    }
}

/// EMG signature for ALS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlsEmgSignature {
    /// Presence of fibrillation potentials (0-4, 0=none, 4=abundant)
    pub fibrillations: u8,
    /// Positive sharp waves (0-4)
    pub positive_sharp_waves: u8,
    /// Fasciculation potentials per minute
    pub fasciculation_rate: f64,
    /// Motor unit action potential amplitude increase (ratio)
    pub muap_amplitude_ratio: f64,
    /// MUAP duration increase (ratio)
    pub muap_duration_ratio: f64,
    /// Polyphasia percentage
    pub polyphasia_percent: f64,
    /// Reduced recruitment pattern
    pub recruitment_reduction: f64,
    /// Firing rate instability (coefficient of variation)
    pub firing_cv: f64,
}

impl AlsEmgSignature {
    /// Create signature for early ALS
    pub fn early() -> Self {
        Self {
            fibrillations: 1,
            positive_sharp_waves: 1,
            fasciculation_rate: 2.0,
            muap_amplitude_ratio: 1.3,
            muap_duration_ratio: 1.2,
            polyphasia_percent: 15.0,
            recruitment_reduction: 0.2,
            firing_cv: 0.15,
        }
    }

    /// Create signature for moderate ALS
    pub fn moderate() -> Self {
        Self {
            fibrillations: 3,
            positive_sharp_waves: 3,
            fasciculation_rate: 5.0,
            muap_amplitude_ratio: 2.0,
            muap_duration_ratio: 1.8,
            polyphasia_percent: 35.0,
            recruitment_reduction: 0.5,
            firing_cv: 0.25,
        }
    }

    /// Create signature for advanced ALS
    pub fn advanced() -> Self {
        Self {
            fibrillations: 4,
            positive_sharp_waves: 4,
            fasciculation_rate: 8.0,
            muap_amplitude_ratio: 3.0,
            muap_duration_ratio: 2.5,
            polyphasia_percent: 50.0,
            recruitment_reduction: 0.8,
            firing_cv: 0.40,
        }
    }
}

/// Respiratory status in ALS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlsRespiratoryStatus {
    /// Forced vital capacity (% predicted)
    pub fvc_percent: f64,
    /// Maximal inspiratory pressure (cm H2O)
    pub mip: f64,
    /// Sniff nasal inspiratory pressure (cm H2O)
    pub snip: f64,
    /// Peak cough flow (L/min)
    pub pcf: f64,
    /// Nocturnal desaturation (hours with SpO2 < 90%)
    pub nocturnal_desat_hours: f64,
    /// NIV requirement
    pub niv_required: bool,
    /// Hours of NIV per day
    pub niv_hours: f64,
}

impl AlsRespiratoryStatus {
    /// Create normal respiratory status
    pub fn normal() -> Self {
        Self {
            fvc_percent: 95.0,
            mip: -80.0,
            snip: -70.0,
            pcf: 400.0,
            nocturnal_desat_hours: 0.0,
            niv_required: false,
            niv_hours: 0.0,
        }
    }

    /// Create status indicating NIV threshold
    pub fn niv_threshold() -> Self {
        Self {
            fvc_percent: 50.0,
            mip: -40.0,
            snip: -40.0,
            pcf: 270.0,
            nocturnal_desat_hours: 2.0,
            niv_required: true,
            niv_hours: 8.0,
        }
    }
}

/// Comprehensive ALS disease model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlsModel {
    /// Disease stage
    pub stage: AlsStage,
    /// Onset type (limb vs bulbar)
    pub onset_type: AlsOnsetType,
    /// Primary affected regions
    pub affected_regions: Vec<BodyRegion>,
    /// Functional rating
    pub alsfrs_r: AlsFunctionalRating,
    /// EMG signature
    pub emg_signature: AlsEmgSignature,
    /// Respiratory status
    pub respiratory: AlsRespiratoryStatus,
    /// Disease duration (months)
    pub duration_months: f64,
    /// Progression rate (ALSFRS-R points lost per month)
    pub progression_rate: f64,
    /// Laterality
    pub laterality: Laterality,
}

/// ALS onset type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlsOnsetType {
    /// Limb onset (most common, ~70%)
    LimbOnset,
    /// Bulbar onset (~25%)
    BulbarOnset,
    /// Respiratory onset (rare, ~5%)
    RespiratoryOnset,
}

impl AlsModel {
    /// Create early limb-onset ALS model
    pub fn early_limb_onset() -> Self {
        Self {
            stage: AlsStage::Stage1,
            onset_type: AlsOnsetType::LimbOnset,
            affected_regions: vec![BodyRegion::UpperLimb],
            alsfrs_r: AlsFunctionalRating {
                speech: 4,
                salivation: 4,
                swallowing: 4,
                handwriting: 3,
                cutting_food: 3,
                dressing: 4,
                turning_bed: 4,
                walking: 4,
                climbing_stairs: 4,
                dyspnea: 4,
                orthopnea: 4,
                respiratory_insufficiency: 4,
            },
            emg_signature: AlsEmgSignature::early(),
            respiratory: AlsRespiratoryStatus::normal(),
            duration_months: 6.0,
            progression_rate: 0.9,
            laterality: Laterality::Predominant(true),
        }
    }

    /// Create moderate ALS model
    pub fn moderate() -> Self {
        Self {
            stage: AlsStage::Stage2b,
            onset_type: AlsOnsetType::LimbOnset,
            affected_regions: vec![BodyRegion::UpperLimb, BodyRegion::LowerLimb],
            alsfrs_r: AlsFunctionalRating {
                speech: 4,
                salivation: 3,
                swallowing: 3,
                handwriting: 2,
                cutting_food: 2,
                dressing: 2,
                turning_bed: 3,
                walking: 2,
                climbing_stairs: 2,
                dyspnea: 3,
                orthopnea: 4,
                respiratory_insufficiency: 4,
            },
            emg_signature: AlsEmgSignature::moderate(),
            respiratory: AlsRespiratoryStatus {
                fvc_percent: 70.0,
                mip: -55.0,
                snip: -50.0,
                pcf: 320.0,
                nocturnal_desat_hours: 0.5,
                niv_required: false,
                niv_hours: 0.0,
            },
            duration_months: 18.0,
            progression_rate: 0.9,
            laterality: Laterality::Bilateral,
        }
    }

    /// Create advanced ALS model
    pub fn advanced() -> Self {
        Self {
            stage: AlsStage::Stage4b,
            onset_type: AlsOnsetType::LimbOnset,
            affected_regions: vec![
                BodyRegion::UpperLimb,
                BodyRegion::LowerLimb,
                BodyRegion::Bulbar,
                BodyRegion::Trunk,
            ],
            alsfrs_r: AlsFunctionalRating {
                speech: 2,
                salivation: 2,
                swallowing: 2,
                handwriting: 0,
                cutting_food: 0,
                dressing: 1,
                turning_bed: 1,
                walking: 0,
                climbing_stairs: 0,
                dyspnea: 2,
                orthopnea: 1,
                respiratory_insufficiency: 1,
            },
            emg_signature: AlsEmgSignature::advanced(),
            respiratory: AlsRespiratoryStatus::niv_threshold(),
            duration_months: 36.0,
            progression_rate: 0.9,
            laterality: Laterality::Bilateral,
        }
    }

    /// Generate multi-modal disease signature
    pub fn to_disease_signature(&self) -> DiseaseSignature {
        let severity = 1.0 - (self.alsfrs_r.total_score() as f64 / 48.0);

        // EMG: reduced amplitude, fasciculations, fibrillations
        let emg_mod = SignalModulation {
            amplitude_factor: 1.0 / self.emg_signature.muap_amplitude_ratio,
            frequency_shift: 0.0,
            noise_level: self.emg_signature.firing_cv * 0.1,
            tremor: if self.emg_signature.fasciculation_rate > 0.0 {
                Some((self.emg_signature.fasciculation_rate, 0.05))
            } else {
                None
            },
            latency_increase: 0.0,
            variability_factor: 1.0 + self.emg_signature.firing_cv,
        };

        // Force: weakness
        let force_mod = SignalModulation::weakness(severity);

        // Gait: slowed if lower limb affected
        let gait_mod = if self.affected_regions.contains(&BodyRegion::LowerLimb) {
            SignalModulation {
                amplitude_factor: 1.0 - severity * 0.5,
                latency_increase: severity * 0.2,
                variability_factor: 1.0 + severity * 0.6,
                ..Default::default()
            }
        } else {
            SignalModulation::default()
        };

        // HRV: autonomic involvement in later stages
        let hrv_mod = if severity > 0.5 {
            SignalModulation {
                variability_factor: 1.0 - severity * 0.3,
                ..Default::default()
            }
        } else {
            SignalModulation::default()
        };

        // Voice: bulbar involvement
        let voice_mod = if self.affected_regions.contains(&BodyRegion::Bulbar) {
            let bulbar_severity = 1.0 - (self.alsfrs_r.bulbar_score() as f64 / 12.0);
            SignalModulation {
                amplitude_factor: 1.0 - bulbar_severity * 0.4,
                frequency_shift: -bulbar_severity * 20.0, // lower pitch
                noise_level: bulbar_severity * 0.1,
                variability_factor: 1.0 + bulbar_severity * 0.5,
                ..Default::default()
            }
        } else {
            SignalModulation::default()
        };

        DiseaseSignature {
            disease: "als".to_string(),
            stage: self.stage.to_disease_stage(),
            affected_regions: self.affected_regions.clone(),
            laterality: self.laterality,
            emg_modulation: emg_mod,
            force_modulation: force_mod,
            gait_modulation: gait_mod,
            hrv_modulation: hrv_mod,
            eeg_modulation: SignalModulation::default(),
            voice_modulation: voice_mod,
            eye_modulation: SignalModulation::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alsfrs_scoring() {
        let healthy = AlsFunctionalRating::healthy();
        assert_eq!(healthy.total_score(), 48);
        assert_eq!(healthy.bulbar_score(), 12);
    }

    #[test]
    fn test_als_stages() {
        assert_eq!(AlsStage::Stage1.to_disease_stage(), DiseaseStage::Early);
        assert_eq!(AlsStage::Stage4b.to_disease_stage(), DiseaseStage::EndStage);
    }

    #[test]
    fn test_als_model_creation() {
        let early = AlsModel::early_limb_onset();
        assert_eq!(early.stage, AlsStage::Stage1);
        assert!(early.alsfrs_r.total_score() > 40);

        let advanced = AlsModel::advanced();
        assert!(advanced.alsfrs_r.total_score() < 20);
    }

    #[test]
    fn test_disease_signature() {
        let model = AlsModel::moderate();
        let sig = model.to_disease_signature();

        assert_eq!(sig.disease, "als");
        assert!(sig.force_modulation.amplitude_factor < 1.0);
    }
}
