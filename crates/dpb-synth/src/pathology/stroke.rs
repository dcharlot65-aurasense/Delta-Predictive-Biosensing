//! Stroke (Cerebrovascular Accident) disease model.
//!
//! Stroke produces focal neurological deficits based on:
//! - Lesion location (anterior/posterior circulation)
//! - Lesion size and mechanism (ischemic vs hemorrhagic)
//! - Time since onset and recovery phase
//!
//! This module models hemiparesis, aphasia, neglect, ataxia,
//! and recovery trajectories following stroke.

use super::{BodyRegion, DiseaseSignature, DiseaseStage, Laterality, SignalModulation};
use serde::{Deserialize, Serialize};

/// Stroke type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrokeType {
    /// Ischemic stroke (most common, ~85%)
    Ischemic,
    /// Intracerebral hemorrhage
    Hemorrhagic,
    /// Subarachnoid hemorrhage
    Subarachnoid,
    /// Transient ischemic attack
    Tia,
}

/// Stroke location
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrokeLocation {
    /// Middle cerebral artery (most common)
    Mca,
    /// Anterior cerebral artery
    Aca,
    /// Posterior cerebral artery
    Pca,
    /// Basilar artery (brainstem)
    Basilar,
    /// Cerebellar
    Cerebellar,
    /// Lacunar (small vessel)
    Lacunar,
    /// Thalamic
    Thalamic,
    /// Brainstem (other)
    Brainstem,
}

impl StrokeLocation {
    /// Get typical clinical features
    pub fn clinical_features(&self) -> Vec<StrokeDeficit> {
        match self {
            StrokeLocation::Mca => vec![
                StrokeDeficit::Hemiparesis,
                StrokeDeficit::SensoryLoss,
                StrokeDeficit::Aphasia, // if dominant
                StrokeDeficit::Neglect, // if non-dominant
            ],
            StrokeLocation::Aca => vec![
                StrokeDeficit::LegWeakness,
                StrokeDeficit::ExecutiveDysfunction,
            ],
            StrokeLocation::Pca => vec![
                StrokeDeficit::VisualFieldDefect,
                StrokeDeficit::MemoryImpairment,
            ],
            StrokeLocation::Basilar | StrokeLocation::Brainstem => vec![
                StrokeDeficit::Ataxia,
                StrokeDeficit::Dysarthria,
                StrokeDeficit::Dysphagia,
                StrokeDeficit::Vertigo,
            ],
            StrokeLocation::Cerebellar => vec![
                StrokeDeficit::Ataxia,
                StrokeDeficit::Vertigo,
                StrokeDeficit::Dysarthria,
            ],
            StrokeLocation::Lacunar => vec![StrokeDeficit::PureMotor],
            StrokeLocation::Thalamic => vec![StrokeDeficit::SensoryLoss, StrokeDeficit::Pain],
        }
    }
}

/// Stroke deficits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrokeDeficit {
    /// Contralateral hemiparesis
    Hemiparesis,
    /// Leg predominant weakness (ACA)
    LegWeakness,
    /// Pure motor (lacunar)
    PureMotor,
    /// Sensory loss
    SensoryLoss,
    /// Language impairment
    Aphasia,
    /// Spatial neglect
    Neglect,
    /// Visual field cut
    VisualFieldDefect,
    /// Cerebellar ataxia
    Ataxia,
    /// Speech motor impairment
    Dysarthria,
    /// Swallowing impairment
    Dysphagia,
    /// Vertigo/dizziness
    Vertigo,
    /// Memory problems
    MemoryImpairment,
    /// Executive dysfunction
    ExecutiveDysfunction,
    /// Central post-stroke pain
    Pain,
}

/// Stroke severity (NIHSS-based)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrokeSeverity {
    /// NIHSS 0-4
    Minor,
    /// NIHSS 5-15
    Moderate,
    /// NIHSS 16-20
    ModeratelySevere,
    /// NIHSS 21-42
    Severe,
}

impl StrokeSeverity {
    /// Create from NIHSS score
    pub fn from_nihss(nihss: u8) -> Self {
        match nihss {
            0..=4 => StrokeSeverity::Minor,
            5..=15 => StrokeSeverity::Moderate,
            16..=20 => StrokeSeverity::ModeratelySevere,
            _ => StrokeSeverity::Severe,
        }
    }

    /// Get typical recovery potential
    pub fn recovery_potential(&self) -> f64 {
        match self {
            StrokeSeverity::Minor => 0.9,
            StrokeSeverity::Moderate => 0.7,
            StrokeSeverity::ModeratelySevere => 0.5,
            StrokeSeverity::Severe => 0.3,
        }
    }

    /// Convert to disease stage
    pub fn to_disease_stage(self) -> DiseaseStage {
        match self {
            StrokeSeverity::Minor => DiseaseStage::Early,
            StrokeSeverity::Moderate => DiseaseStage::Moderate,
            StrokeSeverity::ModeratelySevere => DiseaseStage::Advanced,
            StrokeSeverity::Severe => DiseaseStage::EndStage,
        }
    }
}

/// Recovery phase after stroke
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecoveryPhase {
    /// Hyperacute (0-24 hours)
    Hyperacute,
    /// Acute (1-7 days)
    Acute,
    /// Early subacute (1 week - 3 months)
    EarlySubacute,
    /// Late subacute (3-6 months)
    LateSubacute,
    /// Chronic (> 6 months)
    Chronic,
}

impl RecoveryPhase {
    /// Days since stroke onset
    pub fn from_days(days: u32) -> Self {
        match days {
            0 => RecoveryPhase::Hyperacute,
            1..=7 => RecoveryPhase::Acute,
            8..=90 => RecoveryPhase::EarlySubacute,
            91..=180 => RecoveryPhase::LateSubacute,
            _ => RecoveryPhase::Chronic,
        }
    }

    /// Expected recovery rate (proportion of remaining deficit recovered per month)
    pub fn recovery_rate(&self) -> f64 {
        match self {
            RecoveryPhase::Hyperacute => 0.0, // Still evolving
            RecoveryPhase::Acute => 0.15,
            RecoveryPhase::EarlySubacute => 0.25, // Peak recovery
            RecoveryPhase::LateSubacute => 0.10,
            RecoveryPhase::Chronic => 0.02, // Plateau
        }
    }
}

/// Motor recovery stage (Brunnstrom)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrunnstromStage {
    /// Flaccidity
    Stage1,
    /// Spasticity develops, synergies begin
    Stage2,
    /// Spasticity peaks, synergy patterns
    Stage3,
    /// Spasticity decreases, some isolated movements
    Stage4,
    /// Synergies wane, more discrete movements
    Stage5,
    /// Coordinated, near-normal movement
    Stage6,
}

impl BrunnstromStage {
    /// Get spasticity level (0-1)
    pub fn spasticity(&self) -> f64 {
        match self {
            BrunnstromStage::Stage1 => 0.0,
            BrunnstromStage::Stage2 => 0.4,
            BrunnstromStage::Stage3 => 1.0,
            BrunnstromStage::Stage4 => 0.6,
            BrunnstromStage::Stage5 => 0.3,
            BrunnstromStage::Stage6 => 0.1,
        }
    }

    /// Get synergy pattern strength (0-1)
    pub fn synergy(&self) -> f64 {
        match self {
            BrunnstromStage::Stage1 => 0.0,
            BrunnstromStage::Stage2 => 0.5,
            BrunnstromStage::Stage3 => 1.0,
            BrunnstromStage::Stage4 => 0.6,
            BrunnstromStage::Stage5 => 0.2,
            BrunnstromStage::Stage6 => 0.0,
        }
    }

    /// Get voluntary control (0-1)
    pub fn voluntary_control(&self) -> f64 {
        match self {
            BrunnstromStage::Stage1 => 0.0,
            BrunnstromStage::Stage2 => 0.1,
            BrunnstromStage::Stage3 => 0.2,
            BrunnstromStage::Stage4 => 0.5,
            BrunnstromStage::Stage5 => 0.8,
            BrunnstromStage::Stage6 => 0.95,
        }
    }
}

/// Comprehensive stroke model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrokeModel {
    /// Stroke type
    pub stroke_type: StrokeType,
    /// Lesion location
    pub location: StrokeLocation,
    /// Affected hemisphere
    pub hemisphere: StrokeHemisphere,
    /// Severity (NIHSS-based)
    pub severity: StrokeSeverity,
    /// NIHSS score (0-42)
    pub nihss: u8,
    /// Days since onset
    pub days_since_onset: u32,
    /// Recovery phase
    pub recovery_phase: RecoveryPhase,
    /// Upper extremity Brunnstrom stage
    pub ue_brunnstrom: BrunnstromStage,
    /// Lower extremity Brunnstrom stage
    pub le_brunnstrom: BrunnstromStage,
    /// Present deficits
    pub deficits: Vec<StrokeDeficit>,
    /// Aphasia type (if present)
    pub aphasia_type: Option<AphasiaType>,
    /// Walking ability
    pub walking_ability: WalkingAbility,
}

/// Affected hemisphere
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrokeHemisphere {
    /// Left hemisphere (right-sided weakness)
    Left,
    /// Right hemisphere (left-sided weakness)
    Right,
    /// Bilateral or midline
    Bilateral,
}

impl StrokeHemisphere {
    /// Get affected side (opposite to lesion)
    pub fn affected_side(&self) -> Laterality {
        match self {
            StrokeHemisphere::Left => Laterality::Right,
            StrokeHemisphere::Right => Laterality::Left,
            StrokeHemisphere::Bilateral => Laterality::Bilateral,
        }
    }
}

/// Aphasia type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AphasiaType {
    /// Non-fluent, poor repetition (Broca's)
    Broca,
    /// Fluent but impaired comprehension (Wernicke's)
    Wernicke,
    /// Both production and comprehension impaired
    Global,
    /// Impaired repetition with preserved fluency
    Conduction,
    /// Impaired naming primarily
    Anomic,
    /// Impaired comprehension, preserved repetition
    Transcortical,
}

/// Walking ability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WalkingAbility {
    /// Unable to walk
    NonAmbulatory,
    /// Requires maximal assist
    MaximalAssist,
    /// Requires moderate assist
    ModerateAssist,
    /// Requires minimal assist or device
    MinimalAssist,
    /// Independent with device
    IndependentDevice,
    /// Fully independent
    Independent,
}

impl WalkingAbility {
    /// Get walking speed factor (0-1)
    pub fn speed_factor(&self) -> f64 {
        match self {
            WalkingAbility::NonAmbulatory => 0.0,
            WalkingAbility::MaximalAssist => 0.15,
            WalkingAbility::ModerateAssist => 0.3,
            WalkingAbility::MinimalAssist => 0.5,
            WalkingAbility::IndependentDevice => 0.7,
            WalkingAbility::Independent => 0.9,
        }
    }
}

impl StrokeModel {
    /// Create acute MCA stroke model
    pub fn acute_mca(hemisphere: StrokeHemisphere, nihss: u8) -> Self {
        let severity = StrokeSeverity::from_nihss(nihss);
        let mut deficits = vec![StrokeDeficit::Hemiparesis, StrokeDeficit::SensoryLoss];

        // Add aphasia for left hemisphere
        let aphasia = if hemisphere == StrokeHemisphere::Left && nihss > 5 {
            deficits.push(StrokeDeficit::Aphasia);
            Some(if nihss > 15 {
                AphasiaType::Global
            } else {
                AphasiaType::Broca
            })
        } else {
            None
        };

        // Add neglect for right hemisphere
        if hemisphere == StrokeHemisphere::Right && nihss > 5 {
            deficits.push(StrokeDeficit::Neglect);
        }

        Self {
            stroke_type: StrokeType::Ischemic,
            location: StrokeLocation::Mca,
            hemisphere,
            severity,
            nihss,
            days_since_onset: 1,
            recovery_phase: RecoveryPhase::Acute,
            ue_brunnstrom: BrunnstromStage::Stage1,
            le_brunnstrom: BrunnstromStage::Stage2,
            deficits,
            aphasia_type: aphasia,
            walking_ability: WalkingAbility::NonAmbulatory,
        }
    }

    /// Create subacute recovery model
    pub fn subacute_recovery(initial_nihss: u8) -> Self {
        let mut model = Self::acute_mca(StrokeHemisphere::Left, initial_nihss);
        model.days_since_onset = 30;
        model.recovery_phase = RecoveryPhase::EarlySubacute;
        model.nihss = (initial_nihss as f64 * 0.6) as u8;
        model.severity = StrokeSeverity::from_nihss(model.nihss);
        model.ue_brunnstrom = BrunnstromStage::Stage3;
        model.le_brunnstrom = BrunnstromStage::Stage4;
        model.walking_ability = WalkingAbility::ModerateAssist;
        model
    }

    /// Create chronic stroke model
    pub fn chronic(initial_nihss: u8) -> Self {
        let mut model = Self::acute_mca(StrokeHemisphere::Left, initial_nihss);
        let recovery = model.severity.recovery_potential();
        model.days_since_onset = 365;
        model.recovery_phase = RecoveryPhase::Chronic;
        model.nihss = ((initial_nihss as f64) * (1.0 - recovery)) as u8;
        model.severity = StrokeSeverity::from_nihss(model.nihss);
        model.ue_brunnstrom = BrunnstromStage::Stage4;
        model.le_brunnstrom = BrunnstromStage::Stage5;
        model.walking_ability = WalkingAbility::IndependentDevice;
        model
    }

    /// Generate multi-modal disease signature
    pub fn to_disease_signature(&self) -> DiseaseSignature {
        let weakness_severity = 1.0 - self.ue_brunnstrom.voluntary_control();
        let spasticity = self.ue_brunnstrom.spasticity();

        // EMG: spasticity, co-contraction, synergy patterns
        let emg_mod = SignalModulation {
            amplitude_factor: 0.5 + 0.5 * self.ue_brunnstrom.voluntary_control(),
            frequency_shift: spasticity * 15.0, // increased firing
            noise_level: 0.03 * self.ue_brunnstrom.synergy(),
            tremor: None,
            latency_increase: 0.05 * weakness_severity,
            variability_factor: 1.0 + weakness_severity * 0.5,
        };

        // Force: weakness pattern
        let force_mod = SignalModulation {
            amplitude_factor: self.ue_brunnstrom.voluntary_control(),
            variability_factor: 1.0 + weakness_severity * 0.8,
            latency_increase: 0.1 * weakness_severity,
            ..Default::default()
        };

        // Gait: hemiplegic pattern
        let gait_mod = SignalModulation {
            amplitude_factor: self.walking_ability.speed_factor(),
            latency_increase: 0.3 * weakness_severity,
            variability_factor: 1.0 + weakness_severity * 0.6,
            ..Default::default()
        };

        // HRV: autonomic changes post-stroke
        let hrv_mod = if self.days_since_onset < 90 {
            SignalModulation {
                variability_factor: 0.7, // reduced HRV acutely
                ..Default::default()
            }
        } else {
            SignalModulation::default()
        };

        // EEG: asymmetry, slowing on affected side
        let eeg_mod = SignalModulation {
            frequency_shift: -2.0 * (self.nihss as f64 / 42.0), // slowing
            variability_factor: 1.0 + 0.3 * (self.nihss as f64 / 42.0),
            ..Default::default()
        };

        // Voice: dysarthria
        let voice_mod =
            if self.deficits.contains(&StrokeDeficit::Dysarthria) || self.aphasia_type.is_some() {
                SignalModulation {
                    amplitude_factor: 0.7,
                    variability_factor: 1.5,
                    latency_increase: 0.1,
                    ..Default::default()
                }
            } else {
                SignalModulation::default()
            };

        // Eye: gaze preference, saccade abnormalities
        let eye_mod = SignalModulation {
            latency_increase: 0.05 * (self.nihss as f64 / 42.0),
            variability_factor: 1.0 + 0.3 * (self.nihss as f64 / 42.0),
            ..Default::default()
        };

        // Map deficits to body regions
        let affected_regions: Vec<BodyRegion> = self
            .deficits
            .iter()
            .filter_map(|d| match d {
                StrokeDeficit::Hemiparesis | StrokeDeficit::PureMotor => {
                    Some(BodyRegion::UpperLimb)
                }
                StrokeDeficit::LegWeakness => Some(BodyRegion::LowerLimb),
                StrokeDeficit::Aphasia | StrokeDeficit::Dysarthria | StrokeDeficit::Dysphagia => {
                    Some(BodyRegion::Bulbar)
                }
                StrokeDeficit::Neglect
                | StrokeDeficit::MemoryImpairment
                | StrokeDeficit::ExecutiveDysfunction => Some(BodyRegion::Cognitive),
                StrokeDeficit::VisualFieldDefect | StrokeDeficit::SensoryLoss => {
                    Some(BodyRegion::Sensory)
                }
                StrokeDeficit::Ataxia => Some(BodyRegion::Trunk),
                _ => None,
            })
            .collect();

        DiseaseSignature {
            disease: "stroke".to_string(),
            stage: self.severity.to_disease_stage(),
            affected_regions,
            laterality: self.hemisphere.affected_side(),
            emg_modulation: emg_mod,
            force_modulation: force_mod,
            gait_modulation: gait_mod,
            hrv_modulation: hrv_mod,
            eeg_modulation: eeg_mod,
            voice_modulation: voice_mod,
            eye_modulation: eye_mod,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stroke_severity() {
        assert_eq!(StrokeSeverity::from_nihss(3), StrokeSeverity::Minor);
        assert_eq!(StrokeSeverity::from_nihss(10), StrokeSeverity::Moderate);
        assert_eq!(StrokeSeverity::from_nihss(25), StrokeSeverity::Severe);
    }

    #[test]
    fn test_recovery_phase() {
        assert_eq!(RecoveryPhase::from_days(0), RecoveryPhase::Hyperacute);
        assert_eq!(RecoveryPhase::from_days(5), RecoveryPhase::Acute);
        assert_eq!(RecoveryPhase::from_days(60), RecoveryPhase::EarlySubacute);
        assert_eq!(RecoveryPhase::from_days(200), RecoveryPhase::Chronic);
    }

    #[test]
    fn test_brunnstrom_progression() {
        assert!(
            BrunnstromStage::Stage1.voluntary_control()
                < BrunnstromStage::Stage6.voluntary_control()
        );
        assert!(BrunnstromStage::Stage3.spasticity() > BrunnstromStage::Stage6.spasticity());
    }

    #[test]
    fn test_acute_stroke_model() {
        let stroke = StrokeModel::acute_mca(StrokeHemisphere::Left, 12);

        assert_eq!(stroke.recovery_phase, RecoveryPhase::Acute);
        assert!(stroke.deficits.contains(&StrokeDeficit::Hemiparesis));
        assert!(stroke.aphasia_type.is_some());
    }

    #[test]
    fn test_chronic_stroke() {
        let stroke = StrokeModel::chronic(15);

        assert_eq!(stroke.recovery_phase, RecoveryPhase::Chronic);
        assert!(stroke.nihss < 15);
        assert!(stroke.walking_ability.speed_factor() > 0.5);
    }

    #[test]
    fn test_disease_signature() {
        let model = StrokeModel::subacute_recovery(12);
        let sig = model.to_disease_signature();

        assert_eq!(sig.disease, "stroke");
        assert!(!sig.affected_regions.is_empty());
    }
}
