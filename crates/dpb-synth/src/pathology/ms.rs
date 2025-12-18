//! Multiple Sclerosis (MS) disease model.
//!
//! MS is an autoimmune demyelinating disease characterized by:
//! - Relapsing-remitting or progressive course
//! - Dissemination in time and space
//! - Optic neuritis, sensory symptoms, motor weakness
//! - Fatigue, cognitive impairment
//! - Cerebellar dysfunction (ataxia, tremor)
//!
//! This module models MS-specific biosignal patterns including
//! visual evoked potentials, motor variability, and fatigue patterns.

use super::{BodyRegion, DiseaseSignature, DiseaseStage, Laterality, SignalModulation};
use serde::{Deserialize, Serialize};

/// MS disease types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MsType {
    /// Relapsing-Remitting MS (most common, ~85%)
    RelapsingRemitting,
    /// Secondary Progressive MS
    SecondaryProgressive,
    /// Primary Progressive MS (~10-15%)
    PrimaryProgressive,
    /// Progressive-Relapsing MS (rare)
    ProgressiveRelapsing,
    /// Clinically Isolated Syndrome
    Cis,
    /// Radiologically Isolated Syndrome
    Ris,
}

impl MsType {
    /// Whether this type has relapses
    pub fn has_relapses(&self) -> bool {
        matches!(
            self,
            MsType::RelapsingRemitting | MsType::ProgressiveRelapsing
        )
    }

    /// Whether this type has progression
    pub fn has_progression(&self) -> bool {
        matches!(
            self,
            MsType::SecondaryProgressive | MsType::PrimaryProgressive | MsType::ProgressiveRelapsing
        )
    }
}

/// Expanded Disability Status Scale (EDSS)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdssScore(f64);

impl EdssScore {
    /// Create new EDSS score (0.0 - 10.0)
    pub fn new(score: f64) -> Option<Self> {
        if (0.0..=10.0).contains(&score) && (score * 2.0).fract() == 0.0 {
            Some(EdssScore(score))
        } else {
            None
        }
    }

    /// Get the score value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Get disability level description
    pub fn description(&self) -> &'static str {
        match self.0 as u8 {
            0 => "Normal neurological exam",
            1 => "No disability, minimal signs in one FS",
            2 => "Minimal disability in one FS",
            3 => "Moderate disability or mild in 3-4 FS, fully ambulatory",
            4 => "Relatively severe disability, fully ambulatory",
            5 => "Disability impairs full daily activities, ambulatory ~500m",
            6 => "Requires unilateral/bilateral assistance to walk ~100m",
            7 => "Unable to walk beyond ~20m, essentially wheelchair-bound",
            8 => "Restricted to bed/chair, retains many self-care functions",
            9 => "Helpless bed patient, can communicate and eat",
            10 => "Death due to MS",
            _ => "Unknown",
        }
    }

    /// Convert to disease stage
    pub fn to_disease_stage(&self) -> DiseaseStage {
        match self.0 {
            x if x < 2.0 => DiseaseStage::Prodromal,
            x if x < 4.0 => DiseaseStage::Early,
            x if x < 6.0 => DiseaseStage::Moderate,
            x if x < 7.5 => DiseaseStage::Advanced,
            _ => DiseaseStage::EndStage,
        }
    }
}

impl Default for EdssScore {
    fn default() -> Self {
        EdssScore(0.0)
    }
}

/// MS relapse event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsRelapse {
    /// Day of onset (from disease start)
    pub onset_day: u32,
    /// Duration in days
    pub duration_days: u32,
    /// Affected functional systems
    pub affected_systems: Vec<MsFunctionalSystem>,
    /// Severity (0-1)
    pub severity: f64,
    /// Whether treated with steroids
    pub steroid_treated: bool,
    /// Residual deficit after recovery (0-1)
    pub residual_deficit: f64,
}

/// MS functional systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MsFunctionalSystem {
    /// Visual (optic neuritis)
    Visual,
    /// Brainstem
    Brainstem,
    /// Pyramidal (motor)
    Pyramidal,
    /// Cerebellar
    Cerebellar,
    /// Sensory
    Sensory,
    /// Bowel/Bladder
    BowelBladder,
    /// Cerebral (cognitive)
    Cerebral,
}

impl MsFunctionalSystem {
    /// Map to body region
    pub fn to_body_region(&self) -> BodyRegion {
        match self {
            MsFunctionalSystem::Visual => BodyRegion::Sensory,
            MsFunctionalSystem::Brainstem => BodyRegion::Bulbar,
            MsFunctionalSystem::Pyramidal => BodyRegion::UpperLimb, // Simplified
            MsFunctionalSystem::Cerebellar => BodyRegion::Trunk,
            MsFunctionalSystem::Sensory => BodyRegion::Sensory,
            MsFunctionalSystem::BowelBladder => BodyRegion::Trunk,
            MsFunctionalSystem::Cerebral => BodyRegion::Cognitive,
        }
    }
}

/// MS symptom profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsSymptomProfile {
    /// Fatigue level (0-10, FSS equivalent)
    pub fatigue_score: f64,
    /// Cognitive impairment (0-1)
    pub cognitive_impairment: f64,
    /// Spasticity severity (Ashworth scale 0-4)
    pub spasticity: f64,
    /// Ataxia severity (0-1)
    pub ataxia: f64,
    /// Sensory deficit severity (0-1)
    pub sensory_deficit: f64,
    /// Visual acuity deficit (logMAR)
    pub visual_deficit: f64,
    /// VEP latency delay (ms)
    pub vep_delay_ms: f64,
    /// Walking speed reduction (fraction)
    pub walking_speed_reduction: f64,
    /// Heat sensitivity present
    pub heat_sensitive: bool,
}

impl MsSymptomProfile {
    /// Create profile for early RRMS
    pub fn early_rrms() -> Self {
        Self {
            fatigue_score: 4.0,
            cognitive_impairment: 0.1,
            spasticity: 0.5,
            ataxia: 0.1,
            sensory_deficit: 0.2,
            visual_deficit: 0.0,
            vep_delay_ms: 10.0,
            walking_speed_reduction: 0.1,
            heat_sensitive: true,
        }
    }

    /// Create profile for moderate MS
    pub fn moderate() -> Self {
        Self {
            fatigue_score: 6.5,
            cognitive_impairment: 0.3,
            spasticity: 2.0,
            ataxia: 0.3,
            sensory_deficit: 0.4,
            visual_deficit: 0.2,
            vep_delay_ms: 25.0,
            walking_speed_reduction: 0.3,
            heat_sensitive: true,
        }
    }

    /// Create profile for advanced/progressive MS
    pub fn advanced() -> Self {
        Self {
            fatigue_score: 8.0,
            cognitive_impairment: 0.5,
            spasticity: 3.0,
            ataxia: 0.5,
            sensory_deficit: 0.6,
            visual_deficit: 0.4,
            vep_delay_ms: 40.0,
            walking_speed_reduction: 0.6,
            heat_sensitive: true,
        }
    }
}

/// Comprehensive MS disease model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsModel {
    /// MS type
    pub ms_type: MsType,
    /// EDSS score
    pub edss: EdssScore,
    /// Disease duration (years)
    pub duration_years: f64,
    /// Number of relapses in past 2 years
    pub recent_relapse_count: u32,
    /// Current relapse status
    pub in_relapse: bool,
    /// Symptom profile
    pub symptoms: MsSymptomProfile,
    /// Affected functional systems
    pub affected_systems: Vec<MsFunctionalSystem>,
    /// On disease-modifying therapy
    pub on_dmt: bool,
    /// Laterality of motor symptoms
    pub laterality: Laterality,
}

impl MsModel {
    /// Create early RRMS model
    pub fn early_rrms() -> Self {
        Self {
            ms_type: MsType::RelapsingRemitting,
            edss: EdssScore::new(2.0).unwrap(),
            duration_years: 2.0,
            recent_relapse_count: 2,
            in_relapse: false,
            symptoms: MsSymptomProfile::early_rrms(),
            affected_systems: vec![MsFunctionalSystem::Sensory, MsFunctionalSystem::Pyramidal],
            on_dmt: true,
            laterality: Laterality::Bilateral,
        }
    }

    /// Create moderate RRMS model
    pub fn moderate_rrms() -> Self {
        Self {
            ms_type: MsType::RelapsingRemitting,
            edss: EdssScore::new(4.0).unwrap(),
            duration_years: 8.0,
            recent_relapse_count: 1,
            in_relapse: false,
            symptoms: MsSymptomProfile::moderate(),
            affected_systems: vec![
                MsFunctionalSystem::Pyramidal,
                MsFunctionalSystem::Cerebellar,
                MsFunctionalSystem::Sensory,
                MsFunctionalSystem::Cerebral,
            ],
            on_dmt: true,
            laterality: Laterality::Predominant(true),
        }
    }

    /// Create SPMS model
    pub fn secondary_progressive() -> Self {
        Self {
            ms_type: MsType::SecondaryProgressive,
            edss: EdssScore::new(6.0).unwrap(),
            duration_years: 15.0,
            recent_relapse_count: 0,
            in_relapse: false,
            symptoms: MsSymptomProfile::advanced(),
            affected_systems: vec![
                MsFunctionalSystem::Pyramidal,
                MsFunctionalSystem::Cerebellar,
                MsFunctionalSystem::Sensory,
                MsFunctionalSystem::Cerebral,
                MsFunctionalSystem::BowelBladder,
            ],
            on_dmt: false,
            laterality: Laterality::Bilateral,
        }
    }

    /// Create relapse event
    pub fn generate_relapse(&self) -> MsRelapse {
        MsRelapse {
            onset_day: 0, // Current
            duration_days: 28,
            affected_systems: self.affected_systems.clone(),
            severity: 0.6,
            steroid_treated: true,
            residual_deficit: 0.1,
        }
    }

    /// Generate multi-modal disease signature
    pub fn to_disease_signature(&self) -> DiseaseSignature {
        let edss_severity = self.edss.value() / 10.0;

        // EMG: spasticity patterns
        let emg_mod = SignalModulation {
            amplitude_factor: 1.0 + self.symptoms.spasticity * 0.1, // increased tone
            frequency_shift: self.symptoms.spasticity * 5.0,
            noise_level: self.symptoms.ataxia * 0.05,
            tremor: if self.symptoms.ataxia > 0.3 {
                Some((4.0, self.symptoms.ataxia * 0.1)) // intention tremor
            } else {
                None
            },
            latency_increase: 0.02 * edss_severity,
            variability_factor: 1.0 + self.symptoms.ataxia,
        };

        // Force: weakness + spasticity
        let force_mod = SignalModulation {
            amplitude_factor: 1.0 - edss_severity * 0.4,
            variability_factor: 1.0 + self.symptoms.ataxia * 0.5,
            ..Default::default()
        };

        // Gait: characteristic MS gait
        let gait_mod = SignalModulation {
            amplitude_factor: 1.0 - self.symptoms.walking_speed_reduction,
            latency_increase: edss_severity * 0.3,
            variability_factor: 1.0 + self.symptoms.ataxia * 0.8,
            tremor: if self.symptoms.ataxia > 0.2 {
                Some((3.0, 0.02))
            } else {
                None
            },
            ..Default::default()
        };

        // HRV: autonomic dysfunction
        let hrv_mod = if self.edss.value() > 4.0 {
            SignalModulation {
                variability_factor: 0.8,
                ..Default::default()
            }
        } else {
            SignalModulation::default()
        };

        // EEG: cognitive involvement
        let eeg_mod = SignalModulation {
            latency_increase: self.symptoms.cognitive_impairment * 0.1, // P300 delay
            variability_factor: 1.0 + self.symptoms.cognitive_impairment * 0.3,
            frequency_shift: -self.symptoms.cognitive_impairment * 2.0, // slowing
            ..Default::default()
        };

        // Voice: mild dysarthria in some cases
        let voice_mod = if self.affected_systems.contains(&MsFunctionalSystem::Brainstem) {
            SignalModulation {
                variability_factor: 1.0 + edss_severity * 0.3,
                frequency_shift: -edss_severity * 10.0,
                ..Default::default()
            }
        } else {
            SignalModulation::default()
        };

        // Eye: VEP delays, saccade abnormalities
        let eye_mod = SignalModulation {
            latency_increase: self.symptoms.vep_delay_ms / 1000.0,
            variability_factor: 1.0 + self.symptoms.ataxia * 0.4,
            ..Default::default()
        };

        DiseaseSignature {
            disease: "ms".to_string(),
            stage: self.edss.to_disease_stage(),
            affected_regions: self.affected_systems.iter().map(|s| s.to_body_region()).collect(),
            laterality: self.laterality,
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
    fn test_edss_creation() {
        assert!(EdssScore::new(4.5).is_some());
        assert!(EdssScore::new(4.3).is_none()); // Invalid increment
        assert!(EdssScore::new(11.0).is_none()); // Out of range
    }

    #[test]
    fn test_edss_stages() {
        assert_eq!(EdssScore::new(1.0).unwrap().to_disease_stage(), DiseaseStage::Prodromal);
        assert_eq!(EdssScore::new(5.0).unwrap().to_disease_stage(), DiseaseStage::Moderate);
        assert_eq!(EdssScore::new(8.0).unwrap().to_disease_stage(), DiseaseStage::EndStage);
    }

    #[test]
    fn test_ms_type() {
        assert!(MsType::RelapsingRemitting.has_relapses());
        assert!(!MsType::PrimaryProgressive.has_relapses());
        assert!(MsType::SecondaryProgressive.has_progression());
    }

    #[test]
    fn test_ms_model() {
        let early = MsModel::early_rrms();
        assert_eq!(early.ms_type, MsType::RelapsingRemitting);
        assert!(early.edss.value() < 3.0);

        let spms = MsModel::secondary_progressive();
        assert_eq!(spms.ms_type, MsType::SecondaryProgressive);
        assert!(spms.edss.value() > 5.0);
    }

    #[test]
    fn test_disease_signature() {
        let model = MsModel::moderate_rrms();
        let sig = model.to_disease_signature();

        assert_eq!(sig.disease, "ms");
        assert!(sig.gait_modulation.variability_factor > 1.0);
    }
}
