//! Medication effect generators for biosignal modulation.
//!
//! This module models how various medications affect biosignals:
//! - Dopaminergic drugs (Parkinson's, restless leg)
//! - Beta-blockers (heart rate, tremor)
//! - Anticholinergics (HRV, cognitive)
//! - Sedatives/anxiolytics (EEG, reaction time)
//! - Stimulants (alertness, heart rate)
//! - Muscle relaxants (EMG, reflexes)
//! - Anticonvulsants (EEG, cognitive)

use super::SignalModulation;
use serde::{Deserialize, Serialize};

// Domain acronyms -- ECG beat annotations, audio codecs, ERP components,
// the SMPL-X body model, drug classes. Camel case would diverge from how
// these are written everywhere they are used.
#[allow(clippy::upper_case_acronyms)]
/// Medication classes with biosignal effects
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MedicationClass {
    /// Dopamine agonists/precursors (Levodopa, pramipexole)
    Dopaminergic,
    /// Beta-adrenergic blockers (propranolol, metoprolol)
    BetaBlocker,
    /// Anticholinergics (benztropine, trihexyphenidyl)
    Anticholinergic,
    /// Benzodiazepines (diazepam, lorazepam)
    Benzodiazepine,
    /// Stimulants (methylphenidate, amphetamines)
    Stimulant,
    /// Muscle relaxants (baclofen, tizanidine)
    MuscleRelaxant,
    /// Anticonvulsants (levetiracetam, carbamazepine)
    Anticonvulsant,
    /// Opioids (morphine, oxycodone)
    Opioid,
    /// SSRIs (fluoxetine, sertraline)
    SSRI,
    /// Antipsychotics (haloperidol, risperidone)
    Antipsychotic,
    /// Alpha-2 agonists (clonidine, guanfacine)
    Alpha2Agonist,
    /// Cholinesterase inhibitors (donepezil, rivastigmine)
    CholinesteraseInhibitor,
}

/// Individual medication model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Medication {
    /// Drug name
    pub name: String,
    /// Medication class
    pub class: MedicationClass,
    /// Dose (mg)
    pub dose_mg: f64,
    /// Maximum daily dose (mg)
    pub max_daily_dose_mg: f64,
    /// Time to peak effect (hours)
    pub tmax_hours: f64,
    /// Half-life (hours)
    pub half_life_hours: f64,
    /// Current plasma level (normalized 0-1)
    pub current_level: f64,
}

impl Medication {
    /// Create levodopa/carbidopa
    pub fn levodopa(dose_mg: f64) -> Self {
        Self {
            name: "levodopa".to_string(),
            class: MedicationClass::Dopaminergic,
            dose_mg,
            max_daily_dose_mg: 1500.0,
            tmax_hours: 1.0,
            half_life_hours: 1.5,
            current_level: 0.8,
        }
    }

    /// Create propranolol
    pub fn propranolol(dose_mg: f64) -> Self {
        Self {
            name: "propranolol".to_string(),
            class: MedicationClass::BetaBlocker,
            dose_mg,
            max_daily_dose_mg: 320.0,
            tmax_hours: 1.5,
            half_life_hours: 4.0,
            current_level: 0.8,
        }
    }

    /// Create lorazepam
    pub fn lorazepam(dose_mg: f64) -> Self {
        Self {
            name: "lorazepam".to_string(),
            class: MedicationClass::Benzodiazepine,
            dose_mg,
            max_daily_dose_mg: 6.0,
            tmax_hours: 2.0,
            half_life_hours: 12.0,
            current_level: 0.8,
        }
    }

    /// Create methylphenidate
    pub fn methylphenidate(dose_mg: f64) -> Self {
        Self {
            name: "methylphenidate".to_string(),
            class: MedicationClass::Stimulant,
            dose_mg,
            max_daily_dose_mg: 60.0,
            tmax_hours: 2.0,
            half_life_hours: 3.0,
            current_level: 0.8,
        }
    }

    /// Create baclofen
    pub fn baclofen(dose_mg: f64) -> Self {
        Self {
            name: "baclofen".to_string(),
            class: MedicationClass::MuscleRelaxant,
            dose_mg,
            max_daily_dose_mg: 80.0,
            tmax_hours: 1.0,
            half_life_hours: 4.0,
            current_level: 0.8,
        }
    }

    /// Create levetiracetam
    pub fn levetiracetam(dose_mg: f64) -> Self {
        Self {
            name: "levetiracetam".to_string(),
            class: MedicationClass::Anticonvulsant,
            dose_mg,
            max_daily_dose_mg: 3000.0,
            tmax_hours: 1.0,
            half_life_hours: 7.0,
            current_level: 0.8,
        }
    }

    /// Create donepezil
    pub fn donepezil(dose_mg: f64) -> Self {
        Self {
            name: "donepezil".to_string(),
            class: MedicationClass::CholinesteraseInhibitor,
            dose_mg,
            max_daily_dose_mg: 23.0,
            tmax_hours: 4.0,
            half_life_hours: 70.0,
            current_level: 0.8,
        }
    }

    /// Calculate drug level at time since dose
    pub fn level_at(&self, hours_since_dose: f64) -> f64 {
        if hours_since_dose < 0.0 {
            return 0.0;
        }

        // Simple one-compartment model
        // Absorption phase
        let ka = 2.0 / self.tmax_hours; // absorption rate constant
        let ke = 0.693 / self.half_life_hours; // elimination rate constant

        if ka == ke {
            // Degenerate case
            return self.current_level * hours_since_dose * (-ke * hours_since_dose).exp();
        }

        // Bateman equation
        let level = (ka / (ka - ke)) *
            ((-ke * hours_since_dose).exp() - (-ka * hours_since_dose).exp());

        (level * self.current_level).clamp(0.0, 1.0)
    }

    /// Get relative dose (fraction of max)
    pub fn relative_dose(&self) -> f64 {
        (self.dose_mg / self.max_daily_dose_mg).min(1.0)
    }
}

/// Medication effect profile on biosignals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicationEffect {
    /// Medication producing the effect
    pub medication: Medication,
    /// Effect on heart rate (% change)
    pub heart_rate_change: f64,
    /// Effect on HRV (multiplier)
    pub hrv_change: f64,
    /// Effect on tremor amplitude (multiplier)
    pub tremor_change: f64,
    /// Effect on muscle tone (multiplier)
    pub muscle_tone_change: f64,
    /// Effect on reaction time (multiplier)
    pub reaction_time_change: f64,
    /// Effect on alertness (% change)
    pub alertness_change: f64,
    /// Effect on EEG alpha power (multiplier)
    pub eeg_alpha_change: f64,
    /// Effect on EEG beta power (multiplier)
    pub eeg_beta_change: f64,
    /// Effect on pupil size (multiplier)
    pub pupil_change: f64,
}

impl MedicationEffect {
    /// Generate effects from medication at current level
    pub fn from_medication(medication: Medication) -> Self {
        let level = medication.current_level;
        let dose_factor = medication.relative_dose();

        let (hr, hrv, tremor, tone, rt, alert, alpha, beta, pupil) =
            match medication.class {
                MedicationClass::Dopaminergic => (
                    0.0,           // minimal HR effect
                    1.0,           // HRV unchanged
                    0.6,           // reduce tremor 40%
                    0.9,           // slight tone reduction
                    0.85,          // improve RT 15%
                    1.1,           // slight alertness increase
                    1.0,           // alpha unchanged
                    1.1,           // slight beta increase
                    1.0,           // pupil unchanged
                ),
                MedicationClass::BetaBlocker => (
                    -15.0 * dose_factor, // reduce HR
                    1.2,                  // increase HRV
                    0.5,                  // reduce tremor 50%
                    1.0,                  // tone unchanged
                    1.1,                  // slight RT slowing
                    0.95,                 // slight sedation
                    1.0,                  // alpha unchanged
                    0.9,                  // reduce beta
                    1.0,                  // pupil unchanged
                ),
                MedicationClass::Anticholinergic => (
                    10.0 * dose_factor, // increase HR
                    0.7,                 // reduce HRV
                    0.8,                 // slight tremor reduction
                    0.85,                // reduce tone
                    1.2,                 // slow RT (cognitive)
                    0.9,                 // reduce alertness
                    0.9,                 // reduce alpha
                    0.95,                // slight beta reduction
                    1.3,                 // mydriasis
                ),
                MedicationClass::Benzodiazepine => (
                    -5.0 * dose_factor, // slight HR reduction
                    1.1,                 // slight HRV increase
                    0.7,                 // reduce tremor
                    0.6,                 // reduce tone
                    1.4,                 // slow RT significantly
                    0.6,                 // reduce alertness
                    0.7,                 // reduce alpha
                    1.3,                 // increase beta (fast activity)
                    1.0,                 // pupil unchanged
                ),
                MedicationClass::Stimulant => (
                    15.0 * dose_factor, // increase HR
                    0.8,                 // reduce HRV
                    1.2,                 // may increase tremor
                    1.1,                 // increase tone
                    0.7,                 // improve RT 30%
                    1.5,                 // increase alertness
                    0.8,                 // reduce alpha
                    1.4,                 // increase beta
                    1.2,                 // mydriasis
                ),
                MedicationClass::MuscleRelaxant => (
                    0.0,            // HR unchanged
                    1.0,            // HRV unchanged
                    0.6,            // reduce tremor
                    0.5,            // significant tone reduction
                    1.1,            // slight RT slowing
                    0.9,            // slight sedation
                    1.0,            // alpha unchanged
                    1.0,            // beta unchanged
                    1.0,            // pupil unchanged
                ),
                MedicationClass::Anticonvulsant => (
                    0.0,            // HR unchanged
                    1.0,            // HRV unchanged
                    0.8,            // slight tremor reduction
                    1.0,            // tone unchanged
                    1.15,           // slight RT slowing
                    0.9,            // slight sedation
                    1.0,            // alpha unchanged
                    0.85,           // reduce beta (stabilize)
                    1.0,            // pupil unchanged
                ),
                MedicationClass::Opioid => (
                    -10.0 * dose_factor, // reduce HR
                    0.8,                  // reduce HRV
                    1.0,                  // tremor unchanged
                    0.8,                  // reduce tone
                    1.5,                  // slow RT significantly
                    0.5,                  // significant sedation
                    0.8,                  // reduce alpha
                    0.9,                  // reduce beta
                    0.7,                  // miosis
                ),
                MedicationClass::SSRI => (
                    5.0 * dose_factor, // slight HR increase
                    0.9,                // slight HRV reduction
                    1.1,                // may increase tremor
                    1.0,                // tone unchanged
                    1.0,                // RT unchanged
                    1.0,                // alertness unchanged
                    1.0,                // alpha unchanged
                    1.0,                // beta unchanged
                    1.1,                // slight mydriasis
                ),
                MedicationClass::Antipsychotic => (
                    0.0,            // HR unchanged
                    0.9,            // slight HRV reduction
                    0.7,            // reduce tremor (but may cause EPS)
                    1.2,            // may increase tone (EPS)
                    1.2,            // slow RT
                    0.7,            // sedation
                    0.9,            // reduce alpha
                    0.95,           // slight beta reduction
                    1.0,            // pupil unchanged
                ),
                MedicationClass::Alpha2Agonist => (
                    -10.0 * dose_factor, // reduce HR
                    1.1,                  // increase HRV
                    0.8,                  // reduce tremor
                    0.9,                  // reduce tone
                    1.1,                  // slight RT slowing
                    0.8,                  // sedation
                    1.0,                  // alpha unchanged
                    0.9,                  // reduce beta
                    1.0,                  // pupil unchanged
                ),
                MedicationClass::CholinesteraseInhibitor => (
                    -5.0 * dose_factor, // slight HR reduction
                    1.1,                 // increase HRV
                    1.1,                 // may increase tremor
                    1.0,                 // tone unchanged
                    0.9,                 // improve RT
                    1.1,                 // improve alertness
                    1.1,                 // increase alpha
                    1.0,                 // beta unchanged
                    0.9,                 // slight miosis
                ),
            };

        Self {
            medication,
            heart_rate_change: hr * level,
            hrv_change: 1.0 + (hrv - 1.0) * level,
            tremor_change: 1.0 + (tremor - 1.0) * level,
            muscle_tone_change: 1.0 + (tone - 1.0) * level,
            reaction_time_change: 1.0 + (rt - 1.0) * level,
            alertness_change: 1.0 + (alert - 1.0) * level,
            eeg_alpha_change: 1.0 + (alpha - 1.0) * level,
            eeg_beta_change: 1.0 + (beta - 1.0) * level,
            pupil_change: 1.0 + (pupil - 1.0) * level,
        }
    }

    /// Convert to signal modulation for EMG
    pub fn to_emg_modulation(&self) -> SignalModulation {
        SignalModulation {
            amplitude_factor: self.muscle_tone_change,
            tremor: if self.tremor_change != 1.0 {
                Some((5.0, 0.1 * self.tremor_change))
            } else {
                None
            },
            variability_factor: 1.0 / self.alertness_change.max(0.5),
            ..Default::default()
        }
    }

    /// Convert to signal modulation for HRV
    pub fn to_hrv_modulation(&self) -> SignalModulation {
        SignalModulation {
            amplitude_factor: 1.0 + self.heart_rate_change / 100.0,
            variability_factor: self.hrv_change,
            ..Default::default()
        }
    }

    /// Convert to signal modulation for EEG
    pub fn to_eeg_modulation(&self) -> SignalModulation {
        SignalModulation {
            amplitude_factor: (self.eeg_alpha_change + self.eeg_beta_change) / 2.0,
            frequency_shift: if self.alertness_change < 0.8 { -1.0 } else { 0.0 },
            latency_increase: (self.reaction_time_change - 1.0) * 0.05,
            ..Default::default()
        }
    }

    /// Convert to signal modulation for eye tracking
    pub fn to_eye_modulation(&self) -> SignalModulation {
        SignalModulation {
            amplitude_factor: self.pupil_change,
            latency_increase: (self.reaction_time_change - 1.0) * 0.1,
            variability_factor: 1.0 / self.alertness_change.max(0.5),
            ..Default::default()
        }
    }
}

/// Combined effects from multiple medications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolypharmacyProfile {
    /// Active medications
    pub medications: Vec<MedicationEffect>,
    /// Net effect on heart rate (% change)
    pub net_heart_rate_change: f64,
    /// Net effect on HRV
    pub net_hrv_change: f64,
    /// Net effect on tremor
    pub net_tremor_change: f64,
    /// Net effect on reaction time
    pub net_reaction_time_change: f64,
    /// Potential interactions
    pub interactions: Vec<DrugInteraction>,
}

/// Drug-drug interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugInteraction {
    /// First drug
    pub drug1: String,
    /// Second drug
    pub drug2: String,
    /// Interaction type
    pub interaction_type: InteractionType,
    /// Severity
    pub severity: InteractionSeverity,
    /// Clinical effect description
    pub effect: String,
}

/// Type of drug interaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionType {
    /// Additive effects
    Additive,
    /// Synergistic (more than additive)
    Synergistic,
    /// Antagonistic (opposing effects)
    Antagonistic,
    /// Pharmacokinetic (metabolism)
    Pharmacokinetic,
}

/// Severity of interaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionSeverity {
    Minor,
    Moderate,
    Major,
    Contraindicated,
}

impl PolypharmacyProfile {
    /// Create profile from list of medications
    pub fn from_medications(medications: Vec<Medication>) -> Self {
        let effects: Vec<MedicationEffect> = medications
            .into_iter()
            .map(MedicationEffect::from_medication)
            .collect();

        // Calculate net effects (simplified additive model)
        let net_heart_rate_change = effects
            .iter()
            .map(|e| e.heart_rate_change)
            .sum::<f64>();

        let net_hrv_change = effects
            .iter()
            .map(|e| e.hrv_change)
            .product::<f64>()
            .powf(1.0 / effects.len().max(1) as f64);

        let net_tremor_change = effects
            .iter()
            .map(|e| e.tremor_change)
            .product::<f64>()
            .powf(1.0 / effects.len().max(1) as f64);

        let net_reaction_time_change = effects
            .iter()
            .map(|e| e.reaction_time_change)
            .product::<f64>()
            .powf(1.0 / effects.len().max(1) as f64);

        // Check for interactions
        let interactions = Self::detect_interactions(&effects);

        Self {
            medications: effects,
            net_heart_rate_change,
            net_hrv_change,
            net_tremor_change,
            net_reaction_time_change,
            interactions,
        }
    }

    /// Detect potential drug interactions
    fn detect_interactions(effects: &[MedicationEffect]) -> Vec<DrugInteraction> {
        let mut interactions = Vec::new();

        for i in 0..effects.len() {
            for j in (i + 1)..effects.len() {
                let class1 = effects[i].medication.class;
                let class2 = effects[j].medication.class;
                let name1 = &effects[i].medication.name;
                let name2 = &effects[j].medication.name;

                // Check for known interaction patterns
                if class1 == MedicationClass::Benzodiazepine
                    && class2 == MedicationClass::Opioid
                {
                    interactions.push(DrugInteraction {
                        drug1: name1.clone(),
                        drug2: name2.clone(),
                        interaction_type: InteractionType::Synergistic,
                        severity: InteractionSeverity::Major,
                        effect: "Additive CNS depression, respiratory risk".to_string(),
                    });
                }

                if class1 == MedicationClass::BetaBlocker
                    && class2 == MedicationClass::Alpha2Agonist
                {
                    interactions.push(DrugInteraction {
                        drug1: name1.clone(),
                        drug2: name2.clone(),
                        interaction_type: InteractionType::Additive,
                        severity: InteractionSeverity::Moderate,
                        effect: "Additive bradycardia risk".to_string(),
                    });
                }

                if class1 == MedicationClass::Anticholinergic
                    && class2 == MedicationClass::CholinesteraseInhibitor
                {
                    interactions.push(DrugInteraction {
                        drug1: name1.clone(),
                        drug2: name2.clone(),
                        interaction_type: InteractionType::Antagonistic,
                        severity: InteractionSeverity::Moderate,
                        effect: "Opposing effects on cholinergic system".to_string(),
                    });
                }
            }
        }

        interactions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medication_creation() {
        let med = Medication::levodopa(100.0);
        assert_eq!(med.class, MedicationClass::Dopaminergic);
        assert!(med.relative_dose() < 0.1);
    }

    #[test]
    fn test_drug_level() {
        let med = Medication::propranolol(40.0);

        // Level should peak around tmax
        let level_early = med.level_at(0.5);
        let level_peak = med.level_at(med.tmax_hours);
        let level_late = med.level_at(10.0);

        assert!(level_peak > level_early);
        assert!(level_peak > level_late);
    }

    #[test]
    fn test_medication_effect() {
        let med = Medication::propranolol(40.0);
        let effect = MedicationEffect::from_medication(med);

        assert!(effect.heart_rate_change < 0.0); // Beta blocker reduces HR
        assert!(effect.tremor_change < 1.0); // Reduces tremor
    }

    #[test]
    fn test_polypharmacy() {
        let meds = vec![
            Medication::propranolol(40.0),
            Medication::levetiracetam(500.0),
        ];

        let profile = PolypharmacyProfile::from_medications(meds);

        assert_eq!(profile.medications.len(), 2);
    }

    #[test]
    fn test_interaction_detection() {
        let meds = vec![
            Medication::lorazepam(1.0),
            Medication {
                name: "morphine".to_string(),
                class: MedicationClass::Opioid,
                dose_mg: 10.0,
                max_daily_dose_mg: 60.0,
                tmax_hours: 1.0,
                half_life_hours: 4.0,
                current_level: 0.8,
            },
        ];

        let profile = PolypharmacyProfile::from_medications(meds);

        assert!(!profile.interactions.is_empty());
        assert!(profile.interactions[0].severity == InteractionSeverity::Major);
    }

    #[test]
    fn test_signal_modulation() {
        let med = Medication::methylphenidate(20.0);
        let effect = MedicationEffect::from_medication(med);

        let hrv_mod = effect.to_hrv_modulation();
        assert!(hrv_mod.amplitude_factor > 1.0); // HR increase
    }
}
