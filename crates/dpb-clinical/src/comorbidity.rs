//! Comorbidity modeling for multi-disease simulation.
//!
//! This module provides tools for modeling interactions between multiple
//! conditions and their effects on biosignal patterns.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Comorbidity model for multi-condition interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComorbidityModel {
    /// Model name.
    pub name: String,
    /// Conditions in the model.
    conditions: HashMap<String, Condition>,
    /// Pairwise interactions.
    interactions: HashMap<(String, String), Interaction>,
    /// Higher-order interactions.
    complex_interactions: Vec<ComplexInteraction>,
}

impl ComorbidityModel {
    /// Create a new comorbidity model.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            conditions: HashMap::new(),
            interactions: HashMap::new(),
            complex_interactions: Vec::new(),
        }
    }

    /// Add a condition to the model.
    pub fn add_condition(&mut self, condition: Condition) {
        self.conditions.insert(condition.code.clone(), condition);
    }

    /// Get a condition by code.
    pub fn get_condition(&self, code: &str) -> Option<&Condition> {
        self.conditions.get(code)
    }

    /// Add a pairwise interaction.
    pub fn add_interaction(&mut self, cond1: &str, cond2: &str, interaction: Interaction) {
        let key = Self::interaction_key(cond1, cond2);
        self.interactions.insert(key, interaction);
    }

    /// Get pairwise interaction.
    pub fn get_interaction(&self, cond1: &str, cond2: &str) -> Option<&Interaction> {
        let key = Self::interaction_key(cond1, cond2);
        self.interactions.get(&key)
    }

    /// Add a complex (higher-order) interaction.
    pub fn add_complex_interaction(&mut self, interaction: ComplexInteraction) {
        self.complex_interactions.push(interaction);
    }

    /// Create canonical key for condition pair.
    fn interaction_key(cond1: &str, cond2: &str) -> (String, String) {
        if cond1 < cond2 {
            (cond1.to_string(), cond2.to_string())
        } else {
            (cond2.to_string(), cond1.to_string())
        }
    }

    /// Calculate combined effect of multiple conditions on a measure.
    pub fn combined_effect(
        &self,
        conditions: &[&str],
        measure: &str,
    ) -> Result<CombinedEffect> {
        if conditions.is_empty() {
            return Ok(CombinedEffect::none());
        }

        // Collect individual effects
        let mut individual_effects = Vec::new();
        for code in conditions {
            if let Some(cond) = self.conditions.get(*code)
                && let Some(effect) = cond.effects.get(measure) {
                    individual_effects.push(*effect);
                }
        }

        // Start with sum of individual effects (additive model)
        let additive_effect: f64 = individual_effects.iter().sum();

        // Add pairwise interactions
        let mut interaction_effect = 0.0;
        for i in 0..conditions.len() {
            for j in i + 1..conditions.len() {
                if let Some(interaction) = self.get_interaction(conditions[i], conditions[j])
                    && let Some(effect_mod) = interaction.effect_modifiers.get(measure) {
                        interaction_effect += effect_mod;
                    }
            }
        }

        // Check for complex interactions
        let condition_set: HashSet<&str> = conditions.iter().copied().collect();
        let mut complex_effect = 0.0;
        for complex in &self.complex_interactions {
            let complex_set: HashSet<&str> = complex.conditions.iter().map(|s| s.as_str()).collect();
            if complex_set.is_subset(&condition_set)
                && let Some(effect_mod) = complex.effect_modifiers.get(measure) {
                    complex_effect += effect_mod;
                }
        }

        let total_effect = additive_effect + interaction_effect + complex_effect;

        Ok(CombinedEffect {
            measure: measure.to_string(),
            additive_effect,
            interaction_effect,
            complex_effect,
            total_effect,
            synergy_index: if additive_effect.abs() > 1e-10 {
                total_effect / additive_effect
            } else {
                1.0
            },
        })
    }

    /// Calculate comorbidity index (Charlson-like).
    pub fn comorbidity_index(&self, conditions: &[&str]) -> f64 {
        let mut index = 0.0;
        for code in conditions {
            if let Some(cond) = self.conditions.get(*code) {
                index += cond.severity_weight;
            }
        }
        index
    }

    /// Get all conditions.
    pub fn conditions(&self) -> Vec<&Condition> {
        self.conditions.values().collect()
    }

    /// List all condition codes.
    pub fn condition_codes(&self) -> Vec<&String> {
        self.conditions.keys().collect()
    }

    /// Simulate biosignal modification for a patient profile.
    pub fn simulate_biosignal_effect(
        &self,
        conditions: &[&str],
        baseline_signal: &[f64],
    ) -> Result<Vec<f64>> {
        // Get amplitude and noise effects
        let amplitude_effect = self.combined_effect(conditions, "amplitude")?;
        let noise_effect = self.combined_effect(conditions, "noise")?;

        let amplitude_multiplier = 1.0 + amplitude_effect.total_effect;
        let noise_level = noise_effect.total_effect.abs();

        // Apply effects to signal
        let modified: Vec<f64> = baseline_signal.iter()
            .map(|&v| {
                let scaled = v * amplitude_multiplier;
                // Add simulated noise (deterministic for reproducibility)
                scaled + noise_level * (scaled.sin() * 0.1)
            })
            .collect();

        Ok(modified)
    }
}

/// A medical condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    /// Condition code (e.g., ICD-10).
    pub code: String,
    /// Display name.
    pub name: String,
    /// Category.
    pub category: ConditionCategory,
    /// Severity weight (for comorbidity index).
    pub severity_weight: f64,
    /// Effects on various measures.
    pub effects: HashMap<String, f64>,
    /// Description.
    pub description: Option<String>,
}

impl Condition {
    /// Create a new condition.
    pub fn new(code: &str, name: &str, category: ConditionCategory) -> Self {
        Self {
            code: code.to_string(),
            name: name.to_string(),
            category,
            severity_weight: 1.0,
            effects: HashMap::new(),
            description: None,
        }
    }

    /// Set severity weight.
    pub fn with_severity(mut self, weight: f64) -> Self {
        self.severity_weight = weight;
        self
    }

    /// Add an effect on a measure.
    pub fn with_effect(mut self, measure: &str, effect: f64) -> Self {
        self.effects.insert(measure.to_string(), effect);
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

/// Category of medical condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionCategory {
    /// Neurological conditions.
    Neurological,
    /// Cardiovascular conditions.
    Cardiovascular,
    /// Psychiatric conditions.
    Psychiatric,
    /// Metabolic conditions.
    Metabolic,
    /// Respiratory conditions.
    Respiratory,
    /// Musculoskeletal conditions.
    Musculoskeletal,
    /// Autoimmune conditions.
    Autoimmune,
    /// Infectious conditions.
    Infectious,
    /// Neoplastic conditions.
    Neoplastic,
    /// Other conditions.
    Other,
}

/// Pairwise interaction between conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    /// First condition.
    pub condition1: String,
    /// Second condition.
    pub condition2: String,
    /// Interaction type.
    pub interaction_type: InteractionType,
    /// Effect modifiers by measure.
    pub effect_modifiers: HashMap<String, f64>,
    /// Description.
    pub description: Option<String>,
}

impl Interaction {
    /// Create a new interaction.
    pub fn new(cond1: &str, cond2: &str, interaction_type: InteractionType) -> Self {
        Self {
            condition1: cond1.to_string(),
            condition2: cond2.to_string(),
            interaction_type,
            effect_modifiers: HashMap::new(),
            description: None,
        }
    }

    /// Add effect modifier.
    pub fn with_effect(mut self, measure: &str, modifier: f64) -> Self {
        self.effect_modifiers.insert(measure.to_string(), modifier);
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

/// Type of interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    /// Effects combine additively.
    Additive,
    /// Effects are greater than sum (positive synergy).
    Synergistic,
    /// Effects are less than sum (negative synergy).
    Antagonistic,
    /// One condition modifies the other.
    Modifying,
    /// No interaction.
    None,
}

/// Complex (higher-order) interaction involving 3+ conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexInteraction {
    /// Conditions involved.
    pub conditions: Vec<String>,
    /// Effect modifiers by measure.
    pub effect_modifiers: HashMap<String, f64>,
    /// Description.
    pub description: Option<String>,
}

impl ComplexInteraction {
    /// Create a new complex interaction.
    pub fn new(conditions: Vec<String>) -> Self {
        Self {
            conditions,
            effect_modifiers: HashMap::new(),
            description: None,
        }
    }

    /// Add effect modifier.
    pub fn with_effect(mut self, measure: &str, modifier: f64) -> Self {
        self.effect_modifiers.insert(measure.to_string(), modifier);
        self
    }
}

/// Combined effect result.
#[derive(Debug, Clone)]
pub struct CombinedEffect {
    /// Measure name.
    pub measure: String,
    /// Sum of individual effects.
    pub additive_effect: f64,
    /// Effect from pairwise interactions.
    pub interaction_effect: f64,
    /// Effect from complex interactions.
    pub complex_effect: f64,
    /// Total combined effect.
    pub total_effect: f64,
    /// Synergy index (total / additive).
    pub synergy_index: f64,
}

impl CombinedEffect {
    /// Create zero effect.
    pub fn none() -> Self {
        Self {
            measure: String::new(),
            additive_effect: 0.0,
            interaction_effect: 0.0,
            complex_effect: 0.0,
            total_effect: 0.0,
            synergy_index: 1.0,
        }
    }

    /// Whether effect is synergistic (greater than additive).
    pub fn is_synergistic(&self) -> bool {
        self.synergy_index > 1.1
    }

    /// Whether effect is antagonistic (less than additive).
    pub fn is_antagonistic(&self) -> bool {
        self.synergy_index < 0.9
    }
}

/// Patient comorbidity profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComorbidityProfile {
    /// Patient ID.
    pub patient_id: String,
    /// Active conditions.
    pub conditions: Vec<PatientCondition>,
    /// Comorbidity index score.
    pub index_score: Option<f64>,
}

impl ComorbidityProfile {
    /// Create new profile.
    pub fn new(patient_id: &str) -> Self {
        Self {
            patient_id: patient_id.to_string(),
            conditions: Vec::new(),
            index_score: None,
        }
    }

    /// Add a condition.
    pub fn add_condition(&mut self, condition: PatientCondition) {
        self.conditions.push(condition);
    }

    /// Get condition codes.
    pub fn condition_codes(&self) -> Vec<&str> {
        self.conditions.iter().map(|c| c.code.as_str()).collect()
    }

    /// Calculate index using a model.
    pub fn calculate_index(&mut self, model: &ComorbidityModel) {
        let codes: Vec<&str> = self.condition_codes();
        self.index_score = Some(model.comorbidity_index(&codes));
    }

    /// Check if patient has a condition.
    pub fn has_condition(&self, code: &str) -> bool {
        self.conditions.iter().any(|c| c.code == code)
    }

    /// Get number of conditions.
    pub fn condition_count(&self) -> usize {
        self.conditions.len()
    }
}

/// A condition in a patient's profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatientCondition {
    /// Condition code.
    pub code: String,
    /// Onset date.
    pub onset_date: Option<String>,
    /// Current status.
    pub status: ConditionStatus,
    /// Severity (patient-specific).
    pub severity: Option<ConditionSeverity>,
}

impl PatientCondition {
    /// Create new patient condition.
    pub fn new(code: &str) -> Self {
        Self {
            code: code.to_string(),
            onset_date: None,
            status: ConditionStatus::Active,
            severity: None,
        }
    }

    /// Set onset date.
    pub fn with_onset(mut self, date: &str) -> Self {
        self.onset_date = Some(date.to_string());
        self
    }

    /// Set status.
    pub fn with_status(mut self, status: ConditionStatus) -> Self {
        self.status = status;
        self
    }

    /// Set severity.
    pub fn with_severity(mut self, severity: ConditionSeverity) -> Self {
        self.severity = Some(severity);
        self
    }
}

/// Status of a condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionStatus {
    /// Currently active.
    Active,
    /// In remission.
    Remission,
    /// Resolved.
    Resolved,
    /// Chronic/Ongoing.
    Chronic,
}

/// Severity of a condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionSeverity {
    /// Mild severity.
    Mild,
    /// Moderate severity.
    Moderate,
    /// Severe.
    Severe,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model() -> ComorbidityModel {
        let mut model = ComorbidityModel::new("Test Model");

        // Add conditions
        let diabetes = Condition::new("E11", "Type 2 Diabetes", ConditionCategory::Metabolic)
            .with_severity(1.0)
            .with_effect("amplitude", -0.1)
            .with_effect("noise", 0.05);

        let hypertension = Condition::new("I10", "Hypertension", ConditionCategory::Cardiovascular)
            .with_severity(1.0)
            .with_effect("amplitude", -0.05)
            .with_effect("noise", 0.03);

        let depression = Condition::new("F32", "Depression", ConditionCategory::Psychiatric)
            .with_severity(1.0)
            .with_effect("amplitude", -0.15)
            .with_effect("noise", 0.08);

        model.add_condition(diabetes);
        model.add_condition(hypertension);
        model.add_condition(depression);

        // Add interaction
        let interaction = Interaction::new("E11", "I10", InteractionType::Synergistic)
            .with_effect("amplitude", -0.05)
            .with_effect("noise", 0.02);
        model.add_interaction("E11", "I10", interaction);

        model
    }

    #[test]
    fn test_single_condition_effect() {
        let model = create_test_model();
        let effect = model.combined_effect(&["E11"], "amplitude").unwrap();

        assert!((effect.total_effect - (-0.1)).abs() < 0.001);
        assert!((effect.additive_effect - (-0.1)).abs() < 0.001);
        assert!((effect.interaction_effect - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_pairwise_interaction() {
        let model = create_test_model();
        let effect = model.combined_effect(&["E11", "I10"], "amplitude").unwrap();

        // Additive: -0.1 + -0.05 = -0.15
        // Interaction: -0.05
        // Total: -0.20
        assert!((effect.additive_effect - (-0.15)).abs() < 0.001);
        assert!((effect.interaction_effect - (-0.05)).abs() < 0.001);
        assert!((effect.total_effect - (-0.20)).abs() < 0.001);
    }

    #[test]
    fn test_comorbidity_index() {
        let model = create_test_model();
        let index = model.comorbidity_index(&["E11", "I10", "F32"]);

        assert!((index - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_synergy_index() {
        let model = create_test_model();
        let effect = model.combined_effect(&["E11", "I10"], "amplitude").unwrap();

        // Total (-0.20) / Additive (-0.15) = 1.33
        assert!(effect.synergy_index > 1.0);
        assert!(effect.is_synergistic());
    }

    #[test]
    fn test_patient_profile() {
        let model = create_test_model();

        let mut profile = ComorbidityProfile::new("P001");
        profile.add_condition(PatientCondition::new("E11").with_severity(ConditionSeverity::Moderate));
        profile.add_condition(PatientCondition::new("I10"));

        assert_eq!(profile.condition_count(), 2);
        assert!(profile.has_condition("E11"));
        assert!(!profile.has_condition("F32"));

        profile.calculate_index(&model);
        assert!((profile.index_score.unwrap() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_biosignal_simulation() {
        let model = create_test_model();
        let baseline = vec![1.0, 0.5, -0.5, -1.0, 0.0];

        let modified = model.simulate_biosignal_effect(&["E11"], &baseline).unwrap();

        // Signal should be modified (amplitude reduced by 10%)
        assert_eq!(modified.len(), baseline.len());
        for (orig, mod_val) in baseline.iter().zip(modified.iter()) {
            // Modified values should be different (amplitude scaled + noise)
            if orig.abs() > 0.1 {
                assert!((mod_val.abs() - orig.abs() * 0.9).abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_complex_interaction() {
        let mut model = create_test_model();

        let complex = ComplexInteraction::new(vec![
            "E11".to_string(),
            "I10".to_string(),
            "F32".to_string(),
        ]).with_effect("amplitude", -0.1);

        model.add_complex_interaction(complex);

        let effect = model.combined_effect(&["E11", "I10", "F32"], "amplitude").unwrap();

        // Should include complex interaction effect
        assert!(effect.complex_effect.abs() > 0.0);
        assert!((effect.complex_effect - (-0.1)).abs() < 0.001);
    }

    #[test]
    fn test_condition_categories() {
        let neurological = Condition::new("G40", "Epilepsy", ConditionCategory::Neurological);
        let cardiac = Condition::new("I25", "CAD", ConditionCategory::Cardiovascular);

        assert_eq!(neurological.category, ConditionCategory::Neurological);
        assert_eq!(cardiac.category, ConditionCategory::Cardiovascular);
    }
}
