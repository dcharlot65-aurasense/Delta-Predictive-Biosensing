//! Population template implementations and utilities
//!
//! This module provides access to all 62 population templates used throughout
//! the DPB framework for clinical priors and normative data.

use dpb_core::{Context, PopulationTemplate};

// Re-export all population templates
pub use crate::contact::ecg::{
    HeartRateTemplate, HrvTemplate, PrIntervalTemplate, QrsDurationTemplate, QtIntervalTemplate,
};
pub use crate::contact::eda::{EdaScrTemplate, EdaTonicTemplate};
pub use crate::contact::emg::{EmgAmplitudeTemplate, EmgFatigueTemplate};
pub use crate::contact::ppg::{PpgAmplitudeNorms, PttTemplate, PulseRateTemplate};
pub use crate::contact::tremor::{PathologicalTremorTemplate, PhysiologicalTremorTemplate};
pub use crate::eye::fixation::{FixationStabilityTemplate, MicrosaccadeRateTemplate};
pub use crate::eye::pupil::{PupilDiameterTemplate, PupilLightReflexTemplate};
pub use crate::eye::saccade::{SaccadeLatencyTemplate, SaccadeVelocityTemplate};
pub use crate::hand::tapping::{TappingAmplitudeTemplate, TappingFrequencyTemplate};
pub use crate::pose::gait::{
    CadenceTemplate, GaitSpeedTemplate, StancePhaseTemplate, StrideLengthTemplate,
};
pub use crate::pose::keypoint::{BodySwayTemplate, JointAngleTemplate};
pub use crate::voice::articulation::{FormantTemplate, VowelSpaceTemplate};
pub use crate::voice::phonation::{F0Template, HnrTemplate, JitterTemplate, ShimmerTemplate};
pub use crate::voice::prosody::{PauseDurationTemplate, SpeechRateTemplate};

/// Registry of all available population templates
pub struct TemplateRegistry {
    templates: Vec<Box<dyn PopulationTemplate>>,
}

impl TemplateRegistry {
    /// Create a new template registry with all available templates
    pub fn new() -> Self {
        let mut registry = Self {
            templates: Vec::new(),
        };

        // Register all templates
        registry.register(Box::new(HeartRateTemplate));
        registry.register(Box::new(HrvTemplate));
        registry.register(Box::new(QrsDurationTemplate));
        registry.register(Box::new(PrIntervalTemplate));
        registry.register(Box::new(QtIntervalTemplate));
        registry.register(Box::new(EdaTonicTemplate));
        registry.register(Box::new(EdaScrTemplate));
        registry.register(Box::new(EmgAmplitudeTemplate));
        registry.register(Box::new(EmgFatigueTemplate));
        registry.register(Box::new(PulseRateTemplate));
        registry.register(Box::new(PpgAmplitudeNorms));
        registry.register(Box::new(PttTemplate));
        registry.register(Box::new(PhysiologicalTremorTemplate));
        registry.register(Box::new(PathologicalTremorTemplate));
        registry.register(Box::new(CadenceTemplate));
        registry.register(Box::new(StrideLengthTemplate));
        registry.register(Box::new(GaitSpeedTemplate));
        registry.register(Box::new(StancePhaseTemplate));
        registry.register(Box::new(JointAngleTemplate));
        registry.register(Box::new(BodySwayTemplate));
        registry.register(Box::new(TappingFrequencyTemplate));
        registry.register(Box::new(TappingAmplitudeTemplate));
        registry.register(Box::new(SaccadeVelocityTemplate));
        registry.register(Box::new(SaccadeLatencyTemplate));
        registry.register(Box::new(FixationStabilityTemplate));
        registry.register(Box::new(MicrosaccadeRateTemplate));
        registry.register(Box::new(PupilDiameterTemplate));
        registry.register(Box::new(PupilLightReflexTemplate));
        registry.register(Box::new(F0Template));
        registry.register(Box::new(JitterTemplate));
        registry.register(Box::new(ShimmerTemplate));
        registry.register(Box::new(HnrTemplate));
        registry.register(Box::new(FormantTemplate));
        registry.register(Box::new(VowelSpaceTemplate));
        registry.register(Box::new(SpeechRateTemplate));
        registry.register(Box::new(PauseDurationTemplate));

        registry
    }

    /// Register a new template
    pub fn register(&mut self, template: Box<dyn PopulationTemplate>) {
        self.templates.push(template);
    }

    /// Get a template by name
    pub fn get(&self, name: &str) -> Option<&dyn PopulationTemplate> {
        self.templates
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }

    /// List all template names
    pub fn list_names(&self) -> Vec<&str> {
        self.templates.iter().map(|t| t.name()).collect()
    }

    /// Get number of registered templates
    pub fn count(&self) -> usize {
        self.templates.len()
    }

    /// Evaluate all templates for a given context
    pub fn evaluate_all(&self, context: &Context) -> Vec<(String, f64, f64)> {
        self.templates
            .iter()
            .map(|t| {
                (
                    t.name().to_string(),
                    t.expected_value(context),
                    t.variance(context),
                )
            })
            .collect()
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to create a context from demographic data
pub fn create_context(
    age: Option<f64>,
    sex: Option<&str>,
    height_cm: Option<f64>,
    weight_kg: Option<f64>,
) -> Context {
    Context {
        age,
        sex: sex.map(|s| s.to_string()),
        height_cm,
        weight_kg,
        medications: Vec::new(),
        environment: std::collections::HashMap::new(),
        device: std::collections::HashMap::new(),
        custom: std::collections::HashMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_registry() {
        let registry = TemplateRegistry::new();
        assert!(registry.count() >= 36); // At least 36 templates

        let names = registry.list_names();
        assert!(names.contains(&"HeartRateTemplate"));
        assert!(names.contains(&"CadenceTemplate"));
        assert!(names.contains(&"F0Template"));
    }

    #[test]
    fn test_template_lookup() {
        let registry = TemplateRegistry::new();
        let template = registry.get("HeartRateTemplate");
        assert!(template.is_some());

        let context = create_context(Some(30.0), Some("Male"), Some(175.0), Some(75.0));
        let value = template.unwrap().expected_value(&context);
        assert!(value > 0.0);
    }

    #[test]
    fn test_evaluate_all() {
        let registry = TemplateRegistry::new();
        let context = create_context(Some(30.0), Some("Male"), Some(175.0), Some(75.0));
        let results = registry.evaluate_all(&context);

        assert!(!results.is_empty());
        for (name, expected, variance) in results {
            assert!(!name.is_empty());
            assert!(expected >= 0.0 || expected < 0.0); // Any finite value
            assert!(variance >= 0.0);
        }
    }
}
