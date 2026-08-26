//! # dpb-clinical
//!
//! Clinical utilities for the Delta-Predictive Biosensing Framework.
//!
//! # ⚠️ Intended use — research and education only
//!
//! This crate is not a medical device. It is not FDA-cleared or CE-marked and has
//! not been validated for diagnosis, treatment, monitoring, or any clinical
//! decision. Outputs named after clinical rating scales are model estimates, not
//! clinical scores, and must not be interpreted as such.
//!
//! Using the de-identification helpers here does not by itself establish HIPAA
//! compliance. Compliance is a property of a covered entity's practices, not of
//! any library.
//!
//! ## Overview
//!
//! This crate provides clinical analysis tools including:
//!
//! - **Ethnic Stratification**: Population-specific normative data
//! - **Treatment Response**: Pre/post intervention modeling
//! - **Comorbidity Modeling**: Multi-disease simulation
//! - **Practice Effects**: Serial testing corrections
//! - **Normative Data**: Age/sex-adjusted reference ranges
//! - **PHI Tools**: HIPAA Safe Harbor de-identification (45 CFR 164.514(b)(2))
//!
//! ## HIPAA Compliance
//!
//! The `phi` module implements the HIPAA Safe Harbor de-identification method
//! (45 CFR 164.514(b)(2)):
//!
//! - Safe Harbor method (removal of 18 identifier types)
//! - Limited Data Set configuration
//! - Research pseudonymization
//! - K-Anonymity and L-Diversity checks
//!
//! ## Clinical Importance
//!
//! Biosignal characteristics vary across populations due to genetic,
//! environmental, and social factors. This crate ensures equitable
//! clinical tools by providing:
//!
//! 1. Population-stratified reference ranges
//! 2. Bias detection and correction
//! 3. Treatment effect size estimation
//! 4. Multi-condition interaction modeling
//!
//! ## Example
//!
//! ```rust
//! use dpb_clinical::{
//!     Demographics, Ethnicity, NormativeDatabase, NormativeReference, PopulationNorms, Sex,
//! };
//!
//! # fn example() -> dpb_clinical::Result<()> {
//! // Describe the patient. Every field is optional.
//! let patient = Demographics::new()
//!     .with_age(45)
//!     .with_sex(Sex::Female)
//!     .with_ethnicity(Ethnicity::EastAsian);
//!
//! // Supply norms for the population that patient falls into. These are YOUR
//! // cited values -- nothing usable ships with the crate.
//! let mut norms = PopulationNorms::new("adult female, East Asian");
//! norms.add_reference("alpha_power", NormativeReference::new(10.0, 2.0));
//!
//! let mut db = NormativeDatabase::new("Example Norms", "1.0");
//! db.add_norms(&patient.population_key(), norms, 250);
//!
//! // Compare a measurement against them.
//! let z_score = db.z_score("alpha_power", 12.5, &patient)?;
//! let percentile = db.percentile("alpha_power", 12.5, &patient)?;
//! println!("z = {z_score:.2}, percentile = {percentile:.1}");
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod demographics;
pub mod normative;
pub mod treatment;
pub mod comorbidity;
pub mod practice_effects;
pub mod phi;

// Re-export the crate's public surface.
//
// Most of these types were public in their modules but unreachable from the
// crate root, so the types needed to CALL the re-exported API -- `Assessment`
// for a `TreatmentResponse`, `ConditionCategory` for a `Condition`,
// `CorrectionMethod` for a `PracticeEffectCorrector` -- had to be named by
// module path, which nothing documented.
pub use error::{ClinicalError, Result};
pub use demographics::{
    AgeGroup, BroadEthnicCategory, DemographicFactor, Demographics, Ethnicity, FactorCategory,
    Handedness, Sex,
};
pub use normative::{
    ChangeClassification, MeasureDefinition, NormativeDatabase, NormativeReference,
    PercentileTable, PopulationNorms, ReliableChangeIndex,
};
pub use treatment::{
    Assessment, EffectInterpretation, EffectSize, EffectType, GroupStats, InterventionModel,
    ResponseClassification, ResponseCriteria, TreatmentGroup, TreatmentResponse,
};
pub use comorbidity::{
    CombinedEffect, ComorbidityModel, ComorbidityProfile, ComplexInteraction, Condition,
    ConditionCategory, ConditionSeverity, ConditionStatus, Interaction, InteractionType,
    PatientCondition,
};
// `practice_effects::EffectType` collides with the treatment one, so it keeps a
// qualified name here.
pub use practice_effects::{
    AssessmentSession, CorrectedScore, CorrectionMethod, EffectType as PracticeEffectType,
    PracticeEffect, PracticeEffectCorrector, PracticeEffectNorms, SRBCalculator,
    SRBClassification, SerialAssessment,
};
pub use phi::{DeIdentifier, DeIdentificationConfig, PatientRecord, DeIdentifiedRecord, PhiIdentifier};

/// Clinical data version.
pub const CLINICAL_DATA_VERSION: &str = "1.1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clinical_data_version() {
        assert!(!CLINICAL_DATA_VERSION.is_empty());
    }
}
