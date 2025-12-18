//! # dpb-clinical
//!
//! Clinical utilities for the Delta-Predictive Biosensing Framework.
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
//! - **HIPAA/PHI Tools**: De-identification and anonymization (NEW)
//!
//! ## HIPAA Compliance
//!
//! The `phi` module provides HIPAA-compliant de-identification tools:
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
//! ```rust,ignore
//! use dpb_clinical::{NormativeDatabase, Population, Demographics};
//!
//! // Load normative data
//! let norms = NormativeDatabase::load("eeg_norms.json")?;
//!
//! // Create patient demographics
//! let patient = Demographics::new()
//!     .with_age(45)
//!     .with_sex(Sex::Female)
//!     .with_ethnicity(Ethnicity::EastAsian);
//!
//! // Get population-specific reference
//! let reference = norms.get_reference(&patient, "alpha_power")?;
//!
//! // Calculate z-score
//! let patient_value = 12.5;
//! let z_score = reference.z_score(patient_value);
//! ```

pub mod error;
pub mod demographics;
pub mod normative;
pub mod treatment;
pub mod comorbidity;
pub mod practice_effects;
pub mod phi;

pub use error::{ClinicalError, Result};
pub use demographics::{Demographics, Sex, Ethnicity, AgeGroup};
pub use normative::{NormativeDatabase, NormativeReference, PopulationNorms};
pub use treatment::{TreatmentResponse, InterventionModel, EffectSize};
pub use comorbidity::{ComorbidityModel, Condition, Interaction, ComorbidityProfile};
pub use practice_effects::{PracticeEffectCorrector, SerialAssessment};
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
