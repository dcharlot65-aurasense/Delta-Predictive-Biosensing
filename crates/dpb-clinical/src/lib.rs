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

pub use error::{ClinicalError, Result};
pub use demographics::{Demographics, Sex, Ethnicity, AgeGroup};
pub use normative::{NormativeDatabase, NormativeReference, PopulationNorms};
pub use treatment::{TreatmentResponse, InterventionModel, EffectSize};
pub use comorbidity::{ComorbidityModel, Condition, Interaction, ComorbidityProfile};
pub use practice_effects::{PracticeEffectCorrector, SerialAssessment};

/// Clinical data version.
pub const CLINICAL_DATA_VERSION: &str = "1.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clinical_data_version() {
        assert!(!CLINICAL_DATA_VERSION.is_empty());
    }
}
