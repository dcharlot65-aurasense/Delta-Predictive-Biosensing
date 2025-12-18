//! Error types for clinical operations.

use thiserror::Error;

/// Result type for clinical operations.
pub type Result<T> = std::result::Result<T, ClinicalError>;

/// Errors that can occur during clinical operations.
#[derive(Error, Debug)]
pub enum ClinicalError {
    /// Demographic data missing or invalid.
    #[error("Invalid demographics: {0}")]
    InvalidDemographics(String),

    /// Population not found in normative database.
    #[error("Population not found: {0}")]
    PopulationNotFound(String),

    /// Measure not found in normative database.
    #[error("Measure not found: {0}")]
    MeasureNotFound(String),

    /// Insufficient data for analysis.
    #[error("Insufficient data: need {required}, have {available}")]
    InsufficientData { required: usize, available: usize },

    /// Age out of range for normative data.
    #[error("Age {age} out of range [{min}, {max}]")]
    AgeOutOfRange { age: u8, min: u8, max: u8 },

    /// Invalid measurement value.
    #[error("Invalid measurement value: {0}")]
    InvalidMeasurement(String),

    /// Statistical computation error.
    #[error("Statistical error: {0}")]
    StatisticalError(String),

    /// Treatment model error.
    #[error("Treatment model error: {0}")]
    TreatmentError(String),

    /// Comorbidity modeling error.
    #[error("Comorbidity error: {0}")]
    ComorbidityError(String),

    /// Missing normative data.
    #[error("Missing normative data: {0}")]
    MissingNormativeData(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl ClinicalError {
    /// Check if this is a data-related error.
    pub fn is_data_error(&self) -> bool {
        matches!(
            self,
            ClinicalError::InvalidDemographics(_)
                | ClinicalError::InvalidMeasurement(_)
                | ClinicalError::InsufficientData { .. }
        )
    }
}

impl From<serde_json::Error> for ClinicalError {
    fn from(err: serde_json::Error) -> Self {
        ClinicalError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ClinicalError::PopulationNotFound("test_pop".to_string());
        assert!(err.to_string().contains("test_pop"));
    }

    #[test]
    fn test_is_data_error() {
        assert!(ClinicalError::InvalidDemographics("test".into()).is_data_error());
        assert!(ClinicalError::InsufficientData {
            required: 10,
            available: 5
        }
        .is_data_error());
        assert!(!ClinicalError::StatisticalError("test".into()).is_data_error());
    }
}
