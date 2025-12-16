//! Error types for the DPB framework.

use thiserror::Error;

/// Main error type for the DPB framework.
#[derive(Error, Debug)]
pub enum DpbError {
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// GPU-related errors
    #[error("GPU error: {0}")]
    Gpu(String),

    /// WGPU-specific errors
    #[error("WGPU error: {0}")]
    Wgpu(#[from] wgpu::Error),

    /// Signal processing errors
    #[error("Signal processing error: {0}")]
    SignalProcessing(String),

    /// Invalid dimensions
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Invalid parameters
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Data validation errors
    #[error("Data validation error: {0}")]
    DataValidation(String),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization errors
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Feature extraction errors
    #[error("Feature extraction error: {0}")]
    FeatureExtraction(String),

    /// Encoding errors
    #[error("Encoding error: {0}")]
    Encoding(String),

    /// Network errors
    #[error("Network error: {0}")]
    Network(String),

    /// Training errors
    #[error("Training error: {0}")]
    Training(String),

    /// Hardware export errors
    #[error("Hardware export error: {0}")]
    HardwareExport(String),

    /// Resource not found
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    /// Out of bounds access
    #[error("Out of bounds: {0}")]
    OutOfBounds(String),

    /// Timeout error
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// General error with message
    #[error("{0}")]
    Other(String),
}

impl From<serde_json::Error> for DpbError {
    fn from(err: serde_json::Error) -> Self {
        DpbError::Serialization(err.to_string())
    }
}

impl From<bincode::Error> for DpbError {
    fn from(err: bincode::Error) -> Self {
        DpbError::Serialization(err.to_string())
    }
}

/// Result type alias using DpbError.
pub type Result<T> = std::result::Result<T, DpbError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DpbError::Config("test config error".to_string());
        assert_eq!(err.to_string(), "Configuration error: test config error");
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let dpb_err: DpbError = io_err.into();
        assert!(dpb_err.to_string().contains("file not found"));
    }
}
