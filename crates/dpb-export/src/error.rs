//! Error types for model export operations.

use thiserror::Error;

/// Result type for export operations.
pub type Result<T> = std::result::Result<T, ExportError>;

/// Errors that can occur during model export.
///
/// This enum is marked as non-exhaustive to allow adding new error variants
/// in future versions without breaking downstream code.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ExportError {
    /// IO error during file operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid model configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Unsupported export format.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// ONNX-specific error.
    #[error("ONNX error: {0}")]
    Onnx(String),

    /// Model validation failed.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Encoder not exportable.
    #[error("Encoder not exportable: {0}")]
    NotExportable(String),

    /// Missing required field.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Version incompatibility.
    #[error("Version incompatible: expected {expected}, got {actual}")]
    VersionMismatch {
        expected: String,
        actual: String,
    },

    /// Feature not enabled.
    #[error("Feature not enabled: {0}. Enable the '{0}' feature in Cargo.toml")]
    FeatureNotEnabled(String),
}

impl ExportError {
    /// Create an ONNX error.
    pub fn onnx(msg: impl Into<String>) -> Self {
        ExportError::Onnx(msg.into())
    }

    /// Create a validation error.
    pub fn validation(msg: impl Into<String>) -> Self {
        ExportError::Validation(msg.into())
    }

    /// Create an invalid config error.
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        ExportError::InvalidConfig(msg.into())
    }

    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, ExportError::Io(_) | ExportError::Validation(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = ExportError::onnx("test onnx error");
        assert!(matches!(err, ExportError::Onnx(_)));
        assert!(err.to_string().contains("ONNX error"));

        let err = ExportError::validation("test validation error");
        assert!(matches!(err, ExportError::Validation(_)));

        let err = ExportError::invalid_config("test config error");
        assert!(matches!(err, ExportError::InvalidConfig(_)));
    }

    #[test]
    fn test_error_display() {
        let err = ExportError::UnsupportedFormat("custom".to_string());
        assert!(err.to_string().contains("Unsupported format"));

        let err = ExportError::MissingField("threshold".to_string());
        assert!(err.to_string().contains("Missing required field"));

        let err = ExportError::VersionMismatch {
            expected: "1.0".to_string(),
            actual: "2.0".to_string(),
        };
        assert!(err.to_string().contains("expected 1.0"));
        assert!(err.to_string().contains("got 2.0"));
    }

    #[test]
    fn test_is_recoverable() {
        let io_err = ExportError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(io_err.is_recoverable());

        let validation_err = ExportError::validation("bad data");
        assert!(validation_err.is_recoverable());

        let onnx_err = ExportError::onnx("onnx failed");
        assert!(!onnx_err.is_recoverable());

        let feature_err = ExportError::FeatureNotEnabled("tensorflow".to_string());
        assert!(!feature_err.is_recoverable());
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let export_err: ExportError = io_err.into();
        assert!(matches!(export_err, ExportError::Io(_)));
    }

    #[test]
    fn test_from_json_error() {
        let json_result: std::result::Result<i32, _> = serde_json::from_str("not valid json");
        let json_err = json_result.unwrap_err();
        let export_err: ExportError = json_err.into();
        assert!(matches!(export_err, ExportError::Json(_)));
    }
}
