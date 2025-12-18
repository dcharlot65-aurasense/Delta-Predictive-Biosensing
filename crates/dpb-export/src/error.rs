//! Error types for model export operations.

use thiserror::Error;

/// Result type for export operations.
pub type Result<T> = std::result::Result<T, ExportError>;

/// Errors that can occur during model export.
#[derive(Error, Debug)]
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
