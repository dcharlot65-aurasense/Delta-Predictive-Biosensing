//! Error types for LSL integration

use thiserror::Error;

/// Result type for LSL operations.
pub type Result<T> = std::result::Result<T, LslError>;

/// Errors that can occur during LSL operations.
#[derive(Error, Debug)]
pub enum LslError {
    /// LSL stream not found during resolution.
    #[error("Stream not found: {name} (timeout: {timeout_sec}s)")]
    StreamNotFound {
        name: String,
        timeout_sec: f64,
    },

    /// Connection to stream timed out.
    #[error("Connection timeout after {0}ms")]
    ConnectionTimeout(u64),

    /// Pull operation timed out.
    #[error("Pull timeout after {0}s")]
    PullTimeout(f64),

    /// Channel count mismatch between expected and actual.
    #[error("Channel mismatch: expected {expected}, got {actual}")]
    ChannelMismatch {
        expected: usize,
        actual: usize,
    },

    /// Sample rate mismatch.
    #[error("Sample rate mismatch: expected {expected}, got {actual}")]
    SampleRateMismatch {
        expected: f64,
        actual: f64,
    },

    /// Invalid stream configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Stream is not open or has been closed.
    #[error("Stream not open")]
    StreamNotOpen,

    /// Internal LSL library error.
    #[error("LSL internal error: {0}")]
    Internal(String),

    /// Network error.
    #[error("Network error: {0}")]
    Network(String),

    /// Encoding error during pipeline operation.
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Buffer overflow.
    #[error("Buffer overflow: {0} samples lost")]
    BufferOverflow(usize),

    /// liblsl not found or not installed.
    #[error("liblsl not found. Please install from https://github.com/sccn/liblsl")]
    LibraryNotFound,

    /// Feature not supported.
    #[error("Feature not supported: {0}")]
    NotSupported(String),
}

impl LslError {
    /// Check if this is a timeout error.
    pub fn is_timeout(&self) -> bool {
        matches!(
            self,
            LslError::ConnectionTimeout(_) | LslError::PullTimeout(_)
        )
    }

    /// Check if this is a recoverable error.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            LslError::PullTimeout(_) | LslError::BufferOverflow(_)
        )
    }
}
