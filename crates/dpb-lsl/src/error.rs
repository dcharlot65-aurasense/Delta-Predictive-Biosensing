//! Error types for LSL integration

use thiserror::Error;

/// Result type for LSL operations.
pub type Result<T> = std::result::Result<T, LslError>;

/// Errors that can occur during LSL operations.
///
/// This enum is marked as non-exhaustive to allow adding new error variants
/// in future versions without breaking downstream code.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum LslError {
    /// LSL stream not found during resolution.
    #[error("Stream not found: {name} (timeout: {timeout_sec}s)")]
    StreamNotFound { name: String, timeout_sec: f64 },

    /// Connection to stream timed out.
    #[error("Connection timeout after {0}ms")]
    ConnectionTimeout(u64),

    /// Pull operation timed out.
    #[error("Pull timeout after {0}s")]
    PullTimeout(f64),

    /// Channel count mismatch between expected and actual.
    #[error("Channel mismatch: expected {expected}, got {actual}")]
    ChannelMismatch { expected: usize, actual: usize },

    /// Sample rate mismatch.
    #[error("Sample rate mismatch: expected {expected}, got {actual}")]
    SampleRateMismatch { expected: f64, actual: f64 },

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

    /// Generic timeout error.
    #[error("Timeout during {operation} after {timeout_sec}s")]
    Timeout { operation: String, timeout_sec: f64 },

    /// Connection lost during operation.
    #[error("Connection lost: {0}")]
    ConnectionLost(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl LslError {
    /// Check if this is a timeout error.
    pub fn is_timeout(&self) -> bool {
        matches!(
            self,
            LslError::ConnectionTimeout(_) | LslError::PullTimeout(_) | LslError::Timeout { .. }
        )
    }

    /// Check if this is a recoverable error.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, LslError::PullTimeout(_) | LslError::BufferOverflow(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_not_found_error() {
        let err = LslError::StreamNotFound {
            name: "TestStream".to_string(),
            timeout_sec: 5.0,
        };
        assert!(err.to_string().contains("TestStream"));
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn test_connection_timeout_error() {
        let err = LslError::ConnectionTimeout(1000);
        assert!(err.to_string().contains("1000"));
        assert!(err.is_timeout());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_pull_timeout_error() {
        let err = LslError::PullTimeout(0.5);
        assert!(err.to_string().contains("0.5"));
        assert!(err.is_timeout());
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_channel_mismatch_error() {
        let err = LslError::ChannelMismatch {
            expected: 8,
            actual: 4,
        };
        assert!(err.to_string().contains("expected 8"));
        assert!(err.to_string().contains("got 4"));
        assert!(!err.is_timeout());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_sample_rate_mismatch_error() {
        let err = LslError::SampleRateMismatch {
            expected: 256.0,
            actual: 512.0,
        };
        assert!(err.to_string().contains("256"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn test_invalid_config_error() {
        let err = LslError::InvalidConfig("Bad setting".to_string());
        assert!(err.to_string().contains("Bad setting"));
    }

    #[test]
    fn test_stream_not_open_error() {
        let err = LslError::StreamNotOpen;
        assert!(err.to_string().contains("not open"));
    }

    #[test]
    fn test_internal_error() {
        let err = LslError::Internal("liblsl crashed".to_string());
        assert!(err.to_string().contains("liblsl crashed"));
    }

    #[test]
    fn test_network_error() {
        let err = LslError::Network("Connection refused".to_string());
        assert!(err.to_string().contains("Connection refused"));
    }

    #[test]
    fn test_encoding_error() {
        let err = LslError::EncodingError("Invalid threshold".to_string());
        assert!(err.to_string().contains("Invalid threshold"));
    }

    #[test]
    fn test_buffer_overflow_error() {
        let err = LslError::BufferOverflow(100);
        assert!(err.to_string().contains("100"));
        assert!(err.is_recoverable());
        assert!(!err.is_timeout());
    }

    #[test]
    fn test_library_not_found_error() {
        let err = LslError::LibraryNotFound;
        assert!(err.to_string().contains("liblsl"));
    }

    #[test]
    fn test_not_supported_error() {
        let err = LslError::NotSupported("64-bit integers".to_string());
        assert!(err.to_string().contains("64-bit integers"));
    }

    #[test]
    fn test_is_timeout() {
        assert!(LslError::ConnectionTimeout(1000).is_timeout());
        assert!(LslError::PullTimeout(1.0).is_timeout());
        assert!(!LslError::StreamNotOpen.is_timeout());
        assert!(!LslError::BufferOverflow(10).is_timeout());
    }

    #[test]
    fn test_is_recoverable() {
        assert!(LslError::PullTimeout(1.0).is_recoverable());
        assert!(LslError::BufferOverflow(10).is_recoverable());
        assert!(!LslError::ConnectionTimeout(1000).is_recoverable());
        assert!(!LslError::StreamNotOpen.is_recoverable());
        assert!(!LslError::Internal("error".to_string()).is_recoverable());
    }
}
