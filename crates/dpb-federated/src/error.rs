//! Error types for federated learning operations.

use thiserror::Error;

/// Result type for federated learning operations.
pub type Result<T> = std::result::Result<T, FederatedError>;

/// Errors that can occur during federated learning.
///
/// This enum is marked as non-exhaustive to allow adding new error variants
/// in future versions without breaking downstream code.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum FederatedError {
    /// Client not registered with server.
    #[error("Client not registered: {0}")]
    ClientNotRegistered(String),

    /// Client already registered.
    #[error("Client already registered: {0}")]
    ClientAlreadyRegistered(String),

    /// Not enough clients for aggregation.
    #[error("Insufficient clients: need {required}, have {available}")]
    InsufficientClients { required: usize, available: usize },

    /// Model dimension mismatch between clients.
    #[error("Model dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Invalid model weights.
    #[error("Invalid model weights: {0}")]
    InvalidWeights(String),

    /// Round number out of sync.
    #[error("Round mismatch: server at {server_round}, client sent {client_round}")]
    RoundMismatch { server_round: u64, client_round: u64 },

    /// Privacy budget exhausted.
    #[error("Privacy budget exhausted: epsilon={epsilon}, delta={delta}")]
    PrivacyBudgetExhausted { epsilon: f64, delta: f64 },

    /// Encryption/decryption error.
    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    /// Communication error.
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Timeout waiting for clients.
    #[error("Timeout waiting for {expected} clients, received {received}")]
    Timeout { expected: usize, received: usize },

    /// Server not ready.
    #[error("Server not ready: {0}")]
    ServerNotReady(String),

    /// Training error.
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Invalid state transition.
    #[error("Invalid state transition: {from} -> {to}")]
    InvalidStateTransition { from: String, to: String },
}

impl FederatedError {
    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            FederatedError::Timeout { .. }
                | FederatedError::CommunicationError(_)
                | FederatedError::InsufficientClients { .. }
        )
    }

    /// Check if this is a privacy-related error.
    pub fn is_privacy_error(&self) -> bool {
        matches!(
            self,
            FederatedError::PrivacyBudgetExhausted { .. } | FederatedError::CryptoError(_)
        )
    }
}

impl From<serde_json::Error> for FederatedError {
    fn from(err: serde_json::Error) -> Self {
        FederatedError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for FederatedError {
    fn from(err: std::io::Error) -> Self {
        FederatedError::CommunicationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_clients_error() {
        let err = FederatedError::InsufficientClients {
            required: 5,
            available: 3,
        };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("3"));
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_privacy_budget_exhausted() {
        let err = FederatedError::PrivacyBudgetExhausted {
            epsilon: 1.0,
            delta: 1e-5,
        };
        assert!(err.is_privacy_error());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = FederatedError::DimensionMismatch {
            expected: 100,
            actual: 50,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_timeout_is_recoverable() {
        let err = FederatedError::Timeout {
            expected: 10,
            received: 5,
        };
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_crypto_error_is_privacy_error() {
        let err = FederatedError::CryptoError("decryption failed".to_string());
        assert!(err.is_privacy_error());
    }
}
