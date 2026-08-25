//! # dpb-federated
//!
//! Privacy-preserving federated learning for the Delta-Predictive Biosensing Framework.
//!
//! ## Overview
//!
//! This crate enables multi-site collaborative training of spike encoding models
//! without sharing raw biosignal data. Key features:
//!
//! - **Federated Averaging (FedAvg)**: Standard federated learning algorithm
//! - **Differential Privacy**: Optional Gaussian/Laplace noise injection with
//!   gradient clipping
//! - **Communication Efficiency**: Gradient compression and sparse updates
//!
//! ## ⚠️ What this crate does NOT provide
//!
//! **There is no secure aggregation and no cryptographic protection of model
//! updates.** Model updates are transmitted in plaintext. Do not use this crate
//! in a threat model where the aggregation server is untrusted, or where model
//! updates must be confidential in transit or at rest. Transport security is
//! entirely your responsibility.
//!
//! The differential-privacy budget accountant uses **basic composition**, which
//! is an approximation. It is not a certified DP accountant (it is not
//! Rényi/moments-accountant based), and it must not be relied upon for a formal
//! privacy guarantee or for any regulatory privacy claim.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Federated Learning System                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
//! │  │ Site A  │    │ Site B  │    │ Site C  │    │ Site N  │      │
//! │  │ Client  │    │ Client  │    │ Client  │    │ Client  │      │
//! │  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘      │
//! │       │              │              │              │            │
//! │       │    Model Updates (ΔW) — PLAINTEXT         │            │
//! │       └──────────────┼──────────────┼──────────────┘            │
//! │                      ▼                                          │
//! │              ┌───────────────┐                                  │
//! │              │  Aggregation  │                                  │
//! │              │    Server     │                                  │
//! │              └───────┬───────┘                                  │
//! │                      │                                          │
//! │                      ▼                                          │
//! │              ┌───────────────┐                                  │
//! │              │ Global Model  │                                  │
//! │              │   Update      │                                  │
//! │              └───────────────┘                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use dpb_federated::{FederatedClient, FederatedServer, FedConfig};
//!
//! // Server setup
//! let config = FedConfig::default()
//!     .with_rounds(100)
//!     .with_min_clients(3);
//! let mut server = FederatedServer::new(config);
//!
//! // Client setup at each site
//! let client = FederatedClient::new("site_a", local_model);
//!
//! // Training loop
//! for round in 0..100 {
//!     // Clients train locally
//!     let update = client.train_round(&local_data)?;
//!
//!     // Server aggregates
//!     server.receive_update(client.id(), update)?;
//!     let global_model = server.aggregate()?;
//!
//!     // Clients sync
//!     client.sync_model(&global_model)?;
//! }
//! ```

pub mod error;
pub mod config;
pub mod model;
pub mod client;
pub mod server;
pub mod aggregation;
pub mod privacy;
pub mod compression;

// Re-export the crate's public surface.
//
// These types were public in their modules but invisible at the crate root, so
// `use dpb_federated::*` -- the obvious way to reach for them, and what the
// integration suite does -- brought in barely a third of the API.
pub use error::{FederatedError, Result};
pub use config::{AggregationStrategy, ClientSelection, FedConfig};
pub use model::{
    CompressionInfo, ModelMetadata, ModelUpdate, ModelWeights, ParameterDelta, Tensor,
};
pub use client::{ClientState, FederatedClient, LocalDataset, TrainingHistory};
pub use server::{
    ClientInfo, ClientStatus, FederatedServer, RoundInfo, ServerHistory, ServerState,
};
pub use aggregation::{
    Aggregator, FedAvg, MedianAggregator, TrimmedMeanAggregator, WeightedAvg,
};
pub use privacy::{
    DifferentialPrivacy, LocalDP, PrivacyAccountant, PrivacyConfig, PrivacyMechanism,
};
pub use compression::{CompressionStrategy, GradientCompressor, SparseTensor};

/// Federated learning protocol version.
pub const PROTOCOL_VERSION: &str = "1.0";

/// Maximum supported number of clients.
pub const MAX_CLIENTS: usize = 1000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_version() {
        assert_eq!(PROTOCOL_VERSION, "1.0");
    }

    #[test]
    fn test_max_clients() {
        assert!(MAX_CLIENTS >= 100);
    }
}
