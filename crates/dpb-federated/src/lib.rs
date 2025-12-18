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
//! - **Differential Privacy**: Optional noise injection for privacy guarantees
//! - **Secure Aggregation**: Cryptographic protection of model updates
//! - **Communication Efficiency**: Gradient compression and sparse updates
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
//! │       │    Encrypted Model Updates (ΔW)           │            │
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

pub use error::{FederatedError, Result};
pub use config::FedConfig;
pub use model::{ModelUpdate, ModelWeights, ParameterDelta};
pub use client::FederatedClient;
pub use server::FederatedServer;
pub use aggregation::{Aggregator, FedAvg, WeightedAvg};
pub use privacy::{PrivacyConfig, DifferentialPrivacy};

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
