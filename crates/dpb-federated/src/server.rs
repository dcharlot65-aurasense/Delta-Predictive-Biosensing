//! Federated learning server for aggregation.

use crate::{
    aggregation::{create_aggregator, Aggregator},
    config::FedConfig,
    model::{ModelUpdate, ModelWeights},
    FederatedError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Federated learning server.
///
/// Coordinates training rounds and aggregates client updates.
pub struct FederatedServer {
    /// Server configuration.
    config: FedConfig,
    /// Global model weights.
    global_model: ModelWeights,
    /// Current round number.
    current_round: u64,
    /// Registered clients.
    clients: HashMap<String, ClientInfo>,
    /// Updates received in current round.
    pending_updates: Vec<ModelUpdate>,
    /// Aggregation strategy.
    aggregator: Box<dyn Aggregator>,
    /// Server state.
    state: ServerState,
    /// Training history.
    history: ServerHistory,
}

impl FederatedServer {
    /// Create a new federated server.
    pub fn new(config: FedConfig) -> Self {
        let aggregator = create_aggregator(config.aggregation);

        Self {
            config,
            global_model: ModelWeights::default(),
            current_round: 0,
            clients: HashMap::new(),
            pending_updates: Vec::new(),
            aggregator,
            state: ServerState::Idle,
            history: ServerHistory::new(),
        }
    }

    /// Create with an initial model.
    pub fn with_model(config: FedConfig, model: ModelWeights) -> Self {
        let mut server = Self::new(config);
        server.global_model = model;
        server
    }

    /// Register a new client.
    pub fn register_client(&mut self, client_id: &str) -> Result<()> {
        if self.clients.contains_key(client_id) {
            return Err(FederatedError::ClientAlreadyRegistered(client_id.to_string()));
        }

        info!("Registering client: {}", client_id);
        self.clients.insert(
            client_id.to_string(),
            ClientInfo {
                id: client_id.to_string(),
                registered_at: chrono::Utc::now(),
                last_seen: chrono::Utc::now(),
                rounds_participated: 0,
                total_samples: 0,
                status: ClientStatus::Active,
            },
        );

        Ok(())
    }

    /// Unregister a client.
    pub fn unregister_client(&mut self, client_id: &str) -> Result<()> {
        if self.clients.remove(client_id).is_none() {
            return Err(FederatedError::ClientNotRegistered(client_id.to_string()));
        }
        info!("Unregistered client: {}", client_id);
        Ok(())
    }

    /// Get the global model.
    pub fn global_model(&self) -> &ModelWeights {
        &self.global_model
    }

    /// Get current round.
    pub fn current_round(&self) -> u64 {
        self.current_round
    }

    /// Get number of registered clients.
    pub fn num_clients(&self) -> usize {
        self.clients.len()
    }

    /// Get server state.
    pub fn state(&self) -> &ServerState {
        &self.state
    }

    /// Get training history.
    pub fn history(&self) -> &ServerHistory {
        &self.history
    }

    /// Start a new training round.
    pub fn start_round(&mut self) -> Result<()> {
        if self.clients.len() < self.config.min_clients {
            return Err(FederatedError::InsufficientClients {
                required: self.config.min_clients,
                available: self.clients.len(),
            });
        }

        self.current_round += 1;
        self.pending_updates.clear();
        self.state = ServerState::WaitingForUpdates;

        info!(
            "Started round {} with {} registered clients",
            self.current_round,
            self.clients.len()
        );

        Ok(())
    }

    /// Receive an update from a client.
    pub fn receive_update(&mut self, update: ModelUpdate) -> Result<()> {
        // Verify client is registered
        if !self.clients.contains_key(&update.client_id) {
            return Err(FederatedError::ClientNotRegistered(update.client_id.clone()));
        }

        // Verify round number
        if update.round != self.current_round {
            return Err(FederatedError::RoundMismatch {
                server_round: self.current_round,
                client_round: update.round,
            });
        }

        // Check if we already have an update from this client
        if self.pending_updates.iter().any(|u| u.client_id == update.client_id) {
            warn!(
                "Duplicate update from client {} in round {}",
                update.client_id, self.current_round
            );
            return Ok(());
        }

        debug!(
            "Received update from {} ({} samples)",
            update.client_id, update.num_samples
        );

        // Update client info
        if let Some(client) = self.clients.get_mut(&update.client_id) {
            client.last_seen = chrono::Utc::now();
            client.rounds_participated += 1;
            client.total_samples += update.num_samples;
        }

        self.pending_updates.push(update);

        Ok(())
    }

    /// Check if enough updates received to aggregate.
    pub fn can_aggregate(&self) -> bool {
        self.pending_updates.len() >= self.config.min_clients
    }

    /// Get number of pending updates.
    pub fn num_pending_updates(&self) -> usize {
        self.pending_updates.len()
    }

    /// Aggregate received updates and update global model.
    pub fn aggregate(&mut self) -> Result<&ModelWeights> {
        if !self.can_aggregate() {
            return Err(FederatedError::InsufficientClients {
                required: self.config.min_clients,
                available: self.pending_updates.len(),
            });
        }

        info!(
            "Aggregating {} updates for round {}",
            self.pending_updates.len(),
            self.current_round
        );

        // Perform aggregation
        let aggregated_delta = self.aggregator.aggregate(&self.pending_updates)?;

        // Apply to global model
        aggregated_delta.apply_to(&mut self.global_model)?;
        self.global_model.version = self.current_round;

        // Record history
        let total_samples: usize = self.pending_updates.iter().map(|u| u.num_samples).sum();
        let avg_loss: f32 = self
            .pending_updates
            .iter()
            .filter_map(|u| u.loss)
            .sum::<f32>()
            / self.pending_updates.len() as f32;

        self.history.record_round(
            self.current_round,
            self.pending_updates.len(),
            total_samples,
            avg_loss,
        );

        self.state = ServerState::RoundComplete;
        self.pending_updates.clear();

        info!(
            "Round {} complete. Avg loss: {:.4}",
            self.current_round, avg_loss
        );

        Ok(&self.global_model)
    }

    /// Run the full federated learning training loop.
    pub fn run_training<F>(&mut self, mut round_callback: F) -> Result<()>
    where
        F: FnMut(&mut Self, u64) -> Result<()>,
    {
        for _ in 0..self.config.num_rounds {
            self.start_round()?;

            // Call the callback to let external code coordinate clients
            round_callback(self, self.current_round)?;

            // Aggregate if we have enough updates
            if self.can_aggregate() {
                self.aggregate()?;
            } else {
                warn!(
                    "Round {} incomplete: only {} of {} required updates",
                    self.current_round,
                    self.pending_updates.len(),
                    self.config.min_clients
                );
            }
        }

        self.state = ServerState::TrainingComplete;
        info!("Training complete after {} rounds", self.config.num_rounds);

        Ok(())
    }

    /// Select clients for the current round.
    pub fn select_clients(&self) -> Vec<String> {
        use rand::seq::SliceRandom;

        let mut active_clients: Vec<String> = self
            .clients
            .iter()
            .filter(|(_, info)| info.status == ClientStatus::Active)
            .map(|(id, _)| id.clone())
            .collect();

        if self.config.client_fraction < 1.0 {
            let num_to_select =
                (active_clients.len() as f64 * self.config.client_fraction).ceil() as usize;
            let num_to_select = num_to_select.max(self.config.min_clients);

            let mut rng = rand::rng();
            active_clients.shuffle(&mut rng);
            active_clients.truncate(num_to_select);
        }

        active_clients
    }
}

/// Server state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServerState {
    /// Server is idle.
    Idle,
    /// Waiting for client updates.
    WaitingForUpdates,
    /// Aggregating updates.
    Aggregating,
    /// Round complete.
    RoundComplete,
    /// Training complete.
    TrainingComplete,
    /// Error state.
    Error(String),
}

/// Information about a registered client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    /// Client ID.
    pub id: String,
    /// Registration timestamp.
    pub registered_at: chrono::DateTime<chrono::Utc>,
    /// Last activity timestamp.
    pub last_seen: chrono::DateTime<chrono::Utc>,
    /// Number of rounds participated.
    pub rounds_participated: u64,
    /// Total samples trained on.
    pub total_samples: usize,
    /// Client status.
    pub status: ClientStatus,
}

/// Client status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientStatus {
    /// Client is active and available.
    Active,
    /// Client is temporarily unavailable.
    Unavailable,
    /// Client has been dropped.
    Dropped,
}

/// Training history for the server.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerHistory {
    /// Rounds completed.
    pub rounds: Vec<RoundInfo>,
}

impl ServerHistory {
    /// Create new history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a completed round.
    pub fn record_round(
        &mut self,
        round: u64,
        num_clients: usize,
        total_samples: usize,
        avg_loss: f32,
    ) {
        self.rounds.push(RoundInfo {
            round,
            num_clients,
            total_samples,
            avg_loss,
            timestamp: chrono::Utc::now(),
        });
    }

    /// Get the last recorded loss.
    pub fn last_loss(&self) -> Option<f32> {
        self.rounds.last().map(|r| r.avg_loss)
    }
}

/// Information about a training round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundInfo {
    /// Round number.
    pub round: u64,
    /// Number of participating clients.
    pub num_clients: usize,
    /// Total samples used.
    pub total_samples: usize,
    /// Average loss across clients.
    pub avg_loss: f32,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ParameterDelta, Tensor};

    fn create_test_update(client_id: &str, round: u64, values: Vec<f32>) -> ModelUpdate {
        let delta = ParameterDelta::new(
            round - 1,
            [("w".to_string(), Tensor::new(values, vec![4]))]
                .into_iter()
                .collect(),
        );
        ModelUpdate::new(client_id, round, delta, 100)
    }

    #[test]
    fn test_server_creation() {
        let config = FedConfig::new().with_min_clients(2);
        let server = FederatedServer::new(config);

        assert_eq!(server.current_round(), 0);
        assert_eq!(server.num_clients(), 0);
    }

    #[test]
    fn test_client_registration() {
        let config = FedConfig::new();
        let mut server = FederatedServer::new(config);

        server.register_client("client_a").unwrap();
        server.register_client("client_b").unwrap();

        assert_eq!(server.num_clients(), 2);

        // Duplicate registration should fail
        assert!(server.register_client("client_a").is_err());
    }

    #[test]
    fn test_start_round_insufficient_clients() {
        let config = FedConfig::new().with_min_clients(3);
        let mut server = FederatedServer::new(config);

        server.register_client("client_a").unwrap();
        server.register_client("client_b").unwrap();

        assert!(server.start_round().is_err());
    }

    #[test]
    fn test_receive_update_unregistered() {
        let config = FedConfig::new();
        let mut server = FederatedServer::new(config);

        let update = create_test_update("unknown", 1, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(server.receive_update(update).is_err());
    }

    #[test]
    fn test_full_round() {
        let config = FedConfig::new().with_min_clients(2);
        let model = ModelWeights::zeros(&[("w", vec![4])]);
        let mut server = FederatedServer::with_model(config, model);

        server.register_client("client_a").unwrap();
        server.register_client("client_b").unwrap();

        server.start_round().unwrap();
        assert_eq!(server.current_round(), 1);

        let update_a = create_test_update("client_a", 1, vec![1.0, 2.0, 3.0, 4.0]);
        let update_b = create_test_update("client_b", 1, vec![2.0, 4.0, 6.0, 8.0]);

        server.receive_update(update_a).unwrap();
        server.receive_update(update_b).unwrap();

        assert!(server.can_aggregate());
        let global = server.aggregate().unwrap();

        // FedAvg with equal samples: (1+2)/2=1.5, (2+4)/2=3, etc.
        assert_eq!(global.get("w").unwrap().data, vec![1.5, 3.0, 4.5, 6.0]);
    }

    #[test]
    fn test_round_mismatch() {
        let config = FedConfig::new().with_min_clients(1);
        let mut server = FederatedServer::new(config);

        server.register_client("client_a").unwrap();
        server.start_round().unwrap();

        let update = create_test_update("client_a", 5, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(server.receive_update(update).is_err());
    }

    #[test]
    fn test_server_history() {
        let mut history = ServerHistory::new();
        history.record_round(1, 3, 300, 0.5);
        history.record_round(2, 3, 300, 0.3);

        assert_eq!(history.rounds.len(), 2);
        assert_eq!(history.last_loss(), Some(0.3));
    }

    #[test]
    fn test_client_selection() {
        let config = FedConfig::new().with_client_fraction(0.5);
        let mut server = FederatedServer::new(config);

        for i in 0..10 {
            server.register_client(&format!("client_{}", i)).unwrap();
        }

        let selected = server.select_clients();
        assert!(selected.len() >= 2); // min_clients default is 2
        assert!(selected.len() <= 10);
    }
}
