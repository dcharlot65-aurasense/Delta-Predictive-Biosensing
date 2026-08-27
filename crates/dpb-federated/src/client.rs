//! Federated learning client for local training.

use crate::{
    config::FedConfig,
    model::{ModelUpdate, ModelWeights, ParameterDelta},
    privacy::DifferentialPrivacy,
    FederatedError, Result,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Federated learning client.
///
/// Runs local training and produces model updates for the server.
pub struct FederatedClient {
    /// Client identifier.
    id: String,
    /// Local model weights.
    model: ModelWeights,
    /// Configuration.
    config: FedConfig,
    /// Current round number.
    current_round: u64,
    /// Local differential privacy (optional).
    privacy: Option<DifferentialPrivacy>,
    /// Training state.
    state: ClientState,
    /// Training history.
    history: TrainingHistory,
}

impl FederatedClient {
    /// Create a new federated client.
    pub fn new(id: &str, model: ModelWeights) -> Self {
        Self {
            id: id.to_string(),
            model,
            config: FedConfig::default(),
            current_round: 0,
            privacy: None,
            state: ClientState::Idle,
            history: TrainingHistory::new(),
        }
    }

    /// Create with configuration.
    pub fn with_config(id: &str, model: ModelWeights, config: FedConfig) -> Self {
        let privacy = config.privacy.as_ref().map(|p| DifferentialPrivacy::new(p.clone()));

        Self {
            id: id.to_string(),
            model,
            config,
            current_round: 0,
            privacy,
            state: ClientState::Idle,
            history: TrainingHistory::new(),
        }
    }

    /// Get client ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get current model weights.
    pub fn model(&self) -> &ModelWeights {
        &self.model
    }

    /// Get current round.
    pub fn current_round(&self) -> u64 {
        self.current_round
    }

    /// Get client state.
    pub fn state(&self) -> &ClientState {
        &self.state
    }

    /// Get training history.
    pub fn history(&self) -> &TrainingHistory {
        &self.history
    }

    /// Synchronize with global model from server.
    pub fn sync_model(&mut self, global_model: &ModelWeights) -> Result<()> {
        info!("Client {} syncing model (version {})", self.id, global_model.version);

        // Verify dimensions match
        for (name, tensor) in &global_model.parameters {
            if let Some(local_tensor) = self.model.parameters.get(name)
                && local_tensor.data.len() != tensor.data.len() {
                    return Err(FederatedError::DimensionMismatch {
                        expected: local_tensor.data.len(),
                        actual: tensor.data.len(),
                    });
                }
        }

        self.model = global_model.clone();
        self.current_round = global_model.version;
        self.state = ClientState::Idle;

        Ok(())
    }

    /// Perform local training and produce an update.
    ///
    /// This is a simplified training loop. In practice, you would integrate
    /// with actual neural network training code.
    pub fn train_round(&mut self, data: &LocalDataset) -> Result<ModelUpdate> {
        self.state = ClientState::Training;
        info!(
            "Client {} starting local training (round {}, {} samples)",
            self.id, self.current_round, data.num_samples
        );

        // Store initial weights
        let initial_model = self.model.clone();

        // Simulate local training
        // In practice, this would be replaced with actual gradient descent
        let mut total_loss = 0.0f32;

        for epoch in 0..self.config.local_epochs {
            let epoch_loss = self.train_epoch(data)?;
            total_loss += epoch_loss;
            debug!(
                "Client {} epoch {}: loss = {:.4}",
                self.id, epoch, epoch_loss
            );
        }

        let avg_loss = total_loss / self.config.local_epochs as f32;

        // Compute delta
        let mut delta = ParameterDelta::from_models(&initial_model, &self.model)?;

        // Apply differential privacy if configured
        if let Some(ref mut dp) = self.privacy {
            let mut delta_weights = ModelWeights::new(delta.changes.clone());
            dp.privatize(&mut delta_weights)?;
            delta.changes = delta_weights.parameters;
        }

        // Create update
        let update = ModelUpdate::new(&self.id, self.current_round, delta, data.num_samples)
            .with_loss(avg_loss)
            .with_metric("epochs", self.config.local_epochs as f32);

        // Record in history
        self.history.record_round(self.current_round, avg_loss, data.num_samples);

        self.state = ClientState::WaitingForServer;
        Ok(update)
    }

    /// Train one epoch (simplified).
    fn train_epoch(&mut self, data: &LocalDataset) -> Result<f32> {
        // This is a placeholder for actual training
        // In practice, you would:
        // 1. Iterate over batches
        // 2. Compute gradients
        // 3. Update weights

        let mut loss = 0.0f32;
        let lr = self.config.learning_rate as f32;
        let _momentum = self.config.momentum as f32;
        let weight_decay = self.config.weight_decay as f32;

        // Simulate gradient updates with mock gradients
        for tensor in self.model.parameters.values_mut() {
            for (i, value) in tensor.data.iter_mut().enumerate() {
                // Mock gradient (would come from backprop)
                let grad = data.mock_gradient(i);

                // Weight decay
                *value *= 1.0 - lr * weight_decay;

                // Gradient update
                *value -= lr * grad;

                // Accumulate loss
                loss += grad * grad;
            }
        }

        Ok(loss / self.model.num_parameters as f32)
    }

    /// Check if local training is complete.
    pub fn is_ready(&self) -> bool {
        matches!(self.state, ClientState::WaitingForServer)
    }
}

/// Client state machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientState {
    /// Client is idle, waiting to start.
    Idle,
    /// Client is downloading global model.
    Downloading,
    /// Client is training locally.
    Training,
    /// Client is uploading update.
    Uploading,
    /// Client is waiting for server aggregation.
    WaitingForServer,
    /// Client encountered an error.
    Error(String),
}

/// Local dataset representation.
// Held but not consulted yet; kept so a caller's input is not silently
// discarded.
#[allow(dead_code)]
pub struct LocalDataset {
    /// Number of samples.
    pub num_samples: usize,
    /// Feature dimension.
    pub feature_dim: usize,
    /// Mock data (for demonstration).
    data: Vec<f32>,
}

impl LocalDataset {
    /// Create a new local dataset.
    pub fn new(num_samples: usize, feature_dim: usize) -> Self {
        Self {
            num_samples,
            feature_dim,
            data: vec![0.0; num_samples * feature_dim],
        }
    }

    /// Create with random data.
    pub fn random(num_samples: usize, feature_dim: usize) -> Self {
        use rand::RngExt;
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..num_samples * feature_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        Self {
            num_samples,
            feature_dim,
            data,
        }
    }

    /// Mock gradient based on index (for demo).
    fn mock_gradient(&self, _index: usize) -> f32 {
        use rand::RngExt;
        let mut rng = rand::rng();
        0.01 * rng.random_range(-1.0..1.0)
    }
}

/// Training history for a client.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Loss per round.
    pub losses: Vec<f32>,
    /// Samples per round.
    pub samples: Vec<usize>,
    /// Rounds participated.
    pub rounds: Vec<u64>,
}

impl TrainingHistory {
    /// Create new history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a training round.
    pub fn record_round(&mut self, round: u64, loss: f32, num_samples: usize) {
        self.rounds.push(round);
        self.losses.push(loss);
        self.samples.push(num_samples);
    }

    /// Get average loss.
    pub fn average_loss(&self) -> f32 {
        if self.losses.is_empty() {
            return 0.0;
        }
        self.losses.iter().sum::<f32>() / self.losses.len() as f32
    }

    /// Total samples trained on.
    pub fn total_samples(&self) -> usize {
        self.samples.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let model = ModelWeights::zeros(&[("w", vec![10])]);
        let client = FederatedClient::new("test_client", model);

        assert_eq!(client.id(), "test_client");
        assert_eq!(client.current_round(), 0);
        assert_eq!(client.state(), &ClientState::Idle);
    }

    #[test]
    fn test_client_sync_model() {
        let model = ModelWeights::zeros(&[("w", vec![10])]);
        let mut client = FederatedClient::new("test", model);

        let mut new_model = ModelWeights::zeros(&[("w", vec![10])]);
        new_model.version = 5;

        client.sync_model(&new_model).unwrap();
        assert_eq!(client.current_round(), 5);
    }

    #[test]
    fn test_client_sync_dimension_mismatch() {
        let model = ModelWeights::zeros(&[("w", vec![10])]);
        let mut client = FederatedClient::new("test", model);

        let new_model = ModelWeights::zeros(&[("w", vec![20])]);
        assert!(client.sync_model(&new_model).is_err());
    }

    #[test]
    fn test_local_dataset() {
        let dataset = LocalDataset::new(100, 10);
        assert_eq!(dataset.num_samples, 100);
        assert_eq!(dataset.feature_dim, 10);
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();
        history.record_round(0, 1.0, 100);
        history.record_round(1, 0.5, 100);
        history.record_round(2, 0.25, 100);

        assert_eq!(history.rounds.len(), 3);
        assert!((history.average_loss() - 0.583).abs() < 0.01);
        assert_eq!(history.total_samples(), 300);
    }

    #[test]
    fn test_client_train_round() {
        let model = ModelWeights::zeros(&[("w", vec![10])]);
        let config = FedConfig::new().with_local_training(1, 32);
        let mut client = FederatedClient::with_config("test", model, config);

        let data = LocalDataset::new(100, 10);
        let update = client.train_round(&data).unwrap();

        assert_eq!(update.client_id, "test");
        assert_eq!(update.num_samples, 100);
        assert!(update.loss.is_some());
    }
}
