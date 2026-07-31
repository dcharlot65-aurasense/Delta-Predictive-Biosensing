//! Configuration for federated learning.

use serde::{Deserialize, Serialize};
use crate::privacy::PrivacyConfig;

/// Configuration for federated learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FedConfig {
    /// Number of training rounds.
    pub num_rounds: u64,
    /// Minimum number of clients required per round.
    pub min_clients: usize,
    /// Maximum number of clients per round (for sampling).
    pub max_clients: Option<usize>,
    /// Fraction of clients to sample each round (0.0-1.0).
    pub client_fraction: f64,
    /// Local epochs per round.
    pub local_epochs: usize,
    /// Local batch size.
    pub batch_size: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Momentum for SGD.
    pub momentum: f64,
    /// Weight decay (L2 regularization).
    pub weight_decay: f64,
    /// Privacy configuration (optional).
    pub privacy: Option<PrivacyConfig>,
    /// Enable gradient compression.
    pub compression_enabled: bool,
    /// Compression ratio (0.0-1.0, lower = more compression).
    pub compression_ratio: f64,
    /// Timeout for client responses (seconds).
    pub client_timeout_sec: u64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Aggregation strategy.
    pub aggregation: AggregationStrategy,
}

impl Default for FedConfig {
    fn default() -> Self {
        Self {
            num_rounds: 100,
            min_clients: 2,
            max_clients: None,
            client_fraction: 1.0,
            local_epochs: 1,
            batch_size: 32,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 1e-4,
            privacy: None,
            compression_enabled: false,
            compression_ratio: 0.1,
            client_timeout_sec: 300,
            seed: None,
            aggregation: AggregationStrategy::FedAvg,
        }
    }
}

impl FedConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of training rounds.
    pub fn with_rounds(mut self, rounds: u64) -> Self {
        self.num_rounds = rounds;
        self
    }

    /// Set the minimum number of clients.
    pub fn with_min_clients(mut self, min: usize) -> Self {
        self.min_clients = min;
        self
    }

    /// Set the client sampling fraction.
    pub fn with_client_fraction(mut self, fraction: f64) -> Self {
        self.client_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set local training parameters.
    pub fn with_local_training(mut self, epochs: usize, batch_size: usize) -> Self {
        self.local_epochs = epochs;
        self.batch_size = batch_size;
        self
    }

    /// Set the learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Enable differential privacy.
    pub fn with_privacy(mut self, config: PrivacyConfig) -> Self {
        self.privacy = Some(config);
        self
    }

    /// Enable gradient compression.
    pub fn with_compression(mut self, ratio: f64) -> Self {
        self.compression_enabled = true;
        self.compression_ratio = ratio.clamp(0.01, 1.0);
        self
    }


    /// Set the aggregation strategy.
    pub fn with_aggregation(mut self, strategy: AggregationStrategy) -> Self {
        self.aggregation = strategy;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::Result<()> {
        if self.num_rounds == 0 {
            return Err(crate::FederatedError::ConfigError(
                "num_rounds must be positive".into(),
            ));
        }

        if self.min_clients == 0 {
            return Err(crate::FederatedError::ConfigError(
                "min_clients must be positive".into(),
            ));
        }

        if self.learning_rate <= 0.0 {
            return Err(crate::FederatedError::ConfigError(
                "learning_rate must be positive".into(),
            ));
        }

        if let Some(ref privacy) = self.privacy {
            privacy.validate()?;
        }

        Ok(())
    }
}

/// Aggregation strategy for combining client updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Standard Federated Averaging.
    FedAvg,
    /// Weighted average by number of samples.
    WeightedAvg,
    /// Median aggregation (robust to outliers).
    Median,
    /// Trimmed mean (removes extremes).
    TrimmedMean,
    /// Coordinate-wise median.
    CoordinateMedian,
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::FedAvg
    }
}

/// Client selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClientSelection {
    /// Select all available clients.
    All,
    /// Random sampling.
    Random,
    /// Round-robin selection.
    RoundRobin,
    /// Priority-based (by data quality/quantity).
    Priority,
}

impl Default for ClientSelection {
    fn default() -> Self {
        Self::All
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FedConfig::default();
        assert_eq!(config.num_rounds, 100);
        assert_eq!(config.min_clients, 2);
        assert_eq!(config.local_epochs, 1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = FedConfig::new()
            .with_rounds(50)
            .with_min_clients(5)
            .with_learning_rate(0.001)
            .with_client_fraction(0.5)
            .with_compression(0.1);

        assert_eq!(config.num_rounds, 50);
        assert_eq!(config.min_clients, 5);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.client_fraction, 0.5);
        assert!(config.compression_enabled);
    }

    #[test]
    fn test_config_validation_zero_rounds() {
        let config = FedConfig::new().with_rounds(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_clients() {
        let config = FedConfig::new().with_min_clients(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_client_fraction_clamped() {
        let config = FedConfig::new().with_client_fraction(1.5);
        assert_eq!(config.client_fraction, 1.0);

        let config = FedConfig::new().with_client_fraction(-0.5);
        assert_eq!(config.client_fraction, 0.0);
    }

    #[test]
    fn test_aggregation_strategies() {
        assert_eq!(AggregationStrategy::default(), AggregationStrategy::FedAvg);

        let strategies = [
            AggregationStrategy::FedAvg,
            AggregationStrategy::WeightedAvg,
            AggregationStrategy::Median,
            AggregationStrategy::TrimmedMean,
            AggregationStrategy::CoordinateMedian,
        ];

        for strategy in &strategies {
            let config = FedConfig::new().with_aggregation(*strategy);
            assert_eq!(config.aggregation, *strategy);
        }
    }

    #[test]
    fn test_config_serialization() {
        let config = FedConfig::new()
            .with_rounds(25)
            .with_min_clients(3);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: FedConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.num_rounds, config.num_rounds);
        assert_eq!(deserialized.min_clients, config.min_clients);
    }
}
