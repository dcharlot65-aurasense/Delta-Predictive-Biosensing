//! Training algorithms and utilities for SNNs

pub mod loss;
pub mod optimizer;
pub mod surrogate;
pub mod trainer;

pub use loss::{
    LossFunction, SpikeCountLoss, SpikeTimingLoss, SpikingCrossEntropy, TemporalCrossEntropy,
};
pub use optimizer::{AdamOptimizer, Optimizer, SGDOptimizer};
pub use surrogate::{BPTT, OTTT, SLTT, SurrogateGradient, SurrogateType};
pub use trainer::{StepReport, TrainableLayer, Trainer};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub num_epochs: usize,
    /// Gradient clipping threshold
    pub grad_clip: Option<f32>,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            num_epochs: 100,
            grad_clip: Some(1.0),
            weight_decay: 0.0001,
            early_stopping_patience: Some(10),
        }
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Training losses per epoch
    pub train_losses: Vec<f32>,
    /// Validation losses per epoch
    pub val_losses: Vec<f32>,
    /// Training accuracies per epoch
    pub train_accuracies: Vec<f32>,
    /// Validation accuracies per epoch
    pub val_accuracies: Vec<f32>,
    /// Total training time (seconds)
    pub training_time: f32,
}

impl TrainingStats {
    pub fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
            training_time: 0.0,
        }
    }

    pub fn add_epoch(&mut self, train_loss: f32, val_loss: f32, train_acc: f32, val_acc: f32) {
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
        self.train_accuracies.push(train_acc);
        self.val_accuracies.push(val_acc);
    }

    pub fn best_val_loss(&self) -> Option<f32> {
        self.val_losses.iter().fold(None, |acc, &loss| {
            Some(acc.map_or(loss, |min_loss| min_loss.min(loss)))
        })
    }

    pub fn best_val_accuracy(&self) -> Option<f32> {
        self.val_accuracies.iter().fold(None, |acc, &accuracy| {
            Some(acc.map_or(accuracy, |max_acc| max_acc.max(accuracy)))
        })
    }
}

impl Default for TrainingStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_training_stats() {
        let mut stats = TrainingStats::new();
        stats.add_epoch(0.5, 0.6, 0.8, 0.75);
        stats.add_epoch(0.4, 0.55, 0.85, 0.80);

        assert_eq!(stats.best_val_loss(), Some(0.55));
        assert_eq!(stats.best_val_accuracy(), Some(0.80));
    }
}
