//! Self-Distillation and Progressive Compression
//!
//! This module implements self-distillation techniques where a model distills
//! knowledge to itself, and progressive distillation for iterative compression.

use crate::{SNNError, SNNResult};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Self-distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfDistillationConfig {
    /// Temperature for self-distillation
    pub temperature: f32,
    /// Number of distillation epochs
    pub num_epochs: usize,
    /// Weight for soft targets
    pub soft_weight: f32,
    /// Weight for hard targets
    pub hard_weight: f32,
    /// Use ensemble of previous checkpoints
    pub use_ensemble: bool,
    /// Number of checkpoints for ensemble
    pub ensemble_size: usize,
}

impl Default for SelfDistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 3.0,
            num_epochs: 10,
            soft_weight: 0.5,
            hard_weight: 0.5,
            use_ensemble: false,
            ensemble_size: 3,
        }
    }
}

impl SelfDistillationConfig {
    /// Validate configuration
    pub fn validate(&self) -> SNNResult<()> {
        if self.temperature <= 0.0 {
            return Err(SNNError::InvalidConfig(
                "Temperature must be positive".to_string(),
            ));
        }
        if (self.soft_weight + self.hard_weight - 1.0).abs() > 1e-5 {
            return Err(SNNError::InvalidConfig(
                "Soft and hard weights should sum to 1.0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Self-distillation: model distills to itself
pub struct SelfDistillation {
    /// Configuration
    pub config: SelfDistillationConfig,
    /// Model checkpoints for ensemble
    checkpoints: Vec<ModelCheckpoint>,
    /// Training epoch
    current_epoch: usize,
}

/// Model checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelCheckpoint {
    /// Epoch number
    epoch: usize,
    /// Model state (placeholder - in real implementation would store actual weights)
    state_hash: u64,
    /// Validation accuracy
    accuracy: f32,
}

impl SelfDistillation {
    /// Create a new self-distillation instance
    pub fn new(config: SelfDistillationConfig) -> SNNResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            checkpoints: Vec::new(),
            current_epoch: 0,
        })
    }

    /// Add a model checkpoint
    pub fn add_checkpoint(&mut self, accuracy: f32) {
        let checkpoint = ModelCheckpoint {
            epoch: self.current_epoch,
            state_hash: self.compute_state_hash(),
            accuracy,
        };

        self.checkpoints.push(checkpoint);

        // Keep only the most recent checkpoints
        if self.checkpoints.len() > self.config.ensemble_size {
            self.checkpoints.remove(0);
        }

        self.current_epoch += 1;
    }

    /// Compute hash of model state (placeholder)
    fn compute_state_hash(&self) -> u64 {
        // In real implementation, would hash actual model weights
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.current_epoch.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute self-distillation loss
    pub fn compute_loss(
        &self,
        current_output: &Array2<f32>,
        target_labels: &Array1<usize>,
        previous_outputs: &[Array2<f32>],
    ) -> SNNResult<f32> {
        let _batch_size = current_output.shape()[0];
        let _num_classes = current_output.shape()[1];

        // Hard target loss (cross-entropy)
        let hard_loss = self.compute_hard_loss(current_output, target_labels)?;

        // Soft target loss (from previous checkpoints or self)
        let soft_loss = if !previous_outputs.is_empty() {
            self.compute_soft_loss(current_output, previous_outputs)?
        } else {
            // Self-distillation: use current output as soft target
            self.compute_self_soft_loss(current_output)?
        };

        Ok(self.config.soft_weight * soft_loss + self.config.hard_weight * hard_loss)
    }

    /// Compute hard target loss
    fn compute_hard_loss(&self, output: &Array2<f32>, labels: &Array1<usize>) -> SNNResult<f32> {
        let batch_size = output.shape()[0];
        let mut loss = 0.0;
        let eps = 1e-8;

        for b in 0..batch_size {
            if b >= labels.len() {
                break;
            }

            let label = labels[b];
            if label >= output.shape()[1] {
                return Err(SNNError::InvalidConfig(format!(
                    "Label {} out of range",
                    label
                )));
            }

            // Softmax
            let row = output.row(b);
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let prob = (row[label] - max_val).exp() / exp_sum;

            loss -= prob.max(eps).ln();
        }

        Ok(loss / batch_size as f32)
    }

    /// Compute soft target loss from previous outputs
    fn compute_soft_loss(
        &self,
        current_output: &Array2<f32>,
        previous_outputs: &[Array2<f32>],
    ) -> SNNResult<f32> {
        let batch_size = current_output.shape()[0];
        let num_classes = current_output.shape()[1];

        // Average previous outputs (ensemble)
        let mut avg_output = Array2::zeros((batch_size, num_classes));
        for prev_output in previous_outputs {
            avg_output += prev_output;
        }
        avg_output /= previous_outputs.len() as f32;

        // KL divergence with temperature scaling
        let current_probs = self.temperature_softmax(current_output);
        let target_probs = self.temperature_softmax(&avg_output);

        self.kl_divergence(&target_probs, &current_probs)
    }

    /// Compute self soft loss (regularization)
    fn compute_self_soft_loss(&self, output: &Array2<f32>) -> SNNResult<f32> {
        // Encourage confident predictions (entropy minimization)
        let probs = self.temperature_softmax(output);
        let batch_size = probs.shape()[0];
        let mut entropy = 0.0;
        let eps = 1e-8;

        for b in 0..batch_size {
            for p in probs.row(b).iter() {
                let p_safe = p.max(eps);
                entropy -= p_safe * p_safe.ln();
            }
        }

        Ok(entropy / batch_size as f32)
    }

    /// Temperature-scaled softmax
    fn temperature_softmax(&self, logits: &Array2<f32>) -> Array2<f32> {
        let (batch_size, num_classes) = (logits.shape()[0], logits.shape()[1]);
        let mut softmax = Array2::zeros((batch_size, num_classes));

        for b in 0..batch_size {
            let scaled = logits.row(b).mapv(|x| x / self.config.temperature);
            let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_vals.iter().sum();

            for c in 0..num_classes {
                softmax[[b, c]] = exp_vals[c] / sum_exp;
            }
        }

        softmax
    }

    /// KL divergence
    fn kl_divergence(&self, p: &Array2<f32>, q: &Array2<f32>) -> SNNResult<f32> {
        let batch_size = p.shape()[0];
        let num_classes = p.shape()[1];
        let mut kl = 0.0;
        let eps = 1e-8;

        for b in 0..batch_size {
            for c in 0..num_classes {
                let p_val = p[[b, c]].max(eps);
                let q_val = q[[b, c]].max(eps);
                kl += p_val * (p_val / q_val).ln();
            }
        }

        Ok(kl * self.config.temperature * self.config.temperature / batch_size as f32)
    }

    /// Get number of checkpoints
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }
}

/// Born-Again Networks: train multiple generations
pub struct BornAgainNetworks {
    /// Number of generations
    pub num_generations: usize,
    /// Improvement threshold to continue
    pub improvement_threshold: f32,
    /// Generation history
    generations: Vec<GenerationInfo>,
}

/// Information about a generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationInfo {
    generation: usize,
    accuracy: f32,
    model_size: usize,
}

impl BornAgainNetworks {
    pub fn new(num_generations: usize) -> Self {
        Self {
            num_generations,
            improvement_threshold: 0.001,
            generations: Vec::new(),
        }
    }

    /// Record a generation
    pub fn record_generation(&mut self, generation: usize, accuracy: f32, model_size: usize) {
        self.generations.push(GenerationInfo {
            generation,
            accuracy,
            model_size,
        });
    }

    /// Check if should continue training
    pub fn should_continue(&self) -> bool {
        if self.generations.len() < 2 {
            return true;
        }

        let last = &self.generations[self.generations.len() - 1];
        let prev = &self.generations[self.generations.len() - 2];

        let improvement = last.accuracy - prev.accuracy;
        improvement > self.improvement_threshold
    }

    /// Get best generation
    pub fn best_generation(&self) -> Option<&GenerationInfo> {
        self.generations
            .iter()
            .max_by(|a, b| a.accuracy.total_cmp(&b.accuracy))
    }
}

/// Progressive distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveConfig {
    /// Number of stages
    pub num_stages: usize,
    /// Compression ratio per stage
    pub stage_compression: Vec<f32>,
    /// Epochs per stage
    pub epochs_per_stage: Vec<usize>,
    /// Temperature schedule
    pub temperature_schedule: Vec<f32>,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            num_stages: 3,
            stage_compression: vec![0.7, 0.5, 0.3],
            epochs_per_stage: vec![10, 10, 10],
            temperature_schedule: vec![4.0, 3.0, 2.0],
        }
    }
}

impl ProgressiveConfig {
    /// Validate configuration
    pub fn validate(&self) -> SNNResult<()> {
        if self.num_stages == 0 {
            return Err(SNNError::InvalidConfig(
                "Number of stages must be positive".to_string(),
            ));
        }

        if self.stage_compression.len() != self.num_stages {
            return Err(SNNError::InvalidConfig(
                "Stage compression ratios must match number of stages".to_string(),
            ));
        }

        if self.epochs_per_stage.len() != self.num_stages {
            return Err(SNNError::InvalidConfig(
                "Epochs per stage must match number of stages".to_string(),
            ));
        }

        if self.temperature_schedule.len() != self.num_stages {
            return Err(SNNError::InvalidConfig(
                "Temperature schedule must match number of stages".to_string(),
            ));
        }

        // Check that compression ratios are decreasing
        for i in 1..self.stage_compression.len() {
            if self.stage_compression[i] >= self.stage_compression[i - 1] {
                return Err(SNNError::InvalidConfig(
                    "Stage compression ratios should be decreasing".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Progressive distillation: iterative compression
pub struct ProgressiveDistillation {
    /// Configuration
    pub config: ProgressiveConfig,
    /// Current stage
    current_stage: usize,
    /// Stage history
    stages: Vec<StageInfo>,
}

/// Information about a compression stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageInfo {
    stage: usize,
    compression_ratio: f32,
    accuracy: f32,
    model_size: usize,
}

impl ProgressiveDistillation {
    /// Create a new progressive distillation
    pub fn new(config: ProgressiveConfig) -> SNNResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            current_stage: 0,
            stages: Vec::new(),
        })
    }

    /// Get the current stage configuration, or `None` once every stage has run.
    pub fn try_current_stage_config(&self) -> Option<StageConfig> {
        if self.is_complete() {
            return None;
        }
        Some(self.current_stage_config())
    }

    /// Get current stage configuration
    ///
    /// # Panics
    ///
    /// Panics once [`is_complete`](Self::is_complete) is true, since there is no
    /// current stage to describe. Use
    /// [`try_current_stage_config`](Self::try_current_stage_config) when the
    /// schedule may already have finished.
    pub fn current_stage_config(&self) -> StageConfig {
        let idx = self.current_stage;
        StageConfig {
            stage: self.current_stage,
            compression_ratio: self.config.stage_compression[idx],
            num_epochs: self.config.epochs_per_stage[idx],
            temperature: self.config.temperature_schedule[idx],
        }
    }

    /// Advance to next stage
    pub fn advance_stage(&mut self, accuracy: f32, model_size: usize) -> bool {
        // Indexing `stage_compression[current_stage]` below panics once every
        // stage has been run, so an extra call would abort rather than simply
        // report that there is nothing left to do.
        if self.is_complete() {
            return false;
        }

        self.stages.push(StageInfo {
            stage: self.current_stage,
            compression_ratio: self.config.stage_compression[self.current_stage],
            accuracy,
            model_size,
        });

        self.current_stage += 1;
        self.current_stage < self.config.num_stages
    }

    /// Get number of completed stages
    pub fn completed_stages(&self) -> usize {
        self.stages.len()
    }

    /// Get stage history
    pub fn stage_history(&self) -> &[StageInfo] {
        &self.stages
    }

    /// Check if final stage
    /// Whether the stage now in progress is the last one.
    ///
    /// Distinct from [`is_complete`](Self::is_complete): being ON the final
    /// stage is not the same as having finished it. The previous
    /// `current_stage >= num_stages - 1` was true in both states, so callers
    /// could not tell "one stage left to run" from "nothing left to run" --
    /// and it underflowed for a zero-stage config.
    pub fn is_final_stage(&self) -> bool {
        self.current_stage + 1 == self.config.num_stages
    }

    /// Whether every stage has been run.
    ///
    /// The complement of [`advance_stage`](Self::advance_stage)'s return value.
    pub fn is_complete(&self) -> bool {
        self.current_stage >= self.config.num_stages
    }
}

/// Configuration for a compression stage
#[derive(Debug, Clone)]
pub struct StageConfig {
    pub stage: usize,
    pub compression_ratio: f32,
    pub num_epochs: usize,
    pub temperature: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_distillation_config() {
        let config = SelfDistillationConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid = config.clone();
        invalid.temperature = -1.0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_self_distillation() {
        let config = SelfDistillationConfig::default();
        let mut distiller = SelfDistillation::new(config).unwrap();

        assert_eq!(distiller.num_checkpoints(), 0);

        distiller.add_checkpoint(0.85);
        assert_eq!(distiller.num_checkpoints(), 1);

        distiller.add_checkpoint(0.87);
        assert_eq!(distiller.num_checkpoints(), 2);
    }

    #[test]
    fn test_self_distillation_loss() {
        let config = SelfDistillationConfig::default();
        let distiller = SelfDistillation::new(config).unwrap();

        let output = Array2::from_shape_vec((2, 3), vec![2.0, 1.0, 0.5, 0.5, 2.0, 1.0]).unwrap();

        let labels = Array1::from_vec(vec![0, 1]);

        let loss = distiller.compute_loss(&output, &labels, &[]).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_self_distillation_with_previous() {
        let config = SelfDistillationConfig::default();
        let distiller = SelfDistillation::new(config).unwrap();

        let current = Array2::from_shape_vec((2, 3), vec![2.0, 1.0, 0.5, 0.5, 2.0, 1.0]).unwrap();

        let previous = Array2::from_shape_vec((2, 3), vec![1.8, 1.2, 0.6, 0.6, 1.9, 1.1]).unwrap();

        let labels = Array1::from_vec(vec![0, 1]);

        let loss = distiller
            .compute_loss(&current, &labels, &[previous])
            .unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_born_again_networks() {
        let mut ban = BornAgainNetworks::new(5);

        ban.record_generation(0, 0.85, 10000);
        ban.record_generation(1, 0.87, 10000);
        ban.record_generation(2, 0.88, 10000);

        assert!(ban.should_continue());

        let best = ban.best_generation().unwrap();
        assert_eq!(best.generation, 2);
        assert_eq!(best.accuracy, 0.88);
    }

    #[test]
    fn test_born_again_no_improvement() {
        let mut ban = BornAgainNetworks::new(5);
        ban.improvement_threshold = 0.01;

        ban.record_generation(0, 0.85, 10000);
        ban.record_generation(1, 0.851, 10000); // < 1% improvement

        assert!(!ban.should_continue());
    }

    #[test]
    fn test_progressive_config() {
        let config = ProgressiveConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_progressive_config_invalid() {
        let config = ProgressiveConfig {
            stage_compression: vec![0.7, 0.8, 0.5], // Not decreasing
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_progressive_distillation() {
        let config = ProgressiveConfig {
            num_stages: 3,
            stage_compression: vec![0.7, 0.5, 0.3],
            epochs_per_stage: vec![10, 10, 10],
            temperature_schedule: vec![4.0, 3.0, 2.0],
        };

        let mut progressive = ProgressiveDistillation::new(config).unwrap();

        assert_eq!(progressive.completed_stages(), 0);
        assert!(!progressive.is_final_stage());

        let stage_config = progressive.current_stage_config();
        assert_eq!(stage_config.stage, 0);
        assert_eq!(stage_config.compression_ratio, 0.7);

        let has_next = progressive.advance_stage(0.85, 7000);
        assert!(has_next);
        assert_eq!(progressive.completed_stages(), 1);
    }

    #[test]
    fn test_progressive_distillation_complete() {
        let config = ProgressiveConfig {
            num_stages: 2,
            stage_compression: vec![0.7, 0.5],
            epochs_per_stage: vec![10, 10],
            temperature_schedule: vec![4.0, 3.0],
        };

        let mut progressive = ProgressiveDistillation::new(config).unwrap();

        progressive.advance_stage(0.85, 7000);
        // One of two stages is done: we are now ON the final stage, but the
        // schedule is not finished.
        assert!(progressive.is_final_stage());
        assert!(!progressive.is_complete());

        let has_next = progressive.advance_stage(0.83, 5000);
        assert!(!has_next); // No more stages
        assert!(progressive.is_complete());
        assert!(progressive.try_current_stage_config().is_none());

        // Advancing again is a no-op, not a panic, and records no extra stage.
        let completed = progressive.completed_stages();
        assert!(!progressive.advance_stage(0.80, 4000));
        assert_eq!(progressive.completed_stages(), completed);
    }

    #[test]
    fn test_temperature_softmax() {
        let config = SelfDistillationConfig::default();
        let distiller = SelfDistillation::new(config).unwrap();

        let logits = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let softmax = distiller.temperature_softmax(&logits);

        // Check that probabilities sum to 1
        for b in 0..2 {
            let sum: f32 = softmax.row(b).sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_checkpoint_limit() {
        let config = SelfDistillationConfig {
            ensemble_size: 3,
            ..Default::default()
        };

        let mut distiller = SelfDistillation::new(config).unwrap();

        for i in 0..5 {
            distiller.add_checkpoint(0.8 + i as f32 * 0.01);
        }

        // Should keep only last 3 checkpoints
        assert_eq!(distiller.num_checkpoints(), 3);
    }
}
