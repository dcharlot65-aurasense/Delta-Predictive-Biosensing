//! Privacy-preserving mechanisms for federated learning.

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use crate::{FederatedError, Result, model::ModelWeights};

/// Configuration for differential privacy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Privacy budget epsilon (smaller = more private).
    pub epsilon: f64,
    /// Privacy budget delta (probability of privacy breach).
    pub delta: f64,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f32,
    /// Noise multiplier (sigma = multiplier * sensitivity / epsilon).
    pub noise_multiplier: f64,
    /// Target delta for moments accountant.
    pub target_delta: f64,
    /// Mechanism type.
    pub mechanism: PrivacyMechanism,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            max_grad_norm: 1.0,
            noise_multiplier: 1.0,
            target_delta: 1e-5,
            mechanism: PrivacyMechanism::Gaussian,
        }
    }
}

impl PrivacyConfig {
    /// Create a new privacy configuration.
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            ..Default::default()
        }
    }

    /// Set the maximum gradient norm.
    pub fn with_max_grad_norm(mut self, norm: f32) -> Self {
        self.max_grad_norm = norm;
        self
    }

    /// Set the noise multiplier.
    pub fn with_noise_multiplier(mut self, multiplier: f64) -> Self {
        self.noise_multiplier = multiplier;
        self
    }

    /// Set the privacy mechanism.
    pub fn with_mechanism(mut self, mechanism: PrivacyMechanism) -> Self {
        self.mechanism = mechanism;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.epsilon <= 0.0 {
            return Err(FederatedError::ConfigError(
                "epsilon must be positive".into(),
            ));
        }
        if self.delta <= 0.0 || self.delta >= 1.0 {
            return Err(FederatedError::ConfigError(
                "delta must be in (0, 1)".into(),
            ));
        }
        if self.max_grad_norm <= 0.0 {
            return Err(FederatedError::ConfigError(
                "max_grad_norm must be positive".into(),
            ));
        }
        Ok(())
    }

    /// Calculate noise standard deviation.
    pub fn noise_std(&self) -> f64 {
        self.noise_multiplier * (self.max_grad_norm as f64) / self.epsilon
    }
}

/// Privacy mechanism type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyMechanism {
    /// Gaussian (normal) noise mechanism.
    Gaussian,
    /// Laplace noise mechanism.
    Laplace,
    /// Discrete Gaussian for integer-valued data.
    DiscreteGaussian,
}

/// Differential privacy engine.
pub struct DifferentialPrivacy {
    config: PrivacyConfig,
    /// Cumulative epsilon spent.
    epsilon_spent: f64,
    /// Cumulative delta spent.
    delta_spent: f64,
    /// Number of compositions.
    num_compositions: u64,
}

impl DifferentialPrivacy {
    /// Create a new DP engine with the given configuration.
    pub fn new(config: PrivacyConfig) -> Self {
        Self {
            config,
            epsilon_spent: 0.0,
            delta_spent: 0.0,
            num_compositions: 0,
        }
    }

    /// Get remaining privacy budget.
    pub fn remaining_budget(&self) -> (f64, f64) {
        (
            self.config.epsilon - self.epsilon_spent,
            self.config.delta - self.delta_spent,
        )
    }

    /// Check if budget is exhausted.
    pub fn is_budget_exhausted(&self) -> bool {
        self.epsilon_spent >= self.config.epsilon || self.delta_spent >= self.config.delta
    }

    /// Apply differential privacy to model weights.
    pub fn privatize(&mut self, weights: &mut ModelWeights) -> Result<()> {
        if self.is_budget_exhausted() {
            return Err(FederatedError::PrivacyBudgetExhausted {
                epsilon: self.config.epsilon,
                delta: self.config.delta,
            });
        }

        // Clip gradients to bounded sensitivity
        weights.clip_norm(self.config.max_grad_norm);

        // Add noise
        let noise_std = self.config.noise_std() as f32;
        let mut rng = rand::rng();

        for tensor in weights.parameters.values_mut() {
            for value in &mut tensor.data {
                *value += match self.config.mechanism {
                    PrivacyMechanism::Gaussian => sample_gaussian(&mut rng, noise_std),
                    PrivacyMechanism::Laplace => sample_laplace(&mut rng, noise_std),
                    PrivacyMechanism::DiscreteGaussian => {
                        sample_discrete_gaussian(&mut rng, noise_std)
                    }
                };
            }
        }

        // Approximate budget tracking. NOTE: this is NOT a valid composition
        // bound and NOT a certified DP accountant (no Renyi/moments accountant).
        // Do not rely on epsilon_spent for a formal privacy guarantee.
        self.epsilon_spent += self.config.epsilon / self.config.noise_multiplier;
        self.delta_spent += self.config.delta;
        self.num_compositions += 1;

        Ok(())
    }

    /// Apply privacy to a single vector.
    pub fn privatize_vector(&mut self, data: &mut [f32]) -> Result<()> {
        if self.is_budget_exhausted() {
            return Err(FederatedError::PrivacyBudgetExhausted {
                epsilon: self.config.epsilon,
                delta: self.config.delta,
            });
        }

        // Clip to max norm
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > self.config.max_grad_norm {
            let scale = self.config.max_grad_norm / norm;
            for v in data.iter_mut() {
                *v *= scale;
            }
        }

        // Add noise
        let noise_std = self.config.noise_std() as f32;
        let mut rng = rand::rng();

        for value in data.iter_mut() {
            *value += match self.config.mechanism {
                PrivacyMechanism::Gaussian => sample_gaussian(&mut rng, noise_std),
                PrivacyMechanism::Laplace => sample_laplace(&mut rng, noise_std),
                PrivacyMechanism::DiscreteGaussian => sample_discrete_gaussian(&mut rng, noise_std),
            };
        }

        self.epsilon_spent += self.config.epsilon / self.config.noise_multiplier;
        self.delta_spent += self.config.delta;
        self.num_compositions += 1;

        Ok(())
    }

    /// Get the configuration.
    pub fn config(&self) -> &PrivacyConfig {
        &self.config
    }

    /// Get number of compositions performed.
    pub fn num_compositions(&self) -> u64 {
        self.num_compositions
    }
}

/// Sample from Gaussian distribution.
fn sample_gaussian<R: Rng>(rng: &mut R, std: f32) -> f32 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0, std as f64).unwrap();
    normal.sample(rng) as f32
}

/// Sample from Laplace distribution.
fn sample_laplace<R: Rng>(rng: &mut R, scale: f32) -> f32 {
    let u: f64 = rng.random_range(-0.5..0.5);
    let sign = if u >= 0.0 { 1.0 } else { -1.0 };
    (sign * (scale as f64) * (1.0 - 2.0 * u.abs()).ln()) as f32
}

/// Sample from discrete Gaussian (rounded).
fn sample_discrete_gaussian<R: Rng>(rng: &mut R, std: f32) -> f32 {
    sample_gaussian(rng, std).round()
}

/// Privacy accountant for tracking cumulative privacy loss.
#[derive(Debug, Clone)]
pub struct PrivacyAccountant {
    /// Target epsilon.
    target_epsilon: f64,
    /// Target delta.
    target_delta: f64,
    /// History of noise multipliers.
    noise_history: Vec<f64>,
    /// History of sample rates.
    sample_rate_history: Vec<f64>,
}

impl PrivacyAccountant {
    /// Create a new privacy accountant.
    pub fn new(target_epsilon: f64, target_delta: f64) -> Self {
        Self {
            target_epsilon,
            target_delta,
            noise_history: Vec::new(),
            sample_rate_history: Vec::new(),
        }
    }

    /// Record a step of the training.
    pub fn step(&mut self, noise_multiplier: f64, sample_rate: f64) {
        self.noise_history.push(noise_multiplier);
        self.sample_rate_history.push(sample_rate);
    }

    /// Compute current (epsilon, delta) using basic composition.
    pub fn compute_epsilon_basic(&self) -> (f64, f64) {
        let epsilon: f64 = self
            .noise_history
            .iter()
            .zip(&self.sample_rate_history)
            .map(|(sigma, q)| q / sigma)
            .sum();

        let delta = self.target_delta * self.noise_history.len() as f64;

        (epsilon, delta)
    }

    /// Check if target budget would be exceeded.
    pub fn would_exceed_budget(&self, noise_multiplier: f64, sample_rate: f64) -> bool {
        let mut temp = self.clone();
        temp.step(noise_multiplier, sample_rate);
        let (eps, delta) = temp.compute_epsilon_basic();
        eps > self.target_epsilon || delta > self.target_delta
    }

    /// Number of steps recorded.
    pub fn num_steps(&self) -> usize {
        self.noise_history.len()
    }
}

/// Local differential privacy for client-side noise injection.
pub struct LocalDP {
    epsilon: f64,
    mechanism: PrivacyMechanism,
}

impl LocalDP {
    /// Create a new local DP mechanism.
    pub fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            mechanism: PrivacyMechanism::Laplace,
        }
    }

    /// Randomized response for binary data.
    pub fn randomized_response(&self, value: bool) -> bool {
        let mut rng = rand::rng();
        let p = 1.0 / (1.0 + self.epsilon.exp());

        if rng.random::<f64>() < p {
            !value // Flip with probability p
        } else {
            value
        }
    }

    /// Add noise to a numeric value.
    pub fn add_noise(&self, value: f32, sensitivity: f32) -> f32 {
        let scale = sensitivity / self.epsilon as f32;
        let mut rng = rand::rng();
        value + sample_laplace(&mut rng, scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_config_default() {
        let config = PrivacyConfig::default();
        assert_eq!(config.epsilon, 1.0);
        assert_eq!(config.delta, 1e-5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_privacy_config_invalid_epsilon() {
        let config = PrivacyConfig::new(-1.0, 1e-5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_privacy_config_invalid_delta() {
        let config = PrivacyConfig::new(1.0, 1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_differential_privacy_budget() {
        let config = PrivacyConfig::new(1.0, 1e-5);
        let dp = DifferentialPrivacy::new(config);
        assert!(!dp.is_budget_exhausted());

        let (eps, delta) = dp.remaining_budget();
        assert_eq!(eps, 1.0);
        assert_eq!(delta, 1e-5);
    }

    #[test]
    fn test_privatize_vector() {
        let config = PrivacyConfig::new(10.0, 1e-5).with_noise_multiplier(0.1);
        let mut dp = DifferentialPrivacy::new(config);

        let mut data = vec![1.0, 2.0, 3.0];
        let original = data.clone();

        dp.privatize_vector(&mut data).unwrap();

        // Data should be different after noise
        assert!(data.iter().zip(&original).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn test_privacy_accountant() {
        let mut accountant = PrivacyAccountant::new(10.0, 1e-5);

        accountant.step(1.0, 0.01);
        accountant.step(1.0, 0.01);

        let (eps, _) = accountant.compute_epsilon_basic();
        assert!(eps > 0.0);
        assert_eq!(accountant.num_steps(), 2);
    }

    #[test]
    fn test_local_dp_randomized_response() {
        let ldp = LocalDP::new(1.0);

        // Run many trials - should flip some
        let mut flips = 0;
        for _ in 0..1000 {
            if !ldp.randomized_response(true) {
                flips += 1;
            }
        }

        // Should have some flips but not all
        assert!(flips > 50);
        assert!(flips < 950);
    }

    #[test]
    fn test_local_dp_noise() {
        let ldp = LocalDP::new(1.0);
        let value = 5.0f32;

        let mut different = 0;
        for _ in 0..100 {
            if (ldp.add_noise(value, 1.0) - value).abs() > 1e-6 {
                different += 1;
            }
        }

        assert!(different > 90); // Most should be different
    }
}
