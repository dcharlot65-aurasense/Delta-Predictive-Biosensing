//! Optimizers for SNN training

use crate::SNNResult;
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Base trait for optimizers
pub trait Optimizer: Send + Sync {
    /// Update parameters given gradients
    fn step(&mut self, params: &mut [&mut Array2<f32>], grads: &[&Array2<f32>]) -> SNNResult<()>;

    /// Zero gradients
    fn zero_grad(&mut self);

    /// Get learning rate
    fn learning_rate(&self) -> f32;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f32);
}

/// Stochastic Gradient Descent with momentum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGDOptimizer {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum coefficient
    pub momentum: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Momentum buffers (velocity)
    #[serde(skip)]
    pub velocity: Vec<Array2<f32>>,
}

impl SGDOptimizer {
    pub fn new(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocity: Vec::new(),
        }
    }

    fn ensure_velocity(&mut self, params: &[&mut Array2<f32>]) {
        if self.velocity.len() != params.len() {
            self.velocity = params.iter().map(|p| Array2::zeros(p.raw_dim())).collect();
        }
    }
}

impl Optimizer for SGDOptimizer {
    fn step(&mut self, params: &mut [&mut Array2<f32>], grads: &[&Array2<f32>]) -> SNNResult<()> {
        self.ensure_velocity(params);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Add weight decay to gradient
            let mut effective_grad = (*grad).clone();
            if self.weight_decay > 0.0 {
                let param_ref: &Array2<f32> = param;
                effective_grad += &(param_ref * self.weight_decay);
            }

            // Update velocity: v = momentum * v - lr * grad
            self.velocity[i] = &self.velocity[i] * self.momentum - &effective_grad * self.learning_rate;

            // Update parameters: param = param + velocity
            **param += &self.velocity[i];
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are typically stored in the layers, not the optimizer
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// Adam optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamOptimizer {
    /// Learning rate
    pub learning_rate: f32,
    /// Beta1 (exponential decay rate for first moment)
    pub beta1: f32,
    /// Beta2 (exponential decay rate for second moment)
    pub beta2: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// First moment estimates (mean)
    #[serde(skip)]
    pub m: Vec<Array2<f32>>,
    /// Second moment estimates (variance)
    #[serde(skip)]
    pub v: Vec<Array2<f32>>,
    /// Time step
    pub t: usize,
}

impl AdamOptimizer {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    pub fn default_config(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 0.0001)
    }

    fn ensure_moments(&mut self, params: &[&mut Array2<f32>]) {
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| Array2::zeros(p.raw_dim())).collect();
            self.v = params.iter().map(|p| Array2::zeros(p.raw_dim())).collect();
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, params: &mut [&mut Array2<f32>], grads: &[&Array2<f32>]) -> SNNResult<()> {
        self.ensure_moments(params);
        self.t += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Add weight decay to gradient
            let mut effective_grad = (*grad).clone();
            if self.weight_decay > 0.0 {
                let param_ref: &Array2<f32> = param;
                effective_grad += &(param_ref * self.weight_decay);
            }

            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            self.m[i] = &self.m[i] * self.beta1 + &effective_grad * (1.0 - self.beta1);

            // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            let grad_squared = effective_grad.mapv(|x| x * x);
            self.v[i] = &self.v[i] * self.beta2 + &grad_squared * (1.0 - self.beta2);

            // Compute bias-corrected moment estimates
            let m_hat = &self.m[i] / bias_correction1;
            let v_hat = &self.v[i] / bias_correction2;

            // Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
            let update = m_hat.mapv(|m_val| m_val * self.learning_rate)
                / v_hat.mapv(|v_val| v_val.sqrt() + self.epsilon);

            **param -= &update;
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        // Gradients are stored in layers
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub enum LRScheduler {
    /// Constant learning rate
    Constant,
    /// Step decay: multiply by gamma every step_size epochs
    StepDecay { step_size: usize, gamma: f32 },
    /// Exponential decay: lr = lr_0 * gamma^epoch
    ExponentialDecay { gamma: f32 },
    /// Cosine annealing
    CosineAnnealing { t_max: usize, eta_min: f32 },
}

impl LRScheduler {
    pub fn step(&self, initial_lr: f32, epoch: usize) -> f32 {
        match self {
            LRScheduler::Constant => initial_lr,
            LRScheduler::StepDecay { step_size, gamma } => {
                initial_lr * gamma.powi((epoch / step_size) as i32)
            }
            LRScheduler::ExponentialDecay { gamma } => initial_lr * gamma.powi(epoch as i32),
            LRScheduler::CosineAnnealing { t_max, eta_min } => {
                let cos_term = (1.0 + (std::f32::consts::PI * epoch as f32 / *t_max as f32).cos()) / 2.0;
                eta_min + (initial_lr - eta_min) * cos_term
            }
        }
    }
}

/// Gradient clipping utility
pub fn clip_gradients(grads: &mut [&mut Array2<f32>], max_norm: f32) {
    // Compute total gradient norm
    let mut total_norm_sq = 0.0;
    for grad in grads.iter() {
        total_norm_sq += grad.iter().map(|&g| g * g).sum::<f32>();
    }
    let total_norm = total_norm_sq.sqrt();

    // Clip if necessary
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-8);
        for grad in grads {
            **grad *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sgd_optimizer() {
        let mut optimizer = SGDOptimizer::new(0.01, 0.9, 0.0);

        let mut param = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grad = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 4.0]).unwrap();

        // Capture initial value before mutable borrow
        let initial_value = param[[0, 0]];

        let mut params = vec![&mut param];
        let grads = vec![&grad];

        optimizer.step(&mut params, &grads).unwrap();

        // Parameter should have changed
        assert_ne!(params[0][[0, 0]], initial_value);
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = AdamOptimizer::default_config(0.001);

        let mut param = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grad = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        // Capture initial value before mutable borrow
        let initial_value = param[[0, 0]];

        let mut params = vec![&mut param];
        let grads = vec![&grad];

        optimizer.step(&mut params, &grads).unwrap();

        assert_ne!(params[0][[0, 0]], initial_value);
        assert_eq!(optimizer.t, 1);
    }

    #[test]
    fn test_lr_scheduler_step_decay() {
        let scheduler = LRScheduler::StepDecay {
            step_size: 10,
            gamma: 0.1,
        };

        let lr0 = scheduler.step(1.0, 0);
        let lr10 = scheduler.step(1.0, 10);
        let lr20 = scheduler.step(1.0, 20);

        assert!((lr0 - 1.0).abs() < 1e-6);
        assert!((lr10 - 0.1).abs() < 1e-6);
        assert!((lr20 - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut grad1 = Array2::from_shape_vec((2, 2), vec![10.0, 10.0, 10.0, 10.0]).unwrap();
        let mut grad2 = Array2::from_shape_vec((2, 2), vec![10.0, 10.0, 10.0, 10.0]).unwrap();

        let mut grads = vec![&mut grad1, &mut grad2];

        clip_gradients(&mut grads, 1.0);

        // Gradients should be scaled down
        assert!(grad1[[0, 0]] < 10.0);
        assert!(grad2[[0, 0]] < 10.0);
    }
}
