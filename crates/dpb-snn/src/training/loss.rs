//! Loss functions for SNN training

use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{Array1, Array2, Array3, Axis, s};
use serde::{Deserialize, Serialize};

/// Base trait for loss functions
pub trait LossFunction: Send + Sync {
    /// Compute loss
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32>;

    /// Compute gradient with respect to predictions
    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>>;
}

/// Cross-entropy loss based on spike rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingCrossEntropy {
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl SpikingCrossEntropy {
    pub fn new() -> Self {
        Self { eps: 1e-8 }
    }
}

impl Default for SpikingCrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFunction for SpikingCrossEntropy {
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        // Compute spike rates for each neuron
        let rates = predictions.spike_rate();
        let (batch_size, num_neurons) = (rates.shape()[0], rates.shape()[1]);

        if targets.shape() != [batch_size, num_neurons] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("({}, {})", batch_size, num_neurons),
                actual: format!("{:?}", targets.shape()),
            });
        }

        // Normalize rates to probabilities (softmax)
        let mut loss = 0.0;
        for b in 0..batch_size {
            let rate_sum = rates.row(b).sum();
            if rate_sum > 0.0 {
                for n in 0..num_neurons {
                    let prob = (rates[[b, n]] / rate_sum).max(self.eps);
                    loss -= targets[[b, n]] * prob.ln();
                }
            }
        }

        Ok(loss / batch_size as f32)
    }

    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let rates = predictions.spike_rate();
        let mut gradient = Array3::zeros((batch_size, num_steps, num_neurons));

        // Compute gradient with respect to each spike
        for b in 0..batch_size {
            let rate_sum = rates.row(b).sum().max(self.eps);

            for t in 0..num_steps {
                for n in 0..num_neurons {
                    let prob = (rates[[b, n]] / rate_sum).max(self.eps);
                    gradient[[b, t, n]] = (prob - targets[[b, n]]) / num_steps as f32;
                }
            }
        }

        Ok(gradient)
    }
}

/// Loss based on total spike count
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeCountLoss {
    /// Target spike count per neuron
    pub target_count: f32,
}

impl SpikeCountLoss {
    pub fn new(target_count: f32) -> Self {
        Self { target_count }
    }
}

impl LossFunction for SpikeCountLoss {
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut loss = 0.0;

        for b in 0..batch_size {
            for n in 0..num_neurons {
                // Count spikes for this neuron
                let spike_count: f32 = pred_dense.slice(s![b, .., n]).sum();

                // Target spike count weighted by class probability
                let target_spikes = targets[[b, n]] * self.target_count * num_steps as f32;

                // MSE loss
                loss += (spike_count - target_spikes).powi(2);
            }
        }

        Ok(loss / (batch_size * num_neurons) as f32)
    }

    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut gradient = Array3::zeros((batch_size, num_steps, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                let spike_count: f32 = pred_dense.slice(s![b, .., n]).sum();
                let target_spikes = targets[[b, n]] * self.target_count * num_steps as f32;
                let grad_coef = 2.0 * (spike_count - target_spikes) / (batch_size * num_neurons) as f32;

                for t in 0..num_steps {
                    gradient[[b, t, n]] = grad_coef;
                }
            }
        }

        Ok(gradient)
    }
}

/// Loss based on spike timing (first spike latency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeTimingLoss {
    /// Weight for timing vs count
    pub timing_weight: f32,
}

impl SpikeTimingLoss {
    pub fn new(timing_weight: f32) -> Self {
        Self { timing_weight }
    }
}

impl LossFunction for SpikeTimingLoss {
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut loss = 0.0;

        for b in 0..batch_size {
            for n in 0..num_neurons {
                // Find first spike time
                let mut first_spike_time = num_steps as f32;
                for t in 0..num_steps {
                    if pred_dense[[b, t, n]] > 0.5 {
                        first_spike_time = t as f32;
                        break;
                    }
                }

                // Target: earlier spikes for higher target values
                let desired_time = (1.0 - targets[[b, n]]) * num_steps as f32;

                // Penalize deviation from desired timing
                loss += (first_spike_time - desired_time).powi(2);
            }
        }

        Ok(self.timing_weight * loss / (batch_size * num_neurons) as f32)
    }

    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut gradient = Array3::zeros((batch_size, num_steps, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                // Find first spike
                for t in 0..num_steps {
                    if pred_dense[[b, t, n]] > 0.5 {
                        let desired_time = (1.0 - targets[[b, n]]) * num_steps as f32;
                        let error = t as f32 - desired_time;
                        gradient[[b, t, n]] = 2.0 * self.timing_weight * error / (batch_size * num_neurons) as f32;
                        break;
                    }
                }
            }
        }

        Ok(gradient)
    }
}

/// Temporal cross-entropy loss (considers temporal dynamics)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCrossEntropy {
    /// Temporal kernel for weighting
    pub temporal_kernel: Array1<f32>,
    /// Epsilon
    pub eps: f32,
}

impl TemporalCrossEntropy {
    pub fn new(num_steps: usize) -> Self {
        // Create exponential decay kernel (later spikes weighted less)
        let mut kernel = Array1::zeros(num_steps);
        for t in 0..num_steps {
            kernel[t] = (-0.05 * t as f32).exp();
        }
        let kernel_sum = kernel.sum();
        kernel /= kernel_sum; // Normalize

        Self {
            temporal_kernel: kernel,
            eps: 1e-8,
        }
    }
}

impl LossFunction for TemporalCrossEntropy {
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        // Compute temporally-weighted spike rates
        let mut weighted_rates: Array2<f32> = Array2::zeros((batch_size, num_neurons));
        for b in 0..batch_size {
            for n in 0..num_neurons {
                for t in 0..num_steps.min(self.temporal_kernel.len()) {
                    weighted_rates[[b, n]] += pred_dense[[b, t, n]] * self.temporal_kernel[t];
                }
            }
        }

        // Compute cross-entropy on weighted rates
        let mut loss = 0.0;
        for b in 0..batch_size {
            let rate_sum = weighted_rates.row(b).sum().max(self.eps);
            for n in 0..num_neurons {
                let prob = (weighted_rates[[b, n]] / rate_sum).max(self.eps);
                loss -= targets[[b, n]] * prob.ln();
            }
        }

        Ok(loss / batch_size as f32)
    }

    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut gradient = Array3::zeros((batch_size, num_steps, num_neurons));

        // Simplified gradient computation
        for b in 0..batch_size {
            for t in 0..num_steps.min(self.temporal_kernel.len()) {
                for n in 0..num_neurons {
                    let weight = self.temporal_kernel[t];
                    // Approximate gradient
                    gradient[[b, t, n]] = weight * (pred_dense[[b, t, n]] - targets[[b, n]]);
                }
            }
        }

        Ok(gradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiking_cross_entropy() {
        let loss_fn = SpikingCrossEntropy::new();

        // Create simple predictions (2 batches, 10 steps, 3 neurons)
        let mut pred_data = Array3::zeros((2, 10, 3));
        pred_data[[0, 5, 0]] = 1.0; // Batch 0, neuron 0 spikes
        pred_data[[1, 3, 1]] = 1.0; // Batch 1, neuron 1 spikes

        let predictions = SpikeTensor::from_dense(pred_data, false);

        // Targets: one-hot encoding
        let targets = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

        let loss = loss_fn.compute(&predictions, &targets).unwrap();
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_spike_count_loss() {
        let loss_fn = SpikeCountLoss::new(0.2); // Target 20% spike rate

        let mut pred_data = Array3::zeros((1, 10, 2));
        pred_data[[0, 3, 0]] = 1.0;
        pred_data[[0, 7, 0]] = 1.0; // 2 spikes in 10 steps

        let predictions = SpikeTensor::from_dense(pred_data, false);
        let targets = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();

        let loss = loss_fn.compute(&predictions, &targets).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_spike_timing_loss() {
        let loss_fn = SpikeTimingLoss::new(1.0);

        let mut pred_data = Array3::zeros((1, 10, 2));
        pred_data[[0, 2, 0]] = 1.0; // Early spike

        let predictions = SpikeTensor::from_dense(pred_data, false);
        let targets = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();

        let loss = loss_fn.compute(&predictions, &targets).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_temporal_cross_entropy() {
        let loss_fn = TemporalCrossEntropy::new(10);

        let mut pred_data = Array3::zeros((1, 10, 2));
        pred_data[[0, 1, 0]] = 1.0;

        let predictions = SpikeTensor::from_dense(pred_data, false);
        let targets = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();

        let loss = loss_fn.compute(&predictions, &targets).unwrap();
        assert!(loss.is_finite());
    }
}
