//! Pooling layers for spiking neural networks

use super::SpikingLayer;
use crate::{SNNResult, SpikeTensor};
use ndarray::{Array2, Array3, s};
use serde::{Deserialize, Serialize};

/// 2D Sum Pooling for spike counts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingSumPool2d {
    /// Pool size (height, width)
    pub pool_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// Input shape (for gradient computation)
    #[serde(skip)]
    pub input_shape: Option<(usize, usize, usize, usize)>,
}

impl SpikingSumPool2d {
    /// Create a new sum pooling layer
    pub fn new(pool_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            pool_size,
            stride,
            input_shape: None,
        }
    }

    /// Calculate output dimensions
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = (input_h - self.pool_size.0) / self.stride.0 + 1;
        let out_w = (input_w - self.pool_size.1) / self.stride.1 + 1;
        (out_h, out_w)
    }
}

impl SpikingLayer for SpikingSumPool2d {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, flat_size) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        // For simplicity, assume flat_size can be factored into spatial dimensions
        // In practice, we'd need to track the actual spatial dimensions
        let spatial_dim = (flat_size as f32).sqrt() as usize;
        let _channels = flat_size / (spatial_dim * spatial_dim);

        // Simple spatial pooling over neurons
        let pool_factor = self.pool_size.0 * self.pool_size.1;
        let output_size = flat_size / pool_factor;

        let mut output = Array3::zeros((batch_size, num_steps, output_size));

        for b in 0..batch_size {
            for t in 0..num_steps {
                for i in 0..output_size {
                    let start = i * pool_factor;
                    let end = (start + pool_factor).min(flat_size);
                    let sum: f32 = input_dense.slice(s![b, t, start..end]).sum();
                    output[[b, t, i]] = sum;
                }
            }
        }

        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {
        // No state to reset
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        Vec::new()
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No gradients
    }
}

/// 2D Max Pooling for spikes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingMaxPool2d {
    /// Pool size (height, width)
    pub pool_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// Indices of maximum values (for backprop)
    #[serde(skip)]
    pub max_indices: Option<Vec<Vec<Vec<usize>>>>,
}

impl SpikingMaxPool2d {
    /// Create a new max pooling layer
    pub fn new(pool_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            pool_size,
            stride,
            max_indices: None,
        }
    }

    /// Calculate output dimensions
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = (input_h - self.pool_size.0) / self.stride.0 + 1;
        let out_w = (input_w - self.pool_size.1) / self.stride.1 + 1;
        (out_h, out_w)
    }
}

impl SpikingLayer for SpikingMaxPool2d {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, flat_size) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        let pool_factor = self.pool_size.0 * self.pool_size.1;
        let output_size = flat_size / pool_factor;

        let mut output = Array3::zeros((batch_size, num_steps, output_size));
        let mut indices = vec![vec![vec![0; output_size]; num_steps]; batch_size];

        for b in 0..batch_size {
            for t in 0..num_steps {
                for i in 0..output_size {
                    let start = i * pool_factor;
                    let end = (start + pool_factor).min(flat_size);

                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = start;

                    for idx in start..end {
                        let val = input_dense[[b, t, idx]];
                        if val > max_val {
                            max_val = val;
                            max_idx = idx;
                        }
                    }

                    output[[b, t, i]] = max_val;
                    indices[b][t][i] = max_idx;
                }
            }
        }

        self.max_indices = Some(indices);
        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {
        self.max_indices = None;
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        Vec::new()
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No parameters to have gradients
    }
}

/// Temporal pooling - averages spikes over time windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAvgPool {
    /// Window size
    pub window_size: usize,
    /// Stride
    pub stride: usize,
}

impl TemporalAvgPool {
    pub fn new(window_size: usize, stride: usize) -> Self {
        Self { window_size, stride }
    }

    pub fn output_steps(&self, input_steps: usize) -> usize {
        (input_steps - self.window_size) / self.stride + 1
    }
}

impl SpikingLayer for TemporalAvgPool {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        let output_steps = self.output_steps(num_steps);
        let mut output = Array3::zeros((batch_size, output_steps, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                for t_out in 0..output_steps {
                    let t_start = t_out * self.stride;
                    let t_end = (t_start + self.window_size).min(num_steps);

                    let sum: f32 = input_dense.slice(s![b, t_start..t_end, n]).sum();
                    output[[b, t_out, n]] = sum / self.window_size as f32;
                }
            }
        }

        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {
        // No state
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        Vec::new()
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // No gradients
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_pool_creation() {
        let pool = SpikingSumPool2d::new((2, 2), (2, 2));
        assert_eq!(pool.pool_size, (2, 2));
    }

    #[test]
    fn test_sum_pool_forward() {
        let mut pool = SpikingSumPool2d::new((2, 2), (2, 2));
        let input = SpikeTensor::zeros(2, 10, 16, false);
        let output = pool.forward(&input).unwrap();

        // 16 neurons / 4 (2x2 pool) = 4 output neurons
        assert_eq!(output.shape(), (2, 10, 4));
    }

    #[test]
    fn test_max_pool_forward() {
        let mut pool = SpikingMaxPool2d::new((2, 2), (2, 2));
        let input = SpikeTensor::zeros(1, 5, 16, false);
        let output = pool.forward(&input).unwrap();

        assert_eq!(output.shape(), (1, 5, 4));
        assert!(pool.max_indices.is_some());
    }

    #[test]
    fn test_temporal_avg_pool() {
        let mut pool = TemporalAvgPool::new(5, 2);
        let input = SpikeTensor::zeros(2, 20, 10, false);

        let output_steps = pool.output_steps(20);
        assert_eq!(output_steps, 8); // (20 - 5) / 2 + 1

        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 8, 10));
    }
}
