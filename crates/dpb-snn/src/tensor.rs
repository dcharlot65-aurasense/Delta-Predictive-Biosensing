//! Spike tensor operations and representations
//!
//! This module provides efficient representations for spike data, including
//! dense and sparse formats with conversion utilities.

use crate::{SNNError, SNNResult};
use ndarray::{s, Array, Array1, Array2, Array3, Array4, ArrayD, Axis, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents spike data in different formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpikeRepresentation {
    /// Dense binary tensor (batch, time, neurons)
    Dense(Array3<f32>),
    /// Sparse representation with spike times
    Sparse(SparseSpikes),
}

/// Sparse spike representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseSpikes {
    /// Number of neurons
    pub num_neurons: usize,
    /// Number of time steps
    pub num_steps: usize,
    /// Batch size
    pub batch_size: usize,
    /// Spike events: (batch_idx, time_idx, neuron_idx)
    pub events: Vec<(usize, usize, usize)>,
}

impl SparseSpikes {
    /// Create a new sparse spike representation
    pub fn new(batch_size: usize, num_steps: usize, num_neurons: usize) -> Self {
        Self {
            num_neurons,
            num_steps,
            batch_size,
            events: Vec::new(),
        }
    }

    /// Add a spike event
    pub fn add_spike(&mut self, batch_idx: usize, time_idx: usize, neuron_idx: usize) -> SNNResult<()> {
        if batch_idx >= self.batch_size {
            return Err(SNNError::InvalidConfig(format!(
                "Batch index {} out of bounds (max: {})",
                batch_idx, self.batch_size
            )));
        }
        if time_idx >= self.num_steps {
            return Err(SNNError::InvalidConfig(format!(
                "Time index {} out of bounds (max: {})",
                time_idx, self.num_steps
            )));
        }
        if neuron_idx >= self.num_neurons {
            return Err(SNNError::InvalidConfig(format!(
                "Neuron index {} out of bounds (max: {})",
                neuron_idx, self.num_neurons
            )));
        }
        self.events.push((batch_idx, time_idx, neuron_idx));
        Ok(())
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> Array3<f32> {
        let mut dense = Array3::zeros((self.batch_size, self.num_steps, self.num_neurons));
        for &(b, t, n) in &self.events {
            dense[[b, t, n]] = 1.0;
        }
        dense
    }

    /// Get number of spikes
    pub fn num_spikes(&self) -> usize {
        self.events.len()
    }

    /// Get spikes for a specific batch and time
    pub fn get_spikes_at(&self, batch_idx: usize, time_idx: usize) -> Vec<usize> {
        self.events
            .iter()
            .filter(|&&(b, t, _)| b == batch_idx && t == time_idx)
            .map(|&(_, _, n)| n)
            .collect()
    }
}

/// Main spike tensor structure
#[derive(Debug, Clone)]
pub struct SpikeTensor {
    /// Spike representation
    pub data: SpikeRepresentation,
    /// Gradient (for backpropagation)
    pub grad: Option<Array3<f32>>,
    /// Whether this tensor requires gradients
    pub requires_grad: bool,
}

impl SpikeTensor {
    /// Create a new spike tensor from dense data
    pub fn from_dense(data: Array3<f32>, requires_grad: bool) -> Self {
        Self {
            data: SpikeRepresentation::Dense(data),
            grad: None,
            requires_grad,
        }
    }

    /// Create a new spike tensor from sparse data
    pub fn from_sparse(data: SparseSpikes, requires_grad: bool) -> Self {
        Self {
            data: SpikeRepresentation::Sparse(data),
            grad: None,
            requires_grad,
        }
    }

    /// Create a zero-initialized spike tensor
    pub fn zeros(batch_size: usize, num_steps: usize, num_neurons: usize, requires_grad: bool) -> Self {
        let data = Array3::zeros((batch_size, num_steps, num_neurons));
        Self::from_dense(data, requires_grad)
    }

    /// Get shape (batch_size, num_steps, num_neurons)
    pub fn shape(&self) -> (usize, usize, usize) {
        match &self.data {
            SpikeRepresentation::Dense(arr) => (arr.shape()[0], arr.shape()[1], arr.shape()[2]),
            SpikeRepresentation::Sparse(sparse) => (sparse.batch_size, sparse.num_steps, sparse.num_neurons),
        }
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.shape().0
    }

    /// Get number of time steps
    pub fn num_steps(&self) -> usize {
        self.shape().1
    }

    /// Get number of neurons
    pub fn num_neurons(&self) -> usize {
        self.shape().2
    }

    /// Convert to dense representation
    pub fn to_dense(&self) -> Array3<f32> {
        match &self.data {
            SpikeRepresentation::Dense(arr) => arr.clone(),
            SpikeRepresentation::Sparse(sparse) => sparse.to_dense(),
        }
    }

    /// Convert to sparse representation
    pub fn to_sparse(&self) -> SparseSpikes {
        match &self.data {
            SpikeRepresentation::Dense(arr) => {
                let (batch_size, num_steps, num_neurons) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
                let mut sparse = SparseSpikes::new(batch_size, num_steps, num_neurons);
                for b in 0..batch_size {
                    for t in 0..num_steps {
                        for n in 0..num_neurons {
                            if arr[[b, t, n]] > 0.5 {
                                sparse.add_spike(b, t, n).unwrap();
                            }
                        }
                    }
                }
                sparse
            }
            SpikeRepresentation::Sparse(sparse) => sparse.clone(),
        }
    }

    /// Compute spike rate (spikes per time step)
    pub fn spike_rate(&self) -> Array2<f32> {
        let dense = self.to_dense();
        let (batch_size, num_steps, num_neurons) = (dense.shape()[0], dense.shape()[1], dense.shape()[2]);

        let mut rates = Array2::zeros((batch_size, num_neurons));
        for b in 0..batch_size {
            for n in 0..num_neurons {
                let total_spikes: f32 = dense.slice(s![b, .., n]).sum();
                rates[[b, n]] = total_spikes / num_steps as f32;
            }
        }
        rates
    }

    /// Accumulate gradients
    pub fn accumulate_grad(&mut self, grad: Array3<f32>) -> SNNResult<()> {
        if !self.requires_grad {
            return Ok(());
        }

        if let Some(existing_grad) = &mut self.grad {
            *existing_grad += &grad;
        } else {
            self.grad = Some(grad);
        }
        Ok(())
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = None;
        }
    }

    /// Clone the tensor without gradients
    pub fn detach(&self) -> Self {
        Self {
            data: self.data.clone(),
            grad: None,
            requires_grad: false,
        }
    }
}

/// Spike tensor operations
impl SpikeTensor {
    /// Temporal convolution (spike filtering)
    pub fn temporal_conv(&self, kernel: &Array1<f32>) -> SNNResult<SpikeTensor> {
        let dense = self.to_dense();
        let (batch_size, num_steps, num_neurons) = (dense.shape()[0], dense.shape()[1], dense.shape()[2]);
        let kernel_len = kernel.len();

        if kernel_len > num_steps {
            return Err(SNNError::InvalidConfig(
                "Kernel length exceeds number of time steps".to_string()
            ));
        }

        let mut output = Array3::zeros((batch_size, num_steps, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                for t in 0..num_steps {
                    let mut sum = 0.0;
                    for k in 0..kernel_len {
                        if t >= k {
                            sum += dense[[b, t - k, n]] * kernel[k];
                        }
                    }
                    output[[b, t, n]] = sum;
                }
            }
        }

        Ok(SpikeTensor::from_dense(output, self.requires_grad))
    }

    /// Spatial pooling (max pooling over neurons)
    pub fn spatial_max_pool(&self, pool_size: usize) -> SNNResult<SpikeTensor> {
        let dense = self.to_dense();
        let (batch_size, num_steps, num_neurons) = (dense.shape()[0], dense.shape()[1], dense.shape()[2]);

        if num_neurons % pool_size != 0 {
            return Err(SNNError::InvalidConfig(
                "Number of neurons must be divisible by pool size".to_string()
            ));
        }

        let output_neurons = num_neurons / pool_size;
        let mut output = Array3::zeros((batch_size, num_steps, output_neurons));

        for b in 0..batch_size {
            for t in 0..num_steps {
                for n_out in 0..output_neurons {
                    let start = n_out * pool_size;
                    let end = start + pool_size;
                    let max_val = dense.slice(s![b, t, start..end])
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    output[[b, t, n_out]] = max_val;
                }
            }
        }

        Ok(SpikeTensor::from_dense(output, self.requires_grad))
    }

    /// Temporal pooling (sum over time windows)
    pub fn temporal_sum_pool(&self, pool_size: usize) -> SNNResult<SpikeTensor> {
        let dense = self.to_dense();
        let (batch_size, num_steps, num_neurons) = (dense.shape()[0], dense.shape()[1], dense.shape()[2]);

        if num_steps % pool_size != 0 {
            return Err(SNNError::InvalidConfig(
                "Number of time steps must be divisible by pool size".to_string()
            ));
        }

        let output_steps = num_steps / pool_size;
        let mut output = Array3::zeros((batch_size, output_steps, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                for t_out in 0..output_steps {
                    let start = t_out * pool_size;
                    let end = start + pool_size;
                    let sum: f32 = dense.slice(s![b, start..end, n]).sum();
                    output[[b, t_out, n]] = sum;
                }
            }
        }

        Ok(SpikeTensor::from_dense(output, self.requires_grad))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_spikes() {
        let mut sparse = SparseSpikes::new(2, 10, 5);
        sparse.add_spike(0, 3, 2).unwrap();
        sparse.add_spike(1, 5, 4).unwrap();

        assert_eq!(sparse.num_spikes(), 2);

        let dense = sparse.to_dense();
        assert_eq!(dense[[0, 3, 2]], 1.0);
        assert_eq!(dense[[1, 5, 4]], 1.0);
        assert_eq!(dense[[0, 0, 0]], 0.0);
    }

    #[test]
    fn test_spike_tensor_creation() {
        let tensor = SpikeTensor::zeros(2, 10, 5, true);
        assert_eq!(tensor.shape(), (2, 10, 5));
        assert!(tensor.requires_grad);
    }

    #[test]
    fn test_spike_rate() {
        let mut data = Array3::zeros((1, 10, 3));
        data[[0, 2, 0]] = 1.0;
        data[[0, 5, 0]] = 1.0;
        data[[0, 7, 1]] = 1.0;

        let tensor = SpikeTensor::from_dense(data, false);
        let rates = tensor.spike_rate();

        assert_eq!(rates[[0, 0]], 0.2); // 2 spikes / 10 steps
        assert_eq!(rates[[0, 1]], 0.1); // 1 spike / 10 steps
        assert_eq!(rates[[0, 2]], 0.0); // 0 spikes
    }

    #[test]
    fn test_dense_sparse_conversion() {
        let mut data = Array3::zeros((1, 5, 3));
        data[[0, 1, 0]] = 1.0;
        data[[0, 3, 2]] = 1.0;

        let tensor = SpikeTensor::from_dense(data.clone(), false);
        let sparse = tensor.to_sparse();
        let dense_again = sparse.to_dense();

        assert_eq!(data, dense_again);
    }
}
