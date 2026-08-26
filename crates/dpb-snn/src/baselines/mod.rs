//! # ANN Baselines for SNN Comparison
//!
//! This module provides 44 ANN baseline architectures for fair comparison with SNNs.
//! These baselines serve as reference points for evaluating SNN performance across
//! different architectural paradigms.

pub mod mlp;
pub mod cnn;
pub mod rnn;
pub mod transformer;
pub mod specialized;
pub mod conversion;

use std::f32::consts::PI;

/// Simple tensor type for ANN baselines
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Create a tensor filled with random normal values
    pub fn randn(shape: Vec<usize>, seed: u64) -> Self {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);

        // Simple Box-Muller transform for normal distribution
        let mut rng_state = seed;
        for i in 0..size {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (rng_state as f32) / (u64::MAX as f32);

            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (rng_state as f32) / (u64::MAX as f32);

            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            data.push(z * 0.1); // Scale by 0.1 for initialization
        }

        Self { data, shape }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![1.0; size],
            shape,
        }
    }

    /// Create a tensor from raw data
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_size: usize = shape.iter().product();
        assert_eq!(data.len(), expected_size);
        Self { data, shape }
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Matrix multiplication (2D only)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "incompatible shapes for matmul");

        let m = self.shape[0];
        let n = self.shape[1];
        let p = other.shape[1];

        let mut result = vec![0.0; m * p];

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self.data[i * n + k] * other.data[k * p + j];
                }
                result[i * p + j] = sum;
            }
        }

        Tensor {
            data: result,
            shape: vec![m, p],
        }
    }

    /// ReLU activation
    pub fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Tensor {
        let data = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Tanh activation
    pub fn tanh(&self) -> Tensor {
        let data = self.data.iter().map(|&x| x.tanh()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Softmax activation (along last dimension)
    pub fn softmax(&self) -> Tensor {
        let mut data = self.data.clone();

        if self.shape.len() == 1 {
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = data.iter().map(|&x| (x - max_val).exp()).sum();
            data.iter_mut().for_each(|x| *x = (*x - max_val).exp() / sum);
        } else {
            // For 2D: softmax along last dimension
            let last_dim = *self.shape.last().unwrap();
            let num_groups = self.size() / last_dim;

            for g in 0..num_groups {
                let start = g * last_dim;
                let end = start + last_dim;
                let slice = &mut data[start..end];

                let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let sum: f32 = slice.iter().map(|&x| (x - max_val).exp()).sum();
                slice.iter_mut().for_each(|x| *x = (*x - max_val).exp() / sum);
            }
        }

        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Element-wise addition with trailing-dimension broadcasting.
    ///
    /// Exact-shape addition is the common case. A rank-1 operand whose length
    /// matches the trailing dimension is broadcast across the leading ones, which
    /// is what a bias add is: `[batch, features] + [features]`.
    ///
    /// This previously required exact shape equality and panicked with
    /// "shapes must match for addition ... left: [1, 20] right: [20]" on every
    /// bias add, so every MLP, RNN, CNN and Transformer baseline failed at
    /// construction — 13 tests across four modules from this one assert.
    pub fn add(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
            return Tensor { data, shape: self.shape.clone() };
        }

        if let Some(width) = self.broadcast_width(other) {
            let data = self
                .data
                .iter()
                .enumerate()
                .map(|(i, a)| a + other.data[i % width])
                .collect();
            return Tensor { data, shape: self.shape.clone() };
        }

        panic!(
            "shapes must match for addition (or broadcast a trailing dim): \
             left: {:?} right: {:?}",
            self.shape, other.shape
        );
    }

    /// Length of the trailing dimension when `other` can broadcast against self.
    ///
    /// Returns `Some(width)` when `other` is rank-1, its length equals self's
    /// trailing dimension, and self's total length is a whole number of rows.
    fn broadcast_width(&self, other: &Tensor) -> Option<usize> {
        let width = *self.shape.last()?;
        let ok = other.shape.len() == 1
            && other.shape[0] == width
            && width > 0
            && self.data.len().is_multiple_of(width)
            && other.data.len() == width;
        ok.then_some(width)
    }

    /// Element-wise multiplication with the same trailing-dimension
    /// broadcasting rule as [`Tensor::add`] (per-feature scaling).
    pub fn mul(&self, other: &Tensor) -> Tensor {
        if self.shape != other.shape
            && let Some(width) = self.broadcast_width(other) {
                let data = self
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, a)| a * other.data[i % width])
                    .collect();
                return Tensor { data, shape: self.shape.clone() };
            }
        assert_eq!(self.shape, other.shape, "shapes must match for multiplication");
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(self.size(), new_size, "size must match");
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
        }
    }

    /// 1D convolution (simplified)
    pub fn conv1d(&self, kernel: &Tensor, stride: usize, padding: usize) -> Tensor {
        assert_eq!(self.shape.len(), 3, "conv1d requires 3D input [batch, channels, length]");
        assert_eq!(kernel.shape.len(), 3, "conv1d requires 3D kernel [out_ch, in_ch, kernel_size]");

        let batch = self.shape[0];
        let in_channels = self.shape[1];
        let in_length = self.shape[2];
        let out_channels = kernel.shape[0];
        let kernel_size = kernel.shape[2];

        let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
        let mut result = vec![0.0; batch * out_channels * out_length];

        // Simplified conv1d implementation
        for b in 0..batch {
            for oc in 0..out_channels {
                for ol in 0..out_length {
                    let mut sum = 0.0;
                    for ic in 0..in_channels {
                        for k in 0..kernel_size {
                            let in_pos = ol * stride + k;
                            if in_pos >= padding && in_pos < in_length + padding {
                                let in_idx = b * in_channels * in_length + ic * in_length + (in_pos - padding);
                                let kernel_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                                sum += self.data[in_idx] * kernel.data[kernel_idx];
                            }
                        }
                    }
                    result[b * out_channels * out_length + oc * out_length + ol] = sum;
                }
            }
        }

        Tensor {
            data: result,
            shape: vec![batch, out_channels, out_length],
        }
    }

    /// Max pooling 1D
    pub fn max_pool1d(&self, kernel_size: usize, stride: usize) -> Tensor {
        assert_eq!(self.shape.len(), 3, "max_pool1d requires 3D input");

        let batch = self.shape[0];
        let channels = self.shape[1];
        let in_length = self.shape[2];
        let out_length = (in_length - kernel_size) / stride + 1;

        let mut result = vec![f32::NEG_INFINITY; batch * channels * out_length];

        for b in 0..batch {
            for c in 0..channels {
                for ol in 0..out_length {
                    let start = ol * stride;
                    let mut max_val = f32::NEG_INFINITY;
                    for k in 0..kernel_size {
                        let idx = b * channels * in_length + c * in_length + start + k;
                        max_val = max_val.max(self.data[idx]);
                    }
                    result[b * channels * out_length + c * out_length + ol] = max_val;
                }
            }
        }

        Tensor {
            data: result,
            shape: vec![batch, channels, out_length],
        }
    }

    /// Batch normalization (inference mode)
    pub fn batch_norm(&self, mean: &Tensor, var: &Tensor, gamma: &Tensor, beta: &Tensor) -> Tensor {
        let eps = 1e-5;
        let mut result = self.data.clone();

        // Simplified batch norm for 2D input
        if self.shape.len() == 2 {
            let features = self.shape[1];
            for i in 0..self.shape[0] {
                for j in 0..features {
                    let idx = i * features + j;
                    result[idx] = (result[idx] - mean.data[j]) / (var.data[j] + eps).sqrt();
                    result[idx] = result[idx] * gamma.data[j] + beta.data[j];
                }
            }
        }

        Tensor {
            data: result,
            shape: self.shape.clone(),
        }
    }

    /// Dropout (inference mode - no dropout)
    pub fn dropout(&self, _p: f32) -> Tensor {
        self.clone()
    }

    /// Layer normalization
    pub fn layer_norm(&self, eps: f32) -> Tensor {
        let mut result = self.data.clone();

        if self.shape.len() == 2 {
            let features = self.shape[1];
            for i in 0..self.shape[0] {
                let start = i * features;
                let end = start + features;
                let slice = &self.data[start..end];

                let mean: f32 = slice.iter().sum::<f32>() / features as f32;
                let var: f32 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / features as f32;
                let std = (var + eps).sqrt();

                for j in 0..features {
                    result[start + j] = (result[start + j] - mean) / std;
                }
            }
        }

        Tensor {
            data: result,
            shape: self.shape.clone(),
        }
    }
}

/// Trait for ANN baseline architectures
pub trait ANNBaseline: Send + Sync {
    /// Get the architecture name
    fn name(&self) -> &str;

    /// Forward pass
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Get the number of trainable parameters
    fn num_parameters(&self) -> usize;

    /// Estimate FLOPs per inference
    fn flops_per_inference(&self) -> u64;

    /// Get architecture summary
    fn architecture_summary(&self) -> String;
}

/// Helper function to count parameters in a weight matrix
pub fn count_params(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Helper function to initialize weights with Xavier/Glorot initialization
pub fn xavier_init(shape: Vec<usize>, seed: u64) -> Tensor {
    let fan_in = if shape.len() >= 2 { shape[shape.len() - 2] } else { 1 };
    let fan_out = if !shape.is_empty() { shape[shape.len() - 1] } else { 1 };
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

    let mut tensor = Tensor::randn(shape, seed);
    tensor.data.iter_mut().for_each(|x| *x = (*x).clamp(-limit, limit));
    tensor
}

// Re-export all baseline architectures
pub use mlp::*;
pub use cnn::*;
pub use rnn::*;
pub use transformer::*;
pub use specialized::*;
pub use conversion::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.data.len(), 6);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let t = Tensor::ones(vec![2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert!(t.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_tensor_randn() {
        let t = Tensor::randn(vec![10, 10], 42);
        assert_eq!(t.shape, vec![10, 10]);
        assert_eq!(t.data.len(), 100);
        // Check values are not all zeros
        assert!(t.data.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_tensor_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b);

        assert_eq!(c.shape, vec![2, 2]);
        // [1,2] * [5,6] = 1*5 + 2*7 = 19
        //       [3,4]   [7,8]
        assert_eq!(c.data[0], 19.0);
    }

    #[test]
    fn test_tensor_relu() {
        let t = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let r = t.relu();
        assert_eq!(r.data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tensor_sigmoid() {
        let t = Tensor::from_vec(vec![0.0], vec![1]);
        let s = t.sigmoid();
        assert!((s.data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let s = t.softmax();
        let sum: f32 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_tensor_scale() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let s = t.scale(2.0);
        assert_eq!(s.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_xavier_init() {
        let w = xavier_init(vec![100, 50], 12345);
        assert_eq!(w.shape, vec![100, 50]);

        // Check that values are within Xavier bounds
        let fan_in = 100;
        let fan_out = 50;
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        assert!(w.data.iter().all(|&x| x.abs() <= limit + 1e-6));
    }
}
