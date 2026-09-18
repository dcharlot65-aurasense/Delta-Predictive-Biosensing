//! # ANN Baselines for SNN Comparison
//!
//! This module provides 44 ANN baseline architectures for fair comparison with SNNs.
//! These baselines serve as reference points for evaluating SNN performance across
//! different architectural paradigms.

pub mod cnn;
pub mod conversion;
pub mod mlp;
pub mod rnn;
pub mod specialized;
pub mod transformer;

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
        for _i in 0..size {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = (rng_state as f32) / (u64::MAX as f32);

            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
    /// Build a tensor from a function of the flat index.
    pub fn from_shape_fn(shape: Vec<usize>, f: impl Fn(usize) -> f32) -> Self {
        let n = shape.iter().product();
        Self {
            data: (0..n).map(f).collect(),
            shape,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Matrix multiplication (2D only)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(
            self.shape[1], other.shape[0],
            "incompatible shapes for matmul"
        );

        let m = self.shape[0];
        let n = self.shape[1];
        let p = other.shape[1];
        record_macs((m * n * p) as u64);

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
        let data = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
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
            data.iter_mut()
                .for_each(|x| *x = (*x - max_val).exp() / sum);
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
                slice
                    .iter_mut()
                    .for_each(|x| *x = (*x - max_val).exp() / sum);
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
            let data = self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect();
            return Tensor {
                data,
                shape: self.shape.clone(),
            };
        }

        if let Some(width) = self.broadcast_width(other) {
            let data = self
                .data
                .iter()
                .enumerate()
                .map(|(i, a)| a + other.data[i % width])
                .collect();
            return Tensor {
                data,
                shape: self.shape.clone(),
            };
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
            && let Some(width) = self.broadcast_width(other)
        {
            let data = self
                .data
                .iter()
                .enumerate()
                .map(|(i, a)| a * other.data[i % width])
                .collect();
            return Tensor {
                data,
                shape: self.shape.clone(),
            };
        }
        assert_eq!(
            self.shape, other.shape,
            "shapes must match for multiplication"
        );
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
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
        assert_eq!(
            self.shape.len(),
            3,
            "conv1d requires 3D input [batch, channels, length]"
        );
        assert_eq!(
            kernel.shape.len(),
            3,
            "conv1d requires 3D kernel [out_ch, in_ch, kernel_size]"
        );

        let batch = self.shape[0];
        let in_channels = self.shape[1];
        let in_length = self.shape[2];
        let out_channels = kernel.shape[0];
        let kernel_size = kernel.shape[2];

        let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
        record_macs((batch * out_channels * out_length * in_channels * kernel_size) as u64);
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
                                let in_idx = b * in_channels * in_length
                                    + ic * in_length
                                    + (in_pos - padding);
                                let kernel_idx =
                                    oc * in_channels * kernel_size + ic * kernel_size + k;
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

    /// 2-D convolution over `[batch, in_channels, height, width]`.
    ///
    /// `kernel` is `[out_channels, in_channels, kh, kw]`; padding is zero
    /// padding applied symmetrically. Output spatial size follows the usual
    /// `(n + 2p - k) / s + 1`.
    pub fn conv2d(
        &self,
        kernel: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor {
        assert_eq!(
            self.shape.len(),
            4,
            "conv2d requires 4D input [batch, channels, height, width]"
        );
        assert_eq!(
            kernel.shape.len(),
            4,
            "conv2d requires 4D kernel [out_ch, in_ch, kh, kw]"
        );
        assert_eq!(
            self.shape[1], kernel.shape[1],
            "conv2d input channels {} do not match kernel {}",
            self.shape[1], kernel.shape[1]
        );

        let (batch, in_ch, in_h, in_w) =
            (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (out_ch, kh, kw) = (kernel.shape[0], kernel.shape[2], kernel.shape[3]);
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let out_h = (in_h + 2 * ph).saturating_sub(kh) / sh.max(1) + 1;
        let out_w = (in_w + 2 * pw).saturating_sub(kw) / sw.max(1) + 1;

        record_macs((batch * out_ch * out_h * out_w * in_ch * kh * kw) as u64);
        let mut result = vec![0.0; batch * out_ch * out_h * out_w];
        for b in 0..batch {
            for oc in 0..out_ch {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut sum = 0.0;
                        for ic in 0..in_ch {
                            for ky in 0..kh {
                                let iy = oy * sh + ky;
                                if iy < ph || iy >= in_h + ph {
                                    continue;
                                }
                                let iy = iy - ph;
                                for kx in 0..kw {
                                    let ix = ox * sw + kx;
                                    if ix < pw || ix >= in_w + pw {
                                        continue;
                                    }
                                    let ix = ix - pw;
                                    let in_idx = ((b * in_ch + ic) * in_h + iy) * in_w + ix;
                                    let k_idx = ((oc * in_ch + ic) * kh + ky) * kw + kx;
                                    sum += self.data[in_idx] * kernel.data[k_idx];
                                }
                            }
                        }
                        result[((b * out_ch + oc) * out_h + oy) * out_w + ox] = sum;
                    }
                }
            }
        }

        Tensor {
            data: result,
            shape: vec![batch, out_ch, out_h, out_w],
        }
    }

    /// Depthwise 2-D convolution: one `[1, kh, kw]` filter per input channel.
    ///
    /// `kernel` is `[channels, 1, kh, kw]`. This is the first half of a
    /// depthwise-separable convolution; the pointwise half is [`Self::conv2d`]
    /// with a 1x1 kernel.
    pub fn depthwise_conv2d(
        &self,
        kernel: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor {
        assert_eq!(self.shape.len(), 4, "depthwise_conv2d requires 4D input");
        assert_eq!(kernel.shape.len(), 4, "depthwise_conv2d requires 4D kernel");
        assert_eq!(
            kernel.shape[0], self.shape[1],
            "depthwise_conv2d needs one filter per channel"
        );
        assert_eq!(
            kernel.shape[1], 1,
            "depthwise_conv2d kernel must be [C, 1, kh, kw]"
        );

        let (batch, ch, in_h, in_w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (kh, kw) = (kernel.shape[2], kernel.shape[3]);
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let out_h = (in_h + 2 * ph).saturating_sub(kh) / sh.max(1) + 1;
        let out_w = (in_w + 2 * pw).saturating_sub(kw) / sw.max(1) + 1;

        record_macs((batch * ch * out_h * out_w * kh * kw) as u64);
        let mut result = vec![0.0; batch * ch * out_h * out_w];
        for b in 0..batch {
            for c in 0..ch {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut sum = 0.0;
                        for ky in 0..kh {
                            let iy = oy * sh + ky;
                            if iy < ph || iy >= in_h + ph {
                                continue;
                            }
                            let iy = iy - ph;
                            for kx in 0..kw {
                                let ix = ox * sw + kx;
                                if ix < pw || ix >= in_w + pw {
                                    continue;
                                }
                                let ix = ix - pw;
                                let in_idx = ((b * ch + c) * in_h + iy) * in_w + ix;
                                let k_idx = (c * kh + ky) * kw + kx;
                                sum += self.data[in_idx] * kernel.data[k_idx];
                            }
                        }
                        result[((b * ch + c) * out_h + oy) * out_w + ox] = sum;
                    }
                }
            }
        }

        Tensor {
            data: result,
            shape: vec![batch, ch, out_h, out_w],
        }
    }

    /// Max pooling over `[batch, channels, height, width]`.
    pub fn max_pool2d(&self, kernel_size: usize, stride: usize) -> Tensor {
        assert_eq!(self.shape.len(), 4, "max_pool2d requires 4D input");
        let (batch, ch, in_h, in_w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let stride = stride.max(1);
        let out_h = in_h.saturating_sub(kernel_size) / stride + 1;
        let out_w = in_w.saturating_sub(kernel_size) / stride + 1;

        let mut result = vec![f32::NEG_INFINITY; batch * ch * out_h * out_w];
        for b in 0..batch {
            for c in 0..ch {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let mut best = f32::NEG_INFINITY;
                        for ky in 0..kernel_size {
                            for kx in 0..kernel_size {
                                let iy = oy * stride + ky;
                                let ix = ox * stride + kx;
                                if iy < in_h && ix < in_w {
                                    best =
                                        best.max(self.data[((b * ch + c) * in_h + iy) * in_w + ix]);
                                }
                            }
                        }
                        result[((b * ch + c) * out_h + oy) * out_w + ox] = best;
                    }
                }
            }
        }

        Tensor {
            data: result,
            shape: vec![batch, ch, out_h, out_w],
        }
    }

    /// Global average pooling: `[batch, channels, height, width]` -> `[batch, channels]`.
    pub fn global_avg_pool2d(&self) -> Tensor {
        assert_eq!(self.shape.len(), 4, "global_avg_pool2d requires 4D input");
        let (batch, ch, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let area = (h * w) as f32;
        let mut result = vec![0.0; batch * ch];
        for b in 0..batch {
            for c in 0..ch {
                let base = (b * ch + c) * h * w;
                let sum: f32 = self.data[base..base + h * w].iter().sum();
                result[b * ch + c] = sum / area;
            }
        }
        Tensor {
            data: result,
            shape: vec![batch, ch],
        }
    }

    /// Batch normalization (inference mode), per feature or per channel.
    ///
    /// Handles `[n, features]`, `[n, channels, length]` and
    /// `[n, channels, height, width]`. It previously normalised only rank 2 and
    /// returned every other rank unchanged, so convolutional activations passed
    /// through untouched and a network with batch norm silently had none.
    /// Unsupported ranks now panic rather than quietly doing nothing.
    pub fn batch_norm(&self, mean: &Tensor, var: &Tensor, gamma: &Tensor, beta: &Tensor) -> Tensor {
        let eps = 1e-5;
        let mut result = self.data.clone();

        // Elements sharing a normalisation statistic are contiguous in every
        // supported layout, so one stride pair covers all three ranks.
        let (channels, inner) = match self.shape.len() {
            2 => (self.shape[1], 1),
            3 => (self.shape[1], self.shape[2]),
            4 => (self.shape[1], self.shape[2] * self.shape[3]),
            other => panic!("batch_norm does not support rank {other}"),
        };
        assert!(
            mean.data.len() >= channels
                && var.data.len() >= channels
                && gamma.data.len() >= channels
                && beta.data.len() >= channels,
            "batch_norm statistics must cover {channels} channels"
        );

        for (idx, value) in result.iter_mut().enumerate() {
            let c = (idx / inner) % channels;
            let normalised = (*value - mean.data[c]) / (var.data[c] + eps).sqrt();
            *value = normalised * gamma.data[c] + beta.data[c];
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

        // Normalises over the last axis. Ranks other than 2 and 3 used to fall
        // through and return the input unchanged.
        assert!(
            matches!(self.shape.len(), 2 | 3),
            "layer_norm does not support rank {}",
            self.shape.len()
        );
        {
            let features = self.shape[self.shape.len() - 1];
            for i in 0..(self.size() / features) {
                let start = i * features;
                let end = start + features;
                let slice = &self.data[start..end];

                let mean: f32 = slice.iter().sum::<f32>() / features as f32;
                let var: f32 =
                    slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / features as f32;
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

#[cfg(test)]
mod conv2d_tests {
    use super::*;

    // Reference values produced independently with numpy (see the generator in
    // the commit message): an asymmetric case throughout -- rectangular 5x6
    // input, non-square 3x2 kernel, stride (2, 1), padding (1, 0) -- so that a
    // transposed axis, a swapped stride or a dropped pad cannot agree by
    // symmetry. Pinned to values rather than shapes: a wrong read window still
    // produces a correctly shaped output.
    const X: &[f32] = &[
        -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3,
        2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1,
        4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,
    ];
    const K: &[f32] = &[
        -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, -0.5,
        -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, -0.5, -0.25, 0.0,
        0.25, 0.5, 0.75, 1.0, -0.5,
    ];
    const DK: &[f32] = &[
        -0.4, -0.1, 0.2, 0.5, 0.8, -0.4, -0.1, 0.2, 0.5, 0.8, -0.4, -0.1,
    ];
    const EXPECT_CONV2D: &[f32] = &[
        0.85, 1.05, 1.25, 1.45, 1.65, 4.775, 4.95, 5.125, 5.3, 5.475, 0.5, 0.475, 0.45, 0.425, 0.4,
        0.95, 0.925, 0.9, 0.875, 0.85, 3.375, 3.65, 3.925, 4.2, 4.475, 7.4, 7.675, 7.95, 8.225,
        8.5, 2.45, 2.725, 3.0, 3.275, 3.55, 6.175, 6.55, 6.925, 7.3, 7.675, 9.75, 10.15, 10.55,
        10.95, 11.35,
    ];
    const EXPECT_DW: &[f32] = &[
        -0.85, -0.74, -0.63, -0.52, -0.41, 0.66, 0.72, 0.78, 0.84, 0.9, 0.62, 0.64, 0.66, 0.68,
        0.7, 1.37, 1.45, 1.53, 1.61, 1.69, 2.61, 2.7, 2.79, 2.88, 2.97, 6.2, 6.34, 6.48, 6.62,
        6.76,
    ];

    fn close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() < 1e-4,
                "{what}: element {i} was {g}, numpy says {w}"
            );
        }
    }

    #[test]
    fn conv2d_matches_numpy() {
        let x = Tensor::from_vec(X.to_vec(), vec![1, 2, 5, 6]);
        let k = Tensor::from_vec(K.to_vec(), vec![3, 2, 3, 2]);
        let out = x.conv2d(&k, (2, 1), (1, 0));
        assert_eq!(out.shape, vec![1, 3, 3, 5]);
        close(&out.data, EXPECT_CONV2D, "conv2d");
    }

    #[test]
    fn depthwise_conv2d_matches_numpy() {
        let x = Tensor::from_vec(X.to_vec(), vec![1, 2, 5, 6]);
        let k = Tensor::from_vec(DK.to_vec(), vec![2, 1, 3, 2]);
        let out = x.depthwise_conv2d(&k, (2, 1), (1, 0));
        assert_eq!(out.shape, vec![1, 2, 3, 5]);
        close(&out.data, EXPECT_DW, "depthwise_conv2d");
    }

    /// A 1x1 convolution is a per-pixel channel mix, so it must equal a matmul
    /// over the channel axis -- an identity that holds independently of how
    /// conv2d indexes, and therefore a second opinion on it.
    #[test]
    fn pointwise_conv2d_is_a_channel_matmul() {
        let (ic, oc, h, w) = (3usize, 4usize, 2usize, 3usize);
        let x = Tensor::from_shape_fn(vec![1, ic, h, w], |i| (i % 5) as f32 * 0.3 - 0.6);
        let k = Tensor::from_shape_fn(vec![oc, ic, 1, 1], |i| (i % 7) as f32 * 0.2 - 0.5);
        let got = x.conv2d(&k, (1, 1), (0, 0));

        for o in 0..oc {
            for y in 0..h {
                for xx in 0..w {
                    let mut want = 0.0;
                    for c in 0..ic {
                        want += x.data[(c * h + y) * w + xx] * k.data[o * ic + c];
                    }
                    let got_v = got.data[(o * h + y) * w + xx];
                    assert!(
                        (got_v - want).abs() < 1e-5,
                        "1x1 conv at ({o},{y},{xx}) was {got_v}, channel matmul says {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn max_pool2d_and_global_avg_pool_are_exact() {
        // 1 x 1 x 4 x 4 counting up: pooling 2x2 stride 2 takes the
        // bottom-right of each quadrant.
        let x = Tensor::from_shape_fn(vec![1, 1, 4, 4], |i| i as f32);
        let pooled = x.max_pool2d(2, 2);
        assert_eq!(pooled.shape, vec![1, 1, 2, 2]);
        assert_eq!(pooled.data, vec![5.0, 7.0, 13.0, 15.0]);

        // Mean of 0..16 is 7.5.
        let avg = x.global_avg_pool2d();
        assert_eq!(avg.shape, vec![1, 1]);
        assert!((avg.data[0] - 7.5).abs() < 1e-6);
    }

    /// batch_norm used to return anything that was not rank 2 unchanged.
    #[test]
    fn batch_norm_normalises_convolutional_activations() {
        let x = Tensor::from_shape_fn(vec![1, 2, 2, 2], |i| i as f32);
        let mean = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let var = Tensor::from_vec(vec![4.0, 4.0], vec![2]);
        let gamma = Tensor::from_vec(vec![2.0, 1.0], vec![2]);
        let beta = Tensor::from_vec(vec![0.5, -1.0], vec![2]);

        let out = x.batch_norm(&mean, &var, &gamma, &beta);
        assert_ne!(out.data, x.data, "rank-4 batch_norm was an identity");
        // Channel 0: (0 - 1)/sqrt(4 + 1e-5) * 2 + 0.5 = -0.5 (to tolerance).
        assert!((out.data[0] - -0.5).abs() < 1e-3, "got {}", out.data[0]);
        // Channel 1 starts at element 4: (4 - 5)/2 * 1 - 1 = -1.5.
        assert!((out.data[4] - -1.5).abs() < 1e-3, "got {}", out.data[4]);
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

// ============================================================================
// Multiply-accumulate counting
//
// `flops_per_inference` used to return a constant for most architectures --
// one number regardless of input size, which cannot be right for a
// convolution. Rather than re-derive a formula per architecture and let it
// drift from the code, the tensor operations tally the multiply-accumulates
// they actually perform, and a model reports the total for one real forward
// pass. Each operation records once, with a count it already computed, so the
// tally costs nothing when it is switched off.
// ============================================================================

thread_local! {
    static MAC_TALLY: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Records multiply-accumulates, when counting is active.
fn record_macs(count: u64) {
    MAC_TALLY.with(|tally| {
        if let Some(total) = tally.get() {
            tally.set(Some(total + count));
        }
    });
}

/// Counts the multiply-accumulates performed by tensor operations in `body`.
///
/// Counting is thread-local and does not nest: an inner call would restart the
/// tally, so do not call this from inside a counted region.
pub fn count_macs(body: impl FnOnce()) -> u64 {
    MAC_TALLY.with(|tally| tally.set(Some(0)));
    body();
    MAC_TALLY.with(|tally| tally.replace(None)).unwrap_or(0)
}

/// FLOPs for a counted forward pass, at two per multiply-accumulate.
///
/// One multiply and one add, which is the convention `CNN1D_Small` already
/// used for its fully connected layers.
pub fn flops_of(body: impl FnOnce()) -> u64 {
    2 * count_macs(body)
}

/// Helper function to count parameters in a weight matrix
pub fn count_params(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Helper function to initialize weights with Xavier/Glorot initialization
pub fn xavier_init(shape: Vec<usize>, seed: u64) -> Tensor {
    let fan_in = if shape.len() >= 2 {
        shape[shape.len() - 2]
    } else {
        1
    };
    let fan_out = if !shape.is_empty() {
        shape[shape.len() - 1]
    } else {
        1
    };
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

    let mut tensor = Tensor::randn(shape, seed);
    tensor
        .data
        .iter_mut()
        .for_each(|x| *x = (*x).clamp(-limit, limit));
    tensor
}

// Re-export all baseline architectures
pub use cnn::*;
pub use conversion::*;
pub use mlp::*;
pub use rnn::*;
pub use specialized::*;
pub use transformer::*;

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
