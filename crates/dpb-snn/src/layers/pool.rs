//! Pooling layers for spiking neural networks

use super::SpikingLayer;
use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{Array2, Array3, s};
use serde::{Deserialize, Serialize};

/// Resolves a flat neuron axis into `(channels, height, width)`.
///
/// A `SpikeTensor` is `(batch, steps, flat)` and carries no spatial shape, so a
/// pooling layer has to be told one. When none is set the layer assumes a
/// single square channel, which covers standalone use on an image; anything
/// else is a configuration error rather than a silent reshape.
fn resolve_shape(
    declared: Option<(usize, usize, usize)>,
    flat: usize,
) -> SNNResult<(usize, usize, usize)> {
    match declared {
        Some((c, h, w)) => {
            if c * h * w != flat {
                return Err(SNNError::DimensionMismatch {
                    expected: format!("{c} x {h} x {w} = {}", c * h * w),
                    actual: format!("input size {flat}"),
                });
            }
            Ok((c, h, w))
        }
        None => {
            let side = (flat as f64).sqrt().round() as usize;
            if side * side != flat {
                return Err(SNNError::DimensionMismatch {
                    expected: "a square single-channel input, or an explicit shape via \
                               set_input_shape"
                        .to_string(),
                    actual: format!("{flat} neurons, which is not square"),
                });
            }
            Ok((1, side, side))
        }
    }
}

/// Sum pooling over 2-D feature maps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingSumPool2d {
    /// Pool size (height, width)
    pub pool_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// Input shape as `(channels, height, width)`.
    pub input_shape: Option<(usize, usize, usize)>,
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

    /// Declare the input's `(channels, height, width)`.
    pub fn with_input_shape(mut self, channels: usize, height: usize, width: usize) -> Self {
        self.input_shape = Some((channels, height, width));
        self
    }

    /// Set the input's `(channels, height, width)` in place.
    pub fn set_input_shape(&mut self, channels: usize, height: usize, width: usize) {
        self.input_shape = Some((channels, height, width));
    }

    /// Calculate output dimensions.
    ///
    /// A feature map smaller than the pooling window yields a single clamped
    /// window rather than underflowing: the forward pass already skips
    /// positions past the edge, so it pools whatever is there. A deep stack on
    /// a small input reaches this -- the alternative was `attempt to subtract
    /// with overflow`.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = input_h.saturating_sub(self.pool_size.0) / self.stride.0 + 1;
        let out_w = input_w.saturating_sub(self.pool_size.1) / self.stride.1 + 1;
        (out_h, out_w)
    }

    /// Gradient with respect to the input.
    ///
    /// Every position inside a window contributed equally to that window's sum,
    /// so each receives the window's gradient. Windows overlap when the stride
    /// is smaller than the pool, and those contributions add.
    pub fn backward(&self, flat: usize, output_grad: &Array3<f32>) -> SNNResult<Array3<f32>> {
        let (c, h, w) = resolve_shape(self.input_shape, flat)?;
        let (oh, ow) = self.output_size(h, w);
        let (batch, steps) = (output_grad.shape()[0], output_grad.shape()[1]);

        if output_grad.shape()[2] != c * oh * ow {
            return Err(SNNError::DimensionMismatch {
                expected: format!("output gradient width {}", c * oh * ow),
                actual: format!("{}", output_grad.shape()[2]),
            });
        }

        let mut input_grad = Array3::zeros((batch, steps, flat));
        for b in 0..batch {
            for t in 0..steps {
                for ch in 0..c {
                    for i in 0..oh {
                        for j in 0..ow {
                            let g = output_grad[[b, t, (ch * oh + i) * ow + j]];
                            if g == 0.0 {
                                continue;
                            }
                            for pi in 0..self.pool_size.0 {
                                for pj in 0..self.pool_size.1 {
                                    let (y, x) = (i * self.stride.0 + pi, j * self.stride.1 + pj);
                                    if y < h && x < w {
                                        input_grad[[b, t, (ch * h + y) * w + x]] += g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(input_grad)
    }
}

impl SpikingLayer for SpikingSumPool2d {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, flat) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        let (c, h, w) = resolve_shape(self.input_shape, flat)?;
        let (oh, ow) = self.output_size(h, w);
        let mut output = Array3::zeros((batch_size, num_steps, c * oh * ow));

        for b in 0..batch_size {
            for t in 0..num_steps {
                for ch in 0..c {
                    for i in 0..oh {
                        for j in 0..ow {
                            let mut sum = 0.0;
                            for pi in 0..self.pool_size.0 {
                                for pj in 0..self.pool_size.1 {
                                    let (y, x) = (i * self.stride.0 + pi, j * self.stride.1 + pj);
                                    if y < h && x < w {
                                        sum += input_dense[[b, t, (ch * h + y) * w + x]];
                                    }
                                }
                            }
                            output[[b, t, (ch * oh + i) * ow + j]] = sum;
                        }
                    }
                }
            }
        }

        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {}

    fn parameters(&self) -> Vec<&Array2<f32>> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        Vec::new()
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        Vec::new()
    }

    fn zero_grad(&mut self) {}
}

/// Max pooling over 2-D feature maps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingMaxPool2d {
    /// Pool size (height, width)
    pub pool_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// Input shape as `(channels, height, width)`.
    pub input_shape: Option<(usize, usize, usize)>,
    /// Flat input index that won each output position, kept for the backward
    /// pass. Indexed `[batch][step][output position]`.
    #[serde(skip)]
    pub max_indices: Option<Vec<Vec<Vec<usize>>>>,
}

impl SpikingMaxPool2d {
    /// Create a new max pooling layer
    pub fn new(pool_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            pool_size,
            stride,
            input_shape: None,
            max_indices: None,
        }
    }

    /// Declare the input's `(channels, height, width)`.
    pub fn with_input_shape(mut self, channels: usize, height: usize, width: usize) -> Self {
        self.input_shape = Some((channels, height, width));
        self
    }

    /// Set the input's `(channels, height, width)` in place.
    pub fn set_input_shape(&mut self, channels: usize, height: usize, width: usize) {
        self.input_shape = Some((channels, height, width));
    }

    /// Calculate output dimensions.
    ///
    /// A feature map smaller than the pooling window yields a single clamped
    /// window rather than underflowing: the forward pass already skips
    /// positions past the edge, so it pools whatever is there. A deep stack on
    /// a small input reaches this -- the alternative was `attempt to subtract
    /// with overflow`.
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = input_h.saturating_sub(self.pool_size.0) / self.stride.0 + 1;
        let out_w = input_w.saturating_sub(self.pool_size.1) / self.stride.1 + 1;
        (out_h, out_w)
    }

    /// Gradient with respect to the input.
    ///
    /// Only the position that won each window affected the output, so it
    /// receives the whole gradient and the rest receive none. Requires a
    /// forward pass first, since the winners are what that pass recorded.
    pub fn backward(&self, flat: usize, output_grad: &Array3<f32>) -> SNNResult<Array3<f32>> {
        let indices = self.max_indices.as_ref().ok_or_else(|| {
            SNNError::InvalidConfig(
                "max pooling backward needs a forward pass first: the winning positions are \
                 recorded there"
                    .to_string(),
            )
        })?;

        let (batch, steps, width) = (
            output_grad.shape()[0],
            output_grad.shape()[1],
            output_grad.shape()[2],
        );
        let mut input_grad = Array3::zeros((batch, steps, flat));
        for b in 0..batch.min(indices.len()) {
            for t in 0..steps.min(indices[b].len()) {
                for i in 0..width.min(indices[b][t].len()) {
                    input_grad[[b, t, indices[b][t][i]]] += output_grad[[b, t, i]];
                }
            }
        }
        Ok(input_grad)
    }
}

impl SpikingLayer for SpikingMaxPool2d {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, flat) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        let (c, h, w) = resolve_shape(self.input_shape, flat)?;
        let (oh, ow) = self.output_size(h, w);
        let out_width = c * oh * ow;

        let mut output = Array3::zeros((batch_size, num_steps, out_width));
        let mut indices = vec![vec![vec![0usize; out_width]; num_steps]; batch_size];

        for b in 0..batch_size {
            for t in 0..num_steps {
                for ch in 0..c {
                    for i in 0..oh {
                        for j in 0..ow {
                            let mut best = f32::NEG_INFINITY;
                            let mut best_idx = 0;
                            for pi in 0..self.pool_size.0 {
                                for pj in 0..self.pool_size.1 {
                                    let (y, x) = (i * self.stride.0 + pi, j * self.stride.1 + pj);
                                    if y < h && x < w {
                                        let idx = (ch * h + y) * w + x;
                                        let v = input_dense[[b, t, idx]];
                                        if v > best {
                                            best = v;
                                            best_idx = idx;
                                        }
                                    }
                                }
                            }
                            let out_idx = (ch * oh + i) * ow + j;
                            output[[b, t, out_idx]] = if best.is_finite() { best } else { 0.0 };
                            indices[b][t][out_idx] = best_idx;
                        }
                    }
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

    fn zero_grad(&mut self) {}
}

pub struct TemporalAvgPool {
    /// Window size
    pub window_size: usize,
    /// Stride
    pub stride: usize,
}

impl TemporalAvgPool {
    pub fn new(window_size: usize, stride: usize) -> Self {
        Self {
            window_size,
            stride,
        }
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
