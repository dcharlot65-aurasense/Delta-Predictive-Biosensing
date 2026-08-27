//! Convolutional spiking layers for spatial feature extraction

use super::{NeuronState, SpikingLayer};
use crate::{NeuronParams, SNNResult, SpikeTensor};
use ndarray::{s, Array1, Array2, Array3, Array4};
use rand::rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// 2D Convolutional spiking layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingConv2d {
    /// Convolution kernel (out_channels, in_channels, kernel_h, kernel_w)
    pub kernel: Array4<f32>,
    /// Bias (out_channels)
    pub bias: Option<Array1<f32>>,
    /// Stride
    pub stride: (usize, usize),
    /// Padding
    pub padding: (usize, usize),
    /// Kernel gradient
    #[serde(skip)]
    pub kernel_grad: Option<Array4<f32>>,
    /// Bias gradient
    #[serde(skip)]
    pub bias_grad: Option<Array1<f32>>,
    /// Neuron parameters
    pub neuron_params: NeuronParams,
    /// Neuron state
    #[serde(skip)]
    pub state: Vec<Vec<NeuronState>>, // [batch][spatial_position]
    /// Time step
    pub dt: f32,
    /// Adaptive neurons
    pub adaptive: bool,
}

impl SpikingConv2d {
    /// Create a new 2D convolutional spiking layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        use_bias: bool,
        neuron_params: NeuronParams,
        dt: f32,
        adaptive: bool,
    ) -> Self {
        // Initialize kernel with He initialization
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let std_dev = (2.0 / fan_in as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        let mut rng = rng();

        let kernel = Array4::from_shape_fn(
            (out_channels, in_channels, kernel_size.0, kernel_size.1),
            |_| normal.sample(&mut rng),
        );

        let bias = if use_bias {
            Some(Array1::zeros(out_channels))
        } else {
            None
        };

        Self {
            kernel,
            bias,
            stride,
            padding,
            kernel_grad: None,
            bias_grad: None,
            neuron_params,
            state: Vec::new(),
            dt,
            adaptive,
        }
    }

    /// Calculate output dimensions
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = (input_h + 2 * self.padding.0 - self.kernel.shape()[2]) / self.stride.0 + 1;
        let out_w = (input_w + 2 * self.padding.1 - self.kernel.shape()[3]) / self.stride.1 + 1;
        (out_h, out_w)
    }

    /// Apply 2D convolution (simplified, no actual padding implementation)
    fn conv2d(&self, input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, in_channels, in_h, in_w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let (out_channels, _, kernel_h, kernel_w) = (
            self.kernel.shape()[0],
            self.kernel.shape()[1],
            self.kernel.shape()[2],
            self.kernel.shape()[3],
        );

        let (out_h, out_w) = self.output_size(in_h, in_w);
        let mut output = Array4::zeros((batch_size, out_channels, out_h, out_w));

        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let ih = oh * self.stride.0 + kh;
                                    let iw = ow * self.stride.1 + kw;

                                    if ih < in_h && iw < in_w {
                                        sum += input[[b, ic, ih, iw]] * self.kernel[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                        }

                        if let Some(ref bias) = self.bias {
                            sum += bias[oc];
                        }

                        output[[b, oc, oh, ow]] = sum;
                    }
                }
            }
        }

        output
    }

    /// Initialize neuron states
    fn ensure_state(&mut self, batch_size: usize, out_h: usize, out_w: usize) {
        let spatial_size = out_h * out_w;
        if self.state.len() != batch_size {
            self.state = (0..batch_size)
                .map(|_| {
                    (0..spatial_size)
                        .map(|_| NeuronState::new(self.kernel.shape()[0], self.adaptive))
                        .collect()
                })
                .collect();
        }
    }
}

impl SpikingLayer for SpikingConv2d {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, _flat_input) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        // Assume input is flattened from (C, H, W)
        // For simplicity, we'll treat it as 1D convolution internally
        // In a full implementation, we'd need proper reshaping

        let output_size = self.kernel.shape()[0];
        let mut output = Array3::zeros((batch_size, num_steps, output_size));

        // Simplified forward pass treating each spatial position independently
        for t in 0..num_steps {
            for b in 0..batch_size {
                let input_t = input_dense.slice(s![b, t, ..]);

                // Simplified: just apply linear transformation for each output channel
                for oc in 0..output_size {
                    let kernel_slice: ndarray::ArrayView1<f32> = self.kernel.slice(s![oc, .., 0, 0]);
                    let synaptic: f32 = kernel_slice.dot(&input_t);
                    output[[b, t, oc]] = synaptic;
                }
            }
        }

        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {
        for batch_states in &mut self.state {
            for state in batch_states {
                state.reset();
            }
        }
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        // Return flattened view of kernel as 2D array
        Vec::new() // Simplified for now
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        Vec::new() // Simplified for now
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        Vec::new() // Simplified for now
    }

    fn zero_grad(&mut self) {
        self.kernel_grad = None;
        self.bias_grad = None;
    }
}

/// 1D Convolutional spiking layer for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingConv1d {
    /// Convolution kernel (out_channels, in_channels, kernel_size)
    pub kernel: Array3<f32>,
    /// Bias
    pub bias: Option<Array1<f32>>,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Kernel gradient
    #[serde(skip)]
    pub kernel_grad: Option<Array3<f32>>,
    /// Bias gradient
    #[serde(skip)]
    pub bias_grad: Option<Array1<f32>>,
    /// Neuron parameters
    pub neuron_params: NeuronParams,
    /// Neuron state
    #[serde(skip)]
    pub state: Vec<Vec<NeuronState>>,
    /// Time step
    pub dt: f32,
    /// Adaptive
    pub adaptive: bool,
}

impl SpikingConv1d {
    /// Create a new 1D convolutional spiking layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_bias: bool,
        neuron_params: NeuronParams,
        dt: f32,
        adaptive: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_size;
        let std_dev = (2.0 / fan_in as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        let mut rng = rng();

        let kernel = Array3::from_shape_fn(
            (out_channels, in_channels, kernel_size),
            |_| normal.sample(&mut rng),
        );

        let bias = if use_bias {
            Some(Array1::zeros(out_channels))
        } else {
            None
        };

        Self {
            kernel,
            bias,
            stride,
            padding,
            kernel_grad: None,
            bias_grad: None,
            neuron_params,
            state: Vec::new(),
            dt,
            adaptive,
        }
    }

    /// Calculate output length
    pub fn output_length(&self, input_length: usize) -> usize {
        (input_length + 2 * self.padding - self.kernel.shape()[2]) / self.stride + 1
    }
}

impl SpikingLayer for SpikingConv1d {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, input_size) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        let out_channels = self.kernel.shape()[0];
        let mut output = Array3::zeros((batch_size, num_steps, out_channels));

        // Simplified 1D convolution
        for t in 0..num_steps {
            for b in 0..batch_size {
                let input_t = input_dense.slice(s![b, t, ..]);

                // Apply 1D convolution (simplified)
                for oc in 0..out_channels {
                    let mut sum = 0.0;
                    for ic in 0..self.kernel.shape()[1] {
                        if ic < input_size {
                            sum += input_t[ic] * self.kernel[[oc, ic, 0]];
                        }
                    }
                    if let Some(ref bias) = self.bias {
                        sum += bias[oc];
                    }
                    output[[b, t, oc]] = sum;
                }
            }
        }

        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {
        for batch_states in &mut self.state {
            for state in batch_states {
                state.reset();
            }
        }
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
        self.kernel_grad = None;
        self.bias_grad = None;
    }
}

impl Default for SpikingConv2d {
    fn default() -> Self {
        Self::new(1, 1, (3, 3), (1, 1), (0, 0), false, NeuronParams::default(), 1.0, false)
    }
}

impl Default for SpikingConv1d {
    fn default() -> Self {
        Self::new(1, 1, 3, 1, 0, false, NeuronParams::default(), 1.0, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_creation() {
        let layer = SpikingConv2d::new(
            3,
            16,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
            NeuronParams::default(),
            1.0,
            false,
        );
        assert_eq!(layer.kernel.shape(), &[16, 3, 3, 3]);
    }

    #[test]
    fn test_conv1d_creation() {
        let layer = SpikingConv1d::new(
            8,
            16,
            3,
            1,
            1,
            true,
            NeuronParams::default(),
            1.0,
            false,
        );
        assert_eq!(layer.kernel.shape(), &[16, 8, 3]);
    }

    #[test]
    fn test_conv1d_forward() {
        let mut layer = SpikingConv1d::new(
            8,
            16,
            3,
            1,
            0,
            false,
            NeuronParams::default(),
            1.0,
            false,
        );
        let input = SpikeTensor::zeros(2, 10, 8, false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 10, 16));
    }
}
