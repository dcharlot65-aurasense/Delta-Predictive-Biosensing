//! Convolutional spiking layers for spatial feature extraction

use super::{NeuronState, SpikingLayer};
use crate::{NeuronParams, SNNError, SNNResult, SpikeTensor};
use ndarray::{Array1, Array2, Array3, Array4, s};
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
    /// Neuron state, one entry per batch item covering every output neuron.
    #[serde(skip)]
    pub state: Vec<NeuronState>,
    /// Spatial extent of the input, as `(height, width)`.
    ///
    /// A `SpikeTensor` is `(batch, steps, flat)` and cannot carry `H` and `W`,
    /// so a non-square input has to say so. When this is `None` the layer
    /// assumes a square input and infers the side from `flat / in_channels`,
    /// which covers the usual image case; a flat size that is not
    /// `in_channels * n * n` is then an error rather than a silent reshape.
    pub input_hw: Option<(usize, usize)>,
    /// Time step
    pub dt: f32,
    /// Adaptive neurons
    pub adaptive: bool,
}

// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
impl SpikingConv2d {
    // Channels, kernel, stride, padding, bias and neuron dynamics are all
    // independent; folding them into a config struct would be an API change
    // rather than a simplification.
    #[allow(clippy::too_many_arguments)]
    /// Create a new 2D convolutional spiking layer
    #[allow(clippy::too_many_arguments)]
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
            input_hw: None,
            dt,
            adaptive,
        }
    }

    /// Declare the input's spatial extent, for a non-square input.
    ///
    /// Without this the layer infers a square input from the flat size.
    pub fn with_input_shape(mut self, height: usize, width: usize) -> Self {
        self.input_hw = Some((height, width));
        self
    }

    /// Set the input's spatial extent in place.
    pub fn set_input_shape(&mut self, height: usize, width: usize) {
        self.input_hw = Some((height, width));
    }

    /// Calculate output dimensions
    pub fn output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let out_h = (input_h + 2 * self.padding.0 - self.kernel.shape()[2]) / self.stride.0 + 1;
        let out_w = (input_w + 2 * self.padding.1 - self.kernel.shape()[3]) / self.stride.1 + 1;
        (out_h, out_w)
    }
}

impl SpikingConv2d {
    /// Spatial extent of the input, from `input_hw` or inferred as square.
    fn input_shape(&self, flat: usize) -> SNNResult<(usize, usize)> {
        let in_channels = self.kernel.shape()[1];
        if in_channels == 0 || !flat.is_multiple_of(in_channels) {
            return Err(SNNError::DimensionMismatch {
                expected: format!("a multiple of in_channels {in_channels}"),
                actual: format!("input size {flat}"),
            });
        }
        let plane = flat / in_channels;

        let (h, w) = match self.input_hw {
            Some((h, w)) => {
                if h * w != plane {
                    return Err(SNNError::DimensionMismatch {
                        expected: format!("{in_channels} x {h} x {w} = {}", in_channels * h * w),
                        actual: format!("input size {flat}"),
                    });
                }
                (h, w)
            }
            None => {
                // No declared shape: assume square, and refuse to guess when
                // the plane is not a perfect square rather than truncating.
                let side = (plane as f64).sqrt().round() as usize;
                if side * side != plane {
                    return Err(SNNError::DimensionMismatch {
                        expected: "a square input, or an explicit input shape via set_input_shape"
                            .to_string(),
                        actual: format!("{plane} positions per channel, which is not square"),
                    });
                }
                (side, side)
            }
        };

        let (kh, kw) = (self.kernel.shape()[2], self.kernel.shape()[3]);
        if h + 2 * self.padding.0 < kh || w + 2 * self.padding.1 < kw {
            return Err(SNNError::DimensionMismatch {
                expected: format!("at least {kh}x{kw} after padding"),
                actual: format!("{h}x{w} with padding {:?}", self.padding),
            });
        }
        Ok((h, w))
    }

    /// The input position a kernel tap reads, or `None` inside the zero padding.
    fn source_index(
        &self,
        out_pos: (usize, usize),
        tap: (usize, usize),
        in_hw: (usize, usize),
    ) -> Option<(usize, usize)> {
        let sh = (out_pos.0 * self.stride.0 + tap.0) as isize - self.padding.0 as isize;
        let sw = (out_pos.1 * self.stride.1 + tap.1) as isize - self.padding.1 as isize;
        if sh < 0 || sw < 0 || sh as usize >= in_hw.0 || sw as usize >= in_hw.1 {
            None
        } else {
            Some((sh as usize, sw as usize))
        }
    }

    fn ensure_state(&mut self, batch_size: usize, num_neurons: usize) {
        if self.state.len() != batch_size
            || self.state.first().map(|s| s.v_mem.len()) != Some(num_neurons)
        {
            self.state = (0..batch_size)
                .map(|_| NeuronState::new(num_neurons, self.adaptive))
                .collect();
        }
    }

    /// Synaptic current for one time step, indexed `[oc * OH * OW + oh * OW + ow]`.
    /// Every `(out_index, tap_h, tap_w, in_offset)` whose tap lands inside the
    /// input, where `in_offset` is the position within one input channel plane.
    ///
    /// As in the 1-D layer, tap geometry does not depend on the batch item or
    /// the time step, so it is resolved once per forward pass rather than
    /// `out_channels * in_channels` times per output position.
    fn tap_table(
        &self,
        in_hw: (usize, usize),
        out_hw: (usize, usize),
    ) -> Vec<(usize, usize, usize, usize)> {
        let (kh, kw) = (self.kernel.shape()[2], self.kernel.shape()[3]);
        let mut taps = Vec::with_capacity(out_hw.0 * out_hw.1 * kh * kw);
        for oh in 0..out_hw.0 {
            for ow in 0..out_hw.1 {
                let out_idx = oh * out_hw.1 + ow;
                for i in 0..kh {
                    for j in 0..kw {
                        if let Some((sh, sw)) = self.source_index((oh, ow), (i, j), in_hw) {
                            taps.push((out_idx, i, j, sh * in_hw.1 + sw));
                        }
                    }
                }
            }
        }
        taps
    }

    fn convolve(
        &self,
        input_t: &ndarray::ArrayView1<f32>,
        in_hw: (usize, usize),
        out_hw: (usize, usize),
        taps: &[(usize, usize, usize, usize)],
    ) -> Array1<f32> {
        let out_channels = self.kernel.shape()[0];
        let in_channels = self.kernel.shape()[1];
        let plane = in_hw.0 * in_hw.1;
        let out_plane = out_hw.0 * out_hw.1;

        let mut current = Array1::zeros(out_channels * out_plane);
        for &(out_idx, i, j, in_off) in taps {
            for oc in 0..out_channels {
                let mut acc = 0.0;
                for ic in 0..in_channels {
                    acc += input_t[ic * plane + in_off] * self.kernel[[oc, ic, i, j]];
                }
                current[oc * out_plane + out_idx] += acc;
            }
        }
        if let Some(ref bias) = self.bias {
            for oc in 0..out_channels {
                let b = bias[oc];
                for p in 0..out_plane {
                    current[oc * out_plane + p] += b;
                }
            }
        }
        current
    }

    /// Forward pass that also records what the backward pass needs.
    pub fn forward_recording(
        &mut self,
        input: &SpikeTensor,
    ) -> SNNResult<(SpikeTensor, ConvTrace)> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, flat) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        let in_hw = self.input_shape(flat)?;
        let out_hw = self.output_size(in_hw.0, in_hw.1);
        let num_neurons = self.kernel.shape()[0] * out_hw.0 * out_hw.1;

        let mut output = Array3::zeros((batch_size, num_steps, num_neurons));
        let mut v_mem = Array3::zeros((batch_size, num_steps, num_neurons));
        let mut membrane_updated =
            ndarray::Array3::from_elem((batch_size, num_steps, num_neurons), false);

        self.ensure_state(batch_size, num_neurons);
        let taps = self.tap_table(in_hw, out_hw);

        for t in 0..num_steps {
            for b in 0..batch_size {
                let input_t = input_dense.slice(s![b, t, ..]);
                let current = self.convolve(&input_t, in_hw, out_hw, &taps);

                for n in 0..num_neurons {
                    membrane_updated[[b, t, n]] = self.state[b].refrac[n] <= 0.0;
                }

                let (spikes, v_pre) =
                    self.state[b].update_lif_recording(&current, &self.neuron_params, self.dt);
                output.slice_mut(s![b, t, ..]).assign(&spikes);
                v_mem.slice_mut(s![b, t, ..]).assign(&v_pre);
            }
        }

        Ok((
            SpikeTensor::from_dense(output, input.requires_grad),
            ConvTrace {
                v_mem,
                membrane_updated,
            },
        ))
    }

    /// Backward pass, accumulating kernel and bias gradients.
    ///
    /// Differentiates the same two decay paths as the 1-D layer and gives
    /// refractory steps no surrogate term; see [`SpikingConv1d::backward`] for
    /// the derivation, which is identical bar the extra spatial axis.
    pub fn backward(
        &mut self,
        inputs: &Array3<f32>,
        trace: &ConvTrace,
        output_grad: &Array3<f32>,
        surrogate: &dyn crate::training::surrogate::SurrogateGradient,
    ) -> SNNResult<Array3<f32>> {
        let (batch_size, num_steps, flat) =
            (inputs.shape()[0], inputs.shape()[1], inputs.shape()[2]);
        let in_hw = self.input_shape(flat)?;
        let out_hw = self.output_size(in_hw.0, in_hw.1);
        let out_channels = self.kernel.shape()[0];
        let in_channels = self.kernel.shape()[1];
        let plane = in_hw.0 * in_hw.1;
        let num_neurons = out_channels * out_hw.0 * out_hw.1;

        if output_grad.shape() != [batch_size, num_steps, num_neurons] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("output gradient {batch_size}x{num_steps}x{num_neurons}"),
                actual: format!("{:?}", output_grad.shape()),
            });
        }

        let alpha_syn = (-self.dt / self.neuron_params.tau_syn).exp();
        let alpha_mem = (-self.dt / self.neuron_params.tau_mem).exp();
        let threshold = self.neuron_params.v_threshold;

        let mut kernel_grad = Array4::zeros(self.kernel.raw_dim());
        let mut bias_grad = Array1::zeros(out_channels);
        let mut input_grad = Array3::zeros(inputs.raw_dim());
        let taps = self.tap_table(in_hw, out_hw);
        let out_plane = out_hw.0 * out_hw.1;

        for b in 0..batch_size {
            let mut carry_v = Array1::<f32>::zeros(num_neurons);
            let mut carry_syn = Array1::<f32>::zeros(num_neurons);

            for t in (0..num_steps).rev() {
                let mut d_v = carry_v.clone();
                let mut d_syn_here = Array1::<f32>::zeros(num_neurons);
                let mut next_carry_v = Array1::<f32>::zeros(num_neurons);

                for n in 0..num_neurons {
                    if trace.membrane_updated[[b, t, n]] {
                        d_v[n] += output_grad[[b, t, n]]
                            * surrogate.compute_gradient(trace.v_mem[[b, t, n]], threshold);
                        d_syn_here[n] = d_v[n] * (1.0 - alpha_mem);
                        next_carry_v[n] = d_v[n] * alpha_mem;
                    } else {
                        next_carry_v[n] = d_v[n];
                    }
                }
                carry_v = next_carry_v;

                let d_current = &d_syn_here + &carry_syn;
                carry_syn = &d_current * alpha_syn;

                let x_t = inputs.slice(s![b, t, ..]);

                // One bias contribution per (out_channel, position); the tap
                // loop below visits each of those once per kernel tap.
                for oc in 0..out_channels {
                    for p in 0..out_plane {
                        bias_grad[oc] += d_current[oc * out_plane + p];
                    }
                }

                for &(out_idx, i, j, in_off) in &taps {
                    for oc in 0..out_channels {
                        let g = d_current[oc * out_plane + out_idx];
                        if g == 0.0 {
                            continue;
                        }
                        for ic in 0..in_channels {
                            let idx = ic * plane + in_off;
                            kernel_grad[[oc, ic, i, j]] += g * x_t[idx];
                            input_grad[[b, t, idx]] += g * self.kernel[[oc, ic, i, j]];
                        }
                    }
                }
            }
        }

        match self.kernel_grad {
            Some(ref mut existing) => *existing += &kernel_grad,
            None => self.kernel_grad = Some(kernel_grad),
        }
        if self.bias.is_some() {
            match self.bias_grad {
                Some(ref mut existing) => *existing += &bias_grad,
                None => self.bias_grad = Some(bias_grad),
            }
        }

        Ok(input_grad)
    }
}

impl SpikingLayer for SpikingConv2d {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        self.forward_recording(input).map(|(out, _)| out)
    }

    fn reset_state(&mut self) {
        for state in &mut self.state {
            state.reset();
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
    /// Neuron state, one entry per batch item covering every output neuron.
    #[serde(skip)]
    pub state: Vec<NeuronState>,
    /// Time step
    pub dt: f32,
    /// Adaptive
    pub adaptive: bool,
}

impl SpikingConv1d {
    /// Create a new 1D convolutional spiking layer
    #[allow(clippy::too_many_arguments)]
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

        let kernel = Array3::from_shape_fn((out_channels, in_channels, kernel_size), |_| {
            normal.sample(&mut rng)
        });

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

/// What [`SpikingConv1d::forward_recording`] captured for the backward pass.
#[derive(Debug, Clone)]
pub struct ConvTrace {
    /// Membrane potential per step, before any spike reset.
    pub v_mem: Array3<f32>,
    /// Whether each neuron's membrane integrated on that step, or was held
    /// because the neuron was refractory.
    pub membrane_updated: ndarray::Array3<bool>,
}

impl SpikingConv1d {
    /// Number of input positions along the convolved axis.
    ///
    /// The input arrives flattened as `in_channels * length` per time step, so
    /// the length is recovered by division rather than stored. An input that is
    /// not a whole number of channels is a configuration error, not something
    /// to round away.
    fn input_length(&self, input_size: usize) -> SNNResult<usize> {
        let in_channels = self.kernel.shape()[1];
        if in_channels == 0 || !input_size.is_multiple_of(in_channels) {
            return Err(SNNError::DimensionMismatch {
                expected: format!("a multiple of in_channels {in_channels}"),
                actual: format!("input size {input_size}"),
            });
        }
        let length = input_size / in_channels;
        let kernel_size = self.kernel.shape()[2];
        if length + 2 * self.padding < kernel_size {
            return Err(SNNError::DimensionMismatch {
                expected: format!("length at least {kernel_size} after padding"),
                actual: format!("length {length} with padding {}", self.padding),
            });
        }
        Ok(length)
    }

    /// The input position a kernel tap reads, or `None` when it falls in the
    /// zero padding.
    fn source_index(&self, out_pos: usize, tap: usize, in_len: usize) -> Option<usize> {
        let shifted = (out_pos * self.stride + tap) as isize - self.padding as isize;
        if shifted < 0 || shifted as usize >= in_len {
            None
        } else {
            Some(shifted as usize)
        }
    }

    fn ensure_state(&mut self, batch_size: usize, num_neurons: usize) {
        if self.state.len() != batch_size
            || self.state.first().map(|s| s.v_mem.len()) != Some(num_neurons)
        {
            self.state = (0..batch_size)
                .map(|_| NeuronState::new(num_neurons, self.adaptive))
                .collect();
        }
    }

    /// Every `(out_pos, tap, in_pos)` whose tap lands inside the input.
    ///
    /// Which taps are in bounds depends only on the geometry -- not on the batch
    /// item or the time step -- so this is built once per forward pass. It used
    /// to be decided inside the channel loops, which asked `source_index` the
    /// same question `out_channels * in_channels` times per output position.
    fn tap_table(&self, in_len: usize, out_len: usize) -> Vec<(usize, usize, usize)> {
        let kernel_size = self.kernel.shape()[2];
        let mut taps = Vec::with_capacity(out_len * kernel_size);
        for op in 0..out_len {
            for k in 0..kernel_size {
                if let Some(ip) = self.source_index(op, k, in_len) {
                    taps.push((op, k, ip));
                }
            }
        }
        taps
    }

    /// Synaptic current for one time step, indexed `[out_channel * out_len + pos]`.
    fn convolve(
        &self,
        input_t: &ndarray::ArrayView1<f32>,
        in_len: usize,
        out_len: usize,
        taps: &[(usize, usize, usize)],
    ) -> Array1<f32> {
        let out_channels = self.kernel.shape()[0];
        let in_channels = self.kernel.shape()[1];

        let mut current = Array1::zeros(out_channels * out_len);
        for &(op, k, ip) in taps {
            for oc in 0..out_channels {
                let mut acc = 0.0;
                for ic in 0..in_channels {
                    acc += input_t[ic * in_len + ip] * self.kernel[[oc, ic, k]];
                }
                current[oc * out_len + op] += acc;
            }
        }
        if let Some(ref bias) = self.bias {
            for oc in 0..out_channels {
                let b = bias[oc];
                for op in 0..out_len {
                    current[oc * out_len + op] += b;
                }
            }
        }
        current
    }

    /// Forward pass that also records what the backward pass needs.
    pub fn forward_recording(
        &mut self,
        input: &SpikeTensor,
    ) -> SNNResult<(SpikeTensor, ConvTrace)> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, input_size) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        let in_len = self.input_length(input_size)?;
        let out_len = self.output_length(in_len);
        let out_channels = self.kernel.shape()[0];
        let num_neurons = out_channels * out_len;

        let mut output = Array3::zeros((batch_size, num_steps, num_neurons));
        let mut v_mem = Array3::zeros((batch_size, num_steps, num_neurons));
        let mut membrane_updated =
            ndarray::Array3::from_elem((batch_size, num_steps, num_neurons), false);

        self.ensure_state(batch_size, num_neurons);
        let taps = self.tap_table(in_len, out_len);

        for t in 0..num_steps {
            for b in 0..batch_size {
                let input_t = input_dense.slice(s![b, t, ..]);
                let current = self.convolve(&input_t, in_len, out_len, &taps);

                // Read before the update: a refractory neuron skips the
                // membrane update entirely, and nothing afterwards says which.
                for n in 0..num_neurons {
                    membrane_updated[[b, t, n]] = self.state[b].refrac[n] <= 0.0;
                }

                let (spikes, v_pre) =
                    self.state[b].update_lif_recording(&current, &self.neuron_params, self.dt);
                output.slice_mut(s![b, t, ..]).assign(&spikes);
                v_mem.slice_mut(s![b, t, ..]).assign(&v_pre);
            }
        }

        Ok((
            SpikeTensor::from_dense(output, input.requires_grad),
            ConvTrace {
                v_mem,
                membrane_updated,
            },
        ))
    }

    /// Backward pass, accumulating kernel and bias gradients.
    ///
    /// Returns the gradient with respect to this layer's input. The surrogate is
    /// taken as an argument rather than folded into `output_grad` by the caller,
    /// because the neuron's synaptic and membrane state couple consecutive time
    /// steps: the gradient at step `t` depends on steps after it, so the
    /// substitution has to happen inside the reverse-time loop.
    ///
    /// The same two decay paths as the recurrent layer are differentiated --
    /// `i_syn[t] = a_syn*i_syn[t-1] + I[t]` and
    /// `v[t] = a_mem*v[t-1] + i_syn[t]*(1-a_mem)` -- and refractory steps
    /// contribute no surrogate term while passing membrane gradient back
    /// unattenuated. The spike reset is treated as detached, as is standard for
    /// surrogate-gradient training.
    pub fn backward(
        &mut self,
        inputs: &Array3<f32>,
        trace: &ConvTrace,
        output_grad: &Array3<f32>,
        surrogate: &dyn crate::training::surrogate::SurrogateGradient,
    ) -> SNNResult<Array3<f32>> {
        let (batch_size, num_steps, input_size) =
            (inputs.shape()[0], inputs.shape()[1], inputs.shape()[2]);
        let in_len = self.input_length(input_size)?;
        let out_len = self.output_length(in_len);
        let out_channels = self.kernel.shape()[0];
        let in_channels = self.kernel.shape()[1];
        let num_neurons = out_channels * out_len;

        if output_grad.shape() != [batch_size, num_steps, num_neurons] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("output gradient {batch_size}x{num_steps}x{num_neurons}"),
                actual: format!("{:?}", output_grad.shape()),
            });
        }

        let alpha_syn = (-self.dt / self.neuron_params.tau_syn).exp();
        let alpha_mem = (-self.dt / self.neuron_params.tau_mem).exp();
        let threshold = self.neuron_params.v_threshold;

        let mut kernel_grad = Array3::zeros(self.kernel.raw_dim());
        let mut bias_grad = Array1::zeros(out_channels);
        let mut input_grad = Array3::zeros(inputs.raw_dim());
        let taps = self.tap_table(in_len, out_len);

        for b in 0..batch_size {
            let mut carry_v = Array1::<f32>::zeros(num_neurons);
            let mut carry_syn = Array1::<f32>::zeros(num_neurons);

            for t in (0..num_steps).rev() {
                let mut d_v = carry_v.clone();
                let mut d_syn_here = Array1::<f32>::zeros(num_neurons);
                let mut next_carry_v = Array1::<f32>::zeros(num_neurons);

                for n in 0..num_neurons {
                    if trace.membrane_updated[[b, t, n]] {
                        d_v[n] += output_grad[[b, t, n]]
                            * surrogate.compute_gradient(trace.v_mem[[b, t, n]], threshold);
                        d_syn_here[n] = d_v[n] * (1.0 - alpha_mem);
                        next_carry_v[n] = d_v[n] * alpha_mem;
                    } else {
                        next_carry_v[n] = d_v[n];
                    }
                }
                carry_v = next_carry_v;

                let d_current = &d_syn_here + &carry_syn;
                carry_syn = &d_current * alpha_syn;

                // I[oc, op] = sum_{ic,k} x[ic, op*stride + k - pad] * K[oc,ic,k] + bias[oc]
                let x_t = inputs.slice(s![b, t, ..]);

                // Bias first, and separately: it takes one contribution per
                // (out_channel, position), whereas the tap loop below visits
                // each of those once per kernel tap.
                for oc in 0..out_channels {
                    for op in 0..out_len {
                        bias_grad[oc] += d_current[oc * out_len + op];
                    }
                }

                for &(op, k, ip) in &taps {
                    for oc in 0..out_channels {
                        let g = d_current[oc * out_len + op];
                        if g == 0.0 {
                            continue;
                        }
                        for ic in 0..in_channels {
                            kernel_grad[[oc, ic, k]] += g * x_t[ic * in_len + ip];
                            input_grad[[b, t, ic * in_len + ip]] += g * self.kernel[[oc, ic, k]];
                        }
                    }
                }
            }
        }

        match self.kernel_grad {
            Some(ref mut existing) => *existing += &kernel_grad,
            None => self.kernel_grad = Some(kernel_grad),
        }
        if self.bias.is_some() {
            match self.bias_grad {
                Some(ref mut existing) => *existing += &bias_grad,
                None => self.bias_grad = Some(bias_grad),
            }
        }

        Ok(input_grad)
    }
}

impl SpikingLayer for SpikingConv1d {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        self.forward_recording(input).map(|(out, _)| out)
    }

    fn reset_state(&mut self) {
        for state in &mut self.state {
            state.reset();
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
        Self::new(
            1,
            1,
            (3, 3),
            (1, 1),
            (0, 0),
            false,
            NeuronParams::default(),
            1.0,
            false,
        )
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
        let layer = SpikingConv1d::new(8, 16, 3, 1, 1, true, NeuronParams::default(), 1.0, false);
        assert_eq!(layer.kernel.shape(), &[16, 8, 3]);
    }

    #[test]
    fn test_conv1d_forward() {
        // 8 channels of a length-10 signal, so 80 values per time step.
        // A width-3 kernel with stride 1 and no padding leaves 8 positions,
        // one neuron per (out_channel, position).
        let mut layer =
            SpikingConv1d::new(8, 16, 3, 1, 0, false, NeuronParams::default(), 1.0, false);
        let input = SpikeTensor::zeros(2, 10, 8 * 10, false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 10, 16 * 8));
    }

    /// Padding must preserve the length, which is the point of `padding = 1`
    /// with a width-3 kernel.
    #[test]
    fn conv1d_same_padding_preserves_length() {
        let mut layer =
            SpikingConv1d::new(4, 6, 3, 1, 1, true, NeuronParams::default(), 1.0, false);
        let input = SpikeTensor::zeros(1, 3, 4 * 12, false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 3, 6 * 12));
    }

    /// An input that is not a whole number of channels is a configuration
    /// error, and a signal shorter than the kernel cannot be convolved at all.
    #[test]
    fn conv1d_rejects_impossible_shapes() {
        let mut layer =
            SpikingConv1d::new(4, 6, 3, 1, 0, true, NeuronParams::default(), 1.0, false);
        assert!(layer.forward(&SpikeTensor::zeros(1, 2, 10, false)).is_err());
        assert!(
            layer
                .forward(&SpikeTensor::zeros(1, 2, 4 * 2, false))
                .is_err()
        );
    }

    /// The kernel must actually slide. Under the previous implementation only
    /// tap 0 was ever read, so stride, padding and every other tap were inert;
    /// changing a non-zero tap would not have changed the output.
    #[test]
    fn conv1d_reads_every_kernel_tap() {
        let mut layer =
            SpikingConv1d::new(1, 1, 3, 1, 0, false, NeuronParams::default(), 1.0, false);
        layer.kernel.fill(0.0);
        // Only the last tap is non-zero, so the output must track the input
        // shifted by two positions -- and be identically zero if taps are
        // ignored.
        layer.kernel[[0, 0, 2]] = 5.0;

        let mut dense = Array3::zeros((1, 1, 6));
        dense[[0, 0, 5]] = 1.0;
        let input = SpikeTensor::from_dense(dense, false);

        let (_, trace) = layer.forward_recording(&input).unwrap();
        // Output position 3 reads input positions 3,4,5; only tap 2 is live.
        let v = &trace.v_mem;
        assert!(v[[0, 0, 3]] > 0.0, "last tap contributed nothing: {v:?}");
        for p in 0..3 {
            assert_eq!(v[[0, 0, p]], 0.0, "position {p} should be untouched");
        }
    }

    /// Padding must shift where each output position reads from, not merely
    /// widen the output. Checking the output *shape* does not catch this:
    /// `output_length` computes the shape independently, so an implementation
    /// that ignored the padding offset would still produce a correctly sized
    /// output while reading the wrong inputs.
    #[test]
    fn conv1d_padding_shifts_the_read_window() {
        let mut layer =
            SpikingConv1d::new(1, 1, 3, 1, 1, false, NeuronParams::default(), 1.0, false);
        layer.kernel.fill(0.0);
        // Only tap 0 is live, so output position p reads input position p - 1.
        layer.kernel[[0, 0, 0]] = 3.0;

        let mut dense = Array3::zeros((1, 1, 5));
        dense[[0, 0, 0]] = 1.0;
        let (_, trace) = layer
            .forward_recording(&SpikeTensor::from_dense(dense, false))
            .unwrap();

        // Position 0 reads input position -1, which is padding: zero.
        assert_eq!(
            trace.v_mem[[0, 0, 0]],
            0.0,
            "output 0 read real input instead of the left pad"
        );
        // Position 1 reads input position 0, which is the value that was set.
        assert!(
            trace.v_mem[[0, 0, 1]] > 0.0,
            "output 1 did not read input 0: {:?}",
            trace.v_mem
        );
    }

    /// The tap-table convolution must agree with a direct transcription of the
    /// definition.
    ///
    /// The optimised form hoists the bounds check out of the channel loops and
    /// accumulates per tap rather than per output position, which changes the
    /// order of the floating-point additions. This pins it against a naive
    /// reference written straight from the convolution definition, so a future
    /// restructuring cannot quietly change what the layer computes.
    #[test]
    fn conv1d_matches_a_naive_reference() {
        let (in_ch, out_ch, k, stride, pad, in_len) =
            (3usize, 4usize, 3usize, 2usize, 1usize, 9usize);
        let mut layer = SpikingConv1d::new(
            in_ch,
            out_ch,
            k,
            stride,
            pad,
            true,
            NeuronParams::default(),
            1.0,
            false,
        );
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                for kk in 0..k {
                    layer.kernel[[oc, ic, kk]] =
                        0.3 + 0.17 * (oc as f32) - 0.11 * ((ic * 3 + kk) as f32);
                }
            }
        }
        if let Some(ref mut b) = layer.bias {
            for oc in 0..out_ch {
                b[oc] = 0.07 * (oc as f32) - 0.02;
            }
        }

        let input: Vec<f32> = (0..in_ch * in_len)
            .map(|i| 0.4 * ((i % 5) as f32) - 0.3)
            .collect();
        let out_len = layer.output_length(in_len);

        // Naive reference, straight from the definition.
        let mut expected = vec![0.0f32; out_ch * out_len];
        for oc in 0..out_ch {
            for op in 0..out_len {
                let mut sum = layer.bias.as_ref().map_or(0.0, |b| b[oc]);
                for ic in 0..in_ch {
                    for kk in 0..k {
                        let pos = (op * stride + kk) as isize - pad as isize;
                        if pos >= 0 && (pos as usize) < in_len {
                            sum += input[ic * in_len + pos as usize] * layer.kernel[[oc, ic, kk]];
                        }
                    }
                }
                expected[oc * out_len + op] = sum;
            }
        }

        let arr = Array1::from(input.clone());
        let taps = layer.tap_table(in_len, out_len);
        let got = layer.convolve(&arr.view(), in_len, out_len, &taps);

        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-5 * e.abs().max(1.0),
                "position {i}: optimised {g} vs naive {e}"
            );
        }
    }

    /// The same, for the 2-D layer, on a non-square input with stride and
    /// padding both in play.
    #[test]
    fn conv2d_matches_a_naive_reference() {
        let (in_ch, out_ch, h, w) = (2usize, 3usize, 5usize, 6usize);
        let (kh, kw, sh, sw, ph, pw) = (3usize, 2usize, 2usize, 1usize, 1usize, 0usize);
        let mut layer = SpikingConv2d::new(
            in_ch,
            out_ch,
            (kh, kw),
            (sh, sw),
            (ph, pw),
            true,
            NeuronParams::default(),
            1.0,
            false,
        );
        layer.set_input_shape(h, w);
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                for i in 0..kh {
                    for j in 0..kw {
                        layer.kernel[[oc, ic, i, j]] =
                            0.25 + 0.13 * (oc as f32) - 0.08 * ((ic + i * 2 + j) as f32);
                    }
                }
            }
        }
        if let Some(ref mut b) = layer.bias {
            for oc in 0..out_ch {
                b[oc] = 0.05 * (oc as f32) - 0.01;
            }
        }

        let plane = h * w;
        let input: Vec<f32> = (0..in_ch * plane)
            .map(|i| 0.3 * ((i % 7) as f32) - 0.5)
            .collect();
        let (oh, ow) = layer.output_size(h, w);

        let mut expected = vec![0.0f32; out_ch * oh * ow];
        for oc in 0..out_ch {
            for y in 0..oh {
                for x in 0..ow {
                    let mut sum = layer.bias.as_ref().map_or(0.0, |b| b[oc]);
                    for ic in 0..in_ch {
                        for i in 0..kh {
                            for j in 0..kw {
                                let py = (y * sh + i) as isize - ph as isize;
                                let px = (x * sw + j) as isize - pw as isize;
                                if py >= 0 && px >= 0 && (py as usize) < h && (px as usize) < w {
                                    let idx = ic * plane + py as usize * w + px as usize;
                                    sum += input[idx] * layer.kernel[[oc, ic, i, j]];
                                }
                            }
                        }
                    }
                    expected[(oc * oh + y) * ow + x] = sum;
                }
            }
        }

        let arr = Array1::from(input.clone());
        let taps = layer.tap_table((h, w), (oh, ow));
        let got = layer.convolve(&arr.view(), (h, w), (oh, ow), &taps);

        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-5 * e.abs().max(1.0),
                "position {i}: optimised {g} vs naive {e}"
            );
        }
    }

    // ---- backward-pass validation -------------------------------------
    //
    // Same argument as for the recurrent layer: surrogate-gradient training is
    // not the derivative of a single smooth function, so finite differences
    // against the spiking forward would check the wrong thing. Forward-mode
    // propagation through the same linearised dynamics shares no code with the
    // reverse pass and must agree exactly.

    use crate::training::surrogate::{FastSigmoidSurrogate, SurrogateGradient};

    /// Directional derivative of `sum(output_grad * spikes)` with respect to one
    /// kernel entry or bias, by forward-mode propagation.
    #[allow(clippy::too_many_arguments)]
    fn conv_forward_mode(
        layer: &SpikingConv1d,
        inputs: &Array3<f32>,
        trace: &ConvTrace,
        output_grad: &Array3<f32>,
        surrogate: &dyn SurrogateGradient,
        wrt_kernel: Option<(usize, usize, usize)>,
        wrt_bias: Option<usize>,
    ) -> f32 {
        let (batch, steps, input_size) = (inputs.shape()[0], inputs.shape()[1], inputs.shape()[2]);
        let in_len = layer.input_length(input_size).unwrap();
        let out_len = layer.output_length(in_len);
        let out_channels = layer.kernel.shape()[0];
        let neurons = out_channels * out_len;

        let a_syn = (-layer.dt / layer.neuron_params.tau_syn).exp();
        let a_mem = (-layer.dt / layer.neuron_params.tau_mem).exp();
        let threshold = layer.neuron_params.v_threshold;

        let mut total = 0.0;
        for b in 0..batch {
            let mut d_syn = vec![0.0f32; neurons];
            let mut d_v = vec![0.0f32; neurons];

            for t in 0..steps {
                for oc in 0..out_channels {
                    for op in 0..out_len {
                        let n = oc * out_len + op;

                        // Tangent of this neuron's input current.
                        let mut d_current = 0.0;
                        if let Some((ko, ki, kk)) = wrt_kernel
                            && ko == oc
                            && let Some(ip) = layer.source_index(op, kk, in_len)
                        {
                            d_current += inputs[[b, t, ki * in_len + ip]];
                        }
                        if let Some(bo) = wrt_bias
                            && bo == oc
                        {
                            d_current += 1.0;
                        }

                        d_syn[n] = a_syn * d_syn[n] + d_current;
                        let d_s = if trace.membrane_updated[[b, t, n]] {
                            d_v[n] = a_mem * d_v[n] + d_syn[n] * (1.0 - a_mem);
                            surrogate.compute_gradient(trace.v_mem[[b, t, n]], threshold) * d_v[n]
                        } else {
                            0.0
                        };
                        total += output_grad[[b, t, n]] * d_s;
                    }
                }
            }
        }
        total
    }

    fn conv_fixture() -> (SpikingConv1d, Array3<f32>, Array3<f32>) {
        let (batch, steps, in_channels, in_len) = (2, 5, 2, 7);
        let mut layer = SpikingConv1d::new(
            in_channels,
            3,
            3,
            2,
            1,
            true,
            NeuronParams::default(),
            1.0,
            false,
        );
        for oc in 0..3 {
            for ic in 0..in_channels {
                for k in 0..3 {
                    layer.kernel[[oc, ic, k]] = 0.5 + 0.11 * (oc as f32) - 0.07 * ((ic + k) as f32);
                }
            }
        }
        if let Some(ref mut b) = layer.bias {
            for oc in 0..3 {
                b[oc] = 0.05 * (oc as f32);
            }
        }

        let inputs = Array3::from_shape_fn((batch, steps, in_channels * in_len), |(b, t, i)| {
            (((b * 5 + t * 3 + i) % 4) as f32) * 0.6
        });
        let out_len = layer.output_length(in_len);
        let output_grad = Array3::from_shape_fn((batch, steps, 3 * out_len), |(b, t, n)| {
            0.4 - 0.15 * (((b + t + n) % 3) as f32)
        });
        (layer, inputs, output_grad)
    }

    #[test]
    fn conv1d_backward_matches_forward_mode() {
        let (mut layer, inputs, output_grad) = conv_fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);
        let tensor = SpikeTensor::from_dense(inputs.clone(), false);

        let (spikes, trace) = layer.forward_recording(&tensor).unwrap();
        let fired: f32 = spikes.to_dense().iter().sum();
        let refractory = trace.membrane_updated.iter().filter(|u| !**u).count();
        assert!(fired > 0.0, "fixture never spiked");
        assert!(refractory > 0, "fixture never entered a refractory step");

        layer
            .backward(&inputs, &trace, &output_grad, &surrogate)
            .unwrap();
        let kernel_grad = layer.kernel_grad.clone().unwrap();
        let bias_grad = layer.bias_grad.clone().unwrap();

        let (oc_n, ic_n, k_n) = (
            layer.kernel.shape()[0],
            layer.kernel.shape()[1],
            layer.kernel.shape()[2],
        );
        for oc in 0..oc_n {
            for ic in 0..ic_n {
                for k in 0..k_n {
                    let expected = conv_forward_mode(
                        &layer,
                        &inputs,
                        &trace,
                        &output_grad,
                        &surrogate,
                        Some((oc, ic, k)),
                        None,
                    );
                    let got = kernel_grad[[oc, ic, k]];
                    assert!(
                        (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                        "kernel[{oc},{ic},{k}]: reverse {got} vs forward {expected}"
                    );
                }
            }
            let expected = conv_forward_mode(
                &layer,
                &inputs,
                &trace,
                &output_grad,
                &surrogate,
                None,
                Some(oc),
            );
            assert!(
                (bias_grad[oc] - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "bias[{oc}]: reverse {} vs forward {expected}",
                bias_grad[oc]
            );
        }
    }

    /// Padding contributes no gradient: taps reading the zero pad must not
    /// accumulate into the kernel, and the input gradient must stay in bounds.
    #[test]
    fn conv1d_input_gradient_has_the_input_shape() {
        let (mut layer, inputs, output_grad) = conv_fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);
        let (_, trace) = layer
            .forward_recording(&SpikeTensor::from_dense(inputs.clone(), false))
            .unwrap();

        let grad = layer
            .backward(&inputs, &trace, &output_grad, &surrogate)
            .unwrap();
        assert_eq!(grad.shape(), inputs.shape());
        assert!(grad.iter().all(|g| g.is_finite()));
        assert!(
            grad.iter().any(|g| g.abs() > 0.0),
            "no gradient reached the input"
        );
    }

    // ---- Conv2d backward validation -----------------------------------

    /// Forward-mode directional derivative for one Conv2d kernel entry or bias.
    fn conv2d_forward_mode(
        layer: &SpikingConv2d,
        inputs: &Array3<f32>,
        trace: &ConvTrace,
        output_grad: &Array3<f32>,
        surrogate: &dyn SurrogateGradient,
        wrt_kernel: Option<(usize, usize, usize, usize)>,
        wrt_bias: Option<usize>,
    ) -> f32 {
        let (batch, steps, flat) = (inputs.shape()[0], inputs.shape()[1], inputs.shape()[2]);
        let in_hw = layer.input_shape(flat).unwrap();
        let out_hw = layer.output_size(in_hw.0, in_hw.1);
        let out_channels = layer.kernel.shape()[0];
        let plane = in_hw.0 * in_hw.1;
        let neurons = out_channels * out_hw.0 * out_hw.1;

        let a_syn = (-layer.dt / layer.neuron_params.tau_syn).exp();
        let a_mem = (-layer.dt / layer.neuron_params.tau_mem).exp();
        let threshold = layer.neuron_params.v_threshold;

        let mut total = 0.0;
        for b in 0..batch {
            let mut d_syn = vec![0.0f32; neurons];
            let mut d_v = vec![0.0f32; neurons];

            for t in 0..steps {
                for oc in 0..out_channels {
                    for oh in 0..out_hw.0 {
                        for ow in 0..out_hw.1 {
                            let n = (oc * out_hw.0 + oh) * out_hw.1 + ow;

                            let mut d_current = 0.0;
                            if let Some((ko, ki, ki_h, ki_w)) = wrt_kernel
                                && ko == oc
                                && let Some((sh, sw)) =
                                    layer.source_index((oh, ow), (ki_h, ki_w), in_hw)
                            {
                                d_current += inputs[[b, t, ki * plane + sh * in_hw.1 + sw]];
                            }
                            if let Some(bo) = wrt_bias
                                && bo == oc
                            {
                                d_current += 1.0;
                            }

                            d_syn[n] = a_syn * d_syn[n] + d_current;
                            let d_s = if trace.membrane_updated[[b, t, n]] {
                                d_v[n] = a_mem * d_v[n] + d_syn[n] * (1.0 - a_mem);
                                surrogate.compute_gradient(trace.v_mem[[b, t, n]], threshold)
                                    * d_v[n]
                            } else {
                                0.0
                            };
                            total += output_grad[[b, t, n]] * d_s;
                        }
                    }
                }
            }
        }
        total
    }

    fn conv2d_fixture() -> (SpikingConv2d, Array3<f32>, Array3<f32>) {
        // Deliberately not square: a height/width transposition anywhere in the
        // indexing is invisible on a square input.
        let (batch, steps, in_channels, h, w) = (2, 4, 2, 5, 6);
        let mut layer = SpikingConv2d::new(
            in_channels,
            3,
            (3, 3),
            (2, 2),
            (1, 1),
            true,
            NeuronParams::default(),
            1.0,
            false,
        );
        layer.set_input_shape(h, w);

        for oc in 0..3 {
            for ic in 0..in_channels {
                for i in 0..3 {
                    for j in 0..3 {
                        layer.kernel[[oc, ic, i, j]] =
                            0.4 + 0.09 * (oc as f32) - 0.05 * ((ic + i + j) as f32);
                    }
                }
            }
        }
        if let Some(ref mut b) = layer.bias {
            for oc in 0..3 {
                b[oc] = 0.03 * (oc as f32);
            }
        }

        let inputs = Array3::from_shape_fn((batch, steps, in_channels * h * w), |(b, t, i)| {
            (((b * 3 + t * 5 + i) % 4) as f32) * 0.7
        });
        let (oh, ow) = layer.output_size(h, w);
        let output_grad = Array3::from_shape_fn((batch, steps, 3 * oh * ow), |(b, t, n)| {
            0.35 - 0.12 * (((b + t + n) % 3) as f32)
        });
        (layer, inputs, output_grad)
    }

    #[test]
    fn conv2d_backward_matches_forward_mode() {
        let (mut layer, inputs, output_grad) = conv2d_fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);

        let (spikes, trace) = layer
            .forward_recording(&SpikeTensor::from_dense(inputs.clone(), false))
            .unwrap();
        let fired: f32 = spikes.to_dense().iter().sum();
        let refractory = trace.membrane_updated.iter().filter(|u| !**u).count();
        assert!(fired > 0.0, "fixture never spiked");
        assert!(refractory > 0, "fixture never entered a refractory step");

        layer
            .backward(&inputs, &trace, &output_grad, &surrogate)
            .unwrap();
        let kernel_grad = layer.kernel_grad.clone().unwrap();
        let bias_grad = layer.bias_grad.clone().unwrap();

        let s = layer.kernel.shape().to_vec();
        for oc in 0..s[0] {
            for ic in 0..s[1] {
                for i in 0..s[2] {
                    for j in 0..s[3] {
                        let expected = conv2d_forward_mode(
                            &layer,
                            &inputs,
                            &trace,
                            &output_grad,
                            &surrogate,
                            Some((oc, ic, i, j)),
                            None,
                        );
                        let got = kernel_grad[[oc, ic, i, j]];
                        assert!(
                            (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                            "kernel[{oc},{ic},{i},{j}]: reverse {got} vs forward {expected}"
                        );
                    }
                }
            }
            let expected = conv2d_forward_mode(
                &layer,
                &inputs,
                &trace,
                &output_grad,
                &surrogate,
                None,
                Some(oc),
            );
            assert!(
                (bias_grad[oc] - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "bias[{oc}]: reverse {} vs forward {expected}",
                bias_grad[oc]
            );
        }
    }

    /// Padding must shift the 2-D read window, in both axes. As for the 1-D
    /// layer, the output shape alone cannot catch an ignored offset.
    #[test]
    fn conv2d_padding_shifts_the_read_window() {
        let mut layer = SpikingConv2d::new(
            1,
            1,
            (3, 3),
            (1, 1),
            (1, 1),
            false,
            NeuronParams::default(),
            1.0,
            false,
        );
        layer.set_input_shape(4, 4);
        layer.kernel.fill(0.0);
        // Only the top-left tap is live, so output (i,j) reads input (i-1,j-1).
        layer.kernel[[0, 0, 0, 0]] = 4.0;

        let mut dense = Array3::zeros((1, 1, 16));
        dense[[0, 0, 0]] = 1.0; // input position (0, 0)
        let (_, trace) = layer
            .forward_recording(&SpikeTensor::from_dense(dense, false))
            .unwrap();

        // Output (0,0) reads (-1,-1): padding, so nothing.
        assert_eq!(
            trace.v_mem[[0, 0, 0]],
            0.0,
            "output (0,0) read real input instead of the corner pad"
        );
        // Output (1,1) reads (0,0), the value that was set. Width is 4.
        let (row, col, width) = (1usize, 1usize, 4usize);
        assert!(
            trace.v_mem[[0, 0, row * width + col]] > 0.0,
            "output (1,1) did not read input (0,0): {:?}",
            trace.v_mem
        );
    }

    /// Stride must move the read window by more than one position. Like
    /// padding, this cannot be caught by the gradient check: `source_index` is
    /// shared by the forward pass, the backward pass and the forward-mode
    /// reference, so an error there moves all three together and they still
    /// agree with each other.
    #[test]
    fn conv2d_stride_advances_the_read_window() {
        let mut layer = SpikingConv2d::new(
            1,
            1,
            (2, 2),
            (2, 2),
            (0, 0),
            false,
            NeuronParams::default(),
            1.0,
            false,
        );
        layer.set_input_shape(4, 4);
        layer.kernel.fill(0.0);
        layer.kernel[[0, 0, 0, 0]] = 4.0; // top-left tap only

        let (oh, ow) = layer.output_size(4, 4);
        assert_eq!((oh, ow), (2, 2));

        // Both axes must advance, so both are probed: a stride dropped on only
        // one axis survives a test that looks at the other.
        for (label, input_rc, out_rc) in [
            ("width", (0usize, 2usize), (0usize, 1usize)),
            ("height", (2, 0), (1, 0)),
        ] {
            let mut dense = Array3::zeros((1, 1, 16));
            dense[[0, 0, input_rc.0 * 4 + input_rc.1]] = 1.0;
            let (_, trace) = layer
                .forward_recording(&SpikeTensor::from_dense(dense, false))
                .unwrap();

            let out_idx = out_rc.0 * ow + out_rc.1;
            assert!(
                trace.v_mem[[0, 0, out_idx]] > 0.0,
                "{label}: output {out_rc:?} did not read input {input_rc:?}; \
                 stride is not advancing on that axis: {:?}",
                trace.v_mem
            );
            assert_eq!(
                trace.v_mem[[0, 0, 0]],
                0.0,
                "{label}: output (0,0) should read input (0,0) only"
            );
            layer.reset_state();
        }
    }

    /// A non-square input needs a declared shape; guessing is refused.
    #[test]
    fn conv2d_rejects_an_ambiguous_shape() {
        let mut layer = SpikingConv2d::new(
            1,
            2,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
            NeuronParams::default(),
            1.0,
            false,
        );
        // 12 is not a perfect square, and no shape was declared.
        assert!(layer.forward(&SpikeTensor::zeros(1, 2, 12, false)).is_err());
        // Declared 3x4, it works.
        layer.set_input_shape(3, 4);
        assert!(layer.forward(&SpikeTensor::zeros(1, 2, 12, false)).is_ok());
    }
}
