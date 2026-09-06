//! Recurrent spiking layers for temporal processing

use super::{NeuronState, SpikingLayer};
use crate::{NeuronParams, SNNError, SNNResult, SpikeTensor};
use ndarray::{Array1, Array2, Array3, s};
use rand::rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Recurrent Spiking Neural Network layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingRNN {
    /// Input weight matrix (hidden_size, input_size)
    pub w_input: Array2<f32>,
    /// Recurrent weight matrix (hidden_size, hidden_size)
    pub w_recurrent: Array2<f32>,
    /// Bias
    pub bias: Option<Array1<f32>>,
    /// Weight gradients
    #[serde(skip)]
    pub w_input_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_recurrent_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub bias_grad: Option<Array1<f32>>,
    /// Neuron parameters
    pub neuron_params: NeuronParams,
    /// Neuron state
    #[serde(skip)]
    pub state: Vec<NeuronState>,
    /// Previous output (for recurrence)
    #[serde(skip)]
    pub prev_output: Option<Array2<f32>>,
    /// Time step
    pub dt: f32,
    /// Adaptive
    pub adaptive: bool,
}

impl SpikingRNN {
    /// Create a new recurrent spiking layer
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        use_bias: bool,
        neuron_params: NeuronParams,
        dt: f32,
        adaptive: bool,
    ) -> Self {
        let std_input = (2.0 / (input_size + hidden_size) as f32).sqrt();
        let std_recurrent = (1.0 / hidden_size as f32).sqrt();

        let normal_input = Normal::new(0.0, std_input).unwrap();
        let normal_recurrent = Normal::new(0.0, std_recurrent).unwrap();
        let mut rng = rng();

        let w_input =
            Array2::from_shape_fn((hidden_size, input_size), |_| normal_input.sample(&mut rng));

        let w_recurrent = Array2::from_shape_fn((hidden_size, hidden_size), |_| {
            normal_recurrent.sample(&mut rng)
        });

        let bias = if use_bias {
            Some(Array1::zeros(hidden_size))
        } else {
            None
        };

        Self {
            w_input,
            w_recurrent,
            bias,
            w_input_grad: None,
            w_recurrent_grad: None,
            bias_grad: None,
            neuron_params,
            state: Vec::new(),
            prev_output: None,
            dt,
            adaptive,
        }
    }

    /// Ensure state is initialized
    fn ensure_state(&mut self, batch_size: usize) {
        let hidden_size = self.w_input.shape()[0];
        if self.state.len() != batch_size {
            self.state = (0..batch_size)
                .map(|_| NeuronState::new(hidden_size, self.adaptive))
                .collect();
        }

        // `prev_output` is checked on its own terms, not as a side effect of
        // the neuron state being resized.
        //
        // Both `forward` and `reset_state` set it to None while leaving `state`
        // at its existing length, so the batch-size branch above would not fire
        // and `prev_output` stayed None -- which `forward_step` then unwrapped.
        // The first forward pass survived only because `state` was still empty;
        // every pass after it panicked, so a recurrent layer could be run
        // exactly once.
        let needs_init = self
            .prev_output
            .as_ref()
            .is_none_or(|p| p.dim() != (batch_size, hidden_size));
        if needs_init {
            self.prev_output = Some(Array2::zeros((batch_size, hidden_size)));
        }
    }

    /// Forward pass for single time step
    fn forward_step(&mut self, input: &Array2<f32>) -> SNNResult<Array2<f32>> {
        let batch_size = input.shape()[0];
        let hidden_size = self.w_input.shape()[0];
        self.ensure_state(batch_size);

        let prev_out = self
            .prev_output
            .as_ref()
            .expect("ensure_state guarantees prev_output is present");
        let mut output = Array2::zeros((batch_size, hidden_size));

        for b in 0..batch_size {
            // Compute total input: W_in * x + W_rec * h_prev
            let input_contrib = self.w_input.dot(&input.slice(s![b, ..]));
            let recurrent_contrib = self.w_recurrent.dot(&prev_out.slice(s![b, ..]));
            let mut total_input = input_contrib + recurrent_contrib;

            if let Some(ref bias) = self.bias {
                total_input += bias;
            }

            // Update neuron dynamics
            let spikes = self.state[b].update_lif(&total_input, &self.neuron_params, self.dt);
            output.slice_mut(s![b, ..]).assign(&spikes);
        }

        self.prev_output = Some(output.clone());
        Ok(output)
    }

    /// Forward pass that also records what backpropagation through time needs.
    ///
    /// Returns the output spikes, the membrane potential each neuron reached
    /// *before* its spike reset, and the synaptic current at every step. The
    /// reset in [`NeuronState::update_lif`] destroys the pre-threshold potential
    /// the surrogate takes as its input, so a plain forward pass cannot be
    /// differentiated after the fact.
    pub fn forward_recording(
        &mut self,
        input: &SpikeTensor,
    ) -> SNNResult<(SpikeTensor, RecurrentTrace)> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, input_size) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        if input_size != self.w_input.shape()[1] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("input size {}", self.w_input.shape()[1]),
                actual: format!("input size {}", input_size),
            });
        }

        let hidden_size = self.w_input.shape()[0];
        let mut output = Array3::zeros((batch_size, num_steps, hidden_size));
        let mut v_mem = Array3::zeros((batch_size, num_steps, hidden_size));
        let mut membrane_updated =
            ndarray::Array3::from_elem((batch_size, num_steps, hidden_size), false);

        self.ensure_state(batch_size);
        self.prev_output = Some(Array2::zeros((batch_size, hidden_size)));

        for t in 0..num_steps {
            let prev_out = self
                .prev_output
                .as_ref()
                .expect("set immediately above")
                .clone();
            let mut step_out = Array2::zeros((batch_size, hidden_size));

            for b in 0..batch_size {
                let mut total = self.w_input.dot(&input_dense.slice(s![b, t, ..]));
                total += &self.w_recurrent.dot(&prev_out.slice(s![b, ..]));
                if let Some(ref bias) = self.bias {
                    total += bias;
                }

                // Read the refractory counters before the update: a neuron
                // still refractory skips the membrane update entirely and
                // cannot spike, so the backward pass has to treat that step
                // differently. Nothing after the update reveals which neurons
                // were in that branch.
                for n in 0..hidden_size {
                    membrane_updated[[b, t, n]] = self.state[b].refrac[n] <= 0.0;
                }

                let (spikes, v_pre) =
                    self.state[b].update_lif_recording(&total, &self.neuron_params, self.dt);
                step_out.slice_mut(s![b, ..]).assign(&spikes);
                v_mem.slice_mut(s![b, t, ..]).assign(&v_pre);
            }

            output.slice_mut(s![.., t, ..]).assign(&step_out);
            self.prev_output = Some(step_out);
        }

        Ok((
            SpikeTensor::from_dense(output, input.requires_grad),
            RecurrentTrace {
                v_mem,
                membrane_updated,
            },
        ))
    }

    /// Backpropagation through time.
    ///
    /// Accumulates `dL/dW_input`, `dL/dW_recurrent` and `dL/dbias`, and returns
    /// `dL/dinput` for the layer below.
    ///
    /// Unlike a feedforward layer, the surrogate cannot be applied by the caller
    /// before the call: a spike at step `t` reaches the loss both directly and
    /// through every later step via `w_recurrent`, so the substitution has to
    /// happen inside the reverse-time loop. The surrogate is therefore taken as
    /// an argument.
    ///
    /// Both decay paths of the neuron are differentiated, not just the direct
    /// one. Writing `a_syn = exp(-dt/tau_syn)` and `a_mem = exp(-dt/tau_mem)`,
    /// the forward dynamics are
    ///
    /// ```text
    /// i_syn[t] = a_syn * i_syn[t-1] + I[t]
    /// v[t]     = a_mem * v[t-1] + i_syn[t] * (1 - a_mem)
    /// s[t]     = H(v[t] - threshold)
    /// ```
    ///
    /// so gradient flows back through `v[t-1]` and `i_syn[t-1]` as well as
    /// through `s[t-1]`. Dropping those, as a per-step approximation would,
    /// discards the layer's memory -- which is the entire point of the layer.
    ///
    /// The spike reset is treated as detached, the standard choice for
    /// surrogate-gradient training: the reset is a discontinuity whose
    /// derivative is not usefully defined, and including it destabilises the
    /// backward pass.
    pub fn backward(
        &mut self,
        inputs: &Array3<f32>,
        trace: &RecurrentTrace,
        outputs: &Array3<f32>,
        output_grad: &Array3<f32>,
        surrogate: &dyn crate::training::SurrogateGradient,
    ) -> SNNResult<Array3<f32>> {
        let v_mem = &trace.v_mem;
        let (batch_size, num_steps, input_size) =
            (inputs.shape()[0], inputs.shape()[1], inputs.shape()[2]);
        let hidden_size = self.w_input.shape()[0];

        if input_size != self.w_input.shape()[1] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("input size {}", self.w_input.shape()[1]),
                actual: format!("input size {input_size}"),
            });
        }
        if output_grad.shape() != [batch_size, num_steps, hidden_size] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("output gradient {batch_size}x{num_steps}x{hidden_size}"),
                actual: format!("{:?}", output_grad.shape()),
            });
        }

        let alpha_syn = (-self.dt / self.neuron_params.tau_syn).exp();
        let alpha_mem = (-self.dt / self.neuron_params.tau_mem).exp();
        let threshold = self.neuron_params.v_threshold;

        let mut w_input_grad = Array2::zeros(self.w_input.raw_dim());
        let mut w_rec_grad = Array2::zeros(self.w_recurrent.raw_dim());
        let mut bias_grad = Array1::zeros(hidden_size);
        let mut input_grad = Array3::zeros(inputs.raw_dim());

        for b in 0..batch_size {
            // Gradient arriving at step t from steps after it.
            let mut carry_spike = Array1::<f32>::zeros(hidden_size);
            let mut carry_v = Array1::<f32>::zeros(hidden_size);
            let mut carry_syn = Array1::<f32>::zeros(hidden_size);

            for t in (0..num_steps).rev() {
                // Total gradient on this step's spikes: the loss term plus what
                // the recurrence carried back from step t+1.
                let mut d_spike = output_grad.slice(s![b, t, ..]).to_owned();
                d_spike += &carry_spike;

                // Through the spike function, via the surrogate, plus the
                // membrane's own decay path from t+1.
                //
                // A refractory neuron took neither branch of the forward
                // update: its membrane was held rather than integrated, and its
                // output was zero regardless of the potential. So the surrogate
                // contributes nothing there, the membrane passes gradient back
                // unattenuated instead of by `a_mem`, and no gradient reaches
                // the synaptic current through it.
                let mut d_v = carry_v.clone();
                let mut d_syn_here = Array1::<f32>::zeros(hidden_size);
                let mut next_carry_v = Array1::<f32>::zeros(hidden_size);
                for n in 0..hidden_size {
                    if trace.membrane_updated[[b, t, n]] {
                        d_v[n] +=
                            d_spike[n] * surrogate.compute_gradient(v_mem[[b, t, n]], threshold);
                        d_syn_here[n] = d_v[n] * (1.0 - alpha_mem);
                        next_carry_v[n] = d_v[n] * alpha_mem;
                    } else {
                        next_carry_v[n] = d_v[n];
                    }
                }
                carry_v = next_carry_v;

                // i_syn[t] = a_syn * i_syn[t-1] + I[t]
                let d_current = &d_syn_here + &carry_syn;
                carry_syn = &d_current * alpha_syn;

                // I[t] = W_in . x[t] + W_rec . s[t-1] + bias
                let x_t = inputs.slice(s![b, t, ..]);
                for n in 0..hidden_size {
                    let g = d_current[n];
                    if g == 0.0 {
                        continue;
                    }
                    for j in 0..input_size {
                        w_input_grad[[n, j]] += g * x_t[j];
                    }
                    if t > 0 {
                        for j in 0..hidden_size {
                            w_rec_grad[[n, j]] += g * outputs[[b, t - 1, j]];
                        }
                    }
                }
                bias_grad += &d_current;

                let mut dx = Array1::<f32>::zeros(input_size);
                for n in 0..hidden_size {
                    let g = d_current[n];
                    if g == 0.0 {
                        continue;
                    }
                    for j in 0..input_size {
                        dx[j] += g * self.w_input[[n, j]];
                    }
                }
                input_grad.slice_mut(s![b, t, ..]).assign(&dx);

                // What the previous step's spikes owe the loss.
                carry_spike = super::transpose_dot(&self.w_recurrent, &d_current);
            }
        }

        accumulate(&mut self.w_input_grad, w_input_grad);
        accumulate(&mut self.w_recurrent_grad, w_rec_grad);
        if self.bias.is_some() {
            match self.bias_grad {
                Some(ref mut g) => *g += &bias_grad,
                None => self.bias_grad = Some(bias_grad),
            }
        }

        Ok(input_grad)
    }
}

/// What [`SpikingRNN::forward_recording`] captured for the backward pass.
#[derive(Debug, Clone)]
pub struct RecurrentTrace {
    /// Membrane potential per step, before any spike reset was applied.
    pub v_mem: Array3<f32>,
    /// Whether each neuron's membrane actually integrated on that step.
    ///
    /// False while the neuron is refractory, which the forward pass handles by
    /// skipping the update entirely rather than by scaling it.
    pub membrane_updated: ndarray::Array3<bool>,
}

/// Adds `value` into `slot`, seeding it when this is the first contribution.
fn accumulate(slot: &mut Option<Array2<f32>>, value: Array2<f32>) {
    match slot {
        Some(existing) => *existing += &value,
        None => *slot = Some(value),
    }
}

impl SpikingLayer for SpikingRNN {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, input_size) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        if input_size != self.w_input.shape()[1] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("input size {}", self.w_input.shape()[1]),
                actual: format!("input size {}", input_size),
            });
        }

        let hidden_size = self.w_input.shape()[0];
        let mut output = Array3::zeros((batch_size, num_steps, hidden_size));

        // Reset previous output at start
        self.prev_output = None;

        // Process each time step
        for t in 0..num_steps {
            let input_t = input_dense.slice(s![.., t, ..]).to_owned();
            let output_t = self.forward_step(&input_t)?;
            output.slice_mut(s![.., t, ..]).assign(&output_t);
        }

        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {
        for state in &mut self.state {
            state.reset();
        }
        self.prev_output = None;
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        vec![&self.w_input, &self.w_recurrent]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![&mut self.w_input, &mut self.w_recurrent]
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        vec![self.w_input_grad.as_ref(), self.w_recurrent_grad.as_ref()]
    }

    fn zero_grad(&mut self) {
        self.w_input_grad = None;
        self.w_recurrent_grad = None;
        self.bias_grad = None;
    }
}

/// Spiking LSTM-like layer (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingLSTM {
    /// Input gate weights
    pub w_input_gate: Array2<f32>,
    /// Forget gate weights
    pub w_forget_gate: Array2<f32>,
    /// Output gate weights
    pub w_output_gate: Array2<f32>,
    /// Cell gate weights
    pub w_cell_gate: Array2<f32>,
    /// Recurrent weights for each gate
    pub w_rec_input: Array2<f32>,
    pub w_rec_forget: Array2<f32>,
    pub w_rec_output: Array2<f32>,
    pub w_rec_cell: Array2<f32>,
    /// Gradients, in the same order as the weights above.
    #[serde(skip)]
    pub w_input_gate_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_forget_gate_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_output_gate_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_cell_gate_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_rec_input_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_rec_forget_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_rec_output_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_rec_cell_grad: Option<Array2<f32>>,
    /// Neuron parameters
    pub neuron_params: NeuronParams,
    /// Cell state
    #[serde(skip)]
    pub cell_state: Option<Array2<f32>>,
    /// Hidden state
    #[serde(skip)]
    pub hidden_state: Option<Array2<f32>>,
    /// Neuron state
    #[serde(skip)]
    pub state: Vec<NeuronState>,
    /// Time step
    pub dt: f32,
    /// Adaptive
    pub adaptive: bool,
}

impl SpikingLSTM {
    /// Create a new Spiking LSTM layer
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        neuron_params: NeuronParams,
        dt: f32,
        adaptive: bool,
    ) -> Self {
        let std = (1.0 / hidden_size as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let mut rng = rng();

        let mut init_weights =
            |shape: (usize, usize)| Array2::from_shape_fn(shape, |_| normal.sample(&mut rng));

        Self {
            w_input_gate: init_weights((hidden_size, input_size)),
            w_forget_gate: init_weights((hidden_size, input_size)),
            w_output_gate: init_weights((hidden_size, input_size)),
            w_cell_gate: init_weights((hidden_size, input_size)),
            w_rec_input: init_weights((hidden_size, hidden_size)),
            w_rec_forget: init_weights((hidden_size, hidden_size)),
            w_rec_output: init_weights((hidden_size, hidden_size)),
            w_rec_cell: init_weights((hidden_size, hidden_size)),
            w_input_gate_grad: None,
            w_forget_gate_grad: None,
            w_output_gate_grad: None,
            w_cell_gate_grad: None,
            w_rec_input_grad: None,
            w_rec_forget_grad: None,
            w_rec_output_grad: None,
            w_rec_cell_grad: None,
            neuron_params,
            cell_state: None,
            hidden_state: None,
            state: Vec::new(),
            dt,
            adaptive,
        }
    }

    /// Initialize state
    fn ensure_state(&mut self, batch_size: usize) {
        let hidden_size = self.w_input_gate.shape()[0];
        if self.state.len() != batch_size {
            self.state = (0..batch_size)
                .map(|_| NeuronState::new(hidden_size, self.adaptive))
                .collect();
            self.cell_state = Some(Array2::zeros((batch_size, hidden_size)));
            self.hidden_state = Some(Array2::zeros((batch_size, hidden_size)));
        }
    }
}

/// What [`SpikingLSTM::forward_recording`] captured for the backward pass.
///
/// Every gate activation is kept because each is needed to differentiate the
/// step that produced it, and recomputing them backwards is not possible: the
/// cell state is overwritten as the forward pass advances.
#[derive(Debug, Clone)]
pub struct LstmTrace {
    /// Membrane potential of the output neuron, before any spike reset.
    pub v_mem: Array3<f32>,
    /// Whether each output neuron's membrane integrated on that step.
    pub membrane_updated: ndarray::Array3<bool>,
    /// Input gate activation, `(batch, steps, hidden)`.
    pub i_gate: Array3<f32>,
    /// Forget gate activation.
    pub f_gate: Array3<f32>,
    /// Output gate activation.
    pub o_gate: Array3<f32>,
    /// Cell candidate, already through `tanh`.
    pub g_gate: Array3<f32>,
    /// Cell state after the update.
    pub cell: Array3<f32>,
    /// Hidden state after the update, which is the LIF neuron's input current.
    pub hidden: Array3<f32>,
}

impl SpikingLSTM {
    /// Forward pass that records every intermediate the backward pass needs.
    pub fn forward_recording(
        &mut self,
        input: &SpikeTensor,
    ) -> SNNResult<(SpikeTensor, LstmTrace)> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, input_size) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        if input_size != self.w_input_gate.shape()[1] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("input size {}", self.w_input_gate.shape()[1]),
                actual: format!("input size {input_size}"),
            });
        }

        let hidden_size = self.w_input_gate.shape()[0];
        self.ensure_state(batch_size);

        let dims = (batch_size, num_steps, hidden_size);
        let mut output = Array3::zeros(dims);
        let mut trace = LstmTrace {
            v_mem: Array3::zeros(dims),
            membrane_updated: ndarray::Array3::from_elem(dims, false),
            i_gate: Array3::zeros(dims),
            f_gate: Array3::zeros(dims),
            o_gate: Array3::zeros(dims),
            g_gate: Array3::zeros(dims),
            cell: Array3::zeros(dims),
            hidden: Array3::zeros(dims),
        };

        // Both states start from zero, so the recorded history is complete:
        // step 0's "previous" values are known rather than inherited from
        // whatever a prior call left behind.
        self.cell_state = Some(Array2::zeros((batch_size, hidden_size)));
        self.hidden_state = Some(Array2::zeros((batch_size, hidden_size)));

        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

        for t in 0..num_steps {
            let h_prev = self.hidden_state.as_ref().expect("set above").clone();
            let c_prev = self.cell_state.as_ref().expect("set above").clone();
            let mut h_new = Array2::zeros((batch_size, hidden_size));
            let mut c_new = Array2::zeros((batch_size, hidden_size));

            for b in 0..batch_size {
                let x = input_dense.slice(s![b, t, ..]);
                let h = h_prev.slice(s![b, ..]);

                let i_pre = self.w_input_gate.dot(&x) + self.w_rec_input.dot(&h);
                let f_pre = self.w_forget_gate.dot(&x) + self.w_rec_forget.dot(&h);
                let o_pre = self.w_output_gate.dot(&x) + self.w_rec_output.dot(&h);
                let g_pre = self.w_cell_gate.dot(&x) + self.w_rec_cell.dot(&h);

                for n in 0..hidden_size {
                    let i_val = sigmoid(i_pre[n]);
                    let f_val = sigmoid(f_pre[n]);
                    let o_val = sigmoid(o_pre[n]);
                    let g_val = g_pre[n].tanh();

                    c_new[[b, n]] = f_val * c_prev[[b, n]] + i_val * g_val;
                    h_new[[b, n]] = o_val * c_new[[b, n]].tanh();

                    trace.i_gate[[b, t, n]] = i_val;
                    trace.f_gate[[b, t, n]] = f_val;
                    trace.o_gate[[b, t, n]] = o_val;
                    trace.g_gate[[b, t, n]] = g_val;
                    trace.cell[[b, t, n]] = c_new[[b, n]];
                    trace.hidden[[b, t, n]] = h_new[[b, n]];

                    trace.membrane_updated[[b, t, n]] = self.state[b].refrac[n] <= 0.0;
                }

                let (spikes, v_pre) = self.state[b].update_lif_recording(
                    &h_new.slice(s![b, ..]).to_owned(),
                    &self.neuron_params,
                    self.dt,
                );
                output.slice_mut(s![b, t, ..]).assign(&spikes);
                trace.v_mem.slice_mut(s![b, t, ..]).assign(&v_pre);
            }

            self.cell_state = Some(c_new);
            self.hidden_state = Some(h_new);
        }

        Ok((SpikeTensor::from_dense(output, input.requires_grad), trace))
    }

    /// The eight gate matrices, in a fixed order.
    ///
    /// Order matters: the optimizer keys its moment buffers by position, so
    /// every accessor here agrees on input gates first, then recurrent, each in
    /// input/forget/output/cell order.
    pub fn gate_weights(&self) -> Vec<&Array2<f32>> {
        vec![
            &self.w_input_gate,
            &self.w_forget_gate,
            &self.w_output_gate,
            &self.w_cell_gate,
            &self.w_rec_input,
            &self.w_rec_forget,
            &self.w_rec_output,
            &self.w_rec_cell,
        ]
    }

    /// The gradients that have been accumulated, in [`Self::gate_weights`] order.
    pub fn gate_grads(&self) -> Vec<&Array2<f32>> {
        self.gate_grads_all().into_iter().flatten().collect()
    }

    /// Every gradient slot, present or not, in [`Self::gate_weights`] order.
    pub fn gate_grads_all(&self) -> Vec<Option<&Array2<f32>>> {
        vec![
            self.w_input_gate_grad.as_ref(),
            self.w_forget_gate_grad.as_ref(),
            self.w_output_gate_grad.as_ref(),
            self.w_cell_gate_grad.as_ref(),
            self.w_rec_input_grad.as_ref(),
            self.w_rec_forget_grad.as_ref(),
            self.w_rec_output_grad.as_ref(),
            self.w_rec_cell_grad.as_ref(),
        ]
    }

    /// Mutable access to the accumulated gradients, for clipping.
    pub fn gate_grads_mut(&mut self) -> Vec<&mut Array2<f32>> {
        [
            self.w_input_gate_grad.as_mut(),
            self.w_forget_gate_grad.as_mut(),
            self.w_output_gate_grad.as_mut(),
            self.w_cell_gate_grad.as_mut(),
            self.w_rec_input_grad.as_mut(),
            self.w_rec_forget_grad.as_mut(),
            self.w_rec_output_grad.as_mut(),
            self.w_rec_cell_grad.as_mut(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// Replace the gate matrices, in [`Self::gate_weights`] order.
    ///
    /// A `None` entry leaves that matrix untouched, which is what an optimizer
    /// step does for a matrix that accumulated no gradient.
    pub fn set_gate_weights(&mut self, values: Vec<Option<Array2<f32>>>) {
        let mut it = values.into_iter();
        let mut apply = |slot: &mut Array2<f32>| {
            if let Some(Some(v)) = it.next() {
                *slot = v;
            }
        };
        apply(&mut self.w_input_gate);
        apply(&mut self.w_forget_gate);
        apply(&mut self.w_output_gate);
        apply(&mut self.w_cell_gate);
        apply(&mut self.w_rec_input);
        apply(&mut self.w_rec_forget);
        apply(&mut self.w_rec_output);
        apply(&mut self.w_rec_cell);
    }

    /// Backpropagation through time across all eight weight matrices.
    ///
    /// The forward pass is
    ///
    /// ```text
    /// i[t] = sigmoid(W_i x[t] + R_i h[t-1])
    /// f[t] = sigmoid(W_f x[t] + R_f h[t-1])
    /// o[t] = sigmoid(W_o x[t] + R_o h[t-1])
    /// g[t] = tanh   (W_c x[t] + R_c h[t-1])
    /// c[t] = f[t] * c[t-1] + i[t] * g[t]
    /// h[t] = o[t] * tanh(c[t])
    /// s[t] = LIF(h[t])
    /// ```
    ///
    /// so `h[t]` is the LIF neuron's input current, and gradient reaches
    /// `h[t-1]` through all four gates as well as through the cell path
    /// `c[t-1]`. Both are carried backwards here.
    ///
    /// The LIF chain is the same as in the other layers: surrogate for the
    /// spike, the two decay paths for the neuron state, no surrogate term on a
    /// refractory step, and a detached reset.
    pub fn backward(
        &mut self,
        inputs: &Array3<f32>,
        trace: &LstmTrace,
        output_grad: &Array3<f32>,
        surrogate: &dyn crate::training::surrogate::SurrogateGradient,
    ) -> SNNResult<Array3<f32>> {
        let (batch_size, num_steps, input_size) =
            (inputs.shape()[0], inputs.shape()[1], inputs.shape()[2]);
        let hidden_size = self.w_input_gate.shape()[0];

        if input_size != self.w_input_gate.shape()[1] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("input size {}", self.w_input_gate.shape()[1]),
                actual: format!("input size {input_size}"),
            });
        }
        if output_grad.shape() != [batch_size, num_steps, hidden_size] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("output gradient {batch_size}x{num_steps}x{hidden_size}"),
                actual: format!("{:?}", output_grad.shape()),
            });
        }

        let alpha_syn = (-self.dt / self.neuron_params.tau_syn).exp();
        let alpha_mem = (-self.dt / self.neuron_params.tau_mem).exp();
        let threshold = self.neuron_params.v_threshold;

        let shape_in = (hidden_size, input_size);
        let shape_rec = (hidden_size, hidden_size);
        let mut d_wi = Array2::zeros(shape_in);
        let mut d_wf = Array2::zeros(shape_in);
        let mut d_wo = Array2::zeros(shape_in);
        let mut d_wc = Array2::zeros(shape_in);
        let mut d_ri = Array2::zeros(shape_rec);
        let mut d_rf = Array2::zeros(shape_rec);
        let mut d_ro = Array2::zeros(shape_rec);
        let mut d_rc = Array2::zeros(shape_rec);
        let mut input_grad = Array3::zeros(inputs.raw_dim());

        for b in 0..batch_size {
            let mut carry_v = Array1::<f32>::zeros(hidden_size);
            let mut carry_syn = Array1::<f32>::zeros(hidden_size);
            let mut carry_h = Array1::<f32>::zeros(hidden_size);
            let mut carry_c = Array1::<f32>::zeros(hidden_size);

            for t in (0..num_steps).rev() {
                // --- through the output neuron, to h[t] ---
                let mut d_v = carry_v.clone();
                let mut d_syn_here = Array1::<f32>::zeros(hidden_size);
                let mut next_carry_v = Array1::<f32>::zeros(hidden_size);
                for n in 0..hidden_size {
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

                // h[t] is the LIF's input current, so this is dL/dh[t] from the
                // spike path; the gates at t+1 add their share through carry_h.
                let d_current = &d_syn_here + &carry_syn;
                carry_syn = &d_current * alpha_syn;
                let d_h = &d_current + &carry_h;

                // --- through h[t] = o[t] * tanh(c[t]) ---
                let mut d_i_pre = Array1::<f32>::zeros(hidden_size);
                let mut d_f_pre = Array1::<f32>::zeros(hidden_size);
                let mut d_o_pre = Array1::<f32>::zeros(hidden_size);
                let mut d_g_pre = Array1::<f32>::zeros(hidden_size);
                let mut next_carry_c = Array1::<f32>::zeros(hidden_size);

                for n in 0..hidden_size {
                    let (i_v, f_v, o_v, g_v) = (
                        trace.i_gate[[b, t, n]],
                        trace.f_gate[[b, t, n]],
                        trace.o_gate[[b, t, n]],
                        trace.g_gate[[b, t, n]],
                    );
                    let c_v = trace.cell[[b, t, n]];
                    let tanh_c = c_v.tanh();
                    // c[t-1], which is zero before the first step.
                    let c_prev = if t > 0 {
                        trace.cell[[b, t - 1, n]]
                    } else {
                        0.0
                    };

                    let d_o = d_h[n] * tanh_c;
                    // Both the direct path through tanh(c[t]) and whatever the
                    // next step's cell update carried back.
                    let d_c = d_h[n] * o_v * (1.0 - tanh_c * tanh_c) + carry_c[n];

                    let d_f = d_c * c_prev;
                    let d_i = d_c * g_v;
                    let d_g = d_c * i_v;
                    next_carry_c[n] = d_c * f_v;

                    // Through the gate nonlinearities.
                    d_i_pre[n] = d_i * i_v * (1.0 - i_v);
                    d_f_pre[n] = d_f * f_v * (1.0 - f_v);
                    d_o_pre[n] = d_o * o_v * (1.0 - o_v);
                    d_g_pre[n] = d_g * (1.0 - g_v * g_v);
                }
                carry_c = next_carry_c;

                // --- into the weights, the input, and h[t-1] ---
                let x = inputs.slice(s![b, t, ..]);
                let h_prev: Array1<f32> = if t > 0 {
                    trace.hidden.slice(s![b, t - 1, ..]).to_owned()
                } else {
                    Array1::zeros(hidden_size)
                };

                for n in 0..hidden_size {
                    for j in 0..input_size {
                        d_wi[[n, j]] += d_i_pre[n] * x[j];
                        d_wf[[n, j]] += d_f_pre[n] * x[j];
                        d_wo[[n, j]] += d_o_pre[n] * x[j];
                        d_wc[[n, j]] += d_g_pre[n] * x[j];
                    }
                    for j in 0..hidden_size {
                        d_ri[[n, j]] += d_i_pre[n] * h_prev[j];
                        d_rf[[n, j]] += d_f_pre[n] * h_prev[j];
                        d_ro[[n, j]] += d_o_pre[n] * h_prev[j];
                        d_rc[[n, j]] += d_g_pre[n] * h_prev[j];
                    }
                }

                // Accumulated into one buffer: four transposed dots meant
                // four strided passes and three temporary allocations per
                // (batch, step).
                let mut dx = Array1::<f32>::zeros(input_size);
                super::accumulate_transpose_dot(&mut dx, &self.w_input_gate, &d_i_pre);
                super::accumulate_transpose_dot(&mut dx, &self.w_forget_gate, &d_f_pre);
                super::accumulate_transpose_dot(&mut dx, &self.w_output_gate, &d_o_pre);
                super::accumulate_transpose_dot(&mut dx, &self.w_cell_gate, &d_g_pre);
                input_grad.slice_mut(s![b, t, ..]).assign(&dx);

                carry_h = Array1::<f32>::zeros(hidden_size);
                super::accumulate_transpose_dot(&mut carry_h, &self.w_rec_input, &d_i_pre);
                super::accumulate_transpose_dot(&mut carry_h, &self.w_rec_forget, &d_f_pre);
                super::accumulate_transpose_dot(&mut carry_h, &self.w_rec_output, &d_o_pre);
                super::accumulate_transpose_dot(&mut carry_h, &self.w_rec_cell, &d_g_pre);
            }
        }

        accumulate(&mut self.w_input_gate_grad, d_wi);
        accumulate(&mut self.w_forget_gate_grad, d_wf);
        accumulate(&mut self.w_output_gate_grad, d_wo);
        accumulate(&mut self.w_cell_gate_grad, d_wc);
        accumulate(&mut self.w_rec_input_grad, d_ri);
        accumulate(&mut self.w_rec_forget_grad, d_rf);
        accumulate(&mut self.w_rec_output_grad, d_ro);
        accumulate(&mut self.w_rec_cell_grad, d_rc);

        Ok(input_grad)
    }
}

impl SpikingLayer for SpikingLSTM {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        // Delegates so there is one implementation of the dynamics, and so the
        // cell and hidden states are always initialised. The previous body read
        // `self.hidden_state.as_ref().unwrap()`, which `reset_state` had set to
        // None -- so a reset followed by a forward panicked, and `reset_state`
        // is exactly what a caller does between samples.
        self.forward_recording(input).map(|(out, _)| out)
    }

    fn reset_state(&mut self) {
        for state in &mut self.state {
            state.reset();
        }
        self.cell_state = None;
        self.hidden_state = None;
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        // All eight. Listing only the input-path gates hid the four recurrent
        // matrices from anything iterating parameters -- half the layer.
        self.gate_weights()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![
            &mut self.w_input_gate,
            &mut self.w_forget_gate,
            &mut self.w_output_gate,
            &mut self.w_cell_gate,
            &mut self.w_rec_input,
            &mut self.w_rec_forget,
            &mut self.w_rec_output,
            &mut self.w_rec_cell,
        ]
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        // All eight, in `gate_weights` order -- the previous four `None`s
        // reported every gradient as absent even after `backward` had filled
        // them in.
        self.gate_grads_all()
    }

    fn zero_grad(&mut self) {
        // Not a no-op: `backward` accumulates into all eight gate gradients, so
        // failing to clear them here made them grow without bound across a
        // training run.
        self.w_input_gate_grad = None;
        self.w_forget_gate_grad = None;
        self.w_output_gate_grad = None;
        self.w_cell_gate_grad = None;
        self.w_rec_input_grad = None;
        self.w_rec_forget_grad = None;
        self.w_rec_output_grad = None;
        self.w_rec_cell_grad = None;
    }
}

impl Default for SpikingRNN {
    fn default() -> Self {
        Self::new(1, 1, false, NeuronParams::default(), 1.0, false)
    }
}

impl Default for SpikingLSTM {
    fn default() -> Self {
        Self::new(1, 1, NeuronParams::default(), 1.0, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_creation() {
        let rnn = SpikingRNN::new(10, 20, true, NeuronParams::default(), 1.0, false);
        assert_eq!(rnn.w_input.shape(), &[20, 10]);
        assert_eq!(rnn.w_recurrent.shape(), &[20, 20]);
    }

    #[test]
    fn test_rnn_forward() {
        let mut rnn = SpikingRNN::new(10, 20, true, NeuronParams::default(), 1.0, false);
        let input = SpikeTensor::zeros(2, 15, 10, false);

        let output = rnn.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 15, 20));
    }

    #[test]
    fn test_lstm_creation() {
        let lstm = SpikingLSTM::new(10, 20, NeuronParams::default(), 1.0, false);
        assert_eq!(lstm.w_input_gate.shape(), &[20, 10]);
    }

    #[test]
    fn test_lstm_forward() {
        let mut lstm = SpikingLSTM::new(10, 20, NeuronParams::default(), 1.0, false);
        let input = SpikeTensor::zeros(2, 15, 10, false);

        let output = lstm.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 15, 20));
    }

    // ---- backward-pass validation -------------------------------------
    //
    // Surrogate-gradient training is not the derivative of any single smooth
    // function -- the forward pass uses hard spikes while the backward pass
    // substitutes a smooth derivative -- so a finite-difference check against
    // the spiking forward would compare against the wrong thing and prove
    // nothing.
    //
    // What the backward pass *does* compute is the exact gradient of the
    // surrogate-linearised dynamics. That has a second, structurally
    // independent implementation: propagate tangents forwards through the same
    // linearisation instead of accumulating adjoints backwards. Forward mode
    // and reverse mode share no code and must agree to floating-point
    // precision, which is the standard way to validate a hand-written backward
    // pass.

    use crate::training::surrogate::{FastSigmoidSurrogate, SurrogateGradient};

    /// Which single parameter the forward-mode pass is differentiating.
    enum Wrt {
        Input(usize, usize),
        Recurrent(usize, usize),
        Bias(usize),
    }

    /// Directional derivative of `sum(output_grad * spikes)` with respect to one
    /// parameter, by forward-mode propagation through the linearised dynamics.
    fn forward_mode_grad(
        rnn: &SpikingRNN,
        inputs: &Array3<f32>,
        trace: &RecurrentTrace,
        outputs: &Array3<f32>,
        output_grad: &Array3<f32>,
        surrogate: &dyn SurrogateGradient,
        wrt: &Wrt,
    ) -> f32 {
        let (batch, steps, in_size) = (inputs.shape()[0], inputs.shape()[1], inputs.shape()[2]);
        let hidden = rnn.w_input.shape()[0];
        let a_syn = (-rnn.dt / rnn.neuron_params.tau_syn).exp();
        let a_mem = (-rnn.dt / rnn.neuron_params.tau_mem).exp();
        let threshold = rnn.neuron_params.v_threshold;

        let mut total = 0.0;
        for b in 0..batch {
            let mut d_syn = vec![0.0f32; hidden];
            let mut d_v = vec![0.0f32; hidden];
            let mut d_s_prev = vec![0.0f32; hidden];

            for t in 0..steps {
                let mut d_s = vec![0.0f32; hidden];
                for n in 0..hidden {
                    // Tangent of the total input current at this step.
                    let mut d_current = 0.0;
                    match *wrt {
                        Wrt::Input(p, q) if p == n => d_current += inputs[[b, t, q]],
                        Wrt::Recurrent(p, q) if p == n => {
                            if t > 0 {
                                d_current += outputs[[b, t - 1, q]];
                            }
                        }
                        Wrt::Bias(p) if p == n => d_current += 1.0,
                        _ => {}
                    }
                    // The recurrence carries the previous step's perturbation.
                    for (j, prev) in d_s_prev.iter().enumerate() {
                        d_current += rnn.w_recurrent[[n, j]] * prev;
                    }

                    d_syn[n] = a_syn * d_syn[n] + d_current;

                    if trace.membrane_updated[[b, t, n]] {
                        d_v[n] = a_mem * d_v[n] + d_syn[n] * (1.0 - a_mem);
                        d_s[n] =
                            surrogate.compute_gradient(trace.v_mem[[b, t, n]], threshold) * d_v[n];
                    } else {
                        // Membrane held, output forced to zero.
                        d_s[n] = 0.0;
                    }

                    total += output_grad[[b, t, n]] * d_s[n];
                }
                d_s_prev = d_s;
            }
            let _ = in_size;
        }
        total
    }

    /// Builds a small RNN with deterministic weights, so the test does not
    /// depend on the random initialiser.
    fn fixture() -> (SpikingRNN, Array3<f32>, Array3<f32>) {
        let (batch, steps, in_size, hidden) = (2, 6, 3, 4);
        let mut rnn = SpikingRNN::new(in_size, hidden, true, NeuronParams::default(), 1.0, false);

        for n in 0..hidden {
            for j in 0..in_size {
                rnn.w_input[[n, j]] = 0.6 + 0.13 * (n as f32) - 0.07 * (j as f32);
            }
            for j in 0..hidden {
                rnn.w_recurrent[[n, j]] = 0.21 - 0.05 * ((n + j) as f32);
            }
        }
        if let Some(ref mut b) = rnn.bias {
            for n in 0..hidden {
                b[n] = 0.02 * (n as f32);
            }
        }

        let inputs = Array3::from_shape_fn((batch, steps, in_size), |(b, t, i)| {
            // Deterministic, and varied enough that spikes and refractory
            // periods both occur.
            (((b * 7 + t * 3 + i * 5) % 4) as f32) * 0.5
        });
        let output_grad = Array3::from_shape_fn((batch, steps, hidden), |(b, t, n)| {
            0.3 - 0.1 * (((b + t + n) % 3) as f32)
        });
        (rnn, inputs, output_grad)
    }

    /// Reverse mode must reproduce forward mode for every parameter.
    #[test]
    fn bptt_matches_forward_mode_differentiation() {
        let (mut rnn, inputs, output_grad) = fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);

        let (spikes, trace) = rnn
            .forward_recording(&SpikeTensor::from_dense(inputs.clone(), false))
            .unwrap();
        let outputs = spikes.to_dense();

        // Both the spiking and the refractory branch must be exercised, or the
        // test would not be checking the interesting part.
        let spike_count: f32 = outputs.iter().sum();
        let refractory = trace.membrane_updated.iter().filter(|u| !**u).count();
        assert!(spike_count > 0.0, "fixture never spiked");
        assert!(refractory > 0, "fixture never entered a refractory step");

        rnn.backward(&inputs, &trace, &outputs, &output_grad, &surrogate)
            .unwrap();

        let w_in_grad = rnn.w_input_grad.clone().unwrap();
        let w_rec_grad = rnn.w_recurrent_grad.clone().unwrap();
        let bias_grad = rnn.bias_grad.clone().unwrap();

        let hidden = rnn.w_input.shape()[0];
        let in_size = rnn.w_input.shape()[1];
        let mut compared = 0;

        for n in 0..hidden {
            for j in 0..in_size {
                let expected = forward_mode_grad(
                    &rnn,
                    &inputs,
                    &trace,
                    &outputs,
                    &output_grad,
                    &surrogate,
                    &Wrt::Input(n, j),
                );
                let got = w_in_grad[[n, j]];
                assert!(
                    (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "w_input[{n},{j}]: reverse {got} vs forward {expected}"
                );
                compared += 1;
            }
            for j in 0..hidden {
                let expected = forward_mode_grad(
                    &rnn,
                    &inputs,
                    &trace,
                    &outputs,
                    &output_grad,
                    &surrogate,
                    &Wrt::Recurrent(n, j),
                );
                let got = w_rec_grad[[n, j]];
                assert!(
                    (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "w_recurrent[{n},{j}]: reverse {got} vs forward {expected}"
                );
                compared += 1;
            }
            let expected = forward_mode_grad(
                &rnn,
                &inputs,
                &trace,
                &outputs,
                &output_grad,
                &surrogate,
                &Wrt::Bias(n),
            );
            assert!(
                (bias_grad[n] - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "bias[{n}]: reverse {} vs forward {expected}",
                bias_grad[n]
            );
            compared += 1;
        }
        assert_eq!(compared, hidden * (in_size + hidden) + hidden);
    }

    /// The recurrent weights must actually carry gradient. If the reverse pass
    /// dropped the `w_recurrent^T` term the layer would train as a stack of
    /// independent time steps, and this is what would catch it.
    #[test]
    fn recurrent_path_contributes_gradient() {
        let (mut rnn, inputs, output_grad) = fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);

        let (spikes, trace) = rnn
            .forward_recording(&SpikeTensor::from_dense(inputs.clone(), false))
            .unwrap();
        let outputs = spikes.to_dense();
        rnn.backward(&inputs, &trace, &outputs, &output_grad, &surrogate)
            .unwrap();
        let with_recurrence = rnn.w_input_grad.clone().unwrap();

        // Same trace, but with the recurrent weights zeroed the input gradient
        // must change -- otherwise nothing flowed through them.
        let mut cut = rnn.clone();
        cut.w_recurrent.fill(0.0);
        cut.w_input_grad = None;
        cut.w_recurrent_grad = None;
        cut.bias_grad = None;
        cut.backward(&inputs, &trace, &outputs, &output_grad, &surrogate)
            .unwrap();
        let without = cut.w_input_grad.clone().unwrap();

        let delta: f32 = (&with_recurrence - &without).iter().map(|d| d.abs()).sum();
        assert!(
            delta > 1e-6,
            "zeroing w_recurrent changed nothing; the recurrent path is not wired in"
        );
    }

    /// Gradients accumulate across calls rather than replacing each other,
    /// matching how `SpikingLinear` behaves and what mini-batching needs.
    #[test]
    fn gradients_accumulate_across_calls() {
        let (mut rnn, inputs, output_grad) = fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);
        let tensor = SpikeTensor::from_dense(inputs.clone(), false);

        let (spikes, trace) = rnn.forward_recording(&tensor).unwrap();
        let outputs = spikes.to_dense();

        rnn.backward(&inputs, &trace, &outputs, &output_grad, &surrogate)
            .unwrap();
        let once = rnn.w_input_grad.clone().unwrap();
        rnn.backward(&inputs, &trace, &outputs, &output_grad, &surrogate)
            .unwrap();
        let twice = rnn.w_input_grad.clone().unwrap();

        for (a, b) in once.iter().zip(twice.iter()) {
            assert!(
                (2.0 * a - b).abs() <= 1e-5 * a.abs().max(1.0),
                "{a} then {b}"
            );
        }
    }

    /// A mismatched gradient shape is reported, not silently indexed past.
    #[test]
    fn backward_rejects_a_mismatched_gradient_shape() {
        let (mut rnn, inputs, _) = fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);
        let (spikes, trace) = rnn
            .forward_recording(&SpikeTensor::from_dense(inputs.clone(), false))
            .unwrap();
        let outputs = spikes.to_dense();

        let wrong = Array3::zeros((inputs.shape()[0], inputs.shape()[1], 99));
        assert!(
            rnn.backward(&inputs, &trace, &outputs, &wrong, &surrogate)
                .is_err()
        );
    }

    /// A reset followed by a forward pass must work. The LSTM used to unwrap
    /// the hidden state that `reset_state` had just cleared, so the trainer's
    /// own between-sample reset would have panicked it.
    #[test]
    fn lstm_survives_reset_then_forward() {
        let mut lstm = SpikingLSTM::new(3, 4, NeuronParams::default(), 1.0, false);
        let input = SpikeTensor::zeros(2, 5, 3, false);
        lstm.forward(&input).unwrap();
        lstm.reset_state();
        let out = lstm
            .forward(&input)
            .expect("forward after reset must not panic");
        assert_eq!(out.shape(), (2, 5, 4));
    }

    /// `zero_grad` must actually clear, and `gradients()` must report what
    /// `backward` accumulated -- both were stubs claiming the layer had no
    /// gradient storage, while `backward` filled eight matrices.
    #[test]
    fn lstm_zero_grad_clears_and_gradients_are_reported() {
        let (mut lstm, inputs, output_grad) = lstm_fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);
        let (_, trace) = lstm
            .forward_recording(&SpikeTensor::from_dense(inputs.clone(), false))
            .unwrap();

        lstm.backward(&inputs, &trace, &output_grad, &surrogate)
            .unwrap();
        let reported = lstm.gradients().iter().filter(|g| g.is_some()).count();
        assert_eq!(
            reported, 8,
            "gradients() reported {reported} of 8 gate gradients"
        );

        let before: f32 = lstm
            .gate_grads()
            .iter()
            .map(|g| g.iter().map(|v| v * v).sum::<f32>())
            .sum();
        assert!(
            before > 0.0,
            "no gradient accumulated, so the test proves nothing"
        );

        lstm.zero_grad();
        assert!(
            lstm.gate_grads().is_empty(),
            "zero_grad left gradients in place"
        );

        // And a fresh backward gives the same magnitude, not twice it.
        lstm.backward(&inputs, &trace, &output_grad, &surrogate)
            .unwrap();
        let after: f32 = lstm
            .gate_grads()
            .iter()
            .map(|g| g.iter().map(|v| v * v).sum::<f32>())
            .sum();
        assert!(
            (after - before).abs() <= 1e-4 * before,
            "after clearing, the gradient came back as {after} against {before}"
        );

        // parameters() must expose all eight matrices, not just the input path.
        assert_eq!(lstm.parameters().len(), 8);
        assert_eq!(lstm.parameters_mut().len(), 8);
    }

    // ---- LSTM backward validation -------------------------------------

    /// Which single LSTM weight the forward-mode pass is differentiating.
    #[derive(Clone, Copy)]
    enum LstmWrt {
        /// Input-path weight for one of the four gates.
        Input(usize, usize, usize),
        /// Recurrent-path weight for one of the four gates.
        Rec(usize, usize, usize),
    }

    /// Forward-mode directional derivative of `sum(output_grad * spikes)`.
    ///
    /// Propagates tangents for every intermediate the LSTM carries: the four
    /// gate pre-activations, the cell state, the hidden state and the output
    /// neuron's synaptic and membrane state.
    fn lstm_forward_mode(
        layer: &SpikingLSTM,
        inputs: &Array3<f32>,
        trace: &LstmTrace,
        output_grad: &Array3<f32>,
        surrogate: &dyn SurrogateGradient,
        wrt: LstmWrt,
    ) -> f32 {
        let (batch, steps, in_size) = (inputs.shape()[0], inputs.shape()[1], inputs.shape()[2]);
        let hidden = layer.w_input_gate.shape()[0];
        let a_syn = (-layer.dt / layer.neuron_params.tau_syn).exp();
        let a_mem = (-layer.dt / layer.neuron_params.tau_mem).exp();
        let threshold = layer.neuron_params.v_threshold;

        // gate 0 = input, 1 = forget, 2 = output, 3 = cell candidate
        let w_in = [
            &layer.w_input_gate,
            &layer.w_forget_gate,
            &layer.w_output_gate,
            &layer.w_cell_gate,
        ];
        let w_rec = [
            &layer.w_rec_input,
            &layer.w_rec_forget,
            &layer.w_rec_output,
            &layer.w_rec_cell,
        ];

        let mut total = 0.0;
        for b in 0..batch {
            let mut d_c = vec![0.0f32; hidden];
            let mut d_h = vec![0.0f32; hidden];
            let mut d_syn = vec![0.0f32; hidden];
            let mut d_v = vec![0.0f32; hidden];

            for t in 0..steps {
                let mut d_h_new = vec![0.0f32; hidden];
                let mut d_c_new = vec![0.0f32; hidden];

                for n in 0..hidden {
                    // Tangent of each gate's pre-activation.
                    let mut pre = [0.0f32; 4];
                    for (g, item) in pre.iter_mut().enumerate() {
                        // Direct term, when this is the weight being varied.
                        match wrt {
                            LstmWrt::Input(wg, wn, wj) if wg == g && wn == n => {
                                *item += inputs[[b, t, wj]];
                            }
                            LstmWrt::Rec(wg, wn, wj) if wg == g && wn == n => {
                                *item += if t > 0 {
                                    trace.hidden[[b, t - 1, wj]]
                                } else {
                                    0.0
                                };
                            }
                            _ => {}
                        }
                        // Recurrent term: the previous hidden state moved.
                        for (j, dh) in d_h.iter().enumerate() {
                            *item += w_rec[g][[n, j]] * dh;
                        }
                    }

                    let (i_v, f_v, o_v, g_v) = (
                        trace.i_gate[[b, t, n]],
                        trace.f_gate[[b, t, n]],
                        trace.o_gate[[b, t, n]],
                        trace.g_gate[[b, t, n]],
                    );
                    let d_i = pre[0] * i_v * (1.0 - i_v);
                    let d_f = pre[1] * f_v * (1.0 - f_v);
                    let d_o = pre[2] * o_v * (1.0 - o_v);
                    let d_g = pre[3] * (1.0 - g_v * g_v);

                    let c_prev = if t > 0 {
                        trace.cell[[b, t - 1, n]]
                    } else {
                        0.0
                    };
                    // c = f*c_prev + i*g
                    d_c_new[n] = d_f * c_prev + f_v * d_c[n] + d_i * g_v + i_v * d_g;

                    // h = o * tanh(c)
                    let tanh_c = trace.cell[[b, t, n]].tanh();
                    d_h_new[n] = d_o * tanh_c + o_v * (1.0 - tanh_c * tanh_c) * d_c_new[n];

                    // Through the output neuron, whose input current is h.
                    d_syn[n] = a_syn * d_syn[n] + d_h_new[n];
                    let d_s = if trace.membrane_updated[[b, t, n]] {
                        d_v[n] = a_mem * d_v[n] + d_syn[n] * (1.0 - a_mem);
                        surrogate.compute_gradient(trace.v_mem[[b, t, n]], threshold) * d_v[n]
                    } else {
                        0.0
                    };
                    total += output_grad[[b, t, n]] * d_s;
                }

                d_c = d_c_new;
                d_h = d_h_new;
            }
            let _ = in_size;
            let _ = w_in;
        }
        total
    }

    fn lstm_fixture() -> (SpikingLSTM, Array3<f32>, Array3<f32>) {
        let (batch, steps, in_size, hidden) = (2, 5, 3, 3);
        // h = o * tanh(c) is bounded in (-1, 1), so the default threshold of
        // 1.0 is unreachable and the fixture would never spike -- which would
        // make the surrogate term zero everywhere and the test vacuous.
        let params = NeuronParams {
            v_threshold: 0.15,
            ..NeuronParams::default()
        };
        let mut lstm = SpikingLSTM::new(in_size, hidden, params, 1.0, false);

        // Deterministic weights, so the test does not depend on the initialiser.
        let fill = |m: &mut Array2<f32>, seed: f32| {
            for n in 0..m.shape()[0] {
                for j in 0..m.shape()[1] {
                    m[[n, j]] = seed + 0.13 * (n as f32) - 0.09 * (j as f32);
                }
            }
        };
        fill(&mut lstm.w_input_gate, 0.5);
        fill(&mut lstm.w_forget_gate, 0.4);
        fill(&mut lstm.w_output_gate, 0.6);
        fill(&mut lstm.w_cell_gate, 0.35);
        fill(&mut lstm.w_rec_input, 0.2);
        fill(&mut lstm.w_rec_forget, 0.15);
        fill(&mut lstm.w_rec_output, 0.25);
        fill(&mut lstm.w_rec_cell, 0.1);

        let inputs = Array3::from_shape_fn((batch, steps, in_size), |(b, t, i)| {
            (((b * 7 + t * 3 + i * 2) % 4) as f32) * 0.8
        });
        let output_grad = Array3::from_shape_fn((batch, steps, hidden), |(b, t, n)| {
            0.3 - 0.11 * (((b + t + n) % 3) as f32)
        });
        (lstm, inputs, output_grad)
    }

    #[test]
    fn lstm_backward_matches_forward_mode() {
        let (mut lstm, inputs, output_grad) = lstm_fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);

        let (spikes, trace) = lstm
            .forward_recording(&SpikeTensor::from_dense(inputs.clone(), false))
            .unwrap();
        let fired: f32 = spikes.to_dense().iter().sum();
        assert!(fired > 0.0, "fixture never spiked");

        lstm.backward(&inputs, &trace, &output_grad, &surrogate)
            .unwrap();

        let input_grads = [
            lstm.w_input_gate_grad.clone().unwrap(),
            lstm.w_forget_gate_grad.clone().unwrap(),
            lstm.w_output_gate_grad.clone().unwrap(),
            lstm.w_cell_gate_grad.clone().unwrap(),
        ];
        let rec_grads = [
            lstm.w_rec_input_grad.clone().unwrap(),
            lstm.w_rec_forget_grad.clone().unwrap(),
            lstm.w_rec_output_grad.clone().unwrap(),
            lstm.w_rec_cell_grad.clone().unwrap(),
        ];

        let hidden = lstm.w_input_gate.shape()[0];
        let in_size = lstm.w_input_gate.shape()[1];
        let names = ["input", "forget", "output", "cell"];

        for g in 0..4 {
            for n in 0..hidden {
                for j in 0..in_size {
                    let expected = lstm_forward_mode(
                        &lstm,
                        &inputs,
                        &trace,
                        &output_grad,
                        &surrogate,
                        LstmWrt::Input(g, n, j),
                    );
                    let got = input_grads[g][[n, j]];
                    assert!(
                        (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                        "w_{}_gate[{n},{j}]: reverse {got} vs forward {expected}",
                        names[g]
                    );
                }
                for j in 0..hidden {
                    let expected = lstm_forward_mode(
                        &lstm,
                        &inputs,
                        &trace,
                        &output_grad,
                        &surrogate,
                        LstmWrt::Rec(g, n, j),
                    );
                    let got = rec_grads[g][[n, j]];
                    assert!(
                        (got - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                        "w_rec_{}[{n},{j}]: reverse {got} vs forward {expected}",
                        names[g]
                    );
                }
            }
        }
    }

    /// The cell path must carry gradient across steps. If `carry_c` were
    /// dropped the LSTM would train as if it had no memory, which is the one
    /// thing distinguishing it from a plain gated layer.
    #[test]
    fn lstm_cell_path_carries_gradient() {
        let (mut lstm, inputs, output_grad) = lstm_fixture();
        let surrogate = FastSigmoidSurrogate::new(10.0);
        let (_, trace) = lstm
            .forward_recording(&SpikeTensor::from_dense(inputs.clone(), false))
            .unwrap();
        lstm.backward(&inputs, &trace, &output_grad, &surrogate)
            .unwrap();

        // The forget gate only ever receives gradient through the cell path:
        // f[t] enters the network solely via c[t] = f[t]*c[t-1] + ...
        let forget = lstm.w_forget_gate_grad.clone().unwrap();
        assert!(
            forget.iter().any(|g| g.abs() > 1e-9),
            "forget gate received no gradient, so the cell path is not connected"
        );
    }
}
