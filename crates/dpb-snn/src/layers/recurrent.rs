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

        let w_input = Array2::from_shape_fn((hidden_size, input_size), |_| {
            normal_input.sample(&mut rng)
        });

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

        let mut init_weights = |shape: (usize, usize)| {
            Array2::from_shape_fn(shape, |_| normal.sample(&mut rng))
        };

        Self {
            w_input_gate: init_weights((hidden_size, input_size)),
            w_forget_gate: init_weights((hidden_size, input_size)),
            w_output_gate: init_weights((hidden_size, input_size)),
            w_cell_gate: init_weights((hidden_size, input_size)),
            w_rec_input: init_weights((hidden_size, hidden_size)),
            w_rec_forget: init_weights((hidden_size, hidden_size)),
            w_rec_output: init_weights((hidden_size, hidden_size)),
            w_rec_cell: init_weights((hidden_size, hidden_size)),
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

impl SpikingLayer for SpikingLSTM {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, _input_size) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        self.ensure_state(batch_size);

        let hidden_size = self.w_input_gate.shape()[0];
        let mut output = Array3::zeros((batch_size, num_steps, hidden_size));

        // Simplified LSTM dynamics with spikes
        for t in 0..num_steps {
            let input_t = input_dense.slice(s![.., t, ..]).to_owned();
            let h_prev = self.hidden_state.as_ref().unwrap().clone();
            let c_prev = self.cell_state.as_ref().unwrap().clone();

            let mut h_new = Array2::zeros((batch_size, hidden_size));
            let mut c_new = Array2::zeros((batch_size, hidden_size));

            for b in 0..batch_size {
                // Compute gates (simplified with sigmoid approximation)
                let i_gate = self.w_input_gate.dot(&input_t.slice(s![b, ..]))
                    + self.w_rec_input.dot(&h_prev.slice(s![b, ..]));
                let f_gate = self.w_forget_gate.dot(&input_t.slice(s![b, ..]))
                    + self.w_rec_forget.dot(&h_prev.slice(s![b, ..]));
                let o_gate = self.w_output_gate.dot(&input_t.slice(s![b, ..]))
                    + self.w_rec_output.dot(&h_prev.slice(s![b, ..]));
                let c_gate = self.w_cell_gate.dot(&input_t.slice(s![b, ..]))
                    + self.w_rec_cell.dot(&h_prev.slice(s![b, ..]));

                // Apply sigmoid-like activation (spike-based gating)
                let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

                for i in 0..hidden_size {
                    let i_val = sigmoid(i_gate[i]);
                    let f_val = sigmoid(f_gate[i]);
                    let o_val = sigmoid(o_gate[i]);
                    let c_val = (c_gate[i]).tanh();

                    c_new[[b, i]] = f_val * c_prev[[b, i]] + i_val * c_val;
                    h_new[[b, i]] = o_val * c_new[[b, i]].tanh();
                }

                // Convert to spikes using neuron dynamics
                let spikes = self.state[b].update_lif(&h_new.slice(s![b, ..]).to_owned(),
                    &self.neuron_params, self.dt);
                output.slice_mut(s![b, t, ..]).assign(&spikes);
            }

            self.cell_state = Some(c_new);
            self.hidden_state = Some(h_new);
        }

        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {
        for state in &mut self.state {
            state.reset();
        }
        self.cell_state = None;
        self.hidden_state = None;
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        vec![
            &self.w_input_gate,
            &self.w_forget_gate,
            &self.w_output_gate,
            &self.w_cell_gate,
        ]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![
            &mut self.w_input_gate,
            &mut self.w_forget_gate,
            &mut self.w_output_gate,
            &mut self.w_cell_gate,
        ]
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        vec![None, None, None, None] // Simplified
    }

    fn zero_grad(&mut self) {
        // Simplified - no gradient storage in this implementation
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
}
