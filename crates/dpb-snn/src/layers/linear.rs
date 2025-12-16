//! Linear (fully-connected) spiking layer

use super::{NeuronState, SpikingLayer};
use crate::{NeuronParams, SNNError, SNNResult, SpikeTensor};
use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Linear spiking layer (fully connected)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingLinear {
    /// Weight matrix (output_size, input_size)
    pub weights: Array2<f32>,
    /// Bias (optional)
    pub bias: Option<Array1<f32>>,
    /// Weight gradients
    #[serde(skip)]
    pub weight_grad: Option<Array2<f32>>,
    /// Bias gradients
    #[serde(skip)]
    pub bias_grad: Option<Array1<f32>>,
    /// Neuron parameters
    pub neuron_params: NeuronParams,
    /// Neuron state (per batch)
    #[serde(skip)]
    pub state: Vec<NeuronState>,
    /// Time step size
    pub dt: f32,
    /// Use adaptive neurons
    pub adaptive: bool,
}

impl SpikingLinear {
    /// Create a new linear spiking layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        use_bias: bool,
        neuron_params: NeuronParams,
        dt: f32,
        adaptive: bool,
    ) -> Self {
        // Initialize weights with Xavier/Glorot initialization
        let std_dev = (2.0 / (input_size + output_size) as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        let mut rng = thread_rng();

        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            normal.sample(&mut rng)
        });

        let bias = if use_bias {
            Some(Array1::zeros(output_size))
        } else {
            None
        };

        Self {
            weights,
            bias,
            weight_grad: None,
            bias_grad: None,
            neuron_params,
            state: Vec::new(),
            dt,
            adaptive,
        }
    }

    /// Initialize state for a given batch size
    fn ensure_state(&mut self, batch_size: usize) {
        if self.state.len() != batch_size {
            self.state = (0..batch_size)
                .map(|_| NeuronState::new(self.weights.shape()[0], self.adaptive))
                .collect();
        }
    }

    /// Compute forward pass for a single time step
    fn forward_step(&mut self, input: &Array2<f32>) -> SNNResult<Array2<f32>> {
        let batch_size = input.shape()[0];
        self.ensure_state(batch_size);

        let mut output = Array2::zeros((batch_size, self.weights.shape()[0]));

        for b in 0..batch_size {
            // Compute synaptic input: W * x
            let input_slice = input.slice(s![b, ..]);
            let mut synaptic_input = self.weights.dot(&input_slice);

            // Add bias if present
            if let Some(ref bias) = self.bias {
                synaptic_input += bias;
            }

            // Update neuron dynamics and get spikes
            let spikes = self.state[b].update_lif(&synaptic_input, &self.neuron_params, self.dt);

            output.slice_mut(s![b, ..]).assign(&spikes);
        }

        Ok(output)
    }

    /// Backward pass to compute gradients
    pub fn backward(
        &mut self,
        input: &Array3<f32>,
        output_grad: &Array3<f32>,
    ) -> SNNResult<Array3<f32>> {
        let (batch_size, num_steps, input_size) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
        );
        let output_size = self.weights.shape()[0];

        // Initialize gradients
        let mut weight_grad = Array2::zeros(self.weights.raw_dim());
        let mut bias_grad = if self.bias.is_some() {
            Some(Array1::zeros(output_size))
        } else {
            None
        };
        let mut input_grad = Array3::zeros((batch_size, num_steps, input_size));

        // Backpropagate through time
        for t in (0..num_steps).rev() {
            for b in 0..batch_size {
                let input_t = input.slice(s![b, t, ..]);
                let out_grad_t = output_grad.slice(s![b, t, ..]);

                // Weight gradient: outer product of output_grad and input
                for i in 0..output_size {
                    for j in 0..input_size {
                        weight_grad[[i, j]] += out_grad_t[i] * input_t[j];
                    }
                }

                // Bias gradient
                if let Some(ref mut bg) = bias_grad {
                    for i in 0..output_size {
                        bg[i] += out_grad_t[i];
                    }
                }

                // Input gradient: W^T * output_grad
                let in_grad = self.weights.t().dot(&out_grad_t);
                input_grad.slice_mut(s![b, t, ..]).assign(&in_grad);
            }
        }

        // Store gradients
        self.weight_grad = Some(weight_grad);
        self.bias_grad = bias_grad;

        Ok(input_grad)
    }
}

impl SpikingLayer for SpikingLinear {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, input_size) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        if input_size != self.weights.shape()[1] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("input size {}", self.weights.shape()[1]),
                actual: format!("input size {}", input_size),
            });
        }

        let output_size = self.weights.shape()[0];
        let mut output = Array3::zeros((batch_size, num_steps, output_size));

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
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        vec![&self.weights]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![&mut self.weights]
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        vec![self.weight_grad.as_ref()]
    }

    fn zero_grad(&mut self) {
        self.weight_grad = None;
        self.bias_grad = None;
    }
}

impl Default for SpikingLinear {
    fn default() -> Self {
        // Minimal default - creates a 1x1 layer
        Self::new(1, 1, false, NeuronParams::default(), 1.0, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let layer = SpikingLinear::new(10, 5, true, NeuronParams::default(), 1.0, false);
        assert_eq!(layer.weights.shape(), &[5, 10]);
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_linear_forward() {
        let mut layer = SpikingLinear::new(10, 5, true, NeuronParams::default(), 1.0, false);
        let input = SpikeTensor::zeros(2, 10, 10, false);

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 10, 5));
    }

    #[test]
    fn test_linear_state_reset() {
        let mut layer = SpikingLinear::new(5, 3, false, NeuronParams::default(), 1.0, false);
        let input = SpikeTensor::zeros(1, 5, 5, false);

        // Forward pass to create state
        let _ = layer.forward(&input).unwrap();
        assert_eq!(layer.state.len(), 1);

        // Reset state
        layer.reset_state();
        assert!(layer.state[0].v_mem.iter().all(|&v| v == 0.0));
    }
}
