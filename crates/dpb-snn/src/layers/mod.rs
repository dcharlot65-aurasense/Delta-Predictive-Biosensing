//! Neural network layers for Spiking Neural Networks

pub mod linear;
pub mod conv;
pub mod pool;
pub mod recurrent;
pub mod attention;

pub use linear::SpikingLinear;
pub use conv::{SpikingConv1d, SpikingConv2d};
pub use pool::{SpikingSumPool2d, SpikingMaxPool2d};
pub use recurrent::{SpikingRNN, SpikingLSTM};
pub use attention::SpikingAttention;

use crate::{NeuronParams, SNNResult, SpikeTensor};
use ndarray::{Array, Array1, Array2};
use serde::{Deserialize, Serialize};

/// Base trait for all spiking layers
pub trait SpikingLayer: Send + Sync {
    /// Forward pass through the layer
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor>;

    /// Reset layer state
    fn reset_state(&mut self);

    /// Get layer parameters (for optimization)
    fn parameters(&self) -> Vec<&Array2<f32>>;

    /// Get mutable layer parameters
    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>>;

    /// Get layer gradients
    fn gradients(&self) -> Vec<Option<&Array2<f32>>>;

    /// Zero gradients
    fn zero_grad(&mut self);
}

/// Neuron state for stateful layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    /// Membrane potential
    pub v_mem: Array1<f32>,
    /// Synaptic current
    pub i_syn: Array1<f32>,
    /// Refractory counter
    pub refrac: Array1<f32>,
    /// Adaptive threshold (for adaptive neurons)
    pub threshold_adapt: Option<Array1<f32>>,
}

impl NeuronState {
    /// Create new neuron state
    pub fn new(num_neurons: usize, adaptive: bool) -> Self {
        Self {
            v_mem: Array1::zeros(num_neurons),
            i_syn: Array1::zeros(num_neurons),
            refrac: Array1::zeros(num_neurons),
            threshold_adapt: if adaptive {
                Some(Array1::zeros(num_neurons))
            } else {
                None
            },
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.v_mem.fill(0.0);
        self.i_syn.fill(0.0);
        self.refrac.fill(0.0);
        if let Some(ref mut adapt) = self.threshold_adapt {
            adapt.fill(0.0);
        }
    }

    /// Update neuron dynamics (LIF model)
    pub fn update_lif(
        &mut self,
        input_current: &Array1<f32>,
        params: &NeuronParams,
        dt: f32,
    ) -> Array1<f32> {
        let num_neurons = self.v_mem.len();
        let mut spikes = Array1::zeros(num_neurons);

        // Update synaptic current
        let alpha_syn = (-dt / params.tau_syn).exp();
        self.i_syn = &self.i_syn * alpha_syn + input_current;

        // Update membrane potential (only for non-refractory neurons)
        let alpha_mem = (-dt / params.tau_mem).exp();
        for i in 0..num_neurons {
            if self.refrac[i] <= 0.0 {
                self.v_mem[i] = self.v_mem[i] * alpha_mem + self.i_syn[i] * (1.0 - alpha_mem);

                // Check for spike
                let threshold = params.v_threshold
                    + self.threshold_adapt.as_ref().map(|a| a[i]).unwrap_or(0.0);

                if self.v_mem[i] >= threshold {
                    spikes[i] = 1.0;
                    self.v_mem[i] = params.v_reset;
                    self.refrac[i] = params.t_refrac;

                    // Update adaptive threshold
                    if let Some(ref mut adapt) = self.threshold_adapt {
                        adapt[i] += 0.1; // Simple adaptation increment
                    }
                }
            } else {
                self.refrac[i] -= dt;
            }
        }

        // Decay adaptive threshold
        if let Some(ref mut adapt) = self.threshold_adapt {
            *adapt *= (1.0 - dt / 100.0); // Slow decay
        }

        spikes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_state_creation() {
        let state = NeuronState::new(10, false);
        assert_eq!(state.v_mem.len(), 10);
        assert!(state.threshold_adapt.is_none());

        let state_adaptive = NeuronState::new(10, true);
        assert!(state_adaptive.threshold_adapt.is_some());
    }

    #[test]
    fn test_neuron_state_reset() {
        let mut state = NeuronState::new(5, false);
        state.v_mem[0] = 0.5;
        state.i_syn[1] = 0.3;

        state.reset();
        assert_eq!(state.v_mem[0], 0.0);
        assert_eq!(state.i_syn[1], 0.0);
    }

    #[test]
    fn test_lif_dynamics() {
        let mut state = NeuronState::new(3, false);
        let params = NeuronParams::default();
        let input = Array1::from_vec(vec![0.5, 1.0, 0.3]);

        let spikes = state.update_lif(&input, &params, 1.0);

        // Membrane potential should increase
        assert!(state.v_mem[1] > 0.0);
        // Most likely no spikes on first step
        assert_eq!(spikes.sum(), 0.0);
    }
}
