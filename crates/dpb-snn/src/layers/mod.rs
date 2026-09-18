//! Neural network layers for Spiking Neural Networks

pub mod attention;
pub mod conv;
pub mod linear;
pub mod pool;
pub mod recurrent;

pub use attention::SpikingAttention;
pub use conv::{ConvTrace, SpikingConv1d, SpikingConv2d};
pub use linear::SpikingLinear;
pub use pool::{SpikingMaxPool2d, SpikingSumPool2d};
pub use recurrent::{RecurrentTrace, SpikingLSTM, SpikingRNN};

use crate::{NeuronParams, SNNResult, SpikeTensor};
use ndarray::{Array1, Array2};
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

    /// Total number of trainable parameters in this layer.
    ///
    /// The default sums [`Self::parameters`], which is right only for a layer
    /// whose parameters are all rank 2. That signature yields `Array2`, so it
    /// cannot carry a bias (rank 1) or a convolution kernel (rank 3 or 4):
    /// counting a `SpikingLinear` through it drops the bias, and counting a
    /// convolution through it returns zero for the whole layer. Every layer
    /// holding either overrides this.
    ///
    /// Six `num_parameters` implementations in the fusion module summed
    /// `parameters()` directly and therefore under-reported, a convolution
    /// contributing nothing at all.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.len()).sum()
    }
}

/// Computes `w^T . v` by accumulating over the rows of `w`.
///
/// Mathematically identical to `w.t().dot(v)`, and measurably faster: `w.t()`
/// is a transposed view whose memory access is strided, which drops ndarray off
/// its contiguous path, while `w.row(i)` is contiguous and each step becomes a
/// plain axpy. Measured at 8.8x to 11.9x on square and rectangular weights of
/// 128 to 1024, with results agreeing exactly.
///
/// It matters because the backward passes call this once per (batch, step): a
/// recurrent layer over 32 steps with a batch of 16 does it 512 times, and the
/// LSTM eight times that.
///
/// Rows whose coefficient is zero are skipped -- surrogate gradients are zero
/// wherever the membrane sat outside the surrogate's support, which for a
/// bounded surrogate is most of them.
pub(crate) fn transpose_dot<S>(
    w: &Array2<f32>,
    v: &ndarray::ArrayBase<S, ndarray::Ix1>,
) -> Array1<f32>
where
    S: ndarray::Data<Elem = f32>,
{
    let mut out = Array1::zeros(w.shape()[1]);
    accumulate_transpose_dot(&mut out, w, v);
    out
}

/// Adds `w^T . v` into `out`, for callers summing several such products.
pub(crate) fn accumulate_transpose_dot<S>(
    out: &mut Array1<f32>,
    w: &Array2<f32>,
    v: &ndarray::ArrayBase<S, ndarray::Ix1>,
) where
    S: ndarray::Data<Elem = f32>,
{
    for (i, &coefficient) in v.iter().enumerate() {
        if coefficient != 0.0 {
            out.scaled_add(coefficient, &w.row(i));
        }
    }
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
        self.update_lif_recording(input_current, params, dt).0
    }

    /// Advance one step, also returning the membrane potential each neuron
    /// reached *before* any spike reset was applied.
    ///
    /// Surrogate gradients are a function of how far the potential sat from
    /// threshold at the moment of comparison. [`Self::update_lif`] overwrites
    /// `v_mem` with `v_reset` for every neuron that fired, so that value cannot
    /// be recovered afterwards -- it has to be captured here. Training uses
    /// this; inference uses the simpler wrapper above.
    pub fn update_lif_recording(
        &mut self,
        input_current: &Array1<f32>,
        params: &NeuronParams,
        dt: f32,
    ) -> (Array1<f32>, Array1<f32>) {
        let num_neurons = self.v_mem.len();
        let mut spikes = Array1::zeros(num_neurons);
        // Seeded with the current potential so refractory neurons -- which skip
        // the update below -- still report the value they are holding.
        let mut v_pre_reset = self.v_mem.clone();

        // Update synaptic current
        let alpha_syn = (-dt / params.tau_syn).exp();
        self.i_syn = &self.i_syn * alpha_syn + input_current;

        // Update membrane potential (only for non-refractory neurons)
        let alpha_mem = (-dt / params.tau_mem).exp();
        for i in 0..num_neurons {
            if self.refrac[i] <= 0.0 {
                self.v_mem[i] = self.v_mem[i] * alpha_mem + self.i_syn[i] * (1.0 - alpha_mem);

                // Check for spike
                let threshold =
                    params.v_threshold + self.threshold_adapt.as_ref().map(|a| a[i]).unwrap_or(0.0);

                v_pre_reset[i] = self.v_mem[i];

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
            *adapt *= 1.0 - dt / 100.0; // Slow decay
        }

        (spikes, v_pre_reset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parameter counts must include everything the layer actually holds.
    ///
    /// `parameters()` yields `Array2`, so summing it drops every bias and
    /// returns zero for a convolution -- whose kernel is rank 3 or 4 -- and
    /// the fusion module's `num_parameters` did exactly that. A convolution
    /// contributed nothing to the total.
    #[test]
    fn parameter_counts_include_biases_and_kernels() {
        let params = NeuronParams::default();

        let linear = SpikingLinear::new(16, 8, true, params.clone(), 1.0, false);
        assert_eq!(linear.num_parameters(), 16 * 8 + 8, "weights plus bias");
        assert_eq!(
            linear.parameters().iter().map(|p| p.len()).sum::<usize>(),
            16 * 8,
            "the rank-2 view still cannot see the bias"
        );

        let conv = SpikingConv2d::new(
            3,
            16,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
            params.clone(),
            1.0,
            false,
        );
        assert_eq!(
            conv.num_parameters(),
            16 * 3 * 3 * 3 + 16,
            "kernel plus bias"
        );
        assert_eq!(
            conv.parameters().iter().map(|p| p.len()).sum::<usize>(),
            0,
            "a convolution is invisible to the rank-2 view, which is the point"
        );

        let conv1 = SpikingConv1d::new(4, 8, 5, 1, 2, true, params.clone(), 1.0, false);
        assert_eq!(conv1.num_parameters(), 8 * 4 * 5 + 8);

        let rnn = SpikingRNN::new(16, 12, true, params.clone(), 1.0, false);
        assert_eq!(rnn.num_parameters(), 16 * 12 + 12 * 12 + 12);

        // Multi-head attention must count its heads, not just the output
        // projection it happens to expose.
        let heads = 4;
        let d_model = 16;
        let multi =
            attention::MultiHeadSpikingAttention::new(d_model, heads, params.clone(), 1.0, false);
        let one_head = SpikingAttention::new(d_model, 1, params, 1.0, false).num_parameters();
        assert_eq!(
            multi.num_parameters(),
            heads * one_head + d_model * d_model,
            "heads plus the output projection"
        );
        assert!(
            multi.num_parameters() > multi.parameters().iter().map(|p| p.len()).sum::<usize>(),
            "the rank-2 view sees only the output projection"
        );

        // A layer with no parameters reports none rather than guessing.
        let pool = SpikingMaxPool2d::new((2, 2), (2, 2));
        assert_eq!(pool.num_parameters(), 0);
    }

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
