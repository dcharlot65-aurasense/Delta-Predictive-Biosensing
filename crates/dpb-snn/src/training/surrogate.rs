//! Surrogate gradient methods for training SNNs

use crate::{SNNResult, SpikeTensor};
use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// Types of surrogate gradient functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurrogateType {
    /// Rectangular/box surrogate
    Box,
    /// Triangular surrogate
    Triangle,
    /// Fast sigmoid surrogate
    FastSigmoid,
    /// Exponential surrogate
    Exponential,
    /// SuperSpike surrogate
    SuperSpike,
}

/// Surrogate gradient function trait
pub trait SurrogateGradient: Send + Sync {
    /// Compute surrogate gradient for membrane potential
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32;

    /// Apply surrogate gradient to spike tensor
    fn apply(
        &self,
        spikes: &SpikeTensor,
        v_mem: &Array3<f32>,
        threshold: f32,
    ) -> SNNResult<Array3<f32>>;
}

/// Box surrogate gradient (rectangular window).
///
/// Zero outside `width / 2` of the threshold. That bound is the point of the
/// surrogate, but it also means a neuron whose membrane never comes that close
/// receives no gradient at all and cannot recover -- the dead-neuron problem.
/// If every neuron in a layer is outside the window the layer stops training
/// silently; [`crate::training::StepReport::grad_norm`] going to exactly zero
/// is how that shows up. [`TriangleSurrogate`] has twice the support for the
/// same parameter and [`ExponentialSurrogate`] is never exactly zero, so either
/// is a safer default when a network stops learning for no visible reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxSurrogate {
    /// Width of the box
    pub width: f32,
}

impl BoxSurrogate {
    pub fn new(width: f32) -> Self {
        Self { width }
    }
}

impl SurrogateGradient for BoxSurrogate {
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32 {
        let dist = (v_mem - threshold).abs();
        if dist < self.width / 2.0 {
            1.0 / self.width
        } else {
            0.0
        }
    }

    fn apply(
        &self,
        spikes: &SpikeTensor,
        v_mem: &Array3<f32>,
        threshold: f32,
    ) -> SNNResult<Array3<f32>> {
        let spike_dense = spikes.to_dense();
        let mut gradient = Array3::zeros(spike_dense.raw_dim());

        for ((b, t, n), &spike) in spike_dense.indexed_iter() {
            if spike > 0.5 {
                gradient[[b, t, n]] = self.compute_gradient(v_mem[[b, t, n]], threshold);
            }
        }

        Ok(gradient)
    }
}

/// Triangular surrogate gradient.
///
/// Piecewise linear, peaking at the threshold and falling to zero at
/// `half_width` either side. Its support is `2 * half_width`, against the box's
/// `width`, so it keeps passing gradient where a same-width box has already cut
/// off -- which matters because a box that is everywhere zero makes training a
/// silent no-op.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriangleSurrogate {
    /// Distance from the threshold at which the gradient reaches zero.
    pub half_width: f32,
}

impl TriangleSurrogate {
    pub fn new(half_width: f32) -> Self {
        Self { half_width }
    }
}

impl Default for TriangleSurrogate {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl SurrogateGradient for TriangleSurrogate {
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32 {
        if self.half_width <= 0.0 {
            return 0.0;
        }
        let dist = (v_mem - threshold).abs();
        (1.0 - dist / self.half_width).max(0.0)
    }

    fn apply(
        &self,
        spikes: &SpikeTensor,
        v_mem: &Array3<f32>,
        threshold: f32,
    ) -> SNNResult<Array3<f32>> {
        apply_elementwise(self, spikes, v_mem, threshold)
    }
}

/// Exponential surrogate gradient, `alpha * exp(-beta * |v - threshold|)`.
///
/// Never exactly zero, so a neuron far from threshold still receives some
/// gradient and can recover. That is the dead-neuron problem the bounded
/// surrogates suffer from, and the reason to reach for this one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialSurrogate {
    /// Peak value, at the threshold.
    pub alpha: f32,
    /// Decay rate away from the threshold.
    pub beta: f32,
}

impl ExponentialSurrogate {
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self { alpha, beta }
    }
}

impl Default for ExponentialSurrogate {
    fn default() -> Self {
        Self::new(1.0, 5.0)
    }
}

impl SurrogateGradient for ExponentialSurrogate {
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32 {
        self.alpha * (-self.beta * (v_mem - threshold).abs()).exp()
    }

    fn apply(
        &self,
        spikes: &SpikeTensor,
        v_mem: &Array3<f32>,
        threshold: f32,
    ) -> SNNResult<Array3<f32>> {
        apply_elementwise(self, spikes, v_mem, threshold)
    }
}

/// Evaluates a surrogate at every membrane sample.
///
/// Deliberately not gated on whether the neuron spiked. A surrogate exists to
/// supply a derivative where the true one is zero -- above all for a neuron
/// sitting just below threshold, which is the case a spike gate excludes.
fn apply_elementwise(
    surrogate: &dyn SurrogateGradient,
    spikes: &SpikeTensor,
    v_mem: &Array3<f32>,
    threshold: f32,
) -> SNNResult<Array3<f32>> {
    let dense = spikes.to_dense();
    if dense.shape() != v_mem.shape() {
        return Err(crate::SNNError::DimensionMismatch {
            expected: format!("membrane history shaped {:?}", dense.shape()),
            actual: format!("{:?}", v_mem.shape()),
        });
    }
    let mut gradient = Array3::zeros(dense.raw_dim());
    for ((b, t, n), g) in gradient.indexed_iter_mut() {
        *g = surrogate.compute_gradient(v_mem[[b, t, n]], threshold);
    }
    Ok(gradient)
}

/// Fast sigmoid surrogate gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastSigmoidSurrogate {
    /// Slope parameter
    pub beta: f32,
}

impl FastSigmoidSurrogate {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}

impl SurrogateGradient for FastSigmoidSurrogate {
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32 {
        let x = self.beta * (v_mem - threshold);
        self.beta / (1.0 + x.abs()).powi(2)
    }

    fn apply(
        &self,
        spikes: &SpikeTensor,
        v_mem: &Array3<f32>,
        threshold: f32,
    ) -> SNNResult<Array3<f32>> {
        let spike_dense = spikes.to_dense();
        let mut gradient = Array3::zeros(spike_dense.raw_dim());

        for ((b, t, n), _) in spike_dense.indexed_iter() {
            gradient[[b, t, n]] = self.compute_gradient(v_mem[[b, t, n]], threshold);
        }

        Ok(gradient)
    }
}

/// SuperSpike surrogate gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperSpikeSurrogate {
    /// Scale parameter
    pub beta: f32,
}

impl SuperSpikeSurrogate {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}

impl SurrogateGradient for SuperSpikeSurrogate {
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32 {
        let x = self.beta * (v_mem - threshold);
        1.0 / (1.0 + x.abs()).powi(2)
    }

    fn apply(
        &self,
        spikes: &SpikeTensor,
        v_mem: &Array3<f32>,
        threshold: f32,
    ) -> SNNResult<Array3<f32>> {
        let spike_dense = spikes.to_dense();
        let mut gradient = Array3::zeros(spike_dense.raw_dim());

        for ((b, t, n), _) in spike_dense.indexed_iter() {
            gradient[[b, t, n]] = self.compute_gradient(v_mem[[b, t, n]], threshold);
        }

        Ok(gradient)
    }
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
/// Backpropagation Through Time (BPTT) for SNNs
pub struct BPTT {
    /// Surrogate gradient function
    pub surrogate: Box<dyn SurrogateGradient>,
    /// Number of time steps to backprop through
    pub num_steps: Option<usize>,
}

impl std::fmt::Debug for BPTT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BPTT")
            .field("num_steps", &self.num_steps)
            .finish_non_exhaustive()
    }
}

impl BPTT {
    pub fn new(surrogate_type: SurrogateType, num_steps: Option<usize>) -> Self {
        let surrogate: Box<dyn SurrogateGradient> = match surrogate_type {
            SurrogateType::Box => Box::new(BoxSurrogate::new(1.0)),
            SurrogateType::Triangle => Box::new(BoxSurrogate::new(1.0)), // Simplified
            SurrogateType::FastSigmoid => Box::new(FastSigmoidSurrogate::new(10.0)),
            SurrogateType::Exponential => Box::new(SuperSpikeSurrogate::new(10.0)), // Simplified
            SurrogateType::SuperSpike => Box::new(SuperSpikeSurrogate::new(10.0)),
        };

        Self {
            surrogate,
            num_steps,
        }
    }

    /// Compute gradients through time
    pub fn backward(
        &self,
        output_grad: &Array3<f32>,
        v_mem_history: &[Array3<f32>],
        threshold: f32,
    ) -> SNNResult<Vec<Array3<f32>>> {
        let num_actual_steps = v_mem_history.len();
        let bptt_steps = self
            .num_steps
            .unwrap_or(num_actual_steps)
            .min(num_actual_steps);

        let mut gradients = vec![Array3::zeros(output_grad.raw_dim()); num_actual_steps];

        // Backpropagate through time
        for t in (num_actual_steps - bptt_steps..num_actual_steps).rev() {
            // Compute surrogate gradient for this time step
            let v_mem = &v_mem_history[t];
            let mut grad_t = output_grad.clone();

            // Apply surrogate gradient
            for ((b, _, n), grad_val) in grad_t.indexed_iter_mut() {
                *grad_val *= self.surrogate.compute_gradient(v_mem[[b, t, n]], threshold);
            }

            gradients[t] = grad_t;
        }

        Ok(gradients)
    }
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
/// Online Training Through Time (OTTT)
/// More memory efficient than full BPTT
pub struct OTTT {
    /// Surrogate gradient function
    pub surrogate: Box<dyn SurrogateGradient>,
    /// Eligibility trace decay
    pub trace_decay: f32,
}

impl std::fmt::Debug for OTTT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OTTT")
            .field("trace_decay", &self.trace_decay)
            .finish_non_exhaustive()
    }
}

impl OTTT {
    pub fn new(surrogate_type: SurrogateType, trace_decay: f32) -> Self {
        let surrogate: Box<dyn SurrogateGradient> = match surrogate_type {
            SurrogateType::Box => Box::new(BoxSurrogate::new(1.0)),
            SurrogateType::FastSigmoid => Box::new(FastSigmoidSurrogate::new(10.0)),
            SurrogateType::SuperSpike => Box::new(SuperSpikeSurrogate::new(10.0)),
            _ => Box::new(SuperSpikeSurrogate::new(10.0)),
        };

        Self {
            surrogate,
            trace_decay,
        }
    }
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
/// Spatial Layer-wise Training Through Time (SLTT)
/// Train layers independently to reduce computational cost
pub struct SLTT {
    /// Surrogate gradient function
    pub surrogate: Box<dyn SurrogateGradient>,
    /// Number of layers to train simultaneously
    pub layer_window: usize,
}

impl std::fmt::Debug for SLTT {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SLTT")
            .field("layer_window", &self.layer_window)
            .finish_non_exhaustive()
    }
}

impl SLTT {
    pub fn new(surrogate_type: SurrogateType, layer_window: usize) -> Self {
        let surrogate: Box<dyn SurrogateGradient> = match surrogate_type {
            SurrogateType::Box => Box::new(BoxSurrogate::new(1.0)),
            SurrogateType::FastSigmoid => Box::new(FastSigmoidSurrogate::new(10.0)),
            SurrogateType::SuperSpike => Box::new(SuperSpikeSurrogate::new(10.0)),
            _ => Box::new(SuperSpikeSurrogate::new(10.0)),
        };

        Self {
            surrogate,
            layer_window,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_surrogate() {
        let surrogate = BoxSurrogate::new(1.0);
        let threshold = 1.0;

        // Within window
        let grad1 = surrogate.compute_gradient(1.2, threshold);
        assert!(grad1 > 0.0);

        // Outside window
        let grad2 = surrogate.compute_gradient(2.0, threshold);
        assert_eq!(grad2, 0.0);
    }

    #[test]
    fn test_fast_sigmoid_surrogate() {
        let surrogate = FastSigmoidSurrogate::new(10.0);
        let threshold = 1.0;

        let grad1 = surrogate.compute_gradient(1.0, threshold);
        assert!(grad1 > 0.0);

        let grad2 = surrogate.compute_gradient(1.1, threshold);
        assert!(grad2 > 0.0);
    }

    #[test]
    fn test_superspike_surrogate() {
        let surrogate = SuperSpikeSurrogate::new(10.0);
        let threshold = 1.0;

        let grad = surrogate.compute_gradient(1.1, threshold);
        assert!(grad > 0.0 && grad <= 1.0);
    }

    #[test]
    fn test_bptt_creation() {
        let bptt = BPTT::new(SurrogateType::FastSigmoid, Some(20));
        assert_eq!(bptt.num_steps, Some(20));
    }
}
