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
        // One implementation per kind. Triangle used to resolve to a box and
        // Exponential to SuperSpike, both marked "Simplified", so two of the
        // five kinds silently selected a different function than the one named.
        let surrogate: Box<dyn SurrogateGradient> = match surrogate_type {
            SurrogateType::Box => Box::new(BoxSurrogate::new(2.0)),
            SurrogateType::Triangle => Box::new(TriangleSurrogate::new(1.0)),
            SurrogateType::FastSigmoid => Box::new(FastSigmoidSurrogate::new(10.0)),
            SurrogateType::Exponential => Box::new(ExponentialSurrogate::new(1.0, 5.0)),
            SurrogateType::SuperSpike => Box::new(SuperSpikeSurrogate::new(10.0)),
        };

        Self {
            surrogate,
            num_steps,
        }
    }

    /// Applies the surrogate across the time axis, within the truncation window.
    ///
    /// `output_grad` and `v_mem` are both `(batch, steps, neurons)`. Each
    /// gradient element is scaled by the surrogate evaluated at the membrane
    /// potential of *that same* element. Steps older than `num_steps` from the
    /// end receive zero, which is what truncating backpropagation through time
    /// means: the window is where gradient is allowed to flow.
    ///
    /// What this does NOT do is propagate gradient between steps. It cannot:
    /// carrying `dL/dv[t]` back to `v[t-1]` needs the recurrent weights and the
    /// membrane decay, and this type holds neither. That propagation lives in
    /// the layers' own backward passes -- [`crate::layers::SpikingRNN::backward`]
    /// and its siblings -- which have the parameters to do it.
    ///
    /// The previous version took a slice of per-step arrays and, for each step
    /// `t`, scaled *every* element of a full gradient copy by the membrane at
    /// that one step: the loop discarded each element's own time index
    /// (`for ((b, _, n), g) in ...`) and substituted the outer `t`. It also
    /// indexed `v_mem_history[t][[b, t, n]]`, so a genuine per-step history --
    /// arrays of one step each -- panicked for every `t > 0`.
    pub fn backward(
        &self,
        output_grad: &Array3<f32>,
        v_mem: &Array3<f32>,
        threshold: f32,
    ) -> SNNResult<Array3<f32>> {
        if output_grad.shape() != v_mem.shape() {
            return Err(crate::SNNError::DimensionMismatch {
                expected: format!("membrane history shaped {:?}", output_grad.shape()),
                actual: format!("{:?}", v_mem.shape()),
            });
        }

        let num_steps = output_grad.shape()[1];
        let window = self.num_steps.unwrap_or(num_steps).min(num_steps);
        let first = num_steps - window;

        let mut out = Array3::zeros(output_grad.raw_dim());
        for ((b, t, n), slot) in out.indexed_iter_mut() {
            if t < first {
                continue;
            }
            *slot = output_grad[[b, t, n]]
                * self.surrogate.compute_gradient(v_mem[[b, t, n]], threshold);
        }
        Ok(out)
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

    // ---- BPTT ----------------------------------------------------------

    /// Each gradient element must be scaled by the membrane at its OWN step.
    ///
    /// The old loop discarded the element's time index and used the outer
    /// loop's instead, so every step was scaled by one step's membrane.
    #[test]
    fn bptt_scales_each_step_by_its_own_membrane() {
        let (batch, steps, neurons) = (1, 4, 1);
        let bptt = BPTT::new(SurrogateType::Box, None);
        let threshold = 1.0;

        // Only step 2 sits inside the Box window; the rest are far away.
        let mut v = Array3::from_elem((batch, steps, neurons), 50.0);
        v[[0, 2, 0]] = threshold;
        let grad = Array3::from_elem((batch, steps, neurons), 1.0);

        let out = bptt.backward(&grad, &v, threshold).unwrap();
        assert!(out[[0, 2, 0]] > 0.0, "the in-window step got no gradient");
        for t in [0usize, 1, 3] {
            assert_eq!(
                out[[0, t, 0]],
                0.0,
                "step {t} is far from threshold and should have been zeroed, \
                 but was scaled by another step's membrane"
            );
        }
    }

    /// The truncation window must actually truncate.
    #[test]
    fn bptt_truncation_window_zeroes_older_steps() {
        let (batch, steps, neurons) = (1, 6, 1);
        let bptt = BPTT::new(SurrogateType::Box, Some(2));
        let threshold = 1.0;

        // Every step is in the surrogate's window, so only truncation can
        // zero anything.
        let v = Array3::from_elem((batch, steps, neurons), threshold);
        let grad = Array3::from_elem((batch, steps, neurons), 1.0);

        let out = bptt.backward(&grad, &v, threshold).unwrap();
        for t in 0..4 {
            assert_eq!(out[[0, t, 0]], 0.0, "step {t} is outside the window");
        }
        for t in 4..6 {
            assert!(out[[0, t, 0]] > 0.0, "step {t} is inside the window");
        }
    }

    /// A per-step history is a shape error, not a panic.
    #[test]
    fn bptt_reports_a_shape_mismatch() {
        let bptt = BPTT::new(SurrogateType::FastSigmoid, None);
        let grad = Array3::zeros((1, 4, 2));
        let wrong = Array3::zeros((1, 1, 2));
        assert!(bptt.backward(&grad, &wrong, 1.0).is_err());
    }

    /// Each surrogate kind must select its own implementation here too.
    #[test]
    fn bptt_surrogate_kinds_are_distinct() {
        let probes = [0.2f32, 0.6, 0.95, 1.4, 2.0];
        let mut seen: Vec<Vec<f32>> = Vec::new();
        for kind in [
            SurrogateType::Box,
            SurrogateType::Triangle,
            SurrogateType::FastSigmoid,
            SurrogateType::Exponential,
            SurrogateType::SuperSpike,
        ] {
            let b = BPTT::new(kind, None);
            seen.push(
                probes
                    .iter()
                    .map(|v| b.surrogate.compute_gradient(*v, 1.0))
                    .collect(),
            );
        }
        for i in 0..seen.len() {
            for j in (i + 1)..seen.len() {
                assert_ne!(seen[i], seen[j], "two kinds resolve to the same function");
            }
        }
    }
}
