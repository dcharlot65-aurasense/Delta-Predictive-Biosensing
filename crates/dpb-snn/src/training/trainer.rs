//! Supervised training for feed-forward spiking networks.
//!
//! The pieces this composes already existed separately -- [`LossFunction`]
//! yields a gradient with respect to output spikes, [`SurrogateGradient`]
//! supplies a derivative for the non-differentiable spike, [`SpikingLinear`]
//! turns an output gradient into weight and input gradients, and [`Optimizer`]
//! applies the result. What was missing was the loop that runs them in order,
//! which is why `Trainer.fit` in the Python bindings had nothing to call.
//!
//! # What it trains
//!
//! A stack of [`SpikingLinear`] layers, which is the topology the loss
//! functions and the layer backward pass are written for. Convolutional and
//! recurrent layers have no backward pass yet, so they are not accepted rather
//! than silently mistrained.
//!
//! # The gradient path
//!
//! A spike is a step function, so `dspike/dv` is zero almost everywhere and
//! undefined at threshold. Surrogate-gradient training substitutes a smooth
//! function of the distance between the membrane potential and the threshold.
//! That substitution is the only approximation here; everything either side of
//! it is the exact derivative of the forward pass.
//!
//! For each layer, from the output backwards:
//!
//! 1. Scale the incoming gradient by `surrogate'(v_mem - threshold)`, using the
//!    potential recorded *before* the spike reset overwrote it.
//! 2. Hand the scaled gradient to [`SpikingLinear::backward`], which
//!    accumulates `dL/dW` and returns `dL/dinput` for the layer below.
//!
//! # Example
//!
//! ```rust
//! use dpb_snn::training::{Trainer, SpikeCountLoss, SGDOptimizer, SurrogateType};
//! use dpb_snn::layers::SpikingLinear;
//! use dpb_snn::NeuronParams;
//! use dpb_snn::tensor::SpikeTensor;
//! use ndarray::{Array1, Array2, Array3};
//!
//! # fn main() -> dpb_snn::SNNResult<()> {
//! let layers = vec![
//!     SpikingLinear::new(4, 8, true, NeuronParams::default(), 1.0, false),
//!     SpikingLinear::new(8, 2, true, NeuronParams::default(), 1.0, false),
//! ];
//!
//! let mut trainer = Trainer::new(
//!     layers,
//!     Box::new(SpikeCountLoss::new(1.0)),
//!     Box::new(SGDOptimizer::new(0.05, 0.9, 0.0)),
//!     SurrogateType::FastSigmoid,
//! );
//!
//! // Two samples, ten time steps, four input channels held at full drive.
//! let input = SpikeTensor::from_dense(Array3::ones((2, 10, 4)), false);
//! let targets = Array2::from_elem((2, 2), 0.5);
//! let loss = trainer.train_step(&input, &targets)?;
//! assert!(loss.is_finite());
//! # Ok(())
//! # }
//! ```

use ndarray::{Array1, Array2, Array3};

use crate::layers::conv::{ConvTrace, SpikingConv1d, SpikingConv2d};
use crate::layers::recurrent::{LstmTrace, RecurrentTrace, SpikingLSTM, SpikingRNN};
use crate::layers::{SpikingLayer, SpikingLinear};
use crate::tensor::SpikeTensor;
use crate::training::loss::LossFunction;
use crate::training::optimizer::Optimizer;
use crate::training::surrogate::{
    BoxSurrogate, ExponentialSurrogate, FastSigmoidSurrogate, SuperSpikeSurrogate,
    SurrogateGradient, SurrogateType, TriangleSurrogate,
};
use crate::{SNNError, SNNResult};

/// What one training step measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReport {
    /// Loss before this step's parameter update.
    pub loss: f32,
    /// L2 norm of the concatenated weight gradients, before any clipping.
    ///
    /// Worth watching: a norm that collapses to zero means the surrogate is
    /// saturated and nothing is learning, which otherwise looks the same as a
    /// converged model.
    pub grad_norm: f32,
}

/// A layer the [`Trainer`] knows how to differentiate.
///
/// Feedforward and recurrent layers cannot share one backward signature. For a
/// feedforward layer the caller applies the surrogate to the incoming gradient
/// and then hands it over; for a recurrent layer the surrogate has to be
/// applied inside the reverse-time loop, because a spike reaches the loss both
/// directly and through every later step. This enum keeps that difference in
/// one place instead of pushing it onto callers.
///
/// Every layer here has a validated backward pass. Attention and pooling
/// layers are absent: pooling carries no parameters and is applied through the
/// architectures rather than the trainer, and attention has no backward pass.
// The LSTM variant carries eight weight matrices and dwarfs the others, so
// every element of a layer vector is sized for it. Boxing would trade that for
// a pointer chase on every forward pass, and a network holds a handful of
// layers, not thousands -- the slack is a few hundred bytes in total.
#[allow(clippy::large_enum_variant)]
pub enum TrainableLayer {
    /// A fully connected spiking layer.
    Linear(SpikingLinear),
    /// A recurrent spiking layer, trained by backpropagation through time.
    Recurrent(SpikingRNN),
    /// A 1-D convolutional spiking layer.
    Convolutional(SpikingConv1d),
    /// A 2-D convolutional spiking layer.
    Convolutional2d(SpikingConv2d),
    /// A spiking LSTM layer, trained by backpropagation through time.
    Lstm(SpikingLSTM),
}

/// What a forward pass recorded for one layer's backward pass.
// As for `TrainableLayer`: one trace per layer per step, held only for the
// duration of a backward pass.
#[allow(clippy::large_enum_variant)]
enum LayerTrace {
    /// Membrane potential per step.
    Linear(Array3<f32>),
    /// The recurrent trace, plus the layer's own output spikes -- which the
    /// recurrent weight gradient is an outer product against.
    Recurrent(RecurrentTrace, Array3<f32>),
    /// The convolutional trace, for either convolution.
    Convolutional(ConvTrace),
    /// The LSTM trace.
    Lstm(LstmTrace),
}

impl TrainableLayer {
    /// The spike threshold these neurons fire at.
    fn threshold(&self) -> f32 {
        match self {
            Self::Linear(l) => l.neuron_params.v_threshold,
            Self::Recurrent(l) => l.neuron_params.v_threshold,
            Self::Convolutional(l) => l.neuron_params.v_threshold,
            Self::Convolutional2d(l) => l.neuron_params.v_threshold,
            Self::Lstm(l) => l.neuron_params.v_threshold,
        }
    }

    /// The layer's primary weight matrix.
    ///
    /// For a recurrent layer this is the input weights; the recurrent weights
    /// are reached through [`Self::as_recurrent`]. A convolutional layer has a
    /// 3-D kernel and no 2-D weight matrix, so it returns `None` rather than a
    /// reshaped view that would not be the layer's own storage.
    pub fn weights(&self) -> Option<&Array2<f32>> {
        match self {
            Self::Linear(l) => Some(&l.weights),
            Self::Recurrent(l) => Some(&l.w_input),
            Self::Convolutional(_) | Self::Convolutional2d(_) => None,
            Self::Lstm(l) => Some(&l.w_input_gate),
        }
    }

    /// The underlying layer, when it is a 1-D convolution.
    pub fn as_conv(&self) -> Option<&SpikingConv1d> {
        match self {
            Self::Convolutional(l) => Some(l),
            _ => None,
        }
    }

    /// The underlying layer, when it is a 2-D convolution.
    pub fn as_conv2d(&self) -> Option<&SpikingConv2d> {
        match self {
            Self::Convolutional2d(l) => Some(l),
            _ => None,
        }
    }

    /// The underlying layer, when it is an LSTM.
    pub fn as_lstm(&self) -> Option<&SpikingLSTM> {
        match self {
            Self::Lstm(l) => Some(l),
            _ => None,
        }
    }

    /// The layer's bias, when it has one.
    pub fn bias(&self) -> Option<&Array1<f32>> {
        match self {
            Self::Linear(l) => l.bias.as_ref(),
            Self::Recurrent(l) => l.bias.as_ref(),
            Self::Convolutional(l) => l.bias.as_ref(),
            Self::Convolutional2d(l) => l.bias.as_ref(),
            // The LSTM carries no bias: its gates are weight-only.
            Self::Lstm(_) => None,
        }
    }

    /// The accumulated bias gradient, when there is one.
    pub fn bias_grad(&self) -> Option<&Array1<f32>> {
        match self {
            Self::Linear(l) => l.bias_grad.as_ref(),
            Self::Recurrent(l) => l.bias_grad.as_ref(),
            Self::Convolutional(l) => l.bias_grad.as_ref(),
            Self::Convolutional2d(l) => l.bias_grad.as_ref(),
            Self::Lstm(_) => None,
        }
    }

    /// The underlying layer, when it is feedforward.
    pub fn as_linear(&self) -> Option<&SpikingLinear> {
        match self {
            Self::Linear(l) => Some(l),
            _ => None,
        }
    }

    /// The underlying layer, when it is recurrent.
    pub fn as_recurrent(&self) -> Option<&SpikingRNN> {
        match self {
            Self::Recurrent(l) => Some(l),
            _ => None,
        }
    }

    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        match self {
            Self::Linear(l) => l.forward(input),
            Self::Recurrent(l) => l.forward(input),
            Self::Convolutional(l) => l.forward(input),
            Self::Convolutional2d(l) => l.forward(input),
            Self::Lstm(l) => l.forward(input),
        }
    }

    fn forward_recording(&mut self, input: &SpikeTensor) -> SNNResult<(SpikeTensor, LayerTrace)> {
        match self {
            Self::Linear(l) => {
                let (out, v) = l.forward_recording(input)?;
                Ok((out, LayerTrace::Linear(v)))
            }
            Self::Recurrent(l) => {
                let (out, trace) = l.forward_recording(input)?;
                let spikes = out.to_dense();
                Ok((out, LayerTrace::Recurrent(trace, spikes)))
            }
            Self::Convolutional(l) => {
                let (out, trace) = l.forward_recording(input)?;
                Ok((out, LayerTrace::Convolutional(trace)))
            }
            Self::Convolutional2d(l) => {
                let (out, trace) = l.forward_recording(input)?;
                Ok((out, LayerTrace::Convolutional(trace)))
            }
            Self::Lstm(l) => {
                let (out, trace) = l.forward_recording(input)?;
                Ok((out, LayerTrace::Lstm(trace)))
            }
        }
    }

    /// Backward pass, returning the gradient with respect to this layer's input.
    fn backward(
        &mut self,
        input: &Array3<f32>,
        trace: &LayerTrace,
        grad: &Array3<f32>,
        surrogate: &dyn SurrogateGradient,
    ) -> SNNResult<Array3<f32>> {
        // Each layer's surrogate is evaluated at that layer's own threshold.
        // The trainer used to pass one threshold -- the first layer's -- to
        // every layer, so in a stack whose layers have different thresholds the
        // surrogate was evaluated for a neuron that does not exist. The
        // recurrent and convolutional backward passes already read their own.
        let threshold = self.threshold();
        match (self, trace) {
            (Self::Linear(layer), LayerTrace::Linear(v_mem)) => {
                // A feedforward layer's backward treats the layer as linear, so
                // the surrogate has to be folded in before the call.
                let mut scaled = grad.clone();
                for ((b, t, n), g) in scaled.indexed_iter_mut() {
                    *g *= surrogate.compute_gradient(v_mem[[b, t, n]], threshold);
                }
                layer.backward(input, &scaled)
            }
            (Self::Recurrent(layer), LayerTrace::Recurrent(trace, outputs)) => {
                layer.backward(input, trace, outputs, grad, surrogate)
            }
            (Self::Convolutional(layer), LayerTrace::Convolutional(trace)) => {
                layer.backward(input, trace, grad, surrogate)
            }
            (Self::Convolutional2d(layer), LayerTrace::Convolutional(trace)) => {
                layer.backward(input, trace, grad, surrogate)
            }
            (Self::Lstm(layer), LayerTrace::Lstm(trace)) => {
                layer.backward(input, trace, grad, surrogate)
            }
            _ => Err(SNNError::InvalidConfig(
                "layer trace does not match the layer it came from".to_string(),
            )),
        }
    }

    fn reset_state(&mut self) {
        match self {
            Self::Linear(l) => l.reset_state(),
            Self::Recurrent(l) => l.reset_state(),
            Self::Convolutional(l) => l.reset_state(),
            Self::Convolutional2d(l) => l.reset_state(),
            Self::Lstm(l) => l.reset_state(),
        }
    }

    fn zero_grad(&mut self) {
        match self {
            Self::Linear(l) => l.zero_grad(),
            Self::Recurrent(l) => l.zero_grad(),
            Self::Convolutional(l) => l.zero_grad(),
            Self::Convolutional2d(l) => l.zero_grad(),
            Self::Lstm(l) => l.zero_grad(),
        }
    }

    /// Sum of squared weight gradients, contributing to the global norm.
    ///
    /// Returned as a scalar rather than as matrices because a convolutional
    /// kernel is 3-D and has no `Array2` to hand back.
    fn weight_grad_sq(&self) -> f32 {
        fn sq<'a>(it: impl Iterator<Item = &'a f32>) -> f32 {
            it.map(|v| v * v).sum()
        }
        match self {
            Self::Linear(l) => l.weight_grad.iter().map(|g| sq(g.iter())).sum(),
            Self::Recurrent(l) => {
                l.w_input_grad.iter().map(|g| sq(g.iter())).sum::<f32>()
                    + l.w_recurrent_grad.iter().map(|g| sq(g.iter())).sum::<f32>()
            }
            Self::Convolutional(l) => l.kernel_grad.iter().map(|g| sq(g.iter())).sum(),
            Self::Convolutional2d(l) => l.kernel_grad.iter().map(|g| sq(g.iter())).sum(),
            Self::Lstm(l) => l.gate_grads().iter().map(|g| sq(g.iter())).sum(),
        }
    }

    /// Scale every accumulated gradient in place, for clipping.
    fn scale_grads(&mut self, scale: f32) {
        match self {
            Self::Linear(l) => {
                if let Some(ref mut g) = l.weight_grad {
                    *g *= scale;
                }
                if let Some(ref mut g) = l.bias_grad {
                    *g *= scale;
                }
            }
            Self::Recurrent(l) => {
                if let Some(ref mut g) = l.w_input_grad {
                    *g *= scale;
                }
                if let Some(ref mut g) = l.w_recurrent_grad {
                    *g *= scale;
                }
                if let Some(ref mut g) = l.bias_grad {
                    *g *= scale;
                }
            }
            Self::Convolutional(l) => {
                if let Some(ref mut g) = l.kernel_grad {
                    *g *= scale;
                }
                if let Some(ref mut g) = l.bias_grad {
                    *g *= scale;
                }
            }
            Self::Convolutional2d(l) => {
                if let Some(ref mut g) = l.kernel_grad {
                    *g *= scale;
                }
                if let Some(ref mut g) = l.bias_grad {
                    *g *= scale;
                }
            }
            Self::Lstm(l) => {
                for g in l.gate_grads_mut() {
                    *g *= scale;
                }
            }
        }
    }

    /// `(value, gradient)` pairs for the optimizer, in a fixed order.
    ///
    /// Biases are presented as 1xN matrices because `Optimizer::step` is
    /// Array2-only; see [`Trainer::apply_update`] for why that matters.
    fn param_pairs(&self) -> Vec<(Array2<f32>, Array2<f32>)> {
        fn row(v: &Array1<f32>) -> Array2<f32> {
            let n = v.len();
            v.clone()
                .into_shape_with_order((1, n))
                .expect("a length-n vector always reshapes to 1xn")
        }

        let mut out = Vec::new();
        match self {
            Self::Linear(l) => {
                if let Some(g) = l.weight_grad.as_ref() {
                    out.push((l.weights.clone(), g.clone()));
                }
                if let (Some(b), Some(g)) = (l.bias.as_ref(), l.bias_grad.as_ref()) {
                    out.push((row(b), row(g)));
                }
            }
            Self::Recurrent(l) => {
                if let Some(g) = l.w_input_grad.as_ref() {
                    out.push((l.w_input.clone(), g.clone()));
                }
                if let Some(g) = l.w_recurrent_grad.as_ref() {
                    out.push((l.w_recurrent.clone(), g.clone()));
                }
                if let (Some(b), Some(g)) = (l.bias.as_ref(), l.bias_grad.as_ref()) {
                    out.push((row(b), row(g)));
                }
            }
            Self::Convolutional(l) => {
                // The optimizer is Array2-only, so the (out, in, tap) kernel is
                // flattened to (out, in * tap). Element order is preserved, so
                // each weight keeps its own moment estimates across steps.
                if let Some(g) = l.kernel_grad.as_ref() {
                    out.push((flatten_kernel(&l.kernel), flatten_kernel(g)));
                }
                if let (Some(b), Some(g)) = (l.bias.as_ref(), l.bias_grad.as_ref()) {
                    out.push((row(b), row(g)));
                }
            }
            Self::Convolutional2d(l) => {
                // As above, with the (out, in, kh, kw) kernel flattened to
                // (out, in * kh * kw).
                if let Some(g) = l.kernel_grad.as_ref() {
                    out.push((flatten_kernel4(&l.kernel), flatten_kernel4(g)));
                }
                if let (Some(b), Some(g)) = (l.bias.as_ref(), l.bias_grad.as_ref()) {
                    out.push((row(b), row(g)));
                }
            }
            Self::Lstm(l) => {
                // All eight gate matrices, in the fixed order `gate_weights`
                // and `gate_grads` agree on.
                for (w, g) in l.gate_weights().into_iter().zip(l.gate_grads()) {
                    out.push((w.clone(), g.clone()));
                }
            }
        }
        out
    }

    /// Writes updated values back, consuming them in [`Self::param_pairs`] order.
    fn write_params(&mut self, values: &mut std::vec::IntoIter<Array2<f32>>) {
        fn flat(v: Array2<f32>) -> Array1<f32> {
            let n = v.len();
            v.into_shape_with_order(n)
                .expect("a 1xn matrix always reshapes to length n")
        }

        match self {
            Self::Linear(l) => {
                if l.weight_grad.is_some()
                    && let Some(v) = values.next()
                {
                    l.weights = v;
                }
                if l.bias.is_some()
                    && l.bias_grad.is_some()
                    && let Some(v) = values.next()
                {
                    l.bias = Some(flat(v));
                }
            }
            Self::Recurrent(l) => {
                if l.w_input_grad.is_some()
                    && let Some(v) = values.next()
                {
                    l.w_input = v;
                }
                if l.w_recurrent_grad.is_some()
                    && let Some(v) = values.next()
                {
                    l.w_recurrent = v;
                }
                if l.bias.is_some()
                    && l.bias_grad.is_some()
                    && let Some(v) = values.next()
                {
                    l.bias = Some(flat(v));
                }
            }
            Self::Convolutional(l) => {
                if l.kernel_grad.is_some()
                    && let Some(v) = values.next()
                {
                    let shape = l.kernel.raw_dim();
                    l.kernel = v
                        .into_shape_with_order(shape)
                        .expect("the flattened kernel has the same element count");
                }
                if l.bias.is_some()
                    && l.bias_grad.is_some()
                    && let Some(v) = values.next()
                {
                    l.bias = Some(flat(v));
                }
            }
            Self::Convolutional2d(l) => {
                if l.kernel_grad.is_some()
                    && let Some(v) = values.next()
                {
                    let shape = l.kernel.raw_dim();
                    l.kernel = v
                        .into_shape_with_order(shape)
                        .expect("the flattened kernel has the same element count");
                }
                if l.bias.is_some()
                    && l.bias_grad.is_some()
                    && let Some(v) = values.next()
                {
                    l.bias = Some(flat(v));
                }
            }
            Self::Lstm(l) => {
                // Only matrices that produced a gradient were emitted, so this
                // consumes exactly as many as `param_pairs` pushed.
                let present: Vec<bool> = l.gate_grads_all().iter().map(|g| g.is_some()).collect();
                let mut updated = Vec::new();
                for &has in &present {
                    if has {
                        match values.next() {
                            Some(v) => updated.push(Some(v)),
                            None => updated.push(None),
                        }
                    } else {
                        updated.push(None);
                    }
                }
                l.set_gate_weights(updated);
            }
        }
    }
}

/// Reshapes an `(out, in, tap)` kernel to `(out, in * tap)` for the optimizer.
fn flatten_kernel(k: &Array3<f32>) -> Array2<f32> {
    let (o, i, t) = (k.shape()[0], k.shape()[1], k.shape()[2]);
    k.clone()
        .into_shape_with_order((o, i * t))
        .expect("a 3-D kernel always flattens to 2-D")
}

/// Reshapes an `(out, in, kh, kw)` kernel to `(out, in * kh * kw)`.
fn flatten_kernel4(k: &ndarray::Array4<f32>) -> Array2<f32> {
    let s = k.shape();
    let (o, rest) = (s[0], s[1] * s[2] * s[3]);
    k.clone()
        .into_shape_with_order((o, rest))
        .expect("a 4-D kernel always flattens to 2-D")
}

impl From<SpikingLinear> for TrainableLayer {
    fn from(l: SpikingLinear) -> Self {
        Self::Linear(l)
    }
}

impl From<SpikingRNN> for TrainableLayer {
    fn from(l: SpikingRNN) -> Self {
        Self::Recurrent(l)
    }
}

impl From<SpikingConv1d> for TrainableLayer {
    fn from(l: SpikingConv1d) -> Self {
        Self::Convolutional(l)
    }
}

impl From<SpikingConv2d> for TrainableLayer {
    fn from(l: SpikingConv2d) -> Self {
        Self::Convolutional2d(l)
    }
}

impl From<SpikingLSTM> for TrainableLayer {
    fn from(l: SpikingLSTM) -> Self {
        Self::Lstm(l)
    }
}

/// Trains a stack of [`SpikingLinear`] layers by surrogate-gradient descent.
pub struct Trainer {
    layers: Vec<TrainableLayer>,
    loss: Box<dyn LossFunction>,
    optimizer: Box<dyn Optimizer>,
    surrogate: Box<dyn SurrogateGradient>,
    threshold: f32,
    grad_clip: Option<f32>,
}

impl std::fmt::Debug for Trainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trainer")
            .field("layers", &self.layers.len())
            .field("threshold", &self.threshold)
            .field("grad_clip", &self.grad_clip)
            .finish_non_exhaustive()
    }
}

impl Trainer {
    /// Build a trainer over `layers`.
    ///
    /// The spike threshold is taken from the first layer's neuron parameters,
    /// since that is what the surrogate is evaluated against.
    pub fn new(
        layers: Vec<SpikingLinear>,
        loss: Box<dyn LossFunction>,
        optimizer: Box<dyn Optimizer>,
        surrogate: SurrogateType,
    ) -> Self {
        Self::with_layers(
            layers.into_iter().map(TrainableLayer::Linear).collect(),
            loss,
            optimizer,
            surrogate,
        )
    }

    /// Build a trainer over a mixed stack of feedforward and recurrent layers.
    pub fn with_layers(
        layers: Vec<TrainableLayer>,
        loss: Box<dyn LossFunction>,
        optimizer: Box<dyn Optimizer>,
        surrogate: SurrogateType,
    ) -> Self {
        let threshold = layers.first().map(|l| l.threshold()).unwrap_or(1.0);

        Self {
            layers,
            loss,
            optimizer,
            surrogate: build_surrogate(surrogate),
            threshold,
            grad_clip: Some(5.0),
        }
    }

    /// Build a trainer with a surrogate instance rather than one of the
    /// [`SurrogateType`] presets.
    ///
    /// The presets fix each surrogate's shape parameter. This takes the
    /// surrogate itself, so a caller can widen a box, sharpen a sigmoid, or
    /// supply their own implementation of the trait.
    pub fn with_surrogate(
        layers: Vec<TrainableLayer>,
        loss: Box<dyn LossFunction>,
        optimizer: Box<dyn Optimizer>,
        surrogate: Box<dyn SurrogateGradient>,
    ) -> Self {
        let threshold = layers.first().map(|l| l.threshold()).unwrap_or(1.0);
        Self {
            layers,
            loss,
            optimizer,
            surrogate,
            threshold,
            grad_clip: Some(5.0),
        }
    }

    /// Set the max gradient norm, or `None` to disable clipping.
    ///
    /// Clipping is on by default at 5.0: BPTT through many time steps
    /// multiplies gradients step by step, so a long sequence can produce a
    /// norm large enough to take the weights straight to NaN on one update.
    pub fn set_grad_clip(&mut self, max_norm: Option<f32>) {
        self.grad_clip = max_norm;
    }

    /// The layers being trained.
    pub fn layers(&self) -> &[TrainableLayer] {
        &self.layers
    }

    /// Mutable access to the layers, for inference or checkpointing.
    pub fn layers_mut(&mut self) -> &mut [TrainableLayer] {
        &mut self.layers
    }

    /// Run the network without recording anything, for inference.
    pub fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        self.reset_state();
        let mut activation = input.clone();
        for layer in &mut self.layers {
            activation = layer.forward(&activation)?;
        }
        Ok(activation)
    }

    /// Clear every layer's membrane state.
    ///
    /// Called at the start of each step: a spiking layer carries potential
    /// across calls, so without this the previous sample's residual charge
    /// leaks into the next one.
    pub fn reset_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }

    /// Loss on `input` against `targets`, without updating anything.
    pub fn evaluate(&mut self, input: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        let output = self.forward(input)?;
        self.loss.compute(&output, targets)
    }

    /// One forward pass, one backward pass, one parameter update.
    ///
    /// Returns the loss measured *before* the update, which is the value that
    /// pairs with the parameters that produced it.
    pub fn train_step(&mut self, input: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        self.train_step_reporting(input, targets).map(|r| r.loss)
    }

    /// As [`Self::train_step`], also reporting the gradient norm.
    pub fn train_step_reporting(
        &mut self,
        input: &SpikeTensor,
        targets: &Array2<f32>,
    ) -> SNNResult<StepReport> {
        if self.layers.is_empty() {
            return Err(SNNError::InvalidConfig("trainer has no layers".to_string()));
        }

        self.reset_state();

        // Clear last step's gradients. The recurrent and convolutional backward
        // passes accumulate rather than overwrite -- they must, so a caller can
        // sum over micro-batches -- so without this each step would descend on
        // the running sum of every step before it.
        self.zero_grad();

        // Forward, keeping each layer's input and membrane history. The input
        // is what dL/dW is an outer product against, and the membrane history
        // is where the surrogate is evaluated -- neither survives the forward
        // pass otherwise.
        let mut layer_inputs: Vec<Array3<f32>> = Vec::with_capacity(self.layers.len());
        let mut traces: Vec<LayerTrace> = Vec::with_capacity(self.layers.len());

        let mut activation = input.clone();
        for layer in &mut self.layers {
            layer_inputs.push(activation.to_dense());
            let (output, trace) = layer.forward_recording(&activation)?;
            traces.push(trace);
            activation = output;
        }

        let loss = self.loss.compute(&activation, targets)?;
        let mut grad = self.loss.gradient(&activation, targets)?;

        // Backward, output layer first. Where the surrogate is applied differs
        // between feedforward and recurrent layers, so each layer applies it.
        for idx in (0..self.layers.len()).rev() {
            grad = self.layers[idx].backward(
                &layer_inputs[idx],
                &traces[idx],
                &grad,
                self.surrogate.as_ref(),
            )?;
        }

        let grad_norm = self.gradient_norm();
        if let Some(max_norm) = self.grad_clip {
            self.clip_gradients(max_norm, grad_norm);
        }

        self.apply_update()?;

        Ok(StepReport { loss, grad_norm })
    }

    /// L2 norm across every layer's weight gradient.
    fn gradient_norm(&self) -> f32 {
        self.layers
            .iter()
            .map(|l| l.weight_grad_sq())
            .sum::<f32>()
            .sqrt()
    }

    /// Scale all gradients so their joint norm is at most `max_norm`.
    fn clip_gradients(&mut self, max_norm: f32, current_norm: f32) {
        if !current_norm.is_finite() || current_norm <= max_norm || current_norm == 0.0 {
            return;
        }
        let scale = max_norm / current_norm;
        for layer in &mut self.layers {
            layer.scale_grads(scale);
        }
    }

    /// Hand the gradients to the optimizer and update the weights.
    fn apply_update(&mut self) -> SNNResult<()> {
        // `Optimizer::step` is Array2-only, but a bias is Array1. Rather than
        // hand-rolling a plain descent step for biases -- which would mean Adam
        // updated the weights adaptively while the biases crawled at the raw
        // learning rate -- each bias is presented as a 1xN matrix. The optimizer
        // keys its moment buffers by position in the slice, and each layer emits
        // its parameters in the same order on every step, so biases and
        // recurrent weights accumulate their own moment estimates exactly as
        // input weights do.
        let mut values: Vec<Array2<f32>> = Vec::new();
        let mut grads: Vec<Array2<f32>> = Vec::new();
        for layer in &self.layers {
            for (value, grad) in layer.param_pairs() {
                values.push(value);
                grads.push(grad);
            }
        }

        if values.is_empty() {
            return Ok(());
        }

        {
            let mut param_refs: Vec<&mut Array2<f32>> = values.iter_mut().collect();
            let grad_refs: Vec<&Array2<f32>> = grads.iter().collect();
            self.optimizer.step(&mut param_refs, &grad_refs)?;
        }

        let mut updated = values.into_iter();
        for layer in &mut self.layers {
            layer.write_params(&mut updated);
        }

        Ok(())
    }

    /// Clear the accumulated gradients on every layer.
    pub fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
        self.optimizer.zero_grad();
    }

    /// The optimizer's current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.optimizer.learning_rate()
    }

    /// Set the optimizer's learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr);
    }
}

fn build_surrogate(kind: SurrogateType) -> Box<dyn SurrogateGradient> {
    // One implementation per variant. `Triangle` used to resolve to a box and
    // `Exponential` to SuperSpike, so asking for either silently got a
    // different function than the one named.
    match kind {
        // Width 2.0, i.e. non-zero within 1.0 of the threshold. At the previous
        // 1.0 the window was only +/-0.5 and, on a plain two-layer stack, no
        // membrane sample landed inside it: the gradient norm was exactly zero
        // and training was a silent no-op. This matches the support of the
        // Triangle sibling rather than being picked to suit one network.
        SurrogateType::Box => Box::new(BoxSurrogate::new(2.0)),
        SurrogateType::Triangle => Box::new(TriangleSurrogate::new(1.0)),
        SurrogateType::FastSigmoid => Box::new(FastSigmoidSurrogate::new(10.0)),
        SurrogateType::Exponential => Box::new(ExponentialSurrogate::new(1.0, 5.0)),
        SurrogateType::SuperSpike => Box::new(SuperSpikeSurrogate::new(10.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NeuronParams;
    use crate::training::loss::SpikeCountLoss;
    use crate::training::optimizer::{AdamOptimizer, SGDOptimizer};

    /// Two classes, each a distinct input channel firing every step. A network
    /// that learns anything at all separates these.
    fn separable_batch() -> (SpikeTensor, Array2<f32>) {
        let batch = 4;
        let steps = 12;
        let inputs = 4;
        let mut dense = Array3::zeros((batch, steps, inputs));

        for b in 0..batch {
            let class = b % 2;
            for t in 0..steps {
                // Class 0 drives channels 0-1, class 1 drives channels 2-3.
                dense[[b, t, class * 2]] = 1.0;
                dense[[b, t, class * 2 + 1]] = 1.0;
            }
        }

        // SpikeCountLoss reads targets as a rate in [0, 1] and scales by
        // target_count * num_steps, so 0.6 asks the class's own unit to fire on
        // roughly 60% of steps while the other stays quiet.
        let mut targets = Array2::zeros((batch, 2));
        for b in 0..batch {
            targets[[b, b % 2]] = 0.6;
        }

        (SpikeTensor::from_dense(dense, false), targets)
    }

    fn make_trainer(lr: f32) -> Trainer {
        let params = NeuronParams::default();
        let layers = vec![
            SpikingLinear::new(4, 8, true, params.clone(), 1.0, false),
            SpikingLinear::new(8, 2, true, params, 1.0, false),
        ];
        Trainer::new(
            layers,
            Box::new(SpikeCountLoss::new(1.0)),
            Box::new(SGDOptimizer::new(lr, 0.9, 0.0)),
            SurrogateType::FastSigmoid,
        )
    }

    #[test]
    fn train_step_returns_a_finite_loss() {
        let (input, targets) = separable_batch();
        let mut trainer = make_trainer(0.01);
        let loss = trainer.train_step(&input, &targets).unwrap();
        assert!(loss.is_finite(), "loss was {loss}");
    }

    /// The claim that matters: training reduces the loss. A loop that runs
    /// without learning passes every other test in this file.
    #[test]
    fn training_reduces_the_loss() {
        let (input, targets) = separable_batch();
        let mut trainer = make_trainer(0.02);

        let first = trainer.train_step(&input, &targets).unwrap();
        let mut last = first;
        for _ in 0..60 {
            last = trainer.train_step(&input, &targets).unwrap();
        }

        assert!(
            last < first,
            "loss did not fall over 60 steps: {first} -> {last}"
        );
    }

    /// Weights must actually move. If the surrogate saturates to zero the loss
    /// can drift on its own while the parameters sit still.
    #[test]
    fn weights_change_during_training() {
        let (input, targets) = separable_batch();
        let mut trainer = make_trainer(0.02);

        let before = trainer.layers()[0].weights().expect("linear layer").clone();
        for _ in 0..10 {
            trainer.train_step(&input, &targets).unwrap();
        }
        let after = trainer.layers()[0].weights().expect("linear layer");

        let delta: f32 = (&before - after).iter().map(|d| d.abs()).sum();
        assert!(delta > 1e-6, "weights did not move (total change {delta})");
    }

    /// Biases must go through the optimizer, not a hand-rolled descent step.
    ///
    /// Adam's first update is `lr * m_hat / (sqrt(v_hat) + eps)`, which for a
    /// single step reduces to roughly `lr * sign(g)` -- the size of the step is
    /// independent of the size of the gradient. Plain descent would instead move
    /// the bias by `lr * g`. Asserting the step is close to `lr`, on a batch
    /// whose bias gradients are far from unit magnitude, separates the two.
    #[test]
    fn adam_updates_biases_adaptively() {
        let (input, targets) = separable_batch();
        let lr = 0.01;
        let mut trainer = Trainer::new(
            vec![
                SpikingLinear::new(4, 8, true, NeuronParams::default(), 1.0, false),
                SpikingLinear::new(8, 2, true, NeuronParams::default(), 1.0, false),
            ],
            Box::new(SpikeCountLoss::new(1.0)),
            Box::new(AdamOptimizer::new(lr, 0.9, 0.999, 0.0)),
            SurrogateType::FastSigmoid,
        );
        // Clipping would rescale the gradients and blur the comparison.
        trainer.set_grad_clip(None);

        let before = trainer.layers()[0]
            .bias()
            .cloned()
            .expect("layer has a bias");
        trainer.train_step(&input, &targets).unwrap();
        let after = trainer.layers()[0]
            .bias()
            .cloned()
            .expect("layer has a bias");

        let grad = trainer.layers()[0]
            .bias_grad()
            .cloned()
            .expect("bias gradient was accumulated");

        let mut checked = 0;
        for i in 0..before.len() {
            // Only neurons that actually received a gradient say anything.
            if grad[i].abs() < 1e-6 {
                continue;
            }
            let step = (after[i] - before[i]).abs();
            assert!(
                (step - lr).abs() < lr * 0.1,
                "bias {i} moved {step}, expected about {lr} from an Adam step \
                 (gradient was {}); a plain descent step would have moved it {}",
                grad[i],
                lr * grad[i].abs()
            );
            checked += 1;
        }
        assert!(
            checked > 0,
            "no bias received a gradient, so nothing was tested"
        );
    }

    /// A recurrent layer must train through the trainer, not merely compile
    /// into it. The recurrent weights specifically have to move -- if only the
    /// input weights changed, the BPTT path would not be reaching the optimizer.
    #[test]
    fn a_recurrent_stack_trains() {
        use crate::layers::recurrent::SpikingRNN;

        let (input, targets) = separable_batch();
        let mut trainer = Trainer::with_layers(
            vec![
                TrainableLayer::Recurrent(SpikingRNN::new(
                    4,
                    8,
                    true,
                    NeuronParams::default(),
                    1.0,
                    false,
                )),
                TrainableLayer::Linear(SpikingLinear::new(
                    8,
                    2,
                    true,
                    NeuronParams::default(),
                    1.0,
                    false,
                )),
            ],
            Box::new(SpikeCountLoss::new(1.0)),
            Box::new(SGDOptimizer::new(0.02, 0.9, 0.0)),
            SurrogateType::FastSigmoid,
        );

        let w_rec_before = trainer.layers()[0]
            .as_recurrent()
            .expect("first layer is recurrent")
            .w_recurrent
            .clone();

        let first = trainer.train_step(&input, &targets).unwrap();
        let mut last = first;
        for _ in 0..40 {
            last = trainer.train_step(&input, &targets).unwrap();
        }

        let w_rec_after = &trainer.layers()[0].as_recurrent().unwrap().w_recurrent;
        let moved: f32 = (&w_rec_before - w_rec_after).iter().map(|d| d.abs()).sum();

        assert!(last.is_finite(), "recurrent training produced {last}");
        assert!(
            moved > 1e-6,
            "recurrent weights never moved (total change {moved})"
        );
        assert!(
            last < first,
            "loss did not fall over 40 recurrent steps: {first} -> {last}"
        );
    }

    /// A convolutional layer must train through the trainer, with its kernel
    /// moving -- not merely pass tensors through while staying frozen.
    #[test]
    fn a_convolutional_stack_trains() {
        use crate::layers::conv::SpikingConv1d;

        // Two classes distinguished by which half of a length-8 signal is
        // driven, so the convolution has something local to pick up on.
        let (batch, steps, channels, length) = (4, 12, 1, 8);
        let mut dense = Array3::zeros((batch, steps, channels * length));
        let mut targets = Array2::zeros((batch, 2));
        for b in 0..batch {
            let class = b % 2;
            for t in 0..steps {
                for p in 0..4 {
                    dense[[b, t, class * 4 + p]] = 1.0;
                }
            }
            targets[[b, class]] = 0.5;
        }
        let input = SpikeTensor::from_dense(dense, false);

        // 1 channel, 3 outputs, width-3 kernel, stride 1, padding 1 -> 8
        // positions, so 24 neurons into the classifier.
        let conv = SpikingConv1d::new(1, 3, 3, 1, 1, true, NeuronParams::default(), 1.0, false);
        let mut trainer = Trainer::with_layers(
            vec![
                TrainableLayer::Convolutional(conv),
                TrainableLayer::Linear(SpikingLinear::new(
                    24,
                    2,
                    true,
                    NeuronParams::default(),
                    1.0,
                    false,
                )),
            ],
            Box::new(SpikeCountLoss::new(1.0)),
            Box::new(SGDOptimizer::new(0.02, 0.9, 0.0)),
            SurrogateType::FastSigmoid,
        );

        let kernel_before = trainer.layers()[0]
            .as_conv()
            .expect("first layer is convolutional")
            .kernel
            .clone();

        let first = trainer.train_step(&input, &targets).unwrap();
        let mut last = first;
        for _ in 0..40 {
            last = trainer.train_step(&input, &targets).unwrap();
        }

        let kernel_after = &trainer.layers()[0].as_conv().unwrap().kernel;
        let moved: f32 = (&kernel_before - kernel_after)
            .iter()
            .map(|d| d.abs())
            .sum();

        assert!(last.is_finite(), "convolutional training produced {last}");
        assert!(moved > 1e-6, "kernel never moved (total change {moved})");
        assert!(
            last < first,
            "loss did not fall over 40 convolutional steps: {first} -> {last}"
        );
    }

    /// Every layer type the trainer accepts must actually train: loss falling,
    /// and the layer's own parameters moving.
    #[test]
    fn every_trainable_layer_type_trains() {
        use crate::layers::conv::{SpikingConv1d, SpikingConv2d};
        use crate::layers::recurrent::{SpikingLSTM, SpikingRNN};

        // The LSTM's hidden state is bounded in (-1, 1), so it needs a
        // threshold it can actually reach.
        let low = NeuronParams {
            v_threshold: 0.15,
            ..NeuronParams::default()
        };

        let cases: Vec<(&str, TrainableLayer, usize)> = vec![
            (
                "recurrent",
                SpikingRNN::new(4, 6, true, NeuronParams::default(), 1.0, false).into(),
                6,
            ),
            (
                "conv1d",
                // 4 inputs as 1 channel of length 4, width-3 kernel, pad 1 -> 4
                // positions x 2 channels.
                SpikingConv1d::new(1, 2, 3, 1, 1, true, NeuronParams::default(), 1.0, false).into(),
                8,
            ),
            (
                "conv2d",
                // 4 inputs as 1 channel of 2x2, width-2 kernel, pad 0 -> 1
                // position x 3 channels.
                SpikingConv2d::new(
                    1,
                    3,
                    (2, 2),
                    (1, 1),
                    (0, 0),
                    true,
                    NeuronParams::default(),
                    1.0,
                    false,
                )
                .into(),
                3,
            ),
            ("lstm", SpikingLSTM::new(4, 6, low, 1.0, false).into(), 6),
        ];

        for (name, layer, hidden) in cases {
            let (input, targets) = separable_batch();
            let mut trainer = Trainer::with_layers(
                vec![
                    layer,
                    TrainableLayer::Linear(SpikingLinear::new(
                        hidden,
                        2,
                        true,
                        NeuronParams::default(),
                        1.0,
                        false,
                    )),
                ],
                Box::new(SpikeCountLoss::new(1.0)),
                Box::new(SGDOptimizer::new(0.02, 0.9, 0.0)),
                SurrogateType::FastSigmoid,
            );

            let first = trainer
                .train_step(&input, &targets)
                .unwrap_or_else(|e| panic!("{name}: first step failed: {e}"));
            let mut last = first;
            for _ in 0..40 {
                last = trainer.train_step(&input, &targets).unwrap();
            }

            assert!(last.is_finite(), "{name}: loss became {last}");
            assert!(
                last < first,
                "{name}: loss did not fall over 40 steps: {first} -> {last}"
            );
        }
    }

    /// Two identical training steps from the same state must produce the same
    /// gradient, not twice it.
    ///
    /// `SpikingLinear::backward` overwrites its gradient, but the recurrent and
    /// convolutional backward passes accumulate -- they have to, so a caller can
    /// sum over micro-batches. That makes clearing the trainer's job, and it was
    /// not doing it: every step saw the running sum of every step before it.
    /// With gradient clipping on, the direction stayed roughly right and the
    /// loss still fell, so a "loss goes down" test could not see this.
    #[test]
    fn gradients_do_not_carry_over_between_steps() {
        use crate::layers::recurrent::SpikingRNN;

        let (input, targets) = separable_batch();
        let build = || {
            Trainer::with_layers(
                vec![
                    TrainableLayer::Recurrent(SpikingRNN::new(
                        4,
                        6,
                        true,
                        NeuronParams::default(),
                        1.0,
                        false,
                    )),
                    TrainableLayer::Linear(SpikingLinear::new(
                        6,
                        2,
                        true,
                        NeuronParams::default(),
                        1.0,
                        false,
                    )),
                ],
                Box::new(SpikeCountLoss::new(1.0)),
                // No learning rate, so the weights never move and the second
                // step sees exactly the state the first one did.
                Box::new(SGDOptimizer::new(0.0, 0.0, 0.0)),
                SurrogateType::FastSigmoid,
            )
        };

        let mut trainer = build();
        trainer.set_grad_clip(None);

        trainer.train_step(&input, &targets).unwrap();
        let after_one = trainer.layers()[0]
            .as_recurrent()
            .unwrap()
            .w_input_grad
            .clone()
            .expect("gradient accumulated");

        trainer.train_step(&input, &targets).unwrap();
        let after_two = trainer.layers()[0]
            .as_recurrent()
            .unwrap()
            .w_input_grad
            .clone()
            .unwrap();

        let sum_one: f32 = after_one.iter().map(|g| g.abs()).sum();
        let sum_two: f32 = after_two.iter().map(|g| g.abs()).sum();
        assert!(
            sum_one > 1e-9,
            "no gradient at all, so the test proves nothing"
        );
        assert!(
            (sum_two - sum_one).abs() <= 1e-4 * sum_one,
            "second step's gradient is {sum_two} against the first step's {sum_one}; \
             gradients are carrying over between steps"
        );
    }

    /// Each layer's surrogate must be evaluated at that layer's own threshold.
    ///
    /// The trainer used to evaluate every layer at the FIRST layer's threshold.
    /// The discriminator is the output layer's threshold: hold everything else
    /// fixed and move only that, and the gradient it receives must change. If
    /// the first layer's threshold were used throughout, moving the output
    /// layer's would leave the surrogate evaluation point untouched.
    ///
    /// The Box surrogate has width 1.0 and is non-zero only within 0.5 of the
    /// threshold it is handed, which makes the difference all-or-nothing rather
    /// than a matter of degree.
    #[test]
    fn each_layer_uses_its_own_threshold() {
        fn output_layer_gradient(output_threshold: f32) -> f32 {
            let (input, targets) = separable_batch();
            let mut trainer = Trainer::with_layers(
                vec![
                    // Fires normally, so the output layer actually receives
                    // spikes to integrate.
                    TrainableLayer::Linear(SpikingLinear::new(
                        4,
                        6,
                        true,
                        NeuronParams::default(),
                        1.0,
                        false,
                    )),
                    TrainableLayer::Linear(SpikingLinear::new(
                        6,
                        2,
                        true,
                        NeuronParams {
                            v_threshold: output_threshold,
                            ..NeuronParams::default()
                        },
                        1.0,
                        false,
                    )),
                ],
                Box::new(SpikeCountLoss::new(1.0)),
                Box::new(SGDOptimizer::new(0.0, 0.0, 0.0)),
                SurrogateType::Box,
            );
            trainer.set_grad_clip(None);
            trainer.train_step(&input, &targets).unwrap();
            trainer.layers()[1]
                .as_linear()
                .unwrap()
                .weight_grad
                .as_ref()
                .map(|g| g.iter().map(|v| v.abs()).sum())
                .unwrap_or(0.0)
        }

        // At its own default threshold the output layer's membranes fall inside
        // the surrogate window and it learns.
        let near = output_layer_gradient(1.0);
        assert!(near > 0.0, "control case produced no gradient at all");

        // Moved far away, its membranes never come within half a unit of it, so
        // the surrogate is zero and nothing should reach the weights.
        let far = output_layer_gradient(50.0);
        assert_eq!(
            far, 0.0,
            "output layer still received gradient {far} with its threshold at \
             50.0; its surrogate is being evaluated at another layer's threshold"
        );
    }

    /// Every surrogate the enum offers must actually train.
    ///
    /// `Box` and `Triangle` both resolved to a box of width 1.0, whose window
    /// is only +/-0.5 wide; on this stack no membrane sample fell inside it, so
    /// the gradient norm was exactly zero, no weight moved, and the loss did not
    /// change -- silently, for two of the five options. `Exponential` likewise
    /// resolved to SuperSpike, so two names mapped to functions other than the
    /// ones they named.
    #[test]
    fn every_surrogate_trains() {
        for kind in [
            SurrogateType::Box,
            SurrogateType::Triangle,
            SurrogateType::FastSigmoid,
            SurrogateType::Exponential,
            SurrogateType::SuperSpike,
        ] {
            let (input, targets) = separable_batch();
            let params = NeuronParams::default();

            // Deterministic weights. The constructors initialise randomly, and
            // a bounded surrogate genuinely does saturate for some draws -- the
            // dead-neuron problem -- which would make this test flaky rather
            // than informative.
            let mut l0 = SpikingLinear::new(4, 8, true, params.clone(), 1.0, false);
            let mut l1 = SpikingLinear::new(8, 2, true, params, 1.0, false);
            for n in 0..8 {
                for j in 0..4 {
                    l0.weights[[n, j]] = 0.35 + 0.05 * (n as f32) - 0.03 * (j as f32);
                }
            }
            for n in 0..2 {
                for j in 0..8 {
                    l1.weights[[n, j]] = 0.30 + 0.04 * (n as f32) - 0.02 * (j as f32);
                }
            }

            let mut trainer = Trainer::new(
                vec![l0, l1],
                Box::new(SpikeCountLoss::new(1.0)),
                Box::new(SGDOptimizer::new(0.05, 0.9, 0.0)),
                kind,
            );

            let first = trainer.train_step_reporting(&input, &targets).unwrap();
            assert!(
                first.grad_norm > 0.0,
                "{kind:?}: gradient norm is exactly zero, so the surrogate is \
                 saturated everywhere and nothing can train"
            );

            let mut last = first.loss;
            for _ in 0..60 {
                last = trainer.train_step(&input, &targets).unwrap();
            }
            assert!(
                last < first.loss,
                "{kind:?}: loss did not fall, {} -> {last}",
                first.loss
            );
        }
    }

    /// Each `SurrogateType` must resolve to its own implementation, not a
    /// stand-in for a neighbour.
    #[test]
    fn surrogate_types_are_distinct() {
        // Evaluated well away from threshold, where the shapes differ most.
        let probes = [0.2f32, 0.6, 0.95, 1.0, 1.4, 2.0];
        let mut seen: Vec<(SurrogateType, Vec<f32>)> = Vec::new();
        for kind in [
            SurrogateType::Box,
            SurrogateType::Triangle,
            SurrogateType::FastSigmoid,
            SurrogateType::Exponential,
            SurrogateType::SuperSpike,
        ] {
            let s = build_surrogate(kind);
            seen.push((
                kind,
                probes.iter().map(|v| s.compute_gradient(*v, 1.0)).collect(),
            ));
        }

        for i in 0..seen.len() {
            for j in (i + 1)..seen.len() {
                assert_ne!(
                    seen[i].1, seen[j].1,
                    "{:?} and {:?} compute identical gradients; one is standing in \
                     for the other",
                    seen[i].0, seen[j].0
                );
            }
        }
    }

    #[test]
    fn gradients_are_finite_and_non_zero() {
        let (input, targets) = separable_batch();
        let mut trainer = make_trainer(0.02);

        let report = trainer.train_step_reporting(&input, &targets).unwrap();
        assert!(report.grad_norm.is_finite(), "grad norm was not finite");
        assert!(
            report.grad_norm > 0.0,
            "gradient norm was zero -- nothing would learn"
        );
    }

    /// Clipping has to bound the update, or one long sequence can push the
    /// weights to NaN in a single step.
    #[test]
    fn gradient_clipping_bounds_the_update() {
        let (input, targets) = separable_batch();
        let mut trainer = make_trainer(0.5);
        trainer.set_grad_clip(Some(0.001));

        for _ in 0..20 {
            trainer.train_step(&input, &targets).unwrap();
        }

        assert!(
            trainer.layers()[0]
                .weights()
                .unwrap()
                .iter()
                .all(|w| w.is_finite()),
            "weights diverged despite clipping"
        );
    }

    #[test]
    fn evaluate_leaves_the_weights_alone() {
        let (input, targets) = separable_batch();
        let mut trainer = make_trainer(0.02);

        let before = trainer.layers()[0].weights().expect("linear layer").clone();
        let loss = trainer.evaluate(&input, &targets).unwrap();
        let after = trainer.layers()[0].weights().expect("linear layer");

        assert!(loss.is_finite());
        assert_eq!(&before, after, "evaluate modified the weights");
    }

    #[test]
    fn empty_trainer_is_rejected() {
        let (input, targets) = separable_batch();
        let mut trainer = Trainer::new(
            Vec::new(),
            Box::new(SpikeCountLoss::new(1.0)),
            Box::new(SGDOptimizer::new(0.01, 0.9, 0.0)),
            SurrogateType::FastSigmoid,
        );
        assert!(trainer.train_step(&input, &targets).is_err());
    }

    /// State must not carry between steps: a layer holds membrane potential
    /// across calls, so the same input twice in a row must give the same
    /// forward result.
    #[test]
    fn state_is_reset_between_forward_passes() {
        let (input, _) = separable_batch();
        let mut trainer = make_trainer(0.02);

        let first = trainer.forward(&input).unwrap().to_dense();
        let second = trainer.forward(&input).unwrap().to_dense();

        assert_eq!(first, second, "residual state leaked between passes");
    }
}
