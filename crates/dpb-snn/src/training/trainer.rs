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
//! use ndarray::{Array2, Array3};
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

use ndarray::{Array2, Array3};

use crate::layers::{SpikingLayer, SpikingLinear};
use crate::tensor::SpikeTensor;
use crate::training::loss::LossFunction;
use crate::training::optimizer::Optimizer;
use crate::training::surrogate::{
    BoxSurrogate, FastSigmoidSurrogate, SuperSpikeSurrogate, SurrogateGradient, SurrogateType,
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

/// Trains a stack of [`SpikingLinear`] layers by surrogate-gradient descent.
pub struct Trainer {
    layers: Vec<SpikingLinear>,
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
        let threshold = layers
            .first()
            .map(|l| l.neuron_params.v_threshold)
            .unwrap_or(1.0);

        Self {
            layers,
            loss,
            optimizer,
            surrogate: build_surrogate(surrogate),
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
    pub fn layers(&self) -> &[SpikingLinear] {
        &self.layers
    }

    /// Mutable access to the layers, for inference or checkpointing.
    pub fn layers_mut(&mut self) -> &mut [SpikingLinear] {
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

        // Forward, keeping each layer's input and membrane history. The input
        // is what dL/dW is an outer product against, and the membrane history
        // is where the surrogate is evaluated -- neither survives the forward
        // pass otherwise.
        let mut layer_inputs: Vec<Array3<f32>> = Vec::with_capacity(self.layers.len());
        let mut v_mem_history: Vec<Array3<f32>> = Vec::with_capacity(self.layers.len());

        let mut activation = input.clone();
        for layer in &mut self.layers {
            layer_inputs.push(activation.to_dense());
            let (output, v_mem) = layer.forward_recording(&activation)?;
            v_mem_history.push(v_mem);
            activation = output;
        }

        let loss = self.loss.compute(&activation, targets)?;
        let mut grad = self.loss.gradient(&activation, targets)?;

        // Backward, output layer first.
        for idx in (0..self.layers.len()).rev() {
            let v_mem = &v_mem_history[idx];

            // The surrogate stands in for the spike's derivative.
            for ((b, t, n), g) in grad.indexed_iter_mut() {
                *g *= self
                    .surrogate
                    .compute_gradient(v_mem[[b, t, n]], self.threshold);
            }

            grad = self.layers[idx].backward(&layer_inputs[idx], &grad)?;
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
            .filter_map(|l| l.weight_grad.as_ref())
            .map(|g| g.iter().map(|v| v * v).sum::<f32>())
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
            if let Some(ref mut g) = layer.weight_grad {
                *g *= scale;
            }
            if let Some(ref mut g) = layer.bias_grad {
                *g *= scale;
            }
        }
    }

    /// Hand the gradients to the optimizer and update the weights.
    fn apply_update(&mut self) -> SNNResult<()> {
        // The optimizer borrows parameters mutably and gradients immutably, and
        // both live on the same layer, so the gradients are cloned out first.
        //
        // `Optimizer::step` is Array2-only, but a bias is Array1. Rather than
        // hand-rolling a plain descent step for biases -- which would mean Adam
        // updated the weights adaptively while the biases crawled at the raw
        // learning rate -- each bias is presented as a 1xN matrix. The optimizer
        // keys its moment buffers by position in the slice, and the order here
        // (weights then bias, layer by layer) is the same on every step, so
        // biases accumulate their own moment estimates exactly as weights do.
        struct Slot {
            layer: usize,
            /// True when this slot is the layer's bias rather than its weights.
            is_bias: bool,
            value: Array2<f32>,
            grad: Array2<f32>,
        }

        let mut slots: Vec<Slot> = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(g) = layer.weight_grad.as_ref() {
                slots.push(Slot {
                    layer: i,
                    is_bias: false,
                    value: layer.weights.clone(),
                    grad: g.clone(),
                });
            }
            if let (Some(bias), Some(g)) = (layer.bias.as_ref(), layer.bias_grad.as_ref()) {
                let n = bias.len();
                slots.push(Slot {
                    layer: i,
                    is_bias: true,
                    value: bias.clone().into_shape_with_order((1, n)).map_err(|e| {
                        SNNError::InvalidConfig(format!("bias is not reshapable: {e}"))
                    })?,
                    grad: g.clone().into_shape_with_order((1, n)).map_err(|e| {
                        SNNError::InvalidConfig(format!("bias gradient is not reshapable: {e}"))
                    })?,
                });
            }
        }

        if slots.is_empty() {
            return Ok(());
        }

        // Split the borrows: the optimizer needs `&mut` values and `&` grads at
        // the same time, which it cannot get from one slice of `Slot`.
        let grads: Vec<Array2<f32>> = slots.iter().map(|s| s.grad.clone()).collect();
        let mut values: Vec<Array2<f32>> = slots.iter().map(|s| s.value.clone()).collect();
        {
            let mut param_refs: Vec<&mut Array2<f32>> = values.iter_mut().collect();
            let grad_refs: Vec<&Array2<f32>> = grads.iter().collect();
            self.optimizer.step(&mut param_refs, &grad_refs)?;
        }

        for (slot, updated) in slots.iter().zip(values) {
            let layer = &mut self.layers[slot.layer];
            if slot.is_bias {
                let n = updated.len();
                let flat = updated.into_shape_with_order(n).map_err(|e| {
                    SNNError::InvalidConfig(format!("bias update is not reshapable: {e}"))
                })?;
                layer.bias = Some(flat);
            } else {
                layer.weights = updated;
            }
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
    match kind {
        SurrogateType::Box | SurrogateType::Triangle => Box::new(BoxSurrogate::new(1.0)),
        SurrogateType::FastSigmoid => Box::new(FastSigmoidSurrogate::new(10.0)),
        SurrogateType::Exponential | SurrogateType::SuperSpike => {
            Box::new(SuperSpikeSurrogate::new(10.0))
        }
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

        let before = trainer.layers()[0].weights.clone();
        for _ in 0..10 {
            trainer.train_step(&input, &targets).unwrap();
        }
        let after = &trainer.layers()[0].weights;

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

        let before = trainer.layers()[0].bias.clone().expect("layer has a bias");
        trainer.train_step(&input, &targets).unwrap();
        let after = trainer.layers()[0].bias.clone().expect("layer has a bias");

        let grad = trainer.layers()[0]
            .bias_grad
            .clone()
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
            trainer.layers()[0].weights.iter().all(|w| w.is_finite()),
            "weights diverged despite clipping"
        );
    }

    #[test]
    fn evaluate_leaves_the_weights_alone() {
        let (input, targets) = separable_batch();
        let mut trainer = make_trainer(0.02);

        let before = trainer.layers()[0].weights.clone();
        let loss = trainer.evaluate(&input, &targets).unwrap();
        let after = &trainer.layers()[0].weights;

        assert!(loss.is_finite());
        assert_eq!(before, *after, "evaluate modified the weights");
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
