//! Python bindings for training infrastructure

use crate::snn::{PySequential, PySpikingLinear};
use dpb_snn::layers::SpikingLinear;
use dpb_snn::training::{
    AdamOptimizer, LossFunction, Optimizer, SGDOptimizer, SurrogateType, Trainer as RustTrainer,
};
use dpb_snn::{SpikeCountLoss, SpikeTensor, SpikeTimingLoss, SpikingCrossEntropy};
use numpy::ndarray::{Array2, Array3};
use numpy::{PyReadonlyArray2, PyReadonlyArray3};

/// Lift a (samples x classes) prediction matrix into the single-timestep
/// `SpikeTensor` the loss functions take.
fn predictions_to_tensor(predictions: &PyReadonlyArray2<f32>) -> PyResult<SpikeTensor> {
    let array = predictions.as_array();
    let (rows, cols) = (array.shape()[0], array.shape()[1]);

    let mut tensor = SpikeTensor::zeros(rows, 1, cols, false);
    for i in 0..rows {
        for j in 0..cols {
            tensor.set_spike(i, 0, j, array[[i, j]]).ok();
        }
    }
    Ok(tensor)
}

fn targets_to_array(targets: &PyReadonlyArray2<f32>) -> Array2<f32> {
    targets.as_array().to_owned()
}

use pyo3::PyClassInitializer;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Base loss function
#[pyclass(name = "LossFunction", subclass)]
pub struct PyLossFunction {
    name: String,
}

#[pymethods]
impl PyLossFunction {
    #[new]
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    /// Compute loss
    ///
    /// Args:
    ///     predictions: Model predictions
    ///     targets: Ground truth targets
    ///
    /// Returns:
    ///     float: Loss value
    fn compute(&self, _predictions: Py<PyAny>, _targets: Py<PyAny>) -> PyResult<f32> {
        // Base implementation
        Ok(0.0)
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    fn __repr__(&self) -> String {
        format!("LossFunction(name='{}')", self.name)
    }
}

/// Spike count loss
///
/// Penalizes difference in total spike count.
///
/// Args:
///     weight (float): Scales the computed loss
///     target_rate (float): Firing rate each neuron is penalised for missing,
///         in `[0, 1]`
///
/// Example:
///     >>> loss = SpikeCountLoss(weight=1.0, target_rate=0.5)
///     >>> value = loss.compute(predictions, targets)
#[pyclass(name = "SpikeCountLoss", extends=PyLossFunction)]
pub struct PySpikeCountLoss {
    weight: f32,
    target_rate: f32,
}

#[pymethods]
impl PySpikeCountLoss {
    #[new]
    #[pyo3(signature = (weight=1.0, target_rate=0.5))]
    fn new(weight: f32, target_rate: f32) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyLossFunction {
            name: "SpikeCount".to_string(),
        })
        .add_subclass(Self {
            weight,
            target_rate,
        })
    }

    /// Compute the loss.
    ///
    /// `predictions` and `targets` are (samples x classes) float32 arrays.
    /// The previous signature took untyped objects and returned 0.0 for
    /// every input, which reads as a perfectly fitted model.
    fn compute(
        &self,
        predictions: PyReadonlyArray2<f32>,
        targets: PyReadonlyArray2<f32>,
    ) -> PyResult<f32> {
        let tensor = predictions_to_tensor(&predictions)?;
        let target_array = targets_to_array(&targets);

        // `weight` scales the loss and nothing else. It used to be passed as
        // the Rust loss's `target_count` -- the firing rate being aimed at --
        // and then multiplied in on the way out, so a "scaling factor" moved
        // the target the network was trained toward *and* rescaled the squared
        // error, compounding into roughly weight-cubed. The target rate is now
        // its own parameter.
        SpikeCountLoss::new(self.target_rate)
            .compute(&tensor, &target_array)
            .map(|loss| loss * self.weight)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("spike-count loss: {e}")))
    }
}

/// Spike time loss
///
/// Penalizes differences in spike timing.
///
/// Args:
///     weight (float): Scales the computed loss
///     time_window (float): Recorded for the caller's reference. The Rust
///         timing loss compares first-spike latencies against a target derived
///         from the target array, and takes no matching tolerance, so this does
///         not reach the computation. It used to be passed as that loss's
///         `timing_weight`, which meant widening the documented tolerance
///         multiplied the loss instead of relaxing it.
///
/// Example:
///     >>> loss = SpikeTimeLoss(weight=1.0, time_window=0.01)
///     >>> value = loss.compute(predictions, targets)
#[pyclass(name = "SpikeTimeLoss", extends=PyLossFunction)]
pub struct PySpikeTimeLoss {
    weight: f32,
    time_window: f32,
}

#[pymethods]
impl PySpikeTimeLoss {
    #[new]
    #[pyo3(signature = (weight=1.0, time_window=0.01))]
    fn new(weight: f32, time_window: f32) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyLossFunction {
            name: "SpikeTime".to_string(),
        })
        .add_subclass(Self {
            weight,
            time_window,
        })
    }

    /// Scales the computed loss.
    #[getter]
    fn weight(&self) -> f32 {
        self.weight
    }

    /// The matching tolerance as given.
    ///
    /// Reported back rather than applied: the underlying timing loss compares
    /// first-spike latencies and takes no tolerance parameter. See the class
    /// documentation.
    #[getter]
    fn time_window(&self) -> f32 {
        self.time_window
    }

    /// Compute the loss.
    ///
    /// `predictions` and `targets` are (samples x classes) float32 arrays.
    /// The previous signature took untyped objects and returned 0.0 for
    /// every input, which reads as a perfectly fitted model.
    fn compute(
        &self,
        predictions: PyReadonlyArray2<f32>,
        targets: PyReadonlyArray2<f32>,
    ) -> PyResult<f32> {
        let tensor = predictions_to_tensor(&predictions)?;
        let target_array = targets_to_array(&targets);

        // The Rust loss's parameter is a weight on the timing term, not a
        // matching tolerance. `time_window` used to be passed here, so widening
        // the documented tolerance multiplied the loss rather than relaxing it,
        // and `weight` was then applied on top -- two scalings for one knob.
        // The timing term is left at unit weight and `weight` scales the result,
        // which is what both parameters are documented to do.
        SpikeTimingLoss::new(1.0)
            .compute(&tensor, &target_array)
            .map(|loss| loss * self.weight)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("spike-timing loss: {e}")))
    }
}

/// Cross-entropy loss for spike rates
///
/// Standard cross-entropy on spike rate outputs.
///
/// Example:
///     >>> loss = CrossEntropyLoss()
///     >>> value = loss.compute(predictions, targets)
#[pyclass(name = "CrossEntropyLoss", extends=PyLossFunction)]
pub struct PyCrossEntropyLoss {
    weight: f32,
}

#[pymethods]
impl PyCrossEntropyLoss {
    #[new]
    #[pyo3(signature = (weight=1.0))]
    fn new(weight: f32) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyLossFunction {
            name: "CrossEntropy".to_string(),
        })
        .add_subclass(Self { weight })
    }

    /// Compute the loss.
    ///
    /// `predictions` and `targets` are (samples x classes) float32 arrays.
    /// The previous signature took untyped objects and returned 0.0 for
    /// every input, which reads as a perfectly fitted model.
    fn compute(
        &self,
        predictions: PyReadonlyArray2<f32>,
        targets: PyReadonlyArray2<f32>,
    ) -> PyResult<f32> {
        let tensor = predictions_to_tensor(&predictions)?;
        let target_array = targets_to_array(&targets);

        SpikingCrossEntropy::new()
            .compute(&tensor, &target_array)
            .map(|loss| loss * self.weight)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("cross-entropy loss: {e}"))
            })
    }
}

/// Base optimizer
#[pyclass(name = "Optimizer", subclass)]
pub struct PyOptimizer {
    name: String,
    learning_rate: f32,
}

#[pymethods]
impl PyOptimizer {
    #[new]
    fn new(name: &str, learning_rate: f32) -> Self {
        Self {
            name: name.to_string(),
            learning_rate,
        }
    }

    /// Not usable on its own.
    ///
    /// An optimizer object holds hyperparameters; it holds no parameters and no
    /// gradients, so there is nothing here for it to update. Both of these used
    /// to return Ok(()), so a training loop written around `optimizer.step()`
    /// ran to completion having changed nothing.
    ///
    /// Pass the optimizer to `Trainer`, which owns the parameters and drives the
    /// real update.
    fn step(&mut self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Optimizer.step() does nothing on its own: an optimizer holds no \
             parameters. Pass this optimizer to Trainer(model, optimizer=...) \
             and call Trainer.fit, which performs the update.",
        ))
    }

    /// Not usable on its own; see [`Self::step`].
    fn zero_grad(&mut self) -> PyResult<()> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Optimizer.zero_grad() does nothing on its own: an optimizer holds \
             no gradients. Trainer clears them at each step.",
        ))
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    #[setter]
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn __repr__(&self) -> String {
        format!("Optimizer(name='{}', lr={})", self.name, self.learning_rate)
    }
}

/// Adam optimizer
///
/// Adaptive moment estimation optimizer.
///
/// Args:
///     learning_rate (float): Learning rate
///     beta1 (float): First moment decay rate
///     beta2 (float): Second moment decay rate
///     epsilon (float): Numerical stability constant
///
/// Example:
///     >>> optimizer = Adam(learning_rate=0.001)
///     >>> optimizer.step()
#[pyclass(name = "Adam", extends=PyOptimizer)]
pub struct PyAdam {
    pub(crate) beta1: f32,
    pub(crate) beta2: f32,
    /// Recorded for completeness. The Rust `AdamOptimizer` fixes epsilon at
    /// 1e-8 internally and exposes no way to set it, so this value is reported
    /// back but does not reach the update.
    pub(crate) epsilon: f32,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8))]
    fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyOptimizer {
            name: "Adam".to_string(),
            learning_rate,
        })
        .add_subclass(Self {
            beta1,
            beta2,
            epsilon,
        })
    }

    /// Exponential decay rate for the first moment estimate.
    #[getter]
    fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Exponential decay rate for the second moment estimate.
    #[getter]
    fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Denominator constant.
    ///
    /// Reported back as given, but the Rust optimizer fixes it at 1e-8 and
    /// exposes no way to change it, so setting this does not affect training.
    #[getter]
    fn epsilon(&self) -> f32 {
        self.epsilon
    }
}

/// SGD optimizer
///
/// Stochastic gradient descent with optional momentum.
///
/// Args:
///     learning_rate (float): Learning rate
///     momentum (float): Momentum factor
///     weight_decay (float): L2 regularization
///
/// Example:
///     >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
///     >>> optimizer.step()
#[pyclass(name = "SGD", extends=PyOptimizer)]
pub struct PySGD {
    pub(crate) momentum: f32,
    pub(crate) weight_decay: f32,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (learning_rate=0.01, momentum=0.0, weight_decay=0.0))]
    fn new(learning_rate: f32, momentum: f32, weight_decay: f32) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyOptimizer {
            name: "SGD".to_string(),
            learning_rate,
        })
        .add_subclass(Self {
            momentum,
            weight_decay,
        })
    }

    /// Momentum factor.
    #[getter]
    fn momentum(&self) -> f32 {
        self.momentum
    }

    /// L2 regularization coefficient.
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
}

/// Training callback interface
#[pyclass(name = "Callback")]
// Recorded from the Python-side constructor. The wrapper does not consume
// these yet, but dropping them would silently discard what a caller
// passed through the binding.
#[allow(dead_code)]
pub struct PyCallback {
    name: String,
}

#[pymethods]
impl PyCallback {
    #[new]
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    /// Called at start of training
    fn on_train_begin(&self) -> PyResult<()> {
        Ok(())
    }

    /// Called at end of training
    fn on_train_end(&self) -> PyResult<()> {
        Ok(())
    }

    /// Called at start of epoch
    fn on_epoch_begin(&self, _epoch: usize) -> PyResult<()> {
        Ok(())
    }

    /// Called at end of epoch
    fn on_epoch_end(&self, _epoch: usize, _metrics: &Bound<'_, PyDict>) -> PyResult<()> {
        Ok(())
    }

    /// Called at start of batch
    fn on_batch_begin(&self, _batch: usize) -> PyResult<()> {
        Ok(())
    }

    /// Called at end of batch
    fn on_batch_end(&self, _batch: usize, _loss: f32) -> PyResult<()> {
        Ok(())
    }
}

/// Model trainer
///
/// Main training loop with support for callbacks and metrics.
///
/// Args:
///     model: SNN model to train
///     loss (str or LossFunction): Loss function
///     optimizer (str or Optimizer): Optimizer
///
/// Example:
///     >>> trainer = Trainer(model, loss='spike_count', optimizer='adam')
///     >>> trainer.fit(train_data, epochs=10)
#[pyclass(name = "Trainer")]
pub struct PyTrainer {
    /// The caller's model. Held so trained weights can be written back into
    /// it when `fit` finishes -- otherwise training would update a private
    /// copy and the user's model would be unchanged.
    model: Py<PyAny>,
    loss_name: String,
    optimizer_name: String,
    learning_rate: f32,
    callbacks: Vec<Py<PyAny>>,
    history: HashMap<String, Vec<f32>>,
    /// The Rust trainer doing the actual work.
    inner: RustTrainer,
}

#[pymethods]
impl PyTrainer {
    #[new]
    #[pyo3(signature = (
        model,
        loss="spike_count",
        optimizer=None,
        learning_rate=0.001,
        surrogate="fast_sigmoid",
        target_rate=0.5,
    ))]
    fn new(
        model: Py<PyAny>,
        loss: &str,
        optimizer: Option<Py<PyAny>>,
        learning_rate: f32,
        surrogate: &str,
        target_rate: f32,
        py: Python,
    ) -> PyResult<Self> {
        let layers = collect_layers(model.bind(py))?;

        // `optimizer` accepts a name or an `Adam`/`SGD` instance; an instance
        // brings its own learning rate and hyperparameters.
        let (built, lr, name) = match optimizer {
            Some(ref obj) => optimizer_from_arg(obj.bind(py), learning_rate)?,
            None => (
                build_optimizer("adam", learning_rate)?,
                learning_rate,
                "adam".to_string(),
            ),
        };

        let inner = RustTrainer::new(
            layers,
            build_loss(loss, target_rate)?,
            built,
            parse_surrogate(surrogate)?,
        );
        Ok(Self {
            model,
            loss_name: loss.to_string(),
            optimizer_name: name,
            learning_rate: lr,
            callbacks: Vec::new(),
            history: HashMap::new(),
            inner,
        })
    }

    /// Add a training callback
    fn add_callback(&mut self, callback: Py<PyAny>) {
        self.callbacks.push(callback);
    }

    /// Fit the model on training data.
    ///
    /// Data-loader contract: ``train_data`` is any iterable yielding
    /// ``(inputs, targets)`` pairs, where
    ///
    /// * ``inputs`` has shape ``(batch, time_steps, input_size)`` -- spike
    ///   trains, or any real-valued drive
    /// * ``targets`` has shape ``(batch, output_size)`` -- the desired firing
    ///   rate of each output neuron, in ``[0, 1]``
    ///
    /// Both are converted with ``numpy.asarray(..., dtype='float32')``, so
    /// lists, numpy arrays and anything array-like all work.
    ///
    /// The whole iterable is materialised before the first epoch. That makes
    /// one-shot generators behave correctly across ``epochs > 1``, at the cost
    /// of holding the dataset in memory.
    ///
    /// Args:
    ///     train_data: Iterable of ``(inputs, targets)`` batches
    ///     epochs (int): Number of passes over the data
    ///     validation_data: Optional iterable in the same format
    ///     verbose (bool): Print per-epoch loss
    ///
    /// Returns:
    ///     dict: Training history -- ``loss`` always, ``val_loss`` when
    ///     validation data is supplied
    #[pyo3(signature = (train_data, epochs, validation_data=None, verbose=false))]
    fn fit(
        &mut self,
        train_data: Py<PyAny>,
        epochs: usize,
        validation_data: Option<Py<PyAny>>,
        verbose: bool,
        py: Python,
    ) -> PyResult<HashMap<String, Vec<f32>>> {
        let train = materialize(py, train_data.bind(py))?;
        if train.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "train_data yielded no batches",
            ));
        }
        let validation = match validation_data {
            Some(ref data) => Some(materialize(py, data.bind(py))?),
            None => None,
        };

        for callback in &self.callbacks {
            callback.call_method0(py, "on_train_begin")?;
        }

        for epoch in 0..epochs {
            let mut total = 0.0;
            for (inputs, targets) in &train {
                total += self
                    .inner
                    .train_step(inputs, targets)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            let loss = total / train.len() as f32;
            self.history
                .entry("loss".to_string())
                .or_default()
                .push(loss);

            let val_loss = match validation {
                Some(ref batches) => {
                    let mut total = 0.0;
                    for (inputs, targets) in batches {
                        total += self.inner.evaluate(inputs, targets).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;
                    }
                    let val = total / batches.len() as f32;
                    self.history
                        .entry("val_loss".to_string())
                        .or_default()
                        .push(val);
                    Some(val)
                }
                None => None,
            };

            if verbose {
                match val_loss {
                    Some(val) => println!(
                        "epoch {}/{} - loss {:.4} - val_loss {:.4}",
                        epoch + 1,
                        epochs,
                        loss,
                        val
                    ),
                    None => println!("epoch {}/{} - loss {:.4}", epoch + 1, epochs, loss),
                }
            }

            for callback in &self.callbacks {
                let metrics = PyDict::new(py);
                metrics.set_item("loss", loss)?;
                if let Some(val) = val_loss {
                    metrics.set_item("val_loss", val)?;
                }
                callback.call_method1(py, "on_epoch_end", (epoch, metrics))?;
            }
        }

        for callback in &self.callbacks {
            callback.call_method0(py, "on_train_end")?;
        }

        // Push the trained weights back into the caller's model, so `model`
        // reflects the training that just happened.
        self.write_back(py)?;

        Ok(self.history.clone())
    }

    /// Evaluate the model without updating any weights.
    ///
    /// Args:
    ///     data: Iterable of ``(inputs, targets)`` batches, as for `fit`
    ///
    /// Returns:
    ///     dict: ``{"loss": mean loss over the batches}``
    fn evaluate(&mut self, data: Py<PyAny>, py: Python) -> PyResult<HashMap<String, f32>> {
        let batches = materialize(py, data.bind(py))?;
        if batches.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data yielded no batches",
            ));
        }
        let mut total = 0.0;
        for (inputs, targets) in &batches {
            total += self
                .inner
                .evaluate(inputs, targets)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(HashMap::from([(
            "loss".to_string(),
            total / batches.len() as f32,
        )]))
    }

    /// Name of the loss function this trainer was built with.
    #[getter]
    fn loss(&self) -> String {
        self.loss_name.clone()
    }

    /// Name of the optimizer this trainer was built with.
    #[getter]
    fn optimizer(&self) -> String {
        self.optimizer_name.clone()
    }

    /// Current optimizer learning rate.
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }

    /// Set the optimizer learning rate, e.g. from a scheduler.
    #[setter]
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
        self.inner.set_learning_rate(lr);
    }

    /// Reset the membrane state of every layer.
    fn reset_state(&mut self) {
        self.inner.reset_state();
    }

    /// Get training history
    fn get_history(&self) -> HashMap<String, Vec<f32>> {
        self.history.clone()
    }

    /// Reset training history
    fn reset_history(&mut self) {
        self.history.clear();
    }
}

/// Learning rate scheduler
///
/// Adjusts learning rate during training.
///
/// Args:
///     optimizer: Optimizer to adjust
///     schedule_type (str): Type of schedule ('step', 'exponential', 'cosine')
///     **kwargs: Schedule-specific parameters
///
/// Example:
///     >>> scheduler = LRScheduler(optimizer, schedule_type='step', step_size=10, gamma=0.1)
///     >>> scheduler.step()
#[pyclass(name = "LRScheduler")]
pub struct PyLRScheduler {
    /// The optimizer whose rate this schedules.
    ///
    /// It used to be accepted and dropped, so the scheduler could neither read
    /// the starting rate nor push an updated one back -- `step` incremented a
    /// counter and nothing else, and `get_lr` returned a fixed 0.001.
    optimizer: Py<PyAny>,
    initial_lr: f32,
    schedule_type: String,
    step_size: usize,
    gamma: f32,
    current_step: usize,
}

#[pymethods]
impl PyLRScheduler {
    #[new]
    #[pyo3(signature = (optimizer, schedule_type="step", step_size=10, gamma=0.1))]
    fn new(
        optimizer: Py<PyAny>,
        schedule_type: &str,
        step_size: usize,
        gamma: f32,
        py: Python,
    ) -> PyResult<Self> {
        // `learning_rate` is a pyo3 property, so it is read and written as an
        // attribute rather than called.
        let initial_lr: f32 = optimizer
            .getattr(py, "learning_rate")
            .and_then(|lr| lr.extract(py))
            .unwrap_or(0.001);

        Ok(Self {
            optimizer,
            initial_lr,
            schedule_type: schedule_type.to_string(),
            step_size,
            gamma,
            current_step: 0,
        })
    }

    /// The rate this schedule prescribes at the current step.
    fn scheduled_lr(&self) -> f32 {
        match self.schedule_type.as_str() {
            // Multiply by gamma every `step_size` steps.
            "step" => {
                let decays = self
                    .current_step
                    .checked_div(self.step_size)
                    .unwrap_or_default();
                self.initial_lr * self.gamma.powi(decays as i32)
            }
            // Multiply by gamma every step.
            "exponential" => self.initial_lr * self.gamma.powi(self.current_step as i32),
            // Anything else holds the initial rate.
            _ => self.initial_lr,
        }
    }

    /// Step the scheduler
    fn step(&mut self, py: Python) -> PyResult<()> {
        self.current_step += 1;
        // Push the new rate back to the optimizer, which is the whole point of
        // holding a reference to it.
        let lr = self.scheduled_lr();
        self.optimizer.setattr(py, "learning_rate", lr)?;
        Ok(())
    }

    /// Get current learning rate
    ///
    /// Reports the scheduler's actual rate rather than a fixed 0.001, which
    /// made every schedule look like it was doing nothing.
    fn get_lr(&self) -> f32 {
        self.scheduled_lr()
    }
}

/// Register training module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLossFunction>()?;
    m.add_class::<PySpikeCountLoss>()?;
    m.add_class::<PySpikeTimeLoss>()?;
    m.add_class::<PyCrossEntropyLoss>()?;
    m.add_class::<PyOptimizer>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyCallback>()?;
    m.add_class::<PyTrainer>()?;
    m.add_class::<PyLRScheduler>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Trainer support
// ---------------------------------------------------------------------------

/// Pulls the Rust layers out of a Python model.
///
/// Accepts a `Sequential`, a plain sequence of layers, or a single
/// `SpikingLinear`. Only `SpikingLinear` can be trained: the convolutional and
/// recurrent layers have no backward pass, so admitting them here would build a
/// trainer that silently skipped them.
fn collect_layers(model: &Bound<'_, PyAny>) -> PyResult<Vec<SpikingLinear>> {
    let py = model.py();

    let candidates: Vec<Py<PyAny>> = if let Ok(seq) = model.extract::<PyRef<'_, PySequential>>() {
        seq.layers.iter().map(|l| l.clone_ref(py)).collect()
    } else if let Ok(layer) = model.extract::<PyRef<'_, PySpikingLinear>>() {
        return Ok(vec![layer.inner.clone()]);
    } else {
        model.extract::<Vec<Py<PyAny>>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "model must be a Sequential, a list of layers, or a SpikingLinear",
            )
        })?
    };

    let mut layers = Vec::with_capacity(candidates.len());
    for (i, obj) in candidates.iter().enumerate() {
        let layer = obj
            .bind(py)
            .extract::<PyRef<'_, PySpikingLinear>>()
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "layer {i} is not a SpikingLinear; only SpikingLinear layers can be trained \
                     (conv and recurrent layers have no backward pass)"
                ))
            })?;
        layers.push(layer.inner.clone());
    }
    Ok(layers)
}

fn build_loss(name: &str, target_rate: f32) -> PyResult<Box<dyn LossFunction>> {
    match name {
        "spike_count" | "count" => Ok(Box::new(SpikeCountLoss::new(target_rate))),
        "cross_entropy" | "ce" => Ok(Box::new(SpikingCrossEntropy::new())),
        "spike_timing" | "timing" => Ok(Box::new(SpikeTimingLoss::new(1.0))),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown loss '{other}'; expected one of: spike_count, cross_entropy, spike_timing"
        ))),
    }
}

fn build_optimizer(name: &str, learning_rate: f32) -> PyResult<Box<dyn Optimizer>> {
    match name {
        "adam" => Ok(Box::new(AdamOptimizer::new(learning_rate, 0.9, 0.999, 0.0))),
        "sgd" => Ok(Box::new(SGDOptimizer::new(learning_rate, 0.0, 0.0))),
        "momentum" => Ok(Box::new(SGDOptimizer::new(learning_rate, 0.9, 0.0))),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown optimizer '{other}'; expected one of: adam, sgd, momentum"
        ))),
    }
}

/// Builds the Rust optimizer from whatever the caller passed for `optimizer`.
///
/// Accepts a name, or an `Adam`/`SGD` instance. The instances used to be inert:
/// their hyperparameters were recorded at construction and then never read,
/// because `Trainer` took only a name and built its own optimizer from
/// defaults. Passing `Adam(learning_rate=0.01, beta1=0.5)` therefore trained at
/// whatever `learning_rate` was passed separately, with beta1 = 0.9.
///
/// Returns the optimizer along with the learning rate it was built with, since
/// an instance carries its own and it overrides the `learning_rate` argument.
fn optimizer_from_arg(
    arg: &Bound<'_, PyAny>,
    fallback_lr: f32,
) -> PyResult<(Box<dyn Optimizer>, f32, String)> {
    if let Ok(name) = arg.extract::<String>() {
        return Ok((build_optimizer(&name, fallback_lr)?, fallback_lr, name));
    }

    // An instance carries its own learning rate on the base class.
    let base = arg.extract::<PyRef<'_, PyOptimizer>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "optimizer must be a name ('adam', 'sgd', 'momentum') or an Adam or \
             SGD instance",
        )
    })?;
    let lr = base.learning_rate;
    let name = base.name.clone();
    drop(base);

    if let Ok(adam) = arg.extract::<PyRef<'_, PyAdam>>() {
        return Ok((
            Box::new(AdamOptimizer::new(lr, adam.beta1, adam.beta2, 0.0)),
            lr,
            name,
        ));
    }
    if let Ok(sgd) = arg.extract::<PyRef<'_, PySGD>>() {
        return Ok((
            Box::new(SGDOptimizer::new(lr, sgd.momentum, sgd.weight_decay)),
            lr,
            name,
        ));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "optimizer '{name}' is not a supported instance; use Adam or SGD, or \
         pass a name"
    )))
}

fn parse_surrogate(name: &str) -> PyResult<SurrogateType> {
    match name {
        "fast_sigmoid" => Ok(SurrogateType::FastSigmoid),
        "box" | "rectangular" => Ok(SurrogateType::Box),
        "triangle" | "triangular" => Ok(SurrogateType::Triangle),
        "exponential" => Ok(SurrogateType::Exponential),
        "superspike" | "super_spike" => Ok(SurrogateType::SuperSpike),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown surrogate '{other}'; expected one of: fast_sigmoid, box, triangle, \
             exponential, superspike"
        ))),
    }
}

/// Converts an array-like to a contiguous f32 array via `numpy.asarray`.
///
/// Going through numpy is what lets callers pass lists, nested lists, float64
/// arrays or torch tensors without converting first.
fn as_f32_array<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float32")?;
    py.import("numpy")?
        .call_method("asarray", (obj,), Some(&kwargs))
}

/// Reads the whole data loader into memory as `(inputs, targets)` pairs.
///
/// Materialising up front is deliberate: a Python generator is exhausted by its
/// first pass, so streaming it would make every epoch after the first train on
/// nothing.
fn materialize(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
) -> PyResult<Vec<(SpikeTensor, Array2<f32>)>> {
    let mut batches = Vec::new();
    for (i, item) in data.try_iter()?.enumerate() {
        let item = item?;
        let pair = item.extract::<Vec<Bound<'_, PyAny>>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "batch {i} is not a (inputs, targets) pair"
            ))
        })?;
        if pair.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch {i} has {} elements; expected exactly 2: (inputs, targets)",
                pair.len()
            )));
        }

        let inputs: Array3<f32> = as_f32_array(py, &pair[0])?
            .extract::<PyReadonlyArray3<f32>>()
            .map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "batch {i}: inputs must be 3-D (batch, time_steps, input_size)"
                ))
            })?
            .as_array()
            .to_owned();

        let targets: Array2<f32> = as_f32_array(py, &pair[1])?
            .extract::<PyReadonlyArray2<f32>>()
            .map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "batch {i}: targets must be 2-D (batch, output_size)"
                ))
            })?
            .as_array()
            .to_owned();

        if inputs.shape()[0] != targets.shape()[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "batch {i}: inputs have batch size {} but targets have {}",
                inputs.shape()[0],
                targets.shape()[0]
            )));
        }

        batches.push((SpikeTensor::from_dense(inputs, false), targets));
    }
    Ok(batches)
}

impl PyTrainer {
    /// Copies the trained weights back into the caller's Python layers.
    ///
    /// Without this the model handed to the constructor would still hold its
    /// initial weights after `fit` returned, since the trainer works on clones.
    fn write_back(&self, py: Python<'_>) -> PyResult<()> {
        let model = self.model.bind(py);
        let objects: Vec<Py<PyAny>> = if let Ok(seq) = model.extract::<PyRef<'_, PySequential>>() {
            seq.layers.iter().map(|l| l.clone_ref(py)).collect()
        } else if model.extract::<PyRef<'_, PySpikingLinear>>().is_ok() {
            vec![self.model.clone_ref(py)]
        } else {
            model.extract::<Vec<Py<PyAny>>>()?
        };

        for (obj, trained) in objects.iter().zip(self.inner.layers()) {
            // Only feedforward layers can come back through the Python model:
            // `collect_layers` refuses anything else, so a non-linear layer here
            // would mean the trainer and the model had drifted apart.
            let Some(trained) = trained.as_linear() else {
                continue;
            };
            let mut layer = obj.bind(py).extract::<PyRefMut<'_, PySpikingLinear>>()?;
            // One assignment: the layer keeps no separate copy to refresh.
            layer.inner = trained.clone();
        }
        Ok(())
    }
}
