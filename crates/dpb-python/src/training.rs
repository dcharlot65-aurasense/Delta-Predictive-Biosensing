//! Python bindings for training infrastructure

use dpb_snn::training::LossFunction as _;
use dpb_snn::{SpikeCountLoss, SpikeTensor, SpikeTimingLoss, SpikingCrossEntropy};
use numpy::ndarray::Array2;
use numpy::{PyReadonlyArray2, PyArrayMethods as _};

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
///     weight (float): Loss weight/scaling factor
///
/// Example:
///     >>> loss = SpikeCountLoss(weight=1.0)
///     >>> value = loss.compute(predictions, targets)
#[pyclass(name = "SpikeCountLoss", extends=PyLossFunction)]
pub struct PySpikeCountLoss {
    weight: f32,
}

#[pymethods]
impl PySpikeCountLoss {
    #[new]
    #[pyo3(signature = (weight=1.0))]
    fn new(weight: f32) -> (Self, PyLossFunction) {
        (
            Self { weight },
            PyLossFunction {
                name: "SpikeCount".to_string(),
            },
        )
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

        SpikeCountLoss::new(self.weight)
            .compute(&tensor, &target_array)
            .map(|loss| loss * self.weight)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("spike-count loss: {e}"))
            })
    }
}

/// Spike time loss
///
/// Penalizes differences in spike timing.
///
/// Args:
///     weight (float): Loss weight
///     time_window (float): Time window for matching spikes
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
    fn new(weight: f32, time_window: f32) -> (Self, PyLossFunction) {
        (
            Self {
                weight,
                time_window,
            },
            PyLossFunction {
                name: "SpikeTime".to_string(),
            },
        )
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

        SpikeTimingLoss::new(self.time_window)
            .compute(&tensor, &target_array)
            .map(|loss| loss * self.weight)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("spike-timing loss: {e}"))
            })
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
    fn new(weight: f32) -> (Self, PyLossFunction) {
        (
            Self { weight },
            PyLossFunction {
                name: "CrossEntropy".to_string(),
            },
        )
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

    /// Perform optimization step
    fn step(&mut self) -> PyResult<()> {
        // Base implementation
        Ok(())
    }

    /// Zero gradients
    fn zero_grad(&mut self) -> PyResult<()> {
        // Base implementation
        Ok(())
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
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8))]
    fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> (Self, PyOptimizer) {
        (
            Self {
                beta1,
                beta2,
                epsilon,
            },
            PyOptimizer {
                name: "Adam".to_string(),
                learning_rate,
            },
        )
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
    momentum: f32,
    weight_decay: f32,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (learning_rate=0.01, momentum=0.0, weight_decay=0.0))]
    fn new(learning_rate: f32, momentum: f32, weight_decay: f32) -> (Self, PyOptimizer) {
        (
            Self {
                momentum,
                weight_decay,
            },
            PyOptimizer {
                name: "SGD".to_string(),
                learning_rate,
            },
        )
    }
}

/// Training callback interface
#[pyclass(name = "Callback")]
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
    loss_name: String,
    optimizer_name: String,
    learning_rate: f32,
    callbacks: Vec<Py<PyAny>>,
    history: HashMap<String, Vec<f32>>,
}

#[pymethods]
impl PyTrainer {
    #[new]
    #[pyo3(signature = (model, loss="spike_count", optimizer="adam", learning_rate=0.001))]
    fn new(
        model: Py<PyAny>,
        loss: &str,
        optimizer: &str,
        learning_rate: f32,
    ) -> Self {
        Self {
            loss_name: loss.to_string(),
            optimizer_name: optimizer.to_string(),
            learning_rate,
            callbacks: Vec::new(),
            history: HashMap::new(),
        }
    }

    /// Add a training callback
    fn add_callback(&mut self, callback: Py<PyAny>) {
        self.callbacks.push(callback);
    }

    /// Fit the model on training data
    ///
    /// Args:
    ///     train_data: Training data loader
    ///     epochs (int): Number of epochs
    ///     validation_data: Optional validation data
    ///     verbose (bool): Print progress
    ///
    /// Returns:
    ///     dict: Training history
    #[pyo3(signature = (train_data, epochs, validation_data=None, verbose=false))]
    fn fit(
        &mut self,
        train_data: Py<PyAny>,
        epochs: usize,
        validation_data: Option<Py<PyAny>>,
        verbose: bool,
        py: Python,
    ) -> PyResult<HashMap<String, Vec<f32>>> {
        // Call on_train_begin callbacks
        for callback in &self.callbacks {
            callback.call_method0(py, "on_train_begin")?;
        }

        // Not implemented: a training loop needs a data-loader contract --
        // how to iterate `data`, what a batch looks like, how targets pair with
        // inputs -- and the Python API defines none, taking `data` as an opaque
        // object.
        //
        // This used to run the epoch loop recording a loss of 0.0 each time and
        // printing "loss = 0.0000", which reads as a perfectly converged model.
        // Refusing is the honest answer until the contract exists.
        let _ = (epochs, verbose, py);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Trainer.fit is not implemented: no data-loader contract is defined yet. \
             Use the loss functions and optimizers directly, or drive training from Rust.",
        ))
    }

    /// Evaluate model on data
    ///
    /// Args:
    ///     data: Evaluation data loader
    ///
    /// Returns:
    ///     dict: Evaluation metrics
    fn evaluate(&self, data: Py<PyAny>, py: Python) -> PyResult<HashMap<String, f32>> {
        // Same gap as `fit`: reporting a loss of 0.0 for any input is worse
        // than refusing.
        let _ = (data, py);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Trainer.evaluate is not implemented: no data-loader contract is defined yet",
        ))
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
                let decays = if self.step_size == 0 {
                    0
                } else {
                    self.current_step / self.step_size
                };
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
