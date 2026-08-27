//! Python bindings for Spiking Neural Network layers and models

use crate::types::PySpikeEvent;
use dpb_snn::layers::SpikingLayer;
use dpb_snn::{NeuronParams, SpikeTensor, SpikingLinear};

/// Bin a spike train into the dense (batch, step, neuron) tensor the layers take.
///
/// A spike train carries event TIMES; a `SpikeTensor` carries a value per
/// timestep, so the events are quantised onto a `dt` grid.
fn train_to_tensor(train: &PySpikeTrain, dt: f32, num_neurons: usize) -> SpikeTensor {
    let num_steps = ((train.duration / dt as f64).ceil() as usize).max(1);
    let mut tensor = SpikeTensor::zeros(1, num_steps, num_neurons.max(1), false);

    for event in &train.events {
        let step = (event.timestamp / dt as f64) as usize;
        let neuron = event.channel as usize;
        if step < num_steps && neuron < num_neurons {
            // Magnitude carries the event's weight; polarity its sign.
            let value = event.magnitude * event.polarity as f32;
            tensor.set_spike(0, step, neuron, value).ok();
        }
    }
    tensor
}

/// The inverse: read a tensor back out as timed events.
fn tensor_to_train(tensor: &SpikeTensor, dt: f32) -> PySpikeTrain {
    let (_, num_steps, num_neurons) = tensor.shape();
    let dense = tensor.to_dense();

    let mut events = Vec::new();
    for step in 0..num_steps {
        for neuron in 0..num_neurons {
            let value = dense[[0, step, neuron]];
            if value != 0.0 {
                events.push(PySpikeEvent::new(
                    step as f64 * dt as f64,
                    neuron as u32,
                    if value >= 0.0 { 1 } else { -1 },
                    value.abs(),
                ));
            }
        }
    }

    PySpikeTrain::new(
        Some(events),
        num_steps as f64 * dt as f64,
        num_neurons as u32,
    )
}

use crate::types::PySpikeTrain;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::PyClassInitializer;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Base spiking layer
#[pyclass(name = "SpikingLayer", subclass)]
pub struct PySpikingLayer {
    name: String,
    input_size: usize,
    output_size: usize,
}

#[pymethods]
impl PySpikingLayer {
    #[new]
    fn new(name: &str, input_size: usize, output_size: usize) -> Self {
        Self {
            name: name.to_string(),
            input_size,
            output_size,
        }
    }

    /// Forward pass through the layer
    ///
    /// Args:
    ///     input_spikes (SpikeTrain): Input spike train
    ///     dt (float): Time step in seconds
    ///
    /// Returns:
    ///     SpikeTrain: Output spike train
    fn forward(&self, _input_spikes: &PySpikeTrain, _dt: f32) -> PyResult<PySpikeTrain> {
        // Base implementation - override in subclasses
        Ok(PySpikeTrain::new(None, 0.0, self.output_size as u32))
    }

    /// Reset layer state
    fn reset(&mut self) {
        // Base implementation
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn input_size(&self) -> usize {
        self.input_size
    }

    #[getter]
    fn output_size(&self) -> usize {
        self.output_size
    }

    fn __repr__(&self) -> String {
        format!(
            "SpikingLayer(name='{}', input={}, output={})",
            self.name, self.input_size, self.output_size
        )
    }
}

/// Spiking linear (fully connected) layer
///
/// Fully connected spiking layer with configurable neuron model.
///
/// Args:
///     input_size (int): Number of input neurons
///     output_size (int): Number of output neurons
///     neuron (str): Neuron model type ('lif', 'alif', 'izhikevich')
///     weight_init (str): Weight initialization ('uniform', 'normal', 'xavier')
///
/// Example:
///     >>> layer = SpikingLinear(input_size=100, output_size=50, neuron='lif')
///     >>> output = layer.forward(input_spikes, dt=0.001)
#[pyclass(name = "SpikingLinear", extends=PySpikingLayer)]
// Recorded from the Python-side constructor. The wrapper does not consume
// these yet, but dropping them would silently discard what a caller
// passed through the binding.
#[allow(dead_code)]
pub struct PySpikingLinear {
    neuron_type: String,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    /// The library layer this delegates to.
    inner: SpikingLinear,
}

#[pymethods]
impl PySpikingLinear {
    // pyo3's signature attribute names these parameters, so they cannot be
    // underscored; the body ignores the ones it has not wired up yet.
    #[allow(unused_variables)]
    #[new]
    #[pyo3(signature = (input_size, output_size, neuron="lif", weight_init="uniform"))]
    fn new(
        input_size: usize,
        output_size: usize,
        neuron: &str,
        weight_init: &str,
    ) -> PyClassInitializer<Self> {
        // Initialize weights
        let weights = vec![vec![0.1; input_size]; output_size];
        let biases = vec![0.0; output_size];

        PyClassInitializer::from(PySpikingLayer {
            name: "SpikingLinear".to_string(),
            input_size,
            output_size,
        })
        .add_subclass(Self {
            neuron_type: neuron.to_string(),
            weights,
            biases,
            inner: SpikingLinear::new(
                input_size,
                output_size,
                true,
                NeuronParams::default(),
                1.0,
                neuron.eq_ignore_ascii_case("alif"),
            ),
        })
    }

    /// Get weights as numpy array
    fn get_weights(&self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
        Ok(PyArray2::from_vec2(py, &self.weights)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into())
    }

    /// Set weights from numpy array
    fn set_weights(&mut self, weights: PyReadonlyArray2<f32>) -> PyResult<()> {
        let array = weights.as_array();
        let shape = array.shape();

        if shape[0] != self.weights.len() || shape[1] != self.weights[0].len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Weight shape mismatch",
            ));
        }

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                self.weights[i][j] = array[[i, j]];
            }
        }

        Ok(())
    }

    fn forward(&mut self, input_spikes: &PySpikeTrain, dt: f32) -> PyResult<PySpikeTrain> {
        // Delegated to `dpb_snn::SpikingLinear`. The train is binned onto a `dt`
        // grid on the way in and read back out as timed events.
        let input_size = self.weights.first().map_or(0, |row| row.len());
        let tensor = train_to_tensor(input_spikes, dt, input_size);

        let output = self
            .inner
            .forward(&tensor)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("forward failed: {e}")))?;

        Ok(tensor_to_train(&output, dt))
    }

    fn reset(&mut self) {
        self.inner.reset_state();
    }
}

/// Spiking convolutional layer (2D)
///
/// 2D convolutional layer for spiking networks.
///
/// Args:
///     in_channels (int): Number of input channels
///     out_channels (int): Number of output channels
///     kernel_size (int): Size of convolutional kernel
///     stride (int): Stride of convolution
///     padding (int): Padding size
///     neuron (str): Neuron model type
///
/// Example:
///     >>> layer = SpikingConv2d(in_channels=1, out_channels=32, kernel_size=3)
///     >>> output = layer.forward(input_spikes, dt=0.001)
#[pyclass(name = "SpikingConv2d", extends=PySpikingLayer)]
// Recorded from the Python-side constructor. The wrapper does not consume
// these yet, but dropping them would silently discard what a caller
// passed through the binding.
#[allow(dead_code)]
pub struct PySpikingConv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    neuron_type: String,
}

#[pymethods]
impl PySpikingConv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size=3, stride=1, padding=0, neuron="lif"))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        neuron: &str,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PySpikingLayer {
            name: "SpikingConv2d".to_string(),
            input_size: in_channels,
            output_size: out_channels,
        })
        .add_subclass(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            neuron_type: neuron.to_string(),
        })
    }

    #[getter]
    fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    #[getter]
    fn stride(&self) -> usize {
        self.stride
    }

    #[getter]
    fn padding(&self) -> usize {
        self.padding
    }
}

/// Spiking recurrent layer
///
/// Recurrent layer with feedback connections.
///
/// Args:
///     input_size (int): Number of input neurons
///     hidden_size (int): Number of hidden/recurrent neurons
///     neuron (str): Neuron model type
///
/// Example:
///     >>> layer = SpikingRecurrent(input_size=50, hidden_size=100, neuron='lif')
///     >>> output = layer.forward(input_spikes, dt=0.001)
#[pyclass(name = "SpikingRecurrent", extends=PySpikingLayer)]
// Recorded from the Python-side constructor. The wrapper does not consume
// these yet, but dropping them would silently discard what a caller
// passed through the binding.
#[allow(dead_code)]
pub struct PySpikingRecurrent {
    hidden_size: usize,
    neuron_type: String,
}

#[pymethods]
impl PySpikingRecurrent {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, neuron="lif"))]
    fn new(input_size: usize, hidden_size: usize, neuron: &str) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PySpikingLayer {
            name: "SpikingRecurrent".to_string(),
            input_size,
            output_size: hidden_size,
        })
        .add_subclass(Self {
            hidden_size,
            neuron_type: neuron.to_string(),
        })
    }

    #[getter]
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Spiking pooling layer
///
/// Pooling layer for spike trains (max-pooling or average-pooling).
///
/// Args:
///     pool_size (int): Size of pooling window
///     pool_type (str): Type of pooling ('max', 'avg')
///
/// Example:
///     >>> layer = SpikingPooling(pool_size=2, pool_type='max')
///     >>> output = layer.forward(input_spikes, dt=0.001)
#[pyclass(name = "SpikingPooling", extends=PySpikingLayer)]
pub struct PySpikingPooling {
    pool_size: usize,
    pool_type: String,
}

#[pymethods]
impl PySpikingPooling {
    #[new]
    #[pyo3(signature = (pool_size=2, pool_type="max"))]
    fn new(pool_size: usize, pool_type: &str) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PySpikingLayer {
            name: "SpikingPooling".to_string(),
            input_size: 0,
            output_size: 0,
        })
        .add_subclass(Self {
            pool_size,
            pool_type: pool_type.to_string(),
        })
    }

    #[getter]
    fn pool_size(&self) -> usize {
        self.pool_size
    }

    #[getter]
    fn pool_type(&self) -> String {
        self.pool_type.clone()
    }
}

/// Sequential SNN model
///
/// Stack multiple spiking layers sequentially.
///
/// Args:
///     layers (list): List of spiking layers
///
/// Example:
///     >>> model = Sequential([
///     ...     SpikingLinear(100, 64, neuron='lif'),
///     ...     SpikingLinear(64, 10, neuron='lif'),
///     ... ])
///     >>> output = model.forward(input_spikes, dt=0.001)
#[pyclass(name = "Sequential")]
pub struct PySequential {
    layers: Vec<Py<PyAny>>,
}

#[pymethods]
impl PySequential {
    #[new]
    fn new(layers: Vec<Py<PyAny>>) -> Self {
        Self { layers }
    }

    /// Forward pass through all layers
    fn forward(&self, input_spikes: &PySpikeTrain, dt: f32, py: Python) -> PyResult<PySpikeTrain> {
        let mut current = input_spikes.clone();

        for layer in &self.layers {
            // Call forward method on each layer
            current = layer
                .call_method1(py, "forward", (current, dt))?
                .extract(py)?;
        }

        Ok(current)
    }

    /// Reset all layers
    fn reset(&mut self, py: Python) -> PyResult<()> {
        for layer in &self.layers {
            layer.call_method0(py, "reset")?;
        }
        Ok(())
    }

    /// Get number of layers
    fn __len__(&self) -> usize {
        self.layers.len()
    }

    /// Get layer by index
    fn __getitem__(&self, idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        self.layers
            .get(idx)
            .map(|obj| obj.clone_ref(py))
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Index out of range"))
    }

    fn __repr__(&self) -> String {
        format!("Sequential(layers={})", self.layers.len())
    }
}

/// SNN Network builder
///
/// Utility class for building complex SNN architectures.
///
/// Example:
///     >>> builder = SNNBuilder()
///     >>> builder.add_linear(100, 64, neuron='lif')
///     >>> builder.add_linear(64, 10, neuron='lif')
///     >>> model = builder.build()
#[pyclass(name = "SNNBuilder")]
// Recorded from the Python-side constructor. The wrapper does not consume
// these yet, but dropping them would silently discard what a caller
// passed through the binding.
#[allow(dead_code)]
pub struct PySNNBuilder {
    layers: Vec<Py<PyAny>>,
    config: HashMap<String, String>,
}

#[pymethods]
impl PySNNBuilder {
    #[new]
    fn new() -> Self {
        Self {
            layers: Vec::new(),
            config: HashMap::new(),
        }
    }

    /// Add a linear layer
    fn add_linear(
        &mut self,
        py: Python,
        input_size: usize,
        output_size: usize,
        neuron: Option<&str>,
    ) -> PyResult<()> {
        let neuron_type = neuron.unwrap_or("lif");
        let layer = Py::new(
            py,
            PySpikingLinear::new(input_size, output_size, neuron_type, "uniform"),
        )?;
        self.layers.push(layer.into_any());
        Ok(())
    }

    /// Add a convolutional layer
    fn add_conv2d(
        &mut self,
        py: Python,
        in_channels: usize,
        out_channels: usize,
        kernel_size: Option<usize>,
    ) -> PyResult<()> {
        let kernel = kernel_size.unwrap_or(3);
        let layer = Py::new(
            py,
            PySpikingConv2d::new(in_channels, out_channels, kernel, 1, 0, "lif"),
        )?;
        self.layers.push(layer.into_any());
        Ok(())
    }

    /// Add a recurrent layer
    fn add_recurrent(&mut self, py: Python, input_size: usize, hidden_size: usize) -> PyResult<()> {
        let layer = Py::new(py, PySpikingRecurrent::new(input_size, hidden_size, "lif"))?;
        self.layers.push(layer.into_any());
        Ok(())
    }

    /// Add a pooling layer
    fn add_pooling(&mut self, py: Python, pool_size: Option<usize>) -> PyResult<()> {
        let size = pool_size.unwrap_or(2);
        let layer = Py::new(py, PySpikingPooling::new(size, "max"))?;
        self.layers.push(layer.into_any());
        Ok(())
    }

    /// Build the sequential model
    fn build(&self, py: Python) -> PyResult<PySequential> {
        let layers: Vec<Py<PyAny>> = self.layers.iter().map(|obj| obj.clone_ref(py)).collect();
        Ok(PySequential::new(layers))
    }

    /// Get number of layers
    fn __len__(&self) -> usize {
        self.layers.len()
    }
}

/// Register SNN module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpikingLayer>()?;
    m.add_class::<PySpikingLinear>()?;
    m.add_class::<PySpikingConv2d>()?;
    m.add_class::<PySpikingRecurrent>()?;
    m.add_class::<PySpikingPooling>()?;
    m.add_class::<PySequential>()?;
    m.add_class::<PySNNBuilder>()?;
    Ok(())
}
