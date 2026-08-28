//! Python bindings for the Delta-Predictive Biosensing Framework
//!
//! # Status
//!
//! **Wired to the Rust crates**, and behaving exactly as they do:
//!
//! - `encoders` -- all five call [`dpb_encoders`].
//! - `neurons.LifNeuron` -- [`dpb_neurons::LifNeuron`]; biophysical units in
//!   consequence.
//! - `snn.SpikingLinear` -- [`dpb_snn::SpikingLinear`], with spike trains binned
//!   onto the timestep grid and read back as timed events.
//! - `synth` ECG, PPG and EMG generators, including the ECG's event-level
//!   ground truth.
//! - `metrics` ROC and AUC -- [`dpb_core::validation::roc`].
//! - `training` spike-count, spike-timing and cross-entropy losses.
//! - `gpu` device enumeration, availability and shader validation, via wgpu and
//!   [`dpb_core::gpu`].
//!
//! **Implemented here** rather than delegated, because the Rust crates have no
//! equivalent: the van Rossum spike distance, and the learning-rate schedules.
//!
//! **Training** runs through `Trainer`, which drives the Rust
//! surrogate-gradient trainer over a stack of `SpikingLinear` layers. `fit`
//! takes any iterable of `(inputs, targets)` pairs -- `inputs` shaped
//! `(batch, time_steps, input_size)`, `targets` shaped `(batch, output_size)`
//! as per-neuron firing rates in `[0, 1]`. Trained weights are written back
//! into the model that was passed in.
//!
//! **Not implemented, and refusing rather than pretending:** GPU buffer
//! allocation, transfer and dispatch (the binding owns no device or queue).
//! These raise `NotImplementedError` instead of reporting a successful
//! allocation.
//!
//! The `gpu` profiler measures WALL-CLOCK time, not GPU timestamps, and says so
//! at the call site.

use pyo3::prelude::*;

// Module declarations
mod encoders;
mod gpu;
mod metrics;
mod neurons;
mod numpy_utils;
mod snn;
mod synth;
mod training;
mod types;

/// Delta-Predictive Biosensing Framework
///
/// A neuromorphic signal processing framework for biosensor data.
///
/// Modules:
///     encoders: Event-based signal encoders
///     neurons: Neuron model implementations
///     snn: Spiking Neural Network layers and models
///     training: Training infrastructure
///     synth: Synthetic data generators
///     metrics: Evaluation metrics
///     gpu: GPU device management
///
/// Example:
///     >>> import dpb
///     >>> encoder = dpb.encoders.LevelCrossingEncoder(threshold=0.5)
///     >>> signal = dpb.TimeSeries(data=numpy_array, sample_rate=250.0)
///     >>> events = encoder.encode(signal)
#[pymodule]
fn dpb(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "AuraSense Tech Corporation")?;
    m.add(
        "__doc__",
        "Delta-Predictive Biosensing Framework - Neuromorphic signal processing",
    )?;

    // Register core types
    m.add_class::<types::PySpikeEvent>()?;
    m.add_class::<types::PySpikeTrain>()?;
    m.add_class::<types::PyTimeSeries>()?;
    m.add_class::<types::PyGroundTruth>()?;
    m.add_class::<types::PyContext>()?;

    // Register submodules
    let encoders_module = PyModule::new(py, "encoders")?;
    encoders::register_module(&encoders_module)?;
    m.add_submodule(&encoders_module)?;

    let neurons_module = PyModule::new(py, "neurons")?;
    neurons::register_module(&neurons_module)?;
    m.add_submodule(&neurons_module)?;

    let snn_module = PyModule::new(py, "snn")?;
    snn::register_module(&snn_module)?;
    m.add_submodule(&snn_module)?;

    let training_module = PyModule::new(py, "training")?;
    training::register_module(&training_module)?;
    m.add_submodule(&training_module)?;

    let synth_module = PyModule::new(py, "synth")?;
    synth::register_module(&synth_module)?;
    m.add_submodule(&synth_module)?;

    let metrics_module = PyModule::new(py, "metrics")?;
    metrics::register_module(&metrics_module)?;
    m.add_submodule(&metrics_module)?;

    let gpu_module = PyModule::new(py, "gpu")?;
    gpu::register_module(&gpu_module)?;
    m.add_submodule(&gpu_module)?;

    Ok(())
}
