//! Python bindings for the Delta-Predictive Biosensing Framework
//!
//! # Most of this module is not wired to the library yet
//!
//! **Wired, and behaving exactly as the Rust crates do:**
//!
//! - `encoders` -- all five call [`dpb_encoders`]. `LevelCrossingEncoder` has
//!   the `Delta` default and its one-quantum reconstruction bound; the
//!   derivative, template-deviation, ECG and PPG encoders return real events
//!   where they previously returned an empty list.
//! - `neurons.LifNeuron` -- delegates to [`dpb_neurons::LifNeuron`]. Its units
//!   are biophysical (mV, ms) in consequence, matching the library.
//!
//! **Not wired.** `snn`, `synth`, `training`, `metrics` and `gpu` are still
//! standalone placeholders that import none of the DPB crates, several
//! labelled as such in comments. They have the SHAPE of the library's API but
//! their own behaviour, and corrections to the Rust code do not reach them.
//! Do not attribute results from those modules to DPB.

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
    m.add("__doc__", "Delta-Predictive Biosensing Framework - Neuromorphic signal processing")?;

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
