//! Python bindings for event-based encoders

use crate::types::{PySpikeEvent, PySpikeTrain, PyTimeSeries};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Base encoder trait - all encoders implement this
#[pyclass(name = "EventEncoder", subclass)]
pub struct PyEventEncoder {
    name: String,
    config: HashMap<String, f64>,
}

#[pymethods]
impl PyEventEncoder {
    #[new]
    #[pyo3(signature = (name="base", config=None))]
    fn new(name: &str, config: Option<HashMap<String, f64>>) -> Self {
        Self {
            name: name.to_string(),
            config: config.unwrap_or_default(),
        }
    }

    /// Encode a time series signal to spike events
    fn encode(&self, _signal: &PyTimeSeries) -> PyResult<PySpikeTrain> {
        // Base implementation - override in subclasses
        Ok(PySpikeTrain::new(None, 0.0, 1))
    }

    /// Get encoder name
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// Get encoder configuration
    #[getter]
    fn config(&self) -> HashMap<String, f64> {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!("EventEncoder(name='{}')", self.name)
    }
}

/// Level crossing encoder
///
/// Generates spike events when signal crosses threshold levels.
///
/// Args:
///     threshold (float): Crossing threshold
///     positive_polarity (bool): Emit positive polarity on up-crossing
///     negative_polarity (bool): Emit negative polarity on down-crossing
///
/// Example:
///     >>> encoder = LevelCrossingEncoder(threshold=0.5)
///     >>> events = encoder.encode(signal)
#[pyclass(name = "LevelCrossingEncoder", extends=PyEventEncoder)]
pub struct PyLevelCrossingEncoder {
    threshold: f64,
    positive_polarity: bool,
    negative_polarity: bool,
}

#[pymethods]
impl PyLevelCrossingEncoder {
    #[new]
    #[pyo3(signature = (threshold=0.5, positive_polarity=true, negative_polarity=true))]
    fn new(threshold: f64, positive_polarity: bool, negative_polarity: bool) -> (Self, PyEventEncoder) {
        let mut config = HashMap::new();
        config.insert("threshold".to_string(), threshold);
        config.insert("positive_polarity".to_string(), if positive_polarity { 1.0 } else { 0.0 });
        config.insert("negative_polarity".to_string(), if negative_polarity { 1.0 } else { 0.0 });

        (
            Self {
                threshold,
                positive_polarity,
                negative_polarity,
            },
            PyEventEncoder {
                name: "LevelCrossing".to_string(),
                config,
            },
        )
    }

    /// Encode signal to spike train
    fn encode(&self, signal: &PyTimeSeries, py: Python) -> PyResult<PySpikeTrain> {
        let duration = signal.duration(py);
        let num_channels = signal.num_channels(py);
        let num_samples = signal.num_samples(py);

        // Placeholder implementation - would call Rust encoder
        let mut events = Vec::new();

        // Simple level crossing detection (simplified for example)
        for ch in 0..num_channels {
            let channel_data = signal.get_channel(py, ch)?;
            let readonly = channel_data.bind(py).readonly();
            let samples = readonly.as_slice()?;

            let mut last_val = 0.0;
            for (i, &val) in samples.iter().enumerate() {
                let t = i as f64 / signal.sample_rate;

                // Up-crossing
                if self.positive_polarity && last_val < self.threshold as f32 && val >= self.threshold as f32 {
                    events.push(PySpikeEvent::new(t, ch as u32, 1, 1.0));
                }
                // Down-crossing
                if self.negative_polarity && last_val > self.threshold as f32 && val <= self.threshold as f32 {
                    events.push(PySpikeEvent::new(t, ch as u32, -1, 1.0));
                }

                last_val = val;
            }
        }

        Ok(PySpikeTrain::new(Some(events), duration, num_channels as u32))
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.threshold
    }

    #[setter]
    fn set_threshold(&mut self, value: f64) {
        self.threshold = value;
    }
}

/// Template deviation encoder
///
/// Generates spikes based on deviation from a population template.
///
/// Args:
///     template_type (str): Type of template ('ecg_rpeak', 'ppg_peak', etc.)
///     deviation_threshold (float): Minimum deviation to generate spike
///     window_size (int): Window size for template matching
///
/// Example:
///     >>> encoder = TemplateDeviationEncoder(template_type='ecg_rpeak')
///     >>> events = encoder.encode(signal)
#[pyclass(name = "TemplateDeviationEncoder", extends=PyEventEncoder)]
pub struct PyTemplateDeviationEncoder {
    template_type: String,
    deviation_threshold: f64,
    window_size: usize,
}

#[pymethods]
impl PyTemplateDeviationEncoder {
    #[new]
    #[pyo3(signature = (template_type="generic", deviation_threshold=0.1, window_size=100))]
    fn new(template_type: &str, deviation_threshold: f64, window_size: usize) -> (Self, PyEventEncoder) {
        let mut config = HashMap::new();
        config.insert("deviation_threshold".to_string(), deviation_threshold);
        config.insert("window_size".to_string(), window_size as f64);

        (
            Self {
                template_type: template_type.to_string(),
                deviation_threshold,
                window_size,
            },
            PyEventEncoder {
                name: "TemplateDeviation".to_string(),
                config,
            },
        )
    }

    fn encode(&self, signal: &PyTimeSeries, py: Python) -> PyResult<PySpikeTrain> {
        let duration = signal.duration(py);
        let num_channels = signal.num_channels(py);

        // Placeholder - would implement actual template matching
        let events = Vec::new();

        Ok(PySpikeTrain::new(Some(events), duration, num_channels as u32))
    }

    #[getter]
    fn template_type(&self) -> String {
        self.template_type.clone()
    }
}

/// Derivative-based encoder
///
/// Generates spikes based on signal derivative changes.
///
/// Args:
///     threshold (float): Derivative threshold
///     order (int): Derivative order (1 or 2)
///
/// Example:
///     >>> encoder = DerivativeEncoder(threshold=0.1, order=1)
///     >>> events = encoder.encode(signal)
#[pyclass(name = "DerivativeEncoder", extends=PyEventEncoder)]
pub struct PyDerivativeEncoder {
    threshold: f64,
    order: usize,
}

#[pymethods]
impl PyDerivativeEncoder {
    #[new]
    #[pyo3(signature = (threshold=0.1, order=1))]
    fn new(threshold: f64, order: usize) -> (Self, PyEventEncoder) {
        let mut config = HashMap::new();
        config.insert("threshold".to_string(), threshold);
        config.insert("order".to_string(), order as f64);

        (
            Self { threshold, order },
            PyEventEncoder {
                name: "Derivative".to_string(),
                config,
            },
        )
    }

    fn encode(&self, signal: &PyTimeSeries, py: Python) -> PyResult<PySpikeTrain> {
        let duration = signal.duration(py);
        let num_channels = signal.num_channels(py);

        // Placeholder - would implement derivative-based encoding
        let events = Vec::new();

        Ok(PySpikeTrain::new(Some(events), duration, num_channels as u32))
    }
}

/// ECG R-peak encoder
///
/// Specialized encoder for ECG R-peak detection.
///
/// Args:
///     threshold (float): Detection threshold
///     refractory_ms (float): Refractory period in milliseconds
///     min_rr_interval (float): Minimum RR interval in seconds
///
/// Example:
///     >>> encoder = EcgRPeakEncoder(threshold=0.5, refractory_ms=200)
///     >>> events = encoder.encode(ecg_signal)
#[pyclass(name = "EcgRPeakEncoder", extends=PyEventEncoder)]
pub struct PyEcgRPeakEncoder {
    threshold: f64,
    refractory_ms: f64,
    min_rr_interval: f64,
}

#[pymethods]
impl PyEcgRPeakEncoder {
    #[new]
    #[pyo3(signature = (threshold=0.5, refractory_ms=200.0, min_rr_interval=0.4))]
    fn new(threshold: f64, refractory_ms: f64, min_rr_interval: f64) -> (Self, PyEventEncoder) {
        let mut config = HashMap::new();
        config.insert("threshold".to_string(), threshold);
        config.insert("refractory_ms".to_string(), refractory_ms);
        config.insert("min_rr_interval".to_string(), min_rr_interval);

        (
            Self {
                threshold,
                refractory_ms,
                min_rr_interval,
            },
            PyEventEncoder {
                name: "EcgRPeak".to_string(),
                config,
            },
        )
    }

    fn encode(&self, signal: &PyTimeSeries, py: Python) -> PyResult<PySpikeTrain> {
        let duration = signal.duration(py);
        let num_channels = signal.num_channels(py);

        // Placeholder - would implement R-peak detection
        let events = Vec::new();

        Ok(PySpikeTrain::new(Some(events), duration, num_channels as u32))
    }
}

/// PPG peak encoder
///
/// Specialized encoder for PPG (photoplethysmography) peak detection.
///
/// Args:
///     threshold (float): Detection threshold
///     min_peak_distance (float): Minimum distance between peaks in seconds
///
/// Example:
///     >>> encoder = PpgPeakEncoder(threshold=0.3, min_peak_distance=0.5)
///     >>> events = encoder.encode(ppg_signal)
#[pyclass(name = "PpgPeakEncoder", extends=PyEventEncoder)]
pub struct PyPpgPeakEncoder {
    threshold: f64,
    min_peak_distance: f64,
}

#[pymethods]
impl PyPpgPeakEncoder {
    #[new]
    #[pyo3(signature = (threshold=0.3, min_peak_distance=0.5))]
    fn new(threshold: f64, min_peak_distance: f64) -> (Self, PyEventEncoder) {
        let mut config = HashMap::new();
        config.insert("threshold".to_string(), threshold);
        config.insert("min_peak_distance".to_string(), min_peak_distance);

        (
            Self {
                threshold,
                min_peak_distance,
            },
            PyEventEncoder {
                name: "PpgPeak".to_string(),
                config,
            },
        )
    }

    fn encode(&self, signal: &PyTimeSeries, py: Python) -> PyResult<PySpikeTrain> {
        let duration = signal.duration(py);
        let num_channels = signal.num_channels(py);

        // Placeholder - would implement PPG peak detection
        let events = Vec::new();

        Ok(PySpikeTrain::new(Some(events), duration, num_channels as u32))
    }
}

/// Factory function to create encoders by name
///
/// Args:
///     name (str): Encoder name ('level_crossing', 'template_deviation', etc.)
///     config (dict): Configuration dictionary
///
/// Returns:
///     EventEncoder: Configured encoder instance
///
/// Example:
///     >>> encoder = create_encoder('level_crossing', {'threshold': 0.5})
#[pyfunction]
fn create_encoder(name: &str, config: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let encoder: Py<PyAny> = match name {
            "level_crossing" => {
                let threshold = config
                    .and_then(|c| c.get_item("threshold").ok().flatten())
                    .map(|v| v.extract::<f64>().unwrap_or(0.5))
                    .unwrap_or(0.5);

                Py::new(py, PyLevelCrossingEncoder::new(threshold, true, true))?.into_any()
            }
            "template_deviation" => {
                let template_type = config
                    .and_then(|c| c.get_item("template_type").ok().flatten())
                    .map(|v| v.extract::<String>().unwrap_or_else(|_| "generic".to_string()))
                    .unwrap_or_else(|| "generic".to_string());

                Py::new(py, PyTemplateDeviationEncoder::new(&template_type, 0.1, 100))?.into_any()
            }
            "derivative" => {
                let threshold = config
                    .and_then(|c| c.get_item("threshold").ok().flatten())
                    .map(|v| v.extract::<f64>().unwrap_or(0.1))
                    .unwrap_or(0.1);

                Py::new(py, PyDerivativeEncoder::new(threshold, 1))?.into_any()
            }
            "ecg_rpeak" => {
                Py::new(py, PyEcgRPeakEncoder::new(0.5, 200.0, 0.4))?.into_any()
            }
            "ppg_peak" => {
                Py::new(py, PyPpgPeakEncoder::new(0.3, 0.5))?.into_any()
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown encoder: {}",
                    name
                )))
            }
        };

        Ok(encoder)
    })
}

/// Register encoder module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEventEncoder>()?;
    m.add_class::<PyLevelCrossingEncoder>()?;
    m.add_class::<PyTemplateDeviationEncoder>()?;
    m.add_class::<PyDerivativeEncoder>()?;
    m.add_class::<PyEcgRPeakEncoder>()?;
    m.add_class::<PyPpgPeakEncoder>()?;
    m.add_function(wrap_pyfunction!(create_encoder, m)?)?;
    Ok(())
}
