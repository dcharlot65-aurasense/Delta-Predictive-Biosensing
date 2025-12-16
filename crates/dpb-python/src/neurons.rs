//! Python bindings for neuron models

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Base neuron model trait
#[pyclass(name = "NeuronModel", subclass)]
pub struct PyNeuronModel {
    name: String,
    params: HashMap<String, f64>,
}

#[pymethods]
impl PyNeuronModel {
    #[new]
    #[pyo3(signature = (name="base", params=None))]
    fn new(name: &str, params: Option<HashMap<String, f64>>) -> Self {
        Self {
            name: name.to_string(),
            params: params.unwrap_or_default(),
        }
    }

    /// Step the neuron model forward one timestep
    ///
    /// Args:
    ///     input_current (float): Input current
    ///     dt (float): Time step in seconds
    ///
    /// Returns:
    ///     bool: True if neuron spiked
    fn step(&mut self, _input_current: f32, _dt: f32) -> PyResult<bool> {
        // Base implementation - override in subclasses
        Ok(false)
    }

    /// Reset neuron state
    fn reset(&mut self) {
        // Base implementation - override in subclasses
    }

    /// Get neuron name
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// Get neuron parameters
    #[getter]
    fn params(&self) -> HashMap<String, f64> {
        self.params.clone()
    }

    fn __repr__(&self) -> String {
        format!("NeuronModel(name='{}')", self.name)
    }
}

/// Leaky Integrate-and-Fire (LIF) neuron
///
/// Classic spiking neuron model with leak and threshold.
///
/// Args:
///     tau (float): Membrane time constant in seconds
///     threshold (float): Spike threshold voltage
///     reset (float): Reset voltage after spike
///     refractory_period (float): Refractory period in seconds
///
/// Example:
///     >>> neuron = LifNeuron(tau=0.02, threshold=1.0, reset=0.0)
///     >>> spiked = neuron.step(input_current=0.5, dt=0.001)
#[pyclass(name = "LifNeuron", extends=PyNeuronModel)]
pub struct PyLifNeuron {
    tau: f64,
    threshold: f64,
    reset: f64,
    refractory_period: f64,
    voltage: f64,
    refractory_time: f64,
}

#[pymethods]
impl PyLifNeuron {
    #[new]
    #[pyo3(signature = (tau=0.02, threshold=1.0, reset=0.0, refractory_period=0.002))]
    fn new(tau: f64, threshold: f64, reset: f64, refractory_period: f64) -> (Self, PyNeuronModel) {
        let mut params = HashMap::new();
        params.insert("tau".to_string(), tau);
        params.insert("threshold".to_string(), threshold);
        params.insert("reset".to_string(), reset);
        params.insert("refractory_period".to_string(), refractory_period);

        (
            Self {
                tau,
                threshold,
                reset,
                refractory_period,
                voltage: 0.0,
                refractory_time: 0.0,
            },
            PyNeuronModel {
                name: "LIF".to_string(),
                params,
            },
        )
    }

    fn step(&mut self, input_current: f32, dt: f32) -> bool {
        // Update refractory period
        if self.refractory_time > 0.0 {
            self.refractory_time -= dt as f64;
            return false;
        }

        // Integrate membrane potential
        let decay = (-dt as f64 / self.tau).exp();
        self.voltage = self.voltage * decay + input_current as f64 * (1.0 - decay);

        // Check for spike
        if self.voltage >= self.threshold {
            self.voltage = self.reset;
            self.refractory_time = self.refractory_period;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.voltage = 0.0;
        self.refractory_time = 0.0;
    }

    #[getter]
    fn voltage(&self) -> f64 {
        self.voltage
    }

    #[getter]
    fn tau(&self) -> f64 {
        self.tau
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// Adaptive Leaky Integrate-and-Fire (ALIF) neuron
///
/// LIF neuron with adaptive threshold.
///
/// Args:
///     tau (float): Membrane time constant
///     threshold (float): Base spike threshold
///     reset (float): Reset voltage
///     tau_adaptation (float): Adaptation time constant
///     adaptation_increment (float): Threshold increment per spike
///
/// Example:
///     >>> neuron = AlifNeuron(tau=0.02, threshold=1.0, tau_adaptation=0.1)
///     >>> spiked = neuron.step(input_current=0.5, dt=0.001)
#[pyclass(name = "AlifNeuron", extends=PyNeuronModel)]
pub struct PyAlifNeuron {
    tau: f64,
    threshold: f64,
    reset: f64,
    tau_adaptation: f64,
    adaptation_increment: f64,
    voltage: f64,
    threshold_adaptive: f64,
}

#[pymethods]
impl PyAlifNeuron {
    #[new]
    #[pyo3(signature = (tau=0.02, threshold=1.0, reset=0.0, tau_adaptation=0.1, adaptation_increment=0.1))]
    fn new(
        tau: f64,
        threshold: f64,
        reset: f64,
        tau_adaptation: f64,
        adaptation_increment: f64,
    ) -> (Self, PyNeuronModel) {
        let mut params = HashMap::new();
        params.insert("tau".to_string(), tau);
        params.insert("threshold".to_string(), threshold);
        params.insert("reset".to_string(), reset);
        params.insert("tau_adaptation".to_string(), tau_adaptation);
        params.insert("adaptation_increment".to_string(), adaptation_increment);

        (
            Self {
                tau,
                threshold,
                reset,
                tau_adaptation,
                adaptation_increment,
                voltage: 0.0,
                threshold_adaptive: threshold,
            },
            PyNeuronModel {
                name: "ALIF".to_string(),
                params,
            },
        )
    }

    fn step(&mut self, input_current: f32, dt: f32) -> bool {
        // Decay adaptive threshold
        let decay = (-dt as f64 / self.tau_adaptation).exp();
        self.threshold_adaptive = self.threshold + (self.threshold_adaptive - self.threshold) * decay;

        // Integrate membrane potential
        let mem_decay = (-dt as f64 / self.tau).exp();
        self.voltage = self.voltage * mem_decay + input_current as f64 * (1.0 - mem_decay);

        // Check for spike
        if self.voltage >= self.threshold_adaptive {
            self.voltage = self.reset;
            self.threshold_adaptive += self.adaptation_increment;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.voltage = 0.0;
        self.threshold_adaptive = self.threshold;
    }

    #[getter]
    fn voltage(&self) -> f64 {
        self.voltage
    }
}

/// Izhikevich neuron model
///
/// Biologically realistic neuron model with rich dynamics.
///
/// Args:
///     a (float): Time scale of recovery variable
///     b (float): Sensitivity of recovery variable
///     c (float): After-spike reset value of voltage
///     d (float): After-spike reset increment of recovery
///
/// Example:
///     >>> neuron = IzhikevichNeuron(a=0.02, b=0.2, c=-65.0, d=8.0)
///     >>> spiked = neuron.step(input_current=10.0, dt=0.001)
#[pyclass(name = "IzhikevichNeuron", extends=PyNeuronModel)]
pub struct PyIzhikevichNeuron {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    v: f64,
    u: f64,
}

#[pymethods]
impl PyIzhikevichNeuron {
    #[new]
    #[pyo3(signature = (a=0.02, b=0.2, c=-65.0, d=8.0))]
    fn new(a: f64, b: f64, c: f64, d: f64) -> (Self, PyNeuronModel) {
        let mut params = HashMap::new();
        params.insert("a".to_string(), a);
        params.insert("b".to_string(), b);
        params.insert("c".to_string(), c);
        params.insert("d".to_string(), d);

        (
            Self {
                a,
                b,
                c,
                d,
                v: c,
                u: b * c,
            },
            PyNeuronModel {
                name: "Izhikevich".to_string(),
                params,
            },
        )
    }

    fn step(&mut self, input_current: f32, dt: f32) -> bool {
        let i = input_current as f64;
        let dt_ms = dt as f64 * 1000.0; // Convert to ms for Izhikevich model

        // Integrate using Euler method
        let dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i) * dt_ms;
        let du = (self.a * (self.b * self.v - self.u)) * dt_ms;

        self.v += dv;
        self.u += du;

        // Check for spike
        if self.v >= 30.0 {
            self.v = self.c;
            self.u += self.d;
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.v = self.c;
        self.u = self.b * self.c;
    }

    #[getter]
    fn voltage(&self) -> f64 {
        self.v
    }

    #[getter]
    fn recovery(&self) -> f64 {
        self.u
    }
}

/// Hodgkin-Huxley neuron model
///
/// Detailed biophysical neuron model.
///
/// Args:
///     gNa (float): Sodium conductance
///     gK (float): Potassium conductance
///     gL (float): Leak conductance
///     ENa (float): Sodium reversal potential
///     EK (float): Potassium reversal potential
///     EL (float): Leak reversal potential
///
/// Example:
///     >>> neuron = HodgkinHuxleyNeuron()
///     >>> spiked = neuron.step(input_current=10.0, dt=0.01)
#[pyclass(name = "HodgkinHuxleyNeuron", extends=PyNeuronModel)]
pub struct PyHodgkinHuxleyNeuron {
    g_na: f64,
    g_k: f64,
    g_l: f64,
    e_na: f64,
    e_k: f64,
    e_l: f64,
    v: f64,
    m: f64,
    h: f64,
    n: f64,
}

#[pymethods]
impl PyHodgkinHuxleyNeuron {
    #[new]
    #[pyo3(signature = (gNa=120.0, gK=36.0, gL=0.3, ENa=50.0, EK=-77.0, EL=-54.387))]
    fn new(gNa: f64, gK: f64, gL: f64, ENa: f64, EK: f64, EL: f64) -> (Self, PyNeuronModel) {
        let mut params = HashMap::new();
        params.insert("gNa".to_string(), gNa);
        params.insert("gK".to_string(), gK);
        params.insert("gL".to_string(), gL);
        params.insert("ENa".to_string(), ENa);
        params.insert("EK".to_string(), EK);
        params.insert("EL".to_string(), EL);

        (
            Self {
                g_na: gNa,
                g_k: gK,
                g_l: gL,
                e_na: ENa,
                e_k: EK,
                e_l: EL,
                v: -65.0,
                m: 0.05,
                h: 0.6,
                n: 0.32,
            },
            PyNeuronModel {
                name: "HodgkinHuxley".to_string(),
                params,
            },
        )
    }

    fn step(&mut self, input_current: f32, dt: f32) -> bool {
        let i_ext = input_current as f64;

        // Calculate ionic currents (simplified)
        let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);

        // Update voltage
        let dv = (i_ext - i_na - i_k - i_l) * dt as f64;
        let old_v = self.v;
        self.v += dv;

        // Update gating variables (simplified)
        self.m += dt as f64 * (0.1 * (self.v + 40.0) / (1.0 - (-0.1 * (self.v + 40.0)).exp()) * (1.0 - self.m) - 4.0 * ((-self.v - 65.0) / 18.0).exp() * self.m);
        self.h += dt as f64 * (0.07 * ((-self.v - 65.0) / 20.0).exp() * (1.0 - self.h) - (1.0 / (1.0 + ((-self.v - 35.0) / 10.0).exp())) * self.h);
        self.n += dt as f64 * (0.01 * (self.v + 55.0) / (1.0 - (-0.1 * (self.v + 55.0)).exp()) * (1.0 - self.n) - 0.125 * ((-self.v - 65.0) / 80.0).exp() * self.n);

        // Detect spike (crossing 0 mV threshold)
        old_v < 0.0 && self.v >= 0.0
    }

    fn reset(&mut self) {
        self.v = -65.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.32;
    }
}

/// Factory function to create neurons by name
///
/// Args:
///     name (str): Neuron model name ('lif', 'alif', 'izhikevich', 'hodgkin_huxley')
///     config (dict): Configuration dictionary
///
/// Returns:
///     NeuronModel: Configured neuron instance
///
/// Example:
///     >>> neuron = create_neuron('lif', {'tau': 0.02, 'threshold': 1.0})
#[pyfunction]
fn create_neuron(name: &str, config: Option<&Bound<'_, PyDict>>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let neuron: PyObject = match name {
            "lif" => {
                let tau = config
                    .and_then(|c| c.get_item("tau").ok().flatten())
                    .map(|v| v.extract::<f64>().unwrap_or(0.02))
                    .unwrap_or(0.02);
                let threshold = config
                    .and_then(|c| c.get_item("threshold").ok().flatten())
                    .map(|v| v.extract::<f64>().unwrap_or(1.0))
                    .unwrap_or(1.0);

                Py::new(py, PyLifNeuron::new(tau, threshold, 0.0, 0.002))?.into_py(py)
            }
            "alif" => {
                Py::new(py, PyAlifNeuron::new(0.02, 1.0, 0.0, 0.1, 0.1))?.into_py(py)
            }
            "izhikevich" => {
                Py::new(py, PyIzhikevichNeuron::new(0.02, 0.2, -65.0, 8.0))?.into_py(py)
            }
            "hodgkin_huxley" => {
                Py::new(py, PyHodgkinHuxleyNeuron::new(120.0, 36.0, 0.3, 50.0, -77.0, -54.387))?.into_py(py)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown neuron model: {}",
                    name
                )))
            }
        };

        Ok(neuron)
    })
}

/// Register neuron module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNeuronModel>()?;
    m.add_class::<PyLifNeuron>()?;
    m.add_class::<PyAlifNeuron>()?;
    m.add_class::<PyIzhikevichNeuron>()?;
    m.add_class::<PyHodgkinHuxleyNeuron>()?;
    m.add_function(wrap_pyfunction!(create_neuron, m)?)?;
    Ok(())
}
