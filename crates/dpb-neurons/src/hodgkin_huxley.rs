//! Biophysical neuron models based on conductances.
//!
//! This module implements detailed conductance-based models:
//! 1. Hodgkin-Huxley: Full conductance-based model with Na+, K+, and leak
//! 2. FitzHugh-Nagumo: Simplified 2D reduction of HH
//! 3. Morris-Lecar: 2D model with Ca++ and K+ dynamics

use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

// ============================================================================
// 1. Hodgkin-Huxley Neuron
// ============================================================================

/// Hodgkin-Huxley neuron state with gating variables.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct HodgkinHuxleyState {
    /// Membrane potential in mV
    pub v: f32,
    /// Sodium activation gating variable
    pub m: f32,
    /// Sodium inactivation gating variable
    pub h: f32,
    /// Potassium activation gating variable
    pub n: f32,
}

impl Default for HodgkinHuxleyState {
    fn default() -> Self {
        // Initialize at resting potential equilibrium
        Self {
            v: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.32,
        }
    }
}

/// Hodgkin-Huxley neuron configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct HodgkinHuxleyConfig {
    /// Membrane capacitance in μF/cm²
    pub c_m: f32,
    /// Sodium conductance in mS/cm²
    pub g_na: f32,
    /// Potassium conductance in mS/cm²
    pub g_k: f32,
    /// Leak conductance in mS/cm²
    pub g_l: f32,
    /// Sodium reversal potential in mV
    pub e_na: f32,
    /// Potassium reversal potential in mV
    pub e_k: f32,
    /// Leak reversal potential in mV
    pub e_l: f32,
    /// Spike threshold in mV (for spike detection)
    pub v_thresh: f32,
}

impl Default for HodgkinHuxleyConfig {
    fn default() -> Self {
        Self {
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.387,
            v_thresh: 0.0, // Detect spikes when crossing 0 mV
        }
    }
}

impl NeuronConfig for HodgkinHuxleyConfig {
    fn validate(&self) -> Result<(), String> {
        if self.c_m <= 0.0 || self.g_na <= 0.0 || self.g_k <= 0.0 || self.g_l <= 0.0 {
            return Err("Conductances and capacitance must be positive".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Hodgkin-Huxley neuron model with detailed ion channel dynamics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HodgkinHuxleyNeuron {
    pub state: HodgkinHuxleyState,
    pub config: HodgkinHuxleyConfig,
    /// Track if we crossed threshold (for spike detection)
    crossed_threshold: bool,
}

impl HodgkinHuxleyNeuron {
    pub fn new(config: HodgkinHuxleyConfig) -> Self {
        Self {
            state: HodgkinHuxleyState::default(),
            config,
            crossed_threshold: false,
        }
    }

    /// Alpha function for m (sodium activation).
    fn alpha_m(v: f32) -> f32 {
        let x = v + 40.0;
        if x.abs() < 1e-4 {
            1.0 // Limit as x -> 0
        } else {
            0.1 * x / (1.0 - (-x / 10.0).exp())
        }
    }

    /// Beta function for m.
    fn beta_m(v: f32) -> f32 {
        4.0 * (-(v + 65.0) / 18.0).exp()
    }

    /// Alpha function for h (sodium inactivation).
    fn alpha_h(v: f32) -> f32 {
        0.07 * (-(v + 65.0) / 20.0).exp()
    }

    /// Beta function for h.
    fn beta_h(v: f32) -> f32 {
        1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
    }

    /// Alpha function for n (potassium activation).
    fn alpha_n(v: f32) -> f32 {
        let x = v + 55.0;
        if x.abs() < 1e-4 {
            0.1 // Limit as x -> 0
        } else {
            0.01 * x / (1.0 - (-x / 10.0).exp())
        }
    }

    /// Beta function for n.
    fn beta_n(v: f32) -> f32 {
        0.125 * (-(v + 65.0) / 80.0).exp()
    }
}

impl MembraneDynamics for HodgkinHuxleyNeuron {
    fn membrane_potential(&self) -> f32 {
        self.state.v
    }

    fn set_membrane_potential(&mut self, v: f32) {
        self.state.v = v;
    }

    fn rest_potential(&self) -> f32 {
        -65.0
    }
}

impl NeuronModel for HodgkinHuxleyNeuron {
    type Config = HodgkinHuxleyConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        let v = self.state.v;

        // Compute gating variable derivatives
        let alpha_m = Self::alpha_m(v);
        let beta_m = Self::beta_m(v);
        let alpha_h = Self::alpha_h(v);
        let beta_h = Self::beta_h(v);
        let alpha_n = Self::alpha_n(v);
        let beta_n = Self::beta_n(v);

        // Update gating variables
        self.state.m += (alpha_m * (1.0 - self.state.m) - beta_m * self.state.m) * dt;
        self.state.h += (alpha_h * (1.0 - self.state.h) - beta_h * self.state.h) * dt;
        self.state.n += (alpha_n * (1.0 - self.state.n) - beta_n * self.state.n) * dt;

        // Clamp gating variables to [0, 1]
        self.state.m = self.state.m.clamp(0.0, 1.0);
        self.state.h = self.state.h.clamp(0.0, 1.0);
        self.state.n = self.state.n.clamp(0.0, 1.0);

        // Compute ionic currents
        let i_na = self.config.g_na * self.state.m.powi(3) * self.state.h * (v - self.config.e_na);
        let i_k = self.config.g_k * self.state.n.powi(4) * (v - self.config.e_k);
        let i_l = self.config.g_l * (v - self.config.e_l);

        // Update membrane potential
        let dv = (-i_na - i_k - i_l + input_current) / self.config.c_m;
        self.state.v += dv * dt;

        // Spike detection: crossing threshold upward
        let spiked = if !self.crossed_threshold && v >= self.config.v_thresh {
            self.crossed_threshold = true;
            true
        } else if v < self.config.v_thresh - 10.0 {
            self.crossed_threshold = false;
            false
        } else {
            false
        };

        spiked
    }

    fn reset(&mut self) {
        // HH naturally resets through dynamics, but we can reinitialize
        self.state = HodgkinHuxleyState::default();
        self.crossed_threshold = false;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// ============================================================================
// 2. FitzHugh-Nagumo Neuron
// ============================================================================

/// FitzHugh-Nagumo neuron state (2D simplified model).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct FitzHughNagumoState {
    /// Membrane potential (fast variable)
    pub v: f32,
    /// Recovery variable (slow variable)
    pub w: f32,
    pub _padding: [f32; 2],
}

impl Default for FitzHughNagumoState {
    fn default() -> Self {
        Self {
            v: 0.0,
            w: 0.0,
            _padding: [0.0; 2],
        }
    }
}

/// FitzHugh-Nagumo configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct FitzHughNagumoConfig {
    /// Time scale separation
    pub epsilon: f32,
    /// Cubic nonlinearity parameter
    pub a: f32,
    /// Recovery coupling
    pub b: f32,
    /// External drive
    pub i_ext: f32,
    /// Spike threshold
    pub v_thresh: f32,
    pub _padding: [f32; 3],
}

impl Default for FitzHughNagumoConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.08,
            a: 0.7,
            b: 0.8,
            i_ext: 0.0,
            v_thresh: 0.5,
            _padding: [0.0; 3],
        }
    }
}

impl NeuronConfig for FitzHughNagumoConfig {
    fn validate(&self) -> Result<(), String> {
        if self.epsilon <= 0.0 {
            return Err("Epsilon must be positive".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// FitzHugh-Nagumo neuron - simplified 2D model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitzHughNagumoNeuron {
    pub state: FitzHughNagumoState,
    pub config: FitzHughNagumoConfig,
    crossed_threshold: bool,
}

impl FitzHughNagumoNeuron {
    pub fn new(config: FitzHughNagumoConfig) -> Self {
        Self {
            state: FitzHughNagumoState::default(),
            config,
            crossed_threshold: false,
        }
    }
}

impl MembraneDynamics for FitzHughNagumoNeuron {
    fn membrane_potential(&self) -> f32 {
        self.state.v
    }

    fn set_membrane_potential(&mut self, v: f32) {
        self.state.v = v;
    }

    fn rest_potential(&self) -> f32 {
        0.0
    }
}

impl NeuronModel for FitzHughNagumoNeuron {
    type Config = FitzHughNagumoConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        // FitzHugh-Nagumo equations:
        // dv/dt = v - v³/3 - w + I
        // dw/dt = ε(v + a - bw)

        let v = self.state.v;
        let w = self.state.w;

        let dv = v - v.powi(3) / 3.0 - w + input_current + self.config.i_ext;
        let dw = self.config.epsilon * (v + self.config.a - self.config.b * w);

        self.state.v += dv * dt;
        self.state.w += dw * dt;

        // Spike detection
        let spiked = if !self.crossed_threshold && self.state.v >= self.config.v_thresh {
            self.crossed_threshold = true;
            true
        } else if self.state.v < self.config.v_thresh - 0.2 {
            self.crossed_threshold = false;
            false
        } else {
            false
        };

        spiked
    }

    fn reset(&mut self) {
        self.state = FitzHughNagumoState::default();
        self.crossed_threshold = false;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// ============================================================================
// 3. Morris-Lecar Neuron
// ============================================================================

/// Morris-Lecar neuron state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct MorrisLecarState {
    /// Membrane potential in mV
    pub v: f32,
    /// K+ channel activation
    pub n: f32,
    pub _padding: [f32; 2],
}

impl Default for MorrisLecarState {
    fn default() -> Self {
        Self {
            v: -60.0,
            n: 0.0,
            _padding: [0.0; 2],
        }
    }
}

/// Morris-Lecar configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct MorrisLecarConfig {
    /// Membrane capacitance in μF/cm²
    pub c_m: f32,
    /// Ca++ conductance in mS/cm²
    pub g_ca: f32,
    /// K+ conductance in mS/cm²
    pub g_k: f32,
    /// Leak conductance in mS/cm²
    pub g_l: f32,
    /// Ca++ reversal potential in mV
    pub e_ca: f32,
    /// K+ reversal potential in mV
    pub e_k: f32,
    /// Leak reversal potential in mV
    pub e_l: f32,
    /// Ca++ activation midpoint in mV
    pub v1: f32,
    /// Ca++ activation slope in mV
    pub v2: f32,
    /// K+ activation midpoint in mV
    pub v3: f32,
    /// K+ activation slope in mV
    pub v4: f32,
    /// K+ time constant in ms
    pub phi: f32,
    /// Spike threshold in mV
    pub v_thresh: f32,
    pub _padding: [f32; 3],
}

impl Default for MorrisLecarConfig {
    fn default() -> Self {
        Self {
            c_m: 20.0,
            g_ca: 4.4,
            g_k: 8.0,
            g_l: 2.0,
            e_ca: 120.0,
            e_k: -84.0,
            e_l: -60.0,
            v1: -1.2,
            v2: 18.0,
            v3: 2.0,
            v4: 30.0,
            phi: 0.04,
            v_thresh: -20.0,
            _padding: [0.0; 3],
        }
    }
}

impl NeuronConfig for MorrisLecarConfig {
    fn validate(&self) -> Result<(), String> {
        if self.c_m <= 0.0 || self.g_ca <= 0.0 || self.g_k <= 0.0 || self.g_l <= 0.0 {
            return Err("Conductances and capacitance must be positive".to_string());
        }
        if self.phi <= 0.0 {
            return Err("Phi must be positive".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Morris-Lecar neuron - 2D model with Ca++ and K+ dynamics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorrisLecarNeuron {
    pub state: MorrisLecarState,
    pub config: MorrisLecarConfig,
    crossed_threshold: bool,
}

impl MorrisLecarNeuron {
    pub fn new(config: MorrisLecarConfig) -> Self {
        Self {
            state: MorrisLecarState::default(),
            config,
            crossed_threshold: false,
        }
    }

    /// Steady-state Ca++ activation.
    fn m_inf(&self, v: f32) -> f32 {
        0.5 * (1.0 + ((v - self.config.v1) / self.config.v2).tanh())
    }

    /// Steady-state K+ activation.
    fn n_inf(&self, v: f32) -> f32 {
        0.5 * (1.0 + ((v - self.config.v3) / self.config.v4).tanh())
    }

    /// K+ activation time constant.
    fn tau_n(&self, v: f32) -> f32 {
        1.0 / (self.config.phi * ((v - self.config.v3) / (2.0 * self.config.v4)).cosh())
    }
}

impl MembraneDynamics for MorrisLecarNeuron {
    fn membrane_potential(&self) -> f32 {
        self.state.v
    }

    fn set_membrane_potential(&mut self, v: f32) {
        self.state.v = v;
    }

    fn rest_potential(&self) -> f32 {
        self.config.e_l
    }
}

impl NeuronModel for MorrisLecarNeuron {
    type Config = MorrisLecarConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        let v = self.state.v;
        let n = self.state.n;

        // Compute steady states
        let m_inf = self.m_inf(v);
        let n_inf = self.n_inf(v);
        let tau_n = self.tau_n(v);

        // Ionic currents
        let i_ca = self.config.g_ca * m_inf * (v - self.config.e_ca);
        let i_k = self.config.g_k * n * (v - self.config.e_k);
        let i_l = self.config.g_l * (v - self.config.e_l);

        // Update membrane potential
        let dv = (-i_ca - i_k - i_l + input_current) / self.config.c_m;
        self.state.v += dv * dt;

        // Update K+ activation
        let dn = (n_inf - n) / tau_n;
        self.state.n += dn * dt;

        // Clamp n to [0, 1]
        self.state.n = self.state.n.clamp(0.0, 1.0);

        // Spike detection
        let spiked = if !self.crossed_threshold && self.state.v >= self.config.v_thresh {
            self.crossed_threshold = true;
            true
        } else if self.state.v < self.config.v_thresh - 10.0 {
            self.crossed_threshold = false;
            false
        } else {
            false
        };

        spiked
    }

    fn reset(&mut self) {
        self.state = MorrisLecarState::default();
        self.crossed_threshold = false;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hodgkin_huxley_default() {
        let neuron = HodgkinHuxleyNeuron::new(HodgkinHuxleyConfig::default());
        assert_eq!(neuron.membrane_potential(), -65.0);
        assert!(neuron.state.m > 0.0 && neuron.state.m < 1.0);
        assert!(neuron.state.h > 0.0 && neuron.state.h < 1.0);
        assert!(neuron.state.n > 0.0 && neuron.state.n < 1.0);
    }

    #[test]
    fn test_hodgkin_huxley_spike() {
        let mut neuron = HodgkinHuxleyNeuron::new(HodgkinHuxleyConfig::default());
        let mut spike_count = 0;

        // Apply current pulse
        for _ in 0..1000 {
            if neuron.update(10.0, 0.01) {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0, "HH neuron should spike with input");
    }

    #[test]
    fn test_fitzhugh_nagumo() {
        let mut neuron = FitzHughNagumoNeuron::new(FitzHughNagumoConfig::default());
        let v_initial = neuron.membrane_potential();

        neuron.update(0.5, 0.1);

        assert!(neuron.membrane_potential() != v_initial);
    }

    #[test]
    fn test_fitzhugh_nagumo_spike() {
        let mut neuron = FitzHughNagumoNeuron::new(FitzHughNagumoConfig::default());
        let mut spike_count = 0;

        for _ in 0..1000 {
            if neuron.update(0.5, 0.1) {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0);
    }

    #[test]
    fn test_morris_lecar() {
        let neuron = MorrisLecarNeuron::new(MorrisLecarConfig::default());
        assert_eq!(neuron.membrane_potential(), -60.0);
        assert_eq!(neuron.state.n, 0.0);
    }

    #[test]
    fn test_morris_lecar_spike() {
        let mut neuron = MorrisLecarNeuron::new(MorrisLecarConfig::default());
        let mut spike_count = 0;

        // Morris-Lecar needs sufficient current to overcome K+ conductance
        for _ in 0..2000 {
            if neuron.update(100.0, 0.1) {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0);
    }

    #[test]
    fn test_config_validations() {
        let mut hh_config = HodgkinHuxleyConfig::default();
        assert!(hh_config.validate().is_ok());
        hh_config.g_na = -1.0;
        assert!(hh_config.validate().is_err());

        let mut fhn_config = FitzHughNagumoConfig::default();
        assert!(fhn_config.validate().is_ok());
        fhn_config.epsilon = 0.0;
        assert!(fhn_config.validate().is_err());

        let mut ml_config = MorrisLecarConfig::default();
        assert!(ml_config.validate().is_ok());
        ml_config.phi = -0.1;
        assert!(ml_config.validate().is_err());
    }
}
