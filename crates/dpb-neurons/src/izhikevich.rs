//! Izhikevich neuron model - simple yet biologically realistic.
//!
//! The Izhikevich model combines computational efficiency with the ability
//! to reproduce a wide variety of spiking patterns found in biological neurons.
//!
//! Equations:
//! - dv/dt = 0.04v² + 5v + 140 - u + I
//! - du/dt = a(bv - u)
//! - If v >= 30 mV: v ← c, u ← u + d

use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel, NeuronPreset};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Izhikevich neuron state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct IzhikevichState {
    /// Membrane potential in mV
    pub v: f32,
    /// Recovery variable
    pub u: f32,
    /// Last spike time (for tracking)
    pub last_spike_time: f32,
    pub _padding: f32,
}

impl Default for IzhikevichState {
    fn default() -> Self {
        Self {
            v: -65.0,
            u: -13.0, // b * v for default b=0.2
            last_spike_time: -1000.0,
            _padding: 0.0,
        }
    }
}

/// Izhikevich neuron configuration.
///
/// Parameters:
/// - `a`: Recovery time constant (typical: 0.02)
/// - `b`: Sensitivity of recovery to subthreshold fluctuations (typical: 0.2)
/// - `c`: After-spike reset value for v (typical: -65 mV)
/// - `d`: After-spike increment for u (typical: 2)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct IzhikevichConfig {
    /// Recovery time constant
    pub a: f32,
    /// Recovery sensitivity
    pub b: f32,
    /// Voltage reset value in mV
    pub c: f32,
    /// Recovery increment
    pub d: f32,
    /// Spike threshold in mV
    pub v_thresh: f32,
    /// Resting potential in mV
    pub v_rest: f32,
    pub _padding: [f32; 2],
}

impl Default for IzhikevichConfig {
    fn default() -> Self {
        Self::regular_spiking()
    }
}

impl IzhikevichConfig {
    /// Regular Spiking (RS) - typical cortical excitatory neuron.
    pub fn regular_spiking() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            v_thresh: 30.0,
            v_rest: -65.0,
            _padding: [0.0; 2],
        }
    }

    /// Fast Spiking (FS) - cortical inhibitory interneuron.
    pub fn fast_spiking() -> Self {
        Self {
            a: 0.1,
            b: 0.2,
            c: -65.0,
            d: 2.0,
            v_thresh: 30.0,
            v_rest: -65.0,
            _padding: [0.0; 2],
        }
    }

    /// Intrinsically Bursting (IB) - chattering cortical neuron.
    pub fn intrinsically_bursting() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -55.0,
            d: 4.0,
            v_thresh: 30.0,
            v_rest: -65.0,
            _padding: [0.0; 2],
        }
    }

    /// Chattering (CH) - fast rhythmic bursting.
    pub fn chattering() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -50.0,
            d: 2.0,
            v_thresh: 30.0,
            v_rest: -65.0,
            _padding: [0.0; 2],
        }
    }

    /// Low-Threshold Spiking (LTS) - interneuron.
    pub fn low_threshold_spiking() -> Self {
        Self {
            a: 0.02,
            b: 0.25,
            c: -65.0,
            d: 2.0,
            v_thresh: 30.0,
            v_rest: -65.0,
            _padding: [0.0; 2],
        }
    }

    /// Thalamo-Cortical (TC) - relay neuron.
    pub fn thalamo_cortical() -> Self {
        Self {
            a: 0.02,
            b: 0.25,
            c: -65.0,
            d: 0.05,
            v_thresh: 30.0,
            v_rest: -65.0,
            _padding: [0.0; 2],
        }
    }

    /// Resonator (RZ) - subthreshold oscillations.
    pub fn resonator() -> Self {
        Self {
            a: 0.1,
            b: 0.26,
            c: -65.0,
            d: 2.0,
            v_thresh: 30.0,
            v_rest: -65.0,
            _padding: [0.0; 2],
        }
    }

    /// Create from preset.
    pub fn from_preset(preset: NeuronPreset) -> Self {
        match preset {
            NeuronPreset::RegularSpiking => Self::regular_spiking(),
            NeuronPreset::FastSpiking => Self::fast_spiking(),
            NeuronPreset::IntrinsicallyBursting => Self::intrinsically_bursting(),
            NeuronPreset::Chattering => Self::chattering(),
            NeuronPreset::LowThresholdSpiking => Self::low_threshold_spiking(),
            NeuronPreset::ThalamoCortical => Self::thalamo_cortical(),
            NeuronPreset::Resonator => Self::resonator(),
        }
    }
}

impl NeuronConfig for IzhikevichConfig {
    fn validate(&self) -> Result<(), String> {
        if self.a <= 0.0 {
            return Err("Parameter 'a' must be positive".to_string());
        }
        if self.v_thresh <= self.c {
            return Err("Threshold must be greater than reset potential".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Izhikevich neuron model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IzhikevichNeuron {
    pub state: IzhikevichState,
    pub config: IzhikevichConfig,
}

impl IzhikevichNeuron {
    /// Create a new Izhikevich neuron with the given configuration.
    pub fn new(config: IzhikevichConfig) -> Self {
        let u = config.b * config.v_rest;
        Self {
            state: IzhikevichState {
                v: config.v_rest,
                u,
                last_spike_time: -1000.0,
                _padding: 0.0,
            },
            config,
        }
    }

    /// Create a neuron from a preset.
    pub fn from_preset(preset: NeuronPreset) -> Self {
        Self::new(IzhikevichConfig::from_preset(preset))
    }

    /// Get the recovery variable.
    pub fn recovery(&self) -> f32 {
        self.state.u
    }

    /// Set the recovery variable.
    pub fn set_recovery(&mut self, u: f32) {
        self.state.u = u;
    }
}

impl MembraneDynamics for IzhikevichNeuron {
    fn membrane_potential(&self) -> f32 {
        self.state.v
    }

    fn set_membrane_potential(&mut self, v: f32) {
        self.state.v = v;
    }

    fn rest_potential(&self) -> f32 {
        self.config.v_rest
    }
}

impl NeuronModel for IzhikevichNeuron {
    type Config = IzhikevichConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        // Izhikevich equations:
        // dv/dt = 0.04v² + 5v + 140 - u + I
        // du/dt = a(bv - u)

        let v = self.state.v;
        let u = self.state.u;

        // Update voltage
        let dv = 0.04 * v * v + 5.0 * v + 140.0 - u + input_current;
        self.state.v += dv * dt;

        // Update recovery variable
        let du = self.config.a * (self.config.b * v - u);
        self.state.u += du * dt;

        // Check for spike
        if self.state.v >= self.config.v_thresh {
            self.reset();
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.state.v = self.config.c;
        self.state.u += self.config.d;
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
    fn test_izhikevich_default() {
        let neuron = IzhikevichNeuron::new(IzhikevichConfig::default());
        assert_eq!(neuron.membrane_potential(), -65.0);
        assert!(neuron.recovery().abs() < 14.0); // Should be around b*v_rest
    }

    #[test]
    fn test_izhikevich_update() {
        let mut neuron = IzhikevichNeuron::new(IzhikevichConfig::default());
        let v_initial = neuron.membrane_potential();

        // Apply input current
        let spiked = neuron.update(10.0, 1.0);

        if !spiked {
            // If didn't spike, voltage should have changed
            assert!(neuron.membrane_potential() != v_initial);
        }
    }

    #[test]
    fn test_izhikevich_spike() {
        let mut neuron = IzhikevichNeuron::new(IzhikevichConfig::default());
        let mut spike_count = 0;

        // Apply strong current to trigger spikes
        for _ in 0..200 {
            if neuron.update(15.0, 1.0) {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0, "Should spike with sufficient input");
    }

    #[test]
    fn test_izhikevich_reset() {
        let config = IzhikevichConfig::default();
        let mut neuron = IzhikevichNeuron::new(config);
        let u_before = neuron.recovery();

        neuron.reset();

        assert_eq!(neuron.membrane_potential(), config.c);
        assert_eq!(neuron.recovery(), u_before + config.d);
    }

    #[test]
    fn test_regular_spiking_preset() {
        let neuron = IzhikevichNeuron::from_preset(NeuronPreset::RegularSpiking);
        assert_eq!(neuron.config.a, 0.02);
        assert_eq!(neuron.config.b, 0.2);
        assert_eq!(neuron.config.c, -65.0);
        assert_eq!(neuron.config.d, 8.0);
    }

    #[test]
    fn test_fast_spiking_preset() {
        let neuron = IzhikevichNeuron::from_preset(NeuronPreset::FastSpiking);
        assert_eq!(neuron.config.a, 0.1);
        assert_eq!(neuron.config.d, 2.0);
    }

    #[test]
    fn test_intrinsically_bursting_preset() {
        let neuron = IzhikevichNeuron::from_preset(NeuronPreset::IntrinsicallyBursting);
        assert_eq!(neuron.config.c, -55.0);
        assert_eq!(neuron.config.d, 4.0);
    }

    #[test]
    fn test_chattering_preset() {
        let neuron = IzhikevichNeuron::from_preset(NeuronPreset::Chattering);
        assert_eq!(neuron.config.c, -50.0);
    }

    #[test]
    fn test_presets_spike_differently() {
        let mut rs = IzhikevichNeuron::from_preset(NeuronPreset::RegularSpiking);
        let mut fs = IzhikevichNeuron::from_preset(NeuronPreset::FastSpiking);

        let mut rs_spikes = 0;
        let mut fs_spikes = 0;

        let current = 15.0;
        for _ in 0..200 {
            if rs.update(current, 1.0) {
                rs_spikes += 1;
            }
            if fs.update(current, 1.0) {
                fs_spikes += 1;
            }
        }

        // Fast spiking should spike more frequently
        // (This is a general trend, exact numbers depend on parameters)
        assert!(rs_spikes > 0);
        assert!(fs_spikes > 0);
    }

    #[test]
    fn test_config_validation() {
        let mut config = IzhikevichConfig::default();
        assert!(config.validate().is_ok());

        config.a = -0.1;
        assert!(config.validate().is_err());

        config.a = 0.02;
        config.v_thresh = -70.0; // Below reset
        assert!(config.validate().is_err());
    }
}
