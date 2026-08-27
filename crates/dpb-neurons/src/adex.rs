//! Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
//!
//! The AdEx model combines the exponential spike mechanism of the ELIFwith spike-triggered adaptation, providing a good balance between
//! biological realism and computational efficiency.
//!
//! Equations:
//! - C_m * dv/dt = -g_L(v - E_L) + g_L*Δ_T*exp((v-V_T)/Δ_T) - w + I
//! - τ_w * dw/dt = a(v - E_L) - w
//! - If v >= V_spike: v ← V_reset, w ← w + b

use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// AdEx neuron state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct AdExState {
    /// Membrane potential in mV
    pub v: f32,
    /// Adaptation current in nA
    pub w: f32,
    /// Refractory timer in ms
    pub refrac_timer: f32,
    pub _padding: f32,
}

impl Default for AdExState {
    fn default() -> Self {
        Self {
            v: -70.0,
            w: 0.0,
            refrac_timer: 0.0,
            _padding: 0.0,
        }
    }
}

/// AdEx neuron configuration.
///
/// Default parameters based on Brette & Gerstner (2005).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct AdExConfig {
    /// Membrane capacitance in pF
    pub c_m: f32,
    /// Leak conductance in nS
    pub g_l: f32,
    /// Leak reversal potential in mV
    pub e_l: f32,
    /// Spike slope factor in mV
    pub delta_t: f32,
    /// Spike threshold in mV
    pub v_thresh: f32,
    /// Spike cut-off potential in mV
    pub v_spike: f32,
    /// Reset potential in mV
    pub v_reset: f32,
    /// Adaptation time constant in ms
    pub tau_w: f32,
    /// Subthreshold adaptation in nS
    pub a: f32,
    /// Spike-triggered adaptation in nA
    pub b: f32,
    /// Refractory period in ms
    pub tau_refrac: f32,
    pub _padding: f32,
}

impl Default for AdExConfig {
    fn default() -> Self {
        Self {
            c_m: 281.0,      // pF
            g_l: 30.0,       // nS
            e_l: -70.6,      // mV
            delta_t: 2.0,    // mV
            v_thresh: -50.4, // mV
            v_spike: 20.0,   // mV
            v_reset: -70.6,  // mV
            tau_w: 144.0,    // ms
            a: 4.0,          // nS
            b: 0.0805,       // nA
            tau_refrac: 0.0, // ms
            _padding: 0.0,
        }
    }
}

impl AdExConfig {
    /// Regular spiking configuration.
    pub fn regular_spiking() -> Self {
        Self::default()
    }

    /// Bursting configuration.
    pub fn bursting() -> Self {
        Self {
            c_m: 200.0,
            g_l: 10.0,
            e_l: -58.0,
            delta_t: 2.0,
            v_thresh: -50.0,
            v_spike: 20.0,
            v_reset: -46.0,
            tau_w: 120.0,
            a: 4.0,
            b: 0.5,
            tau_refrac: 0.0,
            _padding: 0.0,
        }
    }

    /// Adapting configuration with strong adaptation.
    pub fn adapting() -> Self {
        Self {
            c_m: 281.0,
            g_l: 30.0,
            e_l: -70.6,
            delta_t: 2.0,
            v_thresh: -50.4,
            v_spike: 20.0,
            v_reset: -70.6,
            tau_w: 144.0,
            a: 4.0,
            b: 0.5, // Increased adaptation
            tau_refrac: 0.0,
            _padding: 0.0,
        }
    }

    /// Fast spiking configuration.
    pub fn fast_spiking() -> Self {
        Self {
            c_m: 150.0,
            g_l: 30.0,
            e_l: -70.0,
            delta_t: 0.5,
            v_thresh: -50.0,
            v_spike: 20.0,
            v_reset: -70.0,
            tau_w: 10.0,
            a: 0.0,
            b: 0.0,
            tau_refrac: 2.0,
            _padding: 0.0,
        }
    }
}

impl NeuronConfig for AdExConfig {
    fn validate(&self) -> Result<(), String> {
        if self.c_m <= 0.0 {
            return Err("Capacitance must be positive".to_string());
        }
        if self.g_l <= 0.0 {
            return Err("Leak conductance must be positive".to_string());
        }
        if self.delta_t <= 0.0 {
            return Err("Spike slope factor must be positive".to_string());
        }
        if self.tau_w <= 0.0 {
            return Err("Adaptation time constant must be positive".to_string());
        }
        if self.v_spike <= self.v_thresh {
            return Err("Spike potential must be greater than threshold".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Adaptive Exponential Integrate-and-Fire neuron.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdExNeuron {
    pub state: AdExState,
    pub config: AdExConfig,
}

impl AdExNeuron {
    /// Create a new AdEx neuron with the given configuration.
    pub fn new(config: AdExConfig) -> Self {
        Self {
            state: AdExState {
                v: config.e_l,
                w: 0.0,
                refrac_timer: 0.0,
                _padding: 0.0,
            },
            config,
        }
    }

    /// Get the adaptation current.
    pub fn adaptation(&self) -> f32 {
        self.state.w
    }

    /// Set the adaptation current.
    pub fn set_adaptation(&mut self, w: f32) {
        self.state.w = w;
    }
}

impl MembraneDynamics for AdExNeuron {
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

impl NeuronModel for AdExNeuron {
    type Config = AdExConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        // Update refractory period
        if self.state.refrac_timer > 0.0 {
            self.state.refrac_timer -= dt;
            if self.state.refrac_timer < 0.0 {
                self.state.refrac_timer = 0.0;
            }
            return false;
        }

        // AdEx membrane equation:
        // C_m * dv/dt = -g_L(v - E_L) + g_L*Δ_T*exp((v-V_T)/Δ_T) - w + I

        let v = self.state.v;
        let w = self.state.w;

        // Leak current
        let i_leak = -self.config.g_l * (v - self.config.e_l);

        // Exponential spike current
        let exp_arg = (v - self.config.v_thresh) / self.config.delta_t;
        let i_exp = if exp_arg > 10.0 {
            // Prevent overflow
            self.config.g_l * self.config.delta_t * 22026.0 // exp(10)
        } else {
            self.config.g_l * self.config.delta_t * exp_arg.exp()
        };

        // Total membrane current
        let dv = (i_leak + i_exp - w + input_current) / self.config.c_m;
        self.state.v += dv * dt;

        // Adaptation current dynamics:
        // τ_w * dw/dt = a(v - E_L) - w
        let dw = (self.config.a * (v - self.config.e_l) - w) / self.config.tau_w;
        self.state.w += dw * dt;

        // Check for spike
        if self.state.v >= self.config.v_spike {
            self.reset();
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.state.v = self.config.v_reset;
        self.state.w += self.config.b; // Spike-triggered adaptation
        self.state.refrac_timer = self.config.tau_refrac;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh
    }

    fn is_refractory(&self) -> bool {
        self.state.refrac_timer > 0.0
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adex_default() {
        let neuron = AdExNeuron::new(AdExConfig::default());
        assert_eq!(neuron.membrane_potential(), neuron.config.e_l);
        assert_eq!(neuron.adaptation(), 0.0);
    }

    #[test]
    fn test_adex_update() {
        let mut neuron = AdExNeuron::new(AdExConfig::default());
        let v_initial = neuron.membrane_potential();

        // Apply input current
        neuron.update(100.0, 1.0);

        // Voltage should depolarize
        assert!(neuron.membrane_potential() > v_initial);
    }

    #[test]
    fn test_adex_spike() {
        let mut neuron = AdExNeuron::new(AdExConfig::default());
        let mut spike_count = 0;

        // Apply strong current (needs to exceed rheobase ~600 pA for default params)
        for _ in 0..1000 {
            if neuron.update(800.0, 0.1) {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0, "Neuron should spike with sufficient input");
    }

    #[test]
    fn test_adex_adaptation() {
        let mut neuron = AdExNeuron::new(AdExConfig::default());
        let w_initial = neuron.adaptation();

        // Trigger spike (needs to exceed rheobase ~600 pA)
        for _ in 0..1000 {
            neuron.update(800.0, 0.1);
        }

        // Adaptation should have increased
        assert!(neuron.adaptation() > w_initial);
    }

    #[test]
    fn test_adex_reset() {
        let config = AdExConfig::default();
        let mut neuron = AdExNeuron::new(config);
        let w_before = neuron.adaptation();

        neuron.reset();

        assert_eq!(neuron.membrane_potential(), config.v_reset);
        assert_eq!(neuron.adaptation(), w_before + config.b);
    }

    #[test]
    fn test_adex_presets() {
        let rs = AdExNeuron::new(AdExConfig::regular_spiking());
        let fs = AdExNeuron::new(AdExConfig::fast_spiking());
        let burst = AdExNeuron::new(AdExConfig::bursting());

        assert_eq!(rs.membrane_potential(), rs.config.e_l);
        assert_eq!(fs.membrane_potential(), fs.config.e_l);
        assert_eq!(burst.membrane_potential(), burst.config.e_l);

        // Fast spiking has no adaptation
        assert_eq!(fs.config.a, 0.0);
        assert_eq!(fs.config.b, 0.0);
    }

    #[test]
    fn test_adex_bursting() {
        let mut neuron = AdExNeuron::new(AdExConfig::bursting());
        let mut spikes = Vec::new();

        // Record spike times
        for t in 0..1000 {
            if neuron.update(300.0, 0.1) {
                spikes.push(t);
            }
        }

        assert!(!spikes.is_empty(), "Bursting neuron should spike");
    }

    #[test]
    fn test_config_validation() {
        let mut config = AdExConfig::default();
        assert!(config.validate().is_ok());

        config.c_m = 0.0;
        assert!(config.validate().is_err());

        config.c_m = 281.0;
        config.g_l = -10.0;
        assert!(config.validate().is_err());

        config.g_l = 30.0;
        config.delta_t = 0.0;
        assert!(config.validate().is_err());

        config.delta_t = 2.0;
        config.v_spike = -60.0; // Below threshold
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_exponential_overflow_protection() {
        let mut neuron = AdExNeuron::new(AdExConfig::default());

        // Set voltage way above threshold to test overflow protection
        neuron.set_membrane_potential(100.0);

        // Should not panic
        neuron.update(0.0, 1.0);
    }
}
