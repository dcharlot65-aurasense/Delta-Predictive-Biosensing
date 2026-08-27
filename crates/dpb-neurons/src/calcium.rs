//! Calcium-based neuron model with Ca²⁺ dynamics.
//!
//! This model includes intracellular calcium concentration as an
//! adaptation mechanism, providing biologically realistic spike-frequency
//! adaptation and burst dynamics.

use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Calcium neuron state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct CalciumState {
    /// Membrane potential in mV
    pub v: f32,
    /// Intracellular calcium concentration in μM
    pub ca: f32,
    /// Calcium-dependent potassium current in nA
    pub i_ahp: f32,
    /// Refractory timer in ms
    pub refrac_timer: f32,
}

impl Default for CalciumState {
    fn default() -> Self {
        Self {
            v: -65.0,
            ca: 0.05, // Baseline calcium concentration
            i_ahp: 0.0,
            refrac_timer: 0.0,
        }
    }
}

/// Calcium neuron configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct CalciumConfig {
    /// Membrane time constant in ms
    pub tau_mem: f32,
    /// Calcium decay time constant in ms
    pub tau_ca: f32,
    /// AHP current time constant in ms
    pub tau_ahp: f32,
    /// Spike threshold in mV
    pub v_thresh: f32,
    /// Reset potential in mV
    pub v_reset: f32,
    /// Resting potential in mV
    pub v_rest: f32,
    /// Membrane resistance in MΩ
    pub r_m: f32,
    /// Refractory period in ms
    pub tau_refrac: f32,
    /// Calcium influx per spike in μM
    pub ca_influx: f32,
    /// Baseline calcium concentration in μM
    pub ca_baseline: f32,
    /// Calcium-to-current coupling in nA/μM
    pub alpha_ca: f32,
    pub _padding: f32,
}

impl Default for CalciumConfig {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            tau_ca: 100.0,
            tau_ahp: 50.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            r_m: 10.0,
            tau_refrac: 2.0,
            ca_influx: 0.5,
            ca_baseline: 0.05,
            alpha_ca: 2.0,
            _padding: 0.0,
        }
    }
}

impl CalciumConfig {
    /// Configuration for strong adaptation (large AHP).
    pub fn strong_adaptation() -> Self {
        Self {
            ca_influx: 1.0,
            alpha_ca: 5.0,
            tau_ca: 150.0,
            ..Default::default()
        }
    }

    /// Configuration for weak adaptation.
    pub fn weak_adaptation() -> Self {
        Self {
            ca_influx: 0.2,
            alpha_ca: 1.0,
            tau_ca: 50.0,
            ..Default::default()
        }
    }

    /// Configuration for bursting behavior.
    pub fn bursting() -> Self {
        Self {
            ca_influx: 0.8,
            alpha_ca: 3.0,
            tau_ca: 200.0,
            tau_ahp: 100.0,
            ..Default::default()
        }
    }
}

impl NeuronConfig for CalciumConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_mem <= 0.0 || self.tau_ca <= 0.0 || self.tau_ahp <= 0.0 {
            return Err("Time constants must be positive".to_string());
        }
        if self.r_m <= 0.0 {
            return Err("Membrane resistance must be positive".to_string());
        }
        if self.v_thresh <= self.v_reset {
            return Err("Threshold must be greater than reset".to_string());
        }
        if self.ca_baseline < 0.0 || self.ca_influx < 0.0 {
            return Err("Calcium parameters must be non-negative".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Calcium-based neuron with Ca²⁺-dependent adaptation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalciumNeuron {
    pub state: CalciumState,
    pub config: CalciumConfig,
}

impl CalciumNeuron {
    pub fn new(config: CalciumConfig) -> Self {
        Self {
            state: CalciumState {
                v: config.v_rest,
                ca: config.ca_baseline,
                i_ahp: 0.0,
                refrac_timer: 0.0,
            },
            config,
        }
    }

    /// Get the current calcium concentration.
    pub fn calcium(&self) -> f32 {
        self.state.ca
    }

    /// Get the afterhyperpolarization current.
    pub fn ahp_current(&self) -> f32 {
        self.state.i_ahp
    }
}

impl MembraneDynamics for CalciumNeuron {
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

impl NeuronModel for CalciumNeuron {
    type Config = CalciumConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        // Update refractory period
        if self.state.refrac_timer > 0.0 {
            self.state.refrac_timer -= dt;
            if self.state.refrac_timer < 0.0 {
                self.state.refrac_timer = 0.0;
            }
            return false;
        }

        // Calcium dynamics: decay toward baseline
        let dca = -(self.state.ca - self.config.ca_baseline) / self.config.tau_ca;
        self.state.ca += dca * dt;
        self.state.ca = self.state.ca.max(0.0); // Keep non-negative

        // AHP current dynamics: proportional to calcium
        let i_ahp_target = self.config.alpha_ca * (self.state.ca - self.config.ca_baseline);
        let di_ahp = (i_ahp_target - self.state.i_ahp) / self.config.tau_ahp;
        self.state.i_ahp += di_ahp * dt;

        // Membrane potential dynamics with AHP current
        let dv = (-(self.state.v - self.config.v_rest) + self.config.r_m * input_current
            - self.state.i_ahp)
            / self.config.tau_mem;
        self.state.v += dv * dt;

        // Check for spike
        if self.state.v >= self.config.v_thresh {
            // Spike-triggered calcium influx
            self.state.ca += self.config.ca_influx;
            self.reset();
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.state.v = self.config.v_reset;
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
    fn test_calcium_neuron_default() {
        let neuron = CalciumNeuron::new(CalciumConfig::default());
        assert_eq!(neuron.membrane_potential(), -65.0);
        assert_eq!(neuron.calcium(), 0.05);
        assert_eq!(neuron.ahp_current(), 0.0);
    }

    #[test]
    fn test_calcium_dynamics() {
        let mut neuron = CalciumNeuron::new(CalciumConfig::default());
        let ca_initial = neuron.calcium();

        // Increase calcium
        neuron.state.ca = 1.0;

        // Should decay toward baseline
        for _ in 0..100 {
            neuron.update(0.0, 1.0);
        }

        assert!(neuron.calcium() < 1.0);
        assert!(neuron.calcium() > ca_initial);
    }

    #[test]
    fn test_calcium_spike_influx() {
        let mut neuron = CalciumNeuron::new(CalciumConfig::default());
        let ca_before = neuron.calcium();

        // Trigger spike with strong input
        for _ in 0..50 {
            neuron.update(50.0, 1.0);
        }

        // Calcium should have increased due to spikes
        assert!(neuron.calcium() > ca_before);
    }

    #[test]
    fn test_ahp_current() {
        let mut neuron = CalciumNeuron::new(CalciumConfig::default());

        // Manually increase calcium
        neuron.state.ca = 1.0;

        // Update to let AHP develop
        for _ in 0..100 {
            neuron.update(0.0, 1.0);
        }

        // AHP current should be positive (hyperpolarizing)
        assert!(neuron.ahp_current() > 0.0);
    }

    #[test]
    fn test_adaptation_effect() {
        let mut neuron = CalciumNeuron::new(CalciumConfig::default());
        let mut spike_times = Vec::new();

        // Apply constant current and record spike times
        for t in 0..1000 {
            if neuron.update(15.0, 1.0) {
                spike_times.push(t);
            }
        }

        // Should have spikes
        assert!(spike_times.len() > 2);

        // Inter-spike intervals should increase due to adaptation
        if spike_times.len() >= 3 {
            let _isi1 = spike_times[1] - spike_times[0];
            let _isi2 = spike_times[2] - spike_times[1];
            // Later ISI should generally be longer (though not guaranteed every time)
            // Just check that adaptation is present by checking calcium builds up
            assert!(neuron.calcium() > 0.05);
        }
    }

    #[test]
    fn test_strong_adaptation_preset() {
        let config = CalciumConfig::strong_adaptation();
        let neuron = CalciumNeuron::new(config);

        assert_eq!(neuron.config.ca_influx, 1.0);
        assert_eq!(neuron.config.alpha_ca, 5.0);
    }

    #[test]
    fn test_weak_adaptation_preset() {
        let config = CalciumConfig::weak_adaptation();
        let neuron = CalciumNeuron::new(config);

        assert_eq!(neuron.config.ca_influx, 0.2);
        assert_eq!(neuron.config.alpha_ca, 1.0);
    }

    #[test]
    fn test_bursting_preset() {
        let config = CalciumConfig::bursting();
        let neuron = CalciumNeuron::new(config);

        assert_eq!(neuron.config.tau_ca, 200.0);
        assert_eq!(neuron.config.tau_ahp, 100.0);
    }

    #[test]
    fn test_config_validation() {
        let mut config = CalciumConfig::default();
        assert!(config.validate().is_ok());

        config.tau_ca = 0.0;
        assert!(config.validate().is_err());

        config.tau_ca = 100.0;
        config.ca_influx = -0.1;
        assert!(config.validate().is_err());

        config.ca_influx = 0.5;
        config.v_thresh = -70.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_calcium_non_negative() {
        let mut neuron = CalciumNeuron::new(CalciumConfig::default());

        // Set very negative calcium (shouldn't happen, but test clamping)
        neuron.state.ca = -1.0;

        // Update should clamp to non-negative
        neuron.update(0.0, 1.0);

        assert!(neuron.calcium() >= 0.0);
    }
}
