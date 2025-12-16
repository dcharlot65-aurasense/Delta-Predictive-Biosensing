//! Spike Response Model (SRM) neuron.
//!
//! The SRM is a kernel-based neuron model where the membrane potential
//! is computed as a convolution of input spikes with synaptic and
//! refractory kernels.

use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Spike time record for kernel convolution.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpikeTime {
    /// Time of the spike in ms
    pub time: f32,
    /// Weight/amplitude
    pub weight: f32,
}

/// SRM neuron state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SrmState {
    /// Current membrane potential in mV
    pub v: f32,
    /// History of input spike times
    pub spike_history: VecDeque<SpikeTime>,
    /// Time of last output spike
    pub last_spike_time: f32,
    /// Current time in ms
    pub current_time: f32,
}

impl Default for SrmState {
    fn default() -> Self {
        Self {
            v: -65.0,
            spike_history: VecDeque::new(),
            last_spike_time: -1000.0,
            current_time: 0.0,
        }
    }
}

/// SRM neuron configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SrmConfig {
    /// Spike threshold in mV
    pub v_thresh: f32,
    /// Reset potential in mV
    pub v_reset: f32,
    /// Resting potential in mV
    pub v_rest: f32,
    /// Synaptic time constant in ms
    pub tau_syn: f32,
    /// Membrane time constant in ms
    pub tau_mem: f32,
    /// Refractory time constant in ms
    pub tau_refrac: f32,
    /// Refractory amplitude in mV
    pub eta_0: f32,
    /// Max spike history length
    pub max_history: usize,
}

impl Default for SrmConfig {
    fn default() -> Self {
        Self {
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            tau_syn: 5.0,
            tau_mem: 20.0,
            tau_refrac: 2.0,
            eta_0: -10.0,
            max_history: 100,
        }
    }
}

impl NeuronConfig for SrmConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_syn <= 0.0 || self.tau_mem <= 0.0 || self.tau_refrac <= 0.0 {
            return Err("Time constants must be positive".to_string());
        }
        if self.v_thresh <= self.v_reset {
            return Err("Threshold must be greater than reset".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Spike Response Model neuron.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SrmNeuron {
    pub state: SrmState,
    pub config: SrmConfig,
}

impl SrmNeuron {
    pub fn new(config: SrmConfig) -> Self {
        Self {
            state: SrmState::default(),
            config,
        }
    }

    /// Synaptic kernel (epsilon): double exponential.
    fn epsilon_kernel(&self, t: f32) -> f32 {
        if t <= 0.0 {
            return 0.0;
        }
        let tau_syn = self.config.tau_syn;
        let tau_mem = self.config.tau_mem;

        if (tau_syn - tau_mem).abs() < 1e-6 {
            // Limit case when time constants are equal
            (t / tau_syn) * (-t / tau_syn).exp()
        } else {
            let norm = 1.0 / (tau_syn - tau_mem);
            norm * ((-t / tau_mem).exp() - (-t / tau_syn).exp())
        }
    }

    /// Refractory kernel (eta): exponential decay.
    fn eta_kernel(&self, t: f32) -> f32 {
        if t <= 0.0 {
            return 0.0;
        }
        self.config.eta_0 * (-t / self.config.tau_refrac).exp()
    }

    /// Compute membrane potential from kernels.
    fn compute_potential(&self) -> f32 {
        let mut v = self.config.v_rest;

        // Add contribution from input spikes
        for spike in &self.state.spike_history {
            let dt = self.state.current_time - spike.time;
            v += spike.weight * self.epsilon_kernel(dt);
        }

        // Subtract refractory effect from last output spike
        let dt_refrac = self.state.current_time - self.state.last_spike_time;
        v += self.eta_kernel(dt_refrac);

        v
    }

    /// Add input spike to history.
    pub fn add_input_spike(&mut self, weight: f32) {
        self.state.spike_history.push_back(SpikeTime {
            time: self.state.current_time,
            weight,
        });

        // Trim old spikes
        while self.state.spike_history.len() > self.config.max_history {
            self.state.spike_history.pop_front();
        }
    }

    /// Clean up old spike history.
    fn prune_history(&mut self) {
        // Remove spikes older than 5*tau_mem (negligible contribution)
        let cutoff_time = self.state.current_time - 5.0 * self.config.tau_mem;
        while let Some(spike) = self.state.spike_history.front() {
            if spike.time < cutoff_time {
                self.state.spike_history.pop_front();
            } else {
                break;
            }
        }
    }
}

impl MembraneDynamics for SrmNeuron {
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

impl NeuronModel for SrmNeuron {
    type Config = SrmConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        self.state.current_time += dt;

        // Convert input current to spike-like input (simple approximation)
        if input_current > 0.0 {
            self.add_input_spike(input_current * dt);
        }

        // Compute potential from kernels
        self.state.v = self.compute_potential();

        // Prune old spikes periodically
        if self.state.spike_history.len() > self.config.max_history / 2 {
            self.prune_history();
        }

        // Check for spike
        if self.state.v >= self.config.v_thresh {
            self.reset();
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.state.last_spike_time = self.state.current_time;
        self.state.v = self.config.v_reset;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh
    }

    fn is_refractory(&self) -> bool {
        let dt = self.state.current_time - self.state.last_spike_time;
        dt < self.config.tau_refrac
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srm_default() {
        let neuron = SrmNeuron::new(SrmConfig::default());
        assert_eq!(neuron.membrane_potential(), -65.0);
        assert_eq!(neuron.state.spike_history.len(), 0);
    }

    #[test]
    fn test_srm_add_spike() {
        let mut neuron = SrmNeuron::new(SrmConfig::default());
        neuron.add_input_spike(10.0);

        assert_eq!(neuron.state.spike_history.len(), 1);
        assert_eq!(neuron.state.spike_history[0].weight, 10.0);
    }

    #[test]
    fn test_srm_epsilon_kernel() {
        let neuron = SrmNeuron::new(SrmConfig::default());

        // At t=0, should be 0
        assert_eq!(neuron.epsilon_kernel(0.0), 0.0);

        // At negative t, should be 0
        assert_eq!(neuron.epsilon_kernel(-1.0), 0.0);

        // At positive t, should be positive
        assert!(neuron.epsilon_kernel(5.0) > 0.0);
    }

    #[test]
    fn test_srm_eta_kernel() {
        let neuron = SrmNeuron::new(SrmConfig::default());

        // At t=0, should be eta_0
        let eta_0_val = neuron.eta_kernel(0.0);
        assert_eq!(eta_0_val, 0.0); // At exactly 0

        // At positive t, should decay
        let eta_1 = neuron.eta_kernel(1.0);
        let eta_2 = neuron.eta_kernel(2.0);
        assert!(eta_1.abs() > eta_2.abs());
    }

    #[test]
    fn test_srm_update() {
        let mut neuron = SrmNeuron::new(SrmConfig::default());
        let v_initial = neuron.membrane_potential();

        neuron.update(50.0, 1.0);

        // Potential should change
        assert!(neuron.membrane_potential() != v_initial);
    }

    #[test]
    fn test_srm_spike() {
        let mut neuron = SrmNeuron::new(SrmConfig::default());
        let mut spike_count = 0;

        for _ in 0..500 {
            if neuron.update(100.0, 0.1) {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0);
    }

    #[test]
    fn test_srm_history_pruning() {
        let config = SrmConfig {
            max_history: 10,
            ..Default::default()
        };
        let mut neuron = SrmNeuron::new(config);

        // Add many spikes
        for _ in 0..20 {
            neuron.add_input_spike(1.0);
            neuron.state.current_time += 1.0;
        }

        // Should be limited by max_history
        assert!(neuron.state.spike_history.len() <= 10);
    }

    #[test]
    fn test_config_validation() {
        let mut config = SrmConfig::default();
        assert!(config.validate().is_ok());

        config.tau_syn = 0.0;
        assert!(config.validate().is_err());

        config.tau_syn = 5.0;
        config.v_thresh = -70.0;
        assert!(config.validate().is_err());
    }
}
