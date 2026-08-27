//! Core traits and types for neuron models in the DPB framework.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Membrane dynamics trait - base trait for all models with membrane potentials.
pub trait MembraneDynamics {
    /// Get the current membrane potential in millivolts (mV).
    fn membrane_potential(&self) -> f32;

    /// Set the membrane potential to a specific value.
    fn set_membrane_potential(&mut self, v: f32);

    /// Check if the neuron is at rest.
    fn is_at_rest(&self) -> bool {
        (self.membrane_potential() - self.rest_potential()).abs() < 1e-6
    }

    /// Get the resting potential in mV.
    fn rest_potential(&self) -> f32;
}

/// Neuron model trait - extends MembraneDynamics with spiking dynamics.
pub trait NeuronModel: MembraneDynamics {
    /// Configuration type for this neuron model.
    type Config: NeuronConfig;

    /// Update the neuron state with input current and return whether it spiked.
    ///
    /// # Arguments
    /// * `input_current` - Input current in nanoamperes (nA)
    /// * `dt` - Time step in milliseconds (ms)
    ///
    /// # Returns
    /// `true` if the neuron spiked, `false` otherwise
    fn update(&mut self, input_current: f32, dt: f32) -> bool;

    /// Reset the neuron to its reset state after a spike.
    fn reset(&mut self);

    /// Get the spike threshold in mV.
    fn threshold(&self) -> f32;

    /// Check if the neuron is currently in refractory period.
    fn is_refractory(&self) -> bool {
        false // Default: no refractory period
    }

    /// Get the configuration of this neuron.
    fn config(&self) -> &Self::Config;
}

/// Configuration trait for neuron models.
pub trait NeuronConfig: Clone + Send + Sync {
    /// Validate the configuration parameters.
    fn validate(&self) -> Result<(), String>;

    /// Get a default configuration.
    fn default_config() -> Self;
}

/// Generic neuron state that can be used for simple models.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct NeuronState {
    /// Membrane potential in mV
    pub v: f32,
    /// Adaptation/recovery variable
    pub u: f32,
    /// Refractory timer in ms
    pub refrac_timer: f32,
    /// Last spike time in ms
    pub last_spike_time: f32,
}

impl Default for NeuronState {
    fn default() -> Self {
        Self {
            v: -65.0,
            u: 0.0,
            refrac_timer: 0.0,
            last_spike_time: -1000.0,
        }
    }
}

impl NeuronState {
    /// Create a new neuron state with given membrane potential.
    pub fn new(v: f32) -> Self {
        Self {
            v,
            u: 0.0,
            refrac_timer: 0.0,
            last_spike_time: -1000.0,
        }
    }

    /// Reset to resting state.
    pub fn reset(&mut self, v_rest: f32) {
        self.v = v_rest;
        self.u = 0.0;
        self.refrac_timer = 0.0;
    }

    /// Check if in refractory period.
    pub fn is_refractory(&self) -> bool {
        self.refrac_timer > 0.0
    }

    /// Update refractory timer.
    pub fn update_refractory(&mut self, dt: f32) {
        if self.refrac_timer > 0.0 {
            self.refrac_timer -= dt;
            if self.refrac_timer < 0.0 {
                self.refrac_timer = 0.0;
            }
        }
    }
}

/// Spike event information.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpikeEvent {
    /// Time of the spike in ms
    pub time: f32,
    /// Neuron index that spiked
    pub neuron_id: usize,
    /// Membrane potential at spike
    pub v_spike: f32,
}

/// Synaptic input types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SynapticInput {
    /// Current-based input in nA
    Current(f32),
    /// Conductance-based excitatory input (mS, mV)
    Excitatory { g: f32, e_rev: f32 },
    /// Conductance-based inhibitory input (mS, mV)
    Inhibitory { g: f32, e_rev: f32 },
}

impl SynapticInput {
    /// Convert to effective current given membrane potential.
    pub fn to_current(self, v: f32) -> f32 {
        match self {
            SynapticInput::Current(i) => i,
            SynapticInput::Excitatory { g, e_rev } => g * (e_rev - v),
            SynapticInput::Inhibitory { g, e_rev } => g * (e_rev - v),
        }
    }
}

/// Neuron parameter presets for common configurations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NeuronPreset {
    /// Regular spiking cortical neuron
    RegularSpiking,
    /// Fast spiking interneuron
    FastSpiking,
    /// Intrinsically bursting
    IntrinsicallyBursting,
    /// Chattering
    Chattering,
    /// Low-threshold spiking
    LowThresholdSpiking,
    /// Thalamo-cortical relay
    ThalamoCortical,
    /// Resonator
    Resonator,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_state_default() {
        let state = NeuronState::default();
        assert_eq!(state.v, -65.0);
        assert_eq!(state.u, 0.0);
        assert!(!state.is_refractory());
    }

    #[test]
    fn test_neuron_state_reset() {
        let mut state = NeuronState::new(-50.0);
        state.u = 10.0;
        state.refrac_timer = 2.0;

        state.reset(-65.0);
        assert_eq!(state.v, -65.0);
        assert_eq!(state.u, 0.0);
        assert_eq!(state.refrac_timer, 0.0);
    }

    #[test]
    fn test_refractory_period() {
        let mut state = NeuronState {
            refrac_timer: 2.0,
            ..Default::default()
        };

        assert!(state.is_refractory());

        state.update_refractory(1.0);
        assert_eq!(state.refrac_timer, 1.0);
        assert!(state.is_refractory());

        state.update_refractory(2.0);
        assert_eq!(state.refrac_timer, 0.0);
        assert!(!state.is_refractory());
    }

    #[test]
    fn test_synaptic_input_current() {
        let input = SynapticInput::Current(5.0);
        assert_eq!(input.to_current(-65.0), 5.0);
    }

    #[test]
    fn test_synaptic_input_excitatory() {
        let input = SynapticInput::Excitatory { g: 1.0, e_rev: 0.0 };
        let current = input.to_current(-65.0);
        assert_eq!(current, 1.0 * (0.0 - (-65.0))); // Should be 65.0
    }

    #[test]
    fn test_synaptic_input_inhibitory() {
        let input = SynapticInput::Inhibitory { g: 1.0, e_rev: -75.0 };
        let current = input.to_current(-65.0);
        assert_eq!(current, 1.0 * (-75.0 - (-65.0))); // Should be -10.0
    }
}
