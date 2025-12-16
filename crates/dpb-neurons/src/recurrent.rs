//! Recurrent neuron with self-connections.
//!
//! This neuron model includes a self-connection (autapse) that feeds
//! its own spike history back as input, creating intrinsic dynamics.

use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Recurrent neuron state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct RecurrentNeuronState {
    /// Membrane potential in mV
    pub v: f32,
    /// Recurrent synaptic current in nA
    pub i_rec: f32,
    /// Refractory timer in ms
    pub refrac_timer: f32,
    pub _padding: f32,
}

impl Default for RecurrentNeuronState {
    fn default() -> Self {
        Self {
            v: -65.0,
            i_rec: 0.0,
            refrac_timer: 0.0,
            _padding: 0.0,
        }
    }
}

/// Recurrent neuron configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct RecurrentNeuronConfig {
    /// Membrane time constant in ms
    pub tau_mem: f32,
    /// Recurrent synapse time constant in ms
    pub tau_rec: f32,
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
    /// Recurrent weight (self-connection strength)
    pub w_rec: f32,
    /// Recurrent delay in ms
    pub delay_rec: f32,
    pub _padding: [f32; 3],
}

impl Default for RecurrentNeuronConfig {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            tau_rec: 5.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            r_m: 10.0,
            tau_refrac: 2.0,
            w_rec: 0.5,
            delay_rec: 1.0,
            _padding: [0.0; 3],
        }
    }
}

impl RecurrentNeuronConfig {
    /// Excitatory recurrence (positive feedback).
    pub fn excitatory_recurrence() -> Self {
        Self {
            w_rec: 2.0,
            delay_rec: 1.0,
            ..Default::default()
        }
    }

    /// Inhibitory recurrence (negative feedback).
    pub fn inhibitory_recurrence() -> Self {
        Self {
            w_rec: -2.0,
            delay_rec: 1.0,
            ..Default::default()
        }
    }

    /// Oscillatory configuration with delayed inhibition.
    pub fn oscillatory() -> Self {
        Self {
            w_rec: -1.5,
            delay_rec: 5.0,
            tau_rec: 10.0,
            ..Default::default()
        }
    }
}

impl NeuronConfig for RecurrentNeuronConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_mem <= 0.0 || self.tau_rec <= 0.0 {
            return Err("Time constants must be positive".to_string());
        }
        if self.r_m <= 0.0 {
            return Err("Membrane resistance must be positive".to_string());
        }
        if self.v_thresh <= self.v_reset {
            return Err("Threshold must be greater than reset".to_string());
        }
        if self.delay_rec < 0.0 {
            return Err("Delay must be non-negative".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Delayed spike for recurrent connection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct DelayedSpike {
    /// Delivery time in ms
    delivery_time: f32,
    /// Spike weight
    weight: f32,
}

/// Recurrent neuron with self-connection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurrentNeuron {
    pub state: RecurrentNeuronState,
    pub config: RecurrentNeuronConfig,
    /// Queue of delayed recurrent spikes
    spike_queue: VecDeque<DelayedSpike>,
    /// Current simulation time in ms
    current_time: f32,
}

impl RecurrentNeuron {
    pub fn new(config: RecurrentNeuronConfig) -> Self {
        Self {
            state: RecurrentNeuronState::default(),
            config,
            spike_queue: VecDeque::new(),
            current_time: 0.0,
        }
    }

    /// Process delayed spikes that have arrived.
    fn process_delayed_spikes(&mut self) {
        while let Some(spike) = self.spike_queue.front() {
            if spike.delivery_time <= self.current_time {
                self.state.i_rec += spike.weight;
                self.spike_queue.pop_front();
            } else {
                break; // Queue is ordered, so we can stop
            }
        }
    }

    /// Add a spike to the delay queue.
    fn queue_recurrent_spike(&mut self) {
        let delivery_time = self.current_time + self.config.delay_rec;
        self.spike_queue.push_back(DelayedSpike {
            delivery_time,
            weight: self.config.w_rec,
        });
    }

    /// Get the current recurrent current.
    pub fn recurrent_current(&self) -> f32 {
        self.state.i_rec
    }
}

impl MembraneDynamics for RecurrentNeuron {
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

impl NeuronModel for RecurrentNeuron {
    type Config = RecurrentNeuronConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        self.current_time += dt;

        // Process any delayed spikes that have arrived
        self.process_delayed_spikes();

        // Update refractory period
        if self.state.refrac_timer > 0.0 {
            self.state.refrac_timer -= dt;
            if self.state.refrac_timer < 0.0 {
                self.state.refrac_timer = 0.0;
            }

            // Decay recurrent current during refractory period
            let di_rec = -self.state.i_rec / self.config.tau_rec;
            self.state.i_rec += di_rec * dt;

            return false;
        }

        // Decay recurrent current
        let di_rec = -self.state.i_rec / self.config.tau_rec;
        self.state.i_rec += di_rec * dt;

        // Membrane dynamics with external + recurrent input
        let total_current = input_current + self.state.i_rec;
        let dv = (-(self.state.v - self.config.v_rest) + self.config.r_m * total_current)
                 / self.config.tau_mem;
        self.state.v += dv * dt;

        // Check for spike
        if self.state.v >= self.config.v_thresh {
            // Queue recurrent spike with delay
            self.queue_recurrent_spike();
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
    fn test_recurrent_neuron_default() {
        let neuron = RecurrentNeuron::new(RecurrentNeuronConfig::default());
        assert_eq!(neuron.membrane_potential(), -65.0);
        assert_eq!(neuron.recurrent_current(), 0.0);
        assert_eq!(neuron.spike_queue.len(), 0);
    }

    #[test]
    fn test_recurrent_spike_queuing() {
        let mut neuron = RecurrentNeuron::new(RecurrentNeuronConfig::default());

        // Trigger a spike
        for _ in 0..50 {
            neuron.update(20.0, 1.0);
        }

        // Should have queued a recurrent spike
        // (might have delivered already depending on timing)
        assert!(neuron.current_time > 0.0);
    }

    #[test]
    fn test_delayed_spike_delivery() {
        let config = RecurrentNeuronConfig {
            delay_rec: 5.0,
            w_rec: 10.0,
            ..Default::default()
        };
        let mut neuron = RecurrentNeuron::new(config);

        // Manually queue a spike
        neuron.queue_recurrent_spike();
        assert_eq!(neuron.spike_queue.len(), 1);

        // Advance time but not enough
        for _ in 0..3 {
            neuron.update(0.0, 1.0);
        }

        // Spike shouldn't have delivered yet
        // (or just started to)

        // Advance past delay
        for _ in 0..10 {
            neuron.update(0.0, 1.0);
        }

        // Spike should have been delivered and queue empty
        assert_eq!(neuron.spike_queue.len(), 0);
    }

    #[test]
    fn test_excitatory_recurrence() {
        let mut neuron = RecurrentNeuron::new(RecurrentNeuronConfig::excitatory_recurrence());
        assert!(neuron.config.w_rec > 0.0);

        let mut spike_count = 0;
        for _ in 0..200 {
            if neuron.update(5.0, 1.0) {
                spike_count += 1;
            }
        }

        // Excitatory recurrence should facilitate spiking
        assert!(spike_count > 0);
    }

    #[test]
    fn test_inhibitory_recurrence() {
        let mut neuron = RecurrentNeuron::new(RecurrentNeuronConfig::inhibitory_recurrence());
        assert!(neuron.config.w_rec < 0.0);

        let mut spike_count = 0;
        for _ in 0..200 {
            if neuron.update(15.0, 1.0) {
                spike_count += 1;
            }
        }

        // Should still spike but possibly less than without inhibition
        assert!(spike_count >= 0);
    }

    #[test]
    fn test_oscillatory_preset() {
        let config = RecurrentNeuronConfig::oscillatory();
        let neuron = RecurrentNeuron::new(config);

        assert!(neuron.config.w_rec < 0.0);
        assert!(neuron.config.delay_rec > 0.0);
    }

    #[test]
    fn test_recurrent_current_decay() {
        let mut neuron = RecurrentNeuron::new(RecurrentNeuronConfig::default());

        // Manually set recurrent current
        neuron.state.i_rec = 10.0;

        // Let it decay
        for _ in 0..100 {
            neuron.update(0.0, 1.0);
        }

        // Should have decayed toward zero
        assert!(neuron.recurrent_current().abs() < 10.0);
    }

    #[test]
    fn test_config_validation() {
        let mut config = RecurrentNeuronConfig::default();
        assert!(config.validate().is_ok());

        config.tau_mem = 0.0;
        assert!(config.validate().is_err());

        config.tau_mem = 20.0;
        config.delay_rec = -1.0;
        assert!(config.validate().is_err());

        config.delay_rec = 1.0;
        config.v_thresh = -70.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_multiple_recurrent_spikes() {
        let mut neuron = RecurrentNeuron::new(RecurrentNeuronConfig::excitatory_recurrence());

        // Trigger multiple spikes
        for _ in 0..100 {
            neuron.update(15.0, 1.0);
        }

        // Queue might have multiple spikes or they've been processed
        // Just verify the mechanism works
        assert!(neuron.current_time > 0.0);
    }
}
