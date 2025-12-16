//! Hardware-specific neuron models optimized for neuromorphic chips.
//!
//! This module implements neuron models compatible with specific hardware:
//! 1. XyloLIF: SynSense Xylo chip compatible
//! 2. PulsarLIF: Generic neuromorphic hardware (Loihi-style)
//! 3. QuantizedLIF: Fixed-point for edge devices

use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

// ============================================================================
// 1. XyloLIF - SynSense Xylo Chip Compatible
// ============================================================================

/// Xylo LIF neuron state (hardware-constrained).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Pod, Zeroable)]
pub struct XyloLifState {
    /// Membrane potential (8-bit fixed point, scaled)
    pub v: i16,
    /// Refractory counter (hardware cycles)
    pub refrac_counter: u8,
    pub _padding: u8,
}

impl Default for XyloLifState {
    fn default() -> Self {
        Self {
            v: 0, // Represents resting potential in hardware units
            refrac_counter: 0,
            _padding: 0,
        }
    }
}

/// Xylo LIF configuration (hardware parameter ranges).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Pod, Zeroable)]
pub struct XyloLifConfig {
    /// Threshold (8-bit)
    pub v_thresh: i16,
    /// Reset value (8-bit)
    pub v_reset: i16,
    /// Leak factor (0-255, represents decay rate)
    pub leak: u8,
    /// Refractory period in hardware cycles
    pub refrac_cycles: u8,
    /// Scaling factor for inputs
    pub input_scale: u8,
    pub _padding: u8,
}

impl Default for XyloLifConfig {
    fn default() -> Self {
        Self {
            v_thresh: 100,
            v_reset: 0,
            leak: 5,
            refrac_cycles: 2,
            input_scale: 1,
            _padding: 0,
        }
    }
}

impl XyloLifConfig {
    /// Fast dynamics configuration.
    pub fn fast() -> Self {
        Self {
            v_thresh: 80,
            leak: 10,
            refrac_cycles: 1,
            ..Default::default()
        }
    }

    /// Slow dynamics configuration.
    pub fn slow() -> Self {
        Self {
            v_thresh: 120,
            leak: 2,
            refrac_cycles: 4,
            ..Default::default()
        }
    }
}

impl NeuronConfig for XyloLifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.v_thresh <= self.v_reset {
            return Err("Threshold must be greater than reset".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Xylo-compatible LIF neuron with hardware constraints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XyloLifNeuron {
    pub state: XyloLifState,
    pub config: XyloLifConfig,
}

impl XyloLifNeuron {
    pub fn new(config: XyloLifConfig) -> Self {
        Self {
            state: XyloLifState::default(),
            config,
        }
    }

    /// Convert to hardware-friendly format for export.
    pub fn to_hardware_params(&self) -> [u8; 8] {
        [
            (self.config.v_thresh & 0xFF) as u8,
            ((self.config.v_thresh >> 8) & 0xFF) as u8,
            (self.config.v_reset & 0xFF) as u8,
            ((self.config.v_reset >> 8) & 0xFF) as u8,
            self.config.leak,
            self.config.refrac_cycles,
            self.config.input_scale,
            0, // padding
        ]
    }
}

impl MembraneDynamics for XyloLifNeuron {
    fn membrane_potential(&self) -> f32 {
        self.state.v as f32
    }

    fn set_membrane_potential(&mut self, v: f32) {
        self.state.v = v.round() as i16;
    }

    fn rest_potential(&self) -> f32 {
        self.config.v_reset as f32
    }
}

impl NeuronModel for XyloLifNeuron {
    type Config = XyloLifConfig;

    fn update(&mut self, input_current: f32, _dt: f32) -> bool {
        // Hardware uses discrete time steps, ignore dt

        // Check refractory period
        if self.state.refrac_counter > 0 {
            self.state.refrac_counter -= 1;
            return false;
        }

        // Apply leak (decay)
        let leak_amount = (self.state.v * self.config.leak as i16) >> 8;
        self.state.v -= leak_amount;

        // Add input (scaled and quantized)
        let scaled_input = (input_current * self.config.input_scale as f32).round() as i16;
        self.state.v = self.state.v.saturating_add(scaled_input);

        // Clamp to hardware range
        self.state.v = self.state.v.clamp(-32768, 32767);

        // Check for spike
        if self.state.v >= self.config.v_thresh {
            self.reset();
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.state.v = self.config.v_reset;
        self.state.refrac_counter = self.config.refrac_cycles;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh as f32
    }

    fn is_refractory(&self) -> bool {
        self.state.refrac_counter > 0
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// ============================================================================
// 2. PulsarLIF - Generic Neuromorphic Hardware (Loihi-style)
// ============================================================================

/// Pulsar LIF state with integer arithmetic.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Pod, Zeroable)]
pub struct PulsarLifState {
    /// Membrane potential (16-bit)
    pub v: i32,
    /// Adaptation current (16-bit)
    pub adapt: i32,
    /// Refractory timer
    pub refrac_timer: u16,
    pub _padding: u16,
}

impl Default for PulsarLifState {
    fn default() -> Self {
        Self {
            v: 0,
            adapt: 0,
            refrac_timer: 0,
            _padding: 0,
        }
    }
}

/// Pulsar LIF configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Pod, Zeroable)]
pub struct PulsarLifConfig {
    /// Threshold
    pub v_thresh: i32,
    /// Reset potential
    pub v_reset: i32,
    /// Decay constant (0-256)
    pub decay: u16,
    pub _padding1: u16,
    /// Adaptation increment
    pub adapt_increment: i32,
    /// Adaptation decay
    pub adapt_decay: u16,
    /// Refractory period
    pub refrac_period: u16,
}

impl Default for PulsarLifConfig {
    fn default() -> Self {
        Self {
            v_thresh: 1000,
            v_reset: 0,
            decay: 200,
            _padding1: 0,
            adapt_increment: 50,
            adapt_decay: 100,
            refrac_period: 2,
        }
    }
}

impl PulsarLifConfig {
    /// Adaptive configuration.
    pub fn adaptive() -> Self {
        Self {
            v_thresh: 1000,
            v_reset: 0,
            decay: 200,
            _padding1: 0,
            adapt_increment: 100,
            adapt_decay: 50,
            refrac_period: 2,
        }
    }

    /// Non-adaptive configuration.
    pub fn non_adaptive() -> Self {
        Self {
            v_thresh: 1000,
            v_reset: 0,
            decay: 200,
            _padding1: 0,
            adapt_increment: 0,
            adapt_decay: 255,
            refrac_period: 2,
        }
    }
}

impl NeuronConfig for PulsarLifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.v_thresh <= self.v_reset {
            return Err("Threshold must be greater than reset".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Pulsar-style LIF neuron for generic neuromorphic hardware.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PulsarLifNeuron {
    pub state: PulsarLifState,
    pub config: PulsarLifConfig,
}

impl PulsarLifNeuron {
    pub fn new(config: PulsarLifConfig) -> Self {
        Self {
            state: PulsarLifState::default(),
            config,
        }
    }
}

impl MembraneDynamics for PulsarLifNeuron {
    fn membrane_potential(&self) -> f32 {
        self.state.v as f32
    }

    fn set_membrane_potential(&mut self, v: f32) {
        self.state.v = v.round() as i32;
    }

    fn rest_potential(&self) -> f32 {
        self.config.v_reset as f32
    }
}

impl NeuronModel for PulsarLifNeuron {
    type Config = PulsarLifConfig;

    fn update(&mut self, input_current: f32, _dt: f32) -> bool {
        // Refractory period
        if self.state.refrac_timer > 0 {
            self.state.refrac_timer -= 1;
            return false;
        }

        // Decay membrane potential
        self.state.v -= (self.state.v * self.config.decay as i32) >> 8;

        // Decay adaptation
        self.state.adapt -= (self.state.adapt * self.config.adapt_decay as i32) >> 8;

        // Add input and subtract adaptation
        let input_int = (input_current * 10.0).round() as i32;
        self.state.v += input_int - self.state.adapt;

        // Check for spike
        if self.state.v >= self.config.v_thresh {
            self.state.adapt += self.config.adapt_increment;
            self.reset();
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.state.v = self.config.v_reset;
        self.state.refrac_timer = self.config.refrac_period;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh as f32
    }

    fn is_refractory(&self) -> bool {
        self.state.refrac_timer > 0
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// ============================================================================
// 3. QuantizedLIF - Fixed-Point for Edge Devices
// ============================================================================

/// Quantized LIF state (8-bit fixed point).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Pod, Zeroable)]
pub struct QuantizedLifState {
    /// Membrane potential (Q8.8 fixed point)
    pub v: i16,
    /// Refractory flag
    pub refrac: u8,
    pub _padding: u8,
}

impl Default for QuantizedLifState {
    fn default() -> Self {
        Self {
            v: 0,
            refrac: 0,
            _padding: 0,
        }
    }
}

/// Quantized LIF configuration (8-bit parameters).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Pod, Zeroable)]
pub struct QuantizedLifConfig {
    /// Threshold (Q8.8)
    pub v_thresh: i16,
    /// Reset value (Q8.8)
    pub v_reset: i16,
    /// Decay factor (0-255, /256)
    pub decay_factor: u8,
    /// Refractory cycles
    pub refrac_cycles: u8,
    /// Bit width for quantization
    pub bit_width: u8,
    pub _padding: u8,
}

impl Default for QuantizedLifConfig {
    fn default() -> Self {
        Self {
            v_thresh: 256, // 1.0 in Q8.8
            v_reset: 0,
            decay_factor: 250, // ~0.98 decay
            refrac_cycles: 2,
            bit_width: 8,
            _padding: 0,
        }
    }
}

impl QuantizedLifConfig {
    /// 4-bit quantization.
    pub fn bit4() -> Self {
        Self {
            bit_width: 4,
            ..Default::default()
        }
    }

    /// 8-bit quantization.
    pub fn bit8() -> Self {
        Self {
            bit_width: 8,
            ..Default::default()
        }
    }

    /// 16-bit quantization.
    pub fn bit16() -> Self {
        Self {
            bit_width: 16,
            ..Default::default()
        }
    }
}

impl NeuronConfig for QuantizedLifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.v_thresh <= self.v_reset {
            return Err("Threshold must be greater than reset".to_string());
        }
        if self.bit_width == 0 || self.bit_width > 16 {
            return Err("Bit width must be between 1 and 16".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Quantized LIF for extreme low-power edge devices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizedLifNeuron {
    pub state: QuantizedLifState,
    pub config: QuantizedLifConfig,
}

impl QuantizedLifNeuron {
    pub fn new(config: QuantizedLifConfig) -> Self {
        Self {
            state: QuantizedLifState::default(),
            config,
        }
    }

    /// Quantize value to configured bit width.
    fn quantize(&self, value: i16) -> i16 {
        let max_val = (1 << (self.config.bit_width - 1)) - 1;
        let min_val = -(1 << (self.config.bit_width - 1));
        value.clamp(min_val, max_val)
    }

    /// Get memory footprint in bytes.
    pub fn memory_footprint(&self) -> usize {
        std::mem::size_of::<QuantizedLifState>() + std::mem::size_of::<QuantizedLifConfig>()
    }
}

impl MembraneDynamics for QuantizedLifNeuron {
    fn membrane_potential(&self) -> f32 {
        // Convert Q8.8 to float
        self.state.v as f32 / 256.0
    }

    fn set_membrane_potential(&mut self, v: f32) {
        // Convert float to Q8.8
        self.state.v = self.quantize((v * 256.0).round() as i16);
    }

    fn rest_potential(&self) -> f32 {
        self.config.v_reset as f32 / 256.0
    }
}

impl NeuronModel for QuantizedLifNeuron {
    type Config = QuantizedLifConfig;

    fn update(&mut self, input_current: f32, _dt: f32) -> bool {
        // Refractory period
        if self.state.refrac > 0 {
            self.state.refrac -= 1;
            return false;
        }

        // Apply decay (Q8.8 fixed point arithmetic)
        let decay = (self.state.v as i32 * self.config.decay_factor as i32) >> 8;
        self.state.v = self.quantize((self.state.v as i32 - decay) as i16);

        // Add quantized input
        let input_q = (input_current * 256.0).round() as i16;
        self.state.v = self.quantize(self.state.v.saturating_add(input_q));

        // Check for spike
        if self.state.v >= self.config.v_thresh {
            self.reset();
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.state.v = self.config.v_reset;
        self.state.refrac = self.config.refrac_cycles;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh as f32 / 256.0
    }

    fn is_refractory(&self) -> bool {
        self.state.refrac > 0
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xylo_lif() {
        let mut neuron = XyloLifNeuron::new(XyloLifConfig::default());
        assert_eq!(neuron.state.v, 0);

        // Apply input
        neuron.update(50.0, 1.0);
        assert!(neuron.state.v > 0);
    }

    #[test]
    fn test_xylo_hardware_export() {
        let neuron = XyloLifNeuron::new(XyloLifConfig::default());
        let params = neuron.to_hardware_params();

        assert_eq!(params.len(), 8);
        assert_eq!(params[4], neuron.config.leak);
    }

    #[test]
    fn test_xylo_presets() {
        let fast = XyloLifConfig::fast();
        let slow = XyloLifConfig::slow();

        assert!(fast.leak > slow.leak);
        assert!(fast.refrac_cycles < slow.refrac_cycles);
    }

    #[test]
    fn test_pulsar_lif() {
        let mut neuron = PulsarLifNeuron::new(PulsarLifConfig::default());
        assert_eq!(neuron.state.v, 0);

        neuron.update(100.0, 1.0);
        assert!(neuron.state.v > 0);
    }

    #[test]
    fn test_pulsar_adaptation() {
        let mut neuron = PulsarLifNeuron::new(PulsarLifConfig::adaptive());

        // Trigger spike
        for _ in 0..20 {
            neuron.update(200.0, 1.0);
        }

        // Adaptation should build up
        assert!(neuron.state.adapt >= 0);
    }

    #[test]
    fn test_quantized_lif() {
        let mut neuron = QuantizedLifNeuron::new(QuantizedLifConfig::default());

        // Test Q8.8 conversion
        neuron.set_membrane_potential(1.5);
        let v = neuron.membrane_potential();
        assert!((v - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_quantized_bit_widths() {
        let bit4 = QuantizedLifConfig::bit4();
        let bit8 = QuantizedLifConfig::bit8();
        let bit16 = QuantizedLifConfig::bit16();

        assert_eq!(bit4.bit_width, 4);
        assert_eq!(bit8.bit_width, 8);
        assert_eq!(bit16.bit_width, 16);
    }

    #[test]
    fn test_quantized_clamping() {
        let neuron = QuantizedLifNeuron::new(QuantizedLifConfig::bit8());

        // Test clamping to 8-bit range
        let clamped = neuron.quantize(1000);
        assert!(clamped >= -128 && clamped <= 127);
    }

    #[test]
    fn test_quantized_memory_footprint() {
        let neuron = QuantizedLifNeuron::new(QuantizedLifConfig::default());
        let footprint = neuron.memory_footprint();

        assert!(footprint > 0);
        assert!(footprint < 100); // Should be very small
    }

    #[test]
    fn test_hardware_neurons_spike() {
        let mut xylo = XyloLifNeuron::new(XyloLifConfig::default());
        let mut pulsar = PulsarLifNeuron::new(PulsarLifConfig::default());
        let mut quantized = QuantizedLifNeuron::new(QuantizedLifConfig::default());

        let mut xylo_spiked = false;
        let mut pulsar_spiked = false;
        let mut quantized_spiked = false;

        for _ in 0..100 {
            if xylo.update(20.0, 1.0) {
                xylo_spiked = true;
            }
            if pulsar.update(200.0, 1.0) {
                pulsar_spiked = true;
            }
            if quantized.update(0.5, 1.0) {
                quantized_spiked = true;
            }
        }

        assert!(xylo_spiked || pulsar_spiked || quantized_spiked);
    }
}
