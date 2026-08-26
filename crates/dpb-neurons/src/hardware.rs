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
///
/// Xylo carries **16-bit** synaptic and membrane state per neuron, with 8-bit
/// synaptic weights. The doc comments here previously described these as 8-bit
/// while the fields were already `i16`, so the stated hardware constraint and
/// the actual one disagreed.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Pod, Zeroable)]
#[derive(Default)]
pub struct XyloLifState {
    /// Membrane potential (16-bit hardware state).
    pub v: i16,
    /// Synaptic current (16-bit hardware state).
    ///
    /// Xylo accumulates weighted input into a synaptic state that decays on its
    /// own time constant before reaching the membrane. Injecting input straight
    /// into `v` collapses two state variables into one and removes the synaptic
    /// filter entirely.
    pub i_syn: i16,
    /// Refractory counter (hardware cycles)
    pub refrac_counter: u8,
    pub _padding: u8,
}


/// Bit-shift decay, as Xylo approximates an exponential.
///
/// The hardware computes `v' = v - (v >> dash)`. When the shift underflows to
/// zero the decay is linear instead, so a small state still reaches rest rather
/// than sticking. Negative values need no special case: an arithmetic shift
/// right already yields -1 for small magnitudes, which steps them toward zero.
#[inline]
fn xylo_bitshift_decay(value: i16, dash: u8) -> i16 {
    let shifted = value >> dash.min(15);
    let decay = if shifted == 0 && value > 0 { 1 } else { shifted };
    value.saturating_sub(decay)
}

/// Xylo LIF configuration (hardware parameter ranges).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Pod, Zeroable)]
pub struct XyloLifConfig {
    /// Firing threshold (16-bit)
    pub v_thresh: i16,
    /// Reset value (16-bit)
    pub v_reset: i16,
    /// Membrane decay, as a right-bit-shift amount.
    ///
    /// Xylo decays state by `v -= v >> dash`, so this is a SHIFT COUNT and not a
    /// rate: the field is 4 bits wide on the hardware, valid over 0..=15, and
    /// **smaller values decay faster** -- `dash_mem = 1` halves the membrane
    /// each step while `dash_mem = 15` barely moves it.
    ///
    /// This replaces a `leak` field that was applied as `(v * leak) >> 8`, a
    /// multiplicative rate the hardware does not implement, and one that
    /// overflowed: `v * leak` is `i16` arithmetic, so any membrane above 6553
    /// panicked in debug and wrapped in release, reachable through
    /// `set_membrane_potential` with the DEFAULT config.
    pub dash_mem: u8,
    /// Synaptic decay, as a right-bit-shift amount. Same 0..=15 range and the
    /// same "smaller is faster" sense as [`Self::dash_mem`].
    pub dash_syn: u8,
    /// Refractory period in hardware cycles
    pub refrac_cycles: u8,
    /// Scaling factor for inputs
    pub input_scale: u8,
}

impl Default for XyloLifConfig {
    fn default() -> Self {
        Self {
            v_thresh: 100,
            v_reset: 0,
            dash_mem: 4,
            dash_syn: 3,
            refrac_cycles: 2,
            input_scale: 1,
        }
    }
}

impl XyloLifConfig {
    /// Fast dynamics configuration.
    pub fn fast() -> Self {
        // Smaller dash = faster decay.
        Self {
            v_thresh: 80,
            dash_mem: 2,
            dash_syn: 1,
            refrac_cycles: 1,
            ..Default::default()
        }
    }

    /// Slow dynamics configuration.
    pub fn slow() -> Self {
        // Larger dash = slower decay.
        Self {
            v_thresh: 120,
            dash_mem: 8,
            dash_syn: 6,
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
        // The dash fields are 4 bits wide on the hardware.
        if self.dash_mem > 15 {
            return Err(format!("dash_mem must be 0..=15, got {}", self.dash_mem));
        }
        if self.dash_syn > 15 {
            return Err(format!("dash_syn must be 0..=15, got {}", self.dash_syn));
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
            self.config.dash_mem,
            self.config.dash_syn,
            self.config.refrac_cycles,
            self.config.input_scale,
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

        // Weighted input accumulates into the SYNAPTIC state, which then drives
        // the membrane -- the two-stage path the hardware implements. Float to
        // int casts saturate in Rust, so an out-of-range input pins at the
        // 16-bit bound rather than wrapping.
        let scaled_input = (input_current * self.config.input_scale as f32).round() as i16;
        self.state.i_syn = self.state.i_syn.saturating_add(scaled_input);
        self.state.v = self.state.v.saturating_add(self.state.i_syn);

        // Each state decays on its own bit-shift constant.
        self.state.i_syn = xylo_bitshift_decay(self.state.i_syn, self.config.dash_syn);
        self.state.v = xylo_bitshift_decay(self.state.v, self.config.dash_mem);

        // No clamp to [-32768, 32767] here: that is exactly the range of `i16`,
        // so the previous clamp could never do anything, and the saturating
        // arithmetic above already holds the bound.

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
#[derive(Default)]
pub struct PulsarLifState {
    /// Membrane potential (16-bit)
    pub v: i32,
    /// Adaptation current (16-bit)
    pub adapt: i32,
    /// Refractory timer
    pub refrac_timer: u16,
    pub _padding: u16,
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
#[derive(Default)]
pub struct QuantizedLifState {
    /// Membrane potential (Q8.8 fixed point)
    pub v: i16,
    /// Refractory flag
    pub refrac: u8,
    pub _padding: u8,
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
        // Use i32 to avoid overflow for 16-bit width
        let max_val: i32 = (1_i32 << (self.config.bit_width - 1)) - 1;
        let min_val: i32 = -(1_i32 << (self.config.bit_width - 1));
        (value as i32).clamp(min_val, max_val) as i16
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
        assert_eq!(params[4], neuron.config.dash_mem);
        assert_eq!(params[5], neuron.config.dash_syn);
    }

    #[test]
    fn test_xylo_presets() {
        let fast = XyloLifConfig::fast();
        let slow = XyloLifConfig::slow();

        // Bit-shift decay: a SMALLER dash decays faster, so the fast preset
        // must have the smaller value. Read as a rate -- which is what the old
        // `leak` field was -- this comparison points the other way, and did.
        assert!(fast.dash_mem < slow.dash_mem);
        assert!(fast.dash_syn < slow.dash_syn);
        assert!(fast.refrac_cycles < slow.refrac_cycles);
    }

    #[test]
    fn test_pulsar_lif() {
        let mut neuron = PulsarLifNeuron::new(PulsarLifConfig::default());
        assert_eq!(neuron.state.v, 0);

        // Use smaller input that doesn't trigger spike (threshold is 1000)
        // input_int = (50.0 * 10.0) = 500
        neuron.update(50.0, 1.0);
        assert!(neuron.state.v > 0, "Voltage should increase with sub-threshold input");
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
        // Use 16-bit config to properly test Q8.8 conversion
        let mut neuron = QuantizedLifNeuron::new(QuantizedLifConfig::bit16());

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
        assert!((-128..=127).contains(&clamped));
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

    /// The membrane update must not overflow for any reachable state.
    ///
    /// Regression: decay was `(v * leak) >> 8` in `i16` arithmetic, so any
    /// membrane above 6553 overflowed -- a panic in debug, a silent wrap in
    /// release -- and `set_membrane_potential` reaches that with the DEFAULT
    /// config.
    #[test]
    fn test_xylo_no_overflow_across_range() {
        for v in [i16::MIN, -20000, -1, 0, 1, 6553, 20000, i16::MAX] {
            for dash in 0u8..=15 {
                let config = XyloLifConfig { dash_mem: dash, dash_syn: dash, ..Default::default() };
                let mut neuron = XyloLifNeuron::new(config);
                neuron.state.v = v;
                neuron.state.i_syn = v;
                // Must not panic, and must stay inside the 16-bit state.
                let _ = neuron.update(1000.0, 1.0);
            }
        }
    }

    /// Bit-shift decay must move state toward rest and actually reach it.
    #[test]
    fn test_xylo_bitshift_decay_reaches_rest() {
        // `v - (v >> dash)`, with a linear step when the shift underflows.
        assert_eq!(xylo_bitshift_decay(1024, 1), 512);
        assert_eq!(xylo_bitshift_decay(1024, 4), 960);
        assert_eq!(xylo_bitshift_decay(0, 4), 0);

        // Without the linear fallback a small positive state would stick
        // forever, since `1 >> 4 == 0`.
        assert_eq!(xylo_bitshift_decay(1, 4), 0);

        // Left to itself the membrane must settle at rest, not stall short.
        let config = XyloLifConfig { v_thresh: 30000, ..Default::default() };
        let mut neuron = XyloLifNeuron::new(config);
        neuron.state.v = 5000;
        for _ in 0..10_000 {
            neuron.update(0.0, 1.0);
        }
        assert_eq!(neuron.state.v, 0, "membrane stalled at {}", neuron.state.v);
    }

    /// A smaller dash must decay faster -- the sense that a rate-shaped field
    /// gets backwards.
    #[test]
    fn test_xylo_smaller_dash_decays_faster() {
        let decay_after = |dash: u8| {
            let config = XyloLifConfig { dash_mem: dash, v_thresh: 30000, ..Default::default() };
            let mut neuron = XyloLifNeuron::new(config);
            neuron.state.v = 10000;
            for _ in 0..10 {
                neuron.update(0.0, 1.0);
            }
            neuron.state.v
        };
        assert!(
            decay_after(1) < decay_after(8),
            "dash 1 must decay further in 10 steps than dash 8"
        );
    }

    /// Configuration must reject dash values wider than the hardware field.
    #[test]
    fn test_xylo_rejects_out_of_range_dash() {
        assert!(XyloLifConfig::default().validate().is_ok());
        assert!(XyloLifConfig { dash_mem: 16, ..Default::default() }.validate().is_err());
        assert!(XyloLifConfig { dash_syn: 16, ..Default::default() }.validate().is_err());
        assert!(XyloLifConfig { dash_mem: 15, dash_syn: 15, ..Default::default() }.validate().is_ok());
    }

}
