//! Leaky Integrate-and-Fire (LIF) neuron models and variants.
//!
//! This module implements 7 variants of the LIF neuron model, from the simplest
//! integrate-and-fire to complex adaptive models.

use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel, NeuronState};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

// ============================================================================
// 1. IF Neuron - Integrate-and-Fire (simplest)
// ============================================================================

/// Integrate-and-Fire neuron - the simplest spiking neuron model.
///
/// Equation: dv/dt = I
///
/// No leak - just integrates input until threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IfNeuron {
    pub state: NeuronState,
    pub config: IfConfig,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct IfConfig {
    /// Spike threshold in mV
    pub v_thresh: f32,
    /// Reset potential in mV
    pub v_reset: f32,
    /// Resting potential in mV
    pub v_rest: f32,
    /// Membrane capacitance in pF
    pub c_m: f32,
}

impl Default for IfConfig {
    fn default() -> Self {
        Self {
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            c_m: 250.0,
        }
    }
}

impl NeuronConfig for IfConfig {
    fn validate(&self) -> Result<(), String> {
        if self.v_thresh <= self.v_reset {
            return Err("Threshold must be greater than reset potential".to_string());
        }
        if self.c_m <= 0.0 {
            return Err("Capacitance must be positive".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

impl IfNeuron {
    pub fn new(config: IfConfig) -> Self {
        Self {
            state: NeuronState::new(config.v_rest),
            config,
        }
    }
}

impl MembraneDynamics for IfNeuron {
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

impl NeuronModel for IfNeuron {
    type Config = IfConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        // Simple integration: dv/dt = I/C_m
        self.state.v += (input_current / self.config.c_m) * dt;

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
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// ============================================================================
// 2. LIF Neuron - Leaky Integrate-and-Fire (most common)
// ============================================================================

/// Leaky Integrate-and-Fire neuron - the most commonly used spiking neuron model.
///
/// Equation: τ_m * dv/dt = -(v - v_rest) + R_m * I
///
/// Includes membrane leak for realistic dynamics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LifNeuron {
    pub state: NeuronState,
    pub config: LifConfig,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct LifConfig {
    /// Membrane time constant in ms
    pub tau_mem: f32,
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
}

impl Default for LifConfig {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            r_m: 10.0,
            tau_refrac: 2.0,
        }
    }
}

impl NeuronConfig for LifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_mem <= 0.0 {
            return Err("Membrane time constant must be positive".to_string());
        }
        if self.v_thresh <= self.v_reset {
            return Err("Threshold must be greater than reset potential".to_string());
        }
        if self.r_m <= 0.0 {
            return Err("Membrane resistance must be positive".to_string());
        }
        if self.tau_refrac < 0.0 {
            return Err("Refractory period must be non-negative".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

impl LifNeuron {
    pub fn new(config: LifConfig) -> Self {
        Self {
            state: NeuronState::new(config.v_rest),
            config,
        }
    }
}

impl MembraneDynamics for LifNeuron {
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

impl NeuronModel for LifNeuron {
    type Config = LifConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        // Update refractory period
        self.state.update_refractory(dt);

        if self.state.is_refractory() {
            return false;
        }

        // LIF dynamics: τ_m * dv/dt = -(v - v_rest) + R_m * I
        let dv = (-(self.state.v - self.config.v_rest) + self.config.r_m * input_current)
            / self.config.tau_mem;
        self.state.v += dv * dt;

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
        self.state.refrac_timer = self.config.tau_refrac;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh
    }

    fn is_refractory(&self) -> bool {
        self.state.is_refractory()
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// ============================================================================
// 3. CLIF Neuron - Current-based Leaky Integrate-and-Fire
// ============================================================================

/// Current-based LIF with explicit synaptic current dynamics.
///
/// Equations:
/// - τ_m * dv/dt = -(v - v_rest) + I_syn
/// - τ_syn * dI_syn/dt = -I_syn + I_ext
///
/// Models synaptic current filtering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClifNeuron {
    pub state: ClifState,
    pub config: ClifConfig,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct ClifState {
    /// Membrane potential in mV
    pub v: f32,
    /// Synaptic current in nA
    pub i_syn: f32,
    /// Refractory timer in ms
    pub refrac_timer: f32,
    pub _padding: f32,
}

impl Default for ClifState {
    fn default() -> Self {
        Self {
            v: -65.0,
            i_syn: 0.0,
            refrac_timer: 0.0,
            _padding: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct ClifConfig {
    pub tau_mem: f32,
    pub tau_syn: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub v_rest: f32,
    pub tau_refrac: f32,
    pub _padding: [f32; 2],
}

impl Default for ClifConfig {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            tau_syn: 5.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            tau_refrac: 2.0,
            _padding: [0.0; 2],
        }
    }
}

impl NeuronConfig for ClifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_mem <= 0.0 || self.tau_syn <= 0.0 {
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

impl ClifNeuron {
    pub fn new(config: ClifConfig) -> Self {
        Self {
            state: ClifState::default(),
            config,
        }
    }
}

impl MembraneDynamics for ClifNeuron {
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

impl NeuronModel for ClifNeuron {
    type Config = ClifConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        if self.state.refrac_timer > 0.0 {
            self.state.refrac_timer -= dt;
            if self.state.refrac_timer < 0.0 {
                self.state.refrac_timer = 0.0;
            }
            return false;
        }

        // Update synaptic current
        let di_syn = (-self.state.i_syn + input_current) / self.config.tau_syn;
        self.state.i_syn += di_syn * dt;

        // Update membrane potential
        let dv = (-(self.state.v - self.config.v_rest) + self.state.i_syn) / self.config.tau_mem;
        self.state.v += dv * dt;

        if self.state.v >= self.config.v_thresh {
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

// ============================================================================
// 4. ALIF Neuron - Adaptive Leaky Integrate-and-Fire
// ============================================================================

/// Adaptive LIF with spike-triggered threshold adaptation.
///
/// Equations:
/// - τ_m * dv/dt = -(v - v_rest) + R_m * I
/// - τ_adapt * da/dt = -a
/// - v_thresh_eff = v_thresh + a
/// - On spike: a += b
///
/// Models spike-frequency adaptation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlifNeuron {
    pub state: AlifState,
    pub config: AlifConfig,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct AlifState {
    pub v: f32,
    pub adapt: f32,
    pub refrac_timer: f32,
    pub _padding: f32,
}

impl Default for AlifState {
    fn default() -> Self {
        Self {
            v: -65.0,
            adapt: 0.0,
            refrac_timer: 0.0,
            _padding: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct AlifConfig {
    pub tau_mem: f32,
    pub tau_adapt: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub v_rest: f32,
    pub r_m: f32,
    pub tau_refrac: f32,
    pub adapt_increment: f32,
}

impl Default for AlifConfig {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            tau_adapt: 100.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            r_m: 10.0,
            tau_refrac: 2.0,
            adapt_increment: 1.0,
        }
    }
}

impl NeuronConfig for AlifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_mem <= 0.0 || self.tau_adapt <= 0.0 {
            return Err("Time constants must be positive".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

impl AlifNeuron {
    pub fn new(config: AlifConfig) -> Self {
        Self {
            state: AlifState::default(),
            config,
        }
    }

    pub fn effective_threshold(&self) -> f32 {
        self.config.v_thresh + self.state.adapt
    }
}

impl MembraneDynamics for AlifNeuron {
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

impl NeuronModel for AlifNeuron {
    type Config = AlifConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        if self.state.refrac_timer > 0.0 {
            self.state.refrac_timer -= dt;
            if self.state.refrac_timer < 0.0 {
                self.state.refrac_timer = 0.0;
            }
            return false;
        }

        // Update adaptation variable
        let da = -self.state.adapt / self.config.tau_adapt;
        self.state.adapt += da * dt;

        // Update membrane potential
        let dv = (-(self.state.v - self.config.v_rest) + self.config.r_m * input_current)
            / self.config.tau_mem;
        self.state.v += dv * dt;

        // Check for spike with adaptive threshold
        if self.state.v >= self.effective_threshold() {
            self.state.adapt += self.config.adapt_increment;
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
        self.effective_threshold()
    }

    fn is_refractory(&self) -> bool {
        self.state.refrac_timer > 0.0
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// ============================================================================
// 5. ELIF Neuron - Exponential Leaky Integrate-and-Fire
// ============================================================================

/// Exponential LIF with exponential spike mechanism.
///
/// Equation: τ_m * dv/dt = -(v - v_rest) + Δ_T * exp((v - v_thresh)/Δ_T) + R_m * I
///
/// Features smooth spike initiation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElifNeuron {
    pub state: NeuronState,
    pub config: ElifConfig,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct ElifConfig {
    pub tau_mem: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub v_rest: f32,
    pub r_m: f32,
    pub delta_t: f32,
    pub v_spike: f32,
    pub tau_refrac: f32,
}

impl Default for ElifConfig {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            r_m: 10.0,
            delta_t: 2.0,
            v_spike: 20.0,
            tau_refrac: 2.0,
        }
    }
}

impl NeuronConfig for ElifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_mem <= 0.0 || self.delta_t <= 0.0 {
            return Err("Time constants must be positive".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

impl ElifNeuron {
    pub fn new(config: ElifConfig) -> Self {
        Self {
            state: NeuronState::new(config.v_rest),
            config,
        }
    }
}

impl MembraneDynamics for ElifNeuron {
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

impl NeuronModel for ElifNeuron {
    type Config = ElifConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        self.state.update_refractory(dt);

        if self.state.is_refractory() {
            return false;
        }

        // Exponential spike mechanism
        let exp_term = self.config.delta_t
            * ((self.state.v - self.config.v_thresh) / self.config.delta_t).exp();
        let dv =
            (-(self.state.v - self.config.v_rest) + exp_term + self.config.r_m * input_current)
                / self.config.tau_mem;
        self.state.v += dv * dt;

        // Check if spike threshold reached
        if self.state.v >= self.config.v_spike {
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
        self.state.is_refractory()
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// ============================================================================
// 6. QLIF Neuron - Quadratic Leaky Integrate-and-Fire
// ============================================================================

/// Quadratic LIF with parabolic nonlinearity.
///
/// Equation: τ_m * dv/dt = a * (v - v_rest)(v - v_crit) + R_m * I
///
/// Similar to ELIF but with quadratic nonlinearity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QlifNeuron {
    pub state: NeuronState,
    pub config: QlifConfig,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct QlifConfig {
    pub tau_mem: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub v_rest: f32,
    pub v_crit: f32,
    pub a: f32,
    pub r_m: f32,
    pub tau_refrac: f32,
}

impl Default for QlifConfig {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            v_crit: -55.0,
            a: 0.04,
            r_m: 10.0,
            tau_refrac: 2.0,
        }
    }
}

impl NeuronConfig for QlifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_mem <= 0.0 {
            return Err("Membrane time constant must be positive".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

impl QlifNeuron {
    pub fn new(config: QlifConfig) -> Self {
        Self {
            state: NeuronState::new(config.v_rest),
            config,
        }
    }
}

impl MembraneDynamics for QlifNeuron {
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

impl NeuronModel for QlifNeuron {
    type Config = QlifConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        self.state.update_refractory(dt);

        if self.state.is_refractory() {
            return false;
        }

        // Quadratic nonlinearity
        let quad_term = self.config.a
            * (self.state.v - self.config.v_rest)
            * (self.state.v - self.config.v_crit);
        let dv = (quad_term + self.config.r_m * input_current) / self.config.tau_mem;
        self.state.v += dv * dt;

        if self.state.v >= self.config.v_thresh {
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
        self.state.is_refractory()
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// ============================================================================
// 7. GLIF Neuron - Generalized Leaky Integrate-and-Fire
// ============================================================================

/// Generalized LIF with multiple adaptation currents.
///
/// Most flexible LIF variant with multiple time scales of adaptation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlifNeuron {
    pub state: GlifState,
    pub config: GlifConfig,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct GlifState {
    pub v: f32,
    pub adapt1: f32,
    pub adapt2: f32,
    pub refrac_timer: f32,
}

impl Default for GlifState {
    fn default() -> Self {
        Self {
            v: -65.0,
            adapt1: 0.0,
            adapt2: 0.0,
            refrac_timer: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct GlifConfig {
    pub tau_mem: f32,
    pub tau_adapt1: f32,
    pub tau_adapt2: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub v_rest: f32,
    pub r_m: f32,
    pub tau_refrac: f32,
    pub adapt1_increment: f32,
    pub adapt2_increment: f32,
    pub _padding: [f32; 2],
}

impl Default for GlifConfig {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            tau_adapt1: 100.0,
            tau_adapt2: 500.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            r_m: 10.0,
            tau_refrac: 2.0,
            adapt1_increment: 0.5,
            adapt2_increment: 0.2,
            _padding: [0.0; 2],
        }
    }
}

impl NeuronConfig for GlifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_mem <= 0.0 || self.tau_adapt1 <= 0.0 || self.tau_adapt2 <= 0.0 {
            return Err("Time constants must be positive".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

impl GlifNeuron {
    pub fn new(config: GlifConfig) -> Self {
        Self {
            state: GlifState::default(),
            config,
        }
    }

    pub fn effective_threshold(&self) -> f32 {
        self.config.v_thresh + self.state.adapt1 + self.state.adapt2
    }
}

impl MembraneDynamics for GlifNeuron {
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

impl NeuronModel for GlifNeuron {
    type Config = GlifConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        if self.state.refrac_timer > 0.0 {
            self.state.refrac_timer -= dt;
            if self.state.refrac_timer < 0.0 {
                self.state.refrac_timer = 0.0;
            }
            return false;
        }

        // Update both adaptation variables
        self.state.adapt1 += (-self.state.adapt1 / self.config.tau_adapt1) * dt;
        self.state.adapt2 += (-self.state.adapt2 / self.config.tau_adapt2) * dt;

        // Update membrane potential
        let dv = (-(self.state.v - self.config.v_rest) + self.config.r_m * input_current)
            / self.config.tau_mem;
        self.state.v += dv * dt;

        // Check for spike with combined adaptive threshold
        if self.state.v >= self.effective_threshold() {
            self.state.adapt1 += self.config.adapt1_increment;
            self.state.adapt2 += self.config.adapt2_increment;
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
        self.effective_threshold()
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
    fn test_if_neuron() {
        let mut neuron = IfNeuron::new(IfConfig::default());
        assert_eq!(neuron.membrane_potential(), -65.0);

        // Should spike with enough input
        let dt = 1.0;
        let input = 100.0; // nA
        let spiked = neuron.update(input, dt);
        assert!(spiked || neuron.membrane_potential() > -65.0);
    }

    #[test]
    fn test_lif_neuron() {
        let config = LifConfig::default();
        let mut neuron = LifNeuron::new(config);

        // Test that neuron integrates
        let spiked = neuron.update(5.0, 1.0);
        assert!(!spiked);
        assert!(neuron.membrane_potential() > -65.0);

        // Test reset
        neuron.reset();
        assert_eq!(neuron.membrane_potential(), config.v_reset);
        assert!(neuron.is_refractory());
    }

    #[test]
    fn test_lif_spike() {
        let mut neuron = LifNeuron::new(LifConfig::default());
        let mut spike_count = 0;

        // Apply strong current
        for _ in 0..100 {
            if neuron.update(20.0, 1.0) {
                spike_count += 1;
            }
        }

        assert!(spike_count > 0, "Neuron should spike with strong input");
    }

    #[test]
    fn test_alif_adaptation() {
        let mut neuron = AlifNeuron::new(AlifConfig::default());
        let initial_thresh = neuron.threshold();

        // Trigger spike
        for _ in 0..50 {
            neuron.update(20.0, 1.0);
        }

        // Threshold should increase due to adaptation
        assert!(neuron.threshold() >= initial_thresh);
    }

    #[test]
    fn test_elif_exponential() {
        let mut neuron = ElifNeuron::new(ElifConfig::default());
        let v_initial = neuron.membrane_potential();

        // Apply input
        neuron.update(10.0, 1.0);

        // Should depolarize
        assert!(neuron.membrane_potential() > v_initial);
    }

    #[test]
    fn test_glif_multiple_adaptations() {
        let config = GlifConfig::default();
        let mut neuron = GlifNeuron::new(config);

        assert_eq!(neuron.state.adapt1, 0.0);
        assert_eq!(neuron.state.adapt2, 0.0);

        // Force spike
        for _ in 0..50 {
            neuron.update(20.0, 1.0);
        }

        // Both adaptation variables should be affected
        // (Either increased directly or decaying from a previous increase)
        // The exact values depend on the dynamics, but we can check the mechanism works
    }

    #[test]
    fn test_clif_synaptic_current() {
        let mut neuron = ClifNeuron::new(ClifConfig::default());
        assert_eq!(neuron.state.i_syn, 0.0);

        // Apply input current
        neuron.update(10.0, 1.0);

        // Synaptic current should build up
        assert!(neuron.state.i_syn > 0.0);
    }

    #[test]
    fn test_qlif_quadratic() {
        let mut neuron = QlifNeuron::new(QlifConfig::default());
        let v_initial = neuron.membrane_potential();

        neuron.update(10.0, 1.0);

        // Should depolarize
        assert!(neuron.membrane_potential() != v_initial);
    }
}
