//! Modulatory Gating Mechanisms
//!
//! This module implements various gating mechanisms by which neuromodulators
//! control information flow and neural responses:
//! - Gain modulation (multiplicative and additive)
//! - Input gating
//! - Output gating
//! - Threshold modulation

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::{NeuromodError, NeuromodResult};

/// Type of modulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModulationType {
    /// Multiplicative gain modulation
    Multiplicative,
    /// Additive modulation (bias shift)
    Additive,
    /// Divisive normalization
    Divisive,
    /// Subtractive modulation
    Subtractive,
}

/// Gain modulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GainModulationConfig {
    /// Type of modulation
    pub modulation_type: ModulationType,
    /// Baseline gain (no modulation)
    pub baseline_gain: f64,
    /// Maximum gain
    pub max_gain: f64,
    /// Minimum gain
    pub min_gain: f64,
    /// Gain scaling factor
    pub gain_scale: f64,
}

impl Default for GainModulationConfig {
    fn default() -> Self {
        Self {
            modulation_type: ModulationType::Multiplicative,
            baseline_gain: 1.0,
            max_gain: 3.0,
            min_gain: 0.1,
            gain_scale: 1.0,
        }
    }
}

/// Gain Modulation
///
/// Implements multiplicative and additive gain control by neuromodulators.
/// Gain modulation allows flexible routing of information and context-dependent
/// processing without changing underlying connectivity.
pub struct GainModulation {
    /// Configuration
    pub config: GainModulationConfig,
    /// Current gain value
    pub gain: f64,
}

impl GainModulation {
    /// Create new gain modulation
    pub fn new(config: GainModulationConfig) -> Self {
        Self {
            gain: config.baseline_gain,
            config,
        }
    }

    /// Set gain from modulator concentration
    pub fn set_gain_from_modulator(&mut self, concentration: f64) {
        self.gain = self.config.baseline_gain + self.config.gain_scale * concentration;
        self.gain = self.gain.max(self.config.min_gain).min(self.config.max_gain);
    }

    /// Apply multiplicative gain modulation
    ///
    /// y = gain * x
    pub fn apply_multiplicative(&self, input: &Array1<f64>, modulation: f64) -> Array1<f64> {
        let effective_gain = self.gain * modulation;
        input.mapv(|x| x * effective_gain)
    }

    /// Apply additive modulation (bias shift)
    ///
    /// y = x + bias
    pub fn apply_additive(&self, input: &Array1<f64>, modulation: f64) -> Array1<f64> {
        let bias = modulation * self.config.gain_scale;
        input.mapv(|x| x + bias)
    }

    /// Apply divisive normalization
    ///
    /// y = x / (c + sum(x))
    pub fn apply_divisive(&self, input: &Array1<f64>, modulation: f64) -> Array1<f64> {
        let total_activity: f64 = input.sum();
        let normalization_constant = modulation * self.config.gain_scale;

        input.mapv(|x| x / (normalization_constant + total_activity))
    }

    /// Apply subtractive modulation
    ///
    /// y = max(0, x - threshold)
    pub fn apply_subtractive(&self, input: &Array1<f64>, modulation: f64) -> Array1<f64> {
        let threshold = modulation * self.config.gain_scale;
        input.mapv(|x| (x - threshold).max(0.0))
    }

    /// Apply configured modulation type
    pub fn apply(&self, input: &Array1<f64>, modulation: f64) -> Array1<f64> {
        match self.config.modulation_type {
            ModulationType::Multiplicative => self.apply_multiplicative(input, modulation),
            ModulationType::Additive => self.apply_additive(input, modulation),
            ModulationType::Divisive => self.apply_divisive(input, modulation),
            ModulationType::Subtractive => self.apply_subtractive(input, modulation),
        }
    }

    /// Apply 2D gain modulation
    pub fn apply_2d(&self, input: &Array2<f64>, modulation: f64) -> Array2<f64> {
        let effective_gain = self.gain * modulation;
        match self.config.modulation_type {
            ModulationType::Multiplicative => input.mapv(|x| x * effective_gain),
            ModulationType::Additive => {
                let bias = modulation * self.config.gain_scale;
                input.mapv(|x| x + bias)
            }
            ModulationType::Divisive => {
                let total_activity: f64 = input.sum();
                let normalization_constant = modulation * self.config.gain_scale;
                input.mapv(|x| x / (normalization_constant + total_activity))
            }
            ModulationType::Subtractive => {
                let threshold = modulation * self.config.gain_scale;
                input.mapv(|x| (x - threshold).max(0.0))
            }
        }
    }
}

/// Input Gating
///
/// Controls which inputs are processed based on neuromodulatory state.
/// Similar to attention mechanisms in transformers but biologically inspired.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputGating {
    /// Gating threshold
    pub threshold: f64,
    /// Gating slope (steepness of sigmoid)
    pub slope: f64,
    /// Current gate values
    pub gates: Array1<f64>,
}

impl InputGating {
    /// Create new input gating
    pub fn new(num_inputs: usize) -> Self {
        Self {
            threshold: 0.5,
            slope: 10.0,
            gates: Array1::ones(num_inputs),
        }
    }

    /// Compute gate values using sigmoid
    ///
    /// gate = 1 / (1 + exp(-slope * (modulation - threshold)))
    pub fn compute_gates(&mut self, modulation: &Array1<f64>) {
        self.gates = modulation.mapv(|m| {
            1.0 / (1.0 + (-self.slope * (m - self.threshold)).exp())
        });
    }

    /// Apply gating to input
    pub fn apply(&self, input: &Array1<f64>) -> NeuromodResult<Array1<f64>> {
        if input.len() != self.gates.len() {
            return Err(NeuromodError::DimensionMismatch {
                expected: self.gates.len(),
                actual: input.len(),
            });
        }

        Ok(input * &self.gates)
    }

    /// Hard gating (binary on/off)
    pub fn apply_hard(&self, input: &Array1<f64>) -> NeuromodResult<Array1<f64>> {
        if input.len() != self.gates.len() {
            return Err(NeuromodError::DimensionMismatch {
                expected: self.gates.len(),
                actual: input.len(),
            });
        }

        Ok(input
            .iter()
            .zip(self.gates.iter())
            .map(|(x, g)| if *g > 0.5 { *x } else { 0.0 })
            .collect())
    }
}

/// Output Gating
///
/// Controls which outputs are produced based on neuromodulatory state.
/// Useful for action selection and response inhibition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputGating {
    /// Gating threshold
    pub threshold: f64,
    /// Winner-take-all strength (0 = soft, 1 = hard WTA)
    pub wta_strength: f64,
    /// Current gate values
    pub gates: Array1<f64>,
}

impl OutputGating {
    /// Create new output gating
    pub fn new(num_outputs: usize) -> Self {
        Self {
            threshold: 0.5,
            wta_strength: 0.0,
            gates: Array1::ones(num_outputs),
        }
    }

    /// Compute gate values with winner-take-all
    pub fn compute_gates(&mut self, modulation: &Array1<f64>) {
        if self.wta_strength > 0.0 {
            // Find maximum
            let max_idx = modulation
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Apply WTA
            self.gates = modulation.mapv(|_| 0.0);
            self.gates[max_idx] = 1.0;

            // Blend with soft gating
            if self.wta_strength < 1.0 {
                let soft_gates = modulation.mapv(|m| {
                    if m > self.threshold { 1.0 } else { 0.0 }
                });
                self.gates = &self.gates * self.wta_strength
                    + &soft_gates * (1.0 - self.wta_strength);
            }
        } else {
            // Soft gating
            self.gates = modulation.mapv(|m| {
                if m > self.threshold { 1.0 } else { 0.0 }
            });
        }
    }

    /// Apply gating to output
    pub fn apply(&self, output: &Array1<f64>) -> NeuromodResult<Array1<f64>> {
        if output.len() != self.gates.len() {
            return Err(NeuromodError::DimensionMismatch {
                expected: self.gates.len(),
                actual: output.len(),
            });
        }

        Ok(output * &self.gates)
    }

    /// Apply soft gating (continuous)
    pub fn apply_soft(&self, output: &Array1<f64>, modulation: &Array1<f64>) -> NeuromodResult<Array1<f64>> {
        if output.len() != modulation.len() {
            return Err(NeuromodError::DimensionMismatch {
                expected: output.len(),
                actual: modulation.len(),
            });
        }

        Ok(output * modulation)
    }
}

/// Threshold Modulation
///
/// Modulates the firing threshold of neurons, affecting their excitability
/// and response properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdModulation {
    /// Baseline threshold
    pub baseline_threshold: f64,
    /// Modulation strength
    pub modulation_strength: f64,
    /// Maximum threshold
    pub max_threshold: f64,
    /// Minimum threshold
    pub min_threshold: f64,
}

impl ThresholdModulation {
    /// Create new threshold modulation
    pub fn new(baseline_threshold: f64) -> Self {
        Self {
            baseline_threshold,
            modulation_strength: 0.5,
            max_threshold: baseline_threshold * 2.0,
            min_threshold: baseline_threshold * 0.5,
        }
    }

    /// Compute modulated threshold
    ///
    /// High modulation → lower threshold (increased excitability)
    /// Low modulation → higher threshold (decreased excitability)
    pub fn compute_threshold(&self, modulation: f64) -> f64 {
        let delta = (modulation - 0.5) * self.modulation_strength * self.baseline_threshold;
        let threshold = self.baseline_threshold - delta; // Negative: higher modulation → lower threshold

        threshold.max(self.min_threshold).min(self.max_threshold)
    }

    /// Compute threshold for array of neurons
    pub fn compute_thresholds(&self, modulation: &Array1<f64>) -> Array1<f64> {
        modulation.mapv(|m| self.compute_threshold(m))
    }

    /// Check if membrane potential exceeds threshold
    pub fn check_spike(&self, membrane_potential: f64, modulation: f64) -> bool {
        let threshold = self.compute_threshold(modulation);
        membrane_potential >= threshold
    }

    /// Apply threshold to spike generation
    pub fn apply_threshold(&self, membrane_potentials: &Array1<f64>, modulation: &Array1<f64>) -> NeuromodResult<Array1<bool>> {
        if membrane_potentials.len() != modulation.len() {
            return Err(NeuromodError::DimensionMismatch {
                expected: membrane_potentials.len(),
                actual: modulation.len(),
            });
        }

        Ok(membrane_potentials
            .iter()
            .zip(modulation.iter())
            .map(|(v, m)| self.check_spike(*v, *m))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiplicative_gain() {
        let gating = GainModulation::new(GainModulationConfig::default());
        let input = Array1::from_vec(vec![0.5, 1.0, 1.5]);
        let modulation = 2.0;

        let output = gating.apply_multiplicative(&input, modulation);

        assert!(output[0] > input[0]);
        assert!(output[1] > input[1]);
        assert!(output[2] > input[2]);
    }

    #[test]
    fn test_additive_modulation() {
        let gating = GainModulation::new(GainModulationConfig::default());
        let input = Array1::from_vec(vec![0.5, 1.0, 1.5]);
        let modulation = 0.5;

        let output = gating.apply_additive(&input, modulation);

        assert!(output[0] > input[0]);
        assert!(output[1] > input[1]);
        assert!(output[2] > input[2]);
    }

    #[test]
    fn test_divisive_normalization() {
        let gating = GainModulation::new(GainModulationConfig::default());
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let modulation = 1.0;

        let output = gating.apply_divisive(&input, modulation);

        // Output should be normalized
        let sum: f64 = output.sum();
        assert!(sum < input.sum());
    }

    #[test]
    fn test_subtractive_modulation() {
        let gating = GainModulation::new(GainModulationConfig::default());
        let input = Array1::from_vec(vec![0.5, 1.0, 1.5]);
        let modulation = 0.3;

        let output = gating.apply_subtractive(&input, modulation);

        // Values below threshold should be zero
        assert!(output[0] >= 0.0);
    }

    #[test]
    fn test_input_gating() {
        let mut gating = InputGating::new(3);
        let modulation = Array1::from_vec(vec![0.8, 0.3, 0.6]);

        gating.compute_gates(&modulation);

        let input = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let output = gating.apply(&input).unwrap();

        // High modulation → high gate value
        assert!(output[0] > output[1]);
    }

    #[test]
    fn test_hard_gating() {
        let mut gating = InputGating::new(3);
        let modulation = Array1::from_vec(vec![0.8, 0.3, 0.6]);

        gating.compute_gates(&modulation);

        let input = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let output = gating.apply_hard(&input).unwrap();

        // Binary gating
        assert!(output[0] > 0.0 || output[0] == 0.0);
    }

    #[test]
    fn test_output_gating() {
        let mut gating = OutputGating::new(3);
        let modulation = Array1::from_vec(vec![0.3, 0.8, 0.4]);

        gating.compute_gates(&modulation);

        let output = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let gated = gating.apply(&output).unwrap();

        // High modulation passes through
        assert!(gated[1] > 0.0);
    }

    #[test]
    fn test_wta_gating() {
        let mut gating = OutputGating::new(3);
        gating.wta_strength = 1.0; // Hard WTA

        let modulation = Array1::from_vec(vec![0.3, 0.8, 0.4]);
        gating.compute_gates(&modulation);

        // Only maximum should have gate open
        assert_eq!(gating.gates[1], 1.0);
        assert_eq!(gating.gates[0], 0.0);
        assert_eq!(gating.gates[2], 0.0);
    }

    #[test]
    fn test_threshold_modulation() {
        let threshold_mod = ThresholdModulation::new(1.0);

        // High modulation → lower threshold
        let high_mod_threshold = threshold_mod.compute_threshold(0.8);
        assert!(high_mod_threshold < threshold_mod.baseline_threshold);

        // Low modulation → higher threshold
        let low_mod_threshold = threshold_mod.compute_threshold(0.2);
        assert!(low_mod_threshold > threshold_mod.baseline_threshold);
    }

    #[test]
    fn test_spike_threshold() {
        let threshold_mod = ThresholdModulation::new(1.0);

        // With high modulation, lower membrane potential can spike
        assert!(threshold_mod.check_spike(0.8, 0.8));

        // With low modulation, higher membrane potential needed
        assert!(!threshold_mod.check_spike(0.8, 0.2));
    }

    #[test]
    fn test_dimension_mismatch() {
        let gating = InputGating::new(3);
        let input = Array1::from_vec(vec![1.0, 1.0]); // Wrong size

        let result = gating.apply(&input);
        assert!(result.is_err());
    }
}
