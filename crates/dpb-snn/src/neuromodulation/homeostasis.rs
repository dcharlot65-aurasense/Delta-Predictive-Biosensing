//! Homeostatic Plasticity Mechanisms
//!
//! This module implements homeostatic mechanisms that maintain network stability:
//! - Firing rate homeostasis
//! - Synaptic scaling
//! - Intrinsic plasticity
//! - Metaplasticity (plasticity of plasticity)
//! - Sleep-dependent consolidation

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Homeostatic plasticity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostaticConfig {
    /// Target firing rate (Hz)
    pub target_rate: f64,
    /// Time constant for homeostasis (ms)
    pub tau_homeostasis: f64,
    /// Synaptic scaling rate
    pub scaling_rate: f64,
    /// Intrinsic plasticity rate
    pub intrinsic_rate: f64,
    /// Enable metaplasticity
    pub use_metaplasticity: bool,
    /// Metaplasticity time constant (ms)
    pub tau_metaplasticity: f64,
}

impl Default for HomeostaticConfig {
    fn default() -> Self {
        Self {
            target_rate: 5.0,
            tau_homeostasis: 10000.0,
            scaling_rate: 0.0001,
            intrinsic_rate: 0.00001,
            use_metaplasticity: true,
            tau_metaplasticity: 50000.0,
        }
    }
}

/// Firing Rate Homeostasis
///
/// Maintains average firing rate near target through negative feedback
/// mechanisms including synaptic scaling and intrinsic plasticity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiringRateHomeostasis {
    /// Target firing rate (Hz)
    pub target_rate: f64,
    /// Current average firing rate (Hz)
    pub current_rate: f64,
    /// Time constant (ms)
    pub tau: f64,
    /// Firing rate history (for moving average)
    pub rate_history: Vec<f64>,
    /// Maximum history length
    pub max_history: usize,
}

impl FiringRateHomeostasis {
    /// Create new firing rate homeostasis
    pub fn new(target_rate: f64, tau: f64) -> Self {
        Self {
            target_rate,
            current_rate: target_rate,
            tau,
            rate_history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Update average firing rate
    pub fn update(&mut self, dt: f64, instantaneous_rate: f64) {
        // Exponential moving average
        let alpha = dt / self.tau;
        self.current_rate = (1.0 - alpha) * self.current_rate + alpha * instantaneous_rate;

        // Add to history
        if self.rate_history.len() >= self.max_history {
            self.rate_history.remove(0);
        }
        self.rate_history.push(instantaneous_rate);
    }

    /// Get homeostatic error (how far from target)
    pub fn get_error(&self) -> f64 {
        self.current_rate - self.target_rate
    }

    /// Get relative error (normalized by target)
    pub fn get_relative_error(&self) -> f64 {
        if self.target_rate > 0.0 {
            self.get_error() / self.target_rate
        } else {
            0.0
        }
    }

    /// Check if firing rate is within acceptable range
    pub fn is_stable(&self, tolerance: f64) -> bool {
        self.get_relative_error().abs() < tolerance
    }

    /// Get mean firing rate from history
    pub fn get_mean_rate(&self) -> f64 {
        if self.rate_history.is_empty() {
            self.current_rate
        } else {
            self.rate_history.iter().sum::<f64>() / self.rate_history.len() as f64
        }
    }
}

/// Synaptic Scaling
///
/// Global multiplicative scaling of synaptic weights to maintain
/// firing rate homeostasis while preserving relative weight structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticScaling {
    /// Scaling rate
    pub rate: f64,
    /// Current scaling factor
    pub scale_factor: f64,
    /// Minimum scaling factor
    pub min_scale: f64,
    /// Maximum scaling factor
    pub max_scale: f64,
}

impl SynapticScaling {
    /// Create new synaptic scaling
    pub fn new(rate: f64) -> Self {
        Self {
            rate,
            scale_factor: 1.0,
            min_scale: 0.1,
            max_scale: 10.0,
        }
    }

    /// Update scaling factor based on firing rate error
    pub fn update(&mut self, dt: f64, rate_error: f64) {
        // If firing too much → scale down synapses
        // If firing too little → scale up synapses
        let delta_scale = -self.rate * rate_error * dt;
        self.scale_factor += delta_scale;

        // Apply bounds
        self.scale_factor = self.scale_factor.max(self.min_scale).min(self.max_scale);
    }

    /// Apply scaling to weights
    pub fn apply(&self, weights: &Array2<f64>) -> Array2<f64> {
        weights.mapv(|w| w * self.scale_factor)
    }

    /// Apply scaling to weights (in-place)
    pub fn apply_inplace(&self, weights: &mut Array2<f64>) {
        weights.mapv_inplace(|w| w * self.scale_factor);
    }

    /// Get scaling factor for individual neuron
    pub fn get_neuron_scale(&self, firing_rate: f64, target_rate: f64) -> f64 {
        if firing_rate > 0.0 {
            target_rate / firing_rate
        } else {
            1.0
        }
    }

    /// Apply per-neuron scaling
    pub fn apply_per_neuron(
        &self,
        weights: &mut Array2<f64>,
        firing_rates: &Array1<f64>,
        target_rate: f64,
    ) {
        let (num_post, _num_pre) = weights.dim();

        for i in 0..num_post {
            let scale = self.get_neuron_scale(firing_rates[i], target_rate);
            for j in 0..weights.ncols() {
                weights[[i, j]] *= scale;
            }
        }
    }
}

/// Intrinsic Plasticity
///
/// Adjusts neuron's intrinsic excitability (threshold, leak, etc.)
/// to maintain target firing rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrinsicPlasticity {
    /// Plasticity rate
    pub rate: f64,
    /// Current threshold adjustments
    pub threshold_adjustments: Array1<f64>,
    /// Current leak adjustments
    pub leak_adjustments: Array1<f64>,
    /// Maximum adjustment
    pub max_adjustment: f64,
}

impl IntrinsicPlasticity {
    /// Create new intrinsic plasticity
    pub fn new(num_neurons: usize, rate: f64) -> Self {
        Self {
            rate,
            threshold_adjustments: Array1::zeros(num_neurons),
            leak_adjustments: Array1::zeros(num_neurons),
            max_adjustment: 0.5,
        }
    }

    /// Update intrinsic properties based on firing rate error
    pub fn update(&mut self, dt: f64, firing_rates: &Array1<f64>, target_rate: f64) {
        let errors = firing_rates.mapv(|r| r - target_rate);

        // If firing too much → increase threshold (decrease excitability)
        // If firing too little → decrease threshold (increase excitability)
        let threshold_updates = errors.mapv(|e| self.rate * e * dt);
        self.threshold_adjustments = &self.threshold_adjustments + &threshold_updates;

        // Apply bounds
        self.threshold_adjustments
            .mapv_inplace(|adj| adj.max(-self.max_adjustment).min(self.max_adjustment));
    }

    /// Get adjusted threshold for neuron
    pub fn get_adjusted_threshold(&self, neuron_idx: usize, base_threshold: f64) -> f64 {
        base_threshold + self.threshold_adjustments[neuron_idx]
    }

    /// Get all adjusted thresholds
    pub fn get_adjusted_thresholds(&self, base_threshold: f64) -> Array1<f64> {
        self.threshold_adjustments.mapv(|adj| base_threshold + adj)
    }

    /// Reset adjustments
    pub fn reset(&mut self) {
        self.threshold_adjustments.fill(0.0);
        self.leak_adjustments.fill(0.0);
    }
}

/// Metaplasticity
///
/// "Plasticity of plasticity" - modulates learning rates based on
/// recent activity to prevent runaway potentiation/depression and
/// maintain stable learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metaplasticity {
    /// Time constant (ms)
    pub tau: f64,
    /// Activity history for each synapse
    pub activity_history: Array2<f64>,
    /// Plasticity threshold
    pub plasticity_threshold: f64,
    /// Maximum learning rate modulation
    pub max_modulation: f64,
}

impl Metaplasticity {
    /// Create new metaplasticity
    pub fn new(shape: (usize, usize), tau: f64) -> Self {
        Self {
            tau,
            activity_history: Array2::zeros(shape),
            plasticity_threshold: 0.5,
            max_modulation: 2.0,
        }
    }

    /// Update activity history
    pub fn update(&mut self, dt: f64, synaptic_activity: &Array2<f64>) {
        let alpha = dt / self.tau;
        self.activity_history = &self.activity_history * (1.0 - alpha) + synaptic_activity * alpha;
    }

    /// Get learning rate modulation based on recent activity
    ///
    /// BCM-like rule: if recent activity is high, reduce plasticity
    /// to prevent runaway potentiation
    pub fn get_learning_rate_modulation(&self, synapse_activity: f64) -> f64 {
        let relative_activity = synapse_activity / (self.plasticity_threshold + 1e-6);

        // Sliding threshold: high recent activity → lower learning rate
        let modulation = 1.0 / (1.0 + relative_activity);

        modulation
            .max(1.0 / self.max_modulation)
            .min(self.max_modulation)
    }

    /// Get modulation for all synapses
    pub fn get_modulation_matrix(&self) -> Array2<f64> {
        self.activity_history
            .mapv(|activity| self.get_learning_rate_modulation(activity))
    }

    /// Apply modulation to learning rates
    pub fn modulate_learning_rates(&self, base_rates: &Array2<f64>) -> Array2<f64> {
        let modulation = self.get_modulation_matrix();
        base_rates * &modulation
    }
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
/// Sleep-dependent consolidation
///
/// Models memory consolidation during sleep-like states with
/// reduced activity and modified plasticity rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SleepPhase {
    /// Awake state
    Wake,
    /// Slow-wave sleep (consolidation)
    SlowWave,
    /// REM sleep (replay)
    REM,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepConsolidation {
    /// Current sleep phase
    pub phase: SleepPhase,
    /// Synaptic downscaling rate during sleep
    pub downscaling_rate: f64,
    /// Replay probability during REM
    pub replay_probability: f64,
    /// Time in current phase (ms)
    pub time_in_phase: f64,
}

impl Default for SleepConsolidation {
    fn default() -> Self {
        Self {
            phase: SleepPhase::Wake,
            downscaling_rate: 0.0001,
            replay_probability: 0.1,
            time_in_phase: 0.0,
        }
    }
}

impl SleepConsolidation {
    /// Create new sleep consolidation
    pub fn new() -> Self {
        Self::default()
    }

    /// Set sleep phase
    pub fn set_phase(&mut self, phase: SleepPhase) {
        if self.phase != phase {
            self.time_in_phase = 0.0;
        }
        self.phase = phase;
    }

    /// Update time in phase
    pub fn update(&mut self, dt: f64) {
        self.time_in_phase += dt;
    }

    /// Apply synaptic downscaling during sleep
    pub fn apply_downscaling(&self, weights: &Array2<f64>, dt: f64) -> Array2<f64> {
        match self.phase {
            SleepPhase::Wake => weights.clone(),
            SleepPhase::SlowWave => {
                // Downscale all synapses proportionally
                let scale = 1.0 - self.downscaling_rate * dt;
                weights.mapv(|w| w * scale)
            }
            SleepPhase::REM => weights.clone(),
        }
    }

    /// Check if should replay pattern
    pub fn should_replay(&self) -> bool {
        self.phase == SleepPhase::REM && rand::random::<f64>() < self.replay_probability
    }

    /// Get learning rate modulation for current phase
    pub fn get_learning_rate_modulation(&self) -> f64 {
        match self.phase {
            SleepPhase::Wake => 1.0,
            SleepPhase::SlowWave => 0.1, // Reduced plasticity
            SleepPhase::REM => 0.5,      // Moderate plasticity
        }
    }
}

/// Homeostatic Plasticity
///
/// Integrates multiple homeostatic mechanisms to maintain network stability
/// while allowing learning.
pub struct HomeostaticPlasticity {
    /// Configuration
    pub config: HomeostaticConfig,
    /// Firing rate homeostasis
    pub firing_rate: FiringRateHomeostasis,
    /// Synaptic scaling
    pub synaptic_scaling: SynapticScaling,
    /// Intrinsic plasticity
    pub intrinsic_plasticity: Option<IntrinsicPlasticity>,
    /// Metaplasticity
    pub metaplasticity: Option<Metaplasticity>,
    /// Sleep consolidation
    pub sleep: SleepConsolidation,
}

impl HomeostaticPlasticity {
    /// Create new homeostatic plasticity
    pub fn new(config: HomeostaticConfig) -> Self {
        Self {
            firing_rate: FiringRateHomeostasis::new(config.target_rate, config.tau_homeostasis),
            synaptic_scaling: SynapticScaling::new(config.scaling_rate),
            intrinsic_plasticity: None,
            metaplasticity: None,
            sleep: SleepConsolidation::new(),
            config,
        }
    }

    /// Initialize intrinsic plasticity
    pub fn with_intrinsic_plasticity(mut self, num_neurons: usize) -> Self {
        self.intrinsic_plasticity = Some(IntrinsicPlasticity::new(
            num_neurons,
            self.config.intrinsic_rate,
        ));
        self
    }

    /// Initialize metaplasticity
    pub fn with_metaplasticity(mut self, shape: (usize, usize)) -> Self {
        if self.config.use_metaplasticity {
            self.metaplasticity = Some(Metaplasticity::new(shape, self.config.tau_metaplasticity));
        }
        self
    }

    /// Update all homeostatic mechanisms
    pub fn update(&mut self, dt: f64, firing_rates: &Array1<f64>) {
        // Update firing rate tracking
        let mean_rate = firing_rates.mean().unwrap_or(0.0);
        self.firing_rate.update(dt, mean_rate);

        // Update synaptic scaling
        let rate_error = self.firing_rate.get_error();
        self.synaptic_scaling.update(dt, rate_error);

        // Update intrinsic plasticity
        if let Some(ref mut intrinsic) = self.intrinsic_plasticity {
            intrinsic.update(dt, firing_rates, self.config.target_rate);
        }

        // Update sleep consolidation
        self.sleep.update(dt);
    }

    /// Update synaptic scaling for network weights
    pub fn update_synaptic_scaling(&mut self, firing_rates: &Array1<f64>, dt: f64) {
        let mean_rate = firing_rates.mean().unwrap_or(0.0);
        let rate_error = mean_rate - self.config.target_rate;
        self.synaptic_scaling.update(dt, rate_error);
    }

    /// Apply homeostatic scaling to weights
    pub fn apply_to_weights(&self, weights: &Array2<f64>) -> Array2<f64> {
        let mut scaled_weights = self.synaptic_scaling.apply(weights);

        // Apply sleep-dependent downscaling
        scaled_weights = self.sleep.apply_downscaling(&scaled_weights, 1.0);

        scaled_weights
    }

    /// Get adjusted thresholds
    pub fn get_adjusted_thresholds(&self, base_threshold: f64) -> Option<Array1<f64>> {
        self.intrinsic_plasticity
            .as_ref()
            .map(|ip| ip.get_adjusted_thresholds(base_threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_firing_rate_homeostasis() {
        let mut homeostasis = FiringRateHomeostasis::new(5.0, 1000.0);

        // Simulate high firing rate
        for _ in 0..100 {
            homeostasis.update(10.0, 10.0);
        }

        assert!(homeostasis.current_rate > 5.0);
        assert!(homeostasis.get_error() > 0.0);
    }

    #[test]
    fn test_synaptic_scaling() {
        let mut scaling = SynapticScaling::new(0.001);
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // High firing rate → scale down
        scaling.update(1.0, 2.0);
        assert!(scaling.scale_factor < 1.0);

        let scaled = scaling.apply(&weights);
        assert!(scaled[[0, 0]] < weights[[0, 0]]);
    }

    #[test]
    fn test_intrinsic_plasticity() {
        let mut intrinsic = IntrinsicPlasticity::new(3, 0.0001);
        let firing_rates = Array1::from_vec(vec![10.0, 3.0, 5.0]);
        let target_rate = 5.0;

        intrinsic.update(1.0, &firing_rates, target_rate);

        // High firing rate → positive adjustment (increase threshold)
        assert!(intrinsic.threshold_adjustments[0] > 0.0);
        // Low firing rate → negative adjustment (decrease threshold)
        assert!(intrinsic.threshold_adjustments[1] < 0.0);
    }

    #[test]
    fn test_metaplasticity() {
        let mut metaplasticity = Metaplasticity::new((2, 2), 1000.0);
        let activity = Array2::from_shape_vec((2, 2), vec![0.1, 0.9, 0.3, 0.7]).unwrap();

        metaplasticity.update(10.0, &activity);

        // High activity → lower modulation (reduced learning rate)
        let modulation = metaplasticity.get_learning_rate_modulation(0.9);
        assert!(modulation < 1.0);
    }

    #[test]
    fn test_sleep_phases() {
        let mut sleep = SleepConsolidation::new();

        sleep.set_phase(SleepPhase::SlowWave);
        assert_eq!(sleep.phase, SleepPhase::SlowWave);

        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let downscaled = sleep.apply_downscaling(&weights, 100.0);

        // Weights should be downscaled during sleep
        assert!(downscaled[[0, 0]] < weights[[0, 0]]);
    }

    #[test]
    fn test_homeostatic_plasticity() {
        let config = HomeostaticConfig::default();
        let mut homeostasis = HomeostaticPlasticity::new(config);

        let firing_rates = Array1::from_vec(vec![10.0, 8.0, 12.0]);
        homeostasis.update(1.0, &firing_rates);

        // Should track firing rate
        assert!(homeostasis.firing_rate.current_rate > 0.0);
    }

    #[test]
    fn test_stability_check() {
        let mut homeostasis = FiringRateHomeostasis::new(5.0, 1000.0);
        homeostasis.current_rate = 5.1;

        assert!(homeostasis.is_stable(0.1));

        homeostasis.current_rate = 10.0;
        assert!(!homeostasis.is_stable(0.1));
    }
}
