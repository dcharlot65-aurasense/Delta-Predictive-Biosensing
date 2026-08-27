//! Dopaminergic System Implementation
//!
//! This module implements the dopamine neuromodulation system, including:
//! - Reward prediction error (RPE) computation
//! - Temporal difference (TD) learning
//! - Phasic vs tonic dopamine dynamics
//! - D1/D2 receptor effects
//! - Striatal modulation
//! - VTA/SNc modeling

use serde::{Deserialize, Serialize};

use super::modulators::{Dopamine, ModulatorySystem};
use super::{NeuromodError, NeuromodResult};

/// Dopamine release mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DopamineMode {
    /// Tonic firing (baseline, ~5 Hz)
    Tonic,
    /// Phasic burst firing (>20 Hz)
    Phasic,
    /// Pause in firing (<5 Hz)
    Pause,
}

/// Dopamine receptor types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceptorType {
    /// D1-like receptors (D1, D5) - excitatory, cAMP increase
    D1,
    /// D2-like receptors (D2, D3, D4) - inhibitory, cAMP decrease
    D2,
}

/// Striatal regions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrialRegion {
    /// Dorsal striatum (motor, procedural learning)
    Dorsal,
    /// Ventral striatum (motivation, reward)
    Ventral,
}

/// Reward Prediction Error
///
/// RPE = r(t) + γ * V(t+1) - V(t)
/// where r is reward, γ is discount, V is value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardPredictionError {
    /// Current RPE value
    pub rpe: f64,
    /// Discount factor (0-1)
    pub gamma: f64,
    /// Learning rate for value updates
    pub alpha: f64,
    /// Current state value estimate
    pub value: f64,
    /// RPE history (for analysis)
    pub history: Vec<f64>,
}

impl RewardPredictionError {
    /// Create new RPE calculator
    pub fn new(gamma: f64, alpha: f64) -> Self {
        Self {
            rpe: 0.0,
            gamma,
            alpha,
            value: 0.0,
            history: Vec::new(),
        }
    }

    /// Compute RPE based on reward and next state value
    pub fn compute(&mut self, reward: f64, next_value: f64) -> f64 {
        self.rpe = reward + self.gamma * next_value - self.value;
        self.history.push(self.rpe);
        self.rpe
    }

    /// Update value estimate
    pub fn update_value(&mut self, reward: f64, next_value: f64) {
        let rpe = self.compute(reward, next_value);
        self.value += self.alpha * rpe;
    }

    /// Get sign of RPE (positive, negative, or zero)
    pub fn sign(&self) -> i8 {
        if self.rpe > 0.01 {
            1
        } else if self.rpe < -0.01 {
            -1
        } else {
            0
        }
    }
}

/// VTA (Ventral Tegmental Area) response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VTAResponse {
    /// Baseline firing rate (Hz)
    pub baseline_rate: f64,
    /// Burst firing rate (Hz)
    pub burst_rate: f64,
    /// Pause firing rate (Hz)
    pub pause_rate: f64,
    /// Current firing rate (Hz)
    pub current_rate: f64,
}

impl Default for VTAResponse {
    fn default() -> Self {
        Self {
            baseline_rate: 5.0,
            burst_rate: 20.0,
            pause_rate: 0.0,
            current_rate: 5.0,
        }
    }
}

impl VTAResponse {
    /// Update firing rate based on RPE
    pub fn update(&mut self, rpe: f64) {
        self.current_rate = if rpe > 0.1 {
            // Positive RPE → burst
            self.burst_rate
        } else if rpe < -0.1 {
            // Negative RPE → pause
            self.pause_rate
        } else {
            // No clear RPE → baseline
            self.baseline_rate
        };
    }

    /// Get dopamine release based on firing rate
    pub fn get_release(&self, dt: f64) -> f64 {
        // Release proportional to firing rate
        (self.current_rate / 1000.0) * dt
    }
}

/// SNc (Substantia Nigra pars compacta) response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SNcResponse {
    /// Baseline activity
    pub baseline: f64,
    /// Current activity level
    pub activity: f64,
    /// Time constant for activity decay
    pub tau: f64,
}

impl Default for SNcResponse {
    fn default() -> Self {
        Self {
            baseline: 5.0,
            activity: 5.0,
            tau: 100.0,
        }
    }
}

impl SNcResponse {
    /// Update activity
    pub fn update(&mut self, dt: f64, input: f64) {
        // Integrate input with decay
        self.activity += input;
        self.activity += (self.baseline - self.activity) * (dt / self.tau);
    }

    /// Get dopamine release
    pub fn get_release(&self, dt: f64) -> f64 {
        (self.activity / 1000.0) * dt
    }
}

/// Dopamine system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DopamineConfig {
    /// TD learning discount factor
    pub gamma: f64,
    /// Value learning rate
    pub alpha: f64,
    /// Phasic threshold multiplier
    pub phasic_threshold: f64,
    /// D1 receptor weight
    pub d1_weight: f64,
    /// D2 receptor weight
    pub d2_weight: f64,
    /// Enable VTA modeling
    pub use_vta: bool,
    /// Enable SNc modeling
    pub use_snc: bool,
}

impl Default for DopamineConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            alpha: 0.1,
            phasic_threshold: 1.5,
            d1_weight: 1.0,
            d2_weight: -0.5,
            use_vta: true,
            use_snc: false,
        }
    }
}

/// Dopamine System
///
/// Comprehensive model of the dopaminergic system including:
/// - Reward prediction error computation
/// - VTA/SNc nucleus modeling
/// - Phasic and tonic release dynamics
/// - D1/D2 receptor effects
/// - Striatal modulation
pub struct DopamineSystem {
    /// Configuration
    pub config: DopamineConfig,
    /// Dopamine concentration and dynamics
    pub dopamine: Dopamine,
    /// Reward prediction error
    pub rpe: RewardPredictionError,
    /// VTA response (reward signaling)
    pub vta: VTAResponse,
    /// SNc response (motor learning)
    pub snc: SNcResponse,
    /// Current dopamine mode
    pub mode: DopamineMode,
}

impl DopamineSystem {
    /// Create new dopamine system
    pub fn new(config: DopamineConfig) -> Self {
        Self {
            rpe: RewardPredictionError::new(config.gamma, config.alpha),
            dopamine: Dopamine::default(),
            vta: VTAResponse::default(),
            snc: SNcResponse::default(),
            mode: DopamineMode::Tonic,
            config,
        }
    }

    /// Compute reward prediction error
    /// Computes the RPE **without** learning from it.
    ///
    /// Use [`update_from_rpe`](Self::update_from_rpe) to advance the system;
    /// this is the read-only variant for inspecting what the error would be.
    pub fn compute_rpe(&mut self, reward: f64, next_value: f64) -> f64 {
        self.rpe.compute(reward, next_value)
    }

    /// Update dopamine release based on RPE
    pub fn update_from_rpe(&mut self, dt: f64, reward: f64, next_value: f64) -> NeuromodResult<()> {
        // Compute the RPE *and learn from it*.
        //
        // This previously called `compute_rpe`, which reads the error without
        // touching the value estimate, so `value` stayed 0.0 for the lifetime of
        // the system and `alpha` did nothing. A TD learner that never updates
        // its estimate has no predictions to err against: `rpe` collapsed to
        // `reward + gamma * next_value`, so the system could neither habituate
        // to a fully predicted reward nor pause when an expected reward was
        // withheld -- the two phenomena that define phasic dopamine coding
        // (Schultz, Dayan & Montague 1997).
        self.rpe.update_value(reward, next_value);
        let rpe = self.rpe.rpe;

        // Update VTA response
        if self.config.use_vta {
            self.vta.update(rpe);
        }

        // Determine release mode
        self.mode = if rpe > 0.1 {
            DopamineMode::Phasic
        } else if rpe < -0.1 {
            DopamineMode::Pause
        } else {
            DopamineMode::Tonic
        };

        // Compute release amount
        let release = if self.config.use_vta {
            self.vta.get_release(dt)
        } else {
            // Simple release proportional to RPE
            (rpe.max(0.0) / 100.0) * dt
        };

        // Update concentration
        self.dopamine.concentration.update(dt, release);

        // Update receptors
        let conc = self.dopamine.concentration.concentration;
        self.dopamine.d1_receptors.update(dt, conc);
        self.dopamine.d2_receptors.update(dt, conc);

        Ok(())
    }

    /// Update with direct release signal
    pub fn update_with_release(&mut self, dt: f64, release: f64) -> NeuromodResult<()> {
        self.dopamine.concentration.update(dt, release);

        let conc = self.dopamine.concentration.concentration;
        self.dopamine.d1_receptors.update(dt, conc);
        self.dopamine.d2_receptors.update(dt, conc);

        Ok(())
    }

    /// Get D1 receptor effect (facilitates LTP)
    pub fn get_d1_effect(&self) -> f64 {
        self.config.d1_weight * self.dopamine.d1_receptors.occupancy
    }

    /// Get D2 receptor effect (facilitates LTD)
    pub fn get_d2_effect(&self) -> f64 {
        self.config.d2_weight * self.dopamine.d2_receptors.occupancy
    }

    /// Get combined receptor effect
    pub fn get_receptor_effect(&self) -> f64 {
        self.get_d1_effect() + self.get_d2_effect()
    }

    /// Check if in phasic mode
    pub fn is_phasic(&self) -> bool {
        self.mode == DopamineMode::Phasic
    }

    /// Check if in pause mode
    pub fn is_pause(&self) -> bool {
        self.mode == DopamineMode::Pause
    }

    /// Get striatal modulation for a specific region
    pub fn get_striatal_modulation(&self, region: StrialRegion) -> f64 {
        match region {
            StrialRegion::Dorsal => {
                // Dorsal striatum: motor learning
                // Primarily influenced by SNc
                if self.config.use_snc {
                    self.snc.activity / 10.0
                } else {
                    self.get_concentration()
                }
            }
            StrialRegion::Ventral => {
                // Ventral striatum: reward processing
                // Primarily influenced by VTA
                if self.config.use_vta {
                    self.vta.current_rate / 10.0
                } else {
                    self.get_concentration()
                }
            }
        }
    }

    /// Get eligibility for learning (positive RPE enhances learning)
    pub fn get_learning_eligibility(&self) -> f64 {
        // Learning is enhanced when RPE is positive
        (self.rpe.rpe).clamp(0.0, 1.0)
    }

    /// Apply dopamine modulation to synaptic change
    pub fn modulate_plasticity(&self, base_weight_change: f64) -> f64 {
        // Dopamine gates plasticity
        // Positive dopamine (phasic) → enhance plasticity
        // Low dopamine (pause) → reduce plasticity
        let modulation = match self.mode {
            DopamineMode::Phasic => 2.0, // Enhance
            DopamineMode::Tonic => 1.0,  // Normal
            DopamineMode::Pause => 0.1,  // Suppress
        };

        // The receptor term keeps its SIGN. `d2_weight` is negative by
        // construction, so a D2-dominated state yields a net negative effect and
        // inverts the weight change -- D1 (Go/LTP) and D2 (NoGo/LTD) push in
        // opposite directions, which is the reason the two families carry
        // opposite-signed weights. Taking `.abs()` here made a D2-dominated
        // state potentiate exactly as strongly as a D1-dominated one, erasing
        // that opposition and leaving `d2_weight`'s sign with no effect on
        // anything.
        base_weight_change * modulation * self.get_receptor_effect()
    }
}

impl ModulatorySystem for DopamineSystem {
    fn get_concentration(&self) -> f64 {
        self.dopamine.concentration.concentration
    }

    fn set_concentration(&mut self, concentration: f64) -> NeuromodResult<()> {
        if concentration < 0.0 || concentration > self.dopamine.concentration.max_concentration {
            return Err(NeuromodError::ConcentrationOutOfBounds {
                value: concentration,
                min: 0.0,
                max: self.dopamine.concentration.max_concentration,
            });
        }
        self.dopamine.concentration.concentration = concentration;
        // Setting a concentration outright must carry the receptors with
        // it; leaving them at their previous occupancy would report a
        // receptor-derived effect for a concentration that is no longer
        // present.
        self.dopamine.d1_receptors.equilibrate(concentration);
        self.dopamine.d2_receptors.equilibrate(concentration);
        Ok(())
    }

    fn update(&mut self, dt: f64, release: f64) -> NeuromodResult<()> {
        self.update_with_release(dt, release)
    }

    fn modulator_type(&self) -> super::modulators::NeuromodulatorType {
        super::modulators::NeuromodulatorType::Dopamine
    }

    fn reset(&mut self) {
        self.dopamine.concentration.reset();
        self.dopamine.d1_receptors.reset();
        self.dopamine.d2_receptors.reset();
        self.rpe.value = 0.0;
        self.rpe.rpe = 0.0;
        self.vta = VTAResponse::default();
        self.snc = SNcResponse::default();
        self.mode = DopamineMode::Tonic;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpe_computation() {
        let mut rpe = RewardPredictionError::new(0.99, 0.1);

        // Test positive RPE (reward > expected)
        let reward = 1.0;
        let next_value = 0.5;
        let computed_rpe = rpe.compute(reward, next_value);
        assert!(computed_rpe > 0.0);
        assert_eq!(rpe.sign(), 1);

        // Test negative RPE (reward < expected)
        rpe.value = 1.0;
        let computed_rpe = rpe.compute(0.0, 0.0);
        assert!(computed_rpe < 0.0);
        assert_eq!(rpe.sign(), -1);
    }

    #[test]
    fn test_vta_response() {
        let mut vta = VTAResponse::default();

        // Test burst response to positive RPE
        vta.update(0.5);
        assert_eq!(vta.current_rate, vta.burst_rate);

        // Test pause response to negative RPE
        vta.update(-0.5);
        assert_eq!(vta.current_rate, vta.pause_rate);

        // Test baseline response to no RPE
        vta.update(0.0);
        assert_eq!(vta.current_rate, vta.baseline_rate);
    }

    #[test]
    fn test_dopamine_system() {
        let mut system = DopamineSystem::new(DopamineConfig::default());

        // Test with positive reward
        system.update_from_rpe(1.0, 1.0, 0.0).unwrap();
        assert!(system.is_phasic());
        assert!(system.get_concentration() > 0.1);

        // A dopamine pause comes from OMITTING AN EXPECTED reward, not from a
        // zero reward as such (Schultz 1997). Under the TD rule
        // `rpe = reward + gamma * next_value - value`, passing a high
        // `next_value` is a POSITIVE prediction error -- the successor state is
        // valuable -- so it produces a burst, not a pause.
        //
        // Build up an expectation first, then withhold the reward.
        for _ in 0..20 {
            system.update_from_rpe(1.0, 1.0, 0.0).unwrap();
        }
        assert!(
            !system.is_pause(),
            "a trained, rewarded system must not be in pause"
        );

        // Reward omitted, and no valuable successor state: rpe goes negative.
        system.update_from_rpe(1.0, 0.0, 0.0).unwrap();
        assert!(
            system.is_pause(),
            "omitting an expected reward must produce a pause"
        );
    }

    #[test]
    fn test_receptor_effects() {
        let mut system = DopamineSystem::new(DopamineConfig::default());

        // Increase concentration
        system.set_concentration(0.5).unwrap();

        // Update receptors
        system.dopamine.d1_receptors.update(100.0, 0.5);
        system.dopamine.d2_receptors.update(100.0, 0.5);

        let d1_effect = system.get_d1_effect();
        let d2_effect = system.get_d2_effect();

        assert!(d1_effect > 0.0);
        assert!(d2_effect < 0.0);
    }

    #[test]
    fn test_plasticity_modulation() {
        let mut system = DopamineSystem::new(DopamineConfig::default());
        let base_change = 0.1;

        system.dopamine.d1_receptors.occupancy = 0.5;

        // The contract is ORDERING ACROSS MODES at a fixed receptor state:
        // phasic potentiates more than tonic, tonic more than pause. Comparing
        // a single mode against the raw `base_change` instead is not a property
        // of the design -- it holds only when the receptor term happens to
        // exceed 0.5, and at exactly 0.5 the phasic gain of 2.0 cancels it to
        // give back `base_change` unchanged.
        system.mode = DopamineMode::Phasic;
        let phasic = system.modulate_plasticity(base_change);
        system.mode = DopamineMode::Tonic;
        let tonic = system.modulate_plasticity(base_change);
        system.mode = DopamineMode::Pause;
        let pause = system.modulate_plasticity(base_change);

        assert!(phasic > tonic, "phasic {phasic} must exceed tonic {tonic}");
        assert!(tonic > pause, "tonic {tonic} must exceed pause {pause}");
        assert!(
            pause < base_change,
            "pause {pause} must suppress below {base_change}"
        );

        // D2 dominance inverts the sign: dopamine acting through D2 drives
        // depression, not potentiation.
        system.mode = DopamineMode::Phasic;
        system.dopamine.d1_receptors.occupancy = 0.0;
        system.dopamine.d2_receptors.occupancy = 0.5;
        let d2_dominated = system.modulate_plasticity(base_change);
        assert!(
            d2_dominated < 0.0,
            "D2-dominated modulation must invert the weight change, got {d2_dominated}"
        );
    }

    #[test]
    fn test_striatal_modulation() {
        let mut system = DopamineSystem::new(DopamineConfig::default());

        system.update_from_rpe(1.0, 1.0, 0.0).unwrap();

        let dorsal_mod = system.get_striatal_modulation(StrialRegion::Dorsal);
        let ventral_mod = system.get_striatal_modulation(StrialRegion::Ventral);

        assert!(dorsal_mod > 0.0);
        assert!(ventral_mod > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut system = DopamineSystem::new(DopamineConfig::default());

        system.update_from_rpe(1.0, 1.0, 0.0).unwrap();
        system.reset();

        assert_eq!(system.mode, DopamineMode::Tonic);
        assert_eq!(system.rpe.value, 0.0);
        assert_eq!(system.rpe.rpe, 0.0);
    }

    /// The value estimate must actually track experience.
    ///
    /// Regression: `update_from_rpe` called the read-only `compute_rpe`, so
    /// `value` never moved off 0.0 and the prediction error was just the reward.
    /// The signature of phasic dopamine is that the response SHRINKS as a reward
    /// becomes predicted, and goes negative when a predicted reward is withheld;
    /// neither is possible if nothing is ever learned.
    #[test]
    fn test_value_estimate_learns() {
        let mut system = DopamineSystem::new(DopamineConfig::default());

        let first = system.compute_rpe(1.0, 0.0);
        system.update_from_rpe(1.0, 1.0, 0.0).unwrap();
        for _ in 0..30 {
            system.update_from_rpe(1.0, 1.0, 0.0).unwrap();
        }

        // Same reward, now predicted: the error must have shrunk toward zero.
        let predicted = system.compute_rpe(1.0, 0.0);
        assert!(
            predicted < first,
            "a repeated reward must become predicted: {first} -> {predicted}"
        );
        assert!(
            predicted.abs() < 0.2,
            "a fully predicted reward should elicit little error, got {predicted}"
        );

        // Withhold it: the error must go negative.
        let omitted = system.compute_rpe(0.0, 0.0);
        assert!(
            omitted < 0.0,
            "omitting a predicted reward must give a negative error, got {omitted}"
        );
    }
}
