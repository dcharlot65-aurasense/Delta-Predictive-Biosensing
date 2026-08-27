//! Cholinergic System Implementation
//!
//! This module implements the acetylcholine (ACh) neuromodulation system, including:
//! - Attention gating and arousal
//! - Learning rate modulation
//! - Memory consolidation effects
//! - Nicotinic and muscarinic receptor dynamics
//! - Basal forebrain modeling

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::modulators::{ModulatorySystem, ModulatorConcentration, Acetylcholine};
use super::{NeuromodError, NeuromodResult};

/// Acetylcholine receptor types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceptorType {
    /// Nicotinic receptors (ionotropic, fast)
    Nicotinic,
    /// Muscarinic receptors (metabotropic, slow)
    Muscarinic,
}

/// Attention state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionState {
    /// Low attention/arousal
    Low,
    /// Moderate attention
    Moderate,
    /// High attention/arousal
    High,
    /// Focused attention
    Focused,
}

/// Basal forebrain regions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BasalForebrainRegion {
    /// Medial septum (hippocampal projections)
    MedialSeptum,
    /// Nucleus basalis (cortical projections)
    NucleusBasalis,
    /// Diagonal band (olfactory projections)
    DiagonalBand,
}

/// Choline acetyltransferase (ChAT) activity
///
/// Enzyme that synthesizes acetylcholine from choline and acetyl-CoA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChAT {
    /// Maximum synthesis rate
    pub vmax: f64,
    /// Michaelis constant (affinity)
    pub km: f64,
    /// Current synthesis rate
    pub rate: f64,
}

impl Default for ChAT {
    fn default() -> Self {
        Self {
            vmax: 1.0,
            km: 0.1,
            rate: 0.5,
        }
    }
}

impl ChAT {
    /// Compute synthesis rate using Michaelis-Menten kinetics
    pub fn compute_synthesis_rate(&mut self, substrate_concentration: f64) -> f64 {
        self.rate = self.vmax * substrate_concentration / (self.km + substrate_concentration);
        self.rate
    }
}

/// Acetylcholine system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcetylcholineConfig {
    /// Attention threshold for modulation
    pub attention_threshold: f64,
    /// Learning rate scaling factor
    pub learning_rate_scale: f64,
    /// Nicotinic receptor weight
    pub nicotinic_weight: f64,
    /// Muscarinic receptor weight
    pub muscarinic_weight: f64,
    /// Enable basal forebrain modeling
    pub use_basal_forebrain: bool,
    /// Signal-to-noise enhancement factor
    pub snr_enhancement: f64,
}

impl Default for AcetylcholineConfig {
    fn default() -> Self {
        Self {
            attention_threshold: 0.3,
            learning_rate_scale: 2.0,
            nicotinic_weight: 1.0,
            muscarinic_weight: 0.5,
            use_basal_forebrain: true,
            snr_enhancement: 1.5,
        }
    }
}

/// Acetylcholine System
///
/// Comprehensive model of the cholinergic system including:
/// - Attention and arousal modulation
/// - Learning rate enhancement
/// - Memory consolidation
/// - Nicotinic and muscarinic receptor effects
/// - Basal forebrain nucleus modeling
pub struct AcetylcholineSystem {
    /// Configuration
    pub config: AcetylcholineConfig,
    /// Acetylcholine concentration and dynamics
    pub acetylcholine: Acetylcholine,
    /// ChAT enzyme activity
    pub chat: ChAT,
    /// Current attention state
    pub attention_state: AttentionState,
    /// Basal forebrain activity by region
    pub basal_forebrain: [f64; 3],
}

impl AcetylcholineSystem {
    /// Create new acetylcholine system
    pub fn new(config: AcetylcholineConfig) -> Self {
        Self {
            config,
            acetylcholine: Acetylcholine::default(),
            chat: ChAT::default(),
            attention_state: AttentionState::Moderate,
            basal_forebrain: [0.5, 0.5, 0.5],
        }
    }

    /// Update attention state based on concentration
    pub fn update_attention_state(&mut self) {
        let conc = self.acetylcholine.concentration.concentration;
        let baseline = self.acetylcholine.concentration.baseline;
        let relative = conc / baseline;

        self.attention_state = if relative > 2.0 {
            AttentionState::Focused
        } else if relative > 1.5 {
            AttentionState::High
        } else if relative > 0.8 {
            AttentionState::Moderate
        } else {
            AttentionState::Low
        };
    }

    /// Get attention level (0-1)
    pub fn get_attention_level(&self) -> f64 {
        match self.attention_state {
            AttentionState::Low => 0.3,
            AttentionState::Moderate => 0.6,
            AttentionState::High => 0.85,
            AttentionState::Focused => 1.0,
        }
    }

    /// Modulate learning rate based on ACh level
    ///
    /// High ACh → increased learning rate (enhanced plasticity)
    /// Low ACh → decreased learning rate
    pub fn modulate_learning_rate(&self, base_learning_rate: f64) -> f64 {
        let attention = self.get_attention_level();
        let modulation = 1.0 + (attention - 0.5) * self.config.learning_rate_scale;
        base_learning_rate * modulation.max(0.1)
    }

    /// Apply attention gating to input
    ///
    /// Enhances signal-to-noise ratio by amplifying attended signals
    /// and suppressing noise/distractors
    pub fn apply_attention_gating(&self, input: &Array1<f64>, salience: &Array1<f64>) -> Array1<f64> {
        let attention = self.get_attention_level();
        let enhancement = self.config.snr_enhancement * attention;

        input
            .iter()
            .zip(salience.iter())
            .map(|(x, s)| {
                // Attended inputs (high salience) are enhanced
                // Unattended inputs are suppressed
                x * (1.0 + enhancement * s)
            })
            .collect()
    }

    /// Get nicotinic receptor effect (fast, phasic)
    ///
    /// Nicotinic receptors mediate fast excitation and attention
    pub fn get_nicotinic_effect(&self) -> f64 {
        self.config.nicotinic_weight * self.acetylcholine.nicotinic_receptors.occupancy
    }

    /// Get muscarinic receptor effect (slow, tonic)
    ///
    /// Muscarinic receptors mediate slow modulation and memory
    pub fn get_muscarinic_effect(&self) -> f64 {
        self.config.muscarinic_weight * self.acetylcholine.muscarinic_receptors.occupancy
    }

    /// Get combined receptor effect
    pub fn get_receptor_effect(&self) -> f64 {
        self.get_nicotinic_effect() + self.get_muscarinic_effect()
    }

    /// Update basal forebrain region
    pub fn update_basal_forebrain(&mut self, region: BasalForebrainRegion, activity: f64) {
        let idx = match region {
            BasalForebrainRegion::MedialSeptum => 0,
            BasalForebrainRegion::NucleusBasalis => 1,
            BasalForebrainRegion::DiagonalBand => 2,
        };
        self.basal_forebrain[idx] = activity.clamp(0.0, 1.0);
    }

    /// Get region-specific ACh release
    pub fn get_regional_release(&self, region: BasalForebrainRegion) -> f64 {
        let idx = match region {
            BasalForebrainRegion::MedialSeptum => 0,
            BasalForebrainRegion::NucleusBasalis => 1,
            BasalForebrainRegion::DiagonalBand => 2,
        };
        self.basal_forebrain[idx] * self.get_concentration()
    }

    /// Modulate memory consolidation
    ///
    /// ACh has a U-shaped effect on memory:
    /// - Moderate ACh during encoding enhances learning
    /// - Low ACh during consolidation prevents interference
    pub fn get_consolidation_factor(&self, phase: ConsolidationPhase) -> f64 {
        let conc = self.acetylcholine.concentration.relative_concentration();

        // `conc` is RELATIVE to the tonic baseline: 1.0 at rest, above 1.0 when
        // elevated, and unbounded above. The other two arms previously used
        // constants chosen for an absolute 0-1 concentration, which does not
        // hold here -- `1.0 - conc.min(0.8)` saturated for any relative level
        // at or above 0.8, so the consolidation factor was pinned at 0.2 for
        // every concentration at or above rest and could not express "low ACh
        // favours consolidation" at all. Each arm below is therefore written
        // against the ratio directly, and each is neutral (1.0) at baseline.
        match phase {
            ConsolidationPhase::Encoding => {
                // High ACh favours encoding: rises with concentration.
                conc
            }
            ConsolidationPhase::Consolidation => {
                // Low ACh favours consolidation -- the hyperbolic dual of the
                // encoding arm. Neutral at baseline, 2.0 as ACh goes to zero,
                // falling toward zero as ACh rises. Bounded, so no singularity.
                2.0 / (1.0 + conc)
            }
            ConsolidationPhase::Retrieval => {
                // Moderate ACh favours retrieval: peaks at the tonic baseline
                // and falls off in either direction.
                1.0 / (1.0 + (conc - 1.0).abs())
            }
        }
    }

    /// Apply cholinergic gain modulation
    ///
    /// Multiplicative gain on neural responses
    pub fn apply_gain_modulation(&self, activity: f64) -> f64 {
        let gain = 1.0 + self.get_receptor_effect();
        activity * gain
    }

    /// Reduce noise in signal
    ///
    /// ACh reduces noise and enhances signal quality
    pub fn reduce_noise(&self, signal: &Array1<f64>, noise_level: f64) -> Array1<f64> {
        let attention = self.get_attention_level();
        let noise_reduction = 1.0 - (attention * 0.5);

        signal.mapv(|x| {
            // High ACh → less noise
            x + noise_level * noise_reduction * (rand::random::<f64>() - 0.5)
        })
    }

    /// Update with attention/arousal signal
    pub fn update_with_attention(&mut self, dt: f64, attention_signal: f64) -> NeuromodResult<()> {
        // Convert attention signal to release
        let release = attention_signal * 0.01;

        // Update concentration
        self.acetylcholine.concentration.update(dt, release);

        // Update receptors
        let conc = self.acetylcholine.concentration.concentration;
        self.acetylcholine.nicotinic_receptors.update(dt, conc);
        self.acetylcholine.muscarinic_receptors.update(dt, conc);

        // Update attention state
        self.update_attention_state();

        Ok(())
    }
}

impl ModulatorySystem for AcetylcholineSystem {
    fn get_concentration(&self) -> f64 {
        self.acetylcholine.concentration.concentration
    }

    fn set_concentration(&mut self, concentration: f64) -> NeuromodResult<()> {
        if concentration < 0.0 || concentration > self.acetylcholine.concentration.max_concentration {
            return Err(NeuromodError::ConcentrationOutOfBounds {
                value: concentration,
                min: 0.0,
                max: self.acetylcholine.concentration.max_concentration,
            });
        }
        self.acetylcholine.concentration.concentration = concentration;
        // Setting a concentration outright must carry the receptors with
        // it; leaving them at their previous occupancy would report a
        // receptor-derived effect for a concentration that is no longer
        // present.
        self.acetylcholine.nicotinic_receptors.equilibrate(concentration);
        self.acetylcholine.muscarinic_receptors.equilibrate(concentration);
        self.update_attention_state();
        Ok(())
    }

    fn update(&mut self, dt: f64, release: f64) -> NeuromodResult<()> {
        self.acetylcholine.concentration.update(dt, release);

        let conc = self.acetylcholine.concentration.concentration;
        self.acetylcholine.nicotinic_receptors.update(dt, conc);
        self.acetylcholine.muscarinic_receptors.update(dt, conc);

        self.update_attention_state();

        Ok(())
    }

    fn modulator_type(&self) -> super::modulators::NeuromodulatorType {
        super::modulators::NeuromodulatorType::Acetylcholine
    }

    fn reset(&mut self) {
        self.acetylcholine.concentration.reset();
        self.acetylcholine.nicotinic_receptors.reset();
        self.acetylcholine.muscarinic_receptors.reset();
        self.attention_state = AttentionState::Moderate;
        self.basal_forebrain = [0.5, 0.5, 0.5];
        self.chat = ChAT::default();
    }
}

/// Memory consolidation phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsolidationPhase {
    /// Learning/encoding phase
    Encoding,
    /// Offline consolidation phase
    Consolidation,
    /// Memory retrieval phase
    Retrieval,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_state() {
        let mut system = AcetylcholineSystem::new(AcetylcholineConfig::default());

        // Low concentration
        system.set_concentration(0.05).unwrap();
        assert_eq!(system.attention_state, AttentionState::Low);

        // High concentration
        system.set_concentration(0.25).unwrap();
        assert_eq!(system.attention_state, AttentionState::Focused);
    }

    #[test]
    fn test_learning_rate_modulation() {
        let mut system = AcetylcholineSystem::new(AcetylcholineConfig::default());
        let base_lr = 0.01;

        // Low attention
        system.attention_state = AttentionState::Low;
        let modulated_lr = system.modulate_learning_rate(base_lr);
        assert!(modulated_lr < base_lr);

        // High attention
        system.attention_state = AttentionState::High;
        let modulated_lr = system.modulate_learning_rate(base_lr);
        assert!(modulated_lr > base_lr);
    }

    #[test]
    fn test_attention_gating() {
        let system = AcetylcholineSystem::new(AcetylcholineConfig::default());
        let input = Array1::from_vec(vec![0.5, 0.3, 0.7]);
        let salience = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let gated = system.apply_attention_gating(&input, &salience);

        // High salience items should be enhanced
        assert!(gated[0] > input[0]);
        assert!(gated[2] > input[2]);
        // Low salience item should be relatively unchanged
        assert!((gated[1] - input[1]).abs() < 0.1);
    }

    #[test]
    fn test_receptor_effects() {
        let mut system = AcetylcholineSystem::new(AcetylcholineConfig::default());

        // Increase concentration
        system.set_concentration(0.5).unwrap();

        // Update receptors
        system.acetylcholine.nicotinic_receptors.update(100.0, 0.5);
        system.acetylcholine.muscarinic_receptors.update(100.0, 0.5);

        let nicotinic = system.get_nicotinic_effect();
        let muscarinic = system.get_muscarinic_effect();

        assert!(nicotinic > 0.0);
        assert!(muscarinic > 0.0);
    }

    #[test]
    fn test_consolidation_phases() {
        let mut system = AcetylcholineSystem::new(AcetylcholineConfig::default());

        // Encoding: high ACh is good
        system.set_concentration(0.3).unwrap();
        let encoding_factor = system.get_consolidation_factor(ConsolidationPhase::Encoding);
        assert!(encoding_factor > 1.0);

        // Consolidation: low ACh is good
        system.set_concentration(0.05).unwrap();
        let consolidation_factor = system.get_consolidation_factor(ConsolidationPhase::Consolidation);
        assert!(consolidation_factor > 0.5);
    }

    #[test]
    fn test_basal_forebrain() {
        let mut system = AcetylcholineSystem::new(AcetylcholineConfig::default());

        system.update_basal_forebrain(BasalForebrainRegion::NucleusBasalis, 0.8);
        let release = system.get_regional_release(BasalForebrainRegion::NucleusBasalis);

        assert!(release > 0.0);
    }

    #[test]
    fn test_chat_synthesis() {
        let mut chat = ChAT::default();

        let rate1 = chat.compute_synthesis_rate(0.05);
        let rate2 = chat.compute_synthesis_rate(0.5);

        // Higher substrate → higher synthesis rate
        assert!(rate2 > rate1);
    }

    #[test]
    fn test_gain_modulation() {
        let mut system = AcetylcholineSystem::new(AcetylcholineConfig::default());
        system.set_concentration(0.3).unwrap();
        system.acetylcholine.nicotinic_receptors.occupancy = 0.5;

        let activity = 0.5;
        let modulated = system.apply_gain_modulation(activity);

        assert!(modulated > activity);
    }
}
