//! Neuromodulation Systems for Spiking Neural Networks
//!
//! This module implements biologically-inspired neuromodulation systems that
//! regulate learning, attention, and plasticity in SNNs. Neuromodulators like
//! dopamine, acetylcholine, serotonin, and norepinephrine play crucial roles
//! in brain function by modulating synaptic transmission and plasticity.
//!
//! # Overview
//!
//! Neuromodulatory systems provide global and local signals that:
//! - Modulate synaptic plasticity (learning rate, eligibility)
//! - Gate information flow (attention, gain control)
//! - Regulate network dynamics (excitability, timing)
//! - Enable reward-based learning (reinforcement learning)
//! - Maintain homeostatic balance (stability, robustness)
//!
//! # Key Concepts
//!
//! ## Dopamine
//! - **Function**: Reward signaling, motivation, learning
//! - **Mechanism**: Reward prediction error (RPE) signals
//! - **Effects**: Modulates STDP, enables TD learning
//! - **Receptors**: D1 (excitatory), D2 (inhibitory)
//!
//! ## Acetylcholine
//! - **Function**: Attention, arousal, learning
//! - **Mechanism**: Gain modulation, noise reduction
//! - **Effects**: Increases learning rate, sharpens selectivity
//! - **Receptors**: Nicotinic (fast), muscarinic (slow)
//!
//! ## Serotonin
//! - **Function**: Mood regulation, temporal processing
//! - **Mechanism**: Alters time constants, modulates inhibition
//! - **Effects**: Patience, impulse control, timing
//!
//! ## Norepinephrine
//! - **Function**: Arousal, alertness, attention
//! - **Mechanism**: Network gain modulation
//! - **Effects**: Enhanced signal-to-noise, rapid adaptation
//!
//! # Examples
//!
//! ## Reward-Modulated STDP
//!
//! ```rust
//! use dpb_snn::neuromodulation::*;
//!
//! // Create dopamine system
//! let mut da_system = DopamineSystem::new(DopamineConfig::default());
//!
//! // Create reward-modulated STDP
//! let mut reward_stdp = RewardModulatedSTDP::new(
//!     0.01,  // learning rate
//!     20.0,  // eligibility trace time constant
//! );
//!
//! // Simulate reward prediction error
//! let reward = 1.0;
//! let predicted_value = 0.5;
//! let rpe = da_system.compute_rpe(reward, predicted_value);
//!
//! // Apply three-factor learning rule
//! # let pre_trace = 0.5;
//! # let post_trace = 0.5;
//! # let eligibility = 0.3;
//! let weight_update = reward_stdp.compute_update(
//!     pre_trace,
//!     post_trace,
//!     eligibility,
//!     rpe,
//! );
//! ```
//!
//! ## Attention Gating
//!
//! ```rust
//! use dpb_snn::neuromodulation::*;
//! use ndarray::Array1;
//!
//! // Create acetylcholine system
//! let mut ach_system = AcetylcholineSystem::new(AcetylcholineConfig::default());
//!
//! // Create gain modulation
//! let mut gating = GainModulation::new(GainModulationConfig::default());
//!
//! // Modulate input based on attention
//! # let input = Array1::from_vec(vec![0.5; 10]);
//! let attention_level = ach_system.get_concentration();
//! let modulated = gating.apply_multiplicative(&input, attention_level);
//! ```
//!
//! ## Homeostatic Plasticity
//!
//! ```rust
//! use dpb_snn::neuromodulation::*;
//!
//! // Create homeostatic plasticity mechanism
//! let mut homeostasis = HomeostaticPlasticity::new(HomeostaticConfig {
//!     target_rate: 5.0,  // target firing rate (Hz)
//!     tau_homeostasis: 10000.0,  // homeostatic time constant (ms)
//!     ..Default::default()
//! });
//!
//! // Update based on firing rates
//! # let firing_rates = ndarray::Array1::from_vec(vec![3.0; 10]);
//! homeostasis.update_synaptic_scaling(&firing_rates, 1.0);
//! ```

pub mod modulators;
pub mod dopamine;
pub mod acetylcholine;
pub mod reward;
pub mod gating;
pub mod homeostasis;
pub mod integration;

// Re-export commonly used types
pub use modulators::{
    Neuromodulator, NeuromodulatorType, ModulatorySystem,
    ModulatorConcentration, DiffusionModel, ReceptorBinding,
    Dopamine, Acetylcholine, Serotonin, Norepinephrine,
};

pub use dopamine::{
    DopamineSystem, DopamineConfig, RewardPredictionError,
    DopamineMode, ReceptorType as DopamineReceptorType,
    StrialRegion, VTAResponse, SNcResponse,
};

pub use acetylcholine::{
    AcetylcholineSystem, AcetylcholineConfig,
    ReceptorType as AChReceptorType, AttentionState,
    BasalForebrainRegion, ChAT,
};

pub use reward::{
    RewardModulatedSTDP, RewardSignal, EligibilityTrace,
    TemporalCreditAssignment, IntrinsicMotivation,
    RewardShaping, ThreeFactorRule,
};

pub use gating::{
    GainModulation, GainModulationConfig, ModulationType,
    InputGating, OutputGating, ThresholdModulation,
};

pub use homeostasis::{
    HomeostaticPlasticity, HomeostaticConfig, FiringRateHomeostasis,
    SynapticScaling, IntrinsicPlasticity, Metaplasticity,
    SleepConsolidation,
};

pub use integration::{
    ModulatoryNetwork, ModulatoryNetworkConfig,
    ModulatorInteraction, SpatialScope, TemporalCoordination,
    StateDependent, BrainState,
};

/// Common result type for neuromodulation operations
pub type NeuromodResult<T> = Result<T, NeuromodError>;

/// Errors that can occur in neuromodulation systems
#[derive(Debug, thiserror::Error)]
pub enum NeuromodError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Concentration out of bounds: {value}, valid range [{min}, {max}]")]
    ConcentrationOutOfBounds { value: f64, min: f64, max: f64 },

    #[error("Incompatible modulator types: {0}")]
    IncompatibleModulators(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid time constant: {0}")]
    InvalidTimeConstant(f64),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Ensure all modules are properly exported
        let _ = DopamineSystem::new(DopamineConfig::default());
        let _ = AcetylcholineSystem::new(AcetylcholineConfig::default());
        let _ = RewardModulatedSTDP::new(0.01, 20.0);
        let _ = GainModulation::new(GainModulationConfig::default());
        let _ = HomeostaticPlasticity::new(HomeostaticConfig::default());
    }
}
