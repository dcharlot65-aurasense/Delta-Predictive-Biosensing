//! Neuromodulation System Integration
//!
//! This module integrates multiple neuromodulatory systems to model
//! realistic brain dynamics including:
//! - Multi-modulator interactions
//! - Global vs local modulation
//! - Temporal coordination
//! - State-dependent processing

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::dopamine::{DopamineSystem, DopamineConfig};
use super::acetylcholine::{AcetylcholineSystem, AcetylcholineConfig};
use super::modulators::{Neuromodulator, NeuromodulatorType, ModulatorySystem};
use super::{NeuromodError, NeuromodResult};

/// Brain state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrainState {
    /// Alert/active state
    Alert,
    /// Focused attention
    Focused,
    /// Relaxed/idle state
    Relaxed,
    /// Sleep state
    Sleep,
    /// Stress/arousal state
    Stressed,
}

/// Spatial scope of modulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialScope {
    /// Global/broadcast modulation
    Global,
    /// Regional modulation
    Regional,
    /// Local/targeted modulation
    Local,
}

/// Modulator interaction type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModulatorInteraction {
    /// Modulators act independently
    Independent,
    /// Synergistic (multiplicative)
    Synergistic,
    /// Antagonistic (competitive)
    Antagonistic,
    /// Sequential (one enables/modulates the other)
    Sequential,
}

/// State-dependent processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDependent {
    /// Current brain state
    pub state: BrainState,
    /// State transition probabilities
    pub transition_probs: HashMap<(BrainState, BrainState), f64>,
    /// Time in current state (ms)
    pub time_in_state: f64,
}

impl Default for StateDependent {
    fn default() -> Self {
        let mut sd = Self {
            state: BrainState::Alert,
            transition_probs: HashMap::new(),
            time_in_state: 0.0,
        };

        // Set default transition probabilities
        sd.set_default_transitions();
        sd
    }
}

impl StateDependent {
    /// Create new state-dependent processor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default state transition probabilities
    fn set_default_transitions(&mut self) {
        use BrainState::*;

        // Alert → Focused
        self.transition_probs.insert((Alert, Focused), 0.3);
        // Alert → Relaxed
        self.transition_probs.insert((Alert, Relaxed), 0.2);

        // Focused → Alert
        self.transition_probs.insert((Focused, Alert), 0.4);
        // Focused → Relaxed
        self.transition_probs.insert((Focused, Relaxed), 0.1);

        // Relaxed → Alert
        self.transition_probs.insert((Relaxed, Alert), 0.3);
        // Relaxed → Sleep
        self.transition_probs.insert((Relaxed, Sleep), 0.1);

        // Sleep → Relaxed
        self.transition_probs.insert((Sleep, Relaxed), 0.5);
    }

    /// Update state
    pub fn update(&mut self, dt: f64) {
        self.time_in_state += dt;
    }

    /// Try state transition based on modulatory signals
    pub fn try_transition(&mut self, dopamine: f64, acetylcholine: f64) -> bool {
        use BrainState::*;

        let new_state = match self.state {
            Alert => {
                if acetylcholine > 0.7 {
                    Focused
                } else if acetylcholine < 0.3 {
                    Relaxed
                } else {
                    Alert
                }
            }
            Focused => {
                if acetylcholine < 0.5 {
                    Alert
                } else {
                    Focused
                }
            }
            Relaxed => {
                if acetylcholine > 0.6 || dopamine > 0.6 {
                    Alert
                } else if acetylcholine < 0.2 {
                    Sleep
                } else {
                    Relaxed
                }
            }
            Sleep => {
                if acetylcholine > 0.4 {
                    Relaxed
                } else {
                    Sleep
                }
            }
            Stressed => {
                if acetylcholine < 0.3 && dopamine < 0.3 {
                    Relaxed
                } else {
                    Stressed
                }
            }
        };

        if new_state != self.state {
            self.state = new_state;
            self.time_in_state = 0.0;
            true
        } else {
            false
        }
    }

    /// Get processing mode for current state
    pub fn get_processing_mode(&self) -> (f64, f64) {
        // Returns (learning_rate_modulation, exploration_factor)
        match self.state {
            BrainState::Alert => (1.0, 0.3),
            BrainState::Focused => (1.5, 0.1),
            BrainState::Relaxed => (0.5, 0.2),
            BrainState::Sleep => (0.1, 0.0),
            BrainState::Stressed => (0.7, 0.5),
        }
    }
}

/// Temporal coordination of modulatory signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCoordination {
    /// Oscillation frequency (Hz)
    pub frequency: f64,
    /// Current phase (0-2π)
    pub phase: f64,
    /// Amplitude
    pub amplitude: f64,
    /// Enable oscillatory modulation
    pub enable_oscillation: bool,
}

impl Default for TemporalCoordination {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            phase: 0.0,
            amplitude: 0.2,
            enable_oscillation: false,
        }
    }
}

impl TemporalCoordination {
    /// Update phase
    pub fn update(&mut self, dt: f64) {
        if self.enable_oscillation {
            let angular_freq = 2.0 * std::f64::consts::PI * self.frequency / 1000.0; // Convert Hz to rad/ms
            self.phase += angular_freq * dt;
            self.phase %= 2.0 * std::f64::consts::PI;
        }
    }

    /// Get oscillatory modulation
    pub fn get_modulation(&self) -> f64 {
        if self.enable_oscillation {
            1.0 + self.amplitude * self.phase.sin()
        } else {
            1.0
        }
    }

    /// Set theta oscillation (4-8 Hz)
    pub fn set_theta(&mut self) {
        self.frequency = 6.0;
        self.amplitude = 0.3;
        self.enable_oscillation = true;
    }

    /// Set gamma oscillation (30-80 Hz)
    pub fn set_gamma(&mut self) {
        self.frequency = 40.0;
        self.amplitude = 0.2;
        self.enable_oscillation = true;
    }
}

/// Modulatory network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModulatoryNetworkConfig {
    /// Enable dopamine system
    pub use_dopamine: bool,
    /// Enable acetylcholine system
    pub use_acetylcholine: bool,
    /// Enable serotonin system
    pub use_serotonin: bool,
    /// Enable norepinephrine system
    pub use_norepinephrine: bool,
    /// Modulator interaction type
    pub interaction_type: ModulatorInteraction,
    /// Enable state-dependent processing
    pub use_state_dependent: bool,
    /// Enable temporal coordination
    pub use_temporal_coordination: bool,
}

impl Default for ModulatoryNetworkConfig {
    fn default() -> Self {
        Self {
            use_dopamine: true,
            use_acetylcholine: true,
            use_serotonin: false,
            use_norepinephrine: false,
            interaction_type: ModulatorInteraction::Independent,
            use_state_dependent: true,
            use_temporal_coordination: false,
        }
    }
}

/// Integrated modulatory network
///
/// Coordinates multiple neuromodulatory systems to create
/// realistic brain dynamics with context-dependent processing.
pub struct ModulatoryNetwork {
    /// Configuration
    pub config: ModulatoryNetworkConfig,
    /// Dopamine system
    pub dopamine: Option<DopamineSystem>,
    /// Acetylcholine system
    pub acetylcholine: Option<AcetylcholineSystem>,
    /// Other modulators
    pub modulators: HashMap<NeuromodulatorType, Neuromodulator>,
    /// State-dependent processing
    pub state_dependent: Option<StateDependent>,
    /// Temporal coordination
    pub temporal_coordination: TemporalCoordination,
}

impl ModulatoryNetwork {
    /// Create new modulatory network
    pub fn new(config: ModulatoryNetworkConfig) -> Self {
        // Initialize systems based on config - capture flags before moving config
        let use_dopamine = config.use_dopamine;
        let use_acetylcholine = config.use_acetylcholine;
        let use_serotonin = config.use_serotonin;
        let use_norepinephrine = config.use_norepinephrine;
        let use_state_dependent = config.use_state_dependent;

        let mut network = Self {
            dopamine: None,
            acetylcholine: None,
            modulators: HashMap::new(),
            state_dependent: None,
            temporal_coordination: TemporalCoordination::default(),
            config,
        };

        // Initialize systems based on config
        if use_dopamine {
            network.dopamine = Some(DopamineSystem::new(DopamineConfig::default()));
        }

        if use_acetylcholine {
            network.acetylcholine = Some(AcetylcholineSystem::new(AcetylcholineConfig::default()));
        }

        if use_serotonin {
            network.modulators.insert(
                NeuromodulatorType::Serotonin,
                Neuromodulator::new(NeuromodulatorType::Serotonin),
            );
        }

        if use_norepinephrine {
            network.modulators.insert(
                NeuromodulatorType::Norepinephrine,
                Neuromodulator::new(NeuromodulatorType::Norepinephrine),
            );
        }

        if use_state_dependent {
            network.state_dependent = Some(StateDependent::new());
        }

        network
    }

    /// Update all modulatory systems
    pub fn update(&mut self, dt: f64) -> NeuromodResult<()> {
        // Update temporal coordination
        self.temporal_coordination.update(dt);

        // Update dopamine
        if let Some(ref mut da) = self.dopamine {
            da.update(dt, 0.0)?;
        }

        // Update acetylcholine
        if let Some(ref mut ach) = self.acetylcholine {
            ach.update(dt, 0.0)?;
        }

        // Update other modulators
        for modulator in self.modulators.values_mut() {
            modulator.concentration.update(dt, 0.0);
            modulator.update_receptors(dt);
        }

        // Update state-dependent processing
        // Get concentrations first to avoid borrow checker issues
        let da_conc = self.get_dopamine_concentration();
        let ach_conc = self.get_acetylcholine_concentration();

        if let Some(ref mut state) = self.state_dependent {
            state.update(dt);
            // Try state transition based on modulatory levels
            state.try_transition(da_conc, ach_conc);
        }

        Ok(())
    }

    /// Update with reward signal (for dopamine)
    pub fn update_with_reward(&mut self, dt: f64, reward: f64, next_value: f64) -> NeuromodResult<()> {
        if let Some(ref mut da) = self.dopamine {
            da.update_from_rpe(dt, reward, next_value)?;
        }
        self.update(dt)?;
        Ok(())
    }

    /// Update with attention signal (for acetylcholine)
    pub fn update_with_attention(&mut self, dt: f64, attention: f64) -> NeuromodResult<()> {
        if let Some(ref mut ach) = self.acetylcholine {
            ach.update_with_attention(dt, attention)?;
        }
        self.update(dt)?;
        Ok(())
    }

    /// Get dopamine concentration
    pub fn get_dopamine_concentration(&self) -> f64 {
        self.dopamine
            .as_ref()
            .map(|da| da.get_concentration())
            .unwrap_or(0.1)
    }

    /// Get acetylcholine concentration
    pub fn get_acetylcholine_concentration(&self) -> f64 {
        self.acetylcholine
            .as_ref()
            .map(|ach| ach.get_concentration())
            .unwrap_or(0.1)
    }

    /// Get combined modulatory effect
    pub fn get_combined_effect(&self) -> f64 {
        match self.config.interaction_type {
            ModulatorInteraction::Independent => {
                // Average effect
                let mut effects = Vec::new();
                if let Some(ref da) = self.dopamine {
                    effects.push(da.get_receptor_effect());
                }
                if let Some(ref ach) = self.acetylcholine {
                    effects.push(ach.get_receptor_effect());
                }

                if effects.is_empty() {
                    1.0
                } else {
                    effects.iter().sum::<f64>() / effects.len() as f64
                }
            }
            ModulatorInteraction::Synergistic => {
                // Multiplicative effect
                let mut effect = 1.0;
                if let Some(ref da) = self.dopamine {
                    effect *= 1.0 + da.get_receptor_effect();
                }
                if let Some(ref ach) = self.acetylcholine {
                    effect *= 1.0 + ach.get_receptor_effect();
                }
                effect
            }
            ModulatorInteraction::Antagonistic => {
                // Competitive effect
                let da_effect = self
                    .dopamine
                    .as_ref()
                    .map(|da| da.get_receptor_effect())
                    .unwrap_or(0.0);
                let ach_effect = self
                    .acetylcholine
                    .as_ref()
                    .map(|ach| ach.get_receptor_effect())
                    .unwrap_or(0.0);

                da_effect - ach_effect * 0.5
            }
            ModulatorInteraction::Sequential => {
                // ACh enables DA effect
                let da_effect = self
                    .dopamine
                    .as_ref()
                    .map(|da| da.get_receptor_effect())
                    .unwrap_or(0.0);
                let ach_gate = self
                    .acetylcholine
                    .as_ref()
                    .map(|ach| ach.get_attention_level())
                    .unwrap_or(1.0);

                da_effect * ach_gate
            }
        }
    }

    /// Get learning rate modulation
    pub fn get_learning_rate_modulation(&self) -> f64 {
        let mut modulation = 1.0;

        // Dopamine modulates based on reward
        if let Some(ref da) = self.dopamine {
            modulation *= 1.0 + da.get_learning_eligibility();
        }

        // Acetylcholine modulates based on attention
        if let Some(ref ach) = self.acetylcholine {
            let attention = ach.get_attention_level();
            modulation *= 0.5 + attention;
        }

        // State-dependent modulation
        if let Some(ref state) = self.state_dependent {
            let (state_mod, _) = state.get_processing_mode();
            modulation *= state_mod;
        }

        // Temporal coordination
        modulation *= self.temporal_coordination.get_modulation();

        modulation
    }

    /// Get exploration factor (for action selection)
    pub fn get_exploration_factor(&self) -> f64 {
        if let Some(ref state) = self.state_dependent {
            let (_, exploration) = state.get_processing_mode();
            exploration
        } else {
            0.2 // Default
        }
    }

    /// Get current brain state
    pub fn get_brain_state(&self) -> Option<BrainState> {
        self.state_dependent.as_ref().map(|s| s.state)
    }

    /// Set brain state
    pub fn set_brain_state(&mut self, state: BrainState) {
        if let Some(ref mut sd) = self.state_dependent {
            sd.state = state;
            sd.time_in_state = 0.0;
        }
    }

    /// Apply modulatory effects to synaptic plasticity
    pub fn modulate_weight_change(&self, base_weight_change: f64) -> f64 {
        let modulation = self.get_combined_effect();
        base_weight_change * modulation
    }

    /// Apply gating to input based on attention
    pub fn gate_input(&self, input: &Array1<f64>, salience: &Array1<f64>) -> Array1<f64> {
        if let Some(ref ach) = self.acetylcholine {
            ach.apply_attention_gating(input, salience)
        } else {
            input.clone()
        }
    }

    /// Reset all systems
    pub fn reset(&mut self) {
        if let Some(ref mut da) = self.dopamine {
            da.reset();
        }
        if let Some(ref mut ach) = self.acetylcholine {
            ach.reset();
        }
        for modulator in self.modulators.values_mut() {
            modulator.concentration.reset();
        }
        if let Some(ref mut state) = self.state_dependent {
            state.state = BrainState::Alert;
            state.time_in_state = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_dependent() {
        let mut state = StateDependent::new();
        assert_eq!(state.state, BrainState::Alert);

        // High acetylcholine → focused
        state.try_transition(0.5, 0.8);
        assert_eq!(state.state, BrainState::Focused);

        // Low acetylcholine → back to alert
        state.try_transition(0.5, 0.4);
        assert_eq!(state.state, BrainState::Alert);
    }

    #[test]
    fn test_temporal_coordination() {
        let mut coord = TemporalCoordination::default();
        coord.set_theta();

        coord.update(100.0);
        let mod1 = coord.get_modulation();

        coord.update(100.0);
        let mod2 = coord.get_modulation();

        // Modulation should vary with oscillation
        assert_ne!(mod1, mod2);
    }

    #[test]
    fn test_modulatory_network() {
        let config = ModulatoryNetworkConfig::default();
        let mut network = ModulatoryNetwork::new(config);

        assert!(network.dopamine.is_some());
        assert!(network.acetylcholine.is_some());

        network.update(1.0).unwrap();

        let da_conc = network.get_dopamine_concentration();
        assert!(da_conc > 0.0);
    }

    #[test]
    fn test_reward_update() {
        let config = ModulatoryNetworkConfig::default();
        let mut network = ModulatoryNetwork::new(config);

        network.update_with_reward(1.0, 1.0, 0.0).unwrap();

        let da_conc = network.get_dopamine_concentration();
        assert!(da_conc > 0.1);
    }

    #[test]
    fn test_attention_update() {
        let config = ModulatoryNetworkConfig::default();
        let mut network = ModulatoryNetwork::new(config);

        network.update_with_attention(1.0, 0.8).unwrap();

        let ach_conc = network.get_acetylcholine_concentration();
        assert!(ach_conc > 0.1);
    }

    #[test]
    fn test_combined_effect() {
        let config = ModulatoryNetworkConfig::default();
        let network = ModulatoryNetwork::new(config);

        let effect = network.get_combined_effect();
        assert!(effect > 0.0);
    }

    #[test]
    fn test_learning_rate_modulation() {
        let config = ModulatoryNetworkConfig::default();
        let network = ModulatoryNetwork::new(config);

        let modulation = network.get_learning_rate_modulation();
        assert!(modulation > 0.0);
    }

    #[test]
    fn test_synergistic_interaction() {
        let mut config = ModulatoryNetworkConfig::default();
        config.interaction_type = ModulatorInteraction::Synergistic;

        let mut network = ModulatoryNetwork::new(config);

        // Set high concentrations
        if let Some(ref mut da) = network.dopamine {
            da.set_concentration(0.5).unwrap();
        }
        if let Some(ref mut ach) = network.acetylcholine {
            ach.set_concentration(0.5).unwrap();
        }

        let effect = network.get_combined_effect();
        // Synergistic should amplify
        assert!(effect > 1.0);
    }

    #[test]
    fn test_brain_state_transition() {
        let config = ModulatoryNetworkConfig::default();
        let mut network = ModulatoryNetwork::new(config);

        network.set_brain_state(BrainState::Relaxed);
        assert_eq!(network.get_brain_state(), Some(BrainState::Relaxed));

        // High attention should transition to alert
        network.update_with_attention(1.0, 0.8).unwrap();
        network.update(100.0).unwrap();
    }

    #[test]
    fn test_input_gating() {
        let config = ModulatoryNetworkConfig::default();
        let mut network = ModulatoryNetwork::new(config);

        // Set high attention
        network.update_with_attention(1.0, 0.8).unwrap();

        let input = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        let salience = Array1::from_vec(vec![1.0, 0.0, 1.0]);

        let gated = network.gate_input(&input, &salience);

        // High salience items should be enhanced
        assert!(gated[0] > input[0]);
        assert!(gated[2] > input[2]);
    }
}
