//! Hardware constraints for neuromorphic platforms
//!
//! Defines hardware-specific constraints including neuron counts, synapse limits,
//! weight precision, delay ranges, and supported neuron models.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use crate::NeuronModel;

/// Complete hardware constraints for a neuromorphic platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConstraints {
    /// Neuron-related constraints
    pub neuron: NeuronConstraints,
    /// Synapse-related constraints
    pub synapse: SynapseConstraints,
    /// Supported neuron models
    pub supported_models: SupportedNeuronModels,
    /// Weight quantization
    pub weight_precision: WeightBitDepth,
    /// Delay constraints
    pub delay_range: DelayRange,
    /// Platform name
    pub platform_name: String,
}

impl HardwareConstraints {
    /// Create Loihi 1 constraints
    pub fn loihi_constraints() -> Self {
        Self {
            neuron: NeuronConstraints {
                max_neurons_per_core: 1024,
                max_neurons_total: 131_072, // 128 cores
                max_compartments_per_neuron: 1,
                supports_adaptation: true,
            },
            synapse: SynapseConstraints {
                max_synapses_per_neuron: 4096,
                max_fanin: 4096,
                max_fanout: None, // Limited by target fanin
                supports_plastic: true,
                supports_delays: true,
            },
            supported_models: SupportedNeuronModels {
                models: vec!["LIF".to_string(), "CUBA".to_string(), "ALIF".to_string()]
                    .into_iter()
                    .collect(),
                default_model: "LIF".to_string(),
            },
            weight_precision: WeightBitDepth::Bits8,
            delay_range: DelayRange {
                min_delay: 1,
                max_delay: 63,
                delay_resolution: 1,
            },
            platform_name: "Intel Loihi".to_string(),
        }
    }

    /// Create Loihi 2 constraints
    pub fn loihi2_constraints() -> Self {
        Self {
            neuron: NeuronConstraints {
                max_neurons_per_core: 1024,
                max_neurons_total: 1_048_576, // 1024 cores (8 chips)
                max_compartments_per_neuron: 8, // Multi-compartment support
                supports_adaptation: true,
            },
            synapse: SynapseConstraints {
                max_synapses_per_neuron: 8192,
                max_fanin: 8192,
                max_fanout: None,
                supports_plastic: true,
                supports_delays: true,
            },
            supported_models: SupportedNeuronModels {
                models: vec![
                    "LIF".to_string(),
                    "CUBA".to_string(),
                    "ALIF".to_string(),
                    "Izhikevich".to_string(),
                ]
                .into_iter()
                .collect(),
                default_model: "LIF".to_string(),
            },
            weight_precision: WeightBitDepth::Bits8,
            delay_range: DelayRange {
                min_delay: 1,
                max_delay: 127,
                delay_resolution: 1,
            },
            platform_name: "Intel Loihi 2".to_string(),
        }
    }

    /// Create SpiNNaker 1 constraints
    pub fn spinnaker_constraints() -> Self {
        Self {
            neuron: NeuronConstraints {
                max_neurons_per_core: 256, // Typical, not hard limit
                max_neurons_total: 1_000_000, // 48-chip board
                max_compartments_per_neuron: 1,
                supports_adaptation: true,
            },
            synapse: SynapseConstraints {
                max_synapses_per_neuron: 10_000, // Memory limited
                max_fanin: 10_000,
                max_fanout: Some(10_000),
                supports_plastic: true,
                supports_delays: true,
            },
            supported_models: SupportedNeuronModels {
                models: vec![
                    "LIF".to_string(),
                    "IF_curr_exp".to_string(),
                    "IF_cond_exp".to_string(),
                    "Izhikevich".to_string(),
                ]
                .into_iter()
                .collect(),
                default_model: "IF_curr_exp".to_string(),
            },
            weight_precision: WeightBitDepth::Bits16, // Software implementation
            delay_range: DelayRange {
                min_delay: 1,
                max_delay: 144,
                delay_resolution: 1,
            },
            platform_name: "SpiNNaker".to_string(),
        }
    }

    /// Create SpiNNaker 2 constraints
    pub fn spinnaker2_constraints() -> Self {
        Self {
            neuron: NeuronConstraints {
                max_neurons_per_core: 512,
                max_neurons_total: 10_000_000, // Large-scale system
                max_compartments_per_neuron: 1,
                supports_adaptation: true,
            },
            synapse: SynapseConstraints {
                max_synapses_per_neuron: 50_000,
                max_fanin: 50_000,
                max_fanout: Some(50_000),
                supports_plastic: true,
                supports_delays: true,
            },
            supported_models: SupportedNeuronModels {
                models: vec![
                    "LIF".to_string(),
                    "IF_curr_exp".to_string(),
                    "IF_cond_exp".to_string(),
                    "Izhikevich".to_string(),
                    "AdEx".to_string(),
                ]
                .into_iter()
                .collect(),
                default_model: "IF_curr_exp".to_string(),
            },
            weight_precision: WeightBitDepth::Bits32, // Higher precision
            delay_range: DelayRange {
                min_delay: 1,
                max_delay: 256,
                delay_resolution: 1,
            },
            platform_name: "SpiNNaker 2".to_string(),
        }
    }

    /// Create BrainScaleS 1 constraints (analog, limited resources)
    pub fn brainscales_constraints() -> Self {
        Self {
            neuron: NeuronConstraints {
                max_neurons_per_core: 512, // Per wafer module
                max_neurons_total: 512,
                max_compartments_per_neuron: 1,
                supports_adaptation: true,
            },
            synapse: SynapseConstraints {
                max_synapses_per_neuron: 14_336, // Crossbar limited
                max_fanin: 14_336,
                max_fanout: Some(14_336),
                supports_plastic: true, // Limited STDP
                supports_delays: false, // No explicit delays
            },
            supported_models: SupportedNeuronModels {
                models: vec!["AdEx".to_string()].into_iter().collect(),
                default_model: "AdEx".to_string(),
            },
            weight_precision: WeightBitDepth::Bits4, // Limited DAC resolution
            delay_range: DelayRange {
                min_delay: 0, // No delay support
                max_delay: 0,
                delay_resolution: 0,
            },
            platform_name: "BrainScaleS".to_string(),
        }
    }

    /// Create BrainScaleS 2 constraints
    pub fn brainscales2_constraints() -> Self {
        Self {
            neuron: NeuronConstraints {
                max_neurons_per_core: 512,
                max_neurons_total: 512,
                max_compartments_per_neuron: 1,
                supports_adaptation: true,
            },
            synapse: SynapseConstraints {
                max_synapses_per_neuron: 256, // Per neuron routing
                max_fanin: 256,
                max_fanout: Some(256),
                supports_plastic: true,
                supports_delays: false,
            },
            supported_models: SupportedNeuronModels {
                models: vec!["AdEx".to_string(), "LIF".to_string()]
                    .into_iter()
                    .collect(),
                default_model: "AdEx".to_string(),
            },
            weight_precision: WeightBitDepth::Bits6,
            delay_range: DelayRange {
                min_delay: 0,
                max_delay: 0,
                delay_resolution: 0,
            },
            platform_name: "BrainScaleS-2".to_string(),
        }
    }

    /// Validate network against constraints
    pub fn validate_network(
        &self,
        num_neurons: usize,
        num_synapses: usize,
        max_fanin: usize,
    ) -> Result<(), String> {
        // Check neuron count
        if num_neurons > self.neuron.max_neurons_total {
            return Err(format!(
                "Network has {} neurons, but {} only supports {} neurons",
                num_neurons, self.platform_name, self.neuron.max_neurons_total
            ));
        }

        // Check fanin
        if max_fanin > self.synapse.max_fanin {
            return Err(format!(
                "Network has max fanin of {}, but {} only supports {}",
                max_fanin, self.platform_name, self.synapse.max_fanin
            ));
        }

        // Check average synapses per neuron
        let avg_synapses = num_synapses / num_neurons.max(1);
        if avg_synapses > self.synapse.max_synapses_per_neuron {
            return Err(format!(
                "Network has average {} synapses/neuron, but {} only supports {}",
                avg_synapses, self.platform_name, self.synapse.max_synapses_per_neuron
            ));
        }

        Ok(())
    }

    /// Check if a neuron model is supported
    pub fn supports_model(&self, model: &str) -> bool {
        self.supported_models.models.contains(model)
    }

    /// Estimate number of cores needed
    pub fn estimate_cores(&self, num_neurons: usize) -> usize {
        num_neurons.div_ceil(self.neuron.max_neurons_per_core)
    }
}

/// Neuron-related constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronConstraints {
    /// Maximum neurons per core
    pub max_neurons_per_core: usize,
    /// Maximum total neurons
    pub max_neurons_total: usize,
    /// Maximum compartments per neuron (for multi-compartment models)
    pub max_compartments_per_neuron: usize,
    /// Supports adaptive threshold
    pub supports_adaptation: bool,
}

/// Synapse-related constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseConstraints {
    /// Maximum synapses per neuron
    pub max_synapses_per_neuron: usize,
    /// Maximum fanin (incoming connections)
    pub max_fanin: usize,
    /// Maximum fanout (outgoing connections)
    pub max_fanout: Option<usize>,
    /// Supports plastic synapses (STDP, etc.)
    pub supports_plastic: bool,
    /// Supports synaptic delays
    pub supports_delays: bool,
}

/// Weight quantization bit depth
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightBitDepth {
    /// 4-bit weights
    Bits4,
    /// 6-bit weights
    Bits6,
    /// 8-bit weights
    Bits8,
    /// 16-bit weights
    Bits16,
    /// 32-bit weights (float)
    Bits32,
}

impl WeightBitDepth {
    /// Get number of bits
    pub fn bits(&self) -> u8 {
        match self {
            Self::Bits4 => 4,
            Self::Bits6 => 6,
            Self::Bits8 => 8,
            Self::Bits16 => 16,
            Self::Bits32 => 32,
        }
    }

    /// Get range of representable values (for signed weights)
    pub fn range(&self) -> (i32, i32) {
        let bits = self.bits() as u32;
        let max = (1 << (bits - 1)) - 1;
        let min = -(1 << (bits - 1));
        (min, max)
    }

    /// Get scale factor for quantization
    pub fn scale_factor(&self) -> f32 {
        let (min, max) = self.range();
        (max - min) as f32
    }
}

/// Delay range constraints
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DelayRange {
    /// Minimum delay (timesteps)
    pub min_delay: usize,
    /// Maximum delay (timesteps)
    pub max_delay: usize,
    /// Delay resolution (timesteps)
    pub delay_resolution: usize,
}

impl DelayRange {
    /// Validate a delay value
    pub fn validate(&self, delay: usize) -> Result<(), String> {
        if delay < self.min_delay {
            return Err(format!("Delay {} is below minimum {}", delay, self.min_delay));
        }
        if delay > self.max_delay {
            return Err(format!("Delay {} exceeds maximum {}", delay, self.max_delay));
        }
        if self.delay_resolution > 1 && !delay.is_multiple_of(self.delay_resolution) {
            return Err(format!(
                "Delay {} is not a multiple of resolution {}",
                delay, self.delay_resolution
            ));
        }
        Ok(())
    }

    /// Quantize delay to nearest valid value
    pub fn quantize(&self, delay: f32) -> usize {
        let delay = delay.round() as usize;
        let delay = delay.clamp(self.min_delay, self.max_delay);

        if self.delay_resolution > 1 {
            // Round to nearest multiple of resolution
            ((delay + self.delay_resolution / 2) / self.delay_resolution) * self.delay_resolution
        } else {
            delay
        }
    }
}

/// Supported neuron models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportedNeuronModels {
    /// Set of supported model names
    pub models: HashSet<String>,
    /// Default model to use
    pub default_model: String,
}

impl SupportedNeuronModels {
    /// Check if a model is supported
    pub fn supports(&self, model: &str) -> bool {
        self.models.contains(model)
    }

    /// Get closest supported model (simple heuristic)
    pub fn closest_model(&self, requested: &str) -> String {
        if self.supports(requested) {
            return requested.to_string();
        }

        // Try to find similar model
        let requested_lower = requested.to_lowercase();
        for model in &self.models {
            if model.to_lowercase().contains(&requested_lower)
                || requested_lower.contains(&model.to_lowercase())
            {
                return model.clone();
            }
        }

        // Return default
        self.default_model.clone()
    }

    /// Map generic neuron model to hardware-specific model
    pub fn map_model(&self, generic: NeuronModel) -> String {
        match generic {
            NeuronModel::LIF => {
                if self.supports("LIF") {
                    "LIF".to_string()
                } else if self.supports("IF_curr_exp") {
                    "IF_curr_exp".to_string()
                } else {
                    self.default_model.clone()
                }
            }
            NeuronModel::AdaptiveLIF => {
                if self.supports("ALIF") {
                    "ALIF".to_string()
                } else if self.supports("AdEx") {
                    "AdEx".to_string()
                } else {
                    self.default_model.clone()
                }
            }
            NeuronModel::Izhikevich => {
                if self.supports("Izhikevich") {
                    "Izhikevich".to_string()
                } else {
                    self.default_model.clone()
                }
            }
            _ => self.default_model.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loihi_constraints() {
        let constraints = HardwareConstraints::loihi_constraints();
        assert_eq!(constraints.neuron.max_neurons_per_core, 1024);
        assert_eq!(constraints.synapse.max_fanin, 4096);
        assert_eq!(constraints.weight_precision.bits(), 8);
    }

    #[test]
    fn test_loihi2_improvements() {
        let loihi1 = HardwareConstraints::loihi_constraints();
        let loihi2 = HardwareConstraints::loihi2_constraints();

        // Loihi 2 has more neurons and synapses
        assert!(loihi2.neuron.max_neurons_total > loihi1.neuron.max_neurons_total);
        assert!(loihi2.synapse.max_fanin > loihi1.synapse.max_fanin);
        assert!(loihi2.neuron.max_compartments_per_neuron > loihi1.neuron.max_compartments_per_neuron);
    }

    #[test]
    fn test_validate_network() {
        let constraints = HardwareConstraints::loihi_constraints();

        // Valid network
        assert!(constraints.validate_network(1000, 10000, 100).is_ok());

        // Too many neurons
        assert!(constraints
            .validate_network(200_000, 1_000_000, 100)
            .is_err());

        // Too large fanin
        assert!(constraints.validate_network(1000, 10000, 5000).is_err());
    }

    #[test]
    fn test_weight_bit_depth() {
        assert_eq!(WeightBitDepth::Bits8.bits(), 8);
        assert_eq!(WeightBitDepth::Bits8.range(), (-128, 127));

        assert_eq!(WeightBitDepth::Bits4.bits(), 4);
        assert_eq!(WeightBitDepth::Bits4.range(), (-8, 7));
    }

    #[test]
    fn test_delay_range_validation() {
        let delay_range = DelayRange {
            min_delay: 1,
            max_delay: 63,
            delay_resolution: 1,
        };

        assert!(delay_range.validate(10).is_ok());
        assert!(delay_range.validate(0).is_err());
        assert!(delay_range.validate(100).is_err());
    }

    #[test]
    fn test_delay_quantization() {
        let delay_range = DelayRange {
            min_delay: 1,
            max_delay: 63,
            delay_resolution: 1,
        };

        assert_eq!(delay_range.quantize(10.3), 10);
        assert_eq!(delay_range.quantize(10.7), 11);
        assert_eq!(delay_range.quantize(0.5), 1); // Clamped to min
        assert_eq!(delay_range.quantize(100.0), 63); // Clamped to max
    }

    #[test]
    fn test_supported_models() {
        let constraints = HardwareConstraints::loihi_constraints();
        assert!(constraints.supports_model("LIF"));
        assert!(!constraints.supports_model("HodgkinHuxley"));

        let mapped = constraints.supported_models.map_model(NeuronModel::LIF);
        assert_eq!(mapped, "LIF");
    }

    #[test]
    fn test_estimate_cores() {
        let constraints = HardwareConstraints::loihi_constraints();

        assert_eq!(constraints.estimate_cores(1024), 1);
        assert_eq!(constraints.estimate_cores(1025), 2);
        assert_eq!(constraints.estimate_cores(2048), 2);
        assert_eq!(constraints.estimate_cores(2049), 3);
    }

    #[test]
    fn test_brainscales_analog_constraints() {
        let bs = HardwareConstraints::brainscales_constraints();

        // Analog hardware has lower weight precision
        assert_eq!(bs.weight_precision, WeightBitDepth::Bits4);

        // No delay support
        assert!(!bs.synapse.supports_delays);
        assert_eq!(bs.delay_range.max_delay, 0);

        // Limited neurons
        assert_eq!(bs.neuron.max_neurons_total, 512);
    }

    #[test]
    fn test_spinnaker_software_flexibility() {
        let spinnaker = HardwareConstraints::spinnaker_constraints();

        // Software implementation allows higher precision
        assert_eq!(spinnaker.weight_precision, WeightBitDepth::Bits16);

        // Supports many neuron models
        assert!(spinnaker.supports_model("IF_curr_exp"));
        assert!(spinnaker.supports_model("Izhikevich"));
    }
}
