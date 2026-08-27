//! Operation-level power models for neuromorphic computing primitives.
//!
//! This module provides fine-grained power estimation for individual
//! operations in neuromorphic systems: synaptic operations, MAC operations,
//! spike transmission, and neuron updates.

use super::{ModelStats, PowerEstimator, PowerMetrics};

/// Synaptic operation power estimator.
///
/// Models energy consumption for a single synaptic operation (synop):
/// spike arrival, weight lookup, and current injection.
///
/// Reference: Multiple neuromorphic platforms report 5-50 pJ/synop
#[derive(Debug, Clone)]
pub struct SynapticOpEstimator {
    /// Energy per synaptic operation in picojoules
    pub energy_per_synop_pj: f64,
    /// Weight memory access energy in picojoules
    pub weight_access_energy_pj: f64,
    /// Current injection energy in picojoules
    pub current_injection_energy_pj: f64,
}

impl Default for SynapticOpEstimator {
    fn default() -> Self {
        Self {
            energy_per_synop_pj: 10.0,          // Total: 10 pJ/synop
            weight_access_energy_pj: 3.0,       // Memory access: 3 pJ
            current_injection_energy_pj: 7.0,   // Analog integration: 7 pJ
        }
    }
}

impl SynapticOpEstimator {
    /// Creates an ultra-efficient synaptic estimator (digital)
    pub fn ultra_efficient() -> Self {
        Self {
            energy_per_synop_pj: 5.0,
            weight_access_energy_pj: 1.5,
            current_injection_energy_pj: 3.5,
        }
    }

    /// Creates an analog synaptic estimator (higher power but faster)
    pub fn analog() -> Self {
        Self {
            energy_per_synop_pj: 15.0,
            weight_access_energy_pj: 5.0,
            current_injection_energy_pj: 10.0,
        }
    }
}

impl PowerEstimator for SynapticOpEstimator {
    fn name(&self) -> &str {
        "Synaptic Operation"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Total number of synaptic operations
        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        // Total energy for all synaptic operations
        let total_energy_j = synops * self.energy_per_synop_pj * 1e-12;

        // Assume 1ms per timestep for time calculation
        let inference_time_s = (model.timesteps as f64) * 0.001;

        // Average power = energy / time
        let dynamic_power_mw = (total_energy_j / inference_time_s) * 1000.0;
        let static_power_mw = 0.1; // Minimal static power for synaptic circuits

        let total_power_mw = dynamic_power_mw + static_power_mw;
        let energy_per_inference_uj = total_energy_j * 1e6;

        let efficiency_tops_per_w = (synops / inference_time_s / 1e12) / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: synops as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        // Energy to process one spike across average fan-out
        self.energy_per_synop_pj * 1e-12
    }

    fn energy_per_synop(&self) -> f64 {
        self.energy_per_synop_pj * 1e-12
    }
}

/// Multiply-accumulate (MAC) operation power estimator.
///
/// Models energy for traditional MAC operations in ANNs.
///
/// Reference: Energy ranges from 1 pJ (advanced digital) to 100+ pJ (older tech)
#[derive(Debug, Clone)]
pub struct MACEstimator {
    /// Energy per MAC operation in picojoules
    pub energy_per_mac_pj: f64,
    /// Technology node in nanometers
    pub technology_nm: u32,
    /// Bit width for computations
    pub bit_width: usize,
}

impl Default for MACEstimator {
    fn default() -> Self {
        Self {
            energy_per_mac_pj: 20.0,  // 20 pJ for 16-bit at 28nm
            technology_nm: 28,
            bit_width: 16,
        }
    }
}

impl MACEstimator {
    /// Creates a 7nm advanced node estimator
    pub fn advanced_node() -> Self {
        Self {
            energy_per_mac_pj: 1.0,
            technology_nm: 7,
            bit_width: 8,
        }
    }

    /// Creates a high-precision estimator (FP32)
    pub fn high_precision() -> Self {
        Self {
            energy_per_mac_pj: 100.0,
            technology_nm: 28,
            bit_width: 32,
        }
    }

    /// Creates a low-precision estimator (INT4)
    pub fn low_precision() -> Self {
        Self {
            energy_per_mac_pj: 5.0,
            technology_nm: 16,
            bit_width: 4,
        }
    }
}

impl PowerEstimator for MACEstimator {
    fn name(&self) -> &str {
        "MAC Operation"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // For ANNs: MACs ≈ num_synapses per timestep
        // For SNNs: effective MACs = synops (spike-gated)
        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let macs = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        let total_energy_j = macs * self.energy_per_mac_pj * 1e-12;

        let inference_time_s = (model.timesteps as f64) * 0.001;
        let dynamic_power_mw = (total_energy_j / inference_time_s) * 1000.0;
        let static_power_mw = 0.5; // Small static power

        let total_power_mw = dynamic_power_mw + static_power_mw;
        let energy_per_inference_uj = total_energy_j * 1e6;

        let efficiency_tops_per_w = (macs / inference_time_s / 1e12) / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: macs as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        // MAC doesn't have spikes, return MAC energy
        self.energy_per_mac_pj * 1e-12
    }

    fn energy_per_synop(&self) -> f64 {
        self.energy_per_mac_pj * 1e-12
    }
}

/// Spike transmission power estimator.
///
/// Models energy for transmitting spikes across the network/NoC.
///
/// Reference: Spike transmission costs vary from 0.1 pJ (on-chip)
/// to 10+ pJ (inter-chip communication)
#[derive(Debug, Clone)]
pub struct SpikeTransmitEstimator {
    /// Energy per spike transmission in picojoules
    pub energy_per_spike_pj: f64,
    /// Communication type
    pub comm_type: CommunicationType,
}

/// How far a spike has to travel, which sets its transport energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommunicationType {
    /// On-chip local communication
    OnChipLocal,
    /// On-chip global (across chip)
    OnChipGlobal,
    /// Inter-chip communication
    InterChip,
}

impl Default for SpikeTransmitEstimator {
    fn default() -> Self {
        Self {
            energy_per_spike_pj: 1.0,
            comm_type: CommunicationType::OnChipLocal,
        }
    }
}

impl SpikeTransmitEstimator {
    /// Creates an on-chip local transmission estimator
    pub fn on_chip_local() -> Self {
        Self {
            energy_per_spike_pj: 0.5,
            comm_type: CommunicationType::OnChipLocal,
        }
    }

    /// Creates an on-chip global transmission estimator
    pub fn on_chip_global() -> Self {
        Self {
            energy_per_spike_pj: 2.0,
            comm_type: CommunicationType::OnChipGlobal,
        }
    }

    /// Creates an inter-chip transmission estimator
    pub fn inter_chip() -> Self {
        Self {
            energy_per_spike_pj: 10.0,
            comm_type: CommunicationType::InterChip,
        }
    }
}

impl PowerEstimator for SpikeTransmitEstimator {
    fn name(&self) -> &str {
        match self.comm_type {
            CommunicationType::OnChipLocal => "Spike Transmission (On-Chip Local)",
            CommunicationType::OnChipGlobal => "Spike Transmission (On-Chip Global)",
            CommunicationType::InterChip => "Spike Transmission (Inter-Chip)",
        }
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Total spikes generated
        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;

        // Average fan-out per neuron
        let avg_fanout = (model.num_synapses as f64) / (model.num_neurons as f64);

        // Total spike transmissions
        let total_transmissions = total_spikes * avg_fanout;

        let total_energy_j = total_transmissions * self.energy_per_spike_pj * 1e-12;

        let inference_time_s = (model.timesteps as f64) * 0.001;
        let dynamic_power_mw = (total_energy_j / inference_time_s) * 1000.0;
        let static_power_mw = 0.05; // Minimal static power for routing

        let total_power_mw = dynamic_power_mw + static_power_mw;
        let energy_per_inference_uj = total_energy_j * 1e6;

        // Efficiency based on spike transmissions
        let efficiency_tops_per_w = (total_transmissions / inference_time_s / 1e12) /
            (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: total_transmissions as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        self.energy_per_spike_pj * 1e-12
    }

    fn energy_per_synop(&self) -> f64 {
        // For spike transmission, synop is equivalent to spike
        self.energy_per_spike_pj * 1e-12
    }
}

/// Neuron update power estimator.
///
/// Models energy for updating neuron membrane potential and spike generation.
///
/// Reference: Neuron updates typically cost 0.1-5 pJ depending on complexity
#[derive(Debug, Clone)]
pub struct NeuronUpdateEstimator {
    /// Energy per neuron update in picojoules
    pub energy_per_update_pj: f64,
    /// Neuron model complexity (LIF, AdEx, Izhikevich, etc.)
    pub model_complexity: NeuronComplexity,
}

/// Neuron model complexity, which sets the per-update compute cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronComplexity {
    /// Simple LIF (Leaky Integrate-and-Fire)
    SimpleLIF,
    /// Adaptive LIF
    AdaptiveLIF,
    /// Izhikevich model
    Izhikevich,
    /// Hodgkin-Huxley (complex)
    HodgkinHuxley,
}

impl Default for NeuronUpdateEstimator {
    fn default() -> Self {
        Self {
            energy_per_update_pj: 1.0,
            model_complexity: NeuronComplexity::SimpleLIF,
        }
    }
}

impl NeuronUpdateEstimator {
    /// Creates a simple LIF neuron estimator
    pub fn simple_lif() -> Self {
        Self {
            energy_per_update_pj: 0.5,
            model_complexity: NeuronComplexity::SimpleLIF,
        }
    }

    /// Creates an adaptive LIF neuron estimator
    pub fn adaptive_lif() -> Self {
        Self {
            energy_per_update_pj: 1.5,
            model_complexity: NeuronComplexity::AdaptiveLIF,
        }
    }

    /// Creates an Izhikevich neuron estimator
    pub fn izhikevich() -> Self {
        Self {
            energy_per_update_pj: 3.0,
            model_complexity: NeuronComplexity::Izhikevich,
        }
    }

    /// Creates a Hodgkin-Huxley neuron estimator
    pub fn hodgkin_huxley() -> Self {
        Self {
            energy_per_update_pj: 10.0,
            model_complexity: NeuronComplexity::HodgkinHuxley,
        }
    }
}

impl PowerEstimator for NeuronUpdateEstimator {
    fn name(&self) -> &str {
        match self.model_complexity {
            NeuronComplexity::SimpleLIF => "Neuron Update (Simple LIF)",
            NeuronComplexity::AdaptiveLIF => "Neuron Update (Adaptive LIF)",
            NeuronComplexity::Izhikevich => "Neuron Update (Izhikevich)",
            NeuronComplexity::HodgkinHuxley => "Neuron Update (Hodgkin-Huxley)",
        }
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Every neuron updates every timestep
        let total_updates = (model.num_neurons as f64) * (model.timesteps as f64);

        let total_energy_j = total_updates * self.energy_per_update_pj * 1e-12;

        let inference_time_s = (model.timesteps as f64) * 0.001;
        let dynamic_power_mw = (total_energy_j / inference_time_s) * 1000.0;
        let static_power_mw = 0.02; // Minimal static power

        let total_power_mw = dynamic_power_mw + static_power_mw;
        let energy_per_inference_uj = total_energy_j * 1e6;

        let efficiency_tops_per_w = (total_updates / inference_time_s / 1e12) /
            (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: total_updates as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        // Neuron update happens regardless of spike
        // But spike detection adds minimal overhead
        self.energy_per_update_pj * 1e-12 * 1.1
    }

    fn energy_per_synop(&self) -> f64 {
        self.energy_per_update_pj * 1e-12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synaptic_op_estimator() {
        let estimator = SynapticOpEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert!(metrics.synops_per_inference > 0);
        assert_eq!(estimator.name(), "Synaptic Operation");
    }

    #[test]
    fn test_synaptic_variants() {
        let efficient = SynapticOpEstimator::ultra_efficient();
        let analog = SynapticOpEstimator::analog();

        assert!(analog.energy_per_synop_pj > efficient.energy_per_synop_pj);
    }

    #[test]
    fn test_mac_estimator() {
        let estimator = MACEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert_eq!(estimator.name(), "MAC Operation");
    }

    #[test]
    fn test_mac_technology_scaling() {
        let advanced = MACEstimator::advanced_node();
        let default = MACEstimator::default();

        // Advanced node should be more efficient
        assert!(advanced.energy_per_mac_pj < default.energy_per_mac_pj);
    }

    #[test]
    fn test_mac_precision_scaling() {
        let high = MACEstimator::high_precision();
        let low = MACEstimator::low_precision();

        // Higher precision should cost more energy
        assert!(high.energy_per_mac_pj > low.energy_per_mac_pj);
    }

    #[test]
    fn test_spike_transmit_estimator() {
        let estimator = SpikeTransmitEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert!(estimator.name().contains("Spike Transmission"));
    }

    #[test]
    fn test_spike_transmit_variants() {
        let local = SpikeTransmitEstimator::on_chip_local();
        let global = SpikeTransmitEstimator::on_chip_global();
        let inter = SpikeTransmitEstimator::inter_chip();

        // Inter-chip should cost more than global, which costs more than local
        assert!(inter.energy_per_spike_pj > global.energy_per_spike_pj);
        assert!(global.energy_per_spike_pj > local.energy_per_spike_pj);
    }

    #[test]
    fn test_neuron_update_estimator() {
        let estimator = NeuronUpdateEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        // Number of operations should equal neurons * timesteps
        assert_eq!(metrics.synops_per_inference, (stats.num_neurons * stats.timesteps) as u64);
    }

    #[test]
    fn test_neuron_complexity_scaling() {
        let lif = NeuronUpdateEstimator::simple_lif();
        let adaptive = NeuronUpdateEstimator::adaptive_lif();
        let izhikevich = NeuronUpdateEstimator::izhikevich();
        let hh = NeuronUpdateEstimator::hodgkin_huxley();

        // More complex models should cost more energy
        assert!(adaptive.energy_per_update_pj > lif.energy_per_update_pj);
        assert!(izhikevich.energy_per_update_pj > adaptive.energy_per_update_pj);
        assert!(hh.energy_per_update_pj > izhikevich.energy_per_update_pj);
    }

    #[test]
    fn test_spike_rate_impact() {
        let estimator = SynapticOpEstimator::default();
        // Reassigned below, so it stays mut.
        let mut stats = ModelStats {
            spike_rate: 0.01,
            ..Default::default()
        };
        let low_activity = estimator.estimate_inference(&stats);

        stats.spike_rate = 0.5;
        let high_activity = estimator.estimate_inference(&stats);

        // Higher spike rate should result in more operations and energy
        assert!(high_activity.synops_per_inference > low_activity.synops_per_inference);
        assert!(high_activity.energy_per_inference_uj > low_activity.energy_per_inference_uj);
    }
}
