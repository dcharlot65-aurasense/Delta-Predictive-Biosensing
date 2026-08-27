//! Neuromorphic chip power models.
//!
//! This module provides power estimation models for commercial and research
//! neuromorphic hardware platforms based on published literature and datasheets.

use super::{ModelStats, PowerEstimator, PowerMetrics};

/// SynSense Xylo power estimator.
///
/// Xylo is an ultra-low-power neuromorphic audio processor.
/// Reference: SynSense Xylo datasheet (2022)
/// Typical power: ~1 mW active, ~5 µW standby
#[derive(Debug, Clone)]
pub struct XyloEstimator {
    /// Base dynamic power in mW
    pub base_dynamic_power_mw: f64,
    /// Static power in mW
    pub static_power_mw: f64,
    /// Energy per synaptic operation in picojoules
    /// Energy per synaptic operation, in picojoules.
    pub energy_per_synop_pj: f64,
}

impl Default for XyloEstimator {
    fn default() -> Self {
        Self {
            base_dynamic_power_mw: 0.8,    // 0.8 mW typical active
            static_power_mw: 0.005,         // 5 µW standby
            energy_per_synop_pj: 5.0,       // ~5 pJ/synop
        }
    }
}

impl PowerEstimator for XyloEstimator {
    fn name(&self) -> &str {
        "SynSense Xylo"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        // Dynamic power scales with activity
        let dynamic_power_mw = self.base_dynamic_power_mw * (1.0 + model.spike_rate);

        // Total power
        let total_power_mw = dynamic_power_mw + self.static_power_mw;

        // Energy per inference (power * time)
        // Assume 1ms per timestep
        let inference_time_s = (model.timesteps as f64) * 0.001;
        let energy_per_inference_uj = total_power_mw * inference_time_s;

        // Efficiency in TOPS/W
        let ops_per_second = synops / inference_time_s;
        let efficiency_tops_per_w = (ops_per_second / 1e12) / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw: self.static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: synops as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        // Energy to transmit a spike (estimated)
        0.1e-12 // 0.1 pJ
    }

    fn energy_per_synop(&self) -> f64 {
        self.energy_per_synop_pj * 1e-12 // Convert pJ to J
    }
}

/// Intel Loihi power estimator.
///
/// Loihi is Intel's neuromorphic research chip.
/// Reference: Davies et al., "Loihi: A Neuromorphic Manycore Processor" (2018)
/// Typical power: 60-100 mW per core
#[derive(Debug, Clone)]
pub struct LoihiEstimator {
    /// Idle power drawn by one core, in milliwatts.
    pub base_power_per_core_mw: f64,
    /// Number of neuromorphic cores in the configuration.
    pub num_cores: usize,
    /// Energy per synaptic operation, in picojoules.
    pub energy_per_synop_pj: f64,
}

impl Default for LoihiEstimator {
    fn default() -> Self {
        Self {
            base_power_per_core_mw: 80.0,
            num_cores: 128,
            energy_per_synop_pj: 23.6, // 23.6 pJ/synop (from paper)
        }
    }
}

impl PowerEstimator for LoihiEstimator {
    fn name(&self) -> &str {
        "Intel Loihi"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Estimate number of cores needed
        let neurons_per_core = 1024;
        let cores_needed = ((model.num_neurons as f64) / (neurons_per_core as f64)).ceil() as usize;
        let active_cores = cores_needed.min(self.num_cores);

        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        // Dynamic power based on active cores and spike activity
        let activity_factor = 0.3 + 0.7 * model.spike_rate;
        let dynamic_power_mw = (active_cores as f64) * self.base_power_per_core_mw * activity_factor;

        // Static power (leakage) - approximately 10% of max power
        let static_power_mw = (active_cores as f64) * self.base_power_per_core_mw * 0.1;

        let total_power_mw = dynamic_power_mw + static_power_mw;

        // Assume 1ms per timestep
        let inference_time_s = (model.timesteps as f64) * 0.001;
        let energy_per_inference_uj = total_power_mw * inference_time_s;

        let ops_per_second = synops / inference_time_s;
        let efficiency_tops_per_w = (ops_per_second / 1e12) / (total_power_mw / 1000.0);

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
        1.0e-12 // 1 pJ per spike
    }

    fn energy_per_synop(&self) -> f64 {
        self.energy_per_synop_pj * 1e-12
    }
}

/// SpiNNaker power estimator.
///
/// SpiNNaker is a massively parallel neuromorphic computing platform.
/// Reference: Furber et al., "The SpiNNaker Project" (2014)
/// Typical power: 1W per chip (18 ARM cores)
#[derive(Debug, Clone)]
pub struct SpinnAkerEstimator {
    /// Power drawn by one chip, in watts.
    pub power_per_chip_w: f64,
    /// ARM cores on each chip.
    pub cores_per_chip: usize,
    /// Number of chips in the configuration.
    pub num_chips: usize,
}

impl Default for SpinnAkerEstimator {
    fn default() -> Self {
        Self {
            power_per_chip_w: 1.0,
            cores_per_chip: 18,
            num_chips: 1,
        }
    }
}

impl PowerEstimator for SpinnAkerEstimator {
    fn name(&self) -> &str {
        "SpiNNaker"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Estimate cores needed (approximately 1000 neurons per core)
        let neurons_per_core = 1000;
        let cores_needed = ((model.num_neurons as f64) / (neurons_per_core as f64)).ceil() as usize;
        let chips_needed = ((cores_needed as f64) / (self.cores_per_chip as f64)).ceil() as usize;
        let active_chips = chips_needed.min(self.num_chips).max(1);

        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        // Power scales with number of active chips
        let activity_factor = 0.4 + 0.6 * model.spike_rate;
        let total_power_mw = (active_chips as f64) * self.power_per_chip_w * 1000.0 * activity_factor;

        // SpiNNaker has significant static power
        let static_power_mw = total_power_mw * 0.2;
        let dynamic_power_mw = total_power_mw - static_power_mw;

        let inference_time_s = (model.timesteps as f64) * 0.001;
        let energy_per_inference_uj = total_power_mw * inference_time_s;

        let ops_per_second = synops / inference_time_s;
        let efficiency_tops_per_w = (ops_per_second / 1e12) / (total_power_mw / 1000.0);

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
        10.0e-12 // 10 pJ per spike (higher due to ARM processors)
    }

    fn energy_per_synop(&self) -> f64 {
        50.0e-12 // 50 pJ/synop
    }
}

/// IBM TrueNorth power estimator.
///
/// TrueNorth is IBM's neuromorphic chip with 1 million neurons.
/// Reference: Merolla et al., "A million spiking-neuron integrated circuit" (2014)
/// Typical power: 70 mW for full chip
#[derive(Debug, Clone)]
pub struct TrueNorthEstimator {
    /// Power drawn by a full chip, in milliwatts.
    pub base_power_mw: f64,
    /// Neurons available on one chip.
    pub neurons_per_chip: usize,
}

impl Default for TrueNorthEstimator {
    fn default() -> Self {
        Self {
            base_power_mw: 70.0,
            neurons_per_chip: 1_000_000,
        }
    }
}

impl PowerEstimator for TrueNorthEstimator {
    fn name(&self) -> &str {
        "IBM TrueNorth"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Calculate fraction of chip used
        let utilization = (model.num_neurons as f64) / (self.neurons_per_chip as f64);
        let utilization = utilization.min(1.0);

        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        // TrueNorth has very consistent power consumption
        let activity_factor = 0.6 + 0.4 * model.spike_rate;
        let total_power_mw = self.base_power_mw * utilization * activity_factor;

        // Very low static power
        let static_power_mw = total_power_mw * 0.05;
        let dynamic_power_mw = total_power_mw - static_power_mw;

        let inference_time_s = (model.timesteps as f64) * 0.001;
        let energy_per_inference_uj = total_power_mw * inference_time_s;

        let ops_per_second = synops / inference_time_s;
        let efficiency_tops_per_w = (ops_per_second / 1e12) / (total_power_mw / 1000.0);

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
        0.26e-12 // 0.26 pJ per spike (from paper)
    }

    fn energy_per_synop(&self) -> f64 {
        26.0e-12 // 26 pJ/synop
    }
}

/// BrainScaleS power estimator.
///
/// BrainScaleS is an analog neuromorphic system operating at accelerated time.
/// Reference: Schemmel et al., "A wafer-scale neuromorphic hardware system" (2010)
/// Typical power: 1-2 W per wafer module
#[derive(Debug, Clone)]
pub struct BrainScaleSEstimator {
    /// Power drawn by one wafer module, in watts.
    pub power_per_wafer_w: f64,
    /// Ratio of emulated time to wall-clock time. BrainScaleS runs in
    /// accelerated time, so a factor of 1000 means one second of modelled
    /// activity completes in a millisecond.
    pub speedup_factor: f64,
}

impl Default for BrainScaleSEstimator {
    fn default() -> Self {
        Self {
            power_per_wafer_w: 1.5,
            speedup_factor: 10000.0, // 10,000x faster than biological time
        }
    }
}

impl PowerEstimator for BrainScaleSEstimator {
    fn name(&self) -> &str {
        "BrainScaleS"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        // Analog system has relatively constant power
        let activity_factor = 0.5 + 0.5 * model.spike_rate;
        let total_power_mw = self.power_per_wafer_w * 1000.0 * activity_factor;

        // Analog circuits have higher leakage
        let static_power_mw = total_power_mw * 0.3;
        let dynamic_power_mw = total_power_mw - static_power_mw;

        // Account for speedup in time calculation
        let biological_time_s = (model.timesteps as f64) * 0.001;
        let inference_time_s = biological_time_s / self.speedup_factor;
        let energy_per_inference_uj = total_power_mw * inference_time_s;

        let ops_per_second = synops / inference_time_s;
        let efficiency_tops_per_w = (ops_per_second / 1e12) / (total_power_mw / 1000.0);

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
        5.0e-12 // 5 pJ per spike (analog)
    }

    fn energy_per_synop(&self) -> f64 {
        15.0e-12 // 15 pJ/synop
    }
}

/// BrainChip Akida power estimator.
///
/// Akida is a commercial neuromorphic processor for edge AI.
/// Reference: BrainChip Akida datasheet (2021)
/// Typical power: 10-20 mW for typical workloads
#[derive(Debug, Clone)]
pub struct AkidaEstimator {
    /// Idle power, in milliwatts.
    pub base_power_mw: f64,
    /// Power drawn at full utilisation, in milliwatts.
    pub max_power_mw: f64,
}

impl Default for AkidaEstimator {
    fn default() -> Self {
        Self {
            base_power_mw: 10.0,
            max_power_mw: 100.0,
        }
    }
}

impl PowerEstimator for AkidaEstimator {
    fn name(&self) -> &str {
        "BrainChip Akida"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        // Power scales with activity
        let activity_factor = model.spike_rate / 0.5; // Normalized to 50% activity
        let dynamic_power_mw = self.base_power_mw +
            (self.max_power_mw - self.base_power_mw) * activity_factor.min(1.0);

        let static_power_mw = self.base_power_mw * 0.1;
        let total_power_mw = dynamic_power_mw + static_power_mw;

        let inference_time_s = (model.timesteps as f64) * 0.001;
        let energy_per_inference_uj = total_power_mw * inference_time_s;

        let ops_per_second = synops / inference_time_s;
        let efficiency_tops_per_w = (ops_per_second / 1e12) / (total_power_mw / 1000.0);

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
        0.5e-12 // 0.5 pJ per spike
    }

    fn energy_per_synop(&self) -> f64 {
        10.0e-12 // 10 pJ/synop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xylo_estimator() {
        let estimator = XyloEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert!(metrics.total_power_mw < 10.0); // Should be very low power
        assert_eq!(estimator.name(), "SynSense Xylo");
    }

    #[test]
    fn test_loihi_estimator() {
        let estimator = LoihiEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 10.0);
        assert!(metrics.synops_per_inference > 0);
        assert_eq!(estimator.name(), "Intel Loihi");
    }

    #[test]
    fn test_spinnaker_estimator() {
        let estimator = SpinnAkerEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert_eq!(estimator.name(), "SpiNNaker");
    }

    #[test]
    fn test_truenorth_estimator() {
        let estimator = TrueNorthEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert!(metrics.total_power_mw < 100.0);
        assert_eq!(estimator.name(), "IBM TrueNorth");
    }

    #[test]
    fn test_brainscales_estimator() {
        let estimator = BrainScaleSEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        // Should have very fast inference due to speedup
        assert!(metrics.energy_per_inference_uj < 1.0);
        assert_eq!(estimator.name(), "BrainScaleS");
    }

    #[test]
    fn test_akida_estimator() {
        let estimator = AkidaEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert!(metrics.total_power_mw < 200.0);
        assert_eq!(estimator.name(), "BrainChip Akida");
    }

    #[test]
    fn test_spike_rate_scaling() {
        let estimator = XyloEstimator::default();
        let mut stats = ModelStats::default();

        stats.spike_rate = 0.01;
        let low_activity = estimator.estimate_inference(&stats);

        stats.spike_rate = 0.5;
        let high_activity = estimator.estimate_inference(&stats);

        // Higher spike rate should consume more power
        assert!(high_activity.total_power_mw > low_activity.total_power_mw);
    }
}
