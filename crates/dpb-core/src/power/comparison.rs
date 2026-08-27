//! Comparative power analysis utilities.
//!
//! This module provides tools for comparing power consumption across
//! different paradigms (SNN vs ANN) and analyzing the impact of
//! sparsity, dynamic activity, and static leakage.

use super::{ModelStats, PowerEstimator, PowerMetrics};

/// SNN vs ANN power comparison estimator.
///
/// Compares the power consumption of spiking neural networks against
/// traditional artificial neural networks for the same task.
///
/// Reference: SNNs can achieve 10-100× energy savings for sparse,
/// event-driven workloads due to temporal sparsity.
#[derive(Debug, Clone)]
pub struct SNNvsANNEstimator {
    /// Energy per MAC operation in ANN (picojoules)
    pub ann_energy_per_mac_pj: f64,
    /// Energy per synaptic operation in SNN (picojoules)
    pub snn_energy_per_synop_pj: f64,
    /// Equivalent MAC operations per SNN timestep
    pub timesteps_per_ann_inference: usize,
}

impl Default for SNNvsANNEstimator {
    fn default() -> Self {
        Self {
            ann_energy_per_mac_pj: 50.0,     // Conservative 16-bit MAC estimate
            snn_energy_per_synop_pj: 5.0,    // Efficient neuromorphic synop
            timesteps_per_ann_inference: 1,   // ANN is single forward pass
        }
    }
}

impl SNNvsANNEstimator {
    /// Estimates ANN power consumption for comparison
    pub fn estimate_ann(&self, model: &ModelStats) -> PowerMetrics {
        // ANN performs MACs for every synapse in a single forward pass
        let total_macs = model.num_synapses as f64;

        let total_energy_j = total_macs * self.ann_energy_per_mac_pj * 1e-12;

        // ANN inference is typically faster (no timesteps)
        let inference_time_s = 0.001; // 1ms for typical forward pass

        let dynamic_power_mw = (total_energy_j / inference_time_s) * 1000.0;
        let static_power_mw = 10.0; // Baseline static power for ANN accelerator

        let total_power_mw = dynamic_power_mw + static_power_mw;
        let energy_per_inference_uj = total_energy_j * 1e6;

        let ops_per_second = total_macs / inference_time_s;
        let efficiency_tops_per_w = (ops_per_second / 1e12) / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: total_macs as u64,
            efficiency_tops_per_w,
        }
    }

    /// Estimates SNN power consumption for comparison
    pub fn estimate_snn(&self, model: &ModelStats) -> PowerMetrics {
        // SNN only performs operations when spikes occur (temporal sparsity)
        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        // Dynamic energy from synaptic operations
        let dynamic_energy_j = synops * self.snn_energy_per_synop_pj * 1e-12;

        let inference_time_s = (model.timesteps as f64) * 0.001;

        // Very low static power for event-driven neuromorphic
        let static_power_mw = 0.01; // 10 µW static power
        let static_energy_j = (static_power_mw / 1000.0) * inference_time_s;

        let total_energy_j = dynamic_energy_j + static_energy_j;
        let energy_per_inference_uj = total_energy_j * 1e6;

        let dynamic_power_mw = (dynamic_energy_j / inference_time_s) * 1000.0;
        let total_power_mw = dynamic_power_mw + static_power_mw;

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

    /// Computes the energy efficiency ratio (SNN / ANN)
    pub fn efficiency_ratio(&self, model: &ModelStats) -> f64 {
        let ann_metrics = self.estimate_ann(model);
        let snn_metrics = self.estimate_snn(model);

        snn_metrics.energy_per_inference_uj / ann_metrics.energy_per_inference_uj
    }
}

impl PowerEstimator for SNNvsANNEstimator {
    fn name(&self) -> &str {
        "SNN vs ANN Comparison"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Returns SNN metrics by default
        self.estimate_snn(model)
    }

    fn energy_per_spike(&self) -> f64 {
        self.snn_energy_per_synop_pj * 1e-12
    }

    fn energy_per_synop(&self) -> f64 {
        self.snn_energy_per_synop_pj * 1e-12
    }
}

/// Sparsity-aware power estimator.
///
/// Accounts for both temporal sparsity (spike rate) and structural
/// sparsity (pruned connections) in power estimation.
///
/// Reference: Exploiting sparsity can reduce energy by 10-100×
#[derive(Debug, Clone)]
pub struct SparsityAwareEstimator {
    /// Base energy per operation (dense case) in picojoules
    pub base_energy_per_op_pj: f64,
    /// Overhead for sparsity detection/routing in picojoules
    pub sparsity_overhead_pj: f64,
    /// Structural sparsity (fraction of weights that are zero)
    pub structural_sparsity: f64,
}

impl Default for SparsityAwareEstimator {
    fn default() -> Self {
        Self {
            base_energy_per_op_pj: 10.0,
            sparsity_overhead_pj: 1.0,    // Small overhead for sparsity logic
            structural_sparsity: 0.0,      // No pruning by default
        }
    }
}

impl SparsityAwareEstimator {
    /// Creates an estimator with pruned network
    pub fn with_pruning(pruning_ratio: f64) -> Self {
        Self {
            base_energy_per_op_pj: 10.0,
            sparsity_overhead_pj: 1.0,
            structural_sparsity: pruning_ratio,
        }
    }

    /// Computes effective sparsity combining temporal and structural
    pub fn effective_sparsity(&self, temporal_sparsity: f64) -> f64 {
        // Combined sparsity: (1 - temporal) × (1 - structural)
        let active_temporal = temporal_sparsity;
        let active_structural = 1.0 - self.structural_sparsity;
        active_temporal * active_structural
    }
}

impl PowerEstimator for SparsityAwareEstimator {
    fn name(&self) -> &str {
        "Sparsity-Aware"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Temporal sparsity from spike rate
        let temporal_activity = model.spike_rate;

        // Effective activity considering both temporal and structural sparsity
        let effective_activity = self.effective_sparsity(temporal_activity);

        // Total potential operations
        let max_ops = (model.num_synapses as f64) * (model.timesteps as f64);

        // Actual operations performed
        let actual_ops = max_ops * effective_activity;

        // Energy savings from sparsity
        let compute_energy_j = actual_ops * self.base_energy_per_op_pj * 1e-12;

        // Overhead for managing sparsity (e.g., indexing, routing)
        let overhead_energy_j = max_ops * self.sparsity_overhead_pj * 1e-12 * 0.1; // 10% ops checked

        let total_energy_j = compute_energy_j + overhead_energy_j;

        let inference_time_s = (model.timesteps as f64) * 0.001;
        let dynamic_power_mw = (total_energy_j / inference_time_s) * 1000.0;
        let static_power_mw = 0.5;

        let total_power_mw = dynamic_power_mw + static_power_mw;
        let energy_per_inference_uj = total_energy_j * 1e6;

        let ops_per_second = actual_ops / inference_time_s;
        let efficiency_tops_per_w = (ops_per_second / 1e12) / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: actual_ops as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        (self.base_energy_per_op_pj + self.sparsity_overhead_pj) * 1e-12
    }

    fn energy_per_synop(&self) -> f64 {
        self.base_energy_per_op_pj * 1e-12
    }
}

/// Dynamic power estimator (activity-dependent power).
///
/// Models power that scales with computational activity,
/// as opposed to static leakage power.
///
/// P_dynamic = α × C × V² × f × activity
#[derive(Debug, Clone)]
pub struct DynamicPowerEstimator {
    /// Switching activity factor (0-1)
    pub activity_factor: f64,
    /// Effective capacitance in farads
    pub capacitance_f: f64,
    /// Supply voltage in volts
    pub voltage_v: f64,
    /// Clock frequency in Hz
    pub frequency_hz: f64,
}

impl Default for DynamicPowerEstimator {
    fn default() -> Self {
        Self {
            activity_factor: 0.3,         // 30% switching activity
            capacitance_f: 1e-12,         // 1 pF
            voltage_v: 1.0,               // 1.0V supply
            frequency_hz: 100e6,          // 100 MHz
        }
    }
}

impl DynamicPowerEstimator {
    /// Creates a high-performance estimator (higher voltage, frequency)
    pub fn high_performance() -> Self {
        Self {
            activity_factor: 0.5,
            capacitance_f: 2e-12,
            voltage_v: 1.2,
            frequency_hz: 1e9, // 1 GHz
        }
    }

    /// Creates an ultra-low-power estimator
    pub fn ultra_low_power() -> Self {
        Self {
            activity_factor: 0.1,
            capacitance_f: 0.5e-12,
            voltage_v: 0.6,
            frequency_hz: 10e6, // 10 MHz
        }
    }

    /// Computes dynamic power using CMOS power equation
    pub fn compute_dynamic_power(&self, activity: f64) -> f64 {
        // P = α × C × V² × f
        let power_w = activity * self.capacitance_f *
                     self.voltage_v.powi(2) * self.frequency_hz;
        power_w * 1000.0 // Convert to mW
    }
}

impl PowerEstimator for DynamicPowerEstimator {
    fn name(&self) -> &str {
        "Dynamic Power"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Activity scales with spike rate
        let effective_activity = self.activity_factor * model.spike_rate / 0.1; // Normalized to 10%

        let dynamic_power_mw = self.compute_dynamic_power(effective_activity);
        let static_power_mw = 0.0; // This estimator only models dynamic power

        let total_power_mw = dynamic_power_mw;

        let inference_time_s = (model.timesteps as f64) * 0.001;
        let energy_per_inference_uj = total_power_mw * inference_time_s;

        // Estimate operations based on activity
        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

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
        // Energy per switching event
        self.capacitance_f * self.voltage_v.powi(2)
    }

    fn energy_per_synop(&self) -> f64 {
        self.energy_per_spike()
    }
}

/// Leakage (static) power estimator.
///
/// Models static power consumption due to transistor leakage,
/// which is independent of activity and scales with temperature
/// and technology node.
///
/// P_leakage = V × I_leakage
#[derive(Debug, Clone)]
pub struct LeakagePowerEstimator {
    /// Leakage current per transistor in amperes
    pub leakage_current_per_transistor_a: f64,
    /// Number of transistors
    pub num_transistors: u64,
    /// Supply voltage in volts
    pub voltage_v: f64,
    /// Temperature in Celsius
    pub temperature_c: f64,
}

impl Default for LeakagePowerEstimator {
    fn default() -> Self {
        Self {
            leakage_current_per_transistor_a: 1e-12, // 1 pA per transistor
            num_transistors: 1_000_000,               // 1M transistors
            voltage_v: 1.0,
            temperature_c: 25.0,
        }
    }
}

impl LeakagePowerEstimator {
    /// Creates an advanced node estimator (higher leakage)
    pub fn advanced_node() -> Self {
        Self {
            leakage_current_per_transistor_a: 10e-12, // 10 pA (7nm)
            num_transistors: 100_000_000,              // 100M transistors
            voltage_v: 0.8,
            temperature_c: 25.0,
        }
    }

    /// Creates a legacy node estimator (lower leakage)
    pub fn legacy_node() -> Self {
        Self {
            leakage_current_per_transistor_a: 0.1e-12, // 0.1 pA (90nm)
            num_transistors: 1_000_000,
            voltage_v: 1.2,
            temperature_c: 25.0,
        }
    }

    /// Computes leakage power with temperature scaling
    pub fn compute_leakage_power(&self) -> f64 {
        // Leakage increases exponentially with temperature
        // Approximation: 2× per 10°C increase from 25°C
        let temp_factor = 2.0_f64.powf((self.temperature_c - 25.0) / 10.0);

        let total_leakage_current_a = (self.num_transistors as f64) *
                                      self.leakage_current_per_transistor_a *
                                      temp_factor;

        let power_w = self.voltage_v * total_leakage_current_a;
        power_w * 1000.0 // Convert to mW
    }
}

impl PowerEstimator for LeakagePowerEstimator {
    fn name(&self) -> &str {
        "Leakage Power"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Leakage is independent of activity
        let static_power_mw = self.compute_leakage_power();
        let dynamic_power_mw = 0.0; // This estimator only models leakage

        let total_power_mw = static_power_mw;

        let inference_time_s = (model.timesteps as f64) * 0.001;
        let energy_per_inference_uj = total_power_mw * inference_time_s;

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: 0, // Leakage is not operation-dependent
            efficiency_tops_per_w: 0.0,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        0.0 // Leakage is not spike-dependent
    }

    fn energy_per_synop(&self) -> f64 {
        0.0 // Leakage is not operation-dependent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snn_vs_ann_estimator() {
        let estimator = SNNvsANNEstimator::default();
        let stats = ModelStats {
            spike_rate: 0.01, // 1% activity - very sparse for clear SNN advantage
            ..Default::default()
        };

        let ann_metrics = estimator.estimate_ann(&stats);
        let snn_metrics = estimator.estimate_snn(&stats);

        // SNN should be more efficient for sparse activity
        assert!(snn_metrics.energy_per_inference_uj < ann_metrics.energy_per_inference_uj,
            "SNN: {} µJ, ANN: {} µJ", snn_metrics.energy_per_inference_uj, ann_metrics.energy_per_inference_uj);

        let ratio = estimator.efficiency_ratio(&stats);
        assert!(ratio < 1.0, "Efficiency ratio should be < 1.0, got {}", ratio); // SNN uses less energy
    }

    #[test]
    fn test_snn_advantage_increases_with_sparsity() {
        let estimator = SNNvsANNEstimator::default();

        let sparse_stats = ModelStats {
            spike_rate: 0.01, // 1% activity
            ..Default::default()
        };

        let dense_stats = ModelStats {
            spike_rate: 0.5, // 50% activity
            ..Default::default()
        };

        let sparse_ratio = estimator.efficiency_ratio(&sparse_stats);
        let dense_ratio = estimator.efficiency_ratio(&dense_stats);

        // SNN advantage should be greater for sparse activity
        assert!(sparse_ratio < dense_ratio);
    }

    #[test]
    fn test_sparsity_aware_estimator() {
        let estimator = SparsityAwareEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert_eq!(estimator.name(), "Sparsity-Aware");
    }

    #[test]
    fn test_pruning_reduces_energy() {
        let no_pruning = SparsityAwareEstimator::default();
        let with_pruning = SparsityAwareEstimator::with_pruning(0.5); // 50% pruned

        let stats = ModelStats::default();

        let no_pruning_metrics = no_pruning.estimate_inference(&stats);
        let pruned_metrics = with_pruning.estimate_inference(&stats);

        // Pruning should reduce both operations and energy
        assert!(pruned_metrics.synops_per_inference < no_pruning_metrics.synops_per_inference);
        assert!(pruned_metrics.energy_per_inference_uj < no_pruning_metrics.energy_per_inference_uj);
    }

    #[test]
    fn test_effective_sparsity() {
        let estimator = SparsityAwareEstimator::with_pruning(0.5); // 50% structural

        let temporal_10 = 0.1; // 10% temporal
        let effective = estimator.effective_sparsity(temporal_10);

        // Effective = 0.1 × (1 - 0.5) = 0.05
        assert!((effective - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_dynamic_power_estimator() {
        let estimator = DynamicPowerEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.dynamic_power_mw > 0.0);
        assert_eq!(metrics.static_power_mw, 0.0); // Only dynamic power
        assert_eq!(estimator.name(), "Dynamic Power");
    }

    #[test]
    fn test_dynamic_power_scaling() {
        let low_power = DynamicPowerEstimator::ultra_low_power();
        let high_perf = DynamicPowerEstimator::high_performance();

        let stats = ModelStats::default();

        let lp_metrics = low_power.estimate_inference(&stats);
        let hp_metrics = high_perf.estimate_inference(&stats);

        // High performance should consume more power
        assert!(hp_metrics.dynamic_power_mw > lp_metrics.dynamic_power_mw);
    }

    #[test]
    fn test_voltage_frequency_impact() {
        let low_v = DynamicPowerEstimator {
            voltage_v: 0.8,
            frequency_hz: 100e6,
            ..Default::default()
        };

        let high_v = DynamicPowerEstimator {
            voltage_v: 1.2,
            frequency_hz: 100e6,
            ..Default::default()
        };

        let power_low = low_v.compute_dynamic_power(0.5);
        let power_high = high_v.compute_dynamic_power(0.5);

        // Power scales with V²
        let expected_ratio = (1.2_f64 / 0.8_f64).powi(2);
        let actual_ratio = power_high / power_low;
        assert!((actual_ratio - expected_ratio).abs() < 0.01);
    }

    #[test]
    fn test_leakage_power_estimator() {
        let estimator = LeakagePowerEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.static_power_mw > 0.0);
        assert_eq!(metrics.dynamic_power_mw, 0.0); // Only static power
        assert_eq!(estimator.name(), "Leakage Power");
    }

    #[test]
    fn test_leakage_temperature_scaling() {
        let cold = LeakagePowerEstimator {
            temperature_c: 0.0,
            ..Default::default()
        };

        let hot = LeakagePowerEstimator {
            temperature_c: 85.0,
            ..Default::default()
        };

        let power_cold = cold.compute_leakage_power();
        let power_hot = hot.compute_leakage_power();

        // Leakage should increase with temperature
        assert!(power_hot > power_cold);
    }

    #[test]
    fn test_leakage_technology_scaling() {
        let advanced = LeakagePowerEstimator::advanced_node();
        let legacy = LeakagePowerEstimator::legacy_node();

        let advanced_power = advanced.compute_leakage_power();
        let legacy_power = legacy.compute_leakage_power();

        // Advanced nodes have more transistors and higher leakage per transistor
        assert!(advanced_power > legacy_power);
    }

    #[test]
    fn test_leakage_independent_of_activity() {
        let estimator = LeakagePowerEstimator::default();

        let low_activity = ModelStats {
            spike_rate: 0.01,
            ..Default::default()
        };

        let high_activity = ModelStats {
            spike_rate: 0.9,
            ..Default::default()
        };

        let low_metrics = estimator.estimate_inference(&low_activity);
        let high_metrics = estimator.estimate_inference(&high_activity);

        // Leakage power should be the same regardless of activity
        assert_eq!(low_metrics.static_power_mw, high_metrics.static_power_mw);
    }

    #[test]
    fn test_all_comparison_estimators() {
        let stats = ModelStats::default();

        let estimators: Vec<Box<dyn PowerEstimator>> = vec![
            Box::new(SNNvsANNEstimator::default()),
            Box::new(SparsityAwareEstimator::default()),
            Box::new(DynamicPowerEstimator::default()),
            Box::new(LeakagePowerEstimator::default()),
        ];

        for estimator in estimators {
            let metrics = estimator.estimate_inference(&stats);
            assert!(metrics.total_power_mw >= 0.0);
            assert!(!estimator.name().is_empty());
        }
    }
}
