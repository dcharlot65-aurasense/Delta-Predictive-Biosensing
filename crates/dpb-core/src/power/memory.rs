//! Memory access power models.
//!
//! This module provides power estimation for different types of memory
//! accesses in neuromorphic and conventional systems.

use super::{ModelStats, PowerEstimator, PowerMetrics};

/// Memory access power estimator.
///
/// Models energy consumption for different types of memory access.
///
/// Reference: Energy per access varies significantly by memory type:
/// - On-chip SRAM: 1-5 pJ per access
/// - DRAM: 10-100 pJ per access
/// - Flash/NVM: 100-1000 pJ per access
#[derive(Debug, Clone)]
pub struct MemoryAccessEstimator {
    /// Energy per read access in picojoules
    pub energy_per_read_pj: f64,
    /// Energy per write access in picojoules
    pub energy_per_write_pj: f64,
    /// Memory type
    pub memory_type: MemoryType,
    /// Access width in bytes
    pub access_width_bytes: usize,
}

// These are established domain acronyms -- clinical file formats, ECG
// beat annotations, and hardware terms. Camel-casing them (Pvc, Wfdb,
// Dram) would make this harder to read for anyone who works with them.
/// Where a value lives in the memory hierarchy.
///
/// Access energy differs by orders of magnitude between these, so the
/// distinction dominates any power estimate.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryType {
    /// On-chip SRAM (register file, L1 cache)
    OnChipSRAM,
    /// L2/L3 Cache
    Cache,
    /// Main memory (DRAM)
    DRAM,
    /// Non-volatile memory (Flash, ReRAM, etc.)
    NonVolatile,
}

impl Default for MemoryAccessEstimator {
    fn default() -> Self {
        // Default to on-chip SRAM (most common for neuromorphic weights)
        Self {
            energy_per_read_pj: 2.0,
            energy_per_write_pj: 3.0,
            memory_type: MemoryType::OnChipSRAM,
            access_width_bytes: 4,
        }
    }
}

impl MemoryAccessEstimator {
    /// Creates an on-chip SRAM estimator
    pub fn on_chip_sram() -> Self {
        Self {
            energy_per_read_pj: 2.0,
            energy_per_write_pj: 3.0,
            memory_type: MemoryType::OnChipSRAM,
            access_width_bytes: 4,
        }
    }

    /// Creates a cache memory estimator (L2/L3)
    pub fn cache() -> Self {
        Self {
            energy_per_read_pj: 10.0,
            energy_per_write_pj: 15.0,
            memory_type: MemoryType::Cache,
            access_width_bytes: 64, // Cache line
        }
    }

    /// Creates a DRAM estimator
    pub fn dram() -> Self {
        Self {
            energy_per_read_pj: 50.0,
            energy_per_write_pj: 80.0,
            memory_type: MemoryType::DRAM,
            access_width_bytes: 8,
        }
    }

    /// Creates a non-volatile memory estimator
    pub fn non_volatile() -> Self {
        Self {
            energy_per_read_pj: 20.0,
            energy_per_write_pj: 500.0, // Writes are expensive in NVM
            memory_type: MemoryType::NonVolatile,
            access_width_bytes: 4,
        }
    }

    /// Estimates energy for a specific number of reads and writes
    pub fn estimate_accesses(&self, num_reads: u64, num_writes: u64) -> f64 {
        let read_energy_j = (num_reads as f64) * self.energy_per_read_pj * 1e-12;
        let write_energy_j = (num_writes as f64) * self.energy_per_write_pj * 1e-12;
        read_energy_j + write_energy_j
    }
}

impl PowerEstimator for MemoryAccessEstimator {
    fn name(&self) -> &str {
        match self.memory_type {
            MemoryType::OnChipSRAM => "Memory Access (On-Chip SRAM)",
            MemoryType::Cache => "Memory Access (Cache)",
            MemoryType::DRAM => "Memory Access (DRAM)",
            MemoryType::NonVolatile => "Memory Access (Non-Volatile)",
        }
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // Memory access patterns for SNN inference:
        // 1. Weight reads: one read per synaptic operation
        // 2. Neuron state reads/writes: neurons × timesteps
        // 3. Spike buffer reads/writes: spikes × fanout

        let total_spikes = (model.num_neurons as f64) * (model.timesteps as f64) * model.spike_rate;
        let synops = (total_spikes * (model.num_synapses as f64)) / (model.num_neurons as f64);

        // Weight reads (one per synop)
        let weight_reads = synops;

        // Neuron state updates (read-modify-write per timestep)
        let neuron_state_reads = (model.num_neurons * model.timesteps) as f64;
        let neuron_state_writes = neuron_state_reads;

        // Spike buffer operations
        let spike_buffer_writes = total_spikes;
        let spike_buffer_reads = total_spikes;

        // Total accesses
        let total_reads = weight_reads + neuron_state_reads + spike_buffer_reads;
        let total_writes = neuron_state_writes + spike_buffer_writes;

        // Energy calculation
        let read_energy_j = total_reads * self.energy_per_read_pj * 1e-12;
        let write_energy_j = total_writes * self.energy_per_write_pj * 1e-12;
        let total_energy_j = read_energy_j + write_energy_j;

        // Time calculation (1ms per timestep)
        let inference_time_s = (model.timesteps as f64) * 0.001;

        let dynamic_power_mw = (total_energy_j / inference_time_s) * 1000.0;

        // Static power depends on memory type
        let static_power_mw = match self.memory_type {
            MemoryType::OnChipSRAM => 0.1,   // Minimal leakage
            MemoryType::Cache => 1.0,        // Some leakage
            MemoryType::DRAM => 10.0,        // Refresh power
            MemoryType::NonVolatile => 0.01, // Very low retention power
        };

        let total_power_mw = dynamic_power_mw + static_power_mw;
        let energy_per_inference_uj = total_energy_j * 1e6;

        // Memory operations per second
        let total_accesses = total_reads + total_writes;
        let accesses_per_second = total_accesses / inference_time_s;
        let efficiency_tops_per_w = (accesses_per_second / 1e12) / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: total_accesses as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        // Spike typically involves one write and one read
        (self.energy_per_read_pj + self.energy_per_write_pj) * 1e-12
    }

    fn energy_per_synop(&self) -> f64 {
        // Synaptic operation requires weight read
        self.energy_per_read_pj * 1e-12
    }
}

/// Memory bandwidth estimator for system-level analysis
#[derive(Debug, Clone)]
pub struct MemoryBandwidthEstimator {
    /// Bandwidth in GB/s
    pub bandwidth_gbs: f64,
    /// Energy per byte transferred in picojoules
    pub energy_per_byte_pj: f64,
}

impl MemoryBandwidthEstimator {
    /// Estimates power for sustained bandwidth usage
    pub fn estimate_power(&self, utilization: f64) -> f64 {
        // Power = bandwidth × utilization × energy_per_byte
        let bytes_per_second = self.bandwidth_gbs * 1e9 * utilization;
        let power_w = bytes_per_second * self.energy_per_byte_pj * 1e-12;
        power_w * 1000.0 // Convert to mW
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_access_estimator() {
        let estimator = MemoryAccessEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert!(metrics.synops_per_inference > 0);
        assert!(estimator.name().contains("Memory Access"));
    }

    #[test]
    fn test_memory_types() {
        let sram = MemoryAccessEstimator::on_chip_sram();
        let cache = MemoryAccessEstimator::cache();
        let dram = MemoryAccessEstimator::dram();
        let nvm = MemoryAccessEstimator::non_volatile();

        // Verify energy ordering for reads
        assert!(sram.energy_per_read_pj < cache.energy_per_read_pj);
        assert!(cache.energy_per_read_pj < dram.energy_per_read_pj);

        // NVM writes should be most expensive
        assert!(nvm.energy_per_write_pj > dram.energy_per_write_pj);
        assert!(nvm.energy_per_write_pj > cache.energy_per_write_pj);
        assert!(nvm.energy_per_write_pj > sram.energy_per_write_pj);
    }

    #[test]
    fn test_memory_access_estimation() {
        let estimator = MemoryAccessEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        // Should account for weight reads, neuron state, and spike buffers
        assert!(metrics.synops_per_inference > (stats.num_neurons * stats.timesteps) as u64);
    }

    #[test]
    fn test_read_write_asymmetry() {
        let estimators = vec![
            MemoryAccessEstimator::on_chip_sram(),
            MemoryAccessEstimator::cache(),
            MemoryAccessEstimator::dram(),
            MemoryAccessEstimator::non_volatile(),
        ];

        for estimator in estimators {
            // Writes should generally be more expensive than reads
            assert!(estimator.energy_per_write_pj >= estimator.energy_per_read_pj);
        }
    }

    #[test]
    fn test_dram_static_power() {
        let dram = MemoryAccessEstimator::dram();
        let sram = MemoryAccessEstimator::on_chip_sram();
        let stats = ModelStats::default();

        let dram_metrics = dram.estimate_inference(&stats);
        let sram_metrics = sram.estimate_inference(&stats);

        // DRAM should have higher static power due to refresh
        assert!(dram_metrics.static_power_mw > sram_metrics.static_power_mw);
    }

    #[test]
    fn test_estimate_accesses() {
        let estimator = MemoryAccessEstimator::default();

        let energy_j = estimator.estimate_accesses(1000, 500);

        let expected_j =
            (1000.0 * estimator.energy_per_read_pj + 500.0 * estimator.energy_per_write_pj) * 1e-12;

        assert!((energy_j - expected_j).abs() < 1e-15);
    }

    #[test]
    fn test_memory_bandwidth_estimator() {
        let estimator = MemoryBandwidthEstimator {
            bandwidth_gbs: 100.0,
            energy_per_byte_pj: 10.0,
        };

        // 50% utilization
        let power_mw = estimator.estimate_power(0.5);
        assert!(power_mw > 0.0);

        // 100% utilization should use twice as much power as 50%
        let power_full = estimator.estimate_power(1.0);
        let power_half = estimator.estimate_power(0.5);
        assert!((power_full / power_half - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_spike_rate_impact_on_memory() {
        let estimator = MemoryAccessEstimator::default();
        // Reassigned below, so it stays mut.
        let mut stats = ModelStats {
            spike_rate: 0.01,
            ..Default::default()
        };
        let low_activity = estimator.estimate_inference(&stats);

        stats.spike_rate = 0.5;
        let high_activity = estimator.estimate_inference(&stats);

        // Higher spike rate means more spike buffer accesses and synops
        assert!(high_activity.synops_per_inference > low_activity.synops_per_inference);
        assert!(high_activity.energy_per_inference_uj > low_activity.energy_per_inference_uj);
    }

    #[test]
    fn test_weight_precision_impact() {
        let stats = ModelStats {
            weight_bits: 8,
            ..Default::default()
        };

        let sram = MemoryAccessEstimator::on_chip_sram();
        let metrics = sram.estimate_inference(&stats);

        // Verify reasonable power consumption
        assert!(metrics.total_power_mw > 0.0);
        assert!(metrics.total_power_mw < 1000.0); // Should be reasonable for SRAM
    }
}
