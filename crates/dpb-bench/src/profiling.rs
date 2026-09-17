//! Performance profiling tools
//!
//! This module provides profiling utilities for measuring time, memory,
//! spike activity, and energy consumption.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Result from profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    /// Profiler name
    pub name: String,
    /// Elapsed time (ms)
    pub elapsed_ms: f64,
    /// Memory used (bytes)
    pub memory_bytes: usize,
    /// Number of spikes
    pub num_spikes: usize,
    /// Sparsity (0.0-1.0)
    pub sparsity: f64,
    /// Estimated energy (mJ)
    pub energy_mj: f64,
    /// Additional metrics
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for ProfileResult {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            elapsed_ms: 0.0,
            memory_bytes: 0,
            num_spikes: 0,
            sparsity: 0.0,
            energy_mj: 0.0,
            metadata: std::collections::HashMap::new(),
        }
    }
}

/// Time profiler
#[derive(Debug)]
pub struct TimeProfiler {
    name: String,
    start: Option<Instant>,
    elapsed: Duration,
    measurements: Vec<Duration>,
}

impl TimeProfiler {
    /// Create a new time profiler
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: None,
            elapsed: Duration::default(),
            measurements: Vec::new(),
        }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Stop timing
    pub fn stop(&mut self) {
        if let Some(start) = self.start.take() {
            let duration = start.elapsed();
            self.elapsed += duration;
            self.measurements.push(duration);
        }
    }

    /// Reset profiler
    pub fn reset(&mut self) {
        self.start = None;
        self.elapsed = Duration::default();
        self.measurements.clear();
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed.as_secs_f64() * 1000.0
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> f64 {
        self.elapsed.as_secs_f64() * 1_000_000.0
    }

    /// Get average time per measurement
    pub fn average_ms(&self) -> f64 {
        if self.measurements.is_empty() {
            0.0
        } else {
            let total: Duration = self.measurements.iter().sum();
            total.as_secs_f64() * 1000.0 / self.measurements.len() as f64
        }
    }

    /// Get minimum time
    pub fn min_ms(&self) -> f64 {
        self.measurements
            .iter()
            .min()
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }

    /// Get maximum time
    pub fn max_ms(&self) -> f64 {
        self.measurements
            .iter()
            .max()
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }

    /// Get standard deviation
    pub fn std_dev_ms(&self) -> f64 {
        if self.measurements.len() < 2 {
            return 0.0;
        }

        let avg = self.average_ms();
        let variance: f64 = self
            .measurements
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() * 1000.0 - avg;
                diff * diff
            })
            .sum::<f64>()
            / (self.measurements.len() - 1) as f64;

        variance.sqrt()
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get number of measurements
    pub fn num_measurements(&self) -> usize {
        self.measurements.len()
    }
}

/// Memory profiler
#[derive(Debug)]
pub struct MemoryProfiler {
    name: String,
    peak_bytes: usize,
    current_bytes: usize,
    allocations: Vec<usize>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            peak_bytes: 0,
            current_bytes: 0,
            allocations: Vec::new(),
        }
    }

    /// Record allocation
    pub fn allocate(&mut self, bytes: usize) {
        self.current_bytes += bytes;
        self.allocations.push(bytes);
        if self.current_bytes > self.peak_bytes {
            self.peak_bytes = self.current_bytes;
        }
    }

    /// Record deallocation
    pub fn deallocate(&mut self, bytes: usize) {
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
    }

    /// Reset profiler
    pub fn reset(&mut self) {
        self.peak_bytes = 0;
        self.current_bytes = 0;
        self.allocations.clear();
    }

    /// Get peak memory usage in bytes
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Get peak memory usage in MB
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / 1_048_576.0
    }

    /// Get current memory usage
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Get total allocations
    pub fn total_allocations(&self) -> usize {
        self.allocations.len()
    }

    /// Get average allocation size
    pub fn average_allocation(&self) -> f64 {
        if self.allocations.is_empty() {
            0.0
        } else {
            self.allocations.iter().sum::<usize>() as f64 / self.allocations.len() as f64
        }
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Spike profiler for SNN analysis
#[derive(Debug)]
pub struct SpikeProfiler {
    name: String,
    total_spikes: usize,
    total_neurons: usize,
    total_timesteps: usize,
    spike_times: Vec<Vec<usize>>,
}

impl SpikeProfiler {
    /// Create a new spike profiler
    pub fn new(name: &str, num_neurons: usize, num_timesteps: usize) -> Self {
        Self {
            name: name.to_string(),
            total_spikes: 0,
            total_neurons: num_neurons,
            total_timesteps: num_timesteps,
            spike_times: vec![Vec::new(); num_neurons],
        }
    }

    /// Record a spike
    pub fn record_spike(&mut self, neuron_id: usize, timestep: usize) {
        if neuron_id < self.total_neurons {
            self.total_spikes += 1;
            self.spike_times[neuron_id].push(timestep);
        }
    }

    /// Record multiple spikes
    pub fn record_spikes(&mut self, spikes: &[(usize, usize)]) {
        for &(neuron_id, timestep) in spikes {
            self.record_spike(neuron_id, timestep);
        }
    }

    /// Get sparsity (fraction of possible spikes that occurred)
    pub fn sparsity(&self) -> f64 {
        let max_spikes = self.total_neurons * self.total_timesteps;
        if max_spikes == 0 {
            0.0
        } else {
            self.total_spikes as f64 / max_spikes as f64
        }
    }

    /// Get average firing rate (spikes per neuron per timestep)
    pub fn average_firing_rate(&self) -> f64 {
        if self.total_neurons == 0 || self.total_timesteps == 0 {
            0.0
        } else {
            self.total_spikes as f64 / (self.total_neurons * self.total_timesteps) as f64
        }
    }

    /// Get number of active neurons
    pub fn num_active_neurons(&self) -> usize {
        self.spike_times
            .iter()
            .filter(|spikes| !spikes.is_empty())
            .count()
    }

    /// Get inter-spike intervals for a neuron
    pub fn inter_spike_intervals(&self, neuron_id: usize) -> Vec<usize> {
        if neuron_id >= self.total_neurons || self.spike_times[neuron_id].len() < 2 {
            return Vec::new();
        }

        let spikes = &self.spike_times[neuron_id];
        spikes.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Get coefficient of variation (CV) for ISIs
    pub fn coefficient_of_variation(&self, neuron_id: usize) -> f64 {
        let isis = self.inter_spike_intervals(neuron_id);
        if isis.len() < 2 {
            return 0.0;
        }

        let mean = isis.iter().sum::<usize>() as f64 / isis.len() as f64;
        let variance = isis
            .iter()
            .map(|&isi| {
                let diff = isi as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (isis.len() - 1) as f64;

        variance.sqrt() / mean
    }

    /// Reset profiler
    pub fn reset(&mut self) {
        self.total_spikes = 0;
        for spikes in &mut self.spike_times {
            spikes.clear();
        }
    }

    /// Get total spikes
    pub fn total_spikes(&self) -> usize {
        self.total_spikes
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Energy estimator for neuromorphic vs. conventional computing
#[derive(Debug)]
pub struct EnergyEstimator {
    name: String,
    /// Energy model type
    model: EnergyModel,
    /// Total energy consumed (mJ)
    total_energy_mj: f64,
}

// Domain notation -- process nodes, file formats. Camel case would
// diverge from how these are written everywhere else.
/// Energy models for different hardware
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub enum EnergyModel {
    /// Neuromorphic hardware (energy per spike-synaptic operation)
    Neuromorphic {
        /// Energy per spike-synaptic operation, in picojoules.
        energy_per_spike_op_pj: f64,
    },
    /// CMOS ANN (energy per MAC operation)
    CMOS {
        /// Energy per multiply-accumulate, in picojoules.
        energy_per_mac_pj: f64,
    },
    /// Custom model
    Custom {
        /// Energy per operation, in picojoules.
        energy_per_op_pj: f64,
    },
}

impl Default for EnergyModel {
    fn default() -> Self {
        EnergyModel::Neuromorphic {
            energy_per_spike_op_pj: 50.0, // 50 pJ per spike-synaptic operation
        }
    }
}

impl EnergyEstimator {
    /// Create a new energy estimator
    pub fn new(name: &str, model: EnergyModel) -> Self {
        Self {
            name: name.to_string(),
            model,
            total_energy_mj: 0.0,
        }
    }

    /// Create neuromorphic estimator
    pub fn neuromorphic(name: &str) -> Self {
        Self::new(name, EnergyModel::default())
    }

    /// Create CMOS estimator
    pub fn cmos(name: &str) -> Self {
        Self::new(
            name,
            EnergyModel::CMOS {
                // 45 nm float32 multiply (3.7 pJ) plus add (0.9 pJ), after
                // Horowitz's 45 nm energy table -- the same figure baselines.rs
                // uses. This was written 4600.0: the field is in picojoules and
                // the comment already said 4.6 pJ, so every CMOS estimate was
                // 1000x too high, and any SNN-vs-CMOS efficiency ratio built on
                // it was inflated by the same factor.
                energy_per_mac_pj: 4.6,
            },
        )
    }

    /// Record operations
    pub fn record_operations(&mut self, num_operations: usize) {
        let energy_per_op = match self.model {
            EnergyModel::Neuromorphic {
                energy_per_spike_op_pj,
            } => energy_per_spike_op_pj,
            EnergyModel::CMOS { energy_per_mac_pj } => energy_per_mac_pj,
            EnergyModel::Custom { energy_per_op_pj } => energy_per_op_pj,
        };

        // Convert pJ to mJ: pJ * 1e-9 = mJ
        self.total_energy_mj += num_operations as f64 * energy_per_op * 1e-9;
    }

    /// Record spike operations (for neuromorphic)
    pub fn record_spikes(&mut self, num_spikes: usize, fanout: usize) {
        let num_ops = num_spikes * fanout;
        self.record_operations(num_ops);
    }

    /// Record MAC operations (for ANNs)
    pub fn record_macs(&mut self, num_macs: usize) {
        self.record_operations(num_macs);
    }

    /// Get total energy in mJ
    pub fn total_energy_mj(&self) -> f64 {
        self.total_energy_mj
    }

    /// Get total energy in μJ
    pub fn total_energy_uj(&self) -> f64 {
        self.total_energy_mj * 1000.0
    }

    /// Get energy efficiency ratio vs baseline
    pub fn efficiency_vs(&self, baseline_mj: f64) -> f64 {
        if self.total_energy_mj > 0.0 {
            baseline_mj / self.total_energy_mj
        } else {
            0.0
        }
    }

    /// Reset estimator
    pub fn reset(&mut self) {
        self.total_energy_mj = 0.0;
    }

    /// Get name
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_time_profiler() {
        let mut profiler = TimeProfiler::new("test");

        profiler.start();
        thread::sleep(Duration::from_millis(10));
        profiler.stop();

        assert!(profiler.elapsed_ms() >= 10.0);
        assert_eq!(profiler.num_measurements(), 1);
    }

    #[test]
    fn test_time_profiler_multiple() {
        let mut profiler = TimeProfiler::new("test");

        for _ in 0..3 {
            profiler.start();
            thread::sleep(Duration::from_millis(5));
            profiler.stop();
        }

        assert_eq!(profiler.num_measurements(), 3);
        assert!(profiler.average_ms() >= 5.0);
        assert!(profiler.std_dev_ms() >= 0.0);
    }

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new("test");

        profiler.allocate(1024);
        assert_eq!(profiler.current_bytes(), 1024);

        profiler.allocate(2048);
        assert_eq!(profiler.current_bytes(), 3072);
        assert_eq!(profiler.peak_bytes(), 3072);

        profiler.deallocate(1024);
        assert_eq!(profiler.current_bytes(), 2048);
        assert_eq!(profiler.peak_bytes(), 3072);
    }

    #[test]
    fn test_spike_profiler() {
        let mut profiler = SpikeProfiler::new("test", 100, 50);

        profiler.record_spike(0, 10);
        profiler.record_spike(0, 20);
        profiler.record_spike(1, 15);

        assert_eq!(profiler.total_spikes(), 3);
        assert_eq!(profiler.num_active_neurons(), 2);

        let sparsity = profiler.sparsity();
        assert!(sparsity > 0.0 && sparsity < 1.0);

        let isis = profiler.inter_spike_intervals(0);
        assert_eq!(isis, vec![10]);
    }

    /// Energy is pinned to exact values, not compared.
    ///
    /// This test used to assert that 100 000 CMOS MACs cost more than 100 000
    /// neuromorphic spike-ops. That only held because the CMOS constant was
    /// 1000x too large; with the correct 4.6 pJ it is false, and the test was
    /// holding the unit error in place. The ordering was never a sound thing to
    /// assert either: 4.6 pJ is a bare-gate figure and 50 pJ a measured
    /// neuromorphic-chip one, so the two are not on the same basis. Exact
    /// values check the arithmetic and units, which is what this can vouch for.
    #[test]
    fn test_energy_estimator() {
        let mut estimator = EnergyEstimator::neuromorphic("test");
        estimator.record_spikes(1000, 100); // 100 000 spike-ops at 50 pJ
        assert!(
            (estimator.total_energy_mj() - 5.0e-3).abs() < 1e-12,
            "100 000 ops x 50 pJ is 5e-3 mJ, got {}",
            estimator.total_energy_mj()
        );

        let mut cmos = EnergyEstimator::cmos("cmos");
        cmos.record_macs(100_000); // 100 000 MACs at 4.6 pJ
        assert!(
            (cmos.total_energy_mj() - 4.6e-4).abs() < 1e-12,
            "100 000 MACs x 4.6 pJ is 4.6e-4 mJ, got {}",
            cmos.total_energy_mj()
        );
    }

    #[test]
    fn test_profile_result() {
        let result = ProfileResult {
            name: "test".to_string(),
            elapsed_ms: 10.5,
            memory_bytes: 1024,
            num_spikes: 100,
            sparsity: 0.05,
            energy_mj: 0.5,
            metadata: std::collections::HashMap::new(),
        };

        assert_eq!(result.name, "test");
        assert_eq!(result.elapsed_ms, 10.5);
    }
}
