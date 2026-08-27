//! Power estimation module for neuromorphic and conventional computing systems.
//!
//! This module provides comprehensive power estimation capabilities for:
//! - Neuromorphic hardware platforms (Xylo, Loihi, SpiNNaker, TrueNorth, BrainScaleS, Akida)
//! - Conventional digital platforms (CPU, GPU, FPGA, ASIC)
//! - Operation-level models (synaptic ops, MAC operations, memory access)
//! - Comparative analysis (SNN vs ANN, sparsity-aware, dynamic vs static power)
//!
//! # Example
//!
//! ```rust
//! use dpb_core::power::{PowerEstimator, ModelStats, XyloEstimator};
//!
//! let stats = ModelStats {
//!     num_neurons: 1000,
//!     num_synapses: 100_000,
//!     num_layers: 3,
//!     timesteps: 100,
//!     spike_rate: 0.05,
//!     weight_bits: 8,
//! };
//!
//! let estimator = XyloEstimator::default();
//! let metrics = estimator.estimate_inference(&stats);
//!
//! println!("Total power: {} mW", metrics.total_power_mw);
//! println!("Energy per inference: {} µJ", metrics.energy_per_inference_uj);
//! ```

pub mod comparison;
pub mod digital;
pub mod memory;
pub mod neuromorphic;
pub mod synaptic;

// Re-export all estimators
pub use comparison::{
    DynamicPowerEstimator, LeakagePowerEstimator, SNNvsANNEstimator, SparsityAwareEstimator,
};
pub use digital::{AsicEstimator, CpuEstimator, FpgaEstimator, GpuEstimator};
pub use memory::MemoryAccessEstimator;
pub use neuromorphic::{
    AkidaEstimator, BrainScaleSEstimator, LoihiEstimator, SpinnAkerEstimator, TrueNorthEstimator,
    XyloEstimator,
};
pub use synaptic::{
    MACEstimator, NeuronUpdateEstimator, SpikeTransmitEstimator, SynapticOpEstimator,
};

/// Statistics describing a neural network model for power estimation.
#[derive(Debug, Clone)]
pub struct ModelStats {
    /// Number of neurons in the network
    pub num_neurons: usize,
    /// Number of synapses (connections) in the network
    pub num_synapses: usize,
    /// Number of layers in the network
    pub num_layers: usize,
    /// Number of timesteps for inference
    pub timesteps: usize,
    /// Average spike rate (spikes per neuron per timestep)
    pub spike_rate: f64,
    /// Weight precision in bits
    pub weight_bits: usize,
}

impl Default for ModelStats {
    fn default() -> Self {
        Self {
            num_neurons: 1000,
            num_synapses: 100_000,
            num_layers: 3,
            timesteps: 100,
            spike_rate: 0.1,
            weight_bits: 8,
        }
    }
}

/// Detailed power consumption metrics.
#[derive(Debug, Clone)]
pub struct PowerMetrics {
    /// Dynamic power consumption in milliwatts
    pub dynamic_power_mw: f64,
    /// Static (leakage) power consumption in milliwatts
    pub static_power_mw: f64,
    /// Total power consumption in milliwatts
    pub total_power_mw: f64,
    /// Energy per inference in microjoules
    pub energy_per_inference_uj: f64,
    /// Number of synaptic operations per inference
    pub synops_per_inference: u64,
    /// Computational efficiency in tera-ops per watt
    pub efficiency_tops_per_w: f64,
}

impl Default for PowerMetrics {
    fn default() -> Self {
        Self {
            dynamic_power_mw: 0.0,
            static_power_mw: 0.0,
            total_power_mw: 0.0,
            energy_per_inference_uj: 0.0,
            synops_per_inference: 0,
            efficiency_tops_per_w: 0.0,
        }
    }
}

/// Core trait for power estimation across different hardware platforms.
///
/// This trait provides methods to estimate power consumption for both
/// inference and training on various neuromorphic and conventional platforms.
pub trait PowerEstimator: Send + Sync {
    /// Returns the name of this power estimator.
    fn name(&self) -> &str;

    /// Estimates power consumption for a single inference.
    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics;

    /// Estimates power consumption for training over multiple epochs.
    fn estimate_training(&self, model: &ModelStats, epochs: usize) -> PowerMetrics {
        let mut inference_metrics = self.estimate_inference(model);

        // Training typically has 2-3x higher power than inference
        let training_factor = 2.5;
        inference_metrics.dynamic_power_mw *= training_factor;
        inference_metrics.total_power_mw =
            inference_metrics.dynamic_power_mw + inference_metrics.static_power_mw;
        inference_metrics.energy_per_inference_uj *= training_factor * epochs as f64;

        inference_metrics
    }

    /// Returns the energy per spike transmission in joules.
    fn energy_per_spike(&self) -> f64;

    /// Returns the energy per synaptic operation in joules.
    fn energy_per_synop(&self) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_stats_default() {
        let stats = ModelStats::default();
        assert_eq!(stats.num_neurons, 1000);
        assert_eq!(stats.num_synapses, 100_000);
        assert!(stats.spike_rate > 0.0);
    }

    #[test]
    fn test_power_metrics_default() {
        let metrics = PowerMetrics::default();
        assert_eq!(metrics.total_power_mw, 0.0);
        assert_eq!(metrics.synops_per_inference, 0);
    }

    #[test]
    fn test_all_estimators() {
        let stats = ModelStats::default();

        // Neuromorphic estimators
        let estimators: Vec<Box<dyn PowerEstimator>> = vec![
            Box::new(XyloEstimator::default()),
            Box::new(LoihiEstimator::default()),
            Box::new(SpinnAkerEstimator::default()),
            Box::new(TrueNorthEstimator::default()),
            Box::new(BrainScaleSEstimator::default()),
            Box::new(AkidaEstimator::default()),
            Box::new(GpuEstimator::default()),
            Box::new(CpuEstimator::default()),
            Box::new(FpgaEstimator::default()),
            Box::new(AsicEstimator::default()),
            Box::new(SynapticOpEstimator::default()),
            Box::new(MACEstimator::default()),
            Box::new(MemoryAccessEstimator::default()),
            Box::new(SpikeTransmitEstimator::default()),
            Box::new(NeuronUpdateEstimator::default()),
            Box::new(SNNvsANNEstimator::default()),
            Box::new(SparsityAwareEstimator::default()),
            Box::new(DynamicPowerEstimator::default()),
            Box::new(LeakagePowerEstimator::default()),
        ];

        // Verify all 19 estimators work
        assert_eq!(estimators.len(), 19);

        for estimator in &estimators {
            let metrics = estimator.estimate_inference(&stats);
            assert!(metrics.total_power_mw >= 0.0);
            assert!(estimator.energy_per_spike() >= 0.0);
            assert!(estimator.energy_per_synop() >= 0.0);
            assert!(!estimator.name().is_empty());
        }
    }
}
