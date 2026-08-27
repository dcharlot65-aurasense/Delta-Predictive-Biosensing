//! SNN efficiency metrics for neuromorphic hardware evaluation.

use super::MetricTrait;
use crate::error::{DpbError, Result};

/// Spike Rate (spikes per second).
#[derive(Debug, Clone)]
pub struct SpikeRate {
    total_spikes: f64,
    total_time: f64,
}

impl SpikeRate {
    /// Creates a new [`SpikeRate`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            total_spikes: 0.0,
            total_time: 0.0,
        }
    }
}

impl Default for SpikeRate {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for SpikeRate {
    fn name(&self) -> &str {
        "spike_rate"
    }

    fn compute(&self, spikes: &[f32], time_steps: &[f32]) -> Result<f64> {
        if spikes.len() != time_steps.len() {
            return Err(DpbError::Other("Spikes and time steps must have the same length".to_string()));
        }
        if time_steps.is_empty() {
            return Ok(0.0);
        }

        let total_spikes: f64 = spikes.iter().map(|&s| s as f64).sum();
        let total_time: f64 = time_steps.iter().map(|&t| t as f64).sum();

        if total_time == 0.0 {
            Ok(0.0)
        } else {
            Ok(total_spikes / total_time)
        }
    }

    fn update(&mut self, spikes: &[f32], time_steps: &[f32]) {
        self.total_spikes += spikes.iter().map(|&s| s as f64).sum::<f64>();
        self.total_time += time_steps.iter().map(|&t| t as f64).sum::<f64>();
    }

    fn result(&self) -> f64 {
        if self.total_time == 0.0 {
            0.0
        } else {
            self.total_spikes / self.total_time
        }
    }

    fn reset(&mut self) {
        self.total_spikes = 0.0;
        self.total_time = 0.0;
    }
}

/// Sparsity (proportion of zero activations).
#[derive(Debug, Clone)]
pub struct Sparsity {
    zero_count: f64,
    total_count: f64,
}

impl Sparsity {
    /// Creates a new [`Sparsity`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            zero_count: 0.0,
            total_count: 0.0,
        }
    }
}

impl Default for Sparsity {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for Sparsity {
    fn name(&self) -> &str {
        "sparsity"
    }

    fn compute(&self, activations: &[f32], _unused: &[f32]) -> Result<f64> {
        if activations.is_empty() {
            return Ok(0.0);
        }

        let zeros = activations.iter().filter(|&&a| a.abs() < 1e-6).count();
        Ok(zeros as f64 / activations.len() as f64)
    }

    fn update(&mut self, activations: &[f32], _unused: &[f32]) {
        self.zero_count += activations.iter().filter(|&&a| a.abs() < 1e-6).count() as f64;
        self.total_count += activations.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.total_count == 0.0 {
            0.0
        } else {
            self.zero_count / self.total_count
        }
    }

    fn reset(&mut self) {
        self.zero_count = 0.0;
        self.total_count = 0.0;
    }
}

/// Synaptic Operations (SynOps).
#[derive(Debug, Clone)]
pub struct SynapticOperations {
    total_synops: f64,
    total_inferences: f64,
}

impl SynapticOperations {
    /// Creates a new [`SynapticOperations`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            total_synops: 0.0,
            total_inferences: 0.0,
        }
    }
}

impl Default for SynapticOperations {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for SynapticOperations {
    fn name(&self) -> &str {
        "synops"
    }

    fn compute(&self, spike_counts: &[f32], synapse_counts: &[f32]) -> Result<f64> {
        if spike_counts.len() != synapse_counts.len() {
            return Err(DpbError::Other("Spike counts and synapse counts must have the same length".to_string()));
        }

        let synops: f64 = spike_counts
            .iter()
            .zip(synapse_counts.iter())
            .map(|(s, w)| (*s as f64) * (*w as f64))
            .sum();

        Ok(synops)
    }

    fn update(&mut self, spike_counts: &[f32], synapse_counts: &[f32]) {
        let synops: f64 = spike_counts
            .iter()
            .zip(synapse_counts.iter())
            .map(|(s, w)| (*s as f64) * (*w as f64))
            .sum();

        self.total_synops += synops;
        self.total_inferences += 1.0;
    }

    fn result(&self) -> f64 {
        if self.total_inferences == 0.0 {
            0.0
        } else {
            self.total_synops / self.total_inferences
        }
    }

    fn reset(&mut self) {
        self.total_synops = 0.0;
        self.total_inferences = 0.0;
    }
}

/// Energy per Inference (joules).
#[derive(Debug, Clone)]
pub struct EnergyPerInference {
    total_energy: f64,
    total_inferences: f64,
}

impl EnergyPerInference {
    /// Creates a new [`EnergyPerInference`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            total_energy: 0.0,
            total_inferences: 0.0,
        }
    }
}

impl Default for EnergyPerInference {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for EnergyPerInference {
    fn name(&self) -> &str {
        "energy_per_inference"
    }

    fn compute(&self, energy_samples: &[f32], _unused: &[f32]) -> Result<f64> {
        if energy_samples.is_empty() {
            return Ok(0.0);
        }

        let total: f64 = energy_samples.iter().map(|&e| e as f64).sum();
        Ok(total / energy_samples.len() as f64)
    }

    fn update(&mut self, energy_samples: &[f32], _unused: &[f32]) {
        self.total_energy += energy_samples.iter().map(|&e| e as f64).sum::<f64>();
        self.total_inferences += energy_samples.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.total_inferences == 0.0 {
            0.0
        } else {
            self.total_energy / self.total_inferences
        }
    }

    fn reset(&mut self) {
        self.total_energy = 0.0;
        self.total_inferences = 0.0;
    }
}

/// Latency to First Spike (milliseconds).
#[derive(Debug, Clone)]
pub struct LatencyToFirstSpike {
    sum_latency: f64,
    count: f64,
}

impl LatencyToFirstSpike {
    /// Creates a new [`LatencyToFirstSpike`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            sum_latency: 0.0,
            count: 0.0,
        }
    }
}

impl Default for LatencyToFirstSpike {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for LatencyToFirstSpike {
    fn name(&self) -> &str {
        "latency_to_first_spike"
    }

    fn compute(&self, spike_times: &[f32], _unused: &[f32]) -> Result<f64> {
        if spike_times.is_empty() {
            return Ok(0.0);
        }

        // Find first non-zero spike time
        let first_spike = spike_times
            .iter()
            .find(|&&t| t > 0.0)
            .map(|&t| t as f64)
            .unwrap_or(0.0);

        Ok(first_spike)
    }

    fn update(&mut self, spike_times: &[f32], _unused: &[f32]) {
        if let Some(&first) = spike_times.iter().find(|&&t| t > 0.0) {
            self.sum_latency += first as f64;
            self.count += 1.0;
        }
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            0.0
        } else {
            self.sum_latency / self.count
        }
    }

    fn reset(&mut self) {
        self.sum_latency = 0.0;
        self.count = 0.0;
    }
}

/// Temporal Coding Efficiency.
#[derive(Debug, Clone)]
pub struct TemporalCodingEfficiency {
    information_bits: f64,
    total_spikes: f64,
}

impl TemporalCodingEfficiency {
    /// Creates a new [`TemporalCodingEfficiency`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            information_bits: 0.0,
            total_spikes: 0.0,
        }
    }
}

impl Default for TemporalCodingEfficiency {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for TemporalCodingEfficiency {
    fn name(&self) -> &str {
        "temporal_coding_efficiency"
    }

    fn compute(&self, spike_train: &[f32], _unused: &[f32]) -> Result<f64> {
        if spike_train.is_empty() {
            return Ok(0.0);
        }

        let spike_count = spike_train.iter().filter(|&&s| s > 0.0).count() as f64;

        if spike_count == 0.0 {
            return Ok(0.0);
        }

        // Compute entropy as a measure of information
        let prob = spike_count / spike_train.len() as f64;
        let entropy = if prob > 0.0 && prob < 1.0 {
            -prob * prob.log2() - (1.0 - prob) * (1.0 - prob).log2()
        } else {
            0.0
        };

        // Efficiency: bits per spike
        Ok(entropy * spike_train.len() as f64 / spike_count.max(1.0))
    }

    fn update(&mut self, spike_train: &[f32], _unused: &[f32]) {
        let spike_count = spike_train.iter().filter(|&&s| s > 0.0).count() as f64;
        let prob = spike_count / spike_train.len() as f64;

        let entropy = if prob > 0.0 && prob < 1.0 {
            -prob * prob.log2() - (1.0 - prob) * (1.0 - prob).log2()
        } else {
            0.0
        };

        self.information_bits += entropy * spike_train.len() as f64;
        self.total_spikes += spike_count;
    }

    fn result(&self) -> f64 {
        if self.total_spikes == 0.0 {
            0.0
        } else {
            self.information_bits / self.total_spikes
        }
    }

    fn reset(&mut self) {
        self.information_bits = 0.0;
        self.total_spikes = 0.0;
    }
}

/// Information Rate (bits per spike).
#[derive(Debug, Clone)]
pub struct InformationRate {
    total_bits: f64,
    total_spikes: f64,
}

impl InformationRate {
    /// Creates a new [`InformationRate`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            total_bits: 0.0,
            total_spikes: 0.0,
        }
    }
}

impl Default for InformationRate {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for InformationRate {
    fn name(&self) -> &str {
        "information_rate"
    }

    fn compute(&self, spikes: &[f32], bits: &[f32]) -> Result<f64> {
        if spikes.len() != bits.len() {
            return Err(DpbError::Other("Spikes and bits must have the same length".to_string()));
        }

        let total_spikes: f64 = spikes.iter().map(|&s| s as f64).sum();
        let total_bits: f64 = bits.iter().map(|&b| b as f64).sum();

        if total_spikes == 0.0 {
            Ok(0.0)
        } else {
            Ok(total_bits / total_spikes)
        }
    }

    fn update(&mut self, spikes: &[f32], bits: &[f32]) {
        self.total_spikes += spikes.iter().map(|&s| s as f64).sum::<f64>();
        self.total_bits += bits.iter().map(|&b| b as f64).sum::<f64>();
    }

    fn result(&self) -> f64 {
        if self.total_spikes == 0.0 {
            0.0
        } else {
            self.total_bits / self.total_spikes
        }
    }

    fn reset(&mut self) {
        self.total_bits = 0.0;
        self.total_spikes = 0.0;
    }
}

/// Membrane Utilization (average membrane potential usage).
#[derive(Debug, Clone)]
pub struct MembraneUtilization {
    sum_utilization: f64,
    count: f64,
}

impl MembraneUtilization {
    /// Creates a new [`MembraneUtilization`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            sum_utilization: 0.0,
            count: 0.0,
        }
    }
}

impl Default for MembraneUtilization {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for MembraneUtilization {
    fn name(&self) -> &str {
        "membrane_utilization"
    }

    fn compute(&self, potentials: &[f32], thresholds: &[f32]) -> Result<f64> {
        if potentials.len() != thresholds.len() {
            return Err(DpbError::Other("Potentials and thresholds must have the same length".to_string()));
        }
        if potentials.is_empty() {
            return Ok(0.0);
        }

        let utilization: f64 = potentials
            .iter()
            .zip(thresholds.iter())
            .map(|(p, t)| {
                if *t > 0.0 {
                    ((*p as f64) / (*t as f64)).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .sum();

        Ok(utilization / potentials.len() as f64)
    }

    fn update(&mut self, potentials: &[f32], thresholds: &[f32]) {
        let utilization: f64 = potentials
            .iter()
            .zip(thresholds.iter())
            .map(|(p, t)| {
                if *t > 0.0 {
                    ((*p as f64) / (*t as f64)).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .sum();

        self.sum_utilization += utilization;
        self.count += potentials.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            0.0
        } else {
            self.sum_utilization / self.count
        }
    }

    fn reset(&mut self) {
        self.sum_utilization = 0.0;
        self.count = 0.0;
    }
}

/// Weight Sparsity (proportion of zero weights).
#[derive(Debug, Clone)]
pub struct WeightSparsity {
    zero_weights: f64,
    total_weights: f64,
}

impl WeightSparsity {
    /// Creates a new [`WeightSparsity`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            zero_weights: 0.0,
            total_weights: 0.0,
        }
    }
}

impl Default for WeightSparsity {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for WeightSparsity {
    fn name(&self) -> &str {
        "weight_sparsity"
    }

    fn compute(&self, weights: &[f32], _unused: &[f32]) -> Result<f64> {
        if weights.is_empty() {
            return Ok(0.0);
        }

        let zeros = weights.iter().filter(|&&w| w.abs() < 1e-6).count();
        Ok(zeros as f64 / weights.len() as f64)
    }

    fn update(&mut self, weights: &[f32], _unused: &[f32]) {
        self.zero_weights += weights.iter().filter(|&&w| w.abs() < 1e-6).count() as f64;
        self.total_weights += weights.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.total_weights == 0.0 {
            0.0
        } else {
            self.zero_weights / self.total_weights
        }
    }

    fn reset(&mut self) {
        self.zero_weights = 0.0;
        self.total_weights = 0.0;
    }
}

/// Activation Sparsity (proportion of zero activations).
#[derive(Debug, Clone)]
pub struct ActivationSparsity {
    zero_activations: f64,
    total_activations: f64,
}

impl ActivationSparsity {
    /// Creates a new [`ActivationSparsity`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            zero_activations: 0.0,
            total_activations: 0.0,
        }
    }
}

impl Default for ActivationSparsity {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for ActivationSparsity {
    fn name(&self) -> &str {
        "activation_sparsity"
    }

    fn compute(&self, activations: &[f32], _unused: &[f32]) -> Result<f64> {
        if activations.is_empty() {
            return Ok(0.0);
        }

        let zeros = activations.iter().filter(|&&a| a.abs() < 1e-6).count();
        Ok(zeros as f64 / activations.len() as f64)
    }

    fn update(&mut self, activations: &[f32], _unused: &[f32]) {
        self.zero_activations += activations.iter().filter(|&&a| a.abs() < 1e-6).count() as f64;
        self.total_activations += activations.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.total_activations == 0.0 {
            0.0
        } else {
            self.zero_activations / self.total_activations
        }
    }

    fn reset(&mut self) {
        self.zero_activations = 0.0;
        self.total_activations = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_rate() {
        let mut sr = SpikeRate::new();
        let spikes = vec![10.0, 20.0, 15.0];
        let times = vec![1.0, 1.0, 1.0];

        let result = sr.compute(&spikes, &times).unwrap();
        assert_eq!(result, 45.0 / 3.0);

        sr.update(&spikes, &times);
        assert_eq!(sr.result(), 45.0 / 3.0);
    }

    #[test]
    fn test_sparsity() {
        let sparsity = Sparsity::new();
        let activations = vec![0.0, 0.0, 1.0, 0.0];
        let unused = vec![0.0; 4];

        let result = sparsity.compute(&activations, &unused).unwrap();
        assert_eq!(result, 0.75);
    }

    #[test]
    fn test_synops() {
        let synops = SynapticOperations::new();
        let spikes = vec![10.0, 20.0];
        let synapses = vec![100.0, 200.0];

        let result = synops.compute(&spikes, &synapses).unwrap();
        assert_eq!(result, 5000.0);  // 10*100 + 20*200
    }

    #[test]
    fn test_energy_per_inference() {
        let mut energy = EnergyPerInference::new();
        let samples = vec![1.0, 2.0, 3.0];
        let unused = vec![0.0; 3];

        let result = energy.compute(&samples, &unused).unwrap();
        assert_eq!(result, 2.0);

        energy.update(&samples, &unused);
        assert_eq!(energy.result(), 2.0);
    }

    #[test]
    fn test_latency_to_first_spike() {
        let latency = LatencyToFirstSpike::new();
        let spike_times = vec![0.0, 0.0, 5.0, 10.0];
        let unused = vec![0.0; 4];

        let result = latency.compute(&spike_times, &unused).unwrap();
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_information_rate() {
        let mut ir = InformationRate::new();
        let spikes = vec![10.0, 20.0];
        let bits = vec![100.0, 200.0];

        let result = ir.compute(&spikes, &bits).unwrap();
        assert_eq!(result, 300.0 / 30.0);

        ir.update(&spikes, &bits);
        assert_eq!(ir.result(), 300.0 / 30.0);
    }

    #[test]
    fn test_membrane_utilization() {
        let mem_util = MembraneUtilization::new();
        let potentials = vec![0.5, 0.75, 1.0];
        let thresholds = vec![1.0, 1.0, 1.0];

        let result = mem_util.compute(&potentials, &thresholds).unwrap();
        assert_eq!(result, 0.75);  // (0.5 + 0.75 + 1.0) / 3
    }

    #[test]
    fn test_weight_sparsity() {
        let ws = WeightSparsity::new();
        let weights = vec![0.0, 0.5, 0.0, 1.0];
        let unused = vec![0.0; 4];

        let result = ws.compute(&weights, &unused).unwrap();
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_activation_sparsity() {
        let as_metric = ActivationSparsity::new();
        let activations = vec![0.0, 0.0, 0.0, 1.0];
        let unused = vec![0.0; 4];

        let result = as_metric.compute(&activations, &unused).unwrap();
        assert_eq!(result, 0.75);
    }
}
