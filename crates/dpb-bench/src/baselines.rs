//! Baseline comparison implementations
//!
//! This module provides baseline implementations for comparing SNN performance
//! against traditional methods and ANNs.

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Result from a baseline model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResult {
    /// Model name
    pub name: String,
    /// Accuracy (0.0-1.0)
    pub accuracy: f64,
    /// Inference latency (ms)
    pub latency_ms: f64,
    /// Energy estimate (mJ)
    pub energy_mj: f64,
    /// Memory usage (bytes)
    pub memory_bytes: usize,
    /// Number of operations
    pub num_operations: usize,
}

/// Comparison metrics between models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    /// Accuracy difference (SNN - baseline)
    pub accuracy_delta: f64,
    /// Speedup factor (baseline_time / snn_time)
    pub speedup: f64,
    /// Energy efficiency (baseline_energy / snn_energy)
    pub energy_efficiency: f64,
    /// Memory ratio (baseline_mem / snn_mem)
    pub memory_ratio: f64,
    /// Sparsity (for SNNs)
    pub sparsity: f64,
}

impl ComparisonMetrics {
    /// Compute comparison metrics
    pub fn compute(snn_result: &BaselineResult, baseline_result: &BaselineResult) -> Self {
        Self {
            accuracy_delta: snn_result.accuracy - baseline_result.accuracy,
            speedup: baseline_result.latency_ms / snn_result.latency_ms.max(0.001),
            energy_efficiency: baseline_result.energy_mj / snn_result.energy_mj.max(0.001),
            memory_ratio: baseline_result.memory_bytes as f64 / snn_result.memory_bytes.max(1) as f64,
            sparsity: 0.0, // To be filled by caller
        }
    }
}

/// Simple ANN baseline for comparison
#[derive(Debug, Clone)]
pub struct ANNBaseline {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Weights (simplified)
    weights: Vec<Vec<f32>>,
}

impl ANNBaseline {
    /// Create a new ANN baseline
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        use rand::RngExt;
        let mut rng = rand::rng();

        // Initialize random weights
        let mut weights = Vec::new();
        let mut prev_dim = input_dim;

        for &hidden_dim in &hidden_dims {
            let layer_weights: Vec<f32> = (0..prev_dim * hidden_dim)
                .map(|_| rng.random::<f32>() * 0.1 - 0.05)
                .collect();
            weights.push(layer_weights);
            prev_dim = hidden_dim;
        }

        // Output layer
        let output_weights: Vec<f32> = (0..prev_dim * output_dim)
            .map(|_| rng.random::<f32>() * 0.1 - 0.05)
            .collect();
        weights.push(output_weights);

        Self {
            input_dim,
            hidden_dims,
            output_dim,
            weights,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut activations = input.to_vec();

        for (layer_idx, layer_weights) in self.weights.iter().enumerate() {
            let output_dim = if layer_idx < self.hidden_dims.len() {
                self.hidden_dims[layer_idx]
            } else {
                self.output_dim
            };

            let mut output = vec![0.0; output_dim];

            for i in 0..output_dim {
                for j in 0..activations.len() {
                    output[i] += activations[j] * layer_weights[i * activations.len() + j];
                }
                // ReLU activation
                output[i] = output[i].max(0.0);
            }

            activations = output;
        }

        activations
    }

    /// Run benchmark
    pub fn benchmark(&self, inputs: &[Vec<f32>]) -> BaselineResult {
        let start = Instant::now();
        let mut total_ops = 0;

        for input in inputs {
            let _ = self.forward(input);
        }

        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0 / inputs.len() as f64;

        // Count operations
        let mut prev_dim = self.input_dim;
        for &hidden_dim in &self.hidden_dims {
            total_ops += prev_dim * hidden_dim * 2; // multiply-add
            prev_dim = hidden_dim;
        }
        total_ops += prev_dim * self.output_dim * 2;
        total_ops *= inputs.len();

        // Estimate energy (based on MAC operations)
        let energy_per_mac = 4.6e-12; // J (45nm CMOS)
        let energy_mj = total_ops as f64 * energy_per_mac * 1000.0;

        // Memory usage
        let memory_bytes = self.weights.iter().map(|w| w.len() * 4).sum();

        BaselineResult {
            name: "ANN".to_string(),
            accuracy: 0.0, // To be set by caller
            latency_ms,
            energy_mj,
            memory_bytes,
            num_operations: total_ops,
        }
    }
}

/// Conventional signal processing baseline
#[derive(Debug, Clone)]
pub struct ConventionalBaseline {
    /// Method name
    pub method: String,
}

impl ConventionalBaseline {
    /// Create new conventional baseline
    pub fn new(method: &str) -> Self {
        Self {
            method: method.to_string(),
        }
    }

    /// Peak detection (simple threshold)
    pub fn detect_peaks(&self, signal: &[f32], threshold: f32) -> Vec<usize> {
        let mut peaks = Vec::new();

        for i in 1..signal.len() - 1 {
            if signal[i] > threshold && signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
                peaks.push(i);
            }
        }

        peaks
    }

    /// Moving average filter
    pub fn moving_average(&self, signal: &[f32], window_size: usize) -> Vec<f32> {
        let mut filtered = Vec::with_capacity(signal.len());

        for i in 0..signal.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(signal.len());

            let sum: f32 = signal[start..end].iter().sum();
            filtered.push(sum / (end - start) as f32);
        }

        filtered
    }

    /// Run benchmark
    pub fn benchmark(&self, signals: &[Vec<f32>]) -> BaselineResult {
        let start = Instant::now();
        let mut total_ops = 0;

        for signal in signals {
            match self.method.as_str() {
                "peak_detection" => {
                    let _ = self.detect_peaks(signal, 0.5);
                    total_ops += signal.len() * 3; // comparisons
                }
                "moving_average" => {
                    let _ = self.moving_average(signal, 5);
                    total_ops += signal.len() * 10; // additions
                }
                _ => {}
            }
        }

        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0 / signals.len() as f64;

        // Much lower energy for simple operations
        let energy_per_op = 0.1e-12; // J
        let energy_mj = total_ops as f64 * energy_per_op * 1000.0;

        BaselineResult {
            name: format!("Conventional_{}", self.method),
            accuracy: 0.0,
            latency_ms,
            energy_mj,
            memory_bytes: 1024, // Minimal memory
            num_operations: total_ops,
        }
    }
}

/// SNN baseline for reference
#[derive(Debug, Clone)]
pub struct SNNBaseline {
    /// Number of neurons
    pub num_neurons: usize,
    /// Number of synapses
    pub num_synapses: usize,
    /// Simulation timesteps
    pub timesteps: usize,
}

impl SNNBaseline {
    /// Create new SNN baseline
    pub fn new(num_neurons: usize, num_synapses: usize, timesteps: usize) -> Self {
        Self {
            num_neurons,
            num_synapses,
            timesteps,
        }
    }

    /// Estimate SNN performance
    pub fn benchmark(&self, num_samples: usize, avg_sparsity: f64) -> BaselineResult {
        // SNN operations are sparse and event-driven
        let spikes_per_neuron = self.timesteps as f64 * avg_sparsity;
        let total_spikes = self.num_neurons as f64 * spikes_per_neuron * num_samples as f64;

        // Each spike processes its synapses
        let avg_fanout = self.num_synapses as f64 / self.num_neurons as f64;
        let total_ops = (total_spikes * avg_fanout) as usize;

        // Neuromorphic energy per spike-synaptic operation
        let energy_per_spike_op = 50e-12; // J (much lower than MAC)
        let energy_mj = total_ops as f64 * energy_per_spike_op * 1000.0;

        // Latency depends on timesteps
        let latency_per_step = 0.01; // ms
        let latency_ms = self.timesteps as f64 * latency_per_step;

        // Memory for neuron states and synaptic weights
        let memory_bytes = self.num_neurons * 16 + self.num_synapses * 4;

        BaselineResult {
            name: "SNN".to_string(),
            accuracy: 0.0,
            latency_ms,
            energy_mj,
            memory_bytes,
            num_operations: total_ops,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ann_baseline() {
        let ann = ANNBaseline::new(10, vec![20, 10], 3);
        let input = vec![0.5; 10];
        let output = ann.forward(&input);

        assert_eq!(output.len(), 3);

        let inputs = vec![vec![0.5; 10]; 100];
        let result = ann.benchmark(&inputs);

        assert!(result.latency_ms > 0.0);
        assert!(result.num_operations > 0);
    }

    #[test]
    fn test_conventional_baseline() {
        let baseline = ConventionalBaseline::new("peak_detection");
        let signal = vec![0.0, 0.5, 1.0, 0.5, 0.0, 0.3, 0.9, 0.2];
        let peaks = baseline.detect_peaks(&signal, 0.4);

        assert!(!peaks.is_empty());
    }

    #[test]
    fn test_snn_baseline() {
        let snn = SNNBaseline::new(100, 1000, 50);
        let result = snn.benchmark(10, 0.1);

        assert!(result.latency_ms > 0.0);
        assert!(result.energy_mj > 0.0);
    }

    #[test]
    fn test_comparison_metrics() {
        let snn = BaselineResult {
            name: "SNN".to_string(),
            accuracy: 0.95,
            latency_ms: 1.0,
            energy_mj: 0.1,
            memory_bytes: 1000,
            num_operations: 500,
        };

        let ann = BaselineResult {
            name: "ANN".to_string(),
            accuracy: 0.96,
            latency_ms: 5.0,
            energy_mj: 1.0,
            memory_bytes: 10000,
            num_operations: 5000,
        };

        let metrics = ComparisonMetrics::compute(&snn, &ann);

        assert!((metrics.accuracy_delta - (-0.01)).abs() < 1e-6);
        assert_eq!(metrics.speedup, 5.0);
        assert_eq!(metrics.energy_efficiency, 10.0);
    }
}
