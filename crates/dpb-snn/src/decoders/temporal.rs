//! Temporal pattern-based spike decoders

use super::Decoder;
use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{Array1, Array2, Array3, s};
use serde::{Deserialize, Serialize};

/// Temporal pattern decoder - recognizes specific spike patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatternDecoder {
    /// Number of output classes
    pub num_classes: usize,
    /// Template patterns for each class (class, time, neurons)
    pub templates: Vec<Array2<f32>>,
    /// Similarity threshold
    pub threshold: f32,
}

impl TemporalPatternDecoder {
    pub fn new(num_classes: usize, threshold: f32) -> Self {
        Self {
            num_classes,
            templates: Vec::new(),
            threshold,
        }
    }

    pub fn with_templates(mut self, templates: Vec<Array2<f32>>) -> Self {
        self.templates = templates;
        self
    }

    /// Compute similarity between spike train and template
    fn compute_similarity(&self, spikes: &Array2<f32>, template: &Array2<f32>) -> f32 {
        if spikes.shape() != template.shape() {
            return 0.0;
        }

        // Normalized cross-correlation
        let dot_product: f32 = spikes
            .iter()
            .zip(template.iter())
            .map(|(s, t)| s * t)
            .sum();

        let spike_norm: f32 = spikes.iter().map(|s| s * s).sum::<f32>().sqrt();
        let template_norm: f32 = template.iter().map(|t| t * t).sum::<f32>().sqrt();

        if spike_norm > 0.0 && template_norm > 0.0 {
            dot_product / (spike_norm * template_norm)
        } else {
            0.0
        }
    }
}

impl Decoder for TemporalPatternDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, self.num_classes));

        for b in 0..batch_size {
            let spike_pattern = spike_dense.slice(s![b, .., ..]).to_owned();

            // Compute similarity to each template
            for (c, template) in self.templates.iter().enumerate() {
                if c >= self.num_classes {
                    break;
                }

                let similarity = self.compute_similarity(&spike_pattern, template);
                output[[b, c]] = if similarity > self.threshold {
                    similarity
                } else {
                    0.0
                };
            }

            // Normalize to probabilities
            let sum: f32 = output.row(b).sum();
            if sum > 0.0 {
                for c in 0..self.num_classes {
                    output[[b, c]] /= sum;
                }
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_classes
    }
}

/// Latency decoder - uses relative spike timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyDecoder {
    /// Number of output neurons
    pub num_outputs: usize,
    /// Time window for considering spikes
    pub time_window: usize,
    /// Latency weights (earlier = higher weight)
    pub use_exponential_weights: bool,
    /// Decay constant for exponential weights
    pub decay_constant: f32,
}

impl LatencyDecoder {
    pub fn new(num_outputs: usize, time_window: usize) -> Self {
        Self {
            num_outputs,
            time_window,
            use_exponential_weights: true,
            decay_constant: 0.1,
        }
    }

    fn compute_latency_score(&self, spike_times: &[usize]) -> f32 {
        if spike_times.is_empty() {
            return 0.0;
        }

        if self.use_exponential_weights {
            // Weighted sum with exponential decay
            spike_times
                .iter()
                .map(|&t| (-self.decay_constant * t as f32).exp())
                .sum()
        } else {
            // Simple inverse latency
            spike_times
                .iter()
                .map(|&t| 1.0 / (t as f32 + 1.0))
                .sum()
        }
    }
}

impl Decoder for LatencyDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        if num_neurons != self.num_outputs {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} neurons", self.num_outputs),
                actual: format!("{} neurons", num_neurons),
            });
        }

        let mut output = Array2::zeros((batch_size, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                // Collect spike times for this neuron
                let mut spike_times = Vec::new();
                for t in 0..num_steps.min(self.time_window) {
                    if spike_dense[[b, t, n]] > 0.5 {
                        spike_times.push(t);
                    }
                }

                output[[b, n]] = self.compute_latency_score(&spike_times);
            }
        }

        // Normalize
        for mut row in output.rows_mut() {
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_outputs
    }
}

/// Inter-spike interval (ISI) decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ISIDecoder {
    /// Number of output neurons
    pub num_outputs: usize,
    /// Target ISI patterns for each class
    pub target_isis: Vec<f32>,
}

impl ISIDecoder {
    pub fn new(num_outputs: usize) -> Self {
        Self {
            num_outputs,
            target_isis: Vec::new(),
        }
    }

    fn compute_mean_isi(&self, spike_train: &[f32]) -> f32 {
        let spike_times: Vec<usize> = spike_train
            .iter()
            .enumerate()
            .filter(|(_, s)| **s > 0.5)
            .map(|(t, _)| t)
            .collect();

        if spike_times.len() < 2 {
            return f32::INFINITY;
        }

        let isis: Vec<f32> = spike_times
            .windows(2)
            .map(|w| (w[1] - w[0]) as f32)
            .collect();

        isis.iter().sum::<f32>() / isis.len() as f32
    }
}

impl Decoder for ISIDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                let spike_train: Vec<f32> = spike_dense.slice(s![b, .., n]).to_vec();
                let mean_isi = self.compute_mean_isi(&spike_train);

                // Score based on ISI (lower ISI = higher score)
                output[[b, n]] = if mean_isi.is_finite() {
                    1.0 / (mean_isi + 1.0)
                } else {
                    0.0
                };
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_outputs
    }
}

/// Burst decoder - detects burst patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstDecoder {
    /// Number of output neurons
    pub num_outputs: usize,
    /// Burst threshold (spikes within window)
    pub burst_threshold: usize,
    /// Time window for burst detection
    pub burst_window: usize,
}

impl BurstDecoder {
    pub fn new(num_outputs: usize, burst_threshold: usize, burst_window: usize) -> Self {
        Self {
            num_outputs,
            burst_threshold,
            burst_window,
        }
    }

    fn detect_bursts(&self, spike_train: &[f32]) -> usize {
        let spike_times: Vec<usize> = spike_train
            .iter()
            .enumerate()
            .filter(|(_, s)| **s > 0.5)
            .map(|(t, _)| t)
            .collect();

        let mut burst_count = 0;

        for i in 0..spike_times.len() {
            let start_time = spike_times[i];
            let spikes_in_window = spike_times
                .iter()
                .filter(|&&t| t >= start_time && t < start_time + self.burst_window)
                .count();

            if spikes_in_window >= self.burst_threshold {
                burst_count += 1;
            }
        }

        burst_count
    }
}

impl Decoder for BurstDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                let spike_train: Vec<f32> = spike_dense.slice(s![b, .., n]).to_vec();
                let burst_count = self.detect_bursts(&spike_train);
                output[[b, n]] = burst_count as f32;
            }
        }

        // Normalize
        for mut row in output.rows_mut() {
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_latency_decoder() {
        let decoder = LatencyDecoder::new(3, 20);

        let mut spike_data = Array3::zeros((1, 20, 3));
        spike_data[[0, 2, 0]] = 1.0; // Early spike
        spike_data[[0, 15, 1]] = 1.0; // Late spike
        spike_data[[0, 5, 2]] = 1.0; // Medium spike

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        // Earlier spike should have higher score
        assert!(output[[0, 0]] > output[[0, 1]]);
        assert!(output[[0, 2]] > output[[0, 1]]);
    }

    #[test]
    fn test_isi_decoder() {
        let decoder = ISIDecoder::new(2);

        let mut spike_data = Array3::zeros((1, 10, 2));
        // Regular ISI for neuron 0
        spike_data[[0, 2, 0]] = 1.0;
        spike_data[[0, 4, 0]] = 1.0;
        spike_data[[0, 6, 0]] = 1.0;

        // Irregular ISI for neuron 1
        spike_data[[0, 1, 1]] = 1.0;
        spike_data[[0, 8, 1]] = 1.0;

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert!(output[[0, 0]] > 0.0);
        assert!(output[[0, 1]] > 0.0);
    }

    #[test]
    fn test_burst_decoder() {
        let decoder = BurstDecoder::new(2, 3, 5);

        let mut spike_data = Array3::zeros((1, 20, 2));
        // Burst in neuron 0
        spike_data[[0, 2, 0]] = 1.0;
        spike_data[[0, 3, 0]] = 1.0;
        spike_data[[0, 4, 0]] = 1.0;

        // No burst in neuron 1
        spike_data[[0, 1, 1]] = 1.0;
        spike_data[[0, 10, 1]] = 1.0;

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        // Neuron 0 should have higher score due to burst
        assert!(output[[0, 0]] > output[[0, 1]]);
    }
}
