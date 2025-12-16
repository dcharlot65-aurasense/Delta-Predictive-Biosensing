//! Rate-based spike decoders

use super::Decoder;
use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{Array1, Array2, Axis, s};
use serde::{Deserialize, Serialize};

/// Spike rate decoder - decodes based on average firing rate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeRateDecoder {
    /// Number of output neurons
    pub num_outputs: usize,
    /// Time window for rate computation (None = use all time)
    pub time_window: Option<usize>,
    /// Apply softmax to rates
    pub use_softmax: bool,
}

impl SpikeRateDecoder {
    pub fn new(num_outputs: usize, time_window: Option<usize>, use_softmax: bool) -> Self {
        Self {
            num_outputs,
            time_window,
            use_softmax,
        }
    }

    fn softmax(&self, rates: &mut Array2<f32>) {
        for mut row in rates.rows_mut() {
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();

            for val in row.iter_mut() {
                *val = (*val - max_val).exp() / exp_sum;
            }
        }
    }
}

impl Decoder for SpikeRateDecoder {
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

        // Compute firing rates
        let window_size = self.time_window.unwrap_or(num_steps);
        let start_time = num_steps.saturating_sub(window_size);

        let mut rates = Array2::zeros((batch_size, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                let spike_count: f32 = spike_dense.slice(s![b, start_time..num_steps, n]).sum();
                rates[[b, n]] = spike_count / window_size as f32;
            }
        }

        // Apply softmax if requested
        if self.use_softmax {
            self.softmax(&mut rates);
        }

        Ok(rates)
    }

    fn output_dim(&self) -> usize {
        self.num_outputs
    }
}

/// First spike decoder - uses time of first spike
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstSpikeDecoder {
    /// Number of output neurons
    pub num_outputs: usize,
    /// Maximum time to consider
    pub max_time: usize,
}

impl FirstSpikeDecoder {
    pub fn new(num_outputs: usize, max_time: usize) -> Self {
        Self {
            num_outputs,
            max_time,
        }
    }
}

impl Decoder for FirstSpikeDecoder {
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
                // Find first spike time
                let mut first_spike = self.max_time as f32;
                for t in 0..num_steps.min(self.max_time) {
                    if spike_dense[[b, t, n]] > 0.5 {
                        first_spike = t as f32;
                        break;
                    }
                }

                // Convert to score (earlier = higher score)
                output[[b, n]] = 1.0 - (first_spike / self.max_time as f32);
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_outputs
    }
}

/// Population decoder - uses weighted population voting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationDecoder {
    /// Number of classes
    pub num_classes: usize,
    /// Neurons per class
    pub neurons_per_class: usize,
    /// Voting weights for each neuron
    pub weights: Option<Array2<f32>>,
}

impl PopulationDecoder {
    pub fn new(num_classes: usize, neurons_per_class: usize) -> Self {
        Self {
            num_classes,
            neurons_per_class,
            weights: None,
        }
    }

    pub fn with_weights(mut self, weights: Array2<f32>) -> Self {
        self.weights = Some(weights);
        self
    }
}

impl Decoder for PopulationDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let total_neurons = self.num_classes * self.neurons_per_class;
        if num_neurons != total_neurons {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} neurons", total_neurons),
                actual: format!("{} neurons", num_neurons),
            });
        }

        let mut output = Array2::zeros((batch_size, self.num_classes));

        for b in 0..batch_size {
            for c in 0..self.num_classes {
                let neuron_start = c * self.neurons_per_class;
                let neuron_end = neuron_start + self.neurons_per_class;

                // Sum spikes from neurons representing this class
                let mut class_activity = 0.0;
                for n in neuron_start..neuron_end {
                    let spike_count: f32 = spike_dense.slice(s![b, .., n]).sum();

                    // Apply weight if provided
                    let weight = if let Some(ref w) = self.weights {
                        w[[c, n - neuron_start]]
                    } else {
                        1.0
                    };

                    class_activity += spike_count * weight;
                }

                output[[b, c]] = class_activity / num_steps as f32;
            }
        }

        // Normalize to probabilities
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
        self.num_classes
    }
}

/// Maximum spike count decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxSpikeDecoder {
    pub num_outputs: usize,
}

impl MaxSpikeDecoder {
    pub fn new(num_outputs: usize) -> Self {
        Self { num_outputs }
    }
}

impl Decoder for MaxSpikeDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();

        // One-hot encode based on maximum rate
        let mut output = Array2::zeros(rates.raw_dim());

        for (b, row) in rates.rows().into_iter().enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            output[[b, max_idx]] = 1.0;
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
    fn test_spike_rate_decoder() {
        let decoder = SpikeRateDecoder::new(3, None, false);

        let mut spike_data = Array3::zeros((2, 10, 3));
        spike_data[[0, 2, 0]] = 1.0;
        spike_data[[0, 5, 0]] = 1.0;
        spike_data[[1, 3, 1]] = 1.0;

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output[[0, 0]], 0.2); // 2 spikes / 10 steps
        assert_eq!(output[[1, 1]], 0.1); // 1 spike / 10 steps
    }

    #[test]
    fn test_spike_rate_decoder_with_softmax() {
        let decoder = SpikeRateDecoder::new(3, None, true);

        let mut spike_data = Array3::zeros((1, 10, 3));
        spike_data[[0, 2, 0]] = 1.0;
        spike_data[[0, 5, 1]] = 1.0;

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        // Check probabilities sum to 1
        let sum: f32 = output.row(0).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_first_spike_decoder() {
        let decoder = FirstSpikeDecoder::new(3, 10);

        let mut spike_data = Array3::zeros((1, 10, 3));
        spike_data[[0, 2, 0]] = 1.0; // Early spike
        spike_data[[0, 8, 1]] = 1.0; // Late spike

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        // Earlier spike should have higher score
        assert!(output[[0, 0]] > output[[0, 1]]);
    }

    #[test]
    fn test_population_decoder() {
        let decoder = PopulationDecoder::new(2, 2); // 2 classes, 2 neurons each

        let mut spike_data = Array3::zeros((1, 10, 4));
        // Class 0 neurons (0, 1) spike more
        spike_data[[0, 2, 0]] = 1.0;
        spike_data[[0, 5, 0]] = 1.0;
        spike_data[[0, 7, 1]] = 1.0;

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        // Class 0 should have higher probability
        assert!(output[[0, 0]] > output[[0, 1]]);
    }

    #[test]
    fn test_max_spike_decoder() {
        let decoder = MaxSpikeDecoder::new(3);

        let mut spike_data = Array3::zeros((2, 10, 3));
        spike_data[[0, 2, 0]] = 1.0;
        spike_data[[0, 5, 0]] = 1.0; // Neuron 0 has most spikes in batch 0
        spike_data[[1, 3, 2]] = 1.0; // Neuron 2 has most spikes in batch 1

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output[[0, 0]], 1.0);
        assert_eq!(output[[1, 2]], 1.0);
    }
}
