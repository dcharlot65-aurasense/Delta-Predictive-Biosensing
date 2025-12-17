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

/// Windowed rate decoder - sliding window rate computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowedRateDecoder {
    pub num_outputs: usize,
    pub window_size: usize,
    pub stride: usize,
}

impl WindowedRateDecoder {
    pub fn new(num_outputs: usize, window_size: usize, stride: usize) -> Self {
        Self {
            num_outputs,
            window_size,
            stride,
        }
    }
}

impl Decoder for WindowedRateDecoder {
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

        let num_windows = (num_steps.saturating_sub(self.window_size)) / self.stride + 1;
        let mut output = Array2::zeros((batch_size, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                let mut max_rate = 0.0f32;

                for w in 0..num_windows {
                    let start = w * self.stride;
                    let end = (start + self.window_size).min(num_steps);

                    let spike_count: f32 = spike_dense.slice(s![b, start..end, n]).sum();
                    let rate = spike_count / self.window_size as f32;
                    max_rate = max_rate.max(rate);
                }

                output[[b, n]] = max_rate;
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_outputs
    }
}

/// Exponential rate decoder - exponentially weighted moving average
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialRateDecoder {
    pub num_outputs: usize,
    pub alpha: f32, // Smoothing factor (0 < alpha <= 1)
}

impl ExponentialRateDecoder {
    pub fn new(num_outputs: usize, alpha: f32) -> Self {
        Self {
            num_outputs,
            alpha: alpha.clamp(0.01, 1.0),
        }
    }
}

impl Decoder for ExponentialRateDecoder {
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
                let mut ema = 0.0f32;

                for t in 0..num_steps {
                    let spike_val = spike_dense[[b, t, n]];
                    ema = self.alpha * spike_val + (1.0 - self.alpha) * ema;
                }

                output[[b, n]] = ema;
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_outputs
    }
}

/// Adaptive rate decoder - adaptive threshold based on activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRateDecoder {
    pub num_outputs: usize,
    pub percentile: f32, // Activity percentile to use as threshold (0-100)
}

impl AdaptiveRateDecoder {
    pub fn new(num_outputs: usize, percentile: f32) -> Self {
        Self {
            num_outputs,
            percentile: percentile.clamp(0.0, 100.0),
        }
    }
}

impl Decoder for AdaptiveRateDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let (batch_size, num_neurons) = (rates.shape()[0], rates.shape()[1]);

        if num_neurons != self.num_outputs {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} neurons", self.num_outputs),
                actual: format!("{} neurons", num_neurons),
            });
        }

        let mut output = Array2::zeros((batch_size, num_neurons));

        for b in 0..batch_size {
            let mut sorted_rates: Vec<f32> = rates.row(b).to_vec();
            sorted_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let threshold_idx = ((self.percentile / 100.0) * sorted_rates.len() as f32) as usize;
            let threshold = sorted_rates.get(threshold_idx).copied().unwrap_or(0.0);

            for n in 0..num_neurons {
                let rate = rates[[b, n]];
                output[[b, n]] = if rate > threshold { rate } else { 0.0 };
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_outputs
    }
}

/// Normalized rate decoder - min-max normalization per batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedRateDecoder {
    pub num_outputs: usize,
}

impl NormalizedRateDecoder {
    pub fn new(num_outputs: usize) -> Self {
        Self { num_outputs }
    }
}

impl Decoder for NormalizedRateDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let (batch_size, num_neurons) = (rates.shape()[0], rates.shape()[1]);

        if num_neurons != self.num_outputs {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} neurons", self.num_outputs),
                actual: format!("{} neurons", num_neurons),
            });
        }

        let mut output = rates.clone();

        for mut row in output.rows_mut() {
            let min_val = row.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_val - min_val;

            if range > 1e-8 {
                for val in row.iter_mut() {
                    *val = (*val - min_val) / range;
                }
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_outputs
    }
}

/// Weighted rate decoder - channel importance weighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedRateDecoder {
    pub num_outputs: usize,
    pub weights: Array1<f32>,
}

impl WeightedRateDecoder {
    pub fn new(num_outputs: usize) -> Self {
        Self {
            num_outputs,
            weights: Array1::from_elem(num_outputs, 1.0),
        }
    }

    pub fn with_weights(mut self, weights: Array1<f32>) -> Self {
        self.weights = weights;
        self
    }
}

impl Decoder for WeightedRateDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let (batch_size, num_neurons) = (rates.shape()[0], rates.shape()[1]);

        if num_neurons != self.num_outputs {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} neurons", self.num_outputs),
                actual: format!("{} neurons", num_neurons),
            });
        }

        if self.weights.len() != num_neurons {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} weights", num_neurons),
                actual: format!("{} weights", self.weights.len()),
            });
        }

        let mut output = Array2::zeros((batch_size, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                output[[b, n]] = rates[[b, n]] * self.weights[n];
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

    #[test]
    fn test_windowed_rate_decoder() {
        let decoder = WindowedRateDecoder::new(2, 5, 2);

        let mut spike_data = Array3::zeros((1, 10, 2));
        spike_data[[0, 2, 0]] = 1.0;
        spike_data[[0, 3, 0]] = 1.0;

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert!(output[[0, 0]] > 0.0);
    }

    #[test]
    fn test_exponential_rate_decoder() {
        let decoder = ExponentialRateDecoder::new(2, 0.5);

        let mut spike_data = Array3::zeros((1, 10, 2));
        spike_data[[0, 8, 0]] = 1.0;
        spike_data[[0, 9, 0]] = 1.0;

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert!(output[[0, 0]] > 0.0);
    }

    #[test]
    fn test_normalized_rate_decoder() {
        let decoder = NormalizedRateDecoder::new(3);

        let mut spike_data = Array3::zeros((1, 10, 3));
        spike_data[[0, 2, 0]] = 1.0;
        spike_data[[0, 5, 0]] = 1.0;
        spike_data[[0, 3, 1]] = 1.0;

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        let max_val = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        assert!((max_val - 1.0).abs() < 1e-5);
    }
}
