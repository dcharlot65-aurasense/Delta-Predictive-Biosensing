//! ANN-to-SNN conversion utilities

use crate::{
    architectures::{FeedforwardSNN, SNNArchitecture},
    layers::{SpikingLayer, SpikingLinear},
    SNNConfig, SNNError, SNNResult, SpikeTensor,
};
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

/// ANN to SNN converter
#[derive(Debug, Clone)]
pub struct ANNToSNNConverter {
    /// Weight normalization method
    pub weight_norm: WeightNormalization,
    /// Threshold balancing method
    pub threshold_balance: ThresholdBalancing,
    /// Target spike rate
    pub target_spike_rate: f32,
    /// Number of time steps for SNN simulation
    pub num_steps: usize,
}

impl ANNToSNNConverter {
    pub fn new(
        weight_norm: WeightNormalization,
        threshold_balance: ThresholdBalancing,
        target_spike_rate: f32,
        num_steps: usize,
    ) -> Self {
        Self {
            weight_norm,
            threshold_balance,
            target_spike_rate,
            num_steps,
        }
    }

    /// Convert ANN weights to SNN
    pub fn convert_weights(
        &self,
        ann_weights: &[Array2<f32>],
        config: SNNConfig,
    ) -> SNNResult<FeedforwardSNN> {
        // Extract layer sizes from weight matrices
        let mut layer_sizes = vec![ann_weights[0].shape()[1]]; // Input size
        for w in ann_weights {
            layer_sizes.push(w.shape()[0]); // Output size of each layer
        }

        // Create SNN with same architecture
        let mut snn = FeedforwardSNN::new(layer_sizes, config, true);

        // Copy and normalize weights
        for (i, ann_weight) in ann_weights.iter().enumerate() {
            if let Some(snn_layer) = snn.layer_mut(i) {
                let normalized_weight = self.weight_norm.normalize(ann_weight);
                snn_layer.weights = normalized_weight;

                // Apply threshold balancing
                let threshold_scale = self.threshold_balance.compute_scale(&snn_layer.weights);
                snn_layer.neuron_params.v_threshold *= threshold_scale;
            }
        }

        Ok(snn)
    }

    /// Calibrate SNN thresholds using data
    pub fn calibrate(
        &self,
        snn: &mut FeedforwardSNN,
        calibration_data: &[SpikeTensor],
    ) -> SNNResult<()> {
        // Forward pass through calibration data to collect activation statistics
        let mut layer_activations: Vec<Vec<f32>> = vec![Vec::new(); snn.num_layers()];

        for data in calibration_data {
            let mut current = data.clone();

            for (layer_idx, layer) in snn.layers.iter_mut().enumerate() {
                current = layer.forward(&current)?;

                // Collect spike rates
                let rates = current.spike_rate();
                let mean_rate = rates.mean().unwrap_or(0.0);
                layer_activations[layer_idx].push(mean_rate);
            }

            snn.reset();
        }

        // Adjust thresholds based on activation statistics
        for (layer_idx, layer) in snn.layers.iter_mut().enumerate() {
            if let Some(&mean_activation) = layer_activations[layer_idx].first() {
                if mean_activation > 0.0 {
                    // Scale threshold to achieve target spike rate
                    let scale = mean_activation / self.target_spike_rate;
                    layer.neuron_params.v_threshold *= scale;
                }
            }
        }

        Ok(())
    }
}

/// Weight normalization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightNormalization {
    /// No normalization
    None,
    /// Data-based normalization
    DataBased,
    /// Model-based normalization
    ModelBased,
    /// Max-weight normalization
    MaxWeight,
}

impl WeightNormalization {
    pub fn normalize(&self, weights: &Array2<f32>) -> Array2<f32> {
        match self {
            WeightNormalization::None => weights.clone(),
            WeightNormalization::DataBased => {
                // Normalize by 99th percentile
                let mut sorted: Vec<f32> = weights.iter().map(|&x| x.abs()).collect();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let percentile_99 = sorted[(sorted.len() as f32 * 0.99) as usize];

                if percentile_99 > 0.0 {
                    weights / percentile_99
                } else {
                    weights.clone()
                }
            }
            WeightNormalization::ModelBased => {
                // Normalize by maximum absolute weight per neuron
                let mut normalized = weights.clone();
                for i in 0..weights.shape()[0] {
                    let row_max = weights
                        .row(i)
                        .iter()
                        .map(|x| x.abs())
                        .fold(0.0f32, |a, b| a.max(b));

                    if row_max > 0.0 {
                        for j in 0..weights.shape()[1] {
                            normalized[[i, j]] = weights[[i, j]] / row_max;
                        }
                    }
                }
                normalized
            }
            WeightNormalization::MaxWeight => {
                // Simple max normalization
                let max_weight = weights.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b));
                if max_weight > 0.0 {
                    weights / max_weight
                } else {
                    weights.clone()
                }
            }
        }
    }
}

/// Threshold balancing methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdBalancing {
    /// No balancing
    None,
    /// Weight-based balancing
    WeightBased,
    /// Layer-wise balancing
    LayerWise,
}

impl ThresholdBalancing {
    pub fn compute_scale(&self, weights: &Array2<f32>) -> f32 {
        match self {
            ThresholdBalancing::None => 1.0,
            ThresholdBalancing::WeightBased => {
                // Scale based on average incoming weight magnitude
                let weight_sum: f32 = weights.iter().map(|x| x.abs()).sum();
                let weight_count = (weights.shape()[0] * weights.shape()[1]) as f32;
                let avg_weight = weight_sum / weight_count;

                if avg_weight > 0.0 {
                    1.0 / avg_weight
                } else {
                    1.0
                }
            }
            ThresholdBalancing::LayerWise => {
                // Scale based on maximum weight per output neuron
                let mut max_incoming = 0.0f32;
                for i in 0..weights.shape()[0] {
                    let row_sum: f32 = weights.row(i).iter().map(|x| x.abs()).sum();
                    max_incoming = max_incoming.max(row_sum);
                }

                if max_incoming > 0.0f32 {
                    1.0f32 / max_incoming
                } else {
                    1.0f32
                }
            }
        }
    }
}

/// Calibration pipeline for converted SNNs
pub struct CalibrationPipeline {
    /// Number of calibration samples
    pub num_samples: usize,
    /// Target accuracy threshold
    pub target_accuracy: f32,
    /// Maximum calibration iterations
    pub max_iterations: usize,
}

impl CalibrationPipeline {
    pub fn new(num_samples: usize, target_accuracy: f32, max_iterations: usize) -> Self {
        Self {
            num_samples,
            target_accuracy,
            max_iterations,
        }
    }

    /// Run calibration pipeline
    pub fn calibrate(
        &self,
        converter: &ANNToSNNConverter,
        snn: &mut FeedforwardSNN,
        calibration_data: &[SpikeTensor],
        validation_fn: impl Fn(&FeedforwardSNN) -> f32,
    ) -> SNNResult<CalibrationStats> {
        let mut stats = CalibrationStats::new();

        for iteration in 0..self.max_iterations {
            // Run calibration
            converter.calibrate(snn, &calibration_data[..self.num_samples.min(calibration_data.len())])?;

            // Validate
            let accuracy = validation_fn(snn);
            stats.accuracies.push(accuracy);

            if accuracy >= self.target_accuracy {
                stats.converged = true;
                stats.final_accuracy = accuracy;
                return Ok(stats);
            }
        }

        stats.converged = false;
        stats.final_accuracy = *stats.accuracies.last().unwrap_or(&0.0);
        Ok(stats)
    }
}

/// Calibration statistics
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    pub accuracies: Vec<f32>,
    pub converged: bool,
    pub final_accuracy: f32,
}

impl CalibrationStats {
    pub fn new() -> Self {
        Self {
            accuracies: Vec::new(),
            converged: false,
            final_accuracy: 0.0,
        }
    }
}

impl Default for CalibrationStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_normalization_none() {
        let weights = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let normalized = WeightNormalization::None.normalize(&weights);
        assert_eq!(weights, normalized);
    }

    #[test]
    fn test_weight_normalization_max_weight() {
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let normalized = WeightNormalization::MaxWeight.normalize(&weights);

        // Should be normalized by max (4.0)
        assert_eq!(normalized[[1, 1]], 1.0);
        assert_eq!(normalized[[0, 0]], 0.25);
    }

    #[test]
    fn test_threshold_balancing_none() {
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let scale = ThresholdBalancing::None.compute_scale(&weights);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_threshold_balancing_weight_based() {
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let scale = ThresholdBalancing::WeightBased.compute_scale(&weights);
        assert_eq!(scale, 1.0); // Average weight is 1.0, so scale is 1/1
    }

    #[test]
    fn test_ann_to_snn_converter() {
        let converter = ANNToSNNConverter::new(
            WeightNormalization::MaxWeight,
            ThresholdBalancing::WeightBased,
            0.2,
            100,
        );

        let ann_weights = vec![
            Array2::from_shape_vec((5, 10), vec![1.0; 50]).unwrap(),
            Array2::from_shape_vec((3, 5), vec![1.0; 15]).unwrap(),
        ];

        let snn = converter.convert_weights(&ann_weights, SNNConfig::default()).unwrap();
        assert_eq!(snn.num_layers(), 2);
    }
}
