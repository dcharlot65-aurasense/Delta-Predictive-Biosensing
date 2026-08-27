//! ANN-to-SNN conversion utilities for baseline models

use super::Tensor;

/// Weight normalization methods for ANN-to-SNN conversion
#[derive(Debug, Clone, Copy)]
pub enum WeightNormalizationMethod {
    /// Data-based normalization using max activation
    DataBased,
    /// Model-based normalization using weight statistics
    ModelBased,
    /// Combined data and model normalization
    Hybrid,
    /// No normalization
    None,
}

/// Threshold balancing strategies
#[derive(Debug, Clone, Copy)]
pub enum ThresholdBalancingStrategy {
    /// Fixed threshold for all layers
    Fixed(f32),
    /// Layer-wise adaptive threshold
    LayerWise,
    /// Percentile-based threshold
    Percentile(f32),
    /// Learned threshold
    Learned,
}

/// ANN-to-SNN conversion configuration
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Weight normalization method
    pub weight_norm: WeightNormalizationMethod,
    /// Threshold balancing strategy
    pub threshold_strategy: ThresholdBalancingStrategy,
    /// Number of time steps for SNN simulation
    pub num_timesteps: usize,
    /// Enable bias correction
    pub bias_correction: bool,
    /// Enable batch normalization folding
    pub fold_batchnorm: bool,
    /// Clip negative weights (for ReLU networks)
    pub clip_negative_weights: bool,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            weight_norm: WeightNormalizationMethod::DataBased,
            threshold_strategy: ThresholdBalancingStrategy::LayerWise,
            num_timesteps: 100,
            bias_correction: true,
            fold_batchnorm: true,
            clip_negative_weights: false,
        }
    }
}

/// ANN-to-SNN converter
pub struct ANNToSNNConverter {
    config: ConversionConfig,
    layer_thresholds: Vec<f32>,
    layer_scales: Vec<f32>,
}

impl ANNToSNNConverter {
    /// Create a new converter with default configuration
    pub fn new() -> Self {
        Self {
            config: ConversionConfig::default(),
            layer_thresholds: Vec::new(),
            layer_scales: Vec::new(),
        }
    }

    /// Create a converter with custom configuration
    pub fn with_config(config: ConversionConfig) -> Self {
        Self {
            config,
            layer_thresholds: Vec::new(),
            layer_scales: Vec::new(),
        }
    }

    /// Normalize weights for a layer
    pub fn normalize_weights(&self, weights: &Tensor, layer_idx: usize) -> Tensor {
        match self.config.weight_norm {
            WeightNormalizationMethod::DataBased => {
                self.data_based_normalization(weights, layer_idx)
            }
            WeightNormalizationMethod::ModelBased => {
                self.model_based_normalization(weights)
            }
            WeightNormalizationMethod::Hybrid => {
                // Combine both methods
                let data_norm = self.data_based_normalization(weights, layer_idx);
                self.model_based_normalization(&data_norm)
            }
            WeightNormalizationMethod::None => weights.clone(),
        }
    }

    /// Data-based normalization using max activation
    fn data_based_normalization(&self, weights: &Tensor, layer_idx: usize) -> Tensor {
        let scale = if layer_idx < self.layer_scales.len() {
            self.layer_scales[layer_idx]
        } else {
            1.0
        };

        if scale == 0.0 || scale == 1.0 {
            return weights.clone();
        }

        weights.scale(1.0 / scale)
    }

    /// Model-based normalization using weight statistics
    fn model_based_normalization(&self, weights: &Tensor) -> Tensor {
        // Find max absolute weight
        let max_weight = weights.data.iter()
            .map(|&w| w.abs())
            .fold(0.0f32, |a, b| a.max(b));

        if max_weight == 0.0 {
            return weights.clone();
        }

        // Normalize to [-1, 1]
        weights.scale(1.0 / max_weight)
    }

    /// Calibrate the converter using sample data
    pub fn calibrate(&mut self, layers: &[Tensor], sample_inputs: &[Tensor]) {
        self.layer_thresholds.clear();
        self.layer_scales.clear();

        // Compute layer-wise statistics
        for layer in layers.iter() {
            let mut max_activation = 0.0f32;

            // Process each sample
            for sample in sample_inputs {
                // Simplified: compute approximate activation
                let activation = self.compute_layer_activation(layer, sample);
                max_activation = max_activation.max(activation);
            }

            // Set scale and threshold
            self.layer_scales.push(max_activation.max(1.0));

            let threshold = match self.config.threshold_strategy {
                ThresholdBalancingStrategy::Fixed(t) => t,
                ThresholdBalancingStrategy::LayerWise => max_activation / 2.0,
                ThresholdBalancingStrategy::Percentile(p) => max_activation * p,
                ThresholdBalancingStrategy::Learned => 1.0, // Would be learned during conversion
            };

            self.layer_thresholds.push(threshold);
        }
    }

    /// Compute approximate layer activation (simplified)
    fn compute_layer_activation(&self, _layer: &Tensor, _input: &Tensor) -> f32 {
        // Simplified: return a fixed value
        // In practice, this would perform actual forward pass
        1.0
    }

    /// Get threshold for a specific layer
    pub fn get_threshold(&self, layer_idx: usize) -> f32 {
        if layer_idx < self.layer_thresholds.len() {
            self.layer_thresholds[layer_idx]
        } else {
            1.0 // Default threshold
        }
    }

    /// Fold batch normalization into preceding layer
    pub fn fold_batchnorm_into_conv(
        &self,
        conv_weight: &Tensor,
        conv_bias: Option<&Tensor>,
        bn_mean: &Tensor,
        bn_var: &Tensor,
        bn_gamma: &Tensor,
        bn_beta: &Tensor,
    ) -> (Tensor, Tensor) {
        let eps = 1e-5;

        // Compute scale factor: gamma / sqrt(var + eps)
        let scale_factors: Vec<f32> = (0..bn_gamma.data.len())
            .map(|i| bn_gamma.data[i] / (bn_var.data[i] + eps).sqrt())
            .collect();

        // Scale weights
        let mut new_weight_data = conv_weight.data.clone();
        let out_channels = conv_weight.shape[0];

        for oc in 0..out_channels {
            let scale = scale_factors[oc];
            // Scale all weights for this output channel
            let channel_size = conv_weight.data.len() / out_channels;
            for i in 0..channel_size {
                new_weight_data[oc * channel_size + i] *= scale;
            }
        }

        // Compute new bias: gamma * (bias - mean) / sqrt(var + eps) + beta
        let mut new_bias_data = vec![0.0; out_channels];
        for oc in 0..out_channels {
            let old_bias = if let Some(bias) = conv_bias {
                bias.data[oc]
            } else {
                0.0
            };

            new_bias_data[oc] = scale_factors[oc] * (old_bias - bn_mean.data[oc]) + bn_beta.data[oc];
        }

        (
            Tensor {
                data: new_weight_data,
                shape: conv_weight.shape.clone(),
            },
            Tensor {
                data: new_bias_data,
                shape: vec![out_channels],
            },
        )
    }

    /// Convert activation function to spike encoding
    pub fn convert_activation_to_spikes(&self, activation: &Tensor) -> Tensor {
        // Convert continuous activation to spike rate
        // Higher activation = higher spike rate
        let num_timesteps = self.config.num_timesteps as f32;

        let spike_data: Vec<f32> = activation.data.iter()
            .map(|&act| {
                // Clip to [0, 1] range
                let rate = act.clamp(0.0, 1.0);
                // Convert to spike count over timesteps
                (rate * num_timesteps).round() / num_timesteps
            })
            .collect();

        Tensor {
            data: spike_data,
            shape: activation.shape.clone(),
        }
    }

    /// Estimate accuracy loss from conversion
    pub fn estimate_accuracy_loss(&self, num_layers: usize) -> f32 {
        // Rough estimate based on number of layers and timesteps
        let timestep_factor = (self.config.num_timesteps as f32).log2() / 10.0;
        let layer_factor = (num_layers as f32) * 0.01;

        // Lower is better (less accuracy loss)
        (layer_factor - timestep_factor).max(0.0)
    }

    /// Generate conversion report
    pub fn conversion_report(&self) -> String {
        let mut report = String::from("ANN-to-SNN Conversion Report\n");
        report.push_str("================================\n\n");

        report.push_str("Configuration:\n");
        report.push_str(&format!("  Weight Normalization: {:?}\n", self.config.weight_norm));
        report.push_str(&format!("  Threshold Strategy: {:?}\n", self.config.threshold_strategy));
        report.push_str(&format!("  Time Steps: {}\n", self.config.num_timesteps));
        report.push_str(&format!("  Bias Correction: {}\n", self.config.bias_correction));
        report.push_str(&format!("  BatchNorm Folding: {}\n\n", self.config.fold_batchnorm));

        if !self.layer_thresholds.is_empty() {
            report.push_str("Layer Statistics:\n");
            for (i, (threshold, scale)) in self.layer_thresholds.iter()
                .zip(self.layer_scales.iter())
                .enumerate()
            {
                report.push_str(&format!("  Layer {}: threshold={:.4}, scale={:.4}\n",
                                        i, threshold, scale));
            }
        }

        report.push_str(&format!("\nEstimated Accuracy Loss: {:.2}%\n",
                                self.estimate_accuracy_loss(self.layer_thresholds.len()) * 100.0));

        report
    }
}

impl Default for ANNToSNNConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to convert a full ANN model to SNN
pub fn convert_model_to_snn(
    weights: Vec<Tensor>,
    biases: Vec<Tensor>,
    sample_data: &[Tensor],
    config: Option<ConversionConfig>,
) -> (Vec<Tensor>, Vec<Tensor>, ANNToSNNConverter) {
    let mut converter = if let Some(cfg) = config {
        ANNToSNNConverter::with_config(cfg)
    } else {
        ANNToSNNConverter::new()
    };

    // Calibrate converter
    converter.calibrate(&weights, sample_data);

    // Normalize all weights
    let normalized_weights: Vec<Tensor> = weights.iter()
        .enumerate()
        .map(|(i, w)| converter.normalize_weights(w, i))
        .collect();

    // Process biases if bias correction is enabled
    let processed_biases = if converter.config.bias_correction {
        biases.clone()
    } else {
        biases.iter()
            .map(|b| Tensor::zeros(b.shape.clone()))
            .collect()
    };

    (normalized_weights, processed_biases, converter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = ANNToSNNConverter::new();
        assert_eq!(converter.config.num_timesteps, 100);
    }

    #[test]
    fn test_custom_config() {
        let config = ConversionConfig {
            weight_norm: WeightNormalizationMethod::ModelBased,
            threshold_strategy: ThresholdBalancingStrategy::Fixed(0.5),
            num_timesteps: 200,
            bias_correction: false,
            fold_batchnorm: false,
            clip_negative_weights: true,
        };

        let converter = ANNToSNNConverter::with_config(config);
        assert_eq!(converter.config.num_timesteps, 200);
        assert!(!converter.config.bias_correction);
    }

    #[test]
    fn test_model_based_normalization() {
        let converter = ANNToSNNConverter::new();
        let weights = Tensor::from_vec(vec![0.5, 1.0, -2.0, 0.8], vec![2, 2]);

        let normalized = converter.model_based_normalization(&weights);
        assert_eq!(normalized.shape, vec![2, 2]);

        // All weights should be in [-1, 1]
        assert!(normalized.data.iter().all(|&w| w.abs() <= 1.0 + 1e-6));
    }

    #[test]
    fn test_fold_batchnorm() {
        let converter = ANNToSNNConverter::new();

        let conv_weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let conv_bias = Tensor::from_vec(vec![0.1, 0.2], vec![2]);
        let bn_mean = Tensor::from_vec(vec![0.5, 0.6], vec![2]);
        let bn_var = Tensor::from_vec(vec![1.0, 1.5], vec![2]);
        let bn_gamma = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
        let bn_beta = Tensor::from_vec(vec![0.0, 0.0], vec![2]);

        let (new_weight, new_bias) = converter.fold_batchnorm_into_conv(
            &conv_weight,
            Some(&conv_bias),
            &bn_mean,
            &bn_var,
            &bn_gamma,
            &bn_beta,
        );

        assert_eq!(new_weight.shape, vec![2, 2]);
        assert_eq!(new_bias.shape, vec![2]);
    }

    #[test]
    fn test_activation_to_spikes() {
        let converter = ANNToSNNConverter::new();
        let activation = Tensor::from_vec(vec![0.0, 0.5, 1.0, 1.5], vec![4]);

        let spikes = converter.convert_activation_to_spikes(&activation);
        assert_eq!(spikes.shape, vec![4]);

        // All spike rates should be in [0, 1]
        assert!(spikes.data.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_conversion_report() {
        let mut converter = ANNToSNNConverter::new();
        converter.layer_thresholds = vec![1.0, 1.5, 2.0];
        converter.layer_scales = vec![2.0, 3.0, 4.0];

        let report = converter.conversion_report();
        assert!(report.contains("ANN-to-SNN Conversion Report"));
        assert!(report.contains("Layer 0"));
        assert!(report.contains("Layer 1"));
        assert!(report.contains("Layer 2"));
    }

    #[test]
    fn test_convert_model() {
        let weights = vec![
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0], vec![2, 2]),
        ];

        let biases = vec![
            Tensor::from_vec(vec![0.1, 0.2], vec![2]),
            Tensor::from_vec(vec![0.3, 0.4], vec![2]),
        ];

        let sample_data = vec![
            Tensor::from_vec(vec![1.0, 0.5], vec![2]),
        ];

        let (norm_weights, norm_biases, converter) = convert_model_to_snn(
            weights,
            biases,
            &sample_data,
            None,
        );

        assert_eq!(norm_weights.len(), 2);
        assert_eq!(norm_biases.len(), 2);
        assert_eq!(converter.layer_thresholds.len(), 2);
    }
}
