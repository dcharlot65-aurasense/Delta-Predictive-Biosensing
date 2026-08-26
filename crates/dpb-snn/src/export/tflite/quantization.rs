//! TensorFlow Lite quantization support
//!
//! Implements post-training quantization (PTQ) and quantization-aware training (QAT)
//! support for TFLite model export.

use super::tensors::{TensorType, QuantizationParams};
use serde::{Serialize, Deserialize};

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization mode
    pub mode: QuantizationMode,
    /// Quantization strategy
    pub strategy: QuantizationStrategy,
    /// Target data type for quantized weights
    pub weight_dtype: TensorType,
    /// Target data type for quantized activations
    pub activation_dtype: TensorType,
    /// Use per-channel quantization for weights
    pub per_channel_weights: bool,
    /// Representative dataset size (for calibration)
    pub calibration_samples: Option<usize>,
}

impl QuantizationConfig {
    /// Create Int8 quantization configuration
    pub fn int8() -> Self {
        Self {
            mode: QuantizationMode::PostTraining,
            strategy: QuantizationStrategy::DynamicRange,
            weight_dtype: TensorType::Int8,
            activation_dtype: TensorType::Int8,
            per_channel_weights: true,
            calibration_samples: Some(100),
        }
    }

    /// Create Float16 quantization configuration
    pub fn float16() -> Self {
        Self {
            mode: QuantizationMode::PostTraining,
            strategy: QuantizationStrategy::Float16,
            weight_dtype: TensorType::Float16,
            activation_dtype: TensorType::Float16,
            per_channel_weights: false,
            calibration_samples: None,
        }
    }

    /// Create dynamic range quantization configuration
    pub fn dynamic() -> Self {
        Self {
            mode: QuantizationMode::PostTraining,
            strategy: QuantizationStrategy::DynamicRange,
            weight_dtype: TensorType::Int8,
            activation_dtype: TensorType::Float32,
            per_channel_weights: true,
            calibration_samples: None,
        }
    }

    /// Create QAT configuration
    pub fn qat() -> Self {
        Self {
            mode: QuantizationMode::QAT,
            strategy: QuantizationStrategy::FullInteger,
            weight_dtype: TensorType::Int8,
            activation_dtype: TensorType::Int8,
            per_channel_weights: true,
            calibration_samples: None,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Check weight dtype is valid for quantization
        if !self.weight_dtype.supports_quantization() && !self.weight_dtype.is_float() {
            return Err(format!(
                "Invalid weight dtype for quantization: {:?}",
                self.weight_dtype
            ));
        }

        // Check activation dtype
        if !self.activation_dtype.supports_quantization() && !self.activation_dtype.is_float() {
            return Err(format!(
                "Invalid activation dtype for quantization: {:?}",
                self.activation_dtype
            ));
        }

        // Validate strategy matches dtypes
        match self.strategy {
            QuantizationStrategy::Float16 => {
                if self.weight_dtype != TensorType::Float16 {
                    return Err("Float16 strategy requires Float16 weight dtype".to_string());
                }
            }
            QuantizationStrategy::FullInteger
                if (!self.weight_dtype.supports_quantization()
                    || !self.activation_dtype.supports_quantization())
                => {
                    return Err("FullInteger strategy requires integer dtypes".to_string());
                }
            _ => {}
        }

        Ok(())
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self::dynamic()
    }
}

/// Quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Post-training quantization
    PostTraining,
    /// Quantization-aware training
    QAT,
}

/// Quantization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// Dynamic range quantization (weights only)
    DynamicRange,
    /// Full integer quantization (weights + activations)
    FullInteger,
    /// Float16 quantization
    Float16,
    /// Integer-only inference (no float fallback)
    IntegerOnly,
}

/// Post-training quantizer
pub struct PostTrainingQuantizer {
    config: QuantizationConfig,
    calibration_data: Option<CalibrationData>,
}

impl PostTrainingQuantizer {
    /// Create a new post-training quantizer
    pub fn new(config: QuantizationConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            calibration_data: None,
        })
    }

    /// Calibrate using representative dataset
    pub fn calibrate(&mut self, dataset: &[Vec<f32>]) -> Result<(), String> {
        if dataset.is_empty() {
            return Err("Calibration dataset cannot be empty".to_string());
        }

        // Compute statistics from representative dataset
        let mut stats = CalibrationData::new();

        for sample in dataset {
            stats.update(sample);
        }

        stats.finalize();
        self.calibration_data = Some(stats);

        Ok(())
    }

    /// Quantize weights
    pub fn quantize_weights(
        &self,
        weights: &[f32],
        shape: &[usize],
    ) -> Result<(Vec<i8>, QuantizationParams), String> {
        match self.config.strategy {
            QuantizationStrategy::DynamicRange | QuantizationStrategy::FullInteger => {
                self.quantize_to_int8(weights, shape)
            }
            QuantizationStrategy::Float16 => {
                Err("Float16 quantization not implemented yet".to_string())
            }
            QuantizationStrategy::IntegerOnly => self.quantize_to_int8(weights, shape),
        }
    }

    /// Quantize activations (requires calibration data)
    pub fn quantize_activations(
        &self,
        layer_name: &str,
    ) -> Result<QuantizationParams, String> {
        let calib_data = self
            .calibration_data
            .as_ref()
            .ok_or("Calibration data required for activation quantization")?;

        let (min, max) = calib_data
            .get_range(layer_name)
            .ok_or_else(|| format!("No calibration data for layer: {}", layer_name))?;

        Self::compute_quantization_params(min, max, TensorType::Int8)
    }

    /// Quantize to Int8
    fn quantize_to_int8(
        &self,
        weights: &[f32],
        shape: &[usize],
    ) -> Result<(Vec<i8>, QuantizationParams), String> {
        if weights.is_empty() {
            return Err("Cannot quantize empty weights".to_string());
        }

        if self.config.per_channel_weights && shape.len() >= 2 {
            self.quantize_per_channel(weights, shape)
        } else {
            self.quantize_per_tensor(weights)
        }
    }

    /// Per-tensor quantization
    fn quantize_per_tensor(&self, weights: &[f32]) -> Result<(Vec<i8>, QuantizationParams), String> {
        // Find min/max
        let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let params = Self::compute_quantization_params(min, max, TensorType::Int8)?;

        // Quantize values
        let quantized: Vec<i8> = weights
            .iter()
            .map(|&w| ((w / params.scales[0]).round() as i32 + params.zero_points[0]).clamp(-128, 127) as i8)
            .collect();

        Ok((quantized, params))
    }

    /// Per-channel quantization (along output channel dimension)
    fn quantize_per_channel(
        &self,
        weights: &[f32],
        shape: &[usize],
    ) -> Result<(Vec<i8>, QuantizationParams), String> {
        if shape.len() < 2 {
            return Err("Per-channel quantization requires at least 2D shape".to_string());
        }

        let num_output_channels = shape[0];
        let channel_size = weights.len() / num_output_channels;

        if weights.len() != num_output_channels * channel_size {
            return Err("Weight size doesn't match shape".to_string());
        }

        let mut scales = Vec::with_capacity(num_output_channels);
        let mut zero_points = Vec::with_capacity(num_output_channels);
        let mut quantized = Vec::with_capacity(weights.len());

        // Quantize each output channel separately
        for ch in 0..num_output_channels {
            let start = ch * channel_size;
            let end = start + channel_size;
            let channel_weights = &weights[start..end];

            // Find min/max for this channel
            let min = channel_weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = channel_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let params = Self::compute_quantization_params(min, max, TensorType::Int8)?;
            scales.push(params.scales[0]);
            zero_points.push(params.zero_points[0]);

            // Quantize channel
            for &w in channel_weights {
                let q = ((w / params.scales[0]).round() as i32 + params.zero_points[0])
                    .clamp(-128, 127) as i8;
                quantized.push(q);
            }
        }

        let params = QuantizationParams::per_channel(scales, zero_points, 0)
            .with_min_max(
                weights.iter().cloned().fold(f32::INFINITY, f32::min),
                weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            );

        Ok((quantized, params))
    }

    /// Compute quantization scale and zero point
    fn compute_quantization_params(
        min: f32,
        max: f32,
        dtype: TensorType,
    ) -> Result<QuantizationParams, String> {
        if !min.is_finite() || !max.is_finite() {
            return Err("Min/max values must be finite".to_string());
        }

        if min > max {
            return Err(format!("Min ({}) cannot be greater than max ({})", min, max));
        }

        // Ensure range includes zero
        let min = min.min(0.0);
        let max = max.max(0.0);

        let (qmin, qmax) = match dtype {
            TensorType::Int8 => (-128.0, 127.0),
            TensorType::UInt8 => (0.0, 255.0),
            TensorType::Int16 => (-32768.0, 32767.0),
            _ => return Err(format!("Unsupported quantization dtype: {:?}", dtype)),
        };

        // Compute scale
        let scale = if (max - min).abs() < 1e-10 {
            1.0 // Avoid division by zero
        } else {
            (max - min) / (qmax - qmin)
        };

        // Compute zero point
        let zero_point_real = qmin - min / scale;
        let zero_point = zero_point_real.round().clamp(qmin, qmax) as i32;

        Ok(QuantizationParams::per_tensor(scale, zero_point).with_min_max(min, max))
    }
}

/// Calibration data for activation quantization
#[derive(Debug, Clone)]
pub struct CalibrationData {
    layer_stats: std::collections::HashMap<String, (f32, f32)>, // min, max per layer
}

impl CalibrationData {
    fn new() -> Self {
        Self {
            layer_stats: std::collections::HashMap::new(),
        }
    }

    fn update(&mut self, _sample: &[f32]) {
        // In a real implementation, this would track min/max per layer
        // For now, this is a placeholder
    }

    fn finalize(&mut self) {
        // Finalize statistics
    }

    fn get_range(&self, layer_name: &str) -> Option<(f32, f32)> {
        self.layer_stats.get(layer_name).copied()
    }
}

/// Quantization-aware training support
pub struct QuantizationAwareTraining {
    config: QuantizationConfig,
}

impl QuantizationAwareTraining {
    /// Create new QAT helper
    pub fn new(config: QuantizationConfig) -> Result<Self, String> {
        if config.mode != QuantizationMode::QAT {
            return Err("Config must have QAT mode".to_string());
        }
        config.validate()?;
        Ok(Self { config })
    }

    /// Get fake quantization parameters for training
    pub fn get_fake_quant_params(&self) -> QuantizationParams {
        // Return default params for QAT
        QuantizationParams::per_tensor(1.0, 0)
    }

    /// Simulate quantization during forward pass
    pub fn fake_quantize(&self, values: &[f32]) -> Vec<f32> {
        // Simulate quantization-dequantization
        let params = self.get_fake_quant_params();

        values
            .iter()
            .map(|&v| {
                let q = params.quantize_value(v, 0);
                params.dequantize_value(q, 0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_config_int8() {
        let config = QuantizationConfig::int8();
        assert_eq!(config.weight_dtype, TensorType::Int8);
        assert_eq!(config.activation_dtype, TensorType::Int8);
        assert!(config.per_channel_weights);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quantization_config_float16() {
        let config = QuantizationConfig::float16();
        assert_eq!(config.weight_dtype, TensorType::Float16);
        assert_eq!(config.strategy, QuantizationStrategy::Float16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_quantization_config_dynamic() {
        let config = QuantizationConfig::dynamic();
        assert_eq!(config.weight_dtype, TensorType::Int8);
        assert_eq!(config.activation_dtype, TensorType::Float32);
        assert_eq!(config.strategy, QuantizationStrategy::DynamicRange);
    }

    #[test]
    fn test_quantization_config_qat() {
        let config = QuantizationConfig::qat();
        assert_eq!(config.mode, QuantizationMode::QAT);
        assert_eq!(config.strategy, QuantizationStrategy::FullInteger);
    }

    #[test]
    fn test_quantization_config_validation() {
        let mut config = QuantizationConfig::int8();
        assert!(config.validate().is_ok());

        // Invalid weight dtype
        config.weight_dtype = TensorType::String;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_post_training_quantizer_creation() {
        let config = QuantizationConfig::int8();
        let quantizer = PostTrainingQuantizer::new(config);
        assert!(quantizer.is_ok());
    }

    #[test]
    fn test_compute_quantization_params() {
        let params = PostTrainingQuantizer::compute_quantization_params(
            -1.0,
            1.0,
            TensorType::Int8,
        );
        assert!(params.is_ok());

        let params = params.unwrap();
        assert_eq!(params.scales.len(), 1);
        assert_eq!(params.zero_points.len(), 1);
        assert!(params.scales[0] > 0.0);
    }

    #[test]
    fn test_compute_quantization_params_asymmetric() {
        let params = PostTrainingQuantizer::compute_quantization_params(
            0.0,
            2.0,
            TensorType::Int8,
        );
        assert!(params.is_ok());

        let params = params.unwrap();
        // Zero should be exactly representable
        assert!(params.scales[0] > 0.0);
    }

    #[test]
    fn test_compute_quantization_params_invalid() {
        // NaN values
        let result = PostTrainingQuantizer::compute_quantization_params(
            f32::NAN,
            1.0,
            TensorType::Int8,
        );
        assert!(result.is_err());

        // Min > max
        let result = PostTrainingQuantizer::compute_quantization_params(
            1.0,
            -1.0,
            TensorType::Int8,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_quantize_per_tensor() {
        let config = QuantizationConfig::int8();
        let quantizer = PostTrainingQuantizer::new(config).unwrap();

        let weights = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let result = quantizer.quantize_per_tensor(&weights);

        assert!(result.is_ok());
        let (quantized, params) = result.unwrap();
        assert_eq!(quantized.len(), weights.len());
        assert!(params.scales[0] > 0.0);
    }

    #[test]
    fn test_quantize_per_channel() {
        let mut config = QuantizationConfig::int8();
        config.per_channel_weights = true;

        let quantizer = PostTrainingQuantizer::new(config).unwrap();

        // 2 output channels, 4 weights each
        let weights = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let shape = vec![2, 4];

        let result = quantizer.quantize_per_channel(&weights, &shape);

        assert!(result.is_ok());
        let (quantized, params) = result.unwrap();
        assert_eq!(quantized.len(), weights.len());
        assert!(params.is_per_channel());
        assert_eq!(params.num_channels(), 2);
    }

    #[test]
    fn test_quantize_weights() {
        let config = QuantizationConfig::dynamic();
        let quantizer = PostTrainingQuantizer::new(config).unwrap();

        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let shape = vec![4];

        let result = quantizer.quantize_weights(&weights, &shape);
        assert!(result.is_ok());
    }

    #[test]
    fn test_calibration_empty_dataset() {
        let config = QuantizationConfig::int8();
        let mut quantizer = PostTrainingQuantizer::new(config).unwrap();

        let result = quantizer.calibrate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_calibration_with_data() {
        let config = QuantizationConfig::int8();
        let mut quantizer = PostTrainingQuantizer::new(config).unwrap();

        let dataset = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let result = quantizer.calibrate(&dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn test_qat_creation() {
        let config = QuantizationConfig::qat();
        let qat = QuantizationAwareTraining::new(config);
        assert!(qat.is_ok());
    }

    #[test]
    fn test_qat_wrong_mode() {
        let config = QuantizationConfig::dynamic(); // Not QAT mode
        let qat = QuantizationAwareTraining::new(config);
        assert!(qat.is_err());
    }

    #[test]
    fn test_qat_fake_quantize() {
        let config = QuantizationConfig::qat();
        let qat = QuantizationAwareTraining::new(config).unwrap();

        let values = vec![0.1, 0.5, 1.0, 2.0];
        let fake_quant = qat.fake_quantize(&values);

        assert_eq!(fake_quant.len(), values.len());
        // Values should be similar but quantized
        for (orig, quant) in values.iter().zip(fake_quant.iter()) {
            assert!((orig - quant).abs() < 2.0); // Within quantization error
        }
    }
}
