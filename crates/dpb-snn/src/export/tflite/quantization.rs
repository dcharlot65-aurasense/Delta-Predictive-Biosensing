//! TensorFlow Lite quantization support
//!
//! Implements post-training quantization (PTQ) and quantization-aware training (QAT)
//! support for TFLite model export.

use super::tensors::{QuantizationParams, TensorType};
use serde::{Deserialize, Serialize};

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
                    || !self.activation_dtype.supports_quantization()) =>
            {
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

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
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

    /// Name a single unnamed calibration tensor is recorded under.
    ///
    /// [`Self::calibrate`] takes samples with no layer names attached, so the
    /// range it observes is filed here and [`Self::quantize_activations`]
    /// accepts this name.
    pub const DEFAULT_LAYER: &'static str = "default";

    /// Calibrate from a representative dataset.
    ///
    /// The samples carry no layer identity, so the observed range is recorded
    /// under [`Self::DEFAULT_LAYER`]. Use [`Self::calibrate_layer`] to record
    /// ranges per layer, which is what `quantize_activations` looks up.
    ///
    /// This used to walk the dataset calling a no-op `update`, so nothing was
    /// ever recorded: calibration reported success and `quantize_activations`
    /// then failed for every layer with "No calibration data".
    pub fn calibrate(&mut self, dataset: &[Vec<f32>]) -> Result<(), String> {
        self.calibrate_layer(Self::DEFAULT_LAYER, dataset)
    }

    /// Calibrate one named layer from its representative activations.
    pub fn calibrate_layer(
        &mut self,
        layer_name: &str,
        dataset: &[Vec<f32>],
    ) -> Result<(), String> {
        if dataset.is_empty() {
            return Err("Calibration dataset cannot be empty".to_string());
        }

        let mut stats = self
            .calibration_data
            .take()
            .unwrap_or_else(CalibrationData::new);
        for sample in dataset {
            stats.update(layer_name, sample);
        }
        stats.finalize();

        if stats.get_range(layer_name).is_none() {
            self.calibration_data = Some(stats);
            return Err(format!(
                "Calibration data for '{layer_name}' contained no finite values"
            ));
        }

        self.calibration_data = Some(stats);
        Ok(())
    }

    /// Layers that currently have calibration data.
    pub fn calibrated_layers(&self) -> Vec<String> {
        self.calibration_data
            .as_ref()
            .map(CalibrationData::layers)
            .unwrap_or_default()
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
    pub fn quantize_activations(&self, layer_name: &str) -> Result<QuantizationParams, String> {
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
    fn quantize_per_tensor(
        &self,
        weights: &[f32],
    ) -> Result<(Vec<i8>, QuantizationParams), String> {
        // Find min/max
        let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let params = Self::compute_quantization_params(min, max, TensorType::Int8)?;

        // Quantize values
        let quantized: Vec<i8> = weights
            .iter()
            .map(|&w| {
                ((w / params.scales[0]).round() as i32 + params.zero_points[0]).clamp(-128, 127)
                    as i8
            })
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
            let min = channel_weights
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let max = channel_weights
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

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

        let params = QuantizationParams::per_channel(scales, zero_points, 0).with_min_max(
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
            return Err(format!(
                "Min ({}) cannot be greater than max ({})",
                min, max
            ));
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

    /// Folds one sample into the running range for `layer`.
    ///
    /// This used to take `_sample` and do nothing, so `layer_stats` stayed
    /// empty however much calibration data was supplied, and every later
    /// `get_range` returned None.
    fn update(&mut self, layer: &str, sample: &[f32]) {
        // Non-finite values would poison the range and make the derived scale
        // useless for every other value in the tensor.
        let finite = sample.iter().copied().filter(|v| v.is_finite());
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for v in finite {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if lo > hi {
            return; // nothing usable in this sample
        }

        let entry = self
            .layer_stats
            .entry(layer.to_string())
            .or_insert((f32::INFINITY, f32::NEG_INFINITY));
        entry.0 = entry.0.min(lo);
        entry.1 = entry.1.max(hi);
    }

    /// Widens any degenerate range so a scale can be derived from it.
    ///
    /// A layer whose activations were constant gives min == max, and a zero
    /// range yields a zero scale and a division by zero downstream.
    fn finalize(&mut self) {
        for (min, max) in self.layer_stats.values_mut() {
            if (*max - *min).abs() < f32::EPSILON {
                *min -= 0.5;
                *max += 0.5;
            }
            // The int8 zero point can only represent zero if the range spans
            // it, and activation ranges are expected to include zero.
            *min = min.min(0.0);
            *max = max.max(0.0);
        }
    }

    /// Layers that have calibration data.
    fn layers(&self) -> Vec<String> {
        let mut names: Vec<String> = self.layer_stats.keys().cloned().collect();
        names.sort();
        names
    }

    fn get_range(&self, layer_name: &str) -> Option<(f32, f32)> {
        self.layer_stats.get(layer_name).copied()
    }
}

/// Quantization-aware training support
// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
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

    /// Identity fake-quantization parameters: scale 1, zero point 0.
    ///
    /// Kept for callers that want the unscaled reference, but note what it
    /// means: with an int8 grid of unit spacing, every value smaller than 0.5
    /// rounds to zero. Typical weights are far smaller than that, so applying
    /// these to a real tensor erases it. [`Self::fake_quantize`] derives a
    /// scale from the data instead; use [`Self::fake_quant_params_for`] to see
    /// the parameters it would use.
    pub fn get_fake_quant_params(&self) -> QuantizationParams {
        QuantizationParams::per_tensor(1.0, 0)
    }

    /// Quantization parameters covering `values`, as a QAT step would derive
    /// them from the batch it is looking at.
    pub fn fake_quant_params_for(&self, values: &[f32]) -> Result<QuantizationParams, String> {
        if values.is_empty() {
            return Err("Cannot derive quantization parameters from no values".to_string());
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in values.iter().filter(|v| v.is_finite()) {
            min = min.min(v);
            max = max.max(v);
        }
        if min > max {
            return Err("Values contain nothing finite".to_string());
        }
        // The grid must span zero, and must not be degenerate.
        min = min.min(0.0);
        max = max.max(0.0);
        if (max - min).abs() < f32::EPSILON {
            min -= 0.5;
            max += 0.5;
        }
        PostTrainingQuantizer::compute_quantization_params(min, max, TensorType::Int8)
    }

    /// Simulate quantization during the forward pass.
    ///
    /// The scale comes from the values themselves. It used to come from
    /// `get_fake_quant_params`, which is scale 1.0: on an int8 grid of unit
    /// spacing every weight below 0.5 rounds to zero, so fake-quantizing a
    /// normally initialised layer returned all zeros and QAT trained against a
    /// network that had been erased.
    pub fn fake_quantize(&self, values: &[f32]) -> Vec<f32> {
        let Ok(params) = self.fake_quant_params_for(values) else {
            return values.to_vec();
        };

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
        let params =
            PostTrainingQuantizer::compute_quantization_params(-1.0, 1.0, TensorType::Int8);
        assert!(params.is_ok());

        let params = params.unwrap();
        assert_eq!(params.scales.len(), 1);
        assert_eq!(params.zero_points.len(), 1);
        assert!(params.scales[0] > 0.0);
    }

    #[test]
    fn test_compute_quantization_params_asymmetric() {
        let params = PostTrainingQuantizer::compute_quantization_params(0.0, 2.0, TensorType::Int8);
        assert!(params.is_ok());

        let params = params.unwrap();
        // Zero should be exactly representable
        assert!(params.scales[0] > 0.0);
    }

    #[test]
    fn test_compute_quantization_params_invalid() {
        // NaN values
        let result =
            PostTrainingQuantizer::compute_quantization_params(f32::NAN, 1.0, TensorType::Int8);
        assert!(result.is_err());

        // Min > max
        let result =
            PostTrainingQuantizer::compute_quantization_params(1.0, -1.0, TensorType::Int8);
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

    // ---- calibration and QAT -------------------------------------------

    fn qat_config() -> QuantizationConfig {
        QuantizationConfig {
            mode: QuantizationMode::QAT,
            ..QuantizationConfig::default()
        }
    }

    /// Calibration must record a range, and activation quantization must then
    /// succeed. `update` used to ignore its sample entirely, so calibration
    /// reported success while recording nothing and every later lookup failed.
    #[test]
    fn calibration_records_a_range() {
        let mut q = PostTrainingQuantizer::new(QuantizationConfig::default()).unwrap();
        let data = vec![vec![-2.0, 0.5, 1.5], vec![0.0, 3.0, -1.0]];

        assert!(
            q.quantize_activations(PostTrainingQuantizer::DEFAULT_LAYER)
                .is_err(),
            "there is no calibration data yet"
        );

        q.calibrate(&data).unwrap();
        assert_eq!(
            q.calibrated_layers(),
            vec![PostTrainingQuantizer::DEFAULT_LAYER.to_string()]
        );

        let params = q
            .quantize_activations(PostTrainingQuantizer::DEFAULT_LAYER)
            .expect("activation quantization should succeed after calibration");
        assert!(
            params.scales[0] > 0.0,
            "a zero scale cannot represent anything"
        );
    }

    /// Ranges accumulate across samples and across calls, per layer.
    #[test]
    fn calibration_tracks_each_layer_separately() {
        let mut q = PostTrainingQuantizer::new(QuantizationConfig::default()).unwrap();
        q.calibrate_layer("narrow", &[vec![-0.1, 0.1]]).unwrap();
        q.calibrate_layer("wide", &[vec![-8.0, 8.0]]).unwrap();

        assert_eq!(q.calibrated_layers(), vec!["narrow", "wide"]);
        let narrow = q.quantize_activations("narrow").unwrap().scales[0];
        let wide = q.quantize_activations("wide").unwrap().scales[0];
        assert!(
            wide > narrow,
            "a wider activation range needs a coarser scale: {wide} vs {narrow}"
        );
        assert!(q.quantize_activations("absent").is_err());
    }

    /// A constant layer must not produce a zero scale.
    #[test]
    fn calibration_widens_a_degenerate_range() {
        let mut q = PostTrainingQuantizer::new(QuantizationConfig::default()).unwrap();
        q.calibrate(&[vec![2.0, 2.0, 2.0]]).unwrap();
        let params = q
            .quantize_activations(PostTrainingQuantizer::DEFAULT_LAYER)
            .unwrap();
        assert!(params.scales[0] > 0.0);
    }

    /// Fake quantization must preserve the tensor, not erase it.
    ///
    /// It used a scale of 1.0, so on an int8 grid of unit spacing every value
    /// below 0.5 rounded to zero -- which is every weight in a normally
    /// initialised layer.
    #[test]
    fn fake_quantization_preserves_typical_weights() {
        let qat = QuantizationAwareTraining::new(qat_config()).unwrap();
        let weights: Vec<f32> = (0..64)
            .map(|i| (i as f32 - 32.0) * 0.01) // -0.32 .. 0.31
            .collect();

        let out = qat.fake_quantize(&weights);
        assert_eq!(out.len(), weights.len());

        let magnitude: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(
            magnitude > 0.0,
            "fake quantization returned an all-zero tensor: the whole layer was erased"
        );

        // Round-trip error must be a fraction of the range, not the whole of it.
        let worst = weights
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let span = 0.64f32;
        assert!(
            worst < span / 100.0,
            "int8 round-trip error {worst} is too large for a range of {span}"
        );
    }

    /// The identity parameters are still available, and still destroy small
    /// weights -- which is why `fake_quantize` no longer uses them.
    #[test]
    fn identity_params_are_documented_as_destructive() {
        let qat = QuantizationAwareTraining::new(qat_config()).unwrap();
        let params = qat.get_fake_quant_params();
        assert_eq!(params.scales[0], 1.0);

        let small = 0.05f32;
        let q = params.quantize_value(small, 0);
        assert_eq!(
            params.dequantize_value(q, 0),
            0.0,
            "unit scale should round a typical weight to zero"
        );
    }
}
