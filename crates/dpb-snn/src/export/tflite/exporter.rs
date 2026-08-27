//! Main TensorFlow Lite exporter
//!
//! Exports SNN models to TensorFlow Lite format with quantization support.

use super::super::config::{ExportMetadata, LayerConfig};
use super::super::weights::ModelWeights;
use super::flatbuffer::{FlatBufferBuilder, SubgraphBuilder};
use super::metadata::{ContentType, TFLiteMetadata, TensorMetadata};
use super::operators::{
    BuiltinOperator, OperatorOptions, OperatorRegistry, OperatorType, TFLiteOperator,
};
use super::quantization::{PostTrainingQuantizer, QuantizationConfig};
use super::tensors::{TFLiteTensor, TensorShape, TensorType};
use super::validation::{TFLiteValidator, ValidationResult};

use serde::{Deserialize, Serialize};

/// TensorFlow Lite export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFLiteConfig {
    /// Model name
    pub model_name: String,
    /// Quantization configuration (if any)
    pub quantization: Option<QuantizationConfig>,
    /// Input tensor names
    pub input_names: Vec<String>,
    /// Output tensor names
    pub output_names: Vec<String>,
    /// Include metadata
    pub include_metadata: bool,
    /// Validate before export
    pub validate: bool,
    /// Operator registry (for custom mappings)
    operator_registry: OperatorRegistry,
}

impl TFLiteConfig {
    /// Create a new TFLite configuration
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            quantization: None,
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            include_metadata: true,
            validate: true,
            operator_registry: OperatorRegistry::new(),
        }
    }

    /// Set quantization configuration
    pub fn with_quantization(mut self, config: QuantizationConfig) -> Self {
        self.quantization = Some(config);
        self
    }

    /// Set input tensor names
    pub fn with_inputs(mut self, names: Vec<String>) -> Self {
        self.input_names = names;
        self
    }

    /// Set output tensor names
    pub fn with_outputs(mut self, names: Vec<String>) -> Self {
        self.output_names = names;
        self
    }

    /// Disable metadata inclusion
    pub fn without_metadata(mut self) -> Self {
        self.include_metadata = false;
        self
    }

    /// Disable validation
    pub fn without_validation(mut self) -> Self {
        self.validate = false;
        self
    }
}

impl Default for TFLiteConfig {
    fn default() -> Self {
        Self::new("model".to_string())
    }
}

/// TFLite export result
#[derive(Debug)]
pub struct TFLiteExportResult {
    /// Exported model bytes (.tflite format)
    pub model_bytes: Vec<u8>,
    /// Export metadata
    pub metadata: ExportMetadata,
    /// Validation result (if validation was enabled)
    pub validation: Option<ValidationResult>,
    /// Export warnings
    pub warnings: Vec<String>,
}

impl TFLiteExportResult {
    /// Save model to file
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, &self.model_bytes).map_err(|e| format!("Failed to write file: {}", e))
    }

    /// Get model size in bytes
    pub fn size_bytes(&self) -> usize {
        self.model_bytes.len()
    }

    /// Get model size in megabytes
    pub fn size_mb(&self) -> f64 {
        self.size_bytes() as f64 / 1_000_000.0
    }
}

/// TensorFlow Lite exporter
pub struct TFLiteExporter {
    config: TFLiteConfig,
}

impl TFLiteExporter {
    /// Create a new TFLite exporter
    pub fn new(config: TFLiteConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self {
            config: TFLiteConfig::default(),
        }
    }

    /// Export SNN model to TFLite format
    pub fn export_model(
        &self,
        weights: &ModelWeights,
        layer_configs: &[LayerConfig],
        input_shape: &[usize],
    ) -> Result<TFLiteExportResult, String> {
        let mut warnings = Vec::new();

        // Initialize quantizer if needed
        let quantizer = if let Some(ref quant_config) = self.config.quantization {
            Some(PostTrainingQuantizer::new(quant_config.clone())?)
        } else {
            None
        };

        // Create FlatBuffer builder
        let mut fb_builder = FlatBufferBuilder::new()
            .with_description(format!("SNN model: {}", self.config.model_name));

        // Create subgraph
        let mut subgraph_builder = SubgraphBuilder::new("main".to_string());

        // Add input tensor
        let input_tensor = TFLiteTensor::new(
            self.config
                .input_names
                .first()
                .unwrap_or(&"input".to_string())
                .clone(),
            TensorShape::from_usize(input_shape.to_vec()),
            TensorType::Float32,
        );
        subgraph_builder.add_tensor(input_tensor);

        // Convert layers to TFLite operators
        let mut prev_output = self
            .config
            .input_names
            .first()
            .unwrap_or(&"input".to_string())
            .clone();

        for (i, (layer_config, layer_weights)) in
            layer_configs.iter().zip(weights.layers.iter()).enumerate()
        {
            let layer_name = format!("layer_{}", i);

            // Get operator type from registry
            let op_type = self
                .config
                .operator_registry
                .get_operator(&layer_config.layer_type.to_string())
                .cloned()
                .unwrap_or(OperatorType::Builtin(BuiltinOperator::FullyConnected));

            // Add weight buffer
            let _weight_buffer_idx = if let Some(ref quantizer) = quantizer {
                // Quantize weights
                let (quantized_weights, quant_params) = quantizer.quantize_weights(
                    &layer_weights
                        .weights
                        .iter()
                        .map(|&w| w as f32)
                        .collect::<Vec<_>>(),
                    &layer_weights.shape,
                )?;

                // Add quantized weights to buffer
                let buffer_idx = fb_builder
                    .buffer_manager()
                    .add_i8_buffer(&quantized_weights);

                // Create weight tensor with quantization
                let weight_tensor = TFLiteTensor::new(
                    format!("{}_weight", layer_name),
                    TensorShape::from_usize(layer_weights.shape.clone()),
                    TensorType::Int8,
                )
                .with_quantization(quant_params)
                .with_buffer(buffer_idx);

                subgraph_builder.add_tensor(weight_tensor);
                buffer_idx
            } else {
                // Add float weights to buffer
                let float_weights: Vec<f32> =
                    layer_weights.weights.iter().map(|&w| w as f32).collect();
                let buffer_idx = fb_builder.buffer_manager().add_f32_buffer(&float_weights);

                // Create weight tensor
                let weight_tensor = TFLiteTensor::new(
                    format!("{}_weight", layer_name),
                    TensorShape::from_usize(layer_weights.shape.clone()),
                    TensorType::Float32,
                )
                .with_buffer(buffer_idx);

                subgraph_builder.add_tensor(weight_tensor);
                buffer_idx
            };

            // Add bias if present
            if let Some(ref bias) = layer_weights.bias {
                let bias_buffer_idx = if quantizer.is_some() {
                    // For quantized models, bias is typically Int32
                    let bias_i32: Vec<i32> = bias.iter().map(|&b| (b * 1000.0) as i32).collect();
                    let bytes: Vec<u8> = bias_i32.iter().flat_map(|&i| i.to_le_bytes()).collect();
                    fb_builder.buffer_manager().add_buffer(bytes)
                } else {
                    let bias_f32: Vec<f32> = bias.iter().map(|&b| b as f32).collect();
                    fb_builder.buffer_manager().add_f32_buffer(&bias_f32)
                };

                let bias_tensor = TFLiteTensor::new(
                    format!("{}_bias", layer_name),
                    TensorShape::from_usize(vec![bias.len()]),
                    if quantizer.is_some() {
                        TensorType::Int32
                    } else {
                        TensorType::Float32
                    },
                )
                .with_buffer(bias_buffer_idx);

                subgraph_builder.add_tensor(bias_tensor);
            }

            // Create output tensor
            let output_name = if i == layer_configs.len() - 1 {
                self.config
                    .output_names
                    .first()
                    .unwrap_or(&"output".to_string())
                    .clone()
            } else {
                format!("{}_output", layer_name)
            };

            let output_tensor = TFLiteTensor::new(
                output_name.clone(),
                TensorShape::from_usize(vec![1, layer_config.output_size]), // Simplified shape
                TensorType::Float32,
            );
            subgraph_builder.add_tensor(output_tensor);

            // Create operator
            let input_idx = subgraph_builder.get_tensor_index(&prev_output).unwrap();
            let weight_idx = subgraph_builder
                .get_tensor_index(&format!("{}_weight", layer_name))
                .unwrap();
            let output_idx = subgraph_builder.get_tensor_index(&output_name).unwrap();

            let mut operator_inputs = vec![input_idx, weight_idx];
            if layer_weights.bias.is_some() {
                let bias_idx = subgraph_builder
                    .get_tensor_index(&format!("{}_bias", layer_name))
                    .unwrap();
                operator_inputs.push(bias_idx);
            }

            let operator = TFLiteOperator::new(op_type.clone(), operator_inputs, vec![output_idx])
                .with_options(OperatorOptions::None);

            subgraph_builder.add_operator(operator);

            // Warn about spiking layers
            if layer_config.layer_type.to_string().contains("Spiking") {
                warnings.push(format!(
                    "Layer {} is a spiking layer - temporal dynamics are approximated in TFLite",
                    layer_name
                ));
            }

            prev_output = output_name;
        }

        // Set subgraph inputs/outputs
        subgraph_builder.set_inputs_by_name(&self.config.input_names)?;
        subgraph_builder.set_outputs_by_name(&self.config.output_names)?;

        // Build subgraph
        let subgraph = subgraph_builder.build();

        // Validate if requested
        let validation = if self.config.validate {
            let validator = TFLiteValidator::new();
            let result = validator.validate_subgraph(&subgraph);

            if !result.is_valid() {
                return Err(format!("Validation failed:\n{}", result.summary()));
            }

            // Add warnings from validation
            for warning in &result.warnings {
                warnings.push(warning.description());
            }

            Some(result)
        } else {
            None
        };

        // Add subgraph to builder
        fb_builder.add_subgraph(subgraph);

        // Add metadata if requested
        if self.config.include_metadata {
            let metadata = self.create_metadata(input_shape, weights);
            let metadata_bytes = metadata.to_json_bytes()?;
            fb_builder.add_metadata("metadata".to_string(), metadata_bytes);
        }

        // Build FlatBuffer
        let model_bytes = fb_builder.build()?;

        // Create export metadata
        let export_metadata = ExportMetadata::new(&self.config.model_name, "tflite")
            .with_shapes(
                input_shape.to_vec(),
                vec![layer_configs.last().map(|l| l.output_size).unwrap_or(0)],
            )
            .with_total_params(weights.total_params())
            .with_quantization(self.config.quantization.is_some());

        Ok(TFLiteExportResult {
            model_bytes,
            metadata: export_metadata,
            validation,
            warnings,
        })
    }

    /// Create TFLite metadata
    fn create_metadata(&self, input_shape: &[usize], weights: &ModelWeights) -> TFLiteMetadata {
        let mut metadata = TFLiteMetadata::new(
            self.config.model_name.clone(),
            format!(
                "SNN model exported to TFLite: {}",
                weights.metadata.model_name
            ),
        )
        .with_version("1.0.0".to_string())
        .with_author("dpb-snn framework".to_string());

        // Add input metadata
        let input_meta = TensorMetadata::new(
            self.config
                .input_names
                .first()
                .unwrap_or(&"input".to_string())
                .clone(),
            ContentType::FeatureVector,
        )
        .with_description(format!("Input tensor of shape {:?}", input_shape));

        metadata.add_input(input_meta);

        // Add output metadata
        let output_meta = TensorMetadata::new(
            self.config
                .output_names
                .first()
                .unwrap_or(&"output".to_string())
                .clone(),
            ContentType::FeatureVector,
        )
        .with_description("Model output".to_string());

        metadata.add_output(output_meta);

        // Add custom properties
        metadata.add_property("framework".to_string(), "dpb-snn".to_string());
        metadata.add_property(
            "quantized".to_string(),
            self.config.quantization.is_some().to_string(),
        );

        metadata
    }

    /// Validate configuration
    pub fn validate_config(&self) -> Result<(), String> {
        if self.config.model_name.is_empty() {
            return Err("Model name cannot be empty".to_string());
        }

        if self.config.input_names.is_empty() {
            return Err("Must specify at least one input name".to_string());
        }

        if self.config.output_names.is_empty() {
            return Err("Must specify at least one output name".to_string());
        }

        if let Some(ref quant_config) = self.config.quantization {
            quant_config.validate()?;
        }

        Ok(())
    }
}

// Re-export for convenience

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::config::LayerType;

    fn create_test_weights() -> ModelWeights {
        let mut weights = ModelWeights::new("test_model".to_string());

        let layer = crate::export::weights::LayerWeights::new(
            "layer1".to_string(),
            "Linear".to_string(),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        )
        .with_bias(vec![0.1, 0.2]);

        weights.add_layer(layer);
        weights.update_checksum();

        weights
    }

    fn create_test_layers() -> Vec<LayerConfig> {
        vec![LayerConfig::new(
            "layer1".to_string(),
            LayerType::SpikingLinear,
            128,
            64,
        )]
    }

    #[test]
    fn test_tflite_config_creation() {
        let config = TFLiteConfig::new("test_model".to_string());

        assert_eq!(config.model_name, "test_model");
        assert!(config.quantization.is_none());
        assert!(config.include_metadata);
        assert!(config.validate);
    }

    #[test]
    fn test_tflite_config_builder() {
        let config = TFLiteConfig::new("test".to_string())
            .with_inputs(vec!["input1".to_string()])
            .with_outputs(vec!["output1".to_string()])
            .without_metadata()
            .without_validation();

        assert_eq!(config.input_names, vec!["input1".to_string()]);
        assert_eq!(config.output_names, vec!["output1".to_string()]);
        assert!(!config.include_metadata);
        assert!(!config.validate);
    }

    #[test]
    fn test_tflite_config_with_quantization() {
        let quant = QuantizationConfig::int8();
        let config = TFLiteConfig::new("test".to_string()).with_quantization(quant);

        assert!(config.quantization.is_some());
    }

    #[test]
    fn test_tflite_exporter_creation() {
        let config = TFLiteConfig::default();
        let exporter = TFLiteExporter::new(config);

        assert_eq!(exporter.config.model_name, "model");
    }

    #[test]
    fn test_tflite_exporter_validate_config() {
        let config = TFLiteConfig::new("test".to_string());
        let exporter = TFLiteExporter::new(config);

        assert!(exporter.validate_config().is_ok());
    }

    #[test]
    fn test_tflite_exporter_validate_config_empty_name() {
        let config = TFLiteConfig::new("".to_string());
        let exporter = TFLiteExporter::new(config);

        assert!(exporter.validate_config().is_err());
    }

    #[test]
    fn test_tflite_export_basic() {
        let config = TFLiteConfig::new("test_model".to_string());
        let exporter = TFLiteExporter::new(config);

        let weights = create_test_weights();
        let layers = create_test_layers();
        let input_shape = vec![128];

        let result = exporter.export_model(&weights, &layers, &input_shape);

        assert!(result.is_ok(), "export failed: {:?}", result.as_ref().err());
        let export = result.unwrap();
        assert!(!export.model_bytes.is_empty());
        assert_eq!(export.metadata.export_format, "tflite");
    }

    #[test]
    fn test_tflite_export_with_quantization() {
        let quant = QuantizationConfig::int8();
        let config = TFLiteConfig::new("test_model".to_string()).with_quantization(quant);
        let exporter = TFLiteExporter::new(config);

        let weights = create_test_weights();
        let layers = create_test_layers();
        let input_shape = vec![128];

        let result = exporter.export_model(&weights, &layers, &input_shape);

        assert!(result.is_ok());
        let export = result.unwrap();
        assert!(export.metadata.quantized);
    }

    #[test]
    fn test_tflite_export_without_validation() {
        let config = TFLiteConfig::new("test_model".to_string()).without_validation();
        let exporter = TFLiteExporter::new(config);

        let weights = create_test_weights();
        let layers = create_test_layers();
        let input_shape = vec![128];

        let result = exporter.export_model(&weights, &layers, &input_shape);

        assert!(result.is_ok());
        let export = result.unwrap();
        assert!(export.validation.is_none());
    }

    #[test]
    fn test_tflite_export_result_size() {
        let config = TFLiteConfig::new("test_model".to_string());
        let exporter = TFLiteExporter::new(config);

        let weights = create_test_weights();
        let layers = create_test_layers();
        let input_shape = vec![128];

        let result = exporter
            .export_model(&weights, &layers, &input_shape)
            .unwrap();

        assert!(result.size_bytes() > 0);
        assert!(result.size_mb() > 0.0);
    }

    #[test]
    fn test_create_metadata() {
        let config = TFLiteConfig::new("test_model".to_string());
        let exporter = TFLiteExporter::new(config);

        let weights = create_test_weights();
        let input_shape = vec![128];

        let metadata = exporter.create_metadata(&input_shape, &weights);

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.inputs.len(), 1);
        assert_eq!(metadata.outputs.len(), 1);
        assert!(metadata.custom_properties.contains_key("framework"));
    }
}
