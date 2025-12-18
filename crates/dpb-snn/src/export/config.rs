use serde::{Serialize, Deserialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Layer configuration for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub name: String,
    pub layer_type: LayerType,
    pub input_size: usize,
    pub output_size: usize,
    pub activation: Option<String>,
    pub parameters: serde_json::Value,
}

impl LayerConfig {
    /// Create a new layer configuration
    pub fn new(
        name: String,
        layer_type: LayerType,
        input_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            name,
            layer_type,
            input_size,
            output_size,
            activation: None,
            parameters: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Set activation function
    pub fn with_activation(mut self, activation: String) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Set parameters
    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = parameters;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Layer name cannot be empty".to_string());
        }

        if self.input_size == 0 {
            return Err(format!("Layer '{}' has zero input size", self.name));
        }

        if self.output_size == 0 {
            return Err(format!("Layer '{}' has zero output size", self.name));
        }

        // Validate layer-specific constraints
        match &self.layer_type {
            LayerType::SpikingConv1d { kernel_size, stride, padding } => {
                if *kernel_size == 0 {
                    return Err(format!("Layer '{}' has zero kernel size", self.name));
                }
                if *stride == 0 {
                    return Err(format!("Layer '{}' has zero stride", self.name));
                }
            }
            LayerType::SpikingConv2d { kernel_size, stride, padding } => {
                if kernel_size.0 == 0 || kernel_size.1 == 0 {
                    return Err(format!("Layer '{}' has zero kernel size", self.name));
                }
                if stride.0 == 0 || stride.1 == 0 {
                    return Err(format!("Layer '{}' has zero stride", self.name));
                }
            }
            LayerType::Dropout { rate } => {
                if *rate < 0.0 || *rate >= 1.0 {
                    return Err(format!(
                        "Layer '{}' has invalid dropout rate: {}",
                        self.name, rate
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    SpikingLinear,
    SpikingConv1d { 
        kernel_size: usize, 
        stride: usize, 
        padding: usize 
    },
    SpikingConv2d { 
        kernel_size: (usize, usize), 
        stride: (usize, usize), 
        padding: (usize, usize) 
    },
    SpikingRecurrent,
    Encoder { 
        encoder_type: String 
    },
    Decoder { 
        decoder_type: String 
    },
    BatchNorm,
    Dropout { 
        rate: f64 
    },
}

impl fmt::Display for LayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerType::SpikingLinear => write!(f, "SpikingLinear"),
            LayerType::SpikingConv1d { .. } => write!(f, "SpikingConv1d"),
            LayerType::SpikingConv2d { .. } => write!(f, "SpikingConv2d"),
            LayerType::SpikingRecurrent => write!(f, "SpikingRecurrent"),
            LayerType::Encoder { encoder_type } => write!(f, "Encoder({})", encoder_type),
            LayerType::Decoder { decoder_type } => write!(f, "Decoder({})", decoder_type),
            LayerType::BatchNorm => write!(f, "BatchNorm"),
            LayerType::Dropout { rate } => write!(f, "Dropout({})", rate),
        }
    }
}

/// Complete model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub layers: Vec<LayerConfig>,
    pub neuron_model: String,
    pub time_steps: usize,
    pub sample_rate: Option<f64>,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            layers: Vec::new(),
            neuron_model: "LIF".to_string(), // Default to Leaky Integrate-and-Fire
            time_steps: 100,
            sample_rate: None,
        }
    }

    /// Set input shape
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = shape;
        self
    }

    /// Set output shape
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = shape;
        self
    }

    /// Set neuron model
    pub fn with_neuron_model(mut self, model: String) -> Self {
        self.neuron_model = model;
        self
    }

    /// Set time steps
    pub fn with_time_steps(mut self, steps: usize) -> Self {
        self.time_steps = steps;
        self
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Add a layer configuration
    pub fn add_layer(&mut self, layer: LayerConfig) {
        self.layers.push(layer);
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| "{}".to_string())
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json)
            .map_err(|e| format!("Failed to parse JSON: {}", e))
    }

    /// Validate the entire model configuration
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Validate name
        if self.name.is_empty() {
            errors.push("Model name cannot be empty".to_string());
        }

        // Validate version
        if self.version.is_empty() {
            errors.push("Model version cannot be empty".to_string());
        }

        // Validate shapes
        if self.input_shape.is_empty() {
            errors.push("Input shape cannot be empty".to_string());
        } else if self.input_shape.iter().any(|&s| s == 0) {
            errors.push("Input shape contains zero dimensions".to_string());
        }

        if self.output_shape.is_empty() {
            errors.push("Output shape cannot be empty".to_string());
        } else if self.output_shape.iter().any(|&s| s == 0) {
            errors.push("Output shape contains zero dimensions".to_string());
        }

        // Validate layers
        if self.layers.is_empty() {
            errors.push("Model must have at least one layer".to_string());
        }

        for (i, layer) in self.layers.iter().enumerate() {
            if let Err(e) = layer.validate() {
                errors.push(format!("Layer {}: {}", i, e));
            }
        }

        // Validate time steps
        if self.time_steps == 0 {
            errors.push("Time steps must be greater than zero".to_string());
        }

        // Validate sample rate if present
        if let Some(rate) = self.sample_rate {
            if rate <= 0.0 {
                errors.push(format!("Sample rate must be positive: {}", rate));
            }
        }

        // Validate layer connections
        if self.layers.len() > 1 {
            for i in 0..self.layers.len() - 1 {
                let current_output = self.layers[i].output_size;
                let next_input = self.layers[i + 1].input_size;

                // For most layers, output of one should match input of next
                // Skip validation for certain layer types that change dimensions
                let skip_validation = matches!(
                    self.layers[i].layer_type,
                    LayerType::BatchNorm | LayerType::Dropout { .. }
                );

                if !skip_validation && current_output != next_input {
                    errors.push(format!(
                        "Layer dimension mismatch: layer {} output ({}) != layer {} input ({})",
                        i, current_output, i + 1, next_input
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get total number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer by name
    pub fn get_layer(&self, name: &str) -> Option<&LayerConfig> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Get layer by index
    pub fn get_layer_by_index(&self, index: usize) -> Option<&LayerConfig> {
        self.layers.get(index)
    }
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub model_name: String,
    pub export_format: String,
    pub export_version: String,
    pub created_at: String,
    pub framework_version: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub total_params: usize,
    pub quantized: bool,
    pub notes: Option<String>,
}

impl ExportMetadata {
    /// Create new export metadata
    pub fn new(model_name: &str, format: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            export_format: format.to_string(),
            export_version: "1.0.0".to_string(),
            created_at: Self::current_timestamp(),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            total_params: 0,
            quantized: false,
            notes: None,
        }
    }

    /// Set input and output shapes
    pub fn with_shapes(mut self, input: Vec<usize>, output: Vec<usize>) -> Self {
        self.input_shape = input;
        self.output_shape = output;
        self
    }

    /// Set total parameters
    pub fn with_total_params(mut self, params: usize) -> Self {
        self.total_params = params;
        self
    }

    /// Mark as quantized
    pub fn with_quantization(mut self, quantized: bool) -> Self {
        self.quantized = quantized;
        self
    }

    /// Add notes
    pub fn with_notes(mut self, notes: String) -> Self {
        self.notes = Some(notes);
        self
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| "{}".to_string())
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json)
            .map_err(|e| format!("Failed to parse JSON: {}", e))
    }

    /// Get current timestamp as ISO 8601 string
    fn current_timestamp() -> String {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => {
                let seconds = duration.as_secs();
                format!("timestamp_{}", seconds)
            }
            Err(_) => "unknown_time".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_config_creation() {
        let layer = LayerConfig::new(
            "test_layer".to_string(),
            LayerType::SpikingLinear,
            128,
            64,
        );

        assert_eq!(layer.name, "test_layer");
        assert_eq!(layer.input_size, 128);
        assert_eq!(layer.output_size, 64);
        assert!(layer.activation.is_none());
    }

    #[test]
    fn test_layer_config_with_activation() {
        let layer = LayerConfig::new(
            "test".to_string(),
            LayerType::SpikingLinear,
            128,
            64,
        ).with_activation("relu".to_string());

        assert_eq!(layer.activation, Some("relu".to_string()));
    }

    #[test]
    fn test_layer_config_validation_success() {
        let layer = LayerConfig::new(
            "test".to_string(),
            LayerType::SpikingLinear,
            128,
            64,
        );

        assert!(layer.validate().is_ok());
    }

    #[test]
    fn test_layer_config_validation_empty_name() {
        let layer = LayerConfig::new(
            "".to_string(),
            LayerType::SpikingLinear,
            128,
            64,
        );

        assert!(layer.validate().is_err());
    }

    #[test]
    fn test_layer_config_validation_zero_input() {
        let layer = LayerConfig::new(
            "test".to_string(),
            LayerType::SpikingLinear,
            0,
            64,
        );

        assert!(layer.validate().is_err());
    }

    #[test]
    fn test_layer_config_validation_invalid_dropout() {
        let layer = LayerConfig::new(
            "test".to_string(),
            LayerType::Dropout { rate: 1.5 },
            128,
            128,
        );

        assert!(layer.validate().is_err());
    }

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig::new("test_model");

        assert_eq!(config.name, "test_model");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.neuron_model, "LIF");
        assert_eq!(config.time_steps, 100);
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::new("test_model")
            .with_input_shape(vec![128])
            .with_output_shape(vec![10])
            .with_neuron_model("Izhikevich".to_string())
            .with_time_steps(200)
            .with_sample_rate(1000.0);

        assert_eq!(config.input_shape, vec![128]);
        assert_eq!(config.output_shape, vec![10]);
        assert_eq!(config.neuron_model, "Izhikevich");
        assert_eq!(config.time_steps, 200);
        assert_eq!(config.sample_rate, Some(1000.0));
    }

    #[test]
    fn test_model_config_add_layer() {
        let mut config = ModelConfig::new("test_model");
        let layer = LayerConfig::new(
            "layer1".to_string(),
            LayerType::SpikingLinear,
            128,
            64,
        );

        config.add_layer(layer);

        assert_eq!(config.num_layers(), 1);
        assert_eq!(config.get_layer("layer1").unwrap().name, "layer1");
    }

    #[test]
    fn test_model_config_validation_empty() {
        let config = ModelConfig::new("test");
        let result = config.validate();

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_model_config_validation_success() {
        let mut config = ModelConfig::new("test_model")
            .with_input_shape(vec![128])
            .with_output_shape(vec![10]);

        let layer = LayerConfig::new(
            "layer1".to_string(),
            LayerType::SpikingLinear,
            128,
            10,
        );
        config.add_layer(layer);

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_model_config_validation_dimension_mismatch() {
        let mut config = ModelConfig::new("test_model")
            .with_input_shape(vec![128])
            .with_output_shape(vec![10]);

        config.add_layer(LayerConfig::new(
            "layer1".to_string(),
            LayerType::SpikingLinear,
            128,
            64,
        ));

        config.add_layer(LayerConfig::new(
            "layer2".to_string(),
            LayerType::SpikingLinear,
            32,  // Mismatch: should be 64
            10,
        ));

        let result = config.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("dimension mismatch")));
    }

    #[test]
    fn test_model_config_json_roundtrip() {
        let mut config = ModelConfig::new("test_model")
            .with_input_shape(vec![128])
            .with_output_shape(vec![10]);

        config.add_layer(LayerConfig::new(
            "layer1".to_string(),
            LayerType::SpikingLinear,
            128,
            10,
        ));

        let json = config.to_json();
        let loaded = ModelConfig::from_json(&json).unwrap();

        assert_eq!(loaded.name, config.name);
        assert_eq!(loaded.num_layers(), config.num_layers());
    }

    #[test]
    fn test_export_metadata_creation() {
        let metadata = ExportMetadata::new("test_model", "onnx");

        assert_eq!(metadata.model_name, "test_model");
        assert_eq!(metadata.export_format, "onnx");
        assert!(!metadata.quantized);
    }

    #[test]
    fn test_export_metadata_builder() {
        let metadata = ExportMetadata::new("test_model", "onnx")
            .with_shapes(vec![128], vec![10])
            .with_total_params(1000)
            .with_quantization(true)
            .with_notes("Test export".to_string());

        assert_eq!(metadata.input_shape, vec![128]);
        assert_eq!(metadata.output_shape, vec![10]);
        assert_eq!(metadata.total_params, 1000);
        assert!(metadata.quantized);
        assert_eq!(metadata.notes, Some("Test export".to_string()));
    }

    #[test]
    fn test_export_metadata_json_roundtrip() {
        let metadata = ExportMetadata::new("test_model", "onnx")
            .with_shapes(vec![128], vec![10]);

        let json = metadata.to_json();
        let loaded = ExportMetadata::from_json(&json).unwrap();

        assert_eq!(loaded.model_name, metadata.model_name);
        assert_eq!(loaded.input_shape, metadata.input_shape);
    }

    #[test]
    fn test_layer_type_display() {
        assert_eq!(LayerType::SpikingLinear.to_string(), "SpikingLinear");
        assert_eq!(
            LayerType::Encoder { encoder_type: "rate".to_string() }.to_string(),
            "Encoder(rate)"
        );
        assert_eq!(
            LayerType::Dropout { rate: 0.5 }.to_string(),
            "Dropout(0.5)"
        );
    }
}
