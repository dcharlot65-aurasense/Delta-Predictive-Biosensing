//! Model metadata and information structures.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Metadata for exported models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique model identifier.
    pub id: String,
    /// Model name.
    pub name: String,
    /// Model version.
    pub version: String,
    /// Model description.
    pub description: String,
    /// Creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Last modified timestamp.
    pub modified_at: DateTime<Utc>,
    /// Author information.
    pub author: Option<String>,
    /// License information.
    pub license: Option<String>,
    /// Custom key-value metadata.
    pub custom: HashMap<String, String>,
    /// Model inputs specification.
    pub inputs: Vec<TensorSpec>,
    /// Model outputs specification.
    pub outputs: Vec<TensorSpec>,
}

impl ModelMetadata {
    /// Create new metadata with default values.
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "DPB Model".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: String::new(),
            created_at: now,
            modified_at: now,
            author: Some("AuraSense Tech Corporation".to_string()),
            license: Some("MIT".to_string()),
            custom: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add custom metadata.
    pub fn add(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.custom.insert(key.into(), value.into());
        self.modified_at = Utc::now();
    }

    /// Set the model name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
        self.modified_at = Utc::now();
    }

    /// Set the model version.
    pub fn set_version(&mut self, version: impl Into<String>) {
        self.version = version.into();
        self.modified_at = Utc::now();
    }

    /// Set the model description.
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.description = description.into();
        self.modified_at = Utc::now();
    }

    /// Set the author.
    pub fn set_author(&mut self, author: impl Into<String>) {
        self.author = Some(author.into());
        self.modified_at = Utc::now();
    }

    /// Add an input specification.
    pub fn add_input(&mut self, spec: TensorSpec) {
        self.inputs.push(spec);
        self.modified_at = Utc::now();
    }

    /// Add an output specification.
    pub fn add_output(&mut self, spec: TensorSpec) {
        self.outputs.push(spec);
        self.modified_at = Utc::now();
    }

    /// Get a custom metadata value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.custom.get(key).map(|s| s.as_str())
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Tensor specification for model inputs/outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name.
    pub name: String,
    /// Data type.
    pub dtype: DataType,
    /// Shape (use -1 for dynamic dimensions).
    pub shape: Vec<i64>,
    /// Optional description.
    pub description: Option<String>,
}

impl TensorSpec {
    /// Create a new tensor specification.
    pub fn new(name: impl Into<String>, dtype: DataType, shape: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
            description: None,
        }
    }

    /// Add a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Create a float32 tensor spec.
    pub fn float32(name: impl Into<String>, shape: Vec<i64>) -> Self {
        Self::new(name, DataType::Float32, shape)
    }

    /// Create a float64 tensor spec.
    pub fn float64(name: impl Into<String>, shape: Vec<i64>) -> Self {
        Self::new(name, DataType::Float64, shape)
    }

    /// Create an int32 tensor spec.
    pub fn int32(name: impl Into<String>, shape: Vec<i64>) -> Self {
        Self::new(name, DataType::Int32, shape)
    }

    /// Check if shape contains dynamic dimensions.
    pub fn is_dynamic(&self) -> bool {
        self.shape.iter().any(|&d| d < 0)
    }
}

/// Data types for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
    String,
}

impl DataType {
    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int8 | DataType::UInt8 | DataType::Bool => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 => 4,
            DataType::Int64 | DataType::UInt64 => 8,
            DataType::String => 0, // Variable
        }
    }

    /// Convert to ONNX tensor type.
    pub fn to_onnx_type(&self) -> i32 {
        match self {
            DataType::Float32 => 1,  // FLOAT
            DataType::Float64 => 11, // DOUBLE
            DataType::Int8 => 3,     // INT8
            DataType::Int16 => 5,    // INT16
            DataType::Int32 => 6,    // INT32
            DataType::Int64 => 7,    // INT64
            DataType::UInt8 => 2,    // UINT8
            DataType::UInt16 => 4,   // UINT16
            DataType::UInt32 => 12,  // UINT32
            DataType::UInt64 => 13,  // UINT64
            DataType::Bool => 9,     // BOOL
            DataType::String => 8,   // STRING
        }
    }
}

/// Detailed model information for documentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Basic metadata.
    pub metadata: ModelMetadata,
    /// Encoder type.
    pub encoder_type: String,
    /// Number of input channels.
    pub num_channels: usize,
    /// Expected sample rate.
    pub sample_rate: f64,
    /// Training information.
    pub training: Option<TrainingInfo>,
    /// Performance metrics.
    pub metrics: Option<ModelMetrics>,
}

impl ModelInfo {
    /// Create new model info.
    pub fn new(encoder_type: impl Into<String>, num_channels: usize, sample_rate: f64) -> Self {
        Self {
            metadata: ModelMetadata::new(),
            encoder_type: encoder_type.into(),
            num_channels,
            sample_rate,
            training: None,
            metrics: None,
        }
    }

    /// Add training information.
    pub fn with_training(mut self, training: TrainingInfo) -> Self {
        self.training = Some(training);
        self
    }

    /// Add performance metrics.
    pub fn with_metrics(mut self, metrics: ModelMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }
}

/// Training information for documentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingInfo {
    /// Training dataset name.
    pub dataset: String,
    /// Number of training samples.
    pub num_samples: usize,
    /// Training duration in seconds.
    pub duration_sec: f64,
    /// Epochs trained.
    pub epochs: Option<usize>,
    /// Final loss value.
    pub final_loss: Option<f64>,
}

/// Model performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Spike rate (spikes/second).
    pub spike_rate: f64,
    /// Compression ratio.
    pub compression_ratio: f64,
    /// Signal-to-noise ratio.
    pub snr: Option<f64>,
    /// Reconstruction error.
    pub reconstruction_error: Option<f64>,
    /// Processing latency in milliseconds.
    pub latency_ms: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_creation() {
        let metadata = ModelMetadata::new();
        assert_eq!(metadata.name, "DPB Model");
        assert!(metadata.author.is_some());
        assert_eq!(metadata.author.as_deref(), Some("AuraSense Tech Corporation"));
        assert!(metadata.license.is_some());
        assert!(!metadata.id.is_empty());
    }

    #[test]
    fn test_model_metadata_setters() {
        let mut metadata = ModelMetadata::new();
        let original_modified = metadata.modified_at;

        // Small delay to ensure timestamp changes
        std::thread::sleep(std::time::Duration::from_millis(10));

        metadata.set_name("Test Model");
        assert_eq!(metadata.name, "Test Model");
        assert!(metadata.modified_at >= original_modified);

        metadata.set_version("1.0.0");
        assert_eq!(metadata.version, "1.0.0");

        metadata.set_description("A test model");
        assert_eq!(metadata.description, "A test model");

        metadata.set_author("Test Author");
        assert_eq!(metadata.author, Some("Test Author".to_string()));
    }

    #[test]
    fn test_model_metadata_custom() {
        let mut metadata = ModelMetadata::new();
        metadata.add("encoder_type", "level_crossing");
        metadata.add("threshold", "0.1");

        assert_eq!(metadata.get("encoder_type"), Some("level_crossing"));
        assert_eq!(metadata.get("threshold"), Some("0.1"));
        assert_eq!(metadata.get("nonexistent"), None);
    }

    #[test]
    fn test_model_metadata_io_specs() {
        let mut metadata = ModelMetadata::new();

        let input = TensorSpec::float32("input", vec![-1, 8, 256]);
        let output = TensorSpec::float32("spikes", vec![-1, 8, -1]);

        metadata.add_input(input);
        metadata.add_output(output);

        assert_eq!(metadata.inputs.len(), 1);
        assert_eq!(metadata.outputs.len(), 1);
        assert_eq!(metadata.inputs[0].name, "input");
        assert_eq!(metadata.outputs[0].name, "spikes");
    }

    #[test]
    fn test_tensor_spec_creation() {
        let spec = TensorSpec::new("data", DataType::Float32, vec![1, 8, 256]);
        assert_eq!(spec.name, "data");
        assert_eq!(spec.dtype, DataType::Float32);
        assert_eq!(spec.shape, vec![1, 8, 256]);
        assert!(spec.description.is_none());
        assert!(!spec.is_dynamic());
    }

    #[test]
    fn test_tensor_spec_with_description() {
        let spec = TensorSpec::float32("input", vec![-1, 8, 256])
            .with_description("Input EEG signal");

        assert_eq!(spec.description, Some("Input EEG signal".to_string()));
    }

    #[test]
    fn test_tensor_spec_type_helpers() {
        let f32_spec = TensorSpec::float32("a", vec![1]);
        assert_eq!(f32_spec.dtype, DataType::Float32);

        let f64_spec = TensorSpec::float64("b", vec![2]);
        assert_eq!(f64_spec.dtype, DataType::Float64);

        let i32_spec = TensorSpec::int32("c", vec![3]);
        assert_eq!(i32_spec.dtype, DataType::Int32);
    }

    #[test]
    fn test_tensor_spec_dynamic() {
        let static_spec = TensorSpec::float32("static", vec![1, 8, 256]);
        assert!(!static_spec.is_dynamic());

        let dynamic_spec = TensorSpec::float32("dynamic", vec![-1, 8, -1]);
        assert!(dynamic_spec.is_dynamic());
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::Float32.size(), 4);
        assert_eq!(DataType::Float64.size(), 8);
        assert_eq!(DataType::Int8.size(), 1);
        assert_eq!(DataType::Int16.size(), 2);
        assert_eq!(DataType::Int32.size(), 4);
        assert_eq!(DataType::Int64.size(), 8);
        assert_eq!(DataType::UInt8.size(), 1);
        assert_eq!(DataType::Bool.size(), 1);
        assert_eq!(DataType::String.size(), 0);
    }

    #[test]
    fn test_data_type_to_onnx() {
        assert_eq!(DataType::Float32.to_onnx_type(), 1);
        assert_eq!(DataType::Float64.to_onnx_type(), 11);
        assert_eq!(DataType::Int32.to_onnx_type(), 6);
        assert_eq!(DataType::Int64.to_onnx_type(), 7);
        assert_eq!(DataType::Bool.to_onnx_type(), 9);
        assert_eq!(DataType::String.to_onnx_type(), 8);
    }

    #[test]
    fn test_model_info_creation() {
        let info = ModelInfo::new("level_crossing", 8, 256.0);
        assert_eq!(info.encoder_type, "level_crossing");
        assert_eq!(info.num_channels, 8);
        assert_eq!(info.sample_rate, 256.0);
        assert!(info.training.is_none());
        assert!(info.metrics.is_none());
    }

    #[test]
    fn test_model_info_with_training() {
        let training = TrainingInfo {
            dataset: "test_dataset".to_string(),
            num_samples: 10000,
            duration_sec: 120.0,
            epochs: Some(100),
            final_loss: Some(0.001),
        };

        let info = ModelInfo::new("delta", 4, 512.0)
            .with_training(training.clone());

        assert!(info.training.is_some());
        let t = info.training.unwrap();
        assert_eq!(t.dataset, "test_dataset");
        assert_eq!(t.num_samples, 10000);
    }

    #[test]
    fn test_model_info_with_metrics() {
        let metrics = ModelMetrics {
            spike_rate: 15.5,
            compression_ratio: 10.2,
            snr: Some(25.0),
            reconstruction_error: Some(0.05),
            latency_ms: Some(1.5),
        };

        let info = ModelInfo::new("temporal_contrast", 16, 1024.0)
            .with_metrics(metrics.clone());

        assert!(info.metrics.is_some());
        let m = info.metrics.unwrap();
        assert_eq!(m.spike_rate, 15.5);
        assert_eq!(m.compression_ratio, 10.2);
    }

    #[test]
    fn test_metadata_serialization() {
        let metadata = ModelMetadata::new();
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("DPB Model"));
        assert!(json.contains("AuraSense"));

        let deserialized: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, metadata.name);
        assert_eq!(deserialized.version, metadata.version);
    }

    #[test]
    fn test_tensor_spec_serialization() {
        let spec = TensorSpec::float32("input", vec![-1, 8, 256])
            .with_description("EEG data");

        let json = serde_json::to_string(&spec).unwrap();
        let deserialized: TensorSpec = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, spec.name);
        assert_eq!(deserialized.dtype, spec.dtype);
        assert_eq!(deserialized.shape, spec.shape);
        assert_eq!(deserialized.description, spec.description);
    }

    #[test]
    fn test_default_metadata() {
        let default = ModelMetadata::default();
        let new = ModelMetadata::new();
        assert_eq!(default.name, new.name);
        assert_eq!(default.author, new.author);
    }
}
