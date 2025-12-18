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
