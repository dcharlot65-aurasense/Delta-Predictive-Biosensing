//! Mobile model format with compression and quantization support.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Model format error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum ModelError {
    /// Invalid model format
    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    /// Unsupported model version
    #[error("Unsupported model version: {version}. Supported range: {min}-{max}")]
    UnsupportedVersion {
        /// Version found in the file.
        version: u32,
        /// Oldest version this build reads.
        min: u32,
        /// Newest version this build reads.
        max: u32,
    },

    /// Corrupted model data
    #[error("Corrupted model data: {0}")]
    CorruptedData(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Compression error
    #[error("Compression error: {0}")]
    CompressionError(String),
}

/// Model format version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Major version
    pub major: u16,

    /// Minor version
    pub minor: u16,

    /// Patch version
    pub patch: u16,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Get version as u32 (major << 16 | minor << 8 | patch)
    pub fn as_u32(&self) -> u32 {
        ((self.major as u32) << 16) | ((self.minor as u32) << 8) | (self.patch as u32)
    }

    /// Create from u32
    pub fn from_u32(version: u32) -> Self {
        Self {
            major: ((version >> 16) & 0xFFFF) as u16,
            minor: ((version >> 8) & 0xFF) as u16,
            patch: (version & 0xFF) as u16,
        }
    }
}

impl fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Quantization type for model weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization (32-bit float)
    Float32,

    /// 16-bit float (half precision)
    Float16,

    /// 8-bit signed integer
    Int8,

    /// 4-bit integer (experimental)
    Int4,
}

impl QuantizationType {
    /// Get bytes per weight for this quantization type
    pub fn bytes_per_weight(&self) -> usize {
        match self {
            QuantizationType::Float32 => 4,
            QuantizationType::Float16 => 2,
            QuantizationType::Int8 => 1,
            QuantizationType::Int4 => 1, // 2 weights per byte
        }
    }

    /// Check if this quantization type requires scale/zero-point parameters
    pub fn requires_quantization_params(&self) -> bool {
        matches!(self, QuantizationType::Int8 | QuantizationType::Int4)
    }
}

/// Model format identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// Delta-Predictive-Biosensing native format
    DpbNative,

    /// ONNX format (future support)
    Onnx,

    /// TensorFlow Lite format (future support)
    TfLite,
}

/// Layer information in mobile model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Layer name
    pub name: String,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Layer type
    pub layer_type: LayerType,

    /// Quantization type for this layer
    pub quantization: QuantizationType,

    /// Weight data (serialized)
    pub weights: Vec<u8>,

    /// Bias data (optional)
    pub bias: Option<Vec<u8>>,

    /// Quantization scale (for Int8/Int4)
    pub scale: Option<f32>,

    /// Quantization zero point (for Int8/Int4)
    pub zero_point: Option<i8>,
}

/// Layer type in SNN
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Fully connected layer
    FullyConnected,

    /// Spiking layer (LIF, Izhikevich, etc.)
    Spiking,

    /// Encoding layer
    Encoding,

    /// Readout layer
    Readout,

    /// Custom layer
    Custom,
}

/// Mobile model with compressed and quantized weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileModel {
    /// Model version
    pub version: ModelVersion,

    /// Model format
    pub format: ModelFormat,

    /// Model name/identifier
    pub name: String,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Layers in the model
    pub layers: Vec<LayerInfo>,

    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetadata {
    /// Model description
    pub description: String,

    /// Model author
    pub author: String,

    /// Creation timestamp (Unix epoch)
    pub created_at: u64,

    /// Last modified timestamp
    pub modified_at: u64,

    /// Model accuracy (if known)
    pub accuracy: Option<f32>,

    /// Training dataset info
    pub training_dataset: Option<String>,

    /// Additional tags
    pub tags: Vec<String>,
}

impl MobileModel {
    /// Create a new mobile model
    pub fn new(name: String, input_dim: usize, output_dim: usize) -> Self {
        Self {
            version: ModelVersion::new(1, 0, 0),
            format: ModelFormat::DpbNative,
            name,
            input_dim,
            output_dim,
            layers: Vec::new(),
            metadata: ModelMetadata::default(),
        }
    }

    /// Load model from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        // Validate minimum size
        if bytes.len() < 16 {
            return Err(ModelError::InvalidFormat("Model data too small".into()));
        }

        // Check magic number (first 4 bytes: "DPB\0")
        if &bytes[0..4] != b"DPB\0" {
            return Err(ModelError::InvalidFormat("Invalid magic number".into()));
        }

        // Check version (next 4 bytes)
        let version_u32 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let version = ModelVersion::from_u32(version_u32);

        // Validate version
        if version.as_u32() < crate::MIN_MODEL_VERSION
            || version.as_u32() > crate::MAX_MODEL_VERSION
        {
            return Err(ModelError::UnsupportedVersion {
                version: version.as_u32(),
                min: crate::MIN_MODEL_VERSION,
                max: crate::MAX_MODEL_VERSION,
            });
        }

        // Deserialize model data (skip 16-byte header)
        let (model, _): (Self, usize) =
            bincode::serde::decode_from_slice(&bytes[16..], bincode::config::standard())
                .map_err(|e| ModelError::DeserializationError(e.to_string()))?;

        // Validate model structure
        model.validate()?;

        Ok(model)
    }

    /// Serialize model to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, ModelError> {
        // Validate before serialization
        self.validate()?;

        let mut bytes = Vec::new();

        // Magic number
        bytes.extend_from_slice(b"DPB\0");

        // Version
        bytes.extend_from_slice(&self.version.as_u32().to_le_bytes());

        // Reserved bytes (8 bytes for future use)
        bytes.extend_from_slice(&[0u8; 8]);

        // Serialize model data
        let model_bytes = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| ModelError::SerializationError(e.to_string()))?;

        bytes.extend_from_slice(&model_bytes);

        Ok(bytes)
    }

    /// Validate model structure
    pub fn validate(&self) -> Result<(), ModelError> {
        // Check input/output dimensions
        if self.input_dim == 0 {
            return Err(ModelError::InvalidFormat(
                "Input dimension cannot be zero".into(),
            ));
        }

        if self.output_dim == 0 {
            return Err(ModelError::InvalidFormat(
                "Output dimension cannot be zero".into(),
            ));
        }

        // Check layers
        if self.layers.is_empty() {
            return Err(ModelError::InvalidFormat(
                "Model must have at least one layer".into(),
            ));
        }

        // Validate layer dimensions match
        for (i, layer) in self.layers.iter().enumerate() {
            if i == 0 && layer.input_dim != self.input_dim {
                return Err(ModelError::InvalidFormat(format!(
                    "First layer input dimension {} doesn't match model input dimension {}",
                    layer.input_dim, self.input_dim
                )));
            }

            if i == self.layers.len() - 1 && layer.output_dim != self.output_dim {
                return Err(ModelError::InvalidFormat(format!(
                    "Last layer output dimension {} doesn't match model output dimension {}",
                    layer.output_dim, self.output_dim
                )));
            }

            if i > 0 && layer.input_dim != self.layers[i - 1].output_dim {
                return Err(ModelError::InvalidFormat(format!(
                    "Layer {} input dimension {} doesn't match previous layer output dimension {}",
                    i,
                    layer.input_dim,
                    self.layers[i - 1].output_dim
                )));
            }
        }

        Ok(())
    }

    /// Add a layer to the model
    pub fn add_layer(&mut self, layer: LayerInfo) {
        self.layers.push(layer);
    }

    /// Get layer sizes
    pub fn layer_sizes(&self) -> Vec<usize> {
        self.layers.iter().map(|l| l.output_dim).collect()
    }

    /// Get total weight bytes
    pub fn weight_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights.len() + l.bias.as_ref().map(|b| b.len()).unwrap_or(0))
            .sum()
    }

    /// Get model compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let uncompressed_size: usize = self
            .layers
            .iter()
            .map(|l| l.input_dim * l.output_dim * 4) // 4 bytes per float32
            .sum();

        let compressed_size = self.weight_bytes();

        if compressed_size == 0 {
            return 1.0;
        }

        uncompressed_size as f32 / compressed_size as f32
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version() {
        let version = ModelVersion::new(1, 2, 3);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);

        let version_u32 = version.as_u32();
        let version2 = ModelVersion::from_u32(version_u32);
        assert_eq!(version, version2);
    }

    #[test]
    fn test_quantization_type() {
        assert_eq!(QuantizationType::Float32.bytes_per_weight(), 4);
        assert_eq!(QuantizationType::Float16.bytes_per_weight(), 2);
        assert_eq!(QuantizationType::Int8.bytes_per_weight(), 1);
        assert!(QuantizationType::Int8.requires_quantization_params());
        assert!(!QuantizationType::Float32.requires_quantization_params());
    }

    #[test]
    fn test_mobile_model_creation() {
        let model = MobileModel::new("test_model".into(), 10, 5);
        assert_eq!(model.name, "test_model");
        assert_eq!(model.input_dim, 10);
        assert_eq!(model.output_dim, 5);
        assert_eq!(model.layers.len(), 0);
    }

    #[test]
    fn test_model_validation_empty() {
        let model = MobileModel::new("test".into(), 10, 5);
        assert!(model.validate().is_err());
    }

    #[test]
    fn test_model_serialization() {
        let mut model = MobileModel::new("test_model".into(), 10, 5);

        // Add a layer
        let layer = LayerInfo {
            name: "layer1".into(),
            input_dim: 10,
            output_dim: 5,
            layer_type: LayerType::FullyConnected,
            quantization: QuantizationType::Float32,
            weights: vec![0u8; 200],   // 10 * 5 * 4 bytes
            bias: Some(vec![0u8; 20]), // 5 * 4 bytes
            scale: None,
            zero_point: None,
        };
        model.add_layer(layer);

        // Serialize
        let bytes = model.to_bytes().unwrap();

        // Check magic number
        assert_eq!(&bytes[0..4], b"DPB\0");

        // Deserialize
        let model2 = MobileModel::from_bytes(&bytes).unwrap();
        assert_eq!(model2.name, model.name);
        assert_eq!(model2.input_dim, model.input_dim);
        assert_eq!(model2.output_dim, model.output_dim);
        assert_eq!(model2.layers.len(), model.layers.len());
    }
}
