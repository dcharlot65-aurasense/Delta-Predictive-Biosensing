//! # dpb-export
//!
//! Model export capabilities for the Delta-Predictive Biosensing (DPB) Framework.
//!
//! ## Overview
//!
//! This crate provides tools for exporting trained spike encoding models to
//! standard formats for deployment across different platforms:
//!
//! - **ONNX**: Open Neural Network Exchange format for universal deployment
//! - **JSON**: Portable configuration and parameter export
//! - **Binary**: Optimized format for embedded deployment
//!
//! ## Features
//!
//! - `onnx` - Enable ONNX export support (requires tract-onnx)
//! - `tensorflow` - Enable TensorFlow SavedModel export
//! - `pytorch` - Enable PyTorch export support
//! - `full` - Enable all export formats
//!
//! ## Example
//!
//! ```rust,ignore
//! use dpb_export::{ModelExporter, ExportFormat, OnnxExporter};
//!
//! // Create exporter for a trained encoder
//! let exporter = ModelExporter::new()
//!     .with_metadata("encoder_type", "level_crossing")
//!     .with_metadata("version", "0.4.0");
//!
//! // Export to ONNX
//! exporter.export_onnx("model.onnx", &encoder_params)?;
//!
//! // Export to JSON for configuration
//! exporter.export_json("config.json")?;
//! ```

pub mod error;
pub mod metadata;
pub mod onnx;
pub mod json;
pub mod binary;
pub mod encoder_export;
pub mod fpga;
pub mod neuromorphic;

pub use error::{ExportError, Result};
pub use metadata::{ModelMetadata, ModelInfo};
pub use encoder_export::{EncoderExport, EncoderParams, ExportableEncoder};

#[cfg(feature = "onnx")]
pub use onnx::OnnxExporter;

pub use json::JsonExporter;
pub use binary::BinaryExporter;

/// Supported export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// ONNX format for universal deployment.
    Onnx,
    /// JSON format for configuration and parameters.
    Json,
    /// Binary format for embedded systems.
    Binary,
    /// TensorFlow SavedModel format.
    TensorFlow,
    /// PyTorch format.
    PyTorch,
}

impl ExportFormat {
    /// Get the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Onnx => "onnx",
            ExportFormat::Json => "json",
            ExportFormat::Binary => "dpb",
            ExportFormat::TensorFlow => "pb",
            ExportFormat::PyTorch => "pt",
        }
    }

    /// Check if this format is supported in the current build.
    pub fn is_supported(&self) -> bool {
        match self {
            ExportFormat::Onnx => cfg!(feature = "onnx"),
            ExportFormat::Json => true,
            ExportFormat::Binary => true,
            ExportFormat::TensorFlow => cfg!(feature = "tensorflow"),
            ExportFormat::PyTorch => cfg!(feature = "pytorch"),
        }
    }
}

/// Main model exporter with builder pattern.
pub struct ModelExporter {
    /// Model metadata.
    metadata: ModelMetadata,
}

impl ModelExporter {
    /// Create a new model exporter.
    pub fn new() -> Self {
        Self {
            metadata: ModelMetadata::new(),
        }
    }

    /// Add metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.add(key, value);
        self
    }

    /// Set the model name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.metadata.set_name(name);
        self
    }

    /// Set the model version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.metadata.set_version(version);
        self
    }

    /// Set the model description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.metadata.set_description(description);
        self
    }

    /// Export to JSON format.
    pub fn export_json<E: ExportableEncoder>(
        &self,
        path: impl AsRef<std::path::Path>,
        encoder: &E,
    ) -> Result<()> {
        let exporter = JsonExporter::new(self.metadata.clone());
        exporter.export(path, encoder)
    }

    /// Export to binary format.
    pub fn export_binary<E: ExportableEncoder>(
        &self,
        path: impl AsRef<std::path::Path>,
        encoder: &E,
    ) -> Result<()> {
        let exporter = BinaryExporter::new(self.metadata.clone());
        exporter.export(path, encoder)
    }

    /// Export to ONNX format.
    #[cfg(feature = "onnx")]
    pub fn export_onnx<E: ExportableEncoder>(
        &self,
        path: impl AsRef<std::path::Path>,
        encoder: &E,
    ) -> Result<()> {
        let exporter = OnnxExporter::new(self.metadata.clone());
        exporter.export(path, encoder)
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
}

impl Default for ModelExporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder_export::MockEncoder;
    use tempfile::tempdir;

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Onnx.extension(), "onnx");
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::Binary.extension(), "dpb");
        assert_eq!(ExportFormat::TensorFlow.extension(), "pb");
        assert_eq!(ExportFormat::PyTorch.extension(), "pt");
    }

    #[test]
    fn test_export_format_is_supported() {
        // JSON and Binary are always supported
        assert!(ExportFormat::Json.is_supported());
        assert!(ExportFormat::Binary.is_supported());

        // ONNX support depends on feature flag
        #[cfg(feature = "onnx")]
        assert!(ExportFormat::Onnx.is_supported());
        #[cfg(not(feature = "onnx"))]
        assert!(!ExportFormat::Onnx.is_supported());

        // TensorFlow support depends on feature flag
        #[cfg(feature = "tensorflow")]
        assert!(ExportFormat::TensorFlow.is_supported());
        #[cfg(not(feature = "tensorflow"))]
        assert!(!ExportFormat::TensorFlow.is_supported());

        // PyTorch support depends on feature flag
        #[cfg(feature = "pytorch")]
        assert!(ExportFormat::PyTorch.is_supported());
        #[cfg(not(feature = "pytorch"))]
        assert!(!ExportFormat::PyTorch.is_supported());
    }

    #[test]
    fn test_export_format_equality() {
        assert_eq!(ExportFormat::Onnx, ExportFormat::Onnx);
        assert_ne!(ExportFormat::Onnx, ExportFormat::Json);
        assert_ne!(ExportFormat::Binary, ExportFormat::TensorFlow);
    }

    #[test]
    fn test_export_format_clone() {
        let format = ExportFormat::Json;
        let cloned = format.clone();
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_export_format_copy() {
        let format = ExportFormat::Binary;
        let copied: ExportFormat = format; // Copy, not move
        assert_eq!(format, copied);
    }

    #[test]
    fn test_export_format_debug() {
        let format = ExportFormat::Onnx;
        let debug_str = format!("{:?}", format);
        assert!(debug_str.contains("Onnx"));
    }

    #[test]
    fn test_model_exporter_new() {
        let exporter = ModelExporter::new();
        assert_eq!(exporter.metadata().name, "DPB Model");
    }

    #[test]
    fn test_model_exporter_default() {
        let exporter = ModelExporter::default();
        assert_eq!(exporter.metadata().name, "DPB Model");
    }

    #[test]
    fn test_model_exporter_with_name() {
        let exporter = ModelExporter::new()
            .with_name("Custom Model");
        assert_eq!(exporter.metadata().name, "Custom Model");
    }

    #[test]
    fn test_model_exporter_with_version() {
        let exporter = ModelExporter::new()
            .with_version("2.0.0");
        assert_eq!(exporter.metadata().version, "2.0.0");
    }

    #[test]
    fn test_model_exporter_with_description() {
        let exporter = ModelExporter::new()
            .with_description("A test model for unit testing");
        assert_eq!(exporter.metadata().description, "A test model for unit testing");
    }

    #[test]
    fn test_model_exporter_with_metadata() {
        let exporter = ModelExporter::new()
            .with_metadata("encoder_type", "level_crossing")
            .with_metadata("channels", "8");

        assert_eq!(exporter.metadata().get("encoder_type"), Some("level_crossing"));
        assert_eq!(exporter.metadata().get("channels"), Some("8"));
    }

    #[test]
    fn test_model_exporter_builder_chain() {
        let exporter = ModelExporter::new()
            .with_name("Chained Model")
            .with_version("1.0.0")
            .with_description("Testing builder pattern")
            .with_metadata("key1", "value1")
            .with_metadata("key2", "value2");

        let metadata = exporter.metadata();
        assert_eq!(metadata.name, "Chained Model");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.description, "Testing builder pattern");
        assert_eq!(metadata.get("key1"), Some("value1"));
        assert_eq!(metadata.get("key2"), Some("value2"));
    }

    #[test]
    fn test_model_exporter_export_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.json");

        let exporter = ModelExporter::new()
            .with_name("JSON Export Test");

        let encoder = MockEncoder::level_crossing(8, 256.0, 0.1);
        let result = exporter.export_json(&path, &encoder);

        assert!(result.is_ok());
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("JSON Export Test"));
    }

    #[test]
    fn test_model_exporter_export_binary() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.dpb");

        let exporter = ModelExporter::new()
            .with_name("Binary Export Test");

        let encoder = MockEncoder::level_crossing(8, 256.0, 0.1);
        let result = exporter.export_binary(&path, &encoder);

        assert!(result.is_ok());
        assert!(path.exists());

        // Verify it's a valid DPB file
        let content = std::fs::read(&path).unwrap();
        assert_eq!(&content[0..4], b"DPB\x00");
    }

    #[test]
    fn test_model_exporter_metadata_accessor() {
        let exporter = ModelExporter::new()
            .with_name("Accessor Test")
            .with_metadata("custom_key", "custom_value");

        let metadata = exporter.metadata();
        assert_eq!(metadata.name, "Accessor Test");
        assert_eq!(metadata.get("custom_key"), Some("custom_value"));
    }
}
