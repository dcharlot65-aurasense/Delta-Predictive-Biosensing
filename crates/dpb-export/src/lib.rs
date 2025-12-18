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
