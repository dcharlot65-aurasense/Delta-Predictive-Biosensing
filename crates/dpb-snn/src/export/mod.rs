//! Model export functionality for deployment
//!
//! Provides ONNX export capabilities and other deployment formats
//! for trained spiking neural networks.

pub mod config;
pub mod onnx;
pub mod tflite;
pub mod weights;

pub use config::{ExportMetadata, LayerConfig, LayerType, ModelConfig};
pub use onnx::{ExportResult, OnnxConfig, OnnxExporter};
pub use tflite::{
    QuantizationConfig, QuantizationMode, TFLiteConfig, TFLiteExportResult, TFLiteExporter,
};
pub use weights::{ModelWeights, WeightExporter, WeightFormat};
