//! TensorFlow Lite export functionality
//!
//! Exports trained spiking neural networks to TensorFlow Lite format for
//! deployment on mobile and embedded devices.
//!
//! # Features
//!
//! - FlatBuffer generation for TFLite format
//! - Operator mapping from SNN ops to TFLite built-in/custom ops
//! - Post-training quantization (Int8, Float16)
//! - Model metadata embedding
//! - Validation and compatibility checks
//!
//! # Example
//!
//! ```rust
//! use dpb_snn::export::tflite::*;
//!
//! # fn example() -> Result<(), String> {
//! // Create TFLite exporter with quantization
//! let config = TFLiteConfig::default()
//!     .with_quantization(QuantizationConfig::int8());
//!
//! let exporter = TFLiteExporter::new(config);
//!
//! // Export model (assuming you have weights and layer configs)
//! # let weights = dpb_snn::export::weights::ModelWeights::new("test".to_string());
//! # let layer_configs = vec![];
//! # let input_shape = vec![128];
//! let result = exporter.export_model(&weights, &layer_configs, &input_shape)?;
//!
//! // Save to file
//! std::fs::write("model.tflite", &result.model_bytes).unwrap();
//! # Ok(())
//! # }
//! ```

pub mod exporter;
pub mod operators;
pub mod tensors;
pub mod quantization;
pub mod flatbuffer;
pub mod metadata;
pub mod validation;

pub use exporter::{TFLiteExporter, TFLiteConfig, TFLiteExportResult};
pub use operators::{TFLiteOperator, BuiltinOperator, CustomOperator, OperatorVersion};
pub use tensors::{TFLiteTensor, TensorType, TensorShape, QuantizationParams};
pub use quantization::{
    QuantizationConfig, QuantizationMode, QuantizationStrategy,
    PostTrainingQuantizer, QuantizationAwareTraining,
};
pub use flatbuffer::{FlatBufferBuilder, SubgraphBuilder, BufferManager};
pub use metadata::{TFLiteMetadata, ModelDescription, TensorMetadata};
pub use validation::{TFLiteValidator, ValidationResult, CompatibilityWarning};
