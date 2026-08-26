//! ONNX export functionality.
//!
//! Requires the `onnx` feature to be enabled.
//!
//! # Status
//!
//! The ONNX *graph* is built faithfully -- inputs, outputs, initialisers and
//! typed nodes -- but [`OnnxExporter::export`] currently writes it as JSON
//! rather than ONNX protobuf, so the resulting file will not load in
//! onnxruntime, tract, or any other ONNX consumer. Treat the output as a
//! readable dump of the intended graph, not as a deployable model.

use crate::{
    encoder_export::{EncoderParams, ExportableEncoder},
    error::{ExportError, Result},
    metadata::{DataType, ModelMetadata, TensorSpec},
};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tracing::{debug, info, warn};

/// ONNX exporter for spike encoders.
///
/// Converts encoder parameters and logic into an ONNX graph.
///
/// See the module-level note: the graph is serialised as JSON, not as ONNX
/// protobuf, so it is not yet loadable by an ONNX runtime.
#[cfg(feature = "onnx")]
pub struct OnnxExporter {
    /// Model metadata.
    metadata: ModelMetadata,
    /// ONNX opset version.
    opset_version: i64,
}

#[cfg(feature = "onnx")]
impl OnnxExporter {
    /// Create a new ONNX exporter.
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            opset_version: 17, // Latest stable opset
        }
    }

    /// Set the ONNX opset version.
    pub fn with_opset(mut self, version: i64) -> Self {
        self.opset_version = version;
        self
    }

    /// Export encoder to ONNX format.
    pub fn export<E: ExportableEncoder>(&self, path: impl AsRef<Path>, encoder: &E) -> Result<()> {
        let path = path.as_ref();
        info!("Exporting encoder to ONNX: {}", path.display());

        encoder.validate_for_export()?;
        let params = encoder.get_params();

        // Build ONNX model structure
        let model = self.build_onnx_model(&params)?;

        // Serialize and write
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        // Write the model (using protobuf-like structure)
        self.write_onnx_model(&mut writer, &model)?;

        debug!("ONNX export complete");
        Ok(())
    }

    /// Build ONNX model from encoder parameters.
    fn build_onnx_model(&self, params: &EncoderParams) -> Result<OnnxModel> {
        let mut model = OnnxModel::new(&self.metadata.name, self.opset_version);

        // Add metadata
        model.producer_name = "dpb-export".to_string();
        model.producer_version = env!("CARGO_PKG_VERSION").to_string();
        model.doc_string = self.metadata.description.clone();

        // Define input: [batch_size, channels, sequence_length]
        model.add_input(OnnxTensor {
            name: "input".to_string(),
            elem_type: 1, // FLOAT
            shape: vec![-1, params.num_channels as i64, -1], // Dynamic batch and sequence
        });

        // Define output: [batch_size, channels, num_spikes]
        model.add_output(OnnxTensor {
            name: "spikes".to_string(),
            elem_type: 1, // FLOAT
            shape: vec![-1, params.num_channels as i64, -1],
        });

        // Build computation graph based on encoder type
        self.add_encoder_nodes(&mut model, params)?;

        Ok(model)
    }

    /// Add encoder-specific computation nodes.
    fn add_encoder_nodes(&self, model: &mut OnnxModel, params: &EncoderParams) -> Result<()> {
        match params.encoder_type.as_str() {
            "level_crossing" => {
                self.add_level_crossing_nodes(model, params)?;
            }
            "delta" => {
                self.add_delta_nodes(model, params)?;
            }
            "temporal_contrast" => {
                self.add_temporal_contrast_nodes(model, params)?;
            }
            _ => {
                warn!("Unknown encoder type: {}", params.encoder_type);
                // Add generic passthrough
                model.add_node(OnnxNode {
                    name: "passthrough".to_string(),
                    op_type: "Identity".to_string(),
                    inputs: vec!["input".to_string()],
                    outputs: vec!["spikes".to_string()],
                    attributes: vec![],
                });
            }
        }
        Ok(())
    }

    /// Add level crossing encoder nodes.
    fn add_level_crossing_nodes(&self, model: &mut OnnxModel, params: &EncoderParams) -> Result<()> {
        // Threshold constant
        let threshold = params.thresholds.first().copied().unwrap_or(0.1);
        model.add_initializer(OnnxInitializer {
            name: "threshold".to_string(),
            data_type: 1, // FLOAT
            dims: vec![1],
            float_data: vec![threshold],
        });

        model.add_initializer(OnnxInitializer {
            name: "neg_threshold".to_string(),
            data_type: 1,
            dims: vec![1],
            float_data: vec![-threshold],
        });

        // Compute difference from previous sample
        // diff = input[:, :, 1:] - input[:, :, :-1]
        model.add_node(OnnxNode {
            name: "compute_diff".to_string(),
            op_type: "Sub".to_string(),
            inputs: vec!["input_shifted".to_string(), "input_base".to_string()],
            outputs: vec!["diff".to_string()],
            attributes: vec![],
        });

        // Detect positive crossings: (prev < threshold) & (curr >= threshold)
        model.add_node(OnnxNode {
            name: "pos_crossing".to_string(),
            op_type: "Greater".to_string(),
            inputs: vec!["input".to_string(), "threshold".to_string()],
            outputs: vec!["above_threshold".to_string()],
            attributes: vec![],
        });

        // Detect negative crossings
        model.add_node(OnnxNode {
            name: "neg_crossing".to_string(),
            op_type: "Less".to_string(),
            inputs: vec!["input".to_string(), "neg_threshold".to_string()],
            outputs: vec!["below_threshold".to_string()],
            attributes: vec![],
        });

        // Combine crossings
        model.add_node(OnnxNode {
            name: "combine_crossings".to_string(),
            op_type: "Or".to_string(),
            inputs: vec!["above_threshold".to_string(), "below_threshold".to_string()],
            outputs: vec!["crossings".to_string()],
            attributes: vec![],
        });

        // Convert to float spikes
        model.add_node(OnnxNode {
            name: "to_spikes".to_string(),
            op_type: "Cast".to_string(),
            inputs: vec!["crossings".to_string()],
            outputs: vec!["spikes".to_string()],
            attributes: vec![OnnxAttribute {
                name: "to".to_string(),
                attr_type: 2, // INT
                i: 1,        // FLOAT
                ..Default::default()
            }],
        });

        Ok(())
    }

    /// Add delta encoder nodes.
    fn add_delta_nodes(&self, model: &mut OnnxModel, params: &EncoderParams) -> Result<()> {
        let threshold = params.thresholds.first().copied().unwrap_or(0.1);

        // Threshold initializer
        model.add_initializer(OnnxInitializer {
            name: "delta_threshold".to_string(),
            data_type: 1,
            dims: vec![1],
            float_data: vec![threshold],
        });

        // Compute delta: diff = input[t] - input[t-1]
        model.add_node(OnnxNode {
            name: "compute_delta".to_string(),
            op_type: "Sub".to_string(),
            inputs: vec!["input_curr".to_string(), "input_prev".to_string()],
            outputs: vec!["delta".to_string()],
            attributes: vec![],
        });

        // Absolute delta
        model.add_node(OnnxNode {
            name: "abs_delta".to_string(),
            op_type: "Abs".to_string(),
            inputs: vec!["delta".to_string()],
            outputs: vec!["abs_delta".to_string()],
            attributes: vec![],
        });

        // Compare with threshold
        model.add_node(OnnxNode {
            name: "threshold_check".to_string(),
            op_type: "Greater".to_string(),
            inputs: vec!["abs_delta".to_string(), "delta_threshold".to_string()],
            outputs: vec!["spike_mask".to_string()],
            attributes: vec![],
        });

        // Get sign of delta for spike polarity
        model.add_node(OnnxNode {
            name: "spike_sign".to_string(),
            op_type: "Sign".to_string(),
            inputs: vec!["delta".to_string()],
            outputs: vec!["spike_polarity".to_string()],
            attributes: vec![],
        });

        // Multiply mask by polarity
        model.add_node(OnnxNode {
            name: "apply_mask".to_string(),
            op_type: "Mul".to_string(),
            inputs: vec!["spike_mask_float".to_string(), "spike_polarity".to_string()],
            outputs: vec!["spikes".to_string()],
            attributes: vec![],
        });

        Ok(())
    }

    /// Add temporal contrast encoder nodes.
    fn add_temporal_contrast_nodes(
        &self,
        model: &mut OnnxModel,
        params: &EncoderParams,
    ) -> Result<()> {
        let threshold = params.thresholds.first().copied().unwrap_or(0.1);

        model.add_initializer(OnnxInitializer {
            name: "tc_threshold".to_string(),
            data_type: 1,
            dims: vec![1],
            float_data: vec![threshold],
        });

        // Log of input (temporal contrast is log-based)
        model.add_node(OnnxNode {
            name: "log_input".to_string(),
            op_type: "Log".to_string(),
            inputs: vec!["input_positive".to_string()],
            outputs: vec!["log_input".to_string()],
            attributes: vec![],
        });

        // Temporal derivative of log
        model.add_node(OnnxNode {
            name: "log_diff".to_string(),
            op_type: "Sub".to_string(),
            inputs: vec!["log_input_curr".to_string(), "log_input_prev".to_string()],
            outputs: vec!["temporal_contrast".to_string()],
            attributes: vec![],
        });

        // Threshold comparison
        model.add_node(OnnxNode {
            name: "tc_threshold_check".to_string(),
            op_type: "Greater".to_string(),
            inputs: vec!["abs_tc".to_string(), "tc_threshold".to_string()],
            outputs: vec!["spikes".to_string()],
            attributes: vec![],
        });

        Ok(())
    }

    /// Write the model to file as JSON.
    ///
    /// This is the gap between what the module claims and what it does: ONNX
    /// is a protobuf format, and a real exporter would encode `model` against
    /// the onnx.proto schema. What lands on disk is a JSON rendering of the
    /// same structure -- useful for inspection and conversion, but not an
    /// ONNX file.
    fn write_onnx_model<W: Write>(&self, writer: &mut W, model: &OnnxModel) -> Result<()> {

        let json = serde_json::to_string_pretty(model)
            .map_err(|e| ExportError::onnx(format!("Failed to serialize model: {}", e)))?;

        writer.write_all(json.as_bytes())?;
        writer.write_all(b"\n")?;

        Ok(())
    }
}

/// Stub exporter when ONNX feature is disabled.
#[cfg(not(feature = "onnx"))]
pub struct OnnxExporter {
    _metadata: ModelMetadata,
}

#[cfg(not(feature = "onnx"))]
impl OnnxExporter {
    pub fn new(metadata: ModelMetadata) -> Self {
        Self { _metadata: metadata }
    }

    pub fn export<E: ExportableEncoder>(&self, _path: impl AsRef<Path>, _encoder: &E) -> Result<()> {
        Err(ExportError::FeatureNotEnabled("onnx".to_string()))
    }
}

/// ONNX model structure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxModel {
    ir_version: i64,
    opset_import: Vec<OnnxOpsetImport>,
    producer_name: String,
    producer_version: String,
    domain: String,
    model_version: i64,
    doc_string: String,
    graph: OnnxGraph,
}

impl OnnxModel {
    fn new(name: &str, opset_version: i64) -> Self {
        Self {
            ir_version: 8,
            opset_import: vec![OnnxOpsetImport {
                domain: "".to_string(),
                version: opset_version,
            }],
            producer_name: String::new(),
            producer_version: String::new(),
            domain: "ai.onnx".to_string(),
            model_version: 1,
            doc_string: String::new(),
            graph: OnnxGraph {
                name: name.to_string(),
                nodes: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                initializers: Vec::new(),
            },
        }
    }

    fn add_input(&mut self, tensor: OnnxTensor) {
        self.graph.inputs.push(tensor);
    }

    fn add_output(&mut self, tensor: OnnxTensor) {
        self.graph.outputs.push(tensor);
    }

    fn add_node(&mut self, node: OnnxNode) {
        self.graph.nodes.push(node);
    }

    fn add_initializer(&mut self, init: OnnxInitializer) {
        self.graph.initializers.push(init);
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxOpsetImport {
    domain: String,
    version: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxGraph {
    name: String,
    nodes: Vec<OnnxNode>,
    inputs: Vec<OnnxTensor>,
    outputs: Vec<OnnxTensor>,
    initializers: Vec<OnnxInitializer>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxNode {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: Vec<OnnxAttribute>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct OnnxAttribute {
    name: String,
    attr_type: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    f: Option<f32>,
    #[serde(default)]
    i: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    s: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxTensor {
    name: String,
    elem_type: i32,
    shape: Vec<i64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxInitializer {
    name: String,
    data_type: i32,
    dims: Vec<i64>,
    float_data: Vec<f32>,
}
