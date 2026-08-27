//! ONNX export functionality.
//!
//! Requires the `onnx` feature to be enabled.
//!
//! Encoders are emitted as ONNX `ModelProto` files at opset 17, serialised
//! through [`crate::protobuf`] rather than a generated schema binding. The
//! output loads in onnxruntime and passes `onnx.checker` with `full_check`.
//!
//! Each encoder becomes a graph over `[batch, channels, time]`, with the batch
//! and time axes left symbolic. All three compare consecutive samples, so the
//! output is one sample shorter than the input -- the first sample has no
//! predecessor to compare against.

use crate::metadata::ModelMetadata;

// Everything below the exporter is gated on `onnx`, so these are too --
// without the feature the module is just the stub that reports the feature
// is off.
#[cfg(feature = "onnx")]
use crate::encoder_export::{EncoderParams, ExportableEncoder};
#[cfg(feature = "onnx")]
use crate::protobuf::Writer;
#[cfg(feature = "onnx")]
use std::fs::File;
#[cfg(feature = "onnx")]
use std::io::Write;
#[cfg(feature = "onnx")]
use tracing::{debug, info, warn};

/// ONNX exporter for spike encoders.
///
/// Converts encoder parameters and logic into an ONNX graph.
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
            elem_type: 1,                                    // FLOAT
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

    /// Emit the pair of `Slice` nodes every encoder here needs.
    ///
    /// All three encoders compare each sample with the one before it, which in
    /// ONNX means slicing the time axis twice and lining the halves up:
    /// `prev` is `src[..., ..-1]` and `curr` is `src[..., 1..]`. Both come out
    /// one sample shorter than the input, which is inherent -- the first
    /// sample has no predecessor to compare against.
    ///
    /// Returns the two output names, in `(prev, curr)` order.
    fn add_time_shift(&self, model: &mut OnnxModel, src: &str, tag: &str) -> (String, String) {
        // Slice takes starts/ends/axes as inputs from opset 10 on, not as
        // attributes, so they have to be initializers rather than literals.
        let starts_prev = format!("{tag}_starts_prev");
        let ends_prev = format!("{tag}_ends_prev");
        let starts_curr = format!("{tag}_starts_curr");
        let ends_curr = format!("{tag}_ends_curr");
        let axes = format!("{tag}_axes");

        for (name, value) in [
            (&starts_prev, 0),
            (&ends_prev, -1),
            (&starts_curr, 1),
            // ONNX clamps an out-of-range end, so INT64_MAX means "to the end".
            (&ends_curr, i64::MAX),
            (&axes, TIME_AXIS),
        ] {
            model.add_initializer(OnnxInitializer::int64(name, vec![1], vec![value]));
        }

        let prev = format!("{tag}_prev");
        let curr = format!("{tag}_curr");

        model.add_node(OnnxNode {
            name: format!("{tag}_slice_prev"),
            op_type: "Slice".to_string(),
            inputs: vec![src.to_string(), starts_prev, ends_prev, axes.clone()],
            outputs: vec![prev.clone()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: format!("{tag}_slice_curr"),
            op_type: "Slice".to_string(),
            inputs: vec![src.to_string(), starts_curr, ends_curr, axes],
            outputs: vec![curr.clone()],
            attributes: vec![],
        });

        (prev, curr)
    }

    /// Cast a boolean tensor to float, which is what the graph declares it
    /// outputs. Comparison ops produce `bool`, so this is never optional.
    fn add_bool_to_float(&self, model: &mut OnnxModel, src: &str, name: &str, out: &str) {
        model.add_node(OnnxNode {
            name: name.to_string(),
            op_type: "Cast".to_string(),
            inputs: vec![src.to_string()],
            outputs: vec![out.to_string()],
            attributes: vec![OnnxAttribute {
                name: "to".to_string(),
                attr_type: attr_type::INT,
                i: i64::from(data_type::FLOAT),
                ..Default::default()
            }],
        });
    }

    /// Add level crossing encoder nodes.
    ///
    /// A spike wherever the sample-to-sample change leaves the +/- threshold
    /// band. Comparing the raw sample against the threshold instead would be a
    /// level *detector*, which is a different thing.
    fn add_level_crossing_nodes(
        &self,
        model: &mut OnnxModel,
        params: &EncoderParams,
    ) -> Result<()> {
        let threshold = params.thresholds.first().copied().unwrap_or(0.1);
        model.add_initializer(OnnxInitializer::float(
            "threshold",
            vec![1],
            vec![threshold],
        ));
        model.add_initializer(OnnxInitializer::float(
            "neg_threshold",
            vec![1],
            vec![-threshold],
        ));

        let (prev, curr) = self.add_time_shift(model, "input", "lc");

        model.add_node(OnnxNode {
            name: "compute_diff".to_string(),
            op_type: "Sub".to_string(),
            inputs: vec![curr, prev],
            outputs: vec!["diff".to_string()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: "pos_crossing".to_string(),
            op_type: "Greater".to_string(),
            inputs: vec!["diff".to_string(), "threshold".to_string()],
            outputs: vec!["above_threshold".to_string()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: "neg_crossing".to_string(),
            op_type: "Less".to_string(),
            inputs: vec!["diff".to_string(), "neg_threshold".to_string()],
            outputs: vec!["below_threshold".to_string()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: "combine_crossings".to_string(),
            op_type: "Or".to_string(),
            inputs: vec!["above_threshold".to_string(), "below_threshold".to_string()],
            outputs: vec!["crossings".to_string()],
            attributes: vec![],
        });
        self.add_bool_to_float(model, "crossings", "to_spikes", "spikes");

        Ok(())
    }

    /// Add delta encoder nodes.
    ///
    /// Magnitude gates the spike and the sign of the change carries its
    /// polarity, so the output is -1, 0 or +1 per sample.
    fn add_delta_nodes(&self, model: &mut OnnxModel, params: &EncoderParams) -> Result<()> {
        let threshold = params.thresholds.first().copied().unwrap_or(0.05);
        model.add_initializer(OnnxInitializer::float(
            "delta_threshold",
            vec![1],
            vec![threshold],
        ));

        let (prev, curr) = self.add_time_shift(model, "input", "delta");

        model.add_node(OnnxNode {
            name: "compute_delta".to_string(),
            op_type: "Sub".to_string(),
            inputs: vec![curr, prev],
            outputs: vec!["delta".to_string()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: "abs_delta_node".to_string(),
            op_type: "Abs".to_string(),
            inputs: vec!["delta".to_string()],
            outputs: vec!["abs_delta".to_string()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: "threshold_check".to_string(),
            op_type: "Greater".to_string(),
            inputs: vec!["abs_delta".to_string(), "delta_threshold".to_string()],
            outputs: vec!["spike_mask".to_string()],
            attributes: vec![],
        });
        self.add_bool_to_float(model, "spike_mask", "mask_to_float", "spike_mask_float");
        model.add_node(OnnxNode {
            name: "spike_sign".to_string(),
            op_type: "Sign".to_string(),
            inputs: vec!["delta".to_string()],
            outputs: vec!["spike_polarity".to_string()],
            attributes: vec![],
        });
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
    ///
    /// Contrast is the change in `log` intensity, so the input has to be made
    /// strictly positive first: `Log` of zero is -inf and of a negative is
    /// NaN, either of which would poison every downstream comparison.
    fn add_temporal_contrast_nodes(
        &self,
        model: &mut OnnxModel,
        params: &EncoderParams,
    ) -> Result<()> {
        let threshold = params.thresholds.first().copied().unwrap_or(0.1);
        model.add_initializer(OnnxInitializer::float(
            "tc_threshold",
            vec![1],
            vec![threshold],
        ));
        model.add_initializer(OnnxInitializer::float(
            "tc_epsilon",
            vec![1],
            vec![LOG_EPSILON],
        ));

        model.add_node(OnnxNode {
            name: "abs_input".to_string(),
            op_type: "Abs".to_string(),
            inputs: vec!["input".to_string()],
            outputs: vec!["abs_input_out".to_string()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: "offset_input".to_string(),
            op_type: "Add".to_string(),
            inputs: vec!["abs_input_out".to_string(), "tc_epsilon".to_string()],
            outputs: vec!["input_positive".to_string()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: "log_input_node".to_string(),
            op_type: "Log".to_string(),
            inputs: vec!["input_positive".to_string()],
            outputs: vec!["log_input".to_string()],
            attributes: vec![],
        });

        let (prev, curr) = self.add_time_shift(model, "log_input", "tc");

        model.add_node(OnnxNode {
            name: "log_diff".to_string(),
            op_type: "Sub".to_string(),
            inputs: vec![curr, prev],
            outputs: vec!["temporal_contrast".to_string()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: "abs_contrast".to_string(),
            op_type: "Abs".to_string(),
            inputs: vec!["temporal_contrast".to_string()],
            outputs: vec!["abs_tc".to_string()],
            attributes: vec![],
        });
        model.add_node(OnnxNode {
            name: "tc_threshold_check".to_string(),
            op_type: "Greater".to_string(),
            inputs: vec!["abs_tc".to_string(), "tc_threshold".to_string()],
            outputs: vec!["tc_mask".to_string()],
            attributes: vec![],
        });
        self.add_bool_to_float(model, "tc_mask", "tc_to_spikes", "spikes");

        Ok(())
    }

    /// Write the model as an ONNX protobuf `ModelProto`.
    fn write_onnx_model<W: Write>(&self, writer: &mut W, model: &OnnxModel) -> Result<()> {
        writer.write_all(&model.to_proto())?;
        Ok(())
    }
}

/// Stub exporter when ONNX feature is disabled.
#[cfg(not(feature = "onnx"))]
pub struct OnnxExporter {
    _metadata: ModelMetadata,
}

#[cfg(feature = "onnx")]
#[cfg(not(feature = "onnx"))]
impl OnnxExporter {
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            _metadata: metadata,
        }
    }

    pub fn export<E: ExportableEncoder>(
        &self,
        _path: impl AsRef<Path>,
        _encoder: &E,
    ) -> Result<()> {
        Err(ExportError::FeatureNotEnabled("onnx".to_string()))
    }
}

// =============================================================================
// Protobuf serialisation
//
// Field numbers below come from onnx.proto (ONNX IR). They are part of the
// wire format, so they are fixed -- changing one silently produces a file that
// decodes into the wrong fields rather than failing loudly.
//
//   https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
// =============================================================================

#[cfg(feature = "onnx")]
/// `TensorProto.DataType` values used here.
mod data_type {
    /// IEEE-754 single precision.
    pub const FLOAT: i32 = 1;
    /// Signed 64-bit integer.
    pub const INT64: i32 = 7;
}

#[cfg(feature = "onnx")]
/// Time is the last axis of the `[batch, channels, time]` layout these
/// encoders declare, so that is what gets sliced.
const TIME_AXIS: i64 = 2;

#[cfg(feature = "onnx")]
/// Added before `Log` so the argument is strictly positive. Small enough not
/// to shift the contrast of any real signal, large enough that `log` of it is
/// finite in f32.
const LOG_EPSILON: f32 = 1e-6;

#[cfg(feature = "onnx")]
/// `AttributeProto.AttributeType` values used here.
mod attr_type {
    /// A single `float`.
    pub const FLOAT: i32 = 1;
    /// A single `int64`.
    pub const INT: i32 = 2;
    /// A UTF-8 `string`.
    pub const STRING: i32 = 3;
}

#[cfg(feature = "onnx")]
impl OnnxModel {
    /// Encodes this model as an ONNX `ModelProto`.
    fn to_proto(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.int64(1, self.ir_version);
        w.string(2, &self.producer_name);
        w.string(3, &self.producer_version);
        w.string(4, &self.domain);
        w.int64(5, self.model_version);
        w.string(6, &self.doc_string);
        w.message(7, &self.graph.to_proto());
        for opset in &self.opset_import {
            let mut o = Writer::new();
            o.string(1, &opset.domain);
            o.int64(2, opset.version);
            w.message(8, &o);
        }
        w.finish()
    }
}

#[cfg(feature = "onnx")]
impl OnnxGraph {
    /// Encodes this graph as a `GraphProto`.
    fn to_proto(&self) -> Writer {
        let mut w = Writer::new();
        for node in &self.nodes {
            w.message(1, &node.to_proto());
        }
        w.string(2, &self.name);
        for init in &self.initializers {
            w.message(5, &init.to_proto());
        }
        for input in &self.inputs {
            w.message(11, &input.to_value_info());
        }
        for output in &self.outputs {
            w.message(12, &output.to_value_info());
        }
        w
    }
}

#[cfg(feature = "onnx")]
impl OnnxNode {
    /// Encodes this node as a `NodeProto`.
    fn to_proto(&self) -> Writer {
        let mut w = Writer::new();
        for input in &self.inputs {
            w.string(1, input);
        }
        for output in &self.outputs {
            w.string(2, output);
        }
        w.string(3, &self.name);
        w.string(4, &self.op_type);
        for attr in &self.attributes {
            w.message(5, &attr.to_proto());
        }
        w
    }
}

#[cfg(feature = "onnx")]
impl OnnxAttribute {
    /// Encodes this attribute as an `AttributeProto`.
    fn to_proto(&self) -> Writer {
        let mut w = Writer::new();
        w.string(1, &self.name);
        if let Some(f) = self.f {
            w.float(2, f);
            w.int32(20, attr_type::FLOAT);
        } else if let Some(s) = &self.s {
            w.bytes(4, s.as_bytes());
            w.int32(20, attr_type::STRING);
        } else {
            w.int64(3, self.i);
            w.int32(20, attr_type::INT);
        }
        w
    }
}

#[cfg(feature = "onnx")]
impl OnnxTensor {
    /// Encodes this tensor as a graph-level `ValueInfoProto`.
    ///
    /// A non-negative extent becomes `dim_value`. A negative one is this
    /// crate's marker for "unknown", and ONNX spells that as a symbolic
    /// `dim_param` -- a negative `dim_value` is not valid in a model file.
    fn to_value_info(&self) -> Writer {
        let mut shape = Writer::new();
        for (axis, &extent) in self.shape.iter().enumerate() {
            let mut dim = Writer::new();
            if extent >= 0 {
                dim.int64(1, extent);
            } else {
                dim.string(2, &Self::symbolic_dim(axis));
            }
            shape.message(1, &dim);
        }

        let mut tensor_type = Writer::new();
        tensor_type.int32(1, self.elem_type);
        tensor_type.message(2, &shape);

        let mut type_proto = Writer::new();
        type_proto.message(1, &tensor_type);

        let mut w = Writer::new();
        w.string(1, &self.name);
        w.message(2, &type_proto);
        w
    }

    /// Names a dynamic axis. Axis 0 is the batch by convention here; the rest
    /// get a positional name so two dynamic axes stay distinguishable.
    fn symbolic_dim(axis: usize) -> String {
        if axis == 0 {
            "batch".to_string()
        } else {
            format!("dim_{axis}")
        }
    }
}

#[cfg(feature = "onnx")]
impl OnnxInitializer {
    /// Encodes this initializer as a `TensorProto`.
    fn to_proto(&self) -> Writer {
        let mut w = Writer::new();
        w.packed_int64(1, &self.dims);
        w.int32(2, self.data_type);
        match self.data_type {
            data_type::FLOAT => w.packed_float(4, &self.float_data),
            data_type::INT64 => w.packed_int64(7, &self.int64_data),
            _ => {}
        }
        w.string(8, &self.name);
        w
    }
}

#[cfg(feature = "onnx")]
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

#[cfg(feature = "onnx")]
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

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxOpsetImport {
    domain: String,
    version: i64,
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxGraph {
    name: String,
    nodes: Vec<OnnxNode>,
    inputs: Vec<OnnxTensor>,
    outputs: Vec<OnnxTensor>,
    initializers: Vec<OnnxInitializer>,
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxNode {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: Vec<OnnxAttribute>,
}

#[cfg(feature = "onnx")]
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

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OnnxTensor {
    name: String,
    elem_type: i32,
    shape: Vec<i64>,
}

#[cfg(feature = "onnx")]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct OnnxInitializer {
    name: String,
    data_type: i32,
    dims: Vec<i64>,
    float_data: Vec<f32>,
    int64_data: Vec<i64>,
}

#[cfg(feature = "onnx")]
impl OnnxInitializer {
    /// A float tensor constant.
    fn float(name: &str, dims: Vec<i64>, data: Vec<f32>) -> Self {
        Self {
            name: name.to_string(),
            data_type: data_type::FLOAT,
            dims,
            float_data: data,
            int64_data: Vec::new(),
        }
    }

    /// An int64 tensor constant -- what `Slice` wants for starts/ends/axes.
    fn int64(name: &str, dims: Vec<i64>, data: Vec<i64>) -> Self {
        Self {
            name: name.to_string(),
            data_type: data_type::INT64,
            dims,
            float_data: Vec::new(),
            int64_data: data,
        }
    }
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;
    use crate::encoder_export::MockEncoder;

    fn model_for(enc: &MockEncoder) -> OnnxModel {
        let mut meta = ModelMetadata::new();
        meta.name = "test".to_string();
        let exporter = OnnxExporter::new(meta);
        exporter
            .build_onnx_model(&enc.get_params())
            .expect("graph builds")
    }

    /// Every node input must already exist: a graph input, an initializer, or
    /// the output of an earlier node. ONNX requires nodes in topological
    /// order, and a name nothing produces is a dangling edge -- both are
    /// rejected by onnx.checker, and both were present before.
    fn assert_graph_is_closed_and_sorted(model: &OnnxModel) {
        let mut available: Vec<String> = model
            .graph
            .inputs
            .iter()
            .map(|t| t.name.clone())
            .chain(model.graph.initializers.iter().map(|i| i.name.clone()))
            .collect();

        for node in &model.graph.nodes {
            for input in &node.inputs {
                assert!(
                    available.contains(input),
                    "node {:?} consumes {:?}, which no earlier node or initializer produces \
                     (available: {:?})",
                    node.name,
                    input,
                    available
                );
            }
            available.extend(node.outputs.iter().cloned());
        }

        for out in &model.graph.outputs {
            assert!(
                available.contains(&out.name),
                "graph declares output {:?} but no node produces it",
                out.name
            );
        }
    }

    #[test]
    fn level_crossing_graph_is_well_formed() {
        assert_graph_is_closed_and_sorted(&model_for(&MockEncoder::level_crossing(4, 1000.0, 0.1)));
    }

    #[test]
    fn delta_graph_is_well_formed() {
        assert_graph_is_closed_and_sorted(&model_for(&MockEncoder::delta(2, 500.0, 0.05, 8)));
    }

    #[test]
    fn temporal_contrast_graph_is_well_formed() {
        assert_graph_is_closed_and_sorted(&model_for(&MockEncoder::temporal_contrast(
            3, 2000.0, 0.2, 0.001,
        )));
    }

    /// Comparison operators yield bool, but the graph declares a float output,
    /// so a Cast has to sit between them.
    #[test]
    fn spikes_are_produced_as_float() {
        for model in [
            model_for(&MockEncoder::level_crossing(1, 1000.0, 0.1)),
            model_for(&MockEncoder::delta(1, 500.0, 0.05, 8)),
            model_for(&MockEncoder::temporal_contrast(1, 2000.0, 0.2, 0.001)),
        ] {
            let producer = model
                .graph
                .nodes
                .iter()
                .find(|n| n.outputs.iter().any(|o| o == "spikes"))
                .expect("something must produce spikes");
            assert!(
                matches!(producer.op_type.as_str(), "Cast" | "Mul"),
                "spikes came from {:?}, which yields bool rather than float",
                producer.op_type
            );
        }
    }

    /// A dynamic extent is a symbolic dim_param in ONNX. A negative dim_value
    /// is not valid in a model file, so the encoder must not emit one.
    #[test]
    fn dynamic_axes_become_symbolic_not_negative() {
        let model = model_for(&MockEncoder::delta(2, 500.0, 0.05, 8));
        let tensor = &model.graph.inputs[0];
        assert!(
            tensor.shape.iter().any(|&d| d < 0),
            "this test is only meaningful while the input has a dynamic axis"
        );

        let bytes = tensor.to_value_info().finish();
        // dim_param is field 2 of Dimension (wire type 2 -> tag 0x12) and
        // carries the name; "batch" must appear for the dynamic batch axis.
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("batch"), "no symbolic batch dim in {bytes:?}");
    }

    /// The serialised bytes must be a protobuf ModelProto, not JSON.
    #[test]
    fn serialises_as_protobuf_not_json() {
        let bytes = model_for(&MockEncoder::delta(2, 500.0, 0.05, 8)).to_proto();
        assert!(!bytes.is_empty());
        assert_ne!(bytes[0], b'{', "still emitting JSON");
        // ModelProto.ir_version is field 1, varint -> first tag byte is 0x08.
        assert_eq!(bytes[0], 0x08, "expected ir_version tag first");
        assert_eq!(bytes[1], 8, "ir_version should be 8");
        // producer_name is field 2, length-delimited -> tag 0x12.
        assert_eq!(bytes[2], 0x12);
    }

    #[test]
    fn export_writes_a_protobuf_file() {
        let dir = std::env::temp_dir().join("dpb_onnx_export_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("delta.onnx");

        let mut meta = ModelMetadata::new();
        meta.name = "delta".to_string();
        OnnxExporter::new(meta)
            .export(&path, &MockEncoder::delta(2, 500.0, 0.05, 8))
            .expect("export succeeds");

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes[0], 0x08, "file does not start with a ModelProto tag");
        let _ = std::fs::remove_file(&path);
    }
}
