use super::config::{ExportMetadata, LayerConfig, LayerType};
use super::weights::ModelWeights;
use serde::{Deserialize, Serialize};

/// ONNX export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxConfig {
    pub opset_version: i64,
    pub optimize: bool,
    pub quantize: Option<QuantizationConfig>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub dynamic_axes: Option<Vec<(String, Vec<usize>)>>,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            opset_version: 13,
            optimize: true,
            quantize: None,
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            dynamic_axes: None,
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub mode: QuantizationMode,
    pub weight_bits: u8,
    pub activation_bits: u8,
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMode {
    Dynamic,
    Static,
    QAT, // Quantization-aware training
}

/// Export result containing model bytes and metadata
#[derive(Debug)]
pub struct ExportResult {
    pub model_bytes: Vec<u8>,
    pub metadata: ExportMetadata,
    pub warnings: Vec<String>,
}

/// ONNX model exporter
pub struct OnnxExporter {
    config: OnnxConfig,
}

impl OnnxExporter {
    /// Create a new ONNX exporter with the given configuration
    pub fn new(config: OnnxConfig) -> Self {
        Self { config }
    }

    /// Create a new ONNX exporter with default configuration
    pub fn with_default_config() -> Self {
        Self {
            config: OnnxConfig::default(),
        }
    }

    /// Export SNN model to ONNX format
    /// Since SNNs have temporal dynamics, this exports an unrolled/approximated version
    pub fn export_snn_model(
        &self,
        weights: &ModelWeights,
        layer_configs: &[LayerConfig],
        input_shape: &[usize],
    ) -> Result<ExportResult, String> {
        let mut warnings = Vec::new();

        // Build ONNX graph
        let mut graph = OnnxGraph::new();

        // Add input tensor
        let input_tensor = OnnxTensor {
            name: self
                .config
                .input_names
                .first()
                .unwrap_or(&"input".to_string())
                .clone(),
            shape: input_shape.iter().map(|&x| x as i64).collect(),
            data_type: OnnxDataType::Float32,
            data: None,
        };
        graph.inputs.push(input_tensor);

        // Convert each layer to ONNX nodes
        let graph_output_name = self
            .config
            .output_names
            .first()
            .cloned()
            .unwrap_or_else(|| "output".to_string());

        let mut prev_output = self
            .config
            .input_names
            .first()
            .unwrap_or(&"input".to_string())
            .clone();

        for (i, (layer_config, layer_weights)) in
            layer_configs.iter().zip(weights.layers.iter()).enumerate()
        {
            let layer_name = format!("layer_{}", i);

            // Add weight initializer
            let weight_tensor = OnnxTensor {
                name: format!("{}_weight", layer_name),
                shape: layer_weights.shape.iter().map(|&x| x as i64).collect(),
                data_type: OnnxDataType::Float32,
                data: Some(self.weights_to_bytes(&layer_weights.weights)),
            };
            graph.initializers.push(weight_tensor);

            // Add bias if present
            if let Some(ref bias) = layer_weights.bias {
                let bias_tensor = OnnxTensor {
                    name: format!("{}_bias", layer_name),
                    shape: vec![bias.len() as i64],
                    data_type: OnnxDataType::Float32,
                    data: Some(self.weights_to_bytes(bias)),
                };
                graph.initializers.push(bias_tensor);
            }

            // Create node based on layer type
            // The last layer must emit the graph's declared output, or nothing
            // produces it and the graph is disconnected -- which is exactly
            // what onnx.checker reports as "Graph output 'output' is not an
            // output of any node in graph".
            let node_output = if i + 1 == layer_configs.len() {
                graph_output_name.clone()
            } else {
                format!("{}_output", layer_name)
            };
            let node = self.create_layer_node(
                layer_config,
                &layer_name,
                &prev_output,
                &node_output,
                layer_weights.bias.is_some(),
            )?;

            graph.nodes.push(node);
            prev_output = node_output;

            // Warn about spiking behavior
            if layer_config.layer_type.to_string().contains("Spiking") {
                warnings.push(format!(
                    "Layer {} is a spiking layer - temporal dynamics are approximated in ONNX",
                    layer_name
                ));
            }
        }

        // Add output tensor, shaped from the last layer rather than left as a
        // single dynamic dimension: -1 for the batch, then the layer's width.
        let output_width = layer_configs
            .last()
            .map(|l| l.output_size as i64)
            .unwrap_or(-1);
        let output_tensor = OnnxTensor {
            name: graph_output_name.clone(),
            shape: vec![-1, output_width],
            data_type: OnnxDataType::Float32,
            data: None,
        };
        graph.outputs.push(output_tensor);

        // Serialize graph to ONNX format (simplified - in reality would use protobuf)
        let model_bytes = self.serialize_graph(&graph)?;

        // Apply optimizations if enabled
        let optimized_bytes = if self.config.optimize {
            self.optimize_model(&model_bytes)?
        } else {
            model_bytes
        };

        // Apply quantization if configured
        let final_bytes = if let Some(ref quant_config) = self.config.quantize {
            self.quantize_model(&optimized_bytes, quant_config)?
        } else {
            optimized_bytes
        };

        // Create metadata
        let metadata = ExportMetadata::new(&weights.metadata.model_name, "onnx")
            .with_shapes(input_shape.to_vec(), vec![]);

        Ok(ExportResult {
            model_bytes: final_bytes,
            metadata,
            warnings,
        })
    }

    /// Export encoder to ONNX
    pub fn export_encoder(
        &self,
        encoder_type: &str,
        params: &serde_json::Value,
        input_shape: &[usize],
    ) -> Result<ExportResult, String> {
        let mut graph = OnnxGraph::new();
        let mut warnings = Vec::new();

        // Add input
        let input_tensor = OnnxTensor {
            name: "encoder_input".to_string(),
            shape: input_shape.iter().map(|&x| x as i64).collect(),
            data_type: OnnxDataType::Float32,
            data: None,
        };
        graph.inputs.push(input_tensor);

        // Create encoder-specific nodes
        match encoder_type {
            "rate" => {
                warnings.push("Rate encoder exported as identity function".to_string());
                // Rate encoding can be approximated as scaling
                let node = OnnxNode {
                    op_type: "Identity".to_string(),
                    name: "rate_encoder".to_string(),
                    inputs: vec!["encoder_input".to_string()],
                    outputs: vec!["encoder_output".to_string()],
                    attributes: vec![],
                };
                graph.nodes.push(node);
            }
            "latency" => {
                warnings.push("Latency encoder not fully supported in ONNX".to_string());
                return Err(
                    "Latency encoder requires temporal dynamics not supported in ONNX".to_string(),
                );
            }
            "population" => {
                warnings.push("Population encoder exported with approximation".to_string());
                // Add nodes for population encoding
                self.add_population_encoding_nodes(&mut graph, params)?;
            }
            _ => {
                return Err(format!("Unknown encoder type: {}", encoder_type));
            }
        }

        // Add output
        let output_tensor = OnnxTensor {
            name: "encoder_output".to_string(),
            shape: vec![-1],
            data_type: OnnxDataType::Float32,
            data: None,
        };
        graph.outputs.push(output_tensor);

        let model_bytes = self.serialize_graph(&graph)?;
        let metadata = ExportMetadata::new(&format!("{}_encoder", encoder_type), "onnx")
            .with_shapes(input_shape.to_vec(), vec![]);

        Ok(ExportResult {
            model_bytes,
            metadata,
            warnings,
        })
    }

    /// Export decoder to ONNX
    pub fn export_decoder(
        &self,
        decoder_type: &str,
        _params: &serde_json::Value,
        input_shape: &[usize],
    ) -> Result<ExportResult, String> {
        let mut graph = OnnxGraph::new();
        let warnings = Vec::new();

        // Add input
        let input_tensor = OnnxTensor {
            name: "decoder_input".to_string(),
            shape: input_shape.iter().map(|&x| x as i64).collect(),
            data_type: OnnxDataType::Float32,
            data: None,
        };
        graph.inputs.push(input_tensor);

        // Create decoder-specific nodes
        match decoder_type {
            "spike_count" => {
                // Sum over time dimension (approximation)
                let node = OnnxNode {
                    op_type: "ReduceSum".to_string(),
                    name: "spike_count_decoder".to_string(),
                    inputs: vec!["decoder_input".to_string()],
                    outputs: vec!["decoder_output".to_string()],
                    attributes: vec![
                        ("axes".to_string(), OnnxAttribute::Ints(vec![1])),
                        ("keepdims".to_string(), OnnxAttribute::Int(0)),
                    ],
                };
                graph.nodes.push(node);
            }
            "rate" => {
                // Average over time dimension
                let node = OnnxNode {
                    op_type: "ReduceMean".to_string(),
                    name: "rate_decoder".to_string(),
                    inputs: vec!["decoder_input".to_string()],
                    outputs: vec!["decoder_output".to_string()],
                    attributes: vec![
                        ("axes".to_string(), OnnxAttribute::Ints(vec![1])),
                        ("keepdims".to_string(), OnnxAttribute::Int(0)),
                    ],
                };
                graph.nodes.push(node);
            }
            _ => {
                return Err(format!("Unknown decoder type: {}", decoder_type));
            }
        }

        // Add output
        let output_tensor = OnnxTensor {
            name: "decoder_output".to_string(),
            shape: vec![-1],
            data_type: OnnxDataType::Float32,
            data: None,
        };
        graph.outputs.push(output_tensor);

        let model_bytes = self.serialize_graph(&graph)?;
        let metadata = ExportMetadata::new(&format!("{}_decoder", decoder_type), "onnx")
            .with_shapes(input_shape.to_vec(), vec![]);

        Ok(ExportResult {
            model_bytes,
            metadata,
            warnings,
        })
    }

    /// Validate exported model
    /// Structural check on exported ONNX bytes.
    ///
    /// ONNX has no magic number -- a `.onnx` file is a bare protobuf message --
    /// so the check is that the bytes decode as one and carry the fields a
    /// ModelProto must have. The previous body claimed, in a comment, to check
    /// a magic number and in fact only checked that four bytes were present:
    /// `b"junk"` passed, and so did any other four bytes.
    pub fn validate(&self, model_bytes: &[u8]) -> Result<(), String> {
        if model_bytes.is_empty() {
            return Err("Model bytes are empty".to_string());
        }

        let mut offset = 0usize;
        let mut seen_ir_version = false;
        let mut seen_graph = false;

        while offset < model_bytes.len() {
            let (tag, used) = read_varint(model_bytes, offset)
                .ok_or_else(|| format!("truncated field tag at byte {offset}"))?;
            offset += used;

            let field = tag >> 3;
            let wire = tag & 0x7;

            match wire {
                // varint
                0 => {
                    let (_, used) = read_varint(model_bytes, offset)
                        .ok_or_else(|| format!("truncated varint at byte {offset}"))?;
                    offset += used;
                    if field == 1 {
                        seen_ir_version = true;
                    }
                }
                // 64-bit
                1 => offset = offset.checked_add(8).ok_or("length overflow")?,
                // length-delimited
                2 => {
                    let (len, used) = read_varint(model_bytes, offset)
                        .ok_or_else(|| format!("truncated length at byte {offset}"))?;
                    offset += used;
                    let end = offset
                        .checked_add(len as usize)
                        .ok_or("field length overflows the buffer")?;
                    if end > model_bytes.len() {
                        return Err(format!(
                            "field {field} claims {len} bytes but only {} remain",
                            model_bytes.len() - offset
                        ));
                    }
                    // ModelProto field 7 is the graph.
                    if field == 7 {
                        seen_graph = true;
                    }
                    offset = end;
                }
                // 32-bit
                5 => offset = offset.checked_add(4).ok_or("length overflow")?,
                other => return Err(format!("unknown wire type {other} for field {field}")),
            }

            if offset > model_bytes.len() {
                return Err("a field ran past the end of the buffer".to_string());
            }
        }

        if !seen_ir_version {
            return Err("no ir_version field: this is not an ONNX ModelProto".to_string());
        }
        if !seen_graph {
            return Err("no graph field: the model contains no graph".to_string());
        }
        Ok(())
    }

    // Helper methods

    fn weights_to_bytes(&self, weights: &[f64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(weights.len() * 4);
        for &w in weights {
            bytes.extend_from_slice(&(w as f32).to_le_bytes());
        }
        bytes
    }

    fn create_layer_node(
        &self,
        layer_config: &LayerConfig,
        name: &str,
        input: &str,
        output: &str,
        has_bias: bool,
    ) -> Result<OnnxNode, String> {
        // Match the layer type itself, not its name.
        //
        // This used to stringify the type and test the result for substrings,
        // then emit attributes marked "Default": every exported convolution
        // claimed a 3x3 kernel with stride 1 and padding 1, and every recurrent
        // layer claimed hidden_size 128, whatever the model actually had. The
        // variants carry those values -- `Display` is what drops them, with
        // `{ .. }` -- so a 7x1 stride-2 convolution exported as a 3x3 stride-1
        // one and the file described a different network than the one saved.
        let (op_type, attributes) = match &layer_config.layer_type {
            LayerType::SpikingLinear => (
                "Gemm".to_string(),
                vec![
                    ("alpha".to_string(), OnnxAttribute::Float(1.0)),
                    (
                        "beta".to_string(),
                        OnnxAttribute::Float(if has_bias { 1.0 } else { 0.0 }),
                    ),
                    ("transB".to_string(), OnnxAttribute::Int(1)),
                ],
            ),
            LayerType::SpikingConv1d {
                kernel_size,
                stride,
                padding,
            } => (
                "Conv".to_string(),
                vec![
                    (
                        "kernel_shape".to_string(),
                        OnnxAttribute::Ints(vec![*kernel_size as i64]),
                    ),
                    (
                        "strides".to_string(),
                        OnnxAttribute::Ints(vec![*stride as i64]),
                    ),
                    // ONNX `pads` lists the start padding for every spatial
                    // axis, then the end padding for every axis.
                    (
                        "pads".to_string(),
                        OnnxAttribute::Ints(vec![*padding as i64, *padding as i64]),
                    ),
                ],
            ),
            LayerType::SpikingConv2d {
                kernel_size,
                stride,
                padding,
            } => (
                "Conv".to_string(),
                vec![
                    (
                        "kernel_shape".to_string(),
                        OnnxAttribute::Ints(vec![kernel_size.0 as i64, kernel_size.1 as i64]),
                    ),
                    (
                        "strides".to_string(),
                        OnnxAttribute::Ints(vec![stride.0 as i64, stride.1 as i64]),
                    ),
                    (
                        "pads".to_string(),
                        OnnxAttribute::Ints(vec![
                            padding.0 as i64,
                            padding.1 as i64,
                            padding.0 as i64,
                            padding.1 as i64,
                        ]),
                    ),
                ],
            ),
            LayerType::SpikingRecurrent => (
                "LSTM".to_string(),
                vec![(
                    "hidden_size".to_string(),
                    // The layer's own output width, not a constant 128.
                    OnnxAttribute::Int(layer_config.output_size as i64),
                )],
            ),
            // Encoders, decoders, batch norm and dropout have no ONNX operator
            // here; passing the tensor through unchanged is at least honest
            // about that.
            _ => ("Identity".to_string(), vec![]),
        };

        let mut inputs = vec![input.to_string(), format!("{}_weight", name)];
        if has_bias {
            inputs.push(format!("{}_bias", name));
        }

        Ok(OnnxNode {
            op_type,
            name: name.to_string(),
            inputs,
            outputs: vec![output.to_string()],
            attributes,
        })
    }

    fn add_population_encoding_nodes(
        &self,
        graph: &mut OnnxGraph,
        _params: &serde_json::Value,
    ) -> Result<(), String> {
        // Add Gaussian receptive field nodes (simplified)
        let node = OnnxNode {
            op_type: "Mul".to_string(),
            name: "population_scale".to_string(),
            inputs: vec!["encoder_input".to_string(), "scale_factor".to_string()],
            outputs: vec!["encoder_output".to_string()],
            attributes: vec![],
        };
        graph.nodes.push(node);

        // Add scale factor initializer
        let scale_tensor = OnnxTensor {
            name: "scale_factor".to_string(),
            shape: vec![1],
            data_type: OnnxDataType::Float32,
            data: Some(vec![0, 0, 128, 63]), // 1.0 in little-endian f32
        };
        graph.initializers.push(scale_tensor);

        Ok(())
    }

    /// Serialise the graph as an ONNX `ModelProto`.
    ///
    /// This used to write JSON, under a comment describing that as a
    /// placeholder -- so every file this exporter produced was named `.onnx`
    /// and was not ONNX. Nothing detected it, because `validate` checked only
    /// that four bytes were present.
    ///
    /// ONNX is a bare protobuf message, so it is written directly here rather
    /// than pulling in a code generator for the handful of messages involved.
    /// Field numbers are from onnx.proto.
    fn serialize_graph(&self, graph: &OnnxGraph) -> Result<Vec<u8>, String> {
        let mut graph_buf = Vec::new();
        for node in &graph.nodes {
            pb_message(&mut graph_buf, 1, &encode_node(node));
        }
        pb_string(&mut graph_buf, 2, "dpb_graph");
        for tensor in &graph.initializers {
            pb_message(&mut graph_buf, 5, &encode_tensor(tensor));
        }
        for tensor in &graph.inputs {
            pb_message(&mut graph_buf, 11, &encode_value_info(tensor));
        }
        for tensor in &graph.outputs {
            pb_message(&mut graph_buf, 12, &encode_value_info(tensor));
        }

        let mut model = Vec::new();
        // ir_version 8 is the ONNX 1.13 IR, which covers opset 13-19.
        const IR_VERSION: u64 = 8;
        pb_varint_field(&mut model, 1, IR_VERSION);
        pb_string(&mut model, 2, "dpb-snn");
        pb_string(&mut model, 3, env!("CARGO_PKG_VERSION"));
        pb_message(&mut model, 7, &graph_buf);

        // opset_import: domain "" at the configured version.
        let mut opset = Vec::new();
        pb_string(&mut opset, 1, "");
        pb_varint_field(&mut opset, 2, self.config.opset_version as u64);
        pb_message(&mut model, 8, &opset);

        Ok(model)
    }

    fn optimize_model(&self, model_bytes: &[u8]) -> Result<Vec<u8>, String> {
        // Placeholder for model optimization
        // In a real implementation, would apply graph optimizations like:
        // - Constant folding
        // - Dead code elimination
        // - Operator fusion
        Ok(model_bytes.to_vec())
    }

    fn quantize_model(
        &self,
        model_bytes: &[u8],
        _config: &QuantizationConfig,
    ) -> Result<Vec<u8>, String> {
        // Placeholder for quantization
        // In a real implementation, would quantize weights and activations
        Ok(model_bytes.to_vec())
    }
}

/// ONNX graph builder (simplified representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxGraph {
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<OnnxTensor>,
    pub outputs: Vec<OnnxTensor>,
    pub initializers: Vec<OnnxTensor>,
}

impl OnnxGraph {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxNode {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<(String, OnnxAttribute)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: OnnxDataType,
    pub data: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OnnxDataType {
    Float32,
    Float16,
    Int32,
    Int8,
    Uint8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_config_default() {
        let config = OnnxConfig::default();
        assert_eq!(config.opset_version, 13);
        assert!(config.optimize);
        assert!(config.quantize.is_none());
        assert_eq!(config.input_names, vec!["input".to_string()]);
        assert_eq!(config.output_names, vec!["output".to_string()]);
    }

    #[test]
    fn test_onnx_exporter_creation() {
        let exporter = OnnxExporter::with_default_config();
        assert_eq!(exporter.config.opset_version, 13);

        let custom_config = OnnxConfig {
            opset_version: 14,
            ..Default::default()
        };
        let exporter2 = OnnxExporter::new(custom_config);
        assert_eq!(exporter2.config.opset_version, 14);
    }

    #[test]
    fn test_weights_to_bytes() {
        let exporter = OnnxExporter::with_default_config();
        let weights = vec![1.0, 2.0, 3.0];
        let bytes = exporter.weights_to_bytes(&weights);

        assert_eq!(bytes.len(), 12); // 3 floats * 4 bytes each
    }

    #[test]
    fn test_onnx_graph_creation() {
        let graph = OnnxGraph::new();
        assert!(graph.nodes.is_empty());
        assert!(graph.inputs.is_empty());
        assert!(graph.outputs.is_empty());
        assert!(graph.initializers.is_empty());
    }

    #[test]
    fn test_validate_empty_model() {
        let exporter = OnnxExporter::with_default_config();
        let result = exporter.validate(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_validate_too_short() {
        let exporter = OnnxExporter::with_default_config();
        assert!(exporter.validate(&[1, 2, 3]).is_err());
    }

    /// Five arbitrary bytes are not an ONNX model.
    ///
    /// This test used to assert the opposite -- that `[1, 2, 3, 4, 5]`
    /// validates -- which was true only because `validate` checked the length
    /// and nothing else. It was pinning the defect rather than the behaviour.
    #[test]
    fn test_validate_rejects_arbitrary_bytes() {
        let exporter = OnnxExporter::with_default_config();
        assert!(exporter.validate(&[1, 2, 3, 4, 5]).is_err());
    }

    // ---- exported convolution attributes -------------------------------

    fn attr_ints(node: &OnnxNode, key: &str) -> Vec<i64> {
        node.attributes
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| match v {
                OnnxAttribute::Ints(vals) => vals.clone(),
                other => panic!("{key} is {other:?}, not Ints"),
            })
            .unwrap_or_else(|| panic!("node has no {key} attribute"))
    }

    /// A convolution must export its own kernel, stride and padding.
    ///
    /// The exporter used to stringify the layer type and match substrings, then
    /// emit hardcoded 3x3 / stride 1 / pad 1 attributes marked "Default" -- so
    /// the file described a different network than the one being saved.
    #[test]
    fn conv2d_exports_its_real_geometry() {
        let exporter = OnnxExporter::new(OnnxConfig::default());
        let layer = LayerConfig::new(
            "c1".to_string(),
            LayerType::SpikingConv2d {
                kernel_size: (7, 5),
                stride: (2, 3),
                padding: (3, 2),
            },
            8,
            16,
        );

        let node = exporter
            .create_layer_node(&layer, "c1", "in", "out", true)
            .unwrap();

        assert_eq!(node.op_type, "Conv");
        assert_eq!(attr_ints(&node, "kernel_shape"), vec![7, 5]);
        assert_eq!(attr_ints(&node, "strides"), vec![2, 3]);
        // ONNX orders pads as all starts then all ends.
        assert_eq!(attr_ints(&node, "pads"), vec![3, 2, 3, 2]);
    }

    #[test]
    fn conv1d_exports_its_real_geometry() {
        let exporter = OnnxExporter::new(OnnxConfig::default());
        let layer = LayerConfig::new(
            "c".to_string(),
            LayerType::SpikingConv1d {
                kernel_size: 9,
                stride: 4,
                padding: 2,
            },
            4,
            6,
        );

        let node = exporter
            .create_layer_node(&layer, "c", "in", "out", false)
            .unwrap();
        assert_eq!(attr_ints(&node, "kernel_shape"), vec![9]);
        assert_eq!(attr_ints(&node, "strides"), vec![4]);
        assert_eq!(attr_ints(&node, "pads"), vec![2, 2]);
    }

    /// Two convolutions that differ must export differently. Under the old
    /// code every convolution exported identically, so this is the check that
    /// would have caught it regardless of which constants were chosen.
    #[test]
    fn different_convolutions_export_differently() {
        let exporter = OnnxExporter::new(OnnxConfig::default());
        let make = |k: (usize, usize), s: (usize, usize), p: (usize, usize)| {
            let layer = LayerConfig::new(
                "c".to_string(),
                LayerType::SpikingConv2d {
                    kernel_size: k,
                    stride: s,
                    padding: p,
                },
                8,
                16,
            );
            exporter
                .create_layer_node(&layer, "c", "in", "out", true)
                .unwrap()
                .attributes
        };

        // OnnxAttribute has no PartialEq, so compare the extracted integers.
        let ints = |attrs: Vec<(String, OnnxAttribute)>| -> Vec<Vec<i64>> {
            ["kernel_shape", "strides", "pads"]
                .iter()
                .map(|key| {
                    attrs
                        .iter()
                        .find(|(k, _)| k == key)
                        .map(|(_, v)| match v {
                            OnnxAttribute::Ints(vals) => vals.clone(),
                            other => panic!("{key} is {other:?}"),
                        })
                        .unwrap_or_default()
                })
                .collect()
        };

        assert_ne!(
            ints(make((3, 3), (1, 1), (1, 1))),
            ints(make((5, 5), (2, 2), (0, 0))),
            "two different convolutions produced identical ONNX attributes"
        );
    }

    /// A recurrent layer's hidden size is its own width, not a constant.
    #[test]
    fn recurrent_exports_its_own_hidden_size() {
        let exporter = OnnxExporter::new(OnnxConfig::default());
        let layer = LayerConfig::new("r".to_string(), LayerType::SpikingRecurrent, 10, 37);
        let node = exporter
            .create_layer_node(&layer, "r", "in", "out", true)
            .unwrap();
        assert_eq!(node.op_type, "LSTM");
        let hidden = node
            .attributes
            .iter()
            .find(|(k, _)| k == "hidden_size")
            .map(|(_, v)| match v {
                OnnxAttribute::Int(i) => *i,
                other => panic!("hidden_size is {other:?}"),
            })
            .expect("no hidden_size attribute");
        assert_eq!(hidden, 37, "hidden size was not taken from the layer");
    }
}

/// Reads one protobuf varint, returning its value and the bytes consumed.
///
/// Returns None when the buffer ends mid-varint, which is what a truncated
/// file looks like.
fn read_varint(bytes: &[u8], mut offset: usize) -> Option<(u64, usize)> {
    let start = offset;
    let mut value = 0u64;
    let mut shift = 0u32;
    while offset < bytes.len() {
        let byte = bytes[offset];
        offset += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some((value, offset - start));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod validate_tests {
    use super::*;

    /// Arbitrary bytes are not an ONNX model.
    ///
    /// `validate` used to claim, in a comment, to check a magic number, and in
    /// fact checked only that four bytes were present -- so `b"junk"` passed.
    #[test]
    fn arbitrary_bytes_are_rejected() {
        let exporter = OnnxExporter::new(OnnxConfig::default());
        for bytes in [
            b"junk".to_vec(),
            vec![0u8; 32],
            vec![0xff; 16],
            b"not an onnx file at all".to_vec(),
        ] {
            assert!(
                exporter.validate(&bytes).is_err(),
                "accepted {} bytes of non-ONNX data",
                bytes.len()
            );
        }
        assert!(exporter.validate(&[]).is_err(), "accepted an empty file");
    }

    /// A truncated field is reported rather than read past.
    #[test]
    fn truncated_fields_are_rejected() {
        let exporter = OnnxExporter::new(OnnxConfig::default());
        // field 7 (graph), wire type 2, claiming 200 bytes that are not there.
        let bytes = vec![(7 << 3) | 2, 200, 1, 2, 3];
        assert!(exporter.validate(&bytes).is_err());
    }

    /// A model missing its graph is reported.
    #[test]
    fn a_model_without_a_graph_is_rejected() {
        let exporter = OnnxExporter::new(OnnxConfig::default());
        // Only ir_version (field 1, varint).
        let bytes = vec![1 << 3, 9];
        let err = exporter.validate(&bytes).unwrap_err();
        assert!(err.contains("graph"), "unexpected error: {err}");
    }

    /// The graph must be connected: the declared output has to be produced by
    /// a node.
    ///
    /// It was not. The last layer emitted `layer_N_output` while the graph
    /// declared an output named `output`, so nothing produced it -- which
    /// `onnx.checker` reports as "Graph output 'output' is not an output of any
    /// node in graph". Nothing here noticed, because the file was JSON and
    /// `validate` checked only its length.
    #[test]
    fn the_graph_output_is_produced_by_a_node() {
        use crate::export::config::{LayerConfig, LayerType};
        use crate::export::weights::{LayerWeights, ModelWeights, WeightMetadata};

        let layers = vec![
            LayerConfig::new("a".into(), LayerType::SpikingLinear, 4, 8),
            LayerConfig::new("b".into(), LayerType::SpikingLinear, 8, 3),
        ];
        let weights = ModelWeights {
            layers: vec![
                LayerWeights {
                    name: "a".into(),
                    layer_type: "SpikingLinear".into(),
                    weights: vec![0.1; 32],
                    shape: vec![8, 4],
                    bias: Some(vec![0.0; 8]),
                },
                LayerWeights {
                    name: "b".into(),
                    layer_type: "SpikingLinear".into(),
                    weights: vec![0.1; 24],
                    shape: vec![3, 8],
                    bias: Some(vec![0.0; 3]),
                },
            ],
            metadata: WeightMetadata {
                model_name: "t".into(),
                version: "1".into(),
                created_at: "now".into(),
                total_params: 0,
                checksum: String::new(),
            },
        };

        let exporter = OnnxExporter::new(OnnxConfig::default());
        let result = exporter
            .export_snn_model(&weights, &layers, &[1, 4])
            .expect("export");

        // The bytes must be a protobuf ModelProto, not JSON.
        assert!(
            exporter.validate(&result.model_bytes).is_ok(),
            "the exporter produced bytes its own validator rejects"
        );
        assert_ne!(
            result.model_bytes.first(),
            Some(&b'{'),
            "the model was serialised as JSON, not ONNX"
        );
    }

    /// Bytes this exporter produced must validate.
    #[test]
    fn exported_bytes_validate() {
        // ir_version, then an empty graph.
        let bytes = vec![1 << 3, 9, (7 << 3) | 2, 0];
        let exporter = OnnxExporter::new(OnnxConfig::default());
        exporter
            .validate(&bytes)
            .expect("a well-formed ModelProto must validate");
    }
}

// ---------------------------------------------------------------------------
// Minimal protobuf encoding for ONNX
// ---------------------------------------------------------------------------

fn pb_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

fn pb_tag(buf: &mut Vec<u8>, field: u32, wire: u32) {
    pb_varint(buf, u64::from(field << 3 | wire));
}

fn pb_varint_field(buf: &mut Vec<u8>, field: u32, value: u64) {
    pb_tag(buf, field, 0);
    pb_varint(buf, value);
}

fn pb_int64(buf: &mut Vec<u8>, field: u32, value: i64) {
    pb_tag(buf, field, 0);
    pb_varint(buf, value as u64);
}

fn pb_float(buf: &mut Vec<u8>, field: u32, value: f32) {
    pb_tag(buf, field, 5);
    buf.extend_from_slice(&value.to_le_bytes());
}

fn pb_bytes(buf: &mut Vec<u8>, field: u32, value: &[u8]) {
    pb_tag(buf, field, 2);
    pb_varint(buf, value.len() as u64);
    buf.extend_from_slice(value);
}

fn pb_string(buf: &mut Vec<u8>, field: u32, value: &str) {
    pb_bytes(buf, field, value.as_bytes());
}

fn pb_message(buf: &mut Vec<u8>, field: u32, body: &[u8]) {
    pb_bytes(buf, field, body);
}

/// ONNX `TensorProto.DataType` codes.
fn onnx_data_type(dtype: OnnxDataType) -> i64 {
    match dtype {
        OnnxDataType::Float32 => 1,
        OnnxDataType::Float16 => 10,
        OnnxDataType::Int32 => 6,
        OnnxDataType::Int8 => 3,
        OnnxDataType::Uint8 => 2,
    }
}

fn encode_attribute(name: &str, attr: &OnnxAttribute) -> Vec<u8> {
    let mut buf = Vec::new();
    pb_string(&mut buf, 1, name);
    // AttributeProto.AttributeType: FLOAT 1, INT 2, STRING 3, FLOATS 6, INTS 7
    match attr {
        OnnxAttribute::Float(v) => {
            pb_float(&mut buf, 2, *v as f32);
            pb_varint_field(&mut buf, 20, 1);
        }
        OnnxAttribute::Int(v) => {
            pb_int64(&mut buf, 3, *v);
            pb_varint_field(&mut buf, 20, 2);
        }
        OnnxAttribute::String(v) => {
            pb_bytes(&mut buf, 4, v.as_bytes());
            pb_varint_field(&mut buf, 20, 3);
        }
        OnnxAttribute::Floats(vs) => {
            for v in vs {
                pb_float(&mut buf, 7, *v as f32);
            }
            pb_varint_field(&mut buf, 20, 6);
        }
        OnnxAttribute::Ints(vs) => {
            for v in vs {
                pb_int64(&mut buf, 8, *v);
            }
            pb_varint_field(&mut buf, 20, 7);
        }
    }
    buf
}

fn encode_node(node: &OnnxNode) -> Vec<u8> {
    let mut buf = Vec::new();
    for input in &node.inputs {
        pb_string(&mut buf, 1, input);
    }
    for output in &node.outputs {
        pb_string(&mut buf, 2, output);
    }
    pb_string(&mut buf, 3, &node.name);
    pb_string(&mut buf, 4, &node.op_type);
    for (name, attr) in &node.attributes {
        pb_message(&mut buf, 5, &encode_attribute(name, attr));
    }
    buf
}

fn encode_tensor(tensor: &OnnxTensor) -> Vec<u8> {
    let mut buf = Vec::new();
    for dim in &tensor.shape {
        pb_int64(&mut buf, 1, *dim);
    }
    pb_int64(&mut buf, 2, onnx_data_type(tensor.data_type));
    pb_string(&mut buf, 8, &tensor.name);
    if let Some(ref data) = tensor.data {
        pb_bytes(&mut buf, 9, data);
    }
    buf
}

/// A `ValueInfoProto`: a name and a tensor type with its shape.
fn encode_value_info(tensor: &OnnxTensor) -> Vec<u8> {
    // TensorShapeProto.Dimension: dim_value is field 1.
    let mut shape = Vec::new();
    for dim in &tensor.shape {
        let mut d = Vec::new();
        pb_int64(&mut d, 1, *dim);
        pb_message(&mut shape, 1, &d);
    }

    // TypeProto.Tensor: elem_type 1, shape 2.
    let mut tensor_type = Vec::new();
    pb_int64(&mut tensor_type, 1, onnx_data_type(tensor.data_type));
    pb_message(&mut tensor_type, 2, &shape);

    // TypeProto: tensor_type is field 1.
    let mut type_proto = Vec::new();
    pb_message(&mut type_proto, 1, &tensor_type);

    let mut buf = Vec::new();
    pb_string(&mut buf, 1, &tensor.name);
    pb_message(&mut buf, 2, &type_proto);
    buf
}
