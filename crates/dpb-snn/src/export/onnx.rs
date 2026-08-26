use serde::{Serialize, Deserialize};
use super::config::{LayerConfig, ExportMetadata};
use super::weights::ModelWeights;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationMode {
    Dynamic,
    Static,
    QAT,  // Quantization-aware training
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
            name: self.config.input_names.first()
                .unwrap_or(&"input".to_string())
                .clone(),
            shape: input_shape.iter().map(|&x| x as i64).collect(),
            data_type: OnnxDataType::Float32,
            data: None,
        };
        graph.inputs.push(input_tensor);
        
        // Convert each layer to ONNX nodes
        let mut prev_output = self.config.input_names.first()
            .unwrap_or(&"input".to_string())
            .clone();
            
        for (i, (layer_config, layer_weights)) in layer_configs.iter()
            .zip(weights.layers.iter())
            .enumerate() 
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
            let node_output = format!("{}_output", layer_name);
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
        
        // Add output tensor
        let output_tensor = OnnxTensor {
            name: self.config.output_names.first()
                .unwrap_or(&"output".to_string())
                .clone(),
            shape: vec![-1],  // Dynamic shape
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
                return Err("Latency encoder requires temporal dynamics not supported in ONNX".to_string());
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
        params: &serde_json::Value,
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
    pub fn validate(&self, model_bytes: &[u8]) -> Result<(), String> {
        // Basic validation checks
        if model_bytes.is_empty() {
            return Err("Model bytes are empty".to_string());
        }
        
        // Check for ONNX magic number (simplified)
        if model_bytes.len() < 4 {
            return Err("Model bytes too short to be valid ONNX".to_string());
        }
        
        // More validation would happen here in a real implementation
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
        let layer_type_str = layer_config.layer_type.to_string();
        
        let (op_type, attributes) = if layer_type_str.contains("Linear") {
            ("Gemm".to_string(), vec![
                ("alpha".to_string(), OnnxAttribute::Float(1.0)),
                ("beta".to_string(), OnnxAttribute::Float(if has_bias { 1.0 } else { 0.0 })),
                ("transB".to_string(), OnnxAttribute::Int(1)),
            ])
        } else if layer_type_str.contains("Conv1d") {
            ("Conv".to_string(), vec![
                ("kernel_shape".to_string(), OnnxAttribute::Ints(vec![3])),  // Default
                ("strides".to_string(), OnnxAttribute::Ints(vec![1])),
                ("pads".to_string(), OnnxAttribute::Ints(vec![1, 1])),
            ])
        } else if layer_type_str.contains("Conv2d") {
            ("Conv".to_string(), vec![
                ("kernel_shape".to_string(), OnnxAttribute::Ints(vec![3, 3])),  // Default
                ("strides".to_string(), OnnxAttribute::Ints(vec![1, 1])),
                ("pads".to_string(), OnnxAttribute::Ints(vec![1, 1, 1, 1])),
            ])
        } else if layer_type_str.contains("Recurrent") {
            ("LSTM".to_string(), vec![
                ("hidden_size".to_string(), OnnxAttribute::Int(128)),  // Default
            ])
        } else {
            // Default to identity for unknown types
            ("Identity".to_string(), vec![])
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
    
    fn serialize_graph(&self, graph: &OnnxGraph) -> Result<Vec<u8>, String> {
        // In a real implementation, this would use protobuf to serialize to ONNX format
        // For now, we'll serialize to JSON as a placeholder
        serde_json::to_vec(graph)
            .map_err(|e| format!("Failed to serialize graph: {}", e))
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
        let result = exporter.validate(&[1, 2, 3]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_validate_valid_length() {
        let exporter = OnnxExporter::with_default_config();
        let result = exporter.validate(&[1, 2, 3, 4, 5]);
        assert!(result.is_ok());
    }
}
