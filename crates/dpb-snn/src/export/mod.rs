//! Model export utilities for hardware deployment

use crate::{
    architectures::{FeedforwardSNN, SNNArchitecture},
    layers::SpikingLinear,
    SNNConfig, SNNError, SNNResult,
};
use ndarray::{Array2, Array4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Neural Intermediate Representation (NIR) export
/// Based on neuromorphic-systems/nir specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NIRExporter {
    /// NIR version
    pub version: String,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

impl NIRExporter {
    pub fn new() -> Self {
        Self {
            version: "1.0".to_string(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Export SNN to NIR format
    pub fn export<P: AsRef<Path>>(
        &self,
        network: &FeedforwardSNN,
        path: P,
    ) -> SNNResult<()> {
        let nir_graph = self.convert_to_nir(network)?;
        let json = serde_json::to_string_pretty(&nir_graph)
            .map_err(|e| SNNError::Export(e.to_string()))?;

        let mut file = File::create(path)
            .map_err(|e| SNNError::Export(e.to_string()))?;

        file.write_all(json.as_bytes())
            .map_err(|e| SNNError::Export(e.to_string()))?;

        Ok(())
    }

    fn convert_to_nir(&self, network: &FeedforwardSNN) -> SNNResult<NIRGraph> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Convert each layer to NIR node
        for (i, layer) in network.layers.iter().enumerate() {
            let node = NIRNode {
                id: format!("layer_{}", i),
                node_type: "LIF".to_string(),
                parameters: self.extract_layer_params(layer),
            };
            nodes.push(node);

            // Add edge to next layer
            if i < network.layers.len() - 1 {
                edges.push(NIREdge {
                    source: format!("layer_{}", i),
                    target: format!("layer_{}", i + 1),
                });
            }
        }

        Ok(NIRGraph {
            version: self.version.clone(),
            nodes,
            edges,
            metadata: self.metadata.clone(),
        })
    }

    fn extract_layer_params(&self, layer: &SpikingLinear) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        params.insert(
            "input_size".to_string(),
            serde_json::json!(layer.weights.shape()[1]),
        );
        params.insert(
            "output_size".to_string(),
            serde_json::json!(layer.weights.shape()[0]),
        );
        params.insert(
            "tau_mem".to_string(),
            serde_json::json!(layer.neuron_params.tau_mem),
        );
        params.insert(
            "tau_syn".to_string(),
            serde_json::json!(layer.neuron_params.tau_syn),
        );
        params.insert(
            "v_threshold".to_string(),
            serde_json::json!(layer.neuron_params.v_threshold),
        );

        params
    }
}

impl Default for NIRExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// NIR Graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NIRGraph {
    pub version: String,
    pub nodes: Vec<NIRNode>,
    pub edges: Vec<NIREdge>,
    pub metadata: HashMap<String, String>,
}

/// NIR Node (layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NIRNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// NIR Edge (connection)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NIREdge {
    pub source: String,
    pub target: String,
}

/// ONNX export (simplified stub)
#[derive(Debug, Clone)]
pub struct ONNXExporter {
    /// ONNX opset version
    pub opset_version: i64,
}

impl ONNXExporter {
    pub fn new(opset_version: i64) -> Self {
        Self { opset_version }
    }

    /// Export to ONNX format (stub)
    pub fn export<P: AsRef<Path>>(
        &self,
        _network: &dyn SNNArchitecture,
        _path: P,
    ) -> SNNResult<()> {
        // Full ONNX export would require the onnx crate
        // This is a stub implementation
        Err(SNNError::Export(
            "ONNX export not fully implemented - requires onnx crate".to_string(),
        ))
    }
}

/// TensorFlow Lite export (stub)
#[derive(Debug, Clone)]
pub struct TFLiteExporter {
    /// Quantization settings
    pub quantize: bool,
}

impl TFLiteExporter {
    pub fn new(quantize: bool) -> Self {
        Self { quantize }
    }

    /// Export to TFLite format (stub)
    pub fn export<P: AsRef<Path>>(
        &self,
        _network: &dyn SNNArchitecture,
        _path: P,
    ) -> SNNResult<()> {
        // Full TFLite export would require tflite bindings
        // This is a stub implementation
        Err(SNNError::Export(
            "TFLite export not fully implemented - requires tflite bindings".to_string(),
        ))
    }
}

/// Custom binary format for efficient loading
#[derive(Debug, Clone)]
pub struct BinaryExporter;

impl BinaryExporter {
    /// Export network weights to binary format
    pub fn export_weights<P: AsRef<Path>>(
        network: &FeedforwardSNN,
        path: P,
    ) -> SNNResult<()> {
        use std::io::BufWriter;

        let file = File::create(path).map_err(|e| SNNError::Export(e.to_string()))?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer
            .write_all(b"DPBSNN")
            .map_err(|e| SNNError::Export(e.to_string()))?;

        // Write number of layers
        let num_layers = network.layers.len() as u32;
        writer
            .write_all(&num_layers.to_le_bytes())
            .map_err(|e| SNNError::Export(e.to_string()))?;

        // Write each layer's weights
        for layer in &network.layers {
            let (rows, cols) = (layer.weights.shape()[0] as u32, layer.weights.shape()[1] as u32);

            writer
                .write_all(&rows.to_le_bytes())
                .map_err(|e| SNNError::Export(e.to_string()))?;
            writer
                .write_all(&cols.to_le_bytes())
                .map_err(|e| SNNError::Export(e.to_string()))?;

            // Write weight data
            for &weight in layer.weights.iter() {
                writer
                    .write_all(&weight.to_le_bytes())
                    .map_err(|e| SNNError::Export(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Import network weights from binary format
    pub fn import_weights<P: AsRef<Path>>(
        network: &mut FeedforwardSNN,
        path: P,
    ) -> SNNResult<()> {
        use std::io::{BufReader, Read};

        let file = File::open(path).map_err(|e| SNNError::Export(e.to_string()))?;
        let mut reader = BufReader::new(file);

        // Read and verify header
        let mut header = [0u8; 6];
        reader
            .read_exact(&mut header)
            .map_err(|e| SNNError::Export(e.to_string()))?;

        if &header != b"DPBSNN" {
            return Err(SNNError::Export("Invalid file format".to_string()));
        }

        // Read number of layers
        let mut num_layers_bytes = [0u8; 4];
        reader
            .read_exact(&mut num_layers_bytes)
            .map_err(|e| SNNError::Export(e.to_string()))?;
        let num_layers = u32::from_le_bytes(num_layers_bytes) as usize;

        if num_layers != network.layers.len() {
            return Err(SNNError::Export(format!(
                "Layer count mismatch: expected {}, got {}",
                network.layers.len(),
                num_layers
            )));
        }

        // Read each layer's weights
        for layer in &mut network.layers {
            let mut dims = [0u8; 8];
            reader
                .read_exact(&mut dims)
                .map_err(|e| SNNError::Export(e.to_string()))?;

            let rows = u32::from_le_bytes([dims[0], dims[1], dims[2], dims[3]]) as usize;
            let cols = u32::from_le_bytes([dims[4], dims[5], dims[6], dims[7]]) as usize;

            if rows != layer.weights.shape()[0] || cols != layer.weights.shape()[1] {
                return Err(SNNError::Export(format!(
                    "Weight shape mismatch for layer: expected ({}, {}), got ({}, {})",
                    layer.weights.shape()[0],
                    layer.weights.shape()[1],
                    rows,
                    cols
                )));
            }

            // Read weights
            for i in 0..rows {
                for j in 0..cols {
                    let mut weight_bytes = [0u8; 4];
                    reader
                        .read_exact(&mut weight_bytes)
                        .map_err(|e| SNNError::Export(e.to_string()))?;
                    layer.weights[[i, j]] = f32::from_le_bytes(weight_bytes);
                }
            }
        }

        Ok(())
    }
}

/// Hardware configuration for neuromorphic chips
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Target hardware platform
    pub platform: NeuromorphicPlatform,
    /// Time step in microseconds
    pub time_step_us: u32,
    /// Number of available neurons
    pub num_neurons: usize,
    /// Synapse density limit
    pub max_synapses_per_neuron: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuromorphicPlatform {
    /// Intel Loihi
    Loihi,
    /// IBM TrueNorth
    TrueNorth,
    /// BrainScaleS
    BrainScaleS,
    /// SpiNNaker
    SpiNNaker,
    /// Custom FPGA
    CustomFPGA,
}

impl HardwareConfig {
    pub fn loihi_default() -> Self {
        Self {
            platform: NeuromorphicPlatform::Loihi,
            time_step_us: 1000, // 1ms
            num_neurons: 131072,
            max_synapses_per_neuron: 4096,
        }
    }

    pub fn spinnaker_default() -> Self {
        Self {
            platform: NeuromorphicPlatform::SpiNNaker,
            time_step_us: 1000,
            num_neurons: 16384,
            max_synapses_per_neuron: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nir_exporter_creation() {
        let exporter = NIRExporter::new()
            .with_metadata("model_name".to_string(), "test_snn".to_string())
            .with_metadata("author".to_string(), "dpb".to_string());

        assert_eq!(exporter.version, "1.0");
        assert_eq!(exporter.metadata.len(), 2);
    }

    #[test]
    fn test_hardware_config() {
        let config = HardwareConfig::loihi_default();
        assert_eq!(config.platform, NeuromorphicPlatform::Loihi);
        assert_eq!(config.num_neurons, 131072);
    }

    #[test]
    fn test_nir_graph_serialization() {
        let graph = NIRGraph {
            version: "1.0".to_string(),
            nodes: vec![
                NIRNode {
                    id: "layer_0".to_string(),
                    node_type: "LIF".to_string(),
                    parameters: HashMap::new(),
                },
            ],
            edges: vec![],
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("layer_0"));
    }
}
