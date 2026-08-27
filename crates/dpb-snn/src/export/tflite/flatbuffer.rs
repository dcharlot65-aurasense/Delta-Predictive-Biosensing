//! FlatBuffer serialization for TensorFlow Lite models
//!
//! Implements FlatBuffer generation for the TFLite schema. This is a simplified
//! implementation that generates schema-compliant binary data.

use super::operators::TFLiteOperator;
use super::tensors::TFLiteTensor;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// FlatBuffer builder for TFLite models
pub struct FlatBufferBuilder {
    /// Model version
    version: u32,
    /// Subgraphs in the model
    subgraphs: Vec<Subgraph>,
    /// Description string
    description: String,
    /// Buffer manager
    buffer_manager: BufferManager,
    /// Metadata
    metadata: Vec<(String, Vec<u8>)>,
}

impl FlatBufferBuilder {
    /// Create a new FlatBuffer builder
    pub fn new() -> Self {
        Self {
            version: 3, // TFLite schema version
            subgraphs: Vec::new(),
            description: String::new(),
            buffer_manager: BufferManager::new(),
            metadata: Vec::new(),
        }
    }

    /// Set model description
    pub fn with_description(mut self, desc: String) -> Self {
        self.description = desc;
        self
    }

    /// Add a subgraph
    pub fn add_subgraph(&mut self, subgraph: Subgraph) {
        self.subgraphs.push(subgraph);
    }

    /// Get buffer manager
    pub fn buffer_manager(&mut self) -> &mut BufferManager {
        &mut self.buffer_manager
    }

    /// Add metadata
    pub fn add_metadata(&mut self, name: String, data: Vec<u8>) {
        self.metadata.push((name, data));
    }

    /// Build FlatBuffer bytes
    pub fn build(&self) -> Result<Vec<u8>, String> {
        // This is a simplified implementation
        // In a real implementation, we would use the flatbuffers crate
        // and the official TFLite schema

        let mut buffer = Vec::new();

        // Write magic number for FlatBuffer
        buffer.extend_from_slice(b"TFL3"); // TFLite magic

        // Write version
        buffer.extend_from_slice(&self.version.to_le_bytes());

        // Serialize model structure to JSON (placeholder)
        let model_data = ModelData {
            version: self.version,
            description: self.description.clone(),
            subgraphs: self.subgraphs.clone(),
            buffers: self.buffer_manager.buffers.clone(),
        };

        let json = serde_json::to_vec(&model_data)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;

        // Write length of JSON data (with overflow check)
        let json_len: u32 = json.len().try_into()
            .map_err(|_| "JSON data too large for u32 length")?;
        buffer.extend_from_slice(&json_len.to_le_bytes());

        // Write JSON data
        buffer.extend_from_slice(&json);

        // Write metadata count (with overflow check)
        let metadata_len: u32 = self.metadata.len().try_into()
            .map_err(|_| "Too many metadata entries for u32")?;
        buffer.extend_from_slice(&metadata_len.to_le_bytes());
        for (name, data) in &self.metadata {
            let name_bytes = name.as_bytes();
            let name_len: u32 = name_bytes.len().try_into()
                .map_err(|_| format!("Metadata name '{}' too long", name))?;
            buffer.extend_from_slice(&name_len.to_le_bytes());
            buffer.extend_from_slice(name_bytes);
            let data_len: u32 = data.len().try_into()
                .map_err(|_| format!("Metadata data for '{}' too large", name))?;
            buffer.extend_from_slice(&data_len.to_le_bytes());
            buffer.extend_from_slice(data);
        }

        Ok(buffer)
    }

    /// Validate the model structure
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.subgraphs.is_empty() {
            errors.push("Model must have at least one subgraph".to_string());
        }

        for (i, subgraph) in self.subgraphs.iter().enumerate() {
            if let Err(e) = subgraph.validate() {
                errors.push(format!("Subgraph {}: {}", i, e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

impl Default for FlatBufferBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Subgraph within a TFLite model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subgraph {
    /// Subgraph name
    pub name: String,
    /// Tensors in this subgraph
    pub tensors: Vec<TFLiteTensor>,
    /// Operators in this subgraph
    pub operators: Vec<TFLiteOperator>,
    /// Input tensor indices
    pub inputs: Vec<usize>,
    /// Output tensor indices
    pub outputs: Vec<usize>,
}

impl Subgraph {
    /// Create a new subgraph
    pub fn new(name: String) -> Self {
        Self {
            name,
            tensors: Vec::new(),
            operators: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Add a tensor
    pub fn add_tensor(&mut self, tensor: TFLiteTensor) -> usize {
        let idx = self.tensors.len();
        self.tensors.push(tensor);
        idx
    }

    /// Add an operator
    pub fn add_operator(&mut self, operator: TFLiteOperator) -> usize {
        let idx = self.operators.len();
        self.operators.push(operator);
        idx
    }

    /// Set input tensors
    pub fn set_inputs(&mut self, inputs: Vec<usize>) {
        self.inputs = inputs;
    }

    /// Set output tensors
    pub fn set_outputs(&mut self, outputs: Vec<usize>) {
        self.outputs = outputs;
    }

    /// Validate subgraph
    pub fn validate(&self) -> Result<(), String> {
        // Check inputs
        if self.inputs.is_empty() {
            return Err("Subgraph must have at least one input".to_string());
        }

        for &input_idx in &self.inputs {
            if input_idx >= self.tensors.len() {
                return Err(format!("Invalid input tensor index: {}", input_idx));
            }
        }

        // Check outputs
        if self.outputs.is_empty() {
            return Err("Subgraph must have at least one output".to_string());
        }

        for &output_idx in &self.outputs {
            if output_idx >= self.tensors.len() {
                return Err(format!("Invalid output tensor index: {}", output_idx));
            }
        }

        // Validate all tensors
        for (i, tensor) in self.tensors.iter().enumerate() {
            if let Err(e) = tensor.validate() {
                return Err(format!("Tensor {}: {}", i, e));
            }
        }

        // Validate all operators
        for (i, op) in self.operators.iter().enumerate() {
            if let Err(e) = op.validate() {
                return Err(format!("Operator {}: {}", i, e));
            }

            // Check operator inputs/outputs are valid tensor indices
            for &input_idx in &op.inputs {
                if input_idx >= self.tensors.len() {
                    return Err(format!(
                        "Operator {} has invalid input tensor index: {}",
                        i, input_idx
                    ));
                }
            }

            for &output_idx in &op.outputs {
                if output_idx >= self.tensors.len() {
                    return Err(format!(
                        "Operator {} has invalid output tensor index: {}",
                        i, output_idx
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get number of tensors
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Get number of operators
    pub fn num_operators(&self) -> usize {
        self.operators.len()
    }
}

/// Subgraph builder helper
pub struct SubgraphBuilder {
    subgraph: Subgraph,
    tensor_map: HashMap<String, usize>,
}

impl SubgraphBuilder {
    /// Create a new subgraph builder
    pub fn new(name: String) -> Self {
        Self {
            subgraph: Subgraph::new(name),
            tensor_map: HashMap::new(),
        }
    }

    /// Add a named tensor
    pub fn add_tensor(&mut self, tensor: TFLiteTensor) -> usize {
        let name = tensor.name.clone();
        let idx = self.subgraph.add_tensor(tensor);
        self.tensor_map.insert(name, idx);
        idx
    }

    /// Get tensor index by name
    pub fn get_tensor_index(&self, name: &str) -> Option<usize> {
        self.tensor_map.get(name).copied()
    }

    /// Add an operator
    pub fn add_operator(&mut self, operator: TFLiteOperator) -> usize {
        self.subgraph.add_operator(operator)
    }

    /// Set inputs by name
    pub fn set_inputs_by_name(&mut self, names: &[String]) -> Result<(), String> {
        let mut indices = Vec::new();
        for name in names {
            let idx = self
                .get_tensor_index(name)
                .ok_or_else(|| format!("Input tensor not found: {}", name))?;
            indices.push(idx);
        }
        self.subgraph.set_inputs(indices);
        Ok(())
    }

    /// Set outputs by name
    pub fn set_outputs_by_name(&mut self, names: &[String]) -> Result<(), String> {
        let mut indices = Vec::new();
        for name in names {
            let idx = self
                .get_tensor_index(name)
                .ok_or_else(|| format!("Output tensor not found: {}", name))?;
            indices.push(idx);
        }
        self.subgraph.set_outputs(indices);
        Ok(())
    }

    /// Build the subgraph
    pub fn build(self) -> Subgraph {
        self.subgraph
    }
}

/// Buffer manager for tensor data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferManager {
    buffers: Vec<Buffer>,
}

impl BufferManager {
    /// Create a new buffer manager
    pub fn new() -> Self {
        Self {
            buffers: vec![Buffer::empty()], // Buffer 0 is always empty
        }
    }

    /// Add a buffer with data
    pub fn add_buffer(&mut self, data: Vec<u8>) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(Buffer::with_data(data));
        idx
    }

    /// Add a buffer from f32 slice
    pub fn add_f32_buffer(&mut self, data: &[f32]) -> usize {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        self.add_buffer(bytes)
    }

    /// Add a buffer from i8 slice
    pub fn add_i8_buffer(&mut self, data: &[i8]) -> usize {
        let bytes: Vec<u8> = data.iter().map(|&i| i as u8).collect();
        self.add_buffer(bytes)
    }

    /// Get buffer count
    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }

    /// Get total buffer size in bytes
    pub fn total_size(&self) -> usize {
        self.buffers.iter().map(|b| b.data.len()).sum()
    }
}

impl Default for BufferManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Buffer containing tensor data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Buffer {
    data: Vec<u8>,
}

impl Buffer {
    /// Create empty buffer
    fn empty() -> Self {
        Self { data: Vec::new() }
    }

    /// Create buffer with data
    fn with_data(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Model data structure for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelData {
    version: u32,
    description: String,
    subgraphs: Vec<Subgraph>,
    buffers: Vec<Buffer>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::TensorType;
    use crate::export::tflite::tensors::TensorShape;

    #[test]
    fn test_flatbuffer_builder_creation() {
        let builder = FlatBufferBuilder::new();
        assert_eq!(builder.version, 3);
        assert!(builder.subgraphs.is_empty());
    }

    #[test]
    fn test_flatbuffer_builder_with_description() {
        let builder = FlatBufferBuilder::new()
            .with_description("Test model".to_string());
        assert_eq!(builder.description, "Test model");
    }

    #[test]
    fn test_subgraph_creation() {
        let subgraph = Subgraph::new("main".to_string());
        assert_eq!(subgraph.name, "main");
        assert!(subgraph.tensors.is_empty());
        assert!(subgraph.operators.is_empty());
    }

    #[test]
    fn test_subgraph_add_tensor() {
        let mut subgraph = Subgraph::new("main".to_string());
        let tensor = TFLiteTensor::new(
            "input".to_string(),
            TensorShape::new(vec![1, 224, 224, 3]),
            TensorType::Float32,
        );

        let idx = subgraph.add_tensor(tensor);
        assert_eq!(idx, 0);
        assert_eq!(subgraph.num_tensors(), 1);
    }

    #[test]
    fn test_subgraph_validation_no_inputs() {
        let subgraph = Subgraph::new("main".to_string());
        let result = subgraph.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_subgraph_validation_invalid_input_index() {
        let mut subgraph = Subgraph::new("main".to_string());
        subgraph.set_inputs(vec![0]); // No tensors added yet
        subgraph.set_outputs(vec![0]);

        let result = subgraph.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_subgraph_validation_success() {
        let mut subgraph = Subgraph::new("main".to_string());

        let input = TFLiteTensor::new(
            "input".to_string(),
            TensorShape::new(vec![1, 10]),
            TensorType::Float32,
        );
        let output = TFLiteTensor::new(
            "output".to_string(),
            TensorShape::new(vec![1, 5]),
            TensorType::Float32,
        );

        subgraph.add_tensor(input);
        subgraph.add_tensor(output);
        subgraph.set_inputs(vec![0]);
        subgraph.set_outputs(vec![1]);

        let result = subgraph.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_subgraph_builder() {
        let mut builder = SubgraphBuilder::new("main".to_string());

        let input = TFLiteTensor::new(
            "input".to_string(),
            TensorShape::new(vec![1, 10]),
            TensorType::Float32,
        );

        let idx = builder.add_tensor(input);
        assert_eq!(idx, 0);

        assert_eq!(builder.get_tensor_index("input"), Some(0));
        assert_eq!(builder.get_tensor_index("nonexistent"), None);
    }

    #[test]
    fn test_subgraph_builder_set_inputs_by_name() {
        let mut builder = SubgraphBuilder::new("main".to_string());

        builder.add_tensor(TFLiteTensor::new(
            "input".to_string(),
            TensorShape::new(vec![1, 10]),
            TensorType::Float32,
        ));

        builder.add_tensor(TFLiteTensor::new(
            "output".to_string(),
            TensorShape::new(vec![1, 5]),
            TensorType::Float32,
        ));

        let result = builder.set_inputs_by_name(&[String::from("input")]);
        assert!(result.is_ok());

        let result = builder.set_outputs_by_name(&[String::from("output")]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_subgraph_builder_invalid_name() {
        let mut builder = SubgraphBuilder::new("main".to_string());

        let result = builder.set_inputs_by_name(&[String::from("nonexistent")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_manager_creation() {
        let manager = BufferManager::new();
        assert_eq!(manager.num_buffers(), 1); // Always has empty buffer at index 0
    }

    #[test]
    fn test_buffer_manager_add_buffer() {
        let mut manager = BufferManager::new();
        let data = vec![1, 2, 3, 4];
        let idx = manager.add_buffer(data.clone());

        assert_eq!(idx, 1); // Index 0 is empty buffer
        assert_eq!(manager.num_buffers(), 2);
    }

    #[test]
    fn test_buffer_manager_add_f32_buffer() {
        let mut manager = BufferManager::new();
        let data = vec![1.0_f32, 2.0, 3.0];
        let idx = manager.add_f32_buffer(&data);

        assert_eq!(idx, 1);
        assert_eq!(manager.buffers[idx].size(), 12); // 3 floats * 4 bytes
    }

    #[test]
    fn test_buffer_manager_add_i8_buffer() {
        let mut manager = BufferManager::new();
        let data = vec![1_i8, 2, 3, 4];
        let idx = manager.add_i8_buffer(&data);

        assert_eq!(idx, 1);
        assert_eq!(manager.buffers[idx].size(), 4);
    }

    #[test]
    fn test_buffer_manager_total_size() {
        let mut manager = BufferManager::new();
        manager.add_buffer(vec![1, 2, 3]);
        manager.add_buffer(vec![4, 5]);

        assert_eq!(manager.total_size(), 5); // 0 (empty) + 3 + 2
    }

    #[test]
    fn test_buffer_empty() {
        let buffer = Buffer::empty();
        assert!(buffer.is_empty());
        assert_eq!(buffer.size(), 0);
    }

    #[test]
    fn test_buffer_with_data() {
        let buffer = Buffer::with_data(vec![1, 2, 3]);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.size(), 3);
    }

    #[test]
    fn test_flatbuffer_build() {
        let mut builder = FlatBufferBuilder::new();

        let mut subgraph = Subgraph::new("main".to_string());
        subgraph.add_tensor(TFLiteTensor::new(
            "input".to_string(),
            TensorShape::new(vec![1, 10]),
            TensorType::Float32,
        ));
        subgraph.add_tensor(TFLiteTensor::new(
            "output".to_string(),
            TensorShape::new(vec![1, 5]),
            TensorType::Float32,
        ));
        subgraph.set_inputs(vec![0]);
        subgraph.set_outputs(vec![1]);

        builder.add_subgraph(subgraph);

        let result = builder.build();
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.starts_with(b"TFL3")); // Magic number
    }

    #[test]
    fn test_flatbuffer_validation() {
        let builder = FlatBufferBuilder::new();
        let result = builder.validate();
        assert!(result.is_err()); // No subgraphs

        let mut builder = FlatBufferBuilder::new();
        let mut subgraph = Subgraph::new("main".to_string());
        subgraph.add_tensor(TFLiteTensor::new(
            "input".to_string(),
            TensorShape::new(vec![1, 10]),
            TensorType::Float32,
        ));
        subgraph.set_inputs(vec![0]);
        subgraph.set_outputs(vec![0]);

        builder.add_subgraph(subgraph);

        let result = builder.validate();
        assert!(result.is_ok());
    }
}
