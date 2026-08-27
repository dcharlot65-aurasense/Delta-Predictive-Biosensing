use serde::{Serialize, Deserialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

/// Weight export format
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum WeightFormat {
    Binary,      // Raw binary (most compact)
    Json,        // JSON (human-readable)
    Safetensors, // HuggingFace safetensors format
    Npz,         // NumPy compressed format
}

/// Model weights container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    pub layers: Vec<LayerWeights>,
    pub metadata: WeightMetadata,
}

impl ModelWeights {
    /// Create a new model weights container
    pub fn new(model_name: String) -> Self {
        Self {
            layers: Vec::new(),
            metadata: WeightMetadata {
                model_name,
                version: "1.0.0".to_string(),
                created_at: Self::current_timestamp(),
                total_params: 0,
                checksum: String::new(),
            },
        }
    }

    /// Add a layer's weights
    pub fn add_layer(&mut self, layer: LayerWeights) {
        self.metadata.total_params += layer.weights.len();
        if let Some(ref bias) = layer.bias {
            self.metadata.total_params += bias.len();
        }
        self.layers.push(layer);
    }

    /// Update the checksum
    pub fn update_checksum(&mut self) {
        self.metadata.checksum = WeightExporter::compute_checksum(self);
    }

    /// Get total number of parameters
    pub fn total_params(&self) -> usize {
        self.metadata.total_params
    }

    /// Get current timestamp as ISO 8601 string
    fn current_timestamp() -> String {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => {
                let seconds = duration.as_secs();
                format!("timestamp_{}", seconds)
            }
            Err(_) => "unknown_time".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    pub name: String,
    pub layer_type: String,
    pub weights: Vec<f64>,  // Flattened
    pub shape: Vec<usize>,
    pub bias: Option<Vec<f64>>,
}

impl LayerWeights {
    /// Create new layer weights
    pub fn new(
        name: String,
        layer_type: String,
        weights: Vec<f64>,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            name,
            layer_type,
            weights,
            shape,
            bias: None,
        }
    }

    /// Add bias to the layer
    pub fn with_bias(mut self, bias: Vec<f64>) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Get the number of parameters in this layer
    pub fn param_count(&self) -> usize {
        let weight_count = self.weights.len();
        let bias_count = self.bias.as_ref().map(|b| b.len()).unwrap_or(0);
        weight_count + bias_count
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightMetadata {
    pub model_name: String,
    pub version: String,
    pub created_at: String,
    pub total_params: usize,
    pub checksum: String,
}

/// Weight exporter
pub struct WeightExporter;

impl WeightExporter {
    /// Export weights to bytes in specified format
    pub fn export(weights: &ModelWeights, format: WeightFormat) -> Result<Vec<u8>, String> {
        match format {
            WeightFormat::Binary => Self::export_binary(weights),
            WeightFormat::Json => Self::export_json(weights),
            WeightFormat::Safetensors => Self::export_safetensors(weights),
            WeightFormat::Npz => Self::export_npz(weights),
        }
    }
    
    /// Load weights from bytes
    pub fn load(data: &[u8], format: WeightFormat) -> Result<ModelWeights, String> {
        match format {
            WeightFormat::Binary => Self::load_binary(data),
            WeightFormat::Json => Self::load_json(data),
            WeightFormat::Safetensors => Self::load_safetensors(data),
            WeightFormat::Npz => Self::load_npz(data),
        }
    }
    
    /// Compute checksum for weights
    pub fn compute_checksum(weights: &ModelWeights) -> String {
        let mut hasher = DefaultHasher::new();
        
        // Hash all layer weights
        for layer in &weights.layers {
            layer.name.hash(&mut hasher);
            layer.layer_type.hash(&mut hasher);
            
            // Hash weights
            for &w in &layer.weights {
                w.to_bits().hash(&mut hasher);
            }
            
            // Hash bias if present
            if let Some(ref bias) = layer.bias {
                for &b in bias {
                    b.to_bits().hash(&mut hasher);
                }
            }
        }
        
        format!("{:016x}", hasher.finish())
    }
    
    /// Quantize weights to specified bit width
    pub fn quantize(weights: &mut ModelWeights, bits: u8) -> Result<(), String> {
        if bits == 0 || bits > 32 {
            return Err(format!("Invalid bit width: {}. Must be 1-32", bits));
        }

        let levels = (1u64 << bits) - 1;
        
        for layer in &mut weights.layers {
            Self::quantize_vec(&mut layer.weights, levels)?;
            
            if let Some(ref mut bias) = layer.bias {
                Self::quantize_vec(bias, levels)?;
            }
        }
        
        weights.update_checksum();
        Ok(())
    }
    
    /// Prune small weights below threshold
    pub fn prune(weights: &mut ModelWeights, threshold: f64) -> usize {
        let mut pruned_count = 0;
        
        for layer in &mut weights.layers {
            pruned_count += Self::prune_vec(&mut layer.weights, threshold);
            
            if let Some(ref mut bias) = layer.bias {
                pruned_count += Self::prune_vec(bias, threshold);
            }
        }
        
        weights.update_checksum();
        pruned_count
    }
    
    // Private helper methods
    
    fn export_binary(weights: &ModelWeights) -> Result<Vec<u8>, String> {
        // Simple binary format: metadata length (8 bytes) + metadata (JSON) + weights (raw f64)
        let metadata_json = serde_json::to_vec(&weights.metadata)
            .map_err(|e| format!("Failed to serialize metadata: {}", e))?;
        
        let metadata_len = metadata_json.len() as u64;
        let mut bytes = Vec::new();
        
        // Write metadata length
        bytes.extend_from_slice(&metadata_len.to_le_bytes());
        
        // Write metadata
        bytes.extend_from_slice(&metadata_json);
        
        // Write number of layers
        bytes.extend_from_slice(&(weights.layers.len() as u64).to_le_bytes());
        
        // Write each layer
        for layer in &weights.layers {
            let layer_json = serde_json::to_vec(layer)
                .map_err(|e| format!("Failed to serialize layer: {}", e))?;
            let layer_len = layer_json.len() as u64;
            
            bytes.extend_from_slice(&layer_len.to_le_bytes());
            bytes.extend_from_slice(&layer_json);
        }
        
        Ok(bytes)
    }
    
    fn load_binary(data: &[u8]) -> Result<ModelWeights, String> {
        if data.len() < 8 {
            return Err("Data too short to contain metadata length".to_string());
        }
        
        let mut offset = 0;
        
        // Read metadata length
        let metadata_len = u64::from_le_bytes(
            data[offset..offset + 8].try_into()
                .map_err(|_| "Failed to read metadata length")?
        ) as usize;
        offset += 8;
        
        // Read metadata
        if data.len() < offset + metadata_len {
            return Err("Data too short to contain metadata".to_string());
        }
        let metadata: WeightMetadata = serde_json::from_slice(&data[offset..offset + metadata_len])
            .map_err(|e| format!("Failed to deserialize metadata: {}", e))?;
        offset += metadata_len;
        
        // Read number of layers
        if data.len() < offset + 8 {
            return Err("Data too short to contain layer count".to_string());
        }
        let layer_count = u64::from_le_bytes(
            data[offset..offset + 8].try_into()
                .map_err(|_| "Failed to read layer count")?
        ) as usize;
        offset += 8;
        
        // Read each layer
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            if data.len() < offset + 8 {
                return Err("Data too short to contain layer length".to_string());
            }
            let layer_len = u64::from_le_bytes(
                data[offset..offset + 8].try_into()
                    .map_err(|_| "Failed to read layer length")?
            ) as usize;
            offset += 8;
            
            if data.len() < offset + layer_len {
                return Err("Data too short to contain layer data".to_string());
            }
            let layer: LayerWeights = serde_json::from_slice(&data[offset..offset + layer_len])
                .map_err(|e| format!("Failed to deserialize layer: {}", e))?;
            offset += layer_len;
            
            layers.push(layer);
        }
        
        Ok(ModelWeights { layers, metadata })
    }
    
    fn export_json(weights: &ModelWeights) -> Result<Vec<u8>, String> {
        serde_json::to_vec_pretty(weights)
            .map_err(|e| format!("Failed to serialize to JSON: {}", e))
    }
    
    fn load_json(data: &[u8]) -> Result<ModelWeights, String> {
        serde_json::from_slice(data)
            .map_err(|e| format!("Failed to deserialize from JSON: {}", e))
    }
    
    fn export_safetensors(_weights: &ModelWeights) -> Result<Vec<u8>, String> {
        // Placeholder for safetensors format
        // In a real implementation, would use the safetensors library
        Err("Safetensors export not yet implemented".to_string())
    }
    
    fn load_safetensors(_data: &[u8]) -> Result<ModelWeights, String> {
        // Placeholder for safetensors format
        Err("Safetensors load not yet implemented".to_string())
    }
    
    fn export_npz(_weights: &ModelWeights) -> Result<Vec<u8>, String> {
        // Placeholder for NPZ format
        // In a real implementation, would use numpy file format
        Err("NPZ export not yet implemented".to_string())
    }
    
    fn load_npz(_data: &[u8]) -> Result<ModelWeights, String> {
        // Placeholder for NPZ format
        Err("NPZ load not yet implemented".to_string())
    }
    
    fn quantize_vec(vec: &mut [f64], levels: u64) -> Result<(), String> {
        // Find min and max for scaling
        let min = vec.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        if min == max {
            // All values are the same, no quantization needed
            return Ok(());
        }
        
        let range = max - min;
        let scale = levels as f64 / range;
        
        for value in vec.iter_mut() {
            // Quantize
            let quantized = ((*value - min) * scale).round() as u64;
            // Dequantize
            *value = (quantized as f64 / scale) + min;
        }
        
        Ok(())
    }
    
    fn prune_vec(vec: &mut [f64], threshold: f64) -> usize {
        let mut count = 0;
        let abs_threshold = threshold.abs();
        
        for value in vec.iter_mut() {
            if value.abs() < abs_threshold {
                *value = 0.0;
                count += 1;
            }
        }
        
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_weights() -> ModelWeights {
        let mut weights = ModelWeights::new("test_model".to_string());
        
        let layer1 = LayerWeights::new(
            "layer1".to_string(),
            "Linear".to_string(),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ).with_bias(vec![0.1, 0.2]);
        
        weights.add_layer(layer1);
        weights.update_checksum();
        
        weights
    }

    #[test]
    fn test_model_weights_creation() {
        let weights = create_test_weights();
        assert_eq!(weights.layers.len(), 1);
        assert_eq!(weights.total_params(), 6); // 4 weights + 2 bias
    }

    #[test]
    fn test_checksum_computation() {
        let weights = create_test_weights();
        let checksum1 = WeightExporter::compute_checksum(&weights);
        let checksum2 = WeightExporter::compute_checksum(&weights);
        
        // Same weights should produce same checksum
        assert_eq!(checksum1, checksum2);
        assert_eq!(checksum1.len(), 16); // Hex string length
    }

    #[test]
    fn test_binary_export_load_roundtrip() {
        let weights = create_test_weights();
        
        let bytes = WeightExporter::export(&weights, WeightFormat::Binary).unwrap();
        assert!(!bytes.is_empty());
        
        let loaded = WeightExporter::load(&bytes, WeightFormat::Binary).unwrap();
        assert_eq!(loaded.layers.len(), weights.layers.len());
        assert_eq!(loaded.metadata.model_name, weights.metadata.model_name);
    }

    #[test]
    fn test_json_export_load_roundtrip() {
        let weights = create_test_weights();
        
        let bytes = WeightExporter::export(&weights, WeightFormat::Json).unwrap();
        assert!(!bytes.is_empty());
        
        let loaded = WeightExporter::load(&bytes, WeightFormat::Json).unwrap();
        assert_eq!(loaded.layers.len(), weights.layers.len());
        assert_eq!(loaded.layers[0].weights, weights.layers[0].weights);
    }

    #[test]
    fn test_quantization() {
        // Create weights with non-uniform values to test quantization
        let mut weights = ModelWeights::new("test_model".to_string());
        weights.add_layer(
            LayerWeights::new(
                "layer1".to_string(),
                "Linear".to_string(),
                vec![0.123, 0.456, 0.789, 1.234, 2.345, 3.456],
                vec![6],
            ),
        );
        weights.update_checksum();

        let original_weights = weights.layers[0].weights.clone();

        // Use 3 bits (7 levels) to ensure quantization error
        WeightExporter::quantize(&mut weights, 3).unwrap();

        // Weights should be different but similar
        // With limited quantization levels, values should change
        let changed = weights.layers[0].weights.iter()
            .zip(original_weights.iter())
            .any(|(q, o)| (q - o).abs() > 1e-10);
        assert!(changed, "Quantization should change at least some weights");

        // Check that quantized values are close to originals
        for (q, o) in weights.layers[0].weights.iter().zip(original_weights.iter()) {
            assert!((q - o).abs() < 1.0); // Should be within quantization error
        }
    }

    #[test]
    fn test_quantization_invalid_bits() {
        let mut weights = create_test_weights();
        
        assert!(WeightExporter::quantize(&mut weights, 0).is_err());
        assert!(WeightExporter::quantize(&mut weights, 33).is_err());
    }

    #[test]
    fn test_pruning() {
        let mut weights = create_test_weights();
        weights.layers[0].weights = vec![0.001, 2.0, 0.002, 4.0];
        
        let pruned_count = WeightExporter::prune(&mut weights, 0.01);
        
        assert_eq!(pruned_count, 2); // Two small weights should be pruned
        assert_eq!(weights.layers[0].weights[0], 0.0);
        assert_eq!(weights.layers[0].weights[1], 2.0);
        assert_eq!(weights.layers[0].weights[2], 0.0);
        assert_eq!(weights.layers[0].weights[3], 4.0);
    }

    #[test]
    fn test_layer_param_count() {
        let layer = LayerWeights::new(
            "test".to_string(),
            "Linear".to_string(),
            vec![1.0, 2.0, 3.0],
            vec![3],
        ).with_bias(vec![0.1, 0.2]);
        
        assert_eq!(layer.param_count(), 5); // 3 weights + 2 bias
    }

    #[test]
    fn test_safetensors_not_implemented() {
        let weights = create_test_weights();
        assert!(WeightExporter::export(&weights, WeightFormat::Safetensors).is_err());
    }

    #[test]
    fn test_npz_not_implemented() {
        let weights = create_test_weights();
        assert!(WeightExporter::export(&weights, WeightFormat::Npz).is_err());
    }
}
