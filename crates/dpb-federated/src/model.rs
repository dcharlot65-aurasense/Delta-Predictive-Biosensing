//! Model weights and updates for federated learning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Model weights represented as named tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    /// Unique identifier.
    pub id: String,
    /// Version/round number.
    pub version: u64,
    /// Named parameter tensors.
    pub parameters: HashMap<String, Tensor>,
    /// Total number of parameters.
    pub num_parameters: usize,
    /// Model architecture metadata.
    pub metadata: ModelMetadata,
}

impl ModelWeights {
    /// Create new model weights.
    pub fn new(parameters: HashMap<String, Tensor>) -> Self {
        let num_parameters: usize = parameters.values().map(|t| t.data.len()).sum();

        Self {
            id: Uuid::new_v4().to_string(),
            version: 0,
            parameters,
            num_parameters,
            metadata: ModelMetadata::default(),
        }
    }

    /// Create empty model weights with given parameter names and shapes.
    pub fn zeros(shapes: &[(&str, Vec<usize>)]) -> Self {
        let mut parameters = HashMap::new();

        for (name, shape) in shapes {
            let size: usize = shape.iter().product();
            parameters.insert(
                name.to_string(),
                Tensor {
                    data: vec![0.0; size],
                    shape: shape.clone(),
                },
            );
        }

        Self::new(parameters)
    }

    /// Get a parameter tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.parameters.get(name)
    }

    /// Set a parameter tensor.
    pub fn set(&mut self, name: &str, tensor: Tensor) {
        self.parameters.insert(name.to_string(), tensor);
        self.num_parameters = self.parameters.values().map(|t| t.data.len()).sum();
    }

    /// Add another model's weights (element-wise).
    pub fn add(&mut self, other: &ModelWeights) -> crate::Result<()> {
        for (name, tensor) in &other.parameters {
            if let Some(self_tensor) = self.parameters.get_mut(name) {
                if self_tensor.data.len() != tensor.data.len() {
                    return Err(crate::FederatedError::DimensionMismatch {
                        expected: self_tensor.data.len(),
                        actual: tensor.data.len(),
                    });
                }
                for (a, b) in self_tensor.data.iter_mut().zip(&tensor.data) {
                    *a += b;
                }
            }
        }
        Ok(())
    }

    /// Scale all weights by a factor.
    pub fn scale(&mut self, factor: f32) {
        for tensor in self.parameters.values_mut() {
            for v in &mut tensor.data {
                *v *= factor;
            }
        }
    }

    /// Compute L2 norm of all weights.
    pub fn l2_norm(&self) -> f32 {
        let sum: f32 = self
            .parameters
            .values()
            .flat_map(|t| &t.data)
            .map(|v| v * v)
            .sum();
        sum.sqrt()
    }

    /// Clip weights to a maximum L2 norm.
    pub fn clip_norm(&mut self, max_norm: f32) {
        let current_norm = self.l2_norm();
        if current_norm > max_norm {
            self.scale(max_norm / current_norm);
        }
    }

    /// Flatten all parameters into a single vector.
    pub fn flatten(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.num_parameters);
        // Sort keys for deterministic ordering
        let mut keys: Vec<_> = self.parameters.keys().collect();
        keys.sort();
        for key in keys {
            flat.extend(&self.parameters[key].data);
        }
        flat
    }

    /// Unflatten a vector back into parameters.
    pub fn unflatten(&mut self, flat: &[f32]) -> crate::Result<()> {
        if flat.len() != self.num_parameters {
            return Err(crate::FederatedError::DimensionMismatch {
                expected: self.num_parameters,
                actual: flat.len(),
            });
        }

        let mut offset = 0;
        let mut keys: Vec<_> = self.parameters.keys().cloned().collect();
        keys.sort();

        for key in keys {
            let tensor = self.parameters.get_mut(&key).unwrap();
            let len = tensor.data.len();
            tensor.data.copy_from_slice(&flat[offset..offset + len]);
            offset += len;
        }

        Ok(())
    }
}

impl Default for ModelWeights {
    fn default() -> Self {
        Self::new(HashMap::new())
    }
}

/// A tensor (multi-dimensional array).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Flattened data in row-major order.
    pub data: Vec<f32>,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Create a tensor of zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Create a tensor of ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![1.0; size],
            shape,
        }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Model update (delta) from a client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    /// Client identifier.
    pub client_id: String,
    /// Round number this update is for.
    pub round: u64,
    /// Parameter deltas.
    pub delta: ParameterDelta,
    /// Number of local samples used.
    pub num_samples: usize,
    /// Local training loss.
    pub loss: Option<f32>,
    /// Training metrics.
    pub metrics: HashMap<String, f32>,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ModelUpdate {
    /// Create a new model update.
    pub fn new(client_id: &str, round: u64, delta: ParameterDelta, num_samples: usize) -> Self {
        Self {
            client_id: client_id.to_string(),
            round,
            delta,
            num_samples,
            loss: None,
            metrics: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Set the training loss.
    pub fn with_loss(mut self, loss: f32) -> Self {
        self.loss = Some(loss);
        self
    }

    /// Add a metric.
    pub fn with_metric(mut self, name: &str, value: f32) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }
}

/// Parameter delta (change from base model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDelta {
    /// Base model version this delta is relative to.
    pub base_version: u64,
    /// Parameter changes.
    pub changes: HashMap<String, Tensor>,
    /// Whether the delta is compressed.
    pub compressed: bool,
    /// Compression metadata (if compressed).
    pub compression_info: Option<CompressionInfo>,
}

impl ParameterDelta {
    /// Create a new parameter delta.
    pub fn new(base_version: u64, changes: HashMap<String, Tensor>) -> Self {
        Self {
            base_version,
            changes,
            compressed: false,
            compression_info: None,
        }
    }

    /// Compute delta from two model weights.
    pub fn from_models(before: &ModelWeights, after: &ModelWeights) -> crate::Result<Self> {
        let mut changes = HashMap::new();

        for (name, after_tensor) in &after.parameters {
            if let Some(before_tensor) = before.parameters.get(name) {
                if before_tensor.data.len() != after_tensor.data.len() {
                    return Err(crate::FederatedError::DimensionMismatch {
                        expected: before_tensor.data.len(),
                        actual: after_tensor.data.len(),
                    });
                }

                let delta_data: Vec<f32> = before_tensor
                    .data
                    .iter()
                    .zip(&after_tensor.data)
                    .map(|(b, a)| a - b)
                    .collect();

                changes.insert(
                    name.clone(),
                    Tensor {
                        data: delta_data,
                        shape: after_tensor.shape.clone(),
                    },
                );
            }
        }

        Ok(Self::new(before.version, changes))
    }

    /// Apply delta to model weights.
    pub fn apply_to(&self, weights: &mut ModelWeights) -> crate::Result<()> {
        for (name, delta_tensor) in &self.changes {
            if let Some(weight_tensor) = weights.parameters.get_mut(name) {
                if weight_tensor.data.len() != delta_tensor.data.len() {
                    return Err(crate::FederatedError::DimensionMismatch {
                        expected: weight_tensor.data.len(),
                        actual: delta_tensor.data.len(),
                    });
                }
                for (w, d) in weight_tensor.data.iter_mut().zip(&delta_tensor.data) {
                    *w += d;
                }
            }
        }
        weights.version = self.base_version + 1;
        Ok(())
    }
}

/// Compression metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    /// Original size in bytes.
    pub original_size: usize,
    /// Compressed size in bytes.
    pub compressed_size: usize,
    /// Compression algorithm.
    pub algorithm: String,
    /// Sparsity ratio (for sparse compression).
    pub sparsity: Option<f32>,
}

/// Model architecture metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name.
    pub name: String,
    /// Model type (e.g., "encoder", "classifier").
    pub model_type: String,
    /// Architecture description.
    pub architecture: String,
    /// Input shape.
    pub input_shape: Vec<usize>,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Additional metadata.
    pub extra: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_weights_creation() {
        let mut params = HashMap::new();
        params.insert(
            "layer1".to_string(),
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        );

        let weights = ModelWeights::new(params);
        assert_eq!(weights.num_parameters, 4);
        assert!(weights.get("layer1").is_some());
    }

    #[test]
    fn test_model_weights_zeros() {
        let weights = ModelWeights::zeros(&[("w1", vec![10, 5]), ("w2", vec![5, 3])]);
        assert_eq!(weights.num_parameters, 65);
        assert_eq!(weights.get("w1").unwrap().data.len(), 50);
        assert_eq!(weights.get("w2").unwrap().data.len(), 15);
    }

    #[test]
    fn test_model_weights_scale() {
        let mut weights = ModelWeights::zeros(&[("w", vec![4])]);
        weights.set("w".into(), Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]));
        weights.scale(2.0);
        assert_eq!(weights.get("w").unwrap().data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_model_weights_l2_norm() {
        let mut weights = ModelWeights::zeros(&[("w", vec![3])]);
        weights.set("w".into(), Tensor::new(vec![3.0, 4.0, 0.0], vec![3]));
        assert!((weights.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_model_weights_clip_norm() {
        let mut weights = ModelWeights::zeros(&[("w", vec![2])]);
        weights.set("w".into(), Tensor::new(vec![3.0, 4.0], vec![2]));
        weights.clip_norm(2.5);
        assert!((weights.l2_norm() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_model_weights_flatten_unflatten() {
        let mut weights = ModelWeights::zeros(&[("a", vec![2]), ("b", vec![3])]);
        weights.set("a".into(), Tensor::new(vec![1.0, 2.0], vec![2]));
        weights.set("b".into(), Tensor::new(vec![3.0, 4.0, 5.0], vec![3]));

        let flat = weights.flatten();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut new_weights = ModelWeights::zeros(&[("a", vec![2]), ("b", vec![3])]);
        new_weights.unflatten(&flat).unwrap();
        assert_eq!(new_weights.get("a").unwrap().data, vec![1.0, 2.0]);
        assert_eq!(new_weights.get("b").unwrap().data, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_parameter_delta() {
        let before = ModelWeights::zeros(&[("w", vec![3])]);
        let mut after = ModelWeights::zeros(&[("w", vec![3])]);
        after.set("w".into(), Tensor::new(vec![1.0, 2.0, 3.0], vec![3]));

        let delta = ParameterDelta::from_models(&before, &after).unwrap();
        assert_eq!(delta.changes["w"].data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_apply_delta() {
        let mut weights = ModelWeights::zeros(&[("w", vec![3])]);
        let delta = ParameterDelta::new(
            0,
            [("w".to_string(), Tensor::new(vec![1.0, 2.0, 3.0], vec![3]))]
                .into_iter()
                .collect(),
        );

        delta.apply_to(&mut weights).unwrap();
        assert_eq!(weights.get("w").unwrap().data, vec![1.0, 2.0, 3.0]);
        assert_eq!(weights.version, 1);
    }

    #[test]
    fn test_tensor_zeros_ones() {
        let zeros = Tensor::zeros(vec![2, 3]);
        assert_eq!(zeros.numel(), 6);
        assert!(zeros.data.iter().all(|&v| v == 0.0));

        let ones = Tensor::ones(vec![3, 4]);
        assert_eq!(ones.numel(), 12);
        assert!(ones.data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_model_update() {
        let delta = ParameterDelta::new(0, HashMap::new());
        let update = ModelUpdate::new("client_1", 5, delta, 100)
            .with_loss(0.5)
            .with_metric("accuracy", 0.95);

        assert_eq!(update.client_id, "client_1");
        assert_eq!(update.round, 5);
        assert_eq!(update.num_samples, 100);
        assert_eq!(update.loss, Some(0.5));
        assert_eq!(update.metrics.get("accuracy"), Some(&0.95));
    }
}
