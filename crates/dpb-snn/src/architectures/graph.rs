//! Graph-based SNN architectures for skeleton and keypoint data

use super::SNNArchitecture;
use crate::{
    NeuronParams, SNNConfig, SNNError, SNNResult, SpikeTensor,
    layers::{NeuronState, SpikingLayer, SpikingLinear},
};
use ndarray::{Array1, Array2, Array3, s};
use serde::{Deserialize, Serialize};

/// Spiking Graph Convolutional Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingGCN {
    /// Graph convolutional layers
    #[serde(skip)]
    pub gcn_layers: Vec<SpikingGraphConvLayer>,
    /// Output layer
    #[serde(skip)]
    pub output_layer: SpikingLinear,
    /// Adjacency matrix (normalized)
    pub adjacency: Array2<f32>,
    /// Network configuration
    pub config: SNNConfig,
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of output classes
    pub num_classes: usize,
}

impl SpikingGCN {
    /// Create a new Spiking GCN
    pub fn new(
        num_nodes: usize,
        input_features: usize,
        hidden_features: Vec<usize>,
        num_classes: usize,
        adjacency: Array2<f32>,
        config: SNNConfig,
    ) -> Self {
        let mut gcn_layers = Vec::new();

        // Normalize adjacency matrix (add self-loops and degree normalization)
        let adj_normalized = Self::normalize_adjacency(&adjacency);

        // Create GCN layers
        let mut in_features = input_features;
        for &out_features in &hidden_features {
            gcn_layers.push(SpikingGraphConvLayer::new(
                in_features,
                out_features,
                adj_normalized.clone(),
                config.neuron_params.clone(),
                config.dt,
            ));
            in_features = out_features;
        }

        // Output layer (per-node or global)
        let output_layer = SpikingLinear::new(
            in_features * num_nodes, // Flattened node features
            num_classes,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        Self {
            gcn_layers,
            output_layer,
            adjacency: adj_normalized,
            config,
            num_nodes,
            num_classes,
        }
    }

    /// Normalize adjacency matrix: D^(-1/2) (A + I) D^(-1/2)
    fn normalize_adjacency(adj: &Array2<f32>) -> Array2<f32> {
        let n = adj.shape()[0];
        let mut adj_with_loops = adj.clone();

        // Add self-loops
        for i in 0..n {
            adj_with_loops[[i, i]] += 1.0;
        }

        // Compute degree matrix
        let mut degree = Array1::zeros(n);
        for i in 0..n {
            degree[i] = adj_with_loops.row(i).sum();
        }

        // D^(-1/2)
        let degree_inv_sqrt: Array1<f32> =
            degree.mapv(|d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 });

        // Normalize: D^(-1/2) A D^(-1/2)
        let mut normalized = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                normalized[[i, j]] =
                    degree_inv_sqrt[i] * adj_with_loops[[i, j]] * degree_inv_sqrt[j];
            }
        }

        normalized
    }

    /// Create GCN for skeleton data (common skeleton graph)
    pub fn for_skeleton(
        skeleton_type: SkeletonType,
        input_features: usize,
        hidden_features: Vec<usize>,
        num_classes: usize,
        config: SNNConfig,
    ) -> Self {
        let (num_nodes, adjacency) = skeleton_type.get_adjacency();
        Self::new(
            num_nodes,
            input_features,
            hidden_features,
            num_classes,
            adjacency,
            config,
        )
    }
}

impl SNNArchitecture for SpikingGCN {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let mut current = input.clone();

        // Forward through GCN layers
        for layer in &mut self.gcn_layers {
            current = layer.forward(&current)?;
        }

        // Flatten node features for output layer
        let dense = current.to_dense();
        let (batch_size, num_steps, features) =
            (dense.shape()[0], dense.shape()[1], dense.shape()[2]);

        let flattened = dense
            .to_shape((batch_size, num_steps, features))
            .unwrap()
            .to_owned();
        let flattened_tensor = SpikeTensor::from_dense(flattened, current.requires_grad);

        // Forward through output layer
        let output = self.output_layer.forward(&flattened_tensor)?;

        Ok(output)
    }

    fn reset(&mut self) {
        for layer in &mut self.gcn_layers {
            layer.reset_state();
        }
        self.output_layer.reset_state();
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        let mut params = Vec::new();
        for layer in &self.gcn_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.output_layer.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        let mut params = Vec::new();
        for layer in &mut self.gcn_layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.output_layer.parameters_mut());
        params
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.gcn_layers {
            layer.zero_grad();
        }
        self.output_layer.zero_grad();
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

/// Single Graph Convolutional Layer with spiking neurons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingGraphConvLayer {
    /// Weight matrix
    pub weights: Array2<f32>,
    /// Adjacency matrix
    pub adjacency: Array2<f32>,
    /// Weight gradient
    #[serde(skip)]
    pub weight_grad: Option<Array2<f32>>,
    /// Neuron parameters
    pub neuron_params: NeuronParams,
    /// Neuron state (per node)
    #[serde(skip)]
    pub state: Vec<Vec<NeuronState>>, // [batch][node]
    /// Time step
    pub dt: f32,
}

impl SpikingGraphConvLayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        adjacency: Array2<f32>,
        neuron_params: NeuronParams,
        dt: f32,
    ) -> Self {
        use rand::rng;
        use rand_distr::{Distribution, Normal};

        let std = (2.0 / (in_features + out_features) as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let mut rng = rng();

        let weights =
            Array2::from_shape_fn((out_features, in_features), |_| normal.sample(&mut rng));

        Self {
            weights,
            adjacency,
            weight_grad: None,
            neuron_params,
            state: Vec::new(),
            dt,
        }
    }

    fn ensure_state(&mut self, batch_size: usize, num_nodes: usize) {
        if self.state.len() != batch_size {
            self.state = (0..batch_size)
                .map(|_| {
                    (0..num_nodes)
                        .map(|_| NeuronState::new(self.weights.shape()[0], false))
                        .collect()
                })
                .collect();
        }
    }
}

impl SpikingLayer for SpikingGraphConvLayer {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, flat_features) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        // Assume features are arranged as [node1_features, node2_features, ...]
        let num_nodes = self.adjacency.shape()[0];
        let in_features = self.weights.shape()[1];
        let out_features = self.weights.shape()[0];

        if flat_features != num_nodes * in_features {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} nodes × {} features", num_nodes, in_features),
                actual: format!("{} flat features", flat_features),
            });
        }

        self.ensure_state(batch_size, num_nodes);

        let mut output = Array3::zeros((batch_size, num_steps, num_nodes * out_features));

        for t in 0..num_steps {
            for b in 0..batch_size {
                // Reshape input to (num_nodes, in_features)
                let input_t = input_dense.slice(s![b, t, ..]);
                let node_features = input_t
                    .to_shape((num_nodes, in_features))
                    .unwrap()
                    .to_owned();

                // Graph convolution: A X W
                // First: X W
                let transformed = node_features.dot(&self.weights.t());

                // Then: A (X W)
                let aggregated = self.adjacency.dot(&transformed);

                // Apply neuron dynamics to each node
                for n in 0..num_nodes {
                    let node_input = aggregated.row(n).to_owned();
                    let spikes =
                        self.state[b][n].update_lif(&node_input, &self.neuron_params, self.dt);

                    // Store spikes for this node
                    for f in 0..out_features {
                        output[[b, t, n * out_features + f]] = spikes[f];
                    }
                }
            }
        }

        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {
        for batch_states in &mut self.state {
            for state in batch_states {
                state.reset();
            }
        }
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        vec![&self.weights]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![&mut self.weights]
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        vec![self.weight_grad.as_ref()]
    }

    fn zero_grad(&mut self) {
        self.weight_grad = None;
    }
}

/// Predefined skeleton types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkeletonType {
    /// 18-keypoint skeleton (COCO format)
    COCO18,
    /// 25-keypoint skeleton (OpenPose)
    OpenPose25,
    /// 33-keypoint skeleton (MediaPipe)
    MediaPipe33,
}

impl SkeletonType {
    /// Get adjacency matrix for skeleton type
    pub fn get_adjacency(&self) -> (usize, Array2<f32>) {
        match self {
            SkeletonType::COCO18 => {
                let num_nodes = 18;
                let mut adj = Array2::zeros((num_nodes, num_nodes));

                // Define skeleton connectivity (COCO format)
                let edges = vec![
                    (0, 1),
                    (0, 14),
                    (0, 15),
                    (14, 16),
                    (15, 17), // Head
                    (0, 2),
                    (2, 4),
                    (0, 3),
                    (3, 5), // Arms
                    (0, 8),
                    (8, 10),
                    (0, 9),
                    (9, 11), // Torso to hips
                    (8, 6),
                    (6, 12),
                    (9, 7),
                    (7, 13), // Legs
                ];

                for (i, j) in edges {
                    adj[[i, j]] = 1.0;
                    adj[[j, i]] = 1.0; // Symmetric
                }

                (num_nodes, adj)
            }
            SkeletonType::OpenPose25 => {
                let num_nodes = 25;
                let adj = Array2::zeros((num_nodes, num_nodes));
                // Simplified - would need proper connectivity
                (num_nodes, adj)
            }
            SkeletonType::MediaPipe33 => {
                let num_nodes = 33;
                let adj = Array2::zeros((num_nodes, num_nodes));
                // Simplified - would need proper connectivity
                (num_nodes, adj)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacency_normalization() {
        let adj = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            .unwrap();

        let normalized = SpikingGCN::normalize_adjacency(&adj);
        assert_eq!(normalized.shape(), &[3, 3]);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((normalized[[i, j]] - normalized[[j, i]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_spiking_gcn_creation() {
        let adj = Array2::eye(5);
        let gcn = SpikingGCN::new(5, 3, vec![8, 16], 4, adj, SNNConfig::default());
        assert_eq!(gcn.num_nodes, 5);
        assert_eq!(gcn.num_classes, 4);
        assert_eq!(gcn.gcn_layers.len(), 2);
    }

    #[test]
    fn test_skeleton_adjacency() {
        let (num_nodes, adj) = SkeletonType::COCO18.get_adjacency();
        assert_eq!(num_nodes, 18);
        assert_eq!(adj.shape(), &[18, 18]);

        // Check symmetry
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                assert_eq!(adj[[i, j]], adj[[j, i]]);
            }
        }
    }

    #[test]
    fn test_gcn_for_skeleton() {
        let gcn =
            SpikingGCN::for_skeleton(SkeletonType::COCO18, 3, vec![8], 4, SNNConfig::default());
        assert_eq!(gcn.num_nodes, 18);
    }
}
