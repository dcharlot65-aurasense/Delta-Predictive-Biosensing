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
    pub gcn_layers: Vec<SpikingGraphConvLayer>,
    /// Output layer
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
    /// Node count and symmetric adjacency for this skeleton layout.
    ///
    /// The edges are the anatomical bones of each format, in that format's own
    /// keypoint order. Two of the three used to return an all-zero matrix with
    /// a comment reading "Simplified - would need proper connectivity"; with no
    /// edges the graph convolution has no neighbours to gather from, so the GCN
    /// silently degenerates into a per-node MLP -- structurally intact, and
    /// blind to the skeleton it exists to exploit.
    ///
    /// COCO18's edges were present but wrong: limb chains skipped their middle
    /// joint (shoulder straight to wrist) and several edges crossed between
    /// limbs (right elbow to left shoulder, right hip to left elbow).
    pub fn get_adjacency(&self) -> (usize, Array2<f32>) {
        // Order matches each format's published keypoint indices.
        let (num_nodes, edges): (usize, &[(usize, usize)]) = match self {
            // 0 nose, 1 neck, 2-4 right arm, 5-7 left arm, 8-10 right leg,
            // 11-13 left leg, 14/16 right eye/ear, 15/17 left eye/ear.
            SkeletonType::COCO18 => (
                18,
                &[
                    (0, 1),
                    (0, 14),
                    (14, 16),
                    (0, 15),
                    (15, 17),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (1, 5),
                    (5, 6),
                    (6, 7),
                    (1, 8),
                    (8, 9),
                    (9, 10),
                    (1, 11),
                    (11, 12),
                    (12, 13),
                ],
            ),
            // OpenPose BODY_25: 8 is the mid-hip root, 19-24 are the feet.
            SkeletonType::OpenPose25 => (
                25,
                &[
                    (0, 1),
                    (0, 15),
                    (15, 17),
                    (0, 16),
                    (16, 18),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (1, 5),
                    (5, 6),
                    (6, 7),
                    (1, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (8, 12),
                    (12, 13),
                    (13, 14),
                    (11, 22),
                    (22, 23),
                    (11, 24),
                    (14, 19),
                    (19, 20),
                    (14, 21),
                ],
            ),
            // MediaPipe Pose: POSE_CONNECTIONS verbatim, including the hand
            // and foot triangles that close on themselves.
            //
            // Worth knowing before using this for a GCN: the published list is
            // not connected. It forms three components -- the face (0-8), the
            // mouth pair (9, 10), and the body (11-32) -- because MediaPipe
            // draws no edge from the head to the shoulders. Message passing
            // therefore never carries information between head and body,
            // however many layers are stacked. That is the format's own
            // topology, not an omission here, and inventing a neck edge would
            // change what the format means; a caller who needs head-body
            // coupling should add that edge deliberately.
            SkeletonType::MediaPipe33 => (
                33,
                &[
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 7),
                    (0, 4),
                    (4, 5),
                    (5, 6),
                    (6, 8),
                    (9, 10),
                    (11, 12),
                    (11, 13),
                    (13, 15),
                    (15, 17),
                    (15, 19),
                    (15, 21),
                    (17, 19),
                    (12, 14),
                    (14, 16),
                    (16, 18),
                    (16, 20),
                    (16, 22),
                    (18, 20),
                    (11, 23),
                    (12, 24),
                    (23, 24),
                    (23, 25),
                    (25, 27),
                    (27, 29),
                    (27, 31),
                    (29, 31),
                    (24, 26),
                    (26, 28),
                    (28, 30),
                    (28, 32),
                    (30, 32),
                ],
            ),
        };

        let mut adj = Array2::zeros((num_nodes, num_nodes));
        for &(i, j) in edges {
            adj[[i, j]] = 1.0;
            adj[[j, i]] = 1.0;
        }
        (num_nodes, adj)
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

    // ---- skeleton adjacency --------------------------------------------

    /// Every layout must describe a real skeleton: symmetric, self-loop free,
    /// with every joint attached to something.
    ///
    /// Two of the three used to return an all-zero matrix, which leaves the
    /// graph convolution with no neighbours to gather from -- a GCN that has
    /// quietly become a per-node MLP.
    #[test]
    fn every_skeleton_has_a_connected_graph() {
        for skeleton in [
            SkeletonType::COCO18,
            SkeletonType::OpenPose25,
            SkeletonType::MediaPipe33,
        ] {
            let (n, adj) = skeleton.get_adjacency();
            assert_eq!(adj.shape(), &[n, n], "{skeleton:?}: wrong shape");

            let total: f32 = adj.iter().sum();
            assert!(total > 0.0, "{skeleton:?}: adjacency is entirely zero");

            for i in 0..n {
                assert_eq!(adj[[i, i]], 0.0, "{skeleton:?}: node {i} links to itself");
                let degree: f32 = adj.row(i).sum();
                assert!(
                    degree > 0.0,
                    "{skeleton:?}: node {i} is isolated, so it can never exchange \
                     information with the rest of the skeleton"
                );
                for j in 0..n {
                    assert_eq!(
                        adj[[i, j]],
                        adj[[j, i]],
                        "{skeleton:?}: edge {i}-{j} is not symmetric"
                    );
                }
            }
        }
    }

    /// The whole skeleton must be one connected component: a limb that is not
    /// reachable from the torso cannot influence it, whatever the depth.
    #[test]
    fn every_skeleton_is_a_single_component() {
        for skeleton in [
            SkeletonType::COCO18,
            SkeletonType::OpenPose25,
            SkeletonType::MediaPipe33,
        ] {
            let (n, adj) = skeleton.get_adjacency();
            let mut seen = vec![false; n];
            let mut stack = vec![0usize];
            seen[0] = true;
            while let Some(node) = stack.pop() {
                for j in 0..n {
                    if adj[[node, j]] > 0.0 && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            let unreachable: Vec<usize> = (0..n).filter(|&i| !seen[i]).collect();
            // MediaPipe's published POSE_CONNECTIONS is genuinely in three
            // pieces -- face, mouth pair, body -- so starting from the nose
            // reaches only the face. COCO and OpenPose are single skeletons.
            let expected: Vec<usize> = match skeleton {
                SkeletonType::MediaPipe33 => (9..33).collect(),
                _ => vec![],
            };
            assert_eq!(
                unreachable, expected,
                "{skeleton:?}: unexpected disconnected nodes"
            );
        }
    }

    /// Limb chains must pass through their middle joint, and must not cross
    /// between limbs -- the two ways COCO18's edge list was wrong.
    #[test]
    fn coco18_limbs_are_anatomical() {
        let (_, adj) = SkeletonType::COCO18.get_adjacency();
        let linked = |a: usize, b: usize| adj[[a, b]] > 0.0;

        // Right arm: neck-shoulder-elbow-wrist, each link present...
        for (a, b) in [(1, 2), (2, 3), (3, 4)] {
            assert!(linked(a, b), "right arm is missing the link {a}-{b}");
        }
        // ...and no shortcut past the elbow.
        assert!(
            !linked(2, 4),
            "shoulder links straight to wrist, skipping the elbow"
        );

        // Left arm likewise.
        for (a, b) in [(1, 5), (5, 6), (6, 7)] {
            assert!(linked(a, b), "left arm is missing the link {a}-{b}");
        }
        assert!(!linked(5, 7), "left shoulder links straight to left wrist");

        // Legs pass through the knee.
        assert!(linked(8, 9) && linked(9, 10), "right leg chain is broken");
        assert!(!linked(8, 10), "right hip links straight to right ankle");

        // No cross-limb edges: right elbow to left shoulder, right hip to left
        // elbow, right knee to left hip were all present.
        for (a, b) in [(3, 5), (8, 6), (9, 11)] {
            assert!(!linked(a, b), "cross-limb edge {a}-{b} is present");
        }
    }
}
