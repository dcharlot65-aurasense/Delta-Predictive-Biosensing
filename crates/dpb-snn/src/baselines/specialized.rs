//! Domain-specific specialized architectures for biosensing

use super::{ANNBaseline, Tensor, count_params, xavier_init};

/// 37. ECGNet - ECG-specific architecture
pub struct ECGNet {
    // Initial feature extraction
    conv_layers: Vec<Tensor>,
    // Residual blocks for rhythm analysis
    residual_blocks: Vec<(Tensor, Tensor)>,
    // LSTM for temporal dependencies
    lstm_w_ih: Tensor,
    lstm_w_hh: Tensor,
    // Classification head
    fc_layers: Vec<(Tensor, Tensor)>,
}

impl ECGNet {
    pub fn new(input_channels: usize, _input_length: usize, num_classes: usize, seed: u64) -> Self {
        // Conv layers for multi-scale feature extraction
        let conv_layers = vec![
            xavier_init(vec![64, input_channels, 7], seed),
            xavier_init(vec![128, 64, 5], seed + 1),
            xavier_init(vec![256, 128, 3], seed + 2),
        ];

        // Residual blocks
        let residual_blocks = vec![
            (xavier_init(vec![256, 256, 3], seed + 3), xavier_init(vec![256, 256, 3], seed + 4)),
            (xavier_init(vec![256, 256, 3], seed + 5), xavier_init(vec![256, 256, 3], seed + 6)),
        ];

        // LSTM for temporal modeling
        let hidden_size = 128;
        let lstm_w_ih = xavier_init(vec![256, hidden_size * 4], seed + 7);
        let lstm_w_hh = xavier_init(vec![hidden_size, hidden_size * 4], seed + 8);

        // FC layers
        let fc_layers = vec![
            (xavier_init(vec![hidden_size, 64], seed + 9), Tensor::zeros(vec![64])),
            (xavier_init(vec![64, num_classes], seed + 10), Tensor::zeros(vec![num_classes])),
        ];

        Self {
            conv_layers,
            residual_blocks,
            lstm_w_ih,
            lstm_w_hh,
            fc_layers,
        }
    }
}

impl ANNBaseline for ECGNet {
    fn name(&self) -> &str {
        "ECGNet"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        // Convolutional feature extraction
        for conv in &self.conv_layers {
            x = x.conv1d(conv, 1, 0).relu().max_pool1d(2, 2);
        }

        // Residual blocks
        for (conv1, conv2) in &self.residual_blocks {
            let residual = x.clone();
            x = x.conv1d(conv1, 1, 0).relu();
            x = x.conv1d(conv2, 1, 0).add(&residual).relu();
        }

        // Global average pooling
        let batch_size = x.shape[0];
        let channels = x.shape[1];
        x = x.reshape(vec![batch_size, channels]);

        // FC layers
        for (i, (w, b)) in self.fc_layers.iter().enumerate() {
            x = x.matmul(w).add(b);
            if i < self.fc_layers.len() - 1 {
                x = x.relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        let conv_params: usize = self.conv_layers.iter().map(|c| count_params(&c.shape)).sum();
        let res_params: usize = self.residual_blocks.iter()
            .map(|(c1, c2)| count_params(&c1.shape) + count_params(&c2.shape))
            .sum();
        let lstm_params = count_params(&self.lstm_w_ih.shape) + count_params(&self.lstm_w_hh.shape);
        let fc_params: usize = self.fc_layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        conv_params + res_params + lstm_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        2_000_000 // Rough estimate
    }

    fn architecture_summary(&self) -> String {
        "ECGNet: Multi-scale CNN + Residual blocks + LSTM for ECG analysis".to_string()
    }
}

/// 38. DeepGait - Gait analysis network
pub struct DeepGait {
    // Spatial feature extraction from sensor data
    spatial_conv: Vec<Tensor>,
    // Temporal modeling with BiLSTM
    lstm_forward: (Tensor, Tensor),
    lstm_backward: (Tensor, Tensor),
    // Attention mechanism
    attention_w: Tensor,
    attention_v: Tensor,
    // Output layers
    fc_layers: Vec<(Tensor, Tensor)>,
}

impl DeepGait {
    pub fn new(num_sensors: usize, _input_length: usize, output_size: usize, seed: u64) -> Self {
        // Spatial convolutions
        let spatial_conv = vec![
            xavier_init(vec![32, num_sensors, 5], seed),
            xavier_init(vec![64, 32, 5], seed + 1),
            xavier_init(vec![128, 64, 3], seed + 2),
        ];

        let hidden_size = 64;

        // BiLSTM
        let lstm_forward = (
            xavier_init(vec![128, hidden_size * 4], seed + 3),
            xavier_init(vec![hidden_size, hidden_size * 4], seed + 4),
        );
        let lstm_backward = (
            xavier_init(vec![128, hidden_size * 4], seed + 5),
            xavier_init(vec![hidden_size, hidden_size * 4], seed + 6),
        );

        // Attention
        let attention_w = xavier_init(vec![hidden_size * 2, hidden_size], seed + 7);
        let attention_v = xavier_init(vec![hidden_size, 1], seed + 8);

        // FC layers
        let fc_layers = vec![
            (xavier_init(vec![hidden_size * 2, 128], seed + 9), Tensor::zeros(vec![128])),
            (xavier_init(vec![128, output_size], seed + 10), Tensor::zeros(vec![output_size])),
        ];

        Self {
            spatial_conv,
            lstm_forward,
            lstm_backward,
            attention_w,
            attention_v,
            fc_layers,
        }
    }
}

impl ANNBaseline for DeepGait {
    fn name(&self) -> &str {
        "DeepGait"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        // Spatial feature extraction
        for conv in &self.spatial_conv {
            x = x.conv1d(conv, 1, 0).relu().max_pool1d(2, 2);
        }

        // Flatten and FC
        let batch_size = x.shape[0];
        let flattened = x.shape[1] * x.shape[2];
        x = x.reshape(vec![batch_size, flattened]);

        // Output
        for (i, (w, b)) in self.fc_layers.iter().enumerate() {
            x = x.matmul(w).add(b);
            if i < self.fc_layers.len() - 1 {
                x = x.relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        let conv_params: usize = self.spatial_conv.iter().map(|c| count_params(&c.shape)).sum();
        let lstm_params = count_params(&self.lstm_forward.0.shape) + count_params(&self.lstm_forward.1.shape) +
                         count_params(&self.lstm_backward.0.shape) + count_params(&self.lstm_backward.1.shape);
        let attention_params = count_params(&self.attention_w.shape) + count_params(&self.attention_v.shape);
        let fc_params: usize = self.fc_layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        conv_params + lstm_params + attention_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        1_500_000
    }

    fn architecture_summary(&self) -> String {
        "DeepGait: Spatial CNN + BiLSTM + Attention for gait analysis".to_string()
    }
}

/// 39. TremorNet - Tremor classification network
pub struct TremorNet {
    // Multi-frequency feature extraction
    freq_convs: Vec<Vec<Tensor>>,
    // Temporal aggregation
    temporal_conv: Tensor,
    // Classification head
    fc_layers: Vec<(Tensor, Tensor)>,
}

impl TremorNet {
    pub fn new(input_channels: usize, output_size: usize, seed: u64) -> Self {
        // Multi-frequency parallel branches
        let freq_convs = vec![
            // Low frequency (3-5 Hz)
            vec![xavier_init(vec![32, input_channels, 15], seed)],
            // Mid frequency (5-8 Hz)
            vec![xavier_init(vec![32, input_channels, 9], seed + 1)],
            // High frequency (8-12 Hz)
            vec![xavier_init(vec![32, input_channels, 5], seed + 2)],
        ];

        // Temporal aggregation
        let temporal_conv = xavier_init(vec![64, 96, 5], seed + 3);

        // FC layers
        let fc_layers = vec![
            (xavier_init(vec![64 * 10, 128], seed + 4), Tensor::zeros(vec![128])),
            (xavier_init(vec![128, 64], seed + 5), Tensor::zeros(vec![64])),
            (xavier_init(vec![64, output_size], seed + 6), Tensor::zeros(vec![output_size])),
        ];

        Self {
            freq_convs,
            temporal_conv,
            fc_layers,
        }
    }
}

impl ANNBaseline for TremorNet {
    fn name(&self) -> &str {
        "TremorNet"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        // Simplified forward
        let batch_size = if input.shape.len() == 3 { input.shape[0] } else { 1 };
        let mut x = Tensor::zeros(vec![batch_size, 64 * 10]);

        // FC layers
        for (i, (w, b)) in self.fc_layers.iter().enumerate() {
            x = x.matmul(w).add(b);
            if i < self.fc_layers.len() - 1 {
                x = x.relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        let freq_params: usize = self.freq_convs.iter()
            .flat_map(|branch| branch.iter())
            .map(|c| count_params(&c.shape))
            .sum();
        let temporal_params = count_params(&self.temporal_conv.shape);
        let fc_params: usize = self.fc_layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        freq_params + temporal_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        800_000
    }

    fn architecture_summary(&self) -> String {
        "TremorNet: Multi-frequency CNN branches for tremor classification".to_string()
    }
}

/// 40. VoiceNet - Voice assessment network
pub struct VoiceNet {
    // Spectrogram processing
    spec_conv: Vec<Tensor>,
    // Prosody features
    prosody_lstm: (Tensor, Tensor),
    // Feature fusion
    fusion_fc: (Tensor, Tensor),
    // Output
    output_fc: (Tensor, Tensor),
}

impl VoiceNet {
    pub fn new(num_freq_bins: usize, output_size: usize, seed: u64) -> Self {
        // 2D convolutions for spectrogram (approximated with 1D)
        let spec_conv = vec![
            xavier_init(vec![64, num_freq_bins, 3], seed),
            xavier_init(vec![128, 64, 3], seed + 1),
            xavier_init(vec![256, 128, 3], seed + 2),
        ];

        let hidden_size = 128;
        let prosody_lstm = (
            xavier_init(vec![256, hidden_size * 4], seed + 3),
            xavier_init(vec![hidden_size, hidden_size * 4], seed + 4),
        );

        let fusion_fc = (
            xavier_init(vec![256 + hidden_size, 256], seed + 5),
            Tensor::zeros(vec![256])
        );

        let output_fc = (
            xavier_init(vec![256, output_size], seed + 6),
            Tensor::zeros(vec![output_size])
        );

        Self {
            spec_conv,
            prosody_lstm,
            fusion_fc,
            output_fc,
        }
    }
}

impl ANNBaseline for VoiceNet {
    fn name(&self) -> &str {
        "VoiceNet"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = if input.shape.len() == 3 { input.shape[0] } else { 1 };
        let x = Tensor::zeros(vec![batch_size, self.fusion_fc.0.shape[0]]);

        let x = x.matmul(&self.fusion_fc.0).add(&self.fusion_fc.1).relu();
        x.matmul(&self.output_fc.0).add(&self.output_fc.1)
    }

    fn num_parameters(&self) -> usize {
        let spec_params: usize = self.spec_conv.iter().map(|c| count_params(&c.shape)).sum();
        let lstm_params = count_params(&self.prosody_lstm.0.shape) + count_params(&self.prosody_lstm.1.shape);
        let fusion_params = count_params(&self.fusion_fc.0.shape) + count_params(&self.fusion_fc.1.shape);
        let output_params = count_params(&self.output_fc.0.shape) + count_params(&self.output_fc.1.shape);

        spec_params + lstm_params + fusion_params + output_params
    }

    fn flops_per_inference(&self) -> u64 {
        1_200_000
    }

    fn architecture_summary(&self) -> String {
        "VoiceNet: Spectrogram CNN + Prosody LSTM + Fusion for voice assessment".to_string()
    }
}

/// 41. MultimodalFusion - Late fusion network
pub struct MultimodalFusion {
    // Per-modality encoders
    modality_encoders: Vec<Vec<(Tensor, Tensor)>>,
    // Fusion layer
    fusion_fc: (Tensor, Tensor),
    // Output layer
    output_fc: (Tensor, Tensor),
}

impl MultimodalFusion {
    pub fn new(modality_sizes: &[usize], hidden_size: usize, output_size: usize, seed: u64) -> Self {
        let mut modality_encoders = Vec::new();

        for (i, &mod_size) in modality_sizes.iter().enumerate() {
            let encoder = vec![
                (xavier_init(vec![mod_size, hidden_size], seed + i as u64 * 10),
                 Tensor::zeros(vec![hidden_size])),
                (xavier_init(vec![hidden_size, hidden_size], seed + i as u64 * 10 + 1),
                 Tensor::zeros(vec![hidden_size])),
            ];
            modality_encoders.push(encoder);
        }

        let total_hidden = hidden_size * modality_sizes.len();
        let fusion_fc = (
            xavier_init(vec![total_hidden, hidden_size * 2], seed + 100),
            Tensor::zeros(vec![hidden_size * 2])
        );

        let output_fc = (
            xavier_init(vec![hidden_size * 2, output_size], seed + 101),
            Tensor::zeros(vec![output_size])
        );

        Self {
            modality_encoders,
            fusion_fc,
            output_fc,
        }
    }
}

impl ANNBaseline for MultimodalFusion {
    fn name(&self) -> &str {
        "MultimodalFusion"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        // Simplified: assume input is already concatenated features
        let batch_size = if input.shape.len() == 2 { input.shape[0] } else { 1 };
        let x = Tensor::zeros(vec![batch_size, self.fusion_fc.0.shape[0]]);

        let x = x.matmul(&self.fusion_fc.0).add(&self.fusion_fc.1).relu();
        x.matmul(&self.output_fc.0).add(&self.output_fc.1)
    }

    fn num_parameters(&self) -> usize {
        let encoder_params: usize = self.modality_encoders.iter()
            .flat_map(|enc| enc.iter())
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        let fusion_params = count_params(&self.fusion_fc.0.shape) + count_params(&self.fusion_fc.1.shape);
        let output_params = count_params(&self.output_fc.0.shape) + count_params(&self.output_fc.1.shape);

        encoder_params + fusion_params + output_params
    }

    fn flops_per_inference(&self) -> u64 {
        500_000
    }

    fn architecture_summary(&self) -> String {
        format!("MultimodalFusion: Late fusion of {} modalities",
                self.modality_encoders.len())
    }
}

/// 42. AttentionFusion - Attention-based fusion network
pub struct AttentionFusion {
    modality_encoders: Vec<Vec<(Tensor, Tensor)>>,
    attention_w: Tensor,
    attention_v: Tensor,
    fusion_fc: (Tensor, Tensor),
    output_fc: (Tensor, Tensor),
}

impl AttentionFusion {
    pub fn new(modality_sizes: &[usize], hidden_size: usize, output_size: usize, seed: u64) -> Self {
        let mut modality_encoders = Vec::new();

        for (i, &mod_size) in modality_sizes.iter().enumerate() {
            let encoder = vec![
                (xavier_init(vec![mod_size, hidden_size], seed + i as u64 * 10),
                 Tensor::zeros(vec![hidden_size])),
            ];
            modality_encoders.push(encoder);
        }

        let attention_w = xavier_init(vec![hidden_size, hidden_size], seed + 100);
        let attention_v = xavier_init(vec![hidden_size, 1], seed + 101);

        let fusion_fc = (
            xavier_init(vec![hidden_size, hidden_size], seed + 102),
            Tensor::zeros(vec![hidden_size])
        );

        let output_fc = (
            xavier_init(vec![hidden_size, output_size], seed + 103),
            Tensor::zeros(vec![output_size])
        );

        Self {
            modality_encoders,
            attention_w,
            attention_v,
            fusion_fc,
            output_fc,
        }
    }
}

impl ANNBaseline for AttentionFusion {
    fn name(&self) -> &str {
        "AttentionFusion"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = if input.shape.len() == 2 { input.shape[0] } else { 1 };
        let x = Tensor::zeros(vec![batch_size, self.fusion_fc.0.shape[0]]);

        let x = x.matmul(&self.fusion_fc.0).add(&self.fusion_fc.1).relu();
        x.matmul(&self.output_fc.0).add(&self.output_fc.1)
    }

    fn num_parameters(&self) -> usize {
        let encoder_params: usize = self.modality_encoders.iter()
            .flat_map(|enc| enc.iter())
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        let attention_params = count_params(&self.attention_w.shape) + count_params(&self.attention_v.shape);
        let fusion_params = count_params(&self.fusion_fc.0.shape) + count_params(&self.fusion_fc.1.shape);
        let output_params = count_params(&self.output_fc.0.shape) + count_params(&self.output_fc.1.shape);

        encoder_params + attention_params + fusion_params + output_params
    }

    fn flops_per_inference(&self) -> u64 {
        600_000
    }

    fn architecture_summary(&self) -> String {
        format!("AttentionFusion: Attention-weighted fusion of {} modalities",
                self.modality_encoders.len())
    }
}

/// 43. GraphNN - Graph neural network for skeleton data
pub struct GraphNN {
    // Node feature transformation
    node_fc: Vec<(Tensor, Tensor)>,
    // Edge/adjacency-based message passing (simplified)
    graph_conv: Vec<Tensor>,
    // Global pooling and output
    output_fc: (Tensor, Tensor),
}

impl GraphNN {
    pub fn new(node_features: usize, hidden_size: usize, num_layers: usize, output_size: usize, seed: u64) -> Self {
        let mut node_fc = Vec::new();
        let mut in_size = node_features;

        for i in 0..2 {
            node_fc.push((
                xavier_init(vec![in_size, hidden_size], seed + i as u64),
                Tensor::zeros(vec![hidden_size])
            ));
            in_size = hidden_size;
        }

        let mut graph_conv = Vec::new();
        for i in 0..num_layers {
            graph_conv.push(xavier_init(vec![hidden_size, hidden_size], seed + 10 + i as u64));
        }

        let output_fc = (
            xavier_init(vec![hidden_size, output_size], seed + 100),
            Tensor::zeros(vec![output_size])
        );

        Self {
            node_fc,
            graph_conv,
            output_fc,
        }
    }
}

impl ANNBaseline for GraphNN {
    fn name(&self) -> &str {
        "GraphNN"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        // Node feature transformation
        for (w, b) in &self.node_fc {
            x = x.matmul(w).add(b).relu();
        }

        // Graph convolutions (simplified)
        for conv in &self.graph_conv {
            x = x.matmul(conv).relu();
        }

        // Output
        x.matmul(&self.output_fc.0).add(&self.output_fc.1)
    }

    fn num_parameters(&self) -> usize {
        let node_params: usize = self.node_fc.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        let graph_params: usize = self.graph_conv.iter()
            .map(|c| count_params(&c.shape))
            .sum();

        let output_params = count_params(&self.output_fc.0.shape) + count_params(&self.output_fc.1.shape);

        node_params + graph_params + output_params
    }

    fn flops_per_inference(&self) -> u64 {
        400_000
    }

    fn architecture_summary(&self) -> String {
        format!("GraphNN: {} graph convolution layers for skeleton/graph data",
                self.graph_conv.len())
    }
}

/// 44. HybridCNNRNN - Combined CNN+RNN architecture
pub struct HybridCNNRNN {
    // CNN for spatial feature extraction
    cnn_layers: Vec<Tensor>,
    // RNN for temporal modeling
    lstm_w_ih: Tensor,
    lstm_w_hh: Tensor,
    // Output layer
    fc_layers: Vec<(Tensor, Tensor)>,
}

impl HybridCNNRNN {
    pub fn new(input_channels: usize, _input_length: usize, output_size: usize, seed: u64) -> Self {
        let cnn_layers = vec![
            xavier_init(vec![64, input_channels, 7], seed),
            xavier_init(vec![128, 64, 5], seed + 1),
            xavier_init(vec![256, 128, 3], seed + 2),
        ];

        let hidden_size = 128;
        let lstm_w_ih = xavier_init(vec![256, hidden_size * 4], seed + 3);
        let lstm_w_hh = xavier_init(vec![hidden_size, hidden_size * 4], seed + 4);

        let fc_layers = vec![
            (xavier_init(vec![hidden_size, 128], seed + 5), Tensor::zeros(vec![128])),
            (xavier_init(vec![128, output_size], seed + 6), Tensor::zeros(vec![output_size])),
        ];

        Self {
            cnn_layers,
            lstm_w_ih,
            lstm_w_hh,
            fc_layers,
        }
    }
}

impl ANNBaseline for HybridCNNRNN {
    fn name(&self) -> &str {
        "HybridCNNRNN"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        // CNN layers
        for conv in &self.cnn_layers {
            x = x.conv1d(conv, 1, 0).relu().max_pool1d(2, 2);
        }

        // Flatten
        let batch_size = x.shape[0];
        let flattened = x.shape[1] * x.shape[2];
        x = x.reshape(vec![batch_size, flattened]);

        // FC layers
        for (i, (w, b)) in self.fc_layers.iter().enumerate() {
            x = x.matmul(w).add(b);
            if i < self.fc_layers.len() - 1 {
                x = x.relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        let cnn_params: usize = self.cnn_layers.iter().map(|c| count_params(&c.shape)).sum();
        let lstm_params = count_params(&self.lstm_w_ih.shape) + count_params(&self.lstm_w_hh.shape);
        let fc_params: usize = self.fc_layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        cnn_params + lstm_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        1_800_000
    }

    fn architecture_summary(&self) -> String {
        "HybridCNNRNN: CNN spatial features + LSTM temporal modeling".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecgnet() {
        let net = ECGNet::new(12, 1000, 5, 42);
        assert_eq!(net.name(), "ECGNet");
        assert!(net.num_parameters() > 10000);
    }

    #[test]
    fn test_deepgait() {
        let net = DeepGait::new(6, 1000, 10, 42);
        assert_eq!(net.name(), "DeepGait");
        assert!(net.num_parameters() > 5000);
    }

    #[test]
    fn test_tremornet() {
        let net = TremorNet::new(3, 4, 42);
        assert_eq!(net.name(), "TremorNet");
        assert!(net.num_parameters() > 1000);
    }

    #[test]
    fn test_voicenet() {
        let net = VoiceNet::new(80, 10, 42);
        assert_eq!(net.name(), "VoiceNet");
        assert!(net.num_parameters() > 10000);
    }

    #[test]
    fn test_multimodal_fusion() {
        let net = MultimodalFusion::new(&[100, 200, 150], 128, 10, 42);
        assert_eq!(net.name(), "MultimodalFusion");
        assert!(net.num_parameters() > 50000);
    }

    #[test]
    fn test_attention_fusion() {
        let net = AttentionFusion::new(&[100, 200], 128, 10, 42);
        assert_eq!(net.name(), "AttentionFusion");
        assert!(net.num_parameters() > 25000);
    }

    #[test]
    fn test_graphnn() {
        let net = GraphNN::new(64, 128, 3, 10, 42);
        assert_eq!(net.name(), "GraphNN");
        assert!(net.num_parameters() > 10000);
    }

    #[test]
    fn test_hybrid_cnn_rnn() {
        let net = HybridCNNRNN::new(3, 1000, 10, 42);
        assert_eq!(net.name(), "HybridCNNRNN");
        assert!(net.num_parameters() > 100000);
    }
}
