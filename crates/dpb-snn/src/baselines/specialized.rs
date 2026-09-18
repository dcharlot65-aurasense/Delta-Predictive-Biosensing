//! Domain-specific specialized architectures for biosensing

use super::rnn::{last_step, lstm_sequence, reverse_time};
use super::{ANNBaseline, Tensor, count_params, flops_of, xavier_init};

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
    /// Input shape this model was configured for, so that
    /// `flops_per_inference` can report the work for one real pass.
    input_shape: Vec<usize>,
}

impl ECGNet {
    pub fn new(input_channels: usize, input_length: usize, num_classes: usize, seed: u64) -> Self {
        // Conv layers for multi-scale feature extraction
        let conv_layers = vec![
            xavier_init(vec![64, input_channels, 7], seed),
            xavier_init(vec![128, 64, 5], seed + 1),
            xavier_init(vec![256, 128, 3], seed + 2),
        ];

        // Residual blocks
        let residual_blocks = vec![
            (
                xavier_init(vec![256, 256, 3], seed + 3),
                xavier_init(vec![256, 256, 3], seed + 4),
            ),
            (
                xavier_init(vec![256, 256, 3], seed + 5),
                xavier_init(vec![256, 256, 3], seed + 6),
            ),
        ];

        // LSTM for temporal modeling
        let hidden_size = 128;
        let lstm_w_ih = xavier_init(vec![256, hidden_size * 4], seed + 7);
        let lstm_w_hh = xavier_init(vec![hidden_size, hidden_size * 4], seed + 8);

        // FC layers
        let fc_layers = vec![
            (
                xavier_init(vec![hidden_size, 64], seed + 9),
                Tensor::zeros(vec![64]),
            ),
            (
                xavier_init(vec![64, num_classes], seed + 10),
                Tensor::zeros(vec![num_classes]),
            ),
        ];

        Self {
            input_shape: vec![1, input_channels, input_length],
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

        // Convolutions keep their length so the residual blocks below can add
        // their input back; with zero padding the two branches differed and the
        // addition panicked on any input.
        for conv in &self.conv_layers {
            let pad = conv.shape[2] / 2;
            x = x.conv1d(conv, 1, pad).relu().max_pool1d(2, 2);
        }

        for (conv1, conv2) in &self.residual_blocks {
            let residual = x.clone();
            x = x.conv1d(conv1, 1, conv1.shape[2] / 2).relu();
            x = x.conv1d(conv2, 1, conv2.shape[2] / 2).add(&residual).relu();
        }

        // Temporal modelling over the convolutional features. The LSTM weights
        // were allocated and counted but never used: the old code reshaped
        // straight to [batch, channels] and called it global average pooling,
        // which is only valid when the sequence is already one sample long.
        let sequence = super::channels_to_time(&x);
        let states = lstm_sequence(&sequence, &self.lstm_w_ih, &self.lstm_w_hh);
        let mut x = last_step(&states);

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
        let conv_params: usize = self
            .conv_layers
            .iter()
            .map(|c| count_params(&c.shape))
            .sum();
        let res_params: usize = self
            .residual_blocks
            .iter()
            .map(|(c1, c2)| count_params(&c1.shape) + count_params(&c2.shape))
            .sum();
        let lstm_params = count_params(&self.lstm_w_ih.shape) + count_params(&self.lstm_w_hh.shape);
        let fc_params: usize = self
            .fc_layers
            .iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        conv_params + res_params + lstm_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        // Counted, not estimated: the tensor operations tally the
        // multiply-accumulates a real forward pass performs, at two FLOPs
        // each. This used to be a constant, which cannot be right for a
        // convolution -- it did not move when the input got longer.
        flops_of(|| {
            let _ = self.forward(&Tensor::zeros(self.input_shape.clone()));
        })
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
    /// Input shape this model was configured for, so that
    /// `flops_per_inference` can report the work for one real pass.
    input_shape: Vec<usize>,
}

impl DeepGait {
    pub fn new(num_sensors: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
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
            (
                xavier_init(vec![hidden_size * 2, 128], seed + 9),
                Tensor::zeros(vec![128]),
            ),
            (
                xavier_init(vec![128, output_size], seed + 10),
                Tensor::zeros(vec![output_size]),
            ),
        ];

        Self {
            input_shape: vec![1, num_sensors, input_length],
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

        for conv in &self.spatial_conv {
            let pad = conv.shape[2] / 2;
            x = x.conv1d(conv, 1, pad).relu().max_pool1d(2, 2);
        }

        // Bidirectional recurrence over the spatial features, then attention
        // across time. Both the backward LSTM and the attention weights used to
        // be stored, counted, and never read: the old path flattened the
        // convolution output straight into the classifier.
        let sequence = super::channels_to_time(&x);
        let forward = lstm_sequence(&sequence, &self.lstm_forward.0, &self.lstm_forward.1);
        let backward = lstm_sequence(
            &reverse_time(&sequence),
            &self.lstm_backward.0,
            &self.lstm_backward.1,
        );
        let backward = reverse_time(&backward);

        let (batch_size, steps, hidden) = (forward.shape[0], forward.shape[1], forward.shape[2]);
        let mut joined = vec![0.0; batch_size * steps * hidden * 2];
        for b in 0..batch_size {
            for t in 0..steps {
                let dst = (b * steps + t) * hidden * 2;
                let src = (b * steps + t) * hidden;
                joined[dst..dst + hidden].copy_from_slice(&forward.data[src..src + hidden]);
                joined[dst + hidden..dst + 2 * hidden]
                    .copy_from_slice(&backward.data[src..src + hidden]);
            }
        }
        let joined = Tensor::from_vec(joined, vec![batch_size, steps, hidden * 2]);

        // Additive attention over time steps.
        let flat = joined.reshape(vec![batch_size * steps, hidden * 2]);
        let scores = flat
            .matmul(&self.attention_w)
            .tanh()
            .matmul(&self.attention_v);
        let mut context = vec![0.0; batch_size * hidden * 2];
        for b in 0..batch_size {
            let at = |t: usize| scores.data[b * steps + t];
            let max = (0..steps).map(at).fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = (0..steps).map(|t| (at(t) - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            for (t, w) in weights.iter().enumerate() {
                let alpha = w / total;
                for j in 0..hidden * 2 {
                    context[b * hidden * 2 + j] +=
                        alpha * joined.data[(b * steps + t) * hidden * 2 + j];
                }
            }
        }
        let mut x = Tensor::from_vec(context, vec![batch_size, hidden * 2]);

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
        let conv_params: usize = self
            .spatial_conv
            .iter()
            .map(|c| count_params(&c.shape))
            .sum();
        let lstm_params = count_params(&self.lstm_forward.0.shape)
            + count_params(&self.lstm_forward.1.shape)
            + count_params(&self.lstm_backward.0.shape)
            + count_params(&self.lstm_backward.1.shape);
        let attention_params =
            count_params(&self.attention_w.shape) + count_params(&self.attention_v.shape);
        let fc_params: usize = self
            .fc_layers
            .iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        conv_params + lstm_params + attention_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        // Counted, not estimated: the tensor operations tally the
        // multiply-accumulates a real forward pass performs, at two FLOPs
        // each. This used to be a constant, which cannot be right for a
        // convolution -- it did not move when the input got longer.
        flops_of(|| {
            let _ = self.forward(&Tensor::zeros(self.input_shape.clone()));
        })
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
    /// Input shape this model was configured for, so that
    /// `flops_per_inference` can report the work for one real pass.
    input_shape: Vec<usize>,
}

impl TremorNet {
    pub fn new(input_channels: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
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
            (
                xavier_init(vec![64 * 10, 128], seed + 4),
                Tensor::zeros(vec![128]),
            ),
            (
                xavier_init(vec![128, 64], seed + 5),
                Tensor::zeros(vec![64]),
            ),
            (
                xavier_init(vec![64, output_size], seed + 6),
                Tensor::zeros(vec![output_size]),
            ),
        ];

        Self {
            input_shape: vec![1, input_channels, input_length],
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
        // Three parallel branches with different kernel widths, which is how
        // this model separates tremor bands; their outputs are concatenated
        // along channels and aggregated in time. All of it used to be skipped:
        // the forward pass started from a zero vector, so the branch and
        // aggregation weights never multiplied anything.
        let branch_outputs: Vec<Tensor> = self
            .freq_convs
            .iter()
            .map(|branch| {
                let mut y = input.clone();
                for conv in branch {
                    y = y.conv1d(conv, 1, conv.shape[2] / 2).relu();
                }
                y
            })
            .collect();

        let batch_size = branch_outputs[0].shape[0];
        let length = branch_outputs[0].shape[2];
        let total_channels: usize = branch_outputs.iter().map(|b| b.shape[1]).sum();
        let mut merged = vec![0.0; batch_size * total_channels * length];
        let mut offset = 0;
        for branch in &branch_outputs {
            let channels = branch.shape[1];
            for b in 0..batch_size {
                for c in 0..channels {
                    let src = (b * channels + c) * length;
                    let dst = (b * total_channels + offset + c) * length;
                    merged[dst..dst + length].copy_from_slice(&branch.data[src..src + length]);
                }
            }
            offset += channels;
        }
        let merged = Tensor::from_vec(merged, vec![batch_size, total_channels, length]);

        let aggregated = merged
            .conv1d(&self.temporal_conv, 1, self.temporal_conv.shape[2] / 2)
            .relu();
        // The classifier expects ten positions per channel however long the
        // recording is.
        let pooled = super::adaptive_avg_pool1d(&aggregated, 10);
        let mut x = pooled.reshape(vec![batch_size, pooled.shape[1] * 10]);

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
        let freq_params: usize = self
            .freq_convs
            .iter()
            .flat_map(|branch| branch.iter())
            .map(|c| count_params(&c.shape))
            .sum();
        let temporal_params = count_params(&self.temporal_conv.shape);
        let fc_params: usize = self
            .fc_layers
            .iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        freq_params + temporal_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        // Counted, not estimated: the tensor operations tally the
        // multiply-accumulates a real forward pass performs, at two FLOPs
        // each. This used to be a constant, which cannot be right for a
        // convolution -- it did not move when the input got longer.
        flops_of(|| {
            let _ = self.forward(&Tensor::zeros(self.input_shape.clone()));
        })
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
    /// Input shape this model was configured for, so that
    /// `flops_per_inference` can report the work for one real pass.
    input_shape: Vec<usize>,
}

impl VoiceNet {
    pub fn new(num_freq_bins: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
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
            Tensor::zeros(vec![256]),
        );

        let output_fc = (
            xavier_init(vec![256, output_size], seed + 6),
            Tensor::zeros(vec![output_size]),
        );

        Self {
            input_shape: vec![1, num_freq_bins, input_length],
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
        // Two branches over the same spectrogram: pooled spectral features,
        // and a recurrent prosody summary. The fusion layer is sized for both
        // concatenated, which is why it is 256 + hidden wide. Neither branch
        // used to run -- the forward pass began at a zero vector.
        let mut spectral = input.clone();
        for conv in &self.spec_conv {
            let pad = conv.shape[2] / 2;
            spectral = spectral.conv1d(conv, 1, pad).relu().max_pool1d(2, 2);
        }

        let sequence = super::channels_to_time(&spectral);
        let prosody = last_step(&lstm_sequence(
            &sequence,
            &self.prosody_lstm.0,
            &self.prosody_lstm.1,
        ));
        let pooled = super::global_avg_pool1d(&spectral);

        let batch_size = pooled.shape[0];
        let (spec_dim, prosody_dim) = (pooled.shape[1], prosody.shape[1]);
        let mut fused = vec![0.0; batch_size * (spec_dim + prosody_dim)];
        for b in 0..batch_size {
            let dst = b * (spec_dim + prosody_dim);
            fused[dst..dst + spec_dim]
                .copy_from_slice(&pooled.data[b * spec_dim..(b + 1) * spec_dim]);
            fused[dst + spec_dim..dst + spec_dim + prosody_dim]
                .copy_from_slice(&prosody.data[b * prosody_dim..(b + 1) * prosody_dim]);
        }
        let fused = Tensor::from_vec(fused, vec![batch_size, spec_dim + prosody_dim]);

        let x = fused
            .matmul(&self.fusion_fc.0)
            .add(&self.fusion_fc.1)
            .relu();
        x.matmul(&self.output_fc.0).add(&self.output_fc.1)
    }

    fn num_parameters(&self) -> usize {
        let spec_params: usize = self.spec_conv.iter().map(|c| count_params(&c.shape)).sum();
        let lstm_params =
            count_params(&self.prosody_lstm.0.shape) + count_params(&self.prosody_lstm.1.shape);
        let fusion_params =
            count_params(&self.fusion_fc.0.shape) + count_params(&self.fusion_fc.1.shape);
        let output_params =
            count_params(&self.output_fc.0.shape) + count_params(&self.output_fc.1.shape);

        spec_params + lstm_params + fusion_params + output_params
    }

    fn flops_per_inference(&self) -> u64 {
        // Counted, not estimated: the tensor operations tally the
        // multiply-accumulates a real forward pass performs, at two FLOPs
        // each. This used to be a constant, which cannot be right for a
        // convolution -- it did not move when the input got longer.
        flops_of(|| {
            let _ = self.forward(&Tensor::zeros(self.input_shape.clone()));
        })
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
    /// Input shape this model was configured for, so that
    /// `flops_per_inference` can report the work for one real pass.
    input_shape: Vec<usize>,
}

impl MultimodalFusion {
    pub fn new(
        modality_sizes: &[usize],
        hidden_size: usize,
        output_size: usize,
        seed: u64,
    ) -> Self {
        let mut modality_encoders = Vec::new();

        for (i, &mod_size) in modality_sizes.iter().enumerate() {
            let encoder = vec![
                (
                    xavier_init(vec![mod_size, hidden_size], seed + i as u64 * 10),
                    Tensor::zeros(vec![hidden_size]),
                ),
                (
                    xavier_init(vec![hidden_size, hidden_size], seed + i as u64 * 10 + 1),
                    Tensor::zeros(vec![hidden_size]),
                ),
            ];
            modality_encoders.push(encoder);
        }

        let total_hidden = hidden_size * modality_sizes.len();
        let fusion_fc = (
            xavier_init(vec![total_hidden, hidden_size * 2], seed + 100),
            Tensor::zeros(vec![hidden_size * 2]),
        );

        let output_fc = (
            xavier_init(vec![hidden_size * 2, output_size], seed + 101),
            Tensor::zeros(vec![output_size]),
        );

        Self {
            input_shape: vec![1, modality_sizes.iter().sum::<usize>()],
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
        // The input is the modalities concatenated along the feature axis, in
        // the order the encoders were built. Each is sliced out, encoded, and
        // the results concatenated again for fusion. None of this used to
        // happen: the forward pass started from a zero vector, so every
        // encoder was stored, counted, and never applied.
        let batch_size = input.shape[0];
        let encoded = encode_modalities(input, &self.modality_encoders);
        let x = concat_features(&encoded, batch_size);

        let x = x.matmul(&self.fusion_fc.0).add(&self.fusion_fc.1).relu();
        x.matmul(&self.output_fc.0).add(&self.output_fc.1)
    }

    fn num_parameters(&self) -> usize {
        let encoder_params: usize = self
            .modality_encoders
            .iter()
            .flat_map(|enc| enc.iter())
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        let fusion_params =
            count_params(&self.fusion_fc.0.shape) + count_params(&self.fusion_fc.1.shape);
        let output_params =
            count_params(&self.output_fc.0.shape) + count_params(&self.output_fc.1.shape);

        encoder_params + fusion_params + output_params
    }

    fn flops_per_inference(&self) -> u64 {
        // Counted, not estimated: the tensor operations tally the
        // multiply-accumulates a real forward pass performs, at two FLOPs
        // each. This used to be a constant, which cannot be right for a
        // convolution -- it did not move when the input got longer.
        flops_of(|| {
            let _ = self.forward(&Tensor::zeros(self.input_shape.clone()));
        })
    }

    fn architecture_summary(&self) -> String {
        format!(
            "MultimodalFusion: Late fusion of {} modalities",
            self.modality_encoders.len()
        )
    }
}

/// 42. AttentionFusion - Attention-based fusion network
pub struct AttentionFusion {
    modality_encoders: Vec<Vec<(Tensor, Tensor)>>,
    attention_w: Tensor,
    attention_v: Tensor,
    fusion_fc: (Tensor, Tensor),
    output_fc: (Tensor, Tensor),
    /// Input shape this model was configured for, so that
    /// `flops_per_inference` can report the work for one real pass.
    input_shape: Vec<usize>,
}

impl AttentionFusion {
    pub fn new(
        modality_sizes: &[usize],
        hidden_size: usize,
        output_size: usize,
        seed: u64,
    ) -> Self {
        let mut modality_encoders = Vec::new();

        for (i, &mod_size) in modality_sizes.iter().enumerate() {
            let encoder = vec![(
                xavier_init(vec![mod_size, hidden_size], seed + i as u64 * 10),
                Tensor::zeros(vec![hidden_size]),
            )];
            modality_encoders.push(encoder);
        }

        let attention_w = xavier_init(vec![hidden_size, hidden_size], seed + 100);
        let attention_v = xavier_init(vec![hidden_size, 1], seed + 101);

        let fusion_fc = (
            xavier_init(vec![hidden_size, hidden_size], seed + 102),
            Tensor::zeros(vec![hidden_size]),
        );

        let output_fc = (
            xavier_init(vec![hidden_size, output_size], seed + 103),
            Tensor::zeros(vec![output_size]),
        );

        Self {
            input_shape: vec![1, modality_sizes.iter().sum::<usize>()],
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
        // Attention decides how much each modality contributes, which is the
        // whole difference from plain concatenation fusion. The encoders and
        // both attention matrices used to be unreachable.
        let batch_size = input.shape[0];
        let encoded = encode_modalities(input, &self.modality_encoders);
        let hidden = encoded[0].shape[1];

        let mut context = vec![0.0; batch_size * hidden];
        for b in 0..batch_size {
            let scores: Vec<f32> = encoded
                .iter()
                .map(|e| {
                    let row = Tensor::from_vec(
                        e.data[b * hidden..(b + 1) * hidden].to_vec(),
                        vec![1, hidden],
                    );
                    row.matmul(&self.attention_w)
                        .tanh()
                        .matmul(&self.attention_v)
                        .data[0]
                })
                .collect();
            let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            for (m, w) in weights.iter().enumerate() {
                let alpha = w / total;
                for j in 0..hidden {
                    context[b * hidden + j] += alpha * encoded[m].data[b * hidden + j];
                }
            }
        }
        let x = Tensor::from_vec(context, vec![batch_size, hidden]);

        let x = x.matmul(&self.fusion_fc.0).add(&self.fusion_fc.1).relu();
        x.matmul(&self.output_fc.0).add(&self.output_fc.1)
    }

    fn num_parameters(&self) -> usize {
        let encoder_params: usize = self
            .modality_encoders
            .iter()
            .flat_map(|enc| enc.iter())
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        let attention_params =
            count_params(&self.attention_w.shape) + count_params(&self.attention_v.shape);
        let fusion_params =
            count_params(&self.fusion_fc.0.shape) + count_params(&self.fusion_fc.1.shape);
        let output_params =
            count_params(&self.output_fc.0.shape) + count_params(&self.output_fc.1.shape);

        encoder_params + attention_params + fusion_params + output_params
    }

    fn flops_per_inference(&self) -> u64 {
        // Counted, not estimated: the tensor operations tally the
        // multiply-accumulates a real forward pass performs, at two FLOPs
        // each. This used to be a constant, which cannot be right for a
        // convolution -- it did not move when the input got longer.
        flops_of(|| {
            let _ = self.forward(&Tensor::zeros(self.input_shape.clone()));
        })
    }

    fn architecture_summary(&self) -> String {
        format!(
            "AttentionFusion: Attention-weighted fusion of {} modalities",
            self.modality_encoders.len()
        )
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
    /// Input shape this model was configured for, so that
    /// `flops_per_inference` can report the work for one real pass.
    input_shape: Vec<usize>,
}

impl GraphNN {
    pub fn new(
        node_features: usize,
        num_nodes: usize,
        hidden_size: usize,
        num_layers: usize,
        output_size: usize,
        seed: u64,
    ) -> Self {
        let mut node_fc = Vec::new();
        let mut in_size = node_features;

        for i in 0..2 {
            node_fc.push((
                xavier_init(vec![in_size, hidden_size], seed + i as u64),
                Tensor::zeros(vec![hidden_size]),
            ));
            in_size = hidden_size;
        }

        let mut graph_conv = Vec::new();
        for i in 0..num_layers {
            graph_conv.push(xavier_init(
                vec![hidden_size, hidden_size],
                seed + 10 + i as u64,
            ));
        }

        let output_fc = (
            xavier_init(vec![hidden_size, output_size], seed + 100),
            Tensor::zeros(vec![output_size]),
        );

        Self {
            input_shape: vec![1, num_nodes, node_features],
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
        // Input is [batch, nodes, features]. Node-wise layers act on each node
        // independently, so the node axis is folded into the batch for them;
        // `matmul` is rank-2 only, and passing a rank-3 tensor used to panic.
        let (batch_size, nodes, features) = match input.shape.len() {
            3 => (input.shape[0], input.shape[1], input.shape[2]),
            2 => (input.shape[0], 1, input.shape[1]),
            other => panic!("GraphNN needs rank 2 or 3 input, got rank {other}"),
        };
        let mut x = input.reshape(vec![batch_size * nodes, features]);

        for (w, b) in &self.node_fc {
            x = x.matmul(w).add(b).relu();
        }

        // Graph convolution proper: transform, then average over each node's
        // neighbourhood. No adjacency reaches this interface, so the graph is
        // taken to be complete, which makes the aggregation a mean over all
        // nodes -- the standard reading when structure is unknown. Without the
        // aggregation step these were plain per-node matmuls and nothing about
        // them was graph-like.
        let hidden = x.shape[1];
        for conv in &self.graph_conv {
            let transformed = x.matmul(conv);
            let mut aggregated = vec![0.0; batch_size * nodes * hidden];
            for b in 0..batch_size {
                for j in 0..hidden {
                    let mut sum = 0.0;
                    for n in 0..nodes {
                        sum += transformed.data[(b * nodes + n) * hidden + j];
                    }
                    let mean = sum / nodes as f32;
                    for n in 0..nodes {
                        aggregated[(b * nodes + n) * hidden + j] = mean;
                    }
                }
            }
            x = Tensor::from_vec(aggregated, vec![batch_size * nodes, hidden]).relu();
        }

        // Readout: mean over nodes gives one vector per graph.
        let mut pooled = vec![0.0; batch_size * hidden];
        for b in 0..batch_size {
            for j in 0..hidden {
                let mut sum = 0.0;
                for n in 0..nodes {
                    sum += x.data[(b * nodes + n) * hidden + j];
                }
                pooled[b * hidden + j] = sum / nodes as f32;
            }
        }
        let pooled = Tensor::from_vec(pooled, vec![batch_size, hidden]);

        pooled.matmul(&self.output_fc.0).add(&self.output_fc.1)
    }

    fn num_parameters(&self) -> usize {
        let node_params: usize = self
            .node_fc
            .iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        let graph_params: usize = self.graph_conv.iter().map(|c| count_params(&c.shape)).sum();

        let output_params =
            count_params(&self.output_fc.0.shape) + count_params(&self.output_fc.1.shape);

        node_params + graph_params + output_params
    }

    fn flops_per_inference(&self) -> u64 {
        // Counted, not estimated: the tensor operations tally the
        // multiply-accumulates a real forward pass performs, at two FLOPs
        // each. This used to be a constant, which cannot be right for a
        // convolution -- it did not move when the input got longer.
        flops_of(|| {
            let _ = self.forward(&Tensor::zeros(self.input_shape.clone()));
        })
    }

    fn architecture_summary(&self) -> String {
        format!(
            "GraphNN: {} graph convolution layers for skeleton/graph data",
            self.graph_conv.len()
        )
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
    /// Input shape this model was configured for, so that
    /// `flops_per_inference` can report the work for one real pass.
    input_shape: Vec<usize>,
}

impl HybridCNNRNN {
    pub fn new(input_channels: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
        let cnn_layers = vec![
            xavier_init(vec![64, input_channels, 7], seed),
            xavier_init(vec![128, 64, 5], seed + 1),
            xavier_init(vec![256, 128, 3], seed + 2),
        ];

        let hidden_size = 128;
        let lstm_w_ih = xavier_init(vec![256, hidden_size * 4], seed + 3);
        let lstm_w_hh = xavier_init(vec![hidden_size, hidden_size * 4], seed + 4);

        let fc_layers = vec![
            (
                xavier_init(vec![hidden_size, 128], seed + 5),
                Tensor::zeros(vec![128]),
            ),
            (
                xavier_init(vec![128, output_size], seed + 6),
                Tensor::zeros(vec![output_size]),
            ),
        ];

        Self {
            input_shape: vec![1, input_channels, input_length],
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

        for conv in &self.cnn_layers {
            let pad = conv.shape[2] / 2;
            x = x.conv1d(conv, 1, pad).relu().max_pool1d(2, 2);
        }

        // The recurrent half of a model called Hybrid CNN-RNN. Its weights were
        // allocated and counted, and the old forward pass flattened the
        // convolution output straight into a classifier sized for the LSTM,
        // which is why it panicked.
        let sequence = super::channels_to_time(&x);
        let mut x = last_step(&lstm_sequence(&sequence, &self.lstm_w_ih, &self.lstm_w_hh));

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
        let fc_params: usize = self
            .fc_layers
            .iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        cnn_params + lstm_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        // Counted, not estimated: the tensor operations tally the
        // multiply-accumulates a real forward pass performs, at two FLOPs
        // each. This used to be a constant, which cannot be right for a
        // convolution -- it did not move when the input got longer.
        flops_of(|| {
            let _ = self.forward(&Tensor::zeros(self.input_shape.clone()));
        })
    }

    fn architecture_summary(&self) -> String {
        "HybridCNNRNN: CNN spatial features + LSTM temporal modeling".to_string()
    }
}

/// Splits a concatenated feature vector into per-modality slices and runs each
/// modality's encoder stack over its own slice.
fn encode_modalities(input: &Tensor, encoders: &[Vec<(Tensor, Tensor)>]) -> Vec<Tensor> {
    let batch_size = input.shape[0];
    let features = input.shape[1];
    let expected: usize = encoders.iter().map(|e| e[0].0.shape[0]).sum();
    assert_eq!(
        features, expected,
        "input has {features} features but the modalities need {expected}"
    );

    let mut offset = 0;
    let mut out = Vec::with_capacity(encoders.len());
    for encoder in encoders {
        let width = encoder[0].0.shape[0];
        let mut slice = vec![0.0; batch_size * width];
        for b in 0..batch_size {
            let src = b * features + offset;
            slice[b * width..(b + 1) * width].copy_from_slice(&input.data[src..src + width]);
        }
        let mut x = Tensor::from_vec(slice, vec![batch_size, width]);
        for (w, bias) in encoder {
            x = x.matmul(w).add(bias).relu();
        }
        out.push(x);
        offset += width;
    }
    out
}

/// Concatenates per-modality encodings along the feature axis.
fn concat_features(parts: &[Tensor], batch_size: usize) -> Tensor {
    let total: usize = parts.iter().map(|p| p.shape[1]).sum();
    let mut data = vec![0.0; batch_size * total];
    let mut offset = 0;
    for part in parts {
        let width = part.shape[1];
        for b in 0..batch_size {
            let dst = b * total + offset;
            data[dst..dst + width].copy_from_slice(&part.data[b * width..(b + 1) * width]);
        }
        offset += width;
    }
    Tensor::from_vec(data, vec![batch_size, total])
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
        let net = TremorNet::new(3, 256, 4, 42);
        assert_eq!(net.name(), "TremorNet");
        assert!(net.num_parameters() > 1000);
    }

    #[test]
    fn test_voicenet() {
        let net = VoiceNet::new(80, 128, 10, 42);
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
        let net = GraphNN::new(64, 16, 128, 3, 10, 42);
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
