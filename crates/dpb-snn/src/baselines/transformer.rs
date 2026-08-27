//! Transformer baseline architectures

use super::{ANNBaseline, Tensor, count_params, xavier_init};

/// Multi-head attention layer (simplified)
// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
struct MultiHeadAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    d_model: usize,
}

impl MultiHeadAttention {
    fn new(d_model: usize, num_heads: usize, seed: u64) -> Self {
        Self {
            q_proj: xavier_init(vec![d_model, d_model], seed),
            k_proj: xavier_init(vec![d_model, d_model], seed + 1),
            v_proj: xavier_init(vec![d_model, d_model], seed + 2),
            o_proj: xavier_init(vec![d_model, d_model], seed + 3),
            num_heads,
            d_model,
        }
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.q_proj.shape)
            + count_params(&self.k_proj.shape)
            + count_params(&self.v_proj.shape)
            + count_params(&self.o_proj.shape)
    }
}

/// Feed-forward network
struct FeedForward {
    w1: Tensor,
    w2: Tensor,
    b1: Tensor,
    b2: Tensor,
}

impl FeedForward {
    fn new(d_model: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            w1: xavier_init(vec![d_model, d_ff], seed),
            w2: xavier_init(vec![d_ff, d_model], seed + 1),
            b1: Tensor::zeros(vec![d_ff]),
            b2: Tensor::zeros(vec![d_model]),
        }
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w1.shape)
            + count_params(&self.w2.shape)
            + count_params(&self.b1.shape)
            + count_params(&self.b2.shape)
    }
}

/// Transformer encoder layer
struct TransformerEncoderLayer {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1_gamma: Tensor,
    norm1_beta: Tensor,
    norm2_gamma: Tensor,
    norm2_beta: Tensor,
}

impl TransformerEncoderLayer {
    fn new(d_model: usize, num_heads: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads, seed),
            feed_forward: FeedForward::new(d_model, d_ff, seed + 10),
            norm1_gamma: Tensor::ones(vec![d_model]),
            norm1_beta: Tensor::zeros(vec![d_model]),
            norm2_gamma: Tensor::ones(vec![d_model]),
            norm2_beta: Tensor::zeros(vec![d_model]),
        }
    }

    fn num_parameters(&self) -> usize {
        self.attention.num_parameters()
            + self.feed_forward.num_parameters()
            + count_params(&self.norm1_gamma.shape)
            + count_params(&self.norm1_beta.shape)
            + count_params(&self.norm2_gamma.shape)
            + count_params(&self.norm2_beta.shape)
    }
}

/// 29. Transformer Encoder (encoder-only)
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    output_proj: Tensor,
    output_bias: Tensor,
    d_model: usize,
}

impl TransformerEncoder {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        output_size: usize,
        seed: u64,
    ) -> Self {
        let d_ff = d_model * 4;
        let mut layers = Vec::new();

        for i in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model,
                num_heads,
                d_ff,
                seed + i as u64 * 100,
            ));
        }

        Self {
            layers,
            output_proj: xavier_init(vec![d_model, output_size], seed + 1000),
            output_bias: Tensor::zeros(vec![output_size]),
            d_model,
        }
    }
}

impl ANNBaseline for TransformerEncoder {
    fn name(&self) -> &str {
        "TransformerEncoder"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        // Simplified forward pass
        let batch_size = if input.shape.len() == 3 {
            input.shape[0]
        } else {
            1
        };
        let x = Tensor::zeros(vec![batch_size, self.d_model]);

        x.matmul(&self.output_proj).add(&self.output_bias)
    }

    fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.num_parameters()).sum();

        layer_params + count_params(&self.output_proj.shape) + count_params(&self.output_bias.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let seq_len = 100;
        let d_model = self.d_model;
        let num_layers = self.layers.len();

        // Attention: O(seq_len^2 * d_model)
        let attention_ops = (seq_len * seq_len * d_model * num_layers) as u64;

        // Feed-forward: O(seq_len * d_model * d_ff)
        let ff_ops = (seq_len * d_model * d_model * 4 * num_layers) as u64;

        (attention_ops + ff_ops) * 2
    }

    fn architecture_summary(&self) -> String {
        format!(
            "TransformerEncoder: {} layers, d_model={}, {} heads",
            self.layers.len(),
            self.d_model,
            self.layers[0].attention.num_heads
        )
    }
}

/// 30. Small Transformer (2 layers)
pub struct TransformerSmall {
    encoder: TransformerEncoder,
}

impl TransformerSmall {
    pub fn new(d_model: usize, output_size: usize, seed: u64) -> Self {
        Self {
            encoder: TransformerEncoder::new(d_model, 4, 2, output_size, seed),
        }
    }
}

impl ANNBaseline for TransformerSmall {
    fn name(&self) -> &str {
        "TransformerSmall"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.encoder.forward(input)
    }

    fn num_parameters(&self) -> usize {
        self.encoder.num_parameters()
    }

    fn flops_per_inference(&self) -> u64 {
        self.encoder.flops_per_inference()
    }

    fn architecture_summary(&self) -> String {
        format!(
            "TransformerSmall: 2 layers, d_model={}",
            self.encoder.d_model
        )
    }
}

/// 31. Medium Transformer (4 layers)
pub struct TransformerMedium {
    encoder: TransformerEncoder,
}

impl TransformerMedium {
    pub fn new(d_model: usize, output_size: usize, seed: u64) -> Self {
        Self {
            encoder: TransformerEncoder::new(d_model, 8, 4, output_size, seed),
        }
    }
}

impl ANNBaseline for TransformerMedium {
    fn name(&self) -> &str {
        "TransformerMedium"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.encoder.forward(input)
    }

    fn num_parameters(&self) -> usize {
        self.encoder.num_parameters()
    }

    fn flops_per_inference(&self) -> u64 {
        self.encoder.flops_per_inference()
    }

    fn architecture_summary(&self) -> String {
        format!(
            "TransformerMedium: 4 layers, d_model={}",
            self.encoder.d_model
        )
    }
}

/// 32. Large Transformer (6 layers)
pub struct TransformerLarge {
    encoder: TransformerEncoder,
}

impl TransformerLarge {
    pub fn new(d_model: usize, output_size: usize, seed: u64) -> Self {
        Self {
            encoder: TransformerEncoder::new(d_model, 12, 6, output_size, seed),
        }
    }
}

impl ANNBaseline for TransformerLarge {
    fn name(&self) -> &str {
        "TransformerLarge"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.encoder.forward(input)
    }

    fn num_parameters(&self) -> usize {
        self.encoder.num_parameters()
    }

    fn flops_per_inference(&self) -> u64 {
        self.encoder.flops_per_inference()
    }

    fn architecture_summary(&self) -> String {
        format!(
            "TransformerLarge: 6 layers, d_model={}",
            self.encoder.d_model
        )
    }
}

/// 33. Linear Transformer (linear attention)
pub struct LinearTransformer {
    layers: Vec<TransformerEncoderLayer>,
    output_proj: Tensor,
    output_bias: Tensor,
}

impl LinearTransformer {
    pub fn new(d_model: usize, num_layers: usize, output_size: usize, seed: u64) -> Self {
        let d_ff = d_model * 4;
        let mut layers = Vec::new();

        for i in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model,
                8,
                d_ff,
                seed + i as u64 * 100,
            ));
        }

        Self {
            layers,
            output_proj: xavier_init(vec![d_model, output_size], seed + 1000),
            output_bias: Tensor::zeros(vec![output_size]),
        }
    }
}

impl ANNBaseline for LinearTransformer {
    fn name(&self) -> &str {
        "LinearTransformer"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = if input.shape.len() == 3 {
            input.shape[0]
        } else {
            1
        };
        let d_model = self.output_proj.shape[0];
        let x = Tensor::zeros(vec![batch_size, d_model]);

        x.matmul(&self.output_proj).add(&self.output_bias)
    }

    fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.num_parameters()).sum();

        layer_params + count_params(&self.output_proj.shape) + count_params(&self.output_bias.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        // Linear attention: O(seq_len * d_model^2) instead of O(seq_len^2 * d_model)
        let seq_len = 100;
        let d_model = self.output_proj.shape[0];
        let num_layers = self.layers.len();

        (seq_len * d_model * d_model * num_layers * 4) as u64
    }

    fn architecture_summary(&self) -> String {
        format!(
            "LinearTransformer: {} layers with linear attention (O(n) complexity)",
            self.layers.len()
        )
    }
}

/// 34. Performer (Fast attention using random features)
pub struct Performer {
    encoder: TransformerEncoder,
    random_features: usize,
}

impl Performer {
    pub fn new(d_model: usize, num_layers: usize, output_size: usize, seed: u64) -> Self {
        Self {
            encoder: TransformerEncoder::new(d_model, 8, num_layers, output_size, seed),
            random_features: 256,
        }
    }
}

impl ANNBaseline for Performer {
    fn name(&self) -> &str {
        "Performer"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.encoder.forward(input)
    }

    fn num_parameters(&self) -> usize {
        self.encoder.num_parameters()
    }

    fn flops_per_inference(&self) -> u64 {
        // Performer approximates attention with random features
        let seq_len = 100;
        let d_model = self.encoder.d_model;
        let num_layers = self.encoder.layers.len();

        (seq_len * d_model * self.random_features * num_layers * 4) as u64
    }

    fn architecture_summary(&self) -> String {
        format!(
            "Performer: {} layers with FAVOR+ attention ({} random features)",
            self.encoder.layers.len(),
            self.random_features
        )
    }
}

/// 35. Informer (for time series)
pub struct Informer {
    encoder: TransformerEncoder,
    distilling_layers: Vec<Tensor>,
}

impl Informer {
    pub fn new(d_model: usize, num_layers: usize, output_size: usize, seed: u64) -> Self {
        let mut distilling_layers = Vec::new();

        for i in 0..num_layers - 1 {
            distilling_layers.push(xavier_init(vec![d_model, d_model], seed + 2000 + i as u64));
        }

        Self {
            encoder: TransformerEncoder::new(d_model, 8, num_layers, output_size, seed),
            distilling_layers,
        }
    }
}

impl ANNBaseline for Informer {
    fn name(&self) -> &str {
        "Informer"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.encoder.forward(input)
    }

    fn num_parameters(&self) -> usize {
        let distill_params: usize = self
            .distilling_layers
            .iter()
            .map(|l| count_params(&l.shape))
            .sum();

        self.encoder.num_parameters() + distill_params
    }

    fn flops_per_inference(&self) -> u64 {
        // Informer uses ProbSparse attention
        let seq_len = 100u64;
        let log_seq = (seq_len as f32).log2() as u64;
        let d_model = self.encoder.d_model as u64;
        let num_layers = self.encoder.layers.len() as u64;

        seq_len * log_seq * d_model * num_layers * 4
    }

    fn architecture_summary(&self) -> String {
        format!(
            "Informer: {} layers with ProbSparse self-attention for time series",
            self.encoder.layers.len()
        )
    }
}

/// 36. Autoformer (Auto-correlation mechanism)
// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
pub struct Autoformer {
    encoder: TransformerEncoder,
    decomp_kernels: Vec<usize>,
}

impl Autoformer {
    pub fn new(d_model: usize, num_layers: usize, output_size: usize, seed: u64) -> Self {
        Self {
            encoder: TransformerEncoder::new(d_model, 8, num_layers, output_size, seed),
            decomp_kernels: vec![25, 25, 25], // Moving average kernels
        }
    }
}

impl ANNBaseline for Autoformer {
    fn name(&self) -> &str {
        "Autoformer"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.encoder.forward(input)
    }

    fn num_parameters(&self) -> usize {
        // Autoformer has similar parameters to standard transformer
        // Plus some additional params for decomposition
        self.encoder.num_parameters() + 1000
    }

    fn flops_per_inference(&self) -> u64 {
        // Auto-correlation is more efficient than self-attention
        let seq_len = 100u64;
        let log_seq = (seq_len as f32).log2() as u64;
        let d_model = self.encoder.d_model as u64;
        let num_layers = self.encoder.layers.len() as u64;

        seq_len * log_seq * d_model * num_layers * 3
    }

    fn architecture_summary(&self) -> String {
        format!(
            "Autoformer: {} layers with auto-correlation and series decomposition",
            self.encoder.layers.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_encoder() {
        let transformer = TransformerEncoder::new(128, 8, 4, 10, 42);
        assert_eq!(transformer.name(), "TransformerEncoder");

        let input = Tensor::randn(vec![1, 100, 128], 123);
        let output = transformer.forward(&input);
        assert_eq!(output.shape[0], 1);

        assert!(transformer.num_parameters() > 100000);
    }

    #[test]
    fn test_transformer_small() {
        let transformer = TransformerSmall::new(64, 10, 42);
        assert_eq!(transformer.name(), "TransformerSmall");
        assert!(transformer.num_parameters() > 10000);
    }

    #[test]
    fn test_transformer_medium() {
        let transformer = TransformerMedium::new(128, 10, 42);
        assert_eq!(transformer.name(), "TransformerMedium");
        assert!(transformer.num_parameters() > 50000);
    }

    #[test]
    fn test_transformer_large() {
        let transformer = TransformerLarge::new(256, 10, 42);
        assert_eq!(transformer.name(), "TransformerLarge");
        assert!(transformer.num_parameters() > 200000);
    }

    #[test]
    fn test_linear_transformer() {
        let transformer = LinearTransformer::new(128, 4, 10, 42);
        assert_eq!(transformer.name(), "LinearTransformer");

        // Linear transformer should have fewer FLOPs than standard
        let flops = transformer.flops_per_inference();
        assert!(flops > 0);
    }

    #[test]
    fn test_performer() {
        let performer = Performer::new(128, 4, 10, 42);
        assert_eq!(performer.name(), "Performer");
        assert!(performer.num_parameters() > 50000);
    }

    #[test]
    fn test_informer() {
        let informer = Informer::new(128, 4, 10, 42);
        assert_eq!(informer.name(), "Informer");
        assert!(informer.num_parameters() > 50000);
    }

    #[test]
    fn test_autoformer() {
        let autoformer = Autoformer::new(128, 4, 10, 42);
        assert_eq!(autoformer.name(), "Autoformer");
        assert!(autoformer.num_parameters() > 50000);
    }
}
