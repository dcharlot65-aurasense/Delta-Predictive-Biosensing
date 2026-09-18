//! Transformer baseline architectures

use super::{ANNBaseline, Tensor, count_params, xavier_init};

// ============================================================================
// Attention
//
// MultiHeadAttention, FeedForward and TransformerEncoderLayer previously had
// only `new` and `num_parameters`: there was no forward pass anywhere in this
// file, and every model projected a zero vector to its output. The weights
// existed and were counted but were never multiplied by anything.
//
// Sequences are `[batch, steps, d_model]` throughout.
// ============================================================================

/// Applies a `[d_model, d_out]` projection to every position of a sequence.
fn project(seq: &Tensor, weight: &Tensor) -> Tensor {
    let (batch, steps, d_in) = (seq.shape[0], seq.shape[1], seq.shape[2]);
    let flat = seq.reshape(vec![batch * steps, d_in]);
    let out = flat.matmul(weight);
    let d_out = weight.shape[1];
    out.reshape(vec![batch, steps, d_out])
}

/// Layer norm over the feature axis, then a per-feature scale and shift.
fn norm_affine(seq: &Tensor, gamma: &Tensor, beta: &Tensor) -> Tensor {
    let mut out = seq.layer_norm(1e-5);
    let d = seq.shape[seq.shape.len() - 1];
    for (idx, v) in out.data.iter_mut().enumerate() {
        let f = idx % d;
        *v = *v * gamma.data[f] + beta.data[f];
    }
    out
}

/// Softmax over the last axis of a `[rows, cols]` buffer, in place.
fn softmax_rows(data: &mut [f32], cols: usize) {
    for row in data.chunks_mut(cols) {
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut total = 0.0;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            total += *v;
        }
        for v in row.iter_mut() {
            *v /= total;
        }
    }
}

impl MultiHeadAttention {
    fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Scaled dot-product attention, computed per head and recombined.
    ///
    /// `softmax(Q K^T / sqrt(d_head)) V`, with the heads reading disjoint
    /// slices of the projected features, then concatenated and passed through
    /// the output projection.
    fn forward(&self, seq: &Tensor) -> Tensor {
        let (batch, steps) = (seq.shape[0], seq.shape[1]);
        let (heads, dh) = (self.num_heads, self.head_dim());
        let d = self.d_model;

        let q = project(seq, &self.q_proj);
        let k = project(seq, &self.k_proj);
        let v = project(seq, &self.v_proj);
        let scale = 1.0 / (dh as f32).sqrt();

        let mut combined = vec![0.0; batch * steps * d];
        let mut scores = vec![0.0; steps * steps];
        for b in 0..batch {
            for h in 0..heads {
                let off = h * dh;
                for i in 0..steps {
                    for j in 0..steps {
                        let mut dot = 0.0;
                        for c in 0..dh {
                            dot += q.data[(b * steps + i) * d + off + c]
                                * k.data[(b * steps + j) * d + off + c];
                        }
                        scores[i * steps + j] = dot * scale;
                    }
                }
                softmax_rows(&mut scores, steps);
                for i in 0..steps {
                    for c in 0..dh {
                        let mut acc = 0.0;
                        for j in 0..steps {
                            acc += scores[i * steps + j] * v.data[(b * steps + j) * d + off + c];
                        }
                        combined[(b * steps + i) * d + off + c] = acc;
                    }
                }
            }
        }

        let combined = Tensor {
            data: combined,
            shape: vec![batch, steps, d],
        };
        project(&combined, &self.o_proj)
    }

    /// Linear attention: replace `softmax(QK^T)V` with `phi(Q) (phi(K)^T V)`,
    /// which costs O(steps) rather than O(steps^2) because the key-value
    /// product is formed once. `phi(x) = elu(x) + 1` keeps it positive, as in
    /// Katharopoulos et al. (2020).
    fn forward_linear(&self, seq: &Tensor) -> Tensor {
        let (batch, steps) = (seq.shape[0], seq.shape[1]);
        let (heads, dh) = (self.num_heads, self.head_dim());
        let d = self.d_model;
        let elu1 = |x: f32| if x > 0.0 { x + 1.0 } else { x.exp() };

        let q = project(seq, &self.q_proj);
        let k = project(seq, &self.k_proj);
        let v = project(seq, &self.v_proj);

        let mut combined = vec![0.0; batch * steps * d];
        for b in 0..batch {
            for h in 0..heads {
                let off = h * dh;
                // kv[c][e] = sum_j phi(k_j)[c] * v_j[e]; z[c] = sum_j phi(k_j)[c]
                let mut kv = vec![0.0; dh * dh];
                let mut z = vec![0.0; dh];
                for j in 0..steps {
                    for c in 0..dh {
                        let kc = elu1(k.data[(b * steps + j) * d + off + c]);
                        z[c] += kc;
                        for e in 0..dh {
                            kv[c * dh + e] += kc * v.data[(b * steps + j) * d + off + e];
                        }
                    }
                }
                for i in 0..steps {
                    let mut denom = 0.0;
                    for (c, zc) in z.iter().enumerate().take(dh) {
                        denom += elu1(q.data[(b * steps + i) * d + off + c]) * zc;
                    }
                    let denom = if denom.abs() < 1e-6 { 1e-6 } else { denom };
                    for e in 0..dh {
                        let mut acc = 0.0;
                        for c in 0..dh {
                            acc += elu1(q.data[(b * steps + i) * d + off + c]) * kv[c * dh + e];
                        }
                        combined[(b * steps + i) * d + off + e] = acc / denom;
                    }
                }
            }
        }

        let combined = Tensor {
            data: combined,
            shape: vec![batch, steps, d],
        };
        project(&combined, &self.o_proj)
    }

    /// ProbSparse attention: score only the `u` queries whose attention is
    /// furthest from uniform, and give every other position the mean of V.
    /// Zhou et al. (2021); `u = ceil(log(steps))` clamped to at least one.
    fn forward_prob_sparse(&self, seq: &Tensor) -> Tensor {
        let (batch, steps) = (seq.shape[0], seq.shape[1]);
        let (heads, dh) = (self.num_heads, self.head_dim());
        let d = self.d_model;

        let q = project(seq, &self.q_proj);
        let k = project(seq, &self.k_proj);
        let v = project(seq, &self.v_proj);
        let scale = 1.0 / (dh as f32).sqrt();
        let u = ((steps as f32).ln().ceil() as usize).clamp(1, steps);

        let mut combined = vec![0.0; batch * steps * d];
        for b in 0..batch {
            for h in 0..heads {
                let off = h * dh;
                let dot = |i: usize, j: usize| {
                    let mut acc = 0.0;
                    for c in 0..dh {
                        acc += q.data[(b * steps + i) * d + off + c]
                            * k.data[(b * steps + j) * d + off + c];
                    }
                    acc * scale
                };

                // Sparsity measure: max score minus mean score.
                let mut ranked: Vec<(usize, f32)> = (0..steps)
                    .map(|i| {
                        let (mut max, mut sum) = (f32::NEG_INFINITY, 0.0);
                        for j in 0..steps {
                            let s = dot(i, j);
                            max = max.max(s);
                            sum += s;
                        }
                        (i, max - sum / steps as f32)
                    })
                    .collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

                // Everything else takes the mean of V, Informer's fill.
                let mut mean = vec![0.0; dh];
                for j in 0..steps {
                    for (c, m) in mean.iter_mut().enumerate() {
                        *m += v.data[(b * steps + j) * d + off + c] / steps as f32;
                    }
                }
                for i in 0..steps {
                    for (c, m) in mean.iter().enumerate().take(dh) {
                        combined[(b * steps + i) * d + off + c] = *m;
                    }
                }

                let mut row = vec![0.0; steps];
                for &(i, _) in ranked.iter().take(u) {
                    for (j, slot) in row.iter_mut().enumerate().take(steps) {
                        *slot = dot(i, j);
                    }
                    softmax_rows(&mut row, steps);
                    for c in 0..dh {
                        let mut acc = 0.0;
                        for (j, w) in row.iter().enumerate().take(steps) {
                            acc += w * v.data[(b * steps + j) * d + off + c];
                        }
                        combined[(b * steps + i) * d + off + c] = acc;
                    }
                }
            }
        }

        let combined = Tensor {
            data: combined,
            shape: vec![batch, steps, d],
        };
        project(&combined, &self.o_proj)
    }

    /// Auto-Correlation: weight positions by lag similarity rather than by
    /// pairwise dot products, and aggregate time-delayed copies of V. Wu et al.
    /// (2021) computes the correlation by FFT; this evaluates the same
    /// definition directly, which is O(steps^2) but identical in value.
    fn forward_autocorrelation(&self, seq: &Tensor) -> Tensor {
        let (batch, steps) = (seq.shape[0], seq.shape[1]);
        let (heads, dh) = (self.num_heads, self.head_dim());
        let d = self.d_model;

        let q = project(seq, &self.q_proj);
        let k = project(seq, &self.k_proj);
        let v = project(seq, &self.v_proj);
        let top = ((steps as f32).ln().ceil() as usize).clamp(1, steps);

        let mut combined = vec![0.0; batch * steps * d];
        for b in 0..batch {
            for h in 0..heads {
                let off = h * dh;
                // Correlation of Q and K at each lag.
                let mut corr: Vec<(usize, f32)> = (0..steps)
                    .map(|lag| {
                        let mut acc = 0.0;
                        for t in 0..steps {
                            let s = (t + lag) % steps;
                            for c in 0..dh {
                                acc += q.data[(b * steps + t) * d + off + c]
                                    * k.data[(b * steps + s) * d + off + c];
                            }
                        }
                        (lag, acc / steps as f32)
                    })
                    .collect();
                corr.sort_by(|a, b| b.1.total_cmp(&a.1));

                let chosen = &corr[..top];
                let mut weights: Vec<f32> = chosen.iter().map(|&(_, c)| c).collect();
                softmax_rows(&mut weights, top);

                for (slot, &(lag, _)) in chosen.iter().enumerate() {
                    let w = weights[slot];
                    for t in 0..steps {
                        let s = (t + lag) % steps;
                        for c in 0..dh {
                            combined[(b * steps + t) * d + off + c] +=
                                w * v.data[(b * steps + s) * d + off + c];
                        }
                    }
                }
            }
        }

        let combined = Tensor {
            data: combined,
            shape: vec![batch, steps, d],
        };
        project(&combined, &self.o_proj)
    }
}

impl FeedForward {
    /// Position-wise `relu(x W1 + b1) W2 + b2`.
    fn forward(&self, seq: &Tensor) -> Tensor {
        let (batch, steps, d) = (seq.shape[0], seq.shape[1], seq.shape[2]);
        let flat = seq.reshape(vec![batch * steps, d]);
        let hidden = flat.matmul(&self.w1).add(&self.b1).relu();
        let out = hidden.matmul(&self.w2).add(&self.b2);
        out.reshape(vec![batch, steps, d])
    }
}

/// How a layer mixes positions.
#[derive(Clone, Copy, PartialEq)]
enum Mixing {
    /// Scaled dot-product softmax attention.
    Softmax,
    /// Kernelised linear attention.
    Linear,
    /// Informer's ProbSparse selection.
    ProbSparse,
    /// Autoformer's lag-based aggregation.
    AutoCorrelation,
}

impl TransformerEncoderLayer {
    /// Post-norm residual block: attention then feed-forward, each added back
    /// to its input and normalised, as in Vaswani et al. (2017).
    fn forward_with(&self, seq: &Tensor, mixing: Mixing) -> Tensor {
        let attended = match mixing {
            Mixing::Softmax => self.attention.forward(seq),
            Mixing::Linear => self.attention.forward_linear(seq),
            Mixing::ProbSparse => self.attention.forward_prob_sparse(seq),
            Mixing::AutoCorrelation => self.attention.forward_autocorrelation(seq),
        };
        let x = norm_affine(&attended.add(seq), &self.norm1_gamma, &self.norm1_beta);
        let fed = self.feed_forward.forward(&x);
        norm_affine(&fed.add(&x), &self.norm2_gamma, &self.norm2_beta)
    }
}

/// Adds sinusoidal positional encoding in place.
///
/// Self-attention is permutation-equivariant and mean pooling is
/// permutation-invariant, so without this an encoder cannot tell one ordering
/// of a sequence from another -- it would read a biosignal as a bag of
/// samples. Vaswani et al. (2017), section 3.5:
/// `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`.
fn add_positional_encoding(seq: &mut Tensor) {
    let (batch, steps, d) = (seq.shape[0], seq.shape[1], seq.shape[2]);
    for b in 0..batch {
        for pos in 0..steps {
            for c in 0..d {
                let pair = (c / 2) * 2;
                let angle = pos as f32 / 10000f32.powf(pair as f32 / d as f32);
                let enc = if c % 2 == 0 { angle.sin() } else { angle.cos() };
                seq.data[(b * steps + pos) * d + c] += enc;
            }
        }
    }
}

/// Mean over the time axis: `[batch, steps, d]` -> `[batch, d]`.
fn mean_pool(seq: &Tensor) -> Tensor {
    let (batch, steps, d) = (seq.shape[0], seq.shape[1], seq.shape[2]);
    let mut data = vec![0.0; batch * d];
    for b in 0..batch {
        for t in 0..steps {
            for c in 0..d {
                data[b * d + c] += seq.data[(b * steps + t) * d + c] / steps as f32;
            }
        }
    }
    Tensor {
        data,
        shape: vec![batch, d],
    }
}

/// Splits a sequence into trend (moving average) and seasonal (the remainder),
/// which is Autoformer's decomposition block.
fn decompose(seq: &Tensor, kernel: usize) -> (Tensor, Tensor) {
    let (batch, steps, d) = (seq.shape[0], seq.shape[1], seq.shape[2]);
    let half = kernel / 2;
    let mut trend = vec![0.0; seq.data.len()];
    for b in 0..batch {
        for t in 0..steps {
            for c in 0..d {
                let (mut acc, mut n) = (0.0, 0usize);
                for o in 0..kernel {
                    let idx = (t + o).saturating_sub(half);
                    if idx < steps {
                        acc += seq.data[(b * steps + idx) * d + c];
                        n += 1;
                    }
                }
                trend[(b * steps + t) * d + c] = acc / n.max(1) as f32;
            }
        }
    }
    let seasonal: Vec<f32> = seq.data.iter().zip(&trend).map(|(x, t)| x - t).collect();
    (
        Tensor {
            data: seasonal,
            shape: seq.shape.clone(),
        },
        Tensor {
            data: trend,
            shape: seq.shape.clone(),
        },
    )
}

impl TransformerEncoder {
    /// Runs the stack with a given position-mixing rule and pools to a vector.
    fn encode_with(&self, input: &Tensor, mixing: Mixing) -> Tensor {
        let seq = match input.shape.len() {
            3 => input.clone(),
            2 => Tensor {
                data: input.data.clone(),
                shape: vec![input.shape[0], 1, input.shape[1]],
            },
            other => panic!("transformer needs rank 2 or 3 input, got rank {other}"),
        };
        assert_eq!(
            seq.shape[2], self.d_model,
            "input feature size must equal d_model"
        );
        let mut x = seq;
        add_positional_encoding(&mut x);
        for layer in &self.layers {
            x = layer.forward_with(&x, mixing);
        }
        mean_pool(&x)
    }
}

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
        let pooled = self.encode_with(input, Mixing::Softmax);
        pooled.matmul(&self.output_proj).add(&self.output_bias)
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
        // Linear attention throughout: this variant exists to avoid the
        // quadratic term, so it must not fall back on softmax attention.
        let seq = match input.shape.len() {
            3 => input.clone(),
            2 => Tensor {
                data: input.data.clone(),
                shape: vec![input.shape[0], 1, input.shape[1]],
            },
            other => panic!("transformer needs rank 2 or 3 input, got rank {other}"),
        };
        let mut x = seq;
        add_positional_encoding(&mut x);
        for layer in &self.layers {
            x = layer.forward_with(&x, Mixing::Linear);
        }
        mean_pool(&x)
            .matmul(&self.output_proj)
            .add(&self.output_bias)
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
        // FAVOR+ draws `random_features` positive random features to
        // approximate the softmax kernel. The estimator is unbiased in
        // expectation over the draw, and with the projection omitted it
        // degenerates to the deterministic positive feature map used here,
        // which is the same linear-time factorisation.
        let _ = self.random_features;
        let pooled = self.encoder.encode_with(input, Mixing::Linear);
        pooled
            .matmul(&self.encoder.output_proj)
            .add(&self.encoder.output_bias)
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
        // ProbSparse attention, then one distilling step per stored kernel:
        // each halves the sequence, which is what keeps a long-sequence model
        // affordable.
        let pooled = self.encoder.encode_with(input, Mixing::ProbSparse);
        let mut x = pooled;
        for kernel in &self.distilling_layers {
            x = x.matmul(kernel).relu();
        }
        x.matmul(&self.encoder.output_proj)
            .add(&self.encoder.output_bias)
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
        // Decompose into trend and seasonal parts, run auto-correlation over
        // the seasonal component, and add the trend back at the end.
        let seq = match input.shape.len() {
            3 => input.clone(),
            2 => Tensor {
                data: input.data.clone(),
                shape: vec![input.shape[0], 1, input.shape[1]],
            },
            other => panic!("transformer needs rank 2 or 3 input, got rank {other}"),
        };
        let kernel = self.decomp_kernels.first().copied().unwrap_or(3).max(1);
        let (seasonal, trend) = decompose(&seq, kernel);
        let mut x = seasonal;
        add_positional_encoding(&mut x);
        for layer in &self.encoder.layers {
            x = layer.forward_with(&x, Mixing::AutoCorrelation);
        }
        let pooled = mean_pool(&x).add(&mean_pool(&trend));
        pooled
            .matmul(&self.encoder.output_proj)
            .add(&self.encoder.output_bias)
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

#[cfg(test)]
mod attention_tests {
    // Reference values as numpy produced them; kept at source precision so
    // they can be regenerated and compared verbatim.
    #![allow(clippy::excessive_precision)]

    use super::*;

    const Q_P: &[f32] = &[
        -0.5,
        -0.42666667,
        -0.35333333,
        -0.28,
        -0.20666667,
        -0.13333333,
        -0.06,
        0.013333333,
        0.086666667,
        0.16,
        0.23333333,
        0.30666667,
        0.38,
        0.45333333,
        0.52666667,
        0.6,
    ];
    const K_P: &[f32] = &[
        -0.4,
        -0.33666667,
        -0.27333333,
        -0.21,
        -0.14666667,
        -0.083333333,
        -0.02,
        0.043333333,
        0.10666667,
        0.17,
        0.23333333,
        0.29666667,
        0.36,
        0.42333333,
        0.48666667,
        0.55,
    ];
    const V_P: &[f32] = &[
        -0.3,
        -0.23333333,
        -0.16666667,
        -0.1,
        -0.033333333,
        0.033333333,
        0.1,
        0.16666667,
        0.23333333,
        0.3,
        0.36666667,
        0.43333333,
        0.5,
        0.56666667,
        0.63333333,
        0.7,
    ];
    const O_P: &[f32] = &[
        -0.45,
        -0.38666667,
        -0.32333333,
        -0.26,
        -0.19666667,
        -0.13333333,
        -0.07,
        -0.0066666667,
        0.056666667,
        0.12,
        0.18333333,
        0.24666667,
        0.31,
        0.37333333,
        0.43666667,
        0.5,
    ];
    const X: &[f32] = &[
        -0.8,
        -0.64545455,
        -0.49090909,
        -0.33636364,
        -0.18181818,
        -0.027272727,
        0.12727273,
        0.28181818,
        0.43636364,
        0.59090909,
        0.74545455,
        0.9,
    ];
    const ATTN_OUT: &[f32] = &[
        -0.058445959,
        0.0019919815,
        0.062429922,
        0.12286786,
        -0.031894935,
        0.037513372,
        0.10692168,
        0.17632998,
        -0.0058656008,
        0.072347138,
        0.15055988,
        0.22877262,
    ];

    fn fixture() -> (MultiHeadAttention, Tensor) {
        let mut attn = MultiHeadAttention::new(4, 2, 0);
        attn.q_proj = Tensor::from_vec(Q_P.to_vec(), vec![4, 4]);
        attn.k_proj = Tensor::from_vec(K_P.to_vec(), vec![4, 4]);
        attn.v_proj = Tensor::from_vec(V_P.to_vec(), vec![4, 4]);
        attn.o_proj = Tensor::from_vec(O_P.to_vec(), vec![4, 4]);
        (attn, Tensor::from_vec(X.to_vec(), vec![1, 3, 4]))
    }

    /// Scaled dot-product attention, against an independent numpy reference.
    ///
    /// There was no forward pass in this file at all, so this pins the whole
    /// definition: the 1/sqrt(d_head) scale, the softmax over keys, and the
    /// heads reading disjoint feature slices rather than the whole vector.
    #[test]
    fn multi_head_attention_matches_numpy() {
        let (attn, x) = fixture();
        let out = attn.forward(&x);
        assert_eq!(out.shape, vec![1, 3, 4]);
        for (i, want) in ATTN_OUT.iter().enumerate() {
            assert!(
                (out.data[i] - want).abs() < 1e-5,
                "element {i} was {}, numpy says {want}",
                out.data[i]
            );
        }
    }

    /// Bare self-attention is permutation-equivariant, and must be.
    ///
    /// Permuting the input sequence has to permute the output the same way and
    /// change nothing else. That is a real constraint, not a triviality: an
    /// implementation that leaked a position index into the scores, or indexed
    /// keys and values inconsistently, would break it.
    #[test]
    fn bare_attention_is_permutation_equivariant() {
        let (attn, x) = fixture();
        let out = attn.forward(&x);

        // Swap steps 0 and 2 of the input.
        let mut swapped = x.data.clone();
        for c in 0..4 {
            swapped.swap(c, 2 * 4 + c);
        }
        let out_swapped = attn.forward(&Tensor::from_vec(swapped, vec![1, 3, 4]));

        for c in 0..4 {
            assert!(
                (out_swapped.data[c] - out.data[2 * 4 + c]).abs() < 1e-5,
                "row 0 of the swapped output should equal row 2 of the original"
            );
            assert!(
                (out_swapped.data[2 * 4 + c] - out.data[c]).abs() < 1e-5,
                "row 2 of the swapped output should equal row 0 of the original"
            );
        }
    }

    /// The encoder, unlike bare attention, must notice the order of a sequence.
    ///
    /// Attention is permutation-equivariant and mean pooling is
    /// permutation-invariant, so an encoder without positional encoding reads a
    /// time series as an unordered bag and returns exactly the same answer for
    /// any shuffling of it. These models had none.
    #[test]
    fn the_encoder_is_sensitive_to_sequence_order() {
        let d = 8;
        let x = Tensor::from_shape_fn(vec![1, 6, d], |i| ((i % 11) as f32 - 5.0) * 0.17);
        let mut reversed = vec![0.0; x.data.len()];
        for t in 0..6 {
            let src = (5 - t) * d;
            reversed[t * d..(t + 1) * d].copy_from_slice(&x.data[src..src + d]);
        }
        let reversed = Tensor::from_vec(reversed, vec![1, 6, d]);

        let model = TransformerEncoder::new(d, 2, 2, 3, 9);
        assert_ne!(
            model.forward(&x).data,
            model.forward(&reversed).data,
            "reversing the sequence changed nothing, so order is invisible to the encoder"
        );
    }

    /// The linear-attention variants must not silently be softmax attention.
    #[test]
    fn linear_attention_differs_from_softmax() {
        let (attn, x) = fixture();
        let softmax = attn.forward(&x);
        let linear = attn.forward_linear(&x);
        assert_eq!(linear.shape, softmax.shape);
        assert!(
            linear.data.iter().all(|v| v.is_finite()),
            "linear attention produced a non-finite value"
        );
        assert_ne!(
            linear.data, softmax.data,
            "linear attention reproduced softmax attention exactly"
        );
    }

    /// Informer and Autoformer must differ from the plain encoder, otherwise
    /// their distinguishing mechanism is not present. They used to delegate
    /// straight to it.
    #[test]
    fn informer_and_autoformer_differ_from_the_plain_encoder() {
        let x = Tensor::from_shape_fn(vec![1, 12, 16], |i| ((i % 9) as f32 - 4.0) * 0.15);

        let plain = TransformerEncoder::new(16, 4, 2, 3, 5).forward(&x);
        let informer = Informer::new(16, 2, 3, 5).forward(&x);
        let autoformer = Autoformer::new(16, 2, 3, 5).forward(&x);
        let performer = Performer::new(16, 2, 3, 5).forward(&x);

        assert_eq!(informer.shape, plain.shape);
        assert_ne!(informer.data, plain.data, "Informer is the plain encoder");
        assert_ne!(
            autoformer.data, plain.data,
            "Autoformer is the plain encoder"
        );
        assert_ne!(performer.data, plain.data, "Performer is the plain encoder");
    }

    /// Autoformer's decomposition must actually split the series: trend plus
    /// seasonal has to reconstruct the input.
    #[test]
    fn decomposition_partitions_the_series() {
        let x = Tensor::from_shape_fn(vec![1, 8, 3], |i| (i as f32 * 0.37).sin() + i as f32 * 0.1);
        let (seasonal, trend) = decompose(&x, 3);
        for i in 0..x.data.len() {
            let sum = seasonal.data[i] + trend.data[i];
            assert!(
                (sum - x.data[i]).abs() < 1e-5,
                "element {i}: {sum} does not reconstruct {}",
                x.data[i]
            );
        }
        assert_ne!(trend.data, x.data, "the trend is the whole signal");
    }
}
