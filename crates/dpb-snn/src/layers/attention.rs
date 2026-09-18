//! Spiking attention mechanisms

use super::{NeuronState, SpikingLayer};
use crate::{NeuronParams, SNNError, SNNResult, SpikeTensor};
use ndarray::{Array2, Array3, s};
use rand::rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Spiking Self-Attention layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingAttention {
    /// Query projection weights
    pub w_query: Array2<f32>,
    /// Key projection weights
    pub w_key: Array2<f32>,
    /// Value projection weights
    pub w_value: Array2<f32>,
    /// Output projection weights
    pub w_output: Array2<f32>,
    /// Embedding dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub d_head: usize,
    /// Scaling factor
    pub scale: f32,
    /// Weight gradients
    #[serde(skip)]
    pub w_query_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_key_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_value_grad: Option<Array2<f32>>,
    #[serde(skip)]
    pub w_output_grad: Option<Array2<f32>>,
    /// Neuron parameters
    pub neuron_params: NeuronParams,
    /// Neuron state
    #[serde(skip)]
    pub state: Vec<NeuronState>,
    /// Time step
    pub dt: f32,
    /// Adaptive
    pub adaptive: bool,
}

impl SpikingAttention {
    /// Create a new spiking attention layer
    pub fn new(
        d_model: usize,
        num_heads: usize,
        neuron_params: NeuronParams,
        dt: f32,
        adaptive: bool,
    ) -> Self {
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model must be divisible by num_heads"
        );
        let d_head = d_model / num_heads;
        let scale = 1.0 / (d_head as f32).sqrt();

        let std = (2.0 / d_model as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let mut rng = rng();

        let mut init_weights =
            |shape: (usize, usize)| Array2::from_shape_fn(shape, |_| normal.sample(&mut rng));

        Self {
            w_query: init_weights((d_model, d_model)),
            w_key: init_weights((d_model, d_model)),
            w_value: init_weights((d_model, d_model)),
            w_output: init_weights((d_model, d_model)),
            d_model,
            num_heads,
            d_head,
            scale,
            w_query_grad: None,
            w_key_grad: None,
            w_value_grad: None,
            w_output_grad: None,
            neuron_params,
            state: Vec::new(),
            dt,
            adaptive,
        }
    }

    /// Scaled dot-product self-attention over the time axis, for one batch item.
    ///
    /// `q`, `k`, `v` are `(num_steps, d_model)`. Attention runs *within* a
    /// sample, across its own time steps: that is what self-attention in a
    /// sequence model means, and the only axis a temporal layer can meaningfully
    /// attend over.
    ///
    /// It used to run across the batch axis -- `forward` handed it one time step
    /// at a time, so the "sequence" it attended over was the set of independent
    /// samples in the batch. Two things followed. Samples contaminated each
    /// other, so a result depended on who else was in the batch and changed with
    /// batch size; and at batch size 1, the usual single-subject case, the
    /// softmax was over one element, therefore exactly 1.0, so the output was
    /// just the value vector and `w_query`/`w_key` had no effect at all.
    ///
    /// Attention is non-causal: a step attends over the whole window, later
    /// steps included. That is standard for a sequence model reading a complete
    /// recording, but it means this layer cannot be used for streaming
    /// inference, where a causal mask would be required.
    fn attention_step(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let num_steps = q.shape()[0];
        let d_head = self.d_model / self.num_heads;
        let mut output = Array2::zeros((num_steps, self.d_model));

        // Each head attends over its own slice of the feature axis. Previously
        // every head saw the full d_model and `num_heads` changed nothing but
        // the scale factor.
        for h in 0..self.num_heads {
            let lo = h * d_head;
            let hi = lo + d_head;

            let mut scores = Array2::zeros((num_steps, num_steps));
            for i in 0..num_steps {
                for j in 0..num_steps {
                    let mut dot = 0.0;
                    for c in lo..hi {
                        dot += q[[i, c]] * k[[j, c]];
                    }
                    scores[[i, j]] = dot * self.scale;
                }
            }

            for i in 0..num_steps {
                let row = scores.slice(s![i, ..]);
                let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let total: f32 = exps.iter().sum();

                for (j, e) in exps.iter().enumerate() {
                    let w = e / total;
                    if w == 0.0 {
                        continue;
                    }
                    for c in lo..hi {
                        output[[i, c]] += w * v[[j, c]];
                    }
                }
            }
        }

        output
    }

    /// Ensure state is initialized
    fn ensure_state(&mut self, batch_size: usize) {
        if self.state.len() != batch_size {
            self.state = (0..batch_size)
                .map(|_| NeuronState::new(self.d_model, self.adaptive))
                .collect();
        }
    }
}

impl SpikingLayer for SpikingAttention {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let input_dense = input.to_dense();
        let (batch_size, num_steps, feature_dim) = (
            input_dense.shape()[0],
            input_dense.shape()[1],
            input_dense.shape()[2],
        );

        if feature_dim != self.d_model {
            return Err(SNNError::DimensionMismatch {
                expected: format!("d_model {}", self.d_model),
                actual: format!("feature_dim {}", feature_dim),
            });
        }

        self.ensure_state(batch_size);

        let mut output = Array3::zeros((batch_size, num_steps, self.d_model));

        // One sample at a time, attending over that sample's own time steps.
        // The previous loop was over time, handing attention a batch-wide slice.
        for b in 0..batch_size {
            let mut query = Array2::zeros((num_steps, self.d_model));
            let mut key = Array2::zeros((num_steps, self.d_model));
            let mut value = Array2::zeros((num_steps, self.d_model));

            for t in 0..num_steps {
                let inp = input_dense.slice(s![b, t, ..]);
                query.slice_mut(s![t, ..]).assign(&self.w_query.dot(&inp));
                key.slice_mut(s![t, ..]).assign(&self.w_key.dot(&inp));
                value.slice_mut(s![t, ..]).assign(&self.w_value.dot(&inp));
            }

            let attended = self.attention_step(&query, &key, &value);

            // The neuron integrates along time, so the output projection and the
            // membrane update stay in time order.
            for t in 0..num_steps {
                let proj = self.w_output.dot(&attended.slice(s![t, ..]));
                let spikes = self.state[b].update_lif(&proj, &self.neuron_params, self.dt);
                output.slice_mut(s![b, t, ..]).assign(&spikes);
            }
        }

        Ok(SpikeTensor::from_dense(output, input.requires_grad))
    }

    fn reset_state(&mut self) {
        for state in &mut self.state {
            state.reset();
        }
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        vec![&self.w_query, &self.w_key, &self.w_value, &self.w_output]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![
            &mut self.w_query,
            &mut self.w_key,
            &mut self.w_value,
            &mut self.w_output,
        ]
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        vec![
            self.w_query_grad.as_ref(),
            self.w_key_grad.as_ref(),
            self.w_value_grad.as_ref(),
            self.w_output_grad.as_ref(),
        ]
    }

    fn zero_grad(&mut self) {
        self.w_query_grad = None;
        self.w_key_grad = None;
        self.w_value_grad = None;
        self.w_output_grad = None;
    }
}

/// Multi-head spiking attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiHeadSpikingAttention {
    /// Individual attention heads
    pub heads: Vec<SpikingAttention>,
    /// Output projection
    pub w_output: Array2<f32>,
    /// Output gradient
    #[serde(skip)]
    pub w_output_grad: Option<Array2<f32>>,
}

impl MultiHeadSpikingAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        neuron_params: NeuronParams,
        dt: f32,
        adaptive: bool,
    ) -> Self {
        let heads = (0..num_heads)
            .map(|_| SpikingAttention::new(d_model, num_heads, neuron_params.clone(), dt, adaptive))
            .collect();

        let std = (1.0 / d_model as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        let mut rng = rng();

        let w_output = Array2::from_shape_fn((d_model, d_model), |_| normal.sample(&mut rng));

        Self {
            heads,
            w_output,
            w_output_grad: None,
        }
    }
}

impl SpikingLayer for MultiHeadSpikingAttention {
    /// Runs every head and combines them through the output projection.
    ///
    /// The heads used to be computed and then discarded -- `forward` returned
    /// the first and dropped the rest, and `w_output` was never applied at all,
    /// so every head after the first was pure cost and the output projection
    /// was dead weight that nothing read.
    ///
    /// Each head is a full `SpikingAttention` over `d_model`, so their outputs
    /// cannot be concatenated into `d_model` the way a single module's internal
    /// heads are; they are averaged instead, which keeps the scale independent
    /// of the head count. The result is a linear projection of the heads' spike
    /// trains rather than a spike train itself -- as with the pooling layers,
    /// which likewise carry sums and averages in a `SpikeTensor`.
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        if self.heads.is_empty() {
            return Err(SNNError::InvalidConfig(
                "No attention heads found".to_string(),
            ));
        }

        let mut summed: Option<Array3<f32>> = None;
        for head in &mut self.heads {
            let out = head.forward(input)?.to_dense();
            summed = Some(match summed {
                Some(acc) => acc + out,
                None => out,
            });
        }

        let mut combined = summed.expect("at least one head, checked above");
        combined /= self.heads.len() as f32;

        let (batch, steps, d_model) = (
            combined.shape()[0],
            combined.shape()[1],
            combined.shape()[2],
        );
        if d_model != self.w_output.shape()[1] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("head width {}", self.w_output.shape()[1]),
                actual: format!("{d_model}"),
            });
        }

        let mut projected = Array3::zeros((batch, steps, self.w_output.shape()[0]));
        for b in 0..batch {
            for t in 0..steps {
                let row = combined.slice(s![b, t, ..]);
                projected
                    .slice_mut(s![b, t, ..])
                    .assign(&self.w_output.dot(&row));
            }
        }

        Ok(SpikeTensor::from_dense(projected, input.requires_grad))
    }

    fn reset_state(&mut self) {
        for head in &mut self.heads {
            head.reset_state();
        }
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        vec![&self.w_output]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![&mut self.w_output]
    }

    fn gradients(&self) -> Vec<Option<&Array2<f32>>> {
        vec![self.w_output_grad.as_ref()]
    }

    fn zero_grad(&mut self) {
        self.w_output_grad = None;
        for head in &mut self.heads {
            head.zero_grad();
        }
    }

    fn num_parameters(&self) -> usize {
        // `parameters` exposes only the output projection, so counting through
        // it reports a multi-head attention as though it had no heads.
        self.heads.iter().map(|h| h.num_parameters()).sum::<usize>() + self.w_output.len()
    }
}

impl Default for SpikingAttention {
    fn default() -> Self {
        Self::new(64, 4, NeuronParams::default(), 1.0, false)
    }
}

impl Default for MultiHeadSpikingAttention {
    fn default() -> Self {
        Self::new(64, 4, NeuronParams::default(), 1.0, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let attention = SpikingAttention::new(64, 4, NeuronParams::default(), 1.0, false);
        assert_eq!(attention.d_model, 64);
        assert_eq!(attention.num_heads, 4);
        assert_eq!(attention.d_head, 16);
    }

    #[test]
    fn test_attention_forward() {
        let mut attention = SpikingAttention::new(64, 4, NeuronParams::default(), 1.0, false);
        let input = SpikeTensor::zeros(2, 10, 64, false);

        let output = attention.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 10, 64));
    }

    #[test]
    fn test_multi_head_attention() {
        let mut mha = MultiHeadSpikingAttention::new(64, 4, NeuronParams::default(), 1.0, false);
        let input = SpikeTensor::zeros(2, 10, 64, false);

        let output = mha.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 10, 64));
    }

    /// Deterministic weights, so a test measures the layer rather than the
    /// draw. The constructors initialise randomly, and some draws never reach
    /// threshold at all -- an all-zero output would make every comparison below
    /// pass for the wrong reason.
    fn fixed_attention(d_model: usize, num_heads: usize) -> SpikingAttention {
        let mut layer =
            SpikingAttention::new(d_model, num_heads, NeuronParams::default(), 1.0, false);
        let fill = |m: &mut Array2<f32>, seed: f32| {
            for i in 0..m.shape()[0] {
                for j in 0..m.shape()[1] {
                    m[[i, j]] = seed + 0.11 * (i as f32) - 0.07 * (j as f32);
                }
            }
        };
        fill(&mut layer.w_query, 0.30);
        fill(&mut layer.w_key, 0.25);
        fill(&mut layer.w_value, 0.40);
        fill(&mut layer.w_output, 0.35);
        layer
    }

    fn ramp(batch: usize, steps: usize, d_model: usize, seed: f32) -> Array3<f32> {
        Array3::from_shape_fn((batch, steps, d_model), |(b, t, c)| {
            ((seed + (b * 5 + t * 3 + c * 2) as f32) % 5.0) * 0.4
        })
    }

    /// A sample's output must not depend on who else is in the batch.
    ///
    /// Attention used to run across the batch axis, so a softmax mixed
    /// independent samples together: running a subject alone and running the
    /// same subject alongside another gave different answers. For a library
    /// that scores clinical recordings, that is the difference between a result
    /// and an artefact of how the work was grouped.
    #[test]
    fn samples_do_not_leak_into_each_other() {
        let (d_model, steps) = (8usize, 30usize);
        let a = ramp(1, steps, d_model, 0.0);
        let b = ramp(1, steps, d_model, 2.0);

        let mut layer = fixed_attention(d_model, 2);
        let alone = layer
            .forward(&SpikeTensor::from_dense(a.clone(), false))
            .unwrap()
            .to_dense();
        assert!(
            alone.iter().sum::<f32>() > 0.0,
            "the lone sample never spiked, so the comparison would be vacuous"
        );

        let mut both = Array3::zeros((2, steps, d_model));
        both.slice_mut(s![0, .., ..])
            .assign(&a.slice(s![0, .., ..]));
        both.slice_mut(s![1, .., ..])
            .assign(&b.slice(s![0, .., ..]));

        let mut layer = fixed_attention(d_model, 2);
        let batched = layer
            .forward(&SpikeTensor::from_dense(both, false))
            .unwrap()
            .to_dense();

        for t in 0..steps {
            for c in 0..d_model {
                assert_eq!(
                    alone[[0, t, c]],
                    batched[[0, t, c]],
                    "step {t}, channel {c}: sample 0 changed when another sample \
                     joined the batch"
                );
            }
        }
    }

    /// The query projection must affect the output.
    ///
    /// Attending across the batch meant that at batch size 1 -- single-subject
    /// inference -- the softmax was over one element and therefore exactly 1.0,
    /// so `w_query` and `w_key` had no effect whatsoever and half the layer's
    /// parameters were dead.
    #[test]
    fn query_weights_affect_the_output() {
        let (d_model, steps) = (8usize, 30usize);
        let input = SpikeTensor::from_dense(ramp(1, steps, d_model, 0.0), false);

        let mut layer = fixed_attention(d_model, 2);
        let before = layer.forward(&input).unwrap().to_dense();
        assert!(
            before.iter().sum::<f32>() > 0.0,
            "no spikes at all, so a difference could not show up"
        );

        let mut layer = fixed_attention(d_model, 2);
        layer.w_query.mapv_inplace(|w| w * -3.0);
        let after = layer.forward(&input).unwrap().to_dense();

        let delta: f32 = (&before - &after).iter().map(|d| d.abs()).sum();
        assert!(
            delta > 0.0,
            "scaling w_query by -3 changed nothing; the query projection is dead"
        );
    }

    /// Every head must reach the output, through the output projection.
    #[test]
    fn multi_head_uses_every_head_and_the_output_projection() {
        // d_model must divide by num_heads.
        let (d_model, heads, steps) = (8usize, 4usize, 30usize);
        let input = SpikeTensor::from_dense(ramp(1, steps, d_model, 0.0), false);

        let build = || {
            let mut mha =
                MultiHeadSpikingAttention::new(d_model, heads, NeuronParams::default(), 1.0, false);
            for (i, h) in mha.heads.iter_mut().enumerate() {
                let seed = 0.3 + 0.05 * i as f32;
                for m in [
                    &mut h.w_query,
                    &mut h.w_key,
                    &mut h.w_value,
                    &mut h.w_output,
                ] {
                    for a in 0..m.shape()[0] {
                        for b in 0..m.shape()[1] {
                            m[[a, b]] = seed + 0.09 * (a as f32) - 0.06 * (b as f32);
                        }
                    }
                }
            }
            for a in 0..d_model {
                for b in 0..d_model {
                    mha.w_output[[a, b]] = 0.2 + 0.03 * (a as f32) - 0.02 * (b as f32);
                }
            }
            mha
        };

        let base = build().forward(&input).unwrap().to_dense();

        // Perturbing the LAST head must change the result: previously only the
        // first head's output was returned.
        let mut altered = build();
        altered
            .heads
            .last_mut()
            .unwrap()
            .w_value
            .mapv_inplace(|w| w * -4.0);
        let after_head = altered.forward(&input).unwrap().to_dense();
        assert!(
            (&base - &after_head).iter().map(|d| d.abs()).sum::<f32>() > 0.0,
            "changing the last head changed nothing; heads after the first are discarded"
        );

        // And the output projection must be applied at all.
        let mut scaled = build();
        scaled.w_output.mapv_inplace(|w| w * 7.0);
        let after_proj = scaled.forward(&input).unwrap().to_dense();
        assert!(
            (&base - &after_proj).iter().map(|d| d.abs()).sum::<f32>() > 0.0,
            "scaling w_output changed nothing; the output projection is never applied"
        );
    }
}
