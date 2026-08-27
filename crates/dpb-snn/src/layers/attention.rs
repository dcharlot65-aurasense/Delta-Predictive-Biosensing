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

    /// Compute attention for a single time step
    fn attention_step(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
    ) -> Array2<f32> {
        let batch_size = query.shape()[0];
        let _seq_len = query.shape()[0]; // Simplified: treat batch as sequence

        // Compute attention scores: Q @ K^T / sqrt(d_head)
        let mut scores = Array2::zeros((batch_size, batch_size));
        for i in 0..batch_size {
            for j in 0..batch_size {
                let q = query.slice(s![i, ..]);
                let k = key.slice(s![j, ..]);
                let score = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * self.scale;
                scores[[i, j]] = score;
            }
        }

        // Apply softmax (spike-based approximation)
        let mut attention_weights = Array2::zeros((batch_size, batch_size));
        for i in 0..batch_size {
            let row = scores.slice(s![i, ..]);
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_row: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_row.iter().sum();

            for j in 0..batch_size {
                attention_weights[[i, j]] = exp_row[j] / sum_exp;
            }
        }

        // Apply attention to values: attention_weights @ V
        let mut output = Array2::zeros((batch_size, self.d_model));
        for i in 0..batch_size {
            for j in 0..batch_size {
                let weight = attention_weights[[i, j]];
                let v = value.slice(s![j, ..]);
                for k in 0..self.d_model {
                    output[[i, k]] += weight * v[k];
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

        // Process each time step
        for t in 0..num_steps {
            let input_t = input_dense.slice(s![.., t, ..]).to_owned();

            // Project to Q, K, V
            let mut query = Array2::zeros((batch_size, self.d_model));
            let mut key = Array2::zeros((batch_size, self.d_model));
            let mut value = Array2::zeros((batch_size, self.d_model));

            for b in 0..batch_size {
                let inp = input_t.slice(s![b, ..]);
                query.slice_mut(s![b, ..]).assign(&self.w_query.dot(&inp));
                key.slice_mut(s![b, ..]).assign(&self.w_key.dot(&inp));
                value.slice_mut(s![b, ..]).assign(&self.w_value.dot(&inp));
            }

            // Compute attention
            let attended = self.attention_step(&query, &key, &value);

            // Output projection
            for b in 0..batch_size {
                let att = attended.slice(s![b, ..]);
                let proj = self.w_output.dot(&att);

                // Convert to spikes using neuron dynamics
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
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        // Process through each head
        let mut head_outputs = Vec::new();
        for head in &mut self.heads {
            head_outputs.push(head.forward(input)?);
        }

        // Concatenate head outputs (simplified)
        // In practice, we'd need proper concatenation and reshaping
        head_outputs
            .into_iter()
            .next()
            .ok_or_else(|| SNNError::InvalidConfig("No attention heads found".to_string()))
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
}
