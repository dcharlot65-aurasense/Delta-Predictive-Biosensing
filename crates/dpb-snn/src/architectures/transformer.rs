//! Spiking Transformer architecture

use super::SNNArchitecture;
use crate::{
    SNNConfig, SNNError, SNNResult, SpikeTensor,
    layers::{SpikingAttention, SpikingLayer, SpikingLinear},
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Spiking Transformer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingTransformer {
    /// Embedding layer
    pub embedding: Option<SpikingLinear>,
    /// Transformer blocks
    pub blocks: Vec<SpikingTransformerBlock>,
    /// Output projection
    pub output_proj: SpikingLinear,
    /// Network configuration
    pub config: SNNConfig,
    /// Model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of blocks
    pub num_blocks: usize,
    /// Output size
    pub output_size: usize,
}

impl SpikingTransformer {
    /// Create a new Spiking Transformer
    pub fn new(
        input_size: usize,
        d_model: usize,
        num_heads: usize,
        num_blocks: usize,
        output_size: usize,
        config: SNNConfig,
    ) -> Self {
        // Embedding layer if input_size != d_model
        let embedding = if input_size != d_model {
            Some(SpikingLinear::new(
                input_size,
                d_model,
                true,
                config.neuron_params.clone(),
                config.dt,
                false,
            ))
        } else {
            None
        };

        // Create transformer blocks
        let blocks = (0..num_blocks)
            .map(|_| {
                SpikingTransformerBlock::new(
                    d_model,
                    num_heads,
                    d_model * 4, // FFN hidden size (typical: 4x d_model)
                    config.neuron_params.clone(),
                    config.dt,
                )
            })
            .collect();

        // Output projection
        let output_proj = SpikingLinear::new(
            d_model,
            output_size,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        Self {
            embedding,
            blocks,
            output_proj,
            config,
            d_model,
            num_heads,
            num_blocks,
            output_size,
        }
    }
}

impl SNNArchitecture for SpikingTransformer {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        // Embed input if needed
        let mut current = if let Some(ref mut emb) = self.embedding {
            emb.forward(input)?
        } else {
            input.clone()
        };

        // Forward through transformer blocks
        for block in &mut self.blocks {
            current = block.forward(&current)?;
        }

        // Output projection
        current = self.output_proj.forward(&current)?;

        Ok(current)
    }

    fn reset(&mut self) {
        if let Some(ref mut emb) = self.embedding {
            emb.reset_state();
        }
        for block in &mut self.blocks {
            block.reset();
        }
        self.output_proj.reset_state();
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        let mut params = Vec::new();
        if let Some(ref emb) = self.embedding {
            params.extend(emb.parameters());
        }
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.output_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        let mut params = Vec::new();
        if let Some(ref mut emb) = self.embedding {
            params.extend(emb.parameters_mut());
        }
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        params.extend(self.output_proj.parameters_mut());
        params
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut emb) = self.embedding {
            emb.zero_grad();
        }
        for block in &mut self.blocks {
            block.zero_grad();
        }
        self.output_proj.zero_grad();
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

/// Single Transformer Block with spiking neurons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingTransformerBlock {
    /// Self-attention layer
    pub attention: SpikingAttention,
    /// Feed-forward network
    pub ffn: Vec<SpikingLinear>,
    /// Layer normalization parameters (simplified)
    pub norm1_scale: f32,
    pub norm2_scale: f32,
}

impl SpikingTransformerBlock {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        ffn_hidden_size: usize,
        neuron_params: crate::NeuronParams,
        dt: f32,
    ) -> Self {
        let attention = SpikingAttention::new(d_model, num_heads, neuron_params.clone(), dt, false);

        // Feed-forward network: d_model -> ffn_hidden -> d_model
        let ffn = vec![
            SpikingLinear::new(
                d_model,
                ffn_hidden_size,
                true,
                neuron_params.clone(),
                dt,
                false,
            ),
            SpikingLinear::new(ffn_hidden_size, d_model, true, neuron_params, dt, false),
        ];

        Self {
            attention,
            ffn,
            norm1_scale: 1.0,
            norm2_scale: 1.0,
        }
    }

    pub fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        // Self-attention with residual connection
        let attn_out = self.attention.forward(input)?;
        let after_attn = self.add_residual(input, &attn_out)?;

        // Feed-forward with residual connection
        let mut ffn_out = after_attn.clone();
        for layer in &mut self.ffn {
            ffn_out = layer.forward(&ffn_out)?;
        }
        let output = self.add_residual(&after_attn, &ffn_out)?;

        Ok(output)
    }

    fn add_residual(&self, residual: &SpikeTensor, main: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let res_dense = residual.to_dense();
        let main_dense = main.to_dense();

        if res_dense.shape() != main_dense.shape() {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{:?}", res_dense.shape()),
                actual: format!("{:?}", main_dense.shape()),
            });
        }

        let combined = res_dense + main_dense;
        Ok(SpikeTensor::from_dense(
            combined,
            residual.requires_grad || main.requires_grad,
        ))
    }

    pub fn reset(&mut self) {
        self.attention.reset_state();
        for layer in &mut self.ffn {
            layer.reset_state();
        }
    }

    pub fn parameters(&self) -> Vec<&Array2<f32>> {
        let mut params = self.attention.parameters();
        for layer in &self.ffn {
            params.extend(layer.parameters());
        }
        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        let mut params = self.attention.parameters_mut();
        for layer in &mut self.ffn {
            params.extend(layer.parameters_mut());
        }
        params
    }

    pub fn zero_grad(&mut self) {
        self.attention.zero_grad();
        for layer in &mut self.ffn {
            layer.zero_grad();
        }
    }
}

/// Builder for Spiking Transformer
pub struct SpikingTransformerBuilder {
    input_size: usize,
    d_model: usize,
    num_heads: usize,
    num_blocks: usize,
    output_size: usize,
    config: SNNConfig,
}

impl SpikingTransformerBuilder {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            d_model: 512,
            num_heads: 8,
            num_blocks: 6,
            output_size,
            config: SNNConfig::default(),
        }
    }

    pub fn d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        self
    }

    pub fn num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    pub fn num_blocks(mut self, num_blocks: usize) -> Self {
        self.num_blocks = num_blocks;
        self
    }

    pub fn config(mut self, config: SNNConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> SNNResult<SpikingTransformer> {
        if !self.d_model.is_multiple_of(self.num_heads) {
            return Err(SNNError::InvalidConfig(
                "d_model must be divisible by num_heads".to_string(),
            ));
        }

        Ok(SpikingTransformer::new(
            self.input_size,
            self.d_model,
            self.num_heads,
            self.num_blocks,
            self.output_size,
            self.config,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_creation() {
        let transformer = SpikingTransformer::new(64, 128, 4, 2, 10, SNNConfig::default());
        assert_eq!(transformer.d_model, 128);
        assert_eq!(transformer.num_heads, 4);
        assert_eq!(transformer.num_blocks, 2);
    }

    #[test]
    fn test_transformer_forward() {
        let mut transformer = SpikingTransformer::new(64, 128, 4, 2, 10, SNNConfig::default());
        let input = SpikeTensor::zeros(2, 20, 64, false);

        let output = transformer.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 20, 10));
    }

    #[test]
    fn test_transformer_builder() {
        let transformer = SpikingTransformerBuilder::new(64, 10)
            .d_model(256)
            .num_heads(8)
            .num_blocks(4)
            .build()
            .unwrap();

        assert_eq!(transformer.d_model, 256);
        assert_eq!(transformer.num_heads, 8);
        assert_eq!(transformer.num_blocks, 4);
    }

    #[test]
    fn test_transformer_block() {
        let mut block =
            SpikingTransformerBlock::new(128, 4, 512, crate::NeuronParams::default(), 1.0);
        let input = SpikeTensor::zeros(1, 10, 128, false);

        let output = block.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 10, 128));
    }
}
