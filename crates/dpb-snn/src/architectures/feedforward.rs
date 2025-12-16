//! Feedforward SNN architectures

use super::SNNArchitecture;
use crate::{
    layers::{SpikingLayer, SpikingLinear},
    SNNConfig, SNNError, SNNResult, SpikeTensor,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Feedforward Spiking Neural Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedforwardSNN {
    /// Network layers
    #[serde(skip)]
    pub layers: Vec<SpikingLinear>,
    /// Layer sizes
    pub layer_sizes: Vec<usize>,
    /// Network configuration
    pub config: SNNConfig,
    /// Use bias in layers
    pub use_bias: bool,
}

impl FeedforwardSNN {
    /// Create a new feedforward SNN
    pub fn new(layer_sizes: Vec<usize>, config: SNNConfig, use_bias: bool) -> Self {
        if layer_sizes.len() < 2 {
            panic!("Need at least 2 layer sizes (input and output)");
        }

        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let layer = SpikingLinear::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                use_bias,
                config.neuron_params.clone(),
                config.dt,
                false, // Non-adaptive for simplicity
            );
            layers.push(layer);
        }

        Self {
            layers,
            layer_sizes,
            config,
            use_bias,
        }
    }

    /// Create a standard MLP-style SNN
    pub fn mlp(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        config: SNNConfig,
    ) -> Self {
        let mut layer_sizes = vec![input_size];
        layer_sizes.extend(hidden_sizes);
        layer_sizes.push(output_size);

        Self::new(layer_sizes, config, true)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer by index
    pub fn layer(&self, index: usize) -> Option<&SpikingLinear> {
        self.layers.get(index)
    }

    /// Get mutable layer by index
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut SpikingLinear> {
        self.layers.get_mut(index)
    }
}

impl SNNArchitecture for FeedforwardSNN {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let mut current = input.clone();

        // Check input size
        if current.num_neurons() != self.layer_sizes[0] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("input size {}", self.layer_sizes[0]),
                actual: format!("input size {}", current.num_neurons()),
            });
        }

        // Forward through all layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            current = layer.forward(&current).map_err(|e| {
                SNNError::Layer(format!("Error in layer {}: {}", i, e))
            })?;
        }

        Ok(current)
    }

    fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset_state();
        }
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

/// Builder for FeedforwardSNN
pub struct FeedforwardSNNBuilder {
    layer_sizes: Vec<usize>,
    config: SNNConfig,
    use_bias: bool,
}

impl FeedforwardSNNBuilder {
    /// Create a new builder with input size
    pub fn new(input_size: usize) -> Self {
        Self {
            layer_sizes: vec![input_size],
            config: SNNConfig::default(),
            use_bias: true,
        }
    }

    /// Add a hidden layer
    pub fn add_layer(mut self, size: usize) -> Self {
        self.layer_sizes.push(size);
        self
    }

    /// Set network configuration
    pub fn config(mut self, config: SNNConfig) -> Self {
        self.config = config;
        self
    }

    /// Set whether to use bias
    pub fn use_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Build the network
    pub fn build(self) -> SNNResult<FeedforwardSNN> {
        if self.layer_sizes.len() < 2 {
            return Err(SNNError::InvalidConfig(
                "Need at least input and output layers".to_string(),
            ));
        }

        Ok(FeedforwardSNN::new(
            self.layer_sizes,
            self.config,
            self.use_bias,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_creation() {
        let snn = FeedforwardSNN::new(vec![10, 20, 5], SNNConfig::default(), true);
        assert_eq!(snn.num_layers(), 2);
        assert_eq!(snn.layer_sizes, vec![10, 20, 5]);
    }

    #[test]
    fn test_feedforward_mlp() {
        let snn = FeedforwardSNN::mlp(10, vec![20, 15], 5, SNNConfig::default());
        assert_eq!(snn.num_layers(), 3);
        assert_eq!(snn.layer_sizes, vec![10, 20, 15, 5]);
    }

    #[test]
    fn test_feedforward_forward() {
        let mut snn = FeedforwardSNN::new(vec![10, 20, 5], SNNConfig::default(), true);
        let input = SpikeTensor::zeros(2, 50, 10, false);

        let output = snn.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 50, 5));
    }

    #[test]
    fn test_feedforward_builder() {
        let snn = FeedforwardSNNBuilder::new(10)
            .add_layer(20)
            .add_layer(15)
            .add_layer(5)
            .use_bias(true)
            .build()
            .unwrap();

        assert_eq!(snn.num_layers(), 3);
        assert_eq!(snn.layer_sizes, vec![10, 20, 15, 5]);
    }

    #[test]
    fn test_feedforward_reset() {
        let mut snn = FeedforwardSNN::new(vec![5, 10, 3], SNNConfig::default(), true);
        let input = SpikeTensor::zeros(1, 10, 5, false);

        // Forward pass to create state
        let _ = snn.forward(&input).unwrap();

        // Reset should clear all states
        snn.reset();
        assert!(snn.layers[0].state.is_empty() || snn.layers[0].state[0].v_mem.iter().all(|&v| v == 0.0));
    }
}
