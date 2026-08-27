//! Recurrent SNN architectures for temporal processing

use super::SNNArchitecture;
use crate::{
    layers::{SpikingLayer, SpikingLinear, SpikingRNN},
    SNNConfig, SNNError, SNNResult, SpikeTensor,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Recurrent Spiking Neural Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentSNN {
    /// Input projection layer
    #[serde(skip)]
    pub input_layer: Option<SpikingLinear>,
    /// Recurrent layers
    #[serde(skip)]
    pub recurrent_layers: Vec<SpikingRNN>,
    /// Output layer
    #[serde(skip)]
    pub output_layer: SpikingLinear,
    /// Network configuration
    pub config: SNNConfig,
    /// Layer sizes
    pub layer_sizes: Vec<usize>,
}

impl RecurrentSNN {
    /// Create a new recurrent SNN
    ///
    /// # Errors
    /// Returns `SNNError::InvalidConfig` if `hidden_sizes` is empty.
    #[must_use = "this Result may contain an error that should be handled"]
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        config: SNNConfig,
    ) -> SNNResult<Self> {
        if hidden_sizes.is_empty() {
            return Err(SNNError::InvalidConfig(
                "Need at least one hidden layer".to_string(),
            ));
        }

        // Input projection if first hidden size differs from input
        let input_layer = if input_size != hidden_sizes[0] {
            Some(SpikingLinear::new(
                input_size,
                hidden_sizes[0],
                true,
                config.neuron_params.clone(),
                config.dt,
                false,
            ))
        } else {
            None
        };

        // Create recurrent layers
        let mut recurrent_layers = Vec::new();
        for i in 0..hidden_sizes.len() {
            let h_in = if i == 0 { hidden_sizes[0] } else { hidden_sizes[i - 1] };
            let h_out = hidden_sizes[i];

            recurrent_layers.push(SpikingRNN::new(
                h_in,
                h_out,
                true,
                config.neuron_params.clone(),
                config.dt,
                false,
            ));
        }

        // Output layer
        let output_layer = SpikingLinear::new(
            *hidden_sizes.last().unwrap(),
            output_size,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        let mut layer_sizes = vec![input_size];
        layer_sizes.extend(hidden_sizes);
        layer_sizes.push(output_size);

        Ok(Self {
            input_layer,
            recurrent_layers,
            output_layer,
            config,
            layer_sizes,
        })
    }
}

impl SNNArchitecture for RecurrentSNN {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        // Project input if needed
        let mut current = if let Some(ref mut input_layer) = self.input_layer {
            input_layer.forward(input)?
        } else {
            input.clone()
        };

        // Forward through recurrent layers
        for layer in &mut self.recurrent_layers {
            current = layer.forward(&current)?;
        }

        // Forward through output layer
        current = self.output_layer.forward(&current)?;

        Ok(current)
    }

    fn reset(&mut self) {
        if let Some(ref mut layer) = self.input_layer {
            layer.reset_state();
        }
        for layer in &mut self.recurrent_layers {
            layer.reset_state();
        }
        self.output_layer.reset_state();
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        let mut params = Vec::new();
        if let Some(ref layer) = self.input_layer {
            params.extend(layer.parameters());
        }
        for layer in &self.recurrent_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.output_layer.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        let mut params = Vec::new();
        if let Some(ref mut layer) = self.input_layer {
            params.extend(layer.parameters_mut());
        }
        for layer in &mut self.recurrent_layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.output_layer.parameters_mut());
        params
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut layer) = self.input_layer {
            layer.zero_grad();
        }
        for layer in &mut self.recurrent_layers {
            layer.zero_grad();
        }
        self.output_layer.zero_grad();
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

/// Liquid State Machine (LSM)
/// A reservoir computing approach with a randomly connected spiking reservoir
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidStateMachine {
    /// Reservoir layer (randomly initialized, typically not trained)
    #[serde(skip)]
    pub reservoir: SpikingRNN,
    /// Readout layer (trained)
    #[serde(skip)]
    pub readout: SpikingLinear,
    /// Network configuration
    pub config: SNNConfig,
    /// Reservoir size
    pub reservoir_size: usize,
    /// Output size
    pub output_size: usize,
    /// Spectral radius for reservoir weights
    pub spectral_radius: f32,
}

impl LiquidStateMachine {
    /// Create a new Liquid State Machine
    pub fn new(
        input_size: usize,
        reservoir_size: usize,
        output_size: usize,
        spectral_radius: f32,
        config: SNNConfig,
    ) -> Self {
        // Create reservoir with random connectivity
        let mut reservoir = SpikingRNN::new(
            input_size,
            reservoir_size,
            false,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        // Scale recurrent weights to desired spectral radius
        // This is a simplified version - proper implementation would compute actual eigenvalues
        reservoir.w_recurrent *= spectral_radius / reservoir_size as f32;

        // Readout layer
        let readout = SpikingLinear::new(
            reservoir_size,
            output_size,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        Self {
            reservoir,
            readout,
            config,
            reservoir_size,
            output_size,
            spectral_radius,
        }
    }

    /// Get reservoir state (for analysis)
    pub fn get_reservoir_state(&self) -> Vec<Array1<f32>> {
        self.reservoir
            .state
            .iter()
            .map(|s| s.v_mem.clone())
            .collect()
    }
}

impl SNNArchitecture for LiquidStateMachine {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        // Forward through reservoir (typically frozen)
        let reservoir_out = self.reservoir.forward(input)?;

        // Forward through readout layer (trainable)
        let output = self.readout.forward(&reservoir_out)?;

        Ok(output)
    }

    fn reset(&mut self) {
        self.reservoir.reset_state();
        self.readout.reset_state();
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        // Only readout parameters are trainable in standard LSM
        self.readout.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        // Only readout parameters are trainable in standard LSM
        self.readout.parameters_mut()
    }

    fn zero_grad(&mut self) {
        // Only zero readout gradients
        self.readout.zero_grad();
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

/// Echo State Network (ESN) style SNN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoStateSNN {
    /// Reservoir
    #[serde(skip)]
    pub reservoir: SpikingRNN,
    /// Output weights (trained with simple regression)
    pub output_weights: Array2<f32>,
    /// Config
    pub config: SNNConfig,
}

impl EchoStateSNN {
    pub fn new(
        input_size: usize,
        reservoir_size: usize,
        output_size: usize,
        config: SNNConfig,
    ) -> Self {
        let reservoir = SpikingRNN::new(
            input_size,
            reservoir_size,
            false,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        let output_weights = Array2::zeros((output_size, reservoir_size));

        Self {
            reservoir,
            output_weights,
            config,
        }
    }

    /// Train readout using ridge regression
    pub fn train_readout(
        &mut self,
        states: &Array2<f32>,
        _targets: &Array2<f32>,
        ridge_param: f32,
    ) -> SNNResult<()> {
        // Simplified ridge regression: W = Y X^T (X X^T + λI)^-1
        // In practice, use proper linear algebra library for this

        let n_samples = states.shape()[0];
        let reservoir_size = states.shape()[1];

        // Compute X X^T
        let mut xxt = Array2::zeros((reservoir_size, reservoir_size));
        for i in 0..reservoir_size {
            for j in 0..reservoir_size {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += states[[k, i]] * states[[k, j]];
                }
                xxt[[i, j]] = sum;
            }
        }

        // Add ridge parameter to diagonal
        for i in 0..reservoir_size {
            xxt[[i, i]] += ridge_param;
        }

        // This is a placeholder - would need matrix inversion
        // self.output_weights = targets.t().dot(&states).dot(&xxt_inv);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recurrent_snn_creation() {
        let rsnn = RecurrentSNN::new(10, vec![20, 15], 5, SNNConfig::default()).unwrap();
        assert_eq!(rsnn.layer_sizes, vec![10, 20, 15, 5]);
    }

    #[test]
    fn test_recurrent_snn_empty_hidden() {
        let result = RecurrentSNN::new(10, vec![], 5, SNNConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_recurrent_snn_forward() {
        let mut rsnn = RecurrentSNN::new(10, vec![20], 5, SNNConfig::default()).unwrap();
        let input = SpikeTensor::zeros(2, 30, 10, false);

        let output = rsnn.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 30, 5));
    }

    #[test]
    fn test_lsm_creation() {
        let lsm = LiquidStateMachine::new(10, 100, 5, 0.9, SNNConfig::default());
        assert_eq!(lsm.reservoir_size, 100);
        assert_eq!(lsm.output_size, 5);
    }

    #[test]
    fn test_lsm_forward() {
        let mut lsm = LiquidStateMachine::new(10, 50, 3, 0.9, SNNConfig::default());
        let input = SpikeTensor::zeros(1, 20, 10, false);

        let output = lsm.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 20, 3));
    }

    #[test]
    fn test_lsm_reservoir_state() {
        let lsm = LiquidStateMachine::new(5, 20, 2, 0.9, SNNConfig::default());
        let states = lsm.get_reservoir_state();
        assert_eq!(states.len(), 0); // No state until forward pass
    }
}
