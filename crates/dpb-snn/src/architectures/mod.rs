//! Complete SNN architectures

pub mod feedforward;
pub mod convolutional;
pub mod recurrent;
pub mod graph;
pub mod transformer;

pub use feedforward::FeedforwardSNN;
pub use convolutional::{ConvolutionalSNN, SpikingVGG, SpikingResNet};
pub use recurrent::{RecurrentSNN, LiquidStateMachine};
pub use graph::SpikingGCN;
pub use transformer::SpikingTransformer;

use crate::{SNNConfig, SNNResult, SpikeTensor};

/// Base trait for all SNN architectures
pub trait SNNArchitecture: Send + Sync {
    /// Forward pass through the entire network
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor>;

    /// Reset all network states
    fn reset(&mut self);

    /// Get trainable parameters
    fn parameters(&self) -> Vec<&ndarray::Array2<f32>>;

    /// Get mutable trainable parameters
    fn parameters_mut(&mut self) -> Vec<&mut ndarray::Array2<f32>>;

    /// Zero all gradients
    fn zero_grad(&mut self);

    /// Get network configuration
    fn config(&self) -> &SNNConfig;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Simple test to verify module structure compiles
        assert!(true);
    }
}
