//! Complete SNN architectures

pub mod convolutional;
pub mod feedforward;
pub mod graph;
pub mod recurrent;
pub mod transformer;

pub use convolutional::{ConvolutionalSNN, SpikingResNet, SpikingVGG};
pub use feedforward::FeedforwardSNN;
pub use graph::SpikingGCN;
pub use recurrent::{LiquidStateMachine, RecurrentSNN};
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

    #[test]
    fn test_module_exports() {
        // Compiling this test is the check -- it names the module's items.
        // An assert!(true) added nothing to that.
    }

    // ---- save/load round-trip -----------------------------------------
    //
    // Every architecture derives Serialize/Deserialize, and every one of them
    // used to mark its layers `#[serde(skip)]`. A saved model was therefore its
    // configuration and nothing else: FeedforwardSNN(vec![4, 8, 2]) serialised
    // to 190 bytes, deserialised with zero layers, and its forward pass then
    // returned the input unchanged -- an identity function, of the wrong output
    // width, reported as success. These tests pin the round-trip.

    use super::{ConvolutionalSNN, FeedforwardSNN, RecurrentSNN, SNNArchitecture};
    use crate::SNNConfig;
    use crate::tensor::SpikeTensor;
    use ndarray::Array3;

    /// Drive strong enough that the network actually spikes; comparing two
    /// silent networks would pass no matter what was lost.
    fn drive(batch: usize, steps: usize, width: usize) -> SpikeTensor {
        SpikeTensor::from_dense(
            Array3::from_shape_fn((batch, steps, width), |(_, t, c)| {
                if (t + c) % 2 == 0 { 1.0 } else { 0.4 }
            }),
            false,
        )
    }

    #[test]
    fn feedforward_survives_a_save_load_round_trip() {
        let mut net = FeedforwardSNN::new(vec![6, 10, 3], SNNConfig::default(), true).unwrap();
        // Deterministic, and large enough to fire.
        for layer in &mut net.layers {
            let (rows, cols) = (layer.weights.shape()[0], layer.weights.shape()[1]);
            for i in 0..rows {
                for j in 0..cols {
                    layer.weights[[i, j]] = 0.45 + 0.05 * (i as f32) - 0.02 * (j as f32);
                }
            }
        }

        let input = drive(2, 24, 6);
        let before = net.forward(&input).unwrap().to_dense();
        assert!(
            before.iter().sum::<f32>() > 0.0,
            "the network never spiked, so the round-trip comparison would be vacuous"
        );

        let json = serde_json::to_string(&net).unwrap();
        let mut back: FeedforwardSNN = serde_json::from_str(&json).unwrap();

        assert_eq!(
            back.layers.len(),
            net.layers.len(),
            "layers were dropped by the round-trip"
        );
        let after = back.forward(&input).unwrap().to_dense();
        assert_eq!(
            before.shape(),
            after.shape(),
            "output shape changed across the round-trip"
        );
        assert_eq!(
            before, after,
            "the reloaded network computes something else"
        );
    }

    /// The same for the convolutional stack, whose kernels are 4-D.
    #[test]
    fn convolutional_survives_a_save_load_round_trip() {
        let mut net = ConvolutionalSNN::new(1, 4, SNNConfig::default());
        let input = drive(1, 6, 8 * 8);

        let before = net.forward(&input).unwrap().to_dense();
        let json = serde_json::to_string(&net).unwrap();
        let mut back: ConvolutionalSNN = serde_json::from_str(&json).unwrap();

        assert_eq!(back.conv_layers.len(), net.conv_layers.len());
        assert_eq!(back.fc_layers.len(), net.fc_layers.len());
        let after = back.forward(&input).unwrap().to_dense();
        assert_eq!(before.shape(), after.shape());
        assert_eq!(
            before, after,
            "the reloaded network computes something else"
        );
    }

    /// And the recurrent stack, whose weights include the recurrent matrix.
    #[test]
    fn recurrent_survives_a_save_load_round_trip() {
        let mut net = RecurrentSNN::new(5, vec![8], 3, SNNConfig::default()).unwrap();
        let input = drive(2, 20, 5);

        let before = net.forward(&input).unwrap().to_dense();
        let json = serde_json::to_string(&net).unwrap();
        let mut back: RecurrentSNN = serde_json::from_str(&json).unwrap();

        assert_eq!(back.recurrent_layers.len(), net.recurrent_layers.len());
        let after = back.forward(&input).unwrap().to_dense();
        assert_eq!(before.shape(), after.shape());
        assert_eq!(
            before, after,
            "the reloaded network computes something else"
        );
    }
}
