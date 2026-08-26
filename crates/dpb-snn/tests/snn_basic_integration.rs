//! Basic integration tests for SNN pipelines
//!
//! These tests verify basic end-to-end functionality without relying on
//! advanced features that may not be fully implemented.

use dpb_snn::*;
use dpb_snn::architectures::SNNArchitecture;
use dpb_snn::tensor::SpikeTensor;
use dpb_snn::decoders::{SpikeRateDecoder, Decoder};
use ndarray::Array3;

/// Create a simple test spike tensor
fn create_test_spikes(batch_size: usize, num_timesteps: usize, num_neurons: usize) -> SpikeTensor {
    let mut dense = Array3::zeros((batch_size, num_timesteps, num_neurons));

    // Add some spikes
    for b in 0..batch_size {
        for n in 0..num_neurons {
            for t in (n * 5..num_timesteps).step_by(15) {
                if t < num_timesteps {
                    dense[[b, t, n]] = 1.0;
                }
            }
        }
    }

    SpikeTensor::from_dense(dense, false)
}

#[test]
fn test_basic_snn_forward_pass() {
    println!("Testing basic SNN forward pass");

    let batch_size = 2;
    let num_input = 32;
    let num_hidden = 16;
    let num_output = 8;
    let num_timesteps = 50;

    // Create config
    let config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..Default::default()
    };

    // Create SNN
    let mut snn = FeedforwardSNN::new(
        vec![num_input, num_hidden, num_output],
        config,
        true,
    ).expect("Failed to create SNN");

    // Create input
    let input = create_test_spikes(batch_size, num_timesteps, num_input);

    // Forward pass
    let output = snn.forward(&input).expect("Forward pass failed");

    // Verify output shape
    let (out_batch, out_time, out_neurons) = output.shape();
    assert_eq!(out_batch, batch_size);
    assert_eq!(out_time, num_timesteps);
    assert_eq!(out_neurons, num_output);

    println!("  ✓ Forward pass successful");
    println!("    Input shape: ({}, {}, {})", batch_size, num_timesteps, num_input);
    println!("    Output shape: ({}, {}, {})", out_batch, out_time, out_neurons);
}

#[test]
fn test_snn_with_decoder() {
    println!("Testing SNN with rate decoder");

    let batch_size = 4;
    let num_input = 64;
    let num_output = 10;
    let num_timesteps = 100;

    // Create config
    let config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..Default::default()
    };

    // Create SNN
    let mut snn = FeedforwardSNN::new(
        vec![num_input, 32, num_output],
        config,
        true,
    ).expect("Failed to create SNN");

    // Create input
    let input = create_test_spikes(batch_size, num_timesteps, num_input);

    // Forward pass
    let output = snn.forward(&input).expect("Forward pass failed");

    // Decode
    let decoder = SpikeRateDecoder::new(num_output, None, false);
    let decoded = decoder.decode(&output).expect("Decoding failed");

    // Verify decoded shape
    assert_eq!(decoded.shape(), &[batch_size, num_output]);

    // Verify all values are valid
    for val in decoded.iter() {
        assert!(val.is_finite(), "Decoded value should be finite");
        assert!(*val >= 0.0, "Decoded value should be non-negative");
    }

    println!("  ✓ Decoding successful");
    println!("    Decoded shape: {:?}", decoded.shape());
}

#[test]
fn test_multiple_snn_architectures() {
    println!("Testing multiple SNN architectures");

    let batch_size = 2;
    let num_input = 32;
    let num_output = 5;
    let num_timesteps = 50;

    let config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..Default::default()
    };

    let input = create_test_spikes(batch_size, num_timesteps, num_input);

    // Test different architectures
    let architectures = vec![
        ("Shallow", vec![num_input, num_output]),
        ("Medium", vec![num_input, 20, num_output]),
        ("Deep", vec![num_input, 32, 16, num_output]),
    ];

    for (name, layers) in architectures {
        let mut snn = FeedforwardSNN::new(layers.clone(), config.clone(), true)
            .expect("Failed to create SNN");
        let output = snn.forward(&input).unwrap_or_else(|_| panic!("{} forward failed", name));

        let (out_batch, out_time, out_neurons) = output.shape();
        assert_eq!(out_batch, batch_size);
        assert_eq!(out_neurons, num_output);

        println!("  ✓ {} architecture: {:?}", name, layers);
    }
}

#[test]
fn test_spike_rate_computation() {
    println!("Testing spike rate computation");

    let batch_size = 3;
    let num_timesteps = 100;
    let num_neurons = 20;

    let spikes = create_test_spikes(batch_size, num_timesteps, num_neurons);

    // Get spike rate
    let spike_rate = spikes.spike_rate();

    // Verify shape
    assert_eq!(spike_rate.shape(), &[batch_size, num_neurons]);

    // Verify all rates are in valid range [0, 1]
    for &rate in spike_rate.iter() {
        assert!((0.0..=1.0).contains(&rate), "Spike rate {} out of range", rate);
    }

    println!("  ✓ Spike rate computation successful");
    println!("    Shape: {:?}", spike_rate.shape());
}

#[test]
fn test_neuron_models() {
    println!("Testing different neuron models");

    let batch_size = 2;
    let num_input = 16;
    let num_output = 4;
    let num_timesteps = 50;

    let input = create_test_spikes(batch_size, num_timesteps, num_input);

    let models = vec![
        NeuronModel::LIF,
        NeuronModel::AdaptiveLIF,
    ];

    for model in models {
        let config = SNNConfig {
            dt: 1.0,
            num_steps: num_timesteps,
            neuron_model: model,
            neuron_params: NeuronParams::default(),
            ..Default::default()
        };

        let mut snn = FeedforwardSNN::new(
            vec![num_input, num_output],
            config,
            true,
        ).expect("Failed to create SNN");

        let output = snn.forward(&input).expect("Forward pass failed");

        let (out_batch, _, out_neurons) = output.shape();
        assert_eq!(out_batch, batch_size);
        assert_eq!(out_neurons, num_output);

        println!("  ✓ {:?} model working", model);
    }
}

#[test]
fn test_batch_processing() {
    println!("Testing batch processing");

    let num_input = 32;
    let num_output = 8;
    let num_timesteps = 50;

    let config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..Default::default()
    };

    let mut snn = FeedforwardSNN::new(
        vec![num_input, 20, num_output],
        config,
        true,
    ).expect("Failed to create SNN");

    // Test different batch sizes
    let batch_sizes = vec![1, 2, 4, 8];

    for batch_size in batch_sizes {
        let input = create_test_spikes(batch_size, num_timesteps, num_input);
        let output = snn.forward(&input).expect("Forward pass failed");

        let (out_batch, _, _) = output.shape();
        assert_eq!(out_batch, batch_size);

        println!("  ✓ Batch size {} processed", batch_size);
    }
}
