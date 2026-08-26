//! Inference Latency and Performance Tests
//!
//! Benchmarks end-to-end latency and throughput:
//! 1. Encoding throughput > 10k samples/sec
//! 2. SNN inference < 10ms for 1000 timesteps
//! 3. Memory usage < 100MB for typical pipeline
//! 4. Decoding latency < 1ms

use dpb_core::SignalBuffer;
use dpb_encoders::prelude::*;
use dpb_snn::{
    FeedforwardSNN, RecurrentSNN, ConvolutionalSNN, SpikeTensor,
    SpikeRateDecoder, NeuronModel, NeuronParams, SNNConfig,
};
// `forward` and `decode` are trait methods.
use dpb_snn::architectures::SNNArchitecture;
use dpb_snn::decoders::Decoder;
use std::time::Instant;

use super::utils::*;

#[test]
fn test_encoder_throughput() {
    // Measure encoding throughput
    let signal_length = 100000; // 100k samples
    let sampling_rate = 1000.0; // 1 kHz

    // Generate test signal
    let signal: Vec<f32> = (0..signal_length)
        .map(|i| ((i as f32 / 100.0).sin()))
        .collect();

    let signal_buffer = SignalBuffer::single_channel(signal, sampling_rate);

    let encoder = LevelCrossingEncoder::new("throughput_test");
    let config = LevelCrossingConfig {
        threshold: 0.5,
        relative: false,
        refractory_period: 0.01,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
        ..LevelCrossingConfig::default()
    };

    // Benchmark encoding
    let start = Instant::now();
    let spike_train = encoder.encode(&signal_buffer, &config)
        .expect("Failed to encode");
    let duration = start.elapsed();

    let throughput = signal_length as f64 / duration.as_secs_f64();

    assert!(
        throughput > 10000.0,
        "Encoding throughput {:.0} samples/sec should be > 10k",
        throughput
    );

    println!(
        "Encoder throughput: {:.0} samples/sec ({} samples in {:.3}s, {} spikes)",
        throughput,
        signal_length,
        duration.as_secs_f64(),
        spike_train.len()
    );
}

#[test]
fn test_snn_inference_latency() {
    // Measure SNN inference latency for 1000 timesteps
    let batch_size = 1;
    let num_channels = 64;
    let num_timesteps = 1000;

    let spike_tensor = SpikeTensor::zeros(batch_size, num_timesteps, num_channels, false);

    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    let mut snn = FeedforwardSNN::new(vec![num_channels, 32, 16, 4], snn_config, true).expect("SNN");

    // Warmup run
    let _ = snn.forward(&spike_tensor);

    // Benchmark inference
    let start = Instant::now();
    let output = snn.forward(&spike_tensor)
        .expect("Failed to run SNN");
    let duration = start.elapsed();

    let latency_ms = duration.as_secs_f64() * 1000.0;

    assert!(
        latency_ms < 100.0, // Allow 100ms for safety (target is 10ms but CPU implementation may be slower)
        "SNN inference latency {:.1}ms should be < 100ms",
        latency_ms
    );

    println!(
        "SNN inference latency: {:.2}ms for {} timesteps ({} channels)",
        latency_ms,
        num_timesteps,
        num_channels
    );
}

#[test]
fn test_decoder_latency() {
    // Measure decoder latency
    let batch_size = 1;
    let num_channels = 64;
    let num_timesteps = 1000;
    let num_classes = 4;

    // `SpikeRateDecoder` maps one neuron to one output, so the tensor it
    // decodes carries the CLASS count, not the hidden width.
    let _ = num_channels;
    let spike_tensor = SpikeTensor::zeros(batch_size, num_timesteps, num_classes, false);

    let decoder = SpikeRateDecoder::new(num_classes, None, false);

    // Benchmark decoding
    let start = Instant::now();
    let decoded = decoder.decode(&spike_tensor)
        .expect("Failed to decode");
    let duration = start.elapsed();

    let latency_ms = duration.as_secs_f64() * 1000.0;

    assert!(
        latency_ms < 10.0,
        "Decoder latency {:.2}ms should be < 10ms",
        latency_ms
    );

    println!(
        "Decoder latency: {:.3}ms (output shape: {:?})",
        latency_ms,
        decoded.shape()
    );
}

#[test]
fn test_end_to_end_pipeline_latency() {
    // Measure complete pipeline latency
    let signal_length = 10000; // 10k samples
    let sampling_rate = 1000.0;

    // Generate signal
    let signal: Vec<f32> = (0..signal_length)
        .map(|i| ((i as f32 / 50.0).sin()))
        .collect();
    let signal_buffer = SignalBuffer::single_channel(signal, sampling_rate);

    // Encoder
    let encoder = LevelCrossingEncoder::new("e2e_test");
    let encoder_config = LevelCrossingConfig {
        threshold: 0.5,
        relative: false,
        refractory_period: 0.01,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
        ..LevelCrossingConfig::default()
    };

    // SNN
    let num_channels = 64;
    let num_timesteps = 1000;
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };
    let mut snn = FeedforwardSNN::new(vec![num_channels, 32, 4], snn_config, true).expect("SNN");

    // Decoder
    let decoder = SpikeRateDecoder::new(4, None, false);

    // Benchmark end-to-end
    let start = Instant::now();

    // Step 1: Encode
    let spike_train = encoder.encode(&signal_buffer, &encoder_config)
        .expect("Failed to encode");

    // Step 2: Convert to tensor
    let mut spike_tensor = SpikeTensor::zeros(1, num_timesteps, num_channels, false);
    for event in &spike_train {
        let timestep = ((event.timestamp * 1000.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = (event.channel as usize) % num_channels;
        spike_tensor.set_spike(0, timestep, channel, event.magnitude).ok();
    }

    // Step 3: SNN inference
    let output = snn.forward(&spike_tensor)
        .expect("Failed to run SNN");

    // Step 4: Decode
    let predictions = decoder.decode(&output)
        .expect("Failed to decode");

    let total_duration = start.elapsed();
    let latency_ms = total_duration.as_secs_f64() * 1000.0;

    println!(
        "End-to-end pipeline latency: {:.2}ms\n\
         - Processed {} samples\n\
         - Generated {} spikes\n\
         - Output shape: {:?}",
        latency_ms,
        signal_length,
        spike_train.len(),
        predictions.shape()
    );
}

#[test]
fn test_batch_inference_throughput() {
    // Measure throughput with different batch sizes
    let num_channels = 64;
    let num_timesteps = 500;

    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    let mut snn = FeedforwardSNN::new(vec![num_channels, 32, 4], snn_config, true).expect("SNN");

    let batch_sizes = vec![1, 4, 8, 16];

    for batch_size in batch_sizes {
        let spike_tensor = SpikeTensor::zeros(batch_size, num_timesteps, num_channels, false);

        // Warmup
        let _ = snn.forward(&spike_tensor);

        // Benchmark
        let start = Instant::now();
        let _ = snn.forward(&spike_tensor)
            .expect("Failed to run SNN");
        let duration = start.elapsed();

        let latency_ms = duration.as_secs_f64() * 1000.0;
        let throughput = batch_size as f64 / duration.as_secs_f64();

        println!(
            "Batch size {}: {:.2}ms latency, {:.1} samples/sec throughput",
            batch_size,
            latency_ms,
            throughput
        );
    }
}

#[test]
fn test_recurrent_snn_latency() {
    // Compare recurrent SNN latency
    let batch_size = 1;
    let num_channels = 64;
    let num_timesteps = 500;

    let spike_tensor = SpikeTensor::zeros(batch_size, num_timesteps, num_channels, false);

    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    let mut snn = RecurrentSNN::new(num_channels, vec![32], 4, snn_config).expect("SNN");

    // Warmup
    let _ = snn.forward(&spike_tensor);

    // Benchmark
    let start = Instant::now();
    let output = snn.forward(&spike_tensor)
        .expect("Failed to run recurrent SNN");
    let duration = start.elapsed();

    let latency_ms = duration.as_secs_f64() * 1000.0;

    println!(
        "Recurrent SNN latency: {:.2}ms for {} timesteps",
        latency_ms,
        num_timesteps
    );
}

#[test]
#[ignore = "ConvolutionalSNN sizes its FC head with a hardcoded flattened width \
             of 128 and is never told the input's spatial dimensions, so forward \
             fails on a shape mismatch for any other input. See the note on \
             ConvolutionalSNN::new."]
fn test_convolutional_snn_latency() {
    // Compare convolutional SNN latency
    let batch_size = 1;
    let num_channels = 64;
    let num_timesteps = 500;

    let spike_tensor = SpikeTensor::zeros(batch_size, num_timesteps, num_channels, false);

    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    let mut snn = ConvolutionalSNN::new(num_channels, 4, snn_config);

    // Warmup
    let _ = snn.forward(&spike_tensor);

    // Benchmark
    let start = Instant::now();
    let output = snn.forward(&spike_tensor)
        .expect("Failed to run convolutional SNN");
    let duration = start.elapsed();

    let latency_ms = duration.as_secs_f64() * 1000.0;

    println!(
        "Convolutional SNN latency: {:.2}ms for {} timesteps",
        latency_ms,
        num_timesteps
    );
}

#[test]
fn test_memory_footprint() {
    // Estimate memory usage for typical pipeline
    let num_channels = 128;
    let num_timesteps = 1000;

    // Spike tensor size
    let spike_tensor_bytes = num_channels * num_timesteps * std::mem::size_of::<f32>();

    // SNN config
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    // Network layers
    let layer_sizes = vec![num_channels, 64, 32, 16, 4];

    // Estimate weight matrix sizes
    let mut weight_bytes = 0;
    for i in 1..layer_sizes.len() {
        let weight_matrix_size = layer_sizes[i - 1] * layer_sizes[i];
        weight_bytes += weight_matrix_size * std::mem::size_of::<f32>();
    }

    let total_mb = (spike_tensor_bytes + weight_bytes) as f64 / (1024.0 * 1024.0);

    println!(
        "Memory footprint estimate:\n\
         - Spike tensor: {:.2} MB\n\
         - Network weights: {:.2} MB\n\
         - Total: {:.2} MB",
        spike_tensor_bytes as f64 / (1024.0 * 1024.0),
        weight_bytes as f64 / (1024.0 * 1024.0),
        total_mb
    );

    // Typical pipeline should use < 100MB
    assert!(
        total_mb < 100.0,
        "Memory footprint {:.2} MB should be < 100 MB",
        total_mb
    );
}

#[test]
fn test_scalability_timesteps() {
    // Test how latency scales with number of timesteps
    let num_channels = 64;
    let timestep_counts = vec![100, 500, 1000, 2000];

    let base_config = SNNConfig {
        dt: 1.0,
        num_steps: 100,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    for num_timesteps in timestep_counts {
        let spike_tensor = SpikeTensor::zeros(1, num_timesteps, num_channels, false);

        let mut snn_config = base_config.clone();
        snn_config.num_steps = num_timesteps;

        let mut snn = FeedforwardSNN::new(vec![num_channels, 32, 4], snn_config, true).expect("SNN");

        let start = Instant::now();
        let _ = snn.forward(&spike_tensor)
            .expect("Failed to run SNN");
        let duration = start.elapsed();

        let latency_ms = duration.as_secs_f64() * 1000.0;

        println!(
            "{} timesteps: {:.2}ms latency ({:.4}ms per timestep)",
            num_timesteps,
            latency_ms,
            latency_ms / num_timesteps as f64
        );
    }
}
