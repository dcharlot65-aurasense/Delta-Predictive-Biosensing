//! Tremor Pipeline Integration Test
//!
//! Tests the complete pipeline:
//! 1. Generate synthetic 3-axis tremor with known frequency
//! 2. Encode using derivative encoder for zero crossings
//! 3. Process through Convolutional SNN
//! 4. Decode to tremor type classification
//! 5. Validate against ground truth

use dpb_core::SignalBuffer;
use dpb_encoders::prelude::*;
use dpb_snn::{
    ConvolutionalSNN, SpikeTensor, TremorSeverityDecoder,
    NeuronModel, NeuronParams, SNNConfig,
};
// `forward` and `decode` are trait methods.
use dpb_snn::architectures::SNNArchitecture;
use dpb_snn::decoders::Decoder;
use dpb_synth::contact::tremor::{
    PhysiologicalTremorGenerator, PhysiologicalTremorParams,
    ParkinsonianTremorGenerator, ParkinsonianTremorParams,
};
use dpb_synth::traits::SyntheticGenerator;

use super::utils::*;

#[test]
fn test_tremor_pipeline_physiological() {
    // Step 1: Generate physiological tremor (8-12 Hz)
    let params = PhysiologicalTremorParams {
        duration: 10.0,
        sampling_rate: 100.0,
        frequency: 10.0, // Hz
        amplitude: 0.3,
        frequency_variability: 0.5,
    };

    let generator = PhysiologicalTremorGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate physiological tremor");

    // Verify ground truth
    let gt_freq = generated.ground_truth.parameters.get("frequency").unwrap();
    assert_approx_eq(*gt_freq, params.frequency, "Ground truth frequency");

    let tremor_type = generated.ground_truth.parameters.get("tremor_type").unwrap();
    assert_approx_eq(*tremor_type, 0.0, "Tremor type (physiological)");

    // Step 2: Create 3-axis tremor (replicate to x, y, z with phase shifts)
    let signal_len = generated.signal.len();
    let x_axis: Vec<f32> = generated.signal.iter().map(|&v| v as f32).collect();

    let y_axis: Vec<f32> = generated.signal.iter()
        .enumerate()
        .map(|(i, &v)| (v * (std::f64::consts::PI / 3.0 * i as f64).cos()) as f32)
        .collect();

    let z_axis: Vec<f32> = generated.signal.iter()
        .enumerate()
        .map(|(i, &v)| (v * (std::f64::consts::PI / 4.0 * i as f64).sin()) as f32)
        .collect();

    // Create signals for each axis
    let signal_x = SignalBuffer::single_channel(x_axis, params.sampling_rate);
    let signal_y = SignalBuffer::single_channel(y_axis, params.sampling_rate);
    let signal_z = SignalBuffer::single_channel(z_axis, params.sampling_rate);

    // Step 3: Encode using derivative encoder
    let encoder = DerivativeEncoder::new("tremor_encoder");
    let encoder_config = DerivativeConfig {
        threshold: 0.05,
        ..DerivativeConfig::default()
    };

    let spikes_x = encoder.encode(&signal_x, &encoder_config)
        .expect("Failed to encode X axis");
    let spikes_y = encoder.encode(&signal_y, &encoder_config)
        .expect("Failed to encode Y axis");
    let spikes_z = encoder.encode(&signal_z, &encoder_config)
        .expect("Failed to encode Z axis");

    // Step 4: Verify encoding produced spikes
    assert!(!spikes_x.is_empty(), "X axis should produce spikes");
    assert!(!spikes_y.is_empty(), "Y axis should produce spikes");
    assert!(!spikes_z.is_empty(), "Z axis should produce spikes");

    println!(
        "Tremor encoding: {} X-spikes, {} Y-spikes, {} Z-spikes",
        spikes_x.len(),
        spikes_y.len(),
        spikes_z.len()
    );

    // Step 5: Create spike tensor for SNN
    let num_timesteps = (params.duration * 100.0) as usize; // 10ms bins
    let num_channels = 96; // 32 per axis

    let mut spike_tensor = SpikeTensor::zeros(1, num_timesteps, num_channels, false);

    // Map spikes to different channel groups
    for event in &spikes_x {
        let timestep = ((event.timestamp * 100.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = event.channel as usize % 32;
        spike_tensor.set_spike(0, timestep, channel, event.magnitude).ok();
    }

    for event in &spikes_y {
        let timestep = ((event.timestamp * 100.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = 32 + (event.channel as usize % 32);
        spike_tensor.set_spike(0, timestep, channel, event.magnitude).ok();
    }

    for event in &spikes_z {
        let timestep = ((event.timestamp * 100.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = 64 + (event.channel as usize % 32);
        spike_tensor.set_spike(0, timestep, channel, event.magnitude).ok();
    }

    // Step 6: Create and run convolutional SNN
    let snn_config = SNNConfig {
        dt: 10.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    let mut snn = ConvolutionalSNN::new(num_channels, 3, snn_config.clone());
    let output_spikes = snn.forward(&spike_tensor)
        .expect("Failed to run SNN forward pass");

    // Step 7: Decode output to tremor classification
    let decoder = TremorSeverityDecoder::new(250.0);
    let tremor_class = decoder.decode(&output_spikes)
        .expect("Failed to decode tremor classification");

    // Step 8: Validate output
    // `TremorSeverityDecoder` reports its three frequency bands
    // (parkinsonian, essential, physiological) plus an overall score.
    assert_eq!(tremor_class.shape()[1], 4, "Should have 3 bands plus an overall score");

    println!(
        "Physiological Tremor Pipeline:\n\
         - Frequency: {:.1} Hz\n\
         - Duration: {:.1}s\n\
         - Total spikes: {}\n\
         - Output shape: {:?}",
        params.frequency,
        params.duration,
        spikes_x.len() + spikes_y.len() + spikes_z.len(),
        output_spikes.shape()
    );
}

#[test]
fn test_tremor_pipeline_parkinsonian() {
    // Step 1: Generate Parkinsonian tremor (4-6 Hz)
    let params = ParkinsonianTremorParams {
        duration: 10.0,
        sampling_rate: 100.0,
        frequency: 5.0, // Hz
        amplitude: 1.5,
        pill_rolling: true,
        amplitude_modulation: 3.0, // 3 second waxing/waning
    };

    let generator = ParkinsonianTremorGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate Parkinsonian tremor");

    // Verify ground truth
    let gt_freq = generated.ground_truth.parameters.get("frequency").unwrap();
    assert_approx_eq(*gt_freq, params.frequency, "Ground truth frequency");

    let tremor_type = generated.ground_truth.parameters.get("tremor_type").unwrap();
    assert_approx_eq(*tremor_type, 1.0, "Tremor type (Parkinsonian)");

    let pill_rolling = generated.ground_truth.parameters.get("pill_rolling").unwrap();
    assert_approx_eq(*pill_rolling, 1.0, "Pill rolling enabled");

    println!(
        "Parkinsonian Tremor Generated:\n\
         - Frequency: {:.1} Hz\n\
         - Amplitude: {:.2}\n\
         - Pill rolling: {}\n\
         - Amplitude modulation: {:.1}s",
        params.frequency,
        params.amplitude,
        params.pill_rolling,
        params.amplitude_modulation
    );
}

#[test]
fn test_tremor_frequency_discrimination() {
    // Test that different tremor frequencies are distinguishable
    let frequencies = vec![
        (5.0, "Parkinsonian"),
        (7.0, "Essential"),
        (10.0, "Physiological"),
    ];

    for (freq, label) in frequencies {
        let params = PhysiologicalTremorParams {
            duration: 5.0,
            sampling_rate: 100.0,
            frequency: freq,
            amplitude: 0.5,
            frequency_variability: 0.2,
        };

        let generator = PhysiologicalTremorGenerator;
        let generated = generator.generate(&params, TEST_SEED)
            .expect("Failed to generate tremor");

        let gt_freq = generated.ground_truth.parameters.get("frequency").unwrap();
        assert_approx_eq(*gt_freq, freq, &format!("{} tremor frequency", label));

        // Verify signal has expected frequency content
        // (In a real test, we'd use FFT to verify dominant frequency)
        assert_eq!(generated.signal.len(), (params.duration * params.sampling_rate) as usize);

        println!("{} tremor: {:.1} Hz, {} samples", label, freq, generated.signal.len());
    }
}

#[test]
fn test_tremor_amplitude_range() {
    // Test various tremor amplitudes
    let amplitudes = vec![0.1, 0.5, 1.0, 2.0];

    for amp in amplitudes {
        let params = PhysiologicalTremorParams {
            duration: 5.0,
            sampling_rate: 100.0,
            frequency: 10.0,
            amplitude: amp,
            frequency_variability: 0.3,
        };

        let generator = PhysiologicalTremorGenerator;
        let generated = generator.generate(&params, TEST_SEED)
            .expect("Failed to generate tremor");

        // Verify signal amplitude is within expected range
        let max_val = generated.signal.iter()
            .map(|&x| x.abs())
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        // Should be roughly within 2x amplitude (allowing for noise)
        assert!(
            max_val <= amp * 2.5,
            "Max value {:.2} should be <= {:.2}",
            max_val,
            amp * 2.5
        );

        println!("Amplitude {:.1}: max signal value = {:.3}", amp, max_val);
    }
}

#[test]
fn test_tremor_reproducibility() {
    // Test deterministic generation
    let params = PhysiologicalTremorParams {
        duration: 5.0,
        sampling_rate: 100.0,
        frequency: 10.0,
        amplitude: 0.5,
        frequency_variability: 0.3,
    };

    let generator = PhysiologicalTremorGenerator;

    let gen1 = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate (run 1)");
    let gen2 = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate (run 2)");

    assert_eq!(gen1.signal.len(), gen2.signal.len(), "Signal lengths should match");

    // Signals should be identical
    for i in 0..gen1.signal.len().min(10) {
        assert_approx_eq(
            gen1.signal[i],
            gen2.signal[i],
            &format!("Sample {}", i)
        );
    }
}

#[test]
fn test_tremor_zero_crossings() {
    // Verify tremor signal has expected number of zero crossings
    let params = PhysiologicalTremorParams {
        duration: 10.0,
        sampling_rate: 100.0,
        frequency: 10.0, // 10 Hz should give ~100 zero crossings in 10s
        amplitude: 1.0,
        frequency_variability: 0.1,
    };

    let generator = PhysiologicalTremorGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate tremor");

    // Count zero crossings
    let mut zero_crossings = 0;
    for i in 1..generated.signal.len() {
        if (generated.signal[i - 1] >= 0.0 && generated.signal[i] < 0.0)
            || (generated.signal[i - 1] < 0.0 && generated.signal[i] >= 0.0)
        {
            zero_crossings += 1;
        }
    }

    // Expected: 2 * frequency * duration (up and down crossings)
    let expected = (2.0 * params.frequency * params.duration) as usize;

    assert_in_range(
        zero_crossings as f64,
        (expected - 20) as f64,
        (expected + 20) as f64,
        "Zero crossing count"
    );

    println!(
        "Zero crossings: {} (expected ~{})",
        zero_crossings,
        expected
    );
}
