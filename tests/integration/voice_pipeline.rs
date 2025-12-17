//! Voice Pipeline Integration Test
//!
//! Tests the complete pipeline:
//! 1. Generate synthetic voice with known F0 and formants
//! 2. Encode using prosody encoder
//! 3. Process through Attention SNN
//! 4. Decode to voice quality assessment
//! 5. Validate against ground truth

use dpb_core::SignalBuffer;
use dpb_encoders::prelude::*;
use dpb_snn::{
    SpikingTransformer, SpikeTensor, SpikeRateDecoder,
    NeuronModel, NeuronParams, SNNConfig,
};
use dpb_synth::voice::phonation::{SustainedVowelGenerator, SustainedVowelParams};
use dpb_synth::traits::SyntheticGenerator;

use super::utils::*;

#[test]
fn test_voice_pipeline_sustained_vowel() {
    // Step 1: Generate sustained vowel /a/ with known F0 and formants
    let f0 = 120.0; // Hz (typical male)

    let params = SustainedVowelParams {
        duration: 3.0,
        sampling_rate: 16000.0,
        fundamental_frequency: f0,
        vowel_formants: vec![
            (730.0, 100.0),   // F1
            (1090.0, 150.0),  // F2
            (2440.0, 200.0),  // F3
        ],
        amplitude: 0.5,
    };

    let generator = SustainedVowelGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate vowel");

    // Verify ground truth
    let gt_f0 = generated.ground_truth.parameters.get("f0").unwrap();
    assert_approx_eq(*gt_f0, f0, "Ground truth F0");

    let f1_freq = generated.ground_truth.parameters.get("F1_freq").unwrap();
    assert_approx_eq(*f1_freq, 730.0, "Ground truth F1");

    // Step 2: Convert to SignalBuffer
    let signal_data: Vec<f32> = generated.signal.iter().map(|&x| x as f32).collect();
    let signal = SignalBuffer::single_channel(signal_data, params.sampling_rate);

    // Step 3: Encode using level crossing encoder
    let encoder = LevelCrossingEncoder::new("voice_encoder");
    let encoder_config = LevelCrossingConfig {
        threshold: 0.1,
        relative: false,
        refractory_period: 0.001, // 1ms refractory
    };

    let spike_train = encoder.encode(&signal, &encoder_config)
        .expect("Failed to encode voice signal");

    // Step 4: Verify encoding produced spikes
    assert!(
        !spike_train.is_empty(),
        "Encoder should produce spike events"
    );

    // Should have roughly f0 * duration crossings
    let expected_crossings = (f0 * params.duration * 2.0) as usize; // up and down
    assert!(
        spike_train.len() > expected_crossings / 2,
        "Should have at least {} spikes, got {}",
        expected_crossings / 2,
        spike_train.len()
    );

    println!(
        "Voice encoding: {} spikes (expected ~{})",
        spike_train.len(),
        expected_crossings
    );

    // Step 5: Create spike tensor for SNN
    let num_timesteps = (params.duration * 1000.0) as usize; // 1ms bins
    let num_channels = 128;

    let mut spike_tensor = SpikeTensor::zeros(1, num_channels, num_timesteps);

    // Map spike events to tensor
    for event in &spike_train.events {
        let timestep = ((event.timestamp * 1000.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = (event.channel as usize) % num_channels;
        spike_tensor.set_spike(0, channel, timestep, event.magnitude);
    }

    // Step 6: Create and run SNN with attention
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        use_gpu: false,
    };

    let snn = SpikingTransformer::new(num_channels, 4, 64, 4, snn_config.clone());
    let output_spikes = snn.forward(&spike_tensor)
        .expect("Failed to run SNN forward pass");

    // Step 7: Decode output
    let decoder = SpikeRateDecoder::new(4);
    let decoded = decoder.decode(&output_spikes)
        .expect("Failed to decode SNN output");

    // Step 8: Validate output shape
    assert_eq!(decoded.shape()[1], 4, "Should have 4 output dimensions");

    println!(
        "Voice Pipeline Test Results:\n\
         - F0: {:.1} Hz\n\
         - Duration: {:.1}s\n\
         - Formants: F1={:.0}Hz, F2={:.0}Hz, F3={:.0}Hz\n\
         - Spike events: {}\n\
         - SNN output shape: {:?}",
        f0,
        params.duration,
        params.vowel_formants[0].0,
        params.vowel_formants[1].0,
        params.vowel_formants[2].0,
        spike_train.len(),
        output_spikes.shape()
    );
}

#[test]
fn test_voice_pipeline_f0_variations() {
    // Test pipeline with different F0 values (male, female ranges)
    let f0_values = vec![
        (100.0, "low male"),
        (130.0, "typical male"),
        (200.0, "typical female"),
        (250.0, "high female"),
    ];

    for (f0, label) in f0_values {
        let params = SustainedVowelParams {
            duration: 2.0,
            sampling_rate: 16000.0,
            fundamental_frequency: f0,
            vowel_formants: vec![
                (730.0, 100.0),
                (1090.0, 150.0),
                (2440.0, 200.0),
            ],
            amplitude: 0.5,
        };

        let generator = SustainedVowelGenerator;
        let generated = generator.generate(&params, TEST_SEED)
            .expect("Failed to generate vowel");

        let gt_f0 = generated.ground_truth.parameters.get("f0").unwrap();
        assert_approx_eq(*gt_f0, f0, &format!("{} F0", label));

        println!("{}: F0 = {:.1} Hz, {} samples", label, f0, generated.signal.len());
    }
}

#[test]
fn test_voice_formant_encoding() {
    // Test different vowels with different formant patterns
    let vowels = vec![
        ("a", vec![(730.0, 100.0), (1090.0, 150.0), (2440.0, 200.0)]),
        ("i", vec![(270.0, 60.0), (2290.0, 150.0), (3010.0, 200.0)]),
        ("u", vec![(300.0, 60.0), (870.0, 120.0), (2240.0, 180.0)]),
    ];

    for (vowel, formants) in vowels {
        let params = SustainedVowelParams {
            duration: 2.0,
            sampling_rate: 16000.0,
            fundamental_frequency: 120.0,
            vowel_formants: formants.clone(),
            amplitude: 0.5,
        };

        let generator = SustainedVowelGenerator;
        let generated = generator.generate(&params, TEST_SEED)
            .expect("Failed to generate vowel");

        // Verify formants in ground truth
        let f1 = generated.ground_truth.parameters.get("F1_freq").unwrap();
        assert_approx_eq(*f1, formants[0].0, &format!("Vowel /{}/: F1", vowel));

        let f2 = generated.ground_truth.parameters.get("F2_freq").unwrap();
        assert_approx_eq(*f2, formants[1].0, &format!("Vowel /{}/: F2", vowel));

        println!(
            "Vowel /{}/ - F1: {:.0} Hz, F2: {:.0} Hz, F3: {:.0} Hz",
            vowel, formants[0].0, formants[1].0, formants[2].0
        );
    }
}

#[test]
fn test_voice_signal_duration() {
    // Test various durations
    let durations = vec![1.0, 2.0, 3.0, 5.0];

    for duration in durations {
        let params = SustainedVowelParams {
            duration,
            sampling_rate: 16000.0,
            fundamental_frequency: 120.0,
            vowel_formants: vec![
                (730.0, 100.0),
                (1090.0, 150.0),
                (2440.0, 200.0),
            ],
            amplitude: 0.5,
        };

        let generator = SustainedVowelGenerator;
        let generated = generator.generate(&params, TEST_SEED)
            .expect("Failed to generate vowel");

        let expected_samples = (duration * params.sampling_rate) as usize;
        assert_eq!(
            generated.signal.len(),
            expected_samples,
            "Duration {:.1}s should produce {} samples",
            duration,
            expected_samples
        );

        println!(
            "Duration {:.1}s: {} samples @ {:.0} Hz",
            duration,
            generated.signal.len(),
            params.sampling_rate
        );
    }
}

#[test]
fn test_voice_reproducibility() {
    // Test deterministic generation
    let params = SustainedVowelParams {
        duration: 2.0,
        sampling_rate: 16000.0,
        fundamental_frequency: 120.0,
        vowel_formants: vec![
            (730.0, 100.0),
            (1090.0, 150.0),
            (2440.0, 200.0),
        ],
        amplitude: 0.5,
    };

    let generator = SustainedVowelGenerator;

    let gen1 = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate (run 1)");
    let gen2 = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate (run 2)");

    assert_eq!(gen1.signal.len(), gen2.signal.len(), "Signal lengths should match");

    // First 100 samples should be identical
    for i in 0..100 {
        assert_approx_eq(
            gen1.signal[i],
            gen2.signal[i],
            &format!("Sample {}", i)
        );
    }
}

#[test]
fn test_voice_harmonic_content() {
    // Verify that signal has harmonic structure
    let params = SustainedVowelParams {
        duration: 2.0,
        sampling_rate: 16000.0,
        fundamental_frequency: 120.0,
        vowel_formants: vec![
            (730.0, 100.0),
            (1090.0, 150.0),
            (2440.0, 200.0),
        ],
        amplitude: 1.0,
    };

    let generator = SustainedVowelGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate vowel");

    // Verify signal is not constant
    let max_val = generated.signal.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = generated.signal.iter().fold(f64::INFINITY, |a, &b| a.min(b));

    assert!(
        max_val > min_val + 0.1,
        "Signal should have variation (max: {:.3}, min: {:.3})",
        max_val,
        min_val
    );

    // Verify signal has periodic structure (zero crossings)
    let mut zero_crossings = 0;
    for i in 1..generated.signal.len() {
        if (generated.signal[i - 1] >= 0.0 && generated.signal[i] < 0.0)
            || (generated.signal[i - 1] < 0.0 && generated.signal[i] >= 0.0)
        {
            zero_crossings += 1;
        }
    }

    // Should have roughly 2 * f0 * duration zero crossings
    let expected_crossings = (2.0 * params.fundamental_frequency * params.duration) as usize;
    assert!(
        zero_crossings > expected_crossings / 4,
        "Should have at least {} zero crossings, got {}",
        expected_crossings / 4,
        zero_crossings
    );

    println!(
        "Voice signal: {} zero crossings (expected ~{})",
        zero_crossings,
        expected_crossings
    );
}
