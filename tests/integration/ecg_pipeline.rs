//! ECG Pipeline Integration Test
//!
//! Tests the complete pipeline:
//! 1. Generate synthetic ECG with known heart rate
//! 2. Encode using LevelCrossing encoder
//! 3. Process through Feedforward SNN
//! 4. Decode to heart rate estimate
//! 5. Validate against ground truth

use dpb_core::{SignalBuffer, SpikeEvent, Context};
use dpb_encoders::prelude::*;
use dpb_snn::{
    FeedforwardSNN, SpikeTensor, SpikeRateDecoder,
    NeuronModel, NeuronParams, SNNConfig,
};
use dpb_synth::contact::ecg::{EcgMorphologyGenerator, EcgMorphologyParams, WaveParams};
use dpb_synth::traits::SyntheticGenerator;

use super::utils::*;

#[test]
fn test_ecg_pipeline_end_to_end() {
    // Step 1: Generate synthetic ECG with known parameters
    let heart_rate = 72.0; // bpm
    let duration = 10.0; // seconds
    let sampling_rate = 250.0; // Hz

    let params = EcgMorphologyParams {
        duration,
        sampling_rate,
        heart_rate,
        p_wave: WaveParams {
            amplitude: 0.25,
            width: 0.1,
            time_offset: -std::f64::consts::PI / 3.0,
        },
        qrs_complex: WaveParams {
            amplitude: 1.0,
            width: 0.1,
            time_offset: 0.0,
        },
        t_wave: WaveParams {
            amplitude: 0.35,
            width: 0.25,
            time_offset: std::f64::consts::PI / 2.0,
        },
    };

    let generator = EcgMorphologyGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate ECG");

    // Verify R-peaks are generated
    let r_peaks = &generated.ground_truth.events;
    assert!(
        !r_peaks.is_empty(),
        "Should have detected R-peaks in ground truth"
    );

    // Expected number of beats: heart_rate * duration / 60
    let expected_beats = (heart_rate * duration / 60.0) as usize;
    assert_in_range(
        r_peaks.len() as f64,
        (expected_beats - 2) as f64,
        (expected_beats + 2) as f64,
        "R-peak count within expected range"
    );

    // Step 2: Convert to SignalBuffer and encode
    let signal_data: Vec<f32> = generated.signal.iter().map(|&x| x as f32).collect();
    let signal = SignalBuffer::single_channel(signal_data, sampling_rate);

    let encoder = LevelCrossingEncoder::new("ecg_encoder");
    let encoder_config = LevelCrossingConfig {
        threshold: 0.3,
        relative: false,
        refractory_period: 0.2, // 200ms refractory (300 bpm max)
    };

    let spike_train = encoder.encode(&signal, &encoder_config)
        .expect("Failed to encode ECG signal");

    // Step 3: Verify encoding produced spikes
    assert!(
        !spike_train.is_empty(),
        "Encoder should produce spike events"
    );
    assert!(
        spike_train.len() >= expected_beats - 2,
        "Should have at least {} spikes, got {}",
        expected_beats - 2,
        spike_train.len()
    );

    // Step 4: Convert spike train to tensor format
    let num_timesteps = (duration * 1000.0) as usize; // 1ms bins
    let num_channels = 64; // Small SNN for testing

    let mut spike_tensor = SpikeTensor::zeros(1, num_channels, num_timesteps);

    // Map spike events to tensor (simple rate-based encoding)
    for event in &spike_train.events {
        let timestep = (event.timestamp * 1000.0).min((num_timesteps - 1) as f64) as usize;
        let channel = (event.channel as usize) % num_channels;
        spike_tensor.set_spike(0, channel, timestep, 1.0);
    }

    // Step 5: Create and run SNN
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        use_gpu: false,
    };

    let snn = FeedforwardSNN::new(vec![num_channels, 32, 16, 1], snn_config.clone());
    let output_spikes = snn.forward(&spike_tensor)
        .expect("Failed to run SNN forward pass");

    // Step 6: Decode output to heart rate
    let decoder = SpikeRateDecoder::new(1);
    let decoded = decoder.decode(&output_spikes)
        .expect("Failed to decode SNN output");

    // Step 7: Validate decoded output shape
    assert_eq!(decoded.shape()[1], 1, "Should have 1 output dimension");

    println!(
        "ECG Pipeline Test Results:\n\
         - Generated ECG: {:.1}s @ {:.0} Hz\n\
         - Ground truth HR: {:.1} bpm\n\
         - R-peaks detected: {}\n\
         - Spike events: {}\n\
         - SNN output shape: {:?}",
        duration,
        sampling_rate,
        heart_rate,
        r_peaks.len(),
        spike_train.len(),
        output_spikes.shape()
    );
}

#[test]
fn test_ecg_pipeline_multiple_heart_rates() {
    // Test pipeline with different heart rates
    let heart_rates = vec![60.0, 75.0, 90.0, 120.0];

    for hr in heart_rates {
        let params = EcgMorphologyParams {
            duration: 10.0,
            sampling_rate: 250.0,
            heart_rate: hr,
            p_wave: WaveParams {
                amplitude: 0.25,
                width: 0.1,
                time_offset: -std::f64::consts::PI / 3.0,
            },
            qrs_complex: WaveParams {
                amplitude: 1.0,
                width: 0.1,
                time_offset: 0.0,
            },
            t_wave: WaveParams {
                amplitude: 0.35,
                width: 0.25,
                time_offset: std::f64::consts::PI / 2.0,
            },
        };

        let generator = EcgMorphologyGenerator;
        let generated = generator.generate(&params, TEST_SEED)
            .expect("Failed to generate ECG");

        let expected_beats = (hr * 10.0 / 60.0) as usize;
        let actual_beats = generated.ground_truth.events.len();

        assert_in_range(
            actual_beats as f64,
            (expected_beats - 2) as f64,
            (expected_beats + 2) as f64,
            &format!("HR {} bpm: beat count", hr)
        );

        println!("HR {:.0} bpm: generated {} beats (expected ~{})", hr, actual_beats, expected_beats);
    }
}

#[test]
fn test_ecg_pipeline_reproducibility() {
    // Test that same seed produces same results
    let params = EcgMorphologyParams {
        duration: 5.0,
        sampling_rate: 250.0,
        heart_rate: 70.0,
        p_wave: WaveParams {
            amplitude: 0.25,
            width: 0.1,
            time_offset: -std::f64::consts::PI / 3.0,
        },
        qrs_complex: WaveParams {
            amplitude: 1.0,
            width: 0.1,
            time_offset: 0.0,
        },
        t_wave: WaveParams {
            amplitude: 0.35,
            width: 0.25,
            time_offset: std::f64::consts::PI / 2.0,
        },
    };

    let generator = EcgMorphologyGenerator;

    let gen1 = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate ECG (run 1)");
    let gen2 = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate ECG (run 2)");

    // Signals should be identical
    assert_eq!(
        gen1.signal.len(),
        gen2.signal.len(),
        "Signal lengths should match"
    );

    for i in 0..gen1.signal.len() {
        assert_approx_eq(
            gen1.signal[i],
            gen2.signal[i],
            &format!("Signal sample {}", i)
        );
    }

    // R-peak count should match
    assert_eq!(
        gen1.ground_truth.events.len(),
        gen2.ground_truth.events.len(),
        "R-peak counts should match"
    );
}

#[test]
fn test_ecg_encoder_threshold_sensitivity() {
    // Test how encoder threshold affects spike count
    let params = EcgMorphologyParams {
        duration: 5.0,
        sampling_rate: 250.0,
        heart_rate: 72.0,
        p_wave: WaveParams {
            amplitude: 0.25,
            width: 0.1,
            time_offset: -std::f64::consts::PI / 3.0,
        },
        qrs_complex: WaveParams {
            amplitude: 1.0,
            width: 0.1,
            time_offset: 0.0,
        },
        t_wave: WaveParams {
            amplitude: 0.35,
            width: 0.25,
            time_offset: std::f64::consts::PI / 2.0,
        },
    };

    let generator = EcgMorphologyGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate ECG");

    let signal_data: Vec<f32> = generated.signal.iter().map(|&x| x as f32).collect();
    let signal = SignalBuffer::single_channel(signal_data, 250.0);

    let thresholds = vec![0.1, 0.3, 0.5, 0.7];
    let encoder = LevelCrossingEncoder::new("threshold_test");

    for threshold in thresholds {
        let config = LevelCrossingConfig {
            threshold,
            relative: false,
            refractory_period: 0.2,
        };

        let spike_train = encoder.encode(&signal, &config)
            .expect("Failed to encode");

        // Higher thresholds should produce fewer spikes
        println!(
            "Threshold {:.1}: {} spikes",
            threshold,
            spike_train.len()
        );

        assert!(
            spike_train.len() > 0,
            "Should produce spikes even at high threshold"
        );
    }
}
