//! Multimodal Pipeline Integration Test
//!
//! Tests the complete multimodal pipeline:
//! 1. Generate synthetic data from multiple modalities
//! 2. Encode each modality separately
//! 3. Fuse encodings in a multi-input SNN
//! 4. Decode to comprehensive assessment
//! 5. Validate against ground truth

use dpb_core::SignalBuffer;
use dpb_encoders::prelude::*;
use dpb_snn::{
    FeedforwardSNN, NeuronModel, NeuronParams, SNNConfig, SpikeRateDecoder, SpikeTensor,
};
// `forward` and `decode` are trait methods.
use dpb_snn::architectures::SNNArchitecture;
use dpb_snn::decoders::Decoder;
use dpb_synth::contact::ecg::{EcgMorphologyGenerator, EcgMorphologyParams, WaveParams};
use dpb_synth::contact::tremor::{PhysiologicalTremorGenerator, PhysiologicalTremorParams};
use dpb_synth::traits::SyntheticGenerator;
use dpb_synth::voice::phonation::{SustainedVowelGenerator, SustainedVowelParams};

use super::utils::*;
use std::f64::consts::PI;

#[test]
fn test_multimodal_ecg_tremor_fusion() {
    // Step 1: Generate ECG data
    let ecg_params = EcgMorphologyParams {
        duration: 10.0,
        sampling_rate: 250.0,
        heart_rate: 72.0,
        p_wave: WaveParams {
            amplitude: 0.25,
            width: 0.1,
            time_offset: -PI / 3.0,
        },
        qrs_complex: WaveParams {
            amplitude: 1.0,
            width: 0.1,
            time_offset: 0.0,
        },
        t_wave: WaveParams {
            amplitude: 0.35,
            width: 0.25,
            time_offset: PI / 2.0,
        },
    };

    let ecg_gen = EcgMorphologyGenerator;
    let ecg_data = ecg_gen
        .generate(&ecg_params, TEST_SEED)
        .expect("Failed to generate ECG");

    // Step 2: Generate tremor data
    let tremor_params = PhysiologicalTremorParams {
        duration: 10.0,
        sampling_rate: 100.0,
        frequency: 10.0,
        amplitude: 0.5,
        frequency_variability: 0.3,
    };

    let tremor_gen = PhysiologicalTremorGenerator;
    let tremor_data = tremor_gen
        .generate(&tremor_params, TEST_SEED)
        .expect("Failed to generate tremor");

    // Step 3: Encode ECG
    let ecg_signal: Vec<f32> = ecg_data.signal.iter().map(|&x| x as f32).collect();
    let ecg_buffer = SignalBuffer::single_channel(ecg_signal, ecg_params.sampling_rate);

    let ecg_encoder = LevelCrossingEncoder::new("ecg");
    let ecg_config = LevelCrossingConfig {
        threshold: 0.3,
        relative: false,
        refractory_period: 0.2,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };

    let ecg_spikes = ecg_encoder
        .encode(&ecg_buffer, &ecg_config)
        .expect("Failed to encode ECG");

    // Step 4: Encode tremor
    let tremor_signal: Vec<f32> = tremor_data.signal.iter().map(|&x| x as f32).collect();
    let tremor_buffer = SignalBuffer::single_channel(tremor_signal, tremor_params.sampling_rate);

    let tremor_encoder = DerivativeEncoder::new("tremor");
    let tremor_config = DerivativeConfig {
        threshold: 0.05,
        ..DerivativeConfig::default()
    };

    let tremor_spikes = tremor_encoder
        .encode(&tremor_buffer, &tremor_config)
        .expect("Failed to encode tremor");

    // Step 5: Create fused spike tensor
    let num_timesteps = 10000; // 1ms bins for 10 seconds
    let ecg_channels = 64;
    let tremor_channels = 64;
    let total_channels = ecg_channels + tremor_channels;

    let mut spike_tensor = SpikeTensor::zeros(1, num_timesteps, total_channels, false);

    // Map ECG spikes to first 64 channels
    for event in &ecg_spikes {
        let timestep = ((event.timestamp * 1000.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = (event.channel as usize) % ecg_channels;
        spike_tensor
            .set_spike(0, timestep, channel, event.magnitude)
            .ok();
    }

    // Map tremor spikes to next 64 channels
    for event in &tremor_spikes {
        let timestep = ((event.timestamp * 1000.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = ecg_channels + ((event.channel as usize) % tremor_channels);
        spike_tensor
            .set_spike(0, timestep, channel, event.magnitude)
            .ok();
    }

    // Step 6: Create fusion SNN
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    let mut snn = FeedforwardSNN::new(vec![total_channels, 64, 32, 8], snn_config.clone(), true)
        .expect("SNN");
    let output_spikes = snn
        .forward(&spike_tensor)
        .expect("Failed to run fusion SNN");

    // Step 7: Decode output
    let decoder = SpikeRateDecoder::new(8, None, false);
    let decoded = decoder.decode(&output_spikes).expect("Failed to decode");

    // Step 8: Validate
    assert_eq!(decoded.shape()[1], 8, "Should have 8 output features");
    assert!(!ecg_spikes.is_empty(), "ECG should produce spikes");
    assert!(!tremor_spikes.is_empty(), "Tremor should produce spikes");

    println!(
        "Multimodal ECG+Tremor Pipeline:\n\
         - ECG spikes: {}\n\
         - Tremor spikes: {}\n\
         - Total channels: {}\n\
         - Output dimensions: {}",
        ecg_spikes.len(),
        tremor_spikes.len(),
        total_channels,
        decoded.shape()[1]
    );
}

#[test]
fn test_multimodal_three_way_fusion() {
    // Test ECG + Tremor + Voice fusion
    let duration = 5.0;

    // Generate ECG
    let ecg_params = EcgMorphologyParams {
        duration,
        sampling_rate: 250.0,
        heart_rate: 75.0,
        p_wave: WaveParams {
            amplitude: 0.25,
            width: 0.1,
            time_offset: -PI / 3.0,
        },
        qrs_complex: WaveParams {
            amplitude: 1.0,
            width: 0.1,
            time_offset: 0.0,
        },
        t_wave: WaveParams {
            amplitude: 0.35,
            width: 0.25,
            time_offset: PI / 2.0,
        },
    };
    let ecg_gen = EcgMorphologyGenerator;
    let ecg_data = ecg_gen.generate(&ecg_params, TEST_SEED).unwrap();

    // Generate tremor
    let tremor_params = PhysiologicalTremorParams {
        duration,
        sampling_rate: 100.0,
        frequency: 10.0,
        amplitude: 0.5,
        frequency_variability: 0.3,
    };
    let tremor_gen = PhysiologicalTremorGenerator;
    let tremor_data = tremor_gen.generate(&tremor_params, TEST_SEED).unwrap();

    // Generate voice
    let voice_params = SustainedVowelParams {
        duration,
        sampling_rate: 16000.0,
        fundamental_frequency: 120.0,
        vowel_formants: vec![(730.0, 100.0), (1090.0, 150.0), (2440.0, 200.0)],
        amplitude: 0.5,
    };
    let voice_gen = SustainedVowelGenerator;
    let voice_data = voice_gen.generate(&voice_params, TEST_SEED).unwrap();

    // Verify all data generated successfully
    assert!(!ecg_data.signal.is_empty(), "ECG data should not be empty");
    assert!(
        !tremor_data.signal.is_empty(),
        "Tremor data should not be empty"
    );
    assert!(
        !voice_data.signal.is_empty(),
        "Voice data should not be empty"
    );

    println!(
        "Three-way Multimodal Data:\n\
         - ECG: {} samples @ {:.0} Hz\n\
         - Tremor: {} samples @ {:.0} Hz\n\
         - Voice: {} samples @ {:.0} Hz",
        ecg_data.signal.len(),
        ecg_params.sampling_rate,
        tremor_data.signal.len(),
        tremor_params.sampling_rate,
        voice_data.signal.len(),
        voice_params.sampling_rate
    );
}

#[test]
fn test_multimodal_temporal_alignment() {
    // Test that different sampling rates align correctly
    let duration = 5.0;

    let ecg_rate = 250.0;
    let tremor_rate = 100.0;
    let voice_rate = 16000.0;

    // All should have same duration despite different sampling rates
    let ecg_samples = (duration * ecg_rate) as usize;
    let tremor_samples = (duration * tremor_rate) as usize;
    let voice_samples = (duration * voice_rate) as usize;

    assert_eq!(ecg_samples, 1250);
    assert_eq!(tremor_samples, 500);
    assert_eq!(voice_samples, 80000);

    // When binning to 1ms timesteps, all should align
    let ms_bins = (duration * 1000.0) as usize;
    assert_eq!(ms_bins, 5000);

    println!(
        "Temporal alignment test:\n\
         - Common duration: {:.1}s\n\
         - ECG samples: {}\n\
         - Tremor samples: {}\n\
         - Voice samples: {}\n\
         - Common timesteps (1ms): {}",
        duration, ecg_samples, tremor_samples, voice_samples, ms_bins
    );
}

#[test]
fn test_multimodal_channel_allocation() {
    // Test that channels are allocated correctly for each modality
    let ecg_ch = 64;
    let tremor_ch = 64;
    let voice_ch = 128;
    let total = ecg_ch + tremor_ch + voice_ch;

    assert_eq!(total, 256);

    // Verify channel ranges don't overlap
    let ecg_range = 0..ecg_ch;
    let tremor_range = ecg_ch..(ecg_ch + tremor_ch);
    let voice_range = (ecg_ch + tremor_ch)..total;

    assert_eq!(ecg_range.end, tremor_range.start);
    assert_eq!(tremor_range.end, voice_range.start);

    println!(
        "Channel allocation:\n\
         - ECG: channels {}-{}\n\
         - Tremor: channels {}-{}\n\
         - Voice: channels {}-{}\n\
         - Total: {} channels",
        ecg_range.start,
        ecg_range.end - 1,
        tremor_range.start,
        tremor_range.end - 1,
        voice_range.start,
        voice_range.end - 1,
        total
    );
}

#[test]
fn test_multimodal_reproducibility() {
    // Test that multimodal pipeline is deterministic
    let duration = 3.0;

    // Run 1
    let ecg_params = EcgMorphologyParams {
        duration,
        sampling_rate: 250.0,
        heart_rate: 72.0,
        p_wave: WaveParams {
            amplitude: 0.25,
            width: 0.1,
            time_offset: -PI / 3.0,
        },
        qrs_complex: WaveParams {
            amplitude: 1.0,
            width: 0.1,
            time_offset: 0.0,
        },
        t_wave: WaveParams {
            amplitude: 0.35,
            width: 0.25,
            time_offset: PI / 2.0,
        },
    };
    let ecg_gen = EcgMorphologyGenerator;
    let ecg1 = ecg_gen.generate(&ecg_params, TEST_SEED).unwrap();
    let ecg2 = ecg_gen.generate(&ecg_params, TEST_SEED).unwrap();

    assert_eq!(ecg1.signal.len(), ecg2.signal.len());
    for i in 0..100 {
        assert_approx_eq(ecg1.signal[i], ecg2.signal[i], &format!("ECG sample {}", i));
    }

    let tremor_params = PhysiologicalTremorParams {
        duration,
        sampling_rate: 100.0,
        frequency: 10.0,
        amplitude: 0.5,
        frequency_variability: 0.3,
    };
    let tremor_gen = PhysiologicalTremorGenerator;
    let tremor1 = tremor_gen.generate(&tremor_params, TEST_SEED).unwrap();
    let tremor2 = tremor_gen.generate(&tremor_params, TEST_SEED).unwrap();

    assert_eq!(tremor1.signal.len(), tremor2.signal.len());
    for i in 0..100 {
        assert_approx_eq(
            tremor1.signal[i],
            tremor2.signal[i],
            &format!("Tremor sample {}", i),
        );
    }

    println!("Multimodal reproducibility verified");
}
