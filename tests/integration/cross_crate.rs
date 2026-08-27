//! Cross-Crate API Compatibility Tests
//!
//! Validates that all crates work together seamlessly:
//! 1. dpb-core types are compatible across all crates
//! 2. dpb-synth → dpb-encoders → dpb-snn pipeline works
//! 3. Trait implementations are consistent
//! 4. Data formats are compatible

use dpb_core::{
    Context, EventEncoder, PopulationTemplate, Signal, SignalBuffer, SpikeEvent, SpikeTrain,
};
use dpb_encoders::prelude::*;
use dpb_snn::{
    FeedforwardSNN, NeuronModel, NeuronParams, SNNConfig, SpikeRateDecoder, SpikeTensor,
};
// `forward` and `decode` are trait methods.
use dpb_snn::architectures::SNNArchitecture;
use dpb_snn::decoders::Decoder;
use dpb_synth::contact::ecg::{EcgMorphologyGenerator, EcgMorphologyParams, WaveParams};
use dpb_synth::traits::SyntheticGenerator;

use super::utils::*;
use std::f64::consts::PI;

#[test]
fn test_spike_event_compatibility() {
    // Verify SpikeEvent is compatible across crates
    let event = SpikeEvent::new(0.001, 5, 1, 1.0);

    // Should work with SpikeTrain from dpb-core
    let mut train = SpikeTrain::new(10);
    train.add_event(event);

    assert_eq!(train.len(), 1);
    assert_eq!(train.events[0].timestamp, 0.001);

    println!(
        "SpikeEvent compatibility: timestamp={:.3}, channel={}, polarity={}",
        event.timestamp, event.channel, event.polarity
    );
}

#[test]
fn test_signal_buffer_trait_impl() {
    // Verify SignalBuffer implements Signal trait
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let signal = SignalBuffer::single_channel(data.clone(), 1000.0);

    // Test Signal trait methods
    assert_eq!(signal.sample_rate(), 1000.0);
    assert_eq!(signal.channels(), 1);
    assert_eq!(signal.samples().len(), 5);

    println!(
        "SignalBuffer trait: {} channels @ {:.0} Hz, {} samples",
        signal.channels(),
        signal.sample_rate(),
        signal.samples().len()
    );
}

#[test]
fn test_context_usage_across_crates() {
    // Test Context from dpb-core works with encoders
    let context = Context {
        age: Some(30.0),
        sex: Some("Male".to_string()),
        height_cm: Some(175.0),
        weight_kg: Some(75.0),
        ..Context::default()
    };

    // Use with population template from dpb-encoders
    let hr_template = HeartRateTemplate;
    let expected_hr = hr_template.expected_value(&context);
    let variance = hr_template.variance(&context);

    assert!(expected_hr > 0.0, "Expected HR should be positive");
    assert!(variance > 0.0, "Variance should be positive");

    println!(
        "Context compatibility: age={:.0}, expected HR={:.1} ± {:.1} bpm",
        context.age.unwrap(),
        expected_hr,
        variance.sqrt()
    );
}

#[test]
fn test_synth_to_encoder_pipeline() {
    // Test dpb-synth → dpb-encoders integration
    let params = EcgMorphologyParams {
        duration: 5.0,
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

    // Generate with dpb-synth
    let generator = EcgMorphologyGenerator;
    let generated = generator
        .generate(&params, TEST_SEED)
        .expect("Failed to generate");

    // Convert to dpb-core SignalBuffer
    let signal_data: Vec<f32> = generated.signal.iter().map(|&x| x as f32).collect();
    let signal = SignalBuffer::single_channel(signal_data, params.sampling_rate);

    // Encode with dpb-encoders
    let encoder = LevelCrossingEncoder::new("cross_crate_test");
    let config = LevelCrossingConfig {
        threshold: 0.3,
        relative: false,
        refractory_period: 0.2,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };

    let spike_train = encoder.encode(&signal, &config).expect("Failed to encode");

    assert!(!spike_train.is_empty(), "Should produce spikes");

    println!(
        "Synth→Encoder pipeline: {} samples → {} spikes",
        generated.signal.len(),
        spike_train.len()
    );
}

#[test]
fn test_encoder_to_snn_pipeline() {
    // Test dpb-encoders → dpb-snn integration
    let data = [0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
    let signal = SignalBuffer::single_channel(data.iter().map(|&x| x as f32).collect(), 100.0);

    // Encode
    let encoder = LevelCrossingEncoder::new("encoder_to_snn");
    let config = LevelCrossingConfig {
        threshold: 0.3,
        relative: false,
        refractory_period: 0.0,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };

    let spike_train = encoder.encode(&signal, &config).expect("Failed to encode");

    // Convert to SpikeTensor (dpb-snn)
    let num_timesteps = 100;
    let num_channels = 16;
    let mut spike_tensor = SpikeTensor::zeros(1, num_timesteps, num_channels, false);

    for event in &spike_train {
        let timestep = ((event.timestamp * 100.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = (event.channel as usize) % num_channels;
        spike_tensor
            .set_spike(0, timestep, channel, event.magnitude)
            .ok();
    }

    // Run through SNN
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    let mut snn = FeedforwardSNN::new(vec![num_channels, 8, 4], snn_config, true).expect("SNN");
    let output = snn.forward(&spike_tensor).expect("Failed to run SNN");

    assert!(output.shape().0 == 1, "Batch size should match");

    println!(
        "Encoder→SNN pipeline: {} spikes → SNN output shape {:?}",
        spike_train.len(),
        output.shape()
    );
}

#[test]
fn test_full_pipeline_integration() {
    // Test dpb-synth → dpb-core → dpb-encoders → dpb-snn → decoders
    let duration = 5.0;

    // 1. Generate (dpb-synth)
    let params = EcgMorphologyParams {
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

    let generator = EcgMorphologyGenerator;
    let generated = generator.generate(&params, TEST_SEED).unwrap();

    // 2. Convert to SignalBuffer (dpb-core)
    let signal_data: Vec<f32> = generated.signal.iter().map(|&x| x as f32).collect();
    let signal = SignalBuffer::single_channel(signal_data, params.sampling_rate);

    // 3. Encode (dpb-encoders)
    let encoder = LevelCrossingEncoder::new("full_pipeline");
    let config = LevelCrossingConfig {
        threshold: 0.3,
        relative: false,
        refractory_period: 0.2,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };
    let spike_train = encoder.encode(&signal, &config).unwrap();

    // 4. Convert to SpikeTensor (dpb-snn)
    let num_timesteps = (duration * 1000.0) as usize;
    let num_channels = 64;
    let mut spike_tensor = SpikeTensor::zeros(1, num_timesteps, num_channels, false);

    for event in &spike_train {
        let timestep = ((event.timestamp * 1000.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = (event.channel as usize) % num_channels;
        spike_tensor
            .set_spike(0, timestep, channel, event.magnitude)
            .ok();
    }

    // 5. SNN inference (dpb-snn)
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };
    let mut snn = FeedforwardSNN::new(vec![num_channels, 32, 4], snn_config, true).expect("SNN");
    let output = snn.forward(&spike_tensor).unwrap();

    // 6. Decode (dpb-snn)
    let decoder = SpikeRateDecoder::new(4, None, false);
    let predictions = decoder.decode(&output).unwrap();

    println!(
        "Full pipeline integration:\n\
         - Generated: {} samples\n\
         - Encoded: {} spikes\n\
         - SNN output: {:?}\n\
         - Predictions: {:?}",
        generated.signal.len(),
        spike_train.len(),
        output.shape(),
        predictions.shape()
    );
}

#[test]
fn test_event_encoder_trait() {
    // Verify EventEncoder trait works across implementations
    let data = [0.0, 1.0, 0.0, -1.0];
    let signal = SignalBuffer::single_channel(data.iter().map(|&x| x as f32).collect(), 100.0);

    // Test with LevelCrossingEncoder
    let encoder1 = LevelCrossingEncoder::new("trait_test_1");
    let config1 = LevelCrossingConfig {
        threshold: 0.5,
        relative: false,
        refractory_period: 0.0,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };
    let spikes1 = encoder1.encode(&signal, &config1).unwrap();

    // Test with DerivativeEncoder
    let encoder2 = DerivativeEncoder::new("trait_test_2");
    let config2 = DerivativeConfig {
        threshold: 0.1,
        ..DerivativeConfig::default()
    };
    let spikes2 = encoder2.encode(&signal, &config2).unwrap();

    // Both should produce SpikeTrain
    assert!(!spikes1.is_empty() || !spikes2.is_empty());

    println!(
        "EventEncoder trait: LevelCrossing={} spikes, Derivative={} spikes",
        spikes1.len(),
        spikes2.len()
    );
}

#[test]
fn test_data_format_consistency() {
    // Verify data formats are consistent across crates

    // Create spike event
    let event = SpikeEvent::new(1.0, 0, 1, 0.5);

    // Add to spike train
    let mut train = SpikeTrain::new(1);
    train.add_event(event);

    // Verify properties
    assert_eq!(train.len(), 1);
    assert_eq!(train.events[0].channel, 0);
    assert_eq!(train.events[0].polarity, 1);
    assert_eq!(train.events[0].magnitude, 0.5);

    println!("Data format consistency verified: SpikeEvent → SpikeTrain");
}

#[test]
fn test_neuron_model_compatibility() {
    // Test that neuron models work consistently
    let models = vec![
        NeuronModel::LIF,
        NeuronModel::AdaptiveLIF,
        NeuronModel::Izhikevich,
    ];

    let num_channels = 16;
    let num_timesteps = 100;

    for model in models {
        let snn_config = SNNConfig {
            dt: 1.0,
            num_steps: num_timesteps,
            neuron_model: model,
            neuron_params: NeuronParams::default(),
        };

        let mut snn = FeedforwardSNN::new(vec![num_channels, 8, 4], snn_config, true).expect("SNN");
        let spike_tensor = SpikeTensor::zeros(1, num_timesteps, num_channels, false);

        let output = snn
            .forward(&spike_tensor)
            .unwrap_or_else(|_| panic!("Failed with neuron model {:?}", model));

        println!(
            "Neuron model {:?}: output shape {:?}",
            model,
            output.shape()
        );
    }
}

#[test]
fn test_template_registry_integration() {
    // Test that template registry works with contexts
    let registry = templates::TemplateRegistry::new();

    let context = Context {
        age: Some(30.0),
        sex: Some("Male".to_string()),
        height_cm: Some(175.0),
        weight_kg: Some(75.0),
        ..Context::default()
    };

    let results = registry.evaluate_all(&context);

    assert!(!results.is_empty(), "Should have template results");

    println!("Template registry: {} templates evaluated", results.len());
}

#[test]
fn test_serialization_compatibility() {
    // Test that types can be serialized/deserialized across crates
    let event = SpikeEvent::new(1.5, 3, 1, 0.8);

    // SpikeEvent implements Serialize/Deserialize
    let serialized = serde_json::to_string(&event).expect("Failed to serialize");

    let deserialized: SpikeEvent =
        serde_json::from_str(&serialized).expect("Failed to deserialize");

    assert_eq!(event.timestamp, deserialized.timestamp);
    assert_eq!(event.channel, deserialized.channel);

    println!("Serialization compatibility verified");
}
