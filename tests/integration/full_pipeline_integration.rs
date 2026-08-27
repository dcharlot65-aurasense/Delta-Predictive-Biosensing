//! Full system integration tests combining multiple crates
//!
//! These tests exercise the complete DPB framework pipeline:
//! Synthesis → Processing → Encoding → SNN → Decoding → Analysis

use dpb_core::SignalBuffer;
use dpb_encoders::prelude::*;
use dpb_snn::{
    FeedforwardSNN, SpikeTensor, SpikeRateDecoder, Decoder,
    NeuronModel, NeuronParams, SNNConfig,
};
use dpb_snn::architectures::SNNArchitecture;
use dpb_synth::contact::ecg::{EcgMorphologyGenerator, EcgMorphologyParams, WaveParams};
use dpb_synth::contact::respiratory::{RespiratoryWaveformGenerator, RespiratoryParams};
use dpb_synth::traits::SyntheticGenerator;
use dpb_core::signal::*;

const TEST_SEED: u64 = 42;

#[test]
fn test_synthetic_to_analysis_pipeline() {
    println!("Full Pipeline Test: Synthesis → Analysis");

    // Step 1: Generate synthetic ECG using dpb-synth
    let duration = 30.0; // seconds
    let sampling_rate = 250.0; // Hz
    let heart_rate = 75.0; // bpm

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
    let generated = generator
        .generate(&params, TEST_SEED)
        .expect("Failed to generate ECG");

    println!("  ✓ Generated ECG: {:.1}s @ {:.0} Hz", duration, sampling_rate);
    println!("    Ground truth R-peaks: {}", generated.ground_truth.events.len());

    // Step 2: Process with dpb-core signal processing
    let signal_data: Vec<f32> = generated.signal.iter().map(|&x| x as f32).collect();
    let signal = SignalBuffer::single_channel(signal_data.clone(), sampling_rate);

    // Apply bandpass filter
    let lowpass = IirFilter::butterworth_lowpass(4, 40.0, sampling_rate)
        .expect("Failed to create lowpass");
    let highpass = IirFilter::butterworth_highpass(4, 0.5, sampling_rate)
        .expect("Failed to create highpass");

    // `filter` takes an ArrayView and returns an Array directly.
    let as_f64: Vec<f64> = signal_data.iter().map(|&x| x as f64).collect();
    let mut lowpass = lowpass;
    let mut highpass = highpass;
    let filtered_low = lowpass.filter(ndarray::ArrayView1::from(&as_f64));
    let filtered_arr = highpass.filter(filtered_low.view());
    let filtered: Vec<f64> = filtered_arr.to_vec();

    println!("  ✓ Applied bandpass filtering (0.5-40 Hz)");

    // Detect R-peaks
    let detector = PanTompkinsDetector::new(sampling_rate).expect("detector");
    let detected_peaks = detector.detect_r_peaks(&filtered).expect("Failed to detect R-peaks");

    println!("  ✓ Detected {} R-peaks", detected_peaks.len());

    // Compute HRV
    if detected_peaks.len() >= 2 {
        let rr_intervals: Vec<f64> = detected_peaks
            .windows(2)
            .map(|w| (w[1].index - w[0].index) as f64 / sampling_rate * 1000.0)
            .collect();

        let hrv_analyzer = HrvAnalyzer::new();
        let hrv = hrv_analyzer
            .compute_time_domain(&rr_intervals)
            .expect("Failed to compute HRV");

        println!("  ✓ HRV Analysis:");
        println!("    Mean RR: {:.1} ms", hrv.mean_rr_ms);
        println!("    SDNN: {:.1} ms", hrv.sdnn_ms);
        println!("    RMSSD: {:.1} ms", hrv.rmssd_ms);
    }

    // Step 3: Encode with dpb-encoders
    let encoder = LevelCrossingEncoder::new("ecg_encoder");
    let encoder_config = LevelCrossingConfig {
        threshold: 0.3,
        relative: false,
        refractory_period: 0.2,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };

    let spike_train = encoder
        .encode(&signal, &encoder_config)
        .expect("Failed to encode signal");

    println!("  ✓ Encoded to {} spike events", spike_train.len());

    // Step 4: Run through dpb-snn
    let num_timesteps = (duration * 1000.0) as usize;
    let num_channels = 64;
    let num_output = 1;

    // Create dense spike tensor
    let mut spike_tensor_dense = ndarray::Array3::zeros((1, num_timesteps, num_channels));

    for event in &spike_train {
        let timestep = (event.timestamp * 1000.0).min((num_timesteps - 1) as f64) as usize;
        let channel = (event.channel as usize) % num_channels;
        spike_tensor_dense[[0, timestep, channel]] = 1.0;
    }

    let spike_tensor = SpikeTensor::from_dense(spike_tensor_dense, false);

    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    let mut snn = FeedforwardSNN::new(vec![num_channels, 32, 16, num_output], snn_config, true).expect("SNN");
    let output_spikes = snn.forward(&spike_tensor).expect("Failed to run SNN");

    println!("  ✓ SNN inference completed");

    // Step 5: Decode output
    let decoder = SpikeRateDecoder::new(num_output, None, false);
    let decoded = decoder.decode(&output_spikes).expect("Failed to decode");

    println!("  ✓ Decoded output shape: {:?}", decoded.shape());

    // Verify outputs
    assert_eq!(decoded.shape()[1], num_output);

    println!("  ✓ Full pipeline completed successfully!");
}

#[test]
fn test_multimodal_synthesis_and_fusion() {
    println!("Multimodal Pipeline Test: ECG + Respiratory");

    let duration = 60.0;
    let sampling_rate = 250.0;

    // Step 1: Generate ECG
    let ecg_params = EcgMorphologyParams {
        duration,
        sampling_rate,
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

    let ecg_generator = EcgMorphologyGenerator;
    let ecg_data = ecg_generator
        .generate(&ecg_params, TEST_SEED)
        .expect("Failed to generate ECG");

    println!("  ✓ Generated ECG signal");

    // Step 2: Generate Respiratory signal
    let resp_params = RespiratoryParams {
        duration,
        sampling_rate,
        respiratory_rate: 15.0, // breaths per minute
        amplitude: 0.5,
        inspiration_ratio: 0.4,
    };

    let resp_generator = RespiratoryWaveformGenerator;
    let resp_data = resp_generator
        .generate(&resp_params, TEST_SEED + 1)
        .expect("Failed to generate respiratory signal");

    println!("  ✓ Generated respiratory signal");

    // Step 3: Process both signals
    let ecg_signal_f32: Vec<f32> = ecg_data.signal.iter().map(|&x| x as f32).collect();
    let resp_signal_f32: Vec<f32> = resp_data.signal.iter().map(|&x| x as f32).collect();

    let ecg_buffer = SignalBuffer::single_channel(ecg_signal_f32, sampling_rate);
    let resp_buffer = SignalBuffer::single_channel(resp_signal_f32, sampling_rate);

    println!("  ✓ Created signal buffers");

    // Step 4: Analyze ECG
    let ecg_detector = PanTompkinsDetector::new(sampling_rate).expect("detector");
    let ecg_signal_f64: Vec<f64> = ecg_data.signal.to_vec();
    let r_peaks = ecg_detector
        .detect_r_peaks(&ecg_signal_f64)
        .expect("Failed to detect R-peaks");

    println!("  ✓ ECG: Detected {} R-peaks", r_peaks.len());

    // Step 5: Analyze Respiratory
    let resp_analyzer = RespiratoryAnalyzer::new(sampling_rate);
    let resp_rate = resp_analyzer
        .calculate_respiratory_rate(resp_data.signal.view())
        .expect("Failed to analyze respiratory");

    println!("  ✓ Respiratory: Rate = {:.1} br/min", resp_rate);

    // Step 6: Encode both modalities
    let ecg_encoder = LevelCrossingEncoder::new("ecg");
    let resp_encoder = LevelCrossingEncoder::new("resp");

    let ecg_config = LevelCrossingConfig {
        threshold: 0.3,
        relative: false,
        refractory_period: 0.2,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };

    let resp_config = LevelCrossingConfig {
        threshold: 0.2,
        relative: false,
        refractory_period: 0.5,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };

    let ecg_spikes = ecg_encoder
        .encode(&ecg_buffer, &ecg_config)
        .expect("Failed to encode ECG");
    let resp_spikes = resp_encoder
        .encode(&resp_buffer, &resp_config)
        .expect("Failed to encode respiratory");

    println!("  ✓ Encoded ECG: {} spikes", ecg_spikes.len());
    println!("  ✓ Encoded Respiratory: {} spikes", resp_spikes.len());

    // Step 7: Fuse modalities (simple concatenation)
    let num_timesteps = (duration * 1000.0) as usize;
    let ecg_channels = 32;
    let resp_channels = 16;
    let total_channels = ecg_channels + resp_channels;

    // Create dense fused tensor
    let mut fused_tensor_dense = ndarray::Array3::zeros((1, num_timesteps, total_channels));

    // Add ECG spikes to first channels
    for event in &ecg_spikes {
        let t = (event.timestamp * 1000.0).min((num_timesteps - 1) as f64) as usize;
        let c = (event.channel as usize) % ecg_channels;
        fused_tensor_dense[[0, t, c]] = 1.0;
    }

    // Add respiratory spikes to remaining channels
    for event in &resp_spikes {
        let t = (event.timestamp * 1000.0).min((num_timesteps - 1) as f64) as usize;
        let c = ecg_channels + ((event.channel as usize) % resp_channels);
        fused_tensor_dense[[0, t, c]] = 1.0;
    }

    let fused_tensor = SpikeTensor::from_dense(fused_tensor_dense, false);

    println!("  ✓ Fused modalities: {} total channels", total_channels);

    // Step 8: Run through fusion SNN
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    let mut fusion_snn = FeedforwardSNN::new(vec![total_channels, 32, 8], snn_config, true).expect("SNN");
    let output = fusion_snn
        .forward(&fused_tensor)
        .expect("Failed to run fusion SNN");

    println!("  ✓ Fusion SNN output shape: {:?}", output.shape());

    // Step 9: Decode fused output
    let decoder = SpikeRateDecoder::new(8, None, false);
    let decoded = decoder.decode(&output).expect("Failed to decode");

    println!("  ✓ Decoded fusion output: {:?}", decoded.shape());

    println!("  ✓ Multimodal pipeline completed successfully!");
}

#[test]
fn test_normative_comparison_pipeline() {
    println!("Normative Comparison Pipeline Test");

    // Step 1: Generate signals with known characteristics
    let duration = 20.0;
    let sampling_rate = 250.0;

    // Generate multiple ECG samples with different heart rates
    let heart_rates = [60.0, 70.0, 80.0, 90.0, 100.0];
    let mut all_hrv_metrics = Vec::new();

    for (i, &hr) in heart_rates.iter().enumerate() {
        let params = EcgMorphologyParams {
            duration,
            sampling_rate,
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
        let generated = generator
            .generate(&params, TEST_SEED + i as u64)
            .expect("Failed to generate ECG");

        // Detect R-peaks
        let detector = PanTompkinsDetector::new(sampling_rate).expect("detector");
        let signal_f64: Vec<f64> = generated.signal.to_vec();
        let peaks = detector
            .detect_r_peaks(&signal_f64)
            .expect("Failed to detect R-peaks");

        if peaks.len() >= 2 {
            let rr_intervals: Vec<f64> = peaks
                .windows(2)
                .map(|w| (w[1].index - w[0].index) as f64 / sampling_rate * 1000.0)
                .collect();

            let analyzer = HrvAnalyzer::new();
            let hrv = analyzer
                .compute_time_domain(&rr_intervals)
                .expect("Failed to compute HRV");

            all_hrv_metrics.push((hr, hrv.mean_rr_ms, hrv.sdnn_ms));

            println!(
                "  Sample {}: HR={:.0} bpm, Mean RR={:.1} ms, SDNN={:.1} ms",
                i + 1,
                hr,
                hrv.mean_rr_ms,
                hrv.sdnn_ms
            );
        }
    }

    // Step 2: Compute normative statistics
    let mean_rr_values: Vec<f64> = all_hrv_metrics.iter().map(|(_, rr, _)| *rr).collect();
    let sdnn_values: Vec<f64> = all_hrv_metrics.iter().map(|(_, _, sdnn)| *sdnn).collect();

    let mean_rr_avg: f64 = mean_rr_values.iter().sum::<f64>() / mean_rr_values.len() as f64;
    let sdnn_avg: f64 = sdnn_values.iter().sum::<f64>() / sdnn_values.len() as f64;

    println!("\n  Normative Statistics:");
    println!("    Mean RR Average: {:.1} ms", mean_rr_avg);
    println!("    SDNN Average: {:.1} ms", sdnn_avg);

    // Compute standard deviations
    let mean_rr_std: f64 = (mean_rr_values
        .iter()
        .map(|x| (x - mean_rr_avg).powi(2))
        .sum::<f64>()
        / mean_rr_values.len() as f64)
        .sqrt();

    let sdnn_std: f64 = (sdnn_values
        .iter()
        .map(|x| (x - sdnn_avg).powi(2))
        .sum::<f64>()
        / sdnn_values.len() as f64)
        .sqrt();

    println!("    Mean RR Std: {:.1} ms", mean_rr_std);
    println!("    SDNN Std: {:.1} ms", sdnn_std);

    // Step 3: Compute z-scores for each sample
    println!("\n  Z-Scores:");
    for (i, (hr, mean_rr, sdnn)) in all_hrv_metrics.iter().enumerate() {
        let mean_rr_zscore = (mean_rr - mean_rr_avg) / mean_rr_std;
        let sdnn_zscore = (sdnn - sdnn_avg) / sdnn_std;

        println!(
            "    Sample {} (HR={:.0}): Mean RR z={:.2}, SDNN z={:.2}",
            i + 1,
            hr,
            mean_rr_zscore,
            sdnn_zscore
        );

        // Verify z-scores are reasonable
        assert!(
            mean_rr_zscore.abs() < 3.0,
            "Z-score should be within ±3"
        );
    }

    println!("\n  ✓ Normative comparison pipeline completed!");
}

#[test]
fn test_real_time_streaming_simulation() {
    println!("Real-time Streaming Simulation Test");

    let sampling_rate = 250.0;
    let chunk_duration = 1.0; // 1 second chunks
    let chunk_size = (sampling_rate * chunk_duration) as usize;
    let num_chunks = 10;

    println!("  Configuration:");
    println!("    Sample rate: {:.0} Hz", sampling_rate);
    println!("    Chunk size: {} samples ({:.1}s)", chunk_size, chunk_duration);
    println!("    Total chunks: {}", num_chunks);

    // Initialize processing pipeline components
    let ecg_params = EcgMorphologyParams {
        duration: chunk_duration,
        sampling_rate,
        heart_rate: 75.0,
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
    let encoder = LevelCrossingEncoder::new("streaming");
    let encoder_config = LevelCrossingConfig {
        threshold: 0.3,
        relative: false,
        refractory_period: 0.2,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };

    let mut total_latency = 0.0;
    let mut total_spikes = 0;

    // Simulate streaming chunks
    for chunk_idx in 0..num_chunks {
        let start_time = std::time::Instant::now();

        // Step 1: Generate chunk
        let generated = generator
            .generate(&ecg_params, TEST_SEED + chunk_idx as u64)
            .expect("Failed to generate chunk");

        // Step 2: Process chunk
        let signal_data: Vec<f32> = generated.signal.iter().map(|&x| x as f32).collect();
        let signal = SignalBuffer::single_channel(signal_data, sampling_rate);

        // Step 3: Encode chunk
        let spikes = encoder
            .encode(&signal, &encoder_config)
            .expect("Failed to encode chunk");

        total_spikes += spikes.len();

        let latency = start_time.elapsed().as_secs_f64() * 1000.0;
        total_latency += latency;

        if chunk_idx % 3 == 0 {
            println!(
                "  Chunk {}: processed in {:.2}ms, {} spikes",
                chunk_idx,
                latency,
                spikes.len()
            );
        }

        // Verify latency constraint (should be much less than chunk duration)
        let max_latency_ms = chunk_duration * 1000.0 * 0.5; // 50% of chunk duration
        assert!(
            latency < max_latency_ms,
            "Chunk {} latency {:.2}ms exceeds limit {:.2}ms",
            chunk_idx,
            latency,
            max_latency_ms
        );
    }

    let mean_latency = total_latency / num_chunks as f64;
    println!("\n  Summary:");
    println!("    Mean latency: {:.2}ms", mean_latency);
    println!("    Total spikes: {}", total_spikes);
    println!("    Chunks/second: {:.1}", 1000.0 / mean_latency);

    println!("  ✓ Streaming simulation completed successfully!");
}

#[test]
fn test_end_to_end_feature_extraction() {
    println!("End-to-End Feature Extraction Test");

    let duration = 30.0;
    let sampling_rate = 250.0;

    // Generate ECG
    let params = EcgMorphologyParams {
        duration,
        sampling_rate,
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
    let generated = generator
        .generate(&params, TEST_SEED)
        .expect("Failed to generate ECG");

    println!("  ✓ Generated ECG signal");

    // Extract features using different approaches
    let mut features = std::collections::HashMap::new();

    // 1. Time-domain features (R-peaks, HRV)
    let detector = PanTompkinsDetector::new(sampling_rate).expect("detector");
    let signal_f64: Vec<f64> = generated.signal.to_vec();
    let peaks = detector
        .detect_r_peaks(&signal_f64)
        .expect("Failed to detect R-peaks");

    features.insert("num_peaks", peaks.len() as f64);

    if peaks.len() >= 2 {
        let rr_intervals: Vec<f64> = peaks
            .windows(2)
            .map(|w| (w[1].index - w[0].index) as f64 / sampling_rate * 1000.0)
            .collect();

        let analyzer = HrvAnalyzer::new();
        let hrv = analyzer
            .compute_time_domain(&rr_intervals)
            .expect("Failed to compute HRV");

        features.insert("mean_rr", hrv.mean_rr_ms);
        features.insert("sdnn", hrv.sdnn_ms);
        features.insert("rmssd", hrv.rmssd_ms);
    }

    // 2. Frequency-domain features
    // `PpgAnalyzer` exposes stagewise methods rather than one `analyze`.
    let ppg_analyzer = PpgAnalyzer::new(sampling_rate);
    let ppg_peaks = ppg_analyzer
        .detect_peaks(generated.signal.view())
        .expect("Failed to detect PPG peaks");
    let mean_amplitude = if ppg_peaks.is_empty() {
        0.0
    } else {
        ppg_peaks.iter().map(|&i| generated.signal[i]).sum::<f64>() / ppg_peaks.len() as f64
    };

    features.insert("mean_amplitude", mean_amplitude);

    // 3. Spike-based features
    let signal_f32: Vec<f32> = generated.signal.iter().map(|&x| x as f32).collect();
    let signal = SignalBuffer::single_channel(signal_f32, sampling_rate);

    let encoder = LevelCrossingEncoder::new("feature_ext");
    let encoder_config = LevelCrossingConfig {
        threshold: 0.3,
        relative: false,
        refractory_period: 0.2,
        // Detection semantics: one event per crossing of the level. The
        // default `Delta` mode instead emits one event per threshold of
        // travel, which is the reconstructable sampling behaviour.
        mode: LevelCrossingMode::FixedLevel,
    };

    let spikes = encoder
        .encode(&signal, &encoder_config)
        .expect("Failed to encode");

    features.insert("spike_count", spikes.len() as f64);
    features.insert("spike_rate", spikes.len() as f64 / duration);

    // Print extracted features
    println!("\n  Extracted Features:");
    let mut feature_names: Vec<_> = features.keys().collect();
    feature_names.sort();

    for name in feature_names {
        println!("    {}: {:.3}", name, features[name]);
    }

    // Verify all features are valid
    for (name, &value) in &features {
        assert!(
            value.is_finite(),
            "Feature {} has invalid value: {}",
            name,
            value
        );
        assert!(value >= 0.0, "Feature {} should be non-negative: {}", name, value);
    }

    println!("\n  ✓ Feature extraction completed with {} features!", features.len());
}
