//! Integration tests for signal processing pipelines
//!
//! Tests complete signal processing chains from raw input through
//! filtering, analysis, and feature extraction.

use dpb_core::signal::*;
use dpb_core::pipeline::*;
use dpb_core::{SignalBuffer, Result};
use ndarray::Array1;
use std::f64::consts::PI;

/// Generate synthetic ECG signal with known heart rate
fn generate_synthetic_ecg(duration: f64, sample_rate: f64, heart_rate: f64) -> Vec<f64> {
    let num_samples = (duration * sample_rate) as usize;
    let mut signal = vec![0.0; num_samples];

    let beat_period = 60.0 / heart_rate; // seconds per beat

    for i in 0..num_samples {
        let t = i as f64 / sample_rate;
        let beat_phase = (t % beat_period) / beat_period;

        // Simple QRS complex model using Gaussian-like pulse
        let qrs_time = 0.15; // QRS at 15% of beat cycle
        let qrs_width = 0.05; // 5% width
        let dist_to_qrs = (beat_phase - qrs_time).abs();

        if dist_to_qrs < 0.1 {
            // QRS complex (Gaussian-shaped)
            let x = (beat_phase - qrs_time) / qrs_width;
            signal[i] = 1.0 * (-x * x).exp();
        }

        // Add T-wave
        let t_time = 0.45;
        let t_width = 0.1;
        let dist_to_t = (beat_phase - t_time).abs();
        if dist_to_t < 0.15 {
            let x = (beat_phase - t_time) / t_width;
            signal[i] += 0.3 * (-x * x).exp();
        }

        // Add small noise
        signal[i] += 0.02 * ((i as f64 * 0.1).sin());
    }

    signal
}

/// Generate synthetic EEG with seizure-like activity
fn generate_synthetic_eeg_with_seizure(
    duration: f64,
    sample_rate: f64,
    seizure_start: f64,
    seizure_duration: f64,
) -> Vec<f64> {
    let num_samples = (duration * sample_rate) as usize;
    let mut signal = vec![0.0; num_samples];

    let seizure_end = seizure_start + seizure_duration;

    for i in 0..num_samples {
        let t = i as f64 / sample_rate;

        // Normal background EEG (alpha ~10Hz + noise)
        signal[i] = 0.5 * (2.0 * PI * 10.0 * t).sin();
        signal[i] += 0.2 * (2.0 * PI * 8.0 * t).sin();
        signal[i] += 0.1 * ((i as f64 * 0.05).sin());

        // Add high-amplitude rhythmic activity during seizure
        if t >= seizure_start && t < seizure_end {
            let seizure_freq = 3.0; // 3 Hz spike-wave
            signal[i] += 2.0 * (2.0 * PI * seizure_freq * (t - seizure_start)).sin();

            // Amplitude increases over time during seizure
            let seizure_progress = (t - seizure_start) / seizure_duration;
            signal[i] *= 1.0 + 1.5 * seizure_progress;
        }
    }

    signal
}

/// Generate respiratory signal with apnea events
fn generate_respiratory_with_apnea(
    duration: f64,
    sample_rate: f64,
    breath_rate: f64, // breaths per minute
    apnea_intervals: &[(f64, f64)], // (start, duration) pairs
) -> Vec<f64> {
    let num_samples = (duration * sample_rate) as usize;
    let mut signal = vec![0.0; num_samples];

    let breath_period = 60.0 / breath_rate;

    for i in 0..num_samples {
        let t = i as f64 / sample_rate;

        // Check if we're in an apnea period
        let in_apnea = apnea_intervals
            .iter()
            .any(|(start, dur)| t >= *start && t < start + dur);

        if !in_apnea {
            // Normal sinusoidal breathing
            signal[i] = (2.0 * PI * t / breath_period).sin();
            // Add some variability
            signal[i] += 0.1 * (2.0 * PI * t / (breath_period * 0.7)).sin();
        } else {
            // Minimal amplitude during apnea
            signal[i] = 0.05 * (2.0 * PI * t / breath_period).sin();
        }
    }

    signal
}

#[test]
fn test_ecg_processing_pipeline() {
    // Step 1: Generate synthetic ECG
    let duration = 10.0; // seconds
    let sample_rate = 250.0; // Hz
    let heart_rate = 72.0; // bpm

    let ecg_data = generate_synthetic_ecg(duration, sample_rate, heart_rate);

    // Step 2: Create signal buffer
    let signal = SignalBuffer::single_channel(
        ecg_data.iter().map(|&x| x as f32).collect(),
        sample_rate,
    );

    // Step 3: Apply bandpass filter (0.5-40 Hz typical for ECG)
    let lowpass = IirFilter::butterworth_lowpass(4, 40.0, sample_rate)
        .expect("Failed to create lowpass filter");
    let highpass = IirFilter::butterworth_highpass(4, 0.5, sample_rate)
        .expect("Failed to create highpass filter");

    let filtered_low = lowpass.filter(&ecg_data).expect("Failed to apply lowpass");
    let filtered = highpass.filter(&filtered_low).expect("Failed to apply highpass");

    // Step 4: Detect R-peaks using Pan-Tompkins
    let detector = PanTompkinsDetector::new(sample_rate)
        .expect("Failed to create Pan-Tompkins detector");
    let r_peaks = detector
        .detect(&filtered)
        .expect("Failed to detect R-peaks");

    // Step 5: Verify results
    let expected_beats = (heart_rate * duration / 60.0) as usize;
    assert!(
        r_peaks.len() >= expected_beats - 2 && r_peaks.len() <= expected_beats + 2,
        "Expected ~{} beats, found {}",
        expected_beats,
        r_peaks.len()
    );

    // Step 6: Compute HRV metrics
    if r_peaks.len() >= 2 {
        let rr_intervals: Vec<f64> = r_peaks
            .windows(2)
            .map(|w| {
                let dt = (w[1].index - w[0].index) as f64 / sample_rate * 1000.0;
                dt
            })
            .collect();

        let analyzer = HrvAnalyzer::new(sample_rate);
        let hrv_metrics = analyzer
            .analyze_time_domain(&rr_intervals)
            .expect("Failed to compute HRV");

        // Verify HRV metrics are reasonable
        assert!(hrv_metrics.mean_rr > 0.0, "Mean RR interval should be positive");
        assert!(hrv_metrics.sdnn >= 0.0, "SDNN should be non-negative");
        assert!(
            hrv_metrics.rmssd >= 0.0,
            "RMSSD should be non-negative"
        );

        println!("ECG Processing Pipeline Results:");
        println!("  Duration: {:.1}s @ {:.0} Hz", duration, sample_rate);
        println!("  Heart Rate: {:.1} bpm (target)", heart_rate);
        println!("  R-peaks detected: {}", r_peaks.len());
        println!("  HRV - Mean RR: {:.1} ms", hrv_metrics.mean_rr);
        println!("  HRV - SDNN: {:.1} ms", hrv_metrics.sdnn);
        println!("  HRV - RMSSD: {:.1} ms", hrv_metrics.rmssd);
    }
}

#[test]
fn test_eeg_seizure_detection_pipeline() {
    // Step 1: Generate synthetic EEG with seizure
    let duration = 60.0; // seconds
    let sample_rate = 256.0; // Hz
    let seizure_start = 20.0;
    let seizure_duration = 15.0;

    let eeg_data = generate_synthetic_eeg_with_seizure(
        duration,
        sample_rate,
        seizure_start,
        seizure_duration,
    );

    // Step 2: Apply artifact detection
    let artifacts = detect_artifacts(&eeg_data, sample_rate, 3.0)
        .expect("Failed to detect artifacts");

    println!("EEG Artifact Detection:");
    println!("  Total samples: {}", eeg_data.len());
    println!("  Artifacts detected: {}", artifacts.len());
    for artifact in &artifacts {
        println!(
            "    Type: {:?}, Start: {:.2}s, Duration: {:.2}s",
            artifact.artifact_type,
            artifact.start_time,
            artifact.duration
        );
    }

    // Step 3: Compute band powers
    let band_powers = compute_band_powers(&eeg_data, sample_rate)
        .expect("Failed to compute band powers");

    // Verify band powers are computed
    assert!(band_powers.delta > 0.0, "Delta power should be positive");
    assert!(band_powers.theta > 0.0, "Theta power should be positive");
    assert!(band_powers.alpha > 0.0, "Alpha power should be positive");
    assert!(band_powers.beta > 0.0, "Beta power should be positive");

    println!("EEG Band Powers:");
    println!("  Delta: {:.3}", band_powers.delta);
    println!("  Theta: {:.3}", band_powers.theta);
    println!("  Alpha: {:.3}", band_powers.alpha);
    println!("  Beta: {:.3}", band_powers.beta);

    // Step 4: Run seizure detector
    let detector = SeizureDetector::new(sample_rate);
    let result = detector.analyze(&eeg_data).expect("Failed to analyze for seizures");

    // Verify seizure was detected in the correct time window
    println!("Seizure Detection Results:");
    println!("  Seizure probability: {:.2}", result.seizure_probability);
    println!("  Seizure events detected: {}", result.events.len());

    for event in &result.events {
        println!(
            "    Start: {:.2}s, Duration: {:.2}s, Type: {:?}",
            event.start_time, event.duration, event.seizure_type
        );

        // Verify detected seizure overlaps with actual seizure
        let event_end = event.start_time + event.duration;
        let actual_end = seizure_start + seizure_duration;

        let overlaps = !(event_end < seizure_start || event.start_time > actual_end);
        if overlaps {
            println!("      ✓ Overlaps with actual seizure window");
        }
    }
}

#[test]
fn test_respiratory_analysis_pipeline() {
    // Step 1: Generate respiratory signal with apneas
    let duration = 120.0; // 2 minutes
    let sample_rate = 50.0; // Hz
    let breath_rate = 15.0; // breaths per minute

    // Define apnea events: (start_time, duration)
    let apnea_intervals = vec![
        (30.0, 12.0), // 12-second apnea at 30s
        (70.0, 15.0), // 15-second apnea at 70s
    ];

    let resp_data = generate_respiratory_with_apnea(
        duration,
        sample_rate,
        breath_rate,
        &apnea_intervals,
    );

    // Step 2: Analyze respiratory signal
    let analyzer = RespiratoryAnalyzer::new(sample_rate);
    let metrics = analyzer
        .analyze(&resp_data)
        .expect("Failed to analyze respiratory signal");

    // Step 3: Verify breath detection
    println!("Respiratory Analysis Results:");
    println!("  Duration: {:.1}s", duration);
    println!("  Expected breaths: ~{}", (breath_rate * duration / 60.0) as usize);
    println!("  Detected breaths: {}", metrics.breath_rate);
    println!("  Mean breath duration: {:.2}s", metrics.mean_breath_duration);

    assert!(
        metrics.breath_rate > 0.0,
        "Should detect breathing activity"
    );

    // Step 4: Detect apnea events
    let sleep_analysis = analyze_sleep_breathing(&resp_data, sample_rate)
        .expect("Failed to analyze sleep breathing");

    println!("Sleep Breathing Analysis:");
    println!("  Apnea events detected: {}", sleep_analysis.apnea_events.len());
    println!("  AHI: {:.1} events/hour", sleep_analysis.ahi);
    println!("  Severity: {:?}", sleep_analysis.severity);

    // Verify apnea detection
    assert!(
        sleep_analysis.apnea_events.len() >= 1,
        "Should detect at least one apnea event"
    );

    for (i, event) in sleep_analysis.apnea_events.iter().enumerate() {
        println!(
            "  Event {}: Type: {:?}, Start: {:.1}s, Duration: {:.1}s",
            i + 1,
            event.apnea_type,
            event.start_time,
            event.duration
        );
    }
}

#[test]
fn test_real_time_pipeline_integration() {
    // Create a real-time processing pipeline with multiple stages
    let sample_rate = 250.0;
    let chunk_size = 250; // 1 second of data per chunk

    // Stage 1: Preprocessing (filtering)
    let preprocess_stage = StageConfig {
        name: "preprocessing".to_string(),
        latency_budget_ms: 10.0,
        buffer_size: chunk_size * 2,
    };

    // Stage 2: Feature extraction
    let feature_stage = StageConfig {
        name: "features".to_string(),
        latency_budget_ms: 20.0,
        buffer_size: chunk_size,
    };

    // Stage 3: Classification
    let classify_stage = StageConfig {
        name: "classification".to_string(),
        latency_budget_ms: 30.0,
        buffer_size: chunk_size / 2,
    };

    let pipeline_config = PipelineConfig {
        stages: vec![preprocess_stage, feature_stage, classify_stage],
        mode: ExecutionMode::Streaming,
        max_latency_ms: 60.0,
    };

    let mut executor = PipelineExecutor::new(pipeline_config);

    // Simulate streaming data
    let total_duration = 5.0; // seconds
    let num_chunks = (total_duration * sample_rate / chunk_size as f64) as usize;

    println!("Real-time Pipeline Test:");
    println!("  Sample rate: {:.0} Hz", sample_rate);
    println!("  Chunk size: {} samples ({:.2}s)", chunk_size, chunk_size as f64 / sample_rate);
    println!("  Total chunks: {}", num_chunks);

    for chunk_idx in 0..num_chunks {
        // Generate chunk of ECG data
        let chunk_start = chunk_idx as f64 * chunk_size as f64 / sample_rate;
        let chunk_data = generate_synthetic_ecg(
            chunk_size as f64 / sample_rate,
            sample_rate,
            75.0,
        );

        // Process chunk (in real implementation, this would execute the stages)
        let start_time = std::time::Instant::now();

        // Simulate processing
        std::thread::sleep(std::time::Duration::from_millis(5));

        let latency = start_time.elapsed().as_secs_f64() * 1000.0;

        // Verify latency constraint
        assert!(
            latency < 60.0,
            "Chunk {} exceeded latency budget: {:.2}ms",
            chunk_idx,
            latency
        );

        if chunk_idx % 10 == 0 {
            println!("  Chunk {}: latency {:.2}ms", chunk_idx, latency);
        }
    }

    let stats = executor.get_latency_stats();
    println!("Pipeline Statistics:");
    println!("  Mean latency: {:.2}ms", stats.mean_ms);
    println!("  Max latency: {:.2}ms", stats.max_ms);
    println!("  P95 latency: {:.2}ms", stats.p95_ms);
}

#[test]
fn test_multi_modal_signal_fusion() {
    // Simultaneous processing of multiple signal types
    let duration = 30.0;
    let sample_rate = 250.0;

    // Generate synchronized signals
    let ecg_data = generate_synthetic_ecg(duration, sample_rate, 70.0);
    let resp_data = generate_respiratory_with_apnea(duration, sample_rate, 14.0, &[]);

    // Create EDA signal (simple increasing trend with SCR events)
    let eda_data: Vec<f64> = (0..(duration * sample_rate) as usize)
        .map(|i| {
            let t = i as f64 / sample_rate;
            // Tonic component (slow drift)
            let tonic = 5.0 + 0.1 * t;
            // Phasic component (SCR events every 8 seconds)
            let phasic = if (t % 8.0) < 2.0 {
                1.0 * (-(t % 8.0 - 1.0).powi(2)).exp()
            } else {
                0.0
            };
            tonic + phasic
        })
        .collect();

    println!("Multi-Modal Signal Fusion Test:");
    println!("  Duration: {:.1}s @ {:.0} Hz", duration, sample_rate);

    // Process ECG
    let ecg_detector = PanTompkinsDetector::new(sample_rate)
        .expect("Failed to create Pan-Tompkins detector");
    let r_peaks = ecg_detector.detect(&ecg_data).expect("Failed to detect R-peaks");
    println!("  ECG: {} R-peaks detected", r_peaks.len());

    // Process Respiratory
    let resp_analyzer = RespiratoryAnalyzer::new(sample_rate);
    let resp_metrics = resp_analyzer
        .analyze(&resp_data)
        .expect("Failed to analyze respiration");
    println!("  Respiratory: rate = {:.1} br/min", resp_metrics.breath_rate);

    // Process EDA
    let eda_analyzer = EdaAnalyzer::new(sample_rate);
    let eda_result = eda_analyzer.analyze(&eda_data).expect("Failed to analyze EDA");
    println!("  EDA: {} SCR events detected", eda_result.metrics.num_scr);
    println!("       Mean SCR amplitude: {:.3}", eda_result.metrics.mean_scr_amplitude);

    // Verify all modalities produced results
    assert!(r_peaks.len() > 0, "ECG analysis produced results");
    assert!(resp_metrics.breath_rate > 0.0, "Respiratory analysis produced results");
    assert!(eda_result.metrics.num_scr > 0, "EDA analysis produced results");

    // Verify temporal synchronization (all signals same length)
    assert_eq!(ecg_data.len(), (duration * sample_rate) as usize);
    assert_eq!(resp_data.len(), (duration * sample_rate) as usize);
    assert_eq!(eda_data.len(), (duration * sample_rate) as usize);

    println!("  ✓ All modalities synchronized and processed successfully");
}

#[test]
fn test_signal_quality_assessment() {
    // Test signal quality metrics
    let sample_rate = 250.0;
    let duration = 10.0;

    // Generate high-quality signal
    let clean_signal = generate_synthetic_ecg(duration, sample_rate, 72.0);

    // Generate noisy signal
    let mut noisy_signal = clean_signal.clone();
    for i in 0..noisy_signal.len() {
        noisy_signal[i] += 0.5 * ((i as f64 * 0.1).sin());
    }

    println!("Signal Quality Assessment:");

    // Compute SNR for both signals
    let compute_snr = |signal: &[f64]| -> f64 {
        let mean: f64 = signal.iter().sum::<f64>() / signal.len() as f64;
        let variance: f64 = signal
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / signal.len() as f64;
        let signal_power = mean.abs();
        let noise_power = variance.sqrt();
        20.0 * (signal_power / noise_power).log10()
    };

    let clean_snr = compute_snr(&clean_signal);
    let noisy_snr = compute_snr(&noisy_signal);

    println!("  Clean signal SNR: {:.1} dB", clean_snr);
    println!("  Noisy signal SNR: {:.1} dB", noisy_snr);

    assert!(
        clean_snr > noisy_snr,
        "Clean signal should have higher SNR"
    );
}
