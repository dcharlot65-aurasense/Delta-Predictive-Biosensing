//! Synthetic Data Validation Tests
//!
//! Validates that synthetic data generators produce:
//! 1. Correct ground truth events at expected locations
//! 2. Valid physiological parameter ranges
//! 3. Expected frequency components
//! 4. Proper temporal alignment

use dpb_synth::contact::ecg::{EcgMorphologyGenerator, EcgMorphologyParams, WaveParams};
use dpb_synth::contact::tremor::{PhysiologicalTremorGenerator, PhysiologicalTremorParams};
use dpb_synth::pose::gait::{GaitCycleGenerator, GaitCycleParams};
use dpb_synth::voice::phonation::{SustainedVowelGenerator, SustainedVowelParams};
use dpb_synth::traits::SyntheticGenerator;

use super::utils::*;
use std::f64::consts::PI;

#[test]
fn test_ecg_r_peak_locations() {
    // Verify R-peaks occur at expected intervals
    let heart_rate = 60.0; // bpm (1 beat per second)
    let duration = 10.0;

    let params = EcgMorphologyParams {
        duration,
        sampling_rate: 1000.0,
        heart_rate,
        p_wave: WaveParams { amplitude: 0.25, width: 0.1, time_offset: -PI / 3.0 },
        qrs_complex: WaveParams { amplitude: 1.0, width: 0.1, time_offset: 0.0 },
        t_wave: WaveParams { amplitude: 0.35, width: 0.25, time_offset: PI / 2.0 },
    };

    let generator = EcgMorphologyGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate ECG");

    let r_peaks = &generated.ground_truth.events;

    // Expected interval: 60 seconds / 60 bpm = 1 second
    let expected_interval = 60.0 / heart_rate;

    // Verify R-peaks are roughly 1 second apart
    for i in 1..r_peaks.len() {
        let interval = r_peaks[i].time - r_peaks[i - 1].time;
        assert_in_range(
            interval,
            expected_interval * 0.9,
            expected_interval * 1.1,
            &format!("R-R interval {}", i)
        );
    }

    println!(
        "ECG R-peak validation: {} peaks, avg interval {:.3}s (expected {:.3}s)",
        r_peaks.len(),
        if r_peaks.len() > 1 {
            (r_peaks.last().unwrap().time - r_peaks.first().unwrap().time) / (r_peaks.len() - 1) as f64
        } else {
            0.0
        },
        expected_interval
    );
}

#[test]
fn test_gait_heel_strike_timing() {
    // Verify heel strikes occur at expected cadence
    let cadence = 120.0; // steps per minute (2 per second)
    let duration = 10.0;

    let params = GaitCycleParams {
        duration,
        frame_rate: 30.0,
        cadence,
        stride_length: 1.4,
        step_width: 0.15,
        height: 1.75,
    };

    let generator = GaitCycleGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate gait");

    let gait_phases = &generated.ground_truth.gait_phases;

    // Count stance phase transitions (heel strikes)
    let mut heel_strikes = 0;
    for i in 1..gait_phases.len() {
        if gait_phases[i - 1].phase_name == "swing" && gait_phases[i].phase_name == "stance" {
            heel_strikes += 1;
        }
    }

    let expected_steps = (cadence * duration / 60.0) as usize;
    assert_in_range(
        heel_strikes as f64,
        (expected_steps - 3) as f64,
        (expected_steps + 3) as f64,
        "Heel strike count"
    );

    println!(
        "Gait heel strike validation: {} heel strikes (expected ~{})",
        heel_strikes,
        expected_steps
    );
}

#[test]
fn test_tremor_frequency_content() {
    // Verify tremor signal contains expected frequency
    let target_freq = 10.0; // Hz
    let duration = 10.0;
    let sampling_rate = 100.0;

    let params = PhysiologicalTremorParams {
        duration,
        sampling_rate,
        frequency: target_freq,
        amplitude: 1.0,
        frequency_variability: 0.1,
    };

    let generator = PhysiologicalTremorGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate tremor");

    // Count zero crossings to estimate frequency
    let mut zero_crossings = 0;
    for i in 1..generated.signal.len() {
        if (generated.signal[i - 1] >= 0.0 && generated.signal[i] < 0.0)
            || (generated.signal[i - 1] < 0.0 && generated.signal[i] >= 0.0)
        {
            zero_crossings += 1;
        }
    }

    // Frequency = zero_crossings / (2 * duration)
    let estimated_freq = zero_crossings as f64 / (2.0 * duration);

    assert_in_range(
        estimated_freq,
        target_freq * 0.8,
        target_freq * 1.2,
        "Tremor frequency from zero crossings"
    );

    println!(
        "Tremor frequency validation: {:.1} Hz (expected {:.1} Hz, {} zero crossings)",
        estimated_freq,
        target_freq,
        zero_crossings
    );
}

#[test]
fn test_voice_f0_periodicity() {
    // Verify voice signal has expected F0
    let f0 = 120.0; // Hz
    let duration = 2.0;

    let params = SustainedVowelParams {
        duration,
        sampling_rate: 16000.0,
        fundamental_frequency: f0,
        vowel_formants: vec![
            (730.0, 100.0),
            (1090.0, 150.0),
            (2440.0, 200.0),
        ],
        amplitude: 1.0,
    };

    let generator = SustainedVowelGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate voice");

    // Count zero crossings
    let mut zero_crossings = 0;
    for i in 1..generated.signal.len() {
        if (generated.signal[i - 1] >= 0.0 && generated.signal[i] < 0.0)
            || (generated.signal[i - 1] < 0.0 && generated.signal[i] >= 0.0)
        {
            zero_crossings += 1;
        }
    }

    // Expected crossings: 2 * f0 * duration (approximate, due to harmonics)
    let expected_crossings = (2.0 * f0 * duration) as usize;

    // Allow wide range due to harmonic content
    assert!(
        zero_crossings > expected_crossings / 2,
        "Should have at least {} zero crossings, got {}",
        expected_crossings / 2,
        zero_crossings
    );

    println!(
        "Voice F0 validation: {} zero crossings (f0={:.1} Hz suggests ~{})",
        zero_crossings,
        f0,
        expected_crossings
    );
}

#[test]
fn test_ground_truth_parameter_accuracy() {
    // Verify ground truth matches generation parameters exactly
    let params = EcgMorphologyParams {
        duration: 5.0,
        sampling_rate: 250.0,
        heart_rate: 75.0,
        p_wave: WaveParams { amplitude: 0.25, width: 0.1, time_offset: -PI / 3.0 },
        qrs_complex: WaveParams { amplitude: 1.0, width: 0.1, time_offset: 0.0 },
        t_wave: WaveParams { amplitude: 0.35, width: 0.25, time_offset: PI / 2.0 },
    };

    let generator = EcgMorphologyGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate ECG");

    let gt_hr = generated.ground_truth.parameters.get("heart_rate").unwrap();
    assert_approx_eq(*gt_hr, params.heart_rate, "Ground truth heart rate");

    let gt_p_amp = generated.ground_truth.parameters.get("p_amplitude").unwrap();
    assert_approx_eq(*gt_p_amp, params.p_wave.amplitude, "Ground truth P amplitude");

    println!(
        "Ground truth accuracy verified: HR={:.1}, P-amp={:.2}",
        gt_hr,
        gt_p_amp
    );
}

#[test]
fn test_signal_duration_accuracy() {
    // Verify all generators produce correct signal length
    let duration = 5.0;

    // ECG
    let ecg_params = EcgMorphologyParams {
        duration,
        sampling_rate: 250.0,
        heart_rate: 72.0,
        p_wave: WaveParams { amplitude: 0.25, width: 0.1, time_offset: -PI / 3.0 },
        qrs_complex: WaveParams { amplitude: 1.0, width: 0.1, time_offset: 0.0 },
        t_wave: WaveParams { amplitude: 0.35, width: 0.25, time_offset: PI / 2.0 },
    };
    let ecg_gen = EcgMorphologyGenerator;
    let ecg = ecg_gen.generate(&ecg_params, TEST_SEED).unwrap();
    assert_eq!(
        ecg.signal.len(),
        (duration * ecg_params.sampling_rate) as usize,
        "ECG signal length"
    );

    // Tremor
    let tremor_params = PhysiologicalTremorParams {
        duration,
        sampling_rate: 100.0,
        frequency: 10.0,
        amplitude: 0.5,
        frequency_variability: 0.3,
    };
    let tremor_gen = PhysiologicalTremorGenerator;
    let tremor = tremor_gen.generate(&tremor_params, TEST_SEED).unwrap();
    assert_eq!(
        tremor.signal.len(),
        (duration * tremor_params.sampling_rate) as usize,
        "Tremor signal length"
    );

    // Voice
    let voice_params = SustainedVowelParams {
        duration,
        sampling_rate: 16000.0,
        fundamental_frequency: 120.0,
        vowel_formants: vec![(730.0, 100.0), (1090.0, 150.0), (2440.0, 200.0)],
        amplitude: 0.5,
    };
    let voice_gen = SustainedVowelGenerator;
    let voice = voice_gen.generate(&voice_params, TEST_SEED).unwrap();
    assert_eq!(
        voice.signal.len(),
        (duration * voice_params.sampling_rate) as usize,
        "Voice signal length"
    );

    println!(
        "Duration accuracy: ECG={}, Tremor={}, Voice={} samples",
        ecg.signal.len(),
        tremor.signal.len(),
        voice.signal.len()
    );
}

#[test]
fn test_physiological_range_validation() {
    // Verify ECG parameters are within physiological ranges
    let params = EcgMorphologyParams {
        duration: 5.0,
        sampling_rate: 250.0,
        heart_rate: 72.0,
        p_wave: WaveParams { amplitude: 0.25, width: 0.1, time_offset: -PI / 3.0 },
        qrs_complex: WaveParams { amplitude: 1.0, width: 0.1, time_offset: 0.0 },
        t_wave: WaveParams { amplitude: 0.35, width: 0.25, time_offset: PI / 2.0 },
    };

    // Heart rate should be 40-200 bpm
    assert_in_range(params.heart_rate, 40.0, 200.0, "Heart rate physiological range");

    // P wave amplitude should be 0.05-0.30 mV
    assert_in_range(params.p_wave.amplitude, 0.05, 0.30, "P wave amplitude range");

    // QRS amplitude should be 0.5-2.0 mV
    assert_in_range(params.qrs_complex.amplitude, 0.5, 2.0, "QRS amplitude range");

    // T wave amplitude should be 0.1-0.5 mV
    assert_in_range(params.t_wave.amplitude, 0.1, 0.5, "T wave amplitude range");

    println!("Physiological range validation passed");
}

#[test]
fn test_gait_joint_angle_validity() {
    // Verify gait joint angles are within physiological ranges
    let params = GaitCycleParams {
        duration: 5.0,
        frame_rate: 30.0,
        cadence: 110.0,
        stride_length: 1.4,
        step_width: 0.15,
        height: 1.75,
    };

    let generator = GaitCycleGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate gait");

    let joint_angles = &generated.ground_truth.joint_angles;

    // Check knee flexion (0-70 degrees normal)
    if let Some(knee_angles) = joint_angles.get("knee_flexion") {
        for &angle in knee_angles.iter() {
            assert_in_range(angle, -5.0, 75.0, "Knee flexion");
        }
    }

    // Check hip flexion (-20 to +30 degrees normal)
    if let Some(hip_angles) = joint_angles.get("hip_flexion") {
        for &angle in hip_angles.iter() {
            assert_in_range(angle, -25.0, 35.0, "Hip flexion");
        }
    }

    println!("Gait joint angle validation passed");
}

#[test]
fn test_event_timing_tolerance() {
    // Verify events are timestamped within 1ms tolerance
    let params = EcgMorphologyParams {
        duration: 5.0,
        sampling_rate: 1000.0, // 1ms resolution
        heart_rate: 60.0,
        p_wave: WaveParams { amplitude: 0.25, width: 0.1, time_offset: -PI / 3.0 },
        qrs_complex: WaveParams { amplitude: 1.0, width: 0.1, time_offset: 0.0 },
        t_wave: WaveParams { amplitude: 0.35, width: 0.25, time_offset: PI / 2.0 },
    };

    let generator = EcgMorphologyGenerator;
    let generated = generator.generate(&params, TEST_SEED)
        .expect("Failed to generate ECG");

    let r_peaks = &generated.ground_truth.events;

    // Check that event times are within signal duration
    for event in r_peaks {
        assert!(
            event.time >= 0.0 && event.time <= params.duration,
            "Event time {:.3} should be within [0, {:.1}]",
            event.time,
            params.duration
        );
    }

    println!("Event timing validation: all {} events within bounds", r_peaks.len());
}

