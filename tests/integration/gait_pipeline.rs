//! Gait Pipeline Integration Test
//!
//! Tests the complete pipeline:
//! 1. Generate synthetic gait keypoints with known parameters
//! 2. Encode using keypoint deviation encoder
//! 3. Process through Recurrent SNN
//! 4. Decode to UPDRS gait score
//! 5. Validate against ground truth

use dpb_core::SignalBuffer;
use dpb_encoders::prelude::*;
use dpb_snn::{GaitScoreDecoder, NeuronModel, NeuronParams, RecurrentSNN, SNNConfig, SpikeTensor};
// `forward` and `decode` are trait methods.
use dpb_snn::architectures::SNNArchitecture;
use dpb_snn::decoders::Decoder;
use dpb_synth::pose::gait::{GaitCycleGenerator, GaitCycleParams};
use dpb_synth::traits::SyntheticGenerator;

use super::utils::*;

#[test]
fn test_gait_pipeline_end_to_end() {
    // Step 1: Generate synthetic gait with known parameters
    let params = GaitCycleParams {
        duration: 10.0,
        frame_rate: 30.0,
        cadence: 110.0, // normal gait
        stride_length: 1.4,
        step_width: 0.15,
        height: 1.75,
    };

    let generator = GaitCycleGenerator;
    let generated = generator
        .generate(&params, TEST_SEED)
        .expect("Failed to generate gait data");

    // Verify gait phases are generated
    let gait_phases = &generated.ground_truth.gait_phases;
    assert!(
        !gait_phases.is_empty(),
        "Should have gait phases in ground truth"
    );

    let n_frames = generated.signal.len();
    assert_eq!(
        n_frames,
        (params.duration * params.frame_rate) as usize,
        "Should generate correct number of frames"
    );

    // Step 2: Extract ankle trajectories for encoding
    // Right ankle is keypoint 28, left ankle is 27
    let right_ankle_y: Vec<f32> = generated
        .signal
        .iter()
        .map(|frame| frame[28][1] as f32)
        .collect();

    let left_ankle_y: Vec<f32> = generated
        .signal
        .iter()
        .map(|frame| frame[27][1] as f32)
        .collect();

    // Create signals for heel strikes (ankle vertical position)
    let right_signal = SignalBuffer::single_channel(right_ankle_y, params.frame_rate);
    let left_signal = SignalBuffer::single_channel(left_ankle_y, params.frame_rate);

    // Step 3: Encode heel strikes using derivative encoder
    let encoder = DerivativeEncoder::new("gait_encoder");
    let encoder_config = DerivativeConfig {
        threshold: 0.01, // Detect vertical motion changes
        ..DerivativeConfig::default()
    };

    let right_spikes = encoder
        .encode(&right_signal, &encoder_config)
        .expect("Failed to encode right ankle");
    let left_spikes = encoder
        .encode(&left_signal, &encoder_config)
        .expect("Failed to encode left ankle");

    // Step 4: Verify encoding produced spikes
    assert!(
        !right_spikes.is_empty() && !left_spikes.is_empty(),
        "Encoder should produce spike events for both legs"
    );

    println!(
        "Gait encoding: {} right spikes, {} left spikes",
        right_spikes.len(),
        left_spikes.len()
    );

    // Step 5: Create spike tensor for SNN
    let num_timesteps = (params.duration * 100.0) as usize; // 10ms bins
    let num_channels = 64;

    let mut spike_tensor = SpikeTensor::zeros(1, num_timesteps, num_channels, false);

    // Map right ankle spikes to first half of channels
    for event in &right_spikes {
        let timestep = ((event.timestamp * 100.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = event.channel as usize % (num_channels / 2);
        spike_tensor
            .set_spike(0, timestep, channel, event.magnitude)
            .ok();
    }

    // Map left ankle spikes to second half of channels
    for event in &left_spikes {
        let timestep = ((event.timestamp * 100.0).min((num_timesteps - 1) as f64)) as usize;
        let channel = (num_channels / 2) + (event.channel as usize % (num_channels / 2));
        spike_tensor
            .set_spike(0, timestep, channel, event.magnitude)
            .ok();
    }

    // Step 6: Create and run recurrent SNN
    let snn_config = SNNConfig {
        dt: 10.0, // 10ms timesteps
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    let mut snn = RecurrentSNN::new(num_channels, vec![32], 4, snn_config.clone()).expect("SNN");
    let output_spikes = snn
        .forward(&spike_tensor)
        .expect("Failed to run SNN forward pass");

    // Step 7: Decode output to gait score
    let decoder = GaitScoreDecoder::new(1);
    let gait_score = decoder
        .decode(&output_spikes)
        .expect("Failed to decode gait score");

    // Step 8: Validate output shape
    // `GaitScoreDecoder` defines five features: cadence, stride length,
    // speed, balance and freezing.
    assert_eq!(gait_score.shape()[1], 5, "Should have 5 gait metrics");

    // Verify joint angles in ground truth
    let joint_angles = &generated.ground_truth.joint_angles;
    assert!(
        joint_angles.contains_key("knee_flexion"),
        "Should have knee flexion data"
    );
    assert!(
        joint_angles.contains_key("hip_flexion"),
        "Should have hip flexion data"
    );

    println!(
        "Gait Pipeline Test Results:\n\
         - Duration: {:.1}s @ {:.0} fps\n\
         - Gait phases: {}\n\
         - Right spikes: {}\n\
         - Left spikes: {}\n\
         - Output shape: {:?}",
        params.duration,
        params.frame_rate,
        gait_phases.len(),
        right_spikes.len(),
        left_spikes.len(),
        output_spikes.shape()
    );
}

#[test]
fn test_gait_pipeline_cadence_variations() {
    // Test pipeline with different cadences (slow, normal, fast)
    let cadences = vec![(90.0, "slow"), (110.0, "normal"), (130.0, "fast")];

    for (cadence, label) in cadences {
        let params = GaitCycleParams {
            duration: 10.0,
            frame_rate: 30.0,
            cadence,
            stride_length: 1.4,
            step_width: 0.15,
            height: 1.75,
        };

        let generator = GaitCycleGenerator;
        let generated = generator
            .generate(&params, TEST_SEED)
            .expect("Failed to generate gait");

        let expected_steps = (cadence * params.duration / 60.0) as usize;
        let frames = generated.signal.len();

        assert_eq!(
            frames,
            (params.duration * params.frame_rate) as usize,
            "Cadence {}: should generate correct frame count",
            label
        );

        println!(
            "Cadence {} ({:.0} spm): {} frames, ~{} steps",
            label, cadence, frames, expected_steps
        );
    }
}

#[test]
fn test_gait_stance_swing_ratio() {
    // Verify normal gait has correct stance:swing ratio (60:40)
    let params = GaitCycleParams {
        duration: 10.0,
        frame_rate: 30.0,
        cadence: 110.0,
        stride_length: 1.4,
        step_width: 0.15,
        height: 1.75,
    };

    let generator = GaitCycleGenerator;
    let generated = generator
        .generate(&params, TEST_SEED)
        .expect("Failed to generate gait");

    let gait_phases = &generated.ground_truth.gait_phases;

    let stance_count = gait_phases
        .iter()
        .filter(|p| p.phase_name == "stance")
        .count();

    let swing_count = gait_phases
        .iter()
        .filter(|p| p.phase_name == "swing")
        .count();

    let total = stance_count + swing_count;
    let stance_ratio = stance_count as f64 / total as f64;

    // Normal gait should be ~60% stance, 40% swing
    assert_in_range(stance_ratio, 0.55, 0.65, "Stance phase ratio");

    println!(
        "Gait phase distribution: {:.1}% stance, {:.1}% swing",
        stance_ratio * 100.0,
        (1.0 - stance_ratio) * 100.0
    );
}

#[test]
fn test_gait_joint_angle_ranges() {
    // Verify joint angles are within physiological ranges
    let params = GaitCycleParams {
        duration: 5.0,
        frame_rate: 30.0,
        cadence: 110.0,
        stride_length: 1.4,
        step_width: 0.15,
        height: 1.75,
    };

    let generator = GaitCycleGenerator;
    let generated = generator
        .generate(&params, TEST_SEED)
        .expect("Failed to generate gait");

    let joint_angles = &generated.ground_truth.joint_angles;

    // Check knee flexion (should be 0-70 degrees)
    if let Some(knee_angles) = joint_angles.get("knee_flexion") {
        for &angle in knee_angles.iter() {
            assert_in_range(
                angle,
                -5.0, // slight hyperextension ok
                70.0, // max flexion in swing
                "Knee flexion angle",
            );
        }
        println!(
            "Knee flexion range: {:.1} to {:.1} degrees",
            knee_angles.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            knee_angles.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );
    }

    // Check hip flexion (should be -20 to +30 degrees)
    if let Some(hip_angles) = joint_angles.get("hip_flexion") {
        for &angle in hip_angles.iter() {
            assert_in_range(angle, -25.0, 35.0, "Hip flexion angle");
        }
        println!(
            "Hip flexion range: {:.1} to {:.1} degrees",
            hip_angles.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            hip_angles.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );
    }
}

#[test]
fn test_gait_reproducibility() {
    // Test deterministic generation with same seed
    let params = GaitCycleParams {
        duration: 5.0,
        frame_rate: 30.0,
        cadence: 110.0,
        stride_length: 1.4,
        step_width: 0.15,
        height: 1.75,
    };

    let generator = GaitCycleGenerator;

    let gen1 = generator
        .generate(&params, TEST_SEED)
        .expect("Failed to generate (run 1)");
    let gen2 = generator
        .generate(&params, TEST_SEED)
        .expect("Failed to generate (run 2)");

    assert_eq!(
        gen1.signal.len(),
        gen2.signal.len(),
        "Frame count should match"
    );
    assert_eq!(
        gen1.ground_truth.gait_phases.len(),
        gen2.ground_truth.gait_phases.len(),
        "Gait phase count should match"
    );

    // First frame should be identical
    for kp in 0..gen1.signal[0].len() {
        for coord in 0..3 {
            assert_approx_eq(
                gen1.signal[0][kp][coord],
                gen2.signal[0][kp][coord],
                &format!("Frame 0, keypoint {}, coord {}", kp, coord),
            );
        }
    }
}
