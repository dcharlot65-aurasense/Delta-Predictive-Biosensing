//! Demonstration of 60-second landmark streaming for MediaPipe/OpenPose
//!
//! This example shows two ways to generate synthetic landmark streams:
//! 1. Natural language API - parse content descriptions like "60 seconds of parkinsonian walking"
//! 2. Programmatic API - use builders and direct parameter configuration
//!
//! Run with: cargo run --example landmark_streaming_demo

use dpb_synth::{
    // Natural language content description API
    ContentDescription, ContentDescriptionBuilder, Severity, LandmarkFormat,
    generate_from_description, stream_from_description,
    // Low-level streaming API
    streaming::{
        ClinicalGaitType, ClinicalPoseFrame, FrameStreamingGenerator,
        StreamingClinicalPose, StreamingClinicalPoseParams,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DPB Landmark Streaming Demonstration ===\n");

    // =========================================================================
    // Method 1: Natural Language API
    // =========================================================================
    println!("--- Method 1: Natural Language Content Description ---\n");

    // Example 1: Simple 60-second normal walking
    let desc = ContentDescription::parse("60 seconds of walking at 30 fps")?;
    println!("Parsed: '60 seconds of walking at 30 fps'");
    println!("  Duration: {:.1}s", desc.duration_secs);
    println!("  Frame rate: {:.0} fps", desc.frame_rate);
    println!("  Gait type: {:?}", desc.gait_type);
    println!("  Total frames: {}", desc.total_frames());
    println!();

    // Example 2: Parkinsonian gait with severity
    let desc = ContentDescription::parse(
        "60 seconds of parkinsonian walking with mild shuffling at 30 fps"
    )?;
    println!("Parsed: '60 seconds of parkinsonian walking with mild shuffling'");
    println!("  Gait type: {:?}", desc.gait_type);
    println!("  Severity: {:?}", desc.severity);
    println!("  Shuffling: {:?}", desc.shuffling_override);
    println!();

    // Example 3: Complex description
    let desc = ContentDescription::parse(
        "2 minutes of festinating gait with severe shuffling and reduced arm swing at 60 fps seed 42"
    )?;
    println!("Parsed: '2 minutes of festinating gait with severe shuffling...'");
    println!("  Duration: {:.1}s", desc.duration_secs);
    println!("  Frame rate: {:.0} fps", desc.frame_rate);
    println!("  Gait type: {:?}", desc.gait_type);
    println!("  Severity: {:?}", desc.severity);
    println!("  Total frames: {}", desc.total_frames());
    println!("  Seed: {:?}", desc.seed);
    println!();

    // =========================================================================
    // Method 2: Builder API (Programmatic)
    // =========================================================================
    println!("--- Method 2: Programmatic Builder API ---\n");

    let desc = ContentDescriptionBuilder::new()
        .duration_secs(60.0)
        .frame_rate(30.0)
        .parkinsonian()
        .mild()
        .shuffling(0.3)
        .arm_swing_reduction(0.4)
        .trunk_flexion(10.0)
        .height(1.75)
        .seed(42)
        .build();

    println!("Built ContentDescription:");
    println!("  Duration: {:.1}s", desc.duration_secs);
    println!("  Frame rate: {:.0} fps", desc.frame_rate);
    println!("  Gait type: {:?}", desc.gait_type);
    println!("  Total frames: {}", desc.total_frames());
    println!();

    // =========================================================================
    // Generating Landmarks
    // =========================================================================
    println!("--- Generating 60-Second Landmark Stream ---\n");

    // Generate a 60-second stream
    let content = "60 seconds of parkinsonian walking with mild shuffling at 30 fps";
    println!("Request: '{}'", content);

    let desc = ContentDescription::parse(content)?;
    let landmarks = desc.generate_landmarks()?;

    println!("Generated {} frames", landmarks.len());
    println!();

    // Analyze first few frames
    println!("First 5 frames:");
    for (i, frame) in landmarks.iter().take(5).enumerate() {
        println!("  Frame {}: phase={:.3}, step_length={:.3}m, arm_swing={:.3}, freezing={}",
            i, frame.gait_phase, frame.step_length, frame.arm_swing, frame.freezing);
    }
    println!();

    // Analyze keypoints (MediaPipe 33 keypoints)
    let first_frame = &landmarks[0];
    println!("Frame 0 keypoint structure (MediaPipe format - 33 keypoints):");
    println!("  Total keypoints: {}", first_frame.keypoints.len());

    // Print key body landmarks
    let keypoint_names = [
        (0, "Nose"),
        (11, "Left Shoulder"),
        (12, "Right Shoulder"),
        (23, "Left Hip"),
        (24, "Right Hip"),
        (25, "Left Knee"),
        (26, "Right Knee"),
        (27, "Left Ankle"),
        (28, "Right Ankle"),
    ];

    println!("  Sample keypoints:");
    for (idx, name) in keypoint_names.iter() {
        if *idx < first_frame.keypoints.len() {
            let [x, y, z] = first_frame.keypoints[*idx];
            println!("    {}: [{:.3}, {:.3}, {:.3}]", name, x, y, z);
        }
    }
    println!();

    // =========================================================================
    // Streaming Generation (Memory-Efficient)
    // =========================================================================
    println!("--- Streaming Generation (Memory-Efficient) ---\n");

    let stream = stream_from_description("10 seconds at 30 fps")?;
    println!("Created streaming iterator");
    println!("  Remaining frames: {}", stream.remaining());

    let mut frame_count = 0;
    let mut max_swing = 0.0f64;
    let mut min_step = f64::MAX;

    for frame in stream {
        frame_count += 1;
        max_swing = max_swing.max(frame.arm_swing);
        min_step = min_step.min(frame.step_length);
    }

    println!("Processed {} frames via streaming", frame_count);
    println!("  Max arm swing: {:.3}", max_swing);
    println!("  Min step length: {:.3}m", min_step);
    println!();

    // =========================================================================
    // Comparing Gait Types
    // =========================================================================
    println!("--- Comparing Clinical Gait Types ---\n");

    let gait_types = [
        ("Normal", ClinicalGaitType::Normal),
        ("Parkinsonian", ClinicalGaitType::Parkinsonian),
        ("Festinating", ClinicalGaitType::Festinating),
        ("Freezing Episodes", ClinicalGaitType::FreezingEpisodes),
    ];

    for (name, gait_type) in gait_types.iter() {
        let desc = ContentDescriptionBuilder::new()
            .duration_secs(5.0)
            .frame_rate(30.0)
            .gait_type(*gait_type)
            .moderate()
            .seed(42)
            .build();

        let frames = desc.generate_landmarks()?;

        // Calculate averages
        let avg_step_length: f64 = frames.iter().map(|f| f.step_length).sum::<f64>() / frames.len() as f64;
        let avg_arm_swing: f64 = frames.iter().map(|f| f.arm_swing).sum::<f64>() / frames.len() as f64;
        let freezing_frames = frames.iter().filter(|f| f.freezing).count();

        println!("{} gait:", name);
        println!("  Avg step length: {:.3}m", avg_step_length);
        println!("  Avg arm swing: {:.3}", avg_arm_swing);
        println!("  Freezing frames: {} ({:.1}%)", freezing_frames, 100.0 * freezing_frames as f64 / frames.len() as f64);
        println!();
    }

    // =========================================================================
    // Low-Level API (Direct StreamingClinicalPose)
    // =========================================================================
    println!("--- Low-Level Streaming API ---\n");

    let generator = StreamingClinicalPose;
    let params = StreamingClinicalPoseParams {
        frame_rate: 60.0,
        gait_type: ClinicalGaitType::Parkinsonian,
        cadence: 90.0, // steps per minute
        stride_length: 0.6,
        step_width: 0.18,
        height: 1.70,
        shuffling_severity: 0.6,
        arm_swing_asymmetry: 0.4,
        arm_swing_reduction: 0.7,
        trunk_flexion: 20.0,
        festination_factor: 0.0,
        freezing_probability: 0.05,
        freezing_duration: (1.5, 4.0),
        noise: 0.02,
        duration: Some(60.0),
    };

    let mut state = generator.init_state(&params, 12345);

    println!("Low-level parameters:");
    println!("  Frame rate: {} fps", params.frame_rate);
    println!("  Cadence: {} steps/min", params.cadence);
    println!("  Stride length: {}m", params.stride_length);
    println!("  Shuffling: {}", params.shuffling_severity);
    println!("  Freezing prob: {}", params.freezing_probability);

    // Generate first few frames
    let mut frames: Vec<ClinicalPoseFrame> = Vec::with_capacity(10);
    for _ in 0..10 {
        frames.push(generator.next_frame(&mut state));
    }

    println!("  Generated 10 frames at 60fps");
    println!("  Time elapsed: {:.3}s", 10.0 / 60.0);
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("=== Summary ===\n");
    println!("The DPB synthetic landmark generator supports:");
    println!("  - MediaPipe 33-keypoint format");
    println!("  - Clinical gait patterns: Normal, Parkinsonian, Festinating, Freezing");
    println!("  - Configurable: duration, frame rate, severity, specific parameters");
    println!("  - Natural language parsing: '60 seconds of parkinsonian walking'");
    println!("  - Streaming generation for memory efficiency");
    println!("  - Reproducible with seed control");
    println!();
    println!("To generate a 60-second video/stream of landmarks:");
    println!("  let landmarks = generate_from_description(");
    println!("      \"60 seconds of parkinsonian walking at 30 fps\"");
    println!("  )?;");
    println!("  // Returns 1800 frames with 33 keypoints each");

    Ok(())
}
