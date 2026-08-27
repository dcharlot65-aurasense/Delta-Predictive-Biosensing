//! Example demonstrating Level 3 audio generation backends
//!
//! This example shows how to use the Festival TTS and Praat backends
//! for synthetic voice audio generation.
//!
//! Run with:
//! ```bash
//! cargo run --example level3_audio_backends
//! ```

use dpb_synth::level3::audio::{AudioBackend, Level3AudioGenerator, VoiceAudioParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Level 3 Audio Generation - Backend Examples\n");

    let generator = Level3AudioGenerator::new();

    // Example 1: Festival TTS Backend
    println!("=== Festival TTS Backend ===");
    let festival_params = VoiceAudioParams {
        text: "The patient exhibits reduced vocal intensity and monotone speech.".to_string(),
        backend: AudioBackend::Festival,
        voice: Some("kal_diphone".to_string()),
        speaking_rate: 140.0,
        pitch_mean: 120.0,
        pitch_std: 15.0,
        volume: 0.7,
        hypophonia_severity: 0.3,
        monotonicity: 0.5,
        dysarthria_severity: 0.0,
        tremor_frequency: 0.0,
        tremor_amplitude: 0.0,
        output_path: "/tmp/festival_example.wav".to_string(),
        ground_truth_path: "/tmp/festival_ground_truth.json".to_string(),
        sample_rate: 16000,
        format: "wav".to_string(),
    };

    match generator.check_backend(AudioBackend::Festival)? {
        true => {
            println!("Festival is available");
            match generator.generate_with_festival(&festival_params) {
                Ok(output) => {
                    println!("✓ Generated audio: {:?}", output.audio_path);
                    println!("  Duration: {:.2}s", output.duration_sec);
                    println!("  Ground truth: {:?}", output.ground_truth_path);
                }
                Err(e) => println!("✗ Generation failed: {}", e),
            }
        }
        false => println!("Festival not installed (this is expected in CI/CD)"),
    }

    println!();

    // Example 2: Praat Backend with Voice Tremor
    println!("=== Praat Backend with Tremor ===");
    let praat_params = VoiceAudioParams {
        text: "Parkinsonian voice characteristics with tremor.".to_string(),
        backend: AudioBackend::Praat,
        voice: None,
        speaking_rate: 130.0,
        pitch_mean: 150.0,
        pitch_std: 20.0,
        volume: 0.6,
        hypophonia_severity: 0.4,
        monotonicity: 0.6,
        dysarthria_severity: 0.0,
        tremor_frequency: 5.0, // 5 Hz tremor (typical for PD)
        tremor_amplitude: 0.4,
        output_path: "/tmp/praat_tremor_example.wav".to_string(),
        ground_truth_path: "/tmp/praat_tremor_ground_truth.json".to_string(),
        sample_rate: 16000,
        format: "wav".to_string(),
    };

    match generator.check_backend(AudioBackend::Praat)? {
        true => {
            println!("Praat is available");
            match generator.generate_with_praat(&praat_params) {
                Ok(output) => {
                    println!("✓ Generated audio: {:?}", output.audio_path);
                    println!("  Duration: {:.2}s", output.duration_sec);
                    println!("  Ground truth: {:?}", output.ground_truth_path);
                    println!("  Metadata: {:?}", output.metadata);

                    // Load and inspect ground truth
                    if let Ok(gt) = generator.load_voice_ground_truth(&output.ground_truth_path) {
                        println!("\n  Ground Truth Details:");
                        println!("    F0 samples: {}", gt.f0_contour.len());
                        if let Some(formants) = &gt.formants {
                            println!("    Formant frames: {}", formants.len());
                            if let Some(first) = formants.first() {
                                println!(
                                    "    First frame - F1: {:.1} Hz, F2: {:.1} Hz, F3: {:.1} Hz",
                                    first.f1, first.f2, first.f3
                                );
                            }
                        }
                    }
                }
                Err(e) => println!("✗ Generation failed: {}", e),
            }
        }
        false => println!("Praat not installed (this is expected in CI/CD)"),
    }

    println!();

    // Example 3: Comparing backends
    println!("=== Backend Comparison ===");
    println!("espeak-ng: Fast, lightweight TTS (good for rapid prototyping)");
    println!("Festival:  Classic research TTS with extensive voice control");
    println!("Praat:     Acoustic modification tool (best for pathological features)");

    println!("\n=== Parameter Effects ===");
    println!("hypophonia_severity: 0.0-1.0 (reduces volume)");
    println!("monotonicity: 0.0-1.0 (flattens pitch variation)");
    println!("tremor_frequency: 4-6 Hz typical for PD voice tremor");
    println!("tremor_amplitude: 0.0-1.0 (tremor strength)");

    Ok(())
}
