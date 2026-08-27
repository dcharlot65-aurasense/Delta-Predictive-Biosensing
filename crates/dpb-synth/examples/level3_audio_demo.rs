//! Level 3 Audio Generation Demo
//!
//! Demonstrates usage of full audio waveform generators using WORLD vocoder.
//!
//! Run with:
//! ```bash
//! cargo run --example level3_audio_demo
//! ```
//!
//! Prerequisites:
//! - Python 3 with pyworld installed
//! - See tools/level3_audio/README.md for setup

use dpb_synth::SyntheticGenerator;
use dpb_synth::level3::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Level 3 Audio Generation Demo ===\n");

    // Demo 1: Sustained Vowel with Tremor (PD-like)
    demo_sustained_vowel_tremor()?;

    // Demo 2: Connected Speech with Monotone (PD-like)
    demo_connected_speech_monotone()?;

    // Demo 3: Diadochokinesis (DDK)
    demo_diadochokinesis()?;

    // Demo 4: Reading Passage
    demo_reading_passage()?;

    println!("\n=== Demo Complete ===");
    println!("\nNote: To save audio files, use the Python interface or add WAV export to Rust.");
    println!("See tools/level3_audio/README.md for details.");

    Ok(())
}

fn demo_sustained_vowel_tremor() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 1: Sustained Vowel with Vocal Tremor");
    println!("{}", "-".repeat(50));

    let generator = SustainedVowelGenerator::new();

    // Normal vowel
    let normal_params = SustainedVowelParams {
        duration_sec: 3.0,
        f0_mean: 120.0,
        f0_std: 2.0,
        jitter_percent: 0.5,
        shimmer_percent: 3.0,
        hnr_db: 22.0,
        tremor_frequency: 0.0,
        tremor_amplitude: 0.0,
        vowel: "a".to_string(),
        sample_rate: 16000,
        seed: 42,
    };

    println!("Generating normal /a/ vowel...");
    let normal_result = generator.generate(&normal_params, 42)?;
    println!("  Samples: {}", normal_result.signal.len());
    println!(
        "  Duration: {:.2} sec",
        normal_result.signal.len() as f64 / normal_result.sampling_rate
    );

    let f0_mean = normal_result
        .ground_truth
        .f0_contour
        .iter()
        .filter(|&&f| f > 0.0)
        .sum::<f64>()
        / normal_result.ground_truth.f0_contour.len() as f64;
    println!("  Mean F0: {:.1} Hz", f0_mean);

    // Pathological vowel with tremor (Parkinson's Disease)
    let tremor_params = SustainedVowelParams {
        tremor_frequency: 5.0,  // 5 Hz tremor
        tremor_amplitude: 10.0, // 10 Hz amplitude
        jitter_percent: 1.5,    // Increased jitter
        ..normal_params
    };

    println!("\nGenerating /a/ with vocal tremor (PD-like)...");
    let tremor_result = generator.generate(&tremor_params, 42)?;
    println!("  Samples: {}", tremor_result.signal.len());

    let tremor_f0_mean = tremor_result
        .ground_truth
        .f0_contour
        .iter()
        .filter(|&&f| f > 0.0)
        .sum::<f64>()
        / tremor_result.ground_truth.f0_contour.len() as f64;
    let tremor_f0_std = {
        let f0_voiced: Vec<f64> = tremor_result
            .ground_truth
            .f0_contour
            .iter()
            .copied()
            .filter(|&f| f > 0.0)
            .collect();
        let mean = tremor_f0_mean;
        let variance =
            f0_voiced.iter().map(|&f| (f - mean).powi(2)).sum::<f64>() / f0_voiced.len() as f64;
        variance.sqrt()
    };
    println!(
        "  Mean F0: {:.1} Hz (std: {:.1} Hz)",
        tremor_f0_mean, tremor_f0_std
    );
    println!("  Tremor visible in F0 variation!");

    println!();
    Ok(())
}

fn demo_connected_speech_monotone() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 2: Connected Speech with Monotone");
    println!("{}", "-".repeat(50));

    let generator = ConnectedSpeechGenerator::new();

    // Normal speech
    let normal_params = ConnectedSpeechParams {
        duration_sec: 3.0,
        base_f0: 120.0,
        f0_range: 50.0,
        speech_rate: 1.0,
        hypophonia_db: 0.0,
        monotone_factor: 0.0,
        tremor_frequency: 0.0,
        tremor_amplitude: 0.0,
        pause_probability: 0.1,
        sample_rate: 16000,
        seed: 42,
    };

    println!("Generating normal connected speech...");
    let normal_result = generator.generate(&normal_params, 42)?;
    let events = normal_result.ground_truth.events.as_ref().unwrap();
    let syllables: Vec<_> = events
        .iter()
        .filter(|e| e.event_type == "syllable")
        .collect();
    let pauses: Vec<_> = events.iter().filter(|e| e.event_type == "pause").collect();

    println!("  Samples: {}", normal_result.signal.len());
    println!("  Syllables: {}, Pauses: {}", syllables.len(), pauses.len());

    // Pathological speech (monotone + hypophonia)
    let pd_params = ConnectedSpeechParams {
        monotone_factor: 0.8, // Reduced prosodic variation
        hypophonia_db: 6.0,   // 6 dB reduction
        speech_rate: 0.8,     // Slower
        ..normal_params
    };

    println!("\nGenerating PD-like speech (monotone + hypophonia)...");
    let pd_result = generator.generate(&pd_params, 42)?;
    let pd_events = pd_result.ground_truth.events.as_ref().unwrap();
    let pd_syllables: Vec<_> = pd_events
        .iter()
        .filter(|e| e.event_type == "syllable")
        .collect();

    println!("  Samples: {}", pd_result.signal.len());
    println!("  Syllables: {}", pd_syllables.len());
    println!("  Reduced prosody and amplitude!");

    println!();
    Ok(())
}

fn demo_diadochokinesis() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 3: Diadochokinesis (Rapid Syllable Repetition)");
    println!("{}", "-".repeat(50));

    let generator = DiadochokinesisGenerator::new();

    // AMR (Alternating Motion Rate) - single syllable
    let amr_params = DiadochokinesisParams {
        syllables: "pa".to_string(),
        repetitions: 10,
        target_rate: 6.0, // 6 syllables/second
        rate_variability: 0.1,
        amplitude_variability: 0.05,
        f0_mean: 120.0,
        sample_rate: 16000,
        seed: 42,
    };

    println!("Generating AMR: /pa/-/pa/-/pa/ (10 repetitions)...");
    let amr_result = generator.generate(&amr_params, 42)?;
    let amr_events = amr_result.ground_truth.events.as_ref().unwrap();

    println!("  Samples: {}", amr_result.signal.len());
    println!("  Events: {}", amr_events.len());
    println!("  Target rate: {} syllables/sec", amr_params.target_rate);

    if let Some(actual_rate) = amr_result.ground_truth.parameters.get("actual_rate") {
        println!("  Actual rate: {:.2} syllables/sec", actual_rate);
    }

    // SMR (Sequential Motion Rate) - syllable sequence
    let smr_params = DiadochokinesisParams {
        syllables: "pa-ta-ka".to_string(),
        repetitions: 5, // 5 sequences = 15 syllables
        ..amr_params
    };

    println!("\nGenerating SMR: /pa/-/ta/-/ka/ (5 sequences)...");
    let smr_result = generator.generate(&smr_params, 42)?;
    let smr_events = smr_result.ground_truth.events.as_ref().unwrap();

    println!("  Samples: {}", smr_result.signal.len());
    println!("  Events: {} syllables", smr_events.len());

    // Show first few syllables
    println!("  Syllable sequence:");
    for (i, event) in smr_events.iter().take(9).enumerate() {
        if let Some(serde_json::Value::String(syl)) = event.attributes.get("syllable") {
            print!("    {}: /{}/", i + 1, syl);
            if (i + 1) % 3 == 0 {
                println!();
            }
        }
    }

    println!();
    Ok(())
}

fn demo_reading_passage() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 4: Reading Passage");
    println!("{}", "-".repeat(50));

    let generator = ReadingPassageGenerator::new();

    let params = ReadingPassageParams {
        passage: "standard".to_string(),
        base_f0: 120.0,
        speech_rate: 1.0,
        articulation_precision: 1.0,
        breath_pause_regularity: 1.0,
        sample_rate: 16000,
        seed: 42,
    };

    println!("Generating reading passage...");
    let result = generator.generate(&params, 42)?;

    println!("  Samples: {}", result.signal.len());
    println!(
        "  Duration: {:.2} sec",
        result.signal.len() as f64 / result.sampling_rate
    );
    println!("  F0 frames: {}", result.ground_truth.f0_contour.len());

    // Calculate F0 statistics
    let f0_voiced: Vec<f64> = result
        .ground_truth
        .f0_contour
        .iter()
        .copied()
        .filter(|&f| f > 0.0)
        .collect();

    if !f0_voiced.is_empty() {
        let f0_mean = f0_voiced.iter().sum::<f64>() / f0_voiced.len() as f64;
        let f0_min = f0_voiced.iter().copied().fold(f64::INFINITY, f64::min);
        let f0_max = f0_voiced.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        println!(
            "  F0 range: {:.1} - {:.1} Hz (mean: {:.1} Hz)",
            f0_min, f0_max, f0_mean
        );
    }

    println!();
    Ok(())
}
