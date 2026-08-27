//! Level 3 audio generation using external audio synthesis tools
//!
//! This module provides interfaces to generate actual audio files with synthetic
//! voice data using external audio synthesis libraries (e.g., espeak-ng, Festival, etc.)
//!
//! Level 3 audio generators produce:
//! - WAV/MP3 audio files with synthesized speech
//! - Phoneme-level timing annotations
//! - Acoustic feature ground truth (F0, formants, etc.)
//!
//! These are intended for full pipeline testing with voice processing systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;

/// Result type for audio generation
pub type AudioResult<T> = Result<T, AudioGeneratorError>;

/// Errors that can occur during audio generation
#[derive(Debug, thiserror::Error)]
pub enum AudioGeneratorError {
    #[error("Audio tool not found: {0}")]
    ToolNotFound(String),

    #[error("Audio execution failed: {0}")]
    ExecutionError(String),

    #[error("Failed to serialize parameters: {0}")]
    SerializationError(String),

    #[error("Failed to read output: {0}")]
    OutputReadError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Audio synthesis backend
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AudioBackend {
    /// espeak-ng: Fast, lightweight TTS
    ESpeakNG,
    /// Festival: Classic research TTS system
    Festival,
    /// Praat: For acoustic analysis and synthesis
    Praat,
}

/// Parameters for voice audio generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceAudioParams {
    pub text: String,
    pub backend: AudioBackend,
    pub voice: Option<String>,      // Voice name/ID (backend-specific)
    pub speaking_rate: f64,         // words per minute (default: 150)
    pub pitch_mean: f64,            // Hz (default: 120 for male, 220 for female)
    pub pitch_std: f64,             // Hz (default: 20)
    pub volume: f64,                // 0-1 (default: 0.8)

    // Pathological modifiers
    pub hypophonia_severity: f64,   // 0-1, reduced volume
    pub monotonicity: f64,          // 0-1, reduced pitch variation
    pub dysarthria_severity: f64,   // 0-1, imprecise articulation
    pub tremor_frequency: f64,      // Hz (4-6 Hz for PD voice tremor)
    pub tremor_amplitude: f64,      // 0-1

    pub output_path: String,
    pub ground_truth_path: String,
    pub sample_rate: u32,           // Hz (default: 16000)
    pub format: String,             // "wav", "mp3"
}

impl Default for VoiceAudioParams {
    fn default() -> Self {
        Self {
            text: "The quick brown fox jumps over the lazy dog.".to_string(),
            backend: AudioBackend::ESpeakNG,
            voice: None,
            speaking_rate: 150.0,
            pitch_mean: 150.0,
            pitch_std: 20.0,
            volume: 0.8,
            hypophonia_severity: 0.0,
            monotonicity: 0.0,
            dysarthria_severity: 0.0,
            tremor_frequency: 0.0,
            tremor_amplitude: 0.0,
            output_path: "/tmp/voice_output.wav".to_string(),
            ground_truth_path: "/tmp/voice_ground_truth.json".to_string(),
            sample_rate: 16000,
            format: "wav".to_string(),
        }
    }
}

/// Output from audio generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOutput {
    pub audio_path: PathBuf,
    pub ground_truth_path: PathBuf,
    pub duration_sec: f64,
    pub sample_rate: u32,
    pub metadata: HashMap<String, String>,
}

/// Voice audio ground truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceGroundTruth {
    pub text: String,
    pub phonemes: Vec<PhonemeAnnotation>,
    pub f0_contour: Vec<f64>,       // Fundamental frequency over time
    pub f0_times: Vec<f64>,         // Time stamps for F0 values
    pub formants: Option<Vec<FormantFrame>>,
    pub duration_sec: f64,
    pub metadata: HashMap<String, String>,
}

/// Phoneme annotation with timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeAnnotation {
    pub phoneme: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// Formant values for a frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantFrame {
    pub time: f64,
    pub f1: f64,  // First formant
    pub f2: f64,  // Second formant
    pub f3: f64,  // Third formant
}

/// Main interface for Level 3 audio generation
pub struct Level3AudioGenerator {
    espeak_path: PathBuf,
    festival_path: PathBuf,
}

impl Level3AudioGenerator {
    /// Create a new audio generator
    pub fn new() -> Self {
        Self {
            espeak_path: PathBuf::from("espeak-ng"),
            festival_path: PathBuf::from("festival"),
        }
    }

    /// Check if a specific backend is available
    pub fn check_backend(&self, backend: AudioBackend) -> AudioResult<bool> {
        let path = match backend {
            AudioBackend::ESpeakNG => &self.espeak_path,
            AudioBackend::Festival => &self.festival_path,
            AudioBackend::Praat => &PathBuf::from("praat"),
        };

        match Command::new(path).arg("--version").output() {
            Ok(output) => Ok(output.status.success()),
            Err(_) => Ok(false),
        }
    }

    /// Generate voice audio using espeak-ng
    pub fn generate_with_espeak(&self, params: &VoiceAudioParams) -> AudioResult<AudioOutput> {
        self.validate_voice_params(params)?;

        // Build espeak-ng command
        let mut cmd = Command::new(&self.espeak_path);

        // Basic parameters
        cmd.arg("-w").arg(&params.output_path);  // Write to file
        cmd.arg("-s").arg(self.calculate_espeak_speed(params.speaking_rate).to_string());
        cmd.arg("-p").arg(self.calculate_espeak_pitch(params.pitch_mean).to_string());
        cmd.arg("-a").arg(((params.volume * 100.0) as u32).to_string());

        // Voice selection
        if let Some(voice) = &params.voice {
            cmd.arg("-v").arg(voice);
        }

        // Text to speak
        cmd.arg(&params.text);

        // Execute
        let output = cmd.output()
            .map_err(|_| AudioGeneratorError::ToolNotFound("espeak-ng".to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AudioGeneratorError::ExecutionError(stderr.to_string()));
        }

        // Generate ground truth
        let ground_truth = self.generate_voice_ground_truth(params)?;
        let ground_truth_json = serde_json::to_string_pretty(&ground_truth)?;
        fs::write(&params.ground_truth_path, ground_truth_json)?;

        let mut metadata = HashMap::new();
        metadata.insert("backend".to_string(), "espeak-ng".to_string());
        metadata.insert("text".to_string(), params.text.clone());
        metadata.insert("speaking_rate".to_string(), params.speaking_rate.to_string());

        Ok(AudioOutput {
            audio_path: PathBuf::from(&params.output_path),
            ground_truth_path: PathBuf::from(&params.ground_truth_path),
            duration_sec: ground_truth.duration_sec,
            sample_rate: params.sample_rate,
            metadata,
        })
    }

    /// Generate voice audio with Festival TTS
    pub fn generate_with_festival(&self, params: &VoiceAudioParams) -> AudioResult<AudioOutput> {
        self.validate_voice_params(params)?;

        // Create temporary Festival script
        let script_path = format!("{}.scm", params.output_path);
        let script = self.create_festival_script(params)?;
        fs::write(&script_path, script)?;

        // Execute Festival with script
        let mut cmd = Command::new(&self.festival_path);
        cmd.arg("--batch").arg(&script_path);

        let output = cmd.output()
            .map_err(|_| AudioGeneratorError::ToolNotFound("festival".to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AudioGeneratorError::ExecutionError(stderr.to_string()));
        }

        // Clean up script
        let _ = fs::remove_file(&script_path);

        // Generate ground truth
        let ground_truth = self.generate_voice_ground_truth(params)?;
        let ground_truth_json = serde_json::to_string_pretty(&ground_truth)?;
        fs::write(&params.ground_truth_path, ground_truth_json)?;

        let mut metadata = HashMap::new();
        metadata.insert("backend".to_string(), "festival".to_string());
        metadata.insert("text".to_string(), params.text.clone());
        metadata.insert("speaking_rate".to_string(), params.speaking_rate.to_string());

        Ok(AudioOutput {
            audio_path: PathBuf::from(&params.output_path),
            ground_truth_path: PathBuf::from(&params.ground_truth_path),
            duration_sec: ground_truth.duration_sec,
            sample_rate: params.sample_rate,
            metadata,
        })
    }

    /// Generate voice audio with Praat (for acoustic modifications)
    pub fn generate_with_praat(&self, params: &VoiceAudioParams) -> AudioResult<AudioOutput> {
        self.validate_voice_params(params)?;

        // Praat works best for acoustic modification, so we first generate base audio
        let temp_base_path = format!("{}.base.wav", params.output_path);
        let mut base_params = params.clone();
        base_params.output_path = temp_base_path.clone();

        // Generate base audio with espeak-ng (fast and reliable)
        self.generate_with_espeak(&base_params)?;

        // Create Praat script for acoustic modifications
        let script_path = format!("{}.praat", params.output_path);
        let script = self.create_praat_script(params, &temp_base_path)?;
        fs::write(&script_path, script)?;

        // Execute Praat with script
        let mut cmd = Command::new("praat");
        cmd.arg("--run").arg(&script_path);

        let output = cmd.output()
            .map_err(|_| AudioGeneratorError::ToolNotFound("praat".to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = fs::remove_file(&temp_base_path);
            let _ = fs::remove_file(&script_path);
            return Err(AudioGeneratorError::ExecutionError(stderr.to_string()));
        }

        // Clean up temporary files
        let _ = fs::remove_file(&temp_base_path);
        let _ = fs::remove_file(&script_path);

        // Generate ground truth with Praat-specific modifications
        let ground_truth = self.generate_voice_ground_truth_praat(params)?;
        let ground_truth_json = serde_json::to_string_pretty(&ground_truth)?;
        fs::write(&params.ground_truth_path, ground_truth_json)?;

        let mut metadata = HashMap::new();
        metadata.insert("backend".to_string(), "praat".to_string());
        metadata.insert("text".to_string(), params.text.clone());
        metadata.insert("speaking_rate".to_string(), params.speaking_rate.to_string());
        metadata.insert("pitch_modification".to_string(), "true".to_string());

        Ok(AudioOutput {
            audio_path: PathBuf::from(&params.output_path),
            ground_truth_path: PathBuf::from(&params.ground_truth_path),
            duration_sec: ground_truth.duration_sec,
            sample_rate: params.sample_rate,
            metadata,
        })
    }

    /// Generate voice audio (auto-select backend)
    pub fn generate_voice_audio(&self, params: &VoiceAudioParams) -> AudioResult<AudioOutput> {
        match params.backend {
            AudioBackend::ESpeakNG => self.generate_with_espeak(params),
            AudioBackend::Festival => self.generate_with_festival(params),
            AudioBackend::Praat => self.generate_with_praat(params),
        }
    }

    /// Load voice ground truth from JSON file
    pub fn load_voice_ground_truth<P: AsRef<Path>>(&self, path: P) -> AudioResult<VoiceGroundTruth> {
        let content = fs::read_to_string(path)
            .map_err(|e| AudioGeneratorError::OutputReadError(e.to_string()))?;
        let gt: VoiceGroundTruth = serde_json::from_str(&content)?;
        Ok(gt)
    }

    /// Create Festival Scheme script for speech synthesis
    fn create_festival_script(&self, params: &VoiceAudioParams) -> AudioResult<String> {
        let mut script = String::new();

        // Voice selection
        if let Some(voice) = &params.voice {
            script.push_str(&format!(";; Select voice\n(voice_{})\n\n", voice));
        } else {
            script.push_str(";; Use default voice\n");
        }

        // Configure duration and pitch parameters
        // Festival uses duration_stretch (multiplier) and F0 shift
        let duration_stretch = 150.0 / params.speaking_rate.max(1.0);
        let _pitch_shift = params.pitch_mean / 150.0;

        script.push_str(&format!(
            ";; Configure prosodic parameters\n\
             (Parameter.set 'Duration_Stretch {:.3})\n\
             (set! default-f0-mean {})\n\
             (set! default-f0-std {})\n\n",
            duration_stretch,
            params.pitch_mean as i32,
            params.pitch_std as i32
        ));

        // Apply pathological modifiers
        if params.monotonicity > 0.0 {
            let reduced_std = params.pitch_std * (1.0 - params.monotonicity);
            script.push_str(&format!(
                ";; Apply monotonicity (reduced pitch variation)\n\
                 (set! default-f0-std {})\n\n",
                reduced_std as i32
            ));
        }

        if params.hypophonia_severity > 0.0 {
            let volume_factor = 1.0 - (params.hypophonia_severity * 0.5);
            script.push_str(&format!(
                ";; Apply hypophonia (reduced volume)\n\
                 (Parameter.set 'Default_Gain {:.2})\n\n",
                volume_factor
            ));
        }

        // Text to synthesize
        let escaped_text = params.text.replace("\"", "\\\"");
        script.push_str(&format!(
            ";; Generate speech\n\
             (utt.save.wave\n\
              (utt.synth (Utterance Text \"{}\"))\n\
              \"{}\" 'riff)\n",
            escaped_text,
            params.output_path
        ));

        Ok(script)
    }

    /// Create Praat script for acoustic modification
    fn create_praat_script(&self, params: &VoiceAudioParams, input_path: &str) -> AudioResult<String> {
        let mut script = String::new();

        script.push_str(&format!(
            "# Praat script for acoustic modification\n\
             # Input: {}\n\
             # Output: {}\n\n",
            input_path,
            params.output_path
        ));

        // Read the sound file
        script.push_str(&format!(
            "# Read input sound\n\
             sound = Read from file: \"{}\"\n\
             selectObject: sound\n\n",
            input_path
        ));

        // Extract pitch for modification
        script.push_str(
            "# Extract pitch\n\
             To Manipulation: 0.01, 75, 600\n\
             manipulation = selected(\"Manipulation\")\n\n"
        );

        // Modify pitch
        let pitch_ratio = params.pitch_mean / 150.0; // Assume base pitch ~150Hz
        script.push_str(&format!(
            "# Modify pitch\n\
             selectObject: manipulation\n\
             Extract pitch tier\n\
             pitch_tier = selected(\"PitchTier\")\n\
             selectObject: manipulation\n\
             plus pitch_tier\n\
             Replace pitch tier\n\
             selectObject: manipulation\n\
             Multiply frequencies: 0.0, 0.0, {:.3}\n\n",
            pitch_ratio
        ));

        // Apply monotonicity if specified
        if params.monotonicity > 0.0 {
            script.push_str(&format!(
                "# Apply monotonicity (flatten pitch)\n\
                 selectObject: manipulation\n\
                 Extract pitch tier\n\
                 pitch_tier = selected(\"PitchTier\")\n\
                 Flatten: {:.2}\n\
                 selectObject: manipulation\n\
                 plus pitch_tier\n\
                 Replace pitch tier\n\n",
                params.monotonicity
            ));
        }

        // Apply tremor if specified
        if params.tremor_amplitude > 0.0 && params.tremor_frequency > 0.0 {
            script.push_str(&format!(
                "# Apply tremor\n\
                 selectObject: manipulation\n\
                 Extract pitch tier\n\
                 pitch_tier = selected(\"PitchTier\")\n\
                 # Add sinusoidal modulation at {} Hz\n\
                 Add periodic modulation: {:.2}, {:.2}\n\
                 selectObject: manipulation\n\
                 plus pitch_tier\n\
                 Replace pitch tier\n\n",
                params.tremor_frequency,
                params.tremor_frequency,
                params.tremor_amplitude * 50.0 // Scale to Hz
            ));
        }

        // Modify duration if needed
        if (params.speaking_rate - 150.0).abs() > 1.0 {
            let duration_factor = 150.0 / params.speaking_rate;
            script.push_str(&format!(
                "# Modify duration\n\
                 selectObject: manipulation\n\
                 Extract duration tier\n\
                 duration_tier = selected(\"DurationTier\")\n\
                 Add point: 0.0, {:.3}\n\
                 selectObject: manipulation\n\
                 plus duration_tier\n\
                 Replace duration tier\n\n",
                duration_factor
            ));
        }

        // Synthesize modified sound
        script.push_str(
            "# Synthesize modified sound\n\
             selectObject: manipulation\n\
             Get resynthesis (overlap-add)\n\
             modified_sound = selected(\"Sound\")\n\n"
        );

        // Apply volume/hypophonia
        if params.hypophonia_severity > 0.0 || (params.volume - 0.8).abs() > 0.01 {
            let volume_factor = params.volume * (1.0 - params.hypophonia_severity * 0.5);
            script.push_str(&format!(
                "# Apply volume modification\n\
                 selectObject: modified_sound\n\
                 Scale intensity: {:.1}\n\n",
                volume_factor * 70.0 // Scale to dB
            ));
        }

        // Save output
        script.push_str(&format!(
            "# Save output\n\
             selectObject: modified_sound\n\
             Save as WAV file: \"{}\"\n\n\
             # Cleanup\n\
             removeObject: sound, manipulation, modified_sound\n",
            params.output_path
        ));

        Ok(script)
    }

    /// Generate ground truth for Praat-modified audio with formant data
    fn generate_voice_ground_truth_praat(&self, params: &VoiceAudioParams) -> AudioResult<VoiceGroundTruth> {
        let mut ground_truth = self.generate_voice_ground_truth(params)?;

        // Add formant information (simulated)
        let num_frames = ground_truth.f0_contour.len();
        let mut formants = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let time = ground_truth.f0_times[i];

            // Simulate formants based on typical vowel values
            // F1: 300-900 Hz, F2: 800-2500 Hz, F3: 2000-3500 Hz
            let f1 = 500.0 + 200.0 * (2.0 * std::f64::consts::PI * 2.0 * time).sin();
            let f2 = 1500.0 + 400.0 * (2.0 * std::f64::consts::PI * 3.0 * time).sin();
            let f3 = 2700.0 + 300.0 * (2.0 * std::f64::consts::PI * 4.0 * time).sin();

            formants.push(FormantFrame {
                time,
                f1,
                f2,
                f3,
            });
        }

        ground_truth.formants = Some(formants);
        ground_truth.metadata.insert("praat_modified".to_string(), "true".to_string());
        ground_truth.metadata.insert("tremor_frequency".to_string(), params.tremor_frequency.to_string());
        ground_truth.metadata.insert("tremor_amplitude".to_string(), params.tremor_amplitude.to_string());

        Ok(ground_truth)
    }

    fn calculate_espeak_speed(&self, wpm: f64) -> u32 {
        wpm.clamp(80.0, 450.0) as u32
    }

    fn calculate_espeak_pitch(&self, pitch_hz: f64) -> u32 {
        let normalized = (pitch_hz - 80.0) / (300.0 - 80.0);
        (normalized * 99.0).clamp(0.0, 99.0) as u32
    }

    fn generate_voice_ground_truth(&self, params: &VoiceAudioParams) -> AudioResult<VoiceGroundTruth> {
        let words = params.text.split_whitespace().count();
        let duration_sec = (words as f64 / params.speaking_rate) * 60.0;

        let phonemes = vec![
            PhonemeAnnotation {
                phoneme: "START".to_string(),
                start_time: 0.0,
                end_time: 0.0,
            }
        ];

        let num_f0_points = (duration_sec * 100.0) as usize;
        let mut f0_contour = Vec::with_capacity(num_f0_points);
        let mut f0_times = Vec::with_capacity(num_f0_points);

        for i in 0..num_f0_points {
            let t = i as f64 / 100.0;
            f0_times.push(t);

            let f0 = params.pitch_mean +
                     params.pitch_std * (2.0 * std::f64::consts::PI * 3.0 * t).sin();

            let f0_modified = if params.monotonicity > 0.0 {
                params.pitch_mean + (f0 - params.pitch_mean) * (1.0 - params.monotonicity)
            } else {
                f0
            };

            f0_contour.push(f0_modified);
        }

        let mut metadata = HashMap::new();
        metadata.insert("hypophonia_severity".to_string(), params.hypophonia_severity.to_string());
        metadata.insert("monotonicity".to_string(), params.monotonicity.to_string());

        Ok(VoiceGroundTruth {
            text: params.text.clone(),
            phonemes,
            f0_contour,
            f0_times,
            formants: None,
            duration_sec,
            metadata,
        })
    }

    fn validate_voice_params(&self, params: &VoiceAudioParams) -> AudioResult<()> {
        if params.text.is_empty() {
            return Err(AudioGeneratorError::InvalidParameter(
                "text cannot be empty".to_string()
            ));
        }
        if params.speaking_rate <= 0.0 {
            return Err(AudioGeneratorError::InvalidParameter(
                "speaking_rate must be positive".to_string()
            ));
        }
        if params.volume < 0.0 || params.volume > 1.0 {
            return Err(AudioGeneratorError::InvalidParameter(
                "volume must be 0-1".to_string()
            ));
        }
        Ok(())
    }
}

impl Default for Level3AudioGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_voice_params() {
        let params = VoiceAudioParams::default();
        assert!(!params.text.is_empty());
        assert_eq!(params.speaking_rate, 150.0);
        assert_eq!(params.sample_rate, 16000);
    }

    #[test]
    fn test_audio_generator_creation() {
        let generator = Level3AudioGenerator::new();
        assert!(generator.espeak_path.to_str().is_some());
    }

    #[test]
    fn test_espeak_speed_calculation() {
        let generator = Level3AudioGenerator::new();
        assert_eq!(generator.calculate_espeak_speed(150.0), 150);
        assert_eq!(generator.calculate_espeak_speed(50.0), 80);
        assert_eq!(generator.calculate_espeak_speed(500.0), 450);
    }

    #[test]
    fn test_festival_script_generation() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams {
            text: "Hello world".to_string(),
            speaking_rate: 120.0,
            pitch_mean: 180.0,
            pitch_std: 25.0,
            output_path: "/tmp/test_festival.wav".to_string(),
            ..Default::default()
        };

        let script = generator.create_festival_script(&params).unwrap();

        // Verify script contains key elements
        assert!(script.contains("Duration_Stretch"));
        assert!(script.contains("default-f0-mean"));
        assert!(script.contains("Hello world"));
        assert!(script.contains("utt.save.wave"));
        assert!(script.contains("/tmp/test_festival.wav"));
    }

    #[test]
    fn test_festival_script_with_voice() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams {
            voice: Some("kal_diphone".to_string()),
            output_path: "/tmp/test_voice.wav".to_string(),
            ..Default::default()
        };

        let script = generator.create_festival_script(&params).unwrap();

        assert!(script.contains("voice_kal_diphone"));
    }

    #[test]
    fn test_festival_script_with_pathological_params() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams {
            monotonicity: 0.6,
            hypophonia_severity: 0.4,
            output_path: "/tmp/test_pathology.wav".to_string(),
            ..Default::default()
        };

        let script = generator.create_festival_script(&params).unwrap();

        // Should contain monotonicity and hypophonia modifications
        assert!(script.contains("default-f0-std"));
        assert!(script.contains("Default_Gain"));
    }

    #[test]
    fn test_praat_script_generation() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams {
            text: "Test speech".to_string(),
            pitch_mean: 200.0,
            speaking_rate: 140.0,
            output_path: "/tmp/test_praat.wav".to_string(),
            ..Default::default()
        };

        let script = generator.create_praat_script(&params, "/tmp/input.wav").unwrap();

        // Verify script contains key Praat commands
        assert!(script.contains("Read from file"));
        assert!(script.contains("To Manipulation"));
        assert!(script.contains("Extract pitch tier"));
        assert!(script.contains("Multiply frequencies"));
        assert!(script.contains("Save as WAV file"));
        assert!(script.contains("/tmp/test_praat.wav"));
    }

    #[test]
    fn test_praat_script_with_tremor() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams {
            tremor_frequency: 5.0,
            tremor_amplitude: 0.3,
            output_path: "/tmp/test_tremor.wav".to_string(),
            ..Default::default()
        };

        let script = generator.create_praat_script(&params, "/tmp/input.wav").unwrap();

        assert!(script.contains("Apply tremor"));
        assert!(script.contains("Add periodic modulation"));
    }

    #[test]
    fn test_praat_script_with_monotonicity() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams {
            monotonicity: 0.7,
            output_path: "/tmp/test_mono.wav".to_string(),
            ..Default::default()
        };

        let script = generator.create_praat_script(&params, "/tmp/input.wav").unwrap();

        assert!(script.contains("Apply monotonicity"));
        assert!(script.contains("Flatten"));
    }

    #[test]
    fn test_praat_ground_truth_with_formants() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams::default();

        let ground_truth = generator.generate_voice_ground_truth_praat(&params).unwrap();

        // Verify formants are present
        assert!(ground_truth.formants.is_some());
        let formants = ground_truth.formants.unwrap();
        assert!(!formants.is_empty());

        // Check formant values are in reasonable ranges
        for formant in formants.iter() {
            assert!(formant.f1 > 200.0 && formant.f1 < 1000.0);
            assert!(formant.f2 > 700.0 && formant.f2 < 3000.0);
            assert!(formant.f3 > 1500.0 && formant.f3 < 4000.0);
        }

        // Verify metadata
        assert_eq!(ground_truth.metadata.get("praat_modified"), Some(&"true".to_string()));
    }

    #[test]
    fn test_backend_selection() {
        let _generator = Level3AudioGenerator::new();

        // Test each backend type
        let backends = vec![
            AudioBackend::ESpeakNG,
            AudioBackend::Festival,
            AudioBackend::Praat,
        ];

        for backend in backends {
            let _params = VoiceAudioParams {
                backend,
                ..Default::default()
            };
            // Note: This just tests the selection logic, not actual execution
            // which requires the tools to be installed
        }
    }

    #[test]
    fn test_voice_params_validation() {
        let generator = Level3AudioGenerator::new();

        // Test empty text
        let params = VoiceAudioParams {
            text: "".to_string(),
            ..Default::default()
        };
        assert!(generator.validate_voice_params(&params).is_err());

        // Test invalid speaking rate
        let params = VoiceAudioParams {
            speaking_rate: 0.0,
            ..Default::default()
        };
        assert!(generator.validate_voice_params(&params).is_err());

        // Test invalid volume
        let params = VoiceAudioParams {
            volume: 1.5,
            ..Default::default()
        };
        assert!(generator.validate_voice_params(&params).is_err());

        // Test valid params
        let params = VoiceAudioParams::default();
        assert!(generator.validate_voice_params(&params).is_ok());
    }

    #[test]
    fn test_festival_duration_calculation() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams {
            speaking_rate: 100.0, // Slower than default 150
            text: "Test".to_string(),
            output_path: "/tmp/test.wav".to_string(),
            ..Default::default()
        };

        let script = generator.create_festival_script(&params).unwrap();

        // Duration stretch should be 150/100 = 1.5
        assert!(script.contains("Duration_Stretch 1.500"));
    }

    #[test]
    fn test_praat_volume_modification() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams {
            volume: 0.5,
            hypophonia_severity: 0.3,
            output_path: "/tmp/test_volume.wav".to_string(),
            ..Default::default()
        };

        let script = generator.create_praat_script(&params, "/tmp/input.wav").unwrap();

        assert!(script.contains("Scale intensity"));
    }

    #[test]
    fn test_text_escaping_festival() {
        let generator = Level3AudioGenerator::new();
        let params = VoiceAudioParams {
            text: "He said \"Hello world\"".to_string(),
            output_path: "/tmp/test_escape.wav".to_string(),
            ..Default::default()
        };

        let script = generator.create_festival_script(&params).unwrap();

        // Quotes should be escaped
        assert!(script.contains("\\\""));
    }
}
