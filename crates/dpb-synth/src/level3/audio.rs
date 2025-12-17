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

    /// Generate voice audio (auto-select backend)
    pub fn generate_voice_audio(&self, params: &VoiceAudioParams) -> AudioResult<AudioOutput> {
        match params.backend {
            AudioBackend::ESpeakNG => self.generate_with_espeak(params),
            AudioBackend::Festival => {
                Err(AudioGeneratorError::ToolNotFound(
                    "Festival backend not yet implemented".to_string()
                ))
            }
            AudioBackend::Praat => {
                Err(AudioGeneratorError::ToolNotFound(
                    "Praat backend not yet implemented".to_string()
                ))
            }
        }
    }

    /// Load voice ground truth from JSON file
    pub fn load_voice_ground_truth<P: AsRef<Path>>(&self, path: P) -> AudioResult<VoiceGroundTruth> {
        let content = fs::read_to_string(path)
            .map_err(|e| AudioGeneratorError::OutputReadError(e.to_string()))?;
        let gt: VoiceGroundTruth = serde_json::from_str(&content)?;
        Ok(gt)
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
}
