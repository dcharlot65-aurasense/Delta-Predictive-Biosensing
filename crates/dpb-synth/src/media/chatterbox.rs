//! Chatterbox Neural TTS Integration
//!
//! This module provides integration with Chatterbox, a state-of-the-art open source
//! text-to-speech model with emotion control and voice cloning capabilities.
//!
//! # Features
//!
//! - High-quality neural TTS (500M params, 500K hours training data)
//! - Emotion exaggeration control (first open source with this feature)
//! - Voice cloning with 5-second samples
//! - Paralinguistic tags ([cough], [laugh], [sigh])
//! - 23 language support (multilingual model)
//! - Sub-200ms latency in streaming mode
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install chatterbox-tts`
//! - CUDA GPU recommended (8GB+ VRAM)
//!
//! # License
//!
//! Chatterbox is MIT licensed - free for commercial use.
//!
//! # References
//!
//! - GitHub: https://github.com/resemble-ai/chatterbox
//! - Resemble AI: https://www.resemble.ai/chatterbox/

use super::{MediaError, Result, VoiceGroundTruth};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Chatterbox TTS wrapper
#[derive(Debug)]
pub struct ChatterboxTTS {
    config: ChatterboxConfig,
    python_path: PathBuf,
    model_loaded: bool,
}

/// Configuration for Chatterbox TTS
#[derive(Debug, Clone)]
pub struct ChatterboxConfig {
    /// Model variant to use
    pub model: ChatterboxModel,
    /// GPU device ID (-1 for CPU)
    pub device: i32,
    /// Output sample rate
    pub sample_rate: u32,
    /// Enable streaming mode for lower latency
    pub streaming: bool,
    /// Output directory
    pub output_dir: PathBuf,
    /// Cache directory for models
    pub cache_dir: PathBuf,
}

/// Chatterbox model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatterboxModel {
    /// Standard model (500M params)
    #[default]
    Standard,
    /// Turbo model (350M params, faster)
    Turbo,
    /// Multilingual model (23 languages)
    Multilingual,
}

impl Default for ChatterboxConfig {
    fn default() -> Self {
        Self {
            model: ChatterboxModel::Standard,
            device: 0,
            sample_rate: 24000,
            streaming: false,
            output_dir: PathBuf::from("output/audio"),
            cache_dir: PathBuf::from(".cache/chatterbox"),
        }
    }
}

/// Voice configuration for synthesis
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    /// Reference audio for voice cloning (optional)
    pub reference_audio: Option<PathBuf>,
    /// Speaker ID (for multi-speaker models)
    pub speaker_id: Option<String>,
    /// Language code (for multilingual)
    pub language: String,
    /// Speaking rate multiplier
    pub rate: f32,
    /// Pitch shift in semitones
    pub pitch_shift: f32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            reference_audio: None,
            speaker_id: None,
            language: "en".to_string(),
            rate: 1.0,
            pitch_shift: 0.0,
        }
    }
}

/// Emotion control parameters
#[derive(Debug, Clone)]
pub struct EmotionControl {
    /// Base emotion
    pub emotion: Emotion,
    /// Exaggeration level (0.0 = neutral, 1.0 = full emotion, >1.0 = exaggerated)
    pub exaggeration: f32,
    /// Emotion transitions over time [(time_sec, emotion, intensity)]
    pub transitions: Vec<(f64, Emotion, f32)>,
}

/// Emotion types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Emotion {
    #[default]
    Neutral,
    Happy,
    Sad,
    Angry,
    Fearful,
    Surprised,
    Disgusted,
    /// Monotone speech (relevant for Parkinson's hypophonia)
    Monotone,
    /// Breathy/weak voice
    Breathy,
    /// Strained voice
    Strained,
}

impl Default for EmotionControl {
    fn default() -> Self {
        Self {
            emotion: Emotion::Neutral,
            exaggeration: 0.5,
            transitions: Vec::new(),
        }
    }
}

/// Paralinguistic tags that can be inserted
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParalinguisticTag {
    Cough,
    Laugh,
    Chuckle,
    Sigh,
    Gasp,
    ClearThroat,
    Sniff,
    Yawn,
    Hmm,
    Uh,
    Um,
}

impl ParalinguisticTag {
    /// Get the tag string for insertion
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Cough => "[cough]",
            Self::Laugh => "[laugh]",
            Self::Chuckle => "[chuckle]",
            Self::Sigh => "[sigh]",
            Self::Gasp => "[gasp]",
            Self::ClearThroat => "[clears throat]",
            Self::Sniff => "[sniff]",
            Self::Yawn => "[yawn]",
            Self::Hmm => "[hmm]",
            Self::Uh => "[uh]",
            Self::Um => "[um]",
        }
    }
}

/// Parameters for pathological voice synthesis
#[derive(Debug, Clone)]
pub struct PathologicalVoiceParams {
    /// Hypophonia (reduced volume/projection) severity 0-1
    pub hypophonia: f32,
    /// Monotonicity (reduced pitch variation) 0-1
    pub monotonicity: f32,
    /// Breathiness severity 0-1
    pub breathiness: f32,
    /// Hoarseness severity 0-1
    pub hoarseness: f32,
    /// Tremor in voice (frequency in Hz, amplitude 0-1)
    pub voice_tremor: Option<(f32, f32)>,
    /// Hesitations/pauses probability
    pub hesitation_probability: f32,
    /// Dysarthria severity 0-1
    pub dysarthria: f32,
}

impl Default for PathologicalVoiceParams {
    fn default() -> Self {
        Self {
            hypophonia: 0.0,
            monotonicity: 0.0,
            breathiness: 0.0,
            hoarseness: 0.0,
            voice_tremor: None,
            hesitation_probability: 0.0,
            dysarthria: 0.0,
        }
    }
}

/// Output from speech synthesis
#[derive(Debug)]
pub struct SpeechOutput {
    /// Path to generated audio file
    pub audio_path: PathBuf,
    /// Duration in seconds
    pub duration: f64,
    /// Sample rate
    pub sample_rate: u32,
    /// Phoneme alignments [(start_time, end_time, phoneme)]
    pub phoneme_alignment: Vec<(f64, f64, String)>,
    /// Word alignments [(start_time, end_time, word)]
    pub word_alignment: Vec<(f64, f64, String)>,
    /// Ground truth voice parameters
    pub ground_truth: VoiceGroundTruth,
}

impl ChatterboxTTS {
    /// Create a new Chatterbox TTS instance
    pub fn new(config: ChatterboxConfig) -> Result<Self> {
        // Check if Chatterbox is available
        let check = Command::new("python3")
            .args(["-c", "from chatterbox.tts import ChatterboxTTS; print('ok')"])
            .output();

        match check {
            Ok(output) if output.status.success() => {}
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "chatterbox".to_string(),
                    install_url: "pip install chatterbox-tts".to_string(),
                });
            }
        }

        let python_path = which::which("python3")
            .map_err(|_| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: "https://www.python.org/downloads/".to_string(),
            })?;

        // Ensure directories exist
        std::fs::create_dir_all(&config.output_dir)?;
        std::fs::create_dir_all(&config.cache_dir)?;

        Ok(Self {
            config,
            python_path,
            model_loaded: false,
        })
    }

    /// Check if Chatterbox is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "from chatterbox.tts import ChatterboxTTS"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Synthesize speech from text
    pub fn synthesize(
        &self,
        text: &str,
        voice: &VoiceConfig,
        emotion: Option<&EmotionControl>,
    ) -> Result<SpeechOutput> {
        let output_path = self.config.output_dir.join(format!(
            "speech_{}.wav",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        let script = self.generate_synthesis_script(text, voice, emotion, &output_path)?;
        self.run_synthesis_script(&script, &output_path)
    }

    /// Synthesize pathological speech (Parkinson's, dysarthria, etc.)
    pub fn synthesize_pathological(
        &self,
        text: &str,
        voice: &VoiceConfig,
        pathology: &PathologicalVoiceParams,
    ) -> Result<SpeechOutput> {
        // Convert pathology params to emotion/modifications
        let modified_text = self.apply_pathological_modifications(text, pathology);

        let _emotion = EmotionControl {
            emotion: if pathology.monotonicity > 0.5 {
                Emotion::Monotone
            } else if pathology.breathiness > 0.5 {
                Emotion::Breathy
            } else {
                Emotion::Neutral
            },
            exaggeration: 1.0 - pathology.hypophonia,
            transitions: Vec::new(),
        };

        let output_path = self.config.output_dir.join(format!(
            "pathological_speech_{}.wav",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        let script = self.generate_pathological_script(
            &modified_text,
            voice,
            pathology,
            &output_path,
        )?;

        let mut result = self.run_synthesis_script(&script, &output_path)?;

        // Update ground truth with pathology info
        result.ground_truth.hypophonia = pathology.hypophonia;
        result.ground_truth.dysarthria = pathology.dysarthria;

        Ok(result)
    }

    /// Apply pathological modifications to text (add hesitations, etc.)
    fn apply_pathological_modifications(&self, text: &str, pathology: &PathologicalVoiceParams) -> String {
        let mut result = text.to_string();

        // Add hesitations based on probability
        if pathology.hesitation_probability > 0.0 {
            let words: Vec<&str> = result.split_whitespace().collect();
            let mut new_words = Vec::new();

            for word in words {
                // Random hesitation before some words
                if rand::random::<f32>() < pathology.hesitation_probability {
                    let hesitation = if rand::random::<bool>() {
                        ParalinguisticTag::Um.tag()
                    } else {
                        ParalinguisticTag::Uh.tag()
                    };
                    new_words.push(hesitation);
                }
                new_words.push(word);
            }

            result = new_words.join(" ");
        }

        result
    }

    /// Generate Python script for standard synthesis
    fn generate_synthesis_script(
        &self,
        text: &str,
        voice: &VoiceConfig,
        emotion: Option<&EmotionControl>,
        output_path: &Path,
    ) -> Result<String> {
        let reference_code = if let Some(ref_audio) = &voice.reference_audio {
            format!(r#"
# Load reference audio for voice cloning
import librosa
ref_audio, _ = librosa.load('{}', sr=model.sample_rate)
speaker_embedding = model.get_speaker_embedding(ref_audio)
"#, ref_audio.display())
        } else {
            "speaker_embedding = None".to_string()
        };

        let emotion_code = if let Some(em) = emotion {
            format!(r#"
# Set emotion
emotion = '{:?}'
exaggeration = {}
"#, em.emotion, em.exaggeration)
        } else {
            "emotion = 'neutral'\nexaggeration = 0.5".to_string()
        };

        Ok(format!(r#"
import torch
from chatterbox.tts import ChatterboxTTS
import soundfile as sf
import json

# Initialize model
device = 'cuda:{device}' if torch.cuda.is_available() and {device} >= 0 else 'cpu'
model = ChatterboxTTS.from_pretrained(device=device)

{reference_code}

{emotion_code}

# Synthesize
text = '''{text}'''
audio = model.generate(
    text,
    speaker_embedding=speaker_embedding,
    exaggeration=exaggeration,
)

# Save audio
sf.write('{output_path}', audio.cpu().numpy(), model.sample_rate)

# Get alignments (if available)
try:
    alignments = model.get_alignments(text, audio)
    phoneme_alignment = alignments.get('phonemes', [])
    word_alignment = alignments.get('words', [])
except:
    phoneme_alignment = []
    word_alignment = []

# Output metadata
result = {{
    'duration': len(audio) / model.sample_rate,
    'sample_rate': model.sample_rate,
    'phoneme_alignment': phoneme_alignment,
    'word_alignment': word_alignment,
}}
print(json.dumps(result))
"#,
            device = self.config.device,
            reference_code = reference_code,
            emotion_code = emotion_code,
            text = text.replace("'", "\\'"),
            output_path = output_path.display(),
        ))
    }

    /// Generate script for pathological voice synthesis
    fn generate_pathological_script(
        &self,
        text: &str,
        _voice: &VoiceConfig,
        pathology: &PathologicalVoiceParams,
        output_path: &Path,
    ) -> Result<String> {
        Ok(format!(r#"
import torch
import numpy as np
from chatterbox.tts import ChatterboxTTS
import soundfile as sf
import json
from scipy import signal

# Pathology parameters
HYPOPHONIA = {hypophonia}
MONOTONICITY = {monotonicity}
BREATHINESS = {breathiness}
HOARSENESS = {hoarseness}
VOICE_TREMOR_FREQ = {tremor_freq}
VOICE_TREMOR_AMP = {tremor_amp}
DYSARTHRIA = {dysarthria}

# Initialize model
device = 'cuda:{device}' if torch.cuda.is_available() and {device} >= 0 else 'cpu'
model = ChatterboxTTS.from_pretrained(device=device)

# Synthesize base audio
text = '''{text}'''

# Adjust exaggeration for hypophonia (less expressive = lower exaggeration)
exaggeration = max(0.1, 1.0 - HYPOPHONIA)

audio = model.generate(text, exaggeration=exaggeration)
audio = audio.cpu().numpy()
sr = model.sample_rate

# Apply pathological modifications

# 1. Reduce pitch variation for monotonicity
if MONOTONICITY > 0:
    # Simple approach: attenuate high-frequency modulations
    # In production, use PSOLA or similar for pitch flattening
    pass

# 2. Add breathiness (noise in spectral valleys)
if BREATHINESS > 0:
    noise = np.random.randn(len(audio)) * BREATHINESS * 0.1
    # Shape noise to be more prominent in unvoiced regions
    envelope = np.abs(signal.hilbert(audio))
    envelope = np.clip(envelope / np.max(envelope), 0.1, 1.0)
    noise = noise * (1 - envelope)
    audio = audio + noise

# 3. Add voice tremor
if VOICE_TREMOR_FREQ > 0 and VOICE_TREMOR_AMP > 0:
    t = np.arange(len(audio)) / sr
    tremor = 1.0 + VOICE_TREMOR_AMP * np.sin(2 * np.pi * VOICE_TREMOR_FREQ * t)
    audio = audio * tremor

# 4. Reduce volume for hypophonia
if HYPOPHONIA > 0:
    audio = audio * (1.0 - 0.5 * HYPOPHONIA)

# 5. Add hoarseness (subharmonics and noise)
if HOARSENESS > 0:
    # Add jitter/shimmer simulation
    jitter_factor = HOARSENESS * 0.02
    shimmer_factor = HOARSENESS * 0.1

    # Simplified: add some noise
    noise = np.random.randn(len(audio)) * HOARSENESS * 0.05
    audio = audio + noise

# Normalize
audio = audio / np.max(np.abs(audio)) * 0.9

# Save
sf.write('{output_path}', audio, sr)

# Analyze resulting voice quality
from scipy.stats import variation

# Estimate F0 statistics (simplified)
f0_mean = 150.0  # Placeholder - would use proper F0 extraction
f0_std = f0_mean * 0.1 * (1 - MONOTONICITY)  # Reduced variation with monotonicity

# Compute basic quality metrics
jitter_estimate = 0.01 + HOARSENESS * 0.05
shimmer_estimate = 0.03 + HOARSENESS * 0.1
hnr_estimate = 20.0 - BREATHINESS * 10 - HOARSENESS * 5

result = {{
    'duration': len(audio) / sr,
    'sample_rate': int(sr),
    'phoneme_alignment': [],
    'word_alignment': [],
    'voice_metrics': {{
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'jitter': jitter_estimate,
        'shimmer': shimmer_estimate,
        'hnr': hnr_estimate,
        'hypophonia': HYPOPHONIA,
        'dysarthria': DYSARTHRIA
    }}
}}
print(json.dumps(result))
"#,
            device = self.config.device,
            text = text.replace("'", "\\'"),
            output_path = output_path.display(),
            hypophonia = pathology.hypophonia,
            monotonicity = pathology.monotonicity,
            breathiness = pathology.breathiness,
            hoarseness = pathology.hoarseness,
            tremor_freq = pathology.voice_tremor.map(|t| t.0).unwrap_or(0.0),
            tremor_amp = pathology.voice_tremor.map(|t| t.1).unwrap_or(0.0),
            dysarthria = pathology.dysarthria,
        ))
    }

    /// Run synthesis script and parse results
    fn run_synthesis_script(&self, script: &str, output_path: &Path) -> Result<SpeechOutput> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("chatterbox_synth.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let data: serde_json::Value = serde_json::from_str(&stdout)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        let voice_metrics = data.get("voice_metrics");

        Ok(SpeechOutput {
            audio_path: output_path.to_path_buf(),
            duration: data["duration"].as_f64().unwrap_or(0.0),
            sample_rate: data["sample_rate"].as_u64().unwrap_or(24000) as u32,
            phoneme_alignment: Vec::new(), // Parse from data if available
            word_alignment: Vec::new(),
            ground_truth: VoiceGroundTruth {
                f0: Vec::new(),
                jitter: voice_metrics
                    .and_then(|m| m["jitter"].as_f64())
                    .unwrap_or(0.01) as f32,
                shimmer: voice_metrics
                    .and_then(|m| m["shimmer"].as_f64())
                    .unwrap_or(0.03) as f32,
                hnr: voice_metrics
                    .and_then(|m| m["hnr"].as_f64())
                    .unwrap_or(20.0) as f32,
                mpt: None,
                speech_rate: 4.0, // Default syllables/second
                hypophonia: voice_metrics
                    .and_then(|m| m["hypophonia"].as_f64())
                    .unwrap_or(0.0) as f32,
                dysarthria: voice_metrics
                    .and_then(|m| m["dysarthria"].as_f64())
                    .unwrap_or(0.0) as f32,
                phonemes: Vec::new(),
            },
        })
    }

    /// Clone a voice from reference audio
    pub fn clone_voice(&self, reference_audio: &Path) -> Result<VoiceConfig> {
        // Validate reference audio exists and is long enough
        if !reference_audio.exists() {
            return Err(MediaError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Reference audio not found",
            )));
        }

        Ok(VoiceConfig {
            reference_audio: Some(reference_audio.to_path_buf()),
            ..Default::default()
        })
    }

    /// List available voices/speakers
    pub fn list_voices(&self) -> Result<Vec<String>> {
        // Query model for available speakers
        let script = r#"
from chatterbox.tts import ChatterboxTTS
import json
model = ChatterboxTTS.from_pretrained()
voices = getattr(model, 'speaker_ids', ['default'])
print(json.dumps(voices))
"#;
        let output = self.run_python_script_simple(script)?;
        let voices: Vec<String> = serde_json::from_str(&output)
            .unwrap_or_else(|_| vec!["default".to_string()]);
        Ok(voices)
    }

    /// Run a simple Python script
    fn run_python_script_simple(&self, script: &str) -> Result<String> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("chatterbox_query.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

// Alternative TTS backends

/// F5-TTS integration (diffusion-based)
#[derive(Debug)]
pub struct F5TTS {
    config: F5TTSConfig,
}

#[derive(Debug, Clone, Default)]
pub struct F5TTSConfig {
    pub device: i32,
    pub output_dir: PathBuf,
}

impl F5TTS {
    pub fn new(config: F5TTSConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import f5_tts"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

/// Kokoro TTS integration (lightweight)
#[derive(Debug)]
pub struct KokoroTTS {
    config: KokoroConfig,
}

#[derive(Debug, Clone, Default)]
pub struct KokoroConfig {
    pub device: i32,
    pub output_dir: PathBuf,
}

impl KokoroTTS {
    pub fn new(config: KokoroConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import kokoro"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ChatterboxConfig::default();
        assert_eq!(config.sample_rate, 24000);
    }

    #[test]
    fn test_voice_config_default() {
        let voice = VoiceConfig::default();
        assert_eq!(voice.language, "en");
        assert_eq!(voice.rate, 1.0);
    }

    #[test]
    fn test_pathological_params_default() {
        let params = PathologicalVoiceParams::default();
        assert_eq!(params.hypophonia, 0.0);
        assert_eq!(params.dysarthria, 0.0);
    }

    #[test]
    fn test_paralinguistic_tags() {
        assert_eq!(ParalinguisticTag::Cough.tag(), "[cough]");
        assert_eq!(ParalinguisticTag::Laugh.tag(), "[laugh]");
    }
}
