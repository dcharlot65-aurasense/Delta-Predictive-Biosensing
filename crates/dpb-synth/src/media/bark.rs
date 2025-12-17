//! Bark Audio Generation Integration
//!
//! Bark is a transformer-based text-to-audio model from Suno.ai that can
//! generate speech, music, and sound effects.
//!
//! # Features
//!
//! - Text-to-speech with various speakers
//! - Non-speech sounds (laughter, sighing, coughing)
//! - Music generation
//! - Environmental sounds
//! - MIT License
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install bark`
//! - GPU with 8GB+ VRAM recommended

use super::{MediaError, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Bark audio generator
#[derive(Debug)]
pub struct BarkGenerator {
    config: BarkConfig,
    python_path: PathBuf,
}

/// Configuration for Bark
#[derive(Debug, Clone)]
pub struct BarkConfig {
    /// Output directory
    pub output_dir: PathBuf,
    /// Sample rate (default 24000)
    pub sample_rate: u32,
    /// Use small model (faster, less quality)
    pub use_small: bool,
    /// GPU device ID (-1 for CPU)
    pub device: i32,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for BarkConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("output/bark"),
            sample_rate: 24000,
            use_small: false,
            device: 0,
            seed: None,
        }
    }
}

/// Voice presets available in Bark
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarkSpeaker {
    /// English speakers (v2)
    EnglishSpeaker0,
    EnglishSpeaker1,
    EnglishSpeaker2,
    EnglishSpeaker3,
    EnglishSpeaker4,
    EnglishSpeaker5,
    EnglishSpeaker6,
    EnglishSpeaker7,
    EnglishSpeaker8,
    EnglishSpeaker9,
    /// Chinese speaker
    ChineseSpeaker0,
    /// French speaker
    FrenchSpeaker0,
    /// German speaker
    GermanSpeaker0,
    /// Hindi speaker
    HindiSpeaker0,
    /// Italian speaker
    ItalianSpeaker0,
    /// Japanese speaker
    JapaneseSpeaker0,
    /// Korean speaker
    KoreanSpeaker0,
    /// Polish speaker
    PolishSpeaker0,
    /// Portuguese speaker
    PortugueseSpeaker0,
    /// Russian speaker
    RussianSpeaker0,
    /// Spanish speaker
    SpanishSpeaker0,
    /// Turkish speaker
    TurkishSpeaker0,
}

impl BarkSpeaker {
    fn preset_name(&self) -> &'static str {
        match self {
            BarkSpeaker::EnglishSpeaker0 => "v2/en_speaker_0",
            BarkSpeaker::EnglishSpeaker1 => "v2/en_speaker_1",
            BarkSpeaker::EnglishSpeaker2 => "v2/en_speaker_2",
            BarkSpeaker::EnglishSpeaker3 => "v2/en_speaker_3",
            BarkSpeaker::EnglishSpeaker4 => "v2/en_speaker_4",
            BarkSpeaker::EnglishSpeaker5 => "v2/en_speaker_5",
            BarkSpeaker::EnglishSpeaker6 => "v2/en_speaker_6",
            BarkSpeaker::EnglishSpeaker7 => "v2/en_speaker_7",
            BarkSpeaker::EnglishSpeaker8 => "v2/en_speaker_8",
            BarkSpeaker::EnglishSpeaker9 => "v2/en_speaker_9",
            BarkSpeaker::ChineseSpeaker0 => "v2/zh_speaker_0",
            BarkSpeaker::FrenchSpeaker0 => "v2/fr_speaker_0",
            BarkSpeaker::GermanSpeaker0 => "v2/de_speaker_0",
            BarkSpeaker::HindiSpeaker0 => "v2/hi_speaker_0",
            BarkSpeaker::ItalianSpeaker0 => "v2/it_speaker_0",
            BarkSpeaker::JapaneseSpeaker0 => "v2/ja_speaker_0",
            BarkSpeaker::KoreanSpeaker0 => "v2/ko_speaker_0",
            BarkSpeaker::PolishSpeaker0 => "v2/pl_speaker_0",
            BarkSpeaker::PortugueseSpeaker0 => "v2/pt_speaker_0",
            BarkSpeaker::RussianSpeaker0 => "v2/ru_speaker_0",
            BarkSpeaker::SpanishSpeaker0 => "v2/es_speaker_0",
            BarkSpeaker::TurkishSpeaker0 => "v2/tr_speaker_0",
        }
    }
}

/// Special audio tags supported by Bark
#[derive(Debug, Clone)]
pub enum BarkTag {
    /// Laughter [laughs]
    Laughs,
    /// Sighing [sighs]
    Sighs,
    /// Coughing [coughs]
    Coughs,
    /// Clearing throat [clears throat]
    ClearsThroat,
    /// Gasping [gasps]
    Gasps,
    /// Music notation ♪
    Music,
    /// Emphasis with CAPS
    Emphasis(String),
    /// Hesitation with ...
    Hesitation,
    /// Custom tag
    Custom(String),
}

impl BarkTag {
    fn to_text(&self) -> String {
        match self {
            BarkTag::Laughs => "[laughs]".to_string(),
            BarkTag::Sighs => "[sighs]".to_string(),
            BarkTag::Coughs => "[coughs]".to_string(),
            BarkTag::ClearsThroat => "[clears throat]".to_string(),
            BarkTag::Gasps => "[gasps]".to_string(),
            BarkTag::Music => "♪".to_string(),
            BarkTag::Emphasis(text) => text.to_uppercase(),
            BarkTag::Hesitation => "...".to_string(),
            BarkTag::Custom(tag) => format!("[{}]", tag),
        }
    }
}

/// Audio prompt for Bark
#[derive(Debug, Clone)]
pub struct BarkPrompt {
    /// Text content with optional tags
    pub text: String,
    /// Speaker preset
    pub speaker: BarkSpeaker,
    /// Additional tags to insert
    pub tags: Vec<(usize, BarkTag)>,
}

impl BarkPrompt {
    /// Create a simple text prompt
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            speaker: BarkSpeaker::EnglishSpeaker0,
            tags: Vec::new(),
        }
    }

    /// Set speaker
    pub fn with_speaker(mut self, speaker: BarkSpeaker) -> Self {
        self.speaker = speaker;
        self
    }

    /// Add tag at position
    pub fn with_tag(mut self, position: usize, tag: BarkTag) -> Self {
        self.tags.push((position, tag));
        self
    }

    /// Add cough at position
    pub fn with_cough(self, position: usize) -> Self {
        self.with_tag(position, BarkTag::Coughs)
    }

    /// Add laugh at position
    pub fn with_laugh(self, position: usize) -> Self {
        self.with_tag(position, BarkTag::Laughs)
    }

    /// Build final text with tags
    fn build_text(&self) -> String {
        let mut result = self.text.clone();
        let mut tags: Vec<_> = self.tags.clone();
        tags.sort_by(|a, b| b.0.cmp(&a.0)); // Sort descending to insert from end

        for (pos, tag) in tags {
            let pos = pos.min(result.len());
            result.insert_str(pos, &format!(" {} ", tag.to_text()));
        }
        result
    }
}

/// Preset prompts for biosignal scenarios
pub struct BarkBioPrompts;

impl BarkBioPrompts {
    /// Normal speech for voice biomarker baseline
    pub fn normal_speech() -> BarkPrompt {
        BarkPrompt::text(
            "The rainbow is a division of white light into many beautiful colors. \
             These take the shape of a long round arch, with its path high above, \
             and its two ends apparently beyond the horizon."
        ).with_speaker(BarkSpeaker::EnglishSpeaker0)
    }

    /// Speech with coughing (respiratory symptom)
    pub fn speech_with_cough() -> BarkPrompt {
        BarkPrompt::text(
            "The rainbow is a division of white light into many beautiful colors."
        ).with_speaker(BarkSpeaker::EnglishSpeaker0)
        .with_cough(35)
    }

    /// Hesitant speech (cognitive/motor symptom)
    pub fn hesitant_speech() -> BarkPrompt {
        BarkPrompt::text(
            "The rainbow... is a division... of white light... into many... beautiful colors."
        ).with_speaker(BarkSpeaker::EnglishSpeaker0)
    }

    /// Breathy/fatigued speech
    pub fn fatigued_speech() -> BarkPrompt {
        BarkPrompt::text(
            "The rainbow is a division of white light into many beautiful colors."
        ).with_speaker(BarkSpeaker::EnglishSpeaker0)
        .with_tag(0, BarkTag::Sighs)
    }

    /// Sustained vowel "ahhh" for voice analysis
    pub fn sustained_vowel() -> BarkPrompt {
        BarkPrompt::text("Ahhhhhhhhhhhhhhhhh")
            .with_speaker(BarkSpeaker::EnglishSpeaker0)
    }

    /// Counting task
    pub fn counting_task() -> BarkPrompt {
        BarkPrompt::text(
            "One, two, three, four, five, six, seven, eight, nine, ten."
        ).with_speaker(BarkSpeaker::EnglishSpeaker0)
    }
}

/// Output from Bark generation
#[derive(Debug)]
pub struct BarkOutput {
    /// Path to generated audio file
    pub audio_path: PathBuf,
    /// Duration in seconds
    pub duration: f64,
    /// Sample rate
    pub sample_rate: u32,
    /// Generation time in seconds
    pub generation_time: f64,
    /// Speaker used
    pub speaker: String,
    /// Seed used
    pub seed: u64,
}

impl BarkGenerator {
    /// Create a new Bark generator
    pub fn new(config: BarkConfig) -> Result<Self> {
        // Check for bark
        let check = Command::new("python3")
            .args(["-c", "from bark import SAMPLE_RATE, generate_audio, preload_models; print('ok')"])
            .output();

        match check {
            Ok(output) if output.status.success() => {}
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "bark".to_string(),
                    install_url: "pip install bark".to_string(),
                });
            }
        }

        let python_path = which::which("python3")
            .map_err(|_| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: "https://www.python.org/downloads/".to_string(),
            })?;

        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self {
            config,
            python_path,
        })
    }

    /// Check if Bark is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "from bark import generate_audio"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Generate audio from prompt
    pub fn generate(&self, prompt: &BarkPrompt) -> Result<BarkOutput> {
        let output_path = self.config.output_dir.join(format!(
            "bark_{}.wav",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        let script = self.generate_script(prompt, &output_path)?;
        self.run_script(&script, &output_path, prompt.speaker.preset_name())
    }

    /// Generate non-speech audio (sound effects)
    pub fn generate_sound_effect(&self, description: &str) -> Result<BarkOutput> {
        let prompt = BarkPrompt::text(description);
        self.generate(&prompt)
    }

    /// Generate music
    pub fn generate_music(&self, description: &str) -> Result<BarkOutput> {
        let text = format!("♪ {} ♪", description);
        let prompt = BarkPrompt::text(text);
        self.generate(&prompt)
    }

    fn generate_script(&self, prompt: &BarkPrompt, output_path: &Path) -> Result<String> {
        let text = prompt.build_text();
        let speaker = prompt.speaker.preset_name();

        let seed_code = if let Some(seed) = self.config.seed {
            format!("import numpy as np; np.random.seed({})", seed)
        } else {
            "import numpy as np; seed = np.random.randint(0, 2**32-1); np.random.seed(seed)".to_string()
        };

        Ok(format!(r#"
import os
os.environ["SUNO_USE_SMALL_MODELS"] = "{use_small}"

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import time
import json

{seed_code}
actual_seed = np.random.get_state()[1][0]

# Preload models
print("Loading Bark models...")
preload_models()

# Generate
print("Generating audio...")
text = '''{text}'''
history_prompt = "{speaker}"

start_time = time.time()
audio_array = generate_audio(text, history_prompt=history_prompt)
generation_time = time.time() - start_time

# Save
print("Saving audio...")
write_wav('{output_path}', SAMPLE_RATE, audio_array)

# Calculate duration
duration = len(audio_array) / SAMPLE_RATE

result = {{
    'duration': duration,
    'sample_rate': SAMPLE_RATE,
    'generation_time': generation_time,
    'speaker': history_prompt,
    'seed': int(actual_seed)
}}
print('RESULT_JSON:' + json.dumps(result))
"#,
            use_small = if self.config.use_small { "True" } else { "False" },
            seed_code = seed_code,
            text = text.replace("'", "\\'"),
            speaker = speaker,
            output_path = output_path.display(),
        ))
    }

    fn run_script(&self, script: &str, output_path: &Path, speaker: &str) -> Result<BarkOutput> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("bark_generate.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        // Parse result
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result found".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(BarkOutput {
            audio_path: output_path.to_path_buf(),
            duration: data["duration"].as_f64().unwrap_or(0.0),
            sample_rate: data["sample_rate"].as_u64().unwrap_or(24000) as u32,
            generation_time: data["generation_time"].as_f64().unwrap_or(0.0),
            speaker: speaker.to_string(),
            seed: data["seed"].as_u64().unwrap_or(0),
        })
    }

    /// Generate batch with variations
    pub fn generate_batch(
        &self,
        base_prompt: &BarkPrompt,
        speakers: &[BarkSpeaker],
    ) -> Result<Vec<BarkOutput>> {
        let mut results = Vec::new();
        for speaker in speakers {
            let prompt = BarkPrompt {
                text: base_prompt.text.clone(),
                speaker: *speaker,
                tags: base_prompt.tags.clone(),
            };
            results.push(self.generate(&prompt)?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BarkConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert!(!config.use_small);
    }

    #[test]
    fn test_prompt_builder() {
        let prompt = BarkPrompt::text("Hello world")
            .with_speaker(BarkSpeaker::EnglishSpeaker1)
            .with_cough(5);

        assert!(prompt.text.contains("Hello"));
        assert_eq!(prompt.speaker, BarkSpeaker::EnglishSpeaker1);
        assert_eq!(prompt.tags.len(), 1);
    }

    #[test]
    fn test_tag_building() {
        let prompt = BarkPrompt::text("Hello world")
            .with_cough(5)
            .with_laugh(0);

        let text = prompt.build_text();
        assert!(text.contains("[coughs]"));
        assert!(text.contains("[laughs]"));
    }

    #[test]
    fn test_bio_prompts() {
        let normal = BarkBioPrompts::normal_speech();
        assert!(normal.text.contains("rainbow"));

        let with_cough = BarkBioPrompts::speech_with_cough();
        assert!(!with_cough.tags.is_empty());
    }

    #[test]
    fn test_speaker_presets() {
        assert_eq!(BarkSpeaker::EnglishSpeaker0.preset_name(), "v2/en_speaker_0");
        assert_eq!(BarkSpeaker::ChineseSpeaker0.preset_name(), "v2/zh_speaker_0");
    }
}
