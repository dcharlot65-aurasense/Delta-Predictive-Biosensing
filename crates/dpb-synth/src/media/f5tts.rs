//! F5-TTS Fast Zero-Shot Voice Cloning Integration
//!
//! F5-TTS is a fast, high-quality text-to-speech model with zero-shot
//! voice cloning capabilities from a short audio reference.
//!
//! # Features
//!
//! - Zero-shot voice cloning from 3-10 second samples
//! - Fast generation (near real-time)
//! - High-quality output
//! - MIT License
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install f5-tts`
//! - GPU recommended for best performance

use super::{MediaError, Result, VoiceGroundTruth};
use std::path::{Path, PathBuf};
use std::process::Command;

/// F5-TTS voice synthesizer
#[derive(Debug)]
pub struct F5TTSGenerator {
    config: F5TTSConfig,
    python_path: PathBuf,
}

/// Configuration for F5-TTS
#[derive(Debug, Clone)]
pub struct F5TTSConfig {
    /// Output directory
    pub output_dir: PathBuf,
    /// Sample rate (default 24000)
    pub sample_rate: u32,
    /// GPU device ID (-1 for CPU)
    pub device: i32,
    /// Model variant
    pub model: F5Model,
    /// Seed for reproducibility
    pub seed: Option<u64>,
    /// Speed factor (1.0 = normal)
    pub speed: f32,
}

impl Default for F5TTSConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("output/f5tts"),
            sample_rate: 24000,
            device: 0,
            model: F5Model::F5TTS,
            seed: None,
            speed: 1.0,
        }
    }
}

/// F5-TTS model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F5Model {
    /// F5-TTS base model (fastest)
    F5TTS,
    /// E2-TTS (better quality, slower)
    E2TTS,
}

impl F5Model {
    fn model_id(&self) -> &'static str {
        match self {
            F5Model::F5TTS => "F5-TTS",
            F5Model::E2TTS => "E2-TTS",
        }
    }
}

/// Voice prompt for F5-TTS
#[derive(Debug, Clone)]
pub struct F5TTSPrompt {
    /// Text to synthesize
    pub text: String,
    /// Reference audio for voice cloning
    pub reference_audio: PathBuf,
    /// Transcript of reference audio (optional, improves quality)
    pub reference_transcript: Option<String>,
}

impl F5TTSPrompt {
    /// Create a new prompt with text and reference audio
    pub fn new(text: impl Into<String>, reference_audio: PathBuf) -> Self {
        Self {
            text: text.into(),
            reference_audio,
            reference_transcript: None,
        }
    }

    /// Add reference transcript
    pub fn with_transcript(mut self, transcript: impl Into<String>) -> Self {
        self.reference_transcript = Some(transcript.into());
        self
    }
}

/// Pathological voice modification parameters
#[derive(Debug, Clone)]
pub struct PathologicalVoiceParams {
    /// Hypophonia (reduced volume) severity 0-1
    pub hypophonia: f32,
    /// Breathiness severity 0-1
    pub breathiness: f32,
    /// Tremor in voice (frequency in Hz, 0 for none)
    pub vocal_tremor_freq: f32,
    /// Tremor amplitude 0-1
    pub vocal_tremor_amp: f32,
    /// Speech rate modification (1.0 = normal, <1 slower)
    pub speech_rate: f32,
    /// Add dysfluencies (repetitions, blocks)
    pub dysfluencies: bool,
}

impl Default for PathologicalVoiceParams {
    fn default() -> Self {
        Self {
            hypophonia: 0.0,
            breathiness: 0.0,
            vocal_tremor_freq: 0.0,
            vocal_tremor_amp: 0.0,
            speech_rate: 1.0,
            dysfluencies: false,
        }
    }
}

/// Preset voice modifications for specific conditions
pub struct PathologicalVoicePresets;

impl PathologicalVoicePresets {
    /// Early Parkinson's voice characteristics
    pub fn parkinsons_early() -> PathologicalVoiceParams {
        PathologicalVoiceParams {
            hypophonia: 0.3,
            breathiness: 0.2,
            vocal_tremor_freq: 5.0,
            vocal_tremor_amp: 0.1,
            speech_rate: 0.95,
            dysfluencies: false,
        }
    }

    /// Advanced Parkinson's voice characteristics
    pub fn parkinsons_advanced() -> PathologicalVoiceParams {
        PathologicalVoiceParams {
            hypophonia: 0.6,
            breathiness: 0.4,
            vocal_tremor_freq: 5.5,
            vocal_tremor_amp: 0.25,
            speech_rate: 0.8,
            dysfluencies: true,
        }
    }

    /// Essential tremor voice
    pub fn essential_tremor() -> PathologicalVoiceParams {
        PathologicalVoiceParams {
            hypophonia: 0.1,
            breathiness: 0.1,
            vocal_tremor_freq: 6.0,
            vocal_tremor_amp: 0.15,
            speech_rate: 0.9,
            dysfluencies: false,
        }
    }

    /// Dysarthria characteristics
    pub fn dysarthria() -> PathologicalVoiceParams {
        PathologicalVoiceParams {
            hypophonia: 0.2,
            breathiness: 0.3,
            vocal_tremor_freq: 0.0,
            vocal_tremor_amp: 0.0,
            speech_rate: 0.7,
            dysfluencies: true,
        }
    }
}

/// Output from F5-TTS generation
#[derive(Debug)]
pub struct F5TTSOutput {
    /// Path to generated audio file
    pub audio_path: PathBuf,
    /// Duration in seconds
    pub duration: f64,
    /// Sample rate
    pub sample_rate: u32,
    /// Generation time
    pub generation_time: f64,
    /// Real-time factor (generation_time / duration)
    pub rtf: f64,
    /// Seed used
    pub seed: u64,
    /// Voice ground truth parameters
    pub ground_truth: Option<VoiceGroundTruth>,
}

impl F5TTSGenerator {
    /// Create a new F5-TTS generator
    pub fn new(config: F5TTSConfig) -> Result<Self> {
        // Check for f5-tts
        let check = Command::new("python3")
            .args(["-c", "from f5_tts.infer.utils_infer import infer_process; print('ok')"])
            .output();

        // Try alternative import
        let check_alt = Command::new("python3")
            .args(["-c", "import f5_tts; print('ok')"])
            .output();

        let available = check.map(|o| o.status.success()).unwrap_or(false)
            || check_alt.map(|o| o.status.success()).unwrap_or(false);

        if !available {
            return Err(MediaError::ToolNotFound {
                tool: "f5-tts".to_string(),
                install_url: "pip install f5-tts".to_string(),
            });
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

    /// Check if F5-TTS is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import f5_tts"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Generate speech from prompt
    pub fn generate(&self, prompt: &F5TTSPrompt) -> Result<F5TTSOutput> {
        let output_path = self.config.output_dir.join(format!(
            "f5tts_{}.wav",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        let script = self.generate_script(prompt, &output_path)?;
        self.run_script(&script, &output_path)
    }

    /// Generate speech with pathological modifications
    pub fn generate_pathological(
        &self,
        prompt: &F5TTSPrompt,
        params: &PathologicalVoiceParams,
    ) -> Result<F5TTSOutput> {
        let output_path = self.config.output_dir.join(format!(
            "f5tts_pathological_{}.wav",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        let script = self.generate_pathological_script(prompt, params, &output_path)?;
        self.run_script(&script, &output_path)
    }

    fn generate_script(&self, prompt: &F5TTSPrompt, output_path: &Path) -> Result<String> {
        let ref_transcript = prompt.reference_transcript
            .as_deref()
            .unwrap_or("");

        let seed_code = if let Some(seed) = self.config.seed {
            format!("import torch; torch.manual_seed({}); import numpy as np; np.random.seed({})", seed, seed)
        } else {
            "import numpy as np; seed = np.random.randint(0, 2**32-1); np.random.seed(seed); import torch; torch.manual_seed(seed)".to_string()
        };

        Ok(format!(r#"
import time
import json
import soundfile as sf
{seed_code}

try:
    from f5_tts.api import F5TTS
    use_api = True
except ImportError:
    from f5_tts.infer.utils_infer import load_model, infer_process, preprocess_ref_audio_text
    use_api = False

# Setup device
device = 'cuda:{device}' if {device} >= 0 else 'cpu'

# Text to synthesize
text = '''{text}'''
ref_audio = '{ref_audio}'
ref_text = '''{ref_transcript}'''

print("Loading F5-TTS model...")
start_time = time.time()

if use_api:
    # Use newer API
    tts = F5TTS(device=device)

    print("Generating speech...")
    gen_start = time.time()
    audio, sr, _ = tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text if ref_text else None,
        gen_text=text,
        speed={speed}
    )
    generation_time = time.time() - gen_start

else:
    # Use older interface
    model = load_model("{model}", device=device)

    # Preprocess reference
    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio, ref_text)

    print("Generating speech...")
    gen_start = time.time()
    audio, sr = infer_process(
        model,
        ref_audio_proc,
        ref_text_proc,
        text,
        device=device,
        speed={speed}
    )
    generation_time = time.time() - gen_start

# Save audio
sf.write('{output_path}', audio, sr)

# Calculate metrics
duration = len(audio) / sr
rtf = generation_time / duration if duration > 0 else 0

try:
    actual_seed = np.random.get_state()[1][0]
except:
    actual_seed = 0

result = {{
    'duration': duration,
    'sample_rate': int(sr),
    'generation_time': generation_time,
    'rtf': rtf,
    'seed': int(actual_seed)
}}
print('RESULT_JSON:' + json.dumps(result))
"#,
            seed_code = seed_code,
            device = self.config.device,
            text = prompt.text.replace("'", "\\'"),
            ref_audio = prompt.reference_audio.display(),
            ref_transcript = ref_transcript.replace("'", "\\'"),
            speed = self.config.speed,
            model = self.config.model.model_id(),
            output_path = output_path.display(),
        ))
    }

    fn generate_pathological_script(
        &self,
        prompt: &F5TTSPrompt,
        params: &PathologicalVoiceParams,
        output_path: &Path,
    ) -> Result<String> {
        let ref_transcript = prompt.reference_transcript
            .as_deref()
            .unwrap_or("");

        Ok(format!(r#"
import time
import json
import numpy as np
import soundfile as sf
from scipy import signal

try:
    from f5_tts.api import F5TTS
    use_api = True
except ImportError:
    from f5_tts.infer.utils_infer import load_model, infer_process, preprocess_ref_audio_text
    use_api = False

device = 'cuda:{device}' if {device} >= 0 else 'cpu'

text = '''{text}'''
ref_audio = '{ref_audio}'
ref_text = '''{ref_transcript}'''

# Pathological parameters
hypophonia = {hypophonia}
breathiness = {breathiness}
tremor_freq = {tremor_freq}
tremor_amp = {tremor_amp}
speech_rate = {speech_rate}

print("Loading F5-TTS model...")
start_time = time.time()

if use_api:
    tts = F5TTS(device=device)
    print("Generating base speech...")
    gen_start = time.time()
    audio, sr, _ = tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text if ref_text else None,
        gen_text=text,
        speed=speech_rate
    )
else:
    model = load_model("{model}", device=device)
    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio, ref_text)
    print("Generating base speech...")
    gen_start = time.time()
    audio, sr = infer_process(model, ref_audio_proc, ref_text_proc, text, device=device, speed=speech_rate)

generation_time = time.time() - gen_start

# Apply pathological modifications
print("Applying pathological modifications...")

# Convert to float
audio = audio.astype(np.float32)

# Hypophonia (reduce volume)
if hypophonia > 0:
    audio = audio * (1.0 - hypophonia * 0.7)

# Add breathiness (filtered noise)
if breathiness > 0:
    noise = np.random.randn(len(audio)) * breathiness * 0.1
    # High-pass filter the noise for breath-like sound
    b, a = signal.butter(4, 2000 / (sr / 2), 'high')
    breath_noise = signal.filtfilt(b, a, noise)
    audio = audio + breath_noise

# Add vocal tremor (amplitude modulation)
if tremor_freq > 0 and tremor_amp > 0:
    t = np.arange(len(audio)) / sr
    tremor = 1.0 + tremor_amp * np.sin(2 * np.pi * tremor_freq * t)
    audio = audio * tremor

# Normalize
max_val = np.max(np.abs(audio))
if max_val > 0:
    audio = audio / max_val * 0.95

# Save
sf.write('{output_path}', audio, sr)

duration = len(audio) / sr
rtf = generation_time / duration if duration > 0 else 0

result = {{
    'duration': duration,
    'sample_rate': int(sr),
    'generation_time': generation_time,
    'rtf': rtf,
    'seed': 0,
    'pathological_params': {{
        'hypophonia': hypophonia,
        'breathiness': breathiness,
        'vocal_tremor_freq': tremor_freq,
        'vocal_tremor_amp': tremor_amp,
        'speech_rate': speech_rate
    }}
}}
print('RESULT_JSON:' + json.dumps(result))
"#,
            device = self.config.device,
            text = prompt.text.replace("'", "\\'"),
            ref_audio = prompt.reference_audio.display(),
            ref_transcript = ref_transcript.replace("'", "\\'"),
            hypophonia = params.hypophonia,
            breathiness = params.breathiness,
            tremor_freq = params.vocal_tremor_freq,
            tremor_amp = params.vocal_tremor_amp,
            speech_rate = params.speech_rate,
            model = self.config.model.model_id(),
            output_path = output_path.display(),
        ))
    }

    fn run_script(&self, script: &str, output_path: &Path) -> Result<F5TTSOutput> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("f5tts_generate.py");
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

        // Build ground truth if pathological params present
        let ground_truth = data["pathological_params"].as_object().map(|params| {
            VoiceGroundTruth {
                f0: Vec::new(),
                jitter: 0.0,
                shimmer: 0.0,
                hnr: 0.0,
                mpt: None,
                speech_rate: params.get("speech_rate")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
                hypophonia: params.get("hypophonia")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                dysarthria: 0.0,
                phonemes: Vec::new(),
            }
        });

        Ok(F5TTSOutput {
            audio_path: output_path.to_path_buf(),
            duration: data["duration"].as_f64().unwrap_or(0.0),
            sample_rate: data["sample_rate"].as_u64().unwrap_or(24000) as u32,
            generation_time: data["generation_time"].as_f64().unwrap_or(0.0),
            rtf: data["rtf"].as_f64().unwrap_or(0.0),
            seed: data["seed"].as_u64().unwrap_or(0),
            ground_truth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = F5TTSConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.speed, 1.0);
    }

    #[test]
    fn test_prompt_builder() {
        let prompt = F5TTSPrompt::new("Hello world", PathBuf::from("ref.wav"))
            .with_transcript("Reference text");

        assert_eq!(prompt.text, "Hello world");
        assert_eq!(prompt.reference_transcript, Some("Reference text".to_string()));
    }

    #[test]
    fn test_pathological_presets() {
        let early_pd = PathologicalVoicePresets::parkinsons_early();
        assert!(early_pd.hypophonia > 0.0);
        assert!(early_pd.vocal_tremor_freq > 0.0);

        let advanced_pd = PathologicalVoicePresets::parkinsons_advanced();
        assert!(advanced_pd.hypophonia > early_pd.hypophonia);
    }

    #[test]
    fn test_model_ids() {
        assert_eq!(F5Model::F5TTS.model_id(), "F5-TTS");
        assert_eq!(F5Model::E2TTS.model_id(), "E2-TTS");
    }
}
