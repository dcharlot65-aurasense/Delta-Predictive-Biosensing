//! WORLD Vocoder-based Audio Generation
//!
//! Generates actual audio waveforms using Python/pyworld for pathological voice synthesis.
//! Uses subprocess calls to Python scripts with WORLD vocoder.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::traits::{GeneratedData, GroundTruth, SyntheticGenerator};

/// Parameters for sustained vowel generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SustainedVowelParams {
    pub duration_sec: f64,
    pub f0_mean: f64,
    pub f0_std: f64,
    pub jitter_percent: f64,
    pub shimmer_percent: f64,
    pub hnr_db: f64,
    pub tremor_frequency: f64,
    pub tremor_amplitude: f64,
    pub vowel: String, // "a", "i", "u", "e", "o"
    pub sample_rate: u32,
    pub seed: u64,
}

impl Default for SustainedVowelParams {
    fn default() -> Self {
        Self {
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
        }
    }
}

/// Parameters for connected speech generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectedSpeechParams {
    pub duration_sec: f64,
    pub base_f0: f64,
    pub f0_range: f64,
    pub speech_rate: f64,
    pub hypophonia_db: f64,
    pub monotone_factor: f64,
    pub tremor_frequency: f64,
    pub tremor_amplitude: f64,
    pub pause_probability: f64,
    pub sample_rate: u32,
    pub seed: u64,
}

impl Default for ConnectedSpeechParams {
    fn default() -> Self {
        Self {
            duration_sec: 5.0,
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
        }
    }
}

/// Parameters for diadochokinesis (rapid syllable repetition)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiadochokinesisParams {
    pub syllables: String, // "pa-ta-ka" or "pa" for AMR
    pub repetitions: u32,
    pub target_rate: f64,
    pub rate_variability: f64,
    pub amplitude_variability: f64,
    pub f0_mean: f64,
    pub sample_rate: u32,
    pub seed: u64,
}

impl Default for DiadochokinesisParams {
    fn default() -> Self {
        Self {
            syllables: "pa-ta-ka".to_string(),
            repetitions: 10,
            target_rate: 6.0,
            rate_variability: 0.1,
            amplitude_variability: 0.05,
            f0_mean: 120.0,
            sample_rate: 16000,
            seed: 42,
        }
    }
}

/// Parameters for reading passage generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadingPassageParams {
    pub passage: String, // "standard", "rainbow", "grandfather"
    pub base_f0: f64,
    pub speech_rate: f64,
    pub articulation_precision: f64,
    pub breath_pause_regularity: f64,
    pub sample_rate: u32,
    pub seed: u64,
}

impl Default for ReadingPassageParams {
    fn default() -> Self {
        Self {
            passage: "standard".to_string(),
            base_f0: 120.0,
            speech_rate: 1.0,
            articulation_precision: 1.0,
            breath_pause_regularity: 1.0,
            sample_rate: 16000,
            seed: 42,
        }
    }
}

/// Audio ground truth information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGroundTruth {
    /// F0 contour (Hz) at each time frame
    pub f0_contour: Vec<f64>,

    /// Time points for F0 values (seconds)
    pub f0_times: Vec<f64>,

    /// Formant frequencies (F1-F4)
    pub formants: Option<HashMap<String, Vec<f64>>>,

    /// Event markers (syllables, pauses, etc.)
    pub events: Option<Vec<AudioEvent>>,

    /// Generator parameters
    pub parameters: HashMap<String, f64>,

    /// Spectral features (if available)
    pub spectral_features: Option<HashMap<String, Vec<f64>>>,
}

impl GroundTruth for AudioGroundTruth {}

/// Audio event marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEvent {
    pub time: f64,
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(flatten)]
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Audio output from Python generator
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonAudioOutput {
    audio: Vec<f32>,
    sample_rate: u32,
    ground_truth: AudioGroundTruth,
}

/// WORLD vocoder audio generator (calls Python scripts)
pub struct WorldAudioGenerator {
    python_script: PathBuf,
    python_executable: String,
}

impl WorldAudioGenerator {
    /// Create new WORLD vocoder audio generator
    ///
    /// # Arguments
    /// * `python_script_path` - Path to generator.py (optional, uses default if None)
    /// * `python_executable` - Python executable name (default: "python3")
    pub fn new(python_script_path: Option<PathBuf>, python_executable: Option<String>) -> Self {
        let script_path = python_script_path.unwrap_or_else(|| {
            // Default to tools/level3_audio/generator.py relative to crate root
            let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.pop(); // Go up from dpb-synth
            path.pop(); // Go up from crates
            path.push("tools");
            path.push("level3_audio");
            path.push("generator.py");
            path
        });

        Self {
            python_script: script_path,
            python_executable: python_executable.unwrap_or_else(|| "python3".to_string()),
        }
    }

    /// Call Python generator script
    fn call_python_generator(
        &self,
        generator_type: &str,
        params_json: &str,
    ) -> crate::Result<PythonAudioOutput> {
        // Check if script exists
        if !self.python_script.exists() {
            return Err(crate::GeneratorError::ConfigurationError(format!(
                "Python script not found: {}",
                self.python_script.display()
            )));
        }

        // Execute Python script
        let output = Command::new(&self.python_executable)
            .arg(&self.python_script)
            .arg(generator_type)
            .arg(params_json)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| {
                crate::GeneratorError::ComputationError(format!(
                    "Failed to execute Python script: {}",
                    e
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(crate::GeneratorError::ComputationError(format!(
                "Python script failed: {}",
                stderr
            )));
        }

        // Parse JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: PythonAudioOutput = serde_json::from_str(&stdout).map_err(|e| {
            crate::GeneratorError::ComputationError(format!("Failed to parse JSON output: {}", e))
        })?;

        Ok(result)
    }

    /// Generate sustained vowel
    pub fn generate_sustained_vowel(
        &self,
        params: &SustainedVowelParams,
    ) -> crate::Result<GeneratedData<Array1<f32>, AudioGroundTruth>> {
        let params_json = serde_json::to_string(params)
            .map_err(|e| crate::GeneratorError::InvalidParameter(e.to_string()))?;

        let result = self.call_python_generator("sustained_vowel", &params_json)?;

        Ok(GeneratedData::new(
            Array1::from_vec(result.audio),
            result.ground_truth,
            result.sample_rate as f64,
        ))
    }

    /// Generate connected speech
    pub fn generate_connected_speech(
        &self,
        params: &ConnectedSpeechParams,
    ) -> crate::Result<GeneratedData<Array1<f32>, AudioGroundTruth>> {
        let params_json = serde_json::to_string(params)
            .map_err(|e| crate::GeneratorError::InvalidParameter(e.to_string()))?;

        let result = self.call_python_generator("connected_speech", &params_json)?;

        Ok(GeneratedData::new(
            Array1::from_vec(result.audio),
            result.ground_truth,
            result.sample_rate as f64,
        ))
    }

    /// Generate diadochokinesis sequence
    pub fn generate_diadochokinesis(
        &self,
        params: &DiadochokinesisParams,
    ) -> crate::Result<GeneratedData<Array1<f32>, AudioGroundTruth>> {
        let params_json = serde_json::to_string(params)
            .map_err(|e| crate::GeneratorError::InvalidParameter(e.to_string()))?;

        let result = self.call_python_generator("diadochokinesis", &params_json)?;

        Ok(GeneratedData::new(
            Array1::from_vec(result.audio),
            result.ground_truth,
            result.sample_rate as f64,
        ))
    }

    /// Generate reading passage
    pub fn generate_reading_passage(
        &self,
        params: &ReadingPassageParams,
    ) -> crate::Result<GeneratedData<Array1<f32>, AudioGroundTruth>> {
        let params_json = serde_json::to_string(params)
            .map_err(|e| crate::GeneratorError::InvalidParameter(e.to_string()))?;

        let result = self.call_python_generator("reading_passage", &params_json)?;

        Ok(GeneratedData::new(
            Array1::from_vec(result.audio),
            result.ground_truth,
            result.sample_rate as f64,
        ))
    }
}

/// Sustained vowel generator implementing SyntheticGenerator trait
pub struct SustainedVowelGenerator {
    engine: WorldAudioGenerator,
}

impl SustainedVowelGenerator {
    pub fn new() -> Self {
        Self {
            engine: WorldAudioGenerator::new(None, None),
        }
    }

    pub fn with_custom_python(script_path: PathBuf, python_exe: String) -> Self {
        Self {
            engine: WorldAudioGenerator::new(Some(script_path), Some(python_exe)),
        }
    }
}

impl Default for SustainedVowelGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticGenerator for SustainedVowelGenerator {
    type Output = Array1<f32>;
    type GroundTruth = AudioGroundTruth;
    type Parameters = SustainedVowelParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        let mut params = params.clone();
        params.seed = seed;
        self.engine.generate_sustained_vowel(&params)
    }

    fn default_params() -> Self::Parameters {
        SustainedVowelParams::default()
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration_sec <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "duration_sec must be positive".to_string(),
            ));
        }

        if params.f0_mean <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "f0_mean must be positive".to_string(),
            ));
        }

        if !["a", "i", "u", "e", "o"].contains(&params.vowel.as_str()) {
            return Err(crate::GeneratorError::InvalidParameter(format!(
                "Invalid vowel: {}. Must be one of: a, i, u, e, o",
                params.vowel
            )));
        }

        Ok(())
    }
}

/// Connected speech generator implementing SyntheticGenerator trait
pub struct ConnectedSpeechGenerator {
    engine: WorldAudioGenerator,
}

impl ConnectedSpeechGenerator {
    pub fn new() -> Self {
        Self {
            engine: WorldAudioGenerator::new(None, None),
        }
    }
}

impl Default for ConnectedSpeechGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticGenerator for ConnectedSpeechGenerator {
    type Output = Array1<f32>;
    type GroundTruth = AudioGroundTruth;
    type Parameters = ConnectedSpeechParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        let mut params = params.clone();
        params.seed = seed;
        self.engine.generate_connected_speech(&params)
    }

    fn default_params() -> Self::Parameters {
        ConnectedSpeechParams::default()
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration_sec <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "duration_sec must be positive".to_string(),
            ));
        }

        if params.speech_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "speech_rate must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

/// Diadochokinesis generator implementing SyntheticGenerator trait
pub struct DiadochokinesisGenerator {
    engine: WorldAudioGenerator,
}

impl DiadochokinesisGenerator {
    pub fn new() -> Self {
        Self {
            engine: WorldAudioGenerator::new(None, None),
        }
    }
}

impl Default for DiadochokinesisGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticGenerator for DiadochokinesisGenerator {
    type Output = Array1<f32>;
    type GroundTruth = AudioGroundTruth;
    type Parameters = DiadochokinesisParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        let mut params = params.clone();
        params.seed = seed;
        self.engine.generate_diadochokinesis(&params)
    }

    fn default_params() -> Self::Parameters {
        DiadochokinesisParams::default()
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.repetitions == 0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "repetitions must be positive".to_string(),
            ));
        }

        if params.target_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "target_rate must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

/// Reading passage generator implementing SyntheticGenerator trait
pub struct ReadingPassageGenerator {
    engine: WorldAudioGenerator,
}

impl ReadingPassageGenerator {
    pub fn new() -> Self {
        Self {
            engine: WorldAudioGenerator::new(None, None),
        }
    }
}

impl Default for ReadingPassageGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticGenerator for ReadingPassageGenerator {
    type Output = Array1<f32>;
    type GroundTruth = AudioGroundTruth;
    type Parameters = ReadingPassageParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        let mut params = params.clone();
        params.seed = seed;
        self.engine.generate_reading_passage(&params)
    }

    fn default_params() -> Self::Parameters {
        ReadingPassageParams::default()
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.speech_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "speech_rate must be positive".to_string(),
            ));
        }

        if params.articulation_precision < 0.0 || params.articulation_precision > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "articulation_precision must be between 0 and 1".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = SustainedVowelParams::default();
        assert_eq!(params.vowel, "a");
        assert_eq!(params.sample_rate, 16000);
    }

    #[test]
    fn test_validate_params() {
        let mut params = SustainedVowelParams::default();
        assert!(SustainedVowelGenerator::validate_params(&params).is_ok());

        params.duration_sec = -1.0;
        assert!(SustainedVowelGenerator::validate_params(&params).is_err());

        params.duration_sec = 3.0;
        params.vowel = "invalid".to_string();
        assert!(SustainedVowelGenerator::validate_params(&params).is_err());
    }

    #[test]
    fn test_connected_speech_params() {
        let params = ConnectedSpeechParams::default();
        assert_eq!(params.speech_rate, 1.0);
        assert!(ConnectedSpeechGenerator::validate_params(&params).is_ok());
    }

    #[test]
    fn test_ddk_params() {
        let params = DiadochokinesisParams::default();
        assert_eq!(params.syllables, "pa-ta-ka");
        assert!(DiadochokinesisGenerator::validate_params(&params).is_ok());
    }
}
