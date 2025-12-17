//! LTX-Video Diffusion-Based Video Generation
//!
//! This module provides integration with LTX-Video, a fast DiT-based
//! (Diffusion Transformer) video generation model.
//!
//! # Features
//!
//! - Fast generation (faster than real-time)
//! - 30 FPS at 1216x704 resolution
//! - Text-to-video and image-to-video
//! - ComfyUI compatible
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install diffusers transformers accelerate`
//! - GPU with 8GB+ VRAM recommended
//!
//! # References
//!
//! - Lightricks LTX-Video
//! - Hugging Face Diffusers integration

use super::{MediaError, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// LTX-Video generator
#[derive(Debug)]
pub struct LTXVideoGenerator {
    config: DiffusionConfig,
    python_path: PathBuf,
}

/// Configuration for diffusion-based video generation
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Model ID on Hugging Face
    pub model_id: String,
    /// Output directory
    pub output_dir: PathBuf,
    /// Output resolution (width, height)
    pub resolution: (u32, u32),
    /// Frames per second
    pub fps: u32,
    /// Number of frames to generate
    pub num_frames: u32,
    /// Number of inference steps
    pub num_inference_steps: u32,
    /// Guidance scale for CFG
    pub guidance_scale: f32,
    /// GPU device ID (-1 for CPU)
    pub device: i32,
    /// Seed for reproducibility (None for random)
    pub seed: Option<u64>,
    /// Enable attention slicing for memory efficiency
    pub attention_slicing: bool,
    /// Enable VAE tiling for memory efficiency
    pub vae_tiling: bool,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            model_id: "Lightricks/LTX-Video".to_string(),
            output_dir: PathBuf::from("output/video"),
            resolution: (704, 480),  // Lower default for faster generation
            fps: 24,
            num_frames: 49,  // ~2 seconds at 24fps
            num_inference_steps: 30,
            guidance_scale: 7.5,
            device: 0,
            seed: None,
            attention_slicing: true,
            vae_tiling: true,
        }
    }
}

/// Video generation prompt
#[derive(Debug, Clone)]
pub struct VideoPrompt {
    /// Main text prompt describing the video
    pub prompt: String,
    /// Negative prompt (things to avoid)
    pub negative_prompt: Option<String>,
    /// Reference image for image-to-video
    pub reference_image: Option<PathBuf>,
    /// Prompt for end frame (for interpolation)
    pub end_prompt: Option<String>,
}

impl VideoPrompt {
    /// Create a simple text-to-video prompt
    pub fn text(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            reference_image: None,
            end_prompt: None,
        }
    }

    /// Add negative prompt
    pub fn with_negative(mut self, negative: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative.into());
        self
    }

    /// Add reference image for image-to-video
    pub fn with_reference(mut self, image: PathBuf) -> Self {
        self.reference_image = Some(image);
        self
    }
}

/// Preset prompts for biosignal-relevant scenarios
pub struct BiomechanicalPrompts;

impl BiomechanicalPrompts {
    /// Walking/gait video prompt
    pub fn gait_normal() -> VideoPrompt {
        VideoPrompt::text(
            "A person walking naturally in a clinical room, side view, \
             full body visible, neutral lighting, white background, \
             medical examination setting, smooth continuous motion"
        ).with_negative("blurry, distorted, multiple people, partial body")
    }

    /// Parkinsonian gait prompt
    pub fn gait_parkinsons() -> VideoPrompt {
        VideoPrompt::text(
            "An elderly person walking with shuffling gait, reduced arm swing, \
             slightly stooped posture, clinical room, side view, full body visible, \
             slow careful steps, medical examination"
        ).with_negative("running, jumping, blurry, distorted")
    }

    /// Hand tremor prompt
    pub fn hand_tremor() -> VideoPrompt {
        VideoPrompt::text(
            "Close-up of a hand with visible tremor shaking, \
             fingers extended, medical examination, neutral background, \
             clear focus on hand movement, clinical lighting"
        ).with_negative("blurry, multiple hands, face")
    }

    /// Finger tapping task
    pub fn finger_tapping() -> VideoPrompt {
        VideoPrompt::text(
            "Close-up of hand performing finger tapping task, \
             index finger repeatedly touching thumb, clinical setting, \
             clear view of finger movement, medical examination"
        ).with_negative("blurry, face, full body")
    }

    /// Standing balance/postural
    pub fn postural_stability() -> VideoPrompt {
        VideoPrompt::text(
            "Person standing still maintaining balance, front view, \
             full body visible, clinical room, subtle body sway, \
             medical examination, neutral expression"
        ).with_negative("walking, sitting, multiple people")
    }
}

/// Output from video generation
#[derive(Debug)]
pub struct GeneratedVideo {
    /// Path to generated video file
    pub video_path: PathBuf,
    /// Duration in seconds
    pub duration: f64,
    /// Resolution (width, height)
    pub resolution: (u32, u32),
    /// Frames per second
    pub fps: u32,
    /// Generation time in seconds
    pub generation_time: f64,
    /// Seed used for generation
    pub seed: u64,
}

impl LTXVideoGenerator {
    /// Create a new LTX-Video generator
    pub fn new(config: DiffusionConfig) -> Result<Self> {
        // Check if required packages are available
        let check = Command::new("python3")
            .args(["-c", "import diffusers; import torch; print('ok')"])
            .output();

        match check {
            Ok(output) if output.status.success() => {}
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "diffusers".to_string(),
                    install_url: "pip install diffusers transformers accelerate torch".to_string(),
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

    /// Check if LTX-Video dependencies are available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import diffusers; from diffusers import DiffusionPipeline"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Generate video from text prompt
    pub fn generate(&self, prompt: &VideoPrompt) -> Result<GeneratedVideo> {
        let output_path = self.config.output_dir.join(format!(
            "generated_{}.mp4",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        let script = if prompt.reference_image.is_some() {
            self.generate_i2v_script(prompt, &output_path)?
        } else {
            self.generate_t2v_script(prompt, &output_path)?
        };

        self.run_generation_script(&script, &output_path)
    }

    /// Generate text-to-video script
    fn generate_t2v_script(&self, prompt: &VideoPrompt, output_path: &Path) -> Result<String> {
        let negative = prompt.negative_prompt
            .as_deref()
            .unwrap_or("low quality, blurry, distorted");

        let seed_code = if let Some(seed) = self.config.seed {
            format!("generator = torch.Generator(device=device).manual_seed({})", seed)
        } else {
            "import random; seed = random.randint(0, 2**32-1); generator = torch.Generator(device=device).manual_seed(seed)".to_string()
        };

        Ok(format!(r#"
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
import time

# Setup device
device = 'cuda:{device}' if torch.cuda.is_available() and {device} >= 0 else 'cpu'
print(f"Using device: {{device}}")

# Load model
print("Loading LTX-Video model...")
pipe = LTXPipeline.from_pretrained(
    "{model_id}",
    torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
)
pipe = pipe.to(device)

# Memory optimizations
if {attention_slicing}:
    pipe.enable_attention_slicing()
if {vae_tiling}:
    pipe.enable_vae_tiling()

# Setup generator for reproducibility
{seed_code}
actual_seed = generator.initial_seed()

# Generate
print("Generating video...")
prompt = '''{prompt}'''
negative_prompt = '''{negative}'''

start_time = time.time()
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width={width},
    height={height},
    num_frames={num_frames},
    num_inference_steps={num_steps},
    guidance_scale={guidance_scale},
    generator=generator,
).frames[0]
generation_time = time.time() - start_time

# Export to video
print("Exporting video...")
export_to_video(output, '{output_path}', fps={fps})

# Print metadata
import json
result = {{
    'duration': {num_frames} / {fps},
    'resolution': [{width}, {height}],
    'fps': {fps},
    'generation_time': generation_time,
    'seed': actual_seed
}}
print('RESULT_JSON:' + json.dumps(result))
"#,
            device = self.config.device,
            model_id = self.config.model_id,
            attention_slicing = self.config.attention_slicing,
            vae_tiling = self.config.vae_tiling,
            seed_code = seed_code,
            prompt = prompt.prompt.replace("'", "\\'"),
            negative = negative.replace("'", "\\'"),
            width = self.config.resolution.0,
            height = self.config.resolution.1,
            num_frames = self.config.num_frames,
            num_steps = self.config.num_inference_steps,
            guidance_scale = self.config.guidance_scale,
            output_path = output_path.display(),
            fps = self.config.fps,
        ))
    }

    /// Generate image-to-video script
    fn generate_i2v_script(&self, prompt: &VideoPrompt, output_path: &Path) -> Result<String> {
        let reference_image = prompt.reference_image
            .as_ref()
            .ok_or_else(|| MediaError::InvalidConfig("Reference image required for I2V".to_string()))?;

        Ok(format!(r#"
import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import time

device = 'cuda:{device}' if torch.cuda.is_available() and {device} >= 0 else 'cpu'

# Load model
pipe = LTXImageToVideoPipeline.from_pretrained(
    "{model_id}",
    torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
)
pipe = pipe.to(device)

if {attention_slicing}:
    pipe.enable_attention_slicing()

# Load reference image
image = load_image("{reference_image}")

# Generate
start_time = time.time()
output = pipe(
    prompt='''{prompt}''',
    image=image,
    width={width},
    height={height},
    num_frames={num_frames},
    num_inference_steps={num_steps},
).frames[0]
generation_time = time.time() - start_time

export_to_video(output, '{output_path}', fps={fps})

import json
result = {{
    'duration': {num_frames} / {fps},
    'resolution': [{width}, {height}],
    'fps': {fps},
    'generation_time': generation_time,
    'seed': 0
}}
print('RESULT_JSON:' + json.dumps(result))
"#,
            device = self.config.device,
            model_id = self.config.model_id,
            attention_slicing = self.config.attention_slicing,
            reference_image = reference_image.display(),
            prompt = prompt.prompt.replace("'", "\\'"),
            width = self.config.resolution.0,
            height = self.config.resolution.1,
            num_frames = self.config.num_frames,
            num_steps = self.config.num_inference_steps,
            output_path = output_path.display(),
            fps = self.config.fps,
        ))
    }

    /// Run generation script and parse results
    fn run_generation_script(&self, script: &str, output_path: &Path) -> Result<GeneratedVideo> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("ltx_generate.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        // Parse result from stdout
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result found".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(GeneratedVideo {
            video_path: output_path.to_path_buf(),
            duration: data["duration"].as_f64().unwrap_or(0.0),
            resolution: (
                data["resolution"][0].as_u64().unwrap_or(704) as u32,
                data["resolution"][1].as_u64().unwrap_or(480) as u32,
            ),
            fps: data["fps"].as_u64().unwrap_or(24) as u32,
            generation_time: data["generation_time"].as_f64().unwrap_or(0.0),
            seed: data["seed"].as_u64().unwrap_or(0),
        })
    }

    /// Generate batch of videos with variations
    pub fn generate_batch(
        &self,
        base_prompt: &VideoPrompt,
        variations: &[String],
    ) -> Result<Vec<GeneratedVideo>> {
        let mut results = Vec::new();

        for variation in variations {
            let mut prompt = base_prompt.clone();
            prompt.prompt = format!("{}, {}", base_prompt.prompt, variation);
            results.push(self.generate(&prompt)?);
        }

        Ok(results)
    }
}

/// Wan 2.2 video generator (higher quality, slower)
#[derive(Debug)]
pub struct Wan22Generator {
    config: DiffusionConfig,
}

impl Wan22Generator {
    pub fn new(mut config: DiffusionConfig) -> Result<Self> {
        config.model_id = "Wan-AI/Wan2.2-T2V-A14B".to_string();
        Ok(Self { config })
    }

    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import diffusers; print('ok')"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

/// HunyuanVideo generator
#[derive(Debug)]
pub struct HunyuanVideoGenerator {
    config: DiffusionConfig,
}

impl HunyuanVideoGenerator {
    pub fn new(mut config: DiffusionConfig) -> Result<Self> {
        config.model_id = "tencent/HunyuanVideo".to_string();
        Ok(Self { config })
    }

    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import diffusers; print('ok')"])
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
        let config = DiffusionConfig::default();
        assert_eq!(config.fps, 24);
        assert_eq!(config.num_frames, 49);
    }

    #[test]
    fn test_prompt_builder() {
        let prompt = VideoPrompt::text("test")
            .with_negative("bad quality");
        assert_eq!(prompt.prompt, "test");
        assert_eq!(prompt.negative_prompt, Some("bad quality".to_string()));
    }

    #[test]
    fn test_biomechanical_prompts() {
        let gait = BiomechanicalPrompts::gait_normal();
        assert!(gait.prompt.contains("walking"));

        let tremor = BiomechanicalPrompts::hand_tremor();
        assert!(tremor.prompt.contains("tremor"));
    }
}
