//! CogVideoX Video Generation Integration
//!
//! CogVideoX is an open-source video generation model from Tsinghua University
//! available in 2B and 5B parameter versions.
//!
//! # Features
//!
//! - Text-to-video and image-to-video generation
//! - 6-second video clips at up to 720p
//! - Better temporal coherence than many alternatives
//! - Apache 2.0 license
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install diffusers transformers accelerate`
//! - GPU with 12GB+ VRAM (2B) or 24GB+ (5B)

use super::{MediaError, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// CogVideoX generator
#[derive(Debug)]
pub struct CogVideoXGenerator {
    config: CogVideoConfig,
    python_path: PathBuf,
}

/// Configuration for CogVideoX
#[derive(Debug, Clone)]
pub struct CogVideoConfig {
    /// Model variant (2b or 5b)
    pub model_variant: CogVideoVariant,
    /// Output directory
    pub output_dir: PathBuf,
    /// Output resolution (width, height)
    pub resolution: (u32, u32),
    /// Frames per second
    pub fps: u32,
    /// Number of frames (max 49 for ~6 seconds)
    pub num_frames: u32,
    /// Number of inference steps
    pub num_inference_steps: u32,
    /// Guidance scale for classifier-free guidance
    pub guidance_scale: f32,
    /// GPU device ID (-1 for CPU)
    pub device: i32,
    /// Seed for reproducibility
    pub seed: Option<u64>,
    /// Enable CPU offload for memory efficiency
    pub cpu_offload: bool,
    /// Enable VAE slicing for memory efficiency
    pub vae_slicing: bool,
}

/// CogVideoX model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CogVideoVariant {
    /// 2B parameter model - faster, less VRAM
    V2B,
    /// 5B parameter model - higher quality
    V5B,
    /// 5B parameter image-to-video model
    V5B_I2V,
}

impl CogVideoVariant {
    fn model_id(&self) -> &'static str {
        match self {
            CogVideoVariant::V2B => "THUDM/CogVideoX-2b",
            CogVideoVariant::V5B => "THUDM/CogVideoX-5b",
            CogVideoVariant::V5B_I2V => "THUDM/CogVideoX-5b-I2V",
        }
    }

    fn pipeline_class(&self) -> &'static str {
        match self {
            CogVideoVariant::V2B | CogVideoVariant::V5B => "CogVideoXPipeline",
            CogVideoVariant::V5B_I2V => "CogVideoXImageToVideoPipeline",
        }
    }
}

impl Default for CogVideoConfig {
    fn default() -> Self {
        Self {
            model_variant: CogVideoVariant::V2B,
            output_dir: PathBuf::from("output/cogvideo"),
            resolution: (720, 480),
            fps: 8,
            num_frames: 49, // ~6 seconds at 8fps
            num_inference_steps: 50,
            guidance_scale: 6.0,
            device: 0,
            seed: None,
            cpu_offload: true,
            vae_slicing: true,
        }
    }
}

/// Video generation prompt for CogVideoX
#[derive(Debug, Clone)]
pub struct CogVideoPrompt {
    /// Main text prompt
    pub prompt: String,
    /// Negative prompt
    pub negative_prompt: Option<String>,
    /// Reference image for I2V mode
    pub reference_image: Option<PathBuf>,
}

impl CogVideoPrompt {
    /// Create text-to-video prompt
    pub fn text(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            reference_image: None,
        }
    }

    /// Add negative prompt
    pub fn with_negative(mut self, negative: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative.into());
        self
    }

    /// Set reference image for I2V
    pub fn with_image(mut self, image: PathBuf) -> Self {
        self.reference_image = Some(image);
        self
    }
}

/// Preset prompts for biosignal scenarios
pub struct CogVideoBioPrompts;

impl CogVideoBioPrompts {
    /// Normal walking gait
    pub fn normal_gait() -> CogVideoPrompt {
        CogVideoPrompt::text(
            "A person walking naturally down a hallway, side view, full body visible, \
             smooth continuous motion, neutral lighting, clinical setting"
        ).with_negative("blurry, distorted, partial body, jerky motion")
    }

    /// Parkinsonian shuffling gait
    pub fn parkinsons_gait() -> CogVideoPrompt {
        CogVideoPrompt::text(
            "An elderly person walking with small shuffling steps, reduced arm swing, \
             slightly stooped posture, side view, full body, slow deliberate movement"
        ).with_negative("running, jumping, blurry, distorted")
    }

    /// Hand at rest with tremor
    pub fn resting_tremor() -> CogVideoPrompt {
        CogVideoPrompt::text(
            "Close-up of a hand resting on a table with visible tremor, \
             rhythmic shaking motion, medical examination setting, clear focus"
        ).with_negative("blurry, multiple hands, action movement")
    }

    /// Finger-to-nose task
    pub fn finger_nose_task() -> CogVideoPrompt {
        CogVideoPrompt::text(
            "Person performing finger-to-nose test, extending arm and touching nose tip, \
             neurological examination, clinical setting, clear view of arm movement"
        ).with_negative("blurry, partial view, multiple people")
    }

    /// Speaking with dysarthria
    pub fn dysarthric_speech() -> CogVideoPrompt {
        CogVideoPrompt::text(
            "Close-up of person speaking with effort, visible lip and jaw movement, \
             frontal view face, neutral background, medical setting"
        ).with_negative("blurry, side profile, multiple faces")
    }
}

/// Output from CogVideoX generation
#[derive(Debug)]
pub struct CogVideoOutput {
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
    /// Seed used
    pub seed: u64,
    /// Model variant used
    pub model_variant: CogVideoVariant,
}

impl CogVideoXGenerator {
    /// Create a new CogVideoX generator
    pub fn new(config: CogVideoConfig) -> Result<Self> {
        // Check for required packages
        let check = Command::new("python3")
            .args(["-c", "import diffusers; from diffusers import CogVideoXPipeline; print('ok')"])
            .output();

        match check {
            Ok(output) if output.status.success() => {}
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "diffusers[cogvideo]".to_string(),
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

    /// Check if CogVideoX is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "from diffusers import CogVideoXPipeline"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Generate video from prompt
    pub fn generate(&self, prompt: &CogVideoPrompt) -> Result<CogVideoOutput> {
        let output_path = self.config.output_dir.join(format!(
            "cogvideo_{}.mp4",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        let script = self.generate_script(prompt, &output_path)?;
        self.run_script(&script, &output_path)
    }

    fn generate_script(&self, prompt: &CogVideoPrompt, output_path: &Path) -> Result<String> {
        let negative = prompt.negative_prompt
            .as_deref()
            .unwrap_or("low quality, blurry, distorted, watermark");

        let seed_code = if let Some(seed) = self.config.seed {
            format!("generator = torch.Generator(device='cuda').manual_seed({})", seed)
        } else {
            "import random; seed = random.randint(0, 2**32-1); generator = torch.Generator(device='cuda').manual_seed(seed)".to_string()
        };

        let model_id = self.config.model_variant.model_id();
        let pipeline_class = self.config.model_variant.pipeline_class();

        let image_load = if let Some(ref img) = prompt.reference_image {
            format!(r#"
from diffusers.utils import load_image
image = load_image("{}")
"#, img.display())
        } else {
            String::new()
        };

        let generate_call = if prompt.reference_image.is_some() {
            "image=image,\n    "
        } else {
            ""
        };

        Ok(format!(r#"
import torch
from diffusers import {pipeline_class}
from diffusers.utils import export_to_video
import time
import json

# Setup
device = 'cuda:{device}' if torch.cuda.is_available() and {device} >= 0 else 'cpu'
print(f"Using device: {{device}}")

# Load model
print("Loading CogVideoX model ({model_id})...")
pipe = {pipeline_class}.from_pretrained(
    "{model_id}",
    torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
)

# Memory optimizations
if {cpu_offload}:
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)

if {vae_slicing}:
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

{image_load}

# Generator for reproducibility
{seed_code}
actual_seed = generator.initial_seed()

# Generate
print("Generating video...")
prompt = '''{prompt}'''
negative_prompt = '''{negative}'''

start_time = time.time()
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    {generate_call}num_videos_per_prompt=1,
    num_inference_steps={num_steps},
    num_frames={num_frames},
    guidance_scale={guidance_scale},
    generator=generator,
).frames[0]
generation_time = time.time() - start_time

# Export
print("Exporting video...")
export_to_video(video, '{output_path}', fps={fps})

# Result
result = {{
    'duration': {num_frames} / {fps},
    'resolution': [{width}, {height}],
    'fps': {fps},
    'generation_time': generation_time,
    'seed': actual_seed
}}
print('RESULT_JSON:' + json.dumps(result))
"#,
            pipeline_class = pipeline_class,
            device = self.config.device,
            model_id = model_id,
            cpu_offload = self.config.cpu_offload,
            vae_slicing = self.config.vae_slicing,
            image_load = image_load,
            seed_code = seed_code,
            prompt = prompt.prompt.replace("'", "\\'"),
            negative = negative.replace("'", "\\'"),
            generate_call = generate_call,
            num_steps = self.config.num_inference_steps,
            num_frames = self.config.num_frames,
            guidance_scale = self.config.guidance_scale,
            output_path = output_path.display(),
            fps = self.config.fps,
            width = self.config.resolution.0,
            height = self.config.resolution.1,
        ))
    }

    fn run_script(&self, script: &str, output_path: &Path) -> Result<CogVideoOutput> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("cogvideo_generate.py");
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

        Ok(CogVideoOutput {
            video_path: output_path.to_path_buf(),
            duration: data["duration"].as_f64().unwrap_or(0.0),
            resolution: (
                data["resolution"][0].as_u64().unwrap_or(720) as u32,
                data["resolution"][1].as_u64().unwrap_or(480) as u32,
            ),
            fps: data["fps"].as_u64().unwrap_or(8) as u32,
            generation_time: data["generation_time"].as_f64().unwrap_or(0.0),
            seed: data["seed"].as_u64().unwrap_or(0),
            model_variant: self.config.model_variant,
        })
    }

    /// Generate multiple variations
    pub fn generate_batch(
        &self,
        base_prompt: &CogVideoPrompt,
        variations: &[String],
    ) -> Result<Vec<CogVideoOutput>> {
        let mut results = Vec::new();
        for variation in variations {
            let mut prompt = base_prompt.clone();
            prompt.prompt = format!("{}, {}", base_prompt.prompt, variation);
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
        let config = CogVideoConfig::default();
        assert_eq!(config.fps, 8);
        assert_eq!(config.num_frames, 49);
        assert_eq!(config.model_variant, CogVideoVariant::V2B);
    }

    #[test]
    fn test_prompt_builder() {
        let prompt = CogVideoPrompt::text("test")
            .with_negative("bad quality");
        assert_eq!(prompt.prompt, "test");
        assert_eq!(prompt.negative_prompt, Some("bad quality".to_string()));
    }

    #[test]
    fn test_bio_prompts() {
        let gait = CogVideoBioPrompts::normal_gait();
        assert!(gait.prompt.contains("walking"));

        let tremor = CogVideoBioPrompts::resting_tremor();
        assert!(tremor.prompt.contains("tremor"));
    }

    #[test]
    fn test_model_variants() {
        assert_eq!(CogVideoVariant::V2B.model_id(), "THUDM/CogVideoX-2b");
        assert_eq!(CogVideoVariant::V5B.model_id(), "THUDM/CogVideoX-5b");
    }
}
