//! Level 3 video generation using Blender
//!
//! This module provides interfaces to generate actual video files with synthetic
//! gait, hand movement, and finger tapping data using Blender as a rendering engine.
//!
//! Level 3 generators produce:
//! - MP4 video files with rendered 3D animations
//! - Frame-by-frame ground truth (MediaPipe keypoints)
//! - Kinematic data (joint angles, velocities, etc.)
//!
//! These are intended for full pipeline testing with MediaPipe and video processing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;

/// Result type for video generation
pub type VideoResult<T> = Result<T, VideoGeneratorError>;

/// Errors that can occur during video generation
#[derive(Debug, thiserror::Error)]
pub enum VideoGeneratorError {
    #[error("Blender not found at path: {0}")]
    BlenderNotFound(String),

    #[error("Script not found: {0}")]
    ScriptNotFound(PathBuf),

    #[error("Blender execution failed: {0}")]
    BlenderExecutionError(String),

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

/// Parameters for gait video generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitVideoParams {
    pub duration_sec: f64,
    pub cadence: f64,           // steps per minute
    pub stride_length: f64,     // meters
    pub velocity: f64,          // m/s
    pub asymmetry: f64,         // 0-1
    pub shuffling_severity: f64, // 0-1
    pub festination: bool,
    pub freezing_probability: f64, // 0-1
    pub arm_swing_amplitude: f64,  // radians
    pub trunk_sway: f64,        // meters
    pub fps: u32,
    pub resolution: (u32, u32),
    pub camera_view: String,    // "side", "front", "oblique", "top"
    pub output_path: String,
    pub ground_truth_path: String,
    pub seed: u64,
    pub height: f64,            // meters
    pub render: bool,
}

impl Default for GaitVideoParams {
    fn default() -> Self {
        Self {
            duration_sec: 10.0,
            cadence: 100.0,
            stride_length: 0.7,
            velocity: 1.2,
            asymmetry: 0.0,
            shuffling_severity: 0.0,
            festination: false,
            freezing_probability: 0.0,
            arm_swing_amplitude: 0.3,
            trunk_sway: 0.02,
            fps: 30,
            resolution: (1920, 1080),
            camera_view: "side".to_string(),
            output_path: "/tmp/gait_output.mp4".to_string(),
            ground_truth_path: "/tmp/gait_ground_truth.json".to_string(),
            seed: 42,
            height: 1.75,
            render: true,
        }
    }
}

/// Parameters for hand video generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandVideoParams {
    pub duration_sec: f64,
    pub task: String,           // "rest", "tapping", "spiral", etc.
    pub tremor_frequency: f64,  // Hz
    pub tremor_amplitude: f64,  // meters
    pub tapping_frequency: f64, // Hz
    pub tapping_amplitude: f64, // meters
    pub bradykinesia_severity: f64, // 0-1
    pub dyskinesia_severity: f64,   // 0-1
    pub fps: u32,
    pub resolution: (u32, u32),
    pub camera_view: String,
    pub output_path: String,
    pub ground_truth_path: String,
    pub seed: u64,
    pub render: bool,
}

impl Default for HandVideoParams {
    fn default() -> Self {
        Self {
            duration_sec: 10.0,
            task: "rest".to_string(),
            tremor_frequency: 0.0,
            tremor_amplitude: 0.0,
            tapping_frequency: 2.0,
            tapping_amplitude: 0.05,
            bradykinesia_severity: 0.0,
            dyskinesia_severity: 0.0,
            fps: 30,
            resolution: (1280, 720),
            camera_view: "top".to_string(),
            output_path: "/tmp/hand_output.mp4".to_string(),
            ground_truth_path: "/tmp/hand_ground_truth.json".to_string(),
            seed: 42,
            render: true,
        }
    }
}

/// Parameters for finger tapping video generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TappingVideoParams {
    pub duration_sec: f64,
    pub target_frequency: f64,      // Hz
    pub amplitude_reduction: f64,   // 0-1
    pub frequency_reduction: f64,   // 0-1
    pub hesitations: u32,           // Number of freezing episodes
    pub irregularity: f64,          // 0-1
    pub fatigue_factor: f64,        // 0-1
    pub fps: u32,
    pub resolution: (u32, u32),
    pub output_path: String,
    pub ground_truth_path: String,
    pub seed: u64,
    pub render: bool,
}

impl Default for TappingVideoParams {
    fn default() -> Self {
        Self {
            duration_sec: 10.0,
            target_frequency: 2.0,
            amplitude_reduction: 0.0,
            frequency_reduction: 0.0,
            hesitations: 0,
            irregularity: 0.0,
            fatigue_factor: 0.0,
            fps: 60,
            resolution: (1280, 720),
            output_path: "/tmp/tapping_output.mp4".to_string(),
            ground_truth_path: "/tmp/tapping_ground_truth.json".to_string(),
            seed: 42,
            render: true,
        }
    }
}

/// Output from video generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoOutput {
    pub video_path: Option<PathBuf>,
    pub ground_truth_path: PathBuf,
    pub num_frames: usize,
    pub metadata: HashMap<String, String>,
}

/// MediaPipe pose keypoint data (33 landmarks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseGroundTruth {
    pub keypoints: Vec<Vec<[f64; 3]>>,  // frames x 33 landmarks x [x, y, z]
    pub format: String,
    pub num_frames: usize,
    pub num_landmarks: usize,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// MediaPipe hand keypoint data (21 landmarks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandGroundTruth {
    pub keypoints: Vec<Vec<[f64; 3]>>,  // frames x 21 landmarks x [x, y, z]
    pub format: String,
    pub num_frames: usize,
    pub num_landmarks: usize,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Finger tapping ground truth data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TappingGroundTruth {
    pub tap_events: Vec<TapEvent>,
    pub num_taps: usize,
    pub mean_frequency: f64,
    pub interval_coefficient_of_variation: f64,
    pub amplitude_trend: f64,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Single tap event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TapEvent {
    pub time: f64,
    pub amplitude: f64,
    pub index: usize,
}

/// Main interface for Level 3 video generation
pub struct Level3VideoGenerator {
    blender_path: PathBuf,
    script_dir: PathBuf,
}

impl Level3VideoGenerator {
    /// Create a new video generator
    ///
    /// # Arguments
    /// * `blender_path` - Path to Blender executable (default: "blender")
    /// * `script_dir` - Directory containing Python scripts (default: "tools/level3_video")
    pub fn new(blender_path: Option<PathBuf>, script_dir: Option<PathBuf>) -> Self {
        let blender_path = blender_path.unwrap_or_else(|| PathBuf::from("blender"));
        let script_dir = script_dir.unwrap_or_else(|| {
            // Try to find script directory relative to workspace root
            let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            current_dir.join("tools").join("level3_video")
        });

        Self {
            blender_path,
            script_dir,
        }
    }

    /// Check if Blender is available
    pub fn check_blender(&self) -> VideoResult<bool> {
        match Command::new(&self.blender_path)
            .arg("--version")
            .output()
        {
            Ok(output) => Ok(output.status.success()),
            Err(_) => Ok(false),
        }
    }

    /// Get Blender version string
    pub fn blender_version(&self) -> VideoResult<String> {
        let output = Command::new(&self.blender_path)
            .arg("--version")
            .output()
            .map_err(|_| VideoGeneratorError::BlenderNotFound(
                self.blender_path.display().to_string()
            ))?;

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Generate gait video
    pub fn generate_gait_video(&self, params: &GaitVideoParams) -> VideoResult<VideoOutput> {
        self.validate_gait_params(params)?;

        let script_path = self.script_dir.join("gait_generator.py");
        if !script_path.exists() {
            return Err(VideoGeneratorError::ScriptNotFound(script_path));
        }

        let params_json = serde_json::to_string(params)
            .map_err(|e| VideoGeneratorError::SerializationError(e.to_string()))?;

        println!("Calling Blender to generate gait video...");
        let output = Command::new(&self.blender_path)
            .args([
                "--background",
                "--python",
                script_path.to_str().unwrap(),
                "--",
                &params_json,
            ])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VideoGeneratorError::BlenderExecutionError(stderr.to_string()));
        }

        // Parse ground truth
        let ground_truth_path = PathBuf::from(&params.ground_truth_path);
        let video_path = if params.render {
            Some(PathBuf::from(&params.output_path))
        } else {
            None
        };

        let mut metadata = HashMap::new();
        metadata.insert("generator".to_string(), "gait_generator.py".to_string());
        metadata.insert("cadence".to_string(), params.cadence.to_string());
        metadata.insert("stride_length".to_string(), params.stride_length.to_string());

        Ok(VideoOutput {
            video_path,
            ground_truth_path,
            num_frames: (params.duration_sec * params.fps as f64) as usize,
            metadata,
        })
    }

    /// Generate hand movement video
    pub fn generate_hand_video(&self, params: &HandVideoParams) -> VideoResult<VideoOutput> {
        self.validate_hand_params(params)?;

        let script_path = self.script_dir.join("hand_generator.py");
        if !script_path.exists() {
            return Err(VideoGeneratorError::ScriptNotFound(script_path));
        }

        let params_json = serde_json::to_string(params)
            .map_err(|e| VideoGeneratorError::SerializationError(e.to_string()))?;

        println!("Calling Blender to generate hand video...");
        let output = Command::new(&self.blender_path)
            .args([
                "--background",
                "--python",
                script_path.to_str().unwrap(),
                "--",
                &params_json,
            ])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VideoGeneratorError::BlenderExecutionError(stderr.to_string()));
        }

        let ground_truth_path = PathBuf::from(&params.ground_truth_path);
        let video_path = if params.render {
            Some(PathBuf::from(&params.output_path))
        } else {
            None
        };

        let mut metadata = HashMap::new();
        metadata.insert("generator".to_string(), "hand_generator.py".to_string());
        metadata.insert("task".to_string(), params.task.clone());
        metadata.insert("tremor_frequency".to_string(), params.tremor_frequency.to_string());

        Ok(VideoOutput {
            video_path,
            ground_truth_path,
            num_frames: (params.duration_sec * params.fps as f64) as usize,
            metadata,
        })
    }

    /// Generate finger tapping video
    pub fn generate_tapping_video(&self, params: &TappingVideoParams) -> VideoResult<VideoOutput> {
        self.validate_tapping_params(params)?;

        let script_path = self.script_dir.join("tapping_generator.py");
        if !script_path.exists() {
            return Err(VideoGeneratorError::ScriptNotFound(script_path));
        }

        let params_json = serde_json::to_string(params)
            .map_err(|e| VideoGeneratorError::SerializationError(e.to_string()))?;

        println!("Calling Blender to generate tapping video...");
        let output = Command::new(&self.blender_path)
            .args([
                "--background",
                "--python",
                script_path.to_str().unwrap(),
                "--",
                &params_json,
            ])
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VideoGeneratorError::BlenderExecutionError(stderr.to_string()));
        }

        let ground_truth_path = PathBuf::from(&params.ground_truth_path);
        let video_path = if params.render {
            Some(PathBuf::from(&params.output_path))
        } else {
            None
        };

        let mut metadata = HashMap::new();
        metadata.insert("generator".to_string(), "tapping_generator.py".to_string());
        metadata.insert("target_frequency".to_string(), params.target_frequency.to_string());

        Ok(VideoOutput {
            video_path,
            ground_truth_path,
            num_frames: (params.duration_sec * params.fps as f64) as usize,
            metadata,
        })
    }

    /// Load pose ground truth from JSON file
    pub fn load_pose_ground_truth<P: AsRef<Path>>(&self, path: P) -> VideoResult<PoseGroundTruth> {
        let content = fs::read_to_string(path)
            .map_err(|e| VideoGeneratorError::OutputReadError(e.to_string()))?;
        let gt: PoseGroundTruth = serde_json::from_str(&content)?;
        Ok(gt)
    }

    /// Load hand ground truth from JSON file
    pub fn load_hand_ground_truth<P: AsRef<Path>>(&self, path: P) -> VideoResult<HandGroundTruth> {
        let content = fs::read_to_string(path)
            .map_err(|e| VideoGeneratorError::OutputReadError(e.to_string()))?;
        let gt: HandGroundTruth = serde_json::from_str(&content)?;
        Ok(gt)
    }

    /// Load tapping ground truth from JSON file
    pub fn load_tapping_ground_truth<P: AsRef<Path>>(&self, path: P) -> VideoResult<TappingGroundTruth> {
        let content = fs::read_to_string(path)
            .map_err(|e| VideoGeneratorError::OutputReadError(e.to_string()))?;
        let gt: TappingGroundTruth = serde_json::from_str(&content)?;
        Ok(gt)
    }

    // Validation methods

    fn validate_gait_params(&self, params: &GaitVideoParams) -> VideoResult<()> {
        if params.duration_sec <= 0.0 {
            return Err(VideoGeneratorError::InvalidParameter(
                "duration_sec must be positive".to_string()
            ));
        }
        if params.cadence <= 0.0 {
            return Err(VideoGeneratorError::InvalidParameter(
                "cadence must be positive".to_string()
            ));
        }
        if params.asymmetry < 0.0 || params.asymmetry > 1.0 {
            return Err(VideoGeneratorError::InvalidParameter(
                "asymmetry must be 0-1".to_string()
            ));
        }
        Ok(())
    }

    fn validate_hand_params(&self, params: &HandVideoParams) -> VideoResult<()> {
        if params.duration_sec <= 0.0 {
            return Err(VideoGeneratorError::InvalidParameter(
                "duration_sec must be positive".to_string()
            ));
        }
        if params.tremor_frequency < 0.0 {
            return Err(VideoGeneratorError::InvalidParameter(
                "tremor_frequency cannot be negative".to_string()
            ));
        }
        Ok(())
    }

    fn validate_tapping_params(&self, params: &TappingVideoParams) -> VideoResult<()> {
        if params.duration_sec <= 0.0 {
            return Err(VideoGeneratorError::InvalidParameter(
                "duration_sec must be positive".to_string()
            ));
        }
        if params.target_frequency <= 0.0 {
            return Err(VideoGeneratorError::InvalidParameter(
                "target_frequency must be positive".to_string()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_gait_params() {
        let params = GaitVideoParams::default();
        assert_eq!(params.duration_sec, 10.0);
        assert_eq!(params.fps, 30);
        assert_eq!(params.cadence, 100.0);
    }

    #[test]
    fn test_default_hand_params() {
        let params = HandVideoParams::default();
        assert_eq!(params.task, "rest");
        assert_eq!(params.fps, 30);
    }

    #[test]
    fn test_default_tapping_params() {
        let params = TappingVideoParams::default();
        assert_eq!(params.target_frequency, 2.0);
        assert_eq!(params.fps, 60);
    }

    #[test]
    fn test_generator_creation() {
        let generator = Level3VideoGenerator::new(None, None);
        assert!(generator.blender_path.to_str().is_some());
    }
}
