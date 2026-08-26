//! Advanced Synthetic Media Generation
//!
//! This module provides integrations with state-of-the-art open source tools for
//! creating highly realistic synthetic biosignal data with perfect ground truth.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    DPB Synthetic Media Pipeline                  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
//! │  │   MOTION CORE   │    │   VIDEO RENDER  │    │ AUDIO SYNTH  │ │
//! │  ├─────────────────┤    ├─────────────────┤    ├──────────────┤ │
//! │  │ • MuJoCo        │───▶│ • Blender       │    │ • Chatterbox │ │
//! │  │ • OpenSim       │    │ • LTX-Video     │    │ • F5-TTS     │ │
//! │  │ • Pathology     │    │ • Wan 2.2       │    │ • Kokoro     │ │
//! │  └────────┬────────┘    └────────┬────────┘    └──────┬───────┘ │
//! │           └──────────────────────┼─────────────────────┘         │
//! │                                  ▼                               │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │                  BIOSIGNAL EXTRACTION                      │  │
//! │  │  MediaPipe Pose → Gait/Tremor kinematics                  │  │
//! │  │  Audio Analysis → Voice biomarkers                        │  │
//! │  │  Ground Truth   → Perfect labels from simulation params   │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Supported Tools
//!
//! ## Motion Simulation
//! - **MuJoCo**: Physics-accurate human motion with musculoskeletal models
//! - **OpenSim**: Biomechanical gait analysis with pathological tremor support
//! - **PyBullet**: Lightweight physics simulation (zlib license)
//!
//! ## Video Generation
//! - **Blender**: Procedural animation with Python scripting
//! - **LTX-Video**: Fast diffusion-based video generation (DiT architecture)
//! - **CogVideoX**: 2B/5B parameter video generation (Apache 2.0)
//! - **Wan 2.2**: State-of-the-art MoE diffusion for highest quality
//!
//! ## Audio Generation
//! - **Chatterbox**: Neural TTS with emotion control (MIT licensed)
//! - **F5-TTS**: Fast zero-shot voice cloning (MIT licensed)
//! - **Bark**: Non-speech audio, music, sound effects (MIT licensed)
//! - **Kokoro**: Lightweight 82M param model (Apache 2.0)
//! - **XTTS-v2**: Voice cloning with 6-second samples
//!
//! ## Pose Extraction
//! - **MediaPipe**: Real-time 33-landmark pose estimation
//! - **OpenPose**: 135-keypoint body+hands+face (research license)
//!
//! ## Human Body Models
//! - **SMPL-X**: Parametric body model with hands and face (10,475 vertices)
//!
//! # Usage
//!
//! ```rust,ignore
//! use dpb_synth::media::{
//!     MediaPipeline, PipelineConfig, MotionBackend, VideoBackend, AudioBackend
//! };
//!
//! // Create unified pipeline
//! let config = PipelineConfig::default()
//!     .with_motion_backend(MotionBackend::MuJoCo)
//!     .with_video_backend(VideoBackend::Blender)
//!     .with_audio_backend(AudioBackend::Chatterbox);
//!
//! let pipeline = MediaPipeline::new(config)?;
//! ```

pub mod mujoco;
pub mod opensim;
pub mod pybullet;
pub mod blender;
pub mod chatterbox;
pub mod f5tts;
pub mod bark;
pub mod ltx_video;
pub mod cogvideo;
pub mod mediapipe;
pub mod openpose;
pub mod smplx;
pub mod pipeline;

pub use mujoco::{MuJoCoSimulator, MuJoCoConfig, MusculoskeletalModel, MotionTrajectory};
pub use opensim::{OpenSimBridge, GaitModel, TremorModel, PathologyParams};
pub use pybullet::{PyBulletSimulator, PyBulletConfig, PyBulletGaitParams, PyBulletTremorParams};
pub use blender::{BlenderRenderer, BlenderConfig, RenderOutput, AnimationScript};
pub use chatterbox::{ChatterboxTTS, VoiceConfig, EmotionControl, SpeechOutput};
pub use f5tts::{F5TTSGenerator, F5TTSConfig, F5TTSPrompt, PathologicalVoiceParams};
pub use bark::{BarkGenerator, BarkConfig, BarkPrompt, BarkSpeaker};
pub use ltx_video::{LTXVideoGenerator, DiffusionConfig, VideoPrompt};
pub use cogvideo::{CogVideoXGenerator, CogVideoConfig, CogVideoPrompt, CogVideoVariant};
pub use mediapipe::{MediaPipeExtractor, PoseEstimate, HandLandmarks, FaceMesh};
pub use openpose::{OpenPoseExtractor, OpenPoseConfig, OpenPosePose, OpenPoseFrame};
pub use smplx::{SMPLXModel, SMPLXConfig, BodyPose, BodyShape, SMPLXAnimation};
pub use pipeline::{MediaPipeline, PipelineConfig, SyntheticScenario, ScenarioOutput};

use thiserror::Error;
use std::path::PathBuf;

/// Errors from media generation
#[derive(Debug, Error)]
pub enum MediaError {
    #[error("Tool not found: {tool}. Install from: {install_url}")]
    ToolNotFound { tool: String, install_url: String },

    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Python error: {0}")]
    PythonError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("GPU not available for: {0}")]
    GpuUnavailable(String),

    #[error("Timeout after {0} seconds")]
    Timeout(u64),
}

pub type Result<T> = std::result::Result<T, MediaError>;

/// Motion simulation backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MotionBackend {
    /// MuJoCo physics simulation (recommended)
    #[default]
    MuJoCo,
    /// OpenSim biomechanical simulation
    OpenSim,
    /// PyBullet lightweight physics (zlib license)
    PyBullet,
    /// SMPL-X parametric body model
    SMPLX,
    /// Procedural animation (no physics)
    Procedural,
}

/// Video rendering backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoBackend {
    /// Blender 3D rendering (highest quality)
    #[default]
    Blender,
    /// LTX-Video diffusion (fast, good quality)
    LTXVideo,
    /// CogVideoX diffusion (2B/5B models, Apache 2.0)
    CogVideoX,
    /// Wan 2.2 diffusion (best quality, slower)
    Wan22,
    /// Simple 2D skeleton rendering (fastest)
    Skeleton2D,
}

/// Audio synthesis backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioBackend {
    /// Chatterbox TTS (MIT, emotion control)
    #[default]
    Chatterbox,
    /// F5-TTS (fast zero-shot cloning, MIT)
    F5TTS,
    /// Bark (non-speech audio, music, effects, MIT)
    Bark,
    /// Kokoro (lightweight, Apache 2.0)
    Kokoro,
    /// XTTS-v2 (voice cloning)
    XTTSv2,
    /// espeak-ng (fallback, always available)
    EspeakNG,
}

/// Common ground truth data for all generated media
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct MediaGroundTruth {
    /// Timestamps in seconds
    pub timestamps: Vec<f64>,
    /// Joint angles (if motion data)
    pub joint_angles: Option<JointAngles>,
    /// 3D joint positions (if motion data)
    pub joint_positions: Option<Vec<JointPositions3D>>,
    /// Tremor parameters (if applicable)
    pub tremor: Option<TremorGroundTruth>,
    /// Gait parameters (if applicable)
    pub gait: Option<GaitGroundTruth>,
    /// Voice parameters (if applicable)
    pub voice: Option<VoiceGroundTruth>,
    /// Clinical severity scores
    pub severity: Option<SeverityScores>,
}

/// Joint angles for a single frame
#[derive(Debug, Clone)]
pub struct JointAngles {
    /// Hip flexion/extension angles (left, right) in radians
    pub hip_flexion: [f32; 2],
    /// Knee flexion angles (left, right) in radians
    pub knee_flexion: [f32; 2],
    /// Ankle dorsiflexion angles (left, right) in radians
    pub ankle_dorsiflexion: [f32; 2],
    /// Shoulder flexion angles (left, right) in radians
    pub shoulder_flexion: [f32; 2],
    /// Elbow flexion angles (left, right) in radians
    pub elbow_flexion: [f32; 2],
    /// Wrist flexion angles (left, right) in radians
    pub wrist_flexion: [f32; 2],
    /// Trunk flexion angle in radians
    pub trunk_flexion: f32,
}

/// 3D joint positions for a single frame
#[derive(Debug, Clone)]
pub struct JointPositions3D {
    /// Position in meters [x, y, z]
    pub position: [f32; 3],
    /// Joint name (e.g., "left_ankle", "right_wrist")
    pub joint_name: String,
}

/// Ground truth for tremor
#[derive(Debug, Clone)]
pub struct TremorGroundTruth {
    /// Tremor frequency in Hz
    pub frequency: f32,
    /// Tremor amplitude in meters
    pub amplitude: f32,
    /// Tremor type
    pub tremor_type: TremorType,
    /// Affected joints
    pub affected_joints: Vec<String>,
    /// Time-varying amplitude envelope
    pub amplitude_envelope: Vec<f32>,
}

/// Types of tremor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TremorType {
    /// Resting tremor (typical in Parkinson's)
    Resting,
    /// Action tremor
    Action,
    /// Postural tremor
    Postural,
    /// Intention tremor (cerebellar)
    Intention,
    /// Essential tremor
    Essential,
}

/// Ground truth for gait
#[derive(Debug, Clone)]
pub struct GaitGroundTruth {
    /// Stride length in meters
    pub stride_length: f32,
    /// Cadence in steps per minute
    pub cadence: f32,
    /// Gait speed in m/s
    pub speed: f32,
    /// Double support time as fraction of gait cycle
    pub double_support_fraction: f32,
    /// Step width in meters
    pub step_width: f32,
    /// Arm swing asymmetry (0 = symmetric)
    pub arm_swing_asymmetry: f32,
    /// Festination present
    pub festination: bool,
    /// Freezing of gait episodes (start_time, duration)
    pub freezing_episodes: Vec<(f64, f64)>,
}

/// Ground truth for voice
#[derive(Debug, Clone)]
pub struct VoiceGroundTruth {
    /// Fundamental frequency in Hz
    pub f0: Vec<f32>,
    /// Jitter (cycle-to-cycle variation)
    pub jitter: f32,
    /// Shimmer (amplitude variation)
    pub shimmer: f32,
    /// Harmonics-to-noise ratio in dB
    pub hnr: f32,
    /// Maximum phonation time in seconds
    pub mpt: Option<f32>,
    /// Speech rate in syllables per second
    pub speech_rate: f32,
    /// Hypophonia severity (0-1)
    pub hypophonia: f32,
    /// Dysarthria severity (0-1)
    pub dysarthria: f32,
    /// Phoneme boundaries (time, phoneme)
    pub phonemes: Vec<(f64, String)>,
}

/// Clinical severity scores
#[derive(Debug, Clone)]
pub struct SeverityScores {
    /// Overall UPDRS motor score (0-108)
    pub updrs_motor: Option<f32>,
    /// Tremor subscore (0-28)
    pub updrs_tremor: Option<f32>,
    /// Rigidity subscore (0-20)
    pub updrs_rigidity: Option<f32>,
    /// Bradykinesia subscore (0-36)
    pub updrs_bradykinesia: Option<f32>,
    /// Gait/posture subscore (0-16)
    pub updrs_gait_posture: Option<f32>,
    /// Hoehn & Yahr stage (0-5)
    pub hoehn_yahr: Option<f32>,
}

impl Default for JointAngles {
    fn default() -> Self {
        Self {
            hip_flexion: [0.0; 2],
            knee_flexion: [0.0; 2],
            ankle_dorsiflexion: [0.0; 2],
            shoulder_flexion: [0.0; 2],
            elbow_flexion: [0.0; 2],
            wrist_flexion: [0.0; 2],
            trunk_flexion: 0.0,
        }
    }
}


/// Check if a tool is available in the system PATH
pub fn check_tool_available(tool: &str) -> bool {
    std::process::Command::new("which")
        .arg(tool)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if Python package is installed
pub fn check_python_package(package: &str) -> bool {
    std::process::Command::new("python3")
        .args(["-c", &format!("import {}", package)])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Get available backends based on installed tools
pub fn get_available_backends() -> AvailableBackends {
    AvailableBackends {
        motion: vec![
            (MotionBackend::Procedural, true),
            (MotionBackend::MuJoCo, check_python_package("mujoco")),
            (MotionBackend::OpenSim, check_python_package("opensim")),
            (MotionBackend::PyBullet, check_python_package("pybullet")),
            (MotionBackend::SMPLX, check_python_package("smplx")),
        ],
        video: vec![
            (VideoBackend::Skeleton2D, true),
            (VideoBackend::Blender, check_tool_available("blender")),
            (VideoBackend::LTXVideo, check_python_package("diffusers")),
            (VideoBackend::CogVideoX, check_python_package("diffusers")),
            (VideoBackend::Wan22, check_python_package("diffusers")),
        ],
        audio: vec![
            (AudioBackend::EspeakNG, check_tool_available("espeak-ng")),
            (AudioBackend::Chatterbox, check_python_package("chatterbox")),
            (AudioBackend::F5TTS, check_python_package("f5_tts")),
            (AudioBackend::Bark, check_python_package("bark")),
            (AudioBackend::Kokoro, check_python_package("kokoro")),
            (AudioBackend::XTTSv2, check_python_package("TTS")),
        ],
    }
}

/// Available backends report
#[derive(Debug)]
pub struct AvailableBackends {
    pub motion: Vec<(MotionBackend, bool)>,
    pub video: Vec<(VideoBackend, bool)>,
    pub audio: Vec<(AudioBackend, bool)>,
}

impl AvailableBackends {
    /// Get the best available motion backend
    pub fn best_motion(&self) -> MotionBackend {
        for (backend, available) in &self.motion {
            if *available && *backend != MotionBackend::Procedural {
                return *backend;
            }
        }
        MotionBackend::Procedural
    }

    /// Get the best available video backend
    pub fn best_video(&self) -> VideoBackend {
        for (backend, available) in &self.video {
            if *available && *backend != VideoBackend::Skeleton2D {
                return *backend;
            }
        }
        VideoBackend::Skeleton2D
    }

    /// Get the best available audio backend
    pub fn best_audio(&self) -> AudioBackend {
        for (backend, available) in &self.audio {
            if *available && *backend != AudioBackend::EspeakNG {
                return *backend;
            }
        }
        AudioBackend::EspeakNG
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        let backends = get_available_backends();
        // At minimum, procedural and skeleton2D should be available
        assert!(backends.motion.iter().any(|(b, a)| *b == MotionBackend::Procedural && *a));
        assert!(backends.video.iter().any(|(b, a)| *b == VideoBackend::Skeleton2D && *a));
    }

    #[test]
    fn test_default_ground_truth() {
        let gt = MediaGroundTruth::default();
        assert!(gt.timestamps.is_empty());
        assert!(gt.joint_angles.is_none());
    }
}
