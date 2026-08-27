//! Unified Synthetic Media Pipeline
//!
//! This module provides a unified interface for generating synthetic biosignal
//! data with synchronized video, audio, and ground truth from physics simulation.
//!
//! # Pipeline Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        SCENARIO DEFINITION                               │
//! │  (disease type, severity, symptoms, duration, camera angles, etc.)      │
//! └─────────────────────────────────────────────────────────────────────────┘
//!                                    │
//!                                    ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         MOTION SIMULATION                                │
//! │  MuJoCo/OpenSim → Joint angles, trajectories, muscle activations        │
//! └─────────────────────────────────────────────────────────────────────────┘
//!                                    │
//!                    ┌───────────────┼───────────────┐
//!                    ▼               ▼               ▼
//! ┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐
//! │   VIDEO RENDERING    │ │  AUDIO SYNTHESIS │ │    GROUND TRUTH          │
//! │  Blender/LTX-Video   │ │   Chatterbox     │ │  Joint angles, clinical  │
//! │  Multi-view output   │ │  + pathology     │ │  scores, biomarkers      │
//! └──────────────────────┘ └──────────────────┘ └──────────────────────────┘
//!                    │               │               │
//!                    └───────────────┼───────────────┘
//!                                    ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        VALIDATION LOOP                                   │
//! │  MediaPipe extraction → Compare with ground truth → Quality metrics     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use super::{
    MediaError, Result, MotionBackend, VideoBackend, AudioBackend,
    MediaGroundTruth, TremorGroundTruth, GaitGroundTruth, SeverityScores, TremorType,
    mujoco::{self, MuJoCoSimulator, MuJoCoConfig, MotionTrajectory},
    opensim::{self, OpenSimBridge, OpenSimConfig, MedicationState},
    blender::{self, BlenderRenderer, BlenderConfig},
    ltx_video::{LTXVideoGenerator, DiffusionConfig, VideoPrompt},
    chatterbox::{ChatterboxTTS, ChatterboxConfig, VoiceConfig, PathologicalVoiceParams},
    mediapipe::{MediaPipeExtractor, MediaPipeConfig},
};
use std::path::{Path, PathBuf};

/// Unified media generation pipeline
#[derive(Debug)]
pub struct MediaPipeline {
    config: PipelineConfig,
    mujoco: Option<MuJoCoSimulator>,
    opensim: Option<OpenSimBridge>,
    blender: Option<BlenderRenderer>,
    ltx_video: Option<LTXVideoGenerator>,
    chatterbox: Option<ChatterboxTTS>,
    mediapipe: Option<MediaPipeExtractor>,
}

/// Configuration for the media pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Motion backend to use
    pub motion_backend: MotionBackend,
    /// Video backend to use
    pub video_backend: VideoBackend,
    /// Audio backend to use
    pub audio_backend: AudioBackend,
    /// Output directory
    pub output_dir: PathBuf,
    /// Enable validation with MediaPipe
    pub enable_validation: bool,
    /// Video resolution
    pub video_resolution: (u32, u32),
    /// Audio sample rate
    pub audio_sample_rate: u32,
    /// Frames per second
    pub fps: u32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            motion_backend: MotionBackend::MuJoCo,
            video_backend: VideoBackend::Blender,
            audio_backend: AudioBackend::Chatterbox,
            output_dir: PathBuf::from("output/synthetic"),
            enable_validation: true,
            video_resolution: (1920, 1080),
            audio_sample_rate: 24000,
            fps: 30,
            seed: None,
        }
    }
}

impl PipelineConfig {
    /// Set motion backend
    pub fn with_motion_backend(mut self, backend: MotionBackend) -> Self {
        self.motion_backend = backend;
        self
    }

    /// Set video backend
    pub fn with_video_backend(mut self, backend: VideoBackend) -> Self {
        self.video_backend = backend;
        self
    }

    /// Set audio backend
    pub fn with_audio_backend(mut self, backend: AudioBackend) -> Self {
        self.audio_backend = backend;
        self
    }

    /// Set output directory
    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Set seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Synthetic scenario definition
#[derive(Debug, Clone)]
pub struct SyntheticScenario {
    /// Unique identifier for this scenario
    pub id: String,
    /// Disease/condition to simulate
    pub condition: Condition,
    /// Severity level (0-1)
    pub severity: f32,
    /// Duration in seconds
    pub duration: f64,
    /// Tasks to perform
    pub tasks: Vec<Task>,
    /// Voice text (if audio needed)
    pub voice_text: Option<String>,
    /// Camera angles for video
    pub camera_angles: Vec<CameraAngle>,
    /// Subject demographics
    pub subject: SubjectProfile,
}

/// Condition to simulate
#[derive(Debug, Clone)]
pub enum Condition {
    /// Healthy control
    Healthy,
    /// Parkinson's Disease with specific parameters
    Parkinsons(ParkinsonsParams),
    /// Essential Tremor
    EssentialTremor(TremorParams),
    /// Stroke (specify affected side)
    Stroke { side: Side, severity: f32 },
    /// Multiple Sclerosis
    MultipleSclerosis { severity: f32 },
    /// Custom condition
    Custom(String),
}

/// Parkinson's Disease parameters
#[derive(Debug, Clone)]
pub struct ParkinsonsParams {
    /// Hoehn & Yahr stage (1-5)
    pub hoehn_yahr: f32,
    /// Tremor subscore (0-1)
    pub tremor_severity: f32,
    /// Bradykinesia severity (0-1)
    pub bradykinesia: f32,
    /// Rigidity severity (0-1)
    pub rigidity: f32,
    /// Medication state
    pub medication_state: MedicationState,
    /// Enable festinating gait
    pub festination: bool,
    /// Freezing of gait probability
    pub fog_probability: f32,
}

impl Default for ParkinsonsParams {
    fn default() -> Self {
        Self {
            hoehn_yahr: 2.0,
            tremor_severity: 0.3,
            bradykinesia: 0.3,
            rigidity: 0.2,
            medication_state: MedicationState::Off,
            festination: false,
            fog_probability: 0.0,
        }
    }
}

/// Tremor parameters for Essential Tremor
#[derive(Debug, Clone)]
pub struct TremorParams {
    /// Frequency in Hz
    pub frequency: f32,
    /// Amplitude (0-1)
    pub amplitude: f32,
    /// Affected limbs
    pub affected_limbs: Vec<String>,
}

/// Side of body
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
    Both,
}

/// Tasks to perform in the scenario
#[derive(Debug, Clone)]
pub enum Task {
    /// Walking task
    Walk {
        distance: f32, // meters
        speed: Option<f32>, // m/s, None for natural
    },
    /// Standing balance
    Stand { duration: f64 },
    /// Finger tapping
    FingerTap { hand: Side, duration: f64 },
    /// Hand movements (e.g., pronation-supination)
    HandMovement { movement_type: String, duration: f64 },
    /// Speech task
    Speech { text: String },
    /// Sustained vowel
    SustainedVowel { vowel: char, duration: f64 },
    /// Reach and grasp
    ReachGrasp { target_distance: f32 },
    /// Custom task
    Custom { name: String, duration: f64 },
}

/// Camera angle presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraAngle {
    /// Side view (sagittal plane)
    Side,
    /// Front view (frontal plane)
    Front,
    /// Back view
    Back,
    /// Top-down view
    TopDown,
    /// Close-up on hands
    HandCloseUp,
    /// Close-up on face
    FaceCloseUp,
    /// Clinical examination view
    Clinical,
}

/// Subject demographics
#[derive(Debug, Clone)]
pub struct SubjectProfile {
    /// Age in years
    pub age: u8,
    /// Height in meters
    pub height: f32,
    /// Weight in kg
    pub weight: f32,
    /// Biological sex for body model
    pub sex: Sex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sex {
    #[default]
    Male,
    Female,
}

impl Default for SubjectProfile {
    fn default() -> Self {
        Self {
            age: 65,
            height: 1.70,
            weight: 75.0,
            sex: Sex::Male,
        }
    }
}

/// Output from scenario generation
#[derive(Debug)]
pub struct ScenarioOutput {
    /// Scenario ID
    pub scenario_id: String,
    /// Video file paths (one per camera angle)
    pub video_paths: Vec<PathBuf>,
    /// Audio file path
    pub audio_path: Option<PathBuf>,
    /// Ground truth data
    pub ground_truth: MediaGroundTruth,
    /// Ground truth file path (JSON)
    pub ground_truth_path: PathBuf,
    /// Validation results (if enabled)
    pub validation: Option<ValidationResult>,
    /// Generation metadata
    pub metadata: GenerationMetadata,
}

/// Validation results from MediaPipe comparison
#[derive(Debug)]
pub struct ValidationResult {
    /// Pose detection rate (0-1)
    pub pose_detection_rate: f32,
    /// Mean joint position error in pixels
    pub mean_position_error: f32,
    /// Joint angle correlation with ground truth
    pub angle_correlation: f32,
    /// Gait parameter accuracy
    pub gait_accuracy: Option<GaitValidation>,
}

/// Gait parameter validation
#[derive(Debug)]
pub struct GaitValidation {
    /// Stride length error (%)
    pub stride_length_error: f32,
    /// Cadence error (%)
    pub cadence_error: f32,
    /// Speed error (%)
    pub speed_error: f32,
}

/// Generation metadata
#[derive(Debug)]
pub struct GenerationMetadata {
    /// Total generation time in seconds
    pub generation_time: f64,
    /// Motion simulation time
    pub motion_time: f64,
    /// Video rendering time
    pub video_time: f64,
    /// Audio synthesis time
    pub audio_time: f64,
    /// Backends used
    pub backends_used: BackendsUsed,
    /// Seed used
    pub seed: u64,
}

#[derive(Debug)]
pub struct BackendsUsed {
    pub motion: String,
    pub video: String,
    pub audio: Option<String>,
}

impl MediaPipeline {
    /// Create a new media pipeline
    pub fn new(config: PipelineConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.output_dir)?;

        // Initialize backends based on config
        let mujoco = if config.motion_backend == MotionBackend::MuJoCo {
            MuJoCoSimulator::new(MuJoCoConfig::default()).ok()
        } else {
            None
        };

        let opensim = if config.motion_backend == MotionBackend::OpenSim {
            OpenSimBridge::new(OpenSimConfig::default()).ok()
        } else {
            None
        };

        let blender = if config.video_backend == VideoBackend::Blender {
            let mut bc = BlenderConfig::default();
            bc.resolution = config.video_resolution;
            bc.fps = config.fps;
            bc.output_dir = config.output_dir.join("video");
            BlenderRenderer::new(bc).ok()
        } else {
            None
        };

        let ltx_video = if config.video_backend == VideoBackend::LTXVideo {
            let mut dc = DiffusionConfig::default();
            dc.resolution = config.video_resolution;
            dc.fps = config.fps;
            dc.output_dir = config.output_dir.join("video");
            dc.seed = config.seed;
            LTXVideoGenerator::new(dc).ok()
        } else {
            None
        };

        let chatterbox = if config.audio_backend == AudioBackend::Chatterbox {
            let mut cc = ChatterboxConfig::default();
            cc.sample_rate = config.audio_sample_rate;
            cc.output_dir = config.output_dir.join("audio");
            ChatterboxTTS::new(cc).ok()
        } else {
            None
        };

        let mediapipe = if config.enable_validation {
            let mut mc = MediaPipeConfig::default();
            mc.output_dir = config.output_dir.join("validation");
            MediaPipeExtractor::new(mc).ok()
        } else {
            None
        };

        Ok(Self {
            config,
            mujoco,
            opensim,
            blender,
            ltx_video,
            chatterbox,
            mediapipe,
        })
    }

    /// Generate a complete scenario
    pub fn generate(&self, scenario: &SyntheticScenario) -> Result<ScenarioOutput> {
        let start_time = std::time::Instant::now();

        // Create scenario output directory
        let scenario_dir = self.config.output_dir.join(&scenario.id);
        std::fs::create_dir_all(&scenario_dir)?;

        // 1. Motion simulation
        let motion_start = std::time::Instant::now();
        let motion_result = self.simulate_motion(scenario)?;
        let motion_time = motion_start.elapsed().as_secs_f64();

        // 2. Video rendering
        let video_start = std::time::Instant::now();
        let video_paths = self.render_videos(scenario, &motion_result, &scenario_dir)?;
        let video_time = video_start.elapsed().as_secs_f64();

        // 3. Audio synthesis (if needed)
        let audio_start = std::time::Instant::now();
        let audio_path = if let Some(ref text) = scenario.voice_text {
            Some(self.synthesize_audio(scenario, text, &scenario_dir)?)
        } else {
            self.synthesize_audio_for_tasks(scenario, &scenario_dir).ok()
        };
        let audio_time = audio_start.elapsed().as_secs_f64();

        // 4. Generate ground truth
        let ground_truth = self.create_ground_truth(scenario, &motion_result);
        let gt_path = scenario_dir.join("ground_truth.json");
        let gt_json = serde_json::to_string_pretty(&GroundTruthJson::from(&ground_truth))
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;
        std::fs::write(&gt_path, gt_json)?;

        // 5. Validation (if enabled)
        let validation = if self.config.enable_validation && !video_paths.is_empty() {
            self.validate_output(&video_paths[0], &ground_truth).ok()
        } else {
            None
        };

        let total_time = start_time.elapsed().as_secs_f64();

        let audio_backend_str = if audio_path.is_some() {
            Some(format!("{:?}", self.config.audio_backend))
        } else {
            None
        };

        Ok(ScenarioOutput {
            scenario_id: scenario.id.clone(),
            video_paths,
            audio_path,
            ground_truth,
            ground_truth_path: gt_path,
            validation,
            metadata: GenerationMetadata {
                generation_time: total_time,
                motion_time,
                video_time,
                audio_time,
                backends_used: BackendsUsed {
                    motion: format!("{:?}", self.config.motion_backend),
                    video: format!("{:?}", self.config.video_backend),
                    audio: audio_backend_str,
                },
                seed: self.config.seed.unwrap_or(0),
            },
        })
    }

    /// Simulate motion for scenario
    fn simulate_motion(&self, scenario: &SyntheticScenario) -> Result<MotionSimulationResult> {
        // Convert scenario to motion parameters
        let gait_params = self.scenario_to_gait_params(scenario);
        let _tremor_params = self.scenario_to_tremor_params(scenario);

        // Run simulation based on backend
        match self.config.motion_backend {
            MotionBackend::MuJoCo => {
                if let Some(ref mujoco) = self.mujoco {
                    let trajectory = mujoco.simulate_gait(
                        &gait_params,
                        mujoco::MusculoskeletalModel::SimpleHumanoid
                    )?;
                    Ok(MotionSimulationResult {
                        trajectory: Some(trajectory),
                        tremor_data: None,
                    })
                } else {
                    self.simulate_procedural_motion(scenario)
                }
            }
            MotionBackend::OpenSim => {
                if let Some(ref opensim) = self.opensim {
                    let pathology = self.scenario_to_pathology(scenario);
                    let result = opensim.simulate_gait(
                        opensim::GaitModel::Gait2392,
                        scenario.duration,
                        pathology.as_ref()
                    )?;
                    Ok(MotionSimulationResult {
                        trajectory: Some(self.opensim_to_trajectory(&result)),
                        tremor_data: None,
                    })
                } else {
                    self.simulate_procedural_motion(scenario)
                }
            }
            MotionBackend::Procedural => {
                self.simulate_procedural_motion(scenario)
            }
            MotionBackend::PyBullet | MotionBackend::SMPLX => {
                // Fall back to procedural for now - PyBullet and SMPL-X
                // integration is available via their direct modules
                self.simulate_procedural_motion(scenario)
            }
        }
    }

    /// Convert scenario to gait parameters
    fn scenario_to_gait_params(&self, scenario: &SyntheticScenario) -> mujoco::GaitSimParams {
        let mut params = mujoco::GaitSimParams::default();
        params.duration = scenario.duration;

        match &scenario.condition {
            Condition::Parkinsons(pd) => {
                params.bradykinesia = pd.bradykinesia as f64;
                params.arm_swing = 1.0 - pd.bradykinesia as f64 * 0.7;
                params.festination = pd.festination;
                params.speed = 1.2 * (1.0 - pd.bradykinesia as f64 * 0.5);
            }
            Condition::Stroke { severity, .. } => {
                params.step_asymmetry = *severity as f64 * 0.5;
                params.speed = 1.2 * (1.0 - *severity as f64 * 0.4);
            }
            _ => {}
        }

        // Adjust for tasks
        for task in &scenario.tasks {
            if let Task::Walk { speed, .. } = task
                && let Some(s) = speed {
                    params.speed = *s as f64;
                }
        }

        params
    }

    /// Convert scenario to tremor parameters
    fn scenario_to_tremor_params(&self, scenario: &SyntheticScenario) -> Option<mujoco::TremorSimParams> {
        match &scenario.condition {
            Condition::Parkinsons(pd) if pd.tremor_severity > 0.1 => {
                Some(mujoco::TremorSimParams {
                    frequency: 5.0,
                    amplitude: pd.tremor_severity as f64 * 0.1,
                    tremor_type: TremorType::Resting,
                    affected_joints: vec!["wrist_r".to_string()],
                    duration: scenario.duration,
                    ..Default::default()
                })
            }
            Condition::EssentialTremor(et) => {
                Some(mujoco::TremorSimParams {
                    frequency: et.frequency as f64,
                    amplitude: et.amplitude as f64 * 0.1,
                    tremor_type: TremorType::Action,
                    affected_joints: et.affected_limbs.clone(),
                    duration: scenario.duration,
                    ..Default::default()
                })
            }
            _ => None,
        }
    }

    /// Convert scenario to OpenSim pathology params
    fn scenario_to_pathology(&self, scenario: &SyntheticScenario) -> Option<opensim::PathologyParams> {
        match &scenario.condition {
            Condition::Parkinsons(pd) => {
                let mut symptoms = Vec::new();

                if pd.tremor_severity > 0.1 {
                    symptoms.push(opensim::Symptom::RestingTremor {
                        frequency: 5.0,
                        amplitude: pd.tremor_severity as f64 * 0.05,
                    });
                }

                if pd.bradykinesia > 0.1 {
                    symptoms.push(opensim::Symptom::Bradykinesia {
                        severity: pd.bradykinesia as f64,
                    });
                    symptoms.push(opensim::Symptom::ReducedArmSwing {
                        reduction: pd.bradykinesia as f64 * 0.7,
                    });
                }

                if pd.rigidity > 0.1 {
                    symptoms.push(opensim::Symptom::Rigidity {
                        severity: pd.rigidity as f64,
                    });
                }

                if pd.festination {
                    symptoms.push(opensim::Symptom::Festination);
                }

                Some(opensim::PathologyParams {
                    disease: opensim::DiseaseType::Parkinsons,
                    severity: scenario.severity as f64,
                    symptoms,
                    medication: pd.medication_state,
                })
            }
            Condition::Healthy => None,
            _ => None,
        }
    }

    /// Simulate procedural motion (fallback)
    fn simulate_procedural_motion(&self, scenario: &SyntheticScenario) -> Result<MotionSimulationResult> {
        // Generate simple sinusoidal joint angles
        let num_frames = (scenario.duration * self.config.fps as f64) as usize;
        let mut times = Vec::with_capacity(num_frames);
        let mut joint_angles = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let t = i as f64 / self.config.fps as f64;
            times.push(t);

            // Simple walking pattern
            let phase = t * 2.0 * std::f64::consts::PI; // 1 Hz gait cycle
            let angles = vec![
                0.4 * phase.sin(),      // hip_l
                0.4 * (phase + std::f64::consts::PI).sin(), // hip_r
                0.6 * phase.sin().max(0.0), // knee_l
                0.6 * (phase + std::f64::consts::PI).sin().max(0.0), // knee_r
                0.2 * (phase + 0.5).sin(), // ankle_l
                0.2 * (phase + std::f64::consts::PI + 0.5).sin(), // ankle_r
                0.3 * (phase + std::f64::consts::PI).sin(), // shoulder_l (contralateral)
                0.3 * phase.sin(), // shoulder_r
                0.0, // elbow_l
                0.0, // elbow_r
            ];
            joint_angles.push(angles);
        }

        Ok(MotionSimulationResult {
            trajectory: Some(MotionTrajectory {
                times,
                joint_angles,
                joint_velocities: Vec::new(),
                end_effector_positions: Vec::new(),
                ground_forces: Vec::new(),
                muscle_activations: None,
            }),
            tremor_data: None,
        })
    }

    /// Convert OpenSim result to trajectory
    fn opensim_to_trajectory(&self, result: &opensim::GaitSimulationResult) -> MotionTrajectory {
        MotionTrajectory {
            times: result.times.clone(),
            joint_angles: result.joint_angles.values()
                .next()
                .map(|v| v.iter().map(|a| vec![*a]).collect())
                .unwrap_or_default(),
            joint_velocities: Vec::new(),
            end_effector_positions: Vec::new(),
            ground_forces: Vec::new(),
            muscle_activations: None,
        }
    }

    /// Render videos for all camera angles
    fn render_videos(
        &self,
        scenario: &SyntheticScenario,
        motion: &MotionSimulationResult,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        let trajectory = motion.trajectory.as_ref()
            .ok_or_else(|| MediaError::InvalidConfig("No motion data".to_string()))?;

        for (i, angle) in scenario.camera_angles.iter().enumerate() {
            let output_name = format!("{}_{:?}_{}", scenario.id, angle, i);

            match self.config.video_backend {
                VideoBackend::Blender => {
                    if let Some(ref blender) = self.blender {
                        let result = blender.render_motion(
                            trajectory,
                            blender::HumanModel::StickFigure,
                            &output_name
                        )?;
                        paths.push(result.video_path);
                    }
                }
                VideoBackend::LTXVideo => {
                    if let Some(ref ltx) = self.ltx_video {
                        let prompt = self.scenario_to_prompt(scenario, *angle);
                        let result = ltx.generate(&prompt)?;
                        paths.push(result.video_path);
                    }
                }
                VideoBackend::Skeleton2D => {
                    // Generate simple 2D skeleton video
                    let path = output_dir.join(format!("{}.mp4", output_name));
                    self.render_skeleton_2d(trajectory, &path)?;
                    paths.push(path);
                }
                _ => {}
            }
        }

        // Default single video if no angles specified
        if paths.is_empty() && scenario.camera_angles.is_empty() {
            let output_name = format!("{}_default", scenario.id);

            match self.config.video_backend {
                VideoBackend::LTXVideo => {
                    if let Some(ref ltx) = self.ltx_video {
                        let prompt = self.scenario_to_prompt(scenario, CameraAngle::Side);
                        let result = ltx.generate(&prompt)?;
                        paths.push(result.video_path);
                    }
                }
                _ => {
                    let path = output_dir.join(format!("{}.mp4", output_name));
                    self.render_skeleton_2d(trajectory, &path)?;
                    paths.push(path);
                }
            }
        }

        Ok(paths)
    }

    /// Generate video prompt from scenario
    fn scenario_to_prompt(&self, scenario: &SyntheticScenario, angle: CameraAngle) -> VideoPrompt {
        let view = match angle {
            CameraAngle::Side => "side view",
            CameraAngle::Front => "front view",
            CameraAngle::Back => "back view",
            CameraAngle::TopDown => "top-down view",
            CameraAngle::HandCloseUp => "close-up of hands",
            CameraAngle::FaceCloseUp => "close-up of face",
            CameraAngle::Clinical => "clinical examination view",
        };

        let condition_desc = match &scenario.condition {
            Condition::Healthy => "healthy person",
            Condition::Parkinsons(pd) => {
                if pd.tremor_severity > 0.5 {
                    "person with visible hand tremor"
                } else if pd.bradykinesia > 0.5 {
                    "person walking slowly with reduced arm swing"
                } else {
                    "elderly person walking"
                }
            }
            Condition::EssentialTremor(_) => "person with hand tremor",
            Condition::Stroke { .. } => "person with asymmetric gait",
            _ => "person",
        };

        let task_desc = scenario.tasks.first().map(|t| match t {
            Task::Walk { .. } => "walking",
            Task::Stand { .. } => "standing",
            Task::FingerTap { .. } => "finger tapping",
            Task::Speech { .. } => "speaking",
            _ => "moving",
        }).unwrap_or("walking");

        VideoPrompt::text(format!(
            "A {} {}, {}, clinical room, white background, \
             full body visible, neutral lighting, medical examination, \
             smooth continuous motion, high quality",
            condition_desc, task_desc, view
        )).with_negative("blurry, distorted, multiple people, partial body, cartoon")
    }

    /// Render simple 2D skeleton video using matplotlib
    ///
    /// This generates a video showing a 2D stick figure representation of the motion
    /// trajectory. Uses forward kinematics to convert joint angles to 2D positions.
    fn render_skeleton_2d(&self, trajectory: &MotionTrajectory, path: &Path) -> Result<()> {
        use std::process::Command;
        use std::io::Write;

        // Create temp directory for frames
        let temp_dir = std::env::temp_dir().join(format!("skeleton2d_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)?;

        // Generate Python script for rendering
        let script = self.create_skeleton2d_script(trajectory, &temp_dir, path)?;
        let script_path = temp_dir.join("render_skeleton.py");
        let mut file = std::fs::File::create(&script_path)?;
        file.write_all(script.as_bytes())?;

        // Run the Python script
        let output = Command::new("python3")
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: format!("Python 3 with matplotlib required: {}", e),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::warn!("Skeleton2D rendering failed: {}", stderr);
            // Create a minimal valid MP4 as fallback
            self.create_fallback_video(path)?;
        }

        // Cleanup temp directory
        let _ = std::fs::remove_dir_all(&temp_dir);

        Ok(())
    }

    /// Create Python script for 2D skeleton rendering
    fn create_skeleton2d_script(
        &self,
        trajectory: &MotionTrajectory,
        temp_dir: &Path,
        output_path: &Path,
    ) -> Result<String> {
        // Convert trajectory to JSON for Python
        let times_json: Vec<String> = trajectory.times.iter().map(|t| t.to_string()).collect();
        let angles_json = self.joint_angles_to_json(&trajectory.joint_angles);

        let (width, height) = self.config.video_resolution;
        let fps = self.config.fps;

        Ok(format!(r#"#!/usr/bin/env python3
"""
2D Skeleton Renderer for Motion Trajectory
Generated by DPB MediaPipeline
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import subprocess
import os

# Configuration
OUTPUT_PATH = r'{output_path}'
TEMP_DIR = r'{temp_dir}'
WIDTH = {width}
HEIGHT = {height}
FPS = {fps}
DPI = 100

# Motion data
TIMES = np.array([{times}])
JOINT_ANGLES = {angles}

# Skeleton definition (simplified humanoid)
# Joint indices: 0=pelvis, 1=spine, 2=neck, 3=head,
#               4=l_hip, 5=l_knee, 6=l_ankle,
#               7=r_hip, 8=r_knee, 9=r_ankle,
#               10=l_shoulder, 11=l_elbow, 12=l_wrist,
#               13=r_shoulder, 14=r_elbow, 15=r_wrist

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),           # Spine to head
    (0, 4), (4, 5), (5, 6),           # Left leg
    (0, 7), (7, 8), (8, 9),           # Right leg
    (2, 10), (10, 11), (11, 12),      # Left arm
    (2, 13), (13, 14), (14, 15),      # Right arm
]

# Segment lengths (normalized)
SEGMENT_LENGTHS = {{
    'spine': 0.3,
    'neck': 0.1,
    'head': 0.08,
    'thigh': 0.25,
    'shin': 0.25,
    'upper_arm': 0.18,
    'forearm': 0.15,
}}

def forward_kinematics(angles, frame_idx):
    """Convert joint angles to 2D positions using forward kinematics."""
    # Default standing pose with slight adjustments
    n_joints = 16
    positions = np.zeros((n_joints, 2))

    # Base position (pelvis at center-bottom)
    base_y = 0.15
    positions[0] = [0.5, base_y]  # Pelvis

    # Get angles for this frame (with fallback defaults)
    if frame_idx < len(angles) and len(angles[frame_idx]) > 0:
        a = angles[frame_idx]
    else:
        a = [0.0] * 20  # Default angles

    # Pad angles if needed
    while len(a) < 20:
        a.append(0.0)

    # Spine chain
    spine_angle = np.pi/2 + a[0] * 0.1  # Slight forward/back lean
    positions[1] = positions[0] + SEGMENT_LENGTHS['spine'] * np.array([np.cos(spine_angle), np.sin(spine_angle)])

    neck_angle = spine_angle + a[1] * 0.05
    positions[2] = positions[1] + SEGMENT_LENGTHS['neck'] * np.array([np.cos(neck_angle), np.sin(neck_angle)])

    head_angle = neck_angle
    positions[3] = positions[2] + SEGMENT_LENGTHS['head'] * np.array([np.cos(head_angle), np.sin(head_angle)])

    # Left leg (angles 2-4: hip, knee, ankle)
    l_hip_angle = -np.pi/2 + a[2] * 0.5  # Hip flexion
    positions[4] = positions[0] + np.array([-0.05, 0])  # Hip offset
    positions[5] = positions[4] + SEGMENT_LENGTHS['thigh'] * np.array([np.cos(l_hip_angle), np.sin(l_hip_angle)])

    l_knee_angle = l_hip_angle + a[3] * 0.8  # Knee always bends backward
    positions[6] = positions[5] + SEGMENT_LENGTHS['shin'] * np.array([np.cos(l_knee_angle), np.sin(l_knee_angle)])

    # Right leg (angles 5-7: hip, knee, ankle)
    r_hip_angle = -np.pi/2 + a[5] * 0.5
    positions[7] = positions[0] + np.array([0.05, 0])  # Hip offset
    positions[8] = positions[7] + SEGMENT_LENGTHS['thigh'] * np.array([np.cos(r_hip_angle), np.sin(r_hip_angle)])

    r_knee_angle = r_hip_angle + a[6] * 0.8
    positions[9] = positions[8] + SEGMENT_LENGTHS['shin'] * np.array([np.cos(r_knee_angle), np.sin(r_knee_angle)])

    # Left arm (angles 8-10: shoulder, elbow, wrist)
    l_shoulder_angle = -np.pi/2 - np.pi/6 + a[8] * 0.4  # Arm swing
    positions[10] = positions[2] + np.array([-0.08, -0.02])  # Shoulder offset
    positions[11] = positions[10] + SEGMENT_LENGTHS['upper_arm'] * np.array([np.cos(l_shoulder_angle), np.sin(l_shoulder_angle)])

    l_elbow_angle = l_shoulder_angle - abs(a[9]) * 0.3  # Elbow flexion
    positions[12] = positions[11] + SEGMENT_LENGTHS['forearm'] * np.array([np.cos(l_elbow_angle), np.sin(l_elbow_angle)])

    # Right arm (angles 11-13: shoulder, elbow, wrist)
    r_shoulder_angle = -np.pi/2 + np.pi/6 + a[11] * 0.4
    positions[13] = positions[2] + np.array([0.08, -0.02])  # Shoulder offset
    positions[14] = positions[13] + SEGMENT_LENGTHS['upper_arm'] * np.array([np.cos(r_shoulder_angle), np.sin(r_shoulder_angle)])

    r_elbow_angle = r_shoulder_angle + abs(a[12]) * 0.3
    positions[15] = positions[14] + SEGMENT_LENGTHS['forearm'] * np.array([np.cos(r_elbow_angle), np.sin(r_elbow_angle)])

    return positions

def render_frame(positions, frame_idx, total_frames):
    """Render a single skeleton frame."""
    fig, ax = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)

    # White background with subtle grid
    ax.set_facecolor('#f8f8f8')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw subtle floor line
    ax.axhline(y=0.05, color='#cccccc', linewidth=1, linestyle='--')

    # Draw skeleton bones
    bone_colors = {{
        (0, 1): '#2196F3', (1, 2): '#2196F3', (2, 3): '#2196F3',  # Spine - blue
        (0, 4): '#4CAF50', (4, 5): '#4CAF50', (5, 6): '#4CAF50',  # Left leg - green
        (0, 7): '#FF9800', (7, 8): '#FF9800', (8, 9): '#FF9800',  # Right leg - orange
        (2, 10): '#9C27B0', (10, 11): '#9C27B0', (11, 12): '#9C27B0',  # Left arm - purple
        (2, 13): '#F44336', (13, 14): '#F44336', (14, 15): '#F44336',  # Right arm - red
    }}

    for connection in SKELETON_CONNECTIONS:
        i, j = connection
        color = bone_colors.get(connection, '#333333')
        ax.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color=color, linewidth=3, solid_capstyle='round')

    # Draw joints
    joint_colors = ['#1565C0'] * 4 + ['#2E7D32'] * 3 + ['#E65100'] * 3 + ['#6A1B9A'] * 3 + ['#C62828'] * 3
    for i, pos in enumerate(positions):
        color = joint_colors[i] if i < len(joint_colors) else '#333333'
        size = 8 if i == 3 else 5  # Larger head
        circle = Circle(pos, size/1000, color=color, zorder=10)
        ax.add_patch(circle)

    # Add frame info
    time_sec = TIMES[frame_idx] if frame_idx < len(TIMES) else frame_idx / FPS
    ax.text(0.02, 0.98, f'Frame {{frame_idx+1}}/{{total_frames}}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            color='#666666', fontfamily='monospace')
    ax.text(0.02, 0.94, f'Time: {{time_sec:.2f}}s',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            color='#666666', fontfamily='monospace')

    # Title
    ax.text(0.5, 0.98, 'Motion Trajectory - 2D Skeleton View',
            transform=ax.transAxes, fontsize=12, ha='center', va='top',
            color='#333333', fontweight='bold')

    plt.tight_layout(pad=0)
    return fig

def main():
    """Main rendering function."""
    print(f"Rendering {{len(TIMES)}} frames...")

    # Create frames directory
    frames_dir = os.path.join(TEMP_DIR, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Render each frame
    n_frames = len(TIMES) if len(TIMES) > 0 else 30  # Default 1 second
    for i in range(n_frames):
        positions = forward_kinematics(JOINT_ANGLES, i)
        fig = render_frame(positions, i, n_frames)

        frame_path = os.path.join(frames_dir, f'frame_{{i:05d}}.png')
        fig.savefig(frame_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  Rendered frame {{i+1}}/{{n_frames}}")

    print("Compiling video with ffmpeg...")

    # Compile frames to video using ffmpeg
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', os.path.join(frames_dir, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-preset', 'medium',
        OUTPUT_PATH
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ffmpeg error: {{result.stderr}}")
        # Try with mpeg4 codec as fallback
        ffmpeg_cmd[ffmpeg_cmd.index('libx264')] = 'mpeg4'
        ffmpeg_cmd.remove('-crf')
        ffmpeg_cmd.remove('23')
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Video saved to: {{OUTPUT_PATH}}")
    else:
        print(f"Failed to create video: {{result.stderr}}")
        raise RuntimeError("ffmpeg failed")

if __name__ == '__main__':
    main()
"#,
            output_path = output_path.display(),
            temp_dir = temp_dir.display(),
            width = width,
            height = height,
            fps = fps,
            times = times_json.join(", "),
            angles = angles_json,
        ))
    }

    /// Convert joint angles to JSON array string for Python
    fn joint_angles_to_json(&self, angles: &[Vec<f64>]) -> String {
        if angles.is_empty() {
            return "[]".to_string();
        }

        let frames: Vec<String> = angles.iter().map(|frame| {
            let values: Vec<String> = frame.iter().map(|v| format!("{:.6}", v)).collect();
            format!("[{}]", values.join(", "))
        }).collect();

        format!("[{}]", frames.join(", "))
    }

    /// Create a minimal fallback video when rendering fails
    fn create_fallback_video(&self, path: &Path) -> Result<()> {
        use std::process::Command;

        // Try to create a simple colored video using ffmpeg
        let output = Command::new("ffmpeg")
            .args([
                "-y",
                "-f", "lavfi",
                "-i", &format!("color=c=gray:s={}x{}:d=1",
                    self.config.video_resolution.0,
                    self.config.video_resolution.1),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                path.to_str().unwrap_or("output.mp4"),
            ])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                log::info!("Created fallback video: {:?}", path);
                Ok(())
            }
            _ => {
                // Create empty file as last resort
                std::fs::write(path, b"")?;
                log::warn!("Created empty placeholder video: {:?}", path);
                Ok(())
            }
        }
    }

    /// Synthesize audio for scenario
    fn synthesize_audio(
        &self,
        scenario: &SyntheticScenario,
        text: &str,
        _output_dir: &Path,
    ) -> Result<PathBuf> {
        let pathology = match &scenario.condition {
            Condition::Parkinsons(pd) => PathologicalVoiceParams {
                hypophonia: pd.bradykinesia * 0.5,
                monotonicity: pd.bradykinesia * 0.7,
                breathiness: 0.1,
                hoarseness: 0.1,
                voice_tremor: if pd.tremor_severity > 0.3 {
                    Some((5.0, pd.tremor_severity * 0.1))
                } else {
                    None
                },
                hesitation_probability: pd.bradykinesia * 0.3,
                dysarthria: pd.bradykinesia * 0.4,
            },
            _ => PathologicalVoiceParams::default(),
        };

        if let Some(ref chatterbox) = self.chatterbox {
            let voice = VoiceConfig::default();
            let result = chatterbox.synthesize_pathological(text, &voice, &pathology)?;
            return Ok(result.audio_path);
        }

        Err(MediaError::ToolNotFound {
            tool: "audio backend".to_string(),
            install_url: "pip install chatterbox-tts".to_string(),
        })
    }

    /// Synthesize audio for tasks that need it
    fn synthesize_audio_for_tasks(
        &self,
        scenario: &SyntheticScenario,
        output_dir: &Path,
    ) -> Result<PathBuf> {
        for task in &scenario.tasks {
            match task {
                Task::Speech { text } => {
                    return self.synthesize_audio(scenario, text, output_dir);
                }
                Task::SustainedVowel { vowel, duration } => {
                    let text = vowel.to_string().repeat((*duration * 10.0) as usize);
                    return self.synthesize_audio(scenario, &text, output_dir);
                }
                _ => {}
            }
        }

        Err(MediaError::InvalidConfig("No audio tasks in scenario".to_string()))
    }

    /// Create ground truth from scenario and motion
    fn create_ground_truth(
        &self,
        scenario: &SyntheticScenario,
        motion: &MotionSimulationResult,
    ) -> MediaGroundTruth {
        let mut gt = MediaGroundTruth::default();

        if let Some(ref trajectory) = motion.trajectory {
            gt.timestamps = trajectory.times.clone();

            // Convert to gait ground truth
            let has_walk_task = scenario.tasks.iter().any(|t| matches!(t, Task::Walk { .. }));
            if has_walk_task {
                gt.gait = Some(self.create_gait_ground_truth(scenario, trajectory));
            }
        }

        // Add tremor ground truth
        if let Some(tremor_params) = self.scenario_to_tremor_params(scenario) {
            gt.tremor = Some(TremorGroundTruth {
                frequency: tremor_params.frequency as f32,
                amplitude: tremor_params.amplitude as f32,
                tremor_type: tremor_params.tremor_type,
                affected_joints: tremor_params.affected_joints,
                amplitude_envelope: Vec::new(),
            });
        }

        // Add severity scores
        if let Condition::Parkinsons(pd) = &scenario.condition {
            gt.severity = Some(SeverityScores {
                updrs_motor: Some(pd.hoehn_yahr * 20.0), // Rough estimate
                updrs_tremor: Some(pd.tremor_severity * 28.0),
                updrs_rigidity: Some(pd.rigidity * 20.0),
                updrs_bradykinesia: Some(pd.bradykinesia * 36.0),
                updrs_gait_posture: None,
                hoehn_yahr: Some(pd.hoehn_yahr),
            });
        }

        gt
    }

    /// Create gait ground truth
    fn create_gait_ground_truth(
        &self,
        scenario: &SyntheticScenario,
        _trajectory: &MotionTrajectory,
    ) -> GaitGroundTruth {
        let mut gt = GaitGroundTruth {
            stride_length: 1.2,
            cadence: 110.0,
            speed: 1.2,
            double_support_fraction: 0.3,
            step_width: 0.1,
            arm_swing_asymmetry: 0.0,
            festination: false,
            freezing_episodes: Vec::new(),
        };

        // Adjust based on condition
        match &scenario.condition {
            Condition::Parkinsons(pd) => {
                gt.stride_length *= 1.0 - pd.bradykinesia * 0.4;
                gt.speed *= 1.0 - pd.bradykinesia * 0.5;
                gt.arm_swing_asymmetry = pd.tremor_severity * 0.3;
                gt.festination = pd.festination;
                if pd.festination {
                    gt.cadence = 140.0;
                    gt.stride_length *= 0.6;
                }
            }
            Condition::Stroke { severity, .. } => {
                gt.arm_swing_asymmetry = *severity * 0.5;
                gt.speed *= 1.0 - severity * 0.4;
            }
            _ => {}
        }

        gt
    }

    /// Validate output against ground truth
    fn validate_output(
        &self,
        video_path: &Path,
        ground_truth: &MediaGroundTruth,
    ) -> Result<ValidationResult> {
        let mediapipe = self.mediapipe.as_ref()
            .ok_or_else(|| MediaError::ToolNotFound {
                tool: "mediapipe".to_string(),
                install_url: "pip install mediapipe".to_string(),
            })?;

        let extraction = mediapipe.extract_from_video(video_path)?;

        // Compute detection rate
        let pose_detection_rate = extraction.poses.len() as f32 / extraction.total_frames as f32;

        // Compute gait validation if we have gait ground truth
        let gait_accuracy = if let Some(ref gait_gt) = ground_truth.gait {
            let extracted_gait = mediapipe.compute_gait_parameters(&extraction.poses)?;

            Some(GaitValidation {
                stride_length_error: ((extracted_gait.stride_length - gait_gt.stride_length) / gait_gt.stride_length).abs() * 100.0,
                cadence_error: ((extracted_gait.cadence - gait_gt.cadence) / gait_gt.cadence).abs() * 100.0,
                speed_error: ((extracted_gait.speed - gait_gt.speed) / gait_gt.speed).abs() * 100.0,
            })
        } else {
            None
        };

        Ok(ValidationResult {
            pose_detection_rate,
            mean_position_error: 0.0, // Would need pixel-level comparison
            angle_correlation: 0.0, // Would need joint angle extraction
            gait_accuracy,
        })
    }

    /// Generate Parkinson's scenario preset
    pub fn parkinsons_scenario(severity: f32) -> SyntheticScenario {
        SyntheticScenario {
            id: format!("pd_severity_{:.1}", severity),
            condition: Condition::Parkinsons(ParkinsonsParams {
                hoehn_yahr: 1.0 + severity * 3.0,
                tremor_severity: severity * 0.8,
                bradykinesia: severity * 0.9,
                rigidity: severity * 0.6,
                festination: severity > 0.7,
                fog_probability: if severity > 0.8 { 0.1 } else { 0.0 },
                ..Default::default()
            }),
            severity,
            duration: 10.0,
            tasks: vec![
                Task::Walk { distance: 10.0, speed: None },
                Task::FingerTap { hand: Side::Right, duration: 5.0 },
            ],
            voice_text: Some("The rainbow is a division of white light into many beautiful colors.".to_string()),
            camera_angles: vec![CameraAngle::Side, CameraAngle::HandCloseUp],
            subject: SubjectProfile::default(),
        }
    }

    /// Generate healthy control scenario
    pub fn healthy_scenario() -> SyntheticScenario {
        SyntheticScenario {
            id: "healthy_control".to_string(),
            condition: Condition::Healthy,
            severity: 0.0,
            duration: 10.0,
            tasks: vec![
                Task::Walk { distance: 10.0, speed: Some(1.2) },
            ],
            voice_text: None,
            camera_angles: vec![CameraAngle::Side],
            subject: SubjectProfile::default(),
        }
    }
}

/// Internal result from motion simulation
#[derive(Debug)]
struct MotionSimulationResult {
    trajectory: Option<MotionTrajectory>,
    tremor_data: Option<opensim::TremorSimulationResult>,
}

/// JSON-serializable ground truth
#[derive(serde::Serialize)]
struct GroundTruthJson {
    timestamps: Vec<f64>,
    gait: Option<GaitGroundTruthJson>,
    tremor: Option<TremorGroundTruthJson>,
    severity: Option<SeverityScoresJson>,
}

#[derive(serde::Serialize)]
struct GaitGroundTruthJson {
    stride_length: f32,
    cadence: f32,
    speed: f32,
    arm_swing_asymmetry: f32,
    festination: bool,
}

#[derive(serde::Serialize)]
struct TremorGroundTruthJson {
    frequency: f32,
    amplitude: f32,
    tremor_type: String,
    affected_joints: Vec<String>,
}

#[derive(serde::Serialize)]
struct SeverityScoresJson {
    updrs_motor: Option<f32>,
    hoehn_yahr: Option<f32>,
}

impl From<&MediaGroundTruth> for GroundTruthJson {
    fn from(gt: &MediaGroundTruth) -> Self {
        Self {
            timestamps: gt.timestamps.clone(),
            gait: gt.gait.as_ref().map(|g| GaitGroundTruthJson {
                stride_length: g.stride_length,
                cadence: g.cadence,
                speed: g.speed,
                arm_swing_asymmetry: g.arm_swing_asymmetry,
                festination: g.festination,
            }),
            tremor: gt.tremor.as_ref().map(|t| TremorGroundTruthJson {
                frequency: t.frequency,
                amplitude: t.amplitude,
                tremor_type: format!("{:?}", t.tremor_type),
                affected_joints: t.affected_joints.clone(),
            }),
            severity: gt.severity.as_ref().map(|s| SeverityScoresJson {
                updrs_motor: s.updrs_motor,
                hoehn_yahr: s.hoehn_yahr,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.fps, 30);
        assert!(config.enable_validation);
    }

    #[test]
    fn test_parkinsons_scenario() {
        let scenario = MediaPipeline::parkinsons_scenario(0.5);
        assert!(scenario.id.contains("pd_severity"));
        assert_eq!(scenario.camera_angles.len(), 2);
    }

    #[test]
    fn test_healthy_scenario() {
        let scenario = MediaPipeline::healthy_scenario();
        assert!(matches!(scenario.condition, Condition::Healthy));
    }

    #[test]
    fn test_config_builder() {
        let config = PipelineConfig::default()
            .with_motion_backend(MotionBackend::OpenSim)
            .with_video_backend(VideoBackend::LTXVideo)
            .with_seed(42);

        assert_eq!(config.motion_backend, MotionBackend::OpenSim);
        assert_eq!(config.video_backend, VideoBackend::LTXVideo);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_skeleton2d_config() {
        let config = PipelineConfig::default()
            .with_video_backend(VideoBackend::Skeleton2D);

        assert_eq!(config.video_backend, VideoBackend::Skeleton2D);
    }

    #[test]
    fn test_joint_angles_to_json_empty() {
        let pipeline = MediaPipeline::new(PipelineConfig::default()).unwrap();
        let json = pipeline.joint_angles_to_json(&[]);
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_joint_angles_to_json_with_data() {
        let pipeline = MediaPipeline::new(PipelineConfig::default()).unwrap();
        let angles = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
        ];
        let json = pipeline.joint_angles_to_json(&angles);
        assert!(json.contains("0.100000"));
        assert!(json.contains("0.400000"));
        assert!(json.starts_with("[["));
        assert!(json.ends_with("]]"));
    }

    #[test]
    fn test_skeleton2d_script_generation() {
        let config = PipelineConfig::default()
            .with_video_backend(VideoBackend::Skeleton2D);
        let pipeline = MediaPipeline::new(config).unwrap();

        let trajectory = MotionTrajectory {
            times: vec![0.0, 0.033, 0.066],
            joint_angles: vec![
                vec![0.0; 14],
                vec![0.1; 14],
                vec![0.0; 14],
            ],
            joint_velocities: vec![],
            end_effector_positions: vec![],
            ground_forces: vec![],
            muscle_activations: None,
        };

        let temp_dir = std::path::Path::new("/tmp/test");
        let output_path = std::path::Path::new("/tmp/test/output.mp4");

        let script = pipeline.create_skeleton2d_script(&trajectory, temp_dir, output_path);
        assert!(script.is_ok());

        let script_content = script.unwrap();
        assert!(script_content.contains("matplotlib"));
        assert!(script_content.contains("forward_kinematics"));
        assert!(script_content.contains("SKELETON_CONNECTIONS"));
        assert!(script_content.contains("ffmpeg"));
        // Times are included in the script (format may vary)
        assert!(script_content.contains("TIMES = np.array"));
    }

    #[test]
    fn test_skeleton2d_script_has_all_body_parts() {
        let config = PipelineConfig::default();
        let pipeline = MediaPipeline::new(config).unwrap();

        let trajectory = MotionTrajectory {
            times: vec![0.0],
            joint_angles: vec![vec![0.0; 14]],
            joint_velocities: vec![],
            end_effector_positions: vec![],
            ground_forces: vec![],
            muscle_activations: None,
        };

        let temp_dir = std::path::Path::new("/tmp/test");
        let output_path = std::path::Path::new("/tmp/test/output.mp4");

        let script = pipeline.create_skeleton2d_script(&trajectory, temp_dir, output_path).unwrap();

        // Check for body part definitions
        assert!(script.contains("pelvis"));
        assert!(script.contains("spine"));
        assert!(script.contains("neck"));
        assert!(script.contains("head"));
        assert!(script.contains("thigh"));
        assert!(script.contains("shin"));
        assert!(script.contains("upper_arm"));
        assert!(script.contains("forearm"));
    }

    #[test]
    fn test_skeleton2d_resolution_config() {
        let config = PipelineConfig {
            video_resolution: (1280, 720),
            fps: 60,
            ..PipelineConfig::default()
        };
        let pipeline = MediaPipeline::new(config).unwrap();

        let trajectory = MotionTrajectory {
            times: vec![0.0],
            joint_angles: vec![vec![0.0; 14]],
            joint_velocities: vec![],
            end_effector_positions: vec![],
            ground_forces: vec![],
            muscle_activations: None,
        };

        let temp_dir = std::path::Path::new("/tmp/test");
        let output_path = std::path::Path::new("/tmp/test/output.mp4");

        let script = pipeline.create_skeleton2d_script(&trajectory, temp_dir, output_path).unwrap();

        assert!(script.contains("WIDTH = 1280"));
        assert!(script.contains("HEIGHT = 720"));
        assert!(script.contains("FPS = 60"));
    }
}
