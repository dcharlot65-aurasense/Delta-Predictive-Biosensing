//! SMPL/SMPL-X Human Body Model Renderer
//!
//! This module provides integration with SMPL (Skinned Multi-Person Linear Model)
//! and PyTorch3D for realistic human body rendering.
//!
//! # Overview
//!
//! SMPL is a parametric human body model that represents the body as:
//! - **Shape parameters (β)**: 10 parameters controlling body shape (height, weight, etc.)
//! - **Pose parameters (θ)**: 72 parameters (24 joints × 3 axis-angle rotations)
//! - **Global translation**: 3D position in world space
//!
//! This renderer uses PyTorch3D for differentiable rendering, enabling:
//! - Realistic human mesh rendering
//! - Multiple camera views
//! - Texture and lighting control
//! - Ground truth depth maps and segmentation
//!
//! # Requirements
//!
//! - Python 3.8+
//! - PyTorch 1.10+
//! - PyTorch3D
//! - smplx package
//! - SMPL model files (downloaded from https://smpl.is.tue.mpg.de/)
//!
//! # Usage
//!
//! ```rust,ignore
//! use dpb_synth::level3::smpl::{SmplRenderer, SmplParams, SmplPose, SmplShape};
//! use std::path::Path;
//!
//! let renderer = SmplRenderer::new(SmplParams::default())?;
//!
//! // Create a pose and shape
//! let pose = SmplPose::from_axis_angles(&[0.0; 72]);
//! let shape = SmplShape::default();
//!
//! // Render frame
//! let output = renderer.render_pose(&pose, &shape, Path::new("output.png"))?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Result type for SMPL rendering
pub type SmplResult<T> = Result<T, SmplError>;

/// Errors that can occur during SMPL rendering
#[derive(Debug, thiserror::Error)]
pub enum SmplError {
    #[error("Python not found: {0}")]
    PythonNotFound(String),

    #[error("PyTorch3D not installed: {0}")]
    PyTorch3DNotFound(String),

    #[error("SMPL model not found: {0}")]
    SmplModelNotFound(PathBuf),

    #[error("Script not found: {0}")]
    ScriptNotFound(PathBuf),

    #[error("Render failed: {0}")]
    RenderError(String),

    #[error("Invalid pose parameters: {0}")]
    InvalidPose(String),

    #[error("Invalid shape parameters: {0}")]
    InvalidShape(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// SMPL body model variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SmplBodyModel {
    /// Original SMPL (10 shape params, 72 pose params)
    #[default]
    Smpl,
    /// SMPL+H with hand articulation
    SmplH,
    /// SMPL-X with hands and face
    SmplX,
    /// STAR model (alternative to SMPL)
    Star,
}

impl SmplBodyModel {
    /// Get the number of shape parameters
    pub fn num_shape_params(&self) -> usize {
        match self {
            SmplBodyModel::Smpl => 10,
            SmplBodyModel::SmplH => 16,
            SmplBodyModel::SmplX => 10,
            SmplBodyModel::Star => 10,
        }
    }

    /// Get the number of pose parameters (axis-angle)
    pub fn num_pose_params(&self) -> usize {
        match self {
            SmplBodyModel::Smpl => 72,   // 24 joints × 3
            SmplBodyModel::SmplH => 156, // 52 joints × 3
            SmplBodyModel::SmplX => 165, // 55 joints × 3 (body + hands + face)
            SmplBodyModel::Star => 72,
        }
    }

    /// Get the number of joints
    pub fn num_joints(&self) -> usize {
        match self {
            SmplBodyModel::Smpl => 24,
            SmplBodyModel::SmplH => 52,
            SmplBodyModel::SmplX => 55,
            SmplBodyModel::Star => 24,
        }
    }
}

/// SMPL pose representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmplPose {
    /// Axis-angle rotations for each joint
    pub body_pose: Vec<f64>,
    /// Global orientation (root rotation)
    pub global_orient: [f64; 3],
    /// Global translation
    pub transl: [f64; 3],
    /// Left hand pose (for SMPL-H/X)
    pub left_hand_pose: Option<Vec<f64>>,
    /// Right hand pose (for SMPL-H/X)
    pub right_hand_pose: Option<Vec<f64>>,
    /// Jaw pose (for SMPL-X)
    pub jaw_pose: Option<[f64; 3]>,
    /// Eye poses (for SMPL-X)
    pub leye_pose: Option<[f64; 3]>,
    pub reye_pose: Option<[f64; 3]>,
    /// Expression coefficients (for SMPL-X)
    pub expression: Option<Vec<f64>>,
}

impl SmplPose {
    /// Create a neutral (rest) pose
    pub fn neutral() -> Self {
        Self {
            body_pose: vec![0.0; 69], // 23 joints × 3 (excluding root)
            global_orient: [0.0; 3],
            transl: [0.0, 0.0, 0.0],
            left_hand_pose: None,
            right_hand_pose: None,
            jaw_pose: None,
            leye_pose: None,
            reye_pose: None,
            expression: None,
        }
    }

    /// Create pose from flat axis-angle array
    pub fn from_axis_angles(angles: &[f64]) -> SmplResult<Self> {
        if angles.len() < 72 {
            return Err(SmplError::InvalidPose(format!(
                "Expected at least 72 pose params, got {}",
                angles.len()
            )));
        }

        Ok(Self {
            body_pose: angles[3..72].to_vec(),
            global_orient: [angles[0], angles[1], angles[2]],
            transl: [0.0, 0.0, 0.0],
            left_hand_pose: None,
            right_hand_pose: None,
            jaw_pose: None,
            leye_pose: None,
            reye_pose: None,
            expression: None,
        })
    }

    /// Create a walking pose at a given phase (0.0 - 1.0)
    pub fn walking(phase: f64) -> Self {
        let mut pose = Self::neutral();
        let phase_rad = phase * 2.0 * std::f64::consts::PI;

        // Hip flexion (alternating)
        let hip_angle = 0.4 * phase_rad.sin();
        pose.body_pose[0] = hip_angle; // Left hip X
        pose.body_pose[3] = -hip_angle; // Right hip X

        // Knee flexion (synchronized with hip)
        let knee_angle = 0.3 * (phase_rad + 0.5).sin().max(0.0);
        pose.body_pose[9] = knee_angle; // Left knee
        pose.body_pose[12] = 0.3 * (phase_rad - 0.5 + std::f64::consts::PI).sin().max(0.0); // Right knee

        // Arm swing (opposite to legs)
        let arm_angle = 0.3 * phase_rad.sin();
        pose.body_pose[48] = -arm_angle; // Left shoulder
        pose.body_pose[51] = arm_angle; // Right shoulder

        // Slight elbow bend
        pose.body_pose[54] = 0.2; // Left elbow
        pose.body_pose[57] = 0.2; // Right elbow

        pose
    }

    /// Create a Parkinsonian gait pose
    pub fn parkinsonian_gait(phase: f64, severity: f64) -> Self {
        let mut pose = Self::walking(phase);

        // Reduced arm swing
        let arm_reduction = 1.0 - severity * 0.8;
        pose.body_pose[48] *= arm_reduction;
        pose.body_pose[51] *= arm_reduction;

        // Forward trunk lean
        pose.global_orient[0] = severity * 0.3;

        // Reduced stride (smaller hip angles)
        let stride_reduction = 1.0 - severity * 0.5;
        pose.body_pose[0] *= stride_reduction;
        pose.body_pose[3] *= stride_reduction;

        // Flexed knees
        pose.body_pose[9] += severity * 0.2;
        pose.body_pose[12] += severity * 0.2;

        pose
    }

    /// Index of the first parameter of a MANO hand joint.
    ///
    /// SMPL-H carries 45 numbers per hand: fifteen joints as axis-angle
    /// triplets, ordered index, middle, pinky, ring, thumb, each finger
    /// proximal to distal. The ordering is MANO's and is what the model
    /// weights expect.
    /// Index finger is MANO joint 0, thumb is joint 12; each joint occupies
    /// three consecutive axis-angle parameters.
    const MANO_JOINT_PARAMS: usize = 3;
    const MANO_INDEX_PROXIMAL: usize = 0;
    const MANO_THUMB_PROXIMAL: usize = 12 * Self::MANO_JOINT_PARAMS;

    /// Create a finger tapping pose.
    ///
    /// `thumb_angle` and `index_angle` are flexion angles in radians, applied
    /// to the proximal joint of each digit -- the joint a single flexion angle
    /// describes. The remaining joints stay neutral: distributing one angle
    /// across a finger needs a coupling ratio that is a modelling choice, not
    /// something this signature supplies, so none is assumed here.
    ///
    /// Flexion is taken about the first axis of the triplet, which is the
    /// convention the body poses in this file use (`body_pose[0]` is hip X).
    ///
    /// Both angles used to be discarded: the function returned a flat hand
    /// whatever it was asked for, so a caller sweeping tap amplitude got the
    /// same pose every time.
    pub fn finger_tapping(thumb_angle: f64, index_angle: f64) -> Self {
        let mut pose = Self::neutral();

        // Position arm for tapping (seated position)
        pose.body_pose[48] = -0.5; // Shoulder forward
        pose.body_pose[54] = 1.2; // Elbow bent

        let mut hand = vec![0.0; 45];
        hand[Self::MANO_INDEX_PROXIMAL] = index_angle;
        hand[Self::MANO_THUMB_PROXIMAL] = thumb_angle;
        pose.left_hand_pose = Some(hand);

        pose
    }
}

/// SMPL shape parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmplShape {
    /// Beta coefficients (typically 10)
    pub betas: Vec<f64>,
}

impl SmplShape {
    /// Average/neutral body shape
    pub fn neutral() -> Self {
        Self {
            betas: vec![0.0; 10],
        }
    }

    /// Create custom shape from beta values
    pub fn from_betas(betas: &[f64]) -> Self {
        Self {
            betas: betas.to_vec(),
        }
    }

    /// Tall body type
    pub fn tall() -> Self {
        let mut betas = vec![0.0; 10];
        betas[0] = 2.0; // Height increase
        Self { betas }
    }

    /// Heavy body type
    pub fn heavy() -> Self {
        let mut betas = vec![0.0; 10];
        betas[1] = 2.0; // Weight increase
        Self { betas }
    }

    /// Elderly body type (stooped posture modifier)
    pub fn elderly() -> Self {
        let mut betas = vec![0.0; 10];
        betas[0] = -0.5; // Slightly shorter
        betas[1] = 0.5; // Slightly heavier
        Self { betas }
    }
}

/// Camera parameters for rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmplCamera {
    /// Camera position in world coordinates
    pub position: [f64; 3],
    /// Camera look-at point
    pub target: [f64; 3],
    /// Camera up vector
    pub up: [f64; 3],
    /// Field of view in degrees
    pub fov: f64,
    /// Near clipping plane
    pub near: f64,
    /// Far clipping plane
    pub far: f64,
}

impl Default for SmplCamera {
    fn default() -> Self {
        Self {
            position: [0.0, 1.0, 3.0],
            target: [0.0, 1.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 60.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl SmplCamera {
    /// Front view camera
    pub fn front() -> Self {
        Self::default()
    }

    /// Side view camera
    pub fn side() -> Self {
        Self {
            position: [3.0, 1.0, 0.0],
            target: [0.0, 1.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 60.0,
            near: 0.1,
            far: 100.0,
        }
    }

    /// Top-down camera
    pub fn top() -> Self {
        Self {
            position: [0.0, 5.0, 0.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 0.0, -1.0],
            fov: 60.0,
            near: 0.1,
            far: 100.0,
        }
    }

    /// Oblique 45-degree view
    pub fn oblique() -> Self {
        Self {
            position: [2.0, 1.5, 2.0],
            target: [0.0, 1.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 60.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

/// Lighting parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmplLighting {
    /// Ambient light color (RGB, 0-1)
    pub ambient_color: [f64; 3],
    /// Ambient light intensity
    pub ambient_intensity: f64,
    /// Directional light positions (list of 3D positions)
    pub light_positions: Vec<[f64; 3]>,
    /// Directional light colors
    pub light_colors: Vec<[f64; 3]>,
    /// Directional light intensities
    pub light_intensities: Vec<f64>,
}

impl Default for SmplLighting {
    fn default() -> Self {
        Self {
            ambient_color: [1.0, 1.0, 1.0],
            ambient_intensity: 0.3,
            light_positions: vec![[2.0, 3.0, 2.0], [-2.0, 3.0, -2.0]],
            light_colors: vec![[1.0, 1.0, 1.0], [0.8, 0.8, 1.0]],
            light_intensities: vec![0.7, 0.3],
        }
    }
}

/// Parameters for SMPL rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmplParams {
    /// Body model variant
    pub model: SmplBodyModel,
    /// Path to SMPL model files
    pub model_path: PathBuf,
    /// Output resolution (width, height)
    pub resolution: (u32, u32),
    /// Frame rate for video output
    pub fps: u32,
    /// Camera parameters
    pub camera: SmplCamera,
    /// Lighting parameters
    pub lighting: SmplLighting,
    /// Background color (RGBA)
    pub background_color: [f64; 4],
    /// Body texture/material
    pub body_color: [f64; 3],
    /// Whether to render wireframe overlay
    pub wireframe: bool,
    /// Whether to output depth map
    pub output_depth: bool,
    /// Whether to output segmentation mask
    pub output_segmentation: bool,
    /// Whether to output joint positions
    pub output_joints: bool,
    /// Anti-aliasing samples
    pub aa_samples: u32,
    /// Python executable path
    pub python_path: PathBuf,
    /// Script directory
    pub script_dir: PathBuf,
}

impl Default for SmplParams {
    fn default() -> Self {
        Self {
            model: SmplBodyModel::Smpl,
            model_path: PathBuf::from("models/smpl"),
            resolution: (640, 480),
            fps: 30,
            camera: SmplCamera::default(),
            lighting: SmplLighting::default(),
            background_color: [0.9, 0.9, 0.9, 1.0],
            body_color: [0.7, 0.6, 0.5],
            wireframe: false,
            output_depth: false,
            output_segmentation: false,
            output_joints: true,
            aa_samples: 4,
            python_path: PathBuf::from("python3"),
            script_dir: PathBuf::from("tools/level3_video"),
        }
    }
}

impl SmplParams {
    /// High quality rendering settings
    pub fn high_quality() -> Self {
        Self {
            resolution: (1920, 1080),
            aa_samples: 8,
            output_depth: true,
            output_segmentation: true,
            ..Default::default()
        }
    }

    /// Fast preview settings
    pub fn preview() -> Self {
        Self {
            resolution: (320, 240),
            aa_samples: 1,
            wireframe: true,
            ..Default::default()
        }
    }

    /// Clinical analysis settings
    pub fn clinical() -> Self {
        Self {
            resolution: (1280, 720),
            output_joints: true,
            output_depth: true,
            background_color: [1.0, 1.0, 1.0, 1.0],
            ..Default::default()
        }
    }
}

/// Output from SMPL rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmplOutput {
    /// Rendered image path
    pub image_path: PathBuf,
    /// Depth map path (if requested)
    pub depth_path: Option<PathBuf>,
    /// Segmentation mask path (if requested)
    pub segmentation_path: Option<PathBuf>,
    /// 3D joint positions (if requested)
    pub joints_3d: Option<Vec<[f64; 3]>>,
    /// 2D joint positions projected to image (if requested)
    pub joints_2d: Option<Vec<[f64; 2]>>,
    /// Mesh vertices
    pub vertices: Option<Vec<[f64; 3]>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Main SMPL renderer interface
pub struct SmplRenderer {
    params: SmplParams,
}

impl SmplRenderer {
    /// Create a new SMPL renderer
    pub fn new(params: SmplParams) -> SmplResult<Self> {
        Ok(Self { params })
    }

    /// Check if Python and required packages are available
    pub fn check_requirements(&self) -> SmplResult<bool> {
        let output = Command::new(&self.params.python_path)
            .args([
                "-c",
                "import torch; import pytorch3d; import smplx; print('OK')",
            ])
            .output()
            .map_err(|e| SmplError::PythonNotFound(e.to_string()))?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            Ok(stdout.contains("OK"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(SmplError::PyTorch3DNotFound(stderr.to_string()))
        }
    }

    /// Get PyTorch3D version
    pub fn pytorch3d_version(&self) -> SmplResult<String> {
        let output = Command::new(&self.params.python_path)
            .args(["-c", "import pytorch3d; print(pytorch3d.__version__)"])
            .output()
            .map_err(|e| SmplError::PythonNotFound(e.to_string()))?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Update rendering parameters
    pub fn set_params(&mut self, params: SmplParams) {
        self.params = params;
    }

    /// Render a single pose
    pub fn render_pose(
        &self,
        pose: &SmplPose,
        shape: &SmplShape,
        output_path: &Path,
    ) -> SmplResult<SmplOutput> {
        let script_path = self.params.script_dir.join("smpl_render.py");
        if !script_path.exists() {
            return Err(SmplError::ScriptNotFound(script_path));
        }

        // Serialize parameters
        let render_params = serde_json::json!({
            "model_type": format!("{:?}", self.params.model).to_lowercase(),
            "model_path": self.params.model_path,
            "resolution": self.params.resolution,
            "pose": pose,
            "shape": shape,
            "camera": self.params.camera,
            "lighting": self.params.lighting,
            "background_color": self.params.background_color,
            "body_color": self.params.body_color,
            "wireframe": self.params.wireframe,
            "output_depth": self.params.output_depth,
            "output_segmentation": self.params.output_segmentation,
            "output_joints": self.params.output_joints,
            "aa_samples": self.params.aa_samples,
            "output_path": output_path,
        });

        let params_json = serde_json::to_string(&render_params)?;

        let output = Command::new(&self.params.python_path)
            .args([script_path.to_str().unwrap(), "--params", &params_json])
            .output()
            .map_err(|e| SmplError::RenderError(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SmplError::RenderError(stderr.to_string()));
        }

        // Parse output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: SmplOutput = serde_json::from_str(&stdout).unwrap_or_else(|_| SmplOutput {
            image_path: output_path.to_path_buf(),
            depth_path: None,
            segmentation_path: None,
            joints_3d: None,
            joints_2d: None,
            vertices: None,
            metadata: HashMap::new(),
        });

        Ok(result)
    }

    /// Render a sequence of poses
    pub fn render_sequence(
        &self,
        poses: &[SmplPose],
        shape: &SmplShape,
        output_dir: &Path,
        prefix: &str,
    ) -> SmplResult<Vec<SmplOutput>> {
        std::fs::create_dir_all(output_dir)?;

        let mut outputs = Vec::with_capacity(poses.len());

        for (i, pose) in poses.iter().enumerate() {
            let filename = format!("{}_{:06}.png", prefix, i);
            let output_path = output_dir.join(&filename);

            let output = self.render_pose(pose, shape, &output_path)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Render a gait animation
    pub fn render_gait_animation(
        &self,
        duration: f64,
        cadence: f64, // steps per minute
        shape: &SmplShape,
        output_dir: &Path,
        parkinsonian_severity: f64,
    ) -> SmplResult<Vec<SmplOutput>> {
        let num_frames = (duration * self.params.fps as f64) as usize;
        let step_duration = 60.0 / cadence;
        let dt = 1.0 / self.params.fps as f64;

        let poses: Vec<SmplPose> = (0..num_frames)
            .map(|i| {
                let t = i as f64 * dt;
                let phase = (t / step_duration) % 1.0;

                if parkinsonian_severity > 0.0 {
                    SmplPose::parkinsonian_gait(phase, parkinsonian_severity)
                } else {
                    SmplPose::walking(phase)
                }
            })
            .collect();

        self.render_sequence(&poses, shape, output_dir, "gait")
    }

    /// Convert keypoints from StreamingClinicalPose to SMPL pose
    pub fn pose_from_keypoints(keypoints: &[[f64; 3]; 33]) -> SmplResult<SmplPose> {
        // This is a simplified mapping - a full implementation would use
        // inverse kinematics to estimate joint angles from keypoint positions
        let mut pose = SmplPose::neutral();

        // Calculate some basic joint angles from keypoint positions
        // Hip angles from hip-knee vectors
        let left_hip = keypoints[23];
        let left_knee = keypoints[25];
        let right_hip = keypoints[24];
        let right_knee = keypoints[26];

        // Simplified angle estimation
        let left_hip_angle = (left_knee[2] - left_hip[2]).atan2(left_hip[1] - left_knee[1]);
        let right_hip_angle = (right_knee[2] - right_hip[2]).atan2(right_hip[1] - right_knee[1]);

        pose.body_pose[0] = left_hip_angle;
        pose.body_pose[3] = right_hip_angle;

        // Knee angles
        let left_ankle = keypoints[27];
        let right_ankle = keypoints[28];

        let left_knee_angle =
            (left_ankle[2] - left_knee[2]).atan2(left_knee[1] - left_ankle[1]) - left_hip_angle;
        let right_knee_angle = (right_ankle[2] - right_knee[2])
            .atan2(right_knee[1] - right_ankle[1])
            - right_hip_angle;

        pose.body_pose[9] = left_knee_angle.max(0.0);
        pose.body_pose[12] = right_knee_angle.max(0.0);

        // Global position from hip center
        pose.transl = [
            (left_hip[0] + right_hip[0]) / 2.0,
            (left_hip[1] + right_hip[1]) / 2.0,
            (left_hip[2] + right_hip[2]) / 2.0,
        ];

        Ok(pose)
    }

    /// Get the SMPL render script content
    pub fn get_script_template() -> String {
        generate_smpl_render_script()
    }
}

/// Generate the Python render script content
pub fn generate_smpl_render_script() -> String {
    r#"#!/usr/bin/env python3
"""
SMPL/PyTorch3D Renderer for DPB Framework

This script renders SMPL body models using PyTorch3D.
It is called from the Rust interface with JSON parameters.

Requirements:
    pip install torch pytorch3d smplx trimesh

Usage:
    python smpl_render.py --params '{"model_type": "smpl", ...}'
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    import pytorch3d
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        PointLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        TexturesVertex,
    )
    import smplx
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(json.dumps({"error": f"Missing dependency: {e}"}))
    sys.exit(1)


def create_smpl_model(model_type, model_path, device):
    """Create SMPL/SMPL-X model."""
    model_type = model_type.lower()

    if model_type == "smpl":
        model = smplx.create(model_path, model_type="smpl", gender="neutral")
    elif model_type == "smplh":
        model = smplx.create(model_path, model_type="smplh", gender="neutral")
    elif model_type == "smplx":
        model = smplx.create(model_path, model_type="smplx", gender="neutral", use_face_contour=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def setup_renderer(params, device):
    """Setup PyTorch3D renderer."""
    width, height = params["resolution"]

    # Camera
    cam_params = params["camera"]
    cameras = FoVPerspectiveCameras(
        device=device,
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.tensor([cam_params["position"]], device=device),
        fov=cam_params["fov"],
        znear=cam_params["near"],
        zfar=cam_params["far"],
    )

    # Rasterization
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Lighting
    light_params = params["lighting"]
    lights = PointLights(
        device=device,
        ambient_color=[[light_params["ambient_intensity"]] * 3],
        diffuse_color=[[0.7, 0.7, 0.7]],
        specular_color=[[0.2, 0.2, 0.2]],
        location=[light_params["light_positions"][0]],
    )

    # Renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    return renderer, cameras


def render_smpl(params):
    """Main rendering function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = create_smpl_model(params["model_type"], params["model_path"], device)

    # Setup pose and shape
    pose_data = params["pose"]
    shape_data = params["shape"]

    body_pose = torch.tensor([pose_data["body_pose"]], dtype=torch.float32, device=device)
    global_orient = torch.tensor([pose_data["global_orient"]], dtype=torch.float32, device=device)
    betas = torch.tensor([shape_data["betas"]], dtype=torch.float32, device=device)
    transl = torch.tensor([pose_data["transl"]], dtype=torch.float32, device=device)

    # Forward pass
    output = model(
        body_pose=body_pose,
        global_orient=global_orient,
        betas=betas,
        transl=transl,
        return_verts=True,
    )

    vertices = output.vertices
    faces = torch.tensor(model.faces.astype(np.int64), device=device).unsqueeze(0)

    # Create mesh with vertex colors
    body_color = params["body_color"]
    verts_rgb = torch.tensor([[body_color] * vertices.shape[1]], dtype=torch.float32, device=device)
    textures = TexturesVertex(verts_features=verts_rgb)

    mesh = Meshes(verts=vertices, faces=faces, textures=textures)

    # Render
    renderer, cameras = setup_renderer(params, device)
    images = renderer(mesh)

    # Save image
    output_path = Path(params["output_path"])
    img = images[0, ..., :3].cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)

    # Prepare output
    result = {
        "image_path": str(output_path),
        "depth_path": None,
        "segmentation_path": None,
        "joints_3d": output.joints[0].cpu().tolist() if params["output_joints"] else None,
        "joints_2d": None,  # Would need projection
        "vertices": None,
        "metadata": {
            "model_type": params["model_type"],
            "device": str(device),
        }
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="SMPL Renderer")
    parser.add_argument("--params", type=str, required=True, help="JSON parameters")
    args = parser.parse_args()

    try:
        params = json.loads(args.params)
        result = render_smpl(params)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smpl_pose_neutral() {
        let pose = SmplPose::neutral();
        assert_eq!(pose.body_pose.len(), 69);
        assert_eq!(pose.global_orient, [0.0; 3]);
    }

    #[test]
    fn test_smpl_pose_from_axis_angles() {
        let angles = vec![0.1; 72];
        let pose = SmplPose::from_axis_angles(&angles).unwrap();
        assert_eq!(pose.global_orient, [0.1, 0.1, 0.1]);
        assert_eq!(pose.body_pose.len(), 69);
    }

    #[test]
    fn test_smpl_pose_invalid() {
        let angles = vec![0.1; 10]; // Too few
        let result = SmplPose::from_axis_angles(&angles);
        assert!(result.is_err());
    }

    #[test]
    fn test_smpl_pose_walking() {
        let pose = SmplPose::walking(0.0);
        assert!(!pose.body_pose.is_empty());

        let pose2 = SmplPose::walking(0.5);
        // Different phases should produce different poses
        assert_ne!(pose.body_pose[0], pose2.body_pose[0]);
    }

    #[test]
    fn test_smpl_pose_parkinsonian() {
        let normal = SmplPose::walking(0.25);
        let parkinsonian = SmplPose::parkinsonian_gait(0.25, 0.8);

        // Forward trunk lean
        assert!(parkinsonian.global_orient[0] > normal.global_orient[0]);
    }

    #[test]
    fn test_smpl_shape_presets() {
        let neutral = SmplShape::neutral();
        assert_eq!(neutral.betas.len(), 10);
        assert!(neutral.betas.iter().all(|&b| b == 0.0));

        let tall = SmplShape::tall();
        assert!(tall.betas[0] > 0.0);

        let heavy = SmplShape::heavy();
        assert!(heavy.betas[1] > 0.0);
    }

    #[test]
    fn test_smpl_camera_presets() {
        let front = SmplCamera::front();
        assert!(front.position[2] > 0.0); // In front of origin

        let side = SmplCamera::side();
        assert!(side.position[0] > 0.0); // To the side

        let top = SmplCamera::top();
        assert!(top.position[1] > 0.0); // Above
    }

    #[test]
    fn test_body_model_params() {
        assert_eq!(SmplBodyModel::Smpl.num_joints(), 24);
        assert_eq!(SmplBodyModel::Smpl.num_pose_params(), 72);
        assert_eq!(SmplBodyModel::Smpl.num_shape_params(), 10);

        assert_eq!(SmplBodyModel::SmplX.num_joints(), 55);
    }

    #[test]
    fn test_smpl_params_presets() {
        let default = SmplParams::default();
        assert_eq!(default.resolution, (640, 480));

        let hq = SmplParams::high_quality();
        assert_eq!(hq.resolution, (1920, 1080));
        assert!(hq.output_depth);

        let preview = SmplParams::preview();
        assert_eq!(preview.resolution, (320, 240));
        assert!(preview.wireframe);
    }

    #[test]
    fn test_generate_script() {
        let script = generate_smpl_render_script();
        assert!(script.contains("pytorch3d"));
        assert!(script.contains("smplx"));
        assert!(script.contains("def render_smpl"));
    }
}

#[cfg(test)]
mod finger_tapping_tests {
    use super::*;

    /// The tap angles must reach the hand.
    ///
    /// They were discarded, so every call returned a flat hand and a caller
    /// sweeping amplitude saw no change. A test checking only that the pose has
    /// 45 hand parameters would have passed throughout.
    #[test]
    fn tap_angles_reach_the_proximal_joints() {
        let pose = SmplPose::finger_tapping(0.7, 0.4);
        let hand = pose.left_hand_pose.expect("SMPL-H hand pose");
        assert_eq!(
            hand.len(),
            45,
            "SMPL-H has 15 joints as axis-angle triplets"
        );

        assert!((hand[SmplPose::MANO_THUMB_PROXIMAL] - 0.7).abs() < 1e-12);
        assert!((hand[SmplPose::MANO_INDEX_PROXIMAL] - 0.4).abs() < 1e-12);

        // Thumb and index are distinct joints, 12 apart in MANO order.
        assert_ne!(SmplPose::MANO_THUMB_PROXIMAL, SmplPose::MANO_INDEX_PROXIMAL);

        // Nothing else moved.
        let touched = [SmplPose::MANO_THUMB_PROXIMAL, SmplPose::MANO_INDEX_PROXIMAL];
        for (i, v) in hand.iter().enumerate() {
            if !touched.contains(&i) {
                assert_eq!(*v, 0.0, "joint parameter {i} should be neutral");
            }
        }
    }

    /// Different amplitudes must give different poses.
    #[test]
    fn tap_amplitude_changes_the_pose() {
        let small = SmplPose::finger_tapping(0.1, 0.1).left_hand_pose.unwrap();
        let large = SmplPose::finger_tapping(1.2, 0.9).left_hand_pose.unwrap();
        assert_ne!(small, large, "hand pose does not depend on the tap angles");

        // And zero really is the neutral hand.
        let zero = SmplPose::finger_tapping(0.0, 0.0).left_hand_pose.unwrap();
        assert_eq!(zero, vec![0.0; 45]);
    }
}
