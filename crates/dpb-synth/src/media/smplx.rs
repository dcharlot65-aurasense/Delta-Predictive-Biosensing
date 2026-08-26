//! SMPL-X Human Body Model Integration
//!
//! SMPL-X (SMPL eXpressive) is a statistical body model that captures body,
//! hand, and facial expressions in a unified representation.
//!
//! # Features
//!
//! - 10,475 vertices for body + hands + face
//! - Parametric control over pose and shape
//! - Realistic deformations
//! - Compatible with animation and rendering pipelines
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install smplx torch`
//! - SMPL-X model files (registration required)

use super::{MediaError, Result, JointAngles, JointPositions3D};
use std::path::{Path, PathBuf};
use std::process::Command;

/// SMPL-X body model interface
#[derive(Debug)]
pub struct SMPLXModel {
    config: SMPLXConfig,
    python_path: PathBuf,
}

/// Configuration for SMPL-X
#[derive(Debug, Clone)]
pub struct SMPLXConfig {
    /// Path to SMPL-X model files
    pub model_path: PathBuf,
    /// Output directory
    pub output_dir: PathBuf,
    /// Model gender (neutral, male, female)
    pub gender: Gender,
    /// Number of shape components to use
    pub num_betas: u32,
    /// Number of expression components
    pub num_expression: u32,
    /// Use hand PCA space
    pub use_pca: bool,
    /// Number of hand PCA components
    pub num_pca_comps: u32,
    /// GPU device (-1 for CPU)
    pub device: i32,
}

impl Default for SMPLXConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/smplx"),
            output_dir: PathBuf::from("output/smplx"),
            gender: Gender::Neutral,
            num_betas: 10,
            num_expression: 10,
            use_pca: true,
            num_pca_comps: 12,
            device: 0,
        }
    }
}

/// Gender for body model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    Neutral,
    Male,
    Female,
}

impl Gender {
    fn as_str(&self) -> &'static str {
        match self {
            Gender::Neutral => "neutral",
            Gender::Male => "male",
            Gender::Female => "female",
        }
    }
}

/// Body pose parameters for SMPL-X
#[derive(Debug, Clone)]
pub struct BodyPose {
    /// Global body orientation (axis-angle, 3 values)
    pub global_orient: [f32; 3],
    /// Body pose (21 joints × 3 axis-angle = 63 values)
    pub body_pose: Vec<f32>,
    /// Left hand pose (PCA or full)
    pub left_hand_pose: Vec<f32>,
    /// Right hand pose (PCA or full)
    pub right_hand_pose: Vec<f32>,
    /// Jaw pose (axis-angle, 3 values)
    pub jaw_pose: [f32; 3],
    /// Left eye gaze direction
    pub leye_pose: [f32; 3],
    /// Right eye gaze direction
    pub reye_pose: [f32; 3],
}

impl Default for BodyPose {
    fn default() -> Self {
        Self {
            global_orient: [0.0; 3],
            body_pose: vec![0.0; 63],
            left_hand_pose: vec![0.0; 12], // PCA components
            right_hand_pose: vec![0.0; 12],
            jaw_pose: [0.0; 3],
            leye_pose: [0.0; 3],
            reye_pose: [0.0; 3],
        }
    }
}

/// Body shape parameters
#[derive(Debug, Clone)]
pub struct BodyShape {
    /// Shape coefficients (betas)
    pub betas: Vec<f32>,
    /// Expression coefficients
    pub expression: Vec<f32>,
}

impl Default for BodyShape {
    fn default() -> Self {
        Self {
            betas: vec![0.0; 10],
            expression: vec![0.0; 10],
        }
    }
}

/// SMPL-X joint names (55 joints)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SMPLXJoint {
    Pelvis = 0,
    LeftHip = 1,
    RightHip = 2,
    Spine1 = 3,
    LeftKnee = 4,
    RightKnee = 5,
    Spine2 = 6,
    LeftAnkle = 7,
    RightAnkle = 8,
    Spine3 = 9,
    LeftFoot = 10,
    RightFoot = 11,
    Neck = 12,
    LeftCollar = 13,
    RightCollar = 14,
    Head = 15,
    LeftShoulder = 16,
    RightShoulder = 17,
    LeftElbow = 18,
    RightElbow = 19,
    LeftWrist = 20,
    RightWrist = 21,
    // Hand joints start at 22
    LeftIndex1 = 22,
    LeftIndex2 = 23,
    LeftIndex3 = 24,
    LeftMiddle1 = 25,
    LeftMiddle2 = 26,
    LeftMiddle3 = 27,
    LeftPinky1 = 28,
    LeftPinky2 = 29,
    LeftPinky3 = 30,
    LeftRing1 = 31,
    LeftRing2 = 32,
    LeftRing3 = 33,
    LeftThumb1 = 34,
    LeftThumb2 = 35,
    LeftThumb3 = 36,
    RightIndex1 = 37,
    RightIndex2 = 38,
    RightIndex3 = 39,
    RightMiddle1 = 40,
    RightMiddle2 = 41,
    RightMiddle3 = 42,
    RightPinky1 = 43,
    RightPinky2 = 44,
    RightPinky3 = 45,
    RightRing1 = 46,
    RightRing2 = 47,
    RightRing3 = 48,
    RightThumb1 = 49,
    RightThumb2 = 50,
    RightThumb3 = 51,
    // Face
    Jaw = 52,
    LeftEye = 53,
    RightEye = 54,
}

/// Output from SMPL-X forward pass
#[derive(Debug)]
pub struct SMPLXOutput {
    /// Vertex positions (10,475 × 3)
    pub vertices: Vec<[f32; 3]>,
    /// Joint positions (55 × 3)
    pub joints: Vec<JointPositions3D>,
    /// Joint rotations as matrices
    pub joint_transforms: Vec<[[f32; 4]; 4]>,
    /// Face landmarks (if available)
    pub face_landmarks: Option<Vec<[f32; 3]>>,
}

/// Animation sequence using SMPL-X
#[derive(Debug, Clone)]
pub struct SMPLXAnimation {
    /// Timestamps for each frame
    pub timestamps: Vec<f64>,
    /// Pose parameters per frame
    pub poses: Vec<BodyPose>,
    /// Shape (typically constant)
    pub shape: BodyShape,
    /// Translation per frame
    pub translations: Vec<[f32; 3]>,
}

impl SMPLXAnimation {
    /// Create empty animation
    pub fn new(shape: BodyShape) -> Self {
        Self {
            timestamps: Vec::new(),
            poses: Vec::new(),
            shape,
            translations: Vec::new(),
        }
    }

    /// Add a frame
    pub fn add_frame(&mut self, timestamp: f64, pose: BodyPose, translation: [f32; 3]) {
        self.timestamps.push(timestamp);
        self.poses.push(pose);
        self.translations.push(translation);
    }

    /// Get duration
    pub fn duration(&self) -> f64 {
        self.timestamps.last().copied().unwrap_or(0.0)
    }

    /// Get frame count
    pub fn num_frames(&self) -> usize {
        self.timestamps.len()
    }
}

/// Preset poses for specific scenarios
pub struct SMPLXPosePresets;

impl SMPLXPosePresets {
    /// T-pose (default)
    pub fn t_pose() -> BodyPose {
        let mut pose = BodyPose::default();
        // Arms horizontal
        pose.body_pose[16 * 3] = 0.0; // Left shoulder X
        pose.body_pose[16 * 3 + 2] = std::f32::consts::FRAC_PI_2; // Left shoulder Z (outward)
        pose.body_pose[17 * 3] = 0.0; // Right shoulder X
        pose.body_pose[17 * 3 + 2] = -std::f32::consts::FRAC_PI_2; // Right shoulder Z (outward)
        pose
    }

    /// A-pose (arms slightly down)
    pub fn a_pose() -> BodyPose {
        let mut pose = BodyPose::default();
        pose.body_pose[16 * 3 + 2] = std::f32::consts::FRAC_PI_4;
        pose.body_pose[17 * 3 + 2] = -std::f32::consts::FRAC_PI_4;
        pose
    }

    /// Standing idle
    pub fn standing() -> BodyPose {
        let mut pose = BodyPose::default();
        // Slight natural arm position
        pose.body_pose[16 * 3] = 0.1;
        pose.body_pose[17 * 3] = 0.1;
        pose
    }

    /// Walking pose at phase (0-1)
    pub fn walking(phase: f32) -> BodyPose {
        let mut pose = BodyPose::default();
        let angle = phase * 2.0 * std::f32::consts::PI;

        // Hip flexion/extension
        pose.body_pose[3] = 0.5 * angle.sin(); // Left hip
        pose.body_pose[2 * 3] = -0.5 * angle.sin(); // Right hip

        // Knee flexion
        pose.body_pose[4 * 3] = 0.8 * (0.5 - 0.5 * angle.cos()).max(0.0); // Left knee
        pose.body_pose[5 * 3] = 0.8 * (0.5 + 0.5 * angle.cos()).max(0.0); // Right knee

        // Arm swing (opposite to legs)
        pose.body_pose[16 * 3] = -0.3 * angle.sin(); // Left shoulder
        pose.body_pose[17 * 3] = 0.3 * angle.sin(); // Right shoulder

        pose
    }

    /// Hand with tremor
    pub fn tremor_hand(tremor_offset: f32) -> BodyPose {
        let mut pose = BodyPose::default();
        // Add tremor to right hand joints
        for i in 0..12 {
            pose.right_hand_pose[i] = tremor_offset * (i as f32 * 0.1).sin();
        }
        pose
    }
}

impl SMPLXModel {
    /// Create a new SMPL-X model interface
    pub fn new(config: SMPLXConfig) -> Result<Self> {
        // Check for smplx
        let check = Command::new("python3")
            .args(["-c", "import smplx; import torch; print('ok')"])
            .output();

        match check {
            Ok(output) if output.status.success() => {}
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "smplx".to_string(),
                    install_url: "pip install smplx torch".to_string(),
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

    /// Check if SMPL-X is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import smplx"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Forward pass: get vertices and joints from pose/shape
    pub fn forward(&self, pose: &BodyPose, shape: &BodyShape) -> Result<SMPLXOutput> {
        let script = self.generate_forward_script(pose, shape)?;
        self.run_forward(&script)
    }

    /// Generate animation sequence
    pub fn generate_animation(&self, animation: &SMPLXAnimation) -> Result<Vec<SMPLXOutput>> {
        let mut outputs = Vec::new();
        for (pose, _translation) in animation.poses.iter().zip(&animation.translations) {
            outputs.push(self.forward(pose, &animation.shape)?);
        }
        Ok(outputs)
    }

    /// Export mesh to OBJ file
    pub fn export_obj(&self, pose: &BodyPose, shape: &BodyShape, output_path: &Path) -> Result<()> {
        let script = self.generate_export_script(pose, shape, output_path)?;
        self.run_export(&script)
    }

    fn generate_forward_script(&self, pose: &BodyPose, shape: &BodyShape) -> Result<String> {
        let body_pose_json = serde_json::to_string(&pose.body_pose)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;
        let left_hand_json = serde_json::to_string(&pose.left_hand_pose)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;
        let right_hand_json = serde_json::to_string(&pose.right_hand_pose)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;
        let betas_json = serde_json::to_string(&shape.betas)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;
        let expression_json = serde_json::to_string(&shape.expression)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(format!(r#"
import torch
import smplx
import json
import numpy as np

device = 'cuda:{device}' if {device} >= 0 and torch.cuda.is_available() else 'cpu'

# Load model
model = smplx.create(
    model_path='{model_path}',
    model_type='smplx',
    gender='{gender}',
    num_betas={num_betas},
    num_expression_coeffs={num_expression},
    use_pca={use_pca},
    num_pca_comps={num_pca_comps}
).to(device)

# Prepare parameters
global_orient = torch.tensor([{go0}, {go1}, {go2}], dtype=torch.float32, device=device).unsqueeze(0)
body_pose = torch.tensor({body_pose_json}, dtype=torch.float32, device=device).unsqueeze(0)
left_hand_pose = torch.tensor({left_hand_json}, dtype=torch.float32, device=device).unsqueeze(0)
right_hand_pose = torch.tensor({right_hand_json}, dtype=torch.float32, device=device).unsqueeze(0)
jaw_pose = torch.tensor([{jp0}, {jp1}, {jp2}], dtype=torch.float32, device=device).unsqueeze(0)
leye_pose = torch.tensor([{le0}, {le1}, {le2}], dtype=torch.float32, device=device).unsqueeze(0)
reye_pose = torch.tensor([{re0}, {re1}, {re2}], dtype=torch.float32, device=device).unsqueeze(0)
betas = torch.tensor({betas_json}, dtype=torch.float32, device=device).unsqueeze(0)
expression = torch.tensor({expression_json}, dtype=torch.float32, device=device).unsqueeze(0)

# Forward pass
output = model(
    global_orient=global_orient,
    body_pose=body_pose,
    left_hand_pose=left_hand_pose,
    right_hand_pose=right_hand_pose,
    jaw_pose=jaw_pose,
    leye_pose=leye_pose,
    reye_pose=reye_pose,
    betas=betas,
    expression=expression
)

# Extract results
vertices = output.vertices.detach().cpu().numpy()[0].tolist()
joints = output.joints.detach().cpu().numpy()[0].tolist()

# Joint names
joint_names = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

joints_named = []
for i, j in enumerate(joints):
    name = joint_names[i] if i < len(joint_names) else f'joint_{{i}}'
    joints_named.append({{'name': name, 'position': j}})

result = {{
    'num_vertices': len(vertices),
    'num_joints': len(joints),
    'vertices': vertices[:100],  # First 100 for size
    'joints': joints_named
}}
print('RESULT_JSON:' + json.dumps(result))
"#,
            device = self.config.device,
            model_path = self.config.model_path.display(),
            gender = self.config.gender.as_str(),
            num_betas = self.config.num_betas,
            num_expression = self.config.num_expression,
            use_pca = self.config.use_pca,
            num_pca_comps = self.config.num_pca_comps,
            go0 = pose.global_orient[0],
            go1 = pose.global_orient[1],
            go2 = pose.global_orient[2],
            body_pose_json = body_pose_json,
            left_hand_json = left_hand_json,
            right_hand_json = right_hand_json,
            jp0 = pose.jaw_pose[0],
            jp1 = pose.jaw_pose[1],
            jp2 = pose.jaw_pose[2],
            le0 = pose.leye_pose[0],
            le1 = pose.leye_pose[1],
            le2 = pose.leye_pose[2],
            re0 = pose.reye_pose[0],
            re1 = pose.reye_pose[1],
            re2 = pose.reye_pose[2],
            betas_json = betas_json,
            expression_json = expression_json,
        ))
    }

    fn generate_export_script(&self, pose: &BodyPose, shape: &BodyShape, output_path: &Path) -> Result<String> {
        let body_pose_json = serde_json::to_string(&pose.body_pose)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;
        let betas_json = serde_json::to_string(&shape.betas)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(format!(r#"
import torch
import smplx
import trimesh

device = 'cpu'

model = smplx.create(
    model_path='{model_path}',
    model_type='smplx',
    gender='{gender}'
).to(device)

body_pose = torch.tensor({body_pose_json}, dtype=torch.float32).unsqueeze(0)
betas = torch.tensor({betas_json}, dtype=torch.float32).unsqueeze(0)

output = model(body_pose=body_pose, betas=betas)

vertices = output.vertices.detach().numpy()[0]
faces = model.faces

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.export('{output_path}')

print('RESULT_JSON:{{"status": "ok"}}')
"#,
            model_path = self.config.model_path.display(),
            gender = self.config.gender.as_str(),
            body_pose_json = body_pose_json,
            betas_json = betas_json,
            output_path = output_path.display(),
        ))
    }

    fn run_forward(&self, script: &str) -> Result<SMPLXOutput> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("smplx_forward.py");
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
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result found".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        let joints = data["joints"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|j| {
                        let pos = j["position"].as_array()?;
                        Some(JointPositions3D {
                            position: [
                                pos.first()?.as_f64()? as f32,
                                pos.get(1)?.as_f64()? as f32,
                                pos.get(2)?.as_f64()? as f32,
                            ],
                            joint_name: j["name"].as_str()?.to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(SMPLXOutput {
            vertices: Vec::new(), // Full vertices omitted for size
            joints,
            joint_transforms: Vec::new(),
            face_landmarks: None,
        })
    }

    fn run_export(&self, script: &str) -> Result<()> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("smplx_export.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        Ok(())
    }

    /// Convert SMPL-X joints to DPB JointAngles
    pub fn to_joint_angles(&self, output: &SMPLXOutput) -> JointAngles {
        let find_joint = |name: &str| -> Option<[f32; 3]> {
            output.joints.iter()
                .find(|j| j.joint_name == name)
                .map(|j| j.position)
        };

        // Calculate angles from joint positions
        // This is simplified - real implementation would use rotation matrices
        let hip_left = find_joint("left_hip").unwrap_or([0.0; 3]);
        let hip_right = find_joint("right_hip").unwrap_or([0.0; 3]);
        let knee_left = find_joint("left_knee").unwrap_or([0.0; 3]);
        let knee_right = find_joint("right_knee").unwrap_or([0.0; 3]);

        JointAngles {
            hip_flexion: [
                (hip_left[0]).atan2(hip_left[2]),
                (hip_right[0]).atan2(hip_right[2]),
            ],
            knee_flexion: [
                (knee_left[0] - hip_left[0]).atan2(knee_left[2] - hip_left[2]),
                (knee_right[0] - hip_right[0]).atan2(knee_right[2] - hip_right[2]),
            ],
            ankle_dorsiflexion: [0.0; 2],
            shoulder_flexion: [0.0; 2],
            elbow_flexion: [0.0; 2],
            wrist_flexion: [0.0; 2],
            trunk_flexion: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SMPLXConfig::default();
        assert_eq!(config.gender, Gender::Neutral);
        assert_eq!(config.num_betas, 10);
    }

    #[test]
    fn test_pose_default() {
        let pose = BodyPose::default();
        assert_eq!(pose.body_pose.len(), 63);
        assert_eq!(pose.left_hand_pose.len(), 12);
    }

    #[test]
    fn test_presets() {
        let t_pose = SMPLXPosePresets::t_pose();
        assert!(t_pose.body_pose[16 * 3 + 2].abs() > 1.0); // Arms out

        let walk = SMPLXPosePresets::walking(0.25);
        assert!(walk.body_pose[3] != 0.0); // Hip moving
    }

    #[test]
    fn test_animation() {
        let shape = BodyShape::default();
        let mut anim = SMPLXAnimation::new(shape);

        anim.add_frame(0.0, BodyPose::default(), [0.0, 0.0, 0.0]);
        anim.add_frame(1.0, BodyPose::default(), [0.0, 0.0, 1.0]);

        assert_eq!(anim.num_frames(), 2);
        assert_eq!(anim.duration(), 1.0);
    }

    #[test]
    fn test_gender() {
        assert_eq!(Gender::Neutral.as_str(), "neutral");
        assert_eq!(Gender::Male.as_str(), "male");
        assert_eq!(Gender::Female.as_str(), "female");
    }
}
