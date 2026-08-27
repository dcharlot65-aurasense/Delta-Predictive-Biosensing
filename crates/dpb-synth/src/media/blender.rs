//! Blender 3D Rendering Pipeline
//!
//! This module provides advanced integration with Blender for high-quality
//! 3D rendering of motion capture and synthetic human animations.
//!
//! # Features
//!
//! - Python scripting for procedural animation
//! - Import from MuJoCo/OpenSim trajectories
//! - Realistic human models and materials
//! - Camera control for multi-view rendering
//! - Background/environment customization
//!
//! # Requirements
//!
//! - Blender 3.x or 4.x
//! - Available in PATH or specify path explicitly
//!
//! # References
//!
//! - Blender Python API: https://docs.blender.org/api/current/

use super::{MediaError, MotionTrajectory, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Blender rendering pipeline
#[derive(Debug)]
pub struct BlenderRenderer {
    config: BlenderConfig,
    blender_path: PathBuf,
}

/// Configuration for Blender rendering
#[derive(Debug, Clone)]
pub struct BlenderConfig {
    /// Path to Blender executable (if not in PATH)
    pub blender_executable: Option<PathBuf>,
    /// Output directory for rendered files
    pub output_dir: PathBuf,
    /// Render resolution
    pub resolution: (u32, u32),
    /// Frames per second
    pub fps: u32,
    /// Render engine (EEVEE for fast, Cycles for quality)
    pub render_engine: RenderEngine,
    /// Number of samples (for Cycles)
    pub samples: u32,
    /// Enable GPU rendering
    pub use_gpu: bool,
    /// Background color/HDRI
    pub background: Background,
    /// Camera setup
    pub camera: CameraSetup,
}

/// Render engine selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderEngine {
    /// EEVEE - fast real-time renderer
    #[default]
    Eevee,
    /// Cycles - path-traced ray tracer
    Cycles,
    /// Workbench - simple solid view
    Workbench,
}

/// Background configuration
#[derive(Debug, Clone)]
pub enum Background {
    /// Solid color (RGB)
    SolidColor([f32; 3]),
    /// HDRI environment map
    Hdri(PathBuf),
    /// Studio lighting preset
    Studio(StudioPreset),
    /// Transparent background
    Transparent,
}

/// Studio lighting presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StudioPreset {
    #[default]
    Basic,
    ThreePoint,
    Rim,
    Outdoor,
    Clinical,
}

/// Camera setup configuration
#[derive(Debug, Clone)]
pub struct CameraSetup {
    /// Camera type
    pub camera_type: CameraType,
    /// Position [x, y, z]
    pub position: [f32; 3],
    /// Look-at target [x, y, z]
    pub target: [f32; 3],
    /// Field of view in degrees
    pub fov: f32,
    /// Enable motion tracking (follow subject)
    pub track_subject: bool,
}

/// Camera types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraType {
    #[default]
    Perspective,
    Orthographic,
    Panoramic,
}

impl Default for CameraSetup {
    fn default() -> Self {
        Self {
            camera_type: CameraType::Perspective,
            position: [3.0, -3.0, 2.0],
            target: [0.0, 0.0, 1.0],
            fov: 50.0,
            track_subject: true,
        }
    }
}

impl Default for Background {
    fn default() -> Self {
        Background::Studio(StudioPreset::Basic)
    }
}

impl Default for BlenderConfig {
    fn default() -> Self {
        Self {
            blender_executable: None,
            output_dir: PathBuf::from("output/video"),
            resolution: (1920, 1080),
            fps: 30,
            render_engine: RenderEngine::Eevee,
            samples: 64,
            use_gpu: true,
            background: Background::default(),
            camera: CameraSetup::default(),
        }
    }
}

/// Output from rendering
#[derive(Debug)]
pub struct RenderOutput {
    /// Path to rendered video file
    pub video_path: PathBuf,
    /// Individual frame paths (if image sequence)
    pub frame_paths: Option<Vec<PathBuf>>,
    /// Total duration in seconds
    pub duration: f64,
    /// Render time in seconds
    pub render_time: f64,
    /// Resolution used
    pub resolution: (u32, u32),
}

/// Animation script for Blender
#[derive(Debug, Clone)]
pub struct AnimationScript {
    /// Frame data
    pub frames: Vec<AnimationFrame>,
    /// Armature/skeleton name
    pub armature_name: String,
    /// Bone mapping from data to Blender bones
    pub bone_mapping: std::collections::HashMap<String, String>,
}

/// Single animation frame
#[derive(Debug, Clone, serde::Serialize)]
pub struct AnimationFrame {
    /// Frame number
    pub frame: u32,
    /// Bone rotations (bone_name -> euler angles in radians)
    pub bone_rotations: std::collections::HashMap<String, [f32; 3]>,
    /// Root position
    pub root_position: [f32; 3],
    /// Root rotation
    pub root_rotation: [f32; 3],
}

/// Human model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HumanModel {
    /// Simple stick figure
    #[default]
    StickFigure,
    /// Basic humanoid mesh
    BasicHumanoid,
    /// Detailed human (MakeHuman export)
    DetailedHuman,
    /// MBLab character
    MBLab,
    /// Custom model from file
    Custom,
}

impl BlenderRenderer {
    /// Create a new Blender renderer
    pub fn new(config: BlenderConfig) -> Result<Self> {
        // Find Blender executable
        let blender_path = if let Some(ref path) = config.blender_executable {
            if path.exists() {
                path.clone()
            } else {
                return Err(MediaError::ToolNotFound {
                    tool: "blender".to_string(),
                    install_url: "https://www.blender.org/download/".to_string(),
                });
            }
        } else {
            which::which("blender").map_err(|_| MediaError::ToolNotFound {
                tool: "blender".to_string(),
                install_url: "https://www.blender.org/download/".to_string(),
            })?
        };

        // Verify Blender works
        let version_check = Command::new(&blender_path).args(["--version"]).output();

        match version_check {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                log::info!("Blender: {}", version.lines().next().unwrap_or("unknown"));
            }
            _ => {
                return Err(MediaError::ExecutionFailed(
                    "Failed to run Blender".to_string(),
                ));
            }
        }

        // Ensure output directory exists
        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self {
            config,
            blender_path,
        })
    }

    /// Check if Blender is available
    pub fn is_available() -> bool {
        which::which("blender").is_ok()
    }

    /// Render motion trajectory to video
    pub fn render_motion(
        &self,
        trajectory: &MotionTrajectory,
        model: HumanModel,
        output_name: &str,
    ) -> Result<RenderOutput> {
        let animation = self.convert_trajectory_to_animation(trajectory)?;
        self.render_animation(&animation, model, output_name)
    }

    /// Render animation script to video
    pub fn render_animation(
        &self,
        animation: &AnimationScript,
        model: HumanModel,
        output_name: &str,
    ) -> Result<RenderOutput> {
        let output_path = self.config.output_dir.join(format!("{}.mp4", output_name));
        let script = self.generate_render_script(animation, model, &output_path)?;

        // Write script to temp file
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("blender_render.py");
        std::fs::write(&script_path, &script)?;

        // Run Blender in background mode
        let start_time = std::time::Instant::now();
        let output = Command::new(&self.blender_path)
            .args(["--background", "--python", script_path.to_str().unwrap()])
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        let render_time = start_time.elapsed().as_secs_f64();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::ExecutionFailed(format!(
                "Blender render failed: {}",
                stderr
            )));
        }

        let duration = animation.frames.len() as f64 / self.config.fps as f64;

        Ok(RenderOutput {
            video_path: output_path,
            frame_paths: None,
            duration,
            render_time,
            resolution: self.config.resolution,
        })
    }

    /// Convert motion trajectory to animation script
    fn convert_trajectory_to_animation(
        &self,
        trajectory: &MotionTrajectory,
    ) -> Result<AnimationScript> {
        let mut frames = Vec::new();
        let fps = self.config.fps as f64;

        for (i, time) in trajectory.times.iter().enumerate() {
            let frame_num = (time * fps) as u32;

            let angles = if i < trajectory.joint_angles.len() {
                &trajectory.joint_angles[i]
            } else {
                continue;
            };

            let mut bone_rotations = std::collections::HashMap::new();

            // Map joint angles to bone rotations (simplified mapping)
            // In production, this would use proper skeleton mapping
            if angles.len() >= 10 {
                bone_rotations.insert("hip.L".to_string(), [angles[0] as f32, 0.0, 0.0]);
                bone_rotations.insert("hip.R".to_string(), [angles[1] as f32, 0.0, 0.0]);
                bone_rotations.insert("knee.L".to_string(), [angles[2] as f32, 0.0, 0.0]);
                bone_rotations.insert("knee.R".to_string(), [angles[3] as f32, 0.0, 0.0]);
                bone_rotations.insert("ankle.L".to_string(), [angles[4] as f32, 0.0, 0.0]);
                bone_rotations.insert("ankle.R".to_string(), [angles[5] as f32, 0.0, 0.0]);
                bone_rotations.insert("shoulder.L".to_string(), [angles[6] as f32, 0.0, 0.0]);
                bone_rotations.insert("shoulder.R".to_string(), [angles[7] as f32, 0.0, 0.0]);
                bone_rotations.insert("elbow.L".to_string(), [angles[8] as f32, 0.0, 0.0]);
                bone_rotations.insert("elbow.R".to_string(), [angles[9] as f32, 0.0, 0.0]);
            }

            frames.push(AnimationFrame {
                frame: frame_num,
                bone_rotations,
                root_position: [0.0, frame_num as f32 * 0.05, 0.0], // Forward walking
                root_rotation: [0.0, 0.0, 0.0],
            });
        }

        Ok(AnimationScript {
            frames,
            armature_name: "Armature".to_string(),
            bone_mapping: std::collections::HashMap::new(),
        })
    }

    /// Generate Blender Python script for rendering
    fn generate_render_script(
        &self,
        animation: &AnimationScript,
        model: HumanModel,
        output_path: &Path,
    ) -> Result<String> {
        let background_code = match &self.config.background {
            Background::SolidColor(rgb) => format!(
                r#"
world.node_tree.nodes["Background"].inputs[0].default_value = ({}, {}, {}, 1)
"#,
                rgb[0], rgb[1], rgb[2]
            ),
            Background::Transparent => r#"
bpy.context.scene.render.film_transparent = True
"#
            .to_string(),
            Background::Studio(preset) => format!(
                r#"
# Studio lighting: {:?}
bpy.ops.object.light_add(type='AREA', location=(2, -2, 3))
key_light = bpy.context.object
key_light.data.energy = 500

bpy.ops.object.light_add(type='AREA', location=(-2, -2, 2))
fill_light = bpy.context.object
fill_light.data.energy = 200

bpy.ops.object.light_add(type='AREA', location=(0, 2, 2))
back_light = bpy.context.object
back_light.data.energy = 300
"#,
                preset
            ),
            Background::Hdri(path) => format!(
                r#"
# Load HDRI
world.use_nodes = True
env_tex = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
env_tex.image = bpy.data.images.load('{}')
world.node_tree.links.new(env_tex.outputs[0], world.node_tree.nodes["Background"].inputs[0])
"#,
                path.display()
            ),
        };

        let model_code = match model {
            HumanModel::StickFigure => STICK_FIGURE_CODE,
            HumanModel::BasicHumanoid => BASIC_HUMANOID_CODE,
            _ => STICK_FIGURE_CODE,
        };

        let render_engine = match self.config.render_engine {
            RenderEngine::Eevee => "BLENDER_EEVEE",
            RenderEngine::Cycles => "CYCLES",
            RenderEngine::Workbench => "BLENDER_WORKBENCH",
        };

        // Serialize animation frames
        let frames_json = serde_json::to_string(&animation.frames)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(format!(
            r#"
import bpy
import json
import math

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Scene setup
scene = bpy.context.scene
scene.render.resolution_x = {res_x}
scene.render.resolution_y = {res_y}
scene.render.fps = {fps}
scene.render.engine = '{render_engine}'

if '{render_engine}' == 'CYCLES':
    scene.cycles.samples = {samples}
    if {use_gpu}:
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'

# World/Background
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
{background_code}

# Create human model
{model_code}

# Load animation data
animation_data = json.loads('''{frames_json}''')

# Apply animation
armature = bpy.data.objects.get('Armature')
if armature:
    for frame_data in animation_data:
        frame_num = frame_data['frame']
        scene.frame_set(frame_num)

        # Set root position
        root_pos = frame_data['root_position']
        armature.location = (root_pos[0], root_pos[1], root_pos[2])
        armature.keyframe_insert(data_path='location', frame=frame_num)

        # Set bone rotations
        for bone_name, rotation in frame_data['bone_rotations'].items():
            if bone_name in armature.pose.bones:
                bone = armature.pose.bones[bone_name]
                bone.rotation_euler = (rotation[0], rotation[1], rotation[2])
                bone.keyframe_insert(data_path='rotation_euler', frame=frame_num)

# Set frame range
scene.frame_start = 0
scene.frame_end = max(f['frame'] for f in animation_data) if animation_data else 1

# Camera setup
bpy.ops.object.camera_add(location=({cam_pos_x}, {cam_pos_y}, {cam_pos_z}))
camera = bpy.context.object
scene.camera = camera

# Point camera at target
target_loc = ({cam_target_x}, {cam_target_y}, {cam_target_z})
direction = [target_loc[i] - camera.location[i] for i in range(3)]
camera.rotation_euler = (
    math.atan2(direction[2], math.sqrt(direction[0]**2 + direction[1]**2)) + math.pi/2,
    0,
    math.atan2(direction[0], -direction[1])
)
camera.data.angle = math.radians({fov})

# Track constraint for following subject
if {track_subject}:
    track = camera.constraints.new(type='TRACK_TO')
    if armature:
        track.target = armature
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

# Floor
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
floor = bpy.context.object
floor_mat = bpy.data.materials.new("Floor")
floor_mat.use_nodes = True
floor_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)
floor.data.materials.append(floor_mat)

# Output settings
scene.render.filepath = '{output_path}'
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'

# Render animation
bpy.ops.render.render(animation=True)

print('Render complete!')
"#,
            res_x = self.config.resolution.0,
            res_y = self.config.resolution.1,
            fps = self.config.fps,
            render_engine = render_engine,
            samples = self.config.samples,
            use_gpu = self.config.use_gpu,
            background_code = background_code,
            model_code = model_code,
            frames_json = frames_json,
            cam_pos_x = self.config.camera.position[0],
            cam_pos_y = self.config.camera.position[1],
            cam_pos_z = self.config.camera.position[2],
            cam_target_x = self.config.camera.target[0],
            cam_target_y = self.config.camera.target[1],
            cam_target_z = self.config.camera.target[2],
            fov = self.config.camera.fov,
            track_subject = self.config.camera.track_subject,
            output_path = output_path.display(),
        ))
    }

    /// Import BVH motion capture file
    pub fn import_bvh(&self, bvh_path: &Path, output_name: &str) -> Result<RenderOutput> {
        let output_path = self.config.output_dir.join(format!("{}.mp4", output_name));

        let script = format!(
            r#"
import bpy

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import BVH
bpy.ops.import_anim.bvh(filepath='{}')

# Setup render
scene = bpy.context.scene
scene.render.resolution_x = {}
scene.render.resolution_y = {}
scene.render.filepath = '{}'
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'

# Add camera
bpy.ops.object.camera_add(location=(5, -5, 3))
scene.camera = bpy.context.object

# Render
bpy.ops.render.render(animation=True)
"#,
            bvh_path.display(),
            self.config.resolution.0,
            self.config.resolution.1,
            output_path.display(),
        );

        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("blender_bvh.py");
        std::fs::write(&script_path, &script)?;

        let start_time = std::time::Instant::now();
        let output = Command::new(&self.blender_path)
            .args(["--background", "--python", script_path.to_str().unwrap()])
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        let render_time = start_time.elapsed().as_secs_f64();

        if !output.status.success() {
            return Err(MediaError::ExecutionFailed(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ));
        }

        Ok(RenderOutput {
            video_path: output_path,
            frame_paths: None,
            duration: 0.0, // Would need to parse from BVH
            render_time,
            resolution: self.config.resolution,
        })
    }
}

/// Code to create a stick figure armature in Blender
const STICK_FIGURE_CODE: &str = r#"
# Create stick figure armature
bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
armature = bpy.context.object
armature.name = 'Armature'

# Get edit bones
arm = armature.data
bones = arm.edit_bones

# Remove default bone
bones.remove(bones[0])

# Create skeleton
# Spine
root = bones.new('root')
root.head = (0, 0, 0)
root.tail = (0, 0, 0.2)

spine = bones.new('spine')
spine.head = (0, 0, 0.2)
spine.tail = (0, 0, 0.5)
spine.parent = root

chest = bones.new('chest')
chest.head = (0, 0, 0.5)
chest.tail = (0, 0, 0.8)
chest.parent = spine

neck = bones.new('neck')
neck.head = (0, 0, 0.8)
neck.tail = (0, 0, 0.9)
neck.parent = chest

head = bones.new('head')
head.head = (0, 0, 0.9)
head.tail = (0, 0, 1.1)
head.parent = neck

# Left leg
hip_l = bones.new('hip.L')
hip_l.head = (-0.1, 0, 0.2)
hip_l.tail = (-0.1, 0, -0.2)
hip_l.parent = root

knee_l = bones.new('knee.L')
knee_l.head = (-0.1, 0, -0.2)
knee_l.tail = (-0.1, 0, -0.6)
knee_l.parent = hip_l

ankle_l = bones.new('ankle.L')
ankle_l.head = (-0.1, 0, -0.6)
ankle_l.tail = (-0.1, 0.1, -0.65)
ankle_l.parent = knee_l

# Right leg
hip_r = bones.new('hip.R')
hip_r.head = (0.1, 0, 0.2)
hip_r.tail = (0.1, 0, -0.2)
hip_r.parent = root

knee_r = bones.new('knee.R')
knee_r.head = (0.1, 0, -0.2)
knee_r.tail = (0.1, 0, -0.6)
knee_r.parent = hip_r

ankle_r = bones.new('ankle.R')
ankle_r.head = (0.1, 0, -0.6)
ankle_r.tail = (0.1, 0.1, -0.65)
ankle_r.parent = knee_r

# Left arm
shoulder_l = bones.new('shoulder.L')
shoulder_l.head = (-0.15, 0, 0.75)
shoulder_l.tail = (-0.4, 0, 0.75)
shoulder_l.parent = chest

elbow_l = bones.new('elbow.L')
elbow_l.head = (-0.4, 0, 0.75)
elbow_l.tail = (-0.4, 0, 0.5)
elbow_l.parent = shoulder_l

wrist_l = bones.new('wrist.L')
wrist_l.head = (-0.4, 0, 0.5)
wrist_l.tail = (-0.4, 0, 0.4)
wrist_l.parent = elbow_l

# Right arm
shoulder_r = bones.new('shoulder.R')
shoulder_r.head = (0.15, 0, 0.75)
shoulder_r.tail = (0.4, 0, 0.75)
shoulder_r.parent = chest

elbow_r = bones.new('elbow.R')
elbow_r.head = (0.4, 0, 0.75)
elbow_r.tail = (0.4, 0, 0.5)
elbow_r.parent = shoulder_r

wrist_r = bones.new('wrist.R')
wrist_r.head = (0.4, 0, 0.5)
wrist_r.tail = (0.4, 0, 0.4)
wrist_r.parent = elbow_r

# Exit edit mode
bpy.ops.object.mode_set(mode='OBJECT')

# Add stick visualization
bpy.ops.object.select_all(action='DESELECT')
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
armature.data.display_type = 'STICK'

# Scale up
armature.scale = (1.7, 1.7, 1.7)
bpy.ops.object.transform_apply(scale=True)
"#;

/// Code to create basic humanoid mesh
const BASIC_HUMANOID_CODE: &str = r#"
# Create basic humanoid with metaballs
import bpy

# First create armature (same as stick figure)
bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 1))
armature = bpy.context.object
armature.name = 'Armature'

# Create body mesh using primitives
def add_body_part(name, loc, scale, parent_bone=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=loc, scale=scale)
    obj = bpy.context.object
    obj.name = name
    # Add skin material
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.6, 0.5, 1)
    obj.data.materials.append(mat)
    return obj

# Torso
add_body_part("torso", (0, 0, 1.2), (0.25, 0.15, 0.35))

# Head
add_body_part("head", (0, 0, 1.7), (0.12, 0.12, 0.15))

# Arms
add_body_part("upper_arm_L", (-0.35, 0, 1.3), (0.05, 0.05, 0.15))
add_body_part("forearm_L", (-0.35, 0, 1.0), (0.04, 0.04, 0.14))
add_body_part("upper_arm_R", (0.35, 0, 1.3), (0.05, 0.05, 0.15))
add_body_part("forearm_R", (0.35, 0, 1.0), (0.04, 0.04, 0.14))

# Legs
add_body_part("thigh_L", (-0.1, 0, 0.7), (0.08, 0.08, 0.22))
add_body_part("shin_L", (-0.1, 0, 0.3), (0.05, 0.05, 0.2))
add_body_part("thigh_R", (0.1, 0, 0.7), (0.08, 0.08, 0.22))
add_body_part("shin_R", (0.1, 0, 0.3), (0.05, 0.05, 0.2))
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BlenderConfig::default();
        assert_eq!(config.resolution, (1920, 1080));
        assert_eq!(config.fps, 30);
    }

    #[test]
    fn test_camera_default() {
        let camera = CameraSetup::default();
        assert_eq!(camera.fov, 50.0);
        assert!(camera.track_subject);
    }
}
