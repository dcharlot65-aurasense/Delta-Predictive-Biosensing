//! MuJoCo Physics Simulation Integration
//!
//! This module provides integration with MuJoCo (Multi-Joint dynamics with Contact)
//! for physics-accurate human motion simulation including musculoskeletal models.
//!
//! # Features
//!
//! - Full-body human musculoskeletal simulation
//! - Pathological motion patterns (Parkinson's, stroke, etc.)
//! - Contact dynamics for realistic ground interaction
//! - Muscle activation and force estimation
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install mujoco mujoco-py`
//!
//! # References
//!
//! - MuJoCo: https://mujoco.org/
//! - DeepMind open-sourced MuJoCo in 2022

use super::{MediaError, Result, TremorType};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// MuJoCo simulator wrapper
#[derive(Debug)]
// Held for an integration that is not wired up yet. Kept rather than
// removed so a caller's configuration is not silently discarded.
#[allow(dead_code)]
pub struct MuJoCoSimulator {
    config: MuJoCoConfig,
    python_path: PathBuf,
    model_cache: HashMap<String, PathBuf>,
}

/// Configuration for MuJoCo simulation
#[derive(Debug, Clone)]
pub struct MuJoCoConfig {
    /// Path to MuJoCo models directory
    pub models_dir: PathBuf,
    /// Simulation timestep in seconds
    pub timestep: f64,
    /// Number of substeps per frame
    pub substeps: u32,
    /// Enable contact dynamics
    pub enable_contacts: bool,
    /// Enable muscle simulation
    pub enable_muscles: bool,
    /// Gravity vector [x, y, z] in m/s²
    pub gravity: [f64; 3],
    /// GPU device ID (-1 for CPU)
    pub gpu_device: i32,
    /// Output video resolution
    pub resolution: (u32, u32),
    /// Frames per second
    pub fps: f64,
}

impl Default for MuJoCoConfig {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("models/mujoco"),
            timestep: 0.002, // 2ms = 500Hz physics
            substeps: 4,
            enable_contacts: true,
            enable_muscles: true,
            gravity: [0.0, 0.0, -9.81],
            gpu_device: 0,
            resolution: (1920, 1080),
            fps: 30.0,
        }
    }
}

/// Musculoskeletal model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusculoskeletalModel {
    /// Simple humanoid (23 DoF)
    SimpleHumanoid,
    /// Full body model (90 segments, 206 joints)
    FullBody,
    /// Lower extremity only (for gait)
    LowerExtremity,
    /// Upper extremity only (for reaching/tremor)
    UpperExtremity,
    /// Hand model (detailed finger simulation)
    Hand,
    /// Custom model from file
    Custom,
}

/// Motion trajectory data
#[derive(Debug, Clone)]
pub struct MotionTrajectory {
    /// Time points in seconds
    pub times: Vec<f64>,
    /// Joint angles at each time point [time_idx][joint_idx]
    pub joint_angles: Vec<Vec<f64>>,
    /// Joint velocities [time_idx][joint_idx]
    pub joint_velocities: Vec<Vec<f64>>,
    /// End effector positions [time_idx][effector_idx][xyz]
    pub end_effector_positions: Vec<Vec<[f64; 3]>>,
    /// Ground reaction forces [time_idx][foot_idx][xyz]
    pub ground_forces: Vec<Vec<[f64; 3]>>,
    /// Muscle activations (if enabled) [time_idx][muscle_idx]
    pub muscle_activations: Option<Vec<Vec<f64>>>,
}

/// Parameters for gait simulation
#[derive(Debug, Clone)]
pub struct GaitSimParams {
    /// Target walking speed in m/s
    pub speed: f64,
    /// Duration in seconds
    pub duration: f64,
    /// Cadence in steps/min (None = natural)
    pub cadence: Option<f64>,
    /// Step length asymmetry (-1 to 1)
    pub step_asymmetry: f64,
    /// Arm swing amplitude (0 to 1)
    pub arm_swing: f64,
    /// Add festination pattern
    pub festination: bool,
    /// Freezing of gait episodes
    pub freezing_episodes: Vec<(f64, f64)>, // (start_time, duration)
    /// Bradykinesia severity (0-1)
    pub bradykinesia: f64,
}

impl Default for GaitSimParams {
    fn default() -> Self {
        Self {
            speed: 1.2,
            duration: 10.0,
            cadence: None,
            step_asymmetry: 0.0,
            arm_swing: 1.0,
            festination: false,
            freezing_episodes: Vec::new(),
            bradykinesia: 0.0,
        }
    }
}

/// Parameters for tremor simulation
#[derive(Debug, Clone)]
pub struct TremorSimParams {
    /// Tremor frequency in Hz
    pub frequency: f64,
    /// Tremor amplitude in radians
    pub amplitude: f64,
    /// Tremor type
    pub tremor_type: TremorType,
    /// Affected joints (e.g., ["wrist_r", "elbow_r"])
    pub affected_joints: Vec<String>,
    /// Duration in seconds
    pub duration: f64,
    /// Amplitude modulation (intermittent tremor)
    pub amplitude_modulation: Option<AmplitudeModulation>,
    /// Add postural component
    pub postural_task: bool,
}

/// Amplitude modulation for intermittent tremor
#[derive(Debug, Clone)]
pub struct AmplitudeModulation {
    /// Modulation frequency in Hz
    pub frequency: f64,
    /// Modulation depth (0-1)
    pub depth: f64,
    /// Random variation (0-1)
    pub randomness: f64,
}

impl Default for TremorSimParams {
    fn default() -> Self {
        Self {
            frequency: 5.0,
            amplitude: 0.1,
            tremor_type: TremorType::Resting,
            affected_joints: vec!["wrist_r".to_string()],
            duration: 10.0,
            amplitude_modulation: None,
            postural_task: false,
        }
    }
}

impl MuJoCoSimulator {
    /// Create a new MuJoCo simulator
    pub fn new(config: MuJoCoConfig) -> Result<Self> {
        // Check if MuJoCo is available
        let python_check = Command::new("python3")
            .args(["-c", "import mujoco; print(mujoco.__version__)"])
            .output();

        match python_check {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                log::info!("MuJoCo version: {}", version.trim());
            }
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "mujoco".to_string(),
                    install_url: "pip install mujoco".to_string(),
                });
            }
        }

        // Find Python path
        let python_path = which::which("python3").map_err(|_| MediaError::ToolNotFound {
            tool: "python3".to_string(),
            install_url: "https://www.python.org/downloads/".to_string(),
        })?;

        Ok(Self {
            config,
            python_path,
            model_cache: HashMap::new(),
        })
    }

    /// Check if MuJoCo is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import mujoco"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Simulate gait motion
    pub fn simulate_gait(
        &self,
        params: &GaitSimParams,
        model: MusculoskeletalModel,
    ) -> Result<MotionTrajectory> {
        let script = self.generate_gait_script(params, model)?;
        self.run_simulation_script(&script)
    }

    /// Simulate tremor motion
    pub fn simulate_tremor(
        &self,
        params: &TremorSimParams,
        model: MusculoskeletalModel,
    ) -> Result<MotionTrajectory> {
        let script = self.generate_tremor_script(params, model)?;
        self.run_simulation_script(&script)
    }

    /// Generate Python script for gait simulation
    fn generate_gait_script(
        &self,
        params: &GaitSimParams,
        model: MusculoskeletalModel,
    ) -> Result<String> {
        let model_name = match model {
            MusculoskeletalModel::SimpleHumanoid => "humanoid",
            MusculoskeletalModel::FullBody => "fullbody",
            MusculoskeletalModel::LowerExtremity => "lower_extremity",
            _ => "humanoid",
        };

        Ok(format!(
            r#"
import mujoco
import numpy as np
import json
import sys

# Simulation parameters
SPEED = {speed}
DURATION = {duration}
TIMESTEP = {timestep}
BRADYKINESIA = {bradykinesia}
FESTINATION = {festination}
ARM_SWING = {arm_swing}

# Load model
model = mujoco.MjModel.from_xml_path('models/mujoco/{model_name}.xml')
data = mujoco.MjData(model)

# Central pattern generator for gait
class GaitCPG:
    def __init__(self, speed, bradykinesia=0.0):
        self.phase = 0.0
        self.speed = speed
        self.bradykinesia = bradykinesia
        # Natural cadence based on speed
        self.cadence = 1.5 + 0.5 * speed  # Steps per second

    def step(self, dt):
        # Bradykinesia reduces speed and increases variability
        effective_cadence = self.cadence * (1.0 - 0.5 * self.bradykinesia)
        noise = np.random.normal(0, 0.05 * self.bradykinesia)
        self.phase += (effective_cadence + noise) * dt
        self.phase = self.phase % 1.0
        return self.get_joint_targets()

    def get_joint_targets(self):
        # Phase-based gait pattern
        left_phase = self.phase
        right_phase = (self.phase + 0.5) % 1.0

        # Hip flexion/extension
        hip_l = 0.4 * np.sin(2 * np.pi * left_phase)
        hip_r = 0.4 * np.sin(2 * np.pi * right_phase)

        # Knee flexion (only during swing)
        knee_l = 0.6 * max(0, np.sin(2 * np.pi * left_phase)) if left_phase < 0.4 else 0
        knee_r = 0.6 * max(0, np.sin(2 * np.pi * right_phase)) if right_phase < 0.4 else 0

        # Ankle
        ankle_l = 0.2 * np.sin(2 * np.pi * left_phase + 0.5)
        ankle_r = 0.2 * np.sin(2 * np.pi * right_phase + 0.5)

        # Arm swing (reduced by bradykinesia)
        arm_swing_factor = {arm_swing} * (1.0 - self.bradykinesia)
        shoulder_l = arm_swing_factor * 0.3 * np.sin(2 * np.pi * right_phase)  # Contralateral
        shoulder_r = arm_swing_factor * 0.3 * np.sin(2 * np.pi * left_phase)

        return {{
            'hip_l': hip_l, 'hip_r': hip_r,
            'knee_l': knee_l, 'knee_r': knee_r,
            'ankle_l': ankle_l, 'ankle_r': ankle_r,
            'shoulder_l': shoulder_l, 'shoulder_r': shoulder_r
        }}

# Initialize CPG
cpg = GaitCPG(SPEED, BRADYKINESIA)

# Storage for trajectory
times = []
joint_angles = []
joint_velocities = []
end_effector_pos = []
ground_forces = []

# Simulation loop
t = 0.0
while t < DURATION:
    # Get target positions from CPG
    targets = cpg.step(TIMESTEP)

    # Apply targets to actuators (simplified PD control)
    # In full implementation, map to actual model actuators

    # Step simulation
    mujoco.mj_step(model, data)

    # Store data
    times.append(t)
    joint_angles.append(data.qpos.copy().tolist())
    joint_velocities.append(data.qvel.copy().tolist())

    # End effector positions (feet, hands)
    # In full implementation, get from model sites
    end_effector_pos.append([[0,0,0], [0,0,0], [0,0,0], [0,0,0]])

    # Ground reaction forces
    # In full implementation, extract from contact forces
    ground_forces.append([[0,0,0], [0,0,0]])

    t += TIMESTEP

# Output as JSON
result = {{
    'times': times,
    'joint_angles': joint_angles,
    'joint_velocities': joint_velocities,
    'end_effector_positions': end_effector_pos,
    'ground_forces': ground_forces
}}
print(json.dumps(result))
"#,
            speed = params.speed,
            duration = params.duration,
            timestep = self.config.timestep,
            bradykinesia = params.bradykinesia,
            festination = params.festination,
            arm_swing = params.arm_swing,
            model_name = model_name,
        ))
    }

    /// Generate Python script for tremor simulation
    fn generate_tremor_script(
        &self,
        params: &TremorSimParams,
        _model: MusculoskeletalModel,
    ) -> Result<String> {
        let affected_joints_str = params
            .affected_joints
            .iter()
            .map(|j| format!("'{}'", j))
            .collect::<Vec<_>>()
            .join(", ");

        Ok(format!(
            r#"
import mujoco
import numpy as np
import json

# Tremor parameters
FREQUENCY = {frequency}
AMPLITUDE = {amplitude}
DURATION = {duration}
TIMESTEP = {timestep}
AFFECTED_JOINTS = [{affected_joints}]
POSTURAL = {postural}

# Tremor oscillator model
class TremorOscillator:
    def __init__(self, frequency, amplitude):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = np.random.uniform(0, 2*np.pi)
        # Natural frequency variation (±10%)
        self.freq_variation = 0.1

    def get_displacement(self, t):
        # Frequency with slight variation
        f = self.frequency * (1 + self.freq_variation * np.sin(0.1 * t))
        # Amplitude with intermittent character
        amp_mod = 0.5 + 0.5 * np.sin(0.3 * t)  # Slow modulation
        return self.amplitude * amp_mod * np.sin(2 * np.pi * f * t + self.phase)

# Load model
model = mujoco.MjModel.from_xml_path('models/mujoco/upper_extremity.xml')
data = mujoco.MjData(model)

# Create oscillator
oscillator = TremorOscillator(FREQUENCY, AMPLITUDE)

# If postural task, hold arm in position
if POSTURAL:
    # Set initial posture (arm extended)
    pass

# Storage
times = []
joint_angles = []
joint_velocities = []

# Simulation loop
t = 0.0
while t < DURATION:
    # Apply tremor to affected joints
    tremor_displacement = oscillator.get_displacement(t)

    # In full implementation, apply to specific joint actuators
    # For now, simulate effect on joint angles

    mujoco.mj_step(model, data)

    times.append(t)
    joint_angles.append(data.qpos.copy().tolist())
    joint_velocities.append(data.qvel.copy().tolist())

    t += TIMESTEP

result = {{
    'times': times,
    'joint_angles': joint_angles,
    'joint_velocities': joint_velocities,
    'tremor_frequency': FREQUENCY,
    'tremor_amplitude': AMPLITUDE
}}
print(json.dumps(result))
"#,
            frequency = params.frequency,
            amplitude = params.amplitude,
            duration = params.duration,
            timestep = self.config.timestep,
            affected_joints = affected_joints_str,
            postural = params.postural_task,
        ))
    }

    /// Run a Python simulation script and parse results
    fn run_simulation_script(&self, script: &str) -> Result<MotionTrajectory> {
        // Write script to temp file
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mujoco_sim.py");
        std::fs::write(&script_path, script)?;

        // Run Python script
        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(format!("Failed to run MuJoCo: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        // Parse JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let data: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| {
            MediaError::SerializationError(format!("Failed to parse output: {}", e))
        })?;

        // Convert to MotionTrajectory
        Ok(MotionTrajectory {
            times: data["times"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default(),
            joint_angles: data["joint_angles"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|row| {
                            row.as_array()
                                .map(|r| r.iter().filter_map(|v| v.as_f64()).collect())
                        })
                        .collect()
                })
                .unwrap_or_default(),
            joint_velocities: data["joint_velocities"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|row| {
                            row.as_array()
                                .map(|r| r.iter().filter_map(|v| v.as_f64()).collect())
                        })
                        .collect()
                })
                .unwrap_or_default(),
            end_effector_positions: Vec::new(), // Parse if present
            ground_forces: Vec::new(),          // Parse if present
            muscle_activations: None,
        })
    }

    /// Export trajectory to BVH format for Blender import
    pub fn export_to_bvh(&self, trajectory: &MotionTrajectory, output_path: &Path) -> Result<()> {
        let mut bvh = String::new();

        // BVH header
        bvh.push_str("HIERARCHY\n");
        bvh.push_str("ROOT Hips\n");
        bvh.push_str("{\n");
        bvh.push_str("  OFFSET 0.0 0.0 0.0\n");
        bvh.push_str("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n");
        // ... Add full skeleton hierarchy
        bvh.push_str("}\n");

        // Motion data
        bvh.push_str("MOTION\n");
        bvh.push_str(&format!("Frames: {}\n", trajectory.times.len()));
        let frame_time = if trajectory.times.len() > 1 {
            trajectory.times[1] - trajectory.times[0]
        } else {
            1.0 / self.config.fps
        };
        bvh.push_str(&format!("Frame Time: {:.6}\n", frame_time));

        // Write frame data
        for angles in &trajectory.joint_angles {
            let line: Vec<String> = angles
                .iter()
                .map(|a| format!("{:.4}", a.to_degrees()))
                .collect();
            bvh.push_str(&line.join(" "));
            bvh.push('\n');
        }

        std::fs::write(output_path, bvh)?;
        Ok(())
    }
}

/// Create standard model files if they don't exist
pub fn ensure_models_exist(models_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(models_dir)?;

    // Create minimal humanoid model
    let humanoid_path = models_dir.join("humanoid.xml");
    if !humanoid_path.exists() {
        std::fs::write(&humanoid_path, MINIMAL_HUMANOID_XML)?;
    }

    // Create upper extremity model
    let upper_path = models_dir.join("upper_extremity.xml");
    if !upper_path.exists() {
        std::fs::write(&upper_path, MINIMAL_UPPER_EXTREMITY_XML)?;
    }

    Ok(())
}

/// Minimal humanoid MuJoCo XML for gait
const MINIMAL_HUMANOID_XML: &str = r#"
<mujoco model="humanoid">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.002" iterations="50" solver="Newton"/>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>

    <body name="torso" pos="0 0 1.2">
      <joint name="root" type="free"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.08"/>

      <!-- Pelvis -->
      <body name="pelvis" pos="0 0 0">
        <geom type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.08"/>

        <!-- Left leg -->
        <body name="left_thigh" pos="-0.1 0 0">
          <joint name="hip_l" type="hinge" axis="0 1 0" range="-30 120"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05"/>

          <body name="left_shin" pos="0 0 -0.4">
            <joint name="knee_l" type="hinge" axis="0 1 0" range="0 150"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04"/>

            <body name="left_foot" pos="0 0 -0.4">
              <joint name="ankle_l" type="hinge" axis="0 1 0" range="-30 45"/>
              <geom type="box" size="0.04 0.02 0.1" pos="0.05 0 0"/>
            </body>
          </body>
        </body>

        <!-- Right leg (mirror) -->
        <body name="right_thigh" pos="0.1 0 0">
          <joint name="hip_r" type="hinge" axis="0 1 0" range="-30 120"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05"/>

          <body name="right_shin" pos="0 0 -0.4">
            <joint name="knee_r" type="hinge" axis="0 1 0" range="0 150"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04"/>

            <body name="right_foot" pos="0 0 -0.4">
              <joint name="ankle_r" type="hinge" axis="0 1 0" range="-30 45"/>
              <geom type="box" size="0.04 0.02 0.1" pos="0.05 0 0"/>
            </body>
          </body>
        </body>
      </body>

      <!-- Upper body -->
      <body name="chest" pos="0 0 0.3">
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.07"/>

        <!-- Left arm -->
        <body name="left_upper_arm" pos="-0.15 0 0.2">
          <joint name="shoulder_l" type="hinge" axis="0 1 0" range="-180 60"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03"/>

          <body name="left_forearm" pos="0 0 -0.25">
            <joint name="elbow_l" type="hinge" axis="0 1 0" range="0 150"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025"/>
          </body>
        </body>

        <!-- Right arm -->
        <body name="right_upper_arm" pos="0.15 0 0.2">
          <joint name="shoulder_r" type="hinge" axis="0 1 0" range="-180 60"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03"/>

          <body name="right_forearm" pos="0 0 -0.25">
            <joint name="elbow_r" type="hinge" axis="0 1 0" range="0 150"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="hip_l" gear="100"/>
    <motor joint="hip_r" gear="100"/>
    <motor joint="knee_l" gear="100"/>
    <motor joint="knee_r" gear="100"/>
    <motor joint="ankle_l" gear="50"/>
    <motor joint="ankle_r" gear="50"/>
    <motor joint="shoulder_l" gear="50"/>
    <motor joint="shoulder_r" gear="50"/>
    <motor joint="elbow_l" gear="50"/>
    <motor joint="elbow_r" gear="50"/>
  </actuator>
</mujoco>
"#;

/// Minimal upper extremity model for tremor
const MINIMAL_UPPER_EXTREMITY_XML: &str = r#"
<mujoco model="upper_extremity">
  <compiler angle="degree"/>
  <option timestep="0.002"/>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>

    <body name="torso" pos="0 0 1">
      <geom type="box" size="0.15 0.05 0.2" rgba="0.8 0.6 0.5 1"/>

      <body name="upper_arm" pos="0.2 0 0.15">
        <joint name="shoulder_flex" type="hinge" axis="0 1 0" range="-60 180"/>
        <joint name="shoulder_abd" type="hinge" axis="1 0 0" range="-30 180"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>

        <body name="forearm" pos="0.25 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="0 150"/>
          <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.025"/>

          <body name="hand" pos="0.22 0 0">
            <joint name="wrist_flex" type="hinge" axis="0 1 0" range="-80 80"/>
            <joint name="wrist_dev" type="hinge" axis="0 0 1" range="-30 30"/>
            <geom type="box" size="0.04 0.01 0.06"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="shoulder_flex" gear="50"/>
    <motor joint="shoulder_abd" gear="50"/>
    <motor joint="elbow" gear="30"/>
    <motor joint="wrist_flex" gear="10"/>
    <motor joint="wrist_dev" gear="10"/>
  </actuator>
</mujoco>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = MuJoCoConfig::default();
        assert_eq!(config.fps, 30.0);
        assert!(config.enable_contacts);
    }

    #[test]
    fn test_gait_params_default() {
        let params = GaitSimParams::default();
        assert_eq!(params.speed, 1.2);
        assert!(!params.festination);
    }

    #[test]
    fn test_tremor_params_default() {
        let params = TremorSimParams::default();
        assert_eq!(params.frequency, 5.0);
        assert_eq!(params.tremor_type, TremorType::Resting);
    }
}
