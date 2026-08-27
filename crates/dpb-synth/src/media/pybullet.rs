//! PyBullet Physics Simulation Integration
//!
//! PyBullet is a Python binding for the Bullet Physics engine, providing
//! a lightweight alternative to MuJoCo for physics simulation.
//!
//! # Features
//!
//! - Real-time physics simulation
//! - Humanoid models (URDF support)
//! - OpenAI Gym locomotion environments
//! - No commercial license required (zlib)
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install pybullet`

use super::{GaitGroundTruth, MediaError, MotionTrajectory, Result};
use std::path::PathBuf;
use std::process::Command;

/// PyBullet physics simulator
#[derive(Debug)]
pub struct PyBulletSimulator {
    config: PyBulletConfig,
    python_path: PathBuf,
}

/// Configuration for PyBullet simulation
#[derive(Debug, Clone)]
pub struct PyBulletConfig {
    /// Output directory for trajectories
    pub output_dir: PathBuf,
    /// Time step in seconds
    pub time_step: f64,
    /// Use GUI for visualization (slower)
    pub use_gui: bool,
    /// Gravity vector [x, y, z]
    pub gravity: [f64; 3],
    /// Number of solver iterations
    pub solver_iterations: u32,
}

impl Default for PyBulletConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("output/pybullet"),
            time_step: 1.0 / 240.0,
            use_gui: false,
            gravity: [0.0, 0.0, -9.81],
            solver_iterations: 50,
        }
    }
}

/// Humanoid model types available in PyBullet
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidModel {
    /// Simple humanoid from pybullet_data
    SimpleHumanoid,
    /// Humanoid from OpenAI Gym
    GymHumanoid,
    /// Cassie biped robot
    Cassie,
    /// Custom URDF model
    Custom,
}

impl HumanoidModel {
    fn urdf_path(&self) -> &'static str {
        match self {
            HumanoidModel::SimpleHumanoid => "humanoid/humanoid.urdf",
            HumanoidModel::GymHumanoid => "humanoid.xml",
            HumanoidModel::Cassie => "cassie/urdf/cassie.urdf",
            HumanoidModel::Custom => "",
        }
    }
}

/// Gait simulation parameters for PyBullet
#[derive(Debug, Clone)]
pub struct PyBulletGaitParams {
    /// Duration in seconds
    pub duration: f64,
    /// Target walking speed in m/s
    pub target_speed: f64,
    /// Step frequency in Hz
    pub step_frequency: f64,
    /// Add shuffling gait (Parkinsonian)
    pub shuffling: bool,
    /// Shuffle severity (0-1)
    pub shuffle_severity: f64,
    /// Festination (progressive speed increase)
    pub festination: bool,
    /// Freezing of gait probability per step
    pub fog_probability: f64,
}

impl Default for PyBulletGaitParams {
    fn default() -> Self {
        Self {
            duration: 5.0,
            target_speed: 1.0,
            step_frequency: 1.8,
            shuffling: false,
            shuffle_severity: 0.0,
            festination: false,
            fog_probability: 0.0,
        }
    }
}

/// Tremor simulation parameters
#[derive(Debug, Clone)]
pub struct PyBulletTremorParams {
    /// Duration in seconds
    pub duration: f64,
    /// Tremor frequency in Hz
    pub frequency: f64,
    /// Tremor amplitude in radians
    pub amplitude: f64,
    /// Affected joints
    pub affected_joints: Vec<String>,
    /// Rest vs action tremor
    pub is_resting: bool,
    /// Add noise to frequency
    pub frequency_variation: f64,
}

impl Default for PyBulletTremorParams {
    fn default() -> Self {
        Self {
            duration: 5.0,
            frequency: 5.0,
            amplitude: 0.05,
            affected_joints: vec!["right_wrist".to_string(), "right_hand".to_string()],
            is_resting: true,
            frequency_variation: 0.5,
        }
    }
}

/// Output from PyBullet simulation
#[derive(Debug)]
pub struct PyBulletOutput {
    /// Joint angle trajectory
    pub trajectory: MotionTrajectory,
    /// Ground truth gait parameters
    pub gait_ground_truth: Option<GaitGroundTruth>,
    /// End-effector positions over time
    pub end_effector_positions: Vec<EndEffectorState>,
    /// Center of mass trajectory
    pub com_trajectory: Vec<[f64; 3]>,
    /// Contact forces over time
    pub contact_forces: Vec<ContactForce>,
}

/// End effector state at a time point
#[derive(Debug, Clone)]
pub struct EndEffectorState {
    /// Time in seconds
    pub time: f64,
    /// Position [x, y, z] in meters
    pub position: [f64; 3],
    /// Velocity [vx, vy, vz] in m/s
    pub velocity: [f64; 3],
    /// End effector name
    pub name: String,
}

/// Contact force at a time point
#[derive(Debug, Clone)]
pub struct ContactForce {
    /// Time in seconds
    pub time: f64,
    /// Force magnitude in Newtons
    pub force: f64,
    /// Contact point [x, y, z]
    pub contact_point: [f64; 3],
    /// Body name in contact
    pub body_name: String,
}

impl PyBulletSimulator {
    /// Create a new PyBullet simulator
    pub fn new(config: PyBulletConfig) -> Result<Self> {
        // Check for pybullet
        let check = Command::new("python3")
            .args(["-c", "import pybullet; print('ok')"])
            .output();

        match check {
            Ok(output) if output.status.success() => {}
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "pybullet".to_string(),
                    install_url: "pip install pybullet".to_string(),
                });
            }
        }

        let python_path = which::which("python3").map_err(|_| MediaError::ToolNotFound {
            tool: "python3".to_string(),
            install_url: "https://www.python.org/downloads/".to_string(),
        })?;

        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self {
            config,
            python_path,
        })
    }

    /// Check if PyBullet is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import pybullet"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Simulate gait with optional pathological features
    pub fn simulate_gait(
        &self,
        model: HumanoidModel,
        params: &PyBulletGaitParams,
    ) -> Result<PyBulletOutput> {
        let script = self.generate_gait_script(model, params)?;
        self.run_simulation(&script)
    }

    /// Simulate tremor
    pub fn simulate_tremor(
        &self,
        model: HumanoidModel,
        params: &PyBulletTremorParams,
    ) -> Result<PyBulletOutput> {
        let script = self.generate_tremor_script(model, params)?;
        self.run_simulation(&script)
    }

    fn generate_gait_script(
        &self,
        model: HumanoidModel,
        params: &PyBulletGaitParams,
    ) -> Result<String> {
        let model_path = model.urdf_path();

        Ok(format!(
            r#"
import pybullet as p
import pybullet_data
import numpy as np
import json
import time

# Setup
if {use_gui}:
    physics_client = p.connect(p.GUI)
else:
    physics_client = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity({gx}, {gy}, {gz})
p.setTimeStep({time_step})
p.setPhysicsEngineParameter(numSolverIterations={solver_iters})

# Load ground plane
plane_id = p.loadURDF("plane.urdf")

# Load humanoid
humanoid_id = p.loadURDF("{model_path}", [0, 0, 1], useFixedBase=False)

# Get joint info
num_joints = p.getNumJoints(humanoid_id)
joint_names = []
joint_indices = []
for i in range(num_joints):
    info = p.getJointInfo(humanoid_id, i)
    if info[2] != p.JOINT_FIXED:
        joint_names.append(info[1].decode('utf-8'))
        joint_indices.append(i)

# Gait parameters
duration = {duration}
target_speed = {target_speed}
step_freq = {step_frequency}
shuffling = {shuffling}
shuffle_severity = {shuffle_severity}
festination = {festination}
fog_prob = {fog_probability}

# Simulation
dt = {time_step}
steps = int(duration / dt)
trajectory = []
com_trajectory = []
contact_forces = []
end_effector_states = []

print("Simulating gait...")
start_time = time.time()

for step in range(steps):
    t = step * dt

    # Simple gait controller
    phase = (t * step_freq * 2 * np.pi) % (2 * np.pi)

    # Modify for shuffling
    step_height = 0.1 if not shuffling else 0.1 * (1 - shuffle_severity)
    step_length = target_speed / step_freq if not shuffling else (target_speed / step_freq) * (1 - shuffle_severity * 0.7)

    # Apply festination
    if festination and t > 1.0:
        effective_speed = target_speed * (1 + 0.1 * (t - 1.0))
    else:
        effective_speed = target_speed

    # Check for freezing
    if fog_prob > 0 and np.random.random() < fog_prob * dt:
        # Freeze for a moment
        for _ in range(int(0.5 / dt)):
            p.stepSimulation()

    # Simple joint targets (simplified gait pattern)
    for i, joint_idx in enumerate(joint_indices):
        if 'hip' in joint_names[i].lower():
            target = 0.3 * np.sin(phase)
        elif 'knee' in joint_names[i].lower():
            target = 0.5 * max(0, np.sin(phase))
        elif 'ankle' in joint_names[i].lower():
            target = 0.1 * np.sin(phase + np.pi/4)
        else:
            target = 0

        p.setJointMotorControl2(
            humanoid_id, joint_idx,
            p.POSITION_CONTROL,
            targetPosition=target,
            force=100
        )

    p.stepSimulation()

    # Record state
    if step % 10 == 0:  # Subsample for efficiency
        joint_states = p.getJointStates(humanoid_id, joint_indices)
        angles = [s[0] for s in joint_states]
        trajectory.append({{'time': t, 'angles': angles, 'names': joint_names}})

        # COM
        pos, _ = p.getBasePositionAndOrientation(humanoid_id)
        com_trajectory.append(list(pos))

        # Contact forces
        contacts = p.getContactPoints(humanoid_id, plane_id)
        for contact in contacts:
            contact_forces.append({{
                'time': t,
                'force': contact[9],
                'point': list(contact[5]),
                'body': 'foot'
            }})

sim_time = time.time() - start_time
print(f"Simulation completed in {{sim_time:.2f}}s")

# Calculate gait metrics
if len(com_trajectory) > 1:
    com_array = np.array(com_trajectory)
    actual_speed = np.mean(np.diff(com_array[:, 0])) / (10 * dt)  # Subsample factor
    stride_length = actual_speed / step_freq
else:
    actual_speed = 0
    stride_length = 0

result = {{
    'trajectory': trajectory,
    'com_trajectory': com_trajectory,
    'contact_forces': contact_forces[:1000],  # Limit size
    'gait_metrics': {{
        'speed': float(actual_speed),
        'stride_length': float(stride_length),
        'cadence': float(step_freq * 60),
        'shuffling': shuffling,
        'festination': festination
    }},
    'sim_time': sim_time
}}

print('RESULT_JSON:' + json.dumps(result))
p.disconnect()
"#,
            use_gui = self.config.use_gui,
            gx = self.config.gravity[0],
            gy = self.config.gravity[1],
            gz = self.config.gravity[2],
            time_step = self.config.time_step,
            solver_iters = self.config.solver_iterations,
            model_path = model_path,
            duration = params.duration,
            target_speed = params.target_speed,
            step_frequency = params.step_frequency,
            shuffling = params.shuffling,
            shuffle_severity = params.shuffle_severity,
            festination = params.festination,
            fog_probability = params.fog_probability,
        ))
    }

    fn generate_tremor_script(
        &self,
        _model: HumanoidModel,
        params: &PyBulletTremorParams,
    ) -> Result<String> {
        let joints_json = serde_json::to_string(&params.affected_joints)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(format!(
            r#"
import pybullet as p
import pybullet_data
import numpy as np
import json
import time

# Setup
physics_client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep({time_step})

# Load arm model (simplified)
plane_id = p.loadURDF("plane.urdf")

# Create a simple arm for tremor simulation
# Using a series of links
arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0.5])

# Tremor parameters
duration = {duration}
freq = {frequency}
amp = {amplitude}
is_resting = {is_resting}
freq_var = {freq_variation}
affected_joints = {joints_json}

# Simulation
dt = {time_step}
steps = int(duration / dt)
trajectory = []
end_effector_states = []

print("Simulating tremor...")
start_time = time.time()

num_joints = p.getNumJoints(arm_id)
for step in range(steps):
    t = step * dt

    # Tremor signal with frequency variation
    inst_freq = freq + freq_var * np.sin(0.5 * t)
    tremor = amp * np.sin(2 * np.pi * inst_freq * t)

    # Apply to joints
    for j in range(num_joints):
        info = p.getJointInfo(arm_id, j)
        joint_name = info[1].decode('utf-8')

        # Base position
        if is_resting:
            base_pos = 0.5  # Rest position
        else:
            base_pos = 0.5 + 0.3 * np.sin(0.5 * t)  # Action movement

        # Add tremor
        target = base_pos + tremor

        p.setJointMotorControl2(
            arm_id, j,
            p.POSITION_CONTROL,
            targetPosition=target,
            force=50
        )

    p.stepSimulation()

    # Record
    if step % 10 == 0:
        joint_states = p.getJointStates(arm_id, range(num_joints))
        angles = [s[0] for s in joint_states]
        trajectory.append({{'time': t, 'angles': angles}})

        # End effector
        ee_state = p.getLinkState(arm_id, num_joints - 1)
        end_effector_states.append({{
            'time': t,
            'position': list(ee_state[0]),
            'velocity': list(ee_state[6]) if len(ee_state) > 6 else [0, 0, 0]
        }})

sim_time = time.time() - start_time

result = {{
    'trajectory': trajectory,
    'end_effector_states': end_effector_states,
    'tremor_params': {{
        'frequency': freq,
        'amplitude': amp,
        'is_resting': is_resting
    }},
    'sim_time': sim_time
}}

print('RESULT_JSON:' + json.dumps(result))
p.disconnect()
"#,
            time_step = self.config.time_step,
            duration = params.duration,
            frequency = params.frequency,
            amplitude = params.amplitude,
            is_resting = params.is_resting,
            freq_variation = params.frequency_variation,
            joints_json = joints_json,
        ))
    }

    fn run_simulation(&self, script: &str) -> Result<PyBulletOutput> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("pybullet_sim.py");
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
        let result_line = stdout
            .lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result found".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        // Parse trajectory
        let mut trajectory = MotionTrajectory {
            times: Vec::new(),
            joint_angles: Vec::new(),
            joint_velocities: Vec::new(),
            end_effector_positions: Vec::new(),
            ground_forces: Vec::new(),
            muscle_activations: None,
        };

        if let Some(traj_array) = data["trajectory"].as_array() {
            for frame in traj_array {
                if let Some(t) = frame["time"].as_f64() {
                    trajectory.times.push(t);
                }
                if let Some(angles) = frame["angles"].as_array() {
                    let angle_vec: Vec<f64> = angles.iter().filter_map(|v| v.as_f64()).collect();
                    trajectory.joint_angles.push(angle_vec);
                    trajectory.joint_velocities.push(Vec::new());
                    trajectory.end_effector_positions.push(Vec::new());
                    trajectory.ground_forces.push(Vec::new());
                }
            }
        }

        // Parse gait ground truth if available
        let gait_ground_truth = data["gait_metrics"]
            .as_object()
            .map(|metrics| GaitGroundTruth {
                stride_length: metrics
                    .get("stride_length")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                cadence: metrics
                    .get("cadence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32,
                speed: metrics.get("speed").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                double_support_fraction: 0.3,
                step_width: 0.1,
                arm_swing_asymmetry: 0.0,
                festination: metrics
                    .get("festination")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                freezing_episodes: Vec::new(),
            });

        // Parse COM trajectory
        let com_trajectory = data["com_trajectory"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        let arr = v.as_array()?;
                        Some([
                            arr.first()?.as_f64()?,
                            arr.get(1)?.as_f64()?,
                            arr.get(2)?.as_f64()?,
                        ])
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Parse end effector states
        let end_effector_positions = data["end_effector_states"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        let pos = v["position"].as_array()?;
                        let vel_arr = v["velocity"].as_array();
                        let velocity = match vel_arr {
                            Some(vel) => [
                                vel.first().and_then(|x| x.as_f64()).unwrap_or(0.0),
                                vel.get(1).and_then(|x| x.as_f64()).unwrap_or(0.0),
                                vel.get(2).and_then(|x| x.as_f64()).unwrap_or(0.0),
                            ],
                            None => [0.0, 0.0, 0.0],
                        };
                        Some(EndEffectorState {
                            time: v["time"].as_f64().unwrap_or(0.0),
                            position: [
                                pos.first()?.as_f64()?,
                                pos.get(1)?.as_f64()?,
                                pos.get(2)?.as_f64()?,
                            ],
                            velocity,
                            name: "end_effector".to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Parse contact forces
        let contact_forces = data["contact_forces"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        let pt = v["point"].as_array()?;
                        Some(ContactForce {
                            time: v["time"].as_f64().unwrap_or(0.0),
                            force: v["force"].as_f64().unwrap_or(0.0),
                            contact_point: [
                                pt.first()?.as_f64()?,
                                pt.get(1)?.as_f64()?,
                                pt.get(2)?.as_f64()?,
                            ],
                            body_name: v["body"].as_str().unwrap_or("unknown").to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(PyBulletOutput {
            trajectory,
            gait_ground_truth,
            end_effector_positions,
            com_trajectory,
            contact_forces,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PyBulletConfig::default();
        assert!((config.time_step - 1.0 / 240.0).abs() < 1e-6);
        assert!(!config.use_gui);
    }

    #[test]
    fn test_gait_params_default() {
        let params = PyBulletGaitParams::default();
        assert_eq!(params.duration, 5.0);
        assert!(!params.shuffling);
    }

    #[test]
    fn test_tremor_params_default() {
        let params = PyBulletTremorParams::default();
        assert_eq!(params.frequency, 5.0);
        assert!(params.is_resting);
    }

    #[test]
    fn test_model_paths() {
        assert_eq!(
            HumanoidModel::SimpleHumanoid.urdf_path(),
            "humanoid/humanoid.urdf"
        );
        assert_eq!(HumanoidModel::GymHumanoid.urdf_path(), "humanoid.xml");
    }
}
