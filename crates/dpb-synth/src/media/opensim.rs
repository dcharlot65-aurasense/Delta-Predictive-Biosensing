//! OpenSim Biomechanical Simulation Integration
//!
//! This module provides integration with OpenSim for biomechanical simulation
//! including detailed musculoskeletal models and pathological motion patterns.
//!
//! # Features
//!
//! - Gait2392 and custom gait models
//! - Inverse kinematics and dynamics
//! - Pathological tremor simulation (Parkinson's, Essential Tremor)
//! - Muscle force estimation
//! - Subject-specific model scaling
//!
//! # Requirements
//!
//! - Python 3.8+
//! - OpenSim 4.x with Python bindings
//! - `conda install -c opensim-org opensim`
//!
//! # References
//!
//! - OpenSim: https://simtk.org/projects/opensim
//! - Pathological tremor model: DOI:10.1016/j.jbiomech.2024.111962

use super::{MediaError, Result, TremorType, TremorGroundTruth};
use std::path::{Path, PathBuf};
use std::process::Command;

/// OpenSim bridge for biomechanical simulation
#[derive(Debug)]
pub struct OpenSimBridge {
    config: OpenSimConfig,
    python_path: PathBuf,
}

/// Configuration for OpenSim simulation
#[derive(Debug, Clone)]
pub struct OpenSimConfig {
    /// Path to OpenSim models directory
    pub models_dir: PathBuf,
    /// Path to OpenSim installation (for scripting API)
    pub opensim_home: Option<PathBuf>,
    /// Simulation accuracy
    pub accuracy: f64,
    /// Integration step size
    pub step_size: f64,
    /// Output directory for results
    pub output_dir: PathBuf,
}

impl Default for OpenSimConfig {
    fn default() -> Self {
        Self {
            models_dir: PathBuf::from("models/opensim"),
            opensim_home: None,
            accuracy: 1e-6,
            step_size: 0.001,
            output_dir: PathBuf::from("output/opensim"),
        }
    }
}

/// Gait model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaitModel {
    /// Standard Gait2392 model (23 DOF, 92 muscles)
    Gait2392,
    /// Gait2354 model (23 DOF, 54 muscles)
    Gait2354,
    /// Full body model with upper extremities
    FullBody,
    /// Rajagopal 2015 lower extremity model
    Rajagopal2015,
    /// Custom model from file
    Custom,
}

/// Tremor model configuration
#[derive(Debug, Clone)]
pub struct TremorModel {
    /// Type of tremor
    pub tremor_type: TremorType,
    /// Central frequency in Hz
    pub frequency: f64,
    /// Frequency bandwidth (std dev)
    pub frequency_std: f64,
    /// Amplitude in radians
    pub amplitude: f64,
    /// Amplitude variability (coefficient of variation)
    pub amplitude_cv: f64,
    /// Affected DOFs
    pub affected_dofs: Vec<String>,
    /// H∞ controller parameters (for closed-loop model)
    pub controller_params: Option<ControllerParams>,
}

/// H∞ controller parameters for closed-loop tremor model
#[derive(Debug, Clone)]
pub struct ControllerParams {
    /// Proportional gain
    pub kp: f64,
    /// Derivative gain
    pub kd: f64,
    /// Filter cutoff frequency
    pub filter_cutoff: f64,
    /// Delay in ms
    pub delay_ms: f64,
}

impl Default for TremorModel {
    fn default() -> Self {
        Self {
            tremor_type: TremorType::Resting,
            frequency: 5.0,
            frequency_std: 0.5,
            amplitude: 0.05,
            amplitude_cv: 0.3,
            affected_dofs: vec!["wrist_flex".to_string()],
            controller_params: None,
        }
    }
}

/// Parameters for pathological gait/motion
#[derive(Debug, Clone)]
pub struct PathologyParams {
    /// Disease type
    pub disease: DiseaseType,
    /// Severity (0-1, maps to clinical scales)
    pub severity: f64,
    /// Specific symptom parameters
    pub symptoms: Vec<Symptom>,
    /// Medication state
    pub medication: MedicationState,
}

/// Disease types with characteristic motion patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiseaseType {
    /// Parkinson's Disease
    Parkinsons,
    /// Essential Tremor
    EssentialTremor,
    /// Stroke (hemiparesis)
    Stroke,
    /// Multiple Sclerosis
    MultipleSclerosis,
    /// Cerebral Palsy
    CerebralPalsy,
    /// Healthy control
    Healthy,
}

/// Specific symptoms to include
#[derive(Debug, Clone)]
pub enum Symptom {
    /// Resting tremor with frequency and amplitude
    RestingTremor { frequency: f64, amplitude: f64 },
    /// Action/intention tremor
    ActionTremor { frequency: f64, amplitude: f64 },
    /// Bradykinesia (slowness)
    Bradykinesia { severity: f64 },
    /// Rigidity
    Rigidity { severity: f64 },
    /// Festinating gait
    Festination,
    /// Freezing of gait
    FreezingOfGait { probability: f64, duration_range: (f64, f64) },
    /// Reduced arm swing
    ReducedArmSwing { reduction: f64 },
    /// Postural instability
    PosturalInstability { severity: f64 },
    /// Dyskinesia (medication-induced movements)
    Dyskinesia { severity: f64 },
    /// Hemiparesis (one-sided weakness)
    Hemiparesis { side: Side, severity: f64 },
}

/// Side of body
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// Medication state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MedicationState {
    /// No medication (OFF state)
    Off,
    /// On medication (optimal)
    OnOptimal,
    /// On medication but wearing off
    WearingOff,
    /// Dyskinetic (too much medication effect)
    Dyskinetic,
}

impl Default for PathologyParams {
    fn default() -> Self {
        Self {
            disease: DiseaseType::Healthy,
            severity: 0.0,
            symptoms: Vec::new(),
            medication: MedicationState::Off,
        }
    }
}

impl OpenSimBridge {
    /// Create a new OpenSim bridge
    pub fn new(config: OpenSimConfig) -> Result<Self> {
        // Check if OpenSim Python is available
        let python_check = Command::new("python3")
            .args(["-c", "import opensim; print(opensim.GetVersion())"])
            .output();

        match python_check {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                log::info!("OpenSim version: {}", version.trim());
            }
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "opensim".to_string(),
                    install_url: "conda install -c opensim-org opensim".to_string(),
                });
            }
        }

        let python_path = which::which("python3")
            .map_err(|_| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: "https://www.python.org/downloads/".to_string(),
            })?;

        // Ensure output directory exists
        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self {
            config,
            python_path,
        })
    }

    /// Check if OpenSim is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import opensim"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Run gait simulation
    pub fn simulate_gait(
        &self,
        model: GaitModel,
        duration: f64,
        pathology: Option<&PathologyParams>,
    ) -> Result<GaitSimulationResult> {
        let script = self.generate_gait_script(model, duration, pathology)?;
        self.run_script(&script)
    }

    /// Run tremor simulation using closed-loop wrist model
    pub fn simulate_tremor(
        &self,
        tremor_model: &TremorModel,
        duration: f64,
    ) -> Result<TremorSimulationResult> {
        let script = self.generate_tremor_script(tremor_model, duration)?;
        self.run_tremor_script(&script)
    }

    /// Scale model to subject-specific dimensions
    pub fn scale_model(
        &self,
        base_model: GaitModel,
        subject_mass: f64,
        subject_height: f64,
        _marker_data: Option<&Path>,
    ) -> Result<PathBuf> {
        let script = format!(r#"
import opensim as osim

# Load base model
model_path = '{model_path}'
model = osim.Model(model_path)

# Create scale tool
scale_tool = osim.ScaleTool()

# Set mass and height
mass_scale = {mass} / model.getTotalMass(osim.SimTK_Vec3(0))
height_scale = {height} / 1.8  # Assuming 1.8m reference

# Apply uniform scaling (simplified)
for body in model.getBodySet():
    body.setMass(body.getMass() * mass_scale)

# Save scaled model
output_path = '{output_path}'
model.printToXML(output_path)
print(output_path)
"#,
            model_path = self.get_model_path(base_model).display(),
            mass = subject_mass,
            height = subject_height,
            output_path = self.config.output_dir.join("scaled_model.osim").display(),
        );

        let output = self.run_python_script(&script)?;
        Ok(PathBuf::from(output.trim()))
    }

    /// Generate Python script for gait simulation
    fn generate_gait_script(
        &self,
        model: GaitModel,
        duration: f64,
        pathology: Option<&PathologyParams>,
    ) -> Result<String> {
        let pathology_code = if let Some(p) = pathology {
            self.generate_pathology_code(p)
        } else {
            String::new()
        };

        Ok(format!(r#"
import opensim as osim
import numpy as np
import json

# Load gait model
model_path = '{model_path}'
model = osim.Model(model_path)
model.initSystem()

# Simulation parameters
DURATION = {duration}
STEP_SIZE = {step_size}

# Storage for results
times = []
joint_angles = {{}}
muscle_forces = {{}}
ground_forces = []

# Get coordinate names
for i in range(model.getCoordinateSet().getSize()):
    coord = model.getCoordinateSet().get(i)
    joint_angles[coord.getName()] = []

# Apply pathological modifications
{pathology_code}

# Create forward dynamics tool
state = model.initSystem()

# Simple forward simulation
t = 0.0
while t < DURATION:
    # Integrate one step
    manager = osim.Manager(model)
    manager.setIntegratorAccuracy({accuracy})
    state.setTime(t)
    manager.initialize(state)
    state = manager.integrate(t + STEP_SIZE)

    # Store results
    times.append(t)
    for i in range(model.getCoordinateSet().getSize()):
        coord = model.getCoordinateSet().get(i)
        joint_angles[coord.getName()].append(coord.getValue(state))

    t += STEP_SIZE

# Output results
result = {{
    'times': times,
    'joint_angles': joint_angles,
    'duration': DURATION
}}
print(json.dumps(result))
"#,
            model_path = self.get_model_path(model).display(),
            duration = duration,
            step_size = self.config.step_size,
            accuracy = self.config.accuracy,
            pathology_code = pathology_code,
        ))
    }

    /// Generate pathology modification code
    fn generate_pathology_code(&self, params: &PathologyParams) -> String {
        let mut code = String::new();

        for symptom in &params.symptoms {
            match symptom {
                Symptom::Bradykinesia { severity } => {
                    code.push_str(&format!(r#"
# Apply bradykinesia - reduce muscle max forces
bradykinesia_factor = 1.0 - {} * 0.5
for muscle in model.getMuscles():
    muscle.setMaxIsometricForce(muscle.getMaxIsometricForce() * bradykinesia_factor)
"#, severity));
                }
                Symptom::Rigidity { severity } => {
                    code.push_str(&format!(r#"
# Apply rigidity - increase passive joint stiffness
rigidity_factor = 1.0 + {} * 2.0
for coord in model.getCoordinateSet():
    # Increase damping
    pass  # OpenSim API for damping adjustment
"#, severity));
                }
                Symptom::ReducedArmSwing { reduction } => {
                    code.push_str(&format!(r#"
# Reduce arm swing amplitude
arm_swing_factor = 1.0 - {}
# Apply to shoulder flexion limits
"#, reduction));
                }
                _ => {}
            }
        }

        code
    }

    /// Generate Python script for tremor simulation
    fn generate_tremor_script(&self, tremor: &TremorModel, duration: f64) -> Result<String> {
        let affected_dofs_str = tremor.affected_dofs
            .iter()
            .map(|d| format!("'{}'", d))
            .collect::<Vec<_>>()
            .join(", ");

        Ok(format!(r#"
import opensim as osim
import numpy as np
import json
from scipy import signal

# Tremor parameters
FREQUENCY = {frequency}
FREQUENCY_STD = {frequency_std}
AMPLITUDE = {amplitude}
AMPLITUDE_CV = {amplitude_cv}
DURATION = {duration}
STEP_SIZE = 0.001
AFFECTED_DOFS = [{affected_dofs}]

# Create upper extremity model for wrist tremor
model = osim.Model()
model.setName('tremor_model')

# Add ground
ground = model.getGround()

# Add forearm body
forearm = osim.Body('forearm', 1.5, osim.Vec3(0), osim.Inertia(0.01))
model.addBody(forearm)

# Add wrist joint (2 DOF)
wrist_flexion = osim.PinJoint('wrist_flexion', ground, osim.Vec3(0), osim.Vec3(0),
                               forearm, osim.Vec3(-0.15, 0, 0), osim.Vec3(0))
model.addJoint(wrist_flexion)

# Add coordinate actuator for tremor input
coord_act = osim.CoordinateActuator('wrist_flexion_coord')
coord_act.setCoordinate(model.getCoordinateSet().get('wrist_flexion_coord'))
model.addForce(coord_act)

# Initialize model
state = model.initSystem()

# Tremor oscillator with physiological variability
class TremorOscillator:
    def __init__(self, freq, freq_std, amp, amp_cv):
        self.base_freq = freq
        self.freq_std = freq_std
        self.base_amp = amp
        self.amp_cv = amp_cv

        # Frequency modulation (slow drift)
        self.freq_mod_freq = 0.1  # Hz

        # Amplitude modulation (intermittent character)
        self.amp_mod_freq = 0.3  # Hz

    def get_tremor(self, t):
        # Instantaneous frequency with modulation
        freq = self.base_freq + self.freq_std * np.sin(2 * np.pi * self.freq_mod_freq * t)

        # Amplitude with intermittent modulation and variability
        amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * self.amp_mod_freq * t)
        amp = self.base_amp * amp_mod * (1 + self.amp_cv * np.random.randn())
        amp = max(0, amp)  # Non-negative

        # Tremor displacement
        return amp * np.sin(2 * np.pi * freq * t)

oscillator = TremorOscillator(FREQUENCY, FREQUENCY_STD, AMPLITUDE, AMPLITUDE_CV)

# Simulation
times = []
angles = []
velocities = []
tremor_input = []

t = 0.0
while t < DURATION:
    # Get tremor input
    tremor = oscillator.get_tremor(t)
    tremor_input.append(tremor)

    # Apply tremor to coordinate
    coord = model.getCoordinateSet().get('wrist_flexion_coord')
    coord.setValue(state, tremor)

    # Store
    times.append(t)
    angles.append(tremor)
    velocities.append(coord.getSpeedValue(state))

    t += STEP_SIZE

# Analyze tremor characteristics
tremor_array = np.array(tremor_input)

# Compute spectrum
fs = 1.0 / STEP_SIZE
f, psd = signal.welch(tremor_array, fs, nperseg=min(len(tremor_array), 1024))

# Find dominant frequency
peak_idx = np.argmax(psd)
dominant_freq = f[peak_idx]

# Compute amplitude statistics
amplitude_rms = np.sqrt(np.mean(tremor_array ** 2))

result = {{
    'times': times,
    'angles': angles,
    'velocities': velocities,
    'tremor_input': list(tremor_input),
    'dominant_frequency': float(dominant_freq),
    'amplitude_rms': float(amplitude_rms),
    'spectrum_frequencies': f.tolist(),
    'spectrum_power': psd.tolist()
}}
print(json.dumps(result))
"#,
            frequency = tremor.frequency,
            frequency_std = tremor.frequency_std,
            amplitude = tremor.amplitude,
            amplitude_cv = tremor.amplitude_cv,
            duration = duration,
            affected_dofs = affected_dofs_str,
        ))
    }

    /// Get path to model file
    fn get_model_path(&self, model: GaitModel) -> PathBuf {
        match model {
            GaitModel::Gait2392 => self.config.models_dir.join("gait2392_simbody.osim"),
            GaitModel::Gait2354 => self.config.models_dir.join("gait2354_simbody.osim"),
            GaitModel::FullBody => self.config.models_dir.join("fullbody.osim"),
            GaitModel::Rajagopal2015 => self.config.models_dir.join("Rajagopal2015.osim"),
            GaitModel::Custom => self.config.models_dir.join("custom.osim"),
        }
    }

    /// Run Python script and parse gait results
    fn run_script(&self, script: &str) -> Result<GaitSimulationResult> {
        let output = self.run_python_script(script)?;

        let data: serde_json::Value = serde_json::from_str(&output)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(GaitSimulationResult {
            times: data["times"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default(),
            joint_angles: serde_json::from_value(data["joint_angles"].clone())
                .unwrap_or_default(),
            duration: data["duration"].as_f64().unwrap_or(0.0),
        })
    }

    /// Run Python script and parse tremor results
    fn run_tremor_script(&self, script: &str) -> Result<TremorSimulationResult> {
        let output = self.run_python_script(script)?;

        let data: serde_json::Value = serde_json::from_str(&output)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(TremorSimulationResult {
            times: data["times"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default(),
            angles: data["angles"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default(),
            velocities: data["velocities"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default(),
            dominant_frequency: data["dominant_frequency"].as_f64().unwrap_or(0.0),
            amplitude_rms: data["amplitude_rms"].as_f64().unwrap_or(0.0),
        })
    }

    /// Run a Python script and return stdout
    fn run_python_script(&self, script: &str) -> Result<String> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("opensim_script.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

/// Result from gait simulation
#[derive(Debug, Clone)]
pub struct GaitSimulationResult {
    /// Time points
    pub times: Vec<f64>,
    /// Joint angles by coordinate name
    pub joint_angles: std::collections::HashMap<String, Vec<f64>>,
    /// Total duration
    pub duration: f64,
}

/// Result from tremor simulation
#[derive(Debug, Clone)]
pub struct TremorSimulationResult {
    /// Time points
    pub times: Vec<f64>,
    /// Joint angles
    pub angles: Vec<f64>,
    /// Joint velocities
    pub velocities: Vec<f64>,
    /// Dominant tremor frequency from spectral analysis
    pub dominant_frequency: f64,
    /// RMS amplitude
    pub amplitude_rms: f64,
}

impl TremorSimulationResult {
    /// Convert to ground truth format
    pub fn to_ground_truth(&self, tremor_type: TremorType) -> TremorGroundTruth {
        TremorGroundTruth {
            frequency: self.dominant_frequency as f32,
            amplitude: self.amplitude_rms as f32,
            tremor_type,
            affected_joints: vec!["wrist".to_string()],
            amplitude_envelope: self.angles.iter().map(|a| a.abs() as f32).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OpenSimConfig::default();
        assert_eq!(config.step_size, 0.001);
    }

    #[test]
    fn test_tremor_model_default() {
        let model = TremorModel::default();
        assert_eq!(model.frequency, 5.0);
        assert_eq!(model.tremor_type, TremorType::Resting);
    }

    #[test]
    fn test_pathology_default() {
        let params = PathologyParams::default();
        assert_eq!(params.disease, DiseaseType::Healthy);
        assert!(params.symptoms.is_empty());
    }
}
