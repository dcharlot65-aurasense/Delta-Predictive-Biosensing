//! Gait cycle and keypoint trajectory generators
//! Based on Winter's biomechanics data

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth, GaitPhase};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Full gait cycle generator with kinematic model
pub struct GaitCycleGenerator;

#[derive(Debug, Clone)]
pub struct GaitCycleParams {
    pub duration: f64,         // seconds
    pub frame_rate: f64,       // fps
    pub cadence: f64,          // steps per minute
    pub stride_length: f64,    // meters
    pub step_width: f64,       // meters
    pub height: f64,           // meters (subject height)
}

impl SyntheticGenerator for GaitCycleGenerator {
    type Output = Vec<Vec<[f64; 3]>>; // frames x keypoints x [x,y,z]
    type GroundTruth = SpatialGroundTruth;
    type Parameters = GaitCycleParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let cycle_duration = 60.0 / params.cadence; // seconds per step
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // MediaPipe 33 keypoints (simplified - using key lower body points)
        let num_keypoints = 33;

        let mut keypoints = Vec::with_capacity(n_frames);
        let mut gait_phases = Vec::new();
        let mut joint_angles = HashMap::new();

        // Initialize joint angle storage
        joint_angles.insert("hip_flexion".to_string(), Vec::new());
        joint_angles.insert("knee_flexion".to_string(), Vec::new());
        joint_angles.insert("ankle_dorsiflexion".to_string(), Vec::new());

        for frame in 0..n_frames {
            let t = frame as f64 * dt;
            let gait_phase = (t % cycle_duration) / cycle_duration; // 0-1

            // Determine gait phase
            let phase_name = if gait_phase < 0.6 {
                "stance"
            } else {
                "swing"
            };

            gait_phases.push(GaitPhase {
                frame,
                phase: gait_phase,
                phase_name: phase_name.to_string(),
            });

            // Generate keypoint positions based on Winter's gait model
            let mut frame_keypoints = vec![[0.0; 3]; num_keypoints];

            // Pelvis center (keypoint 0) - reference point
            let pelvis_y = params.height * 0.55; // pelvis height
            let pelvis_z = t * params.stride_length / cycle_duration; // forward progression
            frame_keypoints[0] = [0.0, pelvis_y, pelvis_z];

            // Right hip (keypoint 24)
            frame_keypoints[24] = [params.step_width / 2.0, pelvis_y, pelvis_z];

            // Left hip (keypoint 23)
            frame_keypoints[23] = [-params.step_width / 2.0, pelvis_y, pelvis_z];

            // Right knee (keypoint 26) - using Winter's knee angle data
            let right_phase = gait_phase;
            let knee_angle = Self::knee_angle_profile(right_phase); // degrees
            let thigh_length = params.height * 0.245;
            let knee_x = params.step_width / 2.0 + thigh_length * knee_angle.to_radians().sin();
            let knee_y = pelvis_y - thigh_length * knee_angle.to_radians().cos();
            frame_keypoints[26] = [knee_x, knee_y, pelvis_z - thigh_length * (PI / 4.0).sin()];

            // Left knee (keypoint 25)
            let left_phase = (gait_phase + 0.5) % 1.0; // opposite phase
            let knee_angle_left = Self::knee_angle_profile(left_phase);
            let knee_x_left = -params.step_width / 2.0 - thigh_length * knee_angle_left.to_radians().sin();
            let knee_y_left = pelvis_y - thigh_length * knee_angle_left.to_radians().cos();
            frame_keypoints[25] = [knee_x_left, knee_y_left, pelvis_z - thigh_length * (PI / 4.0).sin()];

            // Right ankle (keypoint 28)
            let shank_length = params.height * 0.246;
            let ankle_angle = Self::ankle_angle_profile(right_phase);
            let ankle_y = frame_keypoints[26][1] - shank_length;
            frame_keypoints[28] = [knee_x, ankle_y, frame_keypoints[26][2]];

            // Left ankle (keypoint 27)
            let ankle_angle_left = Self::ankle_angle_profile(left_phase);
            let ankle_y_left = frame_keypoints[25][1] - shank_length;
            frame_keypoints[27] = [knee_x_left, ankle_y_left, frame_keypoints[25][2]];

            // Store joint angles
            joint_angles.get_mut("knee_flexion").unwrap().push(knee_angle);
            joint_angles.get_mut("hip_flexion").unwrap().push(Self::hip_angle_profile(right_phase));
            joint_angles.get_mut("ankle_dorsiflexion").unwrap().push(ankle_angle);

            // Fill in upper body keypoints (simplified - roughly stationary)
            for i in 1..23 {
                frame_keypoints[i] = [0.0, params.height * 0.8, pelvis_z];
            }

            keypoints.push(frame_keypoints);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: keypoints.clone(),
            joint_angles,
            gait_phases,
        };

        Ok(GeneratedData::new(keypoints, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        GaitCycleParams {
            duration: 10.0,
            frame_rate: 30.0,
            cadence: 110.0,      // steps per minute
            stride_length: 1.4,  // meters
            step_width: 0.15,    // meters
            height: 1.75,        // meters
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.frame_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("frame_rate must be positive".to_string()));
        }
        if params.cadence <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("cadence must be positive".to_string()));
        }
        Ok(())
    }
}

impl GaitCycleGenerator {
    /// Winter's knee angle profile (degrees) across gait cycle
    fn knee_angle_profile(phase: f64) -> f64 {
        // Loading response: flexion to 15-20 degrees
        // Mid-stance: extension
        // Swing: flexion to 60 degrees
        if phase < 0.12 {
            // Loading response
            15.0 * (phase / 0.12)
        } else if phase < 0.5 {
            // Stance phase extension
            15.0 * (1.0 - (phase - 0.12) / 0.38)
        } else if phase < 0.73 {
            // Swing phase flexion
            60.0 * ((phase - 0.5) / 0.23)
        } else {
            // Terminal swing extension
            60.0 * (1.0 - (phase - 0.73) / 0.27)
        }
    }

    /// Hip angle profile (degrees)
    fn hip_angle_profile(phase: f64) -> f64 {
        // Flexion positive, extension negative
        if phase < 0.5 {
            // Stance: extension
            -10.0 + 30.0 * phase
        } else {
            // Swing: flexion
            30.0 - 40.0 * (phase - 0.5)
        }
    }

    /// Ankle angle profile (degrees, dorsiflexion positive)
    fn ankle_angle_profile(phase: f64) -> f64 {
        if phase < 0.12 {
            // Plantarflexion at loading
            -5.0
        } else if phase < 0.5 {
            // Dorsiflexion in stance
            10.0 * ((phase - 0.12) / 0.38)
        } else {
            // Plantarflexion in swing
            10.0 - 15.0 * ((phase - 0.5) / 0.5)
        }
    }
}

/// Joint angle trajectory generator
pub struct JointAngleTrajectoryGenerator;

#[derive(Debug, Clone)]
pub struct JointAngleParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub joint_name: String,
    pub angle_min: f64,  // degrees
    pub angle_max: f64,  // degrees
    pub frequency: f64,  // Hz
}

impl SyntheticGenerator for JointAngleTrajectoryGenerator {
    type Output = Vec<f64>; // angle time series
    type GroundTruth = SpatialGroundTruth;
    type Parameters = JointAngleParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise = Normal::new(0.0, 1.0).unwrap();

        let angles: Vec<f64> = (0..n_frames)
            .map(|i| {
                let t = i as f64 * dt;
                let mid_angle = (params.angle_max + params.angle_min) / 2.0;
                let amplitude = (params.angle_max - params.angle_min) / 2.0;
                mid_angle + amplitude * (2.0 * PI * params.frequency * t).sin() +
                    noise.sample(&mut rng)
            })
            .collect();

        let mut joint_angles = HashMap::new();
        joint_angles.insert(params.joint_name.clone(), angles.clone());

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(angles, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        JointAngleParams {
            duration: 5.0,
            frame_rate: 30.0,
            joint_name: "knee_flexion".to_string(),
            angle_min: 0.0,
            angle_max: 60.0,
            frequency: 1.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        Ok(())
    }
}

/// Pathological gait generator
pub struct PathologicalGaitGenerator;

#[derive(Debug, Clone)]
pub enum GaitPathology {
    Parkinsonian,  // Shuffling, reduced arm swing, freezing
    Ataxic,        // Wide-based, irregular
    Hemiplegic,    // Asymmetric, circumduction
    Antalgic,      // Pain-avoiding, reduced stance on affected side
}

#[derive(Debug, Clone)]
pub struct PathologicalGaitParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub pathology: GaitPathology,
    pub severity: f64,     // 0-1
    pub baseline_cadence: f64,
    pub height: f64,
}

impl SyntheticGenerator for PathologicalGaitGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = PathologicalGaitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Start with normal gait
        let normal_params = GaitCycleParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cadence: params.baseline_cadence,
            stride_length: 1.4,
            step_width: 0.15,
            height: params.height,
        };

        let normal_gen = GaitCycleGenerator;
        let mut result = normal_gen.generate(&normal_params, seed)?;

        // Apply pathology-specific modifications
        match params.pathology {
            GaitPathology::Parkinsonian => {
                // Reduce stride length and increase cadence
                for frame_kps in &mut result.signal {
                    for kp in frame_kps.iter_mut() {
                        // Reduce forward progression
                        kp[2] *= 1.0 - params.severity * 0.5;
                        // Reduce vertical displacement
                        kp[1] = params.height * 0.55 + (kp[1] - params.height * 0.55) * (1.0 - params.severity * 0.7);
                    }
                }
            }
            GaitPathology::Ataxic => {
                // Add irregular, wide-based gait
                let noise = Normal::new(0.0, 0.05 * params.severity).unwrap();
                for frame_kps in &mut result.signal {
                    for kp in frame_kps.iter_mut() {
                        // Increase lateral sway
                        kp[0] *= 1.0 + params.severity * 2.0;
                        // Add random perturbations
                        kp[0] += noise.sample(&mut rng);
                        kp[1] += noise.sample(&mut rng);
                        kp[2] += noise.sample(&mut rng);
                    }
                }
            }
            GaitPathology::Hemiplegic => {
                // Asymmetric gait - affect right side
                for frame_kps in &mut result.signal {
                    // Right leg keypoints (24, 26, 28)
                    for &idx in &[24, 26, 28] {
                        // Circumduction pattern
                        frame_kps[idx][0] += params.severity * 0.1;
                        frame_kps[idx][1] *= 1.0 - params.severity * 0.1;
                    }
                }
            }
            GaitPathology::Antalgic => {
                // Reduced stance time on affected side
                // This would require phase-specific modifications
                for frame_kps in &mut result.signal {
                    // Reduce weight-bearing on right side
                    for &idx in &[24, 26, 28] {
                        frame_kps[idx][1] *= 1.0 + params.severity * 0.05;
                    }
                }
            }
        }

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        PathologicalGaitParams {
            duration: 10.0,
            frame_rate: 30.0,
            pathology: GaitPathology::Parkinsonian,
            severity: 0.5,
            baseline_cadence: 110.0,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.severity < 0.0 || params.severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("severity must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// MediaPipe keypoint trajectory generator
pub struct KeypointTrajectoryGenerator;

#[derive(Debug, Clone)]
pub struct KeypointTrajectoryParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub gait_params: GaitCycleParams,
}

impl SyntheticGenerator for KeypointTrajectoryGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = KeypointTrajectoryParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        // Delegate to GaitCycleGenerator
        let gait_gen = GaitCycleGenerator;
        gait_gen.generate(&params.gait_params, seed)
    }

    fn default_params() -> Self::Parameters {
        KeypointTrajectoryParams {
            duration: 10.0,
            frame_rate: 30.0,
            gait_params: GaitCycleGenerator::default_params(),
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        GaitCycleGenerator::validate_params(&params.gait_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gait_cycle_generation() {
        let generator = GaitCycleGenerator;
        let params = GaitCycleGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
        assert!(!result.ground_truth.gait_phases.is_empty());
    }

    #[test]
    fn test_pathological_gait() {
        let generator = PathologicalGaitGenerator;
        let params = PathologicalGaitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }
}
