//! Hand movement generators for functional tasks

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Pronation-supination generator (forearm rotation)
pub struct PronationSupinationGenerator;

#[derive(Debug, Clone)]
pub struct PronationSupinationParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub rom: f64,              // Range of motion (degrees)
    pub speed: f64,            // Cycles per second
    pub regularity: f64,       // 0-1 (1 = perfectly regular)
    pub asymmetry: f64,        // 0-1 (0 = symmetric, 1 = highly asymmetric)
}

impl SyntheticGenerator for PronationSupinationGenerator {
    type Output = Vec<f64>; // rotation angle over time (degrees)
    type GroundTruth = SpatialGroundTruth;
    type Parameters = PronationSupinationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let timing_noise = Normal::new(0.0, (1.0 - params.regularity) * 0.05).unwrap();

        let mut angles = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let t = i as f64 * dt;
            let phase = (2.0 * PI * params.speed * t + timing_noise.sample(&mut rng)).rem_euclid(2.0 * PI);

            // Apply asymmetry (different speeds for pronation vs supination)
            let adjusted_phase = if phase < PI {
                // Pronation
                phase * (1.0 + params.asymmetry)
            } else {
                // Supination
                PI + (phase - PI) * (1.0 - params.asymmetry)
            };

            let angle = params.rom * (adjusted_phase / (2.0 * PI) - 0.5);
            angles.push(angle);
        }

        let mut joint_angles = HashMap::new();
        joint_angles.insert("forearm_rotation".to_string(), angles.clone());

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(angles, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        PronationSupinationParams {
            duration: 10.0,
            frame_rate: 60.0,
            rom: 180.0,
            speed: 1.0,
            regularity: 0.85,
            asymmetry: 0.2,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.regularity < 0.0 || params.regularity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("regularity must be 0-1".to_string()));
        }
        if params.asymmetry < 0.0 || params.asymmetry > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("asymmetry must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Hand open-close generator
pub struct HandOpenCloseGenerator;

#[derive(Debug, Clone)]
pub struct HandOpenCloseParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub aperture_range: (f64, f64), // Min and max aperture (cm)
    pub speed: f64,                  // Cycles per second
    pub closure_completeness: f64,   // 0-1 (1 = full closure)
}

impl SyntheticGenerator for HandOpenCloseGenerator {
    type Output = Vec<f64>; // aperture over time (cm)
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HandOpenCloseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let _rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut apertures = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let t = i as f64 * dt;
            let phase = (2.0 * PI * params.speed * t).rem_euclid(2.0 * PI);

            // Sinusoidal open-close pattern
            let normalized = (phase.sin() + 1.0) / 2.0;

            // Apply closure completeness
            let effective_min = params.aperture_range.0 +
                (params.aperture_range.1 - params.aperture_range.0) * (1.0 - params.closure_completeness);

            let aperture = effective_min +
                (params.aperture_range.1 - effective_min) * normalized;

            apertures.push(aperture);
        }

        let mut joint_angles = HashMap::new();
        joint_angles.insert("hand_aperture".to_string(), apertures.clone());

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(apertures, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        HandOpenCloseParams {
            duration: 10.0,
            frame_rate: 60.0,
            aperture_range: (0.0, 20.0),
            speed: 1.0,
            closure_completeness: 0.95,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.aperture_range.0 >= params.aperture_range.1 {
            return Err(crate::GeneratorError::InvalidParameter("aperture_range.0 must be < aperture_range.1".to_string()));
        }
        if params.closure_completeness < 0.0 || params.closure_completeness > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("closure_completeness must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Precision grip generator
pub struct PrecisionGripGenerator;

#[derive(Debug, Clone)]
pub struct PrecisionGripParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub target_force: f64,          // Target force (N)
    pub force_control_noise: f64,   // Force variability (std dev as fraction of target)
    pub tremor_amplitude: f64,      // Tremor amplitude (N)
    pub tremor_frequency: f64,      // Tremor frequency (Hz)
}

impl SyntheticGenerator for PrecisionGripGenerator {
    type Output = Vec<f64>; // grip force over time (N)
    type GroundTruth = SpatialGroundTruth;
    type Parameters = PrecisionGripParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let force_noise = Normal::new(0.0, params.target_force * params.force_control_noise).unwrap();

        let mut forces = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let t = i as f64 * dt;

            // Build up to target force over first 0.5 seconds
            let ramp = if t < 0.5 {
                t / 0.5
            } else {
                1.0
            };

            let base_force = params.target_force * ramp;
            let noise = force_noise.sample(&mut rng);
            let tremor = params.tremor_amplitude * (2.0 * PI * params.tremor_frequency * t).sin();

            let force = (base_force + noise + tremor).max(0.0);
            forces.push(force);
        }

        let mut joint_angles = HashMap::new();
        joint_angles.insert("grip_force".to_string(), forces.clone());

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(forces, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        PrecisionGripParams {
            duration: 5.0,
            frame_rate: 60.0,
            target_force: 10.0,
            force_control_noise: 0.1,
            tremor_amplitude: 0.5,
            tremor_frequency: 8.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.target_force < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("target_force must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Reaching movement generator
pub struct ReachingMovementGenerator;

#[derive(Debug, Clone)]
pub struct ReachingMovementParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub target_position: [f64; 3],  // Target position (cm)
    pub movement_time: f64,          // Time to reach target (s)
    pub speed_profile: SpeedProfile,
    pub accuracy: f64,               // 0-1 (1 = perfect accuracy)
    pub smoothness: f64,             // 0-1 (1 = perfectly smooth)
}

#[derive(Debug, Clone)]
pub enum SpeedProfile {
    Smooth,      // Bell-shaped velocity profile
    Jerky,       // Multiple sub-movements
    Bradykinetic, // Slow, hesitant
}

impl SyntheticGenerator for ReachingMovementGenerator {
    type Output = Vec<[f64; 3]>; // hand position over time [x, y, z]
    type GroundTruth = SpatialGroundTruth;
    type Parameters = ReachingMovementParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let position_noise = Normal::new(0.0, (1.0 - params.accuracy) * 0.5).unwrap();

        let mut positions = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let t = i as f64 * dt;

            let progress = if t >= params.movement_time {
                1.0
            } else {
                match params.speed_profile {
                    SpeedProfile::Smooth => {
                        // Minimum jerk trajectory
                        let tau = t / params.movement_time;
                        10.0 * tau.powi(3) - 15.0 * tau.powi(4) + 6.0 * tau.powi(5)
                    }
                    SpeedProfile::Jerky => {
                        // Multiple sub-movements
                        let tau = t / params.movement_time;
                        let primary = tau;
                        let secondary = 0.2 * (4.0 * PI * tau).sin();
                        (primary + secondary).clamp(0.0, 1.0)
                    }
                    SpeedProfile::Bradykinetic => {
                        // Slow sigmoid
                        let x = 6.0 * (t / params.movement_time - 0.5);
                        1.0 / (1.0 + (-x).exp())
                    }
                }
            };

            // Add smoothness noise
            let smoothness_noise = (1.0 - params.smoothness) * position_noise.sample(&mut rng);

            let position = [
                params.target_position[0] * progress + smoothness_noise,
                params.target_position[1] * progress + smoothness_noise * 0.8,
                params.target_position[2] * progress + smoothness_noise * 0.6,
            ];

            positions.push(position);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: vec![positions.clone()],
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(positions, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        ReachingMovementParams {
            duration: 3.0,
            frame_rate: 60.0,
            target_position: [30.0, 20.0, 15.0],
            movement_time: 2.0,
            speed_profile: SpeedProfile::Smooth,
            accuracy: 0.9,
            smoothness: 0.9,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.accuracy < 0.0 || params.accuracy > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("accuracy must be 0-1".to_string()));
        }
        if params.smoothness < 0.0 || params.smoothness > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("smoothness must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Drawing generator (spiral or line drawing)
pub struct DrawingGenerator;

#[derive(Debug, Clone)]
pub struct DrawingParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub pattern: DrawingPattern,
    pub tremor_amplitude: f64,   // cm
    pub tremor_frequency: f64,   // Hz
    pub accuracy: f64,            // 0-1 (1 = perfect accuracy)
}

#[derive(Debug, Clone)]
pub enum DrawingPattern {
    Spiral {
        revolutions: f64,
        max_radius: f64,
    },
    Line {
        length: f64,
        angle: f64, // degrees
    },
    Circle {
        radius: f64,
    },
}

impl SyntheticGenerator for DrawingGenerator {
    type Output = Vec<[f64; 2]>; // 2D drawing path [x, y]
    type GroundTruth = SpatialGroundTruth;
    type Parameters = DrawingParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let position_noise = Normal::new(0.0, (1.0 - params.accuracy) * 0.2).unwrap();

        let mut path = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let t = i as f64 * dt;
            let progress = t / params.duration;

            let (ideal_x, ideal_y) = match &params.pattern {
                DrawingPattern::Spiral { revolutions, max_radius } => {
                    let angle = 2.0 * PI * revolutions * progress;
                    let radius = max_radius * progress;
                    (
                        radius * angle.cos(),
                        radius * angle.sin(),
                    )
                }
                DrawingPattern::Line { length, angle } => {
                    let angle_rad = angle * PI / 180.0;
                    (
                        length * progress * angle_rad.cos(),
                        length * progress * angle_rad.sin(),
                    )
                }
                DrawingPattern::Circle { radius } => {
                    let angle = 2.0 * PI * progress;
                    (
                        radius * angle.cos(),
                        radius * angle.sin(),
                    )
                }
            };

            // Add tremor
            let tremor = params.tremor_amplitude * (2.0 * PI * params.tremor_frequency * t).sin();

            // Add accuracy noise
            let noise_x = position_noise.sample(&mut rng);
            let noise_y = position_noise.sample(&mut rng);

            let x = ideal_x + tremor + noise_x;
            let y = ideal_y + tremor * 0.8 + noise_y;

            path.push([x, y]);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(path, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        DrawingParams {
            duration: 10.0,
            frame_rate: 60.0,
            pattern: DrawingPattern::Spiral {
                revolutions: 3.0,
                max_radius: 10.0,
            },
            tremor_amplitude: 0.2,
            tremor_frequency: 6.0,
            accuracy: 0.85,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.accuracy < 0.0 || params.accuracy > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("accuracy must be 0-1".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pronation_supination() {
        let generator = PronationSupinationGenerator;
        let params = PronationSupinationGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_hand_open_close() {
        let generator = HandOpenCloseGenerator;
        let params = HandOpenCloseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);

        // Check that aperture stays within range
        for &aperture in &result.signal {
            assert!(aperture >= 0.0 && aperture <= params.aperture_range.1);
        }
    }

    #[test]
    fn test_precision_grip() {
        let generator = PrecisionGripGenerator;
        let params = PrecisionGripGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);

        // Check forces are non-negative
        for &force in &result.signal {
            assert!(force >= 0.0);
        }
    }

    #[test]
    fn test_reaching_movement() {
        let generator = ReachingMovementGenerator;
        let params = ReachingMovementGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_drawing_spiral() {
        let generator = DrawingGenerator;
        let params = DrawingGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_drawing_circle() {
        let generator = DrawingGenerator;
        let mut params = DrawingGenerator::default_params();
        params.pattern = DrawingPattern::Circle { radius: 5.0 };
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }
}
