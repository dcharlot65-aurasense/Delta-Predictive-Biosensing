//! Hand-specific tremor generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Hand postural tremor generator
pub struct HandPosturalTremorGenerator;

#[derive(Debug, Clone)]
pub struct HandPosturalTremorParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub frequency: f64,        // Hz (4-12)
    pub amplitude_x: f64,      // cm
    pub amplitude_y: f64,      // cm
    pub amplitude_z: f64,      // cm
}

impl SyntheticGenerator for HandPosturalTremorGenerator {
    type Output = Vec<[f64; 3]>; // hand position over time [x, y, z]
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HandPosturalTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let phase_x = rng.r#gen_range(0.0..2.0 * PI);
        let phase_y = rng.r#gen_range(0.0..2.0 * PI);
        let phase_z = rng.r#gen_range(0.0..2.0 * PI);

        let positions: Vec<[f64; 3]> = (0..n_frames)
            .map(|i| {
                let t = i as f64 * dt;
                [
                    params.amplitude_x * (2.0 * PI * params.frequency * t + phase_x).sin(),
                    params.amplitude_y * (2.0 * PI * params.frequency * t + phase_y).sin(),
                    params.amplitude_z * (2.0 * PI * params.frequency * t + phase_z).sin(),
                ]
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: vec![positions.clone()],
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(positions, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        HandPosturalTremorParams {
            duration: 10.0,
            frame_rate: 60.0,
            frequency: 8.0,
            amplitude_x: 0.5,
            amplitude_y: 0.5,
            amplitude_z: 0.3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.frequency < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("frequency must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Hand kinetic tremor generator (during movement)
pub struct HandKineticTremorGenerator;

#[derive(Debug, Clone)]
pub struct HandKineticTremorParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub target_position: [f64; 3], // cm
    pub movement_time: f64,         // seconds to reach target
    pub tremor_frequency: f64,      // Hz
    pub tremor_amplitude: f64,      // cm
}

impl SyntheticGenerator for HandKineticTremorGenerator {
    type Output = Vec<[f64; 3]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HandKineticTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let positions: Vec<[f64; 3]> = (0..n_frames)
            .map(|i| {
                let t = i as f64 * dt;

                // Smooth movement to target (using sigmoid)
                let progress = if t < params.movement_time {
                    let x = 10.0 * (t / params.movement_time - 0.5);
                    1.0 / (1.0 + (-x).exp())
                } else {
                    1.0
                };

                // Base position along trajectory
                let base_x = params.target_position[0] * progress;
                let base_y = params.target_position[1] * progress;
                let base_z = params.target_position[2] * progress;

                // Add tremor that increases during movement
                let tremor_scale = if t < params.movement_time {
                    2.0 * progress * (1.0 - progress) // peaks at mid-movement
                } else {
                    0.0
                };

                let tremor = params.tremor_amplitude * tremor_scale *
                    (2.0 * PI * params.tremor_frequency * t).sin();

                [
                    base_x + tremor,
                    base_y + tremor,
                    base_z + tremor * 0.5,
                ]
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: vec![positions.clone()],
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(positions, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        HandKineticTremorParams {
            duration: 5.0,
            frame_rate: 60.0,
            target_position: [30.0, 20.0, 10.0],
            movement_time: 2.0,
            tremor_frequency: 6.0,
            tremor_amplitude: 1.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.movement_time <= 0.0 || params.movement_time > params.duration {
            return Err(crate::GeneratorError::InvalidParameter("movement_time must be positive and <= duration".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hand_postural_tremor() {
        let generator = HandPosturalTremorGenerator;
        let params = HandPosturalTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_hand_kinetic_tremor() {
        let generator = HandKineticTremorGenerator;
        let params = HandKineticTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }
}
