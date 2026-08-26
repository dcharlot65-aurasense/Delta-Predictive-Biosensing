//! Hand-specific tremor generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth};
use rand::{Rng, RngExt, SeedableRng};
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

        let phase_x = rng.random_range(0.0..2.0 * PI);
        let phase_y = rng.random_range(0.0..2.0 * PI);
        let phase_z = rng.random_range(0.0..2.0 * PI);

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

/// Hand rest tremor generator (tremor at rest, e.g., parkinsonian)
pub struct HandRestTremorGenerator;

#[derive(Debug, Clone)]
pub struct HandRestTremorParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub frequency: f64,        // Hz (typically 4-6 for parkinsonian)
    pub amplitude: f64,        // cm
    pub regularity: f64,       // 0-1 (1 = perfectly regular)
    pub suppression_with_action: bool, // If true, tremor reduces during voluntary movement
}

impl SyntheticGenerator for HandRestTremorGenerator {
    type Output = Vec<[f64; 3]>; // hand position over time [x, y, z]
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HandRestTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let phase_noise = Normal::new(0.0, (1.0 - params.regularity) * 0.2).unwrap();
        let amp_noise = Normal::new(1.0, (1.0 - params.regularity) * 0.15).unwrap();

        let base_phase_x = rng.random_range(0.0..2.0 * PI);
        let base_phase_y = rng.random_range(0.0..2.0 * PI);
        let base_phase_z = rng.random_range(0.0..2.0 * PI);

        let positions: Vec<[f64; 3]> = (0..n_frames)
            .map(|i| {
                let t = i as f64 * dt;

                let phase_offset = phase_noise.sample(&mut rng);
                let amp_mod = amp_noise.sample(&mut rng).max(0.1);

                [
                    params.amplitude * amp_mod * (2.0 * PI * params.frequency * t + base_phase_x + phase_offset).sin(),
                    params.amplitude * amp_mod * (2.0 * PI * params.frequency * t + base_phase_y + phase_offset).sin(),
                    params.amplitude * amp_mod * 0.6 * (2.0 * PI * params.frequency * t + base_phase_z + phase_offset).sin(),
                ]
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: vec![positions.clone()],
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(positions, ground_truth, params.frame_rate)
            .with_metadata("tremor_type".to_string(), "rest".to_string()))
    }

    fn default_params() -> Self::Parameters {
        HandRestTremorParams {
            duration: 10.0,
            frame_rate: 60.0,
            frequency: 5.0,
            amplitude: 0.8,
            regularity: 0.7,
            suppression_with_action: true,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.regularity < 0.0 || params.regularity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("regularity must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Intention tremor generator (increases near target)
pub struct IntentionTremorGenerator;

#[derive(Debug, Clone)]
pub struct IntentionTremorParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub target_position: [f64; 3], // cm
    pub movement_time: f64,         // seconds to reach target
    pub base_frequency: f64,        // Hz
    pub base_amplitude: f64,        // cm at rest
    pub precision_demand: f64,      // 0-1 (how precise the target is)
    pub amplitude_scaling: f64,     // Multiplier for amplitude near target
}

impl SyntheticGenerator for IntentionTremorGenerator {
    type Output = Vec<[f64; 3]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = IntentionTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let positions: Vec<[f64; 3]> = (0..n_frames)
            .map(|i| {
                let t = i as f64 * dt;

                // Smooth movement to target
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

                // Distance to target
                let dist = ((params.target_position[0] - base_x).powi(2) +
                           (params.target_position[1] - base_y).powi(2) +
                           (params.target_position[2] - base_z).powi(2)).sqrt();
                let target_dist = (params.target_position[0].powi(2) +
                                  params.target_position[1].powi(2) +
                                  params.target_position[2].powi(2)).sqrt();

                // Intention tremor increases as we approach target
                let proximity = 1.0 - (dist / (target_dist + 0.1));
                let tremor_scale = params.base_amplitude *
                    (1.0 + params.amplitude_scaling * proximity * params.precision_demand);

                let tremor = tremor_scale * (2.0 * PI * params.base_frequency * t).sin();

                [
                    base_x + tremor,
                    base_y + tremor * 0.8,
                    base_z + tremor * 0.6,
                ]
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: vec![positions.clone()],
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(positions, ground_truth, params.frame_rate)
            .with_metadata("tremor_type".to_string(), "intention".to_string()))
    }

    fn default_params() -> Self::Parameters {
        IntentionTremorParams {
            duration: 5.0,
            frame_rate: 60.0,
            target_position: [30.0, 20.0, 10.0],
            movement_time: 3.0,
            base_frequency: 4.0,
            base_amplitude: 0.3,
            precision_demand: 0.8,
            amplitude_scaling: 5.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.precision_demand < 0.0 || params.precision_demand > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("precision_demand must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Tremor intermittency generator (on/off pattern)
pub struct TremorIntermittencyGenerator;

#[derive(Debug, Clone)]
pub struct TremorIntermittencyParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub tremor_frequency: f64,   // Hz
    pub tremor_amplitude: f64,   // cm
    pub duty_cycle: f64,          // 0-1 (fraction of time tremor is active)
    pub burst_duration_mean: f64, // seconds
    pub burst_duration_std: f64,  // seconds
    pub quiet_duration_mean: f64, // seconds
    pub quiet_duration_std: f64,  // seconds
}

impl SyntheticGenerator for TremorIntermittencyGenerator {
    type Output = Vec<[f64; 3]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = TremorIntermittencyParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let burst_dist = Normal::new(params.burst_duration_mean, params.burst_duration_std).unwrap();
        let quiet_dist = Normal::new(params.quiet_duration_mean, params.quiet_duration_std).unwrap();

        let mut positions = Vec::with_capacity(n_frames);
        let mut tremor_active = true;
        let mut state_remaining = burst_dist.sample(&mut rng).max(0.1);

        for i in 0..n_frames {
            let t = i as f64 * dt;

            // Update state if needed
            if state_remaining <= 0.0 {
                tremor_active = !tremor_active;
                state_remaining = if tremor_active {
                    burst_dist.sample(&mut rng).max(0.1)
                } else {
                    quiet_dist.sample(&mut rng).max(0.1)
                };
            }
            state_remaining -= dt;

            // Generate tremor if active
            let tremor = if tremor_active {
                params.tremor_amplitude * (2.0 * PI * params.tremor_frequency * t).sin()
            } else {
                0.0
            };

            positions.push([tremor, tremor * 0.8, tremor * 0.6]);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: vec![positions.clone()],
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(positions, ground_truth, params.frame_rate)
            .with_metadata("tremor_type".to_string(), "intermittent".to_string()))
    }

    fn default_params() -> Self::Parameters {
        TremorIntermittencyParams {
            duration: 20.0,
            frame_rate: 60.0,
            tremor_frequency: 6.0,
            tremor_amplitude: 0.8,
            duty_cycle: 0.6,
            burst_duration_mean: 3.0,
            burst_duration_std: 0.5,
            quiet_duration_mean: 2.0,
            quiet_duration_std: 0.3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.duty_cycle < 0.0 || params.duty_cycle > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("duty_cycle must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Multi-finger tremor generator (per-finger with correlation)
pub struct MultiFingerTremorGenerator;

#[derive(Debug, Clone)]
pub struct MultiFingerTremorParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub frequency: f64,           // Hz
    pub amplitude_per_finger: Vec<f64>, // cm for each finger (5 fingers)
    pub inter_finger_correlation: f64,  // 0-1 (1 = perfectly correlated)
}

impl SyntheticGenerator for MultiFingerTremorGenerator {
    type Output = Vec<Vec<[f64; 3]>>; // [frames][fingers][xyz]
    type GroundTruth = SpatialGroundTruth;
    type Parameters = MultiFingerTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate common phase and per-finger phases
        let common_phase = rng.random_range(0.0..2.0 * PI);
        let finger_phases: Vec<f64> = (0..5)
            .map(|_| rng.random_range(0.0..2.0 * PI))
            .collect();

        let mut positions = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let t = i as f64 * dt;
            let common_tremor = (2.0 * PI * params.frequency * t + common_phase).sin();

            let mut frame_positions = Vec::with_capacity(5);

            for finger_idx in 0..5 {
                let finger_phase = finger_phases[finger_idx];
                let independent_tremor = (2.0 * PI * params.frequency * t + finger_phase).sin();

                // Blend common and independent tremor based on correlation
                let combined_tremor = params.inter_finger_correlation * common_tremor +
                    (1.0 - params.inter_finger_correlation) * independent_tremor;

                let amplitude = if finger_idx < params.amplitude_per_finger.len() {
                    params.amplitude_per_finger[finger_idx]
                } else {
                    0.5 // default amplitude
                };

                frame_positions.push([
                    amplitude * combined_tremor,
                    amplitude * combined_tremor * 0.8,
                    amplitude * combined_tremor * 0.6,
                ]);
            }

            positions.push(frame_positions);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(positions, ground_truth, params.frame_rate)
            .with_metadata("tremor_type".to_string(), "multi_finger".to_string()))
    }

    fn default_params() -> Self::Parameters {
        MultiFingerTremorParams {
            duration: 10.0,
            frame_rate: 60.0,
            frequency: 6.0,
            amplitude_per_finger: vec![0.5, 0.6, 0.7, 0.6, 0.4], // thumb to pinky
            inter_finger_correlation: 0.7,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.inter_finger_correlation < 0.0 || params.inter_finger_correlation > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("inter_finger_correlation must be 0-1".to_string()));
        }
        if !params.amplitude_per_finger.is_empty() && params.amplitude_per_finger.len() != 5 {
            return Err(crate::GeneratorError::InvalidParameter("amplitude_per_finger must have 5 elements or be empty".to_string()));
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

    #[test]
    fn test_hand_rest_tremor() {
        let generator = HandRestTremorGenerator;
        let params = HandRestTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_intention_tremor() {
        let generator = IntentionTremorGenerator;
        let params = IntentionTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_tremor_intermittency() {
        let generator = TremorIntermittencyGenerator;
        let params = TremorIntermittencyGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_multi_finger_tremor() {
        let generator = MultiFingerTremorGenerator;
        let params = MultiFingerTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
        assert_eq!(result.signal[0].len(), 5); // 5 fingers
    }
}
