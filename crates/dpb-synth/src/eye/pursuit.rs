//! Smooth pursuit eye movement generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth};
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Normal smooth pursuit generator (gain ~1.0, phase lag)
pub struct NormalPursuitGenerator;

#[derive(Debug, Clone)]
pub struct NormalPursuitParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub target_frequency: f64,      // Hz (typically 0.2-0.8 Hz)
    pub target_amplitude: f64,      // degrees
    pub pursuit_gain: f64,          // ~0.9-1.0 for normal pursuit
    pub phase_lag: f64,             // radians (typically 0.1-0.3 rad)
    pub catch_up_saccades: bool,    // add corrective saccades
}

impl SyntheticGenerator for NormalPursuitGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = NormalPursuitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise = Normal::new(0.0, 0.05).unwrap();

        let mut gaze_position = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t = i as f64 * dt;

            // Target position (sinusoidal motion)
            let target_phase = 2.0 * std::f64::consts::PI * params.target_frequency * t;
            let target_x = params.target_amplitude * target_phase.sin();

            // Eye position (with gain and phase lag)
            let eye_phase = target_phase - params.phase_lag;
            let mut eye_x = params.pursuit_gain * params.target_amplitude * eye_phase.sin();

            // Add small noise
            eye_x += noise.sample(&mut rng);

            gaze_position.push([eye_x, 0.0]);
        }

        // Add occasional catch-up saccades if enabled
        if params.catch_up_saccades {
            let num_saccades = (params.duration * 0.5) as usize; // ~0.5 per second
            for _ in 0..num_saccades {
                let saccade_idx = rng.random_range(0..n_samples - 10);
                let position_error = rng.random_range(0.5..1.5);

                // Quick correction over a few samples
                for j in 0..5 {
                    if saccade_idx + j < n_samples {
                        gaze_position[saccade_idx + j][0] += position_error * (j as f64 / 5.0);
                    }
                }
            }
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(gaze_position, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        NormalPursuitParams {
            duration: 20.0,
            sampling_rate: 500.0,
            target_frequency: 0.4,
            target_amplitude: 10.0,
            pursuit_gain: 0.95,
            phase_lag: 0.2,
            catch_up_saccades: true,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.pursuit_gain < 0.0 || params.pursuit_gain > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("pursuit_gain should be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Impaired smooth pursuit generator (reduced gain, catch-up saccades)
pub struct ImpairedPursuitGenerator;

#[derive(Debug, Clone)]
pub struct ImpairedPursuitParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub target_frequency: f64,
    pub target_amplitude: f64,
    pub pursuit_gain: f64,          // 0.3-0.7 for impaired pursuit
    pub phase_lag: f64,             // increased (0.4-0.8 rad)
    pub saccadic_pursuit: bool,     // broken/saccadic pursuit pattern
    pub saccade_frequency: f64,     // catch-up saccades per second
}

impl SyntheticGenerator for ImpairedPursuitGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = ImpairedPursuitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise = Normal::new(0.0, 0.1).unwrap();

        let mut gaze_position = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t = i as f64 * dt;

            // Target position
            let target_phase = 2.0 * std::f64::consts::PI * params.target_frequency * t;
            let target_x = params.target_amplitude * target_phase.sin();

            // Eye position (reduced gain, increased lag)
            let eye_phase = target_phase - params.phase_lag;
            let mut eye_x = params.pursuit_gain * params.target_amplitude * eye_phase.sin();

            // Add larger noise for impaired pursuit
            eye_x += noise.sample(&mut rng);

            gaze_position.push([eye_x, 0.0]);
        }

        // Add catch-up saccades (more frequent in impaired pursuit)
        if params.saccadic_pursuit {
            let num_saccades = (params.duration * params.saccade_frequency) as usize;
            for _ in 0..num_saccades {
                let saccade_idx = rng.random_range(0..n_samples - 15);
                let t = saccade_idx as f64 * dt;

                // Calculate position error
                let target_phase = 2.0 * std::f64::consts::PI * params.target_frequency * t;
                let target_x = params.target_amplitude * target_phase.sin();
                let current_x = gaze_position[saccade_idx][0];
                let error = (target_x - current_x) * 0.7; // catch up partially

                // Saccade over several samples
                let saccade_duration = 10; // samples
                for j in 0..saccade_duration {
                    if saccade_idx + j < n_samples {
                        let progress = (j as f64) / (saccade_duration as f64);
                        let s = 10.0 * (progress - 0.5);
                        let position_progress = 1.0 / (1.0 + (-s).exp());
                        gaze_position[saccade_idx + j][0] += error * position_progress;
                    }
                }
            }
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(gaze_position, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        ImpairedPursuitParams {
            duration: 20.0,
            sampling_rate: 500.0,
            target_frequency: 0.4,
            target_amplitude: 10.0,
            pursuit_gain: 0.5,
            phase_lag: 0.6,
            saccadic_pursuit: true,
            saccade_frequency: 2.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.pursuit_gain < 0.0 || params.pursuit_gain > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("pursuit_gain must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Predictive smooth pursuit generator (phase lead, anticipation)
pub struct PredictivePursuitGenerator;

#[derive(Debug, Clone)]
pub struct PredictivePursuitParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub target_frequency: f64,
    pub target_amplitude: f64,
    pub pursuit_gain: f64,          // can be >1.0 with prediction
    pub phase_lead: f64,            // negative lag (anticipation, -0.1 to -0.3 rad)
    pub predictability: f64,        // 0-1 (how predictable the motion is)
    pub learning_period: f64,       // seconds (time to learn the pattern)
}

impl SyntheticGenerator for PredictivePursuitGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = PredictivePursuitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise = Normal::new(0.0, 0.03).unwrap();

        let mut gaze_position = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t = i as f64 * dt;

            // Target position
            let target_phase = 2.0 * std::f64::consts::PI * params.target_frequency * t;
            let target_x = params.target_amplitude * target_phase.sin();

            // Learning factor (gradually improve prediction)
            let learning_progress = (t / params.learning_period).min(1.0);

            // Interpolate between normal lag and predictive lead
            let current_phase_offset = params.phase_lead * learning_progress * params.predictability;

            // Eye position (with phase lead after learning)
            let eye_phase = target_phase - current_phase_offset;
            let mut eye_x = params.pursuit_gain * params.target_amplitude * eye_phase.sin();

            // Less noise with better prediction
            let noise_factor = 1.0 - (learning_progress * params.predictability * 0.5);
            eye_x += noise.sample(&mut rng) * noise_factor;

            gaze_position.push([eye_x, 0.0]);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(gaze_position, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        PredictivePursuitParams {
            duration: 30.0,
            sampling_rate: 500.0,
            target_frequency: 0.5,
            target_amplitude: 8.0,
            pursuit_gain: 1.0,
            phase_lead: -0.2, // negative = lead
            predictability: 0.8,
            learning_period: 10.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.predictability < 0.0 || params.predictability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("predictability must be 0-1".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_pursuit() {
        let generator = NormalPursuitGenerator;
        let params = NormalPursuitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_impaired_pursuit() {
        let generator = ImpairedPursuitGenerator;
        let params = ImpairedPursuitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_predictive_pursuit() {
        let generator = PredictivePursuitGenerator;
        let params = PredictivePursuitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }
}
