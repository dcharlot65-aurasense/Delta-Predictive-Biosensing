//! Fixation generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Stable fixation generator
pub struct StableFixationGenerator;

#[derive(Debug, Clone)]
pub struct StableFixationParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub fixation_position: [f64; 2], // [x, y] degrees
    pub drift_std: f64,               // degrees (typically 0.05-0.1)
}

impl SyntheticGenerator for StableFixationGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = StableFixationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let drift = Normal::new(0.0, params.drift_std).unwrap();

        // Random walk for fixational eye movements
        let mut gaze_position = Vec::with_capacity(n_samples);
        let mut current_pos = params.fixation_position;

        for _ in 0..n_samples {
            current_pos[0] += drift.sample(&mut rng);
            current_pos[1] += drift.sample(&mut rng);

            // Restoring force to prevent excessive drift
            current_pos[0] += (params.fixation_position[0] - current_pos[0]) * 0.01;
            current_pos[1] += (params.fixation_position[1] - current_pos[1]) * 0.01;

            gaze_position.push(current_pos);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(gaze_position, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        StableFixationParams {
            duration: 5.0,
            sampling_rate: 1000.0,
            fixation_position: [0.0, 0.0],
            drift_std: 0.05,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.drift_std < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("drift_std must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Microsaccade generator
pub struct MicrosaccadeGenerator;

#[derive(Debug, Clone)]
pub struct MicrosaccadeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub fixation_position: [f64; 2],
    pub microsaccade_rate: f64,  // per second (typically 1-2)
    pub amplitude_range: (f64, f64), // degrees (typically 0.1-1.0)
}

impl SyntheticGenerator for MicrosaccadeGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = MicrosaccadeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut gaze_position = vec![params.fixation_position; n_samples];
        let mut current_pos = params.fixation_position;

        // Generate microsaccade times
        let num_microsaccades = (params.duration * params.microsaccade_rate) as usize;
        let mut microsaccade_times = Vec::new();
        for _ in 0..num_microsaccades {
            microsaccade_times.push(rng.r#gen_range(0.0..params.duration));
        }
        microsaccade_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for ms_time in microsaccade_times {
            let amplitude = rng.r#gen_range(params.amplitude_range.0..params.amplitude_range.1);
            let angle = rng.r#gen_range(0.0..2.0 * std::f64::consts::PI);

            let target_pos = [
                current_pos[0] + amplitude * angle.cos(),
                current_pos[1] + amplitude * angle.sin(),
            ];

            // Microsaccade duration (very brief, ~10-30ms)
            let duration_ms = 10.0 + amplitude * 20.0;
            let duration_s = duration_ms / 1000.0;

            let start_idx = (ms_time * params.sampling_rate) as usize;
            let end_idx = ((ms_time + duration_s) * params.sampling_rate) as usize;

            for i in start_idx..std::cmp::min(end_idx, n_samples) {
                let t_local = (i - start_idx) as f64 * dt;
                let progress = (t_local / duration_s).min(1.0);
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());

                gaze_position[i] = [
                    current_pos[0] + (target_pos[0] - current_pos[0]) * position_progress,
                    current_pos[1] + (target_pos[1] - current_pos[1]) * position_progress,
                ];
            }

            if end_idx < n_samples {
                current_pos = target_pos;
                // Fill until next microsaccade
                for i in end_idx..n_samples {
                    gaze_position[i] = current_pos;
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
        MicrosaccadeParams {
            duration: 5.0,
            sampling_rate: 1000.0,
            fixation_position: [0.0, 0.0],
            microsaccade_rate: 1.5,
            amplitude_range: (0.1, 0.8),
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        Ok(())
    }
}

/// Square wave jerks generator (pathological)
pub struct SquareWaveJerksGenerator;

#[derive(Debug, Clone)]
pub struct SquareWaveJerksParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub fixation_position: [f64; 2],
    pub jerk_rate: f64,        // per second
    pub jerk_amplitude: f64,   // degrees (typically 0.5-5)
    pub intersaccadic_interval: f64, // seconds (typically 0.2)
}

impl SyntheticGenerator for SquareWaveJerksGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = SquareWaveJerksParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut gaze_position = vec![params.fixation_position; n_samples];

        let num_jerks = (params.duration * params.jerk_rate) as usize;

        for _ in 0..num_jerks {
            let jerk_time = rng.r#gen_range(0.0..params.duration - params.intersaccadic_interval);

            // Random direction
            let direction = if rng.r#gen::<bool>() { 1.0 } else { -1.0 };

            // First saccade (away from fixation)
            let start_idx1 = (jerk_time * params.sampling_rate) as usize;
            let end_idx1 = ((jerk_time + 0.02) * params.sampling_rate) as usize; // 20ms

            for i in start_idx1..std::cmp::min(end_idx1, n_samples) {
                gaze_position[i] = [
                    params.fixation_position[0] + params.jerk_amplitude * direction,
                    params.fixation_position[1],
                ];
            }

            // Intersaccadic interval
            let start_idx2 = ((jerk_time + params.intersaccadic_interval) * params.sampling_rate) as usize;
            let end_idx2 = ((jerk_time + params.intersaccadic_interval + 0.02) * params.sampling_rate) as usize;

            // Second saccade (back to fixation)
            for i in start_idx2..std::cmp::min(end_idx2, n_samples) {
                gaze_position[i] = params.fixation_position;
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
        SquareWaveJerksParams {
            duration: 5.0,
            sampling_rate: 1000.0,
            fixation_position: [0.0, 0.0],
            jerk_rate: 2.0,
            jerk_amplitude: 2.0,
            intersaccadic_interval: 0.2,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_fixation() {
        let generator = StableFixationGenerator;
        let params = StableFixationGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_microsaccade_generation() {
        let generator = MicrosaccadeGenerator;
        let params = MicrosaccadeGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }
}
