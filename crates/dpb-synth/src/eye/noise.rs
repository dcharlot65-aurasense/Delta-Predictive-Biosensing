//! Eye tracking noise and artifact generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Gaze estimation noise generator (angular error std)
pub struct GazeEstimationNoiseGenerator;

#[derive(Debug, Clone)]
pub struct GazeEstimationNoiseParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub base_trajectory: Vec<[f64; 2]>,  // clean gaze trajectory
    pub angular_error_std: f64,           // degrees (typically 0.5-2.0)
    pub systematic_offset: [f64; 2],      // degrees (calibration error)
    pub temporal_drift: f64,              // degrees/second (drift over time)
}

impl SyntheticGenerator for GazeEstimationNoiseGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = GazeEstimationNoiseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, params.angular_error_std).unwrap();
        let dt = 1.0 / params.sampling_rate;

        let noisy_gaze: Vec<[f64; 2]> = params.base_trajectory
            .iter()
            .enumerate()
            .map(|(i, &[x, y])| {
                let t = i as f64 * dt;

                // Random noise
                let noise_x = noise.sample(&mut rng);
                let noise_y = noise.sample(&mut rng);

                // Systematic offset
                let offset_x = params.systematic_offset[0];
                let offset_y = params.systematic_offset[1];

                // Temporal drift
                let drift_x = params.temporal_drift * t;
                let drift_y = params.temporal_drift * t * 0.5;

                [
                    x + noise_x + offset_x + drift_x,
                    y + noise_y + offset_y + drift_y,
                ]
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(noisy_gaze, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        // Generate a simple trajectory
        let duration = 10.0;
        let sampling_rate = 500.0;
        let n_samples = (duration * sampling_rate) as usize;
        let base_trajectory = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sampling_rate;
                [5.0 * (0.5 * t).sin(), 3.0 * (0.3 * t).cos()]
            })
            .collect();

        GazeEstimationNoiseParams {
            duration,
            sampling_rate,
            base_trajectory,
            angular_error_std: 0.8,
            systematic_offset: [0.5, -0.3],
            temporal_drift: 0.1,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.base_trajectory.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("base_trajectory cannot be empty".to_string()));
        }
        Ok(())
    }
}

/// Blink artifact generator (blink rate, duration)
pub struct BlinkArtifactGenerator;

#[derive(Debug, Clone)]
pub struct BlinkArtifactParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub base_trajectory: Vec<[f64; 2]>,
    pub blink_rate: f64,              // blinks per minute (typically 15-20)
    pub blink_duration_mean: f64,     // seconds (typically 0.1-0.4)
    pub blink_duration_std: f64,      // seconds
    pub artifact_type: BlinkArtifactType,
}

#[derive(Debug, Clone)]
pub enum BlinkArtifactType {
    DataLoss,        // NaN or zero values during blink
    LastValue,       // hold last valid value
    LinearDrift,     // linear interpolation with drift
}

impl SyntheticGenerator for BlinkArtifactGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = BlinkArtifactParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let duration_dist = Normal::new(params.blink_duration_mean, params.blink_duration_std).unwrap();

        let mut gaze_with_blinks = params.base_trajectory.clone();
        let mut events = Vec::new();

        // Generate blink times
        let num_blinks = (params.duration * params.blink_rate / 60.0) as usize;
        for _ in 0..num_blinks {
            let blink_time = rng.random_range(0.0..params.duration);
            let blink_duration = duration_dist.sample(&mut rng).max(0.05).min(0.5);

            let start_idx = (blink_time * params.sampling_rate) as usize;
            let end_idx = ((blink_time + blink_duration) * params.sampling_rate) as usize;

            events.push(Event {
                time: blink_time,
                event_type: "blink".to_string(),
                amplitude: Some(blink_duration),
                attributes: HashMap::new(),
            });

            // Apply artifact
            match params.artifact_type {
                BlinkArtifactType::DataLoss => {
                    for i in start_idx..std::cmp::min(end_idx, gaze_with_blinks.len()) {
                        gaze_with_blinks[i] = [f64::NAN, f64::NAN];
                    }
                }
                BlinkArtifactType::LastValue => {
                    if start_idx > 0 {
                        let last_valid = gaze_with_blinks[start_idx - 1];
                        for i in start_idx..std::cmp::min(end_idx, gaze_with_blinks.len()) {
                            gaze_with_blinks[i] = last_valid;
                        }
                    }
                }
                BlinkArtifactType::LinearDrift => {
                    if start_idx > 0 && end_idx < gaze_with_blinks.len() {
                        let start_pos = gaze_with_blinks[start_idx - 1];
                        let end_pos = gaze_with_blinks[end_idx];
                        let n_steps = end_idx - start_idx;

                        for (j, i) in (start_idx..end_idx).enumerate() {
                            let alpha = j as f64 / n_steps as f64;
                            let drift = rng.random_range(-1.0..1.0);
                            gaze_with_blinks[i] = [
                                start_pos[0] + (end_pos[0] - start_pos[0]) * alpha + drift,
                                start_pos[1] + (end_pos[1] - start_pos[1]) * alpha + drift,
                            ];
                        }
                    }
                }
            }
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(gaze_with_blinks, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        // Generate a simple trajectory
        let duration = 30.0;
        let sampling_rate = 500.0;
        let n_samples = (duration * sampling_rate) as usize;
        let base_trajectory = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sampling_rate;
                [8.0 * (0.3 * t).sin(), 5.0 * (0.2 * t).cos()]
            })
            .collect();

        BlinkArtifactParams {
            duration,
            sampling_rate,
            base_trajectory,
            blink_rate: 17.0,
            blink_duration_mean: 0.15,
            blink_duration_std: 0.05,
            artifact_type: BlinkArtifactType::DataLoss,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.base_trajectory.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("base_trajectory cannot be empty".to_string()));
        }
        Ok(())
    }
}

/// Pupil detection failure generator (dropout rate)
pub struct PupilDetectionFailureGenerator;

#[derive(Debug, Clone)]
pub struct PupilDetectionFailureParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub base_pupil_diameter: Array1<f64>,
    pub dropout_rate: f64,            // 0-1 (fraction of samples lost)
    pub burst_failures: bool,         // clustered failures vs random
    pub burst_duration_mean: f64,     // seconds (if burst_failures)
}

impl SyntheticGenerator for PupilDetectionFailureGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PupilDetectionFailureParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut pupil_with_failures = params.base_pupil_diameter.clone();
        let n_samples = pupil_with_failures.len();

        if params.burst_failures {
            // Generate bursts of failures
            let num_bursts = (n_samples as f64 * params.dropout_rate / (params.burst_duration_mean * params.sampling_rate)) as usize;

            for _ in 0..num_bursts {
                let burst_start = rng.random_range(0..n_samples);
                let burst_samples = (params.burst_duration_mean * params.sampling_rate) as usize;

                for i in burst_start..std::cmp::min(burst_start + burst_samples, n_samples) {
                    pupil_with_failures[i] = f64::NAN;
                }
            }
        } else {
            // Random independent failures
            for i in 0..n_samples {
                if rng.random::<f64>() < params.dropout_rate {
                    pupil_with_failures[i] = f64::NAN;
                }
            }
        }

        let mut gt_params = HashMap::new();
        gt_params.insert("dropout_rate".to_string(), params.dropout_rate);
        gt_params.insert("burst_failures".to_string(), if params.burst_failures { 1.0 } else { 0.0 });

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(pupil_with_failures, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        // Generate a simple pupil diameter signal
        let duration = 30.0;
        let sampling_rate = 60.0;
        let n_samples = (duration * sampling_rate) as usize;
        let base_pupil_diameter = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 / sampling_rate;
                    4.0 + 0.5 * (0.1 * t).sin()
                })
                .collect()
        );

        PupilDetectionFailureParams {
            duration,
            sampling_rate,
            base_pupil_diameter,
            dropout_rate: 0.05,
            burst_failures: true,
            burst_duration_mean: 0.5,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.dropout_rate < 0.0 || params.dropout_rate > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("dropout_rate must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Calibration drift generator (drift rate, pattern)
pub struct CalibrationDriftGenerator;

#[derive(Debug, Clone)]
pub struct CalibrationDriftParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub base_trajectory: Vec<[f64; 2]>,
    pub drift_rate: f64,              // degrees per minute
    pub drift_pattern: DriftPattern,
}

#[derive(Debug, Clone)]
pub enum DriftPattern {
    Linear,          // constant drift direction
    Radial,          // drift toward/away from center
    Random,          // random walk drift
}

impl SyntheticGenerator for CalibrationDriftGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = CalibrationDriftParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dt = 1.0 / params.sampling_rate;

        // Random drift direction for linear pattern
        let drift_angle = rng.random_range(0.0..2.0 * std::f64::consts::PI);

        let drifted_gaze: Vec<[f64; 2]> = params.base_trajectory
            .iter()
            .enumerate()
            .map(|(i, &[x, y])| {
                let t = i as f64 * dt;
                let drift_amount = params.drift_rate * (t / 60.0); // convert to minutes

                match params.drift_pattern {
                    DriftPattern::Linear => {
                        let drift_x = drift_amount * drift_angle.cos();
                        let drift_y = drift_amount * drift_angle.sin();
                        [x + drift_x, y + drift_y]
                    }
                    DriftPattern::Radial => {
                        // Drift toward center or away from center
                        let distance = (x * x + y * y).sqrt();
                        if distance > 0.01 {
                            let radial_factor = drift_amount / distance;
                            [x * (1.0 + radial_factor), y * (1.0 + radial_factor)]
                        } else {
                            [x, y]
                        }
                    }
                    DriftPattern::Random => {
                        // Cumulative random walk (implement with mutable state)
                        // For simplicity, use deterministic noise based on time
                        let noise_x = drift_amount * ((t * 7.3).sin());
                        let noise_y = drift_amount * ((t * 5.7).cos());
                        [x + noise_x, y + noise_y]
                    }
                }
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(drifted_gaze, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        // Generate a simple trajectory
        let duration = 60.0;
        let sampling_rate = 500.0;
        let n_samples = (duration * sampling_rate) as usize;
        let base_trajectory = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sampling_rate;
                [6.0 * (0.2 * t).sin(), 4.0 * (0.15 * t).cos()]
            })
            .collect();

        CalibrationDriftParams {
            duration,
            sampling_rate,
            base_trajectory,
            drift_rate: 0.5,
            drift_pattern: DriftPattern::Linear,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.base_trajectory.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("base_trajectory cannot be empty".to_string()));
        }
        Ok(())
    }
}

/// Head movement artifact generator (head pose changes)
pub struct HeadMovementArtifactGenerator;

#[derive(Debug, Clone)]
pub struct HeadMovementArtifactParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub base_trajectory: Vec<[f64; 2]>,
    pub head_movement_frequency: f64,  // Hz (typical head movements)
    pub head_movement_amplitude: f64,  // degrees (gaze shift due to head)
    pub coupling_factor: f64,          // 0-1 (how much head movement affects gaze)
}

impl SyntheticGenerator for HeadMovementArtifactGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HeadMovementArtifactParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dt = 1.0 / params.sampling_rate;

        // Generate head movement
        let phase_offset = rng.random_range(0.0..2.0 * std::f64::consts::PI);

        let gaze_with_head: Vec<[f64; 2]> = params.base_trajectory
            .iter()
            .enumerate()
            .map(|(i, &[x, y])| {
                let t = i as f64 * dt;

                // Sinusoidal head movement
                let head_phase = 2.0 * std::f64::consts::PI * params.head_movement_frequency * t + phase_offset;
                let head_x = params.head_movement_amplitude * head_phase.sin() * params.coupling_factor;
                let head_y = params.head_movement_amplitude * head_phase.cos() * params.coupling_factor * 0.5;

                [x + head_x, y + head_y]
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(gaze_with_head, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        // Generate a simple trajectory
        let duration = 20.0;
        let sampling_rate = 500.0;
        let n_samples = (duration * sampling_rate) as usize;
        let base_trajectory = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sampling_rate;
                [5.0 * (0.4 * t).sin(), 3.0 * (0.3 * t).cos()]
            })
            .collect();

        HeadMovementArtifactParams {
            duration,
            sampling_rate,
            base_trajectory,
            head_movement_frequency: 0.3,
            head_movement_amplitude: 2.0,
            coupling_factor: 0.4,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.coupling_factor < 0.0 || params.coupling_factor > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("coupling_factor must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Glasses/contacts artifact generator (reflection, distortion)
pub struct GlassesContactsArtifactGenerator;

#[derive(Debug, Clone)]
pub struct GlassesContactsArtifactParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub base_trajectory: Vec<[f64; 2]>,
    pub artifact_severity: f64,        // 0-1 (strength of distortion)
    pub reflection_probability: f64,   // 0-1 (probability of reflection artifacts)
    pub edge_distortion: bool,         // more distortion at screen edges
}

impl SyntheticGenerator for GlassesContactsArtifactGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = GlassesContactsArtifactParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, params.artifact_severity).unwrap();

        let distorted_gaze: Vec<[f64; 2]> = params.base_trajectory
            .iter()
            .map(|&[x, y]| {
                // Random reflection artifacts
                let mut out_x = x;
                let mut out_y = y;

                if rng.random::<f64>() < params.reflection_probability {
                    // Sudden offset due to reflection
                    out_x += rng.random_range(-2.0..2.0) * params.artifact_severity;
                    out_y += rng.random_range(-2.0..2.0) * params.artifact_severity;
                }

                // Edge distortion
                if params.edge_distortion {
                    let distance_from_center = (x * x + y * y).sqrt();
                    let edge_factor = (distance_from_center / 15.0).min(1.0);
                    let distortion = noise.sample(&mut rng) * edge_factor;
                    out_x += distortion;
                    out_y += distortion * 0.7;
                }

                // General noise
                out_x += noise.sample(&mut rng) * 0.3;
                out_y += noise.sample(&mut rng) * 0.3;

                [out_x, out_y]
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(distorted_gaze, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        // Generate a simple trajectory
        let duration = 15.0;
        let sampling_rate = 500.0;
        let n_samples = (duration * sampling_rate) as usize;
        let base_trajectory = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sampling_rate;
                [7.0 * (0.3 * t).sin(), 5.0 * (0.25 * t).cos()]
            })
            .collect();

        GlassesContactsArtifactParams {
            duration,
            sampling_rate,
            base_trajectory,
            artifact_severity: 0.6,
            reflection_probability: 0.01,
            edge_distortion: true,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.artifact_severity < 0.0 || params.artifact_severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("artifact_severity must be 0-1".to_string()));
        }
        if params.reflection_probability < 0.0 || params.reflection_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("reflection_probability must be 0-1".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaze_estimation_noise() {
        let generator = GazeEstimationNoiseGenerator;
        let params = GazeEstimationNoiseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.base_trajectory.len());
    }

    #[test]
    fn test_blink_artifact() {
        let generator = BlinkArtifactGenerator;
        let params = BlinkArtifactGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.base_trajectory.len());
    }

    #[test]
    fn test_pupil_detection_failure() {
        let generator = PupilDetectionFailureGenerator;
        let params = PupilDetectionFailureGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.base_pupil_diameter.len());
    }

    #[test]
    fn test_calibration_drift() {
        let generator = CalibrationDriftGenerator;
        let params = CalibrationDriftGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.base_trajectory.len());
    }

    #[test]
    fn test_head_movement_artifact() {
        let generator = HeadMovementArtifactGenerator;
        let params = HeadMovementArtifactGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.base_trajectory.len());
    }

    #[test]
    fn test_glasses_contacts_artifact() {
        let generator = GlassesContactsArtifactGenerator;
        let params = GlassesContactsArtifactGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.base_trajectory.len());
    }
}
