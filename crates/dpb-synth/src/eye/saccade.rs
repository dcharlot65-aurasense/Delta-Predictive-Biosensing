//! Saccade generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth, Event};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Main sequence saccade generator (normal saccades)
pub struct MainSequenceSaccadeGenerator;

#[derive(Debug, Clone)]
pub struct MainSequenceSaccadeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub saccade_count: usize,
    pub amplitude_range: (f64, f64), // degrees
}

impl SyntheticGenerator for MainSequenceSaccadeGenerator {
    type Output = Vec<[f64; 2]>; // gaze position [x, y] degrees
    type GroundTruth = SpatialGroundTruth;
    type Parameters = MainSequenceSaccadeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];
        let mut events = Vec::new();

        // Generate random saccade times
        let mut saccade_times = Vec::new();
        for _ in 0..params.saccade_count {
            saccade_times.push(rng.random_range(0.0..params.duration));
        }
        saccade_times.sort_by(|a, b| a.total_cmp(b));

        let mut current_position = [0.0, 0.0];

        for (saccade_idx, &saccade_time) in saccade_times.iter().enumerate() {
            // Generate saccade amplitude
            let amplitude = rng.random_range(params.amplitude_range.0..params.amplitude_range.1);

            // Main sequence: velocity = 20 * amplitude (deg/s per deg)
            let peak_velocity = 20.0 * amplitude;

            // Duration from main sequence: duration = 2.2 * amplitude + 21 (ms)
            let duration_ms = 2.2 * amplitude + 21.0;
            let duration_s = duration_ms / 1000.0;

            // Target position (random direction)
            let angle = rng.random_range(0.0..2.0 * std::f64::consts::PI);
            let target_position = [
                current_position[0] + amplitude * angle.cos(),
                current_position[1] + amplitude * angle.sin(),
            ];

            events.push(Event {
                time: saccade_time,
                event_type: "saccade".to_string(),
                amplitude: Some(amplitude),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("peak_velocity".to_string(), peak_velocity);
                    attrs.insert("duration".to_string(), duration_s);
                    attrs
                },
            });

            // Generate saccade trajectory
            let start_idx = (saccade_time * params.sampling_rate) as usize;
            let end_idx = ((saccade_time + duration_s) * params.sampling_rate) as usize;

            for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
                let i = start_idx + i_off;
                let t_local = (i - start_idx) as f64 * dt;
                let progress = t_local / duration_s;

                // Sigmoid-shaped velocity profile
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());

                *i_slot = [
                    current_position[0] + (target_position[0] - current_position[0]) * position_progress,
                    current_position[1] + (target_position[1] - current_position[1]) * position_progress,
                ];
            }

            // Update current position for next saccade
            if end_idx < n_samples {
                current_position = target_position;
                // Fill fixation period
                for (i_off, i_slot) in gaze_position[end_idx..n_samples].iter_mut().enumerate() {
                    let i = end_idx + i_off;
                    if saccade_idx + 1 < saccade_times.len() {
                        let next_saccade_idx = (saccade_times[saccade_idx + 1] * params.sampling_rate) as usize;
                        if i >= next_saccade_idx {
                            break;
                        }
                    }
                    *i_slot = current_position;
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
        MainSequenceSaccadeParams {
            duration: 10.0,
            sampling_rate: 500.0, // Hz (eye tracking typically 500-1000 Hz)
            saccade_count: 15,
            amplitude_range: (5.0, 30.0),
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate < 100.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate should be >= 100 Hz for eye tracking".to_string()));
        }
        Ok(())
    }
}

/// Hypometric saccade generator (undershooting)
pub struct HypometricSaccadeGenerator;

#[derive(Debug, Clone)]
pub struct HypometricSaccadeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub target_amplitude: f64,    // degrees
    pub gain: f64,                 // 0-1 (fraction of target reached)
    pub corrective_saccades: bool, // add corrective saccades
}

impl SyntheticGenerator for HypometricSaccadeGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HypometricSaccadeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let _rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];
        

        // Primary saccade at 1 second
        let saccade_time = 1.0;
        let actual_amplitude = params.target_amplitude * params.gain;
        let duration_s = (2.2 * actual_amplitude + 21.0) / 1000.0;

        let target_x = params.target_amplitude;
        let reached_x = actual_amplitude;

        let start_idx = (saccade_time * params.sampling_rate) as usize;
        let end_idx = ((saccade_time + duration_s) * params.sampling_rate) as usize;

        for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
            let i = start_idx + i_off;
            let t_local = (i - start_idx) as f64 * dt;
            let progress = (t_local / duration_s).min(1.0);
            let s = 10.0 * (progress - 0.5);
            let position_progress = 1.0 / (1.0 + (-s).exp());

            *i_slot = [reached_x * position_progress, 0.0];
        }

        let current_position = [reached_x, 0.0];

        // Fill after primary saccade
        for (i_off, i_slot) in gaze_position[end_idx..n_samples].iter_mut().enumerate() {
            let _i = end_idx + i_off;
            *i_slot = current_position;
        }

        // Add corrective saccade if enabled
        if params.corrective_saccades && end_idx < n_samples {
            let corrective_time = saccade_time + duration_s + 0.2; // 200ms later
            let corrective_amplitude = target_x - reached_x;
            let corrective_duration = (2.2 * corrective_amplitude + 21.0) / 1000.0;

            let corr_start_idx = (corrective_time * params.sampling_rate) as usize;
            let corr_end_idx = ((corrective_time + corrective_duration) * params.sampling_rate) as usize;

            for (i_off, i_slot) in gaze_position[corr_start_idx..corr_end_idx.min(n_samples)].iter_mut().enumerate() {
                let i = corr_start_idx + i_off;
                let t_local = (i - corr_start_idx) as f64 * dt;
                let progress = (t_local / corrective_duration).min(1.0);
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());

                *i_slot = [
                    reached_x + corrective_amplitude * position_progress,
                    0.0,
                ];
            }

            // Fill rest
            for (i_off, i_slot) in gaze_position[corr_end_idx..n_samples].iter_mut().enumerate() {
                let _i = corr_end_idx + i_off;
                *i_slot = [target_x, 0.0];
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
        HypometricSaccadeParams {
            duration: 3.0,
            sampling_rate: 500.0,
            target_amplitude: 20.0,
            gain: 0.7,
            corrective_saccades: true,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.gain < 0.0 || params.gain > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("gain must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Saccade latency generator
pub struct SaccadeLatencyGenerator;

#[derive(Debug, Clone)]
pub struct SaccadeLatencyParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub stimulus_times: Vec<f64>,
    pub latency_mean: f64,    // ms (typically 200-250ms)
    pub latency_std: f64,     // ms
    pub amplitude: f64,       // degrees
}

impl SyntheticGenerator for SaccadeLatencyGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = SaccadeLatencyParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let latency_dist = Normal::new(params.latency_mean, params.latency_std).unwrap();

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];
        let mut events = Vec::new();

        for stim_time in &params.stimulus_times {
            let latency = latency_dist.sample(&mut rng).max(50.0) / 1000.0; // convert to seconds
            let saccade_time = stim_time + latency;

            events.push(Event {
                time: saccade_time,
                event_type: "saccade".to_string(),
                amplitude: Some(params.amplitude),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("stimulus_time".to_string(), *stim_time);
                    attrs.insert("latency".to_string(), latency * 1000.0);
                    attrs
                },
            });

            // Generate simple saccade
            let duration = (2.2 * params.amplitude + 21.0) / 1000.0;
            let start_idx = (saccade_time * params.sampling_rate) as usize;
            let end_idx = ((saccade_time + duration) * params.sampling_rate) as usize;

            for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
                let _i = start_idx + i_off;
                *i_slot = [params.amplitude, 0.0];
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
        SaccadeLatencyParams {
            duration: 10.0,
            sampling_rate: 500.0,
            stimulus_times: vec![1.0, 3.0, 5.0, 7.0, 9.0],
            latency_mean: 220.0,
            latency_std: 30.0,
            amplitude: 10.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        Ok(())
    }
}

/// Antisaccade generator
pub struct AntisaccadeGenerator;

#[derive(Debug, Clone)]
pub struct AntisaccadeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub stimulus_times: Vec<f64>,
    pub error_rate: f64,         // 0-1 (prosaccade errors)
    pub latency_correct: f64,    // ms (longer for antisaccades)
    pub latency_error: f64,      // ms (shorter for errors)
    pub amplitude: f64,
}

impl SyntheticGenerator for AntisaccadeGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = AntisaccadeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];

        for stim_time in &params.stimulus_times {
            let is_error = rng.random::<f64>() < params.error_rate;

            let (latency, direction) = if is_error {
                (params.latency_error / 1000.0, 1.0) // toward stimulus
            } else {
                (params.latency_correct / 1000.0, -1.0) // away from stimulus
            };

            let saccade_time = stim_time + latency;
            let duration = (2.2 * params.amplitude + 21.0) / 1000.0;
            let start_idx = (saccade_time * params.sampling_rate) as usize;
            let end_idx = ((saccade_time + duration) * params.sampling_rate) as usize;

            for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
                let _i = start_idx + i_off;
                *i_slot = [params.amplitude * direction, 0.0];
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
        AntisaccadeParams {
            duration: 10.0,
            sampling_rate: 500.0,
            stimulus_times: vec![1.0, 3.0, 5.0, 7.0, 9.0],
            error_rate: 0.2,
            latency_correct: 280.0,
            latency_error: 180.0,
            amplitude: 10.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.error_rate < 0.0 || params.error_rate > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("error_rate must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Hypermetric saccade generator (overshooting, gain >1.0)
pub struct HypermetricSaccadeGenerator;

#[derive(Debug, Clone)]
pub struct HypermetricSaccadeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub target_amplitude: f64,   // degrees
    pub gain: f64,                // >1.0 (overshooting)
    pub corrective_saccades: bool,
}

impl SyntheticGenerator for HypermetricSaccadeGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HypermetricSaccadeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let _rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];
        

        // Primary saccade at 1 second (overshoots target)
        let saccade_time = 1.0;
        let actual_amplitude = params.target_amplitude * params.gain;
        let duration_s = (2.2 * actual_amplitude + 21.0) / 1000.0;

        let target_x = params.target_amplitude;
        let reached_x = actual_amplitude;

        let start_idx = (saccade_time * params.sampling_rate) as usize;
        let end_idx = ((saccade_time + duration_s) * params.sampling_rate) as usize;

        for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
            let i = start_idx + i_off;
            let t_local = (i - start_idx) as f64 * dt;
            let progress = (t_local / duration_s).min(1.0);
            let s = 10.0 * (progress - 0.5);
            let position_progress = 1.0 / (1.0 + (-s).exp());

            *i_slot = [reached_x * position_progress, 0.0];
        }

        let current_position = [reached_x, 0.0];

        // Fill after primary saccade
        for (i_off, i_slot) in gaze_position[end_idx..n_samples].iter_mut().enumerate() {
            let _i = end_idx + i_off;
            *i_slot = current_position;
        }

        // Add corrective saccade if enabled (back to target)
        if params.corrective_saccades && end_idx < n_samples {
            let corrective_time = saccade_time + duration_s + 0.15; // 150ms later
            let corrective_amplitude = (reached_x - target_x).abs();
            let corrective_duration = (2.2 * corrective_amplitude + 21.0) / 1000.0;

            let corr_start_idx = (corrective_time * params.sampling_rate) as usize;
            let corr_end_idx = ((corrective_time + corrective_duration) * params.sampling_rate) as usize;

            for (i_off, i_slot) in gaze_position[corr_start_idx..corr_end_idx.min(n_samples)].iter_mut().enumerate() {
                let i = corr_start_idx + i_off;
                let t_local = (i - corr_start_idx) as f64 * dt;
                let progress = (t_local / corrective_duration).min(1.0);
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());

                *i_slot = [
                    reached_x - corrective_amplitude * position_progress,
                    0.0,
                ];
            }

            // Fill rest
            for (i_off, i_slot) in gaze_position[corr_end_idx..n_samples].iter_mut().enumerate() {
                let _i = corr_end_idx + i_off;
                *i_slot = [target_x, 0.0];
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
        HypermetricSaccadeParams {
            duration: 3.0,
            sampling_rate: 500.0,
            target_amplitude: 20.0,
            gain: 1.2,
            corrective_saccades: true,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.gain <= 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("gain must be >1.0 for hypermetric".to_string()));
        }
        Ok(())
    }
}

/// Express saccade generator (short latency <100ms)
pub struct ExpressSaccadeGenerator;

#[derive(Debug, Clone)]
pub struct ExpressSaccadeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub stimulus_times: Vec<f64>,
    pub latency_mean: f64,    // ms (typically 80-120ms)
    pub latency_std: f64,     // ms
    pub amplitude: f64,       // degrees
    pub gap_paradigm: bool,   // gap before stimulus onset
}

impl SyntheticGenerator for ExpressSaccadeGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = ExpressSaccadeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let latency_dist = Normal::new(params.latency_mean, params.latency_std).unwrap();

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];
        let mut events = Vec::new();

        for stim_time in &params.stimulus_times {
            let latency = latency_dist.sample(&mut rng).clamp(60.0, 120.0) / 1000.0; // express range
            let saccade_time = stim_time + latency;

            events.push(Event {
                time: saccade_time,
                event_type: "express_saccade".to_string(),
                amplitude: Some(params.amplitude),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("stimulus_time".to_string(), *stim_time);
                    attrs.insert("latency".to_string(), latency * 1000.0);
                    attrs.insert("gap_paradigm".to_string(), if params.gap_paradigm { 1.0 } else { 0.0 });
                    attrs
                },
            });

            // Generate saccade
            let duration = (2.2 * params.amplitude + 21.0) / 1000.0;
            let start_idx = (saccade_time * params.sampling_rate) as usize;
            let end_idx = ((saccade_time + duration) * params.sampling_rate) as usize;

            for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
                let i = start_idx + i_off;
                let t_local = (i - start_idx) as f64 * dt;
                let progress = (t_local / duration).min(1.0);
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());
                *i_slot = [params.amplitude * position_progress, 0.0];
            }

            // Fill fixation after saccade
            for (i_off, i_slot) in gaze_position[end_idx..n_samples].iter_mut().enumerate() {
                let i = end_idx + i_off;
                if i >= start_idx {
                    *i_slot = [params.amplitude, 0.0];
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
        ExpressSaccadeParams {
            duration: 10.0,
            sampling_rate: 500.0,
            stimulus_times: vec![1.0, 3.0, 5.0, 7.0, 9.0],
            latency_mean: 90.0,
            latency_std: 10.0,
            amplitude: 10.0,
            gap_paradigm: true,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.latency_mean >= 100.0 {
            return Err(crate::GeneratorError::InvalidParameter("latency_mean should be <100ms for express saccades".to_string()));
        }
        Ok(())
    }
}

/// Delayed saccade generator (long latency >300ms)
pub struct DelayedSaccadeGenerator;

#[derive(Debug, Clone)]
pub struct DelayedSaccadeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub stimulus_times: Vec<f64>,
    pub latency_mean: f64,    // ms (typically 350-500ms)
    pub latency_std: f64,     // ms
    pub amplitude: f64,       // degrees
    pub task_difficulty: f64, // 0-1 (affects latency)
}

impl SyntheticGenerator for DelayedSaccadeGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = DelayedSaccadeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Increased latency based on task difficulty
        let adjusted_mean = params.latency_mean + (params.task_difficulty * 100.0);
        let latency_dist = Normal::new(adjusted_mean, params.latency_std).unwrap();

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];
        let mut events = Vec::new();

        for stim_time in &params.stimulus_times {
            let latency = latency_dist.sample(&mut rng).max(300.0) / 1000.0; // enforce delayed range
            let saccade_time = stim_time + latency;

            events.push(Event {
                time: saccade_time,
                event_type: "delayed_saccade".to_string(),
                amplitude: Some(params.amplitude),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("stimulus_time".to_string(), *stim_time);
                    attrs.insert("latency".to_string(), latency * 1000.0);
                    attrs.insert("task_difficulty".to_string(), params.task_difficulty);
                    attrs
                },
            });

            // Generate saccade
            let duration = (2.2 * params.amplitude + 21.0) / 1000.0;
            let start_idx = (saccade_time * params.sampling_rate) as usize;
            let end_idx = ((saccade_time + duration) * params.sampling_rate) as usize;

            for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
                let i = start_idx + i_off;
                let t_local = (i - start_idx) as f64 * dt;
                let progress = (t_local / duration).min(1.0);
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());
                *i_slot = [params.amplitude * position_progress, 0.0];
            }

            // Fill after
            for (i_off, i_slot) in gaze_position[end_idx..n_samples].iter_mut().enumerate() {
                let _i = end_idx + i_off;
                *i_slot = [params.amplitude, 0.0];
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
        DelayedSaccadeParams {
            duration: 10.0,
            sampling_rate: 500.0,
            stimulus_times: vec![1.0, 3.0, 5.0, 7.0],
            latency_mean: 400.0,
            latency_std: 50.0,
            amplitude: 10.0,
            task_difficulty: 0.5,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.latency_mean < 300.0 {
            return Err(crate::GeneratorError::InvalidParameter("latency_mean should be >=300ms for delayed saccades".to_string()));
        }
        Ok(())
    }
}

/// Corrective saccade generator (post-saccadic correction)
pub struct CorrectiveSaccadeGenerator;

#[derive(Debug, Clone)]
pub struct CorrectiveSaccadeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub target_amplitude: f64,        // degrees
    pub primary_gain: f64,             // 0.7-0.9 (initial undershoot)
    pub corrective_delay: f64,         // ms (typically 100-150ms)
    pub num_corrections: usize,        // number of corrective saccades
    pub correction_accuracy: f64,      // 0-1 (how well corrections work)
}

impl SyntheticGenerator for CorrectiveSaccadeGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = CorrectiveSaccadeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let _rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];
        let mut current_x;
        let mut events = Vec::new();

        // Primary saccade
        let primary_time = 1.0;
        let primary_amplitude = params.target_amplitude * params.primary_gain;
        let primary_duration = (2.2 * primary_amplitude + 21.0) / 1000.0;

        let start_idx = (primary_time * params.sampling_rate) as usize;
        let end_idx = ((primary_time + primary_duration) * params.sampling_rate) as usize;

        for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
            let i = start_idx + i_off;
            let t_local = (i - start_idx) as f64 * dt;
            let progress = (t_local / primary_duration).min(1.0);
            let s = 10.0 * (progress - 0.5);
            let position_progress = 1.0 / (1.0 + (-s).exp());
            *i_slot = [primary_amplitude * position_progress, 0.0];
        }

        current_x = primary_amplitude;
        events.push(Event {
            time: primary_time,
            event_type: "primary_saccade".to_string(),
            amplitude: Some(primary_amplitude),
            attributes: HashMap::new(),
        });

        // Fill between saccades
        for (i_off, i_slot) in gaze_position[end_idx..n_samples].iter_mut().enumerate() {
            let _i = end_idx + i_off;
            *i_slot = [current_x, 0.0];
        }

        // Corrective saccades
        let mut correction_time = primary_time + primary_duration + (params.corrective_delay / 1000.0);

        for correction_num in 0..params.num_corrections {
            if correction_time >= params.duration - 0.5 {
                break;
            }

            let error = params.target_amplitude - current_x;
            if error.abs() < 0.1 {
                break; // close enough
            }

            let correction_amplitude = error * params.correction_accuracy;
            let correction_duration = (2.2 * correction_amplitude.abs() + 21.0) / 1000.0;

            let corr_start_idx = (correction_time * params.sampling_rate) as usize;
            let corr_end_idx = ((correction_time + correction_duration) * params.sampling_rate) as usize;

            for (i_off, i_slot) in gaze_position[corr_start_idx..corr_end_idx.min(n_samples)].iter_mut().enumerate() {
                let i = corr_start_idx + i_off;
                let t_local = (i - corr_start_idx) as f64 * dt;
                let progress = (t_local / correction_duration).min(1.0);
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());
                *i_slot = [current_x + correction_amplitude * position_progress, 0.0];
            }

            current_x += correction_amplitude;
            events.push(Event {
                time: correction_time,
                event_type: format!("corrective_saccade_{}", correction_num + 1),
                amplitude: Some(correction_amplitude.abs()),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("remaining_error".to_string(), (params.target_amplitude - current_x).abs());
                    attrs
                },
            });

            // Fill after this correction
            for (i_off, i_slot) in gaze_position[corr_end_idx..n_samples].iter_mut().enumerate() {
                let _i = corr_end_idx + i_off;
                *i_slot = [current_x, 0.0];
            }

            correction_time += correction_duration + (params.corrective_delay / 1000.0);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(gaze_position, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        CorrectiveSaccadeParams {
            duration: 5.0,
            sampling_rate: 500.0,
            target_amplitude: 20.0,
            primary_gain: 0.75,
            corrective_delay: 120.0,
            num_corrections: 2,
            correction_accuracy: 0.8,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.primary_gain <= 0.0 || params.primary_gain >= 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("primary_gain must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Saccade sequence generator (scanpath pattern)
pub struct SaccadeSequenceGenerator;

#[derive(Debug, Clone)]
pub struct SaccadeSequenceParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub target_positions: Vec<[f64; 2]>, // sequence of x,y positions
    pub fixation_durations: Vec<f64>,    // duration at each target (seconds)
    pub saccade_accuracy: f64,            // 0-1 (spatial accuracy)
}

impl SyntheticGenerator for SaccadeSequenceGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = SaccadeSequenceParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];
        let mut current_pos = [0.0, 0.0];
        let mut current_time = 0.0;
        let mut events = Vec::new();

        let noise_dist = Normal::new(0.0, 0.5 * (1.0 - params.saccade_accuracy)).unwrap();

        for (idx, target_pos) in params.target_positions.iter().enumerate() {
            if current_time >= params.duration {
                break;
            }

            // Add spatial noise to target
            let noisy_target = [
                target_pos[0] + noise_dist.sample(&mut rng),
                target_pos[1] + noise_dist.sample(&mut rng),
            ];

            // Calculate saccade amplitude
            let dx = noisy_target[0] - current_pos[0];
            let dy = noisy_target[1] - current_pos[1];
            let amplitude = (dx * dx + dy * dy).sqrt();

            // Saccade duration
            let saccade_duration = (2.2 * amplitude + 21.0) / 1000.0;

            let start_idx = (current_time * params.sampling_rate) as usize;
            let end_idx = ((current_time + saccade_duration) * params.sampling_rate) as usize;

            // Generate saccade
            for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
                let i = start_idx + i_off;
                let t_local = (i - start_idx) as f64 * dt;
                let progress = (t_local / saccade_duration).min(1.0);
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());

                *i_slot = [
                    current_pos[0] + dx * position_progress,
                    current_pos[1] + dy * position_progress,
                ];
            }

            events.push(Event {
                time: current_time,
                event_type: format!("saccade_to_target_{}", idx),
                amplitude: Some(amplitude),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("target_x".to_string(), target_pos[0]);
                    attrs.insert("target_y".to_string(), target_pos[1]);
                    attrs
                },
            });

            current_pos = noisy_target;
            current_time += saccade_duration;

            // Fixation period
            let fixation_duration = if idx < params.fixation_durations.len() {
                params.fixation_durations[idx]
            } else {
                0.3 // default 300ms
            };

            let fix_start_idx = (current_time * params.sampling_rate) as usize;
            let fix_end_idx = ((current_time + fixation_duration) * params.sampling_rate) as usize;

            for (i_off, i_slot) in gaze_position[fix_start_idx..fix_end_idx.min(n_samples)].iter_mut().enumerate() {
                let _i = fix_start_idx + i_off;
                *i_slot = current_pos;
            }

            current_time += fixation_duration;
        }

        // Fill remaining time
        gaze_position[((current_time * params.sampling_rate) as usize)..n_samples]
            .fill(current_pos);

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(gaze_position, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        SaccadeSequenceParams {
            duration: 10.0,
            sampling_rate: 500.0,
            target_positions: vec![
                [5.0, 0.0],
                [5.0, 5.0],
                [-5.0, 5.0],
                [-5.0, -5.0],
                [0.0, 0.0],
            ],
            fixation_durations: vec![0.5, 0.5, 0.5, 0.5, 0.5],
            saccade_accuracy: 0.9,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.target_positions.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("target_positions cannot be empty".to_string()));
        }
        Ok(())
    }
}

/// Memory-guided saccade generator (delay period, accuracy)
pub struct MemoryGuidedSaccadeGenerator;

#[derive(Debug, Clone)]
pub struct MemoryGuidedSaccadeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub target_positions: Vec<[f64; 2]>,
    pub cue_times: Vec<f64>,           // when target is shown
    pub delay_period: f64,             // seconds (memory delay)
    pub memory_decay: f64,             // 0-1 per second (spatial accuracy loss)
}

impl SyntheticGenerator for MemoryGuidedSaccadeGenerator {
    type Output = Vec<[f64; 2]>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = MemoryGuidedSaccadeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut gaze_position = vec![[0.0, 0.0]; n_samples];
        let mut events = Vec::new();

        for (&cue_time, target_pos) in params.cue_times.iter().zip(&params.target_positions) {
            let saccade_time = cue_time + params.delay_period;

            if saccade_time >= params.duration {
                break;
            }

            // Memory decay increases spatial error
            let memory_error_std = params.memory_decay * params.delay_period;
            let error_dist = Normal::new(0.0, memory_error_std).unwrap();

            let remembered_target = [
                target_pos[0] + error_dist.sample(&mut rng),
                target_pos[1] + error_dist.sample(&mut rng),
            ];

            // Calculate amplitude
            let dx = remembered_target[0];
            let dy = remembered_target[1];
            let amplitude = (dx * dx + dy * dy).sqrt();

            // Saccade duration
            let saccade_duration = (2.2 * amplitude + 21.0) / 1000.0;

            let start_idx = (saccade_time * params.sampling_rate) as usize;
            let end_idx = ((saccade_time + saccade_duration) * params.sampling_rate) as usize;

            // Generate saccade
            for (i_off, i_slot) in gaze_position[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
                let i = start_idx + i_off;
                let t_local = (i - start_idx) as f64 * dt;
                let progress = (t_local / saccade_duration).min(1.0);
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());

                *i_slot = [
                    dx * position_progress,
                    dy * position_progress,
                ];
            }

            events.push(Event {
                time: saccade_time,
                event_type: "memory_guided_saccade".to_string(),
                amplitude: Some(amplitude),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("cue_time".to_string(), cue_time);
                    attrs.insert("delay_period".to_string(), params.delay_period);
                    attrs.insert("target_error".to_string(),
                        ((remembered_target[0] - target_pos[0]).powi(2) +
                         (remembered_target[1] - target_pos[1]).powi(2)).sqrt());
                    attrs
                },
            });

            // Fill after saccade
            for (i_off, i_slot) in gaze_position[end_idx..n_samples].iter_mut().enumerate() {
                let _i = end_idx + i_off;
                *i_slot = remembered_target;
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
        MemoryGuidedSaccadeParams {
            duration: 20.0,
            sampling_rate: 500.0,
            target_positions: vec![
                [10.0, 0.0],
                [0.0, 10.0],
                [-10.0, 0.0],
                [0.0, -10.0],
            ],
            cue_times: vec![1.0, 6.0, 11.0, 16.0],
            delay_period: 2.0,
            memory_decay: 0.5,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.target_positions.len() != params.cue_times.len() {
            return Err(crate::GeneratorError::InvalidParameter("target_positions and cue_times must have same length".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_sequence_saccade() {
        let generator = MainSequenceSaccadeGenerator;
        let params = MainSequenceSaccadeGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_hypometric_saccade() {
        let generator = HypometricSaccadeGenerator;
        let params = HypometricSaccadeGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_hypermetric_saccade() {
        let generator = HypermetricSaccadeGenerator;
        let params = HypermetricSaccadeGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_express_saccade() {
        let generator = ExpressSaccadeGenerator;
        let params = ExpressSaccadeGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_corrective_saccade() {
        let generator = CorrectiveSaccadeGenerator;
        let params = CorrectiveSaccadeGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_saccade_sequence() {
        let generator = SaccadeSequenceGenerator;
        let params = SaccadeSequenceGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_memory_guided_saccade() {
        let generator = MemoryGuidedSaccadeGenerator;
        let params = MemoryGuidedSaccadeGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }
}
