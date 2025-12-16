//! Pupil response generators

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth};
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Pupil light reflex generator
pub struct PupilLightReflexGenerator;

#[derive(Debug, Clone)]
pub struct PupilLightReflexParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_diameter: f64,     // mm (typically 3-5)
    pub light_onset_time: f64,      // seconds
    pub light_intensity: f64,       // 0-1 (fraction of maximum constriction)
    pub constriction_latency: f64,  // seconds (typically 0.2-0.3)
    pub constriction_time: f64,     // seconds (typically 0.5-1.0)
    pub redilation_time: f64,       // seconds (typically 2-4)
}

impl SyntheticGenerator for PupilLightReflexGenerator {
    type Output = Array1<f64>; // pupil diameter over time
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PupilLightReflexParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise = Normal::new(0.0, 0.02).unwrap(); // small physiological noise

        // Maximum constriction (typically 30-40% of baseline)
        let max_constriction = params.baseline_diameter * 0.35 * params.light_intensity;

        let diameter: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;

                let base_diameter = if t < params.light_onset_time {
                    // Before light onset
                    params.baseline_diameter
                } else if t < params.light_onset_time + params.constriction_latency {
                    // Latency period
                    params.baseline_diameter
                } else if t < params.light_onset_time + params.constriction_latency + params.constriction_time {
                    // Constriction phase
                    let t_constrict = t - params.light_onset_time - params.constriction_latency;
                    let progress = t_constrict / params.constriction_time;
                    params.baseline_diameter - max_constriction * progress
                } else {
                    // Redilation phase
                    let t_redilate = t - params.light_onset_time - params.constriction_latency - params.constriction_time;
                    let progress = (t_redilate / params.redilation_time).min(1.0);
                    let current_diameter = params.baseline_diameter - max_constriction;
                    current_diameter + max_constriction * progress
                };

                base_diameter + noise.sample(&mut rng)
            })
            .collect();

        let diameter = Array1::from_vec(diameter);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_diameter".to_string(), params.baseline_diameter);
        gt_params.insert("light_intensity".to_string(), params.light_intensity);
        gt_params.insert("max_constriction".to_string(), max_constriction);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(diameter, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        PupilLightReflexParams {
            duration: 10.0,
            sampling_rate: 60.0,
            baseline_diameter: 4.5,
            light_onset_time: 2.0,
            light_intensity: 1.0,
            constriction_latency: 0.25,
            constriction_time: 0.8,
            redilation_time: 3.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.baseline_diameter <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("baseline_diameter must be positive".to_string()));
        }
        if params.light_intensity < 0.0 || params.light_intensity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("light_intensity must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Cognitive pupil dilation generator (mental workload)
pub struct CognitivePupilDilationGenerator;

#[derive(Debug, Clone)]
pub struct CognitivePupilDilationParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_diameter: f64,
    pub task_segments: Vec<(f64, f64, f64)>, // (start, end, workload 0-1)
    pub dilation_per_workload: f64,          // mm per unit workload
    pub response_time: f64,                  // seconds (tau for exponential)
}

impl SyntheticGenerator for CognitivePupilDilationGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = CognitivePupilDilationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise = Normal::new(0.0, 0.02).unwrap();

        let mut target_diameter = params.baseline_diameter;
        let mut current_diameter = params.baseline_diameter;

        let diameter: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;

                // Determine current workload
                let mut workload = 0.0;
                for (start, end, wl) in &params.task_segments {
                    if t >= *start && t <= *end {
                        workload = *wl;
                        break;
                    }
                }

                // Calculate target diameter
                target_diameter = params.baseline_diameter + params.dilation_per_workload * workload;

                // Exponential approach to target
                let alpha = 1.0 - (-dt / params.response_time).exp();
                current_diameter += alpha * (target_diameter - current_diameter);

                current_diameter + noise.sample(&mut rng)
            })
            .collect();

        let diameter = Array1::from_vec(diameter);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_diameter".to_string(), params.baseline_diameter);
        gt_params.insert("dilation_per_workload".to_string(), params.dilation_per_workload);

        let segments = params.task_segments
            .iter()
            .map(|(start, end, workload)| crate::traits::Segment {
                start: *start,
                end: *end,
                label: format!("workload_{:.1}", workload),
            })
            .collect();

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments,
        };

        Ok(GeneratedData::new(diameter, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        CognitivePupilDilationParams {
            duration: 60.0,
            sampling_rate: 60.0,
            baseline_diameter: 4.0,
            task_segments: vec![
                (10.0, 20.0, 0.5),
                (30.0, 45.0, 0.8),
            ],
            dilation_per_workload: 0.5,
            response_time: 1.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.baseline_diameter <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("baseline_diameter must be positive".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pupil_light_reflex() {
        let generator = PupilLightReflexGenerator;
        let params = PupilLightReflexGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_cognitive_pupil_dilation() {
        let generator = CognitivePupilDilationGenerator;
        let params = CognitivePupilDilationGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
        assert!(!result.ground_truth.segments.is_empty());
    }
}
