//! Pupil response generators

use crate::traits::{GeneratedData, SyntheticGenerator, TimeSeriesGroundTruth};
use ndarray::Array1;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Pupil light reflex generator
pub struct PupilLightReflexGenerator;

#[derive(Debug, Clone)]
pub struct PupilLightReflexParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_diameter: f64,    // mm (typically 3-5)
    pub light_onset_time: f64,     // seconds
    pub light_intensity: f64,      // 0-1 (fraction of maximum constriction)
    pub constriction_latency: f64, // seconds (typically 0.2-0.3)
    pub constriction_time: f64,    // seconds (typically 0.5-1.0)
    pub redilation_time: f64,      // seconds (typically 2-4)
}

impl SyntheticGenerator for PupilLightReflexGenerator {
    type Output = Array1<f64>; // pupil diameter over time
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PupilLightReflexParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
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
                } else if t < params.light_onset_time
                    + params.constriction_latency
                    + params.constriction_time
                {
                    // Constriction phase
                    let t_constrict = t - params.light_onset_time - params.constriction_latency;
                    let progress = t_constrict / params.constriction_time;
                    params.baseline_diameter - max_constriction * progress
                } else {
                    // Redilation phase
                    let t_redilate = t
                        - params.light_onset_time
                        - params.constriction_latency
                        - params.constriction_time;
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

        Ok(GeneratedData::new(
            diameter,
            ground_truth,
            params.sampling_rate,
        ))
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
            return Err(crate::GeneratorError::InvalidParameter(
                "duration must be positive".to_string(),
            ));
        }
        if params.baseline_diameter <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "baseline_diameter must be positive".to_string(),
            ));
        }
        if params.light_intensity < 0.0 || params.light_intensity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "light_intensity must be 0-1".to_string(),
            ));
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

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
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
                target_diameter =
                    params.baseline_diameter + params.dilation_per_workload * workload;

                // Exponential approach to target
                let alpha = 1.0 - (-dt / params.response_time).exp();
                current_diameter += alpha * (target_diameter - current_diameter);

                current_diameter + noise.sample(&mut rng)
            })
            .collect();

        let diameter = Array1::from_vec(diameter);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_diameter".to_string(), params.baseline_diameter);
        gt_params.insert(
            "dilation_per_workload".to_string(),
            params.dilation_per_workload,
        );

        let segments = params
            .task_segments
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

        Ok(GeneratedData::new(
            diameter,
            ground_truth,
            params.sampling_rate,
        ))
    }

    fn default_params() -> Self::Parameters {
        CognitivePupilDilationParams {
            duration: 60.0,
            sampling_rate: 60.0,
            baseline_diameter: 4.0,
            task_segments: vec![(10.0, 20.0, 0.5), (30.0, 45.0, 0.8)],
            dilation_per_workload: 0.5,
            response_time: 1.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "duration must be positive".to_string(),
            ));
        }
        if params.baseline_diameter <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "baseline_diameter must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Hippus generator (physiological pupil oscillation)
pub struct HippusGenerator;

#[derive(Debug, Clone)]
pub struct HippusParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_diameter: f64,     // mm
    pub oscillation_frequency: f64, // Hz (typically 0.05-0.3 Hz)
    pub oscillation_amplitude: f64, // mm (typically 0.1-0.5 mm)
    pub noise_level: f64,           // mm (physiological noise)
}

impl SyntheticGenerator for HippusGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = HippusParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise = Normal::new(0.0, params.noise_level).unwrap();

        let diameter: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;
                let phase = 2.0 * std::f64::consts::PI * params.oscillation_frequency * t;
                let oscillation = params.oscillation_amplitude * phase.sin();

                params.baseline_diameter + oscillation + noise.sample(&mut rng)
            })
            .collect();

        let diameter = Array1::from_vec(diameter);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_diameter".to_string(), params.baseline_diameter);
        gt_params.insert(
            "oscillation_frequency".to_string(),
            params.oscillation_frequency,
        );
        gt_params.insert(
            "oscillation_amplitude".to_string(),
            params.oscillation_amplitude,
        );

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(
            diameter,
            ground_truth,
            params.sampling_rate,
        ))
    }

    fn default_params() -> Self::Parameters {
        HippusParams {
            duration: 60.0,
            sampling_rate: 60.0,
            baseline_diameter: 4.0,
            oscillation_frequency: 0.1,
            oscillation_amplitude: 0.3,
            noise_level: 0.02,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "duration must be positive".to_string(),
            ));
        }
        if params.baseline_diameter <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "baseline_diameter must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Pupil fatigue response generator
pub struct PupilFatigueResponseGenerator;

#[derive(Debug, Clone)]
pub struct PupilFatigueResponseParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_diameter: f64,
    pub task_periods: Vec<(f64, f64)>, // (start, end) times of task periods
    pub fatigue_rate: f64,             // diameter decrease per minute
    pub recovery_rate: f64,            // diameter increase per minute during rest
    pub minimum_diameter: f64,         // mm (fatigue limit)
}

impl SyntheticGenerator for PupilFatigueResponseGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PupilFatigueResponseParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise = Normal::new(0.0, 0.02).unwrap();

        let mut current_diameter = params.baseline_diameter;
        let mut cumulative_fatigue = 0.0;

        let diameter: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;

                // Check if in task period
                let in_task = params
                    .task_periods
                    .iter()
                    .any(|(start, end)| t >= *start && t <= *end);

                if in_task {
                    // Accumulate fatigue
                    cumulative_fatigue += params.fatigue_rate * (dt / 60.0); // per minute
                } else {
                    // Recover from fatigue
                    cumulative_fatigue -= params.recovery_rate * (dt / 60.0);
                    cumulative_fatigue = cumulative_fatigue.max(0.0);
                }

                current_diameter =
                    (params.baseline_diameter - cumulative_fatigue).max(params.minimum_diameter);

                current_diameter + noise.sample(&mut rng)
            })
            .collect();

        let diameter = Array1::from_vec(diameter);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_diameter".to_string(), params.baseline_diameter);
        gt_params.insert("fatigue_rate".to_string(), params.fatigue_rate);
        gt_params.insert("recovery_rate".to_string(), params.recovery_rate);

        let segments = params
            .task_periods
            .iter()
            .enumerate()
            .map(|(idx, (start, end))| crate::traits::Segment {
                start: *start,
                end: *end,
                label: format!("task_period_{}", idx),
            })
            .collect();

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments,
        };

        Ok(GeneratedData::new(
            diameter,
            ground_truth,
            params.sampling_rate,
        ))
    }

    fn default_params() -> Self::Parameters {
        PupilFatigueResponseParams {
            duration: 600.0, // 10 minutes
            sampling_rate: 60.0,
            baseline_diameter: 4.5,
            task_periods: vec![
                (60.0, 180.0),  // 1-3 min
                (240.0, 420.0), // 4-7 min
            ],
            fatigue_rate: 0.5,  // 0.5mm per minute
            recovery_rate: 0.3, // 0.3mm per minute
            minimum_diameter: 2.5,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "duration must be positive".to_string(),
            ));
        }
        if params.baseline_diameter <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "baseline_diameter must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Afferent pupil defect generator (RAPD - Relative Afferent Pupillary Defect)
pub struct AfferentPupilDefectGenerator;

#[derive(Debug, Clone)]
pub struct AfferentPupilDefectParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_diameter: f64,
    pub light_onset_times: Vec<f64>,
    pub affected_eye: bool,        // true = affected, false = normal
    pub defect_severity: f64,      // 0-1 (0 = no defect, 1 = complete)
    pub light_intensity: f64,      // 0-1
    pub swinging_flashlight: bool, // alternating between eyes
}

impl SyntheticGenerator for AfferentPupilDefectGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = AfferentPupilDefectParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise = Normal::new(0.0, 0.02).unwrap();

        // Maximum constriction for normal eye
        let max_constriction_normal = params.baseline_diameter * 0.35 * params.light_intensity;

        // Reduced constriction for affected eye
        let max_constriction_affected = max_constriction_normal * (1.0 - params.defect_severity);

        let _max_constriction = if params.affected_eye {
            max_constriction_affected
        } else {
            max_constriction_normal
        };

        let mut events = Vec::new();
        let mut current_state = params.baseline_diameter;
        let mut target_diameter = params.baseline_diameter;

        let diameter: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;

                // Check for light onset
                for (idx, &onset_time) in params.light_onset_times.iter().enumerate() {
                    if (t - onset_time).abs() < dt * 0.5 {
                        // Light on
                        let eye_being_tested = if params.swinging_flashlight {
                            idx % 2 == 0 // alternate between eyes
                        } else {
                            params.affected_eye
                        };

                        target_diameter = if eye_being_tested && params.affected_eye {
                            // Affected eye - paradoxical dilation
                            params.baseline_diameter - max_constriction_affected
                        } else {
                            // Normal eye
                            params.baseline_diameter - max_constriction_normal
                        };
                    }

                    // Light off (2 seconds after onset)
                    let offset_time = onset_time + 2.0;
                    if (t - offset_time).abs() < dt * 0.5 {
                        target_diameter = params.baseline_diameter;
                    }
                }

                // Exponential approach to target
                let tau = 0.3; // time constant
                let alpha = 1.0 - (-dt / tau).exp();
                current_state += alpha * (target_diameter - current_state);

                current_state + noise.sample(&mut rng)
            })
            .collect();

        let diameter = Array1::from_vec(diameter);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_diameter".to_string(), params.baseline_diameter);
        gt_params.insert("defect_severity".to_string(), params.defect_severity);
        gt_params.insert(
            "affected_eye".to_string(),
            if params.affected_eye { 1.0 } else { 0.0 },
        );

        for (idx, &onset) in params.light_onset_times.iter().enumerate() {
            events.push(crate::traits::Event {
                time: onset,
                event_type: "light_onset".to_string(),
                amplitude: Some(params.light_intensity),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert(
                        "eye_tested".to_string(),
                        if params.swinging_flashlight {
                            (idx % 2) as f64
                        } else {
                            0.0
                        },
                    );
                    attrs
                },
            });
        }

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(
            diameter,
            ground_truth,
            params.sampling_rate,
        ))
    }

    fn default_params() -> Self::Parameters {
        AfferentPupilDefectParams {
            duration: 20.0,
            sampling_rate: 60.0,
            baseline_diameter: 4.5,
            light_onset_times: vec![2.0, 6.0, 10.0, 14.0],
            affected_eye: true,
            defect_severity: 0.6,
            light_intensity: 1.0,
            swinging_flashlight: true,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "duration must be positive".to_string(),
            ));
        }
        if params.defect_severity < 0.0 || params.defect_severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "defect_severity must be 0-1".to_string(),
            ));
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
        assert_eq!(
            result.signal.len(),
            (params.duration * params.sampling_rate) as usize
        );
    }

    #[test]
    fn test_cognitive_pupil_dilation() {
        let generator = CognitivePupilDilationGenerator;
        let params = CognitivePupilDilationGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(
            result.signal.len(),
            (params.duration * params.sampling_rate) as usize
        );
        assert!(!result.ground_truth.segments.is_empty());
    }

    #[test]
    fn test_hippus() {
        let generator = HippusGenerator;
        let params = HippusGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(
            result.signal.len(),
            (params.duration * params.sampling_rate) as usize
        );
    }

    #[test]
    fn test_pupil_fatigue_response() {
        let generator = PupilFatigueResponseGenerator;
        let params = PupilFatigueResponseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(
            result.signal.len(),
            (params.duration * params.sampling_rate) as usize
        );
    }

    #[test]
    fn test_afferent_pupil_defect() {
        let generator = AfferentPupilDefectGenerator;
        let params = AfferentPupilDefectGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(
            result.signal.len(),
            (params.duration * params.sampling_rate) as usize
        );
    }
}
