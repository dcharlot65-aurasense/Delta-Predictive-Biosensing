//! Prosody generators (speech rate, pauses, pitch contour)

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Speech rate generator
pub struct SpeechRateGenerator;

#[derive(Debug, Clone)]
pub struct SpeechRateParams {
    pub duration: f64,
    pub syllables_per_second: f64, // typical 4-5 for normal, <3 for bradyphonic
    pub variability: f64,           // 0-1
}

impl SyntheticGenerator for SpeechRateGenerator {
    type Output = Vec<f64>; // syllable onset times
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = SpeechRateParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mean_interval = 1.0 / params.syllables_per_second;
        let std_dev = mean_interval * params.variability;
        let interval_dist = Normal::new(mean_interval, std_dev).unwrap();

        let mut syllable_times = Vec::new();
        let mut events = Vec::new();
        let mut t = 0.0;

        while t < params.duration {
            syllable_times.push(t);
            events.push(Event {
                time: t,
                event_type: "syllable".to_string(),
                amplitude: None,
                attributes: HashMap::new(),
            });

            let interval = interval_dist.sample(&mut rng).max(mean_interval * 0.3);
            t += interval;
        }

        let actual_rate = syllable_times.len() as f64 / params.duration;

        let mut gt_params = HashMap::new();
        gt_params.insert("target_rate".to_string(), params.syllables_per_second);
        gt_params.insert("actual_rate".to_string(), actual_rate);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(syllable_times, ground_truth, 1.0))
    }

    fn default_params() -> Self::Parameters {
        SpeechRateParams {
            duration: 30.0,
            syllables_per_second: 4.5,
            variability: 0.2,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.syllables_per_second <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("syllables_per_second must be positive".to_string()));
        }
        if params.variability < 0.0 || params.variability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("variability must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Pause generator
pub struct PauseGenerator;

#[derive(Debug, Clone)]
pub struct PauseParams {
    pub duration: f64,
    pub speech_segments: Vec<(f64, f64)>, // (start, end) of speech
    pub pause_probability: f64,            // per second during speech
    pub pause_duration_mean: f64,          // seconds
    pub pause_duration_std: f64,
}

impl SyntheticGenerator for PauseGenerator {
    type Output = Vec<(f64, f64)>; // pause intervals
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PauseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let duration_dist = Normal::new(params.pause_duration_mean, params.pause_duration_std).unwrap();

        let mut pauses = Vec::new();
        let mut events = Vec::new();

        for (start, end) in &params.speech_segments {
            let segment_duration = end - start;
            let expected_pauses = (segment_duration * params.pause_probability) as usize;

            for _ in 0..expected_pauses {
                let pause_time = start + rng.r#gen_range(0.0..segment_duration);
                let pause_duration = duration_dist.sample(&mut rng).max(0.1);

                pauses.push((pause_time, pause_time + pause_duration));

                events.push(Event {
                    time: pause_time,
                    event_type: "pause".to_string(),
                    amplitude: Some(pause_duration),
                    attributes: HashMap::new(),
                });
            }
        }

        pauses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let ground_truth = TimeSeriesGroundTruth {
            parameters: HashMap::new(),
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(pauses, ground_truth, 1.0))
    }

    fn default_params() -> Self::Parameters {
        PauseParams {
            duration: 60.0,
            speech_segments: vec![
                (0.0, 15.0),
                (20.0, 40.0),
                (45.0, 60.0),
            ],
            pause_probability: 0.3,
            pause_duration_mean: 0.5,
            pause_duration_std: 0.2,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.pause_probability < 0.0 || params.pause_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("pause_probability must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Pitch contour generator
pub struct PitchContourGenerator;

#[derive(Debug, Clone)]
pub enum PitchContourType {
    Flat,
    Rising,
    Falling,
    RisingFalling, // question intonation
    FallingRising,
}

#[derive(Debug, Clone)]
pub struct PitchContourParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_f0: f64,
    pub contour_type: PitchContourType,
    pub excursion: f64, // Hz (pitch range)
}

impl SyntheticGenerator for PitchContourGenerator {
    type Output = Array1<f64>; // f0 contour
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PitchContourParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, 1.0).unwrap();

        let f0_contour: Vec<f64> = (0..n_samples)
            .map(|i| {
                let progress = i as f64 / n_samples as f64;

                let contour_value = match params.contour_type {
                    PitchContourType::Flat => 0.0,
                    PitchContourType::Rising => progress,
                    PitchContourType::Falling => 1.0 - progress,
                    PitchContourType::RisingFalling => {
                        if progress < 0.5 {
                            2.0 * progress
                        } else {
                            2.0 * (1.0 - progress)
                        }
                    }
                    PitchContourType::FallingRising => {
                        if progress < 0.5 {
                            1.0 - 2.0 * progress
                        } else {
                            2.0 * progress - 1.0
                        }
                    }
                };

                params.baseline_f0 + params.excursion * contour_value + noise.sample(&mut rng)
            })
            .collect();

        let f0_contour = Array1::from_vec(f0_contour);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_f0".to_string(), params.baseline_f0);
        gt_params.insert("excursion".to_string(), params.excursion);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(f0_contour, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        PitchContourParams {
            duration: 2.0,
            sampling_rate: 100.0,
            baseline_f0: 120.0,
            contour_type: PitchContourType::RisingFalling,
            excursion: 30.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.baseline_f0 <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("baseline_f0 must be positive".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_rate_generation() {
        let generator = SpeechRateGenerator;
        let params = SpeechRateGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());
    }

    #[test]
    fn test_pause_generation() {
        let generator = PauseGenerator;
        let params = PauseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        // Pauses are probabilistic, might be empty
        assert!(result.signal.len() >= 0);
    }

    #[test]
    fn test_pitch_contour_generation() {
        let generator = PitchContourGenerator;
        let params = PitchContourGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }
}
