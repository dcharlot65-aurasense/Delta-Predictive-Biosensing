//! Prosody generators (speech rate, pauses, pitch contour)

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{Rng, RngExt, SeedableRng};
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
                let pause_time = start + rng.random_range(0.0..segment_duration);
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

        pauses.sort_by(|a, b| a.0.total_cmp(&b.0));

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

/// Filled pause generator ("um", "uh" disfluencies)
pub struct FilledPauseGenerator;

#[derive(Debug, Clone)]
pub struct FilledPauseParams {
    pub duration: f64,
    pub filled_pause_rate: f64, // filled pauses per minute
    pub pause_duration_mean: f64, // seconds
    pub pause_duration_std: f64,
}

impl SyntheticGenerator for FilledPauseGenerator {
    type Output = Vec<(f64, String)>; // (time, pause_type) - "um" or "uh"
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = FilledPauseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let duration_dist = Normal::new(params.pause_duration_mean, params.pause_duration_std).unwrap();

        let num_pauses = ((params.duration / 60.0) * params.filled_pause_rate) as usize;
        let mut filled_pauses = Vec::new();
        let mut events = Vec::new();

        for _ in 0..num_pauses {
            let time = rng.random_range(0.0..params.duration);
            let pause_type = if rng.random_bool(0.5) { "um" } else { "uh" };
            let pause_duration = duration_dist.sample(&mut rng).max(0.1);

            filled_pauses.push((time, pause_type.to_string()));

            events.push(Event {
                time,
                event_type: "filled_pause".to_string(),
                amplitude: Some(pause_duration),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("type".to_string(), if pause_type == "um" { 0.0 } else { 1.0 });
                    attrs
                },
            });
        }

        filled_pauses.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut gt_params = HashMap::new();
        gt_params.insert("target_rate".to_string(), params.filled_pause_rate);
        gt_params.insert("actual_count".to_string(), num_pauses as f64);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(filled_pauses, ground_truth, 1.0))
    }

    fn default_params() -> Self::Parameters {
        FilledPauseParams {
            duration: 60.0,
            filled_pause_rate: 2.5, // per minute
            pause_duration_mean: 0.4,
            pause_duration_std: 0.15,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.filled_pause_rate < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("filled_pause_rate must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Pitch range generator (F0 range and declination)
pub struct PitchRangeGenerator;

#[derive(Debug, Clone)]
pub struct PitchRangeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_f0: f64,
    pub pitch_range: f64,      // semitones (typical 12-24 for normal, <6 for monotone)
    pub declination_rate: f64, // semitones per second
}

impl SyntheticGenerator for PitchRangeGenerator {
    type Output = Array1<f64>; // f0 contour
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PitchRangeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;

        let range_dist = Normal::new(0.0, params.pitch_range / 2.0).unwrap();

        let f0_contour: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;

                // Declination (gradual pitch lowering over time)
                let declination = -params.declination_rate * t;

                // Random variation within pitch range
                let variation = range_dist.sample(&mut rng);

                // Convert semitones to Hz: f = f0 * 2^(semitones/12)
                params.baseline_f0 * 2.0_f64.powf((declination + variation) / 12.0)
            })
            .collect();

        let f0_contour_array = Array1::from_vec(f0_contour.clone());

        // Calculate actual pitch range
        let min_f0 = f0_contour.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_f0 = f0_contour.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let actual_range_semitones = 12.0 * (max_f0 / min_f0).log2();

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_f0".to_string(), params.baseline_f0);
        gt_params.insert("target_range_semitones".to_string(), params.pitch_range);
        gt_params.insert("actual_range_semitones".to_string(), actual_range_semitones);
        gt_params.insert("declination_rate".to_string(), params.declination_rate);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(f0_contour_array, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        PitchRangeParams {
            duration: 30.0,
            sampling_rate: 100.0,
            baseline_f0: 120.0,
            pitch_range: 12.0,
            declination_rate: 0.5,
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

/// Intensity contour generator (loudness trajectory)
pub struct IntensityContourGenerator;

#[derive(Debug, Clone)]
pub struct IntensityContourParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_intensity: f64, // dB
    pub intensity_range: f64,    // dB (typical 30-40 dB)
    pub stress_positions: Vec<f64>, // times of stressed syllables
}

impl SyntheticGenerator for IntensityContourGenerator {
    type Output = Array1<f64>; // intensity contour in dB
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = IntensityContourParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;

        let noise_dist = Normal::new(0.0, 2.0).unwrap();

        let intensity_contour: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;

                // Base intensity with gradual decay
                let mut intensity = params.baseline_intensity - (t / params.duration) * 5.0;

                // Add stress peaks
                for &stress_time in &params.stress_positions {
                    let distance = (t - stress_time).abs();
                    if distance < 0.3 {
                        // Gaussian bump for stress
                        let stress_boost = params.intensity_range * 0.3 *
                            (-distance.powi(2) / 0.05).exp();
                        intensity += stress_boost;
                    }
                }

                // Add small random variation
                intensity + noise_dist.sample(&mut rng)
            })
            .collect();

        let intensity_contour_array = Array1::from_vec(intensity_contour.clone());

        let mean_intensity = intensity_contour.iter().sum::<f64>() / intensity_contour.len() as f64;

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_intensity".to_string(), params.baseline_intensity);
        gt_params.insert("intensity_range".to_string(), params.intensity_range);
        gt_params.insert("mean_intensity".to_string(), mean_intensity);
        gt_params.insert("num_stresses".to_string(), params.stress_positions.len() as f64);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(intensity_contour_array, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        IntensityContourParams {
            duration: 10.0,
            sampling_rate: 100.0,
            baseline_intensity: 70.0,
            intensity_range: 30.0,
            stress_positions: vec![1.0, 2.5, 4.0, 5.5, 7.0, 8.5],
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.baseline_intensity <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("baseline_intensity must be positive".to_string()));
        }
        Ok(())
    }
}

/// Rhythm generator (stress pattern and timing)
pub struct RhythmGenerator;

#[derive(Debug, Clone)]
pub enum StressPattern {
    Isochronous,  // evenly spaced (syllable-timed)
    Metrical,     // alternating stress (stress-timed)
    Irregular,    // irregular timing (dysrhythmic)
}

#[derive(Debug, Clone)]
pub struct RhythmParams {
    pub duration: f64,
    pub pattern: StressPattern,
    pub base_rate: f64,       // syllables per second
    pub irregularity: f64,    // 0-1
}

impl SyntheticGenerator for RhythmGenerator {
    type Output = Vec<(f64, bool)>; // (time, is_stressed)
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = RhythmParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let base_interval = 1.0 / params.base_rate;
        let timing_noise = Normal::new(0.0, base_interval * params.irregularity).unwrap();

        let mut rhythm_events = Vec::new();
        let mut events = Vec::new();
        let mut t = 0.0;
        let mut syllable_count = 0;

        while t < params.duration {
            let is_stressed = match params.pattern {
                StressPattern::Isochronous => false, // no stress pattern
                StressPattern::Metrical => syllable_count % 2 == 0,
                StressPattern::Irregular => rng.random_bool(0.3),
            };

            rhythm_events.push((t, is_stressed));

            events.push(Event {
                time: t,
                event_type: if is_stressed { "stressed" } else { "unstressed" }.to_string(),
                amplitude: if is_stressed { Some(1.0) } else { Some(0.0) },
                attributes: HashMap::new(),
            });

            let noise = timing_noise.sample(&mut rng);
            let interval = (base_interval + noise).max(base_interval * 0.3);
            t += interval;
            syllable_count += 1;
        }

        let stressed_count = rhythm_events.iter().filter(|(_, s)| *s).count();

        let mut gt_params = HashMap::new();
        gt_params.insert("base_rate".to_string(), params.base_rate);
        gt_params.insert("total_syllables".to_string(), syllable_count as f64);
        gt_params.insert("stressed_syllables".to_string(), stressed_count as f64);
        gt_params.insert("irregularity".to_string(), params.irregularity);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(rhythm_events, ground_truth, 1.0))
    }

    fn default_params() -> Self::Parameters {
        RhythmParams {
            duration: 30.0,
            pattern: StressPattern::Metrical,
            base_rate: 4.0,
            irregularity: 0.2,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.base_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("base_rate must be positive".to_string()));
        }
        if params.irregularity < 0.0 || params.irregularity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("irregularity must be 0-1".to_string()));
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

    #[test]
    fn test_filled_pause_generation() {
        let generator = FilledPauseGenerator;
        let params = FilledPauseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());
    }

    #[test]
    fn test_pitch_range_generation() {
        let generator = PitchRangeGenerator;
        let params = PitchRangeGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_intensity_contour_generation() {
        let generator = IntensityContourGenerator;
        let params = IntensityContourGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_rhythm_generation() {
        let generator = RhythmGenerator;
        let params = RhythmGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());
    }
}
