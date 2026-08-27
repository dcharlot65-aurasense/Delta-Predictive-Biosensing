//! Pathological voice generators (dysarthria types, hypophonia)

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Hypokinetic dysarthria generator (Parkinsonian speech)
/// Characteristics: reduced loudness, monotone, imprecise articulation, short rushes of speech
pub struct HypokineticDysarthriaGenerator;

#[derive(Debug, Clone)]
pub struct HypokineticDysarthriaParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub severity: f64,           // 0-1 (0 = mild, 1 = severe)
    pub baseline_f0: f64,
    pub baseline_intensity: f64,
}

impl SyntheticGenerator for HypokineticDysarthriaGenerator {
    type Output = HypokineticFeatures;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = HypokineticDysarthriaParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;

        // Reduced pitch range (monotone)
        let pitch_range_reduction = 1.0 - (params.severity * 0.7); // reduce by up to 70%
        let pitch_range_semitones = 12.0 * pitch_range_reduction;

        // Generate F0 contour with reduced variability
        let noise_dist = Normal::new(0.0, pitch_range_semitones / 4.0).unwrap();
        let f0_contour: Vec<f64> = (0..n_samples)
            .map(|_| {
                let variation = noise_dist.sample(&mut rng);
                params.baseline_f0 * 2.0_f64.powf(variation / 12.0)
            })
            .collect();

        // Reduced intensity
        let intensity_reduction = params.severity * 15.0; // up to 15 dB reduction
        let reduced_intensity = params.baseline_intensity - intensity_reduction;

        let intensity_noise = Normal::new(0.0, 2.0 * (1.0 - params.severity)).unwrap();
        let intensity_contour: Vec<f64> = (0..n_samples)
            .map(|_| reduced_intensity + intensity_noise.sample(&mut rng))
            .collect();

        // Imprecise articulation (increased VOT variability)
        let vot_variability = 0.3 + params.severity * 0.4; // 0.3-0.7

        // Short rushes of speech (festination)
        let speech_rate_increase = 1.0 + params.severity * 0.5; // up to 50% faster
        let base_syllable_rate = 4.5 * speech_rate_increase;

        let features = HypokineticFeatures {
            f0_contour: Array1::from_vec(f0_contour.clone()),
            intensity_contour: Array1::from_vec(intensity_contour.clone()),
            pitch_range_semitones,
            mean_intensity: reduced_intensity,
            vot_variability,
            syllable_rate: base_syllable_rate,
        };

        let mut gt_params = HashMap::new();
        gt_params.insert("severity".to_string(), params.severity);
        gt_params.insert("pitch_range_semitones".to_string(), pitch_range_semitones);
        gt_params.insert("mean_intensity_db".to_string(), reduced_intensity);
        gt_params.insert("intensity_reduction_db".to_string(), intensity_reduction);
        gt_params.insert("vot_variability".to_string(), vot_variability);
        gt_params.insert("syllable_rate".to_string(), base_syllable_rate);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(features, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        HypokineticDysarthriaParams {
            duration: 10.0,
            sampling_rate: 100.0,
            severity: 0.6,
            baseline_f0: 120.0,
            baseline_intensity: 70.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.severity < 0.0 || params.severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("severity must be 0-1".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct HypokineticFeatures {
    pub f0_contour: Array1<f64>,
    pub intensity_contour: Array1<f64>,
    pub pitch_range_semitones: f64,
    pub mean_intensity: f64,
    pub vot_variability: f64,
    pub syllable_rate: f64,
}

/// Spastic dysarthria generator (UMN lesion)
/// Characteristics: strained-strangled quality, slow rate, pitch breaks, hypernasality
pub struct SpasticDysarthriaGenerator;

#[derive(Debug, Clone)]
pub struct SpasticDysarthriaParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub severity: f64,
    pub baseline_f0: f64,
}

impl SyntheticGenerator for SpasticDysarthriaGenerator {
    type Output = SpasticFeatures;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = SpasticDysarthriaParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;

        // Slow speech rate
        let rate_reduction = 1.0 - params.severity * 0.5; // up to 50% slower
        let syllable_rate = 4.5 * rate_reduction;

        // Pitch breaks (sudden F0 jumps)
        let pitch_break_probability = params.severity * 0.1; // 0-10% of samples
        let mut f0_contour = Vec::with_capacity(n_samples);
        let mut pitch_breaks = Vec::new();

        let base_noise = Normal::new(0.0, 2.0).unwrap();

        for i in 0..n_samples {
            let base_f0 = params.baseline_f0 + base_noise.sample(&mut rng);

            // Check for pitch break
            if rng.random_bool(pitch_break_probability) {
                // Sudden jump up or down
                let break_magnitude = rng.random_range(5.0..15.0); // semitones
                let break_direction = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                let f0 = base_f0 * 2.0_f64.powf(break_magnitude * break_direction / 12.0);
                f0_contour.push(f0);

                pitch_breaks.push(Event {
                    time: i as f64 / params.sampling_rate,
                    event_type: "pitch_break".to_string(),
                    amplitude: Some(break_magnitude),
                    attributes: HashMap::new(),
                });
            } else {
                f0_contour.push(base_f0);
            }
        }

        // Strained voice quality (increased jitter and shimmer)
        let jitter_percent = 1.5 + params.severity * 2.0; // 1.5-3.5%
        let shimmer_percent = 4.0 + params.severity * 4.0; // 4-8%

        // Slow articulation rate
        let articulation_duration_increase = 1.0 + params.severity * 0.8; // up to 80% longer

        let features = SpasticFeatures {
            f0_contour: Array1::from_vec(f0_contour),
            syllable_rate,
            jitter_percent,
            shimmer_percent,
            articulation_duration_increase,
            num_pitch_breaks: pitch_breaks.len(),
        };

        let mut gt_params = HashMap::new();
        gt_params.insert("severity".to_string(), params.severity);
        gt_params.insert("syllable_rate".to_string(), syllable_rate);
        gt_params.insert("jitter_percent".to_string(), jitter_percent);
        gt_params.insert("shimmer_percent".to_string(), shimmer_percent);
        gt_params.insert("num_pitch_breaks".to_string(), pitch_breaks.len() as f64);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: pitch_breaks,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(features, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        SpasticDysarthriaParams {
            duration: 10.0,
            sampling_rate: 100.0,
            severity: 0.6,
            baseline_f0: 120.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.severity < 0.0 || params.severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("severity must be 0-1".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SpasticFeatures {
    pub f0_contour: Array1<f64>,
    pub syllable_rate: f64,
    pub jitter_percent: f64,
    pub shimmer_percent: f64,
    pub articulation_duration_increase: f64,
    pub num_pitch_breaks: usize,
}

/// Ataxic dysarthria generator (cerebellar damage)
/// Characteristics: irregular articulatory breakdown, scanning speech, excess loudness variation
pub struct AtaxicDysarthriaGenerator;

#[derive(Debug, Clone)]
pub struct AtaxicDysarthriaParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub severity: f64,
    pub baseline_f0: f64,
    pub baseline_intensity: f64,
}

impl SyntheticGenerator for AtaxicDysarthriaGenerator {
    type Output = AtaxicFeatures;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = AtaxicDysarthriaParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;

        // Irregular timing (scanning speech - equal stress on syllables)
        let timing_irregularity = 0.2 + params.severity * 0.6; // 0.2-0.8

        // Excessive pitch variability
        let pitch_variability = 5.0 + params.severity * 15.0; // 5-20 semitones
        let pitch_noise = Normal::new(0.0, pitch_variability).unwrap();

        let f0_contour: Vec<f64> = (0..n_samples)
            .map(|_| {
                let variation = pitch_noise.sample(&mut rng);
                params.baseline_f0 * 2.0_f64.powf(variation / 12.0)
            })
            .collect();

        // Excessive loudness variation
        let intensity_variability = 5.0 + params.severity * 15.0; // 5-20 dB
        let intensity_noise = Normal::new(0.0, intensity_variability).unwrap();

        let intensity_contour: Vec<f64> = (0..n_samples)
            .map(|_| params.baseline_intensity + intensity_noise.sample(&mut rng))
            .collect();

        // Irregular articulation (high consonant imprecision)
        let consonant_precision = 1.0 - (0.3 + params.severity * 0.5); // 0.2-0.7

        // Syllable rate with high variability
        let base_rate = 4.0; // slightly slow
        let _rate_cv = timing_irregularity;

        let features = AtaxicFeatures {
            f0_contour: Array1::from_vec(f0_contour.clone()),
            intensity_contour: Array1::from_vec(intensity_contour.clone()),
            pitch_variability_semitones: pitch_variability,
            intensity_variability_db: intensity_variability,
            timing_irregularity,
            consonant_precision,
            base_syllable_rate: base_rate,
        };

        let f0_std = Self::std_dev(&f0_contour);
        let intensity_std = Self::std_dev(&intensity_contour);

        let mut gt_params = HashMap::new();
        gt_params.insert("severity".to_string(), params.severity);
        gt_params.insert("pitch_variability".to_string(), pitch_variability);
        gt_params.insert("f0_std".to_string(), f0_std);
        gt_params.insert("intensity_variability".to_string(), intensity_variability);
        gt_params.insert("intensity_std".to_string(), intensity_std);
        gt_params.insert("timing_irregularity".to_string(), timing_irregularity);
        gt_params.insert("consonant_precision".to_string(), consonant_precision);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(features, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        AtaxicDysarthriaParams {
            duration: 10.0,
            sampling_rate: 100.0,
            severity: 0.6,
            baseline_f0: 120.0,
            baseline_intensity: 70.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.severity < 0.0 || params.severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("severity must be 0-1".to_string()));
        }
        Ok(())
    }
}

impl AtaxicDysarthriaGenerator {
    fn std_dev(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct AtaxicFeatures {
    pub f0_contour: Array1<f64>,
    pub intensity_contour: Array1<f64>,
    pub pitch_variability_semitones: f64,
    pub intensity_variability_db: f64,
    pub timing_irregularity: f64,
    pub consonant_precision: f64,
    pub base_syllable_rate: f64,
}

/// Hypophonia generator (progressive voice softening)
/// Characteristic of Parkinson's disease - gradual reduction in loudness
pub struct HypophoniaGenerator;

#[derive(Debug, Clone)]
pub struct HypophoniaParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub initial_intensity: f64,  // dB
    pub final_intensity: f64,     // dB
    pub progression_type: ProgressionType,
}

#[derive(Debug, Clone)]
pub enum ProgressionType {
    Linear,      // steady decline
    Exponential, // accelerating decline
    Stepped,     // sudden drops
}

impl SyntheticGenerator for HypophoniaGenerator {
    type Output = Array1<f64>; // intensity trajectory
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = HypophoniaParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;
        let noise_dist = Normal::new(0.0, 1.0).unwrap();

        let intensity_trajectory: Vec<f64> = (0..n_samples)
            .map(|i| {
                let progress = i as f64 / n_samples as f64;

                let base_intensity = match params.progression_type {
                    ProgressionType::Linear => {
                        params.initial_intensity + progress * (params.final_intensity - params.initial_intensity)
                    }
                    ProgressionType::Exponential => {
                        // Use exponential decay
                        let decay_rate = (params.final_intensity / params.initial_intensity).ln();
                        params.initial_intensity * (decay_rate * progress).exp()
                    }
                    ProgressionType::Stepped => {
                        // Step down every 20% of duration
                        let step = (progress * 5.0).floor();
                        let step_size = (params.initial_intensity - params.final_intensity) / 5.0;
                        params.initial_intensity - step * step_size
                    }
                };

                // Add small random variation
                base_intensity + noise_dist.sample(&mut rng)
            })
            .collect();

        let intensity_array = Array1::from_vec(intensity_trajectory.clone());

        let total_reduction = params.initial_intensity - params.final_intensity;
        let mean_intensity = intensity_trajectory.iter().sum::<f64>() / intensity_trajectory.len() as f64;

        let mut gt_params = HashMap::new();
        gt_params.insert("initial_intensity".to_string(), params.initial_intensity);
        gt_params.insert("final_intensity".to_string(), params.final_intensity);
        gt_params.insert("total_reduction_db".to_string(), total_reduction);
        gt_params.insert("mean_intensity".to_string(), mean_intensity);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(intensity_array, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        HypophoniaParams {
            duration: 30.0,
            sampling_rate: 100.0,
            initial_intensity: 70.0,
            final_intensity: 55.0,
            progression_type: ProgressionType::Linear,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.initial_intensity <= params.final_intensity {
            return Err(crate::GeneratorError::InvalidParameter("initial_intensity must be greater than final_intensity".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypokinetic_dysarthria() {
        let generator = HypokineticDysarthriaGenerator;
        let params = HypokineticDysarthriaGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.f0_contour.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_spastic_dysarthria() {
        let generator = SpasticDysarthriaGenerator;
        let params = SpasticDysarthriaGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.f0_contour.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_ataxic_dysarthria() {
        let generator = AtaxicDysarthriaGenerator;
        let params = AtaxicDysarthriaGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.f0_contour.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_hypophonia() {
        let generator = HypophoniaGenerator;
        let params = HypophoniaGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }
}
