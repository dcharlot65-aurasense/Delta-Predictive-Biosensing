//! Articulation generators (vowel space, formant transitions, consonant precision)

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Vowel space generator (F1-F2 positions)
pub struct VowelSpaceGenerator;

#[derive(Debug, Clone)]
pub struct VowelSpaceParams {
    pub num_vowels: usize,
    pub vowel_type: VowelType,
    pub variability: f64, // Hz standard deviation
}

#[derive(Debug, Clone)]
pub enum VowelType {
    CornerVowels,  // /i/, /a/, /u/ (maximum contrast)
    AllVowels,     // Full vowel space
    Reduced,       // Centralized vowels (schwa region)
}

impl SyntheticGenerator for VowelSpaceGenerator {
    type Output = Vec<(f64, f64)>; // (F1, F2) pairs in Hz
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = VowelSpaceParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Canonical formant values for vowels (adult male averages)
        let canonical_vowels = match params.vowel_type {
            VowelType::CornerVowels => vec![
                (270.0, 2290.0),  // /i/ (high front)
                (730.0, 1090.0),  // /a/ (low central)
                (300.0, 870.0),   // /u/ (high back)
            ],
            VowelType::AllVowels => vec![
                (270.0, 2290.0),  // /i/
                (390.0, 1990.0),  // /e/
                (610.0, 1900.0),  // /ɛ/
                (730.0, 1090.0),  // /a/
                (570.0, 840.0),   // /ɔ/
                (440.0, 1020.0),  // /o/
                (300.0, 870.0),   // /u/
            ],
            VowelType::Reduced => vec![
                (500.0, 1500.0),  // schwa region
                (480.0, 1450.0),
                (520.0, 1550.0),
            ],
        };

        let noise_dist = Normal::new(0.0, params.variability).unwrap();
        let mut vowel_points = Vec::new();

        for _ in 0..params.num_vowels {
            // Select random canonical vowel
            let &(f1_base, f2_base) = canonical_vowels.get(rng.random_range(0..canonical_vowels.len())).unwrap();

            // Add variability
            let f1 = (f1_base + noise_dist.sample(&mut rng)).max(200.0);
            let f2 = (f2_base + noise_dist.sample(&mut rng)).max(500.0);

            vowel_points.push((f1, f2));
        }

        // Calculate vowel space area (using convex hull approximation)
        let f1_values: Vec<f64> = vowel_points.iter().map(|(f1, _)| *f1).collect();
        let f2_values: Vec<f64> = vowel_points.iter().map(|(_, f2)| *f2).collect();

        let f1_range = f1_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max) -
                       f1_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let f2_range = f2_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max) -
                       f2_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let vowel_space_area = f1_range * f2_range;

        let mut gt_params = HashMap::new();
        gt_params.insert("vowel_space_area".to_string(), vowel_space_area);
        gt_params.insert("f1_range".to_string(), f1_range);
        gt_params.insert("f2_range".to_string(), f2_range);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(vowel_points, ground_truth, 1.0))
    }

    fn default_params() -> Self::Parameters {
        VowelSpaceParams {
            num_vowels: 50,
            vowel_type: VowelType::CornerVowels,
            variability: 50.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.num_vowels == 0 {
            return Err(crate::GeneratorError::InvalidParameter("num_vowels must be positive".to_string()));
        }
        if params.variability < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("variability must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Vowel centralization generator (vowel reduction)
pub struct VowelCentralizationGenerator;

#[derive(Debug, Clone)]
pub struct VowelCentralizationParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub initial_f1: f64,
    pub initial_f2: f64,
    pub centralization_degree: f64, // 0-1 (0 = no reduction, 1 = full reduction to schwa)
}

impl SyntheticGenerator for VowelCentralizationGenerator {
    type Output = Vec<(f64, f64)>; // (F1, F2) trajectory
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = VowelCentralizationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;

        // Schwa targets (neutral vowel)
        let schwa_f1 = 500.0;
        let schwa_f2 = 1500.0;

        let mut trajectory = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let progress = i as f64 / n_samples as f64;

            // Linear interpolation towards schwa
            let f1 = params.initial_f1 + params.centralization_degree * progress * (schwa_f1 - params.initial_f1);
            let f2 = params.initial_f2 + params.centralization_degree * progress * (schwa_f2 - params.initial_f2);

            trajectory.push((f1, f2));
        }

        let final_f1 = trajectory.last().unwrap().0;
        let final_f2 = trajectory.last().unwrap().1;

        let mut gt_params = HashMap::new();
        gt_params.insert("initial_f1".to_string(), params.initial_f1);
        gt_params.insert("initial_f2".to_string(), params.initial_f2);
        gt_params.insert("final_f1".to_string(), final_f1);
        gt_params.insert("final_f2".to_string(), final_f2);
        gt_params.insert("centralization_degree".to_string(), params.centralization_degree);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(trajectory, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        VowelCentralizationParams {
            duration: 0.5,
            sampling_rate: 100.0,
            initial_f1: 730.0,  // /a/
            initial_f2: 1090.0,
            centralization_degree: 0.7,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.centralization_degree < 0.0 || params.centralization_degree > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("centralization_degree must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Formant transition generator (coarticulation)
pub struct FormantTransitionGenerator;

#[derive(Debug, Clone)]
pub struct FormantTransitionParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub start_formants: Vec<f64>, // F1, F2, F3
    pub end_formants: Vec<f64>,
    pub transition_rate: f64,   // 0-1 (1 = instantaneous, 0 = slow)
    pub smoothness: f64,        // 0-1 (1 = smooth, 0 = abrupt)
}

impl SyntheticGenerator for FormantTransitionGenerator {
    type Output = Vec<Vec<f64>>; // formant trajectories [time][formant]
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = FormantTransitionParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let num_formants = params.start_formants.len();

        let mut trajectory = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let progress = i as f64 / n_samples as f64;

            // Apply smoothness via sigmoid-like function
            let smooth_progress = if params.smoothness > 0.5 {
                // Smooth transition (sigmoid)
                let steepness = 10.0 * params.transition_rate;
                1.0 / (1.0 + (-steepness * (progress - 0.5)).exp())
            } else {
                // More abrupt transition
                progress.powf(2.0 - params.smoothness)
            };

            let mut formants = Vec::with_capacity(num_formants);
            for f in 0..num_formants {
                let start = params.start_formants[f];
                let end = params.end_formants[f];
                let formant_value = start + smooth_progress * (end - start);
                formants.push(formant_value);
            }

            trajectory.push(formants);
        }

        let mut gt_params = HashMap::new();
        gt_params.insert("transition_rate".to_string(), params.transition_rate);
        gt_params.insert("smoothness".to_string(), params.smoothness);
        for (i, &f) in params.start_formants.iter().enumerate() {
            gt_params.insert(format!("start_F{}", i + 1), f);
        }
        for (i, &f) in params.end_formants.iter().enumerate() {
            gt_params.insert(format!("end_F{}", i + 1), f);
        }

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(trajectory, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        FormantTransitionParams {
            duration: 0.15,
            sampling_rate: 1000.0,
            start_formants: vec![270.0, 2290.0, 3010.0],  // /i/
            end_formants: vec![730.0, 1090.0, 2440.0],    // /a/
            transition_rate: 0.7,
            smoothness: 0.8,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.start_formants.len() != params.end_formants.len() {
            return Err(crate::GeneratorError::InvalidParameter("start and end formants must have same length".to_string()));
        }
        if params.transition_rate < 0.0 || params.transition_rate > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("transition_rate must be 0-1".to_string()));
        }
        if params.smoothness < 0.0 || params.smoothness > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("smoothness must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Diadochokinesis generator (alternating motion rate: /pa/-/ta/-/ka/)
pub struct DiadochokinesisGenerator;

#[derive(Debug, Clone)]
pub struct DiadochokinesisParams {
    pub duration: f64,
    pub syllable_sequence: Vec<String>, // e.g., ["pa", "ta", "ka"]
    pub target_rate: f64,               // syllables per second (typical 5-7)
    pub regularity: f64,                // 0-1 (1 = perfectly regular)
}

impl SyntheticGenerator for DiadochokinesisGenerator {
    type Output = Vec<(f64, String)>; // (time, syllable)
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = DiadochokinesisParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let base_interval = 1.0 / params.target_rate;
        let variability = base_interval * (1.0 - params.regularity);
        let timing_dist = Normal::new(base_interval, variability).unwrap();

        let mut syllable_events = Vec::new();
        let mut events = Vec::new();
        let mut t = 0.0;
        let mut syllable_idx = 0;

        while t < params.duration {
            let syllable = &params.syllable_sequence[syllable_idx % params.syllable_sequence.len()];

            syllable_events.push((t, syllable.clone()));

            events.push(Event {
                time: t,
                event_type: "syllable".to_string(),
                amplitude: Some(syllable_idx as f64),
                attributes: HashMap::new(),
            });

            let interval = timing_dist.sample(&mut rng).max(base_interval * 0.3);
            t += interval;
            syllable_idx += 1;
        }

        // Calculate actual rate
        let actual_rate = syllable_events.len() as f64 / params.duration;

        // Calculate regularity (coefficient of variation)
        let intervals: Vec<f64> = syllable_events.windows(2)
            .map(|w| w[1].0 - w[0].0)
            .collect();
        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance = intervals.iter().map(|&i| (i - mean_interval).powi(2)).sum::<f64>() / intervals.len() as f64;
        let cv = variance.sqrt() / mean_interval;

        let mut gt_params = HashMap::new();
        gt_params.insert("target_rate".to_string(), params.target_rate);
        gt_params.insert("actual_rate".to_string(), actual_rate);
        gt_params.insert("target_regularity".to_string(), params.regularity);
        gt_params.insert("coefficient_of_variation".to_string(), cv);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(syllable_events, ground_truth, 1.0))
    }

    fn default_params() -> Self::Parameters {
        DiadochokinesisParams {
            duration: 5.0,
            syllable_sequence: vec!["pa".to_string(), "ta".to_string(), "ka".to_string()],
            target_rate: 6.0,
            regularity: 0.9,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.syllable_sequence.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("syllable_sequence cannot be empty".to_string()));
        }
        if params.target_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("target_rate must be positive".to_string()));
        }
        if params.regularity < 0.0 || params.regularity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("regularity must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Consonant precision generator (articulatory accuracy)
pub struct ConsonantPrecisionGenerator;

#[derive(Debug, Clone)]
pub struct ConsonantPrecisionParams {
    pub num_consonants: usize,
    pub precision: f64,       // 0-1 (1 = perfect, 0 = imprecise)
    pub consonant_type: ConsonantClass,
}

#[derive(Debug, Clone)]
pub enum ConsonantClass {
    Plosives,    // /p/, /t/, /k/
    Fricatives,  // /f/, /s/, /ʃ/
    Affricates,  // /tʃ/, /dʒ/
}

impl SyntheticGenerator for ConsonantPrecisionGenerator {
    type Output = Vec<ConsonantFeatures>; // acoustic features for each consonant
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = ConsonantPrecisionParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Define target features for consonant types
        let (target_burst_duration, target_frication_intensity) = match params.consonant_type {
            ConsonantClass::Plosives => (15.0, 0.0),      // short burst, no frication
            ConsonantClass::Fricatives => (0.0, 70.0),    // no burst, high frication
            ConsonantClass::Affricates => (10.0, 60.0),   // burst + frication
        };

        let imprecision = 1.0 - params.precision;
        let burst_noise = Normal::new(0.0, 5.0 * imprecision).unwrap();
        let frication_noise = Normal::new(0.0, 10.0 * imprecision).unwrap();

        let mut consonant_features = Vec::new();

        for _ in 0..params.num_consonants {
            let burst_duration = (target_burst_duration + burst_noise.sample(&mut rng)).max(0.0);
            let frication_intensity = (target_frication_intensity + frication_noise.sample(&mut rng)).max(0.0);

            // Closure duration also affected by precision
            let closure_duration = 50.0 + (1.0 - params.precision) * 30.0 * rng.random_range(-1.0..1.0);

            consonant_features.push(ConsonantFeatures {
                burst_duration,
                frication_intensity,
                closure_duration: closure_duration.max(20.0),
            });
        }

        // Calculate precision metrics
        let burst_cv = Self::coefficient_of_variation(&consonant_features.iter().map(|c| c.burst_duration).collect::<Vec<_>>());
        let frication_cv = Self::coefficient_of_variation(&consonant_features.iter().map(|c| c.frication_intensity).collect::<Vec<_>>());

        let mut gt_params = HashMap::new();
        gt_params.insert("target_precision".to_string(), params.precision);
        gt_params.insert("burst_cv".to_string(), burst_cv);
        gt_params.insert("frication_cv".to_string(), frication_cv);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(consonant_features, ground_truth, 1.0))
    }

    fn default_params() -> Self::Parameters {
        ConsonantPrecisionParams {
            num_consonants: 30,
            precision: 0.85,
            consonant_type: ConsonantClass::Plosives,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.num_consonants == 0 {
            return Err(crate::GeneratorError::InvalidParameter("num_consonants must be positive".to_string()));
        }
        if params.precision < 0.0 || params.precision > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("precision must be 0-1".to_string()));
        }
        Ok(())
    }
}

impl ConsonantPrecisionGenerator {
    fn coefficient_of_variation(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        if mean == 0.0 {
            return 0.0;
        }
        let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt() / mean
    }
}

#[derive(Debug, Clone)]
pub struct ConsonantFeatures {
    pub burst_duration: f64,      // ms
    pub frication_intensity: f64, // dB
    pub closure_duration: f64,    // ms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vowel_space_generation() {
        let generator = VowelSpaceGenerator;
        let params = VowelSpaceGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.num_vowels);
    }

    #[test]
    fn test_vowel_centralization() {
        let generator = VowelCentralizationGenerator;
        let params = VowelCentralizationGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_formant_transition() {
        let generator = FormantTransitionGenerator;
        let params = FormantTransitionGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_diadochokinesis() {
        let generator = DiadochokinesisGenerator;
        let params = DiadochokinesisGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());
    }

    #[test]
    fn test_consonant_precision() {
        let generator = ConsonantPrecisionGenerator;
        let params = ConsonantPrecisionGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.num_consonants);
    }
}
