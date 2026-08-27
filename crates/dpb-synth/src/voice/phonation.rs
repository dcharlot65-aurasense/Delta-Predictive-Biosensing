//! Phonation generators (source-filter model)

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth};
use ndarray::Array1;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Sustained vowel generator using source-filter model
pub struct SustainedVowelGenerator;

#[derive(Debug, Clone)]
pub struct SustainedVowelParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub fundamental_frequency: f64, // Hz (f0, typical 85-180 for male, 165-255 for female)
    pub vowel_formants: Vec<(f64, f64)>, // (frequency, bandwidth) for F1, F2, F3
    pub amplitude: f64,
}

impl SyntheticGenerator for SustainedVowelGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = SustainedVowelParams;

    fn generate(&self, params: &Self::Parameters, _seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;

        // Generate glottal source (pulse train with harmonics)
        let mut signal = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f64 * dt;

            // Glottal pulse train (sum of harmonics)
            let mut source = 0.0;
            for harmonic in 1..20 {
                let h_freq = params.fundamental_frequency * harmonic as f64;
                if h_freq < params.sampling_rate / 2.0 {
                    // Roll-off with harmonic number
                    let amplitude = 1.0 / harmonic as f64;
                    source += amplitude * (2.0 * PI * h_freq * t).sin();
                }
            }

            // Apply formant filtering (resonances)
            let mut filtered = 0.0;
            for (formant_freq, bandwidth) in &params.vowel_formants {
                // Simple resonance filter
                let q = formant_freq / bandwidth;
                let resonance = 1.0 / (1.0 + ((2.0 * PI * formant_freq * t).sin() / q).powi(2)).sqrt();
                filtered += source * resonance;
            }

            signal.push(params.amplitude * filtered / params.vowel_formants.len() as f64);
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("f0".to_string(), params.fundamental_frequency);
        for (i, (freq, bw)) in params.vowel_formants.iter().enumerate() {
            gt_params.insert(format!("F{}_freq", i + 1), *freq);
            gt_params.insert(format!("F{}_bw", i + 1), *bw);
        }

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        // Formants for vowel /a/ (typical values)
        SustainedVowelParams {
            duration: 3.0,
            sampling_rate: 16000.0,
            fundamental_frequency: 120.0,
            vowel_formants: vec![
                (730.0, 100.0),  // F1
                (1090.0, 150.0), // F2
                (2440.0, 200.0), // F3
            ],
            amplitude: 0.5,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate < 8000.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate should be >= 8000 Hz for voice".to_string()));
        }
        if params.fundamental_frequency <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("fundamental_frequency must be positive".to_string()));
        }
        Ok(())
    }
}

/// Jitter generator (f0 perturbation)
pub struct JitterGenerator;

#[derive(Debug, Clone)]
pub struct JitterParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_f0: f64,
    pub jitter_percent: f64, // % (typically 0.5-1% normal, >1% pathological)
}

impl SyntheticGenerator for JitterGenerator {
    type Output = Vec<f64>; // period-to-period f0 values
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = JitterParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Calculate number of periods
        let num_periods = (params.duration * params.baseline_f0) as usize;

        // Generate f0 values with jitter
        let jitter_std = params.baseline_f0 * (params.jitter_percent / 100.0);
        let jitter_dist = Normal::new(params.baseline_f0, jitter_std).unwrap();

        let f0_values: Vec<f64> = (0..num_periods)
            .map(|_| jitter_dist.sample(&mut rng).max(params.baseline_f0 * 0.5))
            .collect();

        // Calculate actual jitter
        let mut period_diffs = Vec::new();
        for i in 1..f0_values.len() {
            let period_i = 1.0 / f0_values[i];
            let period_prev = 1.0 / f0_values[i - 1];
            period_diffs.push(((period_i - period_prev) / period_prev).abs());
        }
        let actual_jitter = period_diffs.iter().sum::<f64>() / period_diffs.len() as f64 * 100.0;

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_f0".to_string(), params.baseline_f0);
        gt_params.insert("target_jitter_percent".to_string(), params.jitter_percent);
        gt_params.insert("actual_jitter_percent".to_string(), actual_jitter);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(f0_values, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        JitterParams {
            duration: 3.0,
            sampling_rate: 16000.0,
            baseline_f0: 120.0,
            jitter_percent: 0.8,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.jitter_percent < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("jitter_percent must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Shimmer generator (amplitude perturbation)
pub struct ShimmerGenerator;

#[derive(Debug, Clone)]
pub struct ShimmerParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_amplitude: f64,
    pub shimmer_percent: f64, // % (typically 2-3% normal, >3.5% pathological)
}

impl SyntheticGenerator for ShimmerGenerator {
    type Output = Vec<f64>; // period-to-period amplitude values
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = ShimmerParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Approximate number of periods (using typical f0 of 120 Hz)
        let num_periods = (params.duration * 120.0) as usize;

        // Generate amplitude values with shimmer
        let shimmer_std = params.baseline_amplitude * (params.shimmer_percent / 100.0);
        let shimmer_dist = Normal::new(params.baseline_amplitude, shimmer_std).unwrap();

        let amplitude_values: Vec<f64> = (0..num_periods)
            .map(|_| shimmer_dist.sample(&mut rng).max(0.1))
            .collect();

        // Calculate actual shimmer
        let mut amp_diffs = Vec::new();
        for i in 1..amplitude_values.len() {
            amp_diffs.push(((amplitude_values[i] - amplitude_values[i - 1]) / amplitude_values[i - 1]).abs());
        }
        let actual_shimmer = amp_diffs.iter().sum::<f64>() / amp_diffs.len() as f64 * 100.0;

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_amplitude".to_string(), params.baseline_amplitude);
        gt_params.insert("target_shimmer_percent".to_string(), params.shimmer_percent);
        gt_params.insert("actual_shimmer_percent".to_string(), actual_shimmer);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(amplitude_values, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        ShimmerParams {
            duration: 3.0,
            sampling_rate: 16000.0,
            baseline_amplitude: 1.0,
            shimmer_percent: 2.5,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.shimmer_percent < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("shimmer_percent must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Voice tremor generator
pub struct VoiceTremorGenerator;

#[derive(Debug, Clone)]
pub struct VoiceTremorParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_f0: f64,
    pub tremor_frequency: f64, // Hz (typically 4-8 Hz)
    pub tremor_extent: f64,    // Hz (f0 modulation depth)
}

impl SyntheticGenerator for VoiceTremorGenerator {
    type Output = Array1<f64>; // f0 contour over time
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = VoiceTremorParams;

    fn generate(&self, params: &Self::Parameters, _seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;

        let f0_contour: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;
                params.baseline_f0 + params.tremor_extent *
                    (2.0 * PI * params.tremor_frequency * t).sin()
            })
            .collect();

        let f0_contour = Array1::from_vec(f0_contour);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_f0".to_string(), params.baseline_f0);
        gt_params.insert("tremor_frequency".to_string(), params.tremor_frequency);
        gt_params.insert("tremor_extent".to_string(), params.tremor_extent);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(f0_contour, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        VoiceTremorParams {
            duration: 3.0,
            sampling_rate: 16000.0,
            baseline_f0: 120.0,
            tremor_frequency: 5.0,
            tremor_extent: 10.0,
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

/// Voice onset time (VOT) generator
pub struct VoiceOnsetTimeGenerator;

#[derive(Debug, Clone)]
pub struct VoiceOnsetTimeParams {
    pub num_samples: usize,
    pub consonant_type: ConsonantType,
    pub variability: f64, // 0-1
}

#[derive(Debug, Clone)]
pub enum ConsonantType {
    VoicedStop,      // VOT: -100 to 0 ms (e.g., /b/, /d/, /g/)
    VoicelessStop,   // VOT: 25-100 ms (e.g., /p/, /t/, /k/)
    AspiratedStop,   // VOT: >100 ms
}

impl SyntheticGenerator for VoiceOnsetTimeGenerator {
    type Output = Vec<f64>; // VOT values in milliseconds
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = VoiceOnsetTimeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let (mean_vot, std_vot) = match params.consonant_type {
            ConsonantType::VoicedStop => (-50.0, 20.0),
            ConsonantType::VoicelessStop => (60.0, 15.0),
            ConsonantType::AspiratedStop => (120.0, 20.0),
        };

        let std_dev = std_vot * params.variability;
        let vot_dist = Normal::new(mean_vot, std_dev).unwrap();

        let vot_values: Vec<f64> = (0..params.num_samples)
            .map(|_| vot_dist.sample(&mut rng))
            .collect();

        let mut gt_params = HashMap::new();
        gt_params.insert("mean_vot".to_string(), mean_vot);
        gt_params.insert("std_vot".to_string(), std_dev);
        gt_params.insert("actual_mean".to_string(),
            vot_values.iter().sum::<f64>() / vot_values.len() as f64);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(vot_values, ground_truth, 1000.0))
    }

    fn default_params() -> Self::Parameters {
        VoiceOnsetTimeParams {
            num_samples: 50,
            consonant_type: ConsonantType::VoicelessStop,
            variability: 1.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.num_samples == 0 {
            return Err(crate::GeneratorError::InvalidParameter("num_samples must be positive".to_string()));
        }
        if params.variability < 0.0 || params.variability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("variability must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Vocal fry (creaky voice) generator
pub struct VocalFryGenerator;

#[derive(Debug, Clone)]
pub struct VocalFryParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_f0: f64,
    pub fry_f0: f64,           // Hz (typically 20-60 Hz)
    pub fry_irregularity: f64, // 0-1 (period-to-period variation)
}

impl SyntheticGenerator for VocalFryGenerator {
    type Output = Array1<f64>; // audio signal with vocal fry
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = VocalFryParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;

        let irregularity_dist = Normal::new(0.0, params.fry_irregularity).unwrap();

        let mut signal = Vec::with_capacity(n_samples);
        let mut phase = 0.0;

        for i in 0..n_samples {
            let _t = i as f64 * dt;

            // Vocal fry has irregular pulses
            let period_jitter = irregularity_dist.sample(&mut rng);
            let instantaneous_f0 = params.fry_f0 * (1.0 + period_jitter);

            phase += 2.0 * PI * instantaneous_f0 * dt;
            if phase > 2.0 * PI {
                phase -= 2.0 * PI;
            }

            // Create asymmetric pulse characteristic of vocal fry
            let pulse = if phase < PI {
                (phase / PI).powi(3)
            } else {
                -((2.0 * PI - phase) / PI).powi(3)
            };

            signal.push(pulse);
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_f0".to_string(), params.baseline_f0);
        gt_params.insert("fry_f0".to_string(), params.fry_f0);
        gt_params.insert("fry_irregularity".to_string(), params.fry_irregularity);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        VocalFryParams {
            duration: 2.0,
            sampling_rate: 16000.0,
            baseline_f0: 120.0,
            fry_f0: 40.0,
            fry_irregularity: 0.3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.fry_f0 <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("fry_f0 must be positive".to_string()));
        }
        if params.fry_irregularity < 0.0 || params.fry_irregularity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("fry_irregularity must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Breathiness (aspiration noise) generator
pub struct BreathinessGenerator;

#[derive(Debug, Clone)]
pub struct BreathinessParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub harmonic_amplitude: f64,    // amplitude of harmonic component
    pub noise_amplitude: f64,       // amplitude of noise component
    pub fundamental_frequency: f64,
}

impl SyntheticGenerator for BreathinessGenerator {
    type Output = Array1<f64>; // audio signal with breathiness
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = BreathinessParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;

        let noise_dist = Normal::new(0.0, 1.0).unwrap();

        let mut signal = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t = i as f64 * dt;

            // Harmonic component (voiced)
            let harmonic = params.harmonic_amplitude * (2.0 * PI * params.fundamental_frequency * t).sin();

            // Noise component (aspiration)
            let noise = params.noise_amplitude * noise_dist.sample(&mut rng);

            // Breathy voice is a mix of harmonic and noise
            signal.push(harmonic + noise);
        }

        let signal = Array1::from_vec(signal);

        // Calculate harmonic-to-noise ratio (HNR)
        let hnr_db = 20.0 * (params.harmonic_amplitude / params.noise_amplitude).log10();

        let mut gt_params = HashMap::new();
        gt_params.insert("f0".to_string(), params.fundamental_frequency);
        gt_params.insert("harmonic_amplitude".to_string(), params.harmonic_amplitude);
        gt_params.insert("noise_amplitude".to_string(), params.noise_amplitude);
        gt_params.insert("hnr_db".to_string(), hnr_db);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        BreathinessParams {
            duration: 2.0,
            sampling_rate: 16000.0,
            harmonic_amplitude: 0.7,
            noise_amplitude: 0.3,
            fundamental_frequency: 120.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate < 8000.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate should be >= 8000 Hz".to_string()));
        }
        if params.fundamental_frequency <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("fundamental_frequency must be positive".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sustained_vowel_generation() {
        let generator = SustainedVowelGenerator;
        let params = SustainedVowelGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_jitter_generation() {
        let generator = JitterGenerator;
        let params = JitterGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());
    }

    #[test]
    fn test_voice_tremor() {
        let generator = VoiceTremorGenerator;
        let params = VoiceTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_vot_generation() {
        let generator = VoiceOnsetTimeGenerator;
        let params = VoiceOnsetTimeGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.num_samples);
    }

    #[test]
    fn test_vocal_fry_generation() {
        let generator = VocalFryGenerator;
        let params = VocalFryGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_breathiness_generation() {
        let generator = BreathinessGenerator;
        let params = BreathinessGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }
}
