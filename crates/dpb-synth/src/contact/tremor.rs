//! Tremor signal generators

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth};
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Physiological tremor (8-12 Hz)
pub struct PhysiologicalTremorGenerator;

#[derive(Debug, Clone)]
pub struct PhysiologicalTremorParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub frequency: f64,        // Hz (8-12)
    pub amplitude: f64,        // typically small (0.1-0.5 deg)
    pub frequency_variability: f64, // Hz
}

impl SyntheticGenerator for PhysiologicalTremorGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PhysiologicalTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let freq_noise = Normal::new(0.0, params.frequency_variability).unwrap();
        let amp_noise = Normal::new(1.0, 0.1).unwrap(); // 10% amplitude variation

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 * dt;
                    let freq = params.frequency + freq_noise.sample(&mut rng);
                    let sample_val = amp_noise.sample(&mut rng);
                    let amp_mod = f64::max(sample_val, 0.0);
                    params.amplitude * amp_mod * (2.0 * PI * freq * t).sin()
                })
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("frequency".to_string(), params.frequency);
        gt_params.insert("amplitude".to_string(), params.amplitude);
        gt_params.insert("tremor_type".to_string(), 0.0); // physiological

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        PhysiologicalTremorParams {
            duration: 10.0,
            sampling_rate: 100.0,
            frequency: 10.0,
            amplitude: 0.3,
            frequency_variability: 0.5,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.frequency < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("frequency must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Parkinsonian rest tremor (4-6 Hz)
pub struct ParkinsonianTremorGenerator;

#[derive(Debug, Clone)]
pub struct ParkinsonianTremorParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub frequency: f64,        // Hz (4-6)
    pub amplitude: f64,
    pub pill_rolling: bool,    // alternating component
    pub amplitude_modulation: f64, // waxing/waning period (s)
}

impl SyntheticGenerator for ParkinsonianTremorGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = ParkinsonianTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 * dt;

                    // Primary tremor component
                    let primary = params.amplitude * (2.0 * PI * params.frequency * t).sin();

                    // Pill-rolling alternating component (if enabled)
                    let pill_roll = if params.pill_rolling {
                        0.3 * params.amplitude * (2.0 * PI * params.frequency * t + PI / 2.0).sin()
                    } else {
                        0.0
                    };

                    // Amplitude modulation (waxing/waning)
                    let modulation = if params.amplitude_modulation > 0.0 {
                        0.5 + 0.5 * (2.0 * PI * t / params.amplitude_modulation).sin()
                    } else {
                        1.0
                    };

                    (primary + pill_roll) * modulation
                })
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("frequency".to_string(), params.frequency);
        gt_params.insert("amplitude".to_string(), params.amplitude);
        gt_params.insert("tremor_type".to_string(), 1.0); // Parkinsonian
        gt_params.insert("pill_rolling".to_string(), if params.pill_rolling { 1.0 } else { 0.0 });

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        ParkinsonianTremorParams {
            duration: 10.0,
            sampling_rate: 100.0,
            frequency: 5.0,
            amplitude: 2.0,
            pill_rolling: true,
            amplitude_modulation: 3.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.frequency < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("frequency must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Essential tremor (4-12 Hz, postural)
pub struct EssentialTremorGenerator;

#[derive(Debug, Clone)]
pub struct EssentialTremorParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub frequency: f64,        // Hz (4-12, typically 6-8)
    pub amplitude: f64,
    pub harmonics: Vec<f64>,   // harmonic amplitudes
}

impl SyntheticGenerator for EssentialTremorGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = EssentialTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let amp_noise = Normal::new(1.0, 0.15).unwrap();

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 * dt;
                    let sample_val = amp_noise.sample(&mut rng);
                    let amp_mod = f64::max(sample_val, 0.0);

                    // Fundamental frequency
                    let mut sample = params.amplitude * amp_mod * (2.0 * PI * params.frequency * t).sin();

                    // Add harmonics
                    for (harm_idx, &harm_amp) in params.harmonics.iter().enumerate() {
                        let harm_freq = params.frequency * (harm_idx + 2) as f64;
                        sample += params.amplitude * harm_amp * (2.0 * PI * harm_freq * t).sin();
                    }

                    sample
                })
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("frequency".to_string(), params.frequency);
        gt_params.insert("amplitude".to_string(), params.amplitude);
        gt_params.insert("tremor_type".to_string(), 2.0); // Essential

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        EssentialTremorParams {
            duration: 10.0,
            sampling_rate: 100.0,
            frequency: 7.0,
            amplitude: 3.0,
            harmonics: vec![0.3, 0.1], // 2nd and 3rd harmonics
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.frequency < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("frequency must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Cerebellar intention tremor (3-5 Hz)
pub struct CerebellarTremorGenerator;

#[derive(Debug, Clone)]
pub struct CerebellarTremorParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub frequency: f64,        // Hz (3-5)
    pub baseline_amplitude: f64,
    pub movement_times: Vec<(f64, f64)>, // (start, end) of intended movements
    pub amplitude_increase: f64, // fold increase during movement
}

impl SyntheticGenerator for CerebellarTremorGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = CerebellarTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 * dt;

                    // Check if during movement
                    let mut in_movement = false;
                    for (start, end) in &params.movement_times {
                        if t >= *start && t <= *end {
                            in_movement = true;
                            break;
                        }
                    }

                    let amplitude = if in_movement {
                        params.baseline_amplitude * params.amplitude_increase
                    } else {
                        params.baseline_amplitude
                    };

                    // Low-frequency, high-amplitude tremor
                    amplitude * (2.0 * PI * params.frequency * t + rng.r#gen_range(0.0..0.2)).sin()
                })
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("frequency".to_string(), params.frequency);
        gt_params.insert("baseline_amplitude".to_string(), params.baseline_amplitude);
        gt_params.insert("tremor_type".to_string(), 3.0); // Cerebellar

        let segments = params.movement_times
            .iter()
            .map(|(start, end)| crate::traits::Segment {
                start: *start,
                end: *end,
                label: "intention_movement".to_string(),
            })
            .collect();

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments,
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        CerebellarTremorParams {
            duration: 20.0,
            sampling_rate: 100.0,
            frequency: 4.0,
            baseline_amplitude: 0.5,
            movement_times: vec![(3.0, 5.0), (10.0, 12.0), (17.0, 19.0)],
            amplitude_increase: 5.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.frequency < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("frequency must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Tremor modulation generator (amplitude waxing/waning)
pub struct TremorModulationGenerator;

#[derive(Debug, Clone)]
pub struct TremorModulationParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub carrier_frequency: f64,     // Hz (tremor frequency)
    pub carrier_amplitude: f64,
    pub modulation_frequency: f64,  // Hz (modulation)
    pub modulation_depth: f64,      // 0-1
}

impl SyntheticGenerator for TremorModulationGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = TremorModulationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 * dt;

                    // AM modulation: A(1 + m*sin(2πf_m*t)) * sin(2πf_c*t)
                    let modulation = 1.0 + params.modulation_depth *
                        (2.0 * PI * params.modulation_frequency * t).sin();
                    let carrier = (2.0 * PI * params.carrier_frequency * t).sin();

                    params.carrier_amplitude * modulation * carrier
                })
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("carrier_frequency".to_string(), params.carrier_frequency);
        gt_params.insert("modulation_frequency".to_string(), params.modulation_frequency);
        gt_params.insert("modulation_depth".to_string(), params.modulation_depth);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        TremorModulationParams {
            duration: 10.0,
            sampling_rate: 100.0,
            carrier_frequency: 5.0,
            carrier_amplitude: 2.0,
            modulation_frequency: 0.2,
            modulation_depth: 0.8,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.modulation_depth < 0.0 || params.modulation_depth > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("modulation_depth must be 0-1".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physiological_tremor() {
        let generator = PhysiologicalTremorGenerator;
        let params = PhysiologicalTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_parkinsonian_tremor() {
        let generator = ParkinsonianTremorGenerator;
        let params = ParkinsonianTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_cerebellar_tremor() {
        let generator = CerebellarTremorGenerator;
        let params = CerebellarTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.ground_truth.segments.is_empty());
    }
}
