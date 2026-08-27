//! Respiratory signal generators

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Respiratory waveform generator
pub struct RespiratoryWaveformGenerator;

#[derive(Debug, Clone)]
pub struct RespiratoryParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub respiratory_rate: f64, // breaths per minute
    pub amplitude: f64,
    pub inspiration_ratio: f64, // I:E ratio inspiration/(inspiration+expiration)
}

impl SyntheticGenerator for RespiratoryWaveformGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = RespiratoryParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let breath_duration = 60.0 / params.respiratory_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut signal = Vec::with_capacity(n_samples);
        let mut breath_onsets = Vec::new();

        for i in 0..n_samples {
            let t = i as f64 * dt;
            let phase = (t % breath_duration) / breath_duration;

            // Asymmetric breath cycle
            let sample = if phase < params.inspiration_ratio {
                // Inspiration (faster rise)
                let insp_phase = phase / params.inspiration_ratio;
                params.amplitude * (PI * insp_phase).sin()
            } else {
                // Expiration (slower decay)
                let exp_phase = (phase - params.inspiration_ratio) / (1.0 - params.inspiration_ratio);
                params.amplitude * (PI * (1.0 - exp_phase)).sin()
            };

            signal.push(sample + rng.random_range(-0.02..0.02));

            // Detect breath onsets
            if phase < 0.01 {
                breath_onsets.push(Event {
                    time: t,
                    event_type: "breath_onset".to_string(),
                    amplitude: None,
                    attributes: HashMap::new(),
                });
            }
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("respiratory_rate".to_string(), params.respiratory_rate);
        gt_params.insert("ie_ratio".to_string(), params.inspiration_ratio / (1.0 - params.inspiration_ratio));

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: breath_onsets,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        RespiratoryParams {
            duration: 60.0,
            sampling_rate: 50.0,
            respiratory_rate: 15.0,
            amplitude: 1.0,
            inspiration_ratio: 0.4, // I:E = 1:1.5
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.respiratory_rate <= 0.0 || params.respiratory_rate > 60.0 {
            return Err(crate::GeneratorError::InvalidParameter("respiratory_rate must be 0-60 bpm".to_string()));
        }
        if params.inspiration_ratio <= 0.0 || params.inspiration_ratio >= 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("inspiration_ratio must be 0-1".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_respiratory_generation() {
        let generator = RespiratoryWaveformGenerator;
        let params = RespiratoryWaveformGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }
}
