//! Thermal (skin temperature) signal generators

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Skin temperature generator with vasomotor oscillations
pub struct SkinTemperatureGenerator;

#[derive(Debug, Clone)]
pub struct SkinTemperatureParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_temp: f64,          // degrees Celsius (typically 32-34°C)
    pub circadian_amplitude: f64,    // °C (daily variation)
    pub circadian_phase: f64,        // radians (time of day offset)
    pub vasomotor_amplitude: f64,    // °C (blood flow oscillations)
    pub vasomotor_frequency: f64,    // Hz (typically 0.01-0.1 Hz)
    pub measurement_noise: f64,      // °C (sensor noise)
}

impl SyntheticGenerator for SkinTemperatureGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = SkinTemperatureParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise_dist = Normal::new(0.0, params.measurement_noise).unwrap();
        let vasomotor_phase = rng.random_range(0.0..2.0 * PI);

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 * dt;

                    // Circadian rhythm (24-hour cycle)
                    // Using params.duration as scaled time for demonstration
                    let circadian_t = 2.0 * PI * (t / 86400.0); // 86400 sec = 24 hours
                    let circadian = params.circadian_amplitude * (circadian_t + params.circadian_phase).sin();

                    // Vasomotor oscillations (blood flow regulation)
                    let vasomotor = params.vasomotor_amplitude *
                        (2.0 * PI * params.vasomotor_frequency * t + vasomotor_phase).sin();

                    // Measurement noise
                    let noise = noise_dist.sample(&mut rng);

                    params.baseline_temp + circadian + vasomotor + noise
                })
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_temp".to_string(), params.baseline_temp);
        gt_params.insert("circadian_amplitude".to_string(), params.circadian_amplitude);
        gt_params.insert("vasomotor_amplitude".to_string(), params.vasomotor_amplitude);
        gt_params.insert("vasomotor_frequency".to_string(), params.vasomotor_frequency);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        SkinTemperatureParams {
            duration: 600.0,        // 10 minutes
            sampling_rate: 1.0,     // 1 Hz (thermal signals are slow)
            baseline_temp: 33.0,    // °C
            circadian_amplitude: 0.5,
            circadian_phase: 0.0,
            vasomotor_amplitude: 0.2,
            vasomotor_frequency: 0.05, // 0.05 Hz = 20 second period
            measurement_noise: 0.05,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.baseline_temp < 20.0 || params.baseline_temp > 40.0 {
            return Err(crate::GeneratorError::InvalidParameter("baseline_temp must be 20-40°C".to_string()));
        }
        Ok(())
    }
}

/// Thermal response generator (stimulus-evoked temperature changes)
pub struct ThermalResponseGenerator;

#[derive(Debug, Clone)]
pub struct ThermalResponseParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_temp: f64,          // °C
    pub stimulus_times: Vec<f64>,    // seconds
    pub response_amplitude: f64,     // °C (temperature change)
    pub response_latency: f64,       // seconds (delay to peak)
    pub recovery_tau: f64,           // seconds (time constant for recovery)
    pub measurement_noise: f64,      // °C
}

impl SyntheticGenerator for ThermalResponseGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = ThermalResponseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let noise_dist = Normal::new(0.0, params.measurement_noise).unwrap();

        let mut signal = vec![params.baseline_temp; n_samples];
        let mut events = Vec::new();

        // Generate thermal responses for each stimulus
        for &stim_time in &params.stimulus_times {
            events.push(Event {
                time: stim_time,
                event_type: "thermal_stimulus".to_string(),
                amplitude: Some(params.response_amplitude),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("latency".to_string(), params.response_latency);
                    attrs.insert("recovery_tau".to_string(), params.recovery_tau);
                    attrs
                },
            });

            // Generate response waveform
            for i in 0..n_samples {
                let t = i as f64 * dt;

                if t >= stim_time {
                    let delta_t = t - stim_time;

                    // Gamma-like response: rise to peak then exponential decay
                    let response = if delta_t < params.response_latency {
                        // Rising phase (quadratic rise)
                        let progress = delta_t / params.response_latency;
                        params.response_amplitude * progress * progress
                    } else {
                        // Recovery phase (exponential decay from peak)
                        let decay_t = delta_t - params.response_latency;
                        params.response_amplitude * (-decay_t / params.recovery_tau).exp()
                    };

                    signal[i] += response;
                }
            }
        }

        // Add measurement noise
        for sample in signal.iter_mut() {
            *sample += noise_dist.sample(&mut rng);
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_temp".to_string(), params.baseline_temp);
        gt_params.insert("response_amplitude".to_string(), params.response_amplitude);
        gt_params.insert("response_latency".to_string(), params.response_latency);
        gt_params.insert("recovery_tau".to_string(), params.recovery_tau);
        gt_params.insert("num_stimuli".to_string(), params.stimulus_times.len() as f64);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        ThermalResponseParams {
            duration: 300.0,        // 5 minutes
            sampling_rate: 1.0,     // 1 Hz
            baseline_temp: 33.0,
            stimulus_times: vec![30.0, 90.0, 150.0, 210.0, 270.0],
            response_amplitude: 1.0,  // 1°C increase
            response_latency: 5.0,    // 5 seconds to peak
            recovery_tau: 20.0,       // 20 second recovery
            measurement_noise: 0.05,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.response_latency <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("response_latency must be positive".to_string()));
        }
        if params.recovery_tau <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("recovery_tau must be positive".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skin_temperature_generation() {
        let generator = SkinTemperatureGenerator;
        let params = SkinTemperatureGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);

        // Check temperature is in reasonable range
        let mean_temp = result.signal.mean().unwrap();
        assert!((mean_temp - params.baseline_temp).abs() < 2.0);
    }

    #[test]
    fn test_thermal_response_generation() {
        let generator = ThermalResponseGenerator;
        let params = ThermalResponseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
        assert_eq!(result.ground_truth.events.len(), params.stimulus_times.len());
    }
}
