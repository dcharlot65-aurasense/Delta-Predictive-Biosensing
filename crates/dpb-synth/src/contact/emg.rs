//! EMG (electromyography) signal generators

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Surface EMG generator
pub struct SurfaceEmgGenerator;

#[derive(Debug, Clone)]
pub struct SurfaceEmgParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_amplitude: f64,  // baseline noise
    pub contraction_level: f64,   // 0-1 (% of MVC)
}

impl SyntheticGenerator for SurfaceEmgGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = SurfaceEmgParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // EMG is modeled as filtered white noise with amplitude proportional to contraction
        let amplitude = params.baseline_amplitude +
            params.contraction_level * params.baseline_amplitude * 10.0;

        let noise_dist = Normal::new(0.0, amplitude).unwrap();

        // Generate raw EMG signal
        let mut signal: Vec<f64> = (0..n_samples)
            .map(|_| noise_dist.sample(&mut rng))
            .collect();

        // Add MUAP (motor unit action potential) spikes during contraction
        if params.contraction_level > 0.1 {
            let muap_rate = params.contraction_level * 50.0; // spikes per second
            let muap_interval = params.sampling_rate / muap_rate;

            let mut next_muap = rng.random_range(0.0..muap_interval);
            while (next_muap as usize) < n_samples {
                let idx = next_muap as usize;
                // Add biphasic spike
                if idx < n_samples - 5 {
                    signal[idx] += amplitude * 3.0;
                    signal[idx + 1] += amplitude * 5.0;
                    signal[idx + 2] += amplitude * 2.0;
                    signal[idx + 3] -= amplitude * 2.0;
                    signal[idx + 4] -= amplitude * 3.0;
                }
                next_muap += muap_interval + rng.random_range(-muap_interval * 0.3..muap_interval * 0.3);
            }
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("contraction_level".to_string(), params.contraction_level);
        gt_params.insert("baseline_amplitude".to_string(), params.baseline_amplitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        SurfaceEmgParams {
            duration: 10.0,
            sampling_rate: 2000.0,
            baseline_amplitude: 0.01,
            contraction_level: 0.3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate < 500.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate should be >= 500 Hz for EMG".to_string()));
        }
        if params.contraction_level < 0.0 || params.contraction_level > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("contraction_level must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Voluntary contraction generator with fatigue
pub struct VoluntaryContractionGenerator;

#[derive(Debug, Clone)]
pub struct VoluntaryContractionParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_amplitude: f64,
    pub contraction_segments: Vec<(f64, f64, f64)>, // (start, end, MVC%)
    pub fatigue_rate: f64,                          // % decrease per second
}

impl SyntheticGenerator for VoluntaryContractionGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = VoluntaryContractionParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut signal = Vec::with_capacity(n_samples);
        let mut events = Vec::new();

        for i in 0..n_samples {
            let t = i as f64 * dt;

            // Determine current contraction level
            let mut contraction_level = 0.0;
            let mut contraction_duration = 0.0;

            for (start, end, mvc) in &params.contraction_segments {
                if t >= *start && t <= *end {
                    contraction_level = *mvc / 100.0;
                    contraction_duration = t - start;

                    // Add contraction onset event
                    if (t - start).abs() < dt {
                        events.push(Event {
                            time: *start,
                            event_type: "contraction_onset".to_string(),
                            amplitude: Some(*mvc),
                            attributes: HashMap::new(),
                        });
                    }
                    break;
                }
            }

            // Apply fatigue
            if contraction_level > 0.0 {
                let fatigue_factor = 1.0 - (params.fatigue_rate / 100.0 * contraction_duration);
                contraction_level *= fatigue_factor.max(0.2); // minimum 20% of initial
            }

            // Generate EMG signal
            let amplitude = params.baseline_amplitude +
                contraction_level * params.baseline_amplitude * 10.0;
            let sample = amplitude * rng.random_range(-1.0..1.0);

            signal.push(sample);
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("fatigue_rate".to_string(), params.fatigue_rate);

        let segments = params.contraction_segments
            .iter()
            .map(|(start, end, mvc)| crate::traits::Segment {
                start: *start,
                end: *end,
                label: format!("contraction_{}%MVC", mvc),
            })
            .collect();

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments,
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        VoluntaryContractionParams {
            duration: 30.0,
            sampling_rate: 2000.0,
            baseline_amplitude: 0.01,
            contraction_segments: vec![
                (5.0, 10.0, 50.0),
                (15.0, 25.0, 80.0),
            ],
            fatigue_rate: 5.0, // 5% per second
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate < 500.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate should be >= 500 Hz".to_string()));
        }
        Ok(())
    }
}

/// Pathological EMG generator (fasciculations, fibrillations)
pub struct PathologicalEmgGenerator;

#[derive(Debug, Clone)]
pub enum PathologyType {
    Fasciculation, // spontaneous motor unit discharge
    Fibrillation,  // single muscle fiber activity
    Myotonia,      // delayed relaxation
}

#[derive(Debug, Clone)]
pub struct PathologicalEmgParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_amplitude: f64,
    pub pathology_type: PathologyType,
    pub event_rate: f64, // events per second
}

impl SyntheticGenerator for PathologicalEmgGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PathologicalEmgParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Baseline noise
        let noise_dist = Normal::new(0.0, params.baseline_amplitude).unwrap();
        let mut signal: Vec<f64> = (0..n_samples)
            .map(|_| noise_dist.sample(&mut rng))
            .collect();

        let mut events = Vec::new();

        match params.pathology_type {
            PathologyType::Fasciculation => {
                // Large amplitude, irregular spontaneous discharges
                let event_interval = params.sampling_rate / params.event_rate;
                let mut next_event = rng.random_range(0.0..event_interval);

                while (next_event as usize) < n_samples {
                    let idx = next_event as usize;
                    let t = idx as f64 * dt;

                    events.push(Event {
                        time: t,
                        event_type: "fasciculation".to_string(),
                        amplitude: Some(params.baseline_amplitude * 20.0),
                        attributes: HashMap::new(),
                    });

                    // Large triphasic spike
                    if idx < n_samples - 10 {
                        for j in 0..10 {
                            let phase = j as f64 / 10.0 * 2.0 * PI;
                            signal[idx + j] += params.baseline_amplitude * 20.0 * phase.sin();
                        }
                    }

                    next_event += event_interval + rng.random_range(-event_interval * 0.5..event_interval * 0.5);
                }
            }
            PathologyType::Fibrillation => {
                // Small amplitude, high frequency potentials
                let event_interval = params.sampling_rate / params.event_rate;
                let mut next_event = rng.random_range(0.0..event_interval);

                while (next_event as usize) < n_samples {
                    let idx = next_event as usize;
                    let t = idx as f64 * dt;

                    events.push(Event {
                        time: t,
                        event_type: "fibrillation".to_string(),
                        amplitude: Some(params.baseline_amplitude * 3.0),
                        attributes: HashMap::new(),
                    });

                    // Small biphasic spike
                    if idx < n_samples - 3 {
                        signal[idx] += params.baseline_amplitude * 3.0;
                        signal[idx + 1] += params.baseline_amplitude * 4.0;
                        signal[idx + 2] -= params.baseline_amplitude * 3.0;
                    }

                    next_event += event_interval;
                }
            }
            PathologyType::Myotonia => {
                // Prolonged burst after contraction
                // Simulate one myotonic burst in the middle
                let burst_start = n_samples / 2;
                let burst_duration = (2.0 * params.sampling_rate) as usize; // 2 seconds

                events.push(Event {
                    time: burst_start as f64 * dt,
                    event_type: "myotonic_burst".to_string(),
                    amplitude: None,
                    attributes: HashMap::new(),
                });

                for i in burst_start..std::cmp::min(burst_start + burst_duration, n_samples) {
                    let burst_phase = (i - burst_start) as f64 / burst_duration as f64;
                    let amplitude = params.baseline_amplitude * 10.0 * (1.0 - burst_phase); // decay
                    signal[i] += amplitude * rng.random_range(-1.0..1.0);
                }
            }
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("event_rate".to_string(), params.event_rate);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        PathologicalEmgParams {
            duration: 10.0,
            sampling_rate: 2000.0,
            baseline_amplitude: 0.01,
            pathology_type: PathologyType::Fasciculation,
            event_rate: 2.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate < 500.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate should be >= 500 Hz".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_emg_generation() {
        let generator = SurfaceEmgGenerator;
        let params = SurfaceEmgGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_voluntary_contraction() {
        let generator = VoluntaryContractionGenerator;
        let params = VoluntaryContractionGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.ground_truth.segments.is_empty());
    }

    #[test]
    fn test_pathological_emg() {
        let generator = PathologicalEmgGenerator;
        let params = PathologicalEmgGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.ground_truth.events.is_empty());
    }
}
