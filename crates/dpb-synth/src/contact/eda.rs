//! EDA (electrodermal activity) / GSR generators

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal, Exp};
use std::collections::HashMap;

/// Tonic EDA (skin conductance level - SCL) generator
pub struct EdaTonicGenerator;

#[derive(Debug, Clone)]
pub struct EdaTonicParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_scl: f64,    // microsiemens
    pub drift_magnitude: f64,  // slow drift
    pub drift_frequency: f64,  // Hz (very low, ~0.01)
}

impl SyntheticGenerator for EdaTonicGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = EdaTonicParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let phase = rng.random_range(0.0..2.0 * std::f64::consts::PI);
        let noise = Normal::new(0.0, 0.01).unwrap();

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 * dt;
                    let drift = params.drift_magnitude *
                        (2.0 * std::f64::consts::PI * params.drift_frequency * t + phase).sin();
                    params.baseline_scl + drift + noise.sample(&mut rng)
                })
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_scl".to_string(), params.baseline_scl);
        gt_params.insert("drift_magnitude".to_string(), params.drift_magnitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        EdaTonicParams {
            duration: 60.0,
            sampling_rate: 10.0,
            baseline_scl: 5.0,     // microsiemens
            drift_magnitude: 0.5,
            drift_frequency: 0.01,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.baseline_scl < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("baseline_scl must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// SCR (skin conductance response) event generator using Bateman function
pub struct ScrEventGenerator;

#[derive(Debug, Clone)]
pub struct ScrEventParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub event_rate: f64,       // events per minute
    pub amplitude_mean: f64,   // microsiemens
    pub amplitude_std: f64,
    pub rise_time: f64,        // seconds (tau1, typically 1-3s)
    pub recovery_time: f64,    // seconds (tau2, typically 3-10s)
}

impl SyntheticGenerator for ScrEventGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = ScrEventParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate SCR event times (Poisson process)
        let exp_dist = Exp::new(params.event_rate / 60.0).unwrap();
        let mut event_times = Vec::new();
        let mut t = exp_dist.sample(&mut rng);
        while t < params.duration {
            event_times.push(t);
            t += exp_dist.sample(&mut rng);
        }

        // Generate amplitudes
        let amp_dist = Normal::new(params.amplitude_mean, params.amplitude_std).unwrap();
        let amplitudes: Vec<f64> = event_times
            .iter()
            .map(|_| amp_dist.sample(&mut rng).max(0.0))
            .collect();

        // Generate signal using Bateman function for each SCR
        let mut signal = vec![0.0; n_samples];
        let mut events = Vec::new();

        for (event_time, amplitude) in event_times.iter().zip(amplitudes.iter()) {
            events.push(Event {
                time: *event_time,
                event_type: "SCR".to_string(),
                amplitude: Some(*amplitude),
                attributes: HashMap::new(),
            });

            for (i, i_slot) in signal.iter_mut().enumerate() {
                let t = i as f64 * dt;
                if t >= *event_time {
                    let delta_t = t - event_time;
                    // Bateman function: A * (exp(-t/tau2) - exp(-t/tau1))
                    let scr = amplitude *
                        ((-delta_t / params.recovery_time).exp() -
                         (-delta_t / params.rise_time).exp());
                    *i_slot += scr;
                }
            }
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("event_rate".to_string(), params.event_rate);
        gt_params.insert("amplitude_mean".to_string(), params.amplitude_mean);
        gt_params.insert("rise_time".to_string(), params.rise_time);
        gt_params.insert("recovery_time".to_string(), params.recovery_time);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        ScrEventParams {
            duration: 60.0,
            sampling_rate: 10.0,
            event_rate: 3.0,       // 3 SCRs per minute
            amplitude_mean: 0.5,   // microsiemens
            amplitude_std: 0.2,
            rise_time: 1.5,        // seconds
            recovery_time: 5.0,    // seconds
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.rise_time <= 0.0 || params.recovery_time <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("time constants must be positive".to_string()));
        }
        if params.rise_time >= params.recovery_time {
            return Err(crate::GeneratorError::InvalidParameter("rise_time should be < recovery_time".to_string()));
        }
        Ok(())
    }
}

/// Stimulus-locked SCR generator
pub struct StimulusLockedScrGenerator;

#[derive(Debug, Clone)]
pub struct StimulusLockedScrParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub stimulus_times: Vec<f64>, // exact stimulus times
    pub latency_mean: f64,         // mean SCR latency (1-3s)
    pub latency_std: f64,
    pub amplitude_mean: f64,
    pub amplitude_std: f64,
    pub rise_time: f64,
    pub recovery_time: f64,
    pub response_probability: f64, // 0-1, not all stimuli elicit SCR
}

impl SyntheticGenerator for StimulusLockedScrGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = StimulusLockedScrParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let latency_dist = Normal::new(params.latency_mean, params.latency_std).unwrap();
        let amp_dist = Normal::new(params.amplitude_mean, params.amplitude_std).unwrap();

        let mut signal = vec![0.0; n_samples];
        let mut events = Vec::new();

        for stim_time in &params.stimulus_times {
            // Check if this stimulus elicits a response
            if rng.random::<f64>() > params.response_probability {
                continue;
            }

            let latency = latency_dist.sample(&mut rng).max(0.1);
            let amplitude = amp_dist.sample(&mut rng).max(0.0);
            let scr_onset = stim_time + latency;

            events.push(Event {
                time: scr_onset,
                event_type: "stimulus_locked_SCR".to_string(),
                amplitude: Some(amplitude),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("stimulus_time".to_string(), *stim_time);
                    attrs.insert("latency".to_string(), latency);
                    attrs
                },
            });

            // Generate SCR using Bateman function
            for (i, i_slot) in signal.iter_mut().enumerate() {
                let t = i as f64 * dt;
                if t >= scr_onset {
                    let delta_t = t - scr_onset;
                    let scr = amplitude *
                        ((-delta_t / params.recovery_time).exp() -
                         (-delta_t / params.rise_time).exp());
                    *i_slot += scr;
                }
            }
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("latency_mean".to_string(), params.latency_mean);
        gt_params.insert("amplitude_mean".to_string(), params.amplitude_mean);
        gt_params.insert("response_probability".to_string(), params.response_probability);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        StimulusLockedScrParams {
            duration: 60.0,
            sampling_rate: 10.0,
            stimulus_times: vec![5.0, 15.0, 25.0, 35.0, 45.0, 55.0],
            latency_mean: 2.0,
            latency_std: 0.5,
            amplitude_mean: 0.8,
            amplitude_std: 0.3,
            rise_time: 1.5,
            recovery_time: 5.0,
            response_probability: 0.85,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.response_probability < 0.0 || params.response_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("response_probability must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Arousal state generator
pub struct ArousalStateGenerator;

#[derive(Debug, Clone)]
pub enum ArousalState {
    Low,      // relaxed, drowsy
    Medium,   // alert, baseline
    High,     // stressed, aroused
}

#[derive(Debug, Clone)]
pub struct ArousalStateParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub state_sequence: Vec<(f64, ArousalState)>, // (time, state)
    pub transition_duration: f64, // seconds
}

impl SyntheticGenerator for ArousalStateGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = ArousalStateParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Map arousal states to SCL levels
        let state_to_scl = |state: &ArousalState| match state {
            ArousalState::Low => 2.0,
            ArousalState::Medium => 5.0,
            ArousalState::High => 10.0,
        };

        let mut signal = Vec::with_capacity(n_samples);
        let noise = Normal::new(0.0, 0.1).unwrap();

        for i in 0..n_samples {
            let t = i as f64 * dt;

            // Find current state
            let mut target_scl = state_to_scl(&ArousalState::Medium);
            for (state_time, state) in &params.state_sequence {
                if t >= *state_time {
                    target_scl = state_to_scl(state);
                } else {
                    break;
                }
            }

            // Add smooth transitions
            let mut scl = target_scl;
            for j in 0..params.state_sequence.len() {
                let (state_time, _) = &params.state_sequence[j];
                if t >= *state_time && t < state_time + params.transition_duration {
                    let prev_scl = if j > 0 {
                        state_to_scl(&params.state_sequence[j - 1].1)
                    } else {
                        state_to_scl(&ArousalState::Medium)
                    };
                    let transition_progress = (t - state_time) / params.transition_duration;
                    scl = prev_scl + (target_scl - prev_scl) * transition_progress;
                }
            }

            signal.push(scl + noise.sample(&mut rng));
        }

        let signal = Array1::from_vec(signal);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: HashMap::new(),
            events: Vec::new(),
            segments: params.state_sequence
                .windows(2)
                .map(|w| crate::traits::Segment {
                    start: w[0].0,
                    end: w[1].0,
                    label: format!("{:?}", w[0].1),
                })
                .collect(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        ArousalStateParams {
            duration: 180.0, // 3 minutes
            sampling_rate: 10.0,
            state_sequence: vec![
                (0.0, ArousalState::Medium),
                (60.0, ArousalState::High),
                (120.0, ArousalState::Low),
            ],
            transition_duration: 10.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.state_sequence.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("state_sequence cannot be empty".to_string()));
        }
        Ok(())
    }
}

/// EDA artifact generator (movement and electrode artifacts)
pub struct EdaArtifactGenerator;

#[derive(Debug, Clone)]
pub struct EdaArtifactParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub movement_artifact_rate: f64,    // artifacts per minute
    pub movement_amplitude: f64,        // microsiemens
    pub movement_duration: f64,         // seconds
    pub electrode_artifact_rate: f64,   // artifacts per minute
    pub electrode_amplitude: f64,       // microsiemens (spikes)
}

impl SyntheticGenerator for EdaArtifactGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = EdaArtifactParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut signal = vec![0.0; n_samples];
        let mut events = Vec::new();

        // Generate movement artifacts (slow, large deflections)
        let movement_exp = Exp::new(params.movement_artifact_rate / 60.0).unwrap();
        let mut t = movement_exp.sample(&mut rng);
        while t < params.duration {
            let amp_dist = Normal::new(params.movement_amplitude, params.movement_amplitude * 0.3).unwrap();
            let amplitude = amp_dist.sample(&mut rng).abs();
            let duration = params.movement_duration * rng.random_range(0.5..1.5);

            events.push(Event {
                time: t,
                event_type: "movement_artifact".to_string(),
                amplitude: Some(amplitude),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("duration".to_string(), duration);
                    attrs
                },
            });

            // Add slow artifact waveform
            let start_idx = (t * params.sampling_rate) as usize;
            let end_idx = (((t + duration) * params.sampling_rate) as usize).min(n_samples);

            for (i_off, i_slot) in signal[start_idx..end_idx].iter_mut().enumerate() {
                let i = start_idx + i_off;
                let local_t = (i - start_idx) as f64 * dt;
                let envelope = (std::f64::consts::PI * local_t / duration).sin();
                *i_slot += amplitude * envelope;
            }

            t += movement_exp.sample(&mut rng);
        }

        // Generate electrode artifacts (fast spikes)
        let electrode_exp = Exp::new(params.electrode_artifact_rate / 60.0).unwrap();
        let mut t = electrode_exp.sample(&mut rng);
        while t < params.duration {
            let amp_dist = Normal::new(params.electrode_amplitude, params.electrode_amplitude * 0.5).unwrap();
            let amplitude = amp_dist.sample(&mut rng);

            events.push(Event {
                time: t,
                event_type: "electrode_artifact".to_string(),
                amplitude: Some(amplitude),
                attributes: HashMap::new(),
            });

            // Add sharp spike
            let spike_idx = (t * params.sampling_rate) as usize;
            if spike_idx < n_samples {
                signal[spike_idx] += amplitude;

                // Exponential decay
                for i in 1..20.min(n_samples - spike_idx) {
                    let decay = (-(i as f64) * 0.3).exp();
                    signal[spike_idx + i] += amplitude * decay;
                }
            }

            t += electrode_exp.sample(&mut rng);
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("movement_artifact_rate".to_string(), params.movement_artifact_rate);
        gt_params.insert("electrode_artifact_rate".to_string(), params.electrode_artifact_rate);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        EdaArtifactParams {
            duration: 60.0,
            sampling_rate: 10.0,
            movement_artifact_rate: 2.0,     // 2 per minute
            movement_amplitude: 2.0,          // microsiemens
            movement_duration: 3.0,           // seconds
            electrode_artifact_rate: 1.0,     // 1 per minute
            electrode_amplitude: 5.0,         // microsiemens (spikes)
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eda_tonic_generation() {
        let generator = EdaTonicGenerator;
        let params = EdaTonicGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_scr_event_generation() {
        let generator = ScrEventGenerator;
        let params = ScrEventGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
        assert!(!result.ground_truth.events.is_empty());
    }

    #[test]
    fn test_stimulus_locked_scr() {
        let generator = StimulusLockedScrGenerator;
        let params = StimulusLockedScrGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_arousal_state_generation() {
        let generator = ArousalStateGenerator;
        let params = ArousalStateGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
        assert!(!result.ground_truth.segments.is_empty());
    }

    #[test]
    fn test_eda_artifact_generation() {
        let generator = EdaArtifactGenerator;
        let params = EdaArtifactGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
        assert!(!result.ground_truth.events.is_empty());
    }
}
