//! Synchronized multi-signal generators
//!
//! Generators for coupled biosignals with physiological relationships

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// ECG-PPG synchronized generator with pulse transit time (PTT)
pub struct EcgPpgSynchronizedGenerator;

#[derive(Debug, Clone)]
pub struct EcgPpgSyncParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub heart_rate: f64,           // bpm
    pub ptt_mean: f64,             // Pulse Transit Time in ms (typically 150-250ms)
    pub ptt_std: f64,              // ms
    pub pat_offset: f64,           // Pre-ejection period offset (ms)
    pub ppg_amplitude: f64,        // relative PPG amplitude
    pub hrv_std: f64,              // ms (heart rate variability)
}

impl SyntheticGenerator for EcgPpgSynchronizedGenerator {
    type Output = (Array1<f64>, Array1<f64>); // (ECG, PPG)
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = EcgPpgSyncParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let _dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mean_rr = 60.0 / params.heart_rate; // seconds
        let hrv_dist = Normal::new(0.0, params.hrv_std / 1000.0).unwrap();
        let ptt_dist = Normal::new(params.ptt_mean / 1000.0, params.ptt_std / 1000.0).unwrap();

        let mut ecg = vec![0.0; n_samples];
        let mut ppg = vec![0.0; n_samples];
        let mut events = Vec::new();

        let mut t = 0.0;
        let mut beat_count = 0;

        // Generate synchronized heartbeats
        while t < params.duration {
            let rr = (mean_rr + hrv_dist.sample(&mut rng)).max(0.3);
            let ptt = ptt_dist.sample(&mut rng).max(0.05);

            // ECG R-peak
            let r_idx = (t * params.sampling_rate) as usize;
            if r_idx < n_samples {
                // Simple R-peak (spike)
                ecg[r_idx] = 1.0;
                if r_idx > 0 {
                    ecg[r_idx - 1] = 0.3; // Q
                }
                if r_idx + 1 < n_samples {
                    ecg[r_idx + 1] = 0.3; // S
                }
                if r_idx + 3 < n_samples {
                    ecg[r_idx + 3] = 0.4; // T wave
                }

                events.push(Event {
                    time: t,
                    event_type: "R_peak".to_string(),
                    amplitude: Some(1.0),
                    attributes: HashMap::new(),
                });
            }

            // PPG pulse (delayed by PTT + PAT)
            let ppg_time = t + ptt + (params.pat_offset / 1000.0);
            let ppg_idx = (ppg_time * params.sampling_rate) as usize;

            if ppg_idx < n_samples {
                // PPG pulse waveform (systolic peak + dicrotic notch)
                let pulse_duration = 0.3; // 300ms pulse
                let n_pulse_samples = (pulse_duration * params.sampling_rate) as usize;

                for i in 0..n_pulse_samples.min(n_samples - ppg_idx) {
                    let pulse_t = i as f64 / params.sampling_rate;
                    let normalized_t = pulse_t / pulse_duration;

                    // Systolic peak (Gaussian)
                    let systolic = (-((normalized_t - 0.3).powi(2)) / 0.02).exp();

                    // Dicrotic notch (smaller peak)
                    let dicrotic = 0.3 * (-((normalized_t - 0.6).powi(2)) / 0.01).exp();

                    ppg[ppg_idx + i] += params.ppg_amplitude * (systolic + dicrotic);
                }

                events.push(Event {
                    time: ppg_time,
                    event_type: "PPG_peak".to_string(),
                    amplitude: Some(params.ppg_amplitude),
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("ptt".to_string(), ptt * 1000.0);
                        attrs.insert("beat_number".to_string(), beat_count as f64);
                        attrs
                    },
                });
            }

            t += rr;
            beat_count += 1;
        }

        let ecg = Array1::from_vec(ecg);
        let ppg = Array1::from_vec(ppg);

        let mut gt_params = HashMap::new();
        gt_params.insert("heart_rate".to_string(), params.heart_rate);
        gt_params.insert("ptt_mean".to_string(), params.ptt_mean);
        gt_params.insert("ptt_std".to_string(), params.ptt_std);
        gt_params.insert("pat_offset".to_string(), params.pat_offset);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new((ecg, ppg), ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        EcgPpgSyncParams {
            duration: 60.0,
            sampling_rate: 100.0,
            heart_rate: 70.0,
            ptt_mean: 200.0,       // ms
            ptt_std: 20.0,
            pat_offset: 100.0,     // Pre-ejection period
            ppg_amplitude: 1.0,
            hrv_std: 40.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.ptt_mean <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("ptt_mean must be positive".to_string()));
        }
        Ok(())
    }
}

/// Cardiorespiratory coupling generator
pub struct CardiorespiratoryCouplngGenerator;

#[derive(Debug, Clone)]
pub struct CardiorespiratoryParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub mean_hr: f64,              // bpm
    pub respiratory_rate: f64,     // breaths per minute
    pub rsa_amplitude: f64,        // bpm (respiratory sinus arrhythmia)
    pub phase_coupling: f64,       // 0-1 (strength of phase coupling)
}

impl SyntheticGenerator for CardiorespiratoryCouplngGenerator {
    type Output = (Vec<f64>, Array1<f64>); // (RR intervals, respiratory signal)
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = CardiorespiratoryParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let resp_freq = params.respiratory_rate / 60.0; // Hz
        let resp_phase = rng.random_range(0.0..2.0 * PI);

        // Generate respiratory signal
        let respiratory: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * resp_freq * t + resp_phase).sin()
            })
            .collect();

        // Generate RR intervals with RSA
        let mut rr_intervals = Vec::new();
        let mut t = 0.0;
        let mut events = Vec::new();

        while t < params.duration {
            // Get current respiratory phase
            let resp_phase_current = (2.0 * PI * resp_freq * t + resp_phase) % (2.0 * PI);

            // HR modulation by respiration (RSA)
            let hr_modulation = params.rsa_amplitude * resp_phase_current.sin();

            // Phase coupling adds deterministic relationship
            let coupling_effect = if params.phase_coupling > 0.5 {
                // Strong coupling: heartbeats tend to occur at specific respiratory phases
                let preferred_phase = PI / 2.0; // inspiration
                let phase_diff = (resp_phase_current - preferred_phase).abs();
                -params.rsa_amplitude * 0.5 * (phase_diff / PI)
            } else {
                0.0
            };

            let instantaneous_hr = params.mean_hr + hr_modulation + coupling_effect;
            let rr = (60.0 / instantaneous_hr).max(0.3);

            rr_intervals.push(rr);

            // Record phase-locked events
            if params.phase_coupling > 0.7 && (resp_phase_current - PI / 2.0).abs() < 0.2 {
                events.push(Event {
                    time: t,
                    event_type: "phase_locked_beat".to_string(),
                    amplitude: Some(instantaneous_hr),
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("respiratory_phase".to_string(), resp_phase_current);
                        attrs
                    },
                });
            }

            t += rr;
        }

        let respiratory = Array1::from_vec(respiratory);

        let mut gt_params = HashMap::new();
        gt_params.insert("mean_hr".to_string(), params.mean_hr);
        gt_params.insert("respiratory_rate".to_string(), params.respiratory_rate);
        gt_params.insert("rsa_amplitude".to_string(), params.rsa_amplitude);
        gt_params.insert("phase_coupling".to_string(), params.phase_coupling);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new((rr_intervals, respiratory), ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        CardiorespiratoryParams {
            duration: 120.0,
            sampling_rate: 10.0,
            mean_hr: 70.0,
            respiratory_rate: 15.0,
            rsa_amplitude: 5.0,
            phase_coupling: 0.6,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.phase_coupling < 0.0 || params.phase_coupling > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("phase_coupling must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Autonomic state generator (sympathetic/parasympathetic balance)
pub struct AutonomicStateGenerator;

#[derive(Debug, Clone)]
pub enum AutonomicState {
    Parasympathetic,  // Rest and digest
    Balanced,         // Normal
    Sympathetic,      // Fight or flight
}

#[derive(Debug, Clone)]
pub struct AutonomicStateParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub state_transitions: Vec<(f64, AutonomicState)>, // (time, state)
    pub transition_duration: f64, // seconds
}

impl SyntheticGenerator for AutonomicStateGenerator {
    type Output = (Vec<f64>, Array1<f64>, Array1<f64>); // (RR intervals, HRV, EDA)
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = AutonomicStateParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Map states to physiological parameters
        let state_to_params = |state: &AutonomicState| -> (f64, f64, f64) {
            // (HR, HRV, EDA)
            match state {
                AutonomicState::Parasympathetic => (60.0, 80.0, 2.0),   // Low HR, high HRV, low EDA
                AutonomicState::Balanced => (70.0, 50.0, 5.0),          // Normal
                AutonomicState::Sympathetic => (90.0, 20.0, 10.0),      // High HR, low HRV, high EDA
            }
        };

        // Generate RR intervals
        let mut rr_intervals = Vec::new();
        let mut t = 0.0;

        while t < params.duration {
            let (hr, hrv_std, _) = Self::get_current_state_params(t, params, &state_to_params);

            let mean_rr = 60.0 / hr;
            let hrv_noise = Normal::new(0.0, hrv_std / 1000.0).unwrap();
            let rr = (mean_rr + hrv_noise.sample(&mut rng)).max(0.3);

            rr_intervals.push(rr);
            t += rr;
        }

        // Generate HRV time series
        let hrv: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;
                let (_, hrv_std, _) = Self::get_current_state_params(t, params, &state_to_params);
                hrv_std
            })
            .collect();

        // Generate EDA
        let eda: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;
                let (_, _, eda_level) = Self::get_current_state_params(t, params, &state_to_params);
                let noise = Normal::new(0.0, 0.1).unwrap();
                eda_level + noise.sample(&mut rng)
            })
            .collect();

        let hrv = Array1::from_vec(hrv);
        let eda = Array1::from_vec(eda);

        // Create segments for state transitions
        let segments: Vec<_> = params.state_transitions
            .windows(2)
            .map(|w| crate::traits::Segment {
                start: w[0].0,
                end: w[1].0,
                label: format!("{:?}", w[0].1),
            })
            .collect();

        let mut gt_params = HashMap::new();
        gt_params.insert("num_states".to_string(), params.state_transitions.len() as f64);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments,
        };

        Ok(GeneratedData::new((rr_intervals, hrv, eda), ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        AutonomicStateParams {
            duration: 300.0,
            sampling_rate: 10.0,
            state_transitions: vec![
                (0.0, AutonomicState::Balanced),
                (100.0, AutonomicState::Sympathetic),
                (200.0, AutonomicState::Parasympathetic),
            ],
            transition_duration: 20.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.state_transitions.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("state_transitions cannot be empty".to_string()));
        }
        Ok(())
    }
}

impl AutonomicStateGenerator {
    fn get_current_state_params<F>(
        t: f64,
        params: &AutonomicStateParams,
        state_to_params: &F,
    ) -> (f64, f64, f64)
    where
        F: Fn(&AutonomicState) -> (f64, f64, f64),
    {
        // Find current state
        let mut current_state = &params.state_transitions[0].1;
        let mut current_time = params.state_transitions[0].0;
        let mut next_state = None;
        let mut next_time = params.duration;

        for i in 0..params.state_transitions.len() {
            if t >= params.state_transitions[i].0 {
                current_state = &params.state_transitions[i].1;
                current_time = params.state_transitions[i].0;

                if i + 1 < params.state_transitions.len() {
                    next_state = Some(&params.state_transitions[i + 1].1);
                    next_time = params.state_transitions[i + 1].0;
                }
            }
        }

        let current_params = state_to_params(current_state);

        // Apply smooth transition if we're in transition period
        if let Some(next) = next_state
            && t >= current_time && t < current_time + params.transition_duration {
                let next_params = state_to_params(next);
                let progress = (t - current_time) / params.transition_duration;

                // Linear interpolation
                return (
                    current_params.0 + (next_params.0 - current_params.0) * progress,
                    current_params.1 + (next_params.1 - current_params.1) * progress,
                    current_params.2 + (next_params.2 - current_params.2) * progress,
                );
            }

        current_params
    }
}

/// Stress response generator (multi-signal stress markers)
pub struct StressResponseGenerator;

#[derive(Debug, Clone)]
pub struct StressResponseParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub stressor_times: Vec<f64>,  // times of stress events
    pub recovery_tau: f64,         // recovery time constant (s)
}

impl SyntheticGenerator for StressResponseGenerator {
    type Output = (Vec<f64>, Array1<f64>, Array1<f64>); // (RR intervals, EDA, cortisol proxy)
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = StressResponseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut stress_level = vec![0.0; n_samples];
        let mut events = Vec::new();

        // Generate stress level profile
        for &stressor_time in &params.stressor_times {
            events.push(Event {
                time: stressor_time,
                event_type: "stressor".to_string(),
                amplitude: Some(1.0),
                attributes: HashMap::new(),
            });

            for (i, i_slot) in stress_level.iter_mut().enumerate() {
                let t = i as f64 * dt;
                if t >= stressor_time {
                    let delta_t = t - stressor_time;
                    // Exponential rise and fall
                    let rise_tau = 5.0; // 5 seconds to peak
                    let response = if delta_t < rise_tau {
                        1.0 - (-delta_t / rise_tau).exp()
                    } else {
                        (-(delta_t - rise_tau) / params.recovery_tau).exp()
                    };
                    *i_slot += response;
                }
            }
        }

        // Generate RR intervals (decreased by stress)
        let baseline_hr = 70.0;
        let mut rr_intervals = Vec::new();
        let mut t = 0.0;

        while t < params.duration {
            let idx = (t * params.sampling_rate).min(n_samples as f64 - 1.0) as usize;
            let stress = stress_level[idx];

            // HR increases with stress
            let hr = baseline_hr + 20.0 * stress;
            let rr = (60.0 / hr).max(0.3);

            rr_intervals.push(rr);
            t += rr;
        }

        // Generate EDA (increased by stress)
        let baseline_eda = 5.0;
        let noise = Normal::new(0.0, 0.2).unwrap();
        let eda: Vec<f64> = stress_level
            .iter()
            .map(|&stress| baseline_eda + 5.0 * stress + noise.sample(&mut rng))
            .collect();

        // Cortisol proxy (slow response)
        let cortisol: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;
                let mut cortisol_level = 1.0; // baseline

                for &stressor_time in &params.stressor_times {
                    if t >= stressor_time {
                        let delta_t = t - stressor_time;
                        // Delayed rise (15-30 min), slow decay (1-2 hours)
                        let latency = 900.0; // 15 minutes
                        let rise_tau = 900.0; // 15 minutes
                        let decay_tau = 3600.0; // 1 hour

                        if delta_t < latency {
                            // No response yet
                        } else {
                            let t_since_onset = delta_t - latency;
                            if t_since_onset < rise_tau {
                                cortisol_level += 1.0 - (-t_since_onset / rise_tau).exp();
                            } else {
                                cortisol_level += (-(t_since_onset - rise_tau) / decay_tau).exp();
                            }
                        }
                    }
                }

                cortisol_level
            })
            .collect();

        let eda = Array1::from_vec(eda);
        let cortisol = Array1::from_vec(cortisol);

        let mut gt_params = HashMap::new();
        gt_params.insert("num_stressors".to_string(), params.stressor_times.len() as f64);
        gt_params.insert("recovery_tau".to_string(), params.recovery_tau);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new((rr_intervals, eda, cortisol), ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        StressResponseParams {
            duration: 3600.0,      // 1 hour
            sampling_rate: 1.0,    // 1 Hz
            stressor_times: vec![600.0, 1800.0, 3000.0],
            recovery_tau: 300.0,   // 5 minutes
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
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
    fn test_ecg_ppg_sync() {
        let generator = EcgPpgSynchronizedGenerator;
        let params = EcgPpgSynchronizedGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        let (ecg, ppg) = result.signal;
        let expected_len = (params.duration * params.sampling_rate) as usize;
        assert_eq!(ecg.len(), expected_len);
        assert_eq!(ppg.len(), expected_len);
        assert!(!result.ground_truth.events.is_empty());
    }

    #[test]
    fn test_cardiorespiratory_coupling() {
        let generator = CardiorespiratoryCouplngGenerator;
        let params = CardiorespiratoryCouplngGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        let (rr_intervals, respiratory) = result.signal;
        assert!(!rr_intervals.is_empty());
        assert_eq!(respiratory.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_autonomic_state() {
        let generator = AutonomicStateGenerator;
        let params = AutonomicStateGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        let (rr_intervals, hrv, eda) = result.signal;
        let expected_len = (params.duration * params.sampling_rate) as usize;
        assert!(!rr_intervals.is_empty());
        assert_eq!(hrv.len(), expected_len);
        assert_eq!(eda.len(), expected_len);
        assert!(!result.ground_truth.segments.is_empty());
    }

    #[test]
    fn test_stress_response() {
        let generator = StressResponseGenerator;
        let params = StressResponseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        let (rr_intervals, eda, cortisol) = result.signal;
        let expected_len = (params.duration * params.sampling_rate) as usize;
        assert!(!rr_intervals.is_empty());
        assert_eq!(eda.len(), expected_len);
        assert_eq!(cortisol.len(), expected_len);
        assert_eq!(result.ground_truth.events.len(), params.stressor_times.len());
    }
}
