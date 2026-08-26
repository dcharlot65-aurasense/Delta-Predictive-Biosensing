//! ECG signal generators
//!
//! Implements McSharry dynamical model for realistic ECG synthesis with:
//! - P, QRS, T wave morphology
//! - Heart rate variability (HRV)
//! - Arrhythmias (PAC, PVC, AF)
//! - Respiratory sinus arrhythmia (RSA)

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, Event};
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// McSharry ECG morphology generator using coupled ODEs
pub struct EcgMorphologyGenerator;

#[derive(Debug, Clone)]
pub struct EcgMorphologyParams {
    pub duration: f64,           // seconds
    pub sampling_rate: f64,      // Hz
    pub heart_rate: f64,         // bpm
    pub p_wave: WaveParams,
    pub qrs_complex: WaveParams,
    pub t_wave: WaveParams,
}

#[derive(Debug, Clone)]
pub struct WaveParams {
    pub amplitude: f64,  // mV
    pub width: f64,      // radians
    pub time_offset: f64, // radians from R peak
}

impl SyntheticGenerator for EcgMorphologyGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = EcgMorphologyParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let omega = 2.0 * PI * (params.heart_rate / 60.0); // angular frequency

        // McSharry model state variable.
        //
        // The (x, y) limit-cycle pair the original model uses to produce the
        // cardiac phase is not needed here: `theta` is derived directly from
        // `omega * t` below. It was retained only to detect R peaks, which it
        // did wrongly -- see the detection comment in the loop.
        let mut z = 0.0;
        let mut prev_theta = 0.0;

        let mut signal = Vec::with_capacity(n_samples);
        let mut r_peaks = Vec::new();

        // Wave parameters (P, Q, R, S, T)
        let waves = vec![
            ('P', params.p_wave.amplitude, params.p_wave.width, params.p_wave.time_offset),
            ('Q', -0.1, 0.1, -PI / 12.0),
            ('R', 1.0, 0.1, 0.0),
            ('S', -0.2, 0.1, PI / 12.0),
            ('T', params.t_wave.amplitude, params.t_wave.width, params.t_wave.time_offset),
        ];

        for i in 0..n_samples {
            let t = i as f64 * dt;
            let theta = (omega * t).rem_euclid(2.0 * PI);

            // Emit one R peak per cardiac cycle, at the wrap of the cardiac
            // phase.
            //
            // The R wave sits at `theta = 0` (see `waves` above), so the phase
            // wrapping past 2*pi is exactly the R peak, and there is precisely
            // one per beat.
            //
            // This previously tested `x`, the oscillator coordinate, rather
            // than the signal or the phase -- and `x` starts at 1.0 and moves
            // slowly, so `|x - 1| < 0.01` held for a whole run of consecutive
            // samples at the start of the record. The generator therefore
            // labelled 47 "R peaks" one MILLISECOND apart in a 3-second trace
            // that contains three beats. Since these labels are the event-level
            // ground truth the synthetic data exists to provide, anything
            // validated against them was being scored against noise. The
            // oscillator was also never coupled to `omega`, so it bore no
            // relation to the heart rate at all.
            if i > 0 && theta < prev_theta {
                r_peaks.push(Event {
                    time: t,
                    event_type: "R_peak".to_string(),
                    amplitude: Some(z),
                    attributes: HashMap::new(),
                });
            }
            prev_theta = theta;

            // McSharry ODE: dz/dt = sum of Gaussian bumps.
            //
            // The `a` coefficients here are ODE terms, not waveform amplitudes.
            // Integrating `-a * dtheta * exp(-dtheta^2 / 2b^2)` over theta
            // yields `a * b^2 * exp(...)`, so the resulting wave peaks at
            // `a * b^2` -- with the published widths of ~0.1 that is a hundred
            // times smaller than `a`.
            //
            // The loop integrates over TIME while the term is expressed in
            // phase, which contributes a further factor of `1/omega`, so the
            // wave actually peaks at `a * b^2 / omega`.
            //
            // The parameters this generator exposes are documented as
            // amplitudes in mV, so each is converted into the matching ODE
            // coefficient by `a * omega / b^2`. Without it the model produced
            // an ECG spanning 0.004 mV peak to peak where 1 mV was asked for --
            // four hundred times too small, far below any encoder threshold,
            // so every downstream pipeline saw a flat line. Carrying `omega`
            // also keeps the amplitude independent of heart rate, which it
            // otherwise would not be.
            let mut dz_dt = 0.0;
            for (_, ai, bi, thetai) in &waves {
                let delta_theta = (theta - thetai).rem_euclid(2.0 * PI);
                let delta_theta = if delta_theta > PI { delta_theta - 2.0 * PI } else { delta_theta };
                let coefficient = ai * omega / bi.powi(2);
                dz_dt += -coefficient * delta_theta
                    * (-delta_theta.powi(2) / (2.0 * bi.powi(2))).exp();
            }

            // Euler integration of the waveform state.
            z += (dz_dt - z) * dt;

            signal.push(z);
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("heart_rate".to_string(), params.heart_rate);
        gt_params.insert("p_amplitude".to_string(), params.p_wave.amplitude);
        gt_params.insert("qrs_amplitude".to_string(), params.qrs_complex.amplitude);
        gt_params.insert("t_amplitude".to_string(), params.t_wave.amplitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: r_peaks,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        EcgMorphologyParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            heart_rate: 60.0,
            p_wave: WaveParams {
                amplitude: 0.25,
                width: 0.1,
                time_offset: -PI / 3.0,
            },
            qrs_complex: WaveParams {
                amplitude: 1.0,
                width: 0.1,
                time_offset: 0.0,
            },
            t_wave: WaveParams {
                amplitude: 0.35,
                width: 0.25,
                time_offset: PI / 2.0,
            },
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.heart_rate <= 0.0 || params.heart_rate > 300.0 {
            return Err(crate::GeneratorError::InvalidParameter("heart_rate must be 0-300 bpm".to_string()));
        }
        Ok(())
    }
}

/// Heart rate generator with HRV
pub struct HeartRateGenerator;

#[derive(Debug, Clone)]
pub struct HeartRateParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub mean_hr: f64,           // bpm
    pub hrv_std: f64,           // RR interval std dev (ms)
    pub lf_power: f64,          // low frequency power (0.04-0.15 Hz)
    pub hf_power: f64,          // high frequency power (0.15-0.4 Hz)
}

impl SyntheticGenerator for HeartRateGenerator {
    type Output = Vec<f64>; // RR intervals in seconds
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = HeartRateParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mean_rr = 60.0 / params.mean_hr; // seconds

        // Generate RR intervals with HRV components
        let mut rr_intervals = Vec::new();
        let mut t = 0.0;

        while t < params.duration {
            // LF component (0.1 Hz)
            let lf = params.lf_power.sqrt() * (2.0 * PI * 0.1 * t).sin();

            // HF component (0.25 Hz, respiratory)
            let hf = params.hf_power.sqrt() * (2.0 * PI * 0.25 * t).sin();

            // Random component
            let normal = Normal::new(0.0, params.hrv_std / 1000.0).unwrap();
            let noise = normal.sample(&mut rng);

            let rr = mean_rr + lf + hf + noise;
            let rr = rr.max(0.3); // physiological minimum ~200 bpm

            rr_intervals.push(rr);
            t += rr;
        }

        let mut gt_params = HashMap::new();
        gt_params.insert("mean_hr".to_string(), params.mean_hr);
        gt_params.insert("hrv_std".to_string(), params.hrv_std);
        gt_params.insert("lf_power".to_string(), params.lf_power);
        gt_params.insert("hf_power".to_string(), params.hf_power);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(rr_intervals, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        HeartRateParams {
            duration: 60.0,
            sampling_rate: 4.0, // Hz (not used for RR intervals)
            mean_hr: 70.0,
            hrv_std: 50.0,      // ms
            lf_power: 0.01,     // normalized power
            hf_power: 0.005,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.mean_hr <= 0.0 || params.mean_hr > 300.0 {
            return Err(crate::GeneratorError::InvalidParameter("mean_hr must be 0-300 bpm".to_string()));
        }
        Ok(())
    }
}

/// HRV spectral generator
pub struct HrvSpectralGenerator;

#[derive(Debug, Clone)]
pub struct HrvSpectralParams {
    pub duration: f64,
    pub mean_hr: f64,
    pub lf_center: f64,    // Hz (typically 0.1)
    pub lf_width: f64,     // Hz
    pub lf_power: f64,     // ms^2
    pub hf_center: f64,    // Hz (typically 0.25)
    pub hf_width: f64,     // Hz
    pub hf_power: f64,     // ms^2
}

impl SyntheticGenerator for HrvSpectralGenerator {
    type Output = Vec<f64>; // RR intervals
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = HrvSpectralParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mean_rr = 60.0 / params.mean_hr;

        let mut rr_intervals = Vec::new();
        let mut t = 0.0;

        while t < params.duration {
            // LF band (0.04-0.15 Hz)
            let lf_freq = params.lf_center + rng.r#gen_range(-params.lf_width..params.lf_width);
            let lf_phase = rng.r#gen_range(0.0..2.0 * PI);
            let lf = (params.lf_power / 1000.0).sqrt() * (2.0 * PI * lf_freq * t + lf_phase).sin();

            // HF band (0.15-0.4 Hz)
            let hf_freq = params.hf_center + rng.r#gen_range(-params.hf_width..params.hf_width);
            let hf_phase = rng.r#gen_range(0.0..2.0 * PI);
            let hf = (params.hf_power / 1000.0).sqrt() * (2.0 * PI * hf_freq * t + hf_phase).sin();

            let rr = (mean_rr + lf + hf).max(0.3);
            rr_intervals.push(rr);
            t += rr;
        }

        let mut gt_params = HashMap::new();
        gt_params.insert("mean_hr".to_string(), params.mean_hr);
        gt_params.insert("lf_power".to_string(), params.lf_power);
        gt_params.insert("hf_power".to_string(), params.hf_power);
        gt_params.insert("lf_hf_ratio".to_string(), params.lf_power / params.hf_power);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(rr_intervals, ground_truth, 4.0))
    }

    fn default_params() -> Self::Parameters {
        HrvSpectralParams {
            duration: 300.0, // 5 minutes for HRV analysis
            mean_hr: 70.0,
            lf_center: 0.1,
            lf_width: 0.05,
            lf_power: 1000.0, // ms^2
            hf_center: 0.25,
            hf_width: 0.1,
            hf_power: 500.0,  // ms^2
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration < 60.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration should be >= 60s for HRV".to_string()));
        }
        Ok(())
    }
}

/// Arrhythmia generator (PAC, PVC, AF)
pub struct ArrhythmiaGenerator;

#[derive(Debug, Clone)]
pub enum ArrhythmiaType {
    PAC,  // Premature atrial contraction
    PVC,  // Premature ventricular contraction
    AF,   // Atrial fibrillation
}

#[derive(Debug, Clone)]
pub struct ArrhythmiaParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_hr: f64,
    pub arrhythmia_type: ArrhythmiaType,
    pub frequency: f64, // events per minute
}

impl SyntheticGenerator for ArrhythmiaGenerator {
    type Output = Vec<f64>; // RR intervals with arrhythmia
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = ArrhythmiaParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mean_rr = 60.0 / params.baseline_hr;
        let event_interval = 60.0 / params.frequency; // seconds between arrhythmic events

        let mut rr_intervals = Vec::new();
        let mut events = Vec::new();
        let mut t = 0.0;
        let mut next_event_time = rng.r#gen_range(0.0..event_interval);

        while t < params.duration {
            let mut rr = mean_rr + rng.r#gen_range(-0.05..0.05); // normal HRV

            // Check if arrhythmic event should occur
            if t >= next_event_time {
                match params.arrhythmia_type {
                    ArrhythmiaType::PAC => {
                        // Premature beat, then compensatory pause
                        rr *= 0.7; // early beat
                        events.push(Event {
                            time: t,
                            event_type: "PAC".to_string(),
                            amplitude: Some(0.8),
                            attributes: HashMap::new(),
                        });
                    }
                    ArrhythmiaType::PVC => {
                        // Premature ventricular beat with wide QRS
                        rr *= 0.65; // very early beat
                        events.push(Event {
                            time: t,
                            event_type: "PVC".to_string(),
                            amplitude: Some(1.2),
                            attributes: HashMap::new(),
                        });
                    }
                    ArrhythmiaType::AF => {
                        // Irregularly irregular rhythm
                        rr *= rng.r#gen_range(0.6..1.4);
                        events.push(Event {
                            time: t,
                            event_type: "AF_beat".to_string(),
                            amplitude: Some(0.9),
                            attributes: HashMap::new(),
                        });
                    }
                }
                next_event_time += event_interval + rng.r#gen_range(-event_interval * 0.3..event_interval * 0.3);
            }

            rr_intervals.push(rr.max(0.3));
            t += rr;
        }

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_hr".to_string(), params.baseline_hr);
        gt_params.insert("arrhythmia_frequency".to_string(), params.frequency);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(rr_intervals, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        ArrhythmiaParams {
            duration: 60.0,
            sampling_rate: 4.0,
            baseline_hr: 70.0,
            arrhythmia_type: ArrhythmiaType::PVC,
            frequency: 5.0, // 5 PVCs per minute
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.baseline_hr <= 0.0 || params.baseline_hr > 300.0 {
            return Err(crate::GeneratorError::InvalidParameter("baseline_hr must be 0-300 bpm".to_string()));
        }
        Ok(())
    }
}

/// Respiratory sinus arrhythmia (RSA) generator
pub struct RsaGenerator;

#[derive(Debug, Clone)]
pub struct RsaParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub mean_hr: f64,
    pub respiratory_rate: f64, // breaths per minute
    pub rsa_amplitude: f64,    // HR modulation amplitude (bpm)
}

impl SyntheticGenerator for RsaGenerator {
    type Output = Vec<f64>; // RR intervals with RSA
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = RsaParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mean_rr = 60.0 / params.mean_hr;
        let resp_freq = params.respiratory_rate / 60.0; // Hz

        let mut rr_intervals = Vec::new();
        let mut t = 0.0;

        while t < params.duration {
            // RSA modulation (HR increases during inspiration)
            let hr_modulation = params.rsa_amplitude * (2.0 * PI * resp_freq * t).sin();
            let instantaneous_hr = params.mean_hr + hr_modulation;
            let rr = 60.0 / instantaneous_hr;

            // Add small random variation
            let noise = rng.r#gen_range(-0.02..0.02);
            let rr = (rr + noise).max(0.3);

            rr_intervals.push(rr);
            t += rr;
        }

        let mut gt_params = HashMap::new();
        gt_params.insert("mean_hr".to_string(), params.mean_hr);
        gt_params.insert("respiratory_rate".to_string(), params.respiratory_rate);
        gt_params.insert("rsa_amplitude".to_string(), params.rsa_amplitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(rr_intervals, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        RsaParams {
            duration: 60.0,
            sampling_rate: 4.0,
            mean_hr: 70.0,
            respiratory_rate: 15.0, // breaths per minute
            rsa_amplitude: 5.0,     // bpm variation
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.mean_hr <= 0.0 || params.mean_hr > 300.0 {
            return Err(crate::GeneratorError::InvalidParameter("mean_hr must be 0-300 bpm".to_string()));
        }
        if params.respiratory_rate <= 0.0 || params.respiratory_rate > 60.0 {
            return Err(crate::GeneratorError::InvalidParameter("respiratory_rate must be 0-60 bpm".to_string()));
        }
        Ok(())
    }
}

/// Heart rate recovery generator (post-exercise HR decay)
pub struct HeartRateRecoveryGenerator;

#[derive(Debug, Clone)]
pub struct HeartRateRecoveryParams {
    pub duration: f64,          // seconds
    pub sampling_rate: f64,     // Hz
    pub peak_hr: f64,           // bpm (max HR at exercise end)
    pub resting_hr: f64,        // bpm (baseline resting HR)
    pub recovery_tau: f64,      // time constant (seconds) - fitness indicator
    pub hrv_noise: f64,         // beat-to-beat variability (ms)
}

impl SyntheticGenerator for HeartRateRecoveryGenerator {
    type Output = Vec<f64>; // RR intervals
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = HeartRateRecoveryParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, params.hrv_noise / 1000.0).unwrap();

        let mut rr_intervals = Vec::new();
        let mut t = 0.0;
        let mut events = Vec::new();

        // Calculate HR delta for recovery curve
        let hr_delta = params.peak_hr - params.resting_hr;

        // Mark recovery start
        events.push(Event {
            time: 0.0,
            event_type: "recovery_start".to_string(),
            amplitude: Some(params.peak_hr),
            attributes: HashMap::new(),
        });

        while t < params.duration {
            // Exponential decay from peak HR to resting HR
            let current_hr = params.resting_hr + hr_delta * (-t / params.recovery_tau).exp();

            // Convert to RR interval
            let mean_rr = 60.0 / current_hr;

            // Add HRV noise
            let noise = normal.sample(&mut rng);
            let rr = (mean_rr + noise).max(0.3);

            rr_intervals.push(rr);
            t += rr;
        }

        // Calculate recovery metrics
        let hr_1min = if t >= 60.0 {
            let rr_at_1min = rr_intervals.iter()
                .scan(0.0, |acc, &rr| {
                    *acc += rr;
                    Some(*acc)
                })
                .position(|cum_t| cum_t >= 60.0);

            if let Some(idx) = rr_at_1min {
                let rr = rr_intervals[idx];
                60.0 / rr
            } else {
                params.resting_hr
            }
        } else {
            params.resting_hr
        };

        let hr_recovery = params.peak_hr - hr_1min;

        let mut gt_params = HashMap::new();
        gt_params.insert("peak_hr".to_string(), params.peak_hr);
        gt_params.insert("resting_hr".to_string(), params.resting_hr);
        gt_params.insert("recovery_tau".to_string(), params.recovery_tau);
        gt_params.insert("hr_recovery_1min".to_string(), hr_recovery);

        // Mark significant recovery milestones
        let recovery_50_time = params.recovery_tau * (2.0_f64).ln(); // 50% recovery
        if recovery_50_time < params.duration {
            events.push(Event {
                time: recovery_50_time,
                event_type: "recovery_50_percent".to_string(),
                amplitude: Some(params.resting_hr + hr_delta * 0.5),
                attributes: HashMap::new(),
            });
        }

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(rr_intervals, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        HeartRateRecoveryParams {
            duration: 300.0,  // 5 minutes
            sampling_rate: 4.0,
            peak_hr: 170.0,
            resting_hr: 70.0,
            recovery_tau: 60.0,  // 60s tau = good fitness
            hrv_noise: 40.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.peak_hr <= params.resting_hr {
            return Err(crate::GeneratorError::InvalidParameter("peak_hr must be > resting_hr".to_string()));
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
    fn test_ecg_morphology_generation() {
        let generator = EcgMorphologyGenerator;
        let params = EcgMorphologyGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
        assert!(!result.ground_truth.events.is_empty()); // Should have R peaks
    }

    #[test]
    fn test_heart_rate_generation() {
        let generator = HeartRateGenerator;
        let params = HeartRateGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());

        // Check mean HR is approximately correct
        let mean_rr = result.signal.iter().sum::<f64>() / result.signal.len() as f64;
        let mean_hr = 60.0 / mean_rr;
        assert!((mean_hr - params.mean_hr).abs() < 5.0);
    }

    #[test]
    fn test_arrhythmia_generation() {
        let generator = ArrhythmiaGenerator;
        let params = ArrhythmiaGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());
        assert!(!result.ground_truth.events.is_empty()); // Should have arrhythmic events
    }

    #[test]
    fn test_rsa_generation() {
        let generator = RsaGenerator;
        let params = RsaGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());
    }

    #[test]
    fn test_heart_rate_recovery() {
        let generator = HeartRateRecoveryGenerator;
        let params = HeartRateRecoveryGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());

        // Check HR decreases over time
        let early_rr: f64 = result.signal.iter().take(10).sum::<f64>() / 10.0;
        let late_rr: f64 = result.signal.iter().rev().take(10).sum::<f64>() / 10.0;
        assert!(late_rr > early_rr); // RR should increase (HR decreases)
    }
}
