//! PPG (photoplethysmography) signal generators

use crate::traits::{Event, GeneratedData, SyntheticGenerator, TimeSeriesGroundTruth};
use ndarray::Array1;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::PI;

/// PPG waveform generator
pub struct PpgWaveformGenerator;

#[derive(Debug, Clone)]
pub struct PpgWaveformParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub heart_rate: f64,
    pub systolic_amplitude: f64,
    pub dicrotic_notch_amplitude: f64,
    pub diastolic_amplitude: f64,
}

impl SyntheticGenerator for PpgWaveformGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PpgWaveformParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let beat_duration = 60.0 / params.heart_rate;

        let mut signal = Vec::with_capacity(n_samples);
        let mut peaks = Vec::new();
        let _rng = rand::rngs::StdRng::seed_from_u64(seed);

        for i in 0..n_samples {
            let t = i as f64 * dt;
            let phase = (t % beat_duration) / beat_duration; // 0-1 within beat

            // Systolic peak (Gaussian)
            let systolic = params.systolic_amplitude
                * (-((phase - 0.2).powi(2)) / (2.0 * 0.05_f64.powi(2))).exp();

            // Dicrotic notch (inverted Gaussian)
            let dicrotic = -params.dicrotic_notch_amplitude
                * (-((phase - 0.4).powi(2)) / (2.0 * 0.03_f64.powi(2))).exp();

            // Diastolic wave (Gaussian)
            let diastolic = params.diastolic_amplitude
                * (-((phase - 0.5).powi(2)) / (2.0 * 0.1_f64.powi(2))).exp();

            // Baseline decay
            let baseline = 0.1 * (1.0 - phase);

            let sample = systolic + dicrotic + diastolic + baseline;
            signal.push(sample);

            // Detect systolic peaks
            if phase > 0.18 && phase < 0.22 {
                peaks.push(Event {
                    time: t,
                    event_type: "systolic_peak".to_string(),
                    amplitude: Some(sample),
                    attributes: HashMap::new(),
                });
            }
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("heart_rate".to_string(), params.heart_rate);
        gt_params.insert("systolic_amplitude".to_string(), params.systolic_amplitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: peaks,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(
            signal,
            ground_truth,
            params.sampling_rate,
        ))
    }

    fn default_params() -> Self::Parameters {
        PpgWaveformParams {
            duration: 10.0,
            sampling_rate: 100.0,
            heart_rate: 70.0,
            systolic_amplitude: 1.0,
            dicrotic_notch_amplitude: 0.15,
            diastolic_amplitude: 0.3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "duration must be positive".to_string(),
            ));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "sampling_rate must be positive".to_string(),
            ));
        }
        if params.heart_rate <= 0.0 || params.heart_rate > 300.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "heart_rate must be 0-300 bpm".to_string(),
            ));
        }
        Ok(())
    }
}

/// PPG artifact generator (motion artifacts)
pub struct PpgArtifactGenerator;

#[derive(Debug, Clone)]
pub struct PpgArtifactParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub artifact_frequency: f64, // artifacts per minute
    pub artifact_duration: f64,  // seconds
    pub artifact_amplitude: f64,
}

impl SyntheticGenerator for PpgArtifactGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PpgArtifactParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let artifact_interval = 60.0 / params.artifact_frequency;
        let mut next_artifact = rng.random_range(0.0..artifact_interval);

        let mut signal = vec![0.0; n_samples];
        let mut events = Vec::new();

        for (i, i_slot) in signal.iter_mut().enumerate() {
            let t = i as f64 * dt;

            if t >= next_artifact && t < next_artifact + params.artifact_duration {
                // Generate motion artifact (irregular oscillation)
                let phase = (t - next_artifact) / params.artifact_duration;
                let envelope = (PI * phase).sin(); // rise and fall
                let freq = rng.random_range(1.0..5.0);
                *i_slot = params.artifact_amplitude * envelope * (2.0 * PI * freq * t).sin();

                if t == next_artifact || (t - next_artifact) < dt {
                    events.push(Event {
                        time: t,
                        event_type: "motion_artifact".to_string(),
                        amplitude: Some(params.artifact_amplitude),
                        attributes: HashMap::new(),
                    });
                }
            }

            if t >= next_artifact + params.artifact_duration {
                next_artifact += artifact_interval;
            }
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("artifact_frequency".to_string(), params.artifact_frequency);
        gt_params.insert("artifact_amplitude".to_string(), params.artifact_amplitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(
            signal,
            ground_truth,
            params.sampling_rate,
        ))
    }

    fn default_params() -> Self::Parameters {
        PpgArtifactParams {
            duration: 60.0,
            sampling_rate: 100.0,
            artifact_frequency: 5.0, // 5 artifacts per minute
            artifact_duration: 2.0,
            artifact_amplitude: 0.5,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "duration must be positive".to_string(),
            ));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "sampling_rate must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Heart rate recovery generator (post-exercise)
pub struct PulseRateRecoveryGenerator;

#[derive(Debug, Clone)]
pub struct PulseRateRecoveryParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub peak_hr: f64,      // bpm at exercise end
    pub resting_hr: f64,   // bpm at full recovery
    pub recovery_tau: f64, // time constant (seconds)
}

impl SyntheticGenerator for PulseRateRecoveryGenerator {
    type Output = Vec<f64>; // RR intervals
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PulseRateRecoveryParams;

    fn generate(
        &self,
        params: &Self::Parameters,
        seed: u64,
    ) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut rr_intervals = Vec::new();
        let mut t = 0.0;

        while t < params.duration {
            // Exponential recovery: HR(t) = HR_rest + (HR_peak - HR_rest) * exp(-t/tau)
            let hr = params.resting_hr
                + (params.peak_hr - params.resting_hr) * (-t / params.recovery_tau).exp();

            let rr = 60.0 / hr + rng.random_range(-0.02..0.02);
            rr_intervals.push(rr.max(0.3));
            t += rr;
        }

        let mut gt_params = HashMap::new();
        gt_params.insert("peak_hr".to_string(), params.peak_hr);
        gt_params.insert("resting_hr".to_string(), params.resting_hr);
        gt_params.insert("recovery_tau".to_string(), params.recovery_tau);

        // Calculate HR at 1 minute (HRR metric)
        let hr_1min = params.resting_hr
            + (params.peak_hr - params.resting_hr) * (-60.0 / params.recovery_tau).exp();
        let hrr_1min = params.peak_hr - hr_1min;
        gt_params.insert("hrr_1min".to_string(), hrr_1min);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(
            rr_intervals,
            ground_truth,
            params.sampling_rate,
        ))
    }

    fn default_params() -> Self::Parameters {
        PulseRateRecoveryParams {
            duration: 300.0, // 5 minutes
            sampling_rate: 4.0,
            peak_hr: 150.0,
            resting_hr: 70.0,
            recovery_tau: 60.0, // 1 minute time constant
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "duration must be positive".to_string(),
            ));
        }
        if params.peak_hr <= params.resting_hr {
            return Err(crate::GeneratorError::InvalidParameter(
                "peak_hr must be > resting_hr".to_string(),
            ));
        }
        if params.recovery_tau <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter(
                "recovery_tau must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppg_waveform_generation() {
        let generator = PpgWaveformGenerator;
        let params = PpgWaveformGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(
            result.signal.len(),
            (params.duration * params.sampling_rate) as usize
        );
    }

    #[test]
    fn test_ppg_artifact_generation() {
        let generator = PpgArtifactGenerator;
        let params = PpgArtifactGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(
            result.signal.len(),
            (params.duration * params.sampling_rate) as usize
        );
    }

    #[test]
    fn test_heart_rate_recovery() {
        let generator = PulseRateRecoveryGenerator;
        let params = PulseRateRecoveryGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        // Check HR is decreasing over time
        let hr_early = 60.0 / result.signal[0];
        let mut cumulative_time = result.signal[0];

        let mid_idx = result.signal.len() / 2;
        for rr in &result.signal[1..mid_idx] {
            cumulative_time += rr;
        }
        let hr_mid = 60.0 / result.signal[mid_idx];

        assert!(
            hr_early > hr_mid,
            "Heart rate should decrease during recovery"
        );
    }
}
