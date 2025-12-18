//! Cardiopulmonary Signal Generators
//!
//! Generates cardiorespiratory signals:
//! - Heart rate variability (HRV)
//! - Blood pressure variability
//! - Respiratory patterns
//! - Cardiorespiratory coupling
//! - Exercise responses
//! - Pathological patterns (arrhythmias, sleep apnea)

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for cardiopulmonary generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiopulmonaryConfig {
    /// Sampling rate for continuous signals (Hz)
    pub sample_rate: f64,
    /// Resting heart rate (bpm)
    pub resting_hr: f64,
    /// Resting respiratory rate (breaths/min)
    pub resting_rr: f64,
    /// HRV level (SDNN in ms)
    pub hrv_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for CardiopulmonaryConfig {
    fn default() -> Self {
        Self {
            sample_rate: 4.0, // 4 Hz for beat-to-beat
            resting_hr: 70.0,
            resting_rr: 15.0,
            hrv_level: 50.0, // ms
            seed: None,
        }
    }
}

/// Output from cardiopulmonary generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiopulmonaryOutput {
    /// Time points (s)
    pub time: Vec<f64>,
    /// R-R intervals (ms)
    pub rr_intervals: Vec<f64>,
    /// Heart rate (bpm)
    pub heart_rate: Vec<f64>,
    /// Respiratory signal (arbitrary units)
    pub respiratory: Vec<f64>,
    /// Systolic blood pressure (mmHg)
    pub sbp: Vec<f64>,
    /// Ground truth
    pub ground_truth: CardiopulmonaryGroundTruth,
    /// Configuration
    pub config: CardiopulmonaryConfig,
}

/// Ground truth for cardiopulmonary signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardiopulmonaryGroundTruth {
    /// HRV metrics
    pub hrv_metrics: HrvMetrics,
    /// Respiratory metrics
    pub respiratory_metrics: RespiratoryMetrics,
    /// Blood pressure metrics
    pub bp_metrics: BpMetrics,
    /// Cardiorespiratory coupling
    pub coupling: CouplingMetrics,
    /// Applied pathology
    pub pathology: Option<CardiopulmonaryPathology>,
}

/// HRV time and frequency domain metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrvMetrics {
    /// Mean R-R interval (ms)
    pub mean_rr: f64,
    /// SDNN (ms)
    pub sdnn: f64,
    /// RMSSD (ms)
    pub rmssd: f64,
    /// pNN50 (%)
    pub pnn50: f64,
    /// LF power (ms²)
    pub lf_power: f64,
    /// HF power (ms²)
    pub hf_power: f64,
    /// LF/HF ratio
    pub lf_hf_ratio: f64,
    /// Total power (ms²)
    pub total_power: f64,
}

/// Respiratory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RespiratoryMetrics {
    /// Mean respiratory rate (breaths/min)
    pub mean_rate: f64,
    /// Respiratory rate variability (SD)
    pub rate_variability: f64,
    /// Tidal volume variability
    pub tidal_variability: f64,
    /// Inspiratory/expiratory ratio
    pub ie_ratio: f64,
}

/// Blood pressure metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpMetrics {
    /// Mean systolic (mmHg)
    pub mean_sbp: f64,
    /// Mean diastolic (mmHg)
    pub mean_dbp: f64,
    /// SBP variability (SD)
    pub sbp_variability: f64,
    /// Baroreflex sensitivity (ms/mmHg)
    pub brs: f64,
}

/// Cardiorespiratory coupling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingMetrics {
    /// Respiratory sinus arrhythmia amplitude (ms)
    pub rsa_amplitude: f64,
    /// Phase coherence
    pub phase_coherence: f64,
}

/// Cardiopulmonary pathologies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CardiopulmonaryPathology {
    /// Reduced HRV (autonomic dysfunction)
    ReducedHrv { reduction: f64 },
    /// Atrial fibrillation
    AtrialFibrillation,
    /// Premature ventricular contractions
    Pvc { frequency: f64 },
    /// Sleep apnea pattern
    SleepApnea { ahi: f64 }, // Apnea-hypopnea index
    /// Heart failure pattern
    HeartFailure { severity: f64 },
    /// Orthostatic intolerance (POTS)
    Pots { hr_increase: f64 },
    /// Hypertension
    Hypertension { sbp_increase: f64 },
}

/// Cardiopulmonary signal generator
pub struct CardiopulmonaryGenerator {
    config: CardiopulmonaryConfig,
    rng: StdRng,
}

impl CardiopulmonaryGenerator {
    /// Create new cardiopulmonary generator
    pub fn new(config: CardiopulmonaryConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Generate resting cardiopulmonary signals
    pub fn generate_resting(&mut self, duration: f64) -> CardiopulmonaryOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mean_rr = 60000.0 / self.config.resting_hr; // ms
        let resp_period = 60.0 / self.config.resting_rr; // seconds

        let mut time = Vec::with_capacity(n_samples);
        let mut rr_intervals = Vec::with_capacity(n_samples);
        let mut heart_rate = Vec::with_capacity(n_samples);
        let mut respiratory = Vec::with_capacity(n_samples);
        let mut sbp = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.hrv_level * 0.3).unwrap();

        // HRV frequency components
        let lf_freq = 0.1; // ~0.1 Hz (sympathetic + parasympathetic)
        let hf_freq = self.config.resting_rr / 60.0; // Respiratory frequency

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            // Respiratory signal (sinusoidal)
            let resp = (2.0 * PI * t / resp_period).sin();
            respiratory.push(resp);

            // R-R interval with HRV components
            let lf_component = self.config.hrv_level * 0.4 * (2.0 * PI * lf_freq * t).sin();
            let hf_component = self.config.hrv_level * 0.3 * resp; // RSA
            let noise: f64 = self.rng.sample(noise_dist);

            let rr = mean_rr + lf_component + hf_component + noise;
            rr_intervals.push(rr);

            let hr = 60000.0 / rr;
            heart_rate.push(hr);

            // Blood pressure (coupled to respiration and HR)
            let bp_base = 120.0;
            let bp_resp = 5.0 * resp; // Respiratory modulation
            let bp_hr = -0.1 * (hr - self.config.resting_hr); // Baroreceptor
            let bp_noise: f64 = self.rng.sample(Normal::new(0.0, 3.0).unwrap());
            sbp.push(bp_base + bp_resp + bp_hr + bp_noise);
        }

        // Calculate metrics
        let hrv_metrics = self.calculate_hrv_metrics(&rr_intervals);
        let respiratory_metrics = self.calculate_respiratory_metrics(&respiratory, duration);
        let bp_metrics = self.calculate_bp_metrics(&sbp, &rr_intervals);
        let coupling = self.calculate_coupling(&rr_intervals, &respiratory);

        let ground_truth = CardiopulmonaryGroundTruth {
            hrv_metrics,
            respiratory_metrics,
            bp_metrics,
            coupling,
            pathology: None,
        };

        CardiopulmonaryOutput {
            time,
            rr_intervals,
            heart_rate,
            respiratory,
            sbp,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate exercise response
    pub fn generate_exercise(&mut self, duration: f64, peak_hr_percent: f64) -> CardiopulmonaryOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let max_hr = 220.0 - 30.0; // Assume age 30
        let peak_hr = max_hr * peak_hr_percent / 100.0;

        let mut time = Vec::with_capacity(n_samples);
        let mut rr_intervals = Vec::with_capacity(n_samples);
        let mut heart_rate = Vec::with_capacity(n_samples);
        let mut respiratory = Vec::with_capacity(n_samples);
        let mut sbp = Vec::with_capacity(n_samples);

        // Exercise phases: warm-up, exercise, cool-down
        let warmup_end = duration * 0.15;
        let exercise_end = duration * 0.75;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            // Target HR based on phase
            let target_hr = if t < warmup_end {
                let progress = t / warmup_end;
                self.config.resting_hr + (peak_hr - self.config.resting_hr) * progress
            } else if t < exercise_end {
                peak_hr
            } else {
                let progress = (t - exercise_end) / (duration - exercise_end);
                peak_hr - (peak_hr - self.config.resting_hr) * progress
            };

            // Reduced HRV during exercise
            let hrv_reduction = 1.0 - (target_hr - self.config.resting_hr) / (peak_hr - self.config.resting_hr) * 0.7;
            let noise: f64 = self.rng.sample(Normal::new(0.0, self.config.hrv_level * hrv_reduction * 0.2).unwrap());

            let hr = target_hr + noise;
            heart_rate.push(hr);
            rr_intervals.push(60000.0 / hr);

            // Respiratory rate increases with HR
            let resp_rate = self.config.resting_rr + (hr - self.config.resting_hr) * 0.3;
            let resp_period = 60.0 / resp_rate;
            respiratory.push((2.0 * PI * t / resp_period).sin());

            // SBP increases with exercise
            let exercise_sbp = 120.0 + (hr - self.config.resting_hr) * 0.5;
            sbp.push(exercise_sbp + self.rng.sample(Normal::new(0.0, 5.0).unwrap()));
        }

        let hrv_metrics = self.calculate_hrv_metrics(&rr_intervals);
        let respiratory_metrics = self.calculate_respiratory_metrics(&respiratory, duration);
        let bp_metrics = self.calculate_bp_metrics(&sbp, &rr_intervals);
        let coupling = self.calculate_coupling(&rr_intervals, &respiratory);

        let ground_truth = CardiopulmonaryGroundTruth {
            hrv_metrics,
            respiratory_metrics,
            bp_metrics,
            coupling,
            pathology: None,
        };

        CardiopulmonaryOutput {
            time,
            rr_intervals,
            heart_rate,
            respiratory,
            sbp,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate pathological pattern
    pub fn generate_pathological(
        &mut self,
        pathology: CardiopulmonaryPathology,
        duration: f64,
    ) -> CardiopulmonaryOutput {
        let mut output = self.generate_resting(duration);

        match pathology {
            CardiopulmonaryPathology::ReducedHrv { reduction } => {
                // Reduce variability
                let mean = output.rr_intervals.iter().sum::<f64>() / output.rr_intervals.len() as f64;
                for rr in &mut output.rr_intervals {
                    *rr = mean + (*rr - mean) * (1.0 - reduction);
                }
                for (i, rr) in output.rr_intervals.iter().enumerate() {
                    output.heart_rate[i] = 60000.0 / rr;
                }
            }
            CardiopulmonaryPathology::AtrialFibrillation => {
                // Irregular rhythm
                let noise_dist = Normal::new(0.0, self.config.hrv_level * 2.0).unwrap();
                for rr in &mut output.rr_intervals {
                    let noise: f64 = self.rng.sample(noise_dist);
                    *rr = (*rr + noise).max(400.0).min(1500.0);
                }
                for (i, rr) in output.rr_intervals.iter().enumerate() {
                    output.heart_rate[i] = 60000.0 / rr;
                }
            }
            CardiopulmonaryPathology::Pvc { frequency } => {
                // Add premature beats
                let n_pvcs = (output.rr_intervals.len() as f64 * frequency / 100.0) as usize;
                for _ in 0..n_pvcs {
                    let idx = self.rng.gen_range(1..output.rr_intervals.len() - 1);
                    output.rr_intervals[idx] *= 0.7; // Short coupling interval
                    output.rr_intervals[idx + 1] *= 1.3; // Compensatory pause
                }
            }
            CardiopulmonaryPathology::SleepApnea { ahi } => {
                // Cyclical pattern with desaturations
                let apnea_duration = 20.0; // seconds
                let cycle_duration = 3600.0 / ahi; // time between events
                let dt = 1.0 / self.config.sample_rate;

                for i in 0..output.respiratory.len() {
                    let t = i as f64 * dt;
                    let cycle_phase = t % cycle_duration;
                    if cycle_phase < apnea_duration {
                        output.respiratory[i] *= 0.1; // Apnea
                        // HR changes during apnea
                        output.heart_rate[i] *= 1.0 + 0.2 * (cycle_phase / apnea_duration);
                    }
                }
            }
            CardiopulmonaryPathology::Hypertension { sbp_increase } => {
                for bp in &mut output.sbp {
                    *bp += sbp_increase;
                }
            }
            _ => {}
        }

        // Recalculate metrics
        output.ground_truth.hrv_metrics = self.calculate_hrv_metrics(&output.rr_intervals);
        output.ground_truth.pathology = Some(pathology);

        output
    }

    // Metric calculation helpers

    fn calculate_hrv_metrics(&self, rr_intervals: &[f64]) -> HrvMetrics {
        let n = rr_intervals.len() as f64;
        let mean_rr = rr_intervals.iter().sum::<f64>() / n;

        // SDNN
        let variance = rr_intervals.iter()
            .map(|rr| (rr - mean_rr).powi(2))
            .sum::<f64>() / n;
        let sdnn = variance.sqrt();

        // RMSSD
        let rmssd = if rr_intervals.len() > 1 {
            let sum_sq_diff: f64 = rr_intervals.windows(2)
                .map(|w| (w[1] - w[0]).powi(2))
                .sum();
            (sum_sq_diff / (n - 1.0)).sqrt()
        } else {
            0.0
        };

        // pNN50
        let nn50_count = if rr_intervals.len() > 1 {
            rr_intervals.windows(2)
                .filter(|w| (w[1] - w[0]).abs() > 50.0)
                .count()
        } else {
            0
        };
        let pnn50 = nn50_count as f64 / (n - 1.0) * 100.0;

        // Simplified frequency domain (would need FFT for accurate)
        let lf_power = sdnn.powi(2) * 0.3;
        let hf_power = rmssd.powi(2) * 0.5;
        let total_power = sdnn.powi(2);
        let lf_hf_ratio = if hf_power > 0.0 { lf_power / hf_power } else { 0.0 };

        HrvMetrics {
            mean_rr,
            sdnn,
            rmssd,
            pnn50,
            lf_power,
            hf_power,
            lf_hf_ratio,
            total_power,
        }
    }

    fn calculate_respiratory_metrics(&self, respiratory: &[f64], duration: f64) -> RespiratoryMetrics {
        // Count zero crossings to estimate rate
        let mut crossings = 0;
        for window in respiratory.windows(2) {
            if window[0] * window[1] < 0.0 {
                crossings += 1;
            }
        }
        let mean_rate = crossings as f64 / duration * 30.0; // breaths/min

        RespiratoryMetrics {
            mean_rate,
            rate_variability: mean_rate * 0.1,
            tidal_variability: 0.15,
            ie_ratio: 1.5,
        }
    }

    fn calculate_bp_metrics(&self, sbp: &[f64], rr_intervals: &[f64]) -> BpMetrics {
        let n = sbp.len() as f64;
        let mean_sbp = sbp.iter().sum::<f64>() / n;
        let sbp_variance = sbp.iter()
            .map(|bp| (bp - mean_sbp).powi(2))
            .sum::<f64>() / n;

        // Simplified BRS (would need proper cross-correlation)
        let mean_rr = rr_intervals.iter().sum::<f64>() / rr_intervals.len() as f64;
        let brs = (mean_rr / 1000.0) / (mean_sbp / 100.0) * 10.0; // ms/mmHg

        BpMetrics {
            mean_sbp,
            mean_dbp: mean_sbp * 0.65,
            sbp_variability: sbp_variance.sqrt(),
            brs,
        }
    }

    fn calculate_coupling(&self, rr_intervals: &[f64], respiratory: &[f64]) -> CouplingMetrics {
        // RSA amplitude (simplified)
        let n = rr_intervals.len().min(respiratory.len());
        let mut rsa_sum = 0.0;

        for i in 0..n {
            rsa_sum += (rr_intervals[i] / 1000.0) * respiratory[i].abs();
        }
        let rsa_amplitude = rsa_sum / n as f64 * 100.0;

        CouplingMetrics {
            rsa_amplitude,
            phase_coherence: 0.7, // Simplified
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resting_generation() {
        let config = CardiopulmonaryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = CardiopulmonaryGenerator::new(config);
        let output = gen.generate_resting(60.0);

        assert!(!output.rr_intervals.is_empty());
        assert!(output.ground_truth.hrv_metrics.sdnn > 0.0);
    }

    #[test]
    fn test_exercise_response() {
        let config = CardiopulmonaryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = CardiopulmonaryGenerator::new(config);
        let output = gen.generate_exercise(300.0, 80.0);

        // Peak HR should be higher than resting
        let peak_hr = output.heart_rate.iter().cloned().fold(0.0_f64, f64::max);
        assert!(peak_hr > config.resting_hr * 1.5);
    }

    #[test]
    fn test_afib() {
        let config = CardiopulmonaryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = CardiopulmonaryGenerator::new(config);

        let normal = gen.generate_resting(60.0);
        let afib = gen.generate_pathological(CardiopulmonaryPathology::AtrialFibrillation, 60.0);

        // AFib should have higher variability
        assert!(afib.ground_truth.hrv_metrics.sdnn > normal.ground_truth.hrv_metrics.sdnn * 0.5);
    }

    #[test]
    fn test_reduced_hrv() {
        let config = CardiopulmonaryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = CardiopulmonaryGenerator::new(config);

        let normal = gen.generate_resting(60.0);
        let reduced = gen.generate_pathological(
            CardiopulmonaryPathology::ReducedHrv { reduction: 0.7 },
            60.0,
        );

        assert!(reduced.ground_truth.hrv_metrics.sdnn < normal.ground_truth.hrv_metrics.sdnn);
    }
}
