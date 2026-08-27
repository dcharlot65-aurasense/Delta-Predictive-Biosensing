//! EMG (Electromyography) Analysis Module
//!
//! Provides comprehensive EMG signal analysis including:
//! - Muscle activation detection
//! - Fatigue analysis (median frequency shift)
//! - MVC normalization
//! - Burst detection and timing
//! - Co-contraction analysis

use crate::error::{DpbError, Result};
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

/// EMG burst event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmgBurst {
    /// Onset time (samples)
    pub onset: usize,
    /// Offset time (samples)
    pub offset: usize,
    /// Peak amplitude
    pub peak_amplitude: f64,
    /// Mean amplitude during burst
    pub mean_amplitude: f64,
    /// RMS amplitude during burst
    pub rms_amplitude: f64,
    /// Duration (seconds)
    pub duration: f64,
    /// Integrated EMG (area)
    pub iemg: f64,
}

/// Fatigue analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FatigueMetrics {
    /// Initial median frequency (Hz)
    pub initial_mdf: f64,
    /// Final median frequency (Hz)
    pub final_mdf: f64,
    /// Median frequency slope (Hz/s)
    pub mdf_slope: f64,
    /// Initial mean frequency (Hz)
    pub initial_mnf: f64,
    /// Final mean frequency (Hz)
    pub final_mnf: f64,
    /// Mean frequency slope (Hz/s)
    pub mnf_slope: f64,
    /// Initial RMS amplitude
    pub initial_rms: f64,
    /// Final RMS amplitude
    pub final_rms: f64,
    /// RMS slope
    pub rms_slope: f64,
    /// Fatigue index (0-1)
    pub fatigue_index: f64,
}

/// EMG frequency domain metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmgFrequencyMetrics {
    /// Mean frequency (Hz)
    pub mean_frequency: f64,
    /// Median frequency (Hz)
    pub median_frequency: f64,
    /// Peak frequency (Hz)
    pub peak_frequency: f64,
    /// Total power
    pub total_power: f64,
    /// Mean power
    pub mean_power: f64,
    /// Power in low band (20-60 Hz)
    pub low_band_power: f64,
    /// Power in mid band (60-120 Hz)
    pub mid_band_power: f64,
    /// Power in high band (120-250 Hz)
    pub high_band_power: f64,
}

/// EMG time domain metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmgTimeMetrics {
    /// Root mean square
    pub rms: f64,
    /// Mean absolute value
    pub mav: f64,
    /// Integrated EMG
    pub iemg: f64,
    /// Variance
    pub variance: f64,
    /// Waveform length
    pub waveform_length: f64,
    /// Zero crossing rate
    pub zero_crossings: usize,
    /// Slope sign changes
    pub slope_sign_changes: usize,
    /// Willison amplitude (threshold crossings)
    pub willison_amplitude: usize,
}

/// EMG signal analyzer
pub struct EmgAnalyzer {
    sample_rate: f64,
    /// Maximum voluntary contraction amplitude for normalization
    mvc_amplitude: Option<f64>,
    /// Burst detection threshold (fraction of max)
    burst_threshold: f64,
    /// Minimum burst duration (seconds)
    min_burst_duration: f64,
}

impl EmgAnalyzer {
    /// Create a new EMG analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            mvc_amplitude: None,
            burst_threshold: 0.1,
            min_burst_duration: 0.05,
        }
    }

    /// Set MVC amplitude for normalization
    pub fn set_mvc(&mut self, mvc: f64) {
        self.mvc_amplitude = Some(mvc);
    }

    /// Set burst detection parameters
    pub fn set_burst_params(&mut self, threshold: f64, min_duration: f64) {
        self.burst_threshold = threshold;
        self.min_burst_duration = min_duration;
    }

    /// Calculate time domain metrics
    pub fn time_domain_metrics(&self, signal: ArrayView1<f64>) -> EmgTimeMetrics {
        let n = signal.len() as f64;

        // RMS
        let rms = (signal.iter().map(|x| x * x).sum::<f64>() / n).sqrt();

        // Mean Absolute Value
        let mav = signal.iter().map(|x| x.abs()).sum::<f64>() / n;

        // Integrated EMG
        let iemg = signal.iter().map(|x| x.abs()).sum::<f64>() / self.sample_rate;

        // Variance
        let mean = signal.mean().unwrap_or(0.0);
        let variance = signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

        // Waveform Length
        let mut waveform_length = 0.0;
        for i in 1..signal.len() {
            waveform_length += (signal[i] - signal[i - 1]).abs();
        }

        // Zero Crossings
        let mut zero_crossings = 0;
        for i in 1..signal.len() {
            if (signal[i - 1] < 0.0 && signal[i] >= 0.0)
                || (signal[i - 1] >= 0.0 && signal[i] < 0.0)
            {
                zero_crossings += 1;
            }
        }

        // Slope Sign Changes
        let mut slope_sign_changes = 0;
        if signal.len() >= 3 {
            for i in 1..signal.len() - 1 {
                let diff1 = signal[i] - signal[i - 1];
                let diff2 = signal[i + 1] - signal[i];
                if diff1 * diff2 < 0.0 {
                    slope_sign_changes += 1;
                }
            }
        }

        // Willison Amplitude (threshold = 10% of max)
        let max_amp = signal.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let threshold = max_amp * 0.1;
        let mut willison_amplitude = 0;
        for i in 1..signal.len() {
            if (signal[i] - signal[i - 1]).abs() > threshold {
                willison_amplitude += 1;
            }
        }

        EmgTimeMetrics {
            rms,
            mav,
            iemg,
            variance,
            waveform_length,
            zero_crossings,
            slope_sign_changes,
            willison_amplitude,
        }
    }

    /// Calculate frequency domain metrics using simplified PSD estimation
    pub fn frequency_domain_metrics(&self, signal: ArrayView1<f64>) -> Result<EmgFrequencyMetrics> {
        if signal.len() < 64 {
            return Err(DpbError::InvalidDimensions(
                "Signal too short for frequency analysis".to_string(),
            ));
        }

        // Simple power spectral density using periodogram
        let n = signal.len();
        let freq_resolution = self.sample_rate / n as f64;

        // Calculate power at each frequency using Goertzel-like approach
        let max_freq = (self.sample_rate / 2.0).min(500.0);
        let num_freqs = ((max_freq / freq_resolution) as usize).min(n / 2);

        let mut powers = Vec::with_capacity(num_freqs);
        let mut freqs = Vec::with_capacity(num_freqs);

        for k in 1..=num_freqs {
            let freq = k as f64 * freq_resolution;
            let omega = 2.0 * std::f64::consts::PI * freq / self.sample_rate;

            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for (i, &sample) in signal.iter().enumerate() {
                real_sum += sample * (omega * i as f64).cos();
                imag_sum += sample * (omega * i as f64).sin();
            }

            let power = (real_sum.powi(2) + imag_sum.powi(2)) / n as f64;
            powers.push(power);
            freqs.push(freq);
        }

        let total_power: f64 = powers.iter().sum();

        // Mean frequency
        let mean_frequency = if total_power > 0.0 {
            powers
                .iter()
                .zip(freqs.iter())
                .map(|(p, f)| p * f)
                .sum::<f64>()
                / total_power
        } else {
            0.0
        };

        // Median frequency
        let mut cumsum = 0.0;
        let half_power = total_power / 2.0;
        let mut median_frequency = 0.0;
        for (power, freq) in powers.iter().zip(freqs.iter()) {
            cumsum += power;
            if cumsum >= half_power {
                median_frequency = *freq;
                break;
            }
        }

        // Peak frequency
        let peak_idx = powers
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_frequency = freqs.get(peak_idx).copied().unwrap_or(0.0);

        // Band powers
        let low_band_power: f64 = powers
            .iter()
            .zip(freqs.iter())
            .filter(|(_, f)| **f >= 20.0 && **f < 60.0)
            .map(|(p, _)| p)
            .sum();

        let mid_band_power: f64 = powers
            .iter()
            .zip(freqs.iter())
            .filter(|(_, f)| **f >= 60.0 && **f < 120.0)
            .map(|(p, _)| p)
            .sum();

        let high_band_power: f64 = powers
            .iter()
            .zip(freqs.iter())
            .filter(|(_, f)| **f >= 120.0 && **f < 250.0)
            .map(|(p, _)| p)
            .sum();

        Ok(EmgFrequencyMetrics {
            mean_frequency,
            median_frequency,
            peak_frequency,
            total_power,
            mean_power: total_power / num_freqs as f64,
            low_band_power,
            mid_band_power,
            high_band_power,
        })
    }

    /// Calculate rectified and smoothed envelope
    pub fn envelope(&self, signal: ArrayView1<f64>, window_ms: f64) -> Array1<f64> {
        let window_size = (window_ms / 1000.0 * self.sample_rate) as usize;
        let window_size = window_size.max(1);

        let rectified: Vec<f64> = signal.iter().map(|x| x.abs()).collect();
        let mut envelope = Array1::zeros(signal.len());

        for i in 0..signal.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(signal.len());
            let sum: f64 = rectified[start..end].iter().sum();
            envelope[i] = sum / (end - start) as f64;
        }

        envelope
    }

    /// Detect muscle activation bursts
    pub fn detect_bursts(&self, signal: ArrayView1<f64>) -> Vec<EmgBurst> {
        let envelope = self.envelope(signal, 20.0); // 20ms smoothing
        let max_envelope = envelope.iter().cloned().fold(0.0f64, f64::max);
        let threshold = max_envelope * self.burst_threshold;

        let min_samples = (self.min_burst_duration * self.sample_rate) as usize;

        let mut bursts = Vec::new();
        let mut in_burst = false;
        let mut burst_start = 0;

        for i in 0..envelope.len() {
            if envelope[i] > threshold && !in_burst {
                in_burst = true;
                burst_start = i;
            } else if envelope[i] <= threshold && in_burst {
                in_burst = false;
                let burst_end = i;

                if burst_end - burst_start >= min_samples {
                    let burst_signal = signal.slice(ndarray::s![burst_start..burst_end]);

                    let peak_amplitude =
                        burst_signal.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
                    let mean_amplitude = burst_signal.iter().map(|x| x.abs()).sum::<f64>()
                        / burst_signal.len() as f64;
                    let rms_amplitude = (burst_signal.iter().map(|x| x * x).sum::<f64>()
                        / burst_signal.len() as f64)
                        .sqrt();
                    let iemg = burst_signal.iter().map(|x| x.abs()).sum::<f64>() / self.sample_rate;
                    let duration = (burst_end - burst_start) as f64 / self.sample_rate;

                    bursts.push(EmgBurst {
                        onset: burst_start,
                        offset: burst_end,
                        peak_amplitude,
                        mean_amplitude,
                        rms_amplitude,
                        duration,
                        iemg,
                    });
                }
            }
        }

        // Handle burst that extends to end of signal
        if in_burst && signal.len() - burst_start >= min_samples {
            let burst_signal = signal.slice(ndarray::s![burst_start..]);
            let peak_amplitude = burst_signal.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            let mean_amplitude =
                burst_signal.iter().map(|x| x.abs()).sum::<f64>() / burst_signal.len() as f64;
            let rms_amplitude = (burst_signal.iter().map(|x| x * x).sum::<f64>()
                / burst_signal.len() as f64)
                .sqrt();
            let iemg = burst_signal.iter().map(|x| x.abs()).sum::<f64>() / self.sample_rate;
            let duration = (signal.len() - burst_start) as f64 / self.sample_rate;

            bursts.push(EmgBurst {
                onset: burst_start,
                offset: signal.len(),
                peak_amplitude,
                mean_amplitude,
                rms_amplitude,
                duration,
                iemg,
            });
        }

        bursts
    }

    /// Analyze fatigue from sustained contraction
    pub fn analyze_fatigue(
        &self,
        signal: ArrayView1<f64>,
        window_seconds: f64,
    ) -> Result<FatigueMetrics> {
        let window_size = (window_seconds * self.sample_rate) as usize;
        let num_windows = signal.len() / window_size;

        if num_windows < 2 {
            return Err(DpbError::DataValidation(
                "Need at least 2 windows for fatigue analysis".to_string(),
            ));
        }

        let mut mdfs = Vec::new();
        let mut mnfs = Vec::new();
        let mut rms_values = Vec::new();

        for i in 0..num_windows {
            let start = i * window_size;
            let end = start + window_size;
            let window = signal.slice(ndarray::s![start..end]);

            let freq_metrics = self.frequency_domain_metrics(window)?;
            mdfs.push(freq_metrics.median_frequency);
            mnfs.push(freq_metrics.mean_frequency);

            let time_metrics = self.time_domain_metrics(window);
            rms_values.push(time_metrics.rms);
        }

        let initial_mdf = mdfs[0];
        let final_mdf = mdfs[num_windows - 1];
        let mdf_slope = (final_mdf - initial_mdf) / ((num_windows - 1) as f64 * window_seconds);

        let initial_mnf = mnfs[0];
        let final_mnf = mnfs[num_windows - 1];
        let mnf_slope = (final_mnf - initial_mnf) / ((num_windows - 1) as f64 * window_seconds);

        let initial_rms = rms_values[0];
        let final_rms = rms_values[num_windows - 1];
        let rms_slope = (final_rms - initial_rms) / ((num_windows - 1) as f64 * window_seconds);

        // Fatigue index based on MDF decrease
        let mdf_decrease_percent = if initial_mdf > 0.0 {
            ((initial_mdf - final_mdf) / initial_mdf).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(FatigueMetrics {
            initial_mdf,
            final_mdf,
            mdf_slope,
            initial_mnf,
            final_mnf,
            mnf_slope,
            initial_rms,
            final_rms,
            rms_slope,
            fatigue_index: mdf_decrease_percent,
        })
    }

    /// Normalize signal to MVC percentage
    pub fn normalize_to_mvc(&self, signal: ArrayView1<f64>) -> Result<Array1<f64>> {
        let mvc = self
            .mvc_amplitude
            .ok_or_else(|| DpbError::Config("MVC amplitude not set".to_string()))?;

        if mvc <= 0.0 {
            return Err(DpbError::InvalidParameter(
                "MVC amplitude must be positive".to_string(),
            ));
        }

        Ok(signal.mapv(|x| (x.abs() / mvc) * 100.0))
    }

    /// Calculate co-contraction index between agonist and antagonist
    pub fn cocontraction_index(
        &self,
        agonist: ArrayView1<f64>,
        antagonist: ArrayView1<f64>,
    ) -> Result<f64> {
        if agonist.len() != antagonist.len() {
            return Err(DpbError::InvalidDimensions(
                "Signals must have same length".to_string(),
            ));
        }

        let agonist_env = self.envelope(agonist, 20.0);
        let antagonist_env = self.envelope(antagonist, 20.0);

        // Co-contraction = 2 * min(agonist, antagonist) / (agonist + antagonist)
        let mut cocontraction_sum = 0.0;
        let mut total_sum = 0.0;

        for i in 0..agonist.len() {
            let min_val = agonist_env[i].min(antagonist_env[i]);
            let sum_val = agonist_env[i] + antagonist_env[i];
            cocontraction_sum += 2.0 * min_val;
            total_sum += sum_val;
        }

        if total_sum > 0.0 {
            Ok(cocontraction_sum / total_sum)
        } else {
            Ok(0.0)
        }
    }

    /// Estimate muscle activation onset using threshold method
    pub fn detect_onset(&self, signal: ArrayView1<f64>, baseline_samples: usize) -> Option<usize> {
        if signal.len() <= baseline_samples {
            return None;
        }

        let baseline = signal.slice(ndarray::s![..baseline_samples]);
        let baseline_mean = baseline.iter().map(|x| x.abs()).sum::<f64>() / baseline_samples as f64;
        let baseline_std = {
            let variance = baseline
                .iter()
                .map(|x| (x.abs() - baseline_mean).powi(2))
                .sum::<f64>()
                / baseline_samples as f64;
            variance.sqrt()
        };

        let threshold = baseline_mean + 3.0 * baseline_std;

        let envelope = self.envelope(signal, 10.0);
        envelope
            .iter()
            .enumerate()
            .skip(baseline_samples)
            .find(|(_, v)| **v > threshold)
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_synthetic_emg(sample_rate: f64, duration: f64, burst_freq: f64) -> Array1<f64> {
        let n_samples = (sample_rate * duration) as usize;
        let mut signal = Array1::zeros(n_samples);

        // Add noise
        for i in 0..n_samples {
            signal[i] = (i as f64 * 0.1).sin() * 0.01; // Low baseline noise
        }

        // Add bursts
        let burst_duration = 0.2; // 200ms bursts
        let burst_samples = (burst_duration * sample_rate) as usize;
        let period = (sample_rate / burst_freq) as usize;

        for start in (0..n_samples).step_by(period) {
            for i in 0..burst_samples.min(n_samples - start) {
                let t = i as f64 / sample_rate;
                // Simulated EMG during contraction
                signal[start + i] +=
                    (100.0 * t).sin() * 0.5 + (150.0 * t).cos() * 0.3 + (200.0 * t).sin() * 0.2;
            }
        }

        signal
    }

    #[test]
    fn test_time_domain_metrics() {
        let sample_rate = 1000.0;
        let signal = generate_synthetic_emg(sample_rate, 1.0, 2.0);
        let analyzer = EmgAnalyzer::new(sample_rate);

        let metrics = analyzer.time_domain_metrics(signal.view());

        assert!(metrics.rms > 0.0);
        assert!(metrics.mav > 0.0);
        assert!(metrics.iemg > 0.0);
    }

    #[test]
    fn test_frequency_domain_metrics() {
        let sample_rate = 1000.0;
        let signal = generate_synthetic_emg(sample_rate, 1.0, 2.0);
        let analyzer = EmgAnalyzer::new(sample_rate);

        let metrics = analyzer.frequency_domain_metrics(signal.view()).unwrap();

        assert!(metrics.mean_frequency > 0.0);
        assert!(metrics.median_frequency > 0.0);
        assert!(metrics.total_power > 0.0);
    }

    #[test]
    fn test_envelope() {
        let sample_rate = 1000.0;
        let signal = generate_synthetic_emg(sample_rate, 1.0, 2.0);
        let analyzer = EmgAnalyzer::new(sample_rate);

        let env = analyzer.envelope(signal.view(), 20.0);

        assert_eq!(env.len(), signal.len());
        assert!(env.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_burst_detection() {
        let sample_rate = 1000.0;
        let signal = generate_synthetic_emg(sample_rate, 2.0, 2.0);
        let analyzer = EmgAnalyzer::new(sample_rate);

        let bursts = analyzer.detect_bursts(signal.view());

        assert!(!bursts.is_empty());
        for burst in &bursts {
            assert!(burst.duration > 0.0);
            assert!(burst.peak_amplitude > 0.0);
        }
    }

    #[test]
    fn test_fatigue_analysis() {
        let sample_rate = 1000.0;
        // Simulate a sustained contraction with fatigue
        let mut signal = Array1::zeros(10000);
        for i in 0..10000 {
            let t = i as f64 / sample_rate;
            let fatigue_factor = 1.0 - 0.3 * (t / 10.0); // Decreasing frequency over time
            signal[i] = (100.0 * fatigue_factor * t).sin() * 0.5;
        }

        let analyzer = EmgAnalyzer::new(sample_rate);
        let fatigue = analyzer.analyze_fatigue(signal.view(), 1.0).unwrap();

        assert!(fatigue.initial_mdf > 0.0 || fatigue.final_mdf > 0.0);
    }

    #[test]
    fn test_mvc_normalization() {
        let sample_rate = 1000.0;
        let signal = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let mut analyzer = EmgAnalyzer::new(sample_rate);
        analyzer.set_mvc(1.0);

        let normalized = analyzer.normalize_to_mvc(signal.view()).unwrap();

        assert!((normalized[0] - 10.0).abs() < 0.1);
        assert!((normalized[4] - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_cocontraction() {
        let sample_rate = 1000.0;
        let agonist = Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8, 0.9]);
        let antagonist = Array1::from_vec(vec![0.1, 0.2, 0.1, 0.2, 0.1]);
        let analyzer = EmgAnalyzer::new(sample_rate);

        let cci = analyzer
            .cocontraction_index(agonist.view(), antagonist.view())
            .unwrap();

        assert!((0.0..=1.0).contains(&cci));
    }
}
