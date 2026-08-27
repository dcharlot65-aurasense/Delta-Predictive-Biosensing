//! PPG (Photoplethysmography) Analysis Module
//!
//! Provides comprehensive PPG signal analysis including:
//! - Pulse wave morphology analysis
//! - SpO2 estimation
//! - Perfusion index calculation
//! - Pulse rate variability (PRV)
//! - Arterial stiffness indices

use crate::error::{DpbError, Result};
use ndarray::{ArrayView1};
use serde::{Deserialize, Serialize};

/// PPG pulse wave features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulseWaveFeatures {
    /// Systolic peak amplitude
    pub systolic_amplitude: f64,
    /// Diastolic peak amplitude
    pub diastolic_amplitude: f64,
    /// Dicrotic notch amplitude
    pub dicrotic_notch_amplitude: f64,
    /// Systolic peak time (relative to pulse start)
    pub systolic_time: f64,
    /// Diastolic peak time
    pub diastolic_time: f64,
    /// Pulse duration
    pub pulse_duration: f64,
    /// Augmentation index (AI)
    pub augmentation_index: f64,
    /// Reflection index
    pub reflection_index: f64,
    /// Stiffness index (SI)
    pub stiffness_index: f64,
    /// Crest time ratio
    pub crest_time_ratio: f64,
}

impl Default for PulseWaveFeatures {
    fn default() -> Self {
        Self {
            systolic_amplitude: 0.0,
            diastolic_amplitude: 0.0,
            dicrotic_notch_amplitude: 0.0,
            systolic_time: 0.0,
            diastolic_time: 0.0,
            pulse_duration: 0.0,
            augmentation_index: 0.0,
            reflection_index: 0.0,
            stiffness_index: 0.0,
            crest_time_ratio: 0.0,
        }
    }
}

/// SpO2 estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpO2Result {
    /// Estimated SpO2 percentage
    pub spo2: f64,
    /// R-value (ratio of ratios)
    pub r_value: f64,
    /// Perfusion index (%)
    pub perfusion_index: f64,
    /// Signal quality (0-1)
    pub quality: f64,
}

/// Pulse rate variability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrvMetrics {
    /// Mean pulse interval (ms)
    pub mean_pi: f64,
    /// Standard deviation of pulse intervals (ms)
    pub sdnn: f64,
    /// Root mean square of successive differences (ms)
    pub rmssd: f64,
    /// Percentage of successive differences > 50ms
    pub pnn50: f64,
    /// Low frequency power (0.04-0.15 Hz)
    pub lf_power: f64,
    /// High frequency power (0.15-0.4 Hz)
    pub hf_power: f64,
    /// LF/HF ratio
    pub lf_hf_ratio: f64,
}

/// PPG signal analyzer
pub struct PpgAnalyzer {
    sample_rate: f64,
}

impl PpgAnalyzer {
    /// Create a new PPG analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }

    /// Detect pulse peaks in PPG signal
    pub fn detect_peaks(&self, signal: ArrayView1<f64>) -> Result<Vec<usize>> {
        if signal.len() < 10 {
            return Err(DpbError::InvalidDimensions(
                "Signal too short for peak detection".to_string(),
            ));
        }

        let mut peaks = Vec::new();

        // Adaptive threshold based on signal statistics
        let mean = signal.mean().unwrap_or(0.0);
        let std = signal.std(0.0);
        let threshold = mean + 0.3 * std;

        // Minimum peak distance (~40 bpm max)
        let min_distance = (self.sample_rate * 0.4) as usize;

        let mut last_peak = 0;
        for i in 2..signal.len() - 2 {
            if signal[i] > threshold
                && signal[i] > signal[i - 1]
                && signal[i] > signal[i - 2]
                && signal[i] > signal[i + 1]
                && signal[i] > signal[i + 2]
                && (peaks.is_empty() || i - last_peak >= min_distance)
            {
                peaks.push(i);
                last_peak = i;
            }
        }

        Ok(peaks)
    }

    /// Detect pulse onsets (foot of the pulse wave)
    pub fn detect_onsets(&self, signal: ArrayView1<f64>, peaks: &[usize]) -> Vec<usize> {
        let mut onsets = Vec::new();

        for &peak in peaks {
            // Search backwards from peak for minimum
            let search_start = peak.saturating_sub((self.sample_rate * 0.3) as usize);
            let mut min_idx = search_start;
            let mut min_val = signal[search_start];

            for i in search_start..peak {
                if signal[i] < min_val {
                    min_val = signal[i];
                    min_idx = i;
                }
            }

            onsets.push(min_idx);
        }

        onsets
    }

    /// Calculate pulse intervals from peaks
    pub fn calculate_pulse_intervals(&self, peaks: &[usize]) -> Vec<f64> {
        peaks
            .windows(2)
            .map(|w| (w[1] - w[0]) as f64 / self.sample_rate * 1000.0)
            .collect()
    }

    /// Calculate instantaneous heart rate from pulse intervals
    pub fn calculate_heart_rate(&self, pulse_intervals: &[f64]) -> Vec<f64> {
        pulse_intervals
            .iter()
            .map(|pi| 60000.0 / pi)
            .collect()
    }

    /// Analyze pulse wave morphology for a single pulse
    pub fn analyze_pulse_wave(
        &self,
        signal: ArrayView1<f64>,
        onset: usize,
        next_onset: usize,
    ) -> Result<PulseWaveFeatures> {
        if next_onset <= onset {
            return Err(DpbError::InvalidDimensions(
                "Invalid pulse boundaries".to_string(),
            ));
        }

        let pulse = signal.slice(ndarray::s![onset..next_onset]);
        let pulse_len = pulse.len();

        if pulse_len < 5 {
            return Ok(PulseWaveFeatures::default());
        }

        // Find systolic peak (maximum in first 40% of pulse)
        let systolic_search_end = (pulse_len as f64 * 0.4) as usize;
        let mut systolic_idx = 0;
        let mut systolic_amp = pulse[0];
        for i in 0..systolic_search_end.min(pulse_len) {
            if pulse[i] > systolic_amp {
                systolic_amp = pulse[i];
                systolic_idx = i;
            }
        }

        // Find dicrotic notch (minimum in 40-70% of pulse)
        let notch_start = (pulse_len as f64 * 0.4) as usize;
        let notch_end = (pulse_len as f64 * 0.7) as usize;
        let mut notch_idx = notch_start;
        let mut notch_amp = pulse[notch_start.min(pulse_len - 1)];
        for i in notch_start..notch_end.min(pulse_len) {
            if pulse[i] < notch_amp {
                notch_amp = pulse[i];
                notch_idx = i;
            }
        }

        // Find diastolic peak (maximum after notch)
        let mut diastolic_idx = notch_idx;
        let mut diastolic_amp = notch_amp;
        for i in notch_idx..pulse_len {
            if pulse[i] > diastolic_amp {
                diastolic_amp = pulse[i];
                diastolic_idx = i;
            }
        }

        let pulse_duration = pulse_len as f64 / self.sample_rate;
        let systolic_time = systolic_idx as f64 / self.sample_rate;
        let diastolic_time = diastolic_idx as f64 / self.sample_rate;

        // Calculate indices
        let baseline = pulse[0];
        let systolic_height = systolic_amp - baseline;
        let diastolic_height = diastolic_amp - baseline;

        let augmentation_index = if systolic_height > 0.0 {
            (diastolic_height / systolic_height) * 100.0
        } else {
            0.0
        };

        let reflection_index = if systolic_height > 0.0 {
            ((systolic_amp - notch_amp) / systolic_height) * 100.0
        } else {
            0.0
        };

        // Stiffness index (height / transit time)
        let transit_time = diastolic_time - systolic_time;
        let stiffness_index = if transit_time > 0.0 {
            1.0 / transit_time // Simplified, normally uses body height
        } else {
            0.0
        };

        let crest_time_ratio = systolic_time / pulse_duration;

        Ok(PulseWaveFeatures {
            systolic_amplitude: systolic_amp,
            diastolic_amplitude: diastolic_amp,
            dicrotic_notch_amplitude: notch_amp,
            systolic_time,
            diastolic_time,
            pulse_duration,
            augmentation_index,
            reflection_index,
            stiffness_index,
            crest_time_ratio,
        })
    }

    /// Estimate SpO2 from red and infrared PPG signals
    pub fn estimate_spo2(
        &self,
        red_signal: ArrayView1<f64>,
        ir_signal: ArrayView1<f64>,
    ) -> Result<SpO2Result> {
        if red_signal.len() != ir_signal.len() {
            return Err(DpbError::InvalidDimensions(
                "Red and IR signals must have same length".to_string(),
            ));
        }

        // Calculate AC and DC components
        let red_ac = red_signal.std(0.0);
        let red_dc = red_signal.mean().unwrap_or(1.0);
        let ir_ac = ir_signal.std(0.0);
        let ir_dc = ir_signal.mean().unwrap_or(1.0);

        // Calculate ratio of ratios
        let r_value = if ir_ac * red_dc > 0.0 {
            (red_ac / red_dc) / (ir_ac / ir_dc)
        } else {
            0.0
        };

        // Empirical SpO2 calibration curve
        // SpO2 = 110 - 25 * R (typical linear approximation)
        let spo2 = (110.0 - 25.0 * r_value).clamp(70.0, 100.0);

        // Perfusion index (AC/DC ratio in %)
        let perfusion_index = if ir_dc > 0.0 {
            (ir_ac / ir_dc) * 100.0
        } else {
            0.0
        };

        // Signal quality based on perfusion index
        let quality = (perfusion_index / 5.0).clamp(0.0, 1.0);

        Ok(SpO2Result {
            spo2,
            r_value,
            perfusion_index,
            quality,
        })
    }

    /// Calculate pulse rate variability metrics
    pub fn calculate_prv(&self, pulse_intervals: &[f64]) -> Result<PrvMetrics> {
        if pulse_intervals.len() < 10 {
            return Err(DpbError::DataValidation(
                "Need at least 10 pulse intervals for PRV analysis".to_string(),
            ));
        }

        let n = pulse_intervals.len() as f64;

        // Mean pulse interval
        let mean_pi: f64 = pulse_intervals.iter().sum::<f64>() / n;

        // SDNN
        let variance: f64 = pulse_intervals
            .iter()
            .map(|x| (x - mean_pi).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let sdnn = variance.sqrt();

        // RMSSD
        let successive_diffs: Vec<f64> = pulse_intervals
            .windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .collect();
        let rmssd = (successive_diffs.iter().sum::<f64>() / successive_diffs.len() as f64).sqrt();

        // pNN50
        let nn50_count = pulse_intervals
            .windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 50.0)
            .count();
        let pnn50 = (nn50_count as f64 / (pulse_intervals.len() - 1) as f64) * 100.0;

        // Frequency domain (simplified using Welch's method approximation)
        let lf_power = self.estimate_band_power(pulse_intervals, 0.04, 0.15);
        let hf_power = self.estimate_band_power(pulse_intervals, 0.15, 0.4);
        let lf_hf_ratio = if hf_power > 0.0 { lf_power / hf_power } else { 0.0 };

        Ok(PrvMetrics {
            mean_pi,
            sdnn,
            rmssd,
            pnn50,
            lf_power,
            hf_power,
            lf_hf_ratio,
        })
    }

    /// Estimate power in a frequency band (simplified)
    fn estimate_band_power(&self, intervals: &[f64], low_freq: f64, high_freq: f64) -> f64 {
        if intervals.len() < 4 {
            return 0.0;
        }

        // Simplified band power estimation using variance in band
        // In practice, would use proper Welch's method or Lomb-Scargle
        let mean: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let mean_rate = 1000.0 / mean; // Convert to Hz

        // Approximate: LF captures slower variations, HF captures faster
        let center_freq = (low_freq + high_freq) / 2.0;
        let variance: f64 = intervals
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / intervals.len() as f64;

        // Weight by expected contribution at this frequency band
        variance * (1.0 - (center_freq - mean_rate * 0.1).abs().min(1.0))
    }

    /// Calculate perfusion index from PPG signal
    pub fn calculate_perfusion_index(&self, signal: ArrayView1<f64>) -> f64 {
        let dc = signal.mean().unwrap_or(1.0);
        let ac = signal.std(0.0);

        if dc > 0.0 {
            (ac / dc) * 100.0
        } else {
            0.0
        }
    }

    /// Detect motion artifacts in PPG signal
    pub fn detect_artifacts(&self, signal: ArrayView1<f64>) -> Vec<(usize, usize)> {
        let mut artifacts = Vec::new();

        // Calculate local variance in sliding windows
        let window_size = (self.sample_rate * 0.5) as usize;
        let step = window_size / 2;

        let global_std = signal.std(0.0);
        let threshold = global_std * 3.0;

        let mut in_artifact = false;
        let mut artifact_start = 0;

        for i in (0..signal.len()).step_by(step) {
            let end = (i + window_size).min(signal.len());
            let window = signal.slice(ndarray::s![i..end]);
            let local_std = window.std(0.0);

            if local_std > threshold && !in_artifact {
                in_artifact = true;
                artifact_start = i;
            } else if local_std <= threshold && in_artifact {
                in_artifact = false;
                artifacts.push((artifact_start, i));
            }
        }

        if in_artifact {
            artifacts.push((artifact_start, signal.len()));
        }

        artifacts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn generate_synthetic_ppg(sample_rate: f64, duration: f64, heart_rate: f64) -> Array1<f64> {
        let n_samples = (sample_rate * duration) as usize;
        let mut signal = Array1::zeros(n_samples);

        let period = sample_rate * 60.0 / heart_rate;

        for i in 0..n_samples {
            let phase = (i as f64 % period) / period;
            // Simplified pulse wave shape
            let pulse = if phase < 0.2 {
                (phase / 0.2).powi(2)
            } else if phase < 0.35 {
                1.0 - 0.3 * ((phase - 0.2) / 0.15)
            } else if phase < 0.5 {
                0.7 + 0.2 * ((phase - 0.35) / 0.15)
            } else {
                0.9 * (1.0 - (phase - 0.5) / 0.5).powi(2)
            };
            signal[i] = pulse;
        }

        signal
    }

    #[test]
    fn test_peak_detection() {
        let sample_rate = 100.0;
        let signal = generate_synthetic_ppg(sample_rate, 10.0, 75.0);
        let analyzer = PpgAnalyzer::new(sample_rate);

        let peaks = analyzer.detect_peaks(signal.view()).unwrap();

        // At 75 BPM for 10 seconds, expect ~12-13 peaks
        assert!(peaks.len() >= 10 && peaks.len() <= 15);
    }

    #[test]
    fn test_pulse_intervals() {
        let sample_rate = 100.0;
        let signal = generate_synthetic_ppg(sample_rate, 10.0, 60.0);
        let analyzer = PpgAnalyzer::new(sample_rate);

        let peaks = analyzer.detect_peaks(signal.view()).unwrap();
        let intervals = analyzer.calculate_pulse_intervals(&peaks);

        // At 60 BPM, intervals should be ~1000ms
        for interval in intervals {
            assert!(interval > 800.0 && interval < 1200.0);
        }
    }

    #[test]
    fn test_heart_rate_calculation() {
        let intervals = vec![1000.0, 1000.0, 1000.0];
        let analyzer = PpgAnalyzer::new(100.0);
        let hr = analyzer.calculate_heart_rate(&intervals);

        for rate in hr {
            assert!((rate - 60.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_spo2_estimation() {
        let sample_rate = 100.0;
        let red = generate_synthetic_ppg(sample_rate, 5.0, 75.0);
        let ir = generate_synthetic_ppg(sample_rate, 5.0, 75.0) * 1.1;

        let analyzer = PpgAnalyzer::new(sample_rate);
        let result = analyzer.estimate_spo2(red.view(), ir.view()).unwrap();

        assert!(result.spo2 >= 70.0 && result.spo2 <= 100.0);
        assert!(result.perfusion_index >= 0.0);
    }

    #[test]
    fn test_prv_metrics() {
        let intervals: Vec<f64> = (0..50).map(|i| 800.0 + (i as f64 * 0.1).sin() * 50.0).collect();
        let analyzer = PpgAnalyzer::new(100.0);

        let prv = analyzer.calculate_prv(&intervals).unwrap();

        assert!(prv.mean_pi > 0.0);
        assert!(prv.sdnn >= 0.0);
        assert!(prv.rmssd >= 0.0);
        assert!(prv.pnn50 >= 0.0 && prv.pnn50 <= 100.0);
    }

    #[test]
    fn test_perfusion_index() {
        let signal = Array1::from_vec(vec![1.0, 1.1, 0.9, 1.05, 0.95]);
        let analyzer = PpgAnalyzer::new(100.0);

        let pi = analyzer.calculate_perfusion_index(signal.view());
        assert!(pi > 0.0);
    }
}
