//! HRV (Heart Rate Variability) Analysis Module
//!
//! Provides comprehensive HRV analysis including:
//! - Time domain metrics (SDNN, RMSSD, pNN50, etc.)
//! - Frequency domain metrics (VLF, LF, HF power, LF/HF ratio)
//! - RR interval preprocessing and artifact removal
//! - Comprehensive HRV metrics for autonomic nervous system assessment

use crate::error::{DpbError, Result};
use crate::signal::FftProcessor;
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

/// HRV time domain metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrvTimeDomain {
    /// Standard deviation of NN intervals (ms)
    pub sdnn_ms: f64,
    /// Root mean square of successive differences (ms)
    pub rmssd_ms: f64,
    /// Percentage of successive NN intervals > 50ms
    pub pnn50_percent: f64,
    /// Standard deviation of average NN intervals in 5-min segments (ms)
    pub sdann_ms: f64,
    /// HRV triangular index
    pub triangular_index: f64,
    /// Triangular interpolation of NN interval histogram (ms)
    pub tinn_ms: f64,
    /// Mean RR interval (ms)
    pub mean_rr_ms: f64,
    /// Maximum RR interval (ms)
    pub max_rr_ms: f64,
    /// Minimum RR interval (ms)
    pub min_rr_ms: f64,
}

impl Default for HrvTimeDomain {
    fn default() -> Self {
        Self {
            sdnn_ms: 0.0,
            rmssd_ms: 0.0,
            pnn50_percent: 0.0,
            sdann_ms: 0.0,
            triangular_index: 0.0,
            tinn_ms: 0.0,
            mean_rr_ms: 0.0,
            max_rr_ms: 0.0,
            min_rr_ms: 0.0,
        }
    }
}

/// HRV frequency domain metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrvFrequencyDomain {
    /// Very low frequency power (0.003-0.04 Hz) in ms²
    pub vlf_power_ms2: f64,
    /// Low frequency power (0.04-0.15 Hz) in ms²
    pub lf_power_ms2: f64,
    /// High frequency power (0.15-0.4 Hz) in ms²
    pub hf_power_ms2: f64,
    /// LF/HF ratio
    pub lf_hf_ratio: f64,
    /// Total power (ms²)
    pub total_power_ms2: f64,
    /// LF power in normalized units
    pub lf_nu: f64,
    /// HF power in normalized units
    pub hf_nu: f64,
}

impl Default for HrvFrequencyDomain {
    fn default() -> Self {
        Self {
            vlf_power_ms2: 0.0,
            lf_power_ms2: 0.0,
            hf_power_ms2: 0.0,
            lf_hf_ratio: 0.0,
            total_power_ms2: 0.0,
            lf_nu: 0.0,
            hf_nu: 0.0,
        }
    }
}

/// Combined HRV metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrvMetrics {
    /// Time domain metrics
    pub time_domain: HrvTimeDomain,
    /// Frequency domain metrics
    pub frequency_domain: HrvFrequencyDomain,
}

/// HRV analyzer for computing time and frequency domain metrics
pub struct HrvAnalyzer {
    /// Minimum acceptable RR interval (ms)
    min_rr_ms: f64,
    /// Maximum acceptable RR interval (ms)
    max_rr_ms: f64,
    /// Maximum acceptable RR interval change (%)
    max_rr_change_percent: f64,
}

impl HrvAnalyzer {
    /// Create a new HRV analyzer with default thresholds
    pub fn new() -> Self {
        Self {
            min_rr_ms: 300.0,            // 200 bpm
            max_rr_ms: 2000.0,           // 30 bpm
            max_rr_change_percent: 20.0, // 20% max change
        }
    }

    /// Create a new HRV analyzer with custom thresholds
    pub fn with_thresholds(min_rr_ms: f64, max_rr_ms: f64, max_rr_change_percent: f64) -> Self {
        Self {
            min_rr_ms,
            max_rr_ms,
            max_rr_change_percent,
        }
    }

    /// Compute all HRV metrics (time and frequency domain)
    ///
    /// # Arguments
    /// * `rr_intervals` - RR intervals in milliseconds
    /// * `sample_rate` - Effective sampling rate for frequency analysis (Hz)
    ///
    /// # Returns
    /// Complete HRV metrics
    pub fn compute_all(&self, rr_intervals: &[f64], sample_rate: f64) -> Result<HrvMetrics> {
        let time_domain = self.compute_time_domain(rr_intervals)?;
        let frequency_domain = self.compute_frequency_domain(rr_intervals, sample_rate)?;

        Ok(HrvMetrics {
            time_domain,
            frequency_domain,
        })
    }

    /// Compute time domain HRV metrics
    ///
    /// # Arguments
    /// * `rr_intervals` - RR intervals in milliseconds
    ///
    /// # Returns
    /// Time domain HRV metrics
    pub fn compute_time_domain(&self, rr_intervals: &[f64]) -> Result<HrvTimeDomain> {
        if rr_intervals.is_empty() {
            return Err(DpbError::InvalidParameter(
                "RR intervals are empty".to_string(),
            ));
        }

        // Preprocess: remove artifacts and ectopic beats
        let clean_rr = self.preprocess_rr_intervals(rr_intervals)?;

        if clean_rr.len() < 2 {
            return Err(DpbError::InvalidParameter(
                "Insufficient valid RR intervals".to_string(),
            ));
        }

        // Mean RR
        let mean_rr_ms = clean_rr.iter().sum::<f64>() / clean_rr.len() as f64;

        // Min/Max RR
        let min_rr_ms = clean_rr.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_rr_ms = clean_rr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // SDNN: Standard deviation of NN intervals
        let variance = clean_rr
            .iter()
            .map(|x| (x - mean_rr_ms).powi(2))
            .sum::<f64>()
            / clean_rr.len() as f64;
        let sdnn_ms = variance.sqrt();

        // RMSSD: Root mean square of successive differences
        let mut successive_diffs = Vec::new();
        for i in 0..clean_rr.len() - 1 {
            successive_diffs.push(clean_rr[i + 1] - clean_rr[i]);
        }

        let rmssd_ms = if !successive_diffs.is_empty() {
            let sum_sq: f64 = successive_diffs.iter().map(|x| x.powi(2)).sum();
            (sum_sq / successive_diffs.len() as f64).sqrt()
        } else {
            0.0
        };

        // pNN50: Percentage of successive differences > 50ms
        let nn50_count = successive_diffs
            .iter()
            .filter(|&&diff| diff.abs() > 50.0)
            .count();
        let pnn50_percent = if !successive_diffs.is_empty() {
            (nn50_count as f64 / successive_diffs.len() as f64) * 100.0
        } else {
            0.0
        };

        // SDANN: Standard deviation of 5-minute average NN intervals
        let sdann_ms = self.compute_sdann(&clean_rr);

        // Triangular index and TINN
        let (triangular_index, tinn_ms) = self.compute_triangular_metrics(&clean_rr);

        Ok(HrvTimeDomain {
            sdnn_ms,
            rmssd_ms,
            pnn50_percent,
            sdann_ms,
            triangular_index,
            tinn_ms,
            mean_rr_ms,
            max_rr_ms,
            min_rr_ms,
        })
    }

    /// Compute frequency domain HRV metrics
    ///
    /// # Arguments
    /// * `rr_intervals` - RR intervals in milliseconds
    /// * `sample_rate` - Effective sampling rate for frequency analysis (Hz)
    ///
    /// # Returns
    /// Frequency domain HRV metrics
    pub fn compute_frequency_domain(
        &self,
        rr_intervals: &[f64],
        sample_rate: f64,
    ) -> Result<HrvFrequencyDomain> {
        if rr_intervals.is_empty() {
            return Err(DpbError::InvalidParameter(
                "RR intervals are empty".to_string(),
            ));
        }

        // Preprocess RR intervals
        let clean_rr = self.preprocess_rr_intervals(rr_intervals)?;

        if clean_rr.len() < 10 {
            return Err(DpbError::InvalidParameter(
                "Insufficient RR intervals for frequency analysis".to_string(),
            ));
        }

        // Interpolate RR intervals to uniform sampling
        let interpolated = self.interpolate_rr_intervals(&clean_rr, sample_rate)?;

        // Remove DC component
        let mean = interpolated.iter().sum::<f64>() / interpolated.len() as f64;
        let detrended: Vec<f64> = interpolated.iter().map(|x| x - mean).collect();

        // Compute power spectral density using FFT
        let mut fft_processor = FftProcessor::new();
        let spectrum = fft_processor.fft(ArrayView1::from(&detrended))?;

        // Compute power in each frequency band
        let freqs: Vec<f64> = (0..spectrum.len())
            .map(|i| i as f64 * sample_rate / detrended.len() as f64)
            .collect();

        let mut vlf_power_ms2 = 0.0;
        let mut lf_power_ms2 = 0.0;
        let mut hf_power_ms2 = 0.0;

        for (i, &freq) in freqs.iter().enumerate() {
            let power = spectrum[i].norm_sqr();

            if (0.003..0.04).contains(&freq) {
                vlf_power_ms2 += power;
            } else if (0.04..0.15).contains(&freq) {
                lf_power_ms2 += power;
            } else if (0.15..0.4).contains(&freq) {
                hf_power_ms2 += power;
            }
        }

        let total_power_ms2 = vlf_power_ms2 + lf_power_ms2 + hf_power_ms2;

        // LF/HF ratio
        let lf_hf_ratio = if hf_power_ms2 > 0.0 {
            lf_power_ms2 / hf_power_ms2
        } else {
            0.0
        };

        // Normalized units (excluding VLF)
        let lf_hf_sum = lf_power_ms2 + hf_power_ms2;
        let lf_nu = if lf_hf_sum > 0.0 {
            (lf_power_ms2 / lf_hf_sum) * 100.0
        } else {
            0.0
        };
        let hf_nu = if lf_hf_sum > 0.0 {
            (hf_power_ms2 / lf_hf_sum) * 100.0
        } else {
            0.0
        };

        Ok(HrvFrequencyDomain {
            vlf_power_ms2,
            lf_power_ms2,
            hf_power_ms2,
            lf_hf_ratio,
            total_power_ms2,
            lf_nu,
            hf_nu,
        })
    }

    /// Preprocess RR intervals: remove artifacts and ectopic beats
    fn preprocess_rr_intervals(&self, rr_intervals: &[f64]) -> Result<Vec<f64>> {
        let mut clean = Vec::new();

        for (i, &rr) in rr_intervals.iter().enumerate() {
            // Check physiological range
            if rr < self.min_rr_ms || rr > self.max_rr_ms {
                continue;
            }

            // Check for large changes (ectopic beats)
            if i > 0 {
                let prev_rr = clean.last().copied().unwrap_or(rr);
                let change_percent = ((rr - prev_rr).abs() / prev_rr) * 100.0;

                if change_percent > self.max_rr_change_percent {
                    continue;
                }
            }

            clean.push(rr);
        }

        if clean.is_empty() {
            return Err(DpbError::InvalidParameter(
                "No valid RR intervals after preprocessing".to_string(),
            ));
        }

        Ok(clean)
    }

    /// Interpolate RR intervals to uniform sampling
    fn interpolate_rr_intervals(&self, rr_intervals: &[f64], sample_rate: f64) -> Result<Vec<f64>> {
        if rr_intervals.is_empty() {
            return Err(DpbError::InvalidParameter(
                "RR intervals are empty".to_string(),
            ));
        }

        // Compute cumulative time points
        let mut time_points = vec![0.0];
        for &rr in rr_intervals {
            let last_time = *time_points.last().unwrap();
            time_points.push(last_time + rr / 1000.0); // Convert ms to seconds
        }

        // Create uniform time grid
        let duration = *time_points.last().unwrap();
        let num_samples = (duration * sample_rate).ceil() as usize;
        let dt = duration / num_samples as f64;

        let mut interpolated = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f64 * dt;

            // Find surrounding RR intervals
            let mut idx = 0;
            while idx < time_points.len() - 1 && time_points[idx + 1] < t {
                idx += 1;
            }

            // Linear interpolation
            let value = if idx < rr_intervals.len() {
                if idx == 0 {
                    rr_intervals[0]
                } else {
                    let t0 = time_points[idx];
                    let t1 = time_points[idx + 1];
                    let v0 = rr_intervals[idx.saturating_sub(1)];
                    let v1 = rr_intervals[idx];

                    if (t1 - t0).abs() < 1e-10 {
                        v1
                    } else {
                        v0 + (v1 - v0) * (t - t0) / (t1 - t0)
                    }
                }
            } else {
                *rr_intervals.last().unwrap()
            };

            interpolated.push(value);
        }

        Ok(interpolated)
    }

    /// Compute SDANN (standard deviation of 5-minute averages)
    fn compute_sdann(&self, rr_intervals: &[f64]) -> f64 {
        if rr_intervals.is_empty() {
            return 0.0;
        }

        // Segment into 5-minute windows
        let segment_duration_ms = 5.0 * 60.0 * 1000.0; // 5 minutes
        let mut segment_means = Vec::new();

        let mut current_sum = 0.0;
        let mut current_duration = 0.0;
        let mut current_count = 0;

        for &rr in rr_intervals {
            current_sum += rr;
            current_duration += rr;
            current_count += 1;

            if current_duration >= segment_duration_ms {
                segment_means.push(current_sum / current_count as f64);
                current_sum = 0.0;
                current_duration = 0.0;
                current_count = 0;
            }
        }

        // Add final segment if it has data
        if current_count > 0 {
            segment_means.push(current_sum / current_count as f64);
        }

        if segment_means.len() < 2 {
            return 0.0;
        }

        // Compute standard deviation of segment means
        let mean_of_means = segment_means.iter().sum::<f64>() / segment_means.len() as f64;
        let variance = segment_means
            .iter()
            .map(|x| (x - mean_of_means).powi(2))
            .sum::<f64>()
            / segment_means.len() as f64;

        variance.sqrt()
    }

    /// Compute triangular metrics (triangular index and TINN)
    fn compute_triangular_metrics(&self, rr_intervals: &[f64]) -> (f64, f64) {
        if rr_intervals.is_empty() {
            return (0.0, 0.0);
        }

        // Create histogram with 7.8125 ms bins (128 Hz sampling equivalent)
        let bin_size = 7.8125;
        let min_rr = rr_intervals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_rr = rr_intervals
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let num_bins = ((max_rr - min_rr) / bin_size).ceil() as usize + 1;
        let mut histogram = vec![0; num_bins];

        for &rr in rr_intervals {
            let bin = ((rr - min_rr) / bin_size) as usize;
            if bin < num_bins {
                histogram[bin] += 1;
            }
        }

        // Triangular index: total number of RR intervals / height of histogram
        let max_count = histogram.iter().max().copied().unwrap_or(0);
        let triangular_index = if max_count > 0 {
            rr_intervals.len() as f64 / max_count as f64
        } else {
            0.0
        };

        // TINN: baseline width of triangular interpolation
        // Simplified: use range containing 95% of samples
        let sorted_rr: Vec<f64> = {
            let mut sorted = rr_intervals.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            sorted
        };

        let lower_idx = (sorted_rr.len() as f64 * 0.025) as usize;
        let upper_idx = (sorted_rr.len() as f64 * 0.975) as usize;
        let tinn_ms = sorted_rr[upper_idx] - sorted_rr[lower_idx];

        (triangular_index, tinn_ms)
    }
}

impl Default for HrvAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_hrv_analyzer_creation() {
        let analyzer = HrvAnalyzer::new();
        assert_eq!(analyzer.min_rr_ms, 300.0);
        assert_eq!(analyzer.max_rr_ms, 2000.0);
    }

    #[test]
    fn test_time_domain_computation() {
        let analyzer = HrvAnalyzer::new();

        // Regular RR intervals at 60 bpm (1000ms intervals)
        let rr_intervals = vec![1000.0; 100];

        let result = analyzer.compute_time_domain(&rr_intervals);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert_relative_eq!(metrics.mean_rr_ms, 1000.0, epsilon = 1.0);
        assert_relative_eq!(metrics.sdnn_ms, 0.0, epsilon = 1.0);
        assert_relative_eq!(metrics.rmssd_ms, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_time_domain_with_variability() {
        let analyzer = HrvAnalyzer::new();

        // RR intervals with some variability
        let mut rr_intervals = Vec::new();
        for i in 0..50 {
            rr_intervals.push(1000.0 + (i % 10) as f64 * 10.0);
        }

        let result = analyzer.compute_time_domain(&rr_intervals);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.sdnn_ms > 0.0);
        assert!(metrics.rmssd_ms > 0.0);
    }

    #[test]
    fn test_preprocess_rr_intervals() {
        let analyzer = HrvAnalyzer::new();

        // Include some invalid intervals
        let rr_intervals = vec![
            1000.0, // Valid
            1020.0, // Valid
            200.0,  // Too short (invalid)
            1050.0, // Valid
            2500.0, // Too long (invalid)
            1030.0, // Valid
        ];

        let result = analyzer.preprocess_rr_intervals(&rr_intervals);
        assert!(result.is_ok());

        let clean = result.unwrap();
        assert!(clean.len() < rr_intervals.len());
        assert!(clean.iter().all(|&rr| (300.0..=2000.0).contains(&rr)));
    }

    #[test]
    fn test_rmssd_calculation() {
        let analyzer = HrvAnalyzer::new();

        // Alternating RR intervals
        let rr_intervals = vec![1000.0, 1050.0, 1000.0, 1050.0, 1000.0, 1050.0];

        let result = analyzer.compute_time_domain(&rr_intervals);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        // RMSSD should be 50ms for alternating ±50ms differences
        assert_relative_eq!(metrics.rmssd_ms, 50.0, epsilon = 1.0);
    }

    #[test]
    fn test_pnn50_calculation() {
        let analyzer = HrvAnalyzer::new();

        // Create RR intervals with successive differences > 50ms
        let mut rr_intervals = Vec::new();
        for i in 0..20 {
            rr_intervals.push(if i % 2 == 0 { 1000.0 } else { 1100.0 });
        }

        let result = analyzer.compute_time_domain(&rr_intervals);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        // All successive differences are 100ms, so pNN50 should be 100%
        assert_relative_eq!(metrics.pnn50_percent, 100.0, epsilon = 1.0);
    }

    #[test]
    fn test_frequency_domain_computation() {
        let analyzer = HrvAnalyzer::new();

        // Generate RR intervals with a dominant frequency
        let sample_rate = 4.0; // 4 Hz for RR tachogram
        let mut rr_intervals = Vec::new();

        for i in 0..100 {
            let t = i as f64 / 10.0;
            let rr = 1000.0 + 50.0 * (2.0 * PI * 0.1 * t).sin(); // 0.1 Hz oscillation (LF band)
            rr_intervals.push(rr);
        }

        let result = analyzer.compute_frequency_domain(&rr_intervals, sample_rate);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.total_power_ms2 > 0.0);
        assert!(metrics.lf_power_ms2 >= 0.0);
        assert!(metrics.hf_power_ms2 >= 0.0);
    }

    #[test]
    fn test_interpolate_rr_intervals() {
        let analyzer = HrvAnalyzer::new();
        let rr_intervals = vec![1000.0, 1000.0, 1000.0, 1000.0];
        let sample_rate = 4.0;

        let result = analyzer.interpolate_rr_intervals(&rr_intervals, sample_rate);
        assert!(result.is_ok());

        let interpolated = result.unwrap();
        assert!(!interpolated.is_empty());
    }

    #[test]
    fn test_sdann_computation() {
        let analyzer = HrvAnalyzer::new();

        // Create long sequence with varying segments
        let mut rr_intervals = Vec::new();

        // 10 minutes of data at 60 bpm
        rr_intervals.resize(rr_intervals.len() + 600, 1000.0);

        let sdann = analyzer.compute_sdann(&rr_intervals);
        // Should be near zero for constant RR
        assert!(sdann >= 0.0);
    }

    #[test]
    fn test_triangular_metrics() {
        let analyzer = HrvAnalyzer::new();

        let rr_intervals = vec![950.0, 970.0, 1000.0, 1030.0, 1050.0];

        let (tri_index, tinn) = analyzer.compute_triangular_metrics(&rr_intervals);

        assert!(tri_index > 0.0);
        assert!(tinn > 0.0);
    }

    #[test]
    fn test_compute_all_metrics() {
        let analyzer = HrvAnalyzer::new();

        let mut rr_intervals = Vec::new();
        for i in 0..100 {
            rr_intervals.push(1000.0 + (i % 20) as f64 * 5.0);
        }

        let sample_rate = 4.0;
        let result = analyzer.compute_all(&rr_intervals, sample_rate);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.time_domain.sdnn_ms > 0.0);
        assert!(metrics.frequency_domain.total_power_ms2 >= 0.0);
    }

    #[test]
    fn test_lf_hf_ratio() {
        let analyzer = HrvAnalyzer::new();

        let mut rr_intervals = Vec::new();
        for i in 0..100 {
            rr_intervals.push(1000.0 + 20.0 * ((i as f64) / 10.0).sin());
        }

        let result = analyzer.compute_frequency_domain(&rr_intervals, 4.0);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        assert!(metrics.lf_hf_ratio >= 0.0);
    }

    #[test]
    fn test_normalized_units() {
        let analyzer = HrvAnalyzer::new();

        let mut rr_intervals = Vec::new();
        for i in 0..100 {
            rr_intervals.push(1000.0 + 30.0 * ((i as f64) / 5.0).sin());
        }

        let result = analyzer.compute_frequency_domain(&rr_intervals, 4.0);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        // Normalized units should sum to approximately 100
        let sum = metrics.lf_nu + metrics.hf_nu;
        assert!(sum <= 100.0 || sum == 0.0);
    }

    #[test]
    fn test_hrv_metrics_serialization() {
        let time_domain = HrvTimeDomain {
            sdnn_ms: 50.0,
            rmssd_ms: 40.0,
            pnn50_percent: 15.0,
            sdann_ms: 45.0,
            triangular_index: 10.0,
            tinn_ms: 200.0,
            mean_rr_ms: 1000.0,
            max_rr_ms: 1100.0,
            min_rr_ms: 900.0,
        };

        let json = serde_json::to_string(&time_domain).unwrap();
        let deserialized: HrvTimeDomain = serde_json::from_str(&json).unwrap();

        assert_relative_eq!(deserialized.sdnn_ms, 50.0);
        assert_relative_eq!(deserialized.mean_rr_ms, 1000.0);
    }

    #[test]
    fn test_empty_rr_intervals() {
        let analyzer = HrvAnalyzer::new();
        let empty: Vec<f64> = vec![];

        let result = analyzer.compute_time_domain(&empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_thresholds() {
        let analyzer = HrvAnalyzer::with_thresholds(400.0, 1500.0, 15.0);

        assert_eq!(analyzer.min_rr_ms, 400.0);
        assert_eq!(analyzer.max_rr_ms, 1500.0);
        assert_eq!(analyzer.max_rr_change_percent, 15.0);
    }
}
