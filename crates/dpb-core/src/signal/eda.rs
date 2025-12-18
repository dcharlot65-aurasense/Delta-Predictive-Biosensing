//! EDA (Electrodermal Activity) Analysis Module
//!
//! Provides comprehensive EDA signal analysis including:
//! - Tonic/phasic decomposition
//! - SCR (Skin Conductance Response) detection
//! - SCL (Skin Conductance Level) analysis
//! - Sympathetic nervous system activity estimation

use crate::error::{DpbError, Result};
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

/// Skin Conductance Response (SCR) event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrEvent {
    /// Onset time (samples)
    pub onset: usize,
    /// Peak time (samples)
    pub peak: usize,
    /// Recovery time (50% return) (samples)
    pub recovery: usize,
    /// Onset amplitude (microSiemens)
    pub onset_amplitude: f64,
    /// Peak amplitude (microSiemens)
    pub peak_amplitude: f64,
    /// SCR amplitude (peak - onset)
    pub amplitude: f64,
    /// Rise time (seconds)
    pub rise_time: f64,
    /// Recovery time (seconds)
    pub recovery_time: f64,
    /// Area under curve
    pub auc: f64,
}

/// Tonic/Phasic decomposition result
#[derive(Debug, Clone)]
pub struct EdaDecomposition {
    /// Tonic component (SCL)
    pub tonic: Array1<f64>,
    /// Phasic component (SCR)
    pub phasic: Array1<f64>,
    /// Detected SCR events
    pub scr_events: Vec<ScrEvent>,
}

/// EDA metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdaMetrics {
    /// Mean skin conductance level (microSiemens)
    pub mean_scl: f64,
    /// Standard deviation of SCL
    pub scl_std: f64,
    /// Number of SCR events
    pub scr_count: usize,
    /// SCR frequency (events per minute)
    pub scr_frequency: f64,
    /// Mean SCR amplitude
    pub mean_scr_amplitude: f64,
    /// Sum of SCR amplitudes
    pub sum_scr_amplitude: f64,
    /// Mean SCR rise time (seconds)
    pub mean_rise_time: f64,
    /// Mean SCR recovery time (seconds)
    pub mean_recovery_time: f64,
    /// Non-specific SCR count (spontaneous)
    pub ns_scr_count: usize,
}

/// EDA signal analyzer
pub struct EdaAnalyzer {
    sample_rate: f64,
    /// Minimum SCR amplitude threshold (microSiemens)
    scr_threshold: f64,
    /// Minimum rise time (seconds)
    min_rise_time: f64,
    /// Maximum rise time (seconds)
    max_rise_time: f64,
}

impl EdaAnalyzer {
    /// Create a new EDA analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            scr_threshold: 0.01, // 0.01 microSiemens
            min_rise_time: 0.5,  // 500ms minimum
            max_rise_time: 5.0,  // 5s maximum
        }
    }

    /// Create analyzer with custom thresholds
    pub fn with_thresholds(
        sample_rate: f64,
        scr_threshold: f64,
        min_rise_time: f64,
        max_rise_time: f64,
    ) -> Self {
        Self {
            sample_rate,
            scr_threshold,
            min_rise_time,
            max_rise_time,
        }
    }

    /// Decompose EDA into tonic (SCL) and phasic (SCR) components
    pub fn decompose(&self, signal: ArrayView1<f64>) -> Result<EdaDecomposition> {
        if signal.len() < 10 {
            return Err(DpbError::InvalidDimensions(
                "Signal too short for decomposition".to_string(),
            ));
        }

        // Use convex optimization-based decomposition (cvxEDA simplified)
        let tonic = self.extract_tonic(signal)?;
        let phasic = &signal.to_owned() - &tonic;

        // Detect SCR events in phasic component
        let scr_events = self.detect_scr(phasic.view())?;

        Ok(EdaDecomposition {
            tonic,
            phasic,
            scr_events,
        })
    }

    /// Extract tonic component using lowpass filtering
    fn extract_tonic(&self, signal: ArrayView1<f64>) -> Result<Array1<f64>> {
        // Use a very low cutoff (~0.05 Hz) moving average
        let window_size = (self.sample_rate * 4.0) as usize; // 4-second window
        let window_size = window_size.max(3);

        let mut tonic = Array1::zeros(signal.len());
        let half_window = window_size / 2;

        for i in 0..signal.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(signal.len());
            let window = signal.slice(ndarray::s![start..end]);

            // Use median for robustness against SCR peaks
            let mut values: Vec<f64> = window.to_vec();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            tonic[i] = values[values.len() / 2];
        }

        Ok(tonic)
    }

    /// Detect SCR events in phasic component
    fn detect_scr(&self, phasic: ArrayView1<f64>) -> Result<Vec<ScrEvent>> {
        let mut events = Vec::new();

        let min_samples = (self.min_rise_time * self.sample_rate) as usize;
        let max_samples = (self.max_rise_time * self.sample_rate) as usize;

        // Find local maxima that exceed threshold
        let mut i = 1;
        while i < phasic.len() - 1 {
            // Check if this is a peak above threshold
            if phasic[i] > self.scr_threshold
                && phasic[i] >= phasic[i - 1]
                && phasic[i] >= phasic[i + 1]
            {
                // Find onset (search backwards for minimum)
                let search_start = i.saturating_sub(max_samples);
                let mut onset_idx = i - 1;
                let mut onset_val = phasic[i - 1];

                for j in (search_start..i).rev() {
                    if phasic[j] < onset_val {
                        onset_val = phasic[j];
                        onset_idx = j;
                    }
                    // Stop if we start going up again (found the trough)
                    if phasic[j] > onset_val + self.scr_threshold * 0.5 {
                        break;
                    }
                }

                // Validate rise time
                let rise_samples = i - onset_idx;
                if rise_samples >= min_samples && rise_samples <= max_samples {
                    // Find recovery (50% return to baseline)
                    let amplitude = phasic[i] - onset_val;
                    let half_amplitude = onset_val + amplitude * 0.5;

                    let search_end = (i + max_samples * 2).min(phasic.len());
                    let mut recovery_idx = i;

                    for j in i..search_end {
                        if phasic[j] <= half_amplitude {
                            recovery_idx = j;
                            break;
                        }
                        recovery_idx = j;
                    }

                    // Calculate AUC using trapezoidal rule
                    let auc = self.calculate_auc(&phasic, onset_idx, recovery_idx, onset_val);

                    events.push(ScrEvent {
                        onset: onset_idx,
                        peak: i,
                        recovery: recovery_idx,
                        onset_amplitude: onset_val,
                        peak_amplitude: phasic[i],
                        amplitude,
                        rise_time: rise_samples as f64 / self.sample_rate,
                        recovery_time: (recovery_idx - i) as f64 / self.sample_rate,
                        auc,
                    });

                    // Skip past this SCR
                    i = recovery_idx;
                }
            }
            i += 1;
        }

        Ok(events)
    }

    /// Calculate area under curve for SCR
    fn calculate_auc(&self, signal: &ArrayView1<f64>, start: usize, end: usize, baseline: f64) -> f64 {
        let mut auc = 0.0;
        for i in start..end {
            auc += (signal[i] - baseline).max(0.0) / self.sample_rate;
        }
        auc
    }

    /// Calculate comprehensive EDA metrics
    pub fn calculate_metrics(
        &self,
        signal: ArrayView1<f64>,
        duration_seconds: f64,
    ) -> Result<EdaMetrics> {
        let decomposition = self.decompose(signal)?;

        let mean_scl = decomposition.tonic.mean().unwrap_or(0.0);
        let scl_std = decomposition.tonic.std(0.0);

        let scr_count = decomposition.scr_events.len();
        let scr_frequency = scr_count as f64 / (duration_seconds / 60.0);

        let (mean_scr_amplitude, sum_scr_amplitude) = if scr_count > 0 {
            let sum: f64 = decomposition.scr_events.iter().map(|e| e.amplitude).sum();
            (sum / scr_count as f64, sum)
        } else {
            (0.0, 0.0)
        };

        let mean_rise_time = if scr_count > 0 {
            decomposition.scr_events.iter().map(|e| e.rise_time).sum::<f64>() / scr_count as f64
        } else {
            0.0
        };

        let mean_recovery_time = if scr_count > 0 {
            decomposition.scr_events.iter().map(|e| e.recovery_time).sum::<f64>() / scr_count as f64
        } else {
            0.0
        };

        // Non-specific SCRs: those not associated with known stimuli (all in this analysis)
        let ns_scr_count = scr_count;

        Ok(EdaMetrics {
            mean_scl,
            scl_std,
            scr_count,
            scr_frequency,
            mean_scr_amplitude,
            sum_scr_amplitude,
            mean_rise_time,
            mean_recovery_time,
            ns_scr_count,
        })
    }

    /// Estimate sympathetic nervous system activity index
    pub fn estimate_sns_activity(&self, signal: ArrayView1<f64>) -> Result<f64> {
        let decomposition = self.decompose(signal)?;

        // SNS activity index based on SCR frequency and amplitude
        let scr_rate = decomposition.scr_events.len() as f64 / (signal.len() as f64 / self.sample_rate / 60.0);
        let mean_amplitude = if !decomposition.scr_events.is_empty() {
            decomposition.scr_events.iter().map(|e| e.amplitude).sum::<f64>()
                / decomposition.scr_events.len() as f64
        } else {
            0.0
        };

        // Combined index (normalized)
        let sns_index = (scr_rate / 10.0) * 0.5 + (mean_amplitude / 0.5) * 0.5;
        Ok(sns_index.clamp(0.0, 1.0))
    }

    /// Detect stimulus-locked SCR (within time window after stimulus)
    pub fn detect_stimulus_scr(
        &self,
        signal: ArrayView1<f64>,
        stimulus_times: &[usize],
        window_seconds: f64,
    ) -> Result<Vec<Option<ScrEvent>>> {
        let decomposition = self.decompose(signal)?;
        let window_samples = (window_seconds * self.sample_rate) as usize;

        let mut results = Vec::new();

        for &stim_time in stimulus_times {
            let window_end = stim_time + window_samples;

            // Find first SCR in window after stimulus
            let scr = decomposition
                .scr_events
                .iter()
                .find(|e| e.onset >= stim_time && e.onset < window_end)
                .cloned();

            results.push(scr);
        }

        Ok(results)
    }

    /// Calculate SCL slope over time (habituation indicator)
    pub fn calculate_scl_slope(&self, tonic: ArrayView1<f64>) -> f64 {
        if tonic.len() < 2 {
            return 0.0;
        }

        let n = tonic.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = tonic.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in tonic.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator > 0.0 {
            numerator / denominator * self.sample_rate // Slope per second
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_synthetic_eda(sample_rate: f64, duration: f64) -> Array1<f64> {
        let n_samples = (sample_rate * duration) as usize;
        let mut signal = Array1::zeros(n_samples);

        // Base tonic level with slight drift
        for i in 0..n_samples {
            let t = i as f64 / sample_rate;
            signal[i] = 2.0 + 0.1 * (t / 60.0); // Slight increase over time
        }

        // Add some SCR events
        let scr_times = vec![2.0, 5.0, 10.0, 15.0];
        for scr_time in scr_times {
            let scr_start = (scr_time * sample_rate) as usize;
            let rise_samples = (1.5 * sample_rate) as usize;
            let decay_samples = (4.0 * sample_rate) as usize;

            for i in 0..rise_samples {
                if scr_start + i < n_samples {
                    let progress = i as f64 / rise_samples as f64;
                    signal[scr_start + i] += 0.2 * progress;
                }
            }

            for i in 0..decay_samples {
                if scr_start + rise_samples + i < n_samples {
                    let progress = i as f64 / decay_samples as f64;
                    signal[scr_start + rise_samples + i] += 0.2 * (1.0 - progress);
                }
            }
        }

        signal
    }

    #[test]
    fn test_decomposition() {
        let sample_rate = 10.0;
        let signal = generate_synthetic_eda(sample_rate, 20.0);
        let analyzer = EdaAnalyzer::new(sample_rate);

        let decomposition = analyzer.decompose(signal.view()).unwrap();

        assert_eq!(decomposition.tonic.len(), signal.len());
        assert_eq!(decomposition.phasic.len(), signal.len());
    }

    #[test]
    fn test_scr_detection() {
        let sample_rate = 10.0;
        let signal = generate_synthetic_eda(sample_rate, 20.0);
        let analyzer = EdaAnalyzer::new(sample_rate);

        let decomposition = analyzer.decompose(signal.view()).unwrap();

        // Should detect some SCR events
        assert!(!decomposition.scr_events.is_empty());

        for event in &decomposition.scr_events {
            assert!(event.amplitude > 0.0);
            assert!(event.rise_time > 0.0);
        }
    }

    #[test]
    fn test_metrics_calculation() {
        let sample_rate = 10.0;
        let duration = 20.0;
        let signal = generate_synthetic_eda(sample_rate, duration);
        let analyzer = EdaAnalyzer::new(sample_rate);

        let metrics = analyzer.calculate_metrics(signal.view(), duration).unwrap();

        assert!(metrics.mean_scl > 0.0);
        assert!(metrics.scr_frequency >= 0.0);
    }

    #[test]
    fn test_sns_activity() {
        let sample_rate = 10.0;
        let signal = generate_synthetic_eda(sample_rate, 20.0);
        let analyzer = EdaAnalyzer::new(sample_rate);

        let sns = analyzer.estimate_sns_activity(signal.view()).unwrap();

        assert!(sns >= 0.0 && sns <= 1.0);
    }

    #[test]
    fn test_scl_slope() {
        let tonic = Array1::from_vec(vec![1.0, 1.1, 1.2, 1.3, 1.4]);
        let analyzer = EdaAnalyzer::new(10.0);

        let slope = analyzer.calculate_scl_slope(tonic.view());

        assert!(slope > 0.0); // Increasing tonic
    }
}
