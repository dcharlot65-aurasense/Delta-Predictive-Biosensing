//! EEG artifact detection and removal.
//!
//! This module provides tools for detecting and removing common artifacts in EEG signals,
//! including amplitude artifacts, DC offset, trends, and power line noise.

use crate::error::{DpbError, Result};
use crate::signal::filter::IirFilter;
use ndarray::{Array1};
use std::f64::consts::PI;

/// Types of artifacts that can be detected in EEG signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArtifactType {
    /// Amplitude exceeds threshold (e.g., muscle artifact, electrode pop)
    Amplitude,
    /// Signal shows no variation (e.g., electrode disconnection)
    Flatline,
    /// High frequency contamination (e.g., EMG/muscle activity)
    HighFrequency,
}

/// Represents a segment of EEG data containing an artifact.
#[derive(Debug, Clone)]
pub struct ArtifactSegment {
    /// Starting sample index of the artifact
    pub start_sample: usize,
    /// Ending sample index of the artifact
    pub end_sample: usize,
    /// Type of artifact detected
    pub artifact_type: ArtifactType,
    /// Peak amplitude of the artifact (in signal units)
    pub amplitude: f64,
}

/// Detects artifacts in EEG data based on amplitude threshold.
///
/// Identifies segments where the signal amplitude exceeds a specified threshold,
/// which may indicate muscle artifacts, electrode pops, or other non-neural activity.
///
/// # Arguments
///
/// * `eeg` - The input EEG signal
/// * `sample_rate` - Sampling frequency in Hz
/// * `threshold_uv` - Amplitude threshold in microvolts (absolute value)
///
/// # Returns
///
/// A vector of `ArtifactSegment` describing detected artifacts.
///
/// # Example
///
/// ```
/// use dpb_core::signal::eeg::detect_artifacts;
///
/// let eeg_signal = vec![10.0, 15.0, 200.0, 250.0, 20.0, 15.0]; // µV
/// let sample_rate = 250.0;
/// let threshold = 100.0; // µV
/// let artifacts = detect_artifacts(&eeg_signal, sample_rate, threshold).unwrap();
/// assert_eq!(artifacts.len(), 1);
/// ```
pub fn detect_artifacts(
    eeg: &[f64],
    sample_rate: f64,
    threshold_uv: f64,
) -> Result<Vec<ArtifactSegment>> {
    if eeg.is_empty() {
        return Err(DpbError::InvalidDimensions(
            "EEG signal cannot be empty".to_string(),
        ));
    }

    if sample_rate <= 0.0 {
        return Err(DpbError::InvalidParameter(
            "Sample rate must be positive".to_string(),
        ));
    }

    if threshold_uv <= 0.0 {
        return Err(DpbError::InvalidParameter(
            "Threshold must be positive".to_string(),
        ));
    }

    let mut artifacts = Vec::new();
    let mut in_artifact = false;
    let mut artifact_start = 0;
    let mut artifact_peak = 0.0;

    // Minimum artifact duration, in samples.
    //
    // An amplitude artifact is DEFINED by exceeding the threshold, not by
    // lasting a particular time, so this only rejects an empty run. The
    // previous floor of 100 ms discarded anything briefer -- 25 samples at
    // 250 Hz, 100 at 1 kHz -- which covers electrode pops and movement spikes,
    // typically 10-50 ms and the most common amplitude artifacts there are. A
    // 10-sample excursion to 200 uV was reported as no artifact at all. It was
    // also a hidden policy: derived from the sample rate, absent from the
    // signature, and not adjustable by the caller.
    //
    // Flatline detection below keeps its own duration criterion, where a
    // minimum span genuinely is part of the definition.
    const MIN_AMPLITUDE_ARTIFACT_SAMPLES: usize = 1;
    let min_duration = MIN_AMPLITUDE_ARTIFACT_SAMPLES;

    for (i, &value) in eeg.iter().enumerate() {
        let abs_value = value.abs();

        if abs_value > threshold_uv {
            if !in_artifact {
                // Start of new artifact
                in_artifact = true;
                artifact_start = i;
                artifact_peak = abs_value;
            } else {
                // Update peak amplitude
                artifact_peak = artifact_peak.max(abs_value);
            }
        } else if in_artifact {
            // End of artifact
            let duration = i - artifact_start;
            if duration >= min_duration {
                artifacts.push(ArtifactSegment {
                    start_sample: artifact_start,
                    end_sample: i - 1,
                    artifact_type: ArtifactType::Amplitude,
                    amplitude: artifact_peak,
                });
            }
            in_artifact = false;
        }
    }

    // Handle artifact at end of signal
    if in_artifact {
        let duration = eeg.len() - artifact_start;
        if duration >= min_duration {
            artifacts.push(ArtifactSegment {
                start_sample: artifact_start,
                end_sample: eeg.len() - 1,
                artifact_type: ArtifactType::Amplitude,
                amplitude: artifact_peak,
            });
        }
    }

    // Detect flatline artifacts
    let flatline_threshold = 0.5; // Very low variance threshold
    let window_size = (1.0 * sample_rate) as usize; // 1 second windows

    for i in (0..eeg.len()).step_by(window_size / 2) {
        let end = (i + window_size).min(eeg.len());
        if end - i < window_size / 2 {
            break;
        }

        let window = &eeg[i..end];
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let variance = window
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / window.len() as f64;

        if variance < flatline_threshold {
            artifacts.push(ArtifactSegment {
                start_sample: i,
                end_sample: end - 1,
                artifact_type: ArtifactType::Flatline,
                amplitude: variance.sqrt(),
            });
        }
    }

    Ok(artifacts)
}

/// Removes DC offset from an EEG signal.
///
/// Subtracts the mean value from the signal, centering it around zero.
/// This is a common preprocessing step for EEG analysis.
///
/// # Arguments
///
/// * `eeg` - The input EEG signal
///
/// # Returns
///
/// The signal with DC offset removed.
///
/// # Example
///
/// ```
/// use dpb_core::signal::eeg::remove_dc_offset;
///
/// let eeg_signal = vec![10.0, 11.0, 12.0, 13.0];
/// let detrended = remove_dc_offset(&eeg_signal);
/// let mean: f64 = detrended.iter().sum::<f64>() / detrended.len() as f64;
/// assert!((mean.abs() < 1e-10)); // Mean should be ~0
/// ```
pub fn remove_dc_offset(eeg: &[f64]) -> Vec<f64> {
    if eeg.is_empty() {
        return Vec::new();
    }

    let mean = eeg.iter().sum::<f64>() / eeg.len() as f64;
    eeg.iter().map(|&x| x - mean).collect()
}

/// Removes linear trend from an EEG signal.
///
/// Fits a linear trend to the signal and subtracts it, which is useful for
/// removing slow drifts that may be caused by electrode impedance changes
/// or other non-neural sources.
///
/// # Arguments
///
/// * `eeg` - The input EEG signal
///
/// # Returns
///
/// The detrended signal.
///
/// # Example
///
/// ```
/// use dpb_core::signal::eeg::remove_trend;
///
/// let eeg_signal = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Linear trend
/// let detrended = remove_trend(&eeg_signal);
/// // Detrended signal should have minimal slope
/// ```
pub fn remove_trend(eeg: &[f64]) -> Vec<f64> {
    let n = eeg.len();
    if n == 0 {
        return Vec::new();
    }

    if n == 1 {
        return vec![0.0];
    }

    // Compute linear regression: y = mx + b
    let x_mean = (n - 1) as f64 / 2.0;
    let y_mean = eeg.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in eeg.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    let slope = if denominator != 0.0 {
        numerator / denominator
    } else {
        0.0
    };
    let intercept = y_mean - slope * x_mean;

    // Subtract trend
    eeg.iter()
        .enumerate()
        .map(|(i, &y)| y - (slope * i as f64 + intercept))
        .collect()
}

/// Applies a notch filter to remove power line interference.
///
/// Removes narrow-band interference at the specified frequency (typically 50 Hz or 60 Hz)
/// using an IIR notch filter. This is essential for removing electrical noise from
/// the power grid.
///
/// # Arguments
///
/// * `eeg` - The input EEG signal
/// * `sample_rate` - Sampling frequency in Hz
/// * `notch_freq` - Frequency to remove (e.g., 50.0 or 60.0 Hz)
///
/// # Returns
///
/// The filtered signal with power line interference removed.
///
/// # Example
///
/// ```
/// use dpb_core::signal::eeg::apply_notch_filter;
///
/// let eeg_signal = vec![0.1, 0.2, 0.15, 0.25, 0.1, 0.2];
/// let sample_rate = 250.0;
/// let notch_freq = 60.0; // Remove 60 Hz
/// let filtered = apply_notch_filter(&eeg_signal, sample_rate, notch_freq).unwrap();
/// ```
pub fn apply_notch_filter(eeg: &[f64], sample_rate: f64, notch_freq: f64) -> Result<Vec<f64>> {
    if eeg.is_empty() {
        return Err(DpbError::InvalidDimensions(
            "EEG signal cannot be empty".to_string(),
        ));
    }

    if sample_rate <= 0.0 {
        return Err(DpbError::InvalidParameter(
            "Sample rate must be positive".to_string(),
        ));
    }

    if notch_freq <= 0.0 || notch_freq >= sample_rate / 2.0 {
        return Err(DpbError::InvalidParameter(format!(
            "Notch frequency must be between 0 and Nyquist frequency ({})",
            sample_rate / 2.0
        )));
    }

    // Design notch filter parameters
    // Quality factor (bandwidth control)
    let q = 30.0;
    let w0 = 2.0 * PI * notch_freq / sample_rate;

    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();

    // Notch filter coefficients (difference equation)
    // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    let b0 = 1.0;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    // Normalize coefficients
    let b = vec![b0 / a0, b1 / a0, b2 / a0];
    let a = vec![1.0, a1 / a0, a2 / a0];

    // Apply filter
    let mut filter = IirFilter::new(b, a)?;
    let eeg_array = Array1::from_vec(eeg.to_vec());
    let filtered = filter.filter(eeg_array.view());

    Ok(filtered.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_detect_artifacts_amplitude() {
        let sample_rate = 250.0;
        // Signal with a brief high-amplitude artifact
        let mut signal = vec![10.0; 100];
        signal[40..50].fill(200.0); // Artifact

        let artifacts = detect_artifacts(&signal, sample_rate, 100.0).unwrap();

        // Should detect one amplitude artifact
        let amplitude_artifacts: Vec<_> = artifacts
            .iter()
            .filter(|a| a.artifact_type == ArtifactType::Amplitude)
            .collect();

        assert!(!amplitude_artifacts.is_empty());
        assert!(amplitude_artifacts[0].amplitude > 100.0);

        // The segment must cover the samples that actually breached threshold.
        let seg = amplitude_artifacts[0];
        assert_eq!(seg.start_sample, 40);
        assert_eq!(seg.end_sample, 49);
        assert_eq!(seg.amplitude, 200.0);

        // A clean signal must yield nothing.
        let clean = vec![10.0; 100];
        let none = detect_artifacts(&clean, sample_rate, 100.0).unwrap();
        assert!(
            !none.iter().any(|a| a.artifact_type == ArtifactType::Amplitude),
            "flagged an amplitude artifact in a flat 10 uV signal"
        );
    }

    #[test]
    fn test_detect_artifacts_flatline() {
        let sample_rate = 250.0;
        // Create a flatline segment
        let signal = vec![10.0; 500]; // Constant signal

        let artifacts = detect_artifacts(&signal, sample_rate, 1000.0).unwrap();

        // Should detect flatline artifacts
        let flatline_artifacts: Vec<_> = artifacts
            .iter()
            .filter(|a| a.artifact_type == ArtifactType::Flatline)
            .collect();

        assert!(!flatline_artifacts.is_empty());
    }

    #[test]
    fn test_detect_artifacts_invalid_params() {
        let signal = vec![1.0, 2.0, 3.0];

        // Empty signal
        assert!(detect_artifacts(&[], 250.0, 100.0).is_err());

        // Invalid sample rate
        assert!(detect_artifacts(&signal, -10.0, 100.0).is_err());

        // Invalid threshold
        assert!(detect_artifacts(&signal, 250.0, -100.0).is_err());
    }

    #[test]
    fn test_remove_dc_offset() {
        let signal = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let detrended = remove_dc_offset(&signal);

        // Check that mean is approximately zero
        let mean = detrended.iter().sum::<f64>() / detrended.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_remove_dc_offset_empty() {
        let signal: Vec<f64> = vec![];
        let detrended = remove_dc_offset(&signal);
        assert!(detrended.is_empty());
    }

    #[test]
    fn test_remove_trend() {
        // Create signal with linear trend
        let signal: Vec<f64> = (0..100).map(|i| i as f64 * 0.5 + 10.0).collect();
        let detrended = remove_trend(&signal);

        // Detrended signal should have zero mean and minimal slope
        let mean = detrended.iter().sum::<f64>() / detrended.len() as f64;
        assert!(mean.abs() < 1e-10);

        // Check that variance is very small (since we removed the linear component)
        let variance = detrended
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / detrended.len() as f64;
        assert!(variance < 1e-10);
    }

    #[test]
    fn test_remove_trend_empty() {
        let signal: Vec<f64> = vec![];
        let detrended = remove_trend(&signal);
        assert!(detrended.is_empty());
    }

    #[test]
    fn test_remove_trend_single_point() {
        let signal = vec![5.0];
        let detrended = remove_trend(&signal);
        assert_eq!(detrended.len(), 1);
        assert_relative_eq!(detrended[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_notch_filter() {
        let sample_rate = 250.0;
        let duration = 2.0;
        let n = (sample_rate * duration) as usize;

        // Create signal with 60 Hz interference
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                // Clean EEG-like signal (10 Hz) + 60 Hz noise
                (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 60.0 * t).sin()
            })
            .collect();

        let filtered = apply_notch_filter(&signal, sample_rate, 60.0).unwrap();

        assert_eq!(filtered.len(), signal.len());

        // The filtered signal should have reduced 60 Hz component
        // We can verify this by checking that the filtered signal is different from input
        let difference: f64 = signal
            .iter()
            .zip(filtered.iter())
            .map(|(s, f)| (s - f).abs())
            .sum();
        assert!(difference > 0.0);
    }

    #[test]
    fn test_apply_notch_filter_invalid_params() {
        let signal = vec![1.0, 2.0, 3.0];

        // Empty signal
        assert!(apply_notch_filter(&[], 250.0, 60.0).is_err());

        // Invalid sample rate
        assert!(apply_notch_filter(&signal, -10.0, 60.0).is_err());

        // Notch frequency exceeds Nyquist
        assert!(apply_notch_filter(&signal, 250.0, 200.0).is_err());

        // Negative notch frequency
        assert!(apply_notch_filter(&signal, 250.0, -60.0).is_err());
    }

    #[test]
    fn test_artifact_segment_creation() {
        let artifact = ArtifactSegment {
            start_sample: 100,
            end_sample: 150,
            artifact_type: ArtifactType::Amplitude,
            amplitude: 250.0,
        };

        assert_eq!(artifact.start_sample, 100);
        assert_eq!(artifact.end_sample, 150);
        assert_eq!(artifact.artifact_type, ArtifactType::Amplitude);
        assert_relative_eq!(artifact.amplitude, 250.0);
    }

    #[test]
    fn test_artifact_type_equality() {
        assert_eq!(ArtifactType::Amplitude, ArtifactType::Amplitude);
        assert_eq!(ArtifactType::Flatline, ArtifactType::Flatline);
        assert_eq!(ArtifactType::HighFrequency, ArtifactType::HighFrequency);
        assert_ne!(ArtifactType::Amplitude, ArtifactType::Flatline);
    }
}
