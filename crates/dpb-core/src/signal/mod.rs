//! Signal processing utilities for biosensor data.

pub mod eeg;
pub mod fft;
pub mod filter;
pub mod resample;

pub use fft::{FftProcessor, Stft, WindowType, create_window, fft_frequencies, stft_times};
pub use filter::{FirFilter, IirFilter, FilterType, median_filter};
pub use resample::{downsample, resample_linear, resample_to_length, upsample};

use crate::error::{DpbError, Result};
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Normalization methods for signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMethod {
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Min-max normalization to [-1, 1]
    MinMaxSymmetric,
    /// Robust normalization using median and IQR
    Robust,
}

/// Normalizes a signal using the specified method.
pub fn normalize(signal: ArrayView1<f64>, method: NormalizationMethod) -> Result<Array1<f64>> {
    if signal.is_empty() {
        return Err(DpbError::InvalidDimensions(
            "Signal cannot be empty".to_string(),
        ));
    }

    match method {
        NormalizationMethod::ZScore => {
            let mean = signal.mean().unwrap_or(0.0);
            let std = signal.std(0.0);
            if std == 0.0 {
                return Ok(Array1::zeros(signal.len()));
            }
            Ok((signal.to_owned() - mean) / std)
        }
        NormalizationMethod::MinMax => {
            let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (max - min).abs() < 1e-10 {
                return Ok(Array1::zeros(signal.len()));
            }
            Ok((signal.to_owned() - min) / (max - min))
        }
        NormalizationMethod::MinMaxSymmetric => {
            let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (max - min).abs() < 1e-10 {
                return Ok(Array1::zeros(signal.len()));
            }
            Ok(((signal.to_owned() - min) / (max - min)) * 2.0 - 1.0)
        }
        NormalizationMethod::Robust => {
            let mut sorted = signal.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q25 = sorted[sorted.len() / 4];
            let q75 = sorted[3 * sorted.len() / 4];
            let median = sorted[sorted.len() / 2];
            let iqr = q75 - q25;
            if iqr == 0.0 {
                return Ok(Array1::zeros(signal.len()));
            }
            Ok((signal.to_owned() - median) / iqr)
        }
    }
}

/// Normalizes multiple channels independently.
pub fn normalize_channels(
    signals: ArrayView2<f64>,
    method: NormalizationMethod,
) -> Result<ndarray::Array2<f64>> {
    let num_channels = signals.nrows();
    let mut normalized = ndarray::Array2::zeros(signals.dim());

    for i in 0..num_channels {
        let channel = signals.row(i);
        let norm_channel = normalize(channel, method)?;
        normalized.row_mut(i).assign(&norm_channel);
    }

    Ok(normalized)
}

/// Removes DC offset from a signal.
pub fn remove_dc_offset(signal: ArrayView1<f64>) -> Array1<f64> {
    let mean = signal.mean().unwrap_or(0.0);
    signal.to_owned() - mean
}

/// Applies a gain to a signal.
pub fn apply_gain(signal: ArrayView1<f64>, gain: f64) -> Array1<f64> {
    signal.to_owned() * gain
}

/// Clips signal values to a range.
pub fn clip(signal: ArrayView1<f64>, min: f64, max: f64) -> Array1<f64> {
    signal.mapv(|x| x.clamp(min, max))
}

/// Detects zero crossings in a signal.
pub fn zero_crossings(signal: ArrayView1<f64>) -> Vec<usize> {
    let mut crossings = Vec::new();

    for i in 0..signal.len() - 1 {
        if (signal[i] < 0.0 && signal[i + 1] >= 0.0) || (signal[i] >= 0.0 && signal[i + 1] < 0.0) {
            crossings.push(i);
        }
    }

    crossings
}

/// Detects peaks in a signal.
pub fn find_peaks(signal: ArrayView1<f64>, threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();

    for i in 1..signal.len() - 1 {
        if signal[i] > threshold
            && signal[i] > signal[i - 1]
            && signal[i] > signal[i + 1]
        {
            peaks.push(i);
        }
    }

    peaks
}

/// Detects valleys in a signal.
pub fn find_valleys(signal: ArrayView1<f64>, threshold: f64) -> Vec<usize> {
    let mut valleys = Vec::new();

    for i in 1..signal.len() - 1 {
        if signal[i] < threshold
            && signal[i] < signal[i - 1]
            && signal[i] < signal[i + 1]
        {
            valleys.push(i);
        }
    }

    valleys
}

/// Computes the envelope of a signal using Hilbert transform approximation.
pub fn envelope(signal: ArrayView1<f64>) -> Result<Array1<f64>> {
    // Simplified envelope using moving maximum
    let window_size = signal.len().min(20);
    let mut env = Array1::zeros(signal.len());

    for i in 0..signal.len() {
        let start = i.saturating_sub(window_size / 2);
        let end = (i + window_size / 2 + 1).min(signal.len());
        let max_val = signal
            .slice(ndarray::s![start..end])
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max);
        env[i] = max_val;
    }

    Ok(env)
}

/// Computes the root mean square (RMS) of a signal.
pub fn rms(signal: ArrayView1<f64>) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_squares: f64 = signal.iter().map(|x| x * x).sum();
    (sum_squares / signal.len() as f64).sqrt()
}

/// Computes the energy of a signal.
pub fn energy(signal: ArrayView1<f64>) -> f64 {
    signal.iter().map(|x| x * x).sum()
}

/// Computes signal-to-noise ratio (SNR) in dB.
pub fn snr_db(signal: ArrayView1<f64>, noise: ArrayView1<f64>) -> Result<f64> {
    if signal.len() != noise.len() {
        return Err(DpbError::InvalidDimensions(
            "Signal and noise must have same length".to_string(),
        ));
    }

    let signal_power = energy(signal);
    let noise_power = energy(noise);

    if noise_power == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok(10.0 * (signal_power / noise_power).log10())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normalize_zscore() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = normalize(signal.view(), NormalizationMethod::ZScore).unwrap();

        let mean = normalized.mean().unwrap();
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_minmax() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = normalize(signal.view(), NormalizationMethod::MinMax).unwrap();

        assert_relative_eq!(normalized[0], 0.0);
        assert_relative_eq!(normalized[4], 1.0);
    }

    #[test]
    fn test_remove_dc_offset() {
        let signal = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0]);
        let detrended = remove_dc_offset(signal.view());

        let mean = detrended.mean().unwrap();
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_crossings() {
        let signal = Array1::from_vec(vec![-1.0, -0.5, 0.5, 1.0, 0.5, -0.5]);
        let crossings = zero_crossings(signal.view());

        assert_eq!(crossings.len(), 2);
        assert_eq!(crossings[0], 1); // Between index 1 and 2
        assert_eq!(crossings[1], 4); // Between index 4 and 5
    }

    #[test]
    fn test_find_peaks() {
        let signal = Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
        let peaks = find_peaks(signal.view(), 2.0);

        assert_eq!(peaks.len(), 2);
        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
    }

    #[test]
    fn test_rms() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let rms_val = rms(signal.view());

        let expected = ((1.0f64 + 4.0 + 9.0 + 16.0) / 4.0).sqrt();
        assert_relative_eq!(rms_val, expected);
    }

    #[test]
    fn test_energy() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let eng = energy(signal.view());

        assert_relative_eq!(eng, 14.0); // 1 + 4 + 9
    }

    #[test]
    fn test_clip() {
        let signal = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let clipped = clip(signal.view(), -1.0, 1.0);

        assert_eq!(clipped[0], -1.0);
        assert_eq!(clipped[4], 1.0);
        assert_eq!(clipped[2], 0.0);
    }
}
