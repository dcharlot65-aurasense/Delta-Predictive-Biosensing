//! Signal resampling utilities.

use crate::error::{DpbError, Result};
use ndarray::{Array1, ArrayView1};

/// Resamples a signal using linear interpolation.
pub fn resample_linear(
    signal: ArrayView1<f64>,
    original_rate: f64,
    target_rate: f64,
) -> Result<Array1<f64>> {
    if original_rate <= 0.0 || target_rate <= 0.0 {
        return Err(DpbError::InvalidParameter(
            "Sample rates must be positive".to_string(),
        ));
    }

    if signal.is_empty() {
        return Err(DpbError::InvalidDimensions(
            "Signal cannot be empty".to_string(),
        ));
    }

    let original_duration = signal.len() as f64 / original_rate;
    let target_len = (original_duration * target_rate).ceil() as usize;

    let mut resampled = Array1::zeros(target_len);

    for i in 0..target_len {
        let t = i as f64 / target_rate;
        let original_idx = t * original_rate;
        let idx_floor = original_idx.floor() as usize;
        let idx_ceil = (idx_floor + 1).min(signal.len() - 1);
        let fraction = original_idx - idx_floor as f64;

        if idx_floor < signal.len() {
            resampled[i] = signal[idx_floor] * (1.0 - fraction) + signal[idx_ceil] * fraction;
        }
    }

    Ok(resampled)
}

/// Downsamples a signal by a factor (simple decimation).
pub fn downsample(signal: ArrayView1<f64>, factor: usize) -> Result<Array1<f64>> {
    if factor == 0 {
        return Err(DpbError::InvalidParameter(
            "Downsample factor must be positive".to_string(),
        ));
    }

    if factor == 1 {
        return Ok(signal.to_owned());
    }

    let downsampled: Vec<f64> = signal.iter().step_by(factor).copied().collect();
    Ok(Array1::from_vec(downsampled))
}

/// Upsamples a signal by a factor using linear interpolation.
pub fn upsample(signal: ArrayView1<f64>, factor: usize) -> Result<Array1<f64>> {
    if factor == 0 {
        return Err(DpbError::InvalidParameter(
            "Upsample factor must be positive".to_string(),
        ));
    }

    if factor == 1 {
        return Ok(signal.to_owned());
    }

    let new_len = signal.len() * factor;
    let mut upsampled = Array1::zeros(new_len);

    for i in 0..signal.len() - 1 {
        let start = i * factor;
        let value_start = signal[i];
        let value_end = signal[i + 1];

        for j in 0..factor {
            let t = j as f64 / factor as f64;
            upsampled[start + j] = value_start * (1.0 - t) + value_end * t;
        }
    }

    // Handle last sample
    if signal.len() > 0 {
        let last_start = (signal.len() - 1) * factor;
        for j in 0..factor {
            if last_start + j < new_len {
                upsampled[last_start + j] = signal[signal.len() - 1];
            }
        }
    }

    Ok(upsampled)
}

/// Resamples using Fourier method (sinc interpolation).
pub fn resample_fourier(
    signal: ArrayView1<f64>,
    original_rate: f64,
    target_rate: f64,
) -> Result<Array1<f64>> {
    if original_rate <= 0.0 || target_rate <= 0.0 {
        return Err(DpbError::InvalidParameter(
            "Sample rates must be positive".to_string(),
        ));
    }

    // For now, fall back to linear interpolation
    // A full implementation would use FFT-based resampling
    resample_linear(signal, original_rate, target_rate)
}

/// Resamples to match a target length.
pub fn resample_to_length(signal: ArrayView1<f64>, target_len: usize) -> Result<Array1<f64>> {
    if target_len == 0 {
        return Err(DpbError::InvalidParameter(
            "Target length must be positive".to_string(),
        ));
    }

    if signal.len() == target_len {
        return Ok(signal.to_owned());
    }

    let mut resampled = Array1::zeros(target_len);

    for i in 0..target_len {
        let original_idx = i as f64 * (signal.len() - 1) as f64 / (target_len - 1) as f64;
        let idx_floor = original_idx.floor() as usize;
        let idx_ceil = (idx_floor + 1).min(signal.len() - 1);
        let fraction = original_idx - idx_floor as f64;

        resampled[i] = signal[idx_floor] * (1.0 - fraction) + signal[idx_ceil] * fraction;
    }

    Ok(resampled)
}

/// Applies anti-aliasing filter before downsampling.
pub fn resample_with_antialiasing(
    signal: ArrayView1<f64>,
    original_rate: f64,
    target_rate: f64,
) -> Result<Array1<f64>> {
    if original_rate <= 0.0 || target_rate <= 0.0 {
        return Err(DpbError::InvalidParameter(
            "Sample rates must be positive".to_string(),
        ));
    }

    // If downsampling, apply lowpass filter first
    if target_rate < original_rate {
        // Simple moving average as anti-aliasing filter
        let filter_size = (original_rate / target_rate).ceil() as usize;
        let filtered = moving_average(signal, filter_size)?;
        resample_linear(filtered.view(), original_rate, target_rate)
    } else {
        resample_linear(signal, original_rate, target_rate)
    }
}

/// Helper function for moving average.
fn moving_average(signal: ArrayView1<f64>, window_size: usize) -> Result<Array1<f64>> {
    if window_size == 0 {
        return Err(DpbError::InvalidParameter(
            "Window size must be positive".to_string(),
        ));
    }

    let mut result = Array1::zeros(signal.len());

    for i in 0..signal.len() {
        let start = i.saturating_sub(window_size / 2);
        let end = (i + window_size / 2 + 1).min(signal.len());
        let sum: f64 = signal.slice(ndarray::s![start..end]).sum();
        let count = end - start;
        result[i] = sum / count as f64;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_downsample() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let downsampled = downsample(signal.view(), 2).unwrap();

        assert_eq!(downsampled.len(), 3);
        assert_eq!(downsampled[0], 1.0);
        assert_eq!(downsampled[1], 3.0);
        assert_eq!(downsampled[2], 5.0);
    }

    #[test]
    fn test_upsample() {
        let signal = Array1::from_vec(vec![1.0, 3.0]);
        let upsampled = upsample(signal.view(), 2).unwrap();

        assert_eq!(upsampled.len(), 4);
        assert_eq!(upsampled[0], 1.0);
        assert_eq!(upsampled[1], 2.0);
        assert_eq!(upsampled[2], 3.0);
    }

    #[test]
    fn test_resample_linear() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let resampled = resample_linear(signal.view(), 100.0, 50.0).unwrap();

        // Should halve the length
        assert_eq!(resampled.len(), 2);
    }

    #[test]
    fn test_resample_to_length() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let resampled = resample_to_length(signal.view(), 8).unwrap();

        assert_eq!(resampled.len(), 8);
        assert_relative_eq!(resampled[0], 1.0);
    }

    #[test]
    fn test_resample_identity() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let resampled = resample_linear(signal.view(), 100.0, 100.0).unwrap();

        assert_eq!(resampled.len(), signal.len());
    }
}
