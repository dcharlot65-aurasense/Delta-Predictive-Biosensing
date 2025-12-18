//! EEG frequency band analysis.
//!
//! This module provides functionality for analyzing EEG signals in different frequency bands
//! and computing clinically relevant biomarkers.

use crate::error::{DpbError, Result};
use crate::signal::fft::FftProcessor;
use ndarray::ArrayView1;

/// Standard EEG frequency bands.
///
/// The conventional division of EEG frequencies into named bands, each associated
/// with different cognitive and physiological states.
#[derive(Debug, Clone, Copy)]
pub struct EegBands {
    /// Delta band: 0.5-4 Hz (deep sleep, unconsciousness)
    pub delta: (f64, f64),
    /// Theta band: 4-8 Hz (drowsiness, meditation)
    pub theta: (f64, f64),
    /// Alpha band: 8-13 Hz (relaxed wakefulness, eyes closed)
    pub alpha: (f64, f64),
    /// Beta band: 13-30 Hz (active thinking, concentration)
    pub beta: (f64, f64),
    /// Gamma band: 30-100 Hz (cognitive processing, binding)
    pub gamma: (f64, f64),
}

impl Default for EegBands {
    fn default() -> Self {
        Self {
            delta: (0.5, 4.0),
            theta: (4.0, 8.0),
            alpha: (8.0, 13.0),
            beta: (13.0, 30.0),
            gamma: (30.0, 100.0),
        }
    }
}

/// Power values for each EEG frequency band.
///
/// Contains absolute power values (in units of signal²) for each band,
/// plus the total power across all frequencies.
#[derive(Debug, Clone, Default)]
pub struct BandPowers {
    /// Delta band power (0.5-4 Hz)
    pub delta: f64,
    /// Theta band power (4-8 Hz)
    pub theta: f64,
    /// Alpha band power (8-13 Hz)
    pub alpha: f64,
    /// Beta band power (13-30 Hz)
    pub beta: f64,
    /// Gamma band power (30-100 Hz)
    pub gamma: f64,
    /// Total power across all frequencies
    pub total: f64,
}

/// Extracts power in a specific frequency band using FFT.
///
/// Computes the power spectral density and integrates power within the specified
/// frequency range.
///
/// # Arguments
///
/// * `signal` - The input EEG signal
/// * `sample_rate` - Sampling frequency in Hz
/// * `band` - Frequency range as (low_freq, high_freq) in Hz
///
/// # Returns
///
/// The integrated power within the specified frequency band.
///
/// # Example
///
/// ```
/// use dpb_core::signal::eeg::extract_band_power;
///
/// let eeg_signal = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1]; // example data
/// let sample_rate = 250.0; // Hz
/// let alpha_band = (8.0, 13.0); // Hz
/// let power = extract_band_power(&eeg_signal, sample_rate, alpha_band).unwrap();
/// ```
pub fn extract_band_power(signal: &[f64], sample_rate: f64, band: (f64, f64)) -> Result<f64> {
    if signal.is_empty() {
        return Err(DpbError::InvalidDimensions(
            "Signal cannot be empty".to_string(),
        ));
    }

    if sample_rate <= 0.0 {
        return Err(DpbError::InvalidParameter(
            "Sample rate must be positive".to_string(),
        ));
    }

    let (low_freq, high_freq) = band;
    if low_freq < 0.0 || high_freq <= low_freq || high_freq > sample_rate / 2.0 {
        return Err(DpbError::InvalidParameter(format!(
            "Invalid frequency band: ({}, {}). Must satisfy 0 <= low < high <= Nyquist ({})",
            low_freq,
            high_freq,
            sample_rate / 2.0
        )));
    }

    // Create FFT processor and compute PSD
    let mut fft_processor = FftProcessor::new();
    let signal_array = ndarray::Array1::from_vec(signal.to_vec());
    let psd = fft_processor.psd(signal_array.view())?;

    // Compute frequency bins
    let n = signal.len();
    let freq_resolution = sample_rate / n as f64;

    // Find indices corresponding to the frequency band
    let low_idx = (low_freq / freq_resolution).ceil() as usize;
    let high_idx = ((high_freq / freq_resolution).floor() as usize).min(psd.len() - 1);

    // Integrate power in the band
    let band_power: f64 = psd
        .slice(ndarray::s![low_idx..=high_idx])
        .iter()
        .sum::<f64>()
        * freq_resolution;

    Ok(band_power)
}

/// Computes power in all standard EEG frequency bands.
///
/// Analyzes the signal across delta, theta, alpha, beta, and gamma bands
/// using the default band definitions.
///
/// # Arguments
///
/// * `signal` - The input EEG signal
/// * `sample_rate` - Sampling frequency in Hz
///
/// # Returns
///
/// A `BandPowers` struct containing power values for all bands and total power.
///
/// # Example
///
/// ```
/// use dpb_core::signal::eeg::compute_band_powers;
///
/// let eeg_signal = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1]; // example data
/// let sample_rate = 250.0; // Hz
/// let powers = compute_band_powers(&eeg_signal, sample_rate).unwrap();
/// println!("Alpha power: {}", powers.alpha);
/// ```
pub fn compute_band_powers(signal: &[f64], sample_rate: f64) -> Result<BandPowers> {
    let bands = EegBands::default();

    let delta = extract_band_power(signal, sample_rate, bands.delta)?;
    let theta = extract_band_power(signal, sample_rate, bands.theta)?;
    let alpha = extract_band_power(signal, sample_rate, bands.alpha)?;
    let beta = extract_band_power(signal, sample_rate, bands.beta)?;
    let gamma = extract_band_power(signal, sample_rate, bands.gamma)?;

    let total = delta + theta + alpha + beta + gamma;

    Ok(BandPowers {
        delta,
        theta,
        alpha,
        beta,
        gamma,
        total,
    })
}

/// Computes relative band power as a percentage of total power.
///
/// Normalizes absolute band power to a percentage, which is useful for
/// comparing across subjects or sessions with different absolute power levels.
///
/// # Arguments
///
/// * `band_power` - Absolute power in the band of interest
/// * `total_power` - Total power across all frequencies
///
/// # Returns
///
/// Relative power as a value between 0.0 and 1.0.
///
/// # Example
///
/// ```
/// use dpb_core::signal::eeg::relative_band_power;
///
/// let alpha_power = 50.0;
/// let total_power = 200.0;
/// let relative_alpha = relative_band_power(alpha_power, total_power);
/// assert_eq!(relative_alpha, 0.25); // 25%
/// ```
pub fn relative_band_power(band_power: f64, total_power: f64) -> f64 {
    if total_power == 0.0 {
        return 0.0;
    }
    band_power / total_power
}

/// Computes the theta/beta ratio, a biomarker for ADHD.
///
/// The theta/beta ratio (TBR) is elevated in individuals with ADHD compared to
/// neurotypical controls. Higher ratios indicate more theta activity relative to
/// beta, which may reflect attentional deficits.
///
/// # Arguments
///
/// * `signal` - The input EEG signal
/// * `sample_rate` - Sampling frequency in Hz
///
/// # Returns
///
/// The theta/beta power ratio.
///
/// # Example
///
/// ```
/// use dpb_core::signal::eeg::theta_beta_ratio;
///
/// let eeg_signal = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1]; // example data
/// let sample_rate = 250.0; // Hz
/// let tbr = theta_beta_ratio(&eeg_signal, sample_rate).unwrap();
/// println!("Theta/Beta ratio: {:.2}", tbr);
/// ```
pub fn theta_beta_ratio(signal: &[f64], sample_rate: f64) -> Result<f64> {
    let bands = EegBands::default();

    let theta_power = extract_band_power(signal, sample_rate, bands.theta)?;
    let beta_power = extract_band_power(signal, sample_rate, bands.beta)?;

    if beta_power == 0.0 {
        return Err(DpbError::InvalidParameter(
            "Beta power is zero, cannot compute ratio".to_string(),
        ));
    }

    Ok(theta_power / beta_power)
}

/// Computes frontal alpha asymmetry, a biomarker for depression.
///
/// Alpha asymmetry is the difference in alpha power between left and right
/// frontal electrodes. Greater relative right frontal activity (positive values)
/// is associated with depression and withdrawal motivation.
///
/// The asymmetry is computed as: ln(right_alpha) - ln(left_alpha)
///
/// # Arguments
///
/// * `left` - EEG signal from left frontal electrode (e.g., F3)
/// * `right` - EEG signal from right frontal electrode (e.g., F4)
/// * `sample_rate` - Sampling frequency in Hz
///
/// # Returns
///
/// Alpha asymmetry score. Positive values indicate greater relative right activity,
/// negative values indicate greater relative left activity.
///
/// # Example
///
/// ```
/// use dpb_core::signal::eeg::alpha_asymmetry;
///
/// let left_eeg = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1]; // F3
/// let right_eeg = vec![0.15, 0.25, -0.15, 0.35, -0.25, 0.15]; // F4
/// let sample_rate = 250.0; // Hz
/// let asymmetry = alpha_asymmetry(&left_eeg, &right_eeg, sample_rate).unwrap();
/// println!("Alpha asymmetry: {:.3}", asymmetry);
/// ```
pub fn alpha_asymmetry(left: &[f64], right: &[f64], sample_rate: f64) -> Result<f64> {
    if left.len() != right.len() {
        return Err(DpbError::InvalidDimensions(
            "Left and right signals must have the same length".to_string(),
        ));
    }

    let bands = EegBands::default();

    let left_alpha = extract_band_power(left, sample_rate, bands.alpha)?;
    let right_alpha = extract_band_power(right, sample_rate, bands.alpha)?;

    if left_alpha <= 0.0 || right_alpha <= 0.0 {
        return Err(DpbError::InvalidParameter(
            "Alpha power must be positive for log transformation".to_string(),
        ));
    }

    // ln(right) - ln(left)
    Ok(right_alpha.ln() - left_alpha.ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_eeg_bands_default() {
        let bands = EegBands::default();
        assert_eq!(bands.delta, (0.5, 4.0));
        assert_eq!(bands.theta, (4.0, 8.0));
        assert_eq!(bands.alpha, (8.0, 13.0));
        assert_eq!(bands.beta, (13.0, 30.0));
        assert_eq!(bands.gamma, (30.0, 100.0));
    }

    #[test]
    fn test_extract_band_power() {
        // Generate a simple test signal with known frequency content
        let sample_rate = 256.0;
        let duration = 2.0;
        let n = (sample_rate * duration) as usize;

        // Create signal with 10 Hz component (in alpha band)
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            })
            .collect();

        let alpha_power = extract_band_power(&signal, sample_rate, (8.0, 13.0)).unwrap();
        let beta_power = extract_band_power(&signal, sample_rate, (13.0, 30.0)).unwrap();

        // Alpha power should be much greater than beta power
        assert!(alpha_power > beta_power * 10.0);
    }

    #[test]
    fn test_compute_band_powers() {
        let sample_rate = 256.0;
        let signal: Vec<f64> = (0..512).map(|i| (i as f64).sin()).collect();

        let powers = compute_band_powers(&signal, sample_rate).unwrap();

        // Check that total is sum of all bands
        let sum = powers.delta + powers.theta + powers.alpha + powers.beta + powers.gamma;
        assert_relative_eq!(powers.total, sum, epsilon = 1e-6);

        // All powers should be non-negative
        assert!(powers.delta >= 0.0);
        assert!(powers.theta >= 0.0);
        assert!(powers.alpha >= 0.0);
        assert!(powers.beta >= 0.0);
        assert!(powers.gamma >= 0.0);
    }

    #[test]
    fn test_relative_band_power() {
        let band_power = 50.0;
        let total_power = 200.0;
        let relative = relative_band_power(band_power, total_power);
        assert_relative_eq!(relative, 0.25);
    }

    #[test]
    fn test_relative_band_power_zero_total() {
        let relative = relative_band_power(50.0, 0.0);
        assert_eq!(relative, 0.0);
    }

    #[test]
    fn test_theta_beta_ratio() {
        let sample_rate = 256.0;
        let n = 512;

        // Create signal with both theta and beta components
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                // Theta component (6 Hz) + Beta component (20 Hz)
                (2.0 * std::f64::consts::PI * 6.0 * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 20.0 * t).sin()
            })
            .collect();

        let ratio = theta_beta_ratio(&signal, sample_rate).unwrap();
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_alpha_asymmetry() {
        let sample_rate = 256.0;
        let n = 512;

        // Left hemisphere with 10 Hz alpha
        let left: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            })
            .collect();

        // Right hemisphere with stronger 10 Hz alpha (1.5x amplitude)
        let right: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                1.5 * (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            })
            .collect();

        let asymmetry = alpha_asymmetry(&left, &right, sample_rate).unwrap();

        // Right has more power, so asymmetry should be positive
        assert!(asymmetry > 0.0);
    }

    #[test]
    fn test_alpha_asymmetry_length_mismatch() {
        let left = vec![1.0, 2.0, 3.0];
        let right = vec![1.0, 2.0];
        let result = alpha_asymmetry(&left, &right, 256.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_band_power_invalid_params() {
        let signal = vec![1.0, 2.0, 3.0];

        // Invalid sample rate
        assert!(extract_band_power(&signal, -10.0, (8.0, 13.0)).is_err());

        // Empty signal
        assert!(extract_band_power(&[], 256.0, (8.0, 13.0)).is_err());

        // Invalid frequency range (low >= high)
        assert!(extract_band_power(&signal, 256.0, (13.0, 8.0)).is_err());

        // Frequency exceeds Nyquist
        assert!(extract_band_power(&signal, 256.0, (8.0, 200.0)).is_err());
    }
}
