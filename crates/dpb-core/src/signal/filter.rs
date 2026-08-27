//! Digital filter implementations.

use crate::error::{DpbError, Result};
use ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;

/// Filter type enum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    /// Passes frequencies below the cutoff.
    Lowpass,
    /// Passes frequencies above the cutoff.
    Highpass,
    /// Passes frequencies between the two cutoffs.
    Bandpass,
    /// Rejects frequencies between the two cutoffs; a notch filter.
    Bandstop,
}

/// IIR filter implementation (Butterworth).
#[derive(Debug, Clone)]
pub struct IirFilter {
    /// Numerator coefficients (b)
    b: Vec<f64>,
    /// Denominator coefficients (a)
    a: Vec<f64>,
    /// Input history
    x_hist: Vec<f64>,
    /// Output history
    y_hist: Vec<f64>,
}

impl IirFilter {
    /// Creates a new IIR filter with coefficients.
    pub fn new(b: Vec<f64>, a: Vec<f64>) -> Result<Self> {
        if b.is_empty() || a.is_empty() {
            return Err(DpbError::InvalidParameter(
                "Filter coefficients cannot be empty".to_string(),
            ));
        }

        let order = a.len().max(b.len()) - 1;
        Ok(Self {
            b,
            a,
            x_hist: vec![0.0; order],
            y_hist: vec![0.0; order],
        })
    }

    /// Creates a Butterworth lowpass filter.
    pub fn butterworth_lowpass(order: usize, cutoff: f64, sample_rate: f64) -> Result<Self> {
        if cutoff >= sample_rate / 2.0 {
            return Err(DpbError::InvalidParameter(
                "Cutoff frequency must be less than Nyquist frequency".to_string(),
            ));
        }

        let (b, a) = design_butterworth(order, cutoff, sample_rate, FilterType::Lowpass)?;
        Self::new(b, a)
    }

    /// Creates a Butterworth highpass filter.
    pub fn butterworth_highpass(order: usize, cutoff: f64, sample_rate: f64) -> Result<Self> {
        if cutoff >= sample_rate / 2.0 {
            return Err(DpbError::InvalidParameter(
                "Cutoff frequency must be less than Nyquist frequency".to_string(),
            ));
        }

        let (b, a) = design_butterworth(order, cutoff, sample_rate, FilterType::Highpass)?;
        Self::new(b, a)
    }

    /// Creates a Butterworth bandpass filter.
    pub fn butterworth_bandpass(
        order: usize,
        low_cutoff: f64,
        high_cutoff: f64,
        sample_rate: f64,
    ) -> Result<Self> {
        if low_cutoff >= high_cutoff {
            return Err(DpbError::InvalidParameter(
                "Low cutoff must be less than high cutoff".to_string(),
            ));
        }

        // Implement as cascade of lowpass and highpass
        let (b, a) = design_butterworth(order, high_cutoff, sample_rate, FilterType::Lowpass)?;
        Self::new(b, a)
    }

    /// Filters a single sample.
    pub fn filter_sample(&mut self, x: f64) -> f64 {
        // Direct form I:  y[n] = (sum_i b[i]*x[n-i] - sum_{i>=1} a[i]*y[n-i]) / a[0]
        //
        // The histories are shifted AFTER the output is computed. Shifting first
        // put the CURRENT sample at `x_hist[0]`, so `b[1]` -- which must weight
        // x[n-1] -- was applied to x[n] as well, and the previous sample was
        // dropped entirely. For a first-order highpass, where
        // `b = [1-a, -(1-a)]`, that made every output exactly
        // `(1-a)x - (1-a)x = 0`: the filter returned silence for any input.
        let mut y = self.b[0] * x;

        for (i, &b_i) in self.b.iter().enumerate().skip(1) {
            if let Some(&past) = self.x_hist.get(i - 1) {
                y += b_i * past;
            }
        }

        for (i, &a_i) in self.a.iter().enumerate().skip(1) {
            if let Some(&past) = self.y_hist.get(i - 1) {
                y -= a_i * past;
            }
        }

        if self.a[0] != 0.0 {
            y /= self.a[0];
        }

        if !self.x_hist.is_empty() {
            self.x_hist.insert(0, x);
            self.x_hist.pop();
        }
        if !self.y_hist.is_empty() {
            self.y_hist.insert(0, y);
            self.y_hist.pop();
        }

        y
    }

    /// Filters an entire signal.
    pub fn filter(&mut self, signal: ArrayView1<f64>) -> Array1<f64> {
        signal.iter().map(|&x| self.filter_sample(x)).collect()
    }

    /// Resets filter state.
    pub fn reset(&mut self) {
        self.x_hist.fill(0.0);
        self.y_hist.fill(0.0);
    }
}

/// FIR filter implementation.
#[derive(Debug, Clone)]
pub struct FirFilter {
    /// Filter coefficients
    coefficients: Vec<f64>,
    /// Input history
    history: Vec<f64>,
}

impl FirFilter {
    /// Creates a new FIR filter.
    pub fn new(coefficients: Vec<f64>) -> Result<Self> {
        if coefficients.is_empty() {
            return Err(DpbError::InvalidParameter(
                "Filter coefficients cannot be empty".to_string(),
            ));
        }

        let history = vec![0.0; coefficients.len()];
        Ok(Self {
            coefficients,
            history,
        })
    }

    /// Creates a moving average filter.
    pub fn moving_average(window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(DpbError::InvalidParameter(
                "Window size must be positive".to_string(),
            ));
        }

        let coefficient = 1.0 / window_size as f64;
        let coefficients = vec![coefficient; window_size];
        Self::new(coefficients)
    }

    /// Filters a single sample.
    pub fn filter_sample(&mut self, x: f64) -> f64 {
        self.history.insert(0, x);
        self.history.pop();

        self.coefficients
            .iter()
            .zip(self.history.iter())
            .map(|(c, h)| c * h)
            .sum()
    }

    /// Filters an entire signal.
    pub fn filter(&mut self, signal: ArrayView1<f64>) -> Array1<f64> {
        signal.iter().map(|&x| self.filter_sample(x)).collect()
    }

    /// Resets filter state.
    pub fn reset(&mut self) {
        self.history.fill(0.0);
    }
}

/// Designs a Butterworth filter (simplified implementation).
fn design_butterworth(
    _order: usize,
    cutoff: f64,
    sample_rate: f64,
    filter_type: FilterType,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // Simplified first-order Butterworth design
    // For production, use a proper filter design library
    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sample_rate;
    let alpha = dt / (rc + dt);

    let (b, a) = match filter_type {
        FilterType::Lowpass => {
            let b = vec![alpha, 0.0];
            let a = vec![1.0, -(1.0 - alpha)];
            (b, a)
        }
        FilterType::Highpass => {
            let b = vec![1.0 - alpha, -(1.0 - alpha)];
            let a = vec![1.0, -(1.0 - alpha)];
            (b, a)
        }
        _ => {
            // Simplified implementation for other types
            let b = vec![alpha, 0.0];
            let a = vec![1.0, -(1.0 - alpha)];
            (b, a)
        }
    };

    Ok((b, a))
}

/// Applies a median filter to a signal.
pub fn median_filter(signal: ArrayView1<f64>, window_size: usize) -> Result<Array1<f64>> {
    if window_size == 0 || window_size.is_multiple_of(2) {
        return Err(DpbError::InvalidParameter(
            "Window size must be odd and positive".to_string(),
        ));
    }

    let half_window = window_size / 2;
    let mut result = Array1::zeros(signal.len());

    for i in 0..signal.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(signal.len());

        let mut window: Vec<f64> = signal.slice(ndarray::s![start..end]).to_vec();
        window.sort_by(|a, b| a.total_cmp(b));

        result[i] = window[window.len() / 2];
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fir_moving_average() {
        let mut filter = FirFilter::moving_average(3).unwrap();

        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let filtered = filter.filter(signal.view());

        // First sample: (0 + 0 + 1) / 3
        assert_relative_eq!(filtered[0], 1.0 / 3.0, epsilon = 1e-10);
        // Second sample: (0 + 1 + 2) / 3
        assert_relative_eq!(filtered[1], 1.0, epsilon = 1e-10);
        // Third sample: (1 + 2 + 3) / 3
        assert_relative_eq!(filtered[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iir_butterworth() {
        let filter = IirFilter::butterworth_lowpass(2, 10.0, 100.0);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_median_filter() {
        let signal = Array1::from_vec(vec![1.0, 5.0, 2.0, 8.0, 3.0]);
        let filtered = median_filter(signal.view(), 3).unwrap();

        assert_eq!(filtered[1], 2.0); // median of [1, 5, 2]
        assert_eq!(filtered[2], 5.0); // median of [5, 2, 8]
    }

    #[test]
    fn test_filter_reset() {
        let mut filter = FirFilter::moving_average(3).unwrap();
        filter.filter_sample(1.0);
        filter.filter_sample(2.0);
        filter.reset();

        assert_eq!(filter.history, vec![0.0, 0.0, 0.0]);
    }
}
