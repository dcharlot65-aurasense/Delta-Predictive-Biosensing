//! Signal quality metrics for biosignal analysis.

use super::MetricTrait;
use crate::error::{DpbError, Result};

/// Signal-to-Noise Ratio (SNR) in dB.
#[derive(Debug, Clone)]
pub struct SignalToNoiseRatio {
    signal_power: f64,
    noise_power: f64,
    count: f64,
}

impl SignalToNoiseRatio {
    pub fn new() -> Self {
        Self {
            signal_power: 0.0,
            noise_power: 0.0,
            count: 0.0,
        }
    }
}

impl Default for SignalToNoiseRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for SignalToNoiseRatio {
    fn name(&self) -> &str {
        "snr"
    }

    fn compute(&self, signal: &[f32], noise: &[f32]) -> Result<f64> {
        if signal.len() != noise.len() {
            return Err(DpbError::Other("Signal and noise must have the same length".to_string()));
        }
        if signal.is_empty() {
            return Ok(0.0);
        }

        let signal_power: f64 = signal.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>()
            / signal.len() as f64;

        let noise_power: f64 = noise.iter().map(|&n| (n as f64) * (n as f64)).sum::<f64>()
            / noise.len() as f64;

        if noise_power == 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(10.0 * (signal_power / noise_power).log10())
        }
    }

    fn update(&mut self, signal: &[f32], noise: &[f32]) {
        let sig_pow: f64 = signal.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let noi_pow: f64 = noise.iter().map(|&n| (n as f64) * (n as f64)).sum();

        self.signal_power += sig_pow;
        self.noise_power += noi_pow;
        self.count += signal.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 || self.noise_power == 0.0 {
            0.0
        } else {
            let sig_pow = self.signal_power / self.count;
            let noi_pow = self.noise_power / self.count;
            10.0 * (sig_pow / noi_pow).log10()
        }
    }

    fn reset(&mut self) {
        self.signal_power = 0.0;
        self.noise_power = 0.0;
        self.count = 0.0;
    }
}

/// Peak Signal-to-Noise Ratio (PSNR) in dB.
#[derive(Debug, Clone)]
pub struct PeakSignalToNoiseRatio {
    max_signal: f64,
    mse_sum: f64,
    count: f64,
}

impl PeakSignalToNoiseRatio {
    pub fn new() -> Self {
        Self {
            max_signal: 0.0,
            mse_sum: 0.0,
            count: 0.0,
        }
    }
}

impl Default for PeakSignalToNoiseRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for PeakSignalToNoiseRatio {
    fn name(&self) -> &str {
        "psnr"
    }

    fn compute(&self, signal: &[f32], reconstructed: &[f32]) -> Result<f64> {
        if signal.len() != reconstructed.len() {
            return Err(DpbError::Other("Signal and reconstructed must have the same length".to_string()));
        }
        if signal.is_empty() {
            return Ok(0.0);
        }

        let max_signal = signal.iter().map(|&s| s.abs()).fold(0.0f32, f32::max) as f64;

        let mse: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(s, r)| {
                let diff = (*s as f64) - (*r as f64);
                diff * diff
            })
            .sum::<f64>()
            / signal.len() as f64;

        if mse == 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(10.0 * ((max_signal * max_signal) / mse).log10())
        }
    }

    fn update(&mut self, signal: &[f32], reconstructed: &[f32]) {
        let max_sig = signal.iter().map(|&s| s.abs()).fold(0.0f32, f32::max) as f64;
        if max_sig > self.max_signal {
            self.max_signal = max_sig;
        }

        let mse_part: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(s, r)| {
                let diff = (*s as f64) - (*r as f64);
                diff * diff
            })
            .sum();

        self.mse_sum += mse_part;
        self.count += signal.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 || self.mse_sum == 0.0 {
            0.0
        } else {
            let mse = self.mse_sum / self.count;
            10.0 * ((self.max_signal * self.max_signal) / mse).log10()
        }
    }

    fn reset(&mut self) {
        self.max_signal = 0.0;
        self.mse_sum = 0.0;
        self.count = 0.0;
    }
}

/// Cross-correlation coefficient.
#[derive(Debug, Clone)]
pub struct CrossCorrelation {
    sum_x: f64,
    sum_y: f64,
    sum_xy: f64,
    sum_x_sq: f64,
    sum_y_sq: f64,
    count: f64,
}

impl CrossCorrelation {
    pub fn new() -> Self {
        Self {
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xy: 0.0,
            sum_x_sq: 0.0,
            sum_y_sq: 0.0,
            count: 0.0,
        }
    }
}

impl Default for CrossCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for CrossCorrelation {
    fn name(&self) -> &str {
        "cross_correlation"
    }

    fn compute(&self, signal1: &[f32], signal2: &[f32]) -> Result<f64> {
        if signal1.len() != signal2.len() {
            return Err(DpbError::Other("Signals must have the same length".to_string()));
        }
        if signal1.is_empty() {
            return Ok(0.0);
        }

        let n = signal1.len() as f64;
        let sum_x: f64 = signal1.iter().map(|&x| x as f64).sum();
        let sum_y: f64 = signal2.iter().map(|&y| y as f64).sum();
        let sum_xy: f64 = signal1
            .iter()
            .zip(signal2.iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum();
        let sum_x_sq: f64 = signal1.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let sum_y_sq: f64 = signal2.iter().map(|&y| (y as f64) * (y as f64)).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn update(&mut self, signal1: &[f32], signal2: &[f32]) {
        for (x, y) in signal1.iter().zip(signal2.iter()) {
            let x_f64 = *x as f64;
            let y_f64 = *y as f64;

            self.sum_x += x_f64;
            self.sum_y += y_f64;
            self.sum_xy += x_f64 * y_f64;
            self.sum_x_sq += x_f64 * x_f64;
            self.sum_y_sq += y_f64 * y_f64;
            self.count += 1.0;
        }
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            return 0.0;
        }

        let numerator = self.count * self.sum_xy - self.sum_x * self.sum_y;
        let denominator = ((self.count * self.sum_x_sq - self.sum_x * self.sum_x)
            * (self.count * self.sum_y_sq - self.sum_y * self.sum_y))
            .sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn reset(&mut self) {
        self.sum_x = 0.0;
        self.sum_y = 0.0;
        self.sum_xy = 0.0;
        self.sum_x_sq = 0.0;
        self.sum_y_sq = 0.0;
        self.count = 0.0;
    }
}

/// Coherence (frequency domain correlation).
#[derive(Debug, Clone)]
pub struct Coherence {
    cross_spec_sum: f64,
    auto_spec1_sum: f64,
    auto_spec2_sum: f64,
    count: f64,
}

impl Coherence {
    pub fn new() -> Self {
        Self {
            cross_spec_sum: 0.0,
            auto_spec1_sum: 0.0,
            auto_spec2_sum: 0.0,
            count: 0.0,
        }
    }
}

impl Default for Coherence {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for Coherence {
    fn name(&self) -> &str {
        "coherence"
    }

    fn compute(&self, signal1: &[f32], signal2: &[f32]) -> Result<f64> {
        if signal1.len() != signal2.len() {
            return Err(DpbError::Other("Signals must have the same length".to_string()));
        }
        if signal1.is_empty() {
            return Ok(0.0);
        }

        // Simplified coherence as cross-correlation squared
        // In practice, this would involve FFT and frequency domain analysis
        let cross_corr = CrossCorrelation::new().compute(signal1, signal2)?;
        Ok(cross_corr * cross_corr)
    }

    fn update(&mut self, signal1: &[f32], signal2: &[f32]) {
        // Simplified: accumulate cross-correlation squared
        let cross_corr = CrossCorrelation::new().compute(signal1, signal2).unwrap_or(0.0);
        self.cross_spec_sum += cross_corr * cross_corr;
        self.count += 1.0;
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            0.0
        } else {
            self.cross_spec_sum / self.count
        }
    }

    fn reset(&mut self) {
        self.cross_spec_sum = 0.0;
        self.auto_spec1_sum = 0.0;
        self.auto_spec2_sum = 0.0;
        self.count = 0.0;
    }
}

/// Dynamic Time Warping (DTW) distance.
#[derive(Debug, Clone)]
pub struct DtwDistance {
    accumulated_distance: f64,
    count: f64,
}

impl DtwDistance {
    pub fn new() -> Self {
        Self {
            accumulated_distance: 0.0,
            count: 0.0,
        }
    }

    fn compute_dtw(signal1: &[f32], signal2: &[f32]) -> f64 {
        let n = signal1.len();
        let m = signal2.len();

        if n == 0 || m == 0 {
            return 0.0;
        }

        // Initialize DTW matrix
        let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
        dtw[0][0] = 0.0;

        // Fill DTW matrix
        for i in 1..=n {
            for j in 1..=m {
                let cost = ((signal1[i - 1] - signal2[j - 1]) as f64).abs();
                dtw[i][j] = cost + dtw[i - 1][j].min(dtw[i][j - 1]).min(dtw[i - 1][j - 1]);
            }
        }

        dtw[n][m]
    }
}

impl Default for DtwDistance {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for DtwDistance {
    fn name(&self) -> &str {
        "dtw_distance"
    }

    fn compute(&self, signal1: &[f32], signal2: &[f32]) -> Result<f64> {
        if signal1.is_empty() || signal2.is_empty() {
            return Ok(0.0);
        }
        Ok(Self::compute_dtw(signal1, signal2))
    }

    fn update(&mut self, signal1: &[f32], signal2: &[f32]) {
        self.accumulated_distance += Self::compute_dtw(signal1, signal2);
        self.count += 1.0;
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            0.0
        } else {
            self.accumulated_distance / self.count
        }
    }

    fn reset(&mut self) {
        self.accumulated_distance = 0.0;
        self.count = 0.0;
    }
}

/// Fréchet distance (continuous DTW variant).
#[derive(Debug, Clone)]
pub struct FrechetDistance {
    accumulated_distance: f64,
    count: f64,
}

impl FrechetDistance {
    pub fn new() -> Self {
        Self {
            accumulated_distance: 0.0,
            count: 0.0,
        }
    }

    fn compute_frechet(signal1: &[f32], signal2: &[f32]) -> f64 {
        let n = signal1.len();
        let m = signal2.len();

        if n == 0 || m == 0 {
            return 0.0;
        }

        // Simplified Fréchet: Use dynamic programming similar to DTW but with max instead of sum
        let mut ca = vec![vec![f64::NEG_INFINITY; m]; n];

        ca[0][0] = ((signal1[0] - signal2[0]) as f64).abs();

        for i in 1..n {
            ca[i][0] = ((signal1[i] - signal2[0]) as f64).abs().max(ca[i - 1][0]);
        }

        for j in 1..m {
            ca[0][j] = ((signal1[0] - signal2[j]) as f64).abs().max(ca[0][j - 1]);
        }

        for i in 1..n {
            for j in 1..m {
                let dist = ((signal1[i] - signal2[j]) as f64).abs();
                ca[i][j] = dist.max(
                    ca[i - 1][j]
                        .min(ca[i][j - 1])
                        .min(ca[i - 1][j - 1])
                );
            }
        }

        ca[n - 1][m - 1]
    }
}

impl Default for FrechetDistance {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for FrechetDistance {
    fn name(&self) -> &str {
        "frechet_distance"
    }

    fn compute(&self, signal1: &[f32], signal2: &[f32]) -> Result<f64> {
        if signal1.is_empty() || signal2.is_empty() {
            return Ok(0.0);
        }
        Ok(Self::compute_frechet(signal1, signal2))
    }

    fn update(&mut self, signal1: &[f32], signal2: &[f32]) {
        self.accumulated_distance += Self::compute_frechet(signal1, signal2);
        self.count += 1.0;
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            0.0
        } else {
            self.accumulated_distance / self.count
        }
    }

    fn reset(&mut self) {
        self.accumulated_distance = 0.0;
        self.count = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snr() {
        let mut snr = SignalToNoiseRatio::new();
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let noise = vec![0.1, 0.1, 0.1, 0.1];

        let result = snr.compute(&signal, &noise).unwrap();
        assert!(result > 0.0);

        snr.update(&signal, &noise);
        assert!(snr.result() > 0.0);
    }

    #[test]
    fn test_psnr() {
        let psnr = PeakSignalToNoiseRatio::new();
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let reconstructed = vec![1.0, 2.0, 3.0, 4.0];

        let result = psnr.compute(&signal, &reconstructed).unwrap();
        assert_eq!(result, f64::INFINITY);  // Perfect reconstruction
    }

    #[test]
    fn test_cross_correlation() {
        let cc = CrossCorrelation::new();
        let signal1 = vec![1.0, 2.0, 3.0, 4.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0];

        let result = cc.compute(&signal1, &signal2).unwrap();
        assert!((result - 1.0).abs() < 1e-6);  // Perfect correlation
    }

    #[test]
    fn test_coherence() {
        let coh = Coherence::new();
        let signal1 = vec![1.0, 2.0, 3.0, 4.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0];

        let result = coh.compute(&signal1, &signal2).unwrap();
        assert!(result > 0.9);  // High coherence for identical signals
    }

    #[test]
    fn test_dtw_distance() {
        let dtw = DtwDistance::new();
        let signal1 = vec![1.0, 2.0, 3.0, 4.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0];

        let result = dtw.compute(&signal1, &signal2).unwrap();
        assert_eq!(result, 0.0);  // Identical signals
    }

    #[test]
    fn test_frechet_distance() {
        let fd = FrechetDistance::new();
        let signal1 = vec![1.0, 2.0, 3.0, 4.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0];

        let result = fd.compute(&signal1, &signal2).unwrap();
        assert_eq!(result, 0.0);  // Identical signals
    }
}
