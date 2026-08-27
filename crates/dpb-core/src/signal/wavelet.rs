//! Wavelet transform operations for time-frequency analysis.

use ndarray::Array2;
use std::f64::consts::PI;

/// Wavelet family types for wavelet transforms.
///
/// Every variant carries only `f64`/`u8` parameters, so this is `Copy`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveletFamily {
    /// Morlet wavelet with center frequency parameter omega0 (typically 5.0-7.0)
    Morlet {
        /// Centre frequency, typically 5.0-7.0. Higher values trade time
        /// resolution for frequency resolution.
        omega0: f64,
    },
    /// Mexican Hat wavelet (second derivative of Gaussian)
    MexicanHat,
    /// Daubechies wavelet with order (number of vanishing moments)
    Daubechies(u8),
    /// Symlet wavelet with order
    Symlet(u8),
    /// Coiflet wavelet with order
    Coiflet(u8),
}

impl WaveletFamily {
    /// Generates wavelet function at given scale and time points.
    fn generate(&self, t: &[f64], scale: f64) -> Vec<f64> {
        match self {
            WaveletFamily::Morlet { omega0 } => {
                t.iter()
                    .map(|&time| {
                        let scaled_t = time / scale;
                        let envelope = (-scaled_t * scaled_t / 2.0).exp();
                        let wave = (omega0 * scaled_t).cos();
                        envelope * wave / scale.sqrt()
                    })
                    .collect()
            }
            WaveletFamily::MexicanHat => {
                t.iter()
                    .map(|&time| {
                        let scaled_t = time / scale;
                        let t2 = scaled_t * scaled_t;
                        let norm = 2.0 / (3.0_f64.sqrt() * PI.powf(0.25));
                        norm * (1.0 - t2) * (-t2 / 2.0).exp() / scale.sqrt()
                    })
                    .collect()
            }
            WaveletFamily::Daubechies(order) => {
                // Simplified Daubechies approximation (real implementation would use filter coefficients)
                self.daubechies_approx(t, scale, *order)
            }
            WaveletFamily::Symlet(order) => {
                // Symlets are nearly symmetric Daubechies wavelets
                self.daubechies_approx(t, scale, *order)
            }
            WaveletFamily::Coiflet(order) => {
                // Coiflets have more vanishing moments for both wavelet and scaling function
                self.coiflet_approx(t, scale, *order)
            }
        }
    }

    fn daubechies_approx(&self, t: &[f64], scale: f64, order: u8) -> Vec<f64> {
        // Simplified approximation using modulated Gaussian
        let width = 2.0 * order as f64;
        t.iter()
            .map(|&time| {
                let scaled_t = time / scale;
                if scaled_t.abs() > width / 2.0 {
                    0.0
                } else {
                    let envelope = (-(scaled_t * scaled_t) / (width * width / 4.0)).exp();
                    let oscillation = (2.0 * PI * order as f64 * scaled_t / width).sin();
                    envelope * oscillation / scale.sqrt()
                }
            })
            .collect()
    }

    fn coiflet_approx(&self, t: &[f64], scale: f64, order: u8) -> Vec<f64> {
        // Simplified approximation with slightly wider support
        let width = 3.0 * order as f64;
        t.iter()
            .map(|&time| {
                let scaled_t = time / scale;
                if scaled_t.abs() > width / 2.0 {
                    0.0
                } else {
                    let envelope = (-(scaled_t * scaled_t) / (width * width / 4.0)).exp();
                    let oscillation = (2.0 * PI * order as f64 * scaled_t / width).cos();
                    envelope * oscillation / scale.sqrt()
                }
            })
            .collect()
    }
}

/// Continuous Wavelet Transform processor.
pub struct ContinuousWaveletTransform {
    wavelet: WaveletFamily,
    scales: Vec<f64>,
    sample_rate: f64,
}

impl ContinuousWaveletTransform {
    /// Creates a new CWT processor.
    ///
    /// # Arguments
    /// * `wavelet` - The wavelet family to use
    /// * `scales` - Vector of scales for the transform
    /// * `sample_rate` - Sampling rate of the signal in Hz
    pub fn new(wavelet: WaveletFamily, scales: Vec<f64>, sample_rate: f64) -> Self {
        Self {
            wavelet,
            scales,
            sample_rate,
        }
    }

    /// Computes the continuous wavelet transform.
    ///
    /// Returns a 2D array with shape (scales, time) containing the transform coefficients.
    pub fn transform(&self, signal: &[f64]) -> Array2<f64> {
        let n_scales = self.scales.len();
        let n_samples = signal.len();
        let mut result = Array2::zeros((n_scales, n_samples));

        // Generate time vector centered at 0
        let dt = 1.0 / self.sample_rate;
        let total_time = n_samples as f64 * dt;
        let time_vec: Vec<f64> = (0..n_samples)
            .map(|i| (i as f64 * dt) - total_time / 2.0)
            .collect();

        for (scale_idx, &scale) in self.scales.iter().enumerate() {
            let wavelet = self.wavelet.generate(&time_vec, scale);

            // Convolve signal with wavelet at this scale
            for t in 0..n_samples {
                let mut conv_sum = 0.0;
                for (i, &sample) in signal.iter().enumerate().take(n_samples) {
                    let shift = (n_samples / 2 + t).wrapping_sub(i);
                    if shift < n_samples {
                        conv_sum += sample * wavelet[shift];
                    }
                }
                result[[scale_idx, t]] = conv_sum * dt;
            }
        }

        result
    }

    /// Computes the power (squared magnitude) of the CWT coefficients.
    pub fn power(&self, cwt: &Array2<f64>) -> Array2<f64> {
        cwt.mapv(|x| x * x)
    }

    /// Converts scales to frequencies in Hz.
    pub fn scales_to_frequencies(&self) -> Vec<f64> {
        let center_freq = match &self.wavelet {
            WaveletFamily::Morlet { omega0 } => omega0 / (2.0 * PI),
            WaveletFamily::MexicanHat => 1.0 / (2.0 * PI),
            _ => 1.0,
        };

        self.scales
            .iter()
            .map(|&scale| center_freq * self.sample_rate / scale)
            .collect()
    }
}

/// Result of discrete wavelet transform containing approximation and detail coefficients.
#[derive(Debug, Clone)]
pub struct DwtResult {
    /// Approximation coefficients at the finest level
    pub approximation: Vec<f64>,
    /// Detail coefficients at each level (from finest to coarsest)
    pub details: Vec<Vec<f64>>,
    /// Number of decomposition levels
    pub levels: usize,
}

impl DwtResult {
    /// Creates a new DWT result.
    pub fn new(approximation: Vec<f64>, details: Vec<Vec<f64>>, levels: usize) -> Self {
        Self {
            approximation,
            details,
            levels,
        }
    }

    /// Returns the total number of coefficients.
    pub fn total_coefficients(&self) -> usize {
        self.approximation.len() + self.details.iter().map(|d| d.len()).sum::<usize>()
    }
}

/// Discrete Wavelet Transform processor.
pub struct DiscreteWaveletTransform {
    wavelet: WaveletFamily,
    levels: usize,
    // Filter coefficients (low-pass and high-pass)
    h_low: Vec<f64>,
    h_high: Vec<f64>,
    g_low: Vec<f64>,
    g_high: Vec<f64>,
}

impl DiscreteWaveletTransform {
    /// The wavelet family whose filter coefficients this transform uses.
    pub fn wavelet(&self) -> WaveletFamily {
        self.wavelet
    }
    /// Creates a new DWT processor.
    ///
    /// # Arguments
    /// * `wavelet` - The wavelet family to use
    /// * `levels` - Number of decomposition levels
    pub fn new(wavelet: WaveletFamily, levels: usize) -> Self {
        let (h_low, h_high, g_low, g_high) = Self::get_filter_coefficients(&wavelet);

        Self {
            wavelet,
            levels,
            h_low,
            h_high,
            g_low,
            g_high,
        }
    }

    /// Gets filter coefficients for the wavelet family.
    fn get_filter_coefficients(wavelet: &WaveletFamily) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        match wavelet {
            WaveletFamily::Daubechies(2) => {
                // Daubechies-4 (db2) coefficients
                let h_low = vec![
                    0.482962913145,
                    0.836516303738,
                    0.224143868042,
                    -0.129409522551,
                ];
                let h_high = vec![
                    -0.129409522551,
                    -0.224143868042,
                    0.836516303738,
                    -0.482962913145,
                ];
                // Synthesis filters equal the analysis filters.
                //
                // `dwt_step` computes `a[i] = sum_j x[2i+j] * h[j]` and
                // `idwt_step` scatters `a[i] * g[j]` back to index `2i+j`, so
                // synthesis is the TRANSPOSE of analysis -- and for an
                // orthogonal wavelet the inverse is exactly the transpose.
                //
                // These were built from the opposite band and time-reversed:
                // `g_low` came from `h_high` and `g_high` from `h_low`, so
                // reconstruction convolved the approximation coefficients with a
                // highpass-derived filter and the detail coefficients with a
                // lowpass-derived one. Perfect reconstruction was impossible for
                // every wavelet family here.
                let g_low = h_low.clone();
                let g_high = h_high.clone();
                (h_low, h_high, g_low, g_high)
            }
            WaveletFamily::Daubechies(4) => {
                // Daubechies-8 (db4) coefficients
                let h_low = vec![
                    0.230377813309,
                    0.714846570553,
                    0.630880767930,
                    -0.027983769417,
                    -0.187034811719,
                    0.030841381836,
                    0.032883011667,
                    -0.010597401785,
                ];
                let h_high: Vec<f64> = h_low
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| if i % 2 == 0 { x } else { -x })
                    .rev()
                    .collect();
                // See the db2 arm: synthesis is the transpose of analysis.
                let g_low = h_low.clone();
                let g_high = h_high.clone();
                (h_low, h_high, g_low, g_high)
            }
            _ => {
                // Default to Haar wavelet (Daubechies-2)
                // Haar coefficients are exactly 1/sqrt(2).
                let c = std::f64::consts::FRAC_1_SQRT_2;
                let h_low = vec![c, c];
                let h_high = vec![c, -c];
                // See the db2 arm: synthesis is the transpose of analysis.
                let g_low = h_low.clone();
                let g_high = h_high.clone();
                (h_low, h_high, g_low, g_high)
            }
        }
    }

    /// Decomposes a signal using the discrete wavelet transform.
    pub fn decompose(&self, signal: &[f64]) -> DwtResult {
        let mut approximation = signal.to_vec();
        let mut details = Vec::new();

        for _ in 0..self.levels {
            let (approx, detail) = self.dwt_step(&approximation);
            details.push(detail);
            approximation = approx;
        }

        DwtResult::new(approximation, details, self.levels)
    }

    /// Single level of DWT decomposition.
    fn dwt_step(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        let out_len = n.div_ceil(2);

        let mut approximation = Vec::with_capacity(out_len);
        let mut detail = Vec::with_capacity(out_len);

        for i in 0..out_len {
            let mut sum_low = 0.0;
            let mut sum_high = 0.0;

            for (j, (&h_l, &h_h)) in self.h_low.iter().zip(self.h_high.iter()).enumerate() {
                let idx = (2 * i + j) % n;
                sum_low += signal[idx] * h_l;
                sum_high += signal[idx] * h_h;
            }

            approximation.push(sum_low);
            detail.push(sum_high);
        }

        (approximation, detail)
    }

    /// Reconstructs a signal from DWT coefficients.
    pub fn reconstruct(&self, result: &DwtResult) -> Vec<f64> {
        let mut approximation = result.approximation.clone();

        // Reconstruct from coarsest to finest level
        for detail in result.details.iter().rev() {
            approximation = self.idwt_step(&approximation, detail);
        }

        approximation
    }

    /// Single level of inverse DWT.
    fn idwt_step(&self, approximation: &[f64], detail: &[f64]) -> Vec<f64> {
        let out_len = approximation.len() * 2;
        let mut result = vec![0.0; out_len];

        for i in 0..approximation.len() {
            for (j, (&g_l, &g_h)) in self.g_low.iter().zip(self.g_high.iter()).enumerate() {
                let idx = (2 * i + j) % out_len;
                result[idx] += approximation[i] * g_l + detail[i] * g_h;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_wavelet_family_morlet() {
        let wavelet = WaveletFamily::Morlet { omega0: 6.0 };
        let t: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let values = wavelet.generate(&t, 1.0);

        // Check that wavelet has expected properties
        assert_eq!(values.len(), t.len());
        // Central value should be largest (for Morlet)
        let max_idx = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .map(|(idx, _)| idx)
            .unwrap();
        assert!((max_idx as i32 - 50).abs() < 5); // Near center
    }

    #[test]
    fn test_wavelet_family_mexican_hat() {
        let wavelet = WaveletFamily::MexicanHat;
        let t: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        let values = wavelet.generate(&t, 1.0);

        assert_eq!(values.len(), t.len());
        // Mexican hat should have positive center and negative sides
        assert!(values[50] > 0.0); // Center should be positive
    }

    #[test]
    fn test_cwt_transform() {
        let signal: Vec<f64> = (0..128).map(|i| (2.0 * PI * 5.0 * i as f64 / 128.0).sin()).collect();

        let wavelet = WaveletFamily::Morlet { omega0: 6.0 };
        let scales: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let cwt = ContinuousWaveletTransform::new(wavelet, scales, 128.0);

        let result = cwt.transform(&signal);

        assert_eq!(result.nrows(), 10);
        assert_eq!(result.ncols(), 128);
    }

    #[test]
    fn test_cwt_power() {
        let signal: Vec<f64> = (0..64).map(|i| (2.0 * PI * 5.0 * i as f64 / 64.0).sin()).collect();

        let wavelet = WaveletFamily::Morlet { omega0: 6.0 };
        let scales: Vec<f64> = vec![1.0, 2.0, 3.0];
        let cwt = ContinuousWaveletTransform::new(wavelet, scales, 64.0);

        let transform = cwt.transform(&signal);
        let power = cwt.power(&transform);

        assert_eq!(power.shape(), transform.shape());
        // Power should be non-negative
        assert!(power.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_dwt_decompose() {
        let signal: Vec<f64> = (0..128).map(|i| (2.0 * PI * i as f64 / 128.0).sin()).collect();

        let wavelet = WaveletFamily::Daubechies(2);
        let dwt = DiscreteWaveletTransform::new(wavelet, 3);

        let result = dwt.decompose(&signal);

        assert_eq!(result.levels, 3);
        assert_eq!(result.details.len(), 3);
        // Each level should approximately halve the length
        assert!(result.approximation.len() < signal.len() / 4);
    }

    #[test]
    fn test_dwt_reconstruction() {
        let signal: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 64.0).sin()).collect();

        let wavelet = WaveletFamily::Daubechies(2);
        let dwt = DiscreteWaveletTransform::new(wavelet, 2);

        let decomposed = dwt.decompose(&signal);
        let reconstructed = dwt.reconstruct(&decomposed);

        assert_eq!(reconstructed.len(), signal.len());

        // Reconstruction should be close to original
        for (orig, recon) in signal.iter().zip(reconstructed.iter()).take(60) {
            assert_relative_eq!(orig, recon, epsilon = 0.1);
        }
    }

    #[test]
    fn test_dwt_result_coefficients() {
        let approximation = vec![1.0, 2.0, 3.0];
        let details = vec![vec![0.5, 0.6], vec![0.1]];
        let result = DwtResult::new(approximation, details, 2);

        assert_eq!(result.total_coefficients(), 6); // 3 + 2 + 1
        assert_eq!(result.levels, 2);
    }

    #[test]
    fn test_scales_to_frequencies() {
        let wavelet = WaveletFamily::Morlet { omega0: 6.0 };
        let scales: Vec<f64> = vec![1.0, 2.0, 4.0];
        let sample_rate = 100.0;
        let cwt = ContinuousWaveletTransform::new(wavelet, scales, sample_rate);

        let frequencies = cwt.scales_to_frequencies();

        assert_eq!(frequencies.len(), 3);
        // Frequency should be inversely proportional to scale
        assert!(frequencies[0] > frequencies[1]);
        assert!(frequencies[1] > frequencies[2]);
    }
    /// Orthogonal wavelets must reconstruct essentially exactly.
    ///
    /// Regression: the synthesis filters were built from the opposite band, so
    /// reconstruction bore no fixed relationship to the input. The loose
    /// epsilon on the neighbouring test would not have caught a subtler version
    /// of the same error; an orthogonal transform round-trips to numerical
    /// precision, so that is what is asserted.
    #[test]
    fn test_dwt_perfect_reconstruction() {
        let signal: Vec<f64> = (0..64)
            .map(|i| {
                let t = i as f64 / 64.0;
                (2.0 * PI * 3.0 * t).sin() + 0.5 * (2.0 * PI * 11.0 * t).cos()
            })
            .collect();

        // Daubechies(1) falls through to the Haar default arm.
        for wavelet in [
            WaveletFamily::Daubechies(1),
            WaveletFamily::Daubechies(2),
            WaveletFamily::Daubechies(4),
        ] {
            for levels in 1..=2 {
                let dwt = DiscreteWaveletTransform::new(wavelet, levels);
                let reconstructed = dwt.reconstruct(&dwt.decompose(&signal));

                assert_eq!(reconstructed.len(), signal.len());
                for (i, (orig, recon)) in signal.iter().zip(reconstructed.iter()).enumerate() {
                    assert!(
                        (orig - recon).abs() < 1e-9,
                        "{wavelet:?} level {levels} sample {i}: {orig} -> {recon}"
                    );
                }
            }
        }
    }

}
