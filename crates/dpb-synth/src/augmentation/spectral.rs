//! Spectral signal augmentations
//!
//! Provides frequency-domain transformations including magnitude scaling,
//! frequency masking, and time masking.

use super::{SignalAugmentation, random_f64_range, random_usize_range};
use rand::Rng;

/// Magnitude scaling augmentation (amplitude modulation)
pub struct MagnitudeScale {
    /// Range of scaling factors (min, max)
    pub range: (f64, f64),
}

impl MagnitudeScale {
    /// Create a new magnitude scale augmentation
    ///
    /// # Arguments
    /// * `range` - Min and max scaling factors (e.g., (0.8, 1.2))
    pub fn new(range: (f64, f64)) -> Self {
        assert!(range.0 > 0.0 && range.1 >= range.0);
        Self { range }
    }
}

impl SignalAugmentation for MagnitudeScale {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        let scale = random_f64_range(rng, self.range.0, self.range.1);
        signal.iter().map(|&x| x * scale).collect()
    }

    fn name(&self) -> &str {
        "MagnitudeScale"
    }
}

/// Frequency masking augmentation (zero out frequency bands)
pub struct FrequencyMask {
    /// Maximum number of frequency masks to apply
    pub max_masks: usize,
    /// Maximum width of each mask in Hz
    pub max_width_hz: f64,
}

impl FrequencyMask {
    /// Create a new frequency mask augmentation
    ///
    /// # Arguments
    /// * `max_masks` - Maximum number of frequency bands to mask
    /// * `max_width_hz` - Maximum width of each masked band
    pub fn new(max_masks: usize, max_width_hz: f64) -> Self {
        assert!(max_width_hz > 0.0);
        Self { max_masks, max_width_hz }
    }

    fn apply_fft_mask(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        use rustfft::{FftPlanner, num_complex::Complex};

        let n = signal.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        // Convert to complex
        let mut buffer: Vec<Complex<f64>> = signal.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Forward FFT
        fft.process(&mut buffer);

        // Apply frequency masks
        let fs = 250.0;  // Assumed sampling rate
        let freq_resolution = fs / n as f64;
        let num_masks = random_usize_range(rng, 1, self.max_masks);

        for _ in 0..num_masks {
            // Random frequency band to mask
            let center_freq = random_f64_range(rng, 0.0, fs / 2.0);
            let width = random_f64_range(rng, 0.0, self.max_width_hz);

            let freq_start = (center_freq - width / 2.0).max(0.0);
            let freq_end = (center_freq + width / 2.0).min(fs / 2.0);

            // Convert to bin indices
            let bin_start = (freq_start / freq_resolution).floor() as usize;
            let bin_end = ((freq_end / freq_resolution).ceil() as usize).min(n / 2);

            // Zero out the frequency band (and its mirror for real signals)
            for i in bin_start..=bin_end {
                buffer[i] = Complex::new(0.0, 0.0);
                if i > 0 && i < n {
                    buffer[n - i] = Complex::new(0.0, 0.0);
                }
            }
        }

        // Inverse FFT
        ifft.process(&mut buffer);

        // Extract real part and normalize
        buffer.iter()
            .map(|c| c.re / n as f64)
            .collect()
    }
}

impl SignalAugmentation for FrequencyMask {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        if self.max_masks == 0 {
            return signal.to_vec();
        }

        self.apply_fft_mask(signal, rng)
    }

    fn name(&self) -> &str {
        "FrequencyMask"
    }
}

/// Time masking augmentation (zero out time windows)
pub struct TimeMask {
    /// Maximum number of time masks to apply
    pub max_masks: usize,
    /// Maximum width of each mask in samples
    pub max_width_samples: usize,
}

impl TimeMask {
    /// Create a new time mask augmentation
    ///
    /// # Arguments
    /// * `max_masks` - Maximum number of time windows to mask
    /// * `max_width_samples` - Maximum width of each masked window
    pub fn new(max_masks: usize, max_width_samples: usize) -> Self {
        Self { max_masks, max_width_samples }
    }
}

impl SignalAugmentation for TimeMask {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        if self.max_masks == 0 || self.max_width_samples == 0 {
            return signal.to_vec();
        }

        let mut result = signal.to_vec();
        let num_masks = random_usize_range(rng, 1, self.max_masks);

        for _ in 0..num_masks {
            // Random mask width
            let width = random_usize_range(rng, 1, self.max_width_samples.min(signal.len()));

            // Random mask location
            let max_start = if signal.len() > width {
                signal.len() - width
            } else {
                0
            };

            let start = if max_start > 0 {
                random_usize_range(rng, 0, max_start)
            } else {
                0
            };

            // Apply mask
            result[start..(start + width).min(signal.len())].fill(0.0);
        }

        result
    }

    fn name(&self) -> &str {
        "TimeMask"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::f64::consts::PI;

    fn create_test_signal() -> Vec<f64> {
        (0..1000).map(|i| (2.0 * PI * i as f64 / 50.0).sin()).collect()
    }

    #[test]
    fn test_magnitude_scale() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = MagnitudeScale::new((0.8, 1.2));

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());

        // Check that magnitude has changed
        let orig_power: f64 = signal.iter().map(|x| x * x).sum();
        let aug_power: f64 = augmented.iter().map(|x| x * x).sum();

        assert!((aug_power / orig_power - 1.0).abs() > 0.01);  // Should differ by > 1%
    }

    #[test]
    fn test_magnitude_scale_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = MagnitudeScale::new((2.0, 2.0));  // Exact scaling

        let augmented = aug.augment(&signal, &mut rng);

        // Should be exactly doubled
        for (orig, aug) in signal.iter().zip(augmented.iter()) {
            assert!((aug - orig * 2.0).abs() < 1e-10);
        }
    }

    #[test]
    #[should_panic]
    fn test_invalid_magnitude_scale() {
        MagnitudeScale::new((1.2, 0.8));
    }

    #[test]
    fn test_frequency_mask() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = FrequencyMask::new(2, 10.0);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
        // Energy should be reduced due to masking
        let orig_energy: f64 = signal.iter().map(|x| x * x).sum();
        let aug_energy: f64 = augmented.iter().map(|x| x * x).sum();

        assert!(aug_energy < orig_energy);
    }

    #[test]
    fn test_frequency_mask_none() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = FrequencyMask::new(0, 10.0);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented, signal);  // No masking
    }

    #[test]
    fn test_time_mask() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = TimeMask::new(2, 50);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());

        // Should have some zeros from masking
        let num_zeros = augmented.iter().filter(|&&x| x == 0.0).count();
        assert!(num_zeros > 0);
    }

    #[test]
    fn test_time_mask_none() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = TimeMask::new(0, 50);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented, signal);  // No masking
    }

    #[test]
    fn test_time_mask_zero_width() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = TimeMask::new(2, 0);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented, signal);  // No masking
    }

    #[test]
    fn test_spectral_names() {
        assert_eq!(MagnitudeScale::new((0.8, 1.2)).name(), "MagnitudeScale");
        assert_eq!(FrequencyMask::new(2, 10.0).name(), "FrequencyMask");
        assert_eq!(TimeMask::new(2, 50).name(), "TimeMask");
    }
}
