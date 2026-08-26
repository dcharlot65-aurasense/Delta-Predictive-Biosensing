//! Temporal signal augmentations
//!
//! Provides temporal transformations including time warping, shifting, cropping,
//! resampling, and dropout.

use super::{SignalAugmentation, random_f64, random_f64_range, random_usize_range, random_i32_range};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Time warping augmentation using smooth random curves
pub struct TimeWarp {
    /// Standard deviation of the warping magnitude
    pub sigma: f64,
    /// Number of warping control points (knots)
    pub knots: usize,
}

impl TimeWarp {
    /// Create a new time warp augmentation
    ///
    /// # Arguments
    /// * `sigma` - Standard deviation controlling warp magnitude
    /// * `knots` - Number of control points for warping (higher = more complex warping)
    pub fn new(sigma: f64, knots: usize) -> Self {
        assert!(knots >= 2, "Need at least 2 knots for time warping");
        Self { sigma, knots }
    }

    fn generate_warp_curve(&self, length: usize, rng: &mut dyn Rng) -> Vec<f64> {
        let normal = Normal::new(0.0, self.sigma).unwrap();

        // Generate random offsets at knot positions
        let mut knot_offsets = vec![0.0; self.knots];
        for offset in knot_offsets.iter_mut() {
            *offset = normal.sample(rng);
        }

        // Ensure endpoints are fixed (no warping at boundaries)
        knot_offsets[0] = 0.0;
        knot_offsets[self.knots - 1] = 0.0;

        // Interpolate between knots using linear interpolation
        let mut warp = Vec::with_capacity(length);
        for i in 0..length {
            let knot_pos = (i as f64 / length as f64) * (self.knots - 1) as f64;
            let knot_idx = knot_pos.floor() as usize;
            let knot_frac = knot_pos - knot_idx as f64;

            if knot_idx >= self.knots - 1 {
                warp.push(knot_offsets[self.knots - 1]);
            } else {
                let interp = knot_offsets[knot_idx] * (1.0 - knot_frac) +
                           knot_offsets[knot_idx + 1] * knot_frac;
                warp.push(interp);
            }
        }

        warp
    }
}

impl SignalAugmentation for TimeWarp {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        let warp = self.generate_warp_curve(signal.len(), rng);

        let mut result = Vec::with_capacity(signal.len());

        for (i, &warp_offset) in warp.iter().enumerate() {
            // Calculate warped time index
            let warped_idx = (i as f64 + warp_offset * signal.len() as f64 * 0.1)
                .max(0.0)
                .min((signal.len() - 1) as f64);

            // Linear interpolation
            let idx_floor = warped_idx.floor() as usize;
            let idx_ceil = (warped_idx.ceil() as usize).min(signal.len() - 1);
            let frac = warped_idx - idx_floor as f64;

            let value = signal[idx_floor] * (1.0 - frac) + signal[idx_ceil] * frac;
            result.push(value);
        }

        result
    }

    fn name(&self) -> &str {
        "TimeWarp"
    }
}

/// Time shift augmentation (circular shift)
pub struct TimeShift {
    /// Maximum shift in samples (will randomly shift in range [-max, +max])
    pub max_shift_samples: usize,
}

impl TimeShift {
    /// Create a new time shift augmentation
    pub fn new(max_shift_samples: usize) -> Self {
        Self { max_shift_samples }
    }
}

impl SignalAugmentation for TimeShift {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        if self.max_shift_samples == 0 {
            return signal.to_vec();
        }

        let shift = random_i32_range(rng, -(self.max_shift_samples as i32), self.max_shift_samples as i32);
        let n = signal.len() as i32;

        (0..signal.len())
            .map(|i| {
                let shifted_idx = ((i as i32 - shift).rem_euclid(n)) as usize;
                signal[shifted_idx]
            })
            .collect()
    }

    fn name(&self) -> &str {
        "TimeShift"
    }
}

/// Window crop augmentation (extract a random window)
pub struct WindowCrop {
    /// Ratio of signal to keep (0 < ratio <= 1)
    pub crop_ratio: f64,
}

impl WindowCrop {
    /// Create a new window crop augmentation
    ///
    /// # Arguments
    /// * `crop_ratio` - Fraction of signal to retain (e.g., 0.8 = keep 80%)
    pub fn new(crop_ratio: f64) -> Self {
        assert!(crop_ratio > 0.0 && crop_ratio <= 1.0, "Crop ratio must be in (0, 1]");
        Self { crop_ratio }
    }
}

impl SignalAugmentation for WindowCrop {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        if self.crop_ratio >= 1.0 {
            return signal.to_vec();
        }

        let crop_length = (signal.len() as f64 * self.crop_ratio).ceil() as usize;
        let max_start = signal.len() - crop_length;
        let start_idx = if max_start > 0 {
            random_usize_range(rng, 0, max_start)
        } else {
            0
        };

        signal[start_idx..start_idx + crop_length].to_vec()
    }

    fn name(&self) -> &str {
        "WindowCrop"
    }
}

/// Resample augmentation (change effective sampling rate)
pub struct Resample {
    /// Range of resampling rates (e.g., (0.8, 1.2) means 0.8x to 1.2x speed)
    pub rate_range: (f64, f64),
}

impl Resample {
    /// Create a new resample augmentation
    ///
    /// # Arguments
    /// * `rate_range` - Min and max resampling rates (1.0 = no change)
    pub fn new(rate_range: (f64, f64)) -> Self {
        assert!(rate_range.0 > 0.0 && rate_range.1 >= rate_range.0);
        Self { rate_range }
    }
}

impl SignalAugmentation for Resample {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        let rate = random_f64_range(rng, self.rate_range.0, self.rate_range.1);

        let new_length = (signal.len() as f64 * rate).round() as usize;
        let mut result = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_idx = (i as f64 / rate).min((signal.len() - 1) as f64);

            // Linear interpolation
            let idx_floor = src_idx.floor() as usize;
            let idx_ceil = (src_idx.ceil() as usize).min(signal.len() - 1);
            let frac = src_idx - idx_floor as f64;

            let value = signal[idx_floor] * (1.0 - frac) + signal[idx_ceil] * frac;
            result.push(value);
        }

        result
    }

    fn name(&self) -> &str {
        "Resample"
    }
}

/// Random dropout augmentation (randomly zero out samples)
pub struct RandomDropout {
    /// Probability of dropping each sample
    pub dropout_rate: f64,
}

impl RandomDropout {
    /// Create a new random dropout augmentation
    ///
    /// # Arguments
    /// * `dropout_rate` - Probability [0, 1] of dropping each sample
    pub fn new(dropout_rate: f64) -> Self {
        assert!(dropout_rate >= 0.0 && dropout_rate <= 1.0);
        Self { dropout_rate }
    }
}

impl SignalAugmentation for RandomDropout {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        signal.iter()
            .map(|&x| {
                if ((rng.next_u64() as f64) / (u64::MAX as f64)) < self.dropout_rate {
                    0.0
                } else {
                    x
                }
            })
            .collect()
    }

    fn name(&self) -> &str {
        "RandomDropout"
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
    fn test_time_warp() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = TimeWarp::new(0.2, 4);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
        // Should be similar but not identical
        let correlation = signal.iter().zip(augmented.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        assert!(correlation > 0.0);  // Should maintain general structure
    }

    #[test]
    fn test_time_shift() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = TimeShift::new(100);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
        assert_ne!(augmented, signal);  // Should be shifted
    }

    #[test]
    fn test_time_shift_zero() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = TimeShift::new(0);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented, signal);  // No shift
    }

    #[test]
    fn test_window_crop() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = WindowCrop::new(0.8);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), (signal.len() as f64 * 0.8).ceil() as usize);
    }

    #[test]
    fn test_window_crop_full() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = WindowCrop::new(1.0);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented, signal);
    }

    #[test]
    #[should_panic]
    fn test_invalid_crop_ratio() {
        WindowCrop::new(1.5);
    }

    #[test]
    fn test_resample() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = Resample::new((0.8, 1.2));

        let augmented = aug.augment(&signal, &mut rng);

        // Length should be in the expected range
        assert!(augmented.len() >= (signal.len() as f64 * 0.8) as usize);
        assert!(augmented.len() <= (signal.len() as f64 * 1.2) as usize);
    }

    #[test]
    fn test_random_dropout() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = RandomDropout::new(0.1);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());

        // Count zeros (dropped samples)
        let num_zeros = augmented.iter().filter(|&&x| x == 0.0).count();
        assert!(num_zeros > 0);  // Should have some dropouts
    }

    #[test]
    fn test_random_dropout_none() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = RandomDropout::new(0.0);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented, signal);  // No dropout
    }

    #[test]
    fn test_temporal_names() {
        assert_eq!(TimeWarp::new(0.2, 4).name(), "TimeWarp");
        assert_eq!(TimeShift::new(100).name(), "TimeShift");
        assert_eq!(WindowCrop::new(0.8).name(), "WindowCrop");
        assert_eq!(Resample::new((0.8, 1.2)).name(), "Resample");
        assert_eq!(RandomDropout::new(0.1).name(), "RandomDropout");
    }
}
