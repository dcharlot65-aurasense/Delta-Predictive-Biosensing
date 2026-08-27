//! Noise-based signal augmentations
//!
//! Provides various types of noise injection for signal augmentation,
//! including Gaussian, pink noise, baseline wander, powerline noise, and motion artifacts.

use super::{SignalAugmentation, random_f64_range, random_usize_range};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Gaussian (white) noise augmentation with configurable SNR
pub struct GaussianNoise {
    /// Signal-to-noise ratio in decibels
    pub snr_db: f64,
}

impl GaussianNoise {
    /// Create a new Gaussian noise augmentation
    pub fn new(snr_db: f64) -> Self {
        Self { snr_db }
    }
}

impl SignalAugmentation for GaussianNoise {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        // Calculate signal power
        let signal_power = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;

        // Convert SNR from dB to linear scale
        let snr_linear = 10_f64.powf(self.snr_db / 10.0);

        // Calculate noise power
        let noise_power = signal_power / snr_linear;
        let noise_std = noise_power.sqrt();

        // Generate Gaussian noise
        let normal = Normal::new(0.0, noise_std).unwrap();

        signal.iter()
            .map(|&x| x + normal.sample(rng))
            .collect()
    }

    fn name(&self) -> &str {
        "GaussianNoise"
    }
}

/// Pink (1/f) noise augmentation with configurable SNR
pub struct PinkNoise {
    /// Signal-to-noise ratio in decibels
    pub snr_db: f64,
}

impl PinkNoise {
    /// Create a new pink noise augmentation
    pub fn new(snr_db: f64) -> Self {
        Self { snr_db }
    }

    /// Generate pink noise using the Voss-McCartney algorithm
    fn generate_pink_noise(&self, length: usize, rng: &mut dyn Rng) -> Vec<f64> {
        const NUM_GENERATORS: usize = 16;
        let mut generators = [0.0; NUM_GENERATORS];
        let mut counter = 0u32;
        let mut pink = Vec::with_capacity(length);

        for _ in 0..length {
            // Update generators based on counter bits
            let mut sum = 0.0;
            for (i_off, i_slot) in generators[0..NUM_GENERATORS].iter_mut().enumerate() {
                let i = 0 + i_off;
                if counter & (1 << i) != 0 {
                    *i_slot = ((rng.next_u64() as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
                }
                sum += *i_slot;
            }

            pink.push(sum / NUM_GENERATORS as f64);
            counter = counter.wrapping_add(1);
        }

        pink
    }
}

impl SignalAugmentation for PinkNoise {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        // Generate pink noise
        let mut pink_noise = self.generate_pink_noise(signal.len(), rng);

        // Calculate signal and noise power
        let signal_power = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
        let noise_power = pink_noise.iter().map(|x| x * x).sum::<f64>() / pink_noise.len() as f64;

        // Convert SNR from dB to linear scale
        let snr_linear = 10_f64.powf(self.snr_db / 10.0);

        // Scale noise to achieve desired SNR
        let scale = (signal_power / (snr_linear * noise_power)).sqrt();
        pink_noise.iter_mut().for_each(|x| *x *= scale);

        // Add noise to signal
        signal.iter()
            .zip(pink_noise.iter())
            .map(|(s, n)| s + n)
            .collect()
    }

    fn name(&self) -> &str {
        "PinkNoise"
    }
}

/// Baseline wander augmentation (low-frequency drift)
pub struct BaselineWander {
    /// Frequency of baseline wander in Hz
    pub frequency_hz: f64,
    /// Amplitude of baseline wander (relative to signal)
    pub amplitude: f64,
}

impl BaselineWander {
    /// Create a new baseline wander augmentation
    ///
    /// # Arguments
    /// * `frequency_hz` - Frequency of the wandering baseline (typically 0.1-1.0 Hz)
    /// * `amplitude` - Amplitude relative to signal range
    pub fn new(frequency_hz: f64, amplitude: f64) -> Self {
        Self { frequency_hz, amplitude }
    }
}

impl SignalAugmentation for BaselineWander {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        // Assume a typical sampling rate of 250 Hz if not specified
        let fs = 250.0;
        let phase = ((rng.next_u64() as f64) / (u64::MAX as f64)) * 2.0 * PI;

        signal.iter().enumerate()
            .map(|(i, &x)| {
                let t = i as f64 / fs;
                let wander = self.amplitude * (2.0 * PI * self.frequency_hz * t + phase).sin();
                x + wander
            })
            .collect()
    }

    fn name(&self) -> &str {
        "BaselineWander"
    }
}

/// Powerline noise augmentation (50 or 60 Hz interference)
pub struct PowerlineNoise {
    /// Frequency in Hz (typically 50 or 60)
    pub frequency_hz: f64,
    /// Amplitude of the interference
    pub amplitude: f64,
}

impl PowerlineNoise {
    /// Create a new powerline noise augmentation
    ///
    /// # Arguments
    /// * `frequency_hz` - Powerline frequency (50 or 60 Hz)
    /// * `amplitude` - Amplitude of the interference
    pub fn new(frequency_hz: f64, amplitude: f64) -> Self {
        assert!(
            (frequency_hz - 50.0).abs() < 1.0 || (frequency_hz - 60.0).abs() < 1.0,
            "Powerline frequency should be near 50 or 60 Hz"
        );
        Self { frequency_hz, amplitude }
    }
}

impl SignalAugmentation for PowerlineNoise {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        // Assume a typical sampling rate of 250 Hz if not specified
        let fs = 250.0;
        let phase = ((rng.next_u64() as f64) / (u64::MAX as f64)) * 2.0 * PI;

        // Add harmonics for more realistic powerline noise
        signal.iter().enumerate()
            .map(|(i, &x)| {
                let t = i as f64 / fs;
                let fundamental = self.amplitude * (2.0 * PI * self.frequency_hz * t + phase).sin();
                let harmonic2 = (self.amplitude * 0.3) * (2.0 * PI * 2.0 * self.frequency_hz * t + phase).sin();
                let harmonic3 = (self.amplitude * 0.1) * (2.0 * PI * 3.0 * self.frequency_hz * t + phase).sin();
                x + fundamental + harmonic2 + harmonic3
            })
            .collect()
    }

    fn name(&self) -> &str {
        "PowerlineNoise"
    }
}

/// Motion artifact augmentation (sudden signal distortions)
pub struct MotionArtifact {
    /// Probability of artifact occurrence per second
    pub probability: f64,
    /// Duration range in seconds (min, max)
    pub duration_range: (f64, f64),
}

impl MotionArtifact {
    /// Create a new motion artifact augmentation
    ///
    /// # Arguments
    /// * `probability` - Probability of artifact per second
    /// * `duration_range` - Min and max duration in seconds
    pub fn new(probability: f64, duration_range: (f64, f64)) -> Self {
        assert!(duration_range.0 > 0.0 && duration_range.1 >= duration_range.0);
        Self { probability, duration_range }
    }

    fn generate_artifact(&self, length: usize, rng: &mut dyn Rng) -> Vec<f64> {
        let mut artifact = vec![0.0; length];

        // Create a transient with exponential decay
        let peak = ((rng.next_u64() as f64) / (u64::MAX as f64)) * 2.0 - 1.0;  // Random peak amplitude
        let decay_rate = random_f64_range(rng, 5.0, 20.0);

        for (i_off, i_slot) in artifact[0..length].iter_mut().enumerate() {
            let i = 0 + i_off;
            let t = i as f64 / length as f64;
            *i_slot = peak * (-decay_rate * t).exp();
        }

        artifact
    }
}

impl SignalAugmentation for MotionArtifact {
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        let mut result = signal.to_vec();
        let fs = 250.0;  // Assume 250 Hz sampling rate
        let duration_sec = signal.len() as f64 / fs;

        // Determine number of artifacts to add
        let expected_artifacts = self.probability * duration_sec;
        let num_artifacts = if expected_artifacts >= 1.0 {
            random_usize_range(rng, 0, expected_artifacts.ceil() as usize)
        } else if ((rng.next_u64() as f64) / (u64::MAX as f64)) < expected_artifacts {
            1
        } else {
            0
        };

        for _ in 0..num_artifacts {
            // Random artifact location
            let start_idx = random_usize_range(rng, 0, signal.len().saturating_sub(1));

            // Random artifact duration
            let duration_sec = random_f64_range(rng, self.duration_range.0, self.duration_range.1);
            let duration_samples = (duration_sec * fs) as usize;
            let end_idx = (start_idx + duration_samples).min(signal.len());

            // Generate and add artifact
            let artifact_length = end_idx - start_idx;
            if artifact_length > 0 {
                let artifact = self.generate_artifact(artifact_length, rng);
                for (i, &art) in artifact.iter().enumerate() {
                    result[start_idx + i] += art;
                }
            }
        }

        result
    }

    fn name(&self) -> &str {
        "MotionArtifact"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn create_test_signal() -> Vec<f64> {
        (0..1000).map(|i| (2.0 * PI * i as f64 / 50.0).sin()).collect()
    }

    #[test]
    fn test_gaussian_noise() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = GaussianNoise::new(10.0);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
        assert_ne!(augmented, signal);  // Should be different due to noise
    }

    #[test]
    fn test_pink_noise() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = PinkNoise::new(15.0);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
        assert_ne!(augmented, signal);
    }

    #[test]
    fn test_baseline_wander() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = BaselineWander::new(0.5, 0.1);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
        // Check that values have changed
        let changed = signal.iter().zip(augmented.iter())
            .any(|(s, a)| (s - a).abs() > 1e-10);
        assert!(changed);
    }

    #[test]
    fn test_powerline_noise() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = PowerlineNoise::new(60.0, 0.1);

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
    }

    #[test]
    #[should_panic]
    fn test_invalid_powerline_frequency() {
        PowerlineNoise::new(100.0, 0.1);
    }

    #[test]
    fn test_motion_artifact() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = create_test_signal();
        let aug = MotionArtifact::new(0.5, (0.1, 0.3));

        let augmented = aug.augment(&signal, &mut rng);

        assert_eq!(augmented.len(), signal.len());
    }

    #[test]
    fn test_noise_names() {
        assert_eq!(GaussianNoise::new(10.0).name(), "GaussianNoise");
        assert_eq!(PinkNoise::new(10.0).name(), "PinkNoise");
        assert_eq!(BaselineWander::new(0.5, 0.1).name(), "BaselineWander");
        assert_eq!(PowerlineNoise::new(60.0, 0.1).name(), "PowerlineNoise");
        assert_eq!(MotionArtifact::new(0.5, (0.1, 0.3)).name(), "MotionArtifact");
    }
}
