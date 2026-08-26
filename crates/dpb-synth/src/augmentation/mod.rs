//! Signal Augmentation Suite
//!
//! Provides signal augmentation techniques for synthetic data enhancement,
//! including noise injection, temporal transformations, and spectral modifications.

pub mod noise;
pub mod temporal;
pub mod spectral;
pub mod rand_helpers;

pub use rand_helpers::{random_f64, random_f64_range, random_usize_range, random_i32_range};

// Re-export the augmentations themselves.
//
// They were declared `pub` in their submodules but never surfaced here, so
// `use dpb_synth::augmentation::*` -- which is what this crate's own
// documentation tells you to write -- brought in only the RNG helpers.
pub use noise::{BaselineWander, GaussianNoise, MotionArtifact, PinkNoise, PowerlineNoise};
pub use spectral::{FrequencyMask, MagnitudeScale, TimeMask};
pub use temporal::{RandomDropout, Resample, TimeShift, TimeWarp, WindowCrop};

use rand::Rng;

/// Trait for signal augmentation operations
///
/// This trait is dyn-compatible (object-safe) by using `&mut dyn Rng`
/// instead of generic `impl Rng`.
pub trait SignalAugmentation: Send + Sync {
    /// Apply augmentation to a signal
    fn augment(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64>;

    /// Get the name of this augmentation
    fn name(&self) -> &str;
}

/// Pipeline for applying multiple augmentations with configurable probabilities
pub struct AugmentationPipeline {
    augmentations: Vec<(Box<dyn SignalAugmentation>, f64)>,  // (augmentation, probability)
}

impl AugmentationPipeline {
    /// Create a new empty augmentation pipeline
    pub fn new() -> Self {
        Self {
            augmentations: Vec::new(),
        }
    }

    /// Add an augmentation to the pipeline with a given probability
    ///
    /// # Arguments
    /// * `aug` - The augmentation to add
    /// * `probability` - Probability [0, 1] that this augmentation will be applied
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add(mut self, aug: impl SignalAugmentation + 'static, probability: f64) -> Self {
        assert!(probability >= 0.0 && probability <= 1.0, "Probability must be in [0, 1]");
        self.augmentations.push((Box::new(aug), probability));
        self
    }

    /// Apply all augmentations in the pipeline to the signal
    ///
    /// Each augmentation is applied with its configured probability.
    ///
    /// # Arguments
    /// * `signal` - Input signal to augment
    /// * `rng` - Random number generator for stochastic augmentations
    ///
    /// # Returns
    /// Augmented signal
    pub fn apply(&self, signal: &[f64], rng: &mut dyn Rng) -> Vec<f64> {
        let mut result = signal.to_vec();

        for (aug, prob) in &self.augmentations {
            // Generate a random f64 in [0, 1)
            let random_val = (rng.next_u64() as f64) / (u64::MAX as f64);
            if random_val < *prob {
                result = aug.augment(&result, rng);
            }
        }

        result
    }

    /// Get the number of augmentations in the pipeline
    pub fn len(&self) -> usize {
        self.augmentations.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.augmentations.is_empty()
    }
}

impl Default for AugmentationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    struct DummyAugmentation;

    impl SignalAugmentation for DummyAugmentation {
        fn augment(&self, signal: &[f64], _rng: &mut dyn Rng) -> Vec<f64> {
            signal.iter().map(|x| x * 2.0).collect()
        }

        fn name(&self) -> &str {
            "Dummy"
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = AugmentationPipeline::new();
        assert_eq!(pipeline.len(), 0);
        assert!(pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_add() {
        let pipeline = AugmentationPipeline::new()
            .add(DummyAugmentation, 1.0);
        assert_eq!(pipeline.len(), 1);
        assert!(!pipeline.is_empty());
    }

    #[test]
    fn test_pipeline_apply() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = vec![1.0, 2.0, 3.0];

        let pipeline = AugmentationPipeline::new()
            .add(DummyAugmentation, 1.0);  // Always apply

        let result = pipeline.apply(&signal, &mut rng);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_pipeline_probability() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let signal = vec![1.0, 2.0, 3.0];

        let pipeline = AugmentationPipeline::new()
            .add(DummyAugmentation, 0.0);  // Never apply

        let result = pipeline.apply(&signal, &mut rng);
        assert_eq!(result, signal);  // Should be unchanged
    }

    #[test]
    #[should_panic]
    fn test_invalid_probability() {
        AugmentationPipeline::new().add(DummyAugmentation, 1.5);
    }
}
