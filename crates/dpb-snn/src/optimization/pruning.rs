//! Network Pruning for Spiking Neural Networks
//!
//! This module implements various pruning strategies to reduce network size
//! and improve efficiency while maintaining performance.

use ndarray::Array2;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

/// Pruning strategy determines which weights to prune
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningStrategy {
    /// Prune weights below absolute magnitude threshold
    MagnitudeBased { threshold: f64 },

    /// Prune weights with low gradient magnitudes
    GradientBased { threshold: f64 },

    /// Randomly prune a fraction of weights
    Random { ratio: f64 },

    /// Prune lowest magnitude weights to achieve target sparsity
    TopK { target_sparsity: f64 },

    /// Structured pruning (entire neurons/channels)
    Structured { ratio: f64 },

    /// Movement pruning (based on weight movement during training)
    Movement { threshold: f64 },
}

impl PruningStrategy {
    /// Apply pruning strategy to weights
    pub fn prune(&self, weights: &mut Array2<f64>, step: usize) -> PruningStats {
        match self {
            Self::MagnitudeBased { threshold } => self.magnitude_prune(weights, *threshold),
            Self::GradientBased { threshold } => {
                // For gradient-based, we need gradients which aren't tracked here
                // Fall back to magnitude for now
                self.magnitude_prune(weights, *threshold)
            }
            Self::Random { ratio } => self.random_prune(weights, *ratio, step as u64),
            Self::TopK { target_sparsity } => self.topk_prune(weights, *target_sparsity),
            Self::Structured { ratio } => self.structured_prune(weights, *ratio, step as u64),
            Self::Movement { threshold } => self.magnitude_prune(weights, *threshold),
        }
    }

    fn magnitude_prune(&self, weights: &mut Array2<f64>, threshold: f64) -> PruningStats {
        let mut pruned = 0;

        for w in weights.iter_mut() {
            if w.abs() < threshold {
                *w = 0.0;
                pruned += 1;
            }
        }

        PruningStats::measure(weights, pruned)
    }

    fn random_prune(&self, weights: &mut Array2<f64>, ratio: f64, seed: u64) -> PruningStats {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut pruned = 0;

        for w in weights.iter_mut() {
            if *w != 0.0 && rng.random::<f64>() < ratio {
                *w = 0.0;
                pruned += 1;
            }
        }

        PruningStats::measure(weights, pruned)
    }

    fn topk_prune(&self, weights: &mut Array2<f64>, target_sparsity: f64) -> PruningStats {
        let total = weights.len();
        let target_pruned = (total as f64 * target_sparsity) as usize;

        // Collect (index, magnitude) pairs
        let mut indexed_weights: Vec<(usize, f64)> = weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w.abs()))
            .collect();

        // Sort by magnitude (ascending)
        indexed_weights.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Prune lowest magnitude weights
        let mut pruned = 0;
        let shape = weights.shape();
        let ncols = shape[1];

        for (idx, _) in indexed_weights.iter().take(target_pruned) {
            let row = idx / ncols;
            let col = idx % ncols;
            if weights[[row, col]] != 0.0 {
                weights[[row, col]] = 0.0;
                pruned += 1;
            }
        }

        PruningStats::measure(weights, pruned)
    }

    /// Prunes whole rows (neurons), ranked by L1 magnitude.
    ///
    /// Deterministic: rows are selected by magnitude, so `_seed` is unused. It
    /// is kept in the signature to match the other strategies' shape.
    fn structured_prune(&self, weights: &mut Array2<f64>, ratio: f64, _seed: u64) -> PruningStats {
        let n_rows = weights.nrows();
        let n_cols = weights.ncols();

        // Prune entire rows (neurons).
        //
        // Round rather than truncate. `as usize` truncates, so a 3-row layer at
        // ratio 0.33 gave `0.99 -> 0` and pruning silently did nothing at all --
        // a no-op the caller could only detect by inspecting the returned stats.
        // Rounding honours the request at the granularity a row-wise method
        // actually has, and still yields zero rows when the ratio genuinely
        // rounds to none.
        let n_prune_rows = (n_rows as f64 * ratio).round() as usize;
        let mut pruned = 0;

        // Calculate row magnitudes
        let mut row_magnitudes: Vec<(usize, f64)> = (0..n_rows)
            .map(|i| {
                let mag = weights.row(i).iter().map(|&w| w.abs()).sum::<f64>();
                (i, mag)
            })
            .collect();

        // Sort by magnitude (ascending)
        row_magnitudes.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Prune lowest magnitude rows
        for (row_idx, _) in row_magnitudes.iter().take(n_prune_rows) {
            for col in 0..n_cols {
                if weights[[*row_idx, col]] != 0.0 {
                    weights[[*row_idx, col]] = 0.0;
                    pruned += 1;
                }
            }
        }

        PruningStats::measure(weights, pruned)
    }
}

/// Pruning schedule determines when and how much to prune
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningSchedule {
    /// Prune once at a specific step
    OneShot,

    /// Iteratively prune over multiple steps
    Iterative { steps: usize, final_sparsity: f64 },

    /// Gradually increase sparsity over training
    Gradual {
        start_step: usize,
        end_step: usize,
        initial_sparsity: f64,
        final_sparsity: f64,
    },

    /// Exponential schedule
    Exponential {
        start_step: usize,
        frequency: usize,
        final_sparsity: f64,
    },

    /// Polynomial schedule
    Polynomial {
        start_step: usize,
        end_step: usize,
        initial_sparsity: f64,
        final_sparsity: f64,
        power: f64,
    },
}

impl PruningSchedule {
    /// Check if pruning should occur at this step
    pub fn should_prune(&self, step: usize) -> bool {
        match self {
            Self::OneShot => step == 0,
            Self::Iterative { steps, .. } => step < *steps,
            Self::Gradual {
                start_step,
                end_step,
                ..
            } => step >= *start_step && step <= *end_step,
            Self::Exponential {
                start_step,
                frequency,
                ..
            } => step >= *start_step && (step - start_step).is_multiple_of(*frequency),
            Self::Polynomial {
                start_step,
                end_step,
                ..
            } => step >= *start_step && step <= *end_step,
        }
    }

    /// Get target sparsity at this step
    pub fn target_sparsity(&self, step: usize) -> f64 {
        match self {
            Self::OneShot => 0.0,
            Self::Iterative {
                steps,
                final_sparsity,
            } => {
                if step >= *steps {
                    *final_sparsity
                } else {
                    (step as f64 / *steps as f64) * final_sparsity
                }
            }
            Self::Gradual {
                start_step,
                end_step,
                initial_sparsity,
                final_sparsity,
            } => {
                if step < *start_step {
                    *initial_sparsity
                } else if step > *end_step {
                    *final_sparsity
                } else {
                    let progress = (step - start_step) as f64 / (end_step - start_step) as f64;
                    initial_sparsity + progress * (final_sparsity - initial_sparsity)
                }
            }
            Self::Exponential {
                start_step,
                frequency,
                final_sparsity,
            } => {
                if step < *start_step {
                    0.0
                } else {
                    let n_prunes = ((step - start_step) / frequency) as f64;
                    final_sparsity * (1.0 - (-n_prunes / 10.0).exp())
                }
            }
            Self::Polynomial {
                start_step,
                end_step,
                initial_sparsity,
                final_sparsity,
                power,
            } => {
                if step < *start_step {
                    *initial_sparsity
                } else if step > *end_step {
                    *final_sparsity
                } else {
                    let progress = (step - start_step) as f64 / (end_step - start_step) as f64;
                    let progress_powered = progress.powf(*power);
                    initial_sparsity + progress_powered * (final_sparsity - initial_sparsity)
                }
            }
        }
    }
}

/// Network pruner combining strategy and schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPruner {
    /// Pruning strategy
    pub strategy: PruningStrategy,
    /// Pruning schedule
    pub schedule: PruningSchedule,
    /// Current training step
    current_step: usize,
}

impl NetworkPruner {
    /// Create a new network pruner
    pub fn new(strategy: PruningStrategy, schedule: PruningSchedule) -> Self {
        Self {
            strategy,
            schedule,
            current_step: 0,
        }
    }

    /// Prune weights if the schedule calls for it at this step.
    ///
    /// The schedule sets both *when* to prune and *how far*: a gradual or
    /// iterative schedule ramps its target sparsity over training, which is the
    /// point of using one. That ramp used to be ignored -- `should_prune` was
    /// consulted and `target_sparsity` never was, so the strategy pruned to its
    /// own fixed figure every time and a schedule built by `magnitude_gradual`
    /// jumped to its final sparsity on the first pruning step.
    ///
    /// A `TopK` strategy prunes to the schedule's target for this step. The
    /// threshold- and ratio-based strategies have no sparsity target to
    /// override, so they are left as configured; `OneShot`, which reports a
    /// target of 0.0, likewise leaves the strategy alone.
    pub fn prune_weights(&mut self, weights: &mut Array2<f64>, step: usize) -> PruningStats {
        self.current_step = step;

        if self.schedule.should_prune(step) {
            let scheduled = self.schedule.target_sparsity(step);
            match self.strategy {
                PruningStrategy::TopK { .. } if scheduled > 0.0 => PruningStrategy::TopK {
                    target_sparsity: scheduled,
                }
                .prune(weights, step),
                _ => self.strategy.prune(weights, step),
            }
        } else {
            // No pruning this step
            // Nothing pruned this step, but the tensor's existing sparsity is
            // still what should be reported.
            PruningStats::measure(weights, 0)
        }
    }

    /// Compute current sparsity of weights
    pub fn compute_sparsity(&self, weights: &Array2<f64>) -> f64 {
        let total = weights.len();
        let zero_count = weights.iter().filter(|&&w| w == 0.0).count();
        zero_count as f64 / total as f64
    }

    /// Create magnitude-based pruner with gradual schedule
    pub fn magnitude_gradual(
        threshold: f64,
        start_step: usize,
        end_step: usize,
        final_sparsity: f64,
    ) -> Self {
        Self::new(
            PruningStrategy::MagnitudeBased { threshold },
            PruningSchedule::Gradual {
                start_step,
                end_step,
                initial_sparsity: 0.0,
                final_sparsity,
            },
        )
    }

    /// Create Top-K pruner with iterative schedule
    pub fn topk_iterative(final_sparsity: f64, steps: usize) -> Self {
        Self::new(
            PruningStrategy::TopK {
                target_sparsity: final_sparsity,
            },
            PruningSchedule::Iterative {
                steps,
                final_sparsity,
            },
        )
    }

    /// Create structured pruner with one-shot schedule
    pub fn structured_oneshot(ratio: f64) -> Self {
        Self::new(
            PruningStrategy::Structured { ratio },
            PruningSchedule::OneShot,
        )
    }
}

/// Statistics from pruning operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningStats {
    /// Total number of weights
    pub total_weights: usize,
    /// Number of pruned (zero) weights in the tensor as a whole.
    ///
    /// This counts every zero, not only the ones this call created. The
    /// strategies used to report their own delta here, so after the first
    /// pruning step the figure -- and `sparsity` with it -- described how much
    /// had just changed rather than how much was pruned, and
    /// `remaining_weights` and `compression_ratio` were wrong with them.
    pub pruned_weights: usize,
    /// Weights this call zeroed that were not already zero.
    pub newly_pruned: usize,
    /// Sparsity ratio (0 = dense, 1 = all pruned), over the whole tensor.
    pub sparsity: f64,
}

impl PruningStats {
    /// Measures a tensor after pruning, recording how many zeros this call added.
    fn measure(weights: &Array2<f64>, newly_pruned: usize) -> Self {
        let total_weights = weights.len();
        let pruned_weights = weights.iter().filter(|&&w| w == 0.0).count();
        Self {
            total_weights,
            pruned_weights,
            newly_pruned,
            sparsity: if total_weights == 0 {
                0.0
            } else {
                pruned_weights as f64 / total_weights as f64
            },
        }
    }
}

impl PruningStats {
    /// Get number of remaining (non-zero) weights
    pub fn remaining_weights(&self) -> usize {
        self.total_weights - self.pruned_weights
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.total_weights == 0 {
            1.0
        } else {
            self.remaining_weights() as f64 / self.total_weights as f64
        }
    }

    /// Get percentage pruned
    pub fn pruned_percentage(&self) -> f64 {
        self.sparsity * 100.0
    }

    /// Combine stats from multiple layers
    pub fn merge(&self, other: &PruningStats) -> PruningStats {
        let total_weights = self.total_weights + other.total_weights;
        let pruned_weights = self.pruned_weights + other.pruned_weights;
        PruningStats {
            total_weights,
            pruned_weights,
            newly_pruned: self.newly_pruned + other.newly_pruned,
            sparsity: if total_weights == 0 {
                0.0
            } else {
                pruned_weights as f64 / total_weights as f64
            },
        }
    }
}

/// Pruning mask to track which weights are pruned
#[derive(Debug, Clone)]
pub struct PruningMask {
    /// Binary mask (1 = keep, 0 = prune)
    pub mask: Array2<bool>,
}

impl PruningMask {
    /// Create mask from weights
    pub fn from_weights(weights: &Array2<f64>) -> Self {
        let mask = weights.mapv(|w| w != 0.0);
        Self { mask }
    }

    /// Apply mask to weights
    pub fn apply(&self, weights: &mut Array2<f64>) {
        for ((i, j), &keep) in self.mask.indexed_iter() {
            if !keep {
                weights[[i, j]] = 0.0;
            }
        }
    }

    /// Get sparsity
    pub fn sparsity(&self) -> f64 {
        let total = self.mask.len();
        let zeros = self.mask.iter().filter(|&&x| !x).count();
        zeros as f64 / total as f64
    }

    /// Merge two masks (AND operation)
    pub fn intersect(&self, other: &PruningMask) -> PruningMask {
        let mask = &self.mask & &other.mask;
        PruningMask { mask }
    }

    /// Union of two masks (OR operation)
    pub fn union(&self, other: &PruningMask) -> PruningMask {
        let mask = &self.mask | &other.mask;
        PruningMask { mask }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_magnitude_pruning() {
        let mut weights = Array2::from_shape_vec(
            (3, 3),
            vec![0.5, 0.01, 0.9, 0.02, 0.7, 0.03, 0.8, 0.04, 0.6],
        )
        .unwrap();

        let strategy = PruningStrategy::MagnitudeBased { threshold: 0.05 };
        let stats = strategy.prune(&mut weights, 0);

        // Should prune weights < 0.05: 0.01, 0.02, 0.03, 0.04 = 4 weights
        assert_eq!(stats.pruned_weights, 4);
        assert_eq!(stats.total_weights, 9);
        assert_abs_diff_eq!(stats.sparsity, 4.0 / 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_random_pruning() {
        let mut weights = Array2::ones((10, 10));
        let strategy = PruningStrategy::Random { ratio: 0.5 };
        let stats = strategy.prune(&mut weights, 0);

        // Should prune approximately 50%
        assert!(stats.sparsity > 0.3 && stats.sparsity < 0.7);
    }

    #[test]
    fn test_topk_pruning() {
        let mut weights =
            Array2::from_shape_vec((2, 3), vec![0.9, 0.1, 0.5, 0.2, 0.8, 0.3]).unwrap();

        let strategy = PruningStrategy::TopK {
            target_sparsity: 0.5,
        };
        let stats = strategy.prune(&mut weights, 0);

        // Should prune 3 out of 6 weights (50%)
        assert_eq!(stats.pruned_weights, 3);

        // Check that smallest magnitude weights are pruned
        // Sorted by magnitude: 0.1, 0.2, 0.3, 0.5, 0.8, 0.9
        // First 3 should be pruned
        assert_eq!(weights[[0, 1]], 0.0); // 0.1
        assert_eq!(weights[[1, 0]], 0.0); // 0.2
        assert_eq!(weights[[1, 2]], 0.0); // 0.3
    }

    #[test]
    fn test_structured_pruning() {
        let mut weights = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.1, 0.1, 0.1, 0.1, // Low magnitude row
                0.9, 0.9, 0.9, 0.9, // High magnitude row
                0.5, 0.5, 0.5, 0.5, // Medium magnitude row
            ],
        )
        .unwrap();

        let strategy = PruningStrategy::Structured { ratio: 0.33 };
        let stats = strategy.prune(&mut weights, 42);

        // Should prune entire row(s)
        assert!(stats.pruned_weights >= 4); // At least one row
    }

    #[test]
    fn test_oneshot_schedule() {
        let schedule = PruningSchedule::OneShot;

        assert!(schedule.should_prune(0));
        assert!(!schedule.should_prune(1));
        assert!(!schedule.should_prune(100));
    }

    #[test]
    fn test_iterative_schedule() {
        let schedule = PruningSchedule::Iterative {
            steps: 5,
            final_sparsity: 0.9,
        };

        assert!(schedule.should_prune(0));
        assert!(schedule.should_prune(3));
        assert!(!schedule.should_prune(5));

        // Check sparsity progression
        assert_abs_diff_eq!(schedule.target_sparsity(0), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(schedule.target_sparsity(5), 0.9, epsilon = 1e-6);
    }

    #[test]
    fn test_gradual_schedule() {
        let schedule = PruningSchedule::Gradual {
            start_step: 10,
            end_step: 20,
            initial_sparsity: 0.0,
            final_sparsity: 0.8,
        };

        assert!(!schedule.should_prune(5));
        assert!(schedule.should_prune(15));
        assert!(!schedule.should_prune(25));

        assert_abs_diff_eq!(schedule.target_sparsity(10), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(schedule.target_sparsity(15), 0.4, epsilon = 1e-6);
        assert_abs_diff_eq!(schedule.target_sparsity(20), 0.8, epsilon = 1e-6);
    }

    #[test]
    fn test_polynomial_schedule() {
        let schedule = PruningSchedule::Polynomial {
            start_step: 0,
            end_step: 100,
            initial_sparsity: 0.0,
            final_sparsity: 0.9,
            power: 3.0,
        };

        let sparsity_25 = schedule.target_sparsity(25);
        let sparsity_75 = schedule.target_sparsity(75);

        // Cubic schedule should grow slowly at first, then faster
        // At 25%: (0.25)^3 = 0.015625 → 0.9 * 0.015625 = 0.014
        // At 75%: (0.75)^3 = 0.421875 → 0.9 * 0.421875 = 0.38
        assert!(sparsity_25 < 0.1);
        assert!(sparsity_75 > 0.3 && sparsity_75 < 0.5);
    }

    #[test]
    fn test_network_pruner() {
        let mut pruner = NetworkPruner::magnitude_gradual(0.1, 0, 10, 0.5);

        let mut weights = Array2::from_elem((5, 5), 0.5);

        // Step 0: should prune
        let stats = pruner.prune_weights(&mut weights, 0);
        assert_eq!(stats.total_weights, 25);

        // Step 5: should prune (within range)
        let stats = pruner.prune_weights(&mut weights, 5);
        assert!(stats.sparsity >= 0.0);

        // Step 15: should not prune (outside range)
        let _stats = pruner.prune_weights(&mut weights, 15);
    }

    #[test]
    fn test_pruning_stats() {
        let stats = PruningStats {
            total_weights: 100,
            pruned_weights: 75,
            newly_pruned: 75,
            sparsity: 0.75,
        };

        assert_eq!(stats.remaining_weights(), 25);
        assert_abs_diff_eq!(stats.compression_ratio(), 0.25, epsilon = 1e-6);
        assert_abs_diff_eq!(stats.pruned_percentage(), 75.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pruning_stats_merge() {
        let stats1 = PruningStats {
            total_weights: 100,
            pruned_weights: 50,
            newly_pruned: 50,
            sparsity: 0.5,
        };

        let stats2 = PruningStats {
            total_weights: 200,
            pruned_weights: 100,
            newly_pruned: 100,
            sparsity: 0.5,
        };

        let merged = stats1.merge(&stats2);
        assert_eq!(merged.total_weights, 300);
        assert_eq!(merged.pruned_weights, 150);
        assert_abs_diff_eq!(merged.sparsity, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_pruning_mask() {
        let weights = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.5, 0.0]).unwrap();

        let mask = PruningMask::from_weights(&weights);

        assert!(mask.mask[[0, 0]]);
        assert!(!mask.mask[[0, 1]]);
        assert!(mask.mask[[1, 0]]);
        assert!(!mask.mask[[1, 1]]);

        assert_abs_diff_eq!(mask.sparsity(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_pruning_mask_apply() {
        let mut weights = Array2::ones((2, 2));
        let mask = PruningMask {
            mask: Array2::from_shape_vec((2, 2), vec![true, false, true, false]).unwrap(),
        };

        mask.apply(&mut weights);

        assert_eq!(weights[[0, 0]], 1.0);
        assert_eq!(weights[[0, 1]], 0.0);
        assert_eq!(weights[[1, 0]], 1.0);
        assert_eq!(weights[[1, 1]], 0.0);
    }

    #[test]
    fn test_pruning_mask_intersect() {
        let mask1 = PruningMask {
            mask: Array2::from_shape_vec((2, 2), vec![true, true, false, false]).unwrap(),
        };

        let mask2 = PruningMask {
            mask: Array2::from_shape_vec((2, 2), vec![true, false, true, false]).unwrap(),
        };

        let intersect = mask1.intersect(&mask2);

        assert!(intersect.mask[[0, 0]]);
        assert!(!intersect.mask[[0, 1]]);
        assert!(!intersect.mask[[1, 0]]);
        assert!(!intersect.mask[[1, 1]]);
    }

    #[test]
    fn test_pruning_mask_union() {
        let mask1 = PruningMask {
            mask: Array2::from_shape_vec((2, 2), vec![true, false, false, false]).unwrap(),
        };

        let mask2 = PruningMask {
            mask: Array2::from_shape_vec((2, 2), vec![false, true, false, false]).unwrap(),
        };

        let union = mask1.union(&mask2);

        assert!(union.mask[[0, 0]]);
        assert!(union.mask[[0, 1]]);
        assert!(!union.mask[[1, 0]]);
        assert!(!union.mask[[1, 1]]);
    }

    #[test]
    fn test_compute_sparsity() {
        let pruner = NetworkPruner::magnitude_gradual(0.1, 0, 10, 0.5);

        let weights = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.5, 0.0, 0.0, 1.0]).unwrap();

        let sparsity = pruner.compute_sparsity(&weights);
        assert_abs_diff_eq!(sparsity, 3.0 / 6.0, epsilon = 1e-6);
    }

    // ---- schedule and statistics ---------------------------------------

    /// A gradual schedule must ramp the sparsity, not jump to its final value.
    ///
    /// `prune_weights` consulted `should_prune` and never `target_sparsity`, so
    /// the strategy pruned to its own fixed figure every time: the schedule
    /// decided when to prune but never how far, and the ramp was dead code.
    #[test]
    fn gradual_schedule_ramps_the_sparsity() {
        let schedule = PruningSchedule::Gradual {
            start_step: 0,
            end_step: 10,
            initial_sparsity: 0.0,
            final_sparsity: 0.8,
        };
        let mut pruner = NetworkPruner::new(
            PruningStrategy::TopK {
                target_sparsity: 0.8,
            },
            schedule,
        );

        let fresh = || Array2::from_shape_fn((10, 10), |(i, j)| ((i * 10 + j) as f64 + 1.0) * 0.01);

        // Early in the ramp, far less should be pruned than at the end.
        let mut early = fresh();
        let early_stats = pruner.prune_weights(&mut early, 2);

        let mut late = fresh();
        let late_stats = pruner.prune_weights(&mut late, 10);

        assert!(
            early_stats.sparsity < late_stats.sparsity,
            "step 2 pruned {} and step 10 pruned {}; the ramp is being ignored",
            early_stats.sparsity,
            late_stats.sparsity
        );
        assert!(
            early_stats.sparsity < 0.5,
            "step 2 of a 10-step ramp to 0.8 pruned {}, which is already most of it",
            early_stats.sparsity
        );
    }

    /// Sparsity must describe the tensor, not the last call's delta.
    ///
    /// The strategies counted only the weights they newly zeroed, so pruning an
    /// already-sparse tensor reported a fraction of the truth -- and
    /// `remaining_weights` and `compression_ratio` inherited the error.
    #[test]
    fn stats_report_total_sparsity_not_the_delta() {
        let strategy = PruningStrategy::TopK {
            target_sparsity: 0.5,
        };
        let mut w = Array2::from_shape_fn((10, 10), |(i, j)| ((i * 10 + j) as f64 + 1.0) * 0.01);

        let first = strategy.prune(&mut w, 0);
        assert!(
            (first.sparsity - 0.5).abs() < 0.05,
            "first prune: {}",
            first.sparsity
        );
        assert_eq!(first.pruned_weights, 50);
        assert_eq!(first.newly_pruned, 50);

        // Prune again to the same target: almost nothing new, but the tensor is
        // still half zero.
        let second = strategy.prune(&mut w, 1);
        assert_eq!(
            second.pruned_weights, 50,
            "the tensor is still half pruned, whatever this call did"
        );
        assert!(
            (second.sparsity - 0.5).abs() < 0.05,
            "second: {}",
            second.sparsity
        );
        assert_eq!(
            second.newly_pruned, 0,
            "nothing new should have been pruned"
        );
        assert_eq!(second.remaining_weights(), 50);
    }
}
