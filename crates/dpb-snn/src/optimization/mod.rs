//! Optimization techniques for Spiking Neural Networks
//!
//! This module provides various optimization techniques to improve SNN efficiency:
//!
//! - **Pruning** - Remove unnecessary weights to reduce model size
//!   - Magnitude-based pruning
//!   - Gradient-based pruning
//!   - Structured pruning
//!   - Iterative and gradual pruning schedules

pub mod pruning;

// Re-export commonly used types
pub use pruning::{NetworkPruner, PruningMask, PruningSchedule, PruningStats, PruningStrategy};
