//! Balance and Center of Pressure (COP) Signal Generators
//!
//! This module provides synthetic postural control signals:
//! - Static balance (quiet standing COP)
//! - Dynamic balance (limits of stability, weight shifting)
//! - Perturbation responses
//! - Sensory manipulation conditions
//! - Pathological patterns (Parkinson's, vestibular, cerebellar)
//!
//! All generators provide ground truth for algorithm validation.

pub mod cop;
pub mod perturbation;
pub mod sensory;

pub use cop::{
    CopConfig, CopGenerator, CopGroundTruth, CopMetrics, CopOutput, PathologicalBalance,
    StanceCondition, SwaySummary,
};
pub use perturbation::{
    PerturbationConfig, PerturbationDirection, PerturbationGenerator, PerturbationGroundTruth,
    PerturbationOutput, PerturbationType, RecoveryStrategy,
};
pub use sensory::{
    SensoryCondition, SensoryConfig, SensoryGroundTruth, SensoryManipulationGenerator,
    SensoryOutput,
};
