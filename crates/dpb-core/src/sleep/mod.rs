//! Sleep assessment and staging algorithms
//!
//! Provides tools for sleep stage classification and sleep architecture analysis
//! using EEG, HRV, and actigraphy data.

pub mod features;
pub mod staging;

pub use features::{KComplex, SleepFeatureExtractor, SleepSpindle, SlowWave};
pub use staging::{SleepArchitecture, SleepStage, SleepStager};
