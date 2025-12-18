//! Vestibular Signal Generators
//!
//! This module provides synthetic vestibular and oculomotor signals:
//! - Vestibulo-ocular reflex (VOR) responses
//! - Nystagmus patterns
//! - Head impulse test (HIT) responses
//! - Caloric test responses
//! - Pathological patterns (BPPV, vestibular neuritis, bilateral loss)
//!
//! All generators provide ground truth for algorithm validation.

pub mod vor;
pub mod nystagmus;
pub mod caloric;

pub use vor::{
    VorGenerator, VorConfig, VorOutput, VorGroundTruth,
    HeadImpulseResult, VorGain, CatchUpSaccade,
};
pub use nystagmus::{
    NystagmusGenerator, NystagmusConfig, NystagmusOutput,
    NystagmusGroundTruth, NystagmusType, NystagmusDirection,
    BeatInfo,
};
pub use caloric::{
    CaloricGenerator, CaloricConfig, CaloricOutput,
    CaloricGroundTruth, CaloricStimulus, CaloricResponse,
};
