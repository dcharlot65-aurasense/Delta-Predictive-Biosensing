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

pub mod caloric;
pub mod nystagmus;
pub mod vor;

pub use caloric::{
    CaloricConfig, CaloricGenerator, CaloricGroundTruth, CaloricOutput, CaloricResponse,
    CaloricStimulus,
};
pub use nystagmus::{
    BeatInfo, NystagmusConfig, NystagmusDirection, NystagmusGenerator, NystagmusGroundTruth,
    NystagmusOutput, NystagmusType,
};
pub use vor::{
    CatchUpSaccade, HeadImpulseResult, VorConfig, VorGain, VorGenerator, VorGroundTruth, VorOutput,
};
