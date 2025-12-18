//! Neural signal synthesis
//!
//! This module provides synthetic neural signal generation:
//! - EEG with sleep stages and artifacts
//! - ERP components (P300, N400, MMN, etc.)
//! - Sleep microstructure (spindles, K-complexes, slow oscillations)

pub mod eeg;
pub mod erp;
pub mod sleep;

pub use eeg::*;
pub use erp::{
    ErpGenerator, ErpConfig, ErpOutput, ErpGroundTruth,
    ErpComponentType, ErpParadigm, ErpPathology,
    ErpComponent, ErpComponentInfo, Topography,
};
pub use sleep::{
    SleepMicrostructureGenerator, SleepMicroConfig, SleepMicroOutput,
    SleepMicroGroundTruth, SpindleInfo, KComplexInfo, SlowOscillationInfo,
    SleepMicroPathology,
};
