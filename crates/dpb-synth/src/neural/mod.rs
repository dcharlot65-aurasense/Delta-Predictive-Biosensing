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
    ErpComponent, ErpComponentInfo, ErpComponentType, ErpConfig, ErpGenerator, ErpGroundTruth,
    ErpOutput, ErpParadigm, ErpPathology, Topography,
};
pub use sleep::{
    KComplexInfo, SleepMicroConfig, SleepMicroGroundTruth, SleepMicroOutput, SleepMicroPathology,
    SleepMicrostructureGenerator, SlowOscillationInfo, SpindleInfo,
};
