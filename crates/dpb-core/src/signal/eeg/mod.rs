//! EEG signal processing and analysis.
//!
//! This module provides tools for processing and analyzing electroencephalography (EEG) signals,
//! including frequency band analysis, artifact detection, seizure detection, and common
//! preprocessing operations.

pub mod artifacts;
pub mod bands;
pub mod erp;
pub mod seizure;

pub use artifacts::{
    ArtifactSegment, ArtifactType, apply_notch_filter, detect_artifacts, remove_dc_offset,
    remove_trend,
};
pub use bands::{
    BandPowers, EegBands, alpha_asymmetry, compute_band_powers, extract_band_power,
    relative_band_power, theta_beta_ratio,
};
pub use erp::{Erp, ErpAnalyzer, ErpComponent, ErpGenerator, OddballData};
pub use seizure::{
    AlertLevel, EpileptiformSpike, SeizureAnalysisResult, SeizureDetector, SeizureEvent,
    SeizureEvolution, SeizureType, analyze_for_seizures,
};
