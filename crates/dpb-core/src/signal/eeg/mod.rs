//! EEG signal processing and analysis.
//!
//! This module provides tools for processing and analyzing electroencephalography (EEG) signals,
//! including frequency band analysis, artifact detection, and common preprocessing operations.

pub mod artifacts;
pub mod bands;

pub use artifacts::{
    apply_notch_filter, detect_artifacts, remove_dc_offset, remove_trend, ArtifactSegment,
    ArtifactType,
};
pub use bands::{
    alpha_asymmetry, compute_band_powers, extract_band_power, relative_band_power,
    theta_beta_ratio, BandPowers, EegBands,
};
