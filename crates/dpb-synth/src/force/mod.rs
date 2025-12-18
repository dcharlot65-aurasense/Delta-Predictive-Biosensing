//! Force dynamics signal generators
//!
//! This module provides synthetic force signal generation for:
//! - Ground reaction forces (GRF) during walking, running, jumping
//! - Grip strength dynamometry with fatigue modeling
//! - Rate of force development (RFD) profiles
//!
//! All generators provide ground truth annotations for algorithm validation.

pub mod grf;
pub mod grip;
pub mod rfd;

pub use grf::{
    GrfGenerator, GrfConfig, GrfOutput, GrfGroundTruth,
    GaitPhaseLabel, ForceEvent, PathologicalGrf,
};
pub use grip::{
    GripStrengthGenerator, GripConfig, GripOutput, GripGroundTruth,
    GripProtocol, GripPathology,
};
pub use rfd::{
    RfdGenerator, RfdConfig, RfdOutput, RfdGroundTruth,
    RfdType, RfdMetrics,
};
