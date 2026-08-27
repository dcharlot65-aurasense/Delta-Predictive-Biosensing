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
    ForceEvent, GaitPhaseLabel, GrfConfig, GrfGenerator, GrfGroundTruth, GrfOutput, PathologicalGrf,
};
pub use grip::{
    GripConfig, GripEvent, GripEventType, GripGenerator, GripGroundTruth, GripOutput,
    PathologicalGrip,
};
pub use rfd::{PathologicalRfd, RfdConfig, RfdGenerator, RfdGroundTruth, RfdOutput, RfdTaskType};
