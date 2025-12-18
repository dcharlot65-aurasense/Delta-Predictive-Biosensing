//! Somatosensory and proprioceptive assessment
//!
//! Provides algorithms for assessing:
//! - Joint position sense (proprioception)
//! - Vibration perception threshold
//! - Movement detection threshold

pub mod proprioception;
pub mod vibration;

pub use proprioception::{
    Joint, JointPositionSense, JpsMetrics, JpsTrialResult, ProprioceptionTestType, TtdpmResult,
};
pub use vibration::{NeuropathyRisk, VibrationSense, VibrationSite, VptMetrics};
