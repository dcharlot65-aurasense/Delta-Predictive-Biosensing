//! ADHD-specific assessment paradigms
//!
//! Provides specialized tests for ADHD evaluation:
//! - QbTest-style continuous performance test
//! - ADHD-specific eye tracking patterns
//! - Activity monitoring during testing

pub mod eye_tracking;
pub mod paradigms;

pub use eye_tracking::{
    AdhdEyeTracking, AdhdGazeMetrics, FixationStability, ReadingSaccadeMetrics,
};
pub use paradigms::{
    ActivityMetrics, AdhdPattern, AttentionMetrics, ImpulsivityMetrics, QbTest, QbTestMetrics,
    QbTestStimulus,
};
