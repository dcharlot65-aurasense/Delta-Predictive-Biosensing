//! Cognitive assessment paradigms for Delta Predictive Biosensing
//!
//! This crate provides implementations of standard cognitive assessment tasks
//! used in neuropsychological testing and cognitive neuroscience research.
//!
//! ## Modules
//!
//! - `reaction_time` - Simple and choice reaction time paradigms
//! - `working_memory` - N-back and other working memory tasks
//! - `attention` - CPT, Stroop, and sustained attention tasks
//! - `executive` - Executive function tasks (Go/No-Go, Flanker, WCST)
//! - `adhd` - ADHD-specific paradigms (QbTest, eye tracking)

pub mod adhd;
pub mod attention;
pub mod executive;
pub mod reaction_time;
pub mod working_memory;

// Re-export commonly used types
pub use adhd::{
    ActivityMetrics, AdhdEyeTracking, AdhdGazeMetrics, AdhdPattern, AttentionMetrics,
    ImpulsivityMetrics, QbTest, QbTestMetrics,
};
pub use attention::{ContinuousPerformanceTest, CptMetrics, StroopTask, StroopMetrics};
pub use executive::{FlankerMetrics, FlankerTask, GoNoGoMetrics, GoNoGoTask, WcstMetrics, WisconsinCardSort};
pub use reaction_time::{ChoiceReactionTime, ReactionTimeMetrics, SimpleReactionTime};
pub use working_memory::{NBackMetrics, NBackTask};
