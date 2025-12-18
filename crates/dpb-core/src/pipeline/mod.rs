//! Real-time biosignal processing pipeline infrastructure
//!
//! Provides streaming inference capabilities with configurable latency budgets,
//! buffer management, and multi-stage processing.

pub mod buffer;
pub mod stage;
pub mod executor;

pub use buffer::{RingBuffer, SlidingWindow, OverlapBuffer};
pub use stage::{PipelineStage, StageConfig, StageMetrics};
pub use executor::{PipelineExecutor, PipelineConfig, LatencyStats, ExecutionMode};
