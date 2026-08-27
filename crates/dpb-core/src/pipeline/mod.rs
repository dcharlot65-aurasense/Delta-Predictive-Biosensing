//! Real-time biosignal processing pipeline infrastructure
//!
//! Provides streaming inference capabilities with configurable latency budgets,
//! buffer management, and multi-stage processing.

pub mod buffer;
pub mod executor;
pub mod stage;

pub use buffer::{OverlapBuffer, RingBuffer, SlidingWindow};
pub use executor::{ExecutionMode, LatencyStats, PipelineConfig, PipelineExecutor};
pub use stage::{PipelineStage, StageConfig, StageMetrics};
