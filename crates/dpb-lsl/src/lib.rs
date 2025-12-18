//! # dpb-lsl
//!
//! Lab Streaming Layer (LSL) integration for real-time biosignal streaming with the
//! Delta-Predictive Biosensing (DPB) Framework.
//!
//! ## Overview
//!
//! LSL is the de facto standard for synchronized streaming of biosignals in research.
//! This crate provides:
//!
//! - **Inlet**: Receive biosignals from LSL streams
//! - **Outlet**: Send spike trains and processed data over LSL
//! - **Real-time Encoding**: Pipeline for live spike encoding
//! - **Stream Discovery**: Find available LSL streams on the network
//!
//! ## Features
//!
//! - Multi-device synchronization with sub-millisecond precision
//! - Network-transparent data transfer
//! - Async support with Tokio (optional feature)
//! - Integration with existing DPB encoders
//!
//! ## Example
//!
//! ```rust,ignore
//! use dpb_lsl::{LslInlet, LslOutlet, StreamResolver};
//!
//! // Find EEG streams
//! let resolver = StreamResolver::new();
//! let streams = resolver.resolve_by_type("EEG", 5.0)?;
//!
//! // Connect to first stream
//! let inlet = LslInlet::new(&streams[0])?;
//!
//! // Pull samples
//! while let Ok((sample, timestamp)) = inlet.pull_sample(1.0) {
//!     println!("Sample at {}: {:?}", timestamp, sample);
//! }
//! ```
//!
//! ## Note on liblsl Dependency
//!
//! This crate requires the liblsl library to be installed on the system.
//! See <https://github.com/sccn/liblsl> for installation instructions.

pub mod error;
pub mod stream_info;
pub mod inlet;
pub mod outlet;
pub mod resolver;
pub mod pipeline;

pub use error::{LslError, Result};
pub use stream_info::StreamInfo;
pub use inlet::LslInlet;
pub use outlet::LslOutlet;
pub use resolver::StreamResolver;
pub use pipeline::{EncodingPipeline, PipelineConfig};

/// LSL channel format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelFormat {
    Float32,
    Float64,
    String,
    Int32,
    Int16,
    Int8,
    Undefined,
}

impl ChannelFormat {
    /// Bytes per sample for this format.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            ChannelFormat::Float32 => 4,
            ChannelFormat::Float64 => 8,
            ChannelFormat::Int32 => 4,
            ChannelFormat::Int16 => 2,
            ChannelFormat::Int8 => 1,
            ChannelFormat::String => 0, // Variable
            ChannelFormat::Undefined => 0,
        }
    }
}

/// Stream type constants for common biosignal types.
pub mod stream_types {
    /// Electroencephalography
    pub const EEG: &str = "EEG";
    /// Electrocardiography
    pub const ECG: &str = "ECG";
    /// Electromyography
    pub const EMG: &str = "EMG";
    /// Electrooculography
    pub const EOG: &str = "EOG";
    /// Photoplethysmography
    pub const PPG: &str = "PPG";
    /// Electrodermal Activity / Galvanic Skin Response
    pub const EDA: &str = "EDA";
    /// Respiration
    pub const RESP: &str = "Respiration";
    /// Accelerometer
    pub const ACC: &str = "Accelerometer";
    /// Gyroscope
    pub const GYRO: &str = "Gyroscope";
    /// Markers/Events
    pub const MARKERS: &str = "Markers";
    /// Spike trains (DPB custom type)
    pub const SPIKES: &str = "Spikes";
}
