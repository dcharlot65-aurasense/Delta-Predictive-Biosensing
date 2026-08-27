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
//! ```rust,no_run
//! use dpb_lsl::{LslInlet, StreamResolver};
//!
//! # fn example() -> dpb_lsl::Result<()> {
//! // Find EEG streams. `no_run` because this needs a live LSL network.
//! let resolver = StreamResolver::new();
//! let streams = resolver.resolve_by_type("EEG", 5.0)?;
//!
//! // Connect to first stream
//! // `None` takes the default inlet configuration.
//! let mut inlet = LslInlet::new(&streams[0], None)?;
//!
//! // Pull samples
//! while let Ok((sample, timestamp)) = inlet.pull_sample(1.0) {
//!     println!("Sample at {timestamp}: {sample:?}");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Note on liblsl Dependency
//!
//! This crate requires the liblsl library to be installed on the system.
//! See <https://github.com/sccn/liblsl> for installation instructions.
//!
//! ### Native Feature
//!
//! Enable the `native` feature to link against the real liblsl library:
//!
//! ```toml
//! [dependencies]
//! dpb-lsl = { version = "0.1", features = ["native"] }
//! ```
//!
//! Without the `native` feature, mock implementations are used for testing.

pub mod error;
pub mod ffi;
pub mod inlet;
pub mod outlet;
pub mod pipeline;
pub mod resolver;
pub mod stream_info;

#[cfg(feature = "native")]
pub mod native;

pub use error::{LslError, Result};
pub use inlet::LslInlet;
pub use outlet::LslOutlet;
pub use pipeline::{EncodingPipeline, PipelineConfig};
pub use resolver::StreamResolver;
pub use stream_info::StreamInfo;

use serde::{Deserialize, Serialize};

/// LSL channel format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_format_bytes_per_sample() {
        assert_eq!(ChannelFormat::Float32.bytes_per_sample(), 4);
        assert_eq!(ChannelFormat::Float64.bytes_per_sample(), 8);
        assert_eq!(ChannelFormat::Int32.bytes_per_sample(), 4);
        assert_eq!(ChannelFormat::Int16.bytes_per_sample(), 2);
        assert_eq!(ChannelFormat::Int8.bytes_per_sample(), 1);
        assert_eq!(ChannelFormat::String.bytes_per_sample(), 0);
        assert_eq!(ChannelFormat::Undefined.bytes_per_sample(), 0);
    }

    #[test]
    fn test_channel_format_equality() {
        assert_eq!(ChannelFormat::Float32, ChannelFormat::Float32);
        assert_ne!(ChannelFormat::Float32, ChannelFormat::Float64);
        assert_ne!(ChannelFormat::Int32, ChannelFormat::Int16);
    }

    #[test]
    fn test_channel_format_clone() {
        let format = ChannelFormat::Float32;
        let cloned = format;
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_channel_format_copy() {
        let format = ChannelFormat::Int16;
        let copied: ChannelFormat = format; // Copy, not move
        assert_eq!(format, copied);
    }

    #[test]
    fn test_channel_format_debug() {
        let debug_str = format!("{:?}", ChannelFormat::Float64);
        assert!(debug_str.contains("Float64"));
    }

    #[test]
    fn test_stream_types_constants() {
        assert_eq!(stream_types::EEG, "EEG");
        assert_eq!(stream_types::ECG, "ECG");
        assert_eq!(stream_types::EMG, "EMG");
        assert_eq!(stream_types::EOG, "EOG");
        assert_eq!(stream_types::PPG, "PPG");
        assert_eq!(stream_types::EDA, "EDA");
        assert_eq!(stream_types::RESP, "Respiration");
        assert_eq!(stream_types::ACC, "Accelerometer");
        assert_eq!(stream_types::GYRO, "Gyroscope");
        assert_eq!(stream_types::MARKERS, "Markers");
        assert_eq!(stream_types::SPIKES, "Spikes");
    }

    #[test]
    fn test_stream_types_are_not_empty() {
        assert!(!stream_types::EEG.is_empty());
        assert!(!stream_types::ECG.is_empty());
        assert!(!stream_types::SPIKES.is_empty());
    }

    #[test]
    fn test_public_exports() {
        // Verify that public types are accessible
        let _: fn(String, f64) -> LslError =
            |name: String, timeout: f64| LslError::StreamNotFound {
                name,
                timeout_sec: timeout,
            };

        // StreamInfo creation should work
        let info_result = StreamInfo::new(
            "Test",
            stream_types::EEG,
            8,
            256.0,
            ChannelFormat::Float32,
            "source",
        );
        assert!(info_result.is_ok());
    }
}
