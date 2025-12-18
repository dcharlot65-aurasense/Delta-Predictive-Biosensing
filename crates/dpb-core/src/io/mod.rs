//! # I/O Module
//!
//! This module provides support for reading and writing various physiological data formats.
//!
//! ## Supported Formats
//!
//! - **WFDB** ([`wfdb`]) - PhysioNet WFDB (Waveform Database) format, commonly used for
//!   ECG data and other physiological signals (MIT-BIH compatible)
//! - **EDF** ([`edf`]) - European Data Format, widely used for polysomnography and EEG data
//! - **BDF** ([`bdf`]) - BioSemi Data Format, 24-bit variant of EDF for high-resolution recordings
//! - **GDF** ([`gdf`]) - General Data Format, extended EDF with additional features
//! - **XDF** ([`xdf`]) - Extensible Data Format, multi-stream format for Lab Streaming Layer
//!
//! ## Format Detection
//!
//! The [`format_detect`] module provides automatic format detection and unified readers.
//!
//! ## Examples
//!
//! ### Reading WFDB files
//!
//! ```rust,no_run
//! use dpb_core::io::WfdbReader;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let reader = WfdbReader::open(Path::new("data/mitdb/100"))?;
//! let samples = reader.read_all_samples(0)?;
//! println!("Read {} samples from channel 0", samples.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Reading EDF files
//!
//! ```rust,no_run
//! use dpb_core::io::EdfReader;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let reader = EdfReader::open(Path::new("data/sleep.edf"))?;
//! let signal = reader.read_signal(0)?;
//! println!("Read {} samples", signal.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Reading BDF files
//!
//! ```rust,no_run
//! use dpb_core::io::BdfReader;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let reader = BdfReader::open(Path::new("data/biosemi.bdf"))?;
//! let signal = reader.read_signal(0)?;
//! let triggers = reader.read_triggers()?;
//! println!("Read {} samples and {} triggers", signal.len(), triggers.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Reading GDF files
//!
//! ```rust,no_run
//! use dpb_core::io::GdfReader;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let reader = GdfReader::open(Path::new("data/recording.gdf"))?;
//! println!("GDF version: {:?}", reader.header().version);
//! let signal = reader.read_signal(0)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Reading XDF files
//!
//! ```rust,no_run
//! use dpb_core::io::XdfFile;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let xdf = XdfFile::open(Path::new("data/multi_stream.xdf"))?;
//! for stream_id in xdf.stream_ids() {
//!     if let Some(stream) = xdf.stream(stream_id) {
//!         println!("Stream: {} ({} samples)",
//!                  stream.info.name, stream.sample_count());
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Automatic format detection
//!
//! ```rust,no_run
//! use dpb_core::io::{UnifiedReader, UnifiedBiosignalReader};
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Automatically detects format and opens appropriate reader
//! let mut reader = UnifiedReader::open(Path::new("data/unknown_format.edf"))?;
//! println!("Detected format: {}", reader.format_type().name());
//! let signal = reader.read_signal(0)?;
//! # Ok(())
//! # }
//! ```

pub mod bdf;
pub mod edf;
pub mod format_detect;
pub mod gdf;
pub mod wfdb;
pub mod xdf;

// Re-export main types from each format
pub use bdf::{BdfHeader, BdfReader, BdfSignal, BdfTrigger, BdfWriter};
pub use edf::{EdfHeader, EdfReader, EdfSignal, EdfWriter};
pub use format_detect::{
    detect_format, FormatType, UnifiedBiosignalReader, UnifiedReader,
};
pub use gdf::{
    GdfDataType, GdfEvent, GdfHeader, GdfReader, GdfSignal, GdfVersion, GdfWriter,
};
pub use wfdb::{
    AnnotationType, WfdbAnnotation, WfdbHeader, WfdbReader, WfdbSignal, WfdbWriter,
};
pub use xdf::{ChannelFormat, XdfFile, XdfStream, XdfStreamInfo, XdfWriter};
