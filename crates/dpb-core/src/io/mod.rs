//! # I/O Module
//!
//! This module provides support for reading and writing various physiological data formats.
//!
//! ## Supported Formats
//!
//! - **WFDB** ([`wfdb`]) - PhysioNet WFDB (Waveform Database) format, commonly used for
//!   ECG data and other physiological signals (MIT-BIH compatible)
//! - **EDF** ([`edf`]) - European Data Format, widely used for polysomnography and EEG data
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

pub mod edf;
pub mod wfdb;

pub use edf::{EdfHeader, EdfReader, EdfSignal, EdfWriter};
pub use wfdb::{
    AnnotationType, WfdbAnnotation, WfdbHeader, WfdbReader, WfdbSignal, WfdbWriter,
};
