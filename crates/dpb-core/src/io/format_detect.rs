//! # Biosignal Format Auto-Detection
//!
//! This module provides utilities for automatically detecting biosignal file formats
//! and creating unified readers that work across different formats.
//!
//! ## Supported Formats
//!
//! - **EDF** - European Data Format (.edf)
//! - **BDF** - BioSemi Data Format (.bdf)
//! - **GDF** - General Data Format (.gdf)
//! - **XDF** - Extensible Data Format (.xdf)
//! - **WFDB** - PhysioNet Waveform Database (.hea/.dat)
//!
//! ## Detection Strategy
//!
//! 1. Check magic bytes at file start
//! 2. Validate file structure
//! 3. Fall back to extension-based detection
//!
//! ## Example
//!
//! ```rust,no_run
//! use dpb_core::io::{detect_format, FormatType};
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let path = Path::new("data/recording.edf");
//! let format = detect_format(path)?;
//!
//! match format {
//!     FormatType::EDF => println!("Detected EDF format"),
//!     FormatType::BDF => println!("Detected BDF format"),
//!     _ => println!("Other format"),
//! }
//! # Ok(())
//! # }
//! ```

use crate::error::{DpbError, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

// These are established domain acronyms -- clinical file formats, ECG
// beat annotations, and hardware terms. Camel-casing them (Pvc, Wfdb,
// Dram) would make this harder to read for anyone who works with them.
#[allow(clippy::upper_case_acronyms)]
/// Biosignal file format type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatType {
    /// European Data Format
    EDF,
    /// BioSemi Data Format (24-bit EDF variant)
    BDF,
    /// General Data Format
    GDF,
    /// Extensible Data Format (Lab Streaming Layer)
    XDF,
    /// PhysioNet Waveform Database
    WFDB,
    /// Unknown format
    Unknown,
}

impl FormatType {
    /// Get typical file extension for this format
    pub fn extension(&self) -> &str {
        match self {
            Self::EDF => "edf",
            Self::BDF => "bdf",
            Self::GDF => "gdf",
            Self::XDF => "xdf",
            Self::WFDB => "hea",
            Self::Unknown => "",
        }
    }

    /// Get format name
    pub fn name(&self) -> &str {
        match self {
            Self::EDF => "European Data Format",
            Self::BDF => "BioSemi Data Format",
            Self::GDF => "General Data Format",
            Self::XDF => "Extensible Data Format",
            Self::WFDB => "WFDB Format",
            Self::Unknown => "Unknown",
        }
    }

    /// Check if format uses 24-bit samples
    pub fn uses_24bit(&self) -> bool {
        matches!(self, Self::BDF)
    }

    /// Check if format supports multiple streams
    pub fn supports_multiple_streams(&self) -> bool {
        matches!(self, Self::XDF)
    }
}

/// Detect biosignal file format
///
/// # Arguments
///
/// * `path` - Path to the file to analyze
///
/// # Returns
///
/// Detected format type
///
/// # Example
///
/// ```rust,no_run
/// use dpb_core::io::detect_format;
/// use std::path::Path;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let format = detect_format(Path::new("data/recording.edf"))?;
/// println!("Detected: {}", format.name());
/// # Ok(())
/// # }
/// ```
pub fn detect_format(path: &Path) -> Result<FormatType> {
    // First, try magic byte detection
    if let Ok(format) = detect_by_magic_bytes(path)
        && format != FormatType::Unknown {
            return Ok(format);
        }

    // Fall back to extension-based detection
    detect_by_extension(path)
}

/// Detect format by reading magic bytes
fn detect_by_magic_bytes(path: &Path) -> Result<FormatType> {
    let mut file = File::open(path)?;
    let mut header = vec![0u8; 256];

    // Read up to 256 bytes for header analysis
    let bytes_read = file.read(&mut header)?;

    if bytes_read < 8 {
        return Ok(FormatType::Unknown);
    }

    // Check XDF magic bytes (easiest to detect)
    if &header[0..4] == b"XDF:" {
        return Ok(FormatType::XDF);
    }

    // Check GDF version string
    let version_str = String::from_utf8_lossy(&header[0..8]);
    if version_str.starts_with("GDF") {
        return Ok(FormatType::GDF);
    }

    // Check BDF (version byte = 255)
    if header[0] == 0xFF {
        // Additional validation: check if it looks like BDF/EDF structure
        if bytes_read >= 256 && is_valid_edf_like_header(&header) {
            return Ok(FormatType::BDF);
        }
    }

    // Check EDF (version string should be "0       " - 8 spaces with leading 0)
    if header[0] == b'0' && header[1..8].iter().all(|&b| b == b' ' || b == 0)
        && is_valid_edf_like_header(&header) {
            return Ok(FormatType::EDF);
        }

    // Check WFDB header file (text-based)
    if is_text_based(&header[0..bytes_read.min(128)]) {
        // WFDB headers are ASCII text with specific format
        if is_wfdb_header(&header[0..bytes_read]) {
            return Ok(FormatType::WFDB);
        }
    }

    Ok(FormatType::Unknown)
}

/// Detect format by file extension
fn detect_by_extension(path: &Path) -> Result<FormatType> {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    let format = match extension.as_str() {
        "edf" => FormatType::EDF,
        "bdf" => FormatType::BDF,
        "gdf" => FormatType::GDF,
        "xdf" => FormatType::XDF,
        "hea" => FormatType::WFDB,
        _ => FormatType::Unknown,
    };

    Ok(format)
}

/// Check if header looks like valid EDF/BDF structure
fn is_valid_edf_like_header(header: &[u8]) -> bool {
    if header.len() < 256 {
        return false;
    }

    // Check if date field looks valid (dd.mm.yy at offset 168)
    let date_field = &header[168..176];
    if !date_field.iter().all(|&b| b.is_ascii_digit() || b == b'.' || b == b' ') {
        return false;
    }

    // Check if time field looks valid (hh.mm.ss at offset 176)
    let time_field = &header[176..184];
    if !time_field.iter().all(|&b| b.is_ascii_digit() || b == b'.' || b == b' ') {
        return false;
    }

    // Check if number of signals is reasonable (< 1000)
    if let Ok(n_signals_str) = std::str::from_utf8(&header[252..256])
        && let Ok(n_signals) = n_signals_str.trim().parse::<usize>() {
            return n_signals > 0 && n_signals < 1000;
        }

    false
}

/// Check if buffer contains mostly text (ASCII printable)
fn is_text_based(buffer: &[u8]) -> bool {
    let printable_count = buffer.iter().filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace()).count();
    let ratio = printable_count as f64 / buffer.len() as f64;
    ratio > 0.9
}

/// Check if header looks like WFDB format
fn is_wfdb_header(buffer: &[u8]) -> bool {
    if let Ok(text) = std::str::from_utf8(buffer) {
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return false;
        }

        // First line should have: record_name n_signals [sampling_rate] [n_samples]
        let parts: Vec<&str> = lines[0].split_whitespace().collect();
        if parts.len() >= 2 {
            // Second part should be number of signals
            if let Ok(n_signals) = parts[1].parse::<usize>() {
                return n_signals > 0 && n_signals < 100;
            }
        }
    }

    false
}

/// Unified reader trait for different formats
pub trait UnifiedBiosignalReader {
    /// Get number of signals/channels
    fn n_signals(&self) -> usize;

    /// Get sampling rate for a signal
    fn sample_rate(&self, signal_index: usize) -> Result<f64>;

    /// Get signal label
    fn signal_label(&self, signal_index: usize) -> Result<String>;

    /// Read all samples for a signal
    fn read_signal(&mut self, signal_index: usize) -> Result<Vec<f64>>;

    /// Get total number of samples for a signal
    fn n_samples(&self, signal_index: usize) -> Result<usize>;
}

// These are established domain acronyms -- clinical file formats, ECG
// beat annotations, and hardware terms. Camel-casing them (Pvc, Wfdb,
// Dram) would make this harder to read for anyone who works with them.
#[allow(clippy::upper_case_acronyms)]
/// Unified reader that automatically selects the appropriate format reader
pub enum UnifiedReader {
    /// EDF reader
    EDF(crate::io::EdfReader),
    /// BDF reader
    BDF(crate::io::BdfReader),
    /// GDF reader
    GDF(crate::io::GdfReader),
    /// WFDB reader
    WFDB(crate::io::WfdbReader),
}

impl UnifiedReader {
    /// Open a file with automatic format detection
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the biosignal file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dpb_core::io::UnifiedReader;
    /// use std::path::Path;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let reader = UnifiedReader::open(Path::new("data/recording.edf"))?;
    /// println!("Detected format: {:?}", reader.format_type());
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        let format = detect_format(path)?;

        match format {
            FormatType::EDF => {
                let reader = crate::io::EdfReader::open(path)?;
                Ok(Self::EDF(reader))
            }
            FormatType::BDF => {
                let reader = crate::io::BdfReader::open(path)?;
                Ok(Self::BDF(reader))
            }
            FormatType::GDF => {
                let reader = crate::io::GdfReader::open(path)?;
                Ok(Self::GDF(reader))
            }
            FormatType::WFDB => {
                let reader = crate::io::WfdbReader::open(path)?;
                Ok(Self::WFDB(reader))
            }
            FormatType::XDF => Err(DpbError::Other(
                "XDF format requires specialized handling (use XdfFile directly)".to_string(),
            )),
            FormatType::Unknown => Err(DpbError::DataValidation(format!(
                "Unknown or unsupported file format: {}",
                path.display()
            ))),
        }
    }

    /// Get the detected format type
    pub fn format_type(&self) -> FormatType {
        match self {
            Self::EDF(_) => FormatType::EDF,
            Self::BDF(_) => FormatType::BDF,
            Self::GDF(_) => FormatType::GDF,
            Self::WFDB(_) => FormatType::WFDB,
        }
    }
}

impl UnifiedBiosignalReader for UnifiedReader {
    fn n_signals(&self) -> usize {
        match self {
            Self::EDF(reader) => reader.header().n_signals,
            Self::BDF(reader) => reader.header().n_signals,
            Self::GDF(reader) => reader.header().n_signals,
            Self::WFDB(reader) => reader.header().n_signals,
        }
    }

    fn sample_rate(&self, signal_index: usize) -> Result<f64> {
        match self {
            Self::EDF(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.sample_rate(reader.header().record_duration))
            }
            Self::BDF(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.sample_rate(reader.header().record_duration))
            }
            Self::GDF(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.sample_rate(reader.header().record_duration))
            }
            Self::WFDB(reader) => Ok(reader.header().sample_rate),
        }
    }

    fn signal_label(&self, signal_index: usize) -> Result<String> {
        match self {
            Self::EDF(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.label.clone())
            }
            Self::BDF(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.label.clone())
            }
            Self::GDF(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.label.clone())
            }
            Self::WFDB(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.name.clone())
            }
        }
    }

    fn read_signal(&mut self, signal_index: usize) -> Result<Vec<f64>> {
        match self {
            Self::EDF(reader) => reader.read_signal(signal_index),
            Self::BDF(reader) => reader.read_signal(signal_index),
            Self::GDF(reader) => reader.read_signal(signal_index),
            Self::WFDB(reader) => reader.read_all_samples(signal_index),
        }
    }

    fn n_samples(&self, signal_index: usize) -> Result<usize> {
        match self {
            Self::EDF(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.samples_per_record * reader.header().n_records.max(0) as usize)
            }
            Self::BDF(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.samples_per_record * reader.header().n_records.max(0) as usize)
            }
            Self::GDF(reader) => {
                let signal = reader
                    .signals()
                    .get(signal_index)
                    .ok_or_else(|| DpbError::InvalidParameter("Signal index out of range".to_string()))?;
                Ok(signal.samples_per_record * reader.header().n_records.max(0) as usize)
            }
            Self::WFDB(reader) => {
                reader.header().n_samples.ok_or_else(|| {
                    DpbError::Other("Number of samples not specified".to_string())
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_type_properties() {
        assert_eq!(FormatType::EDF.extension(), "edf");
        assert_eq!(FormatType::BDF.extension(), "bdf");
        assert_eq!(FormatType::GDF.extension(), "gdf");
        assert_eq!(FormatType::XDF.extension(), "xdf");
        assert_eq!(FormatType::WFDB.extension(), "hea");

        assert!(FormatType::BDF.uses_24bit());
        assert!(!FormatType::EDF.uses_24bit());

        assert!(FormatType::XDF.supports_multiple_streams());
        assert!(!FormatType::EDF.supports_multiple_streams());
    }

    #[test]
    fn test_extension_detection() {
        let path = Path::new("test.edf");
        let format = detect_by_extension(path).unwrap();
        assert_eq!(format, FormatType::EDF);

        let path = Path::new("recording.bdf");
        let format = detect_by_extension(path).unwrap();
        assert_eq!(format, FormatType::BDF);

        let path = Path::new("data.gdf");
        let format = detect_by_extension(path).unwrap();
        assert_eq!(format, FormatType::GDF);

        let path = Path::new("stream.xdf");
        let format = detect_by_extension(path).unwrap();
        assert_eq!(format, FormatType::XDF);
    }

    #[test]
    fn test_text_detection() {
        let text = b"This is ASCII text\nwith multiple lines";
        assert!(is_text_based(text));

        let binary = &[0x00, 0x01, 0xFF, 0xFE, 0xAB, 0xCD];
        assert!(!is_text_based(binary));
    }

    #[test]
    fn test_wfdb_header_detection() {
        let valid_wfdb = b"100 2 360 650000\n100.dat 212 200(0)/mV";
        assert!(is_wfdb_header(valid_wfdb));

        let invalid = b"Random text that is not WFDB";
        assert!(!is_wfdb_header(invalid));
    }

    #[test]
    fn test_edf_header_validation() {
        let mut header = vec![b' '; 256];
        header[0] = b'0'; // EDF version

        // Set valid date
        header[168..176].copy_from_slice(b"01.01.20");
        // Set valid time
        header[176..184].copy_from_slice(b"12.00.00");
        // Set number of signals
        header[252..256].copy_from_slice(b"   2");

        assert!(is_valid_edf_like_header(&header));
    }

    #[test]
    fn test_format_names() {
        assert_eq!(FormatType::EDF.name(), "European Data Format");
        assert_eq!(FormatType::BDF.name(), "BioSemi Data Format");
        assert_eq!(FormatType::GDF.name(), "General Data Format");
        assert_eq!(FormatType::XDF.name(), "Extensible Data Format");
        assert_eq!(FormatType::WFDB.name(), "WFDB Format");
    }
}
