//! # BDF (BioSemi Data Format) Support
//!
//! This module provides support for reading and writing BioSemi Data Format (BDF) files,
//! which is a 24-bit variant of the EDF format used by BioSemi active electrode systems.
//!
//! ## Format Overview
//!
//! BDF format features:
//! - **Header** (256 + 256*n_signals bytes): Similar to EDF
//! - **24-bit samples**: Higher resolution than EDF's 16-bit
//! - **Status channel**: Contains trigger and sensor status information
//! - **Data Records**: Contiguous blocks of 24-bit samples
//!
//! ## Key Differences from EDF
//!
//! - Uses 24-bit signed integers instead of 16-bit
//! - Version field starts with 0xFF (255) instead of "0"
//! - Status channel contains trigger codes and CMS/DRL information
//!
//! ## References
//!
//! - [BioSemi Data Format](https://www.biosemi.com/faq/file_format.htm)

use crate::error::{DpbError, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// BDF file header
#[derive(Debug, Clone)]
pub struct BdfHeader {
    /// Version identifier (should be 255/0xFF for BDF or "BIOSEMI" string)
    pub version: u8,
    /// Local patient identification
    pub patient_id: String,
    /// Local recording identification
    pub recording_id: String,
    /// Start date of recording (dd.mm.yy format)
    pub start_date: String,
    /// Start time of recording (hh.mm.ss format)
    pub start_time: String,
    /// Number of data records (-1 if unknown)
    pub n_records: i32,
    /// Duration of a data record in seconds
    pub record_duration: f64,
    /// Number of signals in data record
    pub n_signals: usize,
}

impl BdfHeader {
    /// Create a new BDF header
    pub fn new(patient_id: String, recording_id: String, n_signals: usize) -> Self {
        Self {
            version: 255, // BDF identifier
            patient_id,
            recording_id,
            start_date: "01.01.00".to_string(),
            start_time: "00.00.00".to_string(),
            n_records: -1,
            record_duration: 1.0,
            n_signals,
        }
    }

    /// Set start date and time
    pub fn with_start_datetime(mut self, date: String, time: String) -> Self {
        self.start_date = date;
        self.start_time = time;
        self
    }

    /// Set record parameters
    pub fn with_records(mut self, n_records: i32, duration: f64) -> Self {
        self.n_records = n_records;
        self.record_duration = duration;
        self
    }

    /// Check if this is a valid BDF header
    pub fn is_bdf(&self) -> bool {
        self.version == 255
    }
}

/// BDF signal descriptor
#[derive(Debug, Clone)]
pub struct BdfSignal {
    /// Label (e.g., "EEG Fpz-Cz" or "Status")
    pub label: String,
    /// Transducer type (e.g., "Active Electrode")
    pub transducer_type: String,
    /// Physical dimension (e.g., "uV", "Boolean")
    pub physical_dimension: String,
    /// Physical minimum (e.g., -262144.0)
    pub physical_min: f64,
    /// Physical maximum (e.g., 262143.0)
    pub physical_max: f64,
    /// Digital minimum (e.g., -8388608)
    pub digital_min: i32,
    /// Digital maximum (e.g., 8388607)
    pub digital_max: i32,
    /// Prefiltering (e.g., "HP:DC LP:410Hz")
    pub prefiltering: String,
    /// Number of samples in each data record
    pub samples_per_record: usize,
}

impl BdfSignal {
    /// Create a new BDF signal descriptor
    pub fn new(
        label: String,
        physical_dimension: String,
        samples_per_record: usize,
    ) -> Self {
        // Default ranges for 24-bit data
        let digital_min = -8388608; // -2^23
        let digital_max = 8388607; // 2^23 - 1

        Self {
            label,
            transducer_type: String::new(),
            physical_dimension,
            physical_min: -262144.0,
            physical_max: 262143.0,
            digital_min,
            digital_max,
            prefiltering: String::new(),
            samples_per_record,
        }
    }

    /// Create a status channel descriptor
    pub fn status_channel(samples_per_record: usize) -> Self {
        Self {
            label: "Status".to_string(),
            transducer_type: String::new(),
            physical_dimension: "Boolean".to_string(),
            physical_min: -8388608.0,
            physical_max: 8388607.0,
            digital_min: -8388608,
            digital_max: 8388607,
            prefiltering: String::new(),
            samples_per_record,
        }
    }

    /// Set physical range
    pub fn with_physical_range(mut self, min: f64, max: f64) -> Self {
        self.physical_min = min;
        self.physical_max = max;
        self
    }

    /// Set digital range (usually 24-bit range)
    pub fn with_digital_range(mut self, min: i32, max: i32) -> Self {
        self.digital_min = min;
        self.digital_max = max;
        self
    }

    /// Calculate sampling rate for this signal
    pub fn sample_rate(&self, record_duration: f64) -> f64 {
        self.samples_per_record as f64 / record_duration
    }

    /// Convert 24-bit digital value to physical value
    pub fn digital_to_physical(&self, digital: i32) -> f64 {
        let digital_range = self.digital_max - self.digital_min;
        let physical_range = self.physical_max - self.physical_min;

        if digital_range == 0 {
            return self.physical_min;
        }

        let normalized = (digital - self.digital_min) as f64 / digital_range as f64;
        self.physical_min + normalized * physical_range
    }

    /// Convert physical value to 24-bit digital value
    pub fn physical_to_digital(&self, physical: f64) -> i32 {
        let physical_range = self.physical_max - self.physical_min;
        let digital_range = self.digital_max - self.digital_min;

        if physical_range == 0.0 {
            return self.digital_min;
        }

        let normalized = (physical - self.physical_min) / physical_range;
        let digital = self.digital_min as f64 + normalized * digital_range as f64;

        digital
            .round()
            .clamp(self.digital_min as f64, self.digital_max as f64) as i32
    }

    /// Check if this is a status channel
    pub fn is_status_channel(&self) -> bool {
        self.label.to_lowercase().contains("status")
    }
}

/// BDF trigger information extracted from status channel
#[derive(Debug, Clone)]
pub struct BdfTrigger {
    /// Sample index where trigger occurred
    pub sample: usize,
    /// Trigger code (bits 0-15 of status word)
    pub code: u16,
    /// CMS in range flag
    pub cms_in_range: bool,
    /// Battery low flag
    pub battery_low: bool,
}

impl BdfTrigger {
    /// Extract trigger from 24-bit status value
    pub fn from_status_value(sample: usize, status: i32) -> Self {
        let trigger_code = (status & 0xFFFF) as u16;
        let cms_in_range = (status & 0x10000) != 0;
        let battery_low = (status & 0x800000) != 0;

        Self {
            sample,
            code: trigger_code,
            cms_in_range,
            battery_low,
        }
    }

    /// Check if this is a valid trigger (non-zero code)
    pub fn is_valid(&self) -> bool {
        self.code != 0
    }
}

/// BDF file reader
pub struct BdfReader {
    file: File,
    header: BdfHeader,
    signals: Vec<BdfSignal>,
    header_bytes: usize,
    status_channel_index: Option<usize>,
}

impl BdfReader {
    /// Open a BDF file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the BDF file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dpb_core::io::BdfReader;
    /// use std::path::Path;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let reader = BdfReader::open(Path::new("data/biosemi.bdf"))?;
    /// println!("Opened BDF with {} signals", reader.header().n_signals);
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;

        let (header, signals) = Self::read_headers(&mut file)?;

        if !header.is_bdf() {
            return Err(DpbError::DataValidation(
                "Not a valid BDF file (version byte != 255)".to_string(),
            ));
        }

        let header_bytes = 256 + signals.len() * 256;

        // Find status channel
        let status_channel_index = signals
            .iter()
            .position(|s| s.is_status_channel());

        Ok(Self {
            file,
            header,
            signals,
            header_bytes,
            status_channel_index,
        })
    }

    /// Read BDF header and signal headers
    fn read_headers(file: &mut File) -> Result<(BdfHeader, Vec<BdfSignal>)> {
        // Read main header (256 bytes)
        let mut buffer = vec![0u8; 256];
        file.read_exact(&mut buffer)?;

        // Parse main header fields
        let version = buffer[0];
        let patient_id = Self::read_ascii(&buffer[8..88]);
        let recording_id = Self::read_ascii(&buffer[88..168]);
        let start_date = Self::read_ascii(&buffer[168..176]);
        let start_time = Self::read_ascii(&buffer[176..184]);
        let header_bytes = Self::read_ascii(&buffer[184..192])
            .parse::<usize>()
            .map_err(|_| DpbError::DataValidation("Invalid header bytes".to_string()))?;

        let n_records = Self::read_ascii(&buffer[236..244])
            .parse::<i32>()
            .map_err(|_| DpbError::DataValidation("Invalid record count".to_string()))?;

        let record_duration = Self::read_ascii(&buffer[244..252])
            .parse::<f64>()
            .map_err(|_| DpbError::DataValidation("Invalid record duration".to_string()))?;

        let n_signals = Self::read_ascii(&buffer[252..256])
            .parse::<usize>()
            .map_err(|_| DpbError::DataValidation("Invalid signal count".to_string()))?;

        // The declared header size must match the fixed layout: 256 bytes
        // plus 256 per signal. A mismatch means the file is malformed or
        // truncated, and every subsequent offset would be wrong.
        let expected_header_bytes = 256 + 256 * n_signals;
        if header_bytes != expected_header_bytes {
            return Err(DpbError::DataValidation(format!(
                "Header declares {header_bytes} bytes but {n_signals} signals require {expected_header_bytes}"
            )));
        }

        let header = BdfHeader {
            version,
            patient_id,
            recording_id,
            start_date,
            start_time,
            n_records,
            record_duration,
            n_signals,
        };

        // Read signal headers (256 bytes per signal)
        let signal_header_bytes = n_signals * 256;
        let mut signal_buffer = vec![0u8; signal_header_bytes];
        file.read_exact(&mut signal_buffer)?;

        let mut signals = Vec::with_capacity(n_signals);

        // Parse signal headers in BDF order (same as EDF)
        let labels = Self::read_signal_field(&signal_buffer, 0, 16, n_signals);
        let transducer_types = Self::read_signal_field(&signal_buffer, 16, 80, n_signals);
        let dimensions = Self::read_signal_field(&signal_buffer, 80, 8, n_signals);
        let physical_mins = Self::read_signal_field(&signal_buffer, 88, 8, n_signals);
        let physical_maxs = Self::read_signal_field(&signal_buffer, 96, 8, n_signals);
        let digital_mins = Self::read_signal_field(&signal_buffer, 104, 8, n_signals);
        let digital_maxs = Self::read_signal_field(&signal_buffer, 112, 8, n_signals);
        let prefiltering = Self::read_signal_field(&signal_buffer, 120, 80, n_signals);
        let samples_per_record = Self::read_signal_field(&signal_buffer, 200, 8, n_signals);

        for i in 0..n_signals {
            let signal = BdfSignal {
                label: labels[i].clone(),
                transducer_type: transducer_types[i].clone(),
                physical_dimension: dimensions[i].clone(),
                physical_min: physical_mins[i]
                    .parse()
                    .map_err(|_| DpbError::DataValidation("Invalid physical min".to_string()))?,
                physical_max: physical_maxs[i]
                    .parse()
                    .map_err(|_| DpbError::DataValidation("Invalid physical max".to_string()))?,
                digital_min: digital_mins[i]
                    .parse()
                    .map_err(|_| DpbError::DataValidation("Invalid digital min".to_string()))?,
                digital_max: digital_maxs[i]
                    .parse()
                    .map_err(|_| DpbError::DataValidation("Invalid digital max".to_string()))?,
                prefiltering: prefiltering[i].clone(),
                samples_per_record: samples_per_record[i]
                    .parse()
                    .map_err(|_| {
                        DpbError::DataValidation("Invalid samples per record".to_string())
                    })?,
            };
            signals.push(signal);
        }

        Ok((header, signals))
    }

    /// Read ASCII field from buffer and trim whitespace
    fn read_ascii(buffer: &[u8]) -> String {
        String::from_utf8_lossy(buffer).trim().to_string()
    }

    /// Read signal field values for all signals
    fn read_signal_field(
        buffer: &[u8],
        field_offset: usize,
        field_size: usize,
        n_signals: usize,
    ) -> Vec<String> {
        let mut values = Vec::with_capacity(n_signals);
        for i in 0..n_signals {
            let start = field_offset * n_signals + i * field_size;
            let end = start + field_size;
            values.push(Self::read_ascii(&buffer[start..end]));
        }
        values
    }

    /// Get reference to the header
    pub fn header(&self) -> &BdfHeader {
        &self.header
    }

    /// Get reference to signal descriptors
    pub fn signals(&self) -> &[BdfSignal] {
        &self.signals
    }

    /// Get status channel index if present
    pub fn status_channel_index(&self) -> Option<usize> {
        self.status_channel_index
    }

    /// Read 24-bit signed integer from 3 bytes
    fn read_int24(bytes: &[u8]) -> i32 {
        let mut buffer = [0u8; 4];
        buffer[0..3].copy_from_slice(&bytes[0..3]);

        // Sign extend if negative (bit 23 is set)
        if bytes[2] & 0x80 != 0 {
            buffer[3] = 0xFF;
        }

        i32::from_le_bytes(buffer)
    }

    /// Read all samples for a specific signal
    ///
    /// # Arguments
    ///
    /// * `signal_index` - Index of the signal to read (0-based)
    pub fn read_signal(&mut self, signal_index: usize) -> Result<Vec<f64>> {
        if signal_index >= self.header.n_signals {
            return Err(DpbError::InvalidParameter(format!(
                "Signal index {} out of range (0-{})",
                signal_index,
                self.header.n_signals - 1
            )));
        }

        if self.header.n_records < 0 {
            return Err(DpbError::Other(
                "Unknown number of records".to_string(),
            ));
        }

        let signal = &self.signals[signal_index];
        let total_samples = signal.samples_per_record * self.header.n_records as usize;
        let mut samples = Vec::with_capacity(total_samples);

        // Calculate bytes per record (24-bit = 3 bytes per sample)
        let bytes_per_record: usize = self
            .signals
            .iter()
            .map(|s| s.samples_per_record * 3)
            .sum();

        // Calculate offset to this signal's data within each record
        let signal_offset: usize = self.signals[0..signal_index]
            .iter()
            .map(|s| s.samples_per_record * 3)
            .sum();

        // Seek to start of data records
        self.file
            .seek(SeekFrom::Start(self.header_bytes as u64))?;

        // Read each record
        let mut record_buffer = vec![0u8; bytes_per_record];

        for _ in 0..self.header.n_records {
            self.file.read_exact(&mut record_buffer)?;

            // Extract this signal's samples from the record
            let signal_bytes = &record_buffer
                [signal_offset..signal_offset + signal.samples_per_record * 3];

            for chunk in signal_bytes.as_chunks::<3>().0 {
                let digital = Self::read_int24(chunk);
                let physical = signal.digital_to_physical(digital);
                samples.push(physical);
            }
        }

        Ok(samples)
    }

    /// Read status channel and extract triggers
    pub fn read_triggers(&mut self) -> Result<Vec<BdfTrigger>> {
        let status_index = self.status_channel_index.ok_or_else(|| {
            DpbError::Other("No status channel found in BDF file".to_string())
        })?;

        if self.header.n_records < 0 {
            return Err(DpbError::Other(
                "Unknown number of records".to_string(),
            ));
        }

        let signal = &self.signals[status_index];
        let mut triggers = Vec::new();

        let bytes_per_record: usize = self
            .signals
            .iter()
            .map(|s| s.samples_per_record * 3)
            .sum();

        let signal_offset: usize = self.signals[0..status_index]
            .iter()
            .map(|s| s.samples_per_record * 3)
            .sum();

        self.file
            .seek(SeekFrom::Start(self.header_bytes as u64))?;

        let mut record_buffer = vec![0u8; bytes_per_record];
        let mut sample_counter = 0;

        for _ in 0..self.header.n_records {
            self.file.read_exact(&mut record_buffer)?;

            let signal_bytes = &record_buffer
                [signal_offset..signal_offset + signal.samples_per_record * 3];

            for chunk in signal_bytes.as_chunks::<3>().0 {
                let status_value = Self::read_int24(chunk);
                let trigger = BdfTrigger::from_status_value(sample_counter, status_value);

                if trigger.is_valid() {
                    triggers.push(trigger);
                }

                sample_counter += 1;
            }
        }

        Ok(triggers)
    }

    /// Read a specific data record (all signals)
    pub fn read_record(&mut self, record_index: usize) -> Result<Vec<Vec<f64>>> {
        if self.header.n_records < 0 {
            return Err(DpbError::Other(
                "Unknown number of records".to_string(),
            ));
        }

        if record_index >= self.header.n_records as usize {
            return Err(DpbError::InvalidParameter(format!(
                "Record index {} out of range (0-{})",
                record_index,
                self.header.n_records - 1
            )));
        }

        let bytes_per_record: usize = self
            .signals
            .iter()
            .map(|s| s.samples_per_record * 3)
            .sum();

        let record_offset = self.header_bytes + record_index * bytes_per_record;
        self.file
            .seek(SeekFrom::Start(record_offset as u64))?;

        let mut record_buffer = vec![0u8; bytes_per_record];
        self.file.read_exact(&mut record_buffer)?;

        let mut all_samples = Vec::with_capacity(self.header.n_signals);
        let mut offset = 0;

        for signal in &self.signals {
            let mut signal_samples = Vec::with_capacity(signal.samples_per_record);
            let signal_bytes = &record_buffer[offset..offset + signal.samples_per_record * 3];

            for chunk in signal_bytes.as_chunks::<3>().0 {
                let digital = Self::read_int24(chunk);
                let physical = signal.digital_to_physical(digital);
                signal_samples.push(physical);
            }

            all_samples.push(signal_samples);
            offset += signal.samples_per_record * 3;
        }

        Ok(all_samples)
    }
}

/// BDF file writer
pub struct BdfWriter {
    file: File,
    header: BdfHeader,
    signals: Vec<BdfSignal>,
    records_written: usize,
}

impl BdfWriter {
    /// Create a new BDF writer
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the BDF file to create
    /// * `header` - BDF header
    /// * `signals` - Signal descriptors
    pub fn new(path: &Path, header: BdfHeader, signals: Vec<BdfSignal>) -> Result<Self> {
        if signals.len() != header.n_signals {
            return Err(DpbError::InvalidParameter(format!(
                "Signal count mismatch: header specifies {}, got {}",
                header.n_signals,
                signals.len()
            )));
        }

        let file = File::create(path)?;

        Ok(Self {
            file,
            header,
            signals,
            records_written: 0,
        })
    }

    /// Write BDF headers
    fn write_headers(&mut self) -> Result<()> {
        let header_bytes = 256 + self.header.n_signals * 256;

        // Write main header (256 bytes)
        let mut buffer = vec![b' '; 256];

        // Version byte (255 for BDF)
        buffer[0] = self.header.version;

        Self::write_field(&mut buffer, 8, 80, &self.header.patient_id);
        Self::write_field(&mut buffer, 88, 80, &self.header.recording_id);
        Self::write_field(&mut buffer, 168, 8, &self.header.start_date);
        Self::write_field(&mut buffer, 176, 8, &self.header.start_time);
        Self::write_field(&mut buffer, 184, 8, &header_bytes.to_string());
        Self::write_field(&mut buffer, 236, 8, &self.header.n_records.to_string());
        Self::write_field(&mut buffer, 244, 8, &self.header.record_duration.to_string());
        Self::write_field(&mut buffer, 252, 4, &self.header.n_signals.to_string());

        self.file.write_all(&buffer)?;

        // Write signal headers
        let signal_header_size = self.header.n_signals * 256;
        let mut signal_buffer = vec![b' '; signal_header_size];

        Self::write_signal_field(&mut signal_buffer, 0, 16, &self.signals, |s| s.label.clone());
        Self::write_signal_field(&mut signal_buffer, 16, 80, &self.signals, |s| {
            s.transducer_type.clone()
        });
        Self::write_signal_field(&mut signal_buffer, 80, 8, &self.signals, |s| {
            s.physical_dimension.clone()
        });
        Self::write_signal_field(&mut signal_buffer, 88, 8, &self.signals, |s| {
            s.physical_min.to_string()
        });
        Self::write_signal_field(&mut signal_buffer, 96, 8, &self.signals, |s| {
            s.physical_max.to_string()
        });
        Self::write_signal_field(&mut signal_buffer, 104, 8, &self.signals, |s| {
            s.digital_min.to_string()
        });
        Self::write_signal_field(&mut signal_buffer, 112, 8, &self.signals, |s| {
            s.digital_max.to_string()
        });
        Self::write_signal_field(&mut signal_buffer, 120, 80, &self.signals, |s| {
            s.prefiltering.clone()
        });
        Self::write_signal_field(&mut signal_buffer, 200, 8, &self.signals, |s| {
            s.samples_per_record.to_string()
        });

        self.file.write_all(&signal_buffer)?;

        Ok(())
    }

    /// Write ASCII field to buffer
    fn write_field(buffer: &mut [u8], offset: usize, size: usize, value: &str) {
        let bytes = value.as_bytes();
        let len = bytes.len().min(size);
        buffer[offset..offset + len].copy_from_slice(&bytes[..len]);
    }

    /// Write signal field for all signals
    fn write_signal_field<F>(
        buffer: &mut [u8],
        field_offset: usize,
        field_size: usize,
        signals: &[BdfSignal],
        accessor: F,
    ) where
        F: Fn(&BdfSignal) -> String,
    {
        for (i, signal) in signals.iter().enumerate() {
            let start = field_offset * signals.len() + i * field_size;
            let value = accessor(signal);
            Self::write_field(buffer, start, field_size, &value);
        }
    }

    /// Write 24-bit signed integer to 3 bytes
    fn write_int24(value: i32) -> [u8; 3] {
        let bytes = value.to_le_bytes();
        [bytes[0], bytes[1], bytes[2]]
    }

    /// Write a data record
    ///
    /// # Arguments
    ///
    /// * `record` - Vector of signal samples, one vector per signal
    pub fn write_record(&mut self, record: &[Vec<f64>]) -> Result<()> {
        if record.len() != self.header.n_signals {
            return Err(DpbError::InvalidParameter(format!(
                "Record must have {} signals, got {}",
                self.header.n_signals,
                record.len()
            )));
        }

        for (i, samples) in record.iter().enumerate() {
            if samples.len() != self.signals[i].samples_per_record {
                return Err(DpbError::InvalidParameter(format!(
                    "Signal {} must have {} samples, got {}",
                    i,
                    self.signals[i].samples_per_record,
                    samples.len()
                )));
            }

            for &sample in samples {
                let digital = self.signals[i].physical_to_digital(sample);
                let bytes = Self::write_int24(digital);
                self.file.write_all(&bytes)?;
            }
        }

        self.records_written += 1;
        Ok(())
    }

    /// Finalize the BDF file
    pub fn finish(mut self) -> Result<()> {
        self.header.n_records = self.records_written as i32;

        self.file.seek(SeekFrom::Start(0))?;
        self.write_headers()?;

        self.file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int24_conversion() {
        // Test positive value
        let bytes = [0x01, 0x02, 0x03];
        let value = BdfReader::read_int24(&bytes);
        assert_eq!(value, 0x030201);

        // Test negative value (sign bit set)
        let bytes = [0xFF, 0xFF, 0xFF];
        let value = BdfReader::read_int24(&bytes);
        assert_eq!(value, -1);

        // Test zero
        let bytes = [0x00, 0x00, 0x00];
        let value = BdfReader::read_int24(&bytes);
        assert_eq!(value, 0);

        // Test write
        let bytes = BdfWriter::write_int24(0x030201);
        assert_eq!(bytes, [0x01, 0x02, 0x03]);

        let bytes = BdfWriter::write_int24(-1);
        assert_eq!(bytes, [0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_signal_conversion() {
        let signal = BdfSignal::new("EEG".to_string(), "uV".to_string(), 100)
            .with_physical_range(-262144.0, 262143.0)
            .with_digital_range(-8388608, 8388607);

        let physical = signal.digital_to_physical(0);
        assert!((physical - 0.0).abs() < 1.0);

        let physical = signal.digital_to_physical(8388607);
        assert!((physical - 262143.0).abs() < 1.0);
    }

    #[test]
    fn test_trigger_extraction() {
        // Status value with trigger code 15
        let trigger = BdfTrigger::from_status_value(100, 0x0F);
        assert_eq!(trigger.sample, 100);
        assert_eq!(trigger.code, 15);
        assert!(trigger.is_valid());

        // Status value with no trigger
        let trigger = BdfTrigger::from_status_value(200, 0x00);
        assert_eq!(trigger.code, 0);
        assert!(!trigger.is_valid());

        // Status with CMS flag
        let trigger = BdfTrigger::from_status_value(300, 0x10001);
        assert_eq!(trigger.code, 1);
        assert!(trigger.cms_in_range);
    }

    #[test]
    fn test_header_builder() {
        let header = BdfHeader::new(
            "Patient X".to_string(),
            "Recording Y".to_string(),
            4,
        )
        .with_start_datetime("01.01.20".to_string(), "12.00.00".to_string())
        .with_records(100, 1.0);

        assert_eq!(header.version, 255);
        assert!(header.is_bdf());
        assert_eq!(header.n_signals, 4);
    }

    #[test]
    fn test_status_channel() {
        let status = BdfSignal::status_channel(256);
        assert_eq!(status.label, "Status");
        assert!(status.is_status_channel());
        assert_eq!(status.physical_dimension, "Boolean");
    }
}
