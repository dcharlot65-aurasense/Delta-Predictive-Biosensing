//! # EDF (European Data Format) Support
//!
//! This module provides support for reading and writing European Data Format (EDF) files,
//! which are widely used for storing polysomnography, EEG, and other physiological signals.
//!
//! ## Format Overview
//!
//! EDF format consists of:
//! - **Header** (256 bytes): General recording information
//! - **Signal Headers** (256 bytes per signal): Signal-specific metadata
//! - **Data Records**: Contiguous blocks of signal samples
//!
//! EDF+ is an extension that supports:
//! - Interrupted recordings
//! - Annotations (events and timestamps)
//! - Variable record duration
//!
//! ## References
//!
//! - [EDF Specification](https://www.edfplus.info/specs/edf.html)
//! - [EDF+ Specification](https://www.edfplus.info/specs/edfplus.html)

use crate::error::{DpbError, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// EDF file header
#[derive(Debug, Clone)]
pub struct EdfHeader {
    /// Version of data format (typically "0")
    pub version: String,
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

impl EdfHeader {
    /// Create a new EDF header
    pub fn new(patient_id: String, recording_id: String, n_signals: usize) -> Self {
        Self {
            version: "0".to_string(),
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
}

/// Signal-specific metadata
#[derive(Debug, Clone)]
pub struct EdfSignal {
    /// Label (e.g., "EEG Fpz-Cz")
    pub label: String,
    /// Transducer type (e.g., "AgAgCl electrode")
    pub transducer_type: String,
    /// Physical dimension (e.g., "uV", "degreeC")
    pub physical_dimension: String,
    /// Physical minimum (e.g., -500.0 or -500 uV)
    pub physical_min: f64,
    /// Physical maximum (e.g., 500.0 or 500 uV)
    pub physical_max: f64,
    /// Digital minimum (e.g., -2048)
    pub digital_min: i16,
    /// Digital maximum (e.g., 2047)
    pub digital_max: i16,
    /// Prefiltering (e.g., "HP:0.1Hz LP:75Hz")
    pub prefiltering: String,
    /// Number of samples in each data record
    pub samples_per_record: usize,
}

impl EdfSignal {
    /// Create a new EDF signal descriptor
    pub fn new(
        label: String,
        physical_dimension: String,
        samples_per_record: usize,
    ) -> Self {
        Self {
            label,
            transducer_type: String::new(),
            physical_dimension,
            physical_min: -500.0,
            physical_max: 500.0,
            digital_min: -2048,
            digital_max: 2047,
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

    /// Set digital range
    pub fn with_digital_range(mut self, min: i16, max: i16) -> Self {
        self.digital_min = min;
        self.digital_max = max;
        self
    }

    /// Calculate sampling rate for this signal
    pub fn sample_rate(&self, record_duration: f64) -> f64 {
        self.samples_per_record as f64 / record_duration
    }

    /// Convert digital value to physical value
    pub fn digital_to_physical(&self, digital: i16) -> f64 {
        // Widen before subtracting.
        //
        // The standard EDF digital range is [-32768, 32767], whose span is
        // 65535 -- twice `i16::MAX`. Computed in `i16` this overflows: a panic
        // in debug, a wrapped negative range in release, for the single most
        // common EDF configuration there is. The same applies to
        // `digital - self.digital_min` below.
        let digital_range = self.digital_max as i32 - self.digital_min as i32;
        let physical_range = self.physical_max - self.physical_min;

        if digital_range == 0 {
            return self.physical_min;
        }

        let normalized = (digital as i32 - self.digital_min as i32) as f64 / digital_range as f64;
        self.physical_min + normalized * physical_range
    }

    /// Convert physical value to digital value
    pub fn physical_to_digital(&self, physical: f64) -> i16 {
        // Widened for the same reason as `digital_to_physical`.
        let physical_range = self.physical_max - self.physical_min;
        let digital_range = self.digital_max as i32 - self.digital_min as i32;

        if physical_range == 0.0 {
            return self.digital_min;
        }

        let normalized = (physical - self.physical_min) / physical_range;
        let digital = self.digital_min as f64 + normalized * digital_range as f64;

        digital.round().clamp(self.digital_min as f64, self.digital_max as f64) as i16
    }
}

/// EDF file reader
pub struct EdfReader {
    file: File,
    header: EdfHeader,
    signals: Vec<EdfSignal>,
    header_bytes: usize,
}

impl EdfReader {
    /// Open an EDF file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the EDF file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dpb_core::io::EdfReader;
    /// use std::path::Path;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let reader = EdfReader::open(Path::new("data/sleep.edf"))?;
    /// println!("Opened EDF with {} signals", reader.header().n_signals);
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;

        let (header, signals) = Self::read_headers(&mut file)?;
        let header_bytes = 256 + signals.len() * 256;

        Ok(Self {
            file,
            header,
            signals,
            header_bytes,
        })
    }

    /// Read EDF header and signal headers
    fn read_headers(file: &mut File) -> Result<(EdfHeader, Vec<EdfSignal>)> {
        // Read main header (256 bytes)
        let mut buffer = vec![0u8; 256];
        file.read_exact(&mut buffer)?;

        // Parse main header fields
        let version = Self::read_ascii(&buffer[0..8]);
        let patient_id = Self::read_ascii(&buffer[8..88]);
        let recording_id = Self::read_ascii(&buffer[88..168]);
        let start_date = Self::read_ascii(&buffer[168..176]);
        let start_time = Self::read_ascii(&buffer[176..184]);
        let header_bytes = Self::read_ascii(&buffer[184..192])
            .parse::<usize>()
            .map_err(|_| DpbError::DataValidation("Invalid header bytes".to_string()))?;

        // Skip reserved field (44 bytes)
        let n_records = Self::read_ascii(&buffer[236..244])
            .parse::<i32>()
            .map_err(|_| DpbError::DataValidation("Invalid record count".to_string()))?;

        let record_duration = Self::read_ascii(&buffer[244..252])
            .parse::<f64>()
            .map_err(|_| DpbError::DataValidation("Invalid record duration".to_string()))?;

        let n_signals = Self::read_ascii(&buffer[252..256])
            .parse::<usize>()
            .map_err(|_| DpbError::DataValidation("Invalid signal count".to_string()))?;

        let header = EdfHeader {
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

        // Parse signal headers in EDF order
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
            let signal = EdfSignal {
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
                    .map_err(|_| DpbError::DataValidation("Invalid samples per record".to_string()))?,
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
    pub fn header(&self) -> &EdfHeader {
        &self.header
    }

    /// Get reference to signal descriptors
    pub fn signals(&self) -> &[EdfSignal] {
        &self.signals
    }

    /// Check if this is an EDF+ file
    pub fn is_edf_plus(&self) -> bool {
        // EDF+ files have "EDF+C" or "EDF+D" in the reserved field
        // This is a simplified check
        self.header.recording_id.contains("EDF+")
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

        // Calculate bytes per record
        let bytes_per_record: usize = self
            .signals
            .iter()
            .map(|s| s.samples_per_record * 2)
            .sum();

        // Calculate offset to this signal's data within each record
        let signal_offset: usize = self.signals[0..signal_index]
            .iter()
            .map(|s| s.samples_per_record * 2)
            .sum();

        // Seek to start of data records
        self.file
            .seek(SeekFrom::Start(self.header_bytes as u64))?;

        // Read each record
        let mut record_buffer = vec![0u8; bytes_per_record];

        for _ in 0..self.header.n_records {
            self.file
                .read_exact(&mut record_buffer)?;

            // Extract this signal's samples from the record
            let signal_bytes = &record_buffer
                [signal_offset..signal_offset + signal.samples_per_record * 2];

            for chunk in signal_bytes.chunks_exact(2) {
                let digital = i16::from_le_bytes([chunk[0], chunk[1]]);
                let physical = signal.digital_to_physical(digital);
                samples.push(physical);
            }
        }

        Ok(samples)
    }

    /// Read a specific data record (all signals)
    ///
    /// # Arguments
    ///
    /// * `record_index` - Index of the record to read (0-based)
    ///
    /// # Returns
    ///
    /// Vector of signal samples, one vector per signal
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

        // Calculate bytes per record
        let bytes_per_record: usize = self
            .signals
            .iter()
            .map(|s| s.samples_per_record * 2)
            .sum();

        // Seek to the specific record
        let record_offset = self.header_bytes + record_index * bytes_per_record;
        self.file
            .seek(SeekFrom::Start(record_offset as u64))?;

        // Read the record
        let mut record_buffer = vec![0u8; bytes_per_record];
        self.file
            .read_exact(&mut record_buffer)?;

        // Parse samples for each signal
        let mut all_samples = Vec::with_capacity(self.header.n_signals);
        let mut offset = 0;

        for signal in &self.signals {
            let mut signal_samples = Vec::with_capacity(signal.samples_per_record);
            let signal_bytes = &record_buffer[offset..offset + signal.samples_per_record * 2];

            for chunk in signal_bytes.chunks_exact(2) {
                let digital = i16::from_le_bytes([chunk[0], chunk[1]]);
                let physical = signal.digital_to_physical(digital);
                signal_samples.push(physical);
            }

            all_samples.push(signal_samples);
            offset += signal.samples_per_record * 2;
        }

        Ok(all_samples)
    }
}

/// EDF file writer
pub struct EdfWriter {
    file: File,
    header: EdfHeader,
    signals: Vec<EdfSignal>,
    records_written: usize,
}

impl EdfWriter {
    /// Create a new EDF writer
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the EDF file to create
    /// * `header` - EDF header
    /// * `signals` - Signal descriptors
    pub fn new(path: &Path, header: EdfHeader, signals: Vec<EdfSignal>) -> Result<Self> {
        if signals.len() != header.n_signals {
            return Err(DpbError::InvalidParameter(format!(
                "Signal count mismatch: header specifies {}, got {}",
                header.n_signals,
                signals.len()
            )));
        }

        let file = File::create(path)?;

        let mut writer = Self {
            file,
            header,
            signals,
            records_written: 0,
        };

        // Write the headers up front so the sample data that follows starts
        // after them.
        //
        // Without this the first `write_record` began at offset 0, and
        // `finish` -- which seeks to 0 to rewrite the header with the final
        // record count -- then overwrote the first 256 + 256*n bytes of SAMPLE
        // data with it. Every file was both truncated at the front and
        // unreadable, since a reader looking for a header found signal values.
        // `finish` still rewrites this in place once the record count is known.
        writer.write_headers()?;

        Ok(writer)
    }

    /// Write EDF headers
    fn write_headers(&mut self) -> Result<()> {
        // Calculate header size
        let header_bytes = 256 + self.header.n_signals * 256;

        // Write main header (256 bytes)
        let mut buffer = vec![b' '; 256];
        Self::write_field(&mut buffer, 0, 8, &self.header.version);
        Self::write_field(&mut buffer, 8, 80, &self.header.patient_id);
        Self::write_field(&mut buffer, 88, 80, &self.header.recording_id);
        Self::write_field(&mut buffer, 168, 8, &self.header.start_date);
        Self::write_field(&mut buffer, 176, 8, &self.header.start_time);
        Self::write_field(&mut buffer, 184, 8, &header_bytes.to_string());
        // Reserved field at 192 (44 bytes)
        Self::write_field(&mut buffer, 236, 8, &self.header.n_records.to_string());
        Self::write_field(&mut buffer, 244, 8, &self.header.record_duration.to_string());
        Self::write_field(&mut buffer, 252, 4, &self.header.n_signals.to_string());

        self.file
            .write_all(&buffer)?;

        // Write signal headers
        let signal_header_size = self.header.n_signals * 256;
        let mut signal_buffer = vec![b' '; signal_header_size];

        // Write each field for all signals
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
        // Reserved field at 208 (32 bytes per signal)

        self.file
            .write_all(&signal_buffer)?;

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
        signals: &[EdfSignal],
        accessor: F,
    ) where
        F: Fn(&EdfSignal) -> String,
    {
        for (i, signal) in signals.iter().enumerate() {
            let start = field_offset * signals.len() + i * field_size;
            let value = accessor(signal);
            Self::write_field(buffer, start, field_size, &value);
        }
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

        // Verify sample counts and convert to digital values
        for (i, samples) in record.iter().enumerate() {
            if samples.len() != self.signals[i].samples_per_record {
                return Err(DpbError::InvalidParameter(format!(
                    "Signal {} must have {} samples, got {}",
                    i,
                    self.signals[i].samples_per_record,
                    samples.len()
                )));
            }

            // Write samples for this signal
            for &sample in samples {
                let digital = self.signals[i].physical_to_digital(sample);
                self.file
                    .write_all(&digital.to_le_bytes())?;
            }
        }

        self.records_written += 1;
        Ok(())
    }

    /// Finalize the EDF file
    pub fn finish(mut self) -> Result<()> {
        // Update header with actual record count
        self.header.n_records = self.records_written as i32;

        // Seek to beginning and rewrite header
        self.file
            .seek(SeekFrom::Start(0))?;

        self.write_headers()?;

        self.file
            .flush()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_conversion() {
        let signal = EdfSignal::new("EEG".to_string(), "uV".to_string(), 100)
            .with_physical_range(-500.0, 500.0)
            .with_digital_range(-2048, 2047);

        // Test digital to physical
        // Note: digital range -2048 to 2047 is not symmetric, so 0 doesn't map exactly to 0.0
        let physical = signal.digital_to_physical(0);
        assert!((physical - 0.0).abs() < 1.0);

        let physical = signal.digital_to_physical(2047);
        assert!((physical - 500.0).abs() < 1.0);

        let physical = signal.digital_to_physical(-2048);
        assert!((physical - (-500.0)).abs() < 1.0);

        // Test physical to digital
        let digital = signal.physical_to_digital(0.0);
        assert!((digital - 0).abs() <= 1);

        let digital = signal.physical_to_digital(500.0);
        assert_eq!(digital, 2047);

        let digital = signal.physical_to_digital(-500.0);
        assert_eq!(digital, -2048);
    }

    #[test]
    fn test_sample_rate() {
        let signal = EdfSignal::new("EEG".to_string(), "uV".to_string(), 256);
        let sample_rate = signal.sample_rate(1.0);
        assert_eq!(sample_rate, 256.0);

        let sample_rate = signal.sample_rate(2.0);
        assert_eq!(sample_rate, 128.0);
    }

    #[test]
    fn test_header_builder() {
        let header = EdfHeader::new(
            "Patient X".to_string(),
            "Recording Y".to_string(),
            2,
        )
        .with_start_datetime("01.01.20".to_string(), "12.00.00".to_string())
        .with_records(100, 1.0);

        assert_eq!(header.patient_id, "Patient X");
        assert_eq!(header.n_signals, 2);
        assert_eq!(header.n_records, 100);
        assert_eq!(header.record_duration, 1.0);
    }
    /// The standard EDF digital range must not overflow the conversion.
    ///
    /// Regression: `digital_max - digital_min` was computed in `i16`, and the
    /// standard range [-32768, 32767] spans 65535 -- twice `i16::MAX`. Every
    /// conversion panicked in debug and wrapped in release for the most common
    /// configuration the format has.
    #[test]
    fn test_edf_full_digital_range_round_trip() {
        let signal = EdfSignal::new("ECG".to_string(), "mV".to_string(), 256)
            .with_physical_range(-5.0, 5.0)
            .with_digital_range(i16::MIN, i16::MAX);

        for physical in [-5.0, -2.5, 0.0, 1.234, 4.999] {
            let digital = signal.physical_to_digital(physical);
            let back = signal.digital_to_physical(digital);
            assert!(
                (physical - back).abs() < 0.001,
                "{physical} -> {digital} -> {back}"
            );
        }

        // The extremes must map to the extremes, not wrap around.
        assert_eq!(signal.physical_to_digital(-5.0), i16::MIN);
        assert_eq!(signal.physical_to_digital(5.0), i16::MAX);
    }

}
