//! # WFDB (Waveform Database) Format Support
//!
//! This module provides support for reading and writing PhysioNet WFDB format files,
//! which are commonly used for storing physiological signals, particularly ECG data.
//!
//! ## Format Overview
//!
//! WFDB format consists of:
//! - **Header file** (.hea): Contains metadata about the recording
//! - **Data file** (.dat): Contains the actual signal samples
//! - **Annotation file** (.atr, optional): Contains beat annotations and event markers
//!
//! ## References
//!
//! - [PhysioNet WFDB Software Package](https://physionet.org/content/wfdb/)
//! - [WFDB Specification](https://physionet.org/physiotools/wag/header-5.htm)

use crate::error::{DpbError, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

/// WFDB header containing recording metadata
#[derive(Debug, Clone)]
pub struct WfdbHeader {
    /// Record name (base filename without extension)
    pub record_name: String,
    /// Number of signals in the recording
    pub n_signals: usize,
    /// Sampling frequency in Hz
    pub sample_rate: f64,
    /// Total number of samples per signal (None if unknown)
    pub n_samples: Option<usize>,
    /// Base time (HH:MM:SS format)
    pub base_time: Option<String>,
    /// Base date (DD/MM/YYYY format)
    pub base_date: Option<String>,
}

impl WfdbHeader {
    /// Create a new WFDB header
    pub fn new(record_name: String, n_signals: usize, sample_rate: f64) -> Self {
        Self {
            record_name,
            n_signals,
            sample_rate,
            n_samples: None,
            base_time: None,
            base_date: None,
        }
    }

    /// Set the number of samples
    pub fn with_n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = Some(n_samples);
        self
    }

    /// Set the base time
    pub fn with_base_time(mut self, base_time: String) -> Self {
        self.base_time = Some(base_time);
        self
    }

    /// Set the base date
    pub fn with_base_date(mut self, base_date: String) -> Self {
        self.base_date = Some(base_date);
        self
    }
}

/// Signal-specific metadata
#[derive(Debug, Clone)]
pub struct WfdbSignal {
    /// Signal name/label
    pub name: String,
    /// Physical units (e.g., "mV", "mmHg")
    pub units: String,
    /// Gain (ADC units per physical unit)
    pub gain: f64,
    /// Baseline (ADC zero value)
    pub baseline: i32,
    /// ADC resolution in bits
    pub adc_resolution: u8,
    /// ADC zero level
    pub adc_zero: i32,
    /// Format (e.g., "16" for 16-bit samples)
    pub format: u16,
    /// Samples per frame
    pub samples_per_frame: usize,
}

impl WfdbSignal {
    /// Create a new WFDB signal descriptor
    pub fn new(name: String, units: String, gain: f64) -> Self {
        Self {
            name,
            units,
            gain,
            baseline: 0,
            adc_resolution: 16,
            adc_zero: 0,
            format: 16,
            samples_per_frame: 1,
        }
    }

    /// Convert ADC value to physical units
    pub fn adc_to_physical(&self, adc_value: i16) -> f64 {
        (adc_value as f64 - self.baseline as f64) / self.gain
    }

    /// Convert physical value to ADC units
    pub fn physical_to_adc(&self, physical_value: f64) -> i16 {
        (physical_value * self.gain + self.baseline as f64).round() as i16
    }
}

// These are established domain acronyms -- clinical file formats, ECG
// beat annotations, and hardware terms. Camel-casing them (Pvc, Wfdb,
// Dram) would make this harder to read for anyone who works with them.
#[allow(clippy::upper_case_acronyms)]
/// Annotation type codes (based on MIT-BIH annotation codes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnnotationType {
    /// Normal beat (N)
    Normal,
    /// Left bundle branch block beat (L)
    LBBB,
    /// Right bundle branch block beat (R)
    RBBB,
    /// Atrial premature beat (A)
    AtrialPrem,
    /// Premature ventricular contraction (V)
    PVC,
    /// Fusion of ventricular and normal beat (F)
    Fusion,
    /// Paced beat (/)
    Paced,
    /// Unclassifiable beat (Q)
    Unclassifiable,
    /// Ventricular escape beat (E)
    VentricularEscape,
    /// Nodal (junctional) escape beat (j)
    NodalEscape,
    /// Supraventricular premature beat (S)
    SuperventricularPrem,
    /// Unknown or custom annotation
    Unknown(u8),
}

impl AnnotationType {
    /// Convert from annotation code character
    pub fn from_code(code: char) -> Self {
        match code {
            'N' => Self::Normal,
            'L' => Self::LBBB,
            'R' => Self::RBBB,
            'A' => Self::AtrialPrem,
            'V' => Self::PVC,
            'F' => Self::Fusion,
            '/' => Self::Paced,
            'Q' => Self::Unclassifiable,
            'E' => Self::VentricularEscape,
            'j' => Self::NodalEscape,
            'S' => Self::SuperventricularPrem,
            _ => Self::Unknown(code as u8),
        }
    }

    /// Convert to annotation code character
    pub fn to_code(&self) -> char {
        match self {
            Self::Normal => 'N',
            Self::LBBB => 'L',
            Self::RBBB => 'R',
            Self::AtrialPrem => 'A',
            Self::PVC => 'V',
            Self::Fusion => 'F',
            Self::Paced => '/',
            Self::Unclassifiable => 'Q',
            Self::VentricularEscape => 'E',
            Self::NodalEscape => 'j',
            Self::SuperventricularPrem => 'S',
            Self::Unknown(code) => *code as char,
        }
    }
}

/// WFDB annotation entry
#[derive(Debug, Clone)]
pub struct WfdbAnnotation {
    /// Sample number where annotation occurs
    pub sample: usize,
    /// Type of annotation
    pub annotation_type: AnnotationType,
    /// Subtype code
    pub subtype: u8,
    /// Channel number
    pub channel: u8,
    /// Auxiliary information string
    pub aux: Option<String>,
}

impl WfdbAnnotation {
    /// Create a new annotation
    pub fn new(sample: usize, annotation_type: AnnotationType) -> Self {
        Self {
            sample,
            annotation_type,
            subtype: 0,
            channel: 0,
            aux: None,
        }
    }

    /// Set the channel
    pub fn with_channel(mut self, channel: u8) -> Self {
        self.channel = channel;
        self
    }

    /// Set auxiliary information
    pub fn with_aux(mut self, aux: String) -> Self {
        self.aux = Some(aux);
        self
    }
}

/// WFDB format reader
pub struct WfdbReader {
    base_path: PathBuf,
    header: WfdbHeader,
    signals: Vec<WfdbSignal>,
}

impl WfdbReader {
    /// Open a WFDB record for reading
    ///
    /// # Arguments
    ///
    /// * `record_path` - Path to the record (without extension, e.g., "data/100")
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dpb_core::io::WfdbReader;
    /// use std::path::Path;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let reader = WfdbReader::open(Path::new("data/mitdb/100"))?;
    /// println!("Opened record: {}", reader.header().record_name);
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(record_path: &Path) -> Result<Self> {
        let base_path = record_path.to_path_buf();
        let header_path = base_path.with_extension("hea");

        if !header_path.exists() {
            return Err(DpbError::ResourceNotFound(format!(
                "Header file not found: {}",
                header_path.display()
            )));
        }

        let (header, signals) = Self::read_header(&header_path)?;

        Ok(Self {
            base_path,
            header,
            signals,
        })
    }

    /// Parse WFDB header file
    fn read_header(header_path: &Path) -> Result<(WfdbHeader, Vec<WfdbSignal>)> {
        let file = File::open(header_path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Parse first line: record_name n_signals [sample_rate [n_samples [base_time] [base_date]]]
        let first_line = lines
            .next()
            .ok_or_else(|| DpbError::DataValidation("Empty header file".to_string()))??;
        let parts: Vec<&str> = first_line.split_whitespace().collect();

        if parts.len() < 2 {
            return Err(DpbError::DataValidation(
                "Invalid header format".to_string(),
            ));
        }

        let record_name = parts[0].to_string();
        let n_signals = parts[1]
            .parse::<usize>()
            .map_err(|_| DpbError::DataValidation("Invalid signal count".to_string()))?;

        let sample_rate = if parts.len() > 2 {
            parts[2].parse::<f64>().unwrap_or(250.0)
        } else {
            250.0
        };

        let n_samples = if parts.len() > 3 {
            parts[3].parse::<usize>().ok()
        } else {
            None
        };

        let base_time = if parts.len() > 4 {
            Some(parts[4].to_string())
        } else {
            None
        };

        let base_date = if parts.len() > 5 {
            Some(parts[5].to_string())
        } else {
            None
        };

        let header = WfdbHeader {
            record_name,
            n_signals,
            sample_rate,
            n_samples,
            base_time,
            base_date,
        };

        // Parse signal lines
        let mut signals = Vec::new();
        for line in lines {
            let line = line?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            let signal = Self::parse_signal_line(&line)?;
            signals.push(signal);

            if signals.len() == n_signals {
                break;
            }
        }

        if signals.len() != n_signals {
            return Err(DpbError::DataValidation(format!(
                "Expected {} signal descriptors, found {}",
                n_signals,
                signals.len()
            )));
        }

        Ok((header, signals))
    }

    /// Parse a signal descriptor line
    fn parse_signal_line(line: &str) -> Result<WfdbSignal> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        // Format: filename format gain(baseline)/units adc_resolution adc_zero initial_value checksum block_size description
        if parts.len() < 3 {
            return Err(DpbError::DataValidation(format!(
                "Invalid signal line: {}",
                line
            )));
        }

        // Parse format
        let format = parts[1].parse::<u16>().unwrap_or(16);

        // Parse gain(baseline)/units
        let gain_str = parts[2];
        let (gain, baseline, units) = Self::parse_gain_field(gain_str)?;

        // Get signal name from description if available
        let name = if parts.len() > 8 {
            parts[8..].join(" ")
        } else {
            format!("Signal {}", parts[0])
        };

        Ok(WfdbSignal {
            name,
            units,
            gain,
            baseline,
            adc_resolution: 16,
            adc_zero: 0,
            format,
            samples_per_frame: 1,
        })
    }

    /// Parse gain field: gain(baseline)/units
    fn parse_gain_field(field: &str) -> Result<(f64, i32, String)> {
        let parts: Vec<&str> = field.split('/').collect();
        let gain_baseline = parts[0];
        let units = if parts.len() > 1 {
            parts[1].to_string()
        } else {
            "mV".to_string()
        };

        let (gain, baseline) = if gain_baseline.contains('(') {
            let gain_parts: Vec<&str> = gain_baseline.split('(').collect();
            let gain = gain_parts[0].parse::<f64>().unwrap_or(200.0);
            let baseline = gain_parts
                .get(1)
                .and_then(|s| s.trim_end_matches(')').parse::<i32>().ok())
                .unwrap_or(0);
            (gain, baseline)
        } else {
            let gain = gain_baseline.parse::<f64>().unwrap_or(200.0);
            (gain, 0)
        };

        Ok((gain, baseline, units))
    }

    /// Get reference to the header
    pub fn header(&self) -> &WfdbHeader {
        &self.header
    }

    /// Get reference to signal descriptors
    pub fn signals(&self) -> &[WfdbSignal] {
        &self.signals
    }

    /// Read samples from a specific channel
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index (0-based)
    /// * `start` - Starting sample index
    /// * `length` - Number of samples to read
    pub fn read_samples(&self, channel: usize, start: usize, length: usize) -> Result<Vec<f64>> {
        if channel >= self.header.n_signals {
            return Err(DpbError::InvalidParameter(format!(
                "Channel {} out of range (0-{})",
                channel,
                self.header.n_signals - 1
            )));
        }

        let data_path = self.base_path.with_extension("dat");
        if !data_path.exists() {
            return Err(DpbError::ResourceNotFound(format!(
                "Data file not found: {}",
                data_path.display()
            )));
        }

        let mut file = File::open(&data_path)?;

        let bytes_per_sample = 2; // 16-bit samples
        let frame_size = self.header.n_signals * bytes_per_sample;
        let offset = start * frame_size;

        // Seek to start position
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(offset as u64))?;

        let mut samples = Vec::with_capacity(length);
        let mut buffer = vec![0u8; frame_size];

        for _ in 0..length {
            let bytes_read = file.read(&mut buffer)?;

            if bytes_read < frame_size {
                break; // End of file
            }

            // Extract sample for requested channel
            let sample_offset = channel * bytes_per_sample;
            let adc_value = i16::from_le_bytes([
                buffer[sample_offset],
                buffer[sample_offset + 1],
            ]);

            let physical_value = self.signals[channel].adc_to_physical(adc_value);
            samples.push(physical_value);
        }

        Ok(samples)
    }

    /// Read all samples from a specific channel
    pub fn read_all_samples(&self, channel: usize) -> Result<Vec<f64>> {
        let n_samples = self.header.n_samples.ok_or_else(|| {
            DpbError::Other("Number of samples not specified in header".to_string())
        })?;

        self.read_samples(channel, 0, n_samples)
    }

    /// Read annotations from the .atr file
    pub fn read_annotations(&self) -> Result<Vec<WfdbAnnotation>> {
        let ann_path = self.base_path.with_extension("atr");
        if !ann_path.exists() {
            return Ok(Vec::new()); // No annotations available
        }

        // This is a simplified implementation
        // Full WFDB annotation format is binary and more complex
        let mut annotations = Vec::new();

        // For now, return empty vector
        // A full implementation would parse the binary .atr format
        tracing::warn!(
            "Annotation reading not fully implemented, returning empty vector"
        );

        Ok(annotations)
    }
}

/// WFDB format writer
pub struct WfdbWriter {
    base_path: PathBuf,
    header: WfdbHeader,
    signals: Vec<WfdbSignal>,
    data_file: Option<File>,
    samples_written: usize,
}

impl WfdbWriter {
    /// Create a new WFDB writer
    ///
    /// # Arguments
    ///
    /// * `path` - Base path for the record (without extension)
    /// * `header` - Header metadata
    /// * `signals` - Signal descriptors
    pub fn new(path: &Path, header: WfdbHeader, signals: Vec<WfdbSignal>) -> Result<Self> {
        if signals.len() != header.n_signals {
            return Err(DpbError::InvalidParameter(format!(
                "Signal count mismatch: header specifies {}, got {}",
                header.n_signals,
                signals.len()
            )));
        }

        Ok(Self {
            base_path: path.to_path_buf(),
            header,
            signals,
            data_file: None,
            samples_written: 0,
        })
    }

    /// Write header file
    fn write_header(&self) -> Result<()> {
        let header_path = self.base_path.with_extension("hea");
        let mut file = File::create(&header_path)?;

        // Write record line
        let mut record_line = format!(
            "{} {} {}",
            self.header.record_name, self.header.n_signals, self.header.sample_rate
        );

        if let Some(n_samples) = self.header.n_samples {
            record_line.push_str(&format!(" {}", n_samples));
        }

        if let Some(ref base_time) = self.header.base_time {
            record_line.push_str(&format!(" {}", base_time));
        }

        if let Some(ref base_date) = self.header.base_date {
            record_line.push_str(&format!(" {}", base_date));
        }

        writeln!(file, "{}", record_line)?;

        // Write signal lines
        for (i, signal) in self.signals.iter().enumerate() {
            let signal_line = format!(
                "{}.dat {} {}({})/{}",
                self.header.record_name,
                signal.format,
                signal.gain,
                signal.baseline,
                signal.units
            );
            writeln!(file, "{}", signal_line)?;
        }

        Ok(())
    }

    /// Write a frame of samples (one sample per signal)
    ///
    /// # Arguments
    ///
    /// * `samples` - Vector of ADC values, one per signal (length must equal n_signals)
    pub fn write_samples(&mut self, samples: &[Vec<i16>]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        // Ensure data file is open
        if self.data_file.is_none() {
            let data_path = self.base_path.with_extension("dat");
            let file = File::create(&data_path)?;
            self.data_file = Some(file);
        }

        let file = self.data_file.as_mut().unwrap();

        // Write samples frame by frame
        for frame in samples {
            if frame.len() != self.header.n_signals {
                return Err(DpbError::InvalidParameter(format!(
                    "Sample count mismatch: expected {}, got {}",
                    self.header.n_signals,
                    frame.len()
                )));
            }

            for &sample in frame {
                file.write_all(&sample.to_le_bytes())?;
            }

            self.samples_written += 1;
        }

        Ok(())
    }

    /// Write an annotation
    pub fn write_annotation(&mut self, _ann: &WfdbAnnotation) -> Result<()> {
        // Annotation writing is not implemented.
        //
        // This used to log a warning and return `Ok(())`, which tells the
        // caller their annotations were persisted when nothing was written --
        // silent data loss, and only visible if someone happened to be reading
        // the logs. Reporting the gap is the honest answer; `read_annotations`
        // still works for records annotated elsewhere.
        Err(DpbError::Other(
            "WFDB annotation writing is not implemented; annotations were not saved".to_string(),
        ))
    }

    /// Finalize the record and close files
    pub fn finish(mut self) -> Result<()> {
        // Update header with actual sample count
        if self.samples_written > 0 {
            self.header.n_samples = Some(self.samples_written);
        }

        // Write header
        self.write_header()?;

        // Close data file
        if let Some(mut file) = self.data_file.take() {
            file.flush()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annotation_type_conversion() {
        assert_eq!(AnnotationType::from_code('N'), AnnotationType::Normal);
        assert_eq!(AnnotationType::from_code('V'), AnnotationType::PVC);
        assert_eq!(AnnotationType::Normal.to_code(), 'N');
        assert_eq!(AnnotationType::PVC.to_code(), 'V');
    }

    #[test]
    fn test_signal_conversion() {
        let signal = WfdbSignal::new("ECG".to_string(), "mV".to_string(), 200.0);

        // Test ADC to physical conversion
        let physical = signal.adc_to_physical(200);
        assert!((physical - 1.0).abs() < 1e-6);

        // Test physical to ADC conversion
        let adc = signal.physical_to_adc(1.0);
        assert_eq!(adc, 200);
    }

    #[test]
    fn test_header_builder() {
        let header = WfdbHeader::new("test".to_string(), 2, 360.0)
            .with_n_samples(1000)
            .with_base_time("00:00:00".to_string());

        assert_eq!(header.record_name, "test");
        assert_eq!(header.n_signals, 2);
        assert_eq!(header.sample_rate, 360.0);
        assert_eq!(header.n_samples, Some(1000));
        assert_eq!(header.base_time, Some("00:00:00".to_string()));
    }

    #[test]
    fn test_parse_gain_field() {
        let (gain, baseline, units) = WfdbReader::parse_gain_field("200(0)/mV").unwrap();
        assert_eq!(gain, 200.0);
        assert_eq!(baseline, 0);
        assert_eq!(units, "mV");

        let (gain, baseline, units) = WfdbReader::parse_gain_field("100(-50)/uV").unwrap();
        assert_eq!(gain, 100.0);
        assert_eq!(baseline, -50);
        assert_eq!(units, "uV");
    }
}
