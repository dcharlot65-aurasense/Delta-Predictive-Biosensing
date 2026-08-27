//! # GDF (General Data Format) Support
//!
//! This module provides support for reading and writing General Data Format (GDF) files,
//! which is an extension of EDF designed to overcome some of its limitations.
//!
//! ## Format Overview
//!
//! GDF format features:
//! - **Header** (256 bytes fixed + variable): Extended metadata
//! - **Signal Headers**: Enhanced signal information
//! - **Data Records**: Contiguous blocks with multiple data types
//! - **Event Table**: Annotations and markers
//!
//! Supports GDF versions 1.25, 2.20, and later.
//!
//! ## References
//!
//! - [GDF Specification](https://arxiv.org/abs/cs/0608052)
//! - [GDF on BioSig](http://biosig.sourceforge.net/gdf.html)

use crate::error::{DpbError, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// GDF format version
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GdfVersion {
    /// GDF 1.x (1.25)
    V1_25,
    /// GDF 2.x (2.20+)
    V2_20,
    /// GDF 2.x (2.50+)
    V2_50,
    /// Unknown version
    Unknown(f32),
}

impl GdfVersion {
    /// Parse version from string
    fn from_str(s: &str) -> Self {
        let version = s.trim().parse::<f32>().unwrap_or(1.25);
        match version {
            v if (v - 1.25).abs() < 0.01 => Self::V1_25,
            v if (v - 2.20).abs() < 0.01 => Self::V2_20,
            v if (v - 2.50).abs() < 0.01 => Self::V2_50,
            v if v >= 2.20 => Self::V2_20,
            v => Self::Unknown(v),
        }
    }

    /// Get version as float
    fn as_float(&self) -> f32 {
        match self {
            Self::V1_25 => 1.25,
            Self::V2_20 => 2.20,
            Self::V2_50 => 2.50,
            Self::Unknown(v) => *v,
        }
    }
}

/// GDF file header
#[derive(Debug, Clone)]
pub struct GdfHeader {
    /// Format version
    pub version: GdfVersion,
    /// Patient identification
    pub patient_id: String,
    /// Recording identification
    pub recording_id: String,
    /// Start date and time (Unix timestamp for GDF 2.x)
    pub start_time: i64,
    /// Patient birthday (Unix timestamp for GDF 2.x)
    pub patient_birthday: i64,
    /// Header length in bytes
    pub header_bytes: usize,
    /// Equipment provider identification
    pub equipment_id: String,
    /// Laboratory identification
    pub lab_id: String,
    /// Technician identification
    pub technician_id: String,
    /// Number of data records
    pub n_records: i64,
    /// Duration of a data record in seconds
    pub record_duration: f64,
    /// Number of signals
    pub n_signals: usize,
    /// Patient name
    pub patient_name: String,
    /// Patient sex (0: unknown, 1: male, 2: female)
    pub patient_sex: u8,
    /// Patient handedness (0: unknown, 1: right, 2: left, 3: equal)
    pub patient_handedness: u8,
    /// Patient weight in kg
    pub patient_weight: f32,
    /// Patient height in cm
    pub patient_height: f32,
}

impl GdfHeader {
    /// Create a new GDF header
    pub fn new(patient_id: String, recording_id: String, n_signals: usize) -> Self {
        Self {
            version: GdfVersion::V2_20,
            patient_id,
            recording_id,
            start_time: 0,
            patient_birthday: 0,
            header_bytes: 256 + n_signals * 256,
            equipment_id: String::new(),
            lab_id: String::new(),
            technician_id: String::new(),
            n_records: -1,
            record_duration: 1.0,
            n_signals,
            patient_name: String::new(),
            patient_sex: 0,
            patient_handedness: 0,
            patient_weight: 0.0,
            patient_height: 0.0,
        }
    }

    /// Set version
    pub fn with_version(mut self, version: GdfVersion) -> Self {
        self.version = version;
        self
    }

    /// Set start time
    pub fn with_start_time(mut self, timestamp: i64) -> Self {
        self.start_time = timestamp;
        self
    }

    /// Set patient information
    pub fn with_patient_info(
        mut self,
        name: String,
        sex: u8,
        birthday: i64,
        height: f32,
        weight: f32,
    ) -> Self {
        self.patient_name = name;
        self.patient_sex = sex;
        self.patient_birthday = birthday;
        self.patient_height = height;
        self.patient_weight = weight;
        self
    }

    /// Set record parameters
    pub fn with_records(mut self, n_records: i64, duration: f64) -> Self {
        self.n_records = n_records;
        self.record_duration = duration;
        self
    }
}

/// GDF data type codes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GdfDataType {
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit signed integer
    Int16,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit signed integer
    Int32,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit signed integer
    Int64,
    /// 64-bit unsigned integer
    UInt64,
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// 24-bit signed integer (BDF format)
    Int24,
}

impl GdfDataType {
    /// Get data type from code
    fn from_code(code: u16) -> Result<Self> {
        match code {
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt8),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt16),
            5 => Ok(Self::Int32),
            6 => Ok(Self::UInt32),
            7 => Ok(Self::Int64),
            8 => Ok(Self::UInt64),
            16 => Ok(Self::Float32),
            17 => Ok(Self::Float64),
            279 => Ok(Self::Int24), // Special code for 24-bit (255 + 24)
            _ => Err(DpbError::DataValidation(format!(
                "Unknown GDF data type: {}",
                code
            ))),
        }
    }

    /// Get code for data type
    fn to_code(self) -> u16 {
        match self {
            Self::Int8 => 1,
            Self::UInt8 => 2,
            Self::Int16 => 3,
            Self::UInt16 => 4,
            Self::Int32 => 5,
            Self::UInt32 => 6,
            Self::Int64 => 7,
            Self::UInt64 => 8,
            Self::Float32 => 16,
            Self::Float64 => 17,
            Self::Int24 => 279, // 255 + 24
        }
    }

    /// Get size in bytes
    fn size_bytes(&self) -> usize {
        match self {
            Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 => 2,
            Self::Int24 => 3,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
        }
    }
}

/// GDF signal descriptor
#[derive(Debug, Clone)]
pub struct GdfSignal {
    /// Label (e.g., "EEG Fpz-Cz")
    pub label: String,
    /// Transducer type
    pub transducer_type: String,
    /// Physical dimension (e.g., "uV")
    pub physical_dimension: String,
    /// Physical minimum
    pub physical_min: f64,
    /// Physical maximum
    pub physical_max: f64,
    /// Digital minimum
    pub digital_min: f64,
    /// Digital maximum
    pub digital_max: f64,
    /// Prefiltering information
    pub prefiltering: String,
    /// Number of samples in each data record
    pub samples_per_record: usize,
    /// Data type
    pub data_type: GdfDataType,
    /// Sensor position (x, y, z coordinates)
    pub sensor_position: (f32, f32, f32),
    /// Sensor orientation
    pub sensor_orientation: f32,
    /// Low-pass filter setting (Hz)
    pub lowpass: f32,
    /// High-pass filter setting (Hz)
    pub highpass: f32,
    /// Notch filter setting (Hz)
    pub notch: f32,
}

impl GdfSignal {
    /// Create a new GDF signal descriptor
    pub fn new(
        label: String,
        physical_dimension: String,
        samples_per_record: usize,
        data_type: GdfDataType,
    ) -> Self {
        Self {
            label,
            transducer_type: String::new(),
            physical_dimension,
            physical_min: -500.0,
            physical_max: 500.0,
            digital_min: -32768.0,
            digital_max: 32767.0,
            prefiltering: String::new(),
            samples_per_record,
            data_type,
            sensor_position: (0.0, 0.0, 0.0),
            sensor_orientation: 0.0,
            lowpass: 0.0,
            highpass: 0.0,
            notch: 0.0,
        }
    }

    /// Set physical range
    pub fn with_physical_range(mut self, min: f64, max: f64) -> Self {
        self.physical_min = min;
        self.physical_max = max;
        self
    }

    /// Set digital range
    pub fn with_digital_range(mut self, min: f64, max: f64) -> Self {
        self.digital_min = min;
        self.digital_max = max;
        self
    }

    /// Set filter settings
    pub fn with_filters(mut self, highpass: f32, lowpass: f32, notch: f32) -> Self {
        self.highpass = highpass;
        self.lowpass = lowpass;
        self.notch = notch;
        self
    }

    /// Calculate sampling rate
    pub fn sample_rate(&self, record_duration: f64) -> f64 {
        self.samples_per_record as f64 / record_duration
    }

    /// Convert digital value to physical value
    pub fn digital_to_physical(&self, digital: f64) -> f64 {
        let digital_range = self.digital_max - self.digital_min;
        let physical_range = self.physical_max - self.physical_min;

        if digital_range == 0.0 {
            return self.physical_min;
        }

        let normalized = (digital - self.digital_min) / digital_range;
        self.physical_min + normalized * physical_range
    }

    /// Convert physical value to digital value
    pub fn physical_to_digital(&self, physical: f64) -> f64 {
        let physical_range = self.physical_max - self.physical_min;
        let digital_range = self.digital_max - self.digital_min;

        if physical_range == 0.0 {
            return self.digital_min;
        }

        let normalized = (physical - self.physical_min) / physical_range;
        self.digital_min + normalized * digital_range
    }
}

/// GDF event/annotation entry
#[derive(Debug, Clone)]
pub struct GdfEvent {
    /// Sample position
    pub position: u32,
    /// Event type code
    pub event_type: u16,
    /// Event channel (0 = all channels)
    pub channel: u16,
    /// Duration in samples
    pub duration: u32,
    /// Additional description
    pub description: Option<String>,
}

impl GdfEvent {
    /// Create a new event
    pub fn new(position: u32, event_type: u16) -> Self {
        Self {
            position,
            event_type,
            channel: 0,
            duration: 0,
            description: None,
        }
    }

    /// Set channel
    pub fn with_channel(mut self, channel: u16) -> Self {
        self.channel = channel;
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: u32) -> Self {
        self.duration = duration;
        self
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

/// GDF file reader
pub struct GdfReader {
    file: File,
    header: GdfHeader,
    signals: Vec<GdfSignal>,
    events: Vec<GdfEvent>,
}

impl GdfReader {
    /// Open a GDF file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GDF file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dpb_core::io::GdfReader;
    /// use std::path::Path;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let reader = GdfReader::open(Path::new("data/recording.gdf"))?;
    /// println!("GDF version: {:?}", reader.header().version);
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;

        let (header, signals, events) = Self::read_headers(&mut file)?;

        Ok(Self {
            file,
            header,
            signals,
            events,
        })
    }

    /// Read GDF headers and event table
    fn read_headers(file: &mut File) -> Result<(GdfHeader, Vec<GdfSignal>, Vec<GdfEvent>)> {
        // Read fixed header (256 bytes)
        let mut buffer = vec![0u8; 256];
        file.read_exact(&mut buffer)?;

        // Parse version from first 8 bytes
        let version_str = String::from_utf8_lossy(&buffer[0..8])
            .trim()
            .trim_start_matches("GDF")
            .to_string();
        let version = GdfVersion::from_str(&version_str);

        // Parse header fields
        let patient_id = Self::read_ascii(&buffer[8..88]);
        let recording_id = Self::read_ascii(&buffer[88..168]);

        // GDF 2.x uses Unix timestamp (8 bytes at offset 168)
        let start_time = if version.as_float() >= 2.0 {
            i64::from_le_bytes(buffer[168..176].try_into().unwrap())
        } else {
            0
        };

        let patient_birthday = if version.as_float() >= 2.0 {
            i64::from_le_bytes(buffer[176..184].try_into().unwrap())
        } else {
            0
        };

        let header_bytes =
            u64::from_le_bytes(buffer[184..192].try_into().unwrap()) as usize;

        // Equipment provider (GDF 2.x)
        let equipment_id = Self::read_ascii(&buffer[192..200]);

        // Read additional fields
        let n_records = i64::from_le_bytes(buffer[236..244].try_into().unwrap());
        let record_duration = {
            let num = u32::from_le_bytes(buffer[244..248].try_into().unwrap());
            let den = u32::from_le_bytes(buffer[248..252].try_into().unwrap());
            if den > 0 {
                num as f64 / den as f64
            } else {
                1.0
            }
        };
        let n_signals = u16::from_le_bytes(buffer[252..254].try_into().unwrap()) as usize;

        let header = GdfHeader {
            version,
            patient_id,
            recording_id,
            start_time,
            patient_birthday,
            header_bytes,
            equipment_id,
            lab_id: String::new(),
            technician_id: String::new(),
            n_records,
            record_duration,
            n_signals,
            patient_name: String::new(),
            patient_sex: 0,
            patient_handedness: 0,
            patient_weight: 0.0,
            patient_height: 0.0,
        };

        // Read signal headers (256 bytes per signal)
        let signal_header_bytes = n_signals * 256;
        let mut signal_buffer = vec![0u8; signal_header_bytes];
        file.read_exact(&mut signal_buffer)?;

        let mut signals = Vec::with_capacity(n_signals);

        for i in 0..n_signals {
            let offset = i * 256;
            let signal_data = &signal_buffer[offset..offset + 256];

            let label = Self::read_ascii(&signal_data[0..16]);
            let transducer_type = Self::read_ascii(&signal_data[16..96]);
            let physical_dimension = Self::read_ascii(&signal_data[96..104]);
            let physical_min = f64::from_le_bytes(signal_data[104..112].try_into().unwrap());
            let physical_max = f64::from_le_bytes(signal_data[112..120].try_into().unwrap());
            let digital_min = f64::from_le_bytes(signal_data[120..128].try_into().unwrap());
            let digital_max = f64::from_le_bytes(signal_data[128..136].try_into().unwrap());
            let prefiltering = Self::read_ascii(&signal_data[136..204]);
            let samples_per_record =
                u32::from_le_bytes(signal_data[204..208].try_into().unwrap()) as usize;
            let data_type_code = u16::from_le_bytes(signal_data[208..210].try_into().unwrap());
            let data_type = GdfDataType::from_code(data_type_code)?;

            // Read sensor position (GDF 2.x)
            let sensor_position = if version.as_float() >= 2.0 {
                let x = f32::from_le_bytes(signal_data[220..224].try_into().unwrap());
                let y = f32::from_le_bytes(signal_data[224..228].try_into().unwrap());
                let z = f32::from_le_bytes(signal_data[228..232].try_into().unwrap());
                (x, y, z)
            } else {
                (0.0, 0.0, 0.0)
            };

            // Filter settings (GDF 2.x)
            let (highpass, lowpass, notch) = if version.as_float() >= 2.0 {
                let hp = f32::from_le_bytes(signal_data[232..236].try_into().unwrap());
                let lp = f32::from_le_bytes(signal_data[236..240].try_into().unwrap());
                let notch = f32::from_le_bytes(signal_data[240..244].try_into().unwrap());
                (hp, lp, notch)
            } else {
                (0.0, 0.0, 0.0)
            };

            signals.push(GdfSignal {
                label,
                transducer_type,
                physical_dimension,
                physical_min,
                physical_max,
                digital_min,
                digital_max,
                prefiltering,
                samples_per_record,
                data_type,
                sensor_position,
                sensor_orientation: 0.0,
                lowpass,
                highpass,
                notch,
            });
        }

        // Read event table if present (GDF 2.x feature)
        let events = Self::read_event_table(file, &header)?;

        Ok((header, signals, events))
    }

    /// Read event table (simplified)
    fn read_event_table(_file: &mut File, _header: &GdfHeader) -> Result<Vec<GdfEvent>> {
        // Event table reading is simplified for now
        // Full implementation would parse event table at end of file
        Ok(Vec::new())
    }

    /// Read ASCII field from buffer
    fn read_ascii(buffer: &[u8]) -> String {
        String::from_utf8_lossy(buffer)
            .trim_end_matches('\0')
            .trim()
            .to_string()
    }

    /// Get reference to the header
    pub fn header(&self) -> &GdfHeader {
        &self.header
    }

    /// Get reference to signal descriptors
    pub fn signals(&self) -> &[GdfSignal] {
        &self.signals
    }

    /// Get reference to events
    pub fn events(&self) -> &[GdfEvent] {
        &self.events
    }

    /// Read all samples for a specific signal
    pub fn read_signal(&mut self, signal_index: usize) -> Result<Vec<f64>> {
        if signal_index >= self.header.n_signals {
            return Err(DpbError::InvalidParameter(format!(
                "Signal index {} out of range",
                signal_index
            )));
        }

        if self.header.n_records < 0 {
            return Err(DpbError::Other("Unknown number of records".to_string()));
        }

        let signal = &self.signals[signal_index];
        let total_samples = signal.samples_per_record * self.header.n_records as usize;
        let mut samples = Vec::with_capacity(total_samples);

        // Calculate bytes per record
        let bytes_per_record: usize = self
            .signals
            .iter()
            .map(|s| s.samples_per_record * s.data_type.size_bytes())
            .sum();

        // Calculate offset to this signal's data
        let signal_offset: usize = self.signals[0..signal_index]
            .iter()
            .map(|s| s.samples_per_record * s.data_type.size_bytes())
            .sum();

        // Seek to start of data records
        self.file
            .seek(SeekFrom::Start(self.header.header_bytes as u64))?;

        // Read each record
        let mut record_buffer = vec![0u8; bytes_per_record];

        for _ in 0..self.header.n_records {
            self.file.read_exact(&mut record_buffer)?;

            // Extract this signal's samples
            let signal_bytes = &record_buffer
                [signal_offset..signal_offset + signal.samples_per_record * signal.data_type.size_bytes()];

            for i in 0..signal.samples_per_record {
                let offset = i * signal.data_type.size_bytes();
                let digital = Self::read_digital_value(
                    &signal_bytes[offset..offset + signal.data_type.size_bytes()],
                    signal.data_type,
                )?;
                let physical = signal.digital_to_physical(digital);
                samples.push(physical);
            }
        }

        Ok(samples)
    }

    /// Read digital value based on data type
    fn read_digital_value(bytes: &[u8], data_type: GdfDataType) -> Result<f64> {
        Ok(match data_type {
            GdfDataType::Int8 => i8::from_le_bytes([bytes[0]]) as f64,
            GdfDataType::UInt8 => bytes[0] as f64,
            GdfDataType::Int16 => i16::from_le_bytes(bytes[0..2].try_into().unwrap()) as f64,
            GdfDataType::UInt16 => u16::from_le_bytes(bytes[0..2].try_into().unwrap()) as f64,
            GdfDataType::Int32 => i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as f64,
            GdfDataType::UInt32 => u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as f64,
            GdfDataType::Int64 => i64::from_le_bytes(bytes[0..8].try_into().unwrap()) as f64,
            GdfDataType::UInt64 => u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as f64,
            GdfDataType::Float32 => f32::from_le_bytes(bytes[0..4].try_into().unwrap()) as f64,
            GdfDataType::Float64 => f64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            GdfDataType::Int24 => {
                // 24-bit signed integer
                let mut b = [0u8; 4];
                b[0..3].copy_from_slice(&bytes[0..3]);
                // Sign extend if negative
                if bytes[2] & 0x80 != 0 {
                    b[3] = 0xFF;
                }
                i32::from_le_bytes(b) as f64
            }
        })
    }

    /// Read a specific data record
    pub fn read_record(&mut self, record_index: usize) -> Result<Vec<Vec<f64>>> {
        if self.header.n_records < 0 {
            return Err(DpbError::Other("Unknown number of records".to_string()));
        }

        if record_index >= self.header.n_records as usize {
            return Err(DpbError::InvalidParameter(format!(
                "Record index {} out of range",
                record_index
            )));
        }

        let bytes_per_record: usize = self
            .signals
            .iter()
            .map(|s| s.samples_per_record * s.data_type.size_bytes())
            .sum();

        let record_offset = self.header.header_bytes + record_index * bytes_per_record;
        self.file.seek(SeekFrom::Start(record_offset as u64))?;

        let mut record_buffer = vec![0u8; bytes_per_record];
        self.file.read_exact(&mut record_buffer)?;

        let mut all_samples = Vec::with_capacity(self.header.n_signals);
        let mut offset = 0;

        for signal in &self.signals {
            let mut signal_samples = Vec::with_capacity(signal.samples_per_record);
            let signal_bytes =
                &record_buffer[offset..offset + signal.samples_per_record * signal.data_type.size_bytes()];

            for i in 0..signal.samples_per_record {
                let byte_offset = i * signal.data_type.size_bytes();
                let digital = Self::read_digital_value(
                    &signal_bytes[byte_offset..byte_offset + signal.data_type.size_bytes()],
                    signal.data_type,
                )?;
                let physical = signal.digital_to_physical(digital);
                signal_samples.push(physical);
            }

            all_samples.push(signal_samples);
            offset += signal.samples_per_record * signal.data_type.size_bytes();
        }

        Ok(all_samples)
    }
}

/// GDF file writer
pub struct GdfWriter {
    file: File,
    header: GdfHeader,
    signals: Vec<GdfSignal>,
    events: Vec<GdfEvent>,
    records_written: usize,
}

impl GdfWriter {
    /// Create a new GDF writer
    pub fn new(path: &Path, header: GdfHeader, signals: Vec<GdfSignal>) -> Result<Self> {
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
            events: Vec::new(),
            records_written: 0,
        })
    }

    /// Write GDF headers
    fn write_headers(&mut self) -> Result<()> {
        // Write fixed header (256 bytes)
        let mut buffer = vec![0u8; 256];

        // Version string
        let version_str = format!("GDF{:.2}", self.header.version.as_float());
        Self::write_field(&mut buffer, 0, 8, version_str.as_bytes());

        // Patient and recording IDs
        Self::write_field(&mut buffer, 8, 80, self.header.patient_id.as_bytes());
        Self::write_field(&mut buffer, 88, 80, self.header.recording_id.as_bytes());

        // Timestamps (GDF 2.x)
        buffer[168..176].copy_from_slice(&self.header.start_time.to_le_bytes());
        buffer[176..184].copy_from_slice(&self.header.patient_birthday.to_le_bytes());

        // Header size
        buffer[184..192].copy_from_slice(&(self.header.header_bytes as u64).to_le_bytes());

        // Equipment ID
        Self::write_field(&mut buffer, 192, 8, self.header.equipment_id.as_bytes());

        // Record info
        buffer[236..244].copy_from_slice(&self.header.n_records.to_le_bytes());

        // Record duration as fraction
        let num = (self.header.record_duration * 1e9) as u32;
        let den = 1_000_000_000u32;
        buffer[244..248].copy_from_slice(&num.to_le_bytes());
        buffer[248..252].copy_from_slice(&den.to_le_bytes());

        buffer[252..254].copy_from_slice(&(self.header.n_signals as u16).to_le_bytes());

        self.file.write_all(&buffer)?;

        // Write signal headers
        for signal in &self.signals {
            let mut sig_buffer = vec![0u8; 256];

            Self::write_field(&mut sig_buffer, 0, 16, signal.label.as_bytes());
            Self::write_field(&mut sig_buffer, 16, 80, signal.transducer_type.as_bytes());
            Self::write_field(
                &mut sig_buffer,
                96,
                8,
                signal.physical_dimension.as_bytes(),
            );

            sig_buffer[104..112].copy_from_slice(&signal.physical_min.to_le_bytes());
            sig_buffer[112..120].copy_from_slice(&signal.physical_max.to_le_bytes());
            sig_buffer[120..128].copy_from_slice(&signal.digital_min.to_le_bytes());
            sig_buffer[128..136].copy_from_slice(&signal.digital_max.to_le_bytes());

            Self::write_field(&mut sig_buffer, 136, 68, signal.prefiltering.as_bytes());

            sig_buffer[204..208]
                .copy_from_slice(&(signal.samples_per_record as u32).to_le_bytes());
            sig_buffer[208..210].copy_from_slice(&signal.data_type.to_code().to_le_bytes());

            // Sensor position (GDF 2.x)
            sig_buffer[220..224].copy_from_slice(&signal.sensor_position.0.to_le_bytes());
            sig_buffer[224..228].copy_from_slice(&signal.sensor_position.1.to_le_bytes());
            sig_buffer[228..232].copy_from_slice(&signal.sensor_position.2.to_le_bytes());

            // Filter settings
            sig_buffer[232..236].copy_from_slice(&signal.highpass.to_le_bytes());
            sig_buffer[236..240].copy_from_slice(&signal.lowpass.to_le_bytes());
            sig_buffer[240..244].copy_from_slice(&signal.notch.to_le_bytes());

            self.file.write_all(&sig_buffer)?;
        }

        Ok(())
    }

    /// Write field to buffer
    fn write_field(buffer: &mut [u8], offset: usize, size: usize, value: &[u8]) {
        let len = value.len().min(size);
        buffer[offset..offset + len].copy_from_slice(&value[..len]);
    }

    /// Write digital value based on data type
    fn write_digital_value(
        file: &mut File,
        value: f64,
        data_type: GdfDataType,
    ) -> Result<()> {
        match data_type {
            GdfDataType::Int8 => file.write_all(&(value as i8).to_le_bytes())?,
            GdfDataType::UInt8 => file.write_all(&[value as u8])?,
            GdfDataType::Int16 => file.write_all(&(value as i16).to_le_bytes())?,
            GdfDataType::UInt16 => file.write_all(&(value as u16).to_le_bytes())?,
            GdfDataType::Int32 => file.write_all(&(value as i32).to_le_bytes())?,
            GdfDataType::UInt32 => file.write_all(&(value as u32).to_le_bytes())?,
            GdfDataType::Int64 => file.write_all(&(value as i64).to_le_bytes())?,
            GdfDataType::UInt64 => file.write_all(&(value as u64).to_le_bytes())?,
            GdfDataType::Float32 => file.write_all(&(value as f32).to_le_bytes())?,
            GdfDataType::Float64 => file.write_all(&value.to_le_bytes())?,
            GdfDataType::Int24 => {
                let int_val = value as i32;
                let bytes = int_val.to_le_bytes();
                file.write_all(&bytes[0..3])?;
            }
        }
        Ok(())
    }

    /// Write a data record
    pub fn write_record(&mut self, record: &[Vec<f64>]) -> Result<()> {
        if record.len() != self.header.n_signals {
            return Err(DpbError::InvalidParameter(format!(
                "Record must have {} signals",
                self.header.n_signals
            )));
        }

        for (i, samples) in record.iter().enumerate() {
            if samples.len() != self.signals[i].samples_per_record {
                return Err(DpbError::InvalidParameter(format!(
                    "Signal {} must have {} samples",
                    i, self.signals[i].samples_per_record
                )));
            }

            for &sample in samples {
                let digital = self.signals[i].physical_to_digital(sample);
                Self::write_digital_value(&mut self.file, digital, self.signals[i].data_type)?;
            }
        }

        self.records_written += 1;
        Ok(())
    }

    /// Add an event
    pub fn add_event(&mut self, event: GdfEvent) {
        self.events.push(event);
    }

    /// Finalize the GDF file
    pub fn finish(mut self) -> Result<()> {
        // Update header with actual record count
        self.header.n_records = self.records_written as i64;

        // Seek to beginning and rewrite header
        self.file.seek(SeekFrom::Start(0))?;
        self.write_headers()?;

        // Write event table (simplified - not fully implemented)
        // Full implementation would write event table at end

        self.file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdf_version() {
        assert_eq!(GdfVersion::from_str("1.25"), GdfVersion::V1_25);
        assert_eq!(GdfVersion::from_str("2.20"), GdfVersion::V2_20);
        assert_eq!(GdfVersion::V2_20.as_float(), 2.20);
    }

    #[test]
    fn test_data_type_conversion() {
        assert_eq!(GdfDataType::from_code(3).unwrap(), GdfDataType::Int16);
        assert_eq!(GdfDataType::Int16.to_code(), 3);
        assert_eq!(GdfDataType::Int16.size_bytes(), 2);
        assert_eq!(GdfDataType::Int24.size_bytes(), 3);
    }

    #[test]
    fn test_signal_conversion() {
        let signal = GdfSignal::new(
            "EEG".to_string(),
            "uV".to_string(),
            100,
            GdfDataType::Int16,
        )
        .with_physical_range(-500.0, 500.0)
        .with_digital_range(-32768.0, 32767.0);

        // Note: digital range -32768 to 32767 is not symmetric, so 0 doesn't map exactly to 0.0
        let physical = signal.digital_to_physical(0.0);
        assert!((physical - 0.0).abs() < 0.1);

        let physical = signal.digital_to_physical(32767.0);
        assert!((physical - 500.0).abs() < 1.0);
    }

    #[test]
    fn test_event_builder() {
        let event = GdfEvent::new(1000, 1)
            .with_channel(2)
            .with_duration(100)
            .with_description("Stimulus".to_string());

        assert_eq!(event.position, 1000);
        assert_eq!(event.event_type, 1);
        assert_eq!(event.channel, 2);
        assert_eq!(event.duration, 100);
        assert_eq!(event.description, Some("Stimulus".to_string()));
    }
}
