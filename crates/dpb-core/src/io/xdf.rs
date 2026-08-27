//! # XDF (Extensible Data Format) Support
//!
//! This module provides support for reading and writing Extensible Data Format (XDF) files,
//! which is used by Lab Streaming Layer (LSL) for recording multi-modal synchronized data.
//!
//! ## Format Overview
//!
//! XDF format features:
//! - **Multiple Streams**: Can contain different types of data (EEG, markers, video, etc.)
//! - **Chunk-based**: Optimized for streaming and large files
//! - **Clock Synchronization**: Precise time alignment across streams
//! - **XML Headers**: Rich metadata for each stream
//!
//! ## Stream Types
//!
//! Common stream types:
//! - EEG, EMG, ECG (physiological signals)
//! - Markers (event annotations)
//! - Gaze (eye tracking)
//! - MoCap (motion capture)
//! - VideoRectangle (video synchronization)
//!
//! ## References
//!
//! - [XDF Format Specification](https://github.com/sccn/xdf/wiki/Specifications)
//! - [Lab Streaming Layer](https://labstreaminglayer.readthedocs.io/)

use crate::error::{DpbError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, Write};
use std::path::Path;

/// XDF chunk types
#[derive(Debug, Clone, Copy, PartialEq)]
enum ChunkTag {
    /// File header chunk
    FileHeader = 1,
    /// Stream header chunk
    StreamHeader = 2,
    /// Sample data chunk (multiple samples)
    Samples = 3,
    /// Clock offset chunk
    ClockOffset = 4,
    /// Boundary chunk (stream start/end)
    Boundary = 5,
    /// Stream footer chunk
    StreamFooter = 6,
}

impl ChunkTag {
    fn from_u16(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::FileHeader),
            2 => Some(Self::StreamHeader),
            3 => Some(Self::Samples),
            4 => Some(Self::ClockOffset),
            5 => Some(Self::Boundary),
            6 => Some(Self::StreamFooter),
            _ => None,
        }
    }
}

/// XDF channel format
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelFormat {
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// String
    String,
}

impl ChannelFormat {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "float32" => Self::Float32,
            "double64" | "float64" => Self::Float64,
            "int8" => Self::Int8,
            "int16" => Self::Int16,
            "int32" => Self::Int32,
            "int64" => Self::Int64,
            "string" => Self::String,
            _ => Self::Float32, // Default to float32
        }
    }

    fn size_bytes(&self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float64 | Self::Int64 => 8,
            Self::Int8 => 1,
            Self::Int16 => 2,
            Self::String => 0, // Variable length
        }
    }
}

/// XDF stream information
#[derive(Debug, Clone)]
pub struct XdfStreamInfo {
    /// Stream name
    pub name: String,
    /// Stream type (e.g., "EEG", "Markers")
    pub stream_type: String,
    /// Number of channels
    pub channel_count: usize,
    /// Nominal sampling rate (0 for irregular)
    pub nominal_srate: f64,
    /// Channel format
    pub channel_format: ChannelFormat,
    /// Stream source ID
    pub source_id: String,
    /// Channel labels
    pub channel_labels: Vec<String>,
    /// Channel units
    pub channel_units: Vec<String>,
    /// Additional metadata (key-value pairs)
    pub metadata: HashMap<String, String>,
}

impl XdfStreamInfo {
    /// Create a new stream info
    pub fn new(
        name: String,
        stream_type: String,
        channel_count: usize,
        nominal_srate: f64,
    ) -> Self {
        Self {
            name,
            stream_type,
            channel_count,
            nominal_srate,
            channel_format: ChannelFormat::Float32,
            source_id: String::new(),
            channel_labels: vec![String::new(); channel_count],
            channel_units: vec![String::new(); channel_count],
            metadata: HashMap::new(),
        }
    }

    /// Set channel format
    pub fn with_format(mut self, format: ChannelFormat) -> Self {
        self.channel_format = format;
        self
    }

    /// Set channel labels
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.channel_labels = labels;
        self
    }

    /// Set channel units
    pub fn with_units(mut self, units: Vec<String>) -> Self {
        self.channel_units = units;
        self
    }

    /// Check if this is a marker stream
    pub fn is_marker_stream(&self) -> bool {
        self.stream_type.to_lowercase().contains("marker")
    }

    /// Check if this is a regular sampled stream
    pub fn is_regular_stream(&self) -> bool {
        self.nominal_srate > 0.0
    }
}

/// XDF stream data
#[derive(Debug, Clone)]
pub struct XdfStream {
    /// Stream ID
    pub stream_id: u32,
    /// Stream information
    pub info: XdfStreamInfo,
    /// Timestamps for each sample
    pub timestamps: Vec<f64>,
    /// Time series data [samples x channels]
    pub time_series: Vec<Vec<f64>>,
    /// Clock offsets for time synchronization
    pub clock_offsets: Vec<(f64, f64)>, // (time, offset)
}

impl XdfStream {
    /// Create a new stream
    fn new(stream_id: u32, info: XdfStreamInfo) -> Self {
        Self {
            stream_id,
            info,
            timestamps: Vec::new(),
            time_series: Vec::new(),
            clock_offsets: Vec::new(),
        }
    }

    /// Add a sample to the stream
    fn add_sample(&mut self, timestamp: f64, sample: Vec<f64>) {
        self.timestamps.push(timestamp);
        self.time_series.push(sample);
    }

    /// Add clock offset
    fn add_clock_offset(&mut self, time: f64, offset: f64) {
        self.clock_offsets.push((time, offset));
    }

    /// Get corrected timestamp using clock offsets
    pub fn corrected_timestamp(&self, index: usize) -> f64 {
        if index >= self.timestamps.len() {
            return 0.0;
        }

        let timestamp = self.timestamps[index];

        // Find appropriate clock offset
        if self.clock_offsets.is_empty() {
            return timestamp;
        }

        // Linear interpolation of clock offsets
        let mut best_offset = self.clock_offsets[0].1;
        for &(offset_time, offset_value) in &self.clock_offsets {
            if offset_time <= timestamp {
                best_offset = offset_value;
            } else {
                break;
            }
        }

        timestamp + best_offset
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.time_series.len()
    }

    /// Get channel count
    pub fn channel_count(&self) -> usize {
        self.info.channel_count
    }

    /// Get data for a specific channel
    pub fn channel_data(&self, channel: usize) -> Result<Vec<f64>> {
        if channel >= self.info.channel_count {
            return Err(DpbError::InvalidParameter(format!(
                "Channel {} out of range",
                channel
            )));
        }

        Ok(self
            .time_series
            .iter()
            .map(|sample| {
                if channel < sample.len() {
                    sample[channel]
                } else {
                    0.0
                }
            })
            .collect())
    }
}

/// XDF file container
pub struct XdfFile {
    streams: HashMap<u32, XdfStream>,
    file_version: (u8, u8), // (major, minor)
}

impl XdfFile {
    /// Create a new XDF file
    pub fn new() -> Self {
        Self {
            streams: HashMap::new(),
            file_version: (1, 0),
        }
    }

    /// Open an XDF file for reading
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the XDF file
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dpb_core::io::XdfFile;
    /// use std::path::Path;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let xdf = XdfFile::open(Path::new("data/recording.xdf"))?;
    /// println!("Loaded {} streams", xdf.stream_count());
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        Self::read_file(&mut reader)
    }

    /// Read XDF file
    fn read_file<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        // Read magic bytes "XDF:"
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if &magic != b"XDF:" {
            return Err(DpbError::DataValidation(
                "Not a valid XDF file (invalid magic bytes)".to_string(),
            ));
        }

        let mut xdf = Self::new();
        let mut current_streams: HashMap<u32, XdfStream> = HashMap::new();

        // Read chunks until end of file
        loop {
            // Try to read chunk length
            let mut length_buf = [0u8; 4];
            match reader.read_exact(&mut length_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let chunk_length = u32::from_le_bytes(length_buf) as usize;

            if chunk_length == 0 {
                break; // End of file
            }

            // Read chunk tag
            let mut tag_buf = [0u8; 2];
            reader.read_exact(&mut tag_buf)?;
            let tag = u16::from_le_bytes(tag_buf);

            let chunk_tag = ChunkTag::from_u16(tag);

            // Read chunk data
            let data_length = chunk_length.saturating_sub(2);
            let mut chunk_data = vec![0u8; data_length];
            if data_length > 0 {
                reader.read_exact(&mut chunk_data)?;
            }

            // Process chunk
            match chunk_tag {
                Some(ChunkTag::FileHeader) => {
                    xdf.file_version = Self::parse_file_header(&chunk_data)?;
                }
                Some(ChunkTag::StreamHeader) => {
                    let (stream_id, stream_info) = Self::parse_stream_header(&chunk_data)?;
                    let stream = XdfStream::new(stream_id, stream_info);
                    current_streams.insert(stream_id, stream);
                }
                Some(ChunkTag::Samples) => {
                    Self::parse_samples(&chunk_data, &mut current_streams)?;
                }
                Some(ChunkTag::ClockOffset) => {
                    Self::parse_clock_offset(&chunk_data, &mut current_streams)?;
                }
                Some(ChunkTag::StreamFooter) => {
                    // Stream ended, move to final collection
                    let stream_id = u32::from_le_bytes(chunk_data[0..4].try_into().unwrap());
                    if let Some(stream) = current_streams.remove(&stream_id) {
                        xdf.streams.insert(stream_id, stream);
                    }
                }
                _ => {
                    // Unknown or unhandled chunk type, skip
                }
            }
        }

        // Move any remaining streams
        for (id, stream) in current_streams {
            xdf.streams.insert(id, stream);
        }

        Ok(xdf)
    }

    /// Parse file header
    fn parse_file_header(_data: &[u8]) -> Result<(u8, u8)> {
        // File header contains XML with version info
        // Simplified parsing for now
        Ok((1, 0))
    }

    /// Parse stream header (simplified XML parsing)
    fn parse_stream_header(data: &[u8]) -> Result<(u32, XdfStreamInfo)> {
        let stream_id = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let xml_data = &data[4..];
        let xml_str = String::from_utf8_lossy(xml_data);

        // Simplified XML parsing (in production, use proper XML parser)
        let name = Self::extract_xml_value(&xml_str, "name").unwrap_or_default();
        let stream_type = Self::extract_xml_value(&xml_str, "type").unwrap_or_default();
        let channel_count: usize = Self::extract_xml_value(&xml_str, "channel_count")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let nominal_srate: f64 = Self::extract_xml_value(&xml_str, "nominal_srate")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let channel_format = Self::extract_xml_value(&xml_str, "channel_format")
            .map(|s| ChannelFormat::from_str(&s))
            .unwrap_or(ChannelFormat::Float32);

        let mut info = XdfStreamInfo::new(name, stream_type, channel_count, nominal_srate)
            .with_format(channel_format);

        // Extract channel labels if present
        if let Some(labels) = Self::extract_channel_labels(&xml_str, channel_count) {
            info.channel_labels = labels;
        }

        Ok((stream_id, info))
    }

    /// Extract XML value (simplified)
    fn extract_xml_value(xml: &str, tag: &str) -> Option<String> {
        let start_tag = format!("<{}>", tag);
        let end_tag = format!("</{}>", tag);

        if let Some(start_pos) = xml.find(&start_tag) {
            let content_start = start_pos + start_tag.len();
            if let Some(end_pos) = xml[content_start..].find(&end_tag) {
                return Some(xml[content_start..content_start + end_pos].to_string());
            }
        }
        None
    }

    /// Extract channel labels (simplified)
    fn extract_channel_labels(xml: &str, count: usize) -> Option<Vec<String>> {
        let mut labels = Vec::new();
        let mut search_pos = 0;

        while labels.len() < count {
            if let Some(start) = xml[search_pos..].find("<label>") {
                let abs_start = search_pos + start + 7;
                if let Some(end) = xml[abs_start..].find("</label>") {
                    labels.push(xml[abs_start..abs_start + end].to_string());
                    search_pos = abs_start + end;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if labels.is_empty() {
            None
        } else {
            Some(labels)
        }
    }

    /// Parse samples chunk
    fn parse_samples(data: &[u8], streams: &mut HashMap<u32, XdfStream>) -> Result<()> {
        let stream_id = u32::from_le_bytes(data[0..4].try_into().unwrap());

        if let Some(stream) = streams.get_mut(&stream_id) {
            let mut offset = 4;

            // Read number of samples in this chunk
            let num_samples = if offset + 1 < data.len() {
                data[offset] as usize
            } else {
                return Ok(());
            };
            offset += 1;

            let format_size = stream.info.channel_format.size_bytes();
            let channel_count = stream.info.channel_count;

            for _ in 0..num_samples {
                // Read timestamp
                if offset + 8 > data.len() {
                    break;
                }
                let timestamp = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                offset += 8;

                // Read sample data
                let mut sample = Vec::with_capacity(channel_count);

                for _ in 0..channel_count {
                    if offset + format_size > data.len() {
                        break;
                    }

                    let value = match stream.info.channel_format {
                        ChannelFormat::Float32 => {
                            let v = f32::from_le_bytes(
                                data[offset..offset + 4].try_into().unwrap(),
                            );
                            offset += 4;
                            v as f64
                        }
                        ChannelFormat::Float64 => {
                            let v = f64::from_le_bytes(
                                data[offset..offset + 8].try_into().unwrap(),
                            );
                            offset += 8;
                            v
                        }
                        ChannelFormat::Int16 => {
                            let v = i16::from_le_bytes(
                                data[offset..offset + 2].try_into().unwrap(),
                            );
                            offset += 2;
                            v as f64
                        }
                        ChannelFormat::Int32 => {
                            let v = i32::from_le_bytes(
                                data[offset..offset + 4].try_into().unwrap(),
                            );
                            offset += 4;
                            v as f64
                        }
                        _ => {
                            offset += format_size;
                            0.0
                        }
                    };

                    sample.push(value);
                }

                stream.add_sample(timestamp, sample);
            }
        }

        Ok(())
    }

    /// Parse clock offset chunk
    fn parse_clock_offset(data: &[u8], streams: &mut HashMap<u32, XdfStream>) -> Result<()> {
        if data.len() < 20 {
            return Ok(());
        }

        let stream_id = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let collection_time = f64::from_le_bytes(data[4..12].try_into().unwrap());
        let offset_value = f64::from_le_bytes(data[12..20].try_into().unwrap());

        if let Some(stream) = streams.get_mut(&stream_id) {
            stream.add_clock_offset(collection_time, offset_value);
        }

        Ok(())
    }

    /// Get number of streams
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Get all stream IDs
    pub fn stream_ids(&self) -> Vec<u32> {
        self.streams.keys().copied().collect()
    }

    /// Get stream by ID
    pub fn stream(&self, stream_id: u32) -> Option<&XdfStream> {
        self.streams.get(&stream_id)
    }

    /// Get mutable stream by ID
    pub fn stream_mut(&mut self, stream_id: u32) -> Option<&mut XdfStream> {
        self.streams.get_mut(&stream_id)
    }

    /// Find streams by type
    pub fn streams_by_type(&self, stream_type: &str) -> Vec<&XdfStream> {
        self.streams
            .values()
            .filter(|s| s.info.stream_type.to_lowercase() == stream_type.to_lowercase())
            .collect()
    }

    /// Get all marker streams
    pub fn marker_streams(&self) -> Vec<&XdfStream> {
        self.streams
            .values()
            .filter(|s| s.info.is_marker_stream())
            .collect()
    }

    /// Get file version
    pub fn version(&self) -> (u8, u8) {
        self.file_version
    }
}

impl Default for XdfFile {
    fn default() -> Self {
        Self::new()
    }
}

/// XDF file writer (simplified implementation)
pub struct XdfWriter {
    file: File,
    streams: Vec<XdfStreamInfo>,
    stream_id_counter: u32,
}

impl XdfWriter {
    /// Create a new XDF writer
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            file,
            streams: Vec::new(),
            stream_id_counter: 1,
        })
    }

    /// Write file header
    fn write_file_header(&mut self) -> Result<()> {
        // Write magic bytes
        self.file.write_all(b"XDF:")?;

        // Write file header chunk
        let header_xml = b"<?xml version=\"1.0\"?><info><version>1.0</version></info>";
        let chunk_length = (2 + header_xml.len()) as u32;

        self.file.write_all(&chunk_length.to_le_bytes())?;
        self.file
            .write_all(&(ChunkTag::FileHeader as u16).to_le_bytes())?;
        self.file.write_all(header_xml)?;

        Ok(())
    }

    /// Add a stream
    pub fn add_stream(&mut self, info: XdfStreamInfo) -> Result<u32> {
        let stream_id = self.stream_id_counter;
        self.stream_id_counter += 1;

        // Write stream header chunk
        let header_xml = format!(
            "<info><name>{}</name><type>{}</type><channel_count>{}</channel_count>\
             <nominal_srate>{}</nominal_srate><channel_format>{:?}</channel_format></info>",
            info.name, info.stream_type, info.channel_count, info.nominal_srate, info.channel_format
        );

        let chunk_data_len = 4 + header_xml.len();
        let chunk_length = (2 + chunk_data_len) as u32;

        self.file.write_all(&chunk_length.to_le_bytes())?;
        self.file
            .write_all(&(ChunkTag::StreamHeader as u16).to_le_bytes())?;
        self.file.write_all(&stream_id.to_le_bytes())?;
        self.file.write_all(header_xml.as_bytes())?;

        self.streams.push(info);
        Ok(stream_id)
    }

    /// Write samples (simplified)
    pub fn write_samples(
        &mut self,
        stream_id: u32,
        timestamps: &[f64],
        samples: &[Vec<f64>],
    ) -> Result<()> {
        if timestamps.len() != samples.len() {
            return Err(DpbError::InvalidParameter(
                "Timestamp and sample count mismatch".to_string(),
            ));
        }

        // Write in chunks of up to 255 samples
        for chunk_start in (0..samples.len()).step_by(255) {
            let chunk_end = (chunk_start + 255).min(samples.len());
            let chunk_samples = &samples[chunk_start..chunk_end];
            let chunk_timestamps = &timestamps[chunk_start..chunk_end];

            // Calculate chunk size
            let num_samples = chunk_samples.len();
            let sample_size = if !chunk_samples.is_empty() {
                chunk_samples[0].len() * 4 // Assuming float32
            } else {
                0
            };
            let chunk_data_size = 4 + 1 + num_samples * (8 + sample_size);
            let chunk_length = (2 + chunk_data_size) as u32;

            self.file.write_all(&chunk_length.to_le_bytes())?;
            self.file
                .write_all(&(ChunkTag::Samples as u16).to_le_bytes())?;
            self.file.write_all(&stream_id.to_le_bytes())?;
            self.file.write_all(&[num_samples as u8])?;

            for (timestamp, sample) in chunk_timestamps.iter().zip(chunk_samples.iter()) {
                self.file.write_all(&timestamp.to_le_bytes())?;
                for &value in sample {
                    self.file.write_all(&(value as f32).to_le_bytes())?;
                }
            }
        }

        Ok(())
    }

    /// Finalize the file
    pub fn finish(mut self) -> Result<()> {
        // Write end-of-file marker
        self.file.write_all(&0u32.to_le_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_format() {
        assert_eq!(ChannelFormat::from_str("float32"), ChannelFormat::Float32);
        assert_eq!(ChannelFormat::from_str("double64"), ChannelFormat::Float64);
        assert_eq!(ChannelFormat::Float32.size_bytes(), 4);
        assert_eq!(ChannelFormat::Float64.size_bytes(), 8);
    }

    #[test]
    fn test_stream_info() {
        let info = XdfStreamInfo::new("EEG".to_string(), "EEG".to_string(), 32, 250.0)
            .with_format(ChannelFormat::Float32);

        assert_eq!(info.name, "EEG");
        assert_eq!(info.channel_count, 32);
        assert_eq!(info.nominal_srate, 250.0);
        assert!(info.is_regular_stream());
        assert!(!info.is_marker_stream());
    }

    #[test]
    fn test_marker_stream() {
        let info = XdfStreamInfo::new(
            "Events".to_string(),
            "Markers".to_string(),
            1,
            0.0,
        );

        assert!(info.is_marker_stream());
        assert!(!info.is_regular_stream());
    }

    #[test]
    fn test_xdf_stream() {
        let info = XdfStreamInfo::new("Test".to_string(), "EEG".to_string(), 2, 100.0);
        let mut stream = XdfStream::new(1, info);

        stream.add_sample(1.0, vec![10.0, 20.0]);
        stream.add_sample(2.0, vec![15.0, 25.0]);

        assert_eq!(stream.sample_count(), 2);
        assert_eq!(stream.channel_count(), 2);

        let channel_0 = stream.channel_data(0).unwrap();
        assert_eq!(channel_0, vec![10.0, 15.0]);
    }

    #[test]
    fn test_clock_offset() {
        let info = XdfStreamInfo::new("Test".to_string(), "EEG".to_string(), 1, 100.0);
        let mut stream = XdfStream::new(1, info);

        stream.add_sample(1.0, vec![10.0]);
        stream.add_clock_offset(0.5, 0.001);

        let corrected = stream.corrected_timestamp(0);
        assert!((corrected - 1.001).abs() < 1e-6);
    }

    #[test]
    fn test_xml_extraction() {
        let xml = "<info><name>TestStream</name><type>EEG</type></info>";

        assert_eq!(
            XdfFile::extract_xml_value(xml, "name"),
            Some("TestStream".to_string())
        );
        assert_eq!(
            XdfFile::extract_xml_value(xml, "type"),
            Some("EEG".to_string())
        );
    }
}
