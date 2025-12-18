//! LSL stream information and metadata.

use crate::{ChannelFormat, LslError, Result};
use serde::{Deserialize, Serialize};

/// Metadata about an LSL stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInfo {
    /// Stream name (human-readable identifier).
    name: String,
    /// Stream type (e.g., "EEG", "ECG", "Markers").
    stream_type: String,
    /// Number of channels in the stream.
    channel_count: usize,
    /// Nominal sample rate in Hz (0 for irregular rate).
    nominal_srate: f64,
    /// Data format of each sample.
    channel_format: ChannelFormat,
    /// Unique identifier for this stream instance.
    source_id: String,
    /// Hostname of the source machine.
    hostname: Option<String>,
    /// Version of the stream format.
    version: u32,
    /// Session ID for multi-stream synchronization.
    session_id: Option<String>,
    /// Extended XML metadata.
    xml_desc: Option<String>,
}

impl StreamInfo {
    /// Create a new StreamInfo for outlet creation.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable name for the stream
    /// * `stream_type` - Type of data (e.g., "EEG", "ECG", "Markers")
    /// * `channel_count` - Number of channels
    /// * `nominal_srate` - Sample rate in Hz (0 for irregular)
    /// * `channel_format` - Data format
    /// * `source_id` - Unique identifier for this source
    pub fn new(
        name: impl Into<String>,
        stream_type: impl Into<String>,
        channel_count: usize,
        nominal_srate: f64,
        channel_format: ChannelFormat,
        source_id: impl Into<String>,
    ) -> Result<Self> {
        let name = name.into();
        let stream_type = stream_type.into();
        let source_id = source_id.into();

        if name.is_empty() {
            return Err(LslError::InvalidConfig("Stream name cannot be empty".into()));
        }

        if channel_count == 0 {
            return Err(LslError::InvalidConfig(
                "Channel count must be positive".into(),
            ));
        }

        if nominal_srate < 0.0 {
            return Err(LslError::InvalidConfig(
                "Sample rate cannot be negative".into(),
            ));
        }

        Ok(Self {
            name,
            stream_type,
            channel_count,
            nominal_srate,
            channel_format,
            source_id,
            hostname: None,
            version: 1,
            session_id: None,
            xml_desc: None,
        })
    }

    /// Create StreamInfo from discovered stream (internal use).
    #[doc(hidden)]
    pub fn from_resolved(
        name: String,
        stream_type: String,
        channel_count: usize,
        nominal_srate: f64,
        channel_format: ChannelFormat,
        source_id: String,
        hostname: Option<String>,
    ) -> Self {
        Self {
            name,
            stream_type,
            channel_count,
            nominal_srate,
            channel_format,
            source_id,
            hostname,
            version: 1,
            session_id: None,
            xml_desc: None,
        }
    }

    /// Get the stream name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the stream type.
    pub fn stream_type(&self) -> &str {
        &self.stream_type
    }

    /// Get the number of channels.
    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    /// Get the nominal sample rate in Hz.
    pub fn nominal_srate(&self) -> f64 {
        self.nominal_srate
    }

    /// Check if this is an irregular-rate stream.
    pub fn is_irregular_rate(&self) -> bool {
        self.nominal_srate == 0.0
    }

    /// Get the channel format.
    pub fn channel_format(&self) -> ChannelFormat {
        self.channel_format
    }

    /// Get the source ID.
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// Get the hostname if available.
    pub fn hostname(&self) -> Option<&str> {
        self.hostname.as_deref()
    }

    /// Get the protocol version.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Set the session ID for multi-stream synchronization.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Get the session ID if set.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Set extended XML description.
    pub fn with_xml_desc(mut self, xml: impl Into<String>) -> Self {
        self.xml_desc = Some(xml.into());
        self
    }

    /// Get the XML description if set.
    pub fn xml_desc(&self) -> Option<&str> {
        self.xml_desc.as_deref()
    }

    /// Get bytes per sample based on channel format.
    pub fn bytes_per_sample(&self) -> usize {
        self.channel_format.bytes_per_sample() * self.channel_count
    }
}

/// Builder for StreamInfo with extended metadata.
#[derive(Debug, Default)]
pub struct StreamInfoBuilder {
    name: Option<String>,
    stream_type: Option<String>,
    channel_count: Option<usize>,
    nominal_srate: Option<f64>,
    channel_format: Option<ChannelFormat>,
    source_id: Option<String>,
    session_id: Option<String>,
    xml_desc: Option<String>,
}

impl StreamInfoBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the stream name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the stream type.
    pub fn stream_type(mut self, stream_type: impl Into<String>) -> Self {
        self.stream_type = Some(stream_type.into());
        self
    }

    /// Set the channel count.
    pub fn channel_count(mut self, count: usize) -> Self {
        self.channel_count = Some(count);
        self
    }

    /// Set the nominal sample rate.
    pub fn nominal_srate(mut self, srate: f64) -> Self {
        self.nominal_srate = Some(srate);
        self
    }

    /// Set the channel format.
    pub fn channel_format(mut self, format: ChannelFormat) -> Self {
        self.channel_format = Some(format);
        self
    }

    /// Set the source ID.
    pub fn source_id(mut self, id: impl Into<String>) -> Self {
        self.source_id = Some(id.into());
        self
    }

    /// Set the session ID.
    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    /// Set the XML description.
    pub fn xml_desc(mut self, xml: impl Into<String>) -> Self {
        self.xml_desc = Some(xml.into());
        self
    }

    /// Build the StreamInfo.
    pub fn build(self) -> Result<StreamInfo> {
        let name = self
            .name
            .ok_or_else(|| LslError::InvalidConfig("Stream name is required".into()))?;
        let stream_type = self
            .stream_type
            .ok_or_else(|| LslError::InvalidConfig("Stream type is required".into()))?;
        let channel_count = self
            .channel_count
            .ok_or_else(|| LslError::InvalidConfig("Channel count is required".into()))?;
        let nominal_srate = self.nominal_srate.unwrap_or(0.0);
        let channel_format = self.channel_format.unwrap_or(ChannelFormat::Float32);
        let source_id = self.source_id.unwrap_or_default();

        let mut info =
            StreamInfo::new(name, stream_type, channel_count, nominal_srate, channel_format, source_id)?;

        if let Some(session_id) = self.session_id {
            info = info.with_session_id(session_id);
        }

        if let Some(xml) = self.xml_desc {
            info = info.with_xml_desc(xml);
        }

        Ok(info)
    }
}

/// Channel metadata for a single channel in a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelInfo {
    /// Channel label.
    pub label: String,
    /// Physical unit of measurement.
    pub unit: String,
    /// Type of channel (e.g., "EEG", "AUX").
    pub channel_type: String,
}

impl ChannelInfo {
    /// Create new channel info.
    pub fn new(label: impl Into<String>, unit: impl Into<String>, channel_type: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            unit: unit.into(),
            channel_type: channel_type.into(),
        }
    }

    /// Create channel info for an EEG channel.
    pub fn eeg(label: impl Into<String>) -> Self {
        Self::new(label, "µV", "EEG")
    }

    /// Create channel info for an ECG channel.
    pub fn ecg(label: impl Into<String>) -> Self {
        Self::new(label, "mV", "ECG")
    }

    /// Create channel info for a marker channel.
    pub fn marker(label: impl Into<String>) -> Self {
        Self::new(label, "", "Marker")
    }
}
