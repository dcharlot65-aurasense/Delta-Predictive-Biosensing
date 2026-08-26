//! LSL outlet for sending biosignal and spike data.

use crate::{LslError, Result, StreamInfo};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// LSL outlet for streaming data to the network.
///
/// Creates an LSL stream that other applications can connect to
/// for receiving biosignal data, spike trains, or markers.
pub struct LslOutlet {
    /// Stream information for this outlet.
    info: StreamInfo,
    /// Whether the outlet is currently active.
    is_active: Arc<AtomicBool>,
    /// Number of consumers currently connected.
    consumer_count: usize,
}

impl LslOutlet {
    /// Create a new outlet with the given stream info.
    ///
    /// # Arguments
    ///
    /// * `info` - Stream metadata for this outlet
    /// * `chunk_size` - Size of data chunks (0 for default)
    /// * `max_buffered` - Maximum seconds of data to buffer
    ///
    /// # Note
    ///
    /// This creates a mock outlet for API demonstration.
    /// Real implementation requires liblsl bindings.
    pub fn new(info: StreamInfo, chunk_size: usize, max_buffered: f64) -> Result<Self> {
        info!(
            "Creating LSL outlet '{}' ({} channels @ {} Hz)",
            info.name(),
            info.channel_count(),
            info.nominal_srate()
        );

        debug!(
            "Outlet config: chunk_size={}, max_buffered={}s",
            chunk_size, max_buffered
        );

        Ok(Self {
            info,
            is_active: Arc::new(AtomicBool::new(true)),
            consumer_count: 0,
        })
    }

    /// Push a single sample to the outlet.
    ///
    /// # Arguments
    ///
    /// * `data` - Sample data (must match channel count)
    /// * `timestamp` - LSL timestamp (0 for automatic)
    /// * `pushthrough` - Whether to push immediately vs. buffer
    pub fn push_sample(&mut self, data: &[f32], timestamp: f64, pushthrough: bool) -> Result<()> {
        if !self.is_active.load(Ordering::SeqCst) {
            return Err(LslError::StreamNotOpen);
        }

        if data.len() != self.info.channel_count() {
            return Err(LslError::ChannelMismatch {
                expected: self.info.channel_count(),
                actual: data.len(),
            });
        }

        // In real implementation, this calls liblsl push_sample
        // Mock implementation just validates input
        debug!(
            "Pushing sample: {} channels, timestamp={}, pushthrough={}",
            data.len(),
            timestamp,
            pushthrough
        );

        Ok(())
    }

    /// Push a chunk of samples to the outlet.
    ///
    /// # Arguments
    ///
    /// * `data` - Flattened sample data (row-major: sample1_ch1, sample1_ch2, ...)
    /// * `timestamps` - Timestamps for each sample (or empty for automatic)
    /// * `pushthrough` - Whether to push immediately
    pub fn push_chunk(
        &mut self,
        data: &[f32],
        timestamps: &[f64],
        pushthrough: bool,
    ) -> Result<()> {
        if !self.is_active.load(Ordering::SeqCst) {
            return Err(LslError::StreamNotOpen);
        }

        let channels = self.info.channel_count();
        if !data.len().is_multiple_of(channels) {
            return Err(LslError::InvalidConfig(format!(
                "Data length {} is not divisible by channel count {}",
                data.len(),
                channels
            )));
        }

        let num_samples = data.len() / channels;

        if !timestamps.is_empty() && timestamps.len() != num_samples {
            return Err(LslError::InvalidConfig(format!(
                "Timestamp count {} doesn't match sample count {}",
                timestamps.len(),
                num_samples
            )));
        }

        // In real implementation, this calls liblsl push_chunk
        debug!(
            "Pushing chunk: {} samples, {} channels, pushthrough={}",
            num_samples, channels, pushthrough
        );

        Ok(())
    }

    /// Push a marker/event string.
    ///
    /// # Arguments
    ///
    /// * `marker` - Marker string to push
    /// * `timestamp` - LSL timestamp (0 for automatic)
    pub fn push_marker(&mut self, marker: &str, timestamp: f64) -> Result<()> {
        if !self.is_active.load(Ordering::SeqCst) {
            return Err(LslError::StreamNotOpen);
        }

        // Markers typically use string format streams
        debug!("Pushing marker: '{}' at timestamp {}", marker, timestamp);

        Ok(())
    }

    /// Check if anyone is currently connected to this outlet.
    pub fn have_consumers(&self) -> bool {
        self.consumer_count > 0
    }

    /// Wait until at least one consumer connects.
    ///
    /// # Arguments
    ///
    /// * `timeout_sec` - Maximum time to wait
    ///
    /// # Returns
    ///
    /// True if a consumer connected, false if timeout.
    pub fn wait_for_consumers(&mut self, timeout_sec: f64) -> bool {
        if !self.is_active.load(Ordering::SeqCst) {
            return false;
        }

        // In real implementation, this blocks until consumer connects or timeout
        debug!(
            "Waiting for consumers (timeout: {}s)",
            timeout_sec
        );

        // Mock: immediate return
        false
    }

    /// Get the stream info for this outlet.
    pub fn info(&self) -> &StreamInfo {
        &self.info
    }

    /// Check if the outlet is active.
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::SeqCst)
    }

    /// Deactivate the outlet.
    pub fn close(&mut self) {
        if self.is_active.load(Ordering::SeqCst) {
            debug!("Closing outlet");
            self.is_active.store(false, Ordering::SeqCst);
            info!("Outlet '{}' closed", self.info.name());
        }
    }
}

impl Drop for LslOutlet {
    fn drop(&mut self) {
        self.close();
    }
}

/// Specialized outlet for spike train data.
///
/// Optimized for streaming discrete spike events with
/// channel information and optional amplitude data.
pub struct SpikeOutlet {
    /// Underlying LSL outlet.
    outlet: LslOutlet,
    /// Whether to include spike amplitudes.
    include_amplitudes: bool,
}

impl SpikeOutlet {
    /// Create a new spike outlet.
    ///
    /// # Arguments
    ///
    /// * `name` - Stream name
    /// * `source_id` - Unique source identifier
    /// * `num_channels` - Number of spike channels
    /// * `include_amplitudes` - Whether to include amplitude information
    pub fn new(
        name: impl Into<String>,
        source_id: impl Into<String>,
        num_channels: usize,
        include_amplitudes: bool,
    ) -> Result<Self> {
        use crate::{stream_types, ChannelFormat};

        // Spike streams are irregular rate with 2 or 3 values per event:
        // (channel_id, timestamp[, amplitude])
        let values_per_spike = if include_amplitudes { 3 } else { 2 };

        let info = StreamInfo::new(
            name,
            stream_types::SPIKES,
            values_per_spike,
            0.0, // Irregular rate
            ChannelFormat::Float32,
            source_id,
        )?;

        let outlet = LslOutlet::new(info, 0, 360.0)?;

        Ok(Self {
            outlet,
            include_amplitudes,
        })
    }

    /// Push a single spike event.
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel index where spike occurred
    /// * `timestamp` - Time of the spike
    /// * `amplitude` - Optional amplitude (only used if include_amplitudes is true)
    pub fn push_spike(
        &mut self,
        channel: usize,
        timestamp: f64,
        amplitude: Option<f32>,
    ) -> Result<()> {
        let data: Vec<f32> = if self.include_amplitudes {
            vec![
                channel as f32,
                timestamp as f32,
                amplitude.unwrap_or(1.0),
            ]
        } else {
            vec![channel as f32, timestamp as f32]
        };

        self.outlet.push_sample(&data, timestamp, true)
    }

    /// Push multiple spike events.
    ///
    /// # Arguments
    ///
    /// * `spikes` - Iterator of (channel, timestamp, amplitude) tuples
    pub fn push_spikes<I>(&mut self, spikes: I) -> Result<()>
    where
        I: IntoIterator<Item = (usize, f64, Option<f32>)>,
    {
        for (channel, timestamp, amplitude) in spikes {
            self.push_spike(channel, timestamp, amplitude)?;
        }
        Ok(())
    }

    /// Get the underlying outlet's stream info.
    pub fn info(&self) -> &StreamInfo {
        self.outlet.info()
    }

    /// Check if the outlet is active.
    pub fn is_active(&self) -> bool {
        self.outlet.is_active()
    }

    /// Close the outlet.
    pub fn close(&mut self) {
        self.outlet.close();
    }
}

/// Builder for creating outlets with common configurations.
pub struct OutletBuilder {
    name: String,
    stream_type: String,
    channel_count: usize,
    sample_rate: f64,
    source_id: String,
    chunk_size: usize,
    max_buffered: f64,
}

impl OutletBuilder {
    /// Create a new outlet builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stream_type: "Signal".to_string(),
            channel_count: 1,
            sample_rate: 0.0,
            source_id: String::new(),
            chunk_size: 0,
            max_buffered: 360.0,
        }
    }

    /// Set the stream type.
    pub fn stream_type(mut self, stream_type: impl Into<String>) -> Self {
        self.stream_type = stream_type.into();
        self
    }

    /// Set the number of channels.
    pub fn channels(mut self, count: usize) -> Self {
        self.channel_count = count;
        self
    }

    /// Set the sample rate.
    pub fn sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set the source ID.
    pub fn source_id(mut self, id: impl Into<String>) -> Self {
        self.source_id = id.into();
        self
    }

    /// Set the chunk size.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the maximum buffer time.
    pub fn max_buffered(mut self, seconds: f64) -> Self {
        self.max_buffered = seconds;
        self
    }

    /// Build the outlet.
    pub fn build(self) -> Result<LslOutlet> {
        use crate::ChannelFormat;

        let info = StreamInfo::new(
            self.name,
            self.stream_type,
            self.channel_count,
            self.sample_rate,
            ChannelFormat::Float32,
            self.source_id,
        )?;

        LslOutlet::new(info, self.chunk_size, self.max_buffered)
    }
}
