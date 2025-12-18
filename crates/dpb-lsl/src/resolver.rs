//! LSL stream discovery and resolution.

use crate::{ChannelFormat, LslError, Result, StreamInfo};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Stream resolver for discovering LSL streams on the network.
///
/// Provides methods to find streams by name, type, or property queries.
pub struct StreamResolver {
    /// Default timeout for resolution operations.
    default_timeout: Duration,
}

impl StreamResolver {
    /// Create a new stream resolver.
    pub fn new() -> Self {
        Self {
            default_timeout: Duration::from_secs(5),
        }
    }

    /// Create a resolver with custom default timeout.
    pub fn with_timeout(timeout_sec: f64) -> Self {
        Self {
            default_timeout: Duration::from_secs_f64(timeout_sec),
        }
    }

    /// Resolve all streams on the network.
    ///
    /// # Arguments
    ///
    /// * `wait_time` - Minimum time to wait for responses (seconds)
    ///
    /// # Returns
    ///
    /// Vector of discovered stream infos.
    pub fn resolve_all(&self, wait_time: f64) -> Result<Vec<StreamInfo>> {
        info!("Resolving all LSL streams (wait time: {}s)", wait_time);

        // In real implementation, this calls liblsl resolve_streams
        // Mock returns empty list
        let streams = Vec::new();

        debug!("Found {} streams", streams.len());
        Ok(streams)
    }

    /// Resolve streams by type.
    ///
    /// # Arguments
    ///
    /// * `stream_type` - Type to search for (e.g., "EEG", "ECG")
    /// * `timeout_sec` - Maximum time to wait
    ///
    /// # Returns
    ///
    /// Vector of matching stream infos.
    pub fn resolve_by_type(&self, stream_type: &str, timeout_sec: f64) -> Result<Vec<StreamInfo>> {
        info!(
            "Resolving streams of type '{}' (timeout: {}s)",
            stream_type, timeout_sec
        );

        // In real implementation, calls liblsl resolve_byprop
        let streams = Vec::new();

        debug!("Found {} streams of type '{}'", streams.len(), stream_type);
        Ok(streams)
    }

    /// Resolve streams by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Name to search for
    /// * `timeout_sec` - Maximum time to wait
    ///
    /// # Returns
    ///
    /// Vector of matching stream infos.
    pub fn resolve_by_name(&self, name: &str, timeout_sec: f64) -> Result<Vec<StreamInfo>> {
        info!(
            "Resolving streams with name '{}' (timeout: {}s)",
            name, timeout_sec
        );

        // In real implementation, calls liblsl resolve_byprop
        let streams = Vec::new();

        debug!("Found {} streams with name '{}'", streams.len(), name);
        Ok(streams)
    }

    /// Resolve streams by property.
    ///
    /// # Arguments
    ///
    /// * `property` - Property name to match
    /// * `value` - Property value to match
    /// * `timeout_sec` - Maximum time to wait
    pub fn resolve_by_property(
        &self,
        property: &str,
        value: &str,
        timeout_sec: f64,
    ) -> Result<Vec<StreamInfo>> {
        info!(
            "Resolving streams with {}='{}' (timeout: {}s)",
            property, value, timeout_sec
        );

        // In real implementation, calls liblsl resolve_byprop
        let streams = Vec::new();

        debug!(
            "Found {} streams with {}='{}'",
            streams.len(),
            property,
            value
        );
        Ok(streams)
    }

    /// Resolve streams using an XPath-style predicate.
    ///
    /// # Arguments
    ///
    /// * `pred` - XPath predicate string
    /// * `timeout_sec` - Maximum time to wait
    ///
    /// # Example Predicates
    ///
    /// - `name='BioAmp'`
    /// - `type='EEG' and channel_count>4`
    /// - `contains(name,'EEG')`
    pub fn resolve_by_predicate(&self, pred: &str, timeout_sec: f64) -> Result<Vec<StreamInfo>> {
        info!(
            "Resolving streams with predicate '{}' (timeout: {}s)",
            pred, timeout_sec
        );

        // In real implementation, calls liblsl resolve_bypred
        let streams = Vec::new();

        debug!("Found {} streams matching predicate", streams.len());
        Ok(streams)
    }

    /// Wait for a specific stream to appear.
    ///
    /// # Arguments
    ///
    /// * `name` - Stream name to wait for
    /// * `stream_type` - Stream type to match
    /// * `timeout_sec` - Maximum time to wait
    ///
    /// # Returns
    ///
    /// StreamInfo if found, error if timeout.
    pub fn wait_for_stream(
        &self,
        name: &str,
        stream_type: &str,
        timeout_sec: f64,
    ) -> Result<StreamInfo> {
        info!(
            "Waiting for stream '{}' of type '{}' (timeout: {}s)",
            name, stream_type, timeout_sec
        );

        let start = Instant::now();
        let timeout = Duration::from_secs_f64(timeout_sec);

        while start.elapsed() < timeout {
            // Check for matching streams
            let streams = self.resolve_by_name(name, 0.1)?;

            for stream in streams {
                if stream.stream_type() == stream_type {
                    info!("Found stream '{}' of type '{}'", name, stream_type);
                    return Ok(stream);
                }
            }

            // Brief pause before retrying
            std::thread::sleep(Duration::from_millis(100));
        }

        Err(LslError::StreamNotFound {
            name: name.to_string(),
            timeout_sec,
        })
    }

    /// Get the default timeout duration.
    pub fn default_timeout(&self) -> Duration {
        self.default_timeout
    }
}

impl Default for StreamResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Continuous stream watcher that monitors for stream changes.
///
/// Useful for applications that need to react to streams
/// appearing or disappearing on the network.
pub struct StreamWatcher {
    /// Resolver for periodic checks.
    resolver: StreamResolver,
    /// Currently known streams.
    known_streams: Vec<StreamInfo>,
    /// Check interval.
    check_interval: Duration,
}

impl StreamWatcher {
    /// Create a new stream watcher.
    ///
    /// # Arguments
    ///
    /// * `check_interval_sec` - How often to check for changes
    pub fn new(check_interval_sec: f64) -> Self {
        Self {
            resolver: StreamResolver::new(),
            known_streams: Vec::new(),
            check_interval: Duration::from_secs_f64(check_interval_sec),
        }
    }

    /// Check for stream changes.
    ///
    /// # Returns
    ///
    /// Tuple of (new_streams, removed_streams).
    pub fn check(&mut self) -> Result<(Vec<StreamInfo>, Vec<StreamInfo>)> {
        let current = self.resolver.resolve_all(0.5)?;

        let new_streams: Vec<StreamInfo> = current
            .iter()
            .filter(|s| {
                !self
                    .known_streams
                    .iter()
                    .any(|k| k.source_id() == s.source_id())
            })
            .cloned()
            .collect();

        let removed_streams: Vec<StreamInfo> = self
            .known_streams
            .iter()
            .filter(|k| {
                !current
                    .iter()
                    .any(|s| s.source_id() == k.source_id())
            })
            .cloned()
            .collect();

        self.known_streams = current;

        if !new_streams.is_empty() {
            info!("New streams detected: {}", new_streams.len());
        }
        if !removed_streams.is_empty() {
            info!("Streams removed: {}", removed_streams.len());
        }

        Ok((new_streams, removed_streams))
    }

    /// Get currently known streams.
    pub fn known_streams(&self) -> &[StreamInfo] {
        &self.known_streams
    }

    /// Get the check interval.
    pub fn check_interval(&self) -> Duration {
        self.check_interval
    }
}

/// Helper to create standard biosignal stream queries.
pub mod queries {
    /// Query for EEG streams.
    pub fn eeg() -> &'static str {
        "type='EEG'"
    }

    /// Query for ECG streams.
    pub fn ecg() -> &'static str {
        "type='ECG'"
    }

    /// Query for EMG streams.
    pub fn emg() -> &'static str {
        "type='EMG'"
    }

    /// Query for marker streams.
    pub fn markers() -> &'static str {
        "type='Markers'"
    }

    /// Query for spike streams (DPB custom).
    pub fn spikes() -> &'static str {
        "type='Spikes'"
    }

    /// Query for streams from a specific host.
    pub fn from_host(hostname: &str) -> String {
        format!("hostname='{}'", hostname)
    }

    /// Query for streams with minimum channel count.
    pub fn min_channels(count: usize) -> String {
        format!("channel_count>={}", count)
    }

    /// Query for streams with specific sample rate.
    pub fn sample_rate(rate: f64, tolerance: f64) -> String {
        format!(
            "nominal_srate>={} and nominal_srate<={}",
            rate - tolerance,
            rate + tolerance
        )
    }
}
