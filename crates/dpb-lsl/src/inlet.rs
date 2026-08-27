//! LSL inlet for receiving biosignal data.

use crate::{LslError, Result, StreamInfo};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{debug, info, warn};

/// Configuration for LSL inlet behavior.
#[derive(Debug, Clone)]
pub struct InletConfig {
    /// Maximum buffer length in seconds.
    pub max_buflen: f64,
    /// Maximum chunk length for pulling samples.
    pub max_chunklen: usize,
    /// Whether to recover from stream interruptions.
    pub recover: bool,
    /// Processing flags (internal LSL options).
    pub processing_flags: u32,
}

impl Default for InletConfig {
    fn default() -> Self {
        Self {
            max_buflen: 360.0,   // 6 minutes default buffer
            max_chunklen: 0,     // No chunking by default
            recover: true,       // Auto-recover by default
            processing_flags: 0, // No special processing
        }
    }
}

/// LSL inlet for receiving data from a stream.
///
/// Connects to an LSL outlet and pulls samples/chunks from it.
/// Supports both synchronous and buffered operation modes.
pub struct LslInlet {
    /// Stream information for this inlet.
    info: StreamInfo,
    /// Configuration options.
    config: InletConfig,
    /// Whether the inlet is currently open.
    is_open: Arc<AtomicBool>,
    /// Internal sample buffer for chunk operations.
    sample_buffer: Vec<f32>,
    /// Time correction value (offset to synchronize clocks).
    time_correction: f64,
}

impl LslInlet {
    /// Create a new inlet from stream info.
    ///
    /// # Arguments
    ///
    /// * `info` - StreamInfo from a resolved stream
    /// * `config` - Optional inlet configuration
    ///
    /// # Note
    ///
    /// This creates a mock inlet for API demonstration.
    /// Real implementation requires liblsl bindings.
    pub fn new(info: &StreamInfo, config: Option<InletConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();
        let channel_count = info.channel_count();

        info!(
            "Creating LSL inlet for stream '{}' ({} channels @ {} Hz)",
            info.name(),
            channel_count,
            info.nominal_srate()
        );

        Ok(Self {
            info: info.clone(),
            config,
            is_open: Arc::new(AtomicBool::new(false)),
            sample_buffer: vec![0.0; channel_count * 1024], // Pre-allocate buffer
            time_correction: 0.0,
        })
    }

    /// Open the inlet connection.
    ///
    /// # Arguments
    ///
    /// * `timeout_sec` - Timeout in seconds for establishing connection
    pub fn open(&mut self, timeout_sec: f64) -> Result<()> {
        if self.is_open.load(Ordering::SeqCst) {
            warn!("Inlet already open");
            return Ok(());
        }

        debug!("Opening inlet connection (timeout: {}s)", timeout_sec);

        // In real implementation, this would call liblsl to open the inlet
        // For now, we simulate successful connection
        self.is_open.store(true, Ordering::SeqCst);

        info!("Inlet opened for stream '{}'", self.info.name());

        Ok(())
    }

    /// Close the inlet connection.
    pub fn close(&mut self) {
        if self.is_open.load(Ordering::SeqCst) {
            debug!("Closing inlet connection");
            self.is_open.store(false, Ordering::SeqCst);
            info!("Inlet closed");
        }
    }

    /// Check if the inlet is open.
    pub fn is_open(&self) -> bool {
        self.is_open.load(Ordering::SeqCst)
    }

    /// Pull a single sample from the inlet.
    ///
    /// # Arguments
    ///
    /// * `timeout_sec` - Maximum time to wait for a sample
    ///
    /// # Returns
    ///
    /// Tuple of (sample_values, timestamp) or error if timeout/closed.
    pub fn pull_sample(&mut self, timeout_sec: f64) -> Result<(Vec<f32>, f64)> {
        if !self.is_open.load(Ordering::SeqCst) {
            return Err(LslError::StreamNotOpen);
        }

        // In real implementation, this calls liblsl pull_sample
        // Mock implementation returns empty sample
        let sample = vec![0.0f32; self.info.channel_count()];
        let timestamp = 0.0;

        if timeout_sec > 0.0 && timestamp == 0.0 {
            // Simulating no data available within timeout
            return Err(LslError::PullTimeout(timeout_sec));
        }

        Ok((sample, timestamp + self.time_correction))
    }

    /// Pull a chunk of samples from the inlet.
    ///
    /// # Arguments
    ///
    /// * `max_samples` - Maximum number of samples to pull
    /// * `timeout_sec` - Maximum time to wait for samples
    ///
    /// # Returns
    ///
    /// Tuple of (samples_matrix, timestamps) where samples_matrix is
    /// flattened in row-major order (sample1_ch1, sample1_ch2, ..., sample2_ch1, ...).
    pub fn pull_chunk(
        &mut self,
        max_samples: usize,
        timeout_sec: f64,
    ) -> Result<(Vec<f32>, Vec<f64>)> {
        if !self.is_open.load(Ordering::SeqCst) {
            return Err(LslError::StreamNotOpen);
        }

        let channels = self.info.channel_count();
        let buffer_size = max_samples * channels;

        // Ensure buffer is large enough
        if self.sample_buffer.len() < buffer_size {
            self.sample_buffer.resize(buffer_size, 0.0);
        }

        // In real implementation, this calls liblsl pull_chunk
        // Mock returns empty chunk
        let samples_pulled = 0usize;

        if samples_pulled == 0 && timeout_sec > 0.0 {
            return Err(LslError::PullTimeout(timeout_sec));
        }

        let samples = self.sample_buffer[..samples_pulled * channels].to_vec();
        let timestamps: Vec<f64> = (0..samples_pulled).map(|_| self.time_correction).collect();

        Ok((samples, timestamps))
    }

    /// Get the current time correction value.
    ///
    /// This is the offset between the local clock and the source clock.
    pub fn time_correction(&self) -> f64 {
        self.time_correction
    }

    /// Update the time correction by querying the source.
    ///
    /// # Arguments
    ///
    /// * `timeout_sec` - Timeout for the correction query
    pub fn update_time_correction(&mut self, _timeout_sec: f64) -> Result<f64> {
        if !self.is_open.load(Ordering::SeqCst) {
            return Err(LslError::StreamNotOpen);
        }

        // In real implementation, queries liblsl for time correction
        // Mock returns 0
        self.time_correction = 0.0;
        Ok(self.time_correction)
    }

    /// Get the stream info for this inlet.
    pub fn info(&self) -> &StreamInfo {
        &self.info
    }

    /// Get the inlet configuration.
    pub fn config(&self) -> &InletConfig {
        &self.config
    }

    /// Get the number of samples currently buffered.
    ///
    /// # Note
    ///
    /// This is an estimate and may not be exact.
    pub fn samples_available(&self) -> usize {
        if !self.is_open.load(Ordering::SeqCst) {
            return 0;
        }

        // In real implementation, queries liblsl buffer
        0
    }

    /// Check if data was lost due to buffer overflow.
    pub fn was_clock_reset(&self) -> bool {
        // In real implementation, checks liblsl state
        false
    }

    /// Set post-processing options.
    ///
    /// # Arguments
    ///
    /// * `flags` - Processing flags (clock sync, dejitter, etc.)
    pub fn set_processing_flags(&mut self, flags: u32) {
        // In real implementation, configures liblsl processing
        debug!("Setting processing flags: {}", flags);
    }
}

impl Drop for LslInlet {
    fn drop(&mut self) {
        self.close();
    }
}

/// Async inlet wrapper for use with Tokio runtime.
#[cfg(feature = "async")]
pub mod async_inlet {
    use super::*;
    use tokio::sync::mpsc;
    use tokio::task::JoinHandle;

    /// Async wrapper around LslInlet.
    pub struct AsyncLslInlet {
        /// Receiver for samples from background thread.
        rx: mpsc::Receiver<Result<(Vec<f32>, f64)>>,
        /// Handle to the background pulling task.
        _task: JoinHandle<()>,
        /// Stream info reference.
        info: StreamInfo,
        /// Flag to signal shutdown.
        shutdown: Arc<AtomicBool>,
    }

    impl AsyncLslInlet {
        /// Create a new async inlet.
        ///
        /// Spawns a background task that continuously pulls samples
        /// and sends them through a channel.
        pub async fn new(
            info: &StreamInfo,
            config: Option<InletConfig>,
            buffer_size: usize,
        ) -> Result<Self> {
            let mut inlet = LslInlet::new(info, config)?;
            inlet.open(5.0)?;

            let (tx, rx) = mpsc::channel(buffer_size);
            let shutdown = Arc::new(AtomicBool::new(false));
            let shutdown_clone = shutdown.clone();
            let info_clone = info.clone();

            let task = tokio::task::spawn_blocking(move || {
                while !shutdown_clone.load(Ordering::SeqCst) {
                    match inlet.pull_sample(0.1) {
                        Ok(sample) => {
                            if tx.blocking_send(Ok(sample)).is_err() {
                                break;
                            }
                        }
                        Err(LslError::PullTimeout(_)) => continue,
                        Err(e) => {
                            let _ = tx.blocking_send(Err(e));
                            break;
                        }
                    }
                }
            });

            Ok(Self {
                rx,
                _task: task,
                info: info_clone,
                shutdown,
            })
        }

        /// Receive the next sample asynchronously.
        pub async fn recv(&mut self) -> Option<Result<(Vec<f32>, f64)>> {
            self.rx.recv().await
        }

        /// Get the stream info.
        pub fn info(&self) -> &StreamInfo {
            &self.info
        }

        /// Signal shutdown of the background task.
        pub fn shutdown(&self) {
            self.shutdown.store(true, Ordering::SeqCst);
        }
    }

    impl Drop for AsyncLslInlet {
        fn drop(&mut self) {
            self.shutdown();
        }
    }
}
