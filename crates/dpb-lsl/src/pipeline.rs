//! Real-time encoding pipeline for LSL streams.

use crate::{LslError, LslInlet, LslOutlet, Result, StreamInfo, StreamResolver};
use crossbeam_channel::{bounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Configuration for the encoding pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Input stream name pattern.
    pub input_stream: String,
    /// Input stream type.
    pub input_type: String,
    /// Output stream name.
    pub output_name: String,
    /// Output source ID.
    pub output_source_id: String,
    /// Encoder type to use.
    pub encoder_type: EncoderType,
    /// Encoder-specific parameters.
    pub encoder_params: EncoderParams,
    /// Buffer size in samples.
    pub buffer_size: usize,
    /// Enable adaptive thresholds.
    pub adaptive: bool,
    /// Log statistics interval (seconds, 0 to disable).
    pub stats_interval: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            input_stream: String::new(),
            input_type: "EEG".to_string(),
            output_name: "DPB_Spikes".to_string(),
            output_source_id: "dpb_pipeline".to_string(),
            encoder_type: EncoderType::LevelCrossing,
            encoder_params: EncoderParams::default(),
            buffer_size: 1024,
            adaptive: true,
            stats_interval: 60.0,
        }
    }
}

/// Type of encoder to use in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncoderType {
    /// Level crossing detector.
    LevelCrossing,
    /// Delta modulation encoder.
    Delta,
    /// Temporal contrast encoder.
    TemporalContrast,
    /// Send-on-delta encoder.
    SendOnDelta,
}

/// Parameters for encoders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderParams {
    /// Threshold for level crossing / delta encoders.
    pub threshold: f32,
    /// Number of quantization levels.
    pub num_levels: usize,
    /// Reference period for refractory behavior.
    pub refractory_period: f32,
    /// Adaptation rate for adaptive encoders.
    pub adaptation_rate: f32,
}

impl Default for EncoderParams {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            num_levels: 64,
            refractory_period: 0.001,
            adaptation_rate: 0.01,
        }
    }
}

/// Real-time encoding pipeline.
///
/// Connects an LSL input stream to an encoder and outputs
/// spike trains over a new LSL stream.
pub struct EncodingPipeline {
    /// Pipeline configuration.
    config: PipelineConfig,
    /// Running flag.
    running: Arc<AtomicBool>,
    /// Worker thread handle.
    worker: Option<JoinHandle<()>>,
    /// Statistics receiver.
    stats_rx: Option<Receiver<PipelineStats>>,
}

impl EncodingPipeline {
    /// Create a new encoding pipeline.
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            worker: None,
            stats_rx: None,
        }
    }

    /// Start the pipeline.
    ///
    /// # Arguments
    ///
    /// * `timeout_sec` - Timeout for finding the input stream
    pub fn start(&mut self, timeout_sec: f64) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            warn!("Pipeline already running");
            return Ok(());
        }

        info!(
            "Starting encoding pipeline for stream '{}'",
            self.config.input_stream
        );

        // Find input stream
        let resolver = StreamResolver::new();
        let input_info = resolver.wait_for_stream(
            &self.config.input_stream,
            &self.config.input_type,
            timeout_sec,
        )?;

        // Create channels for stats
        let (stats_tx, stats_rx) = bounded(16);
        self.stats_rx = Some(stats_rx);

        // Clone config and flags for worker thread
        let config = self.config.clone();
        let running = self.running.clone();

        running.store(true, Ordering::SeqCst);

        // Spawn worker thread
        let worker = thread::spawn(move || {
            if let Err(e) = run_pipeline_worker(input_info, config, running.clone(), stats_tx) {
                error!("Pipeline worker error: {}", e);
            }
            running.store(false, Ordering::SeqCst);
        });

        self.worker = Some(worker);
        info!("Pipeline started");

        Ok(())
    }

    /// Stop the pipeline.
    pub fn stop(&mut self) {
        if self.running.load(Ordering::SeqCst) {
            info!("Stopping pipeline");
            self.running.store(false, Ordering::SeqCst);

            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }

            info!("Pipeline stopped");
        }
    }

    /// Check if the pipeline is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the latest statistics.
    pub fn get_stats(&self) -> Option<PipelineStats> {
        self.stats_rx.as_ref().and_then(|rx| rx.try_recv().ok())
    }

    /// Get the pipeline configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

impl Drop for EncodingPipeline {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total samples processed.
    pub samples_processed: u64,
    /// Total spikes generated.
    pub spikes_generated: u64,
    /// Current spike rate (spikes/second).
    pub spike_rate: f64,
    /// Samples dropped due to buffer overflow.
    pub samples_dropped: u64,
    /// Average processing latency (seconds).
    pub avg_latency: f64,
    /// Maximum processing latency (seconds).
    pub max_latency: f64,
    /// Pipeline uptime (seconds).
    pub uptime: f64,
}

/// Internal worker function for the pipeline.
fn run_pipeline_worker(
    input_info: StreamInfo,
    config: PipelineConfig,
    running: Arc<AtomicBool>,
    stats_tx: Sender<PipelineStats>,
) -> Result<()> {
    use crate::{stream_types, ChannelFormat, outlet::SpikeOutlet};

    // Create inlet
    let mut inlet = LslInlet::new(&input_info, None)?;
    inlet.open(5.0)?;

    // Create spike outlet
    let mut outlet = SpikeOutlet::new(
        &config.output_name,
        &config.output_source_id,
        input_info.channel_count(),
        true, // Include amplitudes
    )?;

    // Initialize encoder state per channel
    let num_channels = input_info.channel_count();
    let mut thresholds: Vec<f32> = vec![config.encoder_params.threshold; num_channels];
    let mut last_values: Vec<f32> = vec![0.0; num_channels];
    let mut levels: Vec<i32> = vec![0; num_channels];

    // Statistics
    let mut stats = PipelineStats::default();
    let start_time = std::time::Instant::now();
    let mut last_stats_time = start_time;

    info!(
        "Pipeline worker running: {} channels, {:?} encoder",
        num_channels, config.encoder_type
    );

    while running.load(Ordering::SeqCst) {
        // Pull chunk of samples
        match inlet.pull_chunk(config.buffer_size, 0.1) {
            Ok((samples, timestamps)) => {
                if samples.is_empty() {
                    continue;
                }

                let num_samples = samples.len() / num_channels;
                stats.samples_processed += num_samples as u64;

                // Process each sample
                for i in 0..num_samples {
                    let timestamp = if timestamps.len() > i {
                        timestamps[i]
                    } else {
                        0.0
                    };

                    for ch in 0..num_channels {
                        let value = samples[i * num_channels + ch];
                        let threshold = thresholds[ch];

                        // Encode based on selected encoder type
                        let spike = match config.encoder_type {
                            EncoderType::LevelCrossing => {
                                // Level crossing detection
                                let crossed_up =
                                    last_values[ch] < threshold && value >= threshold;
                                let crossed_down =
                                    last_values[ch] > -threshold && value <= -threshold;

                                if crossed_up || crossed_down {
                                    Some(if crossed_up { 1.0f32 } else { -1.0f32 })
                                } else {
                                    None
                                }
                            }
                            EncoderType::Delta => {
                                // Delta encoding
                                let delta = value - last_values[ch];
                                if delta.abs() >= threshold {
                                    Some(delta.signum())
                                } else {
                                    None
                                }
                            }
                            EncoderType::TemporalContrast => {
                                // Temporal contrast (rate of change)
                                let contrast = (value - last_values[ch]).abs();
                                if contrast >= threshold {
                                    Some(contrast)
                                } else {
                                    None
                                }
                            }
                            EncoderType::SendOnDelta => {
                                // Send-on-delta with quantization
                                let step = threshold / config.encoder_params.num_levels as f32;
                                let new_level = (value / step).round() as i32;
                                if new_level != levels[ch] {
                                    levels[ch] = new_level;
                                    Some(value)
                                } else {
                                    None
                                }
                            }
                        };

                        // Output spike if detected
                        if let Some(amplitude) = spike {
                            if let Err(e) = outlet.push_spike(ch, timestamp, Some(amplitude)) {
                                debug!("Failed to push spike: {}", e);
                            } else {
                                stats.spikes_generated += 1;
                            }
                        }

                        // Update state
                        last_values[ch] = value;

                        // Adaptive threshold update
                        if config.adaptive {
                            let alpha = config.encoder_params.adaptation_rate;
                            thresholds[ch] = thresholds[ch] * (1.0 - alpha)
                                + value.abs() * alpha;
                        }
                    }
                }
            }
            Err(LslError::PullTimeout(_)) => {
                // No data available, continue
                continue;
            }
            Err(e) => {
                error!("Error pulling samples: {}", e);
                if !e.is_recoverable() {
                    return Err(e);
                }
            }
        }

        // Update and send statistics periodically
        if config.stats_interval > 0.0 {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(last_stats_time);

            if elapsed >= Duration::from_secs_f64(config.stats_interval) {
                stats.uptime = now.duration_since(start_time).as_secs_f64();
                stats.spike_rate = stats.spikes_generated as f64 / stats.uptime;

                let _ = stats_tx.try_send(stats.clone());
                last_stats_time = now;
            }
        }
    }

    info!("Pipeline worker finished");
    Ok(())
}

/// Builder for encoding pipelines.
pub struct PipelineBuilder {
    config: PipelineConfig,
}

impl PipelineBuilder {
    /// Create a new pipeline builder.
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }

    /// Set the input stream name.
    pub fn input_stream(mut self, name: impl Into<String>) -> Self {
        self.config.input_stream = name.into();
        self
    }

    /// Set the input stream type.
    pub fn input_type(mut self, stream_type: impl Into<String>) -> Self {
        self.config.input_type = stream_type.into();
        self
    }

    /// Set the output stream name.
    pub fn output_name(mut self, name: impl Into<String>) -> Self {
        self.config.output_name = name.into();
        self
    }

    /// Set the output source ID.
    pub fn output_source_id(mut self, id: impl Into<String>) -> Self {
        self.config.output_source_id = id.into();
        self
    }

    /// Set the encoder type.
    pub fn encoder(mut self, encoder_type: EncoderType) -> Self {
        self.config.encoder_type = encoder_type;
        self
    }

    /// Set the threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.config.encoder_params.threshold = threshold;
        self
    }

    /// Set the number of quantization levels.
    pub fn num_levels(mut self, levels: usize) -> Self {
        self.config.encoder_params.num_levels = levels;
        self
    }

    /// Enable or disable adaptive thresholds.
    pub fn adaptive(mut self, enabled: bool) -> Self {
        self.config.adaptive = enabled;
        self
    }

    /// Set the adaptation rate.
    pub fn adaptation_rate(mut self, rate: f32) -> Self {
        self.config.encoder_params.adaptation_rate = rate;
        self
    }

    /// Set the buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Set the statistics reporting interval.
    pub fn stats_interval(mut self, seconds: f64) -> Self {
        self.config.stats_interval = seconds;
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> EncodingPipeline {
        EncodingPipeline::new(self.config)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.input_stream, "");
        assert_eq!(config.input_type, "EEG");
        assert_eq!(config.output_name, "DPB_Spikes");
        assert_eq!(config.output_source_id, "dpb_pipeline");
        assert_eq!(config.encoder_type, EncoderType::LevelCrossing);
        assert_eq!(config.buffer_size, 1024);
        assert!(config.adaptive);
        assert_eq!(config.stats_interval, 60.0);
    }

    #[test]
    fn test_encoder_params_default() {
        let params = EncoderParams::default();
        assert_eq!(params.threshold, 0.1);
        assert_eq!(params.num_levels, 64);
        assert_eq!(params.refractory_period, 0.001);
        assert_eq!(params.adaptation_rate, 0.01);
    }

    #[test]
    fn test_encoder_type_equality() {
        assert_eq!(EncoderType::LevelCrossing, EncoderType::LevelCrossing);
        assert_ne!(EncoderType::LevelCrossing, EncoderType::Delta);
        assert_ne!(EncoderType::TemporalContrast, EncoderType::SendOnDelta);
    }

    #[test]
    fn test_encoder_type_clone() {
        let encoder = EncoderType::Delta;
        let cloned = encoder.clone();
        assert_eq!(encoder, cloned);
    }

    #[test]
    fn test_encoder_type_copy() {
        let encoder = EncoderType::TemporalContrast;
        let copied: EncoderType = encoder;
        assert_eq!(encoder, copied);
    }

    #[test]
    fn test_pipeline_stats_default() {
        let stats = PipelineStats::default();
        assert_eq!(stats.samples_processed, 0);
        assert_eq!(stats.spikes_generated, 0);
        assert_eq!(stats.spike_rate, 0.0);
        assert_eq!(stats.samples_dropped, 0);
        assert_eq!(stats.avg_latency, 0.0);
        assert_eq!(stats.max_latency, 0.0);
        assert_eq!(stats.uptime, 0.0);
    }

    #[test]
    fn test_pipeline_stats_clone() {
        let mut stats = PipelineStats::default();
        stats.samples_processed = 1000;
        stats.spikes_generated = 50;
        stats.spike_rate = 5.0;

        let cloned = stats.clone();
        assert_eq!(cloned.samples_processed, 1000);
        assert_eq!(cloned.spikes_generated, 50);
        assert_eq!(cloned.spike_rate, 5.0);
    }

    #[test]
    fn test_encoding_pipeline_new() {
        let config = PipelineConfig::default();
        let pipeline = EncodingPipeline::new(config.clone());
        assert!(!pipeline.is_running());
        assert_eq!(pipeline.config().input_type, config.input_type);
    }

    #[test]
    fn test_encoding_pipeline_is_not_running_initially() {
        let pipeline = EncodingPipeline::new(PipelineConfig::default());
        assert!(!pipeline.is_running());
    }

    #[test]
    fn test_encoding_pipeline_get_stats_none_initially() {
        let pipeline = EncodingPipeline::new(PipelineConfig::default());
        assert!(pipeline.get_stats().is_none());
    }

    #[test]
    fn test_encoding_pipeline_config_accessor() {
        let mut config = PipelineConfig::default();
        config.input_stream = "TestStream".to_string();
        config.encoder_params.threshold = 0.05;

        let pipeline = EncodingPipeline::new(config);
        assert_eq!(pipeline.config().input_stream, "TestStream");
    }

    #[test]
    fn test_pipeline_builder_new() {
        let builder = PipelineBuilder::new();
        let pipeline = builder.build();
        assert!(!pipeline.is_running());
    }

    #[test]
    fn test_pipeline_builder_default() {
        let builder = PipelineBuilder::default();
        let pipeline = builder.build();
        assert_eq!(pipeline.config().input_type, "EEG");
    }

    #[test]
    fn test_pipeline_builder_input_stream() {
        let pipeline = PipelineBuilder::new()
            .input_stream("MyEEG")
            .build();
        assert_eq!(pipeline.config().input_stream, "MyEEG");
    }

    #[test]
    fn test_pipeline_builder_input_type() {
        let pipeline = PipelineBuilder::new()
            .input_type("ECG")
            .build();
        assert_eq!(pipeline.config().input_type, "ECG");
    }

    #[test]
    fn test_pipeline_builder_output_name() {
        let pipeline = PipelineBuilder::new()
            .output_name("CustomSpikes")
            .build();
        assert_eq!(pipeline.config().output_name, "CustomSpikes");
    }

    #[test]
    fn test_pipeline_builder_output_source_id() {
        let pipeline = PipelineBuilder::new()
            .output_source_id("custom_source")
            .build();
        assert_eq!(pipeline.config().output_source_id, "custom_source");
    }

    #[test]
    fn test_pipeline_builder_encoder() {
        let pipeline = PipelineBuilder::new()
            .encoder(EncoderType::Delta)
            .build();
        assert_eq!(pipeline.config().encoder_type, EncoderType::Delta);
    }

    #[test]
    fn test_pipeline_builder_threshold() {
        let pipeline = PipelineBuilder::new()
            .threshold(0.05)
            .build();
        assert_eq!(pipeline.config().encoder_params.threshold, 0.05);
    }

    #[test]
    fn test_pipeline_builder_num_levels() {
        let pipeline = PipelineBuilder::new()
            .num_levels(128)
            .build();
        assert_eq!(pipeline.config().encoder_params.num_levels, 128);
    }

    #[test]
    fn test_pipeline_builder_adaptive() {
        let pipeline = PipelineBuilder::new()
            .adaptive(false)
            .build();
        assert!(!pipeline.config().adaptive);
    }

    #[test]
    fn test_pipeline_builder_adaptation_rate() {
        let pipeline = PipelineBuilder::new()
            .adaptation_rate(0.05)
            .build();
        assert_eq!(pipeline.config().encoder_params.adaptation_rate, 0.05);
    }

    #[test]
    fn test_pipeline_builder_buffer_size() {
        let pipeline = PipelineBuilder::new()
            .buffer_size(2048)
            .build();
        assert_eq!(pipeline.config().buffer_size, 2048);
    }

    #[test]
    fn test_pipeline_builder_stats_interval() {
        let pipeline = PipelineBuilder::new()
            .stats_interval(30.0)
            .build();
        assert_eq!(pipeline.config().stats_interval, 30.0);
    }

    #[test]
    fn test_pipeline_builder_chained() {
        let pipeline = PipelineBuilder::new()
            .input_stream("TestStream")
            .input_type("EEG")
            .output_name("Spikes")
            .encoder(EncoderType::TemporalContrast)
            .threshold(0.08)
            .adaptive(true)
            .buffer_size(512)
            .build();

        let config = pipeline.config();
        assert_eq!(config.input_stream, "TestStream");
        assert_eq!(config.input_type, "EEG");
        assert_eq!(config.output_name, "Spikes");
        assert_eq!(config.encoder_type, EncoderType::TemporalContrast);
        assert_eq!(config.encoder_params.threshold, 0.08);
        assert!(config.adaptive);
        assert_eq!(config.buffer_size, 512);
    }

    #[test]
    fn test_pipeline_config_serialization() {
        let config = PipelineConfig {
            input_stream: "TestStream".to_string(),
            input_type: "EEG".to_string(),
            output_name: "Spikes".to_string(),
            output_source_id: "test".to_string(),
            encoder_type: EncoderType::Delta,
            encoder_params: EncoderParams::default(),
            buffer_size: 1024,
            adaptive: true,
            stats_interval: 60.0,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("TestStream"));
        assert!(json.contains("Delta"));

        let deserialized: PipelineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.input_stream, config.input_stream);
        assert_eq!(deserialized.encoder_type, config.encoder_type);
    }

    #[test]
    fn test_encoder_params_serialization() {
        let params = EncoderParams {
            threshold: 0.15,
            num_levels: 32,
            refractory_period: 0.002,
            adaptation_rate: 0.02,
        };

        let json = serde_json::to_string(&params).unwrap();
        let deserialized: EncoderParams = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.threshold, params.threshold);
        assert_eq!(deserialized.num_levels, params.num_levels);
        assert_eq!(deserialized.refractory_period, params.refractory_period);
    }

    #[test]
    fn test_encoder_type_serialization() {
        let encoder = EncoderType::SendOnDelta;
        let json = serde_json::to_string(&encoder).unwrap();
        let deserialized: EncoderType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, encoder);
    }

    #[test]
    fn test_all_encoder_types_serialization() {
        let types = vec![
            EncoderType::LevelCrossing,
            EncoderType::Delta,
            EncoderType::TemporalContrast,
            EncoderType::SendOnDelta,
        ];

        for encoder_type in types {
            let json = serde_json::to_string(&encoder_type).unwrap();
            let deserialized: EncoderType = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, encoder_type);
        }
    }
}
