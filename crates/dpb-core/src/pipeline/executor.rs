//! Pipeline executor for real-time signal processing
//!
//! Provides the main pipeline orchestration with latency tracking,
//! windowing, and execution mode management.

use super::buffer::SlidingWindow;
use super::stage::PipelineStage;
use std::time::{Duration, Instant};

/// Execution mode for the pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Drop frames if processing falls behind real-time
    RealTime,
    /// Buffer frames and process them all (may fall behind)
    Buffered,
    /// Block until all processing is complete
    Blocking,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Window size for processing
    pub window_size: usize,
    /// Hop size between windows (for overlap)
    pub hop_size: usize,
    /// Sample rate of input signal
    pub sample_rate: f64,
    /// Maximum allowed latency in milliseconds
    pub max_latency_ms: f64,
    /// Execution mode
    pub execution_mode: ExecutionMode,
}

impl PipelineConfig {
    /// Create a new pipeline configuration
    pub fn new(window_size: usize, hop_size: usize, sample_rate: f64) -> Self {
        Self {
            window_size,
            hop_size,
            sample_rate,
            max_latency_ms: 100.0, // Default 100ms budget
            execution_mode: ExecutionMode::RealTime,
        }
    }

    /// Set the maximum latency budget
    pub fn with_max_latency(mut self, latency_ms: f64) -> Self {
        self.max_latency_ms = latency_ms;
        self
    }

    /// Set the execution mode
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Calculate expected latency per window in milliseconds
    pub fn expected_window_latency_ms(&self) -> f64 {
        (self.hop_size as f64 / self.sample_rate) * 1000.0
    }

    /// Check if configuration is valid
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size == 0 {
            return Err("Window size must be greater than 0".to_string());
        }
        if self.hop_size == 0 {
            return Err("Hop size must be greater than 0".to_string());
        }
        if self.sample_rate <= 0.0 {
            return Err("Sample rate must be positive".to_string());
        }
        if self.max_latency_ms <= 0.0 {
            return Err("Max latency must be positive".to_string());
        }
        Ok(())
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self::new(256, 128, 1000.0) // 256 sample window, 50% overlap, 1kHz
    }
}

/// Latency statistics for pipeline monitoring
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Minimum latency observed (ms)
    pub min_latency_ms: f64,
    /// Maximum latency observed (ms)
    pub max_latency_ms: f64,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// 95th percentile latency (ms)
    pub p95_latency_ms: f64,
    /// 99th percentile latency (ms)
    pub p99_latency_ms: f64,
    /// Rate of deadline misses (0.0 to 1.0)
    pub deadline_miss_rate: f64,
    /// Total samples processed
    pub total_samples: u64,
    /// Number of samples dropped (in RealTime mode)
    pub dropped_samples: u64,
}

impl LatencyStats {
    /// Create new statistics with default values
    pub fn new() -> Self {
        Self {
            min_latency_ms: f64::INFINITY,
            max_latency_ms: 0.0,
            avg_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            deadline_miss_rate: 0.0,
            total_samples: 0,
            dropped_samples: 0,
        }
    }

    /// Calculate statistics from a list of latencies
    pub fn calculate(latencies: &[f64], max_latency: f64, dropped: u64, total: u64) -> Self {
        if latencies.is_empty() {
            return Self::new();
        }

        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let avg = sorted.iter().sum::<f64>() / sorted.len() as f64;

        let p95_idx = ((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1);
        let p99_idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        let p95 = sorted[p95_idx];
        let p99 = sorted[p99_idx];

        let deadline_misses = sorted.iter().filter(|&&l| l > max_latency).count();
        let miss_rate = deadline_misses as f64 / sorted.len() as f64;

        Self {
            min_latency_ms: min,
            max_latency_ms: max,
            avg_latency_ms: avg,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            deadline_miss_rate: miss_rate,
            total_samples: total,
            dropped_samples: dropped,
        }
    }

    /// Get drop rate (0.0 to 1.0)
    pub fn drop_rate(&self) -> f64 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.dropped_samples as f64 / self.total_samples as f64
        }
    }

    /// Check if pipeline is meeting real-time requirements
    pub fn is_realtime(&self, budget_ms: f64) -> bool {
        self.avg_latency_ms <= budget_ms && self.deadline_miss_rate < 0.01
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Main pipeline executor for real-time processing
pub struct PipelineExecutor {
    config: PipelineConfig,
    input_buffer: SlidingWindow<f64>,
    latencies: Vec<f64>,
    stats: LatencyStats,
    start_time: Option<Instant>,
    total_samples: u64,
    dropped_samples: u64,
    last_window_time: Option<Instant>,
}

impl PipelineExecutor {
    /// Create a new pipeline executor
    pub fn new(config: PipelineConfig) -> Self {
        config.validate().expect("Invalid pipeline configuration");

        let input_buffer = SlidingWindow::new(config.window_size, config.hop_size);

        Self {
            config,
            input_buffer,
            latencies: Vec::new(),
            stats: LatencyStats::new(),
            start_time: None,
            total_samples: 0,
            dropped_samples: 0,
            last_window_time: None,
        }
    }

    /// Process a single sample, returns output if window is complete
    pub fn process_sample<F, O>(&mut self, sample: f64, mut processor: F) -> Option<O>
    where
        F: FnMut(&[f64]) -> O,
    {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        self.total_samples += 1;

        // Push sample to window
        if let Some(window) = self.input_buffer.push(sample) {
            let process_start = Instant::now();
            let output = processor(&window);
            let latency = process_start.elapsed();
            let latency_ms = latency.as_secs_f64() * 1000.0;

            // Check if we should drop this window in RealTime mode due to deadline miss
            if self.config.execution_mode == ExecutionMode::RealTime {
                if latency_ms > self.config.max_latency_ms {
                    self.dropped_samples += window.len() as u64;
                    return None;
                }
            }

            self.latencies.push(latency_ms);
            self.last_window_time = Some(Instant::now());

            // Update stats periodically
            if self.latencies.len() >= 100 {
                self.update_stats();
            }

            Some(output)
        } else {
            None
        }
    }

    /// Process a chunk of samples
    pub fn process_chunk<F, O>(&mut self, samples: &[f64], mut processor: F) -> Vec<O>
    where
        F: FnMut(&[f64]) -> O,
    {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        let mut outputs = Vec::new();

        for &sample in samples {
            if let Some(output) = self.process_sample(sample, &mut processor) {
                outputs.push(output);
            }
        }

        outputs
    }

    /// Update statistics from collected latencies
    fn update_stats(&mut self) {
        self.stats = LatencyStats::calculate(
            &self.latencies,
            self.config.max_latency_ms,
            self.dropped_samples,
            self.total_samples,
        );
        // Keep some recent latencies for rolling stats
        if self.latencies.len() > 1000 {
            self.latencies.drain(0..500);
        }
    }

    /// Get current latency statistics
    pub fn latency_stats(&self) -> &LatencyStats {
        &self.stats
    }

    /// Get a copy of latency statistics
    pub fn compute_stats(&mut self) -> LatencyStats {
        self.update_stats();
        self.stats.clone()
    }

    /// Reset the pipeline state
    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.latencies.clear();
        self.stats = LatencyStats::new();
        self.start_time = None;
        self.total_samples = 0;
        self.dropped_samples = 0;
        self.last_window_time = None;
    }

    /// Check if pipeline is within latency budget
    pub fn is_realtime(&self) -> bool {
        self.stats.is_realtime(self.config.max_latency_ms)
    }

    /// Get the pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Get total samples processed
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Get dropped samples count
    pub fn dropped_samples(&self) -> u64 {
        self.dropped_samples
    }

    /// Get elapsed time since first sample
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|t| t.elapsed())
    }
}

/// Builder for complex multi-stage pipelines
pub struct PipelineBuilder {
    stages: Vec<Box<dyn PipelineStage<Input = Vec<f64>, Output = Vec<f64>>>>,
    config: PipelineConfig,
}

impl PipelineBuilder {
    /// Create a new pipeline builder with default configuration
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            config: PipelineConfig::default(),
        }
    }

    /// Create a builder with specific configuration
    pub fn with_config(config: PipelineConfig) -> Self {
        Self {
            stages: Vec::new(),
            config,
        }
    }

    /// Add a processing stage
    pub fn add_stage<S>(mut self, stage: S) -> Self
    where
        S: PipelineStage<Input = Vec<f64>, Output = Vec<f64>> + 'static,
    {
        self.stages.push(Box::new(stage));
        self
    }

    /// Build the final pipeline
    pub fn build(self) -> Result<Pipeline, String> {
        self.config.validate()?;

        if self.stages.is_empty() {
            return Err("Pipeline must have at least one stage".to_string());
        }

        Ok(Pipeline {
            executor: PipelineExecutor::new(self.config),
            stages: self.stages,
        })
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete multi-stage pipeline
pub struct Pipeline {
    executor: PipelineExecutor,
    stages: Vec<Box<dyn PipelineStage<Input = Vec<f64>, Output = Vec<f64>>>>,
}

impl Pipeline {
    /// Process a single sample through all stages
    pub fn process_sample(&mut self, sample: f64) -> Option<Vec<f64>> {
        self.executor.process_sample(sample, |window| {
            let mut data = window.to_vec();
            for stage in &mut self.stages {
                data = stage.process(data);
            }
            data
        })
    }

    /// Process multiple samples through all stages
    pub fn process_chunk(&mut self, samples: &[f64]) -> Vec<Vec<f64>> {
        self.executor.process_chunk(samples, |window| {
            let mut data = window.to_vec();
            for stage in &mut self.stages {
                data = stage.process(data);
            }
            data
        })
    }

    /// Get latency statistics
    pub fn latency_stats(&self) -> &LatencyStats {
        self.executor.latency_stats()
    }

    /// Get computed statistics
    pub fn compute_stats(&mut self) -> LatencyStats {
        self.executor.compute_stats()
    }

    /// Reset the pipeline
    pub fn reset(&mut self) {
        self.executor.reset();
        for stage in &mut self.stages {
            stage.reset();
        }
    }

    /// Check if pipeline meets real-time requirements
    pub fn is_realtime(&self) -> bool {
        self.executor.is_realtime()
    }

    /// Get reference to executor
    pub fn executor(&self) -> &PipelineExecutor {
        &self.executor
    }

    /// Get mutable reference to executor
    pub fn executor_mut(&mut self) -> &mut PipelineExecutor {
        &mut self.executor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::stage::{IdentityStage, ScaleStage};

    #[test]
    fn test_pipeline_config() {
        let config = PipelineConfig::new(256, 128, 1000.0)
            .with_max_latency(50.0)
            .with_execution_mode(ExecutionMode::Buffered);

        assert_eq!(config.window_size, 256);
        assert_eq!(config.hop_size, 128);
        assert_eq!(config.sample_rate, 1000.0);
        assert_eq!(config.max_latency_ms, 50.0);
        assert_eq!(config.execution_mode, ExecutionMode::Buffered);
    }

    #[test]
    fn test_pipeline_config_validation() {
        let config = PipelineConfig::new(0, 128, 1000.0);
        assert!(config.validate().is_err());

        let config = PipelineConfig::new(256, 0, 1000.0);
        assert!(config.validate().is_err());

        let config = PipelineConfig::new(256, 128, -1.0);
        assert!(config.validate().is_err());

        let config = PipelineConfig::new(256, 128, 1000.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_latency_stats_calculation() {
        let latencies = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = LatencyStats::calculate(&latencies, 35.0, 0, 5);

        assert_eq!(stats.min_latency_ms, 10.0);
        assert_eq!(stats.max_latency_ms, 50.0);
        assert_eq!(stats.avg_latency_ms, 30.0);
        assert!(stats.deadline_miss_rate > 0.0); // 40 and 50 miss deadline
    }

    #[test]
    fn test_pipeline_executor_basic() {
        let config = PipelineConfig::new(3, 3, 1000.0);
        let mut executor = PipelineExecutor::new(config);

        let result1 = executor.process_sample(1.0, |w| w.to_vec());
        assert!(result1.is_none()); // Not enough samples yet

        let result2 = executor.process_sample(2.0, |w| w.to_vec());
        assert!(result2.is_none());

        let result3 = executor.process_sample(3.0, |w| w.to_vec());
        assert_eq!(result3, Some(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_pipeline_executor_chunk() {
        let config = PipelineConfig::new(3, 3, 1000.0);
        let mut executor = PipelineExecutor::new(config);

        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let results = executor.process_chunk(&samples, |w| w.to_vec());

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(results[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_pipeline_executor_overlap() {
        let config = PipelineConfig::new(4, 2, 1000.0);
        let mut executor = PipelineExecutor::new(config);

        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let results = executor.process_chunk(&samples, |w| w.to_vec());

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(results[1], vec![3.0, 4.0, 5.0, 6.0]); // 2 sample overlap
    }

    #[test]
    fn test_pipeline_executor_reset() {
        let config = PipelineConfig::new(3, 3, 1000.0);
        let mut executor = PipelineExecutor::new(config);

        executor.process_sample(1.0, |w| w.to_vec());
        executor.process_sample(2.0, |w| w.to_vec());

        executor.reset();

        assert_eq!(executor.total_samples(), 0);
        assert_eq!(executor.dropped_samples(), 0);
    }

    #[test]
    fn test_pipeline_builder() {
        let stage1 = IdentityStage::new("identity", 10);
        let stage2 = ScaleStage::new("scale", 10, 2.0);

        let pipeline = PipelineBuilder::new()
            .add_stage(stage1)
            .add_stage(stage2)
            .build();

        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_pipeline_multi_stage() {
        let config = PipelineConfig::new(3, 3, 1000.0);
        let stage1 = ScaleStage::new("scale", 3, 2.0);
        let stage2 = ScaleStage::new("scale2", 3, 3.0);

        let mut pipeline = PipelineBuilder::with_config(config)
            .add_stage(stage1)
            .add_stage(stage2)
            .build()
            .unwrap();

        let samples = vec![1.0, 2.0, 3.0];
        let results = pipeline.process_chunk(&samples);

        assert_eq!(results.len(), 1);
        // 1*2*3=6, 2*2*3=12, 3*2*3=18
        assert_eq!(results[0], vec![6.0, 12.0, 18.0]);
    }

    #[test]
    fn test_execution_modes() {
        let modes = [
            ExecutionMode::RealTime,
            ExecutionMode::Buffered,
            ExecutionMode::Blocking,
        ];

        for mode in modes {
            let config = PipelineConfig::new(3, 3, 1000.0)
                .with_execution_mode(mode);
            let executor = PipelineExecutor::new(config);
            assert_eq!(executor.config().execution_mode, mode);
        }
    }

    #[test]
    fn test_latency_tracking() {
        let config = PipelineConfig::new(3, 3, 1000.0)
            .with_max_latency(10.0);
        let mut executor = PipelineExecutor::new(config);

        // Process several windows
        for i in 0..300 {
            executor.process_sample(i as f64, |w| w.to_vec());
        }

        let stats = executor.compute_stats();
        assert!(stats.total_samples > 0);
        assert!(stats.min_latency_ms >= 0.0);
        assert!(stats.max_latency_ms >= stats.min_latency_ms);
    }

    #[test]
    fn test_drop_rate_calculation() {
        let stats = LatencyStats {
            total_samples: 100,
            dropped_samples: 10,
            ..Default::default()
        };

        assert_eq!(stats.drop_rate(), 0.1);
    }

    #[test]
    fn test_is_realtime() {
        let stats = LatencyStats {
            avg_latency_ms: 50.0,
            deadline_miss_rate: 0.005,
            ..Default::default()
        };

        assert!(stats.is_realtime(100.0));
        assert!(!stats.is_realtime(40.0));
    }
}
