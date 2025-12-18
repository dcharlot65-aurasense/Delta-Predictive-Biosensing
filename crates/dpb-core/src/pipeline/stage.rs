//! Pipeline stage abstraction and timing infrastructure
//!
//! Provides composable stages for multi-step signal processing pipelines
//! with built-in metrics collection and deadline monitoring.

use std::time::{Duration, Instant};

/// Configuration for a pipeline stage
#[derive(Debug, Clone)]
pub struct StageConfig {
    /// Stage name for identification and logging
    pub name: String,
    /// Expected input size (number of samples)
    pub input_size: usize,
    /// Expected output size (number of samples)
    pub output_size: usize,
    /// Latency budget in milliseconds
    pub latency_budget_ms: f64,
    /// Internal buffer size for stage state
    pub buffer_size: usize,
}

impl StageConfig {
    /// Create a new stage configuration
    pub fn new(name: impl Into<String>, input_size: usize, output_size: usize) -> Self {
        Self {
            name: name.into(),
            input_size,
            output_size,
            latency_budget_ms: 100.0, // Default 100ms budget
            buffer_size: 1024,
        }
    }

    /// Set the latency budget
    pub fn with_latency_budget(mut self, budget_ms: f64) -> Self {
        self.latency_budget_ms = budget_ms;
        self
    }

    /// Set the buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
}

/// Metrics collected for each stage
#[derive(Debug, Clone)]
pub struct StageMetrics {
    /// Stage name
    pub name: String,
    /// Total number of process() calls
    pub total_calls: u64,
    /// Total processing time across all calls
    pub total_time: Duration,
    /// Minimum processing time observed
    pub min_time: Duration,
    /// Maximum processing time observed
    pub max_time: Duration,
    /// Average processing time
    pub avg_time: Duration,
    /// Number of times the deadline was missed
    pub deadline_misses: u64,
}

impl StageMetrics {
    /// Create new metrics for a stage
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            total_calls: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            avg_time: Duration::ZERO,
            deadline_misses: 0,
        }
    }

    /// Update metrics with a new timing measurement
    pub fn record(&mut self, duration: Duration, deadline_ms: f64) {
        self.total_calls += 1;
        self.total_time += duration;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
        self.avg_time = self.total_time / self.total_calls as u32;

        if duration.as_secs_f64() * 1000.0 > deadline_ms {
            self.deadline_misses += 1;
        }
    }

    /// Get the deadline miss rate (0.0 to 1.0)
    pub fn miss_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.deadline_misses as f64 / self.total_calls as f64
        }
    }

    /// Get average latency in milliseconds
    pub fn avg_latency_ms(&self) -> f64 {
        self.avg_time.as_secs_f64() * 1000.0
    }

    /// Get min latency in milliseconds
    pub fn min_latency_ms(&self) -> f64 {
        if self.min_time == Duration::MAX {
            0.0
        } else {
            self.min_time.as_secs_f64() * 1000.0
        }
    }

    /// Get max latency in milliseconds
    pub fn max_latency_ms(&self) -> f64 {
        self.max_time.as_secs_f64() * 1000.0
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.total_calls = 0;
        self.total_time = Duration::ZERO;
        self.min_time = Duration::MAX;
        self.max_time = Duration::ZERO;
        self.avg_time = Duration::ZERO;
        self.deadline_misses = 0;
    }
}

/// Trait for pipeline processing stages
///
/// Implement this trait to create custom processing stages that can be
/// composed into multi-step pipelines.
pub trait PipelineStage: Send + Sync {
    /// Input data type
    type Input;
    /// Output data type
    type Output;

    /// Process input data and produce output
    fn process(&mut self, input: Self::Input) -> Self::Output;

    /// Get the stage name
    fn name(&self) -> &str;

    /// Reset the stage state
    fn reset(&mut self);

    /// Get the stage configuration
    fn config(&self) -> &StageConfig;
}

/// Wrapper that adds timing to any stage
///
/// Automatically tracks execution time and deadline misses for
/// any stage that implements PipelineStage.
pub struct TimedStage<S: PipelineStage> {
    inner: S,
    metrics: StageMetrics,
    deadline_ms: f64,
}

impl<S: PipelineStage> TimedStage<S> {
    /// Create a new timed stage wrapper
    pub fn new(stage: S, deadline_ms: f64) -> Self {
        let name = stage.name().to_string();
        Self {
            inner: stage,
            metrics: StageMetrics::new(name),
            deadline_ms,
        }
    }

    /// Get the current metrics
    pub fn metrics(&self) -> &StageMetrics {
        &self.metrics
    }

    /// Get a mutable reference to metrics
    pub fn metrics_mut(&mut self) -> &mut StageMetrics {
        &mut self.metrics
    }

    /// Process with timing
    pub fn process_timed(&mut self, input: S::Input) -> S::Output {
        let start = Instant::now();
        let output = self.inner.process(input);
        let duration = start.elapsed();

        self.metrics.record(duration, self.deadline_ms);
        output
    }

    /// Get a reference to the inner stage
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get a mutable reference to the inner stage
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    /// Unwrap and return the inner stage
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: PipelineStage> PipelineStage for TimedStage<S> {
    type Input = S::Input;
    type Output = S::Output;

    fn process(&mut self, input: Self::Input) -> Self::Output {
        self.process_timed(input)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.metrics.reset();
    }

    fn config(&self) -> &StageConfig {
        self.inner.config()
    }
}

// Example stages for testing and demonstration

/// Identity stage that passes input through unchanged
#[derive(Debug, Clone)]
pub struct IdentityStage {
    config: StageConfig,
}

impl IdentityStage {
    /// Create a new identity stage
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            config: StageConfig::new(name, size, size),
        }
    }
}

impl PipelineStage for IdentityStage {
    type Input = Vec<f64>;
    type Output = Vec<f64>;

    fn process(&mut self, input: Self::Input) -> Self::Output {
        input
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn reset(&mut self) {
        // No state to reset
    }

    fn config(&self) -> &StageConfig {
        &self.config
    }
}

/// Scale stage that multiplies all values by a constant
#[derive(Debug, Clone)]
pub struct ScaleStage {
    config: StageConfig,
    scale_factor: f64,
}

impl ScaleStage {
    /// Create a new scale stage
    pub fn new(name: impl Into<String>, size: usize, scale_factor: f64) -> Self {
        Self {
            config: StageConfig::new(name, size, size),
            scale_factor,
        }
    }
}

impl PipelineStage for ScaleStage {
    type Input = Vec<f64>;
    type Output = Vec<f64>;

    fn process(&mut self, input: Self::Input) -> Self::Output {
        input.iter().map(|&x| x * self.scale_factor).collect()
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn reset(&mut self) {
        // No state to reset
    }

    fn config(&self) -> &StageConfig {
        &self.config
    }
}

/// Normalize stage that normalizes input to [0, 1] range
#[derive(Debug, Clone)]
pub struct NormalizeStage {
    config: StageConfig,
    min_val: Option<f64>,
    max_val: Option<f64>,
}

impl NormalizeStage {
    /// Create a new normalize stage
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            config: StageConfig::new(name, size, size),
            min_val: None,
            max_val: None,
        }
    }

    /// Create with fixed normalization range
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_val = Some(min);
        self.max_val = Some(max);
        self
    }
}

impl PipelineStage for NormalizeStage {
    type Input = Vec<f64>;
    type Output = Vec<f64>;

    fn process(&mut self, input: Self::Input) -> Self::Output {
        if input.is_empty() {
            return input;
        }

        let min = self.min_val.unwrap_or_else(|| {
            input.iter().cloned().fold(f64::INFINITY, f64::min)
        });

        let max = self.max_val.unwrap_or_else(|| {
            input.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        });

        let range = max - min;
        if range == 0.0 {
            return vec![0.5; input.len()];
        }

        input.iter().map(|&x| (x - min) / range).collect()
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn reset(&mut self) {
        // No state to reset (min/max are configured, not learned)
    }

    fn config(&self) -> &StageConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_config() {
        let config = StageConfig::new("test", 100, 100)
            .with_latency_budget(50.0)
            .with_buffer_size(2048);

        assert_eq!(config.name, "test");
        assert_eq!(config.input_size, 100);
        assert_eq!(config.output_size, 100);
        assert_eq!(config.latency_budget_ms, 50.0);
        assert_eq!(config.buffer_size, 2048);
    }

    #[test]
    fn test_stage_metrics() {
        let mut metrics = StageMetrics::new("test");

        assert_eq!(metrics.total_calls, 0);
        assert_eq!(metrics.deadline_misses, 0);

        metrics.record(Duration::from_millis(10), 50.0);
        assert_eq!(metrics.total_calls, 1);
        assert_eq!(metrics.deadline_misses, 0);

        metrics.record(Duration::from_millis(60), 50.0);
        assert_eq!(metrics.total_calls, 2);
        assert_eq!(metrics.deadline_misses, 1);
        assert_eq!(metrics.miss_rate(), 0.5);
    }

    #[test]
    fn test_identity_stage() {
        let mut stage = IdentityStage::new("identity", 10);
        let input = vec![1.0, 2.0, 3.0];
        let output = stage.process(input.clone());

        assert_eq!(output, input);
    }

    #[test]
    fn test_scale_stage() {
        let mut stage = ScaleStage::new("scale", 10, 2.0);
        let input = vec![1.0, 2.0, 3.0];
        let output = stage.process(input);

        assert_eq!(output, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_normalize_stage() {
        let mut stage = NormalizeStage::new("normalize", 10);
        let input = vec![0.0, 5.0, 10.0];
        let output = stage.process(input);

        assert_eq!(output, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_normalize_stage_with_range() {
        let mut stage = NormalizeStage::new("normalize", 10)
            .with_range(0.0, 10.0);
        let input = vec![0.0, 5.0, 10.0];
        let output = stage.process(input);

        assert_eq!(output, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_timed_stage() {
        let stage = IdentityStage::new("identity", 10);
        let mut timed = TimedStage::new(stage, 50.0);

        let input = vec![1.0, 2.0, 3.0];
        let output = timed.process_timed(input.clone());

        assert_eq!(output, input);
        assert_eq!(timed.metrics().total_calls, 1);
        assert!(timed.metrics().avg_latency_ms() < 50.0); // Should be fast
    }

    #[test]
    fn test_timed_stage_reset() {
        let stage = IdentityStage::new("identity", 10);
        let mut timed = TimedStage::new(stage, 50.0);

        timed.process_timed(vec![1.0, 2.0, 3.0]);
        assert_eq!(timed.metrics().total_calls, 1);

        timed.reset();
        assert_eq!(timed.metrics().total_calls, 0);
    }

    #[test]
    fn test_metrics_min_max() {
        let mut metrics = StageMetrics::new("test");

        metrics.record(Duration::from_millis(10), 100.0);
        metrics.record(Duration::from_millis(50), 100.0);
        metrics.record(Duration::from_millis(30), 100.0);

        assert_eq!(metrics.min_latency_ms(), 10.0);
        assert_eq!(metrics.max_latency_ms(), 50.0);
        assert!((metrics.avg_latency_ms() - 30.0).abs() < 0.1);
    }
}
