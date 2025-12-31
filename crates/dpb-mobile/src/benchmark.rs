//! Performance benchmarking tools for mobile inference.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Benchmark result with detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Model name
    pub model_name: String,

    /// Latency metrics
    pub latency: LatencyMetrics,

    /// Memory metrics
    pub memory: MemoryMetrics,

    /// Throughput metrics
    pub throughput: ThroughputMetrics,

    /// Power metrics (if available)
    pub power: Option<PowerMetrics>,

    /// Device information
    pub device_info: DeviceInfo,
}

/// Latency metrics in milliseconds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Minimum latency
    pub min_ms: f32,

    /// Maximum latency
    pub max_ms: f32,

    /// Average latency
    pub mean_ms: f32,

    /// Median latency
    pub median_ms: f32,

    /// Standard deviation
    pub std_dev_ms: f32,

    /// 95th percentile
    pub p95_ms: f32,

    /// 99th percentile
    pub p99_ms: f32,

    /// Number of samples
    pub num_samples: usize,
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            min_ms: 0.0,
            max_ms: 0.0,
            mean_ms: 0.0,
            median_ms: 0.0,
            std_dev_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            num_samples: 0,
        }
    }
}

impl LatencyMetrics {
    /// Create from samples
    pub fn from_samples(samples_ms: &[f32]) -> Self {
        if samples_ms.is_empty() {
            return Self::default();
        }

        let mut sorted = samples_ms.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let min_ms = sorted[0];
        let max_ms = sorted[sorted.len() - 1];
        let mean_ms = sorted.iter().sum::<f32>() / sorted.len() as f32;

        let median_ms = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let variance = sorted.iter()
            .map(|x| {
                let diff = x - mean_ms;
                diff * diff
            })
            .sum::<f32>() / sorted.len() as f32;
        let std_dev_ms = variance.sqrt();

        let p95_index = (sorted.len() as f32 * 0.95) as usize;
        let p99_index = (sorted.len() as f32 * 0.99) as usize;

        let p95_ms = sorted.get(p95_index).copied().unwrap_or(max_ms);
        let p99_ms = sorted.get(p99_index).copied().unwrap_or(max_ms);

        Self {
            min_ms,
            max_ms,
            mean_ms,
            median_ms,
            std_dev_ms,
            p95_ms,
            p99_ms,
            num_samples: sorted.len(),
        }
    }
}

/// Memory metrics in bytes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage
    pub peak_bytes: usize,

    /// Average memory usage
    pub average_bytes: usize,

    /// Model size
    pub model_size_bytes: usize,

    /// Runtime overhead
    pub runtime_overhead_bytes: usize,

    /// Peak memory usage in MB
    pub peak_mb: f32,

    /// Average memory usage in MB
    pub average_mb: f32,
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            peak_bytes: 0,
            average_bytes: 0,
            model_size_bytes: 0,
            runtime_overhead_bytes: 0,
            peak_mb: 0.0,
            average_mb: 0.0,
        }
    }
}

impl MemoryMetrics {
    /// Create from measurements
    pub fn new(
        peak_bytes: usize,
        average_bytes: usize,
        model_size_bytes: usize,
        runtime_overhead_bytes: usize,
    ) -> Self {
        Self {
            peak_bytes,
            average_bytes,
            model_size_bytes,
            runtime_overhead_bytes,
            peak_mb: peak_bytes as f32 / (1024.0 * 1024.0),
            average_mb: average_bytes as f32 / (1024.0 * 1024.0),
        }
    }
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Inferences per second
    pub inferences_per_second: f32,

    /// Samples per second
    pub samples_per_second: f32,

    /// Total inferences
    pub total_inferences: usize,

    /// Total duration in seconds
    pub total_duration_sec: f32,
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            inferences_per_second: 0.0,
            samples_per_second: 0.0,
            total_inferences: 0,
            total_duration_sec: 0.0,
        }
    }
}

impl ThroughputMetrics {
    /// Create from measurements
    pub fn new(total_inferences: usize, total_duration_sec: f32, batch_size: usize) -> Self {
        let inferences_per_second = total_inferences as f32 / total_duration_sec;
        let samples_per_second = (total_inferences * batch_size) as f32 / total_duration_sec;

        Self {
            inferences_per_second,
            samples_per_second,
            total_inferences,
            total_duration_sec,
        }
    }
}

/// Power metrics (battery impact)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerMetrics {
    /// Average power consumption in watts
    pub average_power_watts: f32,

    /// Peak power consumption in watts
    pub peak_power_watts: f32,

    /// Energy per inference in millijoules
    pub energy_per_inference_mj: f32,

    /// Estimated battery drain percentage per hour
    pub battery_drain_percent_per_hour: f32,
}

impl Default for PowerMetrics {
    fn default() -> Self {
        Self {
            average_power_watts: 0.0,
            peak_power_watts: 0.0,
            energy_per_inference_mj: 0.0,
            battery_drain_percent_per_hour: 0.0,
        }
    }
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device model
    pub model: String,

    /// Operating system
    pub os: String,

    /// OS version
    pub os_version: String,

    /// CPU architecture
    pub cpu_arch: String,

    /// Number of CPU cores
    pub cpu_cores: usize,

    /// GPU model (if available)
    pub gpu_model: Option<String>,

    /// Total RAM in MB
    pub total_ram_mb: usize,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            model: detect_device_model(),
            os: detect_os(),
            os_version: String::from("unknown"),
            cpu_arch: detect_cpu_arch(),
            cpu_cores: detect_cpu_cores(),
            gpu_model: None,
            total_ram_mb: 0,
        }
    }
}

/// Detect device model
fn detect_device_model() -> String {
    #[cfg(target_os = "ios")]
    return String::from("iOS Device");

    #[cfg(target_os = "android")]
    return String::from("Android Device");

    String::from("Unknown Device")
}

/// Detect operating system
fn detect_os() -> String {
    #[cfg(target_os = "ios")]
    return String::from("iOS");

    #[cfg(target_os = "android")]
    return String::from("Android");

    #[cfg(target_os = "linux")]
    return String::from("Linux");

    #[cfg(target_os = "macos")]
    return String::from("macOS");

    #[cfg(target_os = "windows")]
    return String::from("Windows");

    String::from("Unknown")
}

/// Detect CPU architecture
fn detect_cpu_arch() -> String {
    #[cfg(target_arch = "aarch64")]
    return String::from("ARM64");

    #[cfg(target_arch = "arm")]
    return String::from("ARM");

    #[cfg(target_arch = "x86_64")]
    return String::from("x86_64");

    #[cfg(target_arch = "x86")]
    return String::from("x86");

    String::from("Unknown")
}

/// Detect number of CPU cores
fn detect_cpu_cores() -> usize {
    // In a real implementation, this would use platform-specific APIs
    // For now, return a sensible default
    #[cfg(target_os = "ios")]
    return 6; // Typical for modern iPhones

    #[cfg(target_os = "android")]
    return 8; // Typical for modern Android phones

    4 // Default fallback
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,

    /// Number of benchmark iterations
    pub benchmark_iterations: usize,

    /// Batch size
    pub batch_size: usize,

    /// Collect memory metrics
    pub collect_memory: bool,

    /// Collect power metrics
    pub collect_power: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            batch_size: 1,
            collect_memory: true,
            collect_power: false, // Platform-specific
        }
    }
}

/// Benchmark runner
pub struct BenchmarkRunner {
    /// Configuration
    config: BenchmarkConfig,

    /// Collected latency samples
    latency_samples: Vec<f32>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        let capacity = config.benchmark_iterations;
        Self {
            config,
            latency_samples: Vec::with_capacity(capacity),
        }
    }

    /// Record a latency sample
    pub fn record_latency(&mut self, duration_ms: f32) {
        self.latency_samples.push(duration_ms);
    }

    /// Build benchmark result
    pub fn build_result(&self, model_name: String, memory: MemoryMetrics) -> BenchmarkResult {
        let latency = LatencyMetrics::from_samples(&self.latency_samples);

        let total_duration_sec = self.latency_samples.iter().sum::<f32>() / 1000.0;
        let throughput = ThroughputMetrics::new(
            self.latency_samples.len(),
            total_duration_sec,
            self.config.batch_size,
        );

        BenchmarkResult {
            model_name,
            latency,
            memory,
            throughput,
            power: None,
            device_info: DeviceInfo::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }
}

/// Simple timer for measuring durations
pub struct Timer {
    start_time: Option<u64>,
}

impl Timer {
    /// Create a new timer
    pub fn new() -> Self {
        Self { start_time: None }
    }

    /// Start the timer
    pub fn start(&mut self) {
        self.start_time = Some(Self::now_nanos());
    }

    /// Stop the timer and return elapsed milliseconds
    pub fn stop(&mut self) -> f32 {
        match self.start_time.take() {
            Some(start) => {
                let end = Self::now_nanos();
                let elapsed_nanos = end.saturating_sub(start);
                elapsed_nanos as f32 / 1_000_000.0 // Convert to milliseconds
            }
            None => 0.0,
        }
    }

    /// Get current time in nanoseconds
    fn now_nanos() -> u64 {
        // In a real implementation, this would use platform-specific high-resolution timers
        // For now, use a placeholder
        #[cfg(test)]
        return 0;

        #[cfg(not(test))]
        {
            // This would use std::time::Instant or platform APIs
            // For no_std, we'd use platform-specific timing
            0
        }
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_metrics_empty() {
        let metrics = LatencyMetrics::from_samples(&[]);
        assert_eq!(metrics.num_samples, 0);
    }

    #[test]
    fn test_latency_metrics_single() {
        let samples = vec![10.0];
        let metrics = LatencyMetrics::from_samples(&samples);
        assert_eq!(metrics.min_ms, 10.0);
        assert_eq!(metrics.max_ms, 10.0);
        assert_eq!(metrics.mean_ms, 10.0);
        assert_eq!(metrics.num_samples, 1);
    }

    #[test]
    fn test_latency_metrics_multiple() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let metrics = LatencyMetrics::from_samples(&samples);
        assert_eq!(metrics.min_ms, 1.0);
        assert_eq!(metrics.max_ms, 5.0);
        assert_eq!(metrics.mean_ms, 3.0);
        assert_eq!(metrics.median_ms, 3.0);
        assert_eq!(metrics.num_samples, 5);
    }

    #[test]
    fn test_memory_metrics() {
        let metrics = MemoryMetrics::new(1024 * 1024 * 10, 1024 * 1024 * 5, 1024 * 1024 * 2, 1024 * 1024);
        assert_eq!(metrics.peak_mb, 10.0);
        assert_eq!(metrics.average_mb, 5.0);
    }

    #[test]
    fn test_throughput_metrics() {
        let metrics = ThroughputMetrics::new(100, 1.0, 1);
        assert_eq!(metrics.inferences_per_second, 100.0);
        assert_eq!(metrics.samples_per_second, 100.0);
    }

    #[test]
    fn test_throughput_metrics_batch() {
        let metrics = ThroughputMetrics::new(100, 1.0, 4);
        assert_eq!(metrics.inferences_per_second, 100.0);
        assert_eq!(metrics.samples_per_second, 400.0);
    }

    #[test]
    fn test_device_info() {
        let info = DeviceInfo::default();
        assert!(!info.model.is_empty());
        assert!(!info.os.is_empty());
        assert!(!info.cpu_arch.is_empty());
        assert!(info.cpu_cores > 0);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.benchmark_iterations, 100);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_benchmark_runner() {
        let config = BenchmarkConfig::default();
        let mut runner = BenchmarkRunner::new(config);

        runner.record_latency(10.0);
        runner.record_latency(20.0);
        runner.record_latency(30.0);

        assert_eq!(runner.latency_samples.len(), 3);

        let memory = MemoryMetrics::default();
        let result = runner.build_result("test_model".into(), memory);

        assert_eq!(result.model_name, "test_model");
        assert_eq!(result.latency.num_samples, 3);
        assert!(result.throughput.inferences_per_second > 0.0);
    }

    #[test]
    fn test_timer() {
        let mut timer = Timer::new();
        timer.start();
        let elapsed = timer.stop();
        assert!(elapsed >= 0.0);
    }
}
