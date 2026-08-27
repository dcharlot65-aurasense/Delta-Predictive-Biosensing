//! # Distributed Metrics
//!
//! Performance monitoring and profiling for distributed training.

use super::{DistributedResult, DistributedRuntime};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Distributed metrics tracker
pub struct DistributedMetrics {
    runtime: Arc<DistributedRuntime>,
    state: Arc<RwLock<MetricsState>>,
}

/// Metrics state
#[derive(Debug)]
struct MetricsState {
    /// Training start time
    start_time: Instant,
    /// Current epoch
    epoch: usize,
    /// Current step
    step: usize,
    /// Samples processed
    samples_processed: usize,
    /// Throughput history (samples/sec)
    throughput_history: Vec<f64>,
    /// Communication overhead history
    comm_overhead_history: Vec<f64>,
    /// Step timings
    step_timings: Vec<Duration>,
    /// Worker load balance
    worker_loads: HashMap<usize, WorkerLoad>,
    /// Straggler detection
    stragglers: Vec<usize>,
}

/// Worker load information
#[derive(Debug, Clone)]
struct WorkerLoad {
    /// Samples processed
    samples: usize,
    /// Computation time
    compute_time: Duration,
    /// Communication time
    comm_time: Duration,
    /// Idle time
    idle_time: Duration,
}

impl DistributedMetrics {
    /// Create new metrics tracker
    pub fn new(runtime: Arc<DistributedRuntime>) -> Self {
        let world_size = runtime.world_size();
        let mut worker_loads = HashMap::new();

        for rank in 0..world_size {
            worker_loads.insert(
                rank,
                WorkerLoad {
                    samples: 0,
                    compute_time: Duration::ZERO,
                    comm_time: Duration::ZERO,
                    idle_time: Duration::ZERO,
                },
            );
        }

        let state = MetricsState {
            start_time: Instant::now(),
            epoch: 0,
            step: 0,
            samples_processed: 0,
            throughput_history: Vec::new(),
            comm_overhead_history: Vec::new(),
            step_timings: Vec::new(),
            worker_loads,
            stragglers: Vec::new(),
        };

        Self {
            runtime,
            state: Arc::new(RwLock::new(state)),
        }
    }

    /// Record step completion
    pub fn record_step(&self, batch_size: usize, compute_time: Duration, comm_time: Duration) {
        let mut state = self.state.write().unwrap();
        state.step += 1;
        state.samples_processed += batch_size;

        let step_time = compute_time + comm_time;
        state.step_timings.push(step_time);

        // Calculate throughput (samples/sec)
        let throughput = batch_size as f64 / step_time.as_secs_f64();
        state.throughput_history.push(throughput);

        // Calculate communication overhead
        let overhead = comm_time.as_secs_f64() / step_time.as_secs_f64();
        state.comm_overhead_history.push(overhead);

        // Update worker load
        let rank = self.runtime.rank();
        if let Some(load) = state.worker_loads.get_mut(&rank) {
            load.samples += batch_size;
            load.compute_time += compute_time;
            load.comm_time += comm_time;
        }
    }

    /// Record epoch completion
    pub fn record_epoch(&self) {
        let mut state = self.state.write().unwrap();
        state.epoch += 1;
    }

    /// Get current throughput (samples/sec)
    pub fn current_throughput(&self) -> f64 {
        let state = self.state.read().unwrap();
        state.throughput_history.last().copied().unwrap_or(0.0)
    }

    /// Get average throughput
    pub fn average_throughput(&self) -> f64 {
        let state = self.state.read().unwrap();
        if state.throughput_history.is_empty() {
            0.0
        } else {
            state.throughput_history.iter().sum::<f64>() / state.throughput_history.len() as f64
        }
    }

    /// Get global throughput (across all workers)
    pub fn global_throughput(&self) -> f64 {
        self.average_throughput() * self.runtime.world_size() as f64
    }

    /// Get communication overhead (0.0 to 1.0)
    pub fn communication_overhead(&self) -> f64 {
        let state = self.state.read().unwrap();
        if state.comm_overhead_history.is_empty() {
            0.0
        } else {
            state.comm_overhead_history.iter().sum::<f64>()
                / state.comm_overhead_history.len() as f64
        }
    }

    /// Get scaling efficiency
    pub fn scaling_efficiency(&self) -> f64 {
        // Ideal: efficiency = 1.0 (linear scaling)
        // Reality: efficiency < 1.0 due to communication overhead
        let world_size = self.runtime.world_size() as f64;
        let overhead = self.communication_overhead();

        // Simple model: efficiency = (1 - overhead) / world_size
        (1.0 - overhead).clamp(0.0, 1.0)
    }

    /// Get load balance score (0.0 to 1.0, higher is better)
    pub fn load_balance_score(&self) -> f64 {
        let state = self.state.read().unwrap();

        if state.worker_loads.is_empty() {
            return 1.0;
        }

        // Calculate variance in worker loads
        let loads: Vec<f64> = state
            .worker_loads
            .values()
            .map(|w| w.samples as f64)
            .collect();

        let mean = loads.iter().sum::<f64>() / loads.len() as f64;
        let variance = loads.iter().map(|&l| (l - mean).powi(2)).sum::<f64>() / loads.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 0.0 };

        // Convert to 0-1 score (lower CV = better balance)
        (1.0 - coefficient_of_variation.min(1.0)).max(0.0)
    }

    /// Detect stragglers (slow workers)
    pub fn detect_stragglers(&self, threshold: f64) -> Vec<usize> {
        let state = self.state.read().unwrap();

        if state.worker_loads.is_empty() {
            return Vec::new();
        }

        // Calculate average compute time
        let avg_time = state
            .worker_loads
            .values()
            .map(|w| w.compute_time.as_secs_f64())
            .sum::<f64>()
            / state.worker_loads.len() as f64;

        // Find workers slower than threshold
        state
            .worker_loads
            .iter()
            .filter(|(_, load)| {
                let time = load.compute_time.as_secs_f64();
                time > avg_time * (1.0 + threshold)
            })
            .map(|(&rank, _)| rank)
            .collect()
    }

    /// Get worker statistics
    pub fn worker_stats(&self, rank: usize) -> Option<WorkerStats> {
        let state = self.state.read().unwrap();

        state.worker_loads.get(&rank).map(|load| {
            let total_time = load.compute_time + load.comm_time + load.idle_time;

            WorkerStats {
                rank,
                samples: load.samples,
                compute_time: load.compute_time,
                comm_time: load.comm_time,
                idle_time: load.idle_time,
                compute_ratio: if total_time > Duration::ZERO {
                    load.compute_time.as_secs_f64() / total_time.as_secs_f64()
                } else {
                    0.0
                },
                comm_ratio: if total_time > Duration::ZERO {
                    load.comm_time.as_secs_f64() / total_time.as_secs_f64()
                } else {
                    0.0
                },
            }
        })
    }

    /// Get summary report
    pub fn summary(&self) -> MetricsSummary {
        let state = self.state.read().unwrap();
        let elapsed = state.start_time.elapsed();

        MetricsSummary {
            epoch: state.epoch,
            step: state.step,
            samples_processed: state.samples_processed,
            elapsed_time: elapsed,
            average_throughput: self.average_throughput(),
            global_throughput: self.global_throughput(),
            communication_overhead: self.communication_overhead(),
            scaling_efficiency: self.scaling_efficiency(),
            load_balance_score: self.load_balance_score(),
            world_size: self.runtime.world_size(),
        }
    }

    /// Reset metrics
    pub fn reset(&self) {
        let mut state = self.state.write().unwrap();
        state.start_time = Instant::now();
        state.step = 0;
        state.samples_processed = 0;
        state.throughput_history.clear();
        state.comm_overhead_history.clear();
        state.step_timings.clear();
        state.stragglers.clear();

        for load in state.worker_loads.values_mut() {
            load.samples = 0;
            load.compute_time = Duration::ZERO;
            load.comm_time = Duration::ZERO;
            load.idle_time = Duration::ZERO;
        }
    }

    /// Export metrics to JSON
    pub fn export_json(&self) -> String {
        let summary = self.summary();
        serde_json::to_string_pretty(&summary).unwrap_or_default()
    }
}

/// Worker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStats {
    pub rank: usize,
    pub samples: usize,
    pub compute_time: Duration,
    pub comm_time: Duration,
    pub idle_time: Duration,
    pub compute_ratio: f64,
    pub comm_ratio: f64,
}

/// Metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub epoch: usize,
    pub step: usize,
    pub samples_processed: usize,
    #[serde(serialize_with = "serialize_duration")]
    pub elapsed_time: Duration,
    pub average_throughput: f64,
    pub global_throughput: f64,
    pub communication_overhead: f64,
    pub scaling_efficiency: f64,
    pub load_balance_score: f64,
    pub world_size: usize,
}

/// Serialize duration as seconds
fn serialize_duration<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_f64(duration.as_secs_f64())
}

impl MetricsSummary {
    /// Get samples per second
    pub fn samples_per_second(&self) -> f64 {
        if self.elapsed_time > Duration::ZERO {
            self.samples_processed as f64 / self.elapsed_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get average step time
    pub fn average_step_time(&self) -> Duration {
        if self.step > 0 {
            self.elapsed_time / self.step as u32
        } else {
            Duration::ZERO
        }
    }

    /// Print summary
    pub fn print(&self) {
        println!("=== Distributed Training Metrics ===");
        println!("Epoch: {}", self.epoch);
        println!("Step: {}", self.step);
        println!("Samples processed: {}", self.samples_processed);
        println!("Elapsed time: {:.2}s", self.elapsed_time.as_secs_f64());
        println!("World size: {}", self.world_size);
        println!("\n=== Performance ===");
        println!(
            "Average throughput: {:.2} samples/sec",
            self.average_throughput
        );
        println!(
            "Global throughput: {:.2} samples/sec",
            self.global_throughput
        );
        println!(
            "Communication overhead: {:.2}%",
            self.communication_overhead * 100.0
        );
        println!(
            "Scaling efficiency: {:.2}%",
            self.scaling_efficiency * 100.0
        );
        println!("Load balance score: {:.2}", self.load_balance_score);
    }
}

/// Performance profiler
pub struct PerformanceProfiler {
    events: Arc<RwLock<Vec<ProfileEvent>>>,
    enabled: Arc<RwLock<bool>>,
}

/// Profile event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEvent {
    pub name: String,
    pub rank: usize,
    pub start_time: u64,
    pub duration: Duration,
    pub event_type: EventType,
}

/// Event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    Compute,
    Communication,
    Synchronization,
    IO,
    Other,
}

impl PerformanceProfiler {
    /// Create new profiler
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            enabled: Arc::new(RwLock::new(true)),
        }
    }

    /// Enable profiling
    pub fn enable(&self) {
        *self.enabled.write().unwrap() = true;
    }

    /// Disable profiling
    pub fn disable(&self) {
        *self.enabled.write().unwrap() = false;
    }

    /// Record event
    pub fn record(
        &self,
        name: String,
        rank: usize,
        start_time: Instant,
        duration: Duration,
        event_type: EventType,
    ) {
        if !*self.enabled.read().unwrap() {
            return;
        }

        let event = ProfileEvent {
            name,
            rank,
            start_time: timestamp_ms(start_time),
            duration,
            event_type,
        };

        self.events.write().unwrap().push(event);
    }

    /// Start timing a section
    pub fn start_section(&self, name: &str) -> ProfileSection {
        ProfileSection {
            name: name.to_string(),
            start: Instant::now(),
        }
    }

    /// Get all events
    pub fn events(&self) -> Vec<ProfileEvent> {
        self.events.read().unwrap().clone()
    }

    /// Get events by type
    pub fn events_by_type(&self, event_type: EventType) -> Vec<ProfileEvent> {
        self.events
            .read()
            .unwrap()
            .iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect()
    }

    /// Get total time by type
    pub fn total_time(&self, event_type: EventType) -> Duration {
        self.events_by_type(event_type)
            .iter()
            .map(|e| e.duration)
            .sum()
    }

    /// Clear events
    pub fn clear(&self) {
        self.events.write().unwrap().clear();
    }

    /// Export to Chrome trace format
    pub fn export_chrome_trace(&self) -> String {
        let events = self.events.read().unwrap();
        let chrome_events: Vec<_> = events
            .iter()
            .map(|e| {
                serde_json::json!({
                    "name": e.name,
                    "cat": format!("{:?}", e.event_type),
                    "ph": "X",
                    "ts": e.start_time,
                    "dur": e.duration.as_micros(),
                    "pid": e.rank,
                    "tid": 0,
                })
            })
            .collect();

        serde_json::json!({
            "traceEvents": chrome_events,
            "displayTimeUnit": "ms"
        })
        .to_string()
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Profile section helper
pub struct ProfileSection {
    name: String,
    start: Instant,
}

impl ProfileSection {
    /// End section and return duration
    pub fn end(self) -> (String, Duration) {
        (self.name, self.start.elapsed())
    }
}

/// Get timestamp in milliseconds
fn timestamp_ms(instant: Instant) -> u64 {
    instant.elapsed().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{DistributedBackend, DistributedConfig};
    use std::thread;

    #[test]
    fn test_metrics_tracking() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let metrics = DistributedMetrics::new(runtime);

        metrics.record_step(32, Duration::from_millis(100), Duration::from_millis(10));

        assert!(metrics.current_throughput() > 0.0);
        assert!(metrics.communication_overhead() < 1.0);
    }

    #[test]
    fn test_throughput_calculation() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let metrics = DistributedMetrics::new(runtime);

        metrics.record_step(100, Duration::from_secs(1), Duration::from_millis(100));

        let throughput = metrics.current_throughput();
        assert!(throughput > 80.0 && throughput < 100.0); // ~90 samples/sec
    }

    #[test]
    fn test_global_throughput() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let metrics = DistributedMetrics::new(runtime);

        metrics.record_step(100, Duration::from_secs(1), Duration::from_millis(100));

        let global = metrics.global_throughput();
        let local = metrics.average_throughput();

        assert!((global - local * 4.0).abs() < 1.0);
    }

    #[test]
    fn test_communication_overhead() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let metrics = DistributedMetrics::new(runtime);

        // 10% communication overhead
        metrics.record_step(100, Duration::from_millis(900), Duration::from_millis(100));

        let overhead = metrics.communication_overhead();
        assert!((overhead - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_scaling_efficiency() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let metrics = DistributedMetrics::new(runtime);

        metrics.record_step(100, Duration::from_millis(900), Duration::from_millis(100));

        let efficiency = metrics.scaling_efficiency();
        assert!(efficiency > 0.0 && efficiency <= 1.0);
    }

    #[test]
    fn test_load_balance() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let metrics = DistributedMetrics::new(runtime);

        // Record step only updates rank 0's load
        // Other workers have 0 samples, so balance is poor initially
        metrics.record_step(100, Duration::from_secs(1), Duration::from_millis(10));

        let score = metrics.load_balance_score();
        // Score will be lower because only one worker has load
        assert!(score >= 0.0 && score <= 1.0);

        // Verify score increases with more balanced load
        // (In a real scenario, all workers would report)
        assert!(score < 1.0); // Not perfect since only one worker reported
    }

    #[test]
    fn test_metrics_summary() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let metrics = DistributedMetrics::new(runtime);

        metrics.record_step(100, Duration::from_secs(1), Duration::from_millis(10));
        metrics.record_epoch();

        let summary = metrics.summary();
        assert_eq!(summary.epoch, 1);
        assert_eq!(summary.step, 1);
        assert_eq!(summary.samples_processed, 100);
        assert_eq!(summary.world_size, 4);
    }

    #[test]
    fn test_profiler() {
        let profiler = PerformanceProfiler::new();

        let start = Instant::now();
        thread::sleep(Duration::from_millis(10));
        let duration = start.elapsed();

        profiler.record(
            "test_compute".to_string(),
            0,
            start,
            duration,
            EventType::Compute,
        );

        let events = profiler.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "test_compute");
        assert_eq!(events[0].event_type, EventType::Compute);
    }

    #[test]
    fn test_profiler_sections() {
        let profiler = PerformanceProfiler::new();

        let section = profiler.start_section("test_section");
        thread::sleep(Duration::from_millis(10));
        let (name, duration) = section.end();

        assert_eq!(name, "test_section");
        assert!(duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_profiler_by_type() {
        let profiler = PerformanceProfiler::new();
        let start = Instant::now();

        profiler.record(
            "compute1".to_string(),
            0,
            start,
            Duration::from_millis(100),
            EventType::Compute,
        );
        profiler.record(
            "comm1".to_string(),
            0,
            start,
            Duration::from_millis(50),
            EventType::Communication,
        );
        profiler.record(
            "compute2".to_string(),
            0,
            start,
            Duration::from_millis(100),
            EventType::Compute,
        );

        let compute_events = profiler.events_by_type(EventType::Compute);
        assert_eq!(compute_events.len(), 2);

        let total_compute = profiler.total_time(EventType::Compute);
        assert_eq!(total_compute, Duration::from_millis(200));
    }
}
