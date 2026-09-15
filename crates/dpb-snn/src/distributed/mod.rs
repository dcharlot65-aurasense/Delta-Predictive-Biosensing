//! # Distributed Training Infrastructure
//!
//! Distributed training support for spiking neural networks across multiple nodes.
//!
//! ## Features
//!
//! - **Data Parallelism**: Split batches across multiple workers
//! - **Model Parallelism**: Split model layers across multiple workers
//! - **Pipeline Parallelism**: Stage-based execution for large models
//! - **Gradient Aggregation**: AllReduce, AsyncSGD, GossipSGD strategies
//! - **Fault Tolerance**: Heartbeat monitoring, checkpoint/restore, elastic training
//! - **Performance Metrics**: Throughput, communication overhead, load balancing
//!
//! ## Quick Start
//!
//! ```rust
//! use dpb_snn::distributed::*;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize distributed training
//! let config = DistributedConfig::new(4, 0, DistributedBackend::Gloo);
//!
//! let mut runtime = DistributedRuntime::init(config)?;
//!
//! // Create training coordinator
//! let coordinator = TrainingCoordinator::new(
//!     runtime.clone(),
//!     AggregationStrategy::AllReduce,
//! );
//!
//! // Setup data parallelism
//! let partitioner = DataParallel::new(runtime.world_size());
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! The distributed training system is organized into:
//!
//! - [`coordinator`]: Training coordination and gradient aggregation
//! - [`partitioning`]: Data and model partitioning strategies
//! - [`communication`]: Inter-node communication primitives
//! - [`fault_tolerance`]: Failure detection and recovery
//! - [`metrics`]: Performance monitoring and optimization

pub mod communication;
pub mod coordinator;
pub mod fault_tolerance;
pub mod metrics;
pub mod partitioning;

// The module docs above describe a Quick Start in terms of these types,
// reached through `use dpb_snn::distributed::*`. Only the submodules were
// public, so that glob brought in nothing and the example could not compile.
pub use coordinator::{AggregationStrategy, CoordinatorConfig, SyncStats, TrainingCoordinator};
pub use partitioning::{
    DataParallel, ModelParallel, PartitionPlanner, PartitionStrategy, PipelineOp, PipelineParallel,
    PipelineSchedule,
};

use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Distributed training errors
#[derive(Debug, Error)]
pub enum DistributedError {
    #[error("Initialization error: {0}")]
    Initialization(String),

    #[error("Communication error: {0}")]
    Communication(String),

    #[error("Synchronization error: {0}")]
    Synchronization(String),

    #[error("Worker {rank} failed: {reason}")]
    WorkerFailure { rank: usize, reason: String },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    #[error("Partition error: {0}")]
    Partition(String),
}

pub type DistributedResult<T> = Result<T, DistributedError>;

/// Distributed backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributedBackend {
    /// Message Passing Interface (CPU/GPU)
    Mpi,
    /// Gloo collective communications (CPU)
    Gloo,
    /// NVIDIA Collective Communications Library (GPU)
    Nccl,
    /// Mock backend for testing
    Mock,
}

impl DistributedBackend {
    /// Check if backend supports GPU operations
    pub fn supports_gpu(&self) -> bool {
        matches!(self, Self::Nccl | Self::Mpi)
    }

    /// Check if backend is CPU-only
    pub fn cpu_only(&self) -> bool {
        matches!(self, Self::Gloo)
    }

    /// Get backend name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Mpi => "MPI",
            Self::Gloo => "Gloo",
            Self::Nccl => "NCCL",
            Self::Mock => "Mock",
        }
    }
}

/// Distributed training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Total number of workers
    pub world_size: usize,
    /// This worker's rank (0 to world_size - 1)
    pub rank: usize,
    /// Communication backend
    pub backend: DistributedBackend,
    /// Master node address (host:port)
    pub master_addr: String,
    /// Timeout for operations (seconds)
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    /// Enable gradient compression
    #[serde(default)]
    pub compress_gradients: bool,
    /// Gradient compression ratio (0.0 to 1.0)
    #[serde(default = "default_compression_ratio")]
    pub compression_ratio: f32,
}

fn default_timeout() -> u64 {
    300 // 5 minutes
}

fn default_compression_ratio() -> f32 {
    0.01 // 1% of gradients
}

impl DistributedConfig {
    /// Create new configuration
    pub fn new(world_size: usize, rank: usize, backend: DistributedBackend) -> Self {
        Self {
            world_size,
            rank,
            backend,
            master_addr: "localhost:29500".to_string(),
            timeout: default_timeout(),
            compress_gradients: false,
            compression_ratio: default_compression_ratio(),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> DistributedResult<()> {
        if self.world_size == 0 {
            return Err(DistributedError::InvalidConfig(
                "world_size must be greater than 0".to_string(),
            ));
        }

        if self.rank >= self.world_size {
            return Err(DistributedError::InvalidConfig(format!(
                "rank {} must be less than world_size {}",
                self.rank, self.world_size
            )));
        }

        if self.compression_ratio < 0.0 || self.compression_ratio > 1.0 {
            return Err(DistributedError::InvalidConfig(
                "compression_ratio must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if this is the master rank
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }

    /// Get number of workers
    pub fn num_workers(&self) -> usize {
        self.world_size
    }
}

/// Distributed runtime state
#[derive(Clone)]
pub struct DistributedRuntime {
    config: Arc<DistributedConfig>,
    backend: Arc<dyn CommunicationBackend>,
    state: Arc<RwLock<RuntimeState>>,
}

/// Runtime state
#[derive(Debug)]
struct RuntimeState {
    initialized: bool,
    active_workers: Vec<usize>,
    epoch: usize,
    step: usize,
}

impl DistributedRuntime {
    /// Initialize distributed runtime
    pub fn init(config: DistributedConfig) -> DistributedResult<Self> {
        config.validate()?;

        let backend = create_backend(&config)?;

        let state = RuntimeState {
            initialized: true,
            active_workers: (0..config.world_size).collect(),
            epoch: 0,
            step: 0,
        };

        Ok(Self {
            config: Arc::new(config),
            backend,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Get configuration
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }

    /// Get backend
    pub fn backend(&self) -> Arc<dyn CommunicationBackend> {
        Arc::clone(&self.backend)
    }

    /// Get world size
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Check if master
    pub fn is_master(&self) -> bool {
        self.config.is_master()
    }

    /// Barrier synchronization
    pub fn barrier(&self) -> DistributedResult<()> {
        self.backend.barrier()
    }

    /// Get current epoch
    pub fn epoch(&self) -> usize {
        self.state.read().unwrap().epoch
    }

    /// Get current step
    pub fn step(&self) -> usize {
        self.state.read().unwrap().step
    }

    /// Update epoch
    pub fn set_epoch(&self, epoch: usize) {
        self.state.write().unwrap().epoch = epoch;
    }

    /// Update step
    pub fn set_step(&self, step: usize) {
        self.state.write().unwrap().step = step;
    }

    /// Get active workers
    pub fn active_workers(&self) -> Vec<usize> {
        self.state.read().unwrap().active_workers.clone()
    }

    /// Mark worker as failed
    pub fn mark_worker_failed(&self, rank: usize) {
        let mut state = self.state.write().unwrap();
        state.active_workers.retain(|&r| r != rank);
    }

    /// Shutdown runtime
    pub fn shutdown(&self) -> DistributedResult<()> {
        self.backend.shutdown()?;
        self.state.write().unwrap().initialized = false;
        Ok(())
    }

    /// Whether this runtime is still live.
    ///
    /// False once [`Self::shutdown`] has run. The `initialized` flag was set at
    /// construction and then never read or cleared, so a shut-down runtime was
    /// indistinguishable from a live one.
    pub fn is_initialized(&self) -> bool {
        self.state.read().unwrap().initialized
    }
}

/// Communication backend trait
pub trait CommunicationBackend: Send + Sync {
    /// Initialize backend
    fn init(&self, config: &DistributedConfig) -> DistributedResult<()>;

    /// Barrier synchronization
    fn barrier(&self) -> DistributedResult<()>;

    /// Broadcast data from root to all workers
    fn broadcast(&self, data: &mut [f32], root: usize) -> DistributedResult<()>;

    /// All-reduce operation
    fn all_reduce(&self, data: &mut [f32], op: ReduceOp) -> DistributedResult<()>;

    /// Reduce operation
    fn reduce(&self, data: &mut [f32], root: usize, op: ReduceOp) -> DistributedResult<()>;

    /// All-gather operation
    fn all_gather(&self, send_data: &[f32], recv_data: &mut [f32]) -> DistributedResult<()>;

    /// Scatter operation
    fn scatter(
        &self,
        send_data: &[f32],
        recv_data: &mut [f32],
        root: usize,
    ) -> DistributedResult<()>;

    /// Gather operation
    fn gather(
        &self,
        send_data: &[f32],
        recv_data: &mut [f32],
        root: usize,
    ) -> DistributedResult<()>;

    /// Send data to specific rank
    fn send(&self, data: &[f32], dst: usize, tag: i32) -> DistributedResult<()>;

    /// Receive data from specific rank
    fn recv(&self, data: &mut [f32], src: usize, tag: i32) -> DistributedResult<()>;

    /// Shutdown backend
    fn shutdown(&self) -> DistributedResult<()>;
}

/// Reduction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Min,
    Max,
    Product,
}

impl ReduceOp {
    /// Apply reduction operation
    pub fn apply(&self, values: &[f32]) -> f32 {
        match self {
            Self::Sum => values.iter().sum(),
            Self::Mean => values.iter().sum::<f32>() / values.len() as f32,
            Self::Min => values.iter().copied().fold(f32::INFINITY, f32::min),
            Self::Max => values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            Self::Product => values.iter().product(),
        }
    }
}

/// Mock backend for testing
struct MockBackend {
    config: RwLock<Option<DistributedConfig>>,
}

impl MockBackend {
    fn new() -> Self {
        Self {
            config: RwLock::new(None),
        }
    }
}

impl CommunicationBackend for MockBackend {
    fn init(&self, config: &DistributedConfig) -> DistributedResult<()> {
        *self.config.write().unwrap() = Some(config.clone());
        Ok(())
    }

    fn barrier(&self) -> DistributedResult<()> {
        Ok(())
    }

    fn broadcast(&self, _data: &mut [f32], _root: usize) -> DistributedResult<()> {
        Ok(())
    }

    fn all_reduce(&self, data: &mut [f32], op: ReduceOp) -> DistributedResult<()> {
        // In mock mode, just apply operation locally
        if let Some(config) = self.config.read().unwrap().as_ref() {
            if config.world_size > 1 {
                // Simulate reduction
                match op {
                    ReduceOp::Sum | ReduceOp::Mean => {
                        for val in data.iter_mut() {
                            *val *= config.world_size as f32;
                            if matches!(op, ReduceOp::Mean) {
                                *val /= config.world_size as f32;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn reduce(&self, _data: &mut [f32], _root: usize, _op: ReduceOp) -> DistributedResult<()> {
        Ok(())
    }

    fn all_gather(&self, send_data: &[f32], recv_data: &mut [f32]) -> DistributedResult<()> {
        recv_data[..send_data.len()].copy_from_slice(send_data);
        Ok(())
    }

    fn scatter(
        &self,
        send_data: &[f32],
        recv_data: &mut [f32],
        _root: usize,
    ) -> DistributedResult<()> {
        recv_data.copy_from_slice(&send_data[..recv_data.len()]);
        Ok(())
    }

    fn gather(
        &self,
        send_data: &[f32],
        recv_data: &mut [f32],
        _root: usize,
    ) -> DistributedResult<()> {
        recv_data[..send_data.len()].copy_from_slice(send_data);
        Ok(())
    }

    fn send(&self, _data: &[f32], _dst: usize, _tag: i32) -> DistributedResult<()> {
        // Mock implementation - no actual sending
        Ok(())
    }

    fn recv(&self, _data: &mut [f32], _src: usize, _tag: i32) -> DistributedResult<()> {
        // Mock implementation - no actual receiving
        Ok(())
    }

    fn shutdown(&self) -> DistributedResult<()> {
        *self.config.write().unwrap() = None;
        Ok(())
    }
}

/// Create backend based on configuration
fn create_backend(config: &DistributedConfig) -> DistributedResult<Arc<dyn CommunicationBackend>> {
    let backend: Arc<dyn CommunicationBackend> = match config.backend {
        DistributedBackend::Mock => Arc::new(MockBackend::new()),
        DistributedBackend::Mpi => {
            return Err(DistributedError::Backend(
                "MPI backend not yet implemented - use trait-based abstraction".to_string(),
            ));
        }
        DistributedBackend::Gloo => {
            return Err(DistributedError::Backend(
                "Gloo backend not yet implemented - use trait-based abstraction".to_string(),
            ));
        }
        DistributedBackend::Nccl => {
            return Err(DistributedError::Backend(
                "NCCL backend not yet implemented - use trait-based abstraction".to_string(),
            ));
        }
    };

    backend.init(config)?;
    Ok(backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_config_new() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 0);
        assert!(config.is_master());
    }

    #[test]
    fn test_distributed_config_validate() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        assert!(config.validate().is_ok());

        let invalid = DistributedConfig::new(0, 0, DistributedBackend::Mock);
        assert!(invalid.validate().is_err());

        let invalid_rank = DistributedConfig::new(4, 5, DistributedBackend::Mock);
        assert!(invalid_rank.validate().is_err());
    }

    #[test]
    fn test_backend_properties() {
        assert!(DistributedBackend::Nccl.supports_gpu());
        assert!(DistributedBackend::Gloo.cpu_only());
        assert_eq!(DistributedBackend::Mpi.name(), "MPI");
    }

    #[test]
    fn test_reduce_op() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(ReduceOp::Sum.apply(&data), 10.0);
        assert_eq!(ReduceOp::Mean.apply(&data), 2.5);
        assert_eq!(ReduceOp::Min.apply(&data), 1.0);
        assert_eq!(ReduceOp::Max.apply(&data), 4.0);
        assert_eq!(ReduceOp::Product.apply(&data), 24.0);
    }

    #[test]
    fn test_distributed_runtime_init() {
        let config = DistributedConfig::new(2, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();

        assert_eq!(runtime.world_size(), 2);
        assert_eq!(runtime.rank(), 0);
        assert!(runtime.is_master());
    }

    #[test]
    fn test_runtime_state() {
        let config = DistributedConfig::new(2, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();

        assert_eq!(runtime.epoch(), 0);
        assert_eq!(runtime.step(), 0);

        runtime.set_epoch(5);
        runtime.set_step(100);

        assert_eq!(runtime.epoch(), 5);
        assert_eq!(runtime.step(), 100);
    }

    #[test]
    fn test_mock_backend_operations() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();

        // Test barrier
        assert!(runtime.barrier().is_ok());

        // Test all_reduce
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(
            runtime
                .backend()
                .all_reduce(&mut data, ReduceOp::Sum)
                .is_ok()
        );
    }
}
