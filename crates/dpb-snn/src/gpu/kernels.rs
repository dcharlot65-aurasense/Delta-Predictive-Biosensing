//! GPU Kernel Definitions for SNN Operations
//!
//! This module defines trait interfaces and configuration for GPU kernels
//! used in spiking neural network training and inference.
//!
//! ## Kernel Types
//!
//! - **Spike Propagation**: Update neuron states and generate spikes
//! - **Weight Updates**: STDP and gradient-based learning
//! - **Reduction**: Sum, max, min operations across arrays
//! - **Sparse Operations**: Efficient sparse matrix operations
//!
//! ## Design Philosophy
//!
//! Kernels are designed to be:
//! - Backend-agnostic through trait abstraction
//! - Optimized for parallel execution
//! - Memory-efficient with minimal transfers
//! - Configurable for different network sizes

use super::{GpuBuffer, GpuError, GpuResult};
use std::sync::Arc;

/// Configuration for spike propagation kernels
#[derive(Debug, Clone)]
pub struct SpikeKernelConfig {
    /// Number of neurons to process
    pub num_neurons: usize,
    /// Time step size (ms)
    pub dt: f32,
    /// Spike threshold voltage
    pub threshold: f32,
    /// Reset voltage after spike
    pub reset_voltage: f32,
    /// Membrane time constant (ms)
    pub tau_mem: f32,
    /// Synaptic time constant (ms)
    pub tau_syn: f32,
    /// Refractory period (ms)
    pub refractory_period: f32,
}

impl Default for SpikeKernelConfig {
    fn default() -> Self {
        Self {
            num_neurons: 1000,
            dt: 1.0,
            threshold: 1.0,
            reset_voltage: 0.0,
            tau_mem: 20.0,
            tau_syn: 5.0,
            refractory_period: 2.0,
        }
    }
}

/// Trait for spike propagation kernels
///
/// Implements the forward pass of spiking neurons, computing membrane
/// potential updates and spike generation.
pub trait SpikeKernel: Send + Sync {
    /// Execute spike propagation for one time step
    ///
    /// # Arguments
    ///
    /// * `voltages` - Current membrane voltages (input/output)
    /// * `spikes` - Output spike trains (binary)
    /// * `currents` - Input synaptic currents
    /// * `refractory` - Refractory period counters
    /// * `config` - Kernel configuration parameters
    ///
    /// # Returns
    ///
    /// Number of spikes generated in this time step
    fn propagate(
        &self,
        voltages: &Arc<dyn GpuBuffer>,
        spikes: &Arc<dyn GpuBuffer>,
        currents: &Arc<dyn GpuBuffer>,
        refractory: &Arc<dyn GpuBuffer>,
        config: &SpikeKernelConfig,
    ) -> GpuResult<usize>;

    /// Execute spike propagation with spike trace updates
    ///
    /// Updates both spikes and exponential traces for STDP
    fn propagate_with_traces(
        &self,
        voltages: &Arc<dyn GpuBuffer>,
        spikes: &Arc<dyn GpuBuffer>,
        traces: &Arc<dyn GpuBuffer>,
        currents: &Arc<dyn GpuBuffer>,
        refractory: &Arc<dyn GpuBuffer>,
        config: &SpikeKernelConfig,
        trace_decay: f32,
    ) -> GpuResult<usize>;

    /// Compute input currents from weighted spikes
    ///
    /// Performs sparse matrix-vector multiplication for synaptic integration
    fn compute_currents(
        &self,
        output_currents: &Arc<dyn GpuBuffer>,
        input_spikes: &Arc<dyn GpuBuffer>,
        weights: &Arc<dyn GpuBuffer>,
        connectivity: &SparseConnectivity,
    ) -> GpuResult<()>;
}

/// Configuration for weight update kernels
#[derive(Debug, Clone)]
pub struct WeightUpdateConfig {
    /// Number of synapses
    pub num_synapses: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Time constant for LTP (ms)
    pub tau_plus: f32,
    /// Time constant for LTD (ms)
    pub tau_minus: f32,
    /// Maximum weight value
    pub w_max: f32,
    /// Minimum weight value
    pub w_min: f32,
    /// L1 regularization coefficient
    pub l1_lambda: f32,
    /// L2 regularization coefficient
    pub l2_lambda: f32,
}

impl Default for WeightUpdateConfig {
    fn default() -> Self {
        Self {
            num_synapses: 10000,
            learning_rate: 0.001,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_max: 1.0,
            w_min: -1.0,
            l1_lambda: 0.0,
            l2_lambda: 0.0,
        }
    }
}

/// Trait for weight update kernels
///
/// Implements learning rules for synaptic plasticity
pub trait WeightUpdateKernel: Send + Sync {
    /// Apply STDP weight updates
    ///
    /// Updates weights based on spike timing differences between
    /// pre-synaptic and post-synaptic neurons
    fn apply_stdp(
        &self,
        weights: &Arc<dyn GpuBuffer>,
        pre_traces: &Arc<dyn GpuBuffer>,
        post_traces: &Arc<dyn GpuBuffer>,
        pre_spikes: &Arc<dyn GpuBuffer>,
        post_spikes: &Arc<dyn GpuBuffer>,
        connectivity: &SparseConnectivity,
        config: &WeightUpdateConfig,
    ) -> GpuResult<()>;

    /// Apply gradient-based weight updates
    ///
    /// Updates weights using backpropagated gradients
    fn apply_gradients(
        &self,
        weights: &Arc<dyn GpuBuffer>,
        gradients: &Arc<dyn GpuBuffer>,
        config: &WeightUpdateConfig,
    ) -> GpuResult<()>;

    /// Apply weight regularization
    ///
    /// Applies L1/L2 regularization to prevent overfitting
    fn apply_regularization(
        &self,
        weights: &Arc<dyn GpuBuffer>,
        config: &WeightUpdateConfig,
    ) -> GpuResult<()>;

    /// Apply weight constraints (clipping)
    ///
    /// Ensures weights stay within [w_min, w_max]
    fn apply_constraints(
        &self,
        weights: &Arc<dyn GpuBuffer>,
        config: &WeightUpdateConfig,
    ) -> GpuResult<()>;
}

/// Configuration for reduction kernels
#[derive(Debug, Clone)]
pub struct ReductionConfig {
    /// Number of elements to reduce
    pub num_elements: usize,
    /// Reduction operation
    pub operation: ReductionOperation,
    /// Dimension to reduce over (for multi-dimensional arrays)
    pub dim: Option<usize>,
}

/// Reduction operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOperation {
    /// Sum reduction
    Sum,
    /// Maximum value
    Max,
    /// Minimum value
    Min,
    /// Mean (average)
    Mean,
    /// Variance
    Variance,
    /// Standard deviation
    StdDev,
}

/// Trait for reduction kernels
///
/// Implements parallel reduction operations
pub trait ReductionKernel: Send + Sync {
    /// Perform reduction operation
    ///
    /// Reduces input array to a single value or reduced dimensions
    fn reduce(
        &self,
        input: &Arc<dyn GpuBuffer>,
        output: &Arc<dyn GpuBuffer>,
        config: &ReductionConfig,
    ) -> GpuResult<()>;

    /// Reduce along a specific dimension
    ///
    /// For multi-dimensional tensors, reduces along one dimension
    fn reduce_dim(
        &self,
        input: &Arc<dyn GpuBuffer>,
        output: &Arc<dyn GpuBuffer>,
        input_shape: &[usize],
        dim: usize,
        operation: ReductionOperation,
    ) -> GpuResult<()>;
}

/// Sparse connectivity pattern for SNNs
///
/// Represents connectivity using compressed sparse row (CSR) format
#[derive(Debug, Clone)]
pub struct SparseConnectivity {
    /// Number of pre-synaptic neurons
    pub num_pre: usize,
    /// Number of post-synaptic neurons
    pub num_post: usize,
    /// Total number of connections
    pub num_connections: usize,
    /// Row pointers (length: num_post + 1)
    pub indptr: Arc<dyn GpuBuffer>,
    /// Column indices (length: num_connections)
    pub indices: Arc<dyn GpuBuffer>,
}

impl SparseConnectivity {
    /// Create new sparse connectivity pattern
    pub fn new(
        num_pre: usize,
        num_post: usize,
        num_connections: usize,
        indptr: Arc<dyn GpuBuffer>,
        indices: Arc<dyn GpuBuffer>,
    ) -> GpuResult<Self> {
        // Validate buffer sizes
        let expected_indptr_size = (num_post + 1) * std::mem::size_of::<i32>();
        if indptr.size() < expected_indptr_size {
            return Err(GpuError::InvalidBufferSize {
                expected: expected_indptr_size,
                actual: indptr.size(),
            });
        }

        let expected_indices_size = num_connections * std::mem::size_of::<i32>();
        if indices.size() < expected_indices_size {
            return Err(GpuError::InvalidBufferSize {
                expected: expected_indices_size,
                actual: indices.size(),
            });
        }

        Ok(SparseConnectivity {
            num_pre,
            num_post,
            num_connections,
            indptr,
            indices,
        })
    }

    /// Get sparsity ratio (fraction of possible connections that exist)
    pub fn sparsity(&self) -> f32 {
        let total_possible = self.num_pre * self.num_post;
        if total_possible == 0 {
            return 0.0;
        }
        1.0 - (self.num_connections as f32 / total_possible as f32)
    }

    /// Get average degree (average connections per neuron)
    pub fn average_degree(&self) -> f32 {
        if self.num_post == 0 {
            return 0.0;
        }
        self.num_connections as f32 / self.num_post as f32
    }
}

/// Launch configuration for GPU kernels
///
/// Specifies thread/block layout for kernel execution
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Number of thread blocks (grid size)
    pub grid_size: usize,
    /// Number of threads per block
    pub block_size: usize,
    /// Shared memory per block (bytes)
    pub shared_memory: usize,
}

impl LaunchConfig {
    /// Create launch config for 1D problem
    ///
    /// Automatically determines optimal grid/block sizes
    pub fn for_1d(num_elements: usize, preferred_block_size: usize) -> Self {
        let block_size = preferred_block_size.min(1024); // GPU max threads per block
        let grid_size = (num_elements + block_size - 1) / block_size;

        LaunchConfig {
            grid_size,
            block_size,
            shared_memory: 0,
        }
    }

    /// Create launch config for 2D problem
    pub fn for_2d(num_rows: usize, num_cols: usize, block_rows: usize, block_cols: usize) -> Self {
        let grid_rows = (num_rows + block_rows - 1) / block_rows;
        let grid_cols = (num_cols + block_cols - 1) / block_cols;

        LaunchConfig {
            grid_size: grid_rows * grid_cols,
            block_size: block_rows * block_cols,
            shared_memory: 0,
        }
    }

    /// Set shared memory size
    pub fn with_shared_memory(mut self, bytes: usize) -> Self {
        self.shared_memory = bytes;
        self
    }

    /// Total number of threads
    pub fn total_threads(&self) -> usize {
        self.grid_size * self.block_size
    }
}

/// Kernel performance metrics
#[derive(Debug, Clone, Default)]
pub struct KernelMetrics {
    /// Kernel execution time (microseconds)
    pub execution_time_us: f64,
    /// Number of floating point operations
    pub flops: u64,
    /// Memory bandwidth used (bytes)
    pub memory_bandwidth: u64,
    /// Number of kernel launches
    pub num_launches: u64,
}

impl KernelMetrics {
    /// Calculate FLOPS (floating point operations per second)
    pub fn flops(&self) -> f64 {
        if self.execution_time_us == 0.0 {
            return 0.0;
        }
        (self.flops as f64) / (self.execution_time_us / 1_000_000.0)
    }

    /// Calculate memory bandwidth (GB/s)
    pub fn bandwidth_gbs(&self) -> f64 {
        if self.execution_time_us == 0.0 {
            return 0.0;
        }
        (self.memory_bandwidth as f64) / (self.execution_time_us / 1_000_000.0) / 1e9
    }

    /// Add metrics from another kernel execution
    pub fn accumulate(&mut self, other: &KernelMetrics) {
        self.execution_time_us += other.execution_time_us;
        self.flops += other.flops;
        self.memory_bandwidth += other.memory_bandwidth;
        self.num_launches += other.num_launches;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_kernel_config_default() {
        let config = SpikeKernelConfig::default();
        assert_eq!(config.num_neurons, 1000);
        assert_eq!(config.threshold, 1.0);
        assert_eq!(config.tau_mem, 20.0);
    }

    #[test]
    fn test_weight_update_config_default() {
        let config = WeightUpdateConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.tau_plus, 20.0);
        assert_eq!(config.w_max, 1.0);
    }

    #[test]
    fn test_reduction_operations() {
        assert_eq!(ReductionOperation::Sum, ReductionOperation::Sum);
        assert_ne!(ReductionOperation::Sum, ReductionOperation::Max);
    }

    #[test]
    fn test_launch_config_1d() {
        let config = LaunchConfig::for_1d(10000, 256);
        assert_eq!(config.block_size, 256);
        assert_eq!(config.grid_size, 40); // ceil(10000 / 256)
        assert_eq!(config.total_threads(), 10240);
    }

    #[test]
    fn test_launch_config_2d() {
        let config = LaunchConfig::for_2d(1024, 1024, 16, 16);
        assert_eq!(config.block_size, 256); // 16 * 16
        assert_eq!(config.grid_size, 4096); // 64 * 64
    }

    #[test]
    fn test_launch_config_with_shared_memory() {
        let config = LaunchConfig::for_1d(1000, 256).with_shared_memory(4096);
        assert_eq!(config.shared_memory, 4096);
    }

    #[test]
    fn test_kernel_metrics() {
        let mut metrics = KernelMetrics {
            execution_time_us: 1000.0,
            flops: 1_000_000,
            memory_bandwidth: 1_000_000,
            num_launches: 1,
        };

        // 1M operations in 1000 us (1 ms) = 1_000_000 / 0.001 s = 1B FLOPS (1 GFLOPS)
        assert_eq!(metrics.flops(), 1_000_000_000.0);
        // 1 MB in 1000 us (1 ms) = 1_000_000 / 0.001 s = 1B bytes/s = 1 GB/s
        assert_eq!(metrics.bandwidth_gbs(), 1.0);

        let other = KernelMetrics {
            execution_time_us: 500.0,
            flops: 500_000,
            memory_bandwidth: 500_000,
            num_launches: 1,
        };

        metrics.accumulate(&other);
        assert_eq!(metrics.execution_time_us, 1500.0);
        assert_eq!(metrics.flops, 1_500_000);
        assert_eq!(metrics.num_launches, 2);
    }
}
