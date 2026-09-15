//! # Data and Model Partitioning
//!
//! Strategies for partitioning data and models across distributed workers.

use super::{DistributedError, DistributedResult, DistributedRuntime};
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Partitioning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Data parallelism: split batches across workers
    DataParallel,
    /// Model parallelism: split layers across workers
    ModelParallel,
    /// Pipeline parallelism: stage-based execution
    PipelineParallel,
    /// Hybrid: combination of strategies
    Hybrid,
}

impl PartitionStrategy {
    /// Check if strategy uses data parallelism
    pub fn uses_data_parallel(&self) -> bool {
        matches!(self, Self::DataParallel | Self::Hybrid)
    }

    /// Check if strategy uses model parallelism
    pub fn uses_model_parallel(&self) -> bool {
        matches!(self, Self::ModelParallel | Self::Hybrid)
    }

    /// Get memory efficiency score (0.0 to 1.0)
    pub fn memory_efficiency(&self) -> f32 {
        match self {
            Self::DataParallel => 0.5,     // Each worker has full model
            Self::ModelParallel => 0.9,    // Model split across workers
            Self::PipelineParallel => 0.8, // Model staged across workers
            Self::Hybrid => 0.7,           // Balanced approach
        }
    }

    /// Get communication overhead score (0.0 to 1.0, higher = more overhead)
    pub fn communication_overhead(&self) -> f32 {
        match self {
            Self::DataParallel => 0.3,     // Only gradients
            Self::ModelParallel => 0.7,    // Activations and gradients
            Self::PipelineParallel => 0.5, // Sequential communication
            Self::Hybrid => 0.6,           // Combined overhead
        }
    }
}

/// Data parallel partitioner
#[derive(Debug, Clone)]
pub struct DataParallel {
    world_size: usize,
    rank: usize,
    global_batch_size: usize,
    local_batch_size: usize,
}

impl DataParallel {
    /// Create new data parallel partitioner
    pub fn new(world_size: usize) -> Self {
        Self {
            world_size,
            rank: 0,
            global_batch_size: 0,
            local_batch_size: 0,
        }
    }

    /// Initialize with runtime
    pub fn from_runtime(runtime: &DistributedRuntime) -> Self {
        Self {
            world_size: runtime.world_size(),
            rank: runtime.rank(),
            global_batch_size: 0,
            local_batch_size: 0,
        }
    }

    /// Set batch size (global across all workers)
    pub fn set_batch_size(&mut self, global_batch_size: usize) -> DistributedResult<()> {
        if global_batch_size < self.world_size {
            return Err(DistributedError::Partition(format!(
                "Global batch size {} must be >= world size {}",
                global_batch_size, self.world_size
            )));
        }

        self.global_batch_size = global_batch_size;
        self.local_batch_size = global_batch_size / self.world_size;

        Ok(())
    }

    /// Get local batch size for this worker
    pub fn local_batch_size(&self) -> usize {
        self.local_batch_size
    }

    /// Get global batch size
    pub fn global_batch_size(&self) -> usize {
        self.global_batch_size
    }

    /// Get data range for this worker
    pub fn get_partition_range(&self, total_samples: usize) -> Range<usize> {
        let samples_per_worker = total_samples / self.world_size;
        let start = self.rank * samples_per_worker;
        let end = if self.rank == self.world_size - 1 {
            total_samples // Last worker gets remaining samples
        } else {
            start + samples_per_worker
        };

        start..end
    }

    /// Split indices across workers
    pub fn split_indices(&self, indices: &[usize]) -> Vec<usize> {
        let range = self.get_partition_range(indices.len());
        indices[range].to_vec()
    }

    /// Compute effective learning rate with scaling
    pub fn scaled_learning_rate(&self, base_lr: f32) -> f32 {
        // Linear scaling rule: lr * world_size
        base_lr * self.world_size as f32
    }

    /// Compute warmup steps for large batch training
    pub fn warmup_steps(&self, base_warmup: usize) -> usize {
        // Scale warmup proportionally
        base_warmup * self.world_size
    }
}

/// Model parallel partitioner
#[derive(Debug, Clone)]
pub struct ModelParallel {
    world_size: usize,
    rank: usize,
    num_layers: usize,
    layer_assignments: Vec<usize>,
}

impl ModelParallel {
    /// Check that `num_layers` layers can be spread over `world_size` workers.
    ///
    /// Named `new` until now, which it never was: it returns `()`, not `Self`,
    /// so `ModelParallel::new(..)` could not construct anything and the name
    /// promised the opposite of what the signature delivered. Use
    /// [`Self::from_runtime`] to build one.
    pub fn validate(world_size: usize, num_layers: usize) -> DistributedResult<()> {
        if num_layers < world_size {
            return Err(DistributedError::Partition(format!(
                "Number of layers {} must be >= world size {}",
                num_layers, world_size
            )));
        }
        Ok(())
    }

    /// Create from runtime
    pub fn from_runtime(
        runtime: &DistributedRuntime,
        num_layers: usize,
    ) -> DistributedResult<Self> {
        Self::validate(runtime.world_size(), num_layers)?;

        let layer_assignments = Self::assign_layers(num_layers, runtime.world_size());

        Ok(Self {
            world_size: runtime.world_size(),
            rank: runtime.rank(),
            num_layers,
            layer_assignments,
        })
    }

    /// Assign layers to workers (balanced)
    fn assign_layers(num_layers: usize, world_size: usize) -> Vec<usize> {
        let mut assignments = Vec::with_capacity(num_layers);
        let layers_per_worker = (num_layers + world_size - 1) / world_size;

        for layer_idx in 0..num_layers {
            let worker = layer_idx / layers_per_worker;
            assignments.push(worker.min(world_size - 1));
        }

        assignments
    }

    /// Get layers assigned to this worker
    pub fn get_local_layers(&self) -> Vec<usize> {
        self.layer_assignments
            .iter()
            .enumerate()
            .filter(|(_, worker)| **worker == self.rank)
            .map(|(layer_idx, _)| layer_idx)
            .collect()
    }

    /// Check if layer is local to this worker
    pub fn is_local_layer(&self, layer_idx: usize) -> bool {
        layer_idx < self.layer_assignments.len() && self.layer_assignments[layer_idx] == self.rank
    }

    /// Get worker for given layer
    pub fn get_layer_worker(&self, layer_idx: usize) -> Option<usize> {
        self.layer_assignments.get(layer_idx).copied()
    }

    /// Get all layer assignments
    pub fn layer_assignments(&self) -> &[usize] {
        &self.layer_assignments
    }

    /// Number of workers this partition was computed for.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Number of layers this partition covers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// Pipeline parallel partitioner
#[derive(Debug, Clone)]
pub struct PipelineParallel {
    world_size: usize,
    rank: usize,
    num_stages: usize,
    num_microbatches: usize,
    stage_assignments: Vec<Range<usize>>,
}

impl PipelineParallel {
    /// Create from runtime
    pub fn from_runtime(
        runtime: &DistributedRuntime,
        num_layers: usize,
        num_microbatches: usize,
    ) -> DistributedResult<Self> {
        let num_stages = runtime.world_size();

        if num_layers < num_stages {
            return Err(DistributedError::Partition(format!(
                "Number of layers {} must be >= number of stages {}",
                num_layers, num_stages
            )));
        }

        let stage_assignments = Self::assign_stages(num_layers, num_stages);

        Ok(Self {
            world_size: runtime.world_size(),
            rank: runtime.rank(),
            num_stages,
            num_microbatches,
            stage_assignments,
        })
    }

    /// Assign layer ranges to pipeline stages
    fn assign_stages(num_layers: usize, num_stages: usize) -> Vec<Range<usize>> {
        let mut stages = Vec::with_capacity(num_stages);
        let layers_per_stage = num_layers / num_stages;
        let remainder = num_layers % num_stages;

        let mut start = 0;
        for stage in 0..num_stages {
            let extra = if stage < remainder { 1 } else { 0 };
            let end = start + layers_per_stage + extra;
            stages.push(start..end);
            start = end;
        }

        stages
    }

    /// Get stage (worker rank) for this worker
    pub fn stage(&self) -> usize {
        self.rank
    }

    /// Get number of stages
    /// Number of workers this pipeline was computed for.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Number of pipeline stages.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    /// Get layers for this stage
    pub fn get_stage_layers(&self) -> Range<usize> {
        self.stage_assignments[self.rank].clone()
    }

    /// Check if this is the first stage
    pub fn is_first_stage(&self) -> bool {
        self.rank == 0
    }

    /// Check if this is the last stage
    pub fn is_last_stage(&self) -> bool {
        self.rank == self.num_stages - 1
    }

    /// Get previous stage rank
    pub fn prev_stage(&self) -> Option<usize> {
        if self.rank > 0 {
            Some(self.rank - 1)
        } else {
            None
        }
    }

    /// Get next stage rank
    pub fn next_stage(&self) -> Option<usize> {
        if self.rank < self.num_stages - 1 {
            Some(self.rank + 1)
        } else {
            None
        }
    }

    /// Get microbatch size
    pub fn microbatch_size(&self, global_batch_size: usize) -> usize {
        global_batch_size / self.num_microbatches
    }

    /// Get pipeline schedule (forward/backward order)
    pub fn get_schedule(&self) -> PipelineSchedule {
        PipelineSchedule::new(self.num_stages, self.num_microbatches)
    }
}

/// Pipeline schedule (1F1B - One Forward One Backward)
#[derive(Debug, Clone)]
pub struct PipelineSchedule {
    num_stages: usize,
    num_microbatches: usize,
}

impl PipelineSchedule {
    /// Create new schedule
    pub fn new(num_stages: usize, num_microbatches: usize) -> Self {
        Self {
            num_stages,
            num_microbatches,
        }
    }

    /// Get schedule for a specific stage
    pub fn get_stage_schedule(&self, stage: usize) -> Vec<PipelineOp> {
        let mut schedule = Vec::new();

        // Warmup: forward passes
        let warmup_steps = (self.num_stages - stage - 1).min(self.num_microbatches);
        for microbatch in 0..warmup_steps {
            schedule.push(PipelineOp::Forward(microbatch));
        }

        // Steady state: 1F1B
        for microbatch in warmup_steps..self.num_microbatches {
            schedule.push(PipelineOp::Forward(microbatch));
            if microbatch >= warmup_steps {
                schedule.push(PipelineOp::Backward(microbatch - warmup_steps));
            }
        }

        // Cooldown: remaining backward passes
        for microbatch in (self.num_microbatches - warmup_steps)..self.num_microbatches {
            schedule.push(PipelineOp::Backward(microbatch));
        }

        schedule
    }
}

/// Pipeline operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineOp {
    Forward(usize),  // microbatch index
    Backward(usize), // microbatch index
}

/// Partition planner
pub struct PartitionPlanner {
    runtime: DistributedRuntime,
}

impl PartitionPlanner {
    /// Create new planner
    pub fn new(runtime: DistributedRuntime) -> Self {
        Self { runtime }
    }

    /// Recommend partitioning strategy
    pub fn recommend_strategy(
        &self,
        model_size_mb: f32,
        batch_size: usize,
        num_layers: usize,
    ) -> PartitionStrategy {
        let world_size = self.runtime.world_size();
        let available_memory_per_worker = 8000.0; // 8GB assumption

        // Check if model fits in single worker memory
        let model_fits_single = model_size_mb < available_memory_per_worker * 0.8;

        if model_fits_single {
            // Use data parallelism if model fits
            if batch_size >= world_size * 4 {
                PartitionStrategy::DataParallel
            } else {
                // Small batch, consider pipeline
                if num_layers >= world_size * 2 {
                    PartitionStrategy::PipelineParallel
                } else {
                    PartitionStrategy::DataParallel
                }
            }
        } else {
            // Model doesn't fit, need model parallelism
            if num_layers >= world_size {
                if batch_size >= world_size {
                    PartitionStrategy::Hybrid
                } else {
                    PartitionStrategy::PipelineParallel
                }
            } else {
                PartitionStrategy::ModelParallel
            }
        }
    }

    /// Optimize batch size for data parallelism
    pub fn optimize_batch_size(&self, base_batch_size: usize) -> usize {
        // Round up to multiple of world size
        let world_size = self.runtime.world_size();
        ((base_batch_size + world_size - 1) / world_size) * world_size
    }

    /// Optimize microbatch count for pipeline parallelism
    pub fn optimize_microbatches(&self, batch_size: usize) -> usize {
        // Rule of thumb: 4x number of stages for good pipeline efficiency
        let num_stages = self.runtime.world_size();
        let target_microbatches = num_stages * 4;

        // Ensure microbatch size is reasonable (at least 1)
        let max_microbatches = batch_size;
        target_microbatches.min(max_microbatches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{DistributedBackend, DistributedConfig};

    #[test]
    fn test_partition_strategy_properties() {
        assert!(PartitionStrategy::DataParallel.uses_data_parallel());
        assert!(PartitionStrategy::ModelParallel.uses_model_parallel());
        assert!(PartitionStrategy::Hybrid.uses_data_parallel());
        assert!(PartitionStrategy::Hybrid.uses_model_parallel());
    }

    #[test]
    fn test_data_parallel_batch_split() {
        let mut dp = DataParallel::new(4);
        dp.set_batch_size(32).unwrap();

        assert_eq!(dp.local_batch_size(), 8);
        assert_eq!(dp.global_batch_size(), 32);
    }

    #[test]
    fn test_data_parallel_partition_range() {
        let config = DistributedConfig::new(4, 1, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let dp = DataParallel::from_runtime(&runtime);

        let range = dp.get_partition_range(100);
        assert_eq!(range, 25..50); // Worker 1 gets samples 25-49
    }

    #[test]
    fn test_data_parallel_lr_scaling() {
        let dp = DataParallel::new(4);
        let scaled = dp.scaled_learning_rate(0.001);
        assert_eq!(scaled, 0.004); // Linear scaling
    }

    #[test]
    fn test_model_parallel_layer_assignment() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let mp = ModelParallel::from_runtime(&runtime, 8).unwrap();

        let local_layers = mp.get_local_layers();
        assert_eq!(local_layers, vec![0, 1]); // First worker gets first 2 layers
    }

    #[test]
    fn test_model_parallel_layer_lookup() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let mp = ModelParallel::from_runtime(&runtime, 8).unwrap();

        assert!(mp.is_local_layer(0));
        assert!(mp.is_local_layer(1));
        assert!(!mp.is_local_layer(2));

        assert_eq!(mp.get_layer_worker(0), Some(0));
        assert_eq!(mp.get_layer_worker(2), Some(1));
    }

    #[test]
    fn test_pipeline_parallel_stages() {
        let config = DistributedConfig::new(4, 1, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let pp = PipelineParallel::from_runtime(&runtime, 8, 4).unwrap();

        assert_eq!(pp.stage(), 1);
        assert!(!pp.is_first_stage());
        assert!(!pp.is_last_stage());
        assert_eq!(pp.prev_stage(), Some(0));
        assert_eq!(pp.next_stage(), Some(2));

        let layers = pp.get_stage_layers();
        assert_eq!(layers, 2..4); // Stage 1 gets layers 2-3
    }

    #[test]
    fn test_pipeline_schedule() {
        let schedule = PipelineSchedule::new(4, 8);
        let stage1_ops = schedule.get_stage_schedule(1);

        // Check that schedule starts with forwards, then alternates
        assert!(matches!(stage1_ops[0], PipelineOp::Forward(_)));
    }

    #[test]
    fn test_partition_planner_recommend() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let planner = PartitionPlanner::new(runtime);

        // Small model, large batch -> data parallel
        let strategy = planner.recommend_strategy(100.0, 64, 10);
        assert_eq!(strategy, PartitionStrategy::DataParallel);

        // Large model -> model parallel or hybrid
        let strategy = planner.recommend_strategy(10000.0, 64, 100);
        assert!(matches!(
            strategy,
            PartitionStrategy::Hybrid | PartitionStrategy::PipelineParallel
        ));
    }

    #[test]
    fn test_partition_planner_optimize() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let planner = PartitionPlanner::new(runtime);

        // Batch size optimization
        assert_eq!(planner.optimize_batch_size(30), 32); // Round up to multiple of 4
        assert_eq!(planner.optimize_batch_size(32), 32); // Already optimal

        // Microbatch optimization
        let microbatches = planner.optimize_microbatches(64);
        assert_eq!(microbatches, 16); // 4 stages * 4
    }
}
