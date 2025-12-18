//! # Training Coordinator
//!
//! Coordinates distributed training across multiple workers, handling gradient
//! aggregation, synchronization, and checkpoint management.

use super::{DistributedError, DistributedResult, DistributedRuntime, ReduceOp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Gradient aggregation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Synchronous all-reduce (blocking)
    AllReduce,
    /// Asynchronous SGD (non-blocking)
    AsyncSGD,
    /// Decentralized gossip protocol
    GossipSGD,
    /// Hierarchical aggregation (tree-based)
    Hierarchical,
    /// Local SGD (periodic synchronization)
    LocalSGD,
}

impl AggregationStrategy {
    /// Check if strategy is synchronous
    pub fn is_synchronous(&self) -> bool {
        matches!(self, Self::AllReduce | Self::Hierarchical)
    }

    /// Check if strategy requires central coordinator
    pub fn needs_coordinator(&self) -> bool {
        matches!(self, Self::AllReduce | Self::Hierarchical | Self::AsyncSGD)
    }

    /// Get communication pattern
    pub fn communication_pattern(&self) -> &'static str {
        match self {
            Self::AllReduce => "all-to-all",
            Self::AsyncSGD => "worker-to-master",
            Self::GossipSGD => "peer-to-peer",
            Self::Hierarchical => "tree-based",
            Self::LocalSGD => "periodic-all-to-all",
        }
    }
}

/// Training coordinator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Aggregation strategy
    pub strategy: AggregationStrategy,
    /// Synchronization frequency (steps)
    pub sync_frequency: usize,
    /// Enable gradient clipping
    pub gradient_clipping: bool,
    /// Gradient clipping threshold
    pub clip_threshold: f32,
    /// Enable gradient accumulation
    pub gradient_accumulation: bool,
    /// Accumulation steps
    pub accumulation_steps: usize,
    /// Checkpoint frequency (epochs)
    pub checkpoint_frequency: usize,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            strategy: AggregationStrategy::AllReduce,
            sync_frequency: 1,
            gradient_clipping: true,
            clip_threshold: 1.0,
            gradient_accumulation: false,
            accumulation_steps: 1,
            checkpoint_frequency: 1,
        }
    }
}

/// Training coordinator
pub struct TrainingCoordinator {
    runtime: DistributedRuntime,
    config: CoordinatorConfig,
    state: Arc<RwLock<CoordinatorState>>,
}

/// Coordinator internal state
#[derive(Debug)]
struct CoordinatorState {
    /// Accumulated gradients
    gradient_accumulator: HashMap<String, Vec<f32>>,
    /// Number of accumulated steps
    accumulated_steps: usize,
    /// Last synchronization time
    last_sync: Instant,
    /// Synchronization count
    sync_count: usize,
    /// Pending updates (for async strategies)
    pending_updates: Vec<GradientUpdate>,
    /// Worker staleness (for async strategies)
    worker_staleness: HashMap<usize, usize>,
}

/// Gradient update message
#[derive(Debug, Clone)]
struct GradientUpdate {
    /// Source worker rank
    rank: usize,
    /// Parameter name
    param_name: String,
    /// Gradient values
    gradients: Vec<f32>,
    /// Update timestamp
    timestamp: Instant,
    /// Staleness (async strategies)
    staleness: usize,
}

impl TrainingCoordinator {
    /// Create new training coordinator
    pub fn new(runtime: DistributedRuntime, strategy: AggregationStrategy) -> Self {
        let config = CoordinatorConfig {
            strategy,
            ..Default::default()
        };

        Self::with_config(runtime, config)
    }

    /// Create coordinator with custom configuration
    pub fn with_config(runtime: DistributedRuntime, config: CoordinatorConfig) -> Self {
        let state = CoordinatorState {
            gradient_accumulator: HashMap::new(),
            accumulated_steps: 0,
            last_sync: Instant::now(),
            sync_count: 0,
            pending_updates: Vec::new(),
            worker_staleness: HashMap::new(),
        };

        Self {
            runtime,
            config,
            state: Arc::new(RwLock::new(state)),
        }
    }

    /// Get runtime
    pub fn runtime(&self) -> &DistributedRuntime {
        &self.runtime
    }

    /// Get configuration
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }

    /// Synchronize all workers (barrier)
    pub fn barrier(&self) -> DistributedResult<()> {
        self.runtime.barrier()
    }

    /// Aggregate gradients across workers
    pub fn aggregate_gradients(
        &self,
        param_name: &str,
        gradients: &mut [f32],
    ) -> DistributedResult<()> {
        match self.config.strategy {
            AggregationStrategy::AllReduce => self.all_reduce_gradients(gradients),
            AggregationStrategy::AsyncSGD => self.async_sgd_gradients(param_name, gradients),
            AggregationStrategy::GossipSGD => self.gossip_sgd_gradients(param_name, gradients),
            AggregationStrategy::Hierarchical => self.hierarchical_gradients(gradients),
            AggregationStrategy::LocalSGD => self.local_sgd_gradients(param_name, gradients),
        }
    }

    /// AllReduce gradient aggregation (synchronous)
    fn all_reduce_gradients(&self, gradients: &mut [f32]) -> DistributedResult<()> {
        // Clip gradients if enabled
        if self.config.gradient_clipping {
            clip_gradients(gradients, self.config.clip_threshold);
        }

        // All-reduce across workers (sum)
        self.runtime
            .backend()
            .all_reduce(gradients, ReduceOp::Sum)?;

        // Average by world size
        let world_size = self.runtime.world_size() as f32;
        for grad in gradients.iter_mut() {
            *grad /= world_size;
        }

        // Update sync count
        self.state.write().unwrap().sync_count += 1;

        Ok(())
    }

    /// Asynchronous SGD (parameter server style)
    fn async_sgd_gradients(
        &self,
        param_name: &str,
        gradients: &mut [f32],
    ) -> DistributedResult<()> {
        let rank = self.runtime.rank();
        let is_master = self.runtime.is_master();

        if is_master {
            // Master: aggregate updates from workers
            let mut state = self.state.write().unwrap();

            // Process pending updates
            for update in state.pending_updates.drain(..) {
                if update.param_name == param_name {
                    // Apply staleness-aware update
                    let staleness_factor = 1.0 / (1.0 + update.staleness as f32);
                    for (i, &grad) in update.gradients.iter().enumerate() {
                        if i < gradients.len() {
                            gradients[i] += grad * staleness_factor;
                        }
                    }
                }
            }

            // Average
            let num_workers = self.runtime.world_size() as f32;
            for grad in gradients.iter_mut() {
                *grad /= num_workers;
            }
        } else {
            // Worker: send gradients to master
            self.runtime.backend().send(gradients, 0, 0)?;

            // Receive updated gradients
            self.runtime.backend().recv(gradients, 0, 0)?;

            // Update staleness
            let mut state = self.state.write().unwrap();
            *state.worker_staleness.entry(rank).or_insert(0) += 1;
        }

        Ok(())
    }

    /// Gossip SGD (decentralized)
    fn gossip_sgd_gradients(
        &self,
        _param_name: &str,
        gradients: &mut [f32],
    ) -> DistributedResult<()> {
        let rank = self.runtime.rank();
        let world_size = self.runtime.world_size();

        // Select random peer (ring topology for simplicity)
        let peer = (rank + 1) % world_size;

        // Exchange gradients with peer
        let mut peer_gradients = vec![0.0; gradients.len()];

        // Simple exchange (in production, use non-blocking send/recv)
        if rank % 2 == 0 {
            self.runtime.backend().send(gradients, peer, 0)?;
            self.runtime
                .backend()
                .recv(&mut peer_gradients, peer, 0)?;
        } else {
            self.runtime
                .backend()
                .recv(&mut peer_gradients, peer, 0)?;
            self.runtime.backend().send(gradients, peer, 0)?;
        }

        // Average with peer
        for (i, peer_grad) in peer_gradients.iter().enumerate() {
            gradients[i] = (gradients[i] + peer_grad) / 2.0;
        }

        Ok(())
    }

    /// Hierarchical aggregation (tree-based)
    fn hierarchical_gradients(&self, gradients: &mut [f32]) -> DistributedResult<()> {
        let rank = self.runtime.rank();
        let world_size = self.runtime.world_size();

        // Build binary tree structure
        let mut level = 0;
        let mut stride = 1;

        while stride < world_size {
            let is_receiver = rank % (2 * stride) == 0;
            let is_sender = rank % (2 * stride) == stride;

            if is_receiver && rank + stride < world_size {
                // Receive from child
                let mut child_gradients = vec![0.0; gradients.len()];
                self.runtime
                    .backend()
                    .recv(&mut child_gradients, rank + stride, level)?;

                // Aggregate
                for (i, child_grad) in child_gradients.iter().enumerate() {
                    gradients[i] += child_grad;
                }
            } else if is_sender {
                // Send to parent
                let parent = rank - stride;
                self.runtime.backend().send(gradients, parent, level)?;
                break;
            }

            stride *= 2;
            level += 1;
        }

        // Broadcast result from root
        self.runtime.backend().broadcast(gradients, 0)?;

        // Average by world size
        let world_size_f = world_size as f32;
        for grad in gradients.iter_mut() {
            *grad /= world_size_f;
        }

        Ok(())
    }

    /// Local SGD (periodic synchronization)
    fn local_sgd_gradients(
        &self,
        param_name: &str,
        gradients: &mut [f32],
    ) -> DistributedResult<()> {
        // First, accumulate gradients and check if we should sync
        let (should_sync, local_grads) = {
            let mut state = self.state.write().unwrap();

            // Accumulate gradients locally
            {
                let accumulated = state
                    .gradient_accumulator
                    .entry(param_name.to_string())
                    .or_insert_with(|| vec![0.0; gradients.len()]);

                for (i, &grad) in gradients.iter().enumerate() {
                    accumulated[i] += grad;
                }
            } // Drop accumulated reference here

            state.accumulated_steps += 1;

            // Check if we should synchronize
            let should_sync = state.accumulated_steps >= self.config.sync_frequency;

            // Get local gradients if not syncing
            let local_grads = if !should_sync {
                state.gradient_accumulator
                    .get(param_name)
                    .map(|g| g.clone())
            } else {
                None
            };

            (should_sync, local_grads)
        };

        // If not syncing, use local gradients
        if !should_sync {
            if let Some(local) = local_grads {
                gradients.copy_from_slice(&local);
            }
            return Ok(());
        }

        // Synchronize periodically (outside the lock)
        self.runtime
            .backend()
            .all_reduce(gradients, ReduceOp::Sum)?;

        let world_size = self.runtime.world_size() as f32;
        for grad in gradients.iter_mut() {
            *grad /= world_size;
        }

        // Reset accumulator
        let mut state = self.state.write().unwrap();
        state.gradient_accumulator.clear();
        state.accumulated_steps = 0;
        state.last_sync = Instant::now();

        Ok(())
    }

    /// Broadcast model parameters from master to all workers
    pub fn broadcast_parameters(&self, parameters: &mut [f32]) -> DistributedResult<()> {
        self.runtime.backend().broadcast(parameters, 0)
    }

    /// Synchronize checkpoint across workers
    pub fn synchronize_checkpoint(&self, checkpoint_data: &mut [u8]) -> DistributedResult<()> {
        // Convert to f32 for communication (TODO: support byte arrays)
        let len = checkpoint_data.len();
        let mut float_data = vec![0.0f32; len];

        // Convert bytes to floats for transmission
        for (i, &byte) in checkpoint_data.iter().enumerate() {
            float_data[i] = byte as f32;
        }

        // Broadcast
        self.runtime.backend().broadcast(&mut float_data, 0)?;

        // Convert back
        for (i, &val) in float_data.iter().enumerate() {
            checkpoint_data[i] = val as u8;
        }

        Ok(())
    }

    /// Get synchronization statistics
    pub fn sync_stats(&self) -> SyncStats {
        let state = self.state.read().unwrap();
        SyncStats {
            sync_count: state.sync_count,
            time_since_last_sync: state.last_sync.elapsed(),
            accumulated_steps: state.accumulated_steps,
            pending_updates: state.pending_updates.len(),
        }
    }

    /// Reset coordinator state
    pub fn reset(&self) {
        let mut state = self.state.write().unwrap();
        state.gradient_accumulator.clear();
        state.accumulated_steps = 0;
        state.pending_updates.clear();
        state.worker_staleness.clear();
        state.sync_count = 0;
    }
}

/// Synchronization statistics
#[derive(Debug, Clone)]
pub struct SyncStats {
    /// Total synchronization count
    pub sync_count: usize,
    /// Time since last sync
    pub time_since_last_sync: Duration,
    /// Accumulated steps
    pub accumulated_steps: usize,
    /// Pending updates
    pub pending_updates: usize,
}

/// Clip gradients to threshold
fn clip_gradients(gradients: &mut [f32], threshold: f32) {
    let norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();

    if norm > threshold {
        let scale = threshold / norm;
        for grad in gradients.iter_mut() {
            *grad *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{DistributedBackend, DistributedConfig};

    #[test]
    fn test_aggregation_strategy_properties() {
        assert!(AggregationStrategy::AllReduce.is_synchronous());
        assert!(!AggregationStrategy::AsyncSGD.is_synchronous());
        assert!(AggregationStrategy::AllReduce.needs_coordinator());
        assert!(!AggregationStrategy::GossipSGD.needs_coordinator());
    }

    #[test]
    fn test_coordinator_creation() {
        let config = DistributedConfig::new(2, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let coordinator = TrainingCoordinator::new(runtime, AggregationStrategy::AllReduce);

        assert_eq!(
            coordinator.config().strategy,
            AggregationStrategy::AllReduce
        );
    }

    #[test]
    fn test_all_reduce_gradients() {
        let config = DistributedConfig::new(2, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let coordinator = TrainingCoordinator::new(runtime, AggregationStrategy::AllReduce);

        let mut gradients = vec![1.0, 2.0, 3.0, 4.0];
        let original_norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();

        coordinator
            .aggregate_gradients("test", &mut gradients)
            .unwrap();

        // Gradient clipping is enabled by default with threshold 1.0
        // Original norm = sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.477
        // After clipping: gradients are scaled to norm 1.0
        // Mock backend multiplies by world_size (2) during all_reduce
        // Then we divide by world_size (2) again: final norm = 1.0
        let final_norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();

        // The norm should be reduced from original due to clipping
        assert!(final_norm < original_norm);
        // After clipping to 1.0 and averaging, final norm should be ≈ 1.0
        assert!((final_norm - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut gradients = vec![3.0, 4.0]; // norm = 5.0
        clip_gradients(&mut gradients, 2.0);

        // Should be scaled to norm 2.0
        let norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((norm - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sync_stats() {
        let config = DistributedConfig::new(2, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let coordinator = TrainingCoordinator::new(runtime, AggregationStrategy::AllReduce);

        let stats = coordinator.sync_stats();
        assert_eq!(stats.sync_count, 0);
        assert_eq!(stats.accumulated_steps, 0);
    }

    #[test]
    fn test_broadcast_parameters() {
        let config = DistributedConfig::new(2, 0, DistributedBackend::Mock);
        let runtime = DistributedRuntime::init(config).unwrap();
        let coordinator = TrainingCoordinator::new(runtime, AggregationStrategy::AllReduce);

        let mut params = vec![1.0, 2.0, 3.0];
        assert!(coordinator.broadcast_parameters(&mut params).is_ok());
    }
}
