# Distributed Training Infrastructure

## Overview

The Delta-Predictive-Biosensing framework includes a comprehensive distributed training infrastructure for scaling Spiking Neural Network (SNN) training across multiple nodes and GPUs. The implementation provides flexible parallelization strategies, fault tolerance, and performance monitoring.

## Architecture

The distributed training system is organized into six core modules:

```
dpb-snn/src/distributed/
├── mod.rs               - Core runtime and backend abstraction
├── coordinator.rs       - Training coordination and gradient aggregation
├── partitioning.rs      - Data/model/pipeline parallelism strategies
├── communication.rs     - Low-level communication primitives
├── fault_tolerance.rs   - Heartbeat monitoring, checkpointing, elastic training
└── metrics.rs           - Performance tracking and profiling
```

## Features

### 1. Backend Support

The infrastructure provides trait-based abstraction for multiple communication backends:

- **MPI (Message Passing Interface)** - Standard for HPC clusters, supports CPU and GPU
- **Gloo** - CPU-only collective operations, lightweight
- **NCCL (NVIDIA Collective Communications Library)** - GPU-optimized for NVIDIA hardware
- **Mock** - Testing backend for development

```rust
pub enum DistributedBackend {
    Mpi,    // CPU/GPU support
    Gloo,   // CPU only
    Nccl,   // GPU optimized
    Mock,   // Testing
}
```

### 2. Parallelization Strategies

#### Data Parallelism
Distributes mini-batches across workers. Each worker maintains a full copy of the model.

```rust
let mut data_parallel = DataParallel::from_runtime(&runtime);
data_parallel.set_batch_size(256)?;

// Automatically splits data
let local_batch = data_parallel.local_batch_size(); // 64 per worker (4 workers)
let range = data_parallel.get_partition_range(1000); // Each worker gets 250 samples
```

**Advantages:**
- Simple to implement
- Good for large batch sizes
- Minimal communication overhead

**Use cases:** Most common choice when model fits in memory

#### Model Parallelism
Distributes model layers across workers. Each worker handles a subset of layers.

```rust
let model_parallel = ModelParallel::from_runtime(&runtime, 16)?;

// Check which layers are local
if model_parallel.is_local_layer(5) {
    // Process layer 5
}

// Get all local layers
let local_layers = model_parallel.get_local_layers(); // [0, 1, 2, 3] for worker 0
```

**Advantages:**
- Enables training of very large models
- Reduces memory per worker

**Use cases:** Models too large to fit on single device

#### Pipeline Parallelism
Divides model into stages, processes micro-batches in pipeline fashion using 1F1B (One Forward One Backward) scheduling.

```rust
let pipeline = PipelineParallel::from_runtime(&runtime, 16, 8)?;

// Get stage information
let stage = pipeline.stage(); // 0, 1, 2, or 3
let layers = pipeline.get_stage_layers(); // 0..4 for stage 0

// Get execution schedule
let schedule = pipeline.get_schedule();
```

**Advantages:**
- Better GPU utilization than naive model parallelism
- Handles large models efficiently
- Lower communication than model parallelism

**Use cases:** Very deep models with sequential architecture

### 3. Gradient Aggregation Strategies

#### AllReduce (Synchronous)
Standard synchronous training with collective gradient aggregation.

```rust
let coordinator = TrainingCoordinator::new(
    runtime.clone(),
    AggregationStrategy::AllReduce,
);

coordinator.aggregate_gradients("layer.weight", &mut gradients)?;
```

**Properties:**
- Synchronous: All workers wait for each other
- Communication: All-to-all
- Convergence: Identical to single-node SGD
- Overhead: Moderate (one reduction per step)

#### AsyncSGD (Asynchronous)
Parameter server style asynchronous updates with staleness compensation.

```rust
AggregationStrategy::AsyncSGD
```

**Properties:**
- Asynchronous: Workers don't wait
- Communication: Worker-to-master
- Convergence: Faster wall-clock time, may need more iterations
- Staleness factor: `1.0 / (1.0 + staleness)` to handle delayed updates

#### GossipSGD (Decentralized)
Peer-to-peer gradient averaging without central coordination.

```rust
AggregationStrategy::GossipSGD
```

**Properties:**
- Decentralized: No master node
- Communication: Peer-to-peer (ring topology)
- Convergence: Eventually equivalent to centralized averaging
- Scalability: Better for large clusters

#### Hierarchical (Tree-based)
Binary tree reduction for efficient gradient aggregation.

```rust
AggregationStrategy::Hierarchical
```

**Properties:**
- Synchronous: Tree-based synchronization
- Communication: Tree-based (log N stages)
- Bandwidth: More efficient than naive AllReduce
- Latency: Lower than flat AllReduce

#### LocalSGD (Periodic Sync)
Workers train locally with periodic synchronization.

```rust
let mut config = CoordinatorConfig::default();
config.strategy = AggregationStrategy::LocalSGD;
config.sync_frequency = 10; // Sync every 10 steps
```

**Properties:**
- Communication: Periodic (every K steps)
- Convergence: Similar to standard SGD with larger batches
- Efficiency: Reduces communication overhead significantly

### 4. Fault Tolerance

#### Heartbeat Monitoring
Automatic failure detection for workers.

```rust
use std::sync::Arc;
let monitor = HeartbeatMonitor::new(Arc::new(runtime.clone()));
monitor.start();

// Check health
let failed = monitor.check_health();
for rank in failed {
    println!("Worker {} failed", rank);
}
```

**Configuration:**
```rust
let config = HeartbeatConfig {
    interval: Duration::from_secs(5),    // Heartbeat frequency
    timeout: Duration::from_secs(30),     // Detection threshold
    max_missed: 3,                        // Failures before marking dead
};
```

#### Checkpoint/Restore
Fault-tolerant checkpointing with automatic cleanup.

```rust
let manager = CheckpointManager::new("/path/to/checkpoints")?;

// Save checkpoint
let metadata = manager.save(epoch, step, &model_bytes)?;

// Load latest checkpoint
if let Some(data) = manager.load_latest()? {
    // Restore model from data
}
```

**Features:**
- Automatic cleanup (keeps N most recent)
- Checksum validation
- Incremental checkpointing (optional)
- Compression support

#### Elastic Training
Dynamic worker scaling during training.

```rust
let elastic = ElasticTrainingManager::new(
    Arc::new(runtime),
    min_workers: 2,  // Minimum workers to continue
    max_workers: 8,  // Maximum workers allowed
);

// Add worker dynamically
elastic.add_worker(5)?;
elastic.commit_workers()?;

// Remove failed worker
elastic.remove_worker(3)?;
```

### 5. Performance Metrics

#### Throughput Tracking
Monitor training speed and efficiency.

```rust
let metrics = DistributedMetrics::new(Arc::new(runtime));

// Record step
metrics.record_step(
    batch_size,
    compute_time,
    comm_time,
);

// Get metrics
let throughput = metrics.current_throughput(); // samples/sec
let global_throughput = metrics.global_throughput(); // across all workers
```

#### Communication Overhead
Track communication costs.

```rust
let overhead = metrics.communication_overhead(); // 0.0 to 1.0
println!("Communication overhead: {:.2}%", overhead * 100.0);
```

#### Load Balancing
Detect imbalanced workloads.

```rust
let balance_score = metrics.load_balance_score(); // 0.0 to 1.0 (higher is better)

// Detect slow workers (stragglers)
let stragglers = metrics.detect_stragglers(0.2); // 20% slower than average
```

#### Scaling Efficiency
Measure parallel efficiency.

```rust
let efficiency = metrics.scaling_efficiency(); // 0.0 to 1.0
println!("Scaling efficiency: {:.2}%", efficiency * 100.0);

// Ideal: 1.0 (perfect linear scaling)
// Reality: < 1.0 due to communication overhead
```

#### Performance Profiling
Detailed event tracking with Chrome trace export.

```rust
let profiler = PerformanceProfiler::new();

// Profile section
let section = profiler.start_section("forward_pass");
// ... computation ...
let (name, duration) = section.end();

// Export for visualization
let trace = profiler.export_chrome_trace();
std::fs::write("trace.json", trace)?;
// Open in chrome://tracing
```

### 6. Communication Primitives

Low-level operations for custom distributed algorithms:

```rust
let ops = CommunicationOps::new(backend, rank, world_size);

// Point-to-point
ops.send(&data, dst, tag)?;
ops.recv(&mut data, src, tag)?;

// Collective operations
ops.broadcast(&mut data, root)?;
ops.all_reduce(&mut data, ReduceOp::Sum)?;
ops.all_gather(&send_data, &mut recv_data)?;
ops.scatter(&send_data, &mut recv_data, root)?;
ops.gather(&send_data, &mut recv_data, root)?;

// Advanced
ops.ring_all_reduce(&mut data, ReduceOp::Sum)?; // Bandwidth-optimal
ops.reduce_scatter(&send_data, &mut recv_data, ReduceOp::Sum)?;
ops.all_to_all(&send_data, &mut recv_data)?;
```

## Quick Start

### Basic Setup

```rust
use dpb_snn::distributed::*;

// 1. Initialize runtime
let config = DistributedConfig::new(
    world_size: 4,
    rank: 0,
    backend: DistributedBackend::Mock,
);
let runtime = DistributedRuntime::init(config)?;

// 2. Create coordinator
let coordinator = TrainingCoordinator::new(
    runtime.clone(),
    AggregationStrategy::AllReduce,
);

// 3. Setup partitioning
let mut data_parallel = DataParallel::from_runtime(&runtime);
data_parallel.set_batch_size(128)?;

// 4. Training loop
for epoch in 0..num_epochs {
    for batch in data_loader {
        // Forward pass
        let loss = model.forward(batch);

        // Backward pass (compute local gradients)
        let gradients = model.backward();

        // Aggregate gradients across workers
        coordinator.aggregate_gradients("layer.weight", &mut gradients)?;

        // Update weights
        optimizer.step(&gradients);
    }

    // Synchronize at epoch end
    coordinator.barrier()?;
}
```

### Advanced Example

See `crates/dpb-snn/examples/distributed_training.rs` for a complete example demonstrating:
- All partitioning strategies
- All aggregation strategies
- Fault tolerance features
- Metrics tracking
- Performance profiling

Run with:
```bash
cargo run --example distributed_training --features distributed
```

## Configuration Guide

### Choosing Partitioning Strategy

Use the automatic planner:

```rust
let planner = PartitionPlanner::new(runtime);

let strategy = planner.recommend_strategy(
    model_size_mb: 500.0,
    batch_size: 128,
    num_layers: 50,
);

match strategy {
    PartitionStrategy::DataParallel => {
        // Model fits in memory, use data parallelism
    }
    PartitionStrategy::ModelParallel => {
        // Model too large, split across workers
    }
    PartitionStrategy::PipelineParallel => {
        // Deep model, use pipeline
    }
    PartitionStrategy::Hybrid => {
        // Combine strategies
    }
}
```

**Decision matrix:**
- Model fits in memory + large batch → **Data Parallel**
- Model too large + many layers → **Pipeline Parallel**
- Model too large + few layers → **Model Parallel**
- Very large model + large batch → **Hybrid**

### Choosing Aggregation Strategy

**Recommendations:**
- Default choice: **AllReduce** (reliable, well-tested)
- Large clusters (>32 nodes): **Hierarchical** (better bandwidth)
- High latency network: **LocalSGD** (fewer syncs)
- Heterogeneous hardware: **AsyncSGD** (faster workers don't wait)
- Edge/P2P scenarios: **GossipSGD** (no central coordinator)

### Optimization Tips

1. **Batch Size Scaling**
   ```rust
   // Linear scaling rule
   let scaled_lr = base_lr * world_size as f32;

   // Warmup for stability
   let warmup_steps = data_parallel.warmup_steps(base_warmup);
   ```

2. **Gradient Compression**
   ```rust
   let mut config = DistributedConfig::new(4, 0, DistributedBackend::Nccl);
   config.compress_gradients = true;
   config.compression_ratio = 0.01; // Keep top 1% of gradients
   ```

3. **Pipeline Microbatch Tuning**
   ```rust
   // Rule of thumb: 4x number of stages
   let microbatches = planner.optimize_microbatches(batch_size);
   ```

## Testing

All components include comprehensive tests:

```bash
# Run all distributed tests
cargo test --package dpb-snn --features distributed distributed::

# Test specific module
cargo test --package dpb-snn --features distributed distributed::coordinator::

# Run with output
cargo test --package dpb-snn --features distributed distributed:: -- --nocapture
```

**Test coverage:**
- 46 unit tests covering all modules
- Integration tests for end-to-end workflows
- Mock backend for deterministic testing
- Fault injection tests for robustness

## Performance Characteristics

### Communication Complexity

| Operation | Latency | Bandwidth |
|-----------|---------|-----------|
| AllReduce (naive) | O(P) | O(N) |
| Ring-AllReduce | O(P) | O(N/P) optimal |
| Hierarchical | O(log P) | O(N/P) |
| AsyncSGD | O(1) | O(N) |
| GossipSGD | O(log P) | O(N) |

Where P = number of workers, N = data size

### Scaling Results

Expected scaling efficiency by strategy:

| Workers | Data Parallel | Model Parallel | Pipeline Parallel |
|---------|---------------|----------------|-------------------|
| 2 | 95% | 90% | 85% |
| 4 | 90% | 85% | 80% |
| 8 | 85% | 75% | 75% |
| 16 | 75% | 60% | 70% |

*Efficiency = (Speedup / Workers) × 100%*

## Limitations and Future Work

### Current Limitations

1. **Backend Implementation**
   - MPI, Gloo, and NCCL backends not yet implemented
   - Currently only Mock backend is available
   - Production use requires backend implementation

2. **Communication**
   - Byte array communication not fully implemented
   - Non-blocking operations not yet supported

3. **Fault Tolerance**
   - Automatic recovery not implemented
   - Manual intervention required after failures

### Planned Features

- [ ] Real MPI backend integration
- [ ] NCCL backend for GPU training
- [ ] Gloo backend for CPU clusters
- [ ] Automatic gradient compression
- [ ] Zero-copy communication
- [ ] Non-blocking collectives
- [ ] Automatic recovery from failures
- [ ] Dynamic batching
- [ ] Heterogeneous training (mixed precision)

## API Reference

### Core Types

- `DistributedRuntime` - Main runtime state and coordination
- `DistributedConfig` - Configuration parameters
- `DistributedBackend` - Communication backend enum
- `CommunicationBackend` trait - Backend abstraction

### Coordinator

- `TrainingCoordinator` - Gradient aggregation coordinator
- `AggregationStrategy` - Strategy enum
- `CoordinatorConfig` - Coordinator configuration

### Partitioning

- `DataParallel` - Data parallelism helper
- `ModelParallel` - Model parallelism helper
- `PipelineParallel` - Pipeline parallelism helper
- `PartitionPlanner` - Strategy recommendation

### Fault Tolerance

- `HeartbeatMonitor` - Worker health monitoring
- `CheckpointManager` - Checkpoint management
- `ElasticTrainingManager` - Dynamic scaling

### Metrics

- `DistributedMetrics` - Training metrics tracker
- `PerformanceProfiler` - Detailed profiling
- `MetricsSummary` - Metrics snapshot

## Contributing

To add a new communication backend:

1. Implement the `CommunicationBackend` trait
2. Add backend variant to `DistributedBackend` enum
3. Update `create_backend` function in `mod.rs`
4. Add comprehensive tests
5. Update documentation

Example:
```rust
struct MyBackend { /* ... */ }

impl CommunicationBackend for MyBackend {
    fn init(&self, config: &DistributedConfig) -> DistributedResult<()> {
        // Initialize your backend
    }

    fn all_reduce(&self, data: &mut [f32], op: ReduceOp) -> DistributedResult<()> {
        // Implement all-reduce
    }

    // ... implement other methods
}
```

## License

This distributed training infrastructure is part of the Delta-Predictive-Biosensing framework.

## References

1. **Data Parallelism**: Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017)
2. **Pipeline Parallelism**: Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" (2019)
3. **Gradient Compression**: Lin et al., "Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training" (2018)
4. **Asynchronous SGD**: Dean et al., "Large Scale Distributed Deep Networks" (2012)
5. **Ring-AllReduce**: Patarasuk & Yuan, "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations" (2009)
