//! # Distributed Training Example
//!
//! Demonstrates how to use the distributed training infrastructure for SNN training.
//!
//! Run with:
//! ```bash
//! # Single node, 4 workers
//! cargo run --example distributed_training --features distributed
//! ```

#[cfg(feature = "distributed")]
use dpb_snn::distributed::{
    coordinator::{AggregationStrategy, TrainingCoordinator},
    partitioning::{DataParallel, ModelParallel, PipelineParallel},
    fault_tolerance::{HeartbeatMonitor, CheckpointManager},
    metrics::DistributedMetrics,
    DistributedBackend, DistributedConfig, DistributedRuntime,
};

#[cfg(feature = "distributed")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Distributed SNN Training Example ===\n");

    // 1. Initialize distributed runtime
    println!("1. Initializing distributed runtime...");
    let config = DistributedConfig {
        world_size: 4,
        rank: 0,
        backend: DistributedBackend::Mock,
        master_addr: "localhost:29500".to_string(),
        timeout: 300,
        compress_gradients: true,
        compression_ratio: 0.01,
    };

    let runtime = DistributedRuntime::init(config)?;
    println!("   ✓ Runtime initialized: {} workers", runtime.world_size());
    println!("   ✓ This worker rank: {}", runtime.rank());
    println!("   ✓ Backend: {}\n", runtime.config().backend.name());

    // 2. Setup training coordinator
    println!("2. Setting up training coordinator...");
    let coordinator = TrainingCoordinator::new(
        runtime.clone(),
        AggregationStrategy::AllReduce,
    );
    println!("   ✓ Aggregation strategy: AllReduce");
    println!("   ✓ Synchronous training enabled\n");

    // 3. Configure data parallelism
    println!("3. Configuring data parallelism...");
    let mut data_parallel = DataParallel::from_runtime(&runtime);
    data_parallel.set_batch_size(128)?;
    println!("   ✓ Global batch size: {}", data_parallel.global_batch_size());
    println!("   ✓ Local batch size: {}", data_parallel.local_batch_size());
    println!("   ✓ Scaled learning rate: {:.6}\n", data_parallel.scaled_learning_rate(0.001));

    // 4. Setup fault tolerance
    println!("4. Setting up fault tolerance...");
    use std::sync::Arc;
    let heartbeat_monitor = HeartbeatMonitor::new(Arc::new(runtime.clone()));
    heartbeat_monitor.start();
    println!("   ✓ Heartbeat monitoring active");

    let checkpoint_dir = std::env::temp_dir().join("dpb_distributed_checkpoints");
    let checkpoint_manager = CheckpointManager::new(&checkpoint_dir)?;
    println!("   ✓ Checkpoint manager initialized");
    println!("   ✓ Checkpoint directory: {:?}\n", checkpoint_dir);

    // 5. Initialize metrics tracking
    println!("5. Initializing metrics tracking...");
    let metrics = DistributedMetrics::new(Arc::new(runtime.clone()));
    println!("   ✓ Performance metrics enabled");
    println!("   ✓ Tracking: throughput, overhead, load balance\n");

    // 6. Simulate training loop
    println!("6. Simulating distributed training...\n");

    for epoch in 0..3 {
        println!("   Epoch {}/3", epoch + 1);

        // Simulate training steps
        for step in 0..5 {
            // Simulate computation
            let compute_time = std::time::Duration::from_millis(100);

            // Simulate gradient aggregation
            let mut gradients = vec![1.0, 2.0, 3.0, 4.0];
            let comm_start = std::time::Instant::now();
            coordinator.aggregate_gradients("layer.weight", &mut gradients)?;
            let comm_time = comm_start.elapsed();

            // Record metrics
            metrics.record_step(data_parallel.local_batch_size(), compute_time, comm_time);

            if step % 2 == 0 {
                println!("     Step {}: throughput = {:.2} samples/sec",
                    step,
                    metrics.current_throughput()
                );
            }
        }

        metrics.record_epoch();

        // Barrier synchronization at epoch end
        coordinator.barrier()?;

        // Save checkpoint (master only)
        if runtime.is_master() {
            let checkpoint_data = vec![1, 2, 3, 4, 5]; // Dummy model state
            let metadata = checkpoint_manager.save(epoch, step_count(epoch), &checkpoint_data)?;
            println!("     ✓ Checkpoint saved: {}", metadata.id);
        }

        println!();
    }

    // 7. Print final metrics
    println!("7. Training Summary:\n");
    let summary = metrics.summary();
    summary.print();

    // 8. Additional metrics
    println!("\n8. Advanced Metrics:");
    println!("   Communication overhead: {:.2}%", metrics.communication_overhead() * 100.0);
    println!("   Scaling efficiency: {:.2}%", metrics.scaling_efficiency() * 100.0);
    println!("   Load balance score: {:.2}/1.0", metrics.load_balance_score());

    // Detect stragglers (slow workers)
    let stragglers = metrics.detect_stragglers(0.2); // 20% threshold
    if !stragglers.is_empty() {
        println!("   ⚠ Stragglers detected: {:?}", stragglers);
    } else {
        println!("   ✓ No stragglers detected");
    }

    // 9. Fault tolerance status
    println!("\n9. Fault Tolerance Status:");
    println!("   Healthy workers: {}/{}", heartbeat_monitor.healthy_count(), runtime.world_size());
    println!("   Checkpoints available: {}", checkpoint_manager.list_checkpoints().len());

    if let Some(latest) = checkpoint_manager.latest_checkpoint() {
        println!("   Latest checkpoint: epoch {}, step {}", latest.epoch, latest.step);
    }

    // 10. Demonstrate different partitioning strategies
    println!("\n10. Partitioning Strategies:\n");
    demonstrate_partitioning_strategies(&runtime)?;

    // 11. Demonstrate aggregation strategies
    println!("\n11. Aggregation Strategies:\n");
    demonstrate_aggregation_strategies(&runtime)?;

    // 12. Cleanup
    println!("\n12. Cleaning up...");
    runtime.shutdown()?;
    std::fs::remove_dir_all(checkpoint_dir).ok();
    println!("   ✓ Distributed runtime shutdown complete\n");

    println!("=== Training Complete ===");

    Ok(())
}

#[cfg(feature = "distributed")]
fn step_count(epoch: usize) -> usize {
    (epoch + 1) * 5
}

#[cfg(feature = "distributed")]
fn demonstrate_partitioning_strategies(runtime: &DistributedRuntime) -> Result<(), Box<dyn std::error::Error>> {
    println!("   a) Data Parallel:");
    let mut dp = DataParallel::from_runtime(runtime);
    dp.set_batch_size(256)?;
    println!("      - Local batch size: {}", dp.local_batch_size());
    println!("      - Data range: {:?}", dp.get_partition_range(1000));

    println!("\n   b) Model Parallel:");
    let mp = ModelParallel::from_runtime(runtime, 16)?;
    println!("      - Layers per worker: ~{}", 16 / runtime.world_size());
    println!("      - Local layers: {:?}", mp.get_local_layers());

    println!("\n   c) Pipeline Parallel:");
    let pp = PipelineParallel::from_runtime(runtime, 16, 8)?;
    println!("      - Stage: {}/{}", pp.stage() + 1, pp.num_stages());
    println!("      - Stage layers: {:?}", pp.get_stage_layers());
    println!("      - Microbatch size: {}", pp.microbatch_size(256));

    Ok(())
}

#[cfg(feature = "distributed")]
fn demonstrate_aggregation_strategies(runtime: &DistributedRuntime) -> Result<(), Box<dyn std::error::Error>> {
    let strategies = vec![
        AggregationStrategy::AllReduce,
        AggregationStrategy::AsyncSGD,
        AggregationStrategy::GossipSGD,
        AggregationStrategy::Hierarchical,
        AggregationStrategy::LocalSGD,
    ];

    for (i, strategy) in strategies.iter().enumerate() {
        println!("   {}) {:?}:", (b'a' + i as u8) as char, strategy);
        println!("      - Synchronous: {}", strategy.is_synchronous());
        println!("      - Communication: {}", strategy.communication_pattern());
        println!("      - Needs coordinator: {}", strategy.needs_coordinator());
    }

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature.");
    println!("Run with: cargo run --example distributed_training --features distributed");
}
