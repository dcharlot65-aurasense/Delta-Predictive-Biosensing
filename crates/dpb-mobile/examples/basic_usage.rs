//! Basic usage example for dpb-mobile
//!
//! This example demonstrates how to:
//! 1. Create a mobile model
//! 2. Configure and build a runtime
//! 3. Perform inference
//! 4. Apply optimizations
//! 5. Benchmark performance

use dpb_mobile::{
    MobileModel, MobileRuntime,
    model::{LayerInfo, LayerType, QuantizationType},
    optimization::{WeightPruner, PruningStrategy, QuantizationOptimizer},
    benchmark::{BenchmarkRunner, BenchmarkConfig, MemoryMetrics},
};

fn main() {
    println!("=== DPB Mobile Runtime Example ===\n");

    // 1. Create a simple model
    println!("1. Creating model...");
    let mut model = MobileModel::new("example_model".into(), 128, 10);

    // Add layers
    let layer1 = LayerInfo {
        name: "input_layer".into(),
        input_dim: 128,
        output_dim: 256,
        layer_type: LayerType::FullyConnected,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; 128 * 256 * 4],
        bias: Some(vec![0u8; 256 * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer1);

    let layer2 = LayerInfo {
        name: "hidden_layer".into(),
        input_dim: 256,
        output_dim: 128,
        layer_type: LayerType::Spiking,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; 256 * 128 * 4],
        bias: Some(vec![0u8; 128 * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer2);

    let layer3 = LayerInfo {
        name: "output_layer".into(),
        input_dim: 128,
        output_dim: 10,
        layer_type: LayerType::Readout,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; 128 * 10 * 4],
        bias: Some(vec![0u8; 10 * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer3);

    println!("   Model: {}", model.name);
    println!("   Input dimension: {}", model.input_dim());
    println!("   Output dimension: {}", model.output_dim());
    println!("   Number of layers: {}", model.num_layers());
    println!("   Weight size: {} KB\n", model.weight_bytes() / 1024);

    // 2. Serialize and deserialize model
    println!("2. Serializing model...");
    let model_bytes = model.to_bytes().expect("Failed to serialize model");
    println!("   Serialized size: {} KB", model_bytes.len() / 1024);

    let model = MobileModel::from_bytes(&model_bytes)
        .expect("Failed to deserialize model");
    println!("   Model deserialized successfully\n");

    // 3. Apply optimizations
    println!("3. Applying optimizations...");

    // Weight pruning
    let mut pruned_model = model.clone();
    let pruner = WeightPruner::new(PruningStrategy::Magnitude, 0.3);
    let pruning_stats = pruner.prune_model(&mut pruned_model)
        .expect("Failed to prune model");

    println!("   Pruning statistics:");
    println!("     Original weights: {}", pruning_stats.original_weights);
    println!("     Pruned weights: {}", pruning_stats.pruned_weights);
    println!("     Sparsity: {:.1}%", pruning_stats.sparsity() * 100.0);
    println!("     Compression ratio: {:.2}x", pruning_stats.compression_ratio());

    // Quantization
    let mut quantized_model = model.clone();
    let quantizer = QuantizationOptimizer::new(QuantizationType::Int8);
    let quant_stats = quantizer.quantize_model(&mut quantized_model)
        .expect("Failed to quantize model");

    println!("   Quantization statistics:");
    println!("     Original size: {} KB", quant_stats.original_size_bytes / 1024);
    println!("     Quantized size: {} KB", quant_stats.quantized_size_bytes / 1024);
    println!("     Compression ratio: {:.2}x\n", quant_stats.compression_ratio());

    // 4. Create runtime
    println!("4. Creating runtime...");
    let mut runtime = MobileRuntime::builder(model)
        .with_max_memory_mb(100)
        .with_thread_count(2)
        .with_batch_size(1)
        .with_operator_fusion(true)
        .with_memory_planning(true)
        .with_simd(true)
        .with_power_mode(1) // Balanced
        .with_warmup_iterations(5)
        .build()
        .expect("Failed to build runtime");

    println!("   Runtime created successfully");
    println!("   Memory usage: {} KB\n", runtime.memory_usage_bytes() / 1024);

    // 5. Perform inference
    println!("5. Performing inference...");
    let input = vec![0.5f32; 128];

    // Single inference
    let output = runtime.infer(&input)
        .expect("Inference failed");

    println!("   Output shape: {}", output.len());
    println!("   Inference count: {}\n", runtime.inference_count());

    // 6. Benchmark performance
    println!("6. Benchmarking performance...");
    let config = BenchmarkConfig {
        warmup_iterations: 10,
        benchmark_iterations: 100,
        batch_size: 1,
        collect_memory: true,
        collect_power: false,
    };

    let mut benchmark_runner = BenchmarkRunner::new(config.clone());

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = runtime.infer(&input);
    }

    // Benchmark
    runtime.reset_statistics();
    let start = std::time::Instant::now();

    for _ in 0..config.benchmark_iterations {
        let iter_start = std::time::Instant::now();
        let _ = runtime.infer(&input).expect("Inference failed");
        let iter_duration = iter_start.elapsed();
        benchmark_runner.record_latency(iter_duration.as_secs_f32() * 1000.0);
    }

    let total_duration = start.elapsed();

    let memory_metrics = MemoryMetrics::new(
        runtime.memory_usage_bytes(),
        runtime.memory_usage_bytes(),
        model_bytes.len(),
        0,
    );

    let result = benchmark_runner.build_result("example_model".into(), memory_metrics);

    println!("\n   Benchmark Results:");
    println!("   ------------------");
    println!("   Model: {}", result.model_name);
    println!("\n   Latency:");
    println!("     Min: {:.2} ms", result.latency.min_ms);
    println!("     Max: {:.2} ms", result.latency.max_ms);
    println!("     Mean: {:.2} ms", result.latency.mean_ms);
    println!("     Median: {:.2} ms", result.latency.median_ms);
    println!("     Std Dev: {:.2} ms", result.latency.std_dev_ms);
    println!("     P95: {:.2} ms", result.latency.p95_ms);
    println!("     P99: {:.2} ms", result.latency.p99_ms);
    println!("\n   Throughput:");
    println!("     Inferences/sec: {:.0}", result.throughput.inferences_per_second);
    println!("     Samples/sec: {:.0}", result.throughput.samples_per_second);
    println!("\n   Memory:");
    println!("     Peak: {:.2} MB", result.memory.peak_mb);
    println!("     Model size: {:.2} MB", result.memory.model_size_bytes as f32 / (1024.0 * 1024.0));
    println!("\n   Device:");
    println!("     Model: {}", result.device_info.model);
    println!("     OS: {}", result.device_info.os);
    println!("     CPU: {}", result.device_info.cpu_arch);
    println!("     Cores: {}", result.device_info.cpu_cores);

    println!("\n=== Example completed successfully ===");
}
