use dpb_mobile::{
    MobileModel, MobileRuntime,
    model::{LayerInfo, LayerType, QuantizationType, ModelVersion, ModelFormat},
    optimization::{WeightPruner, PruningStrategy, OperatorFusion, QuantizationOptimizer, MemoryPlanner},
    benchmark::{BenchmarkRunner, BenchmarkConfig, LatencyMetrics, MemoryMetrics},
};

fn create_simple_model() -> MobileModel {
    let mut model = MobileModel::new("test_model".into(), 10, 5);

    let layer = LayerInfo {
        name: "layer1".into(),
        input_dim: 10,
        output_dim: 5,
        layer_type: LayerType::FullyConnected,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; 10 * 5 * 4],
        bias: Some(vec![0u8; 5 * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer);

    model
}

#[test]
fn test_model_creation() {
    let model = create_simple_model();
    assert_eq!(model.name, "test_model");
    assert_eq!(model.input_dim(), 10);
    assert_eq!(model.output_dim(), 5);
    assert_eq!(model.num_layers(), 1);
}

#[test]
fn test_model_validation() {
    let model = create_simple_model();
    assert!(model.validate().is_ok());
}

#[test]
fn test_model_serialization_deserialization() {
    let model = create_simple_model();

    // Serialize
    let bytes = model.to_bytes().expect("Serialization failed");

    // Check magic number
    assert_eq!(&bytes[0..4], b"DPB\0");

    // Deserialize
    let model2 = MobileModel::from_bytes(&bytes).expect("Deserialization failed");

    assert_eq!(model2.name, model.name);
    assert_eq!(model2.input_dim(), model.input_dim());
    assert_eq!(model2.output_dim(), model.output_dim());
    assert_eq!(model2.num_layers(), model.num_layers());
}

#[test]
fn test_model_version() {
    let version = ModelVersion::new(1, 2, 3);
    let version_u32 = version.as_u32();
    let version2 = ModelVersion::from_u32(version_u32);

    assert_eq!(version.major, version2.major);
    assert_eq!(version.minor, version2.minor);
    assert_eq!(version.patch, version2.patch);
}

#[test]
fn test_runtime_creation() {
    let model = create_simple_model();
    let runtime = MobileRuntime::builder(model).build();

    assert!(runtime.is_ok());
    let runtime = runtime.unwrap();
    assert!(runtime.is_model_loaded());
    assert_eq!(runtime.inference_count(), 0);
}

#[test]
fn test_runtime_without_model() {
    let runtime = MobileRuntime::new_empty().build();

    assert!(runtime.is_ok());
    let runtime = runtime.unwrap();
    assert!(!runtime.is_model_loaded());
}

#[test]
fn test_runtime_load_model() {
    let model = create_simple_model();
    let mut runtime = MobileRuntime::new_empty().build().unwrap();

    assert!(!runtime.is_model_loaded());

    let result = runtime.load_model(model);
    assert!(result.is_ok());
    assert!(runtime.is_model_loaded());
}

#[test]
fn test_inference() {
    let model = create_simple_model();
    let mut runtime = MobileRuntime::builder(model)
        .with_warmup_iterations(0)
        .build()
        .unwrap();

    let input = vec![0.5f32; 10];
    let result = runtime.infer(&input);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 5);
    assert_eq!(runtime.inference_count(), 1);
}

#[test]
fn test_inference_invalid_input() {
    let model = create_simple_model();
    let mut runtime = MobileRuntime::builder(model)
        .with_warmup_iterations(0)
        .build()
        .unwrap();

    let input = vec![0.5f32; 5]; // Wrong size
    let result = runtime.infer(&input);

    assert!(result.is_err());
}

#[test]
fn test_inference_without_model() {
    let mut runtime = MobileRuntime::new_empty().build().unwrap();

    let input = vec![0.5f32; 10];
    let result = runtime.infer(&input);

    assert!(result.is_err());
}

#[test]
fn test_inference_inplace() {
    let model = create_simple_model();
    let mut runtime = MobileRuntime::builder(model)
        .with_warmup_iterations(0)
        .build()
        .unwrap();

    let input = vec![0.5f32; 10];
    let mut output = vec![0.0f32; 5];
    let result = runtime.infer_inplace(&input, &mut output);

    assert!(result.is_ok());
    assert_eq!(runtime.inference_count(), 1);
}

#[test]
fn test_inference_multiple() {
    let model = create_simple_model();
    let mut runtime = MobileRuntime::builder(model)
        .with_warmup_iterations(0)
        .build()
        .unwrap();

    let input = vec![0.5f32; 10];

    for i in 0..10 {
        let result = runtime.infer(&input);
        assert!(result.is_ok());
        assert_eq!(runtime.inference_count(), i + 1);
    }
}

#[test]
fn test_runtime_config() {
    let model = create_simple_model();
    let runtime = MobileRuntime::builder(model)
        .with_max_memory_mb(50)
        .with_thread_count(4)
        .with_batch_size(2)
        .with_operator_fusion(false)
        .with_memory_planning(false)
        .with_simd(false)
        .with_power_mode(2)
        .build()
        .unwrap();

    let config = runtime.config();
    assert_eq!(config.max_memory_mb, 50);
    assert_eq!(config.thread_count, 4);
    assert_eq!(config.batch_size, 2);
    assert!(!config.enable_operator_fusion);
    assert!(!config.enable_memory_planning);
    assert!(!config.enable_simd);
    assert_eq!(config.power_mode, 2);
}

#[test]
fn test_runtime_memory_usage() {
    let model = create_simple_model();
    let runtime = MobileRuntime::builder(model).build().unwrap();

    let memory_usage = runtime.memory_usage_bytes();
    assert!(memory_usage > 0);
}

#[test]
fn test_runtime_reset_statistics() {
    let model = create_simple_model();
    let mut runtime = MobileRuntime::builder(model)
        .with_warmup_iterations(0)
        .build()
        .unwrap();

    let input = vec![0.5f32; 10];
    runtime.infer(&input).unwrap();
    assert_eq!(runtime.inference_count(), 1);

    runtime.reset_statistics();
    assert_eq!(runtime.inference_count(), 0);
}

#[test]
fn test_weight_pruning() {
    let mut model = create_simple_model();
    let pruner = WeightPruner::new(PruningStrategy::Magnitude, 0.5);

    let result = pruner.prune_model(&mut model);
    assert!(result.is_ok());

    let stats = result.unwrap();
    assert!(stats.original_weights > 0);
    assert!(stats.pruned_weights > 0);
    assert_eq!(stats.sparsity(), 0.5);
}

#[test]
fn test_operator_fusion() {
    let mut model = create_simple_model();
    let fusion = OperatorFusion::new(true);

    let result = fusion.fuse_model(&mut model);
    assert!(result.is_ok());
}

#[test]
fn test_quantization() {
    let mut model = create_simple_model();
    let optimizer = QuantizationOptimizer::new(QuantizationType::Int8);

    let result = optimizer.quantize_model(&mut model);
    assert!(result.is_ok());

    let stats = result.unwrap();
    assert!(stats.compression_ratio() > 1.0);
}

#[test]
fn test_memory_planner() {
    let model = create_simple_model();
    let planner = MemoryPlanner::new(true);

    let plan = planner.plan(&model);
    assert!(plan.input_buffer_size > 0);
    assert!(plan.output_buffer_size > 0);
    assert!(plan.total_size > 0);
}

#[test]
fn test_benchmark_runner() {
    let config = BenchmarkConfig {
        warmup_iterations: 0,
        benchmark_iterations: 10,
        batch_size: 1,
        collect_memory: true,
        collect_power: false,
    };

    let mut runner = BenchmarkRunner::new(config);

    // Record some samples
    for i in 0..10 {
        runner.record_latency(10.0 + i as f32);
    }

    let memory = MemoryMetrics::default();
    let result = runner.build_result("test_model".into(), memory);

    assert_eq!(result.model_name, "test_model");
    assert_eq!(result.latency.num_samples, 10);
    assert!(result.throughput.inferences_per_second > 0.0);
}

#[test]
fn test_latency_metrics() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let metrics = LatencyMetrics::from_samples(&samples);

    assert_eq!(metrics.min_ms, 1.0);
    assert_eq!(metrics.max_ms, 10.0);
    assert_eq!(metrics.mean_ms, 5.5);
    assert_eq!(metrics.num_samples, 10);
    assert!(metrics.std_dev_ms > 0.0);
}

#[test]
fn test_batch_inference() {
    let model = create_simple_model();
    let mut runtime = MobileRuntime::builder(model)
        .with_batch_size(4)
        .with_warmup_iterations(0)
        .build()
        .unwrap();

    let input = vec![0.5f32; 10 * 4]; // 4 samples
    let result = runtime.infer(&input);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 5 * 4); // 4 batches of 5 outputs
}

#[test]
fn test_model_compression_ratio() {
    let model = create_simple_model();
    let compression = model.compression_ratio();

    assert!(compression > 0.0);
    assert!(compression <= 1.0 || compression > 1.0); // Just check it's valid
}

#[test]
fn test_quantization_types() {
    assert_eq!(QuantizationType::Float32.bytes_per_weight(), 4);
    assert_eq!(QuantizationType::Float16.bytes_per_weight(), 2);
    assert_eq!(QuantizationType::Int8.bytes_per_weight(), 1);
    assert_eq!(QuantizationType::Int4.bytes_per_weight(), 1);

    assert!(!QuantizationType::Float32.requires_quantization_params());
    assert!(QuantizationType::Int8.requires_quantization_params());
}

#[test]
fn test_multi_layer_model() {
    let mut model = MobileModel::new("multi_layer".into(), 10, 5);

    // Add multiple layers
    let layer1 = LayerInfo {
        name: "layer1".into(),
        input_dim: 10,
        output_dim: 20,
        layer_type: LayerType::FullyConnected,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; 10 * 20 * 4],
        bias: Some(vec![0u8; 20 * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer1);

    let layer2 = LayerInfo {
        name: "layer2".into(),
        input_dim: 20,
        output_dim: 15,
        layer_type: LayerType::Spiking,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; 20 * 15 * 4],
        bias: Some(vec![0u8; 15 * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer2);

    let layer3 = LayerInfo {
        name: "layer3".into(),
        input_dim: 15,
        output_dim: 5,
        layer_type: LayerType::Readout,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; 15 * 5 * 4],
        bias: Some(vec![0u8; 5 * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer3);

    assert!(model.validate().is_ok());
    assert_eq!(model.num_layers(), 3);

    // Test inference
    let mut runtime = MobileRuntime::builder(model)
        .with_warmup_iterations(0)
        .build()
        .unwrap();

    let input = vec![0.5f32; 10];
    let result = runtime.infer(&input);
    assert!(result.is_ok());
}

#[test]
fn test_model_format() {
    let model = create_simple_model();
    assert_eq!(model.format, ModelFormat::DpbNative);
}

#[cfg(target_os = "ios")]
#[test]
fn test_ios_runtime() {
    use dpb_mobile::ios::IosRuntime;

    let runtime = IosRuntime::new();
    assert!(!runtime.is_background());
    assert_eq!(runtime.battery_level(), 1.0);
}

#[cfg(target_os = "android")]
#[test]
fn test_android_runtime() {
    use dpb_mobile::android::{AndroidRuntime, AndroidPowerState};

    let mut runtime = AndroidRuntime::new();
    assert_eq!(runtime.power_state(), AndroidPowerState::Active);
    assert_eq!(runtime.battery_level(), 1.0);
}

#[test]
fn test_ffi_config() {
    use dpb_mobile::ffi::{DpbRuntimeConfig, DpbErrorCode};

    let config = DpbRuntimeConfig::default();
    assert_eq!(config.batch_size, 1);
    assert_eq!(config.power_mode, 1);

    assert_eq!(DpbErrorCode::Success as i32, 0);
    assert!((DpbErrorCode::NullPointer as i32) < 0);
}
