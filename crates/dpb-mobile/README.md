# dpb-mobile

Mobile runtime for iOS and Android inference in the Delta-Predictive-Biosensing (DPB) Framework.

## Overview

`dpb-mobile` provides an optimized inference runtime for mobile devices, with platform-specific optimizations for iOS and Android. It features:

- **Low memory footprint** - Optimized for constrained mobile environments
- **Quantized model support** - Int8, Float16 quantization for smaller model sizes
- **Platform-specific backends** - Metal (iOS), CoreML (iOS), NNAPI (Android), Vulkan (Android)
- **Battery optimization** - Power-aware inference throttling
- **C FFI** - Easy integration with Swift, Objective-C, Kotlin, and Java
- **Performance benchmarking** - Built-in tools for measuring latency, throughput, and memory usage

## Features

### Core Features

- ✅ Mobile-optimized inference runtime
- ✅ Model compression and quantization
- ✅ Weight pruning
- ✅ Operator fusion
- ✅ Memory planning
- ✅ SIMD optimizations (ARM NEON)
- ✅ C FFI for native integration
- ✅ Comprehensive benchmarking tools

### Platform-Specific Features

#### iOS
- Metal GPU acceleration (planned)
- CoreML integration (planned)
- Background processing support
- Battery and thermal monitoring
- Swift-friendly API wrappers

#### Android
- NNAPI acceleration (planned)
- Vulkan compute backend (planned)
- Doze mode handling
- Thermal throttling support
- JNI bindings structure

## Building

### Prerequisites

1. Install Rust (1.92.0 or later):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install target toolchains:

#### For iOS

```bash
# iOS ARM64 (devices)
rustup target add aarch64-apple-ios

# iOS ARM64 simulator
rustup target add aarch64-apple-ios-sim

# iOS x86_64 simulator (Intel Macs)
rustup target add x86_64-apple-ios
```

#### For Android

```bash
# Android ARM64
rustup target add aarch64-linux-android

# Android ARMv7
rustup target add armv7-linux-androideabi

# Android x86_64 (emulator)
rustup target add x86_64-linux-android

# Android i686 (emulator)
rustup target add i686-linux-android
```

3. Install Android NDK (for Android builds):
```bash
# Using Android Studio
# Or download from: https://developer.android.com/ndk/downloads

# Set NDK path
export ANDROID_NDK_HOME=/path/to/ndk
```

### Build Commands

#### iOS Builds

```bash
# iOS ARM64 (release build for devices)
cargo build --target aarch64-apple-ios --features ios --release

# iOS simulator
cargo build --target aarch64-apple-ios-sim --features ios --release

# With Metal support (when available)
cargo build --target aarch64-apple-ios --features "ios,metal" --release

# With CoreML support (when available)
cargo build --target aarch64-apple-ios --features "ios,coreml" --release
```

#### Android Builds

```bash
# Android ARM64 (release build)
cargo build --target aarch64-linux-android --features android --release

# Android ARMv7 (for older devices)
cargo build --target armv7-linux-androideabi --features android --release

# With NNAPI support (when available)
cargo build --target aarch64-linux-android --features "android,nnapi" --release

# With Vulkan support (when available)
cargo build --target aarch64-linux-android --features "android,vulkan" --release
```

#### Universal Binary (iOS)

Create a universal iOS library:

```bash
# Build for both device and simulator
cargo build --target aarch64-apple-ios --features ios --release
cargo build --target aarch64-apple-ios-sim --features ios --release

# Create universal binary
lipo -create \
    target/aarch64-apple-ios/release/libdpb_mobile.a \
    target/aarch64-apple-ios-sim/release/libdpb_mobile.a \
    -output libdpb_mobile_universal.a
```

#### Host Platform (for testing)

```bash
# Build and test on host platform
cargo build --release
cargo test --release
cargo bench
```

## Usage

### Rust API

```rust
use dpb_mobile::{MobileModel, MobileRuntime};
use dpb_mobile::model::{LayerInfo, LayerType, QuantizationType};

// Load model from bytes
let model_bytes = include_bytes!("path/to/model.dpb");
let model = MobileModel::from_bytes(model_bytes)?;

// Create runtime with configuration
let mut runtime = MobileRuntime::new(model)
    .with_max_memory_mb(50)
    .with_thread_count(2)
    .with_batch_size(1)
    .with_power_mode(1) // Balanced
    .build()?;

// Perform inference
let input = vec![0.5f32; 128];
let output = runtime.infer(&input)?;

// Or use in-place inference (no allocation)
let mut output = vec![0.0f32; 10];
runtime.infer_inplace(&input, &mut output)?;
```

### C FFI

```c
#include "dpb_mobile.h"

// Create runtime
DpbRuntimeConfig config;
dpb_config_default(&config);
config.max_memory_mb = 50;
config.thread_count = 2;

DpbRuntime* runtime = dpb_runtime_create_with_config(&config);

// Load model
int result = dpb_model_load(runtime, model_data, model_size);
if (result != 0) {
    const char* error = dpb_get_error(runtime);
    // Handle error
}

// Perform inference
float input[128];
float output[10];
result = dpb_infer(runtime, input, 128, output, 10);

// Cleanup
dpb_runtime_destroy(runtime);
```

### Swift (iOS)

```swift
import DpbMobile

// Create runtime
let config = DpbRuntimeConfig()
config.maxMemoryMb = 50
config.threadCount = 2

guard let runtime = DpbRuntime.create(config: config) else {
    fatalError("Failed to create runtime")
}

// Load model
guard let modelData = NSData(contentsOfFile: modelPath) else {
    fatalError("Failed to load model")
}

let result = runtime.loadModel(data: modelData)
if result != .success {
    print("Error: \(runtime.lastError ?? "Unknown error")")
}

// Perform inference
let input: [Float] = Array(repeating: 0.5, count: 128)
let output = runtime.infer(input: input)
```

### Kotlin (Android)

```kotlin
import com.dpb.mobile.DpbRuntime
import com.dpb.mobile.DpbRuntimeConfig

// Create runtime
val config = DpbRuntimeConfig().apply {
    maxMemoryMb = 50
    threadCount = 2
}

val runtime = DpbRuntime.create(config)

// Load model
val modelBytes = assets.open("model.dpb").readBytes()
val result = runtime.loadModel(modelBytes)

if (!result.success) {
    Log.e("DPB", "Failed to load model: ${runtime.lastError}")
}

// Perform inference
val input = FloatArray(128) { 0.5f }
val output = runtime.infer(input)
```

## Optimization

### Weight Pruning

Reduce model size by removing small weights:

```rust
use dpb_mobile::optimization::{WeightPruner, PruningStrategy};

let mut model = MobileModel::from_bytes(model_bytes)?;
let pruner = WeightPruner::new(PruningStrategy::Magnitude, 0.5);
let stats = pruner.prune_model(&mut model)?;

println!("Sparsity: {:.1}%", stats.sparsity() * 100.0);
println!("Compression: {:.2}x", stats.compression_ratio());
```

### Quantization

Reduce model size with quantization:

```rust
use dpb_mobile::optimization::QuantizationOptimizer;
use dpb_mobile::model::QuantizationType;

let mut model = MobileModel::from_bytes(model_bytes)?;
let optimizer = QuantizationOptimizer::new(QuantizationType::Int8);
let stats = optimizer.quantize_model(&mut model)?;

println!("Size reduction: {:.1}%",
    (1.0 - 1.0 / stats.compression_ratio()) * 100.0);
```

### Operator Fusion

Improve performance by fusing operations:

```rust
use dpb_mobile::optimization::OperatorFusion;

let mut model = MobileModel::from_bytes(model_bytes)?;
let fusion = OperatorFusion::new(true);
let stats = fusion.fuse_model(&mut model)?;

println!("Fused {} operators", stats.fused_operators);
```

## Benchmarking

Run comprehensive performance benchmarks:

```rust
use dpb_mobile::benchmark::{BenchmarkRunner, BenchmarkConfig};

let config = BenchmarkConfig {
    warmup_iterations: 10,
    benchmark_iterations: 100,
    batch_size: 1,
    collect_memory: true,
    collect_power: false,
};

let mut runner = BenchmarkRunner::new(config);

// Run inference and collect metrics
for _ in 0..100 {
    let start = Instant::now();
    runtime.infer(&input)?;
    let duration = start.elapsed();
    runner.record_latency(duration.as_secs_f32() * 1000.0);
}

let result = runner.build_result("model_name".into(), memory_metrics);
println!("Mean latency: {:.2} ms", result.latency.mean_ms);
println!("P95 latency: {:.2} ms", result.latency.p95_ms);
println!("Throughput: {:.0} inferences/sec",
    result.throughput.inferences_per_second);
```

## Platform-Specific Features

### iOS

```rust
use dpb_mobile::ios::{IosRuntime, MetalConfig, CoreMLConfig};

let mut ios_runtime = IosRuntime::new()
    .with_metal(MetalConfig {
        enabled: true,
        device_index: None,
        max_command_buffers: 3,
        use_shared_memory: true,
    })
    .with_coreml(CoreMLConfig {
        enabled: true,
        compute_units: CoreMLComputeUnits::All,
        allow_compilation: true,
    });

// Update battery status
ios_runtime.update_battery_status(0.5, false);

// Check if should throttle
if ios_runtime.should_throttle() {
    // Reduce inference frequency
}

// Handle background/foreground
ios_runtime.enter_background();
ios_runtime.enter_foreground();
```

### Android

```rust
use dpb_mobile::android::{
    AndroidRuntime, NnapiConfig, VulkanConfig,
    AndroidPowerState, AndroidThermalState
};

let mut android_runtime = AndroidRuntime::new()
    .with_nnapi(NnapiConfig {
        enabled: true,
        device_type: NnapiDeviceType::Any,
        allow_fp16: true,
        cache_dir: Some("/data/local/tmp".into()),
        compilation_preference: NnapiCompilationPreference::FastSingleAnswer,
    })
    .with_vulkan(VulkanConfig {
        enabled: true,
        device_index: None,
        enable_validation: false,
        max_descriptor_sets: 4,
        use_subgroups: true,
    });

// Update device state
android_runtime.update_battery_status(0.5, false, false);
android_runtime.update_power_state(AndroidPowerState::Active);
android_runtime.update_thermal_state(AndroidThermalState::None);

// Get recommended inference frequency
let frequency = android_runtime.recommended_inference_frequency_hz();
println!("Recommended frequency: {} Hz", frequency);

// Check if should block inference
if android_runtime.should_block_inference() {
    // Stop inference completely
}
```

## Testing

Run tests on host platform:

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_inference

# Run benchmarks
cargo bench
```

## Examples

See the `examples/` directory for complete examples:

```bash
# Run basic usage example
cargo run --example basic_usage --release
```

## Performance Tips

1. **Use release builds** - Debug builds are significantly slower
2. **Enable operator fusion** - Reduces overhead from multiple operations
3. **Use memory planning** - Reduces allocations during inference
4. **Enable SIMD** - Leverages ARM NEON instructions
5. **Use quantization** - Int8 models are 4x smaller with minimal accuracy loss
6. **Batch size = 1** - Mobile inference typically processes single samples
7. **Monitor battery** - Throttle inference when battery is low
8. **Respect thermal state** - Reduce frequency during thermal throttling

## Integration with iOS/Android

### iOS Framework

Create an iOS framework:

1. Build static library for all architectures
2. Create module map and headers
3. Package as XCFramework

```bash
# See tools/build_ios_framework.sh for complete script
```

### Android AAR

Create an Android AAR:

1. Build for all Android architectures
2. Create JNI bindings
3. Package as AAR

```bash
# See tools/build_android_aar.sh for complete script
```

## License

This project is licensed under MIT OR Apache-2.0.

## Contributing

Contributions are welcome! Please see the main DPB framework repository for contribution guidelines.

## Support

For issues and questions:
- GitHub Issues: https://github.com/AuraSenseTech/dpb-framework/issues
- Documentation: https://docs.dpb-framework.dev

## Roadmap

- [ ] Metal GPU backend (iOS)
- [ ] CoreML integration (iOS)
- [ ] NNAPI backend (Android)
- [ ] Vulkan compute backend (Android)
- [ ] Dynamic quantization
- [ ] Model profiling tools
- [ ] Automatic batch size tuning
- [ ] Model compression improvements
