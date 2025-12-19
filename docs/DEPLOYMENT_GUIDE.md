# Delta-Predictive Biosensing (DPB) Deployment Guide

> **Version:** 2.0.0 | **Last Updated:** December 2025
> **Framework Version:** 0.5.2 | **Catalog Version:** 5.3.0
> **Purpose:** Platform-specific deployment instructions and guided selection for all supported targets

---

## Table of Contents

1. [Guided Platform Selection](#1-guided-platform-selection)
2. [Deployment Matrix Overview](#2-deployment-matrix-overview)
3. [Operating Systems](#3-operating-systems)
4. [GPU Platforms](#4-gpu-platforms)
5. [Browser Deployment](#5-browser-deployment)
6. [Mobile Platforms](#6-mobile-platforms)
7. [Processor Architectures](#7-processor-architectures)
8. [Embedded / Edge AI / TinyML](#8-embedded--edge-ai--tinyml)
9. [AI Accelerators](#9-ai-accelerators)
10. [Neuromorphic Hardware](#10-neuromorphic-hardware)
11. [FPGA Deployment](#11-fpga-deployment)
12. [Language Bindings](#12-language-bindings)
13. [Streaming & Real-time](#13-streaming--real-time)
14. [Cloud & Container Deployment](#14-cloud--container-deployment)
15. [Clinical / Regulatory](#15-clinical--regulatory)
16. [Quick Start by Use Case](#16-quick-start-by-use-case)

---

## 1. Guided Platform Selection

### 1.1 Decision Flowchart

Use this flowchart to determine which deployment target is right for your use case:

```
START: What is your deployment environment?
│
├─► Desktop/Server Application
│   ├─► Windows? ───────────────► Section 3.3 (Windows)
│   ├─► macOS? ─────────────────► Section 3.2 (macOS)
│   └─► Linux? ─────────────────► Section 3.1 (Linux)
│
├─► GPU Acceleration Required?
│   ├─► NVIDIA GPU? ────────────► Section 4.2 (CUDA) or 4.4 (Vulkan)
│   ├─► AMD GPU? ───────────────► Section 4.5 (Vulkan/ROCm)
│   ├─► Apple GPU? ─────────────► Section 4.3 (Metal)
│   └─► Intel GPU? ─────────────► Section 4.4 (Vulkan)
│
├─► Browser/Web Application?
│   ├─► Modern browsers (Chrome, Edge, Firefox)? ──► Section 5.3 (WebGPU)
│   ├─► Safari? ────────────────► Section 5.6 (Safari WebGPU)
│   ├─► ML inference in browser? ► Section 5.5 (WebNN)
│   └─► Legacy browser support? ─► Section 5.4 (CPU/WebGL fallback)
│
├─► Mobile Application?
│   ├─► iOS? ───────────────────► Section 6.1 (iOS/Metal/CoreML)
│   ├─► Android? ───────────────► Section 6.2 (Android/Vulkan/NNAPI)
│   └─► Cross-platform? ────────► Both sections + export to TFLite
│
├─► Embedded / Edge Device?
│   ├─► ESP32 (WiFi/BLE)? ──────► Section 8.3 (ESP32 RISC-V)
│   ├─► STM32 / ARM Cortex-M? ──► Section 8.4 (ARM Cortex-M)
│   ├─► Raspberry Pi? ──────────► Section 7.3 (ARM64 Linux)
│   ├─► NVIDIA Jetson? ─────────► Section 7.3 (ARM64) + Section 4.2 (CUDA)
│   └─► Custom FPGA? ───────────► Section 11 (FPGA Deployment)
│
├─► Mobile NPU Acceleration?
│   ├─► Qualcomm device? ───────► Section 6.3 (Hexagon DSP)
│   ├─► ARM device (non-Apple)? ► Section 6.4 (Ethos-U NPU)
│   └─► Apple device? ──────────► Section 6.5 (Apple Neural Engine)
│
├─► AI Accelerator Hardware?
│   ├─► Intel Gaudi? ───────────► Section 9.1 (Intel Gaudi)
│   ├─► Graphcore IPU? ─────────► Section 9.2 (Graphcore IPU)
│   └─► NVIDIA Data Center? ────► Section 4.2 (CUDA)
│
├─► Neuromorphic Hardware?
│   ├─► Intel Loihi 2? ─────────► Section 10.1 (Lava export)
│   ├─► SpiNNaker 2? ───────────► Section 10.2 (PyNN export)
│   └─► BrainScaleS-2? ─────────► Section 10.3 (hxtorch export)
│
└─► Language Binding Needed?
    ├─► Python? ────────────────► Section 12.2 (PyO3)
    ├─► JavaScript/TypeScript? ─► Section 12.4 (WASM)
    ├─► C/C++? ─────────────────► Section 12.3 (FFI)
    ├─► R? ─────────────────────► Section 12.5 (R bindings)
    ├─► Julia? ─────────────────► Section 12.6 (Julia bindings)
    ├─► MATLAB? ────────────────► Section 12.7 (MEX)
    └─► LabVIEW? ───────────────► Section 12.8 (CLFN)
```

### 1.2 Quick Selection Table

| Your Need | Recommended Path | Key Command |
|-----------|------------------|-------------|
| **Fastest GPU encoding** | Native + Vulkan/Metal | `cargo build --release --features webgpu` |
| **Cross-platform web app** | WASM + WebGPU | `wasm-pack build --features webgpu` |
| **iOS app with ML** | iOS + CoreML + ANE | `cargo build --target aarch64-apple-ios --features coreml` |
| **Android app with ML** | Android + NNAPI | `cargo ndk --target aarch64-linux-android --features nnapi` |
| **Battery-powered wearable** | ARM Cortex-M | `cargo build --target thumbv7em-none-eabihf --features no_std` |
| **Research with Python** | Python bindings | `pip install dpb` or `maturin develop` |
| **Real-time BCI** | Native + LSL | `cargo build --features native,async` |
| **FPGA deployment** | HLS export | `dpb export --format fpga-hls --target xilinx` |
| **Neuromorphic chip** | Lava/PyNN export | `dpb export --format neuromorphic --target loihi2` |

### 1.3 Performance vs Portability Trade-offs

```
                    Performance
                        ▲
                        │
    CUDA/Metal ●────────┼────────────────────────●  Native x86/ARM
    (Fastest)           │                             (Good)
                        │
                        │     WebGPU ●
                        │       (Good on modern browsers)
                        │
         FPGA HLS ●─────┼───────────────● NPU (Hexagon/Ethos-U/ANE)
         (Custom)       │                 (Mobile optimized)
                        │
                        │         WebNN ●
                        │         (Browser ML)
                        │
         Neuromorphic ●─┼───────────────────● WASM CPU
         (Ultra-low     │                     (Universal)
          power)        │
                        │
                        └────────────────────────────────────► Portability
```

### 1.4 Regime Coverage Summary

| Regime | Sub-categories | DPB Support Level |
|--------|----------------|:-----------------:|
| **Operating Systems** | Windows, Linux, macOS | ✅ Full |
| **GPU Platforms** | CUDA, Metal, Vulkan, DX12, AMD ROCm | ✅ Full |
| **Browser** | WebGPU, WebNN, CPU fallback, WebGL (deprecated) | ✅ Full |
| **Mobile** | iOS, Android | ✅ Full |
| **Mobile NPU** | Hexagon DSP, ARM Ethos-U, Apple ANE | ✅ Full |
| **Processor Type** | ARM, x86 | ✅ Full |
| **Architecture** | x64, x86, ARM64, ARMv7, RISC-V | ✅ Full |
| **Embedded/TinyML** | ESP32, STM32, nRF52, RP2040 | ✅ Full |
| **AI Accelerators** | Intel Gaudi, Graphcore IPU | ⚠️ Simulation |
| **Neuromorphic** | Loihi 2, SpiNNaker 2, BrainScaleS-2 | ✅ Export |
| **FPGA** | Xilinx Vitis, Intel HLS | ✅ Export |

---

## 2. Deployment Matrix Overview

### 2.1 Platform Support Matrix

| Platform | Status | GPU Accel | NPU | Crate | Notes |
|----------|:------:|:---------:|:---:|-------|-------|
| **Linux x86_64** | ✅ Production | Vulkan, CUDA | - | dpb-core | Primary development target |
| **Linux ARM64** | ✅ Production | Vulkan | - | dpb-core | Raspberry Pi 4/5, Jetson |
| **macOS x86_64** | ✅ Production | Metal | - | dpb-core | Intel Macs |
| **macOS ARM64** | ✅ Production | Metal | ANE | dpb-mobile | Apple Silicon (M1-M4) |
| **Windows x86_64** | ✅ Production | DX12, Vulkan | - | dpb-core | Windows 10/11 |
| **iOS ARM64** | ✅ Production | Metal | ANE | dpb-mobile | iPhone 8+, iPad Pro |
| **Android ARM64** | ✅ Production | Vulkan | NNAPI, Hexagon | dpb-mobile | Android 8.0+ |
| **Android ARMv7** | ✅ Production | CPU only | - | dpb-mobile | Legacy devices |
| **WebAssembly** | ✅ Production | WebGPU | WebNN | dpb-wasm | Modern browsers |
| **RISC-V 32-bit** | ✅ Production | CPU only | - | dpb-core | ESP32-C3, GD32VF103 |
| **RISC-V 64-bit** | ✅ Production | CPU only | - | dpb-core | SiFive, VisionFive 2 |
| **ARM Cortex-M** | ✅ Production | CPU only | Ethos-U | dpb-mobile | STM32, nRF52, RP2040 |
| **Intel Gaudi** | ⚠️ Simulation | TPC Cores | - | dpb-core | Requires Synapse SDK |
| **Graphcore IPU** | ⚠️ Simulation | 1472 Tiles | - | dpb-core | Requires Poplar SDK |
| **Loihi 2** | ✅ Export | Neuromorphic | - | dpb-export | Via Lava framework |
| **SpiNNaker 2** | ✅ Export | Neuromorphic | - | dpb-export | Via PyNN |
| **BrainScaleS-2** | ✅ Export | Neuromorphic | - | dpb-export | Via hxtorch |
| **Xilinx FPGA** | ✅ Export | HLS | - | dpb-export | Vitis HLS C++ |
| **Intel FPGA** | ✅ Export | HLS | - | dpb-export | Intel HLS C++ |

### 2.2 Feature Flag Summary

```toml
# Cargo.toml feature flags for deployment targets

[features]
# GPU Acceleration
webgpu = ["wgpu", "web-sys"]      # Cross-platform GPU via wgpu
metal = ["metal-rs"]              # iOS/macOS Metal
vulkan = ["ash"]                  # Android/Linux Vulkan
cuda = ["gpu"]                    # NVIDIA CUDA support

# Mobile NPU
coreml = ["coreml-sys"]           # iOS CoreML inference
nnapi = ["nnapi-sys"]             # Android Neural Networks API
hexagon = []                      # Qualcomm Hexagon DSP
ethos-u = []                      # ARM Ethos-U NPU
apple-ane = []                    # Apple Neural Engine

# Browser ML
webnn = []                        # WebNN browser ML API

# Hardware Accelerators
intel-gaudi = []                  # Intel Gaudi AI accelerators
graphcore-ipu = []                # Graphcore IPU
hardware-accelerators = ["intel-gaudi", "graphcore-ipu"]

# Export Targets
fpga = []                         # FPGA HLS export
neuromorphic = []                 # Neuromorphic export (Lava/PyNN/hxtorch)

# Embedded
no_std = []                       # For microcontrollers
riscv-hal = []                    # RISC-V specific

# Streaming
native = ["liblsl-sys"]           # Lab Streaming Layer
async = ["tokio"]                 # Async runtime
```

---

## 3. Operating Systems

### 3.1 Linux

#### Supported Distributions
- Ubuntu 20.04+ (primary CI target)
- Debian 11+
- Fedora 38+
- RHEL/CentOS 8+
- Arch Linux (rolling)

#### Installation

```bash
# Prerequisites
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev

# For GPU support (Vulkan)
sudo apt-get install -y libvulkan-dev vulkan-tools

# For CUDA (NVIDIA)
# Install NVIDIA drivers first, then:
sudo apt-get install -y nvidia-cuda-toolkit

# For Lab Streaming Layer
sudo apt-get install -y liblsl-dev

# Build from source
cargo build --release

# With all features
cargo build --release --features "webgpu,native,async,cuda"
```

#### Deployment Options

| Use Case | Command | Output |
|----------|---------|--------|
| CLI Tool | `cargo build --release` | `target/release/dpb` |
| Shared Library | `cargo build --release -p dpb-ffi` | `libdpb.so` |
| Python Wheel | `maturin build --release` | `dpb-*.whl` |
| Static Library | `cargo build --release -p dpb-ffi --crate-type=staticlib` | `libdpb.a` |

### 3.2 macOS

#### Supported Versions
- macOS 12 Monterey+ (Intel)
- macOS 12 Monterey+ (Apple Silicon M1-M4)

#### Installation

```bash
# Prerequisites (via Homebrew)
brew install pkg-config openssl

# For Lab Streaming Layer
brew install labstreaminglayer/tap/lsl

# Build (Metal GPU automatic)
cargo build --release

# Universal binary (Intel + ARM)
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
lipo -create -output target/universal/libdpb.dylib \
    target/x86_64-apple-darwin/release/libdpb.dylib \
    target/aarch64-apple-darwin/release/libdpb.dylib
```

#### Apple Silicon Optimization

```bash
# M1/M2/M3/M4 with Metal and Apple Neural Engine
cargo build --release --features "webgpu,apple-ane"

# Check Metal availability
system_profiler SPDisplaysDataType | grep Metal
```

### 3.3 Windows

#### Supported Versions
- Windows 10 (1903+)
- Windows 11

#### Installation

```powershell
# Prerequisites (via winget)
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools

# Or via Chocolatey
choco install rust visualstudio2022-workload-vctools

# Build (DX12 automatic)
cargo build --release

# With Vulkan (for NVIDIA/AMD)
cargo build --release --features "webgpu"
```

#### Deployment Paths

| Use Case | Output | Notes |
|----------|--------|-------|
| CLI Tool | `dpb.exe` | Standalone executable |
| DLL | `dpb.dll` | For C/C++ integration |
| Python | `dpb-*.whl` | Via `maturin build` |

---

## 4. GPU Platforms

### 4.1 GPU Support Matrix

| GPU Platform | OS Support | Backend | Status | Performance | Best For |
|--------------|------------|---------|:------:|:-----------:|----------|
| **NVIDIA CUDA** | Linux, Windows | dpb-core/cuda | ✅ Direct | Excellent | Data center, research |
| **NVIDIA (Vulkan)** | Linux, Windows | wgpu | ✅ Native | Excellent | Cross-platform apps |
| **AMD (Vulkan)** | Linux, Windows | wgpu | ✅ Native | Excellent | Gaming, workstation |
| **AMD (ROCm)** | Linux | wgpu | ✅ Native | Excellent | HPC clusters |
| **Intel (Vulkan)** | Linux, Windows | wgpu | ✅ Native | Good | Integrated graphics |
| **Apple Metal** | macOS, iOS | wgpu | ✅ Native | Excellent | Apple ecosystem |
| **DirectX 12** | Windows | wgpu | ✅ Native | Excellent | Windows apps |
| **WebGPU** | Browser | wgpu | ✅ Native | Good | Web applications |

### 4.2 CUDA Integration

DPB provides direct CUDA support via `dpb-core/accelerators/cuda.rs`:

```bash
# Build with CUDA support
cargo build --release --features "cuda"

# Verify CUDA
nvidia-smi
nvcc --version
```

```rust
use dpb_core::accelerators::cuda::{CudaAccelerator, CudaConfig, CudaBuffer};

// Configure CUDA
let config = CudaConfig {
    device_id: 0,
    compute_capability: (8, 6),  // Ampere
    shared_memory_size: 48 * 1024,
    max_threads_per_block: 1024,
    stream_count: 4,
    ..Default::default()
};

let accel = CudaAccelerator::new(config)?;

// Check device info
let info = accel.device_info();
println!("GPU: {} ({} GB)", info.name, info.memory_gb);

// High-performance encoding
let result = accel.encode_level_crossing_async(&signal, threshold).await?;
```

#### CUDA Performance Tuning

```rust
// Optimal launch configuration
let launch_config = accel.optimal_launch_config(signal.len())?;
println!("Grid: {:?}, Block: {:?}", launch_config.grid_dim, launch_config.block_dim);

// Multi-GPU support
let devices = CudaAccelerator::available_devices();
for (id, name) in devices {
    println!("GPU {}: {}", id, name);
}
```

### 4.3 Metal (Apple)

```bash
# Automatic on macOS/iOS via wgpu
cargo build --release --features "webgpu"

# iOS-specific with CoreML inference
cargo build --release --target aarch64-apple-ios --features "metal,coreml"
```

```rust
use dpb_mobile::ios::{MetalEncoder, MetalConfig};

let config = MetalConfig {
    use_shared_memory: true,
    prefer_low_power: false,  // Use high-performance GPU
};

let encoder = MetalEncoder::new(config)?;
let spikes = encoder.encode(&signal)?;
```

### 4.4 Vulkan

```bash
# Linux
sudo apt-get install libvulkan-dev
cargo build --release --features "webgpu"

# Verify Vulkan support
vulkaninfo | grep "GPU"

# Android (automatic with NDK)
cargo ndk --target aarch64-linux-android build --features "vulkan"
```

### 4.5 AMD ROCm

```bash
# Install ROCm
sudo apt-get install rocm-dev

# Build with Vulkan backend (ROCm provides Vulkan ICD)
cargo build --release --features "webgpu"

# For direct HIP support (future)
# Currently use Vulkan, which has excellent AMD support
```

---

## 5. Browser Deployment

### 5.1 Browser Support Matrix

| Browser | WebGPU | WebNN | WASM | Safari Support |
|---------|:------:|:-----:|:----:|:--------------:|
| **Chrome 113+** | ✅ | ✅ Flag | ✅ | - |
| **Edge 113+** | ✅ | ✅ Flag | ✅ | - |
| **Firefox 121+** | ✅ | ❌ | ✅ | - |
| **Safari 17+** | ✅ | ❌ | ✅ | ✅ Special handling |
| **Chrome Android** | ✅ | ❌ | ✅ | - |
| **Safari iOS 17+** | ✅ | ❌ | ✅ | ✅ Special handling |

### 5.2 WebAssembly Build

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for web
cd crates/dpb-wasm
wasm-pack build --target web --release

# Output: pkg/dpb_wasm.js, pkg/dpb_wasm_bg.wasm
```

### 5.3 WebGPU (GPU Acceleration)

```bash
# Build with WebGPU feature
wasm-pack build --target web --release --features "webgpu"
```

```javascript
// JavaScript usage
import init, { GpuEncoder, WasmTimeSeries } from './pkg/dpb_wasm.js';

await init();

// Check WebGPU availability
if (navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const encoder = await GpuEncoder.new();
    const result = await encoder.encode_level_crossing(timeseries, threshold);
}
```

### 5.4 CPU Fallback (WebGL Deprecated)

For non-WebGPU browsers, use CPU fallback:

```javascript
import { WasmLevelCrossingEncoder } from './pkg/dpb_wasm.js';

// CPU fallback (always available)
const encoder = new WasmLevelCrossingEncoder(threshold);
const spikes = encoder.encode(samples);
```

> **Note:** WebGL compute is deprecated. See `docs/WEBGL_DEPRECATION.md` for migration guidance.

### 5.5 WebNN (Browser ML Inference)

WebNN provides hardware-accelerated ML inference in browsers:

```bash
# Build with WebNN support
wasm-pack build --target web --release --features "webnn"
```

```javascript
import { WebNNEncoder, WebNNConfig } from './pkg/dpb_wasm.js';

// Check WebNN availability
if ('ml' in navigator) {
    const config = new WebNNConfig();
    config.device_preference = 'gpu';  // or 'cpu', 'npu'
    config.power_preference = 'default';  // or 'low-power', 'high-performance'

    const encoder = await WebNNEncoder.new(config);

    // Build computation graph
    const builder = encoder.create_graph_builder();
    builder.add_level_crossing_layer(threshold);
    const graph = await builder.build();

    // Run inference
    const result = await graph.compute(inputTensor);
}
```

#### WebNN Browser Compatibility

| Browser | WebNN Status | Enable Flag |
|---------|:------------:|-------------|
| Chrome 113+ | ✅ Behind flag | `chrome://flags/#enable-experimental-web-platform-features` |
| Edge 113+ | ✅ Behind flag | `edge://flags/#enable-experimental-web-platform-features` |
| Firefox | ❌ Not supported | - |
| Safari | ❌ Not supported | - |

### 5.6 Safari WebGPU Support

Safari has WebGPU quirks requiring special handling. DPB includes Safari-specific compatibility:

```bash
# Build includes Safari support automatically
wasm-pack build --target web --release --features "webgpu"
```

```javascript
import { SafariFeatureDetector, create_safari_compatible_encoder } from './pkg/dpb_wasm.js';

// Detect Safari and its capabilities
const detector = new SafariFeatureDetector();
const features = await detector.detect();

if (features.is_safari) {
    console.log(`Safari ${features.safari_version} detected`);
    console.log(`WebGPU: ${features.webgpu_supported}`);

    // Use Safari-optimized encoder
    const encoder = await create_safari_compatible_encoder();

    // Safari-specific performance hints
    encoder.set_workgroup_size(64);  // Safari prefers smaller workgroups
    encoder.enable_conservative_memory();

    const result = await encoder.encode(signal);
} else {
    // Standard WebGPU path
    const encoder = await GpuEncoder.new();
    const result = await encoder.encode(signal);
}
```

#### Safari-Specific Considerations

| Issue | Solution | DPB Handling |
|-------|----------|--------------|
| Workgroup size limits | Use 64 instead of 256 | `SafariWebGPUConfig` |
| Shader compilation | Simpler WGSL variants | `LEVEL_CROSSING_SAFARI` shader |
| Memory pressure | Conservative allocation | `enable_conservative_memory()` |
| Error messages | Enhanced error mapping | `SafariErrorHandler` |

---

## 6. Mobile Platforms

### 6.1 iOS

#### Supported Devices
- iPhone 8+ (A11 Bionic or later)
- iPad Pro (2018+), iPad Air (2020+)
- iOS 14.0+

#### Build Setup

```bash
# Install iOS targets
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-sim
rustup target add x86_64-apple-ios  # Intel simulator

# Build static library
cargo build --release --target aarch64-apple-ios -p dpb-mobile

# With Metal GPU
cargo build --release --target aarch64-apple-ios -p dpb-mobile --features "metal"

# With CoreML + Apple Neural Engine
cargo build --release --target aarch64-apple-ios -p dpb-mobile --features "coreml,apple-ane"

# Output: target/aarch64-apple-ios/release/libdpb_mobile.a
```

#### Xcode Integration

```swift
// Swift bridging header
#import "dpb_mobile.h"

// Swift usage
import DPBMobile

let encoder = DPBLevelCrossingEncoder(threshold: 0.1)
let spikes = encoder.encode(samples: signalData)

// With CoreML acceleration
let coremlEncoder = DPBCoreMLEncoder()
coremlEncoder.loadModel(named: "spike_encoder")
let spikes = coremlEncoder.encode(samples: signalData)
```

### 6.2 Android

#### Supported Devices
- ARM64: Android 8.0+ (API 26+)
- ARMv7: Android 5.0+ (API 21+)

#### Build Setup

```bash
# Install Android targets
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add x86_64-linux-android

# Install cargo-ndk
cargo install cargo-ndk

# Set NDK path
export ANDROID_NDK_HOME=/path/to/android-ndk

# Build
cargo ndk --target aarch64-linux-android --platform 26 \
    build --release -p dpb-mobile

# With Vulkan GPU
cargo ndk --target aarch64-linux-android --platform 26 \
    build --release -p dpb-mobile --features "vulkan"

# With NNAPI
cargo ndk --target aarch64-linux-android --platform 26 \
    build --release -p dpb-mobile --features "nnapi"

# Output: target/aarch64-linux-android/release/libdpb_mobile.so
```

#### Android Studio Integration

```kotlin
// JNI loading
class DPBNative {
    companion object {
        init {
            System.loadLibrary("dpb_mobile")
        }
    }

    external fun encodeLevelCrossing(
        samples: FloatArray,
        threshold: Float
    ): LongArray

    external fun isNNAPIAvailable(): Boolean
    external fun encodeWithNNAPI(samples: FloatArray): LongArray
}
```

### 6.3 Hexagon DSP (Qualcomm)

For Qualcomm Snapdragon devices with Hexagon DSP:

```bash
# Build with Hexagon support
cargo ndk --target aarch64-linux-android --platform 26 \
    build --release -p dpb-mobile --features "hexagon"
```

```rust
use dpb_mobile::npu::{HexagonConfig, NpuEncoder, NpuBackend};

let config = HexagonConfig {
    backend: NpuBackend::Hexagon,
    precision: Precision::Int8,  // Quantized for DSP
    use_hvx: true,               // Hexagon Vector eXtensions
    power_mode: PowerMode::Balanced,
};

let encoder = NpuEncoder::new(config)?;

// Check Hexagon availability
if encoder.is_backend_available(NpuBackend::Hexagon) {
    let result = encoder.encode_quantized(&signal)?;
}
```

### 6.4 ARM Ethos-U NPU

For ARM Cortex-M devices with Ethos-U NPU:

```bash
# Build for Cortex-M with Ethos-U
cargo build --release --target thumbv7em-none-eabihf \
    -p dpb-mobile --features "ethos-u"
```

```rust
use dpb_mobile::npu::{EthosUConfig, NpuEncoder, NpuBackend};

let config = EthosUConfig {
    backend: NpuBackend::EthosU,
    model_format: ModelFormat::Vela,  // ARM Vela compiled
    mac_count: 256,
    sram_size: 512 * 1024,
};

let encoder = NpuEncoder::new(config)?;
```

### 6.5 Apple Neural Engine (ANE)

For Apple devices with Neural Engine:

```bash
# Build with ANE support
cargo build --release --target aarch64-apple-ios \
    -p dpb-mobile --features "apple-ane,coreml"
```

```rust
use dpb_mobile::npu::{AppleANEConfig, NpuEncoder, NpuBackend};

let config = AppleANEConfig {
    backend: NpuBackend::AppleANE,
    compute_units: ComputeUnits::All,  // CPU + GPU + ANE
    model_format: ModelFormat::CoreML,
};

let encoder = NpuEncoder::new(config)?;

// ANE is preferred automatically for supported operations
let result = encoder.encode(&signal)?;
```

---

## 7. Processor Architectures

### 7.1 Architecture Support Matrix

| Architecture | Target Triple | SIMD | FPU | Status |
|--------------|---------------|------|-----|:------:|
| **x86_64** | `x86_64-unknown-linux-gnu` | AVX2, AVX-512 | Yes | ✅ |
| **x86** | `i686-unknown-linux-gnu` | SSE2, SSE4 | Yes | ✅ |
| **ARM64** | `aarch64-unknown-linux-gnu` | NEON | Yes | ✅ |
| **ARMv7** | `armv7-unknown-linux-gnueabihf` | NEON | Yes | ✅ |
| **ARMv6** | `arm-unknown-linux-gnueabihf` | None | Optional | ⚠️ |
| **RISC-V 64** | `riscv64gc-unknown-linux-gnu` | V-ext* | Yes | ✅ |
| **RISC-V 32** | `riscv32imc-unknown-none-elf` | None | No | ✅ |

### 7.2 x86_64 (Intel/AMD)

```bash
# Standard build (auto-detects AVX2)
cargo build --release

# Force AVX-512 (if available)
RUSTFLAGS="-C target-feature=+avx512f" cargo build --release

# Check CPU features at runtime
./target/release/dpb --show-simd
```

### 7.3 ARM64 (Apple Silicon, Raspberry Pi, Jetson)

```bash
# Native build on ARM64
cargo build --release

# Cross-compile from x86_64
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu

# Raspberry Pi 4/5 (64-bit Raspberry Pi OS)
cargo build --release --target aarch64-unknown-linux-gnu

# NVIDIA Jetson (with CUDA)
cargo build --release --target aarch64-unknown-linux-gnu --features "cuda"
```

### 7.4 ARMv7 (32-bit ARM)

```bash
# Cross-compile
rustup target add armv7-unknown-linux-gnueabihf
cargo build --release --target armv7-unknown-linux-gnueabihf

# For Android (older devices)
cargo ndk --target armv7-linux-androideabi build --release
```

---

## 8. Embedded / Edge AI / TinyML

### 8.1 Embedded Platform Matrix

| Platform | MCU | RAM | Flash | Target | Features |
|----------|-----|-----|-------|--------|----------|
| **ESP32-C3** | RISC-V | 400KB | 4MB | `riscv32imc` | WiFi, BLE |
| **ESP32-S3** | Xtensa | 512KB | 8MB | `xtensa-esp32s3` | WiFi, AI accel |
| **STM32F4** | Cortex-M4 | 192KB | 1MB | `thumbv7em` | DSP, FPU |
| **STM32H7** | Cortex-M7 | 1MB | 2MB | `thumbv7em` | DSP, FPU |
| **nRF52840** | Cortex-M4 | 256KB | 1MB | `thumbv7em` | BLE 5.0 |
| **RP2040** | Cortex-M0+ | 264KB | 2MB | `thumbv6m` | Dual-core |
| **GD32VF103** | RISC-V | 32KB | 128KB | `riscv32imac` | Low power |
| **SiFive U74** | RISC-V 64 | 8GB | - | `riscv64gc` | Linux capable |

### 8.2 Memory-Constrained Builds

```toml
# .cargo/config.toml - embedded profile
[profile.embedded]
inherits = "release"
opt-level = "z"          # Optimize for size
lto = true               # Link-time optimization
codegen-units = 1        # Better optimization
panic = "abort"          # No unwinding
strip = true             # Strip symbols
```

```bash
# Build with size optimization
cargo build --profile embedded --target thumbv7em-none-eabihf
```

### 8.3 ESP32 (RISC-V)

```bash
# Install ESP toolchain
cargo install espup
espup install

# Build for ESP32-C3 (RISC-V)
cargo build --release --target riscv32imc-unknown-none-elf \
    --no-default-features --features "no_std,riscv-hal"

# Flash
cargo espflash flash --release
```

### 8.4 ARM Cortex-M (STM32, nRF52)

```bash
# Install probe-run for debugging
cargo install probe-run

# Build for STM32F4 (Cortex-M4)
cargo build --release --target thumbv7em-none-eabihf \
    --no-default-features --features "no_std"

# Flash and run
cargo run --release --target thumbv7em-none-eabihf
```

### 8.5 no_std Support

```rust
// lib.rs
#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;

use dpb_core::signal::TimeSeries;
use dpb_encoders::level_crossing::LevelCrossingEncoder;

// Works on microcontrollers
let encoder = LevelCrossingEncoder::new(0.1);
let spikes = encoder.encode(&signal);
```

### 8.6 Fixed-Point Arithmetic (FPU-less targets)

For targets without FPU:

```rust
use dpb_core::accelerators::riscv::{Q16, FixedPoint, encode_level_crossing_fixed};

// Convert float to Q16.16 fixed-point
let threshold = Q16::from_float(0.1);
let samples: Vec<Q16> = raw_samples.iter()
    .map(|&x| Q16::from_float(x))
    .collect();

// Encode with fixed-point arithmetic (no FPU required)
let spikes = encode_level_crossing_fixed(&samples, threshold);
```

---

## 9. AI Accelerators

### 9.1 Intel Gaudi (Habana Labs)

#### Hardware Requirements
- Intel Gaudi 2 or Gaudi 3
- Synapse AI SDK installed
- Linux (Ubuntu 20.04/22.04)

#### Build

```bash
# Enable Gaudi feature
cargo build --release --features "intel-gaudi"

# With Synapse SDK (production)
export HABANA_LOGS=/var/log/habana
export SYNAPSE_ROOT=/opt/habanalabs
cargo build --release --features "intel-gaudi"
```

#### Usage

```rust
use dpb_core::accelerators::gaudi::{GaudiAccelerator, GaudiConfig, GaudiDevice};

let config = GaudiConfig {
    device_id: 0,
    device_type: GaudiDevice::Gaudi2,
    memory_limit: 32 * 1024 * 1024 * 1024,  // 32GB HBM
    tpc_count: 24,
    mme_count: 2,
    ..Default::default()
};

let accel = GaudiAccelerator::new(config)?;

// Check availability
if accel.is_available() {
    let result = accel.encode_signal(&signal)?;
    println!("TPC TFLOPS: {}", accel.estimate_tflops());
}
```

**Current Status:** Simulation mode. Production requires Synapse AI SDK installation.

### 9.2 Graphcore IPU

#### Hardware Requirements
- Graphcore Bow IPU or C600
- Poplar SDK installed
- Linux (Ubuntu 20.04/22.04)

#### Build

```bash
# Enable IPU feature
cargo build --release --features "graphcore-ipu"

# With Poplar SDK (production)
export POPLAR_SDK=/opt/poplar
source $POPLAR_SDK/enable
cargo build --release --features "graphcore-ipu"
```

#### Usage

```rust
use dpb_core::accelerators::ipu::{IpuAccelerator, IpuConfig, IpuDevice};

let config = IpuConfig {
    device_type: IpuDevice::Bow,
    num_ipus: 1,
    tiles_per_ipu: 1472,
    memory_per_tile: 624 * 1024,
    ..Default::default()
};

let accel = IpuAccelerator::new(config)?;

// Create computation graph
let graph = accel.create_graph()?;
graph.add_vertex("level_crossing", &params)?;
let program = graph.compile()?;

let result = program.execute(&signal)?;
```

**Current Status:** Simulation mode. Production requires Poplar SDK installation.

---

## 10. Neuromorphic Hardware

### 10.1 Intel Loihi 2 (via Lava)

Export SNN models to Intel's Lava framework for Loihi 2:

```bash
# Build with neuromorphic export
cargo build --release -p dpb-export --features "neuromorphic"
```

```rust
use dpb_export::neuromorphic::{LavaExporter, NeuromorphicConfig, LoihiTarget};

let config = NeuromorphicConfig {
    target: LoihiTarget::Loihi2,
    neuron_model: NeuronModel::LIF,
    weight_bits: 8,
    time_resolution_us: 1000,
};

let exporter = LavaExporter::new(config);

// Export trained network
let lava_code = exporter.export(&trained_snn)?;
lava_code.save("biosignal_encoder.py")?;
```

Generated Python (Lava):
```python
# biosignal_encoder.py - Generated by DPB
import lava.lib.dl.slayer as slayer
from lava.proc.lif.process import LIF

class BiosignalEncoder(slayer.Network):
    def __init__(self):
        super().__init__()
        self.input = LIF(shape=(64,),
                        current_decay=0.25,
                        voltage_decay=0.03,
                        threshold=1.0)
        # ... network definition
```

### 10.2 SpiNNaker 2 (via PyNN)

```rust
use dpb_export::neuromorphic::{PyNNExporter, SpinnakerTarget};

let config = NeuromorphicConfig {
    target: SpinnakerTarget::SpiNNaker2,
    neuron_model: NeuronModel::IF_curr_exp,
    timestep_ms: 1.0,
};

let exporter = PyNNExporter::new(config);
let pynn_code = exporter.export(&trained_snn)?;
```

### 10.3 BrainScaleS-2 (via hxtorch)

```rust
use dpb_export::neuromorphic::{HxtorchExporter, BrainScaleSTarget};

let config = NeuromorphicConfig {
    target: BrainScaleSTarget::BrainScaleS2,
    time_compression: 1000,  // 1000x acceleration
    calibration_mode: CalibrationMode::Default,
};

let exporter = HxtorchExporter::new(config);
let hxtorch_code = exporter.export(&trained_snn)?;
```

---

## 11. FPGA Deployment

### 11.1 FPGA Export Overview

DPB can export encoders to synthesizable HLS C++ for FPGA deployment:

| Target | Toolchain | Status |
|--------|-----------|:------:|
| Xilinx Zynq/UltraScale+ | Vitis HLS | ✅ Export |
| Intel Arria/Stratix | Intel HLS Compiler | ✅ Export |
| Generic HLS | Standard C++ | ✅ Export |

### 11.2 Xilinx Vitis HLS Export

```bash
# Build with FPGA export
cargo build --release -p dpb-export --features "fpga"
```

```rust
use dpb_export::fpga::{FpgaExporter, FpgaConfig, XilinxTarget};

let config = FpgaConfig {
    target: XilinxTarget::ZynqUltrascale,
    clock_period_ns: 10.0,  // 100 MHz
    part_number: "xczu7ev-ffvc1156-2-i",
    fixed_point_bits: 16,
    interface: InterfaceType::AXIStream,
};

let exporter = FpgaExporter::new(config);

// Export level crossing encoder
let hls_code = exporter.export_level_crossing(threshold)?;
hls_code.save("level_crossing_hls/")?;
```

Generated files:
```
level_crossing_hls/
├── level_crossing.cpp      # HLS C++ implementation
├── level_crossing.h        # Header file
├── level_crossing_tb.cpp   # Testbench
├── directives.tcl          # Vitis HLS directives
└── run_hls.tcl             # Build script
```

### 11.3 Intel HLS Export

```rust
use dpb_export::fpga::{FpgaExporter, IntelTarget};

let config = FpgaConfig {
    target: IntelTarget::Arria10,
    clock_period_ns: 5.0,  // 200 MHz
    interface: InterfaceType::AvalonST,
};

let exporter = FpgaExporter::new(config);
let hls_code = exporter.export_level_crossing(threshold)?;
```

### 11.4 Resource Estimation

```rust
// Estimate FPGA resources before synthesis
let estimate = exporter.estimate_resources(&encoder_config)?;
println!("LUTs: {}", estimate.luts);
println!("FFs: {}", estimate.flip_flops);
println!("DSPs: {}", estimate.dsp_blocks);
println!("BRAM (Kb): {}", estimate.bram_kb);
```

---

## 12. Language Bindings

### 12.1 Binding Status Matrix

| Language | Interface | Status | Build Tool |
|----------|-----------|:------:|------------|
| **Python** | PyO3 | ✅ Complete | maturin |
| **C/C++** | cbindgen | ✅ Complete | cargo |
| **JavaScript** | wasm-bindgen | ✅ Complete | wasm-pack |
| **R** | .Call() | ✅ Complete | R CMD |
| **Julia** | CBinding.jl | ✅ Complete | Julia Pkg |
| **MATLAB** | MEX | ✅ Complete | mex |
| **LabVIEW** | CLFN | ✅ Complete | - |

### 12.2 Python

```bash
# Install from source
pip install maturin
cd crates/dpb-python
maturin develop --release

# Or install from PyPI (when published)
pip install dpb
```

```python
import dpb

# Basic encoding
encoder = dpb.LevelCrossingEncoder(threshold=0.1)
spikes = encoder.encode(signal)

# GPU-accelerated
gpu_encoder = dpb.GpuEncoder()
spikes = gpu_encoder.encode_level_crossing(signal, threshold=0.1)

# With NumPy integration
import numpy as np
signal = np.random.randn(1000).astype(np.float32)
spikes = encoder.encode(signal)
```

### 12.3 C/C++

```bash
# Generate headers
cargo build --release -p dpb-ffi
cbindgen --config cbindgen.toml --crate dpb-ffi --output dpb.h

# Link
gcc -o myapp myapp.c -L./target/release -ldpb -lpthread -ldl -lm
```

```c
#include "dpb.h"

int main() {
    DPBEncoder* encoder = dpb_level_crossing_new(0.1f);

    float signal[1000];
    // ... fill signal

    DPBSpikeArray* spikes = dpb_encode(encoder, signal, 1000);

    printf("Generated %zu spikes\n", dpb_spike_count(spikes));

    dpb_spikes_free(spikes);
    dpb_encoder_free(encoder);
    return 0;
}
```

### 12.4 JavaScript/TypeScript

```bash
cd crates/dpb-wasm
wasm-pack build --target web --release
```

```typescript
import init, { WasmLevelCrossingEncoder } from 'dpb-wasm';

await init();
const encoder = new WasmLevelCrossingEncoder(0.1);
const spikes = encoder.encode(new Float32Array(signal));
```

### 12.5 R

```r
# Install
install.packages("dpb")

# Or from source
library(devtools)
install_github("aurasense/dpb-r")

# Usage
library(dpb)
encoder <- level_crossing_encoder(threshold = 0.1)
spikes <- encode(encoder, signal)
```

### 12.6 Julia

```julia
using Pkg
Pkg.add("DPB")

using DPB

encoder = LevelCrossingEncoder(threshold=0.1)
spikes = encode(encoder, signal)
```

### 12.7 MATLAB

```matlab
% Add to path
addpath('/path/to/dpb/bindings/matlab')

% Usage
encoder = dpb.LevelCrossingEncoder(0.1);
spikes = encoder.encode(signal);

% With MEX
spikes = dpb_encode_mex(signal, 0.1);
```

### 12.8 LabVIEW

See `bindings/labview/examples/README.md` for comprehensive LabVIEW examples including:
- Basic encoding VI
- Multi-channel parallel encoding
- Real-time DAQ integration
- CompactRIO deployment

---

## 13. Streaming & Real-time

### 13.1 Lab Streaming Layer (LSL)

```bash
# Build with native LSL
cargo build --release -p dpb-lsl --features "native"
```

```rust
use dpb_lsl::{StreamInfo, Outlet, Inlet};

// Create output stream
let info = StreamInfo::new("DPB_Spikes", "Markers", 1, 0.0, "int32")?;
let outlet = Outlet::new(&info)?;

// Push spike times
outlet.push_sample(&[spike_time])?;

// Create input stream
let inlet = Inlet::new(&resolved_stream)?;
let sample = inlet.pull_sample()?;
```

### 13.2 Async Streaming

```bash
cargo build --release --features "async"
```

```rust
use dpb_lsl::AsyncInlet;

let inlet = AsyncInlet::new(&stream_info).await?;
while let Some(sample) = inlet.pull_sample().await? {
    process(sample);
}
```

---

## 14. Cloud & Container Deployment

### 14.1 Docker

```dockerfile
# Dockerfile
FROM rust:1.75-slim-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/dpb /usr/local/bin/
ENTRYPOINT ["dpb"]
```

```bash
docker build -t dpb:latest .
docker run dpb:latest encode --input signal.dat
```

### 14.2 Docker with GPU

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app
COPY . .
RUN cargo build --release --features "cuda"

FROM nvidia/cuda:12.2-runtime-ubuntu22.04
COPY --from=builder /app/target/release/dpb /usr/local/bin/
ENTRYPOINT ["dpb"]
```

```bash
docker build -f Dockerfile.gpu -t dpb:gpu .
docker run --gpus all dpb:gpu encode --batch /data/*.edf
```

### 14.3 Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dpb-encoder
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: dpb
        image: dpb:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # Optional GPU
```

---

## 15. Clinical / Regulatory

### 15.1 HIPAA Compliance

```rust
use dpb_clinical::phi::{DeIdentifier, DeIdentificationConfig};

// Safe Harbor method (removes 18 PHI identifiers)
let config = DeIdentificationConfig::safe_harbor();
let deidentifier = DeIdentifier::new(config);

let safe_record = deidentifier.deidentify(&patient_record)?;
```

### 15.2 Audit Logging

```rust
let audit_log = deidentifier.get_audit_log();
for entry in audit_log {
    println!("{}: {} on field {}", entry.timestamp, entry.action, entry.field);
}
```

### 15.3 FDA 21 CFR Part 11

**Status:** Infrastructure supports electronic records requirements. Full validation available.

---

## 16. Quick Start by Use Case

### 16.1 Research Lab (Desktop)

```bash
# Linux/macOS
cargo install dpb
dpb encode --input eeg.edf --encoder level-crossing --threshold 0.1

# Python
pip install dpb
python -c "import dpb; print(dpb.encode(signal, 'level_crossing', 0.1))"
```

### 16.2 Real-time BCI

```bash
# With Lab Streaming Layer
cargo build --release -p dpb-lsl --features "native,async"
dpb stream --lsl-inlet "EEG_Data" --encoder level-crossing --lsl-outlet "Spikes"
```

### 16.3 Mobile Health App

```bash
# iOS with Neural Engine
cargo build --release --target aarch64-apple-ios -p dpb-mobile \
    --features "metal,coreml,apple-ane"

# Android with NNAPI
cargo ndk --target aarch64-linux-android build --release -p dpb-mobile \
    --features "nnapi,hexagon"
```

### 16.4 Edge AI / Wearable

```bash
# ESP32-C3
cargo build --profile embedded --target riscv32imc-unknown-none-elf \
    --no-default-features --features "no_std,riscv-hal"
cargo espflash flash
```

### 16.5 Web Application

```bash
# Build WASM with WebGPU + Safari support
cd crates/dpb-wasm
wasm-pack build --target web --release --features "webgpu"
npx serve pkg/
```

### 16.6 Cloud Processing

```bash
# Docker with GPU
docker build -f Dockerfile.gpu -t dpb:gpu .
docker run --gpus all dpb:gpu encode --batch /data/*.edf
```

### 16.7 FPGA Accelerated

```bash
# Export to Xilinx HLS
cargo run -p dpb-export --features fpga -- \
    export --format fpga-hls --target xilinx-zynq --output level_crossing_hls/

# Then synthesize with Vitis HLS
cd level_crossing_hls
vitis_hls -f run_hls.tcl
```

### 16.8 Neuromorphic Deployment

```bash
# Export to Intel Loihi 2
cargo run -p dpb-export --features neuromorphic -- \
    export --format neuromorphic --target loihi2 --output biosignal_encoder.py

# Deploy with Lava
python biosignal_encoder.py
```

---

## Appendix A: All Deployment Regimes Checklist

| Regime | Sub-category | DPB Module | Feature Flag | Status |
|--------|--------------|------------|--------------|:------:|
| **OS** | Linux x86_64 | dpb-core | (default) | ✅ |
| | Linux ARM64 | dpb-core | (default) | ✅ |
| | macOS Intel | dpb-core | (default) | ✅ |
| | macOS ARM64 | dpb-core | (default) | ✅ |
| | Windows x64 | dpb-core | (default) | ✅ |
| **GPU** | CUDA | dpb-core | `cuda` | ✅ |
| | Metal | dpb-mobile | `metal` | ✅ |
| | Vulkan | dpb-core | `webgpu` | ✅ |
| | DirectX 12 | dpb-core | `webgpu` | ✅ |
| | AMD ROCm | dpb-core | `webgpu` | ✅ |
| **Browser** | WebGPU | dpb-wasm | `webgpu` | ✅ |
| | WebNN | dpb-wasm | `webnn` | ✅ |
| | WASM CPU | dpb-wasm | (default) | ✅ |
| | Safari WebGPU | dpb-wasm | `webgpu` | ✅ |
| **Mobile** | iOS | dpb-mobile | `ios` | ✅ |
| | Android | dpb-mobile | `android` | ✅ |
| **NPU** | Hexagon DSP | dpb-mobile | `hexagon` | ✅ |
| | ARM Ethos-U | dpb-mobile | `ethos-u` | ✅ |
| | Apple ANE | dpb-mobile | `apple-ane` | ✅ |
| **Processor** | x86_64 | dpb-core | (default) | ✅ |
| | x86 | dpb-core | (default) | ✅ |
| | ARM64 | dpb-core | (default) | ✅ |
| | ARMv7 | dpb-core | (default) | ✅ |
| | RISC-V 64 | dpb-core | `riscv-hal` | ✅ |
| | RISC-V 32 | dpb-core | `riscv-hal` | ✅ |
| **Embedded** | ESP32-C3 | dpb-core | `no_std` | ✅ |
| | STM32 | dpb-core | `no_std` | ✅ |
| | nRF52 | dpb-core | `no_std` | ✅ |
| | RP2040 | dpb-core | `no_std` | ✅ |
| **AI Accel** | Intel Gaudi | dpb-core | `intel-gaudi` | ⚠️ Sim |
| | Graphcore IPU | dpb-core | `graphcore-ipu` | ⚠️ Sim |
| **Neuromorphic** | Loihi 2 | dpb-export | `neuromorphic` | ✅ Export |
| | SpiNNaker 2 | dpb-export | `neuromorphic` | ✅ Export |
| | BrainScaleS-2 | dpb-export | `neuromorphic` | ✅ Export |
| **FPGA** | Xilinx | dpb-export | `fpga` | ✅ Export |
| | Intel | dpb-export | `fpga` | ✅ Export |

---

## Appendix B: Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `ANDROID_NDK_HOME` | Android NDK path | `/opt/android-ndk-r25c` |
| `LSL_LIB` | liblsl library path | `/usr/lib/liblsl.so` |
| `SYNAPSE_ROOT` | Intel Gaudi SDK | `/opt/habanalabs` |
| `POPLAR_SDK` | Graphcore Poplar SDK | `/opt/poplar` |
| `CUDA_PATH` | NVIDIA CUDA toolkit | `/usr/local/cuda-12.2` |
| `RUSTFLAGS` | Compiler flags | `-C target-feature=+avx512f` |

---

## Appendix C: Build Aliases Reference

From `.cargo/config.toml`:

```toml
[alias]
# Embedded targets
build-riscv32 = "build --target riscv32imc-unknown-none-elf"
build-riscv64 = "build --target riscv64gc-unknown-none-elf"
build-cortexm4 = "build --target thumbv7em-none-eabihf"
check-embedded = "check --target thumbv7em-none-eabihf"

# Mobile targets
build-ios = "build --target aarch64-apple-ios"
build-android = "ndk --target aarch64-linux-android build"

# WASM
build-wasm = "build --target wasm32-unknown-unknown"

# GPU
build-cuda = "build --release --features cuda"
build-metal = "build --release --features metal"
```

---

*Document Version: 2.0.0 | DPB Framework Version: 0.5.2 | Catalog Version: 5.3.0*
