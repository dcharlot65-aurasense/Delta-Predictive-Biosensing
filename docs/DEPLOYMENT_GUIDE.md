# Delta-Predictive Biosensing (DPB) Deployment Guide

> **Version:** 1.0.0 | **Last Updated:** December 2025
> **Purpose:** Platform-specific deployment instructions for all supported targets

---

## Table of Contents

1. [Deployment Matrix Overview](#1-deployment-matrix-overview)
2. [Operating Systems](#2-operating-systems)
3. [GPU Platforms](#3-gpu-platforms)
4. [Browser Deployment](#4-browser-deployment)
5. [Mobile Platforms](#5-mobile-platforms)
6. [Processor Architectures](#6-processor-architectures)
7. [Embedded / Edge AI / TinyML](#7-embedded--edge-ai--tinyml)
8. [AI Accelerators](#8-ai-accelerators)
9. [Neuromorphic Hardware](#9-neuromorphic-hardware)
10. [Language Bindings](#10-language-bindings)
11. [Streaming & Real-time](#11-streaming--real-time)
12. [Cloud & Container Deployment](#12-cloud--container-deployment)
13. [Clinical / Regulatory](#13-clinical--regulatory)
14. [Quick Start by Use Case](#14-quick-start-by-use-case)

---

## 1. Deployment Matrix Overview

### 1.1 Platform Support Matrix

| Platform | Status | GPU Accel | Notes |
|----------|:------:|:---------:|-------|
| **Linux x86_64** | ✅ Production | Vulkan, CUDA* | Primary development target |
| **Linux ARM64** | ✅ Production | Vulkan | Raspberry Pi 4/5, Jetson |
| **macOS x86_64** | ✅ Production | Metal | Intel Macs |
| **macOS ARM64** | ✅ Production | Metal | Apple Silicon (M1/M2/M3) |
| **Windows x86_64** | ✅ Production | DX12, Vulkan | Windows 10/11 |
| **iOS ARM64** | ✅ Production | Metal* | iPhone 8+, iPad Pro |
| **Android ARM64** | ✅ Production | Vulkan*, NNAPI* | Android 8.0+ |
| **Android ARMv7** | ✅ Production | CPU only | Legacy devices |
| **WebAssembly** | ✅ Production | WebGPU* | Modern browsers |
| **RISC-V 32-bit** | ✅ Production | CPU only | ESP32-C3, GD32VF103 |
| **RISC-V 64-bit** | ✅ Production | CPU only | SiFive, VisionFive 2 |
| **ARM Cortex-M** | ✅ Production | CPU only | STM32, nRF52, RP2040 |
| **Intel Gaudi** | ⚠️ Stub | TPC Cores | Requires Synapse SDK |
| **Graphcore IPU** | ⚠️ Stub | 1472 Tiles | Requires Poplar SDK |
| **Loihi/SpiNNaker** | ⚠️ Planned | Neuromorphic | Future implementation |

*Feature-gated; requires explicit opt-in

### 1.2 Feature Flag Summary

```toml
# Cargo.toml feature flags for deployment targets
[features]
# GPU Acceleration
webgpu = ["wgpu", "web-sys"]
metal = ["metal-rs"]           # iOS/macOS
vulkan = ["ash"]               # Android/Linux
coreml = ["coreml-sys"]        # iOS inference
nnapi = ["nnapi-sys"]          # Android inference

# Hardware Accelerators
intel-gaudi = []               # Intel AI accelerators
graphcore-ipu = []             # Graphcore IPU
hardware-accelerators = ["intel-gaudi", "graphcore-ipu"]

# Embedded
no_std = []                    # For microcontrollers
riscv-hal = []                 # RISC-V specific

# Streaming
native = ["liblsl-sys"]        # Lab Streaming Layer
async = ["tokio"]              # Async runtime
```

---

## 2. Operating Systems

### 2.1 Linux

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

# For Lab Streaming Layer
sudo apt-get install -y liblsl-dev

# Build from source
cargo build --release

# With all features
cargo build --release --features "webgpu,native,async"
```

#### Deployment Paths

| Use Case | Command | Output |
|----------|---------|--------|
| CLI Tool | `cargo build --release` | `target/release/dpb` |
| Shared Library | `cargo build --release -p dpb-ffi` | `libdpb.so` |
| Python Wheel | `maturin build --release` | `dpb-*.whl` |

### 2.2 macOS

#### Supported Versions
- macOS 12 Monterey+ (Intel)
- macOS 12 Monterey+ (Apple Silicon)

#### Installation

```bash
# Prerequisites (via Homebrew)
brew install pkg-config openssl

# For Lab Streaming Layer
brew install labstreaminglayer/tap/lsl

# Build
cargo build --release

# Universal binary (Intel + ARM)
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
lipo -create -output target/universal/libdpb.dylib \
    target/x86_64-apple-darwin/release/libdpb.dylib \
    target/aarch64-apple-darwin/release/libdpb.dylib
```

#### Metal GPU Acceleration

```bash
# Metal is automatic via wgpu on macOS
cargo build --release --features "webgpu"
```

### 2.3 Windows

#### Supported Versions
- Windows 10 (1903+)
- Windows 11

#### Installation

```powershell
# Prerequisites (via winget or chocolatey)
winget install Rustlang.Rustup

# Or via Chocolatey
choco install rust visualstudio2022-workload-vctools

# Build
cargo build --release

# With GPU (DX12/Vulkan automatic)
cargo build --release --features "webgpu"
```

#### Deployment Paths

| Use Case | Output | Notes |
|----------|--------|-------|
| CLI Tool | `dpb.exe` | Standalone executable |
| DLL | `dpb.dll` | For C/C++ integration |
| Python | `dpb-*.whl` | Via `maturin build` |

---

## 3. GPU Platforms

### 3.1 GPU Support Matrix

| GPU Platform | OS Support | Backend | Status | Performance |
|--------------|------------|---------|:------:|:-----------:|
| **NVIDIA CUDA** | Linux, Windows | wgpu/Vulkan | ⚠️ Indirect | Good |
| **NVIDIA (Vulkan)** | Linux, Windows | wgpu | ✅ Native | Excellent |
| **AMD (Vulkan)** | Linux, Windows | wgpu | ✅ Native | Excellent |
| **AMD (ROCm)** | Linux | wgpu | ⚠️ Indirect | Good |
| **Intel (Vulkan)** | Linux, Windows | wgpu | ✅ Native | Good |
| **Apple Metal** | macOS, iOS | wgpu | ✅ Native | Excellent |
| **DirectX 12** | Windows | wgpu | ✅ Native | Excellent |
| **WebGPU** | Browser | wgpu | ✅ Native | Good |

### 3.2 CUDA Integration

DPB uses `wgpu` which abstracts GPU backends. For NVIDIA GPUs:

```bash
# Option 1: Vulkan backend (recommended)
# NVIDIA drivers include Vulkan support
cargo build --release --features "webgpu"

# Option 2: Direct CUDA (requires custom integration)
# Currently not implemented - use Vulkan instead
```

**CUDA Limitation:** Direct CUDA kernel calls are not supported. For maximum NVIDIA performance, use Vulkan which has excellent NVIDIA support.

### 3.3 Metal (Apple)

```bash
# Automatic on macOS/iOS
cargo build --release --features "webgpu"

# iOS-specific with CoreML inference
cargo build --release --target aarch64-apple-ios --features "metal,coreml"
```

### 3.4 Vulkan

```bash
# Linux
sudo apt-get install libvulkan-dev
cargo build --release --features "webgpu"

# Verify Vulkan support
vulkaninfo | grep "GPU"

# Android (automatic with NDK)
cargo ndk --target aarch64-linux-android build --features "vulkan"
```

### 3.5 AMD ROCm

```bash
# ROCm provides Vulkan support
# Install ROCm runtime, then use Vulkan backend
sudo apt-get install rocm-dev

# Build with Vulkan (ROCm provides Vulkan ICD)
cargo build --release --features "webgpu"
```

---

## 4. Browser Deployment

### 4.1 Browser Support Matrix

| Browser | WebGPU | WebGL | WASM | WebNN |
|---------|:------:|:-----:|:----:|:-----:|
| **Chrome 113+** | ✅ | ✅ | ✅ | ⚠️ Flag |
| **Edge 113+** | ✅ | ✅ | ✅ | ⚠️ Flag |
| **Firefox 121+** | ✅ | ✅ | ✅ | ❌ |
| **Safari 17+** | ⚠️ Exp | ✅ | ✅ | ❌ |
| **Chrome Android** | ✅ | ✅ | ✅ | ❌ |
| **Safari iOS** | ⚠️ Exp | ✅ | ✅ | ❌ |

### 4.2 WebAssembly Build

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for web
cd crates/dpb-wasm
wasm-pack build --target web --release

# Output: pkg/dpb_wasm.js, pkg/dpb_wasm_bg.wasm
```

### 4.3 WebGPU (GPU Acceleration)

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
    const encoder = await GpuEncoder.new();
    const result = await encoder.encode_level_crossing(timeseries, threshold);
}
```

### 4.4 WebGL Fallback

WebGL compute is limited. For non-WebGPU browsers:

```javascript
// CPU fallback (always available)
import { WasmLevelCrossingEncoder } from './pkg/dpb_wasm.js';

const encoder = new WasmLevelCrossingEncoder(threshold);
const spikes = encoder.encode(samples);
```

### 4.5 WebNN (Experimental)

WebNN support is experimental and browser-specific:

```javascript
// Check WebNN availability
if ('ml' in navigator) {
    // WebNN available - future DPB integration
}
```

**Current Status:** WebNN integration is planned but not yet implemented.

### 4.6 Browser Testing

```bash
# Run browser tests
cd crates/dpb-wasm/tests/web
python -m http.server 8080
# Open http://localhost:8080/index.html
```

---

## 5. Mobile Platforms

### 5.1 iOS

#### Supported Devices
- iPhone 8+ (A11 Bionic or later)
- iPad Pro (2018+)
- iPad Air (2020+)
- iOS 14.0+

#### Build Setup

```bash
# Install iOS targets
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-sim
rustup target add x86_64-apple-ios  # Intel simulator

# Build static library
cargo build --release --target aarch64-apple-ios -p dpb-mobile

# Output: target/aarch64-apple-ios/release/libdpb_mobile.a
```

#### Xcode Integration

```swift
// Swift bridging header
#import "dpb_mobile.h"

// Swift usage
let encoder = DPBLevelCrossingEncoder(threshold: 0.1)
let spikes = encoder.encode(samples: signalData)
```

#### GPU Acceleration (Metal)

```bash
# Build with Metal support
cargo build --release --target aarch64-apple-ios \
    --features "metal" -p dpb-mobile
```

#### CoreML Inference

```bash
# Build with CoreML support
cargo build --release --target aarch64-apple-ios \
    --features "coreml" -p dpb-mobile
```

### 5.2 Android

#### Supported Devices
- ARM64: Android 8.0+ (API 26+)
- ARMv7: Android 5.0+ (API 21+)
- x86_64: Emulator

#### Build Setup

```bash
# Install Android targets
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add x86_64-linux-android
rustup target add i686-linux-android

# Install cargo-ndk
cargo install cargo-ndk

# Set NDK path
export ANDROID_NDK_HOME=/path/to/android-ndk

# Build
cargo ndk --target aarch64-linux-android --platform 26 \
    build --release -p dpb-mobile

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
}
```

#### GPU Acceleration (Vulkan)

```bash
# Build with Vulkan support
cargo ndk --target aarch64-linux-android --platform 26 \
    build --release -p dpb-mobile --features "vulkan"
```

#### NNAPI Inference

```bash
# Build with Android Neural Networks API
cargo ndk --target aarch64-linux-android --platform 26 \
    build --release -p dpb-mobile --features "nnapi"
```

---

## 6. Processor Architectures

### 6.1 Architecture Support Matrix

| Architecture | Target Triple | SIMD | Status |
|--------------|---------------|------|:------:|
| **x86_64** | `x86_64-unknown-linux-gnu` | AVX2, AVX-512 | ✅ |
| **x86** | `i686-unknown-linux-gnu` | SSE2, SSE4 | ✅ |
| **ARM64** | `aarch64-unknown-linux-gnu` | NEON | ✅ |
| **ARMv7** | `armv7-unknown-linux-gnueabihf` | NEON | ✅ |
| **ARMv6** | `arm-unknown-linux-gnueabihf` | None | ⚠️ |
| **RISC-V 64** | `riscv64gc-unknown-linux-gnu` | V-ext* | ✅ |
| **RISC-V 32** | `riscv32imc-unknown-none-elf` | None | ✅ |

### 6.2 x86_64 (Intel/AMD)

```bash
# Standard build (auto-detects AVX2)
cargo build --release

# Force AVX-512 (if available)
RUSTFLAGS="-C target-feature=+avx512f" cargo build --release

# Check CPU features
cargo build --release
./target/release/dpb --show-simd
```

### 6.3 ARM64 (Apple Silicon, Raspberry Pi, Jetson)

```bash
# Native build on ARM64
cargo build --release

# Cross-compile from x86_64
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu

# Raspberry Pi 4/5
cargo build --release --target aarch64-unknown-linux-gnu
```

### 6.4 ARMv7 (32-bit ARM)

```bash
# Cross-compile
rustup target add armv7-unknown-linux-gnueabihf
cargo build --release --target armv7-unknown-linux-gnueabihf

# For Android (older devices)
cargo ndk --target armv7-linux-androideabi build --release
```

---

## 7. Embedded / Edge AI / TinyML

### 7.1 Embedded Platform Matrix

| Platform | MCU | RAM | Flash | Target | Features |
|----------|-----|-----|-------|--------|----------|
| **ESP32-C3** | RISC-V | 400KB | 4MB | `riscv32imc` | WiFi, BLE |
| **ESP32-S3** | RISC-V | 512KB | 8MB | `riscv32imafc` | WiFi, AI accel |
| **STM32F4** | Cortex-M4 | 192KB | 1MB | `thumbv7em` | DSP, FPU |
| **STM32H7** | Cortex-M7 | 1MB | 2MB | `thumbv7em` | DSP, FPU |
| **nRF52840** | Cortex-M4 | 256KB | 1MB | `thumbv7em` | BLE 5.0 |
| **RP2040** | Cortex-M0+ | 264KB | 2MB | `thumbv6m` | Dual-core |
| **GD32VF103** | RISC-V | 32KB | 128KB | `riscv32imac` | Low power |
| **SiFive U74** | RISC-V 64 | 8GB | - | `riscv64gc` | Linux capable |

### 7.2 Build Configuration

```bash
# RISC-V 32-bit (ESP32-C3)
cargo build-riscv32  # Alias defined in .cargo/config.toml

# RISC-V 64-bit (SiFive)
cargo build-riscv64

# ARM Cortex-M4 (STM32F4)
cargo build-cortexm4

# Check embedded targets
cargo check-embedded
```

### 7.3 Memory-Constrained Builds

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

### 7.4 no_std Support

```rust
// Cargo.toml
[features]
no_std = []
std = ["alloc"]

// lib.rs
#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate alloc;
```

```bash
# Build without standard library
cargo build --release --target riscv32imc-unknown-none-elf \
    --no-default-features --features "no_std"
```

### 7.5 Fixed-Point Arithmetic (FPU-less targets)

For targets without FPU, use the RISC-V HAL with Q16 fixed-point:

```rust
use dpb_core::accelerators::riscv::{Q16, FixedPoint, encode_level_crossing_fixed};

// Convert float to Q16.16 fixed-point
let threshold = Q16::from_float(0.1);
let samples: Vec<Q16> = raw_samples.iter()
    .map(|&x| Q16::from_float(x))
    .collect();

// Encode with fixed-point arithmetic
let spikes = encode_level_crossing_fixed(&samples, threshold);
```

### 7.6 Flashing and Debugging

```bash
# Install probe-run for ARM Cortex-M
cargo install probe-run

# Flash and run on STM32
cargo run --release --target thumbv7em-none-eabihf

# For ESP32 (RISC-V)
cargo install espflash
cargo espflash flash --release --target riscv32imc-unknown-none-elf
```

---

## 8. AI Accelerators

### 8.1 Intel Gaudi (Habana Labs)

#### Hardware Requirements
- Intel Gaudi 2 or Gaudi 3
- Synapse AI SDK installed
- Linux (Ubuntu 20.04/22.04)

#### Build

```bash
# Enable Gaudi feature
cargo build --release --features "intel-gaudi"

# With Synapse SDK (when available)
export HABANA_LOGS=/var/log/habana
export SYNAPSE_ROOT=/opt/habanalabs
cargo build --release --features "intel-gaudi"
```

#### Usage

```rust
use dpb_core::accelerators::gaudi::{GaudiAccelerator, GaudiConfig};

let config = GaudiConfig::default();
let accel = GaudiAccelerator::new(config)?;

// Check availability
if accel.is_available() {
    let result = accel.encode_signal(&signal)?;
}
```

**Current Status:** Simulation mode only. Production requires Synapse AI SDK.

### 8.2 Graphcore IPU

#### Hardware Requirements
- Graphcore Bow IPU or C600
- Poplar SDK installed
- Linux (Ubuntu 20.04/22.04)

#### Build

```bash
# Enable IPU feature
cargo build --release --features "graphcore-ipu"

# With Poplar SDK (when available)
export POPLAR_SDK=/opt/poplar
source $POPLAR_SDK/enable
cargo build --release --features "graphcore-ipu"
```

#### Usage

```rust
use dpb_core::accelerators::ipu::{IpuAccelerator, IpuConfig};

let config = IpuConfig::default();
let accel = IpuAccelerator::new(config)?;

// Create computation graph
let graph = accel.create_graph()?;
graph.add_vertex("level_crossing", &params)?;
let program = graph.compile()?;
```

**Current Status:** Simulation mode only. Production requires Poplar SDK.

### 8.3 Google TPU

**Status:** Not currently supported. Consider ONNX export for TPU inference.

### 8.4 AWS Inferentia/Trainium

**Status:** Not currently supported. Consider ONNX export with AWS Neuron SDK.

---

## 9. Neuromorphic Hardware

### 9.1 Support Status

| Platform | Vendor | Status | Export Format |
|----------|--------|:------:|---------------|
| **Intel Loihi 2** | Intel | ⚠️ Planned | Lava/NXSDK |
| **SpiNNaker 2** | Manchester | ⚠️ Planned | PyNN |
| **BrainScaleS-2** | Heidelberg | ⚠️ Planned | PyNN |

### 9.2 Future Integration

Neuromorphic export will be available via:

```rust
// Planned API
use dpb_snn::export::neuromorphic::{LoihiExporter, SpinnakerExporter};

let model = /* trained SNN */;
let loihi_export = LoihiExporter::export(&model)?;
loihi_export.save("model.lava")?;
```

**Current Status:** Architecture defined but not implemented.

---

## 10. Language Bindings

### 10.1 Binding Status Matrix

| Language | Interface | Status | Build Tool |
|----------|-----------|:------:|------------|
| **Python** | PyO3 | ✅ Complete | maturin |
| **C/C++** | cbindgen | ✅ Complete | cargo |
| **JavaScript** | wasm-bindgen | ✅ Complete | wasm-pack |
| **R** | .Call() | 📄 Planned | R CMD |
| **Julia** | CBinding.jl | 📄 Planned | Julia Pkg |
| **MATLAB** | MEX | 📄 Planned | mex |
| **LabVIEW** | CLFN | 📄 Planned | - |

### 10.2 Python

```bash
# Install from source
pip install maturin
cd crates/dpb-python
maturin develop --release

# Usage
import dpb
encoder = dpb.LevelCrossingEncoder(threshold=0.1)
spikes = encoder.encode(signal)
```

### 10.3 C/C++

```bash
# Generate headers
cargo build --release -p dpb-ffi
cbindgen --config cbindgen.toml --crate dpb-ffi --output dpb.h

# Link
gcc -o myapp myapp.c -L./target/release -ldpb -lpthread -ldl -lm
```

### 10.4 JavaScript/TypeScript

```typescript
import init, { WasmLevelCrossingEncoder } from 'dpb-wasm';

await init();
const encoder = new WasmLevelCrossingEncoder(0.1);
const spikes = encoder.encode(new Float32Array(signal));
```

---

## 11. Streaming & Real-time

### 11.1 Lab Streaming Layer (LSL)

```bash
# Build with native LSL
cargo build --release -p dpb-lsl --features "native"

# Usage
use dpb_lsl::{StreamInfo, Outlet};

let info = StreamInfo::new("DPB_Spikes", "Markers", 1, 0.0, "int32")?;
let outlet = Outlet::new(&info)?;
outlet.push_sample(&[spike_time])?;
```

### 11.2 Async Streaming

```bash
# Build with async support
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

## 12. Cloud & Container Deployment

### 12.1 Docker

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

### 12.2 Docker with GPU

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.2-runtime-ubuntu22.04
# ... install Rust and build with Vulkan support
```

### 12.3 Kubernetes

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

## 13. Clinical / Regulatory

### 13.1 HIPAA Compliance

```rust
use dpb_clinical::phi::{DeIdentifier, DeIdentificationConfig};

// Safe Harbor method (removes 18 PHI identifiers)
let config = DeIdentificationConfig::safe_harbor();
let deidentifier = DeIdentifier::new(config);

let safe_record = deidentifier.deidentify(&patient_record)?;
```

### 13.2 Audit Logging

```rust
// All PHI operations are logged
let audit_log = deidentifier.get_audit_log();
for entry in audit_log {
    println!("{}: {} on field {}", entry.timestamp, entry.action, entry.field);
}
```

### 13.3 FDA 21 CFR Part 11

**Status:** Infrastructure supports electronic records requirements. Full validation pending.

---

## 14. Quick Start by Use Case

### 14.1 Research Lab (Desktop)

```bash
# Linux/macOS
cargo install dpb
dpb encode --input eeg.edf --encoder level-crossing --threshold 0.1

# Python
pip install dpb
python -c "import dpb; print(dpb.encode(signal, 'level_crossing', 0.1))"
```

### 14.2 Real-time BCI

```bash
# With Lab Streaming Layer
cargo build --release -p dpb-lsl --features "native,async"

# Stream from EEG device
dpb stream --lsl-inlet "EEG_Data" --encoder level-crossing --lsl-outlet "Spikes"
```

### 14.3 Mobile Health App

```bash
# iOS
cargo build --release --target aarch64-apple-ios -p dpb-mobile --features "metal"

# Android
cargo ndk --target aarch64-linux-android build --release -p dpb-mobile --features "nnapi"
```

### 14.4 Edge AI / Wearable

```bash
# ESP32-C3
cargo build --profile embedded --target riscv32imc-unknown-none-elf \
    --no-default-features --features "no_std,riscv-hal"
cargo espflash flash
```

### 14.5 Web Application

```bash
# Build WASM
cd crates/dpb-wasm
wasm-pack build --target web --release --features "webgpu"

# Serve
npx serve pkg/
```

### 14.6 Cloud Processing

```bash
# Docker with GPU
docker build -f Dockerfile.gpu -t dpb:gpu .
docker run --gpus all dpb:gpu encode --batch /data/*.edf
```

---

## Appendix A: Missing Regimes Identified

Based on your question, here are additional deployment regimes not fully addressed:

| Regime | Current Status | Priority | Notes |
|--------|:-------------:|:--------:|-------|
| **WebGL Compute** | ❌ Not supported | Low | WebGPU supersedes |
| **WebNN** | ❌ Planned | Medium | Browser ML inference |
| **OpenCL** | ❌ Not supported | Low | Vulkan preferred |
| **FPGA (Xilinx/Intel)** | ❌ Not supported | Medium | HLS export possible |
| **TPU** | ❌ Not supported | Low | Use ONNX export |
| **AWS Inferentia** | ❌ Not supported | Medium | Use ONNX + Neuron |
| **Qualcomm Hexagon** | ❌ Not supported | Medium | Mobile DSP |
| **ARM Ethos-U** | ❌ Not supported | Medium | Cortex-M NPU |

---

## Appendix B: Build Aliases Reference

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
```

---

## Appendix C: Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `ANDROID_NDK_HOME` | Android NDK path | `/opt/android-ndk-r25c` |
| `LSL_LIB` | liblsl library path | `/usr/lib/liblsl.so` |
| `SYNAPSE_ROOT` | Intel Gaudi SDK | `/opt/habanalabs` |
| `POPLAR_SDK` | Graphcore Poplar SDK | `/opt/poplar` |
| `RUSTFLAGS` | Compiler flags | `-C target-feature=+avx512f` |

---

*Document Version: 1.0.0 | DPB Framework Version: 0.5.1*
