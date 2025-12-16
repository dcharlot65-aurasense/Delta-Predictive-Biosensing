# DPB Framework: Cross-Platform GPU Implementation Architecture

## Rust Core with GPU Acceleration + Multi-Language Bindings

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           dpb-core (Rust)                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    GPU Compute Abstraction (wgpu)                      │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │ │
│  │  │   Vulkan    │  │    Metal    │  │    DX12     │  │   WebGPU    │  │ │
│  │  │  (Linux/Win)│  │   (macOS)   │  │  (Windows)  │  │  (Browser)  │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      Algorithm Kernels (WGSL)                          │ │
│  │  • Event encoders  • Template matching  • SNN simulation               │ │
│  │  • Signal processing  • FFT/filtering  • Matrix operations             │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         Rust API Layer                                 │ │
│  │  • Type-safe interfaces  • Memory management  • Error handling         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
    │  Python │      │  Julia  │      │ MATLAB  │      │  WASM   │
    │  PyO3 + │      │  jlrs   │      │  MEX +  │      │  wasm-  │
    │ maturin │      │         │      │ C FFI   │      │ bindgen │
    └─────────┘      └─────────┘      └─────────┘      └─────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ NumPy   │      │ Arrays  │      │ Arrays  │      │ Browser │
    │ interop │      │ interop │      │ interop │      │ WebGPU  │
    └─────────┘      └─────────┘      └─────────┘      └─────────┘
```

---

## 2. Core Technology Stack

### 2.1 GPU Compute: wgpu

**Why wgpu:**
- Implements WebGPU specification (future-proof)
- Cross-platform: Vulkan, Metal, DX12, OpenGL (fallback), WebGPU (browser)
- Same Rust code compiles to native AND WASM
- Active development (Mozilla + community)
- No vendor lock-in (unlike CUDA)

```toml
# Cargo.toml - Core dependencies
[dependencies]
wgpu = "23.0"                    # GPU compute
bytemuck = { version = "1.14", features = ["derive"] }  # GPU buffer casting
pollster = "0.4"                 # Async runtime for native
raw-window-handle = "0.6"        # Window integration (if needed)

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = ["console", "GpuDevice", "GpuQueue"] }
console_error_panic_hook = "0.1"
```

### 2.2 Shader Language: WGSL

WebGPU Shading Language (WGSL) is the shader language for wgpu. It runs on all backends.

```wgsl
// Example: Level-crossing event encoder kernel
@group(0) @binding(0) var<storage, read> signal: array<f32>;
@group(0) @binding(1) var<storage, read> thresholds: array<f32>;
@group(0) @binding(2) var<storage, read_write> events: array<Event>;
@group(0) @binding(3) var<storage, read_write> event_count: atomic<u32>;

struct Event {
    timestamp: f32,
    channel: u32,
    polarity: i32,
    magnitude: f32,
}

@compute @workgroup_size(256)
fn level_crossing_encode(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let n = arrayLength(&signal);
    
    if (idx >= n - 1u) { return; }
    
    let current = signal[idx];
    let next = signal[idx + 1u];
    let threshold = thresholds[0];  // Could be per-channel
    
    // Detect upward crossing
    if (current < threshold && next >= threshold) {
        let event_idx = atomicAdd(&event_count, 1u);
        events[event_idx] = Event(
            f32(idx),  // timestamp (sample index)
            0u,        // channel
            1,         // polarity (up)
            next - threshold  // magnitude
        );
    }
    
    // Detect downward crossing
    if (current >= threshold && next < threshold) {
        let event_idx = atomicAdd(&event_count, 1u);
        events[event_idx] = Event(
            f32(idx),
            0u,
            -1,        // polarity (down)
            threshold - next
        );
    }
}
```

### 2.3 Alternative: rust-gpu (Experimental)

For more complex kernels, rust-gpu allows writing GPU shaders in Rust:

```toml
# For shader development in Rust syntax (compiles to SPIR-V)
[dependencies]
spirv-std = "0.9"  # When using rust-gpu
```

---

## 3. Required Crates by Function

### 3.1 Core Compute

| Crate | Version | Purpose |
|-------|---------|---------|
| `wgpu` | 23.0 | GPU abstraction |
| `naga` | 23.0 | Shader compilation (included with wgpu) |
| `bytemuck` | 1.14 | Safe GPU buffer transmutation |
| `glam` | 0.29 | SIMD-accelerated math (vec3, mat4) |
| `nalgebra` | 0.33 | Linear algebra |
| `ndarray` | 0.16 | N-dimensional arrays |
| `rustfft` | 6.2 | FFT (CPU fallback, or as reference) |
| `realfft` | 3.3 | Real-valued FFT |
| `num-complex` | 0.4 | Complex numbers |

### 3.2 Signal Processing

| Crate | Version | Purpose |
|-------|---------|---------|
| `biquad` | 0.4 | IIR filter design |
| `dasp` | 0.11 | Digital audio signal processing |
| `spectrum-analyzer` | 1.5 | Spectral analysis |
| `rubato` | 0.15 | Resampling |
| `hound` | 3.5 | WAV file I/O |

### 3.3 Async & Parallelism

| Crate | Version | Purpose |
|-------|---------|---------|
| `rayon` | 1.10 | CPU parallelism |
| `tokio` | 1.41 | Async runtime (native) |
| `pollster` | 0.4 | Minimal async (for GPU init) |
| `crossbeam` | 0.8 | Lock-free data structures |
| `parking_lot` | 0.12 | Fast mutexes |

### 3.4 Serialization & Data

| Crate | Version | Purpose |
|-------|---------|---------|
| `serde` | 1.0 | Serialization framework |
| `serde_json` | 1.0 | JSON |
| `bincode` | 1.3 | Binary serialization |
| `rmp-serde` | 1.3 | MessagePack |
| `arrow` | 53.0 | Apache Arrow (columnar data) |
| `polars` | 0.44 | DataFrames |

### 3.5 Error Handling & Logging

| Crate | Version | Purpose |
|-------|---------|---------|
| `thiserror` | 2.0 | Error types |
| `anyhow` | 1.0 | Error propagation |
| `tracing` | 0.1 | Structured logging |
| `tracing-subscriber` | 0.3 | Log output |

---

## 4. Language Bindings

### 4.1 Python Bindings (PyO3 + maturin)

**Setup:**
```toml
# Cargo.toml
[lib]
name = "dpb_python"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"  # NumPy interop

[build-dependencies]
pyo3-build-config = "0.22"
```

**Build tool:**
```bash
# Install maturin
pip install maturin

# Build wheel
maturin build --release

# Develop mode (editable install)
maturin develop
```

**Example binding:**
```rust
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

#[pyclass]
struct DPBEncoder {
    inner: dpb_core::LevelCrossingEncoder,
}

#[pymethods]
impl DPBEncoder {
    #[new]
    fn new(threshold: f32) -> Self {
        Self {
            inner: dpb_core::LevelCrossingEncoder::new(threshold),
        }
    }
    
    fn encode<'py>(
        &self,
        py: Python<'py>,
        signal: PyReadonlyArray1<f32>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let signal_slice = signal.as_slice()?;
        let events = self.inner.encode(signal_slice);
        
        // Convert events to numpy array
        let n_events = events.len();
        let mut output = vec![0.0f32; n_events * 4];
        for (i, event) in events.iter().enumerate() {
            output[i * 4] = event.timestamp;
            output[i * 4 + 1] = event.channel as f32;
            output[i * 4 + 2] = event.polarity as f32;
            output[i * 4 + 3] = event.magnitude;
        }
        
        Ok(PyArray2::from_vec2(py, &output.chunks(4).map(|c| c.to_vec()).collect::<Vec<_>>())?)
    }
}

#[pymodule]
fn dpb_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DPBEncoder>()?;
    Ok(())
}
```

**Python usage:**
```python
import numpy as np
import dpb_python as dpb

encoder = dpb.DPBEncoder(threshold=0.1)
signal = np.random.randn(10000).astype(np.float32)
events = encoder.encode(signal)
print(f"Generated {len(events)} events")
```

### 4.2 Julia Bindings

**Option A: jlrs (Recommended)**

```toml
# Cargo.toml
[dependencies]
jlrs = { version = "0.21", features = ["full"] }
```

```rust
use jlrs::prelude::*;

fn encode_signal(signal: TypedArray<f32>) -> JlrsResult<TypedArray<f32>> {
    // Implementation
}

julia_module! {
    become dpb_julia_init;
    
    fn encode_signal(signal: TypedArray<f32>) -> JlrsResult<TypedArray<f32>>;
}
```

**Option B: C FFI + ccall**

```rust
// C-compatible interface
#[no_mangle]
pub extern "C" fn dpb_encode(
    signal: *const f32,
    signal_len: usize,
    threshold: f32,
    events_out: *mut f32,
    events_capacity: usize,
) -> usize {
    // Implementation returning number of events
}
```

```julia
# Julia usage
const libdpb = "path/to/libdpb_core.so"

function encode(signal::Vector{Float32}, threshold::Float32)
    events = Vector{Float32}(undef, length(signal) * 4)
    n_events = ccall(
        (:dpb_encode, libdpb),
        Csize_t,
        (Ptr{Float32}, Csize_t, Float32, Ptr{Float32}, Csize_t),
        signal, length(signal), threshold, events, length(events)
    )
    return reshape(events[1:n_events*4], 4, :)'
end
```

### 4.3 MATLAB Bindings

**Option A: MEX via C FFI**

```c
// dpb_mex.c - MEX wrapper
#include "mex.h"
#include "dpb_core.h"  // Generated C header from Rust

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Get input signal
    float *signal = (float *)mxGetData(prhs[0]);
    size_t signal_len = mxGetNumberOfElements(prhs[0]);
    float threshold = (float)mxGetScalar(prhs[1]);
    
    // Allocate output
    float *events = (float *)mxMalloc(signal_len * 4 * sizeof(float));
    
    // Call Rust function
    size_t n_events = dpb_encode(signal, signal_len, threshold, events, signal_len * 4);
    
    // Create output matrix
    plhs[0] = mxCreateNumericMatrix(n_events, 4, mxSINGLE_CLASS, mxREAL);
    memcpy(mxGetData(plhs[0]), events, n_events * 4 * sizeof(float));
    
    mxFree(events);
}
```

**Generate C header from Rust:**
```toml
# Cargo.toml
[dependencies]
cbindgen = "0.27"
```

```rust
// build.rs
fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::generate(&crate_dir)
        .expect("Unable to generate bindings")
        .write_to_file("include/dpb_core.h");
}
```

**Option B: MATLAB via Python**

```matlab
% Use Python bindings from MATLAB
py.importlib.import_module('dpb_python');
encoder = py.dpb_python.DPBEncoder(0.1);
signal = py.numpy.array(randn(10000, 1), 'float32');
events = encoder.encode(signal);
events_matlab = double(events);
```

### 4.4 WASM + WebGPU Bindings

**Setup:**
```toml
# Cargo.toml
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
web-sys = { version = "0.3", features = [
    "console",
    "Window",
    "Navigator",
    "Gpu",
    "GpuAdapter",
    "GpuDevice",
    "GpuQueue",
    "GpuBuffer",
    "GpuBufferDescriptor",
    "GpuBufferUsage",
    "GpuCommandEncoder",
    "GpuComputePassEncoder",
    "GpuComputePipeline",
    "GpuBindGroup",
    "GpuShaderModule",
]}

[target.'cfg(target_arch = "wasm32")'.dependencies]
console_error_panic_hook = "0.1"
wee_alloc = "0.4"  # Smaller allocator

[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
```

**WASM-compatible code:**
```rust
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

#[wasm_bindgen]
pub struct DPBEncoderWasm {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

#[wasm_bindgen]
impl DPBEncoderWasm {
    #[wasm_bindgen(constructor)]
    pub async fn new() -> Result<DPBEncoderWasm, JsValue> {
        console_error_panic_hook::set_once();
        
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| JsValue::from_str("No adapter found"))?;
        
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        // Create compute pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DPB Encoder Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/encoder.wgsl").into()),
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DPB Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("level_crossing_encode"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        Ok(Self { device, queue, pipeline })
    }
    
    #[wasm_bindgen]
    pub fn encode(&self, signal: &[f32]) -> Vec<f32> {
        // GPU encoding implementation
        // Returns flattened event array
        vec![]  // Placeholder
    }
}
```

**Build for WASM:**
```bash
# Install wasm-pack
cargo install wasm-pack

# Build
wasm-pack build --target web --release

# Output in pkg/ directory:
# - dpb_wasm.js
# - dpb_wasm_bg.wasm
# - dpb_wasm.d.ts (TypeScript types)
```

**JavaScript usage:**
```javascript
import init, { DPBEncoderWasm } from './pkg/dpb_wasm.js';

async function main() {
    await init();
    
    const encoder = await new DPBEncoderWasm();
    const signal = new Float32Array(10000);
    // Fill signal...
    
    const events = encoder.encode(signal);
    console.log(`Generated ${events.length / 4} events`);
}

main();
```

---

## 5. Project Structure

```
dpb-framework/
├── Cargo.toml                 # Workspace root
├── crates/
│   ├── dpb-core/             # Core algorithms (no bindings)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── encoders/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── level_crossing.rs
│   │   │   │   ├── template_deviation.rs
│   │   │   │   ├── derivative.rs
│   │   │   │   └── ...
│   │   │   ├── templates/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── gait.rs
│   │   │   │   ├── tremor.rs
│   │   │   │   ├── saccade.rs
│   │   │   │   ├── voice.rs
│   │   │   │   └── ...
│   │   │   ├── snn/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── neurons.rs
│   │   │   │   ├── layers.rs
│   │   │   │   └── networks.rs
│   │   │   ├── gpu/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── context.rs
│   │   │   │   ├── buffers.rs
│   │   │   │   ├── pipelines.rs
│   │   │   │   └── kernels.rs
│   │   │   └── signal/
│   │   │       ├── mod.rs
│   │   │       ├── filters.rs
│   │   │       ├── fft.rs
│   │   │       └── resample.rs
│   │   └── shaders/
│   │       ├── encoder.wgsl
│   │       ├── snn_lif.wgsl
│   │       ├── template_match.wgsl
│   │       ├── fft.wgsl
│   │       └── ...
│   │
│   ├── dpb-python/           # Python bindings
│   │   ├── Cargo.toml
│   │   ├── pyproject.toml    # Python project config
│   │   ├── src/
│   │   │   └── lib.rs
│   │   └── python/
│   │       └── dpb/
│   │           ├── __init__.py
│   │           └── ...
│   │
│   ├── dpb-julia/            # Julia bindings
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   │
│   ├── dpb-wasm/             # WASM + WebGPU
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   └── lib.rs
│   │   └── www/              # Demo web app
│   │       ├── index.html
│   │       └── index.js
│   │
│   └── dpb-ffi/              # C FFI for MATLAB/others
│       ├── Cargo.toml
│       ├── src/
│       │   └── lib.rs
│       ├── include/
│       │   └── dpb_core.h    # Generated
│       └── matlab/
│           ├── dpb_mex.c
│           └── +dpb/         # MATLAB package
│
├── examples/
│   ├── rust/
│   ├── python/
│   ├── julia/
│   ├── matlab/
│   └── web/
│
├── tests/
│   ├── integration/
│   └── benchmarks/
│
└── docs/
    ├── api/
    └── tutorials/
```

---

## 6. GPU Kernel Categories

### 6.1 Event Encoding Kernels (WGSL)

| Kernel | Input | Output | Parallelization |
|--------|-------|--------|-----------------|
| `level_crossing_1d` | Signal buffer | Event buffer | Per-sample |
| `level_crossing_2d` | Multi-channel signal | Event buffer | Per-sample × channel |
| `template_deviation` | Signal + template | Event buffer | Per-sample |
| `derivative_threshold` | Signal | Event buffer | Per-sample |
| `keypoint_deviation` | Pose keypoints | Event buffer | Per-joint |
| `spectral_event` | FFT magnitudes | Event buffer | Per-frequency bin |

### 6.2 SNN Simulation Kernels

| Kernel | Input | Output | Parallelization |
|--------|-------|--------|-----------------|
| `lif_step` | Spikes + weights + state | New state + output spikes | Per-neuron |
| `alif_step` | Spikes + weights + state + threshold | New state + output | Per-neuron |
| `stdp_update` | Pre/post spikes + weights | Updated weights | Per-synapse |
| `spike_accumulate` | Spike trains | Firing rates | Per-neuron |

### 6.3 Signal Processing Kernels

| Kernel | Input | Output | Parallelization |
|--------|-------|--------|-----------------|
| `fft_radix2` | Time-domain signal | Frequency domain | Butterfly stages |
| `ifft_radix2` | Frequency domain | Time-domain signal | Butterfly stages |
| `filter_iir` | Signal + coefficients | Filtered signal | Per-sample (sequential) |
| `filter_fir` | Signal + coefficients | Filtered signal | Per-sample (parallel) |
| `resample` | Signal + ratio | Resampled signal | Per-output-sample |
| `spectrogram` | Signal + window | STFT output | Per-frame |

### 6.4 Template Matching Kernels

| Kernel | Input | Output | Parallelization |
|--------|-------|--------|-----------------|
| `template_correlate` | Signal + template | Correlation | Per-lag |
| `dtw_distance` | Sequence pair | DTW matrix | Per-cell (wave-front) |
| `z_score_batch` | Signals + stats | Z-scores | Per-sample × channel |

---

## 7. Build & CI/CD Requirements

### 7.1 Rust Toolchain

```bash
# Required components
rustup default stable
rustup target add wasm32-unknown-unknown  # WASM
rustup component add clippy rustfmt       # Linting

# For development
cargo install cargo-watch    # Auto-rebuild
cargo install cargo-nextest  # Fast testing
cargo install wasm-pack      # WASM packaging
cargo install maturin        # Python packaging
```

### 7.2 CI Configuration (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --release
      - run: cargo test
      - run: cargo clippy -- -D warnings
  
  wasm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
      - run: cargo install wasm-pack
      - run: wasm-pack build crates/dpb-wasm --release
  
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install maturin pytest numpy
      - run: cd crates/dpb-python && maturin build --release
      - run: pip install target/wheels/*.whl
      - run: pytest tests/python/

  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo bench
```

### 7.3 Platform Requirements

| Platform | GPU Backend | Requirements |
|----------|-------------|--------------|
| Linux | Vulkan | vulkan-loader, GPU drivers |
| macOS | Metal | macOS 10.15+, Metal-capable GPU |
| Windows | DX12/Vulkan | Windows 10+, DX12 capable GPU |
| Web | WebGPU | Chrome 113+, Edge 113+, Firefox (flag) |
| iOS | Metal | iOS 14+, Metal GPU |
| Android | Vulkan | Android 7+, Vulkan 1.1 |

---

## 8. Performance Targets

### 8.1 Throughput Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Event encoding (1M samples) | <1ms | GPU parallel |
| LIF neuron step (10K neurons) | <100μs | GPU parallel |
| Template matching (1K templates) | <1ms | GPU parallel |
| Full inference (multi-modal) | <10ms | End-to-end |

### 8.2 Memory Targets

| Buffer | Size | Residency |
|--------|------|-----------|
| Signal buffer | 4MB max | GPU |
| Event buffer | 1MB max | GPU |
| SNN weights | 10MB max | GPU |
| Templates | 1MB max | GPU |

### 8.3 WASM Size Targets

| Component | Target | Compression |
|-----------|--------|-------------|
| Core WASM | <500KB | Brotli |
| Shaders | <50KB | Included |
| Total bundle | <1MB | Gzipped |

---

## 9. Testing Strategy

### 9.1 Unit Tests (Rust)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_level_crossing_encoder() {
        let signal = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0];
        let encoder = LevelCrossingEncoder::new(0.3);
        let events = encoder.encode(&signal);
        
        assert_eq!(events.len(), 4);  // 2 up, 2 down crossings
    }
    
    #[tokio::test]
    async fn test_gpu_encoder() {
        let gpu = GpuContext::new().await.unwrap();
        let signal = vec![0.0f32; 10000];
        let events = gpu.encode_level_crossing(&signal, 0.1).await;
        
        // Verify against CPU reference
        let cpu_events = cpu_level_crossing(&signal, 0.1);
        assert_eq!(events.len(), cpu_events.len());
    }
}
```

### 9.2 Cross-Language Tests

```python
# tests/python/test_bindings.py
import numpy as np
import dpb_python as dpb

def test_encoder_matches_reference():
    signal = np.sin(np.linspace(0, 10*np.pi, 1000)).astype(np.float32)
    
    encoder = dpb.DPBEncoder(threshold=0.5)
    events = encoder.encode(signal)
    
    # Compare to known reference
    assert events.shape[1] == 4  # timestamp, channel, polarity, magnitude
    assert len(events) > 0
```

### 9.3 GPU vs CPU Validation

```rust
fn validate_gpu_cpu_equivalence() {
    let signal: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.01).sin()).collect();
    
    let cpu_result = cpu_encode(&signal, 0.1);
    let gpu_result = pollster::block_on(gpu_encode(&signal, 0.1));
    
    assert_eq!(cpu_result.len(), gpu_result.len());
    for (cpu, gpu) in cpu_result.iter().zip(gpu_result.iter()) {
        assert!((cpu.timestamp - gpu.timestamp).abs() < 1e-6);
        assert_eq!(cpu.polarity, gpu.polarity);
    }
}
```

---

## 10. Development Roadmap

### Phase 1: Core Infrastructure (Weeks 1-4)
- [ ] Set up Rust workspace structure
- [ ] Implement wgpu context management
- [ ] Basic WGSL shader compilation pipeline
- [ ] CPU reference implementations for validation

### Phase 2: Event Encoders (Weeks 5-8)
- [ ] Level-crossing encoder (GPU)
- [ ] Template deviation encoder (GPU)
- [ ] Derivative encoder (GPU)
- [ ] Multi-channel batch processing

### Phase 3: SNN Kernels (Weeks 9-12)
- [ ] LIF neuron kernel
- [ ] ALIF neuron kernel  
- [ ] Synaptic integration
- [ ] Recurrent connections

### Phase 4: Language Bindings (Weeks 13-16)
- [ ] Python bindings (PyO3)
- [ ] WASM build + JavaScript API
- [ ] Julia bindings
- [ ] MATLAB MEX wrapper

### Phase 5: Integration & Optimization (Weeks 17-20)
- [ ] End-to-end pipeline
- [ ] Performance profiling
- [ ] Memory optimization
- [ ] Documentation

---

## 11. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| GPU API | wgpu | Cross-platform, WebGPU-compatible |
| Shader language | WGSL | Native to wgpu, runs everywhere |
| Python bindings | PyO3 + maturin | Best Rust-Python integration |
| WASM packaging | wasm-pack | Standard tooling |
| Async runtime | tokio (native), wasm-bindgen-futures (web) | Mature ecosystems |
| Math library | glam + nalgebra | SIMD + full linear algebra |
| Testing | nextest | 3× faster than cargo test |

---

*Document: DPB Implementation Architecture v1.0*
*AuraSense Tech Corporation - December 2025*
