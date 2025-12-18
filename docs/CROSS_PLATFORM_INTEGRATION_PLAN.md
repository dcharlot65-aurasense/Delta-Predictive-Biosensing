# Cross-Platform Integration Plan

> **Delta-Predictive Biosensing (DPB) Framework**
>
> Comprehensive implementation plans for R, LabVIEW, WASM, LSL, and ONNX integration.

**Created**: December 2025
**Status**: Implementation Ready

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [R Language Bindings](#2-r-language-bindings)
3. [LabVIEW Integration](#3-labview-integration)
4. [WebAssembly (WASM) Implementation](#4-webassembly-wasm-implementation)
5. [Lab Streaming Layer (LSL) Integration](#5-lab-streaming-layer-lsl-integration)
6. [ONNX Export Capability](#6-onnx-export-capability)
7. [Implementation Timeline](#7-implementation-timeline)
8. [Testing Strategy](#8-testing-strategy)

---

## 1. Executive Summary

### Current Status

| Platform | Status | Mechanism |
|----------|--------|-----------|
| Python | ✅ Complete | PyO3 + maturin |
| Julia | ✅ Complete | C FFI via ccall |
| MATLAB | ✅ Complete | MEX interface |
| C/C++ | ✅ Complete | cbindgen |
| iOS | ⚠️ Partial | C FFI + Swift |
| Android | ⚠️ Partial | JNI + Kotlin |

### Planned Additions

| Platform | Priority | Implementation |
|----------|----------|----------------|
| **R** | High | C FFI via `.Call()` |
| **LabVIEW** | High | Shared library + VI wrappers |
| **WASM** | High | wasm-bindgen + wasm-pack |
| **LSL** | Medium | liblsl integration |
| **ONNX** | Medium | Export format support |

---

## 2. R Language Bindings

### 2.1 Overview

R is essential for biostatistics, clinical research, and the CRAN ecosystem. We'll use R's native C interface via `.Call()` for maximum performance and compatibility.

### 2.2 Directory Structure

```
bindings/r/
├── DESCRIPTION                 # R package metadata
├── NAMESPACE                   # Exported functions
├── R/
│   ├── dpb.R                  # Main package file
│   ├── timeseries.R           # TimeSeries R6 class
│   ├── spiketrain.R           # SpikeTrain R6 class
│   ├── encoders.R             # Encoder classes
│   └── zzz.R                  # Package load/unload hooks
├── src/
│   ├── Makevars               # Unix build config
│   ├── Makevars.win           # Windows build config
│   ├── init.c                 # R native routine registration
│   └── dpb_r.c                # R <-> C FFI bridge
├── inst/
│   └── libs/                  # Pre-built shared libraries
├── man/                       # Documentation (Rd files)
├── tests/
│   └── testthat/              # Unit tests
├── vignettes/                 # Long-form documentation
└── README.md
```

### 2.3 Implementation Details

#### DESCRIPTION File

```
Package: dpb
Type: Package
Title: Delta-Predictive Biosensing Framework
Version: 0.4.0
Authors@R: person("AuraSense", role = c("aut", "cre"), email = "dev@aurasense.tech")
Description: R interface to the Delta-Predictive Biosensing (DPB) framework for
    neuromorphic signal processing, spike encoding, and biosignal analysis.
License: MIT + file LICENSE
Encoding: UTF-8
LazyData: true
Depends: R (>= 4.0.0)
Imports: R6, methods
Suggests: testthat (>= 3.0.0), knitr, rmarkdown
SystemRequirements: Rust (>= 1.70), Cargo
NeedsCompilation: yes
RoxygenNote: 7.2.3
VignetteBuilder: knitr
```

#### R/dpb.R - Main Package File

```r
#' @useDynLib dpb, .registration = TRUE
#' @importFrom R6 R6Class
NULL

#' Get DPB Version
#'
#' @return Character string with version information
#' @export
#' @examples
#' dpb_version()
dpb_version <- function() {
  .Call(C_dpb_version)
}

#' Get Last Error
#'
#' @return Character string with last error message, or NULL
#' @export
dpb_last_error <- function() {
  .Call(C_dpb_last_error)
}

#' Clear Last Error
#' @export
dpb_clear_error <- function() {
  invisible(.Call(C_dpb_clear_error))
}
```

#### R/timeseries.R - TimeSeries Class

```r
#' TimeSeries Class
#'
#' @description
#' R6 class representing multi-channel time series data.
#'
#' @export
#' @examples
#' # Create a sine wave time series
#' t <- seq(0, 1, length.out = 1000)
#' data <- sin(2 * pi * 10 * t)
#' ts <- TimeSeries$new(data, sample_rate = 1000)
#' print(ts$duration)
TimeSeries <- R6::R6Class("TimeSeries",
  private = list(
    ptr = NULL,

    finalize = function() {
      if (!is.null(private$ptr)) {
        .Call(C_dpb_timeseries_free, private$ptr)
        private$ptr <- NULL
      }
    }
  ),

  public = list(
    #' @description Create a new TimeSeries
    #' @param data Numeric vector or matrix (columns = channels)
    #' @param sample_rate Sampling rate in Hz
    initialize = function(data, sample_rate) {
      if (is.vector(data)) {
        data <- matrix(data, ncol = 1)
      }
      stopifnot(is.matrix(data), is.numeric(data))
      stopifnot(sample_rate > 0)

      # Convert to single precision, column-major (R native)
      data <- as.single(data)
      num_samples <- nrow(data)
      num_channels <- ncol(data)

      private$ptr <- .Call(C_dpb_timeseries_new,
                           data, num_samples, num_channels, sample_rate)

      if (is.null(private$ptr)) {
        stop(paste("Failed to create TimeSeries:", dpb_last_error()))
      }
    },

    #' @description Get raw pointer (internal use)
    get_ptr = function() {
      private$ptr
    }
  ),

  active = list(
    #' @field duration Duration in seconds
    duration = function() {
      .Call(C_dpb_timeseries_duration, private$ptr)
    },

    #' @field num_samples Number of samples
    num_samples = function() {
      .Call(C_dpb_timeseries_num_samples, private$ptr)
    },

    #' @field num_channels Number of channels
    num_channels = function() {
      .Call(C_dpb_timeseries_num_channels, private$ptr)
    },

    #' @field sample_rate Sample rate in Hz
    sample_rate = function() {
      .Call(C_dpb_timeseries_sample_rate, private$ptr)
    },

    #' @field data Raw data as matrix
    data = function() {
      .Call(C_dpb_timeseries_get_data, private$ptr)
    }
  )
)
```

#### R/encoders.R - Encoder Classes

```r
#' Level Crossing Encoder
#'
#' @description
#' Encodes continuous signals into spike trains using level crossing detection.
#'
#' @export
LevelCrossingEncoder <- R6::R6Class("LevelCrossingEncoder",
  private = list(
    ptr = NULL,
    threshold = NULL,

    finalize = function() {
      if (!is.null(private$ptr)) {
        .Call(C_dpb_encoder_free, private$ptr)
        private$ptr <- NULL
      }
    }
  ),

  public = list(
    #' @description Create a new Level Crossing Encoder
    #' @param threshold Threshold for level crossing detection
    initialize = function(threshold = 0.1) {
      stopifnot(threshold > 0)
      private$threshold <- threshold
      private$ptr <- .Call(C_dpb_encoder_level_crossing_new, threshold)

      if (is.null(private$ptr)) {
        stop(paste("Failed to create encoder:", dpb_last_error()))
      }
    },

    #' @description Encode a TimeSeries into a SpikeTrain
    #' @param timeseries TimeSeries object to encode
    #' @return SpikeTrain object
    encode = function(timeseries) {
      stopifnot(inherits(timeseries, "TimeSeries"))

      st_ptr <- .Call(C_dpb_encoder_encode, private$ptr, timeseries$get_ptr())

      if (is.null(st_ptr)) {
        stop(paste("Encoding failed:", dpb_last_error()))
      }

      SpikeTrain$new_from_ptr(st_ptr)
    }
  )
)
```

#### src/dpb_r.c - C Bridge

```c
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "dpb.h"

/* Version */
SEXP C_dpb_version(void) {
    return Rf_mkString(dpb_version());
}

/* Error handling */
SEXP C_dpb_last_error(void) {
    const char* err = dpb_last_error();
    return err ? Rf_mkString(err) : R_NilValue;
}

SEXP C_dpb_clear_error(void) {
    dpb_clear_error();
    return R_NilValue;
}

/* TimeSeries */
SEXP C_dpb_timeseries_new(SEXP data, SEXP num_samples, SEXP num_channels, SEXP sample_rate) {
    float* fdata = (float*)REAL(data);  // R stores as double, need conversion
    size_t ns = (size_t)Rf_asInteger(num_samples);
    size_t nc = (size_t)Rf_asInteger(num_channels);
    double sr = Rf_asReal(sample_rate);

    // Allocate and convert double to float
    float* float_data = (float*)R_alloc(ns * nc, sizeof(float));
    for (size_t i = 0; i < ns * nc; i++) {
        float_data[i] = (float)REAL(data)[i];
    }

    DpbTimeSeries* ts = dpb_timeseries_new(float_data, ns, nc, sr);

    if (!ts) {
        return R_NilValue;
    }

    SEXP ptr = R_MakeExternalPtr(ts, R_NilValue, R_NilValue);
    R_RegisterCFinalizerEx(ptr, (R_CFinalizer_t)dpb_timeseries_free, TRUE);
    return ptr;
}

SEXP C_dpb_timeseries_free(SEXP ptr) {
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    if (ts) {
        dpb_timeseries_free(ts);
        R_ClearExternalPtr(ptr);
    }
    return R_NilValue;
}

SEXP C_dpb_timeseries_duration(SEXP ptr) {
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    return Rf_ScalarReal(dpb_timeseries_duration(ts));
}

SEXP C_dpb_timeseries_num_samples(SEXP ptr) {
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    return Rf_ScalarInteger((int)dpb_timeseries_num_samples(ts));
}

SEXP C_dpb_timeseries_num_channels(SEXP ptr) {
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    return Rf_ScalarInteger((int)dpb_timeseries_num_channels(ts));
}

SEXP C_dpb_timeseries_sample_rate(SEXP ptr) {
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    return Rf_ScalarReal(dpb_timeseries_sample_rate(ts));
}

SEXP C_dpb_timeseries_get_data(SEXP ptr) {
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ptr);
    size_t ns = dpb_timeseries_num_samples(ts);
    size_t nc = dpb_timeseries_num_channels(ts);
    const float* data = dpb_timeseries_get_data(ts);

    SEXP result = PROTECT(Rf_allocMatrix(REALSXP, ns, nc));
    double* rdata = REAL(result);

    // Convert float to double
    for (size_t i = 0; i < ns * nc; i++) {
        rdata[i] = (double)data[i];
    }

    UNPROTECT(1);
    return result;
}

/* Encoder */
SEXP C_dpb_encoder_level_crossing_new(SEXP threshold) {
    DpbEncoder* enc = dpb_encoder_level_crossing_new(Rf_asReal(threshold));
    if (!enc) return R_NilValue;

    SEXP ptr = R_MakeExternalPtr(enc, R_NilValue, R_NilValue);
    R_RegisterCFinalizerEx(ptr, (R_CFinalizer_t)dpb_encoder_free, TRUE);
    return ptr;
}

SEXP C_dpb_encoder_free(SEXP ptr) {
    DpbEncoder* enc = (DpbEncoder*)R_ExternalPtrAddr(ptr);
    if (enc) {
        dpb_encoder_free(enc);
        R_ClearExternalPtr(ptr);
    }
    return R_NilValue;
}

SEXP C_dpb_encoder_encode(SEXP enc_ptr, SEXP ts_ptr) {
    DpbEncoder* enc = (DpbEncoder*)R_ExternalPtrAddr(enc_ptr);
    DpbTimeSeries* ts = (DpbTimeSeries*)R_ExternalPtrAddr(ts_ptr);

    DpbSpikeTrain* st = dpb_encoder_encode(enc, ts);
    if (!st) return R_NilValue;

    SEXP ptr = R_MakeExternalPtr(st, R_NilValue, R_NilValue);
    R_RegisterCFinalizerEx(ptr, (R_CFinalizer_t)dpb_spike_train_free, TRUE);
    return ptr;
}

/* Registration */
static const R_CallMethodDef CallEntries[] = {
    {"C_dpb_version", (DL_FUNC) &C_dpb_version, 0},
    {"C_dpb_last_error", (DL_FUNC) &C_dpb_last_error, 0},
    {"C_dpb_clear_error", (DL_FUNC) &C_dpb_clear_error, 0},
    {"C_dpb_timeseries_new", (DL_FUNC) &C_dpb_timeseries_new, 4},
    {"C_dpb_timeseries_free", (DL_FUNC) &C_dpb_timeseries_free, 1},
    {"C_dpb_timeseries_duration", (DL_FUNC) &C_dpb_timeseries_duration, 1},
    {"C_dpb_timeseries_num_samples", (DL_FUNC) &C_dpb_timeseries_num_samples, 1},
    {"C_dpb_timeseries_num_channels", (DL_FUNC) &C_dpb_timeseries_num_channels, 1},
    {"C_dpb_timeseries_sample_rate", (DL_FUNC) &C_dpb_timeseries_sample_rate, 1},
    {"C_dpb_timeseries_get_data", (DL_FUNC) &C_dpb_timeseries_get_data, 1},
    {"C_dpb_encoder_level_crossing_new", (DL_FUNC) &C_dpb_encoder_level_crossing_new, 1},
    {"C_dpb_encoder_free", (DL_FUNC) &C_dpb_encoder_free, 1},
    {"C_dpb_encoder_encode", (DL_FUNC) &C_dpb_encoder_encode, 2},
    {NULL, NULL, 0}
};

void R_init_dpb(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
```

#### src/Makevars

```makefile
PKG_CFLAGS = -I../inst/include
PKG_LIBS = -L../inst/libs -ldpb_ffi

# For macOS
ifeq ($(shell uname -s),Darwin)
PKG_LIBS += -Wl,-rpath,@loader_path/../libs
endif

# For Linux
ifeq ($(shell uname -s),Linux)
PKG_LIBS += -Wl,-rpath,'$$ORIGIN/../libs'
endif
```

### 2.4 Build Process

```bash
# 1. Build Rust FFI library
cd crates/dpb-ffi
cargo build --release

# 2. Copy library to R package
cp target/release/libdpb_ffi.so bindings/r/inst/libs/
cp include/dpb.h bindings/r/inst/include/

# 3. Build R package
cd bindings/r
R CMD build .
R CMD INSTALL dpb_*.tar.gz
```

### 2.5 Usage Example

```r
library(dpb)

# Check version
print(dpb_version())

# Create synthetic ECG-like signal
t <- seq(0, 10, length.out = 10000)
ecg <- sin(2 * pi * 1.2 * t) + 0.3 * sin(2 * pi * 2.4 * t)

# Create TimeSeries
ts <- TimeSeries$new(ecg, sample_rate = 1000)
print(paste("Duration:", ts$duration, "seconds"))
print(paste("Samples:", ts$num_samples))

# Encode to spikes
encoder <- LevelCrossingEncoder$new(threshold = 0.1)
spikes <- encoder$encode(ts)
print(paste("Generated", spikes$length, "spikes"))

# Access spike data
events <- spikes$get_events()
head(events)
```

---

## 3. LabVIEW Integration

### 3.1 Overview

LabVIEW integration enables real-time data acquisition and hardware control for biosignal systems. We'll provide:
1. Call Library Function Node wrappers
2. Pre-built VI library
3. Example applications

### 3.2 Directory Structure

```
bindings/labview/
├── DPB/
│   ├── DPB.lvlib                    # LabVIEW library
│   ├── VIs/
│   │   ├── Core/
│   │   │   ├── DPB_Initialize.vi
│   │   │   ├── DPB_Version.vi
│   │   │   ├── DPB_LastError.vi
│   │   │   └── DPB_Cleanup.vi
│   │   ├── TimeSeries/
│   │   │   ├── DPB_TimeSeries_Create.vi
│   │   │   ├── DPB_TimeSeries_Properties.vi
│   │   │   ├── DPB_TimeSeries_GetData.vi
│   │   │   └── DPB_TimeSeries_Destroy.vi
│   │   ├── SpikeTrain/
│   │   │   ├── DPB_SpikeTrain_Create.vi
│   │   │   ├── DPB_SpikeTrain_AddEvent.vi
│   │   │   ├── DPB_SpikeTrain_GetEvents.vi
│   │   │   └── DPB_SpikeTrain_Destroy.vi
│   │   ├── Encoders/
│   │   │   ├── DPB_Encoder_LevelCrossing.vi
│   │   │   ├── DPB_Encoder_Encode.vi
│   │   │   └── DPB_Encoder_Destroy.vi
│   │   └── Utilities/
│   │       ├── DPB_ErrorHandler.vi
│   │       └── DPB_TypeDefs.ctl
│   ├── SubVIs/                      # Internal helper VIs
│   │   └── DPB_CallLibrary.vi       # Wrapper for CLF Node
│   └── Examples/
│       ├── DPB_BasicUsage.vi
│       ├── DPB_RealTimeEncoding.vi
│       └── DPB_DAQIntegration.vi
├── libs/
│   ├── win64/
│   │   └── dpb_ffi.dll
│   ├── linux64/
│   │   └── libdpb_ffi.so
│   └── mac64/
│       └── libdpb_ffi.dylib
├── docs/
│   ├── DPB_LabVIEW_Manual.pdf
│   └── API_Reference.html
└── README.md
```

### 3.3 VI Specifications

#### DPB_TimeSeries_Create.vi

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| data | In | DBL Array 2D | Signal data (rows=samples, cols=channels) |
| sample rate | In | DBL | Sample rate in Hz |
| TimeSeries Handle | Out | U64 | Opaque handle |
| error out | Out | Error Cluster | Standard error output |

**Block Diagram Logic:**
```
1. Flatten data array to 1D (column-major)
2. Call Library Function Node:
   - Library: dpb_ffi.dll / libdpb_ffi.so
   - Function: dpb_timeseries_new
   - Parameters:
     - data: Pointer to Array of SGL
     - num_samples: U64
     - num_channels: U64
     - sample_rate: DBL
   - Return: Pointer (U64)
3. Check for NULL return, generate error if needed
4. Output handle
```

#### DPB_Encoder_Encode.vi

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| Encoder Handle | In | U64 | Encoder handle |
| TimeSeries Handle | In | U64 | Input time series |
| SpikeTrain Handle | Out | U64 | Output spike train |
| error in/out | In/Out | Error Cluster | Error propagation |

### 3.4 Call Library Function Configuration

For each C function, the Call Library Function Node configuration:

**dpb_timeseries_new:**
```
Library: <platform>/dpb_ffi.[dll|so|dylib]
Function: dpb_timeseries_new
Calling Convention: C
Thread: Run in UI thread (FALSE - runs in any thread)
Return Type: Pointer-sized Integer (U64)
Parameters:
  - const float* data → Pointer to Array of SGL
  - size_t num_samples → Pointer-sized Unsigned Integer
  - size_t num_channels → Pointer-sized Unsigned Integer
  - double sample_rate → 8-byte Double
```

### 3.5 Error Handling Strategy

```
DPB_ErrorHandler.vi Logic:
1. After each CLF Node call, check return value
2. If return is NULL/0 for pointer-returning functions:
   a. Call dpb_last_error() to get message
   b. Convert to LabVIEW error cluster
   c. Set error code based on DpbErrorCode enum
3. Propagate error cluster through VI chain
```

### 3.6 Real-Time Considerations

For LabVIEW Real-Time targets:

```
Deployment Options:
1. NI Linux RT (cRIO, sbRIO)
   - Cross-compile Rust for aarch64-unknown-linux-gnu
   - Deploy .so to /home/lvuser/natinst/lib/

2. Windows RT
   - Standard Windows x64 DLL

3. Timing Considerations
   - DPB functions are non-blocking
   - Use RT FIFO for data transfer
   - Keep encoding in parallel loop
```

### 3.7 DAQ Integration Example

```
DPB_DAQIntegration.vi:
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐     │
│  │ DAQmx   │───▶│ DPB Create  │───▶│ DPB Encode   │───▶ │
│  │ Read    │    │ TimeSeries  │    │              │     │
│  └─────────┘    └─────────────┘    └──────────────┘     │
│       │                                    │             │
│       │              ┌─────────────────────┘             │
│       │              ▼                                   │
│       │         ┌──────────┐    ┌─────────────┐         │
│       │         │ SpikeTrain│───▶│ Visualization│        │
│       │         │ Events   │    │ / Analysis   │        │
│       │         └──────────┘    └─────────────┘         │
│       │                                                  │
│  Loop with 100ms timeout                                │
└──────────────────────────────────────────────────────────┘
```

---

## 4. WebAssembly (WASM) Implementation

### 4.1 Overview

WASM enables browser-based biosignal processing for telehealth dashboards and web applications without server round-trips.

### 4.2 Crate Structure

```
crates/dpb-wasm/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Main WASM module
│   ├── timeseries.rs       # JS-accessible TimeSeries
│   ├── spiketrain.rs       # JS-accessible SpikeTrain
│   ├── encoders.rs         # Encoder bindings
│   ├── decoders.rs         # Decoder bindings
│   ├── utils.rs            # Utility functions
│   └── error.rs            # Error handling
├── tests/
│   └── web.rs              # Browser tests
├── www/                    # Demo application
│   ├── index.html
│   ├── index.js
│   └── package.json
└── README.md
```

### 4.3 Cargo.toml

```toml
[package]
name = "dpb-wasm"
version = "0.4.0"
edition = "2021"
description = "WebAssembly bindings for DPB Framework"
license = "MIT"

[lib]
crate-type = ["cdylib", "rlib"]

[features]
default = ["console_error_panic_hook"]
# Enable WebGPU compute support
webgpu = ["wgpu/webgpu"]

[dependencies]
wasm-bindgen = "0.2.93"
wasm-bindgen-futures = "0.4.43"
js-sys = "0.3.70"
web-sys = { version = "0.3.70", features = [
    "console",
    "Performance",
    "Window",
    "Document",
    "HtmlCanvasElement",
    "WebGl2RenderingContext",
    "GpuDevice",
    "GpuAdapter",
    "GpuBuffer",
    "GpuBufferDescriptor",
    "GpuCommandEncoder",
    "GpuComputePipeline",
    "GpuShaderModule",
]}
console_error_panic_hook = { version = "0.1.7", optional = true }
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6"

# Core DPB dependencies (no-std compatible portions)
dpb-core = { path = "../dpb-core", default-features = false }
dpb-snn = { path = "../dpb-snn", default-features = false }

[dev-dependencies]
wasm-bindgen-test = "0.3.43"

[profile.release]
opt-level = "z"        # Optimize for size
lto = true             # Link-time optimization
codegen-units = 1      # Better optimization
panic = "abort"        # Smaller code
strip = true           # Strip symbols
```

### 4.4 src/lib.rs

```rust
use wasm_bindgen::prelude::*;

mod timeseries;
mod spiketrain;
mod encoders;
mod error;
mod utils;

pub use timeseries::*;
pub use spiketrain::*;
pub use encoders::*;

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Set up panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get DPB version
#[wasm_bindgen]
pub fn version() -> String {
    format!("DPB WASM v{}", env!("CARGO_PKG_VERSION"))
}

/// Performance measurement
#[wasm_bindgen]
pub struct PerformanceTimer {
    start: f64,
}

#[wasm_bindgen]
impl PerformanceTimer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let window = web_sys::window().expect("no window");
        let performance = window.performance().expect("no performance");
        Self {
            start: performance.now(),
        }
    }

    #[wasm_bindgen]
    pub fn elapsed_ms(&self) -> f64 {
        let window = web_sys::window().expect("no window");
        let performance = window.performance().expect("no performance");
        performance.now() - self.start
    }
}
```

### 4.5 src/timeseries.rs

```rust
use wasm_bindgen::prelude::*;
use js_sys::Float32Array;

/// TimeSeries for browser use
#[wasm_bindgen]
pub struct WasmTimeSeries {
    data: Vec<f32>,
    num_samples: usize,
    num_channels: usize,
    sample_rate: f64,
}

#[wasm_bindgen]
impl WasmTimeSeries {
    /// Create from JavaScript Float32Array
    #[wasm_bindgen(constructor)]
    pub fn new(
        data: Float32Array,
        num_samples: usize,
        num_channels: usize,
        sample_rate: f64,
    ) -> Result<WasmTimeSeries, JsValue> {
        if data.length() as usize != num_samples * num_channels {
            return Err(JsValue::from_str("Data length mismatch"));
        }

        Ok(Self {
            data: data.to_vec(),
            num_samples,
            num_channels,
            sample_rate,
        })
    }

    /// Create from 2D array (channels x samples)
    #[wasm_bindgen(js_name = "fromChannels")]
    pub fn from_channels(
        channels: Vec<Float32Array>,
        sample_rate: f64,
    ) -> Result<WasmTimeSeries, JsValue> {
        if channels.is_empty() {
            return Err(JsValue::from_str("No channels provided"));
        }

        let num_channels = channels.len();
        let num_samples = channels[0].length() as usize;

        // Interleave channels
        let mut data = vec![0.0f32; num_samples * num_channels];
        for (ch_idx, channel) in channels.iter().enumerate() {
            if channel.length() as usize != num_samples {
                return Err(JsValue::from_str("Channel length mismatch"));
            }
            let ch_data = channel.to_vec();
            for (s_idx, &value) in ch_data.iter().enumerate() {
                data[s_idx * num_channels + ch_idx] = value;
            }
        }

        Ok(Self {
            data,
            num_samples,
            num_channels,
            sample_rate,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f64 {
        self.num_samples as f64 / self.sample_rate
    }

    #[wasm_bindgen(getter, js_name = "numSamples")]
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    #[wasm_bindgen(getter, js_name = "numChannels")]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    #[wasm_bindgen(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get data as Float32Array (zero-copy when possible)
    #[wasm_bindgen(js_name = "getData")]
    pub fn get_data(&self) -> Float32Array {
        Float32Array::from(&self.data[..])
    }

    /// Get single channel
    #[wasm_bindgen(js_name = "getChannel")]
    pub fn get_channel(&self, channel: usize) -> Result<Float32Array, JsValue> {
        if channel >= self.num_channels {
            return Err(JsValue::from_str("Channel index out of bounds"));
        }

        let mut channel_data = Vec::with_capacity(self.num_samples);
        for s in 0..self.num_samples {
            channel_data.push(self.data[s * self.num_channels + channel]);
        }

        Ok(Float32Array::from(&channel_data[..]))
    }

    /// Internal: get reference to data for encoding
    pub(crate) fn data_ref(&self) -> &[f32] {
        &self.data
    }
}
```

### 4.6 src/encoders.rs

```rust
use wasm_bindgen::prelude::*;
use crate::timeseries::WasmTimeSeries;
use crate::spiketrain::WasmSpikeTrain;

/// Level Crossing Encoder
#[wasm_bindgen]
pub struct WasmLevelCrossingEncoder {
    threshold: f64,
}

#[wasm_bindgen]
impl WasmLevelCrossingEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f64) -> Result<WasmLevelCrossingEncoder, JsValue> {
        if threshold <= 0.0 {
            return Err(JsValue::from_str("Threshold must be positive"));
        }
        Ok(Self { threshold })
    }

    #[wasm_bindgen]
    pub fn encode(&self, ts: &WasmTimeSeries) -> Result<WasmSpikeTrain, JsValue> {
        let data = ts.data_ref();
        let num_samples = ts.num_samples();
        let num_channels = ts.num_channels();
        let sample_rate = ts.sample_rate();

        let mut spike_train = WasmSpikeTrain::new(num_channels as u32)?;

        // Track last value per channel for level crossing detection
        let mut last_values = vec![0.0f32; num_channels];
        let mut last_levels = vec![0i32; num_channels];

        for s in 0..num_samples {
            let time = s as f64 / sample_rate;

            for ch in 0..num_channels {
                let value = data[s * num_channels + ch];
                let level = (value as f64 / self.threshold).floor() as i32;

                if s > 0 && level != last_levels[ch] {
                    let polarity = if level > last_levels[ch] { 1i8 } else { -1i8 };
                    spike_train.add_event(time, ch as u32, polarity)?;
                }

                last_values[ch] = value;
                last_levels[ch] = level;
            }
        }

        Ok(spike_train)
    }
}

/// Delta Encoder
#[wasm_bindgen]
pub struct WasmDeltaEncoder {
    threshold: f64,
}

#[wasm_bindgen]
impl WasmDeltaEncoder {
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f64) -> Result<WasmDeltaEncoder, JsValue> {
        if threshold <= 0.0 {
            return Err(JsValue::from_str("Threshold must be positive"));
        }
        Ok(Self { threshold })
    }

    #[wasm_bindgen]
    pub fn encode(&self, ts: &WasmTimeSeries) -> Result<WasmSpikeTrain, JsValue> {
        let data = ts.data_ref();
        let num_samples = ts.num_samples();
        let num_channels = ts.num_channels();
        let sample_rate = ts.sample_rate();

        let mut spike_train = WasmSpikeTrain::new(num_channels as u32)?;
        let mut reference = vec![0.0f32; num_channels];

        for s in 0..num_samples {
            let time = s as f64 / sample_rate;

            for ch in 0..num_channels {
                let value = data[s * num_channels + ch];
                let delta = value - reference[ch];

                if delta.abs() as f64 >= self.threshold {
                    let polarity = if delta > 0.0 { 1i8 } else { -1i8 };
                    spike_train.add_event(time, ch as u32, polarity)?;
                    reference[ch] = value;
                }
            }
        }

        Ok(spike_train)
    }
}
```

### 4.7 www/index.html (Demo)

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DPB WASM Demo</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; }
        canvas { border: 1px solid #ccc; margin: 10px 0; }
        #output { background: #f5f5f5; padding: 15px; font-family: monospace; }
        button { padding: 10px 20px; margin: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>DPB WebAssembly Demo</h1>

    <div>
        <button id="runDemo">Run Spike Encoding Demo</button>
        <button id="benchmarkBtn">Run Benchmark</button>
    </div>

    <canvas id="signalCanvas" width="800" height="200"></canvas>
    <canvas id="spikeCanvas" width="800" height="100"></canvas>

    <div id="output"></div>

    <script type="module">
        import init, {
            version,
            WasmTimeSeries,
            WasmLevelCrossingEncoder,
            PerformanceTimer
        } from './pkg/dpb_wasm.js';

        async function main() {
            await init();

            document.getElementById('output').textContent = version();

            document.getElementById('runDemo').onclick = runDemo;
            document.getElementById('benchmarkBtn').onclick = runBenchmark;
        }

        function runDemo() {
            const output = document.getElementById('output');

            // Generate synthetic ECG-like signal
            const sampleRate = 1000;
            const duration = 5; // seconds
            const numSamples = sampleRate * duration;

            const data = new Float32Array(numSamples);
            for (let i = 0; i < numSamples; i++) {
                const t = i / sampleRate;
                data[i] = Math.sin(2 * Math.PI * 1.2 * t) +
                          0.3 * Math.sin(2 * Math.PI * 2.4 * t) +
                          0.1 * Math.sin(2 * Math.PI * 60 * t); // 60Hz noise
            }

            // Create TimeSeries
            const ts = new WasmTimeSeries(data, numSamples, 1, sampleRate);

            // Create encoder
            const encoder = new WasmLevelCrossingEncoder(0.1);

            // Encode
            const timer = new PerformanceTimer();
            const spikes = encoder.encode(ts);
            const elapsed = timer.elapsed_ms();

            output.textContent = `
Version: ${version()}
Duration: ${ts.duration.toFixed(2)}s
Samples: ${ts.numSamples}
Spikes generated: ${spikes.length}
Encoding time: ${elapsed.toFixed(2)}ms
Throughput: ${(numSamples / elapsed * 1000).toFixed(0)} samples/sec
            `;

            // Draw signal
            drawSignal(data, 'signalCanvas');

            // Draw spikes
            drawSpikes(spikes, ts.duration, 'spikeCanvas');
        }

        function runBenchmark() {
            const output = document.getElementById('output');
            const results = [];

            for (const size of [1000, 10000, 100000, 1000000]) {
                const data = new Float32Array(size);
                for (let i = 0; i < size; i++) {
                    data[i] = Math.sin(i * 0.01);
                }

                const ts = new WasmTimeSeries(data, size, 1, 1000);
                const encoder = new WasmLevelCrossingEncoder(0.1);

                const timer = new PerformanceTimer();
                const spikes = encoder.encode(ts);
                const elapsed = timer.elapsed_ms();

                results.push(`${size} samples: ${elapsed.toFixed(2)}ms (${spikes.length} spikes)`);
            }

            output.textContent = 'Benchmark Results:\n' + results.join('\n');
        }

        function drawSignal(data, canvasId) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            ctx.strokeStyle = '#2196F3';
            ctx.beginPath();

            const step = Math.max(1, Math.floor(data.length / canvas.width));
            for (let i = 0; i < data.length; i += step) {
                const x = (i / data.length) * canvas.width;
                const y = canvas.height / 2 - data[i] * 50;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        function drawSpikes(spikes, duration, canvasId) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const events = spikes.getEvents();
            for (const event of events) {
                const x = (event.timestamp / duration) * canvas.width;
                ctx.strokeStyle = event.polarity > 0 ? '#4CAF50' : '#F44336';
                ctx.beginPath();
                ctx.moveTo(x, canvas.height);
                ctx.lineTo(x, event.polarity > 0 ? 20 : canvas.height - 20);
                ctx.stroke();
            }
        }

        main();
    </script>
</body>
</html>
```

### 4.8 Build Commands

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for web
cd crates/dpb-wasm
wasm-pack build --target web --release

# Build for Node.js
wasm-pack build --target nodejs --release

# Build for bundlers (webpack, etc.)
wasm-pack build --target bundler --release

# Run demo server
cd www
python -m http.server 8080
```

---

## 5. Lab Streaming Layer (LSL) Integration

### 5.1 Overview

LSL is the de facto standard for real-time streaming of biosignals in research environments. Integration enables:
- Multi-device synchronization
- Network-transparent data transfer
- Sub-millisecond timing precision

### 5.2 Crate Structure

```
crates/dpb-lsl/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Main module
│   ├── inlet.rs            # LSL inlet (consumer)
│   ├── outlet.rs           # LSL outlet (producer)
│   ├── stream_info.rs      # Stream metadata
│   ├── resolver.rs         # Stream discovery
│   └── dpb_bridge.rs       # DPB <-> LSL conversion
├── examples/
│   ├── receive_eeg.rs
│   ├── send_spikes.rs
│   └── realtime_encoding.rs
└── README.md
```

### 5.3 Cargo.toml

```toml
[package]
name = "dpb-lsl"
version = "0.4.0"
edition = "2021"
description = "LSL integration for real-time biosignal streaming"
license = "MIT"

[dependencies]
# LSL bindings
lsl-sys = "0.2"  # Raw FFI bindings to liblsl

# DPB core
dpb-core = { path = "../dpb-core" }
dpb-snn = { path = "../dpb-snn" }

# Async runtime
tokio = { version = "1.0", features = ["rt-multi-thread", "sync", "time"] }

# Utilities
thiserror = "1.0"
tracing = "0.1"
crossbeam-channel = "0.5"

[build-dependencies]
# For finding liblsl
pkg-config = "0.3"

[dev-dependencies]
tokio-test = "0.4"
```

### 5.4 src/lib.rs

```rust
//! # dpb-lsl
//!
//! Lab Streaming Layer integration for real-time biosignal processing.
//!
//! ## Features
//!
//! - Receive biosignals from LSL streams
//! - Send spike trains over LSL
//! - Real-time encoding pipeline
//! - Multi-stream synchronization

pub mod inlet;
pub mod outlet;
pub mod stream_info;
pub mod resolver;
pub mod dpb_bridge;

pub use inlet::DpbLslInlet;
pub use outlet::DpbLslOutlet;
pub use stream_info::DpbStreamInfo;
pub use resolver::StreamResolver;
pub use dpb_bridge::RealtimeEncoder;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum LslError {
    #[error("LSL stream not found: {0}")]
    StreamNotFound(String),

    #[error("Connection timeout after {0}ms")]
    Timeout(u64),

    #[error("Channel mismatch: expected {expected}, got {actual}")]
    ChannelMismatch { expected: usize, actual: usize },

    #[error("LSL error: {0}")]
    LslInternal(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),
}

pub type Result<T> = std::result::Result<T, LslError>;
```

### 5.5 src/inlet.rs

```rust
use crate::{LslError, Result};
use dpb_core::TimeSeries;
use std::sync::Arc;
use tokio::sync::mpsc;

/// LSL inlet for receiving biosignal streams
pub struct DpbLslInlet {
    inlet: lsl_sys::Inlet,
    stream_info: Arc<crate::DpbStreamInfo>,
    buffer: Vec<f32>,
}

impl DpbLslInlet {
    /// Connect to an LSL stream by name
    pub fn connect(stream_name: &str, timeout_sec: f64) -> Result<Self> {
        // Resolve stream
        let streams = lsl_sys::resolve_stream("name", stream_name, 1, timeout_sec);

        if streams.is_empty() {
            return Err(LslError::StreamNotFound(stream_name.to_string()));
        }

        let info = &streams[0];
        let inlet = lsl_sys::Inlet::new(info, 360, 0, true)?;

        let channel_count = info.channel_count() as usize;

        Ok(Self {
            inlet,
            stream_info: Arc::new(crate::DpbStreamInfo::from_lsl(info)),
            buffer: vec![0.0; channel_count],
        })
    }

    /// Pull a single sample with timestamp
    pub fn pull_sample(&mut self, timeout: f64) -> Result<(Vec<f32>, f64)> {
        let timestamp = self.inlet.pull_sample(&mut self.buffer, timeout)?;
        Ok((self.buffer.clone(), timestamp))
    }

    /// Pull a chunk of samples
    pub fn pull_chunk(
        &mut self,
        max_samples: usize,
        timeout: f64,
    ) -> Result<(Vec<Vec<f32>>, Vec<f64>)> {
        let mut samples = Vec::with_capacity(max_samples);
        let mut timestamps = Vec::with_capacity(max_samples);

        let mut buf = vec![0.0f32; self.stream_info.channel_count * max_samples];
        let mut ts_buf = vec![0.0f64; max_samples];

        let pulled = self.inlet.pull_chunk(&mut buf, &mut ts_buf, timeout)?;

        for i in 0..pulled {
            let start = i * self.stream_info.channel_count;
            let end = start + self.stream_info.channel_count;
            samples.push(buf[start..end].to_vec());
            timestamps.push(ts_buf[i]);
        }

        Ok((samples, timestamps))
    }

    /// Create async channel for continuous reception
    pub fn into_async_receiver(
        mut self,
        buffer_size: usize,
    ) -> mpsc::Receiver<(Vec<f32>, f64)> {
        let (tx, rx) = mpsc::channel(buffer_size);

        tokio::spawn(async move {
            loop {
                match self.pull_sample(1.0) {
                    Ok(sample) => {
                        if tx.send(sample).await.is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(_) => continue, // Timeout, retry
                }
            }
        });

        rx
    }

    /// Get stream info
    pub fn stream_info(&self) -> &crate::DpbStreamInfo {
        &self.stream_info
    }

    /// Get time correction offset
    pub fn time_correction(&self) -> f64 {
        self.inlet.time_correction(1.0).unwrap_or(0.0)
    }
}
```

### 5.6 src/outlet.rs

```rust
use crate::{LslError, Result};
use dpb_core::SpikeTrain;

/// LSL outlet for sending spike trains
pub struct DpbLslOutlet {
    outlet: lsl_sys::Outlet,
    num_channels: usize,
}

impl DpbLslOutlet {
    /// Create outlet for spike data
    pub fn new_spike_stream(
        name: &str,
        num_channels: usize,
        source_id: &str,
    ) -> Result<Self> {
        // Create stream info for spike data
        // Format: [timestamp, channel, polarity, magnitude]
        let info = lsl_sys::StreamInfo::new(
            name,
            "Spikes",
            4, // timestamp, channel, polarity, magnitude
            0.0, // Irregular rate
            lsl_sys::ChannelFormat::Float32,
            source_id,
        )?;

        // Add channel descriptions
        let channels = info.desc().append_child("channels");
        channels.append_child("channel").set_name("timestamp");
        channels.append_child("channel").set_name("channel");
        channels.append_child("channel").set_name("polarity");
        channels.append_child("channel").set_name("magnitude");

        let outlet = lsl_sys::Outlet::new(&info, 0, 360)?;

        Ok(Self {
            outlet,
            num_channels,
        })
    }

    /// Create outlet for continuous biosignal data
    pub fn new_signal_stream(
        name: &str,
        stream_type: &str,
        num_channels: usize,
        sample_rate: f64,
        source_id: &str,
    ) -> Result<Self> {
        let info = lsl_sys::StreamInfo::new(
            name,
            stream_type,
            num_channels as i32,
            sample_rate,
            lsl_sys::ChannelFormat::Float32,
            source_id,
        )?;

        let outlet = lsl_sys::Outlet::new(&info, 0, 360)?;

        Ok(Self {
            outlet,
            num_channels,
        })
    }

    /// Push a single spike event
    pub fn push_spike(
        &self,
        timestamp: f64,
        channel: u32,
        polarity: i8,
        magnitude: f32,
    ) -> Result<()> {
        let sample = [
            timestamp as f32,
            channel as f32,
            polarity as f32,
            magnitude,
        ];
        self.outlet.push_sample(&sample)?;
        Ok(())
    }

    /// Push entire spike train
    pub fn push_spike_train(&self, train: &SpikeTrain) -> Result<()> {
        for event in train.events() {
            self.push_spike(
                event.timestamp,
                event.channel,
                event.polarity,
                event.magnitude,
            )?;
        }
        Ok(())
    }

    /// Push continuous signal sample
    pub fn push_sample(&self, data: &[f32]) -> Result<()> {
        if data.len() != self.num_channels {
            return Err(LslError::ChannelMismatch {
                expected: self.num_channels,
                actual: data.len(),
            });
        }
        self.outlet.push_sample(data)?;
        Ok(())
    }

    /// Push chunk of samples
    pub fn push_chunk(&self, data: &[Vec<f32>]) -> Result<()> {
        for sample in data {
            self.push_sample(sample)?;
        }
        Ok(())
    }

    /// Check if there are consumers
    pub fn have_consumers(&self) -> bool {
        self.outlet.have_consumers()
    }

    /// Wait for consumers
    pub fn wait_for_consumers(&self, timeout: f64) -> bool {
        self.outlet.wait_for_consumers(timeout)
    }
}
```

### 5.7 src/dpb_bridge.rs

```rust
use crate::{DpbLslInlet, DpbLslOutlet, Result};
use dpb_core::{TimeSeries, SpikeTrain};
use dpb_snn::encoders::{Encoder, LevelCrossingEncoder};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Real-time LSL to spike encoding pipeline
pub struct RealtimeEncoder {
    inlet: DpbLslInlet,
    outlet: DpbLslOutlet,
    encoder: Arc<dyn Encoder + Send + Sync>,
    buffer_size: usize,
}

impl RealtimeEncoder {
    pub fn new(
        input_stream: &str,
        output_stream: &str,
        encoder: Arc<dyn Encoder + Send + Sync>,
        buffer_size: usize,
    ) -> Result<Self> {
        let inlet = DpbLslInlet::connect(input_stream, 5.0)?;
        let outlet = DpbLslOutlet::new_spike_stream(
            output_stream,
            inlet.stream_info().channel_count,
            &format!("dpb-encoder-{}", input_stream),
        )?;

        Ok(Self {
            inlet,
            outlet,
            encoder,
            buffer_size,
        })
    }

    /// Run the encoding pipeline
    pub async fn run(&mut self) -> Result<()> {
        let (tx, mut rx) = mpsc::channel::<SpikeTrain>(32);

        let sample_rate = self.inlet.stream_info().sample_rate;
        let num_channels = self.inlet.stream_info().channel_count;

        // Accumulation buffer
        let mut buffer = Vec::with_capacity(self.buffer_size * num_channels);
        let mut sample_count = 0;

        loop {
            // Pull samples
            match self.inlet.pull_sample(0.1) {
                Ok((sample, _timestamp)) => {
                    buffer.extend_from_slice(&sample);
                    sample_count += 1;

                    // Process when buffer is full
                    if sample_count >= self.buffer_size {
                        // Create TimeSeries from buffer
                        let ts = TimeSeries::new(
                            buffer.clone(),
                            sample_count,
                            num_channels,
                            sample_rate,
                        );

                        // Encode
                        let spikes = self.encoder.encode(&ts)?;

                        // Send spikes
                        self.outlet.push_spike_train(&spikes)?;

                        // Clear buffer
                        buffer.clear();
                        sample_count = 0;
                    }
                }
                Err(_) => continue, // Timeout
            }
        }
    }

    /// Run with statistics reporting
    pub async fn run_with_stats(
        &mut self,
        stats_tx: mpsc::Sender<EncodingStats>,
    ) -> Result<()> {
        let mut total_samples = 0u64;
        let mut total_spikes = 0u64;
        let start = std::time::Instant::now();

        // ... similar to run() but with stats collection

        Ok(())
    }
}

/// Encoding statistics
#[derive(Debug, Clone)]
pub struct EncodingStats {
    pub samples_processed: u64,
    pub spikes_generated: u64,
    pub elapsed_seconds: f64,
    pub samples_per_second: f64,
    pub compression_ratio: f64,
}

/// Convenience function to create a default level-crossing pipeline
pub fn create_level_crossing_pipeline(
    input_stream: &str,
    output_stream: &str,
    threshold: f64,
    buffer_ms: f64,
) -> Result<RealtimeEncoder> {
    let encoder = Arc::new(LevelCrossingEncoder::new(threshold));

    // Connect first to get sample rate
    let inlet = DpbLslInlet::connect(input_stream, 5.0)?;
    let sample_rate = inlet.stream_info().sample_rate;
    let buffer_size = (sample_rate * buffer_ms / 1000.0) as usize;

    drop(inlet); // Will reconnect in RealtimeEncoder::new

    RealtimeEncoder::new(input_stream, output_stream, encoder, buffer_size)
}
```

### 5.8 Example: Real-time EEG Encoding

```rust
// examples/realtime_encoding.rs
use dpb_lsl::{create_level_crossing_pipeline, RealtimeEncoder};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Find EEG stream and encode in real-time
    let mut pipeline = create_level_crossing_pipeline(
        "MyEEGStream",      // Input LSL stream name
        "MyEEGSpikes",      // Output LSL stream name
        0.05,               // Threshold (50 µV for EEG)
        100.0,              // Buffer size in ms
    )?;

    println!("Starting real-time encoding pipeline...");
    println!("Input: MyEEGStream -> Output: MyEEGSpikes");

    pipeline.run().await?;

    Ok(())
}
```

---

## 6. ONNX Export Capability

### 6.1 Overview

ONNX (Open Neural Network Exchange) enables deploying trained SNN models to any ONNX-compatible runtime (TensorRT, ONNX Runtime, CoreML, etc.).

### 6.2 Module Structure

```
crates/dpb-export/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── onnx/
│   │   ├── mod.rs
│   │   ├── builder.rs      # ONNX graph builder
│   │   ├── operators.rs    # Custom SNN operators
│   │   ├── converter.rs    # DPB model -> ONNX
│   │   └── validate.rs     # Model validation
│   ├── tflite/
│   │   ├── mod.rs
│   │   └── converter.rs
│   └── neuromorphic/
│       ├── mod.rs
│       ├── loihi.rs
│       └── spinnaker.rs
└── README.md
```

### 6.3 Cargo.toml

```toml
[package]
name = "dpb-export"
version = "0.4.0"
edition = "2021"
description = "Model export for DPB Framework (ONNX, TFLite, Neuromorphic)"
license = "MIT"

[features]
default = ["onnx"]
onnx = ["prost", "prost-build"]
tflite = ["flatbuffers"]
neuromorphic = []

[dependencies]
dpb-core = { path = "../dpb-core" }
dpb-snn = { path = "../dpb-snn" }

# ONNX support
prost = { version = "0.12", optional = true }
prost-types = { version = "0.12", optional = true }

# TFLite support
flatbuffers = { version = "23.5", optional = true }

# Common
thiserror = "1.0"
uuid = { version = "1.0", features = ["v4"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[build-dependencies]
prost-build = { version = "0.12", optional = true }
```

### 6.4 src/onnx/converter.rs

```rust
use crate::onnx::builder::{OnnxGraphBuilder, TensorProto, NodeProto};
use dpb_snn::network::SnnNetwork;
use dpb_snn::neurons::{NeuronModel, LIFNeuron};
use std::collections::HashMap;

/// Convert DPB SNN to ONNX format
pub struct OnnxConverter {
    builder: OnnxGraphBuilder,
    opset_version: i64,
}

impl OnnxConverter {
    pub fn new() -> Self {
        Self {
            builder: OnnxGraphBuilder::new(),
            opset_version: 17,
        }
    }

    /// Convert SNN network to ONNX model
    pub fn convert(&mut self, network: &SnnNetwork) -> Result<Vec<u8>, ExportError> {
        // Clear any previous state
        self.builder.clear();

        // Set model metadata
        self.builder.set_producer("DPB Framework");
        self.builder.set_domain("ai.aurasense.dpb");

        // Add input tensor
        let input_shape = vec![1, network.input_size() as i64]; // [batch, features]
        self.builder.add_input("input", &input_shape, TensorType::Float);

        // Convert each layer
        let mut prev_output = "input".to_string();

        for (layer_idx, layer) in network.layers().enumerate() {
            let layer_name = format!("layer_{}", layer_idx);

            match layer.neuron_type() {
                NeuronModel::LIF(params) => {
                    prev_output = self.convert_lif_layer(
                        &prev_output,
                        &layer_name,
                        layer,
                        params,
                    )?;
                }
                NeuronModel::Izhikevich(params) => {
                    prev_output = self.convert_izhikevich_layer(
                        &prev_output,
                        &layer_name,
                        layer,
                        params,
                    )?;
                }
                _ => {
                    return Err(ExportError::UnsupportedNeuron(
                        format!("{:?}", layer.neuron_type())
                    ));
                }
            }
        }

        // Add output
        self.builder.add_output(&prev_output, TensorType::Float);

        // Build and serialize
        let model = self.builder.build(self.opset_version)?;
        Ok(model.encode_to_vec())
    }

    fn convert_lif_layer(
        &mut self,
        input: &str,
        name: &str,
        layer: &dyn Layer,
        params: &LIFParams,
    ) -> Result<String, ExportError> {
        // LIF can be approximated with standard ONNX ops:
        // v[t] = (1 - dt/tau) * v[t-1] + (dt/tau) * (W @ x[t] + b)
        // spike = v >= threshold
        // v = v * (1 - spike) + v_reset * spike

        let weights = layer.weights();
        let biases = layer.biases();

        // Add weights as initializer
        let w_name = format!("{}_weight", name);
        self.builder.add_initializer(&w_name, weights);

        // Add bias as initializer
        let b_name = format!("{}_bias", name);
        self.builder.add_initializer(&b_name, biases);

        // MatMul: y = W @ x
        let matmul_out = format!("{}_matmul", name);
        self.builder.add_node(NodeProto {
            op_type: "MatMul".to_string(),
            inputs: vec![input.to_string(), w_name],
            outputs: vec![matmul_out.clone()],
            ..Default::default()
        });

        // Add bias: y = y + b
        let add_out = format!("{}_add", name);
        self.builder.add_node(NodeProto {
            op_type: "Add".to_string(),
            inputs: vec![matmul_out, b_name],
            outputs: vec![add_out.clone()],
            ..Default::default()
        });

        // For simplified inference, use ReLU as spike approximation
        // (Full temporal dynamics require custom ops or recurrent structure)
        let relu_out = format!("{}_out", name);
        self.builder.add_node(NodeProto {
            op_type: "Relu".to_string(),
            inputs: vec![add_out],
            outputs: vec![relu_out.clone()],
            ..Default::default()
        });

        Ok(relu_out)
    }

    fn convert_izhikevich_layer(
        &mut self,
        input: &str,
        name: &str,
        layer: &dyn Layer,
        params: &IzhikevichParams,
    ) -> Result<String, ExportError> {
        // Similar structure but with different activation
        // Izhikevich dynamics are complex; use surrogate for inference

        // ... implementation similar to LIF

        Ok(format!("{}_out", name))
    }
}

/// Export trained model to ONNX file
pub fn export_to_onnx(
    network: &SnnNetwork,
    path: &std::path::Path,
) -> Result<(), ExportError> {
    let mut converter = OnnxConverter::new();
    let bytes = converter.convert(network)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Export with custom opset version
pub fn export_to_onnx_with_opset(
    network: &SnnNetwork,
    path: &std::path::Path,
    opset_version: i64,
) -> Result<(), ExportError> {
    let mut converter = OnnxConverter::new();
    converter.opset_version = opset_version;
    let bytes = converter.convert(network)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
```

### 6.5 Custom ONNX Operators for SNNs

```rust
// src/onnx/operators.rs

/// Register custom SNN operators in ONNX
pub fn register_custom_ops() -> HashMap<String, CustomOpDef> {
    let mut ops = HashMap::new();

    // LIF Neuron operator
    ops.insert("dpb.LIFNeuron".to_string(), CustomOpDef {
        domain: "ai.aurasense.dpb",
        version: 1,
        inputs: vec![
            ("input", TensorType::Float),      // Input current
            ("state", TensorType::Float),      // Membrane potential
        ],
        outputs: vec![
            ("spikes", TensorType::Float),     // Output spikes
            ("new_state", TensorType::Float),  // Updated state
        ],
        attributes: vec![
            ("tau", AttrType::Float),          // Time constant
            ("threshold", AttrType::Float),    // Spike threshold
            ("reset", AttrType::Float),        // Reset potential
            ("dt", AttrType::Float),           // Time step
        ],
    });

    // STDP Learning operator
    ops.insert("dpb.STDPUpdate".to_string(), CustomOpDef {
        domain: "ai.aurasense.dpb",
        version: 1,
        inputs: vec![
            ("weights", TensorType::Float),
            ("pre_spikes", TensorType::Float),
            ("post_spikes", TensorType::Float),
        ],
        outputs: vec![
            ("new_weights", TensorType::Float),
        ],
        attributes: vec![
            ("a_plus", AttrType::Float),
            ("a_minus", AttrType::Float),
            ("tau_plus", AttrType::Float),
            ("tau_minus", AttrType::Float),
        ],
    });

    ops
}
```

### 6.6 Usage Example

```rust
use dpb_snn::network::SnnNetworkBuilder;
use dpb_export::onnx::export_to_onnx;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build network
    let network = SnnNetworkBuilder::new()
        .input_size(128)
        .add_lif_layer(64, Default::default())
        .add_lif_layer(32, Default::default())
        .add_output_layer(10)
        .build()?;

    // Train network...

    // Export to ONNX
    export_to_onnx(&network, Path::new("model.onnx"))?;

    println!("Exported model to model.onnx");

    // Validate with ONNX Runtime
    // ort::Session::new("model.onnx")?;

    Ok(())
}
```

---

## 7. Implementation Timeline

### Phase 1: Core Bindings

| Task | Estimated Effort | Dependencies |
|------|------------------|--------------|
| R bindings implementation | 3-4 days | dpb-ffi complete |
| R package testing | 1-2 days | R bindings |
| LabVIEW VI library | 4-5 days | dpb-ffi complete |
| LabVIEW examples | 2 days | VI library |

### Phase 2: Web & Streaming

| Task | Estimated Effort | Dependencies |
|------|------------------|--------------|
| WASM crate implementation | 3-4 days | dpb-core, dpb-snn |
| WASM demo application | 1-2 days | WASM crate |
| LSL integration crate | 4-5 days | liblsl available |
| Real-time pipeline | 2-3 days | LSL crate |

### Phase 3: Export Formats

| Task | Estimated Effort | Dependencies |
|------|------------------|--------------|
| ONNX converter | 4-5 days | dpb-snn |
| Custom ONNX operators | 2-3 days | ONNX converter |
| TFLite export | 3-4 days | Optional |
| Neuromorphic export | 2-3 days | Existing |

---

## 8. Testing Strategy

### 8.1 R Package Tests

```r
# tests/testthat/test-timeseries.R
test_that("TimeSeries creation works", {
  data <- sin(seq(0, 2*pi, length.out = 1000))
  ts <- TimeSeries$new(data, sample_rate = 1000)

  expect_equal(ts$num_samples, 1000)
  expect_equal(ts$sample_rate, 1000)
  expect_equal(ts$duration, 1.0, tolerance = 0.001)
})

test_that("Level crossing encoding works", {
  data <- sin(seq(0, 10*pi, length.out = 10000))
  ts <- TimeSeries$new(data, sample_rate = 1000)
  encoder <- LevelCrossingEncoder$new(threshold = 0.1)
  spikes <- encoder$encode(ts)

  expect_gt(spikes$length, 0)
})
```

### 8.2 WASM Tests

```rust
// tests/web.rs
use wasm_bindgen_test::*;
use dpb_wasm::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_timeseries_creation() {
    let data = js_sys::Float32Array::new_with_length(1000);
    for i in 0..1000 {
        data.set_index(i, (i as f32 * 0.01).sin());
    }

    let ts = WasmTimeSeries::new(data, 1000, 1, 1000.0).unwrap();
    assert_eq!(ts.num_samples(), 1000);
    assert_eq!(ts.num_channels(), 1);
}

#[wasm_bindgen_test]
fn test_encoding() {
    let data = js_sys::Float32Array::new_with_length(1000);
    for i in 0..1000 {
        data.set_index(i, (i as f32 * 0.01).sin());
    }

    let ts = WasmTimeSeries::new(data, 1000, 1, 1000.0).unwrap();
    let encoder = WasmLevelCrossingEncoder::new(0.1).unwrap();
    let spikes = encoder.encode(&ts).unwrap();

    assert!(spikes.length() > 0);
}
```

### 8.3 LSL Integration Tests

```rust
// Requires LSL runtime
#[tokio::test]
async fn test_lsl_roundtrip() {
    // Create outlet
    let outlet = DpbLslOutlet::new_signal_stream(
        "TestStream",
        "EEG",
        4,
        256.0,
        "test-source",
    ).unwrap();

    // Wait for connection
    outlet.wait_for_consumers(5.0);

    // Create inlet in separate task
    let handle = tokio::spawn(async {
        let inlet = DpbLslInlet::connect("TestStream", 5.0).unwrap();
        inlet.pull_sample(1.0).unwrap()
    });

    // Push sample
    outlet.push_sample(&[1.0, 2.0, 3.0, 4.0]).unwrap();

    // Verify reception
    let (sample, _) = handle.await.unwrap();
    assert_eq!(sample, vec![1.0, 2.0, 3.0, 4.0]);
}
```

---

## Appendix A: Platform Support Matrix

| Platform | R | LabVIEW | WASM | LSL | ONNX |
|----------|---|---------|------|-----|------|
| Windows x64 | ✅ | ✅ | N/A | ✅ | ✅ |
| Linux x64 | ✅ | ✅ | N/A | ✅ | ✅ |
| macOS x64 | ✅ | ✅ | N/A | ✅ | ✅ |
| macOS ARM | ✅ | ❌ | N/A | ✅ | ✅ |
| Chrome/Firefox | N/A | N/A | ✅ | N/A | N/A |
| Node.js | N/A | N/A | ✅ | ✅ | ✅ |
| NI Linux RT | ❌ | ✅ | N/A | ✅ | ❌ |

---

*End of Cross-Platform Integration Plan*
