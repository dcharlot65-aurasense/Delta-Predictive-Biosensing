# Rust Standards Review 2025 - Delta Predictive Biosensing

**Review Date:** December 31, 2025
**Rust Version:** 1.92.0
**Edition:** Rust 2024
**Reviewer:** Automated Code Analysis

---

## Executive Summary

This document presents a comprehensive review of the Delta Predictive Biosensing (DPB) Rust codebase against the latest 2025 Rust programming standards, best practices, and SOLID principles. The codebase demonstrates strong foundational patterns but has opportunities for improvement in error handling, memory optimization, async patterns, and type safety.

### Overall Assessment

| Category | Score | Status |
|----------|-------|--------|
| Code Architecture | ★★★★★ | Excellent |
| Error Handling | ★★★☆☆ | Needs Improvement |
| Memory Management | ★★★★☆ | Good |
| Async/Concurrency | ★★★☆☆ | Needs Improvement |
| API Design | ★★★★☆ | Good |
| Type Safety | ★★★★☆ | Good |
| Testing | ★★★★☆ | Good |
| Documentation | ★★★☆☆ | Needs Improvement |

---

## Table of Contents

1. [Standards Reference](#1-standards-reference)
2. [Error Handling Issues](#2-error-handling-issues)
3. [Memory Management Issues](#3-memory-management-issues)
4. [Async/Concurrency Issues](#4-asyncconcurrency-issues)
5. [API Design Issues](#5-api-design-issues)
6. [Type Safety Issues](#6-type-safety-issues)
7. [Testing Issues](#7-testing-issues)
8. [SOLID Principles Analysis](#8-solid-principles-analysis)
9. [Priority Remediation Plan](#9-priority-remediation-plan)

---

## 1. Standards Reference

This review is based on the following 2025 Rust standards and best practices:

### Key Resources Consulted

- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) - Official API design recommendations
- [Microsoft Pragmatic Rust Guidelines](https://microsoft.github.io/rust-guidelines/) - Enterprise-scale Rust practices
- [Rust Design Patterns](https://rust-unofficial.github.io/patterns/) - Idiomatic patterns and anti-patterns
- [Rust 1.92.0 Changelog](https://releases.rs/docs/1.92.0/) - Latest language features
- [Rust Error Handling 2025 Guide](https://markaicode.com/rust-error-handling-2025-guide/) - Modern error handling with thiserror 2.0
- [Zero-Copy Patterns in Rust](https://coinsbench.com/zero-copy-in-rust-challenges-and-solutions-c0d38a6468e9) - Memory optimization techniques
- [Tokio Best Practices](https://tokio.rs/tokio/tutorial/async) - Async programming standards

### Rust 1.92.0 Key Features

- Improved backtraces with `-C panic=abort` on Linux
- Multiple bounds for associated items in trait bounds
- `unused_must_use` lint now ignores `Result<!, E>`
- New `cargo-wizard` for build optimization
- Enhanced rustdoc search capabilities

---

## 2. Error Handling Issues

### 2.1 Critical: `panic!()` in Production Code

**Severity:** 🔴 CRITICAL
**Count:** 17 instances
**Impact:** Application crashes, undefined behavior in FFI contexts

#### Affected Files

| File | Line | Panic Message |
|------|------|---------------|
| `dpb-neurons/src/stochastic.rs` | 310 | `"Output weights not trained. Call train_readout first."` |
| `dpb-neurons/src/reservoir.rs` | 310, 464 | Similar training validation panics |
| `dpb-snn/src/architectures/feedforward.rs` | 29 | `"Need at least 2 layer sizes"` |
| `dpb-snn/src/architectures/recurrent.rs` | 38 | `"Need at least one hidden layer"` |
| `dpb-snn/src/distributed/communication.rs` | 517, 528 | `"Wrong message type"` |
| `dpb-snn/src/fusion/mod.rs` | 299, 315, 352 | `"Expected dense/sparse output"` |
| `dpb-snn/src/fusion/early.rs` | 176 | Type mismatch panic |
| `dpb-snn/src/fusion/late.rs` | 287 | Type mismatch panic |
| `dpb-snn/src/fusion/gated.rs` | 296 | Type mismatch panic |
| `dpb-snn/src/fusion/attention.rs` | 240 | Type mismatch panic |

#### Recommendation

Replace panics with `Result<T, E>` returns:

```rust
// Before (anti-pattern)
pub fn new(layer_sizes: Vec<usize>) -> Self {
    if layer_sizes.len() < 2 {
        panic!("Need at least 2 layer sizes");
    }
    // ...
}

// After (idiomatic Rust 2025)
pub fn new(layer_sizes: Vec<usize>) -> Result<Self, DpbError> {
    if layer_sizes.len() < 2 {
        return Err(DpbError::InvalidDimensions(
            "Need at least 2 layer sizes (input and output)".into()
        ));
    }
    // ...
}
```

---

### 2.2 High: `.unwrap()` and `.expect()` in Production

**Severity:** 🟠 HIGH
**Count:** ~80 instances in production code
**Impact:** Runtime panics on edge cases

#### Key Problem Areas

| File | Issue | Risk |
|------|-------|------|
| `dpb-neurons/src/stochastic.rs` | `Normal::new().unwrap()` | Panics on invalid sigma |
| `dpb-neurons/src/reservoir.rs` | `Normal::new().unwrap()` | Same issue |
| `dpb-encoders/src/balance.rs` | `partial_cmp().unwrap()` | Panics on NaN |
| `dpb-federated/src/compression.rs` | `partial_cmp().unwrap()` x6 | NaN handling |
| `dpb-wasm/src/spiketrain.rs` | `partial_cmp().unwrap()` | NaN timestamps |
| `dpb-python/src/numpy_utils.rs` | Array `.unwrap()` | Shape validation |
| `dpb-core/src/signal/ecg.rs` | `expect()` on filter creation | Invalid params |

#### Recommendation

Handle NaN values explicitly and propagate errors:

```rust
// Before (panics on NaN)
events.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

// After (NaN-safe, Rust 2025 idiom)
events.sort_by(|a, b| {
    a.timestamp
        .partial_cmp(&b.timestamp)
        .unwrap_or(std::cmp::Ordering::Equal)
});

// Or for strict NaN rejection
events.sort_by(|a, b| {
    match a.timestamp.partial_cmp(&b.timestamp) {
        Some(ord) => ord,
        None => {
            tracing::warn!("NaN timestamp detected, treating as equal");
            std::cmp::Ordering::Equal
        }
    }
});
```

---

### 2.3 Medium: Missing `#[non_exhaustive]` on Error Enums

**Severity:** 🟡 MEDIUM
**Count:** 5 error types
**Impact:** Breaking changes when adding error variants

#### Affected Error Types

| File | Type | Variants |
|------|------|----------|
| `dpb-core/src/error.rs` | `DpbError` | 19 |
| `dpb-export/src/error.rs` | `ExportError` | 9 |
| `dpb-federated/src/error.rs` | `FederatedError` | 15 |
| `dpb-clinical/src/error.rs` | `ClinicalError` | 12 |
| `dpb-lsl/src/error.rs` | `LslError` | 16 |

#### Recommendation

```rust
// Add #[non_exhaustive] for forward compatibility
#[derive(Error, Debug)]
#[non_exhaustive]  // Allows adding variants without breaking downstream
pub enum DpbError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("GPU error: {0}")]
    Gpu(String),
    // ...
}
```

---

### 2.4 Low: Missing Error Context

**Severity:** 🟢 LOW
**Count:** Codebase-wide
**Impact:** Difficult debugging, poor error messages

#### Current Pattern

```rust
// Current - minimal context
.map_err(|e| DpbError::Gpu(e.to_string()))?
```

#### Recommended Pattern (thiserror 2.0 + anyhow)

```rust
// Add anyhow for application-level context
use anyhow::{Context, Result};

// Library errors stay with thiserror
#[derive(Error, Debug)]
pub enum DpbError {
    #[error("GPU shader compilation failed: {0}")]
    ShaderCompilation(#[source] wgpu::Error),
}

// Application code adds context
fn load_model(path: &Path) -> Result<Model> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read model from {:?}", path))?;

    Model::from_bytes(&bytes)
        .with_context(|| "Model deserialization failed")?
}
```

---

## 3. Memory Management Issues

### 3.1 High: Unnecessary Cloning

**Severity:** 🟠 HIGH
**Count:** 194 files with `.clone()` calls
**Impact:** Increased memory usage, reduced performance

#### Key Problem Areas

| File | Issue | Impact |
|------|-------|--------|
| `dpb-snn/src/gpu/memory.rs` | Cloning `PoolStats` struct | Memory overhead |
| `dpb-snn/src/gpu/metal.rs` | Cloning GPU pipeline objects | Expensive GPU state copy |
| `dpb-snn/src/architectures/recurrent.rs` | Cloning `Array1` in loops | O(n) memory per iteration |
| `dpb-core/src/types.rs` | Cloning `channel_names` | Unnecessary allocation |
| `dpb-federated/src/server.rs` | Double `client_id.to_string()` | 2x allocation |

#### Recommendations

**Pattern 1: Return references instead of clones**

```rust
// Before
pub fn stats(&self) -> PoolStats {
    self.stats.lock().unwrap().clone()
}

// After - return reference or use Arc
pub fn stats(&self) -> impl std::ops::Deref<Target = PoolStats> + '_ {
    self.stats.lock().unwrap()
}

// Or with Arc for shared ownership
pub fn stats(&self) -> Arc<PoolStats> {
    Arc::clone(&self.stats)  // Cheap reference count increment
}
```

**Pattern 2: Use `Cow<'a, str>` for flexible string handling**

```rust
use std::borrow::Cow;

pub struct ModelConfig<'a> {
    pub name: Cow<'a, str>,  // Borrowed or owned as needed
    pub format: Cow<'static, str>,  // Static strings avoid allocation
}

impl<'a> ModelConfig<'a> {
    pub fn new(name: &'a str) -> Self {
        Self {
            name: Cow::Borrowed(name),
            format: Cow::Borrowed("onnx"),
        }
    }
}
```

---

### 3.2 Medium: Missing `Vec::with_capacity()`

**Severity:** 🟡 MEDIUM
**Count:** 256 files with `Vec::new()`
**Impact:** Unnecessary reallocations

#### Affected Areas

| File | Pattern | Fix |
|------|---------|-----|
| `dpb-snn/src/baselines/rnn.rs` | Loop building `Vec` | Pre-allocate with known size |
| `dpb-snn/src/neuromorphic/brainscales.rs` | Network mapping vectors | Use `.len()` for capacity |
| `dpb-snn/src/explain/visualization.rs` | Raster point collection | Sum spike counts first |

#### Recommendation

```rust
// Before - multiple reallocations
let mut neurons = Vec::new();
for pop in &network.populations {
    neurons.push(process(pop));
}

// After - single allocation
let mut neurons = Vec::with_capacity(network.populations.len());
for pop in &network.populations {
    neurons.push(process(pop));
}

// Even better - use collect with size hint
let neurons: Vec<_> = network.populations
    .iter()
    .map(|pop| process(pop))
    .collect();  // collect() uses size_hint() internally
```

---

### 3.3 Medium: String Parameters Should Use `&str`

**Severity:** 🟡 MEDIUM
**Count:** Multiple function signatures
**Impact:** Unnecessary allocations at API boundaries

#### Affected Signatures

```rust
// Current - forces allocation
pub fn with_sex(mut self, sex: String) -> Self
pub fn add_medication(&mut self, medication: String)
pub fn set_environment(&mut self, key: String, value: f64)
pub fn set_device(&mut self, key: String, value: String)
```

#### Recommended Signatures

```rust
// Flexible - accepts &str, String, &String, Cow<str>, etc.
pub fn with_sex(mut self, sex: impl Into<String>) -> Self {
    self.sex = Some(sex.into());
    self
}

// Or for read-only access
pub fn find_by_name(&self, name: &str) -> Option<&Item>
```

---

### 3.4 Low: Zero-Copy Opportunities

**Severity:** 🟢 LOW
**Impact:** Performance optimization for large data

#### Opportunities

1. **Use `zerocopy` crate for binary parsing**
2. **Use `bytes::Bytes` for shared buffer slices**
3. **Return iterators instead of collecting to Vec**

```rust
// Before - allocates new Vec
pub fn get_spikes_at(&self, batch_idx: usize, time_idx: usize) -> Vec<usize> {
    self.events.iter()
        .filter(|&&(b, t, _)| b == batch_idx && t == time_idx)
        .map(|&(_, _, n)| n)
        .collect()
}

// After - zero allocation, lazy evaluation
pub fn get_spikes_at(&self, batch_idx: usize, time_idx: usize)
    -> impl Iterator<Item = usize> + '_
{
    self.events.iter()
        .filter(move |&&(b, t, _)| b == batch_idx && t == time_idx)
        .map(|&(_, _, n)| n)
}
```

---

## 4. Async/Concurrency Issues

### 4.1 High: `Arc<Mutex<T>>` Overuse

**Severity:** 🟠 HIGH
**Count:** 55 instances of `.lock().unwrap()`
**Impact:** Potential deadlocks, performance bottlenecks

#### Key Problem Areas

| File | Structure | Issue |
|------|-----------|-------|
| `dpb-snn/src/distributed/communication.rs` | `MessageQueue` with `Arc<Mutex<VecDeque>>` | Should use channels |
| `dpb-snn/src/gpu/memory.rs` | 3 separate `Mutex` fields | Should consolidate |
| `dpb-snn/src/gpu/cuda.rs` | Multiple `Arc<Mutex>` | Lock contention risk |

#### Recommendation

Replace `Arc<Mutex<VecDeque>>` with channels:

```rust
// Before - mutex-based message queue (anti-pattern)
pub struct MessageQueue {
    queue: Arc<Mutex<VecDeque<Message>>>,
}

impl MessageQueue {
    pub fn push(&self, message: Message) {
        self.queue.lock().unwrap().push_back(message);
    }

    pub fn pop(&self) -> Option<Message> {
        self.queue.lock().unwrap().pop_front()
    }
}

// After - channel-based (idiomatic async Rust)
use tokio::sync::mpsc;

pub struct MessageQueue {
    sender: mpsc::UnboundedSender<Message>,
    receiver: mpsc::UnboundedReceiver<Message>,
}

impl MessageQueue {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self { sender, receiver }
    }

    pub fn push(&self, message: Message) -> Result<(), SendError<Message>> {
        self.sender.send(message)
    }

    pub async fn pop(&mut self) -> Option<Message> {
        self.receiver.recv().await
    }
}
```

---

### 4.2 Medium: Blocking Operations in Async

**Severity:** 🟡 MEDIUM
**Count:** 3+ locations
**Impact:** Async executor starvation

#### Affected Code

| File | Issue |
|------|-------|
| `dpb-core/src/gpu/backend.rs` | Blocking `recv()` in GPU buffer mapping |
| `dpb-lsl/src/inlet.rs` | `inlet.open()` blocking before `spawn_blocking` |

#### Recommendation

```rust
// Use spawn_blocking for blocking I/O in async context
pub async fn new(info: &StreamInfo) -> Result<Self> {
    // Move blocking operation to dedicated thread
    let inlet = tokio::task::spawn_blocking(move || {
        let mut inlet = LslInlet::new(info)?;
        inlet.open(5.0)?;  // Blocking I/O
        Ok::<_, Error>(inlet)
    }).await??;

    // Continue with async operations
    Ok(Self { inlet })
}
```

---

### 4.3 Medium: Inconsistent Async Semantics

**Severity:** 🟡 MEDIUM
**Count:** 2 functions
**Impact:** Confusing API, wasted async overhead

#### Affected Code

```rust
// File: dpb-core/src/gpu/backend.rs
pub async fn create_default_backend() -> Result<Box<dyn ComputeBackend>> {
    #[cfg(feature = "cuda")]
    {
        if cuda::CUDABackend::is_available() {
            // SYNC - doesn't await anything
            return Ok(Box::new(cuda::CUDABackend::new(0)?));
        }
    }

    // ASYNC - only this path awaits
    Ok(Box::new(WebGPUBackend::new().await?))
}
```

#### Recommendation

Make all paths consistently async or split into separate functions:

```rust
// Option 1: Consistent async
pub async fn create_default_backend() -> Result<Box<dyn ComputeBackend>> {
    #[cfg(feature = "cuda")]
    {
        if cuda::CUDABackend::is_available() {
            return Ok(Box::new(
                tokio::task::spawn_blocking(|| cuda::CUDABackend::new(0))
                    .await??
            ));
        }
    }
    Ok(Box::new(WebGPUBackend::new().await?))
}

// Option 2: Split API
pub fn create_cuda_backend() -> Result<Box<dyn ComputeBackend>>
pub async fn create_webgpu_backend() -> Result<Box<dyn ComputeBackend>>
```

---

### 4.4 Low: Missing Concurrent Primitives

**Severity:** 🟢 LOW
**Count:** 0 uses of `tokio::select!` or `futures::join!`
**Impact:** Sequential execution where parallel is possible

#### Opportunity

```rust
// Before - sequential
let result1 = fetch_data_1().await?;
let result2 = fetch_data_2().await?;

// After - concurrent with join!
let (result1, result2) = tokio::join!(
    fetch_data_1(),
    fetch_data_2()
);
let result1 = result1?;
let result2 = result2?;
```

---

## 5. API Design Issues

### 5.1 Medium: Missing `#[must_use]`

**Severity:** 🟡 MEDIUM
**Count:** Multiple Result-returning functions
**Impact:** Silently ignored errors

#### Affected Code

```rust
// File: dpb-wasm/src/lib.rs
pub fn new() -> Result<PerformanceTimer, JsValue>  // Missing #[must_use]
pub fn elapsed_ms(&self) -> Result<f64, JsValue>   // Missing #[must_use]
pub fn reset(&mut self) -> Result<(), JsValue>     // Missing #[must_use]
```

#### Recommendation

```rust
#[must_use = "this Result may contain an error that should be handled"]
pub fn new() -> Result<PerformanceTimer, JsValue> { ... }
```

---

### 5.2 Low: `Box<dyn Trait>` vs Generics

**Severity:** 🟢 LOW
**Count:** 45 files
**Impact:** Dynamic dispatch overhead

#### Example

```rust
// Before - dynamic dispatch
pub struct FederatedServer {
    aggregator: Box<dyn Aggregator>,
}

// After - monomorphization (zero-cost)
pub struct FederatedServer<A: Aggregator> {
    aggregator: A,
}
```

---

## 6. Type Safety Issues

### 6.1 Medium: Unsafe `as` Casts

**Severity:** 🟡 MEDIUM
**Count:** 15+ instances
**Impact:** Silent overflow on large collections

#### Problem Pattern

```rust
// Files: dpb-encoders, dpb-snn/export, dpb-export, dpb-mobile
let len = collection.len() as u32;  // Overflows if len > u32::MAX
```

#### Recommendation

```rust
// Use TryFrom for checked conversion
let len: u32 = collection.len()
    .try_into()
    .map_err(|_| DpbError::InvalidDimensions("Collection too large for u32"))?;

// Or with explicit handling
let len = u32::try_from(collection.len())
    .expect("Collection size exceeds u32::MAX");
```

---

### 6.2 Low: Limited Newtype Usage

**Severity:** 🟢 LOW
**Count:** 1 newtype found
**Impact:** Potential type confusion

#### Current State

Only `EdssScore(f64)` uses newtype pattern.

#### Recommendation

Add newtypes for domain concepts:

```rust
/// Unique identifier for a signal channel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(pub u32);

/// Index into a neuron population
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NeuronIndex(pub usize);

/// Count of samples in a signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleCount(pub usize);

impl SampleCount {
    pub fn as_duration(&self, sample_rate: f64) -> std::time::Duration {
        std::time::Duration::from_secs_f64(self.0 as f64 / sample_rate)
    }
}
```

---

## 7. Testing Issues

### 7.1 Medium: Limited Async Test Coverage

**Severity:** 🟡 MEDIUM
**Count:** Only 6 `#[tokio::test]` vs extensive async code
**Impact:** Untested async edge cases

#### Current Coverage

| Crate | `#[tokio::test]` Count |
|-------|------------------------|
| dpb-core | 3 |
| dpb-viz | 3 |
| dpb-federated | 0 |
| dpb-lsl | 0 |

#### Recommendation

Add async tests for all async functions:

```rust
#[tokio::test]
async fn test_gpu_context_creation() {
    let config = GpuConfig::default();
    let context = GpuContext::new(&config).await;
    assert!(context.is_ok());
}

#[tokio::test]
async fn test_inlet_streaming() {
    let inlet = AsyncLslInlet::new(&info).await.unwrap();
    let samples = inlet.pull_samples(10).await;
    assert_eq!(samples.len(), 10);
}
```

---

### 7.2 Low: No Property-Based Testing

**Severity:** 🟢 LOW
**Count:** 0 proptest/quickcheck usage
**Impact:** Edge cases may be missed

#### Recommendation

Add proptest for numerical code:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_spike_event_serialization_roundtrip(
        timestamp in 0.0f64..1000.0,
        channel in 0u32..1000,
        polarity in -1i8..=1i8,
        magnitude in 0.0f32..100.0,
    ) {
        let event = SpikeEvent::new(timestamp, channel, polarity, magnitude);
        let serialized = bincode::serialize(&event).unwrap();
        let deserialized: SpikeEvent = bincode::deserialize(&serialized).unwrap();
        prop_assert_eq!(event, deserialized);
    }
}
```

---

## 8. SOLID Principles Analysis

### 8.1 Single Responsibility Principle ✓

**Status:** Well Implemented

The codebase demonstrates excellent module separation:

- **dpb-core**: Foundational types and traits only
- **dpb-encoders**: Event encoding algorithms
- **dpb-snn**: Neural network architectures
- **dpb-export**: Serialization formats

### 8.2 Open/Closed Principle ✓

**Status:** Well Implemented

Traits enable extension without modification:

```rust
pub trait EventEncoder: Send + Sync {
    type Config: Clone + Send + Sync;
    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>>;
}

// New encoders can be added without modifying existing code
impl EventEncoder for CustomEncoder { ... }
```

### 8.3 Liskov Substitution Principle ⚠️

**Status:** Partially Applicable

Rust doesn't have inheritance, but trait implementations should be substitutable. Found issues in `dpb-snn/src/fusion/*` where panics violate expected behavior:

```rust
// Anti-pattern - violates LSP
fn process(&self, input: Representation) -> Representation {
    match input {
        Representation::Dense(d) => self.process_dense(d),
        Representation::Sparse(_) => panic!("Expected dense!"),  // Violates contract
    }
}
```

### 8.4 Interface Segregation Principle ✓

**Status:** Well Implemented

Traits are focused and composable:

```rust
pub trait MembraneDynamics: Send + Sync { ... }
pub trait SynapticModel: Send + Sync { ... }
pub trait PlasticityRule: Send + Sync { ... }
```

### 8.5 Dependency Inversion Principle ✓

**Status:** Well Implemented

High-level modules depend on abstractions:

```rust
pub struct Pipeline<E: EventEncoder, N: SNNNetwork> {
    encoder: E,
    network: N,
}
```

---

## 9. Priority Remediation Plan

### Phase 1: Critical (Week 1-2)

| Issue | Files | Action | Effort |
|-------|-------|--------|--------|
| Remove `panic!()` from production | 17 locations | Convert to `Result<T, E>` | 2 days |
| Fix NaN handling in comparisons | `dpb-encoders`, `dpb-federated` | Add explicit NaN handling | 1 day |
| Add `#[non_exhaustive]` to errors | 5 error types | Simple attribute addition | 2 hours |

### Phase 2: High Priority (Week 3-4)

| Issue | Files | Action | Effort |
|-------|-------|--------|--------|
| Replace `Arc<Mutex<VecDeque>>` | `communication.rs` | Use tokio channels | 2 days |
| Fix `.unwrap()` on distributions | `stochastic.rs`, `reservoir.rs` | Propagate errors | 1 day |
| Add `Vec::with_capacity()` | Architecture modules | Pre-allocate vectors | 1 day |
| Fix blocking in async | `backend.rs`, `inlet.rs` | Use `spawn_blocking` | 1 day |

### Phase 3: Medium Priority (Week 5-6)

| Issue | Files | Action | Effort |
|-------|-------|--------|--------|
| Replace `as` casts with `TryFrom` | 15+ locations | Checked conversions | 2 days |
| Add `#[must_use]` attributes | WASM and public APIs | Annotation pass | 4 hours |
| Reduce unnecessary cloning | Memory-heavy modules | Use references/Cow | 3 days |
| Add async test coverage | federated, lsl crates | Write `#[tokio::test]` | 2 days |

### Phase 4: Low Priority (Ongoing)

| Issue | Action | Effort |
|-------|--------|--------|
| Add newtype wrappers | Create domain types | 2 days |
| Add property-based tests | Integrate proptest | 2 days |
| Improve documentation | Add rustdoc to public items | Ongoing |
| Replace `Box<dyn>` with generics | Where monomorphization helps | As needed |

---

## Appendix A: Clippy Configuration Recommendation

Create `clippy.toml` at workspace root:

```toml
# Clippy configuration for DPB
avoid-breaking-exported-api = false
cognitive-complexity-threshold = 25
too-many-arguments-threshold = 7
type-complexity-threshold = 250
```

Add to `Cargo.toml`:

```toml
[workspace.lints.clippy]
# Deny these in CI
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"

# Warn on these
clone_on_ref_ptr = "warn"
redundant_clone = "warn"
inefficient_to_string = "warn"
large_enum_variant = "warn"
```

---

## Appendix B: Recommended Dependencies Update

```toml
[workspace.dependencies]
# Error handling (upgrade to 2.0)
thiserror = "2.0"
anyhow = "2.0"  # For application-level context

# Property testing (add)
proptest = "1.4"

# Zero-copy utilities (add)
zerocopy = { version = "0.8", features = ["derive"] }
bytes = "1.5"
```

---

## Appendix C: File Reference Quick Lookup

### Critical Priority Files

```
crates/dpb-snn/src/fusion/mod.rs          # 4 panics
crates/dpb-neurons/src/stochastic.rs      # 3 unwraps + 1 panic
crates/dpb-neurons/src/reservoir.rs       # 2 panics + 1 unwrap
crates/dpb-federated/src/compression.rs   # 6 unwraps on floats
crates/dpb-encoders/src/balance.rs        # 3 unwraps on floats
crates/dpb-core/src/signal/ecg.rs         # 2 expect calls
crates/dpb-snn/src/distributed/communication.rs  # MessageQueue anti-pattern
```

### Error Type Files

```
crates/dpb-core/src/error.rs
crates/dpb-export/src/error.rs
crates/dpb-federated/src/error.rs
crates/dpb-clinical/src/error.rs
crates/dpb-lsl/src/error.rs
```

---

*Report generated with Claude Code analysis tools*
*Based on Rust 1.92.0 standards and 2025 best practices*
