//! Mobile Runtime for Delta-Predictive-Biosensing Framework
//!
//! This crate provides optimized mobile inference runtime for iOS and Android platforms.
//! It includes:
//! - Low memory footprint inference
//! - C FFI bindings for native integration
//! - Platform-specific optimizations (Metal, CoreML, NNAPI, Vulkan)
//! - Quantized model support (int8, float16)
//! - Performance benchmarking tools
//!
//! # Platform Support
//!
//! ## iOS
//! Build for iOS devices:
//! ```bash
//! cargo build --target aarch64-apple-ios --features ios --release
//! ```
//!
//! Build for iOS simulator:
//! ```bash
//! cargo build --target aarch64-apple-ios-sim --features ios --release
//! ```
//!
//! ## Android
//! Build for Android ARM64:
//! ```bash
//! cargo build --target aarch64-linux-android --features android --release
//! ```
//!
//! Build for Android ARMv7:
//! ```bash
//! cargo build --target armv7-linux-androideabi --features android --release
//! ```
//!
//! # Features
//!
//! - `ios`: Enable iOS-specific optimizations and APIs
//! - `android`: Enable Android-specific optimizations and APIs
//! - `quantized`: Enable quantized model support
//! - `metal`: Enable Metal GPU backend (iOS)
//! - `coreml`: Enable CoreML integration (iOS)
//! - `nnapi`: Enable Android Neural Networks API
//! - `vulkan`: Enable Vulkan compute backend (Android)
//!
//! # Example Usage
//!
//! ```rust
//! use dpb_mobile::{MobileModel, MobileRuntime};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Describe the model: a name and its input/output widths.
//! let model = MobileModel::new("demo".to_string(), 128, 8);
//!
//! // Create runtime with optimization settings. `MobileRuntime::builder` returns a
//! // builder; `build` yields the runtime.
//! let mut runtime = MobileRuntime::builder(model)
//!     .with_max_memory_mb(50)
//!     .with_thread_count(2)
//!     .build()?;
//!
//! // Perform inference
//! let input = vec![0.5f32; 128];
//! let output = runtime.infer(&input)?;
//! assert_eq!(output.len(), 8);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

// Note: We use std for now as the runtime requires allocation and platform features
// A future no_std version could be created for embedded targets

pub mod runtime;
pub mod model;
pub mod ffi;
pub mod optimization;
pub mod benchmark;
pub mod npu;

#[cfg(target_os = "ios")]
pub mod ios;

#[cfg(target_os = "android")]
pub mod android;

// Re-exports for convenience
pub use runtime::{MobileRuntime, RuntimeConfig, RuntimeError};
pub use model::{MobileModel, ModelFormat, QuantizationType};
pub use optimization::{OptimizationLevel, WeightPruner, OperatorFusion};
pub use benchmark::{BenchmarkResult, LatencyMetrics, MemoryMetrics};

/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimum supported model version (1.0.0 encoded as u32)
pub const MIN_MODEL_VERSION: u32 = 0x00010000; // 1.0.0 = (1 << 16)

/// Maximum supported model version (1.0.0 encoded as u32)
pub const MAX_MODEL_VERSION: u32 = 0x00010000; // 1.0.0 = (1 << 16)

/// Default maximum memory usage in MB
pub const DEFAULT_MAX_MEMORY_MB: usize = 100;

/// Default thread count for inference
pub const DEFAULT_THREAD_COUNT: usize = 2;

/// Mobile runtime result type
pub type Result<T> = core::result::Result<T, RuntimeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_model_version_range() {
        assert!(MIN_MODEL_VERSION <= MAX_MODEL_VERSION);
    }
}
