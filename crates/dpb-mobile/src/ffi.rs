//! C Foreign Function Interface for native mobile integration.
//!
//! This module provides C-compatible functions for integrating the DPB mobile
//! runtime with iOS (Swift/Objective-C) and Android (JNI/Kotlin) applications.

use std::slice;
use std::ptr;
use std::ffi::{c_char, c_int, c_float};
use crate::{MobileRuntime, MobileModel, RuntimeConfig};

/// FFI error codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpbErrorCode {
    /// Success
    Success = 0,

    /// Null pointer error
    NullPointer = -1,

    /// Invalid parameter
    InvalidParameter = -2,

    /// Model not loaded
    ModelNotLoaded = -3,

    /// Invalid input dimensions
    InvalidInputDimensions = -4,

    /// Invalid output dimensions
    InvalidOutputDimensions = -5,

    /// Memory allocation failed
    AllocationFailed = -6,

    /// Inference failed
    InferenceFailed = -7,

    /// Deserialization failed
    DeserializationFailed = -8,

    /// Unknown error
    UnknownError = -99,
}

/// Opaque handle to DPB runtime
#[repr(C)]
pub struct DpbRuntime {
    _private: [u8; 0],
}

/// Opaque handle to DPB model
#[repr(C)]
pub struct DpbModel {
    _private: [u8; 0],
}

/// Runtime configuration for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DpbRuntimeConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,

    /// Number of threads
    pub thread_count: usize,

    /// Batch size
    pub batch_size: usize,

    /// Enable operator fusion (0 = false, 1 = true)
    pub enable_operator_fusion: u8,

    /// Enable memory planning (0 = false, 1 = true)
    pub enable_memory_planning: u8,

    /// Enable SIMD (0 = false, 1 = true)
    pub enable_simd: u8,

    /// Power mode (0 = low, 1 = balanced, 2 = high)
    pub power_mode: u8,

    /// Warmup iterations
    pub warmup_iterations: usize,
}

impl Default for DpbRuntimeConfig {
    fn default() -> Self {
        let config = RuntimeConfig::default();
        Self::from_runtime_config(&config)
    }
}

impl DpbRuntimeConfig {
    fn from_runtime_config(config: &RuntimeConfig) -> Self {
        Self {
            max_memory_mb: config.max_memory_mb,
            thread_count: config.thread_count,
            batch_size: config.batch_size,
            enable_operator_fusion: config.enable_operator_fusion as u8,
            enable_memory_planning: config.enable_memory_planning as u8,
            enable_simd: config.enable_simd as u8,
            power_mode: config.power_mode,
            warmup_iterations: config.warmup_iterations,
        }
    }

    fn to_runtime_config(self) -> RuntimeConfig {
        RuntimeConfig {
            max_memory_mb: self.max_memory_mb,
            thread_count: self.thread_count,
            batch_size: self.batch_size,
            enable_operator_fusion: self.enable_operator_fusion != 0,
            enable_memory_planning: self.enable_memory_planning != 0,
            enable_simd: self.enable_simd != 0,
            power_mode: self.power_mode,
            warmup_iterations: self.warmup_iterations,
        }
    }
}

/// Get library version string
///
/// # Safety
/// Returns a static C string that must not be freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Get default runtime configuration
///
/// # Safety
/// The config pointer must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_config_default(config: *mut DpbRuntimeConfig) -> c_int {
    if config.is_null() {
        return DpbErrorCode::NullPointer as c_int;
    }

    unsafe {
        *config = DpbRuntimeConfig::default();
    }
    DpbErrorCode::Success as c_int
}

/// Create a new runtime with default configuration
///
/// # Safety
/// Returns a pointer that must be freed with dpb_runtime_destroy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_runtime_create() -> *mut DpbRuntime {
    unsafe {
        dpb_runtime_create_with_config(&DpbRuntimeConfig::default())
    }
}

/// Create a new runtime with custom configuration
///
/// # Safety
/// - config must be a valid pointer
/// - Returns a pointer that must be freed with dpb_runtime_destroy
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_runtime_create_with_config(
    config: *const DpbRuntimeConfig,
) -> *mut DpbRuntime {
    if config.is_null() {
        return ptr::null_mut();
    }

    let config = unsafe { (*config).to_runtime_config() };

    match MobileRuntime::new_empty()
        .with_max_memory_mb(config.max_memory_mb)
        .with_thread_count(config.thread_count)
        .with_batch_size(config.batch_size)
        .with_operator_fusion(config.enable_operator_fusion)
        .with_memory_planning(config.enable_memory_planning)
        .with_simd(config.enable_simd)
        .with_power_mode(config.power_mode)
        .with_warmup_iterations(config.warmup_iterations)
        .build()
    {
        Ok(runtime) => Box::into_raw(Box::new(runtime)) as *mut DpbRuntime,
        Err(_) => ptr::null_mut(),
    }
}

/// Destroy a runtime and free its resources
///
/// # Safety
/// - runtime must be a valid pointer created by dpb_runtime_create
/// - runtime must not be used after this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_runtime_destroy(runtime: *mut DpbRuntime) {
    if !runtime.is_null() {
        unsafe {
            let _ = Box::from_raw(runtime as *mut MobileRuntime);
        }
    }
}

/// Load a model from bytes into the runtime
///
/// # Safety
/// - runtime must be a valid pointer
/// - data must be a valid pointer to model_size bytes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_model_load(
    runtime: *mut DpbRuntime,
    data: *const u8,
    data_size: usize,
) -> c_int {
    if runtime.is_null() || data.is_null() {
        return DpbErrorCode::NullPointer as c_int;
    }

    unsafe {
        let runtime = &mut *(runtime as *mut MobileRuntime);
        let data_slice = slice::from_raw_parts(data, data_size);

        match MobileModel::from_bytes(data_slice) {
            Ok(model) => match runtime.load_model(model) {
                Ok(_) => DpbErrorCode::Success as c_int,
                Err(e) => {
                    runtime.set_last_error(e.to_string());
                    DpbErrorCode::AllocationFailed as c_int
                }
            },
            Err(e) => {
                runtime.set_last_error(e.to_string());
                DpbErrorCode::DeserializationFailed as c_int
            }
        }
    }
}

/// Perform inference
///
/// # Safety
/// - runtime must be a valid pointer
/// - input must point to input_size floats
/// - output must point to output_size floats
/// - output_size must match the model's output dimension
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_infer(
    runtime: *mut DpbRuntime,
    input: *const c_float,
    input_size: usize,
    output: *mut c_float,
    output_size: usize,
) -> c_int {
    if runtime.is_null() || input.is_null() || output.is_null() {
        return DpbErrorCode::NullPointer as c_int;
    }

    unsafe {
        let runtime = &mut *(runtime as *mut MobileRuntime);
        let input_slice = slice::from_raw_parts(input, input_size);
        let output_slice = slice::from_raw_parts_mut(output, output_size);

        match runtime.infer_inplace(input_slice, output_slice) {
            Ok(_) => DpbErrorCode::Success as c_int,
            Err(e) => {
                let error_code = match e {
                    crate::RuntimeError::ModelNotLoaded => DpbErrorCode::ModelNotLoaded,
                    crate::RuntimeError::InvalidInputDimensions { .. } => DpbErrorCode::InvalidInputDimensions,
                    crate::RuntimeError::InvalidOutputDimensions { .. } => DpbErrorCode::InvalidOutputDimensions,
                    crate::RuntimeError::AllocationFailed(_) => DpbErrorCode::AllocationFailed,
                    crate::RuntimeError::InferenceFailed(_) => DpbErrorCode::InferenceFailed,
                    _ => DpbErrorCode::UnknownError,
                };
                runtime.set_last_error(e.to_string());
                error_code as c_int
            }
        }
    }
}

/// Get the last error message
///
/// # Safety
/// - runtime must be a valid pointer
/// - Returns a pointer to a C string that is valid until the next error or runtime destruction
/// - Returns null if no error
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_get_error(runtime: *mut DpbRuntime) -> *const c_char {
    if runtime.is_null() {
        return ptr::null();
    }

    unsafe {
        let runtime = &*(runtime as *mut MobileRuntime);
        match runtime.last_error() {
            Some(error) => error.as_ptr() as *const c_char,
            None => ptr::null(),
        }
    }
}

/// Get runtime memory usage in bytes
///
/// # Safety
/// - runtime must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_memory_usage(runtime: *mut DpbRuntime) -> usize {
    if runtime.is_null() {
        return 0;
    }

    unsafe {
        let runtime = &*(runtime as *mut MobileRuntime);
        runtime.memory_usage_bytes()
    }
}

/// Get inference count
///
/// # Safety
/// - runtime must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_inference_count(runtime: *mut DpbRuntime) -> usize {
    if runtime.is_null() {
        return 0;
    }

    unsafe {
        let runtime = &*(runtime as *mut MobileRuntime);
        runtime.inference_count()
    }
}

/// Reset inference statistics
///
/// # Safety
/// - runtime must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_reset_statistics(runtime: *mut DpbRuntime) {
    if runtime.is_null() {
        return;
    }

    unsafe {
        let runtime = &mut *(runtime as *mut MobileRuntime);
        runtime.reset_statistics();
    }
}

/// Check if model is loaded
///
/// # Safety
/// - runtime must be a valid pointer
/// - Returns 1 if model is loaded, 0 otherwise
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_is_model_loaded(runtime: *mut DpbRuntime) -> c_int {
    if runtime.is_null() {
        return 0;
    }

    unsafe {
        let runtime = &*(runtime as *mut MobileRuntime);
        runtime.is_model_loaded() as c_int
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(DpbErrorCode::Success as c_int, 0);
        assert!((DpbErrorCode::NullPointer as c_int) < 0);
    }

    #[test]
    fn test_version() {
        unsafe {
            let version_ptr = dpb_version();
            assert!(!version_ptr.is_null());
        }
    }

    #[test]
    fn test_default_config() {
        unsafe {
            let mut config = DpbRuntimeConfig::default();
            let result = dpb_config_default(&mut config);
            assert_eq!(result, DpbErrorCode::Success as c_int);
            assert_eq!(config.batch_size, 1);
        }
    }

    #[test]
    fn test_runtime_create_destroy() {
        unsafe {
            let runtime = dpb_runtime_create();
            assert!(!runtime.is_null());
            dpb_runtime_destroy(runtime);
        }
    }

    #[test]
    fn test_runtime_create_with_config() {
        unsafe {
            let config = DpbRuntimeConfig {
                max_memory_mb: 50,
                thread_count: 4,
                ..Default::default()
            };

            let runtime = dpb_runtime_create_with_config(&config);
            assert!(!runtime.is_null());

            dpb_runtime_destroy(runtime);
        }
    }

    #[test]
    fn test_is_model_loaded() {
        unsafe {
            let runtime = dpb_runtime_create();
            assert!(!runtime.is_null());

            let loaded = dpb_is_model_loaded(runtime);
            assert_eq!(loaded, 0);

            dpb_runtime_destroy(runtime);
        }
    }

    #[test]
    fn test_memory_usage() {
        unsafe {
            let runtime = dpb_runtime_create();
            assert!(!runtime.is_null());

            let usage = dpb_memory_usage(runtime);
            assert_eq!(usage, 0); // No model loaded

            dpb_runtime_destroy(runtime);
        }
    }
}
