//! Raw FFI bindings to the liblsl C library.
//!
//! This module provides low-level bindings to liblsl. These are unsafe
//! and should not be used directly - use the safe wrappers in other modules.
//!
//! ## liblsl Reference
//!
//! See <https://github.com/sccn/liblsl> for the full API documentation.

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_double, c_int, c_void};

/// Opaque type for LSL stream info.
pub type lsl_streaminfo = *mut c_void;

/// Opaque type for LSL outlet.
pub type lsl_outlet = *mut c_void;

/// Opaque type for LSL inlet.
pub type lsl_inlet = *mut c_void;

/// Opaque type for continuous resolver.
pub type lsl_continuous_resolver = *mut c_void;

/// Channel format constants matching liblsl.
pub mod lsl_channel_format {
    use std::os::raw::c_int;

    /// Single-precision floating point.
    pub const CF_FLOAT32: c_int = 1;
    /// Double-precision floating point.
    pub const CF_DOUBLE64: c_int = 2;
    /// String format (for markers).
    pub const CF_STRING: c_int = 3;
    /// 32-bit signed integer.
    pub const CF_INT32: c_int = 4;
    /// 16-bit signed integer.
    pub const CF_INT16: c_int = 5;
    /// 8-bit signed integer.
    pub const CF_INT8: c_int = 6;
    /// 64-bit signed integer.
    pub const CF_INT64: c_int = 7;
    /// Undefined format.
    pub const CF_UNDEFINED: c_int = 0;
}

/// Error codes from liblsl.
pub mod lsl_error_code {
    use std::os::raw::c_int;

    /// No error.
    pub const LSL_NO_ERROR: c_int = 0;
    /// Timeout occurred.
    pub const LSL_TIMEOUT_ERROR: c_int = -1;
    /// Lost connection to stream.
    pub const LSL_LOST_ERROR: c_int = -2;
    /// Invalid argument.
    pub const LSL_ARGUMENT_ERROR: c_int = -3;
    /// Internal error.
    pub const LSL_INTERNAL_ERROR: c_int = -4;
}

/// Processing flags for outlets.
pub mod lsl_processing {
    use std::os::raw::c_int;

    /// Leave timestamps unmodified.
    pub const PROC_NONE: c_int = 0;
    /// Remove clock drift.
    pub const PROC_CLOCKSYNC: c_int = 1;
    /// Dejitter timestamps.
    pub const PROC_DEJITTER: c_int = 2;
    /// Apply monotonize filter.
    pub const PROC_MONOTONIZE: c_int = 4;
    /// Apply threadsafe processing.
    pub const PROC_THREADSAFE: c_int = 8;
    /// All processing flags.
    pub const PROC_ALL: c_int = 1 | 2 | 4 | 8;
}

// FFI functions are only linked when the native feature is enabled
#[cfg(feature = "native")]
#[link(name = "lsl")]
unsafe extern "C" {
    // ========== Stream Info Functions ==========

    /// Create a new stream info object.
    pub fn lsl_create_streaminfo(
        name: *const c_char,
        stream_type: *const c_char,
        channel_count: c_int,
        nominal_srate: c_double,
        channel_format: c_int,
        source_id: *const c_char,
    ) -> lsl_streaminfo;

    /// Destroy a stream info object.
    pub fn lsl_destroy_streaminfo(info: lsl_streaminfo);

    /// Copy a stream info object.
    pub fn lsl_copy_streaminfo(info: lsl_streaminfo) -> lsl_streaminfo;

    /// Get the name of the stream.
    pub fn lsl_get_name(info: lsl_streaminfo) -> *const c_char;

    /// Get the type of the stream.
    pub fn lsl_get_type(info: lsl_streaminfo) -> *const c_char;

    /// Get the channel count.
    pub fn lsl_get_channel_count(info: lsl_streaminfo) -> c_int;

    /// Get the nominal sampling rate.
    pub fn lsl_get_nominal_srate(info: lsl_streaminfo) -> c_double;

    /// Get the channel format.
    pub fn lsl_get_channel_format(info: lsl_streaminfo) -> c_int;

    /// Get the source ID.
    pub fn lsl_get_source_id(info: lsl_streaminfo) -> *const c_char;

    /// Get the stream version.
    pub fn lsl_get_version(info: lsl_streaminfo) -> c_int;

    /// Get the created_at timestamp.
    pub fn lsl_get_created_at(info: lsl_streaminfo) -> c_double;

    /// Get the stream's unique ID.
    pub fn lsl_get_uid(info: lsl_streaminfo) -> *const c_char;

    /// Get the session ID.
    pub fn lsl_get_session_id(info: lsl_streaminfo) -> *const c_char;

    /// Get the hostname of the origin.
    pub fn lsl_get_hostname(info: lsl_streaminfo) -> *const c_char;

    /// Get the XML description.
    pub fn lsl_get_xml(info: lsl_streaminfo) -> *mut c_char;

    /// Get the number of bytes per channel sample.
    pub fn lsl_get_channel_bytes(info: lsl_streaminfo) -> c_int;

    /// Get the total number of bytes per sample (all channels).
    pub fn lsl_get_sample_bytes(info: lsl_streaminfo) -> c_int;

    // ========== Outlet Functions ==========

    /// Create a new outlet.
    pub fn lsl_create_outlet(
        info: lsl_streaminfo,
        chunk_size: c_int,
        max_buffered: c_int,
    ) -> lsl_outlet;

    /// Destroy an outlet.
    pub fn lsl_destroy_outlet(outlet: lsl_outlet);

    /// Push a sample (float).
    pub fn lsl_push_sample_f(outlet: lsl_outlet, data: *const c_float) -> c_int;

    /// Push a sample (double).
    pub fn lsl_push_sample_d(outlet: lsl_outlet, data: *const c_double) -> c_int;

    /// Push a sample (int32).
    pub fn lsl_push_sample_i(outlet: lsl_outlet, data: *const c_int) -> c_int;

    /// Push a sample (string).
    pub fn lsl_push_sample_str(outlet: lsl_outlet, data: *const *const c_char) -> c_int;

    /// Push a sample with timestamp (float).
    pub fn lsl_push_sample_ft(
        outlet: lsl_outlet,
        data: *const c_float,
        timestamp: c_double,
    ) -> c_int;

    /// Push a sample with timestamp (double).
    pub fn lsl_push_sample_dt(
        outlet: lsl_outlet,
        data: *const c_double,
        timestamp: c_double,
    ) -> c_int;

    /// Push a chunk of samples (float).
    pub fn lsl_push_chunk_f(
        outlet: lsl_outlet,
        data: *const c_float,
        data_elements: c_long,
    ) -> c_int;

    /// Push a chunk with timestamps (float).
    pub fn lsl_push_chunk_ft(
        outlet: lsl_outlet,
        data: *const c_float,
        data_elements: c_long,
        timestamp: c_double,
    ) -> c_int;

    /// Push a chunk with individual timestamps (float).
    pub fn lsl_push_chunk_ftp(
        outlet: lsl_outlet,
        data: *const c_float,
        data_elements: c_long,
        timestamps: *const c_double,
    ) -> c_int;

    /// Check if consumers are connected.
    pub fn lsl_have_consumers(outlet: lsl_outlet) -> c_int;

    /// Wait until consumers connect or timeout.
    pub fn lsl_wait_for_consumers(outlet: lsl_outlet, timeout: c_double) -> c_int;

    /// Get the stream info for an outlet.
    pub fn lsl_get_info(outlet: lsl_outlet) -> lsl_streaminfo;

    // ========== Inlet Functions ==========

    /// Create a new inlet.
    pub fn lsl_create_inlet(
        info: lsl_streaminfo,
        max_buflen: c_int,
        max_chunklen: c_int,
        recover: c_int,
    ) -> lsl_inlet;

    /// Destroy an inlet.
    pub fn lsl_destroy_inlet(inlet: lsl_inlet);

    /// Get full stream info from inlet.
    pub fn lsl_get_fullinfo(
        inlet: lsl_inlet,
        timeout: c_double,
        errcode: *mut c_int,
    ) -> lsl_streaminfo;

    /// Open an inlet stream.
    pub fn lsl_open_stream(inlet: lsl_inlet, timeout: c_double, errcode: *mut c_int);

    /// Close an inlet stream.
    pub fn lsl_close_stream(inlet: lsl_inlet);

    /// Get the time correction offset.
    pub fn lsl_time_correction(
        inlet: lsl_inlet,
        timeout: c_double,
        errcode: *mut c_int,
    ) -> c_double;

    /// Set post-processing options.
    pub fn lsl_set_postprocessing(inlet: lsl_inlet, processing_flags: c_int) -> c_int;

    /// Pull a sample (float).
    pub fn lsl_pull_sample_f(
        inlet: lsl_inlet,
        buffer: *mut c_float,
        buffer_elements: c_int,
        timeout: c_double,
        errcode: *mut c_int,
    ) -> c_double;

    /// Pull a sample (double).
    pub fn lsl_pull_sample_d(
        inlet: lsl_inlet,
        buffer: *mut c_double,
        buffer_elements: c_int,
        timeout: c_double,
        errcode: *mut c_int,
    ) -> c_double;

    /// Pull a sample (int32).
    pub fn lsl_pull_sample_i(
        inlet: lsl_inlet,
        buffer: *mut c_int,
        buffer_elements: c_int,
        timeout: c_double,
        errcode: *mut c_int,
    ) -> c_double;

    /// Pull a sample (string).
    pub fn lsl_pull_sample_str(
        inlet: lsl_inlet,
        buffer: *mut *mut c_char,
        buffer_elements: c_int,
        timeout: c_double,
        errcode: *mut c_int,
    ) -> c_double;

    /// Pull a chunk of samples (float).
    pub fn lsl_pull_chunk_f(
        inlet: lsl_inlet,
        data_buffer: *mut c_float,
        timestamp_buffer: *mut c_double,
        data_buffer_elements: c_long,
        timestamp_buffer_elements: c_long,
        timeout: c_double,
        errcode: *mut c_int,
    ) -> c_long;

    /// Pull a chunk of samples (double).
    pub fn lsl_pull_chunk_d(
        inlet: lsl_inlet,
        data_buffer: *mut c_double,
        timestamp_buffer: *mut c_double,
        data_buffer_elements: c_long,
        timestamp_buffer_elements: c_long,
        timeout: c_double,
        errcode: *mut c_int,
    ) -> c_long;

    /// Get the number of available samples.
    pub fn lsl_samples_available(inlet: lsl_inlet) -> c_int;

    /// Check if the inlet was recently corrected.
    pub fn lsl_was_clock_reset(inlet: lsl_inlet) -> c_int;

    /// Smooth the time stamps.
    pub fn lsl_smoothing_halftime(inlet: lsl_inlet) -> c_float;

    // ========== Resolver Functions ==========

    /// Resolve streams by property.
    pub fn lsl_resolve_byprop(
        buffer: *mut lsl_streaminfo,
        buffer_elements: c_int,
        prop: *const c_char,
        value: *const c_char,
        minimum: c_int,
        timeout: c_double,
    ) -> c_int;

    /// Resolve streams by predicate.
    pub fn lsl_resolve_bypred(
        buffer: *mut lsl_streaminfo,
        buffer_elements: c_int,
        pred: *const c_char,
        minimum: c_int,
        timeout: c_double,
    ) -> c_int;

    /// Resolve all streams.
    pub fn lsl_resolve_all(
        buffer: *mut lsl_streaminfo,
        buffer_elements: c_int,
        timeout: c_double,
    ) -> c_int;

    /// Create a continuous resolver.
    pub fn lsl_create_continuous_resolver(forget_after: c_double) -> lsl_continuous_resolver;

    /// Create a continuous resolver by property.
    pub fn lsl_create_continuous_resolver_byprop(
        prop: *const c_char,
        value: *const c_char,
        forget_after: c_double,
    ) -> lsl_continuous_resolver;

    /// Create a continuous resolver by predicate.
    pub fn lsl_create_continuous_resolver_bypred(
        pred: *const c_char,
        forget_after: c_double,
    ) -> lsl_continuous_resolver;

    /// Get results from continuous resolver.
    pub fn lsl_resolver_results(
        resolver: lsl_continuous_resolver,
        buffer: *mut lsl_streaminfo,
        buffer_elements: c_int,
    ) -> c_int;

    /// Destroy a continuous resolver.
    pub fn lsl_destroy_continuous_resolver(resolver: lsl_continuous_resolver);

    // ========== Utility Functions ==========

    /// Get the protocol version.
    pub fn lsl_protocol_version() -> c_int;

    /// Get the library version.
    pub fn lsl_library_version() -> c_int;

    /// Get the local clock value.
    pub fn lsl_local_clock() -> c_double;

    /// Free a string allocated by LSL.
    pub fn lsl_destroy_string(s: *mut c_char);
}

// Mock implementations when native feature is not enabled
#[cfg(not(feature = "native"))]
pub mod mock {
    use super::*;

    /// Mock: Create stream info.
    /// # Safety
    ///
    /// This is a mock: it ignores every argument and returns a null pointer,
    /// so it dereferences nothing and allocates nothing. It is `unsafe` only
    /// to match the signature of the real liblsl entry point.
    pub unsafe fn lsl_create_streaminfo(
        _name: *const c_char,
        _stream_type: *const c_char,
        _channel_count: c_int,
        _nominal_srate: c_double,
        _channel_format: c_int,
        _source_id: *const c_char,
    ) -> lsl_streaminfo {
        std::ptr::null_mut()
    }

    /// Mock: Destroy stream info.
    ///
    /// # Safety
    ///
    /// This is a mock and does nothing with the pointer, so any value is
    /// accepted. It is `unsafe` only to match the real liblsl signature.
    pub unsafe fn lsl_destroy_streaminfo(_info: lsl_streaminfo) {}

    /// Mock: Get library version.
    pub fn lsl_library_version() -> c_int {
        116 // Version 1.16
    }

    /// Mock: Get local clock.
    pub fn lsl_local_clock() -> c_double {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    /// Mock: Get protocol version.
    pub fn lsl_protocol_version() -> c_int {
        110 // Version 1.10
    }
}

/// Convert a Rust string to a C string for FFI.
pub fn to_c_string(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).unwrap_or_else(|_| std::ffi::CString::new("").unwrap())
}

/// Convert a C string to a Rust string.
///
/// # Safety
/// The pointer must be a valid C string pointer.
pub unsafe fn from_c_string(ptr: *const c_char) -> String {
    unsafe {
        if ptr.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Convert channel format enum to liblsl constant.
pub fn format_to_lsl(format: super::ChannelFormat) -> c_int {
    use super::ChannelFormat;
    match format {
        ChannelFormat::Float32 => lsl_channel_format::CF_FLOAT32,
        ChannelFormat::Float64 => lsl_channel_format::CF_DOUBLE64,
        ChannelFormat::String => lsl_channel_format::CF_STRING,
        ChannelFormat::Int32 => lsl_channel_format::CF_INT32,
        ChannelFormat::Int16 => lsl_channel_format::CF_INT16,
        ChannelFormat::Int8 => lsl_channel_format::CF_INT8,
        ChannelFormat::Undefined => lsl_channel_format::CF_UNDEFINED,
    }
}

/// Convert liblsl constant to channel format enum.
pub fn format_from_lsl(format: c_int) -> super::ChannelFormat {
    use super::ChannelFormat;
    match format {
        lsl_channel_format::CF_FLOAT32 => ChannelFormat::Float32,
        lsl_channel_format::CF_DOUBLE64 => ChannelFormat::Float64,
        lsl_channel_format::CF_STRING => ChannelFormat::String,
        lsl_channel_format::CF_INT32 => ChannelFormat::Int32,
        lsl_channel_format::CF_INT16 => ChannelFormat::Int16,
        lsl_channel_format::CF_INT8 => ChannelFormat::Int8,
        _ => ChannelFormat::Undefined,
    }
}

/// Error code to result conversion.
pub fn check_error(code: c_int) -> super::Result<()> {
    use super::LslError;
    match code {
        lsl_error_code::LSL_NO_ERROR => Ok(()),
        lsl_error_code::LSL_TIMEOUT_ERROR => Err(LslError::Timeout {
            operation: "unknown".to_string(),
            timeout_sec: 0.0,
        }),
        lsl_error_code::LSL_LOST_ERROR => Err(LslError::ConnectionLost(
            "Stream connection lost".to_string(),
        )),
        lsl_error_code::LSL_ARGUMENT_ERROR => {
            Err(LslError::Configuration("Invalid argument".to_string()))
        }
        lsl_error_code::LSL_INTERNAL_ERROR => {
            Err(LslError::Internal("Internal LSL error".to_string()))
        }
        _ => Err(LslError::Internal(format!("Unknown error code: {}", code))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_conversion() {
        use super::super::ChannelFormat;

        assert_eq!(
            format_to_lsl(ChannelFormat::Float32),
            lsl_channel_format::CF_FLOAT32
        );
        assert_eq!(
            format_to_lsl(ChannelFormat::Float64),
            lsl_channel_format::CF_DOUBLE64
        );
        assert_eq!(
            format_to_lsl(ChannelFormat::Int32),
            lsl_channel_format::CF_INT32
        );

        assert_eq!(
            format_from_lsl(lsl_channel_format::CF_FLOAT32),
            ChannelFormat::Float32
        );
        assert_eq!(
            format_from_lsl(lsl_channel_format::CF_DOUBLE64),
            ChannelFormat::Float64
        );
        assert_eq!(format_from_lsl(99), ChannelFormat::Undefined);
    }

    #[test]
    fn test_c_string_conversion() {
        let s = "test_string";
        let c_str = to_c_string(s);
        assert_eq!(c_str.to_str().unwrap(), s);
    }

    #[test]
    fn test_null_c_string() {
        let result = unsafe { from_c_string(std::ptr::null()) };
        assert!(result.is_empty());
    }

    #[test]
    fn test_error_code_conversion() {
        assert!(check_error(lsl_error_code::LSL_NO_ERROR).is_ok());
        assert!(check_error(lsl_error_code::LSL_TIMEOUT_ERROR).is_err());
        assert!(check_error(lsl_error_code::LSL_LOST_ERROR).is_err());
    }

    #[cfg(not(feature = "native"))]
    #[test]
    fn test_mock_functions() {
        let version = mock::lsl_library_version();
        assert!(version > 0);

        let clock = mock::lsl_local_clock();
        assert!(clock > 0.0);
    }
}
