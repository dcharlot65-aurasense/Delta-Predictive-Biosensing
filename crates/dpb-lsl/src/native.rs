//! Native liblsl runtime wrapper.
//!
//! This module provides safe Rust wrappers around the native liblsl C library.
//! It requires the `native` feature to be enabled and liblsl to be installed.
//!
//! ## Installation
//!
//! ### Linux (Debian/Ubuntu)
//! ```bash
//! sudo apt-get install liblsl-dev
//! ```
//!
//! ### macOS
//! ```bash
//! brew install labstreaminglayer/tap/lsl
//! ```
//!
//! ### Windows
//! Download from <https://github.com/sccn/liblsl/releases> and add to PATH.
//!
//! ## Building with Native Support
//!
//! ```bash
//! cargo build -p dpb-lsl --features native
//! ```

#[cfg(feature = "native")]
use crate::ffi;
use crate::{ChannelFormat, LslError, Result};
use std::ffi::CString;

/// Native LSL stream info wrapper.
#[cfg(feature = "native")]
pub struct NativeStreamInfo {
    handle: ffi::lsl_streaminfo,
    owned: bool,
}

#[cfg(feature = "native")]
impl NativeStreamInfo {
    /// Create a new stream info.
    pub fn new(
        name: &str,
        stream_type: &str,
        channel_count: i32,
        nominal_srate: f64,
        channel_format: ChannelFormat,
        source_id: &str,
    ) -> Result<Self> {
        let c_name = CString::new(name).map_err(|e| LslError::Configuration(e.to_string()))?;
        let c_type = CString::new(stream_type).map_err(|e| LslError::Configuration(e.to_string()))?;
        let c_source = CString::new(source_id).map_err(|e| LslError::Configuration(e.to_string()))?;

        let handle = unsafe {
            ffi::lsl_create_streaminfo(
                c_name.as_ptr(),
                c_type.as_ptr(),
                channel_count,
                nominal_srate,
                ffi::format_to_lsl(channel_format),
                c_source.as_ptr(),
            )
        };

        if handle.is_null() {
            return Err(LslError::Internal("Failed to create stream info".to_string()));
        }

        Ok(Self { handle, owned: true })
    }

    /// Create from raw handle (does not take ownership).
    pub(crate) unsafe fn from_raw(handle: ffi::lsl_streaminfo) -> Self {
        Self { handle, owned: false }
    }

    /// Get the raw handle.
    pub(crate) fn as_raw(&self) -> ffi::lsl_streaminfo {
        self.handle
    }

    /// Get the stream name.
    pub fn name(&self) -> String {
        unsafe { ffi::from_c_string(ffi::lsl_get_name(self.handle)) }
    }

    /// Get the stream type.
    pub fn stream_type(&self) -> String {
        unsafe { ffi::from_c_string(ffi::lsl_get_type(self.handle)) }
    }

    /// Get the channel count.
    pub fn channel_count(&self) -> i32 {
        unsafe { ffi::lsl_get_channel_count(self.handle) }
    }

    /// Get the nominal sample rate.
    pub fn nominal_srate(&self) -> f64 {
        unsafe { ffi::lsl_get_nominal_srate(self.handle) }
    }

    /// Get the channel format.
    pub fn channel_format(&self) -> ChannelFormat {
        let format = unsafe { ffi::lsl_get_channel_format(self.handle) };
        ffi::format_from_lsl(format)
    }

    /// Get the source ID.
    pub fn source_id(&self) -> String {
        unsafe { ffi::from_c_string(ffi::lsl_get_source_id(self.handle)) }
    }

    /// Get the unique stream ID.
    pub fn uid(&self) -> String {
        unsafe { ffi::from_c_string(ffi::lsl_get_uid(self.handle)) }
    }

    /// Get the hostname.
    pub fn hostname(&self) -> String {
        unsafe { ffi::from_c_string(ffi::lsl_get_hostname(self.handle)) }
    }

    /// Get the XML description.
    pub fn xml(&self) -> String {
        unsafe {
            let xml_ptr = ffi::lsl_get_xml(self.handle);
            let xml = ffi::from_c_string(xml_ptr);
            ffi::lsl_destroy_string(xml_ptr);
            xml
        }
    }
}

#[cfg(feature = "native")]
impl Drop for NativeStreamInfo {
    fn drop(&mut self) {
        if self.owned && !self.handle.is_null() {
            unsafe { ffi::lsl_destroy_streaminfo(self.handle) };
        }
    }
}

/// Native LSL outlet wrapper.
#[cfg(feature = "native")]
pub struct NativeOutlet {
    handle: ffi::lsl_outlet,
    channel_count: i32,
}

#[cfg(feature = "native")]
impl NativeOutlet {
    /// Create a new outlet.
    pub fn new(info: &NativeStreamInfo, chunk_size: i32, max_buffered: i32) -> Result<Self> {
        let handle = unsafe {
            ffi::lsl_create_outlet(info.as_raw(), chunk_size, max_buffered)
        };

        if handle.is_null() {
            return Err(LslError::Internal("Failed to create outlet".to_string()));
        }

        Ok(Self {
            handle,
            channel_count: info.channel_count(),
        })
    }

    /// Push a sample of f32 values.
    pub fn push_sample_f32(&self, data: &[f32]) -> Result<()> {
        if data.len() != self.channel_count as usize {
            return Err(LslError::Configuration(format!(
                "Expected {} channels, got {}",
                self.channel_count,
                data.len()
            )));
        }

        let result = unsafe { ffi::lsl_push_sample_f(self.handle, data.as_ptr()) };
        if result < 0 {
            ffi::check_error(result)
        } else {
            Ok(())
        }
    }

    /// Push a sample of f64 values.
    pub fn push_sample_f64(&self, data: &[f64]) -> Result<()> {
        if data.len() != self.channel_count as usize {
            return Err(LslError::Configuration(format!(
                "Expected {} channels, got {}",
                self.channel_count,
                data.len()
            )));
        }

        let result = unsafe { ffi::lsl_push_sample_d(self.handle, data.as_ptr()) };
        if result < 0 {
            ffi::check_error(result)
        } else {
            Ok(())
        }
    }

    /// Push a sample with timestamp.
    pub fn push_sample_f32_with_timestamp(&self, data: &[f32], timestamp: f64) -> Result<()> {
        if data.len() != self.channel_count as usize {
            return Err(LslError::Configuration(format!(
                "Expected {} channels, got {}",
                self.channel_count,
                data.len()
            )));
        }

        let result = unsafe { ffi::lsl_push_sample_ft(self.handle, data.as_ptr(), timestamp) };
        if result < 0 {
            ffi::check_error(result)
        } else {
            Ok(())
        }
    }

    /// Push a chunk of samples.
    pub fn push_chunk_f32(&self, data: &[f32]) -> Result<()> {
        let result = unsafe {
            ffi::lsl_push_chunk_f(self.handle, data.as_ptr(), data.len() as i64)
        };
        if result < 0 {
            ffi::check_error(result)
        } else {
            Ok(())
        }
    }

    /// Check if consumers are connected.
    pub fn have_consumers(&self) -> bool {
        unsafe { ffi::lsl_have_consumers(self.handle) != 0 }
    }

    /// Wait for consumers to connect.
    pub fn wait_for_consumers(&self, timeout: f64) -> bool {
        unsafe { ffi::lsl_wait_for_consumers(self.handle, timeout) != 0 }
    }
}

#[cfg(feature = "native")]
impl Drop for NativeOutlet {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::lsl_destroy_outlet(self.handle) };
        }
    }
}

/// Native LSL inlet wrapper.
#[cfg(feature = "native")]
pub struct NativeInlet {
    handle: ffi::lsl_inlet,
    channel_count: i32,
}

#[cfg(feature = "native")]
impl NativeInlet {
    /// Create a new inlet.
    pub fn new(
        info: &NativeStreamInfo,
        max_buflen: i32,
        max_chunklen: i32,
        recover: bool,
    ) -> Result<Self> {
        let handle = unsafe {
            ffi::lsl_create_inlet(
                info.as_raw(),
                max_buflen,
                max_chunklen,
                if recover { 1 } else { 0 },
            )
        };

        if handle.is_null() {
            return Err(LslError::Internal("Failed to create inlet".to_string()));
        }

        Ok(Self {
            handle,
            channel_count: info.channel_count(),
        })
    }

    /// Open the stream connection.
    pub fn open_stream(&self, timeout: f64) -> Result<()> {
        let mut errcode = 0;
        unsafe { ffi::lsl_open_stream(self.handle, timeout, &mut errcode) };
        ffi::check_error(errcode)
    }

    /// Close the stream connection.
    pub fn close_stream(&self) {
        unsafe { ffi::lsl_close_stream(self.handle) };
    }

    /// Pull a sample of f32 values.
    pub fn pull_sample_f32(&self, timeout: f64) -> Result<(Vec<f32>, f64)> {
        let mut buffer = vec![0.0f32; self.channel_count as usize];
        let mut errcode = 0;

        let timestamp = unsafe {
            ffi::lsl_pull_sample_f(
                self.handle,
                buffer.as_mut_ptr(),
                self.channel_count,
                timeout,
                &mut errcode,
            )
        };

        ffi::check_error(errcode)?;
        Ok((buffer, timestamp))
    }

    /// Pull a sample of f64 values.
    pub fn pull_sample_f64(&self, timeout: f64) -> Result<(Vec<f64>, f64)> {
        let mut buffer = vec![0.0f64; self.channel_count as usize];
        let mut errcode = 0;

        let timestamp = unsafe {
            ffi::lsl_pull_sample_d(
                self.handle,
                buffer.as_mut_ptr(),
                self.channel_count,
                timeout,
                &mut errcode,
            )
        };

        ffi::check_error(errcode)?;
        Ok((buffer, timestamp))
    }

    /// Pull a chunk of samples.
    pub fn pull_chunk_f32(&self, max_samples: usize, timeout: f64) -> Result<(Vec<f32>, Vec<f64>)> {
        let buffer_size = max_samples * self.channel_count as usize;
        let mut data_buffer = vec![0.0f32; buffer_size];
        let mut timestamp_buffer = vec![0.0f64; max_samples];
        let mut errcode = 0;

        let samples_read = unsafe {
            ffi::lsl_pull_chunk_f(
                self.handle,
                data_buffer.as_mut_ptr(),
                timestamp_buffer.as_mut_ptr(),
                buffer_size as i64,
                max_samples as i64,
                timeout,
                &mut errcode,
            )
        };

        ffi::check_error(errcode)?;

        let actual_samples = (samples_read as usize) / self.channel_count as usize;
        data_buffer.truncate(samples_read as usize);
        timestamp_buffer.truncate(actual_samples);

        Ok((data_buffer, timestamp_buffer))
    }

    /// Get the number of available samples.
    pub fn samples_available(&self) -> i32 {
        unsafe { ffi::lsl_samples_available(self.handle) }
    }

    /// Get the time correction offset.
    pub fn time_correction(&self, timeout: f64) -> Result<f64> {
        let mut errcode = 0;
        let offset = unsafe { ffi::lsl_time_correction(self.handle, timeout, &mut errcode) };
        ffi::check_error(errcode)?;
        Ok(offset)
    }

    /// Set post-processing options.
    pub fn set_postprocessing(&self, flags: i32) -> Result<()> {
        let result = unsafe { ffi::lsl_set_postprocessing(self.handle, flags) };
        if result < 0 {
            ffi::check_error(result)
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "native")]
impl Drop for NativeInlet {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::lsl_destroy_inlet(self.handle) };
        }
    }
}

/// Resolve streams on the network.
#[cfg(feature = "native")]
pub fn resolve_streams(timeout: f64) -> Result<Vec<NativeStreamInfo>> {
    const MAX_STREAMS: usize = 1024;
    let mut buffer: Vec<ffi::lsl_streaminfo> = vec![std::ptr::null_mut(); MAX_STREAMS];

    let count = unsafe {
        ffi::lsl_resolve_all(buffer.as_mut_ptr(), MAX_STREAMS as i32, timeout)
    };

    if count < 0 {
        return ffi::check_error(count).map(|_| vec![]);
    }

    let streams = buffer
        .into_iter()
        .take(count as usize)
        .filter(|h| !h.is_null())
        .map(|h| unsafe { NativeStreamInfo::from_raw(h) })
        .collect();

    Ok(streams)
}

/// Resolve streams by property.
#[cfg(feature = "native")]
pub fn resolve_by_property(prop: &str, value: &str, timeout: f64) -> Result<Vec<NativeStreamInfo>> {
    const MAX_STREAMS: usize = 1024;
    let mut buffer: Vec<ffi::lsl_streaminfo> = vec![std::ptr::null_mut(); MAX_STREAMS];

    let c_prop = CString::new(prop).map_err(|e| LslError::Configuration(e.to_string()))?;
    let c_value = CString::new(value).map_err(|e| LslError::Configuration(e.to_string()))?;

    let count = unsafe {
        ffi::lsl_resolve_byprop(
            buffer.as_mut_ptr(),
            MAX_STREAMS as i32,
            c_prop.as_ptr(),
            c_value.as_ptr(),
            1, // minimum
            timeout,
        )
    };

    if count < 0 {
        return ffi::check_error(count).map(|_| vec![]);
    }

    let streams = buffer
        .into_iter()
        .take(count as usize)
        .filter(|h| !h.is_null())
        .map(|h| unsafe { NativeStreamInfo::from_raw(h) })
        .collect();

    Ok(streams)
}

/// Get the local clock value.
#[cfg(feature = "native")]
pub fn local_clock() -> f64 {
    unsafe { ffi::lsl_local_clock() }
}

/// Get the library version.
#[cfg(feature = "native")]
pub fn library_version() -> i32 {
    unsafe { ffi::lsl_library_version() }
}

/// Get the protocol version.
#[cfg(feature = "native")]
pub fn protocol_version() -> i32 {
    unsafe { ffi::lsl_protocol_version() }
}

/// Native continuous resolver for background stream discovery.
#[cfg(feature = "native")]
pub struct NativeContinuousResolver {
    handle: ffi::lsl_continuous_resolver,
}

#[cfg(feature = "native")]
impl NativeContinuousResolver {
    /// Create a new continuous resolver.
    pub fn new(forget_after: f64) -> Self {
        let handle = unsafe { ffi::lsl_create_continuous_resolver(forget_after) };
        Self { handle }
    }

    /// Create a resolver for a specific property.
    pub fn with_property(prop: &str, value: &str, forget_after: f64) -> Result<Self> {
        let c_prop = CString::new(prop).map_err(|e| LslError::Configuration(e.to_string()))?;
        let c_value = CString::new(value).map_err(|e| LslError::Configuration(e.to_string()))?;

        let handle = unsafe {
            ffi::lsl_create_continuous_resolver_byprop(
                c_prop.as_ptr(),
                c_value.as_ptr(),
                forget_after,
            )
        };

        Ok(Self { handle })
    }

    /// Get currently resolved streams.
    pub fn results(&self) -> Vec<NativeStreamInfo> {
        const MAX_STREAMS: usize = 1024;
        let mut buffer: Vec<ffi::lsl_streaminfo> = vec![std::ptr::null_mut(); MAX_STREAMS];

        let count = unsafe {
            ffi::lsl_resolver_results(self.handle, buffer.as_mut_ptr(), MAX_STREAMS as i32)
        };

        if count <= 0 {
            return vec![];
        }

        buffer
            .into_iter()
            .take(count as usize)
            .filter(|h| !h.is_null())
            .map(|h| unsafe { NativeStreamInfo::from_raw(h) })
            .collect()
    }
}

#[cfg(feature = "native")]
impl Drop for NativeContinuousResolver {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ffi::lsl_destroy_continuous_resolver(self.handle) };
        }
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    #[test]
    fn test_library_version() {
        let version = library_version();
        assert!(version > 0);
    }

    #[test]
    fn test_local_clock() {
        let t1 = local_clock();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = local_clock();
        assert!(t2 > t1);
    }

    #[test]
    fn test_stream_info_creation() {
        let info = NativeStreamInfo::new(
            "TestStream",
            "EEG",
            8,
            256.0,
            ChannelFormat::Float32,
            "test-source-001",
        )
        .unwrap();

        assert_eq!(info.name(), "TestStream");
        assert_eq!(info.stream_type(), "EEG");
        assert_eq!(info.channel_count(), 8);
        assert!((info.nominal_srate() - 256.0).abs() < 0.001);
    }
}
