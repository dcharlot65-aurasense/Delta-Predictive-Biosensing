//! # DPB FFI - C-Compatible Foreign Function Interface
//!
//! This crate provides a C-compatible API for the DPB framework, enabling
//! integration with Julia, MATLAB, Python (via ctypes), and other languages.
//!
//! ## Safety
//!
//! All exported functions use `#[unsafe(no_mangle)]` and `extern "C"` for C compatibility.
//! Panics are caught and converted to error codes. Memory management follows
//! strict ownership rules:
//!
//! - Objects created by `*_new()` must be freed by `*_free()`
//! - Pointers returned by the library remain valid until freed
//! - The caller must not modify or free library-owned memory
//!
//! ## Error Handling
//!
//! Errors are stored in thread-local storage and can be retrieved via
//! `dpb_last_error()`. Always check return values and error codes.
//!
//! ## Example (C)
//!
//! ```c
//! // Create a time series
//! float data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
//! DpbTimeSeries* ts = dpb_timeseries_new(data, 5, 1, 1000.0);
//! if (!ts) {
//!     printf("Error: %s\n", dpb_last_error());
//!     return 1;
//! }
//!
//! // Get duration
//! double duration = dpb_timeseries_duration(ts);
//! printf("Duration: %f seconds\n", duration);
//!
//! // Clean up
//! dpb_timeseries_free(ts);
//! ```

#![warn(missing_docs)]
#![allow(clippy::missing_safety_doc)]

pub mod types;

use libc::{c_char, c_double, size_t};
use std::ffi::CString;
use std::panic;
use std::ptr;
use std::slice;

use dpb_core::traits::{EventEncoder, Signal};
use dpb_core::{SpikeEvent, SpikeTrain};
use dpb_encoders::base::LevelCrossingConfig;

use types::{DpbEncoder, DpbErrorCode, DpbSpikeTrain, DpbTimeSeries};

// Thread-local storage for error messages
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Sets the last error message.
fn set_last_error(error: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(error);
    });
}

/// Clears the last error message.
fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

/// Catches panics and converts them to error codes.
fn catch_panic<F, T>(f: F, default: T) -> T
where
    F: FnOnce() -> T + panic::UnwindSafe,
{
    match panic::catch_unwind(f) {
        Ok(result) => result,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                format!("Panic: {}", s)
            } else if let Some(s) = e.downcast_ref::<String>() {
                format!("Panic: {}", s)
            } else {
                "Unknown panic".to_string()
            };
            set_last_error(msg);
            default
        }
    }
}

// ============================================================================
// Version Information
// ============================================================================

/// Returns the DPB framework version string.
///
/// The returned pointer is valid for the lifetime of the program.
/// Do NOT free this pointer.
///
/// # Safety
///
/// The returned pointer is to static data and must not be freed.
#[unsafe(no_mangle)]
pub extern "C" fn dpb_version() -> *const c_char {
    static VERSION: &str = concat!("DPB Framework v", env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

// ============================================================================
// Error Handling
// ============================================================================

/// Returns the last error message, or NULL if there is no error.
///
/// The returned string is valid until the next call to any DPB function
/// on the same thread. Do NOT free this pointer.
///
/// # Safety
///
/// The returned pointer is valid until the next DPB function call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn dpb_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        if let Some(ref error) = *e.borrow() {
            // This is a bit tricky - we need to return a C string that outlives this function
            // We'll use a thread-local static to store the C string
            thread_local! {
                static ERROR_CSTRING: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
            }

            ERROR_CSTRING.with(|cs| {
                *cs.borrow_mut() = CString::new(error.as_str()).ok();
                if let Some(ref cstring) = *cs.borrow() {
                    cstring.as_ptr()
                } else {
                    ptr::null()
                }
            })
        } else {
            ptr::null()
        }
    })
}

/// Clears the last error message.
#[unsafe(no_mangle)]
pub extern "C" fn dpb_clear_error() {
    clear_last_error();
}

// ============================================================================
// Memory Management
// ============================================================================

/// Frees a string allocated by the DPB library.
///
/// # Safety
///
/// The pointer must have been returned by a DPB function that allocates strings.
/// Do not call this on strings returned by `dpb_version()` or `dpb_last_error()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_free_string(s: *mut c_char) {
    catch_panic(
        || {
            if !s.is_null() {
                unsafe {
                    drop(CString::from_raw(s));
                }
            }
        },
        (),
    );
}

/// Frees an array allocated by the DPB library.
///
/// # Safety
///
/// The pointer must have been returned by a DPB function that allocates arrays.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_free_array(ptr: *mut f32) {
    catch_panic(
        || {
            if !ptr.is_null() {
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        },
        (),
    );
}

// ============================================================================
// TimeSeries Functions
// ============================================================================

/// Creates a new TimeSeries object.
///
/// # Parameters
///
/// - `data`: Pointer to sample data (will be copied)
/// - `num_samples`: Total number of samples
/// - `num_channels`: Number of channels
/// - `sample_rate`: Sampling rate in Hz
///
/// # Returns
///
/// Pointer to a new DpbTimeSeries, or NULL on error.
///
/// # Safety
///
/// - `data` must point to at least `num_samples` valid f32 values
/// - The returned pointer must be freed with `dpb_timeseries_free()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_timeseries_new(
    data: *const f32,
    num_samples: size_t,
    num_channels: size_t,
    sample_rate: c_double,
) -> *mut DpbTimeSeries {
    catch_panic(
        || {
            clear_last_error();

            if data.is_null() {
                set_last_error("Data pointer is null".to_string());
                return ptr::null_mut();
            }

            if num_samples == 0 {
                set_last_error("Number of samples must be greater than 0".to_string());
                return ptr::null_mut();
            }

            if num_channels == 0 {
                set_last_error("Number of channels must be greater than 0".to_string());
                return ptr::null_mut();
            }

            if sample_rate <= 0.0 {
                set_last_error("Sample rate must be positive".to_string());
                return ptr::null_mut();
            }

            // Copy the data
            let data_vec = unsafe {
                let data_slice = slice::from_raw_parts(data, num_samples);
                data_slice.to_vec()
            };

            // Create the TimeSeries
            let ts = DpbTimeSeries::new(data_vec, num_channels, sample_rate);

            Box::into_raw(Box::new(ts))
        },
        ptr::null_mut(),
    )
}

/// Frees a TimeSeries object.
///
/// # Safety
///
/// The pointer must have been created by `dpb_timeseries_new()`.
/// Do not use the pointer after calling this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_timeseries_free(ts: *mut DpbTimeSeries) {
    catch_panic(
        || {
            if !ts.is_null() {
                unsafe {
                    drop(Box::from_raw(ts));
                }
            }
        },
        (),
    );
}

/// Returns the duration of the time series in seconds.
///
/// # Safety
///
/// `ts` must be a valid pointer to a DpbTimeSeries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_timeseries_duration(ts: *const DpbTimeSeries) -> c_double {
    catch_panic(
        || {
            if ts.is_null() {
                set_last_error("TimeSeries pointer is null".to_string());
                return 0.0;
            }

            unsafe {
                let ts = &*ts;
                let num_samples = ts.buffer.data.len() / ts.buffer.num_channels;
                num_samples as f64 / ts.buffer.sample_rate
            }
        },
        0.0,
    )
}

/// Returns the number of samples in the time series.
///
/// # Safety
///
/// `ts` must be a valid pointer to a DpbTimeSeries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_timeseries_num_samples(ts: *const DpbTimeSeries) -> size_t {
    catch_panic(
        || {
            if ts.is_null() {
                set_last_error("TimeSeries pointer is null".to_string());
                return 0;
            }

            unsafe {
                let ts = &*ts;
                ts.buffer.data.len() / ts.buffer.num_channels
            }
        },
        0,
    )
}

/// Returns the number of channels in the time series.
///
/// # Safety
///
/// `ts` must be a valid pointer to a DpbTimeSeries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_timeseries_num_channels(ts: *const DpbTimeSeries) -> size_t {
    catch_panic(
        || {
            if ts.is_null() {
                set_last_error("TimeSeries pointer is null".to_string());
                return 0;
            }

            unsafe {
                let ts = &*ts;
                ts.buffer.num_channels
            }
        },
        0,
    )
}

/// Returns the sample rate of the time series in Hz.
///
/// # Safety
///
/// `ts` must be a valid pointer to a DpbTimeSeries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_timeseries_sample_rate(ts: *const DpbTimeSeries) -> c_double {
    catch_panic(
        || {
            if ts.is_null() {
                set_last_error("TimeSeries pointer is null".to_string());
                return 0.0;
            }

            unsafe {
                let ts = &*ts;
                ts.buffer.sample_rate
            }
        },
        0.0,
    )
}

/// Returns a pointer to the raw sample data.
///
/// The data is stored in interleaved format (channel 0 sample 0, channel 1 sample 0, ...).
/// Do NOT free this pointer - it is owned by the TimeSeries object.
///
/// # Safety
///
/// - `ts` must be a valid pointer to a DpbTimeSeries
/// - The returned pointer is valid until the TimeSeries is freed
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_timeseries_get_data(ts: *const DpbTimeSeries) -> *const f32 {
    catch_panic(
        || {
            if ts.is_null() {
                set_last_error("TimeSeries pointer is null".to_string());
                return ptr::null();
            }

            unsafe {
                let ts = &*ts;
                ts.buffer.data.as_ptr()
            }
        },
        ptr::null(),
    )
}

// ============================================================================
// SpikeTrain Functions
// ============================================================================

/// Creates a new SpikeTrain object.
///
/// # Parameters
///
/// - `num_channels`: Number of channels
///
/// # Returns
///
/// Pointer to a new DpbSpikeTrain, or NULL on error.
///
/// # Safety
///
/// The returned pointer must be freed with `dpb_spike_train_free()`.
#[unsafe(no_mangle)]
pub extern "C" fn dpb_spike_train_new(num_channels: u32) -> *mut DpbSpikeTrain {
    catch_panic(
        || {
            clear_last_error();

            if num_channels == 0 {
                set_last_error("Number of channels must be greater than 0".to_string());
                return ptr::null_mut();
            }

            let st = DpbSpikeTrain::new(num_channels);
            Box::into_raw(Box::new(st))
        },
        ptr::null_mut(),
    )
}

/// Frees a SpikeTrain object.
///
/// # Safety
///
/// The pointer must have been created by `dpb_spike_train_new()`.
/// Do not use the pointer after calling this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_spike_train_free(st: *mut DpbSpikeTrain) {
    catch_panic(
        || {
            if !st.is_null() {
                unsafe {
                    drop(Box::from_raw(st));
                }
            }
        },
        (),
    );
}

/// Adds a spike event to the spike train.
///
/// # Parameters
///
/// - `st`: Pointer to the SpikeTrain
/// - `timestamp`: Time of the spike in seconds
/// - `channel`: Channel/neuron index
/// - `polarity`: Spike polarity (-1 or +1)
///
/// # Returns
///
/// 0 on success, non-zero on error.
///
/// # Safety
///
/// `st` must be a valid pointer to a DpbSpikeTrain.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_spike_train_add_event(
    st: *mut DpbSpikeTrain,
    timestamp: c_double,
    channel: u32,
    polarity: i8,
) -> i32 {
    catch_panic(
        || {
            if st.is_null() {
                set_last_error("SpikeTrain pointer is null".to_string());
                return DpbErrorCode::NullPointer as i32;
            }

            if polarity != -1 && polarity != 1 {
                set_last_error("Polarity must be -1 or +1".to_string());
                return DpbErrorCode::InvalidParameter as i32;
            }

            unsafe {
                let st = &mut *st;
                let event = SpikeEvent::new(timestamp, channel, polarity, 1.0);
                st.train_mut().add_event(event);

                DpbErrorCode::Success as i32
            }
        },
        DpbErrorCode::Unknown as i32,
    )
}

/// Returns the number of spike events in the train.
///
/// # Safety
///
/// `st` must be a valid pointer to a DpbSpikeTrain.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_spike_train_len(st: *const DpbSpikeTrain) -> size_t {
    catch_panic(
        || {
            if st.is_null() {
                set_last_error("SpikeTrain pointer is null".to_string());
                return 0;
            }

            unsafe {
                let st = &*st;
                st.train().len()
            }
        },
        0,
    )
}

/// Returns the number of channels in the spike train.
///
/// # Safety
///
/// `st` must be a valid pointer to a DpbSpikeTrain.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_spike_train_num_channels(st: *const DpbSpikeTrain) -> u32 {
    catch_panic(
        || {
            if st.is_null() {
                set_last_error("SpikeTrain pointer is null".to_string());
                return 0;
            }

            unsafe {
                let st = &*st;
                st.train().num_channels
            }
        },
        0,
    )
}

/// Gets a spike event from the train by index.
///
/// # Parameters
///
/// - `st`: Pointer to the SpikeTrain
/// - `index`: Event index
/// - `timestamp`: Output parameter for timestamp (can be NULL)
/// - `channel`: Output parameter for channel (can be NULL)
/// - `polarity`: Output parameter for polarity (can be NULL)
/// - `magnitude`: Output parameter for magnitude (can be NULL)
///
/// # Returns
///
/// 0 on success, non-zero on error.
///
/// # Safety
///
/// - `st` must be a valid pointer to a DpbSpikeTrain
/// - Output parameters must be valid pointers or NULL
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_spike_train_get_event(
    st: *const DpbSpikeTrain,
    index: size_t,
    timestamp: *mut c_double,
    channel: *mut u32,
    polarity: *mut i8,
    magnitude: *mut f32,
) -> i32 {
    catch_panic(
        || {
            if st.is_null() {
                set_last_error("SpikeTrain pointer is null".to_string());
                return DpbErrorCode::NullPointer as i32;
            }

            unsafe {
                let st = &*st;
                if index >= st.train().len() {
                    set_last_error(format!("Index {} out of bounds", index));
                    return DpbErrorCode::InvalidParameter as i32;
                }

                let event = &st.train().events[index];

                if !timestamp.is_null() {
                    *timestamp = event.timestamp;
                }
                if !channel.is_null() {
                    *channel = event.channel;
                }
                if !polarity.is_null() {
                    *polarity = event.polarity;
                }
                if !magnitude.is_null() {
                    *magnitude = event.magnitude;
                }

                DpbErrorCode::Success as i32
            }
        },
        DpbErrorCode::Unknown as i32,
    )
}

// ============================================================================
// Encoder Functions
// ============================================================================

/// Creates a new level-crossing encoder.
///
/// # Parameters
///
/// - `threshold`: Threshold value for level crossing detection. Must be
///   finite and positive; anything else returns NULL with the reason in
///   `dpb_last_error()`.
///
/// # Returns
///
/// Pointer to a new DpbEncoder, or NULL on error.
///
/// # Safety
///
/// The returned pointer must be freed with `dpb_encoder_free()`.
#[unsafe(no_mangle)]
pub extern "C" fn dpb_encoder_level_crossing_new(threshold: c_double) -> *mut DpbEncoder {
    catch_panic(
        || {
            clear_last_error();

            match DpbEncoder::new_level_crossing(threshold) {
                Ok(encoder) => Box::into_raw(Box::new(encoder)),
                Err(message) => {
                    set_last_error(message);
                    ptr::null_mut()
                }
            }
        },
        ptr::null_mut(),
    )
}

/// Frees an Encoder object.
///
/// # Safety
///
/// The pointer must have been created by an encoder constructor.
/// Do not use the pointer after calling this function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_encoder_free(enc: *mut DpbEncoder) {
    catch_panic(
        || {
            if !enc.is_null() {
                unsafe {
                    drop(Box::from_raw(enc));
                }
            }
        },
        (),
    );
}

/// Encodes a time series into a spike train.
///
/// # Parameters
///
/// - `enc`: Pointer to the encoder
/// - `ts`: Pointer to the time series to encode
///
/// # Returns
///
/// Pointer to a new DpbSpikeTrain, or NULL on error.
///
/// # Safety
///
/// - `enc` and `ts` must be valid pointers
/// - The returned SpikeTrain must be freed with `dpb_spike_train_free()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dpb_encoder_encode(
    enc: *const DpbEncoder,
    ts: *const DpbTimeSeries,
) -> *mut DpbSpikeTrain {
    catch_panic(
        || {
            if enc.is_null() {
                set_last_error("Encoder pointer is null".to_string());
                return ptr::null_mut();
            }

            if ts.is_null() {
                set_last_error("TimeSeries pointer is null".to_string());
                return ptr::null_mut();
            }

            unsafe {
                let enc = &*enc;
                let ts = &*ts;

                // Encode based on encoder type
                let events = match &enc.encoder {
                    types::EncoderType::LevelCrossing { encoder, threshold } => {
                        let config = LevelCrossingConfig {
                            threshold: *threshold,
                            relative: false,
                            refractory_period: 0.001,
                            // Reference-tracking mode: the only one with a
                            // bounded reconstruction error, so it is what a
                            // C ABI caller should get by default.
                            ..Default::default()
                        };

                        match encoder.encode(&ts.buffer as &dyn Signal, &config) {
                            Ok(events) => events,
                            Err(e) => {
                                set_last_error(format!("Encoding failed: {}", e));
                                return ptr::null_mut();
                            }
                        }
                    }
                };

                // Convert Vec<SpikeEvent> to SpikeTrain
                let num_channels = ts.buffer.num_channels as u32;
                let train = SpikeTrain::from_events(events, num_channels);
                let spike_train = DpbSpikeTrain::from_train(train);
                Box::into_raw(Box::new(spike_train))
            }
        },
        ptr::null_mut(),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes `data` through the public C API at `threshold`, returning the
    /// number of events, or `None` if the encoder refused the threshold.
    unsafe fn events_at(data: &[f32], threshold: f64) -> Option<usize> {
        unsafe {
            let ts = dpb_timeseries_new(data.as_ptr(), data.len(), 1, 1000.0);
            assert!(!ts.is_null());
            let enc = dpb_encoder_level_crossing_new(threshold);
            if enc.is_null() {
                dpb_timeseries_free(ts);
                return None;
            }
            let train = dpb_encoder_encode(enc, ts);
            assert!(!train.is_null(), "encoding a valid signal failed");
            let n = dpb_spike_train_len(train);
            dpb_spike_train_free(train);
            dpb_encoder_free(enc);
            dpb_timeseries_free(ts);
            Some(n)
        }
    }

    /// The threshold a C caller passes has to reach the encoder.
    ///
    /// It used to be dropped in the constructor and replaced by a hardcoded 0.5
    /// at encode time. The bundled C example passes exactly 0.5, so it could
    /// never have noticed; this test deliberately uses thresholds other than
    /// 0.5 and checks the event counts they must produce.
    #[test]
    fn level_crossing_honours_the_callers_threshold() {
        // A ramp from 0 to 3 in steps of 0.01: in delta mode an event fires each
        // time the signal has moved one threshold from the last emitted level.
        let ramp: Vec<f32> = (0..=300).map(|i| i as f32 * 0.01).collect();

        let fine = unsafe { events_at(&ramp, 0.1) }.unwrap();
        let coarse = unsafe { events_at(&ramp, 1.0) }.unwrap();

        // 3.0 of travel is 30 quanta of 0.1 and 3 quanta of 1.0. The old code
        // used 0.5 for both, giving 6 each.
        assert!(
            (29..=30).contains(&fine),
            "threshold 0.1 gave {fine} events"
        );
        assert_eq!(coarse, 3, "threshold 1.0 gave {coarse} events");
        assert_ne!(fine, coarse, "the threshold had no effect");
    }

    /// Thresholds that would silently produce an empty spike train are refused.
    #[test]
    fn level_crossing_rejects_thresholds_that_encode_nothing() {
        let ramp: Vec<f32> = (0..=300).map(|i| i as f32 * 0.01).collect();
        for bad in [0.0, -0.5, f64::NAN, f64::INFINITY, 1e300] {
            assert_eq!(
                unsafe { events_at(&ramp, bad) },
                None,
                "threshold {bad} was accepted"
            );
        }
    }

    #[test]
    fn test_version() {
        let version = dpb_version();
        assert!(!version.is_null());
    }

    #[test]
    fn test_timeseries_lifecycle() {
        unsafe {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
            let ts = dpb_timeseries_new(data.as_ptr(), 5, 1, 1000.0);
            assert!(!ts.is_null());

            let duration = dpb_timeseries_duration(ts);
            assert!((duration - 0.005).abs() < 1e-6);

            let num_samples = dpb_timeseries_num_samples(ts);
            assert_eq!(num_samples, 5);

            let num_channels = dpb_timeseries_num_channels(ts);
            assert_eq!(num_channels, 1);

            dpb_timeseries_free(ts);
        }
    }

    #[test]
    fn test_spike_train_lifecycle() {
        unsafe {
            let st = dpb_spike_train_new(10);
            assert!(!st.is_null());

            let result = dpb_spike_train_add_event(st, 0.001, 5, 1);
            assert_eq!(result, DpbErrorCode::Success as i32);

            let len = dpb_spike_train_len(st);
            assert_eq!(len, 1);

            let mut timestamp = 0.0;
            let mut channel = 0;
            let mut polarity = 0;
            let result = dpb_spike_train_get_event(
                st,
                0,
                &mut timestamp,
                &mut channel,
                &mut polarity,
                ptr::null_mut(),
            );
            assert_eq!(result, DpbErrorCode::Success as i32);
            assert_eq!(timestamp, 0.001);
            assert_eq!(channel, 5);
            assert_eq!(polarity, 1);

            dpb_spike_train_free(st);
        }
    }
}
