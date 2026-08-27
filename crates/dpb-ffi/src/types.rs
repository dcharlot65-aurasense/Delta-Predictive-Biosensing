//! Opaque handle types for FFI.
//!
//! These types are opaque to C clients and can only be manipulated through
//! the exported FFI functions. This ensures memory safety and proper lifetime
//! management across the FFI boundary.

use dpb_core::{SignalBuffer, SpikeTrain};
use dpb_encoders::base::LevelCrossingEncoder;
use std::boxed::Box;

/// Opaque handle to a TimeSeries object.
///
/// This wraps a SignalBuffer (f32-based) for FFI compatibility.
/// C clients receive a pointer to this type but cannot access its contents directly.
#[repr(C)]
pub struct DpbTimeSeries {
    /// Internal buffer - uses f32 for better C compatibility
    pub(crate) buffer: SignalBuffer,
}

impl DpbTimeSeries {
    /// Creates a new TimeSeries from raw data.
    pub fn new(data: Vec<f32>, num_channels: usize, sample_rate: f64) -> Self {
        let buffer = SignalBuffer {
            data,
            num_channels,
            sample_rate,
        };
        Self { buffer }
    }

    /// Gets the underlying SignalBuffer.
    pub fn buffer(&self) -> &SignalBuffer {
        &self.buffer
    }

    /// Gets a mutable reference to the underlying SignalBuffer.
    pub fn buffer_mut(&mut self) -> &mut SignalBuffer {
        &mut self.buffer
    }
}

/// Opaque handle to a SpikeTrain object.
///
/// C clients receive a pointer to this type but cannot access its contents directly.
#[repr(C)]
pub struct DpbSpikeTrain {
    pub(crate) train: SpikeTrain,
}

impl DpbSpikeTrain {
    /// Creates a new SpikeTrain.
    pub fn new(num_channels: u32) -> Self {
        Self {
            train: SpikeTrain::new(num_channels),
        }
    }

    /// Creates from an existing SpikeTrain.
    pub fn from_train(train: SpikeTrain) -> Self {
        Self { train }
    }

    /// Gets the underlying SpikeTrain.
    pub fn train(&self) -> &SpikeTrain {
        &self.train
    }

    /// Gets a mutable reference to the underlying SpikeTrain.
    pub fn train_mut(&mut self) -> &mut SpikeTrain {
        &mut self.train
    }
}

/// Opaque handle to an Encoder object.
///
/// This is a trait object that can hold any encoder type.
#[repr(C)]
pub struct DpbEncoder {
    pub(crate) encoder: EncoderType,
}

/// Internal enum to hold different encoder types.
pub(crate) enum EncoderType {
    LevelCrossing(Box<LevelCrossingEncoder>),
    // Add more encoder types as needed
}

#[allow(dead_code)] // mirrors the modelled surface; this file uses a subset
impl DpbEncoder {
    /// Creates a new level-crossing encoder.
    pub fn new_level_crossing(_threshold: f64) -> Self {
        // Note: threshold is currently not used as LevelCrossingEncoder
        // takes it in the encode() call via config, not in new()
        let encoder = LevelCrossingEncoder::new("ffi_encoder");
        Self {
            encoder: EncoderType::LevelCrossing(Box::new(encoder)),
        }
    }

    /// Gets the encoder type.
    // EncoderType is deliberately crate-internal, so this accessor matches
    // its visibility rather than leaking a private type through a pub fn.
    pub(crate) fn encoder(&self) -> &EncoderType {
        &self.encoder
    }
}

/// Represents an error code for FFI operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpbErrorCode {
    /// Operation succeeded
    Success = 0,
    /// Null pointer was passed
    NullPointer = 1,
    /// Invalid parameter
    InvalidParameter = 2,
    /// Memory allocation failed
    AllocationFailed = 3,
    /// Invalid dimensions
    InvalidDimensions = 4,
    /// Encoding failed
    EncodingFailed = 5,
    /// Unknown error
    Unknown = 99,
}
