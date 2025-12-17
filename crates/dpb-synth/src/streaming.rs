//! Real-time streaming infrastructure for synthetic data generation
//!
//! This module provides traits and utilities for generating synthetic biosignals
//! in real-time, sample-by-sample or frame-by-frame, rather than all at once.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     StreamingGenerator Trait                             │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
//! │  │ init_state()│ -> │ next_sample │ -> │ next_sample │ -> ...          │
//! │  └─────────────┘    └─────────────┘    └─────────────┘                 │
//! │         │                  │                  │                         │
//! │         ▼                  ▼                  ▼                         │
//! │  ┌─────────────────────────────────────────────────────┐               │
//! │  │              RingBuffer (bounded memory)            │               │
//! │  │  [s0][s1][s2][s3][s4][s5][s6][s7] ... [sN]         │               │
//! │  │   ↑ write_pos              ↑ read_pos              │               │
//! │  └─────────────────────────────────────────────────────┘               │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use dpb_synth::streaming::{StreamingGenerator, RingBuffer};
//!
//! // Initialize generator state
//! let mut state = generator.init_state(&params, seed);
//!
//! // Create ring buffer for 1 second at 1000 Hz
//! let mut buffer = RingBuffer::new(1000);
//!
//! // Generate samples in real-time
//! loop {
//!     let sample = generator.next_sample(&mut state);
//!     buffer.push(sample);
//!
//!     // Process when we have enough samples
//!     if buffer.len() >= 256 {
//!         let batch = buffer.read_batch(256);
//!         process(batch);
//!     }
//! }
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Trait for generators that can produce samples incrementally
///
/// Unlike `SyntheticGenerator` which produces all data at once,
/// `StreamingGenerator` maintains state and produces one sample/frame at a time.
pub trait StreamingGenerator: Send + Sync {
    /// Generator state that persists between samples
    type State: Send;

    /// Parameters for configuring the generator
    type Parameters: Clone + Send;

    /// Output type for a single sample (e.g., f64 for ECG, [f64; 3] for 3D position)
    type Sample: Clone + Send;

    /// Initialize generator state from parameters
    ///
    /// This sets up all internal state needed for streaming generation.
    fn init_state(&self, params: &Self::Parameters, seed: u64) -> Self::State;

    /// Generate the next sample, advancing internal state
    ///
    /// This is the core streaming method - call repeatedly to get samples.
    fn next_sample(&self, state: &mut Self::State) -> Self::Sample;

    /// Generate multiple samples at once (batch optimization)
    ///
    /// Default implementation calls `next_sample` repeatedly.
    fn next_samples(&self, state: &mut Self::State, count: usize) -> Vec<Self::Sample> {
        (0..count).map(|_| self.next_sample(state)).collect()
    }

    /// Get the current time/position in the stream
    fn current_time(&self, state: &Self::State) -> f64;

    /// Get the sampling rate in Hz
    fn sampling_rate(&self, params: &Self::Parameters) -> f64;

    /// Reset state to initial conditions (for looping/restart)
    fn reset_state(&self, state: &mut Self::State, params: &Self::Parameters, seed: u64);

    /// Check if the generator has a defined end (finite duration)
    fn is_finite(&self, _params: &Self::Parameters) -> bool {
        false // Default: infinite streaming
    }

    /// Check if the stream has ended (for finite generators)
    fn is_complete(&self, _state: &Self::State, _params: &Self::Parameters) -> bool {
        false
    }
}

/// Trait for frame-based streaming (video, pose data)
pub trait FrameStreamingGenerator: Send + Sync {
    /// Generator state
    type State: Send;

    /// Frame parameters
    type Parameters: Clone + Send;

    /// Single frame output (e.g., Vec<[f64; 3]> for keypoints)
    type Frame: Clone + Send;

    /// Initialize state
    fn init_state(&self, params: &Self::Parameters, seed: u64) -> Self::State;

    /// Generate next frame
    fn next_frame(&self, state: &mut Self::State) -> Self::Frame;

    /// Current frame index
    fn current_frame(&self, state: &Self::State) -> usize;

    /// Frame rate in FPS
    fn frame_rate(&self, params: &Self::Parameters) -> f64;

    /// Reset to beginning
    fn reset_state(&self, state: &mut Self::State, params: &Self::Parameters, seed: u64);
}

/// Ring buffer for efficient streaming data storage
///
/// A fixed-size circular buffer that overwrites oldest data when full.
/// Optimized for streaming scenarios where we need a sliding window.
#[derive(Debug)]
pub struct RingBuffer<T> {
    buffer: Vec<Option<T>>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
    count: usize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        Self {
            buffer: vec![None; capacity],
            capacity,
            write_pos: 0,
            read_pos: 0,
            count: 0,
        }
    }

    /// Push a single value into the buffer
    ///
    /// If the buffer is full, overwrites the oldest value.
    pub fn push(&mut self, value: T) {
        self.buffer[self.write_pos] = Some(value);
        self.write_pos = (self.write_pos + 1) % self.capacity;

        if self.count < self.capacity {
            self.count += 1;
        } else {
            // Buffer is full, advance read position
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
    }

    /// Push multiple values into the buffer
    pub fn push_batch(&mut self, values: &[T]) {
        for value in values {
            self.push(value.clone());
        }
    }

    /// Pop the oldest value from the buffer
    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 {
            return None;
        }

        let value = self.buffer[self.read_pos].take();
        self.read_pos = (self.read_pos + 1) % self.capacity;
        self.count -= 1;
        value
    }

    /// Read a batch of values without removing them
    ///
    /// Returns up to `count` values starting from the oldest.
    pub fn read_batch(&self, count: usize) -> Vec<T> {
        let count = count.min(self.count);
        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            let idx = (self.read_pos + i) % self.capacity;
            if let Some(ref value) = self.buffer[idx] {
                result.push(value.clone());
            }
        }

        result
    }

    /// Read and remove a batch of values
    pub fn pop_batch(&mut self, count: usize) -> Vec<T> {
        let count = count.min(self.count);
        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(value) = self.pop() {
                result.push(value);
            }
        }

        result
    }

    /// Peek at the oldest value without removing it
    pub fn peek(&self) -> Option<&T> {
        if self.count == 0 {
            None
        } else {
            self.buffer[self.read_pos].as_ref()
        }
    }

    /// Peek at the newest value without removing it
    pub fn peek_newest(&self) -> Option<&T> {
        if self.count == 0 {
            None
        } else {
            let idx = if self.write_pos == 0 {
                self.capacity - 1
            } else {
                self.write_pos - 1
            };
            self.buffer[idx].as_ref()
        }
    }

    /// Get the number of elements in the buffer
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all elements from the buffer
    pub fn clear(&mut self) {
        for item in self.buffer.iter_mut() {
            *item = None;
        }
        self.write_pos = 0;
        self.read_pos = 0;
        self.count = 0;
    }

    /// Get all data as a contiguous vector (oldest first)
    pub fn to_vec(&self) -> Vec<T> {
        self.read_batch(self.count)
    }

    /// Get available space in the buffer
    pub fn available(&self) -> usize {
        self.capacity - self.count
    }
}

/// Thread-safe ring buffer using atomic operations
///
/// Suitable for single-producer, single-consumer (SPSC) scenarios.
#[derive(Debug)]
pub struct AtomicRingBuffer<T> {
    buffer: Vec<std::cell::UnsafeCell<Option<T>>>,
    capacity: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
}

// Safety: AtomicRingBuffer is safe to send between threads
unsafe impl<T: Send> Send for AtomicRingBuffer<T> {}
unsafe impl<T: Send> Sync for AtomicRingBuffer<T> {}

impl<T: Clone> AtomicRingBuffer<T> {
    /// Create a new atomic ring buffer
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "AtomicRingBuffer capacity must be > 0");
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(std::cell::UnsafeCell::new(None));
        }

        Self {
            buffer,
            capacity,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
        }
    }

    /// Try to push a value (non-blocking)
    ///
    /// Returns `false` if the buffer is full.
    pub fn try_push(&self, value: T) -> bool {
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        let next_write = (write_pos + 1) % self.capacity;

        // Check if buffer is full
        if next_write == read_pos {
            return false;
        }

        // Safety: We have exclusive access to this slot
        unsafe {
            *self.buffer[write_pos].get() = Some(value);
        }

        self.write_pos.store(next_write, Ordering::Release);
        true
    }

    /// Try to pop a value (non-blocking)
    ///
    /// Returns `None` if the buffer is empty.
    pub fn try_pop(&self) -> Option<T> {
        let read_pos = self.read_pos.load(Ordering::Relaxed);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        // Check if buffer is empty
        if read_pos == write_pos {
            return None;
        }

        // Safety: We have exclusive access to this slot
        let value = unsafe { (*self.buffer[read_pos].get()).take() };

        let next_read = (read_pos + 1) % self.capacity;
        self.read_pos.store(next_read, Ordering::Release);

        value
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.read_pos.load(Ordering::Relaxed) == self.write_pos.load(Ordering::Relaxed)
    }

    /// Get approximate length (may be slightly stale in concurrent scenarios)
    pub fn len(&self) -> usize {
        let write = self.write_pos.load(Ordering::Relaxed);
        let read = self.read_pos.load(Ordering::Relaxed);

        if write >= read {
            write - read
        } else {
            self.capacity - read + write
        }
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Multi-channel ring buffer for synchronized multi-modal streaming
#[derive(Debug)]
pub struct MultiChannelBuffer<T> {
    channels: Vec<RingBuffer<T>>,
    channel_names: Vec<String>,
}

impl<T: Clone> MultiChannelBuffer<T> {
    /// Create a new multi-channel buffer
    pub fn new(channel_names: Vec<String>, capacity_per_channel: usize) -> Self {
        let channels = channel_names
            .iter()
            .map(|_| RingBuffer::new(capacity_per_channel))
            .collect();

        Self {
            channels,
            channel_names,
        }
    }

    /// Push a value to a specific channel by index
    pub fn push_to_channel(&mut self, channel_idx: usize, value: T) {
        if channel_idx < self.channels.len() {
            self.channels[channel_idx].push(value);
        }
    }

    /// Push a value to a channel by name
    pub fn push_to_named_channel(&mut self, channel_name: &str, value: T) {
        if let Some(idx) = self.channel_names.iter().position(|n| n == channel_name) {
            self.channels[idx].push(value);
        }
    }

    /// Push synchronized values to all channels
    pub fn push_synchronized(&mut self, values: Vec<T>) {
        for (channel, value) in self.channels.iter_mut().zip(values) {
            channel.push(value);
        }
    }

    /// Read synchronized batch from all channels
    pub fn read_synchronized_batch(&self, count: usize) -> Vec<Vec<T>> {
        self.channels.iter().map(|c| c.read_batch(count)).collect()
    }

    /// Get channel by name
    pub fn channel(&self, name: &str) -> Option<&RingBuffer<T>> {
        self.channel_names
            .iter()
            .position(|n| n == name)
            .map(|idx| &self.channels[idx])
    }

    /// Get channel by name (mutable)
    pub fn channel_mut(&mut self, name: &str) -> Option<&mut RingBuffer<T>> {
        if let Some(idx) = self.channel_names.iter().position(|n| n == name) {
            Some(&mut self.channels[idx])
        } else {
            None
        }
    }

    /// Get number of channels
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Get channel names
    pub fn channel_names(&self) -> &[String] {
        &self.channel_names
    }

    /// Get minimum length across all channels
    pub fn min_len(&self) -> usize {
        self.channels.iter().map(|c| c.len()).min().unwrap_or(0)
    }

    /// Clear all channels
    pub fn clear_all(&mut self) {
        for channel in &mut self.channels {
            channel.clear();
        }
    }
}

/// Streaming statistics tracker
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total samples generated
    pub samples_generated: u64,
    /// Samples per second (throughput)
    pub samples_per_second: f64,
    /// Buffer underruns (consumer faster than producer)
    pub underruns: u64,
    /// Buffer overruns (producer faster than consumer)
    pub overruns: u64,
    /// Current buffer fill level (0.0 - 1.0)
    pub buffer_fill: f64,
    /// Generation time in seconds
    pub elapsed_time: f64,
}

impl StreamingStats {
    /// Update statistics
    pub fn update(&mut self, samples: u64, elapsed: f64, buffer_len: usize, buffer_capacity: usize) {
        self.samples_generated += samples;
        self.elapsed_time = elapsed;
        self.samples_per_second = if elapsed > 0.0 {
            self.samples_generated as f64 / elapsed
        } else {
            0.0
        };
        self.buffer_fill = buffer_len as f64 / buffer_capacity as f64;
    }

    /// Record an underrun
    pub fn record_underrun(&mut self) {
        self.underruns += 1;
    }

    /// Record an overrun
    pub fn record_overrun(&mut self) {
        self.overruns += 1;
    }
}

/// Configuration for real-time streaming
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size in samples
    pub buffer_size: usize,
    /// Target sample rate (Hz)
    pub sample_rate: f64,
    /// Batch size for processing
    pub batch_size: usize,
    /// Enable real-time pacing (wait between samples)
    pub realtime_pacing: bool,
    /// Maximum latency allowed (seconds)
    pub max_latency: f64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4096,
            sample_rate: 1000.0,
            batch_size: 256,
            realtime_pacing: false,
            max_latency: 0.1, // 100ms
        }
    }
}

impl StreamingConfig {
    /// Create config for ECG streaming (1000 Hz)
    pub fn ecg() -> Self {
        Self {
            buffer_size: 10000, // 10 seconds
            sample_rate: 1000.0,
            batch_size: 256,
            realtime_pacing: true,
            max_latency: 0.05,
        }
    }

    /// Create config for video streaming (30 FPS)
    pub fn video_30fps() -> Self {
        Self {
            buffer_size: 300, // 10 seconds
            sample_rate: 30.0,
            batch_size: 1,
            realtime_pacing: true,
            max_latency: 0.033, // ~1 frame
        }
    }

    /// Create config for high-speed EMG (2000 Hz)
    pub fn emg() -> Self {
        Self {
            buffer_size: 20000, // 10 seconds
            sample_rate: 2000.0,
            batch_size: 512,
            realtime_pacing: true,
            max_latency: 0.02,
        }
    }
}

// ============================================================================
// Example Streaming Generators
// ============================================================================

/// Streaming ECG generator using McSharry model
///
/// This is a streaming version of the ECG morphology generator that produces
/// samples one at a time while maintaining internal ODE state.
pub struct StreamingEcg;

/// State for streaming ECG generation
#[derive(Debug, Clone)]
pub struct StreamingEcgState {
    /// Current x state (ODE)
    pub x: f64,
    /// Current y state (ODE)
    pub y: f64,
    /// Current z state (output)
    pub z: f64,
    /// Current sample index
    pub sample_idx: usize,
    /// Time step
    pub dt: f64,
    /// Angular frequency
    pub omega: f64,
    /// Wave parameters: (name, amplitude, width, time_offset)
    pub waves: Vec<(char, f64, f64, f64)>,
    /// RNG state
    pub rng: rand::rngs::StdRng,
}

/// Parameters for streaming ECG
#[derive(Debug, Clone)]
pub struct StreamingEcgParams {
    /// Sampling rate in Hz
    pub sampling_rate: f64,
    /// Heart rate in BPM
    pub heart_rate: f64,
    /// P wave amplitude
    pub p_amplitude: f64,
    /// T wave amplitude
    pub t_amplitude: f64,
    /// Optional duration limit (None = infinite)
    pub duration: Option<f64>,
}

impl Default for StreamingEcgParams {
    fn default() -> Self {
        Self {
            sampling_rate: 1000.0,
            heart_rate: 60.0,
            p_amplitude: 0.25,
            t_amplitude: 0.35,
            duration: None,
        }
    }
}

impl StreamingGenerator for StreamingEcg {
    type State = StreamingEcgState;
    type Parameters = StreamingEcgParams;
    type Sample = f64;

    fn init_state(&self, params: &Self::Parameters, seed: u64) -> Self::State {
        use rand::SeedableRng;
        use std::f64::consts::PI;

        let dt = 1.0 / params.sampling_rate;
        let omega = 2.0 * PI * (params.heart_rate / 60.0);

        // McSharry wave parameters: P, Q, R, S, T
        let waves = vec![
            ('P', params.p_amplitude, 0.1, -PI / 3.0),
            ('Q', -0.1, 0.1, -PI / 12.0),
            ('R', 1.0, 0.1, 0.0),
            ('S', -0.2, 0.1, PI / 12.0),
            ('T', params.t_amplitude, 0.25, PI / 2.0),
        ];

        StreamingEcgState {
            x: 1.0,
            y: 0.0,
            z: 0.0,
            sample_idx: 0,
            dt,
            omega,
            waves,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    fn next_sample(&self, state: &mut Self::State) -> Self::Sample {
        use std::f64::consts::PI;

        let t = state.sample_idx as f64 * state.dt;
        let theta = (state.omega * t).rem_euclid(2.0 * PI);

        // McSharry ODE: dz/dt = sum of Gaussian bumps
        let mut dz_dt = 0.0;
        for (_, ai, bi, thetai) in &state.waves {
            let delta_theta = (theta - thetai).rem_euclid(2.0 * PI);
            let delta_theta = if delta_theta > PI {
                delta_theta - 2.0 * PI
            } else {
                delta_theta
            };
            dz_dt += -ai * delta_theta * (-delta_theta.powi(2) / (2.0 * bi.powi(2))).exp();
        }

        // Coupled ODEs for circular trajectory
        let alpha = 1.0_f64;
        let dx_dt = alpha * (state.x - state.x.powi(3) / 3.0 - state.y);
        let dy_dt = state.x / alpha;

        // Euler integration
        state.x += dx_dt * state.dt;
        state.y += dy_dt * state.dt;
        state.z += (dz_dt - state.z) * state.dt;

        state.sample_idx += 1;

        state.z
    }

    fn current_time(&self, state: &Self::State) -> f64 {
        state.sample_idx as f64 * state.dt
    }

    fn sampling_rate(&self, params: &Self::Parameters) -> f64 {
        params.sampling_rate
    }

    fn reset_state(&self, state: &mut Self::State, params: &Self::Parameters, seed: u64) {
        *state = self.init_state(params, seed);
    }

    fn is_finite(&self, params: &Self::Parameters) -> bool {
        params.duration.is_some()
    }

    fn is_complete(&self, state: &Self::State, params: &Self::Parameters) -> bool {
        if let Some(duration) = params.duration {
            self.current_time(state) >= duration
        } else {
            false
        }
    }
}

/// Streaming sinusoidal tremor generator
///
/// Generates tremor signal with configurable frequency and amplitude modulation.
pub struct StreamingTremor;

/// State for streaming tremor
#[derive(Debug, Clone)]
pub struct StreamingTremorState {
    pub sample_idx: usize,
    pub dt: f64,
    pub base_freq: f64,
    pub amplitude: f64,
    pub amplitude_var: f64,
    pub rng: rand::rngs::StdRng,
}

/// Parameters for streaming tremor
#[derive(Debug, Clone)]
pub struct StreamingTremorParams {
    pub sampling_rate: f64,
    /// Tremor frequency in Hz (typically 4-12 Hz)
    pub frequency: f64,
    /// Base amplitude
    pub amplitude: f64,
    /// Amplitude variation (0-1)
    pub amplitude_variation: f64,
    /// Duration limit
    pub duration: Option<f64>,
}

impl Default for StreamingTremorParams {
    fn default() -> Self {
        Self {
            sampling_rate: 100.0,
            frequency: 5.0,       // 5 Hz typical for Parkinson's
            amplitude: 1.0,
            amplitude_variation: 0.2,
            duration: None,
        }
    }
}

impl StreamingGenerator for StreamingTremor {
    type State = StreamingTremorState;
    type Parameters = StreamingTremorParams;
    type Sample = f64;

    fn init_state(&self, params: &Self::Parameters, seed: u64) -> Self::State {
        use rand::SeedableRng;

        StreamingTremorState {
            sample_idx: 0,
            dt: 1.0 / params.sampling_rate,
            base_freq: params.frequency,
            amplitude: params.amplitude,
            amplitude_var: params.amplitude_variation,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    fn next_sample(&self, state: &mut Self::State) -> Self::Sample {
        use rand::Rng;
        use std::f64::consts::PI;

        let t = state.sample_idx as f64 * state.dt;

        // Amplitude modulation (slow variation)
        let amp_mod = 1.0 + state.amplitude_var * (0.3 * 2.0 * PI * t).sin();

        // Slight frequency variation
        let freq_noise: f64 = state.rng.gen_range(-0.1..0.1);
        let freq = state.base_freq + freq_noise;

        // Generate tremor signal
        let tremor = state.amplitude * amp_mod * (2.0 * PI * freq * t).sin();

        state.sample_idx += 1;

        tremor
    }

    fn current_time(&self, state: &Self::State) -> f64 {
        state.sample_idx as f64 * state.dt
    }

    fn sampling_rate(&self, params: &Self::Parameters) -> f64 {
        params.sampling_rate
    }

    fn reset_state(&self, state: &mut Self::State, params: &Self::Parameters, seed: u64) {
        *state = self.init_state(params, seed);
    }

    fn is_finite(&self, params: &Self::Parameters) -> bool {
        params.duration.is_some()
    }

    fn is_complete(&self, state: &Self::State, params: &Self::Parameters) -> bool {
        if let Some(duration) = params.duration {
            self.current_time(state) >= duration
        } else {
            false
        }
    }
}

/// Multi-modal streaming generator combining multiple signals
pub struct MultiModalStreaming {
    pub ecg: StreamingEcg,
    pub tremor: StreamingTremor,
}

/// Combined state for multi-modal streaming
pub struct MultiModalState {
    pub ecg_state: StreamingEcgState,
    pub tremor_state: StreamingTremorState,
}

/// Combined parameters
#[derive(Debug, Clone)]
pub struct MultiModalParams {
    pub ecg: StreamingEcgParams,
    pub tremor: StreamingTremorParams,
}

impl Default for MultiModalParams {
    fn default() -> Self {
        Self {
            ecg: StreamingEcgParams::default(),
            tremor: StreamingTremorParams::default(),
        }
    }
}

/// Multi-modal sample output
#[derive(Debug, Clone)]
pub struct MultiModalSample {
    pub ecg: f64,
    pub tremor: f64,
    pub timestamp: f64,
}

impl MultiModalStreaming {
    pub fn new() -> Self {
        Self {
            ecg: StreamingEcg,
            tremor: StreamingTremor,
        }
    }

    pub fn init_state(&self, params: &MultiModalParams, seed: u64) -> MultiModalState {
        MultiModalState {
            ecg_state: self.ecg.init_state(&params.ecg, seed),
            tremor_state: self.tremor.init_state(&params.tremor, seed + 1),
        }
    }

    pub fn next_sample(&self, state: &mut MultiModalState) -> MultiModalSample {
        let ecg = self.ecg.next_sample(&mut state.ecg_state);
        let tremor = self.tremor.next_sample(&mut state.tremor_state);
        let timestamp = self.ecg.current_time(&state.ecg_state);

        MultiModalSample {
            ecg,
            tremor,
            timestamp,
        }
    }
}

impl Default for MultiModalStreaming {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buffer: RingBuffer<i32> = RingBuffer::new(4);

        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 4);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert_eq!(buffer.len(), 3);
        assert!(!buffer.is_full());

        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut buffer: RingBuffer<i32> = RingBuffer::new(3);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert!(buffer.is_full());

        // This should overwrite the oldest (1)
        buffer.push(4);

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop(), Some(2)); // 1 was overwritten
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), Some(4));
    }

    #[test]
    fn test_ring_buffer_batch_operations() {
        let mut buffer: RingBuffer<i32> = RingBuffer::new(10);

        buffer.push_batch(&[1, 2, 3, 4, 5]);
        assert_eq!(buffer.len(), 5);

        let batch = buffer.read_batch(3);
        assert_eq!(batch, vec![1, 2, 3]);
        assert_eq!(buffer.len(), 5); // read_batch doesn't remove

        let popped = buffer.pop_batch(3);
        assert_eq!(popped, vec![1, 2, 3]);
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_ring_buffer_peek() {
        let mut buffer: RingBuffer<i32> = RingBuffer::new(5);

        buffer.push(10);
        buffer.push(20);
        buffer.push(30);

        assert_eq!(buffer.peek(), Some(&10));
        assert_eq!(buffer.peek_newest(), Some(&30));
        assert_eq!(buffer.len(), 3); // Peek doesn't change count
    }

    #[test]
    fn test_ring_buffer_to_vec() {
        let mut buffer: RingBuffer<i32> = RingBuffer::new(5);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        let vec = buffer.to_vec();
        assert_eq!(vec, vec![1, 2, 3]);
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut buffer: RingBuffer<i32> = RingBuffer::new(5);

        buffer.push_batch(&[1, 2, 3, 4, 5]);
        assert!(buffer.is_full());

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_atomic_ring_buffer_basic() {
        let buffer: AtomicRingBuffer<i32> = AtomicRingBuffer::new(4);

        assert!(buffer.is_empty());

        assert!(buffer.try_push(1));
        assert!(buffer.try_push(2));
        assert!(buffer.try_push(3));

        assert_eq!(buffer.len(), 3);

        assert_eq!(buffer.try_pop(), Some(1));
        assert_eq!(buffer.try_pop(), Some(2));
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_atomic_ring_buffer_full() {
        let buffer: AtomicRingBuffer<i32> = AtomicRingBuffer::new(3);

        assert!(buffer.try_push(1));
        assert!(buffer.try_push(2));
        // Buffer is now full (capacity - 1 for SPSC)
        assert!(!buffer.try_push(3)); // Should fail - full
    }

    #[test]
    fn test_multi_channel_buffer() {
        let mut buffer: MultiChannelBuffer<f64> = MultiChannelBuffer::new(
            vec!["ecg".to_string(), "emg".to_string()],
            100,
        );

        buffer.push_to_named_channel("ecg", 1.0);
        buffer.push_to_named_channel("emg", 2.0);
        buffer.push_synchronized(vec![3.0, 4.0]);

        assert_eq!(buffer.channel("ecg").unwrap().len(), 2);
        assert_eq!(buffer.channel("emg").unwrap().len(), 2);

        let synced = buffer.read_synchronized_batch(2);
        assert_eq!(synced[0], vec![1.0, 3.0]); // ECG
        assert_eq!(synced[1], vec![2.0, 4.0]); // EMG
    }

    #[test]
    fn test_streaming_stats() {
        let mut stats = StreamingStats::default();

        stats.update(1000, 1.0, 500, 1000);
        assert_eq!(stats.samples_generated, 1000);
        assert_eq!(stats.samples_per_second, 1000.0);
        assert_eq!(stats.buffer_fill, 0.5);

        stats.record_underrun();
        stats.record_overrun();
        assert_eq!(stats.underruns, 1);
        assert_eq!(stats.overruns, 1);
    }

    #[test]
    fn test_streaming_config_presets() {
        let ecg_config = StreamingConfig::ecg();
        assert_eq!(ecg_config.sample_rate, 1000.0);
        assert!(ecg_config.realtime_pacing);

        let video_config = StreamingConfig::video_30fps();
        assert_eq!(video_config.sample_rate, 30.0);

        let emg_config = StreamingConfig::emg();
        assert_eq!(emg_config.sample_rate, 2000.0);
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let mut buffer: RingBuffer<i32> = RingBuffer::new(4);

        // Fill and drain multiple times to test wraparound
        for round in 0..3 {
            for i in 0..4 {
                buffer.push(round * 10 + i);
            }
            assert!(buffer.is_full());

            for i in 0..4 {
                assert_eq!(buffer.pop(), Some(round * 10 + i));
            }
            assert!(buffer.is_empty());
        }
    }

    // ========== Streaming Generator Tests ==========

    #[test]
    fn test_streaming_ecg_basic() {
        let generator = StreamingEcg;
        let params = StreamingEcgParams::default();
        let mut state = generator.init_state(&params, 42);

        // Generate 1000 samples (1 second at 1000 Hz)
        let samples: Vec<f64> = (0..1000)
            .map(|_| generator.next_sample(&mut state))
            .collect();

        assert_eq!(samples.len(), 1000);
        assert!((generator.current_time(&state) - 1.0).abs() < 0.001);

        // ECG should have variation (not constant)
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!(variance > 0.0, "ECG should have non-zero variance");
    }

    #[test]
    fn test_streaming_ecg_reset() {
        let generator = StreamingEcg;
        let params = StreamingEcgParams::default();
        let mut state = generator.init_state(&params, 42);

        // Generate some samples
        let first_samples: Vec<f64> = (0..100)
            .map(|_| generator.next_sample(&mut state))
            .collect();

        // Reset and generate again with same seed
        generator.reset_state(&mut state, &params, 42);
        let second_samples: Vec<f64> = (0..100)
            .map(|_| generator.next_sample(&mut state))
            .collect();

        // Should produce identical results
        for (a, b) in first_samples.iter().zip(second_samples.iter()) {
            assert!((a - b).abs() < 1e-10, "Reset should produce identical output");
        }
    }

    #[test]
    fn test_streaming_ecg_with_buffer() {
        let generator = StreamingEcg;
        let params = StreamingEcgParams::default();
        let mut state = generator.init_state(&params, 42);

        let mut buffer: RingBuffer<f64> = RingBuffer::new(256);

        // Simulate real-time: generate and buffer
        for _ in 0..500 {
            let sample = generator.next_sample(&mut state);
            buffer.push(sample);
        }

        assert!(buffer.is_full());
        assert_eq!(buffer.len(), 256);

        // Read batch for processing
        let batch = buffer.read_batch(128);
        assert_eq!(batch.len(), 128);
    }

    #[test]
    fn test_streaming_tremor_basic() {
        let generator = StreamingTremor;
        let params = StreamingTremorParams {
            sampling_rate: 100.0,
            frequency: 5.0,
            amplitude: 1.0,
            amplitude_variation: 0.0, // No variation for predictable test
            duration: None,
        };
        let mut state = generator.init_state(&params, 42);

        // Generate 100 samples (1 second at 100 Hz)
        let samples: Vec<f64> = (0..100)
            .map(|_| generator.next_sample(&mut state))
            .collect();

        assert_eq!(samples.len(), 100);

        // At 5 Hz, we should see ~5 complete cycles in 1 second
        // Count zero crossings (rough frequency estimate)
        let zero_crossings: usize = samples.windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();

        // Should be roughly 10 zero crossings (2 per cycle * 5 cycles)
        assert!(zero_crossings >= 8 && zero_crossings <= 12,
            "Expected ~10 zero crossings, got {}", zero_crossings);
    }

    #[test]
    fn test_streaming_ecg_finite_duration() {
        let generator = StreamingEcg;
        let params = StreamingEcgParams {
            duration: Some(0.5), // 0.5 second limit
            ..StreamingEcgParams::default()
        };
        let mut state = generator.init_state(&params, 42);

        assert!(generator.is_finite(&params));
        assert!(!generator.is_complete(&state, &params));

        // Generate until complete
        let mut count = 0;
        while !generator.is_complete(&state, &params) {
            generator.next_sample(&mut state);
            count += 1;
            if count > 1000 { break; } // Safety limit
        }

        assert!(generator.is_complete(&state, &params));
        assert_eq!(count, 500); // 0.5s at 1000 Hz
    }

    #[test]
    fn test_multi_modal_streaming() {
        let generator = MultiModalStreaming::new();
        let params = MultiModalParams::default();
        let mut state = generator.init_state(&params, 42);

        // Generate 100 multi-modal samples
        let samples: Vec<MultiModalSample> = (0..100)
            .map(|_| generator.next_sample(&mut state))
            .collect();

        assert_eq!(samples.len(), 100);

        // Check we have both ECG and tremor data
        let ecg_variance: f64 = {
            let mean = samples.iter().map(|s| s.ecg).sum::<f64>() / 100.0;
            samples.iter().map(|s| (s.ecg - mean).powi(2)).sum::<f64>() / 100.0
        };
        assert!(ecg_variance > 0.0, "Should have ECG variation");

        // Timestamps should be increasing
        for w in samples.windows(2) {
            assert!(w[1].timestamp > w[0].timestamp, "Timestamps should increase");
        }
    }

    #[test]
    fn test_streaming_with_multi_channel_buffer() {
        let ecg_gen = StreamingEcg;
        let tremor_gen = StreamingTremor;

        let ecg_params = StreamingEcgParams::default();
        let tremor_params = StreamingTremorParams::default();

        let mut ecg_state = ecg_gen.init_state(&ecg_params, 42);
        let mut tremor_state = tremor_gen.init_state(&tremor_params, 43);

        let mut buffer: MultiChannelBuffer<f64> = MultiChannelBuffer::new(
            vec!["ecg".to_string(), "tremor".to_string()],
            100,
        );

        // Simulate synchronized streaming
        for _ in 0..50 {
            let ecg = ecg_gen.next_sample(&mut ecg_state);
            let tremor = tremor_gen.next_sample(&mut tremor_state);
            buffer.push_synchronized(vec![ecg, tremor]);
        }

        assert_eq!(buffer.min_len(), 50);

        let synced = buffer.read_synchronized_batch(25);
        assert_eq!(synced.len(), 2); // 2 channels
        assert_eq!(synced[0].len(), 25); // 25 samples each
    }

    #[test]
    fn test_streaming_batch_generation() {
        let generator = StreamingEcg;
        let params = StreamingEcgParams::default();
        let mut state = generator.init_state(&params, 42);

        // Generate batch of 256 samples
        let batch = generator.next_samples(&mut state, 256);
        assert_eq!(batch.len(), 256);

        // Compare with individual generation
        generator.reset_state(&mut state, &params, 42);
        let individual: Vec<f64> = (0..256)
            .map(|_| generator.next_sample(&mut state))
            .collect();

        for (a, b) in batch.iter().zip(individual.iter()) {
            assert!((a - b).abs() < 1e-10, "Batch should match individual");
        }
    }
}
