//! Buffer management for real-time signal processing
//!
//! Provides efficient circular buffers, sliding windows, and overlap buffers
//! for streaming data processing.

use std::collections::VecDeque;

/// Thread-safe ring buffer for streaming data
///
/// A circular buffer implementation that overwrites oldest data when full.
/// Optimized for real-time streaming scenarios.
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
    count: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    /// Create a new ring buffer with the specified capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");
        Self {
            data: vec![T::default(); capacity],
            capacity,
            write_pos: 0,
            read_pos: 0,
            count: 0,
        }
    }

    /// Push a single item into the buffer
    ///
    /// If the buffer is full, the oldest item is overwritten.
    pub fn push(&mut self, item: T) {
        self.data[self.write_pos] = item;
        self.write_pos = (self.write_pos + 1) % self.capacity;

        if self.count < self.capacity {
            self.count += 1;
        } else {
            // Buffer is full, move read position forward
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }
    }

    /// Push multiple items into the buffer
    pub fn push_slice(&mut self, items: &[T]) {
        for item in items {
            self.push(item.clone());
        }
    }

    /// Pop the oldest item from the buffer
    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 {
            return None;
        }

        let item = self.data[self.read_pos].clone();
        self.read_pos = (self.read_pos + 1) % self.capacity;
        self.count -= 1;
        Some(item)
    }

    /// Peek at the oldest item without removing it
    pub fn peek(&self) -> Option<&T> {
        if self.count == 0 {
            None
        } else {
            Some(&self.data[self.read_pos])
        }
    }

    /// Get the current number of items in the buffer
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

    /// Clear all items from the buffer
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.count = 0;
    }

    /// Get all data in correct order (oldest to newest)
    pub fn as_slice(&self) -> Vec<T> {
        if self.count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.count);
        let mut pos = self.read_pos;
        for _ in 0..self.count {
            result.push(self.data[pos].clone());
            pos = (pos + 1) % self.capacity;
        }
        result
    }

    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Sliding window with configurable overlap
///
/// Efficiently manages windowed processing of streaming data with
/// configurable hop size for overlap-add or overlap-save algorithms.
#[derive(Debug, Clone)]
pub struct SlidingWindow<T> {
    buffer: VecDeque<T>,
    window_size: usize,
    hop_size: usize,
    samples_since_last_window: usize,
}

impl<T: Clone + Default> SlidingWindow<T> {
    /// Create a new sliding window
    ///
    /// # Arguments
    /// * `window_size` - Size of each window
    /// * `hop_size` - Number of samples to advance between windows
    ///
    /// # Panics
    /// Panics if window_size or hop_size is 0
    pub fn new(window_size: usize, hop_size: usize) -> Self {
        assert!(window_size > 0, "Window size must be greater than 0");
        assert!(hop_size > 0, "Hop size must be greater than 0");

        Self {
            buffer: VecDeque::with_capacity(window_size + hop_size),
            window_size,
            hop_size,
            samples_since_last_window: 0,
        }
    }

    /// Push a single sample and return a window if ready
    ///
    /// Returns `Some(window)` when enough samples have been collected
    /// for a new window based on the hop size.
    pub fn push(&mut self, sample: T) -> Option<Vec<T>> {
        self.buffer.push_back(sample);
        self.samples_since_last_window += 1;

        // Check if we have enough samples for the first window
        if self.buffer.len() >= self.window_size &&
           self.samples_since_last_window >= self.hop_size {
            self.samples_since_last_window = 0;
            return Some(self.get_current_window());
        }

        None
    }

    /// Push multiple samples and return all ready windows
    ///
    /// May return multiple windows if enough samples are provided.
    pub fn push_slice(&mut self, samples: &[T]) -> Vec<Vec<T>> {
        let mut windows = Vec::new();

        for sample in samples {
            if let Some(window) = self.push(sample.clone()) {
                windows.push(window);
            }
        }

        windows
    }

    /// Get the current window without advancing
    fn get_current_window(&mut self) -> Vec<T> {
        let window: Vec<T> = self.buffer.iter()
            .take(self.window_size)
            .cloned()
            .collect();

        // Remove samples according to hop size to maintain overlap
        for _ in 0..self.hop_size {
            if !self.buffer.is_empty() {
                self.buffer.pop_front();
            }
        }

        window
    }

    /// Get the current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the window buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.samples_since_last_window = 0;
    }

    /// Get window parameters
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get hop size
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Calculate the overlap between windows
    pub fn overlap(&self) -> usize {
        if self.window_size > self.hop_size {
            self.window_size - self.hop_size
        } else {
            0
        }
    }
}

/// Buffer for overlap-add/overlap-save processing
///
/// Manages frame-based processing with overlap for filtering and
/// spectral analysis operations.
#[derive(Debug, Clone)]
pub struct OverlapBuffer {
    frame_size: usize,
    overlap: usize,
    buffer: VecDeque<f64>,
    output_buffer: Vec<f64>,
}

impl OverlapBuffer {
    /// Create a new overlap buffer
    ///
    /// # Arguments
    /// * `frame_size` - Size of each processing frame
    /// * `overlap` - Number of samples to overlap between frames
    pub fn new(frame_size: usize, overlap: usize) -> Self {
        assert!(frame_size > 0, "Frame size must be greater than 0");
        assert!(overlap < frame_size, "Overlap must be less than frame size");

        Self {
            frame_size,
            overlap,
            buffer: VecDeque::with_capacity(frame_size * 2),
            output_buffer: Vec::new(),
        }
    }

    /// Process a frame with overlap-add
    ///
    /// Adds the processed frame to the output buffer, handling overlap appropriately.
    pub fn process_frame(&mut self, frame: &[f64]) -> Vec<f64> {
        assert_eq!(frame.len(), self.frame_size, "Frame size mismatch");

        // Extend buffer with new frame
        self.buffer.extend(frame.iter().cloned());

        // Extract output samples (non-overlapping portion)
        let output_size = self.frame_size - self.overlap;
        let mut output = Vec::with_capacity(output_size);

        for _ in 0..output_size {
            if let Some(sample) = self.buffer.pop_front() {
                output.push(sample);
            }
        }

        output
    }

    /// Add processed samples with overlap-add combination
    ///
    /// Combines overlapping regions by summing them.
    pub fn overlap_add(&mut self, samples: &[f64]) {
        let current_len = self.output_buffer.len();

        // Check if we have overlap samples from a previous frame
        if current_len > 0 && current_len >= self.overlap {
            // We have previous overlap samples at the end
            // Add the first overlap samples to the existing ones
            let overlap_start = current_len - self.overlap;
            for i in 0..self.overlap.min(samples.len()) {
                self.output_buffer[overlap_start + i] += samples[i];
            }
            // Append remaining new samples
            if samples.len() > self.overlap {
                self.output_buffer.extend_from_slice(&samples[self.overlap..]);
            }
        } else {
            // First frame or not enough samples to overlap
            self.output_buffer.extend_from_slice(samples);
        }
    }

    /// Drain output samples
    pub fn drain_output(&mut self, n: usize) -> Vec<f64> {
        let to_drain = n.min(self.output_buffer.len());
        self.output_buffer.drain(0..to_drain).collect()
    }

    /// Get the current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.output_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buffer = RingBuffer::new(3);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());

        buffer.push(1);
        buffer.push(2);
        assert_eq!(buffer.len(), 2);

        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.push(4); // Overwrites 1

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.as_slice(), vec![2, 3, 4]);
    }

    #[test]
    fn test_ring_buffer_push_slice() {
        let mut buffer = RingBuffer::new(5);
        buffer.push_slice(&[1, 2, 3]);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.as_slice(), vec![1, 2, 3]);
    }

    #[test]
    fn test_ring_buffer_peek() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);

        assert_eq!(buffer.peek(), Some(&1));
        assert_eq!(buffer.len(), 2); // Peek doesn't remove
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.clear();

        assert!(buffer.is_empty());
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn test_sliding_window_no_overlap() {
        let mut window = SlidingWindow::new(3, 3);

        assert_eq!(window.push(1.0), None);
        assert_eq!(window.push(2.0), None);

        let result = window.push(3.0);
        assert_eq!(result, Some(vec![1.0, 2.0, 3.0]));

        assert_eq!(window.push(4.0), None);
        assert_eq!(window.push(5.0), None);

        let result = window.push(6.0);
        assert_eq!(result, Some(vec![4.0, 5.0, 6.0]));
    }

    #[test]
    fn test_sliding_window_with_overlap() {
        let mut window = SlidingWindow::new(4, 2);

        // Fill first window
        assert_eq!(window.push(1.0), None);
        assert_eq!(window.push(2.0), None);
        assert_eq!(window.push(3.0), None);

        let result = window.push(4.0);
        assert_eq!(result, Some(vec![1.0, 2.0, 3.0, 4.0]));

        // Next window should have 2-sample overlap
        assert_eq!(window.push(5.0), None);

        let result = window.push(6.0);
        assert_eq!(result, Some(vec![3.0, 4.0, 5.0, 6.0]));
    }

    #[test]
    fn test_sliding_window_push_slice() {
        let mut window = SlidingWindow::new(3, 3);

        let windows = window.push_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(windows[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sliding_window_overlap_calculation() {
        let window = SlidingWindow::<f64>::new(10, 5);
        assert_eq!(window.overlap(), 5);

        let window = SlidingWindow::<f64>::new(10, 10);
        assert_eq!(window.overlap(), 0);
    }

    #[test]
    fn test_overlap_buffer_basic() {
        let mut buffer = OverlapBuffer::new(4, 2);

        let frame1 = vec![1.0, 2.0, 3.0, 4.0];
        let output1 = buffer.process_frame(&frame1);

        // First frame outputs non-overlapping part
        assert_eq!(output1.len(), 2); // frame_size - overlap
    }

    #[test]
    fn test_overlap_buffer_multiple_frames() {
        let mut buffer = OverlapBuffer::new(4, 2);

        let frame1 = vec![1.0, 2.0, 3.0, 4.0];
        let output1 = buffer.process_frame(&frame1);
        assert_eq!(output1, vec![1.0, 2.0]); // First 2 samples (non-overlapping)

        let frame2 = vec![5.0, 6.0, 7.0, 8.0];
        let output2 = buffer.process_frame(&frame2);
        // After first frame, buffer has [3.0, 4.0]
        // Second frame adds [5.0, 6.0, 7.0, 8.0]
        // Buffer becomes [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        // Output first 2: [3.0, 4.0]
        assert_eq!(output2, vec![3.0, 4.0]);
    }

    #[test]
    fn test_overlap_buffer_clear() {
        let mut buffer = OverlapBuffer::new(4, 2);

        let frame = vec![1.0, 2.0, 3.0, 4.0];
        buffer.process_frame(&frame);

        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_overlap_add() {
        let mut buffer = OverlapBuffer::new(6, 2);

        // First frame
        buffer.overlap_add(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output1 = buffer.drain_output(4);
        assert_eq!(output1, vec![1.0, 2.0, 3.0, 4.0]);

        // Second frame - overlapping parts should sum
        buffer.overlap_add(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let output2 = buffer.drain_output(4);
        assert_eq!(output2, vec![5.0 + 7.0, 6.0 + 8.0, 9.0, 10.0]);
    }
}
