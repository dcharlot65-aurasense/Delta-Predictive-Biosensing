//! Core types for the DPB framework.

use crate::error::{DpbError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Newtype Wrappers for Type Safety
// ============================================================================

/// Unique identifier for a signal channel.
///
/// Using a newtype prevents accidentally mixing channel indices with other
/// integer types like neuron indices or sample counts.
///
/// # Examples
///
/// ```
/// use dpb_core::types::ChannelId;
///
/// let ch = ChannelId::new(5);
/// assert_eq!(ch.as_u32(), 5);
///
/// // Convert from u32
/// let ch2: ChannelId = 10.into();
/// assert_eq!(u32::from(ch2), 10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelId(pub u32);

impl ChannelId {
    /// Creates a new channel ID.
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Returns the underlying channel index.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<u32> for ChannelId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<ChannelId> for u32 {
    fn from(id: ChannelId) -> Self {
        id.0
    }
}

impl std::fmt::Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ch{}", self.0)
    }
}

/// Index into a neuron population.
///
/// This newtype ensures neuron indices are not confused with channel IDs
/// or sample counts.
///
/// # Examples
///
/// ```
/// use dpb_core::types::NeuronIndex;
///
/// let idx = NeuronIndex::new(42);
/// assert_eq!(idx.as_usize(), 42);
///
/// // Use in collections
/// let neurons: Vec<NeuronIndex> = (0..10).map(NeuronIndex::new).collect();
/// assert_eq!(neurons.len(), 10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuronIndex(pub usize);

impl NeuronIndex {
    /// Creates a new neuron index.
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// Returns the underlying index value.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

impl From<usize> for NeuronIndex {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<NeuronIndex> for usize {
    fn from(idx: NeuronIndex) -> Self {
        idx.0
    }
}

impl std::fmt::Display for NeuronIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "N{}", self.0)
    }
}

/// Count of samples in a signal.
///
/// Provides type safety and utility methods for working with sample counts.
///
/// # Examples
///
/// ```
/// use dpb_core::types::SampleCount;
///
/// let count = SampleCount::new(1000);
/// let sample_rate = 250.0; // Hz
///
/// // Convert to time
/// let seconds = count.as_seconds(sample_rate);
/// assert_eq!(seconds, 4.0); // 1000 samples / 250 Hz = 4 seconds
///
/// let duration = count.as_duration(sample_rate);
/// assert_eq!(duration.as_secs(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SampleCount(pub usize);

impl SampleCount {
    /// Creates a new sample count.
    #[inline]
    pub const fn new(count: usize) -> Self {
        Self(count)
    }

    /// Returns the underlying count.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }

    /// Converts sample count to duration at given sample rate.
    #[inline]
    pub fn as_duration(self, sample_rate: f64) -> std::time::Duration {
        std::time::Duration::from_secs_f64(self.0 as f64 / sample_rate)
    }

    /// Converts sample count to seconds at given sample rate.
    #[inline]
    pub fn as_seconds(self, sample_rate: f64) -> f64 {
        self.0 as f64 / sample_rate
    }

    /// Creates a sample count from a duration and sample rate.
    #[inline]
    pub fn from_duration(duration: std::time::Duration, sample_rate: f64) -> Self {
        Self((duration.as_secs_f64() * sample_rate) as usize)
    }
}

impl From<usize> for SampleCount {
    fn from(count: usize) -> Self {
        Self(count)
    }
}

impl From<SampleCount> for usize {
    fn from(count: SampleCount) -> Self {
        count.0
    }
}

impl std::fmt::Display for SampleCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} samples", self.0)
    }
}

// ============================================================================
// Core Domain Types
// ============================================================================

/// Represents a single spike event in neuromorphic processing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SpikeEvent {
    /// Timestamp in seconds
    pub timestamp: f64,
    /// Channel/neuron index
    pub channel: u32,
    /// Spike polarity (-1 or +1)
    pub polarity: i8,
    /// Spike magnitude/weight
    pub magnitude: f32,
}

impl SpikeEvent {
    /// Creates a new spike event.
    ///
    /// Polarity should be -1 or +1 but other values are allowed for flexibility.
    pub fn new(timestamp: f64, channel: u32, polarity: i8, magnitude: f32) -> Self {
        Self {
            timestamp,
            channel,
            polarity,
            magnitude,
        }
    }

    /// Creates a new spike event with validation.
    pub fn new_validated(timestamp: f64, channel: u32, polarity: i8, magnitude: f32) -> Result<Self> {
        if polarity != -1 && polarity != 1 {
            return Err(DpbError::InvalidParameter(
                "Polarity must be -1 or +1".to_string(),
            ));
        }
        Ok(Self {
            timestamp,
            channel,
            polarity,
            magnitude,
        })
    }
}

/// A collection of spike events with utility functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeTrain {
    /// Vector of spike events
    pub events: Vec<SpikeEvent>,
    /// Number of channels
    pub num_channels: u32,
}

impl SpikeTrain {
    /// Creates a new spike train.
    pub fn new(num_channels: u32) -> Self {
        Self {
            events: Vec::new(),
            num_channels,
        }
    }

    /// Creates a spike train from events.
    pub fn from_events(events: Vec<SpikeEvent>, num_channels: u32) -> Self {
        Self {
            events,
            num_channels,
        }
    }

    /// Adds a spike event to the train.
    pub fn add_event(&mut self, event: SpikeEvent) {
        self.events.push(event);
    }

    /// Returns the number of spikes.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns true if the spike train is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the duration of the spike train.
    pub fn duration(&self) -> f64 {
        self.events
            .iter()
            .map(|e| e.timestamp)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Sorts events by timestamp.
    pub fn sort_by_time(&mut self) {
        self.events
            .sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
    }

    /// Filters events by channel.
    pub fn filter_by_channel(&self, channel: u32) -> SpikeTrain {
        let events: Vec<SpikeEvent> = self
            .events
            .iter()
            .filter(|e| e.channel == channel)
            .copied()
            .collect();
        SpikeTrain::from_events(events, self.num_channels)
    }

    /// Returns spike rate (spikes per second).
    pub fn spike_rate(&self) -> f64 {
        let duration = self.duration();
        if duration > 0.0 {
            self.events.len() as f64 / duration
        } else {
            0.0
        }
    }

    /// Returns per-channel spike counts.
    pub fn channel_counts(&self) -> HashMap<u32, usize> {
        let mut counts = HashMap::new();
        for event in &self.events {
            *counts.entry(event.channel).or_insert(0) += 1;
        }
        counts
    }
}

/// Multi-channel time series data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Sample data (channels x time)
    pub data: Vec<Vec<f64>>,
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Channel names (optional)
    pub channel_names: Option<Vec<String>>,
}

impl TimeSeries {
    /// Creates a new time series.
    pub fn new(data: Vec<Vec<f64>>, sample_rate: f64) -> Result<Self> {
        if data.is_empty() {
            return Err(DpbError::InvalidDimensions(
                "Data cannot be empty".to_string(),
            ));
        }
        if sample_rate <= 0.0 {
            return Err(DpbError::InvalidParameter(
                "Sample rate must be positive".to_string(),
            ));
        }
        Ok(Self {
            data,
            sample_rate,
            channel_names: None,
        })
    }

    /// Returns the number of channels.
    pub fn num_channels(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of samples.
    pub fn num_samples(&self) -> usize {
        self.data.get(0).map(|v| v.len()).unwrap_or(0)
    }

    /// Returns the duration in seconds.
    pub fn duration(&self) -> f64 {
        self.num_samples() as f64 / self.sample_rate
    }

    /// Sets channel names.
    pub fn with_channel_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.num_channels() {
            return Err(DpbError::InvalidDimensions(
                "Number of names must match number of channels".to_string(),
            ));
        }
        self.channel_names = Some(names);
        Ok(self)
    }

    /// Gets a specific channel.
    pub fn get_channel(&self, index: usize) -> Option<&Vec<f64>> {
        self.data.get(index)
    }

    /// Gets a time slice across all channels.
    pub fn get_time_slice(&self, start: usize, end: usize) -> Result<TimeSeries> {
        if end > self.num_samples() {
            return Err(DpbError::OutOfBounds(
                "Time slice exceeds data length".to_string(),
            ));
        }
        let sliced_data: Vec<Vec<f64>> = self
            .data
            .iter()
            .map(|channel| channel[start..end].to_vec())
            .collect();
        Ok(TimeSeries {
            data: sliced_data,
            sample_rate: self.sample_rate,
            channel_names: self.channel_names.clone(),
        })
    }
}

/// Ground truth annotations for training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    /// Class labels
    pub labels: Vec<u32>,
    /// Continuous values (e.g., regression targets)
    pub values: Option<Vec<f64>>,
    /// Temporal annotations (start, end, label)
    pub temporal: Option<Vec<(f64, f64, u32)>>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl GroundTruth {
    /// Creates a new ground truth with labels.
    pub fn new(labels: Vec<u32>) -> Self {
        Self {
            labels,
            values: None,
            temporal: None,
            metadata: HashMap::new(),
        }
    }

    /// Creates ground truth with continuous values.
    pub fn with_values(labels: Vec<u32>, values: Vec<f64>) -> Result<Self> {
        if labels.len() != values.len() {
            return Err(DpbError::InvalidDimensions(
                "Labels and values must have same length".to_string(),
            ));
        }
        Ok(Self {
            labels,
            values: Some(values),
            temporal: None,
            metadata: HashMap::new(),
        })
    }

    /// Adds temporal annotations.
    pub fn with_temporal(mut self, temporal: Vec<(f64, f64, u32)>) -> Self {
        self.temporal = Some(temporal);
        self
    }

    /// Adds metadata.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Runtime context information for biosensor data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Context {
    /// Subject age in years (can be fractional for infants)
    pub age: Option<f64>,
    /// Subject sex
    pub sex: Option<String>,
    /// Subject height in centimeters
    pub height_cm: Option<f64>,
    /// Subject weight in kilograms
    pub weight_kg: Option<f64>,
    /// Medications
    pub medications: Vec<String>,
    /// Environmental conditions
    pub environment: HashMap<String, f64>,
    /// Device metadata
    pub device: HashMap<String, String>,
    /// Custom attributes
    pub custom: HashMap<String, String>,
}

impl Context {
    /// Creates a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets age.
    pub fn with_age(mut self, age: f64) -> Self {
        self.age = Some(age);
        self
    }

    /// Sets sex.
    pub fn with_sex(mut self, sex: String) -> Self {
        self.sex = Some(sex);
        self
    }

    /// Adds a medication.
    pub fn add_medication(&mut self, medication: String) {
        self.medications.push(medication);
    }

    /// Sets an environmental parameter.
    pub fn set_environment(&mut self, key: String, value: f64) {
        self.environment.insert(key, value);
    }

    /// Sets device metadata.
    pub fn set_device(&mut self, key: String, value: String) {
        self.device.insert(key, value);
    }
}

/// Sensing modality types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Contact-based sensors (ECG, EMG, EEG, etc.)
    Contact,
    /// Pose estimation/skeleton tracking
    Pose,
    /// Hand tracking
    Hand,
    /// Eye tracking
    Eye,
    /// Voice/audio
    Voice,
}

impl Modality {
    /// Returns a string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Contact => "contact",
            Modality::Pose => "pose",
            Modality::Hand => "hand",
            Modality::Eye => "eye",
            Modality::Voice => "voice",
        }
    }
}

/// Signal quality indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalQuality {
    /// Overall quality score [0.0, 1.0]
    pub score: f64,
    /// Signal-to-noise ratio (dB)
    pub snr: Option<f64>,
    /// Missing data percentage
    pub missing_data: f64,
    /// Artifact indicators
    pub artifacts: Vec<ArtifactType>,
    /// Per-channel quality scores
    pub channel_quality: Option<Vec<f64>>,
}

/// Types of signal artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Motion artifacts
    Motion,
    /// Electrical interference
    Electrical,
    /// Baseline drift
    BaselineDrift,
    /// Saturation/clipping
    Saturation,
    /// Missing data
    Missing,
}

impl SignalQuality {
    /// Creates a new signal quality indicator.
    pub fn new(score: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&score) {
            return Err(DpbError::InvalidParameter(
                "Quality score must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(Self {
            score,
            snr: None,
            missing_data: 0.0,
            artifacts: Vec::new(),
            channel_quality: None,
        })
    }

    /// Sets SNR.
    pub fn with_snr(mut self, snr: f64) -> Self {
        self.snr = Some(snr);
        self
    }

    /// Adds an artifact type.
    pub fn add_artifact(&mut self, artifact: ArtifactType) {
        if !self.artifacts.contains(&artifact) {
            self.artifacts.push(artifact);
        }
    }

    /// Checks if signal is acceptable.
    pub fn is_acceptable(&self, threshold: f64) -> bool {
        self.score >= threshold
    }
}

/// A buffer of signal samples that implements the Signal trait.
///
/// This is a simple container for single-channel signal data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalBuffer {
    /// Signal samples
    pub data: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Number of channels
    pub num_channels: usize,
}

impl SignalBuffer {
    /// Creates a new signal buffer.
    pub fn new(data: Vec<f32>, sample_rate: f64) -> Self {
        Self {
            data,
            sample_rate,
            num_channels: 1,
        }
    }

    /// Creates a single-channel signal buffer.
    pub fn single_channel(data: Vec<f32>, sample_rate: f64) -> Self {
        Self {
            data,
            sample_rate,
            num_channels: 1,
        }
    }

    /// Creates a multi-channel signal buffer.
    pub fn multi_channel(data: Vec<f32>, sample_rate: f64, num_channels: usize) -> Self {
        Self {
            data,
            sample_rate,
            num_channels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_event_creation() {
        let event = SpikeEvent::new(1.0, 5, 1, 0.5);
        assert_eq!(event.timestamp, 1.0);
        assert_eq!(event.channel, 5);
        assert_eq!(event.polarity, 1);
        assert_eq!(event.magnitude, 0.5);
    }

    #[test]
    fn test_spike_event_invalid_polarity() {
        let result = SpikeEvent::new_validated(1.0, 5, 2, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_spike_train() {
        let mut train = SpikeTrain::new(10);
        train.add_event(SpikeEvent::new(0.5, 1, 1, 1.0));
        train.add_event(SpikeEvent::new(1.0, 2, -1, 0.8));
        train.add_event(SpikeEvent::new(1.5, 1, 1, 0.9));

        assert_eq!(train.len(), 3);
        assert_eq!(train.duration(), 1.5);

        let filtered = train.filter_by_channel(1);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_time_series() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let ts = TimeSeries::new(data, 100.0).unwrap();

        assert_eq!(ts.num_channels(), 2);
        assert_eq!(ts.num_samples(), 3);
        assert_eq!(ts.duration(), 0.03);
    }

    #[test]
    fn test_ground_truth() {
        let gt = GroundTruth::new(vec![0, 1, 2]);
        assert_eq!(gt.labels.len(), 3);
        assert!(gt.values.is_none());
    }

    #[test]
    fn test_context() {
        let mut ctx = Context::new()
            .with_age(30.0)
            .with_sex("M".to_string());
        ctx.add_medication("aspirin".to_string());
        ctx.set_environment("temperature".to_string(), 22.5);

        assert_eq!(ctx.age, Some(30.0));
        assert_eq!(ctx.medications.len(), 1);
    }

    #[test]
    fn test_signal_quality() {
        let mut sq = SignalQuality::new(0.85).unwrap();
        sq.add_artifact(ArtifactType::Motion);

        assert!(sq.is_acceptable(0.8));
        assert!(!sq.is_acceptable(0.9));
    }

    #[test]
    fn test_modality() {
        assert_eq!(Modality::Contact.as_str(), "contact");
        assert_eq!(Modality::Eye.as_str(), "eye");
    }

    #[test]
    fn test_signal_buffer() {
        let data = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let signal = SignalBuffer::single_channel(data.clone(), 100.0);
        assert_eq!(signal.data.len(), 5);
        assert_eq!(signal.sample_rate, 100.0);
        assert_eq!(signal.num_channels, 1);
    }
}

/// Property-based tests using proptest
#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Test that SpikeEvent serialization round-trips correctly
        #[test]
        fn spike_event_serialization_roundtrip(
            timestamp in 0.0f64..1000.0,
            channel in 0u32..1000,
            polarity in prop_oneof![-1i8, 1i8],
            magnitude in 0.0f32..100.0,
        ) {
            let event = SpikeEvent::new(timestamp, channel, polarity, magnitude);
            let serialized = bincode::serialize(&event).unwrap();
            let deserialized: SpikeEvent = bincode::deserialize(&serialized).unwrap();
            prop_assert_eq!(event, deserialized);
        }

        /// Test that SpikeTrain operations are consistent
        #[test]
        fn spike_train_length_consistency(
            num_events in 0usize..100,
            num_channels in 1u32..100,
        ) {
            let mut train = SpikeTrain::new(num_channels);
            for i in 0..num_events {
                train.add_event(SpikeEvent::new(
                    i as f64 * 0.001,
                    (i as u32) % num_channels,
                    if i % 2 == 0 { 1 } else { -1 },
                    1.0,
                ));
            }
            prop_assert_eq!(train.len(), num_events);
        }

        /// Test that SignalQuality score validation works for all valid values
        #[test]
        fn signal_quality_valid_range(score in 0.0f64..=1.0) {
            let result = SignalQuality::new(score);
            prop_assert!(result.is_ok());
            let sq = result.unwrap();
            prop_assert_eq!(sq.score, score);
        }

        /// Test that SignalQuality rejects invalid scores
        #[test]
        fn signal_quality_invalid_range(
            score in prop_oneof![
                -100.0f64..-0.001,
                1.001f64..100.0
            ]
        ) {
            let result = SignalQuality::new(score);
            prop_assert!(result.is_err());
        }

        /// Test that Context builder methods don't lose data
        #[test]
        fn context_preserves_age(age in 0.0f64..150.0) {
            let ctx = Context::new().with_age(age);
            prop_assert_eq!(ctx.age, Some(age));
        }

        /// Test SpikeEvent validated constructor
        #[test]
        fn spike_event_validated_accepts_valid_polarity(
            timestamp in 0.0f64..1000.0,
            channel in 0u32..1000,
            polarity in prop_oneof![-1i8, 1i8],
            magnitude in 0.0f32..100.0,
        ) {
            let result = SpikeEvent::new_validated(timestamp, channel, polarity, magnitude);
            prop_assert!(result.is_ok());
        }

        /// Test SpikeEvent validated constructor rejects invalid polarity
        #[test]
        fn spike_event_validated_rejects_invalid_polarity(
            timestamp in 0.0f64..1000.0,
            channel in 0u32..1000,
            polarity in (-128i8..=-2).prop_union(2i8..=127),
            magnitude in 0.0f32..100.0,
        ) {
            let result = SpikeEvent::new_validated(timestamp, channel, polarity, magnitude);
            prop_assert!(result.is_err());
        }
    }
}
