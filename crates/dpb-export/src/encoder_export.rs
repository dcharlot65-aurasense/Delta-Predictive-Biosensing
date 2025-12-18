//! Encoder export traits and structures.

use crate::{ExportError, Result};
use serde::{Deserialize, Serialize};

/// Trait for encoders that can be exported.
pub trait ExportableEncoder {
    /// Get the encoder type name.
    fn encoder_type(&self) -> &str;

    /// Get the encoder parameters.
    fn get_params(&self) -> EncoderParams;

    /// Get the number of input channels.
    fn num_channels(&self) -> usize;

    /// Get internal state for stateful encoders.
    fn get_state(&self) -> Option<EncoderState> {
        None
    }

    /// Validate the encoder can be exported.
    fn validate_for_export(&self) -> Result<()> {
        Ok(())
    }
}

/// Generic encoder parameters for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderParams {
    /// Encoder type identifier.
    pub encoder_type: String,
    /// Number of channels.
    pub num_channels: usize,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Threshold value(s).
    pub thresholds: Vec<f32>,
    /// Number of quantization levels (for level-based encoders).
    pub num_levels: Option<usize>,
    /// Refractory period in seconds.
    pub refractory_period: Option<f32>,
    /// Adaptive encoding enabled.
    pub adaptive: bool,
    /// Adaptation rate.
    pub adaptation_rate: Option<f32>,
    /// Additional encoder-specific parameters.
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

impl EncoderParams {
    /// Create new encoder parameters.
    pub fn new(encoder_type: impl Into<String>, num_channels: usize, sample_rate: f64) -> Self {
        Self {
            encoder_type: encoder_type.into(),
            num_channels,
            sample_rate,
            thresholds: Vec::new(),
            num_levels: None,
            refractory_period: None,
            adaptive: false,
            adaptation_rate: None,
            extra: std::collections::HashMap::new(),
        }
    }

    /// Create parameters for a level crossing encoder.
    pub fn level_crossing(num_channels: usize, sample_rate: f64, threshold: f32) -> Self {
        let mut params = Self::new("level_crossing", num_channels, sample_rate);
        params.thresholds = vec![threshold; num_channels];
        params
    }

    /// Create parameters for a delta encoder.
    pub fn delta(num_channels: usize, sample_rate: f64, threshold: f32, num_levels: usize) -> Self {
        let mut params = Self::new("delta", num_channels, sample_rate);
        params.thresholds = vec![threshold; num_channels];
        params.num_levels = Some(num_levels);
        params
    }

    /// Create parameters for a temporal contrast encoder.
    pub fn temporal_contrast(
        num_channels: usize,
        sample_rate: f64,
        threshold: f32,
        refractory_period: f32,
    ) -> Self {
        let mut params = Self::new("temporal_contrast", num_channels, sample_rate);
        params.thresholds = vec![threshold; num_channels];
        params.refractory_period = Some(refractory_period);
        params
    }

    /// Enable adaptive thresholds.
    pub fn with_adaptive(mut self, rate: f32) -> Self {
        self.adaptive = true;
        self.adaptation_rate = Some(rate);
        self
    }

    /// Set per-channel thresholds.
    pub fn with_thresholds(mut self, thresholds: Vec<f32>) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Add extra parameter.
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Validate parameters.
    pub fn validate(&self) -> Result<()> {
        if self.num_channels == 0 {
            return Err(ExportError::validation("num_channels must be positive"));
        }

        if self.sample_rate <= 0.0 {
            return Err(ExportError::validation("sample_rate must be positive"));
        }

        if !self.thresholds.is_empty() && self.thresholds.len() != self.num_channels {
            return Err(ExportError::validation(format!(
                "thresholds length ({}) must match num_channels ({})",
                self.thresholds.len(),
                self.num_channels
            )));
        }

        Ok(())
    }
}

/// Internal state for stateful encoders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderState {
    /// Per-channel state values.
    pub channel_states: Vec<ChannelState>,
    /// Global encoder state.
    pub global: std::collections::HashMap<String, serde_json::Value>,
}

impl EncoderState {
    /// Create new encoder state.
    pub fn new(num_channels: usize) -> Self {
        Self {
            channel_states: (0..num_channels).map(|_| ChannelState::default()).collect(),
            global: std::collections::HashMap::new(),
        }
    }

    /// Add global state.
    pub fn with_global(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.global.insert(key.into(), value);
        self
    }
}

/// Per-channel encoder state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChannelState {
    /// Last value seen.
    pub last_value: f32,
    /// Current threshold (may be adapted).
    pub current_threshold: f32,
    /// Current quantization level.
    pub current_level: i32,
    /// Time since last spike.
    pub time_since_spike: f32,
    /// Running statistics for adaptation.
    pub running_mean: f32,
    /// Running variance for adaptation.
    pub running_var: f32,
    /// Spike count.
    pub spike_count: u64,
}

/// Complete encoder export bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderExport {
    /// Schema version for compatibility.
    pub schema_version: String,
    /// Encoder parameters.
    pub params: EncoderParams,
    /// Optional internal state.
    pub state: Option<EncoderState>,
    /// Export timestamp.
    pub exported_at: chrono::DateTime<chrono::Utc>,
    /// Checksum for integrity verification.
    pub checksum: Option<String>,
}

impl EncoderExport {
    /// Create a new encoder export.
    pub fn new(params: EncoderParams) -> Self {
        Self {
            schema_version: "1.0".to_string(),
            params,
            state: None,
            exported_at: chrono::Utc::now(),
            checksum: None,
        }
    }

    /// Include state in export.
    pub fn with_state(mut self, state: EncoderState) -> Self {
        self.state = Some(state);
        self
    }

    /// Compute and set checksum.
    pub fn with_checksum(mut self) -> Self {
        // Simple checksum based on parameters
        let data = serde_json::to_string(&self.params).unwrap_or_default();
        let checksum = format!("{:x}", md5_hash(&data));
        self.checksum = Some(checksum);
        self
    }

    /// Verify the checksum.
    pub fn verify_checksum(&self) -> bool {
        match &self.checksum {
            Some(expected) => {
                let data = serde_json::to_string(&self.params).unwrap_or_default();
                let actual = format!("{:x}", md5_hash(&data));
                &actual == expected
            }
            None => true, // No checksum to verify
        }
    }

    /// Validate the export.
    pub fn validate(&self) -> Result<()> {
        self.params.validate()?;

        if let Some(state) = &self.state {
            if state.channel_states.len() != self.params.num_channels {
                return Err(ExportError::validation(
                    "state channel count doesn't match params",
                ));
            }
        }

        if !self.verify_checksum() {
            return Err(ExportError::validation("checksum mismatch"));
        }

        Ok(())
    }
}

/// Simple hash function for checksums.
fn md5_hash(data: &str) -> u128 {
    // Simple non-cryptographic hash for checksum purposes
    let mut hash: u128 = 0;
    for (i, byte) in data.bytes().enumerate() {
        hash = hash.wrapping_add((byte as u128) << ((i % 16) * 8));
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Mock encoder for testing export functionality.
#[derive(Debug, Clone)]
pub struct MockEncoder {
    encoder_type: String,
    params: EncoderParams,
}

impl MockEncoder {
    /// Create a mock level crossing encoder.
    pub fn level_crossing(num_channels: usize, sample_rate: f64, threshold: f32) -> Self {
        Self {
            encoder_type: "level_crossing".to_string(),
            params: EncoderParams::level_crossing(num_channels, sample_rate, threshold),
        }
    }

    /// Create a mock delta encoder.
    pub fn delta(num_channels: usize, sample_rate: f64, threshold: f32, num_levels: usize) -> Self {
        Self {
            encoder_type: "delta".to_string(),
            params: EncoderParams::delta(num_channels, sample_rate, threshold, num_levels),
        }
    }
}

impl ExportableEncoder for MockEncoder {
    fn encoder_type(&self) -> &str {
        &self.encoder_type
    }

    fn get_params(&self) -> EncoderParams {
        self.params.clone()
    }

    fn num_channels(&self) -> usize {
        self.params.num_channels
    }
}
