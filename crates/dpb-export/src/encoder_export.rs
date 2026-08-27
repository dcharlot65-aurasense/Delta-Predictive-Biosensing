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

        if let Some(state) = &self.state
            && state.channel_states.len() != self.params.num_channels {
                return Err(ExportError::validation(
                    "state channel count doesn't match params",
                ));
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

    /// Create a mock temporal contrast encoder.
    pub fn temporal_contrast(
        num_channels: usize,
        sample_rate: f64,
        threshold: f32,
        refractory_period: f32,
    ) -> Self {
        Self {
            encoder_type: "temporal_contrast".to_string(),
            params: EncoderParams::temporal_contrast(
                num_channels,
                sample_rate,
                threshold,
                refractory_period,
            ),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_params_new() {
        let params = EncoderParams::new("test_encoder", 8, 256.0);
        assert_eq!(params.encoder_type, "test_encoder");
        assert_eq!(params.num_channels, 8);
        assert_eq!(params.sample_rate, 256.0);
        assert!(params.thresholds.is_empty());
        assert!(params.num_levels.is_none());
        assert!(!params.adaptive);
    }

    #[test]
    fn test_encoder_params_level_crossing() {
        let params = EncoderParams::level_crossing(8, 256.0, 0.1);
        assert_eq!(params.encoder_type, "level_crossing");
        assert_eq!(params.thresholds.len(), 8);
        assert!(params.thresholds.iter().all(|&t| (t - 0.1).abs() < 1e-6));
    }

    #[test]
    fn test_encoder_params_delta() {
        let params = EncoderParams::delta(4, 512.0, 0.05, 16);
        assert_eq!(params.encoder_type, "delta");
        assert_eq!(params.num_channels, 4);
        assert_eq!(params.sample_rate, 512.0);
        assert_eq!(params.thresholds.len(), 4);
        assert_eq!(params.num_levels, Some(16));
    }

    #[test]
    fn test_encoder_params_temporal_contrast() {
        let params = EncoderParams::temporal_contrast(16, 1024.0, 0.08, 0.002);
        assert_eq!(params.encoder_type, "temporal_contrast");
        assert_eq!(params.num_channels, 16);
        assert_eq!(params.refractory_period, Some(0.002));
    }

    #[test]
    fn test_encoder_params_with_adaptive() {
        let params = EncoderParams::level_crossing(8, 256.0, 0.1)
            .with_adaptive(0.01);
        assert!(params.adaptive);
        assert_eq!(params.adaptation_rate, Some(0.01));
    }

    #[test]
    fn test_encoder_params_with_thresholds() {
        let custom_thresholds = vec![0.1, 0.2, 0.15, 0.12];
        let params = EncoderParams::new("custom", 4, 256.0)
            .with_thresholds(custom_thresholds.clone());
        assert_eq!(params.thresholds, custom_thresholds);
    }

    #[test]
    fn test_encoder_params_with_extra() {
        let params = EncoderParams::new("custom", 4, 256.0)
            .with_extra("custom_param", serde_json::json!(42));
        assert!(params.extra.contains_key("custom_param"));
        assert_eq!(params.extra.get("custom_param").unwrap(), &serde_json::json!(42));
    }

    #[test]
    fn test_encoder_params_validate_success() {
        let params = EncoderParams::level_crossing(8, 256.0, 0.1);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_encoder_params_validate_zero_channels() {
        let params = EncoderParams::new("test", 0, 256.0);
        let result = params.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("num_channels"));
    }

    #[test]
    fn test_encoder_params_validate_negative_sample_rate() {
        let params = EncoderParams::new("test", 8, -1.0);
        let result = params.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sample_rate"));
    }

    #[test]
    fn test_encoder_params_validate_threshold_mismatch() {
        let params = EncoderParams::new("test", 8, 256.0)
            .with_thresholds(vec![0.1, 0.2, 0.3]); // Only 3 thresholds for 8 channels
        let result = params.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("thresholds length"));
    }

    #[test]
    fn test_encoder_state_new() {
        let state = EncoderState::new(4);
        assert_eq!(state.channel_states.len(), 4);
        assert!(state.global.is_empty());
        for ch_state in &state.channel_states {
            assert_eq!(ch_state.last_value, 0.0);
            assert_eq!(ch_state.spike_count, 0);
        }
    }

    #[test]
    fn test_encoder_state_with_global() {
        let state = EncoderState::new(2)
            .with_global("iteration", serde_json::json!(100));
        assert!(state.global.contains_key("iteration"));
    }

    #[test]
    fn test_channel_state_default() {
        let state = ChannelState::default();
        assert_eq!(state.last_value, 0.0);
        assert_eq!(state.current_threshold, 0.0);
        assert_eq!(state.current_level, 0);
        assert_eq!(state.time_since_spike, 0.0);
        assert_eq!(state.running_mean, 0.0);
        assert_eq!(state.running_var, 0.0);
        assert_eq!(state.spike_count, 0);
    }

    #[test]
    fn test_encoder_export_new() {
        let params = EncoderParams::level_crossing(8, 256.0, 0.1);
        let export = EncoderExport::new(params);
        assert_eq!(export.schema_version, "1.0");
        assert!(export.state.is_none());
        assert!(export.checksum.is_none());
    }

    #[test]
    fn test_encoder_export_with_state() {
        let params = EncoderParams::level_crossing(4, 256.0, 0.1);
        let state = EncoderState::new(4);
        let export = EncoderExport::new(params).with_state(state);
        assert!(export.state.is_some());
        assert_eq!(export.state.as_ref().unwrap().channel_states.len(), 4);
    }

    #[test]
    fn test_encoder_export_with_checksum() {
        let params = EncoderParams::level_crossing(8, 256.0, 0.1);
        let export = EncoderExport::new(params).with_checksum();
        assert!(export.checksum.is_some());
        assert!(!export.checksum.as_ref().unwrap().is_empty());
    }

    #[test]
    fn test_encoder_export_verify_checksum() {
        let params = EncoderParams::level_crossing(8, 256.0, 0.1);
        let export = EncoderExport::new(params).with_checksum();
        assert!(export.verify_checksum());
    }

    #[test]
    fn test_encoder_export_verify_no_checksum() {
        let params = EncoderParams::level_crossing(8, 256.0, 0.1);
        let export = EncoderExport::new(params);
        // No checksum set, should return true
        assert!(export.verify_checksum());
    }

    #[test]
    fn test_encoder_export_validate_success() {
        let params = EncoderParams::level_crossing(4, 256.0, 0.1);
        let state = EncoderState::new(4);
        let export = EncoderExport::new(params)
            .with_state(state)
            .with_checksum();
        assert!(export.validate().is_ok());
    }

    #[test]
    fn test_encoder_export_validate_state_mismatch() {
        let params = EncoderParams::level_crossing(4, 256.0, 0.1);
        let state = EncoderState::new(8); // Wrong channel count
        let export = EncoderExport::new(params).with_state(state);
        let result = export.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("channel count"));
    }

    #[test]
    fn test_mock_encoder_level_crossing() {
        let encoder = MockEncoder::level_crossing(8, 256.0, 0.1);
        assert_eq!(encoder.encoder_type(), "level_crossing");
        assert_eq!(encoder.num_channels(), 8);

        let params = encoder.get_params();
        assert_eq!(params.encoder_type, "level_crossing");
        assert_eq!(params.sample_rate, 256.0);
    }

    #[test]
    fn test_mock_encoder_delta() {
        let encoder = MockEncoder::delta(4, 512.0, 0.05, 16);
        assert_eq!(encoder.encoder_type(), "delta");
        assert_eq!(encoder.num_channels(), 4);

        let params = encoder.get_params();
        assert_eq!(params.num_levels, Some(16));
    }

    #[test]
    fn test_mock_encoder_validate_for_export() {
        let encoder = MockEncoder::level_crossing(8, 256.0, 0.1);
        assert!(encoder.validate_for_export().is_ok());
    }

    #[test]
    fn test_mock_encoder_get_state() {
        let encoder = MockEncoder::level_crossing(8, 256.0, 0.1);
        // MockEncoder doesn't have state
        assert!(encoder.get_state().is_none());
    }

    #[test]
    fn test_encoder_params_serialization() {
        let params = EncoderParams::level_crossing(8, 256.0, 0.1)
            .with_adaptive(0.01);

        let json = serde_json::to_string(&params).unwrap();
        let deserialized: EncoderParams = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.encoder_type, params.encoder_type);
        assert_eq!(deserialized.num_channels, params.num_channels);
        assert_eq!(deserialized.sample_rate, params.sample_rate);
        assert_eq!(deserialized.adaptive, params.adaptive);
    }

    #[test]
    fn test_encoder_state_serialization() {
        let state = EncoderState::new(4)
            .with_global("test_key", serde_json::json!("test_value"));

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: EncoderState = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.channel_states.len(), state.channel_states.len());
        assert!(deserialized.global.contains_key("test_key"));
    }

    #[test]
    fn test_encoder_export_serialization() {
        let params = EncoderParams::level_crossing(4, 256.0, 0.1);
        let export = EncoderExport::new(params).with_checksum();

        let json = serde_json::to_string(&export).unwrap();
        let deserialized: EncoderExport = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.schema_version, export.schema_version);
        assert_eq!(deserialized.checksum, export.checksum);
    }

    #[test]
    fn test_channel_state_serialization() {
        let state = ChannelState {
            last_value: 1.5,
            current_threshold: 0.1,
            spike_count: 42,
            ..Default::default()
        };

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: ChannelState = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.last_value, state.last_value);
        assert_eq!(deserialized.current_threshold, state.current_threshold);
        assert_eq!(deserialized.spike_count, state.spike_count);
    }
}
