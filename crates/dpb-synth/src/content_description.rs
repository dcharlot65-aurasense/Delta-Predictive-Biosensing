//! Content Description API for Natural Language Landmark Generation
//!
//! This module provides a natural language interface for generating synthetic
//! landmark streams based on human-readable content descriptions.
//!
//! # Overview
//!
//! Instead of manually configuring parameters like cadence, stride length, and
//! shuffling severity, users can describe what they want in natural language:
//!
//! ```rust,ignore
//! use dpb_synth::content_description::ContentDescription;
//!
//! // Parse natural language request
//! let description = ContentDescription::parse("60 seconds of parkinsonian walking with mild shuffling")?;
//!
//! // Generate landmark stream
//! let stream = description.generate_landmarks()?;
//! ```
//!
//! # Supported Content Types
//!
//! ## Gait Patterns
//! - Normal walking
//! - Parkinsonian gait (shuffling, reduced arm swing, stooped posture)
//! - Festinating gait (accelerating small steps)
//! - Freezing episodes
//!
//! ## Modifiers
//! - Duration: "30 seconds", "1 minute", "2 minutes"
//! - Severity: "mild", "moderate", "severe"
//! - Speed: "slow", "normal", "fast"
//! - Frame rate: "30 fps", "60 fps"
//!
//! # Examples
//!
//! ```rust,ignore
//! // Parkinsonian gait with freezing
//! "60 seconds of parkinsonian walk with freezing episodes"
//!
//! // Fast festinating gait
//! "30 seconds of festinating gait at 60 fps"
//!
//! // Mild symptoms
//! "1 minute of walking with mild shuffling and reduced arm swing"
//! ```

use std::str::FromStr;

use crate::streaming::{
    ClinicalGaitType, ClinicalPoseFrame, FrameStreamingGenerator,
    StreamingClinicalPose, StreamingClinicalPoseParams,
};

/// Error type for content description parsing
#[derive(Debug, Clone, PartialEq)]
pub enum ContentDescriptionError {
    /// Could not parse the content description
    ParseError(String),
    /// Invalid duration specified
    InvalidDuration(String),
    /// Unknown gait type
    UnknownGaitType(String),
    /// Invalid severity modifier
    InvalidSeverity(String),
    /// Generation error
    GenerationError(String),
}

impl std::fmt::Display for ContentDescriptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::InvalidDuration(msg) => write!(f, "Invalid duration: {}", msg),
            Self::UnknownGaitType(msg) => write!(f, "Unknown gait type: {}", msg),
            Self::InvalidSeverity(msg) => write!(f, "Invalid severity: {}", msg),
            Self::GenerationError(msg) => write!(f, "Generation error: {}", msg),
        }
    }
}

impl std::error::Error for ContentDescriptionError {}

/// Severity level for clinical symptoms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    /// Minimal symptoms (0.1-0.3)
    Mild,
    /// Noticeable symptoms (0.4-0.6)
    Moderate,
    /// Pronounced symptoms (0.7-1.0)
    Severe,
}

impl Severity {
    /// Convert severity to a numeric value (0-1)
    pub fn as_factor(&self) -> f64 {
        match self {
            Severity::Mild => 0.3,
            Severity::Moderate => 0.5,
            Severity::Severe => 0.8,
        }
    }
}

impl Default for Severity {
    fn default() -> Self {
        Severity::Moderate
    }
}

/// Output format for landmark data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LandmarkFormat {
    /// MediaPipe 33-keypoint format
    MediaPipe,
    /// OpenPose 25-keypoint body format
    OpenPoseBody,
    /// OpenPose 135-keypoint full format (body + hands + face)
    OpenPoseFull,
}

impl Default for LandmarkFormat {
    fn default() -> Self {
        LandmarkFormat::MediaPipe
    }
}

/// Parsed content description with all parameters extracted
#[derive(Debug, Clone)]
pub struct ContentDescription {
    /// Duration in seconds
    pub duration_secs: f64,
    /// Frame rate in FPS
    pub frame_rate: f64,
    /// Gait type
    pub gait_type: ClinicalGaitType,
    /// Overall severity
    pub severity: Severity,
    /// Whether freezing episodes are enabled
    pub with_freezing: bool,
    /// Custom shuffling severity (overrides severity if set)
    pub shuffling_override: Option<f64>,
    /// Custom arm swing reduction (overrides severity if set)
    pub arm_swing_reduction_override: Option<f64>,
    /// Custom trunk flexion in degrees
    pub trunk_flexion_override: Option<f64>,
    /// Subject height in meters
    pub height: f64,
    /// Output format
    pub format: LandmarkFormat,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for ContentDescription {
    fn default() -> Self {
        Self {
            duration_secs: 10.0,
            frame_rate: 30.0,
            gait_type: ClinicalGaitType::Normal,
            severity: Severity::Moderate,
            with_freezing: false,
            shuffling_override: None,
            arm_swing_reduction_override: None,
            trunk_flexion_override: None,
            height: 1.75,
            format: LandmarkFormat::MediaPipe,
            seed: None,
        }
    }
}

impl ContentDescription {
    /// Parse a natural language content description
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let desc = ContentDescription::parse("60 seconds of parkinsonian walking")?;
    /// let desc = ContentDescription::parse("1 minute of normal gait at 60 fps")?;
    /// let desc = ContentDescription::parse("30 seconds of festinating walk with severe shuffling")?;
    /// ```
    pub fn parse(input: &str) -> Result<Self, ContentDescriptionError> {
        let input_lower = input.to_lowercase();
        let mut result = ContentDescription::default();

        // Parse duration
        result.duration_secs = Self::parse_duration(&input_lower)?;

        // Parse frame rate
        if let Some(fps) = Self::parse_frame_rate(&input_lower) {
            result.frame_rate = fps;
        }

        // Parse gait type
        result.gait_type = Self::parse_gait_type(&input_lower);

        // Parse severity modifiers
        result.severity = Self::parse_severity(&input_lower);

        // Check for freezing
        result.with_freezing = input_lower.contains("freez");

        // Parse specific symptom overrides
        result.shuffling_override = Self::parse_shuffling_override(&input_lower);
        result.arm_swing_reduction_override = Self::parse_arm_swing_override(&input_lower);
        result.trunk_flexion_override = Self::parse_trunk_flexion_override(&input_lower);

        // Parse format
        result.format = Self::parse_format(&input_lower);

        // Parse seed
        result.seed = Self::parse_seed(&input_lower);

        Ok(result)
    }

    /// Parse duration from input string
    fn parse_duration(input: &str) -> Result<f64, ContentDescriptionError> {
        // Try to match patterns like "60 seconds", "1 minute", "2 minutes", "1.5 min"

        // Match seconds
        if let Some(secs) = Self::extract_number_before(input, "second") {
            return Ok(secs);
        }
        if let Some(secs) = Self::extract_number_before(input, "sec") {
            return Ok(secs);
        }
        if let Some(secs) = Self::extract_number_before(input, " s ") {
            return Ok(secs);
        }

        // Match minutes
        if let Some(mins) = Self::extract_number_before(input, "minute") {
            return Ok(mins * 60.0);
        }
        if let Some(mins) = Self::extract_number_before(input, "min") {
            return Ok(mins * 60.0);
        }
        if let Some(mins) = Self::extract_number_before(input, " m ") {
            return Ok(mins * 60.0);
        }

        // Match hours (for long recordings)
        if let Some(hours) = Self::extract_number_before(input, "hour") {
            return Ok(hours * 3600.0);
        }
        if let Some(hours) = Self::extract_number_before(input, " h ") {
            return Ok(hours * 3600.0);
        }

        // Default to 10 seconds if no duration specified
        Ok(10.0)
    }

    /// Extract a number that appears before a given suffix
    fn extract_number_before(input: &str, suffix: &str) -> Option<f64> {
        if let Some(idx) = input.find(suffix) {
            // Look for number before suffix
            let before = &input[..idx];
            // Find the last word/number before suffix
            let words: Vec<&str> = before.split_whitespace().collect();
            if let Some(last) = words.last() {
                if let Ok(num) = last.parse::<f64>() {
                    return Some(num);
                }
            }
        }
        None
    }

    /// Parse frame rate from input string
    fn parse_frame_rate(input: &str) -> Option<f64> {
        // Match "60 fps", "30fps", "at 60 fps", etc.
        if let Some(fps) = Self::extract_number_before(input, "fps") {
            return Some(fps);
        }
        if let Some(fps) = Self::extract_number_before(input, " hz") {
            return Some(fps);
        }

        // Check for common frame rates mentioned
        if input.contains("60fps") || input.contains("60 fps") {
            return Some(60.0);
        }
        if input.contains("30fps") || input.contains("30 fps") {
            return Some(30.0);
        }
        if input.contains("24fps") || input.contains("24 fps") {
            return Some(24.0);
        }

        None
    }

    /// Parse gait type from input string
    fn parse_gait_type(input: &str) -> ClinicalGaitType {
        if input.contains("parkinson") {
            ClinicalGaitType::Parkinsonian
        } else if input.contains("festinat") {
            ClinicalGaitType::Festinating
        } else if input.contains("freez") {
            ClinicalGaitType::FreezingEpisodes
        } else {
            ClinicalGaitType::Normal
        }
    }

    /// Parse severity from input string
    fn parse_severity(input: &str) -> Severity {
        if input.contains("mild") || input.contains("slight") || input.contains("minimal") {
            Severity::Mild
        } else if input.contains("severe") || input.contains("pronounced") || input.contains("heavy") {
            Severity::Severe
        } else if input.contains("moderate") || input.contains("medium") {
            Severity::Moderate
        } else {
            // Default to moderate for clinical conditions, mild for normal
            if input.contains("parkinson") || input.contains("festinat") || input.contains("freez") {
                Severity::Moderate
            } else {
                Severity::Mild
            }
        }
    }

    /// Parse shuffling severity override
    fn parse_shuffling_override(input: &str) -> Option<f64> {
        if input.contains("shuffl") {
            // Check for severity modifier near "shuffling"
            if input.contains("mild shuffl") || input.contains("slight shuffl") {
                Some(0.3)
            } else if input.contains("severe shuffl") || input.contains("heavy shuffl") {
                Some(0.8)
            } else if input.contains("moderate shuffl") {
                Some(0.5)
            } else if input.contains("no shuffl") {
                Some(0.0)
            } else {
                Some(0.5) // Default shuffling
            }
        } else {
            None
        }
    }

    /// Parse arm swing reduction override
    fn parse_arm_swing_override(input: &str) -> Option<f64> {
        if input.contains("arm swing") || input.contains("reduced arm") {
            if input.contains("no arm swing") || input.contains("minimal arm") {
                Some(0.9)
            } else if input.contains("reduced arm") || input.contains("limited arm") {
                Some(0.6)
            } else if input.contains("normal arm") {
                Some(0.0)
            } else {
                Some(0.5)
            }
        } else {
            None
        }
    }

    /// Parse trunk flexion override
    fn parse_trunk_flexion_override(input: &str) -> Option<f64> {
        if input.contains("stooped") || input.contains("bent") || input.contains("forward lean") {
            if input.contains("severe") || input.contains("pronounced") {
                Some(25.0)
            } else if input.contains("mild") || input.contains("slight") {
                Some(10.0)
            } else {
                Some(15.0)
            }
        } else {
            None
        }
    }

    /// Parse output format
    fn parse_format(input: &str) -> LandmarkFormat {
        if input.contains("openpose full") || input.contains("openpose 135") {
            LandmarkFormat::OpenPoseFull
        } else if input.contains("openpose") || input.contains("open pose") {
            LandmarkFormat::OpenPoseBody
        } else {
            LandmarkFormat::MediaPipe
        }
    }

    /// Parse random seed
    fn parse_seed(input: &str) -> Option<u64> {
        // Match "seed 42", "seed=42", etc.
        if let Some(idx) = input.find("seed") {
            let after = &input[idx + 4..];
            let words: Vec<&str> = after.split_whitespace().collect();
            if let Some(first) = words.first() {
                let cleaned = first.trim_start_matches('=').trim_start_matches(':');
                if let Ok(seed) = cleaned.parse::<u64>() {
                    return Some(seed);
                }
            }
        }
        None
    }

    /// Convert to StreamingClinicalPoseParams
    pub fn to_pose_params(&self) -> StreamingClinicalPoseParams {
        let severity_factor = self.severity.as_factor();

        // Start with appropriate base parameters
        let mut params = match self.gait_type {
            ClinicalGaitType::Normal => StreamingClinicalPoseParams::default(),
            ClinicalGaitType::Parkinsonian => StreamingClinicalPoseParams::parkinsonian(),
            ClinicalGaitType::Festinating => StreamingClinicalPoseParams::festinating(),
            ClinicalGaitType::FreezingEpisodes => StreamingClinicalPoseParams::with_freezing(),
        };

        // Apply overrides and severity adjustments
        params.frame_rate = self.frame_rate;
        params.duration = Some(self.duration_secs);
        params.height = self.height;

        // Apply shuffling override or severity-based adjustment
        if let Some(shuffling) = self.shuffling_override {
            params.shuffling_severity = shuffling;
        } else if self.gait_type != ClinicalGaitType::Normal {
            params.shuffling_severity = params.shuffling_severity * severity_factor / 0.5;
            params.shuffling_severity = params.shuffling_severity.min(1.0);
        }

        // Apply arm swing override or severity-based adjustment
        if let Some(arm_swing) = self.arm_swing_reduction_override {
            params.arm_swing_reduction = arm_swing;
        } else if self.gait_type != ClinicalGaitType::Normal {
            params.arm_swing_reduction = params.arm_swing_reduction * severity_factor / 0.5;
            params.arm_swing_reduction = params.arm_swing_reduction.min(1.0);
        }

        // Apply trunk flexion override
        if let Some(trunk) = self.trunk_flexion_override {
            params.trunk_flexion = trunk;
        }

        // Enable freezing if requested
        if self.with_freezing && self.gait_type != ClinicalGaitType::FreezingEpisodes {
            params.freezing_probability = 0.1 * severity_factor;
        }

        params
    }

    /// Generate landmark frames from this content description
    ///
    /// Returns a vector of ClinicalPoseFrame with the specified number of frames
    /// based on duration and frame rate.
    pub fn generate_landmarks(&self) -> Result<Vec<ClinicalPoseFrame>, ContentDescriptionError> {
        let params = self.to_pose_params();
        let generator = StreamingClinicalPose;
        let seed = self.seed.unwrap_or(42);

        let mut state = generator.init_state(&params, seed);
        let total_frames = (self.duration_secs * self.frame_rate) as usize;

        let mut frames = Vec::with_capacity(total_frames);
        for _ in 0..total_frames {
            let frame = generator.next_frame(&mut state);
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Generate landmarks as a streaming iterator
    pub fn stream_landmarks(&self) -> LandmarkStream {
        let params = self.to_pose_params();
        let seed = self.seed.unwrap_or(42);

        LandmarkStream::new(params, seed, self.duration_secs, self.frame_rate)
    }

    /// Get total number of frames that will be generated
    pub fn total_frames(&self) -> usize {
        (self.duration_secs * self.frame_rate) as usize
    }

    /// Get the effective duration in seconds
    pub fn duration(&self) -> f64 {
        self.duration_secs
    }
}

impl FromStr for ContentDescription {
    type Err = ContentDescriptionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        ContentDescription::parse(s)
    }
}

/// Streaming iterator for landmark generation
pub struct LandmarkStream {
    generator: StreamingClinicalPose,
    state: <StreamingClinicalPose as FrameStreamingGenerator>::State,
    total_frames: usize,
    current_frame: usize,
}

impl LandmarkStream {
    /// Create a new landmark stream
    pub fn new(params: StreamingClinicalPoseParams, seed: u64, duration: f64, frame_rate: f64) -> Self {
        let generator = StreamingClinicalPose;
        let state = generator.init_state(&params, seed);
        let total_frames = (duration * frame_rate) as usize;

        Self {
            generator,
            state,
            total_frames,
            current_frame: 0,
        }
    }

    /// Get remaining frames
    pub fn remaining(&self) -> usize {
        self.total_frames.saturating_sub(self.current_frame)
    }

    /// Get progress as a fraction (0.0 - 1.0)
    pub fn progress(&self) -> f64 {
        if self.total_frames == 0 {
            1.0
        } else {
            self.current_frame as f64 / self.total_frames as f64
        }
    }
}

impl Iterator for LandmarkStream {
    type Item = ClinicalPoseFrame;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_frame >= self.total_frames {
            return None;
        }

        let frame = self.generator.next_frame(&mut self.state);
        self.current_frame += 1;
        Some(frame)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for LandmarkStream {}

/// Builder for ContentDescription with fluent API
#[derive(Debug, Clone, Default)]
pub struct ContentDescriptionBuilder {
    inner: ContentDescription,
}

impl ContentDescriptionBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set duration in seconds
    pub fn duration_secs(mut self, secs: f64) -> Self {
        self.inner.duration_secs = secs;
        self
    }

    /// Set duration in minutes
    pub fn duration_mins(mut self, mins: f64) -> Self {
        self.inner.duration_secs = mins * 60.0;
        self
    }

    /// Set frame rate
    pub fn frame_rate(mut self, fps: f64) -> Self {
        self.inner.frame_rate = fps;
        self
    }

    /// Set gait type
    pub fn gait_type(mut self, gait_type: ClinicalGaitType) -> Self {
        self.inner.gait_type = gait_type;
        self
    }

    /// Set as normal gait
    pub fn normal(mut self) -> Self {
        self.inner.gait_type = ClinicalGaitType::Normal;
        self
    }

    /// Set as parkinsonian gait
    pub fn parkinsonian(mut self) -> Self {
        self.inner.gait_type = ClinicalGaitType::Parkinsonian;
        self
    }

    /// Set as festinating gait
    pub fn festinating(mut self) -> Self {
        self.inner.gait_type = ClinicalGaitType::Festinating;
        self
    }

    /// Set as freezing episodes gait
    pub fn with_freezing(mut self) -> Self {
        self.inner.gait_type = ClinicalGaitType::FreezingEpisodes;
        self.inner.with_freezing = true;
        self
    }

    /// Set severity
    pub fn severity(mut self, severity: Severity) -> Self {
        self.inner.severity = severity;
        self
    }

    /// Set mild severity
    pub fn mild(mut self) -> Self {
        self.inner.severity = Severity::Mild;
        self
    }

    /// Set moderate severity
    pub fn moderate(mut self) -> Self {
        self.inner.severity = Severity::Moderate;
        self
    }

    /// Set severe severity
    pub fn severe(mut self) -> Self {
        self.inner.severity = Severity::Severe;
        self
    }

    /// Set shuffling severity directly (0-1)
    pub fn shuffling(mut self, severity: f64) -> Self {
        self.inner.shuffling_override = Some(severity.clamp(0.0, 1.0));
        self
    }

    /// Set arm swing reduction directly (0-1)
    pub fn arm_swing_reduction(mut self, reduction: f64) -> Self {
        self.inner.arm_swing_reduction_override = Some(reduction.clamp(0.0, 1.0));
        self
    }

    /// Set trunk flexion in degrees
    pub fn trunk_flexion(mut self, degrees: f64) -> Self {
        self.inner.trunk_flexion_override = Some(degrees.clamp(0.0, 45.0));
        self
    }

    /// Set subject height
    pub fn height(mut self, meters: f64) -> Self {
        self.inner.height = meters;
        self
    }

    /// Set output format
    pub fn format(mut self, format: LandmarkFormat) -> Self {
        self.inner.format = format;
        self
    }

    /// Set random seed
    pub fn seed(mut self, seed: u64) -> Self {
        self.inner.seed = Some(seed);
        self
    }

    /// Build the ContentDescription
    pub fn build(self) -> ContentDescription {
        self.inner
    }
}

/// Convenience function to generate landmarks from a content description string
pub fn generate_from_description(
    description: &str,
) -> Result<Vec<ClinicalPoseFrame>, ContentDescriptionError> {
    ContentDescription::parse(description)?.generate_landmarks()
}

/// Convenience function to create a streaming iterator from a content description string
pub fn stream_from_description(
    description: &str,
) -> Result<LandmarkStream, ContentDescriptionError> {
    Ok(ContentDescription::parse(description)?.stream_landmarks())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duration_seconds() {
        let desc = ContentDescription::parse("60 seconds of walking").unwrap();
        assert_eq!(desc.duration_secs, 60.0);

        let desc = ContentDescription::parse("30 sec walking").unwrap();
        assert_eq!(desc.duration_secs, 30.0);
    }

    #[test]
    fn test_parse_duration_minutes() {
        let desc = ContentDescription::parse("1 minute of walking").unwrap();
        assert_eq!(desc.duration_secs, 60.0);

        let desc = ContentDescription::parse("2 minutes of parkinsonian gait").unwrap();
        assert_eq!(desc.duration_secs, 120.0);

        let desc = ContentDescription::parse("1.5 min walking").unwrap();
        assert_eq!(desc.duration_secs, 90.0);
    }

    #[test]
    fn test_parse_frame_rate() {
        let desc = ContentDescription::parse("60 seconds at 60 fps").unwrap();
        assert_eq!(desc.frame_rate, 60.0);

        let desc = ContentDescription::parse("30 seconds at 30fps").unwrap();
        assert_eq!(desc.frame_rate, 30.0);
    }

    #[test]
    fn test_parse_gait_type() {
        let desc = ContentDescription::parse("60 seconds of normal walking").unwrap();
        assert_eq!(desc.gait_type, ClinicalGaitType::Normal);

        let desc = ContentDescription::parse("60 seconds of parkinsonian gait").unwrap();
        assert_eq!(desc.gait_type, ClinicalGaitType::Parkinsonian);

        let desc = ContentDescription::parse("60 seconds of festinating walk").unwrap();
        assert_eq!(desc.gait_type, ClinicalGaitType::Festinating);

        let desc = ContentDescription::parse("60 seconds with freezing episodes").unwrap();
        assert_eq!(desc.gait_type, ClinicalGaitType::FreezingEpisodes);
    }

    #[test]
    fn test_parse_severity() {
        let desc = ContentDescription::parse("60 seconds of mild parkinsonian gait").unwrap();
        assert_eq!(desc.severity, Severity::Mild);

        let desc = ContentDescription::parse("60 seconds of severe shuffling").unwrap();
        assert_eq!(desc.severity, Severity::Severe);
    }

    #[test]
    fn test_parse_shuffling_override() {
        let desc = ContentDescription::parse("60 seconds with mild shuffling").unwrap();
        assert_eq!(desc.shuffling_override, Some(0.3));

        let desc = ContentDescription::parse("60 seconds with severe shuffling").unwrap();
        assert_eq!(desc.shuffling_override, Some(0.8));
    }

    #[test]
    fn test_parse_complex_description() {
        let desc = ContentDescription::parse(
            "60 seconds of parkinsonian walking with mild shuffling and reduced arm swing at 60 fps"
        ).unwrap();

        assert_eq!(desc.duration_secs, 60.0);
        assert_eq!(desc.frame_rate, 60.0);
        assert_eq!(desc.gait_type, ClinicalGaitType::Parkinsonian);
        assert_eq!(desc.shuffling_override, Some(0.3));
        assert!(desc.arm_swing_reduction_override.is_some());
    }

    #[test]
    fn test_to_pose_params() {
        let desc = ContentDescription::parse("60 seconds of parkinsonian walking").unwrap();
        let params = desc.to_pose_params();

        assert_eq!(params.duration, Some(60.0));
        assert_eq!(params.gait_type, ClinicalGaitType::Parkinsonian);
        assert!(params.shuffling_severity > 0.0);
    }

    #[test]
    fn test_generate_landmarks() {
        let desc = ContentDescription::parse("1 second at 30 fps").unwrap();
        let landmarks = desc.generate_landmarks().unwrap();

        assert_eq!(landmarks.len(), 30);
        for frame in &landmarks {
            assert_eq!(frame.keypoints.len(), 33); // MediaPipe format
        }
    }

    #[test]
    fn test_stream_landmarks() {
        let desc = ContentDescription::parse("1 second at 30 fps").unwrap();
        let stream = desc.stream_landmarks();

        assert_eq!(stream.remaining(), 30);

        let frames: Vec<_> = stream.collect();
        assert_eq!(frames.len(), 30);
    }

    #[test]
    fn test_builder_api() {
        let desc = ContentDescriptionBuilder::new()
            .duration_secs(60.0)
            .frame_rate(60.0)
            .parkinsonian()
            .mild()
            .shuffling(0.4)
            .seed(42)
            .build();

        assert_eq!(desc.duration_secs, 60.0);
        assert_eq!(desc.frame_rate, 60.0);
        assert_eq!(desc.gait_type, ClinicalGaitType::Parkinsonian);
        assert_eq!(desc.severity, Severity::Mild);
        assert_eq!(desc.shuffling_override, Some(0.4));
        assert_eq!(desc.seed, Some(42));
    }

    #[test]
    fn test_60_second_video_generation() {
        // This is the specific test case mentioned by the user
        let desc = ContentDescription::parse(
            "60 seconds of parkinsonian walking with mild shuffling at 30 fps"
        ).unwrap();

        assert_eq!(desc.duration_secs, 60.0);
        assert_eq!(desc.frame_rate, 30.0);
        assert_eq!(desc.total_frames(), 1800); // 60 * 30

        // Generate landmarks
        let landmarks = desc.generate_landmarks().unwrap();
        assert_eq!(landmarks.len(), 1800);

        // Verify frame structure
        let first_frame = &landmarks[0];
        assert_eq!(first_frame.keypoints.len(), 33);
        assert!(first_frame.gait_phase >= 0.0 && first_frame.gait_phase <= 1.0);
    }

    #[test]
    fn test_format_parsing() {
        let desc = ContentDescription::parse("60 seconds openpose format").unwrap();
        assert_eq!(desc.format, LandmarkFormat::OpenPoseBody);

        let desc = ContentDescription::parse("60 seconds openpose full format").unwrap();
        assert_eq!(desc.format, LandmarkFormat::OpenPoseFull);

        let desc = ContentDescription::parse("60 seconds mediapipe").unwrap();
        assert_eq!(desc.format, LandmarkFormat::MediaPipe);
    }

    #[test]
    fn test_seed_parsing() {
        let desc = ContentDescription::parse("60 seconds with seed 12345").unwrap();
        assert_eq!(desc.seed, Some(12345));

        let desc = ContentDescription::parse("60 seconds seed=42").unwrap();
        assert_eq!(desc.seed, Some(42));
    }
}
