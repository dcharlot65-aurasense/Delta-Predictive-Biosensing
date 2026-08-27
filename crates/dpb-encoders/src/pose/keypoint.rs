//! Keypoint deviation encoders for pose analysis

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates for Pose
// ============================================================================

/// Joint angle template (e.g., knee extension during gait)
pub struct JointAngleTemplate;

impl PopulationTemplate for JointAngleTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Example: knee flexion angle during stance
        15.0 // degrees
    }

    fn variance(&self, _context: &Context) -> f64 {
        10.0
    }

    fn name(&self) -> &str {
        "JointAngleTemplate"
    }
}

/// Body sway template
pub struct BodySwayTemplate;

impl PopulationTemplate for BodySwayTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Center of mass sway in cm
        2.0
    }

    fn variance(&self, _context: &Context) -> f64 {
        1.0
    }

    fn name(&self) -> &str {
        "BodySwayTemplate"
    }
}

// ============================================================================
// Keypoint Deviation Encoder
// ============================================================================

/// Configuration for [`KeypointDeviationEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeypointDeviationConfig {
    /// Expected keypoint positions (template)
    pub template: Vec<(f32, f32)>, // (x, y) positions
    /// Deviation threshold in pixels or normalized coordinates
    pub threshold: f32,
    /// Keypoint indices to track
    pub tracked_keypoints: Vec<usize>,
}

impl Default for KeypointDeviationConfig {
    fn default() -> Self {
        Self {
            template: Vec::new(),
            threshold: 0.1,
            tracked_keypoints: Vec::new(),
        }
    }
}

/// Keypoint deviation encoder.
pub struct KeypointDeviationEncoder;

impl KeypointDeviationEncoder {
    /// Creates a new [`KeypointDeviationEncoder`].
    pub fn new() -> Self {
        Self
    }

    fn calculate_distance(&self, p1: (f32, f32), p2: (f32, f32)) -> f32 {
        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;
        (dx * dx + dy * dy).sqrt()
    }
}

impl Default for KeypointDeviationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for KeypointDeviationEncoder {
    type Config = KeypointDeviationConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Assume signal contains interleaved (x, y) coordinates for each keypoint
        let num_keypoints = config.template.len();
        if num_keypoints == 0 {
            return Ok(Vec::new());
        }

        let coords_per_frame = num_keypoints * 2;
        let num_frames = samples.len() / coords_per_frame;

        let mut events = Vec::new();

        for frame_idx in 0..num_frames {
            let frame_offset = frame_idx * coords_per_frame;

            for &keypoint_idx in &config.tracked_keypoints {
                if keypoint_idx >= num_keypoints {
                    continue;
                }

                let x_idx = frame_offset + keypoint_idx * 2;
                let y_idx = frame_offset + keypoint_idx * 2 + 1;

                if y_idx >= samples.len() {
                    break;
                }

                let observed = (samples[x_idx], samples[y_idx]);
                let template = config.template[keypoint_idx];

                let distance = self.calculate_distance(observed, template);

                if distance > config.threshold {
                    let time = frame_idx as f64 * dt;
                    events.push(SpikeEvent::new(time, keypoint_idx as u32, 1, distance));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "KeypointDeviationEncoder"
    }
}

// ============================================================================
// Joint Angle Encoder
// ============================================================================

/// Configuration for [`JointAngleEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointAngleConfig {
    /// Joint definition: (proximal_keypoint, joint_keypoint, distal_keypoint)
    pub joint_definition: (usize, usize, usize),
    /// Expected angle in degrees
    pub expected_angle: f32,
    /// Threshold for deviation (degrees)
    pub threshold: f32,
}

impl Default for JointAngleConfig {
    fn default() -> Self {
        Self {
            joint_definition: (0, 1, 2),
            expected_angle: 180.0, // Straight
            threshold: 15.0,
        }
    }
}

/// Joint angle encoder.
pub struct JointAngleEncoder;

impl JointAngleEncoder {
    /// Creates a new [`JointAngleEncoder`].
    pub fn new() -> Self {
        Self {}
    }

    fn calculate_angle(&self, p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> f32 {
        let v1 = (p1.0 - p2.0, p1.1 - p2.1);
        let v2 = (p3.0 - p2.0, p3.1 - p2.1);

        let dot = v1.0 * v2.0 + v1.1 * v2.1;
        let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
        let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();

        if mag1 == 0.0 || mag2 == 0.0 {
            return 0.0;
        }

        let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
        cos_angle.acos().to_degrees()
    }
}

impl Default for JointAngleEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for JointAngleEncoder {
    type Config = JointAngleConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let (p1_idx, p2_idx, p3_idx) = config.joint_definition;

        // Assume 2D coordinates per keypoint
        let coords_per_keypoint = 2;
        let max_keypoint = p1_idx.max(p2_idx).max(p3_idx);
        let coords_per_frame = (max_keypoint + 1) * coords_per_keypoint;

        if samples.len() < coords_per_frame {
            return Ok(Vec::new());
        }

        let num_frames = samples.len() / coords_per_frame;
        let mut events = Vec::new();

        for frame_idx in 0..num_frames {
            let offset = frame_idx * coords_per_frame;

            let p1 = (
                samples[offset + p1_idx * 2],
                samples[offset + p1_idx * 2 + 1],
            );
            let p2 = (
                samples[offset + p2_idx * 2],
                samples[offset + p2_idx * 2 + 1],
            );
            let p3 = (
                samples[offset + p3_idx * 2],
                samples[offset + p3_idx * 2 + 1],
            );

            let angle = self.calculate_angle(p1, p2, p3);
            let deviation = (angle - config.expected_angle).abs();

            if deviation > config.threshold {
                let time = frame_idx as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, deviation));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "JointAngleEncoder"
    }
}

// ============================================================================
// Body Sway Encoder
// ============================================================================

/// Configuration for [`BodySwayEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodySwayConfig {
    /// Center of mass keypoint index
    pub com_keypoint: usize,
    /// Window size for sway calculation
    pub window_size: usize,
    /// Sway threshold
    pub threshold: f32,
}

impl Default for BodySwayConfig {
    fn default() -> Self {
        Self {
            com_keypoint: 0,
            window_size: 30,
            threshold: 0.05,
        }
    }
}

/// Body sway encoder.
pub struct BodySwayEncoder;

impl BodySwayEncoder {
    /// Creates a new [`BodySwayEncoder`].
    pub fn new() -> Self {
        Self {}
    }

    fn calculate_sway(&self, positions: &[(f32, f32)]) -> f32 {
        if positions.len() < 2 {
            return 0.0;
        }

        let mean_x = positions.iter().map(|p| p.0).sum::<f32>() / positions.len() as f32;
        let mean_y = positions.iter().map(|p| p.1).sum::<f32>() / positions.len() as f32;

        let mut max_dist = 0.0f32;
        for pos in positions {
            let dx = pos.0 - mean_x;
            let dy = pos.1 - mean_y;
            let dist = (dx * dx + dy * dy).sqrt();
            max_dist = max_dist.max(dist);
        }

        max_dist
    }
}

impl Default for BodySwayEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for BodySwayEncoder {
    type Config = BodySwayConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let coords_per_keypoint = 2;
        let coords_per_frame = (config.com_keypoint + 1) * coords_per_keypoint;

        if samples.len() < coords_per_frame * config.window_size {
            return Ok(Vec::new());
        }

        let num_frames = samples.len() / coords_per_frame;
        let mut events = Vec::new();

        for frame_idx in config.window_size..num_frames {
            let mut positions = Vec::new();

            for i in (frame_idx - config.window_size)..frame_idx {
                let offset = i * coords_per_frame + config.com_keypoint * 2;
                if offset + 1 < samples.len() {
                    positions.push((samples[offset], samples[offset + 1]));
                }
            }

            let sway = self.calculate_sway(&positions);

            if sway > config.threshold {
                let time = frame_idx as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, sway));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "BodySwayEncoder"
    }
}
