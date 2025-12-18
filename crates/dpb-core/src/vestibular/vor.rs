//! Vestibulo-Ocular Reflex (VOR) analysis
//!
//! This module provides tools for analyzing the vestibulo-ocular reflex, which
//! stabilizes vision during head movements. Key assessments include:
//!
//! - **VOR Gain**: Ratio of eye velocity to head velocity (normal ~0.8-1.0)
//! - **VOR Asymmetry**: Difference between left and right responses
//! - **Corrective Saccades**: Eye movements that compensate for inadequate VOR
//! - **Video Head Impulse Test (vHIT)**: Rapid head impulse assessment

/// Vestibulo-Ocular Reflex analyzer
///
/// # Example
///
/// ```rust
/// use dpb_core::vestibular::{VorAnalyzer, VhitTrial, HeadDirection};
///
/// let analyzer = VorAnalyzer::new(200.0); // 200 Hz eye tracking
/// let trial = VhitTrial {
///     head_velocity: vec![0.0, 50.0, 100.0, 50.0, 0.0],
///     eye_velocity: vec![0.0, -45.0, -90.0, -45.0, 0.0],
///     direction: HeadDirection::Right,
/// };
/// let metrics = analyzer.analyze_vhit(&trial);
/// println!("VOR gain: {}", metrics.gain);
/// ```
#[derive(Debug, Clone)]
pub struct VorAnalyzer {
    /// Sampling rate in Hz (typically 200-250 Hz for video eye tracking)
    pub sample_rate: f64,
}

/// Comprehensive VOR metrics from head impulse testing
#[derive(Debug, Clone, Default)]
pub struct VorMetrics {
    /// VOR gain (eye velocity / head velocity), normal ~0.8-1.0
    pub gain: f64,
    /// Gain asymmetry between left and right (0.0 = symmetric)
    pub gain_asymmetry: f64,
    /// Phase lag in degrees (normal < 10°)
    pub phase_lag: f64,
    /// Number of covert saccades (during head movement)
    pub covert_saccade_count: usize,
    /// Number of overt saccades (after head movement)
    pub overt_saccade_count: usize,
}

/// Video Head Impulse Test (vHIT) trial data
#[derive(Debug, Clone)]
pub struct VhitTrial {
    /// Head angular velocity (deg/s)
    pub head_velocity: Vec<f64>,
    /// Eye angular velocity (deg/s, opposite sign to head)
    pub eye_velocity: Vec<f64>,
    /// Direction of head movement
    pub direction: HeadDirection,
}

/// Direction of head movement in horizontal plane
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadDirection {
    /// Leftward head rotation
    Left,
    /// Rightward head rotation
    Right,
}

/// Corrective saccade detected during or after head impulse
#[derive(Debug, Clone)]
pub struct CorrectionSaccade {
    /// Latency from head impulse start (ms)
    pub latency_ms: f64,
    /// Saccade amplitude (degrees)
    pub amplitude: f64,
    /// True if saccade occurs during head movement (covert), false if after (overt)
    pub is_covert: bool,
}

impl VorAnalyzer {
    /// Create a new VOR analyzer with the specified sampling rate
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sampling rate in Hz (typically 200-250 Hz for vHIT)
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }

    /// Calculate VOR gain from eye and head velocity data
    ///
    /// Gain is calculated as the ratio of peak eye velocity to peak head velocity.
    /// Normal gain is approximately 0.8-1.0. Values < 0.6 may indicate vestibular hypofunction.
    ///
    /// # Arguments
    ///
    /// * `eye_velocity` - Eye angular velocity (deg/s)
    /// * `head_velocity` - Head angular velocity (deg/s)
    ///
    /// # Returns
    ///
    /// VOR gain (dimensionless ratio)
    pub fn vor_gain(&self, eye_velocity: &[f64], head_velocity: &[f64]) -> f64 {
        if eye_velocity.is_empty() || head_velocity.is_empty() {
            return 0.0;
        }

        // Find peak velocities (absolute values)
        let peak_eye = eye_velocity
            .iter()
            .map(|&v| v.abs())
            .fold(0.0_f64, |a, b| a.max(b));
        let peak_head = head_velocity
            .iter()
            .map(|&v| v.abs())
            .fold(0.0_f64, |a, b| a.max(b));

        if peak_head > 0.0 {
            peak_eye / peak_head
        } else {
            0.0
        }
    }

    /// Calculate asymmetry between left and right VOR gains
    ///
    /// Asymmetry indicates unilateral vestibular dysfunction.
    /// Values > 0.1-0.15 (10-15%) may be clinically significant.
    ///
    /// # Arguments
    ///
    /// * `left_gain` - VOR gain for leftward head impulses
    /// * `right_gain` - VOR gain for rightward head impulses
    ///
    /// # Returns
    ///
    /// Asymmetry ratio: |left - right| / (left + right)
    pub fn vor_asymmetry(&self, left_gain: f64, right_gain: f64) -> f64 {
        let sum = left_gain + right_gain;
        if sum > 0.0 {
            (left_gain - right_gain).abs() / sum
        } else {
            0.0
        }
    }

    /// Analyze a video head impulse test trial
    ///
    /// Computes VOR gain and detects corrective saccades.
    ///
    /// # Arguments
    ///
    /// * `trial` - vHIT trial data including head and eye velocities
    ///
    /// # Returns
    ///
    /// VOR metrics including gain and saccade counts
    pub fn analyze_vhit(&self, trial: &VhitTrial) -> VorMetrics {
        let gain = self.vor_gain(&trial.eye_velocity, &trial.head_velocity);

        // Find when head impulse ends (velocity drops below threshold)
        let head_impulse_end = trial
            .head_velocity
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, v)| v.abs() > 50.0) // 50 deg/s threshold
            .map(|(i, _)| i)
            .unwrap_or(trial.head_velocity.len());

        // Integrate velocities to get positions for saccade detection
        let eye_position = integrate(&trial.eye_velocity, self.sample_rate);

        let saccades = self.detect_corrective_saccades(&eye_position, head_impulse_end);

        let covert_count = saccades.iter().filter(|s| s.is_covert).count();
        let overt_count = saccades.iter().filter(|s| !s.is_covert).count();

        VorMetrics {
            gain,
            gain_asymmetry: 0.0, // Requires left and right trials
            phase_lag: 0.0,      // Requires cross-correlation analysis
            covert_saccade_count: covert_count,
            overt_saccade_count: overt_count,
        }
    }

    /// Detect corrective saccades in eye position data
    ///
    /// Saccades are rapid eye movements that compensate for inadequate VOR.
    /// Covert saccades occur during head movement; overt saccades occur after.
    ///
    /// # Arguments
    ///
    /// * `eye_position` - Eye position over time (degrees)
    /// * `head_impulse_end` - Index where head impulse ends
    ///
    /// # Returns
    ///
    /// Vector of detected corrective saccades
    pub fn detect_corrective_saccades(
        &self,
        eye_position: &[f64],
        head_impulse_end: usize,
    ) -> Vec<CorrectionSaccade> {
        if eye_position.len() < 3 {
            return Vec::new();
        }

        let mut saccades = Vec::new();

        // Calculate velocity from position
        let velocity: Vec<f64> = eye_position
            .windows(2)
            .map(|w| (w[1] - w[0]) * self.sample_rate)
            .collect();

        // Detect saccades using velocity threshold
        let saccade_threshold = 100.0; // deg/s
        let mut in_saccade = false;
        let mut saccade_start = 0;

        for (i, &v) in velocity.iter().enumerate() {
            if !in_saccade && v.abs() > saccade_threshold {
                // Saccade onset
                in_saccade = true;
                saccade_start = i;
            } else if in_saccade && v.abs() < saccade_threshold / 2.0 {
                // Saccade offset
                in_saccade = false;

                if i > saccade_start {
                    let amplitude = (eye_position[i] - eye_position[saccade_start]).abs();
                    let latency_ms = saccade_start as f64 * 1000.0 / self.sample_rate;
                    let is_covert = saccade_start < head_impulse_end;

                    // Only record significant saccades (> 2 degrees)
                    if amplitude > 2.0 {
                        saccades.push(CorrectionSaccade {
                            latency_ms,
                            amplitude,
                            is_covert,
                        });
                    }
                }
            }
        }

        saccades
    }
}

/// Integrate velocity to position using trapezoidal rule
fn integrate(velocity: &[f64], sample_rate: f64) -> Vec<f64> {
    let dt = 1.0 / sample_rate;
    let mut position = Vec::with_capacity(velocity.len());
    let mut pos = 0.0;

    position.push(pos);

    for &v in velocity.iter() {
        pos += v * dt;
        position.push(pos);
    }

    position
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vor_analyzer_creation() {
        let analyzer = VorAnalyzer::new(200.0);
        assert_eq!(analyzer.sample_rate, 200.0);
    }

    #[test]
    fn test_vor_gain() {
        let analyzer = VorAnalyzer::new(200.0);
        let eye_velocity = vec![0.0, -80.0, -100.0, -80.0, 0.0];
        let head_velocity = vec![0.0, 100.0, 125.0, 100.0, 0.0];

        let gain = analyzer.vor_gain(&eye_velocity, &head_velocity);
        // Peak eye = 100, peak head = 125, gain = 100/125 = 0.8
        assert!((gain - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_vor_asymmetry() {
        let analyzer = VorAnalyzer::new(200.0);

        // Symmetric gains
        let asym = analyzer.vor_asymmetry(0.9, 0.9);
        assert_eq!(asym, 0.0);

        // Asymmetric gains
        let asym = analyzer.vor_asymmetry(1.0, 0.8);
        assert!((asym - 0.111).abs() < 0.001);
    }

    #[test]
    fn test_integration() {
        let velocity = vec![10.0, 10.0, 10.0];
        let position = integrate(&velocity, 10.0); // dt = 0.1

        // pos[0] = 0, pos[1] = 1.0, pos[2] = 2.0, pos[3] = 3.0
        assert_eq!(position.len(), 4);
        assert!((position[1] - 1.0).abs() < 0.01);
        assert!((position[2] - 2.0).abs() < 0.01);
        assert!((position[3] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_vhit_trial() {
        let analyzer = VorAnalyzer::new(200.0);
        let trial = VhitTrial {
            head_velocity: vec![0.0, 100.0, 150.0, 100.0, 0.0],
            eye_velocity: vec![0.0, -90.0, -135.0, -90.0, 0.0],
            direction: HeadDirection::Right,
        };

        let metrics = analyzer.analyze_vhit(&trial);
        assert!((metrics.gain - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_head_direction() {
        let left = HeadDirection::Left;
        let right = HeadDirection::Right;
        assert_ne!(left, right);
        assert_eq!(left, HeadDirection::Left);
    }
}
