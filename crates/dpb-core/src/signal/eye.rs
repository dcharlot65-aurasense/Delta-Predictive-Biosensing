//! Eye Tracking Analysis Module
//!
//! Provides comprehensive eye movement analysis including:
//! - Saccade detection and metrics
//! - Fixation identification and analysis
//! - Smooth pursuit analysis
//! - Pupillometry
//! - Blink detection
//! - Gaze pattern analysis

use crate::error::{DpbError, Result};
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

/// Saccade event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Saccade {
    /// Start time (samples)
    pub start: usize,
    /// End time (samples)
    pub end: usize,
    /// Start position (x, y) in degrees
    pub start_pos: (f64, f64),
    /// End position (x, y) in degrees
    pub end_pos: (f64, f64),
    /// Amplitude (degrees)
    pub amplitude: f64,
    /// Duration (milliseconds)
    pub duration: f64,
    /// Peak velocity (degrees/second)
    pub peak_velocity: f64,
    /// Average velocity (degrees/second)
    pub avg_velocity: f64,
    /// Direction (degrees, 0=right, 90=up)
    pub direction: f64,
    /// Latency from stimulus (if applicable)
    pub latency: Option<f64>,
}

/// Fixation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fixation {
    /// Start time (samples)
    pub start: usize,
    /// End time (samples)
    pub end: usize,
    /// Center position (x, y) in degrees
    pub center: (f64, f64),
    /// Duration (milliseconds)
    pub duration: f64,
    /// Dispersion (degrees)
    pub dispersion: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Standard deviation of position
    pub position_std: f64,
}

/// Blink event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blink {
    /// Start time (samples)
    pub start: usize,
    /// End time (samples)
    pub end: usize,
    /// Duration (milliseconds)
    pub duration: f64,
}

/// Smooth pursuit metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothPursuitMetrics {
    /// Gain (eye velocity / target velocity)
    pub gain: f64,
    /// Lag (degrees)
    pub lag: f64,
    /// Position error RMS (degrees)
    pub position_error_rms: f64,
    /// Velocity error RMS (degrees/s)
    pub velocity_error_rms: f64,
    /// Number of catch-up saccades
    pub catchup_saccade_count: usize,
    /// Proportion of smooth pursuit (vs saccades)
    pub pursuit_proportion: f64,
}

/// Pupil metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PupilMetrics {
    /// Mean pupil diameter (mm)
    pub mean_diameter: f64,
    /// Standard deviation of diameter
    pub std_diameter: f64,
    /// Minimum diameter
    pub min_diameter: f64,
    /// Maximum diameter
    pub max_diameter: f64,
    /// Baseline pupil diameter
    pub baseline_diameter: f64,
    /// Peak dilation (relative to baseline)
    pub peak_dilation: f64,
    /// Peak dilation latency (ms)
    pub peak_latency: f64,
    /// Constriction velocity (mm/s)
    pub constriction_velocity: f64,
    /// Dilation velocity (mm/s)
    pub dilation_velocity: f64,
}

/// Gaze pattern summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GazePatternSummary {
    /// Total number of saccades
    pub saccade_count: usize,
    /// Total number of fixations
    pub fixation_count: usize,
    /// Mean saccade amplitude (degrees)
    pub mean_saccade_amplitude: f64,
    /// Mean saccade duration (ms)
    pub mean_saccade_duration: f64,
    /// Mean saccade peak velocity (deg/s)
    pub mean_peak_velocity: f64,
    /// Mean fixation duration (ms)
    pub mean_fixation_duration: f64,
    /// Total fixation time (ms)
    pub total_fixation_time: f64,
    /// Fixation rate (fixations per second)
    pub fixation_rate: f64,
    /// Scanpath length (degrees)
    pub scanpath_length: f64,
    /// Number of blinks
    pub blink_count: usize,
    /// Blink rate (per minute)
    pub blink_rate: f64,
}

/// Eye tracking analyzer
pub struct EyeAnalyzer {
    sample_rate: f64,
    /// Velocity threshold for saccade detection (deg/s)
    velocity_threshold: f64,
    /// Minimum saccade duration (ms)
    min_saccade_duration: f64,
    /// Minimum fixation duration (ms)
    min_fixation_duration: f64,
    /// Maximum fixation dispersion (degrees)
    max_fixation_dispersion: f64,
}

impl EyeAnalyzer {
    /// Create a new eye analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            velocity_threshold: 30.0,      // deg/s
            min_saccade_duration: 10.0,    // ms
            min_fixation_duration: 100.0,  // ms
            max_fixation_dispersion: 1.0,  // degrees
        }
    }

    /// Configure saccade detection parameters
    pub fn configure_saccade_detection(
        &mut self,
        velocity_threshold: f64,
        min_duration: f64,
    ) {
        self.velocity_threshold = velocity_threshold;
        self.min_saccade_duration = min_duration;
    }

    /// Configure fixation detection parameters
    pub fn configure_fixation_detection(
        &mut self,
        min_duration: f64,
        max_dispersion: f64,
    ) {
        self.min_fixation_duration = min_duration;
        self.max_fixation_dispersion = max_dispersion;
    }

    /// Calculate velocity from position data
    pub fn calculate_velocity(
        &self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Array1<f64>> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(DpbError::InvalidDimensions(
                "Position arrays must have same length >= 2".to_string(),
            ));
        }

        let dt = 1000.0 / self.sample_rate; // ms per sample
        let mut velocity = Array1::zeros(x.len());

        for i in 1..x.len() - 1 {
            let dx = x[i + 1] - x[i - 1];
            let dy = y[i + 1] - y[i - 1];
            velocity[i] = (dx * dx + dy * dy).sqrt() / (2.0 * dt) * 1000.0; // deg/s
        }

        // Handle edges
        velocity[0] = velocity[1];
        velocity[x.len() - 1] = velocity[x.len() - 2];

        Ok(velocity)
    }

    /// Detect saccades using velocity threshold
    pub fn detect_saccades(
        &self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Vec<Saccade>> {
        let velocity = self.calculate_velocity(x, y)?;
        let min_samples = (self.min_saccade_duration * self.sample_rate / 1000.0) as usize;
        let min_samples = min_samples.max(2);

        let mut saccades = Vec::new();
        let mut in_saccade = false;
        let mut saccade_start = 0;

        for i in 0..velocity.len() {
            if velocity[i] > self.velocity_threshold && !in_saccade {
                in_saccade = true;
                saccade_start = i;
            } else if velocity[i] <= self.velocity_threshold && in_saccade {
                in_saccade = false;

                if i - saccade_start >= min_samples {
                    let start_pos = (x[saccade_start], y[saccade_start]);
                    let end_pos = (x[i], y[i]);

                    let dx = end_pos.0 - start_pos.0;
                    let dy = end_pos.1 - start_pos.1;
                    let amplitude = (dx * dx + dy * dy).sqrt();

                    let peak_velocity = velocity
                        .slice(ndarray::s![saccade_start..i])
                        .iter()
                        .cloned()
                        .fold(0.0f64, f64::max);

                    let avg_velocity = velocity
                        .slice(ndarray::s![saccade_start..i])
                        .mean()
                        .unwrap_or(0.0);

                    let direction = dy.atan2(dx).to_degrees();
                    let duration = (i - saccade_start) as f64 * 1000.0 / self.sample_rate;

                    saccades.push(Saccade {
                        start: saccade_start,
                        end: i,
                        start_pos,
                        end_pos,
                        amplitude,
                        duration,
                        peak_velocity,
                        avg_velocity,
                        direction,
                        latency: None,
                    });
                }
            }
        }

        Ok(saccades)
    }

    /// Detect fixations using dispersion-based method
    pub fn detect_fixations(
        &self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
    ) -> Result<Vec<Fixation>> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(DpbError::InvalidDimensions(
                "Position arrays must have same length >= 2".to_string(),
            ));
        }

        let min_samples = (self.min_fixation_duration * self.sample_rate / 1000.0) as usize;
        let min_samples = min_samples.max(2);

        let mut fixations = Vec::new();
        let mut window_start = 0;

        while window_start < x.len() - min_samples {
            let mut window_end = window_start + min_samples;

            // Calculate dispersion of initial window
            let window_x = x.slice(ndarray::s![window_start..window_end]);
            let window_y = y.slice(ndarray::s![window_start..window_end]);

            let mut dispersion = self.calculate_dispersion(&window_x, &window_y);

            // Expand window while dispersion is below threshold
            while dispersion < self.max_fixation_dispersion && window_end < x.len() {
                window_end += 1;
                let window_x = x.slice(ndarray::s![window_start..window_end]);
                let window_y = y.slice(ndarray::s![window_start..window_end]);
                dispersion = self.calculate_dispersion(&window_x, &window_y);
            }

            if window_end - window_start >= min_samples {
                // Valid fixation found
                let window_x = x.slice(ndarray::s![window_start..window_end - 1]);
                let window_y = y.slice(ndarray::s![window_start..window_end - 1]);

                let center_x = window_x.mean().unwrap_or(0.0);
                let center_y = window_y.mean().unwrap_or(0.0);

                let position_std = {
                    let var_x = window_x.iter().map(|&v| (v - center_x).powi(2)).sum::<f64>()
                        / window_x.len() as f64;
                    let var_y = window_y.iter().map(|&v| (v - center_y).powi(2)).sum::<f64>()
                        / window_y.len() as f64;
                    ((var_x + var_y) / 2.0).sqrt()
                };

                let duration = (window_end - 1 - window_start) as f64 * 1000.0 / self.sample_rate;

                fixations.push(Fixation {
                    start: window_start,
                    end: window_end - 1,
                    center: (center_x, center_y),
                    duration,
                    dispersion: self.calculate_dispersion(&window_x, &window_y),
                    sample_count: window_end - 1 - window_start,
                    position_std,
                });

                window_start = window_end - 1;
            } else {
                window_start += 1;
            }
        }

        Ok(fixations)
    }

    /// Calculate dispersion (max - min) for x and y
    fn calculate_dispersion(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        let x_range = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - x.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_range = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - y.iter().cloned().fold(f64::INFINITY, f64::min);
        x_range + y_range
    }

    /// Detect blinks from pupil diameter data
    pub fn detect_blinks(&self, pupil_diameter: ArrayView1<f64>) -> Vec<Blink> {
        let mut blinks = Vec::new();

        // Blinks are characterized by sudden drops in pupil diameter (or NaN/0 values)
        let mean_diameter = pupil_diameter
            .iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .sum::<f64>()
            / pupil_diameter
                .iter()
                .filter(|&&v| v > 0.0 && v.is_finite())
                .count() as f64;

        let threshold = mean_diameter * 0.5;

        let mut in_blink = false;
        let mut blink_start = 0;

        for i in 0..pupil_diameter.len() {
            let is_blink_sample = pupil_diameter[i] < threshold
                || pupil_diameter[i].is_nan()
                || pupil_diameter[i] <= 0.0;

            if is_blink_sample && !in_blink {
                in_blink = true;
                blink_start = i;
            } else if !is_blink_sample && in_blink {
                in_blink = false;
                let duration = (i - blink_start) as f64 * 1000.0 / self.sample_rate;

                // Filter out very short events (likely noise) and very long events
                if duration >= 50.0 && duration <= 500.0 {
                    blinks.push(Blink {
                        start: blink_start,
                        end: i,
                        duration,
                    });
                }
            }
        }

        blinks
    }

    /// Analyze pupil diameter
    pub fn analyze_pupil(
        &self,
        pupil_diameter: ArrayView1<f64>,
        baseline_samples: Option<usize>,
    ) -> Result<PupilMetrics> {
        // Filter out blinks and invalid samples
        let valid_samples: Vec<f64> = pupil_diameter
            .iter()
            .filter(|&&v| v > 0.0 && v.is_finite())
            .cloned()
            .collect();

        if valid_samples.is_empty() {
            return Err(DpbError::DataValidation(
                "No valid pupil samples".to_string(),
            ));
        }

        let mean_diameter = valid_samples.iter().sum::<f64>() / valid_samples.len() as f64;
        let variance = valid_samples
            .iter()
            .map(|&v| (v - mean_diameter).powi(2))
            .sum::<f64>()
            / valid_samples.len() as f64;
        let std_diameter = variance.sqrt();

        let min_diameter = valid_samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_diameter = valid_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Baseline calculation
        let baseline_diameter = if let Some(n) = baseline_samples {
            let baseline: Vec<f64> = pupil_diameter
                .iter()
                .take(n)
                .filter(|&&v| v > 0.0 && v.is_finite())
                .cloned()
                .collect();
            if baseline.is_empty() {
                mean_diameter
            } else {
                baseline.iter().sum::<f64>() / baseline.len() as f64
            }
        } else {
            mean_diameter
        };

        let peak_dilation = max_diameter - baseline_diameter;

        // Find peak dilation latency
        let peak_idx = pupil_diameter
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_latency = peak_idx as f64 * 1000.0 / self.sample_rate;

        // Estimate constriction and dilation velocities
        let (constriction_velocity, dilation_velocity) =
            self.estimate_pupil_velocities(pupil_diameter);

        Ok(PupilMetrics {
            mean_diameter,
            std_diameter,
            min_diameter,
            max_diameter,
            baseline_diameter,
            peak_dilation,
            peak_latency,
            constriction_velocity,
            dilation_velocity,
        })
    }

    /// Estimate pupil constriction and dilation velocities
    fn estimate_pupil_velocities(&self, pupil: ArrayView1<f64>) -> (f64, f64) {
        let dt = 1000.0 / self.sample_rate;
        let mut constriction_velocities = Vec::new();
        let mut dilation_velocities = Vec::new();

        for i in 1..pupil.len() {
            if pupil[i].is_finite() && pupil[i - 1].is_finite() && pupil[i] > 0.0 && pupil[i - 1] > 0.0 {
                let velocity = (pupil[i] - pupil[i - 1]) / dt * 1000.0;
                if velocity < 0.0 {
                    constriction_velocities.push(-velocity);
                } else if velocity > 0.0 {
                    dilation_velocities.push(velocity);
                }
            }
        }

        let mean_constriction = if constriction_velocities.is_empty() {
            0.0
        } else {
            constriction_velocities.iter().sum::<f64>() / constriction_velocities.len() as f64
        };

        let mean_dilation = if dilation_velocities.is_empty() {
            0.0
        } else {
            dilation_velocities.iter().sum::<f64>() / dilation_velocities.len() as f64
        };

        (mean_constriction, mean_dilation)
    }

    /// Analyze smooth pursuit performance
    pub fn analyze_smooth_pursuit(
        &self,
        eye_x: ArrayView1<f64>,
        eye_y: ArrayView1<f64>,
        target_x: ArrayView1<f64>,
        target_y: ArrayView1<f64>,
    ) -> Result<SmoothPursuitMetrics> {
        if eye_x.len() != target_x.len() || eye_y.len() != target_y.len() {
            return Err(DpbError::InvalidDimensions(
                "Eye and target arrays must have same length".to_string(),
            ));
        }

        // Position error
        let position_errors: Vec<f64> = (0..eye_x.len())
            .map(|i| {
                let dx = eye_x[i] - target_x[i];
                let dy = eye_y[i] - target_y[i];
                (dx * dx + dy * dy).sqrt()
            })
            .collect();

        let position_error_rms =
            (position_errors.iter().map(|e| e * e).sum::<f64>() / position_errors.len() as f64).sqrt();

        // Velocity calculation
        let eye_velocity = self.calculate_velocity(eye_x, eye_y)?;
        let target_velocity = self.calculate_velocity(target_x, target_y)?;

        // Velocity error
        let velocity_errors: Vec<f64> = (0..eye_velocity.len())
            .map(|i| (eye_velocity[i] - target_velocity[i]).abs())
            .collect();
        let velocity_error_rms =
            (velocity_errors.iter().map(|e| e * e).sum::<f64>() / velocity_errors.len() as f64).sqrt();

        // Gain (eye velocity / target velocity)
        let mean_eye_vel = eye_velocity.mean().unwrap_or(0.0);
        let mean_target_vel = target_velocity.mean().unwrap_or(1.0);
        let gain = if mean_target_vel > 0.0 {
            mean_eye_vel / mean_target_vel
        } else {
            0.0
        };

        // Detect catch-up saccades during pursuit
        let saccades = self.detect_saccades(eye_x, eye_y)?;
        let catchup_saccade_count = saccades.len();

        // Calculate pursuit proportion (time not in saccades)
        let total_saccade_samples: usize = saccades.iter().map(|s| s.end - s.start).sum();
        let pursuit_proportion =
            1.0 - (total_saccade_samples as f64 / eye_x.len() as f64);

        // Lag estimation (simplified - cross-correlation would be better)
        let lag = position_errors.iter().sum::<f64>() / position_errors.len() as f64 * 0.1;

        Ok(SmoothPursuitMetrics {
            gain,
            lag,
            position_error_rms,
            velocity_error_rms,
            catchup_saccade_count,
            pursuit_proportion,
        })
    }

    /// Generate gaze pattern summary
    pub fn summarize_gaze_pattern(
        &self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
        pupil_diameter: Option<ArrayView1<f64>>,
    ) -> Result<GazePatternSummary> {
        let duration_seconds = x.len() as f64 / self.sample_rate;

        let saccades = self.detect_saccades(x, y)?;
        let fixations = self.detect_fixations(x, y)?;

        let blinks = if let Some(pupil) = pupil_diameter {
            self.detect_blinks(pupil)
        } else {
            Vec::new()
        };

        let saccade_count = saccades.len();
        let fixation_count = fixations.len();

        let mean_saccade_amplitude = if saccade_count > 0 {
            saccades.iter().map(|s| s.amplitude).sum::<f64>() / saccade_count as f64
        } else {
            0.0
        };

        let mean_saccade_duration = if saccade_count > 0 {
            saccades.iter().map(|s| s.duration).sum::<f64>() / saccade_count as f64
        } else {
            0.0
        };

        let mean_peak_velocity = if saccade_count > 0 {
            saccades.iter().map(|s| s.peak_velocity).sum::<f64>() / saccade_count as f64
        } else {
            0.0
        };

        let mean_fixation_duration = if fixation_count > 0 {
            fixations.iter().map(|f| f.duration).sum::<f64>() / fixation_count as f64
        } else {
            0.0
        };

        let total_fixation_time: f64 = fixations.iter().map(|f| f.duration).sum();
        let fixation_rate = fixation_count as f64 / duration_seconds;

        // Scanpath length (sum of saccade amplitudes)
        let scanpath_length: f64 = saccades.iter().map(|s| s.amplitude).sum();

        let blink_count = blinks.len();
        let blink_rate = blink_count as f64 / (duration_seconds / 60.0);

        Ok(GazePatternSummary {
            saccade_count,
            fixation_count,
            mean_saccade_amplitude,
            mean_saccade_duration,
            mean_peak_velocity,
            mean_fixation_duration,
            total_fixation_time,
            fixation_rate,
            scanpath_length,
            blink_count,
            blink_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_saccadic_movement(sample_rate: f64, duration: f64) -> (Array1<f64>, Array1<f64>) {
        let n_samples = (sample_rate * duration) as usize;
        let mut x = Array1::zeros(n_samples);
        let mut y = Array1::zeros(n_samples);

        // Generate step-like pattern with smooth transitions
        let saccade_times = vec![0.5, 1.5, 2.5];
        let positions = vec![(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)];

        let mut current_pos = 0;
        for i in 0..n_samples {
            let t = i as f64 / sample_rate;

            // Check if saccade should occur
            for (j, &saccade_t) in saccade_times.iter().enumerate() {
                if t >= saccade_t && t < saccade_t + 0.05 && current_pos == j {
                    current_pos = j + 1;
                }
            }

            x[i] = positions[current_pos.min(positions.len() - 1)].0;
            y[i] = positions[current_pos.min(positions.len() - 1)].1;
        }

        (x, y)
    }

    #[test]
    fn test_velocity_calculation() {
        let sample_rate = 1000.0;
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        let analyzer = EyeAnalyzer::new(sample_rate);
        let velocity = analyzer.calculate_velocity(x.view(), y.view()).unwrap();

        assert_eq!(velocity.len(), 5);
        // Velocity should be approximately 1000 deg/s (1 deg per ms)
        for i in 1..4 {
            assert!(velocity[i] > 500.0);
        }
    }

    #[test]
    fn test_saccade_detection() {
        let sample_rate = 1000.0;
        let (x, y) = generate_saccadic_movement(sample_rate, 3.0);

        let analyzer = EyeAnalyzer::new(sample_rate);
        let saccades = analyzer.detect_saccades(x.view(), y.view()).unwrap();

        // Should detect some saccades
        assert!(!saccades.is_empty());

        for saccade in &saccades {
            assert!(saccade.amplitude > 0.0);
            assert!(saccade.peak_velocity > 0.0);
        }
    }

    #[test]
    fn test_fixation_detection() {
        let sample_rate = 1000.0;
        // Stable gaze with small jitter
        let n_samples = 1000;
        let x: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| 5.0 + 0.05 * (i as f64 * 0.1).sin()));
        let y: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| 5.0 + 0.05 * (i as f64 * 0.15).cos()));

        let analyzer = EyeAnalyzer::new(sample_rate);
        let fixations = analyzer.detect_fixations(x.view(), y.view()).unwrap();

        // Should detect at least one fixation
        assert!(!fixations.is_empty());

        for fixation in &fixations {
            assert!(fixation.duration >= 100.0);
            assert!(fixation.dispersion < 1.0);
        }
    }

    #[test]
    fn test_blink_detection() {
        let sample_rate = 60.0;
        let mut pupil = Array1::from_elem(600, 4.0); // 10 seconds

        // Simulate blinks
        for i in 120..140 {
            pupil[i] = 0.0;
        }
        for i in 360..385 {
            pupil[i] = 0.0;
        }

        let analyzer = EyeAnalyzer::new(sample_rate);
        let blinks = analyzer.detect_blinks(pupil.view());

        assert_eq!(blinks.len(), 2);
    }

    #[test]
    fn test_pupil_analysis() {
        let sample_rate = 60.0;
        let mut pupil = Array1::from_elem(300, 3.5);

        // Simulate dilation response
        for i in 60..180 {
            pupil[i] = 3.5 + 1.0 * ((i - 60) as f64 / 120.0);
        }

        let analyzer = EyeAnalyzer::new(sample_rate);
        let metrics = analyzer.analyze_pupil(pupil.view(), Some(60)).unwrap();

        assert!(metrics.mean_diameter > 0.0);
        assert!(metrics.peak_dilation > 0.0);
    }

    #[test]
    fn test_gaze_summary() {
        let sample_rate = 1000.0;
        let (x, y) = generate_saccadic_movement(sample_rate, 5.0);

        let analyzer = EyeAnalyzer::new(sample_rate);
        let summary = analyzer.summarize_gaze_pattern(x.view(), y.view(), None).unwrap();

        assert!(summary.saccade_count > 0 || summary.fixation_count > 0);
        assert!(summary.fixation_rate >= 0.0);
        assert!(summary.scanpath_length >= 0.0);
    }
}
