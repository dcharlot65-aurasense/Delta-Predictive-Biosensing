//! ADHD-specific eye tracking patterns
//!
//! Implements eye tracking analysis specific to ADHD assessment:
//! - Fixation stability during sustained attention
//! - Saccade metrics during reading
//! - Anticipatory saccade detection

/// ADHD-specific eye tracking analyzer
#[derive(Debug, Clone)]
pub struct AdhdEyeTracking {
    /// Sample rate of eye tracker (Hz)
    pub sample_rate: f64,
    /// Minimum fixation duration (ms)
    pub min_fixation_duration_ms: f64,
    /// Maximum dispersion for fixation (degrees)
    pub max_dispersion_deg: f64,
}

impl Default for AdhdEyeTracking {
    fn default() -> Self {
        Self {
            sample_rate: 120.0,
            min_fixation_duration_ms: 100.0,
            max_dispersion_deg: 1.0,
        }
    }
}

impl AdhdEyeTracking {
    /// Create new ADHD eye tracking analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            ..Default::default()
        }
    }

    /// Analyze fixation stability during sustained attention task
    pub fn fixation_stability(&self, gaze: &[GazePoint]) -> FixationStability {
        if gaze.is_empty() {
            return FixationStability::default();
        }

        // Detect fixations using I-DT algorithm
        let fixations = self.detect_fixations(gaze);

        if fixations.is_empty() {
            return FixationStability::default();
        }

        // Calculate fixation metrics
        let durations: Vec<f64> = fixations.iter().map(|f| f.duration_ms).collect();
        let mean_duration = durations.iter().sum::<f64>() / durations.len() as f64;

        let variance: f64 = durations
            .iter()
            .map(|d| (d - mean_duration).powi(2))
            .sum::<f64>()
            / durations.len() as f64;
        let duration_variability = variance.sqrt();

        // Calculate dispersion within fixations
        let dispersions: Vec<f64> = fixations.iter().map(|f| f.dispersion_deg).collect();
        let mean_dispersion = dispersions.iter().sum::<f64>() / dispersions.len() as f64;

        // Calculate blink rate
        let blinks = self.detect_blinks(gaze);
        let total_time_sec = gaze.len() as f64 / self.sample_rate;
        let blink_rate = if total_time_sec > 0.0 {
            blinks.len() as f64 / (total_time_sec / 60.0) // blinks per minute
        } else {
            0.0
        };

        // Calculate microsaccade rate within fixations
        let microsaccades = self.detect_microsaccades(&fixations, gaze);
        let microsaccade_rate = if !fixations.is_empty() {
            microsaccades.len() as f64 / fixations.len() as f64
        } else {
            0.0
        };

        FixationStability {
            mean_fixation_duration_ms: mean_duration,
            fixation_duration_variability_ms: duration_variability,
            mean_dispersion_deg: mean_dispersion,
            blink_rate_per_min: blink_rate,
            microsaccade_rate,
            n_fixations: fixations.len(),
        }
    }

    /// Analyze saccade metrics during reading task
    pub fn reading_saccades(&self, gaze: &[GazePoint]) -> ReadingSaccadeMetrics {
        if gaze.is_empty() {
            return ReadingSaccadeMetrics::default();
        }

        let saccades = self.detect_saccades(gaze);

        if saccades.is_empty() {
            return ReadingSaccadeMetrics::default();
        }

        // Classify saccades by direction
        let progressive: Vec<&Saccade> = saccades.iter().filter(|s| s.amplitude_deg > 0.0).collect();
        let regressive: Vec<&Saccade> = saccades.iter().filter(|s| s.amplitude_deg < 0.0).collect();

        let regression_rate = if !saccades.is_empty() {
            regressive.len() as f64 / saccades.len() as f64
        } else {
            0.0
        };

        // Calculate saccade amplitudes
        let amplitudes: Vec<f64> = progressive.iter().map(|s| s.amplitude_deg.abs()).collect();
        let mean_forward_amplitude = if amplitudes.is_empty() {
            0.0
        } else {
            amplitudes.iter().sum::<f64>() / amplitudes.len() as f64
        };

        // Calculate fixation durations between saccades
        let fixations = self.detect_fixations(gaze);
        let fixation_durations: Vec<f64> = fixations.iter().map(|f| f.duration_ms).collect();
        let mean_fixation_duration = if fixation_durations.is_empty() {
            0.0
        } else {
            fixation_durations.iter().sum::<f64>() / fixation_durations.len() as f64
        };

        // Detect line return saccades (large leftward movements)
        let line_returns = saccades
            .iter()
            .filter(|s| s.amplitude_deg < -10.0)
            .count();

        ReadingSaccadeMetrics {
            n_saccades: saccades.len(),
            n_fixations: fixations.len(),
            regression_rate,
            mean_forward_amplitude_deg: mean_forward_amplitude,
            mean_fixation_duration_ms: mean_fixation_duration,
            line_returns,
        }
    }

    /// Calculate anticipatory saccade rate
    pub fn anticipatory_saccades(
        &self,
        gaze: &[GazePoint],
        target_times_ms: &[f64],
    ) -> AnticipatoryMetrics {
        if gaze.is_empty() || target_times_ms.is_empty() {
            return AnticipatoryMetrics::default();
        }

        let saccades = self.detect_saccades(gaze);

        // Count saccades that occur before target onset (within 200ms window)
        let mut anticipatory_count = 0;
        let anticipatory_window_ms = 200.0;

        for target_time in target_times_ms {
            for saccade in &saccades {
                let time_before_target = target_time - saccade.onset_time_ms;
                if time_before_target > 0.0 && time_before_target < anticipatory_window_ms {
                    anticipatory_count += 1;
                    break;
                }
            }
        }

        let anticipatory_rate = anticipatory_count as f64 / target_times_ms.len() as f64;

        // Calculate express saccade rate (very short latency saccades after target)
        let mut express_count = 0;
        let express_window_ms = 120.0;

        for target_time in target_times_ms {
            for saccade in &saccades {
                let latency = saccade.onset_time_ms - target_time;
                if latency > 80.0 && latency < express_window_ms {
                    express_count += 1;
                    break;
                }
            }
        }

        let express_saccade_rate = express_count as f64 / target_times_ms.len() as f64;

        AnticipatoryMetrics {
            anticipatory_rate,
            express_saccade_rate,
            n_anticipatory: anticipatory_count,
            n_targets: target_times_ms.len(),
        }
    }

    /// Generate comprehensive ADHD gaze metrics
    pub fn analyze(&self, gaze: &[GazePoint], target_times: Option<&[f64]>) -> AdhdGazeMetrics {
        let fixation_stability = self.fixation_stability(gaze);
        let reading_metrics = self.reading_saccades(gaze);
        let anticipatory = target_times
            .map(|t| self.anticipatory_saccades(gaze, t))
            .unwrap_or_default();

        // Calculate attention stability index
        let attention_stability = self.calculate_attention_stability(&fixation_stability);

        AdhdGazeMetrics {
            fixation_stability,
            reading_metrics,
            anticipatory,
            attention_stability_index: attention_stability,
        }
    }

    /// Calculate attention stability index
    fn calculate_attention_stability(&self, stability: &FixationStability) -> f64 {
        // Lower variability and higher duration = better stability
        let duration_factor = stability.mean_fixation_duration_ms / 300.0; // Normalize to ~300ms
        let variability_factor =
            1.0 - (stability.fixation_duration_variability_ms / stability.mean_fixation_duration_ms)
                .min(1.0);

        ((duration_factor + variability_factor) / 2.0 * 100.0).clamp(0.0, 100.0)
    }

    /// Detect fixations using I-DT algorithm
    fn detect_fixations(&self, gaze: &[GazePoint]) -> Vec<Fixation> {
        let mut fixations = Vec::new();
        let min_samples = (self.min_fixation_duration_ms * self.sample_rate / 1000.0) as usize;

        let mut start_idx = 0;
        while start_idx < gaze.len() {
            let mut end_idx = start_idx + min_samples.min(gaze.len() - start_idx);

            // Expand window while dispersion is below threshold
            while end_idx < gaze.len() {
                let dispersion = self.calculate_dispersion(&gaze[start_idx..end_idx]);
                if dispersion > self.max_dispersion_deg {
                    break;
                }
                end_idx += 1;
            }

            let duration_samples = end_idx - start_idx;
            if duration_samples >= min_samples {
                let duration_ms = duration_samples as f64 / self.sample_rate * 1000.0;
                let centroid = self.calculate_centroid(&gaze[start_idx..end_idx]);
                let dispersion = self.calculate_dispersion(&gaze[start_idx..end_idx]);

                fixations.push(Fixation {
                    start_time_ms: gaze[start_idx].time_ms,
                    duration_ms,
                    x: centroid.0,
                    y: centroid.1,
                    dispersion_deg: dispersion,
                });

                start_idx = end_idx;
            } else {
                start_idx += 1;
            }
        }

        fixations
    }

    /// Detect saccades (simplified velocity-based detection)
    fn detect_saccades(&self, gaze: &[GazePoint]) -> Vec<Saccade> {
        let mut saccades = Vec::new();
        let velocity_threshold = 30.0; // degrees/second

        let mut in_saccade = false;
        let mut saccade_start = 0;

        for i in 1..gaze.len() {
            let dt = (gaze[i].time_ms - gaze[i - 1].time_ms) / 1000.0;
            if dt <= 0.0 {
                continue;
            }

            let dx = gaze[i].x - gaze[i - 1].x;
            let dy = gaze[i].y - gaze[i - 1].y;
            let velocity = (dx * dx + dy * dy).sqrt() / dt;

            if velocity > velocity_threshold && !in_saccade {
                in_saccade = true;
                saccade_start = i - 1;
            } else if velocity < velocity_threshold && in_saccade {
                in_saccade = false;

                let amplitude = gaze[i - 1].x - gaze[saccade_start].x;
                let duration_ms = gaze[i - 1].time_ms - gaze[saccade_start].time_ms;

                saccades.push(Saccade {
                    onset_time_ms: gaze[saccade_start].time_ms,
                    duration_ms,
                    amplitude_deg: amplitude,
                });
            }
        }

        saccades
    }

    /// Detect blinks (missing/invalid data)
    fn detect_blinks(&self, gaze: &[GazePoint]) -> Vec<Blink> {
        let mut blinks = Vec::new();
        let mut in_blink = false;
        let mut blink_start_ms = 0.0;

        for point in gaze {
            if !point.valid && !in_blink {
                in_blink = true;
                blink_start_ms = point.time_ms;
            } else if point.valid && in_blink {
                in_blink = false;
                let duration = point.time_ms - blink_start_ms;
                if duration > 50.0 && duration < 500.0 {
                    // Typical blink duration
                    blinks.push(Blink {
                        time_ms: blink_start_ms,
                        duration_ms: duration,
                    });
                }
            }
        }

        blinks
    }

    /// Detect microsaccades within fixations
    fn detect_microsaccades(&self, fixations: &[Fixation], gaze: &[GazePoint]) -> Vec<Microsaccade> {
        let mut microsaccades = Vec::new();
        let velocity_threshold = 10.0; // Lower threshold for microsaccades

        for fixation in fixations {
            let start_time = fixation.start_time_ms;
            let end_time = start_time + fixation.duration_ms;

            // Get gaze points within this fixation
            let fixation_gaze: Vec<&GazePoint> = gaze
                .iter()
                .filter(|g| g.time_ms >= start_time && g.time_ms <= end_time)
                .collect();

            for i in 1..fixation_gaze.len() {
                let dt = (fixation_gaze[i].time_ms - fixation_gaze[i - 1].time_ms) / 1000.0;
                if dt <= 0.0 {
                    continue;
                }

                let dx = fixation_gaze[i].x - fixation_gaze[i - 1].x;
                let dy = fixation_gaze[i].y - fixation_gaze[i - 1].y;
                let velocity = (dx * dx + dy * dy).sqrt() / dt;
                let amplitude = (dx * dx + dy * dy).sqrt();

                // Microsaccade: small amplitude, high velocity
                if velocity > velocity_threshold && amplitude < 1.0 {
                    microsaccades.push(Microsaccade {
                        time_ms: fixation_gaze[i - 1].time_ms,
                        amplitude_deg: amplitude,
                    });
                }
            }
        }

        microsaccades
    }

    /// Calculate dispersion of gaze points
    fn calculate_dispersion(&self, gaze: &[GazePoint]) -> f64 {
        if gaze.is_empty() {
            return 0.0;
        }

        let x_vals: Vec<f64> = gaze.iter().map(|g| g.x).collect();
        let y_vals: Vec<f64> = gaze.iter().map(|g| g.y).collect();

        let x_range = x_vals.iter().cloned().fold(f64::NAN, f64::max)
            - x_vals.iter().cloned().fold(f64::NAN, f64::min);
        let y_range = y_vals.iter().cloned().fold(f64::NAN, f64::max)
            - y_vals.iter().cloned().fold(f64::NAN, f64::min);

        x_range + y_range
    }

    /// Calculate centroid of gaze points
    fn calculate_centroid(&self, gaze: &[GazePoint]) -> (f64, f64) {
        if gaze.is_empty() {
            return (0.0, 0.0);
        }

        let sum_x: f64 = gaze.iter().map(|g| g.x).sum();
        let sum_y: f64 = gaze.iter().map(|g| g.y).sum();
        let n = gaze.len() as f64;

        (sum_x / n, sum_y / n)
    }
}

/// Single gaze point sample
#[derive(Debug, Clone)]
pub struct GazePoint {
    /// Time in milliseconds
    pub time_ms: f64,
    /// X position (degrees or pixels)
    pub x: f64,
    /// Y position (degrees or pixels)
    pub y: f64,
    /// Whether this sample is valid
    pub valid: bool,
}

/// Detected fixation
#[derive(Debug, Clone)]
pub struct Fixation {
    /// Start time in milliseconds
    pub start_time_ms: f64,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// X centroid
    pub x: f64,
    /// Y centroid
    pub y: f64,
    /// Dispersion (degrees)
    pub dispersion_deg: f64,
}

/// Detected saccade
#[derive(Debug, Clone)]
pub struct Saccade {
    /// Onset time in milliseconds
    pub onset_time_ms: f64,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Amplitude in degrees (positive = rightward for reading)
    pub amplitude_deg: f64,
}

/// Detected blink
#[derive(Debug, Clone)]
pub struct Blink {
    /// Time in milliseconds
    pub time_ms: f64,
    /// Duration in milliseconds
    pub duration_ms: f64,
}

/// Detected microsaccade
#[derive(Debug, Clone)]
pub struct Microsaccade {
    /// Time in milliseconds
    pub time_ms: f64,
    /// Amplitude in degrees
    pub amplitude_deg: f64,
}

/// Fixation stability metrics
#[derive(Debug, Clone, Default)]
pub struct FixationStability {
    /// Mean fixation duration (ms)
    pub mean_fixation_duration_ms: f64,
    /// Variability of fixation duration (SD in ms)
    pub fixation_duration_variability_ms: f64,
    /// Mean dispersion within fixations (degrees)
    pub mean_dispersion_deg: f64,
    /// Blink rate (per minute)
    pub blink_rate_per_min: f64,
    /// Microsaccade rate (per fixation)
    pub microsaccade_rate: f64,
    /// Number of fixations detected
    pub n_fixations: usize,
}

/// Reading saccade metrics
#[derive(Debug, Clone, Default)]
pub struct ReadingSaccadeMetrics {
    /// Total number of saccades
    pub n_saccades: usize,
    /// Total number of fixations
    pub n_fixations: usize,
    /// Proportion of regressive saccades
    pub regression_rate: f64,
    /// Mean amplitude of forward saccades (degrees)
    pub mean_forward_amplitude_deg: f64,
    /// Mean fixation duration (ms)
    pub mean_fixation_duration_ms: f64,
    /// Number of line return saccades
    pub line_returns: usize,
}

/// Anticipatory saccade metrics
#[derive(Debug, Clone, Default)]
pub struct AnticipatoryMetrics {
    /// Rate of anticipatory saccades
    pub anticipatory_rate: f64,
    /// Rate of express saccades
    pub express_saccade_rate: f64,
    /// Number of anticipatory saccades
    pub n_anticipatory: usize,
    /// Number of targets
    pub n_targets: usize,
}

/// Comprehensive ADHD gaze metrics
#[derive(Debug, Clone, Default)]
pub struct AdhdGazeMetrics {
    /// Fixation stability metrics
    pub fixation_stability: FixationStability,
    /// Reading saccade metrics
    pub reading_metrics: ReadingSaccadeMetrics,
    /// Anticipatory saccade metrics
    pub anticipatory: AnticipatoryMetrics,
    /// Overall attention stability index (0-100)
    pub attention_stability_index: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_gaze(n_points: usize) -> Vec<GazePoint> {
        (0..n_points)
            .map(|i| GazePoint {
                time_ms: i as f64 * 8.33, // ~120 Hz
                x: 0.5 + (i as f64 * 0.01).sin() * 0.1,
                y: 0.5 + (i as f64 * 0.01).cos() * 0.1,
                valid: true,
            })
            .collect()
    }

    #[test]
    fn test_fixation_detection() {
        let tracker = AdhdEyeTracking::default();
        let gaze = generate_test_gaze(500);

        let fixations = tracker.detect_fixations(&gaze);
        assert!(!fixations.is_empty());
    }

    #[test]
    fn test_fixation_stability() {
        let tracker = AdhdEyeTracking::default();
        let gaze = generate_test_gaze(1000);

        let stability = tracker.fixation_stability(&gaze);
        assert!(stability.mean_fixation_duration_ms > 0.0);
    }

    #[test]
    fn test_reading_saccades() {
        let tracker = AdhdEyeTracking::default();

        // Generate reading-like gaze pattern
        let mut gaze = Vec::new();
        let mut x = 0.0;
        for i in 0..500 {
            // Simulate reading with occasional regressions
            if i % 30 == 0 {
                x -= 1.0; // Regression
            } else if i % 5 == 0 {
                x += 2.0; // Forward saccade
            }
            gaze.push(GazePoint {
                time_ms: i as f64 * 8.33,
                x,
                y: 0.5,
                valid: true,
            });
        }

        let metrics = tracker.reading_saccades(&gaze);
        assert!(metrics.n_saccades > 0);
    }
}
