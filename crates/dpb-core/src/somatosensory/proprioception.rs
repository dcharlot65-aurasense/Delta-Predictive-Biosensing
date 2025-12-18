//! Proprioception and joint position sense assessment

/// Joint type for proprioceptive testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Joint {
    /// Knee joint
    Knee,
    /// Ankle joint
    Ankle,
    /// Shoulder joint
    Shoulder,
    /// Elbow joint
    Elbow,
    /// Wrist joint
    Wrist,
    /// Hip joint
    Hip,
    /// Finger (MCP/PIP/DIP)
    Finger,
}

/// Proprioceptive test type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProprioceptionTestType {
    /// Joint Position Reproduction - actively match target angle
    ActiveReproduction,
    /// Joint Position Reproduction - passively moved to match
    PassiveReproduction,
    /// Threshold to Detection of Passive Motion
    Ttdpm,
}

/// Joint position sense analyzer
#[derive(Debug, Clone)]
pub struct JointPositionSense {
    /// Joint being tested
    pub joint: Joint,
    /// Sample rate of position data (Hz)
    pub sample_rate: f64,
}

impl JointPositionSense {
    /// Create new JPS analyzer
    pub fn new(joint: Joint, sample_rate: f64) -> Self {
        Self { joint, sample_rate }
    }

    /// Calculate absolute repositioning error
    pub fn repositioning_error(&self, target_angle: f64, reproduced_angle: f64) -> f64 {
        (target_angle - reproduced_angle).abs()
    }

    /// Calculate variable error (consistency across trials)
    pub fn variable_error(&self, errors: &[f64]) -> f64 {
        if errors.is_empty() {
            return 0.0;
        }

        let mean: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
        let variance: f64 = errors
            .iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f64>()
            / errors.len() as f64;

        variance.sqrt()
    }

    /// Calculate constant error (directional bias)
    pub fn constant_error(&self, target_angles: &[f64], reproduced_angles: &[f64]) -> f64 {
        if target_angles.len() != reproduced_angles.len() || target_angles.is_empty() {
            return 0.0;
        }

        let errors: Vec<f64> = target_angles
            .iter()
            .zip(reproduced_angles.iter())
            .map(|(t, r)| r - t)
            .collect();

        errors.iter().sum::<f64>() / errors.len() as f64
    }

    /// Threshold to Detection of Passive Motion (TTDPM)
    pub fn detection_threshold(
        &self,
        movement_velocity_deg_s: f64,
        detection_angle: f64,
    ) -> TtdpmResult {
        TtdpmResult {
            threshold_angle: detection_angle,
            movement_velocity: movement_velocity_deg_s,
            detection_time: detection_angle / movement_velocity_deg_s.max(0.001),
        }
    }

    /// Analyze a series of JPS trials
    pub fn analyze_trials(&self, trials: &[JpsTrialResult]) -> JpsMetrics {
        if trials.is_empty() {
            return JpsMetrics::default();
        }

        let errors: Vec<f64> = trials
            .iter()
            .map(|t| self.repositioning_error(t.target_angle, t.reproduced_angle))
            .collect();

        let mean_absolute_error = errors.iter().sum::<f64>() / errors.len() as f64;
        let variable_error = self.variable_error(&errors);

        let target_angles: Vec<f64> = trials.iter().map(|t| t.target_angle).collect();
        let reproduced_angles: Vec<f64> = trials.iter().map(|t| t.reproduced_angle).collect();
        let constant_error = self.constant_error(&target_angles, &reproduced_angles);

        JpsMetrics {
            mean_absolute_error,
            variable_error,
            constant_error,
            n_trials: trials.len(),
        }
    }

    /// Get age-adjusted percentile for JPS error
    pub fn age_percentile(&self, error_degrees: f64, age: u8) -> f64 {
        let (mean, sd) = match age {
            0..=30 => (2.0, 1.0),
            31..=50 => (3.0, 1.5),
            51..=70 => (4.5, 2.0),
            _ => (6.0, 2.5),
        };

        let z = (error_degrees - mean) / sd;
        normal_cdf(z) * 100.0
    }
}

/// Result from a single JPS trial
#[derive(Debug, Clone)]
pub struct JpsTrialResult {
    /// Target angle in degrees
    pub target_angle: f64,
    /// Reproduced angle in degrees
    pub reproduced_angle: f64,
    /// Time to reproduce in seconds
    pub reproduction_time: f64,
    /// Test type used
    pub test_type: ProprioceptionTestType,
}

/// Aggregated JPS metrics
#[derive(Debug, Clone, Default)]
pub struct JpsMetrics {
    /// Mean absolute error in degrees
    pub mean_absolute_error: f64,
    /// Variable error (SD of errors)
    pub variable_error: f64,
    /// Constant error (directional bias)
    pub constant_error: f64,
    /// Number of trials
    pub n_trials: usize,
}

/// TTDPM result
#[derive(Debug, Clone)]
pub struct TtdpmResult {
    /// Angle at which motion was detected (degrees)
    pub threshold_angle: f64,
    /// Movement velocity (degrees/second)
    pub movement_velocity: f64,
    /// Time to detection (seconds)
    pub detection_time: f64,
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repositioning_error() {
        let jps = JointPositionSense::new(Joint::Knee, 100.0);

        let error = jps.repositioning_error(45.0, 48.0);
        assert!((error - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_variable_error() {
        let jps = JointPositionSense::new(Joint::Knee, 100.0);

        let errors = vec![2.0, 3.0, 2.5, 3.5, 2.0];
        let ve = jps.variable_error(&errors);

        assert!(ve > 0.0 && ve < 1.0);
    }

    #[test]
    fn test_constant_error() {
        let jps = JointPositionSense::new(Joint::Knee, 100.0);

        let targets = vec![30.0, 45.0, 60.0];
        let reproduced = vec![32.0, 47.0, 62.0];

        let ce = jps.constant_error(&targets, &reproduced);
        assert!((ce - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_age_percentile() {
        let jps = JointPositionSense::new(Joint::Knee, 100.0);

        // Normal error for young adult should be around 50th percentile
        let percentile = jps.age_percentile(2.0, 25);
        assert!(percentile > 40.0 && percentile < 60.0);
    }
}
