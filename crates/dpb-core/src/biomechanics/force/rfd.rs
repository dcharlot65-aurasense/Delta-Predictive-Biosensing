//! Rate of force development (RFD) analysis algorithms
//!
//! This module provides tools for analyzing the rate at which force is developed,
//! which is critical for explosive strength assessment and neuromuscular function.

/// Rate of force development analyzer
///
/// RFD is a key metric in explosive strength testing and neuromuscular assessment.
/// This analyzer calculates RFD over various time windows and tracks force timing.
///
/// # Example
///
/// ```
/// use dpb_core::biomechanics::force::RfdAnalyzer;
///
/// let analyzer = RfdAnalyzer::new(1000.0); // 1000 Hz
/// let force = vec![0.0, 100.0, 300.0, 500.0, 600.0];
/// let metrics = analyzer.analyze(&force);
/// println!("Peak RFD: {} N/s", metrics.peak_rfd);
/// ```
#[derive(Debug, Clone)]
pub struct RfdAnalyzer {
    /// Sample rate in Hz
    pub sample_rate: f64,
}

/// Rate of force development metrics over standard time windows
///
/// RFD is typically calculated over early time windows (0-50ms, 0-100ms, etc.)
/// as these reflect neural activation patterns.
#[derive(Debug, Clone, Default)]
pub struct RfdMetrics {
    /// Peak RFD (maximum slope) in N/s
    pub peak_rfd: f64,
    /// RFD in 0-50ms window in N/s
    pub rfd_0_50: f64,
    /// RFD in 0-100ms window in N/s
    pub rfd_0_100: f64,
    /// RFD in 0-200ms window in N/s
    pub rfd_0_200: f64,
    /// RFD in 100-200ms window in N/s
    pub rfd_100_200: f64,
}

/// Force timing metrics relative to peak force
#[derive(Debug, Clone, Default)]
pub struct ForceTiming {
    /// Time to reach 25% of peak force in seconds
    pub time_to_25: f64,
    /// Time to reach 50% of peak force in seconds
    pub time_to_50: f64,
    /// Time to reach 75% of peak force in seconds
    pub time_to_75: f64,
    /// Time to reach 90% of peak force in seconds
    pub time_to_90: f64,
}

impl RfdAnalyzer {
    /// Create a new RFD analyzer with specified sample rate
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sampling frequency in Hz
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }

    /// Perform complete RFD analysis on a force-time curve
    ///
    /// # Arguments
    ///
    /// * `force` - Array of force measurements in Newtons
    ///
    /// # Returns
    ///
    /// Comprehensive RFD metrics over standard time windows
    pub fn analyze(&self, force: &[f64]) -> RfdMetrics {
        RfdMetrics {
            peak_rfd: self.peak_rfd(force),
            rfd_0_50: self.rfd_at_interval(force, 0.0, 50.0),
            rfd_0_100: self.rfd_at_interval(force, 0.0, 100.0),
            rfd_0_200: self.rfd_at_interval(force, 0.0, 200.0),
            rfd_100_200: self.rfd_at_interval(force, 100.0, 200.0),
        }
    }

    /// Calculate peak RFD (maximum slope in force-time curve)
    ///
    /// # Arguments
    ///
    /// * `force` - Array of force measurements in Newtons
    ///
    /// # Returns
    ///
    /// Maximum rate of force development in N/s
    pub fn peak_rfd(&self, force: &[f64]) -> f64 {
        if force.len() < 2 {
            return 0.0;
        }

        let dt = 1.0 / self.sample_rate;
        let mut max_rfd = 0.0;

        for window in force.windows(2) {
            let rfd = (window[1] - window[0]) / dt;
            max_rfd = max_rfd.max(rfd);
        }

        max_rfd
    }

    /// Calculate RFD over a specific time interval
    ///
    /// # Arguments
    ///
    /// * `force` - Array of force measurements in Newtons
    /// * `start_ms` - Start of interval in milliseconds
    /// * `end_ms` - End of interval in milliseconds
    ///
    /// # Returns
    ///
    /// Average RFD over the interval in N/s
    pub fn rfd_at_interval(&self, force: &[f64], start_ms: f64, end_ms: f64) -> f64 {
        if force.is_empty() {
            return 0.0;
        }

        // Convert ms to sample indices
        let start_idx = (start_ms / 1000.0 * self.sample_rate) as usize;
        let end_idx = ((end_ms / 1000.0 * self.sample_rate) as usize).min(force.len() - 1);

        if end_idx <= start_idx || end_idx >= force.len() {
            return 0.0;
        }

        let delta_force = force[end_idx] - force[start_idx];
        let delta_time = (end_idx - start_idx) as f64 / self.sample_rate;

        if delta_time == 0.0 {
            return 0.0;
        }

        delta_force / delta_time
    }

    /// Calculate time to reach various force levels
    ///
    /// # Arguments
    ///
    /// * `force` - Array of force measurements in Newtons
    ///
    /// # Returns
    ///
    /// Timing metrics for reaching different percentages of peak force
    pub fn time_to_force_levels(&self, force: &[f64]) -> ForceTiming {
        if force.is_empty() {
            return ForceTiming::default();
        }

        let peak = force.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        ForceTiming {
            time_to_25: self.time_to_threshold(force, peak * 0.25),
            time_to_50: self.time_to_threshold(force, peak * 0.50),
            time_to_75: self.time_to_threshold(force, peak * 0.75),
            time_to_90: self.time_to_threshold(force, peak * 0.90),
        }
    }

    /// Find time to reach a specific force threshold
    fn time_to_threshold(&self, force: &[f64], threshold: f64) -> f64 {
        force
            .iter()
            .position(|&f| f >= threshold)
            .map(|idx| idx as f64 / self.sample_rate)
            .unwrap_or(0.0)
    }

    /// Calculate contractile impulse over a time window
    ///
    /// Contractile impulse is the integral of force over time, representing
    /// the total force-generating capacity.
    ///
    /// # Arguments
    ///
    /// * `force` - Array of force measurements in Newtons
    /// * `time_window_ms` - Duration of window in milliseconds
    ///
    /// # Returns
    ///
    /// Contractile impulse in Newton-seconds
    pub fn contractile_impulse(&self, force: &[f64], time_window_ms: f64) -> f64 {
        if force.is_empty() {
            return 0.0;
        }

        let window_samples = (time_window_ms / 1000.0 * self.sample_rate) as usize;
        let end_idx = window_samples.min(force.len());

        let dt = 1.0 / self.sample_rate;
        force[0..end_idx].iter().sum::<f64>() * dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_rfd() {
        let analyzer = RfdAnalyzer::new(1000.0); // 1000 Hz
        let force = vec![0.0, 100.0, 250.0, 400.0, 500.0];

        let peak_rfd = analyzer.peak_rfd(&force);
        // Max delta is 150N in 0.001s = 150000 N/s
        assert_eq!(peak_rfd, 150000.0);
    }

    #[test]
    fn test_rfd_at_interval() {
        let analyzer = RfdAnalyzer::new(1000.0);
        let force = vec![0.0; 50]; // 50ms of zeros
        let mut force = force;
        force.extend(vec![500.0; 100]); // Then jump to 500N

        // RFD from 0-100ms should be 5000 N/s (500N / 0.1s)
        let rfd = analyzer.rfd_at_interval(&force, 0.0, 100.0);
        assert_eq!(rfd, 5000.0);
    }

    #[test]
    fn test_time_to_force_levels() {
        let analyzer = RfdAnalyzer::new(1000.0);
        // Linear ramp from 0 to 100N over 100 samples (0.1s)
        let force: Vec<f64> = (0..=100).map(|i| i as f64).collect();

        let timing = analyzer.time_to_force_levels(&force);

        // 25% of 100 = 25, should occur at index 25 = 0.025s
        assert!((timing.time_to_25 - 0.025).abs() < 0.001);
        assert!((timing.time_to_50 - 0.050).abs() < 0.001);
        assert!((timing.time_to_75 - 0.075).abs() < 0.001);
        assert!((timing.time_to_90 - 0.090).abs() < 0.001);
    }

    #[test]
    fn test_contractile_impulse() {
        let analyzer = RfdAnalyzer::new(1000.0);
        let force = vec![100.0; 100]; // 100N for 100 samples = 0.1s

        let impulse = analyzer.contractile_impulse(&force, 100.0);
        // 100N * 0.1s = 10 N·s
        assert!((impulse - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_analyze() {
        let analyzer = RfdAnalyzer::new(1000.0);
        let force = vec![0.0; 50];
        let mut force = force;
        force.extend(vec![500.0; 200]);

        let metrics = analyzer.analyze(&force);
        assert!(metrics.peak_rfd > 0.0);
        assert!(metrics.rfd_0_100 > 0.0);
    }

    #[test]
    fn test_empty_force() {
        let analyzer = RfdAnalyzer::new(1000.0);
        let force: Vec<f64> = vec![];

        let metrics = analyzer.analyze(&force);
        assert_eq!(metrics.peak_rfd, 0.0);
    }
}
