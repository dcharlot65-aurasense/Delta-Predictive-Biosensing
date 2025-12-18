//! Grip strength analysis algorithms
//!
//! This module provides tools for analyzing grip strength measurements, including:
//! - Peak force detection
//! - Rate of force development
//! - Force steadiness during sustained contractions
//! - Fatigue index calculation
//! - Age and sex-based normative data

/// Grip strength analyzer for processing force-time curves from handgrip dynamometry
///
/// # Example
///
/// ```
/// use dpb_core::biomechanics::force::GripStrengthAnalyzer;
///
/// let analyzer = GripStrengthAnalyzer::new(1000.0); // 1000 Hz sample rate
/// let force_curve = vec![0.0, 50.0, 100.0, 120.0, 110.0, 100.0];
/// let metrics = analyzer.analyze(&force_curve);
/// println!("Peak force: {} N", metrics.peak_force);
/// ```
#[derive(Debug, Clone)]
pub struct GripStrengthAnalyzer {
    /// Sample rate in Hz
    pub sample_rate: f64,
}

/// Comprehensive grip strength metrics from a force-time curve
#[derive(Debug, Clone, Default)]
pub struct GripMetrics {
    /// Peak force achieved in Newtons
    pub peak_force: f64,
    /// Time to reach peak force in seconds
    pub time_to_peak: f64,
    /// Rate of force development in N/s
    pub rate_of_force_dev: f64,
    /// Force steadiness during hold phase (coefficient of variation)
    pub force_steadiness: f64,
    /// Fatigue index as percentage decline in force
    pub fatigue_index: f64,
}

impl GripStrengthAnalyzer {
    /// Create a new grip strength analyzer with specified sample rate
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sampling frequency in Hz
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }

    /// Perform complete analysis of a grip strength force curve
    ///
    /// # Arguments
    ///
    /// * `force_curve` - Array of force measurements in Newtons
    ///
    /// # Returns
    ///
    /// Complete grip strength metrics
    pub fn analyze(&self, force_curve: &[f64]) -> GripMetrics {
        if force_curve.is_empty() {
            return GripMetrics::default();
        }

        let peak_force = self.peak_force(force_curve);
        let time_to_peak = self.time_to_peak(force_curve);
        let rate_of_force_dev = self.rate_of_force_development(force_curve);

        // For force steadiness, use the middle 50% of the curve as the "hold" phase
        let hold_start = force_curve.len() / 4;
        let hold_end = (force_curve.len() * 3) / 4;
        let force_steadiness = if hold_end > hold_start {
            self.force_steadiness(force_curve, hold_start, hold_end)
        } else {
            0.0
        };

        // Use 10% of samples as window for fatigue calculation
        let window_samples = (force_curve.len() / 10).max(5);
        let fatigue_index = self.fatigue_index(force_curve, window_samples);

        GripMetrics {
            peak_force,
            time_to_peak,
            rate_of_force_dev,
            force_steadiness,
            fatigue_index,
        }
    }

    /// Calculate peak force in the force curve
    ///
    /// # Arguments
    ///
    /// * `force_curve` - Array of force measurements in Newtons
    ///
    /// # Returns
    ///
    /// Maximum force value in Newtons
    pub fn peak_force(&self, force_curve: &[f64]) -> f64 {
        force_curve
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0)
    }

    /// Calculate time to reach peak force
    ///
    /// # Arguments
    ///
    /// * `force_curve` - Array of force measurements in Newtons
    ///
    /// # Returns
    ///
    /// Time to peak in seconds
    pub fn time_to_peak(&self, force_curve: &[f64]) -> f64 {
        if force_curve.is_empty() {
            return 0.0;
        }

        let peak = self.peak_force(force_curve);
        let peak_idx = force_curve
            .iter()
            .position(|&f| f == peak)
            .unwrap_or(0);

        peak_idx as f64 / self.sample_rate
    }

    /// Calculate rate of force development (maximum slope)
    ///
    /// # Arguments
    ///
    /// * `force_curve` - Array of force measurements in Newtons
    ///
    /// # Returns
    ///
    /// Maximum rate of force development in N/s
    pub fn rate_of_force_development(&self, force_curve: &[f64]) -> f64 {
        if force_curve.len() < 2 {
            return 0.0;
        }

        let dt = 1.0 / self.sample_rate;
        let mut max_rfd = 0.0;

        for window in force_curve.windows(2) {
            let rfd = (window[1] - window[0]) / dt;
            max_rfd = max_rfd.max(rfd);
        }

        max_rfd
    }

    /// Calculate force steadiness during sustained contraction
    ///
    /// Force steadiness is measured as the coefficient of variation (CV)
    /// during the hold phase. Lower values indicate steadier force production.
    ///
    /// # Arguments
    ///
    /// * `force_curve` - Array of force measurements in Newtons
    /// * `hold_start` - Starting index of hold phase
    /// * `hold_end` - Ending index of hold phase
    ///
    /// # Returns
    ///
    /// Coefficient of variation (std/mean) during hold phase
    pub fn force_steadiness(&self, force_curve: &[f64], hold_start: usize, hold_end: usize) -> f64 {
        if hold_end <= hold_start || hold_end > force_curve.len() {
            return 0.0;
        }

        let hold_data = &force_curve[hold_start..hold_end];
        if hold_data.is_empty() {
            return 0.0;
        }

        let mean: f64 = hold_data.iter().sum::<f64>() / hold_data.len() as f64;
        if mean == 0.0 {
            return 0.0;
        }

        let variance: f64 = hold_data
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / hold_data.len() as f64;

        let std = variance.sqrt();
        (std / mean) * 100.0 // Return as percentage
    }

    /// Calculate fatigue index as the decline in force over time
    ///
    /// Compares average force in initial window to average force in final window.
    ///
    /// # Arguments
    ///
    /// * `force_curve` - Array of force measurements in Newtons
    /// * `window_samples` - Size of window for averaging
    ///
    /// # Returns
    ///
    /// Fatigue index as percentage decline (0-100)
    pub fn fatigue_index(&self, force_curve: &[f64], window_samples: usize) -> f64 {
        if force_curve.len() < window_samples * 2 {
            return 0.0;
        }

        let initial_window = &force_curve[0..window_samples];
        let final_start = force_curve.len() - window_samples;
        let final_window = &force_curve[final_start..];

        let initial_avg: f64 = initial_window.iter().sum::<f64>() / window_samples as f64;
        let final_avg: f64 = final_window.iter().sum::<f64>() / window_samples as f64;

        if initial_avg == 0.0 {
            return 0.0;
        }

        ((initial_avg - final_avg) / initial_avg * 100.0).max(0.0)
    }
}

/// Age and sex-based grip strength normative data
///
/// Based on normative data from sarcopenia research and clinical guidelines.
pub struct GripStrengthNorms;

impl GripStrengthNorms {
    /// Get approximate percentile for grip strength
    ///
    /// This is a simplified normative model. For clinical use, consult
    /// published normative tables specific to your population.
    ///
    /// # Arguments
    ///
    /// * `force_kg` - Grip strength in kilograms
    /// * `age` - Age in years
    /// * `is_male` - true for male, false for female
    ///
    /// # Returns
    ///
    /// Approximate percentile (0-100)
    pub fn percentile(force_kg: f64, age: u8, is_male: bool) -> f64 {
        // Simplified percentile estimation based on population norms
        // These are rough approximations and should be validated against
        // specific normative datasets for clinical use

        let (mean, std) = if is_male {
            // Male norms (simplified)
            let mean = if age < 30 {
                46.0
            } else if age < 50 {
                44.0
            } else if age < 70 {
                40.0
            } else {
                34.0
            };
            (mean, 8.0)
        } else {
            // Female norms (simplified)
            let mean = if age < 30 {
                28.0
            } else if age < 50 {
                26.0
            } else if age < 70 {
                24.0
            } else {
                20.0
            };
            (mean, 5.0)
        };

        // Convert to z-score and then approximate percentile
        let z_score = (force_kg - mean) / std;
        Self::z_to_percentile(z_score)
    }

    /// Sarcopenia cutoff values based on clinical criteria
    ///
    /// Based on European Working Group on Sarcopenia in Older People (EWGSOP2)
    /// and Asian Working Group for Sarcopenia (AWGS) criteria.
    ///
    /// # Arguments
    ///
    /// * `is_male` - true for male, false for female
    ///
    /// # Returns
    ///
    /// Cutoff value in kilograms below which sarcopenia may be indicated
    pub fn sarcopenia_cutoff(is_male: bool) -> f64 {
        if is_male {
            27.0 // <27 kg for males
        } else {
            16.0 // <16 kg for females
        }
    }

    /// Convert z-score to approximate percentile using error function
    fn z_to_percentile(z: f64) -> f64 {
        // Approximate CDF of standard normal using error function
        let percentile = 50.0 * (1.0 + Self::erf(z / std::f64::consts::SQRT_2));
        percentile.clamp(0.0, 100.0)
    }

    /// Approximation of the error function for percentile calculation
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 =  0.254829592;
        let a2 = -0.284496736;
        let a3 =  1.421413741;
        let a4 = -1.453152027;
        let a5 =  1.061405429;
        let p  =  0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_force() {
        let analyzer = GripStrengthAnalyzer::new(100.0);
        let force = vec![0.0, 50.0, 100.0, 120.0, 110.0, 100.0];
        assert_eq!(analyzer.peak_force(&force), 120.0);
    }

    #[test]
    fn test_time_to_peak() {
        let analyzer = GripStrengthAnalyzer::new(100.0);
        let force = vec![0.0, 50.0, 100.0, 120.0, 110.0, 100.0];
        assert_eq!(analyzer.time_to_peak(&force), 0.03); // 3/100
    }

    #[test]
    fn test_force_steadiness() {
        let analyzer = GripStrengthAnalyzer::new(100.0);
        // Constant force should have CV near 0
        let force = vec![100.0; 10];
        let cv = analyzer.force_steadiness(&force, 0, 10);
        assert!(cv < 0.01);
    }

    #[test]
    fn test_fatigue_index() {
        let analyzer = GripStrengthAnalyzer::new(100.0);
        // Force declining from 100 to 50
        let mut force = vec![100.0; 10];
        force.extend(vec![50.0; 10]);
        let fatigue = analyzer.fatigue_index(&force, 5);
        assert!(fatigue > 45.0 && fatigue < 55.0); // ~50% decline
    }

    #[test]
    fn test_sarcopenia_cutoffs() {
        assert_eq!(GripStrengthNorms::sarcopenia_cutoff(true), 27.0);
        assert_eq!(GripStrengthNorms::sarcopenia_cutoff(false), 16.0);
    }

    #[test]
    fn test_analyze() {
        let analyzer = GripStrengthAnalyzer::new(1000.0);
        let force = vec![0.0; 100]; // Should handle edge cases
        let metrics = analyzer.analyze(&force);
        assert_eq!(metrics.peak_force, 0.0);
    }
}
