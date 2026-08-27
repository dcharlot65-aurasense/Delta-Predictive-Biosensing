//! Center of Pressure (COP) analysis for balance assessment
//!
//! This module provides tools for analyzing center of pressure data from force plates
//! to assess postural control and balance. Key metrics include:
//!
//! - **Sway Area**: 95% confidence ellipse area (spatial extent)
//! - **Path Length**: Total distance traveled by COP
//! - **Mean Velocity**: Average speed of COP movement
//! - **RMS Displacement**: Root mean square displacement in AP and ML directions
//! - **Mean Frequency**: Dominant frequency of postural sway

use std::f64::consts::PI;

/// Center of pressure analyzer for balance assessment
///
/// # Example
///
/// ```rust
/// use dpb_core::biomechanics::balance::CopAnalyzer;
///
/// let analyzer = CopAnalyzer::new(100.0); // 100 Hz sampling
/// let cop_x = vec![0.0, 1.0, 2.0, 1.5, 1.0]; // mm
/// let cop_y = vec![0.0, 0.5, 1.0, 0.8, 0.5]; // mm
/// let metrics = analyzer.analyze(&cop_x, &cop_y);
/// println!("Sway area: {} mm²", metrics.sway_area);
/// ```
#[derive(Debug, Clone)]
pub struct CopAnalyzer {
    /// Sampling rate in Hz
    pub sample_rate: f64,
}

/// Comprehensive COP metrics for balance assessment
///
/// All spatial measurements are in millimeters (mm) unless otherwise noted.
#[derive(Debug, Clone, Default)]
pub struct CopMetrics {
    /// 95% confidence ellipse area (mm²)
    pub sway_area: f64,
    /// Total path length - distance traveled by COP (mm)
    pub path_length: f64,
    /// Mean velocity of COP movement (mm/s)
    pub mean_velocity: f64,
    /// Root mean square displacement in anterior-posterior direction (mm)
    pub rms_ap: f64,
    /// Root mean square displacement in medial-lateral direction (mm)
    pub rms_ml: f64,
    /// Mean frequency of sway in anterior-posterior direction (Hz)
    pub mean_frequency_ap: f64,
    /// Mean frequency of sway in medial-lateral direction (Hz)
    pub mean_frequency_ml: f64,
    /// Range (max - min) in anterior-posterior direction (mm)
    pub range_ap: f64,
    /// Range (max - min) in medial-lateral direction (mm)
    pub range_ml: f64,
}

impl CopAnalyzer {
    /// Create a new COP analyzer with the specified sampling rate
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sampling rate in Hz (e.g., 100.0 for 100 Hz)
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }

    /// Perform comprehensive COP analysis on the provided data
    ///
    /// # Arguments
    ///
    /// * `cop_x` - COP positions in medial-lateral direction (mm)
    /// * `cop_y` - COP positions in anterior-posterior direction (mm)
    ///
    /// # Returns
    ///
    /// A `CopMetrics` structure containing all computed metrics
    pub fn analyze(&self, cop_x: &[f64], cop_y: &[f64]) -> CopMetrics {
        CopMetrics {
            sway_area: self.sway_area(cop_x, cop_y),
            path_length: self.path_length(cop_x, cop_y),
            mean_velocity: self.mean_velocity(cop_x, cop_y),
            rms_ap: self.rms_displacement(cop_y),
            rms_ml: self.rms_displacement(cop_x),
            mean_frequency_ap: self.mean_frequency(cop_y),
            mean_frequency_ml: self.mean_frequency(cop_x),
            range_ap: range(cop_y),
            range_ml: range(cop_x),
        }
    }

    /// Calculate 95% confidence ellipse area
    ///
    /// This represents the spatial extent of postural sway.
    /// Larger values indicate greater postural instability.
    ///
    /// # Arguments
    ///
    /// * `cop_x` - COP positions in medial-lateral direction (mm)
    /// * `cop_y` - COP positions in anterior-posterior direction (mm)
    ///
    /// # Returns
    ///
    /// Area of 95% confidence ellipse in mm²
    pub fn sway_area(&self, cop_x: &[f64], cop_y: &[f64]) -> f64 {
        confidence_ellipse_area(cop_x, cop_y)
    }

    /// Calculate total path length (sway path)
    ///
    /// Sum of Euclidean distances between consecutive COP positions.
    /// Higher values indicate more postural corrections.
    ///
    /// # Arguments
    ///
    /// * `cop_x` - COP positions in medial-lateral direction (mm)
    /// * `cop_y` - COP positions in anterior-posterior direction (mm)
    ///
    /// # Returns
    ///
    /// Total path length in mm
    pub fn path_length(&self, cop_x: &[f64], cop_y: &[f64]) -> f64 {
        if cop_x.len() < 2 {
            return 0.0;
        }

        cop_x
            .windows(2)
            .zip(cop_y.windows(2))
            .map(|(x_win, y_win)| {
                let dx = x_win[1] - x_win[0];
                let dy = y_win[1] - y_win[0];
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    }

    /// Calculate mean sway velocity
    ///
    /// Path length divided by time duration.
    ///
    /// # Arguments
    ///
    /// * `cop_x` - COP positions in medial-lateral direction (mm)
    /// * `cop_y` - COP positions in anterior-posterior direction (mm)
    ///
    /// # Returns
    ///
    /// Mean velocity in mm/s
    pub fn mean_velocity(&self, cop_x: &[f64], cop_y: &[f64]) -> f64 {
        if cop_x.is_empty() {
            return 0.0;
        }

        let path = self.path_length(cop_x, cop_y);
        let duration = (cop_x.len() - 1) as f64 / self.sample_rate;

        if duration > 0.0 { path / duration } else { 0.0 }
    }

    /// Calculate RMS (root mean square) displacement
    ///
    /// Measure of the average magnitude of oscillation around the mean position.
    ///
    /// # Arguments
    ///
    /// * `cop` - COP positions in one direction (mm)
    ///
    /// # Returns
    ///
    /// RMS displacement in mm
    pub fn rms_displacement(&self, cop: &[f64]) -> f64 {
        if cop.is_empty() {
            return 0.0;
        }

        let mean: f64 = cop.iter().sum::<f64>() / cop.len() as f64;
        let variance: f64 = cop.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / cop.len() as f64;

        variance.sqrt()
    }

    /// Calculate mean frequency of sway using spectral centroid
    ///
    /// Represents the "center of mass" of the power spectrum.
    /// Higher frequencies may indicate more rigid postural control.
    ///
    /// # Arguments
    ///
    /// * `cop` - COP positions in one direction (mm)
    ///
    /// # Returns
    ///
    /// Mean frequency in Hz
    pub fn mean_frequency(&self, cop: &[f64]) -> f64 {
        if cop.len() < 4 {
            return 0.0;
        }

        // Simple implementation using zero-crossing rate as a proxy
        // For a more accurate implementation, use FFT
        let mean: f64 = cop.iter().sum::<f64>() / cop.len() as f64;
        let centered: Vec<f64> = cop.iter().map(|&x| x - mean).collect();

        let zero_crossings = centered.windows(2).filter(|w| w[0] * w[1] < 0.0).count();

        // Estimate frequency from zero-crossing rate
        // Each cycle has 2 zero crossings
        (zero_crossings as f64 / 2.0) * self.sample_rate / cop.len() as f64
    }

    /// Calculate Romberg quotient (eyes closed / eyes open)
    ///
    /// Measures the contribution of vision to postural control.
    /// Values > 1 indicate reliance on vision; > 2 may suggest vestibular dysfunction.
    ///
    /// # Arguments
    ///
    /// * `eyes_open` - COP metrics with eyes open
    /// * `eyes_closed` - COP metrics with eyes closed
    ///
    /// # Returns
    ///
    /// Romberg quotient (ratio of sway areas)
    pub fn romberg_quotient(eyes_open: &CopMetrics, eyes_closed: &CopMetrics) -> f64 {
        if eyes_open.sway_area > 0.0 {
            eyes_closed.sway_area / eyes_open.sway_area
        } else {
            0.0
        }
    }
}

/// Calculate 95% confidence ellipse area from 2D data
///
/// Uses the covariance matrix to determine the ellipse parameters.
/// The 95% confidence region corresponds to chi-squared with 2 df = 5.991.
///
/// # Arguments
///
/// * `x` - X coordinates
/// * `y` - Y coordinates
///
/// # Returns
///
/// Area of 95% confidence ellipse
fn confidence_ellipse_area(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 || x.len() != y.len() {
        return 0.0;
    }

    let n = x.len() as f64;

    // Calculate means
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    // Calculate covariance matrix elements
    let var_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f64>() / n;
    let var_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<f64>() / n;
    let cov_xy: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / n;

    // Calculate eigenvalues of covariance matrix
    let trace = var_x + var_y;
    let det = var_x * var_y - cov_xy * cov_xy;
    let discriminant = (trace * trace - 4.0 * det).max(0.0).sqrt();

    let lambda1 = (trace + discriminant) / 2.0;
    let lambda2 = (trace - discriminant) / 2.0;

    // 95% confidence corresponds to chi-squared(2) = 5.991
    let chi2_95 = 5.991;

    // Ellipse area = π * a * b, where a and b are semi-axes
    PI * chi2_95 * lambda1.sqrt() * lambda2.sqrt()
}

/// Calculate range (max - min) of a signal
fn range(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let min = data
        .iter()
        .fold(f64::INFINITY, |a, &b| if b < a { b } else { a });
    let max = data
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| if b > a { b } else { a });

    max - min
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cop_analyzer_creation() {
        let analyzer = CopAnalyzer::new(100.0);
        assert_eq!(analyzer.sample_rate, 100.0);
    }

    #[test]
    fn test_path_length() {
        let analyzer = CopAnalyzer::new(100.0);

        // Open path along the axes: 3 across, 4 up, 3 back.
        let path = analyzer.path_length(&[0.0, 3.0, 3.0, 0.0], &[0.0, 0.0, 4.0, 4.0]);
        assert!((path - 10.0).abs() < 0.01, "got {path}, expected 3 + 4 + 3");

        // Closed 3-4-5 triangle, so the last leg is the hypotenuse. This is
        // what the original expectation of 12 described, but the points it
        // used returned to (0, 4) rather than the origin, tracing 10.
        let path = analyzer.path_length(&[0.0, 3.0, 3.0, 0.0], &[0.0, 0.0, 4.0, 0.0]);
        assert!((path - 12.0).abs() < 0.01, "got {path}, expected 3 + 4 + 5");
    }

    #[test]
    fn test_mean_velocity() {
        let analyzer = CopAnalyzer::new(100.0);
        let cop_x = vec![0.0, 10.0]; // 10mm in 0.01s
        let cop_y = vec![0.0, 0.0];

        let velocity = analyzer.mean_velocity(&cop_x, &cop_y);
        // 10mm / 0.01s = 1000 mm/s
        assert!((velocity - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_rms_displacement() {
        let analyzer = CopAnalyzer::new(100.0);
        let cop = vec![-1.0, 0.0, 1.0]; // Mean = 0, variance = 2/3

        let rms = analyzer.rms_displacement(&cop);
        let expected = (2.0 / 3.0_f64).sqrt();
        assert!((rms - expected).abs() < 0.01);
    }

    #[test]
    fn test_range() {
        let data = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(range(&data), 4.0);
    }

    #[test]
    fn test_romberg_quotient() {
        let eyes_open = CopMetrics {
            sway_area: 100.0,
            ..Default::default()
        };
        let eyes_closed = CopMetrics {
            sway_area: 200.0,
            ..Default::default()
        };

        let quotient = CopAnalyzer::romberg_quotient(&eyes_open, &eyes_closed);
        assert_eq!(quotient, 2.0);
    }

    #[test]
    fn test_confidence_ellipse_area() {
        // Simple test with unit circle data
        let x = vec![1.0, 0.0, -1.0, 0.0];
        let y = vec![0.0, 1.0, 0.0, -1.0];

        let area = confidence_ellipse_area(&x, &y);
        // Should be a positive value
        assert!(area > 0.0);
    }
}
