//! Longitudinal change detection for normative assessments
//!
//! Provides statistical methods for distinguishing real change from measurement
//! error when tracking individuals over time.
//!
//! # Key Concepts
//!
//! ## Minimal Detectable Change (MDC)
//!
//! The smallest change that can be considered a "real" change beyond measurement
//! error at a given confidence level. Calculated from test-retest reliability.
//!
//! **Formula:**
//! ```text
//! MDC = SEM × √2 × z
//! where SEM = SD × √(1 - ICC)
//! ```
//!
//! ## Standard Error of Measurement (SEM)
//!
//! The standard deviation of measurement errors. Quantifies precision of
//! repeated measurements.
//!
//! ## Reliable Change Index (RCI)
//!
//! Standardized measure of change accounting for regression to the mean and
//! measurement error. RCI > 1.96 indicates significant change at p < 0.05.
//!
//! # Clinical Applications
//!
//! - Treatment efficacy evaluation
//! - Disease progression monitoring
//! - Rehabilitation outcome assessment
//! - Identifying clinically meaningful improvement/decline
//!
//! # ⚠️ The reference values in this module are ILLUSTRATIVE
//!
//! They are placeholders that exist to exercise the API. They are not drawn from
//! any published cohort, they carry no citations, and they must not be used to
//! interpret a measurement from a real person. Any percentile, z-score or
//! classification computed against them demonstrates the arithmetic only.
//!
//! Supply your own cited reference values before drawing research conclusions.
//! See the crate-level documentation for the full statement.
//!
//! Research and educational use only. Not a medical device.

use serde::{Deserialize, Serialize};

/// Minimal detectable change at different confidence levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MinimalDetectableChange {
    /// 90% confidence MDC
    pub mdc_90: f64,
    /// 95% confidence MDC
    pub mdc_95: f64,
    /// Standard error of measurement
    pub sem: f64,
    /// Intraclass correlation coefficient (test-retest reliability)
    pub icc: f64,
}

impl MinimalDetectableChange {
    /// Create MDC from standard deviation and ICC
    ///
    /// # Arguments
    ///
    /// * `std_dev` - Standard deviation of the measure
    /// * `icc` - Intraclass correlation coefficient (0-1)
    ///
    /// # Returns
    ///
    /// MDC values at 90% and 95% confidence
    pub fn from_reliability(std_dev: f64, icc: f64) -> Self {
        // Standard error of measurement: SEM = SD × √(1 - ICC)
        let sem = std_dev * (1.0 - icc).sqrt();

        // Standard error of difference between two measurements
        let se_diff = sem * std::f64::consts::SQRT_2;

        // MDC at different confidence levels (z-scores)
        // 90% CI: z = 1.645
        // 95% CI: z = 1.96
        let mdc_90 = se_diff * 1.645;
        let mdc_95 = se_diff * 1.96;

        Self {
            mdc_90,
            mdc_95,
            sem,
            icc,
        }
    }

    /// Create MDC from test-retest data
    ///
    /// # Arguments
    ///
    /// * `test_retest` - Pairs of (test, retest) measurements
    ///
    /// # Returns
    ///
    /// MDC calculated from empirical test-retest data
    pub fn from_test_retest(test_retest: &[(f64, f64)]) -> Option<Self> {
        if test_retest.is_empty() {
            return None;
        }

        let n = test_retest.len() as f64;

        // Calculate means
        let mean_test: f64 = test_retest.iter().map(|(t, _)| t).sum::<f64>() / n;
        let mean_retest: f64 = test_retest.iter().map(|(_, r)| r).sum::<f64>() / n;

        // Calculate standard deviations
        let var_test: f64 = test_retest
            .iter()
            .map(|(t, _)| (t - mean_test).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let var_retest: f64 = test_retest
            .iter()
            .map(|(_, r)| (r - mean_retest).powi(2))
            .sum::<f64>()
            / (n - 1.0);

        // Calculate covariance
        let cov: f64 = test_retest
            .iter()
            .map(|(t, r)| (t - mean_test) * (r - mean_retest))
            .sum::<f64>()
            / (n - 1.0);

        // ICC(2,1) = (MSb - MSw) / (MSb + MSw)
        // Simplified: ICC = cov / sqrt(var_test × var_retest)
        let icc = cov / (var_test * var_retest).sqrt();
        let icc = icc.clamp(0.0, 1.0);

        // Average SD
        let std_dev = ((var_test + var_retest) / 2.0).sqrt();

        Some(Self::from_reliability(std_dev, icc))
    }

    /// Check if change exceeds 90% confidence MDC
    pub fn is_real_change_90(&self, change: f64) -> bool {
        change.abs() >= self.mdc_90
    }

    /// Check if change exceeds 95% confidence MDC
    pub fn is_real_change_95(&self, change: f64) -> bool {
        change.abs() >= self.mdc_95
    }

    /// Get the confidence level for a given change magnitude
    ///
    /// Returns approximate confidence level (0.0-1.0) that the change is real
    pub fn confidence_level(&self, change: f64) -> f64 {
        let z = change.abs() / (self.sem * std::f64::consts::SQRT_2);

        // Convert z to two-tailed probability
        let p = 2.0 * (1.0 - normal_cdf(z));
        (1.0 - p).clamp(0.0, 1.0)
    }
}

/// Status of change between two measurements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeStatus {
    /// No significant change detected
    NoChange,
    /// Improved beyond 90% MDC
    ImprovedMdc90,
    /// Improved beyond 95% MDC
    ImprovedMdc95,
    /// Declined beyond 90% MDC
    DeclinedMdc90,
    /// Declined beyond 95% MDC
    DeclinedMdc95,
}

impl ChangeStatus {
    /// Get descriptive label
    pub fn label(&self) -> &'static str {
        match self {
            ChangeStatus::NoChange => "No Significant Change",
            ChangeStatus::ImprovedMdc90 => "Improved (90% confidence)",
            ChangeStatus::ImprovedMdc95 => "Improved (95% confidence)",
            ChangeStatus::DeclinedMdc90 => "Declined (90% confidence)",
            ChangeStatus::DeclinedMdc95 => "Declined (95% confidence)",
        }
    }

    /// Check if change is statistically significant
    pub fn is_significant(&self) -> bool {
        !matches!(self, ChangeStatus::NoChange)
    }

    /// Check if change represents improvement
    pub fn is_improvement(&self) -> bool {
        matches!(
            self,
            ChangeStatus::ImprovedMdc90 | ChangeStatus::ImprovedMdc95
        )
    }

    /// Check if change represents decline
    pub fn is_decline(&self) -> bool {
        matches!(
            self,
            ChangeStatus::DeclinedMdc90 | ChangeStatus::DeclinedMdc95
        )
    }
}

/// Calculate minimal detectable change from test-retest reliability
///
/// # Arguments
///
/// * `test_retest` - Pairs of (test, retest) measurements from same individuals
/// * `confidence` - Confidence level (e.g., 0.90 or 0.95)
///
/// # Returns
///
/// MDC structure with SEM, ICC, and MDC values
pub fn calculate_mdc(test_retest: &[(f64, f64)], _confidence: f64) -> Option<MinimalDetectableChange> {
    MinimalDetectableChange::from_test_retest(test_retest)
}

/// Determine if change is real based on MDC
///
/// # Arguments
///
/// * `baseline` - Baseline measurement
/// * `followup` - Follow-up measurement
/// * `mdc` - Minimal detectable change thresholds
/// * `higher_is_better` - Whether higher values indicate improvement
///
/// # Returns
///
/// Change status indicating direction and confidence
pub fn is_real_change(
    baseline: f64,
    followup: f64,
    mdc: &MinimalDetectableChange,
    higher_is_better: bool,
) -> ChangeStatus {
    let change = followup - baseline;

    // Determine if improved or declined based on metric direction
    let improved = if higher_is_better {
        change > 0.0
    } else {
        change < 0.0
    };

    // Check MDC thresholds
    if mdc.is_real_change_95(change) {
        if improved {
            ChangeStatus::ImprovedMdc95
        } else {
            ChangeStatus::DeclinedMdc95
        }
    } else if mdc.is_real_change_90(change) {
        if improved {
            ChangeStatus::ImprovedMdc90
        } else {
            ChangeStatus::DeclinedMdc90
        }
    } else {
        ChangeStatus::NoChange
    }
}

/// Calculate Reliable Change Index (RCI)
///
/// RCI accounts for both measurement error and regression to the mean.
/// |RCI| > 1.96 indicates significant change at p < 0.05.
///
/// # Arguments
///
/// * `baseline` - Baseline measurement
/// * `followup` - Follow-up measurement
/// * `sem` - Standard error of measurement
///
/// # Returns
///
/// RCI value (z-score of change)
pub fn calculate_reliable_change_index(baseline: f64, followup: f64, sem: f64) -> f64 {
    let change = followup - baseline;
    let se_diff = sem * std::f64::consts::SQRT_2;

    change / se_diff
}

/// Check if RCI indicates significant change
///
/// # Arguments
///
/// * `rci` - Reliable change index value
/// * `alpha` - Significance level (e.g., 0.05 for p < 0.05)
///
/// # Returns
///
/// True if change is statistically significant
pub fn rci_is_significant(rci: f64, alpha: f64) -> bool {
    // Convert alpha to z-score (two-tailed)
    let z_critical = if (alpha - 0.05).abs() < 0.001 {
        1.96
    } else if (alpha - 0.01).abs() < 0.001 {
        2.576
    } else if (alpha - 0.10).abs() < 0.001 {
        1.645
    } else {
        // Default to 0.05
        1.96
    };

    rci.abs() >= z_critical
}

/// Longitudinal change analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeAnalysis {
    /// Baseline value
    pub baseline: f64,
    /// Follow-up value
    pub followup: f64,
    /// Raw change (followup - baseline)
    pub raw_change: f64,
    /// Percent change
    pub percent_change: f64,
    /// Change status
    pub status: ChangeStatus,
    /// Reliable change index
    pub rci: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Minimal detectable change used
    pub mdc: MinimalDetectableChange,
}

impl ChangeAnalysis {
    /// Perform longitudinal change analysis
    ///
    /// # Arguments
    ///
    /// * `baseline` - Baseline measurement
    /// * `followup` - Follow-up measurement
    /// * `mdc` - Minimal detectable change thresholds
    /// * `higher_is_better` - Whether higher values indicate improvement
    ///
    /// # Returns
    ///
    /// Complete change analysis
    pub fn analyze(
        baseline: f64,
        followup: f64,
        mdc: MinimalDetectableChange,
        higher_is_better: bool,
    ) -> Self {
        let raw_change = followup - baseline;
        let percent_change = if baseline.abs() > f64::EPSILON {
            (raw_change / baseline.abs()) * 100.0
        } else {
            0.0
        };

        let status = is_real_change(baseline, followup, &mdc, higher_is_better);
        let rci = calculate_reliable_change_index(baseline, followup, mdc.sem);
        let confidence = mdc.confidence_level(raw_change);

        Self {
            baseline,
            followup,
            raw_change,
            percent_change,
            status,
            rci,
            confidence,
            mdc,
        }
    }

    /// Check if change is clinically significant
    ///
    /// Combines statistical significance (MDC95) with a minimum effect size
    pub fn is_clinically_significant(&self, min_percent_change: f64) -> bool {
        // NOTE the parentheses. `&&` binds tighter than `||`, so without them
        // this read as `Improved || (Declined && threshold)` — the effect-size
        // threshold was ignored entirely for improvements, so any improvement
        // reaching MDC95 counted as clinically significant no matter how small,
        // while declines were held to the threshold. Both conditions must apply
        // in both directions, per this method's own documentation.
        (self.status == ChangeStatus::ImprovedMdc95
            || self.status == ChangeStatus::DeclinedMdc95)
            && self.percent_change.abs() >= min_percent_change
    }
}

/// Approximate normal CDF using Abramowitz and Stegun formula
fn normal_cdf(z: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let z = z.abs();

    let t = 1.0 / (1.0 + P * z);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-z * z / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdc_from_reliability() {
        let mdc = MinimalDetectableChange::from_reliability(10.0, 0.90);

        // SEM = 10 * sqrt(1 - 0.90) = 10 * sqrt(0.1) ≈ 3.16
        assert!((mdc.sem - 3.16).abs() < 0.01);

        // SE_diff = 3.16 * sqrt(2) ≈ 4.47
        // MDC95 = 4.47 * 1.96 ≈ 8.76
        assert!((mdc.mdc_95 - 8.76).abs() < 0.1);

        // MDC90 should be less than MDC95
        assert!(mdc.mdc_90 < mdc.mdc_95);
    }

    #[test]
    fn test_mdc_from_test_retest() {
        let data = vec![
            (100.0, 102.0),
            (95.0, 97.0),
            (105.0, 103.0),
            (98.0, 100.0),
            (102.0, 104.0),
        ];

        let mdc = MinimalDetectableChange::from_test_retest(&data);
        assert!(mdc.is_some());

        let mdc = mdc.unwrap();
        assert!(mdc.icc >= 0.0 && mdc.icc <= 1.0);
        assert!(mdc.sem > 0.0);
        assert!(mdc.mdc_95 > mdc.mdc_90);
    }

    #[test]
    fn test_is_real_change_improvement() {
        let mdc = MinimalDetectableChange::from_reliability(10.0, 0.90);

        // Small change - should be NoChange
        let status = is_real_change(100.0, 105.0, &mdc, true);
        assert_eq!(status, ChangeStatus::NoChange);

        // Large improvement - should exceed MDC95
        let status = is_real_change(100.0, 115.0, &mdc, true);
        assert_eq!(status, ChangeStatus::ImprovedMdc95);

        // Same change but lower-is-better metric
        let status = is_real_change(100.0, 115.0, &mdc, false);
        assert_eq!(status, ChangeStatus::DeclinedMdc95);
    }

    #[test]
    fn test_is_real_change_decline() {
        let mdc = MinimalDetectableChange::from_reliability(10.0, 0.90);

        // Large decline for higher-is-better metric
        let status = is_real_change(100.0, 85.0, &mdc, true);
        assert_eq!(status, ChangeStatus::DeclinedMdc95);

        // Same change but lower-is-better metric (improvement)
        let status = is_real_change(100.0, 85.0, &mdc, false);
        assert_eq!(status, ChangeStatus::ImprovedMdc95);
    }

    #[test]
    fn test_calculate_rci() {
        let sem = 3.16;

        // No change
        let rci = calculate_reliable_change_index(100.0, 100.0, sem);
        assert_eq!(rci, 0.0);

        // Moderate change
        let rci = calculate_reliable_change_index(100.0, 110.0, sem);
        // RCI = 10 / (3.16 * sqrt(2)) ≈ 2.24
        assert!((rci - 2.24).abs() < 0.1);
        assert!(rci_is_significant(rci, 0.05));
    }

    #[test]
    fn test_rci_significance() {
        // RCI > 1.96 is significant at p < 0.05
        assert!(rci_is_significant(2.0, 0.05));
        assert!(!rci_is_significant(1.5, 0.05));

        // RCI > 2.576 is significant at p < 0.01
        assert!(rci_is_significant(2.8, 0.01));
        assert!(!rci_is_significant(2.0, 0.01));
    }

    #[test]
    fn test_change_analysis() {
        let mdc = MinimalDetectableChange::from_reliability(10.0, 0.90);

        let analysis = ChangeAnalysis::analyze(100.0, 115.0, mdc, true);

        assert_eq!(analysis.baseline, 100.0);
        assert_eq!(analysis.followup, 115.0);
        assert_eq!(analysis.raw_change, 15.0);
        assert!((analysis.percent_change - 15.0).abs() < 0.1);
        assert_eq!(analysis.status, ChangeStatus::ImprovedMdc95);
        assert!(analysis.rci > 1.96);
        assert!(analysis.confidence > 0.95);
    }

    #[test]
    fn test_clinically_significant() {
        let mdc = MinimalDetectableChange::from_reliability(10.0, 0.90);

        // Large statistically significant change
        let analysis = ChangeAnalysis::analyze(100.0, 120.0, mdc, true);
        assert!(analysis.is_clinically_significant(10.0));

        // Statistically significant but below clinical threshold
        let analysis = ChangeAnalysis::analyze(100.0, 110.0, mdc, true);
        assert!(!analysis.is_clinically_significant(15.0));

        // Not statistically significant
        let analysis = ChangeAnalysis::analyze(100.0, 105.0, mdc, true);
        assert!(!analysis.is_clinically_significant(5.0));
    }

    #[test]
    fn test_confidence_level() {
        let mdc = MinimalDetectableChange::from_reliability(10.0, 0.90);

        // Large change should have high confidence
        let conf = mdc.confidence_level(15.0);
        assert!(conf > 0.90);

        // Small change should have lower confidence
        let conf = mdc.confidence_level(3.0);
        assert!(conf < 0.70);

        // No change should have ~0 confidence
        let conf = mdc.confidence_level(0.0);
        assert!(conf < 0.01);
    }

    #[test]
    fn test_change_status_methods() {
        assert!(ChangeStatus::ImprovedMdc95.is_significant());
        assert!(ChangeStatus::ImprovedMdc95.is_improvement());
        assert!(!ChangeStatus::ImprovedMdc95.is_decline());

        assert!(ChangeStatus::DeclinedMdc90.is_significant());
        assert!(!ChangeStatus::DeclinedMdc90.is_improvement());
        assert!(ChangeStatus::DeclinedMdc90.is_decline());

        assert!(!ChangeStatus::NoChange.is_significant());
        assert!(!ChangeStatus::NoChange.is_improvement());
        assert!(!ChangeStatus::NoChange.is_decline());
    }

    #[test]
    fn test_mdc_empty_data() {
        let mdc = MinimalDetectableChange::from_test_retest(&[]);
        assert!(mdc.is_none());
    }

    #[test]
    fn test_percent_change_zero_baseline() {
        let mdc = MinimalDetectableChange::from_reliability(10.0, 0.90);
        let analysis = ChangeAnalysis::analyze(0.0, 10.0, mdc, true);

        // Should handle division by zero gracefully
        assert_eq!(analysis.percent_change, 0.0);
    }
}
