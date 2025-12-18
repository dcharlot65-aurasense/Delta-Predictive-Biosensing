//! Test-retest and inter-rater reliability metrics
//!
//! Implements:
//! - Intraclass Correlation Coefficient (ICC)
//! - Standard Error of Measurement (SEM)
//! - Minimal Detectable Change (MDC)
//! - Coefficient of Variation

/// ICC model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccModel {
    /// One-way random effects model
    OneWayRandom,
    /// Two-way random effects model
    TwoWayRandom,
    /// Two-way mixed effects model
    TwoWayMixed,
}

/// ICC type for agreement vs consistency
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccType {
    /// Single measures
    SingleMeasures,
    /// Average measures
    AverageMeasures,
    /// Absolute agreement
    AbsoluteAgreement,
    /// Consistency
    Consistency,
}

/// Test-retest reliability analyzer
#[derive(Debug, Clone)]
pub struct TestRetestReliability {
    /// ICC model to use
    pub model: IccModel,
    /// ICC type
    pub icc_type: IccType,
    /// Confidence level for intervals
    pub confidence_level: f64,
}

impl Default for TestRetestReliability {
    fn default() -> Self {
        Self {
            model: IccModel::TwoWayMixed,
            icc_type: IccType::AbsoluteAgreement,
            confidence_level: 0.95,
        }
    }
}

impl TestRetestReliability {
    /// Create new reliability analyzer
    pub fn new(model: IccModel, icc_type: IccType) -> Self {
        Self {
            model,
            icc_type,
            confidence_level: 0.95,
        }
    }

    /// Calculate test-retest reliability metrics
    pub fn analyze(&self, test1: &[f64], test2: &[f64]) -> ReliabilityMetrics {
        if test1.len() != test2.len() || test1.is_empty() {
            return ReliabilityMetrics::default();
        }

        let n = test1.len();

        // Calculate means
        let mean1 = test1.iter().sum::<f64>() / n as f64;
        let mean2 = test2.iter().sum::<f64>() / n as f64;
        let grand_mean = (mean1 + mean2) / 2.0;

        // Calculate variance components for ICC
        // Between-subjects variance
        let subject_means: Vec<f64> = test1
            .iter()
            .zip(test2.iter())
            .map(|(t1, t2)| (t1 + t2) / 2.0)
            .collect();

        let ss_between: f64 = subject_means
            .iter()
            .map(|m| (m - grand_mean).powi(2))
            .sum::<f64>()
            * 2.0;

        // Within-subjects variance
        let ss_within: f64 = test1
            .iter()
            .zip(test2.iter())
            .zip(subject_means.iter())
            .map(|((t1, t2), sm)| (t1 - sm).powi(2) + (t2 - sm).powi(2))
            .sum();

        // Mean squares
        let ms_between = ss_between / (n - 1) as f64;
        let ms_within = ss_within / n as f64;

        // ICC(2,1) - Two-way random, single measures, absolute agreement
        let icc = if (ms_between + ms_within) > 0.0 {
            (ms_between - ms_within) / (ms_between + ms_within)
        } else {
            0.0
        };

        // Standard Error of Measurement
        let pooled_sd = self.pooled_sd(test1, test2);
        let sem = pooled_sd * (1.0 - icc).sqrt();

        // Minimal Detectable Change (95% confidence)
        let mdc_95 = sem * 1.96 * std::f64::consts::SQRT_2;
        let mdc_90 = sem * 1.645 * std::f64::consts::SQRT_2;

        // Coefficient of Variation
        let cv = self.coefficient_of_variation(test1, test2);

        // Limits of Agreement
        let differences: Vec<f64> = test1
            .iter()
            .zip(test2.iter())
            .map(|(t1, t2)| t1 - t2)
            .collect();

        let mean_diff = differences.iter().sum::<f64>() / n as f64;
        let sd_diff = (differences
            .iter()
            .map(|d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1) as f64)
            .sqrt();

        let loa_lower = mean_diff - 1.96 * sd_diff;
        let loa_upper = mean_diff + 1.96 * sd_diff;

        // ICC confidence interval (Fisher transformation)
        let (icc_lower, icc_upper) = self.icc_confidence_interval(icc, n);

        ReliabilityMetrics {
            icc,
            icc_lower,
            icc_upper,
            sem,
            mdc_95,
            mdc_90,
            coefficient_of_variation: cv,
            mean_difference: mean_diff,
            sd_difference: sd_diff,
            loa_lower,
            loa_upper,
            n_subjects: n,
        }
    }

    /// Calculate inter-rater reliability
    pub fn inter_rater_reliability(&self, rater_scores: &[Vec<f64>]) -> f64 {
        if rater_scores.is_empty() || rater_scores[0].is_empty() {
            return 0.0;
        }

        let n_raters = rater_scores.len();
        let n_subjects = rater_scores[0].len();

        // Ensure all raters rated same number of subjects
        if !rater_scores.iter().all(|r| r.len() == n_subjects) {
            return 0.0;
        }

        // Calculate grand mean
        let total: f64 = rater_scores.iter().flat_map(|r| r.iter()).sum();
        let grand_mean = total / (n_raters * n_subjects) as f64;

        // Calculate subject means
        let subject_means: Vec<f64> = (0..n_subjects)
            .map(|s| {
                rater_scores.iter().map(|r| r[s]).sum::<f64>() / n_raters as f64
            })
            .collect();

        // Between-subjects sum of squares
        let ss_between: f64 = subject_means
            .iter()
            .map(|m| (m - grand_mean).powi(2))
            .sum::<f64>()
            * n_raters as f64;

        // Total sum of squares
        let ss_total: f64 = rater_scores
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| (x - grand_mean).powi(2))
            .sum();

        // Within-subjects sum of squares
        let ss_within = ss_total - ss_between;

        // Mean squares
        let ms_between = ss_between / (n_subjects - 1) as f64;
        let ms_within = ss_within / ((n_subjects - 1) * (n_raters - 1)) as f64;

        // ICC(2,k) for average measures
        if (ms_between + (n_raters as f64 - 1.0) * ms_within) > 0.0 {
            (ms_between - ms_within) / ms_between
        } else {
            0.0
        }
    }

    /// Calculate Cronbach's alpha for internal consistency
    pub fn cronbachs_alpha(&self, items: &[Vec<f64>]) -> f64 {
        if items.is_empty() || items[0].is_empty() {
            return 0.0;
        }

        let k = items.len() as f64; // Number of items
        let n = items[0].len();

        // Calculate variance of each item
        let item_variances: Vec<f64> = items.iter().map(|item| {
            let mean = item.iter().sum::<f64>() / n as f64;
            item.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        }).collect();

        // Calculate total score variance
        let total_scores: Vec<f64> = (0..n)
            .map(|i| items.iter().map(|item| item[i]).sum())
            .collect();

        let total_mean = total_scores.iter().sum::<f64>() / n as f64;
        let total_variance = total_scores
            .iter()
            .map(|x| (x - total_mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;

        if total_variance <= 0.0 {
            return 0.0;
        }

        let sum_item_variance: f64 = item_variances.iter().sum();

        (k / (k - 1.0)) * (1.0 - sum_item_variance / total_variance)
    }

    /// Calculate pooled standard deviation
    fn pooled_sd(&self, test1: &[f64], test2: &[f64]) -> f64 {
        let n = test1.len();
        if n < 2 {
            return 0.0;
        }

        let mean1 = test1.iter().sum::<f64>() / n as f64;
        let mean2 = test2.iter().sum::<f64>() / n as f64;

        let var1: f64 = test1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n - 1) as f64;
        let var2: f64 = test2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n - 1) as f64;

        ((var1 + var2) / 2.0).sqrt()
    }

    /// Calculate coefficient of variation
    fn coefficient_of_variation(&self, test1: &[f64], test2: &[f64]) -> f64 {
        let n = test1.len();
        if n == 0 {
            return 0.0;
        }

        let differences: Vec<f64> = test1
            .iter()
            .zip(test2.iter())
            .map(|(t1, t2)| (t1 - t2).abs())
            .collect();

        let mean_abs_diff = differences.iter().sum::<f64>() / n as f64;
        let mean_value = (test1.iter().sum::<f64>() + test2.iter().sum::<f64>()) / (2.0 * n as f64);

        if mean_value > 0.0 {
            (mean_abs_diff / mean_value) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate ICC confidence interval using Fisher transformation
    fn icc_confidence_interval(&self, icc: f64, n: usize) -> (f64, f64) {
        // Fisher z transformation
        let z = 0.5 * ((1.0 + icc) / (1.0 - icc + 1e-10)).ln();

        // Standard error of z
        let se_z = 1.0 / ((n - 3) as f64).sqrt();

        // Critical value for 95% CI
        let z_crit = 1.96;

        // Confidence limits in z scale
        let z_lower = z - z_crit * se_z;
        let z_upper = z + z_crit * se_z;

        // Back-transform to ICC scale
        let icc_lower = (z_lower.exp().powi(2) - 1.0) / (z_lower.exp().powi(2) + 1.0);
        let icc_upper = (z_upper.exp().powi(2) - 1.0) / (z_upper.exp().powi(2) + 1.0);

        (icc_lower.max(-1.0), icc_upper.min(1.0))
    }

    /// Interpret ICC value
    pub fn interpret_icc(icc: f64) -> IccInterpretation {
        if icc < 0.5 {
            IccInterpretation::Poor
        } else if icc < 0.75 {
            IccInterpretation::Moderate
        } else if icc < 0.9 {
            IccInterpretation::Good
        } else {
            IccInterpretation::Excellent
        }
    }
}

/// Reliability metrics result
#[derive(Debug, Clone, Default)]
pub struct ReliabilityMetrics {
    /// Intraclass correlation coefficient
    pub icc: f64,
    /// Lower bound of ICC 95% CI
    pub icc_lower: f64,
    /// Upper bound of ICC 95% CI
    pub icc_upper: f64,
    /// Standard error of measurement
    pub sem: f64,
    /// Minimal detectable change (95%)
    pub mdc_95: f64,
    /// Minimal detectable change (90%)
    pub mdc_90: f64,
    /// Coefficient of variation (%)
    pub coefficient_of_variation: f64,
    /// Mean difference between tests
    pub mean_difference: f64,
    /// SD of differences
    pub sd_difference: f64,
    /// Lower limit of agreement
    pub loa_lower: f64,
    /// Upper limit of agreement
    pub loa_upper: f64,
    /// Number of subjects
    pub n_subjects: usize,
}

/// ICC interpretation category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccInterpretation {
    /// ICC < 0.5
    Poor,
    /// ICC 0.5-0.75
    Moderate,
    /// ICC 0.75-0.9
    Good,
    /// ICC > 0.9
    Excellent,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icc_calculation() {
        let reliability = TestRetestReliability::default();

        // High reliability data (small differences)
        let test1 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let test2 = vec![11.0, 19.0, 31.0, 39.0, 51.0];

        let metrics = reliability.analyze(&test1, &test2);

        assert!(metrics.icc > 0.9);
        assert!(metrics.sem < 5.0);
    }

    #[test]
    fn test_low_reliability() {
        let reliability = TestRetestReliability::default();

        // Low reliability data (large differences)
        let test1 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let test2 = vec![50.0, 10.0, 40.0, 20.0, 30.0];

        let metrics = reliability.analyze(&test1, &test2);

        assert!(metrics.icc < 0.5);
    }

    #[test]
    fn test_inter_rater() {
        let reliability = TestRetestReliability::default();

        // Three raters, five subjects
        let rater_scores = vec![
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![11.0, 19.0, 31.0, 39.0, 51.0],
            vec![10.0, 21.0, 29.0, 41.0, 49.0],
        ];

        let icc = reliability.inter_rater_reliability(&rater_scores);
        assert!(icc > 0.9);
    }

    #[test]
    fn test_icc_interpretation() {
        assert_eq!(TestRetestReliability::interpret_icc(0.3), IccInterpretation::Poor);
        assert_eq!(TestRetestReliability::interpret_icc(0.6), IccInterpretation::Moderate);
        assert_eq!(TestRetestReliability::interpret_icc(0.8), IccInterpretation::Good);
        assert_eq!(TestRetestReliability::interpret_icc(0.95), IccInterpretation::Excellent);
    }
}
