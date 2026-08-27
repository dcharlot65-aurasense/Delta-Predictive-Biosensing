//! Criterion validity and sensitivity to change
//!
//! Implements:
//! - Pearson and Spearman correlation
//! - Bland-Altman analysis
//! - Sensitivity to change metrics
//! - Regression analysis

/// Criterion validity analyzer
#[derive(Debug, Clone)]
pub struct CriterionValidity {
    /// Confidence level for intervals
    pub confidence_level: f64,
}

impl Default for CriterionValidity {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
        }
    }
}

impl CriterionValidity {
    /// Calculate criterion validity metrics
    pub fn analyze(&self, algorithm_scores: &[f64], gold_standard: &[f64]) -> ValidityMetrics {
        if algorithm_scores.len() != gold_standard.len() || algorithm_scores.is_empty() {
            return ValidityMetrics::default();
        }

        let n = algorithm_scores.len();

        // Pearson correlation
        let pearson_r = self.pearson_correlation(algorithm_scores, gold_standard);

        // Spearman correlation
        let spearman_rho = self.spearman_correlation(algorithm_scores, gold_standard);

        // Linear regression
        let (slope, intercept, r_squared) = self.linear_regression(algorithm_scores, gold_standard);

        // Bland-Altman analysis
        let bland_altman = self.bland_altman(algorithm_scores, gold_standard);

        // Mean absolute error
        let mae: f64 = algorithm_scores
            .iter()
            .zip(gold_standard.iter())
            .map(|(a, g)| (a - g).abs())
            .sum::<f64>()
            / n as f64;

        // Root mean square error
        let rmse: f64 = (algorithm_scores
            .iter()
            .zip(gold_standard.iter())
            .map(|(a, g)| (a - g).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();

        // Concordance correlation coefficient (Lin's CCC)
        let ccc = self.concordance_correlation(algorithm_scores, gold_standard);

        ValidityMetrics {
            pearson_r,
            spearman_rho,
            r_squared,
            slope,
            intercept,
            bland_altman,
            mean_absolute_error: mae,
            root_mean_square_error: rmse,
            concordance_correlation: ccc,
            n_observations: n,
        }
    }

    /// Calculate Pearson correlation coefficient
    pub fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let covariance: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / (n - 1.0);

        let std_x = (x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let std_y = (y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        if std_x * std_y > 0.0 {
            covariance / (std_x * std_y)
        } else {
            0.0
        }
    }

    /// Calculate Spearman rank correlation
    pub fn spearman_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        // Convert to ranks
        let ranks_x = self.compute_ranks(x);
        let ranks_y = self.compute_ranks(y);

        // Pearson on ranks
        self.pearson_correlation(&ranks_x, &ranks_y)
    }

    /// Compute ranks for values
    fn compute_ranks(&self, values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut ranks = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            // Find tied values
            while j < n - 1 && indexed[j].1 == indexed[j + 1].1 {
                j += 1;
            }
            // Average rank for ties
            let avg_rank = (i + j) as f64 / 2.0 + 1.0;
            for k in i..=j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j + 1;
        }
        ranks
    }

    /// Linear regression
    pub fn linear_regression(&self, x: &[f64], y: &[f64]) -> (f64, f64, f64) {
        if x.len() != y.len() || x.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return (0.0, sum_y / n, 0.0);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        // R-squared
        let mean_y = sum_y / n;
        let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
            .sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        (slope, intercept, r_squared)
    }

    /// Bland-Altman analysis
    pub fn bland_altman(&self, method1: &[f64], method2: &[f64]) -> BlandAltmanResult {
        if method1.len() != method2.len() || method1.is_empty() {
            return BlandAltmanResult::default();
        }

        let n = method1.len();

        // Calculate means and differences
        let means: Vec<f64> = method1
            .iter()
            .zip(method2.iter())
            .map(|(m1, m2)| (m1 + m2) / 2.0)
            .collect();

        let differences: Vec<f64> = method1
            .iter()
            .zip(method2.iter())
            .map(|(m1, m2)| m1 - m2)
            .collect();

        let mean_diff = differences.iter().sum::<f64>() / n as f64;
        let sd_diff = (differences
            .iter()
            .map(|d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1) as f64)
            .sqrt();

        // Limits of agreement
        let loa_lower = mean_diff - 1.96 * sd_diff;
        let loa_upper = mean_diff + 1.96 * sd_diff;

        // Confidence intervals for LOA (Bland & Altman, 1999)
        let se_mean = sd_diff / (n as f64).sqrt();
        let se_loa = sd_diff * (3.0 / n as f64).sqrt();

        let mean_ci = (mean_diff - 1.96 * se_mean, mean_diff + 1.96 * se_mean);
        let loa_lower_ci = (loa_lower - 1.96 * se_loa, loa_lower + 1.96 * se_loa);
        let loa_upper_ci = (loa_upper - 1.96 * se_loa, loa_upper + 1.96 * se_loa);

        // Check for proportional bias (regression of difference on mean)
        let (slope, _, r_squared) = self.linear_regression(&means, &differences);
        let proportional_bias = r_squared > 0.1 && slope.abs() > 0.1;

        BlandAltmanResult {
            mean_difference: mean_diff,
            sd_difference: sd_diff,
            loa_lower,
            loa_upper,
            mean_ci,
            loa_lower_ci,
            loa_upper_ci,
            proportional_bias,
            proportional_bias_slope: slope,
        }
    }

    /// Calculate Lin's concordance correlation coefficient
    fn concordance_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / n;
        let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / n;

        let covar: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / n;

        let denominator = var_x + var_y + (mean_x - mean_y).powi(2);

        if denominator > 0.0 {
            2.0 * covar / denominator
        } else {
            0.0
        }
    }

    /// Analyze sensitivity to change
    pub fn sensitivity_to_change(
        &self,
        pre_treatment: &[f64],
        post_treatment: &[f64],
    ) -> ChangeMetrics {
        if pre_treatment.len() != post_treatment.len() || pre_treatment.is_empty() {
            return ChangeMetrics::default();
        }

        let n = pre_treatment.len();

        // Calculate changes
        let changes: Vec<f64> = pre_treatment
            .iter()
            .zip(post_treatment.iter())
            .map(|(pre, post)| post - pre)
            .collect();

        let mean_change = changes.iter().sum::<f64>() / n as f64;
        let sd_change = (changes
            .iter()
            .map(|c| (c - mean_change).powi(2))
            .sum::<f64>()
            / (n - 1) as f64)
            .sqrt();

        // Baseline SD
        let mean_pre = pre_treatment.iter().sum::<f64>() / n as f64;
        let sd_pre = (pre_treatment
            .iter()
            .map(|x| (x - mean_pre).powi(2))
            .sum::<f64>()
            / (n - 1) as f64)
            .sqrt();

        // Effect size (Cohen's d)
        let cohens_d = if sd_pre > 0.0 {
            mean_change / sd_pre
        } else {
            0.0
        };

        // Standardized Response Mean (SRM)
        let srm = if sd_change > 0.0 {
            mean_change / sd_change
        } else {
            0.0
        };

        // Paired t-test
        let t_statistic = if sd_change > 0.0 {
            mean_change / (sd_change / (n as f64).sqrt())
        } else {
            0.0
        };

        // Percent change
        let percent_change = if mean_pre.abs() > 0.0 {
            (mean_change / mean_pre) * 100.0
        } else {
            0.0
        };

        // Responder rate (using 0.5 SD as MCID approximation)
        let mcid = sd_pre * 0.5;
        let responders = changes.iter().filter(|&c| c.abs() >= mcid).count();
        let responder_rate = responders as f64 / n as f64 * 100.0;

        ChangeMetrics {
            mean_change,
            sd_change,
            cohens_d,
            standardized_response_mean: srm,
            t_statistic,
            percent_change,
            responder_rate,
            n_subjects: n,
        }
    }
}

/// Validity metrics result
#[derive(Debug, Clone, Default)]
pub struct ValidityMetrics {
    /// Pearson correlation
    pub pearson_r: f64,
    /// Spearman rank correlation
    pub spearman_rho: f64,
    /// R-squared from regression
    pub r_squared: f64,
    /// Regression slope
    pub slope: f64,
    /// Regression intercept
    pub intercept: f64,
    /// Bland-Altman analysis results
    pub bland_altman: BlandAltmanResult,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Root mean square error
    pub root_mean_square_error: f64,
    /// Lin's concordance correlation coefficient
    pub concordance_correlation: f64,
    /// Number of observations
    pub n_observations: usize,
}

/// Bland-Altman analysis result
#[derive(Debug, Clone, Default)]
pub struct BlandAltmanResult {
    /// Mean difference (bias)
    pub mean_difference: f64,
    /// SD of differences
    pub sd_difference: f64,
    /// Lower limit of agreement
    pub loa_lower: f64,
    /// Upper limit of agreement
    pub loa_upper: f64,
    /// 95% CI for mean difference
    pub mean_ci: (f64, f64),
    /// 95% CI for lower LOA
    pub loa_lower_ci: (f64, f64),
    /// 95% CI for upper LOA
    pub loa_upper_ci: (f64, f64),
    /// Whether proportional bias is present
    pub proportional_bias: bool,
    /// Slope of proportional bias
    pub proportional_bias_slope: f64,
}

/// Change metrics for sensitivity to change
#[derive(Debug, Clone, Default)]
pub struct ChangeMetrics {
    /// Mean change
    pub mean_change: f64,
    /// SD of change
    pub sd_change: f64,
    /// Cohen's d effect size
    pub cohens_d: f64,
    /// Standardized response mean
    pub standardized_response_mean: f64,
    /// Paired t-statistic
    pub t_statistic: f64,
    /// Percent change from baseline
    pub percent_change: f64,
    /// Responder rate (%)
    pub responder_rate: f64,
    /// Number of subjects
    pub n_subjects: usize,
}

impl ChangeMetrics {
    /// Interpret effect size
    pub fn interpret_effect_size(&self) -> EffectSizeInterpretation {
        let d = self.cohens_d.abs();
        if d < 0.2 {
            EffectSizeInterpretation::Negligible
        } else if d < 0.5 {
            EffectSizeInterpretation::Small
        } else if d < 0.8 {
            EffectSizeInterpretation::Medium
        } else {
            EffectSizeInterpretation::Large
        }
    }
}

/// Effect size interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectSizeInterpretation {
    /// Below Cohen's small threshold: no practically meaningful difference.
    Negligible,
    /// Cohen's "small" band.
    Small,
    /// Cohen's "medium" band.
    Medium,
    /// Cohen's "large" band or above.
    Large,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation() {
        let validity = CriterionValidity::default();

        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let r = validity.pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bland_altman() {
        let validity = CriterionValidity::default();

        let method1 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let method2 = vec![11.0, 21.0, 29.0, 41.0, 49.0];

        let ba = validity.bland_altman(&method1, &method2);

        // Mean difference should be close to 0
        assert!(ba.mean_difference.abs() < 1.0);
    }

    #[test]
    fn test_sensitivity_to_change() {
        let validity = CriterionValidity::default();

        // Large improvement
        let pre = vec![10.0, 12.0, 11.0, 13.0, 10.0];
        let post = vec![20.0, 22.0, 21.0, 23.0, 20.0];

        let change = validity.sensitivity_to_change(&pre, &post);

        assert!(change.cohens_d > 2.0); // Large effect
        assert_eq!(
            change.interpret_effect_size(),
            EffectSizeInterpretation::Large
        );
    }

    #[test]
    fn test_concordance_correlation() {
        let validity = CriterionValidity::default();

        // High agreement
        let x = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let y = vec![10.5, 19.5, 30.5, 39.5, 50.5];

        let metrics = validity.analyze(&x, &y);
        assert!(metrics.concordance_correlation > 0.95);
    }
}
