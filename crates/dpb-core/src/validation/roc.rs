//! Receiver Operating Characteristic (ROC) analysis
//!
//! Implements:
//! - ROC curve generation
//! - Area Under Curve (AUC) calculation
//! - Optimal cutoff selection
//! - Sensitivity/specificity at cutoffs

/// ROC curve analyzer
#[derive(Debug, Clone)]
pub struct RocAnalyzer {
    /// Number of threshold points to evaluate
    pub n_thresholds: usize,
}

impl Default for RocAnalyzer {
    fn default() -> Self {
        Self { n_thresholds: 100 }
    }
}

impl RocAnalyzer {
    /// Generate ROC curve from scores and labels
    pub fn analyze(&self, scores: &[f64], labels: &[bool]) -> RocCurve {
        if scores.len() != labels.len() || scores.is_empty() {
            return RocCurve::default();
        }

        // Count positives and negatives
        let n_positive = labels.iter().filter(|&&l| l).count();
        let n_negative = labels.len() - n_positive;

        if n_positive == 0 || n_negative == 0 {
            return RocCurve::default();
        }

        // Sort by score descending
        let mut indexed: Vec<(f64, bool)> = scores
            .iter()
            .cloned()
            .zip(labels.iter().cloned())
            .collect();
        indexed.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Generate ROC points
        let mut points = Vec::new();
        let mut tp = 0;
        let mut fp = 0;

        // Add (0, 0) point
        points.push(RocPoint {
            threshold: f64::INFINITY,
            true_positive_rate: 0.0,
            false_positive_rate: 0.0,
            sensitivity: 0.0,
            specificity: 1.0,
            ppv: 0.0,
            npv: 0.0,
            accuracy: n_negative as f64 / (n_positive + n_negative) as f64,
            f1_score: 0.0,
            youden_index: -1.0,
        });

        for (i, (score, label)) in indexed.iter().enumerate() {
            if *label {
                tp += 1;
            } else {
                fp += 1;
            }

            let tpr = tp as f64 / n_positive as f64;
            let fpr = fp as f64 / n_negative as f64;
            let tn = n_negative - fp;
            let fn_count = n_positive - tp;

            let sensitivity = tpr;
            let specificity = 1.0 - fpr;

            let ppv = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };

            let npv = if tn + fn_count > 0 {
                tn as f64 / (tn + fn_count) as f64
            } else {
                0.0
            };

            let accuracy = (tp + tn) as f64 / (n_positive + n_negative) as f64;

            let f1 = if ppv + sensitivity > 0.0 {
                2.0 * ppv * sensitivity / (ppv + sensitivity)
            } else {
                0.0
            };

            let youden = sensitivity + specificity - 1.0;

            // Emit one point per DISTINCT threshold, not per instance.
            //
            // Instances sharing a score cannot be separated by any threshold, so
            // they must advance the curve together. Emitting a point inside a
            // tie group draws a staircase whose shape depends on the order the
            // tied instances happen to be sorted into, and the trapezoidal AUC
            // then depends on that order too: eight instances all scored 0.5,
            // four positive and four negative, gave 0.375 with negatives first
            // and 0.625 with positives first, where the only correct answer for
            // a score that carries no information is 0.5.
            let last_of_tie_group =
                i + 1 == indexed.len() || indexed[i + 1].0 != *score;
            if !last_of_tie_group {
                continue;
            }

            points.push(RocPoint {
                threshold: *score,
                true_positive_rate: tpr,
                false_positive_rate: fpr,
                sensitivity,
                specificity,
                ppv,
                npv,
                accuracy,
                f1_score: f1,
                youden_index: youden,
            });
        }

        // Calculate AUC using trapezoidal rule
        let auc = self.calculate_auc(&points);

        // Find optimal cutoffs
        let optimal_youden = self.find_optimal_youden(&points);
        let optimal_closest_to_01 = self.find_optimal_closest(&points);

        // Calculate confidence interval for AUC (DeLong method approximation)
        let (auc_lower, auc_upper) = self.auc_confidence_interval(auc, n_positive, n_negative);

        RocCurve {
            points,
            auc,
            auc_lower,
            auc_upper,
            optimal_youden,
            optimal_closest_to_01,
            n_positive,
            n_negative,
        }
    }

    /// Calculate AUC using trapezoidal rule
    fn calculate_auc(&self, points: &[RocPoint]) -> f64 {
        if points.len() < 2 {
            return 0.5;
        }

        let mut auc = 0.0;
        for i in 1..points.len() {
            let width = points[i].false_positive_rate - points[i - 1].false_positive_rate;
            let height = (points[i].true_positive_rate + points[i - 1].true_positive_rate) / 2.0;
            auc += width * height;
        }

        auc.clamp(0.0, 1.0)
    }

    /// Find optimal cutoff using Youden's index
    fn find_optimal_youden(&self, points: &[RocPoint]) -> OptimalCutoff {
        let best = points
            .iter()
            .max_by(|a, b| a.youden_index.total_cmp(&b.youden_index))
            .unwrap();

        OptimalCutoff {
            threshold: best.threshold,
            sensitivity: best.sensitivity,
            specificity: best.specificity,
            ppv: best.ppv,
            npv: best.npv,
            accuracy: best.accuracy,
            youden_index: best.youden_index,
            method: CutoffMethod::YoudenIndex,
        }
    }

    /// Find optimal cutoff closest to (0, 1) corner
    fn find_optimal_closest(&self, points: &[RocPoint]) -> OptimalCutoff {
        let best = points
            .iter()
            .min_by(|a, b| {
                let dist_a = a.false_positive_rate.powi(2) + (1.0 - a.true_positive_rate).powi(2);
                let dist_b = b.false_positive_rate.powi(2) + (1.0 - b.true_positive_rate).powi(2);
                dist_a.total_cmp(&dist_b)
            })
            .unwrap();

        OptimalCutoff {
            threshold: best.threshold,
            sensitivity: best.sensitivity,
            specificity: best.specificity,
            ppv: best.ppv,
            npv: best.npv,
            accuracy: best.accuracy,
            youden_index: best.youden_index,
            method: CutoffMethod::ClosestToCorner,
        }
    }

    /// Calculate AUC confidence interval
    fn auc_confidence_interval(&self, auc: f64, n_pos: usize, n_neg: usize) -> (f64, f64) {
        // Hanley-McNeil approximation
        let q1 = auc / (2.0 - auc);
        let q2 = 2.0 * auc * auc / (1.0 + auc);

        let se = ((auc * (1.0 - auc) + (n_pos as f64 - 1.0) * (q1 - auc * auc)
            + (n_neg as f64 - 1.0) * (q2 - auc * auc))
            / (n_pos as f64 * n_neg as f64))
            .sqrt();

        let z = 1.96; // 95% CI
        let lower = (auc - z * se).max(0.0);
        let upper = (auc + z * se).min(1.0);

        (lower, upper)
    }

    /// Get sensitivity at specific specificity
    pub fn sensitivity_at_specificity(&self, curve: &RocCurve, target_specificity: f64) -> f64 {
        let target_fpr = 1.0 - target_specificity;

        // Find points bracketing target FPR
        for i in 1..curve.points.len() {
            if curve.points[i].false_positive_rate >= target_fpr {
                // Linear interpolation
                let fpr1 = curve.points[i - 1].false_positive_rate;
                let fpr2 = curve.points[i].false_positive_rate;
                let tpr1 = curve.points[i - 1].true_positive_rate;
                let tpr2 = curve.points[i].true_positive_rate;

                if (fpr2 - fpr1).abs() < 1e-10 {
                    return tpr1;
                }

                let t = (target_fpr - fpr1) / (fpr2 - fpr1);
                return tpr1 + t * (tpr2 - tpr1);
            }
        }

        curve.points.last().map(|p| p.true_positive_rate).unwrap_or(0.0)
    }

    /// Get specificity at specific sensitivity
    pub fn specificity_at_sensitivity(&self, curve: &RocCurve, target_sensitivity: f64) -> f64 {
        // Find points bracketing target TPR
        for i in 1..curve.points.len() {
            if curve.points[i].true_positive_rate >= target_sensitivity {
                // Linear interpolation
                let tpr1 = curve.points[i - 1].true_positive_rate;
                let tpr2 = curve.points[i].true_positive_rate;
                let fpr1 = curve.points[i - 1].false_positive_rate;
                let fpr2 = curve.points[i].false_positive_rate;

                if (tpr2 - tpr1).abs() < 1e-10 {
                    return 1.0 - fpr1;
                }

                let t = (target_sensitivity - tpr1) / (tpr2 - tpr1);
                let fpr = fpr1 + t * (fpr2 - fpr1);
                return 1.0 - fpr;
            }
        }

        0.0
    }

    /// Interpret AUC value
    pub fn interpret_auc(auc: f64) -> AucInterpretation {
        if auc < 0.6 {
            AucInterpretation::Failed
        } else if auc < 0.7 {
            AucInterpretation::Poor
        } else if auc < 0.8 {
            AucInterpretation::Fair
        } else if auc < 0.9 {
            AucInterpretation::Good
        } else {
            AucInterpretation::Excellent
        }
    }

    /// Compare two ROC curves (DeLong test approximation)
    pub fn compare_curves(&self, curve1: &RocCurve, curve2: &RocCurve) -> RocComparison {
        let diff = curve1.auc - curve2.auc;

        // Approximate SE for difference
        let se1 = (curve1.auc_upper - curve1.auc_lower) / (2.0 * 1.96);
        let se2 = (curve2.auc_upper - curve2.auc_lower) / (2.0 * 1.96);
        let se_diff = (se1 * se1 + se2 * se2).sqrt();

        let z = if se_diff > 0.0 { diff / se_diff } else { 0.0 };

        // Two-tailed p-value
        let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));

        RocComparison {
            auc_difference: diff,
            z_statistic: z,
            p_value,
            significant: p_value < 0.05,
        }
    }
}

/// ROC curve result
#[derive(Debug, Clone, Default)]
pub struct RocCurve {
    /// ROC curve points
    pub points: Vec<RocPoint>,
    /// Area under curve
    pub auc: f64,
    /// Lower bound of AUC 95% CI
    pub auc_lower: f64,
    /// Upper bound of AUC 95% CI
    pub auc_upper: f64,
    /// Optimal cutoff by Youden's index
    pub optimal_youden: OptimalCutoff,
    /// Optimal cutoff closest to (0,1)
    pub optimal_closest_to_01: OptimalCutoff,
    /// Number of positive cases
    pub n_positive: usize,
    /// Number of negative cases
    pub n_negative: usize,
}

/// Single point on ROC curve
#[derive(Debug, Clone, Default)]
pub struct RocPoint {
    /// Threshold value
    pub threshold: f64,
    /// True positive rate (sensitivity)
    pub true_positive_rate: f64,
    /// False positive rate (1 - specificity)
    pub false_positive_rate: f64,
    /// Sensitivity
    pub sensitivity: f64,
    /// Specificity
    pub specificity: f64,
    /// Positive predictive value
    pub ppv: f64,
    /// Negative predictive value
    pub npv: f64,
    /// Overall accuracy
    pub accuracy: f64,
    /// F1 score
    pub f1_score: f64,
    /// Youden's index (sensitivity + specificity - 1)
    pub youden_index: f64,
}

/// Optimal cutoff result
#[derive(Debug, Clone, Default)]
pub struct OptimalCutoff {
    /// Threshold value
    pub threshold: f64,
    /// Sensitivity at this threshold
    pub sensitivity: f64,
    /// Specificity at this threshold
    pub specificity: f64,
    /// Positive predictive value
    pub ppv: f64,
    /// Negative predictive value
    pub npv: f64,
    /// Overall accuracy
    pub accuracy: f64,
    /// Youden's index
    pub youden_index: f64,
    /// Method used for selection
    pub method: CutoffMethod,
}

/// Method for optimal cutoff selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CutoffMethod {
    /// Maximize Youden's index
    #[default]
    YoudenIndex,
    /// Minimize distance to (0, 1) corner
    ClosestToCorner,
    /// Fixed sensitivity level
    FixedSensitivity,
    /// Fixed specificity level
    FixedSpecificity,
}

/// AUC interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AucInterpretation {
    /// AUC < 0.6
    Failed,
    /// AUC 0.6-0.7
    Poor,
    /// AUC 0.7-0.8
    Fair,
    /// AUC 0.8-0.9
    Good,
    /// AUC > 0.9
    Excellent,
}

/// ROC curve comparison result
#[derive(Debug, Clone)]
pub struct RocComparison {
    /// Difference in AUC
    pub auc_difference: f64,
    /// Z statistic
    pub z_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether difference is significant
    pub significant: bool,
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
    fn test_perfect_roc() {
        let analyzer = RocAnalyzer::default();

        // Perfect separation
        let scores = vec![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let labels = vec![false, false, false, false, true, true, true, true];

        let curve = analyzer.analyze(&scores, &labels);

        assert!(curve.auc > 0.99);
    }

    #[test]
    fn test_random_roc() {
        let analyzer = RocAnalyzer::default();

        // Random (no discrimination)
        let scores = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let labels = vec![false, true, false, true, false, true, false, true];

        let curve = analyzer.analyze(&scores, &labels);

        // A score carrying no information has AUC exactly 0.5.
        assert!(
            (curve.auc - 0.5).abs() < 1e-9,
            "tied scores must give AUC 0.5, got {}",
            curve.auc
        );

        // ...and that must not depend on how the ties happen to be ordered.
        let reversed_labels: Vec<bool> = labels.iter().rev().cloned().collect();
        let reversed = analyzer.analyze(&scores, &reversed_labels);
        assert!(
            (reversed.auc - curve.auc).abs() < 1e-9,
            "AUC changed with tie order: {} vs {}",
            curve.auc,
            reversed.auc
        );
    }

    #[test]
    fn test_optimal_cutoff() {
        let analyzer = RocAnalyzer::default();

        let scores = vec![0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let labels = vec![false, false, false, true, true, true, true, true];

        let curve = analyzer.analyze(&scores, &labels);

        // Optimal should be around 0.45-0.55
        assert!(curve.optimal_youden.threshold >= 0.4 && curve.optimal_youden.threshold <= 0.6);
    }

    #[test]
    fn test_auc_interpretation() {
        assert_eq!(RocAnalyzer::interpret_auc(0.55), AucInterpretation::Failed);
        assert_eq!(RocAnalyzer::interpret_auc(0.65), AucInterpretation::Poor);
        assert_eq!(RocAnalyzer::interpret_auc(0.75), AucInterpretation::Fair);
        assert_eq!(RocAnalyzer::interpret_auc(0.85), AucInterpretation::Good);
        assert_eq!(RocAnalyzer::interpret_auc(0.95), AucInterpretation::Excellent);
    }

}
