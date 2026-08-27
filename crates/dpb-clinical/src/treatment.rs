//! Treatment response and intervention modeling.
//!
//! This module provides tools for modeling pre/post intervention outcomes,
//! calculating effect sizes, and tracking treatment response over time.

use crate::{ClinicalError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Treatment response tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreatmentResponse {
    /// Participant/patient ID.
    pub participant_id: String,
    /// Treatment/intervention name.
    pub treatment: String,
    /// Baseline assessments.
    pub baseline: Assessment,
    /// Follow-up assessments.
    pub followups: Vec<Assessment>,
    /// Response classification.
    pub classification: Option<ResponseClassification>,
}

impl TreatmentResponse {
    /// Create new treatment response tracker.
    pub fn new(participant_id: &str, treatment: &str, baseline: Assessment) -> Self {
        Self {
            participant_id: participant_id.to_string(),
            treatment: treatment.to_string(),
            baseline,
            followups: Vec::new(),
            classification: None,
        }
    }

    /// Add follow-up assessment.
    pub fn add_followup(&mut self, assessment: Assessment) {
        self.followups.push(assessment);
    }

    /// Get latest followup.
    pub fn latest_followup(&self) -> Option<&Assessment> {
        self.followups.last()
    }

    /// Calculate change from baseline for a measure.
    pub fn change_from_baseline(&self, measure: &str) -> Option<f64> {
        let baseline_val = self.baseline.get_value(measure)?;
        let followup_val = self.latest_followup()?.get_value(measure)?;
        Some(followup_val - baseline_val)
    }

    /// Calculate percent change from baseline.
    pub fn percent_change(&self, measure: &str) -> Option<f64> {
        let baseline_val = self.baseline.get_value(measure)?;
        if baseline_val.abs() < 1e-10 {
            return None;
        }
        let change = self.change_from_baseline(measure)?;
        Some(change / baseline_val * 100.0)
    }

    /// Calculate effect size (Cohen's d) for a measure.
    pub fn effect_size(&self, measure: &str, pooled_sd: f64) -> Option<f64> {
        let change = self.change_from_baseline(measure)?;
        Some(change / pooled_sd)
    }

    /// Classify response based on criteria.
    pub fn classify_response(&mut self, criteria: &ResponseCriteria) -> ResponseClassification {
        let classification =
            if let Some(raw_change) = self.change_from_baseline(&criteria.primary_measure) {
                // `change_from_baseline` is followup - baseline, so its sign depends on
                // the measure's direction. Normalise to a magnitude where positive
                // always means "got better", per criteria.higher_is_better. Without
                // this every lower-is-better measure (depression, UPDRS, TUG, Trail
                // Making, reaction time) classifies backwards.
                let improvement = if criteria.higher_is_better {
                    raw_change
                } else {
                    -raw_change
                };

                // Compared by magnitude so a caller may pass either a signed threshold
                // matching their direction (e.g. -12.5 for a lower-is-better measure)
                // or a plain magnitude (12.5). Both mean the same thing.
                let required = criteria.response_threshold.abs();

                if improvement >= required {
                    let reached_remission = criteria.remission_threshold.and_then(|remission| {
                        let val = self
                            .latest_followup()?
                            .get_value(&criteria.primary_measure)?;
                        // Remission is an absolute cut-off, so it is crossed in the
                        // direction the measure improves.
                        Some(if criteria.higher_is_better {
                            val >= remission
                        } else {
                            val <= remission
                        })
                    });
                    match reached_remission {
                        Some(true) => ResponseClassification::Remission,
                        _ => ResponseClassification::Response,
                    }
                } else if improvement > 0.0 {
                    ResponseClassification::PartialResponse
                } else {
                    ResponseClassification::NoResponse
                }
            } else {
                ResponseClassification::Unknown
            };

        self.classification = Some(classification);
        classification
    }

    /// Get trajectory for a measure across all timepoints.
    pub fn trajectory(&self, measure: &str) -> Vec<(f64, f64)> {
        let mut points = Vec::new();

        if let Some(val) = self.baseline.get_value(measure) {
            points.push((self.baseline.time_point, val));
        }

        for followup in &self.followups {
            if let Some(val) = followup.get_value(measure) {
                points.push((followup.time_point, val));
            }
        }

        points
    }
}

/// Assessment at a single timepoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assessment {
    /// Time point (e.g., weeks from baseline).
    pub time_point: f64,
    /// Assessment label (e.g., "Week 0", "Week 12").
    pub label: String,
    /// Measure values.
    values: HashMap<String, f64>,
    /// Assessment date (optional).
    pub date: Option<String>,
    /// Notes.
    pub notes: Option<String>,
}

impl Assessment {
    /// Create new assessment.
    pub fn new(time_point: f64, label: &str) -> Self {
        Self {
            time_point,
            label: label.to_string(),
            values: HashMap::new(),
            date: None,
            notes: None,
        }
    }

    /// Add a measure value.
    pub fn add_value(&mut self, measure: &str, value: f64) {
        self.values.insert(measure.to_string(), value);
    }

    /// Get a measure value.
    pub fn get_value(&self, measure: &str) -> Option<f64> {
        self.values.get(measure).copied()
    }

    /// Get all measures.
    pub fn measures(&self) -> Vec<&String> {
        self.values.keys().collect()
    }

    /// Set date.
    pub fn with_date(mut self, date: &str) -> Self {
        self.date = Some(date.to_string());
        self
    }

    /// Set notes.
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = Some(notes.to_string());
        self
    }
}

/// Response classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseClassification {
    /// Full remission.
    Remission,
    /// Response (met threshold).
    Response,
    /// Partial response.
    PartialResponse,
    /// No response.
    NoResponse,
    /// Deterioration.
    Deterioration,
    /// Unknown/Unable to classify.
    Unknown,
}

impl ResponseClassification {
    /// Whether this is a positive response.
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            ResponseClassification::Remission | ResponseClassification::Response
        )
    }
}

/// Criteria for response classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCriteria {
    /// Primary outcome measure.
    pub primary_measure: String,
    /// Threshold for response (change from baseline).
    pub response_threshold: f64,
    /// Threshold for remission (absolute value).
    pub remission_threshold: Option<f64>,
    /// Whether higher values are better.
    pub higher_is_better: bool,
}

impl ResponseCriteria {
    /// Create new response criteria.
    pub fn new(measure: &str, threshold: f64) -> Self {
        Self {
            primary_measure: measure.to_string(),
            response_threshold: threshold,
            remission_threshold: None,
            higher_is_better: true,
        }
    }

    /// Set remission threshold.
    pub fn with_remission(mut self, threshold: f64) -> Self {
        self.remission_threshold = Some(threshold);
        self
    }

    /// Set direction.
    pub fn with_direction(mut self, higher_is_better: bool) -> Self {
        self.higher_is_better = higher_is_better;
        self
    }
}

/// Intervention model for group comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionModel {
    /// Model name.
    pub name: String,
    /// Treatment groups.
    groups: HashMap<String, TreatmentGroup>,
    /// Primary outcome.
    pub primary_outcome: String,
    /// Secondary outcomes.
    pub secondary_outcomes: Vec<String>,
}

impl InterventionModel {
    /// Create new intervention model.
    pub fn new(name: &str, primary_outcome: &str) -> Self {
        Self {
            name: name.to_string(),
            groups: HashMap::new(),
            primary_outcome: primary_outcome.to_string(),
            secondary_outcomes: Vec::new(),
        }
    }

    /// Add secondary outcome.
    pub fn add_secondary_outcome(&mut self, outcome: &str) {
        self.secondary_outcomes.push(outcome.to_string());
    }

    /// Add a treatment group.
    pub fn add_group(&mut self, name: &str, group: TreatmentGroup) {
        self.groups.insert(name.to_string(), group);
    }

    /// Get a group.
    pub fn get_group(&self, name: &str) -> Option<&TreatmentGroup> {
        self.groups.get(name)
    }

    /// Calculate between-group effect size.
    pub fn between_group_effect_size(
        &self,
        group1: &str,
        group2: &str,
        measure: &str,
    ) -> Result<EffectSize> {
        let g1 = self.groups.get(group1).ok_or_else(|| {
            ClinicalError::InvalidConfiguration(format!("Group {} not found", group1))
        })?;
        let g2 = self.groups.get(group2).ok_or_else(|| {
            ClinicalError::InvalidConfiguration(format!("Group {} not found", group2))
        })?;

        let stats1 = g1.get_stats(measure).ok_or_else(|| {
            ClinicalError::InvalidConfiguration(format!("No stats for {} in {}", measure, group1))
        })?;
        let stats2 = g2.get_stats(measure).ok_or_else(|| {
            ClinicalError::InvalidConfiguration(format!("No stats for {} in {}", measure, group2))
        })?;

        EffectSize::cohens_d(
            stats1.mean,
            stats2.mean,
            stats1.sd,
            stats2.sd,
            stats1.n,
            stats2.n,
        )
    }

    /// Calculate within-group effect sizes for all groups.
    pub fn within_group_effect_sizes(&self, measure: &str) -> HashMap<String, Result<EffectSize>> {
        self.groups
            .iter()
            .map(|(name, group)| {
                let effect = group.within_group_effect_size(measure);
                (name.clone(), effect)
            })
            .collect()
    }

    /// Get response rates for all groups.
    pub fn response_rates(&self) -> HashMap<String, f64> {
        self.groups
            .iter()
            .map(|(name, group)| (name.clone(), group.response_rate()))
            .collect()
    }

    /// Calculate number needed to treat (NNT).
    pub fn nnt(&self, treatment: &str, control: &str) -> Result<f64> {
        let treatment_group = self.groups.get(treatment).ok_or_else(|| {
            ClinicalError::InvalidConfiguration("Treatment group not found".to_string())
        })?;
        let control_group = self.groups.get(control).ok_or_else(|| {
            ClinicalError::InvalidConfiguration("Control group not found".to_string())
        })?;

        let treatment_rate = treatment_group.response_rate();
        let control_rate = control_group.response_rate();

        let ard = treatment_rate - control_rate; // Absolute risk difference

        if ard.abs() < 1e-10 {
            return Err(ClinicalError::InvalidConfiguration(
                "No difference in response rates".to_string(),
            ));
        }

        Ok(1.0 / ard.abs())
    }
}

/// A treatment group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreatmentGroup {
    /// Group name.
    pub name: String,
    /// Treatment description.
    pub treatment: String,
    /// Number of participants.
    pub n: usize,
    /// Summary statistics by measure.
    stats: HashMap<String, GroupStats>,
    /// Individual responses.
    responses: Vec<TreatmentResponse>,
}

impl TreatmentGroup {
    /// Create new treatment group.
    pub fn new(name: &str, treatment: &str) -> Self {
        Self {
            name: name.to_string(),
            treatment: treatment.to_string(),
            n: 0,
            stats: HashMap::new(),
            responses: Vec::new(),
        }
    }

    /// Add summary statistics for a measure.
    pub fn add_stats(&mut self, measure: &str, stats: GroupStats) {
        self.stats.insert(measure.to_string(), stats);
    }

    /// Get stats for a measure.
    pub fn get_stats(&self, measure: &str) -> Option<&GroupStats> {
        self.stats.get(measure)
    }

    /// Add individual response.
    pub fn add_response(&mut self, response: TreatmentResponse) {
        self.n += 1;
        self.responses.push(response);
    }

    /// Calculate response rate.
    pub fn response_rate(&self) -> f64 {
        if self.responses.is_empty() {
            return 0.0;
        }

        let responders = self
            .responses
            .iter()
            .filter(|r| r.classification.map(|c| c.is_positive()).unwrap_or(false))
            .count();

        responders as f64 / self.responses.len() as f64
    }

    /// Calculate within-group effect size.
    pub fn within_group_effect_size(&self, measure: &str) -> Result<EffectSize> {
        let stats = self.get_stats(measure).ok_or_else(|| {
            ClinicalError::InvalidConfiguration(format!("No stats for {}", measure))
        })?;

        Ok(EffectSize {
            value: stats.effect_size,
            effect_type: EffectType::CohensD,
            ci_lower: None,
            ci_upper: None,
            interpretation: interpret_effect_size(stats.effect_size),
        })
    }
}

/// Group summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupStats {
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub sd: f64,
    /// Sample size.
    pub n: usize,
    /// Mean change from baseline.
    pub mean_change: Option<f64>,
    /// SD of change.
    pub sd_change: Option<f64>,
    /// Effect size (within-group).
    pub effect_size: f64,
}

impl GroupStats {
    /// Create new group stats.
    pub fn new(mean: f64, sd: f64, n: usize) -> Self {
        Self {
            mean,
            sd,
            n,
            mean_change: None,
            sd_change: None,
            effect_size: 0.0,
        }
    }

    /// Set change statistics.
    pub fn with_change(mut self, mean_change: f64, sd_change: f64) -> Self {
        self.mean_change = Some(mean_change);
        self.sd_change = Some(sd_change);
        if sd_change > 0.0 {
            self.effect_size = mean_change / sd_change;
        }
        self
    }
}

/// Effect size calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSize {
    /// Effect size value.
    pub value: f64,
    /// Type of effect size.
    pub effect_type: EffectType,
    /// Lower confidence interval.
    pub ci_lower: Option<f64>,
    /// Upper confidence interval.
    pub ci_upper: Option<f64>,
    /// Qualitative interpretation.
    pub interpretation: EffectInterpretation,
}

impl EffectSize {
    /// Calculate Cohen's d.
    pub fn cohens_d(
        mean1: f64,
        mean2: f64,
        sd1: f64,
        sd2: f64,
        n1: usize,
        n2: usize,
    ) -> Result<Self> {
        // Pooled standard deviation
        let pooled_sd = (((n1 - 1) as f64 * sd1.powi(2) + (n2 - 1) as f64 * sd2.powi(2))
            / (n1 + n2 - 2) as f64)
            .sqrt();

        if pooled_sd.abs() < 1e-10 {
            return Err(ClinicalError::InvalidConfiguration(
                "Cannot calculate effect size with zero SD".to_string(),
            ));
        }

        let d = (mean1 - mean2) / pooled_sd;

        // Calculate confidence interval
        let se =
            ((n1 + n2) as f64 / (n1 * n2) as f64 + d.powi(2) / (2.0 * (n1 + n2) as f64)).sqrt();
        let ci_lower = d - 1.96 * se;
        let ci_upper = d + 1.96 * se;

        Ok(Self {
            value: d,
            effect_type: EffectType::CohensD,
            ci_lower: Some(ci_lower),
            ci_upper: Some(ci_upper),
            interpretation: interpret_effect_size(d),
        })
    }

    /// Calculate Hedges' g (bias-corrected Cohen's d).
    pub fn hedges_g(
        mean1: f64,
        mean2: f64,
        sd1: f64,
        sd2: f64,
        n1: usize,
        n2: usize,
    ) -> Result<Self> {
        let mut effect = Self::cohens_d(mean1, mean2, sd1, sd2, n1, n2)?;

        // Correction factor
        let df = (n1 + n2 - 2) as f64;
        let j = 1.0 - (3.0 / (4.0 * df - 1.0));

        effect.value *= j;
        effect.effect_type = EffectType::HedgesG;
        effect.ci_lower = effect.ci_lower.map(|v| v * j);
        effect.ci_upper = effect.ci_upper.map(|v| v * j);
        effect.interpretation = interpret_effect_size(effect.value);

        Ok(effect)
    }

    /// Calculate Glass's delta.
    pub fn glass_delta(mean_treatment: f64, mean_control: f64, sd_control: f64) -> Result<Self> {
        if sd_control.abs() < 1e-10 {
            return Err(ClinicalError::InvalidConfiguration(
                "Cannot calculate effect size with zero SD".to_string(),
            ));
        }

        let delta = (mean_treatment - mean_control) / sd_control;

        Ok(Self {
            value: delta,
            effect_type: EffectType::GlassDelta,
            ci_lower: None,
            ci_upper: None,
            interpretation: interpret_effect_size(delta),
        })
    }

    /// Calculate correlation coefficient r from d.
    pub fn d_to_r(&self) -> f64 {
        let d = self.value;
        d / (d.powi(2) + 4.0).sqrt()
    }

    /// Create from correlation.
    pub fn from_correlation(r: f64) -> Self {
        Self {
            value: r,
            effect_type: EffectType::CorrelationR,
            ci_lower: None,
            ci_upper: None,
            interpretation: interpret_correlation(r),
        }
    }
}

/// Type of effect size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectType {
    /// Cohen's d.
    CohensD,
    /// Hedges' g (bias-corrected).
    HedgesG,
    /// Glass's delta.
    GlassDelta,
    /// Pearson correlation r.
    CorrelationR,
    /// Eta-squared.
    EtaSquared,
    /// Odds ratio.
    OddsRatio,
}

/// Qualitative interpretation of effect size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectInterpretation {
    /// Negligible effect.
    Negligible,
    /// Small effect.
    Small,
    /// Medium effect.
    Medium,
    /// Large effect.
    Large,
    /// Very large effect.
    VeryLarge,
}

impl std::fmt::Display for EffectInterpretation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EffectInterpretation::Negligible => write!(f, "Negligible"),
            EffectInterpretation::Small => write!(f, "Small"),
            EffectInterpretation::Medium => write!(f, "Medium"),
            EffectInterpretation::Large => write!(f, "Large"),
            EffectInterpretation::VeryLarge => write!(f, "Very Large"),
        }
    }
}

/// Interpret Cohen's d effect size.
fn interpret_effect_size(d: f64) -> EffectInterpretation {
    let d = d.abs();
    match d {
        d if d < 0.2 => EffectInterpretation::Negligible,
        d if d < 0.5 => EffectInterpretation::Small,
        d if d < 0.8 => EffectInterpretation::Medium,
        d if d < 1.2 => EffectInterpretation::Large,
        _ => EffectInterpretation::VeryLarge,
    }
}

/// Interpret correlation coefficient.
fn interpret_correlation(r: f64) -> EffectInterpretation {
    let r = r.abs();
    match r {
        r if r < 0.1 => EffectInterpretation::Negligible,
        r if r < 0.3 => EffectInterpretation::Small,
        r if r < 0.5 => EffectInterpretation::Medium,
        r if r < 0.7 => EffectInterpretation::Large,
        _ => EffectInterpretation::VeryLarge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_treatment_response() {
        let mut baseline = Assessment::new(0.0, "Baseline");
        baseline.add_value("symptom_score", 25.0);

        let mut response = TreatmentResponse::new("P001", "Drug A", baseline);

        let mut week12 = Assessment::new(12.0, "Week 12");
        week12.add_value("symptom_score", 10.0);
        response.add_followup(week12);

        let change = response.change_from_baseline("symptom_score").unwrap();
        assert!((change - (-15.0)).abs() < 0.001);

        let pct_change = response.percent_change("symptom_score").unwrap();
        assert!((pct_change - (-60.0)).abs() < 0.001);
    }

    #[test]
    fn test_effect_size_calculation() {
        // Treatment vs control
        let effect = EffectSize::cohens_d(
            110.0, // treatment mean
            100.0, // control mean
            15.0,  // treatment SD
            15.0,  // control SD
            50,    // treatment n
            50,    // control n
        )
        .unwrap();

        assert!((effect.value - 0.67).abs() < 0.1);
        assert_eq!(effect.interpretation, EffectInterpretation::Medium);
    }

    #[test]
    fn test_hedges_g() {
        let effect = EffectSize::hedges_g(110.0, 100.0, 15.0, 15.0, 10, 10).unwrap();

        // Hedges' g should be slightly smaller than Cohen's d for small samples
        let cohens = EffectSize::cohens_d(110.0, 100.0, 15.0, 15.0, 10, 10).unwrap();
        assert!(effect.value.abs() < cohens.value.abs());
    }

    /// Build a two-timepoint response for `measure`, baseline -> followup.
    fn response_for(measure: &str, baseline_val: f64, followup_val: f64) -> TreatmentResponse {
        let mut baseline = Assessment::new(0.0, "Baseline");
        baseline.add_value(measure, baseline_val);
        let mut r = TreatmentResponse::new("P001", "Therapy", baseline);
        let mut followup = Assessment::new(8.0, "Week 8");
        followup.add_value(measure, followup_val);
        r.add_followup(followup);
        r
    }

    #[test]
    fn lower_is_better_improvement_reaches_remission() {
        // Depression 25 -> 8. A 17-point drop clears the 12.5 response threshold,
        // and 8 is at or below the remission cut-off of 10.
        let mut r = response_for("depression", 25.0, 8.0);
        let criteria = ResponseCriteria::new("depression", -12.5)
            .with_remission(10.0)
            .with_direction(false);
        assert_eq!(
            r.classify_response(&criteria),
            ResponseClassification::Remission,
            "a 25->8 drop on a lower-is-better measure is remission, not failure"
        );
    }

    #[test]
    fn lower_is_better_deterioration_is_no_response() {
        // Depression got worse: 25 -> 30. Must never read as improvement.
        let mut r = response_for("depression", 25.0, 30.0);
        let criteria = ResponseCriteria::new("depression", 12.5).with_direction(false);
        assert_eq!(
            r.classify_response(&criteria),
            ResponseClassification::NoResponse
        );
    }

    #[test]
    fn lower_is_better_partial_improvement() {
        // 25 -> 20: real but below the 12.5 threshold.
        let mut r = response_for("depression", 25.0, 20.0);
        let criteria = ResponseCriteria::new("depression", 12.5).with_direction(false);
        assert_eq!(
            r.classify_response(&criteria),
            ResponseClassification::PartialResponse
        );
    }

    #[test]
    fn higher_is_better_direction_still_works() {
        // Function score 40 -> 60 on a higher-is-better measure, remission at 55.
        let mut r = response_for("function", 40.0, 60.0);
        let criteria = ResponseCriteria::new("function", 12.5)
            .with_remission(55.0)
            .with_direction(true);
        assert_eq!(
            r.classify_response(&criteria),
            ResponseClassification::Remission
        );

        // And a decline on the same measure is not a response.
        let mut worse = response_for("function", 40.0, 30.0);
        assert_eq!(
            worse.classify_response(&criteria),
            ResponseClassification::NoResponse
        );
    }

    #[test]
    fn signed_and_unsigned_thresholds_agree() {
        // -12.5 and 12.5 must mean the same thing for a lower-is-better measure.
        let signed = ResponseCriteria::new("depression", -12.5).with_direction(false);
        let unsigned = ResponseCriteria::new("depression", 12.5).with_direction(false);
        let mut a = response_for("depression", 25.0, 8.0);
        let mut b = response_for("depression", 25.0, 8.0);
        assert_eq!(a.classify_response(&signed), b.classify_response(&unsigned));
    }

    #[test]
    fn test_intervention_model() {
        let mut model = InterventionModel::new("RCT Study", "symptom_score");

        let mut treatment = TreatmentGroup::new("Treatment", "Drug A");
        treatment.add_stats("symptom_score", GroupStats::new(10.0, 5.0, 50));

        let mut control = TreatmentGroup::new("Control", "Placebo");
        control.add_stats("symptom_score", GroupStats::new(20.0, 5.0, 50));

        model.add_group("treatment", treatment);
        model.add_group("control", control);

        let effect = model
            .between_group_effect_size("treatment", "control", "symptom_score")
            .unwrap();
        assert!((effect.value.abs() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_d_to_r_conversion() {
        let effect = EffectSize::cohens_d(110.0, 100.0, 15.0, 15.0, 50, 50).unwrap();
        let r = effect.d_to_r();

        // r should be smaller than d
        assert!(r.abs() < effect.value.abs());
        assert!(r > 0.0 && r < 1.0);
    }

    #[test]
    fn test_nnt_calculation() {
        let mut model = InterventionModel::new("NNT Test", "response");

        let mut treatment = TreatmentGroup::new("Treatment", "Drug");
        let mut control = TreatmentGroup::new("Control", "Placebo");

        // Add mock responses (60% response in treatment, 30% in control)
        for i in 0..10 {
            let baseline = Assessment::new(0.0, "Baseline");
            let mut t_response =
                TreatmentResponse::new(&format!("T{}", i), "Drug", baseline.clone());
            t_response.classification = Some(if i < 6 {
                ResponseClassification::Response
            } else {
                ResponseClassification::NoResponse
            });
            treatment.add_response(t_response);

            let mut c_response = TreatmentResponse::new(&format!("C{}", i), "Placebo", baseline);
            c_response.classification = Some(if i < 3 {
                ResponseClassification::Response
            } else {
                ResponseClassification::NoResponse
            });
            control.add_response(c_response);
        }

        model.add_group("treatment", treatment);
        model.add_group("control", control);

        let nnt = model.nnt("treatment", "control").unwrap();
        // NNT = 1 / (0.6 - 0.3) = 3.33
        assert!((nnt - 3.33).abs() < 0.1);
    }
}
