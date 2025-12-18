//! Practice effects and serial testing corrections.
//!
//! This module provides tools for correcting practice effects in
//! longitudinal cognitive and behavioral assessments.

use crate::{ClinicalError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Practice effect corrector for serial assessments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PracticeEffectCorrector {
    /// Corrector name/ID.
    pub name: String,
    /// Practice effect estimates by measure.
    effects: HashMap<String, PracticeEffect>,
    /// Correction method.
    pub method: CorrectionMethod,
}

impl PracticeEffectCorrector {
    /// Create a new corrector.
    pub fn new(name: &str, method: CorrectionMethod) -> Self {
        Self {
            name: name.to_string(),
            effects: HashMap::new(),
            method,
        }
    }

    /// Add practice effect for a measure.
    pub fn add_effect(&mut self, measure: &str, effect: PracticeEffect) {
        self.effects.insert(measure.to_string(), effect);
    }

    /// Get practice effect for a measure.
    pub fn get_effect(&self, measure: &str) -> Option<&PracticeEffect> {
        self.effects.get(measure)
    }

    /// Correct a score for practice effects.
    pub fn correct_score(
        &self,
        measure: &str,
        raw_score: f64,
        assessment_number: usize,
        test_retest_interval: Option<f64>,
    ) -> Result<CorrectedScore> {
        let effect = self.effects.get(measure)
            .ok_or_else(|| ClinicalError::MissingNormativeData(format!("No practice effect data for {}", measure)))?;

        let correction = effect.calculate_correction(assessment_number, test_retest_interval);
        let corrected_score = raw_score - correction;

        Ok(CorrectedScore {
            measure: measure.to_string(),
            raw_score,
            corrected_score,
            correction_applied: correction,
            assessment_number,
            method: self.method,
        })
    }

    /// Correct multiple scores.
    pub fn correct_scores(
        &self,
        scores: &[(String, f64)],
        assessment_number: usize,
        test_retest_interval: Option<f64>,
    ) -> Vec<Result<CorrectedScore>> {
        scores.iter()
            .map(|(measure, score)| {
                self.correct_score(measure, *score, assessment_number, test_retest_interval)
            })
            .collect()
    }

    /// List all measures with practice effects.
    pub fn measures(&self) -> Vec<&String> {
        self.effects.keys().collect()
    }
}

/// Practice effect parameters for a measure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PracticeEffect {
    /// Measure name.
    pub measure: String,
    /// Expected gain from first to second assessment.
    pub first_retest_gain: f64,
    /// Expected gain from second to third assessment.
    pub second_retest_gain: f64,
    /// Asymptotic gain (maximum cumulative practice effect).
    pub asymptotic_gain: f64,
    /// Decay rate (effect diminishes over longer intervals).
    pub decay_rate: f64,
    /// Standard error of the practice effect.
    pub standard_error: Option<f64>,
    /// Whether effect is additive or multiplicative.
    pub effect_type: EffectType,
}

impl PracticeEffect {
    /// Create a new practice effect.
    pub fn new(measure: &str, first_gain: f64) -> Self {
        Self {
            measure: measure.to_string(),
            first_retest_gain: first_gain,
            second_retest_gain: first_gain * 0.5, // Typical diminishing returns
            asymptotic_gain: first_gain * 1.5,
            decay_rate: 0.0,
            standard_error: None,
            effect_type: EffectType::Additive,
        }
    }

    /// Set second retest gain.
    pub fn with_second_gain(mut self, gain: f64) -> Self {
        self.second_retest_gain = gain;
        self
    }

    /// Set asymptotic gain.
    pub fn with_asymptote(mut self, asymptote: f64) -> Self {
        self.asymptotic_gain = asymptote;
        self
    }

    /// Set decay rate.
    pub fn with_decay(mut self, rate: f64) -> Self {
        self.decay_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set standard error.
    pub fn with_se(mut self, se: f64) -> Self {
        self.standard_error = Some(se);
        self
    }

    /// Set effect type.
    pub fn with_type(mut self, effect_type: EffectType) -> Self {
        self.effect_type = effect_type;
        self
    }

    /// Calculate correction for a given assessment number.
    pub fn calculate_correction(&self, assessment_number: usize, interval_days: Option<f64>) -> f64 {
        if assessment_number <= 1 {
            return 0.0; // No correction for first assessment
        }

        // Base cumulative effect using exponential approach to asymptote
        let n = assessment_number as f64 - 1.0;
        let base_effect = self.asymptotic_gain * (1.0 - (-n / 2.0).exp());

        // Apply decay if interval provided
        let decay_factor = if let Some(days) = interval_days {
            if self.decay_rate > 0.0 {
                (-self.decay_rate * days / 365.0).exp()
            } else {
                1.0
            }
        } else {
            1.0
        };

        base_effect * decay_factor
    }

    /// Get expected score at assessment N given baseline.
    pub fn expected_score(&self, baseline: f64, assessment_number: usize) -> f64 {
        let correction = self.calculate_correction(assessment_number, None);
        match self.effect_type {
            EffectType::Additive => baseline + correction,
            EffectType::Multiplicative => baseline * (1.0 + correction / 100.0),
        }
    }
}

/// Type of practice effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectType {
    /// Effect is added to score.
    Additive,
    /// Effect is a percentage multiplier.
    Multiplicative,
}

/// Correction method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionMethod {
    /// Simple subtraction of expected practice effect.
    SimpleSubtraction,
    /// Regression-based correction.
    RegressionBased,
    /// Standardized regression-based (SRB).
    StandardizedRegressionBased,
    /// Reliable change with practice effect.
    ReliableChangeWithPractice,
}

/// Result of score correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectedScore {
    /// Measure name.
    pub measure: String,
    /// Original raw score.
    pub raw_score: f64,
    /// Practice-corrected score.
    pub corrected_score: f64,
    /// Amount of correction applied.
    pub correction_applied: f64,
    /// Assessment number.
    pub assessment_number: usize,
    /// Correction method used.
    pub method: CorrectionMethod,
}

impl CorrectedScore {
    /// Get the difference between raw and corrected.
    pub fn correction_magnitude(&self) -> f64 {
        self.correction_applied.abs()
    }

    /// Whether a significant correction was applied.
    pub fn was_significant_correction(&self, threshold: f64) -> bool {
        self.correction_applied.abs() >= threshold
    }
}

/// Serial assessment tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerialAssessment {
    /// Participant ID.
    pub participant_id: String,
    /// Assessment sessions.
    sessions: Vec<AssessmentSession>,
}

impl SerialAssessment {
    /// Create new serial assessment.
    pub fn new(participant_id: &str) -> Self {
        Self {
            participant_id: participant_id.to_string(),
            sessions: Vec::new(),
        }
    }

    /// Add an assessment session.
    pub fn add_session(&mut self, session: AssessmentSession) {
        self.sessions.push(session);
        // Sort by date/time
        self.sessions.sort_by(|a, b| a.session_number.cmp(&b.session_number));
    }

    /// Get session count.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Get a specific session.
    pub fn get_session(&self, index: usize) -> Option<&AssessmentSession> {
        self.sessions.get(index)
    }

    /// Get all sessions.
    pub fn sessions(&self) -> &[AssessmentSession] {
        &self.sessions
    }

    /// Get baseline session.
    pub fn baseline(&self) -> Option<&AssessmentSession> {
        self.sessions.first()
    }

    /// Get latest session.
    pub fn latest(&self) -> Option<&AssessmentSession> {
        self.sessions.last()
    }

    /// Calculate change from baseline for a measure.
    pub fn change_from_baseline(&self, measure: &str) -> Option<f64> {
        let baseline = self.baseline()?.get_score(measure)?;
        let latest = self.latest()?.get_score(measure)?;
        Some(latest - baseline)
    }

    /// Get trajectory for a measure.
    pub fn trajectory(&self, measure: &str) -> Vec<(usize, f64)> {
        self.sessions.iter()
            .filter_map(|s| {
                s.get_score(measure).map(|score| (s.session_number, score))
            })
            .collect()
    }

    /// Apply practice effect corrections.
    pub fn apply_corrections(
        &self,
        corrector: &PracticeEffectCorrector,
        measure: &str,
    ) -> Vec<CorrectedScore> {
        self.sessions.iter()
            .filter_map(|session| {
                let raw = session.get_score(measure)?;
                let interval = self.days_since_baseline(session.session_number);
                corrector.correct_score(measure, raw, session.session_number, interval).ok()
            })
            .collect()
    }

    /// Calculate days since baseline for a session.
    fn days_since_baseline(&self, session_number: usize) -> Option<f64> {
        if session_number <= 1 {
            return Some(0.0);
        }

        let baseline = self.baseline()?;
        let session = self.sessions.iter().find(|s| s.session_number == session_number)?;

        // If dates available, calculate actual interval
        if let (Some(b_date), Some(s_date)) = (&baseline.date, &session.date) {
            // Simplified: assume dates are in days format for this example
            // In practice, would parse actual dates
            return None;
        }

        // Default: estimate based on typical intervals
        Some((session_number - 1) as f64 * 90.0) // Assume ~90 days between sessions
    }
}

/// A single assessment session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentSession {
    /// Session number (1 = baseline).
    pub session_number: usize,
    /// Session label.
    pub label: String,
    /// Date of session.
    pub date: Option<String>,
    /// Scores by measure.
    scores: HashMap<String, f64>,
    /// Notes.
    pub notes: Option<String>,
}

impl AssessmentSession {
    /// Create new session.
    pub fn new(session_number: usize, label: &str) -> Self {
        Self {
            session_number,
            label: label.to_string(),
            date: None,
            scores: HashMap::new(),
            notes: None,
        }
    }

    /// Set date.
    pub fn with_date(mut self, date: &str) -> Self {
        self.date = Some(date.to_string());
        self
    }

    /// Add a score.
    pub fn add_score(&mut self, measure: &str, score: f64) {
        self.scores.insert(measure.to_string(), score);
    }

    /// Get a score.
    pub fn get_score(&self, measure: &str) -> Option<f64> {
        self.scores.get(measure).copied()
    }

    /// Get all measures.
    pub fn measures(&self) -> Vec<&String> {
        self.scores.keys().collect()
    }

    /// Is this the baseline session?
    pub fn is_baseline(&self) -> bool {
        self.session_number == 1
    }
}

/// Standardized Regression-Based (SRB) change calculator.
#[derive(Debug, Clone)]
pub struct SRBCalculator {
    /// Regression coefficients by measure (intercept, slope).
    coefficients: HashMap<String, (f64, f64)>,
    /// Standard error of estimate by measure.
    see: HashMap<String, f64>,
    /// Confidence level for determining reliable change.
    confidence: f64,
}

impl SRBCalculator {
    /// Create new SRB calculator.
    pub fn new(confidence: f64) -> Self {
        Self {
            coefficients: HashMap::new(),
            see: HashMap::new(),
            confidence: confidence.clamp(0.80, 0.99),
        }
    }

    /// Add regression parameters for a measure.
    pub fn add_measure(&mut self, measure: &str, intercept: f64, slope: f64, see: f64) {
        self.coefficients.insert(measure.to_string(), (intercept, slope));
        self.see.insert(measure.to_string(), see);
    }

    /// Calculate predicted retest score.
    pub fn predict(&self, measure: &str, baseline: f64) -> Option<f64> {
        let (intercept, slope) = self.coefficients.get(measure)?;
        Some(intercept + slope * baseline)
    }

    /// Calculate SRB z-score.
    pub fn srb_z(&self, measure: &str, baseline: f64, observed: f64) -> Option<f64> {
        let predicted = self.predict(measure, baseline)?;
        let see = self.see.get(measure)?;
        if *see < 1e-10 {
            return None;
        }
        Some((observed - predicted) / see)
    }

    /// Classify change.
    pub fn classify_change(
        &self,
        measure: &str,
        baseline: f64,
        observed: f64,
    ) -> Option<SRBClassification> {
        let z = self.srb_z(measure, baseline, observed)?;
        let critical_z = match self.confidence {
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.28,
        };

        Some(if z >= critical_z {
            SRBClassification::ReliableImprovement
        } else if z <= -critical_z {
            SRBClassification::ReliableDecline
        } else {
            SRBClassification::NoChange
        })
    }
}

/// SRB classification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SRBClassification {
    /// Reliable improvement.
    ReliableImprovement,
    /// No reliable change.
    NoChange,
    /// Reliable decline.
    ReliableDecline,
}

/// Multi-measure practice effect norms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PracticeEffectNorms {
    /// Norms name.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// Effects by measure.
    effects: HashMap<String, PracticeEffect>,
    /// Population description.
    pub population: String,
    /// Sample size.
    pub sample_size: usize,
}

impl PracticeEffectNorms {
    /// Create new norms.
    pub fn new(name: &str, population: &str, sample_size: usize) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            effects: HashMap::new(),
            population: population.to_string(),
            sample_size,
        }
    }

    /// Add effect.
    pub fn add_effect(&mut self, effect: PracticeEffect) {
        self.effects.insert(effect.measure.clone(), effect);
    }

    /// Get effect.
    pub fn get_effect(&self, measure: &str) -> Option<&PracticeEffect> {
        self.effects.get(measure)
    }

    /// Create corrector from these norms.
    pub fn to_corrector(&self, method: CorrectionMethod) -> PracticeEffectCorrector {
        let mut corrector = PracticeEffectCorrector::new(&self.name, method);
        for (measure, effect) in &self.effects {
            corrector.add_effect(measure, effect.clone());
        }
        corrector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_practice_effect_calculation() {
        let effect = PracticeEffect::new("memory", 5.0)
            .with_second_gain(2.5)
            .with_asymptote(8.0);

        // First assessment: no correction
        assert!((effect.calculate_correction(1, None) - 0.0).abs() < 0.001);

        // Second assessment: significant correction
        let correction2 = effect.calculate_correction(2, None);
        assert!(correction2 > 0.0 && correction2 < 8.0);

        // Third assessment: larger correction
        let correction3 = effect.calculate_correction(3, None);
        assert!(correction3 > correction2);

        // Approaches asymptote
        let correction10 = effect.calculate_correction(10, None);
        assert!((correction10 - 8.0).abs() < 1.0);
    }

    #[test]
    fn test_decay_effect() {
        let effect = PracticeEffect::new("attention", 5.0)
            .with_decay(0.5);

        let short_interval = effect.calculate_correction(2, Some(30.0));
        let long_interval = effect.calculate_correction(2, Some(365.0));

        // Longer interval should have smaller effect due to decay
        assert!(long_interval < short_interval);
    }

    #[test]
    fn test_corrector() {
        let mut corrector = PracticeEffectCorrector::new("Test", CorrectionMethod::SimpleSubtraction);
        corrector.add_effect("memory", PracticeEffect::new("memory", 5.0));

        let result = corrector.correct_score("memory", 100.0, 2, None).unwrap();

        assert_eq!(result.raw_score, 100.0);
        assert!(result.corrected_score < 100.0);
        assert!(result.correction_applied > 0.0);
    }

    #[test]
    fn test_serial_assessment() {
        let mut serial = SerialAssessment::new("P001");

        let mut session1 = AssessmentSession::new(1, "Baseline");
        session1.add_score("memory", 50.0);
        serial.add_session(session1);

        let mut session2 = AssessmentSession::new(2, "Week 12");
        session2.add_score("memory", 55.0);
        serial.add_session(session2);

        assert_eq!(serial.session_count(), 2);

        let change = serial.change_from_baseline("memory").unwrap();
        assert!((change - 5.0).abs() < 0.001);

        let trajectory = serial.trajectory("memory");
        assert_eq!(trajectory.len(), 2);
        assert_eq!(trajectory[0], (1, 50.0));
        assert_eq!(trajectory[1], (2, 55.0));
    }

    #[test]
    fn test_srb_calculator() {
        let mut srb = SRBCalculator::new(0.90);
        // Typical regression: predicted = 10 + 0.8 * baseline
        srb.add_measure("memory", 10.0, 0.8, 5.0);

        let predicted = srb.predict("memory", 50.0).unwrap();
        assert!((predicted - 50.0).abs() < 0.001); // 10 + 0.8*50 = 50

        // Observed score much higher than predicted
        let z = srb.srb_z("memory", 50.0, 65.0).unwrap();
        assert!((z - 3.0).abs() < 0.001); // (65-50)/5 = 3.0

        let classification = srb.classify_change("memory", 50.0, 65.0).unwrap();
        assert_eq!(classification, SRBClassification::ReliableImprovement);
    }

    #[test]
    fn test_expected_score() {
        let effect = PracticeEffect::new("memory", 5.0)
            .with_asymptote(8.0);

        let baseline = 50.0;
        let expected2 = effect.expected_score(baseline, 2);
        let expected3 = effect.expected_score(baseline, 3);

        assert!(expected2 > baseline);
        assert!(expected3 > expected2);
    }

    #[test]
    fn test_norms_to_corrector() {
        let mut norms = PracticeEffectNorms::new("Standard Norms", "Healthy Adults", 500);
        norms.add_effect(PracticeEffect::new("memory", 5.0));
        norms.add_effect(PracticeEffect::new("attention", 3.0));

        let corrector = norms.to_corrector(CorrectionMethod::SimpleSubtraction);
        assert_eq!(corrector.measures().len(), 2);
    }

    #[test]
    fn test_multiplicative_effect() {
        let effect = PracticeEffect::new("reaction_time", 5.0)
            .with_type(EffectType::Multiplicative);

        let baseline = 500.0; // ms
        let expected = effect.expected_score(baseline, 2);

        // Multiplicative: baseline * (1 + correction/100)
        assert!(expected > baseline);
    }

    #[test]
    fn test_correction_with_multiple_measures() {
        let mut corrector = PracticeEffectCorrector::new("Multi", CorrectionMethod::SimpleSubtraction);
        corrector.add_effect("memory", PracticeEffect::new("memory", 5.0));
        corrector.add_effect("attention", PracticeEffect::new("attention", 3.0));

        let scores = vec![
            ("memory".to_string(), 60.0),
            ("attention".to_string(), 45.0),
        ];

        let results = corrector.correct_scores(&scores, 2, None);
        assert_eq!(results.len(), 2);

        for result in results {
            assert!(result.is_ok());
        }
    }
}
