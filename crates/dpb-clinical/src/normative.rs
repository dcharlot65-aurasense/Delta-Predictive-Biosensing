//! Normative databases for population-stratified reference ranges.
//!
//! This module provides infrastructure for managing population-specific
//! normative data, enabling equitable clinical assessments across diverse
//! patient populations.

use crate::{demographics::Demographics, ClinicalError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Normative database for population-stratified reference data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormativeDatabase {
    /// Database name.
    pub name: String,
    /// Database version.
    pub version: String,
    /// Description.
    pub description: Option<String>,
    /// Population norms by key.
    norms: HashMap<String, PopulationNorms>,
    /// Measures available in this database.
    measures: Vec<MeasureDefinition>,
    /// Sample sizes by population.
    sample_sizes: HashMap<String, usize>,
}

impl NormativeDatabase {
    /// Create a new normative database.
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            description: None,
            norms: HashMap::new(),
            measures: Vec::new(),
            sample_sizes: HashMap::new(),
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Add a measure definition.
    pub fn add_measure(&mut self, measure: MeasureDefinition) {
        self.measures.push(measure);
    }

    /// Add population norms.
    pub fn add_norms(&mut self, population_key: &str, norms: PopulationNorms, sample_size: usize) {
        self.norms.insert(population_key.to_string(), norms);
        self.sample_sizes.insert(population_key.to_string(), sample_size);
    }

    /// Get norms for a specific population.
    pub fn get_norms(&self, population_key: &str) -> Option<&PopulationNorms> {
        self.norms.get(population_key)
    }

    /// Get norms for demographics, with fallback to broader categories.
    pub fn get_norms_for_demographics(&self, demographics: &Demographics) -> Option<&PopulationNorms> {
        let key = demographics.population_key();

        // Try exact match first
        if let Some(norms) = self.norms.get(&key) {
            return Some(norms);
        }

        // Try fallback keys
        for fallback in self.fallback_keys(demographics) {
            if let Some(norms) = self.norms.get(&fallback) {
                return Some(norms);
            }
        }

        None
    }

    /// Generate fallback keys for demographics.
    fn fallback_keys(&self, demographics: &Demographics) -> Vec<String> {
        let mut keys = Vec::new();

        // Age group + sex (any ethnicity)
        if let (Some(age_group), Some(sex)) = (demographics.age_group(), demographics.sex) {
            keys.push(format!("{:?}_{:?}_Unknown", age_group, sex));
        }

        // Age group only
        if let Some(age_group) = demographics.age_group() {
            keys.push(format!("{:?}_Unknown_Unknown", age_group));
        }

        // General population
        keys.push("Unknown_Unknown_Unknown".to_string());

        keys
    }

    /// List all population keys.
    pub fn population_keys(&self) -> Vec<&String> {
        self.norms.keys().collect()
    }

    /// Get sample size for population.
    pub fn sample_size(&self, population_key: &str) -> Option<usize> {
        self.sample_sizes.get(population_key).copied()
    }

    /// Get all measures.
    pub fn measures(&self) -> &[MeasureDefinition] {
        &self.measures
    }

    /// Calculate z-score for a value.
    pub fn z_score(
        &self,
        measure: &str,
        value: f64,
        demographics: &Demographics,
    ) -> Result<f64> {
        let norms = self
            .get_norms_for_demographics(demographics)
            .ok_or_else(|| ClinicalError::MissingNormativeData(demographics.population_key()))?;

        let reference = norms
            .get_reference(measure)
            .ok_or_else(|| ClinicalError::MissingNormativeData(measure.to_string()))?;

        Ok(reference.z_score(value))
    }

    /// Calculate percentile for a value.
    pub fn percentile(
        &self,
        measure: &str,
        value: f64,
        demographics: &Demographics,
    ) -> Result<f64> {
        let z = self.z_score(measure, value, demographics)?;
        Ok(z_to_percentile(z))
    }

    /// Determine if value is impaired (below cutoff).
    pub fn is_impaired(
        &self,
        measure: &str,
        value: f64,
        demographics: &Demographics,
        cutoff_sd: f64,
    ) -> Result<bool> {
        let z = self.z_score(measure, value, demographics)?;
        Ok(z < -cutoff_sd)
    }
}

/// Norms for a specific population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationNorms {
    /// Population description.
    pub description: String,
    /// Reference values by measure.
    references: HashMap<String, NormativeReference>,
    /// Reliability coefficients by measure.
    reliability: HashMap<String, f64>,
}

impl PopulationNorms {
    /// Create new population norms.
    pub fn new(description: &str) -> Self {
        Self {
            description: description.to_string(),
            references: HashMap::new(),
            reliability: HashMap::new(),
        }
    }

    /// Add a reference for a measure.
    pub fn add_reference(&mut self, measure: &str, reference: NormativeReference) {
        self.references.insert(measure.to_string(), reference);
    }

    /// Add reliability coefficient.
    pub fn add_reliability(&mut self, measure: &str, reliability: f64) {
        self.reliability.insert(measure.to_string(), reliability.clamp(0.0, 1.0));
    }

    /// Get reference for a measure.
    pub fn get_reference(&self, measure: &str) -> Option<&NormativeReference> {
        self.references.get(measure)
    }

    /// Get reliability for a measure.
    pub fn get_reliability(&self, measure: &str) -> Option<f64> {
        self.reliability.get(measure).copied()
    }

    /// List all measures with references.
    pub fn measures(&self) -> Vec<&String> {
        self.references.keys().collect()
    }
}

/// Reference statistics for normative comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormativeReference {
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub sd: f64,
    /// Median (if available).
    pub median: Option<f64>,
    /// Skewness (if available).
    pub skewness: Option<f64>,
    /// Kurtosis (if available).
    pub kurtosis: Option<f64>,
    /// Percentile values (if available).
    pub percentiles: Option<PercentileTable>,
    /// Standard error of measurement.
    pub sem: Option<f64>,
    /// Whether higher scores are better.
    pub higher_is_better: bool,
}

impl NormativeReference {
    /// Create a new reference with mean and SD.
    pub fn new(mean: f64, sd: f64) -> Self {
        Self {
            mean,
            sd,
            median: None,
            skewness: None,
            kurtosis: None,
            percentiles: None,
            sem: None,
            higher_is_better: true,
        }
    }

    /// Set median.
    pub fn with_median(mut self, median: f64) -> Self {
        self.median = Some(median);
        self
    }

    /// Set skewness.
    pub fn with_skewness(mut self, skewness: f64) -> Self {
        self.skewness = Some(skewness);
        self
    }

    /// Set SEM.
    pub fn with_sem(mut self, sem: f64) -> Self {
        self.sem = Some(sem);
        self
    }

    /// Set higher_is_better flag.
    pub fn with_direction(mut self, higher_is_better: bool) -> Self {
        self.higher_is_better = higher_is_better;
        self
    }

    /// Set percentile table.
    pub fn with_percentiles(mut self, percentiles: PercentileTable) -> Self {
        self.percentiles = Some(percentiles);
        self
    }

    /// Calculate z-score.
    pub fn z_score(&self, value: f64) -> f64 {
        if self.sd.abs() < 1e-10 {
            0.0
        } else {
            (value - self.mean) / self.sd
        }
    }

    /// Calculate T-score (mean=50, SD=10).
    pub fn t_score(&self, value: f64) -> f64 {
        let z = self.z_score(value);
        50.0 + z * 10.0
    }

    /// Calculate scaled score (mean=10, SD=3).
    pub fn scaled_score(&self, value: f64) -> f64 {
        let z = self.z_score(value);
        10.0 + z * 3.0
    }

    /// Get percentile from percentile table (if available) or estimate from z.
    pub fn percentile(&self, value: f64) -> f64 {
        if let Some(ref table) = self.percentiles {
            table.interpolate(value)
        } else {
            z_to_percentile(self.z_score(value))
        }
    }

    /// Get confidence interval for the score.
    pub fn confidence_interval(&self, value: f64, confidence: f64) -> Option<(f64, f64)> {
        let sem = self.sem?;
        let z_crit = confidence_to_z(confidence);
        let margin = z_crit * sem;
        Some((value - margin, value + margin))
    }

    /// Check if value is in normal range.
    pub fn is_normal(&self, value: f64, sd_threshold: f64) -> bool {
        let z = self.z_score(value).abs();
        z <= sd_threshold
    }

    /// Get qualitative descriptor for z-score.
    pub fn qualitative_descriptor(&self, value: f64) -> &'static str {
        let z = self.z_score(value);
        match z {
            z if z >= 2.0 => "Very Superior",
            z if z >= 1.3 => "Superior",
            z if z >= 0.7 => "High Average",
            z if z >= -0.7 => "Average",
            z if z >= -1.3 => "Low Average",
            z if z >= -2.0 => "Borderline",
            _ => "Extremely Low",
        }
    }
}

/// Percentile lookup table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileTable {
    /// Percentile values (0-100).
    percentiles: Vec<f64>,
    /// Corresponding raw scores.
    scores: Vec<f64>,
}

impl PercentileTable {
    /// Create from percentile-score pairs.
    pub fn new(percentiles: Vec<f64>, scores: Vec<f64>) -> Self {
        Self { percentiles, scores }
    }

    /// Create standard percentile table.
    pub fn standard() -> Self {
        Self {
            percentiles: vec![1.0, 2.0, 5.0, 9.0, 16.0, 25.0, 37.0, 50.0, 63.0, 75.0, 84.0, 91.0, 95.0, 98.0, 99.0],
            scores: vec![-2.33, -2.05, -1.65, -1.34, -1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0, 1.34, 1.65, 2.05, 2.33],
        }
    }

    /// Interpolate percentile for a score.
    pub fn interpolate(&self, score: f64) -> f64 {
        if self.scores.is_empty() {
            return 50.0;
        }

        // Find bracketing scores
        for i in 0..self.scores.len() - 1 {
            if score >= self.scores[i] && score <= self.scores[i + 1] {
                // Linear interpolation
                let t = (score - self.scores[i]) / (self.scores[i + 1] - self.scores[i]);
                return self.percentiles[i] + t * (self.percentiles[i + 1] - self.percentiles[i]);
            }
        }

        // Extrapolate
        if score < self.scores[0] {
            self.percentiles[0]
        } else {
            *self.percentiles.last().unwrap()
        }
    }
}

/// Definition of a measure in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureDefinition {
    /// Measure name/code.
    pub name: String,
    /// Display name.
    pub display_name: String,
    /// Description.
    pub description: Option<String>,
    /// Units.
    pub units: Option<String>,
    /// Domain (e.g., "attention", "memory").
    pub domain: Option<String>,
    /// Whether higher is better.
    pub higher_is_better: bool,
    /// Minimum valid value.
    pub min_value: Option<f64>,
    /// Maximum valid value.
    pub max_value: Option<f64>,
}

impl MeasureDefinition {
    /// Create new measure definition.
    pub fn new(name: &str, display_name: &str) -> Self {
        Self {
            name: name.to_string(),
            display_name: display_name.to_string(),
            description: None,
            units: None,
            domain: None,
            higher_is_better: true,
            min_value: None,
            max_value: None,
        }
    }

    /// Set description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set units.
    pub fn with_units(mut self, units: &str) -> Self {
        self.units = Some(units.to_string());
        self
    }

    /// Set domain.
    pub fn with_domain(mut self, domain: &str) -> Self {
        self.domain = Some(domain.to_string());
        self
    }

    /// Set direction.
    pub fn with_direction(mut self, higher_is_better: bool) -> Self {
        self.higher_is_better = higher_is_better;
        self
    }

    /// Set valid range.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Validate a value.
    pub fn validate(&self, value: f64) -> bool {
        if let Some(min) = self.min_value {
            if value < min {
                return false;
            }
        }
        if let Some(max) = self.max_value {
            if value > max {
                return false;
            }
        }
        true
    }
}

/// Convert z-score to percentile using normal CDF approximation.
fn z_to_percentile(z: f64) -> f64 {
    // Approximation of normal CDF
    let x = z.abs();
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

    if z >= 0.0 {
        (1.0 - p) * 100.0
    } else {
        p * 100.0
    }
}

/// Convert confidence level to z-critical value.
fn confidence_to_z(confidence: f64) -> f64 {
    match confidence {
        c if c >= 0.99 => 2.576,
        c if c >= 0.95 => 1.96,
        c if c >= 0.90 => 1.645,
        c if c >= 0.85 => 1.44,
        c if c >= 0.80 => 1.28,
        _ => 1.96,
    }
}

/// Reliable change index calculation.
#[derive(Debug, Clone)]
pub struct ReliableChangeIndex {
    /// Standard error of difference.
    pub se_diff: f64,
    /// Critical value for significance.
    pub critical_value: f64,
}

impl ReliableChangeIndex {
    /// Create RCI from test reliability and SD.
    pub fn new(reliability: f64, sd: f64, confidence: f64) -> Self {
        let sem = sd * (1.0 - reliability).sqrt();
        let se_diff = (2.0 * sem * sem).sqrt();
        let critical_value = confidence_to_z(confidence);

        Self {
            se_diff,
            critical_value,
        }
    }

    /// Calculate RCI value.
    pub fn calculate(&self, baseline: f64, followup: f64) -> f64 {
        (followup - baseline) / self.se_diff
    }

    /// Determine if change is reliable.
    pub fn is_reliable(&self, baseline: f64, followup: f64) -> bool {
        self.calculate(baseline, followup).abs() >= self.critical_value
    }

    /// Determine change direction.
    pub fn change_classification(&self, baseline: f64, followup: f64) -> ChangeClassification {
        let rci = self.calculate(baseline, followup);
        if rci >= self.critical_value {
            ChangeClassification::ReliableImprovement
        } else if rci <= -self.critical_value {
            ChangeClassification::ReliableDecline
        } else {
            ChangeClassification::NoChange
        }
    }
}

/// Classification of change over time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeClassification {
    /// Reliable improvement.
    ReliableImprovement,
    /// No reliable change.
    NoChange,
    /// Reliable decline.
    ReliableDecline,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demographics::{Sex, Ethnicity};

    #[test]
    fn test_z_score_calculation() {
        let reference = NormativeReference::new(100.0, 15.0);

        assert!((reference.z_score(100.0) - 0.0).abs() < 0.001);
        assert!((reference.z_score(115.0) - 1.0).abs() < 0.001);
        assert!((reference.z_score(85.0) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_t_score_calculation() {
        let reference = NormativeReference::new(100.0, 15.0);

        assert!((reference.t_score(100.0) - 50.0).abs() < 0.001);
        assert!((reference.t_score(115.0) - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_normative_database() {
        let mut db = NormativeDatabase::new("Test DB", "1.0");

        let mut norms = PopulationNorms::new("Adult Males European");
        norms.add_reference("memory", NormativeReference::new(100.0, 15.0));

        db.add_norms("Adult_Male_European", norms, 500);

        let demo = Demographics::new()
            .with_age(35)
            .with_sex(Sex::Male)
            .with_ethnicity(Ethnicity::European);

        let z = db.z_score("memory", 115.0, &demo).unwrap();
        assert!((z - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_qualitative_descriptor() {
        let reference = NormativeReference::new(100.0, 15.0);

        assert_eq!(reference.qualitative_descriptor(130.0), "Very Superior");
        assert_eq!(reference.qualitative_descriptor(100.0), "Average");
        // 70 is z = -2.0 exactly. Under the conventional Wechsler bands that is
        // the bottom of Borderline (70-79); Extremely Low is <= 69. This
        // assertion previously expected "Extremely Low" and failed — the band
        // boundaries in qualitative_descriptor are right and the expectation was
        // wrong. Changing the classifier to satisfy it would have mislabelled
        // every borderline score as extremely low.
        assert_eq!(reference.qualitative_descriptor(70.0), "Borderline");
        assert_eq!(reference.qualitative_descriptor(69.0), "Extremely Low");
    }

    #[test]
    fn qualitative_descriptor_band_boundaries() {
        // qualitative_descriptor is defined on the Z-SCORE (see its doc), with
        // cut-offs at +/-0.7, +/-1.3 and +/-2.0. Those approximate — but do not
        // exactly reproduce — the Wechsler standard-score bands: with SD 15 a
        // score of 110 is z = 0.67, just under the 0.7 cut-off, so it lands in
        // Average where the Wechsler band (110-119) would say High Average.
        //
        // This pins the classifier's ACTUAL contract so a future edit cannot
        // shift a classification unnoticed. It deliberately does not "correct"
        // the cut-offs to the score bands: that would change how real
        // measurements are labelled and needs a clinical decision, not a test.
        let r = NormativeReference::new(100.0, 15.0);
        for (score, expected) in [
            (130.0, "Very Superior"),  // z =  2.00
            (125.0, "Superior"),       // z =  1.67
            (121.0, "Superior"),       // z =  1.40
            (115.0, "High Average"),   // z =  1.00
            (111.0, "High Average"),   // z =  0.73
            (100.0, "Average"),        // z =  0.00
            (90.0, "Average"),         // z = -0.67
            (85.0, "Low Average"),     // z = -1.00
            (80.0, "Borderline"),      // z = -1.33, just past the -1.3 cut-off
            (75.0, "Borderline"),      // z = -1.67
            (70.0, "Borderline"),      // z = -2.00 exactly
            (69.0, "Extremely Low"),   // z = -2.07
            (50.0, "Extremely Low"),   // z = -3.33
        ] {
            assert_eq!(
                r.qualitative_descriptor(score),
                expected,
                "score {score} (z = {:.2}) misclassified",
                r.z_score(score)
            );
        }
    }

    #[test]
    fn test_reliable_change_index() {
        let rci = ReliableChangeIndex::new(0.90, 15.0, 0.95);

        // Large improvement should be reliable
        let classification = rci.change_classification(100.0, 120.0);
        assert_eq!(classification, ChangeClassification::ReliableImprovement);

        // Small change should not be reliable
        let classification = rci.change_classification(100.0, 102.0);
        assert_eq!(classification, ChangeClassification::NoChange);
    }

    #[test]
    fn test_percentile_conversion() {
        assert!((z_to_percentile(0.0) - 50.0).abs() < 1.0);
        assert!(z_to_percentile(1.0) > 80.0);
        assert!(z_to_percentile(-1.0) < 20.0);
    }

    #[test]
    fn test_confidence_interval() {
        let reference = NormativeReference::new(100.0, 15.0).with_sem(4.5);

        let ci = reference.confidence_interval(100.0, 0.95).unwrap();
        assert!(ci.0 < 100.0 && ci.1 > 100.0);
        assert!((ci.1 - ci.0) > 15.0); // CI should be reasonably wide
    }
}
