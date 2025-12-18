//! Pediatric normative data (0-17 years)
//!
//! Provides age-stratified normative reference values for pediatric populations,
//! accounting for rapid developmental changes across infancy, childhood, and adolescence.
//!
//! # Developmental Stages
//!
//! - **Infant** (0-12 months): Rapid neuromotor development
//! - **Toddler** (1-3 years): Locomotion emergence and refinement
//! - **Preschool** (3-5 years): Motor skill consolidation
//! - **School Age** (6-12 years): Continued cognitive and motor maturation
//! - **Adolescent** (13-17 years): Approaching adult-level performance
//!
//! # Clinical Applications
//!
//! - Developmental delay screening
//! - Tracking growth-related changes
//! - Identifying early neurological impairments
//! - Longitudinal developmental monitoring

use crate::{Demographics, MetricType, NormativeStats, NormsError, Result, Sex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Developmental stages for pediatric assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DevelopmentalStage {
    /// 0-12 months
    Infant,
    /// 1-3 years (13-36 months)
    Toddler,
    /// 3-5 years (37-60 months)
    Preschool,
    /// 6-12 years (61-144 months)
    SchoolAge,
    /// 13-17 years (145-216 months)
    Adolescent,
}

impl DevelopmentalStage {
    /// Get the age range in months for this stage
    pub fn age_range_months(&self) -> (u16, u16) {
        match self {
            DevelopmentalStage::Infant => (0, 12),
            DevelopmentalStage::Toddler => (13, 36),
            DevelopmentalStage::Preschool => (37, 60),
            DevelopmentalStage::SchoolAge => (61, 144),
            DevelopmentalStage::Adolescent => (145, 216),
        }
    }

    /// Get developmental stage from age in months
    pub fn from_age_months(age_months: u16) -> Option<Self> {
        match age_months {
            0..=12 => Some(DevelopmentalStage::Infant),
            13..=36 => Some(DevelopmentalStage::Toddler),
            37..=60 => Some(DevelopmentalStage::Preschool),
            61..=144 => Some(DevelopmentalStage::SchoolAge),
            145..=216 => Some(DevelopmentalStage::Adolescent),
            _ => None,
        }
    }

    /// Get descriptive label
    pub fn label(&self) -> &'static str {
        match self {
            DevelopmentalStage::Infant => "Infant",
            DevelopmentalStage::Toddler => "Toddler",
            DevelopmentalStage::Preschool => "Preschool",
            DevelopmentalStage::SchoolAge => "School-Age",
            DevelopmentalStage::Adolescent => "Adolescent",
        }
    }
}

/// Age range for pediatric norms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PediatricAgeRange {
    /// Minimum age in months (inclusive)
    pub min_months: u16,
    /// Maximum age in months (inclusive)
    pub max_months: u16,
    /// Developmental stage
    pub stage: DevelopmentalStage,
}

impl PediatricAgeRange {
    /// Create a new pediatric age range
    pub fn new(min_months: u16, max_months: u16) -> Self {
        let stage = DevelopmentalStage::from_age_months((min_months + max_months) / 2)
            .unwrap_or(DevelopmentalStage::Infant);
        Self {
            min_months,
            max_months,
            stage,
        }
    }

    /// Check if age falls within this range
    pub fn contains(&self, age_months: u16) -> bool {
        age_months >= self.min_months && age_months <= self.max_months
    }
}

/// Percentile values for reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percentiles {
    /// 3rd percentile
    pub p3: f64,
    /// 5th percentile
    pub p5: f64,
    /// 10th percentile
    pub p10: f64,
    /// 25th percentile (Q1)
    pub p25: f64,
    /// 50th percentile (median)
    pub p50: f64,
    /// 75th percentile (Q3)
    pub p75: f64,
    /// 90th percentile
    pub p90: f64,
    /// 95th percentile
    pub p95: f64,
    /// 97th percentile
    pub p97: f64,
}

impl Percentiles {
    /// Create from mean and standard deviation assuming normal distribution
    pub fn from_normal(mean: f64, std_dev: f64) -> Self {
        Self {
            p3: mean - 1.88 * std_dev,
            p5: mean - 1.645 * std_dev,
            p10: mean - 1.28 * std_dev,
            p25: mean - 0.674 * std_dev,
            p50: mean,
            p75: mean + 0.674 * std_dev,
            p90: mean + 1.28 * std_dev,
            p95: mean + 1.645 * std_dev,
            p97: mean + 1.88 * std_dev,
        }
    }

    /// Get value at specific percentile (interpolated)
    pub fn at(&self, percentile: f64) -> Option<f64> {
        if !(0.0..=100.0).contains(&percentile) {
            return None;
        }

        // Linear interpolation between known percentiles
        let value = if percentile <= 3.0 {
            self.p3
        } else if percentile <= 5.0 {
            self.p3 + (self.p5 - self.p3) * (percentile - 3.0) / 2.0
        } else if percentile <= 10.0 {
            self.p5 + (self.p10 - self.p5) * (percentile - 5.0) / 5.0
        } else if percentile <= 25.0 {
            self.p10 + (self.p25 - self.p10) * (percentile - 10.0) / 15.0
        } else if percentile <= 50.0 {
            self.p25 + (self.p50 - self.p25) * (percentile - 25.0) / 25.0
        } else if percentile <= 75.0 {
            self.p50 + (self.p75 - self.p50) * (percentile - 50.0) / 25.0
        } else if percentile <= 90.0 {
            self.p75 + (self.p90 - self.p75) * (percentile - 75.0) / 15.0
        } else if percentile <= 95.0 {
            self.p90 + (self.p95 - self.p90) * (percentile - 90.0) / 5.0
        } else if percentile <= 97.0 {
            self.p95 + (self.p97 - self.p95) * (percentile - 95.0) / 2.0
        } else {
            self.p97
        };

        Some(value)
    }
}

/// Pediatric normative reference for a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PediatricReference {
    /// Metric identifier
    pub metric_id: String,
    /// Age range for this reference
    pub age_range: PediatricAgeRange,
    /// Sex (None = combined sexes)
    pub sex: Option<Sex>,
    /// Percentile values
    pub percentiles: Percentiles,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
}

impl PediatricReference {
    /// Create a new pediatric reference
    pub fn new(
        metric_id: String,
        min_months: u16,
        max_months: u16,
        sex: Option<Sex>,
        mean: f64,
        std_dev: f64,
    ) -> Self {
        Self {
            metric_id,
            age_range: PediatricAgeRange::new(min_months, max_months),
            sex,
            percentiles: Percentiles::from_normal(mean, std_dev),
            mean,
            std_dev,
        }
    }

    /// Check if this reference matches the given criteria
    pub fn matches(&self, metric_id: &str, age_months: u16, sex: Sex) -> bool {
        // Check metric
        if self.metric_id != metric_id {
            return false;
        }

        // Check age range
        if !self.age_range.contains(age_months) {
            return false;
        }

        // Check sex (None means applies to both)
        if let Some(ref_sex) = self.sex {
            if ref_sex != sex {
                return false;
            }
        }

        true
    }

    /// Calculate z-score for a given value
    pub fn z_score(&self, value: f64) -> f64 {
        (value - self.mean) / self.std_dev
    }

    /// Calculate percentile for a given value (approximate using normal CDF)
    pub fn percentile(&self, value: f64) -> f64 {
        let z = self.z_score(value);
        normal_cdf(z) * 100.0
    }
}

/// Pediatric normative database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PediatricNormativeDb {
    /// All reference values, indexed by metric ID
    references: HashMap<String, Vec<PediatricReference>>,
}

impl PediatricNormativeDb {
    /// Create a new empty database
    pub fn new() -> Self {
        Self {
            references: HashMap::new(),
        }
    }

    /// Create database with default pediatric norms
    pub fn with_defaults() -> Self {
        let mut db = Self::new();
        db.populate_defaults();
        db
    }

    /// Add a reference to the database
    pub fn add_reference(&mut self, reference: PediatricReference) {
        self.references
            .entry(reference.metric_id.clone())
            .or_insert_with(Vec::new)
            .push(reference);
    }

    /// Lookup the best matching reference
    pub fn lookup(
        &self,
        metric: &str,
        age_months: u16,
        sex: Sex,
    ) -> Option<&PediatricReference> {
        let refs = self.references.get(metric)?;

        // Find exact match first (sex-specific and age-matched)
        refs.iter()
            .filter(|r| r.matches(metric, age_months, sex))
            .max_by_key(|r| {
                // Prefer sex-specific over combined
                let sex_score = if r.sex.is_some() { 2 } else { 1 };
                // Prefer narrower age ranges
                let age_span = r.age_range.max_months - r.age_range.min_months;
                let age_score = 1000 - age_span as i32;
                sex_score * 10000 + age_score
            })
    }

    /// Get percentile for a value
    pub fn get_percentile(
        &self,
        metric: &str,
        value: f64,
        age_months: u16,
        sex: Sex,
    ) -> Option<f64> {
        let reference = self.lookup(metric, age_months, sex)?;
        Some(reference.percentile(value))
    }

    /// Get z-score for a value
    pub fn get_z_score(
        &self,
        metric: &str,
        value: f64,
        age_months: u16,
        sex: Sex,
    ) -> Option<f64> {
        let reference = self.lookup(metric, age_months, sex)?;
        Some(reference.z_score(value))
    }

    /// Populate with default pediatric normative data
    fn populate_defaults(&mut self) {
        // === Heart Rate Variability (HRV SDNN) ===
        // Based on Task Force standards and pediatric literature
        // Infant: 70-100 ms, Toddler: 85-115 ms, Preschool: 95-125 ms,
        // School-age: 110-140 ms, Adolescent: 120-150 ms

        self.add_reference(PediatricReference::new(
            "hrv_sdnn".to_string(),
            0, 12,    // Infant
            None,
            85.0, 20.0,
        ));

        self.add_reference(PediatricReference::new(
            "hrv_sdnn".to_string(),
            13, 36,   // Toddler
            None,
            100.0, 22.0,
        ));

        self.add_reference(PediatricReference::new(
            "hrv_sdnn".to_string(),
            37, 60,   // Preschool
            None,
            110.0, 25.0,
        ));

        self.add_reference(PediatricReference::new(
            "hrv_sdnn".to_string(),
            61, 144,  // School-age
            None,
            125.0, 28.0,
        ));

        self.add_reference(PediatricReference::new(
            "hrv_sdnn".to_string(),
            145, 216, // Adolescent
            None,
            135.0, 30.0,
        ));

        // === Gait Velocity (m/s) ===
        // Toddlers start walking around 12 months, velocity increases with age

        self.add_reference(PediatricReference::new(
            "gait_velocity".to_string(),
            13, 36,   // Toddler (early walkers)
            None,
            0.65, 0.15,
        ));

        self.add_reference(PediatricReference::new(
            "gait_velocity".to_string(),
            37, 60,   // Preschool
            None,
            0.90, 0.12,
        ));

        self.add_reference(PediatricReference::new(
            "gait_velocity".to_string(),
            61, 144,  // School-age
            None,
            1.15, 0.15,
        ));

        self.add_reference(PediatricReference::new(
            "gait_velocity".to_string(),
            145, 216, // Adolescent (approaching adult values)
            Some(Sex::Male),
            1.30, 0.12,
        ));

        self.add_reference(PediatricReference::new(
            "gait_velocity".to_string(),
            145, 216, // Adolescent
            Some(Sex::Female),
            1.25, 0.12,
        ));

        // === Reaction Time (ms) ===
        // Simple reaction time decreases with age/maturation

        self.add_reference(PediatricReference::new(
            "simple_reaction_time".to_string(),
            61, 96,   // 5-8 years
            None,
            450.0, 80.0,
        ));

        self.add_reference(PediatricReference::new(
            "simple_reaction_time".to_string(),
            97, 144,  // 9-12 years
            None,
            350.0, 60.0,
        ));

        self.add_reference(PediatricReference::new(
            "simple_reaction_time".to_string(),
            145, 180, // 13-15 years
            None,
            290.0, 50.0,
        ));

        self.add_reference(PediatricReference::new(
            "simple_reaction_time".to_string(),
            181, 216, // 16-17 years (approaching adult)
            None,
            260.0, 45.0,
        ));

        // === Grip Strength (kg) ===
        // Sex differences emerge in adolescence

        self.add_reference(PediatricReference::new(
            "grip_strength".to_string(),
            61, 96,   // 5-8 years
            None,
            10.0, 3.0,
        ));

        self.add_reference(PediatricReference::new(
            "grip_strength".to_string(),
            97, 144,  // 9-12 years
            None,
            18.0, 5.0,
        ));

        self.add_reference(PediatricReference::new(
            "grip_strength".to_string(),
            145, 180, // 13-15 years
            Some(Sex::Male),
            32.0, 8.0,
        ));

        self.add_reference(PediatricReference::new(
            "grip_strength".to_string(),
            145, 180, // 13-15 years
            Some(Sex::Female),
            24.0, 6.0,
        ));

        self.add_reference(PediatricReference::new(
            "grip_strength".to_string(),
            181, 216, // 16-17 years
            Some(Sex::Male),
            42.0, 9.0,
        ));

        self.add_reference(PediatricReference::new(
            "grip_strength".to_string(),
            181, 216, // 16-17 years
            Some(Sex::Female),
            28.0, 6.0,
        ));

        // === HRV RMSSD (ms) ===
        // Parasympathetic marker, increases with age

        self.add_reference(PediatricReference::new(
            "hrv_rmssd".to_string(),
            0, 12,    // Infant
            None,
            35.0, 15.0,
        ));

        self.add_reference(PediatricReference::new(
            "hrv_rmssd".to_string(),
            13, 60,   // Toddler + Preschool
            None,
            42.0, 18.0,
        ));

        self.add_reference(PediatricReference::new(
            "hrv_rmssd".to_string(),
            61, 144,  // School-age
            None,
            50.0, 20.0,
        ));

        self.add_reference(PediatricReference::new(
            "hrv_rmssd".to_string(),
            145, 216, // Adolescent
            None,
            55.0, 22.0,
        ));
    }

    /// Get all metrics in the database
    pub fn metrics(&self) -> Vec<String> {
        self.references.keys().cloned().collect()
    }

    /// Get all references for a specific metric
    pub fn get_all_references(&self, metric: &str) -> Option<&Vec<PediatricReference>> {
        self.references.get(metric)
    }
}

impl Default for PediatricNormativeDb {
    fn default() -> Self {
        Self::with_defaults()
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
    fn test_developmental_stage_from_age() {
        assert_eq!(
            DevelopmentalStage::from_age_months(6),
            Some(DevelopmentalStage::Infant)
        );
        assert_eq!(
            DevelopmentalStage::from_age_months(24),
            Some(DevelopmentalStage::Toddler)
        );
        assert_eq!(
            DevelopmentalStage::from_age_months(48),
            Some(DevelopmentalStage::Preschool)
        );
        assert_eq!(
            DevelopmentalStage::from_age_months(96),
            Some(DevelopmentalStage::SchoolAge)
        );
        assert_eq!(
            DevelopmentalStage::from_age_months(180),
            Some(DevelopmentalStage::Adolescent)
        );
    }

    #[test]
    fn test_pediatric_age_range() {
        let range = PediatricAgeRange::new(61, 144);
        assert!(range.contains(96));
        assert!(!range.contains(50));
        assert_eq!(range.stage, DevelopmentalStage::SchoolAge);
    }

    #[test]
    fn test_percentiles_from_normal() {
        let percentiles = Percentiles::from_normal(100.0, 15.0);

        // Median should equal mean
        assert!((percentiles.p50 - 100.0).abs() < 0.01);

        // Check ordering
        assert!(percentiles.p3 < percentiles.p5);
        assert!(percentiles.p5 < percentiles.p10);
        assert!(percentiles.p90 < percentiles.p95);
        assert!(percentiles.p95 < percentiles.p97);
    }

    #[test]
    fn test_pediatric_db_lookup() {
        let db = PediatricNormativeDb::with_defaults();

        // Lookup HRV for school-age child
        let reference = db.lookup("hrv_sdnn", 96, Sex::Male);
        assert!(reference.is_some());

        let reference = reference.unwrap();
        assert_eq!(reference.metric_id, "hrv_sdnn");
        assert!(reference.age_range.contains(96));
    }

    #[test]
    fn test_pediatric_db_percentile() {
        let db = PediatricNormativeDb::with_defaults();

        // Check percentile calculation for grip strength
        let percentile = db.get_percentile("grip_strength", 18.0, 96, Sex::Male);
        assert!(percentile.is_some());

        let p = percentile.unwrap();
        // 18 kg is the mean for 9-12 year olds, should be around 50th percentile
        assert!((p - 50.0).abs() < 10.0);
    }

    #[test]
    fn test_pediatric_db_z_score() {
        let db = PediatricNormativeDb::with_defaults();

        let z = db.get_z_score("grip_strength", 18.0, 96, Sex::Male);
        assert!(z.is_some());

        // Mean should have z-score near 0
        assert!(z.unwrap().abs() < 0.5);
    }

    #[test]
    fn test_sex_specific_adolescent_norms() {
        let db = PediatricNormativeDb::with_defaults();

        // Adolescent grip strength should differ by sex
        let male_ref = db.lookup("grip_strength", 180, Sex::Male);
        let female_ref = db.lookup("grip_strength", 180, Sex::Female);

        assert!(male_ref.is_some());
        assert!(female_ref.is_some());

        // Males should have higher mean grip strength
        assert!(male_ref.unwrap().mean > female_ref.unwrap().mean);
    }

    #[test]
    fn test_reaction_time_improvement_with_age() {
        let db = PediatricNormativeDb::with_defaults();

        // Younger children should have slower reaction times
        let young_ref = db.lookup("simple_reaction_time", 72, Sex::Male);
        let older_ref = db.lookup("simple_reaction_time", 180, Sex::Male);

        assert!(young_ref.is_some());
        assert!(older_ref.is_some());

        // Reaction time should decrease (improve) with age
        assert!(young_ref.unwrap().mean > older_ref.unwrap().mean);
    }

    #[test]
    fn test_percentile_at_interpolation() {
        let percentiles = Percentiles::from_normal(100.0, 15.0);

        // Test interpolation
        let p40 = percentiles.at(40.0);
        assert!(p40.is_some());

        let val = p40.unwrap();
        // Should be between p25 and p50
        assert!(val > percentiles.p25);
        assert!(val < percentiles.p50);
    }
}
