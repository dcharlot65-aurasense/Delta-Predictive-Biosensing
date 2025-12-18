//! Geriatric normative data extensions (80+ years)
//!
//! Provides fine-grained age stratification for older adults (80+) and accounts
//! for frailty heterogeneity within age groups.
//!
//! # Age Groups
//!
//! - **Young-Old** (65-74): Early retirement age, generally healthy
//! - **Middle-Old** (75-84): Increased chronic conditions, functional decline
//! - **Old-Old** (85-94): High heterogeneity, frailty common
//! - **Oldest-Old** (95+): Exceptional longevity, extreme frailty variation
//!
//! # Frailty Adjustment
//!
//! Traditional age-stratified norms fail to capture the high heterogeneity
//! in older adults. Frailty-adjusted norms account for accumulated deficits
//! and provide more personalized reference values.
//!
//! # Clinical Applications
//!
//! - Fall risk assessment
//! - Frailty screening and monitoring
//! - Functional decline detection
//! - Rehabilitation outcome prediction

use crate::{Demographics, MetricType, NormativeStats, NormsError, Result, Sex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Geriatric age groups for fine-grained stratification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeriatricAgeGroup {
    /// 65-74 years
    YoungOld,
    /// 75-84 years
    MiddleOld,
    /// 85-94 years
    OldOld,
    /// 95+ years
    Oldest,
}

impl GeriatricAgeGroup {
    /// Get age range for this group
    pub fn age_range(&self) -> (u8, u8) {
        match self {
            GeriatricAgeGroup::YoungOld => (65, 74),
            GeriatricAgeGroup::MiddleOld => (75, 84),
            GeriatricAgeGroup::OldOld => (85, 94),
            GeriatricAgeGroup::Oldest => (95, 120),
        }
    }

    /// Get age group from age in years
    pub fn from_age(age: u8) -> Option<Self> {
        match age {
            65..=74 => Some(GeriatricAgeGroup::YoungOld),
            75..=84 => Some(GeriatricAgeGroup::MiddleOld),
            85..=94 => Some(GeriatricAgeGroup::OldOld),
            95..=120 => Some(GeriatricAgeGroup::Oldest),
            _ => None,
        }
    }

    /// Get descriptive label
    pub fn label(&self) -> &'static str {
        match self {
            GeriatricAgeGroup::YoungOld => "Young-Old",
            GeriatricAgeGroup::MiddleOld => "Middle-Old",
            GeriatricAgeGroup::OldOld => "Old-Old",
            GeriatricAgeGroup::Oldest => "Oldest-Old",
        }
    }

    /// Check if age falls within this group
    pub fn contains(&self, age: u8) -> bool {
        let (min, max) = self.age_range();
        age >= min && age <= max
    }
}

/// Frailty adjustment for normative values
///
/// Accounts for accumulated health deficits that create heterogeneity
/// within age groups. Frailty Index ranges from 0 (no deficits) to 1
/// (maximum frailty).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FrailtyAdjustment {
    /// Frailty Index (0-1 scale, proportion of deficits present)
    pub frailty_index: f64,
    /// Adjustment factor to apply to expected value
    /// < 1.0 for metrics where frailty decreases performance
    /// > 1.0 for metrics where frailty increases values
    pub adjustment_factor: f64,
}

impl FrailtyAdjustment {
    /// Create a new frailty adjustment
    pub fn new(frailty_index: f64, adjustment_factor: f64) -> Self {
        Self {
            frailty_index: frailty_index.clamp(0.0, 1.0),
            adjustment_factor,
        }
    }

    /// Apply adjustment to a normative value
    pub fn apply(&self, baseline_value: f64) -> f64 {
        baseline_value * self.adjustment_factor
    }
}

/// Frailty category for clinical interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrailtyCategory {
    /// FI < 0.10: Robust, few health deficits
    Robust,
    /// 0.10 <= FI < 0.21: Pre-frail, some deficits emerging
    PreFrail,
    /// 0.21 <= FI < 0.45: Frail, multiple deficits
    Frail,
    /// FI >= 0.45: Severely frail, high mortality risk
    SeverelyFrail,
}

impl FrailtyCategory {
    /// Categorize based on Frailty Index value
    pub fn from_index(fi: f64) -> Self {
        if fi < 0.10 {
            FrailtyCategory::Robust
        } else if fi < 0.21 {
            FrailtyCategory::PreFrail
        } else if fi < 0.45 {
            FrailtyCategory::Frail
        } else {
            FrailtyCategory::SeverelyFrail
        }
    }

    /// Get descriptive label
    pub fn label(&self) -> &'static str {
        match self {
            FrailtyCategory::Robust => "Robust",
            FrailtyCategory::PreFrail => "Pre-Frail",
            FrailtyCategory::Frail => "Frail",
            FrailtyCategory::SeverelyFrail => "Severely Frail",
        }
    }

    /// Get clinical description
    pub fn description(&self) -> &'static str {
        match self {
            FrailtyCategory::Robust => "Few health deficits, low mortality risk",
            FrailtyCategory::PreFrail => "Some deficits emerging, increased vulnerability",
            FrailtyCategory::Frail => "Multiple health deficits, high intervention need",
            FrailtyCategory::SeverelyFrail => "Severe deficits, very high mortality risk",
        }
    }
}

/// Geriatric normative reference with optional frailty adjustments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeriatricReference {
    /// Metric identifier
    pub metric_id: String,
    /// Age group
    pub age_group: GeriatricAgeGroup,
    /// Sex (None = combined sexes)
    pub sex: Option<Sex>,
    /// Baseline percentiles (for robust individuals, FI ~0)
    pub percentiles: crate::pediatric::Percentiles,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Optional frailty adjustments at different FI levels
    pub frailty_adjustments: Option<Vec<FrailtyAdjustment>>,
}

impl GeriatricReference {
    /// Create a new geriatric reference without frailty adjustments
    pub fn new(
        metric_id: String,
        age_group: GeriatricAgeGroup,
        sex: Option<Sex>,
        mean: f64,
        std_dev: f64,
    ) -> Self {
        Self {
            metric_id,
            age_group,
            sex,
            percentiles: crate::pediatric::Percentiles::from_normal(mean, std_dev),
            mean,
            std_dev,
            frailty_adjustments: None,
        }
    }

    /// Add frailty adjustments
    pub fn with_frailty_adjustments(mut self, adjustments: Vec<FrailtyAdjustment>) -> Self {
        self.frailty_adjustments = Some(adjustments);
        self
    }

    /// Check if this reference matches the given criteria
    pub fn matches(&self, metric_id: &str, age: u8, sex: Sex) -> bool {
        // Check metric
        if self.metric_id != metric_id {
            return false;
        }

        // Check age group
        if !self.age_group.contains(age) {
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

    /// Get frailty-adjusted mean value
    pub fn frailty_adjusted_mean(&self, frailty_index: f64) -> f64 {
        if let Some(ref adjustments) = self.frailty_adjustments {
            // Find closest frailty adjustment
            if let Some(adjustment) = adjustments
                .iter()
                .min_by_key(|a| ((a.frailty_index - frailty_index).abs() * 1000.0) as i32)
            {
                return adjustment.apply(self.mean);
            }
        }

        // No adjustments available, return baseline
        self.mean
    }

    /// Calculate z-score for a given value, accounting for frailty if available
    pub fn z_score(&self, value: f64, frailty_index: Option<f64>) -> f64 {
        let expected_mean = if let Some(fi) = frailty_index {
            self.frailty_adjusted_mean(fi)
        } else {
            self.mean
        };

        (value - expected_mean) / self.std_dev
    }

    /// Calculate percentile for a given value
    pub fn percentile(&self, value: f64, frailty_index: Option<f64>) -> f64 {
        let z = self.z_score(value, frailty_index);
        normal_cdf(z) * 100.0
    }
}

/// Geriatric normative database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeriatricNormativeDb {
    /// All reference values, indexed by metric ID
    references: HashMap<String, Vec<GeriatricReference>>,
}

impl GeriatricNormativeDb {
    /// Create a new empty database
    pub fn new() -> Self {
        Self {
            references: HashMap::new(),
        }
    }

    /// Create database with default geriatric norms
    pub fn with_defaults() -> Self {
        let mut db = Self::new();
        db.populate_defaults();
        db
    }

    /// Add a reference to the database
    pub fn add_reference(&mut self, reference: GeriatricReference) {
        self.references
            .entry(reference.metric_id.clone())
            .or_insert_with(Vec::new)
            .push(reference);
    }

    /// Lookup the best matching reference
    pub fn lookup(&self, metric: &str, age: u8, sex: Sex) -> Option<&GeriatricReference> {
        let refs = self.references.get(metric)?;

        refs.iter()
            .filter(|r| r.matches(metric, age, sex))
            .max_by_key(|r| {
                // Prefer sex-specific over combined
                if r.sex.is_some() { 2 } else { 1 }
            })
    }

    /// Get percentile for a value, accounting for frailty if provided
    pub fn get_percentile(
        &self,
        metric: &str,
        value: f64,
        age: u8,
        sex: Sex,
        frailty_index: Option<f64>,
    ) -> Option<f64> {
        let reference = self.lookup(metric, age, sex)?;
        Some(reference.percentile(value, frailty_index))
    }

    /// Get z-score for a value, accounting for frailty if provided
    pub fn get_z_score(
        &self,
        metric: &str,
        value: f64,
        age: u8,
        sex: Sex,
        frailty_index: Option<f64>,
    ) -> Option<f64> {
        let reference = self.lookup(metric, age, sex)?;
        Some(reference.z_score(value, frailty_index))
    }

    /// Populate with default geriatric normative data
    fn populate_defaults(&mut self) {
        // === Gait Velocity (m/s) ===
        // Critical functional measure, highly predictive of mortality
        // Frailty adjustments: frail individuals walk ~30% slower

        let gait_velocity_young_old_male = GeriatricReference::new(
            "gait_velocity".to_string(),
            GeriatricAgeGroup::YoungOld,
            Some(Sex::Male),
            1.15, 0.20,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.05, 1.0),   // Robust
            FrailtyAdjustment::new(0.15, 0.90),  // Pre-frail
            FrailtyAdjustment::new(0.30, 0.75),  // Frail
            FrailtyAdjustment::new(0.50, 0.60),  // Severely frail
        ]);
        self.add_reference(gait_velocity_young_old_male);

        let gait_velocity_young_old_female = GeriatricReference::new(
            "gait_velocity".to_string(),
            GeriatricAgeGroup::YoungOld,
            Some(Sex::Female),
            1.10, 0.18,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.05, 1.0),
            FrailtyAdjustment::new(0.15, 0.90),
            FrailtyAdjustment::new(0.30, 0.75),
            FrailtyAdjustment::new(0.50, 0.60),
        ]);
        self.add_reference(gait_velocity_young_old_female);

        let gait_velocity_middle_old_male = GeriatricReference::new(
            "gait_velocity".to_string(),
            GeriatricAgeGroup::MiddleOld,
            Some(Sex::Male),
            1.05, 0.22,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.10, 1.0),
            FrailtyAdjustment::new(0.20, 0.85),
            FrailtyAdjustment::new(0.35, 0.70),
            FrailtyAdjustment::new(0.55, 0.55),
        ]);
        self.add_reference(gait_velocity_middle_old_male);

        let gait_velocity_middle_old_female = GeriatricReference::new(
            "gait_velocity".to_string(),
            GeriatricAgeGroup::MiddleOld,
            Some(Sex::Female),
            0.98, 0.20,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.10, 1.0),
            FrailtyAdjustment::new(0.20, 0.85),
            FrailtyAdjustment::new(0.35, 0.70),
            FrailtyAdjustment::new(0.55, 0.55),
        ]);
        self.add_reference(gait_velocity_middle_old_female);

        let gait_velocity_old_old = GeriatricReference::new(
            "gait_velocity".to_string(),
            GeriatricAgeGroup::OldOld,
            None,
            0.85, 0.25,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.15, 1.0),
            FrailtyAdjustment::new(0.28, 0.80),
            FrailtyAdjustment::new(0.42, 0.65),
            FrailtyAdjustment::new(0.60, 0.50),
        ]);
        self.add_reference(gait_velocity_old_old);

        let gait_velocity_oldest = GeriatricReference::new(
            "gait_velocity".to_string(),
            GeriatricAgeGroup::Oldest,
            None,
            0.70, 0.28,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.20, 1.0),
            FrailtyAdjustment::new(0.35, 0.75),
            FrailtyAdjustment::new(0.50, 0.60),
        ]);
        self.add_reference(gait_velocity_oldest);

        // === Grip Strength (kg) ===
        // Sex-specific, declines with age and frailty

        let grip_young_old_male = GeriatricReference::new(
            "grip_strength".to_string(),
            GeriatricAgeGroup::YoungOld,
            Some(Sex::Male),
            38.0, 9.5,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.05, 1.0),
            FrailtyAdjustment::new(0.15, 0.85),
            FrailtyAdjustment::new(0.30, 0.70),
            FrailtyAdjustment::new(0.50, 0.55),
        ]);
        self.add_reference(grip_young_old_male);

        let grip_young_old_female = GeriatricReference::new(
            "grip_strength".to_string(),
            GeriatricAgeGroup::YoungOld,
            Some(Sex::Female),
            23.0, 5.5,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.05, 1.0),
            FrailtyAdjustment::new(0.15, 0.85),
            FrailtyAdjustment::new(0.30, 0.70),
            FrailtyAdjustment::new(0.50, 0.55),
        ]);
        self.add_reference(grip_young_old_female);

        let grip_middle_old_male = GeriatricReference::new(
            "grip_strength".to_string(),
            GeriatricAgeGroup::MiddleOld,
            Some(Sex::Male),
            33.0, 9.0,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.10, 1.0),
            FrailtyAdjustment::new(0.20, 0.80),
            FrailtyAdjustment::new(0.35, 0.65),
        ]);
        self.add_reference(grip_middle_old_male);

        let grip_middle_old_female = GeriatricReference::new(
            "grip_strength".to_string(),
            GeriatricAgeGroup::MiddleOld,
            Some(Sex::Female),
            20.0, 5.0,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.10, 1.0),
            FrailtyAdjustment::new(0.20, 0.80),
            FrailtyAdjustment::new(0.35, 0.65),
        ]);
        self.add_reference(grip_middle_old_female);

        let grip_old_old = GeriatricReference::new(
            "grip_strength".to_string(),
            GeriatricAgeGroup::OldOld,
            None,
            22.0, 7.0,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.15, 1.0),
            FrailtyAdjustment::new(0.28, 0.75),
            FrailtyAdjustment::new(0.42, 0.60),
        ]);
        self.add_reference(grip_old_old);

        let grip_oldest = GeriatricReference::new(
            "grip_strength".to_string(),
            GeriatricAgeGroup::Oldest,
            None,
            18.0, 6.0,
        );
        self.add_reference(grip_oldest);

        // === HRV SDNN (ms) ===
        // Cardiac autonomic function, decreases with age and frailty

        let hrv_young_old = GeriatricReference::new(
            "hrv_sdnn".to_string(),
            GeriatricAgeGroup::YoungOld,
            None,
            80.0, 28.0,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.05, 1.0),
            FrailtyAdjustment::new(0.15, 0.85),
            FrailtyAdjustment::new(0.30, 0.70),
        ]);
        self.add_reference(hrv_young_old);

        let hrv_middle_old = GeriatricReference::new(
            "hrv_sdnn".to_string(),
            GeriatricAgeGroup::MiddleOld,
            None,
            65.0, 25.0,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.10, 1.0),
            FrailtyAdjustment::new(0.20, 0.80),
            FrailtyAdjustment::new(0.35, 0.65),
        ]);
        self.add_reference(hrv_middle_old);

        let hrv_old_old = GeriatricReference::new(
            "hrv_sdnn".to_string(),
            GeriatricAgeGroup::OldOld,
            None,
            50.0, 20.0,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.15, 1.0),
            FrailtyAdjustment::new(0.28, 0.75),
            FrailtyAdjustment::new(0.42, 0.60),
        ]);
        self.add_reference(hrv_old_old);

        let hrv_oldest = GeriatricReference::new(
            "hrv_sdnn".to_string(),
            GeriatricAgeGroup::Oldest,
            None,
            40.0, 18.0,
        );
        self.add_reference(hrv_oldest);

        // === Timed Up and Go (seconds) ===
        // Fall risk predictor, higher is worse (frailty increases time)

        let tug_young_old = GeriatricReference::new(
            "timed_up_and_go".to_string(),
            GeriatricAgeGroup::YoungOld,
            None,
            9.0, 2.0,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.05, 1.0),
            FrailtyAdjustment::new(0.15, 1.20),  // Frailty increases time
            FrailtyAdjustment::new(0.30, 1.50),
            FrailtyAdjustment::new(0.50, 2.0),
        ]);
        self.add_reference(tug_young_old);

        let tug_middle_old = GeriatricReference::new(
            "timed_up_and_go".to_string(),
            GeriatricAgeGroup::MiddleOld,
            None,
            10.5, 2.5,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.10, 1.0),
            FrailtyAdjustment::new(0.20, 1.25),
            FrailtyAdjustment::new(0.35, 1.60),
            FrailtyAdjustment::new(0.55, 2.2),
        ]);
        self.add_reference(tug_middle_old);

        let tug_old_old = GeriatricReference::new(
            "timed_up_and_go".to_string(),
            GeriatricAgeGroup::OldOld,
            None,
            12.5, 3.5,
        ).with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.15, 1.0),
            FrailtyAdjustment::new(0.28, 1.35),
            FrailtyAdjustment::new(0.42, 1.80),
        ]);
        self.add_reference(tug_old_old);

        let tug_oldest = GeriatricReference::new(
            "timed_up_and_go".to_string(),
            GeriatricAgeGroup::Oldest,
            None,
            15.0, 5.0,
        );
        self.add_reference(tug_oldest);
    }

    /// Get all metrics in the database
    pub fn metrics(&self) -> Vec<String> {
        self.references.keys().cloned().collect()
    }

    /// Get all references for a specific metric
    pub fn get_all_references(&self, metric: &str) -> Option<&Vec<GeriatricReference>> {
        self.references.get(metric)
    }
}

impl Default for GeriatricNormativeDb {
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
    fn test_geriatric_age_group_from_age() {
        assert_eq!(
            GeriatricAgeGroup::from_age(70),
            Some(GeriatricAgeGroup::YoungOld)
        );
        assert_eq!(
            GeriatricAgeGroup::from_age(80),
            Some(GeriatricAgeGroup::MiddleOld)
        );
        assert_eq!(
            GeriatricAgeGroup::from_age(90),
            Some(GeriatricAgeGroup::OldOld)
        );
        assert_eq!(
            GeriatricAgeGroup::from_age(98),
            Some(GeriatricAgeGroup::Oldest)
        );
    }

    #[test]
    fn test_frailty_category() {
        assert_eq!(FrailtyCategory::from_index(0.05), FrailtyCategory::Robust);
        assert_eq!(FrailtyCategory::from_index(0.15), FrailtyCategory::PreFrail);
        assert_eq!(FrailtyCategory::from_index(0.30), FrailtyCategory::Frail);
        assert_eq!(FrailtyCategory::from_index(0.50), FrailtyCategory::SeverelyFrail);
    }

    #[test]
    fn test_frailty_adjustment() {
        let adj = FrailtyAdjustment::new(0.25, 0.80);
        assert_eq!(adj.frailty_index, 0.25);
        assert_eq!(adj.apply(100.0), 80.0);
    }

    #[test]
    fn test_geriatric_db_lookup() {
        let db = GeriatricNormativeDb::with_defaults();

        // Lookup gait velocity for 85-year-old male
        let reference = db.lookup("gait_velocity", 85, Sex::Male);
        assert!(reference.is_some());

        let r = reference.unwrap();
        assert_eq!(r.age_group, GeriatricAgeGroup::OldOld);
    }

    #[test]
    fn test_frailty_adjusted_percentile() {
        let db = GeriatricNormativeDb::with_defaults();

        // For a 70-year-old with gait velocity of 1.0 m/s
        // Without frailty context, evaluate against baseline
        let p_baseline = db.get_percentile("gait_velocity", 1.0, 70, Sex::Male, None);
        assert!(p_baseline.is_some());

        // With frailty index 0.3 (frail), same velocity should be better percentile
        let p_frail = db.get_percentile("gait_velocity", 1.0, 70, Sex::Male, Some(0.30));
        assert!(p_frail.is_some());

        // Frail individual with 1.0 m/s is performing better relative to frail norms
        assert!(p_frail.unwrap() > p_baseline.unwrap());
    }

    #[test]
    fn test_sex_specific_grip_strength() {
        let db = GeriatricNormativeDb::with_defaults();

        let male_ref = db.lookup("grip_strength", 70, Sex::Male);
        let female_ref = db.lookup("grip_strength", 70, Sex::Female);

        assert!(male_ref.is_some());
        assert!(female_ref.is_some());

        // Males should have higher mean grip strength
        assert!(male_ref.unwrap().mean > female_ref.unwrap().mean);
    }

    #[test]
    fn test_age_related_decline() {
        let db = GeriatricNormativeDb::with_defaults();

        let young_old_ref = db.lookup("hrv_sdnn", 70, Sex::Male);
        let old_old_ref = db.lookup("hrv_sdnn", 90, Sex::Male);

        assert!(young_old_ref.is_some());
        assert!(old_old_ref.is_some());

        // HRV should decline with age
        assert!(young_old_ref.unwrap().mean > old_old_ref.unwrap().mean);
    }

    #[test]
    fn test_tug_frailty_increases_time() {
        let reference = GeriatricReference::new(
            "timed_up_and_go".to_string(),
            GeriatricAgeGroup::YoungOld,
            None,
            9.0,
            2.0,
        )
        .with_frailty_adjustments(vec![
            FrailtyAdjustment::new(0.05, 1.0),
            FrailtyAdjustment::new(0.30, 1.5),
        ]);

        // Baseline
        let baseline = reference.frailty_adjusted_mean(0.05);
        assert_eq!(baseline, 9.0);

        // Frail (should increase time)
        let frail = reference.frailty_adjusted_mean(0.30);
        assert_eq!(frail, 13.5);
    }

    #[test]
    fn test_frailty_adjustment_clamping() {
        let adj = FrailtyAdjustment::new(1.5, 0.8); // FI > 1.0 should clamp
        assert_eq!(adj.frailty_index, 1.0);

        let adj2 = FrailtyAdjustment::new(-0.1, 0.8); // FI < 0.0 should clamp
        assert_eq!(adj2.frailty_index, 0.0);
    }
}
