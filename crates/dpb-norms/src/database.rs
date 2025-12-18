//! Normative database implementation
//!
//! Provides age and sex-stratified normative data for comparing individual
//! assessment results to population-based reference values.

use crate::{
    Demographics, DemographicsFilter, MetricDirection, MetricType, NormativeComparison,
    NormativeStats, NormsError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single entry in the normative table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormativeEntry {
    /// Metric type
    pub metric: MetricType,
    /// Age range (min, max inclusive)
    pub age_range: (u8, u8),
    /// Sex (None = both sexes combined)
    pub sex: Option<crate::demographics::Sex>,
    /// Normative statistics
    pub stats: NormativeStats,
}

impl NormativeEntry {
    /// Check if this entry matches the given demographics
    pub fn matches(&self, demographics: &Demographics) -> bool {
        // Check age range
        if demographics.age < self.age_range.0 || demographics.age > self.age_range.1 {
            return false;
        }

        // Check sex if specified
        if let Some(sex) = self.sex {
            if demographics.sex != sex {
                return false;
            }
        }

        true
    }

    /// Get the specificity score (higher = more specific match)
    pub fn specificity(&self) -> u8 {
        let mut score = 0;

        // Narrower age range is more specific
        let age_span = self.age_range.1 - self.age_range.0;
        if age_span < 10 {
            score += 3;
        } else if age_span < 20 {
            score += 2;
        } else if age_span < 40 {
            score += 1;
        }

        // Sex-specific is more specific than combined
        if self.sex.is_some() {
            score += 2;
        }

        score
    }
}

/// Table of normative entries for a specific metric
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NormativeTable {
    /// All entries for this metric
    entries: Vec<NormativeEntry>,
}

impl NormativeTable {
    /// Create a new empty table
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an entry to the table
    pub fn add_entry(&mut self, entry: NormativeEntry) {
        self.entries.push(entry);
    }

    /// Find the best matching entry for given demographics
    pub fn find_best_match(&self, demographics: &Demographics) -> Option<&NormativeEntry> {
        self.entries
            .iter()
            .filter(|e| e.matches(demographics))
            .max_by_key(|e| e.specificity())
    }

    /// Get all matching entries
    pub fn find_all_matches(&self, demographics: &Demographics) -> Vec<&NormativeEntry> {
        self.entries
            .iter()
            .filter(|e| e.matches(demographics))
            .collect()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Comprehensive normative database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormativeDatabase {
    /// Name/version of the database
    pub name: String,
    /// Description
    pub description: String,
    /// Tables for each metric
    tables: HashMap<MetricType, NormativeTable>,
    /// Default reliability coefficients (ICC) for MDC calculation
    reliability: HashMap<MetricType, f64>,
}

impl NormativeDatabase {
    /// Create a new empty database
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            tables: HashMap::new(),
            reliability: HashMap::new(),
        }
    }

    /// Create a database with default normative data
    pub fn with_defaults() -> Self {
        let mut db = Self::new(
            "DPB Default Norms",
            "Default normative database for Delta-Predictive Biosensing assessments",
        );
        db.populate_defaults();
        db
    }

    /// Add a normative entry
    pub fn add_entry(&mut self, entry: NormativeEntry) {
        self.tables
            .entry(entry.metric)
            .or_insert_with(NormativeTable::new)
            .add_entry(entry);
    }

    /// Set reliability coefficient for a metric
    pub fn set_reliability(&mut self, metric: MetricType, icc: f64) {
        self.reliability.insert(metric, icc);
    }

    /// Get percentile for a value
    pub fn percentile(
        &self,
        metric: MetricType,
        value: f64,
        demographics: &Demographics,
    ) -> Result<f64> {
        let stats = self.get_stats(metric, demographics)?;
        Ok(stats.percentile(value))
    }

    /// Get z-score for a value
    pub fn z_score(
        &self,
        metric: MetricType,
        value: f64,
        demographics: &Demographics,
    ) -> Result<f64> {
        let stats = self.get_stats(metric, demographics)?;
        Ok(stats.z_score(value))
    }

    /// Get reference range (5th-95th percentile)
    pub fn reference_range(
        &self,
        metric: MetricType,
        demographics: &Demographics,
    ) -> Result<(f64, f64)> {
        let stats = self.get_stats(metric, demographics)?;
        Ok(stats.reference_range())
    }

    /// Get minimal detectable change
    pub fn minimal_detectable_change(&self, metric: MetricType) -> Result<f64> {
        // Use default ICC if not specified
        let icc = self.reliability.get(&metric).copied().unwrap_or(0.85);

        // Get average stats across age groups
        let table = self.tables.get(&metric)
            .ok_or_else(|| NormsError::MetricNotFound(format!("{:?}", metric)))?;

        if table.is_empty() {
            return Err(NormsError::InsufficientData);
        }

        // Calculate average SD
        let avg_sd: f64 = table.entries.iter()
            .map(|e| e.stats.std_dev)
            .sum::<f64>() / table.entries.len() as f64;

        // MDC95 = SEM * sqrt(2) * 1.96
        let sem = avg_sd * (1.0 - icc).sqrt();
        let mdc95 = sem * std::f64::consts::SQRT_2 * 1.96;

        Ok(mdc95)
    }

    /// Get normative statistics for a metric
    pub fn get_stats(
        &self,
        metric: MetricType,
        demographics: &Demographics,
    ) -> Result<&NormativeStats> {
        let table = self.tables.get(&metric)
            .ok_or_else(|| NormsError::MetricNotFound(format!("{:?}", metric)))?;

        let entry = table.find_best_match(demographics)
            .ok_or_else(|| NormsError::DemographicsOutOfRange(
                format!("No norms for age {} {:?}", demographics.age, demographics.sex)
            ))?;

        Ok(&entry.stats)
    }

    /// Compare a value to normative data
    pub fn compare(
        &self,
        metric: MetricType,
        value: f64,
        demographics: &Demographics,
    ) -> Result<NormativeComparison> {
        let stats = self.get_stats(metric, demographics)?;
        let direction = metric.direction();

        Ok(NormativeComparison {
            metric,
            value,
            percentile: stats.percentile(value),
            z_score: stats.z_score(value),
            impairment: stats.impairment_level(value, direction),
            reference_range: stats.reference_range(),
            mdc95: stats.mdc95,
            demographics: demographics.clone(),
        })
    }

    /// Get all supported metrics
    pub fn supported_metrics(&self) -> Vec<MetricType> {
        self.tables.keys().copied().collect()
    }

    /// Check if metric is supported
    pub fn has_metric(&self, metric: MetricType) -> bool {
        self.tables.contains_key(&metric)
    }

    /// Populate with default normative data
    fn populate_defaults(&mut self) {
        // === Cognitive Metrics ===

        // Simple Reaction Time (ms) - based on literature
        self.add_age_sex_norms(
            MetricType::SimpleReactionTime,
            &[
                ((18, 29), None, 250.0, 40.0, 500),
                ((30, 39), None, 260.0, 42.0, 450),
                ((40, 49), None, 275.0, 45.0, 400),
                ((50, 59), None, 290.0, 50.0, 380),
                ((60, 69), None, 310.0, 55.0, 350),
                ((70, 79), None, 340.0, 65.0, 300),
                ((80, 89), None, 380.0, 80.0, 200),
            ],
        );
        self.set_reliability(MetricType::SimpleReactionTime, 0.85);

        // Choice Reaction Time (ms)
        self.add_age_sex_norms(
            MetricType::ChoiceReactionTime,
            &[
                ((18, 29), None, 350.0, 50.0, 500),
                ((30, 39), None, 365.0, 52.0, 450),
                ((40, 49), None, 385.0, 58.0, 400),
                ((50, 59), None, 410.0, 65.0, 380),
                ((60, 69), None, 450.0, 75.0, 350),
                ((70, 79), None, 500.0, 90.0, 300),
                ((80, 89), None, 560.0, 110.0, 200),
            ],
        );
        self.set_reliability(MetricType::ChoiceReactionTime, 0.82);

        // Trail Making A (seconds)
        self.add_age_sex_norms(
            MetricType::TrailMakingA,
            &[
                ((18, 29), None, 22.0, 7.0, 500),
                ((30, 39), None, 24.0, 8.0, 450),
                ((40, 49), None, 28.0, 9.0, 400),
                ((50, 59), None, 32.0, 10.0, 380),
                ((60, 69), None, 38.0, 12.0, 350),
                ((70, 79), None, 48.0, 16.0, 300),
                ((80, 89), None, 62.0, 22.0, 200),
            ],
        );
        self.set_reliability(MetricType::TrailMakingA, 0.79);

        // Trail Making B (seconds)
        self.add_age_sex_norms(
            MetricType::TrailMakingB,
            &[
                ((18, 29), None, 48.0, 15.0, 500),
                ((30, 39), None, 52.0, 16.0, 450),
                ((40, 49), None, 60.0, 18.0, 400),
                ((50, 59), None, 72.0, 22.0, 380),
                ((60, 69), None, 90.0, 28.0, 350),
                ((70, 79), None, 115.0, 38.0, 300),
                ((80, 89), None, 150.0, 55.0, 200),
            ],
        );
        self.set_reliability(MetricType::TrailMakingB, 0.82);

        // MoCA Total (0-30)
        self.add_age_sex_norms(
            MetricType::MocaTotal,
            &[
                ((18, 59), None, 27.5, 2.0, 1000),
                ((60, 69), None, 26.5, 2.2, 800),
                ((70, 79), None, 25.5, 2.5, 600),
                ((80, 89), None, 24.0, 3.0, 400),
            ],
        );
        self.set_reliability(MetricType::MocaTotal, 0.92);

        // === Motor Metrics ===

        // Gait Velocity (m/s) - sex-stratified
        self.add_age_sex_norms(
            MetricType::GaitVelocity,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 1.35, 0.15, 300),
                ((18, 29), Some(crate::demographics::Sex::Female), 1.30, 0.14, 300),
                ((30, 39), Some(crate::demographics::Sex::Male), 1.32, 0.15, 280),
                ((30, 39), Some(crate::demographics::Sex::Female), 1.28, 0.14, 280),
                ((40, 49), Some(crate::demographics::Sex::Male), 1.28, 0.16, 260),
                ((40, 49), Some(crate::demographics::Sex::Female), 1.24, 0.15, 260),
                ((50, 59), Some(crate::demographics::Sex::Male), 1.22, 0.18, 250),
                ((50, 59), Some(crate::demographics::Sex::Female), 1.18, 0.16, 250),
                ((60, 69), Some(crate::demographics::Sex::Male), 1.15, 0.20, 240),
                ((60, 69), Some(crate::demographics::Sex::Female), 1.10, 0.18, 240),
                ((70, 79), Some(crate::demographics::Sex::Male), 1.05, 0.22, 200),
                ((70, 79), Some(crate::demographics::Sex::Female), 0.98, 0.20, 200),
                ((80, 89), Some(crate::demographics::Sex::Male), 0.90, 0.25, 150),
                ((80, 89), Some(crate::demographics::Sex::Female), 0.82, 0.22, 150),
            ],
        );
        self.set_reliability(MetricType::GaitVelocity, 0.95);

        // Timed Up and Go (seconds)
        self.add_age_sex_norms(
            MetricType::TimedUpAndGo,
            &[
                ((18, 39), None, 7.0, 1.2, 400),
                ((40, 49), None, 7.5, 1.3, 350),
                ((50, 59), None, 8.0, 1.5, 320),
                ((60, 69), None, 9.0, 2.0, 300),
                ((70, 79), None, 10.5, 2.5, 250),
                ((80, 89), None, 12.5, 3.5, 180),
            ],
        );
        self.set_reliability(MetricType::TimedUpAndGo, 0.97);

        // Grip Strength (kg) - highly sex-stratified
        self.add_age_sex_norms(
            MetricType::GripStrength,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 48.0, 8.0, 300),
                ((18, 29), Some(crate::demographics::Sex::Female), 30.0, 5.5, 300),
                ((30, 39), Some(crate::demographics::Sex::Male), 47.0, 8.5, 280),
                ((30, 39), Some(crate::demographics::Sex::Female), 29.0, 5.5, 280),
                ((40, 49), Some(crate::demographics::Sex::Male), 45.0, 9.0, 260),
                ((40, 49), Some(crate::demographics::Sex::Female), 28.0, 5.5, 260),
                ((50, 59), Some(crate::demographics::Sex::Male), 42.0, 9.0, 250),
                ((50, 59), Some(crate::demographics::Sex::Female), 26.0, 5.5, 250),
                ((60, 69), Some(crate::demographics::Sex::Male), 38.0, 9.5, 240),
                ((60, 69), Some(crate::demographics::Sex::Female), 23.0, 5.5, 240),
                ((70, 79), Some(crate::demographics::Sex::Male), 33.0, 9.0, 200),
                ((70, 79), Some(crate::demographics::Sex::Female), 20.0, 5.0, 200),
                ((80, 89), Some(crate::demographics::Sex::Male), 27.0, 8.0, 150),
                ((80, 89), Some(crate::demographics::Sex::Female), 16.0, 4.5, 150),
            ],
        );
        self.set_reliability(MetricType::GripStrength, 0.95);

        // === Balance Metrics ===

        // Sway Area (cm²)
        self.add_age_sex_norms(
            MetricType::SwayArea,
            &[
                ((18, 39), None, 2.5, 1.0, 400),
                ((40, 49), None, 3.0, 1.2, 350),
                ((50, 59), None, 3.5, 1.5, 320),
                ((60, 69), None, 4.5, 2.0, 300),
                ((70, 79), None, 6.0, 2.8, 250),
                ((80, 89), None, 8.5, 4.0, 180),
            ],
        );
        self.set_reliability(MetricType::SwayArea, 0.80);

        // Berg Balance Scale (0-56)
        self.add_age_sex_norms(
            MetricType::BergBalanceScale,
            &[
                ((60, 69), None, 54.0, 2.0, 300),
                ((70, 79), None, 52.0, 3.0, 250),
                ((80, 89), None, 48.0, 5.0, 180),
            ],
        );
        self.set_reliability(MetricType::BergBalanceScale, 0.98);

        // === Physiological Metrics ===

        // HRV SDNN (ms)
        self.add_age_sex_norms(
            MetricType::HrvSdnn,
            &[
                ((18, 29), None, 140.0, 40.0, 400),
                ((30, 39), None, 125.0, 38.0, 380),
                ((40, 49), None, 110.0, 35.0, 350),
                ((50, 59), None, 95.0, 32.0, 320),
                ((60, 69), None, 80.0, 28.0, 300),
                ((70, 79), None, 65.0, 25.0, 250),
                ((80, 89), None, 50.0, 20.0, 180),
            ],
        );
        self.set_reliability(MetricType::HrvSdnn, 0.85);

        // HRV RMSSD (ms)
        self.add_age_sex_norms(
            MetricType::HrvRmssd,
            &[
                ((18, 29), None, 45.0, 20.0, 400),
                ((30, 39), None, 38.0, 18.0, 380),
                ((40, 49), None, 32.0, 15.0, 350),
                ((50, 59), None, 26.0, 12.0, 320),
                ((60, 69), None, 22.0, 10.0, 300),
                ((70, 79), None, 18.0, 8.0, 250),
                ((80, 89), None, 14.0, 6.0, 180),
            ],
        );
        self.set_reliability(MetricType::HrvRmssd, 0.82);

        // === Sleep Metrics ===

        // Total Sleep Time (min)
        self.add_age_sex_norms(
            MetricType::TotalSleepTime,
            &[
                ((18, 29), None, 450.0, 45.0, 400),
                ((30, 39), None, 440.0, 50.0, 380),
                ((40, 49), None, 425.0, 55.0, 350),
                ((50, 59), None, 410.0, 60.0, 320),
                ((60, 69), None, 390.0, 65.0, 300),
                ((70, 79), None, 370.0, 70.0, 250),
                ((80, 89), None, 350.0, 75.0, 180),
            ],
        );
        self.set_reliability(MetricType::TotalSleepTime, 0.75);

        // Sleep Efficiency (%)
        self.add_age_sex_norms(
            MetricType::SleepEfficiency,
            &[
                ((18, 29), None, 92.0, 5.0, 400),
                ((30, 39), None, 90.0, 6.0, 380),
                ((40, 49), None, 88.0, 7.0, 350),
                ((50, 59), None, 85.0, 8.0, 320),
                ((60, 69), None, 82.0, 9.0, 300),
                ((70, 79), None, 78.0, 10.0, 250),
                ((80, 89), None, 74.0, 12.0, 180),
            ],
        );
        self.set_reliability(MetricType::SleepEfficiency, 0.78);
    }

    /// Helper to add multiple normative entries
    fn add_age_sex_norms(
        &mut self,
        metric: MetricType,
        data: &[((u8, u8), Option<crate::demographics::Sex>, f64, f64, usize)],
    ) {
        for &(age_range, sex, mean, std_dev, n) in data {
            let icc = self.reliability.get(&metric).copied().unwrap_or(0.85);
            let stats = NormativeStats::with_reliability(mean, std_dev, n, icc);

            self.add_entry(NormativeEntry {
                metric,
                age_range,
                sex,
                stats,
            });
        }
    }
}

impl Default for NormativeDatabase {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demographics::Sex;

    #[test]
    fn test_database_creation() {
        let db = NormativeDatabase::with_defaults();
        assert!(!db.name.is_empty());
        assert!(db.has_metric(MetricType::SimpleReactionTime));
        assert!(db.has_metric(MetricType::GaitVelocity));
    }

    #[test]
    fn test_percentile_lookup() {
        let db = NormativeDatabase::with_defaults();
        let demo = Demographics::new(30, Sex::Male);

        // Average RT should be around 50th percentile
        let p = db.percentile(MetricType::SimpleReactionTime, 260.0, &demo).unwrap();
        assert!((p - 50.0).abs() < 10.0);

        // Fast RT should be higher percentile (but RT is lower-is-better)
        let p_fast = db.percentile(MetricType::SimpleReactionTime, 200.0, &demo).unwrap();
        assert!(p_fast < 20.0); // Lower percentile for faster time
    }

    #[test]
    fn test_z_score_lookup() {
        let db = NormativeDatabase::with_defaults();
        let demo = Demographics::new(30, Sex::Male);

        // Mean should have z = 0
        let z = db.z_score(MetricType::SimpleReactionTime, 260.0, &demo).unwrap();
        assert!(z.abs() < 0.5);

        // 1 SD above mean
        let z_high = db.z_score(MetricType::SimpleReactionTime, 302.0, &demo).unwrap();
        assert!((z_high - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_sex_stratified_norms() {
        let db = NormativeDatabase::with_defaults();

        let male = Demographics::new(50, Sex::Male);
        let female = Demographics::new(50, Sex::Female);

        // Grip strength should differ by sex
        let male_grip = db.get_stats(MetricType::GripStrength, &male).unwrap();
        let female_grip = db.get_stats(MetricType::GripStrength, &female).unwrap();

        assert!(male_grip.mean > female_grip.mean);
    }

    #[test]
    fn test_age_effects() {
        let db = NormativeDatabase::with_defaults();

        let young = Demographics::new(25, Sex::Male);
        let old = Demographics::new(75, Sex::Male);

        // RT should be faster in young
        let young_rt = db.get_stats(MetricType::SimpleReactionTime, &young).unwrap();
        let old_rt = db.get_stats(MetricType::SimpleReactionTime, &old).unwrap();

        assert!(young_rt.mean < old_rt.mean);
    }

    #[test]
    fn test_mdc_calculation() {
        let db = NormativeDatabase::with_defaults();

        let mdc = db.minimal_detectable_change(MetricType::GaitVelocity).unwrap();

        // MDC should be positive and reasonable
        assert!(mdc > 0.0);
        assert!(mdc < 0.5); // Should be less than 0.5 m/s for gait velocity
    }

    #[test]
    fn test_compare() {
        let db = NormativeDatabase::with_defaults();
        let demo = Demographics::new(65, Sex::Female);

        let result = db.compare(MetricType::GaitVelocity, 0.95, &demo).unwrap();

        assert_eq!(result.metric, MetricType::GaitVelocity);
        assert_eq!(result.value, 0.95);
        assert!(result.percentile >= 0.0 && result.percentile <= 100.0);
        assert!(result.reference_range.0 < result.reference_range.1);
    }

    #[test]
    fn test_reference_range() {
        let db = NormativeDatabase::with_defaults();
        let demo = Demographics::new(50, Sex::Male);

        let (low, high) = db.reference_range(MetricType::GripStrength, &demo).unwrap();

        // Reference range should contain the mean
        let stats = db.get_stats(MetricType::GripStrength, &demo).unwrap();
        assert!(low < stats.mean);
        assert!(high > stats.mean);
    }
}
