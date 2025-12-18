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

        // === Pain/Sensory Metrics (Phase F) ===

        // Pressure Pain Threshold (kPa) - sex-stratified, highly variable
        self.add_age_sex_norms(
            MetricType::PressurePainThreshold,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 450.0, 120.0, 200),
                ((18, 29), Some(crate::demographics::Sex::Female), 350.0, 100.0, 200),
                ((30, 39), Some(crate::demographics::Sex::Male), 440.0, 115.0, 180),
                ((30, 39), Some(crate::demographics::Sex::Female), 340.0, 95.0, 180),
                ((40, 49), Some(crate::demographics::Sex::Male), 420.0, 110.0, 160),
                ((40, 49), Some(crate::demographics::Sex::Female), 330.0, 90.0, 160),
                ((50, 59), Some(crate::demographics::Sex::Male), 400.0, 105.0, 150),
                ((50, 59), Some(crate::demographics::Sex::Female), 315.0, 85.0, 150),
                ((60, 69), Some(crate::demographics::Sex::Male), 380.0, 100.0, 140),
                ((60, 69), Some(crate::demographics::Sex::Female), 300.0, 80.0, 140),
                ((70, 89), Some(crate::demographics::Sex::Male), 360.0, 95.0, 100),
                ((70, 89), Some(crate::demographics::Sex::Female), 285.0, 75.0, 100),
            ],
        );
        self.set_reliability(MetricType::PressurePainThreshold, 0.88);

        // Pain Tolerance (kPa) - higher than threshold
        self.add_age_sex_norms(
            MetricType::PainTolerance,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 650.0, 150.0, 200),
                ((18, 29), Some(crate::demographics::Sex::Female), 520.0, 130.0, 200),
                ((30, 49), Some(crate::demographics::Sex::Male), 620.0, 145.0, 180),
                ((30, 49), Some(crate::demographics::Sex::Female), 500.0, 125.0, 180),
                ((50, 69), Some(crate::demographics::Sex::Male), 580.0, 140.0, 150),
                ((50, 69), Some(crate::demographics::Sex::Female), 470.0, 120.0, 150),
                ((70, 89), Some(crate::demographics::Sex::Male), 540.0, 135.0, 100),
                ((70, 89), Some(crate::demographics::Sex::Female), 440.0, 115.0, 100),
            ],
        );
        self.set_reliability(MetricType::PainTolerance, 0.85);

        // CPM Effect (% pain reduction) - conditioned pain modulation
        self.add_age_sex_norms(
            MetricType::CpmEffect,
            &[
                ((18, 39), None, 35.0, 15.0, 200),
                ((40, 59), None, 30.0, 14.0, 180),
                ((60, 79), None, 22.0, 12.0, 150),
                ((80, 89), None, 15.0, 10.0, 80),
            ],
        );
        self.set_reliability(MetricType::CpmEffect, 0.75);

        // Vibration Threshold (μm) - increases with age
        self.add_age_sex_norms(
            MetricType::VibrationThreshold,
            &[
                ((18, 29), None, 0.5, 0.2, 300),
                ((30, 39), None, 0.6, 0.25, 280),
                ((40, 49), None, 0.8, 0.3, 260),
                ((50, 59), None, 1.2, 0.4, 250),
                ((60, 69), None, 2.0, 0.6, 240),
                ((70, 79), None, 3.5, 1.0, 200),
                ((80, 89), None, 5.5, 1.5, 150),
            ],
        );
        self.set_reliability(MetricType::VibrationThreshold, 0.82);

        // Joint Position Error (degrees) - proprioception
        self.add_age_sex_norms(
            MetricType::JointPositionError,
            &[
                ((18, 39), None, 2.5, 1.0, 300),
                ((40, 49), None, 3.0, 1.2, 280),
                ((50, 59), None, 3.5, 1.4, 260),
                ((60, 69), None, 4.5, 1.8, 240),
                ((70, 79), None, 6.0, 2.2, 200),
                ((80, 89), None, 8.0, 3.0, 150),
            ],
        );
        self.set_reliability(MetricType::JointPositionError, 0.78);

        // === Additional Balance Metrics ===

        // Sway Velocity (cm/s)
        self.add_age_sex_norms(
            MetricType::SwayVelocity,
            &[
                ((18, 39), None, 0.8, 0.25, 400),
                ((40, 49), None, 0.95, 0.30, 350),
                ((50, 59), None, 1.1, 0.35, 320),
                ((60, 69), None, 1.4, 0.45, 300),
                ((70, 79), None, 1.8, 0.55, 250),
                ((80, 89), None, 2.4, 0.70, 180),
            ],
        );
        self.set_reliability(MetricType::SwayVelocity, 0.82);

        // Sway Path Length (cm)
        self.add_age_sex_norms(
            MetricType::SwayPathLength,
            &[
                ((18, 39), None, 25.0, 8.0, 400),
                ((40, 49), None, 30.0, 10.0, 350),
                ((50, 59), None, 36.0, 12.0, 320),
                ((60, 69), None, 45.0, 15.0, 300),
                ((70, 79), None, 58.0, 20.0, 250),
                ((80, 89), None, 75.0, 28.0, 180),
            ],
        );
        self.set_reliability(MetricType::SwayPathLength, 0.80);

        // Limits of Stability - Max Excursion (%)
        self.add_age_sex_norms(
            MetricType::LosMaxExcursion,
            &[
                ((18, 39), None, 95.0, 8.0, 350),
                ((40, 49), None, 92.0, 9.0, 320),
                ((50, 59), None, 88.0, 10.0, 300),
                ((60, 69), None, 82.0, 12.0, 280),
                ((70, 79), None, 74.0, 14.0, 220),
                ((80, 89), None, 65.0, 16.0, 150),
            ],
        );
        self.set_reliability(MetricType::LosMaxExcursion, 0.85);

        // Limits of Stability - Reaction Time (ms)
        self.add_age_sex_norms(
            MetricType::LosReactionTime,
            &[
                ((18, 39), None, 450.0, 80.0, 350),
                ((40, 49), None, 480.0, 90.0, 320),
                ((50, 59), None, 520.0, 100.0, 300),
                ((60, 69), None, 580.0, 120.0, 280),
                ((70, 79), None, 660.0, 150.0, 220),
                ((80, 89), None, 780.0, 180.0, 150),
            ],
        );
        self.set_reliability(MetricType::LosReactionTime, 0.82);

        // === Additional Motor Metrics ===

        // Rate of Force Development (N/s)
        self.add_age_sex_norms(
            MetricType::RateOfForceDevelopment,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 1200.0, 300.0, 200),
                ((18, 29), Some(crate::demographics::Sex::Female), 750.0, 200.0, 200),
                ((30, 39), Some(crate::demographics::Sex::Male), 1100.0, 280.0, 180),
                ((30, 39), Some(crate::demographics::Sex::Female), 700.0, 185.0, 180),
                ((40, 49), Some(crate::demographics::Sex::Male), 950.0, 260.0, 160),
                ((40, 49), Some(crate::demographics::Sex::Female), 620.0, 170.0, 160),
                ((50, 59), Some(crate::demographics::Sex::Male), 800.0, 240.0, 150),
                ((50, 59), Some(crate::demographics::Sex::Female), 530.0, 155.0, 150),
                ((60, 69), Some(crate::demographics::Sex::Male), 650.0, 220.0, 140),
                ((60, 69), Some(crate::demographics::Sex::Female), 440.0, 140.0, 140),
                ((70, 89), Some(crate::demographics::Sex::Male), 480.0, 180.0, 100),
                ((70, 89), Some(crate::demographics::Sex::Female), 320.0, 120.0, 100),
            ],
        );
        self.set_reliability(MetricType::RateOfForceDevelopment, 0.90);

        // Tapping Frequency (Hz)
        self.add_age_sex_norms(
            MetricType::TappingFrequency,
            &[
                ((18, 29), None, 6.0, 0.8, 300),
                ((30, 39), None, 5.8, 0.9, 280),
                ((40, 49), None, 5.5, 0.9, 260),
                ((50, 59), None, 5.2, 1.0, 250),
                ((60, 69), None, 4.8, 1.1, 240),
                ((70, 79), None, 4.3, 1.2, 200),
                ((80, 89), None, 3.7, 1.3, 150),
            ],
        );
        self.set_reliability(MetricType::TappingFrequency, 0.88);

        // Stride Length (m) - sex-stratified due to height differences
        self.add_age_sex_norms(
            MetricType::StrideLength,
            &[
                ((18, 39), Some(crate::demographics::Sex::Male), 1.50, 0.12, 250),
                ((18, 39), Some(crate::demographics::Sex::Female), 1.35, 0.11, 250),
                ((40, 59), Some(crate::demographics::Sex::Male), 1.45, 0.13, 220),
                ((40, 59), Some(crate::demographics::Sex::Female), 1.30, 0.12, 220),
                ((60, 79), Some(crate::demographics::Sex::Male), 1.35, 0.15, 180),
                ((60, 79), Some(crate::demographics::Sex::Female), 1.20, 0.14, 180),
                ((80, 89), Some(crate::demographics::Sex::Male), 1.20, 0.18, 120),
                ((80, 89), Some(crate::demographics::Sex::Female), 1.05, 0.16, 120),
            ],
        );
        self.set_reliability(MetricType::StrideLength, 0.92);

        // Cadence (steps/min)
        self.add_age_sex_norms(
            MetricType::Cadence,
            &[
                ((18, 39), None, 115.0, 10.0, 400),
                ((40, 49), None, 112.0, 11.0, 350),
                ((50, 59), None, 108.0, 12.0, 320),
                ((60, 69), None, 104.0, 13.0, 300),
                ((70, 79), None, 98.0, 14.0, 250),
                ((80, 89), None, 90.0, 16.0, 180),
            ],
        );
        self.set_reliability(MetricType::Cadence, 0.90);

        // === Additional Physiological Metrics ===

        // Heart Rate (bpm) - resting
        self.add_age_sex_norms(
            MetricType::HeartRate,
            &[
                ((18, 29), None, 68.0, 10.0, 500),
                ((30, 39), None, 70.0, 10.0, 480),
                ((40, 49), None, 72.0, 11.0, 450),
                ((50, 59), None, 74.0, 11.0, 420),
                ((60, 69), None, 72.0, 10.0, 400),
                ((70, 79), None, 70.0, 10.0, 350),
                ((80, 89), None, 72.0, 12.0, 250),
            ],
        );
        self.set_reliability(MetricType::HeartRate, 0.92);

        // Respiratory Rate (breaths/min)
        self.add_age_sex_norms(
            MetricType::RespiratoryRate,
            &[
                ((18, 39), None, 14.0, 2.5, 400),
                ((40, 59), None, 15.0, 3.0, 380),
                ((60, 79), None, 16.0, 3.5, 300),
                ((80, 89), None, 18.0, 4.0, 200),
            ],
        );
        self.set_reliability(MetricType::RespiratoryRate, 0.75);

        // SpO2 (%) - oxygen saturation
        self.add_age_sex_norms(
            MetricType::SpO2,
            &[
                ((18, 59), None, 97.5, 1.0, 500),
                ((60, 69), None, 97.0, 1.2, 400),
                ((70, 79), None, 96.5, 1.5, 300),
                ((80, 89), None, 95.5, 2.0, 200),
            ],
        );
        self.set_reliability(MetricType::SpO2, 0.85);

        // Blood Pressure Systolic (mmHg)
        self.add_age_sex_norms(
            MetricType::BpSystolic,
            &[
                ((18, 29), None, 115.0, 10.0, 500),
                ((30, 39), None, 118.0, 11.0, 480),
                ((40, 49), None, 122.0, 12.0, 450),
                ((50, 59), None, 128.0, 14.0, 420),
                ((60, 69), None, 132.0, 16.0, 400),
                ((70, 79), None, 138.0, 18.0, 350),
                ((80, 89), None, 142.0, 20.0, 250),
            ],
        );
        self.set_reliability(MetricType::BpSystolic, 0.88);

        // Blood Pressure Diastolic (mmHg)
        self.add_age_sex_norms(
            MetricType::BpDiastolic,
            &[
                ((18, 29), None, 72.0, 8.0, 500),
                ((30, 39), None, 75.0, 8.0, 480),
                ((40, 49), None, 78.0, 9.0, 450),
                ((50, 59), None, 80.0, 9.0, 420),
                ((60, 69), None, 78.0, 10.0, 400),
                ((70, 79), None, 76.0, 10.0, 350),
                ((80, 89), None, 74.0, 11.0, 250),
            ],
        );
        self.set_reliability(MetricType::BpDiastolic, 0.86);

        // HRV pNN50 (%)
        self.add_age_sex_norms(
            MetricType::HrvPnn50,
            &[
                ((18, 29), None, 20.0, 12.0, 400),
                ((30, 39), None, 16.0, 10.0, 380),
                ((40, 49), None, 12.0, 8.0, 350),
                ((50, 59), None, 8.0, 6.0, 320),
                ((60, 69), None, 5.0, 4.0, 300),
                ((70, 79), None, 3.0, 3.0, 250),
                ((80, 89), None, 2.0, 2.0, 180),
            ],
        );
        self.set_reliability(MetricType::HrvPnn50, 0.80);

        // === Additional Cognitive Metrics ===

        // Digit Span Forward
        self.add_age_sex_norms(
            MetricType::DigitSpanForward,
            &[
                ((18, 39), None, 7.0, 1.2, 400),
                ((40, 59), None, 6.8, 1.3, 350),
                ((60, 79), None, 6.2, 1.4, 280),
                ((80, 89), None, 5.5, 1.5, 180),
            ],
        );
        self.set_reliability(MetricType::DigitSpanForward, 0.80);

        // Digit Span Backward
        self.add_age_sex_norms(
            MetricType::DigitSpanBackward,
            &[
                ((18, 39), None, 5.5, 1.3, 400),
                ((40, 59), None, 5.2, 1.4, 350),
                ((60, 79), None, 4.5, 1.5, 280),
                ((80, 89), None, 3.8, 1.6, 180),
            ],
        );
        self.set_reliability(MetricType::DigitSpanBackward, 0.78);

        // Verbal Fluency (words/min)
        self.add_age_sex_norms(
            MetricType::VerbalFluency,
            &[
                ((18, 39), None, 18.0, 5.0, 400),
                ((40, 59), None, 16.0, 5.0, 350),
                ((60, 79), None, 13.0, 4.5, 280),
                ((80, 89), None, 10.0, 4.0, 180),
            ],
        );
        self.set_reliability(MetricType::VerbalFluency, 0.82);

        // Stroop Interference (ms)
        self.add_age_sex_norms(
            MetricType::StroopInterference,
            &[
                ((18, 29), None, 40.0, 20.0, 400),
                ((30, 39), None, 45.0, 22.0, 380),
                ((40, 49), None, 55.0, 25.0, 350),
                ((50, 59), None, 70.0, 30.0, 320),
                ((60, 69), None, 90.0, 38.0, 280),
                ((70, 79), None, 120.0, 50.0, 220),
                ((80, 89), None, 160.0, 65.0, 150),
            ],
        );
        self.set_reliability(MetricType::StroopInterference, 0.75);

        // === Tremor Metrics ===

        // Tremor Amplitude (mm) - physiological tremor increases with age
        self.add_age_sex_norms(
            MetricType::TremorAmplitude,
            &[
                ((18, 39), None, 0.15, 0.08, 350),
                ((40, 59), None, 0.22, 0.12, 320),
                ((60, 79), None, 0.35, 0.18, 280),
                ((80, 89), None, 0.50, 0.25, 180),
            ],
        );
        self.set_reliability(MetricType::TremorAmplitude, 0.85);

        // Tremor Frequency (Hz) - typically 8-12 Hz physiological
        self.add_age_sex_norms(
            MetricType::TremorFrequency,
            &[
                ((18, 39), None, 10.0, 1.5, 350),
                ((40, 59), None, 9.5, 1.8, 320),
                ((60, 79), None, 8.5, 2.0, 280),
                ((80, 89), None, 7.5, 2.2, 180),
            ],
        );
        self.set_reliability(MetricType::TremorFrequency, 0.88);
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
