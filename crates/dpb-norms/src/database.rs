//! Normative database implementation
//!
//! Provides age and sex-stratified normative data for comparing individual
//! assessment results to population-based reference values.
//!
//! # ⚠️ The reference values in this module are ILLUSTRATIVE
//!
//! They are placeholders that exist to exercise the API. They are not drawn from
//! any published cohort, they carry no citations, and they must not be used to
//! interpret a measurement from a real person. Any percentile, z-score or
//! classification computed against them demonstrates the arithmetic only.
//!
//! Supply your own cited reference values before drawing research conclusions.
//! See the crate-level documentation for the full statement.
//!
//! Research and educational use only. Not a medical device.

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
                ((18, 29), None, 250.0, 40.0),
                ((30, 39), None, 260.0, 42.0),
                ((40, 49), None, 275.0, 45.0),
                ((50, 59), None, 290.0, 50.0),
                ((60, 69), None, 310.0, 55.0),
                ((70, 79), None, 340.0, 65.0),
                ((80, 89), None, 380.0, 80.0),
            ],
        );
        self.set_reliability(MetricType::SimpleReactionTime, 0.85);

        // Choice Reaction Time (ms)
        self.add_age_sex_norms(
            MetricType::ChoiceReactionTime,
            &[
                ((18, 29), None, 350.0, 50.0),
                ((30, 39), None, 365.0, 52.0),
                ((40, 49), None, 385.0, 58.0),
                ((50, 59), None, 410.0, 65.0),
                ((60, 69), None, 450.0, 75.0),
                ((70, 79), None, 500.0, 90.0),
                ((80, 89), None, 560.0, 110.0),
            ],
        );
        self.set_reliability(MetricType::ChoiceReactionTime, 0.82);

        // Trail Making A (seconds)
        self.add_age_sex_norms(
            MetricType::TrailMakingA,
            &[
                ((18, 29), None, 22.0, 7.0),
                ((30, 39), None, 24.0, 8.0),
                ((40, 49), None, 28.0, 9.0),
                ((50, 59), None, 32.0, 10.0),
                ((60, 69), None, 38.0, 12.0),
                ((70, 79), None, 48.0, 16.0),
                ((80, 89), None, 62.0, 22.0),
            ],
        );
        self.set_reliability(MetricType::TrailMakingA, 0.79);

        // Trail Making B (seconds)
        self.add_age_sex_norms(
            MetricType::TrailMakingB,
            &[
                ((18, 29), None, 48.0, 15.0),
                ((30, 39), None, 52.0, 16.0),
                ((40, 49), None, 60.0, 18.0),
                ((50, 59), None, 72.0, 22.0),
                ((60, 69), None, 90.0, 28.0),
                ((70, 79), None, 115.0, 38.0),
                ((80, 89), None, 150.0, 55.0),
            ],
        );
        self.set_reliability(MetricType::TrailMakingB, 0.82);

        // MoCA Total (0-30)
        self.add_age_sex_norms(
            MetricType::MocaTotal,
            &[
                ((18, 59), None, 27.5, 2.0),
                ((60, 69), None, 26.5, 2.2),
                ((70, 79), None, 25.5, 2.5),
                ((80, 89), None, 24.0, 3.0),
            ],
        );
        self.set_reliability(MetricType::MocaTotal, 0.92);

        // === Motor Metrics ===

        // Gait Velocity (m/s) - sex-stratified
        self.add_age_sex_norms(
            MetricType::GaitVelocity,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 1.35, 0.15),
                ((18, 29), Some(crate::demographics::Sex::Female), 1.30, 0.14),
                ((30, 39), Some(crate::demographics::Sex::Male), 1.32, 0.15),
                ((30, 39), Some(crate::demographics::Sex::Female), 1.28, 0.14),
                ((40, 49), Some(crate::demographics::Sex::Male), 1.28, 0.16),
                ((40, 49), Some(crate::demographics::Sex::Female), 1.24, 0.15),
                ((50, 59), Some(crate::demographics::Sex::Male), 1.22, 0.18),
                ((50, 59), Some(crate::demographics::Sex::Female), 1.18, 0.16),
                ((60, 69), Some(crate::demographics::Sex::Male), 1.15, 0.20),
                ((60, 69), Some(crate::demographics::Sex::Female), 1.10, 0.18),
                ((70, 79), Some(crate::demographics::Sex::Male), 1.05, 0.22),
                ((70, 79), Some(crate::demographics::Sex::Female), 0.98, 0.20),
                ((80, 89), Some(crate::demographics::Sex::Male), 0.90, 0.25),
                ((80, 89), Some(crate::demographics::Sex::Female), 0.82, 0.22),
            ],
        );
        self.set_reliability(MetricType::GaitVelocity, 0.95);

        // Timed Up and Go (seconds)
        self.add_age_sex_norms(
            MetricType::TimedUpAndGo,
            &[
                ((18, 39), None, 7.0, 1.2),
                ((40, 49), None, 7.5, 1.3),
                ((50, 59), None, 8.0, 1.5),
                ((60, 69), None, 9.0, 2.0),
                ((70, 79), None, 10.5, 2.5),
                ((80, 89), None, 12.5, 3.5),
            ],
        );
        self.set_reliability(MetricType::TimedUpAndGo, 0.97);

        // Grip Strength (kg) - highly sex-stratified
        self.add_age_sex_norms(
            MetricType::GripStrength,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 48.0, 8.0),
                ((18, 29), Some(crate::demographics::Sex::Female), 30.0, 5.5),
                ((30, 39), Some(crate::demographics::Sex::Male), 47.0, 8.5),
                ((30, 39), Some(crate::demographics::Sex::Female), 29.0, 5.5),
                ((40, 49), Some(crate::demographics::Sex::Male), 45.0, 9.0),
                ((40, 49), Some(crate::demographics::Sex::Female), 28.0, 5.5),
                ((50, 59), Some(crate::demographics::Sex::Male), 42.0, 9.0),
                ((50, 59), Some(crate::demographics::Sex::Female), 26.0, 5.5),
                ((60, 69), Some(crate::demographics::Sex::Male), 38.0, 9.5),
                ((60, 69), Some(crate::demographics::Sex::Female), 23.0, 5.5),
                ((70, 79), Some(crate::demographics::Sex::Male), 33.0, 9.0),
                ((70, 79), Some(crate::demographics::Sex::Female), 20.0, 5.0),
                ((80, 89), Some(crate::demographics::Sex::Male), 27.0, 8.0),
                ((80, 89), Some(crate::demographics::Sex::Female), 16.0, 4.5),
            ],
        );
        self.set_reliability(MetricType::GripStrength, 0.95);

        // === Balance Metrics ===

        // Sway Area (cm²)
        self.add_age_sex_norms(
            MetricType::SwayArea,
            &[
                ((18, 39), None, 2.5, 1.0),
                ((40, 49), None, 3.0, 1.2),
                ((50, 59), None, 3.5, 1.5),
                ((60, 69), None, 4.5, 2.0),
                ((70, 79), None, 6.0, 2.8),
                ((80, 89), None, 8.5, 4.0),
            ],
        );
        self.set_reliability(MetricType::SwayArea, 0.80);

        // Berg Balance Scale (0-56)
        self.add_age_sex_norms(
            MetricType::BergBalanceScale,
            &[
                ((60, 69), None, 54.0, 2.0),
                ((70, 79), None, 52.0, 3.0),
                ((80, 89), None, 48.0, 5.0),
            ],
        );
        self.set_reliability(MetricType::BergBalanceScale, 0.98);

        // === Physiological Metrics ===

        // HRV SDNN (ms)
        self.add_age_sex_norms(
            MetricType::HrvSdnn,
            &[
                ((18, 29), None, 140.0, 40.0),
                ((30, 39), None, 125.0, 38.0),
                ((40, 49), None, 110.0, 35.0),
                ((50, 59), None, 95.0, 32.0),
                ((60, 69), None, 80.0, 28.0),
                ((70, 79), None, 65.0, 25.0),
                ((80, 89), None, 50.0, 20.0),
            ],
        );
        self.set_reliability(MetricType::HrvSdnn, 0.85);

        // HRV RMSSD (ms)
        self.add_age_sex_norms(
            MetricType::HrvRmssd,
            &[
                ((18, 29), None, 45.0, 20.0),
                ((30, 39), None, 38.0, 18.0),
                ((40, 49), None, 32.0, 15.0),
                ((50, 59), None, 26.0, 12.0),
                ((60, 69), None, 22.0, 10.0),
                ((70, 79), None, 18.0, 8.0),
                ((80, 89), None, 14.0, 6.0),
            ],
        );
        self.set_reliability(MetricType::HrvRmssd, 0.82);

        // === Sleep Metrics ===

        // Total Sleep Time (min)
        self.add_age_sex_norms(
            MetricType::TotalSleepTime,
            &[
                ((18, 29), None, 450.0, 45.0),
                ((30, 39), None, 440.0, 50.0),
                ((40, 49), None, 425.0, 55.0),
                ((50, 59), None, 410.0, 60.0),
                ((60, 69), None, 390.0, 65.0),
                ((70, 79), None, 370.0, 70.0),
                ((80, 89), None, 350.0, 75.0),
            ],
        );
        self.set_reliability(MetricType::TotalSleepTime, 0.75);

        // Sleep Efficiency (%)
        self.add_age_sex_norms(
            MetricType::SleepEfficiency,
            &[
                ((18, 29), None, 92.0, 5.0),
                ((30, 39), None, 90.0, 6.0),
                ((40, 49), None, 88.0, 7.0),
                ((50, 59), None, 85.0, 8.0),
                ((60, 69), None, 82.0, 9.0),
                ((70, 79), None, 78.0, 10.0),
                ((80, 89), None, 74.0, 12.0),
            ],
        );
        self.set_reliability(MetricType::SleepEfficiency, 0.78);

        // === Pain/Sensory Metrics (Phase F) ===

        // Pressure Pain Threshold (kPa) - sex-stratified, highly variable
        self.add_age_sex_norms(
            MetricType::PressurePainThreshold,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 450.0, 120.0),
                ((18, 29), Some(crate::demographics::Sex::Female), 350.0, 100.0),
                ((30, 39), Some(crate::demographics::Sex::Male), 440.0, 115.0),
                ((30, 39), Some(crate::demographics::Sex::Female), 340.0, 95.0),
                ((40, 49), Some(crate::demographics::Sex::Male), 420.0, 110.0),
                ((40, 49), Some(crate::demographics::Sex::Female), 330.0, 90.0),
                ((50, 59), Some(crate::demographics::Sex::Male), 400.0, 105.0),
                ((50, 59), Some(crate::demographics::Sex::Female), 315.0, 85.0),
                ((60, 69), Some(crate::demographics::Sex::Male), 380.0, 100.0),
                ((60, 69), Some(crate::demographics::Sex::Female), 300.0, 80.0),
                ((70, 89), Some(crate::demographics::Sex::Male), 360.0, 95.0),
                ((70, 89), Some(crate::demographics::Sex::Female), 285.0, 75.0),
            ],
        );
        self.set_reliability(MetricType::PressurePainThreshold, 0.88);

        // Pain Tolerance (kPa) - higher than threshold
        self.add_age_sex_norms(
            MetricType::PainTolerance,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 650.0, 150.0),
                ((18, 29), Some(crate::demographics::Sex::Female), 520.0, 130.0),
                ((30, 49), Some(crate::demographics::Sex::Male), 620.0, 145.0),
                ((30, 49), Some(crate::demographics::Sex::Female), 500.0, 125.0),
                ((50, 69), Some(crate::demographics::Sex::Male), 580.0, 140.0),
                ((50, 69), Some(crate::demographics::Sex::Female), 470.0, 120.0),
                ((70, 89), Some(crate::demographics::Sex::Male), 540.0, 135.0),
                ((70, 89), Some(crate::demographics::Sex::Female), 440.0, 115.0),
            ],
        );
        self.set_reliability(MetricType::PainTolerance, 0.85);

        // CPM Effect (% pain reduction) - conditioned pain modulation
        self.add_age_sex_norms(
            MetricType::CpmEffect,
            &[
                ((18, 39), None, 35.0, 15.0),
                ((40, 59), None, 30.0, 14.0),
                ((60, 79), None, 22.0, 12.0),
                ((80, 89), None, 15.0, 10.0),
            ],
        );
        self.set_reliability(MetricType::CpmEffect, 0.75);

        // Vibration Threshold (μm) - increases with age
        self.add_age_sex_norms(
            MetricType::VibrationThreshold,
            &[
                ((18, 29), None, 0.5, 0.2),
                ((30, 39), None, 0.6, 0.25),
                ((40, 49), None, 0.8, 0.3),
                ((50, 59), None, 1.2, 0.4),
                ((60, 69), None, 2.0, 0.6),
                ((70, 79), None, 3.5, 1.0),
                ((80, 89), None, 5.5, 1.5),
            ],
        );
        self.set_reliability(MetricType::VibrationThreshold, 0.82);

        // Joint Position Error (degrees) - proprioception
        self.add_age_sex_norms(
            MetricType::JointPositionError,
            &[
                ((18, 39), None, 2.5, 1.0),
                ((40, 49), None, 3.0, 1.2),
                ((50, 59), None, 3.5, 1.4),
                ((60, 69), None, 4.5, 1.8),
                ((70, 79), None, 6.0, 2.2),
                ((80, 89), None, 8.0, 3.0),
            ],
        );
        self.set_reliability(MetricType::JointPositionError, 0.78);

        // === Additional Balance Metrics ===

        // Sway Velocity (cm/s)
        self.add_age_sex_norms(
            MetricType::SwayVelocity,
            &[
                ((18, 39), None, 0.8, 0.25),
                ((40, 49), None, 0.95, 0.30),
                ((50, 59), None, 1.1, 0.35),
                ((60, 69), None, 1.4, 0.45),
                ((70, 79), None, 1.8, 0.55),
                ((80, 89), None, 2.4, 0.70),
            ],
        );
        self.set_reliability(MetricType::SwayVelocity, 0.82);

        // Sway Path Length (cm)
        self.add_age_sex_norms(
            MetricType::SwayPathLength,
            &[
                ((18, 39), None, 25.0, 8.0),
                ((40, 49), None, 30.0, 10.0),
                ((50, 59), None, 36.0, 12.0),
                ((60, 69), None, 45.0, 15.0),
                ((70, 79), None, 58.0, 20.0),
                ((80, 89), None, 75.0, 28.0),
            ],
        );
        self.set_reliability(MetricType::SwayPathLength, 0.80);

        // Limits of Stability - Max Excursion (%)
        self.add_age_sex_norms(
            MetricType::LosMaxExcursion,
            &[
                ((18, 39), None, 95.0, 8.0),
                ((40, 49), None, 92.0, 9.0),
                ((50, 59), None, 88.0, 10.0),
                ((60, 69), None, 82.0, 12.0),
                ((70, 79), None, 74.0, 14.0),
                ((80, 89), None, 65.0, 16.0),
            ],
        );
        self.set_reliability(MetricType::LosMaxExcursion, 0.85);

        // Limits of Stability - Reaction Time (ms)
        self.add_age_sex_norms(
            MetricType::LosReactionTime,
            &[
                ((18, 39), None, 450.0, 80.0),
                ((40, 49), None, 480.0, 90.0),
                ((50, 59), None, 520.0, 100.0),
                ((60, 69), None, 580.0, 120.0),
                ((70, 79), None, 660.0, 150.0),
                ((80, 89), None, 780.0, 180.0),
            ],
        );
        self.set_reliability(MetricType::LosReactionTime, 0.82);

        // === Additional Motor Metrics ===

        // Rate of Force Development (N/s)
        self.add_age_sex_norms(
            MetricType::RateOfForceDevelopment,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 1200.0, 300.0),
                ((18, 29), Some(crate::demographics::Sex::Female), 750.0, 200.0),
                ((30, 39), Some(crate::demographics::Sex::Male), 1100.0, 280.0),
                ((30, 39), Some(crate::demographics::Sex::Female), 700.0, 185.0),
                ((40, 49), Some(crate::demographics::Sex::Male), 950.0, 260.0),
                ((40, 49), Some(crate::demographics::Sex::Female), 620.0, 170.0),
                ((50, 59), Some(crate::demographics::Sex::Male), 800.0, 240.0),
                ((50, 59), Some(crate::demographics::Sex::Female), 530.0, 155.0),
                ((60, 69), Some(crate::demographics::Sex::Male), 650.0, 220.0),
                ((60, 69), Some(crate::demographics::Sex::Female), 440.0, 140.0),
                ((70, 89), Some(crate::demographics::Sex::Male), 480.0, 180.0),
                ((70, 89), Some(crate::demographics::Sex::Female), 320.0, 120.0),
            ],
        );
        self.set_reliability(MetricType::RateOfForceDevelopment, 0.90);

        // Tapping Frequency (Hz)
        self.add_age_sex_norms(
            MetricType::TappingFrequency,
            &[
                ((18, 29), None, 6.0, 0.8),
                ((30, 39), None, 5.8, 0.9),
                ((40, 49), None, 5.5, 0.9),
                ((50, 59), None, 5.2, 1.0),
                ((60, 69), None, 4.8, 1.1),
                ((70, 79), None, 4.3, 1.2),
                ((80, 89), None, 3.7, 1.3),
            ],
        );
        self.set_reliability(MetricType::TappingFrequency, 0.88);

        // Stride Length (m) - sex-stratified due to height differences
        self.add_age_sex_norms(
            MetricType::StrideLength,
            &[
                ((18, 39), Some(crate::demographics::Sex::Male), 1.50, 0.12),
                ((18, 39), Some(crate::demographics::Sex::Female), 1.35, 0.11),
                ((40, 59), Some(crate::demographics::Sex::Male), 1.45, 0.13),
                ((40, 59), Some(crate::demographics::Sex::Female), 1.30, 0.12),
                ((60, 79), Some(crate::demographics::Sex::Male), 1.35, 0.15),
                ((60, 79), Some(crate::demographics::Sex::Female), 1.20, 0.14),
                ((80, 89), Some(crate::demographics::Sex::Male), 1.20, 0.18),
                ((80, 89), Some(crate::demographics::Sex::Female), 1.05, 0.16),
            ],
        );
        self.set_reliability(MetricType::StrideLength, 0.92);

        // Cadence (steps/min)
        self.add_age_sex_norms(
            MetricType::Cadence,
            &[
                ((18, 39), None, 115.0, 10.0),
                ((40, 49), None, 112.0, 11.0),
                ((50, 59), None, 108.0, 12.0),
                ((60, 69), None, 104.0, 13.0),
                ((70, 79), None, 98.0, 14.0),
                ((80, 89), None, 90.0, 16.0),
            ],
        );
        self.set_reliability(MetricType::Cadence, 0.90);

        // === Additional Physiological Metrics ===

        // Heart Rate (bpm) - resting
        self.add_age_sex_norms(
            MetricType::HeartRate,
            &[
                ((18, 29), None, 68.0, 10.0),
                ((30, 39), None, 70.0, 10.0),
                ((40, 49), None, 72.0, 11.0),
                ((50, 59), None, 74.0, 11.0),
                ((60, 69), None, 72.0, 10.0),
                ((70, 79), None, 70.0, 10.0),
                ((80, 89), None, 72.0, 12.0),
            ],
        );
        self.set_reliability(MetricType::HeartRate, 0.92);

        // Respiratory Rate (breaths/min)
        self.add_age_sex_norms(
            MetricType::RespiratoryRate,
            &[
                ((18, 39), None, 14.0, 2.5),
                ((40, 59), None, 15.0, 3.0),
                ((60, 79), None, 16.0, 3.5),
                ((80, 89), None, 18.0, 4.0),
            ],
        );
        self.set_reliability(MetricType::RespiratoryRate, 0.75);

        // SpO2 (%) - oxygen saturation
        self.add_age_sex_norms(
            MetricType::SpO2,
            &[
                ((18, 59), None, 97.5, 1.0),
                ((60, 69), None, 97.0, 1.2),
                ((70, 79), None, 96.5, 1.5),
                ((80, 89), None, 95.5, 2.0),
            ],
        );
        self.set_reliability(MetricType::SpO2, 0.85);

        // Blood Pressure Systolic (mmHg)
        self.add_age_sex_norms(
            MetricType::BpSystolic,
            &[
                ((18, 29), None, 115.0, 10.0),
                ((30, 39), None, 118.0, 11.0),
                ((40, 49), None, 122.0, 12.0),
                ((50, 59), None, 128.0, 14.0),
                ((60, 69), None, 132.0, 16.0),
                ((70, 79), None, 138.0, 18.0),
                ((80, 89), None, 142.0, 20.0),
            ],
        );
        self.set_reliability(MetricType::BpSystolic, 0.88);

        // Blood Pressure Diastolic (mmHg)
        self.add_age_sex_norms(
            MetricType::BpDiastolic,
            &[
                ((18, 29), None, 72.0, 8.0),
                ((30, 39), None, 75.0, 8.0),
                ((40, 49), None, 78.0, 9.0),
                ((50, 59), None, 80.0, 9.0),
                ((60, 69), None, 78.0, 10.0),
                ((70, 79), None, 76.0, 10.0),
                ((80, 89), None, 74.0, 11.0),
            ],
        );
        self.set_reliability(MetricType::BpDiastolic, 0.86);

        // HRV pNN50 (%)
        self.add_age_sex_norms(
            MetricType::HrvPnn50,
            &[
                ((18, 29), None, 20.0, 12.0),
                ((30, 39), None, 16.0, 10.0),
                ((40, 49), None, 12.0, 8.0),
                ((50, 59), None, 8.0, 6.0),
                ((60, 69), None, 5.0, 4.0),
                ((70, 79), None, 3.0, 3.0),
                ((80, 89), None, 2.0, 2.0),
            ],
        );
        self.set_reliability(MetricType::HrvPnn50, 0.80);

        // === Additional Cognitive Metrics ===

        // Digit Span Forward
        self.add_age_sex_norms(
            MetricType::DigitSpanForward,
            &[
                ((18, 39), None, 7.0, 1.2),
                ((40, 59), None, 6.8, 1.3),
                ((60, 79), None, 6.2, 1.4),
                ((80, 89), None, 5.5, 1.5),
            ],
        );
        self.set_reliability(MetricType::DigitSpanForward, 0.80);

        // Digit Span Backward
        self.add_age_sex_norms(
            MetricType::DigitSpanBackward,
            &[
                ((18, 39), None, 5.5, 1.3),
                ((40, 59), None, 5.2, 1.4),
                ((60, 79), None, 4.5, 1.5),
                ((80, 89), None, 3.8, 1.6),
            ],
        );
        self.set_reliability(MetricType::DigitSpanBackward, 0.78);

        // Verbal Fluency (words/min)
        self.add_age_sex_norms(
            MetricType::VerbalFluency,
            &[
                ((18, 39), None, 18.0, 5.0),
                ((40, 59), None, 16.0, 5.0),
                ((60, 79), None, 13.0, 4.5),
                ((80, 89), None, 10.0, 4.0),
            ],
        );
        self.set_reliability(MetricType::VerbalFluency, 0.82);

        // Stroop Interference (ms)
        self.add_age_sex_norms(
            MetricType::StroopInterference,
            &[
                ((18, 29), None, 40.0, 20.0),
                ((30, 39), None, 45.0, 22.0),
                ((40, 49), None, 55.0, 25.0),
                ((50, 59), None, 70.0, 30.0),
                ((60, 69), None, 90.0, 38.0),
                ((70, 79), None, 120.0, 50.0),
                ((80, 89), None, 160.0, 65.0),
            ],
        );
        self.set_reliability(MetricType::StroopInterference, 0.75);

        // === Tremor Metrics ===

        // Tremor Amplitude (mm) - physiological tremor increases with age
        self.add_age_sex_norms(
            MetricType::TremorAmplitude,
            &[
                ((18, 39), None, 0.15, 0.08),
                ((40, 59), None, 0.22, 0.12),
                ((60, 79), None, 0.35, 0.18),
                ((80, 89), None, 0.50, 0.25),
            ],
        );
        self.set_reliability(MetricType::TremorAmplitude, 0.85);

        // Tremor Frequency (Hz) - typically 8-12 Hz physiological
        self.add_age_sex_norms(
            MetricType::TremorFrequency,
            &[
                ((18, 39), None, 10.0, 1.5),
                ((40, 59), None, 9.5, 1.8),
                ((60, 79), None, 8.5, 2.0),
                ((80, 89), None, 7.5, 2.2),
            ],
        );
        self.set_reliability(MetricType::TremorFrequency, 0.88);

        // === Additional Cognitive Metrics ===

        // Reaction Time Variability (Coefficient of Variation, %)
        self.add_age_sex_norms(
            MetricType::ReactionTimeVariability,
            &[
                ((18, 29), None, 12.0, 4.0),
                ((30, 49), None, 14.0, 4.5),
                ((50, 69), None, 18.0, 5.5),
                ((70, 89), None, 24.0, 7.0),
            ],
        );
        self.set_reliability(MetricType::ReactionTimeVariability, 0.75);

        // N-Back Accuracy (%)
        self.add_age_sex_norms(
            MetricType::NBackAccuracy,
            &[
                ((18, 29), None, 85.0, 8.0),
                ((30, 49), None, 82.0, 9.0),
                ((50, 69), None, 75.0, 12.0),
                ((70, 89), None, 65.0, 15.0),
            ],
        );
        self.set_reliability(MetricType::NBackAccuracy, 0.78);

        // N-Back d-prime (signal detection)
        self.add_age_sex_norms(
            MetricType::NBackDPrime,
            &[
                ((18, 29), None, 2.8, 0.6),
                ((30, 49), None, 2.5, 0.7),
                ((50, 69), None, 2.0, 0.8),
                ((70, 89), None, 1.5, 0.9),
            ],
        );
        self.set_reliability(MetricType::NBackDPrime, 0.80);

        // CPT Omissions (miss rate, %)
        self.add_age_sex_norms(
            MetricType::CptOmissions,
            &[
                ((18, 29), None, 2.0, 2.5),
                ((30, 49), None, 3.5, 3.0),
                ((50, 69), None, 6.0, 4.5),
                ((70, 89), None, 12.0, 7.0),
            ],
        );
        self.set_reliability(MetricType::CptOmissions, 0.72);

        // CPT Commissions (false alarm rate, %)
        self.add_age_sex_norms(
            MetricType::CptCommissions,
            &[
                ((18, 29), None, 15.0, 8.0),
                ((30, 49), None, 12.0, 7.0),
                ((50, 69), None, 10.0, 6.0),
                ((70, 89), None, 8.0, 5.0),
            ],
        );
        self.set_reliability(MetricType::CptCommissions, 0.70);

        // Trail Making B-A (executive function indicator, seconds)
        self.add_age_sex_norms(
            MetricType::TrailMakingBMinusA,
            &[
                ((18, 29), None, 25.0, 12.0),
                ((30, 49), None, 30.0, 15.0),
                ((50, 69), None, 45.0, 22.0),
                ((70, 89), None, 70.0, 35.0),
            ],
        );
        self.set_reliability(MetricType::TrailMakingBMinusA, 0.78);

        // === Additional Motor Metrics ===

        // Stride Time Variability (CV, %)
        self.add_age_sex_norms(
            MetricType::StrideTimeVariability,
            &[
                ((18, 39), None, 2.5, 0.8),
                ((40, 59), None, 3.0, 1.0),
                ((60, 74), None, 3.8, 1.3),
                ((75, 89), None, 5.0, 2.0),
            ],
        );
        self.set_reliability(MetricType::StrideTimeVariability, 0.85);

        // Double Support Time (% of gait cycle)
        self.add_age_sex_norms(
            MetricType::DoubleSupportTime,
            &[
                ((18, 39), None, 22.0, 3.0),
                ((40, 59), None, 24.0, 3.5),
                ((60, 74), None, 27.0, 4.0),
                ((75, 89), None, 32.0, 5.0),
            ],
        );
        self.set_reliability(MetricType::DoubleSupportTime, 0.88);

        // Tapping Variability (CV, %)
        self.add_age_sex_norms(
            MetricType::TappingVariability,
            &[
                ((18, 39), None, 8.0, 3.0),
                ((40, 59), None, 10.0, 4.0),
                ((60, 74), None, 14.0, 5.0),
                ((75, 89), None, 20.0, 7.0),
            ],
        );
        self.set_reliability(MetricType::TappingVariability, 0.82);

        // UPDRS Motor Score (0-132)
        self.add_age_sex_norms(
            MetricType::UpdrsMotor,
            &[
                ((18, 49), None, 0.0, 1.0),  // Healthy baseline
                ((50, 69), None, 2.0, 3.0),
                ((70, 89), None, 5.0, 5.0),
            ],
        );
        self.set_reliability(MetricType::UpdrsMotor, 0.92);

        // === Additional Balance Metrics ===

        // Romberg Quotient (eyes closed / eyes open sway ratio)
        self.add_age_sex_norms(
            MetricType::RombergQuotient,
            &[
                ((18, 39), None, 1.3, 0.3),
                ((40, 59), None, 1.5, 0.4),
                ((60, 74), None, 1.8, 0.5),
                ((75, 89), None, 2.2, 0.7),
            ],
        );
        self.set_reliability(MetricType::RombergQuotient, 0.75);

        // Limits of Stability - Directional Control (%)
        self.add_age_sex_norms(
            MetricType::LosDirectionalControl,
            &[
                ((18, 39), None, 82.0, 8.0),
                ((40, 59), None, 78.0, 10.0),
                ((60, 74), None, 72.0, 12.0),
                ((75, 89), None, 62.0, 15.0),
            ],
        );
        self.set_reliability(MetricType::LosDirectionalControl, 0.82);

        // === Additional Physiological Metrics ===

        // HRV LF/HF Ratio (autonomic balance)
        self.add_age_sex_norms(
            MetricType::HrvLfHf,
            &[
                ((18, 29), None, 1.5, 0.8),
                ((30, 49), None, 2.0, 1.0),
                ((50, 69), None, 2.5, 1.2),
                ((70, 89), None, 3.0, 1.5),
            ],
        );
        self.set_reliability(MetricType::HrvLfHf, 0.75);

        // === Additional Sleep Metrics ===

        // Sleep Onset Latency (minutes)
        self.add_age_sex_norms(
            MetricType::SleepOnsetLatency,
            &[
                ((18, 39), None, 12.0, 8.0),
                ((40, 59), None, 15.0, 10.0),
                ((60, 74), None, 20.0, 12.0),
                ((75, 89), None, 25.0, 15.0),
            ],
        );
        self.set_reliability(MetricType::SleepOnsetLatency, 0.70);

        // Wake After Sleep Onset (minutes)
        self.add_age_sex_norms(
            MetricType::WakeAfterSleepOnset,
            &[
                ((18, 39), None, 15.0, 12.0),
                ((40, 59), None, 25.0, 18.0),
                ((60, 74), None, 40.0, 25.0),
                ((75, 89), None, 60.0, 35.0),
            ],
        );
        self.set_reliability(MetricType::WakeAfterSleepOnset, 0.72);

        // REM Percentage (%)
        self.add_age_sex_norms(
            MetricType::RemPercent,
            &[
                ((18, 39), None, 22.0, 4.0),
                ((40, 59), None, 20.0, 4.5),
                ((60, 74), None, 18.0, 5.0),
                ((75, 89), None, 15.0, 5.5),
            ],
        );
        self.set_reliability(MetricType::RemPercent, 0.75);

        // Deep Sleep Percentage (N3, %)
        self.add_age_sex_norms(
            MetricType::DeepSleepPercent,
            &[
                ((18, 29), None, 20.0, 5.0),
                ((30, 49), None, 15.0, 5.0),
                ((50, 69), None, 10.0, 4.0),
                ((70, 89), None, 5.0, 3.0),
            ],
        );
        self.set_reliability(MetricType::DeepSleepPercent, 0.78);

        // === Composite Metrics ===

        // Cognitive Composite (standardized, mean=100, SD=15)
        self.add_age_sex_norms(
            MetricType::CognitiveComposite,
            &[
                ((18, 29), None, 105.0, 15.0),
                ((30, 49), None, 100.0, 15.0),
                ((50, 69), None, 95.0, 15.0),
                ((70, 89), None, 88.0, 16.0),
            ],
        );
        self.set_reliability(MetricType::CognitiveComposite, 0.90);

        // Motor Composite (standardized, mean=100, SD=15)
        self.add_age_sex_norms(
            MetricType::MotorComposite,
            &[
                ((18, 29), None, 105.0, 14.0),
                ((30, 49), None, 100.0, 14.0),
                ((50, 69), None, 92.0, 15.0),
                ((70, 89), None, 82.0, 16.0),
            ],
        );
        self.set_reliability(MetricType::MotorComposite, 0.92);

        // Global Composite (standardized, mean=100, SD=15)
        self.add_age_sex_norms(
            MetricType::GlobalComposite,
            &[
                ((18, 29), None, 105.0, 14.0),
                ((30, 49), None, 100.0, 14.0),
                ((50, 69), None, 94.0, 15.0),
                ((70, 89), None, 85.0, 16.0),
            ],
        );
        self.set_reliability(MetricType::GlobalComposite, 0.93);

        // Frailty Index (0-1 scale, proportion of deficits)
        self.add_age_sex_norms(
            MetricType::FrailtyIndex,
            &[
                ((18, 49), None, 0.05, 0.03),
                ((50, 64), None, 0.10, 0.06),
                ((65, 74), None, 0.18, 0.10),
                ((75, 84), None, 0.28, 0.12),
                ((85, 99), None, 0.38, 0.15),
            ],
        );
        self.set_reliability(MetricType::FrailtyIndex, 0.88);

        // === PPG Metrics ===

        // Pulse Transit Time (ms) - increases with arterial stiffness/age
        self.add_age_sex_norms(
            MetricType::PulseTransitTime,
            &[
                ((18, 29), None, 280.0, 40.0),
                ((30, 39), None, 260.0, 38.0),
                ((40, 49), None, 240.0, 35.0),
                ((50, 59), None, 220.0, 32.0),
                ((60, 69), None, 200.0, 30.0),
                ((70, 79), None, 180.0, 28.0),
                ((80, 89), None, 160.0, 25.0),
            ],
        );
        self.set_reliability(MetricType::PulseTransitTime, 0.85);

        // Pulse Wave Velocity (m/s) - increases with age/arterial stiffness
        self.add_age_sex_norms(
            MetricType::PulseWaveVelocity,
            &[
                ((18, 29), None, 6.5, 1.0),
                ((30, 39), None, 7.0, 1.1),
                ((40, 49), None, 7.8, 1.3),
                ((50, 59), None, 8.8, 1.5),
                ((60, 69), None, 10.0, 1.8),
                ((70, 79), None, 11.5, 2.2),
                ((80, 89), None, 13.0, 2.5),
            ],
        );
        self.set_reliability(MetricType::PulseWaveVelocity, 0.88);

        // Augmentation Index (%) - increases with vascular aging
        self.add_age_sex_norms(
            MetricType::AugmentationIndex,
            &[
                ((18, 29), None, 5.0, 8.0),
                ((30, 39), None, 12.0, 9.0),
                ((40, 49), None, 20.0, 10.0),
                ((50, 59), None, 28.0, 10.0),
                ((60, 69), None, 32.0, 11.0),
                ((70, 79), None, 35.0, 12.0),
                ((80, 89), None, 38.0, 12.0),
            ],
        );
        self.set_reliability(MetricType::AugmentationIndex, 0.82);

        // Stiffness Index (m/s)
        self.add_age_sex_norms(
            MetricType::StiffnessIndex,
            &[
                ((18, 29), None, 6.0, 1.2),
                ((30, 49), None, 7.0, 1.4),
                ((50, 69), None, 8.5, 1.8),
                ((70, 89), None, 10.5, 2.2),
            ],
        );
        self.set_reliability(MetricType::StiffnessIndex, 0.80);

        // Perfusion Index (%)
        self.add_age_sex_norms(
            MetricType::PerfusionIndex,
            &[
                ((18, 39), None, 5.0, 3.0),
                ((40, 59), None, 4.0, 2.5),
                ((60, 79), None, 3.0, 2.0),
                ((80, 89), None, 2.0, 1.5),
            ],
        );
        self.set_reliability(MetricType::PerfusionIndex, 0.75);

        // PRV SDNN (ms) - similar to HRV SDNN
        self.add_age_sex_norms(
            MetricType::PrvSdnn,
            &[
                ((18, 29), None, 135.0, 38.0),
                ((30, 39), None, 120.0, 36.0),
                ((40, 49), None, 105.0, 33.0),
                ((50, 59), None, 90.0, 30.0),
                ((60, 69), None, 75.0, 26.0),
                ((70, 79), None, 60.0, 22.0),
                ((80, 89), None, 48.0, 18.0),
            ],
        );
        self.set_reliability(MetricType::PrvSdnn, 0.84);

        // === EDA Metrics ===

        // Skin Conductance Level (μS) - baseline arousal
        self.add_age_sex_norms(
            MetricType::SkinConductanceLevel,
            &[
                ((18, 29), None, 5.0, 3.0),
                ((30, 49), None, 4.5, 2.8),
                ((50, 69), None, 3.5, 2.5),
                ((70, 89), None, 2.5, 2.0),
            ],
        );
        self.set_reliability(MetricType::SkinConductanceLevel, 0.75);

        // SCR Frequency (events/min) - spontaneous responses
        self.add_age_sex_norms(
            MetricType::ScrFrequency,
            &[
                ((18, 29), None, 8.0, 5.0),
                ((30, 49), None, 6.0, 4.0),
                ((50, 69), None, 4.0, 3.0),
                ((70, 89), None, 2.5, 2.0),
            ],
        );
        self.set_reliability(MetricType::ScrFrequency, 0.72);

        // SCR Amplitude (μS)
        self.add_age_sex_norms(
            MetricType::ScrAmplitude,
            &[
                ((18, 29), None, 0.5, 0.3),
                ((30, 49), None, 0.4, 0.25),
                ((50, 69), None, 0.3, 0.2),
                ((70, 89), None, 0.2, 0.15),
            ],
        );
        self.set_reliability(MetricType::ScrAmplitude, 0.78);

        // NS-SCR Count (non-specific, per 5 min)
        self.add_age_sex_norms(
            MetricType::NsScrCount,
            &[
                ((18, 29), None, 25.0, 15.0),
                ((30, 49), None, 20.0, 12.0),
                ((50, 69), None, 12.0, 8.0),
                ((70, 89), None, 6.0, 5.0),
            ],
        );
        self.set_reliability(MetricType::NsScrCount, 0.70);

        // EDA Recovery Time (s) - half-recovery
        self.add_age_sex_norms(
            MetricType::EdaRecoveryTime,
            &[
                ((18, 29), None, 3.0, 1.0),
                ((30, 49), None, 3.5, 1.2),
                ((50, 69), None, 4.5, 1.5),
                ((70, 89), None, 6.0, 2.0),
            ],
        );
        self.set_reliability(MetricType::EdaRecoveryTime, 0.75);

        // === EEG Band Power Metrics ===

        // Delta Power (μV²) - eyes closed resting
        self.add_age_sex_norms(
            MetricType::EegDeltaPower,
            &[
                ((18, 29), None, 15.0, 8.0),
                ((30, 49), None, 18.0, 10.0),
                ((50, 69), None, 22.0, 12.0),
                ((70, 89), None, 28.0, 15.0),
            ],
        );
        self.set_reliability(MetricType::EegDeltaPower, 0.80);

        // Theta Power (μV²)
        self.add_age_sex_norms(
            MetricType::EegThetaPower,
            &[
                ((18, 29), None, 12.0, 6.0),
                ((30, 49), None, 14.0, 7.0),
                ((50, 69), None, 16.0, 8.0),
                ((70, 89), None, 20.0, 10.0),
            ],
        );
        self.set_reliability(MetricType::EegThetaPower, 0.78);

        // Alpha Power (μV²) - dominant rhythm
        self.add_age_sex_norms(
            MetricType::EegAlphaPower,
            &[
                ((18, 29), None, 35.0, 18.0),
                ((30, 49), None, 32.0, 16.0),
                ((50, 69), None, 28.0, 14.0),
                ((70, 89), None, 22.0, 12.0),
            ],
        );
        self.set_reliability(MetricType::EegAlphaPower, 0.85);

        // Beta Power (μV²)
        self.add_age_sex_norms(
            MetricType::EegBetaPower,
            &[
                ((18, 29), None, 8.0, 4.0),
                ((30, 49), None, 10.0, 5.0),
                ((50, 69), None, 12.0, 6.0),
                ((70, 89), None, 10.0, 5.0),
            ],
        );
        self.set_reliability(MetricType::EegBetaPower, 0.78);

        // Gamma Power (μV²)
        self.add_age_sex_norms(
            MetricType::EegGammaPower,
            &[
                ((18, 29), None, 2.0, 1.2),
                ((30, 49), None, 2.5, 1.5),
                ((50, 69), None, 2.2, 1.3),
                ((70, 89), None, 1.8, 1.0),
            ],
        );
        self.set_reliability(MetricType::EegGammaPower, 0.72);

        // Alpha/Theta Ratio
        self.add_age_sex_norms(
            MetricType::EegAlphaThetaRatio,
            &[
                ((18, 29), None, 3.0, 1.2),
                ((30, 49), None, 2.5, 1.0),
                ((50, 69), None, 2.0, 0.8),
                ((70, 89), None, 1.5, 0.6),
            ],
        );
        self.set_reliability(MetricType::EegAlphaThetaRatio, 0.80);

        // Alpha Asymmetry (F4-F3, frontal)
        self.add_age_sex_norms(
            MetricType::EegAlphaAsymmetry,
            &[
                ((18, 39), None, 0.0, 0.15),
                ((40, 59), None, 0.0, 0.18),
                ((60, 89), None, 0.0, 0.20),
            ],
        );
        self.set_reliability(MetricType::EegAlphaAsymmetry, 0.75);

        // === Eye Tracking Metrics ===

        // Saccade Peak Velocity (°/s)
        self.add_age_sex_norms(
            MetricType::SaccadePeakVelocity,
            &[
                ((18, 29), None, 450.0, 60.0),
                ((30, 49), None, 420.0, 65.0),
                ((50, 69), None, 380.0, 70.0),
                ((70, 89), None, 320.0, 80.0),
            ],
        );
        self.set_reliability(MetricType::SaccadePeakVelocity, 0.88);

        // Saccade Amplitude (°)
        self.add_age_sex_norms(
            MetricType::SaccadeAmplitude,
            &[
                ((18, 39), None, 8.0, 3.0),
                ((40, 59), None, 7.5, 3.0),
                ((60, 79), None, 7.0, 3.0),
                ((80, 89), None, 6.5, 3.0),
            ],
        );
        self.set_reliability(MetricType::SaccadeAmplitude, 0.82);

        // Saccade Latency (ms)
        self.add_age_sex_norms(
            MetricType::SaccadeLatency,
            &[
                ((18, 29), None, 180.0, 30.0),
                ((30, 49), None, 195.0, 35.0),
                ((50, 69), None, 220.0, 45.0),
                ((70, 89), None, 260.0, 60.0),
            ],
        );
        self.set_reliability(MetricType::SaccadeLatency, 0.85);

        // Fixation Duration (ms)
        self.add_age_sex_norms(
            MetricType::FixationDuration,
            &[
                ((18, 29), None, 250.0, 80.0),
                ((30, 49), None, 270.0, 90.0),
                ((50, 69), None, 300.0, 100.0),
                ((70, 89), None, 350.0, 120.0),
            ],
        );
        self.set_reliability(MetricType::FixationDuration, 0.80);

        // Fixation Count (per minute)
        self.add_age_sex_norms(
            MetricType::FixationCount,
            &[
                ((18, 39), None, 180.0, 40.0),
                ((40, 59), None, 160.0, 45.0),
                ((60, 79), None, 140.0, 50.0),
                ((80, 89), None, 120.0, 50.0),
            ],
        );
        self.set_reliability(MetricType::FixationCount, 0.78);

        // Pupil Diameter (mm) - baseline
        self.add_age_sex_norms(
            MetricType::PupilDiameter,
            &[
                ((18, 29), None, 4.5, 0.8),
                ((30, 49), None, 4.0, 0.8),
                ((50, 69), None, 3.5, 0.7),
                ((70, 89), None, 3.0, 0.6),
            ],
        );
        self.set_reliability(MetricType::PupilDiameter, 0.85);

        // Pupil Response Latency (ms)
        self.add_age_sex_norms(
            MetricType::PupilResponseLatency,
            &[
                ((18, 29), None, 200.0, 30.0),
                ((30, 49), None, 220.0, 35.0),
                ((50, 69), None, 250.0, 45.0),
                ((70, 89), None, 300.0, 60.0),
            ],
        );
        self.set_reliability(MetricType::PupilResponseLatency, 0.82);

        // Smooth Pursuit Gain
        self.add_age_sex_norms(
            MetricType::SmoothPursuitGain,
            &[
                ((18, 29), None, 0.95, 0.05),
                ((30, 49), None, 0.92, 0.06),
                ((50, 69), None, 0.88, 0.08),
                ((70, 89), None, 0.80, 0.10),
            ],
        );
        self.set_reliability(MetricType::SmoothPursuitGain, 0.85);

        // Blink Rate (per minute)
        self.add_age_sex_norms(
            MetricType::BlinkRate,
            &[
                ((18, 39), None, 17.0, 6.0),
                ((40, 59), None, 18.0, 7.0),
                ((60, 89), None, 16.0, 6.0),
            ],
        );
        self.set_reliability(MetricType::BlinkRate, 0.75);

        // === Voice Metrics ===

        // Voice F0 (Hz) - sex-stratified
        self.add_age_sex_norms(
            MetricType::VoiceF0,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 120.0, 20.0),
                ((18, 29), Some(crate::demographics::Sex::Female), 220.0, 25.0),
                ((30, 49), Some(crate::demographics::Sex::Male), 115.0, 20.0),
                ((30, 49), Some(crate::demographics::Sex::Female), 210.0, 25.0),
                ((50, 69), Some(crate::demographics::Sex::Male), 110.0, 22.0),
                ((50, 69), Some(crate::demographics::Sex::Female), 195.0, 28.0),
                ((70, 89), Some(crate::demographics::Sex::Male), 130.0, 25.0),
                ((70, 89), Some(crate::demographics::Sex::Female), 180.0, 30.0),
            ],
        );
        self.set_reliability(MetricType::VoiceF0, 0.90);

        // Voice F0 Variability (semitones)
        self.add_age_sex_norms(
            MetricType::VoiceF0Variability,
            &[
                ((18, 39), None, 3.0, 1.0),
                ((40, 59), None, 3.5, 1.2),
                ((60, 79), None, 4.0, 1.5),
                ((80, 89), None, 5.0, 2.0),
            ],
        );
        self.set_reliability(MetricType::VoiceF0Variability, 0.82);

        // Voice Jitter (%)
        self.add_age_sex_norms(
            MetricType::VoiceJitter,
            &[
                ((18, 39), None, 0.4, 0.2),
                ((40, 59), None, 0.6, 0.3),
                ((60, 79), None, 0.9, 0.4),
                ((80, 89), None, 1.3, 0.6),
            ],
        );
        self.set_reliability(MetricType::VoiceJitter, 0.85);

        // Voice Shimmer (%)
        self.add_age_sex_norms(
            MetricType::VoiceShimmer,
            &[
                ((18, 39), None, 2.5, 1.2),
                ((40, 59), None, 3.5, 1.5),
                ((60, 79), None, 5.0, 2.0),
                ((80, 89), None, 7.0, 3.0),
            ],
        );
        self.set_reliability(MetricType::VoiceShimmer, 0.83);

        // Voice HNR (dB)
        self.add_age_sex_norms(
            MetricType::VoiceHnr,
            &[
                ((18, 39), None, 22.0, 4.0),
                ((40, 59), None, 20.0, 4.5),
                ((60, 79), None, 17.0, 5.0),
                ((80, 89), None, 14.0, 5.5),
            ],
        );
        self.set_reliability(MetricType::VoiceHnr, 0.85);

        // Speech Rate (syllables/s)
        self.add_age_sex_norms(
            MetricType::SpeechRate,
            &[
                ((18, 39), None, 5.0, 1.0),
                ((40, 59), None, 4.8, 1.0),
                ((60, 79), None, 4.2, 1.2),
                ((80, 89), None, 3.5, 1.2),
            ],
        );
        self.set_reliability(MetricType::SpeechRate, 0.80);

        // Voice Onset Time (ms)
        self.add_age_sex_norms(
            MetricType::VoiceOnsetTime,
            &[
                ((18, 39), None, 25.0, 10.0),
                ((40, 59), None, 30.0, 12.0),
                ((60, 79), None, 40.0, 15.0),
                ((80, 89), None, 55.0, 20.0),
            ],
        );
        self.set_reliability(MetricType::VoiceOnsetTime, 0.78);

        // Maximum Phonation Time (s)
        self.add_age_sex_norms(
            MetricType::MaxPhonationTime,
            &[
                ((18, 29), Some(crate::demographics::Sex::Male), 25.0, 8.0),
                ((18, 29), Some(crate::demographics::Sex::Female), 20.0, 6.0),
                ((30, 49), Some(crate::demographics::Sex::Male), 22.0, 7.0),
                ((30, 49), Some(crate::demographics::Sex::Female), 18.0, 6.0),
                ((50, 69), Some(crate::demographics::Sex::Male), 18.0, 7.0),
                ((50, 69), Some(crate::demographics::Sex::Female), 15.0, 5.0),
                ((70, 89), Some(crate::demographics::Sex::Male), 14.0, 6.0),
                ((70, 89), Some(crate::demographics::Sex::Female), 12.0, 5.0),
            ],
        );
        self.set_reliability(MetricType::MaxPhonationTime, 0.88);

        // === Vestibular Metrics ===

        // VOR Gain
        self.add_age_sex_norms(
            MetricType::VorGain,
            &[
                ((18, 39), None, 1.0, 0.08),
                ((40, 59), None, 0.95, 0.10),
                ((60, 79), None, 0.85, 0.12),
                ((80, 89), None, 0.75, 0.15),
            ],
        );
        self.set_reliability(MetricType::VorGain, 0.88);

        // Canal Paresis (%) - asymmetry measure
        self.add_age_sex_norms(
            MetricType::CanalParesis,
            &[
                ((18, 39), None, 8.0, 5.0),
                ((40, 59), None, 10.0, 6.0),
                ((60, 79), None, 14.0, 8.0),
                ((80, 89), None, 18.0, 10.0),
            ],
        );
        self.set_reliability(MetricType::CanalParesis, 0.82);

        // DVA Score Loss (logMAR)
        self.add_age_sex_norms(
            MetricType::DvaScoreLoss,
            &[
                ((18, 39), None, 0.05, 0.03),
                ((40, 59), None, 0.08, 0.04),
                ((60, 79), None, 0.12, 0.06),
                ((80, 89), None, 0.18, 0.08),
            ],
        );
        self.set_reliability(MetricType::DvaScoreLoss, 0.85);

        // SVV Error (°)
        self.add_age_sex_norms(
            MetricType::SvvError,
            &[
                ((18, 39), None, 1.5, 1.0),
                ((40, 59), None, 2.0, 1.2),
                ((60, 79), None, 2.8, 1.5),
                ((80, 89), None, 4.0, 2.0),
            ],
        );
        self.set_reliability(MetricType::SvvError, 0.80);

        // Head Impulse Gain
        self.add_age_sex_norms(
            MetricType::HeadImpulseGain,
            &[
                ((18, 39), None, 1.0, 0.08),
                ((40, 59), None, 0.95, 0.10),
                ((60, 79), None, 0.88, 0.12),
                ((80, 89), None, 0.78, 0.15),
            ],
        );
        self.set_reliability(MetricType::HeadImpulseGain, 0.85);
    }

    /// Helper to add multiple normative entries
    fn add_age_sex_norms(
        &mut self,
        metric: MetricType,
        data: &[((u8, u8), Option<crate::demographics::Sex>, f64, f64)],
    ) {
        for &(age_range, sex, mean, std_dev) in data {
            let icc = self.reliability.get(&metric).copied().unwrap_or(0.85);
            let stats = NormativeStats::with_reliability(mean, std_dev, icc);

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
