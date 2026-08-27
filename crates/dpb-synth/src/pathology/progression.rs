//! Disease progression modeling.
//!
//! Models temporal evolution of disease states including:
//! - Monotonic decline (ALS, Parkinson's)
//! - Relapsing-remitting patterns (MS)
//! - Recovery trajectories (stroke)
//! - Fluctuating symptoms (Parkinson's)

use super::DiseaseStage;
use serde::{Deserialize, Serialize};

/// Time point in disease course
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePoint {
    /// Time from reference (disease onset, diagnosis, etc.)
    pub time: f64,
    /// Time unit
    pub unit: TimeUnit,
    /// Disease stage at this point
    pub stage: DiseaseStage,
    /// Functional score (normalized 0-1)
    pub functional_score: f64,
    /// Key events at this time
    pub events: Vec<String>,
}

/// Time unit for progression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeUnit {
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

impl TimeUnit {
    /// Convert to days
    pub fn to_days(self, value: f64) -> f64 {
        match self {
            TimeUnit::Hours => value / 24.0,
            TimeUnit::Days => value,
            TimeUnit::Weeks => value * 7.0,
            TimeUnit::Months => value * 30.44,
            TimeUnit::Years => value * 365.25,
        }
    }
}

/// Progression rate classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProgressionRate {
    /// Slower than typical
    Slow,
    /// Typical progression
    Typical,
    /// Faster than typical
    Fast,
    /// Very aggressive
    Aggressive,
}

impl ProgressionRate {
    /// Get multiplier relative to typical
    pub fn multiplier(&self) -> f64 {
        match self {
            ProgressionRate::Slow => 0.5,
            ProgressionRate::Typical => 1.0,
            ProgressionRate::Fast => 1.5,
            ProgressionRate::Aggressive => 2.5,
        }
    }
}

/// Progression pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProgressionPattern {
    /// Steady monotonic decline
    LinearDecline,
    /// Exponential decline (faster early)
    ExponentialDecline,
    /// Step-wise decline with plateaus
    StepwiseDecline,
    /// Relapsing-remitting with recovery
    RelapsingRemitting,
    /// Progressive with superimposed relapses
    ProgressiveRelapsing,
    /// Recovery trajectory (stroke)
    Recovery,
    /// Fluctuating (Parkinson's)
    Fluctuating,
}

/// Disease progression model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiseaseProgression {
    /// Disease name
    pub disease: String,
    /// Progression pattern
    pub pattern: ProgressionPattern,
    /// Progression rate
    pub rate: ProgressionRate,
    /// Time points
    pub trajectory: Vec<TimePoint>,
    /// Initial functional score
    pub initial_score: f64,
    /// Final functional score (if known)
    pub final_score: Option<f64>,
    /// Time to key milestones (in days)
    pub milestones: Vec<(String, f64)>,
}

impl DiseaseProgression {
    /// Generate ALS progression trajectory
    pub fn als(rate: ProgressionRate, months: u32) -> Self {
        let rate_mult = rate.multiplier();
        let mut trajectory = Vec::new();

        for m in 0..=months {
            let t = m as f64;
            // ALS: roughly linear decline of ~0.9 ALSFRS points per month
            let decline = 0.9 * rate_mult * t;
            let score = (1.0 - decline / 48.0).max(0.0);

            let stage = match score {
                x if x > 0.85 => DiseaseStage::Early,
                x if x > 0.60 => DiseaseStage::Moderate,
                x if x > 0.35 => DiseaseStage::Advanced,
                _ => DiseaseStage::EndStage,
            };

            let events = if m == 0 {
                vec!["diagnosis".to_string()]
            } else if score < 0.75 && m > 0 && trajectory.last().map(|p: &TimePoint| p.functional_score >= 0.75).unwrap_or(false) {
                vec!["second_region".to_string()]
            } else if score < 0.50 && trajectory.last().map(|p: &TimePoint| p.functional_score >= 0.50).unwrap_or(false) {
                vec!["third_region".to_string()]
            } else if score < 0.30 && trajectory.last().map(|p: &TimePoint| p.functional_score >= 0.30).unwrap_or(false) {
                vec!["niv_required".to_string()]
            } else {
                vec![]
            };

            trajectory.push(TimePoint {
                time: t,
                unit: TimeUnit::Months,
                stage,
                functional_score: score,
                events,
            });
        }

        Self {
            disease: "als".to_string(),
            pattern: ProgressionPattern::LinearDecline,
            rate,
            trajectory,
            initial_score: 1.0,
            final_score: Some(0.0),
            milestones: vec![
                ("diagnosis".to_string(), 0.0),
                ("second_region".to_string(), 10.0 * 30.0 / rate_mult),
                ("respiratory_support".to_string(), 24.0 * 30.0 / rate_mult),
            ],
        }
    }

    /// Generate MS relapsing-remitting trajectory
    pub fn ms_rrms(years: u32, relapses_per_year: f64) -> Self {
        let mut trajectory = Vec::new();
        let mut current_score = 1.0;
        let mut current_stage;

        // Simulate monthly with relapses
        let total_months = years * 12;
        let relapse_prob_per_month = relapses_per_year / 12.0;

        let mut rng_state: u64 = 42;
        let pseudo_random = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(1103515245).wrapping_add(12345);
            (*state as f64) / (u64::MAX as f64)
        };

        for m in 0..=total_months {
            let mut events = vec![];

            // Check for relapse
            if pseudo_random(&mut rng_state) < relapse_prob_per_month {
                let severity = 0.05 + pseudo_random(&mut rng_state) * 0.15;
                current_score = (current_score - severity).max(0.1);
                events.push("relapse".to_string());

                // Partial recovery over next 2-3 months
                let recovery = severity * 0.7;
                current_score = (current_score + recovery).min(1.0);
            }

            // Background progression (slow)
            current_score = (current_score - 0.002).max(0.1);

            current_stage = match current_score {
                x if x > 0.85 => DiseaseStage::Prodromal,
                x if x > 0.65 => DiseaseStage::Early,
                x if x > 0.45 => DiseaseStage::Moderate,
                x if x > 0.25 => DiseaseStage::Advanced,
                _ => DiseaseStage::EndStage,
            };

            if m % 12 == 0 { // Record yearly
                trajectory.push(TimePoint {
                    time: (m / 12) as f64,
                    unit: TimeUnit::Years,
                    stage: current_stage,
                    functional_score: current_score,
                    events,
                });
            }
        }

        Self {
            disease: "ms_rrms".to_string(),
            pattern: ProgressionPattern::RelapsingRemitting,
            rate: ProgressionRate::Typical,
            trajectory,
            initial_score: 1.0,
            final_score: None,
            milestones: vec![
                ("diagnosis".to_string(), 0.0),
            ],
        }
    }

    /// Generate stroke recovery trajectory
    pub fn stroke_recovery(initial_nihss: u8, days: u32) -> Self {
        let mut trajectory = Vec::new();
        let initial_score = 1.0 - (initial_nihss as f64 / 42.0);

        for d in 0..=days {
            // Recovery follows logarithmic curve
            // Most recovery in first 3 months
            let t = d as f64;
            let recovery_potential = match initial_nihss {
                0..=4 => 0.9,
                5..=15 => 0.7,
                16..=20 => 0.5,
                _ => 0.3,
            };

            // Logarithmic recovery: score = initial + recovery * (1 - exp(-t/tau))
            let tau = 60.0; // time constant in days
            let recovered_fraction = 1.0 - (-t / tau).exp();
            let score = initial_score + (1.0 - initial_score) * recovery_potential * recovered_fraction;

            let stage = match score {
                x if x > 0.85 => DiseaseStage::Early,
                x if x > 0.65 => DiseaseStage::Moderate,
                x if x > 0.45 => DiseaseStage::Advanced,
                _ => DiseaseStage::EndStage,
            };

            let events = if d == 0 {
                vec!["stroke_onset".to_string()]
            } else if d == 7 {
                vec!["acute_phase_end".to_string()]
            } else if d == 90 {
                vec!["subacute_phase_end".to_string()]
            } else {
                vec![]
            };

            if d % 7 == 0 { // Record weekly
                trajectory.push(TimePoint {
                    time: d as f64,
                    unit: TimeUnit::Days,
                    stage,
                    functional_score: score,
                    events,
                });
            }
        }

        Self {
            disease: "stroke".to_string(),
            pattern: ProgressionPattern::Recovery,
            rate: ProgressionRate::Typical,
            trajectory,
            initial_score,
            final_score: None,
            milestones: vec![
                ("stroke_onset".to_string(), 0.0),
                ("hyperacute_end".to_string(), 1.0),
                ("acute_end".to_string(), 7.0),
                ("subacute_end".to_string(), 90.0),
                ("chronic_phase".to_string(), 180.0),
            ],
        }
    }

    /// Get functional score at given time
    pub fn score_at(&self, time: f64, unit: TimeUnit) -> Option<f64> {
        let target_days = unit.to_days(time);

        // Find bracketing points
        let mut prev: Option<&TimePoint> = None;
        for point in &self.trajectory {
            let point_days = point.unit.to_days(point.time);

            if point_days >= target_days {
                if let Some(p) = prev {
                    // Interpolate
                    let prev_days = p.unit.to_days(p.time);
                    let fraction = (target_days - prev_days) / (point_days - prev_days);
                    return Some(p.functional_score + fraction * (point.functional_score - p.functional_score));
                } else {
                    return Some(point.functional_score);
                }
            }
            prev = Some(point);
        }

        self.trajectory.last().map(|p| p.functional_score)
    }

    /// Get stage at given time
    pub fn stage_at(&self, time: f64, unit: TimeUnit) -> Option<DiseaseStage> {
        let target_days = unit.to_days(time);

        for point in self.trajectory.iter().rev() {
            let point_days = point.unit.to_days(point.time);
            if point_days <= target_days {
                return Some(point.stage);
            }
        }

        self.trajectory.first().map(|p| p.stage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_als_progression() {
        let prog = DiseaseProgression::als(ProgressionRate::Typical, 36);

        assert!(!prog.trajectory.is_empty());
        assert_eq!(prog.trajectory[0].functional_score, 1.0);
        assert!(prog.trajectory.last().unwrap().functional_score < 0.5);
    }

    #[test]
    fn test_ms_progression() {
        let prog = DiseaseProgression::ms_rrms(10, 0.5);

        assert!(!prog.trajectory.is_empty());
        // Should have some relapses recorded
    }

    #[test]
    fn test_stroke_recovery() {
        let prog = DiseaseProgression::stroke_recovery(12, 180);

        assert!(!prog.trajectory.is_empty());
        // Score should improve over time
        let initial = prog.trajectory[0].functional_score;
        let final_score = prog.trajectory.last().unwrap().functional_score;
        assert!(final_score > initial);
    }

    #[test]
    fn test_score_interpolation() {
        let prog = DiseaseProgression::als(ProgressionRate::Typical, 12);

        let score = prog.score_at(6.0, TimeUnit::Months).unwrap();
        assert!(score < 1.0);
        assert!(score > 0.5);
    }

    #[test]
    fn test_time_unit_conversion() {
        assert_eq!(TimeUnit::Days.to_days(1.0), 1.0);
        assert_eq!(TimeUnit::Weeks.to_days(1.0), 7.0);
        assert_eq!(TimeUnit::Years.to_days(1.0), 365.25);
    }
}
