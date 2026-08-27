//! Computerized Dynamic Posturography (CDP)
//!
//! This module provides algorithms for analyzing balance and postural control
//! using dynamic posturography, including the Sensory Organization Test (SOT).

/// Sensory Organization Test conditions
///
/// The SOT systematically manipulates visual and somatosensory inputs
/// to assess how well a patient can use each sensory system for balance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SotCondition {
    /// Eyes open, fixed surface, fixed visual surround
    /// Tests: Normal sensory integration
    Condition1,
    /// Eyes closed, fixed surface
    /// Tests: Balance without vision (somatosensory + vestibular)
    Condition2,
    /// Eyes open, fixed surface, sway-referenced visual surround
    /// Tests: Balance with inaccurate vision
    Condition3,
    /// Eyes open, sway-referenced surface, fixed visual surround
    /// Tests: Balance with inaccurate somatosensory
    Condition4,
    /// Eyes closed, sway-referenced surface
    /// Tests: Vestibular-only balance
    Condition5,
    /// Eyes open, sway-referenced surface and visual surround
    /// Tests: Vestibular-only with visual conflict
    Condition6,
}

impl SotCondition {
    /// Get all conditions in order
    pub fn all() -> [SotCondition; 6] {
        [
            SotCondition::Condition1,
            SotCondition::Condition2,
            SotCondition::Condition3,
            SotCondition::Condition4,
            SotCondition::Condition5,
            SotCondition::Condition6,
        ]
    }

    /// Get condition number (1-6)
    pub fn number(&self) -> usize {
        match self {
            SotCondition::Condition1 => 1,
            SotCondition::Condition2 => 2,
            SotCondition::Condition3 => 3,
            SotCondition::Condition4 => 4,
            SotCondition::Condition5 => 5,
            SotCondition::Condition6 => 6,
        }
    }
}

/// Results from Sensory Organization Test
#[derive(Debug, Clone, Default)]
pub struct SotResults {
    /// Equilibrium scores for each condition (0-100)
    pub equilibrium_scores: [f64; 6],
    /// Composite score (weighted average)
    pub composite_score: f64,
    /// Sensory analysis ratios
    pub sensory_analysis: SensoryAnalysis,
    /// Strategy analysis (ankle vs hip)
    pub strategy_scores: [f64; 6],
}

/// Sensory analysis ratios from SOT
///
/// These ratios indicate how well each sensory system is being used
/// for balance control.
#[derive(Debug, Clone, Default)]
pub struct SensoryAnalysis {
    /// Somatosensory ratio: Condition 2 / Condition 1
    /// Normal: > 0.90
    /// Indicates ability to use somatosensory input
    pub somatosensory_ratio: f64,

    /// Visual ratio: Condition 4 / Condition 1
    /// Normal: > 0.80
    /// Indicates ability to use visual input
    pub visual_ratio: f64,

    /// Vestibular ratio: Condition 5 / Condition 1
    /// Normal: > 0.65
    /// Indicates ability to use vestibular input alone
    pub vestibular_ratio: f64,

    /// Visual preference ratio: (Cond3 + Cond6) / (Cond2 + Cond5)
    /// Normal: ~1.0
    /// > 1.0 indicates over-reliance on vision
    pub visual_preference_ratio: f64,
}

/// Dynamic posturography analyzer
#[derive(Debug, Clone)]
pub struct DynamicPosturography {
    /// Sample rate of sway data (Hz)
    pub sample_rate: f64,
    /// Maximum theoretical sway angle (degrees)
    pub max_sway_angle: f64,
}

impl Default for DynamicPosturography {
    fn default() -> Self {
        Self {
            sample_rate: 100.0,
            max_sway_angle: 12.5, // Standard for NeuroCom
        }
    }
}

impl DynamicPosturography {
    /// Create a new analyzer with specified sample rate
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            max_sway_angle: 12.5,
        }
    }

    /// Calculate equilibrium score from sway angle data
    ///
    /// Equilibrium score = (max_sway - actual_sway) / max_sway * 100
    /// Score of 100 = perfect stability, 0 = fall
    pub fn equilibrium_score(&self, sway_angles: &[f64]) -> f64 {
        if sway_angles.is_empty() {
            return 0.0;
        }

        // Find peak-to-peak sway
        let max_sway = sway_angles
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_sway = sway_angles
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let peak_to_peak = max_sway - min_sway;

        // Calculate equilibrium score
        // Theoretical maximum sway before fall is ~12.5 degrees (anterior + posterior)
        let score = ((self.max_sway_angle - peak_to_peak) / self.max_sway_angle) * 100.0;

        score.clamp(0.0, 100.0)
    }

    /// Calculate strategy score (ankle vs hip strategy)
    ///
    /// Based on shear force relative to sway:
    /// - High score (near 100) = ankle strategy (normal)
    /// - Low score (near 0) = hip strategy
    pub fn strategy_score(&self, sway_angles: &[f64], shear_forces: &[f64]) -> f64 {
        if sway_angles.is_empty() || shear_forces.is_empty() {
            return 50.0;
        }

        let sway_range = {
            let max = sway_angles.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = sway_angles.iter().cloned().fold(f64::INFINITY, f64::min);
            max - min
        };

        let shear_range = {
            let max = shear_forces.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = shear_forces.iter().cloned().fold(f64::INFINITY, f64::min);
            max - min
        };

        if sway_range < 0.001 {
            return 100.0;
        }

        // Normalize and calculate ratio
        // Higher shear relative to sway = hip strategy
        let ratio = shear_range / (sway_range * 10.0); // Scaling factor

        (100.0 - ratio * 100.0).clamp(0.0, 100.0)
    }

    /// Analyze complete Sensory Organization Test
    ///
    /// Takes equilibrium scores from all 6 conditions
    pub fn sensory_organization_test(&self, condition_scores: &[f64; 6]) -> SotResults {
        let composite = Self::composite_score(condition_scores);
        let sensory_analysis = Self::sensory_analysis(condition_scores);

        SotResults {
            equilibrium_scores: *condition_scores,
            composite_score: composite,
            sensory_analysis,
            strategy_scores: [0.0; 6], // Would need shear force data
        }
    }

    /// Calculate composite SOT score
    ///
    /// Weighted average giving more weight to difficult conditions
    pub fn composite_score(scores: &[f64; 6]) -> f64 {
        // Standard NeuroCom weighting:
        // Conditions 1 & 2: averaged (1 trial weight)
        // Conditions 3-6: each condition averaged across 3 trials (3 trial weights each)
        // Total: 2 + 12 = 14 trial weights

        let weighted_sum = scores[0] + scores[1] + // Conditions 1-2
            3.0 * (scores[2] + scores[3] + scores[4] + scores[5]); // Conditions 3-6

        weighted_sum / 14.0
    }

    /// Calculate sensory analysis ratios
    pub fn sensory_analysis(scores: &[f64; 6]) -> SensoryAnalysis {
        let c1 = scores[0].max(0.01); // Avoid division by zero

        SensoryAnalysis {
            somatosensory_ratio: scores[1] / c1,
            visual_ratio: scores[3] / c1,
            vestibular_ratio: scores[4] / c1,
            visual_preference_ratio: if (scores[1] + scores[4]) > 0.01 {
                (scores[2] + scores[5]) / (scores[1] + scores[4])
            } else {
                1.0
            },
        }
    }

    /// Interpret sensory analysis results
    pub fn interpret_sensory_analysis(analysis: &SensoryAnalysis) -> SensoryInterpretation {
        SensoryInterpretation {
            somatosensory_dysfunction: analysis.somatosensory_ratio < 0.90,
            visual_dysfunction: analysis.visual_ratio < 0.80,
            vestibular_dysfunction: analysis.vestibular_ratio < 0.65,
            visual_preference: analysis.visual_preference_ratio > 1.10,
        }
    }
}

/// Interpretation of sensory analysis
#[derive(Debug, Clone, Default)]
pub struct SensoryInterpretation {
    /// Possible somatosensory system dysfunction
    pub somatosensory_dysfunction: bool,
    /// Possible visual system dysfunction for balance
    pub visual_dysfunction: bool,
    /// Possible vestibular system dysfunction
    pub vestibular_dysfunction: bool,
    /// Over-reliance on visual input
    pub visual_preference: bool,
}

/// Motor Control Test (MCT) results
#[derive(Debug, Clone, Default)]
pub struct MctResults {
    /// Latency for small backward translations (ms)
    pub latency_small_backward: f64,
    /// Latency for medium backward translations (ms)
    pub latency_medium_backward: f64,
    /// Latency for large backward translations (ms)
    pub latency_large_backward: f64,
    /// Latency for small forward translations (ms)
    pub latency_small_forward: f64,
    /// Latency for medium forward translations (ms)
    pub latency_medium_forward: f64,
    /// Latency for large forward translations (ms)
    pub latency_large_forward: f64,
    /// Weight symmetry (left vs right)
    pub weight_symmetry: f64,
    /// Amplitude scaling
    pub amplitude_scaling: f64,
}

/// Limits of Stability (LOS) test results
#[derive(Debug, Clone, Default)]
pub struct LosResults {
    /// Reaction time to movement onset (seconds)
    pub reaction_time: f64,
    /// Movement velocity (degrees/second)
    pub movement_velocity: f64,
    /// Endpoint excursion (% of theoretical maximum)
    pub endpoint_excursion: f64,
    /// Maximum excursion (% of theoretical maximum)
    pub maximum_excursion: f64,
    /// Directional control (%)
    pub directional_control: f64,
}

/// Limits of Stability analyzer
#[derive(Debug, Clone)]
pub struct LimitsOfStability {
    /// Sample rate (Hz)
    pub sample_rate: f64,
    /// Theoretical maximum lean distance
    pub max_lean: f64,
}

impl Default for LimitsOfStability {
    fn default() -> Self {
        Self {
            sample_rate: 100.0,
            max_lean: 8.0, // degrees
        }
    }
}

impl LimitsOfStability {
    /// Create new analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            max_lean: 8.0,
        }
    }

    /// Analyze a single LOS trial
    ///
    /// cop_x, cop_y: Center of pressure trajectory
    /// target_x, target_y: Target position
    pub fn analyze_trial(
        &self,
        cop_x: &[f64],
        cop_y: &[f64],
        target_x: f64,
        target_y: f64,
    ) -> LosResults {
        if cop_x.len() < 10 || cop_y.len() < 10 {
            return LosResults::default();
        }

        let n = cop_x.len().min(cop_y.len());

        // Calculate path from origin
        let path_distances: Vec<f64> = (0..n)
            .map(|i| (cop_x[i].powi(2) + cop_y[i].powi(2)).sqrt())
            .collect();

        // Reaction time: time to first movement > 10% of path
        let max_distance = path_distances.iter().cloned().fold(0.0, f64::max);
        let threshold = max_distance * 0.1;
        let reaction_samples = path_distances
            .iter()
            .position(|&d| d > threshold)
            .unwrap_or(0);
        let reaction_time = reaction_samples as f64 / self.sample_rate;

        // Movement velocity: average velocity during movement phase
        let mut velocities = Vec::new();
        for i in 1..n {
            let dx = cop_x[i] - cop_x[i - 1];
            let dy = cop_y[i] - cop_y[i - 1];
            let dist = (dx * dx + dy * dy).sqrt();
            velocities.push(dist * self.sample_rate);
        }
        let movement_velocity = if !velocities.is_empty() {
            velocities.iter().sum::<f64>() / velocities.len() as f64
        } else {
            0.0
        };

        // Endpoint excursion: final position as % of target
        let target_distance = (target_x.powi(2) + target_y.powi(2)).sqrt();
        let endpoint_excursion = if target_distance > 0.001 {
            (path_distances.last().unwrap_or(&0.0) / target_distance) * 100.0
        } else {
            0.0
        };

        // Maximum excursion
        let maximum_excursion = if target_distance > 0.001 {
            (max_distance / target_distance) * 100.0
        } else {
            0.0
        };

        // Directional control: how straight was the path
        // 100% = perfectly straight to target
        let path_length: f64 = (1..n)
            .map(|i| {
                let dx = cop_x[i] - cop_x[i - 1];
                let dy = cop_y[i] - cop_y[i - 1];
                (dx * dx + dy * dy).sqrt()
            })
            .sum();

        let directional_control = if path_length > 0.001 {
            ((max_distance / path_length) * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        };

        LosResults {
            reaction_time,
            movement_velocity,
            endpoint_excursion,
            maximum_excursion,
            directional_control,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equilibrium_score() {
        let cdp = DynamicPosturography::default();

        // Perfect stability (no sway)
        let perfect = vec![0.0; 100];
        let score = cdp.equilibrium_score(&perfect);
        assert!((score - 100.0).abs() < 0.01);

        // Maximum sway (should score 0)
        let max_sway: Vec<f64> = (0..100)
            .map(|i| if i < 50 { -6.25 } else { 6.25 })
            .collect();
        let score = cdp.equilibrium_score(&max_sway);
        assert!(score < 5.0);
    }

    #[test]
    fn test_composite_score() {
        // All perfect scores
        let scores = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
        let composite = DynamicPosturography::composite_score(&scores);
        assert!((composite - 100.0).abs() < 0.01);

        // All zeros
        let scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let composite = DynamicPosturography::composite_score(&scores);
        assert!(composite.abs() < 0.01);
    }

    #[test]
    fn test_sensory_analysis() {
        // Normal pattern
        let scores = [95.0, 90.0, 85.0, 80.0, 70.0, 65.0];
        let analysis = DynamicPosturography::sensory_analysis(&scores);

        assert!(analysis.somatosensory_ratio > 0.9);
        assert!(analysis.visual_ratio > 0.8);
        assert!(analysis.vestibular_ratio > 0.7);
    }

    #[test]
    fn test_sot_conditions() {
        let conditions = SotCondition::all();
        assert_eq!(conditions.len(), 6);
        assert_eq!(conditions[0].number(), 1);
        assert_eq!(conditions[5].number(), 6);
    }
}
