//! QbTest-style continuous performance test
//!
//! Implements a comprehensive attention/impulsivity test similar to QbTest,
//! measuring attention, impulsivity, and activity during a sustained attention task.

use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};

/// QbTest-style continuous performance test
#[derive(Debug, Clone)]
pub struct QbTest {
    /// Test duration in minutes (typically 15-20)
    pub duration_minutes: f64,
    /// Target probability (typically 0.25)
    pub target_probability: f64,
    /// Stimulus duration in milliseconds
    pub stimulus_duration_ms: f64,
    /// Inter-stimulus interval in milliseconds
    pub inter_stimulus_interval_ms: f64,
}

impl Default for QbTest {
    fn default() -> Self {
        Self {
            duration_minutes: 15.0,
            target_probability: 0.25,
            stimulus_duration_ms: 200.0,
            inter_stimulus_interval_ms: 1800.0,
        }
    }
}

impl QbTest {
    /// Create new QbTest with custom parameters
    pub fn new(duration_minutes: f64, target_probability: f64) -> Self {
        Self {
            duration_minutes,
            target_probability,
            ..Default::default()
        }
    }

    /// Calculate total number of stimuli in the test
    pub fn total_stimuli(&self) -> usize {
        let trial_duration_ms = self.stimulus_duration_ms + self.inter_stimulus_interval_ms;
        let total_time_ms = self.duration_minutes * 60.0 * 1000.0;
        (total_time_ms / trial_duration_ms) as usize
    }

    /// Generate test stimuli sequence
    pub fn generate_stimuli(&self, seed: Option<u64>) -> Vec<QbTestStimulus> {
        let n_stimuli = self.total_stimuli();
        let mut stimuli = Vec::with_capacity(n_stimuli);

        let mut rng: StdRng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => rand::make_rng::<StdRng>(),
        };

        let trial_duration_ms = self.stimulus_duration_ms + self.inter_stimulus_interval_ms;

        for i in 0..n_stimuli {
            let is_target: bool = rng.random::<f64>() < self.target_probability;
            stimuli.push(QbTestStimulus {
                index: i,
                onset_time_ms: i as f64 * trial_duration_ms,
                is_target,
                stimulus_type: if is_target {
                    StimulusType::Target
                } else {
                    StimulusType::NonTarget
                },
            });
        }

        stimuli
    }

    /// Score responses and generate comprehensive metrics
    pub fn score_responses(
        &self,
        stimuli: &[QbTestStimulus],
        responses: &[QbTestResponse],
    ) -> QbTestMetrics {
        let attention = self.score_attention(stimuli, responses);
        let impulsivity = self.score_impulsivity(stimuli, responses);

        QbTestMetrics {
            attention,
            impulsivity,
            activity: ActivityMetrics::default(), // Filled from motion data separately
        }
    }

    /// Score attention-related metrics
    pub fn score_attention(
        &self,
        stimuli: &[QbTestStimulus],
        responses: &[QbTestResponse],
    ) -> AttentionMetrics {
        let target_stimuli: Vec<&QbTestStimulus> =
            stimuli.iter().filter(|s| s.is_target).collect();

        let mut hits = 0;
        let mut omissions = 0;
        let mut reaction_times = Vec::new();

        for target in &target_stimuli {
            // Find response within valid window (100-1800ms after stimulus)
            let valid_response = responses.iter().find(|r| {
                let rt = r.response_time_ms - target.onset_time_ms;
                rt >= 100.0 && rt <= self.inter_stimulus_interval_ms
            });

            match valid_response {
                Some(resp) => {
                    hits += 1;
                    reaction_times.push(resp.response_time_ms - target.onset_time_ms);
                }
                None => {
                    omissions += 1;
                }
            }
        }

        let mean_rt = if reaction_times.is_empty() {
            0.0
        } else {
            reaction_times.iter().sum::<f64>() / reaction_times.len() as f64
        };

        let rt_variability = if reaction_times.len() < 2 {
            0.0
        } else {
            let variance: f64 = reaction_times
                .iter()
                .map(|rt| (rt - mean_rt).powi(2))
                .sum::<f64>()
                / (reaction_times.len() - 1) as f64;
            variance.sqrt()
        };

        // Calculate coefficient of variation
        let rt_cv = if mean_rt > 0.0 {
            rt_variability / mean_rt * 100.0
        } else {
            0.0
        };

        AttentionMetrics {
            omission_errors: omissions,
            reaction_time_ms: mean_rt,
            rt_variability_ms: rt_variability,
            rt_coefficient_of_variation: rt_cv,
            hit_rate: if target_stimuli.is_empty() {
                0.0
            } else {
                hits as f64 / target_stimuli.len() as f64
            },
            normative_percentile: 0.0, // Filled from norms
        }
    }

    /// Score impulsivity-related metrics
    pub fn score_impulsivity(
        &self,
        stimuli: &[QbTestStimulus],
        responses: &[QbTestResponse],
    ) -> ImpulsivityMetrics {
        let non_target_stimuli: Vec<&QbTestStimulus> =
            stimuli.iter().filter(|s| !s.is_target).collect();

        let mut commission_errors = 0;
        let mut anticipatory_responses = 0;
        let mut multiresponses = 0;

        // Count commission errors (responding to non-targets)
        for non_target in &non_target_stimuli {
            let responses_to_stimulus: Vec<&QbTestResponse> = responses
                .iter()
                .filter(|r| {
                    let rt = r.response_time_ms - non_target.onset_time_ms;
                    rt >= 100.0 && rt <= self.inter_stimulus_interval_ms
                })
                .collect();

            if !responses_to_stimulus.is_empty() {
                commission_errors += 1;
            }

            if responses_to_stimulus.len() > 1 {
                multiresponses += responses_to_stimulus.len() - 1;
            }
        }

        // Count anticipatory responses (RT < 100ms)
        for stimulus in stimuli {
            let antic = responses.iter().any(|r| {
                let rt = r.response_time_ms - stimulus.onset_time_ms;
                rt >= 0.0 && rt < 100.0
            });
            if antic {
                anticipatory_responses += 1;
            }
        }

        let false_alarm_rate = if non_target_stimuli.is_empty() {
            0.0
        } else {
            commission_errors as f64 / non_target_stimuli.len() as f64
        };

        ImpulsivityMetrics {
            commission_errors,
            anticipatory_responses,
            multiresponses,
            false_alarm_rate,
            normative_percentile: 0.0, // Filled from norms
        }
    }

    /// Score activity metrics from head/body motion tracking
    pub fn score_activity(&self, motion_data: &[MotionSample]) -> ActivityMetrics {
        if motion_data.is_empty() {
            return ActivityMetrics::default();
        }

        // Calculate total distance traveled
        let mut total_distance = 0.0;
        for i in 1..motion_data.len() {
            let dx = motion_data[i].x - motion_data[i - 1].x;
            let dy = motion_data[i].y - motion_data[i - 1].y;
            let dz = motion_data[i].z - motion_data[i - 1].z;
            total_distance += (dx * dx + dy * dy + dz * dz).sqrt();
        }

        // Calculate area covered (simplified as bounding box)
        let x_vals: Vec<f64> = motion_data.iter().map(|m| m.x).collect();
        let y_vals: Vec<f64> = motion_data.iter().map(|m| m.y).collect();

        let x_range = x_vals.iter().cloned().fold(f64::NAN, f64::max)
            - x_vals.iter().cloned().fold(f64::NAN, f64::min);
        let y_range = y_vals.iter().cloned().fold(f64::NAN, f64::max)
            - y_vals.iter().cloned().fold(f64::NAN, f64::min);

        let area = x_range * y_range;

        // Count microevents (rapid movements exceeding threshold)
        let velocity_threshold = 0.5; // Arbitrary threshold
        let mut microevents = 0;
        for i in 1..motion_data.len() {
            let dt = motion_data[i].time_ms - motion_data[i - 1].time_ms;
            if dt > 0.0 {
                let dx = motion_data[i].x - motion_data[i - 1].x;
                let dy = motion_data[i].y - motion_data[i - 1].y;
                let dz = motion_data[i].z - motion_data[i - 1].z;
                let velocity = (dx * dx + dy * dy + dz * dz).sqrt() / dt * 1000.0;
                if velocity > velocity_threshold {
                    microevents += 1;
                }
            }
        }

        ActivityMetrics {
            distance: total_distance,
            area,
            microevents,
            normative_percentile: 0.0, // Filled from norms
        }
    }

    /// Calculate d-prime (sensitivity) from hit rate and false alarm rate
    pub fn calculate_d_prime(&self, hit_rate: f64, false_alarm_rate: f64) -> f64 {
        // Clamp to avoid infinite z-scores
        let hr = hit_rate.clamp(0.01, 0.99);
        let far = false_alarm_rate.clamp(0.01, 0.99);

        z_score(hr) - z_score(far)
    }

    /// Calculate response bias (criterion)
    pub fn calculate_response_bias(&self, hit_rate: f64, false_alarm_rate: f64) -> f64 {
        let hr = hit_rate.clamp(0.01, 0.99);
        let far = false_alarm_rate.clamp(0.01, 0.99);

        -0.5 * (z_score(hr) + z_score(far))
    }

    /// Get normative percentile for attention (placeholder - needs normative data)
    pub fn attention_percentile(&self, metrics: &AttentionMetrics, age: u8) -> f64 {
        // Simplified normative lookup
        // In production, this would use age-specific normative tables
        let (mean_rt, sd_rt) = match age {
            6..=8 => (650.0, 150.0),
            9..=11 => (550.0, 120.0),
            12..=17 => (480.0, 100.0),
            _ => (450.0, 90.0),
        };

        let z = (metrics.reaction_time_ms - mean_rt) / sd_rt;
        // Invert so higher percentile = better (faster RT)
        (1.0 - normal_cdf(z)) * 100.0
    }

    /// Get normative percentile for impulsivity
    pub fn impulsivity_percentile(&self, metrics: &ImpulsivityMetrics, age: u8) -> f64 {
        // Normative data for commission errors
        let (mean_ce, sd_ce) = match age {
            6..=8 => (15.0, 8.0),
            9..=11 => (10.0, 6.0),
            12..=17 => (8.0, 5.0),
            _ => (6.0, 4.0),
        };

        let z = (metrics.commission_errors as f64 - mean_ce) / sd_ce;
        // Higher percentile = more errors = worse
        normal_cdf(z) * 100.0
    }
}

/// Stimulus in QbTest
#[derive(Debug, Clone)]
pub struct QbTestStimulus {
    /// Index in sequence
    pub index: usize,
    /// Onset time in milliseconds
    pub onset_time_ms: f64,
    /// Whether this is a target stimulus
    pub is_target: bool,
    /// Stimulus type
    pub stimulus_type: StimulusType,
}

/// Type of stimulus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StimulusType {
    /// Target - respond
    Target,
    /// Non-target - withhold response
    NonTarget,
}

/// Response to QbTest stimulus
#[derive(Debug, Clone)]
pub struct QbTestResponse {
    /// Time of response in milliseconds from test start
    pub response_time_ms: f64,
    /// Response type (button press, etc.)
    pub response_type: ResponseType,
}

/// Type of response
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseType {
    /// Primary button press
    ButtonPress,
    /// Secondary button press (if applicable)
    SecondaryPress,
}

/// Motion sample for activity tracking
#[derive(Debug, Clone)]
pub struct MotionSample {
    /// Time in milliseconds
    pub time_ms: f64,
    /// X position
    pub x: f64,
    /// Y position
    pub y: f64,
    /// Z position
    pub z: f64,
}

/// Comprehensive QbTest metrics
#[derive(Debug, Clone, Default)]
pub struct QbTestMetrics {
    /// Attention-related metrics
    pub attention: AttentionMetrics,
    /// Impulsivity-related metrics
    pub impulsivity: ImpulsivityMetrics,
    /// Activity-related metrics
    pub activity: ActivityMetrics,
}

impl QbTestMetrics {
    /// Calculate composite ADHD score
    pub fn composite_score(&self) -> f64 {
        // Simple weighted average of percentiles
        // Higher score = more ADHD-like pattern
        let attention_component = 100.0 - self.attention.normative_percentile;
        let impulsivity_component = self.impulsivity.normative_percentile;
        let activity_component = self.activity.normative_percentile;

        (attention_component + impulsivity_component + activity_component) / 3.0
    }

    /// Check if pattern suggests ADHD
    pub fn adhd_pattern(&self) -> AdhdPattern {
        let attention_impaired = self.attention.normative_percentile < 16.0;
        let impulsivity_elevated = self.impulsivity.normative_percentile > 84.0;
        let activity_elevated = self.activity.normative_percentile > 84.0;

        if attention_impaired && impulsivity_elevated && activity_elevated {
            AdhdPattern::Combined
        } else if attention_impaired && !impulsivity_elevated {
            AdhdPattern::Inattentive
        } else if impulsivity_elevated && activity_elevated && !attention_impaired {
            AdhdPattern::HyperactiveImpulsive
        } else {
            AdhdPattern::Normal
        }
    }
}

/// ADHD presentation pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdhdPattern {
    /// Normal pattern
    Normal,
    /// Predominantly inattentive
    Inattentive,
    /// Predominantly hyperactive/impulsive
    HyperactiveImpulsive,
    /// Combined presentation
    Combined,
}

/// Attention metrics
#[derive(Debug, Clone, Default)]
pub struct AttentionMetrics {
    /// Number of missed targets (omission errors)
    pub omission_errors: usize,
    /// Mean reaction time in milliseconds
    pub reaction_time_ms: f64,
    /// Reaction time variability (standard deviation)
    pub rt_variability_ms: f64,
    /// Coefficient of variation (CV) of reaction time
    pub rt_coefficient_of_variation: f64,
    /// Hit rate (proportion of targets detected)
    pub hit_rate: f64,
    /// Normative percentile (0-100)
    pub normative_percentile: f64,
}

/// Impulsivity metrics
#[derive(Debug, Clone, Default)]
pub struct ImpulsivityMetrics {
    /// Number of responses to non-targets (commission errors)
    pub commission_errors: usize,
    /// Number of anticipatory responses (RT < 100ms)
    pub anticipatory_responses: usize,
    /// Number of multiple responses to single stimulus
    pub multiresponses: usize,
    /// False alarm rate
    pub false_alarm_rate: f64,
    /// Normative percentile (0-100)
    pub normative_percentile: f64,
}

/// Activity metrics (from motion tracking)
#[derive(Debug, Clone, Default)]
pub struct ActivityMetrics {
    /// Total distance traveled
    pub distance: f64,
    /// Area covered (bounding box)
    pub area: f64,
    /// Number of rapid microevents
    pub microevents: usize,
    /// Normative percentile (0-100)
    pub normative_percentile: f64,
}

fn z_score(p: f64) -> f64 {
    // Approximation of inverse normal CDF
    let p = p.clamp(0.0001, 0.9999);

    if p < 0.5 {
        -rational_approximation((-2.0 * p.ln()).sqrt())
    } else {
        rational_approximation((-2.0 * (1.0 - p).ln()).sqrt())
    }
}

fn rational_approximation(t: f64) -> f64 {
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qbtest_generation() {
        let qb = QbTest::default();
        let stimuli = qb.generate_stimuli(Some(42));

        assert!(!stimuli.is_empty());

        // Check target probability is approximately correct
        let targets: Vec<_> = stimuli.iter().filter(|s| s.is_target).collect();
        let target_ratio = targets.len() as f64 / stimuli.len() as f64;
        assert!((target_ratio - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_attention_scoring() {
        let qb = QbTest::default();
        let stimuli = qb.generate_stimuli(Some(42));

        // Create responses for all targets with RT of 450ms
        let responses: Vec<QbTestResponse> = stimuli
            .iter()
            .filter(|s| s.is_target)
            .map(|s| QbTestResponse {
                response_time_ms: s.onset_time_ms + 450.0,
                response_type: ResponseType::ButtonPress,
            })
            .collect();

        let attention = qb.score_attention(&stimuli, &responses);

        assert_eq!(attention.omission_errors, 0);
        assert!((attention.reaction_time_ms - 450.0).abs() < 1.0);
        assert!((attention.hit_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_d_prime() {
        let qb = QbTest::default();

        // Perfect performance
        let d_perfect = qb.calculate_d_prime(0.99, 0.01);
        assert!(d_perfect > 3.0);

        // Chance performance
        let d_chance = qb.calculate_d_prime(0.5, 0.5);
        assert!(d_chance.abs() < 0.1);
    }

    #[test]
    fn test_adhd_pattern() {
        let mut metrics = QbTestMetrics::default();

        // Normal pattern
        metrics.attention.normative_percentile = 50.0;
        metrics.impulsivity.normative_percentile = 50.0;
        metrics.activity.normative_percentile = 50.0;
        assert_eq!(metrics.adhd_pattern(), AdhdPattern::Normal);

        // Inattentive pattern
        metrics.attention.normative_percentile = 10.0;
        metrics.impulsivity.normative_percentile = 50.0;
        metrics.activity.normative_percentile = 50.0;
        assert_eq!(metrics.adhd_pattern(), AdhdPattern::Inattentive);
    }
}
