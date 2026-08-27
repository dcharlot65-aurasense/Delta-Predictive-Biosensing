//! Executive function assessment tasks
//!
//! This module provides implementations of cognitive tasks that assess
//! executive functions including response inhibition, cognitive flexibility,
//! and interference control.

use rand::RngExt;

/// Go/No-Go task for assessing response inhibition
#[derive(Debug, Clone)]
pub struct GoNoGoTask {
    /// Probability of a Go trial (0.0 - 1.0)
    pub go_probability: f64,
    /// Duration of stimulus presentation in seconds
    pub stimulus_duration: f64,
    /// Maximum time allowed for response in seconds
    pub response_window: f64,
    /// Inter-stimulus interval in seconds
    pub isi: f64,
}

impl Default for GoNoGoTask {
    fn default() -> Self {
        Self {
            go_probability: 0.75, // 75% Go trials
            stimulus_duration: 0.5,
            response_window: 1.0,
            isi: 1.5,
        }
    }
}

/// A single Go/No-Go trial
#[derive(Debug, Clone)]
pub struct GoNoGoTrial {
    /// Trial number
    pub trial_number: usize,
    /// Whether this is a Go trial
    pub is_go: bool,
    /// Onset time in seconds from start
    pub onset_time: f64,
}

/// Response to a Go/No-Go trial
#[derive(Debug, Clone)]
pub struct GoNoGoResponse {
    /// Trial number
    pub trial_number: usize,
    /// Whether the participant responded
    pub responded: bool,
    /// Reaction time in seconds (if responded)
    pub reaction_time: Option<f64>,
}

/// Metrics from Go/No-Go task performance
#[derive(Debug, Clone, Default)]
pub struct GoNoGoMetrics {
    /// Number of commission errors (responding on No-Go trials)
    pub commission_errors: usize,
    /// Number of omission errors (not responding on Go trials)
    pub omission_errors: usize,
    /// Total Go trials
    pub total_go_trials: usize,
    /// Total No-Go trials
    pub total_nogo_trials: usize,
    /// Mean reaction time for correct Go responses (seconds)
    pub mean_rt_go: f64,
    /// Standard deviation of RT for correct Go responses
    pub std_rt_go: f64,
    /// Hit rate (correct Go responses / total Go trials)
    pub hit_rate: f64,
    /// False alarm rate (commission errors / total No-Go trials)
    pub false_alarm_rate: f64,
    /// d-prime (sensitivity)
    pub d_prime: f64,
}

impl GoNoGoTask {
    /// Create a new Go/No-Go task with specified parameters
    pub fn new(go_probability: f64, stimulus_duration: f64, response_window: f64) -> Self {
        Self {
            go_probability: go_probability.clamp(0.0, 1.0),
            stimulus_duration,
            response_window,
            isi: 1.5,
        }
    }

    /// Generate a sequence of Go/No-Go trials
    pub fn generate_trials(&self, n_trials: usize) -> Vec<GoNoGoTrial> {
        let mut rng = rand::rng();
        let mut trials = Vec::with_capacity(n_trials);
        let mut current_time = 0.0;

        for i in 0..n_trials {
            let is_go = rng.random::<f64>() < self.go_probability;
            trials.push(GoNoGoTrial {
                trial_number: i,
                is_go,
                onset_time: current_time,
            });
            current_time += self.stimulus_duration + self.isi;
        }

        trials
    }

    /// Score responses against trials
    pub fn score_responses(
        &self,
        trials: &[GoNoGoTrial],
        responses: &[GoNoGoResponse],
    ) -> GoNoGoMetrics {
        let mut metrics = GoNoGoMetrics::default();
        let mut correct_go_rts = Vec::new();

        // Create a map of trial numbers to responses
        let response_map: std::collections::HashMap<usize, &GoNoGoResponse> =
            responses.iter().map(|r| (r.trial_number, r)).collect();

        for trial in trials {
            let response = response_map.get(&trial.trial_number);

            if trial.is_go {
                metrics.total_go_trials += 1;
                match response {
                    Some(r) if r.responded => {
                        if let Some(rt) = r.reaction_time {
                            correct_go_rts.push(rt);
                        }
                    }
                    _ => {
                        metrics.omission_errors += 1;
                    }
                }
            } else {
                metrics.total_nogo_trials += 1;
                if let Some(r) = response
                    && r.responded
                {
                    metrics.commission_errors += 1;
                }
            }
        }

        // Calculate RT statistics
        if !correct_go_rts.is_empty() {
            metrics.mean_rt_go = correct_go_rts.iter().sum::<f64>() / correct_go_rts.len() as f64;

            if correct_go_rts.len() > 1 {
                let variance = correct_go_rts
                    .iter()
                    .map(|rt| (rt - metrics.mean_rt_go).powi(2))
                    .sum::<f64>()
                    / (correct_go_rts.len() - 1) as f64;
                metrics.std_rt_go = variance.sqrt();
            }
        }

        // Calculate rates
        if metrics.total_go_trials > 0 {
            metrics.hit_rate = (metrics.total_go_trials - metrics.omission_errors) as f64
                / metrics.total_go_trials as f64;
        }
        if metrics.total_nogo_trials > 0 {
            metrics.false_alarm_rate =
                metrics.commission_errors as f64 / metrics.total_nogo_trials as f64;
        }

        // Calculate d-prime
        metrics.d_prime = calculate_d_prime(metrics.hit_rate, metrics.false_alarm_rate);

        metrics
    }
}

/// Flanker task conditions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlankerCondition {
    /// Flankers point same direction as target (>>>>>, <<<<<)
    Congruent,
    /// Flankers point opposite direction (>><>>, <<><<)
    Incongruent,
    /// Neutral flankers (--<--, -->--)
    Neutral,
}

/// Flanker task for assessing selective attention and conflict monitoring
#[derive(Debug, Clone)]
pub struct FlankerTask {
    /// Number of trials per condition
    pub trials_per_condition: usize,
    /// Stimulus duration in seconds
    pub stimulus_duration: f64,
    /// Response window in seconds
    pub response_window: f64,
    /// Inter-stimulus interval in seconds
    pub isi: f64,
}

impl Default for FlankerTask {
    fn default() -> Self {
        Self {
            trials_per_condition: 40,
            stimulus_duration: 0.2,
            response_window: 1.5,
            isi: 1.0,
        }
    }
}

/// A single Flanker trial
#[derive(Debug, Clone)]
pub struct FlankerTrial {
    /// Trial number
    pub trial_number: usize,
    /// Condition type
    pub condition: FlankerCondition,
    /// Target direction (true = right, false = left)
    pub target_right: bool,
    /// Onset time in seconds
    pub onset_time: f64,
}

/// Response to a Flanker trial
#[derive(Debug, Clone)]
pub struct FlankerResponse {
    /// Trial number
    pub trial_number: usize,
    /// Response direction (true = right, false = left)
    pub response_right: bool,
    /// Reaction time in seconds
    pub reaction_time: f64,
}

/// Metrics from Flanker task performance
#[derive(Debug, Clone, Default)]
pub struct FlankerMetrics {
    /// Mean RT for congruent trials (seconds)
    pub congruent_rt: f64,
    /// Mean RT for incongruent trials (seconds)
    pub incongruent_rt: f64,
    /// Mean RT for neutral trials (seconds)
    pub neutral_rt: f64,
    /// Flanker effect (incongruent RT - congruent RT)
    pub flanker_effect: f64,
    /// Accuracy for congruent trials
    pub congruent_accuracy: f64,
    /// Accuracy for incongruent trials
    pub incongruent_accuracy: f64,
    /// Accuracy for neutral trials
    pub neutral_accuracy: f64,
    /// Conflict adaptation effect
    pub conflict_adaptation: f64,
}

impl FlankerTask {
    /// Create a new Flanker task
    pub fn new(trials_per_condition: usize) -> Self {
        Self {
            trials_per_condition,
            ..Default::default()
        }
    }

    /// Generate a randomized sequence of Flanker trials
    pub fn generate_trials(&self) -> Vec<FlankerTrial> {
        let mut rng = rand::rng();
        let conditions = [
            FlankerCondition::Congruent,
            FlankerCondition::Incongruent,
            FlankerCondition::Neutral,
        ];

        let mut trials = Vec::new();
        let mut trial_number = 0;

        for condition in &conditions {
            for _ in 0..self.trials_per_condition {
                trials.push(FlankerTrial {
                    trial_number,
                    condition: *condition,
                    target_right: rng.random(),
                    onset_time: 0.0, // Will be set after shuffling
                });
                trial_number += 1;
            }
        }

        // Shuffle trials
        use rand::seq::SliceRandom;
        trials.shuffle(&mut rng);

        // Set onset times
        let mut current_time = 0.0;
        for (i, trial) in trials.iter_mut().enumerate() {
            trial.trial_number = i;
            trial.onset_time = current_time;
            current_time += self.stimulus_duration + self.isi;
        }

        trials
    }

    /// Score responses against trials
    pub fn score_responses(
        &self,
        trials: &[FlankerTrial],
        responses: &[FlankerResponse],
    ) -> FlankerMetrics {
        let mut congruent_rts = Vec::new();
        let mut incongruent_rts = Vec::new();
        let mut neutral_rts = Vec::new();
        let mut congruent_correct = 0usize;
        let mut incongruent_correct = 0usize;
        let mut neutral_correct = 0usize;
        let mut congruent_total = 0usize;
        let mut incongruent_total = 0usize;
        let mut neutral_total = 0usize;

        let response_map: std::collections::HashMap<usize, &FlankerResponse> =
            responses.iter().map(|r| (r.trial_number, r)).collect();

        for trial in trials {
            if let Some(response) = response_map.get(&trial.trial_number) {
                let correct = response.response_right == trial.target_right;

                match trial.condition {
                    FlankerCondition::Congruent => {
                        congruent_total += 1;
                        if correct {
                            congruent_correct += 1;
                            congruent_rts.push(response.reaction_time);
                        }
                    }
                    FlankerCondition::Incongruent => {
                        incongruent_total += 1;
                        if correct {
                            incongruent_correct += 1;
                            incongruent_rts.push(response.reaction_time);
                        }
                    }
                    FlankerCondition::Neutral => {
                        neutral_total += 1;
                        if correct {
                            neutral_correct += 1;
                            neutral_rts.push(response.reaction_time);
                        }
                    }
                }
            }
        }

        let mean_rt = |rts: &[f64]| -> f64 {
            if rts.is_empty() {
                0.0
            } else {
                rts.iter().sum::<f64>() / rts.len() as f64
            }
        };

        let congruent_rt = mean_rt(&congruent_rts);
        let incongruent_rt = mean_rt(&incongruent_rts);
        let neutral_rt = mean_rt(&neutral_rts);

        FlankerMetrics {
            congruent_rt,
            incongruent_rt,
            neutral_rt,
            flanker_effect: incongruent_rt - congruent_rt,
            congruent_accuracy: if congruent_total > 0 {
                congruent_correct as f64 / congruent_total as f64
            } else {
                0.0
            },
            incongruent_accuracy: if incongruent_total > 0 {
                incongruent_correct as f64 / incongruent_total as f64
            } else {
                0.0
            },
            neutral_accuracy: if neutral_total > 0 {
                neutral_correct as f64 / neutral_total as f64
            } else {
                0.0
            },
            conflict_adaptation: 0.0, // Requires sequential analysis
        }
    }
}

/// Wisconsin Card Sorting Test rule dimensions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WcstDimension {
    Color,
    Shape,
    Number,
}

/// Wisconsin Card Sorting Test for cognitive flexibility
#[derive(Debug, Clone)]
pub struct WisconsinCardSort {
    /// Maximum categories to complete
    pub max_categories: usize,
    /// Consecutive correct responses needed to change rule
    pub criterion: usize,
    /// Maximum total trials
    pub max_trials: usize,
}

impl Default for WisconsinCardSort {
    fn default() -> Self {
        Self {
            max_categories: 6,
            criterion: 10,
            max_trials: 128,
        }
    }
}

/// A WCST card
#[derive(Debug, Clone)]
pub struct WcstCard {
    /// Color (0-3)
    pub color: u8,
    /// Shape (0-3)
    pub shape: u8,
    /// Number of shapes (1-4)
    pub number: u8,
}

/// WCST performance metrics
#[derive(Debug, Clone, Default)]
pub struct WcstMetrics {
    /// Number of categories completed
    pub categories_completed: usize,
    /// Total trials administered
    pub total_trials: usize,
    /// Total errors
    pub total_errors: usize,
    /// Perseverative responses (continuing with old rule)
    pub perseverative_responses: usize,
    /// Perseverative errors
    pub perseverative_errors: usize,
    /// Non-perseverative errors
    pub non_perseverative_errors: usize,
    /// Trials to complete first category
    pub trials_to_first_category: usize,
    /// Conceptual level responses (3+ consecutive correct)
    pub conceptual_level_responses: usize,
    /// Failure to maintain set (error after 5+ correct)
    pub failure_to_maintain_set: usize,
}

impl WisconsinCardSort {
    /// Create a new WCST
    pub fn new(max_categories: usize, criterion: usize) -> Self {
        Self {
            max_categories,
            criterion,
            max_trials: 128,
        }
    }

    /// Generate stimulus cards
    pub fn generate_cards(&self) -> Vec<WcstCard> {
        let mut cards = Vec::new();
        for color in 0..4 {
            for shape in 0..4 {
                for number in 1..=4 {
                    cards.push(WcstCard {
                        color,
                        shape,
                        number,
                    });
                }
            }
        }
        cards
    }

    /// Check if a sort matches the current rule
    pub fn check_match(&self, stimulus: &WcstCard, target: &WcstCard, rule: WcstDimension) -> bool {
        match rule {
            WcstDimension::Color => stimulus.color == target.color,
            WcstDimension::Shape => stimulus.shape == target.shape,
            WcstDimension::Number => stimulus.number == target.number,
        }
    }
}

/// Stroop task (re-exported from attention for executive function assessment)
pub use crate::attention::{StroopCondition, StroopMetrics, StroopTask};

/// Calculate d-prime from hit rate and false alarm rate
fn calculate_d_prime(hit_rate: f64, false_alarm_rate: f64) -> f64 {
    // Adjust extreme values to avoid infinite z-scores
    let hr = hit_rate.clamp(0.01, 0.99);
    let far = false_alarm_rate.clamp(0.01, 0.99);

    // Calculate z-scores using inverse normal approximation
    let z_hit = inverse_normal_cdf(hr);
    let z_fa = inverse_normal_cdf(far);

    z_hit - z_fa
}

/// Approximate inverse normal CDF (probit function)
fn inverse_normal_cdf(p: f64) -> f64 {
    // Rational approximation (Abramowitz and Stegun)
    let p = p.clamp(1e-10, 1.0 - 1e-10);

    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let p = if p < 0.5 { p } else { 1.0 - p };

    let t = (-2.0 * p.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    sign * result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gonogo_generation() {
        let task = GoNoGoTask::default();
        let trials = task.generate_trials(100);
        assert_eq!(trials.len(), 100);

        // Check that go probability is approximately correct
        let go_count = trials.iter().filter(|t| t.is_go).count();
        assert!(go_count > 50 && go_count < 95); // Should be around 75
    }

    #[test]
    fn test_flanker_generation() {
        let task = FlankerTask::new(20);
        let trials = task.generate_trials();
        assert_eq!(trials.len(), 60); // 20 per condition * 3 conditions
    }

    #[test]
    fn test_d_prime_calculation() {
        // Perfect performance
        let d = calculate_d_prime(0.99, 0.01);
        assert!(d > 3.0);

        // Chance performance
        let d = calculate_d_prime(0.5, 0.5);
        assert!(d.abs() < 0.1);
    }

    #[test]
    fn test_wcst_card_generation() {
        let wcst = WisconsinCardSort::default();
        let cards = wcst.generate_cards();
        assert_eq!(cards.len(), 64); // 4 colors * 4 shapes * 4 numbers
    }
}
