//! Reaction time tasks for assessing processing speed and response execution
//!
//! Includes simple reaction time (SRT) and choice reaction time (CRT) paradigms.

use rand::{Rng, RngExt};
use rand_distr::{Distribution, Uniform};

/// A simple reaction time task where participant responds to a single stimulus
#[derive(Debug, Clone)]
pub struct SimpleReactionTime {
    /// Range of intervals between stimuli in milliseconds (min, max)
    pub stimulus_intervals: (f64, f64),
    /// Maximum time allowed for a valid response in milliseconds
    pub response_window: f64,
}

impl Default for SimpleReactionTime {
    fn default() -> Self {
        Self {
            stimulus_intervals: (1000.0, 3000.0),
            response_window: 1000.0,
        }
    }
}

impl SimpleReactionTime {
    /// Create a new simple reaction time task
    pub fn new(stimulus_intervals: (f64, f64), response_window: f64) -> Self {
        Self {
            stimulus_intervals,
            response_window,
        }
    }

    /// Generate a sequence of trials with randomized inter-stimulus intervals
    pub fn generate_trial_sequence(&self, n_trials: usize) -> Vec<Trial> {
        let mut rng = rand::rng();
        let interval_dist = Uniform::new(self.stimulus_intervals.0, self.stimulus_intervals.1).expect("uniform bounds are ordered and finite");

        (0..n_trials)
            .map(|i| Trial {
                trial_number: i,
                stimulus_onset: 0.0, // Will be computed based on previous ISIs
                inter_stimulus_interval: interval_dist.sample(&mut rng),
                stimulus_type: StimulusType::Single,
                correct_response: ResponseType::Go,
            })
            .collect()
    }

    /// Score responses and calculate metrics
    pub fn score_responses(&self, trials: &[Trial], responses: &[Response]) -> ReactionTimeMetrics {
        let mut valid_rts = Vec::new();
        let mut anticipations = 0;
        let mut lapses = 0;

        for (trial, response) in trials.iter().zip(responses.iter()) {
            match response.response_time {
                Some(rt) => {
                    if rt < 100.0 {
                        anticipations += 1;
                    } else if rt > self.response_window {
                        lapses += 1;
                    } else {
                        valid_rts.push(rt);
                    }
                }
                None => {
                    lapses += 1;
                }
            }
        }

        calculate_rt_metrics(valid_rts, anticipations, lapses)
    }
}

/// A choice reaction time task where participant must discriminate between stimuli
#[derive(Debug, Clone)]
pub struct ChoiceReactionTime {
    /// Number of response choices
    pub n_choices: usize,
    /// Probability distribution of each stimulus type
    pub stimulus_probability: Vec<f64>,
}

impl Default for ChoiceReactionTime {
    fn default() -> Self {
        Self {
            n_choices: 2,
            stimulus_probability: vec![0.5, 0.5],
        }
    }
}

impl ChoiceReactionTime {
    /// Create a new choice reaction time task
    pub fn new(n_choices: usize, stimulus_probability: Vec<f64>) -> Self {
        assert_eq!(n_choices, stimulus_probability.len());
        assert!((stimulus_probability.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        Self {
            n_choices,
            stimulus_probability,
        }
    }

    /// Generate a sequence of trials with balanced stimulus types
    pub fn generate_trial_sequence(&self, n_trials: usize) -> Vec<Trial> {
        let mut rng = rand::rng();
        let mut trials = Vec::with_capacity(n_trials);

        for i in 0..n_trials {
            let rand_val: f64 = rng.random();
            let mut cumulative = 0.0;
            let mut stimulus_idx = 0;

            for (idx, &prob) in self.stimulus_probability.iter().enumerate() {
                cumulative += prob;
                if rand_val < cumulative {
                    stimulus_idx = idx;
                    break;
                }
            }

            trials.push(Trial {
                trial_number: i,
                stimulus_onset: 0.0,
                inter_stimulus_interval: rng.random_range(1000.0..3000.0),
                stimulus_type: StimulusType::Choice(stimulus_idx),
                correct_response: ResponseType::Choice(stimulus_idx),
            });
        }

        trials
    }

    /// Score responses and calculate metrics including accuracy
    pub fn score_responses(&self, trials: &[Trial], responses: &[Response]) -> ReactionTimeMetrics {
        let mut valid_rts = Vec::new();
        let mut anticipations = 0;
        let mut lapses = 0;
        let mut errors = 0;
        let mut correct = 0;

        for (trial, response) in trials.iter().zip(responses.iter()) {
            // Check accuracy
            let is_correct = match (&trial.correct_response, &response.response_type) {
                (ResponseType::Choice(correct), ResponseType::Choice(given)) => correct == given,
                _ => false,
            };

            if is_correct {
                correct += 1;
            } else {
                errors += 1;
            }

            // Check RT validity
            match response.response_time {
                Some(rt) => {
                    if rt < 100.0 {
                        anticipations += 1;
                    } else if rt > 2000.0 {
                        lapses += 1;
                    } else if is_correct {
                        valid_rts.push(rt);
                    }
                }
                None => {
                    lapses += 1;
                }
            }
        }

        let mut metrics = calculate_rt_metrics(valid_rts, anticipations, lapses);
        metrics.accuracy = Some((correct as f64) / (trials.len() as f64));
        metrics.errors = Some(errors);
        metrics
    }
}

/// Metrics computed from reaction time data
#[derive(Debug, Clone, Default)]
pub struct ReactionTimeMetrics {
    /// Mean reaction time in milliseconds
    pub mean_rt: f64,
    /// Median reaction time in milliseconds
    pub median_rt: f64,
    /// Standard deviation of reaction times
    pub std_rt: f64,
    /// Coefficient of variation (std/mean)
    pub coefficient_of_variation: f64,
    /// Number of anticipatory responses (RT < 100ms)
    pub anticipations: usize,
    /// Number of lapses (no response or RT > threshold)
    pub lapses: usize,
    /// Inverse efficiency score (mean_rt / accuracy)
    pub inverse_efficiency: Option<f64>,
    /// Accuracy (for choice RT tasks)
    pub accuracy: Option<f64>,
    /// Number of errors (for choice RT tasks)
    pub errors: Option<usize>,
}

/// A single trial in a reaction time task
#[derive(Debug, Clone)]
pub struct Trial {
    /// Trial index
    pub trial_number: usize,
    /// Time of stimulus onset (ms from task start)
    pub stimulus_onset: f64,
    /// Interval before this stimulus (ms)
    pub inter_stimulus_interval: f64,
    /// Type of stimulus presented
    pub stimulus_type: StimulusType,
    /// Correct response for this trial
    pub correct_response: ResponseType,
}

/// Type of stimulus presented
#[derive(Debug, Clone, PartialEq)]
pub enum StimulusType {
    /// Single stimulus (simple RT)
    Single,
    /// One of N choices (choice RT)
    Choice(usize),
}

/// Response recorded from participant
#[derive(Debug, Clone)]
pub struct Response {
    /// Trial number this response corresponds to
    pub trial_number: usize,
    /// Reaction time in milliseconds (None if no response)
    pub response_time: Option<f64>,
    /// Type of response given
    pub response_type: ResponseType,
}

/// Type of response
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseType {
    /// Simple go response
    Go,
    /// Choice response (which option selected)
    Choice(usize),
    /// No response
    NoGo,
}

/// Calculate reaction time statistics from a vector of valid RTs
pub fn calculate_rt_metrics(
    mut valid_rts: Vec<f64>,
    anticipations: usize,
    lapses: usize,
) -> ReactionTimeMetrics {
    if valid_rts.is_empty() {
        return ReactionTimeMetrics::default();
    }

    let n = valid_rts.len();
    let mean_rt = valid_rts.iter().sum::<f64>() / n as f64;

    valid_rts.sort_by(|a, b| a.total_cmp(b));
    let median_rt = if n.is_multiple_of(2) {
        (valid_rts[n / 2 - 1] + valid_rts[n / 2]) / 2.0
    } else {
        valid_rts[n / 2]
    };

    let variance = valid_rts
        .iter()
        .map(|&rt| (rt - mean_rt).powi(2))
        .sum::<f64>()
        / n as f64;
    let std_rt = variance.sqrt();

    let coefficient_of_variation = if mean_rt > 0.0 {
        std_rt / mean_rt
    } else {
        0.0
    };

    ReactionTimeMetrics {
        mean_rt,
        median_rt,
        std_rt,
        coefficient_of_variation,
        anticipations,
        lapses,
        inverse_efficiency: None,
        accuracy: None,
        errors: None,
    }
}

/// Calculate percentile of reaction time distribution
pub fn calculate_percentile(rts: &[f64], percentile: f64) -> f64 {
    assert!(!rts.is_empty() && (0.0..=100.0).contains(&percentile));
    let mut sorted = rts.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let rank = (percentile / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    if lower == upper {
        sorted[lower]
    } else {
        let fraction = rank - lower as f64;
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rt_generation() {
        let task = SimpleReactionTime::default();
        let trials = task.generate_trial_sequence(10);
        assert_eq!(trials.len(), 10);
    }

    #[test]
    fn test_choice_rt_generation() {
        let task = ChoiceReactionTime::new(2, vec![0.5, 0.5]);
        let trials = task.generate_trial_sequence(100);
        assert_eq!(trials.len(), 100);
    }

    #[test]
    fn test_rt_metrics() {
        let rts = vec![250.0, 300.0, 280.0, 320.0, 290.0];
        let metrics = calculate_rt_metrics(rts, 0, 0);
        assert!((metrics.mean_rt - 288.0).abs() < 1.0);
        assert_eq!(metrics.median_rt, 290.0);
    }

    #[test]
    fn test_percentile() {
        let rts = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        assert_eq!(calculate_percentile(&rts, 0.0), 100.0);
        assert_eq!(calculate_percentile(&rts, 50.0), 300.0);
        assert_eq!(calculate_percentile(&rts, 100.0), 500.0);
    }
}
