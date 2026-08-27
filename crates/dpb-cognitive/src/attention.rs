//! Attention assessment tasks
//!
//! Implements continuous performance tests, Stroop task, and trail making tests.

use rand::prelude::*;

/// Continuous Performance Test for sustained attention
#[derive(Debug, Clone)]
pub struct ContinuousPerformanceTest {
    /// Variant of CPT being used
    pub variant: CptVariant,
    /// Duration of the test in milliseconds
    pub duration: f64,
    /// Inter-stimulus interval in milliseconds
    pub isi: f64,
    /// Target stimulus
    pub target: char,
}

/// Variants of Continuous Performance Test
#[derive(Debug, Clone, PartialEq)]
pub enum CptVariant {
    /// Conners CPT-II (respond to all except X)
    CptII,
    /// CPT-X (respond only to X)
    CptX,
    /// Test of Variables of Attention
    Tova,
}

impl Default for ContinuousPerformanceTest {
    fn default() -> Self {
        Self {
            variant: CptVariant::CptX,
            duration: 300000.0, // 5 minutes
            isi: 2000.0,
            target: 'X',
        }
    }
}

impl ContinuousPerformanceTest {
    /// Create a new CPT task
    pub fn new(variant: CptVariant, duration: f64, isi: f64, target: char) -> Self {
        Self {
            variant,
            duration,
            isi,
            target,
        }
    }

    /// Generate trial sequence for CPT
    pub fn generate_trials(&self) -> Vec<CptTrial> {
        let n_trials = (self.duration / self.isi) as usize;
        let mut rng = rand::rng();
        let letters: Vec<char> = ('A'..='Z').collect();
        let mut trials = Vec::with_capacity(n_trials);

        for i in 0..n_trials {
            let stimulus = if rng.random_bool(0.3) {
                self.target
            } else {
                let mut letter = *letters.choose(&mut rng).unwrap();
                while letter == self.target {
                    letter = *letters.choose(&mut rng).unwrap();
                }
                letter
            };

            let is_target = match self.variant {
                CptVariant::CptX => stimulus == self.target,
                CptVariant::CptII => stimulus != self.target,
                CptVariant::Tova => stimulus == self.target,
            };

            trials.push(CptTrial {
                trial_number: i,
                stimulus,
                is_target,
                onset_time: i as f64 * self.isi,
            });
        }

        trials
    }

    /// Score CPT performance
    pub fn score_responses(&self, trials: &[CptTrial], responses: &[CptResponse]) -> CptMetrics {
        let mut omissions = 0;
        let mut commissions = 0;
        let mut hit_rts = Vec::new();
        let mut hits = 0;
        let mut false_alarms = 0;
        // Correct rejections are not counted: they are n_non_targets minus
        // false_alarms, and d' below is computed from n_non_targets directly.

        for (trial, response) in trials.iter().zip(responses.iter()) {
            match (trial.is_target, response.responded) {
                (true, true) => {
                    hits += 1;
                    if let Some(rt) = response.response_time {
                        hit_rts.push(rt);
                    }
                }
                (true, false) => omissions += 1,
                (false, true) => {
                    commissions += 1;
                    false_alarms += 1;
                }
                (false, false) => {}
            }
        }

        let hit_rt = if hit_rts.is_empty() {
            0.0
        } else {
            hit_rts.iter().sum::<f64>() / hit_rts.len() as f64
        };

        let rt_variability = if hit_rts.len() > 1 {
            let mean = hit_rt;
            let variance = hit_rts
                .iter()
                .map(|&rt| (rt - mean).powi(2))
                .sum::<f64>()
                / hit_rts.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let n_targets = trials.iter().filter(|t| t.is_target).count();
        let n_non_targets = trials.len() - n_targets;

        let d_prime = if n_targets > 0 && n_non_targets > 0 {
            let hit_rate = (hits as f64 + 0.5) / (n_targets as f64 + 1.0);
            let fa_rate = (false_alarms as f64 + 0.5) / (n_non_targets as f64 + 1.0);
            crate::working_memory::z_score(hit_rate) - crate::working_memory::z_score(fa_rate)
        } else {
            0.0
        };

        CptMetrics {
            omissions,
            commissions,
            hit_rt,
            rt_variability,
            d_prime,
        }
    }
}

/// CPT performance metrics
#[derive(Debug, Clone, Default)]
pub struct CptMetrics {
    /// Number of missed target responses
    pub omissions: usize,
    /// Number of incorrect responses to non-targets
    pub commissions: usize,
    /// Mean reaction time for hits
    pub hit_rt: f64,
    /// Standard deviation of hit reaction times
    pub rt_variability: f64,
    /// Sensitivity measure
    pub d_prime: f64,
}

/// A single CPT trial
#[derive(Debug, Clone)]
pub struct CptTrial {
    /// Trial number
    pub trial_number: usize,
    /// Stimulus letter
    pub stimulus: char,
    /// Whether this is a target trial
    pub is_target: bool,
    /// Time of stimulus onset
    pub onset_time: f64,
}

/// Response to a CPT trial
#[derive(Debug, Clone)]
pub struct CptResponse {
    /// Trial number
    pub trial_number: usize,
    /// Whether participant responded
    pub responded: bool,
    /// Response time if responded
    pub response_time: Option<f64>,
}

/// Stroop color-word interference task
#[derive(Debug, Clone)]
pub struct StroopTask {
    /// Number of trials per condition
    pub trials_per_condition: usize,
}

impl Default for StroopTask {
    fn default() -> Self {
        Self {
            trials_per_condition: 20,
        }
    }
}

impl StroopTask {
    /// Create a new Stroop task
    pub fn new(trials_per_condition: usize) -> Self {
        Self {
            trials_per_condition,
        }
    }

    /// Generate balanced Stroop trial sequence
    pub fn generate_trials(&self) -> Vec<StroopTrial> {
        let mut rng = rand::rng();
        let colors = ["red", "blue", "green", "yellow"];
        let mut trials = Vec::new();

        // Generate trials for each condition
        for condition in &[
            StroopCondition::Congruent,
            StroopCondition::Incongruent,
            StroopCondition::Neutral,
        ] {
            for _i in 0..self.trials_per_condition {
                let color_idx = rng.random_range(0..colors.len());
                let color = colors[color_idx];

                let word = match condition {
                    StroopCondition::Congruent => color,
                    StroopCondition::Incongruent => {
                        let mut word_idx = rng.random_range(0..colors.len());
                        while word_idx == color_idx {
                            word_idx = rng.random_range(0..colors.len());
                        }
                        colors[word_idx]
                    }
                    StroopCondition::Neutral => "XXXX",
                };

                trials.push(StroopTrial {
                    trial_number: trials.len(),
                    word: word.to_string(),
                    color: color.to_string(),
                    condition: condition.clone(),
                });
            }
        }

        // Shuffle trials
        trials.shuffle(&mut rng);

        // Reassign trial numbers after shuffle
        for (i, trial) in trials.iter_mut().enumerate() {
            trial.trial_number = i;
        }

        trials
    }

    /// Score Stroop performance
    pub fn score_responses(
        &self,
        trials: &[StroopTrial],
        responses: &[StroopResponse],
    ) -> StroopMetrics {
        let mut congruent_rts = Vec::new();
        let mut incongruent_rts = Vec::new();
        let mut neutral_rts = Vec::new();
        let mut errors_by_condition = [0, 0, 0]; // Congruent, Incongruent, Neutral

        for (trial, response) in trials.iter().zip(responses.iter()) {
            let is_correct = response.response.to_lowercase() == trial.color.to_lowercase();

            if let Some(rt) = response.response_time
                && is_correct {
                    match trial.condition {
                        StroopCondition::Congruent => congruent_rts.push(rt),
                        StroopCondition::Incongruent => incongruent_rts.push(rt),
                        StroopCondition::Neutral => neutral_rts.push(rt),
                    }
                }

            if !is_correct {
                let idx = match trial.condition {
                    StroopCondition::Congruent => 0,
                    StroopCondition::Incongruent => 1,
                    StroopCondition::Neutral => 2,
                };
                errors_by_condition[idx] += 1;
            }
        }

        let mean_rt_congruent = mean_or_zero(&congruent_rts);
        let mean_rt_incongruent = mean_or_zero(&incongruent_rts);
        let mean_rt_neutral = mean_or_zero(&neutral_rts);
        let stroop_effect = mean_rt_incongruent - mean_rt_congruent;

        StroopMetrics {
            mean_rt_congruent,
            mean_rt_incongruent,
            mean_rt_neutral,
            stroop_effect,
            errors_congruent: errors_by_condition[0],
            errors_incongruent: errors_by_condition[1],
            errors_neutral: errors_by_condition[2],
        }
    }
}

/// Stroop task conditions
#[derive(Debug, Clone, PartialEq)]
pub enum StroopCondition {
    /// Word and color match
    Congruent,
    /// Word and color mismatch
    Incongruent,
    /// Neutral stimulus (e.g., XXXX)
    Neutral,
}

/// A single Stroop trial
#[derive(Debug, Clone)]
pub struct StroopTrial {
    /// Trial number
    pub trial_number: usize,
    /// Word text
    pub word: String,
    /// Color of the text
    pub color: String,
    /// Trial condition
    pub condition: StroopCondition,
}

/// Response to a Stroop trial
#[derive(Debug, Clone)]
pub struct StroopResponse {
    /// Trial number
    pub trial_number: usize,
    /// Color named by participant
    pub response: String,
    /// Response time
    pub response_time: Option<f64>,
}

/// Stroop performance metrics
#[derive(Debug, Clone, Default)]
pub struct StroopMetrics {
    /// Mean RT for congruent trials
    pub mean_rt_congruent: f64,
    /// Mean RT for incongruent trials
    pub mean_rt_incongruent: f64,
    /// Mean RT for neutral trials
    pub mean_rt_neutral: f64,
    /// Stroop interference effect (incongruent - congruent)
    pub stroop_effect: f64,
    /// Errors on congruent trials
    pub errors_congruent: usize,
    /// Errors on incongruent trials
    pub errors_incongruent: usize,
    /// Errors on neutral trials
    pub errors_neutral: usize,
}

/// Trail Making Test for visual attention and task switching
#[derive(Debug, Clone)]
pub struct TrailMakingTest {
    /// Part A (numbers only) or Part B (alternating numbers and letters)
    pub part: TrailPart,
    /// Number of targets to connect
    pub n_targets: usize,
}

/// Trail Making Test parts
#[derive(Debug, Clone, PartialEq)]
pub enum TrailPart {
    /// Part A: Connect numbers in sequence (1-2-3...)
    A,
    /// Part B: Alternate numbers and letters (1-A-2-B-3-C...)
    B,
}

impl Default for TrailMakingTest {
    fn default() -> Self {
        Self {
            part: TrailPart::A,
            n_targets: 25,
        }
    }
}

impl TrailMakingTest {
    /// Create a new Trail Making Test
    pub fn new(part: TrailPart, n_targets: usize) -> Self {
        Self { part, n_targets }
    }

    /// Generate target sequence
    pub fn generate_sequence(&self) -> Vec<String> {
        match self.part {
            TrailPart::A => (1..=self.n_targets).map(|n| n.to_string()).collect(),
            TrailPart::B => {
                let mut sequence = Vec::new();
                for i in 0..self.n_targets {
                    if i % 2 == 0 {
                        sequence.push((i / 2 + 1).to_string());
                    } else {
                        let letter = (b'A' + (i / 2) as u8) as char;
                        sequence.push(letter.to_string());
                    }
                }
                sequence
            }
        }
    }

    /// Score trail making performance
    pub fn score_performance(&self, completion_time: f64, errors: usize) -> TrailMetrics {
        TrailMetrics {
            completion_time,
            errors,
            part: self.part.clone(),
        }
    }
}

/// Trail Making Test metrics
#[derive(Debug, Clone)]
pub struct TrailMetrics {
    /// Time to complete in milliseconds
    pub completion_time: f64,
    /// Number of errors made
    pub errors: usize,
    /// Which part was completed
    pub part: TrailPart,
}

impl Default for TrailMetrics {
    fn default() -> Self {
        Self {
            completion_time: 0.0,
            errors: 0,
            part: TrailPart::A,
        }
    }
}

/// Calculate mean of a vector or return 0.0 if empty
fn mean_or_zero(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpt_generation() {
        let cpt = ContinuousPerformanceTest::default();
        let trials = cpt.generate_trials();
        assert!(!trials.is_empty());
    }

    #[test]
    fn test_stroop_generation() {
        let stroop = StroopTask::new(10);
        let trials = stroop.generate_trials();
        assert_eq!(trials.len(), 30); // 10 trials x 3 conditions
    }

    #[test]
    fn test_trail_making_sequences() {
        let tmt_a = TrailMakingTest::new(TrailPart::A, 10);
        let seq_a = tmt_a.generate_sequence();
        assert_eq!(seq_a, vec!["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]);

        let tmt_b = TrailMakingTest::new(TrailPart::B, 8);
        let seq_b = tmt_b.generate_sequence();
        assert_eq!(seq_b, vec!["1", "A", "2", "B", "3", "C", "4", "D"]);
    }

    #[test]
    fn test_stroop_effect() {
        let task = StroopTask::new(5);
        let trials = vec![
            StroopTrial {
                trial_number: 0,
                word: "red".to_string(),
                color: "red".to_string(),
                condition: StroopCondition::Congruent,
            },
            StroopTrial {
                trial_number: 1,
                word: "blue".to_string(),
                color: "red".to_string(),
                condition: StroopCondition::Incongruent,
            },
        ];
        let responses = vec![
            StroopResponse {
                trial_number: 0,
                response: "red".to_string(),
                response_time: Some(500.0),
            },
            StroopResponse {
                trial_number: 1,
                response: "red".to_string(),
                response_time: Some(700.0),
            },
        ];

        let metrics = task.score_responses(&trials, &responses);
        assert_eq!(metrics.stroop_effect, 200.0);
    }
}
