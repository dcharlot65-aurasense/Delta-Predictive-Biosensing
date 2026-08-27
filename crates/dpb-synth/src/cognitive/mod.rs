//! Cognitive Task Response Generators
//!
//! Generates behavioral data from cognitive tasks:
//! - Reaction time distributions
//! - Accuracy patterns
//! - Speed-accuracy tradeoffs
//! - Sequential effects
//! - Cognitive fatigue
//! - Pathological patterns (ADHD, dementia, TBI)

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// Configuration for cognitive task generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveConfig {
    /// Mean reaction time (ms)
    pub mean_rt: f64,
    /// RT variability (coefficient of variation)
    pub rt_cv: f64,
    /// Base accuracy (0-1)
    pub accuracy: f64,
    /// Speed-accuracy tradeoff strength
    pub sat_strength: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            mean_rt: 350.0,
            rt_cv: 0.2,
            accuracy: 0.95,
            sat_strength: 0.5,
            seed: None,
        }
    }
}

/// Output from cognitive task generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveOutput {
    /// Trial number
    pub trial: Vec<usize>,
    /// Reaction time (ms)
    pub reaction_time: Vec<f64>,
    /// Response accuracy (correct = true)
    pub correct: Vec<bool>,
    /// Stimulus type per trial
    pub stimulus: Vec<StimulusType>,
    /// Response type per trial
    pub response: Vec<ResponseType>,
    /// Ground truth
    pub ground_truth: CognitiveGroundTruth,
    /// Configuration
    pub config: CognitiveConfig,
}

/// Stimulus types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StimulusType {
    /// Target stimulus requiring response
    Target,
    /// Non-target (go/no-go tasks)
    NonTarget,
    /// Congruent (flanker, Stroop)
    Congruent,
    /// Incongruent
    Incongruent,
    /// Neutral
    Neutral,
    /// Cued
    Cued,
    /// Uncued
    Uncued,
}

/// Response types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ResponseType {
    /// Correct response
    Hit,
    /// Correct rejection
    CorrectRejection,
    /// Incorrect response (error)
    Error,
    /// Missed target
    Miss,
    /// False alarm
    FalseAlarm,
    /// No response
    NoResponse,
}

/// Ground truth for cognitive performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveGroundTruth {
    /// Task type
    pub task: CognitiveTask,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Ex-Gaussian RT distribution parameters
    pub rt_distribution: RtDistribution,
    /// Sequential effects
    pub sequential: SequentialEffects,
    /// Applied pathology
    pub pathology: Option<CognitivePathology>,
}

/// Cognitive task types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CognitiveTask {
    /// Simple reaction time
    SimpleRt,
    /// Choice reaction time
    ChoiceRt,
    /// Go/No-Go
    GoNoGo,
    /// Flanker task
    Flanker,
    /// Stroop task
    Stroop,
    /// N-back working memory
    NBack { n: usize },
    /// Continuous performance test
    Cpt,
    /// Trail Making Test
    TrailMaking { part: char },
    /// Posner cueing task
    PosnerCueing,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Mean RT (ms)
    pub mean_rt: f64,
    /// Median RT (ms)
    pub median_rt: f64,
    /// RT standard deviation (ms)
    pub sd_rt: f64,
    /// Coefficient of variation
    pub cv_rt: f64,
    /// Overall accuracy
    pub accuracy: f64,
    /// Hit rate (for signal detection)
    pub hit_rate: f64,
    /// False alarm rate
    pub false_alarm_rate: f64,
    /// d-prime (sensitivity)
    pub d_prime: f64,
    /// Response bias (criterion)
    pub criterion: f64,
    /// Congruency effect (if applicable)
    pub congruency_effect: Option<f64>,
    /// Post-error slowing (ms)
    pub post_error_slowing: f64,
}

/// RT distribution parameters (ex-Gaussian)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtDistribution {
    /// Mu (Gaussian mean)
    pub mu: f64,
    /// Sigma (Gaussian SD)
    pub sigma: f64,
    /// Tau (exponential component)
    pub tau: f64,
}

/// Sequential effects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialEffects {
    /// Post-error slowing present
    pub post_error_slowing: bool,
    /// Post-conflict adaptation
    pub gratton_effect: bool,
    /// Fatigue slope (ms per 100 trials)
    pub fatigue_slope: f64,
}

/// Cognitive pathologies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CognitivePathology {
    /// ADHD pattern (high variability, lapses)
    Adhd {
        variability_increase: f64,
        lapse_rate: f64,
    },
    /// Mild cognitive impairment
    Mci {
        slowdown: f64,
        accuracy_reduction: f64,
    },
    /// Traumatic brain injury
    Tbi { severity: f64 },
    /// Depression (psychomotor slowing)
    Depression { slowdown: f64 },
    /// Parkinson's (bradykinesia)
    Parkinsons { slowdown: f64, variability: f64 },
    /// Sleep deprivation
    SleepDeprivation { hours_awake: f64 },
    /// Aging effects
    Aging { age: f64 },
}

/// Cognitive task generator
pub struct CognitiveGenerator {
    config: CognitiveConfig,
    rng: StdRng,
}

impl CognitiveGenerator {
    /// Create new cognitive generator
    pub fn new(config: CognitiveConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };
        Self { config, rng }
    }

    /// Generate simple reaction time task
    pub fn generate_simple_rt(&mut self, n_trials: usize) -> CognitiveOutput {
        let mut trial = Vec::with_capacity(n_trials);
        let mut reaction_time = Vec::with_capacity(n_trials);
        let mut correct = Vec::with_capacity(n_trials);
        let mut stimulus = Vec::with_capacity(n_trials);
        let mut response = Vec::with_capacity(n_trials);

        // Ex-Gaussian parameters
        let mu = self.config.mean_rt * 0.7;
        let sigma = self.config.mean_rt * self.config.rt_cv * 0.5;
        let tau = self.config.mean_rt * 0.3;

        let gaussian = Normal::new(mu, sigma).unwrap();

        for i in 0..n_trials {
            trial.push(i);
            stimulus.push(StimulusType::Target);

            // Ex-Gaussian RT
            let gaussian_part: f64 = self.rng.sample(gaussian);
            let exponential_part = -tau * self.rng.random::<f64>().ln();
            let rt = (gaussian_part + exponential_part).max(100.0);

            // Accuracy (simple RT typically has high accuracy)
            let is_correct = self.rng.random::<f64>() < self.config.accuracy;

            if is_correct {
                reaction_time.push(rt);
                correct.push(true);
                response.push(ResponseType::Hit);
            } else {
                // Miss or anticipation
                if self.rng.random::<bool>() {
                    reaction_time.push(rt * 0.5); // Anticipation
                    correct.push(false);
                    response.push(ResponseType::FalseAlarm);
                } else {
                    reaction_time.push(3000.0); // Timeout
                    correct.push(false);
                    response.push(ResponseType::Miss);
                }
            }
        }

        let metrics = self.calculate_metrics(&reaction_time, &correct, &stimulus, &response);
        let rt_distribution = RtDistribution { mu, sigma, tau };
        let sequential = self.calculate_sequential(&reaction_time, &correct);

        let ground_truth = CognitiveGroundTruth {
            task: CognitiveTask::SimpleRt,
            metrics,
            rt_distribution,
            sequential,
            pathology: None,
        };

        CognitiveOutput {
            trial,
            reaction_time,
            correct,
            stimulus,
            response,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate flanker task
    pub fn generate_flanker(
        &mut self,
        n_trials: usize,
        congruent_proportion: f64,
    ) -> CognitiveOutput {
        let mut trial = Vec::with_capacity(n_trials);
        let mut reaction_time = Vec::with_capacity(n_trials);
        let mut correct = Vec::with_capacity(n_trials);
        let mut stimulus = Vec::with_capacity(n_trials);
        let mut response = Vec::with_capacity(n_trials);

        let congruent_rt = self.config.mean_rt;
        let incongruent_rt = self.config.mean_rt * 1.15; // 15% congruency effect

        let congruent_acc = self.config.accuracy;
        let incongruent_acc = self.config.accuracy * 0.92; // Lower accuracy for incongruent

        for i in 0..n_trials {
            trial.push(i);

            let is_congruent = self.rng.random::<f64>() < congruent_proportion;
            let (stim_type, base_rt, base_acc) = if is_congruent {
                (StimulusType::Congruent, congruent_rt, congruent_acc)
            } else {
                (StimulusType::Incongruent, incongruent_rt, incongruent_acc)
            };

            stimulus.push(stim_type);

            // Generate RT
            let sigma = base_rt * self.config.rt_cv;
            let rt_dist = Normal::new(base_rt, sigma).unwrap();
            let rt = self.rng.sample(rt_dist).max(100.0);
            reaction_time.push(rt);

            // Accuracy
            let is_correct = self.rng.random::<f64>() < base_acc;
            correct.push(is_correct);

            response.push(if is_correct {
                ResponseType::Hit
            } else {
                ResponseType::Error
            });
        }

        let metrics = self.calculate_metrics(&reaction_time, &correct, &stimulus, &response);
        let rt_distribution = RtDistribution {
            mu: self.config.mean_rt * 0.7,
            sigma: self.config.mean_rt * self.config.rt_cv * 0.5,
            tau: self.config.mean_rt * 0.3,
        };
        let sequential = self.calculate_sequential(&reaction_time, &correct);

        let ground_truth = CognitiveGroundTruth {
            task: CognitiveTask::Flanker,
            metrics,
            rt_distribution,
            sequential,
            pathology: None,
        };

        CognitiveOutput {
            trial,
            reaction_time,
            correct,
            stimulus,
            response,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate Go/No-Go task
    pub fn generate_go_nogo(&mut self, n_trials: usize, go_proportion: f64) -> CognitiveOutput {
        let mut trial = Vec::with_capacity(n_trials);
        let mut reaction_time = Vec::with_capacity(n_trials);
        let mut correct = Vec::with_capacity(n_trials);
        let mut stimulus = Vec::with_capacity(n_trials);
        let mut response = Vec::with_capacity(n_trials);

        let rt_dist =
            Normal::new(self.config.mean_rt, self.config.mean_rt * self.config.rt_cv).unwrap();

        for i in 0..n_trials {
            trial.push(i);

            let is_go = self.rng.random::<f64>() < go_proportion;
            stimulus.push(if is_go {
                StimulusType::Target
            } else {
                StimulusType::NonTarget
            });

            if is_go {
                // Go trial
                let responded = self.rng.random::<f64>() < self.config.accuracy;
                if responded {
                    let rt = self.rng.sample(rt_dist).max(100.0);
                    reaction_time.push(rt);
                    correct.push(true);
                    response.push(ResponseType::Hit);
                } else {
                    reaction_time.push(3000.0); // Timeout
                    correct.push(false);
                    response.push(ResponseType::Miss);
                }
            } else {
                // No-Go trial
                let false_alarm = self.rng.random::<f64>() > self.config.accuracy * 0.9; // Higher FA on no-go
                if false_alarm {
                    let rt = self.rng.sample(rt_dist).max(100.0) * 0.9; // Faster for impulsive responses
                    reaction_time.push(rt);
                    correct.push(false);
                    response.push(ResponseType::FalseAlarm);
                } else {
                    reaction_time.push(0.0); // No RT for correct rejection
                    correct.push(true);
                    response.push(ResponseType::CorrectRejection);
                }
            }
        }

        let metrics = self.calculate_metrics(&reaction_time, &correct, &stimulus, &response);
        let rt_distribution = RtDistribution {
            mu: self.config.mean_rt * 0.7,
            sigma: self.config.mean_rt * self.config.rt_cv * 0.5,
            tau: self.config.mean_rt * 0.3,
        };
        let sequential = self.calculate_sequential(&reaction_time, &correct);

        let ground_truth = CognitiveGroundTruth {
            task: CognitiveTask::GoNoGo,
            metrics,
            rt_distribution,
            sequential,
            pathology: None,
        };

        CognitiveOutput {
            trial,
            reaction_time,
            correct,
            stimulus,
            response,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate N-back working memory task
    pub fn generate_nback(
        &mut self,
        n_trials: usize,
        n: usize,
        target_proportion: f64,
    ) -> CognitiveOutput {
        let mut trial = Vec::with_capacity(n_trials);
        let mut reaction_time = Vec::with_capacity(n_trials);
        let mut correct = Vec::with_capacity(n_trials);
        let mut stimulus = Vec::with_capacity(n_trials);
        let mut response = Vec::with_capacity(n_trials);

        // N-back is harder than simple RT
        let nback_slowdown = 1.0 + n as f64 * 0.1;
        let nback_accuracy = self.config.accuracy - n as f64 * 0.05;

        let base_rt = self.config.mean_rt * nback_slowdown;
        let rt_dist = Normal::new(base_rt, base_rt * self.config.rt_cv).unwrap();

        for i in 0..n_trials {
            trial.push(i);

            let is_target = self.rng.random::<f64>() < target_proportion;
            stimulus.push(if is_target {
                StimulusType::Target
            } else {
                StimulusType::NonTarget
            });

            if is_target {
                let hit = self.rng.random::<f64>() < nback_accuracy;
                if hit {
                    reaction_time.push(self.rng.sample(rt_dist).max(100.0));
                    correct.push(true);
                    response.push(ResponseType::Hit);
                } else {
                    reaction_time.push(3000.0);
                    correct.push(false);
                    response.push(ResponseType::Miss);
                }
            } else {
                let fa = self.rng.random::<f64>() > nback_accuracy;
                if fa {
                    reaction_time.push(self.rng.sample(rt_dist).max(100.0));
                    correct.push(false);
                    response.push(ResponseType::FalseAlarm);
                } else {
                    reaction_time.push(0.0);
                    correct.push(true);
                    response.push(ResponseType::CorrectRejection);
                }
            }
        }

        let metrics = self.calculate_metrics(&reaction_time, &correct, &stimulus, &response);
        let rt_distribution = RtDistribution {
            mu: base_rt * 0.7,
            sigma: base_rt * self.config.rt_cv * 0.5,
            tau: base_rt * 0.3,
        };
        let sequential = self.calculate_sequential(&reaction_time, &correct);

        let ground_truth = CognitiveGroundTruth {
            task: CognitiveTask::NBack { n },
            metrics,
            rt_distribution,
            sequential,
            pathology: None,
        };

        CognitiveOutput {
            trial,
            reaction_time,
            correct,
            stimulus,
            response,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate pathological cognitive performance
    pub fn generate_pathological(
        &mut self,
        task: CognitiveTask,
        pathology: CognitivePathology,
        n_trials: usize,
    ) -> CognitiveOutput {
        // Modify config based on pathology
        let original_config = self.config.clone();

        match pathology {
            CognitivePathology::Adhd {
                variability_increase,
                lapse_rate,
            } => {
                self.config.rt_cv *= 1.0 + variability_increase;
                self.config.accuracy *= 1.0 - lapse_rate * 0.5;
            }
            CognitivePathology::Mci {
                slowdown,
                accuracy_reduction,
            } => {
                self.config.mean_rt *= 1.0 + slowdown;
                self.config.accuracy *= 1.0 - accuracy_reduction;
            }
            CognitivePathology::Tbi { severity } => {
                self.config.mean_rt *= 1.0 + severity * 0.5;
                self.config.rt_cv *= 1.0 + severity * 0.3;
                self.config.accuracy *= 1.0 - severity * 0.2;
            }
            CognitivePathology::Depression { slowdown } => {
                self.config.mean_rt *= 1.0 + slowdown;
            }
            CognitivePathology::Parkinsons {
                slowdown,
                variability,
            } => {
                self.config.mean_rt *= 1.0 + slowdown;
                self.config.rt_cv *= 1.0 + variability;
            }
            CognitivePathology::SleepDeprivation { hours_awake } => {
                let effect = (hours_awake - 16.0).max(0.0) / 24.0;
                self.config.mean_rt *= 1.0 + effect * 0.3;
                self.config.rt_cv *= 1.0 + effect * 0.5;
                self.config.accuracy *= 1.0 - effect * 0.2;
            }
            CognitivePathology::Aging { age } => {
                let effect = (age - 25.0).max(0.0) / 50.0;
                self.config.mean_rt *= 1.0 + effect * 0.4;
                self.config.accuracy *= 1.0 - effect * 0.1;
            }
        }

        let mut output = match task {
            CognitiveTask::SimpleRt => self.generate_simple_rt(n_trials),
            CognitiveTask::Flanker => self.generate_flanker(n_trials, 0.5),
            CognitiveTask::GoNoGo => self.generate_go_nogo(n_trials, 0.7),
            CognitiveTask::NBack { n } => self.generate_nback(n_trials, n, 0.3),
            _ => self.generate_simple_rt(n_trials),
        };

        output.ground_truth.pathology = Some(pathology);
        self.config = original_config;

        output
    }

    // Helper methods

    fn calculate_metrics(
        &self,
        reaction_time: &[f64],
        correct: &[bool],
        stimulus: &[StimulusType],
        response: &[ResponseType],
    ) -> PerformanceMetrics {
        // Filter valid RTs (exclude timeouts)
        let valid_rts: Vec<f64> = reaction_time
            .iter()
            .zip(correct.iter())
            .filter(|(rt, c)| **c && **rt < 2500.0 && **rt > 0.0)
            .map(|(rt, _)| *rt)
            .collect();

        let mean_rt = if !valid_rts.is_empty() {
            valid_rts.iter().sum::<f64>() / valid_rts.len() as f64
        } else {
            0.0
        };

        let mut sorted_rts = valid_rts.clone();
        sorted_rts.sort_by(|a, b| a.total_cmp(b));
        let median_rt = if !sorted_rts.is_empty() {
            sorted_rts[sorted_rts.len() / 2]
        } else {
            0.0
        };

        let sd_rt = if !valid_rts.is_empty() {
            let variance = valid_rts
                .iter()
                .map(|rt| (rt - mean_rt).powi(2))
                .sum::<f64>()
                / valid_rts.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let cv_rt = if mean_rt > 0.0 { sd_rt / mean_rt } else { 0.0 };

        let accuracy = correct.iter().filter(|&&c| c).count() as f64 / correct.len() as f64;

        // Signal detection
        let hits = response.iter().filter(|&&r| r == ResponseType::Hit).count();
        let misses = response
            .iter()
            .filter(|&&r| r == ResponseType::Miss)
            .count();
        let fas = response
            .iter()
            .filter(|&&r| r == ResponseType::FalseAlarm)
            .count();
        let crs = response
            .iter()
            .filter(|&&r| r == ResponseType::CorrectRejection)
            .count();

        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.5
        };

        let false_alarm_rate = if fas + crs > 0 {
            fas as f64 / (fas + crs) as f64
        } else {
            0.5
        };

        // d-prime (with correction for extreme values)
        let hr_adj = hit_rate.clamp(0.01, 0.99);
        let far_adj = false_alarm_rate.clamp(0.01, 0.99);
        let d_prime = Self::z_score(hr_adj) - Self::z_score(far_adj);
        let criterion = -0.5 * (Self::z_score(hr_adj) + Self::z_score(far_adj));

        // Congruency effect
        let congruent_rts: Vec<f64> = reaction_time
            .iter()
            .zip(stimulus.iter())
            .zip(correct.iter())
            .filter(|((_, s), c)| **s == StimulusType::Congruent && **c)
            .map(|((rt, _), _)| *rt)
            .filter(|rt| *rt > 0.0 && *rt < 2500.0)
            .collect();

        let incongruent_rts: Vec<f64> = reaction_time
            .iter()
            .zip(stimulus.iter())
            .zip(correct.iter())
            .filter(|((_, s), c)| **s == StimulusType::Incongruent && **c)
            .map(|((rt, _), _)| *rt)
            .filter(|rt| *rt > 0.0 && *rt < 2500.0)
            .collect();

        let congruency_effect = if !congruent_rts.is_empty() && !incongruent_rts.is_empty() {
            let cong_mean = congruent_rts.iter().sum::<f64>() / congruent_rts.len() as f64;
            let incong_mean = incongruent_rts.iter().sum::<f64>() / incongruent_rts.len() as f64;
            Some(incong_mean - cong_mean)
        } else {
            None
        };

        // Post-error slowing
        let mut post_error_slowing = 0.0;
        let mut post_error_count = 0;
        for i in 1..correct.len() {
            if !correct[i - 1] && correct[i] && reaction_time[i] > 0.0 && reaction_time[i] < 2500.0
            {
                post_error_slowing += reaction_time[i] - mean_rt;
                post_error_count += 1;
            }
        }
        if post_error_count > 0 {
            post_error_slowing /= post_error_count as f64;
        }

        PerformanceMetrics {
            mean_rt,
            median_rt,
            sd_rt,
            cv_rt,
            accuracy,
            hit_rate,
            false_alarm_rate,
            d_prime,
            criterion,
            congruency_effect,
            post_error_slowing,
        }
    }

    fn calculate_sequential(&self, reaction_time: &[f64], correct: &[bool]) -> SequentialEffects {
        // Check for post-error slowing
        let mut pes_sum = 0.0;
        let mut pes_count = 0;
        let valid_rts: Vec<f64> = reaction_time
            .iter()
            .filter(|&&rt| rt > 0.0 && rt < 2500.0)
            .cloned()
            .collect();
        let mean_rt = if !valid_rts.is_empty() {
            valid_rts.iter().sum::<f64>() / valid_rts.len() as f64
        } else {
            0.0
        };

        for i in 1..correct.len() {
            if !correct[i - 1] && correct[i] && reaction_time[i] > 0.0 {
                pes_sum += reaction_time[i] - mean_rt;
                pes_count += 1;
            }
        }

        let post_error_slowing = pes_count > 0 && (pes_sum / pes_count as f64) > 20.0;

        // Simplified fatigue detection
        let n = reaction_time.len();
        let early_mean = if n > 20 {
            reaction_time[0..20]
                .iter()
                .filter(|&&rt| rt > 0.0 && rt < 2500.0)
                .sum::<f64>()
                / 20.0
        } else {
            mean_rt
        };

        let late_mean = if n > 20 {
            reaction_time[n - 20..]
                .iter()
                .filter(|&&rt| rt > 0.0 && rt < 2500.0)
                .sum::<f64>()
                / 20.0
        } else {
            mean_rt
        };

        let fatigue_slope = (late_mean - early_mean) / (n as f64 / 100.0);

        SequentialEffects {
            post_error_slowing,
            gratton_effect: false, // Would need stimulus sequence analysis
            fatigue_slope,
        }
    }

    fn z_score(p: f64) -> f64 {
        // Approximation of inverse normal CDF
        let p = p.clamp(0.0001, 0.9999);
        if p == 0.5 {
            return 0.0;
        }

        let a = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * a.ln()).sqrt();
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p < 0.5 { -z } else { z }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rt() {
        let config = CognitiveConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CognitiveGenerator::new(config);
        let output = generator.generate_simple_rt(100);

        assert_eq!(output.trial.len(), 100);
        assert!(output.ground_truth.metrics.mean_rt > 0.0);
        assert!(output.ground_truth.metrics.accuracy > 0.8);
    }

    #[test]
    fn test_flanker() {
        let config = CognitiveConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CognitiveGenerator::new(config);
        let output = generator.generate_flanker(200, 0.5);

        // Should have congruency effect
        assert!(output.ground_truth.metrics.congruency_effect.is_some());
        let effect = output.ground_truth.metrics.congruency_effect.unwrap();
        assert!(effect > 0.0); // Incongruent should be slower
    }

    #[test]
    fn test_go_nogo() {
        let config = CognitiveConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CognitiveGenerator::new(config);
        let output = generator.generate_go_nogo(200, 0.7);

        // Should have hits and correct rejections
        let hits = output
            .response
            .iter()
            .filter(|&&r| r == ResponseType::Hit)
            .count();
        let crs = output
            .response
            .iter()
            .filter(|&&r| r == ResponseType::CorrectRejection)
            .count();
        assert!(hits > 0);
        assert!(crs > 0);
    }

    #[test]
    fn test_nback() {
        let config = CognitiveConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CognitiveGenerator::new(config);

        let nback_1 = generator.generate_nback(100, 1, 0.3);
        let nback_2 = generator.generate_nback(100, 2, 0.3);

        // 2-back should be harder (lower d-prime or slower)
        assert!(nback_2.ground_truth.metrics.mean_rt >= nback_1.ground_truth.metrics.mean_rt * 0.9);
    }

    #[test]
    fn test_adhd_pathology() {
        let config = CognitiveConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CognitiveGenerator::new(config);

        let normal = generator.generate_simple_rt(100);
        let adhd = generator.generate_pathological(
            CognitiveTask::SimpleRt,
            CognitivePathology::Adhd {
                variability_increase: 0.5,
                lapse_rate: 0.1,
            },
            100,
        );

        // ADHD should have higher RT variability
        assert!(adhd.ground_truth.metrics.cv_rt > normal.ground_truth.metrics.cv_rt * 0.9);
    }
}
