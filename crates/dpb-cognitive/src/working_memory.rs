//! Working memory assessment tasks
//!
//! Implements n-back tasks for assessing working memory capacity and updating.

use rand::prelude::*;

/// N-back working memory task
#[derive(Debug, Clone)]
pub struct NBackTask {
    /// Level of n-back (1-back, 2-back, etc.)
    pub n_level: usize,
    /// Duration each stimulus is shown in milliseconds
    pub stimulus_duration: f64,
    /// Interval between stimuli in milliseconds
    pub inter_stimulus_interval: f64,
    /// Target percentage of trials that should be targets
    pub target_percentage: f64,
}

impl Default for NBackTask {
    fn default() -> Self {
        Self {
            n_level: 2,
            stimulus_duration: 500.0,
            inter_stimulus_interval: 2000.0,
            target_percentage: 0.3,
        }
    }
}

impl NBackTask {
    /// Create a new n-back task
    pub fn new(
        n_level: usize,
        stimulus_duration: f64,
        inter_stimulus_interval: f64,
        target_percentage: f64,
    ) -> Self {
        assert!(n_level > 0, "N-level must be at least 1");
        assert!(
            target_percentage > 0.0 && target_percentage < 1.0,
            "Target percentage must be between 0 and 1"
        );

        Self {
            n_level,
            stimulus_duration,
            inter_stimulus_interval,
            target_percentage,
        }
    }

    /// Generate a sequence of stimuli with proper n-back targets
    pub fn generate_sequence(
        &self,
        n_trials: usize,
        stimulus_type: StimulusSet,
    ) -> Vec<NBackTrial> {
        let mut rng = rand::rng();
        let stimuli = stimulus_type.get_stimuli();
        let mut sequence = Vec::with_capacity(n_trials);

        // Initialize first n trials randomly (these cannot be targets)
        for i in 0..self.n_level.min(n_trials) {
            let stimulus = stimuli.choose(&mut rng).unwrap().clone();
            sequence.push(NBackTrial {
                trial_number: i,
                stimulus: stimulus.clone(),
                is_target: false,
            });
        }

        // Generate remaining trials with target percentage control
        for i in self.n_level..n_trials {
            let is_target: bool = rng.random_bool(self.target_percentage);

            let stimulus = if is_target {
                // Make it match n positions back
                sequence[i - self.n_level].stimulus.clone()
            } else {
                // Make sure it doesn't match n positions back
                let mut candidate = stimuli.choose(&mut rng).unwrap().clone();
                let target_stimulus = &sequence[i - self.n_level].stimulus;

                // Resample if we accidentally picked a match
                while &candidate == target_stimulus && stimuli.len() > 1 {
                    candidate = stimuli.choose(&mut rng).unwrap().clone();
                }
                candidate
            };

            sequence.push(NBackTrial {
                trial_number: i,
                stimulus,
                is_target,
            });
        }

        sequence
    }

    /// Score participant responses and calculate performance metrics
    pub fn score_responses(
        &self,
        trials: &[NBackTrial],
        responses: &[NBackResponse],
    ) -> NBackMetrics {
        let mut hits = 0;
        let mut misses = 0;
        let mut false_alarms = 0;
        let mut correct_rejections = 0;
        let mut rt_hits = Vec::new();

        for (trial, response) in trials.iter().zip(responses.iter()) {
            match (trial.is_target, response.responded_target) {
                (true, true) => {
                    hits += 1;
                    if let Some(rt) = response.response_time {
                        rt_hits.push(rt);
                    }
                }
                (true, false) => misses += 1,
                (false, true) => false_alarms += 1,
                (false, false) => correct_rejections += 1,
            }
        }

        let mean_rt_hits = if rt_hits.is_empty() {
            0.0
        } else {
            rt_hits.iter().sum::<f64>() / rt_hits.len() as f64
        };

        let d_prime = calculate_d_prime(hits, misses, false_alarms, correct_rejections);
        let response_bias = calculate_response_bias(hits, misses, false_alarms, correct_rejections);

        NBackMetrics {
            hits,
            misses,
            false_alarms,
            correct_rejections,
            d_prime,
            response_bias,
            mean_rt_hits,
        }
    }
}

/// Performance metrics for n-back task
#[derive(Debug, Clone, Default)]
pub struct NBackMetrics {
    /// Number of correctly identified targets
    pub hits: usize,
    /// Number of missed targets
    pub misses: usize,
    /// Number of non-targets incorrectly identified as targets
    pub false_alarms: usize,
    /// Number of correctly identified non-targets
    pub correct_rejections: usize,
    /// Sensitivity index (d-prime)
    pub d_prime: f64,
    /// Response bias (criterion)
    pub response_bias: f64,
    /// Mean reaction time for hits
    pub mean_rt_hits: f64,
}

/// A single trial in an n-back task
#[derive(Debug, Clone)]
pub struct NBackTrial {
    /// Trial index
    pub trial_number: usize,
    /// Stimulus presented
    pub stimulus: Stimulus,
    /// Whether this is a target (matches n-back)
    pub is_target: bool,
}

/// Participant response to an n-back trial
#[derive(Debug, Clone)]
pub struct NBackResponse {
    /// Trial number
    pub trial_number: usize,
    /// Whether participant indicated target
    pub responded_target: bool,
    /// Response time in milliseconds
    pub response_time: Option<f64>,
}

/// Type of stimulus used in n-back task
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stimulus {
    /// Letter stimulus (A-Z)
    Letter(char),
    /// Number stimulus (0-9)
    Number(u8),
    /// Spatial position (x, y coordinates)
    Spatial(u8, u8),
}

/// Set of stimuli to use in the task
#[derive(Debug, Clone)]
pub enum StimulusSet {
    /// Use letters
    Letters,
    /// Use numbers
    Numbers,
    /// Use spatial positions (grid_size x grid_size)
    Spatial(u8),
}

impl StimulusSet {
    /// Get the available stimuli for this set
    pub fn get_stimuli(&self) -> Vec<Stimulus> {
        match self {
            StimulusSet::Letters => {
                ('A'..='Z').map(Stimulus::Letter).collect()
            }
            StimulusSet::Numbers => {
                (0..=9).map(Stimulus::Number).collect()
            }
            StimulusSet::Spatial(grid_size) => {
                let mut positions = Vec::new();
                for x in 0..*grid_size {
                    for y in 0..*grid_size {
                        positions.push(Stimulus::Spatial(x, y));
                    }
                }
                positions
            }
        }
    }
}

/// Calculate d-prime (sensitivity) using signal detection theory
///
/// d' = Z(hit_rate) - Z(false_alarm_rate)
pub fn calculate_d_prime(
    hits: usize,
    misses: usize,
    false_alarms: usize,
    correct_rejections: usize,
) -> f64 {
    // Avoid edge cases (0 or 1) by using log-linear correction
    let n_signal = hits + misses;
    let n_noise = false_alarms + correct_rejections;

    if n_signal == 0 || n_noise == 0 {
        return 0.0;
    }

    // Apply log-linear correction for extreme values
    let hit_rate = (hits as f64 + 0.5) / (n_signal as f64 + 1.0);
    let fa_rate = (false_alarms as f64 + 0.5) / (n_noise as f64 + 1.0);

    // Use inverse normal CDF approximation
    z_score(hit_rate) - z_score(fa_rate)
}

/// Calculate response bias (criterion) using signal detection theory
///
/// c = -0.5 * [Z(hit_rate) + Z(false_alarm_rate)]
pub fn calculate_response_bias(
    hits: usize,
    misses: usize,
    false_alarms: usize,
    correct_rejections: usize,
) -> f64 {
    let n_signal = hits + misses;
    let n_noise = false_alarms + correct_rejections;

    if n_signal == 0 || n_noise == 0 {
        return 0.0;
    }

    let hit_rate = (hits as f64 + 0.5) / (n_signal as f64 + 1.0);
    let fa_rate = (false_alarms as f64 + 0.5) / (n_noise as f64 + 1.0);

    -0.5 * (z_score(hit_rate) + z_score(fa_rate))
}

/// Approximate inverse normal CDF (z-score) using Beasley-Springer-Moro algorithm
pub fn z_score(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "Probability must be between 0 and 1");

    // Constants for approximation
    const A: [f64; 4] = [
        2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637,
    ];

    const B: [f64; 4] = [
        -8.47351093090,
        23.08336743743,
        -21.06224101826,
        3.13082909833,
    ];

    const C: [f64; 9] = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ];

    let y = p - 0.5;

    if y.abs() < 0.42 {
        let r = y * y;
        y * (((A[3] * r + A[2]) * r + A[1]) * r + A[0])
            / ((((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0)
    } else {
        let r = if y > 0.0 { 1.0 - p } else { p };
        let s = r.ln();
        let t = (-s).sqrt();

        let num = ((C[8] * t + C[7]) * t + C[6]) * t + C[5];
        let num = ((num * t + C[4]) * t + C[3]) * t + C[2];
        let num = ((num * t + C[1]) * t + C[0]) * t - s;

        if y < 0.0 {
            -num
        } else {
            num
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nback_sequence_generation() {
        let task = NBackTask::new(2, 500.0, 2000.0, 0.3);
        let sequence = task.generate_sequence(50, StimulusSet::Letters);
        assert_eq!(sequence.len(), 50);

        // First n trials should not be targets
        assert!(!sequence[0].is_target);
        assert!(!sequence[1].is_target);
    }

    #[test]
    fn test_d_prime_calculation() {
        let d_prime = calculate_d_prime(80, 20, 20, 80);
        assert!(d_prime > 0.0); // Should show good discrimination
    }

    #[test]
    fn test_stimulus_sets() {
        let letters = StimulusSet::Letters.get_stimuli();
        assert_eq!(letters.len(), 26);

        let numbers = StimulusSet::Numbers.get_stimuli();
        assert_eq!(numbers.len(), 10);

        let spatial = StimulusSet::Spatial(3).get_stimuli();
        assert_eq!(spatial.len(), 9); // 3x3 grid
    }

    #[test]
    fn test_z_score() {
        assert!((z_score(0.5) - 0.0).abs() < 0.01);
        assert!(z_score(0.84) > 0.9 && z_score(0.84) < 1.1);
        assert!(z_score(0.16) < -0.9 && z_score(0.16) > -1.1);
    }
}
