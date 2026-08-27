//! Reward-Modulated Learning
//!
//! This module implements reward-modulated plasticity mechanisms including:
//! - Three-factor learning rules (pre, post, reward)
//! - Eligibility traces for temporal credit assignment
//! - Reward-modulated STDP
//! - Intrinsic motivation
//! - Reward shaping

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Reward signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardSignal {
    /// Current reward value
    pub value: f64,
    /// Reward history (for moving average)
    pub history: VecDeque<f64>,
    /// Maximum history length
    pub max_history: usize,
    /// Reward baseline (for normalization)
    pub baseline: f64,
}

impl RewardSignal {
    /// Create new reward signal
    pub fn new(max_history: usize) -> Self {
        Self {
            value: 0.0,
            history: VecDeque::with_capacity(max_history),
            max_history,
            baseline: 0.0,
        }
    }

    /// Update reward signal
    pub fn update(&mut self, reward: f64) {
        self.value = reward;

        // Add to history
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(reward);

        // Update baseline (moving average)
        self.baseline = self.history.iter().sum::<f64>() / self.history.len() as f64;
    }

    /// Get normalized reward (subtract baseline)
    pub fn normalized(&self) -> f64 {
        self.value - self.baseline
    }

    /// Get mean reward over history
    pub fn mean(&self) -> f64 {
        if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().sum::<f64>() / self.history.len() as f64
        }
    }

    /// Get reward variance
    pub fn variance(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let mean = self.mean();
        let variance: f64 = self.history.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64;

        variance
    }

    /// Reset reward signal
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.history.clear();
        self.baseline = 0.0;
    }
}

/// Eligibility trace for temporal credit assignment
///
/// Tracks recent synaptic activity to determine which synapses
/// should be modified when a delayed reward arrives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityTrace {
    /// Current trace value
    pub trace: f64,
    /// Time constant for trace decay (ms)
    pub tau: f64,
    /// Maximum trace value
    pub max_trace: f64,
}

impl EligibilityTrace {
    /// Create new eligibility trace
    pub fn new(tau: f64) -> Self {
        Self {
            trace: 0.0,
            tau,
            max_trace: 1.0,
        }
    }

    /// Update trace based on synaptic activity
    pub fn update(&mut self, dt: f64, activity: f64) {
        // Increase trace with activity
        self.trace += activity;

        // Decay trace
        self.trace -= self.trace * (dt / self.tau);

        // Apply bounds
        self.trace = self.trace.max(0.0).min(self.max_trace);
    }

    /// Reset trace
    pub fn reset(&mut self) {
        self.trace = 0.0;
    }

    /// Get trace value
    pub fn get(&self) -> f64 {
        self.trace
    }
}

/// Three-factor learning rule
///
/// Synaptic plasticity depends on three factors:
/// 1. Pre-synaptic activity
/// 2. Post-synaptic activity
/// 3. Neuromodulatory signal (e.g., dopamine/reward)
///
/// Δw = η * pre * post * modulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeFactorRule {
    /// Learning rate
    pub learning_rate: f64,
    /// Minimum weight
    pub w_min: f64,
    /// Maximum weight
    pub w_max: f64,
}

impl ThreeFactorRule {
    /// Create new three-factor rule
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            w_min: 0.0,
            w_max: 1.0,
        }
    }

    /// Compute weight update
    pub fn compute_update(&self, pre_activity: f64, post_activity: f64, modulator: f64) -> f64 {
        self.learning_rate * pre_activity * post_activity * modulator
    }

    /// Apply weight update with bounds
    pub fn update_weight(
        &self,
        weight: f64,
        pre_activity: f64,
        post_activity: f64,
        modulator: f64,
    ) -> f64 {
        let delta_w = self.compute_update(pre_activity, post_activity, modulator);
        (weight + delta_w).max(self.w_min).min(self.w_max)
    }
}

/// Reward-modulated STDP
///
/// Combines spike-timing-dependent plasticity with reward signals.
/// Synaptic changes are stored in eligibility traces and applied when
/// reward arrives, solving the temporal credit assignment problem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardModulatedSTDP {
    /// Base learning rate
    pub learning_rate: f64,
    /// LTP amplitude
    pub a_plus: f64,
    /// LTD amplitude
    pub a_minus: f64,
    /// LTP time constant (ms)
    pub tau_plus: f64,
    /// LTD time constant (ms)
    pub tau_minus: f64,
    /// Eligibility trace time constant (ms)
    pub tau_eligibility: f64,
    /// Minimum weight
    pub w_min: f64,
    /// Maximum weight
    pub w_max: f64,
}

impl RewardModulatedSTDP {
    /// Create new reward-modulated STDP
    pub fn new(learning_rate: f64, tau_eligibility: f64) -> Self {
        Self {
            learning_rate,
            a_plus: 0.01,
            a_minus: 0.01,
            tau_plus: 20.0,
            tau_minus: 20.0,
            tau_eligibility,
            w_min: 0.0,
            w_max: 1.0,
        }
    }

    /// Compute STDP component (without reward)
    pub fn compute_stdp(&self, delta_t: f64) -> f64 {
        if delta_t > 0.0 {
            // Post after pre → LTP
            self.a_plus * (-delta_t / self.tau_plus).exp()
        } else {
            // Pre after post → LTD
            -self.a_minus * (delta_t / self.tau_minus).exp()
        }
    }

    /// Compute weight update with reward modulation
    ///
    /// # Arguments
    /// * `pre_trace` - Pre-synaptic trace
    /// * `post_trace` - Post-synaptic trace
    /// * `eligibility` - Eligibility trace value
    /// * `reward` - Reward signal
    pub fn compute_update(
        &self,
        pre_trace: f64,
        post_trace: f64,
        eligibility: f64,
        reward: f64,
    ) -> f64 {
        // Three-factor rule: pre × post × reward
        // Modified by eligibility trace for temporal credit assignment
        self.learning_rate * pre_trace * post_trace * eligibility * reward
    }

    /// Update weight
    pub fn update_weight(
        &self,
        weight: f64,
        pre_trace: f64,
        post_trace: f64,
        eligibility: f64,
        reward: f64,
    ) -> f64 {
        let delta_w = self.compute_update(pre_trace, post_trace, eligibility, reward);
        (weight + delta_w).max(self.w_min).min(self.w_max)
    }
}

/// Temporal credit assignment
///
/// Assigns credit for rewards to earlier state-action pairs
/// using eligibility traces or n-step returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCreditAssignment {
    /// Discount factor
    pub gamma: f64,
    /// Eligibility trace decay
    pub lambda: f64,
    /// N-step return length
    pub n_steps: usize,
    /// Reward trace history
    pub reward_history: VecDeque<f64>,
}

impl TemporalCreditAssignment {
    /// Create new temporal credit assignment
    pub fn new(gamma: f64, lambda: f64, n_steps: usize) -> Self {
        Self {
            gamma,
            lambda,
            n_steps,
            reward_history: VecDeque::with_capacity(n_steps),
        }
    }

    /// Compute n-step return
    ///
    /// G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γⁿr_{t+n}
    pub fn compute_n_step_return(&self) -> f64 {
        self.reward_history
            .iter()
            .enumerate()
            .map(|(i, &r)| r * self.gamma.powi(i as i32))
            .sum()
    }

    /// Compute TD(λ) return
    ///
    /// Combines n-step returns with exponentially decaying weights
    pub fn compute_td_lambda_return(&self, value: f64) -> f64 {
        let mut td_return = 0.0;
        let mut weight = 1.0;
        let mut total_weight = 0.0;

        for (i, &r) in self.reward_history.iter().enumerate() {
            let discount = self.gamma.powi(i as i32);
            td_return += weight * (r * discount);
            total_weight += weight;
            weight *= self.lambda;
        }

        // Add bootstrapped value estimate
        td_return += weight * (value * self.gamma.powi(self.reward_history.len() as i32));
        total_weight += weight;

        td_return / total_weight
    }

    /// Add reward to history
    pub fn add_reward(&mut self, reward: f64) {
        if self.reward_history.len() >= self.n_steps {
            self.reward_history.pop_front();
        }
        self.reward_history.push_back(reward);
    }

    /// Reset history
    pub fn reset(&mut self) {
        self.reward_history.clear();
    }
}

/// Intrinsic motivation
///
/// Generates internal reward signals based on novelty, curiosity,
/// or prediction error to encourage exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrinsicMotivation {
    /// Novelty bonus scale
    pub novelty_scale: f64,
    /// Prediction error scale
    pub prediction_error_scale: f64,
    /// State visitation counts (for novelty)
    pub visitation_counts: Vec<usize>,
    /// Running prediction error
    pub prediction_error: f64,
}

impl IntrinsicMotivation {
    /// Create new intrinsic motivation
    pub fn new(novelty_scale: f64, prediction_error_scale: f64) -> Self {
        Self {
            novelty_scale,
            prediction_error_scale,
            visitation_counts: Vec::new(),
            prediction_error: 0.0,
        }
    }

    /// Compute novelty bonus
    ///
    /// Higher bonus for less-visited states
    pub fn compute_novelty_bonus(&self, state_id: usize) -> f64 {
        if state_id >= self.visitation_counts.len() {
            // Never seen this state → maximum bonus
            return self.novelty_scale;
        }

        let count = self.visitation_counts[state_id] as f64;
        self.novelty_scale / (count + 1.0).sqrt()
    }

    /// Compute prediction error bonus
    ///
    /// Reward for surprising/unpredictable outcomes
    pub fn compute_prediction_error_bonus(&self) -> f64 {
        self.prediction_error_scale * self.prediction_error.abs()
    }

    /// Update with state visit
    pub fn update_visitation(&mut self, state_id: usize) {
        if state_id >= self.visitation_counts.len() {
            self.visitation_counts.resize(state_id + 1, 0);
        }
        self.visitation_counts[state_id] += 1;
    }

    /// Update prediction error
    pub fn update_prediction_error(&mut self, error: f64) {
        self.prediction_error = error;
    }

    /// Get total intrinsic reward
    pub fn get_intrinsic_reward(&self, state_id: usize) -> f64 {
        self.compute_novelty_bonus(state_id) + self.compute_prediction_error_bonus()
    }
}

/// Reward shaping
///
/// Modifies the reward function to provide additional guidance
/// while preserving optimal policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardShaping {
    /// Discount factor (must match main learning algorithm)
    pub gamma: f64,
    /// Shaping function weight
    pub shaping_weight: f64,
}

impl RewardShaping {
    /// Create new reward shaping
    pub fn new(gamma: f64, shaping_weight: f64) -> Self {
        Self {
            gamma,
            shaping_weight,
        }
    }

    /// Apply potential-based reward shaping
    ///
    /// F(s, s') = γ * Φ(s') - Φ(s)
    ///
    /// Where Φ is a potential function (e.g., distance to goal)
    pub fn shape_reward(
        &self,
        base_reward: f64,
        current_potential: f64,
        next_potential: f64,
    ) -> f64 {
        let shaping = self.gamma * next_potential - current_potential;
        base_reward + self.shaping_weight * shaping
    }

    /// Compute potential from distance to goal
    pub fn distance_potential(&self, distance: f64, max_distance: f64) -> f64 {
        // Higher potential when closer to goal
        1.0 - (distance / max_distance)
    }

    /// Compute potential from progress
    pub fn progress_potential(&self, progress: f64) -> f64 {
        // Linear potential based on task progress
        progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_signal() {
        let mut reward = RewardSignal::new(10);

        reward.update(1.0);
        assert_eq!(reward.value, 1.0);

        reward.update(0.5);
        assert_eq!(reward.mean(), 0.75);

        let normalized = reward.normalized();
        assert_eq!(normalized, 0.5 - 0.75);
    }

    #[test]
    fn test_eligibility_trace() {
        let mut trace = EligibilityTrace::new(20.0);

        // Add activity
        trace.update(1.0, 0.5);
        assert!(trace.get() > 0.0);

        // Decay
        for _ in 0..100 {
            trace.update(1.0, 0.0);
        }
        assert!(trace.get() < 0.1);
    }

    #[test]
    fn test_three_factor_rule() {
        let rule = ThreeFactorRule::new(0.01);

        // Positive modulation
        let update = rule.compute_update(1.0, 1.0, 1.0);
        assert_eq!(update, 0.01);

        // Negative modulation
        let update = rule.compute_update(1.0, 1.0, -1.0);
        assert_eq!(update, -0.01);

        // No modulation
        let update = rule.compute_update(1.0, 1.0, 0.0);
        assert_eq!(update, 0.0);
    }

    #[test]
    fn test_reward_modulated_stdp() {
        let stdp = RewardModulatedSTDP::new(0.01, 20.0);

        // Positive reward → potentiation
        let update = stdp.compute_update(0.5, 0.5, 0.8, 1.0);
        assert!(update > 0.0);

        // Negative reward → depression
        let update = stdp.compute_update(0.5, 0.5, 0.8, -1.0);
        assert!(update < 0.0);

        // No eligibility → no update
        let update = stdp.compute_update(0.5, 0.5, 0.0, 1.0);
        assert_eq!(update, 0.0);
    }

    #[test]
    fn test_temporal_credit_assignment() {
        let mut tca = TemporalCreditAssignment::new(0.99, 0.9, 5);

        tca.add_reward(1.0);
        tca.add_reward(0.5);
        tca.add_reward(0.2);

        let n_step_return = tca.compute_n_step_return();
        assert!(n_step_return > 1.0); // Sum of discounted rewards

        let td_lambda = tca.compute_td_lambda_return(0.0);
        assert!(td_lambda > 0.0);
    }

    #[test]
    fn test_intrinsic_motivation() {
        let mut intrinsic = IntrinsicMotivation::new(1.0, 0.5);

        // First visit → high novelty
        let bonus1 = intrinsic.compute_novelty_bonus(0);
        intrinsic.update_visitation(0);

        // Second visit → lower novelty
        let bonus2 = intrinsic.compute_novelty_bonus(0);
        assert!(bonus1 > bonus2);

        // Prediction error bonus
        intrinsic.update_prediction_error(0.5);
        let pe_bonus = intrinsic.compute_prediction_error_bonus();
        assert!(pe_bonus > 0.0);
    }

    #[test]
    fn test_reward_shaping() {
        let shaping = RewardShaping::new(0.99, 1.0);

        // Moving toward goal
        let shaped = shaping.shape_reward(0.0, 0.3, 0.5);
        assert!(shaped > 0.0);

        // Moving away from goal
        let shaped = shaping.shape_reward(0.0, 0.5, 0.3);
        assert!(shaped < 0.0);
    }

    #[test]
    fn test_stdp_timing() {
        let stdp = RewardModulatedSTDP::new(0.01, 20.0);

        // Pre before post → LTP
        let ltp = stdp.compute_stdp(5.0);
        assert!(ltp > 0.0);

        // Post before pre → LTD
        let ltd = stdp.compute_stdp(-5.0);
        assert!(ltd < 0.0);
    }
}
