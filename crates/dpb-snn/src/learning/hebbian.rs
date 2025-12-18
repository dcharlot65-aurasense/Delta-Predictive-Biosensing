//! Hebbian Learning Rules for Spiking Neural Networks
//!
//! This module implements various Hebbian learning rules used for unsupervised
//! learning in SNNs, including STDP, BCM, and Oja's rule.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f64::consts::E;

/// Trait for Hebbian-style learning rules
pub trait HebbianRule: Send + Sync + std::fmt::Debug {
    /// Compute weight update based on pre- and post-synaptic traces
    ///
    /// # Arguments
    /// * `pre_trace` - Pre-synaptic trace value
    /// * `post_trace` - Post-synaptic trace value
    /// * `weight` - Current synaptic weight
    ///
    /// # Returns
    /// The weight change (Δw)
    fn compute_weight_update(&self, pre_trace: f64, post_trace: f64, weight: f64) -> f64;

    /// Update weight based on pre- and post-synaptic spikes
    fn update_weight(&self, weight: f64, pre_trace: f64, post_trace: f64) -> f64 {
        weight + self.compute_weight_update(pre_trace, post_trace, weight)
    }

    /// Apply weight bounds
    fn apply_bounds(&self, weight: f64, w_min: f64, w_max: f64) -> f64 {
        weight.max(w_min).min(w_max)
    }
}

/// Spike-Timing-Dependent Plasticity (STDP)
///
/// STDP is a biological learning rule where synaptic strength changes
/// based on the relative timing of pre- and post-synaptic spikes.
///
/// Long-Term Potentiation (LTP): Pre before Post → Δw > 0
/// Long-Term Depression (LTD): Post before Pre → Δw < 0
///
/// # References
/// - Bi & Poo (1998). "Synaptic modifications in cultured hippocampal neurons"
/// - Song et al. (2000). "Competitive Hebbian learning through STDP"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDP {
    /// LTP amplitude (typically 0.01)
    pub a_plus: f64,
    /// LTD amplitude (typically 0.01)
    pub a_minus: f64,
    /// LTP time constant in ms (typically 20ms)
    pub tau_plus: f64,
    /// LTD time constant in ms (typically 20ms)
    pub tau_minus: f64,
    /// Weight-dependent scaling (for soft bounds)
    pub weight_dependent: bool,
    /// Maximum weight for soft bounds
    pub w_max: f64,
}

impl Default for STDP {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.01,
            tau_plus: 20.0,
            tau_minus: 20.0,
            weight_dependent: false,
            w_max: 1.0,
        }
    }
}

impl STDP {
    /// Create STDP with custom parameters
    pub fn new(a_plus: f64, a_minus: f64, tau_plus: f64, tau_minus: f64) -> Self {
        Self {
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            weight_dependent: false,
            w_max: 1.0,
        }
    }

    /// Enable weight-dependent STDP (multiplicative)
    pub fn with_weight_dependence(mut self, w_max: f64) -> Self {
        self.weight_dependent = true;
        self.w_max = w_max;
        self
    }

    /// Compute exponential STDP kernel
    ///
    /// # Arguments
    /// * `delta_t` - Time difference (post - pre) in ms
    ///
    /// # Returns
    /// Weight change according to STDP rule
    pub fn stdp_kernel(&self, delta_t: f64) -> f64 {
        if delta_t > 0.0 {
            // LTP: pre before post
            self.a_plus * (-delta_t / self.tau_plus).exp()
        } else {
            // LTD: post before pre
            -self.a_minus * (delta_t / self.tau_minus).exp()
        }
    }

    /// Update spike trace (exponential decay)
    pub fn update_trace(trace: f64, spiked: bool, tau: f64, dt: f64) -> f64 {
        let decay = (-dt / tau).exp();
        if spiked {
            decay * trace + 1.0
        } else {
            decay * trace
        }
    }
}

impl HebbianRule for STDP {
    fn compute_weight_update(&self, pre_trace: f64, post_trace: f64, weight: f64) -> f64 {
        if self.weight_dependent {
            // Multiplicative STDP (soft bounds)
            let ltp = self.a_plus * pre_trace * (self.w_max - weight);
            let ltd = self.a_minus * post_trace * weight;
            ltp - ltd
        } else {
            // Additive STDP
            self.a_plus * pre_trace - self.a_minus * post_trace
        }
    }
}

/// Bienenstock-Cooper-Munro (BCM) Rule
///
/// BCM theory provides a sliding threshold mechanism for synaptic plasticity,
/// where the modification threshold depends on the postsynaptic activity history.
///
/// # References
/// - Bienenstock et al. (1982). "Theory for the development of neuron selectivity"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BCMRule {
    /// Modification threshold (slides based on activity)
    pub theta: f64,
    /// Threshold adaptation time constant in ms
    pub tau_theta: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Exponent for threshold computation (typically 2)
    pub p: f64,
}

impl Default for BCMRule {
    fn default() -> Self {
        Self {
            theta: 1.0,
            tau_theta: 10000.0, // 10 seconds
            learning_rate: 0.001,
            p: 2.0,
        }
    }
}

impl BCMRule {
    /// Create BCM rule with custom parameters
    pub fn new(learning_rate: f64, tau_theta: f64) -> Self {
        Self {
            theta: 1.0,
            tau_theta,
            learning_rate,
            p: 2.0,
        }
    }

    /// Update the modification threshold based on postsynaptic activity
    ///
    /// θ(t+1) = θ(t) + (1/τ_θ)[y^p - θ(t)]
    ///
    /// # Arguments
    /// * `post_activity` - Current postsynaptic activity
    /// * `dt` - Time step in ms
    pub fn update_threshold(&mut self, post_activity: f64, dt: f64) {
        let target = post_activity.powf(self.p);
        self.theta += (dt / self.tau_theta) * (target - self.theta);
    }

    /// BCM learning rule
    ///
    /// Δw = η * x * y * (y - θ)
    ///
    /// where:
    /// - η is the learning rate
    /// - x is presynaptic activity
    /// - y is postsynaptic activity
    /// - θ is the modification threshold
    pub fn bcm_update(&self, pre_activity: f64, post_activity: f64) -> f64 {
        self.learning_rate * pre_activity * post_activity * (post_activity - self.theta)
    }
}

impl HebbianRule for BCMRule {
    fn compute_weight_update(&self, pre_trace: f64, post_trace: f64, _weight: f64) -> f64 {
        self.bcm_update(pre_trace, post_trace)
    }
}

/// Oja's Learning Rule
///
/// Oja's rule is a normalized Hebbian learning rule that prevents unbounded
/// weight growth by including a weight decay term proportional to the square
/// of the postsynaptic activity.
///
/// # References
/// - Oja (1982). "Simplified neuron model as a principal component analyzer"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OjasRule {
    /// Learning rate
    pub learning_rate: f64,
    /// Optional weight normalization
    pub normalize: bool,
}

impl Default for OjasRule {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            normalize: true,
        }
    }
}

impl OjasRule {
    /// Create Oja's rule with custom learning rate
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            normalize: true,
        }
    }

    /// Oja's learning rule
    ///
    /// Δw = η * [y * x - y² * w]
    ///
    /// where:
    /// - η is the learning rate
    /// - x is presynaptic activity
    /// - y is postsynaptic activity
    /// - w is the current weight
    pub fn ojas_update(&self, pre_activity: f64, post_activity: f64, weight: f64) -> f64 {
        self.learning_rate * (post_activity * pre_activity - post_activity.powi(2) * weight)
    }
}

impl HebbianRule for OjasRule {
    fn compute_weight_update(&self, pre_trace: f64, post_trace: f64, weight: f64) -> f64 {
        self.ojas_update(pre_trace, post_trace, weight)
    }
}

/// Covariance Rule (Sejnowski's Rule)
///
/// A normalized Hebbian rule based on the covariance between pre- and post-synaptic activities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovarianceRule {
    /// Learning rate
    pub learning_rate: f64,
    /// Average presynaptic activity
    pub pre_mean: f64,
    /// Average postsynaptic activity
    pub post_mean: f64,
    /// Time constant for mean tracking
    pub tau_mean: f64,
}

impl Default for CovarianceRule {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            pre_mean: 0.0,
            post_mean: 0.0,
            tau_mean: 1000.0,
        }
    }
}

impl CovarianceRule {
    /// Update activity means
    pub fn update_means(&mut self, pre_activity: f64, post_activity: f64, dt: f64) {
        let alpha = dt / self.tau_mean;
        self.pre_mean += alpha * (pre_activity - self.pre_mean);
        self.post_mean += alpha * (post_activity - self.post_mean);
    }
}

impl HebbianRule for CovarianceRule {
    fn compute_weight_update(&self, pre_trace: f64, post_trace: f64, _weight: f64) -> f64 {
        self.learning_rate * (pre_trace - self.pre_mean) * (post_trace - self.post_mean)
    }
}

/// Synaptic trace for STDP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticTrace {
    /// Pre-synaptic trace
    pub pre: f64,
    /// Post-synaptic trace
    pub post: f64,
    /// Time constant for pre-synaptic trace (ms)
    pub tau_pre: f64,
    /// Time constant for post-synaptic trace (ms)
    pub tau_post: f64,
}

impl SynapticTrace {
    /// Create new synaptic trace
    pub fn new(tau_pre: f64, tau_post: f64) -> Self {
        Self {
            pre: 0.0,
            post: 0.0,
            tau_pre,
            tau_post,
        }
    }

    /// Update traces with exponential decay and spike
    pub fn update(&mut self, pre_spike: bool, post_spike: bool, dt: f64) {
        // Exponential decay
        self.pre *= (-dt / self.tau_pre).exp();
        self.post *= (-dt / self.tau_post).exp();

        // Add spike contributions
        if pre_spike {
            self.pre += 1.0;
        }
        if post_spike {
            self.post += 1.0;
        }
    }

    /// Reset traces
    pub fn reset(&mut self) {
        self.pre = 0.0;
        self.post = 0.0;
    }
}

/// Hebbian learning layer
///
/// Applies Hebbian learning to a layer of synapses
#[derive(Debug)]
pub struct HebbianLayer {
    /// Synaptic weights (n_post x n_pre)
    pub weights: Array2<f64>,
    /// Synaptic traces for each synapse
    traces: Vec<SynapticTrace>,
    /// Learning rule
    rule: Box<dyn HebbianRule>,
    /// Weight bounds
    pub w_min: f64,
    pub w_max: f64,
}

impl HebbianLayer {
    /// Create a new Hebbian learning layer
    pub fn new<R: HebbianRule + 'static>(
        n_pre: usize,
        n_post: usize,
        rule: R,
        tau_pre: f64,
        tau_post: f64,
    ) -> Self {
        let weights = Array2::from_elem((n_post, n_pre), 0.1);
        let traces = vec![SynapticTrace::new(tau_pre, tau_post); n_post * n_pre];

        Self {
            weights,
            traces,
            rule: Box::new(rule),
            w_min: 0.0,
            w_max: 1.0,
        }
    }

    /// Update weights based on pre- and post-synaptic spikes
    pub fn update(&mut self, pre_spikes: &Array1<bool>, post_spikes: &Array1<bool>, dt: f64) {
        let n_post = self.weights.nrows();
        let n_pre = self.weights.ncols();

        for i in 0..n_post {
            for j in 0..n_pre {
                let idx = i * n_pre + j;
                let trace = &mut self.traces[idx];

                // Update trace
                trace.update(pre_spikes[j], post_spikes[i], dt);

                // Compute weight update
                let delta_w = self.rule.compute_weight_update(
                    trace.pre,
                    trace.post,
                    self.weights[[i, j]],
                );

                // Apply update with bounds
                self.weights[[i, j]] = self.rule.apply_bounds(
                    self.weights[[i, j]] + delta_w,
                    self.w_min,
                    self.w_max,
                );
            }
        }
    }

    /// Reset all traces
    pub fn reset_traces(&mut self) {
        for trace in &mut self.traces {
            trace.reset();
        }
    }

    /// Get average weight
    pub fn mean_weight(&self) -> f64 {
        self.weights.mean().unwrap_or(0.0)
    }

    /// Get weight sparsity (fraction of near-zero weights)
    pub fn sparsity(&self, threshold: f64) -> f64 {
        let near_zero = self.weights.iter().filter(|&&w| w.abs() < threshold).count();
        near_zero as f64 / self.weights.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_stdp_default() {
        let stdp = STDP::default();
        assert_eq!(stdp.a_plus, 0.01);
        assert_eq!(stdp.a_minus, 0.01);
        assert_eq!(stdp.tau_plus, 20.0);
        assert_eq!(stdp.tau_minus, 20.0);
    }

    #[test]
    fn test_stdp_kernel() {
        let stdp = STDP::default();

        // LTP: pre before post (Δt > 0)
        let ltp = stdp.stdp_kernel(10.0);
        assert!(ltp > 0.0);

        // LTD: post before pre (Δt < 0)
        let ltd = stdp.stdp_kernel(-10.0);
        assert!(ltd < 0.0);

        // At t=0, slight LTP bias
        let zero = stdp.stdp_kernel(0.0);
        assert!(zero >= 0.0);
    }

    #[test]
    fn test_stdp_trace_update() {
        let tau = 20.0;
        let dt = 1.0;

        // No spike
        let trace = STDP::update_trace(1.0, false, tau, dt);
        assert!(trace < 1.0); // Should decay

        // With spike
        let trace = STDP::update_trace(0.5, true, tau, dt);
        assert!(trace > 1.0); // Should increase
    }

    #[test]
    fn test_stdp_weight_update() {
        let stdp = STDP::default();

        // LTP scenario: strong pre trace, weak post trace
        let delta_w = stdp.compute_weight_update(1.0, 0.1, 0.5);
        assert!(delta_w > 0.0);

        // LTD scenario: weak pre trace, strong post trace
        let delta_w = stdp.compute_weight_update(0.1, 1.0, 0.5);
        assert!(delta_w < 0.0);
    }

    #[test]
    fn test_stdp_weight_dependent() {
        let stdp = STDP::default().with_weight_dependence(1.0);

        // For low weight, LTP should be stronger
        let delta_w_low = stdp.compute_weight_update(1.0, 0.0, 0.1);

        // For high weight, LTP should be weaker
        let delta_w_high = stdp.compute_weight_update(1.0, 0.0, 0.9);

        assert!(delta_w_low > delta_w_high);
    }

    #[test]
    fn test_bcm_default() {
        let bcm = BCMRule::default();
        assert_eq!(bcm.theta, 1.0);
        assert_eq!(bcm.learning_rate, 0.001);
    }

    #[test]
    fn test_bcm_threshold_update() {
        let mut bcm = BCMRule::default();
        let initial_theta = bcm.theta;

        // High activity should increase threshold
        bcm.update_threshold(2.0, 100.0);
        assert!(bcm.theta > initial_theta);
    }

    #[test]
    fn test_bcm_update() {
        let bcm = BCMRule {
            theta: 1.0,
            learning_rate: 0.01,
            tau_theta: 1000.0,
            p: 2.0,
        };

        // When post > theta, potentiation
        let delta_w = bcm.bcm_update(1.0, 1.5);
        assert!(delta_w > 0.0);

        // When post < theta, depression
        let delta_w = bcm.bcm_update(1.0, 0.5);
        assert!(delta_w < 0.0);
    }

    #[test]
    fn test_ojas_rule_default() {
        let ojas = OjasRule::default();
        assert_eq!(ojas.learning_rate, 0.001);
        assert!(ojas.normalize);
    }

    #[test]
    fn test_ojas_update() {
        let ojas = OjasRule::new(0.01);

        // Standard Hebbian term minus weight decay
        let delta_w = ojas.ojas_update(1.0, 1.0, 0.5);

        // Should have both Hebbian and decay components
        let hebbian = 0.01 * 1.0 * 1.0;
        let decay = 0.01 * 1.0 * 1.0 * 0.5;
        assert_abs_diff_eq!(delta_w, hebbian - decay, epsilon = 1e-6);
    }

    #[test]
    fn test_covariance_rule() {
        let cov = CovarianceRule::default();

        // With zero means, should be standard Hebbian
        let delta_w = cov.compute_weight_update(1.0, 1.0, 0.5);
        assert_abs_diff_eq!(delta_w, cov.learning_rate * 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_synaptic_trace() {
        let mut trace = SynapticTrace::new(20.0, 20.0);

        // Initial state
        assert_eq!(trace.pre, 0.0);
        assert_eq!(trace.post, 0.0);

        // Update with pre spike
        trace.update(true, false, 1.0);
        assert!(trace.pre > 0.0);
        assert_eq!(trace.post, 0.0);

        // Update with post spike
        trace.update(false, true, 1.0);
        assert!(trace.pre > 0.0); // Decayed but still > 0
        assert!(trace.post > 0.0);

        // Decay without spikes
        let pre_before = trace.pre;
        trace.update(false, false, 1.0);
        assert!(trace.pre < pre_before);
    }

    #[test]
    fn test_hebbian_layer_creation() {
        let stdp = STDP::default();
        let layer = HebbianLayer::new(10, 5, stdp, 20.0, 20.0);

        assert_eq!(layer.weights.shape(), &[5, 10]);
        assert_eq!(layer.traces.len(), 50);
    }

    #[test]
    fn test_hebbian_layer_update() {
        let stdp = STDP::default();
        let mut layer = HebbianLayer::new(3, 2, stdp, 20.0, 20.0);

        // Set initial weights
        layer.weights.fill(0.5);

        let pre_spikes = Array1::from_vec(vec![true, false, true]);
        let post_spikes = Array1::from_vec(vec![true, false]);

        // Update
        layer.update(&pre_spikes, &post_spikes, 1.0);

        // Weights should have changed
        let mean = layer.mean_weight();
        assert!((mean - 0.5).abs() > 1e-6);
    }

    #[test]
    fn test_hebbian_layer_bounds() {
        let stdp = STDP::default();
        let mut layer = HebbianLayer::new(2, 2, stdp, 20.0, 20.0);
        layer.w_min = 0.0;
        layer.w_max = 1.0;

        // Set weights near boundary
        layer.weights.fill(0.99);

        // Strong potentiation
        let pre_spikes = Array1::from_vec(vec![true, true]);
        let post_spikes = Array1::from_vec(vec![true, true]);

        for _ in 0..100 {
            layer.update(&pre_spikes, &post_spikes, 1.0);
        }

        // Should be bounded at w_max
        for &w in layer.weights.iter() {
            assert!(w <= layer.w_max);
            assert!(w >= layer.w_min);
        }
    }

    #[test]
    fn test_hebbian_layer_sparsity() {
        let ojas = OjasRule::new(0.1);
        let mut layer = HebbianLayer::new(10, 10, ojas, 20.0, 20.0);

        // Set some weights to zero
        for i in 0..5 {
            layer.weights[[i, i]] = 0.0;
        }

        let sparsity = layer.sparsity(0.01);
        assert!(sparsity >= 0.05); // At least 5% sparse
    }
}
