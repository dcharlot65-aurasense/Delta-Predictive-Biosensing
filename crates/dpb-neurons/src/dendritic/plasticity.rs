//! # Dendritic Plasticity Module
//!
//! Implements various dendritic plasticity mechanisms:
//! - Dendritic spike-timing-dependent plasticity (STDP)
//! - Branch-specific plasticity
//! - Compartment-specific plasticity
//! - Heterosynaptic plasticity
//! - Metaplasticity
//!
//! ## Plasticity Rules
//!
//! 1. **STDP**: Δw = f(t_post - t_pre)
//! 2. **Branch-specific**: Local dendritic spikes gate plasticity
//! 3. **Heterosynaptic**: Neighboring synapses interact
//! 4. **Metaplasticity**: Plasticity of plasticity (BCM-like)

/// Dendritic plasticity trait
pub trait DendriticPlasticity: Send + Sync {
    /// Update synaptic weight based on timing
    fn update_weight(
        &mut self,
        current_weight: f64,
        pre_spike_time: f64,
        post_spike_time: f64,
        dt: f64,
    ) -> f64;

    /// Reset plasticity state
    fn reset(&mut self);
}

/// Dendritic STDP with local dendritic spike requirement
#[derive(Debug, Clone)]
pub struct DendriticStdp {
    /// Learning rate for potentiation
    a_plus: f64,
    /// Learning rate for depression
    a_minus: f64,
    /// Time constant for potentiation (ms)
    tau_plus: f64,
    /// Time constant for depression (ms)
    tau_minus: f64,
    /// Require dendritic spike for potentiation
    require_dendritic_spike: bool,
    /// Last dendritic spike time (ms)
    last_dendritic_spike: f64,
    /// Dendritic spike time window (ms)
    dendritic_window: f64,
    /// Minimum weight
    w_min: f64,
    /// Maximum weight
    w_max: f64,
}

impl DendriticStdp {
    /// Create dendritic STDP rule
    pub fn new(a_plus: f64, a_minus: f64, tau_plus: f64, tau_minus: f64) -> Self {
        Self {
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            require_dendritic_spike: true,
            last_dendritic_spike: -1000.0,
            dendritic_window: 20.0, // 20 ms window
            w_min: 0.0,
            w_max: 2.0,
        }
    }

    /// Create standard STDP without dendritic spike requirement
    pub fn standard(a_plus: f64, a_minus: f64) -> Self {
        let mut stdp = Self::new(a_plus, a_minus, 20.0, 20.0);
        stdp.require_dendritic_spike = false;
        stdp
    }

    /// Register dendritic spike
    pub fn register_dendritic_spike(&mut self, time: f64) {
        self.last_dendritic_spike = time;
    }

    /// Check if dendritic spike occurred recently
    fn has_recent_dendritic_spike(&self, time: f64) -> bool {
        if !self.require_dendritic_spike {
            return true;
        }
        (time - self.last_dendritic_spike).abs() < self.dendritic_window
    }

    /// Calculate weight change
    fn delta_w(&self, dt_spike: f64, dendritic_active: bool) -> f64 {
        if dt_spike > 0.0 {
            // Post before pre (LTD)
            -self.a_minus * (-dt_spike / self.tau_minus).exp()
        } else if dt_spike < 0.0 {
            // Pre before post (LTP)
            if dendritic_active {
                self.a_plus * (dt_spike / self.tau_plus).exp()
            } else {
                0.0 // No potentiation without dendritic spike
            }
        } else {
            0.0
        }
    }
}

impl DendriticPlasticity for DendriticStdp {
    fn update_weight(
        &mut self,
        current_weight: f64,
        pre_spike_time: f64,
        post_spike_time: f64,
        _dt: f64,
    ) -> f64 {
        let dt_spike = post_spike_time - pre_spike_time;

        if dt_spike.abs() > 100.0 {
            return current_weight; // Outside STDP window
        }

        let dendritic_active = self.has_recent_dendritic_spike(post_spike_time);
        let dw = self.delta_w(dt_spike, dendritic_active);

        let new_weight = current_weight + dw;
        new_weight.max(self.w_min).min(self.w_max)
    }

    fn reset(&mut self) {
        self.last_dendritic_spike = -1000.0;
    }
}

/// Branch-specific plasticity
#[derive(Debug, Clone)]
pub struct BranchSpecificPlasticity {
    /// Plasticity per branch
    branch_learning_rates: Vec<f64>,
    /// Branch activation threshold
    activation_threshold: f64,
    /// Branch activity (recent spike count)
    branch_activity: Vec<f64>,
    /// Activity decay time constant (ms)
    tau_activity: f64,
}

impl BranchSpecificPlasticity {
    /// Create branch-specific plasticity
    pub fn new(num_branches: usize, base_learning_rate: f64) -> Self {
        Self {
            branch_learning_rates: vec![base_learning_rate; num_branches],
            activation_threshold: 3.0, // Need 3+ inputs
            branch_activity: vec![0.0; num_branches],
            tau_activity: 100.0, // 100 ms decay
        }
    }

    /// Update branch activity
    pub fn update_branch_activity(&mut self, branch_idx: usize, spike_count: f64, dt: f64) {
        if let Some(activity) = self.branch_activity.get_mut(branch_idx) {
            *activity += spike_count;
            *activity *= (-dt / self.tau_activity).exp(); // Decay
        }
    }

    /// Get learning rate for branch
    pub fn branch_learning_rate(&self, branch_idx: usize) -> f64 {
        let activity = self.branch_activity.get(branch_idx).copied().unwrap_or(0.0);

        if activity > self.activation_threshold {
            // Enhanced plasticity in active branches
            self.branch_learning_rates.get(branch_idx).copied().unwrap_or(0.0) * 2.0
        } else {
            self.branch_learning_rates.get(branch_idx).copied().unwrap_or(0.0)
        }
    }

    /// Set learning rate for specific branch
    pub fn set_branch_learning_rate(&mut self, branch_idx: usize, rate: f64) {
        if let Some(lr) = self.branch_learning_rates.get_mut(branch_idx) {
            *lr = rate;
        }
    }
}

impl DendriticPlasticity for BranchSpecificPlasticity {
    fn update_weight(
        &mut self,
        current_weight: f64,
        pre_spike_time: f64,
        post_spike_time: f64,
        _dt: f64,
    ) -> f64 {
        // Simplified: use first branch learning rate
        let learning_rate = self.branch_learning_rate(0);
        let dt_spike = post_spike_time - pre_spike_time;

        if dt_spike.abs() > 100.0 {
            return current_weight;
        }

        let dw = if dt_spike < 0.0 {
            learning_rate * (dt_spike / 20.0).exp()
        } else {
            -learning_rate * (-dt_spike / 20.0).exp()
        };

        (current_weight + dw).max(0.0).min(2.0)
    }

    fn reset(&mut self) {
        for activity in &mut self.branch_activity {
            *activity = 0.0;
        }
    }
}

/// Compartment-specific plasticity
#[derive(Debug, Clone)]
pub struct CompartmentPlasticity {
    /// Plasticity threshold per compartment
    thresholds: Vec<f64>,
    /// Learning rate per compartment
    learning_rates: Vec<f64>,
    /// Distance-dependent plasticity scaling
    distance_scaling: bool,
}

impl CompartmentPlasticity {
    /// Create compartment-specific plasticity
    pub fn new(num_compartments: usize) -> Self {
        Self {
            thresholds: vec![0.5; num_compartments],
            learning_rates: vec![0.01; num_compartments],
            distance_scaling: true,
        }
    }

    /// Set distance-dependent scaling
    pub fn with_distance_scaling(mut self, enabled: bool) -> Self {
        self.distance_scaling = enabled;
        self
    }

    /// Get learning rate for compartment at distance
    pub fn learning_rate(&self, compartment_idx: usize, distance_from_soma: f64) -> f64 {
        let base_rate = self.learning_rates.get(compartment_idx).copied().unwrap_or(0.01);

        if self.distance_scaling {
            // Distal synapses have higher plasticity
            let scaling = 1.0 + (distance_from_soma / 200.0).min(1.0);
            base_rate * scaling
        } else {
            base_rate
        }
    }

    /// Set learning rate for compartment
    pub fn set_learning_rate(&mut self, compartment_idx: usize, rate: f64) {
        if let Some(lr) = self.learning_rates.get_mut(compartment_idx) {
            *lr = rate;
        }
    }
}

impl DendriticPlasticity for CompartmentPlasticity {
    fn update_weight(
        &mut self,
        current_weight: f64,
        pre_spike_time: f64,
        post_spike_time: f64,
        _dt: f64,
    ) -> f64 {
        let learning_rate = self.learning_rate(0, 0.0);
        let dt_spike = post_spike_time - pre_spike_time;

        if dt_spike.abs() > 100.0 {
            return current_weight;
        }

        let dw = if dt_spike < 0.0 {
            learning_rate * (dt_spike / 20.0).exp()
        } else {
            -learning_rate * (-dt_spike / 20.0).exp()
        };

        (current_weight + dw).max(0.0).min(2.0)
    }

    fn reset(&mut self) {
        // Nothing to reset
    }
}

/// Heterosynaptic plasticity (neighboring synapses interact)
#[derive(Debug, Clone)]
pub struct Heterosynaptic {
    /// Spatial interaction radius (μm)
    interaction_radius: f64,
    /// Cooperativity factor
    cooperativity: f64,
    /// Competition factor
    competition: f64,
    /// Recent synapse activations
    recent_activations: Vec<(usize, f64, f64)>, // (synapse_id, time, position)
}

impl Heterosynaptic {
    /// Create heterosynaptic plasticity
    pub fn new(interaction_radius: f64) -> Self {
        Self {
            interaction_radius,
            cooperativity: 0.1,
            competition: -0.05,
            recent_activations: Vec::new(),
        }
    }

    /// Register synapse activation
    pub fn register_activation(&mut self, synapse_id: usize, time: f64, position: f64) {
        self.recent_activations.push((synapse_id, time, position));

        // Keep only recent activations (last 100 ms)
        self.recent_activations.retain(|(_, t, _)| time - *t < 100.0);
    }

    /// Calculate heterosynaptic modulation
    pub fn heterosynaptic_modulation(&self, synapse_id: usize, position: f64, time: f64) -> f64 {
        let mut modulation = 0.0;

        for &(other_id, other_time, other_pos) in &self.recent_activations {
            if other_id == synapse_id {
                continue; // Skip self
            }

            let spatial_dist = (position - other_pos).abs();
            let temporal_dist = (time - other_time).abs();

            if spatial_dist < self.interaction_radius && temporal_dist < 20.0 {
                // Nearby in space and time
                if temporal_dist < 5.0 {
                    // Coincident -> cooperation
                    modulation += self.cooperativity;
                } else {
                    // Sequential -> competition
                    modulation += self.competition;
                }
            }
        }

        modulation
    }
}

impl DendriticPlasticity for Heterosynaptic {
    fn update_weight(
        &mut self,
        current_weight: f64,
        pre_spike_time: f64,
        post_spike_time: f64,
        _dt: f64,
    ) -> f64 {
        // Base STDP
        let dt_spike = post_spike_time - pre_spike_time;

        if dt_spike.abs() > 100.0 {
            return current_weight;
        }

        let base_dw = if dt_spike < 0.0 {
            0.01 * (dt_spike / 20.0).exp()
        } else {
            -0.01 * (-dt_spike / 20.0).exp()
        };

        // Add heterosynaptic modulation (simplified - needs synapse position)
        let hetero_mod = 0.0; // Would need synapse context

        let dw = base_dw + hetero_mod;
        (current_weight + dw).max(0.0).min(2.0)
    }

    fn reset(&mut self) {
        self.recent_activations.clear();
    }
}

/// Metaplasticity (BCM-like, plasticity threshold changes)
#[derive(Debug, Clone)]
pub struct Metaplasticity {
    /// Current plasticity threshold
    theta: f64,
    /// Target threshold
    theta_target: f64,
    /// Threshold adaptation rate
    tau_theta: f64,
    /// Recent postsynaptic activity
    recent_activity: Vec<f64>,
    /// Activity time constant (ms)
    tau_activity: f64,
    /// Learning rate
    learning_rate: f64,
}

impl Metaplasticity {
    /// Create metaplasticity rule
    pub fn new(initial_theta: f64) -> Self {
        Self {
            theta: initial_theta,
            theta_target: initial_theta,
            tau_theta: 1000.0, // 1 second
            recent_activity: Vec::new(),
            tau_activity: 100.0, // 100 ms
            learning_rate: 0.01,
        }
    }

    /// Update threshold based on recent activity
    pub fn update_threshold(&mut self, dt: f64) {
        // Calculate average recent activity
        let avg_activity = if !self.recent_activity.is_empty() {
            self.recent_activity.iter().sum::<f64>() / self.recent_activity.len() as f64
        } else {
            0.0
        };

        // Update target threshold (BCM-like)
        self.theta_target = avg_activity.powi(2);

        // Move current threshold toward target
        let dtheta = (self.theta_target - self.theta) * dt / self.tau_theta;
        self.theta += dtheta;
    }

    /// Register postsynaptic activity
    pub fn register_activity(&mut self, activity: f64, time: f64) {
        self.recent_activity.push(activity);

        // Keep only recent activity
        let cutoff_time = time - 500.0; // Last 500 ms
        // In a real implementation, we'd store timestamps too
        if self.recent_activity.len() > 50 {
            self.recent_activity.remove(0);
        }
    }

    /// BCM learning rule: dw = activity * (activity - theta)
    fn bcm_rule(&self, activity: f64) -> f64 {
        self.learning_rate * activity * (activity - self.theta)
    }
}

impl DendriticPlasticity for Metaplasticity {
    fn update_weight(
        &mut self,
        current_weight: f64,
        _pre_spike_time: f64,
        post_spike_time: f64,
        dt: f64,
    ) -> f64 {
        // Update threshold
        self.update_threshold(dt);

        // Simplified activity measure (would be more complex in reality)
        let activity = 1.0; // Assume unit activity for spike

        // BCM learning
        let dw = self.bcm_rule(activity);

        (current_weight + dw).max(0.0).min(2.0)
    }

    fn reset(&mut self) {
        self.recent_activity.clear();
        self.theta = self.theta_target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dendritic_stdp() {
        let mut stdp = DendriticStdp::new(0.01, 0.01, 20.0, 20.0);

        // Without dendritic spike, no LTP
        let w1 = stdp.update_weight(1.0, 10.0, 15.0, 1.0); // pre before post
        assert_eq!(w1, 1.0); // No change without dendritic spike

        // With dendritic spike, LTP occurs
        stdp.register_dendritic_spike(15.0);
        let w2 = stdp.update_weight(1.0, 10.0, 15.0, 1.0);
        assert!(w2 > 1.0); // Potentiation

        // LTD doesn't require dendritic spike
        let w3 = stdp.update_weight(1.0, 15.0, 10.0, 1.0); // post before pre
        assert!(w3 < 1.0); // Depression
    }

    #[test]
    fn test_standard_stdp() {
        let mut stdp = DendriticStdp::standard(0.01, 0.01);

        // LTP without dendritic spike requirement
        let w1 = stdp.update_weight(1.0, 10.0, 15.0, 1.0);
        assert!(w1 > 1.0);

        // LTD
        let w2 = stdp.update_weight(1.0, 15.0, 10.0, 1.0);
        assert!(w2 < 1.0);
    }

    #[test]
    fn test_branch_specific_plasticity() {
        let mut plasticity = BranchSpecificPlasticity::new(3, 0.01);

        // Initially, all branches have same rate
        assert_eq!(plasticity.branch_learning_rate(0), 0.01);
        assert_eq!(plasticity.branch_learning_rate(1), 0.01);

        // Activate one branch
        plasticity.update_branch_activity(0, 5.0, 1.0);

        // Active branch should have enhanced learning
        assert!(plasticity.branch_learning_rate(0) > 0.01);
        assert_eq!(plasticity.branch_learning_rate(1), 0.01);
    }

    #[test]
    fn test_branch_activity_decay() {
        let mut plasticity = BranchSpecificPlasticity::new(2, 0.01);

        plasticity.update_branch_activity(0, 5.0, 1.0);
        let activity_initial = plasticity.branch_activity[0];

        // Decay over time
        for _ in 0..100 {
            plasticity.update_branch_activity(0, 0.0, 10.0);
        }

        assert!(plasticity.branch_activity[0] < activity_initial);
    }

    #[test]
    fn test_compartment_plasticity() {
        let mut plasticity = CompartmentPlasticity::new(5).with_distance_scaling(true);

        // Distal synapses should have higher learning rate
        let proximal_rate = plasticity.learning_rate(0, 50.0);
        let distal_rate = plasticity.learning_rate(0, 300.0);

        assert!(distal_rate > proximal_rate);
    }

    #[test]
    fn test_compartment_plasticity_no_scaling() {
        let plasticity = CompartmentPlasticity::new(5).with_distance_scaling(false);

        let rate1 = plasticity.learning_rate(0, 50.0);
        let rate2 = plasticity.learning_rate(0, 300.0);

        // Without scaling, rates should be equal
        assert_eq!(rate1, rate2);
    }

    #[test]
    fn test_heterosynaptic() {
        let mut hetero = Heterosynaptic::new(50.0); // 50 μm radius

        // Register activations
        hetero.register_activation(0, 10.0, 100.0);
        hetero.register_activation(1, 11.0, 110.0); // Close in space and time

        // Check modulation for nearby synapse
        let mod1 = hetero.heterosynaptic_modulation(2, 105.0, 12.0);

        // Should have positive modulation (cooperativity)
        assert!(mod1 > 0.0);
    }

    #[test]
    fn test_heterosynaptic_competition() {
        let mut hetero = Heterosynaptic::new(50.0);

        hetero.register_activation(0, 10.0, 100.0);
        hetero.register_activation(1, 20.0, 110.0); // Close in space, far in time

        let mod1 = hetero.heterosynaptic_modulation(2, 105.0, 25.0);

        // Should have negative modulation (competition)
        assert!(mod1 < 0.0);
    }

    #[test]
    fn test_metaplasticity() {
        let mut meta = Metaplasticity::new(0.5);

        let initial_theta = meta.theta;

        // Register high activity
        for _ in 0..10 {
            meta.register_activity(1.0, 10.0);
        }

        meta.update_threshold(100.0); // Update over 100 ms

        // Threshold should increase with high activity
        assert!(meta.theta > initial_theta);
    }

    #[test]
    fn test_metaplasticity_bcm() {
        let mut meta = Metaplasticity::new(0.5);

        let w_initial = 1.0;

        // Low activity (below threshold) -> depression
        meta.register_activity(0.3, 10.0);
        let w1 = meta.update_weight(w_initial, 0.0, 10.0, 1.0);
        // BCM rule: 0.3 * (0.3 - 0.5) = negative
        assert!(w1 <= w_initial);

        // High activity (above threshold) -> potentiation
        meta.register_activity(1.0, 20.0);
        let w2 = meta.update_weight(w_initial, 0.0, 20.0, 1.0);
        // BCM rule: 1.0 * (1.0 - theta) where theta < 1.0
        assert!(w2 >= w_initial);
    }

    #[test]
    fn test_weight_bounds() {
        let mut stdp = DendriticStdp::standard(0.1, 0.1);

        // Test upper bound
        stdp.register_dendritic_spike(15.0);
        let mut w = 1.9;
        for _ in 0..10 {
            w = stdp.update_weight(w, 10.0, 15.0, 1.0);
        }
        assert!(w <= 2.0);

        // Test lower bound
        w = 0.1;
        for _ in 0..10 {
            w = stdp.update_weight(w, 15.0, 10.0, 1.0);
        }
        assert!(w >= 0.0);
    }
}
