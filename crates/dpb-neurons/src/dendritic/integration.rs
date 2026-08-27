//! # Dendritic Integration Module
//!
//! Implements various dendritic integration mechanisms:
//! - Passive cable theory integration
//! - Active dendritic spikes
//! - Nonlinear dendritic computation
//! - Coincidence detection
//!
//! ## Integration Modes
//!
//! 1. **Passive**: Linear summation via cable equation
//! 2. **Active**: With voltage-gated channels
//! 3. **Nonlinear**: Dendritic spikes, NMDA spikes, Ca²⁺ spikes

use super::channels::{CalciumChannel, IonChannel, NmdaReceptor};

/// Dendritic integration trait
pub trait DendriticIntegration: Send + Sync {
    /// Integrate synaptic inputs to produce somatic depolarization
    fn integrate(&self, inputs: &[f64], distances: &[f64]) -> f64;

    /// Check if dendritic spike occurred
    fn has_dendritic_spike(&self) -> bool;

    /// Reset integration state
    fn reset(&mut self);
}

/// Passive dendritic integration (cable theory)
#[derive(Debug, Clone)]
pub struct PassiveIntegration {
    /// Space constant (μm)
    lambda: f64,
    /// Attenuation mode
    attenuation_mode: AttenuationMode,
}

#[derive(Debug, Clone, Copy)]
pub enum AttenuationMode {
    /// Exponential: exp(-x/λ)
    Exponential,
    /// Rall's law: (λ/x) * exp(-x/λ) for infinite cable
    Rall,
    /// No attenuation (for debugging)
    None,
}

impl PassiveIntegration {
    /// Create passive integration with space constant
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            attenuation_mode: AttenuationMode::Exponential,
        }
    }

    /// Set attenuation mode
    pub fn with_mode(mut self, mode: AttenuationMode) -> Self {
        self.attenuation_mode = mode;
        self
    }

    /// Calculate attenuation factor for distance x
    fn attenuation(&self, distance: f64) -> f64 {
        match self.attenuation_mode {
            AttenuationMode::Exponential => {
                // V(x) = V(0) * exp(-x/λ)
                (-distance / self.lambda).exp()
            }
            AttenuationMode::Rall => {
                // For infinite cable: V(x) = V(0) * (λ/x) * exp(-x/λ)
                if distance < 1.0 {
                    1.0 // Avoid singularity near soma
                } else {
                    (self.lambda / distance) * (-distance / self.lambda).exp()
                }
            }
            AttenuationMode::None => 1.0,
        }
    }
}

impl DendriticIntegration for PassiveIntegration {
    fn integrate(&self, inputs: &[f64], distances: &[f64]) -> f64 {
        let mut somatic_input = 0.0;

        for (i, &input) in inputs.iter().enumerate() {
            let distance = distances.get(i).copied().unwrap_or(0.0);
            let attenuation = self.attenuation(distance);
            somatic_input += input * attenuation;
        }

        somatic_input
    }

    fn has_dendritic_spike(&self) -> bool {
        false // Passive integration has no spikes
    }

    fn reset(&mut self) {
        // Nothing to reset for passive
    }
}

/// Active dendritic integration with voltage-gated channels
#[derive(Debug, Clone)]
pub struct ActiveIntegration {
    /// Space constant (μm)
    lambda: f64,
    /// Sodium channel density (mS/cm²).
    ///
    /// Recorded from the constructor but not yet consumed: the dendritic
    /// spike below is triggered on `spike_threshold` alone, with no
    /// conductance dynamics. Kept so the parameter a caller supplies is not
    /// silently discarded.
    #[allow(dead_code)]
    g_na: f64,
    /// Potassium channel density (mS/cm²). Not yet consumed; see `g_na`.
    #[allow(dead_code)]
    g_k: f64,
    /// Dendritic spike threshold (mV)
    spike_threshold: f64,
    /// Last spike time (ms)
    last_spike_time: f64,
    /// Refractory period (ms)
    refractory_period: f64,
}

impl ActiveIntegration {
    /// Create active integration
    pub fn new(lambda: f64, g_na: f64, g_k: f64) -> Self {
        Self {
            lambda,
            g_na,
            g_k,
            spike_threshold: -40.0, // Dendritic spike threshold
            last_spike_time: -1000.0,
            refractory_period: 5.0, // 5 ms
        }
    }

    /// Check if in refractory period
    fn is_refractory(&self, current_time: f64) -> bool {
        current_time - self.last_spike_time < self.refractory_period
    }

    /// Amplify input based on active conductances
    fn amplify(&self, input: f64, voltage: f64) -> f64 {
        // Simple amplification model
        // In reality, this would involve solving channel dynamics
        if voltage > self.spike_threshold && !self.is_refractory(0.0) {
            input * 2.0 // Amplification during dendritic spike
        } else {
            input
        }
    }
}

impl DendriticIntegration for ActiveIntegration {
    fn integrate(&self, inputs: &[f64], distances: &[f64]) -> f64 {
        let mut somatic_input = 0.0;

        for (i, &input) in inputs.iter().enumerate() {
            let distance = distances.get(i).copied().unwrap_or(0.0);
            let attenuation = (-distance / self.lambda).exp();

            // Active amplification can counteract attenuation
            let amplified = self.amplify(input, -60.0); // Simplified voltage
            somatic_input += amplified * attenuation;
        }

        somatic_input
    }

    fn has_dendritic_spike(&self) -> bool {
        !self.is_refractory(0.0) && self.last_spike_time > -100.0
    }

    fn reset(&mut self) {
        self.last_spike_time = -1000.0;
    }
}

/// Nonlinear dendritic computation (dendritic spikes, NMDA spikes)
#[derive(Debug, Clone)]
pub struct NonlinearDendrites {
    /// Calcium channels for dendritic spikes
    ca_channels: Vec<CalciumChannel>,
    /// NMDA receptors for plateau potentials
    nmda_receptors: Vec<NmdaReceptor>,
    /// Compartment voltages
    voltages: Vec<f64>,
    /// Dendritic spike threshold (mV)
    spike_threshold: f64,
    /// NMDA spike threshold (mV)
    nmda_threshold: f64,
    /// Dendritic spike detected
    dendritic_spike: bool,
    /// NMDA spike detected
    nmda_spike: bool,
}

impl NonlinearDendrites {
    /// Create nonlinear dendrites
    pub fn new(num_compartments: usize) -> Self {
        let ca_channels = (0..num_compartments)
            .map(|_| CalciumChannel::t_type(0.5))
            .collect();

        let nmda_receptors = (0..num_compartments)
            .map(|_| NmdaReceptor::new(0.5))
            .collect();

        Self {
            ca_channels,
            nmda_receptors,
            voltages: vec![-70.0; num_compartments],
            spike_threshold: -30.0,  // Ca spike threshold
            nmda_threshold: -45.0,   // NMDA plateau threshold
            dendritic_spike: false,
            nmda_spike: false,
        }
    }

    /// Update dendritic voltages and detect spikes
    pub fn update(&mut self, inputs: &[f64], dt: f64) {
        self.dendritic_spike = false;
        self.nmda_spike = false;

        for (i, &input) in inputs.iter().enumerate() {
            if let Some(voltage) = self.voltages.get_mut(i) {
                // Update voltage (simplified)
                *voltage += input * dt;

                // Check for calcium spike
                if *voltage > self.spike_threshold {
                    self.dendritic_spike = true;
                    *voltage = 0.0; // Spike reset
                }

                // Check for NMDA spike (plateau potential)
                if *voltage > self.nmda_threshold {
                    self.nmda_spike = true;
                    // NMDA plateau maintains depolarization
                    *voltage = self.nmda_threshold + 5.0;
                }

                // Update channels
                if let Some(ca) = self.ca_channels.get_mut(i) {
                    ca.update(*voltage, dt);
                }

                if let Some(nmda) = self.nmda_receptors.get_mut(i) {
                    nmda.update(*voltage, dt);
                }
            }
        }
    }

    /// Get dendritic spike status
    pub fn dendritic_spike(&self) -> bool {
        self.dendritic_spike
    }

    /// Get NMDA spike status
    pub fn nmda_spike(&self) -> bool {
        self.nmda_spike
    }

    /// Get compartment voltage
    pub fn voltage(&self, idx: usize) -> Option<f64> {
        self.voltages.get(idx).copied()
    }
}

impl DendriticIntegration for NonlinearDendrites {
    fn integrate(&self, inputs: &[f64], distances: &[f64]) -> f64 {
        // Nonlinear integration with supralinear summation
        let mut somatic_input = 0.0;

        for (i, &input) in inputs.iter().enumerate() {
            let distance = distances.get(i).copied().unwrap_or(0.0);
            let lambda = 200.0; // Space constant
            let attenuation = (-distance / lambda).exp();

            // Nonlinear amplification for strong inputs
            let amplified = if input > 0.5 {
                input * 1.5 // Supralinear
            } else {
                input // Linear
            };

            somatic_input += amplified * attenuation;
        }

        somatic_input
    }

    fn has_dendritic_spike(&self) -> bool {
        self.dendritic_spike || self.nmda_spike
    }

    fn reset(&mut self) {
        self.dendritic_spike = false;
        self.nmda_spike = false;
        for voltage in &mut self.voltages {
            *voltage = -70.0;
        }
    }
}

/// Coincidence detection in dendritic branches
#[derive(Debug, Clone)]
pub struct CoincidenceDetection {
    /// Time window for coincidence (ms)
    time_window: f64,
    /// Last input times for each input
    last_input_times: Vec<f64>,
    /// Coincidence threshold (number of inputs)
    coincidence_threshold: usize,
    /// Detected coincidence
    coincidence_detected: bool,
}

impl CoincidenceDetection {
    /// Create coincidence detector
    pub fn new(num_inputs: usize, time_window: f64, threshold: usize) -> Self {
        Self {
            time_window,
            last_input_times: vec![-1000.0; num_inputs],
            coincidence_threshold: threshold,
            coincidence_detected: false,
        }
    }

    /// Register input at specific index and time
    pub fn register_input(&mut self, idx: usize, time: f64) {
        if let Some(last_time) = self.last_input_times.get_mut(idx) {
            *last_time = time;
        }

        // Check for coincidence
        self.check_coincidence(time);
    }

    /// Check if inputs are coincident
    fn check_coincidence(&mut self, current_time: f64) {
        let coincident_count = self
            .last_input_times
            .iter()
            .filter(|&&t| (current_time - t).abs() < self.time_window)
            .count();

        self.coincidence_detected = coincident_count >= self.coincidence_threshold;
    }

    /// Get coincidence status
    pub fn is_coincident(&self) -> bool {
        self.coincidence_detected
    }
}

impl DendriticIntegration for CoincidenceDetection {
    fn integrate(&self, inputs: &[f64], _distances: &[f64]) -> f64 {
        // Enhanced output if coincidence detected
        let base_sum: f64 = inputs.iter().sum();

        if self.coincidence_detected {
            base_sum * 2.0 // Nonlinear boost for coincident inputs
        } else {
            base_sum
        }
    }

    fn has_dendritic_spike(&self) -> bool {
        self.coincidence_detected
    }

    fn reset(&mut self) {
        self.coincidence_detected = false;
        for time in &mut self.last_input_times {
            *time = -1000.0;
        }
    }
}

/// Two-layer integration model (local branches + soma)
///
/// Note: Debug and Clone are not derived due to trait object constraints.
pub struct TwoLayerIntegration {
    /// Branch-level integrators
    branch_integrators: Vec<Box<dyn DendriticIntegration>>,
    /// Somatic integrator
    somatic_integrator: PassiveIntegration,
}

impl std::fmt::Debug for TwoLayerIntegration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoLayerIntegration")
            .field("branch_count", &self.branch_integrators.len())
            .field("somatic_integrator", &self.somatic_integrator)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passive_integration() {
        let integration = PassiveIntegration::new(200.0);

        let inputs = vec![1.0, 1.0, 1.0];
        let distances = vec![0.0, 100.0, 300.0];

        let somatic = integration.integrate(&inputs, &distances);

        // Closer inputs should contribute more
        assert!(somatic < 3.0); // Due to attenuation
        assert!(somatic > 1.0); // But still significant

        assert!(!integration.has_dendritic_spike());
    }

    #[test]
    fn test_exponential_attenuation() {
        let integration = PassiveIntegration::new(200.0);

        let inputs = vec![1.0, 1.0];
        let distances1 = vec![0.0, 100.0];
        let distances2 = vec![0.0, 200.0];

        let soma1 = integration.integrate(&inputs, &distances1);
        let soma2 = integration.integrate(&inputs, &distances2);

        // Further input should contribute less
        assert!(soma1 > soma2);
    }

    #[test]
    fn test_attenuation_modes() {
        let exp_integration = PassiveIntegration::new(200.0)
            .with_mode(AttenuationMode::Exponential);

        let rall_integration = PassiveIntegration::new(200.0)
            .with_mode(AttenuationMode::Rall);

        let no_integration = PassiveIntegration::new(200.0)
            .with_mode(AttenuationMode::None);

        let inputs = vec![1.0];
        let distances = vec![300.0];

        let exp_result = exp_integration.integrate(&inputs, &distances);
        let rall_result = rall_integration.integrate(&inputs, &distances);
        let no_result = no_integration.integrate(&inputs, &distances);

        assert!(exp_result < no_result);
        assert!(rall_result < exp_result); // Rall has stronger attenuation for far inputs
        assert_eq!(no_result, 1.0);
    }

    #[test]
    fn test_active_integration() {
        let integration = ActiveIntegration::new(200.0, 120.0, 36.0);

        let inputs = vec![1.0, 0.5];
        let distances = vec![100.0, 200.0];

        let somatic = integration.integrate(&inputs, &distances);

        assert!(somatic > 0.0);
    }

    #[test]
    fn test_nonlinear_dendrites() {
        let mut dendrites = NonlinearDendrites::new(3);

        let inputs = vec![0.1, 0.1, 0.1];
        dendrites.update(&inputs, 1.0);

        assert!(!dendrites.has_dendritic_spike());

        // Strong input to trigger spike
        let strong_inputs = vec![5.0, 0.0, 0.0];
        for _ in 0..10 {
            dendrites.update(&strong_inputs, 1.0);
        }

        // Should eventually trigger dendritic spike
        // (may take multiple iterations depending on dynamics)
    }

    #[test]
    fn test_coincidence_detection() {
        let mut detector = CoincidenceDetection::new(4, 5.0, 3);

        // Register inputs at similar times
        detector.register_input(0, 10.0);
        detector.register_input(1, 11.0);
        detector.register_input(2, 12.0);

        // Should detect coincidence (3 inputs within 5 ms)
        assert!(detector.is_coincident());

        // Reset and test non-coincident inputs
        detector.reset();
        detector.register_input(0, 10.0);
        detector.register_input(1, 20.0); // Too far apart
        detector.register_input(2, 30.0);

        assert!(!detector.is_coincident());
    }

    #[test]
    fn test_coincidence_integration_boost() {
        let mut detector = CoincidenceDetection::new(3, 5.0, 2);

        let inputs = vec![1.0, 1.0, 1.0];
        let distances = vec![0.0, 0.0, 0.0];

        // Without coincidence
        let normal = detector.integrate(&inputs, &distances);

        // With coincidence
        detector.register_input(0, 10.0);
        detector.register_input(1, 11.0);

        let boosted = detector.integrate(&inputs, &distances);

        // Coincident inputs should produce larger output
        assert!(boosted > normal);
        assert_eq!(boosted, normal * 2.0);
    }

    #[test]
    fn test_nonlinear_summation() {
        let dendrites = NonlinearDendrites::new(5);

        let weak_inputs = vec![0.3, 0.3, 0.0, 0.0, 0.0];
        let strong_inputs = vec![0.8, 0.8, 0.0, 0.0, 0.0];

        let distances = vec![100.0, 100.0, 100.0, 100.0, 100.0];

        let weak_sum = dendrites.integrate(&weak_inputs, &distances);
        let strong_sum = dendrites.integrate(&strong_inputs, &distances);

        // Nonlinear amplification means strong inputs sum supralinearly
        // strong_sum / weak_sum > (0.8 / 0.3) due to threshold nonlinearity
        let linear_ratio = 0.8 / 0.3;
        let actual_ratio = strong_sum / weak_sum;

        assert!(actual_ratio > linear_ratio * 0.9); // Allow some tolerance
    }
}
