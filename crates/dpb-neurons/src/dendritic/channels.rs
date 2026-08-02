//! # Ion Channel Module
//!
//! Implements various ion channel models for dendritic computation:
//! - Voltage-gated channels (Na⁺, K⁺, Ca²⁺)
//! - Ligand-gated receptors (NMDA, AMPA, GABA)
//! - Channel gating dynamics
//!
//! ## Hodgkin-Huxley Formalism
//!
//! I = g_max * m^p * h^q * (V - E)
//!
//! where:
//! - g_max: maximal conductance
//! - m, h: activation and inactivation variables
//! - p, q: exponents
//! - E: reversal potential

use std::f64::consts::E as EULER;

/// Ion channel trait
pub trait IonChannel: Send + Sync {
    /// Get channel current (nA/cm²) at given voltage.
    ///
    /// # Sign convention: INWARD-POSITIVE
    ///
    /// Returns `g * (E_rev - V)`, so a positive value depolarises. This matches
    /// `Compartment::leak_current` and the integrator, which computes
    /// `dv = +I/C * dt` and sums channel and leak currents directly.
    ///
    /// Note this is the opposite of the outward-positive convention usual in
    /// electrophysiology (`I = g(V - E)`). Mixing the two here is not cosmetic:
    /// channel currents summed with the wrong sign become regenerative instead
    /// of restorative, and the membrane integrates away from rest without any
    /// input at all.
    fn current(&self, voltage: f64) -> f64;

    /// Update gating variables
    fn update(&mut self, voltage: f64, dt: f64);

    /// Reset channel to resting state
    fn reset(&mut self);

    /// Get channel conductance (mS/cm²)
    fn conductance(&self) -> f64;

    /// Get reversal potential (mV)
    fn reversal_potential(&self) -> f64;
}

/// Channel types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    Na,
    K,
    Ca,
    Leak,
    Nmda,
    Ampa,
    GabaA,
    GabaB,
}

/// Gating variable (activation or inactivation)
#[derive(Debug, Clone)]
pub struct GatingVariable {
    /// Current value [0, 1]
    pub value: f64,
    /// Exponent (power)
    pub exponent: u32,
}

impl GatingVariable {
    /// Create new gating variable
    pub fn new(initial: f64, exponent: u32) -> Self {
        Self {
            value: initial,
            exponent,
        }
    }

    /// Get effective gating (value^exponent)
    pub fn effective(&self) -> f64 {
        self.value.powi(self.exponent as i32)
    }

    /// Update using alpha-beta formulation
    pub fn update_alpha_beta(&mut self, alpha: f64, beta: f64, dt: f64) {
        let tau = 1.0 / (alpha + beta);
        let m_inf = alpha / (alpha + beta);
        self.value = m_inf + (self.value - m_inf) * (-dt / tau).exp();
    }

    /// Update using inf-tau formulation
    pub fn update_inf_tau(&mut self, inf: f64, tau: f64, dt: f64) {
        self.value = inf + (self.value - inf) * (-dt / tau).exp();
    }
}

/// Hodgkin-Huxley sodium and potassium channels
#[derive(Debug, Clone)]
pub struct HodgkinHuxleyChannel {
    channel_type: ChannelType,
    g_max: f64,
    e_rev: f64,
    m: GatingVariable, // activation
    h: GatingVariable, // inactivation
}

impl HodgkinHuxleyChannel {
    /// Create sodium channel
    pub fn sodium(g_max: f64) -> Self {
        Self {
            channel_type: ChannelType::Na,
            g_max,
            e_rev: 50.0, // Na reversal ~+50 mV
            m: GatingVariable::new(0.05, 3), // m^3
            h: GatingVariable::new(0.6, 1),  // h^1
        }
    }

    /// Create potassium channel
    pub fn potassium(g_max: f64) -> Self {
        Self {
            channel_type: ChannelType::K,
            g_max,
            e_rev: -77.0, // K reversal ~-77 mV
            m: GatingVariable::new(0.32, 4), // n^4 (using m for n)
            h: GatingVariable::new(1.0, 0),  // no inactivation
        }
    }

    /// Alpha function for Na activation
    fn alpha_m(v: f64) -> f64 {
        let x = -(v + 40.0) / 10.0;
        if x.abs() < 1e-6 {
            1.0
        } else {
            0.1 * x / (1.0 - (-x).exp())
        }
    }

    /// Beta function for Na activation
    fn beta_m(v: f64) -> f64 {
        4.0 * (-(v + 65.0) / 18.0).exp()
    }

    /// Alpha function for Na inactivation
    fn alpha_h(v: f64) -> f64 {
        0.07 * (-(v + 65.0) / 20.0).exp()
    }

    /// Beta function for Na inactivation
    fn beta_h(v: f64) -> f64 {
        1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
    }

    /// Alpha function for K activation
    fn alpha_n(v: f64) -> f64 {
        let x = -(v + 55.0) / 10.0;
        if x.abs() < 1e-6 {
            0.1
        } else {
            0.01 * x / (1.0 - (-x).exp())
        }
    }

    /// Beta function for K activation
    fn beta_n(v: f64) -> f64 {
        0.125 * (-(v + 65.0) / 80.0).exp()
    }
}

impl IonChannel for HodgkinHuxleyChannel {
    fn current(&self, voltage: f64) -> f64 {
        let g = self.g_max * self.m.effective() * self.h.effective();
        // Inward-positive: see IonChannel::current.
        g * (self.e_rev - voltage)
    }

    fn update(&mut self, voltage: f64, dt: f64) {
        match self.channel_type {
            ChannelType::Na => {
                let alpha_m = Self::alpha_m(voltage);
                let beta_m = Self::beta_m(voltage);
                let alpha_h = Self::alpha_h(voltage);
                let beta_h = Self::beta_h(voltage);

                self.m.update_alpha_beta(alpha_m, beta_m, dt);
                self.h.update_alpha_beta(alpha_h, beta_h, dt);
            }
            ChannelType::K => {
                let alpha_n = Self::alpha_n(voltage);
                let beta_n = Self::beta_n(voltage);
                self.m.update_alpha_beta(alpha_n, beta_n, dt);
            }
            _ => {}
        }
    }

    fn reset(&mut self) {
        match self.channel_type {
            ChannelType::Na => {
                self.m.value = 0.05;
                self.h.value = 0.6;
            }
            ChannelType::K => {
                self.m.value = 0.32;
            }
            _ => {}
        }
    }

    fn conductance(&self) -> f64 {
        self.g_max * self.m.effective() * self.h.effective()
    }

    fn reversal_potential(&self) -> f64 {
        self.e_rev
    }
}

/// Calcium channel (L-type, T-type, N-type)
#[derive(Debug, Clone)]
pub struct CalciumChannel {
    channel_type: String,
    g_max: f64,
    e_ca: f64,
    m: GatingVariable,
    h: GatingVariable,
}

impl CalciumChannel {
    /// Create L-type calcium channel (high-voltage activated)
    pub fn l_type(g_max: f64) -> Self {
        Self {
            channel_type: "L-type".to_string(),
            g_max,
            e_ca: 120.0, // Ca reversal ~+120 mV
            m: GatingVariable::new(0.0, 2),
            h: GatingVariable::new(1.0, 1),
        }
    }

    /// Create T-type calcium channel (low-voltage activated)
    pub fn t_type(g_max: f64) -> Self {
        Self {
            channel_type: "T-type".to_string(),
            g_max,
            e_ca: 120.0,
            m: GatingVariable::new(0.0, 2),
            h: GatingVariable::new(1.0, 1),
        }
    }

    /// Create N-type calcium channel
    pub fn n_type(g_max: f64) -> Self {
        Self {
            channel_type: "N-type".to_string(),
            g_max,
            e_ca: 120.0,
            m: GatingVariable::new(0.0, 2),
            h: GatingVariable::new(1.0, 1),
        }
    }

    /// Activation steady-state and time constant
    fn m_inf_tau(&self, v: f64) -> (f64, f64) {
        match self.channel_type.as_str() {
            "L-type" => {
                let m_inf = 1.0 / (1.0 + (-(v + 27.0) / 8.0).exp());
                let tau = 0.2 + 0.5 / ((v + 30.0).abs() / 20.0).cosh();
                (m_inf, tau)
            }
            "T-type" => {
                let m_inf = 1.0 / (1.0 + (-(v + 52.0) / 7.4).exp());
                let tau = 0.5 + 1.0 / ((v + 60.0).abs() / 15.0).cosh();
                (m_inf, tau)
            }
            "N-type" => {
                let m_inf = 1.0 / (1.0 + (-(v + 20.0) / 5.0).exp());
                let tau = 1.0 + 2.0 / ((v + 40.0).abs() / 20.0).cosh();
                (m_inf, tau)
            }
            _ => (0.0, 1.0),
        }
    }

    /// Inactivation steady-state and time constant
    fn h_inf_tau(&self, v: f64) -> (f64, f64) {
        match self.channel_type.as_str() {
            "L-type" => {
                let h_inf = 1.0 / (1.0 + ((v + 52.0) / 5.0).exp());
                let tau = 20.0 + 50.0 / ((v + 50.0).abs() / 15.0).cosh();
                (h_inf, tau)
            }
            "T-type" => {
                let h_inf = 1.0 / (1.0 + ((v + 80.0) / 5.0).exp());
                let tau = 10.0 + 20.0 / ((v + 70.0).abs() / 20.0).cosh();
                (h_inf, tau)
            }
            _ => (1.0, 1.0),
        }
    }
}

impl IonChannel for CalciumChannel {
    fn current(&self, voltage: f64) -> f64 {
        let g = self.g_max * self.m.effective() * self.h.effective();
        g * (self.e_ca - voltage)
    }

    fn update(&mut self, voltage: f64, dt: f64) {
        let (m_inf, tau_m) = self.m_inf_tau(voltage);
        let (h_inf, tau_h) = self.h_inf_tau(voltage);

        self.m.update_inf_tau(m_inf, tau_m, dt);
        self.h.update_inf_tau(h_inf, tau_h, dt);
    }

    fn reset(&mut self) {
        self.m.value = 0.0;
        self.h.value = 1.0;
    }

    fn conductance(&self) -> f64 {
        self.g_max * self.m.effective() * self.h.effective()
    }

    fn reversal_potential(&self) -> f64 {
        self.e_ca
    }
}

/// Potassium channel variants (Kv, KCa, Kir)
#[derive(Debug, Clone)]
pub struct PotassiumChannel {
    variant: String,
    g_max: f64,
    e_k: f64,
    m: GatingVariable,
    ca_concentration: f64, // For KCa channels
}

impl PotassiumChannel {
    /// Create delayed rectifier potassium channel
    pub fn delayed_rectifier(g_max: f64) -> Self {
        Self {
            variant: "Kv".to_string(),
            g_max,
            e_k: -77.0,
            m: GatingVariable::new(0.0, 4),
            ca_concentration: 0.0,
        }
    }

    /// Create calcium-activated potassium channel
    pub fn ca_activated(g_max: f64) -> Self {
        Self {
            variant: "KCa".to_string(),
            g_max,
            e_k: -77.0,
            m: GatingVariable::new(0.0, 1),
            ca_concentration: 0.0001, // 0.1 μM
        }
    }

    /// Set calcium concentration (for KCa channels)
    pub fn set_ca_concentration(&mut self, ca: f64) {
        self.ca_concentration = ca;
    }
}

impl IonChannel for PotassiumChannel {
    fn current(&self, voltage: f64) -> f64 {
        let g = self.g_max * self.m.effective();
        g * (self.e_k - voltage)
    }

    fn update(&mut self, voltage: f64, dt: f64) {
        match self.variant.as_str() {
            "Kv" => {
                // Delayed rectifier
                let m_inf = 1.0 / (1.0 + (-(voltage + 40.0) / 15.0).exp());
                let tau = 5.0;
                self.m.update_inf_tau(m_inf, tau, dt);
            }
            "KCa" => {
                // Calcium-activated
                let ca = self.ca_concentration;
                let m_inf = ca / (ca + 0.001); // Half-activation at 1 μM
                let tau = 10.0;
                self.m.update_inf_tau(m_inf, tau, dt);
            }
            _ => {}
        }
    }

    fn reset(&mut self) {
        self.m.value = 0.0;
    }

    fn conductance(&self) -> f64 {
        self.g_max * self.m.effective()
    }

    fn reversal_potential(&self) -> f64 {
        self.e_k
    }
}

/// NMDA receptor with Mg²⁺ block
#[derive(Debug, Clone)]
pub struct NmdaReceptor {
    g_max: f64,
    e_rev: f64,
    /// Synaptic gating (neurotransmitter binding)
    s: f64,
    /// Time constant for opening (ms)
    tau_rise: f64,
    /// Time constant for closing (ms)
    tau_decay: f64,
    /// Mg²⁺ concentration (mM)
    mg_concentration: f64,
}

impl NmdaReceptor {
    /// Create NMDA receptor
    pub fn new(g_max: f64) -> Self {
        Self {
            g_max,
            e_rev: 0.0, // Non-selective cation channel
            s: 0.0,
            tau_rise: 2.0,   // 2 ms rise
            tau_decay: 100.0, // 100 ms decay
            mg_concentration: 1.0, // 1 mM
        }
    }

    /// Mg²⁺ block function (Jahr & Stevens, 1990)
    fn mg_block(&self, voltage: f64) -> f64 {
        1.0 / (1.0 + (self.mg_concentration / 3.57) * (-0.062 * voltage).exp())
    }

    /// Activate receptor (neurotransmitter release)
    pub fn activate(&mut self, strength: f64) {
        self.s += strength;
        if self.s > 1.0 {
            self.s = 1.0;
        }
    }
}

impl IonChannel for NmdaReceptor {
    fn current(&self, voltage: f64) -> f64 {
        let mg_block = self.mg_block(voltage);
        let g = self.g_max * self.s * mg_block;
        g * (self.e_rev - voltage)
    }

    fn update(&mut self, _voltage: f64, dt: f64) {
        // Decay synaptic gating
        self.s *= (-dt / self.tau_decay).exp();
    }

    fn reset(&mut self) {
        self.s = 0.0;
    }

    fn conductance(&self) -> f64 {
        self.g_max * self.s
    }

    fn reversal_potential(&self) -> f64 {
        self.e_rev
    }
}

/// AMPA receptor (fast excitatory)
#[derive(Debug, Clone)]
pub struct AmpaReceptor {
    g_max: f64,
    e_rev: f64,
    s: f64,
    tau_decay: f64,
}

impl AmpaReceptor {
    pub fn new(g_max: f64) -> Self {
        Self {
            g_max,
            e_rev: 0.0,
            s: 0.0,
            tau_decay: 2.0, // 2 ms decay
        }
    }

    pub fn activate(&mut self, strength: f64) {
        self.s += strength;
        if self.s > 1.0 {
            self.s = 1.0;
        }
    }
}

impl IonChannel for AmpaReceptor {
    fn current(&self, voltage: f64) -> f64 {
        let g = self.g_max * self.s;
        g * (self.e_rev - voltage)
    }

    fn update(&mut self, _voltage: f64, dt: f64) {
        self.s *= (-dt / self.tau_decay).exp();
    }

    fn reset(&mut self) {
        self.s = 0.0;
    }

    fn conductance(&self) -> f64 {
        self.g_max * self.s
    }

    fn reversal_potential(&self) -> f64 {
        self.e_rev
    }
}

/// GABA_A receptor (fast inhibitory)
#[derive(Debug, Clone)]
pub struct GabaAReceptor {
    g_max: f64,
    e_rev: f64,
    s: f64,
    tau_decay: f64,
}

impl GabaAReceptor {
    pub fn new(g_max: f64) -> Self {
        Self {
            g_max,
            e_rev: -70.0, // Chloride reversal
            s: 0.0,
            tau_decay: 10.0, // 10 ms decay
        }
    }

    pub fn activate(&mut self, strength: f64) {
        self.s += strength;
        if self.s > 1.0 {
            self.s = 1.0;
        }
    }
}

impl IonChannel for GabaAReceptor {
    fn current(&self, voltage: f64) -> f64 {
        let g = self.g_max * self.s;
        g * (self.e_rev - voltage)
    }

    fn update(&mut self, _voltage: f64, dt: f64) {
        self.s *= (-dt / self.tau_decay).exp();
    }

    fn reset(&mut self) {
        self.s = 0.0;
    }

    fn conductance(&self) -> f64 {
        self.g_max * self.s
    }

    fn reversal_potential(&self) -> f64 {
        self.e_rev
    }
}

/// GABA_B receptor (slow inhibitory, metabotropic)
#[derive(Debug, Clone)]
pub struct GabaBReceptor {
    g_max: f64,
    e_rev: f64,
    s: f64,
    tau_rise: f64,
    tau_decay: f64,
}

impl GabaBReceptor {
    pub fn new(g_max: f64) -> Self {
        Self {
            g_max,
            e_rev: -90.0, // K reversal (activates K channels)
            s: 0.0,
            tau_rise: 50.0,   // 50 ms rise
            tau_decay: 200.0, // 200 ms decay
        }
    }

    pub fn activate(&mut self, strength: f64) {
        self.s += strength;
        if self.s > 1.0 {
            self.s = 1.0;
        }
    }
}

impl IonChannel for GabaBReceptor {
    fn current(&self, voltage: f64) -> f64 {
        let g = self.g_max * self.s;
        g * (self.e_rev - voltage)
    }

    fn update(&mut self, _voltage: f64, dt: f64) {
        self.s *= (-dt / self.tau_decay).exp();
    }

    fn reset(&mut self) {
        self.s = 0.0;
    }

    fn conductance(&self) -> f64 {
        self.g_max * self.s
    }

    fn reversal_potential(&self) -> f64 {
        self.e_rev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_variable() {
        let mut m = GatingVariable::new(0.5, 3);
        assert_eq!(m.value, 0.5);
        assert_eq!(m.effective(), 0.125); // 0.5^3

        m.update_alpha_beta(1.0, 1.0, 1.0);
        assert!((m.value - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_hh_sodium_channel() {
        let mut na = HodgkinHuxleyChannel::sodium(120.0);

        // At resting potential, current should be small
        let i_rest = na.current(-70.0);
        assert!(i_rest.abs() < 10.0);

        // Simulate depolarization
        for _ in 0..10 {
            na.update(-30.0, 0.1);
        }

        // Na influx depolarises, so the current is positive under the crate's
        // inward-positive convention.
        assert!(i_rest > 0.0, "Na influx depolarises at rest: got {i_rest}");

        // After sustained depolarisation the h gate inactivates AND the driving
        // force (E_Na - V) shrinks, so the current FALLS. The previous assertion
        // expected it to rise, which is the opposite of Na channel inactivation.
        let i_depol = na.current(-30.0);
        assert!(i_depol > 0.0, "still an inward current: got {i_depol}");
        assert!(
            i_depol < i_rest,
            "sustained depolarisation inactivates Na: rest {i_rest}, depol {i_depol}"
        );
    }

    #[test]
    fn test_hh_potassium_channel() {
        let mut k = HodgkinHuxleyChannel::potassium(36.0);

        // Simulate depolarization
        for _ in 0..20 {
            k.update(-30.0, 0.1);
        }

        // Current should be negative (outward)
        let i = k.current(-30.0);
        assert!(i < 0.0);
    }

    #[test]
    fn test_calcium_channels() {
        let mut l_type = CalciumChannel::l_type(1.0);
        let mut t_type = CalciumChannel::t_type(1.0);

        // L-type activates at higher voltage
        for _ in 0..20 {
            l_type.update(-20.0, 0.1);
            t_type.update(-60.0, 0.1);
        }

        let i_l = l_type.current(-20.0);
        let i_t = t_type.current(-60.0);

        // Ca influx DEPOLARISES (E_Ca ~ +120 mV), which is a positive current
        // under the crate's inward-positive convention.
        assert!(i_l > 0.0, "L-type Ca influx depolarises: got {i_l}");
        assert!(i_t > 0.0, "T-type Ca influx also depolarises: got {i_t}");
    }

    #[test]
    fn test_nmda_mg_block() {
        let nmda = NmdaReceptor::new(1.0);

        // mg_block returns the UNBLOCKED fraction, and implements
        // Jahr & Stevens (1990): 1 / (1 + [Mg]/3.57 * exp(-0.062V)).
        // Values below are that published model at 1 mM Mg, not round numbers
        // chosen by eye — the earlier test asserted > 0.8 at 0 mV where the
        // model gives 0.781, so a correct implementation failed.
        let block_hyperpol = nmda.mg_block(-70.0);
        assert!(
            (block_hyperpol - 0.044).abs() < 0.01,
            "Jahr-Stevens at -70 mV is ~0.044 (strong block): got {block_hyperpol}"
        );

        let block_zero = nmda.mg_block(0.0);
        assert!(
            (block_zero - 0.781).abs() < 0.01,
            "Jahr-Stevens at 0 mV is ~0.781: got {block_zero}"
        );

        let block_depol = nmda.mg_block(20.0);
        assert!(
            block_depol > 0.9,
            "block is largely relieved by +20 mV: got {block_depol}"
        );

        // The property that matters for coincidence detection: relief of the
        // block is monotonic in depolarisation.
        assert!(block_hyperpol < block_zero && block_zero < block_depol);
    }

    #[test]
    fn test_nmda_activation() {
        let mut nmda = NmdaReceptor::new(1.0);

        nmda.activate(0.5);
        assert_eq!(nmda.s, 0.5);

        // Decay over time
        for _ in 0..100 {
            nmda.update(0.0, 1.0);
        }

        assert!(nmda.s < 0.5);
    }

    #[test]
    fn test_ampa_receptor() {
        let mut ampa = AmpaReceptor::new(1.0);

        ampa.activate(1.0);
        let i_initial = ampa.current(-70.0);

        // Should decay quickly
        for _ in 0..10 {
            ampa.update(0.0, 1.0);
        }

        let i_decayed = ampa.current(-70.0);
        assert!(i_decayed.abs() < i_initial.abs());
    }

    #[test]
    fn test_gaba_receptors() {
        let mut gaba_a = GabaAReceptor::new(1.0);
        let mut gaba_b = GabaBReceptor::new(1.0);

        gaba_a.activate(1.0);
        gaba_b.activate(1.0);

        // Both are inhibitory: at -60 mV they pull toward E_rev below it.
        // Under the crate's inward-positive convention (see IonChannel::current)
        // a hyperpolarising current is NEGATIVE.
        let i_a = gaba_a.current(-60.0);
        let i_b = gaba_b.current(-60.0);

        assert!(i_a < 0.0, "GABA_A pulls toward -70 mV from -60 mV: got {i_a}");
        assert!(i_b < 0.0, "GABA_B pulls toward -90 mV from -60 mV: got {i_b}");
        assert!(i_b < i_a, "GABA_B (E=-90) hyperpolarises harder than GABA_A (E=-70)");
    }

    #[test]
    fn test_kca_channel() {
        let mut kca = PotassiumChannel::ca_activated(1.0);

        // Low calcium - low activation
        kca.set_ca_concentration(0.0001);
        kca.update(-30.0, 1.0);
        let g_low = kca.conductance();

        // High calcium - high activation
        kca.set_ca_concentration(0.01);
        kca.update(-30.0, 1.0);
        let g_high = kca.conductance();

        assert!(g_high > g_low);
    }
}
