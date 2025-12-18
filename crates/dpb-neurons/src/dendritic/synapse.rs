//! # Dendritic Synapse Module
//!
//! Implements synapses located on dendritic compartments with:
//! - Synaptic placement on dendritic tree
//! - Conductance-based synaptic currents
//! - EPSP/IPSP generation
//! - Temporal and spatial summation
//!
//! ## Synaptic Current
//!
//! I_syn = g_syn(t) * (V - E_syn)
//!
//! where g_syn follows alpha function or bi-exponential

use super::channels::{AmpaReceptor, GabaAReceptor, GabaBReceptor, NmdaReceptor};

/// Synapse type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynapseType {
    Excitatory,  // AMPA/NMDA
    Inhibitory,  // GABA
    Modulatory,  // Neuromodulatory
}

/// Synaptic conductance model
#[derive(Debug, Clone)]
pub enum SynapticConductance {
    /// Alpha function: g(t) = g_max * (t/tau) * exp(1 - t/tau)
    Alpha {
        g_max: f64,
        tau: f64,
        time_since_spike: f64,
    },
    /// Bi-exponential: g(t) = g_max * (exp(-t/tau_decay) - exp(-t/tau_rise))
    BiExponential {
        g_max: f64,
        tau_rise: f64,
        tau_decay: f64,
        time_since_spike: f64,
    },
    /// Dual receptor (AMPA + NMDA for excitatory)
    DualReceptor {
        ampa: AmpaReceptor,
        nmda: NmdaReceptor,
    },
    /// GABA receptors
    GabaReceptors {
        gaba_a: GabaAReceptor,
        gaba_b: Option<GabaBReceptor>,
    },
}

impl SynapticConductance {
    /// Create alpha function conductance
    pub fn alpha(g_max: f64, tau: f64) -> Self {
        Self::Alpha {
            g_max,
            tau,
            time_since_spike: -1000.0, // Long time ago
        }
    }

    /// Create bi-exponential conductance
    pub fn bi_exponential(g_max: f64, tau_rise: f64, tau_decay: f64) -> Self {
        Self::BiExponential {
            g_max,
            tau_rise,
            tau_decay,
            time_since_spike: -1000.0,
        }
    }

    /// Create AMPA+NMDA dual receptor
    pub fn ampa_nmda(g_ampa: f64, g_nmda: f64) -> Self {
        Self::DualReceptor {
            ampa: AmpaReceptor::new(g_ampa),
            nmda: NmdaReceptor::new(g_nmda),
        }
    }

    /// Create GABA receptors
    pub fn gaba(g_gaba_a: f64, g_gaba_b: Option<f64>) -> Self {
        Self::GabaReceptors {
            gaba_a: GabaAReceptor::new(g_gaba_a),
            gaba_b: g_gaba_b.map(GabaBReceptor::new),
        }
    }

    /// Get current conductance value
    pub fn conductance(&self, voltage: f64) -> f64 {
        use std::f64::consts::E as EULER;

        match self {
            Self::Alpha { g_max, tau, time_since_spike } => {
                if *time_since_spike < 0.0 {
                    0.0
                } else {
                    let t = time_since_spike;
                    g_max * (t / tau) * (EULER.powf(1.0 - t / tau))
                }
            }
            Self::BiExponential {
                g_max,
                tau_rise,
                tau_decay,
                time_since_spike,
            } => {
                if *time_since_spike < 0.0 {
                    0.0
                } else {
                    let t = time_since_spike;
                    let norm = 1.0 / ((-tau_rise / tau_decay).exp() - (-tau_decay / tau_rise).exp());
                    g_max * norm * ((-t / tau_decay).exp() - (-t / tau_rise).exp())
                }
            }
            Self::DualReceptor { ampa, nmda } => {
                use super::channels::IonChannel;
                ampa.conductance() + nmda.conductance()
            }
            Self::GabaReceptors { gaba_a, gaba_b } => {
                use super::channels::IonChannel;
                let g_a = gaba_a.conductance();
                let g_b = gaba_b.as_ref().map_or(0.0, |g| g.conductance());
                g_a + g_b
            }
        }
    }

    /// Update conductance dynamics
    pub fn update(&mut self, dt: f64, voltage: f64) {
        match self {
            Self::Alpha { time_since_spike, .. } => {
                if *time_since_spike >= 0.0 {
                    *time_since_spike += dt;
                }
            }
            Self::BiExponential { time_since_spike, .. } => {
                if *time_since_spike >= 0.0 {
                    *time_since_spike += dt;
                }
            }
            Self::DualReceptor { ampa, nmda } => {
                use super::channels::IonChannel;
                ampa.update(voltage, dt);
                nmda.update(voltage, dt);
            }
            Self::GabaReceptors { gaba_a, gaba_b } => {
                use super::channels::IonChannel;
                gaba_a.update(voltage, dt);
                if let Some(g) = gaba_b {
                    g.update(voltage, dt);
                }
            }
        }
    }

    /// Trigger synaptic event
    pub fn trigger(&mut self, weight: f64) {
        match self {
            Self::Alpha { time_since_spike, .. } => {
                *time_since_spike = 0.0;
            }
            Self::BiExponential { time_since_spike, .. } => {
                *time_since_spike = 0.0;
            }
            Self::DualReceptor { ampa, nmda } => {
                ampa.activate(weight);
                nmda.activate(weight);
            }
            Self::GabaReceptors { gaba_a, gaba_b } => {
                gaba_a.activate(weight);
                if let Some(g) = gaba_b {
                    g.activate(weight);
                }
            }
        }
    }

    /// Reset synapse to baseline
    pub fn reset(&mut self) {
        match self {
            Self::Alpha { time_since_spike, .. } => {
                *time_since_spike = -1000.0;
            }
            Self::BiExponential { time_since_spike, .. } => {
                *time_since_spike = -1000.0;
            }
            Self::DualReceptor { ampa, nmda } => {
                use super::channels::IonChannel;
                ampa.reset();
                nmda.reset();
            }
            Self::GabaReceptors { gaba_a, gaba_b } => {
                use super::channels::IonChannel;
                gaba_a.reset();
                if let Some(g) = gaba_b {
                    g.reset();
                }
            }
        }
    }
}

/// Configuration for a dendritic synapse
#[derive(Debug, Clone)]
pub struct SynapseConfig {
    /// Compartment index where synapse is located
    pub compartment_idx: usize,
    /// Synapse type
    pub synapse_type: SynapseType,
    /// Maximum conductance (mS/cm²)
    pub g_max: f64,
    /// Reversal potential (mV)
    pub e_rev: f64,
    /// Initial synaptic weight
    pub weight: f64,
    /// Distance from soma (μm)
    pub distance_from_soma: f64,
}

impl SynapseConfig {
    /// Create excitatory synapse configuration
    pub fn excitatory(compartment_idx: usize, g_max: f64) -> Self {
        Self {
            compartment_idx,
            synapse_type: SynapseType::Excitatory,
            g_max,
            e_rev: 0.0, // Non-selective cation channel
            weight: 1.0,
            distance_from_soma: 0.0,
        }
    }

    /// Create inhibitory synapse configuration
    pub fn inhibitory(compartment_idx: usize, g_max: f64) -> Self {
        Self {
            compartment_idx,
            synapse_type: SynapseType::Inhibitory,
            g_max,
            e_rev: -70.0, // Chloride reversal
            weight: 1.0,
            distance_from_soma: 0.0,
        }
    }
}

/// Dendritic synapse
#[derive(Debug, Clone)]
pub struct DendriticSynapse {
    /// Synapse configuration
    config: SynapseConfig,
    /// Synaptic conductance model
    conductance: SynapticConductance,
    /// Last spike time (ms)
    last_spike_time: f64,
    /// Total number of spikes received
    spike_count: usize,
    /// Running average of inter-spike interval (ms)
    avg_isi: f64,
}

impl DendriticSynapse {
    /// Create a new dendritic synapse
    pub fn new(config: SynapseConfig, conductance: SynapticConductance) -> Self {
        Self {
            config,
            conductance,
            last_spike_time: -1000.0,
            spike_count: 0,
            avg_isi: 0.0,
        }
    }

    /// Create excitatory synapse with AMPA+NMDA
    pub fn excitatory(compartment_idx: usize, g_ampa: f64, g_nmda: f64) -> Self {
        let config = SynapseConfig::excitatory(compartment_idx, g_ampa + g_nmda);
        let conductance = SynapticConductance::ampa_nmda(g_ampa, g_nmda);
        Self::new(config, conductance)
    }

    /// Create inhibitory synapse with GABA
    pub fn inhibitory(compartment_idx: usize, g_gaba: f64) -> Self {
        let config = SynapseConfig::inhibitory(compartment_idx, g_gaba);
        let conductance = SynapticConductance::gaba(g_gaba, None);
        Self::new(config, conductance)
    }

    /// Get compartment index
    pub fn compartment_idx(&self) -> usize {
        self.config.compartment_idx
    }

    /// Get synapse type
    pub fn synapse_type(&self) -> SynapseType {
        self.config.synapse_type
    }

    /// Get synaptic weight
    pub fn weight(&self) -> f64 {
        self.config.weight
    }

    /// Set synaptic weight
    pub fn set_weight(&mut self, weight: f64) {
        self.config.weight = weight;
    }

    /// Modify weight by delta
    pub fn modify_weight(&mut self, delta: f64) {
        self.config.weight += delta;
        // Clip to reasonable range
        self.config.weight = self.config.weight.max(0.0).min(10.0);
    }

    /// Get distance from soma
    pub fn distance_from_soma(&self) -> f64 {
        self.config.distance_from_soma
    }

    /// Set distance from soma
    pub fn set_distance(&mut self, distance: f64) {
        self.config.distance_from_soma = distance;
    }

    /// Receive presynaptic spike
    pub fn receive_spike(&mut self, time: f64) {
        // Update ISI statistics
        if self.last_spike_time > 0.0 {
            let isi = time - self.last_spike_time;
            let alpha = 0.1; // Running average factor
            self.avg_isi = (1.0 - alpha) * self.avg_isi + alpha * isi;
        }

        self.last_spike_time = time;
        self.spike_count += 1;

        // Trigger synaptic conductance
        self.conductance.trigger(self.config.weight);
    }

    /// Calculate synaptic current at given voltage
    pub fn current(&self, voltage: f64) -> f64 {
        let g = self.conductance.conductance(voltage);
        g * (voltage - self.config.e_rev)
    }

    /// Update synapse dynamics
    pub fn update(&mut self, voltage: f64, dt: f64) {
        self.conductance.update(dt, voltage);
    }

    /// Reset synapse
    pub fn reset(&mut self) {
        self.conductance.reset();
        self.last_spike_time = -1000.0;
        self.spike_count = 0;
        self.avg_isi = 0.0;
    }

    /// Get spike count
    pub fn spike_count(&self) -> usize {
        self.spike_count
    }

    /// Get average inter-spike interval
    pub fn avg_isi(&self) -> f64 {
        self.avg_isi
    }

    /// Get firing rate (Hz)
    pub fn firing_rate(&self) -> f64 {
        if self.avg_isi > 0.0 {
            1000.0 / self.avg_isi // Convert ms to Hz
        } else {
            0.0
        }
    }

    /// Calculate EPSP amplitude at soma (approximate)
    pub fn epsp_amplitude(&self, attenuation_factor: f64) -> f64 {
        // Simplified: depends on distance and cable properties
        let peak_conductance = self.config.g_max * self.config.weight;
        let driving_force = -70.0 - self.config.e_rev; // Assuming rest at -70 mV
        let local_epsp = peak_conductance * driving_force;

        // Attenuate based on distance
        local_epsp * attenuation_factor
    }
}

/// Collection of synapses for temporal and spatial summation
#[derive(Debug, Clone)]
pub struct SynapseCollection {
    synapses: Vec<DendriticSynapse>,
}

impl SynapseCollection {
    /// Create empty collection
    pub fn new() -> Self {
        Self {
            synapses: Vec::new(),
        }
    }

    /// Add a synapse
    pub fn add(&mut self, synapse: DendriticSynapse) {
        self.synapses.push(synapse);
    }

    /// Get number of synapses
    pub fn len(&self) -> usize {
        self.synapses.len()
    }

    /// Check if collection is empty
    pub fn is_empty(&self) -> bool {
        self.synapses.is_empty()
    }

    /// Get synapses on a specific compartment
    pub fn on_compartment(&self, compartment_idx: usize) -> Vec<&DendriticSynapse> {
        self.synapses
            .iter()
            .filter(|s| s.compartment_idx() == compartment_idx)
            .collect()
    }

    /// Get mutable synapses on a specific compartment
    pub fn on_compartment_mut(&mut self, compartment_idx: usize) -> Vec<&mut DendriticSynapse> {
        self.synapses
            .iter_mut()
            .filter(|s| s.compartment_idx() == compartment_idx)
            .collect()
    }

    /// Calculate total current on a compartment
    pub fn total_current(&self, compartment_idx: usize, voltage: f64) -> f64 {
        self.on_compartment(compartment_idx)
            .iter()
            .map(|s| s.current(voltage))
            .sum()
    }

    /// Update all synapses
    pub fn update_all(&mut self, voltages: &[f64], dt: f64) {
        for synapse in &mut self.synapses {
            let idx = synapse.compartment_idx();
            if let Some(&voltage) = voltages.get(idx) {
                synapse.update(voltage, dt);
            }
        }
    }

    /// Reset all synapses
    pub fn reset_all(&mut self) {
        for synapse in &mut self.synapses {
            synapse.reset();
        }
    }

    /// Get synapse statistics
    pub fn statistics(&self) -> SynapseStats {
        let excitatory_count = self
            .synapses
            .iter()
            .filter(|s| s.synapse_type() == SynapseType::Excitatory)
            .count();

        let inhibitory_count = self
            .synapses
            .iter()
            .filter(|s| s.synapse_type() == SynapseType::Inhibitory)
            .count();

        let avg_weight = if !self.synapses.is_empty() {
            self.synapses.iter().map(|s| s.weight()).sum::<f64>() / self.synapses.len() as f64
        } else {
            0.0
        };

        SynapseStats {
            total: self.synapses.len(),
            excitatory: excitatory_count,
            inhibitory: inhibitory_count,
            avg_weight,
        }
    }
}

impl Default for SynapseCollection {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about synapse collection
#[derive(Debug, Clone)]
pub struct SynapseStats {
    pub total: usize,
    pub excitatory: usize,
    pub inhibitory: usize,
    pub avg_weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_conductance() {
        let mut g = SynapticConductance::alpha(1.0, 5.0);

        // Before spike
        assert_eq!(g.conductance(0.0), 0.0);

        // Trigger spike
        g.trigger(1.0);
        assert!(g.conductance(0.0) >= 0.0);

        // Should rise then fall
        g.update(1.0, 0.0);
        let g1 = g.conductance(0.0);

        g.update(5.0, 0.0);
        let g2 = g.conductance(0.0);

        g.update(5.0, 0.0);
        let g3 = g.conductance(0.0);

        assert!(g2 > g1);
        assert!(g3 < g2);
    }

    #[test]
    fn test_dendritic_synapse() {
        let mut syn = DendriticSynapse::excitatory(0, 1.0, 0.5);

        assert_eq!(syn.compartment_idx(), 0);
        assert_eq!(syn.synapse_type(), SynapseType::Excitatory);
        assert_eq!(syn.spike_count(), 0);

        // Receive spike
        syn.receive_spike(10.0);
        assert_eq!(syn.spike_count(), 1);

        // Should generate current
        let i = syn.current(-70.0);
        assert!(i < 0.0); // Excitatory at -70 mV
    }

    #[test]
    fn test_synapse_weight_modification() {
        let mut syn = DendriticSynapse::excitatory(0, 1.0, 0.5);

        assert_eq!(syn.weight(), 1.0);

        syn.set_weight(1.5);
        assert_eq!(syn.weight(), 1.5);

        syn.modify_weight(0.2);
        assert_eq!(syn.weight(), 1.7);

        syn.modify_weight(-0.5);
        assert_eq!(syn.weight(), 1.2);
    }

    #[test]
    fn test_synapse_collection() {
        let mut collection = SynapseCollection::new();

        // Add synapses on different compartments
        collection.add(DendriticSynapse::excitatory(0, 1.0, 0.5));
        collection.add(DendriticSynapse::excitatory(1, 1.0, 0.5));
        collection.add(DendriticSynapse::inhibitory(1, 0.8));

        assert_eq!(collection.len(), 3);

        // Check compartment filtering
        let on_comp1 = collection.on_compartment(1);
        assert_eq!(on_comp1.len(), 2);

        // Check statistics
        let stats = collection.statistics();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.excitatory, 2);
        assert_eq!(stats.inhibitory, 1);
    }

    #[test]
    fn test_isi_tracking() {
        let mut syn = DendriticSynapse::excitatory(0, 1.0, 0.5);

        syn.receive_spike(10.0);
        syn.receive_spike(20.0); // ISI = 10 ms

        assert!(syn.avg_isi() > 0.0);

        syn.receive_spike(30.0); // ISI = 10 ms

        // Average should be around 10 ms
        assert!((syn.avg_isi() - 10.0).abs() < 2.0);

        // Firing rate should be ~100 Hz
        let rate = syn.firing_rate();
        assert!((rate - 100.0).abs() < 20.0);
    }

    #[test]
    fn test_temporal_summation() {
        let mut syn = DendriticSynapse::excitatory(0, 1.0, 0.5);

        syn.receive_spike(0.0);
        let i1 = syn.current(-70.0).abs();

        // Second spike before first decays (temporal summation)
        syn.update(-70.0, 2.0);
        syn.receive_spike(2.0);
        let i2 = syn.current(-70.0).abs();

        // Current should be larger due to summation
        assert!(i2 > i1);
    }
}
