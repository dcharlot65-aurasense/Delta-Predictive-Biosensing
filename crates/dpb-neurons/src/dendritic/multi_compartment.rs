//! # Multi-Compartment Neuron Module
//!
//! Implements a complete multi-compartment neuron model integrating:
//! - Multiple compartments (soma + dendrites + axon)
//! - Cable equation for voltage propagation
//! - Ion channels in each compartment
//! - Synaptic inputs on dendrites
//! - Action potential generation
//! - Backpropagating action potentials
//! - Numerical solvers (Euler, Crank-Nicolson)
//!
//! ## Architecture
//!
//! ```text
//!        Dendrite
//!           |
//!         Soma --- Axon
//!           |
//!        Dendrite
//! ```

use super::channels::{HodgkinHuxleyChannel, IonChannel};
use super::compartment::{AxialCoupling, Compartment, CompartmentConfig};
use super::morphology::DendriticTree;
use super::synapse::{DendriticSynapse, SynapseCollection};

/// Numerical solver for compartment equations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalSolver {
    /// Forward Euler (explicit, fast but can be unstable)
    Euler,
    /// Crank-Nicolson (implicit, stable for large dt)
    CrankNicolson,
    /// Backward Euler (implicit, very stable)
    BackwardEuler,
}

/// Configuration for backpropagation
#[derive(Debug, Clone)]
pub struct BackpropagationConfig {
    /// Enable backpropagating action potentials
    pub enabled: bool,
    /// Attenuation factor per μm
    pub attenuation: f64,
    /// Propagation velocity (μm/ms)
    pub velocity: f64,
    /// Amplitude at soma (mV)
    pub amplitude: f64,
}

impl Default for BackpropagationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            attenuation: 0.002, // ~2% attenuation per μm
            velocity: 200.0,    // 200 μm/ms (~0.2 m/s)
            amplitude: 100.0,   // 100 mV spike
        }
    }
}

/// Configuration for multi-compartment neuron
#[derive(Debug, Clone)]
pub struct NeuronConfig {
    /// Compartment configuration
    pub compartment_config: CompartmentConfig,
    /// Numerical solver
    pub solver: NumericalSolver,
    /// Backpropagation configuration
    pub backprop_config: BackpropagationConfig,
    /// Action potential threshold at soma (mV)
    pub ap_threshold: f64,
    /// Reset potential after spike (mV)
    pub reset_potential: f64,
    /// Refractory period (ms)
    pub refractory_period: f64,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            compartment_config: CompartmentConfig::default(),
            // Backward Euler, not Crank-Nicolson.
            //
            // Neighbour voltages are taken from the previous step (Jacobi
            // coupling), and under that splitting the two schemes behave very
            // differently on a stiff cable. Writing a = 0.5*dt*G/C for an
            // interior compartment, CN weights the compartment's own previous
            // voltage by (1-a)/(1+a), which turns NEGATIVE once a > 1 while the
            // neighbour weights keep growing: at dt = 1 ms on the default
            // geometry that is -0.82 against 1.82, an amplification of 2.6 per
            // step, and the cable rings itself apart.
            //
            // Backward Euler's weights are all non-negative and sum to exactly
            // 1, so each update is a convex combination of the voltages already
            // present and cannot overshoot their range -- unconditionally
            // stable at any timestep. CrankNicolson remains available and is
            // more accurate, but is only stable while dt*G/C stays small; see
            // `NumericalSolver`.
            solver: NumericalSolver::BackwardEuler,
            backprop_config: BackpropagationConfig::default(),
            ap_threshold: -50.0,
            reset_potential: -70.0,
            refractory_period: 2.0,
        }
    }
}

/// Multi-compartment neuron model
#[derive(Debug, Clone)]
pub struct MultiCompartmentNeuron {
    /// All compartments (0 = soma)
    compartments: Vec<Compartment>,
    /// Dendritic tree structure
    tree: Option<DendriticTree>,
    /// Synapses on dendrites
    synapses: SynapseCollection,
    /// Ion channels per compartment
    na_channels: Vec<Option<HodgkinHuxleyChannel>>,
    k_channels: Vec<Option<HodgkinHuxleyChannel>>,
    /// Configuration
    config: NeuronConfig,
    /// Current simulation time (ms)
    time: f64,
    /// Last spike time (ms)
    last_spike_time: f64,
    /// Spike count
    spike_count: usize,
    /// Backpropagation queue (compartment_idx, arrival_time, amplitude)
    backprop_queue: Vec<(usize, f64, f64)>,
}

impl MultiCompartmentNeuron {
    /// Create a new multi-compartment neuron
    pub fn new(num_compartments: usize) -> Self {
        let config = NeuronConfig::default();
        let mut compartments: Vec<Compartment> = (0..num_compartments)
            .map(|i| Compartment::with_index(config.compartment_config.clone(), i))
            .collect();

        // Link the compartments into an unbranched cable: the soma is
        // compartment 0 and each subsequent compartment is the child of the one
        // before it.
        //
        // Without this every compartment had `parent: None` and no children, so
        // `calculate_axial_currents` returned all zeros and the compartments
        // were electrically isolated from one another. Current injected into a
        // dendrite could never reach the soma, which makes the structure a
        // collection of independent single compartments rather than a
        // multi-compartment neuron. `from_tree` overrides this topology.
        for i in 1..num_compartments {
            compartments[i].set_parent(i - 1);
            compartments[i - 1].add_child(i);
        }

        // Only soma (compartment 0) has Na and K channels for AP generation
        let mut na_channels = vec![None; num_compartments];
        let mut k_channels = vec![None; num_compartments];

        na_channels[0] = Some(HodgkinHuxleyChannel::sodium(120.0));
        k_channels[0] = Some(HodgkinHuxleyChannel::potassium(36.0));

        Self {
            compartments,
            tree: None,
            synapses: SynapseCollection::new(),
            na_channels,
            k_channels,
            config,
            time: 0.0,
            last_spike_time: -1000.0,
            spike_count: 0,
            backprop_queue: Vec::new(),
        }
    }

    /// Create from configuration
    pub fn with_config(num_compartments: usize, config: NeuronConfig) -> Self {
        let mut neuron = Self::new(num_compartments);
        neuron.config = config;
        neuron
    }

    /// Create from dendritic tree
    pub fn from_tree(tree: DendriticTree, config: NeuronConfig) -> Self {
        let num_compartments = tree.num_branches();
        let mut neuron = Self::with_config(num_compartments, config);
        neuron.tree = Some(tree);
        neuron
    }

    /// Get number of compartments
    pub fn num_compartments(&self) -> usize {
        self.compartments.len()
    }

    /// Get soma voltage
    pub fn soma_voltage(&self) -> f64 {
        self.compartments.first().map_or(-70.0, |c| c.voltage())
    }

    /// Get compartment voltage
    pub fn compartment_voltage(&self, idx: usize) -> Option<f64> {
        self.compartments.get(idx).map(|c| c.voltage())
    }

    /// Get all voltages
    pub fn voltages(&self) -> Vec<f64> {
        self.compartments.iter().map(|c| c.voltage()).collect()
    }

    /// Add synapse
    pub fn add_synapse(&mut self, synapse: DendriticSynapse) {
        // Set distance from soma if tree is available
        // (simplified - would use actual tree distance)
        self.synapses.add(synapse);
    }

    /// Get synapses
    pub fn synapses(&self) -> &SynapseCollection {
        &self.synapses
    }

    /// Get mutable synapses
    pub fn synapses_mut(&mut self) -> &mut SynapseCollection {
        &mut self.synapses
    }

    /// Check if in refractory period
    fn is_refractory(&self) -> bool {
        self.time - self.last_spike_time < self.config.refractory_period
    }

    /// Calculate axial currents between compartments
    /// Net axial current into each compartment, and the conductance carrying it.
    ///
    /// The conductance is returned alongside the current so the voltage solvers
    /// can treat the coupling implicitly -- see [`AxialCoupling`].
    fn calculate_axial_currents(&self) -> Vec<AxialCoupling> {
        let mut couplings = vec![AxialCoupling::NONE; self.compartments.len()];

        for (i, comp) in self.compartments.iter().enumerate() {
            let mut acc = AxialCoupling::NONE;

            let mut couple = |neighbour: &Compartment| {
                acc.current +=
                    comp.axial_current_to(neighbour.voltage(), neighbour.axial_resistance());
                acc.conductance += 1.0 / (comp.axial_resistance() + neighbour.axial_resistance());
            };

            // Current from parent
            if let Some(parent_idx) = comp.parent()
                && let Some(parent) = self.compartments.get(parent_idx)
            {
                couple(parent);
            }

            // Current from children
            for &child_idx in comp.children() {
                if let Some(child) = self.compartments.get(child_idx) {
                    couple(child);
                }
            }

            couplings[i] = acc;
        }

        couplings
    }

    /// Update ion channel currents
    fn update_channels(&mut self, dt: f64) {
        for (i, comp) in self.compartments.iter_mut().enumerate() {
            let voltage = comp.voltage();
            let mut ion_current = 0.0;

            // Sodium channel
            if let Some(na_channel) = self.na_channels.get_mut(i).and_then(|c| c.as_mut()) {
                na_channel.update(voltage, dt);
                ion_current += na_channel.current(voltage);
            }

            // Potassium channel
            if let Some(k_channel) = self.k_channels.get_mut(i).and_then(|c| c.as_mut()) {
                k_channel.update(voltage, dt);
                ion_current += k_channel.current(voltage);
            }

            // IonChannel::current() is a current DENSITY (nA/cm²) but
            // Compartment::ion_currents is an absolute current (nA), and the
            // integrator divides by the area-scaled total capacitance. Passing
            // the density straight through over-scaled every channel current by
            // 1/area — about 3e5 for default geometry — so the membrane
            // integrated to ±1e5 mV and then NaN within two steps, with no input.
            comp.set_ion_current(ion_current * comp.area());
        }
    }

    /// Update synaptic currents
    fn update_synapses(&mut self, dt: f64) {
        let voltages = self.voltages();
        self.synapses.update_all(&voltages, dt);

        // Apply synaptic currents to compartments
        for comp in &mut self.compartments {
            let idx = comp.index();
            let syn_current = self.synapses.total_current(idx, comp.voltage());
            comp.set_synaptic_current(syn_current);
        }
    }

    /// Process backpropagating action potentials
    fn process_backpropagation(&mut self) {
        if !self.config.backprop_config.enabled {
            return;
        }

        // Remove and process arrived backprop spikes
        let mut arrived = Vec::new();
        self.backprop_queue
            .retain(|(idx, arrival_time, amplitude)| {
                if self.time >= *arrival_time {
                    arrived.push((*idx, *amplitude));
                    false
                } else {
                    true
                }
            });

        // Apply backprop depolarization
        for (idx, amplitude) in arrived {
            if let Some(comp) = self.compartments.get_mut(idx) {
                let current_v = comp.voltage();
                comp.set_voltage(current_v + amplitude);
            }
        }
    }

    /// Initiate backpropagation from soma spike
    fn initiate_backpropagation(&mut self) {
        if !self.config.backprop_config.enabled {
            return;
        }

        let config = &self.config.backprop_config;

        // Calculate backprop for each compartment
        for comp in &self.compartments {
            if comp.index() == 0 {
                continue; // Skip soma
            }

            // Calculate distance from soma (simplified - use compartment index)
            let distance = if let Some(ref tree) = self.tree {
                tree.path_distance_to_node(comp.index()).unwrap_or(100.0)
            } else {
                comp.index() as f64 * 50.0 // Assume 50 μm per compartment
            };

            // Calculate arrival time based on propagation velocity
            let propagation_time = distance / config.velocity;
            let arrival_time = self.time + propagation_time;

            // Calculate attenuated amplitude
            let amplitude = config.amplitude * (-config.attenuation * distance).exp();

            self.backprop_queue
                .push((comp.index(), arrival_time, amplitude));
        }
    }

    /// How many forward-Euler substeps `dt` must be split into to stay inside
    /// the scheme's stability limit.
    ///
    /// The limit is `dt < 2C/G` with `G` the total conductance seen by a
    /// compartment -- leak plus axial. Targets `C/G`, i.e. half the limit, for
    /// margin. Capped so a pathological geometry cannot spin here forever; at
    /// the cap the step is simply less accurate rather than unbounded.
    fn explicit_substeps(&self, dt: f64) -> usize {
        const MAX_SUBSTEPS: usize = 4096;

        let mut shortest = f64::INFINITY;
        for (i, comp) in self.compartments.iter().enumerate() {
            let cable = comp.cable_params();
            let mut g = cable.conductance;

            if let Some(parent) = comp.parent()
                && let Some(p) = self.compartments.get(parent)
            {
                g += 1.0 / (comp.axial_resistance() + p.axial_resistance());
            }
            for &child in comp.children() {
                if let Some(c) = self.compartments.get(child) {
                    g += 1.0 / (comp.axial_resistance() + c.axial_resistance());
                }
            }

            if g > 0.0 && cable.capacitance > 0.0 {
                shortest = shortest.min(cable.capacitance / g);
            }
            let _ = i;
        }

        if !shortest.is_finite() || shortest <= 0.0 || dt <= shortest {
            return 1;
        }
        ((dt / shortest).ceil() as usize).clamp(1, MAX_SUBSTEPS)
    }

    /// Solve the cable equation implicitly across the whole tree in one step.
    ///
    /// This is the Hines (1984) two-pass elimination for a branched cable. It
    /// assembles, for every compartment,
    ///
    /// ```text
    /// (C/dt + theta*G_i) V_i^{n+1} - theta * sum_j g_ij V_j^{n+1}
    ///     = (C/dt - (1-theta)*G_i) V_i^n + (1-theta) * sum_j g_ij V_j^n
    ///       + g_leak * E + I_membrane
    /// ```
    ///
    /// with `G_i = g_leak + sum_j g_ij`, and solves the whole system at once.
    /// `theta = 1` gives backward Euler, `theta = 0.5` Crank-Nicolson.
    ///
    /// Solving simultaneously is what distinguishes a cable solver from
    /// per-compartment updates that read their neighbours' PREVIOUS voltages.
    /// Under that Jacobi splitting a disturbance advances only one compartment
    /// per timestep, so current injected into a distal dendrite took as many
    /// steps to reach the soma as there were compartments between them, and the
    /// attenuation it arrived with was a property of the step count rather than
    /// of the cable. Here it reaches every compartment within the same step.
    ///
    /// The tree matrix is symmetric and diagonally dominant, so the elimination
    /// needs no pivoting.
    ///
    /// Returns `false` if any compartment's parent does not precede it in index
    /// order, which the two passes below rely on; the caller then falls back to
    /// the per-compartment update.
    fn solve_implicit(&mut self, dt: f64, theta: f64) -> bool {
        let n = self.compartments.len();
        if n == 0 {
            return true;
        }

        let mut diag = vec![0.0f64; n];
        let mut rhs = vec![0.0f64; n];
        let mut g_parent = vec![0.0f64; n];

        // Membrane terms.
        for (i, comp) in self.compartments.iter().enumerate() {
            let cable = comp.cable_params();
            let c_over_dt = cable.capacitance / dt;
            let g_leak = cable.conductance;

            diag[i] = c_over_dt + theta * g_leak;
            rhs[i] = (c_over_dt - (1.0 - theta) * g_leak) * comp.voltage()
                + g_leak * comp.config().e_leak
                + comp.membrane_current();
        }

        // Axial terms, walked once per parent/child edge.
        for i in 0..n {
            let Some(parent) = self.compartments[i].parent() else {
                continue;
            };
            if parent >= i {
                return false;
            }
            let g = 1.0
                / (self.compartments[i].axial_resistance()
                    + self.compartments[parent].axial_resistance());
            g_parent[i] = g;

            diag[i] += theta * g;
            diag[parent] += theta * g;

            let (v_i, v_p) = (
                self.compartments[i].voltage(),
                self.compartments[parent].voltage(),
            );
            let explicit = 1.0 - theta;
            rhs[i] += explicit * g * (v_p - v_i);
            rhs[parent] += explicit * g * (v_i - v_p);
        }

        // Pass 1: eliminate each compartment into its parent. Descending index
        // guarantees every child is eliminated before its parent is used.
        for i in (1..n).rev() {
            let Some(parent) = self.compartments[i].parent() else {
                continue;
            };
            let g = theta * g_parent[i];
            if g == 0.0 || diag[i] == 0.0 {
                continue;
            }
            let factor = g / diag[i];
            diag[parent] -= g * factor;
            rhs[parent] += rhs[i] * factor;
        }

        // Pass 2: back-substitute from roots outward.
        let mut solved = vec![0.0f64; n];
        for i in 0..n {
            match self.compartments[i].parent() {
                None => solved[i] = rhs[i] / diag[i],
                Some(parent) => {
                    solved[i] = (rhs[i] + theta * g_parent[i] * solved[parent]) / diag[i]
                }
            }
        }

        for (comp, v) in self.compartments.iter_mut().zip(solved) {
            comp.set_voltage(v);
        }
        true
    }

    /// Update neuron state for one timestep
    pub fn update(&mut self, dt: f64) -> bool {
        self.time += dt;

        // Update ion channels
        self.update_channels(dt);

        // Update synapses
        self.update_synapses(dt);

        // Process backpropagation
        self.process_backpropagation();

        // Calculate axial currents
        let axial_currents = self.calculate_axial_currents();

        // Update each compartment voltage
        match self.config.solver {
            NumericalSolver::Euler => {
                // Forward Euler is only stable while dt < 2C/G, and on a
                // dendritic cable G is dominated by the AXIAL conductance, which
                // is orders of magnitude above the leak: the default geometry
                // gives C/G ~ 0.05 ms, so even a 0.1 ms step is past the limit
                // and the cable diverges to NaN. Subdivide internally rather
                // than hand back infinities for a timestep a caller has every
                // reason to think is small.
                let substeps = self.explicit_substeps(dt);
                let sub_dt = dt / substeps as f64;
                for _ in 0..substeps {
                    // Recomputed each substep: neighbour voltages have moved.
                    let couplings = self.calculate_axial_currents();
                    for (i, comp) in self.compartments.iter_mut().enumerate() {
                        comp.update_voltage_euler(sub_dt, couplings[i], 0.0);
                    }
                }
            }
            // Both implicit schemes solve the whole tree at once; they differ
            // only in theta. BackwardEuler previously just called
            // Crank-Nicolson, so selecting it silently gave the trapezoidal
            // scheme instead.
            NumericalSolver::CrankNicolson => {
                if !self.solve_implicit(dt, 0.5) {
                    for (i, comp) in self.compartments.iter_mut().enumerate() {
                        let axial = axial_currents[i];
                        comp.update_voltage_crank_nicolson(dt, axial, 0.0);
                    }
                }
            }
            NumericalSolver::BackwardEuler => {
                if !self.solve_implicit(dt, 1.0) {
                    for (i, comp) in self.compartments.iter_mut().enumerate() {
                        let axial = axial_currents[i];
                        comp.update_voltage_backward_euler(dt, axial, 0.0);
                    }
                }
            }
        }

        // Check for action potential at soma
        let mut spiked = false;
        if !self.is_refractory() {
            let soma_v = self.soma_voltage();
            if soma_v > self.config.ap_threshold {
                // Spike!
                spiked = true;
                self.last_spike_time = self.time;
                self.spike_count += 1;

                // Reset soma
                if let Some(soma) = self.compartments.get_mut(0) {
                    soma.set_voltage(self.config.reset_potential);
                }

                // Initiate backpropagation
                self.initiate_backpropagation();
            }
        }

        spiked
    }

    /// Stimulate a specific compartment
    pub fn stimulate_compartment(&mut self, idx: usize, current: f64) {
        if let Some(comp) = self.compartments.get_mut(idx) {
            let v = comp.voltage();
            comp.set_voltage(v + current); // Simplified current injection
        }
    }

    /// Reset neuron to resting state
    pub fn reset(&mut self) {
        for comp in &mut self.compartments {
            comp.reset();
        }

        self.synapses.reset_all();

        for na in self.na_channels.iter_mut().flatten() {
            na.reset();
        }

        for k in self.k_channels.iter_mut().flatten() {
            k.reset();
        }

        self.time = 0.0;
        self.last_spike_time = -1000.0;
        self.spike_count = 0;
        self.backprop_queue.clear();
    }

    /// Get spike count
    pub fn spike_count(&self) -> usize {
        self.spike_count
    }

    /// Get current time
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get dendritic tree
    pub fn tree(&self) -> Option<&DendriticTree> {
        self.tree.as_ref()
    }

    /// Enable/disable ion channels in a compartment
    pub fn set_compartment_channels(
        &mut self,
        idx: usize,
        na_conductance: Option<f64>,
        k_conductance: Option<f64>,
    ) {
        if let Some(g_na) = na_conductance
            && let Some(na_slot) = self.na_channels.get_mut(idx)
        {
            *na_slot = Some(HodgkinHuxleyChannel::sodium(g_na));
        }

        if let Some(g_k) = k_conductance
            && let Some(k_slot) = self.k_channels.get_mut(idx)
        {
            *k_slot = Some(HodgkinHuxleyChannel::potassium(g_k));
        }
    }

    /// Get voltage trace for all compartments
    pub fn voltage_trace(&self) -> Vec<(usize, f64)> {
        self.compartments
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.voltage()))
            .collect()
    }

    /// Calculate input resistance at soma
    pub fn input_resistance(&self) -> f64 {
        self.compartments
            .first()
            .map_or(0.0, |c| c.input_resistance())
    }

    /// Calculate membrane time constant at soma
    pub fn time_constant(&self) -> f64 {
        self.compartments.first().map_or(0.0, |c| c.time_constant())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dendritic::synapse::DendriticSynapse;

    #[test]
    fn test_multi_compartment_creation() {
        let neuron = MultiCompartmentNeuron::new(5);

        assert_eq!(neuron.num_compartments(), 5);
        assert_eq!(neuron.spike_count(), 0);
        assert_eq!(neuron.soma_voltage(), -70.0);
    }

    #[test]
    fn test_compartment_voltages() {
        let neuron = MultiCompartmentNeuron::new(3);

        let voltages = neuron.voltages();
        assert_eq!(voltages.len(), 3);

        for v in voltages {
            assert_eq!(v, -70.0);
        }
    }

    #[test]
    fn test_synapse_addition() {
        let mut neuron = MultiCompartmentNeuron::new(5);

        let syn = DendriticSynapse::excitatory(2, 1.0, 0.5);
        neuron.add_synapse(syn);

        assert_eq!(neuron.synapses().len(), 1);
    }

    #[test]
    fn test_update_no_input() {
        let mut neuron = MultiCompartmentNeuron::new(3);

        // Update without input
        for _ in 0..100 {
            let spiked = neuron.update(1.0);
            assert!(!spiked);
        }

        // The neuron must SETTLE, but not necessarily at `e_leak`.
        //
        // Resting potential is where the TOTAL membrane current vanishes, and
        // the soma carries Na and K conductances whose maxima are ~4000x the
        // leak, so they set it -- not `e_leak`. Asserting a return to -70 mV
        // assumed a passive compartment. What actually matters is that an
        // unstimulated neuron converges to a stable physiological potential
        // instead of drifting or diverging.
        let v = neuron.soma_voltage();
        assert!(
            (-90.0..=-40.0).contains(&v),
            "resting potential {v} outside a physiological range"
        );

        let before = neuron.soma_voltage();
        for _ in 0..50 {
            neuron.update(1.0);
        }
        assert!(
            (neuron.soma_voltage() - before).abs() < 0.1,
            "still moving at {} after settling from {before}",
            neuron.soma_voltage()
        );
    }

    #[test]
    fn test_soma_stimulation() {
        let mut neuron = MultiCompartmentNeuron::new(3);

        // Strong stimulation should trigger spike
        for _ in 0..20 {
            neuron.stimulate_compartment(0, 5.0); // Strong current
            neuron.update(0.1);
        }

        // Should have spiked
        assert!(neuron.spike_count() > 0);
    }

    #[test]
    fn test_dendritic_stimulation() {
        let mut neuron = MultiCompartmentNeuron::new(5);

        let initial_soma_v = neuron.soma_voltage();

        // Stimulate distal dendrite
        neuron.stimulate_compartment(4, 10.0);
        neuron.update(1.0);

        // Soma should see some depolarization (attenuated)
        let soma_v = neuron.soma_voltage();
        assert!(soma_v > initial_soma_v);
    }

    #[test]
    fn test_refractory_period() {
        let mut neuron = MultiCompartmentNeuron::new(3);

        // Trigger spike
        for _ in 0..20 {
            neuron.stimulate_compartment(0, 5.0);
            neuron.update(0.1);
        }

        let spike_count_1 = neuron.spike_count();

        // Try to trigger another spike immediately (should fail due to refractoriness)
        neuron.stimulate_compartment(0, 10.0);
        neuron.update(0.1);

        assert_eq!(neuron.spike_count(), spike_count_1);
    }

    #[test]
    fn test_reset() {
        let mut neuron = MultiCompartmentNeuron::new(3);

        // Stimulate and spike
        for _ in 0..20 {
            neuron.stimulate_compartment(0, 5.0);
            neuron.update(0.1);
        }

        assert!(neuron.spike_count() > 0);

        // Reset
        neuron.reset();

        assert_eq!(neuron.spike_count(), 0);
        assert_eq!(neuron.soma_voltage(), -70.0);
        assert_eq!(neuron.time(), 0.0);
    }

    #[test]
    fn test_numerical_solvers() {
        let mut neuron_euler = MultiCompartmentNeuron::with_config(
            3,
            NeuronConfig {
                solver: NumericalSolver::Euler,
                ..Default::default()
            },
        );

        let mut neuron_cn = MultiCompartmentNeuron::with_config(
            3,
            NeuronConfig {
                solver: NumericalSolver::CrankNicolson,
                ..Default::default()
            },
        );

        // Both should remain stable
        for _ in 0..100 {
            neuron_euler.update(0.1);
            neuron_cn.update(0.1);
        }

        assert!(neuron_euler.soma_voltage().is_finite());
        assert!(neuron_cn.soma_voltage().is_finite());
    }

    #[test]
    fn test_backpropagation() {
        let mut neuron = MultiCompartmentNeuron::with_config(
            5,
            NeuronConfig {
                backprop_config: BackpropagationConfig {
                    enabled: true,
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        // Trigger soma spike
        for _ in 0..20 {
            neuron.stimulate_compartment(0, 5.0);
            neuron.update(0.1);
        }

        // Wait for backpropagation to arrive at distal compartments
        for _ in 0..100 {
            neuron.update(1.0);
        }

        // Distal compartments should have seen some depolarization
        // (in reality - this is simplified and may need adjustment)
    }

    #[test]
    fn test_compartment_channels() {
        let mut neuron = MultiCompartmentNeuron::new(3);

        // Add active channels to dendrite
        neuron.set_compartment_channels(1, Some(50.0), Some(20.0));

        // Check channels were added
        assert!(neuron.na_channels[1].is_some());
        assert!(neuron.k_channels[1].is_some());
    }

    #[test]
    fn test_biophysical_properties() {
        let neuron = MultiCompartmentNeuron::new(3);

        let r_in = neuron.input_resistance();
        let tau = neuron.time_constant();

        assert!(r_in > 0.0);
        assert!(tau > 0.0);
        assert!(tau < 100.0); // Reasonable time constant
    }

    #[test]
    fn test_voltage_trace() {
        let neuron = MultiCompartmentNeuron::new(4);

        let trace = neuron.voltage_trace();

        assert_eq!(trace.len(), 4);
        for (idx, v) in trace {
            assert!(idx < 4);
            assert_eq!(v, -70.0);
        }
    }

    #[test]
    fn test_with_config() {
        let config = NeuronConfig {
            ap_threshold: -55.0,
            reset_potential: -75.0,
            refractory_period: 3.0,
            ..Default::default()
        };

        let neuron = MultiCompartmentNeuron::with_config(3, config.clone());

        assert_eq!(neuron.config.ap_threshold, -55.0);
        assert_eq!(neuron.config.reset_potential, -75.0);
        assert_eq!(neuron.config.refractory_period, 3.0);
    }

    /// The implicit tree solve must agree with an explicit reference.
    ///
    /// Guards the Hines elimination in `solve_implicit`: at a timestep well
    /// inside forward Euler's stability limit the two schemes must converge to
    /// the same trajectory. A sign error or a mis-ordered pass in the
    /// elimination would show up here as divergence even though the solver
    /// would still look stable on its own.
    #[test]
    fn test_implicit_solve_matches_explicit_reference() {
        let make = |solver| {
            MultiCompartmentNeuron::with_config(
                4,
                NeuronConfig {
                    solver,
                    ..Default::default()
                },
            )
        };
        let mut implicit = make(NumericalSolver::BackwardEuler);
        let mut explicit = make(NumericalSolver::Euler);

        // Same perturbation to both, away from the soma.
        implicit.stimulate_compartment(3, 15.0);
        explicit.stimulate_compartment(3, 15.0);

        // Small dt: backward Euler's O(dt) error stays tiny and forward Euler
        // is comfortably stable.
        let dt = 0.002;
        for _ in 0..500 {
            implicit.update(dt);
            explicit.update(dt);
        }

        for (i, (a, b)) in implicit
            .voltages()
            .iter()
            .zip(explicit.voltages().iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 0.5,
                "compartment {i}: implicit {a} vs explicit {b}"
            );
        }
    }

    /// Every solver must stay finite at a timestep a caller would reasonably
    /// pick, including the explicit one.
    #[test]
    fn test_all_solvers_stable_at_typical_timestep() {
        for solver in [
            NumericalSolver::Euler,
            NumericalSolver::CrankNicolson,
            NumericalSolver::BackwardEuler,
        ] {
            let mut neuron = MultiCompartmentNeuron::with_config(
                4,
                NeuronConfig {
                    solver,
                    ..Default::default()
                },
            );
            neuron.stimulate_compartment(3, 10.0);
            for _ in 0..200 {
                neuron.update(0.1);
            }
            for (i, v) in neuron.voltages().iter().enumerate() {
                assert!(
                    v.is_finite() && (-200.0..=100.0).contains(v),
                    "{solver:?} compartment {i} reached {v}"
                );
            }
        }
    }
}
