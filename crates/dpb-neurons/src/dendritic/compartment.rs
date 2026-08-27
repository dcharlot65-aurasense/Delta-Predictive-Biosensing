//! # Compartment Module
//!
//! Implements single compartment dynamics using the cable equation.
//! Each compartment represents a small section of dendrite with:
//! - Membrane capacitance
//! - Membrane resistance (leak conductance)
//! - Axial resistance (coupling to neighbors)
//! - Ion channel currents
//!
//! ## Cable Equation
//!
//! dV/dt = (1/Cm) * [I_leak + I_channels + I_axial + I_syn + I_ext]
//!
//! where:
//! - Cm: membrane capacitance
//! - I_leak: leak current through membrane
//! - I_channels: active ion channel currents
//! - I_axial: axial current from neighboring compartments
//! - I_syn: synaptic current
//! - I_ext: external applied current

use std::f64::consts::PI;

/// Configuration for a single compartment
#[derive(Debug, Clone)]
pub struct CompartmentConfig {
    /// Membrane capacitance (μF/cm²)
    pub cm: f64,
    /// Membrane leak conductance (mS/cm²)
    pub g_leak: f64,
    /// Leak reversal potential (mV)
    pub e_leak: f64,
    /// Compartment length (μm)
    pub length: f64,
    /// Compartment diameter (μm)
    pub diameter: f64,
    /// Axial resistivity (Ω·cm)
    pub r_axial: f64,
}

impl Default for CompartmentConfig {
    fn default() -> Self {
        Self {
            cm: 1.0,           // 1 μF/cm² - typical membrane capacitance
            g_leak: 0.03,      // 0.03 mS/cm² - leak conductance
            e_leak: -70.0,     // -70 mV - resting potential
            length: 50.0,      // 50 μm - compartment length
            diameter: 2.0,     // 2 μm - dendrite diameter
            r_axial: 100.0,    // 100 Ω·cm - intracellular resistivity
        }
    }
}

/// Cable theory parameters derived from compartment geometry
#[derive(Debug, Clone)]
pub struct CableParams {
    /// Surface area (cm²)
    pub area: f64,
    /// Cross-sectional area (cm²)
    pub cross_section: f64,
    /// Total capacitance (μF)
    pub capacitance: f64,
    /// Leak conductance (mS)
    pub conductance: f64,
    /// Axial resistance (MΩ)
    pub axial_resistance: f64,
}

impl CableParams {
    /// Calculate cable parameters from geometry
    pub fn from_geometry(config: &CompartmentConfig) -> Self {
        // Convert μm to cm
        let length_cm = config.length * 1e-4;
        let diameter_cm = config.diameter * 1e-4;
        let radius_cm = diameter_cm / 2.0;

        // Surface area: π * d * L
        let area = PI * diameter_cm * length_cm;

        // Cross-sectional area: π * r²
        let cross_section = PI * radius_cm * radius_cm;

        // Total capacitance: Cm * area
        let capacitance = config.cm * area;

        // Total leak conductance: g_leak * area
        let conductance = config.g_leak * area;

        // Axial resistance: (R_a * L) / A
        // Axial resistance in kOhm, NOT MOhm.
        //
        // The rest of this model works in mV, ms, uF, mS and uA. Conductance is
        // in mS, so a resistance must be in kOhm to be its reciprocal, and only
        // then does `dV / R` yield uA to match the leak current `g * dV`.
        // Converting to MOhm made every axial current 1000x too large relative
        // to every other current in the same sum.
        let axial_resistance = (config.r_axial * length_cm) / cross_section;
        let axial_resistance = axial_resistance / 1e3; // Ohm -> kOhm

        Self {
            area,
            cross_section,
            capacitance,
            conductance,
            axial_resistance,
        }
    }
}

/// Coupling of a compartment to its neighbours over one timestep.
///
/// The conductance is carried separately from the current because the
/// compartment's own voltage appears on both sides of `sum_j g_j * (V_j - V)`.
/// Collapsing that into a single current term forces the coupling to be
/// integrated explicitly, which diverges once `dt * g / C` exceeds 2 -- routine
/// for fine dendritic compartments at millisecond timesteps, where the axial
/// conductance is orders of magnitude larger than the membrane conductance.
/// Keeping the conductance lets the implicit solvers put `V` on the left.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AxialCoupling {
    /// Net current flowing in from neighbours at their present voltages (uA).
    pub current: f64,
    /// Total coupling conductance to those neighbours (mS).
    pub conductance: f64,
}

impl AxialCoupling {
    /// An isolated compartment: no neighbours, no coupling.
    pub const NONE: Self = Self { current: 0.0, conductance: 0.0 };
}

/// Single compartment in a multi-compartment neuron model
#[derive(Debug, Clone)]
pub struct Compartment {
    /// Configuration
    config: CompartmentConfig,
    /// Cable parameters
    cable: CableParams,
    /// Membrane potential (mV)
    voltage: f64,
    /// Axial currents from neighbors (nA)
    /// Ion channel currents (nA)
    ion_currents: f64,
    /// Synaptic current (nA)
    synaptic_current: f64,
    /// Compartment index in the neuron
    index: usize,
    /// Parent compartment index (None for soma)
    parent: Option<usize>,
    /// Child compartment indices
    children: Vec<usize>,
}

impl Compartment {
    /// Create a new compartment with default configuration
    pub fn new(config: CompartmentConfig) -> Self {
        let cable = CableParams::from_geometry(&config);
        Self {
            voltage: config.e_leak,
            config,
            cable,
            ion_currents: 0.0,
            synaptic_current: 0.0,
            index: 0,
            parent: None,
            children: Vec::new(),
        }
    }

    /// Create a compartment at specific index
    pub fn with_index(config: CompartmentConfig, index: usize) -> Self {
        let mut comp = Self::new(config);
        comp.index = index;
        comp
    }

    /// Get membrane voltage
    pub fn voltage(&self) -> f64 {
        self.voltage
    }

    /// Set membrane voltage
    pub fn set_voltage(&mut self, v: f64) {
        self.voltage = v;
    }

    /// Get compartment index
    pub fn index(&self) -> usize {
        self.index
    }

    /// Set parent compartment
    pub fn set_parent(&mut self, parent_idx: usize) {
        self.parent = Some(parent_idx);
    }

    /// Get parent compartment
    pub fn parent(&self) -> Option<usize> {
        self.parent
    }

    /// Add child compartment
    pub fn add_child(&mut self, child_idx: usize) {
        self.children.push(child_idx);
    }

    /// Get children compartments
    pub fn children(&self) -> &[usize] {
        &self.children
    }

    /// Get surface area
    pub fn area(&self) -> f64 {
        self.cable.area
    }

    /// Get axial resistance
    pub fn axial_resistance(&self) -> f64 {
        self.cable.axial_resistance
    }

    /// Calculate leak current
    pub fn leak_current(&self) -> f64 {
        // I_leak = g_leak * (E_leak - V)
        self.cable.conductance * (self.config.e_leak - self.voltage)
    }

    /// Calculate axial current to a neighboring compartment
    pub fn axial_current_to(&self, neighbor_voltage: f64, neighbor_resistance: f64) -> f64 {
        // I_axial = (V_neighbor - V) / (R_axial + R_neighbor)
        let total_resistance = self.cable.axial_resistance + neighbor_resistance;
        (neighbor_voltage - self.voltage) / total_resistance
    }

    /// Set ion channel current
    pub fn set_ion_current(&mut self, current: f64) {
        self.ion_currents = current;
    }

    /// Add to ion channel current
    pub fn add_ion_current(&mut self, current: f64) {
        self.ion_currents += current;
    }

    /// Set synaptic current
    pub fn set_synaptic_current(&mut self, current: f64) {
        self.synaptic_current = current;
    }

    /// Add to synaptic current
    pub fn add_synaptic_current(&mut self, current: f64) {
        self.synaptic_current += current;
    }

    /// Get total current
    /// Membrane current from ion channels and synapses (uA).
    ///
    /// Excludes the leak, which the solvers handle separately because it is
    /// linear in `V` and therefore belongs on the implicit side.
    pub fn membrane_current(&self) -> f64 {
        self.ion_currents + self.synaptic_current
    }

    pub fn total_current(&self, axial_current: f64, external_current: f64) -> f64 {
        self.leak_current() + self.ion_currents + axial_current
            + self.synaptic_current + external_current
    }

    /// Update membrane potential using forward Euler
    pub fn update_voltage_euler(&mut self, dt: f64, axial: AxialCoupling, external_current: f64) {
        let (g, drive) = self.conductance_and_drive(axial, external_current);
        let dv = ((drive - g * self.voltage) / self.cable.capacitance) * dt;
        self.voltage += dv;
    }

    /// Total conductance seen by this compartment's voltage, and the drive term
    /// it is pulled toward.
    ///
    /// Writes the membrane equation as `C dV/dt = drive - g * V`, which is what
    /// lets the implicit solvers below place `V` on the left-hand side. The
    /// neighbour contribution `sum_j g_j * V_j` is recovered from the coupling
    /// as `current + V * conductance`, since
    /// `current = sum_j g_j * (V_j - V)`.
    fn conductance_and_drive(&self, axial: AxialCoupling, external_current: f64) -> (f64, f64) {
        let g_leak = self.cable.conductance;
        let g = g_leak + axial.conductance;
        let neighbour_drive = axial.current + self.voltage * axial.conductance;
        let drive = g_leak * self.config.e_leak
            + neighbour_drive
            + self.ion_currents
            + self.synaptic_current
            + external_current;
        (g, drive)
    }

    /// Fully implicit (backward Euler) voltage update.
    ///
    /// Unconditionally stable and monotone: the result is a convex combination
    /// of the present voltage and the drive, so it cannot overshoot the range of
    /// voltages present. Preferred over Crank-Nicolson at large timesteps, where
    /// CN stays bounded but rings.
    pub fn update_voltage_backward_euler(
        &mut self,
        dt: f64,
        axial: AxialCoupling,
        external_current: f64,
    ) {
        let (g, drive) = self.conductance_and_drive(axial, external_current);
        let c = self.cable.capacitance;
        self.voltage = (self.voltage + (dt / c) * drive) / (1.0 + dt * g / c);
    }

    /// Update membrane potential using Crank-Nicolson (implicit)
    /// More stable for stiff systems
    pub fn update_voltage_crank_nicolson(
        &mut self,
        dt: f64,
        axial: AxialCoupling,
        external_current: f64,
    ) {
        // dV/dt = (drive - g*V) / C
        // CN: (V_new - V_old)/dt = 0.5 * [f(V_old) + f(V_new)]
        //  => V_new * (1 + 0.5*dt*g/C) = V_old * (1 - 0.5*dt*g/C) + dt/C * drive
        //
        // `g` here includes the AXIAL conductance, not just the leak. Treating
        // only the leak implicitly and passing the axial term through as a fixed
        // current made this explicit in the very conductance that dominates a
        // dendritic cable, so it diverged at any realistic timestep.
        let c = self.cable.capacitance;
        let (g, drive) = self.conductance_and_drive(axial, external_current);

        let alpha = 0.5 * dt * g / c;
        self.voltage = (self.voltage * (1.0 - alpha) + (dt / c) * drive) / (1.0 + alpha);
    }

    /// Reset compartment to resting state
    pub fn reset(&mut self) {
        self.voltage = self.config.e_leak;
        self.ion_currents = 0.0;
        self.synaptic_current = 0.0;
    }

    /// Get configuration
    pub fn config(&self) -> &CompartmentConfig {
        &self.config
    }

    /// Get cable parameters
    pub fn cable_params(&self) -> &CableParams {
        &self.cable
    }

    /// Calculate input resistance (MΩ)
    pub fn input_resistance(&self) -> f64 {
        // R_in ≈ 1/g_leak (simplified, ignoring geometry)
        1.0 / self.cable.conductance
    }

    /// Calculate membrane time constant (ms)
    pub fn time_constant(&self) -> f64 {
        // τ = Cm / g_leak
        self.cable.capacitance / self.cable.conductance
    }

    /// Calculate space constant (electrotonic length, μm)
    pub fn space_constant(&self) -> f64 {
        // λ = sqrt(d / (4 * R_a * g_leak))
        let d = self.config.diameter * 1e-4; // convert to cm
        let ra = self.config.r_axial; // Ohm.cm
        // g_leak is stored in mS/cm^2; the cable formula needs S/cm^2, since
        // 1/g_leak has to come out as the specific membrane resistance in
        // Ohm.cm^2. Using the mS figure directly understated lambda by a factor
        // of sqrt(1000) -- 41 um instead of ~1290 um for the default geometry,
        // i.e. shorter than a single compartment.
        let g_leak = self.config.g_leak * 1e-3; // mS/cm^2 -> S/cm^2

        let lambda_cm = (d / (4.0 * ra * g_leak)).sqrt();
        lambda_cm * 1e4 // convert back to μm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compartment_creation() {
        let config = CompartmentConfig::default();
        let comp = Compartment::new(config.clone());

        assert_eq!(comp.voltage(), config.e_leak);
        assert_eq!(comp.index(), 0);
        assert_eq!(comp.parent(), None);
        assert_eq!(comp.children().len(), 0);
    }

    #[test]
    fn test_cable_params() {
        let config = CompartmentConfig::default();
        let cable = CableParams::from_geometry(&config);

        // Check that parameters are positive
        assert!(cable.area > 0.0);
        assert!(cable.capacitance > 0.0);
        assert!(cable.conductance > 0.0);
        assert!(cable.axial_resistance > 0.0);
    }

    #[test]
    fn test_leak_current() {
        let config = CompartmentConfig::default();
        let mut comp = Compartment::new(config.clone());

        // At rest, leak current should be zero
        let i_leak = comp.leak_current();
        assert!(i_leak.abs() < 1e-10);

        // Depolarize the compartment
        comp.set_voltage(-50.0);
        let i_leak = comp.leak_current();

        // Leak current should be negative (outward)
        assert!(i_leak < 0.0);
    }

    #[test]
    fn test_axial_current() {
        let config = CompartmentConfig::default();
        let comp1 = Compartment::new(config.clone());
        let comp2 = Compartment::new(config.clone());

        // Create voltage difference
        let v1 = -70.0;
        let v2 = -60.0;

        let mut comp1_mut = comp1.clone();
        comp1_mut.set_voltage(v1);

        // Current should flow from comp2 to comp1 (positive)
        let i_axial = comp1_mut.axial_current_to(v2, comp2.axial_resistance());
        assert!(i_axial > 0.0);
    }

    #[test]
    fn test_voltage_update() {
        let config = CompartmentConfig::default();
        let mut comp = Compartment::new(config.clone());

        let v_initial = comp.voltage();

        // Apply a depolarizing current sized from the compartment's own input
        // resistance, rather than a fixed figure.
        //
        // The default geometry is a 50x2 um dendritic segment: ~3 pF and ~94 pS,
        // so its input resistance is ~10 GOhm and a few PICOamps move it tens of
        // millivolts. The literal 0.5 here (documented as "0.5nA", though this
        // model's currents are uA) drove it by ~5 volts.
        let i_inject = 20.0 / comp.input_resistance(); // ~20 mV at steady state
        comp.update_voltage_euler(1.0, AxialCoupling::NONE, i_inject);

        // Voltage should increase
        assert!(comp.voltage() > v_initial);
    }

    #[test]
    fn test_crank_nicolson_stability() {
        let config = CompartmentConfig::default();
        let mut comp = Compartment::new(config.clone());

        // Large timestep (normally unstable for Euler)
        let dt = 5.0; // 5ms
        let v_initial = comp.voltage();

        // ~20 mV of steady-state depolarization -- see `test_voltage_update` for
        // why this is derived from the input resistance and not a fixed figure.
        let i_inject = 20.0 / comp.input_resistance();

        // Should remain stable
        for _ in 0..100 {
            comp.update_voltage_crank_nicolson(dt, AxialCoupling::NONE, i_inject);
        }

        // Should converge to steady state, not explode
        assert!(comp.voltage().is_finite());
        assert!(comp.voltage() > v_initial);
        assert!(comp.voltage() < 0.0); // Still below 0 mV

        // Steady state of `C dV/dt = g(E - V) + I` is `E + I/g`.
        let expected = comp.config().e_leak + i_inject * comp.input_resistance();
        assert!(
            (comp.voltage() - expected).abs() < 0.1,
            "settled at {} rather than {expected}",
            comp.voltage()
        );
    }

    #[test]
    fn test_parent_child_relationships() {
        let config = CompartmentConfig::default();
        let mut soma = Compartment::with_index(config.clone(), 0);
        let mut dend1 = Compartment::with_index(config.clone(), 1);
        let mut dend2 = Compartment::with_index(config.clone(), 2);

        // Build tree: soma -> dend1 -> dend2
        dend1.set_parent(0);
        soma.add_child(1);

        dend2.set_parent(1);
        dend1.add_child(2);

        assert_eq!(soma.parent(), None);
        assert_eq!(soma.children(), &[1]);
        assert_eq!(dend1.parent(), Some(0));
        assert_eq!(dend1.children(), &[2]);
        assert_eq!(dend2.parent(), Some(1));
        assert_eq!(dend2.children().len(), 0);
    }

    #[test]
    fn test_biophysical_properties() {
        let config = CompartmentConfig::default();
        let comp = Compartment::new(config.clone());

        // Test input resistance
        let r_in = comp.input_resistance();
        assert!(r_in > 0.0);

        // Test time constant (should be ~10-30ms for typical neurons)
        let tau = comp.time_constant();
        assert!(tau > 5.0 && tau < 100.0);

        // Test space constant (should be hundreds of μm)
        let lambda = comp.space_constant();
        assert!(lambda > 100.0 && lambda < 10000.0);
    }

    #[test]
    fn test_reset() {
        let config = CompartmentConfig::default();
        let mut comp = Compartment::new(config.clone());

        // Modify state
        comp.set_voltage(-40.0);
        comp.add_ion_current(1.0);
        comp.add_synaptic_current(0.5);

        // Reset
        comp.reset();

        assert_eq!(comp.voltage(), config.e_leak);
        assert_eq!(comp.ion_currents, 0.0);
        assert_eq!(comp.synaptic_current, 0.0);
    }
}
