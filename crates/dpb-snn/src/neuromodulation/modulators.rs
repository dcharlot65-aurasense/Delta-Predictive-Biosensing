//! Core Neuromodulator Types and Dynamics
//!
//! This module implements the fundamental types for representing neuromodulators,
//! their concentrations, diffusion dynamics, and receptor binding kinetics.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::NeuromodResult;

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
/// Types of neuromodulators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NeuromodulatorType {
    /// Dopamine - reward, motivation, learning
    Dopamine,
    /// Acetylcholine - attention, arousal, learning
    Acetylcholine,
    /// Serotonin - mood, timing, patience
    Serotonin,
    /// Norepinephrine - arousal, alertness, stress
    Norepinephrine,
    /// GABA - inhibition (can act as modulator)
    GABA,
    /// Glutamate - excitation (can act as modulator)
    Glutamate,
    /// Custom neuromodulator
    Custom(u8),
}

impl NeuromodulatorType {
    /// Get the typical time constant for this neuromodulator (ms)
    pub fn typical_time_constant(&self) -> f64 {
        match self {
            Self::Dopamine => 200.0,       // Fast phasic, slow tonic
            Self::Acetylcholine => 100.0,  // Fast
            Self::Serotonin => 1000.0,     // Slow
            Self::Norepinephrine => 150.0, // Medium
            Self::GABA => 50.0,            // Fast
            Self::Glutamate => 20.0,       // Very fast
            Self::Custom(_) => 200.0,      // Default
        }
    }

    /// Get typical diffusion radius (μm)
    pub fn typical_diffusion_radius(&self) -> f64 {
        match self {
            Self::Dopamine => 100.0,       // Volume transmission
            Self::Acetylcholine => 50.0,   // Moderate diffusion
            Self::Serotonin => 200.0,      // Wide diffusion
            Self::Norepinephrine => 150.0, // Wide diffusion
            Self::GABA => 10.0,            // Local
            Self::Glutamate => 5.0,        // Very local
            Self::Custom(_) => 100.0,      // Default
        }
    }
}

/// Trait for neuromodulatory systems
pub trait ModulatorySystem: Send + Sync {
    /// Get the current concentration level
    fn get_concentration(&self) -> f64;

    /// Set the concentration level
    fn set_concentration(&mut self, concentration: f64) -> NeuromodResult<()>;

    /// Update concentration based on release and clearance
    fn update(&mut self, dt: f64, release: f64) -> NeuromodResult<()>;

    /// Get the modulator type
    fn modulator_type(&self) -> NeuromodulatorType;

    /// Reset to baseline concentration
    fn reset(&mut self);
}

/// Concentration dynamics for a neuromodulator
///
/// Models the temporal dynamics of neuromodulator concentration including
/// release, diffusion, reuptake, and enzymatic degradation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModulatorConcentration {
    /// Current concentration (normalized 0-1)
    pub concentration: f64,
    /// Baseline/tonic concentration
    pub baseline: f64,
    /// Maximum concentration
    pub max_concentration: f64,
    /// Time constant for clearance (ms)
    pub tau_clearance: f64,
    /// Reuptake rate (1/ms)
    pub reuptake_rate: f64,
    /// Enzymatic degradation rate (1/ms)
    pub degradation_rate: f64,
    /// Diffusion coefficient (μm²/ms)
    pub diffusion_coefficient: f64,
}

impl Default for ModulatorConcentration {
    fn default() -> Self {
        Self {
            concentration: 0.1,
            baseline: 0.1,
            max_concentration: 1.0,
            tau_clearance: 200.0,
            reuptake_rate: 0.001,
            degradation_rate: 0.0005,
            diffusion_coefficient: 0.3,
        }
    }
}

impl ModulatorConcentration {
    /// Create new concentration dynamics
    pub fn new(baseline: f64, tau_clearance: f64) -> Self {
        Self {
            concentration: baseline,
            baseline,
            tau_clearance,
            ..Default::default()
        }
    }

    /// Update concentration with release and clearance
    ///
    /// # Arguments
    /// * `dt` - Time step (ms)
    /// * `release` - Amount of neuromodulator released
    ///
    /// # Returns
    /// New concentration level
    pub fn update(&mut self, dt: f64, release: f64) -> f64 {
        // Release increases concentration
        self.concentration += release;

        // Clearance acts on the excess over the tonic baseline, not on the
        // absolute concentration.
        //
        // Reuptake and enzymatic degradation are the mechanisms BY WHICH
        // released transmitter is cleared -- they are not a separate drain
        // competing with clearance, so they belong in the same rate term.
        // Summing them as an additional decay toward zero put the resting fixed
        // point below `baseline` (0.077 rather than 0.1 at the defaults), which
        // made `baseline` untrue and left `relative_concentration` reading ~0.77
        // at rest instead of 1.0 -- miscalibrating every `is_phasic` threshold
        // downstream. Tonic release holds the floor at `baseline`; these rates
        // set how fast a phasic transient returns to it.
        let rate = 1.0 / self.tau_clearance + self.reuptake_rate + self.degradation_rate;

        // Clamp the step fraction so a large `dt` relaxes to baseline rather
        // than overshooting past it and oscillating.
        let fraction = (rate * dt).clamp(0.0, 1.0);
        self.concentration -= (self.concentration - self.baseline) * fraction;

        // Apply bounds
        self.concentration = self.concentration.max(0.0).min(self.max_concentration);

        self.concentration
    }

    /// Get relative concentration (normalized by baseline)
    pub fn relative_concentration(&self) -> f64 {
        if self.baseline > 0.0 {
            self.concentration / self.baseline
        } else {
            self.concentration
        }
    }

    /// Reset to baseline
    pub fn reset(&mut self) {
        self.concentration = self.baseline;
    }

    /// Check if concentration is at phasic (high) level
    pub fn is_phasic(&self, threshold: f64) -> bool {
        self.relative_concentration() > threshold
    }
}

/// Spatial diffusion model for neuromodulators
///
/// Models volume transmission and spatial spread of neuromodulators
/// using a simplified diffusion equation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionModel {
    /// Diffusion coefficient (μm²/ms)
    pub diffusion_coefficient: f64,
    /// Spatial decay constant (1/μm)
    pub spatial_decay: f64,
    /// Effective radius of influence (μm)
    pub effective_radius: f64,
}

impl Default for DiffusionModel {
    fn default() -> Self {
        Self {
            diffusion_coefficient: 0.3,
            spatial_decay: 0.01,
            effective_radius: 100.0,
        }
    }
}

impl DiffusionModel {
    /// Compute concentration at distance from source
    ///
    /// Uses Gaussian diffusion approximation:
    /// C(r, t) = C₀ * exp(-r² / (4 * D * t)) / (4πDt)^(3/2)
    ///
    /// Simplified for steady-state:
    /// C(r) = C₀ * exp(-λr)
    pub fn concentration_at_distance(&self, source_concentration: f64, distance: f64) -> f64 {
        if distance > self.effective_radius {
            return 0.0;
        }
        source_concentration * (-self.spatial_decay * distance).exp()
    }

    /// Compute diffusion over spatial grid
    pub fn diffuse_2d(&self, concentration: &Array2<f64>, dt: f64, dx: f64) -> Array2<f64> {
        let (height, width) = concentration.dim();
        let mut new_concentration = concentration.clone();

        let diffusion_factor = self.diffusion_coefficient * dt / (dx * dx);

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                // 2D Laplacian (5-point stencil)
                let laplacian = concentration[[i + 1, j]]
                    + concentration[[i - 1, j]]
                    + concentration[[i, j + 1]]
                    + concentration[[i, j - 1]]
                    - 4.0 * concentration[[i, j]];

                new_concentration[[i, j]] += diffusion_factor * laplacian;

                // Apply spatial decay
                new_concentration[[i, j]] *= 1.0 - self.spatial_decay * dt;
            }
        }

        new_concentration
    }
}

/// Receptor binding kinetics
///
/// Models the binding of neuromodulators to receptors using
/// first-order kinetics with association and dissociation rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptorBinding {
    /// Receptor type identifier
    pub receptor_type: String,
    /// Association rate constant (1/(concentration·ms))
    pub k_on: f64,
    /// Dissociation rate constant (1/ms)
    pub k_off: f64,
    /// Current receptor occupancy (0-1)
    pub occupancy: f64,
    /// Maximum effect at full occupancy
    pub max_effect: f64,
    /// Hill coefficient (cooperativity)
    pub hill_coefficient: f64,
}

impl ReceptorBinding {
    /// Create new receptor binding model
    pub fn new(receptor_type: String, k_on: f64, k_off: f64) -> Self {
        Self {
            receptor_type,
            k_on,
            k_off,
            occupancy: 0.0,
            max_effect: 1.0,
            hill_coefficient: 1.0,
        }
    }

    /// Update receptor occupancy based on ligand concentration
    ///
    /// Uses first-order binding kinetics:
    /// dR/dt = k_on * [L] * (1 - R) - k_off * R
    ///
    /// where R is occupancy and [L] is ligand concentration
    pub fn update(&mut self, dt: f64, ligand_concentration: f64) {
        let binding = self.k_on * ligand_concentration * (1.0 - self.occupancy);
        let unbinding = self.k_off * self.occupancy;

        self.occupancy += (binding - unbinding) * dt;
        self.occupancy = self.occupancy.clamp(0.0, 1.0);
    }

    /// Fractional occupancy this receptor settles at for a steady ligand level.
    ///
    /// Setting `dR/dt = k_on*[L]*(1-R) - k_off*R` to zero gives the standard
    /// binding isotherm `R = [L] / (Kd + [L])`.
    pub fn equilibrium_occupancy(&self, ligand: f64) -> f64 {
        if ligand <= 0.0 {
            return 0.0;
        }
        let kd = self.kd();
        ligand / (kd + ligand)
    }

    /// Jump this receptor straight to its equilibrium occupancy for `ligand`.
    ///
    /// Used when concentration is *set* rather than integrated, so the receptor
    /// state stays consistent with the concentration it is supposed to reflect.
    pub fn equilibrate(&mut self, ligand: f64) {
        self.occupancy = self.equilibrium_occupancy(ligand);
    }

    /// Construct a receptor already at equilibrium for `ligand`.
    ///
    /// A modulator at rest is at its tonic level, not at zero occupancy: every
    /// receptor here is deliberately given `Kd` equal to the tonic baseline, so
    /// resting occupancy is 0.5 by design. Starting at 0.0 instead left a freshly
    /// constructed system in a state it can never return to, and made every
    /// receptor-derived gain read zero until enough steps had been simulated.
    pub fn at_equilibrium(receptor_type: String, k_on: f64, k_off: f64, ligand: f64) -> Self {
        let mut r = Self::new(receptor_type, k_on, k_off);
        r.equilibrate(ligand);
        r
    }

    /// Get the equilibrium dissociation constant (Kd)
    pub fn kd(&self) -> f64 {
        self.k_off / self.k_on
    }

    /// Compute effect based on occupancy using Hill equation
    ///
    /// Effect = max_effect * [L]^n / (Kd^n + [L]^n)
    pub fn compute_effect(&self, ligand_concentration: f64) -> f64 {
        let kd = self.kd();
        let normalized = ligand_concentration.powf(self.hill_coefficient);
        let kd_normalized = kd.powf(self.hill_coefficient);

        self.max_effect * normalized / (kd_normalized + normalized)
    }

    /// Reset receptor occupancy
    pub fn reset(&mut self) {
        self.occupancy = 0.0;
    }
}

/// Tonic (resting) concentration shared by the bundled modulators.
///
/// Every bundled receptor is given `Kd` equal to this value, which puts resting
/// occupancy at exactly 0.5.
pub const TONIC_BASELINE: f64 = 0.1;

/// Dopamine neuromodulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dopamine {
    pub concentration: ModulatorConcentration,
    pub diffusion: DiffusionModel,
    pub d1_receptors: ReceptorBinding,
    pub d2_receptors: ReceptorBinding,
}

impl Default for Dopamine {
    fn default() -> Self {
        Self {
            concentration: ModulatorConcentration::new(0.1, 200.0),
            diffusion: DiffusionModel {
                effective_radius: 100.0,
                ..Default::default()
            },
            d1_receptors: ReceptorBinding::at_equilibrium(
                "D1".to_string(),
                0.01,
                0.001,
                TONIC_BASELINE,
            ),
            d2_receptors: ReceptorBinding::at_equilibrium(
                "D2".to_string(),
                0.02,
                0.002,
                TONIC_BASELINE,
            ),
        }
    }
}

/// Acetylcholine neuromodulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Acetylcholine {
    pub concentration: ModulatorConcentration,
    pub diffusion: DiffusionModel,
    pub nicotinic_receptors: ReceptorBinding,
    pub muscarinic_receptors: ReceptorBinding,
}

impl Default for Acetylcholine {
    fn default() -> Self {
        Self {
            concentration: ModulatorConcentration::new(0.1, 100.0),
            diffusion: DiffusionModel {
                effective_radius: 50.0,
                ..Default::default()
            },
            nicotinic_receptors: ReceptorBinding::at_equilibrium(
                "Nicotinic".to_string(),
                0.05,
                0.005,
                TONIC_BASELINE,
            ),
            muscarinic_receptors: ReceptorBinding::at_equilibrium(
                "Muscarinic".to_string(),
                0.01,
                0.001,
                TONIC_BASELINE,
            ),
        }
    }
}

/// Serotonin neuromodulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Serotonin {
    pub concentration: ModulatorConcentration,
    pub diffusion: DiffusionModel,
    pub receptors_5ht1a: ReceptorBinding,
    pub receptors_5ht2a: ReceptorBinding,
}

impl Default for Serotonin {
    fn default() -> Self {
        Self {
            concentration: ModulatorConcentration::new(0.1, 1000.0),
            diffusion: DiffusionModel {
                effective_radius: 200.0,
                ..Default::default()
            },
            receptors_5ht1a: ReceptorBinding::at_equilibrium(
                "5-HT1A".to_string(),
                0.008,
                0.0008,
                TONIC_BASELINE,
            ),
            receptors_5ht2a: ReceptorBinding::at_equilibrium(
                "5-HT2A".to_string(),
                0.01,
                0.001,
                TONIC_BASELINE,
            ),
        }
    }
}

/// Norepinephrine neuromodulator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Norepinephrine {
    pub concentration: ModulatorConcentration,
    pub diffusion: DiffusionModel,
    pub alpha_receptors: ReceptorBinding,
    pub beta_receptors: ReceptorBinding,
}

impl Default for Norepinephrine {
    fn default() -> Self {
        Self {
            concentration: ModulatorConcentration::new(0.1, 150.0),
            diffusion: DiffusionModel {
                effective_radius: 150.0,
                ..Default::default()
            },
            alpha_receptors: ReceptorBinding::at_equilibrium(
                "Alpha".to_string(),
                0.015,
                0.0015,
                TONIC_BASELINE,
            ),
            beta_receptors: ReceptorBinding::at_equilibrium(
                "Beta".to_string(),
                0.02,
                0.002,
                TONIC_BASELINE,
            ),
        }
    }
}

/// Generic neuromodulator interface
pub struct Neuromodulator {
    pub modulator_type: NeuromodulatorType,
    pub concentration: ModulatorConcentration,
    pub diffusion: DiffusionModel,
    pub receptors: HashMap<String, ReceptorBinding>,
}

impl Neuromodulator {
    /// Create a new neuromodulator of the specified type
    pub fn new(modulator_type: NeuromodulatorType) -> Self {
        let tau = modulator_type.typical_time_constant();
        let radius = modulator_type.typical_diffusion_radius();

        Self {
            modulator_type,
            concentration: ModulatorConcentration::new(0.1, tau),
            diffusion: DiffusionModel {
                effective_radius: radius,
                ..Default::default()
            },
            receptors: HashMap::new(),
        }
    }

    /// Add a receptor type
    pub fn add_receptor(&mut self, name: String, k_on: f64, k_off: f64) {
        self.receptors
            .insert(name.clone(), ReceptorBinding::new(name, k_on, k_off));
    }

    /// Update all receptors based on current concentration
    pub fn update_receptors(&mut self, dt: f64) {
        let concentration = self.concentration.concentration;
        for receptor in self.receptors.values_mut() {
            receptor.update(dt, concentration);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modulator_concentration() {
        let mut conc = ModulatorConcentration::new(0.1, 200.0);
        assert_eq!(conc.concentration, 0.1);

        // Test release
        conc.update(1.0, 0.5);
        assert!(conc.concentration > 0.1);

        // Test decay
        for _ in 0..1000 {
            conc.update(1.0, 0.0);
        }
        assert!((conc.concentration - conc.baseline).abs() < 0.01);
    }

    #[test]
    fn test_diffusion_model() {
        let diffusion = DiffusionModel::default();
        let conc_at_origin = 1.0;
        let conc_at_50 = diffusion.concentration_at_distance(conc_at_origin, 50.0);
        let conc_at_100 = diffusion.concentration_at_distance(conc_at_origin, 100.0);

        assert!(conc_at_50 > conc_at_100);
        assert!(conc_at_50 < conc_at_origin);
    }

    #[test]
    fn test_receptor_binding() {
        let mut receptor = ReceptorBinding::new("Test".to_string(), 0.01, 0.001);
        let kd = receptor.kd();
        assert_eq!(kd, 0.1);

        // Test binding dynamics
        receptor.update(1.0, 1.0);
        assert!(receptor.occupancy > 0.0);

        // Test effect computation
        let effect = receptor.compute_effect(1.0);
        assert!(effect > 0.0 && effect <= 1.0);
    }

    #[test]
    fn test_dopamine_default() {
        let da = Dopamine::default();
        assert_eq!(da.concentration.baseline, 0.1);
        assert_eq!(da.d1_receptors.receptor_type, "D1");
        assert_eq!(da.d2_receptors.receptor_type, "D2");
    }

    #[test]
    fn test_neuromodulator_types() {
        assert_eq!(NeuromodulatorType::Dopamine.typical_time_constant(), 200.0);
        assert_eq!(
            NeuromodulatorType::Acetylcholine.typical_time_constant(),
            100.0
        );
        assert!(NeuromodulatorType::Dopamine.typical_diffusion_radius() > 0.0);
    }

    #[test]
    fn test_neuromodulator() {
        let mut nm = Neuromodulator::new(NeuromodulatorType::Dopamine);
        nm.add_receptor("D1".to_string(), 0.01, 0.001);
        nm.add_receptor("D2".to_string(), 0.02, 0.002);

        assert_eq!(nm.receptors.len(), 2);
        nm.update_receptors(1.0);
    }
}
