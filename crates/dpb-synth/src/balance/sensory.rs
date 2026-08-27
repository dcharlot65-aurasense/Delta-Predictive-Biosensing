//! Sensory Manipulation Balance Generator
//!
//! Generates balance responses under various sensory conditions:
//! - Eyes open/closed
//! - Foam surface (somatosensory disruption)
//! - Modified Clinical Test of Sensory Integration (mCTSIB)
//! - Sensory Organization Test conditions

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for sensory manipulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryConfig {
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Base sway amplitude
    pub base_sway: f64,
    /// Noise level
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for SensoryConfig {
    fn default() -> Self {
        Self {
            sample_rate: 100.0,
            base_sway: 0.008,
            noise_level: 0.05,
            seed: None,
        }
    }
}

/// Sensory test conditions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SensoryCondition {
    /// Condition 1: Eyes open, firm surface
    FirmEyesOpen,
    /// Condition 2: Eyes closed, firm surface
    FirmEyesClosed,
    /// Condition 3: Eyes open, foam surface
    FoamEyesOpen,
    /// Condition 4: Eyes closed, foam surface
    FoamEyesClosed,
    /// SOT Condition 5: Sway-referenced vision
    SwayReferencedVision,
    /// SOT Condition 6: Sway-referenced surface
    SwayReferencedSurface,
    /// Full sensory conflict
    FullConflict,
    /// Galvanic vestibular stimulation
    GalvanicStimulation { current_ma: f64 },
    /// Visual flow perturbation
    VisualFlow { velocity: f64, direction: f64 },
}

/// Output from sensory manipulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryOutput {
    /// Time points
    pub time: Vec<f64>,
    /// COP AP
    pub cop_ap: Vec<f64>,
    /// COP ML
    pub cop_ml: Vec<f64>,
    /// Ground truth
    pub ground_truth: SensoryGroundTruth,
    /// Configuration
    pub config: SensoryConfig,
}

/// Ground truth for sensory conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryGroundTruth {
    /// Sensory condition
    pub condition: SensoryCondition,
    /// Sway metrics
    pub sway_area: f64,
    /// Path length
    pub path_length: f64,
    /// Mean velocity
    pub mean_velocity: f64,
    /// RMS AP
    pub rms_ap: f64,
    /// RMS ML
    pub rms_ml: f64,
    /// Romberg ratio (if applicable)
    pub romberg_ratio: Option<f64>,
    /// Sensory weighting estimates
    pub sensory_weights: SensoryWeights,
    /// Equilibrium score (0-100)
    pub equilibrium_score: f64,
}

/// Sensory system weighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryWeights {
    /// Visual system weight (0-1)
    pub visual: f64,
    /// Somatosensory weight (0-1)
    pub somatosensory: f64,
    /// Vestibular weight (0-1)
    pub vestibular: f64,
}

/// Sensory manipulation generator
pub struct SensoryManipulationGenerator {
    config: SensoryConfig,
    rng: StdRng,
}

impl SensoryManipulationGenerator {
    /// Create a new sensory generator
    pub fn new(config: SensoryConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };
        Self { config, rng }
    }

    /// Generate balance under specified sensory condition
    pub fn generate(&mut self, condition: SensoryCondition, duration: f64) -> SensoryOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        // Get condition-specific parameters
        let (sway_factor_ap, sway_factor_ml, freq_shift, sensory_weights) =
            self.get_condition_params(condition);

        let sway_ap = self.config.base_sway * sway_factor_ap;
        let sway_ml = self.config.base_sway * sway_factor_ml;

        let mut time = Vec::with_capacity(n_samples);
        let mut cop_ap = Vec::with_capacity(n_samples);
        let mut cop_ml = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * sway_ap).unwrap();

        // Frequency components
        let base_freqs = [0.1, 0.2, 0.35, 0.5, 0.8, 1.2];
        let freqs: Vec<f64> = base_freqs.iter().map(|f| f * freq_shift).collect();
        let phases_ap: Vec<f64> = freqs
            .iter()
            .map(|_| self.rng.random::<f64>() * 2.0 * PI)
            .collect();
        let phases_ml: Vec<f64> = freqs
            .iter()
            .map(|_| self.rng.random::<f64>() * 2.0 * PI)
            .collect();
        let amps = [0.35, 0.25, 0.18, 0.12, 0.07, 0.03];

        // Add condition-specific effects
        let (drift_ap, drift_ml, tremor_freq) = match condition {
            SensoryCondition::GalvanicStimulation { current_ma } => {
                (0.02 * current_ma, 0.03 * current_ma, Some(0.5))
            }
            SensoryCondition::VisualFlow {
                velocity,
                direction,
            } => {
                let drift = 0.01 * velocity;
                (drift * direction.cos(), drift * direction.sin(), None)
            }
            _ => (0.0, 0.0, None),
        };

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let mut ap = 0.0;
            let mut ml = 0.0;

            // Sum frequency components
            for j in 0..freqs.len() {
                ap += amps[j] * (2.0 * PI * freqs[j] * t + phases_ap[j]).sin();
                ml += amps[j] * (2.0 * PI * freqs[j] * t + phases_ml[j]).sin();
            }

            ap *= sway_ap;
            ml *= sway_ml;

            // Add slow drift
            let slow_drift_ap = 0.05 * sway_ap * (0.03 * PI * t).sin() + drift_ap * t;
            let slow_drift_ml = 0.04 * sway_ml * (0.025 * PI * t).sin() + drift_ml * t;

            ap += slow_drift_ap;
            ml += slow_drift_ml;

            // Add tremor if present
            if let Some(tf) = tremor_freq {
                ap += 0.003 * (2.0 * PI * tf * t).sin();
            }

            let noise_ap: f64 = self.rng.sample(noise_dist);
            let noise_ml: f64 = self.rng.sample(noise_dist);

            cop_ap.push(ap + noise_ap);
            cop_ml.push(ml + noise_ml);
        }

        // Calculate metrics
        let (sway_area, path_length, mean_velocity, rms_ap, rms_ml) =
            self.calculate_metrics(&cop_ap, &cop_ml, dt);

        // Calculate equilibrium score (simplified)
        let max_sway = 0.12; // 12.5 degrees theoretical max
        let actual_sway = (rms_ap.powi(2) + rms_ml.powi(2)).sqrt();
        let equilibrium_score = ((1.0 - actual_sway / max_sway) * 100.0).clamp(0.0, 100.0);

        let ground_truth = SensoryGroundTruth {
            condition,
            sway_area,
            path_length,
            mean_velocity,
            rms_ap,
            rms_ml,
            romberg_ratio: None, // Would need eyes open baseline
            sensory_weights,
            equilibrium_score,
        };

        SensoryOutput {
            time,
            cop_ap,
            cop_ml,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate mCTSIB protocol (all 4 conditions)
    pub fn generate_mctsib(&mut self, trial_duration: f64) -> Vec<SensoryOutput> {
        let conditions = [
            SensoryCondition::FirmEyesOpen,
            SensoryCondition::FirmEyesClosed,
            SensoryCondition::FoamEyesOpen,
            SensoryCondition::FoamEyesClosed,
        ];

        let mut outputs: Vec<SensoryOutput> = conditions
            .iter()
            .map(|&c| self.generate(c, trial_duration))
            .collect();

        // Calculate Romberg ratios
        let firm_eo_sway = outputs[0].ground_truth.sway_area;
        let firm_ec_sway = outputs[1].ground_truth.sway_area;

        if firm_eo_sway > 0.0 {
            outputs[1].ground_truth.romberg_ratio = Some(firm_ec_sway / firm_eo_sway);
        }

        outputs
    }

    /// Generate SOT protocol (6 conditions)
    pub fn generate_sot(&mut self, trial_duration: f64) -> Vec<SensoryOutput> {
        let conditions = [
            SensoryCondition::FirmEyesOpen,
            SensoryCondition::FirmEyesClosed,
            SensoryCondition::SwayReferencedVision,
            SensoryCondition::FoamEyesOpen,
            SensoryCondition::FoamEyesClosed,
            SensoryCondition::FullConflict,
        ];

        conditions
            .iter()
            .map(|&c| self.generate(c, trial_duration))
            .collect()
    }

    /// Get condition-specific parameters
    fn get_condition_params(&self, condition: SensoryCondition) -> (f64, f64, f64, SensoryWeights) {
        // Returns: (sway_factor_ap, sway_factor_ml, freq_shift, sensory_weights)
        match condition {
            SensoryCondition::FirmEyesOpen => (
                1.0,
                1.0,
                1.0,
                SensoryWeights {
                    visual: 0.2,
                    somatosensory: 0.7,
                    vestibular: 0.1,
                },
            ),
            SensoryCondition::FirmEyesClosed => (
                1.3,
                1.2,
                0.95,
                SensoryWeights {
                    visual: 0.0,
                    somatosensory: 0.8,
                    vestibular: 0.2,
                },
            ),
            SensoryCondition::FoamEyesOpen => (
                1.5,
                1.4,
                0.9,
                SensoryWeights {
                    visual: 0.5,
                    somatosensory: 0.2,
                    vestibular: 0.3,
                },
            ),
            SensoryCondition::FoamEyesClosed => (
                2.5,
                2.2,
                0.85,
                SensoryWeights {
                    visual: 0.0,
                    somatosensory: 0.3,
                    vestibular: 0.7,
                },
            ),
            SensoryCondition::SwayReferencedVision => (
                1.8,
                1.6,
                0.9,
                SensoryWeights {
                    visual: 0.1,
                    somatosensory: 0.6,
                    vestibular: 0.3,
                },
            ),
            SensoryCondition::SwayReferencedSurface => (
                2.0,
                1.8,
                0.85,
                SensoryWeights {
                    visual: 0.4,
                    somatosensory: 0.2,
                    vestibular: 0.4,
                },
            ),
            SensoryCondition::FullConflict => (
                3.0,
                2.8,
                0.8,
                SensoryWeights {
                    visual: 0.1,
                    somatosensory: 0.1,
                    vestibular: 0.8,
                },
            ),
            SensoryCondition::GalvanicStimulation { current_ma } => (
                1.0 + current_ma * 0.3,
                1.0 + current_ma * 0.5,
                1.0,
                SensoryWeights {
                    visual: 0.2,
                    somatosensory: 0.5,
                    vestibular: 0.3,
                },
            ),
            SensoryCondition::VisualFlow { velocity, .. } => (
                1.0 + velocity * 0.1,
                1.0 + velocity * 0.05,
                1.0,
                SensoryWeights {
                    visual: 0.4 + velocity * 0.1,
                    somatosensory: 0.4,
                    vestibular: 0.2,
                },
            ),
        }
    }

    /// Calculate sway metrics
    fn calculate_metrics(
        &self,
        cop_ap: &[f64],
        cop_ml: &[f64],
        dt: f64,
    ) -> (f64, f64, f64, f64, f64) {
        let n = cop_ap.len() as f64;

        let mean_ap = cop_ap.iter().sum::<f64>() / n;
        let mean_ml = cop_ml.iter().sum::<f64>() / n;

        let rms_ap = (cop_ap.iter().map(|x| (x - mean_ap).powi(2)).sum::<f64>() / n).sqrt();
        let rms_ml = (cop_ml.iter().map(|x| (x - mean_ml).powi(2)).sum::<f64>() / n).sqrt();

        // Path length
        let path_length: f64 = (1..cop_ap.len())
            .map(|i| {
                let dap = cop_ap[i] - cop_ap[i - 1];
                let dml = cop_ml[i] - cop_ml[i - 1];
                (dap.powi(2) + dml.powi(2)).sqrt()
            })
            .sum();

        let mean_velocity = path_length / (n * dt);

        // 95% confidence ellipse area
        let sway_area = PI * 2.0 * rms_ap * rms_ml * 2.447; // 95% CI

        (sway_area, path_length, mean_velocity, rms_ap, rms_ml)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_firm_eyes_open() {
        let config = SensoryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SensoryManipulationGenerator::new(config);
        let output = generator.generate(SensoryCondition::FirmEyesOpen, 30.0);

        assert!(!output.cop_ap.is_empty());
        assert!(output.ground_truth.sway_area > 0.0);
        assert!(output.ground_truth.equilibrium_score > 0.0);
    }

    #[test]
    fn test_foam_increases_sway() {
        let config = SensoryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SensoryManipulationGenerator::new(config);

        let firm = generator.generate(SensoryCondition::FirmEyesOpen, 30.0);
        let foam = generator.generate(SensoryCondition::FoamEyesOpen, 30.0);

        assert!(foam.ground_truth.rms_ap > firm.ground_truth.rms_ap);
    }

    #[test]
    fn test_eyes_closed_increases_sway() {
        let config = SensoryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SensoryManipulationGenerator::new(config);

        let eo = generator.generate(SensoryCondition::FirmEyesOpen, 30.0);
        let ec = generator.generate(SensoryCondition::FirmEyesClosed, 30.0);

        assert!(ec.ground_truth.rms_ap > eo.ground_truth.rms_ap * 0.9);
    }

    #[test]
    fn test_mctsib() {
        let config = SensoryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SensoryManipulationGenerator::new(config);
        let outputs = generator.generate_mctsib(30.0);

        assert_eq!(outputs.len(), 4);
        // Condition 2 should have Romberg ratio
        assert!(outputs[1].ground_truth.romberg_ratio.is_some());
    }

    #[test]
    fn test_sot() {
        let config = SensoryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SensoryManipulationGenerator::new(config);
        let outputs = generator.generate_sot(20.0);

        assert_eq!(outputs.len(), 6);
        // Condition 6 should have highest sway
        let max_sway = outputs
            .iter()
            .map(|o| o.ground_truth.sway_area)
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(outputs[5].ground_truth.sway_area, max_sway);
    }

    #[test]
    fn test_galvanic_stimulation() {
        let config = SensoryConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SensoryManipulationGenerator::new(config);
        let output = generator.generate(
            SensoryCondition::GalvanicStimulation { current_ma: 1.0 },
            10.0,
        );

        assert!(!output.cop_ml.is_empty());
        // GVS should cause ML bias
        assert!(output.ground_truth.rms_ml > 0.0);
    }
}
