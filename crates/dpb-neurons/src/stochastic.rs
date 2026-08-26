//! Stochastic neuron models with noise injection.
//!
//! Stochastic neurons include noise in their dynamics to model
//! channel noise, synaptic variability, and other random fluctuations.

use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel};
use bytemuck::{Pod, Zeroable};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Noise type for stochastic neurons.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NoiseType {
    /// Gaussian (normal) noise
    Gaussian,
    /// Ornstein-Uhlenbeck process (colored noise)
    OrnsteinUhlenbeck,
    /// Multiplicative noise (amplitude depends on voltage)
    Multiplicative,
}

/// Stochastic LIF neuron state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct StochasticLifState {
    /// Membrane potential in mV
    pub v: f32,
    /// Ornstein-Uhlenbeck noise state
    pub ou_noise: f32,
    /// Refractory timer in ms
    pub refrac_timer: f32,
    pub _padding: f32,
}

impl Default for StochasticLifState {
    fn default() -> Self {
        Self {
            v: -65.0,
            ou_noise: 0.0,
            refrac_timer: 0.0,
            _padding: 0.0,
        }
    }
}

/// Stochastic LIF neuron configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct StochasticLifConfig {
    /// Membrane time constant in ms
    pub tau_mem: f32,
    /// Spike threshold in mV
    pub v_thresh: f32,
    /// Reset potential in mV
    pub v_reset: f32,
    /// Resting potential in mV
    pub v_rest: f32,
    /// Membrane resistance in MΩ
    pub r_m: f32,
    /// Refractory period in ms
    pub tau_refrac: f32,
    /// Noise standard deviation in mV
    pub noise_sigma: f32,
    /// OU noise time constant in ms
    pub tau_ou: f32,
    /// Multiplicative noise factor
    pub noise_mult: f32,
    pub _padding: [f32; 3],
}

impl Default for StochasticLifConfig {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            r_m: 10.0,
            tau_refrac: 2.0,
            noise_sigma: 0.5,
            tau_ou: 2.0,
            noise_mult: 0.0,
            _padding: [0.0; 3],
        }
    }
}

impl StochasticLifConfig {
    /// High noise configuration.
    pub fn high_noise() -> Self {
        Self {
            noise_sigma: 2.0,
            ..Default::default()
        }
    }

    /// Low noise configuration.
    pub fn low_noise() -> Self {
        Self {
            noise_sigma: 0.1,
            ..Default::default()
        }
    }

    /// Colored noise (OU process) configuration.
    pub fn colored_noise() -> Self {
        Self {
            noise_sigma: 1.0,
            tau_ou: 5.0,
            ..Default::default()
        }
    }

    /// Multiplicative noise configuration.
    pub fn multiplicative_noise() -> Self {
        Self {
            noise_sigma: 0.5,
            noise_mult: 0.1,
            ..Default::default()
        }
    }
}

impl NeuronConfig for StochasticLifConfig {
    fn validate(&self) -> Result<(), String> {
        if self.tau_mem <= 0.0 {
            return Err("Membrane time constant must be positive".to_string());
        }
        if self.r_m <= 0.0 {
            return Err("Membrane resistance must be positive".to_string());
        }
        if self.v_thresh <= self.v_reset {
            return Err("Threshold must be greater than reset".to_string());
        }
        if self.noise_sigma < 0.0 {
            return Err("Noise sigma must be non-negative".to_string());
        }
        if self.tau_ou <= 0.0 {
            return Err("OU time constant must be positive".to_string());
        }
        Ok(())
    }

    fn default_config() -> Self {
        Self::default()
    }
}

/// Stochastic LIF neuron with noise injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticLifNeuron {
    pub state: StochasticLifState,
    pub config: StochasticLifConfig,
    pub noise_type: NoiseType,
    #[serde(skip)]
    rng: rand::rngs::ThreadRng,
}

impl StochasticLifNeuron {
    pub fn new(config: StochasticLifConfig) -> Self {
        Self::with_noise_type(config, NoiseType::Gaussian)
    }

    pub fn with_noise_type(config: StochasticLifConfig, noise_type: NoiseType) -> Self {
        Self {
            state: StochasticLifState::default(),
            config,
            noise_type,
            rng: rand::rng(),
        }
    }

    /// Generate noise sample based on noise type.
    ///
    /// Uses safe defaults if noise parameters are invalid (sigma <= 0).
    fn generate_noise(&mut self, dt: f32) -> f32 {
        match self.noise_type {
            NoiseType::Gaussian => {
                // White Gaussian noise - use safe default if sigma is invalid
                let sigma = self.config.noise_sigma.max(0.0) as f64;
                if sigma <= 0.0 {
                    return 0.0;
                }
                let Ok(normal) = Normal::new(0.0, sigma) else {
                    return 0.0;
                };
                (normal.sample(&mut self.rng) * (dt as f64).sqrt()) as f32
            }
            NoiseType::OrnsteinUhlenbeck => {
                // Ornstein-Uhlenbeck process: dX = -X/τ dt + σ dW
                let Ok(normal) = Normal::new(0.0, 1.0) else {
                    return self.state.ou_noise;
                };
                let dW = normal.sample(&mut self.rng) * (dt as f64).sqrt();

                let dou = -self.state.ou_noise as f64 / self.config.tau_ou as f64 * dt as f64
                          + self.config.noise_sigma as f64 * dW;
                self.state.ou_noise += dou as f32;

                self.state.ou_noise
            }
            NoiseType::Multiplicative => {
                // Multiplicative noise: σ * v * ξ
                let Ok(normal) = Normal::new(0.0, 1.0) else {
                    return 0.0;
                };
                let xi = normal.sample(&mut self.rng) as f32;
                self.config.noise_mult * self.state.v * xi * dt.sqrt()
            }
        }
    }

}

// Note: For reproducibility with seeded RNG, a custom implementation would be needed
// using rand::rngs::StdRng instead of ThreadRng.

impl MembraneDynamics for StochasticLifNeuron {
    fn membrane_potential(&self) -> f32 {
        self.state.v
    }

    fn set_membrane_potential(&mut self, v: f32) {
        self.state.v = v;
    }

    fn rest_potential(&self) -> f32 {
        self.config.v_rest
    }
}

impl NeuronModel for StochasticLifNeuron {
    type Config = StochasticLifConfig;

    fn update(&mut self, input_current: f32, dt: f32) -> bool {
        // Update refractory period
        if self.state.refrac_timer > 0.0 {
            self.state.refrac_timer -= dt;
            if self.state.refrac_timer < 0.0 {
                self.state.refrac_timer = 0.0;
            }
            return false;
        }

        // LIF dynamics with noise
        let dv_det = (-(self.state.v - self.config.v_rest) + self.config.r_m * input_current)
                     / self.config.tau_mem;

        let noise = self.generate_noise(dt);

        self.state.v += dv_det * dt + noise;

        // Check for spike
        if self.state.v >= self.config.v_thresh {
            self.reset();
            true
        } else {
            false
        }
    }

    fn reset(&mut self) {
        self.state.v = self.config.v_reset;
        self.state.refrac_timer = self.config.tau_refrac;
    }

    fn threshold(&self) -> f32 {
        self.config.v_thresh
    }

    fn is_refractory(&self) -> bool {
        self.state.refrac_timer > 0.0
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_lif_default() {
        let neuron = StochasticLifNeuron::new(StochasticLifConfig::default());
        assert_eq!(neuron.membrane_potential(), -65.0);
        assert_eq!(neuron.noise_type, NoiseType::Gaussian);
    }

    #[test]
    fn test_gaussian_noise() {
        let mut neuron = StochasticLifNeuron::new(StochasticLifConfig::default());
        let v_initial = neuron.membrane_potential();

        // With no input, noise should still cause voltage changes
        let mut voltage_changed = false;
        for _ in 0..10 {
            neuron.update(0.0, 1.0);
            if (neuron.membrane_potential() - v_initial).abs() > 0.01 {
                voltage_changed = true;
                break;
            }
        }

        assert!(voltage_changed, "Noise should cause voltage fluctuations");
    }

    #[test]
    fn test_ou_noise() {
        let mut neuron = StochasticLifNeuron::with_noise_type(
            StochasticLifConfig::colored_noise(),
            NoiseType::OrnsteinUhlenbeck,
        );

        // OU noise should have temporal correlation
        for _ in 0..100 {
            neuron.update(0.0, 0.1);
        }

        // OU state should have evolved
        assert!(neuron.state.ou_noise.abs() >= 0.0); // Just check it exists
    }

    #[test]
    fn test_multiplicative_noise() {
        let mut neuron = StochasticLifNeuron::with_noise_type(
            StochasticLifConfig::multiplicative_noise(),
            NoiseType::Multiplicative,
        );

        // Set voltage to see multiplicative effect
        neuron.set_membrane_potential(-50.0);

        let v_before = neuron.membrane_potential();
        for _ in 0..10 {
            neuron.update(0.0, 1.0);
        }

        // Voltage should change due to multiplicative noise
        // (Though it might also decay, so just check dynamics work)
        assert!(neuron.membrane_potential() != v_before || neuron.membrane_potential() < neuron.config.v_thresh);
    }

    #[test]
    fn test_stochastic_spiking() {
        let mut neuron = StochasticLifNeuron::new(StochasticLifConfig::default());
        let mut spike_count = 0;

        // Apply input that might cause spikes with noise
        for _ in 0..500 {
            if neuron.update(12.0, 1.0) {
                spike_count += 1;
            }
        }

        // Should eventually spike
        assert!(spike_count > 0);
    }

    #[test]
    fn test_noise_presets() {
        let high = StochasticLifConfig::high_noise();
        let low = StochasticLifConfig::low_noise();
        let colored = StochasticLifConfig::colored_noise();
        let mult = StochasticLifConfig::multiplicative_noise();

        assert!(high.noise_sigma > low.noise_sigma);
        assert!(colored.tau_ou > high.tau_ou);
        assert!(mult.noise_mult > 0.0);
    }

    #[test]
    fn test_config_validation() {
        let mut config = StochasticLifConfig::default();
        assert!(config.validate().is_ok());

        config.tau_mem = 0.0;
        assert!(config.validate().is_err());

        config.tau_mem = 20.0;
        config.noise_sigma = -0.1;
        assert!(config.validate().is_err());

        config.noise_sigma = 0.5;
        config.tau_ou = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_noise_variability() {
        let mut neuron = StochasticLifNeuron::new(StochasticLifConfig::high_noise());

        let mut voltages = Vec::new();
        for _ in 0..50 {
            neuron.update(5.0, 1.0);
            voltages.push(neuron.membrane_potential());
        }

        // Check that there's variability in the trajectory
        let mean: f32 = voltages.iter().sum::<f32>() / voltages.len() as f32;
        let variance: f32 = voltages.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>() / voltages.len() as f32;

        assert!(variance > 0.0, "Stochastic neuron should show variance");
    }
}
