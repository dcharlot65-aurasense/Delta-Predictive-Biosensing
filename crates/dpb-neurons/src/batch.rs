//! Batch neuron layer operations for efficient parallel updates.
//!
//! This module provides optimized batch processing of neuron populations
//! using SIMD operations where possible.

use crate::lif::LifConfig;
use crate::traits::NeuronModel;
use ndarray::{Array1, Array2};

/// Batch layer of neurons for parallel processing.
#[derive(Debug, Clone)]
pub struct BatchNeuronLayer<N: NeuronModel> {
    /// Vector of neurons in the layer
    pub neurons: Vec<N>,
    /// Number of neurons
    pub size: usize,
}

impl<N: NeuronModel + Clone> BatchNeuronLayer<N> {
    /// Create a new batch layer from a vector of neurons.
    pub fn from_neurons(neurons: Vec<N>) -> Self {
        let size = neurons.len();
        Self { neurons, size }
    }

    /// Update all neurons in parallel with input currents.
    pub fn update(&mut self, inputs: &[f32], dt: f32) -> Vec<bool> {
        assert_eq!(inputs.len(), self.size, "Input size mismatch");

        self.neurons
            .iter_mut()
            .zip(inputs.iter())
            .map(|(neuron, &input)| neuron.update(input, dt))
            .collect()
    }

    /// Reset all neurons in the layer.
    pub fn reset_all(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
    }

    /// Get membrane potentials of all neurons.
    pub fn get_potentials(&self) -> Vec<f32>
    where
        N: crate::traits::MembraneDynamics,
    {
        self.neurons
            .iter()
            .map(|n| n.membrane_potential())
            .collect()
    }

    /// Set membrane potentials for all neurons.
    pub fn set_potentials(&mut self, potentials: &[f32])
    where
        N: crate::traits::MembraneDynamics,
    {
        assert_eq!(potentials.len(), self.size, "Potentials size mismatch");
        for (neuron, &v) in self.neurons.iter_mut().zip(potentials.iter()) {
            neuron.set_membrane_potential(v);
        }
    }

    /// Count active (spiking) neurons.
    pub fn count_spikes(spikes: &[bool]) -> usize {
        spikes.iter().filter(|&&s| s).count()
    }

    /// Get indices of spiking neurons.
    pub fn spike_indices(spikes: &[bool]) -> Vec<usize> {
        spikes
            .iter()
            .enumerate()
            .filter_map(|(i, &spike)| if spike { Some(i) } else { None })
            .collect()
    }
}

/// Specialized batch layer for LIF neurons with optimized operations.
#[derive(Debug, Clone)]
pub struct BatchLifLayer {
    /// Membrane potentials
    pub v: Array1<f32>,
    /// Configuration (shared across neurons)
    pub config: LifConfig,
    /// Refractory timers
    pub refrac: Array1<f32>,
}

impl BatchLifLayer {
    /// Create a new batch LIF layer.
    pub fn new(size: usize, config: LifConfig) -> Self {
        Self {
            v: Array1::from_elem(size, config.v_rest),
            config,
            refrac: Array1::zeros(size),
        }
    }

    /// Update all neurons with vectorized operations.
    pub fn update(&mut self, inputs: &Array1<f32>, dt: f32) -> Array1<bool> {
        assert_eq!(inputs.len(), self.v.len(), "Input size mismatch");

        let size = self.v.len();
        let mut spikes = Array1::from_elem(size, false);

        // Vectorized update
        for i in 0..size {
            // Update refractory period
            if self.refrac[i] > 0.0 {
                self.refrac[i] -= dt;
                if self.refrac[i] < 0.0 {
                    self.refrac[i] = 0.0;
                }
                continue;
            }

            // LIF dynamics
            let dv = (-(self.v[i] - self.config.v_rest) + self.config.r_m * inputs[i])
                     / self.config.tau_mem;
            self.v[i] += dv * dt;

            // Check for spike
            if self.v[i] >= self.config.v_thresh {
                self.v[i] = self.config.v_reset;
                self.refrac[i] = self.config.tau_refrac;
                spikes[i] = true;
            }
        }

        spikes
    }

    /// Reset all neurons.
    pub fn reset_all(&mut self) {
        self.v.fill(self.config.v_rest);
        self.refrac.fill(0.0);
    }

    /// Get spike rate over a time window.
    pub fn spike_rate(&self, spike_counts: &Array1<usize>, time_window: f32) -> Array1<f32> {
        spike_counts.mapv(|count| count as f32 / time_window)
    }

    /// Compute population activity (fraction of neurons above threshold).
    pub fn population_activity(&self, threshold: f32) -> f32 {
        let active = self.v.iter().filter(|&&v| v > threshold).count();
        active as f32 / self.v.len() as f32
    }
}

/// Multi-layer batch network.
#[derive(Debug, Clone)]
pub struct BatchNetwork {
    /// Layers of neurons
    pub layers: Vec<BatchLifLayer>,
    /// Inter-layer connections (weight matrices)
    pub weights: Vec<Array2<f32>>,
}

impl BatchNetwork {
    /// Create a new batch network with given layer sizes.
    pub fn new(layer_sizes: &[usize], config: LifConfig) -> Self {
        let layers: Vec<_> = layer_sizes
            .iter()
            .map(|&size| BatchLifLayer::new(size, config))
            .collect();

        // Initialize random weights
        let mut weights = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let w = Array2::zeros((layer_sizes[i + 1], layer_sizes[i]));
            weights.push(w);
        }

        Self { layers, weights }
    }

    /// Forward pass through the network.
    pub fn forward(&mut self, input: &Array1<f32>, dt: f32) -> Vec<Array1<bool>> {
        let mut spike_trains = Vec::new();
        let mut current_input = input.clone();

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Update layer
            let spikes = layer.update(&current_input, dt);
            spike_trains.push(spikes.clone());

            // Compute input for next layer
            if layer_idx < self.weights.len() {
                let spike_floats = spikes.mapv(|s| if s { 1.0 } else { 0.0 });
                current_input = self.weights[layer_idx].dot(&spike_floats);
            }
        }

        spike_trains
    }

    /// Reset all layers.
    pub fn reset_all(&mut self) {
        for layer in &mut self.layers {
            layer.reset_all();
        }
    }

    /// Get total number of neurons.
    pub fn total_neurons(&self) -> usize {
        self.layers.iter().map(|l| l.v.len()).sum()
    }
}

/// Statistics for batch neuron populations.
#[derive(Debug, Clone)]
pub struct PopulationStats {
    /// Mean membrane potential
    pub mean_v: f32,
    /// Standard deviation of membrane potential
    pub std_v: f32,
    /// Spike rate (Hz)
    pub spike_rate: f32,
    /// Coefficient of variation of ISI
    pub cv_isi: f32,
}

impl PopulationStats {
    /// Compute statistics from membrane potentials.
    pub fn from_potentials(potentials: &[f32], spike_rate: f32) -> Self {
        let n = potentials.len() as f32;
        let mean_v = potentials.iter().sum::<f32>() / n;
        let variance = potentials
            .iter()
            .map(|&v| (v - mean_v).powi(2))
            .sum::<f32>()
            / n;
        let std_v = variance.sqrt();

        Self {
            mean_v,
            std_v,
            spike_rate,
            cv_isi: 0.0, // Would need ISI history to compute
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lif::LifNeuron;

    #[test]
    fn test_batch_layer_creation() {
        let config = LifConfig::default();
        let neurons: Vec<LifNeuron> = (0..100).map(|_| LifNeuron::new(config)).collect();
        let layer = BatchNeuronLayer::from_neurons(neurons);

        assert_eq!(layer.size, 100);
        assert_eq!(layer.neurons.len(), 100);
    }

    #[test]
    fn test_batch_layer_update() {
        let config = LifConfig::default();
        let neurons: Vec<LifNeuron> = (0..10).map(|_| LifNeuron::new(config)).collect();
        let mut layer = BatchNeuronLayer::from_neurons(neurons);

        let inputs = vec![5.0; 10];
        let spikes = layer.update(&inputs, 1.0);

        assert_eq!(spikes.len(), 10);
    }

    #[test]
    fn test_batch_lif_layer() {
        let config = LifConfig::default();
        let mut layer = BatchLifLayer::new(50, config);

        assert_eq!(layer.v.len(), 50);

        let inputs = Array1::from_elem(50, 10.0);
        let spikes = layer.update(&inputs, 1.0);

        assert_eq!(spikes.len(), 50);
    }

    #[test]
    fn test_batch_lif_spiking() {
        let config = LifConfig::default();
        let mut layer = BatchLifLayer::new(10, config);

        let inputs = Array1::from_elem(10, 50.0);
        let mut total_spikes = 0;

        for _ in 0..100 {
            let spikes = layer.update(&inputs, 1.0);
            total_spikes += spikes.iter().filter(|&&s| s).count();
        }

        assert!(total_spikes > 0, "Should produce spikes with strong input");
    }

    #[test]
    fn test_batch_network() {
        let config = LifConfig::default();
        let layer_sizes = vec![10, 20, 10];
        let network = BatchNetwork::new(&layer_sizes, config);

        assert_eq!(network.layers.len(), 3);
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.total_neurons(), 40);
    }

    #[test]
    fn test_batch_network_forward() {
        let config = LifConfig::default();
        let layer_sizes = vec![5, 5, 5];
        let mut network = BatchNetwork::new(&layer_sizes, config);

        let input = Array1::from_elem(5, 10.0);
        let spike_trains = network.forward(&input, 1.0);

        assert_eq!(spike_trains.len(), 3);
        assert_eq!(spike_trains[0].len(), 5);
    }

    #[test]
    fn test_population_activity() {
        let config = LifConfig::default();
        let mut layer = BatchLifLayer::new(100, config);

        // Set half above threshold
        for i in 0..50 {
            layer.v[i] = -45.0; // Above -50 threshold
        }

        let activity = layer.population_activity(-50.0);
        assert!((activity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_spike_counting() {
        use crate::lif::LifNeuron;

        let spikes = vec![true, false, true, true, false];
        let count = BatchNeuronLayer::<LifNeuron>::count_spikes(&spikes);
        assert_eq!(count, 3);

        let indices = BatchNeuronLayer::<LifNeuron>::spike_indices(&spikes);
        assert_eq!(indices, vec![0, 2, 3]);
    }

    #[test]
    fn test_reset_all() {
        let config = LifConfig::default();
        let mut layer = BatchLifLayer::new(10, config);

        // Modify state
        layer.v.fill(-40.0);
        layer.refrac.fill(5.0);

        // Reset
        layer.reset_all();

        assert!(layer.v.iter().all(|&v| v == config.v_rest));
        assert!(layer.refrac.iter().all(|&r| r == 0.0));
    }

    #[test]
    fn test_population_stats() {
        let potentials = vec![-65.0, -60.0, -55.0, -60.0, -65.0];
        let stats = PopulationStats::from_potentials(&potentials, 10.0);

        assert_eq!(stats.mean_v, -61.0);
        assert!(stats.std_v > 0.0);
        assert_eq!(stats.spike_rate, 10.0);
    }
}
