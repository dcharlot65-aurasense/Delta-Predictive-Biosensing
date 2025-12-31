//! Reservoir Computing: Echo State Networks and Liquid State Machines
//!
//! This module implements reservoir computing approaches where a fixed, randomly
//! initialized recurrent network (the "reservoir") transforms input into a high-
//! dimensional space, and only the output layer is trained.

use crate::lif::{LifNeuron, LifConfig};
use crate::traits::{MembraneDynamics, NeuronModel};
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

/// Sparsity pattern for reservoir connectivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityPattern {
    /// Indices of non-zero connections (pre_idx, post_idx, weight)
    pub connections: Vec<(usize, usize, f64)>,
    /// Number of pre-synaptic neurons
    pub n_pre: usize,
    /// Number of post-synaptic neurons
    pub n_post: usize,
}

impl SparsityPattern {
    /// Create a new sparse connectivity pattern with random connections
    pub fn random(n_pre: usize, n_post: usize, sparsity: f64, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let uniform = Uniform::new(-1.0, 1.0);

        let n_connections = ((n_pre * n_post) as f64 * (1.0 - sparsity)) as usize;
        let mut connections = Vec::with_capacity(n_connections);

        for _ in 0..n_connections {
            let pre = rng.r#gen_range(0..n_pre);
            let post = rng.r#gen_range(0..n_post);
            let weight = uniform.sample(&mut rng);
            connections.push((pre, post, weight));
        }

        Self {
            connections,
            n_pre,
            n_post,
        }
    }

    /// Convert to dense matrix
    pub fn to_dense(&self) -> Array2<f64> {
        let mut matrix = Array2::zeros((self.n_post, self.n_pre));
        for &(pre, post, weight) in &self.connections {
            matrix[[post, pre]] = weight;
        }
        matrix
    }
}

/// Echo State Network (ESN)
///
/// A recurrent neural network with fixed random weights in the reservoir,
/// where only the output (readout) layer is trained.
///
/// # References
/// - Jaeger, H. (2001). "The echo state approach to analysing and training recurrent neural networks"
#[derive(Debug, Clone)]
pub struct EchoStateNetwork {
    /// Input-to-reservoir weights (N_reservoir x N_input)
    input_weights: Array2<f64>,
    /// Reservoir recurrent weights (N_reservoir x N_reservoir)
    reservoir_weights: Array2<f64>,
    /// Trained output weights (N_output x N_reservoir)
    output_weights: Option<Array2<f64>>,
    /// Spectral radius for reservoir stability
    spectral_radius: f64,
    /// Leak rate (0 = no leak, 1 = instantaneous)
    leak_rate: f64,
    /// Number of neurons in reservoir
    reservoir_size: usize,
    /// Input dimension
    input_size: usize,
    /// Output dimension
    output_size: usize,
    /// Sparsity of reservoir connections (0 = dense, 1 = fully sparse)
    sparsity: f64,
    /// Current reservoir state
    state: Array1<f64>,
}

impl EchoStateNetwork {
    /// Create a new Echo State Network
    pub fn new(input_size: usize, reservoir_size: usize, output_size: usize) -> Self {
        Self {
            input_weights: Array2::zeros((reservoir_size, input_size)),
            reservoir_weights: Array2::zeros((reservoir_size, reservoir_size)),
            output_weights: None,
            spectral_radius: 0.9,
            leak_rate: 1.0,
            reservoir_size,
            input_size,
            output_size,
            sparsity: 0.9,
            state: Array1::zeros(reservoir_size),
        }
    }

    /// Set spectral radius (controls reservoir dynamics)
    pub fn with_spectral_radius(mut self, sr: f64) -> Self {
        self.spectral_radius = sr;
        self
    }

    /// Set leak rate (controls integration timescale)
    pub fn with_leak_rate(mut self, lr: f64) -> Self {
        self.leak_rate = lr;
        self
    }

    /// Set sparsity level
    pub fn with_sparsity(mut self, sp: f64) -> Self {
        self.sparsity = sp;
        self
    }

    /// Initialize reservoir weights randomly
    pub fn initialize(&mut self, seed: u64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let input_dist = Uniform::new(-1.0, 1.0);
        let reservoir_dist = Uniform::new(-1.0, 1.0);

        // Initialize input weights
        for i in 0..self.reservoir_size {
            for j in 0..self.input_size {
                self.input_weights[[i, j]] = input_dist.sample(&mut rng);
            }
        }

        // Initialize sparse reservoir weights
        for i in 0..self.reservoir_size {
            for j in 0..self.reservoir_size {
                if rng.r#gen::<f64>() > self.sparsity {
                    self.reservoir_weights[[i, j]] = reservoir_dist.sample(&mut rng);
                }
            }
        }

        // Scale by spectral radius
        let max_eigenvalue = self.compute_spectral_radius();
        if max_eigenvalue > 0.0 {
            self.reservoir_weights *= self.spectral_radius / max_eigenvalue;
        }
    }

    /// Compute spectral radius (approximate using power iteration)
    fn compute_spectral_radius(&self) -> f64 {
        let n = self.reservoir_size;
        let mut v = Array1::from_vec(vec![1.0 / (n as f64).sqrt(); n]);

        // Power iteration
        for _ in 0..100 {
            let v_new = self.reservoir_weights.dot(&v);
            let norm = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                v = v_new / norm;
            } else {
                break;
            }
        }

        let av = self.reservoir_weights.dot(&v);
        av.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Forward pass through reservoir
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        // Compute new state: x(t+1) = (1-α)x(t) + α·tanh(W_in·u(t) + W·x(t))
        let input_activation = self.input_weights.dot(input);
        let reservoir_activation = self.reservoir_weights.dot(&self.state);
        let activation = input_activation + reservoir_activation;

        // Apply tanh nonlinearity
        let new_state = activation.mapv(|x| x.tanh());

        // Leaky integration
        self.state = self.state.mapv(|x| (1.0 - self.leak_rate) * x)
            + new_state.mapv(|x| self.leak_rate * x);

        self.state.clone()
    }

    /// Train readout layer using ridge regression
    ///
    /// # Arguments
    /// * `states` - Collected reservoir states (N_samples x N_reservoir)
    /// * `targets` - Target outputs (N_samples x N_output)
    /// * `ridge_param` - Regularization parameter (default: 1e-6)
    pub fn train_readout(
        &mut self,
        states: &Array2<f64>,
        targets: &Array2<f64>,
        ridge_param: f64,
    ) -> Result<(), String> {
        if states.nrows() != targets.nrows() {
            return Err("Number of state samples must match number of target samples".to_string());
        }

        if states.ncols() != self.reservoir_size {
            return Err("State dimension must match reservoir size".to_string());
        }

        if targets.ncols() != self.output_size {
            return Err("Target dimension must match output size".to_string());
        }

        // Ridge regression: W_out = (X^T X + λI)^{-1} X^T Y
        // where X = states, Y = targets
        let xt_x = states.t().dot(states);
        let n = xt_x.nrows();
        let mut xt_x_reg = xt_x;

        // Add ridge regularization
        for i in 0..n {
            xt_x_reg[[i, i]] += ridge_param;
        }

        // Solve using Cholesky decomposition (assuming positive definite)
        let xt_y = states.t().dot(targets);

        // Simple pseudo-inverse (in practice, use more robust solver)
        match Self::solve_linear_system(&xt_x_reg, &xt_y) {
            Ok(w_out) => {
                self.output_weights = Some(w_out.t().to_owned());
                Ok(())
            }
            Err(e) => Err(format!("Failed to solve linear system: {}", e)),
        }
    }

    /// Simple linear system solver using Gaussian elimination
    fn solve_linear_system(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, String> {
        let n = a.nrows();
        let m = b.ncols();

        if a.ncols() != n {
            return Err("Matrix A must be square".to_string());
        }
        if b.nrows() != n {
            return Err("Matrix dimensions incompatible".to_string());
        }

        // Create augmented matrix [A|B]
        let mut aug = Array2::zeros((n, n + m));
        aug.slice_mut(s![.., ..n]).assign(a);
        aug.slice_mut(s![.., n..]).assign(b);

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..(n + m) {
                    let tmp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            let pivot = aug[[i, i]];
            if pivot.abs() < 1e-10 {
                return Err("Matrix is singular".to_string());
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = aug[[k, i]] / pivot;
                for j in i..(n + m) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array2::zeros((n, m));
        for i in (0..n).rev() {
            for j in 0..m {
                let mut sum = aug[[i, n + j]];
                for k in (i + 1)..n {
                    sum -= aug[[i, k]] * x[[k, j]];
                }
                x[[i, j]] = sum / aug[[i, i]];
            }
        }

        Ok(x)
    }

    /// Predict output for given input
    ///
    /// # Errors
    /// Returns an error if `train_readout` has not been called.
    pub fn predict(&mut self, input: &Array1<f64>) -> Result<Array1<f64>, String> {
        let state = self.forward(input);

        if let Some(ref w_out) = self.output_weights {
            Ok(w_out.dot(&state))
        } else {
            Err("Output weights not trained. Call train_readout first.".to_string())
        }
    }

    /// Reset reservoir state
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Get current reservoir state
    pub fn state(&self) -> &Array1<f64> {
        &self.state
    }
}

/// Liquid State Machine (LSM)
///
/// A reservoir computing approach using spiking neurons (LIF),
/// where the reservoir is a recurrently connected network of spiking neurons.
///
/// # References
/// - Maass, W., Natschläger, T., & Markram, H. (2002). "Real-time computing without stable states"
#[derive(Debug, Clone)]
pub struct LiquidStateMachine {
    /// Spiking neurons in the reservoir
    neurons: Vec<LifNeuron>,
    /// Sparse connectivity pattern
    connectivity: SparsityPattern,
    /// Input-to-reservoir weights
    input_weights: Array2<f64>,
    /// Trained readout weights (if trained)
    readout_weights: Option<Array2<f64>>,
    /// Input dimension
    input_size: usize,
    /// Output dimension
    output_size: usize,
    /// Time step in ms
    dt: f64,
}

impl LiquidStateMachine {
    /// Create a new Liquid State Machine
    pub fn new(
        input_size: usize,
        reservoir_size: usize,
        output_size: usize,
        config: LifConfig,
    ) -> Self {
        let neurons = (0..reservoir_size)
            .map(|_| LifNeuron::new(config.clone()))
            .collect();

        Self {
            neurons,
            connectivity: SparsityPattern {
                connections: Vec::new(),
                n_pre: reservoir_size,
                n_post: reservoir_size,
            },
            input_weights: Array2::zeros((reservoir_size, input_size)),
            readout_weights: None,
            input_size,
            output_size,
            dt: 1.0,
        }
    }

    /// Initialize with random connectivity
    ///
    /// # Panics
    /// This function will not panic. If weight_scale is invalid, weights default to 0.
    pub fn initialize(&mut self, sparsity: f64, weight_scale: f64, seed: u64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let reservoir_size = self.neurons.len();

        // Initialize input weights - use safe default if weight_scale is invalid
        let input_dist = match Normal::new(0.0, weight_scale.abs().max(1e-10)) {
            Ok(dist) => dist,
            Err(_) => return, // Invalid scale, leave weights at default
        };
        for i in 0..reservoir_size {
            for j in 0..self.input_size {
                self.input_weights[[i, j]] = input_dist.sample(&mut rng);
            }
        }

        // Initialize sparse recurrent connectivity
        self.connectivity = SparsityPattern::random(
            reservoir_size,
            reservoir_size,
            sparsity,
            seed,
        );
    }

    /// Forward pass for one time step
    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let reservoir_size = self.neurons.len();
        let mut spikes = Array1::zeros(reservoir_size);

        // Apply input to reservoir
        let input_currents = self.input_weights.dot(input);

        // Compute recurrent currents from previous spikes
        let mut recurrent_currents = Array1::<f64>::zeros(reservoir_size);
        for &(pre, post, weight) in &self.connectivity.connections {
            if pre < reservoir_size && post < reservoir_size {
                let pre_v = self.neurons[pre].membrane_potential() as f64;
                recurrent_currents[post] += weight * pre_v;
            }
        }

        // Update neurons
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let total_current = input_currents[i] + recurrent_currents[i];
            let spiked = neuron.update(total_current as f32, self.dt as f32);
            spikes[i] = if spiked { 1.0 } else { 0.0 };
        }

        spikes
    }

    /// Train readout layer using spike states
    pub fn train_readout(
        &mut self,
        spike_trains: &Array2<f64>,
        targets: &Array2<f64>,
        ridge_param: f64,
    ) -> Result<(), String> {
        if spike_trains.nrows() != targets.nrows() {
            return Err("Number of samples must match".to_string());
        }

        // Use ridge regression similar to ESN
        let xt_x = spike_trains.t().dot(spike_trains);
        let n = xt_x.nrows();
        let mut xt_x_reg = xt_x;

        for i in 0..n {
            xt_x_reg[[i, i]] += ridge_param;
        }

        let xt_y = spike_trains.t().dot(targets);

        match EchoStateNetwork::solve_linear_system(&xt_x_reg, &xt_y) {
            Ok(w_out) => {
                self.readout_weights = Some(w_out.t().to_owned());
                Ok(())
            }
            Err(e) => Err(format!("Failed to solve linear system: {}", e)),
        }
    }

    /// Predict using trained readout
    ///
    /// # Errors
    /// Returns an error if `train_readout` has not been called.
    pub fn predict(&mut self, input: &Array1<f64>) -> Result<Array1<f64>, String> {
        let spikes = self.forward(input);

        if let Some(ref w_out) = self.readout_weights {
            Ok(w_out.dot(&spikes))
        } else {
            Err("Readout weights not trained. Call train_readout first.".to_string())
        }
    }

    /// Reset all neurons
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
    }

    /// Get current spike state
    pub fn spike_state(&self) -> Array1<f64> {
        Array1::from_vec(
            self.neurons
                .iter()
                .map(|n| if n.membrane_potential() > 0.0 { 1.0 } else { 0.0 })
                .collect(),
        )
    }

    /// Get membrane potentials
    pub fn membrane_potentials(&self) -> Array1<f64> {
        Array1::from_vec(
            self.neurons
                .iter()
                .map(|n| n.membrane_potential() as f64)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparsity_pattern_creation() {
        let pattern = SparsityPattern::random(100, 100, 0.9, 42);
        assert_eq!(pattern.n_pre, 100);
        assert_eq!(pattern.n_post, 100);
        assert!(pattern.connections.len() > 0);

        // Check sparsity roughly matches
        let expected_connections = (100 * 100) as f64 * 0.1;
        let actual_connections = pattern.connections.len() as f64;
        assert!((actual_connections - expected_connections).abs() < expected_connections * 0.5);
    }

    #[test]
    fn test_sparsity_pattern_to_dense() {
        let pattern = SparsityPattern {
            connections: vec![(0, 1, 0.5), (2, 3, -0.3)],
            n_pre: 4,
            n_post: 4,
        };

        let dense = pattern.to_dense();
        assert_eq!(dense.shape(), &[4, 4]);
        assert_eq!(dense[[1, 0]], 0.5);
        assert_eq!(dense[[3, 2]], -0.3);
    }

    #[test]
    fn test_esn_creation() {
        let mut esn = EchoStateNetwork::new(10, 100, 5)
            .with_spectral_radius(0.95)
            .with_leak_rate(0.8)
            .with_sparsity(0.9);

        esn.initialize(42);

        assert_eq!(esn.reservoir_size, 100);
        assert_eq!(esn.input_size, 10);
        assert_eq!(esn.output_size, 5);
        assert_eq!(esn.spectral_radius, 0.95);
        assert_eq!(esn.leak_rate, 0.8);
    }

    #[test]
    fn test_esn_forward() {
        let mut esn = EchoStateNetwork::new(5, 50, 3);
        esn.initialize(42);

        let input = Array1::from_vec(vec![1.0, 0.5, -0.5, 0.0, 0.2]);
        let state = esn.forward(&input);

        assert_eq!(state.len(), 50);

        // State should be bounded due to tanh
        for &s in state.iter() {
            assert!(s >= -1.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_esn_training() {
        let mut esn = EchoStateNetwork::new(3, 20, 2);
        esn.initialize(42);

        // Collect some states
        let n_samples = 50;
        let mut states = Array2::zeros((n_samples, 20));
        let mut targets = Array2::zeros((n_samples, 2));

        for i in 0..n_samples {
            let input = Array1::from_vec(vec![
                (i as f64 * 0.1).sin(),
                (i as f64 * 0.1).cos(),
                (i as f64 * 0.05).sin(),
            ]);

            let state = esn.forward(&input);
            states.row_mut(i).assign(&state);

            // Simple target: sin and cos of time
            targets[[i, 0]] = (i as f64 * 0.1).sin();
            targets[[i, 1]] = (i as f64 * 0.1).cos();
        }

        // Train readout
        let result = esn.train_readout(&states, &targets, 1e-6);
        assert!(result.is_ok());
        assert!(esn.output_weights.is_some());
    }

    #[test]
    fn test_esn_predict() {
        let mut esn = EchoStateNetwork::new(2, 30, 1);
        esn.initialize(42);

        // Train on simple function
        let n_samples = 100;
        let mut states = Array2::zeros((n_samples, 30));
        let mut targets = Array2::zeros((n_samples, 1));

        for i in 0..n_samples {
            let x = i as f64 * 0.1;
            let input = Array1::from_vec(vec![x.sin(), x.cos()]);
            let state = esn.forward(&input);
            states.row_mut(i).assign(&state);
            targets[[i, 0]] = (2.0 * x).sin();
        }

        esn.train_readout(&states, &targets, 1e-5).unwrap();

        // Test prediction
        esn.reset();
        let test_input = Array1::from_vec(vec![0.5_f64.sin(), 0.5_f64.cos()]);
        let _ = esn.forward(&test_input);
        let prediction = esn.predict(&test_input);

        assert_eq!(prediction.len(), 1);
    }

    #[test]
    fn test_lsm_creation() {
        let config = LifConfig::default();
        let lsm = LiquidStateMachine::new(5, 50, 3, config);

        assert_eq!(lsm.neurons.len(), 50);
        assert_eq!(lsm.input_size, 5);
        assert_eq!(lsm.output_size, 3);
    }

    #[test]
    fn test_lsm_initialization() {
        let config = LifConfig::default();
        let mut lsm = LiquidStateMachine::new(5, 30, 2, config);
        lsm.initialize(0.9, 0.5, 42);

        assert!(lsm.connectivity.connections.len() > 0);

        // Check input weights are initialized
        let sum: f64 = lsm.input_weights.iter().map(|&x| x.abs()).sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_lsm_forward() {
        let config = LifConfig::default();
        let mut lsm = LiquidStateMachine::new(3, 20, 2, config);
        lsm.initialize(0.9, 1.0, 42);

        let input = Array1::from_vec(vec![10.0, 5.0, -5.0]);
        let spikes = lsm.forward(&input);

        assert_eq!(spikes.len(), 20);

        // Spikes should be binary
        for &s in spikes.iter() {
            assert!(s == 0.0 || s == 1.0);
        }
    }

    #[test]
    fn test_lsm_reset() {
        let config = LifConfig::default();
        let mut lsm = LiquidStateMachine::new(3, 10, 1, config);
        lsm.initialize(0.8, 1.0, 42);

        // Run some steps
        for _ in 0..10 {
            let input = Array1::from_vec(vec![5.0, 5.0, 5.0]);
            lsm.forward(&input);
        }

        // Reset
        lsm.reset();

        let potentials = lsm.membrane_potentials();
        let initial_potential = LifConfig::default().v_rest as f64;

        for &v in potentials.iter() {
            assert!((v - initial_potential).abs() < 0.1);
        }
    }
}
