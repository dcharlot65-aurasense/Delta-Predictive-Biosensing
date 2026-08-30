//! Recurrent SNN architectures for temporal processing

use super::SNNArchitecture;
use crate::{
    SNNConfig, SNNError, SNNResult, SpikeTensor,
    layers::{SpikingLayer, SpikingLinear, SpikingRNN},
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Recurrent Spiking Neural Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentSNN {
    /// Input projection layer
    pub input_layer: Option<SpikingLinear>,
    /// Recurrent layers
    pub recurrent_layers: Vec<SpikingRNN>,
    /// Output layer
    pub output_layer: SpikingLinear,
    /// Network configuration
    pub config: SNNConfig,
    /// Layer sizes
    pub layer_sizes: Vec<usize>,
}

impl RecurrentSNN {
    /// Create a new recurrent SNN
    ///
    /// # Errors
    /// Returns `SNNError::InvalidConfig` if `hidden_sizes` is empty.
    #[must_use = "this Result may contain an error that should be handled"]
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        config: SNNConfig,
    ) -> SNNResult<Self> {
        if hidden_sizes.is_empty() {
            return Err(SNNError::InvalidConfig(
                "Need at least one hidden layer".to_string(),
            ));
        }

        // Input projection if first hidden size differs from input
        let input_layer = if input_size != hidden_sizes[0] {
            Some(SpikingLinear::new(
                input_size,
                hidden_sizes[0],
                true,
                config.neuron_params.clone(),
                config.dt,
                false,
            ))
        } else {
            None
        };

        // Create recurrent layers
        let mut recurrent_layers = Vec::new();
        for i in 0..hidden_sizes.len() {
            let h_in = if i == 0 {
                hidden_sizes[0]
            } else {
                hidden_sizes[i - 1]
            };
            let h_out = hidden_sizes[i];

            recurrent_layers.push(SpikingRNN::new(
                h_in,
                h_out,
                true,
                config.neuron_params.clone(),
                config.dt,
                false,
            ));
        }

        // Output layer
        let output_layer = SpikingLinear::new(
            *hidden_sizes.last().unwrap(),
            output_size,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        let mut layer_sizes = vec![input_size];
        layer_sizes.extend(hidden_sizes);
        layer_sizes.push(output_size);

        Ok(Self {
            input_layer,
            recurrent_layers,
            output_layer,
            config,
            layer_sizes,
        })
    }
}

impl SNNArchitecture for RecurrentSNN {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        // Project input if needed
        let mut current = if let Some(ref mut input_layer) = self.input_layer {
            input_layer.forward(input)?
        } else {
            input.clone()
        };

        // Forward through recurrent layers
        for layer in &mut self.recurrent_layers {
            current = layer.forward(&current)?;
        }

        // Forward through output layer
        current = self.output_layer.forward(&current)?;

        Ok(current)
    }

    fn reset(&mut self) {
        if let Some(ref mut layer) = self.input_layer {
            layer.reset_state();
        }
        for layer in &mut self.recurrent_layers {
            layer.reset_state();
        }
        self.output_layer.reset_state();
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        let mut params = Vec::new();
        if let Some(ref layer) = self.input_layer {
            params.extend(layer.parameters());
        }
        for layer in &self.recurrent_layers {
            params.extend(layer.parameters());
        }
        params.extend(self.output_layer.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        let mut params = Vec::new();
        if let Some(ref mut layer) = self.input_layer {
            params.extend(layer.parameters_mut());
        }
        for layer in &mut self.recurrent_layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.output_layer.parameters_mut());
        params
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut layer) = self.input_layer {
            layer.zero_grad();
        }
        for layer in &mut self.recurrent_layers {
            layer.zero_grad();
        }
        self.output_layer.zero_grad();
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

/// Estimates the spectral radius of a square matrix by power iteration.
///
/// Uses the growth rate of `||W^k v||`, averaged over the later iterations,
/// rather than a Rayleigh quotient: the recurrent matrix is not symmetric, so
/// its dominant eigenvalues can be a complex pair, and a Rayleigh quotient
/// oscillates in that case while the growth rate still converges to the
/// modulus.
///
/// Returns 0.0 for a matrix that annihilates the probe vector.
fn estimate_spectral_radius(w: &Array2<f32>, iterations: usize) -> f32 {
    let n = w.shape()[0];
    if n == 0 {
        return 0.0;
    }

    // Deterministic probe, so the same weights always give the same estimate.
    let mut v = Array1::from_shape_fn(n, |i| ((i % 7) as f32) + 1.0);
    let norm = v.dot(&v).sqrt();
    if norm <= f32::EPSILON {
        return 0.0;
    }
    v /= norm;

    let burn_in = iterations / 2;
    let mut log_growth = 0.0f32;
    let mut counted = 0usize;

    for k in 0..iterations {
        let next = w.dot(&v);
        let growth = next.dot(&next).sqrt();
        if growth <= f32::EPSILON {
            return 0.0;
        }
        if k >= burn_in {
            log_growth += growth.ln();
            counted += 1;
        }
        v = next / growth;
    }

    if counted == 0 {
        0.0
    } else {
        (log_growth / counted as f32).exp()
    }
}

/// Liquid State Machine (LSM)
/// A reservoir computing approach with a randomly connected spiking reservoir
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidStateMachine {
    /// Reservoir layer (randomly initialized, typically not trained)
    pub reservoir: SpikingRNN,
    /// Readout layer (trained)
    pub readout: SpikingLinear,
    /// Network configuration
    pub config: SNNConfig,
    /// Reservoir size
    pub reservoir_size: usize,
    /// Output size
    pub output_size: usize,
    /// Spectral radius for reservoir weights
    pub spectral_radius: f32,
}

impl LiquidStateMachine {
    /// Create a new Liquid State Machine
    pub fn new(
        input_size: usize,
        reservoir_size: usize,
        output_size: usize,
        spectral_radius: f32,
        config: SNNConfig,
    ) -> Self {
        // Create reservoir with random connectivity
        let mut reservoir = SpikingRNN::new(
            input_size,
            reservoir_size,
            false,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        // Scale the recurrent weights so their spectral radius is the one asked
        // for. This used to divide by `reservoir_size`, which is not the
        // spectral radius of anything: for a random matrix with i.i.d. entries
        // of standard deviation s, the radius grows as s*sqrt(N), so dividing
        // by N undershot by a factor of about sqrt(N) -- and the radius is what
        // decides whether the reservoir has any memory at all, so undershooting
        // leaves a state that decays almost immediately.
        let measured = estimate_spectral_radius(&reservoir.w_recurrent, 60);
        if measured > f32::EPSILON {
            reservoir.w_recurrent *= spectral_radius / measured;
        }

        // Readout layer
        let readout = SpikingLinear::new(
            reservoir_size,
            output_size,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        Self {
            reservoir,
            readout,
            config,
            reservoir_size,
            output_size,
            spectral_radius,
        }
    }

    /// Get reservoir state (for analysis)
    pub fn get_reservoir_state(&self) -> Vec<Array1<f32>> {
        self.reservoir
            .state
            .iter()
            .map(|s| s.v_mem.clone())
            .collect()
    }
}

impl SNNArchitecture for LiquidStateMachine {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        // Forward through reservoir (typically frozen)
        let reservoir_out = self.reservoir.forward(input)?;

        // Forward through readout layer (trainable)
        let output = self.readout.forward(&reservoir_out)?;

        Ok(output)
    }

    fn reset(&mut self) {
        self.reservoir.reset_state();
        self.readout.reset_state();
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        // Only readout parameters are trainable in standard LSM
        self.readout.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        // Only readout parameters are trainable in standard LSM
        self.readout.parameters_mut()
    }

    fn zero_grad(&mut self) {
        // Only zero readout gradients
        self.readout.zero_grad();
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

/// Echo State Network (ESN) style SNN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoStateSNN {
    /// Reservoir
    pub reservoir: SpikingRNN,
    /// Output weights (trained with simple regression)
    pub output_weights: Array2<f32>,
    /// Config
    pub config: SNNConfig,
}

impl EchoStateSNN {
    pub fn new(
        input_size: usize,
        reservoir_size: usize,
        output_size: usize,
        config: SNNConfig,
    ) -> Self {
        let reservoir = SpikingRNN::new(
            input_size,
            reservoir_size,
            false,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        let output_weights = Array2::zeros((output_size, reservoir_size));

        Self {
            reservoir,
            output_weights,
            config,
        }
    }

    /// Fit the readout by ridge regression.
    ///
    /// `states` is `(n_samples, reservoir_size)` -- the reservoir's response to
    /// the training input -- and `targets` is `(n_samples, output_size)`. Only
    /// the readout is fitted; the reservoir is left alone, which is the whole
    /// idea of an echo state network.
    ///
    /// Solves `(X'X + lambda I) W = X'Y` by Cholesky decomposition. The matrix
    /// is symmetric and, for `lambda > 0`, positive definite, so Cholesky is
    /// both the right factorisation and cheaper than a general inverse.
    ///
    /// This used to build `X'X`, add the ridge term, and then stop at a comment
    /// reading "would need matrix inversion" -- never assigning
    /// `output_weights`, which are initialised to zero. The readout therefore
    /// stayed zero however much data it was given, and the network predicted
    /// zero for every input, reporting Ok(()) each time.
    pub fn train_readout(
        &mut self,
        states: &Array2<f32>,
        targets: &Array2<f32>,
        ridge_param: f32,
    ) -> SNNResult<()> {
        let (n_samples, reservoir_size) = (states.shape()[0], states.shape()[1]);
        let output_size = self.output_weights.shape()[0];

        if targets.shape()[0] != n_samples {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{n_samples} target rows"),
                actual: format!("{}", targets.shape()[0]),
            });
        }
        if targets.shape()[1] != output_size {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{output_size} target columns"),
                actual: format!("{}", targets.shape()[1]),
            });
        }
        if reservoir_size != self.output_weights.shape()[1] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} state columns", self.output_weights.shape()[1]),
                actual: format!("{reservoir_size}"),
            });
        }
        if ridge_param <= 0.0 {
            return Err(SNNError::InvalidConfig(
                "ridge_param must be positive: it is what makes the normal \
                 equations solvable when the reservoir states are collinear, \
                 which they generally are"
                    .to_string(),
            ));
        }

        // A = X'X + lambda I
        let mut a = states.t().dot(states);
        for i in 0..reservoir_size {
            a[[i, i]] += ridge_param;
        }
        // B = X'Y
        let b = states.t().dot(targets);

        let l = cholesky(&a)?;
        let z = cholesky_solve(&l, &b);

        // Z is (reservoir, output); the readout is stored transposed.
        self.output_weights = z.t().to_owned();
        Ok(())
    }

    /// Apply the trained readout to reservoir states.
    ///
    /// Returns `(n_samples, output_size)`. Without this the fitted weights had
    /// no consumer at all -- nothing in the type read `output_weights`.
    pub fn predict(&self, states: &Array2<f32>) -> SNNResult<Array2<f32>> {
        if states.shape()[1] != self.output_weights.shape()[1] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} state columns", self.output_weights.shape()[1]),
                actual: format!("{}", states.shape()[1]),
            });
        }
        Ok(states.dot(&self.output_weights.t()))
    }
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular `L` with `A = L L'`. Errors rather than
/// producing NaNs if the matrix turns out not to be positive definite, which
/// for these normal equations means the ridge term was too small to lift a
/// singular `X'X`.
fn cholesky(a: &Array2<f32>) -> SNNResult<Array2<f32>> {
    let n = a.shape()[0];
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(SNNError::InvalidConfig(format!(
                        "readout normal equations are not positive definite at \
                         pivot {i} (got {sum}); increase ridge_param"
                    )));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Solves `L L' Z = B` for `Z`, one right-hand side column at a time.
fn cholesky_solve(l: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let n = l.shape()[0];
    let cols = b.shape()[1];
    let mut z = Array2::zeros((n, cols));

    for c in 0..cols {
        // Forward substitution: L y = b
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = b[[i, c]];
            for k in 0..i {
                sum -= l[[i, k]] * y[k];
            }
            y[i] = sum / l[[i, i]];
        }
        // Back substitution: L' x = y
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum -= l[[k, i]] * z[[k, c]];
            }
            z[[i, c]] = sum / l[[i, i]];
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recurrent_snn_creation() {
        let rsnn = RecurrentSNN::new(10, vec![20, 15], 5, SNNConfig::default()).unwrap();
        assert_eq!(rsnn.layer_sizes, vec![10, 20, 15, 5]);
    }

    #[test]
    fn test_recurrent_snn_empty_hidden() {
        let result = RecurrentSNN::new(10, vec![], 5, SNNConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_recurrent_snn_forward() {
        let mut rsnn = RecurrentSNN::new(10, vec![20], 5, SNNConfig::default()).unwrap();
        let input = SpikeTensor::zeros(2, 30, 10, false);

        let output = rsnn.forward(&input).unwrap();
        assert_eq!(output.shape(), (2, 30, 5));
    }

    #[test]
    fn test_lsm_creation() {
        let lsm = LiquidStateMachine::new(10, 100, 5, 0.9, SNNConfig::default());
        assert_eq!(lsm.reservoir_size, 100);
        assert_eq!(lsm.output_size, 5);
    }

    #[test]
    fn test_lsm_forward() {
        let mut lsm = LiquidStateMachine::new(10, 50, 3, 0.9, SNNConfig::default());
        let input = SpikeTensor::zeros(1, 20, 10, false);

        let output = lsm.forward(&input).unwrap();
        assert_eq!(output.shape(), (1, 20, 3));
    }

    #[test]
    fn test_lsm_reservoir_state() {
        let lsm = LiquidStateMachine::new(5, 20, 2, 0.9, SNNConfig::default());
        let states = lsm.get_reservoir_state();
        assert_eq!(states.len(), 0); // No state until forward pass
    }

    // ---- echo state readout -------------------------------------------

    /// The readout must recover a linear map it was trained on.
    ///
    /// `train_readout` used to compute the normal equations and then stop at a
    /// comment, never assigning the weights, which start at zero -- so the
    /// network predicted zero for every input no matter how much data it saw,
    /// returning Ok(()) each time. Fitting an exactly-linear target is the
    /// sharpest check: with a small ridge term the recovered map should be very
    /// close to the true one.
    #[test]
    fn echo_state_readout_recovers_a_linear_map() {
        let (samples, reservoir, outputs) = (60usize, 6usize, 2usize);
        let mut esn = EchoStateSNN::new(3, reservoir, outputs, SNNConfig::default());

        // Deterministic, well-conditioned states.
        let states = Array2::from_shape_fn((samples, reservoir), |(s, r)| {
            ((s * 7 + r * 13) % 11) as f32 * 0.3 - 1.5
        });
        // A known map from states to targets.
        let truth = Array2::from_shape_fn((outputs, reservoir), |(o, r)| {
            0.4 - 0.15 * (o as f32) + 0.07 * (r as f32)
        });
        let targets = states.dot(&truth.t());

        // Before training, the readout is zero and predicts zero.
        let before = esn.predict(&states).unwrap();
        assert_eq!(
            before.iter().map(|v| v.abs()).sum::<f32>(),
            0.0,
            "an untrained readout should predict zero"
        );

        esn.train_readout(&states, &targets, 1e-6).unwrap();

        let learned = &esn.output_weights;
        let err: f32 =
            (learned - &truth).iter().map(|d| d.abs()).sum::<f32>() / (outputs * reservoir) as f32;
        assert!(
            err < 1e-3,
            "readout did not recover the map: mean |error| {err}\nlearned {learned:?}\ntruth {truth:?}"
        );

        // And prediction reproduces the targets.
        let predicted = esn.predict(&states).unwrap();
        let pred_err: f32 = (&predicted - &targets).iter().map(|d| d.abs()).sum::<f32>()
            / (samples * outputs) as f32;
        assert!(
            pred_err < 1e-3,
            "predictions are off by {pred_err} on average"
        );
    }

    /// A larger ridge term must shrink the readout toward zero, which is what
    /// the parameter is for.
    #[test]
    fn echo_state_ridge_parameter_shrinks_the_readout() {
        let (samples, reservoir, outputs) = (40usize, 5usize, 1usize);
        let states = Array2::from_shape_fn((samples, reservoir), |(s, r)| {
            ((s * 5 + r * 3) % 7) as f32 * 0.4 - 1.0
        });
        let truth = Array2::from_shape_fn((outputs, reservoir), |(_, r)| 0.5 + 0.1 * r as f32);
        let targets = states.dot(&truth.t());

        let norm = |ridge: f32| {
            let mut esn = EchoStateSNN::new(2, reservoir, outputs, SNNConfig::default());
            esn.train_readout(&states, &targets, ridge).unwrap();
            esn.output_weights.iter().map(|w| w * w).sum::<f32>().sqrt()
        };

        let light = norm(1e-6);
        let heavy = norm(1e4);
        assert!(
            heavy < light,
            "a heavy ridge ({heavy}) should shrink the readout below a light one ({light})"
        );
    }

    /// Mismatched shapes and a non-positive ridge are reported, not ignored.
    #[test]
    fn echo_state_readout_validates_its_input() {
        let mut esn = EchoStateSNN::new(2, 4, 2, SNNConfig::default());
        let states = Array2::zeros((10, 4));

        // Wrong number of target rows.
        assert!(
            esn.train_readout(&states, &Array2::zeros((9, 2)), 1e-3)
                .is_err()
        );
        // Wrong number of target columns.
        assert!(
            esn.train_readout(&states, &Array2::zeros((10, 3)), 1e-3)
                .is_err()
        );
        // Wrong state width.
        assert!(
            esn.train_readout(&Array2::zeros((10, 5)), &Array2::zeros((10, 2)), 1e-3)
                .is_err()
        );
        // A zero ridge leaves the normal equations singular here.
        assert!(
            esn.train_readout(&states, &Array2::zeros((10, 2)), 0.0)
                .is_err()
        );
    }

    /// The reservoir's spectral radius must be the one requested.
    ///
    /// It used to be set by dividing the weights by `reservoir_size`, which is
    /// not the radius of anything: for i.i.d. entries the radius grows as
    /// s*sqrt(N), so the result undershot by roughly sqrt(N). The radius is
    /// what gives a reservoir its memory, so a reservoir asked for 0.9 and
    /// given 0.09 forgets almost immediately.
    #[test]
    fn liquid_state_machine_hits_its_spectral_radius() {
        for &target in &[0.5f32, 0.9, 1.2] {
            for &size in &[40usize, 120] {
                let lsm = LiquidStateMachine::new(5, size, 3, target, SNNConfig::default());
                let measured = estimate_spectral_radius(&lsm.reservoir.w_recurrent, 200);
                let rel = (measured - target).abs() / target;
                assert!(
                    rel < 0.15,
                    "reservoir of {size} asked for radius {target}, measured {measured}"
                );
            }
        }
    }

    /// The estimator itself must be right on a matrix whose radius is known.
    #[test]
    fn spectral_radius_estimator_is_accurate() {
        // Diagonal: the radius is the largest magnitude on the diagonal.
        let mut d = Array2::zeros((4, 4));
        for (i, v) in [0.3f32, -2.5, 1.1, 0.7].iter().enumerate() {
            d[[i, i]] = *v;
        }
        let est = estimate_spectral_radius(&d, 200);
        assert!(
            (est - 2.5).abs() < 1e-2,
            "diagonal: expected 2.5, got {est}"
        );

        // A rotation-and-scale block has a complex eigenvalue pair of modulus
        // r -- the case a Rayleigh quotient cannot handle.
        let r = 1.7f32;
        let (c, s) = (0.6f32, 0.8f32);
        let mut rot = Array2::zeros((2, 2));
        rot[[0, 0]] = r * c;
        rot[[0, 1]] = -r * s;
        rot[[1, 0]] = r * s;
        rot[[1, 1]] = r * c;
        let est = estimate_spectral_radius(&rot, 200);
        assert!(
            (est - r).abs() < 1e-2,
            "complex pair: expected {r}, got {est}"
        );
    }
}
