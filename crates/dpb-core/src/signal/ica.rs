//! Independent Component Analysis (ICA) for blind source separation.

use crate::error::{DpbError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::{RngExt};

/// Nonlinear function types for FastICA algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonlinearFunction {
    /// logcosh function: G(u) = log(cosh(u))
    LogCosh,
    /// Exponential function: G(u) = -exp(-u²/2)
    Exp,
    /// Cubic function: G(u) = u³
    Cube,
}

impl NonlinearFunction {
    /// Computes the nonlinear function G(u).
    fn g(&self, u: f64) -> f64 {
        match self {
            NonlinearFunction::LogCosh => u.cosh().ln(),
            NonlinearFunction::Exp => -(u * u / 2.0).exp(),
            NonlinearFunction::Cube => u * u * u,
        }
    }

    /// Computes the derivative g'(u).
    fn g_prime(&self, u: f64) -> f64 {
        match self {
            NonlinearFunction::LogCosh => u.tanh(),
            NonlinearFunction::Exp => u * (-(u * u / 2.0).exp()),
            NonlinearFunction::Cube => 3.0 * u * u,
        }
    }

    /// Computes the second derivative g''(u).
    fn g_double_prime(&self, u: f64) -> f64 {
        match self {
            NonlinearFunction::LogCosh => {
                let tanh = u.tanh();
                1.0 - tanh * tanh
            }
            NonlinearFunction::Exp => {
                let exp_term = (-(u * u / 2.0)).exp();
                exp_term * (u * u - 1.0)
            }
            NonlinearFunction::Cube => 6.0 * u,
        }
    }
}

/// FastICA algorithm for Independent Component Analysis.
///
/// Implements the FastICA algorithm for blind source separation, which attempts
/// to separate a multivariate signal into additive, independent components.
pub struct FastICA {
    n_components: usize,
    max_iter: usize,
    tol: f64,
    whiten: bool,
    fun: NonlinearFunction,
    // Fitted parameters
    mixing_matrix: Option<Array2<f64>>,
    unmixing_matrix: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    whitening_matrix: Option<Array2<f64>>,
}

impl FastICA {
    /// Creates a new FastICA instance.
    ///
    /// # Arguments
    /// * `n_components` - Number of independent components to extract
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            whiten: true,
            fun: NonlinearFunction::LogCosh,
            mixing_matrix: None,
            unmixing_matrix: None,
            mean: None,
            whitening_matrix: None,
        }
    }

    /// Sets the maximum number of iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets whether to perform whitening.
    pub fn with_whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    /// Sets the nonlinear function to use.
    pub fn with_function(mut self, fun: NonlinearFunction) -> Self {
        self.fun = fun;
        self
    }

    /// Fits the ICA model to the data.
    ///
    /// # Arguments
    /// * `data` - Input data with shape (n_features, n_samples)
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<()> {
        let (n_features, n_samples) = data.dim();

        if n_samples < self.n_components {
            return Err(DpbError::InvalidDimensions(
                "Number of samples must be >= number of components".to_string(),
            ));
        }

        if n_features < self.n_components {
            return Err(DpbError::InvalidDimensions(
                "Number of features must be >= number of components".to_string(),
            ));
        }

        // Center the data
        let mean = data.mean_axis(Axis(1)).unwrap();
        let mut centered = data.clone();
        for i in 0..n_features {
            for j in 0..n_samples {
                centered[[i, j]] -= mean[i];
            }
        }

        self.mean = Some(mean);

        // Whiten the data if requested
        let mut dewhitening_matrix = None;
        let x_white = if self.whiten {
            let (whitened, whitening, dewhitening) = self.whiten_data(&centered)?;
            self.whitening_matrix = Some(whitening);
            dewhitening_matrix = Some(dewhitening);
            whitened
        } else {
            centered
        };

        // Perform FastICA
        let unmixing = self.fastica_algorithm(&x_white)?;

        // Store the unmixing matrix
        self.unmixing_matrix = Some(unmixing.clone());

        // Compute mixing matrix (pseudo-inverse of unmixing matrix)
        let mixing = self.pseudo_inverse(&unmixing)?;

        // If whitening was applied, adjust mixing matrix.
        //
        // The forward model is `s = U * W_white * x`, so recovering `x` from the
        // sources needs `pinv(W_white) * pinv(U)`. The whitening matrix itself
        // was used here instead of its pseudo-inverse, which is not only the
        // wrong operator but the wrong SHAPE: `W_white` is
        // (components x features) and `pinv(U)` is (components x components),
        // so the product was rejected outright by ndarray -- `get_result`
        // panicked for any input with more features than components, which is
        // the case ICA exists for.
        if let Some(dewhitening) = dewhitening_matrix {
            self.mixing_matrix = Some(dewhitening.dot(&mixing));
        } else {
            self.mixing_matrix = Some(mixing);
        }

        Ok(())
    }

    /// Transforms the data using the fitted ICA model.
    ///
    /// # Arguments
    /// * `data` - Input data with shape (n_features, n_samples)
    ///
    /// # Returns
    /// Independent sources with shape (n_components, n_samples)
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let unmixing = self
            .unmixing_matrix
            .as_ref()
            .ok_or_else(|| DpbError::SignalProcessing("Model not fitted".to_string()))?;

        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| DpbError::SignalProcessing("Model not fitted".to_string()))?;

        let (n_features, n_samples) = data.dim();

        // Center the data
        let mut centered = data.clone();
        for i in 0..n_features {
            for j in 0..n_samples {
                centered[[i, j]] -= mean[i];
            }
        }

        // Apply whitening if it was used during fitting
        let x_transform = if let Some(ref whitening) = self.whitening_matrix {
            whitening.dot(&centered)
        } else {
            centered
        };

        // Apply unmixing matrix
        Ok(unmixing.dot(&x_transform))
    }

    /// Fits the model and transforms the data in one step.
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Returns the ICA result with mixing and unmixing matrices.
    pub fn get_result(&self) -> Result<ICAResult> {
        let mixing = self
            .mixing_matrix
            .as_ref()
            .ok_or_else(|| DpbError::SignalProcessing("Model not fitted".to_string()))?
            .clone();

        let unmixing = self
            .unmixing_matrix
            .as_ref()
            .ok_or_else(|| DpbError::SignalProcessing("Model not fitted".to_string()))?
            .clone();

        Ok(ICAResult {
            mixing_matrix: mixing,
            unmixing_matrix: unmixing,
        })
    }

    /// Whitens the data using eigenvalue decomposition.
    fn whiten_data(
        &self,
        data: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let (n_features, n_samples) = data.dim();

        // Compute covariance matrix
        let cov = data.dot(&data.t()) / n_samples as f64;

        // Eigenvalue decomposition (simplified using SVD approximation)
        let (u, s, _) = self.svd(&cov)?;

        // Compute whitening matrix: K = D^(-1/2) @ U_k^T   (components x features)
        // and its exact inverse map K^+ = U_k @ D^(1/2)     (features x components).
        //
        // `U_k` has orthonormal columns, so `K @ K^+ = I` exactly and no general
        // pseudo-inverse is needed. That matters: `pseudo_inverse` here relies
        // on a simplified `svd` valid only for square matrices, and `K` is not
        // square whenever there are more features than components -- the case
        // ICA exists for.
        let mut whitening = Array2::zeros((self.n_components, n_features));
        let mut dewhitening = Array2::zeros((n_features, self.n_components));
        for i in 0..self.n_components {
            let sqrt_s = s[i].sqrt();
            let scale = 1.0 / (sqrt_s + 1e-10);
            for j in 0..n_features {
                whitening[[i, j]] = scale * u[[j, i]];
                dewhitening[[j, i]] = sqrt_s * u[[j, i]];
            }
        }

        let whitened = whitening.dot(data);

        Ok((whitened, whitening, dewhitening))
    }

    /// Performs the FastICA algorithm.
    fn fastica_algorithm(&self, x_white: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_components, n_samples) = x_white.dim();
        let mut unmixing = Array2::zeros((self.n_components, n_components));

        // Initialize with random weights
        let mut rng = rand::rng();
        for i in 0..self.n_components {
            for j in 0..n_components {
                unmixing[[i, j]] = rng.random_range(-1.0..1.0);
            }
        }

        // Orthogonalize initial weights
        unmixing = self.symmetric_decorrelation(&unmixing)?;

        // Iterate for each component
        for comp_idx in 0..self.n_components {
            let mut w = unmixing.row(comp_idx).to_owned();

            for _iter in 0..self.max_iter {
                // Compute w^T @ X
                let wtx: Array1<f64> = (0..n_samples)
                    .map(|j| {
                        let mut sum = 0.0;
                        for i in 0..n_components {
                            sum += w[i] * x_white[[i, j]];
                        }
                        sum
                    })
                    .collect();

                // Compute g(w^T @ X) and g'(w^T @ X)
                let g_wtx: Array1<f64> = wtx.iter().map(|&x| self.fun.g_prime(x)).collect();
                let gp_wtx: Array1<f64> = wtx.iter().map(|&x| self.fun.g_double_prime(x)).collect();

                // Update rule: w = E{X * g(w^T @ X)} - E{g'(w^T @ X)} * w
                let mut w_new = Array1::zeros(n_components);
                for i in 0..n_components {
                    let mut sum = 0.0;
                    for j in 0..n_samples {
                        sum += x_white[[i, j]] * g_wtx[j];
                    }
                    w_new[i] = sum / n_samples as f64;
                }

                let gp_mean = gp_wtx.mean().unwrap_or(0.0);
                w_new = w_new - gp_mean * &w;

                // Orthogonalize with previous components
                for prev_idx in 0..comp_idx {
                    let prev_w = unmixing.row(prev_idx);
                    let dot: f64 = w_new.iter().zip(prev_w.iter()).map(|(a, b)| a * b).sum();
                    for i in 0..n_components {
                        w_new[i] -= dot * prev_w[i];
                    }
                }

                // Normalize
                let norm = w_new.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    w_new /= norm;
                }

                // Check convergence
                let dot_product: f64 = w.iter().zip(w_new.iter()).map(|(a, b)| a * b).sum();
                if (1.0 - dot_product.abs()) < self.tol {
                    w = w_new;
                    break;
                }

                w = w_new;
            }

            // Update unmixing matrix
            for i in 0..n_components {
                unmixing[[comp_idx, i]] = w[i];
            }
        }

        Ok(unmixing)
    }

    /// Symmetric decorrelation (orthogonalization).
    fn symmetric_decorrelation(&self, w: &Array2<f64>) -> Result<Array2<f64>> {
        let (u, _s, vt) = self.svd(w)?;

        // W = U @ V^T
        let result = u.dot(&vt);
        Ok(result)
    }

    /// Simplified SVD using power iteration (for small matrices).
    fn svd(&self, matrix: &Array2<f64>) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>)> {
        let (rows, cols) = matrix.dim();
        let k = rows.min(cols);

        // For simplicity, use eigenvalue decomposition for square matrices
        // or approximate for rectangular ones
        if rows == cols {
            // Compute eigenvalues and eigenvectors of A @ A^T
            let aat = matrix.dot(&matrix.t());
            let (eigenvectors, eigenvalues) = self.eigen_decomposition(&aat)?;

            let singular_values: Vec<f64> = eigenvalues.iter().map(|&x| x.sqrt()).collect();

            // V^T = (1/s) * U^T @ A
            let mut vt = Array2::zeros((k, cols));
            for i in 0..k {
                if singular_values[i] > 1e-10 {
                    let u_row = eigenvectors.column(i);
                    for j in 0..cols {
                        let mut sum = 0.0;
                        for l in 0..rows {
                            sum += u_row[l] * matrix[[l, j]];
                        }
                        vt[[i, j]] = sum / singular_values[i];
                    }
                }
            }

            Ok((eigenvectors, singular_values, vt))
        } else {
            // Simplified case - return identity-like matrices
            let u = Array2::eye(rows);
            let s = vec![1.0; k];
            let vt = Array2::eye(cols);
            Ok((u, s, vt))
        }
    }

    /// Eigenvalue decomposition using power iteration.
    fn eigen_decomposition(&self, matrix: &Array2<f64>) -> Result<(Array2<f64>, Vec<f64>)> {
        let n = matrix.nrows();
        let mut eigenvectors = Array2::zeros((n, n));
        let mut eigenvalues = Vec::with_capacity(n);

        let mut rng = rand::rng();

        for k in 0..n {
            // Random initial vector
            let mut v: Array1<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();

            // Normalize
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v /= norm;

            // Power iteration
            for _ in 0..100 {
                let mut av = Array1::zeros(n);
                for i in 0..n {
                    for j in 0..n {
                        av[i] += matrix[[i, j]] * v[j];
                    }
                }

                // Orthogonalize against previous eigenvectors
                for prev_k in 0..k {
                    let prev_v = eigenvectors.column(prev_k);
                    let dot: f64 = av.iter().zip(prev_v.iter()).map(|(a, b)| a * b).sum();
                    for i in 0..n {
                        av[i] -= dot * prev_v[i];
                    }
                }

                let norm = av.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                v = av / norm;
            }

            // Compute eigenvalue
            let mut av = Array1::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    av[i] += matrix[[i, j]] * v[j];
                }
            }
            let eigenvalue: f64 = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();

            eigenvalues.push(eigenvalue);
            for i in 0..n {
                eigenvectors[[i, k]] = v[i];
            }
        }

        Ok((eigenvectors, eigenvalues))
    }

    /// Computes pseudo-inverse using SVD.
    /// Moore-Penrose pseudo-inverse.
    ///
    /// Relies on [`svd`](Self::svd), which is a simplified symmetric-eigenvalue
    /// routine and is only valid for SQUARE input. Both call sites pass square
    /// matrices (a covariance and the unmixing matrix); do not extend it to
    /// rectangular input without replacing `svd` first.
    fn pseudo_inverse(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let (u, s, vt) = self.svd(matrix)?;
        let k = s.len();

        // A^+ = V @ S^(-1) @ U^T
        let mut s_inv = Array2::zeros((k, k));
        for i in 0..k {
            if s[i] > 1e-10 {
                s_inv[[i, i]] = 1.0 / s[i];
            }
        }

        let result = vt.t().dot(&s_inv).dot(&u.t());
        Ok(result)
    }
}

/// Result of ICA decomposition.
#[derive(Debug, Clone)]
pub struct ICAResult {
    /// Mixing matrix A where X = A @ S
    pub mixing_matrix: Array2<f64>,
    /// Unmixing matrix W where S = W @ X
    pub unmixing_matrix: Array2<f64>,
}

impl ICAResult {
    /// Creates a new ICA result.
    pub fn new(mixing_matrix: Array2<f64>, unmixing_matrix: Array2<f64>) -> Self {
        Self {
            mixing_matrix,
            unmixing_matrix,
        }
    }

    /// Returns the number of components.
    pub fn n_components(&self) -> usize {
        self.unmixing_matrix.nrows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_nonlinear_functions() {
        let x = 0.5;

        let logcosh = NonlinearFunction::LogCosh;
        assert!(logcosh.g(x) > 0.0);
        assert!(logcosh.g_prime(x) > 0.0 && logcosh.g_prime(x) < 1.0);

        let exp = NonlinearFunction::Exp;
        assert!(exp.g(x) < 0.0);

        let cube = NonlinearFunction::Cube;
        assert_relative_eq!(cube.g(x), 0.125, epsilon = 1e-10);
        assert_relative_eq!(cube.g_prime(x), 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_fastica_initialization() {
        let ica = FastICA::new(3)
            .with_max_iter(100)
            .with_tol(1e-5)
            .with_whiten(true);

        assert_eq!(ica.n_components, 3);
        assert_eq!(ica.max_iter, 100);
        assert_eq!(ica.tol, 1e-5);
        assert!(ica.whiten);
    }

    #[test]
    fn test_ica_simple_mixture() {
        // Create two independent source signals
        let n_samples = 1000;
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / 100.0).collect();

        let s1: Vec<f64> = t.iter().map(|&t| (2.0 * PI * 1.0 * t).sin()).collect();
        let s2: Vec<f64> = t.iter().map(|&t| if (t * 3.0) as i32 % 2 == 0 { 1.0 } else { -1.0 }).collect();

        // Create mixing matrix
        let mixing = Array2::from_shape_vec((2, 2), vec![0.8, 0.6, 0.4, 0.9]).unwrap();

        // Create mixed signals
        let mut sources = Array2::zeros((2, n_samples));
        for i in 0..n_samples {
            sources[[0, i]] = s1[i];
            sources[[1, i]] = s2[i];
        }

        let mixed = mixing.dot(&sources);

        // Apply ICA
        let mut ica = FastICA::new(2).with_max_iter(100);
        let result = ica.fit_transform(&mixed);

        assert!(result.is_ok());
        let separated = result.unwrap();

        assert_eq!(separated.dim(), (2, n_samples));

        // Check that components are separated (they should be uncorrelated)
        let comp1 = separated.row(0);
        let comp2 = separated.row(1);

        let mean1: f64 = comp1.iter().sum::<f64>() / n_samples as f64;
        let mean2: f64 = comp2.iter().sum::<f64>() / n_samples as f64;

        // Components should have near-zero mean
        assert!(mean1.abs() < 0.5);
        assert!(mean2.abs() < 0.5);
    }

    #[test]
    fn test_ica_get_result() {
        let n_samples = 500;
        let n_features = 3;

        // Create random data
        let mut rng = rand::rng();
        let mut data = Array2::zeros((n_features, n_samples));
        for i in 0..n_features {
            for j in 0..n_samples {
                data[[i, j]] = rng.random_range(-1.0..1.0);
            }
        }

        let mut ica = FastICA::new(2);
        ica.fit(&data).unwrap();

        let result = ica.get_result();
        assert!(result.is_ok());

        let ica_result = result.unwrap();
        assert_eq!(ica_result.n_components(), 2);
        assert_eq!(ica_result.unmixing_matrix.nrows(), 2);

        // The mixing matrix maps components back to the ORIGINAL feature space,
        // so it must be (n_features x n_components).
        assert_eq!(
            ica_result.mixing_matrix.dim(),
            (n_features, 2),
            "mixing matrix must map components back to feature space"
        );
    }

    #[test]
    fn test_ica_transform_without_fit() {
        let ica = FastICA::new(2);
        let data = Array2::zeros((2, 100));

        let result = ica.transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_ica_invalid_dimensions() {
        let mut ica = FastICA::new(10);
        let data = Array2::zeros((2, 5)); // Too few samples

        let result = ica.fit(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_ica_result_creation() {
        let mixing = Array2::eye(3);
        let unmixing = Array2::eye(3);

        let result = ICAResult::new(mixing, unmixing);
        assert_eq!(result.n_components(), 3);
    }
}
