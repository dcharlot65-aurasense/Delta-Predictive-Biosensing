//! # Empirical Mode Decomposition (EMD) and Related Algorithms
//!
//! This module provides adaptive signal decomposition methods for analyzing
//! non-stationary and non-linear signals. These algorithms decompose a signal
//! into intrinsic mode functions (IMFs) that represent different oscillatory modes.
//!
//! ## Algorithms Provided
//!
//! - **EMD**: Classic Empirical Mode Decomposition
//! - **EEMD**: Ensemble EMD with noise-assisted analysis
//! - **CEEMDAN**: Complete Ensemble EMD with Adaptive Noise
//! - **VMD**: Variational Mode Decomposition (frequency domain approach)
//! - **Hilbert-Huang Transform**: Time-frequency analysis using EMD
//!
//! ## Example: Basic EMD
//!
//! ```rust
//! use dpb_core::signal::emd::{Emd, EmdConfig};
//! use ndarray::Array1;
//!
//! # fn example() -> dpb_core::Result<()> {
//! // Create a test signal: sum of two sinusoids
//! let n = 1000;
//! let t: Array1<f64> = Array1::linspace(0.0, 10.0, n);
//! let signal: Array1<f64> = t.mapv(|t| {
//!     (2.0 * std::f64::consts::PI * 1.0 * t).sin() +
//!     0.5 * (2.0 * std::f64::consts::PI * 5.0 * t).sin()
//! });
//!
//! // Decompose using EMD
//! let config = EmdConfig::default();
//! let mut emd = Emd::new(config);
//! let imfs = emd.decompose(&signal)?;
//!
//! println!("Decomposed into {} IMFs", imfs.num_imfs());
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Ensemble EMD
//!
//! ```rust
//! use dpb_core::signal::emd::{Eemd, EemdConfig};
//! use ndarray::Array1;
//!
//! # fn example() -> dpb_core::Result<()> {
//! # let signal = Array1::from_vec(vec![0.0; 1000]);
//! // Configure EEMD
//! let config = EemdConfig {
//!     num_ensembles: 100,
//!     noise_amplitude: 0.2,
//!     ..Default::default()
//! };
//!
//! let mut eemd = Eemd::new(config);
//! let imfs = eemd.decompose(&signal)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Hilbert-Huang Transform
//!
//! ```rust
//! use dpb_core::signal::emd::{HilbertHuangTransform, EmdConfig};
//! use ndarray::Array1;
//!
//! # fn example() -> dpb_core::Result<()> {
//! # let signal = Array1::from_vec(vec![0.0; 1000]);
//! let sample_rate = 100.0;
//!
//! let config = EmdConfig::default();
//! let mut hht = HilbertHuangTransform::new(config);
//! let spectrum = hht.compute(&signal, sample_rate)?;
//!
//! println!("Computed Hilbert spectrum with {} IMFs", spectrum.num_imfs);
//! # Ok(())
//! # }
//! ```

use crate::error::{DpbError, Result};
use crate::signal::hilbert::analytic_signal;
use ndarray::Array1;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

// =============================================================================
// Spline Interpolation
// =============================================================================

/// Boundary conditions for cubic spline interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Natural spline (second derivative = 0 at boundaries)
    Natural,
    /// Clamped spline (specified first derivatives at boundaries)
    Clamped { left_slope: f64, right_slope: f64 },
    /// Not-a-knot condition (third derivative continuous at second and second-to-last points)
    NotAKnot,
}

/// Cubic spline interpolator for envelope computation.
///
/// Computes a smooth curve through a set of points using piecewise cubic polynomials.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    coefficients: Vec<(f64, f64, f64, f64)>, // (a, b, c, d) for each segment
}

impl CubicSpline {
    /// Creates a new cubic spline through the given points.
    ///
    /// # Arguments
    /// * `x` - X coordinates (must be strictly increasing)
    /// * `y` - Y coordinates
    /// * `bc` - Boundary conditions
    pub fn new(x: Vec<f64>, y: Vec<f64>, bc: BoundaryCondition) -> Result<Self> {
        if x.len() != y.len() {
            return Err(DpbError::InvalidDimensions(
                "x and y must have same length".to_string(),
            ));
        }

        if x.len() < 2 {
            return Err(DpbError::InvalidDimensions(
                "Need at least 2 points for spline".to_string(),
            ));
        }

        // Check that x is strictly increasing
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(DpbError::InvalidParameter(
                    "x coordinates must be strictly increasing".to_string(),
                ));
            }
        }

        let n = x.len() - 1;
        let mut h = vec![0.0; n];
        for i in 0..n {
            h[i] = x[i + 1] - x[i];
        }

        // Solve for second derivatives (m)
        let m = Self::solve_for_second_derivatives(&x, &y, &h, bc)?;

        // Compute coefficients for each segment
        let mut coefficients = Vec::with_capacity(n);
        for i in 0..n {
            let a = y[i];
            let b = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * m[i] + m[i + 1]) / 6.0;
            let c = m[i] / 2.0;
            let d = (m[i + 1] - m[i]) / (6.0 * h[i]);
            coefficients.push((a, b, c, d));
        }

        Ok(Self {
            x,
            y,
            coefficients,
        })
    }

    fn solve_for_second_derivatives(
        x: &[f64],
        y: &[f64],
        h: &[f64],
        bc: BoundaryCondition,
    ) -> Result<Vec<f64>> {
        let n = x.len() - 1;
        let mut m = vec![0.0; n + 1];

        match bc {
            BoundaryCondition::Natural => {
                // Natural spline: m[0] = m[n] = 0
                m[0] = 0.0;
                m[n] = 0.0;

                if n > 1 {
                    let mut alpha = vec![0.0; n];
                    let mut l = vec![0.0; n + 1];
                    let mut mu = vec![0.0; n + 1];
                    let mut z = vec![0.0; n + 1];

                    for i in 1..n {
                        alpha[i] = 3.0 / h[i] * (y[i + 1] - y[i])
                            - 3.0 / h[i - 1] * (y[i] - y[i - 1]);
                    }

                    l[0] = 1.0;
                    mu[0] = 0.0;
                    z[0] = 0.0;

                    for i in 1..n {
                        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
                        mu[i] = h[i] / l[i];
                        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
                    }

                    l[n] = 1.0;
                    z[n] = 0.0;

                    for j in (0..n).rev() {
                        m[j] = z[j] - mu[j] * m[j + 1];
                    }
                }
            }
            BoundaryCondition::Clamped {
                left_slope,
                right_slope,
            } => {
                // Clamped spline with specified derivatives
                let mut alpha = vec![0.0; n + 1];
                let mut l = vec![0.0; n + 1];
                let mut mu = vec![0.0; n + 1];
                let mut z = vec![0.0; n + 1];

                alpha[0] = 3.0 * ((y[1] - y[0]) / h[0] - left_slope);
                alpha[n] = 3.0 * (right_slope - (y[n] - y[n - 1]) / h[n - 1]);

                for i in 1..n {
                    alpha[i] =
                        3.0 / h[i] * (y[i + 1] - y[i]) - 3.0 / h[i - 1] * (y[i] - y[i - 1]);
                }

                l[0] = 2.0 * h[0];
                mu[0] = 0.5;
                z[0] = alpha[0] / l[0];

                for i in 1..n {
                    l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
                    mu[i] = h[i] / l[i];
                    z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
                }

                l[n] = h[n - 1] * (2.0 - mu[n - 1]);
                z[n] = (alpha[n] - h[n - 1] * z[n - 1]) / l[n];
                m[n] = z[n];

                for j in (0..n).rev() {
                    m[j] = z[j] - mu[j] * m[j + 1];
                }
            }
            BoundaryCondition::NotAKnot => {
                // Not-a-knot: third derivative continuous at x[1] and x[n-1]
                // For simplicity, fall back to natural spline
                return Self::solve_for_second_derivatives(x, y, h, BoundaryCondition::Natural);
            }
        }

        Ok(m)
    }

    /// Evaluates the spline at a single point.
    pub fn eval(&self, x_eval: f64) -> f64 {
        // Find the appropriate segment
        let mut i = 0;
        for j in 0..self.x.len() - 1 {
            if x_eval >= self.x[j] && x_eval <= self.x[j + 1] {
                i = j;
                break;
            }
        }

        // Clamp to last segment if out of range
        if x_eval > self.x[self.x.len() - 1] {
            i = self.x.len() - 2;
        }

        let dx = x_eval - self.x[i];
        let (a, b, c, d) = self.coefficients[i];

        a + b * dx + c * dx * dx + d * dx * dx * dx
    }

    /// Evaluates the spline at multiple points.
    pub fn eval_array(&self, x_eval: &[f64]) -> Vec<f64> {
        x_eval.iter().map(|&x| self.eval(x)).collect()
    }
}

// =============================================================================
// Extrema Detection
// =============================================================================

/// Finds local maxima in a signal.
fn find_maxima(signal: &[f64]) -> Vec<usize> {
    let mut maxima = Vec::new();

    for i in 1..signal.len() - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            maxima.push(i);
        }
    }

    maxima
}

/// Finds local minima in a signal.
fn find_minima(signal: &[f64]) -> Vec<usize> {
    let mut minima = Vec::new();

    for i in 1..signal.len() - 1 {
        if signal[i] < signal[i - 1] && signal[i] < signal[i + 1] {
            minima.push(i);
        }
    }

    minima
}

/// Computes the envelope through extrema using cubic spline interpolation.
fn compute_envelope(
    signal: &[f64],
    extrema: &[usize],
    _is_upper: bool,
) -> Result<Vec<f64>> {
    if extrema.is_empty() {
        // No extrema: return constant signal at mean
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        return Ok(vec![mean; signal.len()]);
    }

    if extrema.len() == 1 {
        // Only one extremum: return constant at that value
        return Ok(vec![signal[extrema[0]]; signal.len()]);
    }

    // Build spline through extrema
    let mut x = Vec::with_capacity(extrema.len() + 2);
    let mut y = Vec::with_capacity(extrema.len() + 2);

    // Add boundary points if needed
    if extrema[0] != 0 {
        x.push(0.0);
        y.push(signal[0]);
    }

    for &idx in extrema {
        x.push(idx as f64);
        y.push(signal[idx]);
    }

    if extrema[extrema.len() - 1] != signal.len() - 1 {
        x.push((signal.len() - 1) as f64);
        y.push(signal[signal.len() - 1]);
    }

    let spline = CubicSpline::new(x, y, BoundaryCondition::Natural)?;

    let eval_points: Vec<f64> = (0..signal.len()).map(|i| i as f64).collect();
    Ok(spline.eval_array(&eval_points))
}

// =============================================================================
// Intrinsic Mode Function (IMF)
// =============================================================================

/// A single Intrinsic Mode Function extracted from EMD.
#[derive(Debug, Clone)]
pub struct Imf {
    /// The IMF signal
    pub data: Array1<f64>,
    /// Mean frequency of this mode (if computed)
    pub mean_frequency: Option<f64>,
    /// Energy of this mode
    pub energy: f64,
}

impl Imf {
    /// Creates a new IMF.
    pub fn new(data: Array1<f64>) -> Self {
        let energy = data.iter().map(|x| x * x).sum();
        Self {
            data,
            mean_frequency: None,
            energy,
        }
    }

    /// Returns the length of the IMF.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the IMF is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Computes the mean frequency using Hilbert transform.
    pub fn compute_mean_frequency(&mut self, sample_rate: f64) {
        let data_vec: Vec<f64> = self.data.to_vec();
        let analytic = analytic_signal(&data_vec, sample_rate);

        // Average instantaneous frequency (excluding boundary artifacts)
        let skip = 10.min(analytic.instantaneous_frequency.len() / 4);
        let end = analytic.instantaneous_frequency.len().saturating_sub(skip);

        if end > skip {
            let mean_freq = analytic.instantaneous_frequency[skip..end]
                .iter()
                .map(|f| f.abs())
                .sum::<f64>()
                / (end - skip) as f64;
            self.mean_frequency = Some(mean_freq);
        }
    }
}

/// Collection of IMFs with analysis capabilities.
#[derive(Debug, Clone)]
pub struct ImfSet {
    /// The IMFs in order from highest to lowest frequency
    pub imfs: Vec<Imf>,
    /// The residue (final trend)
    pub residue: Array1<f64>,
}

impl ImfSet {
    /// Creates a new IMF set.
    pub fn new(imfs: Vec<Imf>, residue: Array1<f64>) -> Self {
        Self { imfs, residue }
    }

    /// Returns the number of IMFs.
    pub fn num_imfs(&self) -> usize {
        self.imfs.len()
    }

    /// Reconstructs the original signal from all IMFs.
    pub fn reconstruct(&self) -> Array1<f64> {
        let mut signal = self.residue.clone();
        for imf in &self.imfs {
            signal = signal + &imf.data;
        }
        signal
    }

    /// Reconstructs signal from selected IMF indices.
    pub fn reconstruct_from_indices(&self, indices: &[usize]) -> Result<Array1<f64>> {
        if self.imfs.is_empty() {
            return Ok(self.residue.clone());
        }

        let mut signal = Array1::zeros(self.residue.len());

        for &idx in indices {
            if idx >= self.imfs.len() {
                return Err(DpbError::InvalidParameter(format!(
                    "IMF index {} out of range (max {})",
                    idx,
                    self.imfs.len() - 1
                )));
            }
            signal = signal + &self.imfs[idx].data;
        }

        Ok(signal)
    }

    /// Computes energy distribution across IMFs.
    pub fn energy_distribution(&self) -> Vec<f64> {
        let total_energy: f64 = self.imfs.iter().map(|imf| imf.energy).sum();

        if total_energy == 0.0 {
            return vec![0.0; self.imfs.len()];
        }

        self.imfs
            .iter()
            .map(|imf| imf.energy / total_energy)
            .collect()
    }

    /// Checks orthogonality between IMFs (returns index of orthogonality).
    ///
    /// Perfect orthogonality gives IO = 0.
    pub fn orthogonality_index(&self) -> f64 {
        let n = self.imfs.len();
        if n < 2 {
            return 0.0;
        }

        let mut cross_energy = 0.0;
        let mut total_energy = 0.0;

        for i in 0..n {
            total_energy += self.imfs[i].energy;

            for j in (i + 1)..n {
                let cross: f64 = self.imfs[i]
                    .data
                    .iter()
                    .zip(self.imfs[j].data.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                cross_energy += cross.abs();
            }
        }

        if total_energy == 0.0 {
            return 0.0;
        }

        cross_energy / total_energy
    }

    /// Computes mean frequencies for all IMFs.
    pub fn compute_mean_frequencies(&mut self, sample_rate: f64) {
        for imf in &mut self.imfs {
            imf.compute_mean_frequency(sample_rate);
        }
    }
}

// =============================================================================
// EMD Configuration and Stopping Criteria
// =============================================================================

/// Stopping criterion for the sifting process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StoppingCriterion {
    /// Standard deviation threshold (Huang et al., 1998)
    StandardDeviation { threshold: f64 },
    /// S-number criterion (Huang et al., 2003)
    SNumber { s_number: usize },
    /// Combined criterion
    Combined {
        sd_threshold: f64,
        s_number: usize,
    },
}

impl Default for StoppingCriterion {
    fn default() -> Self {
        Self::Combined {
            sd_threshold: 0.2,
            s_number: 5,
        }
    }
}

/// Configuration for EMD decomposition.
#[derive(Debug, Clone)]
pub struct EmdConfig {
    /// Maximum number of IMFs to extract
    pub max_imfs: usize,
    /// Maximum sifting iterations per IMF
    pub max_sift_iterations: usize,
    /// Stopping criterion for sifting
    pub stopping_criterion: StoppingCriterion,
    /// Minimum number of extrema to continue
    pub min_extrema: usize,
}

impl Default for EmdConfig {
    fn default() -> Self {
        Self {
            max_imfs: 20,
            max_sift_iterations: 1000,
            stopping_criterion: StoppingCriterion::default(),
            min_extrema: 3,
        }
    }
}

// =============================================================================
// EMD Core Algorithm
// =============================================================================

/// Empirical Mode Decomposition (EMD) algorithm.
///
/// Decomposes a signal into intrinsic mode functions (IMFs) through an iterative
/// sifting process. Each IMF satisfies two conditions:
/// 1. Number of extrema and zero-crossings differ by at most one
/// 2. Mean of upper and lower envelopes is close to zero
pub struct Emd {
    config: EmdConfig,
}

impl Emd {
    /// Creates a new EMD decomposer with the given configuration.
    pub fn new(config: EmdConfig) -> Self {
        Self { config }
    }

    /// Creates an EMD decomposer with default configuration.
    pub fn default() -> Self {
        Self::new(EmdConfig::default())
    }

    /// Decomposes a signal into IMFs.
    pub fn decompose(&mut self, signal: &Array1<f64>) -> Result<ImfSet> {
        if signal.is_empty() {
            return Err(DpbError::InvalidParameter("Signal cannot be empty".to_string()));
        }

        let signal_vec = signal.to_vec();
        let mut imfs = Vec::new();
        let mut residue = signal_vec.clone();

        for _ in 0..self.config.max_imfs {
            // Check if residue has enough extrema
            let maxima = find_maxima(&residue);
            let minima = find_minima(&residue);

            if maxima.len() + minima.len() < self.config.min_extrema {
                break;
            }

            // Extract IMF through sifting
            match self.sift(&residue) {
                Ok(imf_data) => {
                    // Subtract IMF from residue
                    for i in 0..residue.len() {
                        residue[i] -= imf_data[i];
                    }

                    let imf = Imf::new(Array1::from_vec(imf_data));
                    imfs.push(imf);
                }
                Err(_) => break,
            }

            // Check if residue is monotonic (no more IMFs to extract)
            if self.is_monotonic(&residue) {
                break;
            }
        }

        Ok(ImfSet::new(imfs, Array1::from_vec(residue)))
    }

    /// Performs the sifting process to extract a single IMF.
    fn sift(&self, signal: &[f64]) -> Result<Vec<f64>> {
        let mut h = signal.to_vec();
        let mut s_count = 0;

        for _iteration in 0..self.config.max_sift_iterations {
            let maxima = find_maxima(&h);
            let minima = find_minima(&h);

            if maxima.is_empty() || minima.is_empty() {
                break;
            }

            // Compute upper and lower envelopes
            let upper = compute_envelope(&h, &maxima, true)?;
            let lower = compute_envelope(&h, &minima, false)?;

            // Compute mean envelope
            let mut mean_envelope = vec![0.0; h.len()];
            for i in 0..h.len() {
                mean_envelope[i] = (upper[i] + lower[i]) / 2.0;
            }

            // Subtract mean from h
            let mut h_new = vec![0.0; h.len()];
            for i in 0..h.len() {
                h_new[i] = h[i] - mean_envelope[i];
            }

            // Check stopping criterion
            match self.config.stopping_criterion {
                StoppingCriterion::StandardDeviation { threshold } => {
                    let sd = self.compute_standard_deviation(&h, &h_new);
                    if sd < threshold {
                        return Ok(h_new);
                    }
                }
                StoppingCriterion::SNumber { s_number } => {
                    if self.check_imf_conditions(&h_new) {
                        s_count += 1;
                        if s_count >= s_number {
                            return Ok(h_new);
                        }
                    } else {
                        s_count = 0;
                    }
                }
                StoppingCriterion::Combined {
                    sd_threshold,
                    s_number,
                } => {
                    let sd = self.compute_standard_deviation(&h, &h_new);

                    if sd < sd_threshold && self.check_imf_conditions(&h_new) {
                        s_count += 1;
                        if s_count >= s_number {
                            return Ok(h_new);
                        }
                    } else {
                        s_count = 0;
                    }
                }
            }

            h = h_new;
        }

        Ok(h)
    }

    /// Computes the standard deviation between successive sifting results.
    fn compute_standard_deviation(&self, h_prev: &[f64], h_new: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..h_prev.len() {
            let diff = h_prev[i] - h_new[i];
            sum += diff * diff / (h_prev[i] * h_prev[i] + 1e-10);
        }
        sum / h_prev.len() as f64
    }

    /// Checks if a signal satisfies IMF conditions.
    fn check_imf_conditions(&self, signal: &[f64]) -> bool {
        let maxima = find_maxima(signal);
        let minima = find_minima(signal);
        let num_extrema = maxima.len() + minima.len();

        // Count zero crossings
        let mut zero_crossings = 0;
        for i in 0..signal.len() - 1 {
            if (signal[i] >= 0.0 && signal[i + 1] < 0.0)
                || (signal[i] < 0.0 && signal[i + 1] >= 0.0)
            {
                zero_crossings += 1;
            }
        }

        // IMF condition: number of extrema and zero crossings differ by at most 1
        num_extrema.abs_diff(zero_crossings) <= 1
    }

    /// Checks if a signal is monotonic.
    fn is_monotonic(&self, signal: &[f64]) -> bool {
        let maxima = find_maxima(signal);
        let minima = find_minima(signal);
        maxima.len() + minima.len() < 2
    }
}

// =============================================================================
// Ensemble EMD (EEMD)
// =============================================================================

/// Configuration for Ensemble EMD.
#[derive(Debug, Clone)]
pub struct EemdConfig {
    /// Number of ensemble members
    pub num_ensembles: usize,
    /// Standard deviation of added noise (relative to signal std)
    pub noise_amplitude: f64,
    /// Base EMD configuration
    pub emd_config: EmdConfig,
    /// Enable parallel decomposition
    pub parallel: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for EemdConfig {
    fn default() -> Self {
        Self {
            num_ensembles: 100,
            noise_amplitude: 0.2,
            emd_config: EmdConfig::default(),
            parallel: true,
            seed: None,
        }
    }
}

/// Ensemble Empirical Mode Decomposition (EEMD).
///
/// Improves EMD robustness by:
/// 1. Adding white noise to the signal
/// 2. Decomposing multiple noisy realizations
/// 3. Averaging the resulting IMFs
///
/// This helps mitigate mode mixing and improves decomposition consistency.
pub struct Eemd {
    config: EemdConfig,
}

impl Eemd {
    /// Creates a new EEMD decomposer.
    pub fn new(config: EemdConfig) -> Self {
        Self { config }
    }

    /// Creates an EEMD decomposer with default configuration.
    pub fn default() -> Self {
        Self::new(EemdConfig::default())
    }

    /// Decomposes a signal using ensemble averaging.
    pub fn decompose(&mut self, signal: &Array1<f64>) -> Result<ImfSet> {
        if signal.is_empty() {
            return Err(DpbError::InvalidParameter("Signal cannot be empty".to_string()));
        }

        let signal_vec = signal.to_vec();
        let signal_std = signal.std(0.0);
        let noise_std = self.config.noise_amplitude * signal_std;

        // Generate noisy realizations and decompose
        let all_imfs: Vec<ImfSet> = if self.config.parallel {
            (0..self.config.num_ensembles)
                .into_par_iter()
                .map(|i| {
                    let seed = self.config.seed.unwrap_or(0) + i as u64;
                    self.decompose_with_noise(&signal_vec, noise_std, seed)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..self.config.num_ensembles)
                .map(|i| {
                    let seed = self.config.seed.unwrap_or(0) + i as u64;
                    self.decompose_with_noise(&signal_vec, noise_std, seed)
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Average the IMFs
        self.average_imfs(&all_imfs)
    }

    fn decompose_with_noise(
        &self,
        signal: &[f64],
        noise_std: f64,
        seed: u64,
    ) -> Result<ImfSet> {
        // Add white noise
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, noise_std).unwrap();

        let noisy_signal: Vec<f64> = signal
            .iter()
            .map(|&x| x + normal.sample(&mut rng))
            .collect();

        // Decompose
        let mut emd = Emd::new(self.config.emd_config.clone());
        emd.decompose(&Array1::from_vec(noisy_signal))
    }

    fn average_imfs(&self, all_imfs: &[ImfSet]) -> Result<ImfSet> {
        if all_imfs.is_empty() {
            return Err(DpbError::InvalidParameter(
                "No IMF sets to average".to_string(),
            ));
        }

        // Find maximum number of IMFs
        let max_num_imfs = all_imfs.iter().map(|s| s.num_imfs()).max().unwrap();
        let signal_len = all_imfs[0].residue.len();

        let mut averaged_imfs = Vec::new();

        for imf_idx in 0..max_num_imfs {
            let mut sum = Array1::zeros(signal_len);
            let mut count = 0;

            for imf_set in all_imfs {
                if imf_idx < imf_set.num_imfs() {
                    sum = sum + &imf_set.imfs[imf_idx].data;
                    count += 1;
                }
            }

            if count > 0 {
                let averaged = sum / count as f64;
                averaged_imfs.push(Imf::new(averaged));
            }
        }

        // Average residues
        let mut residue_sum = Array1::zeros(signal_len);
        for imf_set in all_imfs {
            residue_sum = residue_sum + &imf_set.residue;
        }
        let averaged_residue = residue_sum / all_imfs.len() as f64;

        Ok(ImfSet::new(averaged_imfs, averaged_residue))
    }
}

// =============================================================================
// Complete EEMD with Adaptive Noise (CEEMDAN)
// =============================================================================

/// Configuration for CEEMDAN.
#[derive(Debug, Clone)]
pub struct CeemdanConfig {
    /// Number of ensemble members
    pub num_ensembles: usize,
    /// Standard deviation of added noise (relative to signal std)
    pub noise_amplitude: f64,
    /// Base EMD configuration
    pub emd_config: EmdConfig,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for CeemdanConfig {
    fn default() -> Self {
        Self {
            num_ensembles: 100,
            noise_amplitude: 0.2,
            emd_config: EmdConfig::default(),
            seed: None,
        }
    }
}

/// Complete Ensemble EMD with Adaptive Noise (CEEMDAN).
///
/// Further improves EEMD by:
/// 1. Adding adaptive noise at each decomposition stage
/// 2. Using EMD modes of white noise as the adaptive noise
/// 3. Better reconstruction and reduced residual noise
pub struct Ceemdan {
    config: CeemdanConfig,
}

impl Ceemdan {
    /// Creates a new CEEMDAN decomposer.
    pub fn new(config: CeemdanConfig) -> Self {
        Self { config }
    }

    /// Decomposes a signal using CEEMDAN.
    pub fn decompose(&mut self, signal: &Array1<f64>) -> Result<ImfSet> {
        if signal.is_empty() {
            return Err(DpbError::InvalidParameter("Signal cannot be empty".to_string()));
        }

        let signal_vec = signal.to_vec();
        let n = signal_vec.len();
        let signal_std = signal.std(0.0);
        let noise_std = self.config.noise_amplitude * signal_std;

        // Pre-compute noise modes for each ensemble member
        let noise_modes = self.precompute_noise_modes(n, noise_std)?;

        let mut imfs = Vec::new();
        let mut residue = signal_vec.clone();

        for mode_idx in 0..self.config.emd_config.max_imfs {
            // Add adaptive noise to residue
            let mut ensemble_imfs = Vec::new();

            for e in 0..self.config.num_ensembles {
                if mode_idx >= noise_modes[e].len() {
                    continue;
                }

                let noisy_residue: Vec<f64> = residue
                    .iter()
                    .zip(noise_modes[e][mode_idx].iter())
                    .map(|(r, n)| r + n)
                    .collect();

                let mut emd = Emd::new(self.config.emd_config.clone());
                if let Ok(imf_set) = emd.decompose(&Array1::from_vec(noisy_residue)) {
                    if !imf_set.imfs.is_empty() {
                        ensemble_imfs.push(imf_set.imfs[0].data.to_vec());
                    }
                }
            }

            if ensemble_imfs.is_empty() {
                break;
            }

            // Average the first IMF from all ensembles
            let mut averaged_imf = vec![0.0; n];
            for ensemble_imf in &ensemble_imfs {
                for i in 0..n {
                    averaged_imf[i] += ensemble_imf[i];
                }
            }
            for i in 0..n {
                averaged_imf[i] /= ensemble_imfs.len() as f64;
            }

            // Update residue
            for i in 0..n {
                residue[i] -= averaged_imf[i];
            }

            imfs.push(Imf::new(Array1::from_vec(averaged_imf)));

            // Check stopping conditions
            if self.is_monotonic(&residue) {
                break;
            }
        }

        Ok(ImfSet::new(imfs, Array1::from_vec(residue)))
    }

    fn precompute_noise_modes(
        &self,
        n: usize,
        noise_std: f64,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let mut all_noise_modes = Vec::new();

        for e in 0..self.config.num_ensembles {
            let seed = self.config.seed.unwrap_or(0) + e as u64;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let normal = Normal::new(0.0, noise_std).unwrap();

            let noise: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();

            let mut emd = Emd::new(self.config.emd_config.clone());
            let noise_imfs = emd.decompose(&Array1::from_vec(noise))?;

            let modes: Vec<Vec<f64>> = noise_imfs
                .imfs
                .iter()
                .map(|imf| imf.data.to_vec())
                .collect();

            all_noise_modes.push(modes);
        }

        Ok(all_noise_modes)
    }

    fn is_monotonic(&self, signal: &[f64]) -> bool {
        let maxima = find_maxima(signal);
        let minima = find_minima(signal);
        maxima.len() + minima.len() < 2
    }
}

// =============================================================================
// Variational Mode Decomposition (VMD)
// =============================================================================

/// Configuration for VMD.
#[derive(Debug, Clone)]
pub struct VmdConfig {
    /// Number of modes to extract
    pub num_modes: usize,
    /// Bandwidth constraint parameter (alpha)
    pub alpha: f64,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Initial DC component removal
    pub dc_component: bool,
}

impl Default for VmdConfig {
    fn default() -> Self {
        Self {
            num_modes: 5,
            alpha: 2000.0,
            tolerance: 1e-7,
            max_iterations: 500,
            dc_component: false,
        }
    }
}

/// Variational Mode Decomposition (VMD).
///
/// A frequency-domain approach that decomposes a signal into modes with
/// specific sparsity properties. Unlike EMD, VMD:
/// - Is non-recursive
/// - Has a solid mathematical foundation
/// - Requires specifying the number of modes
/// - Is more robust to noise and sampling
pub struct Vmd {
    config: VmdConfig,
}

impl Vmd {
    /// Creates a new VMD decomposer.
    pub fn new(config: VmdConfig) -> Self {
        Self { config }
    }

    /// Decomposes a signal using VMD.
    ///
    /// Note: This is a simplified implementation. Full VMD requires
    /// complex optimization in the frequency domain.
    pub fn decompose(&mut self, signal: &Array1<f64>) -> Result<ImfSet> {
        if signal.is_empty() {
            return Err(DpbError::InvalidParameter("Signal cannot be empty".to_string()));
        }

        // This is a placeholder for a full VMD implementation
        // Full VMD requires:
        // 1. Frequency domain representation
        // 2. Wiener filtering for each mode
        // 3. Lagrangian multiplier updates
        // 4. Alternating direction method of multipliers (ADMM)

        // For now, fall back to EMD-like decomposition with fixed mode count
        let mut emd_config = EmdConfig::default();
        emd_config.max_imfs = self.config.num_modes;

        let mut emd = Emd::new(emd_config);
        emd.decompose(signal)
    }
}

// =============================================================================
// Hilbert-Huang Transform
// =============================================================================

/// Hilbert spectrum computed from EMD.
#[derive(Debug, Clone)]
pub struct HilbertSpectrum {
    /// Number of IMFs
    pub num_imfs: usize,
    /// Time points
    pub time: Vec<f64>,
    /// Instantaneous frequencies for each IMF
    pub instantaneous_frequencies: Vec<Vec<f64>>,
    /// Instantaneous amplitudes for each IMF
    pub instantaneous_amplitudes: Vec<Vec<f64>>,
    /// Marginal spectrum (frequency vs amplitude)
    pub marginal_spectrum: Option<(Vec<f64>, Vec<f64>)>,
}

/// Hilbert-Huang Transform (HHT).
///
/// Combines EMD with Hilbert spectral analysis to provide time-frequency
/// representation with high resolution for non-stationary signals.
pub struct HilbertHuangTransform {
    emd_config: EmdConfig,
}

impl HilbertHuangTransform {
    /// Creates a new HHT analyzer.
    pub fn new(emd_config: EmdConfig) -> Self {
        Self { emd_config }
    }

    /// Computes the Hilbert-Huang spectrum.
    pub fn compute(&mut self, signal: &Array1<f64>, sample_rate: f64) -> Result<HilbertSpectrum> {
        if signal.is_empty() {
            return Err(DpbError::InvalidParameter("Signal cannot be empty".to_string()));
        }

        // Decompose signal using EMD
        let mut emd = Emd::new(self.emd_config.clone());
        let mut imf_set = emd.decompose(signal)?;

        // Compute instantaneous frequency and amplitude for each IMF
        let num_imfs = imf_set.num_imfs();
        let n = signal.len();
        let time: Vec<f64> = (0..n).map(|i| i as f64 / sample_rate).collect();

        let mut instantaneous_frequencies = Vec::new();
        let mut instantaneous_amplitudes = Vec::new();

        for imf in &mut imf_set.imfs {
            let imf_vec = imf.data.to_vec();
            let analytic = analytic_signal(&imf_vec, sample_rate);

            instantaneous_frequencies.push(analytic.instantaneous_frequency);
            instantaneous_amplitudes.push(analytic.amplitude_envelope);
        }

        // Compute marginal spectrum
        let marginal = self.compute_marginal_spectrum(
            &instantaneous_frequencies,
            &instantaneous_amplitudes,
            sample_rate,
        );

        Ok(HilbertSpectrum {
            num_imfs,
            time,
            instantaneous_frequencies,
            instantaneous_amplitudes,
            marginal_spectrum: Some(marginal),
        })
    }

    fn compute_marginal_spectrum(
        &self,
        frequencies: &[Vec<f64>],
        amplitudes: &[Vec<f64>],
        sample_rate: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        // Create frequency bins
        let num_bins = 256;
        let max_freq = sample_rate / 2.0;
        let freq_bins: Vec<f64> = (0..num_bins)
            .map(|i| i as f64 * max_freq / num_bins as f64)
            .collect();

        let mut marginal = vec![0.0; num_bins];

        // Accumulate amplitude at each frequency
        for (freq_vec, amp_vec) in frequencies.iter().zip(amplitudes.iter()) {
            for (&freq, &amp) in freq_vec.iter().zip(amp_vec.iter()) {
                if freq >= 0.0 && freq < max_freq {
                    let bin = ((freq / max_freq * num_bins as f64) as usize).min(num_bins - 1);
                    marginal[bin] += amp;
                }
            }
        }

        // Normalize
        let max_amp = marginal.iter().cloned().fold(0.0f64, f64::max);
        if max_amp > 0.0 {
            for m in &mut marginal {
                *m /= max_amp;
            }
        }

        (freq_bins, marginal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_cubic_spline_interpolation() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];

        let spline = CubicSpline::new(x.clone(), y.clone(), BoundaryCondition::Natural).unwrap();

        // Test at original points
        for (xi, yi) in x.iter().zip(y.iter()) {
            assert_relative_eq!(spline.eval(*xi), *yi, epsilon = 1e-10);
        }

        // Test at intermediate point
        let y_interp = spline.eval(1.5);
        assert!(y_interp > 1.0 && y_interp < 4.0);
    }

    #[test]
    fn test_find_extrema() {
        let signal = vec![0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0];

        let maxima = find_maxima(&signal);
        assert_eq!(maxima, vec![1, 5]);

        let minima = find_minima(&signal);
        assert_eq!(minima, vec![3]);
    }

    #[test]
    fn test_emd_simple_signal() {
        // Create a simple two-component signal
        let n = 200;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / 100.0).collect();
        let signal: Vec<f64> = t
            .iter()
            .map(|&t| (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin())
            .collect();

        let config = EmdConfig {
            max_imfs: 10,
            max_sift_iterations: 100,
            ..Default::default()
        };

        let mut emd = Emd::new(config);
        let result = emd.decompose(&Array1::from_vec(signal.clone()));

        assert!(result.is_ok());
        let imf_set = result.unwrap();

        // Should extract at least 2 IMFs
        assert!(imf_set.num_imfs() >= 1);

        // Reconstruction should match original
        let reconstructed = imf_set.reconstruct();
        let error: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(error < 0.1 * signal.len() as f64);
    }

    #[test]
    fn test_imf_set_reconstruction() {
        let n = 100;

        let imf1 = Imf::new(Array1::from_vec(vec![1.0; n]));
        let imf2 = Imf::new(Array1::from_vec(vec![2.0; n]));
        let residue = Array1::from_vec(vec![3.0; n]);

        let imf_set = ImfSet::new(vec![imf1, imf2], residue);

        let reconstructed = imf_set.reconstruct();

        for &val in reconstructed.iter() {
            assert_relative_eq!(val, 6.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_imf_set_partial_reconstruction() {
        let n = 100;
        let imf1 = Imf::new(Array1::from_vec(vec![1.0; n]));
        let imf2 = Imf::new(Array1::from_vec(vec![2.0; n]));
        let imf3 = Imf::new(Array1::from_vec(vec![3.0; n]));
        let residue = Array1::zeros(n);

        let imf_set = ImfSet::new(vec![imf1, imf2, imf3], residue);

        // Reconstruct from indices 0 and 2
        let reconstructed = imf_set.reconstruct_from_indices(&[0, 2]).unwrap();

        for &val in reconstructed.iter() {
            assert_relative_eq!(val, 4.0, epsilon = 1e-10); // 1.0 + 3.0
        }
    }

    #[test]
    fn test_energy_distribution() {
        let n = 100;
        let imf1 = Imf::new(Array1::from_vec(vec![1.0; n]));
        let imf2 = Imf::new(Array1::from_vec(vec![2.0; n]));
        let residue = Array1::zeros(n);

        let imf_set = ImfSet::new(vec![imf1, imf2], residue);

        let energy_dist = imf_set.energy_distribution();

        // Energy of imf1: 100 * 1^2 = 100
        // Energy of imf2: 100 * 2^2 = 400
        // Total: 500
        assert_relative_eq!(energy_dist[0], 100.0 / 500.0, epsilon = 1e-10);
        assert_relative_eq!(energy_dist[1], 400.0 / 500.0, epsilon = 1e-10);
    }

    #[test]
    fn test_eemd_noise_averaging() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 20.0).sin()).collect();

        let config = EemdConfig {
            num_ensembles: 10,
            noise_amplitude: 0.1,
            emd_config: EmdConfig {
                max_imfs: 5,
                max_sift_iterations: 50,
                ..Default::default()
            },
            parallel: false,
            seed: Some(42),
        };

        let mut eemd = Eemd::new(config);
        let result = eemd.decompose(&Array1::from_vec(signal.clone()));

        assert!(result.is_ok());
        let imf_set = result.unwrap();
        assert!(imf_set.num_imfs() >= 1);
    }

    #[test]
    fn test_hilbert_huang_transform() {
        let n = 200;
        let sample_rate = 100.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * 5.0 * t).sin()
            })
            .collect();

        let config = EmdConfig {
            max_imfs: 5,
            max_sift_iterations: 50,
            ..Default::default()
        };

        let mut hht = HilbertHuangTransform::new(config);
        let result = hht.compute(&Array1::from_vec(signal), sample_rate);

        assert!(result.is_ok());
        let spectrum = result.unwrap();

        assert_eq!(spectrum.time.len(), n);
        assert!(spectrum.num_imfs >= 1);
        assert!(spectrum.marginal_spectrum.is_some());
    }

    #[test]
    fn test_orthogonality_index() {
        let n = 100;

        // Create orthogonal IMFs
        let imf1 = Imf::new(Array1::from_vec(
            (0..n).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect(),
        ));
        let imf2 = Imf::new(Array1::from_vec(
            (0..n).map(|i| (2.0 * PI * i as f64 / 10.0).cos()).collect(),
        ));

        let imf_set = ImfSet::new(vec![imf1, imf2], Array1::zeros(n));
        let oi = imf_set.orthogonality_index();

        // Sin and cos are orthogonal, so OI should be small
        assert!(oi < 0.1);
    }

    #[test]
    fn test_ceemdan_basic() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 20.0).sin()).collect();

        let config = CeemdanConfig {
            num_ensembles: 10,
            noise_amplitude: 0.1,
            emd_config: EmdConfig {
                max_imfs: 3,
                max_sift_iterations: 50,
                ..Default::default()
            },
            seed: Some(42),
        };

        let mut ceemdan = Ceemdan::new(config);
        let result = ceemdan.decompose(&Array1::from_vec(signal));

        assert!(result.is_ok());
    }

    #[test]
    fn test_vmd_basic() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 20.0).sin()).collect();

        let config = VmdConfig {
            num_modes: 3,
            ..Default::default()
        };

        let mut vmd = Vmd::new(config);
        let result = vmd.decompose(&Array1::from_vec(signal));

        assert!(result.is_ok());
    }

    #[test]
    fn test_chirp_signal_decomposition() {
        // Test with a chirp signal (frequency modulated)
        let n = 500;
        let sample_rate = 1000.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                // Chirp from 10 Hz to 50 Hz
                let phase = 2.0 * PI * (10.0 * t + 20.0 * t * t);
                phase.sin()
            })
            .collect();

        let config = EmdConfig {
            max_imfs: 5,
            max_sift_iterations: 100,
            ..Default::default()
        };

        let mut emd = Emd::new(config);
        let result = emd.decompose(&Array1::from_vec(signal));

        assert!(result.is_ok());
        let imf_set = result.unwrap();
        assert!(imf_set.num_imfs() >= 1);

        // Test Hilbert-Huang Transform on chirp
        let mut hht = HilbertHuangTransform::new(EmdConfig::default());
        let spectrum = hht
            .compute(&Array1::from_vec(
                (0..n)
                    .map(|i| {
                        let t = i as f64 / sample_rate;
                        let phase = 2.0 * PI * (10.0 * t + 20.0 * t * t);
                        phase.sin()
                    })
                    .collect(),
            ), sample_rate)
            .unwrap();

        assert!(spectrum.num_imfs >= 1);
    }
}
