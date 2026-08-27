//! Surrogate gradient functions for backpropagation through spiking neurons.
//!
//! Since the Heaviside step function (spike generation) has zero gradient everywhere
//! except at the threshold (where it's undefined), we use smooth surrogate functions
//! for the backward pass during training.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Trait for surrogate gradient functions.
pub trait SurrogateGradient: Send + Sync {
    /// Forward pass - typically Heaviside step function.
    fn forward(&self, x: f32) -> f32 {
        if x >= 0.0 { 1.0 } else { 0.0 }
    }

    /// Backward pass - smooth gradient approximation.
    fn backward(&self, x: f32) -> f32;

    /// Get the name of this surrogate function.
    fn name(&self) -> &'static str;
}

/// Fast sigmoid surrogate gradient.
///
/// Formula: g(x) = 1 / (slope * |x| + 1)^2
///
/// This is computationally efficient and works well in practice.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FastSigmoid {
    /// Slope parameter (higher = steeper, closer to Heaviside)
    pub slope: f32,
}

impl Default for FastSigmoid {
    fn default() -> Self {
        Self { slope: 25.0 }
    }
}

impl FastSigmoid {
    pub fn new(slope: f32) -> Self {
        Self { slope }
    }
}

impl SurrogateGradient for FastSigmoid {
    fn backward(&self, x: f32) -> f32 {
        let denom = self.slope * x.abs() + 1.0;
        1.0 / (denom * denom)
    }

    fn name(&self) -> &'static str {
        "FastSigmoid"
    }
}

/// Arctangent surrogate gradient.
///
/// Formula: g(x) = 1 / (π * slope) * 1 / (1 + (slope * x)^2)
///
/// Smooth and differentiable everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arctan {
    /// Slope/temperature parameter
    pub slope: f32,
}

impl Default for Arctan {
    fn default() -> Self {
        Self { slope: 20.0 }
    }
}

impl Arctan {
    pub fn new(slope: f32) -> Self {
        Self { slope }
    }
}

impl SurrogateGradient for Arctan {
    fn backward(&self, x: f32) -> f32 {
        let sx = self.slope * x;
        1.0 / (PI * self.slope * (1.0 + sx * sx))
    }

    fn name(&self) -> &'static str {
        "Arctan"
    }
}

/// Triangular surrogate gradient.
///
/// Formula: g(x) = max(0, 1 - |x|) / width
///
/// Piecewise linear, zero outside [-width, width].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Triangular {
    /// Width of the triangle
    pub width: f32,
}

impl Default for Triangular {
    fn default() -> Self {
        Self { width: 1.0 }
    }
}

impl Triangular {
    pub fn new(width: f32) -> Self {
        Self { width }
    }
}

impl SurrogateGradient for Triangular {
    fn backward(&self, x: f32) -> f32 {
        let normalized = x.abs() / self.width;
        if normalized < 1.0 {
            (1.0 - normalized) / self.width
        } else {
            0.0
        }
    }

    fn name(&self) -> &'static str {
        "Triangular"
    }
}

/// SuperSpike surrogate gradient.
///
/// Formula: g(x) = 1 / (1 + |β*x|)^2
///
/// Proposed in "SuperSpike: Supervised learning in multi-layer spiking neural networks"
/// (Zenke & Ganguli, 2018).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SuperSpike {
    /// Beta parameter (controls steepness)
    pub beta: f32,
}

impl Default for SuperSpike {
    fn default() -> Self {
        Self { beta: 1.0 }
    }
}

impl SuperSpike {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}

impl SurrogateGradient for SuperSpike {
    fn backward(&self, x: f32) -> f32 {
        let denom = 1.0 + (self.beta * x).abs();
        1.0 / (denom * denom)
    }

    fn name(&self) -> &'static str {
        "SuperSpike"
    }
}

/// Multi-Gaussian surrogate gradient.
///
/// Formula: g(x) = 1/√(2π*σ²) * exp(-x²/(2σ²))
///
/// Gaussian derivative, smooth and biologically plausible.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MultiGaussian {
    /// Standard deviation (width of Gaussian)
    pub sigma: f32,
}

impl Default for MultiGaussian {
    fn default() -> Self {
        Self { sigma: 0.5 }
    }
}

impl MultiGaussian {
    pub fn new(sigma: f32) -> Self {
        Self { sigma }
    }
}

impl SurrogateGradient for MultiGaussian {
    fn backward(&self, x: f32) -> f32 {
        let variance = self.sigma * self.sigma;
        let normalization = 1.0 / (2.0 * PI * variance).sqrt();
        normalization * (-x * x / (2.0 * variance)).exp()
    }

    fn name(&self) -> &'static str {
        "MultiGaussian"
    }
}

/// Straight-Through Estimator (STE).
///
/// Formula: g(x) = 1 for |x| < threshold, 0 otherwise
///
/// Simple box function, often used as baseline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StraightThroughEstimator {
    /// Threshold beyond which gradient is zero
    pub threshold: f32,
}

impl Default for StraightThroughEstimator {
    fn default() -> Self {
        Self { threshold: 1.0 }
    }
}

impl StraightThroughEstimator {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl SurrogateGradient for StraightThroughEstimator {
    fn backward(&self, x: f32) -> f32 {
        if x.abs() < self.threshold { 1.0 } else { 0.0 }
    }

    fn name(&self) -> &'static str {
        "StraightThroughEstimator"
    }
}

/// Enum wrapper for all surrogate gradient types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SurrogateFunction {
    FastSigmoid(FastSigmoid),
    Arctan(Arctan),
    Triangular(Triangular),
    SuperSpike(SuperSpike),
    MultiGaussian(MultiGaussian),
    StraightThroughEstimator(StraightThroughEstimator),
}

impl Default for SurrogateFunction {
    fn default() -> Self {
        SurrogateFunction::FastSigmoid(FastSigmoid::default())
    }
}

impl SurrogateGradient for SurrogateFunction {
    fn forward(&self, x: f32) -> f32 {
        match self {
            SurrogateFunction::FastSigmoid(s) => s.forward(x),
            SurrogateFunction::Arctan(s) => s.forward(x),
            SurrogateFunction::Triangular(s) => s.forward(x),
            SurrogateFunction::SuperSpike(s) => s.forward(x),
            SurrogateFunction::MultiGaussian(s) => s.forward(x),
            SurrogateFunction::StraightThroughEstimator(s) => s.forward(x),
        }
    }

    fn backward(&self, x: f32) -> f32 {
        match self {
            SurrogateFunction::FastSigmoid(s) => s.backward(x),
            SurrogateFunction::Arctan(s) => s.backward(x),
            SurrogateFunction::Triangular(s) => s.backward(x),
            SurrogateFunction::SuperSpike(s) => s.backward(x),
            SurrogateFunction::MultiGaussian(s) => s.backward(x),
            SurrogateFunction::StraightThroughEstimator(s) => s.backward(x),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            SurrogateFunction::FastSigmoid(s) => s.name(),
            SurrogateFunction::Arctan(s) => s.name(),
            SurrogateFunction::Triangular(s) => s.name(),
            SurrogateFunction::SuperSpike(s) => s.name(),
            SurrogateFunction::MultiGaussian(s) => s.name(),
            SurrogateFunction::StraightThroughEstimator(s) => s.name(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_sigmoid() {
        let surrogate = FastSigmoid::new(25.0);

        // Forward pass - Heaviside
        assert_eq!(surrogate.forward(1.0), 1.0);
        assert_eq!(surrogate.forward(-1.0), 0.0);
        assert_eq!(surrogate.forward(0.0), 1.0); // At threshold

        // Backward pass - smooth gradient
        let grad_at_zero = surrogate.backward(0.0);
        assert!(grad_at_zero > 0.9); // Should be close to 1.0

        let grad_away = surrogate.backward(1.0);
        assert!(grad_away < grad_at_zero); // Should decrease away from threshold
    }

    #[test]
    fn test_arctan() {
        let surrogate = Arctan::new(20.0);

        let grad_at_zero = surrogate.backward(0.0);
        let grad_away = surrogate.backward(1.0);

        assert!(grad_at_zero > grad_away);
        assert!(grad_at_zero > 0.0);
    }

    #[test]
    fn test_triangular() {
        let surrogate = Triangular::new(1.0);

        // At zero, gradient should be maximum
        assert_eq!(surrogate.backward(0.0), 1.0);

        // At width boundary, gradient should be zero
        assert_eq!(surrogate.backward(1.0), 0.0);
        assert_eq!(surrogate.backward(-1.0), 0.0);

        // Beyond width, gradient is zero
        assert_eq!(surrogate.backward(2.0), 0.0);

        // Within width, decreases linearly
        let grad_half = surrogate.backward(0.5);
        assert!(grad_half > 0.0 && grad_half < 1.0);
    }

    #[test]
    fn test_superspike() {
        let surrogate = SuperSpike::new(1.0);

        let grad_at_zero = surrogate.backward(0.0);
        assert_eq!(grad_at_zero, 1.0);

        let grad_away = surrogate.backward(1.0);
        assert!(grad_away < grad_at_zero);
    }

    #[test]
    fn test_multi_gaussian() {
        let surrogate = MultiGaussian::new(0.5);

        let grad_at_zero = surrogate.backward(0.0);
        assert!(grad_at_zero > 0.0);

        // Gaussian should be symmetric
        let grad_pos = surrogate.backward(0.5);
        let grad_neg = surrogate.backward(-0.5);
        assert!((grad_pos - grad_neg).abs() < 1e-6);

        // Should decrease away from center
        assert!(grad_at_zero > grad_pos);
    }

    #[test]
    fn test_straight_through_estimator() {
        let surrogate = StraightThroughEstimator::new(1.0);

        // Within threshold
        assert_eq!(surrogate.backward(0.0), 1.0);
        assert_eq!(surrogate.backward(0.5), 1.0);
        assert_eq!(surrogate.backward(-0.5), 1.0);

        // At boundary
        assert_eq!(surrogate.backward(1.0), 0.0);
        assert_eq!(surrogate.backward(-1.0), 0.0);

        // Beyond threshold
        assert_eq!(surrogate.backward(2.0), 0.0);
    }

    #[test]
    fn test_surrogate_function_enum() {
        let surrogate = SurrogateFunction::default();
        assert_eq!(surrogate.name(), "FastSigmoid");

        // Test polymorphic dispatch
        let surrogates = vec![
            SurrogateFunction::FastSigmoid(FastSigmoid::default()),
            SurrogateFunction::Arctan(Arctan::default()),
            SurrogateFunction::SuperSpike(SuperSpike::default()),
        ];

        for s in surrogates {
            assert_eq!(s.forward(1.0), 1.0);
            assert_eq!(s.forward(-1.0), 0.0);
            assert!(s.backward(0.0) > 0.0);
        }
    }
}
