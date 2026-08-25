//! # Model Calibration
//!
//! Methods for calibrating neural network confidence scores.
//!
//! ## Overview
//!
//! Model calibration ensures that predicted probabilities accurately reflect
//! true likelihood of correctness. A well-calibrated model should have:
//! - Predicted 80% confidence → 80% actual accuracy
//! - Predicted 90% confidence → 90% actual accuracy
//!
//! This module provides three calibration methods and metrics for evaluation.
//!
//! ## Calibration Methods
//!
//! | Method | Use Case | Complexity |
//! |--------|----------|------------|
//! | [`TemperatureScaling`] | Simple, effective for most cases | O(1) parameters |
//! | [`PlattScaling`] | Binary classification | O(2) parameters |
//! | [`IsotonicCalibration`] | Non-parametric, more flexible | O(n) bins |
//!
//! ## Metrics
//!
//! - [`expected_calibration_error`]: Primary calibration metric (ECE)
//! - [`maximum_calibration_error`]: Worst-case calibration (MCE)
//! - [`brier_score`]: Probability accuracy
//! - [`negative_log_likelihood`]: Log-likelihood loss
//! - [`reliability_diagram`]: Visual calibration assessment
//!
//! ## Example: Temperature Scaling
//!
//! ```rust
//! use dpb_snn::calibration::*;
//!
//! # fn example() {
//! // Validation set logits and labels
//! let logits = vec![
//!     vec![2.0, 1.0, 0.5],  // Sample 1
//!     vec![0.1, 3.0, 0.2],  // Sample 2
//!     vec![1.5, 0.5, 2.5],  // Sample 3
//! ];
//! let labels = vec![0, 1, 2];
//!
//! // Fit temperature parameter
//! let mut calibrator = TemperatureScaling::new();
//! // `fit` reports its own error type, so it is not a `?` candidate here.
//! calibrator.fit(&logits, &labels).expect("fit");
//!
//! // Calibrate test predictions. `calibrate` takes ONE row of logits;
//! // `calibrate_batch` takes many.
//! let test_logits = vec![vec![1.8, 1.2, 0.3]];
//! let calibrated = calibrator.calibrate_batch(&test_logits);
//!
//! println!("Temperature: {:.3}", calibrator.temperature());
//! # }
//! ```
//!
//! ## Example: Isotonic Calibration
//!
//! ```rust
//! use dpb_snn::calibration::*;
//!
//! # fn example() {
//! // For binary classification. Labels are `f64` so that soft targets work too.
//! let probs = vec![0.1, 0.3, 0.5, 0.7, 0.9];
//! let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0];
//!
//! let mut calibrator = IsotonicCalibration::new();
//! calibrator.fit(&probs, &labels).expect("fit");
//!
//! let test_probs = vec![0.4, 0.6, 0.8];
//! let calibrated = calibrator.calibrate_batch(&test_probs);
//! # }
//! ```
//!
//! ## Example: Evaluating Calibration
//!
//! ```rust
//! use dpb_snn::calibration::*;
//!
//! # fn example() {
//! // These metrics are BINARY: a confidence per sample, and whether that
//! // sample was correct. For a multiclass model, take the probability assigned
//! // to the predicted class and whether it matched the label.
//! let confidences = vec![0.7, 0.8, 0.5];
//! let correct = vec![true, true, false];
//!
//! // Compute calibration metrics
//! let ece = expected_calibration_error(&confidences, &correct, 10);
//! let mce = maximum_calibration_error(&confidences, &correct, 10);
//! let brier = brier_score(&confidences, &correct);
//!
//! println!("Expected Calibration Error: {:.4}", ece);
//! println!("Maximum Calibration Error: {:.4}", mce);
//! println!("Brier Score: {:.4}", brier);
//! # }
//! ```
//!
//! ## Example: Uncertainty Quantification
//!
//! ```rust
//! use dpb_snn::calibration::uncertainty::*;
//!
//! # fn example() {
//! // Monte Carlo Dropout for uncertainty estimation.
//! // Note the argument order: samples first, then dropout rate.
//! let mut mc_dropout = MCDropout::new(100, 0.2);  // 100 samples, 20% dropout
//!
//! // Ensemble uncertainty
//! let predictions = vec![
//!     vec![0.7, 0.2, 0.1],  // Model 1
//!     vec![0.6, 0.3, 0.1],  // Model 2
//!     vec![0.8, 0.1, 0.1],  // Model 3
//! ];
//!
//! // The estimator is constructed with the model COUNT; the predictions are
//! // passed to each call.
//! let ensemble = EnsembleUncertainty::new(predictions.len());
//! let (mean, variance) = ensemble.compute_statistics(&predictions);
//! let entropy = ensemble.entropy(&predictions);
//!
//! println!("Prediction: {:?}", mean);
//! println!("Per-class variance: {:?}", variance);
//! println!("Uncertainty (entropy): {:?}", entropy);
//! # }
//! ```
//!
//! ## Reliability Diagram
//!
//! Visualize calibration with reliability diagrams:
//!
//! ```rust
//! use dpb_snn::calibration::*;
//!
//! # fn example() {
//! # let confidences = vec![0.7, 0.6, 0.9, 0.55];
//! # let correct = vec![true, false, true, true];
//! // Generate reliability diagram data
//! let bins = reliability_diagram(&confidences, &correct, 10);
//!
//! for bin in bins {
//!     println!("Confidence: {:.2}, Accuracy: {:.2}, Count: {}",
//!         bin.predicted_prob,
//!         bin.actual_freq,
//!         bin.count
//!     );
//! }
//! # }
//! ```
//!
//! ## When to Use Calibration
//!
//! - **Critical Applications**: Medical diagnosis, safety-critical systems
//! - **Confidence Thresholding**: When filtering predictions by confidence
//! - **Decision Making**: When probabilities guide downstream decisions
//! - **Model Comparison**: Fair comparison of different architectures

pub mod temperature;
pub mod isotonic;
pub mod uncertainty;
pub mod metrics;

pub use temperature::{TemperatureScaling, PlattScaling};
pub use isotonic::IsotonicCalibration;
pub use uncertainty::{
    UncertaintyEstimator, MCDropout, EnsembleUncertainty, ConfidenceInterval, bootstrap_ci,
};
pub use metrics::{
    expected_calibration_error, maximum_calibration_error, reliability_diagram, brier_score,
    negative_log_likelihood, ReliabilityBin,
};
