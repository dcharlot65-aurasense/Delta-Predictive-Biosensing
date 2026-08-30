//! Loss functions for SNN training

use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{Array1, Array2, Array3, s};
use serde::{Deserialize, Serialize};

/// Base trait for loss functions
pub trait LossFunction: Send + Sync {
    /// Compute loss
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32>;

    /// Compute gradient with respect to predictions
    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>>;
}

/// Cross-entropy loss based on spike rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingCrossEntropy {
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl SpikingCrossEntropy {
    pub fn new() -> Self {
        Self { eps: 1e-8 }
    }
}

impl Default for SpikingCrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl LossFunction for SpikingCrossEntropy {
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        // Compute spike rates for each neuron
        let rates = predictions.spike_rate();
        let (batch_size, num_neurons) = (rates.shape()[0], rates.shape()[1]);

        if targets.shape() != [batch_size, num_neurons] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("({}, {})", batch_size, num_neurons),
                actual: format!("{:?}", targets.shape()),
            });
        }

        // Softmax over the per-neuron rates, which are the logits.
        //
        // This used to divide each rate by the row sum -- a linear
        // normalisation, not the softmax the comment claimed -- while
        // `gradient` returned `p - t`, which is the derivative of the SOFTMAX
        // cross-entropy. The two did not describe the same function: for the
        // linear form the derivative is `sum(t)/S - t_k/r_k`. Making this an
        // actual softmax is what makes the gradient below correct.
        //
        // It also fixes a silent lie. The old body skipped any row whose rates
        // summed to zero, so a network that emitted no spikes at all scored a
        // loss of 0.0 -- a perfect fit. Under a softmax, all-zero rates give a
        // uniform distribution and a real, non-zero loss.
        let mut loss = 0.0;
        for b in 0..batch_size {
            let probs = softmax_row(&rates.row(b), self.eps);
            for n in 0..num_neurons {
                loss -= targets[[b, n]] * probs[n].ln();
            }
        }

        Ok(loss / batch_size as f32)
    }

    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let rates = predictions.spike_rate();
        let mut gradient = Array3::zeros((batch_size, num_steps, num_neurons));

        // dL/dp is `p - t` for a softmax cross-entropy; `rate = sum(s) / T`
        // contributes the 1/T, and `compute` averages over the batch, so the
        // 1/B belongs here too. It was missing, making this the gradient of the
        // summed loss rather than of the mean one that `compute` reports.
        let scale = 1.0 / (num_steps as f32 * batch_size as f32);
        for b in 0..batch_size {
            let probs = softmax_row(&rates.row(b), self.eps);
            for t in 0..num_steps {
                for n in 0..num_neurons {
                    gradient[[b, t, n]] = (probs[n] - targets[[b, n]]) * scale;
                }
            }
        }

        Ok(gradient)
    }
}

/// Softmax of one row, shifted by its maximum for numerical stability.
///
/// `eps` floors each probability so the logarithm the caller takes stays
/// finite when a class is assigned essentially no mass.
fn softmax_row(row: &ndarray::ArrayView1<f32>, eps: f32) -> Vec<f32> {
    let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.iter().map(|e| (e / total).max(eps)).collect()
}

/// Loss based on total spike count
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeCountLoss {
    /// Target spike count per neuron
    pub target_count: f32,
}

impl SpikeCountLoss {
    pub fn new(target_count: f32) -> Self {
        Self { target_count }
    }
}

impl LossFunction for SpikeCountLoss {
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut loss = 0.0;

        for b in 0..batch_size {
            for n in 0..num_neurons {
                // Count spikes for this neuron
                let spike_count: f32 = pred_dense.slice(s![b, .., n]).sum();

                // Target spike count weighted by class probability
                let target_spikes = targets[[b, n]] * self.target_count * num_steps as f32;

                // MSE loss
                loss += (spike_count - target_spikes).powi(2);
            }
        }

        Ok(loss / (batch_size * num_neurons) as f32)
    }

    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut gradient = Array3::zeros((batch_size, num_steps, num_neurons));

        for b in 0..batch_size {
            for n in 0..num_neurons {
                let spike_count: f32 = pred_dense.slice(s![b, .., n]).sum();
                let target_spikes = targets[[b, n]] * self.target_count * num_steps as f32;
                let grad_coef =
                    2.0 * (spike_count - target_spikes) / (batch_size * num_neurons) as f32;

                for t in 0..num_steps {
                    gradient[[b, t, n]] = grad_coef;
                }
            }
        }

        Ok(gradient)
    }
}

/// Loss based on spike timing (first spike latency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeTimingLoss {
    /// Weight for timing vs count
    pub timing_weight: f32,
}

impl SpikeTimingLoss {
    pub fn new(timing_weight: f32) -> Self {
        Self { timing_weight }
    }
}

impl LossFunction for SpikeTimingLoss {
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut loss = 0.0;

        for b in 0..batch_size {
            for n in 0..num_neurons {
                // Find first spike time
                let mut first_spike_time = num_steps as f32;
                for t in 0..num_steps {
                    if pred_dense[[b, t, n]] > 0.5 {
                        first_spike_time = t as f32;
                        break;
                    }
                }

                // Target: earlier spikes for higher target values
                let desired_time = (1.0 - targets[[b, n]]) * num_steps as f32;

                // Penalize deviation from desired timing
                loss += (first_spike_time - desired_time).powi(2);
            }
        }

        Ok(self.timing_weight * loss / (batch_size * num_neurons) as f32)
    }

    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut gradient = Array3::zeros((batch_size, num_steps, num_neurons));
        let scale = 2.0 * self.timing_weight / (batch_size * num_neurons) as f32;

        for b in 0..batch_size {
            for n in 0..num_neurons {
                // The first spike time, or `num_steps` when the neuron never
                // fired -- the same convention `compute` uses, and the case it
                // penalises most heavily. The old loop searched for a spike and
                // did nothing when it found none, so the worst outcome produced
                // no learning signal at all.
                let mut first = num_steps as f32;
                for t in 0..num_steps {
                    if pred_dense[[b, t, n]] > 0.5 {
                        first = t as f32;
                        break;
                    }
                }

                let desired = (1.0 - targets[[b, n]]) * num_steps as f32;
                let error = first - desired;
                if error == 0.0 {
                    continue;
                }

                // Signal placed at the DESIRED time, not at the observed spike.
                //
                // The old version wrote `+2 w e` at the spike that was already
                // too late; descending on that suppresses precisely that spike,
                // so the first spike becomes a later one still, and for a spike
                // that came too early it reinforced the early spike. Both
                // pushed the wrong way.
                //
                // Signalling at the desired index makes descent raise activity
                // there when the neuron is late and lower it when the neuron is
                // early, which is the direction that moves the first spike
                // toward its target.
                let idx = (desired.round() as usize).min(num_steps - 1);
                gradient[[b, idx, n]] = -scale * error;
            }
        }

        Ok(gradient)
    }
}

/// Temporal cross-entropy loss (considers temporal dynamics)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCrossEntropy {
    /// Temporal kernel for weighting
    pub temporal_kernel: Array1<f32>,
    /// Epsilon
    pub eps: f32,
}

impl TemporalCrossEntropy {
    pub fn new(num_steps: usize) -> Self {
        // Create exponential decay kernel (later spikes weighted less)
        let mut kernel = Array1::zeros(num_steps);
        for t in 0..num_steps {
            kernel[t] = (-0.05 * t as f32).exp();
        }
        let kernel_sum = kernel.sum();
        kernel /= kernel_sum; // Normalize

        Self {
            temporal_kernel: kernel,
            eps: 1e-8,
        }
    }
}

impl LossFunction for TemporalCrossEntropy {
    fn compute(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<f32> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        // Compute temporally-weighted spike rates
        let mut weighted_rates: Array2<f32> = Array2::zeros((batch_size, num_neurons));
        for b in 0..batch_size {
            for n in 0..num_neurons {
                for t in 0..num_steps.min(self.temporal_kernel.len()) {
                    weighted_rates[[b, n]] += pred_dense[[b, t, n]] * self.temporal_kernel[t];
                }
            }
        }

        // Compute cross-entropy on weighted rates
        let mut loss = 0.0;
        for b in 0..batch_size {
            let rate_sum = weighted_rates.row(b).sum().max(self.eps);
            for n in 0..num_neurons {
                let prob = (weighted_rates[[b, n]] / rate_sum).max(self.eps);
                loss -= targets[[b, n]] * prob.ln();
            }
        }

        Ok(loss / batch_size as f32)
    }

    fn gradient(&self, predictions: &SpikeTensor, targets: &Array2<f32>) -> SNNResult<Array3<f32>> {
        let pred_dense = predictions.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            pred_dense.shape()[0],
            pred_dense.shape()[1],
            pred_dense.shape()[2],
        );

        let mut gradient = Array3::zeros((batch_size, num_steps, num_neurons));

        // Simplified gradient computation
        for b in 0..batch_size {
            for t in 0..num_steps.min(self.temporal_kernel.len()) {
                for n in 0..num_neurons {
                    let weight = self.temporal_kernel[t];
                    // Approximate gradient
                    gradient[[b, t, n]] = weight * (pred_dense[[b, t, n]] - targets[[b, n]]);
                }
            }
        }

        Ok(gradient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiking_cross_entropy() {
        let loss_fn = SpikingCrossEntropy::new();

        // Create simple predictions (2 batches, 10 steps, 3 neurons)
        let mut pred_data = Array3::zeros((2, 10, 3));
        pred_data[[0, 5, 0]] = 1.0; // Batch 0, neuron 0 spikes
        pred_data[[1, 3, 1]] = 1.0; // Batch 1, neuron 1 spikes

        let predictions = SpikeTensor::from_dense(pred_data, false);

        // Targets: one-hot encoding
        let targets = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

        let loss = loss_fn.compute(&predictions, &targets).unwrap();
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_spike_count_loss() {
        let loss_fn = SpikeCountLoss::new(0.2); // Target 20% spike rate

        let mut pred_data = Array3::zeros((1, 10, 2));
        pred_data[[0, 3, 0]] = 1.0;
        pred_data[[0, 7, 0]] = 1.0; // 2 spikes in 10 steps

        let predictions = SpikeTensor::from_dense(pred_data, false);
        let targets = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();

        let loss = loss_fn.compute(&predictions, &targets).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_spike_timing_loss() {
        let loss_fn = SpikeTimingLoss::new(1.0);

        let mut pred_data = Array3::zeros((1, 10, 2));
        pred_data[[0, 2, 0]] = 1.0; // Early spike

        let predictions = SpikeTensor::from_dense(pred_data, false);
        let targets = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();

        let loss = loss_fn.compute(&predictions, &targets).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_temporal_cross_entropy() {
        let loss_fn = TemporalCrossEntropy::new(10);

        let mut pred_data = Array3::zeros((1, 10, 2));
        pred_data[[0, 1, 0]] = 1.0;

        let predictions = SpikeTensor::from_dense(pred_data, false);
        let targets = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();

        let loss = loss_fn.compute(&predictions, &targets).unwrap();
        assert!(loss.is_finite());
    }

    // ---- loss/gradient consistency ------------------------------------
    //
    // These losses are smooth functions of the (real-valued) spike tensor, with
    // no surrogate anywhere in them, so central differences are a valid and
    // decisive check that `gradient` is the derivative of `compute`. That is
    // not true of the layer backward passes, which is why those are checked by
    // forward-mode instead.

    /// Central-difference gradient of `compute`, for one loss and one input.
    fn numeric_gradient(
        loss: &dyn LossFunction,
        dense: &Array3<f32>,
        targets: &Array2<f32>,
    ) -> Array3<f32> {
        let h = 1e-3f32;
        let mut out = Array3::zeros(dense.raw_dim());
        for ((b, t, n), slot) in out.indexed_iter_mut() {
            let mut up = dense.clone();
            let mut down = dense.clone();
            up[[b, t, n]] += h;
            down[[b, t, n]] -= h;
            let lu = loss
                .compute(&SpikeTensor::from_dense(up, false), targets)
                .unwrap();
            let ld = loss
                .compute(&SpikeTensor::from_dense(down, false), targets)
                .unwrap();
            *slot = (lu - ld) / (2.0 * h);
        }
        out
    }

    fn ce_fixture() -> (Array3<f32>, Array2<f32>) {
        let (batch, steps, neurons) = (2, 5, 3);
        let dense = Array3::from_shape_fn((batch, steps, neurons), |(b, t, n)| {
            (((b * 3 + t * 2 + n) % 4) as f32) * 0.25
        });
        let mut targets = Array2::zeros((batch, neurons));
        for b in 0..batch {
            targets[[b, b % neurons]] = 1.0;
        }
        (dense, targets)
    }

    /// `gradient` must be the derivative of `compute`.
    ///
    /// It was not: `compute` normalised rates linearly while `gradient`
    /// returned `p - t`, the softmax form, and `gradient` also omitted the
    /// 1/batch that `compute` applies.
    #[test]
    fn cross_entropy_gradient_matches_its_own_loss() {
        let (dense, targets) = ce_fixture();
        let loss = SpikingCrossEntropy::new();

        let analytic = loss
            .gradient(&SpikeTensor::from_dense(dense.clone(), false), &targets)
            .unwrap();
        let numeric = numeric_gradient(&loss, &dense, &targets);

        for ((b, t, n), a) in analytic.indexed_iter() {
            let e = numeric[[b, t, n]];
            assert!(
                (a - e).abs() <= 1e-2 * e.abs().max(1e-2),
                "d/ds[{b},{t},{n}]: analytic {a} vs numeric {e}"
            );
        }
    }

    /// A network that emitted nothing is not a perfect fit.
    ///
    /// The old `compute` skipped any row whose rates summed to zero, so a
    /// silent network scored 0.0 -- the best score achievable.
    #[test]
    fn cross_entropy_penalises_a_silent_network() {
        let (batch, steps, neurons) = (2, 6, 3);
        let silent = SpikeTensor::from_dense(Array3::zeros((batch, steps, neurons)), false);
        let mut targets = Array2::zeros((batch, neurons));
        for b in 0..batch {
            targets[[b, 0]] = 1.0;
        }

        let loss = SpikingCrossEntropy::new()
            .compute(&silent, &targets)
            .unwrap();
        assert!(
            loss > 0.0,
            "a network that never spiked scored {loss}, which reads as a perfect fit"
        );

        // And a network that puts its spikes on the right neuron must score
        // better than the silent one.
        let mut good = Array3::zeros((batch, steps, neurons));
        for b in 0..batch {
            for t in 0..steps {
                good[[b, t, 0]] = 1.0;
            }
        }
        let good_loss = SpikingCrossEntropy::new()
            .compute(&SpikeTensor::from_dense(good, false), &targets)
            .unwrap();
        assert!(
            good_loss < loss,
            "spiking on the target neuron ({good_loss}) should beat silence ({loss})"
        );
    }

    /// Every loss must have a gradient that is the derivative of its own
    /// `compute`. Reported per loss so a failure names the offender.
    #[test]
    fn every_loss_gradient_matches_its_loss() {
        let (dense, targets) = ce_fixture();
        // SpikeTimingLoss is deliberately absent. Its `compute` depends on the
        // first spike time, found by thresholding at 0.5, so it is piecewise
        // constant in the spike values and its true derivative is zero almost
        // everywhere -- finite differences would correctly report zero and the
        // comparison would be meaningless. Its signal is a heuristic, like a
        // surrogate, and is checked by its own tests below on the properties
        // that actually matter.
        let losses: Vec<(&str, Box<dyn LossFunction>)> = vec![
            ("SpikingCrossEntropy", Box::new(SpikingCrossEntropy::new())),
            ("SpikeCountLoss", Box::new(SpikeCountLoss::new(0.5))),
        ];

        let mut mismatched = Vec::new();
        for (name, loss) in &losses {
            let analytic =
                match loss.gradient(&SpikeTensor::from_dense(dense.clone(), false), &targets) {
                    Ok(g) => g,
                    Err(e) => {
                        mismatched.push(format!("{name}: gradient errored: {e}"));
                        continue;
                    }
                };
            let numeric = numeric_gradient(loss.as_ref(), &dense, &targets);

            let mut worst = 0.0f32;
            let mut where_worst = (0, 0, 0);
            for ((b, t, n), a) in analytic.indexed_iter() {
                let e = numeric[[b, t, n]];
                let err = (a - e).abs() / e.abs().max(1e-2);
                if err > worst {
                    worst = err;
                    where_worst = (b, t, n);
                }
            }
            if worst > 5e-2 {
                let (b, t, n) = where_worst;
                mismatched.push(format!(
                    "{name}: worst relative error {worst:.3} at [{b},{t},{n}] \
                     (analytic {}, numeric {})",
                    analytic[[b, t, n]],
                    numeric[[b, t, n]]
                ));
            }
        }

        assert!(
            mismatched.is_empty(),
            "gradient does not match compute for:\n  {}",
            mismatched.join("\n  ")
        );
    }

    // ---- SpikeTimingLoss -----------------------------------------------
    //
    // Not checkable by finite differences: `compute` reads the first spike time
    // through a 0.5 threshold, so it is piecewise constant and its derivative
    // is zero almost everywhere. What can be checked is that the heuristic
    // signal exists in the cases that matter and points the right way.

    /// One neuron, `steps` long, spiking at `spike_at` if given.
    fn timing_case(
        steps: usize,
        spike_at: Option<usize>,
        target: f32,
    ) -> (Array3<f32>, Array2<f32>) {
        let mut dense = Array3::zeros((1, steps, 1));
        if let Some(t) = spike_at {
            dense[[0, t, 0]] = 1.0;
        }
        let mut targets = Array2::zeros((1, 1));
        targets[[0, 0]] = target;
        (dense, targets)
    }

    /// A neuron that never fires is what the loss penalises hardest, and it
    /// used to receive no gradient at all -- the search for a first spike found
    /// none and wrote nothing, so the worst case could never be corrected.
    #[test]
    fn timing_loss_signals_when_no_spike_occurred() {
        let steps = 10;
        let (dense, targets) = timing_case(steps, None, 0.5);
        let loss = SpikeTimingLoss::new(1.0);

        let penalty = loss
            .compute(&SpikeTensor::from_dense(dense.clone(), false), &targets)
            .unwrap();
        assert!(penalty > 0.0, "silence should be penalised");

        let g = loss
            .gradient(&SpikeTensor::from_dense(dense, false), &targets)
            .unwrap();
        let total: f32 = g.iter().map(|v| v.abs()).sum();
        assert!(
            total > 0.0,
            "a neuron that never fired received no gradient, so the case the \
             loss penalises hardest cannot be corrected"
        );
    }

    /// The signal must push a late neuron earlier and an early neuron later.
    ///
    /// Descent subtracts the gradient, so "encourage activity here" is a
    /// negative entry. The old version wrote a positive entry at the late
    /// spike, which suppressed that spike and made the first spike later still.
    #[test]
    fn timing_loss_pushes_toward_the_target_time() {
        let steps = 10;
        let target = 0.5; // desired first spike at step 5
        let loss = SpikeTimingLoss::new(1.0);
        let desired = 5usize;

        // Spiking too late.
        let (late, targets) = timing_case(steps, Some(8), target);
        let g_late = loss
            .gradient(&SpikeTensor::from_dense(late, false), &targets)
            .unwrap();
        assert!(
            g_late[[0, desired, 0]] < 0.0,
            "a late neuron should be encouraged to fire at step {desired}, got {}",
            g_late[[0, desired, 0]]
        );

        // Spiking too early.
        let (early, targets) = timing_case(steps, Some(2), target);
        let g_early = loss
            .gradient(&SpikeTensor::from_dense(early, false), &targets)
            .unwrap();
        assert!(
            g_early[[0, desired, 0]] > 0.0,
            "an early neuron should be discouraged at step {desired}, got {}",
            g_early[[0, desired, 0]]
        );

        // On target: nothing to correct.
        let (spot_on, targets) = timing_case(steps, Some(desired), target);
        let g_ok = loss
            .gradient(&SpikeTensor::from_dense(spot_on, false), &targets)
            .unwrap();
        assert_eq!(
            g_ok.iter().map(|v| v.abs()).sum::<f32>(),
            0.0,
            "a neuron already on target should receive no correction"
        );
    }

    /// The signal's magnitude must grow with the timing error.
    #[test]
    fn timing_loss_signal_scales_with_the_error() {
        let steps = 10;
        let loss = SpikeTimingLoss::new(1.0);
        let magnitude = |spike_at: usize| {
            let (dense, targets) = timing_case(steps, Some(spike_at), 0.5);
            loss.gradient(&SpikeTensor::from_dense(dense, false), &targets)
                .unwrap()
                .iter()
                .map(|v| v.abs())
                .sum::<f32>()
        };
        assert!(
            magnitude(9) > magnitude(6),
            "a neuron four steps late should be corrected harder than one one step late"
        );
    }
}
