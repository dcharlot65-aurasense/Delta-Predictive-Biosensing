//! Surrogate gradient methods for training SNNs

use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{Array3, Axis};
use serde::{Deserialize, Serialize};

/// Types of surrogate gradient functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurrogateType {
    /// Rectangular/box surrogate
    Box,
    /// Triangular surrogate
    Triangle,
    /// Fast sigmoid surrogate
    FastSigmoid,
    /// Exponential surrogate
    Exponential,
    /// SuperSpike surrogate
    SuperSpike,
}

/// Surrogate gradient function trait
pub trait SurrogateGradient: Send + Sync {
    /// Compute surrogate gradient for membrane potential
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32;

    /// Apply surrogate gradient to spike tensor
    fn apply(&self, spikes: &SpikeTensor, v_mem: &Array3<f32>, threshold: f32) -> SNNResult<Array3<f32>>;
}

/// Box surrogate gradient (rectangular window)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxSurrogate {
    /// Width of the box
    pub width: f32,
}

impl BoxSurrogate {
    pub fn new(width: f32) -> Self {
        Self { width }
    }
}

impl SurrogateGradient for BoxSurrogate {
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32 {
        let dist = (v_mem - threshold).abs();
        if dist < self.width / 2.0 {
            1.0 / self.width
        } else {
            0.0
        }
    }

    fn apply(&self, spikes: &SpikeTensor, v_mem: &Array3<f32>, threshold: f32) -> SNNResult<Array3<f32>> {
        let spike_dense = spikes.to_dense();
        let mut gradient = Array3::zeros(spike_dense.raw_dim());

        for ((b, t, n), &spike) in spike_dense.indexed_iter() {
            if spike > 0.5 {
                gradient[[b, t, n]] = self.compute_gradient(v_mem[[b, t, n]], threshold);
            }
        }

        Ok(gradient)
    }
}

/// Fast sigmoid surrogate gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastSigmoidSurrogate {
    /// Slope parameter
    pub beta: f32,
}

impl FastSigmoidSurrogate {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}

impl SurrogateGradient for FastSigmoidSurrogate {
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32 {
        let x = self.beta * (v_mem - threshold);
        self.beta / (1.0 + x.abs()).powi(2)
    }

    fn apply(&self, spikes: &SpikeTensor, v_mem: &Array3<f32>, threshold: f32) -> SNNResult<Array3<f32>> {
        let spike_dense = spikes.to_dense();
        let mut gradient = Array3::zeros(spike_dense.raw_dim());

        for ((b, t, n), _) in spike_dense.indexed_iter() {
            gradient[[b, t, n]] = self.compute_gradient(v_mem[[b, t, n]], threshold);
        }

        Ok(gradient)
    }
}

/// SuperSpike surrogate gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperSpikeSurrogate {
    /// Scale parameter
    pub beta: f32,
}

impl SuperSpikeSurrogate {
    pub fn new(beta: f32) -> Self {
        Self { beta }
    }
}

impl SurrogateGradient for SuperSpikeSurrogate {
    fn compute_gradient(&self, v_mem: f32, threshold: f32) -> f32 {
        let x = self.beta * (v_mem - threshold);
        1.0 / (1.0 + x.abs()).powi(2)
    }

    fn apply(&self, spikes: &SpikeTensor, v_mem: &Array3<f32>, threshold: f32) -> SNNResult<Array3<f32>> {
        let spike_dense = spikes.to_dense();
        let mut gradient = Array3::zeros(spike_dense.raw_dim());

        for ((b, t, n), _) in spike_dense.indexed_iter() {
            gradient[[b, t, n]] = self.compute_gradient(v_mem[[b, t, n]], threshold);
        }

        Ok(gradient)
    }
}

/// Backpropagation Through Time (BPTT) for SNNs
#[derive(Debug, Clone)]
pub struct BPTT {
    /// Surrogate gradient function
    pub surrogate: Box<dyn SurrogateGradient>,
    /// Number of time steps to backprop through
    pub num_steps: Option<usize>,
}

impl BPTT {
    pub fn new(surrogate_type: SurrogateType, num_steps: Option<usize>) -> Self {
        let surrogate: Box<dyn SurrogateGradient> = match surrogate_type {
            SurrogateType::Box => Box::new(BoxSurrogate::new(1.0)),
            SurrogateType::Triangle => Box::new(BoxSurrogate::new(1.0)), // Simplified
            SurrogateType::FastSigmoid => Box::new(FastSigmoidSurrogate::new(10.0)),
            SurrogateType::Exponential => Box::new(SuperSpikeSurrogate::new(10.0)), // Simplified
            SurrogateType::SuperSpike => Box::new(SuperSpikeSurrogate::new(10.0)),
        };

        Self { surrogate, num_steps }
    }

    /// Compute gradients through time
    pub fn backward(
        &self,
        output_grad: &Array3<f32>,
        v_mem_history: &[Array3<f32>],
        threshold: f32,
    ) -> SNNResult<Vec<Array3<f32>>> {
        let num_actual_steps = v_mem_history.len();
        let bptt_steps = self.num_steps.unwrap_or(num_actual_steps).min(num_actual_steps);

        let mut gradients = vec![Array3::zeros(output_grad.raw_dim()); num_actual_steps];

        // Backpropagate through time
        for t in (num_actual_steps - bptt_steps..num_actual_steps).rev() {
            // Compute surrogate gradient for this time step
            let v_mem = &v_mem_history[t];
            let mut grad_t = output_grad.clone();

            // Apply surrogate gradient
            for ((b, _, n), grad_val) in grad_t.indexed_iter_mut() {
                *grad_val *= self.surrogate.compute_gradient(v_mem[[b, t, n]], threshold);
            }

            gradients[t] = grad_t;
        }

        Ok(gradients)
    }
}

/// Online Training Through Time (OTTT)
/// More memory efficient than full BPTT
#[derive(Debug, Clone)]
pub struct OTTT {
    /// Surrogate gradient function
    pub surrogate: Box<dyn SurrogateGradient>,
    /// Eligibility trace decay
    pub trace_decay: f32,
}

impl OTTT {
    pub fn new(surrogate_type: SurrogateType, trace_decay: f32) -> Self {
        let surrogate: Box<dyn SurrogateGradient> = match surrogate_type {
            SurrogateType::Box => Box::new(BoxSurrogate::new(1.0)),
            SurrogateType::FastSigmoid => Box::new(FastSigmoidSurrogate::new(10.0)),
            SurrogateType::SuperSpike => Box::new(SuperSpikeSurrogate::new(10.0)),
            _ => Box::new(SuperSpikeSurrogate::new(10.0)),
        };

        Self { surrogate, trace_decay }
    }
}

/// Spatial Layer-wise Training Through Time (SLTT)
/// Train layers independently to reduce computational cost
#[derive(Debug, Clone)]
pub struct SLTT {
    /// Surrogate gradient function
    pub surrogate: Box<dyn SurrogateGradient>,
    /// Number of layers to train simultaneously
    pub layer_window: usize,
}

impl SLTT {
    pub fn new(surrogate_type: SurrogateType, layer_window: usize) -> Self {
        let surrogate: Box<dyn SurrogateGradient> = match surrogate_type {
            SurrogateType::Box => Box::new(BoxSurrogate::new(1.0)),
            SurrogateType::FastSigmoid => Box::new(FastSigmoidSurrogate::new(10.0)),
            SurrogateType::SuperSpike => Box::new(SuperSpikeSurrogate::new(10.0)),
            _ => Box::new(SuperSpikeSurrogate::new(10.0)),
        };

        Self { surrogate, layer_window }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_box_surrogate() {
        let surrogate = BoxSurrogate::new(1.0);
        let threshold = 1.0;

        // Within window
        let grad1 = surrogate.compute_gradient(1.2, threshold);
        assert!(grad1 > 0.0);

        // Outside window
        let grad2 = surrogate.compute_gradient(2.0, threshold);
        assert_eq!(grad2, 0.0);
    }

    #[test]
    fn test_fast_sigmoid_surrogate() {
        let surrogate = FastSigmoidSurrogate::new(10.0);
        let threshold = 1.0;

        let grad1 = surrogate.compute_gradient(1.0, threshold);
        assert!(grad1 > 0.0);

        let grad2 = surrogate.compute_gradient(1.1, threshold);
        assert!(grad2 > 0.0);
    }

    #[test]
    fn test_superspike_surrogate() {
        let surrogate = SuperSpikeSurrogate::new(10.0);
        let threshold = 1.0;

        let grad = surrogate.compute_gradient(1.1, threshold);
        assert!(grad > 0.0 && grad <= 1.0);
    }

    #[test]
    fn test_bptt_creation() {
        let bptt = BPTT::new(SurrogateType::FastSigmoid, Some(20));
        assert_eq!(bptt.num_steps, Some(20));
    }
}
