//! Core traits for the DPB framework.

use crate::error::Result;
use crate::types::{Context, GroundTruth, SpikeEvent, TimeSeries};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;

/// Trait for signal data types.
pub trait Signal: Send + Sync {
    /// Returns the signal samples as a slice.
    fn samples(&self) -> &[f32];

    /// Returns the sample rate in Hz.
    fn sample_rate(&self) -> f64;

    /// Returns the duration in seconds.
    fn duration(&self) -> f64 {
        self.samples().len() as f64 / self.sample_rate()
    }

    /// Returns the number of channels.
    fn channels(&self) -> usize {
        1 // Default single channel
    }

    /// Returns the number of channels (alias for channels()).
    fn channel_count(&self) -> usize {
        self.channels()
    }

    /// Returns samples for a specific channel.
    /// For single-channel signals, channel 0 returns all samples, others return None.
    /// For multi-channel signals, this should be overridden.
    fn channel(&self, channel_idx: usize) -> Option<&[f32]> {
        if channel_idx == 0 {
            Some(self.samples())
        } else {
            None
        }
    }
}

/// Trait for encoding analog signals to spike trains.
pub trait EventEncoder: Send + Sync {
    /// Configuration type for this encoder.
    type Config: Clone + Send + Sync;

    /// Returns the name of this encoder.
    fn name(&self) -> &str;

    /// Encodes a signal to spike events using the provided configuration.
    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>>;

    /// Returns the ground truth for encoded data (if applicable).
    fn ground_truth(&self) -> Option<&GroundTruth> {
        None
    }
}

/// Trait for reconstructing an analog signal from spike events.
///
/// The inverse of [`EventEncoder`]. Without it an encoder's output cannot be
/// checked against its input, so no statement about how much information the
/// encoding discarded can be supported — an events-per-second figure means
/// nothing on its own.
///
/// Implementations should document their error bound. For level-crossing
/// encoding the bound is one threshold quantum: the encoder emits whenever the
/// signal departs by `threshold` from the last emitted level, so between events
/// the signal is known to have stayed inside that band.
pub trait EventDecoder: Send + Sync {
    /// Configuration type for this decoder.
    type Config: Clone + Send + Sync;

    /// Returns the name of this decoder.
    fn name(&self) -> &str;

    /// Reconstructs `n_samples` at `sample_rate` from `events`.
    fn reconstruct(
        &self,
        events: &[SpikeEvent],
        sample_rate: f64,
        n_samples: usize,
        config: &Self::Config,
    ) -> Result<Vec<f32>>;

    /// Worst-case absolute error this decoder guarantees, if it guarantees one.
    ///
    /// `None` means no bound is claimed — which is itself worth knowing, and is
    /// the honest answer for encoders that discard information irreversibly.
    fn error_bound(&self, _config: &Self::Config) -> Option<f32> {
        None
    }
}

/// Reconstruction fidelity of an encode/decode round trip.
///
/// Exists so efficiency and fidelity are always reported together. Compression
/// without an error figure is not a result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructionQuality {
    /// Root-mean-square error between original and reconstruction.
    pub rmse: f32,
    /// Largest absolute error at any sample.
    pub max_abs_error: f32,
    /// Signal-to-noise ratio in dB. Infinite for an exact reconstruction.
    pub snr_db: f32,
    /// Original samples divided by emitted events. 1.0 means no reduction.
    pub compression_ratio: f32,
}

impl ReconstructionQuality {
    /// Compares a reconstruction against the original signal.
    ///
    /// `n_events` is carried through so compression and fidelity cannot be
    /// quoted apart from one another.
    pub fn compare(original: &[f32], reconstructed: &[f32], n_events: usize) -> Self {
        let n = original.len().min(reconstructed.len());
        if n == 0 {
            return Self {
                rmse: 0.0,
                max_abs_error: 0.0,
                snr_db: f32::INFINITY,
                compression_ratio: 1.0,
            };
        }
        let mut sq_err = 0.0f64;
        let mut sq_sig = 0.0f64;
        let mut max_abs = 0.0f32;
        for i in 0..n {
            let e = original[i] - reconstructed[i];
            sq_err += (e as f64) * (e as f64);
            sq_sig += (original[i] as f64) * (original[i] as f64);
            max_abs = max_abs.max(e.abs());
        }
        let rmse = (sq_err / n as f64).sqrt() as f32;
        let snr_db = if sq_err <= f64::EPSILON {
            f32::INFINITY
        } else {
            (10.0 * (sq_sig / sq_err).log10()) as f32
        };
        let compression_ratio = if n_events == 0 {
            f32::INFINITY
        } else {
            original.len() as f32 / n_events as f32
        };
        Self {
            rmse,
            max_abs_error: max_abs,
            snr_db,
            compression_ratio,
        }
    }
}

/// Trait for population-based encoding templates.
///
/// Population templates provide clinical reference values based on demographic context
/// (age, sex, etc.) for biosignal analysis.
pub trait PopulationTemplate: Send + Sync {
    /// Returns the name of this template.
    fn name(&self) -> &str;

    /// Returns the expected value based on demographic context.
    fn expected_value(&self, context: &Context) -> f64;

    /// Returns the variance based on demographic context.
    fn variance(&self, context: &Context) -> f64;

    /// Returns the standard deviation based on demographic context.
    fn deviation(&self, context: &Context) -> f64 {
        self.variance(context).sqrt()
    }

    /// Returns the number of neurons in the population (default: 1).
    fn population_size(&self) -> usize {
        1
    }
}

/// Trait for membrane dynamics models.
pub trait MembraneDynamics: Send + Sync {
    /// Updates the membrane potential given input current and time step.
    fn update(&mut self, input_current: f64, dt: f64) -> bool;

    /// Resets the membrane to resting state.
    fn reset(&mut self);

    /// Returns the current membrane potential.
    fn membrane_potential(&self) -> f64;

    /// Returns the spike threshold.
    fn threshold(&self) -> f64;
}

/// Trait for synaptic models.
pub trait SynapticModel: Send + Sync {
    /// Applies synaptic weights to input spikes.
    fn apply_weights(&self, inputs: ArrayView1<f32>, weights: ArrayView2<f32>) -> Array1<f32>;

    /// Updates synaptic weights (for learning).
    fn update_weights(&mut self, delta: ArrayView2<f32>);

    /// Returns the current weight matrix.
    fn weights(&self) -> ArrayView2<'_, f32>;
}

/// Trait for surrogate gradient functions.
pub trait SurrogateGradient: Send + Sync {
    /// Forward pass (spike generation).
    fn forward(&self, membrane_potential: f64, threshold: f64) -> f64;

    /// Backward pass (gradient computation).
    fn backward(&self, membrane_potential: f64, threshold: f64) -> f64;

    /// Returns the gradient scale parameter.
    fn scale(&self) -> f64 {
        1.0
    }
}

/// Trait for spiking neural network layers.
pub trait SpikingLayer: Send + Sync {
    /// Forward pass through the layer.
    fn forward(&mut self, input: ArrayView2<f32>, dt: f64) -> Result<Array2<f32>>;

    /// Resets the layer state.
    fn reset_state(&mut self);

    /// Returns the number of neurons in the layer.
    fn neuron_count(&self) -> usize;

    /// Returns the layer as Any for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns mutable Any for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Trait for complete spiking neural networks.
pub trait SNNNetwork: Send + Sync {
    /// Forward pass through the entire network.
    fn forward(&mut self, input: ArrayView2<f32>, dt: f64) -> Result<Array2<f32>>;

    /// Returns references to all layers.
    fn layers(&self) -> Vec<&dyn SpikingLayer>;

    /// Returns mutable references to all layers.
    fn layers_mut(&mut self) -> Vec<&mut dyn SpikingLayer>;

    /// Resets all network state.
    fn reset(&mut self) {
        for layer in self.layers_mut() {
            layer.reset_state();
        }
    }
}

/// Trait for loss functions in SNN training.
pub trait LossFunction: Send + Sync {
    /// Computes the loss value.
    fn compute(&self, predictions: ArrayView2<f32>, targets: ArrayView2<f32>) -> Result<f64>;

    /// Computes the gradient of the loss.
    fn gradient(
        &self,
        predictions: ArrayView2<f32>,
        targets: ArrayView2<f32>,
    ) -> Result<Array2<f32>>;

    /// Returns the loss function name.
    fn name(&self) -> &str;
}

/// Trait for optimizers.
pub trait Optimizer: Send + Sync {
    /// Performs a single optimization step.
    fn step(&mut self, gradients: &HashMap<String, Array2<f32>>) -> Result<()>;

    /// Returns the learning rate.
    fn learning_rate(&self) -> f64;

    /// Sets the learning rate.
    fn set_learning_rate(&mut self, lr: f64);

    /// Resets optimizer state.
    fn reset(&mut self);
}

/// Trait for evaluation metrics.
pub trait Metric: Send + Sync {
    /// Updates the metric with new predictions and targets.
    fn update(&mut self, predictions: ArrayView1<u32>, targets: ArrayView1<u32>) -> Result<()>;

    /// Computes the final metric value.
    fn compute(&self) -> Result<f64>;

    /// Resets the metric state.
    fn reset(&mut self);

    /// Returns the metric name.
    fn name(&self) -> &str;
}

/// Trait for datasets.
pub trait Dataset: Send + Sync {
    /// Returns the number of samples in the dataset.
    fn len(&self) -> usize;

    /// Returns true if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a single sample and its label.
    fn get(&self, index: usize) -> Result<(TimeSeries, GroundTruth)>;

    /// Returns a batch of samples.
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<(TimeSeries, GroundTruth)>> {
        indices.iter().map(|&i| self.get(i)).collect()
    }
}

/// Trait for synthetic data generation.
pub trait SyntheticGenerator: Send + Sync {
    /// Generates a synthetic sample.
    fn generate(&mut self) -> Result<TimeSeries>;

    /// Returns the ground truth for the last generated sample.
    fn ground_truth(&self) -> Result<GroundTruth>;

    /// Sets the random seed.
    fn set_seed(&mut self, seed: u64);

    /// Returns generator configuration.
    fn config(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Trait for feature extraction.
pub trait FeatureExtractor: Send + Sync {
    /// Extracts features from a time series.
    fn extract(&self, signal: &TimeSeries) -> Result<Array1<f64>>;

    /// Returns the feature dimension.
    fn feature_dim(&self) -> usize;

    /// Returns feature names.
    fn feature_names(&self) -> Vec<String> {
        (0..self.feature_dim())
            .map(|i| format!("feature_{}", i))
            .collect()
    }
}

/// Trait for GPU compute kernels.
pub trait GpuKernel: Send + Sync {
    /// Dispatches the kernel to the GPU.
    ///
    /// Note: The encoder type is opaque to allow for API changes
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        workgroups: (u32, u32, u32),
    ) -> Result<()>;

    /// Returns the bind group for the kernel.
    fn bind_group(&self) -> &wgpu::BindGroup;

    /// Returns the pipeline.
    fn pipeline(&self) -> &wgpu::ComputePipeline;
}

/// Trait for hardware export capabilities.
pub trait HardwareExporter: Send + Sync {
    /// Exports the model to a specific hardware platform.
    fn export(&self, path: &std::path::Path) -> Result<()>;

    /// Returns the target hardware platform.
    fn target_platform(&self) -> &str;

    /// Returns platform-specific configuration.
    fn platform_config(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Trait for power estimation.
pub trait PowerEstimator: Send + Sync {
    /// Estimates power consumption for a given operation.
    fn estimate(&self, operations: u64, data_size: usize) -> Result<f64>;

    /// Returns power consumption in watts.
    fn power_watts(&self) -> f64;

    /// Returns energy consumption in joules.
    fn energy_joules(&self, duration_seconds: f64) -> f64 {
        self.power_watts() * duration_seconds
    }
}

/// Trait for convergence analysis.
pub trait ConvergenceAnalyzer: Send + Sync {
    /// Computes time to reach target accuracy.
    fn time_to_accuracy(&self, target_accuracy: f64) -> Result<Option<f64>>;

    /// Returns the convergence curve (epochs, accuracy).
    fn convergence_curve(&self) -> Result<Vec<(usize, f64)>>;

    /// Returns whether training has converged.
    fn has_converged(&self, tolerance: f64) -> bool;
}

/// Trait for configurable components.
pub trait Configurable: Sized {
    /// Configuration type for this component.
    type Config: Serialize + for<'de> Deserialize<'de>;

    /// Creates an instance from configuration.
    fn from_config(config: &Self::Config) -> Result<Self>;

    /// Returns the current configuration.
    fn to_config(&self) -> Self::Config;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementations for testing
    struct MockSignal {
        data: Vec<f32>,
        rate: f64,
        num_channels: usize,
    }

    impl Signal for MockSignal {
        fn samples(&self) -> &[f32] {
            &self.data
        }

        fn sample_rate(&self) -> f64 {
            self.rate
        }

        fn channels(&self) -> usize {
            self.num_channels
        }
    }

    #[test]
    fn test_signal_trait() {
        let data = vec![0.0_f32; 1000];
        let signal = MockSignal {
            data,
            rate: 100.0,
            num_channels: 1,
        };

        assert_eq!(signal.channels(), 1);
        assert_eq!(signal.duration(), 10.0);
    }

    struct MockMembraneDynamics {
        potential: f64,
        threshold: f64,
    }

    impl MembraneDynamics for MockMembraneDynamics {
        fn update(&mut self, input_current: f64, _dt: f64) -> bool {
            self.potential += input_current;
            if self.potential >= self.threshold {
                self.reset();
                true
            } else {
                false
            }
        }

        fn reset(&mut self) {
            self.potential = 0.0;
        }

        fn membrane_potential(&self) -> f64 {
            self.potential
        }

        fn threshold(&self) -> f64 {
            self.threshold
        }
    }

    #[test]
    fn test_membrane_dynamics() {
        let mut neuron = MockMembraneDynamics {
            potential: 0.0,
            threshold: 1.0,
        };

        assert!(!neuron.update(0.5, 0.001));
        assert_eq!(neuron.membrane_potential(), 0.5);

        assert!(neuron.update(0.6, 0.001));
        assert_eq!(neuron.membrane_potential(), 0.0);
    }
}
