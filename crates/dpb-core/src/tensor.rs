//! Tensor operations for batched spike data.

use crate::error::{DpbError, Result};
use crate::types::{SpikeEvent, SpikeTrain};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

/// Batched spike tensor for efficient processing.
///
/// Shape: (batch_size, num_channels, num_timesteps)
#[derive(Debug, Clone)]
pub struct SpikeTensor {
    /// Spike data (batch, channels, time)
    data: Array3<f32>,
    /// Time step size in seconds
    pub dt: f64,
    /// Number of channels
    pub num_channels: usize,
}

impl SpikeTensor {
    /// Creates a new spike tensor.
    pub fn new(batch_size: usize, num_channels: usize, num_timesteps: usize, dt: f64) -> Result<Self> {
        if batch_size == 0 || num_channels == 0 || num_timesteps == 0 {
            return Err(DpbError::InvalidDimensions(
                "All dimensions must be positive".to_string(),
            ));
        }
        if dt <= 0.0 {
            return Err(DpbError::InvalidParameter(
                "Time step must be positive".to_string(),
            ));
        }

        Ok(Self {
            data: Array3::zeros((batch_size, num_channels, num_timesteps)),
            dt,
            num_channels,
        })
    }

    /// Creates a spike tensor from raw data.
    pub fn from_array(data: Array3<f32>, dt: f64) -> Result<Self> {
        if dt <= 0.0 {
            return Err(DpbError::InvalidParameter(
                "Time step must be positive".to_string(),
            ));
        }
        let num_channels = data.shape()[1];
        Ok(Self {
            data,
            dt,
            num_channels,
        })
    }

    /// Creates a spike tensor from spike trains.
    pub fn from_spike_trains(trains: &[SpikeTrain], num_timesteps: usize, dt: f64) -> Result<Self> {
        if trains.is_empty() {
            return Err(DpbError::InvalidParameter(
                "Spike trains cannot be empty".to_string(),
            ));
        }

        let batch_size = trains.len();
        let num_channels = trains[0].num_channels as usize;

        let mut tensor = Self::new(batch_size, num_channels, num_timesteps, dt)?;

        for (batch_idx, train) in trains.iter().enumerate() {
            for event in &train.events {
                let time_bin = (event.timestamp / dt).floor() as usize;
                if time_bin < num_timesteps {
                    tensor.data[[batch_idx, event.channel as usize, time_bin]] += event.magnitude;
                }
            }
        }

        Ok(tensor)
    }

    /// Returns the shape (batch_size, channels, timesteps).
    pub fn shape(&self) -> (usize, usize, usize) {
        let shape = self.data.shape();
        (shape[0], shape[1], shape[2])
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.data.shape()[0]
    }

    /// Returns the number of timesteps.
    pub fn num_timesteps(&self) -> usize {
        self.data.shape()[2]
    }

    /// Returns the duration in seconds.
    pub fn duration(&self) -> f64 {
        self.num_timesteps() as f64 * self.dt
    }

    /// Returns a view of the data.
    pub fn view(&self) -> ArrayView3<'_, f32> {
        self.data.view()
    }

    /// Returns a mutable reference to the data.
    pub fn data_mut(&mut self) -> &mut Array3<f32> {
        &mut self.data
    }

    /// Gets a single batch item.
    pub fn get_batch(&self, index: usize) -> Result<ArrayView2<'_, f32>> {
        if index >= self.batch_size() {
            return Err(DpbError::OutOfBounds(
                format!("Batch index {} out of bounds", index),
            ));
        }
        Ok(self.data.index_axis(ndarray::Axis(0), index))
    }

    /// Gets a time slice across all batches and channels.
    pub fn get_time_slice(&self, start: usize, end: usize) -> Result<Array3<f32>> {
        if end > self.num_timesteps() {
            return Err(DpbError::OutOfBounds(
                "Time slice exceeds tensor bounds".to_string(),
            ));
        }
        Ok(self.data.slice(ndarray::s![.., .., start..end]).to_owned())
    }

    /// Converts to spike trains.
    pub fn to_spike_trains(&self, threshold: f32) -> Vec<SpikeTrain> {
        let (batch_size, num_channels, num_timesteps) = self.shape();
        let mut trains = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let mut train = SpikeTrain::new(num_channels as u32);

            for channel in 0..num_channels {
                for time_idx in 0..num_timesteps {
                    let value = self.data[[batch_idx, channel, time_idx]];
                    if value.abs() >= threshold {
                        let timestamp = time_idx as f64 * self.dt;
                        let polarity = if value > 0.0 { 1 } else { -1 };
                        let event = SpikeEvent::new(timestamp, channel as u32, polarity, value.abs());
                        train.add_event(event);
                    }
                }
            }

            trains.push(train);
        }

        trains
    }

    /// Applies a threshold to create binary spikes.
    pub fn threshold(&mut self, threshold: f32) {
        self.data.mapv_inplace(|x| if x >= threshold { 1.0 } else { 0.0 });
    }

    /// Computes spike counts per channel.
    pub fn spike_counts(&self) -> Array2<usize> {
        let (batch_size, num_channels, _) = self.shape();
        let mut counts = Array2::zeros((batch_size, num_channels));

        for batch_idx in 0..batch_size {
            for channel in 0..num_channels {
                let count = self.data
                    .index_axis(ndarray::Axis(0), batch_idx)
                    .index_axis(ndarray::Axis(0), channel)
                    .iter()
                    .filter(|&&x| x != 0.0)
                    .count();
                counts[[batch_idx, channel]] = count;
            }
        }

        counts
    }

    /// Computes total spike count.
    pub fn total_spikes(&self) -> usize {
        self.data.iter().filter(|&&x| x != 0.0).count()
    }

    /// Computes mean firing rate (Hz) per channel.
    pub fn mean_firing_rate(&self) -> Array1<f64> {
        let (_, num_channels, _) = self.shape();
        let duration = self.duration();
        let mut rates = Array1::zeros(num_channels);

        for channel in 0..num_channels {
            let count = self.data
                .index_axis(ndarray::Axis(1), channel)
                .iter()
                .filter(|&&x| x != 0.0)
                .count();
            rates[channel] = count as f64 / duration / self.batch_size() as f64;
        }

        rates
    }

    /// Applies a temporal filter.
    pub fn apply_temporal_filter(&mut self, kernel: ArrayView1<f32>) {
        let (batch_size, num_channels, num_timesteps) = self.shape();
        let kernel_size = kernel.len();
        let half_kernel = kernel_size / 2;

        for batch_idx in 0..batch_size {
            for channel in 0..num_channels {
                let mut filtered = Array1::zeros(num_timesteps);

                for t in 0..num_timesteps {
                    let mut sum = 0.0;
                    for k in 0..kernel_size {
                        let t_offset = t as i32 - half_kernel as i32 + k as i32;
                        if t_offset >= 0 && t_offset < num_timesteps as i32 {
                            sum += self.data[[batch_idx, channel, t_offset as usize]] * kernel[k];
                        }
                    }
                    filtered[t] = sum;
                }

                for t in 0..num_timesteps {
                    self.data[[batch_idx, channel, t]] = filtered[t];
                }
            }
        }
    }

    /// Concatenates along the batch dimension.
    pub fn concatenate_batch(tensors: &[SpikeTensor]) -> Result<Self> {
        if tensors.is_empty() {
            return Err(DpbError::InvalidParameter(
                "Cannot concatenate empty tensor list".to_string(),
            ));
        }

        let num_channels = tensors[0].num_channels;
        let num_timesteps = tensors[0].num_timesteps();
        let dt = tensors[0].dt;

        // Verify all tensors have same shape (except batch)
        for tensor in tensors.iter() {
            if tensor.num_channels != num_channels || tensor.num_timesteps() != num_timesteps {
                return Err(DpbError::InvalidDimensions(
                    "All tensors must have same channels and timesteps".to_string(),
                ));
            }
        }

        let arrays: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let concatenated = ndarray::concatenate(
            ndarray::Axis(0),
            &arrays,
        ).map_err(|e| DpbError::Other(format!("Concatenation failed: {}", e)))?;

        Self::from_array(concatenated, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_tensor_creation() {
        let tensor = SpikeTensor::new(4, 10, 100, 0.001).unwrap();
        assert_eq!(tensor.shape(), (4, 10, 100));
        assert_eq!(tensor.batch_size(), 4);
        assert_eq!(tensor.num_channels, 10);
        assert_eq!(tensor.num_timesteps(), 100);
    }

    #[test]
    fn test_from_spike_trains() {
        let mut train1 = SpikeTrain::new(5);
        train1.add_event(SpikeEvent::new(0.001, 0, 1, 1.0));
        train1.add_event(SpikeEvent::new(0.002, 1, 1, 0.8));

        let mut train2 = SpikeTrain::new(5);
        train2.add_event(SpikeEvent::new(0.003, 2, -1, 0.9));

        let tensor = SpikeTensor::from_spike_trains(&[train1, train2], 10, 0.001).unwrap();
        assert_eq!(tensor.batch_size(), 2);
        assert_eq!(tensor.num_channels, 5);
    }

    #[test]
    fn test_to_spike_trains() {
        let mut data = Array3::zeros((1, 3, 5));
        data[[0, 0, 1]] = 1.5;
        data[[0, 1, 3]] = -1.2;

        let tensor = SpikeTensor::from_array(data, 0.001).unwrap();
        let trains = tensor.to_spike_trains(1.0);

        assert_eq!(trains.len(), 1);
        assert_eq!(trains[0].len(), 2);
    }

    #[test]
    fn test_spike_counts() {
        let mut data = Array3::zeros((2, 3, 10));
        data[[0, 0, 1]] = 1.0;
        data[[0, 0, 2]] = 1.0;
        data[[0, 1, 3]] = 1.0;
        data[[1, 2, 5]] = 1.0;

        let tensor = SpikeTensor::from_array(data, 0.001).unwrap();
        let counts = tensor.spike_counts();

        assert_eq!(counts[[0, 0]], 2);
        assert_eq!(counts[[0, 1]], 1);
        assert_eq!(counts[[1, 2]], 1);
    }

    #[test]
    fn test_concatenation() {
        let tensor1 = SpikeTensor::new(2, 5, 10, 0.001).unwrap();
        let tensor2 = SpikeTensor::new(3, 5, 10, 0.001).unwrap();

        let concatenated = SpikeTensor::concatenate_batch(&[tensor1, tensor2]).unwrap();
        assert_eq!(concatenated.batch_size(), 5);
    }

    #[test]
    fn test_mean_firing_rate() {
        let mut data = Array3::zeros((2, 3, 100));
        data[[0, 0, 10]] = 1.0;
        data[[0, 0, 20]] = 1.0;
        data[[1, 0, 30]] = 1.0;

        let tensor = SpikeTensor::from_array(data, 0.001).unwrap();
        let rates = tensor.mean_firing_rate();

        assert!(rates[0] > 0.0);
        assert_eq!(rates[1], 0.0);
    }
}
