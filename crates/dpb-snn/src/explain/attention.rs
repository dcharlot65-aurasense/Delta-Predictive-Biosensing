//! Attention mechanisms for interpretability

/// Temporal attention weights over time
#[derive(Debug, Clone)]
pub struct TemporalAttention {
    pub time_steps: Vec<f64>,
    pub attention_weights: Vec<f64>,
    pub window_size_ms: f64,
}

impl TemporalAttention {
    /// Create a new temporal attention tracker
    pub fn new(duration_ms: f64, resolution_ms: f64) -> Self {
        let n_steps = (duration_ms / resolution_ms).ceil() as usize;
        let time_steps: Vec<f64> = (0..n_steps)
            .map(|i| i as f64 * resolution_ms)
            .collect();

        Self {
            time_steps,
            attention_weights: vec![0.0; n_steps],
            window_size_ms: resolution_ms,
        }
    }

    /// Update attention from spike activity
    pub fn update_from_spikes(&mut self, spike_times: &[f64], weights: &[f64]) {
        assert_eq!(
            spike_times.len(),
            weights.len(),
            "Spike times and weights must have same length"
        );

        for (&spike_time, &weight) in spike_times.iter().zip(weights.iter()) {
            let bin_idx = (spike_time / self.window_size_ms).floor() as usize;
            if bin_idx < self.attention_weights.len() {
                self.attention_weights[bin_idx] += weight.abs();
            }
        }
    }

    /// Get attention at specific time
    pub fn attention_at(&self, time_ms: f64) -> f64 {
        let bin_idx = (time_ms / self.window_size_ms).floor() as usize;
        self.attention_weights.get(bin_idx).copied().unwrap_or(0.0)
    }

    /// Get peak attention times
    pub fn peak_times(&self, n_peaks: usize) -> Vec<f64> {
        let mut indexed_weights: Vec<_> = self
            .attention_weights
            .iter()
            .enumerate()
            .map(|(idx, &weight)| (idx, weight))
            .collect();

        indexed_weights.sort_by(|a, b| b.1.total_cmp(&a.1));

        indexed_weights
            .into_iter()
            .take(n_peaks)
            .map(|(idx, _)| self.time_steps[idx])
            .collect()
    }

    /// Normalize attention weights to sum to 1
    pub fn normalize(&mut self) {
        let sum: f64 = self.attention_weights.iter().sum();
        if sum > 0.0 {
            for weight in &mut self.attention_weights {
                *weight /= sum;
            }
        }
    }

    /// Apply softmax to attention weights
    pub fn softmax(&mut self, temperature: f64) {
        let max_weight = self
            .attention_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute exp and sum
        let mut exp_sum = 0.0;
        for weight in &mut self.attention_weights {
            *weight = ((*weight - max_weight) / temperature).exp();
            exp_sum += *weight;
        }

        // Normalize
        if exp_sum > 0.0 {
            for weight in &mut self.attention_weights {
                *weight /= exp_sum;
            }
        }
    }

    /// Get total attention in time window
    pub fn window_attention(&self, start_ms: f64, end_ms: f64) -> f64 {
        let start_idx = (start_ms / self.window_size_ms).floor() as usize;
        let end_idx = (end_ms / self.window_size_ms).ceil() as usize;

        self.attention_weights[start_idx..end_idx.min(self.attention_weights.len())]
            .iter()
            .sum()
    }
}

/// Spatial attention across input channels/neurons
#[derive(Debug, Clone)]
pub struct SpatialAttention {
    pub channel_ids: Vec<usize>,
    pub attention_weights: Vec<f64>,
}

impl SpatialAttention {
    /// Create a new spatial attention tracker
    pub fn new(n_channels: usize) -> Self {
        Self {
            channel_ids: (0..n_channels).collect(),
            attention_weights: vec![0.0; n_channels],
        }
    }

    /// Update from layer activations
    pub fn update_from_activations(&mut self, activations: &[f64]) {
        assert_eq!(
            activations.len(),
            self.attention_weights.len(),
            "Activations must match number of channels"
        );

        for (weight, &activation) in self.attention_weights.iter_mut().zip(activations.iter()) {
            *weight += activation.abs();
        }
    }

    /// Get top-k attended channels
    pub fn top_channels(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed_weights: Vec<_> = self
            .channel_ids
            .iter()
            .zip(self.attention_weights.iter())
            .map(|(&id, &weight)| (id, weight))
            .collect();

        indexed_weights.sort_by(|a, b| b.1.total_cmp(&a.1));

        indexed_weights.into_iter().take(k).collect()
    }

    /// Normalize attention weights
    pub fn normalize(&mut self) {
        let sum: f64 = self.attention_weights.iter().sum();
        if sum > 0.0 {
            for weight in &mut self.attention_weights {
                *weight /= sum;
            }
        }
    }

    /// Apply softmax to attention weights
    pub fn softmax(&mut self, temperature: f64) {
        let max_weight = self
            .attention_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Compute exp and sum
        let mut exp_sum = 0.0;
        for weight in &mut self.attention_weights {
            *weight = ((*weight - max_weight) / temperature).exp();
            exp_sum += *weight;
        }

        // Normalize
        if exp_sum > 0.0 {
            for weight in &mut self.attention_weights {
                *weight /= exp_sum;
            }
        }
    }

    /// Get attention for specific channel
    pub fn channel_attention(&self, channel_id: usize) -> f64 {
        self.channel_ids
            .iter()
            .position(|&id| id == channel_id)
            .and_then(|idx| self.attention_weights.get(idx))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Combined spatiotemporal attention map
#[derive(Debug, Clone)]
pub struct AttentionMap {
    pub temporal: TemporalAttention,
    pub spatial: SpatialAttention,
    pub cross_attention: Option<Vec<Vec<f64>>>, // Channel x Time
}

impl AttentionMap {
    /// Create a new attention map
    pub fn new(n_channels: usize, duration_ms: f64, resolution_ms: f64) -> Self {
        Self {
            temporal: TemporalAttention::new(duration_ms, resolution_ms),
            spatial: SpatialAttention::new(n_channels),
            cross_attention: None,
        }
    }

    /// Compute joint attention (outer product of spatial and temporal)
    pub fn compute_cross_attention(&mut self) {
        let n_channels = self.spatial.attention_weights.len();
        let n_time_steps = self.temporal.attention_weights.len();

        let mut cross = vec![vec![0.0; n_time_steps]; n_channels];

        for (channel_idx, spatial_weight) in self.spatial.attention_weights.iter().enumerate() {
            for (time_idx, temporal_weight) in self.temporal.attention_weights.iter().enumerate() {
                cross[channel_idx][time_idx] = spatial_weight * temporal_weight;
            }
        }

        self.cross_attention = Some(cross);
    }

    /// Get attention at specific channel and time
    pub fn attention_at(&self, channel: usize, time_ms: f64) -> f64 {
        if let Some(ref cross) = self.cross_attention {
            let time_idx = (time_ms / self.temporal.window_size_ms).floor() as usize;
            cross
                .get(channel)
                .and_then(|row| row.get(time_idx))
                .copied()
                .unwrap_or(0.0)
        } else {
            // Fallback to product of marginals
            let spatial_att = self.spatial.channel_attention(channel);
            let temporal_att = self.temporal.attention_at(time_ms);
            spatial_att * temporal_att
        }
    }

    /// Normalize all attention components
    pub fn normalize_all(&mut self) {
        self.spatial.normalize();
        self.temporal.normalize();

        if self.cross_attention.is_some() {
            self.compute_cross_attention();
        }
    }

    /// Get peak attention location (channel, time)
    pub fn peak_attention(&self) -> Option<(usize, f64)> {
        if let Some(ref cross) = self.cross_attention {
            let mut max_val = f64::NEG_INFINITY;
            let mut max_pos = None;

            for (channel, row) in cross.iter().enumerate() {
                for (time_idx, &val) in row.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_pos = Some((channel, self.temporal.time_steps[time_idx]));
                    }
                }
            }

            max_pos
        } else {
            None
        }
    }

    /// Get attention heatmap data
    pub fn get_heatmap(&self) -> Option<&Vec<Vec<f64>>> {
        self.cross_attention.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_attention_creation() {
        let attention = TemporalAttention::new(100.0, 10.0);
        assert_eq!(attention.time_steps.len(), 10);
        assert_eq!(attention.attention_weights.len(), 10);
        assert_eq!(attention.window_size_ms, 10.0);
    }

    #[test]
    fn test_temporal_attention_update() {
        let mut attention = TemporalAttention::new(100.0, 10.0);

        let spike_times = vec![5.0, 15.0, 25.0];
        let weights = vec![1.0, 2.0, 1.5];

        attention.update_from_spikes(&spike_times, &weights);

        assert_eq!(attention.attention_weights[0], 1.0); // Bin 0
        assert_eq!(attention.attention_weights[1], 2.0); // Bin 1
        assert_eq!(attention.attention_weights[2], 1.5); // Bin 2
    }

    #[test]
    fn test_temporal_attention_normalize() {
        let mut attention = TemporalAttention::new(100.0, 10.0);
        attention.attention_weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        attention.normalize();

        let sum: f64 = attention.attention_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_temporal_peak_times() {
        let mut attention = TemporalAttention::new(100.0, 10.0);
        attention.attention_weights = vec![1.0, 5.0, 2.0, 8.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let peaks = attention.peak_times(3);
        assert_eq!(peaks.len(), 3);
        assert_eq!(peaks[0], 30.0); // Bin 3 has highest weight (8.0)
        assert_eq!(peaks[1], 10.0); // Bin 1 has second highest (5.0)
        assert_eq!(peaks[2], 40.0); // Bin 4 has third highest (3.0)
    }

    #[test]
    fn test_spatial_attention_creation() {
        let attention = SpatialAttention::new(5);
        assert_eq!(attention.channel_ids.len(), 5);
        assert_eq!(attention.attention_weights.len(), 5);
    }

    #[test]
    fn test_spatial_attention_update() {
        let mut attention = SpatialAttention::new(3);
        let activations = vec![1.0, 2.0, 3.0];

        attention.update_from_activations(&activations);

        assert_eq!(attention.attention_weights[0], 1.0);
        assert_eq!(attention.attention_weights[1], 2.0);
        assert_eq!(attention.attention_weights[2], 3.0);
    }

    #[test]
    fn test_spatial_top_channels() {
        let mut attention = SpatialAttention::new(5);
        attention.attention_weights = vec![1.0, 5.0, 2.0, 8.0, 3.0];

        let top = attention.top_channels(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 3); // Channel 3 has highest weight
        assert_eq!(top[1].0, 1); // Channel 1 has second highest
        assert_eq!(top[2].0, 4); // Channel 4 has third highest
    }

    #[test]
    fn test_attention_map_creation() {
        let map = AttentionMap::new(5, 100.0, 10.0);
        assert_eq!(map.spatial.attention_weights.len(), 5);
        assert_eq!(map.temporal.attention_weights.len(), 10);
        assert!(map.cross_attention.is_none());
    }

    #[test]
    fn test_attention_map_cross_attention() {
        let mut map = AttentionMap::new(3, 100.0, 10.0);
        map.spatial.attention_weights = vec![1.0, 2.0, 3.0];
        map.temporal.attention_weights = vec![0.5, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        map.compute_cross_attention();

        assert!(map.cross_attention.is_some());

        let cross = map.cross_attention.as_ref().unwrap();
        assert_eq!(cross.len(), 3); // 3 channels
        assert_eq!(cross[0].len(), 10); // 10 time steps

        // Check some values
        assert_eq!(cross[0][0], 1.0 * 0.5); // Channel 0, Time 0
        assert_eq!(cross[2][1], 3.0 * 1.0); // Channel 2, Time 1
    }

    #[test]
    fn test_attention_map_peak() {
        let mut map = AttentionMap::new(2, 100.0, 10.0);
        map.spatial.attention_weights = vec![1.0, 2.0];
        map.temporal.attention_weights = vec![1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        map.compute_cross_attention();

        let peak = map.peak_attention();
        assert!(peak.is_some());

        let (channel, time) = peak.unwrap();
        assert_eq!(channel, 1); // Channel 1 (weight 2.0)
        assert_eq!(time, 10.0); // Time bin 1 (weight 3.0)
    }

    #[test]
    fn test_temporal_softmax() {
        let mut attention = TemporalAttention::new(100.0, 10.0);
        attention.attention_weights = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        attention.softmax(1.0);

        // Check sum is 1
        let sum: f64 = attention.attention_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check highest input has highest output
        assert!(attention.attention_weights[2] > attention.attention_weights[1]);
        assert!(attention.attention_weights[1] > attention.attention_weights[0]);
    }
}
