//! Spike and neuron importance analysis for explainability

use std::collections::HashMap;

/// Importance score for individual spikes
#[derive(Debug, Clone)]
pub struct SpikeImportance {
    pub neuron_id: usize,
    pub layer: usize,
    pub spike_time: f64,
    pub importance_score: f64,
    pub contribution_to_output: f64,
}

impl SpikeImportance {
    /// Create a new spike importance record
    pub fn new(
        neuron_id: usize,
        layer: usize,
        spike_time: f64,
        importance_score: f64,
        contribution_to_output: f64,
    ) -> Self {
        Self {
            neuron_id,
            layer,
            spike_time,
            importance_score,
            contribution_to_output,
        }
    }
}

/// Aggregated importance for neurons
#[derive(Debug, Clone)]
pub struct NeuronImportance {
    pub neuron_id: usize,
    pub layer: usize,
    pub total_spikes: usize,
    pub mean_importance: f64,
    pub max_importance: f64,
    pub temporal_profile: Vec<f64>,
}

impl NeuronImportance {
    /// Create a new neuron importance record
    pub fn new(
        neuron_id: usize,
        layer: usize,
        total_spikes: usize,
        mean_importance: f64,
        max_importance: f64,
        temporal_profile: Vec<f64>,
    ) -> Self {
        Self {
            neuron_id,
            layer,
            total_spikes,
            mean_importance,
            max_importance,
            temporal_profile,
        }
    }

    /// Get importance at a specific time
    pub fn importance_at(&self, time_ms: f64, bin_size_ms: f64) -> f64 {
        let bin_idx = (time_ms / bin_size_ms).floor() as usize;
        self.temporal_profile.get(bin_idx).copied().unwrap_or(0.0)
    }

    /// Get peak importance time
    pub fn peak_time(&self, bin_size_ms: f64) -> f64 {
        self.temporal_profile
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as f64 * bin_size_ms)
            .unwrap_or(0.0)
    }
}

/// Layer-wise importance summary
#[derive(Debug, Clone)]
pub struct LayerImportance {
    pub layer: usize,
    pub neuron_importances: Vec<NeuronImportance>,
    pub layer_contribution: f64,
}

impl LayerImportance {
    /// Create a new layer importance summary
    pub fn new(layer: usize, neuron_importances: Vec<NeuronImportance>) -> Self {
        let layer_contribution = neuron_importances
            .iter()
            .map(|n| n.mean_importance)
            .sum::<f64>() / neuron_importances.len().max(1) as f64;

        Self {
            layer,
            neuron_importances,
            layer_contribution,
        }
    }

    /// Get top N most important neurons in this layer
    pub fn top_neurons(&self, n: usize) -> Vec<&NeuronImportance> {
        let mut neurons: Vec<_> = self.neuron_importances.iter().collect();
        neurons.sort_by(|a, b| b.mean_importance.total_cmp(&a.mean_importance));
        neurons.into_iter().take(n).collect()
    }
}

/// Compute spike importance using gradient-based method
///
/// This method computes importance by propagating gradients backward through the network.
/// The importance of each spike is determined by its contribution to the output.
///
/// # Arguments
/// * `spike_times` - Per-neuron spike times
/// * `output_gradients` - Gradients at the output layer
/// * `layer_weights` - Weights from each neuron to output neurons (neuron x output)
pub fn compute_spike_importance(
    spike_times: &[Vec<f64>],  // Per-neuron spike times
    output_gradients: &[f64],
    layer_weights: &[Vec<f64>],
) -> Vec<SpikeImportance> {
    let mut importances = Vec::new();

    // For each neuron in the layer
    for (neuron_id, neuron_spikes) in spike_times.iter().enumerate() {
        // Get weights from this neuron to output layer
        let neuron_weights = if neuron_id < layer_weights.len() {
            &layer_weights[neuron_id]
        } else {
            continue;
        };

        // Compute importance for each spike
        for &spike_time in neuron_spikes {
            // Compute weighted gradient contribution
            let mut importance = 0.0;
            let mut contribution = 0.0;

            for (out_idx, &grad) in output_gradients.iter().enumerate() {
                let weight = *neuron_weights.get(out_idx).unwrap_or(&0.0);
                importance += (grad * weight).abs();
                contribution += grad * weight;
            }

            importances.push(SpikeImportance::new(
                neuron_id,
                0, // Layer index would need to be passed in
                spike_time,
                importance,
                contribution,
            ));
        }
    }

    // Normalize importance scores
    let max_importance = importances
        .iter()
        .map(|s| s.importance_score)
        .fold(0.0f64, f64::max);

    if max_importance > 0.0 {
        for spike in &mut importances {
            spike.importance_score /= max_importance;
        }
    }

    importances
}

/// Compute importance by perturbation (spike removal)
///
/// This method computes importance by removing each spike and measuring
/// the change in output. More important spikes cause larger changes.
pub fn compute_importance_by_perturbation(
    spike_trains: &[Vec<f64>],
    output_fn: impl Fn(&[Vec<f64>]) -> f64,
) -> Vec<SpikeImportance> {
    let mut importances = Vec::new();

    // Compute baseline output
    let baseline_output = output_fn(spike_trains);

    // For each neuron
    for (neuron_id, neuron_spikes) in spike_trains.iter().enumerate() {
        // For each spike in this neuron
        for (spike_idx, &spike_time) in neuron_spikes.iter().enumerate() {
            // Create perturbed spike trains (remove this spike)
            let mut perturbed = spike_trains.to_vec();
            perturbed[neuron_id].remove(spike_idx);

            // Compute output with spike removed
            let perturbed_output = output_fn(&perturbed);

            // Importance is the absolute change in output
            let importance = (baseline_output - perturbed_output).abs();
            let contribution = baseline_output - perturbed_output;

            importances.push(SpikeImportance::new(
                neuron_id,
                0,
                spike_time,
                importance,
                contribution,
            ));
        }
    }

    // Normalize importance scores
    let max_importance = importances
        .iter()
        .map(|s| s.importance_score)
        .fold(0.0f64, f64::max);

    if max_importance > 0.0 {
        for spike in &mut importances {
            spike.importance_score /= max_importance;
        }
    }

    importances
}

/// Aggregate spike importance to neuron level
pub fn aggregate_to_neurons(spikes: &[SpikeImportance]) -> Vec<NeuronImportance> {
    let mut neuron_map: HashMap<(usize, usize), Vec<&SpikeImportance>> = HashMap::new();

    // Group spikes by (layer, neuron_id)
    for spike in spikes {
        neuron_map
            .entry((spike.layer, spike.neuron_id))
            .or_insert_with(Vec::new)
            .push(spike);
    }

    // Aggregate for each neuron
    let mut neuron_importances = Vec::new();

    for ((layer, neuron_id), neuron_spikes) in neuron_map {
        let total_spikes = neuron_spikes.len();

        if total_spikes == 0 {
            continue;
        }

        // Compute statistics
        let mean_importance = neuron_spikes
            .iter()
            .map(|s| s.importance_score)
            .sum::<f64>() / total_spikes as f64;

        let max_importance = neuron_spikes
            .iter()
            .map(|s| s.importance_score)
            .fold(0.0f64, f64::max);

        // Create temporal profile (binned)
        let bin_size_ms = 10.0;
        let max_time = neuron_spikes
            .iter()
            .map(|s| s.spike_time)
            .fold(0.0f64, f64::max);
        let n_bins = ((max_time / bin_size_ms).ceil() as usize).max(1);

        let mut temporal_profile = vec![0.0; n_bins];
        for spike in &neuron_spikes {
            let bin_idx = (spike.spike_time / bin_size_ms).floor() as usize;
            if bin_idx < n_bins {
                temporal_profile[bin_idx] += spike.importance_score;
            }
        }

        // Normalize temporal profile
        let max_bin = temporal_profile.iter().copied().fold(0.0f64, f64::max);
        if max_bin > 0.0 {
            for val in &mut temporal_profile {
                *val /= max_bin;
            }
        }

        neuron_importances.push(NeuronImportance::new(
            neuron_id,
            layer,
            total_spikes,
            mean_importance,
            max_importance,
            temporal_profile,
        ));
    }

    neuron_importances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_importance_creation() {
        let spike = SpikeImportance::new(0, 0, 10.0, 0.5, 0.3);
        assert_eq!(spike.neuron_id, 0);
        assert_eq!(spike.layer, 0);
        assert_eq!(spike.spike_time, 10.0);
        assert_eq!(spike.importance_score, 0.5);
        assert_eq!(spike.contribution_to_output, 0.3);
    }

    #[test]
    fn test_neuron_importance_peak_time() {
        let temporal_profile = vec![0.1, 0.3, 0.8, 0.4, 0.2];
        let neuron = NeuronImportance::new(0, 0, 5, 0.5, 0.8, temporal_profile);

        let peak = neuron.peak_time(10.0);
        assert_eq!(peak, 20.0); // Bin 2 * 10ms
    }

    #[test]
    fn test_compute_spike_importance() {
        let spike_times = vec![
            vec![10.0, 20.0, 30.0],
            vec![15.0, 25.0],
        ];
        let output_gradients = vec![1.0, -0.5];
        let layer_weights = vec![
            vec![0.5, 0.3],
            vec![0.2, 0.4],
        ];

        let importances = compute_spike_importance(&spike_times, &output_gradients, &layer_weights);

        assert_eq!(importances.len(), 5); // 3 + 2 spikes

        // Check that importance scores are normalized to [0, 1]
        for imp in &importances {
            assert!(imp.importance_score >= 0.0 && imp.importance_score <= 1.0);
        }
    }

    #[test]
    fn test_compute_importance_by_perturbation() {
        let spike_trains = vec![
            vec![10.0, 20.0],
            vec![15.0],
        ];

        // Simple output function: sum of spike counts
        let output_fn = |trains: &[Vec<f64>]| {
            trains.iter().map(|t| t.len() as f64).sum::<f64>()
        };

        let importances = compute_importance_by_perturbation(&spike_trains, output_fn);

        assert_eq!(importances.len(), 3); // 2 + 1 spikes

        // All spikes should have equal importance since removing any reduces count by 1
        for imp in &importances {
            assert!(imp.importance_score > 0.0);
        }
    }

    #[test]
    fn test_aggregate_to_neurons() {
        let spikes = vec![
            SpikeImportance::new(0, 0, 10.0, 0.5, 0.3),
            SpikeImportance::new(0, 0, 20.0, 0.8, 0.5),
            SpikeImportance::new(1, 0, 15.0, 0.3, 0.2),
        ];

        let neurons = aggregate_to_neurons(&spikes);

        assert_eq!(neurons.len(), 2); // 2 unique neurons

        // Find neuron 0
        let neuron_0 = neurons.iter().find(|n| n.neuron_id == 0).unwrap();
        assert_eq!(neuron_0.total_spikes, 2);
        assert_eq!(neuron_0.mean_importance, (0.5 + 0.8) / 2.0);
        assert_eq!(neuron_0.max_importance, 0.8);
    }

    #[test]
    fn test_layer_importance() {
        let neurons = vec![
            NeuronImportance::new(0, 0, 2, 0.6, 0.8, vec![0.5, 0.7]),
            NeuronImportance::new(1, 0, 3, 0.4, 0.5, vec![0.3, 0.5]),
            NeuronImportance::new(2, 0, 1, 0.8, 0.8, vec![0.8]),
        ];

        let layer = LayerImportance::new(0, neurons);

        assert_eq!(layer.layer, 0);
        assert_eq!(layer.neuron_importances.len(), 3);
        assert!((layer.layer_contribution - 0.6).abs() < 0.01); // (0.6 + 0.4 + 0.8) / 3

        let top = layer.top_neurons(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].neuron_id, 2); // Highest importance
        assert_eq!(top[1].neuron_id, 0); // Second highest
    }
}
