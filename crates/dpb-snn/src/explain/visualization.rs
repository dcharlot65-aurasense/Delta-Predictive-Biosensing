//! Visualization helpers for explainability

use crate::explain::{AttentionMap, FeatureAttribution, SpikeImportance};
use std::collections::HashMap;

/// Heatmap data for visualization
#[derive(Debug, Clone)]
pub struct HeatmapData {
    pub data: Vec<Vec<f64>>,
    pub x_labels: Vec<String>,
    pub y_labels: Vec<String>,
    pub title: String,
    pub colormap: String,
}

impl HeatmapData {
    /// Create a new heatmap data structure
    pub fn new(
        data: Vec<Vec<f64>>,
        x_labels: Vec<String>,
        y_labels: Vec<String>,
        title: String,
        colormap: String,
    ) -> Self {
        Self {
            data,
            x_labels,
            y_labels,
            title,
            colormap,
        }
    }

    /// Get dimensions (rows, cols)
    pub fn dimensions(&self) -> (usize, usize) {
        let rows = self.data.len();
        let cols = self.data.get(0).map(|row| row.len()).unwrap_or(0);
        (rows, cols)
    }

    /// Get value at position
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        self.data.get(row)?.get(col).copied()
    }

    /// Get min and max values
    pub fn value_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for row in &self.data {
            for &val in row {
                min = min.min(val);
                max = max.max(val);
            }
        }

        (min, max)
    }

    /// Normalize data to [0, 1] range
    pub fn normalize(&mut self) {
        let (min, max) = self.value_range();
        let range = max - min;

        if range > 0.0 {
            for row in &mut self.data {
                for val in row {
                    *val = (*val - min) / range;
                }
            }
        }
    }
}

/// Explanation visualizer
pub struct ExplanationVisualizer;

impl ExplanationVisualizer {
    /// Generate heatmap data from attention map
    pub fn attention_heatmap(attention: &AttentionMap) -> HeatmapData {
        // Get or compute cross attention
        let cross_data = if let Some(ref cross) = attention.cross_attention {
            cross.clone()
        } else {
            // Compute on-the-fly
            let n_channels = attention.spatial.attention_weights.len();
            let n_time = attention.temporal.attention_weights.len();

            let mut data = vec![vec![0.0; n_time]; n_channels];
            for (ch, spatial_weight) in attention.spatial.attention_weights.iter().enumerate() {
                for (t, temporal_weight) in attention.temporal.attention_weights.iter().enumerate() {
                    data[ch][t] = spatial_weight * temporal_weight;
                }
            }
            data
        };

        // Generate labels
        let x_labels: Vec<String> = attention
            .temporal
            .time_steps
            .iter()
            .map(|&t| format!("{:.1}ms", t))
            .collect();

        let y_labels: Vec<String> = attention
            .spatial
            .channel_ids
            .iter()
            .map(|&id| format!("Ch{}", id))
            .collect();

        HeatmapData::new(
            cross_data,
            x_labels,
            y_labels,
            "Spatiotemporal Attention".to_string(),
            "viridis".to_string(),
        )
    }

    /// Generate bar chart data from feature attribution
    pub fn attribution_bars(attribution: &FeatureAttribution) -> (Vec<String>, Vec<f64>) {
        let features = attribution.feature_names.clone();
        let values = attribution.attribution_values.clone();

        // Sort by absolute value (descending)
        let mut indexed: Vec<_> = features.into_iter().zip(values).collect();
        indexed.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));

        indexed.into_iter().unzip()
    }

    /// Generate spike raster with importance coloring
    pub fn importance_raster(
        spike_times: &[Vec<f64>],
        importances: &[SpikeImportance],
    ) -> Vec<(f64, usize, f64)> {
        // Create lookup map for importance scores
        let mut importance_map: HashMap<(usize, usize), f64> = HashMap::new();

        for imp in importances {
            // Use time rounded to nearest millisecond as key
            let time_key = (imp.spike_time * 10.0).round() as usize;
            importance_map.insert((imp.neuron_id, time_key), imp.importance_score);
        }

        // Generate raster points with importance
        let mut raster = Vec::new();

        for (neuron_id, neuron_spikes) in spike_times.iter().enumerate() {
            for &spike_time in neuron_spikes {
                let time_key = (spike_time * 10.0).round() as usize;
                let importance = importance_map
                    .get(&(neuron_id, time_key))
                    .copied()
                    .unwrap_or(0.0);

                raster.push((spike_time, neuron_id, importance));
            }
        }

        // Sort by time
        raster.sort_by(|a, b| a.0.total_cmp(&b.0));

        raster
    }

    /// Generate temporal importance curve
    pub fn temporal_importance_curve(
        importances: &[SpikeImportance],
        bin_size_ms: f64,
        duration_ms: f64,
    ) -> Vec<(f64, f64)> {
        let n_bins = (duration_ms / bin_size_ms).ceil() as usize;
        let mut bins = vec![0.0; n_bins];

        // Bin the importance scores
        for imp in importances {
            let bin_idx = (imp.spike_time / bin_size_ms).floor() as usize;
            if bin_idx < n_bins {
                bins[bin_idx] += imp.importance_score;
            }
        }

        // Convert to time-value pairs
        (0..n_bins)
            .map(|i| {
                let time = i as f64 * bin_size_ms;
                (time, bins[i])
            })
            .collect()
    }

    /// Generate neuron importance ranking
    pub fn neuron_ranking(
        importances: &[SpikeImportance],
    ) -> Vec<(usize, usize, f64)> {
        // Group by (layer, neuron_id)
        let mut neuron_scores: HashMap<(usize, usize), f64> = HashMap::new();

        for imp in importances {
            let key = (imp.layer, imp.neuron_id);
            *neuron_scores.entry(key).or_insert(0.0) += imp.importance_score;
        }

        // Convert to sorted list
        let mut ranking: Vec<_> = neuron_scores
            .into_iter()
            .map(|((layer, neuron), score)| (layer, neuron, score))
            .collect();

        ranking.sort_by(|a, b| b.2.total_cmp(&a.2));

        ranking
    }

    /// Generate layer contribution summary
    pub fn layer_contributions(
        importances: &[SpikeImportance],
    ) -> Vec<(usize, f64, usize)> {
        // Group by layer
        let mut layer_data: HashMap<usize, (f64, usize)> = HashMap::new();

        for imp in importances {
            let entry = layer_data.entry(imp.layer).or_insert((0.0, 0));
            entry.0 += imp.importance_score;
            entry.1 += 1;
        }

        // Convert to sorted list (layer, total_importance, spike_count)
        let mut contributions: Vec<_> = layer_data
            .into_iter()
            .map(|(layer, (score, count))| (layer, score, count))
            .collect();

        contributions.sort_by_key(|&(layer, _, _)| layer);

        contributions
    }

    /// Generate contribution breakdown (positive vs negative)
    pub fn contribution_breakdown(
        attribution: &FeatureAttribution,
    ) -> (f64, f64, Vec<String>, Vec<String>) {
        let mut positive_sum = 0.0;
        let mut negative_sum = 0.0;
        let mut positive_features = Vec::new();
        let mut negative_features = Vec::new();

        for (name, &value) in attribution.feature_names.iter().zip(&attribution.attribution_values) {
            if value >= 0.0 {
                positive_sum += value;
                positive_features.push(format!("{}: {:.3}", name, value));
            } else {
                negative_sum += value.abs();
                negative_features.push(format!("{}: {:.3}", name, value));
            }
        }

        (positive_sum, negative_sum, positive_features, negative_features)
    }
}

/// Export explanation to JSON for external visualization
pub fn export_explanation_json(
    attention: Option<&AttentionMap>,
    attribution: Option<&FeatureAttribution>,
    importance: Option<&[SpikeImportance]>,
) -> String {
    // Only the object's MEMBERS go in here; the braces are added at the end.
    //
    // They were previously pushed as elements and then joined with the members
    // using ", ", so the result was `{, "importance": [...], }` -- a comma
    // immediately after the opening brace and before the closing one. That is
    // invalid JSON for every input, including the empty case `{, }`, so this
    // function never once produced parseable output.
    let mut json_parts: Vec<String> = Vec::new();

    // Export attention
    if let Some(att) = attention {
        let spatial_json = format!(
            "\"spatial_attention\": {{\"channels\": {:?}, \"weights\": {}}}",
            att.spatial.channel_ids,
            json_number_array(&att.spatial.attention_weights)
        );

        let temporal_json = format!(
            "\"temporal_attention\": {{\"times\": {}, \"weights\": {}}}",
            json_number_array(&att.temporal.time_steps),
            json_number_array(&att.temporal.attention_weights)
        );

        json_parts.push(spatial_json);
        json_parts.push(temporal_json);
    }

    // Export attribution
    if let Some(attr) = attribution {
        let attr_json = format!(
            "\"attribution\": {{\"features\": {:?}, \"values\": {}, \"baseline\": {}, \"output\": {}}}",
            attr.feature_names,
            json_number_array(&attr.attribution_values),
            json_number(attr.baseline_output),
            json_number(attr.actual_output)
        );

        json_parts.push(attr_json);
    }

    // Export importance
    if let Some(imp) = importance {
        let imp_data: Vec<String> = imp
            .iter()
            .map(|s| {
                format!(
                    "{{\"neuron\": {}, \"layer\": {}, \"time\": {}, \"importance\": {}, \"contribution\": {}}}",
                    s.neuron_id,
                    s.layer,
                    json_number(s.spike_time),
                    json_number(s.importance_score),
                    json_number(s.contribution_to_output)
                )
            })
            .collect();

        let imp_json = format!("\"importance\": [{}]", imp_data.join(", "));
        json_parts.push(imp_json);
    }

    format!("{{{}}}", json_parts.join(", "))
}

/// Format an `f64` as a JSON number.
///
/// JSON has no NaN or infinity, so non-finite values become `null` rather than
/// bare `NaN`/`inf` tokens, which no parser accepts.
fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "null".to_string()
    }
}

/// Format a slice of `f64` as a JSON array, via [`json_number`].
fn json_number_array(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().copied().map(json_number).collect();
    format!("[{}]", items.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explain::{SpatialAttention, TemporalAttention};

    #[test]
    fn test_heatmap_creation() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let x_labels = vec!["t1".to_string(), "t2".to_string(), "t3".to_string()];
        let y_labels = vec!["ch1".to_string(), "ch2".to_string()];

        let heatmap = HeatmapData::new(
            data,
            x_labels,
            y_labels,
            "Test".to_string(),
            "viridis".to_string(),
        );

        assert_eq!(heatmap.dimensions(), (2, 3));
        assert_eq!(heatmap.get(0, 0), Some(1.0));
        assert_eq!(heatmap.get(1, 2), Some(6.0));
    }

    #[test]
    fn test_heatmap_value_range() {
        let data = vec![
            vec![1.0, 5.0, 3.0],
            vec![2.0, 8.0, 4.0],
        ];
        let heatmap = HeatmapData::new(
            data,
            vec![],
            vec![],
            "Test".to_string(),
            "viridis".to_string(),
        );

        let (min, max) = heatmap.value_range();
        assert_eq!(min, 1.0);
        assert_eq!(max, 8.0);
    }

    #[test]
    fn test_heatmap_normalize() {
        let data = vec![
            vec![1.0, 5.0],
            vec![3.0, 9.0],
        ];
        let mut heatmap = HeatmapData::new(
            data,
            vec![],
            vec![],
            "Test".to_string(),
            "viridis".to_string(),
        );

        heatmap.normalize();

        let (min, max) = heatmap.value_range();
        assert!((min - 0.0).abs() < 1e-6);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_attention_heatmap() {
        let mut map = AttentionMap::new(3, 100.0, 10.0);
        map.spatial.attention_weights = vec![1.0, 2.0, 3.0];
        map.temporal.attention_weights = vec![0.5, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        map.compute_cross_attention();

        let heatmap = ExplanationVisualizer::attention_heatmap(&map);

        assert_eq!(heatmap.dimensions(), (3, 10));
        assert_eq!(heatmap.title, "Spatiotemporal Attention");
    }

    #[test]
    fn test_attribution_bars() {
        let names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];
        let values = vec![0.5, 0.8, 0.2];
        let attribution = FeatureAttribution::new(names, values, 0.0, 1.5);

        let (bar_names, bar_values) = ExplanationVisualizer::attribution_bars(&attribution);

        assert_eq!(bar_names.len(), 3);
        assert_eq!(bar_values.len(), 3);

        // Should be sorted by absolute value
        assert_eq!(bar_names[0], "f2"); // 0.8 is highest
        assert_eq!(bar_names[1], "f1"); // 0.5 is second
        assert_eq!(bar_names[2], "f3"); // 0.2 is lowest
    }

    #[test]
    fn test_importance_raster() {
        let spike_times = vec![
            vec![10.0, 20.0],
            vec![15.0, 25.0],
        ];

        let importances = vec![
            SpikeImportance::new(0, 0, 10.0, 0.8, 0.5),
            SpikeImportance::new(0, 0, 20.0, 0.6, 0.3),
            SpikeImportance::new(1, 0, 15.0, 0.9, 0.7),
        ];

        let raster = ExplanationVisualizer::importance_raster(&spike_times, &importances);

        assert_eq!(raster.len(), 4); // 4 total spikes

        // Check that spikes are sorted by time
        assert!(raster[0].0 <= raster[1].0);
        assert!(raster[1].0 <= raster[2].0);
        assert!(raster[2].0 <= raster[3].0);
    }

    #[test]
    fn test_temporal_importance_curve() {
        let importances = vec![
            SpikeImportance::new(0, 0, 5.0, 0.5, 0.3),
            SpikeImportance::new(1, 0, 15.0, 0.8, 0.5),
            SpikeImportance::new(2, 0, 25.0, 0.6, 0.4),
            SpikeImportance::new(3, 0, 27.0, 0.7, 0.45),
        ];

        let curve = ExplanationVisualizer::temporal_importance_curve(&importances, 10.0, 50.0);

        assert_eq!(curve.len(), 5); // 50ms / 10ms = 5 bins

        // Check bin values (with tolerance for floating-point precision)
        assert!((curve[0].1 - 0.5).abs() < 1e-10); // Bin 0 has one spike
        assert!((curve[1].1 - 0.8).abs() < 1e-10); // Bin 1 has one spike
        assert!((curve[2].1 - 1.3).abs() < 1e-10); // Bin 2 has two spikes
    }

    #[test]
    fn test_neuron_ranking() {
        let importances = vec![
            SpikeImportance::new(0, 0, 10.0, 0.5, 0.3),
            SpikeImportance::new(0, 0, 20.0, 0.3, 0.2),
            SpikeImportance::new(1, 0, 15.0, 0.9, 0.7),
            SpikeImportance::new(2, 1, 25.0, 0.6, 0.4),
        ];

        let ranking = ExplanationVisualizer::neuron_ranking(&importances);

        assert_eq!(ranking.len(), 3); // 3 unique neurons

        // Check highest ranked neuron
        assert_eq!(ranking[0].1, 1); // Neuron 1 has highest total importance (0.9)
        assert_eq!(ranking[1].1, 0); // Neuron 0 has second highest (0.5 + 0.3 = 0.8)
    }

    #[test]
    fn test_layer_contributions() {
        let importances = vec![
            SpikeImportance::new(0, 0, 10.0, 0.5, 0.3),
            SpikeImportance::new(1, 0, 15.0, 0.8, 0.5),
            SpikeImportance::new(2, 1, 20.0, 0.6, 0.4),
            SpikeImportance::new(3, 1, 25.0, 0.7, 0.5),
        ];

        let contributions = ExplanationVisualizer::layer_contributions(&importances);

        assert_eq!(contributions.len(), 2); // 2 layers

        // Check layer 0
        assert_eq!(contributions[0].0, 0);
        assert!((contributions[0].1 - 1.3).abs() < 1e-10); // 0.5 + 0.8
        assert_eq!(contributions[0].2, 2); // 2 spikes

        // Check layer 1
        assert_eq!(contributions[1].0, 1);
        assert!((contributions[1].1 - 1.3).abs() < 1e-10); // 0.6 + 0.7
        assert_eq!(contributions[1].2, 2); // 2 spikes
    }

    #[test]
    fn test_contribution_breakdown() {
        let names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];
        let values = vec![0.5, -0.3, 0.2];
        let attribution = FeatureAttribution::new(names, values, 0.0, 0.4);

        let (pos_sum, neg_sum, pos_features, neg_features) =
            ExplanationVisualizer::contribution_breakdown(&attribution);

        assert_eq!(pos_sum, 0.7);
        assert_eq!(neg_sum, 0.3);
        assert_eq!(pos_features.len(), 2);
        assert_eq!(neg_features.len(), 1);
    }

    #[test]
    fn test_export_json() {
        let names = vec!["f1".to_string(), "f2".to_string()];
        let values = vec![0.5, 0.3];
        let attribution = FeatureAttribution::new(names, values, 0.0, 0.8);

        let json = export_explanation_json(None, Some(&attribution), None);

        assert!(json.contains("attribution"));
        assert!(json.contains("features"));
        assert!(json.contains("values"));
    }
    /// Whatever the inputs, the exporter must emit parseable JSON.
    ///
    /// Regression: the braces were pushed into the same vector as the members
    /// and joined with them, so every output carried a comma straight after `{`
    /// and before `}` -- `{, }` for the empty case. Nothing this function
    /// produced had ever parsed.
    #[test]
    fn test_export_explanation_json_is_valid_json() {
        // Empty: the degenerate case that produced `{, }`.
        let empty = export_explanation_json(None, None, None);
        assert_eq!(empty, "{}", "empty export should be an empty object");
        serde_json::from_str::<serde_json::Value>(&empty).expect("empty export must parse");

        // With importance records, including non-finite values, which have no
        // JSON representation and must become null rather than bare NaN.
        let importance = vec![
            SpikeImportance {
                neuron_id: 3,
                layer: 1,
                spike_time: 12.5,
                importance_score: 0.75,
                contribution_to_output: -0.25,
            },
            SpikeImportance {
                neuron_id: 4,
                layer: 1,
                spike_time: 20.0,
                importance_score: f64::NAN,
                contribution_to_output: f64::INFINITY,
            },
        ];

        let json = export_explanation_json(None, None, Some(&importance));
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("export with importance must parse");

        let records = parsed
            .get("importance")
            .and_then(|v| v.as_array())
            .expect("importance array present");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["neuron"], 3);
        assert!(
            records[1]["importance"].is_null(),
            "NaN must serialise as null, got {}",
            records[1]["importance"]
        );
    }

}
