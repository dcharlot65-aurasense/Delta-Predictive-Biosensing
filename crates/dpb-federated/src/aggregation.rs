//! Aggregation strategies for federated learning.

use crate::{
    config::AggregationStrategy,
    model::{ModelUpdate, ModelWeights, ParameterDelta, Tensor},
    FederatedError, Result,
};
use std::collections::HashMap;

/// Trait for aggregation algorithms.
pub trait Aggregator: Send + Sync {
    /// Aggregate model updates into a single update.
    fn aggregate(&self, updates: &[ModelUpdate]) -> Result<ParameterDelta>;

    /// Get the aggregation strategy type.
    fn strategy(&self) -> AggregationStrategy;
}

/// Federated Averaging (FedAvg) aggregator.
///
/// Computes the weighted average of client updates based on
/// the number of local samples.
pub struct FedAvg;

impl FedAvg {
    /// Create a new FedAvg aggregator.
    pub fn new() -> Self {
        Self
    }
}

impl Default for FedAvg {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for FedAvg {
    fn aggregate(&self, updates: &[ModelUpdate]) -> Result<ParameterDelta> {
        if updates.is_empty() {
            return Err(FederatedError::InsufficientClients {
                required: 1,
                available: 0,
            });
        }

        // Calculate total samples
        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();

        if total_samples == 0 {
            return Err(FederatedError::InvalidWeights(
                "Total samples is zero".into(),
            ));
        }

        // Get all parameter names from first update
        let param_names: Vec<String> = updates[0].delta.changes.keys().cloned().collect();

        let mut aggregated_changes = HashMap::new();

        for name in &param_names {
            // Collect all tensors for this parameter
            let tensors: Vec<(&Tensor, f32)> = updates
                .iter()
                .filter_map(|u| {
                    u.delta.changes.get(name).map(|t| {
                        (t, u.num_samples as f32 / total_samples as f32)
                    })
                })
                .collect();

            if tensors.is_empty() {
                continue;
            }

            let shape = tensors[0].0.shape.clone();
            let size = tensors[0].0.data.len();

            // Weighted average
            let mut aggregated_data = vec![0.0f32; size];
            for (tensor, weight) in &tensors {
                if tensor.data.len() != size {
                    return Err(FederatedError::DimensionMismatch {
                        expected: size,
                        actual: tensor.data.len(),
                    });
                }
                for (i, v) in tensor.data.iter().enumerate() {
                    aggregated_data[i] += v * weight;
                }
            }

            aggregated_changes.insert(
                name.clone(),
                Tensor {
                    data: aggregated_data,
                    shape,
                },
            );
        }

        Ok(ParameterDelta::new(
            updates[0].delta.base_version,
            aggregated_changes,
        ))
    }

    fn strategy(&self) -> AggregationStrategy {
        AggregationStrategy::FedAvg
    }
}

/// Simple weighted average aggregator.
pub struct WeightedAvg {
    /// Custom weights per client (if None, uses sample counts).
    weights: Option<HashMap<String, f32>>,
}

impl WeightedAvg {
    /// Create a new weighted average aggregator.
    pub fn new() -> Self {
        Self { weights: None }
    }

    /// Create with custom client weights.
    pub fn with_weights(weights: HashMap<String, f32>) -> Self {
        Self {
            weights: Some(weights),
        }
    }
}

impl Default for WeightedAvg {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for WeightedAvg {
    fn aggregate(&self, updates: &[ModelUpdate]) -> Result<ParameterDelta> {
        if updates.is_empty() {
            return Err(FederatedError::InsufficientClients {
                required: 1,
                available: 0,
            });
        }

        // Determine weights
        let client_weights: Vec<f32> = if let Some(ref custom_weights) = self.weights {
            updates
                .iter()
                .map(|u| *custom_weights.get(&u.client_id).unwrap_or(&1.0))
                .collect()
        } else {
            let total: usize = updates.iter().map(|u| u.num_samples).sum();
            updates
                .iter()
                .map(|u| u.num_samples as f32 / total as f32)
                .collect()
        };

        // Normalize weights
        let weight_sum: f32 = client_weights.iter().sum();
        let normalized_weights: Vec<f32> = client_weights.iter().map(|w| w / weight_sum).collect();

        // Get parameter names
        let param_names: Vec<String> = updates[0].delta.changes.keys().cloned().collect();
        let mut aggregated_changes = HashMap::new();

        for name in &param_names {
            let tensors: Vec<(&Tensor, f32)> = updates
                .iter()
                .zip(&normalized_weights)
                .filter_map(|(u, w)| u.delta.changes.get(name).map(|t| (t, *w)))
                .collect();

            if tensors.is_empty() {
                continue;
            }

            let shape = tensors[0].0.shape.clone();
            let size = tensors[0].0.data.len();

            let mut aggregated_data = vec![0.0f32; size];
            for (tensor, weight) in &tensors {
                for (i, v) in tensor.data.iter().enumerate() {
                    aggregated_data[i] += v * weight;
                }
            }

            aggregated_changes.insert(name.clone(), Tensor::new(aggregated_data, shape));
        }

        Ok(ParameterDelta::new(
            updates[0].delta.base_version,
            aggregated_changes,
        ))
    }

    fn strategy(&self) -> AggregationStrategy {
        AggregationStrategy::WeightedAvg
    }
}

/// Median aggregator (robust to Byzantine clients).
pub struct MedianAggregator;

impl MedianAggregator {
    /// Create a new median aggregator.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MedianAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl Aggregator for MedianAggregator {
    fn aggregate(&self, updates: &[ModelUpdate]) -> Result<ParameterDelta> {
        if updates.is_empty() {
            return Err(FederatedError::InsufficientClients {
                required: 1,
                available: 0,
            });
        }

        let param_names: Vec<String> = updates[0].delta.changes.keys().cloned().collect();
        let mut aggregated_changes = HashMap::new();

        for name in &param_names {
            let tensors: Vec<&Tensor> = updates
                .iter()
                .filter_map(|u| u.delta.changes.get(name))
                .collect();

            if tensors.is_empty() {
                continue;
            }

            let shape = tensors[0].shape.clone();
            let size = tensors[0].data.len();

            let mut aggregated_data = vec![0.0f32; size];

            // Compute coordinate-wise median
            for i in 0..size {
                let mut values: Vec<f32> = tensors.iter().map(|t| t.data[i]).collect();
                values.sort_by(|a, b| a.total_cmp(b));
                aggregated_data[i] = median(&values);
            }

            aggregated_changes.insert(name.clone(), Tensor::new(aggregated_data, shape));
        }

        Ok(ParameterDelta::new(
            updates[0].delta.base_version,
            aggregated_changes,
        ))
    }

    fn strategy(&self) -> AggregationStrategy {
        AggregationStrategy::Median
    }
}

/// Trimmed mean aggregator (removes extreme values).
pub struct TrimmedMeanAggregator {
    /// Fraction to trim from each end (0.0-0.5).
    trim_fraction: f32,
}

impl TrimmedMeanAggregator {
    /// Create a new trimmed mean aggregator.
    pub fn new(trim_fraction: f32) -> Self {
        Self {
            trim_fraction: trim_fraction.clamp(0.0, 0.49),
        }
    }
}

impl Default for TrimmedMeanAggregator {
    fn default() -> Self {
        Self::new(0.1) // Trim 10% from each end by default
    }
}

impl Aggregator for TrimmedMeanAggregator {
    fn aggregate(&self, updates: &[ModelUpdate]) -> Result<ParameterDelta> {
        if updates.is_empty() {
            return Err(FederatedError::InsufficientClients {
                required: 1,
                available: 0,
            });
        }

        let param_names: Vec<String> = updates[0].delta.changes.keys().cloned().collect();
        let mut aggregated_changes = HashMap::new();
        let n = updates.len();
        let trim_count = (n as f32 * self.trim_fraction) as usize;

        for name in &param_names {
            let tensors: Vec<&Tensor> = updates
                .iter()
                .filter_map(|u| u.delta.changes.get(name))
                .collect();

            if tensors.is_empty() {
                continue;
            }

            let shape = tensors[0].shape.clone();
            let size = tensors[0].data.len();

            let mut aggregated_data = vec![0.0f32; size];

            for i in 0..size {
                let mut values: Vec<f32> = tensors.iter().map(|t| t.data[i]).collect();
                values.sort_by(|a, b| a.total_cmp(b));

                // Trim from both ends
                let trimmed = if trim_count > 0 && values.len() > 2 * trim_count {
                    &values[trim_count..values.len() - trim_count]
                } else {
                    &values[..]
                };

                // Compute mean of remaining
                aggregated_data[i] = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
            }

            aggregated_changes.insert(name.clone(), Tensor::new(aggregated_data, shape));
        }

        Ok(ParameterDelta::new(
            updates[0].delta.base_version,
            aggregated_changes,
        ))
    }

    fn strategy(&self) -> AggregationStrategy {
        AggregationStrategy::TrimmedMean
    }
}

/// Create an aggregator from strategy enum.
pub fn create_aggregator(strategy: AggregationStrategy) -> Box<dyn Aggregator> {
    match strategy {
        AggregationStrategy::FedAvg => Box::new(FedAvg::new()),
        AggregationStrategy::WeightedAvg => Box::new(WeightedAvg::new()),
        AggregationStrategy::Median | AggregationStrategy::CoordinateMedian => {
            Box::new(MedianAggregator::new())
        }
        AggregationStrategy::TrimmedMean => Box::new(TrimmedMeanAggregator::default()),
    }
}

/// Compute median of a sorted slice.
fn median(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_update(client_id: &str, values: Vec<f32>, num_samples: usize) -> ModelUpdate {
        let delta = ParameterDelta::new(
            0,
            [("w".to_string(), Tensor::new(values, vec![4]))]
                .into_iter()
                .collect(),
        );
        ModelUpdate::new(client_id, 0, delta, num_samples)
    }

    #[test]
    fn test_fedavg_simple() {
        let updates = vec![
            create_test_update("a", vec![1.0, 2.0, 3.0, 4.0], 100),
            create_test_update("b", vec![2.0, 4.0, 6.0, 8.0], 100),
        ];

        let aggregator = FedAvg::new();
        let result = aggregator.aggregate(&updates).unwrap();

        // Should be simple average since samples are equal
        let expected = vec![1.5, 3.0, 4.5, 6.0];
        assert_eq!(result.changes["w"].data, expected);
    }

    #[test]
    fn test_fedavg_weighted() {
        let updates = vec![
            create_test_update("a", vec![1.0, 1.0, 1.0, 1.0], 100),
            create_test_update("b", vec![3.0, 3.0, 3.0, 3.0], 300),
        ];

        let aggregator = FedAvg::new();
        let result = aggregator.aggregate(&updates).unwrap();

        // Weight: a=0.25, b=0.75 -> 0.25*1 + 0.75*3 = 2.5
        for v in &result.changes["w"].data {
            assert!((*v - 2.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fedavg_empty() {
        let aggregator = FedAvg::new();
        let result = aggregator.aggregate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_median_aggregator() {
        let updates = vec![
            create_test_update("a", vec![1.0, 2.0, 3.0, 4.0], 100),
            create_test_update("b", vec![10.0, 20.0, 30.0, 40.0], 100), // Outlier
            create_test_update("c", vec![2.0, 3.0, 4.0, 5.0], 100),
        ];

        let aggregator = MedianAggregator::new();
        let result = aggregator.aggregate(&updates).unwrap();

        // Median of [1, 10, 2], [2, 20, 3], [3, 30, 4], [4, 40, 5]
        assert_eq!(result.changes["w"].data, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_trimmed_mean() {
        let updates = vec![
            create_test_update("a", vec![1.0], 100),
            create_test_update("b", vec![2.0], 100),
            create_test_update("c", vec![3.0], 100),
            create_test_update("d", vec![100.0], 100), // Outlier
            create_test_update("e", vec![4.0], 100),
        ];

        let aggregator = TrimmedMeanAggregator::new(0.2); // Trim 20% = 1 from each end
        let result = aggregator.aggregate(&updates).unwrap();

        // After sorting: [1, 2, 3, 4, 100], trim 1 from each: [2, 3, 4], mean = 3
        assert!((result.changes["w"].data[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_create_aggregator() {
        let strategies = [
            AggregationStrategy::FedAvg,
            AggregationStrategy::WeightedAvg,
            AggregationStrategy::Median,
            AggregationStrategy::TrimmedMean,
        ];

        for strategy in &strategies {
            let aggregator = create_aggregator(*strategy);
            assert_eq!(aggregator.strategy(), *strategy);
        }
    }

    #[test]
    fn test_median_helper() {
        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[5.0]), 5.0);
        assert_eq!(median(&[]), 0.0);
    }
}
