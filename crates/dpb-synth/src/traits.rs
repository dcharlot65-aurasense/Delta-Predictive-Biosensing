//! Core traits for synthetic data generation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core trait for all synthetic generators
pub trait SyntheticGenerator {
    /// Output type for generated signals
    type Output;

    /// Ground truth type for this generator
    type GroundTruth;

    /// Parameters for this generator
    type Parameters: Clone;

    /// Generate synthetic data with ground truth
    ///
    /// # Arguments
    /// * `params` - Generator parameters
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    /// Tuple of (generated signal, ground truth)
    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>>;

    /// Get default parameters
    fn default_params() -> Self::Parameters;

    /// Validate parameters
    fn validate_params(params: &Self::Parameters) -> crate::Result<()>;
}

/// Generated data with ground truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedData<T, G> {
    /// The generated signal/data
    pub signal: T,

    /// Ground truth information
    pub ground_truth: G,

    /// Sampling rate (Hz)
    pub sampling_rate: f64,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl<T, G> GeneratedData<T, G> {
    pub fn new(signal: T, ground_truth: G, sampling_rate: f64) -> Self {
        Self {
            signal,
            ground_truth,
            sampling_rate,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Ground truth marker trait
pub trait GroundTruth: Clone + Serialize {}

/// Parameter space for sweeps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpace {
    /// Parameter name -> values to sweep
    pub parameters: HashMap<String, Vec<f64>>,
}

impl ParameterSpace {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    pub fn add_parameter(mut self, name: impl Into<String>, values: Vec<f64>) -> Self {
        self.parameters.insert(name.into(), values);
        self
    }

    /// Generate all parameter combinations
    pub fn combinations(&self) -> Vec<HashMap<String, f64>> {
        if self.parameters.is_empty() {
            return vec![HashMap::new()];
        }

        let keys: Vec<_> = self.parameters.keys().cloned().collect();
        let values: Vec<_> = keys.iter().map(|k| &self.parameters[k]).collect();

        Self::cartesian_product(&keys, &values)
    }

    fn cartesian_product(keys: &[String], values: &[&Vec<f64>]) -> Vec<HashMap<String, f64>> {
        if values.is_empty() {
            return vec![HashMap::new()];
        }

        let mut result = Vec::new();
        let first = values[0];
        let rest = &values[1..];
        let rest_keys = &keys[1..];

        for &val in first {
            if rest.is_empty() {
                let mut map = HashMap::new();
                map.insert(keys[0].clone(), val);
                result.push(map);
            } else {
                for mut combo in Self::cartesian_product(rest_keys, rest) {
                    combo.insert(keys[0].clone(), val);
                    result.push(combo);
                }
            }
        }

        result
    }
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Time series ground truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesGroundTruth {
    /// True parameter values
    pub parameters: HashMap<String, f64>,

    /// Event times (e.g., R-peaks, SCR onsets)
    pub events: Vec<Event>,

    /// Segment labels
    pub segments: Vec<Segment>,
}

impl GroundTruth for TimeSeriesGroundTruth {}

/// Event marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Time of event (seconds)
    pub time: f64,

    /// Event type
    pub event_type: String,

    /// Event amplitude/magnitude
    pub amplitude: Option<f64>,

    /// Additional attributes
    pub attributes: HashMap<String, f64>,
}

/// Segment label
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Start time (seconds)
    pub start: f64,

    /// End time (seconds)
    pub end: f64,

    /// Segment label
    pub label: String,
}

/// Spatial data ground truth (e.g., pose keypoints)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialGroundTruth {
    /// True keypoint locations [frames x keypoints x dims]
    pub keypoints: Vec<Vec<[f64; 3]>>,

    /// True joint angles
    pub joint_angles: HashMap<String, Vec<f64>>,

    /// Gait cycle phases
    pub gait_phases: Vec<GaitPhase>,
}

impl GroundTruth for SpatialGroundTruth {}

/// Gait phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitPhase {
    pub frame: usize,
    pub phase: f64,  // 0.0 = heel strike, 1.0 = next heel strike
    pub phase_name: String,  // "stance", "swing", etc.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_space_combinations() {
        let space = ParameterSpace::new()
            .add_parameter("a", vec![1.0, 2.0])
            .add_parameter("b", vec![3.0, 4.0]);

        let combos = space.combinations();
        assert_eq!(combos.len(), 4);
    }

    #[test]
    fn test_generated_data_metadata() {
        let data = GeneratedData::new(
            vec![1.0, 2.0, 3.0],
            TimeSeriesGroundTruth {
                parameters: HashMap::new(),
                events: Vec::new(),
                segments: Vec::new(),
            },
            100.0,
        ).with_metadata("test".to_string(), "value".to_string());

        assert_eq!(data.metadata.get("test").unwrap(), "value");
    }
}
