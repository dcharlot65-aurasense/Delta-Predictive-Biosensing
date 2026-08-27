//! Python bindings for evaluation metrics

use dpb_core::validation::roc::RocAnalyzer;

use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::PyClassInitializer;
use std::collections::HashMap;

/// Base metric class
#[pyclass(name = "Metric", subclass)]
pub struct PyMetric {
    name: String,
}

#[pymethods]
impl PyMetric {
    #[new]
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }

    /// Compute metric value
    ///
    /// Args:
    ///     predictions: Model predictions
    ///     targets: Ground truth targets
    ///
    /// Returns:
    ///     float: Metric value
    fn compute(&self, _predictions: Py<PyAny>, _targets: Py<PyAny>) -> PyResult<f64> {
        // Base implementation
        Ok(0.0)
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    fn __repr__(&self) -> String {
        format!("Metric(name='{}')", self.name)
    }
}

/// Accuracy metric
///
/// Computes classification accuracy.
///
/// Example:
///     >>> metric = Accuracy()
///     >>> value = metric.compute(predictions, targets)
#[pyclass(name = "Accuracy", extends=PyMetric)]
pub struct PyAccuracy;

#[pymethods]
impl PyAccuracy {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMetric {
                name: "Accuracy".to_string(),
            })
            .add_subclass(Self)
}

    fn compute(&self, predictions: PyReadonlyArray1<i32>, targets: PyReadonlyArray1<i32>) -> PyResult<f64> {
        let preds = predictions.as_slice()?;
        let targs = targets.as_slice()?;

        if preds.len() != targs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Predictions and targets must have same length",
            ));
        }

        let correct = preds
            .iter()
            .zip(targs.iter())
            .filter(|(p, t)| p == t)
            .count();

        Ok(correct as f64 / preds.len() as f64)
    }
}

/// Precision metric
///
/// Computes classification precision.
///
/// Example:
///     >>> metric = Precision()
///     >>> value = metric.compute(predictions, targets)
#[pyclass(name = "Precision", extends=PyMetric)]
pub struct PyPrecision;

#[pymethods]
impl PyPrecision {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMetric {
                name: "Precision".to_string(),
            })
            .add_subclass(Self)
}

    fn compute(&self, predictions: PyReadonlyArray1<i32>, targets: PyReadonlyArray1<i32>) -> PyResult<f64> {
        let preds = predictions.as_slice()?;
        let targs = targets.as_slice()?;

        if preds.len() != targs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Predictions and targets must have same length",
            ));
        }

        let mut true_positives = 0;
        let mut false_positives = 0;

        for (p, t) in preds.iter().zip(targs.iter()) {
            if *p == 1 {
                if *t == 1 {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            }
        }

        if true_positives + false_positives == 0 {
            Ok(0.0)
        } else {
            Ok(true_positives as f64 / (true_positives + false_positives) as f64)
        }
    }
}

/// Recall metric
///
/// Computes classification recall (sensitivity).
///
/// Example:
///     >>> metric = Recall()
///     >>> value = metric.compute(predictions, targets)
#[pyclass(name = "Recall", extends=PyMetric)]
pub struct PyRecall;

#[pymethods]
impl PyRecall {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMetric {
                name: "Recall".to_string(),
            })
            .add_subclass(Self)
}

    fn compute(&self, predictions: PyReadonlyArray1<i32>, targets: PyReadonlyArray1<i32>) -> PyResult<f64> {
        let preds = predictions.as_slice()?;
        let targs = targets.as_slice()?;

        if preds.len() != targs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Predictions and targets must have same length",
            ));
        }

        let mut true_positives = 0;
        let mut false_negatives = 0;

        for (p, t) in preds.iter().zip(targs.iter()) {
            if *t == 1 {
                if *p == 1 {
                    true_positives += 1;
                } else {
                    false_negatives += 1;
                }
            }
        }

        if true_positives + false_negatives == 0 {
            Ok(0.0)
        } else {
            Ok(true_positives as f64 / (true_positives + false_negatives) as f64)
        }
    }
}

/// F1 Score metric
///
/// Computes F1 score (harmonic mean of precision and recall).
///
/// Example:
///     >>> metric = F1Score()
///     >>> value = metric.compute(predictions, targets)
#[pyclass(name = "F1Score", extends=PyMetric)]
pub struct PyF1Score;

#[pymethods]
impl PyF1Score {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMetric {
                name: "F1Score".to_string(),
            })
            .add_subclass(Self)
}

    fn compute(&self, predictions: PyReadonlyArray1<i32>, targets: PyReadonlyArray1<i32>) -> PyResult<f64> {
        // Compute precision and recall
        let precision_metric = PyPrecision;
        let recall_metric = PyRecall;

        let precision = precision_metric.compute(predictions.clone(), targets.clone())?;
        let recall = recall_metric.compute(predictions, targets)?;

        if precision + recall == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * precision * recall / (precision + recall))
        }
    }
}

/// Mean Squared Error (MSE) metric
///
/// Computes mean squared error.
///
/// Example:
///     >>> metric = MSE()
///     >>> value = metric.compute(predictions, targets)
#[pyclass(name = "MSE", extends=PyMetric)]
pub struct PyMSE;

#[pymethods]
impl PyMSE {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMetric {
                name: "MSE".to_string(),
            })
            .add_subclass(Self)
}

    fn compute(&self, predictions: PyReadonlyArray1<f32>, targets: PyReadonlyArray1<f32>) -> PyResult<f64> {
        let preds = predictions.as_slice()?;
        let targs = targets.as_slice()?;

        if preds.len() != targs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Predictions and targets must have same length",
            ));
        }

        let mse: f64 = preds
            .iter()
            .zip(targs.iter())
            .map(|(p, t)| {
                let diff = *p as f64 - *t as f64;
                diff * diff
            })
            .sum::<f64>()
            / preds.len() as f64;

        Ok(mse)
    }
}

/// Root Mean Squared Error (RMSE) metric
///
/// Computes root mean squared error.
///
/// Example:
///     >>> metric = RMSE()
///     >>> value = metric.compute(predictions, targets)
#[pyclass(name = "RMSE", extends=PyMetric)]
pub struct PyRMSE;

#[pymethods]
impl PyRMSE {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMetric {
                name: "RMSE".to_string(),
            })
            .add_subclass(Self)
}

    fn compute(&self, predictions: PyReadonlyArray1<f32>, targets: PyReadonlyArray1<f32>) -> PyResult<f64> {
        let mse_metric = PyMSE;
        let mse = mse_metric.compute(predictions, targets)?;
        Ok(mse.sqrt())
    }
}

/// Mean Absolute Error (MAE) metric
///
/// Computes mean absolute error.
///
/// Example:
///     >>> metric = MAE()
///     >>> value = metric.compute(predictions, targets)
#[pyclass(name = "MAE", extends=PyMetric)]
pub struct PyMAE;

#[pymethods]
impl PyMAE {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMetric {
                name: "MAE".to_string(),
            })
            .add_subclass(Self)
}

    fn compute(&self, predictions: PyReadonlyArray1<f32>, targets: PyReadonlyArray1<f32>) -> PyResult<f64> {
        let preds = predictions.as_slice()?;
        let targs = targets.as_slice()?;

        if preds.len() != targs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Predictions and targets must have same length",
            ));
        }

        let mae: f64 = preds
            .iter()
            .zip(targs.iter())
            .map(|(p, t)| (*p as f64 - *t as f64).abs())
            .sum::<f64>()
            / preds.len() as f64;

        Ok(mae)
    }
}

/// Confusion Matrix
///
/// Computes confusion matrix for classification.
///
/// Args:
///     num_classes (int): Number of classes
///
/// Example:
///     >>> cm = ConfusionMatrix(num_classes=3)
///     >>> matrix = cm.compute(predictions, targets)
#[pyclass(name = "ConfusionMatrix")]
pub struct PyConfusionMatrix {
    num_classes: usize,
}

#[pymethods]
impl PyConfusionMatrix {
    #[new]
    fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }

    /// Compute confusion matrix
    fn compute(
        &self,
        predictions: PyReadonlyArray1<i32>,
        targets: PyReadonlyArray1<i32>,
        py: Python,
    ) -> PyResult<Py<PyArray2<i32>>> {
        let preds = predictions.as_slice()?;
        let targs = targets.as_slice()?;

        if preds.len() != targs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Predictions and targets must have same length",
            ));
        }

        // Initialize confusion matrix
        let mut matrix = vec![vec![0i32; self.num_classes]; self.num_classes];

        // Fill confusion matrix
        for (p, t) in preds.iter().zip(targs.iter()) {
            let pred_class = *p as usize;
            let true_class = *t as usize;

            if pred_class < self.num_classes && true_class < self.num_classes {
                matrix[true_class][pred_class] += 1;
            }
        }

        // Convert to numpy array
        let flat: Vec<i32> = matrix.into_iter().flatten().collect();
        let array = PyArray2::from_vec2(
            py,
            &(0..self.num_classes)
                .map(|i| {
                    flat[i * self.num_classes..(i + 1) * self.num_classes].to_vec()
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(array.into())
    }

    fn __repr__(&self) -> String {
        format!("ConfusionMatrix(num_classes={})", self.num_classes)
    }
}

/// ROC Curve computation
///
/// Computes ROC curve points.
///
/// Example:
///     >>> roc = ROCCurve()
///     >>> fpr, tpr, thresholds = roc.compute(predictions, targets)
#[pyclass(name = "ROCCurve")]
pub struct PyROCCurve;

#[pymethods]
impl PyROCCurve {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Compute ROC curve
    fn compute(
        &self,
        predictions: PyReadonlyArray1<f32>,
        targets: PyReadonlyArray1<i32>,
        py: Python,
    ) -> PyResult<(Py<numpy::PyArray1<f64>>, Py<numpy::PyArray1<f64>>, Py<numpy::PyArray1<f64>>)> {
        // Computed by `dpb_core::validation::roc`, which emits one point per
        // DISTINCT threshold so the curve does not depend on the order tied
        // scores happen to arrive in.
        let scores: Vec<f64> = predictions.as_slice()?.iter().map(|&v| v as f64).collect();
        let labels: Vec<bool> = targets.as_slice()?.iter().map(|&v| v != 0).collect();

        let curve = RocAnalyzer::default().analyze(&scores, &labels);

        let fpr: Vec<f64> = curve.points.iter().map(|p| p.false_positive_rate).collect();
        let tpr: Vec<f64> = curve.points.iter().map(|p| p.true_positive_rate).collect();
        let thresholds: Vec<f64> = curve.points.iter().map(|p| p.threshold).collect();

        Ok((
            numpy::PyArray1::from_vec(py, fpr).into(),
            numpy::PyArray1::from_vec(py, tpr).into(),
            numpy::PyArray1::from_vec(py, thresholds).into(),
        ))
    }
}

/// AUC (Area Under Curve) metric
///
/// Computes area under ROC curve.
///
/// Example:
///     >>> metric = AUC()
///     >>> value = metric.compute(predictions, targets)
#[pyclass(name = "AUC", extends=PyMetric)]
pub struct PyAUC;

#[pymethods]
impl PyAUC {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMetric {
                name: "AUC".to_string(),
            })
            .add_subclass(Self)
}

    fn compute(&self, predictions: PyReadonlyArray1<f32>, targets: PyReadonlyArray1<i32>) -> PyResult<f64> {
        // Previously returned a hardcoded 0.5 for any input, which reads as a
        // perfectly uninformative classifier and is indistinguishable from a
        // real result.
        let scores: Vec<f64> = predictions.as_slice()?.iter().map(|&v| v as f64).collect();
        let labels: Vec<bool> = targets.as_slice()?.iter().map(|&v| v != 0).collect();

        Ok(RocAnalyzer::default().analyze(&scores, &labels).auc)
    }
}

/// Spike distance metric
///
/// Computes temporal distance between spike trains.
///
/// Example:
///     >>> metric = SpikeDistance()
///     >>> value = metric.compute(spike_train1, spike_train2)
#[pyclass(name = "SpikeDistance", extends=PyMetric)]
pub struct PySpikeDistance {
    tau: f64,
}

#[pymethods]
impl PySpikeDistance {
    #[new]
    #[pyo3(signature = (tau=0.01))]
    fn new(tau: f64) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyMetric {
                name: "SpikeDistance".to_string(),
            })
            .add_subclass(Self { tau })
}

    fn compute(&self, spikes1: PyReadonlyArray1<f64>, spikes2: PyReadonlyArray1<f64>) -> PyResult<f64> {
        // van Rossum distance, computed here because the Rust crates do not
        // implement a spike metric to delegate to.
        //
        // Each train is convolved with a decaying exponential of time constant
        // `tau` and the L2 distance taken between the results; equivalently,
        // summing the pairwise kernel terms in closed form, which avoids
        // discretising the trains onto a grid.
        //
        // This previously returned 0.0 -- "the trains are identical" -- for
        // every input.
        let a = spikes1.as_slice()?;
        let b = spikes2.as_slice()?;
        let tau = self.tau.max(f64::EPSILON);

        let kernel_sum = |xs: &[f64], ys: &[f64]| -> f64 {
            xs.iter()
                .flat_map(|x| ys.iter().map(move |y| (-(x - y).abs() / tau).exp()))
                .sum::<f64>()
        };

        // |f - g|^2 = <f,f> - 2<f,g> + <g,g>
        let distance_squared =
            kernel_sum(a, a) - 2.0 * kernel_sum(a, b) + kernel_sum(b, b);

        Ok((distance_squared.max(0.0) / (2.0 * tau)).sqrt())
    }
}

/// Metric collection for computing multiple metrics at once
///
/// Args:
///     metrics (list): List of metric instances
///
/// Example:
///     >>> collection = MetricCollection([Accuracy(), Precision(), Recall()])
///     >>> results = collection.compute(predictions, targets)
#[pyclass(name = "MetricCollection")]
pub struct PyMetricCollection {
    metrics: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyMetricCollection {
    #[new]
    fn new(metrics: Vec<Py<PyAny>>) -> Self {
        Self { metrics }
    }

    /// Compute all metrics
    fn compute(
        &self,
        predictions: Py<PyAny>,
        targets: Py<PyAny>,
        py: Python,
    ) -> PyResult<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for metric in &self.metrics {
            let name: String = metric.getattr(py, "name")?.extract(py)?;
            let value: f64 = metric
                .call_method1(py, "compute", (predictions.clone_ref(py), targets.clone_ref(py)))?
                .extract(py)?;
            results.insert(name, value);
        }

        Ok(results)
    }

    fn __len__(&self) -> usize {
        self.metrics.len()
    }
}

/// Register metrics module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMetric>()?;
    m.add_class::<PyAccuracy>()?;
    m.add_class::<PyPrecision>()?;
    m.add_class::<PyRecall>()?;
    m.add_class::<PyF1Score>()?;
    m.add_class::<PyMSE>()?;
    m.add_class::<PyRMSE>()?;
    m.add_class::<PyMAE>()?;
    m.add_class::<PyConfusionMatrix>()?;
    m.add_class::<PyROCCurve>()?;
    m.add_class::<PyAUC>()?;
    m.add_class::<PySpikeDistance>()?;
    m.add_class::<PyMetricCollection>()?;
    Ok(())
}
