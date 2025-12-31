//! Utilities for NumPy array conversion and integration

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Convert PyArray1 to ndarray Array1
pub fn pyarray1_to_array1<T: numpy::Element>(
    py_array: PyReadonlyArray1<T>,
) -> Array1<T>
where
    T: Clone,
{
    py_array.as_array().to_owned()
}

/// Convert PyArray2 to ndarray Array2
pub fn pyarray2_to_array2<T: numpy::Element>(
    py_array: PyReadonlyArray2<T>,
) -> Array2<T>
where
    T: Clone,
{
    py_array.as_array().to_owned()
}

/// Convert ndarray Array1 to PyArray1 (zero-copy when possible)
pub fn array1_to_pyarray1<'py, T: numpy::Element>(
    py: Python<'py>,
    array: Array1<T>,
) -> Bound<'py, PyArray1<T>> {
    PyArray1::from_owned_array(py, array)
}

/// Convert ndarray Array2 to PyArray2 (zero-copy when possible)
pub fn array2_to_pyarray2<'py, T: numpy::Element>(
    py: Python<'py>,
    array: Array2<T>,
) -> Bound<'py, PyArray2<T>> {
    PyArray2::from_owned_array(py, array)
}

/// Convert Vec to PyArray1
pub fn vec_to_pyarray1<'py, T: numpy::Element>(
    py: Python<'py>,
    vec: Vec<T>,
) -> Bound<'py, PyArray1<T>> {
    PyArray1::from_vec(py, vec)
}

/// Convert Vec<Vec<T>> to PyArray2
pub fn vec2_to_pyarray2<'py, T: numpy::Element + Clone>(
    py: Python<'py>,
    data: Vec<Vec<T>>,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    if data.is_empty() {
        return Ok(PyArray2::zeros(py, (0, 0), false));
    }

    let rows = data.len();
    let cols = data[0].len();

    // Validate all rows have same length
    if !data.iter().all(|row| row.len() == cols) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All rows must have the same length",
        ));
    }

    // Flatten data
    let flat: Vec<T> = data.into_iter().flatten().collect();
    let array = Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyArray2::from_owned_array(py, array))
}

/// Create a batch of samples from a list of arrays
pub fn batch_arrays<'py>(
    py: Python<'py>,
    arrays: Vec<Array2<f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if arrays.is_empty() {
        return Ok(PyArray2::zeros(py, (0, 0), false));
    }

    let rows: usize = arrays.iter().map(|a| a.nrows()).sum();
    let cols = arrays[0].ncols();

    // Validate all arrays have same number of columns
    if !arrays.iter().all(|a| a.ncols() == cols) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All arrays must have the same number of columns",
        ));
    }

    // Concatenate arrays
    let mut flat = Vec::with_capacity(rows * cols);
    for array in arrays {
        for row in array.rows() {
            // Use as_slice when contiguous, fall back to iter for non-contiguous views
            if let Some(slice) = row.as_slice() {
                flat.extend_from_slice(slice);
            } else {
                flat.extend(row.iter().copied());
            }
        }
    }

    let batched = Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyArray2::from_owned_array(py, batched))
}

/// Normalize array to zero mean and unit variance
pub fn normalize(mut data: Array2<f32>) -> Array2<f32> {
    let mean = data.mean().unwrap_or(0.0);
    let std = data.std(0.0);

    if std > 1e-8 {
        data.mapv_inplace(|x| (x - mean) / std);
    }

    data
}

/// Standardize array columns independently
pub fn standardize_columns(mut data: Array2<f32>) -> Array2<f32> {
    let ncols = data.ncols();

    for col_idx in 0..ncols {
        let col = data.column(col_idx);
        let mean = col.mean().unwrap_or(0.0);
        let std = col.std(0.0);

        if std > 1e-8 {
            for row_idx in 0..data.nrows() {
                data[[row_idx, col_idx]] = (data[[row_idx, col_idx]] - mean) / std;
            }
        }
    }

    data
}

/// Apply moving average filter
pub fn moving_average(data: ArrayView1<f32>, window_size: usize) -> Array1<f32> {
    let n = data.len();
    let mut result = Array1::zeros(n);

    if window_size == 0 {
        return result;
    }

    for i in 0..n {
        let start = i.saturating_sub(window_size / 2);
        let end = (i + window_size / 2 + 1).min(n);
        let window = data.slice(ndarray::s![start..end]);
        result[i] = window.mean().unwrap_or(0.0);
    }

    result
}

/// Resample data to target length using linear interpolation
pub fn resample(data: ArrayView1<f32>, target_len: usize) -> Array1<f32> {
    let src_len = data.len();
    if src_len == 0 || target_len == 0 {
        return Array1::zeros(target_len);
    }

    if src_len == target_len {
        return data.to_owned();
    }

    let mut result = Array1::zeros(target_len);
    let scale = (src_len - 1) as f32 / (target_len - 1) as f32;

    for i in 0..target_len {
        let pos = i as f32 * scale;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f32;

        if idx + 1 < src_len {
            result[i] = data[idx] * (1.0 - frac) + data[idx + 1] * frac;
        } else {
            result[i] = data[src_len - 1];
        }
    }

    result
}

/// Compute sliding windows from 1D array
pub fn sliding_windows(
    data: ArrayView1<f32>,
    window_size: usize,
    stride: usize,
) -> Array2<f32> {
    let n = data.len();
    if window_size > n || stride == 0 {
        return Array2::zeros((0, window_size));
    }

    let num_windows = ((n - window_size) / stride) + 1;
    let mut windows = Array2::zeros((num_windows, window_size));

    for (win_idx, i) in (0..n - window_size + 1).step_by(stride).enumerate() {
        for j in 0..window_size {
            windows[[win_idx, j]] = data[i + j];
        }
    }

    windows
}

/// Compute root mean square
pub fn rms(data: ArrayView1<f32>) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    (data.mapv(|x| x * x).mean().unwrap_or(0.0)).sqrt()
}

/// Compute peak-to-peak amplitude
pub fn peak_to_peak(data: ArrayView1<f32>) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    max - min
}

/// Find peaks in 1D signal
pub fn find_peaks(data: ArrayView1<f32>, threshold: f32, min_distance: usize) -> Vec<usize> {
    let n = data.len();
    let mut peaks = Vec::new();
    let mut last_peak = 0;

    for i in 1..n - 1 {
        if data[i] > threshold
            && data[i] > data[i - 1]
            && data[i] > data[i + 1]
            && (i - last_peak) >= min_distance
        {
            peaks.push(i);
            last_peak = i;
        }
    }

    peaks
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normalize() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let normalized = normalize(data);
        let mean = normalized.mean().unwrap();
        assert_relative_eq!(mean, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_moving_average() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let smoothed = moving_average(data.view(), 3);
        assert_eq!(smoothed.len(), 5);
    }

    #[test]
    fn test_resample() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let resampled = resample(data.view(), 10);
        assert_eq!(resampled.len(), 10);
    }

    #[test]
    fn test_sliding_windows() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let windows = sliding_windows(data.view(), 3, 1);
        assert_eq!(windows.nrows(), 3);
        assert_eq!(windows.ncols(), 3);
    }

    #[test]
    fn test_find_peaks() {
        let data = Array1::from_vec(vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        let peaks = find_peaks(data.view(), 0.5, 1);
        assert_eq!(peaks, vec![1, 3, 5]);
    }
}
