//! Mathematical utilities and statistical functions.

use crate::error::{DpbError, Result};
use ndarray::{Array1, ArrayView1};

/// Computes the mean of a signal.
pub fn mean(data: ArrayView1<f64>) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.sum() / data.len() as f64
}

/// Computes the variance of a signal.
pub fn variance(data: ArrayView1<f64>, ddof: usize) -> f64 {
    if data.len() <= ddof {
        return 0.0;
    }

    let mean_val = mean(data);
    let sum_squared_diff: f64 = data.iter().map(|x| (x - mean_val).powi(2)).sum();
    sum_squared_diff / (data.len() - ddof) as f64
}

/// Computes the standard deviation of a signal.
pub fn std(data: ArrayView1<f64>, ddof: usize) -> f64 {
    variance(data, ddof).sqrt()
}

/// Computes the median of a signal.
pub fn median(data: ArrayView1<f64>) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Computes a specific percentile.
pub fn percentile(data: ArrayView1<f64>, percentile: f64) -> Result<f64> {
    if !(0.0..=100.0).contains(&percentile) {
        return Err(DpbError::InvalidParameter(
            "Percentile must be between 0 and 100".to_string(),
        ));
    }

    if data.is_empty() {
        return Err(DpbError::InvalidDimensions(
            "Data cannot be empty".to_string(),
        ));
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = (percentile / 100.0) * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;

    let value = sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    Ok(value)
}

/// Computes the interquartile range (IQR).
pub fn iqr(data: ArrayView1<f64>) -> Result<f64> {
    let q25 = percentile(data, 25.0)?;
    let q75 = percentile(data, 75.0)?;
    Ok(q75 - q25)
}

/// Computes the covariance between two signals.
pub fn covariance(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<f64> {
    if x.len() != y.len() {
        return Err(DpbError::InvalidDimensions(
            "Signals must have same length".to_string(),
        ));
    }

    if x.is_empty() {
        return Ok(0.0);
    }

    let mean_x = mean(x);
    let mean_y = mean(y);

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    Ok(cov / x.len() as f64)
}

/// Computes the Pearson correlation coefficient.
pub fn correlation(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<f64> {
    let cov = covariance(x, y)?;
    let std_x = std(x, 0);
    let std_y = std(y, 0);

    if std_x == 0.0 || std_y == 0.0 {
        return Ok(0.0);
    }

    Ok(cov / (std_x * std_y))
}

/// Linear interpolation between two points.
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Linear interpolation for arrays.
pub fn interpolate_linear(x: ArrayView1<f64>, y: ArrayView1<f64>, x_new: ArrayView1<f64>) -> Result<Array1<f64>> {
    if x.len() != y.len() {
        return Err(DpbError::InvalidDimensions(
            "x and y must have same length".to_string(),
        ));
    }

    if x.len() < 2 {
        return Err(DpbError::InvalidDimensions(
            "Need at least 2 points for interpolation".to_string(),
        ));
    }

    let mut result = Array1::zeros(x_new.len());

    for (i, &x_val) in x_new.iter().enumerate() {
        // Find the interval
        let mut idx = 0;
        for j in 0..x.len() - 1 {
            if x_val >= x[j] && x_val <= x[j + 1] {
                idx = j;
                break;
            }
        }

        // Handle extrapolation
        if x_val < x[0] {
            result[i] = y[0];
        } else if x_val > x[x.len() - 1] {
            result[i] = y[y.len() - 1];
        } else {
            let t = (x_val - x[idx]) / (x[idx + 1] - x[idx]);
            result[i] = lerp(y[idx], y[idx + 1], t);
        }
    }

    Ok(result)
}

/// Cubic spline interpolation (simplified).
pub fn interpolate_cubic_spline(x: ArrayView1<f64>, y: ArrayView1<f64>, x_new: ArrayView1<f64>) -> Result<Array1<f64>> {
    // Simplified cubic interpolation using Catmull-Rom splines
    if x.len() != y.len() {
        return Err(DpbError::InvalidDimensions(
            "x and y must have same length".to_string(),
        ));
    }

    if x.len() < 4 {
        // Fall back to linear for insufficient points
        return interpolate_linear(x, y, x_new);
    }

    let mut result = Array1::zeros(x_new.len());

    for (i, &x_val) in x_new.iter().enumerate() {
        // Find segment
        let mut seg = 0;
        for j in 0..x.len() - 1 {
            if x_val >= x[j] && x_val <= x[j + 1] {
                seg = j;
                break;
            }
        }

        // Use linear interpolation at boundaries
        if x_val < x[0] {
            result[i] = y[0];
        } else if x_val > x[x.len() - 1] {
            result[i] = y[y.len() - 1];
        } else if seg == 0 || seg >= x.len() - 2 {
            let t = (x_val - x[seg]) / (x[seg + 1] - x[seg]);
            result[i] = lerp(y[seg], y[seg + 1], t);
        } else {
            // Catmull-Rom interpolation
            let t = (x_val - x[seg]) / (x[seg + 1] - x[seg]);
            result[i] = catmull_rom(y[seg - 1], y[seg], y[seg + 1], y[seg + 2], t);
        }
    }

    Ok(result)
}

/// Catmull-Rom spline interpolation helper.
fn catmull_rom(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;

    0.5 * (
        2.0 * p1 +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
}

/// Computes cumulative sum.
pub fn cumsum(data: ArrayView1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(data.len());
    let mut sum = 0.0;

    for (i, &val) in data.iter().enumerate() {
        sum += val;
        result[i] = sum;
    }

    result
}

/// Computes cumulative product.
pub fn cumprod(data: ArrayView1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(data.len());
    let mut prod = 1.0;

    for (i, &val) in data.iter().enumerate() {
        prod *= val;
        result[i] = prod;
    }

    result
}

/// Computes the difference between consecutive elements.
pub fn diff(data: ArrayView1<f64>) -> Array1<f64> {
    if data.len() < 2 {
        return Array1::zeros(0);
    }

    let mut result = Array1::zeros(data.len() - 1);
    for i in 0..data.len() - 1 {
        result[i] = data[i + 1] - data[i];
    }

    result
}

/// Computes the gradient (numerical derivative).
pub fn gradient(data: ArrayView1<f64>, spacing: f64) -> Array1<f64> {
    if data.len() < 2 {
        return Array1::zeros(data.len());
    }

    let mut result = Array1::zeros(data.len());

    // Forward difference for first point
    result[0] = (data[1] - data[0]) / spacing;

    // Central difference for interior points
    for i in 1..data.len() - 1 {
        result[i] = (data[i + 1] - data[i - 1]) / (2.0 * spacing);
    }

    // Backward difference for last point
    let last = data.len() - 1;
    result[last] = (data[last] - data[last - 1]) / spacing;

    result
}

/// Integrates using the trapezoidal rule.
pub fn trapz(y: ArrayView1<f64>, x: Option<ArrayView1<f64>>) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0;

    if let Some(x_vals) = x {
        if x_vals.len() != y.len() {
            return 0.0;
        }
        for i in 0..y.len() - 1 {
            sum += 0.5 * (y[i] + y[i + 1]) * (x_vals[i + 1] - x_vals[i]);
        }
    } else {
        // Uniform spacing of 1.0
        for i in 0..y.len() - 1 {
            sum += 0.5 * (y[i] + y[i + 1]);
        }
    }

    sum
}

/// Computes the softmax function.
pub fn softmax(data: ArrayView1<f64>) -> Array1<f64> {
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = data.mapv(|x| (x - max).exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum
}

/// Computes the log-softmax function (numerically stable).
pub fn log_softmax(data: ArrayView1<f64>) -> Array1<f64> {
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shifted = data.mapv(|x| x - max);
    let log_sum_exp = shifted.mapv(|x| x.exp()).sum().ln();
    shifted.mapv(|x| x - log_sum_exp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mean() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_relative_eq!(mean(data.view()), 3.0);
    }

    #[test]
    fn test_variance() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let var = variance(data.view(), 0);
        assert!(var > 0.0);
    }

    #[test]
    fn test_median() {
        let data = Array1::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0]);
        assert_relative_eq!(median(data.view()), 3.0);
    }

    #[test]
    fn test_percentile() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let p50 = percentile(data.view(), 50.0).unwrap();
        assert_relative_eq!(p50, 3.0);
    }

    #[test]
    fn test_correlation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let corr = correlation(x.view(), y.view()).unwrap();
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolate_linear() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 4.0]);
        let x_new = Array1::from_vec(vec![0.5, 1.5]);
        let y_new = interpolate_linear(x.view(), y.view(), x_new.view()).unwrap();

        assert_relative_eq!(y_new[0], 0.5);
        assert_relative_eq!(y_new[1], 2.5);
    }

    #[test]
    fn test_cumsum() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let cs = cumsum(data.view());

        assert_eq!(cs[0], 1.0);
        assert_eq!(cs[1], 3.0);
        assert_eq!(cs[2], 6.0);
        assert_eq!(cs[3], 10.0);
    }

    #[test]
    fn test_diff() {
        let data = Array1::from_vec(vec![1.0, 3.0, 6.0, 10.0]);
        let d = diff(data.view());

        assert_eq!(d[0], 2.0);
        assert_eq!(d[1], 3.0);
        assert_eq!(d[2], 4.0);
    }

    #[test]
    fn test_trapz() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let integral = trapz(y.view(), None);

        assert_relative_eq!(integral, 4.0); // (1+2)/2 + (2+3)/2 = 1.5 + 2.5 = 4
    }

    #[test]
    fn test_softmax() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let sm = softmax(data.view());

        assert_relative_eq!(sm.sum(), 1.0, epsilon = 1e-10);
        assert!(sm[2] > sm[1]);
        assert!(sm[1] > sm[0]);
    }
}
