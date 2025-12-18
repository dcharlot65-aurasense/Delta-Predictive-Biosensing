//! Heatmap visualization for weight matrices and activations.
//!
//! This module provides tools for creating heatmaps of weight matrices,
//! layer activations, and correlation matrices with various color scales.
//!
//! # Examples
//!
//! ```rust
//! use dpb_viz::heatmap::{WeightHeatmap, ColorScale};
//! use ndarray::Array2;
//!
//! // Create a weight matrix
//! let weights = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f32).collect()).unwrap();
//!
//! // Create a heatmap
//! let heatmap = WeightHeatmap::new(weights)
//!     .with_color_scale(ColorScale::Viridis)
//!     .with_title("Layer 1 Weights");
//!
//! // Export to SVG
//! let svg = heatmap.to_svg();
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use crate::{Result, VizError};

/// Color scales for heatmap visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScale {
    /// Viridis color scale (perceptually uniform)
    Viridis,
    /// Plasma color scale
    Plasma,
    /// Inferno color scale
    Inferno,
    /// Coolwarm diverging color scale
    Coolwarm,
    /// Grayscale
    Grayscale,
    /// Red-Blue diverging (negative to positive)
    RedBlue,
}

impl ColorScale {
    /// Get the RGB color for a normalized value (0.0 to 1.0)
    pub fn color_at(&self, t: f32) -> (u8, u8, u8) {
        let t = t.clamp(0.0, 1.0);

        match self {
            ColorScale::Viridis => Self::viridis(t),
            ColorScale::Plasma => Self::plasma(t),
            ColorScale::Inferno => Self::inferno(t),
            ColorScale::Coolwarm => Self::coolwarm(t),
            ColorScale::Grayscale => {
                let g = (t * 255.0) as u8;
                (g, g, g)
            }
            ColorScale::RedBlue => Self::red_blue(t),
        }
    }

    /// Convert RGB to hex string
    pub fn to_hex(&self, t: f32) -> String {
        let (r, g, b) = self.color_at(t);
        format!("#{:02X}{:02X}{:02X}", r, g, b)
    }

    // Simplified Viridis colormap
    fn viridis(t: f32) -> (u8, u8, u8) {
        let r = ((-4.54 * t * t + 4.55 * t - 0.26) * 255.0).clamp(0.0, 255.0) as u8;
        let g = ((3.36 * t * t + 0.77 * t + 0.18) * 255.0).clamp(0.0, 255.0) as u8;
        let b = ((-6.43 * t * t + 8.93 * t + 0.36) * 255.0).clamp(0.0, 255.0) as u8;
        (r, g, b)
    }

    // Simplified Plasma colormap
    fn plasma(t: f32) -> (u8, u8, u8) {
        let r = ((0.99 * t) * 255.0).clamp(0.0, 255.0) as u8;
        let g = ((0.65 * (1.0 - (1.0 - t).powf(0.3))) * 255.0).clamp(0.0, 255.0) as u8;
        let b = ((1.0 - t) * 255.0).clamp(0.0, 255.0) as u8;
        (r, g, b)
    }

    // Simplified Inferno colormap
    fn inferno(t: f32) -> (u8, u8, u8) {
        let r = (t * 255.0).clamp(0.0, 255.0) as u8;
        let g = ((t * t) * 255.0).clamp(0.0, 255.0) as u8;
        let b = ((t * t * t * 0.5) * 255.0).clamp(0.0, 255.0) as u8;
        (r, g, b)
    }

    // Coolwarm diverging colormap
    fn coolwarm(t: f32) -> (u8, u8, u8) {
        let r = (t * 255.0) as u8;
        let g = ((1.0 - 2.0 * (t - 0.5).abs()) * 255.0) as u8;
        let b = ((1.0 - t) * 255.0) as u8;
        (r, g, b)
    }

    // Red-Blue diverging colormap
    fn red_blue(t: f32) -> (u8, u8, u8) {
        if t < 0.5 {
            let normalized = t * 2.0;
            ((255.0 * (1.0 - normalized)) as u8, 0, 255)
        } else {
            let normalized = (t - 0.5) * 2.0;
            (255, 0, (255.0 * (1.0 - normalized)) as u8)
        }
    }
}

/// Configuration for heatmap visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapConfig {
    /// Width of each cell in pixels
    pub cell_width: f32,
    /// Height of each cell in pixels
    pub cell_height: f32,
    /// Color scale to use
    pub color_scale: ColorScale,
    /// Whether to show cell values as text
    pub show_values: bool,
    /// Whether to show axis labels
    pub show_labels: bool,
    /// Title of the heatmap
    pub title: Option<String>,
    /// Font size for cell values
    pub font_size: f32,
    /// Whether to normalize values to [0, 1]
    pub normalize: bool,
}

impl Default for HeatmapConfig {
    fn default() -> Self {
        Self {
            cell_width: 20.0,
            cell_height: 20.0,
            color_scale: ColorScale::Viridis,
            show_values: false,
            show_labels: true,
            title: None,
            font_size: 10.0,
            normalize: true,
        }
    }
}

impl HeatmapConfig {
    /// Create a new heatmap configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the cell dimensions
    pub fn with_cell_size(mut self, width: f32, height: f32) -> Self {
        self.cell_width = width;
        self.cell_height = height;
        self
    }

    /// Set the color scale
    pub fn with_color_scale(mut self, scale: ColorScale) -> Self {
        self.color_scale = scale;
        self
    }

    /// Set whether to show cell values
    pub fn with_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Set the title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

/// Heatmap for weight matrices
#[derive(Debug, Clone)]
pub struct WeightHeatmap {
    data: Array2<f32>,
    config: HeatmapConfig,
    min_value: f32,
    max_value: f32,
}

impl WeightHeatmap {
    /// Create a new weight heatmap
    pub fn new(data: Array2<f32>) -> Self {
        let min_value = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_value = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        Self {
            data,
            config: HeatmapConfig::default(),
            min_value,
            max_value,
        }
    }

    /// Create with custom configuration
    pub fn with_config(data: Array2<f32>, config: HeatmapConfig) -> Self {
        let min_value = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_value = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        Self {
            data,
            config,
            min_value,
            max_value,
        }
    }

    /// Set the color scale
    pub fn with_color_scale(mut self, scale: ColorScale) -> Self {
        self.config.color_scale = scale;
        self
    }

    /// Set the title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.config.title = Some(title.into());
        self
    }

    /// Set whether to show values
    pub fn with_values(mut self, show: bool) -> Self {
        self.config.show_values = show;
        self
    }

    /// Normalize a value to [0, 1] range
    fn normalize(&self, value: f32) -> f32 {
        if (self.max_value - self.min_value).abs() < 1e-6 {
            0.5
        } else {
            (value - self.min_value) / (self.max_value - self.min_value)
        }
    }

    /// Get the dimensions of the heatmap
    pub fn dimensions(&self) -> (usize, usize) {
        (self.data.nrows(), self.data.ncols())
    }

    /// Export to SVG format
    pub fn to_svg(&self) -> String {
        let (rows, cols) = self.dimensions();
        let title_height = if self.config.title.is_some() { 40.0 } else { 0.0 };
        let label_margin = if self.config.show_labels { 30.0 } else { 0.0 };

        let width = cols as f32 * self.config.cell_width + label_margin * 2.0;
        let height = rows as f32 * self.config.cell_height + label_margin * 2.0 + title_height;

        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            width, height
        ));
        svg.push('\n');

        // Background
        svg.push_str(r##"  <rect width="100%" height="100%" fill="#FFFFFF"/>"##);
        svg.push('\n');

        // Title
        if let Some(title) = &self.config.title {
            svg.push_str(&format!(
                r#"  <text x="{}" y="25" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold">{}</text>"#,
                width / 2.0,
                title
            ));
            svg.push('\n');
        }

        // Cells
        svg.push_str("  <g>\n");
        for i in 0..rows {
            for j in 0..cols {
                let value = self.data[[i, j]];
                let normalized = self.normalize(value);
                let color = self.config.color_scale.to_hex(normalized);

                let x = label_margin + j as f32 * self.config.cell_width;
                let y = label_margin + title_height + i as f32 * self.config.cell_height;

                svg.push_str(&format!(
                    r##"    <rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="#333" stroke-width="0.5"/>"##,
                    x, y, self.config.cell_width, self.config.cell_height, color
                ));
                svg.push('\n');

                // Cell values
                if self.config.show_values {
                    let text_x = x + self.config.cell_width / 2.0;
                    let text_y = y + self.config.cell_height / 2.0 + self.config.font_size / 3.0;
                    let text_color = if normalized > 0.5 { "#FFF" } else { "#000" };

                    svg.push_str(&format!(
                        r#"    <text x="{}" y="{}" text-anchor="middle" font-size="{}" fill="{}">{:.2}</text>"#,
                        text_x, text_y, self.config.font_size, text_color, value
                    ));
                    svg.push('\n');
                }
            }
        }
        svg.push_str("  </g>\n");

        // Axis labels
        if self.config.show_labels {
            svg.push_str(r##"  <g font-family="Arial" font-size="10" fill="#333">"##);
            svg.push('\n');

            // Column labels (every 5 columns)
            for j in (0..cols).step_by(5) {
                let x = label_margin + (j as f32 + 0.5) * self.config.cell_width;
                let y = label_margin + title_height - 5.0;
                svg.push_str(&format!(
                    r#"    <text x="{}" y="{}" text-anchor="middle">{}</text>"#,
                    x, y, j
                ));
                svg.push('\n');
            }

            // Row labels (every 5 rows)
            for i in (0..rows).step_by(5) {
                let x = label_margin - 10.0;
                let y = label_margin + title_height + (i as f32 + 0.5) * self.config.cell_height;
                svg.push_str(&format!(
                    r#"    <text x="{}" y="{}" text-anchor="end">{}</text>"#,
                    x, y, i
                ));
                svg.push('\n');
            }

            svg.push_str("  </g>\n");
        }

        // Color scale legend
        self.add_color_legend(&mut svg, width, height, title_height, label_margin);

        svg.push_str("</svg>");
        svg
    }

    fn add_color_legend(&self, svg: &mut String, width: f32, _height: f32, title_height: f32, label_margin: f32) {
        let legend_width = 200.0;
        let legend_height = 20.0;
        let legend_x = width - legend_width - label_margin;
        let legend_y = title_height + 10.0;

        svg.push_str("  <g>\n");

        // Draw color gradient
        for i in 0..100 {
            let x = legend_x + (i as f32 / 100.0) * legend_width;
            let t = i as f32 / 100.0;
            let color = self.config.color_scale.to_hex(t);

            svg.push_str(&format!(
                r#"    <rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
                x, legend_y, legend_width / 100.0, legend_height, color
            ));
            svg.push('\n');
        }

        // Legend labels
        svg.push_str(&format!(
            r#"    <text x="{}" y="{}" font-size="10">{:.2}</text>"#,
            legend_x, legend_y + legend_height + 15.0, self.min_value
        ));
        svg.push('\n');

        svg.push_str(&format!(
            r#"    <text x="{}" y="{}" text-anchor="end" font-size="10">{:.2}</text>"#,
            legend_x + legend_width, legend_y + legend_height + 15.0, self.max_value
        ));
        svg.push('\n');

        svg.push_str("  </g>\n");
    }

    /// Export to JSON format
    pub fn to_json(&self) -> serde_json::Value {
        let (rows, cols) = self.dimensions();

        serde_json::json!({
            "dimensions": [rows, cols],
            "min_value": self.min_value,
            "max_value": self.max_value,
            "color_scale": self.config.color_scale,
            "data": self.data.as_slice().unwrap(),
        })
    }
}

/// Heatmap for layer activations
#[derive(Debug, Clone)]
pub struct ActivationHeatmap {
    data: Array2<f32>,
    config: HeatmapConfig,
    layer_name: String,
}

impl ActivationHeatmap {
    /// Create a new activation heatmap
    pub fn new(data: Array2<f32>, layer_name: impl Into<String>) -> Self {
        let layer_name = layer_name.into();
        let mut config = HeatmapConfig::default();
        config.title = Some(format!("{} Activations", layer_name));

        Self {
            data,
            config,
            layer_name,
        }
    }

    /// Set the color scale
    pub fn with_color_scale(mut self, scale: ColorScale) -> Self {
        self.config.color_scale = scale;
        self
    }

    /// Export to SVG (delegates to WeightHeatmap)
    pub fn to_svg(&self) -> String {
        let heatmap = WeightHeatmap::with_config(self.data.clone(), self.config.clone());
        heatmap.to_svg()
    }

    /// Export to JSON
    pub fn to_json(&self) -> serde_json::Value {
        let heatmap = WeightHeatmap::with_config(self.data.clone(), self.config.clone());
        let mut json = heatmap.to_json();
        json["layer_name"] = serde_json::json!(self.layer_name);
        json
    }
}

/// Correlation matrix heatmap
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    data: Array2<f32>,
    labels: Vec<String>,
    config: HeatmapConfig,
}

impl CorrelationMatrix {
    /// Create a correlation matrix from data
    /// Each column is a variable, rows are observations
    pub fn from_data(data: &Array2<f32>) -> Result<Self> {
        let (n_obs, n_vars) = data.dim();

        if n_obs < 2 {
            return Err(VizError::InvalidConfig(
                "Need at least 2 observations for correlation".to_string(),
            ));
        }

        // Compute correlation matrix
        let mut corr = Array2::zeros((n_vars, n_vars));

        for i in 0..n_vars {
            for j in 0..n_vars {
                let col_i = data.column(i);
                let col_j = data.column(j);
                corr[[i, j]] = Self::pearson_correlation(&col_i.to_owned(), &col_j.to_owned());
            }
        }

        let labels = (0..n_vars).map(|i| format!("Var{}", i)).collect();

        let mut config = HeatmapConfig::default();
        config.color_scale = ColorScale::RedBlue;
        config.title = Some("Correlation Matrix".to_string());

        Ok(Self {
            data: corr,
            labels,
            config,
        })
    }

    /// Compute Pearson correlation coefficient
    fn pearson_correlation(x: &Array1<f32>, y: &Array1<f32>) -> f32 {
        let n = x.len() as f32;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        if den_x == 0.0 || den_y == 0.0 {
            0.0
        } else {
            num / (den_x * den_y).sqrt()
        }
    }

    /// Set custom labels
    pub fn with_labels(mut self, labels: Vec<String>) -> Result<Self> {
        if labels.len() != self.data.nrows() {
            return Err(VizError::DimensionMismatch {
                expected: self.data.nrows(),
                actual: labels.len(),
            });
        }
        self.labels = labels;
        Ok(self)
    }

    /// Export to SVG
    pub fn to_svg(&self) -> String {
        let heatmap = WeightHeatmap::with_config(self.data.clone(), self.config.clone());
        heatmap.to_svg()
    }

    /// Export to JSON
    pub fn to_json(&self) -> serde_json::Value {
        let heatmap = WeightHeatmap::with_config(self.data.clone(), self.config.clone());
        let mut json = heatmap.to_json();
        json["labels"] = serde_json::json!(self.labels);
        json
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_scales() {
        assert_eq!(ColorScale::Grayscale.color_at(0.0), (0, 0, 0));
        assert_eq!(ColorScale::Grayscale.color_at(1.0), (255, 255, 255));

        let viridis_color = ColorScale::Viridis.color_at(0.5);
        assert!(viridis_color.0 > 0 || viridis_color.1 > 0 || viridis_color.2 > 0);

        let hex = ColorScale::Grayscale.to_hex(0.5);
        assert!(hex.starts_with('#'));
        assert_eq!(hex.len(), 7);
    }

    #[test]
    fn test_weight_heatmap() {
        let data = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();

        let heatmap = WeightHeatmap::new(data)
            .with_color_scale(ColorScale::Viridis)
            .with_title("Test Heatmap");

        assert_eq!(heatmap.dimensions(), (3, 3));
        assert_eq!(heatmap.min_value, 1.0);
        assert_eq!(heatmap.max_value, 9.0);

        let normalized = heatmap.normalize(5.0);
        assert!((normalized - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_weight_heatmap_svg() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let heatmap = WeightHeatmap::new(data);

        let svg = heatmap.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn test_weight_heatmap_json() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let heatmap = WeightHeatmap::new(data);

        let json = heatmap.to_json();
        assert_eq!(json["dimensions"][0], 2);
        assert_eq!(json["dimensions"][1], 2);
        assert_eq!(json["min_value"], 1.0);
        assert_eq!(json["max_value"], 4.0);
    }

    #[test]
    fn test_activation_heatmap() {
        let data = Array2::from_shape_vec((3, 4), (0..12).map(|x| x as f32).collect()).unwrap();
        let heatmap = ActivationHeatmap::new(data, "layer1");

        let svg = heatmap.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("layer1 Activations"));

        let json = heatmap.to_json();
        assert_eq!(json["layer_name"], "layer1");
    }

    #[test]
    fn test_correlation_matrix() {
        // Create test data: 10 observations, 3 variables
        let data = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0, 5.0, 10.0, 15.0, 6.0,
                12.0, 18.0, 7.0, 14.0, 21.0, 8.0, 16.0, 24.0, 9.0, 18.0, 27.0, 10.0, 20.0, 30.0,
            ],
        )
        .unwrap();

        let corr = CorrelationMatrix::from_data(&data).unwrap();
        assert_eq!(corr.data.nrows(), 3);
        assert_eq!(corr.data.ncols(), 3);

        // Diagonal should be 1.0 (self-correlation)
        for i in 0..3 {
            assert!((corr.data[[i, i]] - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_correlation_invalid_data() {
        // Only 1 observation - should fail
        let data = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(CorrelationMatrix::from_data(&data).is_err());
    }

    #[test]
    fn test_correlation_with_labels() {
        let data = Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            .unwrap();

        let labels = vec!["A".to_string(), "B".to_string()];
        let corr = CorrelationMatrix::from_data(&data)
            .unwrap()
            .with_labels(labels)
            .unwrap();

        assert_eq!(corr.labels.len(), 2);
        assert_eq!(corr.labels[0], "A");
        assert_eq!(corr.labels[1], "B");
    }

    #[test]
    fn test_correlation_wrong_labels() {
        let data = Array2::from_shape_vec((5, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            .unwrap();

        let labels = vec!["A".to_string()]; // Wrong number of labels
        let corr = CorrelationMatrix::from_data(&data).unwrap();
        assert!(corr.with_labels(labels).is_err());
    }

    #[test]
    fn test_normalize_constant_values() {
        let data = Array2::from_shape_vec((2, 2), vec![5.0, 5.0, 5.0, 5.0]).unwrap();
        let heatmap = WeightHeatmap::new(data);

        // All values the same, should normalize to 0.5
        assert_eq!(heatmap.normalize(5.0), 0.5);
    }
}
