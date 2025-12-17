//! Visualization utilities for the DPB framework.
//!
//! This module provides data generators for various visualization types,
//! outputting JSON and SVG for rendering without external dependencies.

pub mod analysis;
pub mod export;
pub mod networks;
pub mod signals;
pub mod spikes;
pub mod training;

// Re-export visualization types
pub use analysis::{ConfusionMatrix, ROCCurve};
pub use export::{SvgBuilder, JsonBuilder};
pub use networks::{ActivationMap, NetworkGraph, WeightHeatmap};
pub use signals::{AnnotatedSignalPlot, MultiChannelPlot, SignalPlot, SpectrogramPlot};
pub use spikes::{FiringRateHeatmap, ISIHistogram, RasterPlot, SpikeHistogram};
pub use training::{GradientFlowPlot, HyperparameterPlot, LearningCurve};

/// Core trait for all visualizations in the DPB framework.
pub trait Visualization: Send + Sync {
    /// Returns the name of this visualization.
    fn name(&self) -> &str;

    /// Renders the visualization as SVG string.
    fn render_svg(&self) -> String;

    /// Renders the visualization data as JSON string.
    fn render_json(&self) -> String;

    /// Returns the dimensions (width, height) in pixels.
    fn dimensions(&self) -> (u32, u32);
}

/// Configuration for plot appearance and behavior.
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Optional plot title.
    pub title: Option<String>,
    /// Optional x-axis label.
    pub x_label: Option<String>,
    /// Optional y-axis label.
    pub y_label: Option<String>,
    /// Colormap name (e.g., "viridis", "plasma", "magma").
    pub colormap: String,
    /// Whether to show grid lines.
    pub show_grid: bool,
    /// Whether to show legend.
    pub show_legend: bool,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: None,
            x_label: None,
            y_label: None,
            colormap: "viridis".to_string(),
            show_grid: true,
            show_legend: true,
        }
    }
}

impl PlotConfig {
    /// Creates a new PlotConfig with custom dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Sets the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the x-axis label.
    pub fn with_x_label(mut self, label: impl Into<String>) -> Self {
        self.x_label = Some(label.into());
        self
    }

    /// Sets the y-axis label.
    pub fn with_y_label(mut self, label: impl Into<String>) -> Self {
        self.y_label = Some(label.into());
        self
    }

    /// Sets the colormap.
    pub fn with_colormap(mut self, colormap: impl Into<String>) -> Self {
        self.colormap = colormap.into();
        self
    }

    /// Sets whether to show grid.
    pub fn with_grid(mut self, show_grid: bool) -> Self {
        self.show_grid = show_grid;
        self
    }

    /// Sets whether to show legend.
    pub fn with_legend(mut self, show_legend: bool) -> Self {
        self.show_legend = show_legend;
        self
    }
}

/// Color utility functions.
pub mod colors {
    /// Converts a normalized value [0, 1] to RGB using the viridis colormap.
    pub fn viridis(t: f32) -> (u8, u8, u8) {
        let t = t.clamp(0.0, 1.0);
        // Simplified viridis approximation
        let r = (68.0 + t * (253.0 - 68.0)) as u8;
        let g = (1.0 + t * (231.0 - 1.0)) as u8;
        let b = (84.0 + t * (37.0 - 84.0)) as u8;
        (r, g, b)
    }

    /// Converts a normalized value [0, 1] to RGB using the plasma colormap.
    pub fn plasma(t: f32) -> (u8, u8, u8) {
        let t = t.clamp(0.0, 1.0);
        let r = (13.0 + t * (240.0 - 13.0)) as u8;
        let g = (8.0 + t * (249.0 - 8.0)) as u8;
        let b = (135.0 + t * (33.0 - 135.0)) as u8;
        (r, g, b)
    }

    /// Converts a normalized value [0, 1] to RGB using the magma colormap.
    pub fn magma(t: f32) -> (u8, u8, u8) {
        let t = t.clamp(0.0, 1.0);
        let r = (0.0 + t * (252.0 - 0.0)) as u8;
        let g = (0.0 + t * (253.0 - 0.0)) as u8;
        let b = (4.0 + t * (191.0 - 4.0)) as u8;
        (r, g, b)
    }

    /// Converts a normalized value [0, 1] to RGB using the specified colormap.
    pub fn colormap(name: &str, t: f32) -> (u8, u8, u8) {
        match name {
            "plasma" => plasma(t),
            "magma" => magma(t),
            _ => viridis(t),
        }
    }

    /// Converts RGB to hex string.
    pub fn rgb_to_hex(r: u8, g: u8, b: u8) -> String {
        format!("#{:02x}{:02x}{:02x}", r, g, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_config_default() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.colormap, "viridis");
        assert!(config.show_grid);
        assert!(config.show_legend);
    }

    #[test]
    fn test_plot_config_builder() {
        let config = PlotConfig::new(1024, 768)
            .with_title("Test Plot")
            .with_x_label("X Axis")
            .with_y_label("Y Axis")
            .with_colormap("plasma")
            .with_grid(false)
            .with_legend(false);

        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert_eq!(config.title, Some("Test Plot".to_string()));
        assert_eq!(config.x_label, Some("X Axis".to_string()));
        assert_eq!(config.y_label, Some("Y Axis".to_string()));
        assert_eq!(config.colormap, "plasma");
        assert!(!config.show_grid);
        assert!(!config.show_legend);
    }

    #[test]
    fn test_colormap_viridis() {
        let (r, g, b) = colors::viridis(0.0);
        assert_eq!(r, 68);
        assert_eq!(g, 1);
        assert_eq!(b, 84);

        let (r, g, b) = colors::viridis(1.0);
        assert_eq!(r, 253);
        assert_eq!(g, 231);
        assert_eq!(b, 37);
    }

    #[test]
    fn test_rgb_to_hex() {
        assert_eq!(colors::rgb_to_hex(255, 0, 0), "#ff0000");
        assert_eq!(colors::rgb_to_hex(0, 255, 0), "#00ff00");
        assert_eq!(colors::rgb_to_hex(0, 0, 255), "#0000ff");
    }
}
