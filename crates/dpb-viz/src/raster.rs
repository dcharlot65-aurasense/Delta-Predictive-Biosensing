//! Spike raster plot generation for visualizing neural activity patterns.
//!
//! This module provides tools for creating 2D and 3D raster plots of spike trains,
//! with support for various color schemes, temporal playback, and multiple export formats.
//!
//! # Examples
//!
//! ```rust
//! use dpb_viz::raster::{RasterPlot, RasterConfig, ColorScheme};
//!
//! // Create a 2D raster plot
//! let config = RasterConfig::new(100, 1000.0) // 100 neurons, 1000ms duration
//!     .with_color_scheme(ColorScheme::Timing)
//!     .with_marker_size(2.0);
//!
//! let mut raster = RasterPlot::with_config(config);
//!
//! // Add spikes
//! raster.add_spike(0, 10.5); // Neuron 0 fires at 10.5ms
//! raster.add_spike(5, 15.2); // Neuron 5 fires at 15.2ms
//! raster.add_spike(0, 25.8); // Neuron 0 fires again at 25.8ms
//!
//! // Export to SVG
//! let svg = raster.to_svg();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Result, VizError};

/// Configuration for raster plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterConfig {
    /// Number of neurons to display
    pub num_neurons: usize,
    /// Duration of the plot in milliseconds
    pub duration_ms: f32,
    /// Width of the plot in pixels
    pub width: u32,
    /// Height of the plot in pixels
    pub height: u32,
    /// Marker size for spikes
    pub marker_size: f32,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Whether to show grid lines
    pub show_grid: bool,
    /// Whether to show axis labels
    pub show_labels: bool,
    /// Background color (RGB hex)
    pub background_color: String,
}

impl Default for RasterConfig {
    fn default() -> Self {
        Self {
            num_neurons: 100,
            duration_ms: 1000.0,
            width: 800,
            height: 600,
            marker_size: 1.5,
            color_scheme: ColorScheme::Uniform,
            show_grid: true,
            show_labels: true,
            background_color: "#FFFFFF".to_string(),
        }
    }
}

impl RasterConfig {
    /// Create a new raster configuration
    pub fn new(num_neurons: usize, duration_ms: f32) -> Self {
        Self {
            num_neurons,
            duration_ms,
            ..Default::default()
        }
    }

    /// Set the plot dimensions
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the marker size
    pub fn with_marker_size(mut self, size: f32) -> Self {
        self.marker_size = size;
        self
    }

    /// Set the color scheme
    pub fn with_color_scheme(mut self, scheme: ColorScheme) -> Self {
        self.color_scheme = scheme;
        self
    }

    /// Set whether to show grid lines
    pub fn with_grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Set whether to show axis labels
    pub fn with_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Set the background color
    pub fn with_background(mut self, color: impl Into<String>) -> Self {
        self.background_color = color.into();
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.num_neurons == 0 {
            return Err(VizError::InvalidConfig(
                "Number of neurons must be > 0".to_string(),
            ));
        }
        if self.duration_ms <= 0.0 {
            return Err(VizError::InvalidConfig(
                "Duration must be > 0".to_string(),
            ));
        }
        if self.width == 0 || self.height == 0 {
            return Err(VizError::InvalidConfig(
                "Dimensions must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Color schemes for spike visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScheme {
    /// All spikes same color
    Uniform,
    /// Color by spike timing (early to late)
    Timing,
    /// Color by neuron ID
    NeuronId,
    /// Color by layer (for 3D rasters)
    Layer,
    /// Color by spike frequency
    Frequency,
}

impl ColorScheme {
    /// Get the color for a spike based on the scheme
    pub fn get_color(&self, neuron_id: usize, time_ms: f32, duration_ms: f32) -> String {
        match self {
            ColorScheme::Uniform => "#000000".to_string(),
            ColorScheme::Timing => {
                // Gradient from blue (early) to red (late)
                let t = (time_ms / duration_ms).clamp(0.0, 1.0);
                let r = (t * 255.0) as u8;
                let b = ((1.0 - t) * 255.0) as u8;
                format!("#{:02X}00{:02X}", r, b)
            }
            ColorScheme::NeuronId => {
                // Hash-based color per neuron
                let hue = ((neuron_id * 137) % 360) as f32;
                Self::hsl_to_hex(hue, 70.0, 50.0)
            }
            ColorScheme::Layer => {
                // Different color per layer (layer determined by neuron_id / 100)
                let layer = neuron_id / 100;
                let colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"];
                colors[layer % colors.len()].to_string()
            }
            ColorScheme::Frequency => "#4A90E2".to_string(), // Default blue, frequency computed later
        }
    }

    fn hsl_to_hex(h: f32, s: f32, l: f32) -> String {
        let h = h / 360.0;
        let s = s / 100.0;
        let l = l / 100.0;

        let q = if l < 0.5 {
            l * (1.0 + s)
        } else {
            l + s - l * s
        };
        let p = 2.0 * l - q;

        let r = (Self::hue_to_rgb(p, q, h + 1.0 / 3.0) * 255.0) as u8;
        let g = (Self::hue_to_rgb(p, q, h) * 255.0) as u8;
        let b = (Self::hue_to_rgb(p, q, h - 1.0 / 3.0) * 255.0) as u8;

        format!("#{:02X}{:02X}{:02X}", r, g, b)
    }

    fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 1.0 / 2.0 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    }
}

/// A spike event in the raster plot
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Spike {
    /// Neuron ID
    pub neuron_id: usize,
    /// Time in milliseconds
    pub time_ms: f32,
    /// Optional layer ID for 3D rasters
    pub layer_id: Option<usize>,
}

/// 2D raster plot for spike visualization
#[derive(Debug, Clone)]
pub struct RasterPlot {
    config: RasterConfig,
    spikes: Vec<Spike>,
    spike_counts: HashMap<usize, usize>, // Neuron ID -> spike count
}

impl RasterPlot {
    /// Create a new raster plot with default configuration
    pub fn new(num_neurons: usize, duration_ms: f32) -> Self {
        Self::with_config(RasterConfig::new(num_neurons, duration_ms))
    }

    /// Create a new raster plot with custom configuration
    pub fn with_config(config: RasterConfig) -> Self {
        Self {
            config,
            spikes: Vec::new(),
            spike_counts: HashMap::new(),
        }
    }

    /// Add a spike to the plot
    pub fn add_spike(&mut self, neuron_id: usize, time_ms: f32) -> Result<()> {
        if neuron_id >= self.config.num_neurons {
            return Err(VizError::InvalidConfig(format!(
                "Neuron ID {} exceeds num_neurons {}",
                neuron_id, self.config.num_neurons
            )));
        }
        if time_ms < 0.0 || time_ms > self.config.duration_ms {
            return Err(VizError::InvalidTimeRange {
                start: 0.0,
                end: self.config.duration_ms,
            });
        }

        self.spikes.push(Spike {
            neuron_id,
            time_ms,
            layer_id: None,
        });

        *self.spike_counts.entry(neuron_id).or_insert(0) += 1;

        Ok(())
    }

    /// Add multiple spikes at once
    pub fn add_spikes(&mut self, spikes: &[(usize, f32)]) -> Result<()> {
        for &(neuron_id, time_ms) in spikes {
            self.add_spike(neuron_id, time_ms)?;
        }
        Ok(())
    }

    /// Get the total number of spikes
    pub fn spike_count(&self) -> usize {
        self.spikes.len()
    }

    /// Get the spike rate for a specific neuron (spikes/sec)
    pub fn neuron_spike_rate(&self, neuron_id: usize) -> f32 {
        let count = self.spike_counts.get(&neuron_id).copied().unwrap_or(0);
        (count as f32) / (self.config.duration_ms / 1000.0)
    }

    /// Get the average spike rate across all neurons (spikes/sec)
    pub fn average_spike_rate(&self) -> f32 {
        let total_spikes = self.spikes.len() as f32;
        let duration_sec = self.config.duration_ms / 1000.0;
        total_spikes / (self.config.num_neurons as f32 * duration_sec)
    }

    /// Clear all spikes
    pub fn clear(&mut self) {
        self.spikes.clear();
        self.spike_counts.clear();
    }

    /// Export to SVG format
    pub fn to_svg(&self) -> String {
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));
        svg.push('\n');

        // Background
        svg.push_str(&format!(
            r#"  <rect width="100%" height="100%" fill="{}"/>"#,
            self.config.background_color
        ));
        svg.push('\n');

        // Grid lines
        if self.config.show_grid {
            svg.push_str(r##"  <g stroke="#E0E0E0" stroke-width="0.5">"##);
            svg.push('\n');

            // Horizontal grid lines (every 10 neurons)
            for i in (0..=self.config.num_neurons).step_by(10) {
                let y = (i as f32 / self.config.num_neurons as f32) * self.config.height as f32;
                svg.push_str(&format!(
                    r#"    <line x1="0" y1="{}" x2="{}" y2="{}"/>"#,
                    y, self.config.width, y
                ));
                svg.push('\n');
            }

            // Vertical grid lines (every 100ms)
            let time_step = 100.0;
            let mut t = 0.0;
            while t <= self.config.duration_ms {
                let x = (t / self.config.duration_ms) * self.config.width as f32;
                svg.push_str(&format!(
                    r#"    <line x1="{}" y1="0" x2="{}" y2="{}"/>"#,
                    x, x, self.config.height
                ));
                svg.push('\n');
                t += time_step;
            }

            svg.push_str("  </g>\n");
        }

        // Spikes
        svg.push_str("  <g>\n");
        for spike in &self.spikes {
            let x = (spike.time_ms / self.config.duration_ms) * self.config.width as f32;
            let y = (spike.neuron_id as f32 / self.config.num_neurons as f32)
                * self.config.height as f32;
            let color = self
                .config
                .color_scheme
                .get_color(spike.neuron_id, spike.time_ms, self.config.duration_ms);

            svg.push_str(&format!(
                r#"    <circle cx="{}" cy="{}" r="{}" fill="{}"/>"#,
                x, y, self.config.marker_size, color
            ));
            svg.push('\n');
        }
        svg.push_str("  </g>\n");

        // Axis labels
        if self.config.show_labels {
            svg.push_str(r##"  <g font-family="Arial" font-size="12" fill="#333">"##);
            svg.push('\n');

            // X-axis label
            svg.push_str(&format!(
                r#"    <text x="{}" y="{}" text-anchor="middle">Time (ms)</text>"#,
                self.config.width / 2,
                self.config.height - 5
            ));
            svg.push('\n');

            // Y-axis label
            svg.push_str(r#"    <text x="10" y="15">Neuron ID</text>"#);
            svg.push('\n');

            svg.push_str("  </g>\n");
        }

        svg.push_str("</svg>");
        svg
    }

    /// Export to JSON format
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "config": {
                "num_neurons": self.config.num_neurons,
                "duration_ms": self.config.duration_ms,
                "width": self.config.width,
                "height": self.config.height,
                "color_scheme": self.config.color_scheme,
            },
            "spikes": self.spikes.iter().map(|s| {
                serde_json::json!({
                    "neuron_id": s.neuron_id,
                    "time_ms": s.time_ms,
                    "layer_id": s.layer_id,
                })
            }).collect::<Vec<_>>(),
            "statistics": {
                "total_spikes": self.spike_count(),
                "average_spike_rate": self.average_spike_rate(),
            }
        })
    }

    /// Get spikes in a time window
    pub fn spikes_in_window(&self, start_ms: f32, end_ms: f32) -> Vec<&Spike> {
        self.spikes
            .iter()
            .filter(|s| s.time_ms >= start_ms && s.time_ms <= end_ms)
            .collect()
    }

    /// Get spike times for a specific neuron
    pub fn neuron_spikes(&self, neuron_id: usize) -> Vec<f32> {
        self.spikes
            .iter()
            .filter(|s| s.neuron_id == neuron_id)
            .map(|s| s.time_ms)
            .collect()
    }
}

/// 3D raster plot with layer information
#[derive(Debug, Clone)]
pub struct RasterPlot3D {
    config: RasterConfig,
    spikes: Vec<Spike>,
    num_layers: usize,
}

impl RasterPlot3D {
    /// Create a new 3D raster plot
    pub fn new(num_neurons: usize, duration_ms: f32, num_layers: usize) -> Self {
        Self {
            config: RasterConfig::new(num_neurons, duration_ms),
            spikes: Vec::new(),
            num_layers,
        }
    }

    /// Add a spike with layer information
    pub fn add_spike(&mut self, neuron_id: usize, time_ms: f32, layer_id: usize) -> Result<()> {
        if neuron_id >= self.config.num_neurons {
            return Err(VizError::InvalidConfig(format!(
                "Neuron ID {} exceeds num_neurons {}",
                neuron_id, self.config.num_neurons
            )));
        }
        if layer_id >= self.num_layers {
            return Err(VizError::InvalidConfig(format!(
                "Layer ID {} exceeds num_layers {}",
                layer_id, self.num_layers
            )));
        }

        self.spikes.push(Spike {
            neuron_id,
            time_ms,
            layer_id: Some(layer_id),
        });

        Ok(())
    }

    /// Export as multiple 2D raster plots (one per layer)
    pub fn to_layer_rasters(&self) -> Vec<RasterPlot> {
        let mut rasters = Vec::new();

        for layer in 0..self.num_layers {
            let mut raster = RasterPlot::with_config(self.config.clone());
            for spike in &self.spikes {
                if spike.layer_id == Some(layer) {
                    let _ = raster.add_spike(spike.neuron_id, spike.time_ms);
                }
            }
            rasters.push(raster);
        }

        rasters
    }

    /// Export to JSON with layer information
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "config": {
                "num_neurons": self.config.num_neurons,
                "duration_ms": self.config.duration_ms,
                "num_layers": self.num_layers,
            },
            "spikes": self.spikes.iter().map(|s| {
                serde_json::json!({
                    "neuron_id": s.neuron_id,
                    "time_ms": s.time_ms,
                    "layer_id": s.layer_id,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raster_config() {
        let config = RasterConfig::new(50, 500.0)
            .with_dimensions(1000, 800)
            .with_marker_size(2.5)
            .with_color_scheme(ColorScheme::Timing);

        assert_eq!(config.num_neurons, 50);
        assert_eq!(config.duration_ms, 500.0);
        assert_eq!(config.width, 1000);
        assert_eq!(config.height, 800);
        assert_eq!(config.marker_size, 2.5);
        assert_eq!(config.color_scheme, ColorScheme::Timing);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_raster_config() {
        let config = RasterConfig::new(0, 1000.0);
        assert!(config.validate().is_err());

        let config = RasterConfig::new(100, -10.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_color_schemes() {
        assert_eq!(ColorScheme::Uniform.get_color(0, 500.0, 1000.0), "#000000");

        let timing_color = ColorScheme::Timing.get_color(0, 500.0, 1000.0);
        assert!(timing_color.starts_with('#'));

        let neuron_color = ColorScheme::NeuronId.get_color(5, 500.0, 1000.0);
        assert!(neuron_color.starts_with('#'));
    }

    #[test]
    fn test_raster_plot_basic() {
        let mut raster = RasterPlot::new(10, 100.0);

        assert!(raster.add_spike(0, 10.0).is_ok());
        assert!(raster.add_spike(5, 50.0).is_ok());
        assert!(raster.add_spike(9, 90.0).is_ok());

        assert_eq!(raster.spike_count(), 3);
    }

    #[test]
    fn test_raster_plot_invalid_spike() {
        let mut raster = RasterPlot::new(10, 100.0);

        // Neuron ID out of range
        assert!(raster.add_spike(10, 50.0).is_err());

        // Time out of range
        assert!(raster.add_spike(5, 150.0).is_err());
        assert!(raster.add_spike(5, -10.0).is_err());
    }

    #[test]
    fn test_spike_rate_calculation() {
        let mut raster = RasterPlot::new(10, 1000.0); // 1 second

        // Add 5 spikes to neuron 0
        for i in 0..5 {
            raster.add_spike(0, i as f32 * 200.0).unwrap();
        }

        // Neuron 0 should have 5 spikes/sec
        assert_eq!(raster.neuron_spike_rate(0), 5.0);

        // Average rate: 5 spikes / 10 neurons / 1 sec = 0.5
        assert_eq!(raster.average_spike_rate(), 0.5);
    }

    #[test]
    fn test_spikes_in_window() {
        let mut raster = RasterPlot::new(10, 1000.0);

        raster.add_spike(0, 100.0).unwrap();
        raster.add_spike(1, 200.0).unwrap();
        raster.add_spike(2, 300.0).unwrap();
        raster.add_spike(3, 400.0).unwrap();

        let window_spikes = raster.spikes_in_window(150.0, 350.0);
        assert_eq!(window_spikes.len(), 2); // Should get spikes at 200 and 300
    }

    #[test]
    fn test_neuron_spikes() {
        let mut raster = RasterPlot::new(10, 1000.0);

        raster.add_spike(5, 100.0).unwrap();
        raster.add_spike(5, 200.0).unwrap();
        raster.add_spike(3, 150.0).unwrap();
        raster.add_spike(5, 300.0).unwrap();

        let neuron5_spikes = raster.neuron_spikes(5);
        assert_eq!(neuron5_spikes.len(), 3);
        assert_eq!(neuron5_spikes, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn test_add_multiple_spikes() {
        let mut raster = RasterPlot::new(10, 1000.0);

        let spikes = vec![(0, 10.0), (1, 20.0), (2, 30.0)];
        assert!(raster.add_spikes(&spikes).is_ok());
        assert_eq!(raster.spike_count(), 3);
    }

    #[test]
    fn test_raster_clear() {
        let mut raster = RasterPlot::new(10, 1000.0);

        raster.add_spike(0, 10.0).unwrap();
        raster.add_spike(1, 20.0).unwrap();
        assert_eq!(raster.spike_count(), 2);

        raster.clear();
        assert_eq!(raster.spike_count(), 0);
    }

    #[test]
    fn test_svg_export() {
        let mut raster = RasterPlot::new(10, 100.0);
        raster.add_spike(0, 10.0).unwrap();
        raster.add_spike(5, 50.0).unwrap();

        let svg = raster.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn test_json_export() {
        let mut raster = RasterPlot::new(10, 100.0);
        raster.add_spike(0, 10.0).unwrap();
        raster.add_spike(5, 50.0).unwrap();

        let json = raster.to_json();
        assert_eq!(json["config"]["num_neurons"], 10);
        assert_eq!(json["spikes"].as_array().unwrap().len(), 2);
        assert_eq!(json["statistics"]["total_spikes"], 2);
    }

    #[test]
    fn test_3d_raster_plot() {
        let mut raster = RasterPlot3D::new(10, 100.0, 3);

        assert!(raster.add_spike(0, 10.0, 0).is_ok());
        assert!(raster.add_spike(5, 50.0, 1).is_ok());
        assert!(raster.add_spike(9, 90.0, 2).is_ok());

        // Invalid layer
        assert!(raster.add_spike(0, 10.0, 3).is_err());

        let layer_rasters = raster.to_layer_rasters();
        assert_eq!(layer_rasters.len(), 3);
        assert_eq!(layer_rasters[0].spike_count(), 1);
        assert_eq!(layer_rasters[1].spike_count(), 1);
        assert_eq!(layer_rasters[2].spike_count(), 1);
    }
}
