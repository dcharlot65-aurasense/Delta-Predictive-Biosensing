//! # DPB Visualization Toolkit
//!
//! Advanced visualization capabilities for the Delta-Predictive Biosensing framework.
//!
//! ## Features
//!
//! - **Real-time Dashboard**: Live monitoring of training metrics, spike rates, and system resources
//! - **Spike Raster Plots**: 2D and 3D visualization of neural activity patterns
//! - **Network Topology**: Interactive visualization of network structure and connectivity
//! - **Heatmaps**: Weight matrices, activations, and correlation visualizations
//! - **Timeline Events**: Temporal visualization of spikes, updates, and predictions
//! - **Multiple Export Formats**: SVG, PNG, JSON, and CSV export capabilities
//!
//! ## Quick Start
//!
//! ```rust
//! use dpb_viz::{RasterPlot, DashboardConfig, NetworkGraph};
//!
//! // Create a spike raster plot. The duration is in milliseconds as f32.
//! let mut raster = RasterPlot::new(100, 1000.0); // 100 neurons, 1000ms duration
//! raster.add_spike(0, 10.5); // Neuron 0 spikes at 10.5ms
//! let svg = raster.to_svg();
//!
//! // Configure a real-time dashboard
//! let config = DashboardConfig::new()
//!     .with_update_interval_ms(100)
//!     .with_port(8080);
//!
//! // Visualize network topology. `add_layer` mutates and returns the new
//! // layer's index, so it does not chain as a builder.
//! let mut graph = NetworkGraph::new();
//! let input = graph.add_layer("input", 784);
//! let hidden = graph.add_layer("hidden", 256);
//! let output = graph.add_layer("output", 10);
//! ```
//!
//! ## Architecture
//!
//! The visualization toolkit is organized into specialized modules:
//!
//! - [`dashboard`]: Real-time monitoring with WebSocket support
//! - [`raster`]: Spike raster plot generation
//! - [`network`]: Network topology visualization
//! - [`heatmap`]: Weight and activation heatmaps
//! - [`timeline`]: Temporal event visualization
//! - [`export`]: Multi-format export utilities

pub mod dashboard;
pub mod raster;
pub mod network;
pub mod heatmap;
pub mod timeline;
pub mod export;

// Re-export commonly used types
pub use dashboard::{DashboardConfig, DashboardServer, MetricPanel};
pub use raster::{RasterPlot, RasterConfig, ColorScheme as RasterColorScheme};
pub use network::{NetworkGraph, NodePositioning, EdgeStyle};
pub use heatmap::{WeightHeatmap, ActivationHeatmap, ColorScale};
pub use timeline::{EventTimeline, TimelineEvent, ZoomLevel};
pub use export::{SvgExporter, JsonExporter, CsvExporter, ExportFormat};

use thiserror::Error;

/// Result type for visualization operations
pub type Result<T> = std::result::Result<T, VizError>;

/// Errors that can occur during visualization operations
#[derive(Debug, Error)]
pub enum VizError {
    /// Invalid dimensions or configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Data dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// IO error during export
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Network error (WebSocket, etc.)
    #[error("Network error: {0}")]
    Network(String),

    /// Export format not supported
    #[error("Unsupported export format: {0}")]
    UnsupportedFormat(String),

    /// Invalid time range
    #[error("Invalid time range: start={start}, end={end}")]
    InvalidTimeRange { start: f32, end: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = VizError::InvalidConfig("test error".to_string());
        assert!(err.to_string().contains("Invalid configuration"));

        let err = VizError::DimensionMismatch {
            expected: 10,
            actual: 5,
        };
        assert!(err.to_string().contains("expected 10"));
        assert!(err.to_string().contains("got 5"));
    }

    #[test]
    fn test_invalid_time_range() {
        let err = VizError::InvalidTimeRange {
            start: 100.0,
            end: 50.0,
        };
        assert!(err.to_string().contains("start=100"));
        assert!(err.to_string().contains("end=50"));
    }
}
