# DPB Visualization Toolkit

Advanced visualization capabilities for the Delta-Predictive Biosensing framework.

## Features

- **Real-time Dashboard**: Live monitoring of training metrics, spike rates, and system resources via WebSocket server
- **Spike Raster Plots**: 2D and 3D visualization of neural activity patterns with customizable color schemes
- **Network Topology**: Interactive visualization of network structure with multiple layout algorithms
- **Heatmaps**: Weight matrices, activations, and correlation visualizations with perceptually-uniform color scales
- **Timeline Events**: Temporal visualization of spikes, updates, and predictions across multiple zoom levels
- **Multiple Export Formats**: SVG, JSON, and CSV export capabilities

## Quick Start

```rust
use dpb_viz::{RasterPlot, DashboardConfig, NetworkGraph};

// Create a spike raster plot
let mut raster = RasterPlot::new(100, 1000.0); // 100 neurons, 1000ms duration
raster.add_spike(0, 10.5).unwrap();
let svg = raster.to_svg();

// Configure a real-time dashboard
let config = DashboardConfig::new()
    .with_update_interval_ms(100)
    .with_port(8080);

// Visualize network topology
let mut graph = NetworkGraph::new();
graph.add_layer("input", 784);
graph.add_layer("hidden", 256);
graph.add_layer("output", 10);
graph.connect_layers(0, 1, 0.8);
```

## Module Overview

### Dashboard (`dashboard`)
Real-time monitoring with WebSocket support for live metric streaming. Includes built-in panels:
- **SpikeRatePanel**: Track neural activity over time
- **LossPanel**: Monitor training and validation loss
- **AccuracyPanel**: Display validation accuracy
- **ResourcePanel**: System resource usage (CPU, GPU, Memory)

### Raster Plots (`raster`)
Spike visualization with multiple color schemes:
- Uniform, Timing-based, Neuron ID-based, Layer-based, Frequency-based
- 2D and 3D raster plot support
- SVG and JSON export

### Network Graphs (`network`)
Topology visualization with layout algorithms:
- Hierarchical, Force-directed, Circular, Grid layouts
- Edge rendering based on weights
- Layer grouping and labeling

### Heatmaps (`heatmap`)
Matrix visualization with perceptually-uniform color scales:
- Viridis, Plasma, Inferno, Coolwarm, Grayscale, RedBlue
- Weight matrices, activation patterns, correlation matrices

### Timeline (`timeline`)
Event tracking across time with zoom levels:
- Millisecond, Second, Minute, Hour precision
- Event types: Spikes, Weight Updates, Predictions, Epochs
- Custom event support

### Export (`export`)
Multi-format export utilities:
- **SVG**: Vector graphics for high-quality rendering
- **JSON**: Web-compatible data format
- **CSV**: Data analysis in external tools
- **Batch export**: Process multiple visualizations

## Architecture

The visualization toolkit is designed for:
- **Performance**: Efficient data structures for large-scale networks
- **Flexibility**: Trait-based custom panel system
- **Extensibility**: Easy integration with external visualization tools
- **Web-ready**: JSON export for web frontends

## Requirements

- Rust 1.92.0 or later (workspace requirement, may work with 1.91.1 using `--ignore-rust-version`)
- For PNG export: External tools like `rsvg-convert` or Inkscape

## Examples

### Real-time Dashboard

```rust
use dpb_viz::dashboard::{DashboardServer, DashboardConfig, SpikeRatePanel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DashboardConfig::new().with_port(8080);
    let mut server = DashboardServer::new(config);

    server.add_panel(Box::new(SpikeRatePanel::new(1000))).await;

    // In training loop:
    server.update_metric("spike_rate", 42.5).await?;

    Ok(())
}
```

### Network Visualization

```rust
use dpb_viz::network::{NetworkGraph, NodePositioning};

let mut graph = NetworkGraph::new()
    .with_positioning(NodePositioning::ForceDirected);

let input = graph.add_layer("input", 784);
let hidden = graph.add_layer("hidden", 256);
let output = graph.add_layer("output", 10);

graph.connect_layers(input, hidden, 0.8);
graph.connect_layers(hidden, output, 0.6);

let svg = graph.to_svg();
```

### Heatmap

```rust
use dpb_viz::heatmap::{WeightHeatmap, ColorScale};
use ndarray::Array2;

let weights = Array2::from_shape_vec((10, 10), (0..100).map(|x| x as f32).collect())?;

let heatmap = WeightHeatmap::new(weights)
    .with_color_scale(ColorScale::Viridis)
    .with_title("Layer 1 Weights");

let svg = heatmap.to_svg();
```

### Timeline Events

```rust
use dpb_viz::timeline::{EventTimeline, TimelineEvent, ZoomLevel};

let mut timeline = EventTimeline::new(0.0, 1000.0)
    .with_zoom_level(ZoomLevel::Milliseconds);

timeline.add_event(TimelineEvent::spike(10.5, 0, "Neuron 0 fires"))?;
timeline.add_event(TimelineEvent::prediction(500.0, "class_0", 0.95))?;

let svg = timeline.to_svg();
```

## Testing

The crate includes comprehensive unit tests:

```bash
cargo test -p dpb-viz
```

75 tests covering all major functionality.

## License

Licensed under either of:
- Apache License, Version 2.0
- MIT License

at your option.

## Contributing

Contributions are welcome! Please see the main repository for guidelines.
