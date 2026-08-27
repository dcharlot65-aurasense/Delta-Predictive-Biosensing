//! Network architecture visualization utilities.

use super::{colors, export::{JsonBuilder, SvgBuilder}, PlotConfig, Visualization};

/// Network layer connectivity diagram.
pub struct NetworkGraph {
    layer_sizes: Vec<usize>,
    layer_names: Vec<String>,
    config: PlotConfig,
}

impl NetworkGraph {
    /// Creates a new network graph.
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let layer_names = (0..layer_sizes.len())
            .map(|i| format!("Layer {}", i))
            .collect();
        Self {
            layer_sizes,
            layer_names,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new network graph with layer names.
    pub fn with_names(layer_sizes: Vec<usize>, layer_names: Vec<String>) -> Self {
        Self {
            layer_sizes,
            layer_names,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new network graph with custom configuration.
    pub fn with_config(
        layer_sizes: Vec<usize>,
        layer_names: Vec<String>,
        config: PlotConfig,
    ) -> Self {
        Self {
            layer_sizes,
            layer_names,
            config,
        }
    }
}

impl Visualization for NetworkGraph {
    fn name(&self) -> &str {
        "NetworkGraph"
    }

    fn render_svg(&self) -> String {
        let mut svg = SvgBuilder::new(self.config.width, self.config.height);
        let margin = 80.0;
        let width = self.config.width as f32;
        let height = self.config.height as f32;

        // Add title if present
        if let Some(ref title) = self.config.title {
            svg.title(title, self.config.width);
        }

        if !self.layer_sizes.is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            let layer_spacing = plot_width / (self.layer_sizes.len() - 1).max(1) as f32;
            let max_neurons = *self.layer_sizes.iter().max().unwrap_or(&1);
            let neuron_radius = 8.0_f32.min(plot_height / (max_neurons as f32 * 3.0));

            // Draw connections between layers
            for layer_idx in 0..self.layer_sizes.len().saturating_sub(1) {
                let curr_size = self.layer_sizes[layer_idx];
                let next_size = self.layer_sizes[layer_idx + 1];

                let x1 = margin + layer_idx as f32 * layer_spacing;
                let x2 = margin + (layer_idx + 1) as f32 * layer_spacing;

                let curr_spacing = plot_height / curr_size as f32;
                let next_spacing = plot_height / next_size as f32;

                // Draw sample connections (limit to avoid clutter)
                let max_connections = 100;
                let conn_step = ((curr_size * next_size) / max_connections).max(1);

                for i in (0..curr_size).step_by(conn_step) {
                    for j in (0..next_size).step_by(conn_step) {
                        let y1 = margin + (i as f32 + 0.5) * curr_spacing;
                        let y2 = margin + (j as f32 + 0.5) * next_spacing;
                        svg.line(x1, y1, x2, y2, "#d1d5db", 0.5);
                    }
                }
            }

            // Draw neurons
            for (layer_idx, &size) in self.layer_sizes.iter().enumerate() {
                let x = margin + layer_idx as f32 * layer_spacing;
                let neuron_spacing = plot_height / size as f32;

                // Determine how many neurons to actually draw (avoid clutter)
                let max_draw = 20;
                let draw_step = (size / max_draw).max(1);

                for i in (0..size).step_by(draw_step) {
                    let y = margin + (i as f32 + 0.5) * neuron_spacing;
                    svg.circle(x, y, neuron_radius, "#2563eb");
                }

                // Add layer label
                let label = if layer_idx < self.layer_names.len() {
                    &self.layer_names[layer_idx]
                } else {
                    "Layer"
                };
                svg.text_anchor(x, height - margin + 25.0, label, 12, "#000000", "middle");
                svg.text_anchor(
                    x,
                    height - margin + 40.0,
                    &format!("({})", size),
                    10,
                    "#6b7280",
                    "middle",
                );
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        let sizes: Vec<f64> = self.layer_sizes.iter().map(|&x| x as f64).collect();
        json.add_string("type", "NetworkGraph")
            .add_number_array("layer_sizes", &sizes)
            .add_string_array("layer_names", &self.layer_names)
            .add_int("num_layers", self.layer_sizes.len() as i64);
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Weight matrix heatmap visualization.
pub struct WeightHeatmap {
    weights: Vec<Vec<f32>>, // [input][output]
    config: PlotConfig,
}

impl WeightHeatmap {
    /// Creates a new weight heatmap.
    pub fn new(weights: Vec<Vec<f32>>) -> Self {
        Self {
            weights,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new weight heatmap with custom configuration.
    pub fn with_config(weights: Vec<Vec<f32>>, config: PlotConfig) -> Self {
        Self { weights, config }
    }
}

impl Visualization for WeightHeatmap {
    fn name(&self) -> &str {
        "WeightHeatmap"
    }

    fn render_svg(&self) -> String {
        let mut svg = SvgBuilder::new(self.config.width, self.config.height);
        let margin = 60.0;
        let width = self.config.width as f32;
        let height = self.config.height as f32;

        // Add title if present
        if let Some(ref title) = self.config.title {
            svg.title(title, self.config.width);
        }

        // Add axes
        svg.axes(margin, &self.config);

        // Plot heatmap
        if !self.weights.is_empty() && !self.weights[0].is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            let num_inputs = self.weights.len();
            let num_outputs = self.weights[0].len();

            // Find min/max for normalization (centered around 0)
            let mut min_weight = f32::INFINITY;
            let mut max_weight = f32::NEG_INFINITY;
            for row in &self.weights {
                for &w in row {
                    min_weight = min_weight.min(w);
                    max_weight = max_weight.max(w);
                }
            }
            // Guard the divisor: an all-zero weight matrix gives abs_max == 0,
            // and weight / 0.0 is NaN, which the colormap cannot render.
            let abs_max = max_weight.abs().max(min_weight.abs());
            let scale = if abs_max < 1e-6 { 1.0 } else { abs_max };

            let cell_width = plot_width / num_inputs as f32;
            let cell_height = plot_height / num_outputs as f32;

            for (i, input_weights) in self.weights.iter().enumerate() {
                for (o, &weight) in input_weights.iter().enumerate() {
                    let x = margin + i as f32 * cell_width;
                    let y = height - margin - (o + 1) as f32 * cell_height;
                    // Normalize to [0, 1] with 0.5 being zero weight
                    let normalized = ((weight / scale) + 1.0) / 2.0;
                    let normalized = normalized.clamp(0.0, 1.0);
                    let (r, g, b) = colors::colormap(&self.config.colormap, normalized);
                    let color = colors::rgb_to_hex(r, g, b);
                    svg.rect(x, y, cell_width, cell_height, &color);
                }
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        let weights_2d: Vec<Vec<f64>> = self
            .weights
            .iter()
            .map(|row| row.iter().map(|&x| x as f64).collect())
            .collect();

        json.add_string("type", "WeightHeatmap")
            .add_number_array_2d("weights", &weights_2d)
            .add_int("num_inputs", self.weights.len() as i64)
            .add_int(
                "num_outputs",
                if self.weights.is_empty() { 0 } else { self.weights[0].len() as i64 },
            );
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Layer activation map over time.
pub struct ActivationMap {
    activations: Vec<Vec<f32>>, // [timestep][neuron]
    timesteps: Vec<f64>,
    config: PlotConfig,
}

impl ActivationMap {
    /// Creates a new activation map.
    pub fn new(activations: Vec<Vec<f32>>, timesteps: Vec<f64>) -> Self {
        Self {
            activations,
            timesteps,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new activation map with custom configuration.
    pub fn with_config(activations: Vec<Vec<f32>>, timesteps: Vec<f64>, config: PlotConfig) -> Self {
        Self {
            activations,
            timesteps,
            config,
        }
    }
}

impl Visualization for ActivationMap {
    fn name(&self) -> &str {
        "ActivationMap"
    }

    fn render_svg(&self) -> String {
        let mut svg = SvgBuilder::new(self.config.width, self.config.height);
        let margin = 60.0;
        let width = self.config.width as f32;
        let height = self.config.height as f32;

        // Add title if present
        if let Some(ref title) = self.config.title {
            svg.title(title, self.config.width);
        }

        // Add axes
        svg.axes(margin, &self.config);

        // Plot activation map
        if !self.activations.is_empty() && !self.activations[0].is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            let num_timesteps = self.activations.len();
            let num_neurons = self.activations[0].len();

            // Find min/max for normalization
            let mut min_act = f32::INFINITY;
            let mut max_act = f32::NEG_INFINITY;
            for row in &self.activations {
                for &act in row {
                    min_act = min_act.min(act);
                    max_act = max_act.max(act);
                }
            }
            let range = if (max_act - min_act).abs() < 1e-6 {
                1.0
            } else {
                max_act - min_act
            };

            let cell_width = plot_width / num_timesteps as f32;
            let cell_height = plot_height / num_neurons as f32;

            for (t, timestep_acts) in self.activations.iter().enumerate() {
                for (n, &act) in timestep_acts.iter().enumerate() {
                    let x = margin + t as f32 * cell_width;
                    let y = height - margin - (n + 1) as f32 * cell_height;
                    let normalized = ((act - min_act) / range).clamp(0.0, 1.0);
                    let (r, g, b) = colors::colormap(&self.config.colormap, normalized);
                    let color = colors::rgb_to_hex(r, g, b);
                    svg.rect(x, y, cell_width, cell_height, &color);
                }
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        let acts_2d: Vec<Vec<f64>> = self
            .activations
            .iter()
            .map(|row| row.iter().map(|&x| x as f64).collect())
            .collect();

        json.add_string("type", "ActivationMap")
            .add_number_array_2d("activations", &acts_2d)
            .add_number_array("timesteps", &self.timesteps)
            .add_int("num_timesteps", self.activations.len() as i64)
            .add_int(
                "num_neurons",
                if self.activations.is_empty() { 0 } else { self.activations[0].len() as i64 },
            );
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_graph() {
        let layer_sizes = vec![784, 128, 64, 10];
        let plot = NetworkGraph::new(layer_sizes);
        assert_eq!(plot.name(), "NetworkGraph");
        assert_eq!(plot.dimensions(), (800, 600));

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("NetworkGraph"));
        assert!(json.contains("layer_sizes"));
    }

    #[test]
    fn test_weight_heatmap() {
        let weights = vec![
            vec![0.5, -0.3, 0.8],
            vec![-0.2, 0.9, -0.1],
            vec![0.1, 0.4, -0.6],
        ];
        let plot = WeightHeatmap::new(weights);
        assert_eq!(plot.name(), "WeightHeatmap");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("WeightHeatmap"));
        assert!(json.contains("weights"));
    }

    #[test]
    fn test_activation_map() {
        let activations = vec![
            vec![0.1, 0.5, 0.9],
            vec![0.3, 0.7, 0.4],
            vec![0.8, 0.2, 0.6],
        ];
        let timesteps = vec![0.0, 0.1, 0.2];
        let plot = ActivationMap::new(activations, timesteps);
        assert_eq!(plot.name(), "ActivationMap");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("ActivationMap"));
        assert!(json.contains("activations"));
    }
}
