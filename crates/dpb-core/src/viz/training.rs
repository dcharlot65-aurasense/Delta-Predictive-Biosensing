//! Training progress visualization utilities.

use super::{
    PlotConfig, Visualization,
    export::{JsonBuilder, SvgBuilder},
};

/// Learning curve showing loss/accuracy over epochs.
pub struct LearningCurve {
    train_loss: Vec<f64>,
    val_loss: Option<Vec<f64>>,
    train_acc: Option<Vec<f64>>,
    val_acc: Option<Vec<f64>>,
    config: PlotConfig,
}

impl LearningCurve {
    /// Creates a new learning curve with only training loss.
    pub fn new(train_loss: Vec<f64>) -> Self {
        Self {
            train_loss,
            val_loss: None,
            train_acc: None,
            val_acc: None,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new learning curve with training and validation loss.
    pub fn with_validation(train_loss: Vec<f64>, val_loss: Vec<f64>) -> Self {
        Self {
            train_loss,
            val_loss: Some(val_loss),
            train_acc: None,
            val_acc: None,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new learning curve with all metrics.
    pub fn with_all_metrics(
        train_loss: Vec<f64>,
        val_loss: Vec<f64>,
        train_acc: Vec<f64>,
        val_acc: Vec<f64>,
    ) -> Self {
        Self {
            train_loss,
            val_loss: Some(val_loss),
            train_acc: Some(train_acc),
            val_acc: Some(val_acc),
            config: PlotConfig::default(),
        }
    }

    /// Creates a new learning curve with custom configuration.
    pub fn with_config(
        train_loss: Vec<f64>,
        val_loss: Option<Vec<f64>>,
        train_acc: Option<Vec<f64>>,
        val_acc: Option<Vec<f64>>,
        config: PlotConfig,
    ) -> Self {
        Self {
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            config,
        }
    }

    fn plot_metric(
        &self,
        svg: &mut SvgBuilder,
        data: &[f64],
        margin: f32,
        width: f32,
        height: f32,
        min_val: f64,
        max_val: f64,
        color: &str,
    ) {
        if data.is_empty() {
            return;
        }

        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;
        let range = if (max_val - min_val).abs() < 1e-9 {
            1.0
        } else {
            max_val - min_val
        };

        let points: Vec<(f32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let x = margin + (i as f32 / (data.len() - 1).max(1) as f32) * plot_width;
                let normalized = ((val - min_val) / range) as f32;
                let y = height - margin - normalized * plot_height;
                (x, y)
            })
            .collect();

        svg.polyline(&points, color, 2.0, "none");
    }
}

impl Visualization for LearningCurve {
    fn name(&self) -> &str {
        "LearningCurve"
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

        // Add grid if enabled
        if self.config.show_grid {
            svg.grid(margin, self.config.width, self.config.height, 10, 8);
        }

        // Add axes
        svg.axes(margin, &self.config);

        // Find global min/max across all metrics for consistent scaling
        let mut all_values = self.train_loss.clone();
        if let Some(ref val_loss) = self.val_loss {
            all_values.extend(val_loss);
        }
        if let Some(ref train_acc) = self.train_acc {
            all_values.extend(train_acc);
        }
        if let Some(ref val_acc) = self.val_acc {
            all_values.extend(val_acc);
        }

        let min_val = all_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Plot metrics
        self.plot_metric(
            &mut svg,
            &self.train_loss,
            margin,
            width,
            height,
            min_val,
            max_val,
            "#2563eb",
        );

        if let Some(ref val_loss) = self.val_loss {
            self.plot_metric(
                &mut svg, val_loss, margin, width, height, min_val, max_val, "#dc2626",
            );
        }

        if let Some(ref train_acc) = self.train_acc {
            self.plot_metric(
                &mut svg, train_acc, margin, width, height, min_val, max_val, "#16a34a",
            );
        }

        if let Some(ref val_acc) = self.val_acc {
            self.plot_metric(
                &mut svg, val_acc, margin, width, height, min_val, max_val, "#ca8a04",
            );
        }

        // Add legend if enabled
        if self.config.show_legend {
            let mut legend_items = vec![("Train Loss", "#2563eb")];
            if self.val_loss.is_some() {
                legend_items.push(("Val Loss", "#dc2626"));
            }
            if self.train_acc.is_some() {
                legend_items.push(("Train Acc", "#16a34a"));
            }
            if self.val_acc.is_some() {
                legend_items.push(("Val Acc", "#ca8a04"));
            }

            let legend_x = width - margin - 100.0;
            let mut legend_y = margin + 20.0;
            for (label, color) in legend_items {
                svg.line(legend_x, legend_y, legend_x + 20.0, legend_y, color, 2.0);
                svg.text(legend_x + 25.0, legend_y + 5.0, label, 12, "#000000");
                legend_y += 20.0;
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        json.add_string("type", "LearningCurve")
            .add_number_array("train_loss", &self.train_loss);

        if let Some(ref val_loss) = self.val_loss {
            json.add_number_array("val_loss", val_loss);
        }
        if let Some(ref train_acc) = self.train_acc {
            json.add_number_array("train_acc", train_acc);
        }
        if let Some(ref val_acc) = self.val_acc {
            json.add_number_array("val_acc", val_acc);
        }

        json.add_int("num_epochs", self.train_loss.len() as i64);
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Gradient flow visualization showing gradient magnitudes by layer.
pub struct GradientFlowPlot {
    gradients: Vec<Vec<f64>>, // [epoch][layer]
    layer_names: Vec<String>,
    config: PlotConfig,
}

impl GradientFlowPlot {
    /// Creates a new gradient flow plot.
    pub fn new(gradients: Vec<Vec<f64>>) -> Self {
        let num_layers = if gradients.is_empty() {
            0
        } else {
            gradients[0].len()
        };
        let layer_names = (0..num_layers).map(|i| format!("Layer {}", i)).collect();
        Self {
            gradients,
            layer_names,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new gradient flow plot with layer names.
    pub fn with_names(gradients: Vec<Vec<f64>>, layer_names: Vec<String>) -> Self {
        Self {
            gradients,
            layer_names,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new gradient flow plot with custom configuration.
    pub fn with_config(
        gradients: Vec<Vec<f64>>,
        layer_names: Vec<String>,
        config: PlotConfig,
    ) -> Self {
        Self {
            gradients,
            layer_names,
            config,
        }
    }
}

impl Visualization for GradientFlowPlot {
    fn name(&self) -> &str {
        "GradientFlowPlot"
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

        // Add grid if enabled
        if self.config.show_grid {
            svg.grid(margin, self.config.width, self.config.height, 10, 8);
        }

        // Add axes
        svg.axes(margin, &self.config);

        // Plot gradient flow
        if !self.gradients.is_empty() && !self.gradients[0].is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            // Find min/max for normalization (use log scale for gradients)
            let mut min_grad = f64::INFINITY;
            let mut max_grad = f64::NEG_INFINITY;
            for epoch in &self.gradients {
                for &grad in epoch {
                    let log_grad = (grad.abs() + 1e-10).ln();
                    min_grad = min_grad.min(log_grad);
                    max_grad = max_grad.max(log_grad);
                }
            }
            let range = if (max_grad - min_grad).abs() < 1e-9 {
                1.0
            } else {
                max_grad - min_grad
            };

            let num_layers = self.gradients[0].len();
            let colors_palette = [
                "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea", "#0891b2",
            ];

            // Plot each layer's gradient over epochs
            for layer_idx in 0..num_layers {
                let points: Vec<(f32, f32)> = self
                    .gradients
                    .iter()
                    .enumerate()
                    .map(|(epoch, grads)| {
                        let x = margin
                            + (epoch as f32 / (self.gradients.len() - 1).max(1) as f32)
                                * plot_width;
                        let grad = grads.get(layer_idx).copied().unwrap_or(0.0);
                        let log_grad = (grad.abs() + 1e-10).ln();
                        let normalized = ((log_grad - min_grad) / range) as f32;
                        let y = height - margin - normalized * plot_height;
                        (x, y)
                    })
                    .collect();

                let color = colors_palette[layer_idx % colors_palette.len()];
                svg.polyline(&points, color, 2.0, "none");
            }

            // Add legend if enabled
            if self.config.show_legend {
                let legend_x = width - margin - 100.0;
                let mut legend_y = margin + 20.0;
                for (idx, name) in self.layer_names.iter().enumerate().take(num_layers) {
                    let color = colors_palette[idx % colors_palette.len()];
                    svg.line(legend_x, legend_y, legend_x + 20.0, legend_y, color, 2.0);
                    svg.text(legend_x + 25.0, legend_y + 5.0, name, 12, "#000000");
                    legend_y += 20.0;
                }
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        json.add_string("type", "GradientFlowPlot")
            .add_number_array_2d("gradients", &self.gradients)
            .add_string_array("layer_names", &self.layer_names)
            .add_int("num_epochs", self.gradients.len() as i64)
            .add_int(
                "num_layers",
                if self.gradients.is_empty() {
                    0
                } else {
                    self.gradients[0].len() as i64
                },
            );
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Hyperparameter search results visualization.
pub struct HyperparameterPlot {
    results: Vec<(f64, f64)>, // (hyperparameter_value, performance_metric)
    hp_name: String,
    metric_name: String,
    config: PlotConfig,
}

impl HyperparameterPlot {
    /// Creates a new hyperparameter plot.
    pub fn new(results: Vec<(f64, f64)>, hp_name: String, metric_name: String) -> Self {
        Self {
            results,
            hp_name,
            metric_name,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new hyperparameter plot with custom configuration.
    pub fn with_config(
        results: Vec<(f64, f64)>,
        hp_name: String,
        metric_name: String,
        config: PlotConfig,
    ) -> Self {
        Self {
            results,
            hp_name,
            metric_name,
            config,
        }
    }
}

impl Visualization for HyperparameterPlot {
    fn name(&self) -> &str {
        "HyperparameterPlot"
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

        // Add grid if enabled
        if self.config.show_grid {
            svg.grid(margin, self.config.width, self.config.height, 10, 8);
        }

        // Add axes with custom labels
        let mut config_with_labels = self.config.clone();
        config_with_labels.x_label = Some(self.hp_name.clone());
        config_with_labels.y_label = Some(self.metric_name.clone());
        svg.axes(margin, &config_with_labels);

        // Plot results
        if !self.results.is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            // Find ranges
            let hp_values: Vec<f64> = self.results.iter().map(|(h, _)| *h).collect();
            let metrics: Vec<f64> = self.results.iter().map(|(_, m)| *m).collect();

            let min_hp = hp_values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_hp = hp_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_metric = metrics.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_metric = metrics.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let hp_range = if (max_hp - min_hp).abs() < 1e-9 {
                1.0
            } else {
                max_hp - min_hp
            };
            let metric_range = if (max_metric - min_metric).abs() < 1e-9 {
                1.0
            } else {
                max_metric - min_metric
            };

            // Plot line
            let points: Vec<(f32, f32)> = self
                .results
                .iter()
                .map(|(hp, metric)| {
                    let x = margin + (((hp - min_hp) / hp_range) as f32) * plot_width;
                    let y = height
                        - margin
                        - (((metric - min_metric) / metric_range) as f32) * plot_height;
                    (x, y)
                })
                .collect();

            svg.polyline(&points, "#2563eb", 2.0, "none");

            // Add scatter points
            for (x, y) in &points {
                svg.circle(*x, *y, 4.0, "#dc2626");
            }

            // Highlight best result
            if let Some((best_idx, _)) = metrics
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                && let Some((x, y)) = points.get(best_idx)
            {
                svg.circle(*x, *y, 7.0, "#16a34a");
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        let hp_values: Vec<f64> = self.results.iter().map(|(h, _)| *h).collect();
        let metrics: Vec<f64> = self.results.iter().map(|(_, m)| *m).collect();

        json.add_string("type", "HyperparameterPlot")
            .add_string("hp_name", &self.hp_name)
            .add_string("metric_name", &self.metric_name)
            .add_number_array("hp_values", &hp_values)
            .add_number_array("metrics", &metrics)
            .add_int("num_trials", self.results.len() as i64);
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
    fn test_learning_curve() {
        let train_loss = vec![1.0, 0.8, 0.6, 0.4, 0.3];
        let val_loss = vec![1.1, 0.9, 0.7, 0.5, 0.4];
        let plot = LearningCurve::with_validation(train_loss, val_loss);
        assert_eq!(plot.name(), "LearningCurve");
        assert_eq!(plot.dimensions(), (800, 600));

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("LearningCurve"));
        assert!(json.contains("train_loss"));
        assert!(json.contains("val_loss"));
    }

    #[test]
    fn test_gradient_flow_plot() {
        let gradients = vec![
            vec![1e-3, 1e-4, 1e-5],
            vec![8e-4, 9e-5, 8e-6],
            vec![6e-4, 7e-5, 6e-6],
        ];
        let plot = GradientFlowPlot::new(gradients);
        assert_eq!(plot.name(), "GradientFlowPlot");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("GradientFlowPlot"));
        assert!(json.contains("gradients"));
    }

    #[test]
    fn test_hyperparameter_plot() {
        let results = vec![(0.001, 0.85), (0.01, 0.92), (0.1, 0.88), (1.0, 0.75)];
        let plot =
            HyperparameterPlot::new(results, "Learning Rate".to_string(), "Accuracy".to_string());
        assert_eq!(plot.name(), "HyperparameterPlot");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("HyperparameterPlot"));
        assert!(json.contains("hp_name"));
        assert!(json.contains("metric_name"));
    }
}
