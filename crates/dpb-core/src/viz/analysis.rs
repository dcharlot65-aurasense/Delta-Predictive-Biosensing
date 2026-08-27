//! Analysis visualization utilities for classification results.

use super::{
    PlotConfig, Visualization, colors,
    export::{JsonBuilder, SvgBuilder},
};

/// Confusion matrix visualization.
pub struct ConfusionMatrix {
    matrix: Vec<Vec<usize>>, // [true_class][predicted_class]
    class_names: Vec<String>,
    config: PlotConfig,
}

impl ConfusionMatrix {
    /// Creates a new confusion matrix.
    pub fn new(matrix: Vec<Vec<usize>>) -> Self {
        let num_classes = matrix.len();
        let class_names = (0..num_classes).map(|i| format!("Class {}", i)).collect();
        Self {
            matrix,
            class_names,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new confusion matrix with class names.
    pub fn with_names(matrix: Vec<Vec<usize>>, class_names: Vec<String>) -> Self {
        Self {
            matrix,
            class_names,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new confusion matrix with custom configuration.
    pub fn with_config(
        matrix: Vec<Vec<usize>>,
        class_names: Vec<String>,
        config: PlotConfig,
    ) -> Self {
        Self {
            matrix,
            class_names,
            config,
        }
    }

    /// Computes accuracy from the confusion matrix.
    pub fn accuracy(&self) -> f64 {
        let total: usize = self.matrix.iter().flat_map(|row| row.iter()).sum();
        if total == 0 {
            return 0.0;
        }
        let correct: usize = (0..self.matrix.len())
            .map(|i| {
                self.matrix
                    .get(i)
                    .and_then(|row| row.get(i))
                    .copied()
                    .unwrap_or(0)
            })
            .sum();
        correct as f64 / total as f64
    }
}

impl Visualization for ConfusionMatrix {
    fn name(&self) -> &str {
        "ConfusionMatrix"
    }

    fn render_svg(&self) -> String {
        let mut svg = SvgBuilder::new(self.config.width, self.config.height);
        let margin = 100.0;
        let width = self.config.width as f32;
        let height = self.config.height as f32;

        // Add title if present
        if let Some(ref title) = self.config.title {
            svg.title(title, self.config.width);
        } else {
            svg.title(
                &format!("Confusion Matrix (Acc: {:.2}%)", self.accuracy() * 100.0),
                self.config.width,
            );
        }

        // Plot confusion matrix
        if !self.matrix.is_empty() && !self.matrix[0].is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            let num_classes = self.matrix.len();

            // Find max value for normalization
            let max_count = self
                .matrix
                .iter()
                .flat_map(|row| row.iter())
                .max()
                .copied()
                .unwrap_or(1);

            let cell_size = (plot_width.min(plot_height) / num_classes as f32).min(80.0);

            // Center the matrix
            let total_width = cell_size * num_classes as f32;
            let total_height = cell_size * num_classes as f32;
            let start_x = margin + (plot_width - total_width) / 2.0;
            let start_y = margin + (plot_height - total_height) / 2.0;

            // Draw cells
            for (true_idx, row) in self.matrix.iter().enumerate() {
                for (pred_idx, &count) in row.iter().enumerate() {
                    let x = start_x + pred_idx as f32 * cell_size;
                    let y = start_y + true_idx as f32 * cell_size;

                    // Color based on count (darker = more predictions)
                    let normalized = if max_count > 0 {
                        count as f32 / max_count as f32
                    } else {
                        0.0
                    };

                    // Use different color for diagonal (correct predictions)
                    let (r, g, b) = if true_idx == pred_idx {
                        // Diagonal: green scale
                        let base = 240 - (normalized * 180.0) as u8;
                        (base, 255, base)
                    } else {
                        // Off-diagonal: red scale
                        let base = 240 - (normalized * 180.0) as u8;
                        (255, base, base)
                    };
                    let color = colors::rgb_to_hex(r, g, b);

                    svg.rect_stroke(x, y, cell_size, cell_size, &color, "#999999", 1.0);

                    // Add count text
                    svg.text_anchor(
                        x + cell_size / 2.0,
                        y + cell_size / 2.0 + 5.0,
                        &count.to_string(),
                        12,
                        "#000000",
                        "middle",
                    );
                }
            }

            // Add axis labels
            svg.text_anchor(
                width / 2.0,
                height - margin / 2.0,
                "Predicted Class",
                14,
                "#000000",
                "middle",
            );
            svg.text_anchor(
                margin / 2.0,
                height / 2.0,
                "True Class",
                14,
                "#000000",
                "middle",
            );

            // Add class labels
            for (i, name) in self.class_names.iter().enumerate().take(num_classes) {
                let label = if name.len() > 10 {
                    format!("{}...", &name[..7])
                } else {
                    name.clone()
                };

                // Top labels (predicted)
                svg.text_anchor(
                    start_x + (i as f32 + 0.5) * cell_size,
                    start_y - 10.0,
                    &label,
                    10,
                    "#000000",
                    "middle",
                );

                // Left labels (true)
                svg.text_anchor(
                    start_x - 10.0,
                    start_y + (i as f32 + 0.5) * cell_size + 4.0,
                    &label,
                    10,
                    "#000000",
                    "end",
                );
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        let matrix_2d: Vec<Vec<f64>> = self
            .matrix
            .iter()
            .map(|row| row.iter().map(|&x| x as f64).collect())
            .collect();

        json.add_string("type", "ConfusionMatrix")
            .add_number_array_2d("matrix", &matrix_2d)
            .add_string_array("class_names", &self.class_names)
            .add_int("num_classes", self.matrix.len() as i64)
            .add_number("accuracy", self.accuracy());
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// ROC curve visualization with AUC.
pub struct ROCCurve {
    fpr: Vec<f64>, // False positive rate
    tpr: Vec<f64>, // True positive rate
    auc: f64,      // Area under curve
    config: PlotConfig,
}

impl ROCCurve {
    /// Creates a new ROC curve.
    pub fn new(fpr: Vec<f64>, tpr: Vec<f64>) -> Self {
        // Compute AUC using trapezoidal rule
        let auc = Self::compute_auc(&fpr, &tpr);
        Self {
            fpr,
            tpr,
            auc,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new ROC curve with pre-computed AUC.
    pub fn with_auc(fpr: Vec<f64>, tpr: Vec<f64>, auc: f64) -> Self {
        Self {
            fpr,
            tpr,
            auc,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new ROC curve with custom configuration.
    pub fn with_config(fpr: Vec<f64>, tpr: Vec<f64>, auc: f64, config: PlotConfig) -> Self {
        Self {
            fpr,
            tpr,
            auc,
            config,
        }
    }

    /// Computes AUC using trapezoidal rule.
    fn compute_auc(fpr: &[f64], tpr: &[f64]) -> f64 {
        if fpr.len() != tpr.len() || fpr.is_empty() {
            return 0.0;
        }

        let mut auc = 0.0;
        for i in 1..fpr.len() {
            let dx = fpr[i] - fpr[i - 1];
            let avg_y = (tpr[i] + tpr[i - 1]) / 2.0;
            auc += dx * avg_y;
        }
        auc
    }
}

impl Visualization for ROCCurve {
    fn name(&self) -> &str {
        "ROCCurve"
    }

    fn render_svg(&self) -> String {
        let mut svg = SvgBuilder::new(self.config.width, self.config.height);
        let margin = 60.0;
        let width = self.config.width as f32;
        let height = self.config.height as f32;

        // Add title if present
        if let Some(ref title) = self.config.title {
            svg.title(title, self.config.width);
        } else {
            svg.title(
                &format!("ROC Curve (AUC = {:.3})", self.auc),
                self.config.width,
            );
        }

        // Add grid if enabled
        if self.config.show_grid {
            svg.grid(margin, self.config.width, self.config.height, 10, 10);
        }

        // Add axes
        let mut config_with_labels = self.config.clone();
        config_with_labels.x_label = Some("False Positive Rate".to_string());
        config_with_labels.y_label = Some("True Positive Rate".to_string());
        svg.axes(margin, &config_with_labels);

        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        // Draw diagonal reference line (random classifier)
        svg.line(
            margin,
            height - margin,
            width - margin,
            margin,
            "#999999",
            1.0,
        );

        // Plot ROC curve
        if !self.fpr.is_empty() && self.fpr.len() == self.tpr.len() {
            let points: Vec<(f32, f32)> = self
                .fpr
                .iter()
                .zip(self.tpr.iter())
                .map(|(&fpr, &tpr)| {
                    let x = margin + (fpr as f32) * plot_width;
                    let y = height - margin - (tpr as f32) * plot_height;
                    (x, y)
                })
                .collect();

            svg.polyline(&points, "#2563eb", 2.5, "none");

            // Highlight the curve start and end points
            if let Some(&(x, y)) = points.first() {
                svg.circle(x, y, 4.0, "#16a34a");
            }
            if let Some(&(x, y)) = points.last() {
                svg.circle(x, y, 4.0, "#dc2626");
            }
        }

        // Add AUC text box
        let text_x = width - margin - 120.0;
        let text_y = height - margin - 30.0;
        svg.rect_stroke(
            text_x,
            text_y - 20.0,
            110.0,
            30.0,
            "#f0f0f0",
            "#999999",
            1.0,
        );
        svg.text(
            text_x + 5.0,
            text_y - 5.0,
            &format!("AUC = {:.4}", self.auc),
            14,
            "#000000",
        );

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        json.add_string("type", "ROCCurve")
            .add_number_array("fpr", &self.fpr)
            .add_number_array("tpr", &self.tpr)
            .add_number("auc", self.auc)
            .add_int("num_points", self.fpr.len() as i64);
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
    fn test_confusion_matrix() {
        let matrix = vec![vec![50, 2, 3], vec![1, 48, 4], vec![2, 3, 47]];
        let class_names = vec!["Cat".to_string(), "Dog".to_string(), "Bird".to_string()];
        let cm = ConfusionMatrix::with_names(matrix, class_names);

        assert_eq!(cm.name(), "ConfusionMatrix");
        assert_eq!(cm.dimensions(), (800, 600));

        // Check accuracy calculation
        let acc = cm.accuracy();
        assert!((acc - 0.9062).abs() < 0.001); // (50+48+47)/160 ≈ 0.90625

        let svg = cm.render_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Confusion Matrix"));

        let json = cm.render_json();
        assert!(json.contains("ConfusionMatrix"));
        assert!(json.contains("matrix"));
        assert!(json.contains("accuracy"));
    }

    #[test]
    fn test_roc_curve() {
        let fpr = vec![0.0, 0.1, 0.2, 0.4, 0.8, 1.0];
        let tpr = vec![0.0, 0.5, 0.7, 0.9, 0.95, 1.0];
        let roc = ROCCurve::new(fpr, tpr);

        assert_eq!(roc.name(), "ROCCurve");
        assert_eq!(roc.dimensions(), (800, 600));

        // AUC should be computed
        assert!(roc.auc > 0.0);
        assert!(roc.auc <= 1.0);

        let svg = roc.render_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("ROC Curve"));
        assert!(svg.contains("AUC"));

        let json = roc.render_json();
        assert!(json.contains("ROCCurve"));
        assert!(json.contains("fpr"));
        assert!(json.contains("tpr"));
        assert!(json.contains("auc"));
    }

    #[test]
    fn test_roc_auc_calculation() {
        // Perfect classifier: AUC should be 1.0
        let fpr = vec![0.0, 0.0, 1.0];
        let tpr = vec![0.0, 1.0, 1.0];
        let roc = ROCCurve::new(fpr, tpr);
        assert!((roc.auc - 1.0).abs() < 0.01);

        // Random classifier: AUC should be ~0.5
        let fpr = vec![0.0, 0.5, 1.0];
        let tpr = vec![0.0, 0.5, 1.0];
        let roc = ROCCurve::new(fpr, tpr);
        assert!((roc.auc - 0.5).abs() < 0.01);
    }
}
