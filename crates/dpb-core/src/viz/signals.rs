//! Signal visualization utilities.

use super::{
    PlotConfig, Visualization, colors,
    export::{JsonBuilder, SvgBuilder},
};

/// Time series waveform plot.
pub struct SignalPlot {
    data: Vec<f32>,
    sample_rate: f64,
    config: PlotConfig,
}

impl SignalPlot {
    /// Creates a new signal plot.
    pub fn new(data: Vec<f32>, sample_rate: f64) -> Self {
        Self {
            data,
            sample_rate,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new signal plot with custom configuration.
    pub fn with_config(data: Vec<f32>, sample_rate: f64, config: PlotConfig) -> Self {
        Self {
            data,
            sample_rate,
            config,
        }
    }
}

impl Visualization for SignalPlot {
    fn name(&self) -> &str {
        "SignalPlot"
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

        // Plot data
        if !self.data.is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            let min_val = self.data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = if (max_val - min_val).abs() < 1e-6 {
                1.0
            } else {
                max_val - min_val
            };

            let points: Vec<(f32, f32)> = self
                .data
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let x = margin + (i as f32 / (self.data.len() - 1) as f32) * plot_width;
                    let normalized = (val - min_val) / range;
                    let y = height - margin - normalized * plot_height;
                    (x, y)
                })
                .collect();

            svg.polyline(&points, "#2563eb", 2.0, "none");
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        json.add_string("type", "SignalPlot")
            .add_number("sample_rate", self.sample_rate)
            .add_number_array(
                "data",
                &self.data.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            )
            .add_int("num_samples", self.data.len() as i64);
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Time-frequency spectrogram plot.
pub struct SpectrogramPlot {
    data: Vec<Vec<f32>>, // [time][frequency]
    sample_rate: f64,
    config: PlotConfig,
}

impl SpectrogramPlot {
    /// Creates a new spectrogram plot.
    pub fn new(data: Vec<Vec<f32>>, sample_rate: f64) -> Self {
        Self {
            data,
            sample_rate,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new spectrogram plot with custom configuration.
    pub fn with_config(data: Vec<Vec<f32>>, sample_rate: f64, config: PlotConfig) -> Self {
        Self {
            data,
            sample_rate,
            config,
        }
    }
}

impl Visualization for SpectrogramPlot {
    fn name(&self) -> &str {
        "SpectrogramPlot"
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

        // Plot spectrogram
        if !self.data.is_empty() && !self.data[0].is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            let time_bins = self.data.len();
            let freq_bins = self.data[0].len();

            // Find min/max for normalization
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for row in &self.data {
                for &val in row {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
            let range = if (max_val - min_val).abs() < 1e-6 {
                1.0
            } else {
                max_val - min_val
            };

            let cell_width = plot_width / time_bins as f32;
            let cell_height = plot_height / freq_bins as f32;

            for (t, time_slice) in self.data.iter().enumerate() {
                for (f, &val) in time_slice.iter().enumerate() {
                    let x = margin + t as f32 * cell_width;
                    let y = height - margin - (f + 1) as f32 * cell_height;
                    let normalized = ((val - min_val) / range).clamp(0.0, 1.0);
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
        let data_2d: Vec<Vec<f64>> = self
            .data
            .iter()
            .map(|row| row.iter().map(|&x| x as f64).collect())
            .collect();

        json.add_string("type", "SpectrogramPlot")
            .add_number("sample_rate", self.sample_rate)
            .add_number_array_2d("data", &data_2d)
            .add_int("time_bins", self.data.len() as i64)
            .add_int(
                "freq_bins",
                if self.data.is_empty() {
                    0
                } else {
                    self.data[0].len() as i64
                },
            );
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Multi-channel signal overlay plot.
pub struct MultiChannelPlot {
    channels: Vec<Vec<f32>>,
    sample_rate: f64,
    channel_names: Vec<String>,
    config: PlotConfig,
}

impl MultiChannelPlot {
    /// Creates a new multi-channel plot.
    pub fn new(channels: Vec<Vec<f32>>, sample_rate: f64) -> Self {
        let channel_names = (0..channels.len())
            .map(|i| format!("Channel {}", i))
            .collect();
        Self {
            channels,
            sample_rate,
            channel_names,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new multi-channel plot with channel names.
    pub fn with_names(
        channels: Vec<Vec<f32>>,
        sample_rate: f64,
        channel_names: Vec<String>,
    ) -> Self {
        Self {
            channels,
            sample_rate,
            channel_names,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new multi-channel plot with custom configuration.
    pub fn with_config(
        channels: Vec<Vec<f32>>,
        sample_rate: f64,
        channel_names: Vec<String>,
        config: PlotConfig,
    ) -> Self {
        Self {
            channels,
            sample_rate,
            channel_names,
            config,
        }
    }
}

impl Visualization for MultiChannelPlot {
    fn name(&self) -> &str {
        "MultiChannelPlot"
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

        // Plot channels
        let colors_palette = [
            "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea", "#0891b2",
        ];

        for (ch_idx, channel) in self.channels.iter().enumerate() {
            if !channel.is_empty() {
                let plot_width = width - 2.0 * margin;
                let plot_height = height - 2.0 * margin;

                // Find global min/max across all channels for consistent scaling
                let mut min_val = f32::INFINITY;
                let mut max_val = f32::NEG_INFINITY;
                for ch in &self.channels {
                    for &val in ch {
                        min_val = min_val.min(val);
                        max_val = max_val.max(val);
                    }
                }
                let range = if (max_val - min_val).abs() < 1e-6 {
                    1.0
                } else {
                    max_val - min_val
                };

                let points: Vec<(f32, f32)> = channel
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        let x = margin + (i as f32 / (channel.len() - 1) as f32) * plot_width;
                        let normalized = (val - min_val) / range;
                        let y = height - margin - normalized * plot_height;
                        (x, y)
                    })
                    .collect();

                let color = colors_palette[ch_idx % colors_palette.len()];
                svg.polyline(&points, color, 2.0, "none");
            }
        }

        // Add legend if enabled
        if self.config.show_legend && !self.channel_names.is_empty() {
            let legend_x = width - margin - 100.0;
            let mut legend_y = margin + 20.0;
            for (idx, name) in self.channel_names.iter().enumerate() {
                let color = colors_palette[idx % colors_palette.len()];
                svg.line(legend_x, legend_y, legend_x + 20.0, legend_y, color, 2.0);
                svg.text(legend_x + 25.0, legend_y + 5.0, name, 12, "#000000");
                legend_y += 20.0;
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        let channels_2d: Vec<Vec<f64>> = self
            .channels
            .iter()
            .map(|ch| ch.iter().map(|&x| x as f64).collect())
            .collect();

        json.add_string("type", "MultiChannelPlot")
            .add_number("sample_rate", self.sample_rate)
            .add_number_array_2d("channels", &channels_2d)
            .add_string_array("channel_names", &self.channel_names)
            .add_int("num_channels", self.channels.len() as i64);
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Signal plot with event markers.
pub struct AnnotatedSignalPlot {
    data: Vec<f32>,
    sample_rate: f64,
    events: Vec<(usize, String)>, // (sample_index, label)
    config: PlotConfig,
}

impl AnnotatedSignalPlot {
    /// Creates a new annotated signal plot.
    pub fn new(data: Vec<f32>, sample_rate: f64, events: Vec<(usize, String)>) -> Self {
        Self {
            data,
            sample_rate,
            events,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new annotated signal plot with custom configuration.
    pub fn with_config(
        data: Vec<f32>,
        sample_rate: f64,
        events: Vec<(usize, String)>,
        config: PlotConfig,
    ) -> Self {
        Self {
            data,
            sample_rate,
            events,
            config,
        }
    }
}

impl Visualization for AnnotatedSignalPlot {
    fn name(&self) -> &str {
        "AnnotatedSignalPlot"
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

        // Plot signal
        if !self.data.is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            let min_val = self.data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = if (max_val - min_val).abs() < 1e-6 {
                1.0
            } else {
                max_val - min_val
            };

            let points: Vec<(f32, f32)> = self
                .data
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let x = margin + (i as f32 / (self.data.len() - 1) as f32) * plot_width;
                    let normalized = (val - min_val) / range;
                    let y = height - margin - normalized * plot_height;
                    (x, y)
                })
                .collect();

            svg.polyline(&points, "#2563eb", 2.0, "none");

            // Add event markers
            for (idx, label) in &self.events {
                if *idx < self.data.len() {
                    let x = margin + (*idx as f32 / (self.data.len() - 1) as f32) * plot_width;
                    // Draw vertical line
                    svg.line(x, margin, x, height - margin, "#dc2626", 1.5);
                    // Draw circle at the signal point
                    let val = self.data[*idx];
                    let normalized = (val - min_val) / range;
                    let y = height - margin - normalized * plot_height;
                    svg.circle(x, y, 4.0, "#dc2626");
                    // Add label
                    svg.text(x + 5.0, margin + 15.0, label, 10, "#dc2626");
                }
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        let events_json = format!(
            "[{}]",
            self.events
                .iter()
                .map(|(idx, label)| format!(r#"{{"index": {}, "label": "{}"}}"#, idx, label))
                .collect::<Vec<_>>()
                .join(", ")
        );

        json.add_string("type", "AnnotatedSignalPlot")
            .add_number("sample_rate", self.sample_rate)
            .add_number_array(
                "data",
                &self.data.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            )
            .add_raw("events", &events_json);
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
    fn test_signal_plot() {
        let data = vec![0.0, 1.0, 0.5, -0.5, -1.0, 0.0];
        let plot = SignalPlot::new(data, 1000.0);
        assert_eq!(plot.name(), "SignalPlot");
        assert_eq!(plot.dimensions(), (800, 600));

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));

        let json = plot.render_json();
        assert!(json.contains("SignalPlot"));
        assert!(json.contains("sample_rate"));
    }

    #[test]
    fn test_spectrogram_plot() {
        let data = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];
        let plot = SpectrogramPlot::new(data, 1000.0);
        assert_eq!(plot.name(), "SpectrogramPlot");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("SpectrogramPlot"));
    }

    #[test]
    fn test_multi_channel_plot() {
        let channels = vec![vec![0.0, 1.0, 0.5], vec![1.0, 0.5, 0.0]];
        let plot = MultiChannelPlot::new(channels, 1000.0);
        assert_eq!(plot.name(), "MultiChannelPlot");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("MultiChannelPlot"));
    }

    #[test]
    fn test_annotated_signal_plot() {
        let data = vec![0.0, 1.0, 0.5, -0.5, -1.0, 0.0];
        let events = vec![(1, "Peak".to_string()), (4, "Trough".to_string())];
        let plot = AnnotatedSignalPlot::new(data, 1000.0, events);
        assert_eq!(plot.name(), "AnnotatedSignalPlot");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Peak"));
        assert!(svg.contains("Trough"));

        let json = plot.render_json();
        assert!(json.contains("AnnotatedSignalPlot"));
        assert!(json.contains("events"));
    }
}
