//! Spike train visualization utilities.

use super::{colors, export::{JsonBuilder, SvgBuilder}, PlotConfig, Visualization};

/// Spike raster plot (neuron x time).
pub struct RasterPlot {
    spikes: Vec<Vec<f64>>, // Vec of spike times for each neuron
    duration: f64,
    config: PlotConfig,
}

impl RasterPlot {
    /// Creates a new raster plot.
    pub fn new(spikes: Vec<Vec<f64>>, duration: f64) -> Self {
        Self {
            spikes,
            duration,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new raster plot with custom configuration.
    pub fn with_config(spikes: Vec<Vec<f64>>, duration: f64, config: PlotConfig) -> Self {
        Self {
            spikes,
            duration,
            config,
        }
    }
}

impl Visualization for RasterPlot {
    fn name(&self) -> &str {
        "RasterPlot"
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

        // Plot raster
        if !self.spikes.is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;
            let neuron_height = plot_height / self.spikes.len() as f32;

            for (neuron_idx, spike_times) in self.spikes.iter().enumerate() {
                let y_center = height - margin - (neuron_idx as f32 + 0.5) * neuron_height;
                for &spike_time in spike_times {
                    if spike_time >= 0.0 && spike_time <= self.duration {
                        let x = margin + (spike_time / self.duration) as f32 * plot_width;
                        // Draw a vertical tick for each spike
                        let tick_height = neuron_height * 0.8;
                        svg.line(
                            x,
                            y_center - tick_height / 2.0,
                            x,
                            y_center + tick_height / 2.0,
                            "#000000",
                            1.5,
                        );
                    }
                }
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        json.add_string("type", "RasterPlot")
            .add_number("duration", self.duration)
            .add_number_array_2d("spikes", &self.spikes)
            .add_int("num_neurons", self.spikes.len() as i64);
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Spike count histogram.
pub struct SpikeHistogram {
    spike_counts: Vec<usize>,
    bin_labels: Vec<String>,
    config: PlotConfig,
}

impl SpikeHistogram {
    /// Creates a new spike histogram.
    pub fn new(spike_counts: Vec<usize>) -> Self {
        let bin_labels = (0..spike_counts.len())
            .map(|i| format!("Bin {}", i))
            .collect();
        Self {
            spike_counts,
            bin_labels,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new spike histogram with custom bin labels.
    pub fn with_labels(spike_counts: Vec<usize>, bin_labels: Vec<String>) -> Self {
        Self {
            spike_counts,
            bin_labels,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new spike histogram with custom configuration.
    pub fn with_config(spike_counts: Vec<usize>, bin_labels: Vec<String>, config: PlotConfig) -> Self {
        Self {
            spike_counts,
            bin_labels,
            config,
        }
    }
}

impl Visualization for SpikeHistogram {
    fn name(&self) -> &str {
        "SpikeHistogram"
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

        // Plot histogram
        if !self.spike_counts.is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            let max_count = *self.spike_counts.iter().max().unwrap_or(&1);
            let bar_width = plot_width / self.spike_counts.len() as f32 * 0.8;
            let bar_spacing = plot_width / self.spike_counts.len() as f32;

            for (i, &count) in self.spike_counts.iter().enumerate() {
                let bar_height = (count as f32 / max_count as f32) * plot_height;
                let x = margin + i as f32 * bar_spacing + bar_spacing * 0.1;
                let y = height - margin - bar_height;

                svg.rect(x, y, bar_width, bar_height, "#2563eb");

                // Add count label on top of bar
                svg.text_anchor(
                    x + bar_width / 2.0,
                    y - 5.0,
                    &count.to_string(),
                    10,
                    "#000000",
                    "middle",
                );
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        let counts: Vec<f64> = self.spike_counts.iter().map(|&x| x as f64).collect();
        json.add_string("type", "SpikeHistogram")
            .add_number_array("counts", &counts)
            .add_string_array("labels", &self.bin_labels);
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Inter-spike interval (ISI) distribution histogram.
pub struct ISIHistogram {
    intervals: Vec<f64>, // ISI values in seconds
    num_bins: usize,
    config: PlotConfig,
}

impl ISIHistogram {
    /// Creates a new ISI histogram.
    pub fn new(intervals: Vec<f64>, num_bins: usize) -> Self {
        Self {
            intervals,
            num_bins,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new ISI histogram with custom configuration.
    pub fn with_config(intervals: Vec<f64>, num_bins: usize, config: PlotConfig) -> Self {
        Self {
            intervals,
            num_bins,
            config,
        }
    }

    /// Computes histogram bins from intervals.
    fn compute_histogram(&self) -> (Vec<usize>, f64, f64) {
        if self.intervals.is_empty() {
            return (vec![0; self.num_bins], 0.0, 1.0);
        }

        let min_isi = self.intervals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_isi = self.intervals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max_isi - min_isi).abs() < 1e-9 {
            1.0
        } else {
            max_isi - min_isi
        };

        let mut bins = vec![0; self.num_bins];
        for &isi in &self.intervals {
            let bin_idx = (((isi - min_isi) / range) * self.num_bins as f64) as usize;
            let bin_idx = bin_idx.min(self.num_bins - 1);
            bins[bin_idx] += 1;
        }

        (bins, min_isi, max_isi)
    }
}

impl Visualization for ISIHistogram {
    fn name(&self) -> &str {
        "ISIHistogram"
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

        // Compute and plot histogram
        let (bins, _min_isi, _max_isi) = self.compute_histogram();
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        if !bins.is_empty() {
            let max_count = *bins.iter().max().unwrap_or(&1);
            let bar_width = plot_width / bins.len() as f32 * 0.9;
            let bar_spacing = plot_width / bins.len() as f32;

            for (i, &count) in bins.iter().enumerate() {
                if count > 0 {
                    let bar_height = (count as f32 / max_count as f32) * plot_height;
                    let x = margin + i as f32 * bar_spacing + bar_spacing * 0.05;
                    let y = height - margin - bar_height;

                    svg.rect(x, y, bar_width, bar_height, "#16a34a");
                }
            }
        }

        svg.build()
    }

    fn render_json(&self) -> String {
        let mut json = JsonBuilder::new();
        let (bins, min_isi, max_isi) = self.compute_histogram();
        let bin_counts: Vec<f64> = bins.iter().map(|&x| x as f64).collect();

        json.add_string("type", "ISIHistogram")
            .add_number_array("intervals", &self.intervals)
            .add_number_array("bin_counts", &bin_counts)
            .add_number("min_isi", min_isi)
            .add_number("max_isi", max_isi)
            .add_int("num_bins", self.num_bins as i64);
        json.build()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

/// Firing rate heatmap over time and neurons.
pub struct FiringRateHeatmap {
    firing_rates: Vec<Vec<f32>>, // [time_bin][neuron]
    time_bins: Vec<f64>,
    config: PlotConfig,
}

impl FiringRateHeatmap {
    /// Creates a new firing rate heatmap.
    pub fn new(firing_rates: Vec<Vec<f32>>, time_bins: Vec<f64>) -> Self {
        Self {
            firing_rates,
            time_bins,
            config: PlotConfig::default(),
        }
    }

    /// Creates a new firing rate heatmap with custom configuration.
    pub fn with_config(firing_rates: Vec<Vec<f32>>, time_bins: Vec<f64>, config: PlotConfig) -> Self {
        Self {
            firing_rates,
            time_bins,
            config,
        }
    }
}

impl Visualization for FiringRateHeatmap {
    fn name(&self) -> &str {
        "FiringRateHeatmap"
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
        if !self.firing_rates.is_empty() && !self.firing_rates[0].is_empty() {
            let plot_width = width - 2.0 * margin;
            let plot_height = height - 2.0 * margin;

            let time_bins = self.firing_rates.len();
            let num_neurons = self.firing_rates[0].len();

            // Find min/max for normalization
            let mut min_rate = f32::INFINITY;
            let mut max_rate = f32::NEG_INFINITY;
            for row in &self.firing_rates {
                for &rate in row {
                    min_rate = min_rate.min(rate);
                    max_rate = max_rate.max(rate);
                }
            }
            let range = if (max_rate - min_rate).abs() < 1e-6 {
                1.0
            } else {
                max_rate - min_rate
            };

            let cell_width = plot_width / time_bins as f32;
            let cell_height = plot_height / num_neurons as f32;

            for (t, time_slice) in self.firing_rates.iter().enumerate() {
                for (n, &rate) in time_slice.iter().enumerate() {
                    let x = margin + t as f32 * cell_width;
                    let y = height - margin - (n + 1) as f32 * cell_height;
                    let normalized = ((rate - min_rate) / range).clamp(0.0, 1.0);
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
        let rates_2d: Vec<Vec<f64>> = self
            .firing_rates
            .iter()
            .map(|row| row.iter().map(|&x| x as f64).collect())
            .collect();

        json.add_string("type", "FiringRateHeatmap")
            .add_number_array_2d("firing_rates", &rates_2d)
            .add_number_array("time_bins", &self.time_bins)
            .add_int("num_time_bins", self.firing_rates.len() as i64)
            .add_int(
                "num_neurons",
                if self.firing_rates.is_empty() { 0 } else { self.firing_rates[0].len() as i64 },
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
    fn test_raster_plot() {
        let spikes = vec![
            vec![0.1, 0.3, 0.5],
            vec![0.2, 0.4],
            vec![0.15, 0.35, 0.55, 0.75],
        ];
        let plot = RasterPlot::new(spikes, 1.0);
        assert_eq!(plot.name(), "RasterPlot");
        assert_eq!(plot.dimensions(), (800, 600));

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("RasterPlot"));
        assert!(json.contains("duration"));
    }

    #[test]
    fn test_spike_histogram() {
        let counts = vec![10, 25, 30, 15, 5];
        let plot = SpikeHistogram::new(counts);
        assert_eq!(plot.name(), "SpikeHistogram");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("SpikeHistogram"));
        assert!(json.contains("counts"));
    }

    #[test]
    fn test_isi_histogram() {
        let intervals = vec![0.01, 0.02, 0.015, 0.025, 0.012, 0.018, 0.022];
        let plot = ISIHistogram::new(intervals, 10);
        assert_eq!(plot.name(), "ISIHistogram");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("ISIHistogram"));
        assert!(json.contains("intervals"));
    }

    #[test]
    fn test_firing_rate_heatmap() {
        let firing_rates = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![1.5, 2.5, 3.5],
        ];
        let time_bins = vec![0.0, 0.1, 0.2];
        let plot = FiringRateHeatmap::new(firing_rates, time_bins);
        assert_eq!(plot.name(), "FiringRateHeatmap");

        let svg = plot.render_svg();
        assert!(svg.contains("<svg"));

        let json = plot.render_json();
        assert!(json.contains("FiringRateHeatmap"));
        assert!(json.contains("firing_rates"));
    }
}
