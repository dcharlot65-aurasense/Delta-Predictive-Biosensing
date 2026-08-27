//! Export utilities for visualization data.
//!
//! This module provides exporters for various formats including SVG, JSON, and CSV,
//! allowing visualization data to be saved and used in external tools.
//!
//! # Examples
//!
//! ```rust
//! use dpb_viz::export::{SvgExporter, JsonExporter, CsvExporter};
//! use std::path::Path;
//!
//! // Export SVG
//! let svg_content = "<svg>...</svg>";
//! # #[cfg(not(test))]
//! SvgExporter::save(svg_content, "output.svg").unwrap();
//!
//! // Export JSON
//! let json_data = serde_json::json!({"key": "value"});
//! # #[cfg(not(test))]
//! JsonExporter::save(&json_data, "output.json").unwrap();
//! ```

use crate::{Result, VizError};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// Scalable Vector Graphics
    Svg,
    /// Portable Network Graphics (requires external tool)
    Png,
    /// JSON format for web rendering
    Json,
    /// CSV format for data analysis
    Csv,
}

impl ExportFormat {
    /// Get the file extension for this format
    pub fn extension(&self) -> &str {
        match self {
            ExportFormat::Svg => "svg",
            ExportFormat::Png => "png",
            ExportFormat::Json => "json",
            ExportFormat::Csv => "csv",
        }
    }

    /// Check if this format is supported
    pub fn is_supported(&self) -> bool {
        match self {
            ExportFormat::Svg | ExportFormat::Json | ExportFormat::Csv => true,
            ExportFormat::Png => false, // Requires external tool
        }
    }
}

/// SVG exporter for vector graphics
pub struct SvgExporter;

impl SvgExporter {
    /// Save SVG content to a file
    pub fn save(content: impl AsRef<str>, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut file = File::create(path)?;
        file.write_all(content.as_ref().as_bytes())?;
        Ok(())
    }

    /// Validate SVG content
    pub fn validate(content: impl AsRef<str>) -> Result<()> {
        let content = content.as_ref();
        if !content.contains("<svg") {
            return Err(VizError::InvalidConfig(
                "Content does not appear to be valid SVG".to_string(),
            ));
        }
        if !content.contains("</svg>") {
            return Err(VizError::InvalidConfig(
                "SVG content is not properly closed".to_string(),
            ));
        }
        Ok(())
    }

    /// Convert SVG to string with proper formatting
    pub fn format(content: impl AsRef<str>) -> String {
        content.as_ref().to_string()
    }

    /// Get the MIME type for SVG
    pub fn mime_type() -> &'static str {
        "image/svg+xml"
    }
}

/// PNG exporter (requires external tool like rsvg-convert or inkscape)
pub struct PngExporter;

impl PngExporter {
    /// Export SVG to PNG using an external tool
    ///
    /// This function will return an error indicating that external tools are needed.
    /// Users should use tools like `rsvg-convert` or `inkscape` to convert SVG to PNG.
    pub fn save(_svg_content: impl AsRef<str>, _output_path: impl AsRef<Path>) -> Result<()> {
        Err(VizError::UnsupportedFormat(
            "PNG export requires external tools. Use rsvg-convert or inkscape to convert SVG files."
                .to_string(),
        ))
    }

    /// Get command line example for converting SVG to PNG
    pub fn conversion_hint() -> &'static str {
        "Use: rsvg-convert -o output.png input.svg"
    }

    /// Get the MIME type for PNG
    pub fn mime_type() -> &'static str {
        "image/png"
    }
}

/// JSON exporter for web rendering and data interchange
pub struct JsonExporter;

impl JsonExporter {
    /// Save JSON data to a file
    pub fn save(data: &serde_json::Value, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, data)?;
        Ok(())
    }

    /// Convert data to pretty-printed JSON string
    pub fn to_string(data: &serde_json::Value) -> Result<String> {
        Ok(serde_json::to_string_pretty(data)?)
    }

    /// Convert data to compact JSON string
    pub fn to_string_compact(data: &serde_json::Value) -> Result<String> {
        Ok(serde_json::to_string(data)?)
    }

    /// Get the MIME type for JSON
    pub fn mime_type() -> &'static str {
        "application/json"
    }

    /// Validate JSON data
    pub fn validate(data: &serde_json::Value) -> Result<()> {
        // Ensure the data can be serialized
        serde_json::to_string(data)?;
        Ok(())
    }
}

/// CSV exporter for data analysis
pub struct CsvExporter;

impl CsvExporter {
    /// Save 2D data as CSV
    pub fn save_matrix(
        data: &[Vec<f32>],
        path: impl AsRef<Path>,
        headers: Option<&[String]>,
    ) -> Result<()> {
        let path = path.as_ref();
        let mut file = File::create(path)?;

        // Write headers if provided
        if let Some(headers) = headers {
            let header_line = headers.join(",");
            writeln!(file, "{}", header_line)?;
        }

        // Write data rows
        for row in data {
            let row_str: Vec<String> = row.iter().map(|v| format!("{}", v)).collect();
            writeln!(file, "{}", row_str.join(","))?;
        }

        Ok(())
    }

    /// Save time series data as CSV
    pub fn save_time_series(
        times: &[f32],
        values: &[f32],
        path: impl AsRef<Path>,
        time_label: &str,
        value_label: &str,
    ) -> Result<()> {
        if times.len() != values.len() {
            return Err(VizError::DimensionMismatch {
                expected: times.len(),
                actual: values.len(),
            });
        }

        let path = path.as_ref();
        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "{},{}", time_label, value_label)?;

        // Write data
        for (time, value) in times.iter().zip(values.iter()) {
            writeln!(file, "{},{}", time, value)?;
        }

        Ok(())
    }

    /// Save spike data as CSV
    pub fn save_spikes(
        spikes: &[(usize, f32)], // (neuron_id, time_ms)
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let path = path.as_ref();
        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "neuron_id,time_ms")?;

        // Write spikes
        for (neuron_id, time_ms) in spikes {
            writeln!(file, "{},{}", neuron_id, time_ms)?;
        }

        Ok(())
    }

    /// Save events as CSV
    pub fn save_events(
        events: &[(f32, String, String)], // (time_ms, event_type, description)
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let path = path.as_ref();
        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "time_ms,event_type,description")?;

        // Write events
        for (time_ms, event_type, description) in events {
            // Escape commas in description
            let escaped_desc = description.replace(',', ";");
            writeln!(file, "{},{},{}", time_ms, event_type, escaped_desc)?;
        }

        Ok(())
    }

    /// Get the MIME type for CSV
    pub fn mime_type() -> &'static str {
        "text/csv"
    }
}

/// Batch exporter for multiple visualizations
pub struct BatchExporter {
    output_dir: String,
    format: ExportFormat,
}

impl BatchExporter {
    /// Create a new batch exporter
    pub fn new(output_dir: impl Into<String>, format: ExportFormat) -> Self {
        Self {
            output_dir: output_dir.into(),
            format,
        }
    }

    /// Export a visualization with automatic naming
    pub fn export(&self, name: &str, content: &str) -> Result<String> {
        let filename = format!("{}.{}", name, self.format.extension());
        let path = Path::new(&self.output_dir).join(filename);

        match self.format {
            ExportFormat::Svg => {
                SvgExporter::save(content, &path)?;
            }
            ExportFormat::Json => {
                let json: serde_json::Value = serde_json::from_str(content)?;
                JsonExporter::save(&json, &path)?;
            }
            ExportFormat::Png => {
                return Err(VizError::UnsupportedFormat(
                    "PNG export not supported in batch mode".to_string(),
                ));
            }
            ExportFormat::Csv => {
                let mut file = File::create(&path)?;
                file.write_all(content.as_bytes())?;
            }
        }

        Ok(path.to_string_lossy().to_string())
    }

    /// Get the output directory
    pub fn output_dir(&self) -> &str {
        &self.output_dir
    }

    /// Get the export format
    pub fn format(&self) -> ExportFormat {
        self.format
    }
}

/// Utility functions for export operations
pub struct ExportUtils;

impl ExportUtils {
    /// Sanitize filename by removing invalid characters
    pub fn sanitize_filename(name: &str) -> String {
        name.chars()
            .map(|c| match c {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => c,
            })
            .collect()
    }

    /// Generate timestamped filename
    pub fn timestamped_filename(base: &str, extension: &str) -> String {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        format!("{}_{}.{}", base, timestamp, extension)
    }

    /// Check if path is writable
    pub fn is_writable(path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            parent.exists() && parent.is_dir()
        } else {
            false
        }
    }

    /// Create output directory if it doesn't exist
    pub fn ensure_output_dir(path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_export_format() {
        assert_eq!(ExportFormat::Svg.extension(), "svg");
        assert_eq!(ExportFormat::Png.extension(), "png");
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::Csv.extension(), "csv");

        assert!(ExportFormat::Svg.is_supported());
        assert!(ExportFormat::Json.is_supported());
        assert!(ExportFormat::Csv.is_supported());
        assert!(!ExportFormat::Png.is_supported());
    }

    #[test]
    fn test_svg_exporter() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.svg");

        let svg_content = r#"<svg width="100" height="100"><circle cx="50" cy="50" r="40"/></svg>"#;

        SvgExporter::save(svg_content, &file_path).unwrap();
        assert!(file_path.exists());

        // Validate
        assert!(SvgExporter::validate(svg_content).is_ok());
        assert!(SvgExporter::validate("not svg").is_err());
        assert!(SvgExporter::validate("<svg>not closed").is_err());
    }

    #[test]
    fn test_svg_mime_type() {
        assert_eq!(SvgExporter::mime_type(), "image/svg+xml");
    }

    #[test]
    fn test_png_exporter() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.png");

        let svg_content = "<svg></svg>";
        let result = PngExporter::save(svg_content, &file_path);
        assert!(result.is_err());

        assert!(PngExporter::conversion_hint().contains("rsvg-convert"));
        assert_eq!(PngExporter::mime_type(), "image/png");
    }

    #[test]
    fn test_json_exporter() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.json");

        let data = serde_json::json!({
            "name": "test",
            "value": 42,
            "array": [1, 2, 3]
        });

        JsonExporter::save(&data, &file_path).unwrap();
        assert!(file_path.exists());

        // Test string conversion
        let json_str = JsonExporter::to_string(&data).unwrap();
        assert!(json_str.contains("test"));

        let compact = JsonExporter::to_string_compact(&data).unwrap();
        assert!(!compact.contains("\n"));

        // Validate
        assert!(JsonExporter::validate(&data).is_ok());
    }

    #[test]
    fn test_json_mime_type() {
        assert_eq!(JsonExporter::mime_type(), "application/json");
    }

    #[test]
    fn test_csv_exporter_matrix() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.csv");

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let headers = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        CsvExporter::save_matrix(&data, &file_path, Some(&headers)).unwrap();
        assert!(file_path.exists());

        // Read and verify
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("A,B,C"));
        assert!(content.contains("1,2,3"));
    }

    #[test]
    fn test_csv_exporter_time_series() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("timeseries.csv");

        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![10.0, 20.0, 15.0, 25.0];

        CsvExporter::save_time_series(&times, &values, &file_path, "Time", "Value").unwrap();
        assert!(file_path.exists());

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("Time,Value"));
        assert!(content.contains("0,10"));
    }

    #[test]
    fn test_csv_exporter_time_series_mismatch() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("timeseries.csv");

        let times = vec![0.0, 1.0, 2.0];
        let values = vec![10.0, 20.0]; // Wrong length

        let result = CsvExporter::save_time_series(&times, &values, &file_path, "Time", "Value");
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_exporter_spikes() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("spikes.csv");

        let spikes = vec![(0, 10.5), (5, 15.2), (3, 20.8)];

        CsvExporter::save_spikes(&spikes, &file_path).unwrap();
        assert!(file_path.exists());

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("neuron_id,time_ms"));
        assert!(content.contains("0,10.5"));
    }

    #[test]
    fn test_csv_exporter_events() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("events.csv");

        let events = vec![
            (10.0, "spike".to_string(), "Neuron fired".to_string()),
            (20.0, "update".to_string(), "Weight changed".to_string()),
        ];

        CsvExporter::save_events(&events, &file_path).unwrap();
        assert!(file_path.exists());

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("time_ms,event_type,description"));
        assert!(content.contains("spike"));
    }

    #[test]
    fn test_csv_mime_type() {
        assert_eq!(CsvExporter::mime_type(), "text/csv");
    }

    #[test]
    fn test_batch_exporter() {
        let temp_dir = TempDir::new().unwrap();
        let exporter = BatchExporter::new(temp_dir.path().to_str().unwrap(), ExportFormat::Svg);

        assert_eq!(exporter.format(), ExportFormat::Svg);
        assert_eq!(exporter.output_dir(), temp_dir.path().to_str().unwrap());

        let svg_content = "<svg></svg>";
        let path = exporter.export("test", svg_content).unwrap();
        assert!(Path::new(&path).exists());
    }

    #[test]
    fn test_batch_exporter_unsupported() {
        let temp_dir = TempDir::new().unwrap();
        let exporter = BatchExporter::new(temp_dir.path().to_str().unwrap(), ExportFormat::Png);

        let result = exporter.export("test", "content");
        assert!(result.is_err());
    }

    #[test]
    fn test_export_utils_sanitize_filename() {
        assert_eq!(
            ExportUtils::sanitize_filename("test/file:name*.svg"),
            "test_file_name_.svg"
        );
        assert_eq!(ExportUtils::sanitize_filename("normal.svg"), "normal.svg");
    }

    #[test]
    fn test_export_utils_timestamped_filename() {
        let filename = ExportUtils::timestamped_filename("test", "svg");
        assert!(filename.starts_with("test_"));
        assert!(filename.ends_with(".svg"));
    }

    #[test]
    fn test_export_utils_is_writable() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        assert!(ExportUtils::is_writable(&file_path));
        assert!(!ExportUtils::is_writable("/nonexistent/path/file.txt"));
    }

    #[test]
    fn test_export_utils_ensure_output_dir() {
        let temp_dir = TempDir::new().unwrap();
        let new_dir = temp_dir.path().join("new_dir");

        assert!(!new_dir.exists());
        ExportUtils::ensure_output_dir(&new_dir).unwrap();
        assert!(new_dir.exists());
    }
}
