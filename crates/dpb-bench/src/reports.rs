//! Benchmark report generation and export
//!
//! This module provides utilities for generating and exporting benchmark reports
//! in various formats (JSON, CSV, Markdown).

use crate::baselines::{BaselineResult, ComparisonMetrics};
use crate::profiling::ProfileResult;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Report output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// Markdown format
    Markdown,
}

/// Complete benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Summary statistics
    pub summary: BenchmarkSummary,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report title
    pub title: String,
    /// Timestamp
    pub timestamp: String,
    /// DPB version
    pub dpb_version: String,
    /// System information
    pub system_info: SystemInfo,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// Architecture
    pub arch: String,
    /// Number of CPU cores
    pub num_cores: usize,
    /// Rust version
    pub rust_version: String,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            num_cores: num_cpus::get(),
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
        }
    }
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Category
    pub category: String,
    /// Profile results
    pub profile: ProfileResult,
    /// Baseline comparison (if available)
    pub baseline: Option<BaselineResult>,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Benchmark summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Total benchmarks run
    pub total_benchmarks: usize,
    /// Total time elapsed (ms)
    pub total_time_ms: f64,
    /// Average latency (ms)
    pub avg_latency_ms: f64,
    /// Total energy (mJ)
    pub total_energy_mj: f64,
    /// Average sparsity
    pub avg_sparsity: f64,
    /// Peak memory (MB)
    pub peak_memory_mb: f64,
}

impl BenchmarkReport {
    /// Create a new benchmark report
    pub fn new(title: &str) -> Self {
        let timestamp = Utc::now().to_rfc3339();

        Self {
            metadata: ReportMetadata {
                title: title.to_string(),
                timestamp,
                dpb_version: env!("CARGO_PKG_VERSION").to_string(),
                system_info: SystemInfo::default(),
                metadata: HashMap::new(),
            },
            results: Vec::new(),
            summary: BenchmarkSummary {
                total_benchmarks: 0,
                total_time_ms: 0.0,
                avg_latency_ms: 0.0,
                total_energy_mj: 0.0,
                avg_sparsity: 0.0,
                peak_memory_mb: 0.0,
            },
        }
    }

    /// Add a benchmark result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
        self.update_summary();
    }

    /// Update summary statistics
    fn update_summary(&mut self) {
        self.summary.total_benchmarks = self.results.len();
        self.summary.total_time_ms = self.results.iter().map(|r| r.profile.elapsed_ms).sum();
        self.summary.avg_latency_ms = if !self.results.is_empty() {
            self.summary.total_time_ms / self.results.len() as f64
        } else {
            0.0
        };
        self.summary.total_energy_mj = self.results.iter().map(|r| r.profile.energy_mj).sum();
        self.summary.avg_sparsity = if !self.results.is_empty() {
            self.results.iter().map(|r| r.profile.sparsity).sum::<f64>() / self.results.len() as f64
        } else {
            0.0
        };
        self.summary.peak_memory_mb = self.results.iter()
            .map(|r| r.profile.memory_bytes as f64 / 1_048_576.0)
            .fold(0.0f64, f64::max);
    }

    /// Export report to file
    pub fn export<P: AsRef<Path>>(&self, path: P, format: ReportFormat) -> anyhow::Result<()> {
        match format {
            ReportFormat::JSON => self.export_json(path),
            ReportFormat::CSV => self.export_csv(path),
            ReportFormat::Markdown => self.export_markdown(path),
        }
    }

    /// Export as JSON
    fn export_json<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Export as CSV
    fn export_csv<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "Name,Category,Elapsed(ms),Memory(bytes),Spikes,Sparsity,Energy(mJ)")?;

        // Write results
        for result in &self.results {
            writeln!(
                file,
                "{},{},{:.3},{},{},{:.4},{:.6}",
                result.name,
                result.category,
                result.profile.elapsed_ms,
                result.profile.memory_bytes,
                result.profile.num_spikes,
                result.profile.sparsity,
                result.profile.energy_mj
            )?;
        }

        Ok(())
    }

    /// Export as Markdown
    fn export_markdown<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        // Write header
        writeln!(file, "# {}", self.metadata.title)?;
        writeln!(file)?;
        writeln!(file, "**Timestamp:** {}", self.metadata.timestamp)?;
        writeln!(file, "**DPB Version:** {}", self.metadata.dpb_version)?;
        writeln!(file)?;

        // Write system info
        writeln!(file, "## System Information")?;
        writeln!(file)?;
        writeln!(file, "- **OS:** {}", self.metadata.system_info.os)?;
        writeln!(file, "- **Architecture:** {}", self.metadata.system_info.arch)?;
        writeln!(file, "- **CPU Cores:** {}", self.metadata.system_info.num_cores)?;
        writeln!(file, "- **Rust Version:** {}", self.metadata.system_info.rust_version)?;
        writeln!(file)?;

        // Write summary
        writeln!(file, "## Summary")?;
        writeln!(file)?;
        writeln!(file, "- **Total Benchmarks:** {}", self.summary.total_benchmarks)?;
        writeln!(file, "- **Total Time:** {:.2} ms", self.summary.total_time_ms)?;
        writeln!(file, "- **Average Latency:** {:.2} ms", self.summary.avg_latency_ms)?;
        writeln!(file, "- **Total Energy:** {:.3} mJ", self.summary.total_energy_mj)?;
        writeln!(file, "- **Average Sparsity:** {:.2}%", self.summary.avg_sparsity * 100.0)?;
        writeln!(file, "- **Peak Memory:** {:.2} MB", self.summary.peak_memory_mb)?;
        writeln!(file)?;

        // Write detailed results
        writeln!(file, "## Detailed Results")?;
        writeln!(file)?;
        writeln!(file, "| Name | Category | Time (ms) | Memory (KB) | Spikes | Sparsity | Energy (mJ) |")?;
        writeln!(file, "|------|----------|-----------|-------------|--------|----------|-------------|")?;

        for result in &self.results {
            writeln!(
                file,
                "| {} | {} | {:.2} | {:.1} | {} | {:.2}% | {:.3} |",
                result.name,
                result.category,
                result.profile.elapsed_ms,
                result.profile.memory_bytes as f64 / 1024.0,
                result.profile.num_spikes,
                result.profile.sparsity * 100.0,
                result.profile.energy_mj
            )?;
        }

        writeln!(file)?;

        Ok(())
    }
}

/// Comparison report between SNN and baselines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// SNN results
    pub snn_results: Vec<BaselineResult>,
    /// ANN results
    pub ann_results: Vec<BaselineResult>,
    /// Conventional results
    pub conventional_results: Vec<BaselineResult>,
    /// Comparison metrics
    pub comparisons: Vec<ComparisonEntry>,
}

/// Entry in comparison report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonEntry {
    /// Benchmark name
    pub name: String,
    /// SNN vs ANN comparison
    pub snn_vs_ann: Option<ComparisonMetrics>,
    /// SNN vs Conventional comparison
    pub snn_vs_conventional: Option<ComparisonMetrics>,
}

impl ComparisonReport {
    /// Create a new comparison report
    pub fn new(title: &str) -> Self {
        let timestamp = Utc::now().to_rfc3339();

        Self {
            metadata: ReportMetadata {
                title: title.to_string(),
                timestamp,
                dpb_version: env!("CARGO_PKG_VERSION").to_string(),
                system_info: SystemInfo::default(),
                metadata: HashMap::new(),
            },
            snn_results: Vec::new(),
            ann_results: Vec::new(),
            conventional_results: Vec::new(),
            comparisons: Vec::new(),
        }
    }

    /// Add comparison
    pub fn add_comparison(
        &mut self,
        name: &str,
        snn: BaselineResult,
        ann: Option<BaselineResult>,
        conventional: Option<BaselineResult>,
    ) {
        self.snn_results.push(snn.clone());

        let snn_vs_ann = ann.as_ref().map(|a| {
            self.ann_results.push(a.clone());
            ComparisonMetrics::compute(&snn, a)
        });

        let snn_vs_conventional = conventional.as_ref().map(|c| {
            self.conventional_results.push(c.clone());
            ComparisonMetrics::compute(&snn, c)
        });

        self.comparisons.push(ComparisonEntry {
            name: name.to_string(),
            snn_vs_ann,
            snn_vs_conventional,
        });
    }

    /// Export report
    pub fn export<P: AsRef<Path>>(&self, path: P, format: ReportFormat) -> anyhow::Result<()> {
        match format {
            ReportFormat::JSON => self.export_json(path),
            ReportFormat::Markdown => self.export_markdown(path),
            _ => Err(anyhow::anyhow!("Format not supported for comparison report")),
        }
    }

    fn export_json<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    fn export_markdown<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "# {}", self.metadata.title)?;
        writeln!(file)?;
        writeln!(file, "**Timestamp:** {}", self.metadata.timestamp)?;
        writeln!(file)?;

        writeln!(file, "## SNN vs ANN Comparison")?;
        writeln!(file)?;
        writeln!(file, "| Benchmark | Accuracy Δ | Speedup | Energy Efficiency |")?;
        writeln!(file, "|-----------|------------|---------|-------------------|")?;

        for comp in &self.comparisons {
            if let Some(ref metrics) = comp.snn_vs_ann {
                writeln!(
                    file,
                    "| {} | {:.2}% | {:.2}x | {:.2}x |",
                    comp.name,
                    metrics.accuracy_delta * 100.0,
                    metrics.speedup,
                    metrics.energy_efficiency
                )?;
            }
        }

        writeln!(file)?;
        Ok(())
    }
}

// Helper module for num_cpus
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_benchmark_report_creation() {
        let report = BenchmarkReport::new("Test Report");
        assert_eq!(report.metadata.title, "Test Report");
        assert_eq!(report.results.len(), 0);
    }

    #[test]
    fn test_add_result() {
        let mut report = BenchmarkReport::new("Test");

        let result = BenchmarkResult {
            name: "test1".to_string(),
            category: "encoding".to_string(),
            profile: ProfileResult {
                name: "test1".to_string(),
                elapsed_ms: 10.0,
                memory_bytes: 1024,
                num_spikes: 100,
                sparsity: 0.05,
                energy_mj: 0.1,
                metadata: HashMap::new(),
            },
            baseline: None,
            metrics: HashMap::new(),
        };

        report.add_result(result);
        assert_eq!(report.summary.total_benchmarks, 1);
        assert_eq!(report.summary.total_time_ms, 10.0);
    }

    #[test]
    fn test_comparison_report() {
        let mut report = ComparisonReport::new("Comparison Test");

        let snn = BaselineResult {
            name: "SNN".to_string(),
            accuracy: 0.95,
            latency_ms: 1.0,
            energy_mj: 0.1,
            memory_bytes: 1000,
            num_operations: 500,
        };

        let ann = BaselineResult {
            name: "ANN".to_string(),
            accuracy: 0.96,
            latency_ms: 5.0,
            energy_mj: 1.0,
            memory_bytes: 10000,
            num_operations: 5000,
        };

        report.add_comparison("test", snn, Some(ann), None);
        assert_eq!(report.comparisons.len(), 1);
    }
}
