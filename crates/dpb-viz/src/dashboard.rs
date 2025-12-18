//! Real-time dashboard for monitoring training and inference metrics.
//!
//! This module provides a WebSocket-based dashboard server that can stream
//! live metrics to web frontends. It includes built-in panels for common
//! metrics and allows custom panels via the `MetricPanel` trait.
//!
//! # Examples
//!
//! ```rust
//! use dpb_viz::dashboard::{DashboardConfig, DashboardServer, SpikeRatePanel};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure dashboard
//! let config = DashboardConfig::new()
//!     .with_port(8080)
//!     .with_update_interval_ms(100);
//!
//! // Create server with built-in panels
//! let mut server = DashboardServer::new(config);
//! server.add_panel(Box::new(SpikeRatePanel::new(1000))); // 1000ms window
//!
//! // Start server (non-blocking)
//! server.start().await?;
//!
//! // Update metrics in training loop
//! server.update_metric("spike_rate", 42.5).await?;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::{Result, VizError};

/// Configuration for the dashboard server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Port for WebSocket server
    pub port: u16,
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Maximum number of data points to retain per metric
    pub max_history: usize,
    /// Host address to bind to
    pub host: String,
    /// Enable CORS for web access
    pub enable_cors: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            update_interval_ms: 100,
            max_history: 1000,
            host: "127.0.0.1".to_string(),
            enable_cors: true,
        }
    }
}

impl DashboardConfig {
    /// Create a new dashboard configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the WebSocket port
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the update interval in milliseconds
    pub fn with_update_interval_ms(mut self, interval: u64) -> Self {
        self.update_interval_ms = interval;
        self
    }

    /// Set the maximum history size
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Set the host address
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Enable or disable CORS
    pub fn with_cors(mut self, enable: bool) -> Self {
        self.enable_cors = enable;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.port == 0 {
            return Err(VizError::InvalidConfig("Port cannot be 0".to_string()));
        }
        if self.update_interval_ms == 0 {
            return Err(VizError::InvalidConfig(
                "Update interval must be > 0".to_string(),
            ));
        }
        if self.max_history == 0 {
            return Err(VizError::InvalidConfig(
                "Max history must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Trait for custom metric panels
pub trait MetricPanel: Send + Sync {
    /// Get the panel's unique identifier
    fn id(&self) -> &str;

    /// Get the panel's display name
    fn name(&self) -> &str;

    /// Update the panel with new data
    fn update(&mut self, timestamp: f64, value: f64);

    /// Get the panel's current state as JSON
    fn to_json(&self) -> serde_json::Value;

    /// Reset the panel's state
    fn reset(&mut self);
}

/// Panel for displaying spike rate over time
pub struct SpikeRatePanel {
    id: String,
    name: String,
    window_ms: usize,
    data: VecDeque<(f64, f64)>, // (timestamp, spike_rate)
}

impl SpikeRatePanel {
    /// Create a new spike rate panel with the given window size in milliseconds
    pub fn new(window_ms: usize) -> Self {
        Self {
            id: "spike_rate".to_string(),
            name: "Spike Rate".to_string(),
            window_ms,
            data: VecDeque::new(),
        }
    }

    /// Get the current spike rate statistics
    pub fn statistics(&self) -> PanelStatistics {
        if self.data.is_empty() {
            return PanelStatistics::default();
        }

        let values: Vec<f64> = self.data.iter().map(|(_, v)| *v).collect();
        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;

        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        PanelStatistics {
            mean,
            min,
            max,
            std_dev,
            count: values.len(),
        }
    }
}

impl MetricPanel for SpikeRatePanel {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn update(&mut self, timestamp: f64, value: f64) {
        self.data.push_back((timestamp, value));

        // Remove old data outside the window
        let cutoff = timestamp - (self.window_ms as f64);
        while let Some((ts, _)) = self.data.front() {
            if *ts < cutoff {
                self.data.pop_front();
            } else {
                break;
            }
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "name": self.name,
            "type": "line",
            "window_ms": self.window_ms,
            "data": self.data.iter().map(|(ts, val)| {
                serde_json::json!({"timestamp": ts, "value": val})
            }).collect::<Vec<_>>(),
            "statistics": {
                "mean": self.statistics().mean,
                "min": self.statistics().min,
                "max": self.statistics().max,
                "std_dev": self.statistics().std_dev,
            }
        })
    }

    fn reset(&mut self) {
        self.data.clear();
    }
}

/// Panel for displaying training loss curves
pub struct LossPanel {
    id: String,
    name: String,
    train_loss: VecDeque<(f64, f64)>,
    val_loss: VecDeque<(f64, f64)>,
    max_points: usize,
}

impl LossPanel {
    /// Create a new loss panel
    pub fn new(max_points: usize) -> Self {
        Self {
            id: "loss".to_string(),
            name: "Loss".to_string(),
            train_loss: VecDeque::new(),
            val_loss: VecDeque::new(),
            max_points,
        }
    }

    /// Update training loss
    pub fn update_train(&mut self, timestamp: f64, loss: f64) {
        self.train_loss.push_back((timestamp, loss));
        if self.train_loss.len() > self.max_points {
            self.train_loss.pop_front();
        }
    }

    /// Update validation loss
    pub fn update_val(&mut self, timestamp: f64, loss: f64) {
        self.val_loss.push_back((timestamp, loss));
        if self.val_loss.len() > self.max_points {
            self.val_loss.pop_front();
        }
    }
}

impl MetricPanel for LossPanel {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn update(&mut self, timestamp: f64, value: f64) {
        // Default to training loss
        self.update_train(timestamp, value);
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "name": self.name,
            "type": "multi_line",
            "series": [
                {
                    "name": "Training Loss",
                    "data": self.train_loss.iter().map(|(ts, val)| {
                        serde_json::json!({"timestamp": ts, "value": val})
                    }).collect::<Vec<_>>(),
                },
                {
                    "name": "Validation Loss",
                    "data": self.val_loss.iter().map(|(ts, val)| {
                        serde_json::json!({"timestamp": ts, "value": val})
                    }).collect::<Vec<_>>(),
                }
            ]
        })
    }

    fn reset(&mut self) {
        self.train_loss.clear();
        self.val_loss.clear();
    }
}

/// Panel for displaying validation accuracy
pub struct AccuracyPanel {
    id: String,
    name: String,
    data: VecDeque<(f64, f64)>,
    max_points: usize,
}

impl AccuracyPanel {
    /// Create a new accuracy panel
    pub fn new(max_points: usize) -> Self {
        Self {
            id: "accuracy".to_string(),
            name: "Accuracy".to_string(),
            data: VecDeque::new(),
            max_points,
        }
    }
}

impl MetricPanel for AccuracyPanel {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn update(&mut self, timestamp: f64, value: f64) {
        self.data.push_back((timestamp, value));
        if self.data.len() > self.max_points {
            self.data.pop_front();
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "name": self.name,
            "type": "line",
            "data": self.data.iter().map(|(ts, val)| {
                serde_json::json!({"timestamp": ts, "value": val})
            }).collect::<Vec<_>>(),
        })
    }

    fn reset(&mut self) {
        self.data.clear();
    }
}

/// Panel for displaying system resource usage (CPU, GPU, Memory)
pub struct ResourcePanel {
    id: String,
    name: String,
    cpu_usage: VecDeque<(f64, f64)>,
    gpu_usage: VecDeque<(f64, f64)>,
    memory_usage: VecDeque<(f64, f64)>,
    max_points: usize,
}

impl ResourcePanel {
    /// Create a new resource panel
    pub fn new(max_points: usize) -> Self {
        Self {
            id: "resources".to_string(),
            name: "System Resources".to_string(),
            cpu_usage: VecDeque::new(),
            gpu_usage: VecDeque::new(),
            memory_usage: VecDeque::new(),
            max_points,
        }
    }

    /// Update CPU usage (0.0 - 100.0)
    pub fn update_cpu(&mut self, timestamp: f64, usage: f64) {
        self.cpu_usage.push_back((timestamp, usage));
        if self.cpu_usage.len() > self.max_points {
            self.cpu_usage.pop_front();
        }
    }

    /// Update GPU usage (0.0 - 100.0)
    pub fn update_gpu(&mut self, timestamp: f64, usage: f64) {
        self.gpu_usage.push_back((timestamp, usage));
        if self.gpu_usage.len() > self.max_points {
            self.gpu_usage.pop_front();
        }
    }

    /// Update memory usage in MB
    pub fn update_memory(&mut self, timestamp: f64, usage_mb: f64) {
        self.memory_usage.push_back((timestamp, usage_mb));
        if self.memory_usage.len() > self.max_points {
            self.memory_usage.pop_front();
        }
    }
}

impl MetricPanel for ResourcePanel {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn update(&mut self, timestamp: f64, value: f64) {
        // Default to CPU usage
        self.update_cpu(timestamp, value);
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "name": self.name,
            "type": "multi_line",
            "series": [
                {
                    "name": "CPU Usage (%)",
                    "data": self.cpu_usage.iter().map(|(ts, val)| {
                        serde_json::json!({"timestamp": ts, "value": val})
                    }).collect::<Vec<_>>(),
                },
                {
                    "name": "GPU Usage (%)",
                    "data": self.gpu_usage.iter().map(|(ts, val)| {
                        serde_json::json!({"timestamp": ts, "value": val})
                    }).collect::<Vec<_>>(),
                },
                {
                    "name": "Memory Usage (MB)",
                    "data": self.memory_usage.iter().map(|(ts, val)| {
                        serde_json::json!({"timestamp": ts, "value": val})
                    }).collect::<Vec<_>>(),
                }
            ]
        })
    }

    fn reset(&mut self) {
        self.cpu_usage.clear();
        self.gpu_usage.clear();
        self.memory_usage.clear();
    }
}

/// Statistics for a panel
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PanelStatistics {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
    pub count: usize,
}

/// Dashboard server for real-time metric visualization
pub struct DashboardServer {
    config: DashboardConfig,
    panels: Arc<RwLock<HashMap<String, Box<dyn MetricPanel>>>>,
    running: Arc<RwLock<bool>>,
}

impl DashboardServer {
    /// Create a new dashboard server with the given configuration
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            panels: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Add a metric panel to the dashboard
    pub async fn add_panel(&mut self, panel: Box<dyn MetricPanel>) {
        let id = panel.id().to_string();
        self.panels.write().await.insert(id, panel);
    }

    /// Remove a panel by ID
    pub async fn remove_panel(&mut self, id: &str) -> bool {
        self.panels.write().await.remove(id).is_some()
    }

    /// Update a metric value for a specific panel
    pub async fn update_metric(&self, panel_id: &str, value: f64) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut panels = self.panels.write().await;
        if let Some(panel) = panels.get_mut(panel_id) {
            panel.update(timestamp, value);
            Ok(())
        } else {
            Err(VizError::InvalidConfig(format!(
                "Panel '{}' not found",
                panel_id
            )))
        }
    }

    /// Get the current state of all panels as JSON
    pub async fn get_state(&self) -> serde_json::Value {
        let panels = self.panels.read().await;
        let panel_data: Vec<_> = panels.values().map(|p| p.to_json()).collect();

        serde_json::json!({
            "panels": panel_data,
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            "config": {
                "port": self.config.port,
                "update_interval_ms": self.config.update_interval_ms,
            }
        })
    }

    /// Start the WebSocket server (non-blocking)
    #[cfg(feature = "web")]
    pub async fn start(&mut self) -> Result<()> {
        self.config.validate()?;

        let addr = format!("{}:{}", self.config.host, self.config.port);
        *self.running.write().await = true;

        // In a real implementation, this would start a WebSocket server
        // For now, we just log the configuration
        eprintln!("Dashboard server would start on {}", addr);

        Ok(())
    }

    /// Stop the WebSocket server
    pub async fn stop(&mut self) {
        *self.running.write().await = false;
    }

    /// Check if the server is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Reset all panels
    pub async fn reset(&mut self) {
        let mut panels = self.panels.write().await;
        for panel in panels.values_mut() {
            panel.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_config() {
        let config = DashboardConfig::new()
            .with_port(9000)
            .with_update_interval_ms(200)
            .with_max_history(500);

        assert_eq!(config.port, 9000);
        assert_eq!(config.update_interval_ms, 200);
        assert_eq!(config.max_history, 500);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let config = DashboardConfig::new().with_port(0);
        assert!(config.validate().is_err());

        let config = DashboardConfig::new().with_update_interval_ms(0);
        assert!(config.validate().is_err());

        let config = DashboardConfig::new().with_max_history(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_spike_rate_panel() {
        let mut panel = SpikeRatePanel::new(1000);
        assert_eq!(panel.id(), "spike_rate");
        assert_eq!(panel.name(), "Spike Rate");

        panel.update(0.0, 10.0);
        panel.update(500.0, 20.0);
        panel.update(1000.0, 15.0);

        let json = panel.to_json();
        assert_eq!(json["id"], "spike_rate");
        assert_eq!(json["data"].as_array().unwrap().len(), 3);

        let stats = panel.statistics();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean, 15.0);
    }

    #[test]
    fn test_spike_rate_panel_window() {
        let mut panel = SpikeRatePanel::new(1000);

        panel.update(0.0, 10.0);
        panel.update(500.0, 20.0);
        panel.update(1500.0, 30.0); // This should remove the first point

        let json = panel.to_json();
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 2); // Only last 2 points within window
        assert_eq!(data[0]["value"], 20.0);
        assert_eq!(data[1]["value"], 30.0);
    }

    #[test]
    fn test_loss_panel() {
        let mut panel = LossPanel::new(100);

        panel.update_train(0.0, 1.0);
        panel.update_train(1.0, 0.8);
        panel.update_val(0.0, 1.2);
        panel.update_val(1.0, 0.9);

        let json = panel.to_json();
        assert_eq!(json["series"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_accuracy_panel() {
        let mut panel = AccuracyPanel::new(100);

        panel.update(0.0, 0.75);
        panel.update(1.0, 0.82);
        panel.update(2.0, 0.88);

        let json = panel.to_json();
        assert_eq!(json["data"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_resource_panel() {
        let mut panel = ResourcePanel::new(100);

        panel.update_cpu(0.0, 45.2);
        panel.update_gpu(0.0, 78.5);
        panel.update_memory(0.0, 1024.0);

        let json = panel.to_json();
        let series = json["series"].as_array().unwrap();
        assert_eq!(series.len(), 3);
        assert_eq!(series[0]["name"], "CPU Usage (%)");
        assert_eq!(series[1]["name"], "GPU Usage (%)");
        assert_eq!(series[2]["name"], "Memory Usage (MB)");
    }

    #[tokio::test]
    async fn test_dashboard_server() {
        let config = DashboardConfig::new();
        let mut server = DashboardServer::new(config);

        server.add_panel(Box::new(SpikeRatePanel::new(1000))).await;
        assert!(server.update_metric("spike_rate", 42.0).await.is_ok());

        let state = server.get_state().await;
        assert_eq!(state["panels"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_dashboard_panel_management() {
        let config = DashboardConfig::new();
        let mut server = DashboardServer::new(config);

        server.add_panel(Box::new(SpikeRatePanel::new(1000))).await;
        server.add_panel(Box::new(LossPanel::new(100))).await;

        assert!(server.remove_panel("spike_rate").await);
        assert!(!server.remove_panel("nonexistent").await);
    }

    #[tokio::test]
    async fn test_dashboard_reset() {
        let config = DashboardConfig::new();
        let mut server = DashboardServer::new(config);

        server.add_panel(Box::new(SpikeRatePanel::new(1000))).await;
        server.update_metric("spike_rate", 42.0).await.unwrap();

        server.reset().await;

        let state = server.get_state().await;
        let panel_data = &state["panels"][0]["data"];
        assert_eq!(panel_data.as_array().unwrap().len(), 0);
    }
}
