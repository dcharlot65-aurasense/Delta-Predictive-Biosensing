//! Temporal event visualization for tracking neural events over time.
//!
//! This module provides tools for visualizing events on a timeline, including
//! spikes, weight updates, predictions, and other temporal events with support
//! for multiple zoom levels and filtering.
//!
//! # Examples
//!
//! ```rust
//! use dpb_viz::timeline::{EventTimeline, TimelineEvent, EventType, ZoomLevel};
//!
//! // Create a timeline
//! let mut timeline = EventTimeline::new(0.0, 1000.0) // 0-1000ms
//!     .with_zoom_level(ZoomLevel::Milliseconds);
//!
//! // Add events
//! timeline.add_event(TimelineEvent::spike(10.5, 0, "Neuron 0 spike"));
//! timeline.add_event(TimelineEvent::prediction(500.0, "class_0", 0.95));
//! timeline.add_event(TimelineEvent::weight_update(750.0, 0, 1, 0.05));
//!
//! // Export to SVG
//! let svg = timeline.to_svg();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::{Result, VizError};

/// Zoom levels for timeline visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoomLevel {
    /// Millisecond precision
    Milliseconds,
    /// Second precision
    Seconds,
    /// Minute precision
    Minutes,
    /// Hour precision
    Hours,
}

impl ZoomLevel {
    /// Get the time unit for this zoom level
    pub fn unit(&self) -> &str {
        match self {
            ZoomLevel::Milliseconds => "ms",
            ZoomLevel::Seconds => "s",
            ZoomLevel::Minutes => "min",
            ZoomLevel::Hours => "h",
        }
    }

    /// Convert time to display value
    pub fn convert_time(&self, time_ms: f32) -> f32 {
        match self {
            ZoomLevel::Milliseconds => time_ms,
            ZoomLevel::Seconds => time_ms / 1000.0,
            ZoomLevel::Minutes => time_ms / 60000.0,
            ZoomLevel::Hours => time_ms / 3600000.0,
        }
    }

    /// Get tick spacing for this zoom level
    pub fn tick_spacing(&self) -> f32 {
        match self {
            ZoomLevel::Milliseconds => 100.0, // 100ms
            ZoomLevel::Seconds => 1000.0,     // 1s
            ZoomLevel::Minutes => 60000.0,    // 1min
            ZoomLevel::Hours => 3600000.0,    // 1h
        }
    }
}

/// Types of events that can be displayed on the timeline
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// Spike event from a neuron
    Spike,
    /// Weight update between neurons
    WeightUpdate,
    /// Prediction made by the network
    Prediction,
    /// Training epoch boundary
    Epoch,
    /// Custom event type
    Custom(String),
}

impl EventType {
    /// Get the color for this event type
    pub fn color(&self) -> &str {
        match self {
            EventType::Spike => "#4A90E2",        // Blue
            EventType::WeightUpdate => "#F5A623", // Orange
            EventType::Prediction => "#7ED321",   // Green
            EventType::Epoch => "#D0021B",        // Red
            EventType::Custom(_) => "#9013FE",    // Purple
        }
    }

    /// Get the symbol for this event type
    pub fn symbol(&self) -> &str {
        match self {
            EventType::Spike => "●",
            EventType::WeightUpdate => "▲",
            EventType::Prediction => "■",
            EventType::Epoch => "◆",
            EventType::Custom(_) => "★",
        }
    }
}

/// An event on the timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Time of the event in milliseconds
    pub time_ms: f32,
    /// Type of event
    pub event_type: EventType,
    /// Description of the event
    pub description: String,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl TimelineEvent {
    /// Create a new timeline event
    pub fn new(time_ms: f32, event_type: EventType, description: impl Into<String>) -> Self {
        Self {
            time_ms,
            event_type,
            description: description.into(),
            metadata: HashMap::new(),
        }
    }

    /// Create a spike event
    pub fn spike(time_ms: f32, neuron_id: usize, description: impl Into<String>) -> Self {
        let mut event = Self::new(time_ms, EventType::Spike, description);
        event
            .metadata
            .insert("neuron_id".to_string(), neuron_id.to_string());
        event
    }

    /// Create a weight update event
    pub fn weight_update(
        time_ms: f32,
        from_neuron: usize,
        to_neuron: usize,
        delta: f32,
    ) -> Self {
        let mut event = Self::new(
            time_ms,
            EventType::WeightUpdate,
            format!("Weight update: {} -> {}", from_neuron, to_neuron),
        );
        event
            .metadata
            .insert("from_neuron".to_string(), from_neuron.to_string());
        event
            .metadata
            .insert("to_neuron".to_string(), to_neuron.to_string());
        event
            .metadata
            .insert("delta".to_string(), format!("{:.4}", delta));
        event
    }

    /// Create a prediction event
    pub fn prediction(time_ms: f32, prediction: impl Into<String>, confidence: f32) -> Self {
        let mut event = Self::new(
            time_ms,
            EventType::Prediction,
            format!("Prediction: {}", prediction.into()),
        );
        event
            .metadata
            .insert("confidence".to_string(), format!("{:.4}", confidence));
        event
    }

    /// Create an epoch boundary event
    pub fn epoch(time_ms: f32, epoch: usize) -> Self {
        let mut event = Self::new(time_ms, EventType::Epoch, format!("Epoch {}", epoch));
        event
            .metadata
            .insert("epoch".to_string(), epoch.to_string());
        event
    }

    /// Create a custom event
    pub fn custom(
        time_ms: f32,
        event_name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let name = event_name.into();
        Self::new(time_ms, EventType::Custom(name), description)
    }

    /// Add metadata to the event
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Configuration for timeline visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineConfig {
    /// Width of the timeline in pixels
    pub width: u32,
    /// Height of the timeline in pixels
    pub height: u32,
    /// Zoom level
    pub zoom_level: ZoomLevel,
    /// Whether to show event descriptions
    pub show_descriptions: bool,
    /// Whether to show grid lines
    pub show_grid: bool,
    /// Event marker size
    pub marker_size: f32,
    /// Font size for labels
    pub font_size: f32,
    /// Background color
    pub background_color: String,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            width: 1200,
            height: 400,
            zoom_level: ZoomLevel::Milliseconds,
            show_descriptions: true,
            show_grid: true,
            marker_size: 8.0,
            font_size: 10.0,
            background_color: "#FFFFFF".to_string(),
        }
    }
}

impl TimelineConfig {
    /// Create a new timeline configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the dimensions
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the zoom level
    pub fn with_zoom_level(mut self, level: ZoomLevel) -> Self {
        self.zoom_level = level;
        self
    }

    /// Set whether to show descriptions
    pub fn with_descriptions(mut self, show: bool) -> Self {
        self.show_descriptions = show;
        self
    }
}

/// Timeline visualization for temporal events
#[derive(Debug, Clone)]
pub struct EventTimeline {
    config: TimelineConfig,
    start_time: f32,
    end_time: f32,
    events: Vec<TimelineEvent>,
    event_lanes: HashMap<EventType, usize>, // Map event types to display lanes
}

impl EventTimeline {
    /// Create a new event timeline
    pub fn new(start_time: f32, end_time: f32) -> Self {
        Self {
            config: TimelineConfig::default(),
            start_time,
            end_time,
            events: Vec::new(),
            event_lanes: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(start_time: f32, end_time: f32, config: TimelineConfig) -> Self {
        Self {
            config,
            start_time,
            end_time,
            events: Vec::new(),
            event_lanes: HashMap::new(),
        }
    }

    /// Set the zoom level
    pub fn with_zoom_level(mut self, level: ZoomLevel) -> Self {
        self.config.zoom_level = level;
        self
    }

    /// Add an event to the timeline
    pub fn add_event(&mut self, event: TimelineEvent) -> Result<()> {
        if event.time_ms < self.start_time || event.time_ms > self.end_time {
            return Err(VizError::InvalidTimeRange {
                start: self.start_time,
                end: self.end_time,
            });
        }

        // Assign lane if not already assigned
        if !self.event_lanes.contains_key(&event.event_type) {
            let lane = self.event_lanes.len();
            self.event_lanes.insert(event.event_type.clone(), lane);
        }

        self.events.push(event);
        Ok(())
    }

    /// Add multiple events
    pub fn add_events(&mut self, events: Vec<TimelineEvent>) -> Result<()> {
        for event in events {
            self.add_event(event)?;
        }
        Ok(())
    }

    /// Filter events by type
    pub fn filter_by_type(&self, event_type: &EventType) -> Vec<&TimelineEvent> {
        self.events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .collect()
    }

    /// Filter events in a time range
    pub fn filter_by_time(&self, start: f32, end: f32) -> Vec<&TimelineEvent> {
        self.events
            .iter()
            .filter(|e| e.time_ms >= start && e.time_ms <= end)
            .collect()
    }

    /// Get the number of events
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get event count by type
    pub fn event_count_by_type(&self, event_type: &EventType) -> usize {
        self.events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .count()
    }

    /// Convert time to x-coordinate
    fn time_to_x(&self, time_ms: f32, margin: f32) -> f32 {
        let range = self.end_time - self.start_time;
        let normalized = (time_ms - self.start_time) / range;
        margin + normalized * (self.config.width as f32 - 2.0 * margin)
    }

    /// Get y-coordinate for an event lane
    fn lane_to_y(&self, lane: usize, margin: f32) -> f32 {
        let num_lanes = self.event_lanes.len().max(1);
        let lane_height = (self.config.height as f32 - 2.0 * margin) / num_lanes as f32;
        margin + (lane as f32 + 0.5) * lane_height
    }

    /// Export to SVG format
    pub fn to_svg(&self) -> String {
        let mut svg = String::new();
        let margin = 50.0;

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));
        svg.push('\n');

        // Background
        svg.push_str(&format!(
            r#"  <rect width="100%" height="100%" fill="{}"/>"#,
            self.config.background_color
        ));
        svg.push('\n');

        // Timeline axis
        svg.push_str(r##"  <g stroke="#333" stroke-width="2">"##);
        svg.push('\n');
        let axis_y = self.config.height as f32 - margin / 2.0;
        svg.push_str(&format!(
            r#"    <line x1="{}" y1="{}" x2="{}" y2="{}"/>"#,
            margin,
            axis_y,
            self.config.width as f32 - margin,
            axis_y
        ));
        svg.push('\n');
        svg.push_str("  </g>\n");

        // Grid lines and time labels
        if self.config.show_grid {
            svg.push_str(r##"  <g stroke="#E0E0E0" stroke-width="0.5" font-size="10" fill="#666">"##);
            svg.push('\n');

            let tick_spacing = self.config.zoom_level.tick_spacing();
            let mut t = self.start_time;
            while t <= self.end_time {
                let x = self.time_to_x(t, margin);
                svg.push_str(&format!(
                    r#"    <line x1="{}" y1="{}" x2="{}" y2="{}"/>"#,
                    x, margin, x, axis_y
                ));
                svg.push('\n');

                // Time label
                let display_time = self.config.zoom_level.convert_time(t);
                svg.push_str(&format!(
                    r#"    <text x="{}" y="{}" text-anchor="middle">{:.1}{}</text>"#,
                    x,
                    self.config.height as f32 - margin / 4.0,
                    display_time,
                    self.config.zoom_level.unit()
                ));
                svg.push('\n');

                t += tick_spacing;
            }

            svg.push_str("  </g>\n");
        }

        // Lane labels
        svg.push_str(r##"  <g font-size="12" fill="#333">"##);
        svg.push('\n');
        for (event_type, &lane) in &self.event_lanes {
            let y = self.lane_to_y(lane, margin);
            let label = match event_type {
                EventType::Spike => "Spikes",
                EventType::WeightUpdate => "Weight Updates",
                EventType::Prediction => "Predictions",
                EventType::Epoch => "Epochs",
                EventType::Custom(name) => name.as_str(),
            };
            svg.push_str(&format!(
                r#"    <text x="5" y="{}" font-weight="bold">{}</text>"#,
                y, label
            ));
            svg.push('\n');
        }
        svg.push_str("  </g>\n");

        // Events
        svg.push_str("  <g>\n");
        for event in &self.events {
            let lane = self.event_lanes[&event.event_type];
            let x = self.time_to_x(event.time_ms, margin);
            let y = self.lane_to_y(lane, margin);
            let color = event.event_type.color();

            // Event marker
            svg.push_str(&format!(
                r##"    <circle cx="{}" cy="{}" r="{}" fill="{}" stroke="#333" stroke-width="1"/>"##,
                x, y, self.config.marker_size, color
            ));
            svg.push('\n');

            // Event description (if enabled)
            if self.config.show_descriptions {
                svg.push_str(&format!(
                    r##"    <text x="{}" y="{}" font-size="{}" fill="#333" text-anchor="middle">{}</text>"##,
                    x,
                    y - self.config.marker_size - 5.0,
                    self.config.font_size,
                    event.description
                ));
                svg.push('\n');
            }
        }
        svg.push_str("  </g>\n");

        // Legend
        self.add_legend(&mut svg, margin);

        svg.push_str("</svg>");
        svg
    }

    fn add_legend(&self, svg: &mut String, margin: f32) {
        svg.push_str(r##"  <g font-size="10" fill="#333">"##);
        svg.push('\n');

        let legend_x = self.config.width as f32 - margin - 150.0;
        let legend_y = margin;

        svg.push_str(&format!(
            r#"    <text x="{}" y="{}" font-weight="bold">Legend</text>"#,
            legend_x, legend_y
        ));
        svg.push('\n');

        let mut offset = 20.0;
        for event_type in self.event_lanes.keys() {
            let color = event_type.color();
            let label = match event_type {
                EventType::Spike => "Spike",
                EventType::WeightUpdate => "Weight Update",
                EventType::Prediction => "Prediction",
                EventType::Epoch => "Epoch",
                EventType::Custom(name) => name.as_str(),
            };

            svg.push_str(&format!(
                r#"    <circle cx="{}" cy="{}" r="5" fill="{}"/>"#,
                legend_x,
                legend_y + offset,
                color
            ));
            svg.push('\n');

            svg.push_str(&format!(
                r#"    <text x="{}" y="{}">{}</text>"#,
                legend_x + 15.0,
                legend_y + offset + 4.0,
                label
            ));
            svg.push('\n');

            offset += 20.0;
        }

        svg.push_str("  </g>\n");
    }

    /// Export to JSON format
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "start_time": self.start_time,
            "end_time": self.end_time,
            "zoom_level": self.config.zoom_level,
            "events": self.events.iter().map(|e| {
                serde_json::json!({
                    "time_ms": e.time_ms,
                    "event_type": e.event_type,
                    "description": e.description,
                    "metadata": e.metadata,
                })
            }).collect::<Vec<_>>(),
            "statistics": {
                "total_events": self.event_count(),
                "spike_events": self.event_count_by_type(&EventType::Spike),
                "weight_updates": self.event_count_by_type(&EventType::WeightUpdate),
                "predictions": self.event_count_by_type(&EventType::Prediction),
            }
        })
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
        self.event_lanes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zoom_level() {
        assert_eq!(ZoomLevel::Milliseconds.unit(), "ms");
        assert_eq!(ZoomLevel::Seconds.unit(), "s");
        assert_eq!(ZoomLevel::Minutes.unit(), "min");
        assert_eq!(ZoomLevel::Hours.unit(), "h");

        assert_eq!(ZoomLevel::Milliseconds.convert_time(1000.0), 1000.0);
        assert_eq!(ZoomLevel::Seconds.convert_time(1000.0), 1.0);
        assert_eq!(ZoomLevel::Minutes.convert_time(60000.0), 1.0);
        assert_eq!(ZoomLevel::Hours.convert_time(3600000.0), 1.0);
    }

    #[test]
    fn test_event_type_properties() {
        assert_eq!(EventType::Spike.color(), "#4A90E2");
        assert_eq!(EventType::WeightUpdate.color(), "#F5A623");
        assert_eq!(EventType::Prediction.color(), "#7ED321");
        assert_eq!(EventType::Epoch.color(), "#D0021B");

        assert_eq!(EventType::Spike.symbol(), "●");
        assert_eq!(EventType::WeightUpdate.symbol(), "▲");
    }

    #[test]
    fn test_timeline_event_creation() {
        let event = TimelineEvent::spike(10.0, 5, "Test spike");
        assert_eq!(event.time_ms, 10.0);
        assert_eq!(event.event_type, EventType::Spike);
        assert_eq!(event.metadata.get("neuron_id").unwrap(), "5");

        let event = TimelineEvent::weight_update(20.0, 1, 2, 0.05);
        assert_eq!(event.event_type, EventType::WeightUpdate);
        assert_eq!(event.metadata.get("from_neuron").unwrap(), "1");
        assert_eq!(event.metadata.get("to_neuron").unwrap(), "2");

        let event = TimelineEvent::prediction(30.0, "class_0", 0.95);
        assert_eq!(event.event_type, EventType::Prediction);
        assert!(event.metadata.get("confidence").unwrap().starts_with("0.95"));

        let event = TimelineEvent::epoch(40.0, 1);
        assert_eq!(event.event_type, EventType::Epoch);
        assert_eq!(event.metadata.get("epoch").unwrap(), "1");

        let event = TimelineEvent::custom(50.0, "test", "Custom event");
        assert!(matches!(event.event_type, EventType::Custom(_)));
    }

    #[test]
    fn test_event_timeline_basic() {
        let mut timeline = EventTimeline::new(0.0, 1000.0);

        assert!(timeline
            .add_event(TimelineEvent::spike(100.0, 0, "Spike 1"))
            .is_ok());
        assert!(timeline
            .add_event(TimelineEvent::spike(200.0, 1, "Spike 2"))
            .is_ok());

        assert_eq!(timeline.event_count(), 2);
        assert_eq!(timeline.event_count_by_type(&EventType::Spike), 2);
    }

    #[test]
    fn test_event_timeline_out_of_range() {
        let mut timeline = EventTimeline::new(0.0, 1000.0);

        // Event before start time
        assert!(timeline
            .add_event(TimelineEvent::spike(-10.0, 0, "Too early"))
            .is_err());

        // Event after end time
        assert!(timeline
            .add_event(TimelineEvent::spike(1100.0, 0, "Too late"))
            .is_err());
    }

    #[test]
    fn test_filter_by_type() {
        let mut timeline = EventTimeline::new(0.0, 1000.0);

        timeline
            .add_event(TimelineEvent::spike(100.0, 0, "Spike 1"))
            .unwrap();
        timeline
            .add_event(TimelineEvent::prediction(200.0, "class_0", 0.9))
            .unwrap();
        timeline
            .add_event(TimelineEvent::spike(300.0, 1, "Spike 2"))
            .unwrap();

        let spikes = timeline.filter_by_type(&EventType::Spike);
        assert_eq!(spikes.len(), 2);

        let predictions = timeline.filter_by_type(&EventType::Prediction);
        assert_eq!(predictions.len(), 1);
    }

    #[test]
    fn test_filter_by_time() {
        let mut timeline = EventTimeline::new(0.0, 1000.0);

        timeline
            .add_event(TimelineEvent::spike(100.0, 0, "Spike 1"))
            .unwrap();
        timeline
            .add_event(TimelineEvent::spike(250.0, 1, "Spike 2"))
            .unwrap();
        timeline
            .add_event(TimelineEvent::spike(500.0, 2, "Spike 3"))
            .unwrap();

        let events = timeline.filter_by_time(200.0, 400.0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].time_ms, 250.0);
    }

    #[test]
    fn test_timeline_svg_export() {
        let mut timeline = EventTimeline::new(0.0, 100.0);
        timeline
            .add_event(TimelineEvent::spike(50.0, 0, "Test"))
            .unwrap();

        let svg = timeline.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn test_timeline_json_export() {
        let mut timeline = EventTimeline::new(0.0, 100.0);
        timeline
            .add_event(TimelineEvent::spike(50.0, 0, "Test"))
            .unwrap();

        let json = timeline.to_json();
        assert_eq!(json["start_time"], 0.0);
        assert_eq!(json["end_time"], 100.0);
        assert_eq!(json["events"].as_array().unwrap().len(), 1);
        assert_eq!(json["statistics"]["total_events"], 1);
    }

    #[test]
    fn test_timeline_clear() {
        let mut timeline = EventTimeline::new(0.0, 1000.0);

        timeline
            .add_event(TimelineEvent::spike(100.0, 0, "Spike"))
            .unwrap();
        assert_eq!(timeline.event_count(), 1);

        timeline.clear();
        assert_eq!(timeline.event_count(), 0);
    }

    #[test]
    fn test_event_with_metadata() {
        let event = TimelineEvent::spike(10.0, 0, "Test")
            .with_metadata("layer", "layer_1")
            .with_metadata("intensity", "0.95");

        assert_eq!(event.metadata.get("layer").unwrap(), "layer_1");
        assert_eq!(event.metadata.get("intensity").unwrap(), "0.95");
    }
}
