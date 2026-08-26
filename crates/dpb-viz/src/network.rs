//! Network topology visualization for neural network architecture.
//!
//! This module provides tools for visualizing network structure, including
//! node positioning algorithms, edge rendering, and layer grouping.
//!
//! # Examples
//!
//! ```rust
//! use dpb_viz::network::{NetworkGraph, NodePositioning, EdgeStyle};
//!
//! // Create a network graph
//! let mut graph = NetworkGraph::new()
//!     .with_positioning(NodePositioning::Hierarchical)
//!     .with_edge_style(EdgeStyle::WeightBased);
//!
//! // Add layers
//! let input_layer = graph.add_layer("input", 784);
//! let hidden_layer = graph.add_layer("hidden", 256);
//! let output_layer = graph.add_layer("output", 10);
//!
//! // Add connections
//! graph.connect_layers(input_layer, hidden_layer, 0.8); // 0.8 = average weight
//! graph.connect_layers(hidden_layer, output_layer, 0.6);
//!
//! // Export to SVG
//! let svg = graph.to_svg();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use glam::Vec2;
use crate::{Result, VizError};

/// Node positioning algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodePositioning {
    /// Hierarchical layout (layers stacked)
    Hierarchical,
    /// Force-directed layout
    ForceDirected,
    /// Circular layout
    Circular,
    /// Grid layout
    Grid,
    /// Custom positions provided by user
    Custom,
}

/// Edge rendering style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeStyle {
    /// Uniform thickness and color
    Uniform,
    /// Thickness based on connection weight
    WeightBased,
    /// Color based on connection weight
    ColorCoded,
    /// Both thickness and color based on weight
    Full,
}

/// Configuration for network graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Width of the visualization in pixels
    pub width: u32,
    /// Height of the visualization in pixels
    pub height: u32,
    /// Node radius in pixels
    pub node_radius: f32,
    /// Node positioning algorithm
    pub positioning: NodePositioning,
    /// Edge rendering style
    pub edge_style: EdgeStyle,
    /// Whether to show node labels
    pub show_labels: bool,
    /// Whether to show layer boundaries
    pub show_layer_boundaries: bool,
    /// Background color
    pub background_color: String,
    /// Node color
    pub node_color: String,
    /// Edge color
    pub edge_color: String,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            width: 1200,
            height: 800,
            node_radius: 5.0,
            positioning: NodePositioning::Hierarchical,
            edge_style: EdgeStyle::WeightBased,
            show_labels: true,
            show_layer_boundaries: true,
            background_color: "#FFFFFF".to_string(),
            node_color: "#4A90E2".to_string(),
            edge_color: "#999999".to_string(),
        }
    }
}

impl NetworkConfig {
    /// Create a new network configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the dimensions
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the node radius
    pub fn with_node_radius(mut self, radius: f32) -> Self {
        self.node_radius = radius;
        self
    }

    /// Set the positioning algorithm
    pub fn with_positioning(mut self, positioning: NodePositioning) -> Self {
        self.positioning = positioning;
        self
    }

    /// Set the edge style
    pub fn with_edge_style(mut self, style: EdgeStyle) -> Self {
        self.edge_style = style;
        self
    }
}

/// A node in the network graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique node ID
    pub id: usize,
    /// Layer this node belongs to
    pub layer_id: usize,
    /// Position in 2D space
    pub position: Vec2,
    /// Optional label
    pub label: Option<String>,
    /// Node color (overrides default if set)
    pub color: Option<String>,
}

/// An edge connecting two nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node ID
    pub from: usize,
    /// Target node ID
    pub to: usize,
    /// Connection weight
    pub weight: f32,
    /// Edge color (overrides default if set)
    pub color: Option<String>,
}

/// A layer in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Layer ID
    pub id: usize,
    /// Layer name
    pub name: String,
    /// Number of nodes in this layer
    pub size: usize,
    /// Node IDs in this layer
    pub node_ids: Vec<usize>,
}

/// Network graph for topology visualization
#[derive(Debug, Clone)]
pub struct NetworkGraph {
    config: NetworkConfig,
    nodes: HashMap<usize, Node>,
    edges: Vec<Edge>,
    layers: Vec<Layer>,
    next_node_id: usize,
    next_layer_id: usize,
}

impl NetworkGraph {
    /// Create a new network graph with default configuration
    pub fn new() -> Self {
        Self::with_config(NetworkConfig::default())
    }

    /// Create a new network graph with custom configuration
    pub fn with_config(config: NetworkConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            edges: Vec::new(),
            layers: Vec::new(),
            next_node_id: 0,
            next_layer_id: 0,
        }
    }

    /// Set the positioning algorithm
    pub fn with_positioning(mut self, positioning: NodePositioning) -> Self {
        self.config.positioning = positioning;
        self
    }

    /// Set the edge style
    pub fn with_edge_style(mut self, style: EdgeStyle) -> Self {
        self.config.edge_style = style;
        self
    }

    /// Add a layer to the network
    pub fn add_layer(&mut self, name: impl Into<String>, size: usize) -> usize {
        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        let mut node_ids = Vec::new();

        // Create nodes for this layer
        for _ in 0..size {
            let node_id = self.next_node_id;
            self.next_node_id += 1;

            let node = Node {
                id: node_id,
                layer_id,
                position: Vec2::ZERO, // Will be positioned later
                label: None,
                color: None,
            };

            self.nodes.insert(node_id, node);
            node_ids.push(node_id);
        }

        let layer = Layer {
            id: layer_id,
            name: name.into(),
            size,
            node_ids,
        };

        self.layers.push(layer);

        // Reposition all nodes
        self.position_nodes();

        layer_id
    }

    /// Connect two layers with edges
    pub fn connect_layers(&mut self, from_layer: usize, to_layer: usize, avg_weight: f32) {
        let from_nodes = self.layers[from_layer].node_ids.clone();
        let to_nodes = self.layers[to_layer].node_ids.clone();

        // Create fully connected edges between layers
        for &from_node in &from_nodes {
            for &to_node in &to_nodes {
                let _ = self.add_edge(from_node, to_node, avg_weight);
            }
        }
    }

    /// Add a single edge between two nodes
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f32) -> Result<()> {
        if !self.nodes.contains_key(&from) {
            return Err(VizError::InvalidConfig(format!("Node {} not found", from)));
        }
        if !self.nodes.contains_key(&to) {
            return Err(VizError::InvalidConfig(format!("Node {} not found", to)));
        }

        self.edges.push(Edge {
            from,
            to,
            weight,
            color: None,
        });

        Ok(())
    }

    /// Position nodes using the configured algorithm
    fn position_nodes(&mut self) {
        match self.config.positioning {
            NodePositioning::Hierarchical => self.position_hierarchical(),
            NodePositioning::ForceDirected => self.position_force_directed(),
            NodePositioning::Circular => self.position_circular(),
            NodePositioning::Grid => self.position_grid(),
            NodePositioning::Custom => {} // User provides positions
        }
    }

    /// Hierarchical layout (layers stacked vertically)
    fn position_hierarchical(&mut self) {
        let num_layers = self.layers.len();
        if num_layers == 0 {
            return;
        }

        let layer_spacing = (self.config.height as f32) / (num_layers + 1) as f32;
        let margin = 50.0;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let y = margin + (layer_idx + 1) as f32 * layer_spacing;
            let num_nodes = layer.size;
            let node_spacing = (self.config.width as f32 - 2.0 * margin) / (num_nodes + 1) as f32;

            for (node_idx, &node_id) in layer.node_ids.iter().enumerate() {
                let x = margin + (node_idx + 1) as f32 * node_spacing;
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.position = Vec2::new(x, y);
                }
            }
        }
    }

    /// Force-directed layout (simplified)
    fn position_force_directed(&mut self) {
        // Start with hierarchical layout
        self.position_hierarchical();

        // Apply simple force-directed adjustments
        let iterations = 50;
        let k = 50.0; // Spring constant
        let damping = 0.9;

        for _ in 0..iterations {
            let mut forces: HashMap<usize, Vec2> = HashMap::new();

            // Repulsive forces between all nodes
            let node_ids: Vec<usize> = self.nodes.keys().copied().collect();
            for &id1 in &node_ids {
                for &id2 in &node_ids {
                    if id1 >= id2 {
                        continue;
                    }

                    let pos1 = self.nodes[&id1].position;
                    let pos2 = self.nodes[&id2].position;
                    let delta = pos1 - pos2;
                    let dist = delta.length().max(1.0);

                    let force = delta.normalize() * (k * k / dist);
                    *forces.entry(id1).or_insert(Vec2::ZERO) += force;
                    *forces.entry(id2).or_insert(Vec2::ZERO) -= force;
                }
            }

            // Attractive forces along edges
            for edge in &self.edges {
                let pos1 = self.nodes[&edge.from].position;
                let pos2 = self.nodes[&edge.to].position;
                let delta = pos2 - pos1;
                let dist = delta.length().max(1.0);

                let force = delta.normalize() * (dist / k);
                *forces.entry(edge.from).or_insert(Vec2::ZERO) += force;
                *forces.entry(edge.to).or_insert(Vec2::ZERO) -= force;
            }

            // Apply forces with damping
            for (id, force) in forces {
                if let Some(node) = self.nodes.get_mut(&id) {
                    node.position += force * damping;

                    // Clamp to bounds
                    node.position.x = node.position.x.clamp(10.0, self.config.width as f32 - 10.0);
                    node.position.y = node.position.y.clamp(10.0, self.config.height as f32 - 10.0);
                }
            }
        }
    }

    /// Circular layout
    fn position_circular(&mut self) {
        let center = Vec2::new(self.config.width as f32 / 2.0, self.config.height as f32 / 2.0);
        let radius = (self.config.width.min(self.config.height) as f32 / 2.0) * 0.8;

        let total_nodes = self.nodes.len();
        let mut node_idx = 0;

        for layer in &self.layers {
            for &node_id in &layer.node_ids {
                let angle = 2.0 * std::f32::consts::PI * (node_idx as f32) / (total_nodes as f32);
                let x = center.x + radius * angle.cos();
                let y = center.y + radius * angle.sin();

                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.position = Vec2::new(x, y);
                }

                node_idx += 1;
            }
        }
    }

    /// Grid layout
    fn position_grid(&mut self) {
        let total_nodes = self.nodes.len();
        let cols = (total_nodes as f32).sqrt().ceil() as usize;
        let rows = total_nodes.div_ceil(cols);

        let margin = 50.0;
        let cell_width = (self.config.width as f32 - 2.0 * margin) / cols as f32;
        let cell_height = (self.config.height as f32 - 2.0 * margin) / rows as f32;

        let mut node_idx = 0;
        for layer in &self.layers {
            for &node_id in &layer.node_ids {
                let row = node_idx / cols;
                let col = node_idx % cols;

                let x = margin + (col as f32 + 0.5) * cell_width;
                let y = margin + (row as f32 + 0.5) * cell_height;

                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.position = Vec2::new(x, y);
                }

                node_idx += 1;
            }
        }
    }

    /// Get edge thickness based on weight
    fn edge_thickness(&self, weight: f32) -> f32 {
        match self.config.edge_style {
            EdgeStyle::WeightBased | EdgeStyle::Full => {
                0.5 + weight.abs() * 3.0
            }
            _ => 1.0,
        }
    }

    /// Get edge color based on weight
    fn edge_color(&self, weight: f32) -> String {
        match self.config.edge_style {
            EdgeStyle::ColorCoded | EdgeStyle::Full => {
                // Positive weights: blue, Negative weights: red
                if weight >= 0.0 {
                    let intensity = (weight * 255.0).min(255.0) as u8;
                    format!("#00{:02X}FF", intensity)
                } else {
                    let intensity = ((-weight) * 255.0).min(255.0) as u8;
                    format!("#{:02X}00FF", intensity)
                }
            }
            _ => self.config.edge_color.clone(),
        }
    }

    /// Export to SVG format
    pub fn to_svg(&self) -> String {
        let mut svg = String::new();

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

        // Layer boundaries
        if self.config.show_layer_boundaries && self.config.positioning == NodePositioning::Hierarchical {
            svg.push_str(r##"  <g stroke="#E0E0E0" stroke-width="1" stroke-dasharray="5,5" fill="none">"##);
            svg.push('\n');

            for layer in &self.layers {
                if !layer.node_ids.is_empty()
                    && let Some(first_node) = self.nodes.get(&layer.node_ids[0]) {
                        let y = first_node.position.y;
                        svg.push_str(&format!(
                            r#"    <line x1="0" y1="{}" x2="{}" y2="{}"/>"#,
                            y, self.config.width, y
                        ));
                        svg.push('\n');
                    }
            }

            svg.push_str("  </g>\n");
        }

        // Edges
        svg.push_str("  <g>\n");
        for edge in &self.edges {
            if let (Some(from_node), Some(to_node)) =
                (self.nodes.get(&edge.from), self.nodes.get(&edge.to))
            {
                let thickness = self.edge_thickness(edge.weight);
                let default_color = self.edge_color(edge.weight);
                let color = edge.color.as_ref().unwrap_or(&default_color);

                svg.push_str(&format!(
                    r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" opacity="0.6"/>"#,
                    from_node.position.x,
                    from_node.position.y,
                    to_node.position.x,
                    to_node.position.y,
                    color,
                    thickness
                ));
                svg.push('\n');
            }
        }
        svg.push_str("  </g>\n");

        // Nodes
        svg.push_str("  <g>\n");
        for node in self.nodes.values() {
            let color = node.color.as_ref().unwrap_or(&self.config.node_color);
            svg.push_str(&format!(
                "    <circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\" stroke=\"#333\" stroke-width=\"1\"/>\n",
                node.position.x, node.position.y, self.config.node_radius, color
            ));
            svg.push('\n');

            // Node labels
            if self.config.show_labels
                && let Some(label) = &node.label {
                    svg.push_str(&format!(
                        "    <text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"8\" fill=\"#333\">{}</text>\n",
                        node.position.x,
                        node.position.y + self.config.node_radius + 10.0,
                        label
                    ));
                }
        }
        svg.push_str("  </g>\n");

        // Layer labels
        if self.config.show_labels {
            svg.push_str(r##"  <g font-family="Arial" font-size="14" font-weight="bold" fill="#333">"##);
            svg.push('\n');

            for layer in &self.layers {
                if !layer.node_ids.is_empty()
                    && let Some(first_node) = self.nodes.get(&layer.node_ids[0]) {
                        svg.push_str(&format!(
                            r#"    <text x="10" y="{}">{} ({})</text>"#,
                            first_node.position.y - 15.0,
                            layer.name,
                            layer.size
                        ));
                        svg.push('\n');
                    }
            }

            svg.push_str("  </g>\n");
        }

        svg.push_str("</svg>");
        svg
    }

    /// Export to JSON format
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "positioning": self.config.positioning,
                "edge_style": self.config.edge_style,
            },
            "layers": self.layers.iter().map(|l| {
                serde_json::json!({
                    "id": l.id,
                    "name": l.name,
                    "size": l.size,
                })
            }).collect::<Vec<_>>(),
            "nodes": self.nodes.values().map(|n| {
                serde_json::json!({
                    "id": n.id,
                    "layer_id": n.layer_id,
                    "position": [n.position.x, n.position.y],
                    "label": n.label,
                })
            }).collect::<Vec<_>>(),
            "edges": self.edges.iter().map(|e| {
                serde_json::json!({
                    "from": e.from,
                    "to": e.to,
                    "weight": e.weight,
                })
            }).collect::<Vec<_>>(),
        })
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

impl Default for NetworkGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_config() {
        let config = NetworkConfig::new()
            .with_dimensions(1000, 800)
            .with_node_radius(8.0)
            .with_positioning(NodePositioning::ForceDirected)
            .with_edge_style(EdgeStyle::Full);

        assert_eq!(config.width, 1000);
        assert_eq!(config.height, 800);
        assert_eq!(config.node_radius, 8.0);
        assert_eq!(config.positioning, NodePositioning::ForceDirected);
        assert_eq!(config.edge_style, EdgeStyle::Full);
    }

    #[test]
    fn test_add_layer() {
        let mut graph = NetworkGraph::new();

        let layer1 = graph.add_layer("input", 10);
        let layer2 = graph.add_layer("output", 5);

        assert_eq!(graph.layer_count(), 2);
        assert_eq!(graph.node_count(), 15);
        assert_eq!(layer1, 0);
        assert_eq!(layer2, 1);
    }

    #[test]
    fn test_connect_layers() {
        let mut graph = NetworkGraph::new();

        let layer1 = graph.add_layer("input", 3);
        let layer2 = graph.add_layer("output", 2);

        graph.connect_layers(layer1, layer2, 0.5);

        // 3 input * 2 output = 6 edges
        assert_eq!(graph.edge_count(), 6);
    }

    #[test]
    fn test_add_edge() {
        let mut graph = NetworkGraph::new();
        graph.add_layer("layer", 2);

        // Node IDs are 0 and 1
        assert!(graph.add_edge(0, 1, 0.8).is_ok());
        assert_eq!(graph.edge_count(), 1);

        // Invalid node ID
        assert!(graph.add_edge(0, 99, 0.5).is_err());
    }

    #[test]
    fn test_hierarchical_positioning() {
        let mut graph = NetworkGraph::new()
            .with_positioning(NodePositioning::Hierarchical);

        let layer1 = graph.add_layer("input", 5);
        let layer2 = graph.add_layer("hidden", 3);
        let layer3 = graph.add_layer("output", 2);

        // Check that nodes have been positioned
        for node in graph.nodes.values() {
            assert!(node.position.x > 0.0);
            assert!(node.position.y > 0.0);
        }

        // Check that layers are at different y positions
        let layer1_y = graph.nodes[&graph.layers[layer1].node_ids[0]].position.y;
        let layer2_y = graph.nodes[&graph.layers[layer2].node_ids[0]].position.y;
        let layer3_y = graph.nodes[&graph.layers[layer3].node_ids[0]].position.y;

        assert!(layer1_y < layer2_y);
        assert!(layer2_y < layer3_y);
    }

    #[test]
    fn test_edge_thickness() {
        let graph = NetworkGraph::new()
            .with_edge_style(EdgeStyle::WeightBased);

        let thickness1 = graph.edge_thickness(0.5);
        let thickness2 = graph.edge_thickness(1.0);

        assert!(thickness2 > thickness1);
    }

    #[test]
    fn test_edge_color() {
        let graph = NetworkGraph::new()
            .with_edge_style(EdgeStyle::ColorCoded);

        let color_pos = graph.edge_color(0.5);
        let color_neg = graph.edge_color(-0.5);

        assert!(color_pos.starts_with('#'));
        assert!(color_neg.starts_with('#'));
        assert_ne!(color_pos, color_neg);
    }

    #[test]
    fn test_svg_export() {
        let mut graph = NetworkGraph::new();
        graph.add_layer("input", 3);
        graph.add_layer("output", 2);
        graph.connect_layers(0, 1, 0.5);

        let svg = graph.to_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<line"));
    }

    #[test]
    fn test_json_export() {
        let mut graph = NetworkGraph::new();
        graph.add_layer("input", 3);
        graph.add_layer("output", 2);

        let json = graph.to_json();
        assert_eq!(json["layers"].as_array().unwrap().len(), 2);
        assert_eq!(json["nodes"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn test_circular_positioning() {
        let mut graph = NetworkGraph::new()
            .with_positioning(NodePositioning::Circular);

        graph.add_layer("layer1", 8);

        // All nodes should be positioned on a circle
        let center = Vec2::new(
            graph.config.width as f32 / 2.0,
            graph.config.height as f32 / 2.0,
        );

        let first_node = graph.nodes.values().next().unwrap();
        let radius = (first_node.position - center).length();

        for node in graph.nodes.values() {
            let dist = (node.position - center).length();
            assert!((dist - radius).abs() < 1.0); // Allow small floating point error
        }
    }

    #[test]
    fn test_grid_positioning() {
        let mut graph = NetworkGraph::new()
            .with_positioning(NodePositioning::Grid);

        graph.add_layer("layer1", 9); // 3x3 grid

        // Check that all nodes are positioned
        for node in graph.nodes.values() {
            assert!(node.position.x > 0.0);
            assert!(node.position.y > 0.0);
        }
    }
}
