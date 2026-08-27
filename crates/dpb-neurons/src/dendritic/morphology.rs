//! # Dendritic Morphology Module
//!
//! Implements dendritic tree structure and morphology loading from SWC files.
//!
//! ## SWC Format
//!
//! Standard format for neuronal morphology:
//! ```text
//! # Sample points: id type x y z radius parent
//! 1 1 0.0 0.0 0.0 10.0 -1  # Soma
//! 2 3 0.0 0.0 20.0 1.0 1   # Dendrite
//! 3 3 0.0 0.0 40.0 0.8 2   # Dendrite
//! ```
//!
//! Types:
//! - 1: soma
//! - 2: axon
//! - 3: basal dendrite
//! - 4: apical dendrite

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// SWC point types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwcType {
    Undefined = 0,
    Soma = 1,
    Axon = 2,
    BasalDendrite = 3,
    ApicalDendrite = 4,
}

impl From<i32> for SwcType {
    fn from(value: i32) -> Self {
        match value {
            1 => SwcType::Soma,
            2 => SwcType::Axon,
            3 => SwcType::BasalDendrite,
            4 => SwcType::ApicalDendrite,
            _ => SwcType::Undefined,
        }
    }
}

/// A point in SWC format
#[derive(Debug, Clone)]
pub struct SwcPoint {
    /// Point ID
    pub id: usize,
    /// Point type
    pub swc_type: SwcType,
    /// X coordinate (μm)
    pub x: f64,
    /// Y coordinate (μm)
    pub y: f64,
    /// Z coordinate (μm)
    pub z: f64,
    /// Radius (μm)
    pub radius: f64,
    /// Parent point ID (-1 for root)
    pub parent: i32,
}

impl SwcPoint {
    /// Calculate distance to another point
    pub fn distance_to(&self, other: &SwcPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculate Euclidean distance from origin
    pub fn distance_from_origin(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// A node in the dendritic tree
#[derive(Debug, Clone)]
pub struct BranchNode {
    /// Compartment index
    pub compartment_idx: usize,
    /// Parent node index (None for root)
    pub parent: Option<usize>,
    /// Child node indices
    pub children: Vec<usize>,
    /// Branch order (0 for soma, increases with branching)
    pub branch_order: usize,
    /// Path distance from soma (μm)
    pub path_distance: f64,
    /// Euclidean distance from soma (μm)
    pub euclidean_distance: f64,
    /// Segment length (μm)
    pub length: f64,
    /// Segment diameter (μm)
    pub diameter: f64,
    /// Surface area (μm²)
    pub surface_area: f64,
    /// Volume (μm³)
    pub volume: f64,
    /// SWC type
    pub swc_type: SwcType,
}

impl BranchNode {
    /// Create a new branch node
    pub fn new(compartment_idx: usize, length: f64, diameter: f64, swc_type: SwcType) -> Self {
        use std::f64::consts::PI;

        let surface_area = PI * diameter * length;
        let radius = diameter / 2.0;
        let volume = PI * radius * radius * length;

        Self {
            compartment_idx,
            parent: None,
            children: Vec::new(),
            branch_order: 0,
            path_distance: 0.0,
            euclidean_distance: 0.0,
            length,
            diameter,
            surface_area,
            volume,
            swc_type,
        }
    }

    /// Check if this is a terminal branch
    pub fn is_terminal(&self) -> bool {
        self.children.is_empty()
    }

    /// Check if this is a branch point
    pub fn is_branch_point(&self) -> bool {
        self.children.len() > 1
    }
}

/// Morphology data for a neuron
#[derive(Debug, Clone)]
pub struct MorphologyData {
    /// All SWC points
    pub points: Vec<SwcPoint>,
    /// Soma center (μm)
    pub soma_center: (f64, f64, f64),
    /// Total dendritic length (μm)
    pub total_length: f64,
    /// Total dendritic surface area (μm²)
    pub total_surface_area: f64,
    /// Number of dendritic terminals
    pub num_terminals: usize,
    /// Number of branch points
    pub num_branch_points: usize,
}

impl MorphologyData {
    /// Create empty morphology
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            soma_center: (0.0, 0.0, 0.0),
            total_length: 0.0,
            total_surface_area: 0.0,
            num_terminals: 0,
            num_branch_points: 0,
        }
    }

    /// Load morphology from SWC file
    pub fn from_swc<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::new(file);

        let mut points = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            let line = line.trim();

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse SWC line: id type x y z radius parent
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 7 {
                return Err(format!("Invalid SWC line: {}", line));
            }

            let id = parts[0]
                .parse::<usize>()
                .map_err(|_| format!("Invalid ID: {}", parts[0]))?;
            let swc_type = parts[1]
                .parse::<i32>()
                .map_err(|_| format!("Invalid type: {}", parts[1]))?
                .into();
            let x = parts[2]
                .parse::<f64>()
                .map_err(|_| format!("Invalid x: {}", parts[2]))?;
            let y = parts[3]
                .parse::<f64>()
                .map_err(|_| format!("Invalid y: {}", parts[3]))?;
            let z = parts[4]
                .parse::<f64>()
                .map_err(|_| format!("Invalid z: {}", parts[4]))?;
            let radius = parts[5]
                .parse::<f64>()
                .map_err(|_| format!("Invalid radius: {}", parts[5]))?;
            let parent = parts[6]
                .parse::<i32>()
                .map_err(|_| format!("Invalid parent: {}", parts[6]))?;

            points.push(SwcPoint {
                id,
                swc_type,
                x,
                y,
                z,
                radius,
                parent,
            });
        }

        if points.is_empty() {
            return Err("No points found in SWC file".to_string());
        }

        // Calculate statistics
        let mut morphology = Self::new();
        morphology.points = points;
        morphology.calculate_statistics();

        Ok(morphology)
    }

    /// Calculate morphology statistics
    fn calculate_statistics(&mut self) {
        use std::f64::consts::PI;

        // Find soma center
        let soma_points: Vec<_> = self
            .points
            .iter()
            .filter(|p| p.swc_type == SwcType::Soma)
            .collect();

        if !soma_points.is_empty() {
            let n = soma_points.len() as f64;
            self.soma_center = (
                soma_points.iter().map(|p| p.x).sum::<f64>() / n,
                soma_points.iter().map(|p| p.y).sum::<f64>() / n,
                soma_points.iter().map(|p| p.z).sum::<f64>() / n,
            );
        }

        // Build parent-child map
        let mut children_map: HashMap<i32, Vec<usize>> = HashMap::new();
        for point in &self.points {
            if point.parent >= 0 {
                children_map.entry(point.parent).or_default().push(point.id);
            }
        }

        // Calculate length and surface area
        let mut total_length = 0.0;
        let mut total_surface_area = 0.0;

        for point in &self.points {
            if point.parent >= 0 {
                // Find parent point
                if let Some(parent) = self.points.iter().find(|p| p.id == point.parent as usize) {
                    let length = point.distance_to(parent);
                    let diameter = point.radius + parent.radius; // Average diameter
                    let surface_area = PI * diameter * length;

                    total_length += length;
                    total_surface_area += surface_area;
                }
            }
        }

        self.total_length = total_length;
        self.total_surface_area = total_surface_area;

        // Count terminals and branch points
        self.num_terminals = self
            .points
            .iter()
            .filter(|p| !children_map.contains_key(&(p.id as i32)))
            .count();

        self.num_branch_points = children_map
            .values()
            .filter(|children| children.len() > 1)
            .count();
    }
}

impl Default for MorphologyData {
    fn default() -> Self {
        Self::new()
    }
}

/// Dendritic tree structure
#[derive(Debug, Clone)]
pub struct DendriticTree {
    /// All branch nodes
    nodes: Vec<BranchNode>,
    /// Root node index (soma)
    root: Option<usize>,
    /// Morphology data
    morphology: MorphologyData,
}

impl DendriticTree {
    /// Create an empty dendritic tree
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
            morphology: MorphologyData::new(),
        }
    }

    /// Create tree from morphology data
    pub fn from_morphology(morphology: MorphologyData) -> Self {
        let mut tree = Self::new();

        // Convert SWC points to branch nodes
        let mut point_to_node: HashMap<usize, usize> = HashMap::new();

        for (idx, point) in morphology.points.iter().enumerate() {
            let length = if point.parent >= 0 {
                if let Some(parent) = morphology
                    .points
                    .iter()
                    .find(|p| p.id == point.parent as usize)
                {
                    point.distance_to(parent)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let diameter = point.radius * 2.0;
            let node = BranchNode::new(idx, length, diameter, point.swc_type);
            tree.nodes.push(node);
            point_to_node.insert(point.id, idx);
        }

        // Build parent-child relationships
        for point in &morphology.points {
            if let Some(&node_idx) = point_to_node.get(&point.id) {
                if point.parent >= 0 {
                    if let Some(&parent_idx) = point_to_node.get(&(point.parent as usize)) {
                        tree.nodes[node_idx].parent = Some(parent_idx);
                        tree.nodes[parent_idx].children.push(node_idx);
                    }
                } else {
                    // This is the root
                    tree.root = Some(node_idx);
                }
            }
        }

        // Move morphology data into tree (needed for calculate_distances)
        tree.morphology = morphology;

        // Calculate distances and branch orders
        if let Some(root) = tree.root {
            tree.calculate_distances(root, 0.0, 0);
        }

        tree
    }

    /// Recursively calculate distances from soma
    fn calculate_distances(&mut self, node_idx: usize, path_dist: f64, order: usize) {
        let node = &mut self.nodes[node_idx];
        node.path_distance = path_dist;
        node.branch_order = order;

        // Calculate Euclidean distance from soma
        let soma = self.morphology.soma_center;
        if let Some(point) = self.morphology.points.get(node.compartment_idx) {
            let dx = point.x - soma.0;
            let dy = point.y - soma.1;
            let dz = point.z - soma.2;
            node.euclidean_distance = (dx * dx + dy * dy + dz * dz).sqrt();
        }

        let children = node.children.clone();
        let new_path_dist = path_dist + node.length;
        let new_order = if children.len() > 1 { order + 1 } else { order };

        for &child_idx in &children {
            self.calculate_distances(child_idx, new_path_dist, new_order);
        }
    }

    /// Get number of branches
    pub fn num_branches(&self) -> usize {
        self.nodes.len()
    }

    /// Get a branch node
    pub fn get_node(&self, idx: usize) -> Option<&BranchNode> {
        self.nodes.get(idx)
    }

    /// Get root node
    pub fn root(&self) -> Option<usize> {
        self.root
    }

    /// Get all terminal branches
    pub fn terminals(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.is_terminal())
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get all branch points
    pub fn branch_points(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.is_branch_point())
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Get path distance from soma to a node
    pub fn path_distance_to_node(&self, idx: usize) -> Option<f64> {
        self.nodes.get(idx).map(|node| node.path_distance)
    }

    /// Get morphology data
    pub fn morphology(&self) -> &MorphologyData {
        &self.morphology
    }

    /// Find closest node to a 3D point
    pub fn find_closest_node(&self, x: f64, y: f64, z: f64) -> Option<usize> {
        let mut closest_idx = None;
        let mut min_dist = f64::INFINITY;

        for (idx, node) in self.nodes.iter().enumerate() {
            if let Some(point) = self.morphology.points.get(node.compartment_idx) {
                let dx = point.x - x;
                let dy = point.y - y;
                let dz = point.z - z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    closest_idx = Some(idx);
                }
            }
        }

        closest_idx
    }
}

impl Default for DendriticTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swc_point() {
        let p1 = SwcPoint {
            id: 1,
            swc_type: SwcType::Soma,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            radius: 10.0,
            parent: -1,
        };

        let p2 = SwcPoint {
            id: 2,
            swc_type: SwcType::BasalDendrite,
            x: 0.0,
            y: 0.0,
            z: 20.0,
            radius: 1.0,
            parent: 1,
        };

        let dist = p1.distance_to(&p2);
        assert!((dist - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_swc_type_conversion() {
        assert_eq!(SwcType::from(1), SwcType::Soma);
        assert_eq!(SwcType::from(2), SwcType::Axon);
        assert_eq!(SwcType::from(3), SwcType::BasalDendrite);
        assert_eq!(SwcType::from(4), SwcType::ApicalDendrite);
        assert_eq!(SwcType::from(99), SwcType::Undefined);
    }

    #[test]
    fn test_branch_node_creation() {
        let node = BranchNode::new(0, 50.0, 2.0, SwcType::BasalDendrite);

        assert_eq!(node.compartment_idx, 0);
        assert!(node.is_terminal());
        assert!(!node.is_branch_point());
        assert!(node.surface_area > 0.0);
        assert!(node.volume > 0.0);
    }

    #[test]
    fn test_morphology_data() {
        let mut morphology = MorphologyData::new();

        // Create simple Y-shaped morphology
        morphology.points = vec![
            SwcPoint {
                id: 1,
                swc_type: SwcType::Soma,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                radius: 10.0,
                parent: -1,
            },
            SwcPoint {
                id: 2,
                swc_type: SwcType::BasalDendrite,
                x: 0.0,
                y: 0.0,
                z: 50.0,
                radius: 1.0,
                parent: 1,
            },
            SwcPoint {
                id: 3,
                swc_type: SwcType::BasalDendrite,
                x: 30.0,
                y: 0.0,
                z: 80.0,
                radius: 0.8,
                parent: 2,
            },
            SwcPoint {
                id: 4,
                swc_type: SwcType::BasalDendrite,
                x: -30.0,
                y: 0.0,
                z: 80.0,
                radius: 0.8,
                parent: 2,
            },
        ];

        morphology.calculate_statistics();

        assert_eq!(morphology.soma_center, (0.0, 0.0, 0.0));
        assert!(morphology.total_length > 0.0);
        assert!(morphology.total_surface_area > 0.0);
        assert_eq!(morphology.num_terminals, 2); // Two terminals
        assert_eq!(morphology.num_branch_points, 1); // One branch point
    }

    #[test]
    fn test_dendritic_tree() {
        let mut morphology = MorphologyData::new();

        // Create simple linear dendrite
        morphology.points = vec![
            SwcPoint {
                id: 1,
                swc_type: SwcType::Soma,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                radius: 10.0,
                parent: -1,
            },
            SwcPoint {
                id: 2,
                swc_type: SwcType::BasalDendrite,
                x: 0.0,
                y: 0.0,
                z: 50.0,
                radius: 1.0,
                parent: 1,
            },
            SwcPoint {
                id: 3,
                swc_type: SwcType::BasalDendrite,
                x: 0.0,
                y: 0.0,
                z: 100.0,
                radius: 0.8,
                parent: 2,
            },
        ];

        morphology.calculate_statistics();

        let tree = DendriticTree::from_morphology(morphology);

        assert_eq!(tree.num_branches(), 3);
        assert!(tree.root().is_some());
        assert_eq!(tree.terminals().len(), 1);
        assert_eq!(tree.branch_points().len(), 0);
    }

    #[test]
    fn test_find_closest_node() {
        let mut morphology = MorphologyData::new();

        morphology.points = vec![
            SwcPoint {
                id: 1,
                swc_type: SwcType::Soma,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                radius: 10.0,
                parent: -1,
            },
            SwcPoint {
                id: 2,
                swc_type: SwcType::BasalDendrite,
                x: 100.0,
                y: 0.0,
                z: 0.0,
                radius: 1.0,
                parent: 1,
            },
        ];

        morphology.calculate_statistics();
        let tree = DendriticTree::from_morphology(morphology);

        // Should find soma as closest to origin
        let closest = tree.find_closest_node(0.0, 0.0, 0.0);
        assert!(closest.is_some());

        // Should find dendrite as closest to (100, 0, 0)
        let closest = tree.find_closest_node(100.0, 0.0, 0.0);
        assert!(closest.is_some());
    }
}
