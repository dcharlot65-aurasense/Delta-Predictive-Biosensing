//! Network partitioning for multi-core neuromorphic hardware
//!
//! Implements graph-based partitioning algorithms to distribute networks
//! across multiple cores while minimizing inter-core communication.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use crate::SNNResult;
use super::constraints::HardwareConstraints;

/// Network partitioner for multi-core systems
#[derive(Debug)]
pub struct NetworkPartitioner {
    /// Hardware constraints
    constraints: HardwareConstraints,
    /// Partitioning strategy
    strategy: PartitionStrategy,
}

impl NetworkPartitioner {
    /// Create a new network partitioner
    pub fn new(constraints: HardwareConstraints, strategy: PartitionStrategy) -> Self {
        Self {
            constraints,
            strategy,
        }
    }

    /// Partition a network across multiple cores
    pub fn partition(&self, graph: &NetworkGraph) -> SNNResult<PartitionResult> {
        match self.strategy {
            PartitionStrategy::Greedy => self.partition_greedy(graph),
            PartitionStrategy::Metis => self.partition_metis(graph),
            PartitionStrategy::Spectral => self.partition_spectral(graph),
            PartitionStrategy::LoadBalanced => self.partition_load_balanced(graph),
        }
    }

    /// Greedy partitioning (simple, fast)
    fn partition_greedy(&self, graph: &NetworkGraph) -> SNNResult<PartitionResult> {
        let max_neurons_per_core = self.constraints.neuron.max_neurons_per_core;
        let mut assignments = HashMap::new();
        let mut core_loads = Vec::new();
        let mut current_core = 0;
        let mut neurons_in_core = 0;

        // Sort neurons by connectivity (high to low)
        let mut neurons: Vec<usize> = (0..graph.num_neurons).collect();
        neurons.sort_by_key(|&n| std::cmp::Reverse(graph.get_degree(n)));

        for neuron in neurons {
            if neurons_in_core >= max_neurons_per_core {
                core_loads.push(neurons_in_core);
                current_core += 1;
                neurons_in_core = 0;
            }

            assignments.insert(neuron, current_core);
            neurons_in_core += 1;
        }

        if neurons_in_core > 0 {
            core_loads.push(neurons_in_core);
        }

        let metrics = self.calculate_metrics(graph, &assignments)?;

        Ok(PartitionResult {
            assignments,
            num_cores: current_core + 1,
            metrics,
        })
    }

    /// METIS-like partitioning (multilevel graph partitioning)
    fn partition_metis(&self, graph: &NetworkGraph) -> SNNResult<PartitionResult> {
        let num_cores = self.calculate_num_cores(graph.num_neurons);

        // Simplified METIS-like algorithm
        // 1. Coarsening phase
        let coarse_graph = self.coarsen_graph(graph);

        // 2. Initial partitioning
        let mut assignments = self.initial_partition(&coarse_graph, num_cores);

        // 3. Uncoarsening and refinement
        assignments = self.refine_partition(graph, assignments);

        let metrics = self.calculate_metrics(graph, &assignments)?;

        Ok(PartitionResult {
            assignments,
            num_cores,
            metrics,
        })
    }

    /// Spectral partitioning (eigenvalue-based)
    fn partition_spectral(&self, graph: &NetworkGraph) -> SNNResult<PartitionResult> {
        // Simplified spectral partitioning
        // In practice, would compute Laplacian eigenvalues
        // For now, use greedy as fallback
        self.partition_greedy(graph)
    }

    /// Load-balanced partitioning
    fn partition_load_balanced(&self, graph: &NetworkGraph) -> SNNResult<PartitionResult> {
        let num_cores = self.calculate_num_cores(graph.num_neurons);
        let target_load = graph.num_neurons / num_cores;

        let mut assignments = HashMap::new();
        let mut core_loads = vec![0; num_cores];

        // Sort neurons by synapse count
        let mut neurons: Vec<usize> = (0..graph.num_neurons).collect();
        neurons.sort_by_key(|&n| std::cmp::Reverse(graph.get_synapse_count(n)));

        // Assign each neuron to least-loaded core
        for neuron in neurons {
            let min_core = core_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, &load)| load)
                .map(|(idx, _)| idx)
                .unwrap();

            assignments.insert(neuron, min_core);
            core_loads[min_core] += 1;
        }

        let metrics = self.calculate_metrics(graph, &assignments)?;

        Ok(PartitionResult {
            assignments,
            num_cores,
            metrics,
        })
    }

    /// Calculate number of cores needed
    fn calculate_num_cores(&self, num_neurons: usize) -> usize {
        let max_per_core = self.constraints.neuron.max_neurons_per_core;
        (num_neurons + max_per_core - 1) / max_per_core
    }

    /// Coarsen graph for multilevel partitioning
    fn coarsen_graph(&self, graph: &NetworkGraph) -> NetworkGraph {
        // Simplified coarsening - merge connected neurons
        // In practice, would use matching algorithms
        graph.clone()
    }

    /// Initial partition of coarse graph
    fn initial_partition(&self, graph: &NetworkGraph, num_parts: usize) -> HashMap<usize, usize> {
        let mut assignments = HashMap::new();
        let neurons_per_part = graph.num_neurons / num_parts;

        for neuron in 0..graph.num_neurons {
            let core = neuron / neurons_per_part.max(1);
            let core = core.min(num_parts - 1);
            assignments.insert(neuron, core);
        }

        assignments
    }

    /// Refine partition using local search
    fn refine_partition(
        &self,
        graph: &NetworkGraph,
        mut assignments: HashMap<usize, usize>,
    ) -> HashMap<usize, usize> {
        // Kernighan-Lin style refinement
        let max_iterations = 10;

        for _ in 0..max_iterations {
            let mut improved = false;

            for neuron in 0..graph.num_neurons {
                let current_core = *assignments.get(&neuron).unwrap();

                // Try moving to neighbor cores
                let neighbor_cores: HashSet<usize> = graph
                    .get_neighbors(neuron)
                    .iter()
                    .filter_map(|&n| assignments.get(&n).copied())
                    .collect();

                for &new_core in &neighbor_cores {
                    if new_core != current_core {
                        // Check if move improves partition
                        let current_cost = self.calculate_neuron_cost(graph, neuron, &assignments);

                        assignments.insert(neuron, new_core);
                        let new_cost = self.calculate_neuron_cost(graph, neuron, &assignments);

                        if new_cost < current_cost {
                            improved = true;
                        } else {
                            // Revert
                            assignments.insert(neuron, current_core);
                        }
                    }
                }
            }

            if !improved {
                break;
            }
        }

        assignments
    }

    /// Calculate cost for a single neuron placement
    fn calculate_neuron_cost(
        &self,
        graph: &NetworkGraph,
        neuron: usize,
        assignments: &HashMap<usize, usize>,
    ) -> usize {
        let neuron_core = *assignments.get(&neuron).unwrap();
        let mut inter_core_edges = 0;

        for &neighbor in &graph.get_neighbors(neuron) {
            if let Some(&neighbor_core) = assignments.get(&neighbor) {
                if neighbor_core != neuron_core {
                    inter_core_edges += 1;
                }
            }
        }

        inter_core_edges
    }

    /// Calculate partition metrics
    fn calculate_metrics(
        &self,
        graph: &NetworkGraph,
        assignments: &HashMap<usize, usize>,
    ) -> SNNResult<PartitionMetrics> {
        let num_cores = assignments.values().max().map(|&c| c + 1).unwrap_or(0);

        // Count neurons per core
        let mut neurons_per_core = vec![0; num_cores];
        for &core in assignments.values() {
            neurons_per_core[core] += 1;
        }

        // Count inter-core communication
        let mut inter_core_edges = 0;
        let mut intra_core_edges = 0;

        for edge in &graph.edges {
            let src_core = assignments.get(&edge.source).unwrap();
            let tgt_core = assignments.get(&edge.target).unwrap();

            if src_core == tgt_core {
                intra_core_edges += 1;
            } else {
                inter_core_edges += 1;
            }
        }

        let total_edges = graph.edges.len();
        let communication_ratio = if total_edges > 0 {
            inter_core_edges as f32 / total_edges as f32
        } else {
            0.0
        };

        // Calculate load balance
        let avg_load = graph.num_neurons as f32 / num_cores as f32;
        let load_imbalance = neurons_per_core
            .iter()
            .map(|&load| ((load as f32 - avg_load).abs() / avg_load).max(0.0))
            .sum::<f32>()
            / num_cores as f32;

        Ok(PartitionMetrics {
            num_cores,
            neurons_per_core,
            inter_core_edges,
            intra_core_edges,
            communication_ratio,
            load_imbalance,
        })
    }
}

/// Network graph representation
#[derive(Debug, Clone)]
pub struct NetworkGraph {
    /// Number of neurons
    pub num_neurons: usize,
    /// Edges (connections)
    pub edges: Vec<Edge>,
    /// Adjacency list
    adjacency: HashMap<usize, Vec<usize>>,
    /// Synapse counts per neuron
    synapse_counts: HashMap<usize, usize>,
}

impl NetworkGraph {
    /// Create a new network graph
    pub fn new(num_neurons: usize, edges: Vec<Edge>) -> Self {
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut synapse_counts: HashMap<usize, usize> = HashMap::new();

        for edge in &edges {
            adjacency
                .entry(edge.source)
                .or_insert_with(Vec::new)
                .push(edge.target);

            adjacency
                .entry(edge.target)
                .or_insert_with(Vec::new)
                .push(edge.source);

            *synapse_counts.entry(edge.source).or_insert(0) += 1;
            *synapse_counts.entry(edge.target).or_insert(0) += 1;
        }

        Self {
            num_neurons,
            edges,
            adjacency,
            synapse_counts,
        }
    }

    /// Get degree (connectivity) of a neuron
    pub fn get_degree(&self, neuron: usize) -> usize {
        self.adjacency.get(&neuron).map(|v| v.len()).unwrap_or(0)
    }

    /// Get synapse count for a neuron
    pub fn get_synapse_count(&self, neuron: usize) -> usize {
        *self.synapse_counts.get(&neuron).unwrap_or(&0)
    }

    /// Get neighbors of a neuron
    pub fn get_neighbors(&self, neuron: usize) -> Vec<usize> {
        self.adjacency
            .get(&neuron)
            .cloned()
            .unwrap_or_else(Vec::new)
    }
}

/// Graph edge
#[derive(Debug, Clone)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
    pub weight: f32,
}

/// Partitioning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Simple greedy partitioning
    Greedy,
    /// METIS-like multilevel partitioning
    Metis,
    /// Spectral partitioning
    Spectral,
    /// Load-balanced partitioning
    LoadBalanced,
}

/// Core assignment for neurons
pub type CoreAssignment = HashMap<usize, usize>;

/// Partition result
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// Neuron to core assignments
    pub assignments: CoreAssignment,
    /// Number of cores used
    pub num_cores: usize,
    /// Partition quality metrics
    pub metrics: PartitionMetrics,
}

impl PartitionResult {
    /// Get neurons assigned to a specific core
    pub fn get_neurons_on_core(&self, core_id: usize) -> Vec<usize> {
        self.assignments
            .iter()
            .filter(|(_, &c)| c == core_id)
            .map(|(&n, _)| n)
            .collect()
    }

    /// Check if partition is valid
    pub fn is_valid(&self, constraints: &HardwareConstraints) -> bool {
        // Check each core doesn't exceed neuron limit
        for core in 0..self.num_cores {
            let neurons_on_core = self.get_neurons_on_core(core).len();
            if neurons_on_core > constraints.neuron.max_neurons_per_core {
                return false;
            }
        }
        true
    }
}

/// Partition quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionMetrics {
    /// Number of cores used
    pub num_cores: usize,
    /// Neurons per core
    pub neurons_per_core: Vec<usize>,
    /// Number of inter-core edges (communication)
    pub inter_core_edges: usize,
    /// Number of intra-core edges
    pub intra_core_edges: usize,
    /// Ratio of inter-core communication
    pub communication_ratio: f32,
    /// Load imbalance (0 = perfect, higher = worse)
    pub load_imbalance: f32,
}

impl PartitionMetrics {
    /// Get quality score (0-1, higher is better)
    pub fn quality_score(&self) -> f32 {
        let comm_score = 1.0 - self.communication_ratio;
        let balance_score = 1.0 - self.load_imbalance.min(1.0);

        // Weighted average
        0.6 * comm_score + 0.4 * balance_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> NetworkGraph {
        let edges = vec![
            Edge {
                source: 0,
                target: 1,
                weight: 1.0,
            },
            Edge {
                source: 1,
                target: 2,
                weight: 1.0,
            },
            Edge {
                source: 2,
                target: 3,
                weight: 1.0,
            },
            Edge {
                source: 0,
                target: 3,
                weight: 1.0,
            },
        ];

        NetworkGraph::new(4, edges)
    }

    #[test]
    fn test_network_graph_creation() {
        let graph = create_test_graph();
        assert_eq!(graph.num_neurons, 4);
        assert_eq!(graph.edges.len(), 4);
    }

    #[test]
    fn test_graph_degree() {
        let graph = create_test_graph();

        assert_eq!(graph.get_degree(0), 2); // Connected to 1 and 3
        assert_eq!(graph.get_degree(1), 2); // Connected to 0 and 2
        assert_eq!(graph.get_degree(2), 2); // Connected to 1 and 3
        assert_eq!(graph.get_degree(3), 2); // Connected to 0 and 2
    }

    #[test]
    fn test_greedy_partitioning() {
        let graph = create_test_graph();
        let constraints = HardwareConstraints::loihi_constraints();
        let partitioner = NetworkPartitioner::new(constraints, PartitionStrategy::Greedy);

        let result = partitioner.partition(&graph).unwrap();

        assert!(result.num_cores > 0);
        assert_eq!(result.assignments.len(), 4);
    }

    #[test]
    fn test_load_balanced_partitioning() {
        let graph = create_test_graph();
        let constraints = HardwareConstraints::loihi_constraints();
        let partitioner = NetworkPartitioner::new(constraints, PartitionStrategy::LoadBalanced);

        let result = partitioner.partition(&graph).unwrap();

        // Should use minimal cores for small network
        assert!(result.num_cores <= 4);

        // Check load balance
        assert!(result.metrics.load_imbalance < 1.0);
    }

    #[test]
    fn test_partition_validity() {
        let graph = create_test_graph();
        let constraints = HardwareConstraints::loihi_constraints();
        let partitioner = NetworkPartitioner::new(constraints, PartitionStrategy::Greedy);

        let result = partitioner.partition(&graph).unwrap();

        assert!(result.is_valid(&constraints));
    }

    #[test]
    fn test_get_neurons_on_core() {
        let graph = create_test_graph();
        let constraints = HardwareConstraints::loihi_constraints();
        let partitioner = NetworkPartitioner::new(constraints, PartitionStrategy::Greedy);

        let result = partitioner.partition(&graph).unwrap();

        // All neurons should be assigned
        let mut all_neurons = HashSet::new();
        for core in 0..result.num_cores {
            let neurons = result.get_neurons_on_core(core);
            for n in neurons {
                all_neurons.insert(n);
            }
        }

        assert_eq!(all_neurons.len(), 4);
    }

    #[test]
    fn test_partition_metrics() {
        let graph = create_test_graph();
        let constraints = HardwareConstraints::loihi_constraints();
        let partitioner = NetworkPartitioner::new(constraints, PartitionStrategy::Greedy);

        let result = partitioner.partition(&graph).unwrap();

        let metrics = &result.metrics;

        assert!(metrics.communication_ratio >= 0.0 && metrics.communication_ratio <= 1.0);
        assert!(metrics.load_imbalance >= 0.0);
        assert_eq!(
            metrics.inter_core_edges + metrics.intra_core_edges,
            graph.edges.len()
        );
    }

    #[test]
    fn test_quality_score() {
        let metrics = PartitionMetrics {
            num_cores: 2,
            neurons_per_core: vec![50, 50],
            inter_core_edges: 10,
            intra_core_edges: 90,
            communication_ratio: 0.1,
            load_imbalance: 0.0,
        };

        let score = metrics.quality_score();
        assert!(score > 0.8); // Good partition
    }

    #[test]
    fn test_large_network_partitioning() {
        // Create a larger network
        let mut edges = Vec::new();
        for i in 0..1000 {
            edges.push(Edge {
                source: i,
                target: (i + 1) % 1000,
                weight: 1.0,
            });
        }

        let graph = NetworkGraph::new(1000, edges);
        let constraints = HardwareConstraints::loihi_constraints();
        let partitioner = NetworkPartitioner::new(constraints, PartitionStrategy::LoadBalanced);

        let result = partitioner.partition(&graph).unwrap();

        // Should use approximately 1000/1024 = 1 core
        assert_eq!(result.num_cores, 1);
        assert!(result.is_valid(&constraints));
    }
}
