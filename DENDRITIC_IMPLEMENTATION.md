# Dendritic Computation and Multi-Compartment Neuron Models

## Implementation Summary

This document describes the comprehensive dendritic computation and multi-compartment neuron models implemented in the `crates/dpb-neurons/src/dendritic/` module.

## Module Structure

### 1. **mod.rs** - Module Exports
Location: `crates/dpb-neurons/src/dendritic/mod.rs`

Main exports:
- `Compartment`, `DendriticTree`, `MultiCompartmentNeuron`
- `IonChannel` trait and implementations
- `DendriticSynapse`, `DendriticIntegration`, `DendriticPlasticity`

### 2. **compartment.rs** - Single Compartment Implementation
Location: `crates/dpb-neurons/src/dendritic/compartment.rs`

Features:
- `Compartment` struct with membrane potential dynamics
- Cable equation implementation: `dV/dt = (1/Cm) * [I_leak + I_channels + I_axial + I_syn]`
- `CompartmentConfig` with biophysical parameters:
  - Membrane capacitance (Cm): 1.0 μF/cm²
  - Leak conductance (g_leak): 0.03 mS/cm²
  - Leak reversal potential (E_leak): -70 mV
  - Compartment geometry (length, diameter)
  - Axial resistivity (R_a): 100 Ω·cm
- `CableParams` derived from geometry:
  - Surface area, cross-sectional area
  - Total capacitance and conductance
  - Axial resistance between compartments
- Numerical integration:
  - Forward Euler (explicit)
  - Crank-Nicolson (implicit, stable)
- Biophysical properties:
  - Input resistance
  - Membrane time constant
  - Space constant (electrotonic length)

Tests: 11 comprehensive tests covering all functionality

### 3. **morphology.rs** - Dendritic Tree Structure
Location: `crates/dpb-neurons/src/dendritic/morphology.rs`

Features:
- **SWC Format Support**: Standard neuronal morphology file format
  - Point types: soma (1), axon (2), basal dendrite (3), apical dendrite (4)
  - Parser for .swc files with error handling
- `SwcPoint` struct:
  - 3D coordinates (x, y, z) in μm
  - Radius and parent relationships
  - Distance calculations
- `BranchNode` representing dendritic segments:
  - Parent-child relationships
  - Branch order calculation
  - Path distance from soma
  - Euclidean distance from soma
  - Surface area and volume
- `DendriticTree` structure:
  - Build from morphology data
  - Recursive distance calculation
  - Terminal branch identification
  - Branch point identification
  - Closest node queries
- `MorphologyData`:
  - Total dendritic length and surface area
  - Number of terminals and branch points
  - Soma center calculation

Tests: 9 tests for morphology loading and tree construction

### 4. **channels.rs** - Ion Channel Models
Location: `crates/dpb-neurons/src/dendritic/channels.rs`

Features:
- **IonChannel Trait**:
  - `current(voltage)`: Calculate channel current
  - `update(voltage, dt)`: Update gating dynamics
  - `conductance()`: Get current conductance
  - `reversal_potential()`: Get reversal potential

- **Hodgkin-Huxley Channels**:
  - Na⁺ channel: g_max·m³·h·(V-E_Na)
  - K⁺ channel: g_max·n⁴·(V-E_K)
  - Alpha-beta formulation for gating variables
  - Realistic voltage-dependent kinetics

- **Calcium Channels**:
  - L-type (high-voltage activated)
  - T-type (low-voltage activated)
  - N-type
  - Inf-tau formulation for smooth dynamics

- **Potassium Channel Variants**:
  - Kv (delayed rectifier)
  - KCa (calcium-activated)
  - Calcium-dependent activation

- **Neurotransmitter Receptors**:
  - **NMDA**: Mg²⁺ voltage-dependent block (Jahr & Stevens 1990)
  - **AMPA**: Fast excitatory (2 ms decay)
  - **GABA_A**: Fast inhibitory (10 ms decay)
  - **GABA_B**: Slow inhibitory (50 ms rise, 200 ms decay)

- **GatingVariable** helper:
  - Exponential dynamics
  - Alpha-beta or inf-tau updates

Tests: 13 tests for all channel types and gating dynamics

### 5. **synapse.rs** - Dendritic Synapses
Location: `crates/dpb-neurons/src/dendritic/synapse.rs`

Features:
- **SynapticConductance** models:
  - Alpha function: g(t) = g_max·(t/τ)·exp(1-t/τ)
  - Bi-exponential: g(t) = g_max·(exp(-t/τ_decay) - exp(-t/τ_rise))
  - Dual receptor (AMPA+NMDA)
  - GABA receptors (A and B)

- **DendriticSynapse**:
  - Location on dendritic tree
  - Synapse type (excitatory/inhibitory/modulatory)
  - Synaptic weight (modifiable for plasticity)
  - Distance from soma
  - Spike counting and ISI tracking
  - Firing rate calculation
  - EPSP amplitude estimation with attenuation

- **SynapseCollection**:
  - Group synapses by compartment
  - Total current calculation
  - Batch updates
  - Statistics (E/I balance, average weight)

- **Temporal and Spatial Summation**:
  - Multiple synaptic events integrate
  - Location-dependent attenuation

Tests: 9 tests for synaptic dynamics and integration

### 6. **integration.rs** - Dendritic Integration
Location: `crates/dpb-neurons/src/dendritic/integration.rs`

Features:
- **DendriticIntegration Trait**:
  - `integrate(inputs, distances)`: Combine inputs from dendrites
  - `has_dendritic_spike()`: Detect local spikes
  - `reset()`: Reset integration state

- **PassiveIntegration** (Cable Theory):
  - Exponential attenuation: exp(-x/λ)
  - Rall's law for infinite cable
  - Space constant (λ) parameter
  - No active amplification

- **ActiveIntegration**:
  - Voltage-gated Na⁺ and K⁺ channels
  - Dendritic spike detection
  - Active amplification
  - Refractory period

- **NonlinearDendrites**:
  - Calcium spikes (T-type Ca channels)
  - NMDA plateau potentials
  - Supralinear summation
  - Branch-specific computation

- **CoincidenceDetection**:
  - Time window for coincidence (ms)
  - Threshold for minimum inputs
  - Nonlinear boost for coincident inputs
  - Temporal precision

Tests: 11 tests for all integration modes

### 7. **plasticity.rs** - Dendritic Plasticity
Location: `crates/dpb-neurons/src/dendritic/plasticity.rs`

Features:
- **DendriticPlasticity Trait**:
  - `update_weight(w, t_pre, t_post, dt)`: STDP-like updates
  - Weight bounds enforcement

- **DendriticStdp**:
  - Classic STDP: Δw = A·exp(-Δt/τ)
  - Requires local dendritic spike for LTP
  - Time windows (τ_+ and τ_-)
  - Weight bounds (w_min, w_max)

- **BranchSpecificPlasticity**:
  - Different learning rates per branch
  - Activity-dependent enhancement
  - Branch activation threshold
  - Exponential activity decay

- **CompartmentPlasticity**:
  - Compartment-specific thresholds
  - Distance-dependent scaling
  - Distal synapses have higher plasticity
  - Location-aware learning

- **Heterosynaptic**:
  - Spatial interaction radius
  - Cooperativity (nearby + coincident)
  - Competition (nearby + sequential)
  - Synapse interaction tracking

- **Metaplasticity** (BCM-like):
  - Sliding threshold (θ)
  - BCM rule: Δw = activity·(activity - θ)
  - Activity-dependent threshold adaptation
  - Homeostatic regulation

Tests: 14 tests for all plasticity mechanisms

### 8. **multi_compartment.rs** - Complete Multi-Compartment Neuron
Location: `crates/dpb-neurons/src/dendritic/multi_compartment.rs`

Features:
- **MultiCompartmentNeuron**:
  - Multiple compartments (soma + dendrites)
  - Cable equation integration
  - Ion channels per compartment
  - Synaptic inputs
  - Action potential generation at soma
  - Backpropagating action potentials (bAPs)

- **NumericalSolver**:
  - Euler (explicit, fast)
  - Crank-Nicolson (implicit, stable)
  - Backward Euler (fully implicit)

- **BackpropagationConfig**:
  - bAP amplitude at soma
  - Attenuation per μm
  - Propagation velocity (μm/ms)
  - Distance-dependent amplitude

- **Neuron Configuration**:
  - AP threshold (-50 mV)
  - Reset potential (-70 mV)
  - Refractory period (2 ms)
  - Compartment parameters

- **Capabilities**:
  - Stimulate specific compartments
  - Get voltage traces
  - Add/remove ion channels
  - Configure synapses
  - Calculate input resistance
  - Calculate time constant

- **Integration with Morphology**:
  - Can build from `DendriticTree`
  - Respects branch structure
  - Distance calculations

Tests: 13 comprehensive integration tests

## Usage Examples

### Basic Multi-Compartment Neuron

```rust
use dpb_neurons::dendritic::{MultiCompartmentNeuron, DendriticSynapse};

// Create neuron with 10 compartments
let mut neuron = MultiCompartmentNeuron::new(10);

// Add excitatory synapse on distal dendrite (compartment 8)
let synapse = DendriticSynapse::excitatory(8, 1.0, 0.5);
neuron.add_synapse(synapse);

// Simulate
for t in 0..1000 {
    let spike = neuron.update(0.1); // 0.1 ms timestep
    if spike {
        println!("Spike at t={} ms", t * 0.1);
    }
}

println!("Soma voltage: {} mV", neuron.soma_voltage());
```

### Load Morphology from SWC

```rust
use dpb_neurons::dendritic::{MorphologyData, DendriticTree, MultiCompartmentNeuron};

// Load morphology from file
let morphology = MorphologyData::from_swc("neuron.swc").unwrap();

// Build dendritic tree
let tree = DendriticTree::from_morphology(morphology);

// Create neuron from tree
let config = NeuronConfig::default();
let mut neuron = MultiCompartmentNeuron::from_tree(tree, config);
```

### Dendritic Integration

```rust
use dpb_neurons::dendritic::{PassiveIntegration, DendriticIntegration};

// Create passive integrator with space constant 200 μm
let integration = PassiveIntegration::new(200.0);

// Inputs at different locations
let inputs = vec![1.0, 1.0, 1.0];
let distances = vec![0.0, 100.0, 300.0]; // μm from soma

// Integrate to get somatic depolarization
let somatic_input = integration.integrate(&inputs, &distances);
```

### Dendritic Plasticity

```rust
use dpb_neurons::dendritic::DendriticStdp;

// Create dendritic STDP rule
let mut stdp = DendriticStdp::new(0.01, 0.01, 20.0, 20.0);

// Register dendritic spike
stdp.register_dendritic_spike(15.0);

// Update weight based on timing
let weight = 1.0;
let new_weight = stdp.update_weight(weight, 10.0, 15.0, 1.0);
// Pre-before-post with dendritic spike → potentiation
```

## Key Features

### Biophysical Realism
- Hodgkin-Huxley ion channels with realistic kinetics
- Cable theory for voltage propagation
- Morphological structure support (SWC files)
- Distance-dependent attenuation
- Backpropagating action potentials

### Dendritic Computation
- Passive linear summation
- Active dendritic spikes
- NMDA plateau potentials
- Coincidence detection
- Supralinear integration

### Synaptic Mechanisms
- AMPA, NMDA, GABA_A, GABA_B receptors
- Mg²⁺ block in NMDA receptors
- Temporal summation
- Spatial summation with attenuation
- Location-dependent EPSPs

### Plasticity
- Dendritic STDP with local spike requirement
- Branch-specific learning rules
- Distance-dependent plasticity
- Heterosynaptic interactions
- Metaplasticity (BCM-like)

### Numerical Stability
- Multiple solver options (Euler, Crank-Nicolson)
- Implicit methods for stiff equations
- Adaptive timesteps possible
- Bounded weights

## Test Coverage

Total tests: **80+ comprehensive tests**

- Compartment: 11 tests
- Morphology: 9 tests
- Channels: 13 tests
- Synapse: 9 tests
- Integration: 11 tests
- Plasticity: 14 tests
- Multi-compartment: 13 tests

All tests verify:
- Correct biophysical behavior
- Numerical stability
- Edge cases and bounds
- Integration between modules

## File Statistics

```
compartment.rs:        13,767 bytes  (397 lines)
morphology.rs:         18,921 bytes  (569 lines)
channels.rs:           18,973 bytes  (652 lines)
synapse.rs:            17,776 bytes  (642 lines)
integration.rs:        16,325 bytes  (535 lines)
plasticity.rs:         19,429 bytes  (667 lines)
multi_compartment.rs:  20,769 bytes  (718 lines)
mod.rs:                 2,888 bytes   (88 lines)

Total:                128,848 bytes (4,268 lines)
```

## Integration with dpb-neurons

The dendritic module is fully integrated with the `dpb-neurons` crate:

1. Added to `lib.rs` module exports
2. Re-exported main types for convenience
3. Documentation added to crate-level docs
4. Example usage in crate documentation
5. Compatible with existing neuron models

## Future Enhancements

Potential additions:
1. Calcium dynamics and buffering
2. More channel types (A-type K, h-current)
3. Spine modeling
4. Gap junctions
5. GPU acceleration for multi-compartment models
6. Optimization for large morphologies
7. Integration with learning algorithms
8. Visualization tools for morphology and voltage

## References

The implementation is based on established neuroscience literature:

1. Hodgkin & Huxley (1952) - Action potential kinetics
2. Rall (1959, 1964) - Cable theory
3. Jahr & Stevens (1990) - NMDA receptor Mg²⁺ block
4. Magee & Johnston (1997) - Dendritic spikes
5. Larkum et al. (1999) - Dendritic integration
6. Sjöström & Häusser (2006) - Dendritic STDP
7. Bienenstock et al. (1982) - BCM theory

## Conclusion

This implementation provides a comprehensive framework for simulating dendritic computation in spiking neural networks. It combines biophysical realism with computational efficiency, making it suitable for both research and applications in neuromorphic computing and brain-inspired AI.
