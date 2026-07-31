# DPB Framework: Code Design Analysis & Development Plan

## Prioritized Implementation Roadmap for Parallel Development

**Version:** 1.0
**Date:** December 2025
**Purpose:** Technical analysis and phased implementation strategy

---

## Executive Summary

> **Note on this figure.** The 547 total below counts algorithms *specified* across the
> framework documents, including third-party front-ends (MediaPipe, OpenPose, ViTPose,
> Praat, CREPE) that DPB integrates with rather than implements. It is a specification
> inventory, not a count of algorithms implemented in this repository, and must not be
> quoted as one.

This document analyzes the **547 specified algorithms** in the DPB Framework documentation, identifies dependencies, common abstractions, and presents a phased development plan enabling **parallel implementation** by multiple developers/teams.

**Key Findings:**
- **19 core abstractions** can serve 80%+ of algorithm implementations
- **5 independent tracks** can proceed in parallel
- **156 synthetic data generators** mirror encoder structure (develop together)
- **Estimated total effort:** 12-18 months with 4-6 developers

---

## 1. Algorithm Inventory Summary

### 1.1 Total Algorithm Count by Category

| Category | Contact | Non-Contact | Shared | Total |
|----------|---------|-------------|--------|-------|
| Signal Acquisition & Preprocessing | 15 | 28 | 0 | **43** |
| Population Templates/Priors | 17 | 44 | 0 | **61** |
| Event-Based Encoders (DPB Core) | 25 | 52 | 0 | **77** |
| Conventional Baselines | 10 | 20 | 0 | **30** |
| Neuron Models | 0 | 0 | 19 | **19** |
| SNN Architectures | 23 | 32 | 0 | **55** |
| ANN Baseline Architectures | 20 | 24 | 0 | **44** |
| Training Algorithms | 0 | 0 | 25 | **25** |
| Output Decoders | 10 | 38 | 0 | **48** |
| Power/Energy Estimation | 0 | 0 | 19 | **19** |
| Convergence Analysis | 10 | 18 | 0 | **28** |
| Evaluation Metrics | 0 | 0 | 40 | **40** |
| Experiment Infrastructure | 0 | 0 | 28 | **28** |
| Visualization | 0 | 0 | 16 | **16** |
| Hardware Deployment | 0 | 0 | 14 | **14** |
| **Subtotal** | 130 | 256 | 161 | **547** |
| Synthetic Data Generators | 42 | 64 | 8 | **156** |
| **GRAND TOTAL** | 172 | 320 | 211 | **703** |

### 1.2 Complexity Distribution

```
Complexity Assessment (703 algorithms):

LOW (1-2 days):      ████████████████████████████░░░░░░░░░░░░  ~45% (316)
  - Simple encoders, metrics, visualizations, data loaders

MEDIUM (3-7 days):   ████████████████████░░░░░░░░░░░░░░░░░░░░  ~35% (246)
  - Neuron models, SNN layers, feature extractors, templates

HIGH (1-3 weeks):    ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~15% (105)
  - Full SNN architectures, fusion systems, hardware export

VERY HIGH (1+ month): ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~5% (36)
  - Multi-modal fusion, Level 3 video synthesis, hardware deployment
```

---

## 2. Dependency Analysis

### 2.1 Layer Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEPENDENCY FLOW (TOP TO BOTTOM)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LAYER 0: CORE INFRASTRUCTURE (NO DEPENDENCIES)                              │
│  ════════════════════════════════════════════                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ GPU Context │  │ Buffer Mgmt │  │ Math Utils  │  │ Type System │        │
│  │   (wgpu)    │  │  (memory)   │  │ (glam/nal)  │  │  (events)   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                         │
│  LAYER 1: SIGNAL PRIMITIVES        ▼                                         │
│  ══════════════════════════════════                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Filters   │  │    FFT      │  │  Resample   │  │  Windowing  │        │
│  │ (IIR/FIR)   │  │ (GPU/CPU)   │  │             │  │             │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                         │
│  LAYER 2: TEMPLATES & ENCODERS     ▼                                         │
│  ══════════════════════════════════                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Population │  │   Level     │  │  Template   │  │ Derivative  │        │
│  │  Templates  │  │  Crossing   │  │  Deviation  │  │  Encoder    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                         │
│  LAYER 3: NEURON MODELS            ▼                                         │
│  ══════════════════════════════════                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │     LIF     │  │    ALIF     │  │  Izhikevich │  │   Custom    │        │
│  │   Neuron    │  │   Neuron    │  │   Neuron    │  │   Neurons   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                         │
│  LAYER 4: SNN ARCHITECTURES        ▼                                         │
│  ══════════════════════════════════                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Feedfwd    │  │  Recurrent  │  │   Graph     │  │ Transformer │        │
│  │    SNN      │  │    SNN      │  │    SNN      │  │    SNN      │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                         │
│  LAYER 5: TRAINING & OUTPUT        ▼                                         │
│  ══════════════════════════════════                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Surrogate  │  │   Output    │  │  Loss Fns   │  │  Metrics    │        │
│  │  Gradients  │  │  Decoders   │  │             │  │             │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                    │                                         │
│  LAYER 6: INTEGRATION              ▼                                         │
│  ══════════════════════════════════                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Multi-Modal │  │    NIR      │  │  Hardware   │  │   Full      │        │
│  │   Fusion    │  │   Export    │  │  Deployment │  │  Pipeline   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Modality Independence Matrix

| Component | Contact | Pose | Hand | Eye | Voice | Fusion |
|-----------|---------|------|------|-----|-------|--------|
| **Contact** | ● | ○ | ○ | ○ | ○ | ◐ |
| **Pose** | ○ | ● | ◐ | ○ | ○ | ◐ |
| **Hand** | ○ | ◐ | ● | ○ | ○ | ◐ |
| **Eye** | ○ | ○ | ○ | ● | ○ | ◐ |
| **Voice** | ○ | ○ | ○ | ○ | ● | ◐ |
| **Fusion** | ◐ | ◐ | ◐ | ◐ | ◐ | ● |

**Legend:** ● = Self, ◐ = Partial dependency, ○ = Independent

**Key Insight:** Each modality can be developed **independently** until fusion layer.

---

## 3. Common Abstractions & Optimizations

### 3.1 Core Traits (Rust) - 19 Shared Abstractions

These abstractions serve **80%+ of algorithms**:

```rust
// ═══════════════════════════════════════════════════════════════════════════
// TIER 1: FUNDAMENTAL TRAITS (Implement First - Everything Depends On These)
// ═══════════════════════════════════════════════════════════════════════════

/// 1. Event representation - Universal across all encoders
pub struct SpikeEvent {
    pub timestamp: f64,
    pub channel: u32,
    pub polarity: i8,      // -1, 0, +1
    pub magnitude: f32,
}

/// 2. Time series signal - Input to all encoders
pub trait Signal {
    fn samples(&self) -> &[f32];
    fn sample_rate(&self) -> f64;
    fn duration(&self) -> f64;
    fn channel_count(&self) -> usize;
}

/// 3. Event encoder - Core DPB abstraction (77 implementations)
pub trait EventEncoder {
    type Config;
    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Vec<SpikeEvent>;
    fn ground_truth(&self) -> Option<&GroundTruth>;
}

/// 4. Population template - Shared by 61 template types
pub trait PopulationTemplate {
    fn expected_value(&self, context: &Context) -> f64;
    fn variance(&self, context: &Context) -> f64;
    fn deviation(&self, observed: f64, context: &Context) -> f64;
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER 2: NEURON ABSTRACTIONS (19 neuron model implementations)
// ═══════════════════════════════════════════════════════════════════════════

/// 5. Membrane dynamics - All neuron models implement this
pub trait MembraneDynamics {
    fn update(&mut self, input_current: f32, dt: f32) -> bool;  // Returns spike
    fn reset(&mut self);
    fn membrane_potential(&self) -> f32;
    fn threshold(&self) -> f32;
}

/// 6. Synaptic model - Weight application
pub trait SynapticModel {
    fn apply_weights(&self, spikes: &[bool], weights: &[f32]) -> Vec<f32>;
}

/// 7. Surrogate gradient - 6 variants
pub trait SurrogateGradient {
    fn forward(&self, x: f32) -> f32;      // Heaviside step
    fn backward(&self, x: f32) -> f32;     // Smooth approximation
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER 3: NETWORK ABSTRACTIONS (55 SNN architectures)
// ═══════════════════════════════════════════════════════════════════════════

/// 8. Spiking layer - Building block for all SNNs
pub trait SpikingLayer {
    fn forward(&mut self, input: &SpikeTensor, timestep: usize) -> SpikeTensor;
    fn reset_state(&mut self);
    fn neuron_count(&self) -> usize;
}

/// 9. SNN network - Composition of layers
pub trait SNNNetwork {
    fn forward(&mut self, input: &SpikeTensor, num_steps: usize) -> SpikeTensor;
    fn layers(&self) -> &[Box<dyn SpikingLayer>];
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER 4: TRAINING & EVALUATION (25 training + 40 metrics)
// ═══════════════════════════════════════════════════════════════════════════

/// 10. Loss function - All training uses this
pub trait LossFunction {
    fn compute(&self, prediction: &Tensor, target: &Tensor) -> f32;
    fn gradient(&self, prediction: &Tensor, target: &Tensor) -> Tensor;
}

/// 11. Optimizer - Weight updates
pub trait Optimizer {
    fn step(&mut self, params: &mut [f32], grads: &[f32]);
}

/// 12. Metric - All 40 evaluation metrics
pub trait Metric {
    fn update(&mut self, prediction: &[f32], target: &[f32]);
    fn compute(&self) -> f64;
    fn reset(&mut self);
    fn name(&self) -> &str;
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER 5: DATA & IO (28 infrastructure + 156 generators)
// ═══════════════════════════════════════════════════════════════════════════

/// 13. Dataset - Data loading abstraction
pub trait Dataset {
    type Item;
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Option<Self::Item>;
}

/// 14. Synthetic generator - All 156 generators
pub trait SyntheticGenerator {
    type Params;
    type Output;
    fn generate(&self, params: &Self::Params, seed: u64) -> Self::Output;
    fn ground_truth(&self, params: &Self::Params, seed: u64) -> GroundTruth;
}

/// 15. Feature extractor - Non-contact modalities (MediaPipe, etc.)
pub trait FeatureExtractor {
    type Input;    // Video frames, audio samples
    type Output;   // Keypoints, F0, etc.
    fn extract(&self, input: &Self::Input) -> Self::Output;
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER 6: GPU & HARDWARE (14 deployment targets)
// ═══════════════════════════════════════════════════════════════════════════

/// 16. GPU kernel - All compute shaders
pub trait GpuKernel {
    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, workgroups: (u32, u32, u32));
    fn bind_group(&self) -> &wgpu::BindGroup;
}

/// 17. Hardware exporter - NIR and device-specific
pub trait HardwareExporter {
    fn export(&self, network: &dyn SNNNetwork) -> ExportResult;
    fn target_platform(&self) -> Platform;
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER 7: CROSS-CUTTING (Power analysis, convergence)
// ═══════════════════════════════════════════════════════════════════════════

/// 18. Power estimator - All 19 power models
pub trait PowerEstimator {
    fn estimate(&self, network: &dyn SNNNetwork, input: &SpikeTensor) -> PowerMetrics;
}

/// 19. Convergence analyzer - All 28 convergence metrics
pub trait ConvergenceAnalyzer {
    fn time_to_accuracy(&self, threshold: f64) -> Option<f64>;
    fn convergence_curve(&self) -> Vec<(f64, f64)>;
}
```

### 3.2 Optimization Opportunities

| Optimization | Algorithms Affected | Effort | Impact |
|--------------|---------------------|--------|--------|
| **GPU batch encoding** | All 77 encoders | Medium | 10-100x speedup |
| **Unified template storage** | 61 templates | Low | Memory reduction |
| **SIMD neuron updates** | 19 neuron models | Medium | 4-8x speedup |
| **Sparse spike representation** | All SNNs (55) | High | Memory/compute |
| **Shared FFT kernels** | 30+ algorithms | Low | Code reuse |
| **Template caching** | 61 templates | Low | Latency reduction |
| **Lazy evaluation** | All pipelines | Medium | Memory efficiency |

### 3.3 Encoder Family Patterns

```
ENCODER INHERITANCE HIERARCHY:

EventEncoder (base trait)
├── ThresholdEncoder (abstract)
│   ├── LevelCrossingEncoder (13 implementations)
│   │   ├── ECGLevelCrossing
│   │   ├── EDALevelCrossing
│   │   ├── TremorLevelCrossing
│   │   └── ... (10 more)
│   │
│   └── RateThresholdEncoder (8 implementations)
│       ├── JointVelocityEncoder
│       ├── PupilRateEncoder
│       └── ... (6 more)
│
├── TemplateDeviationEncoder (abstract)
│   ├── MorphologyDeviationEncoder (12 implementations)
│   │   ├── ECGTemplateEncoder
│   │   ├── PPGTemplateEncoder
│   │   ├── GaitPhaseEncoder
│   │   └── ... (9 more)
│   │
│   └── TrajectoryDeviationEncoder (15 implementations)
│       ├── KeypointDeviationEncoder
│       ├── SaccadeMainSequenceEncoder
│       ├── FormantDeviationEncoder
│       └── ... (12 more)
│
├── DerivativeEncoder (abstract)
│   ├── ZeroCrossingEncoder (6 implementations)
│   ├── ExtremaEncoder (5 implementations)
│   └── InflectionEncoder (4 implementations)
│
└── DiscreteEventEncoder (abstract)
    ├── HeelStrikeEncoder
    ├── TapOnsetEncoder
    ├── SaccadeOnsetEncoder
    ├── PauseOnsetEncoder
    └── ... (10 more)
```

**Optimization:** Implement 4 abstract encoder classes, derive 77 specific encoders.

---

## 4. Development Phases & Parallel Tracks

### 4.1 Phase Overview

```
TIMELINE (Parallel Tracks):

Month:    1    2    3    4    5    6    7    8    9   10   11   12
          │    │    │    │    │    │    │    │    │    │    │    │
TRACK A:  ████████████████████████████████████████████████████████
Core      │ Infrastructure │ Encoders │ SNNs  │ Training │ Deploy│

TRACK B:  ░░░░████████████████████████████████████████████████████
Contact        │ Templates │ Encoders │ Decoders │ Pipeline │

TRACK C:  ░░░░░░░░████████████████████████████████████████████████
NonContact          │ Pose │ Hand │ Eye │ Voice │ Fusion │

TRACK D:  ░░░░░░░░████████████████████████████████████████████████
Synthetic          │ Level1 │ Level2 │ Level3 (partial) │

TRACK E:  ░░░░░░░░░░░░░░░░████████████████████████████████████████
Validation                   │ Metrics │ Benchmarks │ Tests │

          └────────────────┘ └────────────────────┘ └──────────────┘
           Foundation Phase   Development Phase      Integration Phase
```

### 4.2 Detailed Phase Breakdown

---

## PHASE 1: Foundation (Months 1-3)

### Track A: Core Infrastructure

**Priority: CRITICAL - All other tracks depend on this**

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | GPU context setup (wgpu) | 1 | Medium | None |
| 1-2 | Buffer management system | 1 | Medium | None |
| 2-3 | Type system (SpikeEvent, Signal, etc.) | 5 | Low | None |
| 3-4 | Math utilities (glam, nalgebra integration) | 3 | Low | None |
| 4-5 | Signal primitives (filters, FFT, resample) | 8 | Medium | GPU context |
| 5-6 | Base trait definitions (19 traits) | 19 | Medium | Type system |
| 7-8 | WGSL shader compilation pipeline | 2 | High | GPU context |
| 9-10 | Python bindings scaffolding (PyO3) | 1 | Medium | All above |
| 11-12 | Basic test infrastructure | 5 | Low | All above |

**Deliverables:**
- [ ] `dpb-core` crate with GPU context
- [ ] All 19 core traits defined
- [ ] 8 signal processing primitives
- [ ] Python bindings skeleton
- [ ] CI/CD pipeline

**Team:** 2 developers
**Risk:** GPU compatibility across platforms

---

## PHASE 2: Encoder Development (Months 3-6)

### Track A (continued): Encoder Framework

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | ThresholdEncoder base | 1 | Medium | Phase 1 |
| 2-3 | LevelCrossingEncoder base + GPU kernel | 1 + 1 | Medium | Threshold base |
| 3-4 | TemplateDeviationEncoder base | 1 | Medium | Phase 1 |
| 4-5 | DerivativeEncoder base | 1 | Medium | Phase 1 |
| 5-6 | DiscreteEventEncoder base | 1 | Low | Phase 1 |
| 6-8 | GPU batch encoding system | 1 | High | All bases |

### Track B: Contact Modality Encoders

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | ECG templates (5 priors) | 5 | Low | Phase 1 |
| 2-3 | ECG encoders (R-peak, morphology) | 4 | Medium | ECG templates |
| 3-4 | PPG templates + encoders | 4 + 3 | Medium | Phase 1 |
| 4-5 | EDA templates + encoders | 3 + 4 | Medium | Phase 1 |
| 5-6 | Tremor templates + encoders | 4 + 5 | Medium | Phase 1 |
| 6-8 | EMG, respiration encoders | 6 | Medium | Phase 1 |

### Track C: Non-Contact Modality Encoders

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | Gait templates (11 priors) | 11 | Medium | Phase 1 |
| 2-4 | Gait encoders (heel strike, phase, etc.) | 12 | Medium | Gait templates |
| 4-5 | Hand templates (10 priors) | 10 | Low | Phase 1 |
| 5-6 | Hand encoders (tapping, tremor) | 12 | Medium | Hand templates |
| 6-7 | Eye templates (12 priors) | 12 | Medium | Phase 1 |
| 7-8 | Eye encoders (saccade, pupil, fixation) | 15 | Medium | Eye templates |

### Track D: Synthetic Data Generators

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | Generator base traits | 2 | Low | Phase 1 |
| 2-3 | ECG generator (McSharry model) | 1 | High | Generator traits |
| 3-4 | PPG, EDA generators | 2 | Medium | Generator traits |
| 4-5 | Tremor generators (6 types) | 6 | Medium | Generator traits |
| 5-6 | Gait keypoint generator | 1 | High | Generator traits |
| 6-7 | Hand landmark generator | 1 | Medium | Generator traits |
| 7-8 | Voice feature generators | 4 | Medium | Generator traits |

**Phase 2 Deliverables:**
- [ ] 77 event encoders (all modalities)
- [ ] 61 population templates
- [ ] 42 Level 1 synthetic generators
- [ ] 50 Level 2 synthetic generators
- [ ] GPU-accelerated batch encoding

**Team:** 4-5 developers (1 per track + 1 floating)

---

## PHASE 3: Neural Network Development (Months 5-8)

### Track A: Neuron Models & SNN Infrastructure

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | LIF neuron (CPU + GPU) | 2 | Medium | Phase 1 |
| 2-3 | ALIF, ELIF neurons | 2 | Medium | LIF |
| 3-4 | Izhikevich, AdEx neurons | 2 | Medium | LIF |
| 4-5 | Hardware-specific neurons (Xylo, Pulsar) | 2 | Medium | LIF |
| 5-6 | Remaining neuron models | 11 | Medium | LIF base |
| 6-7 | SpikingLayer trait + Linear layer | 2 | Medium | Neurons |
| 7-8 | Conv2d, Pool2d spiking layers | 2 | High | SpikingLayer |

### Track A (continued): SNN Architectures

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 9-10 | Feedforward SNN builder | 3 | Medium | Layers |
| 10-11 | Recurrent SNN (RSNN) | 3 | High | Feedforward |
| 11-12 | Spiking GCN (skeleton graph) | 2 | High | Feedforward |
| 12-14 | Spiking Transformer | 2 | Very High | All layers |

### Track B: Contact-Specific SNNs

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-4 | Cardiac SNN (HR, HRV analysis) | 3 | Medium | Neuron models |
| 4-6 | Tremor classification SNN | 2 | Medium | Neuron models |
| 6-8 | Multi-signal fusion SNN | 2 | High | Individual SNNs |

### Track C: Non-Contact SNNs

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-3 | Gait analysis SNN | 4 | High | Neuron models |
| 3-5 | Hand movement SNN | 3 | Medium | Neuron models |
| 5-7 | Eye tracking SNN | 4 | Medium | Neuron models |
| 7-9 | Voice analysis SNN | 4 | High | Neuron models |

### Track E: Training Infrastructure

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | Surrogate gradient functions (6 types) | 6 | Low | Phase 1 |
| 2-4 | BPTT training loop | 1 | High | Surrogate grad |
| 4-5 | OTTT (online training) | 1 | High | BPTT |
| 5-6 | SLTT (spatial layer-wise) | 1 | Medium | BPTT |
| 6-7 | ANN-to-SNN conversion | 4 | High | All training |
| 7-8 | STDP unsupervised learning | 2 | Medium | Neuron models |

**Phase 3 Deliverables:**
- [ ] 19 neuron models
- [ ] 55 SNN architectures
- [ ] 25 training algorithms
- [ ] ANN-to-SNN conversion pipeline

**Team:** 4-5 developers

---

## PHASE 4: Output & Evaluation (Months 7-10)

### Track A: Output Decoders

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | Spike rate decoder | 2 | Low | Phase 3 |
| 2-3 | Temporal decoder | 2 | Medium | Phase 3 |
| 3-4 | Clinical score decoders (UPDRS) | 8 | Medium | Rate decoder |
| 4-6 | Regression decoders (HR, tremor freq) | 10 | Medium | Rate decoder |
| 6-8 | Classification decoders | 8 | Medium | Rate decoder |
| 8-10 | Multi-task decoders | 4 | High | All decoders |

### Track E: Evaluation & Benchmarks

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | Classification metrics (10) | 10 | Low | Phase 1 |
| 2-3 | Regression metrics (8) | 8 | Low | Phase 1 |
| 3-4 | Signal quality metrics (6) | 6 | Low | Phase 1 |
| 4-5 | Clinical validity metrics (6) | 6 | Medium | Phase 1 |
| 5-6 | Efficiency metrics (10) | 10 | Medium | Phase 3 |
| 6-8 | Convergence analyzers (18) | 18 | Medium | Efficiency |
| 8-10 | Power estimators (19) | 19 | Medium | Phase 3 |
| 10-12 | Benchmark suite | 5 | High | All metrics |

### Track D (continued): Level 3 Synthetic Data

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-4 | Blender gait video generator | 1 | Very High | Level 2 |
| 4-6 | Hand video generator | 1 | Very High | Level 2 |
| 6-8 | WORLD vocoder audio generator | 1 | High | Level 2 |
| 8-10 | Neural TTS + pathology | 1 | High | WORLD |

**Phase 4 Deliverables:**
- [ ] 48 output decoders
- [ ] 40 evaluation metrics
- [ ] 19 power estimators
- [ ] 18 convergence analyzers
- [ ] 9 Level 3 generators (partial)
- [ ] Comprehensive benchmark suite

---

## PHASE 5: Integration & Deployment (Months 9-12)

### Track A: Multi-Modal Fusion

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-3 | Temporal alignment layer | 2 | High | All modality SNNs |
| 3-5 | Cross-modal attention SNN | 2 | Very High | Alignment |
| 5-7 | Hierarchical fusion | 2 | High | Attention |
| 7-9 | Late fusion / voting | 2 | Medium | All SNNs |
| 9-11 | Full NeuroPlay integration | 1 | Very High | All fusion |

### Track A: Hardware Export

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-2 | NIR export | 2 | Medium | Phase 3 |
| 2-4 | Xylo mapper + quantizer | 3 | High | NIR |
| 4-5 | Pulsar mapper | 2 | High | NIR |
| 5-6 | FPGA HLS4ML export | 2 | High | NIR |
| 6-7 | TFLite/ONNX export | 3 | Medium | Phase 3 |
| 7-8 | WebGPU/WASM deployment | 2 | Medium | Phase 1 |

### Track B + C + D: Integration Testing

| Week | Task | Algorithms | Complexity | Dependencies |
|------|------|------------|------------|--------------|
| 1-4 | End-to-end pipeline tests | 10 | High | All phases |
| 4-6 | Cross-platform validation | 5 | Medium | Hardware export |
| 6-8 | Performance optimization | 5 | High | All systems |
| 8-10 | Documentation | - | Medium | All systems |
| 10-12 | Release preparation | - | Medium | All above |

**Phase 5 Deliverables:**
- [ ] Multi-modal fusion system
- [ ] 14 hardware export targets
- [ ] Full pipeline integration
- [ ] Performance benchmarks
- [ ] Documentation and examples

---

## 5. Parallel Implementation Matrix

### 5.1 Developer Assignment Recommendation

```
TEAM STRUCTURE (6 developers):

Developer A (Lead): Core Infrastructure + GPU
├── Phase 1: GPU context, buffers, traits
├── Phase 2: Encoder framework, GPU kernels
├── Phase 3: Spiking layers, architectures
└── Phase 5: Hardware export, NIR

Developer B: Contact Modalities
├── Phase 2: All contact templates + encoders
├── Phase 3: Contact-specific SNNs
└── Phase 4: Contact output decoders

Developer C: Pose + Hand (Non-Contact)
├── Phase 2: Gait + hand templates + encoders
├── Phase 3: Gait + hand SNNs
└── Phase 4: Movement output decoders

Developer D: Eye + Voice (Non-Contact)
├── Phase 2: Eye + voice templates + encoders
├── Phase 3: Eye + voice SNNs
└── Phase 4: Cognitive/speech output decoders

Developer E: Synthetic Data + Validation
├── Phase 2: Level 1 + Level 2 generators
├── Phase 4: Metrics, benchmarks, Level 3
└── Phase 5: Integration testing

Developer F: Training + Python Bindings
├── Phase 3: Training algorithms, surrogate gradients
├── Phase 4: ANN-to-SNN conversion
└── Phase 5: Python API, documentation
```

### 5.2 Weekly Parallelization Capacity

| Phase | Week | Track A | Track B | Track C | Track D | Track E |
|-------|------|---------|---------|---------|---------|---------|
| 1 | 1-4 | ████ | ░░░░ | ░░░░ | ░░░░ | ░░░░ |
| 1 | 5-8 | ████ | ░░░░ | ░░░░ | ░░░░ | ░░░░ |
| 1 | 9-12 | ████ | ░░░░ | ░░░░ | ░░░░ | ░░░░ |
| 2 | 1-4 | ████ | ████ | ████ | ████ | ░░░░ |
| 2 | 5-8 | ████ | ████ | ████ | ████ | ░░░░ |
| 3 | 1-4 | ████ | ████ | ████ | ████ | ████ |
| 3 | 5-8 | ████ | ████ | ████ | ████ | ████ |
| 4 | 1-4 | ████ | ████ | ████ | ████ | ████ |
| 4 | 5-8 | ████ | ░░░░ | ░░░░ | ████ | ████ |
| 5 | 1-4 | ████ | ████ | ████ | ████ | ████ |
| 5 | 5-8 | ████ | ████ | ████ | ████ | ████ |

**Legend:** ████ = Active development, ░░░░ = Blocked/waiting

---

## 6. Risk Analysis & Mitigation

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| wgpu compatibility issues | Medium | High | Test all backends early (Month 1) |
| SNN accuracy below ANN | Medium | Medium | Implement ANN-to-SNN conversion |
| Hardware export failures | Medium | High | Start NIR testing in Phase 3 |
| Level 3 video quality | High | Medium | Prioritize Level 1+2 validation |
| Multi-modal fusion complexity | High | High | Implement late fusion first |

### 6.2 Dependency Risks

| Blocked Component | Blocking Component | Delay Impact | Mitigation |
|-------------------|-------------------|--------------|------------|
| All encoders | GPU context | 2-3 weeks | CPU fallback |
| All SNNs | Neuron models | 2 weeks | Use snnTorch initially |
| Hardware export | Full pipeline | 4 weeks | Export simple models first |
| Level 3 generators | Level 2 generators | 3 weeks | Can skip Level 3 |

---

## 7. Success Metrics by Phase

| Phase | Key Metric | Target | Validation Method |
|-------|------------|--------|-------------------|
| 1 | GPU kernels working | 100% backends | CI tests |
| 2 | Encoder accuracy | >98% vs ground truth | Synthetic data |
| 3 | SNN vs ANN accuracy | Within 2% | Benchmark datasets |
| 4 | Clinical correlation | ICC >0.8 | Expert validation |
| 5 | Hardware latency | <10ms end-to-end | Xylo deployment |
| 5 | Power consumption | <1mW on Xylo | Hardware measurement |

---

## 8. Quick Start Priority List

### Immediate Actions (Week 1)

1. **Set up repository structure:**
   ```
   dpb-framework/
   ├── Cargo.toml (workspace)
   ├── crates/
   │   ├── dpb-core/
   │   ├── dpb-encoders/
   │   ├── dpb-neurons/
   │   ├── dpb-snn/
   │   ├── dpb-synth/
   │   └── dpb-python/
   ├── shaders/
   ├── examples/
   └── tests/
   ```

2. **Implement SpikeEvent and Signal types**
3. **Set up wgpu context with all backends**
4. **Create CI pipeline for multi-platform builds**

### First Milestone (Month 1)

- [ ] GPU context working on Linux, macOS, Windows
- [ ] 5 core traits implemented
- [ ] 1 encoder (LevelCrossing) working end-to-end
- [ ] 1 synthetic generator producing ground truth
- [ ] Basic Python binding returning events

---

## Appendix A: Algorithm Checklist

### Encoders (77 total)

- [ ] Level Crossing (13): ECG, PPG, EDA, EMG, tremor (3-axis), keypoint (X,Y,Z), gaze, pupil, F0, formant, intensity
- [ ] Template Deviation (27): ECG morphology, PPG waveform, gait phase, joint angle (6), saccade main sequence, vowel space, prosody rhythm, ...
- [ ] Derivative (15): Zero-crossing (6), extrema (5), inflection (4)
- [ ] Discrete Event (22): Heel strike, toe-off, tap onset/offset, saccade onset/offset, pause onset, ...

### Neuron Models (19 total)

- [ ] IF, LIF, CLIF, ALIF, ELIF, QLIF, GLIF
- [ ] Izhikevich, AdEx, Hodgkin-Huxley, FitzHugh-Nagumo, Morris-Lecar
- [ ] SRM, Calcium-based, Recurrent, Stochastic
- [ ] Xylo-LIF, Pulsar-LIF, Quantized-LIF

### SNN Architectures (55 total)

- [ ] Feedforward (8), Recurrent (6), Convolutional (8)
- [ ] Graph (7), Transformer (4)
- [ ] Modality-specific (22): Gait (4), Hand (3), Eye (4), Voice (4), Fusion (7)

---

*Document generated: December 2025*
*DPB Framework Development Plan v1.0*
