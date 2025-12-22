# DPB Algorithm Review: Current State vs. Latest Research (December 2025)

This document provides a comprehensive comparison between the algorithms implemented in the Delta-Predictive Biosensing (DPB) framework and the latest state-of-the-art research from 2024-2025, with a focus on GPU parallelization and performance optimization.

## Executive Summary

| Category | DPB Status | Latest Research | Gap Priority | Performance Impact |
|----------|-----------|-----------------|--------------|-------------------|
| Training Algorithms | ✅ BPTT, OTTT, SLTT | EventProp, Smooth Exact GD | **HIGH** | 3-4× faster, 4× less memory |
| Surrogate Gradients | ✅ FastSigmoid, SuperSpike, Box | Parametric SG, Threshold-robust | **MEDIUM** | 10-20% accuracy improvement |
| GPU Backend | ⚠️ WGSL shaders (wgpu) | CUDA optimized (SpikingJelly) | **HIGH** | 11× faster training |
| Transformer Architectures | ❌ Not implemented | Spikformer, STAtten (CVPR 2025) | **HIGH** | 80%+ ImageNet accuracy |
| Sparse Training | ⚠️ Basic pruning | Dynamic spatio-temporal, u-Ticket | **MEDIUM** | 97% sparsity possible |
| Delay Learning | ❌ Not implemented | EventProp delays, DelGrad | **MEDIUM** | Exact temporal gradients |
| Neuron Models | ✅ Comprehensive (19 models) | Current | LOW | Already state-of-the-art |
| Encoders | ✅ Comprehensive (77+ encoders) | Current | LOW | Already comprehensive |

---

## 1. Training Algorithms

### Current DPB Implementation

**Location:** `crates/dpb-snn/src/training/`

| Algorithm | Implemented | Parameters |
|-----------|------------|------------|
| BPTT | ✅ Yes | `num_steps`, surrogate type |
| OTTT | ✅ Yes | `trace_decay` (eligibility traces) |
| SLTT | ✅ Yes | `layer_window` |

### Latest Research (2024-2025)

#### 1.1 EventProp - Exact Gradient Computation

**What it is:** Event-based backpropagation that computes **exact gradients** (not surrogate approximations) by backpropagating errors at spike times.

**Performance gains:**
- **3× faster** than surrogate gradient methods
- **4× less memory** (only stores states at spike times)
- Supports **delay learning** (synaptic delays as trainable parameters)

**Reference:** [Nature Communications (2025)](https://www.nature.com/articles/s41467-025-65394-8)

**Gap:** DPB uses surrogate gradients which are approximations. EventProp provides mathematically exact gradients.

#### 1.2 Smooth Exact Gradient Descent (January 2025)

**What it is:** Published in Physical Review Letters, this method enables gradient-based spike addition and removal without surrogate approximations.

**Key insight:** Handles discontinuities when spikes appear/disappear during training.

**Reference:** [Physical Review Letters (2025)](https://physics.aps.org/featured-article-pdf/10.1103/PhysRevLett.134.027301)

#### 1.3 Temporally-Truncated Local BPTT

**Performance gains:**
- **89.94% reduction in GPU memory**
- **99.64% reduction in MAC operations**
- Trains AlexNet on CIFAR10-DVS efficiently

**Reference:** [Frontiers in Neuroscience (2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1047008/full)

### Recommendation

```
Priority: HIGH
Action: Implement EventProp algorithm for exact gradient computation
Expected Impact: 3-4× training speedup, 4× memory reduction
GPU Benefit: Highly parallelizable event-based computation
```

---

## 2. Surrogate Gradient Functions

### Current DPB Implementation

**Location:** `crates/dpb-snn/src/training/surrogate.rs`

| Surrogate | Implemented | Formula |
|-----------|------------|---------|
| Box | ✅ Yes | Rectangular window |
| FastSigmoid | ✅ Yes | `β / (1 + |β*(v-θ)|)²` |
| SuperSpike | ✅ Yes | `1 / (1 + |β*(v-θ)|)²` |
| Triangle | ⚠️ Falls back to Box | - |
| Exponential | ⚠️ Falls back to SuperSpike | - |

### Latest Research (2024-2025)

#### 2.1 Parametric Surrogate Gradient (PSG) - December 2024

**What it is:** Learnable surrogate gradient parameters that adapt during training.

**Key insight:** "Most SGs are usually chosen intuitively, leaving whether a SG is optimal as an open question."

**Reference:** [ScienceDirect (2024)](https://www.sciencedirect.com/science/article/abs/pii/S092523122401960X)

#### 2.2 Threshold-robust Surrogate Gradient (TrSG) - November 2025

**Components:**
- **MP-Init:** Membrane Potential Initialization to address temporal covariate shift
- **TrSG:** Threshold-robust surrogate that stabilizes training

**Reference:** [arXiv (2025)](https://arxiv.org/html/2511.08708)

#### 2.3 Masked Surrogate Gradients (MSG) - July 2024

**Problem solved:** Standard surrogates cause SNNs to lose their natural sparsity.

**Solution:** Masks gradient flow to preserve sparsity during training.

#### 2.4 Adaptive Surrogate Gradients for RL - October 2025

**Finding:** Shallower slopes increase gradient magnitude in deeper layers but reduce alignment with true gradients. Adaptive scheduling outperforms fixed slopes.

### Recommendation

```
Priority: MEDIUM
Action:
  1. Implement Parametric Surrogate Gradient (learnable parameters)
  2. Add MP-Init for membrane potential initialization
  3. Implement Masked Surrogate Gradients for sparsity preservation
Expected Impact: 10-20% accuracy improvement, better gradient flow
```

---

## 3. GPU Acceleration Architecture

### Current DPB Implementation

**Location:** `crates/dpb-snn/src/gpu/`

| Backend | Status | Implementation |
|---------|--------|----------------|
| WebGPU (wgpu) | ✅ Active | WGSL compute shaders |
| CUDA | ⚠️ Planned | Feature flag `cuda` |
| Metal | ⚠️ Planned | Feature flag `metal` |

**Current kernels:**
- Spike propagation (256 threads/workgroup)
- STDP weight updates
- Sparse matrix-vector multiplication (CSR format)
- Reduction operations

### Latest Research: Framework Benchmarks

**Reference:** [Open Neuromorphic SNN Benchmarks](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)

| Framework | Forward+Backward Time | Notes |
|-----------|----------------------|-------|
| **SpikingJelly (CuPy)** | **0.26s** | Gold standard for CUDA |
| SLAYER/EXODUS | 0.4-0.5s | 1.5-2× slower than SpikingJelly |
| Pure PyTorch | 2-3s | No custom CUDA |
| Spyx (JAX) | ~0.3s | JIT compilation |

**Key insight:** SpikingJelly with CuPy backend achieves **11× speedup** over pure PyTorch through custom CUDA kernels.

### GPU-RANC Framework (April 2024)

**Achievement:** 780× speedup vs. serial simulation for neuromorphic architectures.

**Reference:** [arXiv (2024)](https://arxiv.org/abs/2404.16208)

### Recommendation for Maximum GPU Parallelization

```
Priority: HIGH
Action: Migrate from WGSL to native CUDA/Metal backends

Implementation strategy:
1. Keep WGSL for WebAssembly/browser deployment
2. Add CuPy-style CUDA kernels for training workloads
3. Use vectorization across time dimension (SLAYER/EXODUS approach)

Expected Impact:
- 11× training speedup (matching SpikingJelly)
- 780× potential for full neuromorphic simulation

Key optimizations:
- Fused forward-backward kernels
- Time-dimension parallelization
- Memory-efficient gradient accumulation
- Sparse activation exploitation
```

---

## 4. Spiking Transformer Architectures

### Current DPB Implementation

**Status:** ❌ Not implemented

DPB has feedforward, recurrent, and convolutional SNN architectures, but no transformer-based models.

### Latest Research (2024-2025)

#### 4.1 Spikformer (ICLR 2023, foundational)

**Key innovation:** Spiking Self-Attention (SSA) - attention without softmax

**Performance:**
- 74.81% ImageNet accuracy
- 3.31× lower energy than equivalent Transformer (11.5 mJ vs 38.3 mJ)

#### 4.2 STAtten - Spiking Transformer with Spatial-Temporal Attention (CVPR 2025)

**Key innovation:** Integrates both spatial AND temporal information in self-attention.

**Problem solved:** Existing spike transformers focus on spatial attention, neglecting temporal dependencies.

**Reference:** [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Spiking_Transformer_with_Spatial-Temporal_Attention_CVPR_2025_paper.pdf)

#### 4.3 HAST - Hybrid Attention Spike Transformer (2025)

**Performance:**
- **80.7% accuracy** at T=10 timesteps
- **81.9% accuracy** at T=16 timesteps
- Beats Spikformer by 1.8%

**Reference:** [IET Cyber-Systems and Robotics](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/csy2.70010)

#### 4.4 SGSAFormer

**Innovation:** Spike Gated Self-Attention (SGSA) with Spike Gated Linear Units (SGLU)

**Reference:** [MDPI Electronics](https://www.mdpi.com/2079-9292/14/1/43)

### Recommendation

```
Priority: HIGH (if classification tasks are needed)
Action: Implement Spikformer-family architectures

Components needed:
1. Spiking Patch Splitting (SPS) embedding
2. Spiking Self-Attention (SSA) blocks
3. Spatial-Temporal Attention (STAtten) for temporal modeling
4. Spike Gated Linear Units (SGLU)

Expected Impact:
- 80%+ ImageNet accuracy possible
- 3× energy efficiency over ANNs
- State-of-the-art on event-based datasets (DVS-CIFAR10, etc.)

GPU optimization:
- Attention computation is highly parallelizable
- Batch processing of Q/K/V spike matrices
- Memory-efficient sparse attention patterns
```

---

## 5. Sparse Training & Pruning

### Current DPB Implementation

**Location:** `crates/dpb-snn/src/optimization/pruning.rs`

| Strategy | Implemented |
|----------|------------|
| Magnitude-based | ✅ Yes |
| Gradient-based | ⚠️ Falls back to magnitude |
| Random | ✅ Yes |
| Top-K | ✅ Yes |
| Structured | ✅ Yes |
| Movement | ⚠️ Falls back to magnitude |

### Latest Research (2024-2025)

#### 5.1 u-Ticket: Workload-Balanced Pruning (2024)

**Key innovation:** Optimizes for neuromorphic hardware utilization during LTH-based pruning.

**Performance:**
- **98% filter sparsity** while maintaining accuracy
- **77% latency reduction**
- **64% energy reduction**
- **100% improvement in PE utilization**

**Reference:** [arXiv](https://arxiv.org/html/2302.06746v2)

#### 5.2 Dynamic Spatio-Temporal Pruning (Frontiers, 2025)

**Key insight:** Prune in both weight AND time dimensions.

**Compatibility:** Works well with Speck and Loihi neuromorphic processors.

**Reference:** [Frontiers in Neuroscience (2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1545583/full)

#### 5.3 NDSNN: Neurogenesis-Inspired Training

**Performance:**
- **99% sparsity** on Tiny-ImageNet with ResNet-19
- **40% training cost reduction** vs. standard LTH

**Reference:** [NSF](https://par.nsf.gov/servlets/purl/10462378)

#### 5.4 SpikeFit (EurIPS 2025)

**Innovation:** Hardware-aware training with Clusterization-Aware Training (CAT)

**Features:**
- 2-, 4-, or 8-bit quantization
- Fisher Spike Contribution (FSC) pruning

**Reference:** [arXiv](https://arxiv.org/html/2510.15542)

### Recommendation

```
Priority: MEDIUM
Action:
1. Implement temporal pruning (time-dimension sparsity)
2. Add u-Ticket for workload-balanced hardware deployment
3. Implement Fisher Spike Contribution (FSC) scoring
4. Add Lottery Ticket Hypothesis search with Early-Time tickets

Expected Impact:
- 97-99% sparsity achievable
- 64-77% latency/energy reduction on neuromorphic hardware
- Enables edge deployment

GPU optimization:
- Sparse tensor operations
- Dynamic mask updates during training
- Hardware utilization monitoring
```

---

## 6. Delay Learning

### Current DPB Implementation

**Status:** ❌ Synaptic delays not trainable

Current implementation uses fixed delays in recurrent connections.

### Latest Research (2024-2025)

#### 6.1 DelGrad: Delay Gradient Learning

**What it is:** Analytically computes exact gradients for synaptic delays.

**Key insight:** Delays encode temporal structure; learning them improves temporal tasks.

**Reference:** [arXiv](https://arxiv.org/html/2507.10568)

#### 6.2 EventProp with Delay Learning

**Features:**
- Exact gradients for both weights AND delays
- Supports recurrent SNNs
- Multiple spikes per neuron

**Reference:** [Nature Communications (2025)](https://www.nature.com/articles/s41467-025-65394-8)

### Recommendation

```
Priority: MEDIUM
Action: Add trainable synaptic delays

Implementation:
1. Delay buffer in spike propagation
2. Delay gradients via EventProp formalism
3. Delay bounds (hardware constraints)

Expected Impact:
- Better temporal pattern recognition
- Essential for biosignal timing tasks
- Enables spike-timing-based learning

GPU optimization:
- Delay indexing with circular buffers
- Batched delay updates
- Sparse delay connectivity
```

---

## 7. Neuron Models - Already Comprehensive

### Current DPB Implementation

**Location:** `crates/dpb-neurons/src/`

| Category | Models Implemented | Status |
|----------|-------------------|--------|
| LIF Variants | IF, LIF, CLIF, ALIF, ELIF, QLIF, GLIF | ✅ Complete |
| Phenomenological | Izhikevich (7 presets) | ✅ Complete |
| Adaptive | AdEx, Calcium | ✅ Complete |
| Biophysical | Hodgkin-Huxley, FitzHugh-Nagumo, Morris-Lecar | ✅ Complete |
| Specialized | SRM, Stochastic, Recurrent | ✅ Complete |
| Hardware | XyloLIF, PulsarLIF, QuantizedLIF | ✅ Complete |
| Dendritic | Multi-compartment, Ion channels, STDP | ✅ Complete |

**Assessment:** DPB's neuron model library is **state-of-the-art** and comprehensive. No significant gaps identified.

---

## 8. Encoders - Already Comprehensive

### Current DPB Implementation

**Location:** `crates/dpb-encoders/src/`

| Category | Encoders | Status |
|----------|----------|--------|
| Base | Level-crossing, Template deviation, Derivative | ✅ Complete |
| ECG | R-peak, Morphology, ST-deviation, HRV | ✅ Complete |
| EEG | Alpha/Beta/Theta/Gamma/Delta bands, ERP, Spindles | ✅ Complete |
| EMG | Burst, Amplitude, Fatigue | ✅ Complete |
| Voice | F0, Jitter, Shimmer, HNR, Formants | ✅ Complete |
| Gait | Heel strike, Cadence, Stride | ✅ Complete |
| Eye | Saccade, Fixation, Pupil | ✅ Complete |
| Balance | CoP sway, Velocity, Stability limits | ✅ Complete |

**Assessment:** 77+ encoders with 61+ population templates. **Comprehensive coverage** of biosignal modalities.

---

## 9. Implementation Roadmap for GPU Optimization

### Phase 1: CUDA Backend (Highest Priority)

```rust
// Target: Match SpikingJelly performance (11× speedup)

// 1. CUDA kernel for fused forward-backward pass
// 2. Time-dimension vectorization (SLAYER approach)
// 3. CuPy integration for dynamic kernel generation
// 4. Memory pooling for gradient storage

// Expected: 0.26s forward+backward for 16k neuron network
```

### Phase 2: EventProp Integration

```rust
// Target: Exact gradients with 3× speedup, 4× memory reduction

// 1. Event-based backward pass
// 2. Spike-time state storage only
// 3. Delay gradient computation
// 4. Recurrent network support
```

### Phase 3: Spiking Transformers

```rust
// Target: 80%+ ImageNet accuracy with spike efficiency

// 1. Spiking Self-Attention (SSA) module
// 2. Spiking Patch Splitting embedding
// 3. Spatial-Temporal Attention (STAtten)
// 4. Spike Gated Linear Units
```

### Phase 4: Advanced Sparse Training

```rust
// Target: 97%+ sparsity with minimal accuracy loss

// 1. Temporal pruning (time-dimension sparsity)
// 2. u-Ticket workload balancing
// 3. Fisher Spike Contribution scoring
// 4. Neurogenesis-inspired dynamic connectivity
```

---

## 10. Summary: Priority Actions

| Priority | Action | Expected Speedup | Memory Impact |
|----------|--------|-----------------|---------------|
| 🔴 HIGH | CUDA backend (SpikingJelly-style) | 11× | Same |
| 🔴 HIGH | EventProp exact gradients | 3× | 4× reduction |
| 🔴 HIGH | Spiking Transformers | N/A (new capability) | Higher |
| 🟡 MEDIUM | Parametric surrogate gradients | 1.2× | Same |
| 🟡 MEDIUM | Temporal pruning | 2× (inference) | 97% reduction |
| 🟡 MEDIUM | Delay learning | N/A (new capability) | Slight increase |
| 🟢 LOW | Neuron models | Already optimal | - |
| 🟢 LOW | Encoders | Already optimal | - |

---

## Sources

### Training Algorithms
- [Direct Training High-Performance Deep SNNs Review](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full)
- [EXODUS: Stable and Efficient SNN Training](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9945199/)
- [Event-Based Delay Learning (Nature 2025)](https://www.nature.com/articles/s41467-025-65394-8)

### Surrogate Gradients
- [Threshold-robust Surrogate Gradient](https://arxiv.org/html/2511.08708)
- [Parametric Surrogate Gradient](https://www.sciencedirect.com/science/article/abs/pii/S092523122401960X)
- [Smooth Exact Gradient Descent (PRL 2025)](https://physics.aps.org/featured-article-pdf/10.1103/PhysRevLett.134.027301)

### GPU Optimization
- [GPU-RANC CUDA Framework](https://arxiv.org/abs/2404.16208)
- [SNN Library Benchmarks](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/)

### Spiking Transformers
- [STAtten CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Spiking_Transformer_with_Spatial-Temporal_Attention_CVPR_2025_paper.pdf)
- [HAST 2025](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/csy2.70010)
- [SGSAFormer](https://www.mdpi.com/2079-9292/14/1/43)

### Sparse Training
- [u-Ticket Workload-Balanced Pruning](https://arxiv.org/html/2302.06746v2)
- [Dynamic Spatio-Temporal Pruning](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1545583/full)
- [SpikeFit EurIPS 2025](https://arxiv.org/html/2510.15542)

---

*Document generated: December 22, 2025*
*DPB Version: 0.1.0*
