# Delta-Predictive Biosensing (DPB) System Catalog v5.3.0

> **Last Updated:** December 2025
> **Framework Version:** 0.5.2
> **Total Modules:** 380+ | **Encoders:** 77+ | **Generators:** 200+ | **Decoders:** 48+
> **Language Bindings:** 7 (Python, Julia, MATLAB, R, LabVIEW, C/C++, JavaScript/WASM)
> **Platform Targets:** 18 (Native, iOS, Android, WASM, Loihi 2, SpiNNaker 2, BrainScaleS-2, LSL, RISC-V, WebGPU, WebNN, CUDA, Intel Gaudi, Graphcore IPU, FPGA, Hexagon DSP, ARM Ethos-U, Apple ANE)
> **HIPAA Compliance:** Safe Harbor, Limited Data Set, Research Pseudonymization
> **Export Formats:** ONNX, TFLite, JSON, Binary, FPGA HLS (Xilinx/Intel), Neuromorphic (Lava/PyNN/hxtorch)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Capability → Implementation Map](#2-capability--implementation-map)
3. [Implementation → Capabilities Map](#3-implementation--capabilities-map)
4. [Cross-Platform Integration](#4-cross-platform-integration)
5. [Dependency Graph](#5-dependency-graph)
6. [Gap Analysis](#6-gap-analysis)
7. [Quick Reference Tables](#7-quick-reference-tables)
8. [Learning Resources](#8-learning-resources)

---

## 1. Executive Summary

The Delta-Predictive Biosensing (DPB) Framework is a comprehensive neuromorphic biosignal processing library providing end-to-end capabilities from signal acquisition to clinical deployment.

### 1.1 System Metrics

| Metric | Count |
|--------|-------|
| Crates | 21 |
| Neuron Models | 19 + Reservoir + Multi-Compartment |
| Event Encoders | 77+ |
| Population Templates | 61+ |
| Synthetic Generators | 200+ |
| Output Decoders | 48+ |
| Signal Augmentations | 13 |
| Calibration Methods | 4 |
| Explainability Tools | 8 |
| Clinical Metrics | 100+ |
| Data Formats | 10 (WFDB, EDF, GDF, BDF, XDF, BIDS, FHIR + auto-detect) |
| **GPU Backends** | 4 (CUDA, Metal, Vulkan, WebGPU) |
| **Neuromorphic Targets** | 3 (Loihi 2, SpiNNaker 2, BrainScaleS-2) |
| **Export Formats** | 8 (ONNX, TFLite, JSON, Binary, Mobile, FPGA HLS, Lava, PyNN) |
| Visualization Types | 6 |
| Normative Databases | Age/Sex stratified (Pediatric/Adult/Geriatric) |
| Learning Resources | 5 Books, 8 Notebooks, 5 Video Scripts |
| **Language Bindings** | 7 (Python, Julia, MATLAB, R, LabVIEW, C/C++, JavaScript) |
| **Web/Streaming** | 3 (WebAssembly, WebNN, Lab Streaming Layer) |
| **Mobile NPUs** | 4 (Qualcomm Hexagon, ARM Ethos-U, Apple ANE, Samsung NPU) |
| **FPGA Targets** | 3 (Xilinx Vitis HLS, Intel HLS, Generic C) |

### 1.2 Crate Overview

| Crate | Purpose | LOC (approx) |
|-------|---------|--------------|
| **dpb-core** | Types, traits, signal processing, pipelines, I/O (10 formats) | 45,000+ |
| **dpb-encoders** | Event-based encoders, population templates | 12,000+ |
| **dpb-neurons** | 19+ neuron models, surrogates, reservoir, dendritic | 15,000+ |
| **dpb-snn** | Architectures, training, calibration, explainability, GPU, distributed, distillation, neuromorphic, neuromodulation | 65,000+ |
| **dpb-synth** | 200+ generators, augmentation, cohorts, pathology | 30,000+ |
| **dpb-norms** | Normative databases (adult, pediatric, geriatric), longitudinal | 10,000+ |
| **dpb-viz** | Dashboards, raster plots, heatmaps, network graphs, timeline | 4,500+ |
| **dpb-mobile** | iOS/Android runtime, FFI, optimization, benchmarking, NPU acceleration (Hexagon, Ethos-U, ANE) | 5,500+ |
| **dpb-cognitive** | Cognitive assessment paradigms | 4,000+ |
| **dpb-python** | PyO3 Python bindings | 3,000+ |
| **dpb-ffi** | C-compatible FFI | 1,500+ |
| **dpb-bench** | Benchmarking suite | 2,500+ |
| **dpb-wasm** | WebAssembly bindings, browser deployment, WebGPU (inc. Safari support), WebNN ML inference | 3,500+ |
| **dpb-lsl** | Lab Streaming Layer integration, liblsl FFI, real-time streaming | 3,000+ |
| **dpb-export** | ONNX, JSON, Binary, FPGA HLS, Neuromorphic (Lava/PyNN/hxtorch) export | 4,500+ |
| **dpb-federated** ✅ NEW | Privacy-preserving distributed training, FedAvg, differential privacy, gradient compression | 3,500+ |
| **dpb-clinical** ✅ NEW | Clinical utilities, ethnic stratification, treatment response, comorbidity, practice effects, HIPAA-compliant PHI de-identification | 4,000+ |

### 1.3 Language Bindings Overview

| Binding | Location | Interface | Status |
|---------|----------|-----------|--------|
| **Python** | `crates/dpb-python/` | PyO3 native extension | ✅ Complete |
| **Julia** | `bindings/julia/` | C FFI via CBinding.jl | ✅ Complete |
| **MATLAB** | `bindings/matlab/` | MEX functions | ✅ Complete |
| **R** ✅ NEW | `bindings/r/` | R6 classes via .Call() | ✅ Complete |
| **LabVIEW** ✅ NEW | `bindings/labview/` | Call Library Function Nodes | ✅ Complete |
| **C/C++** | `crates/dpb-ffi/` | cbindgen C headers | ✅ Complete |
| **JavaScript/WASM** ✅ NEW | `crates/dpb-wasm/` | wasm-bindgen | ✅ Complete |

---

## 2. Capability → Implementation Map

This section answers: **"I want to do X, where is it implemented?"**

### 2.1 Signal Acquisition & I/O

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Read PhysioNet files** | `WfdbReader` | `dpb-core/io/wfdb.rs` | .dat, .hea, annotations |
| **Read EDF/EDF+ files** | `EdfReader` | `dpb-core/io/edf.rs` | Standard polysomnography |
| **Read GDF files** | `GdfReader` | `dpb-core/io/gdf.rs` | General Data Format 1.x/2.x |
| **Read BDF files** | `BdfReader` | `dpb-core/io/bdf.rs` | 24-bit BioSemi |
| **Read XDF files** | `XdfFile` | `dpb-core/io/xdf.rs` | Lab Streaming Layer, multi-stream |
| **Read BIDS datasets** | `BidsDataset` | `dpb-core/io/bids/dataset.rs` | Brain Imaging Data Structure |
| **BIDS EEG extension** | `BidsEeg` | `dpb-core/io/bids/eeg.rs` | EEG-BIDS v1.8+ |
| **BIDS validation** | `BidsValidator` | `dpb-core/io/bids/validation.rs` | Schema compliance |
| **HL7 FHIR resources** | `FhirObservation`, `FhirBundle` | `dpb-core/io/fhir/` | Healthcare interoperability |
| **FHIR client** | `FhirClient` | `dpb-core/io/fhir/client.rs` | REST API integration |
| **Auto-detect format** | `UnifiedReader`, `detect_format` | `dpb-core/io/format_detect.rs` | Magic byte detection |
| **Write signals** | `WfdbWriter`, `EdfWriter`, etc. | `dpb-core/io/*.rs` | Multi-format export |

### 2.2 Signal Processing

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **FFT/Spectral Analysis** | `FftProcessor`, `Stft` | `dpb-core/signal/fft.rs` | Forward/inverse FFT |
| **Filtering (FIR/IIR)** | `FirFilter`, `IirFilter` | `dpb-core/signal/filter.rs` | Bandpass, notch, custom |
| **Resampling** | `downsample`, `upsample` | `dpb-core/signal/resample.rs` | Linear, polyphase |
| **Wavelet Transform** | `ContinuousWaveletTransform`, `DiscreteWaveletTransform` | `dpb-core/signal/wavelet.rs` | Morlet, Daubechies |
| **Hilbert Transform** | `hilbert_transform`, `AnalyticSignal` | `dpb-core/signal/hilbert.rs` | Instantaneous phase/freq |
| **ICA** | `FastICA` | `dpb-core/signal/ica.rs` | Blind source separation |
| **EMD/EEMD** | `EmpiricalModeDecomposition`, `EemdDecomposition` | `dpb-core/signal/emd.rs` | Nonlinear decomposition |
| **CEEMDAN** | `CeemdanDecomposition` | `dpb-core/signal/emd.rs` | Complete ensemble EMD |
| **VMD** | `VariationalModeDecomposition` | `dpb-core/signal/emd.rs` | Variational decomposition |
| **Hilbert-Huang Transform** | `hilbert_huang_transform` | `dpb-core/signal/emd.rs` | Time-frequency from IMFs |

### 2.3 Domain-Specific Analysis

| Domain | Capability | Implementation | Location |
|--------|------------|----------------|----------|
| **ECG** | R-Peak Detection | `PanTompkinsDetector` | `dpb-core/signal/ecg.rs` |
| **ECG** | QRS Morphology | `QrsMorphology`, `BeatType` | `dpb-core/signal/ecg.rs` |
| **ECG** | Arrhythmia Detection | `ArrhythmiaDetector` | `dpb-core/signal/ecg.rs` |
| **HRV** | Time-Domain | `HrvTimeDomain` (SDNN, RMSSD, pNN50) | `dpb-core/signal/hrv.rs` |
| **HRV** | Frequency-Domain | `HrvFrequencyDomain` (VLF, LF, HF) | `dpb-core/signal/hrv.rs` |
| **EEG** | Band Power | `compute_band_powers`, `EegBands` | `dpb-core/signal/eeg/bands.rs` |
| **EEG** | Artifact Detection | `detect_artifacts` | `dpb-core/signal/eeg/artifacts.rs` |
| **EEG** | Seizure Detection | `SeizureDetector` | `dpb-core/signal/eeg/seizure.rs` |
| **EEG** | ERP Analysis | `ErpAnalyzer` | `dpb-core/signal/eeg/erp.rs` |
| **PPG** | Pulse Analysis | `PpgAnalyzer`, `SpO2Result` | `dpb-core/signal/ppg.rs` |
| **EDA** | Decomposition | `EdaDecomposition` | `dpb-core/signal/eda.rs` |
| **EMG** | Burst Detection | `EmgBurst`, `FatigueMetrics` | `dpb-core/signal/emg.rs` |
| **Voice** | Acoustic Features | `F0Metrics`, `JitterMetrics` | `dpb-core/signal/voice.rs` |
| **Eye** | Saccade/Fixation | `Saccade`, `Fixation` | `dpb-core/signal/eye.rs` |
| **Respiratory** | Breath/Apnea | `BreathEvent`, `ApneaEvent` | `dpb-core/signal/respiratory.rs` |
| **Fatigue** | Multi-modal | `IntegratedFatigueMetrics` | `dpb-core/signal/fatigue.rs` |

### 2.4 Neural Network Models

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Point Neurons (19)** | LIF, ALIF, ELIF, Izhikevich, AdEx, HH, etc. | `dpb-neurons/models/*.rs` | Single-compartment |
| **Reservoir Computing** | `EchoStateNetwork`, `LiquidStateMachine` | `dpb-neurons/reservoir.rs` | Echo state, liquid state |
| **Surrogate Gradients** | FastSigmoid, Arctan, SuperSpike, etc. | `dpb-neurons/surrogates.rs` | 6 gradient functions |
| **Multi-Compartment** | `MultiCompartmentNeuron` | `dpb-neurons/dendritic/multi_compartment.rs` | Biologically detailed |
| **Dendritic Morphology** | `DendriticMorphology`, `Segment` | `dpb-neurons/dendritic/morphology.rs` | Soma, axon, dendrite |
| **Ion Channels** | `IonChannel`, `HodgkinHuxleyChannel` | `dpb-neurons/dendritic/channels.rs` | Na, K, Ca, leak |
| **Dendritic Synapses** | `DendriticSynapse`, `SpineCompartment` | `dpb-neurons/dendritic/synapse.rs` | Location-dependent |
| **Dendritic Plasticity** | `DendriticPlasticity`, `BranchSTDP` | `dpb-neurons/dendritic/plasticity.rs` | Branch-specific learning |
| **Cable Equation** | `CableIntegrator` | `dpb-neurons/dendritic/integration.rs` | Multi-scale integration |

### 2.5 Spiking Neural Network Layers & Architectures

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Spiking Layers** | SpikingLinear, SpikingConv1d/2d | `dpb-snn/layers/*.rs` | Basic building blocks |
| **Recurrent Layers** | SpikingRNN, SpikingLSTM | `dpb-snn/layers/*.rs` | Temporal processing |
| **Architectures** | Feedforward, Conv, Recurrent, Transformer | `dpb-snn/architectures/*.rs` | Pre-built networks |
| **Multi-Modal Fusion** | Early, Late, Cross-Modal, Gated | `dpb-snn/fusion/*.rs` | 8 fusion types |
| **ANN-to-SNN** | `ANNToSNNConverter` | `dpb-snn/conversion/*.rs` | Weight normalization |

### 2.6 Training & Learning

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Backprop Through Time** | BPTT trainer | `dpb-snn/training/*.rs` | Standard training |
| **Online Training** | OTTT, SLTT | `dpb-snn/training/*.rs` | Real-time learning |
| **Hebbian Learning** | `STDP`, `BCMRule`, `OjasRule` | `dpb-snn/learning/hebbian.rs` | Unsupervised |
| **Network Pruning** | `NetworkPruner`, `PruningStrategy` | `dpb-snn/optimization/pruning.rs` | Model compression |
| **Knowledge Distillation** | `TeacherStudentTrainer` | `dpb-snn/distillation/teacher_student.rs` | Model compression |
| **Spike Distillation** | `SpikeDistillation` | `dpb-snn/distillation/spike_distillation.rs` | Temporal knowledge |
| **Distillation Losses** | KL, MSE, Temporal, Rate | `dpb-snn/distillation/losses.rs` | Loss functions |
| **Self-Distillation** | `SelfDistillation` | `dpb-snn/distillation/self_distillation.rs` | Born-again networks |
| **Compression Utils** | `CompressionAnalyzer` | `dpb-snn/distillation/compression.rs` | Size/speed metrics |
| **Neuromodulation** | `NeuromodulatorSystem` | `dpb-snn/neuromodulation/modulators.rs` | DA, ACh, NE, 5-HT |
| **Dopamine System** | `DopamineSystem`, `RPEComputer` | `dpb-snn/neuromodulation/dopamine.rs` | Reward prediction |
| **Acetylcholine** | `AcetylcholineSystem` | `dpb-snn/neuromodulation/acetylcholine.rs` | Attention modulation |
| **Reward Learning** | `RewardModulatedSTDP` | `dpb-snn/neuromodulation/reward.rs` | Three-factor learning |
| **Neuromodulatory Gating** | `GatingNetwork` | `dpb-snn/neuromodulation/gating.rs` | Dynamic routing |
| **Homeostasis** | `HomeostaticRegulator` | `dpb-snn/neuromodulation/homeostasis.rs` | Activity regulation |

### 2.7 GPU & Distributed Computing

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **GPU Backend** | `Backend` enum, `GpuDevice` trait | `dpb-snn/gpu/mod.rs` | Abstraction layer |
| **CUDA Support** | `CudaDevice`, `CudaBuffer` | `dpb-snn/gpu/cuda.rs` | NVIDIA GPUs |
| **Metal Support** | `MetalDevice`, `MetalBuffer` | `dpb-snn/gpu/metal.rs` | Apple Silicon |
| **GPU Kernels** | `SpikeKernel`, `WeightUpdateKernel` | `dpb-snn/gpu/kernels.rs` | Compute operations |
| **Memory Management** | `MemoryPool`, `PinnedMemory` | `dpb-snn/gpu/memory.rs` | Efficient allocation |
| **Distributed Config** | `DistributedConfig` | `dpb-snn/distributed/mod.rs` | MPI, Gloo, NCCL |
| **Data Parallel** | `DataParallel` | `dpb-snn/distributed/partitioning.rs` | Batch splitting |
| **Model Parallel** | `ModelParallel` | `dpb-snn/distributed/partitioning.rs` | Layer distribution |
| **Pipeline Parallel** | `PipelineParallel` | `dpb-snn/distributed/partitioning.rs` | 1F1B scheduling |
| **Fault Tolerance** | `CheckpointManager`, `ElasticTrainingManager` | `dpb-snn/distributed/fault_tolerance.rs` | Recovery, elasticity |

### 2.8 Model Calibration & Explainability

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Temperature Scaling** | `TemperatureScaling` | `dpb-snn/calibration/temperature.rs` | Post-hoc calibration |
| **Platt Scaling** | `PlattScaling` | `dpb-snn/calibration/temperature.rs` | Binary classification |
| **Isotonic Calibration** | `IsotonicCalibration` | `dpb-snn/calibration/isotonic.rs` | Non-parametric |
| **Uncertainty** | `MCDropout`, `EnsembleUncertainty` | `dpb-snn/calibration/uncertainty.rs` | Epistemic/aleatoric |
| **Calibration Metrics** | `expected_calibration_error`, `brier_score` | `dpb-snn/calibration/metrics.rs` | ECE, MCE, Brier |
| **Spike Importance** | `SpikeImportance` | `dpb-snn/explain/importance.rs` | Per-spike scores |
| **Attention Maps** | `TemporalAttention`, `SpatialAttention` | `dpb-snn/explain/attention.rs` | Focus visualization |
| **Attribution** | `GradientAttribution`, `IntegratedGradients` | `dpb-snn/explain/attribution.rs` | Feature importance |
| **SHAP** | `SpikeSHAP` | `dpb-snn/explain/attribution.rs` | Shapley values |

### 2.9 Model Export & Deployment

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **ONNX Export** | `OnnxExporter` | `dpb-snn/export/onnx.rs` | Cross-platform |
| **TFLite Export** | `TfLiteExporter` | `dpb-snn/export/tflite/exporter.rs` | Mobile deployment |
| **TFLite Operators** | `TfLiteOperator`, custom ops | `dpb-snn/export/tflite/operators.rs` | SNN-specific ops |
| **TFLite Quantization** | `TfLiteQuantizer` | `dpb-snn/export/tflite/quantization.rs` | Int8, Float16 |
| **TFLite Metadata** | `TfLiteMetadata` | `dpb-snn/export/tflite/metadata.rs` | Model documentation |
| **TFLite Validation** | `TfLiteValidator` | `dpb-snn/export/tflite/validation.rs` | Export verification |
| **Neuromorphic: Loihi** | `LoihiExporter`, `LoihiNetwork` | `dpb-snn/neuromorphic/loihi.rs` | Intel Loihi |
| **Neuromorphic: SpiNNaker** | `SpinnakerExporter` | `dpb-snn/neuromorphic/spinnaker.rs` | Manchester SpiNNaker |
| **Neuromorphic: BrainScaleS** | `BrainscalesExporter` | `dpb-snn/neuromorphic/brainscales.rs` | Heidelberg BrainScaleS |
| **Hardware Constraints** | `HardwareConstraints` | `dpb-snn/neuromorphic/constraints.rs` | Chip-specific limits |
| **Network Partitioning** | `NetworkPartitioner` | `dpb-snn/neuromorphic/partitioning.rs` | Multi-chip mapping |
| **Weight Quantization** | `NeuromorphicQuantizer` | `dpb-snn/neuromorphic/quantization.rs` | Fixed-point weights |
| **Mobile Runtime** | `MobileRuntime` | `dpb-mobile/runtime.rs` | iOS/Android inference |
| **Mobile FFI** | `dpb_runtime_create`, etc. | `dpb-mobile/ffi.rs` | C bindings |

### 2.10 Visualization

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Dashboard Server** | `DashboardServer` | `dpb-viz/dashboard.rs` | WebSocket, real-time |
| **Spike Raster** | `RasterPlot`, `RasterPlot3D` | `dpb-viz/raster.rs` | Spike visualization |
| **Network Graph** | `NetworkGraph` | `dpb-viz/network.rs` | Topology view |
| **Heatmaps** | `WeightHeatmap`, `ActivationHeatmap` | `dpb-viz/heatmap.rs` | Matrix visualization |
| **Timeline** | `EventTimeline` | `dpb-viz/timeline.rs` | Temporal events |
| **Export** | `SvgExporter`, `PngExporter` | `dpb-viz/export.rs` | Multi-format |

### 2.11 Synthetic Data & Augmentation

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **ECG Generation** | Normal, arrhythmias, pathology | `dpb-synth/ecg/*.rs` | 50+ variants |
| **EEG Generation** | Normal, sleep, seizure, ERP | `dpb-synth/eeg/*.rs` | Multi-condition |
| **EMG Generation** | Normal, fatigue, pathology | `dpb-synth/emg/*.rs` | Muscle signals |
| **Virtual Cohorts** | `CohortGenerator`, `VirtualPatient` | `dpb-synth/cohort.rs` | Population simulation |
| **Pathology Models** | ALS, MS, Stroke | `dpb-synth/pathology/*.rs` | Disease progression |
| **Noise Augmentation** | Gaussian, Pink, Powerline | `dpb-synth/augmentation/noise.rs` | 5 noise types |
| **Temporal Augmentation** | Warp, Shift, Crop | `dpb-synth/augmentation/temporal.rs` | 5 temporal |
| **Spectral Augmentation** | Magnitude, Mask | `dpb-synth/augmentation/spectral.rs` | 3 spectral |

### 2.12 Normative Data

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Adult Norms** | Age/sex stratified | `dpb-norms/adult.rs` | 18-64 years |
| **Pediatric Norms** | Developmental stages | `dpb-norms/pediatric.rs` | 0-17 years |
| **Geriatric Norms** | Frailty adjustment | `dpb-norms/geriatric.rs` | 65+ years |
| **Longitudinal** | MDC, RCI, change detection | `dpb-norms/longitudinal.rs` | Serial assessment |

### 2.13 Federated Learning ✅ NEW

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **FedAvg Aggregation** | `FedAvgAggregator` | `dpb-federated/aggregation.rs` | Federated averaging |
| **Weighted Averaging** | `WeightedAverageAggregator` | `dpb-federated/aggregation.rs` | Sample-count weighted |
| **Median Aggregation** | `MedianAggregator` | `dpb-federated/aggregation.rs` | Byzantine-resistant |
| **Trimmed Mean** | `TrimmedMeanAggregator` | `dpb-federated/aggregation.rs` | Outlier-robust |
| **Differential Privacy** | `DifferentialPrivacy`, `PrivacyAccountant` | `dpb-federated/privacy.rs` | Gaussian/Laplace noise |
| **Local DP** | `LocalDP` | `dpb-federated/privacy.rs` | Client-side privacy |
| **Gradient Compression** | `GradientCompressor` | `dpb-federated/compression.rs` | TopK, RandomK, SignSGD |
| **Sparse Tensors** | `SparseTensor` | `dpb-federated/compression.rs` | Compressed communication |
| **Federated Client** | `FederatedClient` | `dpb-federated/client.rs` | Client-side training |
| **Federated Server** | `FederatedServer` | `dpb-federated/server.rs` | Aggregation coordinator |
| **Model Weights** | `ModelWeights`, `Tensor` | `dpb-federated/model.rs` | Weight representation |
| **Fed Config** | `FedConfig`, `FedConfigBuilder` | `dpb-federated/config.rs` | Configuration builder |

### 2.14 Clinical Utilities ✅ NEW

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Demographics** | `Demographics`, `Sex`, `Ethnicity` | `dpb-clinical/demographics.rs` | Patient demographics |
| **Normative Database** | `NormativeDatabase` | `dpb-clinical/normative.rs` | Population norms with stratification |
| **Population Norms** | `PopulationNorms` | `dpb-clinical/normative.rs` | Mean, SD by demographics |
| **Z-Score Calculation** | `calculate_z_score()` | `dpb-clinical/normative.rs` | Standardized scores |
| **Reliable Change Index** | `ReliableChangeIndex` | `dpb-clinical/normative.rs` | RCI calculation |
| **Treatment Response** | `TreatmentResponse` | `dpb-clinical/treatment.rs` | Pre/post intervention |
| **Effect Size** | `EffectSize` (Cohen's d, Hedges' g, Glass's delta) | `dpb-clinical/treatment.rs` | Standardized effect sizes |
| **Intervention Model** | `InterventionModel` | `dpb-clinical/treatment.rs` | Multi-timepoint modeling |
| **Comorbidity Model** | `ComorbidityModel` | `dpb-clinical/comorbidity.rs` | Multi-disease simulation |
| **Condition Interactions** | `Interaction` (synergistic/antagonistic) | `dpb-clinical/comorbidity.rs` | Disease interactions |
| **Practice Effects** | `PracticeEffectCorrector` | `dpb-clinical/practice_effects.rs` | Serial testing correction |
| **SRB Calculator** | `SRBCalculator` | `dpb-clinical/practice_effects.rs` | Standardized regression-based change |

### 2.15 Hardware Accelerators ✅ UPDATED v5.3.0

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Accelerator Trait** | `Accelerator` trait | `dpb-core/accelerators.rs` | Hardware abstraction |
| **CPU Fallback** | `CpuAccelerator` | `dpb-core/accelerators.rs` | AVX2/AVX-512 SIMD |
| **Direct CUDA** ✅ NEW | `CudaAccelerator` | `dpb-core/accelerators/cuda.rs` | Native PTX kernels |
| **CUDA Streams** | `CudaStream` | `dpb-core/accelerators/cuda.rs` | Async operations |
| **CUDA Memory** | `CudaBuffer`, unified memory | `dpb-core/accelerators/cuda.rs` | GPU memory management |
| **PTX Kernels** | Level crossing, delta modulation | `dpb-core/accelerators/kernels/*.ptx` | GPU compute kernels |
| **Intel Gaudi** | `GaudiAccelerator` | `dpb-core/accelerators/gaudi.rs` | Synapse AI SDK, TPC kernels |
| **Gaudi Graphs** | `GaudiGraph`, `GaudiKernel` | `dpb-core/accelerators/gaudi.rs` | Static graph compilation |
| **Graphcore IPU** | `IpuAccelerator` | `dpb-core/accelerators/ipu.rs` | Poplar SDK, BSP model |
| **IPU Vertices** | `IpuVertex`, `IpuGraph` | `dpb-core/accelerators/ipu.rs` | Tile-based computing |
| **PopRT Inference** | `PopRTSession` | `dpb-core/accelerators/ipu.rs` | ONNX model inference |
| **RISC-V HAL** | `RiscVHal` | `dpb-core/accelerators/riscv.rs` | Embedded RISC-V targets |
| **Fixed-Point Q16** | `FixedPoint<FRAC_BITS>`, `Q16` | `dpb-core/accelerators/riscv.rs` | FPU-less arithmetic |
| **Capabilities Query** | `AcceleratorCapabilities` | `dpb-core/accelerators.rs` | Feature detection |
| **WebGPU Browser** | `GpuEncoder` | `dpb-wasm/webgpu.rs` | Browser GPU compute |
| **WGSL Shaders** | Level crossing, delta modulation | `dpb-wasm/webgpu.rs` | Compute shaders |

### 2.16 Browser ML Inference ✅ NEW v5.3.0

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **WebNN API** | `WebNNEncoder` | `dpb-wasm/webnn.rs` | W3C Web Neural Network API |
| **Device Selection** | `WebNNDeviceType` (CPU/GPU/NPU) | `dpb-wasm/webnn.rs` | Backend selection |
| **Power Preference** | `WebNNPowerPreference` | `dpb-wasm/webnn.rs` | Battery optimization |
| **Graph Builder** | `WebNNGraphBuilder` | `dpb-wasm/webnn.rs` | Neural network operations |
| **Feature Detection** | `WebNNFeatures` | `dpb-wasm/webnn.rs` | Browser capability checks |
| **ONNX Loading** | `WebNNModelLoader` | `dpb-wasm/webnn.rs` | Model import |

### 2.17 FPGA Export ✅ NEW v5.3.0

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **HLS Exporter** | `FpgaExporter` | `dpb-export/fpga.rs` | Multi-target export |
| **Xilinx Vitis HLS** | `FpgaTarget::XilinxVitis` | `dpb-export/fpga.rs` | ap_fixed, HLS pragmas |
| **Intel HLS** | `FpgaTarget::IntelHls` | `dpb-export/fpga.rs` | ac_fixed, ihc:: types |
| **Generic HLS** | `FpgaTarget::GenericHls` | `dpb-export/fpga.rs` | Portable C (Catapult, LegUp) |
| **Encoder Config** | `EncoderConfig` | `dpb-export/fpga.rs` | Level crossing, delta, temporal |
| **AXI Stream** | AXI interface wrappers | `dpb-export/fpga.rs` | Streaming data interface |
| **Fixed-Point Types** | `FpgaDataType` | `dpb-export/fpga.rs` | Q16.16, Q8.8, custom |
| **Pipeline Pragmas** | `pipeline_ii`, `unroll_factor` | `dpb-export/fpga.rs` | Performance optimization |

### 2.18 Mobile NPU Acceleration ✅ NEW v5.3.0

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **NPU Backend Detection** | `NpuBackend::detect()` | `dpb-mobile/npu.rs` | Auto-detect NPU |
| **Qualcomm Hexagon DSP** | `NpuBackend::QualcommHexagon` | `dpb-mobile/npu.rs` | HVX vector extensions |
| **Qualcomm HTP** | `NpuBackend::QualcommHtp` | `dpb-mobile/npu.rs` | Tensor processor |
| **ARM Ethos-U55** | `NpuBackend::ArmEthosU55` | `dpb-mobile/npu.rs` | Cortex-M NPU, 32-256 MACs |
| **ARM Ethos-U65** | `NpuBackend::ArmEthosU65` | `dpb-mobile/npu.rs` | Higher throughput, 256-512 MACs |
| **Apple Neural Engine** | `NpuBackend::AppleAne` | `dpb-mobile/npu.rs` | M1/M2/A-series |
| **Samsung NPU** | `NpuBackend::SamsungNpu` | `dpb-mobile/npu.rs` | Exynos devices |
| **MediaTek APU** | `NpuBackend::MediaTekApu` | `dpb-mobile/npu.rs` | Dimensity devices |
| **NPU Encoder** | `NpuEncoder` | `dpb-mobile/npu.rs` | Spike encoding on NPU |
| **INT8 Quantization** | `quantize()`, `dequantize()` | `dpb-mobile/npu.rs` | Efficient inference |
| **Hexagon Config** | `HexagonConfig`, `HvxMode` | `dpb-mobile/npu.rs` | DSP configuration |
| **Ethos-U Config** | `EthosUConfig`, `EthosUVariant` | `dpb-mobile/npu.rs` | NPU configuration |

### 2.19 Neuromorphic Hardware Export ✅ NEW v5.3.0

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **Neuromorphic Exporter** | `NeuromorphicExporter` | `dpb-export/neuromorphic.rs` | Multi-platform export |
| **Intel Loihi 2** | `NeuromorphicTarget::Loihi2` | `dpb-export/neuromorphic.rs` | Lava framework export |
| **SpiNNaker 2** | `NeuromorphicTarget::SpiNNaker2` | `dpb-export/neuromorphic.rs` | sPyNNaker export |
| **BrainScaleS-2** | `NeuromorphicTarget::BrainScaleS2` | `dpb-export/neuromorphic.rs` | hxtorch export |
| **Generic PyNN** | `NeuromorphicTarget::GenericPyNN` | `dpb-export/neuromorphic.rs` | NEST/Brian2/NEURON |
| **Network Config** | `NetworkConfig` | `dpb-export/neuromorphic.rs` | Layers and connections |
| **Layer Config** | `LayerConfig`, `NeuronParams` | `dpb-export/neuromorphic.rs` | Neuron model params |
| **Connection Config** | `ConnectionConfig`, `ConnectionType` | `dpb-export/neuromorphic.rs` | Synaptic connections |
| **Neuron Models** | `NeuronModel` (LIF, CUBA, COBA, AdEx) | `dpb-export/neuromorphic.rs` | Supported neuron types |
| **Hardware Specs** | `HardwareSpecs` | `dpb-export/neuromorphic.rs` | Platform capabilities |

### 2.20 Embedded Targets

| Capability | Implementation | Location | Notes |
|------------|----------------|----------|-------|
| **RISC-V 32-bit IMC** | `riscv32imc-unknown-none-elf` | `.cargo/config.toml` | Common embedded |
| **RISC-V 32-bit IMAC** | `riscv32imac-unknown-none-elf` | `.cargo/config.toml` | ESP32-C3 compatible |
| **RISC-V 32-bit IMAFC** | `riscv32imafc-unknown-none-elf` | `.cargo/config.toml` | ESP32-S3, with FPU |
| **RISC-V 64-bit GC** | `riscv64gc-unknown-none-elf` | `.cargo/config.toml` | SiFive/StarFive |
| **ARM Cortex-M0/M0+** | `thumbv6m-none-eabi` | `.cargo/config.toml` | ARMv6-M |
| **ARM Cortex-M3** | `thumbv7m-none-eabi` | `.cargo/config.toml` | ARMv7-M |
| **ARM Cortex-M4/M7** | `thumbv7em-none-eabihf` | `.cargo/config.toml` | ARMv7E-M with FPU |
| **ARM Cortex-M33** | `thumbv8m.main-none-eabihf` | `.cargo/config.toml` | ARMv8-M |
| **Embedded Profile** | `release-embedded` | `.cargo/config.toml` | Size-optimized builds |

---

## 3. Implementation → Capabilities Map

This section answers: **"I found this code, what does it do?"**

### 3.1 dpb-core (Signal Processing & I/O)

```
dpb-core/
├── signal/
│   ├── mod.rs              → normalize, find_peaks, rms, envelope
│   ├── fft.rs              → FFT, STFT, spectral analysis
│   ├── filter.rs           → FIR/IIR filtering, bandpass, notch
│   ├── resample.rs         → up/downsampling, interpolation
│   ├── wavelet.rs          → CWT, DWT, multi-resolution analysis
│   ├── hilbert.rs          → analytic signal, instantaneous features
│   ├── ica.rs              → FastICA, blind source separation
│   ├── emd.rs ✅ NEW       → EMD, EEMD, CEEMDAN, VMD, Hilbert-Huang
│   ├── ecg.rs              → R-peak, QRS, arrhythmia detection
│   ├── hrv.rs              → time/frequency domain HRV
│   ├── ppg.rs              → pulse detection, SpO2
│   ├── eda.rs              → tonic/phasic, SCR
│   ├── emg.rs              → burst detection, fatigue
│   ├── voice.rs            → F0, jitter, shimmer
│   ├── eye.rs              → saccade, fixation, blink
│   ├── respiratory.rs      → breath, apnea detection
│   ├── fatigue.rs          → multi-modal fatigue
│   └── eeg/
│       ├── bands.rs        → band power analysis
│       ├── artifacts.rs    → artifact detection/removal
│       ├── seizure.rs      → seizure detection
│       └── erp.rs          → ERP component analysis
├── pipeline/
│   ├── buffer.rs           → RingBuffer, SlidingWindow, OverlapBuffer
│   ├── stage.rs            → PipelineStage trait, TimedStage
│   └── executor.rs         → PipelineExecutor, latency tracking
├── io/
│   ├── wfdb.rs             → PhysioNet WFDB format
│   ├── edf.rs              → EDF/EDF+ format
│   ├── gdf.rs              → General Data Format
│   ├── bdf.rs              → BioSemi 24-bit format
│   ├── xdf.rs              → Lab Streaming Layer XDF
│   ├── format_detect.rs    → auto-detection, UnifiedReader
│   ├── bids/ ✅ NEW
│   │   ├── mod.rs          → BIDS exports
│   │   ├── dataset.rs      → BidsDataset, dataset_description.json
│   │   ├── subject.rs      → Subject handling, participant info
│   │   ├── session.rs      → Session management
│   │   ├── eeg.rs          → EEG-BIDS extension
│   │   ├── derivatives.rs  → Processed data handling
│   │   └── validation.rs   → BIDS validator
│   └── fhir/ ✅ NEW
│       ├── mod.rs          → FHIR exports
│       ├── resources.rs    → Patient, Observation, Device
│       ├── observations.rs → Vital signs, waveforms
│       ├── bundles.rs      → Transaction bundles
│       ├── serialization.rs → JSON/XML serialization
│       ├── client.rs       → FHIR REST client
│       └── conversion.rs   → Signal-to-FHIR conversion
└── types/                  → Core type definitions
```

### 3.2 dpb-neurons (Neuron Models)

```
dpb-neurons/
├── models/
│   ├── lif.rs              → Leaky Integrate-and-Fire
│   ├── alif.rs             → Adaptive LIF
│   ├── izhikevich.rs       → Izhikevich model (20+ behaviors)
│   ├── adex.rs             → Adaptive Exponential IF
│   ├── hh.rs               → Hodgkin-Huxley
│   └── ...                 → 14+ more models
├── reservoir.rs            → ESN, LSM reservoir computing
├── surrogates.rs           → 6 surrogate gradient functions
└── dendritic/ ✅ NEW
    ├── mod.rs              → Dendritic computation exports
    ├── compartment.rs      → Compartment struct, cable properties
    ├── morphology.rs       → Tree structure, segment types
    ├── channels.rs         → Ion channels (Na, K, Ca, HCN)
    ├── synapse.rs          → Location-dependent synapses
    ├── integration.rs      → Cable equation solver
    ├── plasticity.rs       → Branch-specific STDP
    └── multi_compartment.rs → Full multi-compartment neuron
```

### 3.3 dpb-snn (Network Training & Deployment)

```
dpb-snn/
├── layers/                 → Spiking layers (Linear, Conv, RNN)
├── architectures/          → Pre-built network architectures
├── training/               → BPTT, OTTT, SLTT trainers
├── decoders/               → Rate, temporal, clinical decoders
├── fusion/                 → Multi-modal fusion (8 types)
├── conversion/             → ANN-to-SNN conversion
├── learning/
│   └── hebbian.rs          → STDP, BCM, Oja unsupervised
├── optimization/
│   └── pruning.rs          → Magnitude, structured pruning
├── calibration/
│   ├── temperature.rs      → Temperature/Platt scaling
│   ├── isotonic.rs         → Isotonic calibration
│   ├── uncertainty.rs      → MC Dropout, ensemble
│   └── metrics.rs          → ECE, Brier, reliability
├── explain/
│   ├── importance.rs       → Spike/neuron importance
│   ├── attention.rs        → Temporal/spatial attention
│   ├── attribution.rs      → Gradients, IG, SHAP
│   └── visualization.rs    → Explanation export
├── export/
│   ├── onnx.rs             → ONNX graph export
│   ├── weights.rs          → Weight serialization
│   ├── config.rs           → Model configuration
│   └── tflite/ ✅ NEW
│       ├── mod.rs          → TFLite exports
│       ├── exporter.rs     → TfLiteExporter main class
│       ├── operators.rs    → TFLite operator mapping
│       ├── tensors.rs      → Tensor serialization
│       ├── quantization.rs → Int8/Float16 quantization
│       ├── flatbuffer.rs   → FlatBuffer generation
│       ├── metadata.rs     → Model metadata
│       └── validation.rs   → Export validation
├── gpu/
│   ├── mod.rs              → Backend abstraction
│   ├── cuda.rs             → CUDA implementation
│   ├── metal.rs            → Metal implementation
│   ├── kernels.rs          → Compute kernels
│   └── memory.rs           → Memory management
├── distributed/
│   ├── mod.rs              → Distributed config
│   ├── coordinator.rs      → Gradient aggregation
│   ├── partitioning.rs     → Data/Model/Pipeline parallel
│   ├── communication.rs    → Message passing
│   ├── fault_tolerance.rs  → Checkpoints, elasticity
│   └── metrics.rs          → Throughput metrics
├── distillation/ ✅ NEW
│   ├── mod.rs              → Knowledge distillation exports
│   ├── teacher_student.rs  → Teacher-student training
│   ├── losses.rs           → KL, MSE, temporal losses
│   ├── spike_distillation.rs → Spike timing transfer
│   ├── compression.rs      → Compression analysis
│   └── self_distillation.rs → Born-again networks
├── neuromorphic/ ✅ NEW
│   ├── mod.rs              → Neuromorphic exports
│   ├── loihi.rs            → Intel Loihi export
│   ├── spinnaker.rs        → SpiNNaker export
│   ├── brainscales.rs      → BrainScaleS export
│   ├── constraints.rs      → Hardware constraints
│   ├── partitioning.rs     → Multi-chip partitioning
│   └── quantization.rs     → Fixed-point conversion
└── neuromodulation/ ✅ NEW
    ├── mod.rs              → Neuromodulation exports
    ├── modulators.rs       → Generic modulator system
    ├── dopamine.rs         → DA, reward prediction error
    ├── acetylcholine.rs    → ACh, attention modulation
    ├── reward.rs           → Reward-modulated STDP
    ├── gating.rs           → Dynamic routing
    ├── homeostasis.rs      → Activity regulation
    └── integration.rs      → Multi-system integration
```

### 3.4 dpb-synth (Synthetic Data)

```
dpb-synth/
├── ecg/                    → ECG generators (50+ variants)
├── eeg/                    → EEG generators
├── emg/                    → EMG generators
├── tremor/                 → Tremor generators
├── gait/                   → Gait generators
├── voice/                  → Voice generators
├── respiratory/            → Respiratory generators
├── pathology/
│   ├── als.rs              → ALS model, ALSFRS-R
│   ├── ms.rs               → MS model, EDSS
│   ├── stroke.rs           → Stroke model, NIHSS
│   ├── progression.rs      → Disease trajectories
│   └── medication.rs       → 12 medication classes
├── augmentation/
│   ├── noise.rs            → 5 noise augmentations
│   ├── temporal.rs         → 5 temporal augmentations
│   └── spectral.rs         → 3 spectral augmentations
└── cohort.rs               → Virtual patient cohorts
```

### 3.5 Auxiliary Crates

```
dpb-viz/                    → Visualization (dashboard, raster, heatmap)
dpb-mobile/                 → Mobile runtime (iOS, Android)
dpb-norms/                  → Normative databases
dpb-cognitive/              → Cognitive assessments
dpb-encoders/               → Event-based encoders
dpb-python/                 → Python bindings
dpb-ffi/                    → C FFI
dpb-bench/                  → Benchmarking
dpb-wasm/ ✅ NEW            → WebAssembly bindings
dpb-lsl/ ✅ NEW             → Lab Streaming Layer integration
dpb-export/ ✅ NEW          → Model export (ONNX, JSON, Binary)
```

### 3.6 Cross-Platform Bindings ✅ NEW

```
bindings/
├── r/ ✅ NEW
│   ├── DESCRIPTION         → CRAN package metadata
│   ├── NAMESPACE           → Export declarations
│   ├── R/
│   │   ├── dpb.R           → Package load, version, error handling
│   │   ├── timeseries.R    → R6 TimeSeries class
│   │   ├── spiketrain.R    → R6 SpikeTrain class
│   │   └── encoders.R      → LevelCrossingEncoder, DeltaEncoder
│   └── src/
│       └── dpb_r.c         → C bridge to DPB FFI
├── labview/ ✅ NEW
│   └── README.md           → Integration guide, VI specifications
├── julia/                  → Julia bindings (C FFI)
└── matlab/                 → MATLAB MEX bindings
```

### 3.7 dpb-wasm (WebAssembly) ✅ UPDATED v5.3.0

```
dpb-wasm/
├── Cargo.toml              → wasm-bindgen 0.2.93, WebGPU, WebNN features
└── src/
    ├── lib.rs              → Module init, PerformanceTimer
    ├── timeseries.rs       → WasmTimeSeries (Float32Array interop)
    ├── spiketrain.rs       → WasmSpikeTrain (event storage)
    ├── encoders.rs         → Level crossing, Delta, Temporal contrast
    ├── utils.rs            → Panic hook, console logging
    ├── webgpu.rs           → WebGPU compute shaders (WGSL)
    └── webnn.rs ✅ NEW     → WebNN ML Inference
                              ├── WebNNEncoder (browser ML API)
                              ├── WebNNConfig (device, power preference)
                              ├── WebNNDeviceType (Cpu, Gpu, Npu)
                              ├── WebNNGraphBuilder (neural network graphs)
                              └── feature_detect(), initialize_encoder()
```

### 3.8 dpb-lsl (Lab Streaming Layer) ✅ NEW

```
dpb-lsl/
├── Cargo.toml              → Optional async/tokio feature
└── src/
    ├── lib.rs              → ChannelFormat, stream_types constants
    ├── error.rs            → LslError enum, Result type
    ├── stream_info.rs      → StreamInfo, StreamInfoBuilder, ChannelInfo
    ├── inlet.rs            → LslInlet, InletConfig, AsyncLslInlet
    ├── outlet.rs           → LslOutlet, SpikeOutlet, OutletBuilder
    ├── resolver.rs         → StreamResolver, StreamWatcher, queries
    └── pipeline.rs         → EncodingPipeline, PipelineConfig, stats
```

### 3.9 dpb-export (Model Export) ✅ UPDATED v5.3.0

```
dpb-export/
├── Cargo.toml              → Features: onnx, tensorflow, pytorch, fpga, neuromorphic
└── src/
    ├── lib.rs              → ExportFormat, ModelExporter
    ├── error.rs            → ExportError enum
    ├── metadata.rs         → ModelMetadata, TensorSpec, DataType
    ├── encoder_export.rs   → EncoderParams, EncoderState, ExportableEncoder
    ├── json.rs             → JsonExporter, PipelineConfig
    ├── binary.rs           → BinaryExporter, BinaryImporter (embedded format)
    ├── onnx.rs             → OnnxExporter, computation graph generation
    ├── fpga.rs ✅ NEW      → FPGA HLS Export
    │                         ├── FpgaExporter (multi-target)
    │                         ├── FpgaTarget (XilinxVitis, IntelHls, GenericHls)
    │                         ├── EncoderConfig (level_crossing, delta, temporal)
    │                         ├── FpgaDataType (Float32, FixedQ16, FixedQ8)
    │                         └── AXI Stream interface generation
    └── neuromorphic.rs ✅ NEW → Neuromorphic Hardware Export
                              ├── NeuromorphicExporter (multi-platform)
                              ├── NeuromorphicTarget (Loihi2, SpiNNaker2, BrainScaleS2, PyNN)
                              ├── NetworkConfig, LayerConfig, ConnectionConfig
                              ├── NeuronModel (LIF, CUBA, COBA, ALIF, AdEx)
                              ├── HardwareSpecs per platform
                              └── Export formats: Lava, sPyNNaker, hxtorch, PyNN
```

### 3.10 dpb-federated (Federated Learning) ✅ NEW

```
dpb-federated/
├── Cargo.toml              → Features: differential-privacy, secure-aggregation, compression, async
└── src/
    ├── lib.rs              → Module exports, architecture diagram
    ├── error.rs            → FederatedError enum
    ├── config.rs           → FedConfig, FedConfigBuilder, AggregationStrategy
    ├── model.rs            → ModelWeights, Tensor, ParameterDelta
    ├── privacy.rs          → DifferentialPrivacy, PrivacyAccountant, LocalDP
    │                         ├── NoiseType (Gaussian, Laplace)
    │                         ├── PrivacyBudget, epsilon/delta tracking
    │                         └── clip_and_add_noise(), compose_privacy()
    ├── aggregation.rs      → Aggregator trait, implementations
    │                         ├── FedAvgAggregator (Federated Averaging)
    │                         ├── WeightedAverageAggregator
    │                         ├── MedianAggregator (Byzantine-resistant)
    │                         └── TrimmedMeanAggregator (outlier-robust)
    ├── client.rs           → FederatedClient, ClientState
    │                         ├── train_local_model()
    │                         ├── compute_update()
    │                         └── apply_privacy()
    ├── server.rs           → FederatedServer, ServerState
    │                         ├── aggregate_updates()
    │                         ├── broadcast_global_model()
    │                         └── coordinate_round()
    └── compression.rs      → Gradient compression
                              ├── GradientCompressor (TopK, RandomK, SignSGD)
                              ├── SparseTensor representation
                              └── ErrorFeedback accumulation
```

### 3.11 dpb-clinical (Clinical Utilities) ✅ NEW

```
dpb-clinical/
├── Cargo.toml              → Clinical analysis dependencies
└── src/
    ├── lib.rs              → Module exports
    ├── error.rs            → ClinicalError enum
    │                         ├── MissingNormativeData
    │                         ├── InvalidConfiguration
    │                         └── InsufficientData
    ├── demographics.rs     → Demographics struct
    │                         ├── Sex (Male, Female, Other)
    │                         ├── Ethnicity (10+ categories)
    │                         ├── AgeGroup (Pediatric, Adult, Geriatric)
    │                         └── Handedness (Right, Left, Ambidextrous)
    ├── normative.rs        → Normative databases
    │                         ├── NormativeDatabase (lookup by demographics)
    │                         ├── NormativeReference (metric/measure definitions)
    │                         ├── PopulationNorms (mean, std, percentiles)
    │                         ├── calculate_z_score(), percentile_rank()
    │                         └── ReliableChangeIndex (RCI calculation)
    ├── treatment.rs        → Treatment response modeling
    │                         ├── TreatmentResponse (pre/post analysis)
    │                         ├── EffectSize variants:
    │                         │   ├── CohensD (standardized mean difference)
    │                         │   ├── HedgesG (small-sample corrected)
    │                         │   └── GlassDelta (control-referenced)
    │                         ├── InterventionModel (multi-timepoint)
    │                         └── clinical_significance()
    ├── comorbidity.rs      → Multi-disease modeling
    │                         ├── ComorbidityModel (disease interactions)
    │                         ├── Condition (name, severity, onset)
    │                         ├── Interaction (synergistic, antagonistic)
    │                         └── CombinedEffect (composite impact)
    └── practice_effects.rs → Serial testing corrections
                              ├── PracticeEffectCorrector
                              ├── SRBCalculator (standardized regression-based)
                              ├── SerialAssessment (multi-timepoint tracking)
                              └── estimate_learning_effect()
```

### 3.12 dpb-core/accelerators ✅ UPDATED v5.3.0

```
dpb-core/accelerators/
├── mod.rs                  → Module exports, Accelerator trait
│   ├── Accelerator trait   → Generic hardware abstraction
│   │   ├── name() -> &str
│   │   ├── capabilities() -> AcceleratorCapabilities
│   │   ├── compute_fft() -> Result
│   │   ├── compute_conv() -> Result
│   │   └── compute_matmul() -> Result
│   └── AcceleratorCapabilities → Feature detection
├── cpu.rs                  → CpuAccelerator (always available)
├── gaudi.rs                → Intel Gaudi (Synapse AI SDK)
├── ipu.rs                  → Graphcore IPU (Poplar SDK)
├── riscv.rs                → RISC-V HAL (embedded, fixed-point)
└── cuda.rs ✅ NEW          → Direct CUDA Support
                              ├── CudaAccelerator (PTX kernel execution)
                              ├── CudaModule (PTX module loading)
                              ├── CudaKernel (kernel launch parameters)
                              ├── CudaStream (async execution, events)
                              ├── CudaBuffer (device memory)
                              └── kernels/
                                  └── level_crossing.ptx (spike detection)
```

### 3.13 dpb-wasm/webgpu ✅ NEW

```
dpb-wasm/src/webgpu.rs
├── GpuEncoder              → Browser-side GPU encoding
│   ├── new() -> Promise<GpuEncoder>
│   ├── encode_level_crossing() -> WasmSpikeTrain
│   └── encode_delta() -> WasmSpikeTrain
├── GpuEncoderConfig        → GPU configuration
│   ├── workgroup_size: u32
│   └── max_spikes: u32
├── WGSL Compute Shaders:
│   ├── level_crossing_shader → Threshold-based spike detection
│   └── delta_modulation_shader → Change-based encoding
└── Internal:
    ├── create_device() → WebGPU device setup
    ├── create_pipeline() → Compute pipeline
    └── execute_shader() → GPU dispatch
```

### 3.14 dpb-mobile/npu ✅ NEW v5.3.0

```
dpb-mobile/src/npu.rs
├── NpuBackend enum         → Backend selection
│   ├── QualcommHexagon     → Hexagon DSP/HVX/HTP
│   ├── ArmEthosU           → ARM Ethos-U55/U65
│   ├── AppleAne            → Apple Neural Engine
│   └── SamsungNpu          → Samsung Exynos NPU
├── NpuEncoder              → Hardware-accelerated spike encoding
│   ├── new(backend, config) -> Result<Self>
│   ├── encode_level_crossing() -> Result<Vec<Spike>>
│   ├── encode_delta() -> Result<Vec<Spike>>
│   └── encode_temporal_contrast() -> Result<Vec<Spike>>
├── HexagonConfig           → Qualcomm Hexagon settings
│   ├── dsp_clock_mhz       → DSP clock speed
│   ├── hvx_threads         → HVX vector threads
│   └── power_level         → Power management
├── EthosUConfig            → ARM Ethos-U settings
│   ├── macs                → MAC operations/cycle
│   ├── sram_kb             → Available SRAM
│   └── burst_length        → Memory burst size
└── AppleAneConfig          → Apple ANE settings
    ├── use_fp16            → Half precision
    └── batch_size          → Inference batch
```

---

## 4. Cross-Platform Integration

This section provides detailed documentation for all cross-platform integrations.

### 4.1 Language Binding Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            LANGUAGE BINDINGS ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Native Rust Crates                    FFI Layer                 Language Bindings  │
│  ─────────────────                     ─────────                 ─────────────────  │
│                                                                                      │
│  ┌─────────────┐                   ┌─────────────┐            ┌─────────────────┐   │
│  │ dpb-core    │──────────────────►│ dpb-ffi     │───────────►│ Python (PyO3)   │   │
│  │ dpb-encoders│                   │ (C headers) │            │ Julia (CBinding)│   │
│  │ dpb-neurons │                   └──────┬──────┘            │ MATLAB (MEX)    │   │
│  │ dpb-snn     │                          │                   │ R (.Call())     │   │
│  └─────────────┘                          │                   │ LabVIEW (CLFN)  │   │
│                                           │                   │ C/C++ (direct)  │   │
│                                           │                   └─────────────────┘   │
│                                           │                                          │
│                                           ▼                                          │
│  ┌─────────────┐                   ┌─────────────┐            ┌─────────────────┐   │
│  │ dpb-wasm    │◄──────────────────│ dpb-export  │───────────►│ ONNX Runtime    │   │
│  │ (browser)   │                   │ (formats)   │            │ TensorFlow Lite │   │
│  └──────┬──────┘                   └─────────────┘            │ Edge Devices    │   │
│         │                                                     └─────────────────┘   │
│         ▼                                                                            │
│  ┌─────────────────┐               ┌─────────────┐            ┌─────────────────┐   │
│  │ JavaScript/     │               │ dpb-lsl     │───────────►│ LSL Ecosystem   │   │
│  │ TypeScript      │               │ (streaming) │            │ OpenBCI, Muse   │   │
│  │ Web Apps        │               └─────────────┘            │ BrainVision     │   │
│  └─────────────────┘                                          └─────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 R Bindings

| Capability | Implementation | Notes |
|------------|----------------|-------|
| **TimeSeries** | `TimeSeries` R6 class | Matrix storage, FFI bridge |
| **SpikeTrain** | `SpikeTrain` R6 class | Event-based storage |
| **Level Crossing Encoder** | `LevelCrossingEncoder` R6 class | Threshold-based encoding |
| **Delta Encoder** | `DeltaEncoder` R6 class | Change-based encoding |
| **File I/O** | `timeseries_from_file()` | CSV, custom formats |
| **Version Info** | `dpb_version()` | Library version |
| **Error Handling** | `dpb_last_error()`, `dpb_clear_error()` | Thread-local errors |

**Installation:**
```r
# From source
install.packages("devtools")
devtools::install_local("bindings/r")

# Usage
library(dpb)
ts <- TimeSeries$new(data_matrix, sample_rate = 256)
encoder <- LevelCrossingEncoder$new(threshold = 0.1, num_channels = 8)
spikes <- encoder$encode(ts)
```

### 4.3 LabVIEW Integration

| Component | VI/Function | Description |
|-----------|-------------|-------------|
| **Library Loading** | Call Library Function Node | Load dpb_ffi.dll/.so/.dylib |
| **TimeSeries** | DPB_TimeSeries_Create.vi | Create from 2D array |
| **Encoding** | DPB_Encoder_Encode.vi | Generic encoder wrapper |
| **Spike Output** | DPB_SpikeTrain_GetData.vi | Extract spike data |
| **Error Handling** | DPB_GetLastError.vi | Thread-safe error messages |

### 4.4 WebAssembly (WASM)

| Capability | Implementation | Notes |
|------------|----------------|-------|
| **WasmTimeSeries** | `WasmTimeSeries` class | Float32Array integration |
| **WasmSpikeTrain** | `WasmSpikeTrain` class | JavaScript event access |
| **Level Crossing** | `WasmLevelCrossingEncoder` | Browser-side encoding |
| **Delta Encoder** | `WasmDeltaEncoder` | Adaptive thresholds |
| **Temporal Contrast** | `WasmTemporalContrastEncoder` | Event-based vision style |
| **Performance Timer** | `PerformanceTimer` | Benchmarking utilities |

**Usage:**
```javascript
import init, { WasmTimeSeries, WasmLevelCrossingEncoder } from 'dpb-wasm';

await init();
const ts = WasmTimeSeries.new(new Float32Array(data), numChannels, sampleRate);
const encoder = WasmLevelCrossingEncoder.new(0.1, numChannels);
const spikes = encoder.encode(ts);
console.log(`Generated ${spikes.spikeCount()} spikes`);
```

### 4.5 Lab Streaming Layer (LSL)

| Capability | Implementation | Notes |
|------------|----------------|-------|
| **Stream Discovery** | `StreamResolver` | Find streams by name/type/property |
| **Data Reception** | `LslInlet` | Pull samples/chunks |
| **Data Transmission** | `LslOutlet` | Push samples/chunks |
| **Spike Streaming** | `SpikeOutlet` | Specialized spike output |
| **Real-time Pipeline** | `EncodingPipeline` | Live encoding with stats |
| **Async Support** | `AsyncLslInlet` | Tokio integration |

**Pipeline Example:**
```rust
use dpb_lsl::{EncodingPipeline, PipelineBuilder, EncoderType};

let pipeline = PipelineBuilder::new()
    .input_stream("MyEEG")
    .input_type("EEG")
    .output_name("DPB_Spikes")
    .encoder(EncoderType::LevelCrossing)
    .threshold(0.1)
    .adaptive(true)
    .build();

pipeline.start(5.0)?;  // 5 second timeout
```

### 4.6 Model Export Formats

| Format | Exporter | Use Case |
|--------|----------|----------|
| **ONNX** | `OnnxExporter` | Universal deployment, cross-platform |
| **JSON** | `JsonExporter` | Configuration, human-readable params |
| **Binary** | `BinaryExporter` | Embedded systems, fast loading |
| **TFLite** | (via dpb-snn) | Mobile deployment |

**Export Example:**
```rust
use dpb_export::{ModelExporter, JsonExporter};

let exporter = ModelExporter::new()
    .with_name("ECG_Encoder")
    .with_version("1.0.0")
    .with_metadata("encoder_type", "level_crossing");

exporter.export_json("model_config.json", &encoder)?;
exporter.export_binary("model.dpb", &encoder)?;

#[cfg(feature = "onnx")]
exporter.export_onnx("model.onnx", &encoder)?;
```

---

## 5. Dependency Graph

### 5.1 Crate Dependencies

```
                         ┌───────────────────────────────────────────────────────────────┐
                         │                   LANGUAGE BINDINGS LAYER                      │
                         │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐       │
                         │  │dpb-python │ │ dpb-ffi   │ │ dpb-wasm  │ │ R/LabVIEW│       │
                         │  │  (PyO3)   │ │(C headers)│ │(WASM+GPU) │ │(bindings)│       │
                         │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └────┬─────┘       │
                         └────────┼─────────────┼─────────────┼────────────┼─────────────┘
                                  │             │             │            │
              ┌───────────────────┼─────────────┼─────────────┼────────────┼──────────────┐
              │                   │             │             │            │              │
              ▼                   ▼             ▼             ▼            ▼              │
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
    │   dpb-bench     │  │   dpb-snn       │  │   dpb-synth     │  │ dpb-federated    │  │
    └────────┬────────┘  │ ✅ +distillation │  └────────┬────────┘  │ ✅ NEW           │  │
             │           │ ✅ +neuromorphic │           │           │ FedAvg, DP       │  │
             │           │ ✅ +neuromodulat │           │           │ Compression      │  │
             │           └────────┬────────┘           │           └────────┬─────────┘  │
             │                    │                    │                    │            │
             │           ┌────────┴────────┐           │                    │            │
             │           │                 │           │                    │            │
             │           ▼                 ▼           │                    │            │
             │  ┌─────────────────┐ ┌─────────────────┐│                    │            │
             │  │  dpb-neurons    │ │  dpb-encoders   ││                    │            │
             │  │ ✅ +dendritic   │ └────────┬────────┘│                    │            │
             │  └────────┬────────┘          │         │                    │            │
             │           │                   │         │                    │            │
             │           └─────────┬─────────┘         │                    │            │
             │                     │                   │                    │            │
             │                     ▼                   │                    │            │
             │          ┌─────────────────┐            │                    │            │
             │          │   dpb-norms     │◄───────────┤                    │            │
             │          └────────┬────────┘            │                    │            │
             │                   │                     │                    │            │
             └───────────────────┼─────────────────────┼────────────────────┼────────────┘
                                 │                     │                    │
           ┌─────────────────────┼─────────────────────┼────────────────────┘
           │                     │                     │
           ▼                     ▼                     ▼
 ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
 │   dpb-viz       │  │   dpb-core      │  │   dpb-mobile    │  │   dpb-lsl       │
 │                 │  │ ✅ +BIDS/FHIR   │  │                 │  │ ✅ +liblsl FFI  │
 │                 │  │ ✅ +EMD         │  │                 │  │ LSL streaming   │
 │                 │  │ ✅ +accelerators│  │                 │  │                 │
 └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
           │                     │                     │                    │
           │                     ▼                     │                    │
           │          ┌─────────────────┐              │                    │
           │          │  dpb-clinical   │◄─────────────┤                    │
           │          │ ✅ NEW          │              │                    │
           │          │ Norms, Tx resp  │              │                    │
           │          │ Comorbidity     │              │                    │
           │          └─────────────────┘              │                    │
           │                     │                     │                    │
           └─────────────────────┼─────────────────────┼────────────────────┘
                                 │                     │
                                 ▼                     ▼
                      ┌─────────────────┐   ┌─────────────────┐
                      │   dpb-export    │   │   CI/CD         │
                      │ ONNX/JSON/Bin   │   │ benchmarks.yml  │
                      └─────────────────┘   │ Regression      │
                                            └─────────────────┘
```

### 5.2 Feature Flags

```
dpb-snn features:
├── gpu                → GPU acceleration (base)
│   ├── cuda           → NVIDIA CUDA support
│   └── metal          → Apple Metal support
├── distributed        → Multi-node training
├── distillation       → Knowledge distillation
├── neuromorphic       → Hardware export (Loihi, SpiNNaker, BrainScaleS)
└── neuromodulation    → Neuromodulatory systems

dpb-neurons features:
└── dendritic          → Multi-compartment neurons

dpb-core features:
├── bids               → BIDS format support
├── fhir               → HL7 FHIR support
├── emd                → Empirical mode decomposition
├── cuda ✅ NEW        → Direct CUDA support (PTX kernels)
├── intel-gaudi        → Intel Gaudi accelerator (Synapse AI)
├── graphcore-ipu      → Graphcore IPU accelerator (Poplar)
├── riscv              → RISC-V embedded HAL
└── hardware-accelerators → All hardware accelerators

dpb-mobile features:
├── ios                → iOS-specific (Metal, CoreML)
├── android            → Android-specific (NNAPI, Vulkan)
├── quantized          → Quantized inference
├── hexagon ✅ NEW     → Qualcomm Hexagon DSP/HVX/HTP
├── ethos-u ✅ NEW     → ARM Ethos-U55/U65 NPU
└── apple-ane ✅ NEW   → Apple Neural Engine

dpb-snn/export features:
├── onnx               → ONNX export
└── tflite             → TensorFlow Lite export

dpb-wasm features:
├── console_error_panic_hook → Better panic messages in browser
├── webgpu             → WebGPU compute shaders (WGSL)
└── webnn ✅ NEW       → WebNN ML inference API

dpb-lsl features:
├── async              → Tokio async runtime support
└── native             → Link against liblsl C library ✅ NEW

dpb-export features:
├── onnx               → ONNX graph export
├── tensorflow         → TensorFlow export (planned)
├── pytorch            → PyTorch export (planned)
├── fpga ✅ NEW        → FPGA HLS export (Xilinx Vitis, Intel HLS)
├── neuromorphic ✅ NEW → Neuromorphic export (Lava, PyNN, hxtorch)
└── full               → All export formats

dpb-federated features: ✅ NEW
├── differential-privacy → Gaussian/Laplace noise, privacy budget
├── secure-aggregation → Cryptographic aggregation
├── compression        → Gradient compression (TopK, RandomK, SignSGD)
└── async              → Async/Tokio support
```

### 5.3 Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPLETE DATA FLOW v5.0                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│  │     I/O      │──►│   Signal     │──►│   Encoders   │──►│     SNN      │             │
│  │ WFDB/EDF/GDF │   │  Processing  │   │   (77+)      │   │  Training    │             │
│  │ BDF/XDF/BIDS │   │  ECG/EEG/EMG │   │              │   │GPU/Dist/Neuro│             │
│  │ FHIR         │   │  EMD/Wavelet │   │              │   │  modulation  │             │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘             │
│         │                  │                  │                   │                     │
│         │                  │                  │                   ▼                     │
│         │                  │                  │           ┌──────────────┐              │
│         │                  │                  │           │  Distillation│              │
│         │                  │                  │           │  Pruning     │              │
│         │                  │                  │           │  Compression │              │
│         │                  │                  │           └──────┬───────┘              │
│         │                  │                  │                   │                     │
│         │                  │                  │                   ▼                     │
│         │                  │                  │    ┌──────────────────────────────┐     │
│         │                  │                  │    │         EXPORT               │     │
│         │                  │                  │    ├──────────────────────────────┤     │
│         │                  │                  │    │ ONNX  │ TFLite │ Neuromorphic│     │
│         │                  │                  │    │       │        │Loihi/SpiNN/ │     │
│         │                  │                  │    │       │        │BrainScaleS  │     │
│         │                  │                  │    └──────────────────────────────┘     │
│         │                  │                  │                   │                     │
│         │                  │                  │                   ▼                     │
│         │                  │                  │           ┌──────────────┐              │
│         │                  │                  │           │   Decoders   │              │
│         │                  │                  │           │ Rate/Temporal│              │
│         │                  │                  │           │  Clinical    │              │
│         │                  │                  │           └──────┬───────┘              │
│         │                  │                  │                   │                     │
│         │                  │                  │     ┌─────────────┼─────────────┐       │
│         │                  │                  │     │             │             │       │
│         │                  │                  │     ▼             ▼             ▼       │
│         │                  │                  │ ┌────────┐  ┌──────────┐  ┌──────────┐ │
│         │                  │                  │ │Calibrate│  │ Explain  │  │   Viz    │ │
│         │                  │                  │ │Uncertain│  │ SHAP/IG  │  │Dashboard │ │
│         │                  │                  │ └────────┘  └──────────┘  └──────────┘ │
│         │                  │                  │                                         │
│         ▼                  ▼                  │                   │                     │
│  ┌──────────────┐   ┌──────────────┐         │                   ▼                     │
│  │    Synth     │   │    Norms     │         │           ┌──────────────┐              │
│  │  Generators  │   │ Pediatric    │         │           │    Mobile    │              │
│  │  Augment     │   │ Adult        │         │           │  iOS/Android │              │
│  │  Cohorts     │   │ Geriatric    │         │           │   Runtime    │              │
│  └──────────────┘   └──────────────┘         │           └──────────────┘              │
│                                              │                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Gap Analysis

This section answers: **"What's needed but missing?"**

### 6.1 Implementation Status Matrix

| Category | Feature | Status | Priority |
|----------|---------|--------|----------|
| **Signal Processing** | FFT/Filtering/Resampling | ✅ Complete | - |
| **Signal Processing** | Wavelet/Hilbert/ICA | ✅ Complete | - |
| **Signal Processing** | EMD/EEMD/CEEMDAN/VMD | ✅ Complete | - |
| **Signal Processing** | ECG/HRV/EEG Analysis | ✅ Complete | - |
| **Data Formats** | WFDB/EDF/GDF/BDF/XDF | ✅ Complete | - |
| **Data Formats** | BIDS | ✅ Complete | - |
| **Data Formats** | HL7 FHIR | ✅ Complete | - |
| **Neural Networks** | Point Neurons (19+) | ✅ Complete | - |
| **Neural Networks** | Reservoir Computing | ✅ Complete | - |
| **Neural Networks** | Multi-Compartment/Dendritic | ✅ Complete | - |
| **Training** | BPTT/OTTT/SLTT | ✅ Complete | - |
| **Training** | Hebbian (STDP/BCM/Oja) | ✅ Complete | - |
| **Training** | Knowledge Distillation | ✅ Complete | - |
| **Training** | Neuromodulation | ✅ Complete | - |
| **Infrastructure** | GPU (CUDA/Metal) | ✅ Complete | - |
| **Infrastructure** | Distributed Training | ✅ Complete | - |
| **Calibration** | Temperature/Isotonic | ✅ Complete | - |
| **Explainability** | Importance/Attention/SHAP | ✅ Complete | - |
| **Export** | ONNX | ✅ Complete | - |
| **Export** | TensorFlow Lite | ✅ Complete | - |
| **Export** | Neuromorphic Hardware | ✅ Complete | - |
| **Export** | JSON/Binary (dpb-export) | ✅ Complete | - |
| **Visualization** | Dashboard/Raster/Heatmap | ✅ Complete | - |
| **Mobile** | iOS/Android Runtime | ✅ Complete | - |
| **Normative** | Pediatric/Adult/Geriatric | ✅ Complete | - |
| **Learning** | Books/Notebooks/Videos | ✅ Complete | - |
| **Bindings** | Python (PyO3) | ✅ Complete | - |
| **Bindings** | Julia (C FFI) | ✅ Complete | - |
| **Bindings** | MATLAB (MEX) | ✅ Complete | - |
| **Bindings** | R (.Call/R6) | ✅ Complete | - |
| **Bindings** | LabVIEW (CLFN) | ✅ Complete | - |
| **Bindings** | C/C++ (cbindgen) | ✅ Complete | - |
| **Web** | WebAssembly (dpb-wasm) | ✅ Complete | - |
| **Streaming** | Lab Streaming Layer (dpb-lsl) | ✅ Complete | - |
| **GPU** | Direct CUDA (PTX kernels) | ✅ Complete | - |
| **Web ML** | WebNN Browser API | ✅ Complete | - |
| **Export** | FPGA HLS (Xilinx/Intel) | ✅ Complete | - |
| **Mobile** | NPU Acceleration (Hexagon/Ethos-U/ANE) | ✅ Complete | - |
| **Export** | Neuromorphic (Lava/PyNN/hxtorch) | ✅ Complete | - |

### 6.2 Recently Completed ✅ (v5.3.0)

| Feature | Implementation | Status |
|---------|----------------|--------|
| **Direct CUDA Support** | `dpb-core/accelerators/cuda.rs` - PTX kernels, CudaAccelerator, CudaStream, CudaBuffer, CudaModule | ✅ Complete |
| **WebNN ML Inference** | `dpb-wasm/webnn.rs` - Browser ML API, WebNNEncoder, device selection (CPU/GPU/NPU), graph builder | ✅ Complete |
| **FPGA HLS Export** | `dpb-export/fpga.rs` - FpgaExporter (Xilinx Vitis, Intel HLS, Generic C), AXI Stream, fixed-point Q16.16 | ✅ Complete |
| **Mobile NPU Acceleration** | `dpb-mobile/npu.rs` - NpuEncoder, Qualcomm Hexagon DSP/HVX/HTP, ARM Ethos-U55/U65, Apple ANE | ✅ Complete |
| **Neuromorphic Export** | `dpb-export/neuromorphic.rs` - Lava (Loihi 2), sPyNNaker (SpiNNaker 2), hxtorch (BrainScaleS-2), PyNN | ✅ Complete |
| **WebGL Deprecation Docs** | `docs/WEBGL_DEPRECATION.md` - Migration guide, browser compatibility matrix, fallback strategy | ✅ Complete |
| **Deployment Guide** | `docs/DEPLOYMENT_GUIDE.md` - 18 platform targets, all deployment regimes documented | ✅ Complete |

### 6.2.1 Previously Completed (v5.2.0)

| Feature | Implementation | Status |
|---------|----------------|--------|
| **HIPAA/PHI De-identification** | `dpb-clinical/phi.rs` - Safe Harbor, Limited Data Set, Research Pseudonymization, 18 PHI identifiers | ✅ Complete |
| **K-Anonymity & L-Diversity** | `dpb-clinical/phi.rs` - Privacy validation with k-anonymity and l-diversity checks | ✅ Complete |
| **Unit Test Coverage** | `dpb-federated/tests/`, `dpb-clinical/tests/` - Comprehensive integration tests | ✅ Complete |
| **WebGPU Browser Testing** | `dpb-wasm/tests/web/index.html` - Full browser test harness with visual UI | ✅ Complete |
| **Real liblsl Runtime** | `dpb-lsl/native.rs`, `build.rs` - Native liblsl FFI with pkg-config detection | ✅ Complete |
| **Intel Gaudi SDK** | `dpb-core/accelerators/gaudi.rs` - Full Synapse AI SDK, TPC kernels, HBM memory, graph compilation | ✅ Complete |
| **Graphcore IPU SDK** | `dpb-core/accelerators/ipu.rs` - Full Poplar SDK, BSP model, tile-based computing, PopRT inference | ✅ Complete |
| **RISC-V HAL** | `dpb-core/accelerators/riscv.rs` - Embedded HAL, fixed-point Q16, DMA, interrupt-driven encoding | ✅ Complete |

### 6.2.2 Previously Completed (v5.1.0)

| Feature | Implementation | Status |
|---------|----------------|--------|
| **Federated Learning** | `dpb-federated` crate - FedAvg, differential privacy, gradient compression, secure aggregation | ✅ Complete |
| **Ethnic Stratification** | `dpb-clinical` - NormativeDatabase with demographic stratification (age/sex/ethnicity) | ✅ Complete |
| **Treatment Response** | `dpb-clinical` - TreatmentResponse, EffectSize (Cohen's d, Hedges' g, Glass's delta) | ✅ Complete |
| **Comorbidity Modeling** | `dpb-clinical` - ComorbidityModel with condition interactions | ✅ Complete |
| **Practice Effects** | `dpb-clinical` - PracticeEffectCorrector, SRBCalculator | ✅ Complete |
| **Performance Regression** | `.github/workflows/benchmarks.yml` - CI/CD benchmarks with regression detection | ✅ Complete |
| **liblsl Integration** | `dpb-lsl/ffi.rs` - Complete liblsl C library bindings | ✅ Complete |
| **Additional Hardware** | `dpb-core/accelerators.rs` - Intel Gaudi, Graphcore IPU abstractions | ✅ Complete |
| **WebGPU Acceleration** | `dpb-wasm/webgpu.rs` - GPU compute shaders for spike encoding | ✅ Complete |
| **RISC-V Targets** | `.cargo/config.toml` - riscv32imc, riscv32imac, riscv64gc configurations | ✅ Complete |

### 6.3 Remaining Gaps

#### HIGH Priority

| Gap | Description | Effort | Impact |
|-----|-------------|--------|--------|
| **Clinical Validation** | Tests against MIT-BIH, CHB-MIT, PhysioNet datasets | Medium | Regulatory compliance |
| **FDA 510(k) Documentation** | Pre-submission documentation templates | High | Regulatory pathway |

#### MEDIUM Priority

| Gap | Description | Effort | Impact |
|-----|-------------|--------|--------|
| **TPC Kernel Optimization** | Optimized TPC-C kernels for Intel Gaudi production | Medium | Performance |
| **IPU Vertex Optimization** | Optimized Poplar vertices for Graphcore IPU production | Medium | Performance |
| **Continuous liblsl Integration** | CI/CD with real liblsl library testing | Low | Production readiness |

#### LOW Priority

| Gap | Description | Effort | Impact |
|-----|-------------|--------|--------|
| **RISC-V Production Testing** | Testing on physical ESP32-C3, SiFive boards | Low | Hardware validation |

### 6.3.1 Completed Gaps (v5.3.0)

| Previously Gap | Now Implementation | Status |
|----------------|-------------------|--------|
| ~~Direct CUDA Support~~ | `dpb-core/accelerators/cuda.rs` | ✅ Complete |
| ~~WebNN Browser API~~ | `dpb-wasm/webnn.rs` | ✅ Complete |
| ~~FPGA HLS Export~~ | `dpb-export/fpga.rs` | ✅ Complete |
| ~~Mobile NPU Acceleration~~ | `dpb-mobile/npu.rs` | ✅ Complete |
| ~~Neuromorphic Export~~ | `dpb-export/neuromorphic.rs` | ✅ Complete |
| ~~WebGL Deprecation Docs~~ | `docs/WEBGL_DEPRECATION.md` | ✅ Complete |
| ~~WebGPU Safari Support~~ | `dpb-wasm/webgpu_safari.rs` | ✅ Complete |

### 6.3.2 Completed Gaps (v5.2.0)

| Previously Gap | Now Implementation | Status |
|----------------|-------------------|--------|
| ~~HIPAA/PHI Tools~~ | `dpb-clinical/phi.rs` | ✅ Complete |
| ~~Unit Test Coverage~~ | `dpb-federated/tests/`, `dpb-clinical/tests/` | ✅ Complete |
| ~~Real liblsl Runtime~~ | `dpb-lsl/native.rs`, `build.rs` | ✅ Complete |
| ~~WebGPU Browser Testing~~ | `dpb-wasm/tests/web/` | ✅ Complete |
| ~~Intel Gaudi SDK~~ | `dpb-core/accelerators/gaudi.rs` | ✅ Complete |
| ~~Graphcore IPU SDK~~ | `dpb-core/accelerators/ipu.rs` | ✅ Complete |
| ~~RISC-V HAL~~ | `dpb-core/accelerators/riscv.rs` | ✅ Complete |

### 6.4 Module Completeness

| Module | Core | Tests | Docs | Examples |
|--------|:----:|:-----:|:----:|:--------:|
| dpb-core/signal | ✅ | ✅ | ✅ | ✅ |
| dpb-core/io (all formats) | ✅ | ✅ | ✅ | ⚠️ |
| dpb-core/io/bids | ✅ | ✅ | ✅ | ⚠️ |
| dpb-core/io/fhir | ✅ | ✅ | ✅ | ⚠️ |
| dpb-core/accelerators | ✅ | ✅ | ✅ | ⚠️ |
| dpb-core/accelerators/gaudi | ✅ | ✅ | ✅ | ⚠️ |
| dpb-core/accelerators/ipu | ✅ | ✅ | ✅ | ⚠️ |
| dpb-core/accelerators/riscv | ✅ | ✅ | ✅ | ⚠️ |
| dpb-core/accelerators/cuda | ✅ | ✅ | ✅ | ⚠️ |
| dpb-neurons/dendritic | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/distillation | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/neuromorphic | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/neuromodulation | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/export/tflite | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/gpu | ✅ | ✅ | ✅ | ✅ |
| dpb-snn/distributed | ✅ | ✅ | ✅ | ✅ |
| dpb-viz | ✅ | ✅ | ✅ | ⚠️ |
| dpb-mobile | ✅ | ✅ | ✅ | ✅ |
| dpb-federated | ✅ | ✅ | ✅ | ⚠️ |
| dpb-clinical | ✅ | ✅ | ✅ | ⚠️ |
| dpb-clinical/phi | ✅ | ✅ | ✅ | ⚠️ |
| dpb-lsl | ✅ | ✅ | ✅ | ✅ |
| dpb-lsl/native | ✅ | ✅ | ✅ | ⚠️ |
| dpb-wasm | ✅ | ✅ | ✅ | ⚠️ |
| dpb-wasm/webnn | ✅ | ✅ | ✅ | ⚠️ |
| dpb-mobile/npu | ✅ | ✅ | ✅ | ⚠️ |
| dpb-export | ✅ | ✅ | ✅ | ✅ |
| dpb-export/fpga | ✅ | ✅ | ✅ | ⚠️ |
| dpb-export/neuromorphic | ✅ | ✅ | ✅ | ⚠️ |
| bindings/r | ✅ | ✅ | ✅ | ✅ |
| bindings/labview | ✅ | - | ✅ | ✅ |

Legend: ✅ Complete | ⚠️ Partial (needs more examples) | ❌ Missing

### 6.5 Recommended Next Steps

1. **Immediate (Testing & Validation)** ✅ Mostly Complete
   - ~~Add unit tests for dpb-federated, dpb-clinical~~ ✅ Complete
   - Run validation against public datasets (MIT-BIH, CHB-MIT, PhysioNet) ⏳ Pending
   - ~~End-to-end browser testing for WebGPU features~~ ✅ Complete
   - Test federated learning with simulated multi-site setup ⏳ Pending

2. **Short-term (Clinical Readiness)** ✅ Mostly Complete
   - ~~Implement HIPAA/PHI de-identification utilities~~ ✅ Complete
   - Add clinical validation test suite ⏳ Pending
   - Create regulatory documentation templates (FDA 510(k)) ⏳ Pending
   - ~~Link dpb-lsl against production liblsl.so~~ ✅ Complete

3. **Medium-term (Hardware Production)** ✅ Complete
   - ~~Integrate Intel Gaudi Synapse AI SDK~~ ✅ Complete (`dpb-core/accelerators/gaudi.rs`)
   - ~~Integrate Graphcore Poplar SDK~~ ✅ Complete (`dpb-core/accelerators/ipu.rs`)
   - ~~RISC-V hardware abstraction layer~~ ✅ Complete (`dpb-core/accelerators/riscv.rs`)
   - ~~Direct CUDA support~~ ✅ Complete (`dpb-core/accelerators/cuda.rs`)
   - ~~Mobile NPU acceleration~~ ✅ Complete (`dpb-mobile/npu.rs`)
   - LabVIEW example VIs and palettes ⏳ Pending

4. **Deployment Expansion (v5.3.0)** ✅ Complete
   - ~~FPGA HLS export path~~ ✅ Complete (`dpb-export/fpga.rs`)
   - ~~Neuromorphic hardware export~~ ✅ Complete (`dpb-export/neuromorphic.rs`)
   - ~~WebNN browser ML API~~ ✅ Complete (`dpb-wasm/webnn.rs`)
   - ~~WebGL deprecation documentation~~ ✅ Complete (`docs/WEBGL_DEPRECATION.md`)
   - ~~Deployment guide for all platforms~~ ✅ Complete (`docs/DEPLOYMENT_GUIDE.md`)

5. **Next Phase (v5.4.0)**
   - Clinical validation against PhysioNet reference datasets
   - FDA 510(k) pre-submission documentation
   - Production TPC-C kernels for Intel Gaudi
   - Production Poplar vertices for Graphcore IPU
   - Physical RISC-V testing (ESP32-C3, SiFive)
   - CUDA kernel optimization and profiling
   - WebNN integration testing across browsers
   - FPGA synthesis testing (Xilinx Zynq, Intel Arria)

---

## 7. Quick Reference Tables

### 7.1 Signal Processing Quick Reference

| Task | Function/Type | Location |
|------|---------------|----------|
| R-peak detection | `PanTompkinsDetector::detect()` | `dpb-core/signal/ecg.rs` |
| HRV analysis | `HrvAnalyzer::analyze()` | `dpb-core/signal/hrv.rs` |
| Band power | `compute_band_powers()` | `dpb-core/signal/eeg/bands.rs` |
| EMD decomposition | `EmpiricalModeDecomposition::decompose()` | `dpb-core/signal/emd.rs` |
| Wavelet transform | `ContinuousWaveletTransform::transform()` | `dpb-core/signal/wavelet.rs` |
| ICA | `FastICA::fit_transform()` | `dpb-core/signal/ica.rs` |

### 7.2 Data Format Quick Reference

| Format | Read | Write | Key Types |
|--------|------|-------|-----------|
| WFDB | `WfdbReader` | `WfdbWriter` | `WfdbSignal`, `WfdbAnnotation` |
| EDF | `EdfReader` | `EdfWriter` | `EdfSignal`, `EdfHeader` |
| GDF | `GdfReader` | `GdfWriter` | `GdfHeader` |
| BDF | `BdfReader` | `BdfWriter` | 24-bit signals |
| XDF | `XdfFile` | - | `XdfStream`, clock sync |
| BIDS | `BidsDataset` | `BidsWriter` | `BidsSubject`, `BidsSession` |
| FHIR | `FhirClient` | `FhirBundle` | `FhirObservation`, `FhirPatient` |
| Auto | `UnifiedReader` | - | Magic byte detection |

### 7.3 Neural Network Quick Reference

| Model Type | Class | Key Methods |
|------------|-------|-------------|
| LIF | `LeakyIntegrateFire` | `forward()`, `reset()` |
| Adaptive LIF | `AdaptiveLIF` | `forward()`, `get_threshold()` |
| Izhikevich | `IzhikevichNeuron` | `forward()`, `set_mode()` |
| Multi-compartment | `MultiCompartmentNeuron` | `step()`, `inject_current()` |
| ESN | `EchoStateNetwork` | `forward()`, `train_readout()` |
| LSM | `LiquidStateMachine` | `forward()`, `get_state()` |

### 7.4 Training Quick Reference

| Method | Class | When to Use |
|--------|-------|-------------|
| BPTT | `BPTTTrainer` | Standard supervised training |
| OTTT | `OTTTTrainer` | Online/streaming data |
| STDP | `STDP` | Unsupervised, local learning |
| Distillation | `TeacherStudentTrainer` | Model compression |
| Reward STDP | `RewardModulatedSTDP` | Reinforcement learning |

### 7.5 Export Quick Reference

| Target | Exporter | Output |
|--------|----------|--------|
| Cross-platform | `OnnxExporter` | `.onnx` file |
| Mobile | `TfLiteExporter` | `.tflite` file |
| Intel Loihi 2 | `NeuromorphicExporter::Loihi2` | Lava Python code |
| SpiNNaker 2 | `NeuromorphicExporter::SpiNNaker2` | sPyNNaker Python |
| BrainScaleS-2 | `NeuromorphicExporter::BrainScaleS2` | hxtorch Python |
| Portable neuromorphic | `NeuromorphicExporter::PyNN` | PyNN-compatible |
| iOS/Android | `MobileRuntime` | Native runtime |
| **JSON** | `JsonExporter` | `.json` config |
| **Binary** | `BinaryExporter` | `.dpb` embedded |
| **Xilinx FPGA** ✅ NEW | `FpgaExporter::XilinxVitis` | Vitis HLS C++ |
| **Intel FPGA** ✅ NEW | `FpgaExporter::IntelHls` | Intel HLS C++ |
| **Generic FPGA** ✅ NEW | `FpgaExporter::GenericHls` | Plain C (portable) |

### 7.6 Neuromodulation Quick Reference

| System | Class | Effect |
|--------|-------|--------|
| Dopamine | `DopamineSystem` | Reward prediction, motivation |
| Acetylcholine | `AcetylcholineSystem` | Attention, learning rate |
| Norepinephrine | `NorepinephrineSystem` | Arousal, gain modulation |
| Serotonin | `SerotoninSystem` | Mood, temporal discounting |

### 7.7 Cross-Platform Quick Reference ✅ NEW

| Platform | Crate/Binding | Key Types/Classes | Build Command |
|----------|---------------|-------------------|---------------|
| **Python** | `dpb-python` | `TimeSeries`, `SpikeTrain`, encoders | `maturin build` |
| **R** | `bindings/r` | `TimeSeries`, `SpikeTrain` R6 classes | `R CMD INSTALL` |
| **Julia** | `bindings/julia` | `DPB.TimeSeries`, `DPB.encode` | Load via `include()` |
| **MATLAB** | `bindings/matlab` | `dpb_timeseries`, `dpb_encode` | MEX compile |
| **LabVIEW** | `bindings/labview` | Call Library Function Nodes | NI LabVIEW |
| **JavaScript** | `dpb-wasm` | `WasmTimeSeries`, `WasmLevelCrossingEncoder` | `wasm-pack build` |
| **LSL** | `dpb-lsl` | `LslInlet`, `LslOutlet`, `EncodingPipeline` | `cargo build -p dpb-lsl` |

### 7.8 LSL Stream Types Quick Reference ✅ NEW

| Constant | Type | Description |
|----------|------|-------------|
| `stream_types::EEG` | `"EEG"` | Electroencephalography |
| `stream_types::ECG` | `"ECG"` | Electrocardiography |
| `stream_types::EMG` | `"EMG"` | Electromyography |
| `stream_types::PPG` | `"PPG"` | Photoplethysmography |
| `stream_types::EDA` | `"EDA"` | Electrodermal activity |
| `stream_types::RESP` | `"Respiration"` | Respiratory signals |
| `stream_types::MARKERS` | `"Markers"` | Event markers |
| `stream_types::SPIKES` | `"Spikes"` | DPB spike trains |

### 7.9 Hardware Accelerator Quick Reference ✅ NEW v5.3.0

| Backend | Class | Feature Flag | Key Capabilities |
|---------|-------|--------------|------------------|
| **CPU** | `CpuAccelerator` | (always) | Reference implementation, all platforms |
| **CUDA** | `CudaAccelerator` | `cuda` | PTX kernels, NVIDIA GPUs, streams, events |
| **Intel Gaudi** | `IntelGaudiAccelerator` | `intel-gaudi` | TPC kernels, HBM memory, graph compilation |
| **Graphcore IPU** | `GraphcoreIpuAccelerator` | `graphcore-ipu` | Poplar SDK, BSP, tile computing |
| **RISC-V** | `RiscVAccelerator` | `riscv` | Embedded HAL, fixed-point Q16.16, DMA |

### 7.10 Mobile NPU Quick Reference ✅ NEW v5.3.0

| Backend | Class/Config | Platform | Key Features |
|---------|--------------|----------|--------------|
| **Qualcomm Hexagon** | `NpuBackend::QualcommHexagon` | Android (Snapdragon) | DSP/HVX/HTP, power levels |
| **ARM Ethos-U** | `NpuBackend::ArmEthosU` | IoT/Embedded | U55/U65, Cortex-M integration |
| **Apple ANE** | `NpuBackend::AppleAne` | iOS/macOS | Neural Engine, FP16 support |
| **Samsung NPU** | `NpuBackend::SamsungNpu` | Android (Exynos) | NPU co-processor |

### 7.11 Web ML Quick Reference ✅ NEW v5.3.0

| API | Class | Browser Support | Key Features |
|-----|-------|-----------------|--------------|
| **WebGPU** | `GpuEncoder` | Chrome 113+, Edge 113+, Firefox 121+ | WGSL compute shaders, full GPU access |
| **WebNN** | `WebNNEncoder` | Chrome/Edge (flag) | ML inference, CPU/GPU/NPU device selection |
| **WASM CPU** | `WasmLevelCrossingEncoder` | All modern browsers | Fallback, always available |

---

## 8. Learning Resources

### 8.1 Documentation Structure

```
docs/learning/
├── DOCUMENTATION_PLAN.md       → Master plan, writing guidelines
├── books/
│   ├── 01_foundations/         → History of biosignal measurement (7 chapters)
│   ├── 02_signals/             → Signal types explained (8 chapters)
│   ├── 03_analysis/            → Processing techniques (8 chapters)
│   ├── 04_intelligence/        → AI and ML concepts (8 chapters)
│   └── 05_practice/            → Real-world applications (8 chapters)
├── notebooks/
│   ├── beginner/               → First steps (3 notebooks)
│   ├── intermediate/           → Building skills (3 notebooks)
│   └── advanced/               → Expert techniques (2 notebooks)
├── reference/
│   ├── glossary.md             → 105 terms defined
│   └── quick_cards/            → 5 quick reference cards
└── media/
    └── video_scripts/          → 5 episode scripts (~38 min total)
```

### 8.2 Learning Path

| Level | Content | Time |
|-------|---------|------|
| **Beginner** | Book 1-2, Notebooks 01-03 | ~8 hours |
| **Intermediate** | Book 3-4, Notebooks 04-06 | ~12 hours |
| **Advanced** | Book 5, Notebooks 07-08 | ~8 hours |
| **Video Series** | Episodes 1-5 | ~40 minutes |

### 8.3 Quick Cards Available

1. **Signal Types** - ECG, EEG, EMG, PPG, EDA at a glance
2. **Normal vs Abnormal** - Reference ranges and warning signs
3. **Analysis Steps** - Standard processing workflow
4. **Code Cheatsheet** - Common operations in code
5. **Troubleshooting** - Common problems and solutions

---

## Appendix A: Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0.0 | Dec 2025 | Initial catalog |
| v2.0.0 | Dec 2025 | ECG/HRV, transforms, pipeline, calibration, explainability |
| v3.0.0 | Dec 2025 | GPU, distributed, dpb-viz, GDF/BDF/XDF, dpb-mobile |
| v4.0.0 | Dec 2025 | Knowledge distillation, BIDS, FHIR, neuromorphic export, TFLite, EMD/EEMD, dendritic computation, neuromodulation, learning library |
| v5.0.0 | Dec 2025 | **R bindings**, **LabVIEW bindings**, **dpb-wasm** (WebAssembly), **dpb-lsl** (Lab Streaming Layer), **dpb-export** (ONNX/JSON/Binary), cross-platform integration |
| v5.1.0 | Dec 2025 | **dpb-federated** (federated learning: FedAvg, differential privacy, gradient compression), **dpb-clinical** (ethnic stratification, treatment response, comorbidity modeling, practice effects), **Hardware accelerators** (Intel Gaudi, Graphcore IPU stubs), **WebGPU acceleration** (browser GPU compute), **RISC-V targets** (embedded microcontroller support), **liblsl FFI** (complete C bindings), **CI/CD benchmarks** (performance regression detection) |

---

## Appendix B: Build Commands

```bash
# Standard build
cargo build --all-features

# With new features
cargo build -p dpb-core --features bids,fhir,emd
cargo build -p dpb-core --features intel-gaudi,graphcore-ipu  # Hardware accelerators
cargo build -p dpb-neurons --features dendritic
cargo build -p dpb-snn --features distillation,neuromorphic,neuromodulation
cargo build -p dpb-snn --features tflite

# GPU features
cargo build -p dpb-snn --features cuda
cargo build -p dpb-snn --features metal

# Mobile builds
cargo build -p dpb-mobile --target aarch64-apple-ios --features ios
cargo ndk --target aarch64-linux-android -- build -p dpb-mobile --features android

# Cross-platform builds
cargo build -p dpb-wasm --target wasm32-unknown-unknown
cargo build -p dpb-wasm --features webgpu  # WebGPU acceleration
wasm-pack build crates/dpb-wasm --target web
cargo build -p dpb-lsl --features async,native  # With liblsl FFI
cargo build -p dpb-export --features onnx

# Federated learning and clinical builds (NEW)
cargo build -p dpb-federated --features differential-privacy,compression
cargo build -p dpb-clinical

# Embedded/RISC-V builds (NEW)
cargo build-riscv32  # Uses .cargo/config.toml alias
cargo build-riscv64
cargo build --target riscv32imc-unknown-none-elf --profile release-embedded

# R package build
R CMD INSTALL bindings/r

# Run all tests
cargo test --all

# Run CI/CD benchmarks
cargo bench --bench signal_processing

# Generate documentation
cargo doc --no-deps --all-features --open
```

---

## Appendix C: File Counts by Module (v5.1.0)

| Module | Files | Approx LOC |
|--------|-------|------------|
| dpb-core/signal | 25+ | 15,000 |
| dpb-core/io (inc. BIDS, FHIR) | 20+ | 12,000 |
| dpb-core/accelerators ✅ NEW | 1 | 500 |
| dpb-neurons (inc. dendritic) | 25+ | 15,000 |
| dpb-snn (all features) | 60+ | 65,000 |
| dpb-synth | 40+ | 30,000 |
| dpb-norms | 5 | 10,000 |
| dpb-viz | 7 | 4,500 |
| dpb-mobile | 8 | 4,000 |
| docs/learning | 55+ | 25,000 |
| dpb-wasm (inc. WebGPU) | 6 | 2,000 |
| dpb-lsl (inc. FFI) | 8 | 3,500 |
| dpb-export | 7 | 2,000 |
| **dpb-federated** ✅ NEW | 9 | 3,500 |
| **dpb-clinical** ✅ NEW | 7 | 2,500 |
| bindings/r | 11 | 1,500 |
| bindings/labview | 1 | 500 |
| .cargo/config.toml ✅ NEW | 1 | 150 |
| .github/workflows ✅ NEW | 1 | 280 |
| **Total** | **310+** | **~197,000** |

---

*End of System Catalog v5.1.0*
