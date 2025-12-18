# Delta-Predictive Biosensing (DPB) System Catalog v4.0.0

> **Last Updated:** December 2025
> **Framework Version:** 0.4.0
> **Total Modules:** 250+ | **Encoders:** 77+ | **Generators:** 200+ | **Decoders:** 48+

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Capability → Implementation Map](#2-capability--implementation-map)
3. [Implementation → Capabilities Map](#3-implementation--capabilities-map)
4. [Dependency Graph](#4-dependency-graph)
5. [Gap Analysis](#5-gap-analysis)
6. [Quick Reference Tables](#6-quick-reference-tables)
7. [Learning Resources](#7-learning-resources)

---

## 1. Executive Summary

The Delta-Predictive Biosensing (DPB) Framework is a comprehensive neuromorphic biosignal processing library providing end-to-end capabilities from signal acquisition to clinical deployment.

### 1.1 System Metrics

| Metric | Count |
|--------|-------|
| Crates | 14 |
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
| GPU Backends | 2 (CUDA, Metal) |
| Neuromorphic Targets | 3 (Loihi, SpiNNaker, BrainScaleS) |
| Export Formats | 3 (ONNX, TFLite, Mobile) |
| Visualization Types | 6 |
| Normative Databases | Age/Sex stratified (Pediatric/Adult/Geriatric) |
| Learning Resources | 5 Books, 8 Notebooks, 5 Video Scripts |

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
| **dpb-mobile** | iOS/Android runtime, FFI, optimization, benchmarking | 4,000+ |
| **dpb-cognitive** | Cognitive assessment paradigms | 4,000+ |
| **dpb-python** | PyO3 Python bindings | 3,000+ |
| **dpb-ffi** | C-compatible FFI | 1,500+ |
| **dpb-bench** | Benchmarking suite | 2,500+ |

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
```

---

## 4. Dependency Graph

### 4.1 Crate Dependencies

```
                                    ┌─────────────────┐
                                    │   dpb-python    │
                                    │   dpb-ffi       │
                                    └────────┬────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
    ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
    │   dpb-bench     │           │   dpb-snn       │           │   dpb-synth     │
    └────────┬────────┘           │ ✅ +distillation │          └────────┬────────┘
             │                    │ ✅ +neuromorphic │                   │
             │                    │ ✅ +neuromodulat │                   │
             │                    └────────┬────────┘                    │
             │                             │                              │
             │                    ┌────────┴────────┐                     │
             │                    │                 │                     │
             │                    ▼                 ▼                     │
             │          ┌─────────────────┐  ┌─────────────────┐         │
             │          │  dpb-neurons    │  │  dpb-encoders   │         │
             │          │ ✅ +dendritic   │  └────────┬────────┘         │
             │          └────────┬────────┘           │                  │
             │                   │                    │                  │
             │                   └──────────┬─────────┘                  │
             │                              │                            │
             │                              ▼                            │
             │                   ┌─────────────────┐                     │
             │                   │   dpb-norms     │◄────────────────────┤
             │                   └────────┬────────┘                     │
             │                            │                              │
             └────────────────────────────┼──────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
          ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
          │   dpb-viz       │  │   dpb-core      │  │   dpb-mobile    │
          │                 │  │ ✅ +BIDS        │  │                 │
          │                 │  │ ✅ +FHIR        │  │                 │
          │                 │  │ ✅ +EMD         │  │                 │
          └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 4.2 Feature Flags

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
└── emd                → Empirical mode decomposition

dpb-mobile features:
├── ios                → iOS-specific (Metal, CoreML)
├── android            → Android-specific (NNAPI, Vulkan)
└── quantized          → Quantized inference

dpb-snn/export features:
├── onnx               → ONNX export
└── tflite             → TensorFlow Lite export
```

### 4.3 Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPLETE DATA FLOW v4.0                                     │
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

## 5. Gap Analysis

This section answers: **"What's needed but missing?"**

### 5.1 Implementation Status Matrix

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
| **Visualization** | Dashboard/Raster/Heatmap | ✅ Complete | - |
| **Mobile** | iOS/Android Runtime | ✅ Complete | - |
| **Normative** | Pediatric/Adult/Geriatric | ✅ Complete | - |
| **Learning** | Books/Notebooks/Videos | ✅ Complete | - |

### 5.2 Remaining Gaps

#### HIGH Priority

| Gap | Description | Effort | Impact |
|-----|-------------|--------|--------|
| **HIPAA/PHI Tools** | Data anonymization, de-identification utilities | Medium | Critical for clinical use |
| **Clinical Validation** | Tests against MIT-BIH, CHB-MIT, PhysioNet datasets | Medium | Regulatory compliance |
| **Unit Test Coverage** | Comprehensive test coverage for all new modules | Medium | Quality assurance |

#### MEDIUM Priority

| Gap | Description | Effort | Impact |
|-----|-------------|--------|--------|
| **Ethnic Stratification** | Population-specific normative data | Medium | Equity in clinical tools |
| **Treatment Response** | Pre/post intervention modeling | Medium | Clinical utility |
| **Performance Regression** | Automated CI/CD benchmarks | Low | Development velocity |
| **Federated Learning** | Privacy-preserving distributed training | High | Multi-site collaboration |
| **Edge Deployment** | WebAssembly, RISC-V targets | Medium | Browser/embedded use |

#### LOW Priority

| Gap | Description | Effort | Impact |
|-----|-------------|--------|--------|
| **Comorbidity Modeling** | Multi-disease simulation | Medium | Research utility |
| **Practice Effects** | Serial testing corrections | Low | Longitudinal accuracy |
| **Additional Hardware** | Intel Gaudi, Graphcore IPU | High | Hardware diversity |
| **Streaming Inference** | Continuous real-time processing | Medium | Real-time applications |

### 5.3 Module Completeness

| Module | Core | Tests | Docs | Examples |
|--------|:----:|:-----:|:----:|:--------:|
| dpb-core/signal | ✅ | ✅ | ✅ | ✅ |
| dpb-core/io (all formats) | ✅ | ✅ | ✅ | ⚠️ |
| dpb-core/io/bids | ✅ | ✅ | ✅ | ⚠️ |
| dpb-core/io/fhir | ✅ | ✅ | ✅ | ⚠️ |
| dpb-neurons/dendritic | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/distillation | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/neuromorphic | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/neuromodulation | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/export/tflite | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/gpu | ✅ | ✅ | ✅ | ✅ |
| dpb-snn/distributed | ✅ | ✅ | ✅ | ✅ |
| dpb-viz | ✅ | ✅ | ✅ | ⚠️ |
| dpb-mobile | ✅ | ✅ | ✅ | ✅ |

Legend: ✅ Complete | ⚠️ Partial (needs more examples) | ❌ Missing

### 5.4 Recommended Next Steps

1. **Immediate (Consolidation)**
   - Add integration examples for BIDS, FHIR, TFLite, neuromorphic export
   - Create end-to-end tutorial notebooks using new features
   - Run validation against public datasets

2. **Short-term (Clinical Readiness)**
   - Implement HIPAA/PHI de-identification utilities
   - Add clinical validation test suite
   - Create regulatory documentation templates

3. **Medium-term (Advanced Features)**
   - Federated learning for multi-site studies
   - WebAssembly compilation for browser deployment
   - Streaming inference pipeline

---

## 6. Quick Reference Tables

### 6.1 Signal Processing Quick Reference

| Task | Function/Type | Location |
|------|---------------|----------|
| R-peak detection | `PanTompkinsDetector::detect()` | `dpb-core/signal/ecg.rs` |
| HRV analysis | `HrvAnalyzer::analyze()` | `dpb-core/signal/hrv.rs` |
| Band power | `compute_band_powers()` | `dpb-core/signal/eeg/bands.rs` |
| EMD decomposition | `EmpiricalModeDecomposition::decompose()` | `dpb-core/signal/emd.rs` |
| Wavelet transform | `ContinuousWaveletTransform::transform()` | `dpb-core/signal/wavelet.rs` |
| ICA | `FastICA::fit_transform()` | `dpb-core/signal/ica.rs` |

### 6.2 Data Format Quick Reference

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

### 6.3 Neural Network Quick Reference

| Model Type | Class | Key Methods |
|------------|-------|-------------|
| LIF | `LeakyIntegrateFire` | `forward()`, `reset()` |
| Adaptive LIF | `AdaptiveLIF` | `forward()`, `get_threshold()` |
| Izhikevich | `IzhikevichNeuron` | `forward()`, `set_mode()` |
| Multi-compartment | `MultiCompartmentNeuron` | `step()`, `inject_current()` |
| ESN | `EchoStateNetwork` | `forward()`, `train_readout()` |
| LSM | `LiquidStateMachine` | `forward()`, `get_state()` |

### 6.4 Training Quick Reference

| Method | Class | When to Use |
|--------|-------|-------------|
| BPTT | `BPTTTrainer` | Standard supervised training |
| OTTT | `OTTTTrainer` | Online/streaming data |
| STDP | `STDP` | Unsupervised, local learning |
| Distillation | `TeacherStudentTrainer` | Model compression |
| Reward STDP | `RewardModulatedSTDP` | Reinforcement learning |

### 6.5 Export Quick Reference

| Target | Exporter | Output |
|--------|----------|--------|
| Cross-platform | `OnnxExporter` | `.onnx` file |
| Mobile | `TfLiteExporter` | `.tflite` file |
| Intel Loihi | `LoihiExporter` | Loihi configuration |
| SpiNNaker | `SpinnakerExporter` | PyNN-compatible |
| BrainScaleS | `BrainscalesExporter` | BrainScaleS mapping |
| iOS/Android | `MobileRuntime` | Native runtime |

### 6.6 Neuromodulation Quick Reference

| System | Class | Effect |
|--------|-------|--------|
| Dopamine | `DopamineSystem` | Reward prediction, motivation |
| Acetylcholine | `AcetylcholineSystem` | Attention, learning rate |
| Norepinephrine | `NorepinephrineSystem` | Arousal, gain modulation |
| Serotonin | `SerotoninSystem` | Mood, temporal discounting |

---

## 7. Learning Resources

### 7.1 Documentation Structure

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

### 7.2 Learning Path

| Level | Content | Time |
|-------|---------|------|
| **Beginner** | Book 1-2, Notebooks 01-03 | ~8 hours |
| **Intermediate** | Book 3-4, Notebooks 04-06 | ~12 hours |
| **Advanced** | Book 5, Notebooks 07-08 | ~8 hours |
| **Video Series** | Episodes 1-5 | ~40 minutes |

### 7.3 Quick Cards Available

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
| v4.0.0 | Dec 2025 | **Knowledge distillation**, **BIDS**, **FHIR**, **neuromorphic export**, **TFLite**, **EMD/EEMD**, **dendritic computation**, **neuromodulation**, **learning library** |

---

## Appendix B: Build Commands

```bash
# Standard build
cargo build --all-features

# With new features
cargo build -p dpb-core --features bids,fhir,emd
cargo build -p dpb-neurons --features dendritic
cargo build -p dpb-snn --features distillation,neuromorphic,neuromodulation
cargo build -p dpb-snn --features tflite

# GPU features
cargo build -p dpb-snn --features cuda
cargo build -p dpb-snn --features metal

# Mobile builds
cargo build -p dpb-mobile --target aarch64-apple-ios --features ios
cargo ndk --target aarch64-linux-android -- build -p dpb-mobile --features android

# Run all tests
cargo test --all

# Generate documentation
cargo doc --no-deps --all-features --open
```

---

## Appendix C: File Counts by Module (v4.0.0)

| Module | Files | Approx LOC |
|--------|-------|------------|
| dpb-core/signal | 25+ | 15,000 |
| dpb-core/io (inc. BIDS, FHIR) | 20+ | 12,000 |
| dpb-neurons (inc. dendritic) | 25+ | 15,000 |
| dpb-snn (all features) | 60+ | 65,000 |
| dpb-synth | 40+ | 30,000 |
| dpb-norms | 5 | 10,000 |
| dpb-viz | 7 | 4,500 |
| dpb-mobile | 8 | 4,000 |
| docs/learning | 55+ | 25,000 |
| **Total** | **250+** | **~180,000** |

---

*End of System Catalog v4.0.0*
