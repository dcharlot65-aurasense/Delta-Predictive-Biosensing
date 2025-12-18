# Delta-Predictive Biosensing (DPB) System Catalog v3.0.0

> **Last Updated:** December 2024
> **Framework Version:** 0.3.0
> **Total Modules:** 200+ | **Encoders:** 77+ | **Generators:** 200+ | **Decoders:** 48+

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Capability → Implementation Map](#2-capability--implementation-map)
3. [Implementation → Capabilities Map](#3-implementation--capabilities-map)
4. [Dependency Graph](#4-dependency-graph)
5. [Gap Analysis & Recommendations](#5-gap-analysis--recommendations)
6. [Quick Reference Tables](#6-quick-reference-tables)

---

## 1. Executive Summary

The Delta-Predictive Biosensing (DPB) Framework is a comprehensive neuromorphic biosignal processing library providing:

| Metric | Count |
|--------|-------|
| Crates | 12 |
| Neuron Models | 19 + Reservoir (ESN/LSM) |
| Event Encoders | 77+ |
| Population Templates | 61+ |
| Synthetic Generators | 200+ |
| Output Decoders | 48+ |
| Signal Augmentations | 13 |
| Calibration Methods | 4 |
| Explainability Tools | 8 |
| Clinical Metrics | 100+ |
| Data Formats | 6 (WFDB, EDF, GDF, BDF, XDF + auto-detect) |
| GPU Backends | 2 (CUDA, Metal) |
| Visualization Types | 6 (Dashboard, Raster, Network, Heatmap, Timeline, Export) |
| Normative Databases | Age/Sex stratified (incl. Pediatric/Geriatric) |
| Integration Tests | 50+ |

### Crate Overview

| Crate | Purpose | LOC (approx) |
|-------|---------|--------------|
| **dpb-core** | Types, traits, signal processing, pipelines, I/O (6 formats) | 35,000+ |
| **dpb-encoders** | Event-based encoders, population templates | 12,000+ |
| **dpb-neurons** | 19 neuron models, surrogate gradients, reservoir computing | 10,000+ |
| **dpb-snn** | SNN architectures, training, calibration, explainability, GPU, distributed | 45,000+ |
| **dpb-synth** | 200+ synthetic generators, augmentation, cohorts, pathology | 30,000+ |
| **dpb-norms** | Normative databases (adult, pediatric, geriatric), longitudinal | 10,000+ |
| **dpb-viz** ✅ NEW | Dashboards, raster plots, heatmaps, network graphs, timeline | 4,500+ |
| **dpb-mobile** ✅ NEW | iOS/Android runtime, FFI, optimization, benchmarking | 4,000+ |
| **dpb-cognitive** | Cognitive assessment paradigms | 4,000+ |
| **dpb-python** | PyO3 Python bindings | 3,000+ |
| **dpb-ffi** | C-compatible FFI | 1,500+ |
| **dpb-bench** | Benchmarking suite | 2,500+ |

---

## 2. Capability → Implementation Map

### 2.1 Signal Acquisition & Preprocessing

| Capability | Implementation | Location |
|------------|----------------|----------|
| **FFT/Spectral Analysis** | `FftProcessor`, `Stft` | `dpb-core/signal/fft.rs` |
| **Filtering (FIR/IIR)** | `FirFilter`, `IirFilter`, `FilterType` | `dpb-core/signal/filter.rs` |
| **Resampling** | `downsample`, `upsample`, `resample_linear` | `dpb-core/signal/resample.rs` |
| **Normalization** | `normalize` (ZScore, MinMax, Robust) | `dpb-core/signal/mod.rs` |
| **DC Offset Removal** | `remove_dc_offset` | `dpb-core/signal/mod.rs` |
| **Peak/Valley Detection** | `find_peaks`, `find_valleys` | `dpb-core/signal/mod.rs` |
| **Envelope Extraction** | `envelope` | `dpb-core/signal/mod.rs` |
| **RMS/Energy Calculation** | `rms`, `energy`, `snr_db` | `dpb-core/signal/mod.rs` |

### 2.2 Advanced Signal Transforms

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Wavelet Transform (CWT)** | `ContinuousWaveletTransform`, `WaveletFamily` | `dpb-core/signal/wavelet.rs` |
| **Wavelet Transform (DWT)** | `DiscreteWaveletTransform`, `DwtResult` | `dpb-core/signal/wavelet.rs` |
| **Hilbert Transform** | `hilbert_transform`, `AnalyticSignal` | `dpb-core/signal/hilbert.rs` |
| **Instantaneous Phase/Freq** | `analytic_signal`, `instantaneous_frequency` | `dpb-core/signal/hilbert.rs` |
| **Independent Component Analysis** | `FastICA`, `ICAResult` | `dpb-core/signal/ica.rs` |

### 2.3 Domain-Specific Signal Analysis

| Domain | Capability | Implementation | Location |
|--------|------------|----------------|----------|
| **EEG** | Band Power Analysis | `compute_band_powers`, `EegBands` | `dpb-core/signal/eeg/bands.rs` |
| **EEG** | Artifact Detection | `detect_artifacts`, `ArtifactType` | `dpb-core/signal/eeg/artifacts.rs` |
| **EEG** | ERP Analysis | `ErpAnalyzer`, `ErpComponent` | `dpb-core/signal/eeg/erp.rs` |
| **EEG** | Seizure Detection | `SeizureDetector`, `SeizureEvent` | `dpb-core/signal/eeg/seizure.rs` |
| **ECG** | R-Peak Detection | `PanTompkinsDetector`, `RPeak` | `dpb-core/signal/ecg.rs` |
| **ECG** | QRS Morphology | `QrsMorphology`, `QrsTemplate`, `BeatType` | `dpb-core/signal/ecg.rs` |
| **ECG** | Arrhythmia Detection | `ArrhythmiaDetector`, `ArrhythmiaAnalysis` | `dpb-core/signal/ecg.rs` |
| **HRV** | Time-Domain Metrics | `HrvTimeDomain` (SDNN, RMSSD, pNN50) | `dpb-core/signal/hrv.rs` |
| **HRV** | Frequency-Domain | `HrvFrequencyDomain` (VLF, LF, HF) | `dpb-core/signal/hrv.rs` |
| **PPG** | Pulse Analysis | `PpgAnalyzer`, `PulseWaveFeatures` | `dpb-core/signal/ppg.rs` |
| **PPG** | SpO2 Estimation | `SpO2Result` | `dpb-core/signal/ppg.rs` |
| **EDA** | Tonic/Phasic Decomposition | `EdaDecomposition` | `dpb-core/signal/eda.rs` |
| **EDA** | SCR Detection | `ScrEvent`, `EdaMetrics` | `dpb-core/signal/eda.rs` |
| **EMG** | Burst Detection | `EmgBurst`, `EmgAnalyzer` | `dpb-core/signal/emg.rs` |
| **EMG** | Fatigue Metrics | `FatigueMetrics` | `dpb-core/signal/emg.rs` |
| **Voice** | F0/Jitter/Shimmer | `F0Metrics`, `JitterMetrics`, `ShimmerMetrics` | `dpb-core/signal/voice.rs` |
| **Eye** | Saccade/Fixation | `Saccade`, `Fixation`, `Blink` | `dpb-core/signal/eye.rs` |
| **Respiratory** | Breath Detection | `BreathEvent`, `RespiratoryAnalyzer` | `dpb-core/signal/respiratory.rs` |
| **Respiratory** | Apnea Detection | `ApneaEvent`, `ApneaType`, `SleepApneaSeverity` | `dpb-core/signal/respiratory.rs` |

### 2.4 Fatigue Detection

| Capability | Implementation | Location |
|------------|----------------|----------|
| **EMG Fatigue** | `EmgFatigueAnalyzer` (MDF slope, spectral shift) | `dpb-core/signal/fatigue.rs` |
| **Force Fatigue** | `ForceFatigueAnalyzer` (decline rate, endurance) | `dpb-core/signal/fatigue.rs` |
| **Cognitive Fatigue** | `CognitiveFatigueAnalyzer` (RT deterioration, lapses) | `dpb-core/signal/fatigue.rs` |
| **Integrated Fatigue** | `IntegratedFatigueMetrics`, `integrate_fatigue` | `dpb-core/signal/fatigue.rs` |

### 2.5 Real-Time Pipeline

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Ring Buffer** | `RingBuffer<T>` | `dpb-core/pipeline/buffer.rs` |
| **Sliding Window** | `SlidingWindow<T>` | `dpb-core/pipeline/buffer.rs` |
| **Overlap Buffer** | `OverlapBuffer` | `dpb-core/pipeline/buffer.rs` |
| **Pipeline Stages** | `PipelineStage`, `TimedStage`, `StageMetrics` | `dpb-core/pipeline/stage.rs` |
| **Pipeline Executor** | `PipelineExecutor`, `ExecutionMode` | `dpb-core/pipeline/executor.rs` |
| **Latency Tracking** | `LatencyStats` (P95, P99, miss rate) | `dpb-core/pipeline/executor.rs` |
| **Pipeline Builder** | `PipelineBuilder`, `Pipeline` | `dpb-core/pipeline/executor.rs` |

### 2.6 Data Format Support

| Capability | Implementation | Location |
|------------|----------------|----------|
| **WFDB/PhysioNet Read** | `WfdbReader`, `WfdbHeader`, `WfdbSignal` | `dpb-core/io/wfdb.rs` |
| **WFDB/PhysioNet Write** | `WfdbWriter` | `dpb-core/io/wfdb.rs` |
| **WFDB Annotations** | `WfdbAnnotation`, `AnnotationType` | `dpb-core/io/wfdb.rs` |
| **EDF/EDF+ Read** | `EdfReader`, `EdfHeader`, `EdfSignal` | `dpb-core/io/edf.rs` |
| **EDF/EDF+ Write** | `EdfWriter` | `dpb-core/io/edf.rs` |
| **GDF Read/Write** ✅ NEW | `GdfReader`, `GdfWriter`, `GdfHeader` | `dpb-core/io/gdf.rs` |
| **BDF Read/Write** ✅ NEW | `BdfReader`, `BdfWriter`, `BdfHeader` | `dpb-core/io/bdf.rs` |
| **XDF Read** ✅ NEW | `XdfFile`, `XdfStream`, clock sync | `dpb-core/io/xdf.rs` |
| **Format Auto-Detect** ✅ NEW | `detect_format`, `UnifiedReader` | `dpb-core/io/format_detect.rs` |

### 2.7 Event-Based Encoding

| Signal Type | Encoders | Location |
|-------------|----------|----------|
| **ECG** | R-Peak, Morphology, ST-Deviation, HRV, Arrhythmia | `dpb-encoders/contact/ecg.rs` |
| **PPG** | Pulse, Amplitude, PTT, SpO2 | `dpb-encoders/contact/ppg.rs` |
| **EDA** | Level-Crossing, SCR, Tonic, Phasic | `dpb-encoders/contact/eda.rs` |
| **EMG** | Burst, Amplitude, Fatigue, MU | `dpb-encoders/contact/emg.rs` |
| **Tremor** | Level-Crossing, Frequency, Amplitude, Type | `dpb-encoders/contact/tremor.rs` |
| **Gait** | Heel-Strike, Toe-Off, Phase, Stride, Asymmetry | `dpb-encoders/pose/gait.rs` |
| **Tapping** | Onset, Aperture, Frequency, Decrement | `dpb-encoders/hand/tapping.rs` |
| **Saccade** | Onset, Main-Sequence, Latency, Accuracy | `dpb-encoders/eye/saccade.rs` |
| **Fixation** | Stability, Drift, Hold | `dpb-encoders/eye/fixation.rs` |
| **Pupil** | Dilation, Light-Reflex, Constriction | `dpb-encoders/eye/pupil.rs` |
| **Voice** | F0, Jitter, Shimmer, HNR, Formant | `dpb-encoders/voice/*.rs` |
| **EEG** | Alpha/Beta/Theta/Gamma Band, Seizure | `dpb-encoders/eeg.rs` |
| **Balance** | COP Velocity, COP Area, Sway | `dpb-encoders/balance.rs` |
| **Force** | GRF Peak, RFD, Grip Strength | `dpb-encoders/force.rs` |

### 2.8 Spiking Neural Networks

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Neuron Models (19)** | LIF, ALIF, ELIF, QLIF, GLIF, Izhikevich, AdEx, HH, etc. | `dpb-neurons/models/*.rs` |
| **Reservoir Computing** | `EchoStateNetwork`, `LiquidStateMachine` | `dpb-neurons/reservoir.rs` |
| **Surrogate Gradients (6)** | FastSigmoid, Arctan, Triangular, SuperSpike, etc. | `dpb-neurons/surrogates.rs` |
| **Spiking Layers** | SpikingLinear, SpikingConv1d/2d, SpikingRNN/LSTM | `dpb-snn/layers/*.rs` |
| **Architectures** | Feedforward, Convolutional, Recurrent, Transformer | `dpb-snn/architectures/*.rs` |
| **Training** | BPTT, OTTT, SLTT | `dpb-snn/training/*.rs` |
| **Hebbian Learning** | `STDP`, `BCMRule`, `OjasRule` | `dpb-snn/learning/hebbian.rs` |
| **Network Pruning** | `NetworkPruner`, `PruningStrategy`, `PruningSchedule` | `dpb-snn/optimization/pruning.rs` |
| **ANN-to-SNN Conversion** | `ANNToSNNConverter`, `WeightNormalization` | `dpb-snn/conversion/*.rs` |
| **Multi-Modal Fusion** | Early, Late, Cross-Modal, Hierarchical, Gated | `dpb-snn/fusion/*.rs` |

### 2.9 GPU Acceleration ✅ NEW

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Backend Abstraction** | `Backend` (Cuda, Metal, Cpu), `GpuDevice` trait | `dpb-snn/gpu/mod.rs` |
| **Auto-Detection** | `auto_detect_backend`, `list_devices` | `dpb-snn/gpu/mod.rs` |
| **CUDA Backend** | `CudaDevice`, `CudaBuffer`, `CudaStream` | `dpb-snn/gpu/cuda.rs` |
| **CUDA Memory Pool** | `CudaMemoryPool`, efficient allocation | `dpb-snn/gpu/cuda.rs` |
| **Metal Backend** | `MetalDevice`, `MetalBuffer`, `MetalComputeEncoder` | `dpb-snn/gpu/metal.rs` |
| **Metal Pipelines** | `MetalPipelineCache`, compute shaders | `dpb-snn/gpu/metal.rs` |
| **Kernel Abstractions** | `SpikeKernel`, `WeightUpdateKernel`, `ReductionKernel` | `dpb-snn/gpu/kernels.rs` |
| **Sparse Connectivity** | `SparseConnectivity` (CSR format) | `dpb-snn/gpu/kernels.rs` |
| **Memory Management** | `MemoryPool`, `PinnedMemory<T>`, `AsyncTransferManager` | `dpb-snn/gpu/memory.rs` |
| **Performance Tracking** | `KernelMetrics`, FLOPS, bandwidth | `dpb-snn/gpu/kernels.rs` |

### 2.10 Distributed Training ✅ NEW

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Configuration** | `DistributedConfig`, `DistributedBackend` (Mpi, Gloo, Nccl) | `dpb-snn/distributed/mod.rs` |
| **Runtime** | `DistributedRuntime`, state management | `dpb-snn/distributed/mod.rs` |
| **Training Coordinator** | `TrainingCoordinator`, barrier sync | `dpb-snn/distributed/coordinator.rs` |
| **Gradient Aggregation** | AllReduce, AsyncSGD, GossipSGD, Hierarchical, LocalSGD | `dpb-snn/distributed/coordinator.rs` |
| **Data Parallel** | `DataParallel`, batch splitting, LR scaling | `dpb-snn/distributed/partitioning.rs` |
| **Model Parallel** | `ModelParallel`, layer distribution | `dpb-snn/distributed/partitioning.rs` |
| **Pipeline Parallel** | `PipelineParallel`, 1F1B scheduling | `dpb-snn/distributed/partitioning.rs` |
| **Communication** | `Message`, Send/Receive, Ring-AllReduce | `dpb-snn/distributed/communication.rs` |
| **Fault Tolerance** | `HeartbeatMonitor`, `CheckpointManager`, `ElasticTrainingManager` | `dpb-snn/distributed/fault_tolerance.rs` |
| **Metrics** | Throughput, scaling efficiency, straggler detection | `dpb-snn/distributed/metrics.rs` |

### 2.11 Model Calibration

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Temperature Scaling** | `TemperatureScaling` | `dpb-snn/calibration/temperature.rs` |
| **Platt Scaling** | `PlattScaling` | `dpb-snn/calibration/temperature.rs` |
| **Isotonic Calibration** | `IsotonicCalibration`, PAVA algorithm | `dpb-snn/calibration/isotonic.rs` |
| **MC Dropout Uncertainty** | `MCDropout`, epistemic/aleatoric | `dpb-snn/calibration/uncertainty.rs` |
| **Ensemble Uncertainty** | `EnsembleUncertainty`, entropy, mutual info | `dpb-snn/calibration/uncertainty.rs` |
| **Bootstrap CI** | `bootstrap_ci`, `ConfidenceInterval` | `dpb-snn/calibration/uncertainty.rs` |
| **ECE Metric** | `expected_calibration_error` | `dpb-snn/calibration/metrics.rs` |
| **Reliability Diagrams** | `reliability_diagram`, `ReliabilityBin` | `dpb-snn/calibration/metrics.rs` |
| **Brier Score** | `brier_score` | `dpb-snn/calibration/metrics.rs` |

### 2.12 Explainability Tools

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Spike Importance** | `SpikeImportance`, `compute_spike_importance` | `dpb-snn/explain/importance.rs` |
| **Neuron Importance** | `NeuronImportance`, `aggregate_to_neurons` | `dpb-snn/explain/importance.rs` |
| **Perturbation Analysis** | `compute_importance_by_perturbation` | `dpb-snn/explain/importance.rs` |
| **Temporal Attention** | `TemporalAttention`, peak detection | `dpb-snn/explain/attention.rs` |
| **Spatial Attention** | `SpatialAttention`, top-k channels | `dpb-snn/explain/attention.rs` |
| **Cross Attention** | `AttentionMap`, spatiotemporal | `dpb-snn/explain/attention.rs` |
| **Gradient Attribution** | `GradientAttribution`, SmoothGrad | `dpb-snn/explain/attribution.rs` |
| **Integrated Gradients** | `IntegratedGradients`, completeness axiom | `dpb-snn/explain/attribution.rs` |
| **Spike SHAP** | `SpikeSHAP`, Shapley values | `dpb-snn/explain/attribution.rs` |
| **Visualization Export** | `ExplanationVisualizer`, JSON export | `dpb-snn/explain/visualization.rs` |

### 2.13 Model Export

| Capability | Implementation | Location |
|------------|----------------|----------|
| **ONNX Export** | `OnnxExporter`, `OnnxConfig`, `OnnxGraph` | `dpb-snn/export/onnx.rs` |
| **Weight Serialization** | `WeightExporter`, Binary/JSON formats | `dpb-snn/export/weights.rs` |
| **Weight Quantization** | `WeightExporter::quantize` (1-32 bit) | `dpb-snn/export/weights.rs` |
| **Weight Pruning** | `WeightExporter::prune` | `dpb-snn/export/weights.rs` |
| **Model Config** | `ModelConfig`, `LayerConfig`, validation | `dpb-snn/export/config.rs` |
| **Export Metadata** | `ExportMetadata` | `dpb-snn/export/config.rs` |

### 2.14 Visualization ✅ NEW (dpb-viz crate)

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Dashboard Server** | `DashboardServer`, `DashboardConfig`, WebSocket | `dpb-viz/dashboard.rs` |
| **Metric Panels** | `SpikeRatePanel`, `LossPanel`, `AccuracyPanel`, `ResourcePanel` | `dpb-viz/dashboard.rs` |
| **Spike Raster Plots** | `RasterPlot`, `RasterPlot3D`, color schemes | `dpb-viz/raster.rs` |
| **Network Topology** | `NetworkGraph`, force-directed, hierarchical layouts | `dpb-viz/network.rs` |
| **Weight Heatmaps** | `WeightHeatmap`, `ActivationHeatmap`, `CorrelationMatrix` | `dpb-viz/heatmap.rs` |
| **Color Scales** | Viridis, Plasma, Inferno, Coolwarm, Grayscale, RedBlue | `dpb-viz/heatmap.rs` |
| **Event Timeline** | `EventTimeline`, `TimelineEvent`, zoom levels | `dpb-viz/timeline.rs` |
| **Export Formats** | `SvgExporter`, `PngExporter`, `JsonExporter`, `CsvExporter` | `dpb-viz/export.rs` |
| **Batch Export** | `BatchExporter`, timestamp generation | `dpb-viz/export.rs` |

### 2.15 Mobile Runtime ✅ NEW (dpb-mobile crate)

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Mobile Runtime** | `MobileRuntime`, batch=1 optimized, pre-allocated buffers | `dpb-mobile/runtime.rs` |
| **Mobile Model Format** | `MobileModel`, quantization (Float16, Int8, Int4) | `dpb-mobile/model.rs` |
| **C FFI** | `dpb_runtime_create`, `dpb_model_load`, `dpb_infer` | `dpb-mobile/ffi.rs` |
| **iOS Integration** | `MetalConfig`, `CoreMLConfig`, `IosRuntime` | `dpb-mobile/ios.rs` |
| **Android Integration** | `NnapiConfig`, `VulkanConfig`, `AndroidRuntime` | `dpb-mobile/android.rs` |
| **Weight Pruning** | `WeightPruner`, magnitude/structured | `dpb-mobile/optimization.rs` |
| **Operator Fusion** | `OperatorFusion`, layer fusion patterns | `dpb-mobile/optimization.rs` |
| **Quantization** | `QuantizationOptimizer`, Int8/Float16 | `dpb-mobile/optimization.rs` |
| **Memory Planning** | `MemoryPlanner`, buffer optimization | `dpb-mobile/optimization.rs` |
| **SIMD Hints** | `SimdOptimizer`, ARM NEON detection | `dpb-mobile/optimization.rs` |
| **Benchmarking** | `BenchmarkRunner`, `LatencyMetrics`, `PowerMetrics` | `dpb-mobile/benchmark.rs` |

### 2.16 Output Decoding (48+ Decoders)

| Category | Decoders | Location |
|----------|----------|----------|
| **Rate-Based** | SpikeRate, FirstSpike, Population, Windowed, Exponential | `dpb-snn/decoders/rate.rs` |
| **Temporal** | TemporalPattern, Latency, ISI, Burst, Phase, RankOrder | `dpb-snn/decoders/temporal.rs` |
| **Clinical Scores** | UPDRS (motor/tremor/brady/rigid/gait), TUG, Berg, MoCA | `dpb-snn/decoders/clinical.rs` |
| **Regression** | HR, HRV, Tremor Freq/Amp, Gait Velocity, RT | `dpb-snn/decoders/regression.rs` |
| **Classification** | Binary, MultiClass, TremorType, SleepStage, Emotion | `dpb-snn/decoders/classification.rs` |

### 2.17 Synthetic Data Generation (200+ Generators)

| Category | Generators | Location |
|----------|------------|----------|
| **ECG** | Normal sinus, arrhythmias, pathologies | `dpb-synth/ecg/*.rs` |
| **EEG** | Normal, sleep stages, seizures, ERPs | `dpb-synth/eeg/*.rs` |
| **EMG** | Normal, fatigue, pathology | `dpb-synth/emg/*.rs` |
| **Tremor** | Rest, postural, kinetic, PD, ET | `dpb-synth/tremor/*.rs` |
| **Gait** | Normal, PD, stroke, aging | `dpb-synth/gait/*.rs` |
| **Voice** | Normal, PD, dysarthria | `dpb-synth/voice/*.rs` |
| **Respiratory** | Normal, apnea patterns | `dpb-synth/respiratory/*.rs` |

### 2.18 Signal Augmentation

| Category | Augmentations | Location |
|----------|---------------|----------|
| **Noise** | GaussianNoise, PinkNoise, BaselineWander, PowerlineNoise, MotionArtifact | `dpb-synth/augmentation/noise.rs` |
| **Temporal** | TimeWarp, TimeShift, WindowCrop, Resample, RandomDropout | `dpb-synth/augmentation/temporal.rs` |
| **Spectral** | MagnitudeScale, FrequencyMask, TimeMask | `dpb-synth/augmentation/spectral.rs` |
| **Pipeline** | `AugmentationPipeline` with probabilities | `dpb-synth/augmentation/mod.rs` |

### 2.19 Pathology Models

| Disease | Features | Location |
|---------|----------|----------|
| **ALS** | ALSFRS-R, EMG signatures, respiratory status | `dpb-synth/pathology/als.rs` |
| **MS** | EDSS, relapse patterns, symptom profiles | `dpb-synth/pathology/ms.rs` |
| **Stroke** | NIHSS, Brunnstrom stages, recovery phases | `dpb-synth/pathology/stroke.rs` |
| **Progression** | Linear, exponential, relapsing trajectories | `dpb-synth/pathology/progression.rs` |
| **Medications** | 12 classes, PK modeling, interactions | `dpb-synth/pathology/medication.rs` |

### 2.20 Virtual Cohorts

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Demographics** | `Demographics`, age/sex/BMI/ethnicity | `dpb-synth/cohort.rs` |
| **Virtual Patients** | `VirtualPatient`, conditions, medications | `dpb-synth/cohort.rs` |
| **Cohort Generation** | `CohortGenerator`, disease prevalence | `dpb-synth/cohort.rs` |

### 2.21 Normative Databases

| Population | Features | Location |
|------------|----------|----------|
| **Adult** | Age/sex stratified, 100+ metrics | `dpb-norms/adult.rs` |
| **Pediatric** | Ages 0-17, developmental stages | `dpb-norms/pediatric.rs` |
| **Geriatric** | 65+, frailty adjustments | `dpb-norms/geriatric.rs` |
| **Longitudinal** | MDC, RCI, change detection | `dpb-norms/longitudinal.rs` |

---

## 3. Implementation → Capabilities Map

### 3.1 dpb-core

```
dpb-core/
├── signal/
│   ├── mod.rs          → FFT, filtering, normalization, envelope, peaks
│   ├── fft.rs          → Spectral analysis, STFT
│   ├── filter.rs       → FIR, IIR, bandpass, notch
│   ├── resampling.rs   → Up/downsampling, interpolation
│   ├── ecg.rs          → Pan-Tompkins, R-peaks, QRS, arrhythmia
│   ├── hrv.rs          → SDNN, RMSSD, pNN50, VLF/LF/HF
│   ├── wavelet.rs      → CWT, DWT, Morlet, Daubechies
│   ├── hilbert.rs      → Analytic signal, inst. phase/freq
│   ├── ica.rs          → FastICA, blind source separation
│   ├── eeg/            → Band power, artifacts, seizure, ERP
│   ├── ppg.rs          → Pulse detection, SpO2, PRV
│   ├── eda.rs          → Tonic/phasic, SCR detection
│   ├── emg.rs          → Burst detection, fatigue
│   ├── voice.rs        → F0, jitter, shimmer, HNR
│   ├── eye.rs          → Saccades, fixations, blinks
│   ├── respiratory.rs  → Breath detection, apnea, AHI
│   └── fatigue.rs      → EMG/force/cognitive fatigue
├── pipeline/
│   ├── mod.rs          → Module exports
│   ├── buffer.rs       → RingBuffer, SlidingWindow, OverlapBuffer
│   ├── stage.rs        → PipelineStage trait, TimedStage
│   └── executor.rs     → PipelineExecutor, latency tracking
├── io/
│   ├── mod.rs          → Module exports, unified reader
│   ├── wfdb.rs         → PhysioNet format read/write
│   ├── edf.rs          → EDF/EDF+ format read/write
│   ├── gdf.rs ✅       → General Data Format read/write
│   ├── bdf.rs ✅       → BioSemi Data Format (24-bit)
│   ├── xdf.rs ✅       → Extensible Data Format (LSL)
│   └── format_detect.rs ✅ → Auto-detection, UnifiedReader
├── types/              → Core type definitions
├── gpu/                → GPU acceleration
└── metrics/            → Performance metrics
```

### 3.2 dpb-neurons

```
dpb-neurons/
├── models/
│   ├── lif.rs          → Leaky Integrate-and-Fire
│   ├── alif.rs         → Adaptive LIF
│   ├── izhikevich.rs   → Izhikevich model
│   ├── adex.rs         → Adaptive Exponential
│   └── ...             → 15+ more models
├── reservoir.rs        → Echo State Network, Liquid State Machine
└── surrogates.rs       → 6 surrogate gradient functions
```

### 3.3 dpb-snn

```
dpb-snn/
├── layers/             → SpikingLinear, SpikingConv, SpikingRNN
├── architectures/      → Feedforward, Conv, Recurrent, Transformer
├── training/           → BPTT, OTTT, SLTT
├── decoders/           → Rate, temporal, clinical decoders
├── fusion/             → Multi-modal fusion (8 types)
├── conversion/         → ANN-to-SNN conversion
├── learning/
│   ├── mod.rs          → Module exports
│   └── hebbian.rs      → STDP, BCM, Oja's rule
├── optimization/
│   ├── mod.rs          → Module exports
│   └── pruning.rs      → Network pruning strategies
├── calibration/
│   ├── mod.rs          → Module exports
│   ├── temperature.rs  → Temperature/Platt scaling
│   ├── isotonic.rs     → Isotonic calibration (PAVA)
│   ├── uncertainty.rs  → MC Dropout, ensemble, bootstrap
│   └── metrics.rs      → ECE, MCE, Brier, reliability
├── explain/
│   ├── mod.rs          → Module exports
│   ├── importance.rs   → Spike/neuron importance
│   ├── attention.rs    → Temporal/spatial attention
│   ├── attribution.rs  → Gradients, IG, SHAP
│   └── visualization.rs → Heatmaps, JSON export
├── export/
│   ├── mod.rs          → Module exports
│   ├── onnx.rs         → ONNX graph export
│   ├── weights.rs      → Weight serialization
│   └── config.rs       → Model configuration
├── gpu/ ✅
│   ├── mod.rs          → Backend enum, GpuDevice trait
│   ├── cuda.rs         → CUDA device, buffers, streams
│   ├── metal.rs        → Metal device, pipelines
│   ├── kernels.rs      → Kernel traits, sparse connectivity
│   └── memory.rs       → Memory pools, pinned memory
└── distributed/ ✅
    ├── mod.rs          → Config, runtime, backends
    ├── coordinator.rs  → Aggregation strategies
    ├── partitioning.rs → Data/Model/Pipeline parallel
    ├── communication.rs → Messages, collectives
    ├── fault_tolerance.rs → Heartbeat, checkpoints
    └── metrics.rs      → Throughput, efficiency
```

### 3.4 dpb-synth

```
dpb-synth/
├── ecg/                → ECG generators
├── eeg/                → EEG generators
├── emg/                → EMG generators
├── tremor/             → Tremor generators
├── gait/               → Gait generators
├── voice/              → Voice generators
├── respiratory/        → Respiratory generators
├── pathology/
│   ├── mod.rs          → Disease stages, body regions
│   ├── als.rs          → ALS model, ALSFRS-R
│   ├── ms.rs           → MS model, EDSS
│   ├── stroke.rs       → Stroke model, NIHSS
│   ├── progression.rs  → Disease trajectories
│   └── medication.rs   → 12 medication classes
├── augmentation/
│   ├── mod.rs          → SignalAugmentation trait, pipeline
│   ├── noise.rs        → 5 noise types
│   ├── temporal.rs     → 5 temporal augmentations
│   ├── spectral.rs     → 3 spectral augmentations
│   └── rand_helpers.rs → RNG utilities
└── cohort.rs           → Virtual patient cohorts
```

### 3.5 dpb-norms

```
dpb-norms/
├── adult.rs            → Adult normative data
├── pediatric.rs        → Pediatric norms (0-17)
├── geriatric.rs        → Geriatric norms (65+, frailty)
└── longitudinal.rs     → MDC, RCI, change detection
```

### 3.6 dpb-viz ✅ NEW

```
dpb-viz/
├── Cargo.toml          → Crate configuration
├── README.md           → Documentation
└── src/
    ├── lib.rs          → Main exports, error types
    ├── dashboard.rs    → DashboardServer, MetricPanels
    ├── raster.rs       → RasterPlot, RasterPlot3D
    ├── network.rs      → NetworkGraph, layouts
    ├── heatmap.rs      → WeightHeatmap, ColorScales
    ├── timeline.rs     → EventTimeline, zoom levels
    └── export.rs       → SVG, PNG, JSON, CSV exporters
```

### 3.7 dpb-mobile ✅ NEW

```
dpb-mobile/
├── Cargo.toml          → crate-type: cdylib, staticlib
├── README.md           → Documentation
├── BUILD_GUIDE.md      → iOS/Android build instructions
└── src/
    ├── lib.rs          → Main exports
    ├── runtime.rs      → MobileRuntime (batch=1)
    ├── model.rs        → MobileModel, quantization
    ├── ffi.rs          → C FFI functions
    ├── ios.rs          → Metal, CoreML integration
    ├── android.rs      → NNAPI, Vulkan integration
    ├── optimization.rs → Pruning, fusion, quantization
    └── benchmark.rs    → Latency, memory, power metrics
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
    └────────┬────────┘           │  (gpu, dist)    │           └────────┬────────┘
             │                    └────────┬────────┘                    │
             │                             │                              │
             │                    ┌────────┴────────┐                     │
             │                    │                 │                     │
             │                    ▼                 ▼                     │
             │          ┌─────────────────┐  ┌─────────────────┐         │
             │          │  dpb-neurons    │  │  dpb-encoders   │         │
             │          └────────┬────────┘  └────────┬────────┘         │
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
          │  (visualization)│  │   (6 formats)   │  │  (iOS/Android)  │
          └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 4.2 Feature Flags

```
dpb-snn features:
├── gpu           → GPU acceleration (base)
│   ├── cuda      → NVIDIA CUDA support
│   └── metal     → Apple Metal support
└── distributed   → Multi-node training

dpb-mobile features:
├── ios           → iOS-specific code
│   ├── metal     → Metal compute
│   └── coreml    → CoreML interop
├── android       → Android-specific code
│   ├── nnapi     → Android NNAPI
│   └── vulkan    → Vulkan compute
└── quantized     → Quantized inference
```

### 4.3 Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE DATA FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │   I/O    │──►│  Signal  │──►│ Encoders │──►│   SNN    │──►│ Decoders │      │
│  │ 6 formats│   │Processing│   │          │   │ GPU/Dist │   │          │      │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘      │
│       │              │              │              │              │              │
│       │              │              │              │              │              │
│       │              ▼              │              ▼              ▼              │
│       │        ┌──────────┐        │        ┌──────────┐   ┌──────────┐        │
│       │        │ Pipeline │        │        │Calibrate │   │  Export  │        │
│       │        │ Buffer   │        │        │Explain   │   │ONNX/Mobile│       │
│       │        │ Latency  │        │        └──────────┘   └──────────┘        │
│       │        └──────────┘        │              │              │              │
│       │                            │              ▼              ▼              │
│       │                            │        ┌──────────┐   ┌──────────┐        │
│       │                            │        │   Viz    │   │  Mobile  │        │
│       │                            │        │Dashboard │   │ Runtime  │        │
│       │                            │        │ Raster   │   │iOS/Android│       │
│       │                            │        └──────────┘   └──────────┘        │
│       │                            │                                            │
│       ▼                            ▼                                            │
│  ┌──────────┐              ┌──────────┐                                        │
│  │  Synth   │              │  Norms   │                                        │
│  │  Augment │              │ Pediatric│                                        │
│  │  Cohort  │              │Geriatric │                                        │
│  └──────────┘              └──────────┘                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Gap Analysis & Recommendations

### 5.1 Summary: All Resolved Gaps

| Category | Gap | Priority | Status |
|----------|-----|----------|--------|
| **Signal Processing** | ECG R-Peak, QRS, Arrhythmia | HIGH | ✅ DONE |
| **Signal Processing** | HRV Time/Frequency Domain | HIGH | ✅ DONE |
| **Signal Processing** | Wavelet/Hilbert/ICA Transforms | MEDIUM | ✅ DONE |
| **Neural Networks** | Reservoir Computing (ESN/LSM) | MEDIUM | ✅ DONE |
| **Neural Networks** | Hebbian Learning (STDP/BCM/Oja) | MEDIUM | ✅ DONE |
| **Neural Networks** | Network Pruning | MEDIUM | ✅ DONE |
| **Infrastructure** | Real-Time Pipeline | HIGH | ✅ DONE |
| **Infrastructure** | Model Calibration | HIGH | ✅ DONE |
| **Infrastructure** | Explainability Tools | HIGH | ✅ DONE |
| **Infrastructure** | ONNX Export | HIGH | ✅ DONE |
| **Infrastructure** | Integration Tests | HIGH | ✅ DONE |
| **Infrastructure** | API Documentation | HIGH | ✅ DONE |
| **Data Formats** | WFDB/EDF Support | HIGH | ✅ DONE |
| **Data Formats** | GDF/BDF/XDF Support | LOW | ✅ DONE |
| **Normative Data** | Pediatric/Geriatric/Longitudinal | MEDIUM | ✅ DONE |
| **Synthetic Data** | Augmentation Suite | HIGH | ✅ DONE |
| **Synthetic Data** | Virtual Cohorts | MEDIUM | ✅ DONE |
| **GPU** | CUDA/Metal Acceleration | MEDIUM | ✅ DONE |
| **Distributed** | Multi-node Training | MEDIUM | ✅ DONE |
| **Visualization** | Dashboards, Rasters, Heatmaps | MEDIUM | ✅ DONE |
| **Mobile** | iOS/Android Runtime | LOW | ✅ DONE |

### 5.2 Remaining Gaps

#### HIGH Priority (Remaining)

| Gap | Description | Recommendation |
|-----|-------------|----------------|
| **HIPAA Compliance** | Data anonymization utilities | Add PHI detection, de-identification |
| **Clinical Validation** | Tests against published datasets | MIT-BIH, CHB-MIT, PhysioNet validation |
| **Tutorial Notebooks** | Jupyter examples for all workflows | Create 10+ tutorial notebooks |

#### MEDIUM Priority (Remaining)

| Gap | Description | Recommendation |
|-----|-------------|----------------|
| **Knowledge Distillation** | SNN compression | Teacher-student training |
| **BIDS Format** | Neuroimaging standard | Brain Imaging Data Structure |
| **HL7 FHIR** | Healthcare interoperability | Add FHIR resources |
| **Neuromorphic Export** | Loihi, SpiNNaker | Hardware-specific formats |
| **Ethnic Stratification** | Population-specific norms | Ethnicity-aware normative data |
| **Treatment Response** | Intervention modeling | Pre/post treatment simulation |
| **Performance Regression** | Automated benchmarks | CI/CD performance tracking |
| **TensorFlow Lite** | Mobile ML export | Add TFLite converter |

#### LOW Priority (Remaining)

| Gap | Description | Recommendation |
|-----|-------------|----------------|
| **EMD/EEMD** | Empirical mode decomposition | Nonlinear signal analysis |
| **Dendritic Computation** | Multi-compartment models | Biologically detailed neurons |
| **Neuromodulation** | Dopamine/ACh modulation | Neuromodulatory learning |
| **Comorbidity Modeling** | Multi-disease simulation | Disease interaction effects |
| **Practice Effects** | Serial testing corrections | Repeated assessment adjustments |

### 5.3 Implementation Completeness Matrix

| Module | Core | Tests | Docs | Examples |
|--------|:----:|:-----:|:----:|:--------:|
| dpb-core/signal | ✅ | ✅ | ✅ | ✅ |
| dpb-core/pipeline | ✅ | ✅ | ✅ | ✅ |
| dpb-core/io | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/gpu | ✅ | ✅ | ✅ | ✅ |
| dpb-snn/distributed | ✅ | ✅ | ✅ | ✅ |
| dpb-snn/calibration | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/explain | ✅ | ✅ | ✅ | ⚠️ |
| dpb-snn/export | ✅ | ✅ | ✅ | ⚠️ |
| dpb-viz | ✅ | ✅ | ✅ | ⚠️ |
| dpb-mobile | ✅ | ✅ | ✅ | ✅ |

Legend: ✅ Complete | ⚠️ Partial | ❌ Missing

---

## 6. Quick Reference Tables

### 6.1 Signal Processing Quick Reference

| Module | Key Types | Functions |
|--------|-----------|-----------|
| `ecg` | `PanTompkinsDetector`, `RPeak`, `ArrhythmiaDetector` | `detect_r_peaks`, `classify_beat` |
| `hrv` | `HrvTimeDomain`, `HrvFrequencyDomain`, `HrvAnalyzer` | `compute_time_domain`, `compute_frequency_domain` |
| `wavelet` | `ContinuousWaveletTransform`, `DiscreteWaveletTransform` | `transform`, `decompose`, `reconstruct` |
| `hilbert` | `AnalyticSignal` | `hilbert_transform`, `analytic_signal` |
| `ica` | `FastICA`, `ICAResult` | `fit`, `transform`, `fit_transform` |

### 6.2 Data Format Quick Reference

| Format | Reader | Writer | Features |
|--------|--------|--------|----------|
| WFDB | `WfdbReader` | `WfdbWriter` | PhysioNet, annotations |
| EDF/EDF+ | `EdfReader` | `EdfWriter` | Standard polysomnography |
| GDF | `GdfReader` | `GdfWriter` | General Data Format 1.x/2.x |
| BDF | `BdfReader` | `BdfWriter` | 24-bit BioSemi |
| XDF | `XdfFile` | - | Lab Streaming Layer, multi-stream |
| Auto | `UnifiedReader` | - | Magic byte detection |

### 6.3 GPU Quick Reference

| Backend | Device | Buffer | Features |
|---------|--------|--------|----------|
| CUDA | `CudaDevice` | `CudaBuffer` | Streams, memory pool |
| Metal | `MetalDevice` | `MetalBuffer` | Pipelines, compute encoder |
| CPU | Fallback | `Vec<T>` | Always available |

### 6.4 Distributed Training Quick Reference

| Strategy | When to Use | Scaling |
|----------|-------------|---------|
| DataParallel | Large batches | Linear |
| ModelParallel | Large models | Sublinear |
| PipelineParallel | Very deep networks | ~Linear |
| AllReduce | Synchronous training | Best accuracy |
| AsyncSGD | High latency | Best throughput |
| GossipSGD | Decentralized | Fault tolerant |

### 6.5 Visualization Quick Reference

| Type | Class | Output Formats |
|------|-------|----------------|
| Dashboard | `DashboardServer` | WebSocket, JSON |
| Raster | `RasterPlot`, `RasterPlot3D` | SVG, JSON |
| Network | `NetworkGraph` | SVG, JSON |
| Heatmap | `WeightHeatmap`, `ActivationHeatmap` | SVG, JSON |
| Timeline | `EventTimeline` | SVG, JSON |
| Export | `BatchExporter` | SVG, PNG, JSON, CSV |

### 6.6 Mobile Quick Reference

| Platform | Config | Backend |
|----------|--------|---------|
| iOS | `IosRuntime` | Metal, CoreML |
| Android | `AndroidRuntime` | NNAPI, Vulkan |
| Both | `MobileRuntime` | CPU (optimized) |

### 6.7 Calibration Quick Reference

| Method | When to Use | Key Metric |
|--------|-------------|------------|
| Temperature Scaling | Quick calibration, most cases | ECE |
| Platt Scaling | Binary classification | Log-loss |
| Isotonic | Non-parametric, more flexible | ECE |
| MC Dropout | Epistemic uncertainty | Variance |
| Ensemble | Robust uncertainty | Entropy |

### 6.8 Explainability Quick Reference

| Method | Output | Use Case |
|--------|--------|----------|
| Spike Importance | Per-spike scores | Identify critical spikes |
| Attention Maps | Time × Channel heatmap | Visualize focus |
| Integrated Gradients | Per-feature attribution | Feature importance |
| SpikeSHAP | Shapley values | Fair attribution |

### 6.9 Augmentation Quick Reference

| Category | Types | Parameters |
|----------|-------|------------|
| Noise | Gaussian, Pink, Baseline, Powerline, Motion | SNR (dB), frequency, amplitude |
| Temporal | Warp, Shift, Crop, Resample, Dropout | sigma, samples, ratio, rate |
| Spectral | Magnitude, FreqMask, TimeMask | range, width, masks |

### 6.10 Export Quick Reference

| Format | Use Case | Key Types |
|--------|----------|-----------|
| ONNX | Deployment, inference | `OnnxExporter`, `OnnxConfig` |
| Binary | Fast storage | `WeightExporter` |
| JSON | Debugging, inspection | `WeightExporter`, `ModelConfig` |
| Mobile | iOS/Android | `MobileModel`, `MobileRuntime` |

---

## Appendix A: Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0.0 | Dec 2024 | Initial catalog |
| v2.0.0 | Dec 2024 | Added ECG/HRV, transforms, pipeline, calibration, explainability, export, tests, docs |
| v3.0.0 | Dec 2024 | Added GPU (CUDA/Metal), distributed training, dpb-viz, GDF/BDF/XDF, dpb-mobile |

---

## Appendix B: Test Coverage

| Crate | Integration Tests | Unit Tests |
|-------|-------------------|------------|
| dpb-core | `signal_processing_integration.rs`, `io_format_integration.rs` | 250+ |
| dpb-snn | `snn_basic_integration.rs`, `snn_pipeline_integration.rs`, `export_integration.rs` | 400+ |
| dpb-snn (gpu) | GPU feature tests | 17 |
| dpb-snn (distributed) | Distributed feature tests | 46 |
| dpb-synth | Augmentation tests | 150+ |
| dpb-viz | Visualization tests | 75 |
| dpb-mobile | `integration_test.rs` | 63 |
| Workspace | `full_pipeline_integration.rs` | - |

---

## Appendix C: Build Commands

```bash
# Standard build
cargo build --all-features

# GPU features
cargo build -p dpb-snn --features cuda
cargo build -p dpb-snn --features metal

# Distributed features
cargo build -p dpb-snn --features distributed

# Mobile builds
cargo build -p dpb-mobile --target aarch64-apple-ios --features ios
cargo ndk --target aarch64-linux-android -- build -p dpb-mobile --features android

# Run all tests
cargo test --all

# Generate documentation
cargo doc --no-deps --all-features --open
```

---

*End of System Catalog v3.0.0*
