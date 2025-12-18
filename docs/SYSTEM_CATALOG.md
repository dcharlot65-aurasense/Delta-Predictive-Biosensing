# Delta-Predictive Biosensing (DPB) System Catalog v1.0.0

> **Last Updated:** December 2024
> **Framework Version:** 0.1.0
> **Total Modules:** 100+ | **Encoders:** 77+ | **Generators:** 200+ | **Decoders:** 48+

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
| Crates | 10 |
| Neuron Models | 19 |
| Event Encoders | 77+ |
| Population Templates | 61+ |
| Synthetic Generators | 200+ |
| Output Decoders | 48+ |
| ANN Baselines | 44 |
| Clinical Metrics | 100+ |
| Normative Databases | Age/Sex stratified for 100+ metrics |

### Crate Overview

| Crate | Purpose | LOC (approx) |
|-------|---------|--------------|
| **dpb-core** | Types, traits, GPU, signal processing | 15,000+ |
| **dpb-encoders** | Event-based encoders, population templates | 12,000+ |
| **dpb-neurons** | 19 neuron models, surrogate gradients | 8,000+ |
| **dpb-snn** | SNN architectures, training, decoders | 20,000+ |
| **dpb-synth** | 200+ synthetic biosignal generators | 25,000+ |
| **dpb-norms** | Normative databases, assessments | 6,000+ |
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

### 2.2 Domain-Specific Signal Analysis

| Domain | Capability | Implementation | Location |
|--------|------------|----------------|----------|
| **EEG** | Band Power Analysis | `compute_band_powers`, `EegBands` | `dpb-core/signal/eeg/bands.rs` |
| **EEG** | Artifact Detection | `detect_artifacts`, `ArtifactType` | `dpb-core/signal/eeg/artifacts.rs` |
| **EEG** | ERP Analysis | `ErpAnalyzer`, `ErpComponent` | `dpb-core/signal/eeg/erp.rs` |
| **EEG** | Seizure Detection | `SeizureDetector`, `SeizureEvent` | `dpb-core/signal/eeg/seizure.rs` |
| **EEG** | Alpha Asymmetry | `alpha_asymmetry` | `dpb-core/signal/eeg/bands.rs` |
| **ECG** | R-Peak Detection | via PPG pulse detection patterns | `dpb-core/signal/ppg.rs` |
| **PPG** | Pulse Analysis | `PpgAnalyzer`, `PulseWaveFeatures` | `dpb-core/signal/ppg.rs` |
| **PPG** | SpO2 Estimation | `SpO2Result` | `dpb-core/signal/ppg.rs` |
| **PPG** | PRV Analysis | `PrvMetrics` | `dpb-core/signal/ppg.rs` |
| **EDA** | Tonic/Phasic Decomposition | `EdaDecomposition` | `dpb-core/signal/eda.rs` |
| **EDA** | SCR Detection | `ScrEvent`, `EdaMetrics` | `dpb-core/signal/eda.rs` |
| **EMG** | Burst Detection | `EmgBurst`, `EmgAnalyzer` | `dpb-core/signal/emg.rs` |
| **EMG** | Fatigue Metrics | `FatigueMetrics` | `dpb-core/signal/emg.rs` |
| **Voice** | F0/Jitter/Shimmer | `F0Metrics`, `JitterMetrics`, `ShimmerMetrics` | `dpb-core/signal/voice.rs` |
| **Voice** | Voice Quality | `VoiceQualityMetrics`, `SpectralVoiceFeatures` | `dpb-core/signal/voice.rs` |
| **Eye** | Saccade/Fixation Detection | `Saccade`, `Fixation`, `Blink` | `dpb-core/signal/eye.rs` |
| **Eye** | Pupil Metrics | `PupilMetrics`, `SmoothPursuitMetrics` | `dpb-core/signal/eye.rs` |
| **Respiratory** | Breath Detection | `BreathEvent`, `RespiratoryAnalyzer` | `dpb-core/signal/respiratory.rs` |
| **Respiratory** | Apnea Detection | `ApneaEvent`, `ApneaType` | `dpb-core/signal/respiratory.rs` |
| **Respiratory** | Sleep Breathing | `SleepBreathingAnalysis`, `SleepApneaSeverity` | `dpb-core/signal/respiratory.rs` |

### 2.3 Fatigue Detection

| Capability | Implementation | Location |
|------------|----------------|----------|
| **EMG Fatigue** | `EmgFatigueAnalyzer` (MDF slope, spectral shift) | `dpb-core/signal/fatigue.rs` |
| **Force Fatigue** | `ForceFatigueAnalyzer` (decline rate, endurance) | `dpb-core/signal/fatigue.rs` |
| **Cognitive Fatigue** | `CognitiveFatigueAnalyzer` (RT deterioration, lapses) | `dpb-core/signal/fatigue.rs` |
| **Integrated Fatigue** | `IntegratedFatigueMetrics`, `integrate_fatigue` | `dpb-core/signal/fatigue.rs` |
| **Classification** | `FatigueType`, `FatigueSeverity` | `dpb-core/signal/fatigue.rs` |

### 2.4 Event-Based Encoding

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
| **Pain** | Threshold, Temporal Summation, QST | `dpb-encoders/pain.rs` |
| **Vestibular** | VOR Gain, Nystagmus, Caloric | `dpb-encoders/vestibular.rs` |

### 2.5 Spiking Neural Networks

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Neuron Models (19)** | LIF, ALIF, ELIF, QLIF, GLIF, Izhikevich, AdEx, HH, etc. | `dpb-neurons/models/*.rs` |
| **Surrogate Gradients (6)** | FastSigmoid, Arctan, Triangular, SuperSpike, etc. | `dpb-neurons/surrogates.rs` |
| **Spiking Layers** | SpikingLinear, SpikingConv1d/2d, SpikingRNN/LSTM | `dpb-snn/layers/*.rs` |
| **Architectures** | Feedforward, Convolutional, Recurrent, Transformer, GCN | `dpb-snn/architectures/*.rs` |
| **Training** | BPTT, OTTT, SLTT | `dpb-snn/training/*.rs` |
| **ANN-to-SNN Conversion** | `ANNToSNNConverter`, `WeightNormalization` | `dpb-snn/conversion/*.rs` |
| **Multi-Modal Fusion** | Early, Late, Cross-Modal, Hierarchical, Gated (8 types) | `dpb-snn/fusion/*.rs` |

### 2.6 Output Decoding (48+ Decoders)

| Category | Decoders | Location |
|----------|----------|----------|
| **Rate-Based** | SpikeRate, FirstSpike, Population, Windowed, Exponential | `dpb-snn/decoders/rate.rs` |
| **Temporal** | TemporalPattern, Latency, ISI, Burst, Phase, RankOrder | `dpb-snn/decoders/temporal.rs` |
| **Clinical Scores** | UPDRS (motor/tremor/brady/rigid/gait), TUG, Berg, MoCA, Tinetti | `dpb-snn/decoders/clinical.rs` |
| **Pain** | VAS, NRS, QST | `dpb-snn/decoders/clinical.rs` |
| **Vestibular** | VOR, Canal Paresis, BPPV | `dpb-snn/decoders/clinical.rs` |
| **Force** | GRF, Grip, RFD | `dpb-snn/decoders/regression.rs` |
| **Regression** | HR, HRV, Tremor Freq/Amp, Gait Velocity, RT, Speech Rate | `dpb-snn/decoders/regression.rs` |
| **Classification** | Binary, MultiClass, TremorType, SleepStage, Emotion | `dpb-snn/decoders/classification.rs` |

### 2.7 Synthetic Data Generation (200+ Generators)

| Domain | Generator Types | Location |
|--------|----------------|----------|
| **ECG** | Morphology, HRV, Arrhythmia, RSA, HR Recovery | `dpb-synth/contact/ecg.rs` |
| **PPG** | Waveform, Artifacts, SpO2 variations | `dpb-synth/contact/ppg.rs` |
| **EDA** | Tonic, SCR, Stimulus-Locked, Arousal, Artifacts | `dpb-synth/contact/eda.rs` |
| **EMG** | Surface, Voluntary, Pathological (fasciculations) | `dpb-synth/contact/emg.rs` |
| **Tremor** | Physiological, Parkinsonian, Essential, Cerebellar | `dpb-synth/contact/tremor.rs` |
| **Noise** | White, Pink, Powerline, Motion, Baseline, Electrode (11 types) | `dpb-synth/contact/noise.rs` |
| **Gait** | Normal, Parkinsonian, Cerebellar, Hemiplegic, Spastic | `dpb-synth/pose/*.rs` |
| **Hand** | Tapping, Movement, Tremor (rest/postural/kinetic) | `dpb-synth/hand/*.rs` |
| **Eye** | Fixation, Saccade, Pursuit, Pupil | `dpb-synth/eye/*.rs` |
| **Voice** | Phonation, Articulation, Prosody | `dpb-synth/voice/*.rs` |
| **EEG** | Resting, Band-specific, Seizure, Artifacts | `dpb-synth/neural/eeg.rs` |
| **ERP** | Oddball, P300, N400, MMN | `dpb-synth/neural/erp.rs` |
| **Sleep** | N1, N2, N3, REM, Full cycles | `dpb-synth/neural/sleep.rs` |
| **Force** | GRF, Grip, RFD | `dpb-synth/force/*.rs` |
| **Balance** | COP, Sway, Perturbation | `dpb-synth/balance/*.rs` |
| **Vestibular** | VOR, Nystagmus, Caloric | `dpb-synth/vestibular/*.rs` |
| **Streaming** | All modalities with real-time support | `dpb-synth/streaming.rs` |

### 2.8 Disease & Pathology Models

| Condition | Model Components | Location |
|-----------|-----------------|----------|
| **ALS** | ALSFRS-R, EMG Signatures, Respiratory Status, Stages | `dpb-synth/pathology/als.rs` |
| **MS** | EDSS, Relapse/Remitting, Symptom Profiles, Types | `dpb-synth/pathology/ms.rs` |
| **Stroke** | NIHSS, Brunnstrom, Aphasia, Recovery Phases | `dpb-synth/pathology/stroke.rs` |
| **Progression** | Linear, Exponential, Relapsing, Recovery curves | `dpb-synth/pathology/progression.rs` |
| **Medications** | 12 classes, PK modeling, Drug interactions | `dpb-synth/pathology/medication.rs` |
| **Parkinson's** | Shuffling, Festination, FOG, Asymmetric gait | `dpb-synth/pose/pathological.rs` |

### 2.9 Normative Databases

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Demographic Filtering** | Age, Sex, Ethnicity, Handedness, Education | `dpb-norms/demographics.rs` |
| **100+ Metrics** | Cardiovascular, Motor, Cognitive, Sensory, Clinical | `dpb-norms/metrics.rs` |
| **Statistical Operations** | Percentile, Z-score, Reference ranges, MDC | `dpb-norms/lib.rs` |
| **Multi-Modal Assessment** | Domain summaries, Profile classification | `dpb-norms/multimodal.rs` |
| **Dissociation Detection** | Cognitive-motor dissociation patterns | `dpb-norms/multimodal.rs` |
| **Report Generation** | `generate_report()` | `dpb-norms/multimodal.rs` |

### 2.10 Cognitive Assessments

| Domain | Tasks | Location |
|--------|-------|----------|
| **Reaction Time** | Simple RT, Choice RT | `dpb-cognitive/reaction_time/` |
| **Working Memory** | N-Back | `dpb-cognitive/working_memory/` |
| **Attention** | CPT, Stroop | `dpb-cognitive/attention/` |
| **Executive Function** | Go/No-Go, Flanker, WCST | `dpb-cognitive/executive/` |
| **ADHD** | QbTest, Eye Tracking | `dpb-cognitive/adhd/` |

### 2.11 Power & Efficiency Analysis

| Capability | Implementation | Location |
|------------|----------------|----------|
| **Neuromorphic Power** | LIF, Izhikevich, Synaptic, Calcium models | `dpb-core/power/*.rs` |
| **Digital Power** | ANN/SNN on GPU/CPU, Memory, Compute | `dpb-core/power/*.rs` |
| **Comparison** | Neuromorphic vs. Digital | `dpb-core/power/*.rs` |

### 2.12 Bindings & Interoperability

| Target | Implementation | Location |
|--------|----------------|----------|
| **Python** | PyO3 wrappers for all major components | `dpb-python/` |
| **C/C++** | Extern "C" FFI with safe memory management | `dpb-ffi/` |
| **GPU** | WebGPU compute shaders | `dpb-core/gpu/`, `dpb-neurons/gpu/` |

---

## 3. Implementation → Capabilities Map

### 3.1 dpb-core

```
dpb-core/
├── types.rs         → SpikeEvent, SpikeTrain, TimeSeries, SignalBuffer, GroundTruth
├── traits.rs        → EventEncoder, SNNNetwork, Dataset, Metric, Optimizer, GpuKernel
├── error.rs         → DpbError, Result type
├── signal/
│   ├── mod.rs       → normalize, remove_dc_offset, find_peaks, rms, energy, snr_db
│   ├── fft.rs       → FftProcessor, Stft, WindowType, fft_frequencies
│   ├── filter.rs    → FirFilter, IirFilter, FilterType, median_filter
│   ├── resample.rs  → downsample, upsample, resample_linear, resample_to_length
│   ├── eeg/         → Band powers, artifacts, ERP, seizure detection
│   ├── eda.rs       → EdaAnalyzer, EdaDecomposition, ScrEvent
│   ├── emg.rs       → EmgAnalyzer, EmgBurst, FatigueMetrics
│   ├── ppg.rs       → PpgAnalyzer, PulseWaveFeatures, SpO2Result, PrvMetrics
│   ├── eye.rs       → EyeAnalyzer, Saccade, Fixation, Blink, PupilMetrics
│   ├── voice.rs     → VoiceAnalyzer, F0/Jitter/Shimmer/VoiceQuality metrics
│   ├── respiratory.rs → RespiratoryAnalyzer, ApneaEvent, SleepBreathingAnalysis
│   └── fatigue.rs   → EmgFatigueAnalyzer, ForceFatigueAnalyzer, CognitiveFatigueAnalyzer
├── metrics/         → Classification, clinical, regression, signal, efficiency metrics
├── gpu/             → GpuContext, GpuBuffer, compute pipelines
├── viz/             → Visualization utilities
├── biomechanics/    → Balance, force analysis
├── cardiopulmonary/ → Gas exchange, ventilatory analysis, VO2
├── vestibular/      → VOR, posturography
├── pain/            → Pain scales, QST
├── sleep/           → Sleep staging
├── somatosensory/   → Proprioception, vibration
└── power/           → Neuromorphic and digital power estimation
```

### 3.2 dpb-encoders

```
dpb-encoders/
├── base/            → LevelCrossing, TemplateDeviation, Derivative, DiscreteEvent encoders
├── contact/
│   ├── ecg.rs       → 5 ECG encoders (R-Peak, Morphology, ST, HRV, Arrhythmia)
│   ├── ppg.rs       → 4 PPG encoders (Pulse, Amplitude, PTT, SpO2)
│   ├── eda.rs       → 4 EDA encoders (Level, SCR, Tonic, Phasic)
│   ├── emg.rs       → 4 EMG encoders (Burst, Amplitude, Fatigue, MU)
│   └── tremor.rs    → 4 Tremor encoders (Level, Frequency, Amplitude, Type)
├── pose/
│   ├── gait.rs      → 7 Gait encoders (HeelStrike, ToeOff, Phase, Stride, Asymmetry, etc.)
│   └── keypoint.rs  → 4 Keypoint encoders (Deviation, JointAngle, Sway, Posture)
├── hand/
│   ├── tapping.rs   → 5 Tapping encoders (Onset, Aperture, Frequency, Decrement, Regularity)
│   └── tremor.rs    → 4 Hand tremor encoders (Rest, Postural, Kinetic, Pinch)
├── eye/
│   ├── saccade.rs   → 5 Saccade encoders (Onset, MainSequence, Latency, Accuracy, Microsaccade)
│   ├── fixation.rs  → 3 Fixation encoders (Stability, Drift, Hold)
│   └── pupil.rs     → 4 Pupil encoders (Dilation, LightReflex, Constriction, Near)
├── voice/
│   ├── phonation.rs → 5 Phonation encoders (F0, Jitter, Shimmer, HNR, Vibrato)
│   ├── articulation.rs → 3 Articulation encoders (Formant, VowelSpace, Consonant)
│   └── prosody.rs   → 4 Prosody encoders (Rate, Pause, Intonation, Stress)
├── eeg.rs           → 6 EEG band + seizure + artifact encoders
├── balance.rs       → 3 Balance encoders (COP velocity/area, sway)
├── force.rs         → 3 Force encoders (GRF, RFD, Grip)
├── pain.rs          → 3 Pain encoders (Threshold, TemporalSummation, QST)
├── vestibular.rs    → 3 Vestibular encoders (VOR, Nystagmus, Caloric)
├── cardiopulmonary.rs → 3 Cardiopulmonary encoders (HRV LF/HF, RespRate, Phase)
├── cognitive.rs     → 3 Cognitive encoders (RT, Accuracy, WM)
└── templates/       → 61+ population templates for clinical priors
```

### 3.3 dpb-neurons

```
dpb-neurons/
├── models/
│   ├── if.rs        → IfNeuron (simple integrate-and-fire)
│   ├── lif.rs       → LifNeuron (leaky IF)
│   ├── clif.rs      → ClifNeuron (current-based LIF)
│   ├── alif.rs      → AlifNeuron (adaptive LIF)
│   ├── elif.rs      → ElifNeuron (exponential LIF)
│   ├── qlif.rs      → QlifNeuron (quadratic LIF)
│   ├── glif.rs      → GlifNeuron (generalized LIF)
│   ├── izhikevich.rs → IzhikevichNeuron
│   ├── adex.rs      → AdExNeuron
│   ├── calcium.rs   → CalciumNeuron
│   ├── hh.rs        → HodgkinHuxleyNeuron
│   ├── fhn.rs       → FitzHughNagumoNeuron
│   ├── morris_lecar.rs → MorrisLecarNeuron
│   ├── srm.rs       → SrmNeuron
│   ├── stochastic.rs → StochasticLifNeuron
│   ├── recurrent.rs → RecurrentNeuron
│   ├── xylo.rs      → XyloLifNeuron (hardware)
│   ├── pulsar.rs    → PulsarLifNeuron (hardware)
│   └── quantized.rs → QuantizedLifNeuron (edge)
├── surrogates.rs    → 6 surrogate gradient functions
├── batch/           → BatchLifLayer, BatchNetwork, PopulationStats
└── gpu/             → GPU-accelerated neuron kernels
```

### 3.4 dpb-snn

```
dpb-snn/
├── tensor/          → SpikeTensor, SpikeRepresentation
├── layers/
│   ├── linear.rs    → SpikingLinear
│   ├── conv.rs      → SpikingConv1d, SpikingConv2d
│   ├── pooling.rs   → SpikingSumPool2d, SpikingMaxPool2d
│   ├── recurrent.rs → SpikingRNN, SpikingLSTM
│   └── attention.rs → SpikingAttention
├── architectures/
│   ├── feedforward.rs → FeedforwardSNN
│   ├── convolutional.rs → ConvolutionalSNN
│   ├── recurrent.rs → RecurrentSNN
│   ├── transformer.rs → SpikingTransformer
│   └── gcn.rs       → SpikingGCN
├── training/
│   ├── bptt.rs      → Backpropagation through time
│   ├── ottt.rs      → Online trace-based learning
│   ├── sltt.rs      → Spike-local learning
│   ├── loss.rs      → Loss functions
│   └── optimizers.rs → Adam, SGD
├── decoders/
│   ├── rate.rs      → 9 rate-based decoders
│   ├── temporal.rs  → 7 temporal decoders
│   ├── clinical.rs  → 27+ clinical scale decoders
│   ├── regression.rs → 10 regression decoders
│   └── classification.rs → 10 classification decoders
├── fusion/          → 8 multi-modal fusion strategies
├── conversion/      → ANN-to-SNN conversion
├── analysis/        → Training analysis tools
├── baselines/       → 44 ANN baseline architectures
└── export/          → ONNX, TorchScript export
```

### 3.5 dpb-synth

```
dpb-synth/
├── traits.rs        → SyntheticGenerator, GroundTruth, ParameterSpace
├── streaming.rs     → StreamingGenerator, RingBuffer, real-time support
├── contact/         → ECG, PPG, EDA, EMG, Tremor, Respiratory, Thermal, Noise
├── eye/             → Fixation, Saccade, Pursuit, Pupil
├── pose/            → Gait (normal + 7 pathological), Keypoint
├── hand/            → Tapping, Movement, Tremor
├── voice/           → Phonation, Articulation, Prosody
├── neural/          → EEG, ERP, Sleep
├── force/           → GRF, Grip, RFD
├── balance/         → COP, Sway, Perturbation
├── vestibular/      → VOR, Nystagmus, Caloric
├── pain/            → Pain progression, QST
├── cardiopulmonary/ → Breathing, HRV
├── cognitive/       → RT, Accuracy, WM load
├── pathology/       → ALS, MS, Stroke, Progression, Medication
├── multimodal.rs    → Multi-modal synchronization
├── level3/          → Audio, Video, SMPL, Style transfer
└── media/           → 20+ external backend integrations
```

### 3.6 dpb-norms

```
dpb-norms/
├── lib.rs           → NormativeStats, ImpairmentLevel, NormativeComparison
├── database.rs      → NormativeDatabase, NormativeEntry, NormativeTable
├── demographics.rs  → Demographics, DemographicsFilter, Sex, Ethnicity, etc.
├── metrics.rs       → MetricType (100+), MetricDomain, MetricDirection
└── multimodal.rs    → MultiModalAssessment, DomainSummary, ProfileClassification
```

---

## 4. Dependency Graph

### 4.1 Crate Dependencies

```
                              ┌─────────────────┐
                              │   dpb-bench     │
                              └────────┬────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐            ┌─────────────────┐            ┌─────────────────┐
│  dpb-python   │            │     dpb-snn     │            │   dpb-synth     │
└───────┬───────┘            └────────┬────────┘            └────────┬────────┘
        │                              │                              │
        │                    ┌─────────┼─────────┐                    │
        │                    │         │         │                    │
        │                    ▼         ▼         ▼                    │
        │            ┌───────────┐ ┌─────────┐ ┌───────────────┐      │
        │            │dpb-neurons│ │dpb-norms│ │ dpb-encoders  │      │
        │            └─────┬─────┘ └────┬────┘ └───────┬───────┘      │
        │                  │            │              │              │
        │                  └────────────┼──────────────┘              │
        │                               │                             │
        └───────────────────────────────┼─────────────────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │     dpb-core      │
                              └───────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │     dpb-ffi       │
                              └───────────────────┘

┌─────────────────┐
│  dpb-cognitive  │  (Standalone - depends only on rand)
└─────────────────┘
```

### 4.2 Module-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA ACQUISITION                                   │
│  (Real sensors or dpb-synth generators)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        dpb-core/signal                                       │
│  ┌─────────┐  ┌────────┐  ┌──────────┐  ┌─────────────────────────────────┐ │
│  │   FFT   │  │ Filter │  │ Resample │  │ Domain-Specific Analyzers      │ │
│  │ (fft.rs)│  │(filter)│  │(resample)│  │ (eeg, eda, emg, ppg, eye, etc.)│ │
│  └────┬────┘  └───┬────┘  └────┬─────┘  └─────────────────┬───────────────┘ │
│       └───────────┴────────────┴──────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         dpb-encoders                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │  Base Encoders │  │ Modality-Spec. │  │   Population   │                 │
│  │ (LevelCrossing,│  │    Encoders    │  │   Templates    │                 │
│  │  Derivative)   │  │ (77+ encoders) │  │ (61+ templates)│                 │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                 │
│          └───────────────────┴───────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │  Spike Events  │
                           │  (SpikeTrain)  │
                           └────────┬───────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           dpb-neurons                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  19 Neuron Models (IF, LIF, ALIF, Izhikevich, HH, AdEx, ...)          │ │
│  │  + 6 Surrogate Gradients                                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            dpb-snn                                           │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │    Layers    │  │ Architectures │  │   Training    │  │   Fusion     │  │
│  │(Linear,Conv, │  │(FF,Conv,RNN,  │  │(BPTT,OTTT,   │  │(8 strategies)│  │
│  │ RNN,Attn)    │  │ Transformer)  │  │ SLTT)        │  │              │  │
│  └──────┬───────┘  └───────┬───────┘  └───────┬───────┘  └──────┬───────┘  │
│         └──────────────────┴──────────────────┴─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       dpb-snn/decoders                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────┐               │
│  │  Rate    │  │ Temporal │  │   Clinical   │  │ Regression │               │
│  │ (9 types)│  │ (7 types)│  │ (27+ scales) │  │ (10 types) │               │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └─────┬──────┘               │
│       └─────────────┴───────────────┴────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          dpb-norms                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Normative Comparison: Z-score, Percentile, Reference Range            │ │
│  │  Multi-Modal Assessment: Domain summaries, Risk stratification         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Clinical Output                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Biomarkers  │  │   Severity   │  │  Progression │  │    Risk      │    │
│  │   Scores     │  │   Staging    │  │   Tracking   │  │ Stratification│    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 External Dependencies

| Category | Dependencies |
|----------|--------------|
| **Math/Linear Algebra** | ndarray, nalgebra, num-complex |
| **FFT** | rustfft |
| **Random** | rand, rand_distr |
| **GPU** | wgpu |
| **Serialization** | serde, serde_json, bincode |
| **Error Handling** | thiserror |
| **Python** | pyo3, numpy |
| **FFI** | libc, cbindgen |
| **Testing** | approx, criterion |

---

## 5. Gap Analysis & Recommendations

### 5.1 Signal Processing Gaps

| Gap | Priority | Recommendation | Effort |
|-----|----------|----------------|--------|
| **ECG R-Peak Detector** | HIGH | Add dedicated R-peak detection algorithm (Pan-Tompkins) | 2-3 days |
| **ECG QRS Morphology** | MEDIUM | Add QRS template matching and classification | 3-4 days |
| **ECG Arrhythmia Detection** | MEDIUM | Add AFib, PVC, PAC detection algorithms | 5-7 days |
| **HRV Time-Domain** | HIGH | Add SDNN, RMSSD, pNN50 calculators | 1-2 days |
| **HRV Frequency-Domain** | HIGH | Add LF/HF power, LF/HF ratio | 2-3 days |
| **Wavelet Transform** | MEDIUM | Add CWT/DWT for multi-resolution analysis | 3-4 days |
| **Hilbert Transform** | MEDIUM | Add proper Hilbert transform for envelope/phase | 2-3 days |
| **Independent Component Analysis** | LOW | Add ICA for artifact removal | 4-5 days |

### 5.2 Encoder Gaps

| Gap | Priority | Recommendation | Effort |
|-----|----------|----------------|--------|
| **Multi-Scale Encoders** | MEDIUM | Add encoders operating at multiple temporal scales | 3-4 days |
| **Learned Encoders** | MEDIUM | Add trainable encoder parameters | 4-5 days |
| **Compression-Aware** | LOW | Add rate-distortion optimization | 3-4 days |
| **Cross-Modal Encoders** | LOW | Add encoders that fuse multiple inputs | 4-5 days |

### 5.3 Neural Network Gaps

| Gap | Priority | Recommendation | Effort |
|-----|----------|----------------|--------|
| **Reservoir Computing** | MEDIUM | Add Echo State Network / Liquid State Machine | 4-5 days |
| **Hebbian Learning** | MEDIUM | Add local Hebbian/anti-Hebbian rules | 3-4 days |
| **Dendritic Computation** | LOW | Add dendritic tree models | 5-7 days |
| **Neuromodulation** | LOW | Add dopamine/acetylcholine modulation | 4-5 days |
| **Pruning/Sparsification** | MEDIUM | Add network pruning strategies | 3-4 days |
| **Knowledge Distillation** | MEDIUM | Add SNN-to-SNN and ANN-to-SNN distillation | 4-5 days |

### 5.4 Clinical/Application Gaps

| Gap | Priority | Recommendation | Effort |
|-----|----------|----------------|--------|
| **Real-Time Pipeline** | HIGH | Add end-to-end real-time inference pipeline | 5-7 days |
| **Model Calibration** | HIGH | Add uncertainty quantification and calibration | 3-4 days |
| **Explainability** | HIGH | Add attention visualization, feature importance | 4-5 days |
| **Longitudinal Tracking** | MEDIUM | Add within-subject change detection | 3-4 days |
| **Confidence Intervals** | MEDIUM | Add bootstrap/Bayesian uncertainty | 3-4 days |
| **HIPAA Compliance** | HIGH | Add data anonymization utilities | 2-3 days |

### 5.5 Normative Database Gaps

| Gap | Priority | Recommendation | Effort |
|-----|----------|----------------|--------|
| **Pediatric Norms** | MEDIUM | Add pediatric reference ranges (ages 2-17) | 3-4 days |
| **Geriatric Norms** | MEDIUM | Add stratification for 80+ age group | 2-3 days |
| **Ethnic Stratification** | MEDIUM | Add ethnicity-specific norms where relevant | 3-4 days |
| **Longitudinal MDC** | MEDIUM | Add within-subject minimal detectable change | 2-3 days |
| **Practice Effects** | LOW | Add practice effect corrections for serial testing | 2-3 days |

### 5.6 Synthetic Data Gaps

| Gap | Priority | Recommendation | Effort |
|-----|----------|----------------|--------|
| **Multi-Patient Cohort** | MEDIUM | Add cohort-level variation generators | 3-4 days |
| **Longitudinal Trajectories** | MEDIUM | Add individual progression curves | 3-4 days |
| **Treatment Response** | MEDIUM | Add intervention effect modeling | 3-4 days |
| **Comorbidity Modeling** | LOW | Add multi-condition interactions | 4-5 days |
| **Augmentation Suite** | HIGH | Add signal augmentation for training | 2-3 days |

### 5.7 Integration Gaps

| Gap | Priority | Recommendation | Effort |
|-----|----------|----------------|--------|
| **WFDB/PhysioNet** | HIGH | Add WFDB format reading/writing | 2-3 days |
| **EDF/EDF+** | HIGH | Add European Data Format support | 2-3 days |
| **BIDS Format** | MEDIUM | Add Brain Imaging Data Structure support | 3-4 days |
| **HL7 FHIR** | MEDIUM | Add healthcare interoperability | 4-5 days |
| **ONNX Runtime** | HIGH | Add ONNX inference for deployment | 3-4 days |
| **TensorFlow Lite** | MEDIUM | Add TFLite export for mobile | 3-4 days |
| **Neuromorphic Hardware** | MEDIUM | Add Loihi/SpiNNaker export | 5-7 days |

### 5.8 Testing & Validation Gaps

| Gap | Priority | Recommendation | Effort |
|-----|----------|----------------|--------|
| **Integration Tests** | HIGH | Add end-to-end pipeline tests | 3-4 days |
| **Clinical Validation** | HIGH | Add tests against published datasets | 5-7 days |
| **Performance Regression** | MEDIUM | Add automated performance tracking | 2-3 days |
| **Fuzzing** | LOW | Add property-based testing for encoders | 3-4 days |

### 5.9 Documentation Gaps

| Gap | Priority | Recommendation | Effort |
|-----|----------|----------------|--------|
| **API Reference** | HIGH | Generate comprehensive rustdoc | 2-3 days |
| **Tutorial Notebooks** | HIGH | Add Jupyter notebooks with examples | 4-5 days |
| **Architecture Guide** | MEDIUM | Add detailed architecture documentation | 2-3 days |
| **Deployment Guide** | MEDIUM | Add production deployment instructions | 2-3 days |
| **Clinical Guide** | MEDIUM | Add clinical interpretation guidelines | 3-4 days |

---

### 5.10 Prioritized Implementation Roadmap

#### Phase 1: Critical Gaps (1-2 weeks)

1. **ECG R-Peak Detection** - Essential for cardiac applications
2. **HRV Time/Frequency Domain** - Core cardiovascular metrics
3. **Real-Time Pipeline** - Production deployment requirement
4. **Data Format Support** - WFDB, EDF for real data
5. **Integration Tests** - Quality assurance

#### Phase 2: High Priority (2-4 weeks)

6. **Model Calibration** - Uncertainty quantification
7. **Explainability Tools** - Clinical acceptance
8. **ONNX Runtime** - Deployment flexibility
9. **Signal Augmentation Suite** - Training data enhancement
10. **API Documentation** - Developer adoption

#### Phase 3: Medium Priority (4-8 weeks)

11. **Reservoir Computing** - Alternative architectures
12. **Network Pruning** - Efficiency optimization
13. **Wavelet Transform** - Multi-resolution analysis
14. **Pediatric/Geriatric Norms** - Extended populations
15. **Longitudinal Tracking** - Disease monitoring

#### Phase 4: Future Enhancements (8+ weeks)

16. **Dendritic Computation** - Advanced neuron models
17. **Neuromodulation** - Biologically plausible learning
18. **Neuromorphic Hardware Export** - Loihi, SpiNNaker
19. **HL7 FHIR** - Healthcare integration
20. **Comorbidity Modeling** - Complex patient simulation

---

## 6. Quick Reference Tables

### 6.1 Encoder Quick Reference

| Modality | Encoder Count | Key Encoders |
|----------|---------------|--------------|
| ECG | 5 | RPeak, Morphology, ST, HRV, Arrhythmia |
| PPG | 4 | Pulse, Amplitude, PTT, SpO2 |
| EDA | 4 | Level, SCR, Tonic, Phasic |
| EMG | 4 | Burst, Amplitude, Fatigue, MU |
| Tremor | 4 | Level, Frequency, Amplitude, Type |
| Gait | 7 | HeelStrike, ToeOff, Phase, Stride, Asymmetry, Velocity, Cadence |
| Tapping | 5 | Onset, Aperture, Frequency, Decrement, Regularity |
| Saccade | 5 | Onset, MainSequence, Latency, Accuracy, Microsaccade |
| Fixation | 3 | Stability, Drift, Hold |
| Pupil | 4 | Dilation, LightReflex, Constriction, Near |
| Voice | 12 | F0, Jitter, Shimmer, HNR, Formant, VowelSpace, Rate, Pause, etc. |
| EEG | 6+ | Alpha, Beta, Theta, Gamma, Seizure, Artifact |
| Balance | 3 | COP Velocity, Area, Sway |
| Force | 3 | GRF, RFD, Grip |
| Pain | 3 | Threshold, TemporalSummation, QST |
| Vestibular | 3 | VOR, Nystagmus, Caloric |

### 6.2 Decoder Quick Reference

| Category | Count | Key Decoders |
|----------|-------|--------------|
| Rate-Based | 9 | SpikeRate, FirstSpike, Population, Windowed, Exponential |
| Temporal | 7 | TemporalPattern, Latency, ISI, Burst, Phase, RankOrder |
| UPDRS | 6 | Motor, Tremor, Bradykinesia, Rigidity, Gait, Total |
| Balance | 4 | Tinetti, MiniBEST, Berg, TUG |
| Pain | 3 | VAS, NRS, QST |
| Vestibular | 3 | VOR, CanalParesis, BPPV |
| Force | 3 | GRF, Grip, RFD |
| Cardiopulmonary | 3 | HRV, Respiratory, VO2 |
| Cognitive | 3 | RT, Attention, WorkingMemory |
| Regression | 10 | HR, HRV, TremorFreq, GaitVelocity, RT, etc. |
| Classification | 10 | Binary, MultiClass, TremorType, SleepStage, etc. |

### 6.3 Neuron Model Quick Reference

| Model | Complexity | Use Case |
|-------|------------|----------|
| IF | Low | Simple spiking, baseline |
| LIF | Low | Standard choice, efficient |
| CLIF | Low | Current-based applications |
| ALIF | Medium | Adaptation, temporal patterns |
| ELIF | Medium | Precise spike timing |
| QLIF | Medium | Sub-threshold dynamics |
| GLIF | High | Multi-timescale adaptation |
| Izhikevich | Medium | Diverse dynamics, efficient |
| AdEx | Medium | Adaptation + exponential |
| Calcium | Medium | Slow adaptation, bursting |
| HH | High | Biophysical accuracy |
| FHN | Medium | Oscillatory dynamics |
| MorrisLecar | Medium | Calcium dynamics |
| SRM | Medium | Kernel-based response |
| Stochastic | Medium | Noisy environments |
| Recurrent | Medium | Self-excitation |
| Xylo/Pulsar/Quantized | Low | Hardware deployment |

### 6.4 Normative Metric Domains

| Domain | Metric Count | Examples |
|--------|--------------|----------|
| Cardiovascular | 10+ | HR, HRV, BP, CO, SpO2 |
| Motor | 15+ | Tremor, Gait, Grip, Tapping, ROM |
| Cognitive | 10+ | RT, Accuracy, WM, Attention |
| Pain | 5+ | VAS, NRS, Threshold, TS |
| Vestibular | 5+ | VOR, HIT, DVA, Caloric |
| Sleep | 5+ | Latency, Efficiency, REM% |
| PPG | 6 | PTT, PWV, AI, SI, PI, PRV-SDNN |
| EDA | 5 | SCL, SCR freq/amp, NS-SCR, Recovery |
| EEG | 7 | Delta/Theta/Alpha/Beta/Gamma power, ratios |
| Eye | 9 | Saccade metrics, Fixation, Pupil |
| Voice | 8 | F0, Jitter, Shimmer, HNR, Rate, VOT |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2024 | Initial comprehensive catalog |

---

*This document is auto-generated and maintained alongside the codebase. For the latest version, regenerate from source.*
