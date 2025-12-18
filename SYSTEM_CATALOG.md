# Delta Predictive Biosensing (DPB) System Catalog

> **Authoritative reference for the DPB neuromorphic biosignal processing framework**
> Version: 0.3.0 | Edition: Rust 2024 | License: MIT OR Apache-2.0

---

## Table of Contents

1. [Framework Overview](#1-framework-overview)
2. [Capability → Implementation](#2-capability--implementation)
3. [Implementation → Capabilities](#3-implementation--capabilities)
4. [Dependency Graph](#4-dependency-graph)
5. [Gap Analysis](#5-gap-analysis)
6. [Quick Reference](#6-quick-reference)

---

## 1. Framework Overview

The Delta Predictive Biosensing Framework is a comprehensive neuromorphic signal processing system for biosignal analysis. It provides:

- **Event-based encoding** of continuous biosignals into spike trains
- **Spiking Neural Networks (SNNs)** for efficient, low-power inference
- **Synthetic data generation** with clinical ground truth
- **Normative databases** for population-based comparison
- **Multi-modal fusion** for integrated assessments

### Crate Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATIONS                                    │
├─────────────────┬─────────────────┬─────────────────────────────────────────┤
│   dpb-python    │    dpb-ffi      │              dpb-bench                  │
│  (PyO3 bindings)│ (C/FFI bindings)│         (Benchmarking suite)            │
├─────────────────┴─────────────────┴─────────────────────────────────────────┤
│                           NEURAL NETWORKS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                              dpb-snn                                         │
│     (Layers, Architectures, Training, 48 Decoders, Fusion, Baselines)       │
├────────────────────────────────┬────────────────────────────────────────────┤
│          dpb-neurons           │              dpb-encoders                  │
│    (19 neuron models, GPU)     │   (103+ encoders, 85+ templates)          │
├────────────────────────────────┴────────────────────────────────────────────┤
│                           DATA & VALIDATION                                  │
├─────────────────────────────────┬───────────────────────────────────────────┤
│           dpb-synth             │              dpb-norms                    │
│   (210+ synthetic generators)   │  (Normative DB, 60 metrics populated)    │
├─────────────────────────────────┴───────────────────────────────────────────┤
│                             FOUNDATION                                       │
├─────────────────────────────────┬───────────────────────────────────────────┤
│           dpb-core              │           dpb-cognitive                   │
│  (Types, Traits, Algorithms,    │     (Cognitive task paradigms)            │
│   Signal Processing, Metrics)   │                                           │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

---

## 2. Capability → Implementation

### 2.1 Signal Acquisition & Preprocessing

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| Signal representation | dpb-core | `types` | `TimeSeries`, `SignalBuffer`, `Modality` |
| IIR filtering (Butterworth) | dpb-core | `signal::filter` | `IirFilter`, `FilterType` |
| Median filtering | dpb-core | `signal::filter` | `median_filter()` |
| FFT/spectral analysis | dpb-core | `signal::fft` | `FftProcessor`, `Stft`, `WindowType` |
| Signal normalization | dpb-core | `signal` | `normalize()`, `NormalizationType` |
| Resampling | dpb-core | `signal::resample` | `downsample()`, `upsample()`, `resample_linear()` |
| Peak/valley detection | dpb-core | `signal` | `find_peaks()`, `find_valleys()` |
| Envelope extraction | dpb-core | `signal` | `envelope()`, `rms()` |
| Artifact detection | dpb-core | `signal::eeg::artifacts` | `detect_artifacts()`, `ArtifactSegment` |

### 2.2 Event-Based Encoding

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| **Base Encoders** |
| Level crossing encoding | dpb-encoders | `base` | `LevelCrossingEncoder` |
| Template deviation encoding | dpb-encoders | `base` | `TemplateDeviationEncoder` |
| Derivative encoding | dpb-encoders | `base` | `DerivativeEncoder` |
| **Contact Biosignals** |
| ECG R-peak encoding | dpb-encoders | `contact::ecg` | `EcgRPeakEncoder`, `EcgMorphologyEncoder` |
| PPG pulse encoding | dpb-encoders | `contact::ppg` | `PpgPulseEncoder`, `PpgAmplitudeEncoder` |
| EDA response encoding | dpb-encoders | `contact::eda` | `EdaScrEncoder`, `EdaTonicEncoder` |
| EMG burst encoding | dpb-encoders | `contact::emg` | `EmgBurstEncoder`, `EmgFatigueEncoder` |
| **Movement/Pose** |
| Gait phase encoding | dpb-encoders | `pose` | `HeelStrikeEncoder`, `ToeOffEncoder`, `GaitPhaseEncoder` |
| Hand movement encoding | dpb-encoders | `hand` | `TapOnsetEncoder`, `TapApertureEncoder` |
| Eye movement encoding | dpb-encoders | `eye` | `SaccadeOnsetEncoder`, `FixationStabilityEncoder` |
| Voice encoding | dpb-encoders | `voice` | `F0Encoder`, `JitterEncoder`, `FormantEncoder` |
| **Balance Encoders** |
| CoP sway encoding | dpb-encoders | `balance` | `CopSwayEncoder`, `SwayAreaTemplate` |
| CoP velocity encoding | dpb-encoders | `balance` | `CopVelocityEncoder`, `SwayVelocityTemplate` |
| Stability limits encoding | dpb-encoders | `balance` | `StabilityLimitEncoder`, `StabilityLimitTemplate` |
| **Force Encoders** |
| GRF phase encoding | dpb-encoders | `force` | `GrfPhaseEncoder`, `PeakGrfTemplate` |
| Grip onset encoding | dpb-encoders | `force` | `GripOnsetEncoder`, `GripStrengthTemplate` |
| RFD encoding | dpb-encoders | `force` | `RfdEncoder`, `RfdTemplate` |
| **Vestibular Encoders** |
| VOR gain encoding | dpb-encoders | `vestibular` | `VorGainEncoder`, `VorGainTemplate` |
| Nystagmus encoding | dpb-encoders | `vestibular` | `NystagmusEncoder`, `NystagmusSPVTemplate` |
| Caloric test encoding | dpb-encoders | `vestibular` | `CaloricEncoder`, `CaloricAsymmetryTemplate` |
| **Pain Encoders** |
| Pain threshold encoding | dpb-encoders | `pain` | `PainThresholdEncoder`, `PressurePainThresholdTemplate` |
| Temporal summation encoding | dpb-encoders | `pain` | `TemporalSummationEncoder`, `WindUpRatioTemplate` |
| CPM encoding | dpb-encoders | `pain` | `CpmEncoder` |
| **Cardiopulmonary Encoders** |
| HRV encoding | dpb-encoders | `cardiopulmonary` | `HrvEncoder`, `RmssdTemplate` |
| Respiratory phase encoding | dpb-encoders | `cardiopulmonary` | `RespiratoryPhaseEncoder`, `RespiratoryRateTemplate` |
| RSA encoding | dpb-encoders | `cardiopulmonary` | `RsaEncoder`, `RsaTemplate` |
| **Cognitive Encoders** |
| Reaction time encoding | dpb-encoders | `cognitive` | `ReactionTimeEncoder`, `SimpleRtTemplate` |
| Error encoding | dpb-encoders | `cognitive` | `ErrorEncoder`, `AccuracyTemplate` |
| Lapse encoding | dpb-encoders | `cognitive` | `LapseEncoder`, `LapseRateTemplate` |
| **EEG Encoders** |
| Alpha band power | dpb-encoders | `eeg` | `AlphaBandEncoder`, `AlphaPowerTemplate` |
| Beta band power | dpb-encoders | `eeg` | `BetaBandEncoder`, `BetaPowerTemplate` |
| Theta band power | dpb-encoders | `eeg` | `ThetaBandEncoder`, `ThetaPowerTemplate` |
| Gamma band power | dpb-encoders | `eeg` | `GammaBandEncoder`, `GammaPowerTemplate` |
| Delta band power | dpb-encoders | `eeg` | `DeltaBandEncoder`, `DeltaPowerTemplate` |
| ERP detection | dpb-encoders | `eeg` | `ErpEncoder` (P300, N100) |
| Sleep spindle detection | dpb-encoders | `eeg` | `SpindleEncoder` |
| Artifact detection | dpb-encoders | `eeg` | `ArtifactEncoder` (blink, muscle, movement) |
| **Templates** |
| Population templates | dpb-encoders | `templates` | `TemplateRegistry` (85+ templates) |

### 2.3 Spiking Neural Networks

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| **Neuron Models (19)** |
| Leaky integrate-and-fire | dpb-neurons | `lif` | `LifNeuron`, `AlifNeuron`, `ClifNeuron`, `ElifNeuron`, `QlifNeuron`, `GlifNeuron` |
| Izhikevich | dpb-neurons | `izhikevich` | `IzhikevichNeuron` |
| Adaptive exponential | dpb-neurons | `adex` | `AdexNeuron` |
| Hodgkin-Huxley | dpb-neurons | `hodgkin_huxley` | `HodgkinHuxleyNeuron`, `FitzHughNagumoNeuron`, `MorrisLecarNeuron` |
| Spike response model | dpb-neurons | `srm` | `SrmNeuron` |
| Hardware-optimized | dpb-neurons | `hardware` | `XyloLifNeuron`, `PulsarLifNeuron`, `QuantizedLifNeuron` |
| GPU-accelerated | dpb-neurons | `gpu` | `GpuLifNeuron`, `GpuAlifNeuron`, `GpuIzhikevichNeuron` |
| **Surrogate Gradients (6)** |
| Gradient functions | dpb-neurons | `surrogate` | `FastSigmoid`, `Arctan`, `Triangular`, `SuperSpike`, `MultiGaussian`, `STE` |
| **Network Layers** |
| Linear layers | dpb-snn | `layers::linear` | `SpikingLinear` |
| Convolutional | dpb-snn | `layers::conv` | `SpikingConv1d`, `SpikingConv2d` |
| Pooling | dpb-snn | `layers::pool` | `SpikingSumPool2d`, `SpikingMaxPool2d` |
| Recurrent | dpb-snn | `layers::recurrent` | `SpikingRNN`, `SpikingLSTM` |
| Attention | dpb-snn | `layers::attention` | `SpikingAttention` |
| **Architectures (5)** |
| Feedforward | dpb-snn | `architectures` | `FeedforwardSNN` |
| Convolutional | dpb-snn | `architectures` | `ConvolutionalSNN` |
| Recurrent | dpb-snn | `architectures` | `RecurrentSNN` |
| Graph neural | dpb-snn | `architectures` | `SpikingGCN` |
| Transformer | dpb-snn | `architectures` | `SpikingTransformer` |

### 2.4 Clinical Decoders (48 total)

| Category | Decoder | Output | Module |
|----------|---------|--------|--------|
| **Rate-Based (9)** |
| Spike rate | `SpikeRateDecoder` | Hz | `decoders::rate` |
| First spike | `FirstSpikeDecoder` | Time-to-first-spike | `decoders::rate` |
| Population | `PopulationDecoder` | Vector code | `decoders::rate` |
| Windowed | `WindowedRateDecoder` | Time-windowed rate | `decoders::rate` |
| **Temporal (7)** |
| Pattern | `TemporalPatternDecoder` | Pattern match | `decoders::temporal` |
| Latency | `LatencyDecoder` | Response latency | `decoders::temporal` |
| ISI | `ISIDecoder` | Inter-spike intervals | `decoders::temporal` |
| Burst | `BurstDecoder` | Burst detection | `decoders::temporal` |
| **Clinical Scales (23)** |
| UPDRS | `UpdrsDecoder` | 0-199 (MDS-UPDRS) | `decoders::clinical` |
| UPDRS-III | `UpdrsMotorDecoder` | 0-132 (motor) | `decoders::clinical` |
| UPDRS Tremor | `UpdrsTremorDecoder` | Tremor subscale | `decoders::clinical` |
| UPDRS Gait | `UpdrsGaitDecoder` | Gait/posture subscale | `decoders::clinical` |
| Hoehn-Yahr | `HoehnYahrDecoder` | 0-5 stage | `decoders::clinical` |
| TUG | `TugDecoder` | Seconds | `decoders::clinical` |
| Berg Balance | `BergBalanceDecoder` | 0-56 | `decoders::clinical` |
| Tinetti | `TinettiDecoder` | 0-28 | `decoders::clinical` |
| MiniBEST | `MiniBESTDecoder` | 0-32 | `decoders::clinical` |
| MoCA | `MocaDecoder` | 0-30 | `decoders::clinical` |
| VoiceHD | `VoiceHdDecoder` | Voice disorder scale | `decoders::clinical` |
| PDQ-39 | `Pdq39Decoder` | Quality of life | `decoders::clinical` |
| SEADL | `SeadlDecoder` | ADL scale | `decoders::clinical` |
| VAS Pain | `VasDecoder` | 0-100mm | `decoders::clinical` |
| NRS Pain | `NrsDecoder` | 0-10 | `decoders::clinical` |
| QST Phenotype | `QstPhenotypeDecoder` | Sensory phenotype | `decoders::clinical` |
| VOR Gain | `VorGainDecoder` | Gain ratio | `decoders::clinical` |
| Canal Paresis | `CanalParesisDecoder` | % asymmetry | `decoders::clinical` |
| BPPV | `BppvDecoder` | Probability | `decoders::clinical` |
| **Regression (10)** |
| Heart rate | `HeartRateDecoder` | BPM | `decoders::regression` |
| HRV | `HrvDecoder` | RMSSD (ms) | `decoders::regression` |
| Tremor freq | `TremorFrequencyDecoder` | Hz | `decoders::regression` |
| Tremor amp | `TremorAmplitudeDecoder` | mm | `decoders::regression` |
| Gait velocity | `GaitVelocityDecoder` | m/s | `decoders::regression` |
| Stride time | `StrideTimeDecoder` | seconds | `decoders::regression` |
| Tapping freq | `TappingFrequencyDecoder` | Hz | `decoders::regression` |
| Reaction time | `ReactionTimeDecoder` | ms | `decoders::regression` |
| Speech rate | `SpeechRateDecoder` | syllables/s | `decoders::regression` |
| **Classification (10)** |
| Binary | `BinaryClassDecoder` | 0/1 | `decoders::classification` |
| Multi-class | `MultiClassDecoder` | Class label | `decoders::classification` |
| Tremor type | `TremorTypeDecoder` | Rest/postural/kinetic | `decoders::classification` |
| Gait phase | `GaitPhaseDecoder` | Stance/swing | `decoders::classification` |
| Sleep stage | `SleepStageDecoder` | W/N1/N2/N3/REM | `decoders::classification` |
| Activity | `ActivityDecoder` | Activity type | `decoders::classification` |

### 2.5 Multi-Modal Fusion

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| Early fusion | dpb-snn | `fusion::early` | `EarlyFusionSNN` |
| Late fusion | dpb-snn | `fusion::late` | `LateFusionSNN` |
| Cross-modal attention | dpb-snn | `fusion::attention` | `CrossModalAttentionSNN` |
| Hierarchical fusion | dpb-snn | `fusion::hierarchical` | `HierarchicalFusionSNN` |
| Temporal alignment | dpb-snn | `fusion::temporal` | `TemporalAlignmentSNN` |
| Gated fusion | dpb-snn | `fusion::gated` | `GatedFusionSNN` |
| Cognitive-motor | dpb-snn | `fusion::cognitive_motor` | `CognitiveMotorFusionSNN` |

### 2.6 Synthetic Data Generation

| Domain | Crate | Module | Generators |
|--------|-------|--------|------------|
| **Contact Biosignals** |
| ECG | dpb-synth | `contact::ecg` | `EcgMorphologyGenerator`, `EcgArrhythmiaGenerator` |
| PPG | dpb-synth | `contact::ppg` | `PpgPulseGenerator`, `PpgRespiratoryGenerator` |
| EDA | dpb-synth | `contact::eda` | `EdaTonicGenerator`, `EdaScrGenerator` |
| EMG | dpb-synth | `contact::emg` | `EmgBurstGenerator`, `EmgFatigueGenerator` |
| Tremor | dpb-synth | `contact::tremor` | `ParkinsonianTremorGenerator`, `PhysiologicalTremorGenerator` |
| **Movement** |
| Gait | dpb-synth | `pose::gait` | `GaitCycleGenerator`, `PathologicalGaitGenerator` |
| Hand | dpb-synth | `hand` | `TappingGenerator`, `BradykineticTappingGenerator` |
| Eye | dpb-synth | `eye` | `SaccadeGenerator`, `FixationGenerator`, `PursuitGenerator` |
| **Voice** |
| Phonation | dpb-synth | `voice::phonation` | `VoiceTremorGenerator`, `HypophoniaGenerator` |
| Articulation | dpb-synth | `voice::articulation` | `DysarthriaGenerator` |
| **Neural** |
| EEG | dpb-synth | `neural::eeg` | `EegGenerator`, `EegArtifactGenerator` |
| ERP | dpb-synth | `neural::erp` | `P300Generator`, `N100Generator` |
| Sleep | dpb-synth | `neural::sleep` | `SpindleGenerator`, `KComplexGenerator` |
| **Balance/Force** |
| CoP | dpb-synth | `balance::cop` | `CopSwayGenerator`, `PathologicalSwayGenerator` |
| GRF | dpb-synth | `force::grf` | `GrfGenerator`, `PathologicalGrfGenerator` |
| Grip | dpb-synth | `force::grip` | `GripStrengthGenerator` |
| **Vestibular** |
| VOR | dpb-synth | `vestibular::vor` | `VorGainGenerator`, `VorAsymmetryGenerator` |
| Nystagmus | dpb-synth | `vestibular::nystagmus` | `NystagmusGenerator`, `BppvGenerator` |
| Caloric | dpb-synth | `vestibular::caloric` | `CaloricResponseGenerator` |
| **Pain** |
| QST | dpb-synth | `pain` | `PainThresholdGenerator`, `CpmGenerator` |
| **Cardiopulmonary** |
| HRV | dpb-synth | `cardiopulmonary` | `HrvGenerator` |
| Respiratory | dpb-synth | `cardiopulmonary` | `RespiratoryGenerator` |
| **Cognitive** |
| Tasks | dpb-synth | `cognitive` | `ReactionTimeGenerator`, `NBackGenerator` |
| **Multi-modal** |
| Parkinson's | dpb-synth | `multimodal` | `FullPDSimulator`, `HealthyAgingSimulator` |
| Coupling | dpb-synth | `multimodal` | `HandVoiceTremorCouplingGenerator`, `GaitSpeechRateCouplingGenerator` |
| **Streaming (Real-time)** |
| ECG | dpb-synth | `streaming` | `StreamingEcg`, `StreamingEcgParams` |
| EEG | dpb-synth | `streaming` | `StreamingEeg`, `StreamingEegParams` |
| PPG | dpb-synth | `streaming` | `StreamingPpg` |
| EMG | dpb-synth | `streaming` | `StreamingEmg` |
| EDA | dpb-synth | `streaming` | `StreamingEda` |
| Respiratory | dpb-synth | `streaming` | `StreamingRespiratory` |
| Pose | dpb-synth | `streaming` | `StreamingPose`, `StreamingClinicalPose` |
| Hand | dpb-synth | `streaming` | `StreamingHand`, `StreamingClinicalHand` |
| Gaze | dpb-synth | `streaming` | `StreamingGaze` |
| Tremor | dpb-synth | `streaming` | `StreamingTremor` |

### 2.7 Normative Comparison

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| Demographics | dpb-norms | `demographics` | `Demographics`, `Sex`, `AgeGroup`, `Ethnicity` |
| Metric types | dpb-norms | `metrics` | `MetricType` (60 types), `MetricDomain`, `MetricDirection` |
| Normative stats | dpb-norms | `lib` | `NormativeStats`, `NormativeComparison` |
| Database | dpb-norms | `database` | `NormativeDatabase`, `NormativeEntry`, `NormativeTable` |
| Impairment levels | dpb-norms | `lib` | `ImpairmentLevel` (Normal → Severe) |

#### 2.7.1 Populated Normative Metrics (60/60 = 100%)

| Domain | Metrics |
|--------|---------|
| **Cognitive (15)** | SimpleReactionTime, ChoiceReactionTime, ReactionTimeVariability, NBackAccuracy, NBackDPrime, CptOmissions, CptCommissions, StroopInterference, TrailMakingA, TrailMakingB, TrailMakingBMinusA, DigitSpanForward, DigitSpanBackward, VerbalFluency, MocaTotal |
| **Motor (13)** | GaitVelocity, StrideLength, StrideTimeVariability, DoubleSupportTime, Cadence, TimedUpAndGo, GripStrength, RateOfForceDevelopment, TappingFrequency, TappingVariability, TremorAmplitude, TremorFrequency, UpdrsMotor |
| **Balance (8)** | SwayArea, SwayPathLength, SwayVelocity, RombergQuotient, BergBalanceScale, LosReactionTime, LosMaxExcursion, LosDirectionalControl |
| **Physiological (10)** | HeartRate, HrvSdnn, HrvRmssd, HrvPnn50, HrvLfHf, RespiratoryRate, SpO2, BpSystolic, BpDiastolic |
| **Sensory (5)** | PressurePainThreshold, PainTolerance, CpmEffect, VibrationThreshold, JointPositionError |
| **Sleep (6)** | TotalSleepTime, SleepEfficiency, SleepOnsetLatency, WakeAfterSleepOnset, RemPercent, DeepSleepPercent |
| **Composite (4)** | CognitiveComposite, MotorComposite, GlobalComposite, FrailtyIndex |

### 2.8 Cognitive Assessment

| Task Type | Crate | Module | Key Types |
|-----------|-------|--------|-----------|
| Reaction time | dpb-cognitive | `reaction_time` | `SimpleReactionTime`, `ChoiceReactionTime` |
| Working memory | dpb-cognitive | `working_memory` | `NBackTask` |
| Sustained attention | dpb-cognitive | `attention` | `ContinuousPerformanceTest` |
| Selective attention | dpb-cognitive | `attention` | `StroopTask` |
| Inhibition | dpb-cognitive | `executive` | `GoNoGoTask` |
| Interference | dpb-cognitive | `executive` | `FlankerTask` |
| Set shifting | dpb-cognitive | `executive` | `WisconsinCardSort` |
| ADHD screening | dpb-cognitive | `adhd` | `QbTest`, `AdhdEyeTracking` |

---

## 3. Implementation → Capabilities

### 3.1 dpb-core

**Purpose**: Foundation crate with core types, traits, and algorithms

| Module | Capabilities |
|--------|--------------|
| `types` | `SpikeEvent`, `SpikeTrain`, `SignalBuffer`, `TimeSeries`, `Context`, `Modality` |
| `traits` | `Signal`, `EventEncoder`, `PopulationTemplate`, `MembraneDynamics`, `SynapticModel`, `SpikingLayer`, `SNNNetwork`, `Dataset`, `Optimizer`, `LossFunction` |
| `tensor` | `SpikeTensor` - batched spike operations with GPU acceleration |
| `signal/filter` | Butterworth IIR, median filter, bandpass, highpass, lowpass |
| `signal/fft` | FFT, STFT, power spectral density, band power extraction |
| `signal/resample` | Upsampling, downsampling, linear interpolation |
| `signal/eeg` | Band power (delta/theta/alpha/beta/gamma), artifact detection, ERP analysis |
| `gpu` | WGPU-based GPU compute, buffer management, kernel execution |
| `metrics` | Classification (accuracy, F1, ROC-AUC), regression (MSE, MAE, R²), clinical (sensitivity, specificity), efficiency (spike rate, energy) |
| `validation` | Reliability (ICC, Cronbach's alpha), validity (ROC curves, MDC), convergence analysis |
| `power` | Neuromorphic power estimation, synaptic operations, memory access |
| `biomechanics` | GRF analysis, grip dynamics, RFD calculation, balance metrics |
| `cardiopulmonary` | VO2 kinetics, ventilatory thresholds, gas exchange |
| `pain` | QST analysis, pain scales, autonomic response |
| `vestibular` | VOR analysis, posturography, canal function |
| `sleep` | Sleep staging, spindle detection, sleep efficiency |
| `viz` | Signal plotting, spike rasters, network visualization, training curves |

### 3.2 dpb-encoders

**Purpose**: Convert continuous biosignals to spike trains

| Module | Capabilities |
|--------|--------------|
| `base` | Level crossing, template deviation, derivative, discrete event encoders |
| `contact/ecg` | R-peak detection, morphology analysis, ST-segment, HRV features |
| `contact/ppg` | Pulse detection, amplitude, transit time, respiratory modulation |
| `contact/eda` | SCR detection, tonic level, phasic response |
| `contact/emg` | Burst detection, amplitude envelope, fatigue tracking |
| `pose/gait` | Heel strike, toe-off, gait phase, stride detection, asymmetry |
| `hand/tapping` | Tap onset, aperture, frequency, decrement detection |
| `eye/saccade` | Saccade onset, main sequence, latency |
| `eye/fixation` | Fixation stability, microsaccades |
| `eye/pupil` | Pupil response, light reflex |
| `voice/phonation` | F0 tracking, jitter, shimmer, HNR |
| `voice/articulation` | Formant tracking, vowel space |
| `voice/prosody` | Speech rate, pause detection, intonation |
| `balance` | CoP sway, velocity, stability limits |
| `force` | GRF phases, grip onset, RFD |
| `vestibular` | VOR gain, nystagmus SPV, caloric asymmetry |
| `pain` | Pain threshold, temporal summation, CPM effect |
| `cardiopulmonary` | HRV features, respiratory phase, RSA |
| `cognitive` | Reaction time, error detection, lapse detection |
| `eeg` | Alpha/beta/theta/gamma/delta power, ERP (P300/N100), spindles, artifacts |
| `templates` | 85+ population templates with age/sex stratification |

### 3.3 dpb-neurons

**Purpose**: Spiking neuron models with GPU support

| Module | Capabilities |
|--------|--------------|
| `lif` | 7 LIF variants: IF, LIF, CLIF, ALIF, ELIF, QLIF, GLIF |
| `izhikevich` | Phenomenological model with various firing patterns |
| `adex` | Adaptive exponential integrate-and-fire |
| `hodgkin_huxley` | Conductance-based models: HH, FitzHugh-Nagumo, Morris-Lecar |
| `srm` | Spike response model with adaptation |
| `stochastic` | Stochastic LIF with noise |
| `recurrent` | Self-connected neurons |
| `hardware` | Xylo, Pulsar, Quantized LIF for hardware deployment |
| `gpu` | GPU-accelerated LIF, ALIF, Izhikevich |
| `surrogate` | 6 surrogate gradients for backpropagation |
| `batch` | Batch processing layers, population statistics |

### 3.4 dpb-snn

**Purpose**: Complete SNN training, inference, and deployment

| Module | Capabilities |
|--------|--------------|
| `layers` | Linear, Conv1d/2d, Pooling, RNN, LSTM, Attention layers |
| `architectures` | Feedforward, Convolutional, Recurrent, GCN, Transformer SNNs |
| `training` | BPTT, OTTT, SLTT; CrossEntropy, SpikeCount, SpikeTiming losses |
| `decoders/rate` | 9 rate-based decoders |
| `decoders/temporal` | 7 temporal pattern decoders |
| `decoders/clinical` | 23 clinical scale decoders (UPDRS, Berg, MoCA, VAS, etc.) |
| `decoders/regression` | 10 continuous value decoders |
| `decoders/classification` | 10 classification decoders |
| `fusion` | 8 multi-modal fusion strategies |
| `baselines` | 44 ANN architectures for comparison |
| `conversion` | ANN-to-SNN conversion with threshold balancing |
| `analysis` | 25+ training analyzers (convergence, gradients, overfitting) |
| `export` | Hardware export capabilities |

### 3.5 dpb-synth

**Purpose**: Synthetic biosignal generation with ground truth

| Module | Capabilities |
|--------|--------------|
| `contact` | ECG, PPG, EDA, EMG, tremor generators |
| `pose` | Gait cycle, pathological gait, variability generators |
| `hand` | Tapping, tremor, movement generators |
| `eye` | Saccade, fixation, pupil, pursuit generators |
| `voice` | Phonation, articulation, prosody, pathological speech |
| `neural` | EEG, ERP, sleep microstructure generators |
| `force` | GRF, grip strength, RFD generators |
| `balance` | CoP sway, perturbation, sensory integration |
| `vestibular` | VOR, nystagmus, caloric response generators |
| `pain` | Pain threshold, CPM, temporal summation |
| `cardiopulmonary` | HRV, respiratory pattern generators |
| `cognitive` | Reaction time, cognitive task generators |
| `multimodal` | Full disease simulators (Parkinson's, healthy aging), cross-modal coupling |
| `streaming` | Real-time streaming generators (ECG, EEG, PPG, EMG, EDA, Pose, Hand, Gaze, Tremor) |
| `media` | Integration with Blender, MediaPipe, OpenSim, MuJoCo, etc. |
| `level3` | Complex clinical simulations with video/audio |

### 3.6 dpb-norms

**Purpose**: Population-based normative comparison

| Module | Capabilities |
|--------|--------------|
| `demographics` | Age, sex, ethnicity, handedness, education filtering |
| `metrics` | 60 metric types across 7 domains with directions and units |
| `database` | Normative lookup with age/sex stratification |
| - | Percentile calculation, z-score computation |
| - | Reference ranges (5th-95th percentile) |
| - | Impairment classification (Normal → Severe) |
| - | MDC (minimal detectable change) calculation |
| - | ICC-based reliability metrics |

### 3.7 dpb-cognitive

**Purpose**: Computerized cognitive assessment tasks

| Module | Capabilities |
|--------|--------------|
| `reaction_time` | Simple RT, choice RT with accuracy and variability |
| `working_memory` | N-back task with d-prime calculation |
| `attention` | CPT with hit/FA rates; Stroop with interference |
| `executive` | Go/No-Go with inhibition; Flanker with congruity; WCST with perseveration |
| `adhd` | QbTest integration, ADHD-specific eye tracking metrics |

---

## 4. Dependency Graph

### 4.1 Crate Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                         APPLICATIONS                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │dpb-python│  │ dpb-ffi  │  │dpb-bench │                       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
│       │             │             │                              │
│       └──────┬──────┴─────────────┘                              │
│              │                                                   │
│              ▼                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                         dpb-snn                            │  │
│  │  (Layers, Architectures, Training, Decoders, Fusion)       │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │                                       │
│         ┌────────────────┼────────────────┐                     │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │dpb-encoders│  │dpb-neurons │  │ dpb-synth  │                │
│  │(103+ enc.) │  │(19 models) │  │(210+ gen.) │                │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘                │
│         │               │               │                       │
│         └───────────────┼───────────────┘                       │
│                         │                                        │
│                         ▼                                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                        dpb-core                            │  │
│  │  (Types, Traits, Signal Processing, GPU, Metrics, Viz)     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐                         │
│  │   dpb-norms    │  │ dpb-cognitive  │  (Standalone modules)   │
│  │ (60 metrics)   │  │  (8+ tasks)    │                         │
│  └────────────────┘  └────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Signal Processing Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Raw Signal │───▶│   Preprocess │───▶│   Encoder    │───▶│  SpikeTrain  │
│   (analog)   │    │  (filter/FFT)│    │ (103+ types) │    │  (events)    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│   Clinical   │◀───│   Decoder    │◀───│     SNN      │◀─────────┘
│    Score     │    │  (48 types)  │    │ (5 arch.)    │
└──────────────┘    └──────────────┘    └──────────────┘
       │
       ▼
┌──────────────┐
│  Normative   │
│  Comparison  │
└──────────────┘
```

### 4.3 External Dependencies

| Category | Libraries |
|----------|-----------|
| **Numerics** | ndarray, nalgebra, num-complex, statrs |
| **GPU** | wgpu, bytemuck |
| **FFT** | rustfft |
| **Serialization** | serde, serde_json, bincode |
| **Async** | tokio, rayon |
| **Random** | rand, rand_distr |
| **Error Handling** | thiserror, anyhow |
| **Logging** | tracing, tracing-subscriber |
| **Python** | pyo3, numpy |
| **Testing** | approx, criterion |

---

## 5. Gap Analysis

### 5.1 Coverage Summary

| Domain | Generators | Encoders | Analysis | Decoders | Norms |
|--------|:----------:|:--------:|:--------:|:--------:|:-----:|
| **ECG/Cardiac** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PPG** | ✓ | ✓ | ○ | ✓ | ○ |
| **EDA** | ✓ | ✓ | ○ | ○ | ○ |
| **EMG** | ✓ | ✓ | ○ | ✓ | ○ |
| **EEG** | ✓ | ✓ | ✓ | ✓ | ○ |
| **Gait/Pose** | ✓ | ✓ | ○ | ✓ | ✓ |
| **Hand/Tremor** | ✓ | ✓ | ○ | ✓ | ✓ |
| **Eye Tracking** | ✓ | ✓ | ○ | ✓ | ○ |
| **Voice** | ✓ | ✓ | ○ | ✓ | ○ |
| **Force (GRF/Grip/RFD)** | ✓ | ✓ | ✓ | ○ | ✓ |
| **Balance (CoP)** | ✓ | ✓ | ○ | ✓ | ✓ |
| **Vestibular (VOR)** | ✓ | ✓ | ✓ | ✓ | ○ |
| **Pain (QST)** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Cardiopulmonary** | ✓ | ✓ | ✓ | ○ | ✓ |
| **Cognitive** | ✓ | ✓ | ✓ | ○ | ✓ |

**Legend:** ✓ = Complete | ○ = Partial/Stub | ✗ = Missing

### 5.2 Completeness Metrics

| Metric | Count | Target | Coverage |
|--------|-------|--------|----------|
| Neuron models | 19 | 20 | **95%** |
| Event encoders | 103+ | 110 | **94%** |
| Population templates | 85+ | 90 | **94%** |
| Synthetic generators | 210+ | 220 | **95%** |
| SNN decoders | 48 | 50 | **96%** |
| ANN baselines | 44 | 50 | **88%** |
| Normative metrics | 60 | 60 | **100%** |
| Streaming generators | 12 | 15 | **80%** |

### 5.3 Remaining Gaps

#### Medium Priority

| Gap | Domain | Impact | Recommended Implementation |
|-----|--------|--------|---------------------------|
| **Multi-modal norms** | Norms | Single-modality only | Add cross-modal normative comparisons |
| **Fatigue detection** | Core | No fatigue-specific analysis | Add `FatigueDetector` for EMG/force/cognitive |
| **Seizure detection** | Core | EEG seizure detection stub | Implement full seizure detection algorithm |
| **Respiratory analysis** | Core | Limited respiratory analysis | Add breath detection, apnea detection |

#### Low Priority

| Gap | Domain | Impact | Recommended Implementation |
|-----|--------|--------|---------------------------|
| **Additional pathologies** | Synth | Some conditions not modeled | Add ALS, MS, stroke-specific patterns |
| **Longitudinal modeling** | Synth | No disease progression | Add progressive disease models |
| **Medication effects** | Synth | No pharmacological modeling | Add medication response generators |
| **Hardware export** | SNN | Limited deployment targets | Add more neuromorphic hardware backends |
| **Visualization tools** | Core | Basic plotting only | Add interactive visualization |

---

## 6. Quick Reference

### 6.1 Common Usage Patterns

```rust
// 1. Encode a signal to spikes
use dpb_encoders::prelude::*;
let encoder = LevelCrossingEncoder::new();
let config = LevelCrossingConfig::default();
let spikes = encoder.encode(&signal, &config)?;

// 2. Create a simple SNN
use dpb_snn::prelude::*;
let network = FeedforwardSNN::new(&[100, 64, 32, 10]);
let output = network.forward(&spikes)?;

// 3. Decode to clinical score
use dpb_snn::decoders::clinical::*;
let decoder = UpdrsMotorDecoder::new();
let score = decoder.decode(&output)?;

// 4. Compare to normative data
use dpb_norms::prelude::*;
let db = NormativeDatabase::with_defaults();
let comparison = db.compare(MetricType::GaitVelocity, 0.95, &demographics)?;

// 5. Generate synthetic EEG with streaming
use dpb_synth::streaming::*;
let generator = StreamingEeg;
let params = StreamingEegParams::eyes_closed_rest();
let mut state = generator.init_state(&params, 42);
let sample = generator.next_sample(&mut state);
```

### 6.2 Crate Feature Flags

| Crate | Feature | Description |
|-------|---------|-------------|
| dpb-core | `gpu` | Enable GPU acceleration |
| dpb-core | `viz` | Enable visualization |
| dpb-neurons | `gpu` | GPU neuron models |
| dpb-snn | `training` | Enable training features |
| dpb-synth | `media` | External tool integration |
| dpb-synth | `level3` | Complex simulations |

### 6.3 CLI Commands

```bash
# Run all tests
cargo test --workspace

# Run benchmarks
cargo bench -p dpb-bench

# Build documentation
cargo doc --workspace --no-deps --open

# Check code coverage
cargo tarpaulin --workspace
```

### 6.4 Key Traits Summary

| Trait | Purpose | Key Methods |
|-------|---------|-------------|
| `Signal` | Time-series data | `samples()`, `sample_rate()`, `channels()` |
| `EventEncoder` | Signal → Spikes | `encode()`, `name()` |
| `PopulationTemplate` | Clinical norms | `expected_value()`, `variance()` |
| `MembraneDynamics` | Neuron state | `membrane_potential()`, `rest_potential()` |
| `NeuronModel` | Spike generation | `update()`, `reset()`, `threshold()` |
| `SurrogateGradient` | Backprop | `forward()`, `backward()` |
| `Decoder` | Spikes → Output | `decode()`, `output_dim()` |
| `SyntheticGenerator` | Data generation | `generate()`, `validate_params()` |
| `StreamingGenerator` | Real-time | `next_sample()`, `init_state()` |

### 6.5 Module Quick Links

| Need | Crate | Module |
|------|-------|--------|
| Encode ECG to spikes | dpb-encoders | `contact::ecg` |
| Encode EEG band power | dpb-encoders | `eeg` |
| Generate synthetic gait | dpb-synth | `pose::gait` |
| Stream real-time EEG | dpb-synth | `streaming` |
| Build feedforward SNN | dpb-snn | `architectures` |
| Decode UPDRS score | dpb-snn | `decoders::clinical` |
| Compare to norms | dpb-norms | `database` |
| Run cognitive task | dpb-cognitive | `reaction_time` |

---

*Last updated: 2025-12-18 | Catalog version: 0.3.0*
