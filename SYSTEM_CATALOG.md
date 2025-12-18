# Delta Predictive Biosensing (DPB) System Catalog

> **Authoritative reference for the DPB neuromorphic biosignal processing framework**
> Version: 0.2.0 | Edition: Rust 2024 | License: MIT OR Apache-2.0

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
│     (Layers, Architectures, Training, 40 Decoders, Fusion, Baselines)       │
├────────────────────────────────┬────────────────────────────────────────────┤
│          dpb-neurons           │              dpb-encoders                  │
│    (19 neuron models, GPU)     │   (95+ encoders, 79+ templates)           │
├────────────────────────────────┴────────────────────────────────────────────┤
│                           DATA & VALIDATION                                  │
├─────────────────────────────────┬───────────────────────────────────────────┤
│           dpb-synth             │              dpb-norms                    │
│   (200+ synthetic generators)   │  (Normative DB, 40+ metrics populated)   │
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
| **Balance Encoders** *(NEW)* |
| CoP sway encoding | dpb-encoders | `balance` | `CopSwayEncoder`, `SwayAreaTemplate` |
| CoP velocity encoding | dpb-encoders | `balance` | `CopVelocityEncoder`, `SwayVelocityTemplate` |
| Stability limits encoding | dpb-encoders | `balance` | `StabilityLimitEncoder`, `StabilityLimitTemplate` |
| **Force Encoders** *(NEW)* |
| GRF phase encoding | dpb-encoders | `force` | `GrfPhaseEncoder`, `PeakGrfTemplate` |
| Grip onset encoding | dpb-encoders | `force` | `GripOnsetEncoder`, `GripStrengthTemplate` |
| RFD encoding | dpb-encoders | `force` | `RfdEncoder`, `RfdTemplate` |
| **Vestibular Encoders** *(NEW)* |
| VOR gain encoding | dpb-encoders | `vestibular` | `VorGainEncoder`, `VorGainTemplate` |
| Nystagmus encoding | dpb-encoders | `vestibular` | `NystagmusEncoder`, `NystagmusSPVTemplate` |
| Caloric test encoding | dpb-encoders | `vestibular` | `CaloricEncoder`, `CaloricAsymmetryTemplate` |
| **Pain Encoders** *(NEW)* |
| Pain threshold encoding | dpb-encoders | `pain` | `PainThresholdEncoder`, `PressurePainThresholdTemplate` |
| Temporal summation encoding | dpb-encoders | `pain` | `TemporalSummationEncoder`, `WindUpRatioTemplate` |
| CPM encoding | dpb-encoders | `pain` | `CpmEncoder` |
| **Cardiopulmonary Encoders** *(NEW)* |
| HRV encoding | dpb-encoders | `cardiopulmonary` | `HrvEncoder`, `RmssdTemplate` |
| Respiratory phase encoding | dpb-encoders | `cardiopulmonary` | `RespiratoryPhaseEncoder`, `RespiratoryRateTemplate` |
| RSA encoding | dpb-encoders | `cardiopulmonary` | `RsaEncoder`, `RsaTemplate` |
| **Cognitive Encoders** *(NEW)* |
| Reaction time encoding | dpb-encoders | `cognitive` | `ReactionTimeEncoder`, `SimpleRtTemplate` |
| Error encoding | dpb-encoders | `cognitive` | `ErrorEncoder`, `AccuracyTemplate` |
| Lapse encoding | dpb-encoders | `cognitive` | `LapseEncoder`, `LapseRateTemplate` |
| **EEG Encoders** *(NEW - v0.2.1)* |
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
| **Tensor Operations** |
| Spike tensor operations | dpb-snn | `tensor` | `SpikeTensor`, `SpikeRepresentation` |
| **Layers** |
| Linear spiking layers | dpb-snn | `layers` | `SpikingLinear` |
| Convolutional spiking layers | dpb-snn | `layers` | `SpikingConv1d`, `SpikingConv2d` |
| Pooling layers | dpb-snn | `layers` | `SpikingSumPool2d`, `SpikingMaxPool2d` |
| Recurrent spiking layers | dpb-snn | `layers` | `SpikingRNN`, `SpikingLSTM` |
| Attention mechanisms | dpb-snn | `layers` | `SpikingAttention` |
| **Architectures** |
| Feedforward architectures | dpb-snn | `architectures` | `FeedforwardSNN` |
| Convolutional architectures | dpb-snn | `architectures` | `ConvolutionalSNN` |
| Recurrent architectures | dpb-snn | `architectures` | `RecurrentSNN` |
| Graph neural networks | dpb-snn | `architectures` | `SpikingGCN` |
| Transformer architectures | dpb-snn | `architectures` | `SpikingTransformer` |
| **Training** |
| BPTT training | dpb-snn | `training` | `BPTT`, `SurrogateGradient` |
| Online training | dpb-snn | `training` | `OTTT`, `SLTT` |
| Loss functions | dpb-snn | `training` | `SpikingCrossEntropy`, `SpikeCountLoss`, `SpikeTimingLoss` |
| ANN→SNN conversion | dpb-snn | `conversion` | `ANNToSNNConverter`, `WeightNormalization` |
| **Decoders (40 types)** |
| Rate decoders | dpb-snn | `decoders::rate` | `RateDecoder`, `PopulationDecoder`, `SoftmaxDecoder` |
| Temporal decoders | dpb-snn | `decoders::temporal` | `LatencyDecoder`, `ISIDecoder`, `PhaseDecoder` |
| Motor decoders | dpb-snn | `decoders::clinical` | `UpdrsDecoder`, `BradykinesiaDecoder` |
| Balance decoders *(NEW)* | dpb-snn | `decoders::clinical` | `BergBalanceDecoder`, `TinettiDecoder`, `MiniBESTDecoder` |
| Pain decoders *(NEW)* | dpb-snn | `decoders::clinical` | `VasDecoder`, `NrsDecoder`, `QstPhenotypeDecoder` |
| Vestibular decoders *(NEW)* | dpb-snn | `decoders::clinical` | `VorGainDecoder`, `CanalParesisDecoder`, `BppvDecoder` |
| Regression decoders | dpb-snn | `decoders::regression` | `LinearRegDecoder`, `MLPRegDecoder` |
| **Fusion** |
| Multi-modal fusion | dpb-snn | `fusion` | `EarlyFusionSNN`, `LateFusionSNN`, `HierarchicalFusionSNN` |

### 2.4 Neuron Models

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| Integrate-and-fire neurons | dpb-neurons | `lif` | `IfNeuron`, `LifNeuron`, `ClifNeuron`, `AlifNeuron`, `ElifNeuron`, `QlifNeuron`, `GlifNeuron` |
| Phenomenological models | dpb-neurons | `izhikevich` | `IzhikevichNeuron` (with presets) |
| Adaptive exponential | dpb-neurons | `adex` | `AdExNeuron` |
| Calcium-based adaptation | dpb-neurons | `calcium` | `CalciumNeuron` |
| Biophysical models | dpb-neurons | `hodgkin_huxley` | `HodgkinHuxleyNeuron`, `FitzHughNagumoNeuron`, `MorrisLecarNeuron` |
| Spike response models | dpb-neurons | `srm` | `SrmNeuron` |
| Stochastic neurons | dpb-neurons | `stochastic` | `StochasticLifNeuron` |
| Hardware-optimized | dpb-neurons | `hardware` | `XyloLifNeuron`, `PulsarLifNeuron`, `QuantizedLifNeuron` |
| Surrogate gradients | dpb-neurons | `surrogate` | `FastSigmoid`, `Arctan`, `SuperSpike`, `MultiGaussian` |
| GPU acceleration | dpb-neurons | `gpu` | `GpuLifNeuron`, `GpuAlifNeuron`, `GpuIzhikevichNeuron` |

### 2.5 Synthetic Data Generation

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| **Contact Biosignals** |
| ECG synthesis | dpb-synth | `contact` | `StreamingEcg` (arrhythmias, HRV, RSA) |
| PPG synthesis | dpb-synth | `contact` | `StreamingPpg` |
| EDA synthesis | dpb-synth | `contact` | `StreamingEda` |
| EMG synthesis | dpb-synth | `contact` | `StreamingEmg` |
| Respiratory synthesis | dpb-synth | `contact` | `StreamingRespiratory` |
| Temperature synthesis | dpb-synth | `contact` | `StreamingThermal` |
| Tremor synthesis | dpb-synth | `contact` | `StreamingTremor` |
| **Biomechanics** |
| Ground reaction force | dpb-synth | `force` | `GrfGenerator` (walking, running, jumping) |
| Grip strength | dpb-synth | `force` | `GripGenerator` (MVC, fatigue, pathologies) |
| Rate of force development | dpb-synth | `force` | `RfdGenerator` (isometric, jumps) |
| Center of pressure | dpb-synth | `balance` | `CopGenerator` (quiet standing, LOS) |
| Perturbation responses | dpb-synth | `balance` | `PerturbationGenerator` |
| Sensory organization | dpb-synth | `balance` | `SensoryManipulationGenerator` (SOT, mCTSIB) |
| **Vestibular** |
| VOR responses | dpb-synth | `vestibular` | `VorGenerator` (sinusoidal, HIT) |
| Nystagmus patterns | dpb-synth | `vestibular` | `NystagmusGenerator` (BPPV, spontaneous) |
| Caloric test | dpb-synth | `vestibular` | `CaloricGenerator` (bithermal) |
| **Pain & Sensory** |
| Pain protocols | dpb-synth | `pain` | `PainGenerator` (QST, CPM, temporal summation) |
| **Cardiopulmonary** |
| HRV/respiratory | dpb-synth | `cardiopulmonary` | `CardiopulmonaryGenerator` (resting, exercise) |
| **Cognitive** |
| Reaction time tasks | dpb-synth | `cognitive` | `CognitiveGenerator` (RT, Flanker, Go/NoGo, N-back) |
| **Motion/Pose** |
| Gait patterns | dpb-synth | `pose` | `StreamingPose`, `StreamingClinicalPose` |
| Hand movements | dpb-synth | `hand` | `StreamingHand`, `StreamingClinicalHand` |
| **Eye Tracking** |
| Gaze patterns | dpb-synth | `eye` | `StreamingGaze` |
| **Voice** |
| Vowel production | dpb-synth | `voice` | `StreamingVowel`, `StreamingDdk` |
| **Neural** |
| EEG synthesis | dpb-synth | `neural` | EEG, ERP generators |

### 2.6 Clinical Analysis Algorithms

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| **EEG Analysis** |
| Frequency band power | dpb-core | `signal::eeg::bands` | `EegBands`, `BandPowers`, `theta_beta_ratio()` |
| ERP extraction | dpb-core | `signal::eeg::erp` | `ErpAnalyzer`, `ErpComponent` (P300, N400, MMN) |
| Sleep staging | dpb-core | `sleep::staging` | `SleepStager`, `SleepArchitecture` |
| Sleep microstructure | dpb-core | `sleep::features` | Spindle, K-complex, slow wave detection |
| **Pain Assessment** |
| QST analysis | dpb-core | `pain::qst` | `QstAnalyzer`, `SensoryPhenotype` |
| Pressure pain threshold | dpb-core | `pain::threshold` | `PressurePainThreshold`, `ConditionedPainModulation` |
| Pain autonomic response | dpb-core | `pain::autonomic` | `PainAutonomicAnalyzer` |
| Pain scales | dpb-core | `pain::scales` | VAS, NRS, McGill, BPI, NPSI |
| **Biomechanics** |
| Grip analysis | dpb-core | `biomechanics::force::grip` | `GripStrengthAnalyzer`, `GripMetrics` |
| GRF analysis | dpb-core | `biomechanics::force::grf` | `GroundReactionForceAnalyzer`, `GaitPhase` |
| RFD analysis | dpb-core | `biomechanics::force::rfd` | `RfdAnalyzer`, `RfdMetrics` |
| **Vestibular** |
| VOR analysis | dpb-core | `vestibular::vor` | `VorAnalyzer`, `VorMetrics` |
| Posturography | dpb-core | `vestibular::posturography` | `PosturographyAnalyzer`, `SotMetrics` |
| **Somatosensory** |
| Proprioception | dpb-core | `somatosensory::proprioception` | `JointPositionSense`, `JpsMetrics` |
| Vibration sense | dpb-core | `somatosensory::vibration` | `VibrationSense`, `VptMetrics` |
| **Cardiopulmonary** |
| VO2 estimation | dpb-core | `cardiopulmonary::vo2` | `Vo2Estimator`, `ExerciseProtocol` |
| Ventilatory threshold | dpb-core | `cardiopulmonary::ventilatory` | `VentilatoryThreshold`, `VtResult` |
| Gas exchange | dpb-core | `cardiopulmonary::gas_exchange` | `GasExchange`, `RespiratoryQuotient` |

### 2.7 Normative Comparison

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| Normative database | dpb-norms | `database` | `NormativeDatabase`, `NormativeEntry` |
| Z-score calculation | dpb-norms | `database` | `NormativeStats::z_score()` |
| Percentile ranking | dpb-norms | `database` | `NormativeStats::percentile()` |
| Impairment classification | dpb-norms | `database` | `ImpairmentLevel` (Normal→Severe) |
| Demographic filtering | dpb-norms | `demographics` | `Demographics`, `DemographicsFilter` |
| MDC calculation | dpb-norms | `database` | `mdc90`, `mdc95` |

#### 2.7.1 Populated Normative Metrics *(40+ metrics)*

| Domain | Metrics | Stratification |
|--------|---------|----------------|
| **Cognitive** | SimpleReactionTime, ChoiceReactionTime, TrailMakingA, TrailMakingB, MocaTotal, DigitSpanForward, DigitSpanBackward, VerbalFluency, StroopInterference | Age |
| **Motor** | GaitVelocity, TimedUpAndGo, GripStrength, RateOfForceDevelopment, TappingFrequency, StrideLength, Cadence | Age, Sex |
| **Balance** | SwayArea, SwayVelocity, SwayPathLength, BergBalanceScale, LosMaxExcursion, LosReactionTime | Age |
| **Physiological** | HrvSdnn, HrvRmssd, HrvPnn50, HeartRate, RespiratoryRate, SpO2, BpSystolic, BpDiastolic | Age |
| **Pain/Sensory** | PressurePainThreshold, PainTolerance, CpmEffect, VibrationThreshold, JointPositionError | Age, Sex |
| **Tremor** | TremorAmplitude, TremorFrequency | Age |
| **Sleep** | TotalSleepTime, SleepEfficiency | Age |

### 2.8 Cognitive Assessment

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| Simple/choice RT | dpb-cognitive | `reaction_time` | `SimpleReactionTime`, `ChoiceReactionTime` |
| Working memory (N-back) | dpb-cognitive | `working_memory` | `NBackTask`, `NBackMetrics` |
| Attention (CPT) | dpb-cognitive | `attention` | `ContinuousPerformanceTest`, `CptMetrics` |
| Stroop task | dpb-cognitive | `attention` | `StroopTask`, `StroopMetrics` |
| Inhibition (Go/No-Go) | dpb-cognitive | `executive` | `GoNoGoTask`, `GoNoGoMetrics` |
| Flanker task | dpb-cognitive | `executive` | `FlankerTask`, `FlankerMetrics` |
| Set shifting (WCST) | dpb-cognitive | `executive` | `WisconsinCardSort`, `WcstMetrics` |
| ADHD assessment | dpb-cognitive | `adhd` | `QbTest`, `AdhdEyeTracking` |

### 2.9 Metrics & Evaluation

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| Classification metrics | dpb-core | `metrics::classification` | `Accuracy`, `Precision`, `Recall`, `F1Score`, `AucRoc` |
| Regression metrics | dpb-core | `metrics::regression` | `MSE`, `RMSE`, `MAE`, `R2Score` |
| Signal quality | dpb-core | `metrics::signal` | `SNR`, `PSNR`, `THD`, `CrestFactor` |
| Clinical validity | dpb-core | `metrics::clinical` | `Sensitivity`, `Specificity`, `LikelihoodRatio` |
| SNN efficiency | dpb-core | `metrics::efficiency` | `SpikeCount`, `SynapticOperations` |
| Power estimation | dpb-core | `power` | `PowerEstimator`, `PowerMetrics` |

### 2.10 Integration & Deployment

| Capability | Crate | Module | Key Types |
|------------|-------|--------|-----------|
| Python bindings | dpb-python | root | `dpb.*` (all submodules) |
| C FFI | dpb-ffi | root | `dpb_*` functions, C header generation |
| Benchmarking | dpb-bench | root | `TimeProfiler`, standard datasets |
| GPU acceleration | dpb-core | `gpu` | WGPU context, compute buffers |

---

## 3. Implementation → Capabilities

### 3.1 dpb-core

```
dpb-core/src/
├── types.rs          → Signal representation, spike events, time series
├── traits.rs         → Core interfaces (Signal, EventEncoder, SyntheticGenerator, etc.)
├── error.rs          → Error handling (DpbError, Result)
├── config.rs         → Framework configuration
├── tensor.rs         → Batched spike tensor operations
├── gpu/              → WGPU-based GPU acceleration
├── signal/
│   ├── mod.rs        → Normalization, peak detection, envelope, RMS
│   ├── filter.rs     → IIR filtering (Butterworth LP/HP/BP/BS)
│   ├── fft.rs        → FFT, STFT, PSD, spectra
│   ├── resample.rs   → Up/downsampling, interpolation
│   └── eeg/
│       ├── bands.rs  → EEG frequency band analysis
│       ├── erp.rs    → Event-related potential extraction
│       └── artifacts.rs → Artifact detection/removal
├── biomechanics/
│   └── force/
│       ├── grip.rs   → Grip strength analysis
│       ├── grf.rs    → Ground reaction force analysis
│       └── rfd.rs    → Rate of force development
├── cardiopulmonary/
│   ├── vo2.rs        → VO2max estimation
│   ├── ventilatory.rs → Ventilatory threshold detection
│   └── gas_exchange.rs → RER, gas exchange metrics
├── pain/
│   ├── qst.rs        → Quantitative sensory testing
│   ├── threshold.rs  → PPT, CPM analysis
│   ├── autonomic.rs  → Pain-autonomic coupling
│   └── scales.rs     → VAS, NRS, McGill, BPI
├── sleep/
│   ├── staging.rs    → Sleep stage classification
│   └── features.rs   → Spindle, K-complex detection
├── somatosensory/
│   ├── proprioception.rs → Joint position sense
│   └── vibration.rs  → Vibration perception threshold
├── vestibular/
│   ├── vor.rs        → VOR gain/phase analysis
│   └── posturography.rs → Dynamic balance assessment
├── metrics/
│   ├── classification.rs → Accuracy, F1, AUC
│   ├── regression.rs → MSE, RMSE, R²
│   ├── signal.rs     → SNR, THD
│   ├── clinical.rs   → Sensitivity, specificity
│   └── efficiency.rs → SNN spike/energy metrics
├── power/            → Hardware power estimation
├── validation/       → ROC analysis, reliability
├── viz/              → Plotting utilities
└── prelude.rs        → Convenience re-exports
```

### 3.2 dpb-encoders

```
dpb-encoders/src/
├── base.rs           → LevelCrossing, TemplateDeviation, Derivative encoders
├── contact/
│   ├── ecg.rs        → ECG encoders (R-peak, morphology, ST, HRV)
│   ├── ppg.rs        → PPG encoders (pulse, amplitude, PTT)
│   ├── eda.rs        → EDA encoders (level crossing, SCR, tonic)
│   ├── emg.rs        → EMG encoders (burst, amplitude, fatigue)
│   └── tremor.rs     → Tremor encoders (level crossing, frequency, amplitude)
├── pose/             → Gait encoders (heel strike, toe off, phase, asymmetry)
├── hand/             → Hand encoders (tap onset, aperture, frequency, decrement)
├── eye/              → Eye encoders (saccade, fixation, microsaccade, pupil)
├── voice/
│   ├── phonation.rs  → F0, jitter, shimmer, HNR encoders
│   ├── articulation.rs → Formant, vowel space encoders
│   └── prosody.rs    → Speech rate, pause, intonation encoders
├── balance.rs        → (NEW) CoP sway, velocity, stability limit encoders
├── force.rs          → (NEW) GRF phase, grip onset, RFD encoders
├── vestibular.rs     → (NEW) VOR gain, nystagmus, caloric encoders
├── pain.rs           → (NEW) Pain threshold, temporal summation, CPM encoders
├── cardiopulmonary.rs → (NEW) HRV, respiratory phase, RSA encoders
├── cognitive.rs      → (NEW) Reaction time, error, lapse encoders
└── templates/        → Population template registry (79+ templates)
```

### 3.3 dpb-neurons

```
dpb-neurons/src/
├── lif/              → 7 LIF variants (IF, LIF, CLIF, ALIF, ELIF, QLIF, GLIF)
├── izhikevich.rs     → Izhikevich model with presets
├── adex.rs           → Adaptive exponential IF
├── calcium.rs        → Calcium-dependent adaptation
├── hodgkin_huxley/   → HH, FitzHugh-Nagumo, Morris-Lecar
├── srm.rs            → Spike response model
├── stochastic.rs     → Stochastic LIF
├── recurrent.rs      → Recurrent dynamics
├── hardware/         → Xylo, Pulsar, Quantized neurons
├── surrogate/        → 6 surrogate gradient functions
├── batch/            → Batch processing layers
├── gpu/              → GPU-accelerated neurons + shaders
└── traits.rs         → NeuronModel, MembraneDynamics, SynapticInput
```

### 3.4 dpb-snn

```
dpb-snn/src/
├── tensor.rs         → SpikeTensor batched operations
├── layers/           → SpikingLinear, Conv1d/2d, Pool, RNN, LSTM, Attention
├── architectures/    → Feedforward, Conv, Recurrent, GCN, Transformer SNNs
├── training/         → BPTT, OTTT, SLTT, loss functions, optimizers
├── conversion/       → ANN→SNN conversion, weight normalization
├── decoders/
│   ├── rate.rs       → Rate-based decoders
│   ├── temporal.rs   → Temporal pattern decoders
│   ├── clinical.rs   → Clinical scale decoders (UPDRS, Berg, Tinetti, MiniBEST, VAS, NRS, VOR, etc.)
│   └── regression.rs → Continuous value decoders
├── fusion/           → Early, Late, Hierarchical, Cross-modal, Gated fusion
├── analysis/         → Convergence analysis, plateau detection, early stopping
├── baselines/        → 44 ANN baseline architectures
└── export/           → Model export for deployment
```

### 3.5 dpb-synth

```
dpb-synth/src/
├── contact/          → ECG, PPG, EMG, EDA, respiratory, thermal, tremor
├── pose/             → Gait, pathological gait, variability
├── hand/             → Hand movement, tapping, tremor
├── eye/              → Saccade, pursuit, fixation, pupil
├── voice/            → Phonation, articulation, prosody, pathological
├── neural/           → EEG, ERP, sleep microstructure
├── force/
│   ├── grf.rs        → Ground reaction force (walking, running, jumping, pathologies)
│   ├── grip.rs       → Grip strength (MVC, sustained, fatigue, pathologies)
│   └── rfd.rs        → Rate of force development (isometric, CMJ, SJ, DJ)
├── balance/
│   ├── cop.rs        → Center of pressure (quiet standing, LOS, pathologies)
│   ├── perturbation.rs → Balance perturbation responses
│   └── sensory.rs    → Sensory manipulation (SOT, mCTSIB)
├── vestibular/
│   ├── vor.rs        → VOR (sinusoidal, head impulse, pathologies)
│   ├── nystagmus.rs  → Nystagmus patterns (BPPV, spontaneous, gaze-evoked)
│   └── caloric.rs    → Caloric test responses
├── pain/             → Pain ramp, temporal summation, CPM, QST battery
├── cardiopulmonary/  → HRV, respiratory, BP (resting, exercise, pathologies)
├── cognitive/        → RT, Flanker, Go/NoGo, N-back (ADHD, MCI, TBI patterns)
├── multimodal/       → Combined modality streaming
├── level3/           → Audio world, SMPL skeleton, style transfer, video
├── media/            → BARK audio processing
├── streaming/        → StreamingGenerator trait, ring buffers
└── traits.rs         → SyntheticGenerator, GroundTruth, ParameterSpace
```

### 3.6 dpb-norms

```
dpb-norms/src/
├── database.rs       → NormativeDatabase with 40+ populated metrics (age/sex-stratified)
├── demographics.rs   → Demographics, Sex, Ethnicity, AgeGroup, filters
└── metrics.rs        → MetricType (60 variants), MetricDomain, MetricDirection
```

### 3.7 dpb-cognitive

```
dpb-cognitive/src/
├── reaction_time.rs  → Simple RT, Choice RT
├── working_memory.rs → N-back task
├── attention.rs      → CPT, Stroop
├── executive.rs      → Go/No-Go, Flanker, WCST
└── adhd.rs           → QbTest, ADHD eye tracking
```

### 3.8 dpb-python

```
dpb-python/src/
├── encoders/         → Python encoder wrappers
├── neurons/          → Python neuron model wrappers
├── snn/              → Python SNN layer/architecture wrappers
├── training/         → Python training infrastructure
├── synth/            → Python synthetic generator wrappers
├── metrics/          → Python metric wrappers
└── gpu/              → Python GPU management
```

### 3.9 dpb-ffi

```
dpb-ffi/src/
├── lib.rs            → C-compatible API entry points
├── types.rs          → C wrapper types (DpbTimeSeries, DpbSpikeTrain, DpbEncoder)
├── timeseries.rs     → Time series FFI functions
├── spiketrain.rs     → Spike train FFI functions
└── encoder.rs        → Encoder FFI functions
```

### 3.10 dpb-bench

```
dpb-bench/src/
├── datasets/         → Standard synthetic datasets
├── baselines/        → ANN comparison baselines
├── profiling/        → Time, memory, spike, energy profiling
├── reports/          → JSON, CSV, Markdown export
└── scenarios/        → Pre-configured benchmark scenarios
```

---

## 4. Dependency Graph

### 4.1 Crate Dependencies

```
                    ┌─────────────────┐
                    │  dpb-cognitive  │  (standalone)
                    └─────────────────┘

                    ┌─────────────────┐
                    │    dpb-core     │  (foundation)
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  dpb-encoders   │ │  dpb-neurons    │ │   dpb-synth     │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         │         ┌─────────────────┐           │
         └────────►│    dpb-snn      │◄──────────┘
                   └────────┬────────┘
                            │
                   ┌─────────────────┐
                   │   dpb-norms     │
                   └────────┬────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  dpb-python     │ │    dpb-ffi      │ │   dpb-bench     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 4.2 Dependency Matrix

| Crate | dpb-core | dpb-encoders | dpb-neurons | dpb-snn | dpb-synth | dpb-norms | dpb-cognitive |
|-------|:--------:|:------------:|:-----------:|:-------:|:---------:|:---------:|:-------------:|
| dpb-core | - | | | | | | |
| dpb-encoders | ✓ | - | | | | | |
| dpb-neurons | ✓ | | - | | | | |
| dpb-synth | ✓ | | | | - | | |
| dpb-snn | ✓ | ✓ | ✓ | - | | | |
| dpb-norms | ✓ | | | | | - | |
| dpb-cognitive | | | | | | | - |
| dpb-python | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| dpb-ffi | ✓ | ✓ | | | ✓ | | |
| dpb-bench | ✓ | ✓ | ✓ | ✓ | ✓ | | |

### 4.3 External Dependencies

| Category | Dependencies |
|----------|--------------|
| **Linear Algebra** | nalgebra, ndarray, glam |
| **GPU** | wgpu, bytemuck |
| **FFT** | rustfft, num-complex |
| **Statistics** | statrs, rand, rand_distr |
| **Serialization** | serde, serde_json, bincode |
| **Async/Parallel** | tokio, rayon, crossbeam |
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

### 5.2 Completed Implementations *(Since v0.1.0)*

#### Phase D - Encoders (18 new encoders)

| Module | Encoders | Templates |
|--------|----------|-----------|
| `balance.rs` | CopSwayEncoder, CopVelocityEncoder, StabilityLimitEncoder | SwayAreaTemplate, SwayVelocityTemplate, StabilityLimitTemplate |
| `force.rs` | GrfPhaseEncoder, GripOnsetEncoder, RfdEncoder | PeakGrfTemplate, GripStrengthTemplate, RfdTemplate |
| `vestibular.rs` | VorGainEncoder, NystagmusEncoder, CaloricEncoder | VorGainTemplate, NystagmusSPVTemplate, CaloricAsymmetryTemplate |
| `pain.rs` | PainThresholdEncoder, TemporalSummationEncoder, CpmEncoder | PressurePainThresholdTemplate, WindUpRatioTemplate |
| `cardiopulmonary.rs` | HrvEncoder, RespiratoryPhaseEncoder, RsaEncoder | RmssdTemplate, RespiratoryRateTemplate, RsaTemplate |
| `cognitive.rs` | ReactionTimeEncoder, ErrorEncoder, LapseEncoder | SimpleRtTemplate, AccuracyTemplate, LapseRateTemplate |

#### Phase E - Decoders (8 new clinical decoders)

| Category | Decoders | Output Scale |
|----------|----------|--------------|
| Balance | TinettiDecoder | 0-28 (Tinetti scale) |
| Balance | MiniBESTDecoder | 0-32 (MiniBEST) |
| Pain | VasDecoder | 0-100mm (Visual Analog Scale) |
| Pain | NrsDecoder | 0-10 (Numeric Rating Scale) |
| Pain | QstPhenotypeDecoder | Sensory phenotype classification |
| Vestibular | VorGainDecoder | VOR gain ratio |
| Vestibular | CanalParesisDecoder | Unilateral weakness % |
| Vestibular | BppvDecoder | BPPV probability |

#### Phase F - Normative Data (25+ new metrics)

| Domain | Metrics Added |
|--------|---------------|
| Pain/Sensory | PressurePainThreshold, PainTolerance, CpmEffect, VibrationThreshold, JointPositionError |
| Balance | SwayVelocity, SwayPathLength, LosMaxExcursion, LosReactionTime |
| Motor | RateOfForceDevelopment, TappingFrequency, StrideLength, Cadence |
| Physiological | HeartRate, RespiratoryRate, SpO2, BpSystolic, BpDiastolic, HrvPnn50 |
| Cognitive | DigitSpanForward, DigitSpanBackward, VerbalFluency, StroopInterference |
| Tremor | TremorAmplitude, TremorFrequency |

### 5.3 Remaining Gaps

#### Recently Completed *(v0.2.1)*

| Gap | Domain | Status |
|-----|--------|--------|
| ~~**EEG encoders**~~ | Encoders | ✓ **COMPLETED** - Alpha, Beta, Theta, Gamma, Delta band encoders + ERP, Spindle, Artifact |
| ~~**Real-time streaming**~~ | Synth | ✓ **COMPLETED** - EEG streaming with band power profiles, artifact generation |
| ~~**Normative database**~~ | Norms | ✓ **COMPLETED** - All 60 metric types now populated (67% → 100%) |

#### Medium Priority Gaps

| Gap | Domain | Impact | Recommended Implementation |
|-----|--------|--------|---------------------------|
| **Multi-modal norms** | Norms | Single-modality only | Add cross-modal normative comparisons |
| **Fatigue detection algorithms** | Core | No fatigue-specific analysis | Add `FatigueDetector` for EMG/force/cognitive |
| **Seizure detection** | Core | EEG seizure detection stub | Implement full seizure detection algorithm |
| **Respiratory analysis** | Core | Limited respiratory analysis | Add breath detection, apnea detection |

#### Low Priority Gaps

| Gap | Domain | Impact | Recommended Implementation |
|-----|--------|--------|---------------------------|
| **Additional pathologies** | Synth | Some conditions not modeled | Add ALS, MS, stroke-specific patterns |
| **Longitudinal modeling** | Synth | No disease progression | Add progressive disease models |
| **Medication effects** | Synth | No pharmacological modeling | Add medication response generators |
| **Hardware export** | SNN | Limited deployment targets | Add more neuromorphic hardware backends |
| **Visualization tools** | Core | Basic plotting only | Add interactive visualization |

### 5.4 Completeness Metrics

| Metric | Previous | Current | Target | Coverage |
|--------|----------|---------|--------|----------|
| Neuron models | 19 | 19 | 20 | 95% |
| Event encoders | 95 | **103+** | 110 | **94%** |
| Population templates | 79 | **85+** | 90 | **94%** |
| Synthetic generators | 200+ | **210+** | 220 | **95%** |
| SNN decoders | 40 | 40 | 50 | 80% |
| ANN baselines | 44 | 44 | 50 | 88% |
| Normative metrics | 40 | **60** | 60 | **100%** |
| Unit tests | 500+ | **530+** | 550 | **96%** |

### 5.5 Recommended Development Priorities

1. ~~**Phase G - EEG Encoders**~~ ✓ **COMPLETED**
   - ✓ Alpha, beta, theta, gamma, delta band power encoders
   - ✓ Spindle encoder, ERP encoder
   - ✓ Artifact detection encoder

2. ~~**Phase H - Real-time Streaming**~~ ✓ **COMPLETED**
   - ✓ EEG streaming with state profiles (wake, sleep stages)
   - ✓ Band power tracking and artifact injection
   - Streaming analysis pipelines (future)

3. **Phase I - Expanded Clinical Decoders**
   - Fatigue detection decoders
   - Seizure detection pipeline
   - Respiratory pattern decoders

4. **Phase J - Advanced Integration**
   - Cross-modal normative comparisons
   - Integrated assessment batteries
   - Composite score generation

---

## 6. Quick Reference

### 6.1 Common Use Cases

| Use Case | Crates | Key Functions |
|----------|--------|---------------|
| Encode ECG to spikes | dpb-encoders | `EcgRPeakEncoder::encode()` |
| Encode balance to spikes | dpb-encoders | `CopSwayEncoder::encode()` |
| Encode pain response | dpb-encoders | `PainThresholdEncoder::encode()` |
| Train SNN classifier | dpb-snn | `FeedforwardSNN::forward()`, `BPTT::step()` |
| Generate synthetic ECG | dpb-synth | `StreamingEcg::generate()` |
| Compare to norms | dpb-norms | `NormativeDatabase::compare()` |
| Decode balance scale | dpb-snn | `TinettiDecoder::decode()` |
| Decode pain scale | dpb-snn | `VasDecoder::decode()` |
| Analyze VOR | dpb-core | `VorAnalyzer::analyze_vhit()` |
| Detect sleep stages | dpb-core | `SleepStager::stage_from_eeg()` |
| Assess pain | dpb-core | `QstAnalyzer::analyze_modality()` |
| Run cognitive task | dpb-cognitive | `NBackTask::run_trial()` |

### 6.2 Type Quick Reference

| Type | Crate | Purpose |
|------|-------|---------|
| `SpikeEvent` | dpb-core | Single spike (timestamp, channel, polarity) |
| `SpikeTrain` | dpb-core | Collection of spikes |
| `TimeSeries` | dpb-core | Multi-channel continuous signal |
| `GroundTruth` | dpb-core | Reference labels/values |
| `SpikeTensor` | dpb-snn | Batched spike representation |
| `NormativeStats` | dpb-norms | Normative statistics with Z-score |
| `Demographics` | dpb-norms | Age, sex, population filters |

### 6.3 Trait Quick Reference

| Trait | Crate | Implementors |
|-------|-------|-------------|
| `EventEncoder` | dpb-core | All encoders in dpb-encoders (95+) |
| `PopulationTemplate` | dpb-core | All templates (79+) |
| `NeuronModel` | dpb-core | All neurons in dpb-neurons |
| `SyntheticGenerator` | dpb-core | All generators in dpb-synth |
| `Metric` | dpb-core | All metrics in dpb-core/metrics |
| `SpikingLayer` | dpb-core | All layers in dpb-snn/layers |
| `Decoder` | dpb-snn | All decoders (40) |

### 6.4 Configuration Quick Reference

| Config | Crate | Configures |
|--------|-------|------------|
| `LifConfig` | dpb-neurons | LIF neuron parameters |
| `SNNConfig` | dpb-snn | SNN architecture parameters |
| `StreamingConfig` | dpb-synth | Streaming generator parameters |
| `GrfConfig` | dpb-synth | GRF generator parameters |
| `Demographics` | dpb-norms | Normative comparison filters |
| `CopSwayConfig` | dpb-encoders | Balance encoder parameters |
| `PainThresholdConfig` | dpb-encoders | Pain encoder parameters |
| `HrvConfig` | dpb-encoders | HRV encoder parameters |

### 6.5 New Encoder Quick Reference

| Encoder | Module | Input Signal | Output Events |
|---------|--------|--------------|---------------|
| `CopSwayEncoder` | balance | CoP trajectory | Sway area exceedance |
| `CopVelocityEncoder` | balance | CoP trajectory | Velocity threshold crossings |
| `StabilityLimitEncoder` | balance | LOS task | Limit exceedance events |
| `GrfPhaseEncoder` | force | GRF waveform | Heel strike, toe-off, peaks |
| `GripOnsetEncoder` | force | Grip force | Force onset, peak, release |
| `RfdEncoder` | force | Force-time curve | RFD threshold events |
| `VorGainEncoder` | vestibular | Head/eye velocity | Gain deviation events |
| `NystagmusEncoder` | vestibular | Eye velocity | Nystagmus beats |
| `CaloricEncoder` | vestibular | Caloric response | Asymmetry events |
| `PainThresholdEncoder` | pain | Pressure/stimulus | Threshold crossings |
| `TemporalSummationEncoder` | pain | Pain ratings | Wind-up events |
| `CpmEncoder` | pain | CPM response | Modulation events |
| `HrvEncoder` | cardiopulmonary | RR intervals | HRV feature events |
| `RespiratoryPhaseEncoder` | cardiopulmonary | Respiratory signal | Inspiration/expiration |
| `RsaEncoder` | cardiopulmonary | RR + respiratory | RSA events |
| `ReactionTimeEncoder` | cognitive | RT data | Fast/slow RT events |
| `ErrorEncoder` | cognitive | Response accuracy | Error events |
| `LapseEncoder` | cognitive | Attention data | Lapse events |

---

## Document Information

| Field | Value |
|-------|-------|
| Version | 2.0.0 |
| Last Updated | 2025-12-18 |
| Generated By | System catalog analysis |
| Framework Version | 0.2.0 |
| Rust Edition | 2024 |
| Changes Since v1.0 | +18 encoders, +8 decoders, +25 normative metrics |

---

*This document serves as the authoritative reference for the Delta Predictive Biosensing framework. For implementation details, refer to the source code and inline documentation.*
