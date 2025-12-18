# DPB Gap Resolution Plan - Parallel Implementation Strategy

> **Created:** December 2025
> **Total Gaps:** 45+
> **Parallel Work Streams:** 8
> **Estimated Total Effort:** 80-100 developer-days (parallelized to ~15-20 days with 8 streams)

---

## Executive Summary

This plan organizes all identified gaps into 8 parallel work streams based on:
1. **Crate boundaries** - Minimize merge conflicts
2. **Dependency ordering** - Foundational work first within each stream
3. **Skill domains** - Group related work together
4. **Independence** - Maximize parallel execution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PARALLEL WORK STREAMS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stream 1    Stream 2    Stream 3    Stream 4    Stream 5    Stream 6    Stream 7    Stream 8  │
│  ─────────   ─────────   ─────────   ─────────   ─────────   ─────────   ─────────   ─────────  │
│  ECG/HRV     Transforms  Neural Net  Clinical    Normative   Synthetic   Integration  Testing   │
│  Signal      & Analysis  & Learning  Pipeline    Database    Data Gen    & Formats    & Docs    │
│  Processing                                                                                      │
│                                                                                                  │
│  dpb-core    dpb-core    dpb-neurons dpb-snn     dpb-norms   dpb-synth   dpb-core    all crates │
│  signal/     signal/     dpb-snn     dpb-core    dpb-core    dpb-bench   dpb-ffi                │
│                                                                                                  │
│  ↓           ↓           ↓           ↓           ↓           ↓           ↓           ↓          │
│  Week 1-2    Week 1-2    Week 1-3    Week 1-3    Week 1-2    Week 1-2    Week 1-2    Week 2-3   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stream 1: ECG & HRV Signal Processing

**Location:** `dpb-core/src/signal/`
**Dependencies:** None (foundational)
**Effort:** 8-10 days

### Tasks (Sequential within stream)

| Order | Task | File | Description | Effort |
|-------|------|------|-------------|--------|
| 1.1 | **ECG R-Peak Detection** | `ecg.rs` (new) | Pan-Tompkins algorithm with adaptive thresholds | 2-3 days |
| 1.2 | **ECG QRS Morphology** | `ecg.rs` | Template matching, QRS classification | 2 days |
| 1.3 | **HRV Time-Domain** | `hrv.rs` (new) | SDNN, RMSSD, pNN50, SDANN, triangular index | 1-2 days |
| 1.4 | **HRV Frequency-Domain** | `hrv.rs` | VLF/LF/HF power, LF/HF ratio, total power | 2 days |
| 1.5 | **ECG Arrhythmia Detection** | `ecg.rs` | AFib, PVC, PAC, bradycardia, tachycardia | 2-3 days |

### Implementation Details

```rust
// ecg.rs - New file structure
pub struct PanTompkinsDetector {
    sample_rate: f64,
    bandpass_filter: BandpassFilter,  // 5-15 Hz
    derivative_kernel: [f64; 5],
    integration_window_ms: f64,       // 150ms typical
    refractory_period_ms: f64,        // 200ms
    adaptive_threshold: AdaptiveThreshold,
}

pub struct RRIntervals {
    intervals_ms: Vec<f64>,
    timestamps: Vec<f64>,
    quality_flags: Vec<QualityFlag>,
}

// hrv.rs - New file structure
pub struct HrvTimeDomain {
    pub sdnn_ms: f64,           // Standard deviation of NN intervals
    pub rmssd_ms: f64,          // Root mean square of successive differences
    pub pnn50_percent: f64,     // Percentage of successive differences > 50ms
    pub sdann_ms: f64,          // SD of 5-min NN averages
    pub triangular_index: f64,  // Total NN / max histogram bin
    pub tinn_ms: f64,           // Triangular interpolation of NN histogram
}

pub struct HrvFrequencyDomain {
    pub vlf_power_ms2: f64,     // 0.003-0.04 Hz
    pub lf_power_ms2: f64,      // 0.04-0.15 Hz
    pub hf_power_ms2: f64,      // 0.15-0.4 Hz
    pub lf_hf_ratio: f64,
    pub total_power_ms2: f64,
    pub lf_nu: f64,             // Normalized units
    pub hf_nu: f64,
}
```

### Acceptance Criteria
- [ ] R-peak detection sensitivity >99% on MIT-BIH arrhythmia database
- [ ] HRV metrics match Kubios/PhysioNet reference implementations within 5%
- [ ] Arrhythmia detection F1 >0.85 for AFib, >0.80 for PVC

---

## Stream 2: Advanced Signal Transforms

**Location:** `dpb-core/src/signal/`
**Dependencies:** None
**Effort:** 8-10 days

### Tasks

| Order | Task | File | Description | Effort |
|-------|------|------|-------------|--------|
| 2.1 | **Wavelet Transform** | `wavelet.rs` (new) | CWT (Morlet, Mexican Hat), DWT (Daubechies, Symlet) | 3-4 days |
| 2.2 | **Hilbert Transform** | `hilbert.rs` (new) | Analytic signal, instantaneous phase/frequency/amplitude | 2 days |
| 2.3 | **Independent Component Analysis** | `ica.rs` (new) | FastICA, artifact removal for EEG/ECG | 3-4 days |
| 2.4 | **Empirical Mode Decomposition** | `emd.rs` (new) | IMF extraction, Hilbert-Huang transform | 2-3 days |

### Implementation Details

```rust
// wavelet.rs
pub enum WaveletFamily {
    Morlet { omega0: f64 },
    MexicanHat,
    Daubechies(u8),  // db1-db20
    Symlet(u8),      // sym2-sym20
    Coiflet(u8),     // coif1-coif5
}

pub struct ContinuousWaveletTransform {
    wavelet: WaveletFamily,
    scales: Vec<f64>,
    sample_rate: f64,
}

pub struct DiscreteWaveletTransform {
    wavelet: WaveletFamily,
    levels: usize,
    mode: ExtensionMode,  // Symmetric, Periodic, Zero
}

// hilbert.rs
pub struct AnalyticSignal {
    pub amplitude_envelope: Vec<f64>,
    pub instantaneous_phase: Vec<f64>,
    pub instantaneous_frequency: Vec<f64>,
}

pub fn hilbert_transform(signal: &[f64]) -> Vec<Complex64>;
pub fn analytic_signal(signal: &[f64], sample_rate: f64) -> AnalyticSignal;

// ica.rs
pub struct FastICA {
    n_components: usize,
    max_iter: usize,
    tol: f64,
    whiten: bool,
    fun: NonlinearFunction,  // LogCosh, Exp, Cube
}

impl FastICA {
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<ICAResult>;
    pub fn unmix(&self, mixed: &Array2<f64>) -> Array2<f64>;
}
```

### Acceptance Criteria
- [ ] Wavelet reconstruction error <1e-10 for DWT
- [ ] Hilbert transform matches scipy.signal.hilbert within 1e-6
- [ ] ICA successfully separates 3+ sources in synthetic mixtures

---

## Stream 3: Neural Network Enhancements

**Location:** `dpb-neurons/`, `dpb-snn/`
**Dependencies:** None
**Effort:** 12-15 days

### Tasks

| Order | Task | File | Description | Effort |
|-------|------|------|-------------|--------|
| 3.1 | **Reservoir Computing** | `dpb-neurons/src/reservoir.rs` | Echo State Network, Liquid State Machine | 4-5 days |
| 3.2 | **Hebbian Learning** | `dpb-snn/src/learning/hebbian.rs` | STDP variants, BCM, Oja's rule | 3-4 days |
| 3.3 | **Network Pruning** | `dpb-snn/src/optimization/pruning.rs` | Magnitude, gradient, lottery ticket | 3-4 days |
| 3.4 | **Knowledge Distillation** | `dpb-snn/src/training/distillation.rs` | ANN→SNN, SNN→SNN distillation | 3-4 days |
| 3.5 | **Dendritic Computation** | `dpb-neurons/src/models/dendritic.rs` | Multi-compartment, dendritic spikes | 3-4 days |

### Implementation Details

```rust
// reservoir.rs
pub struct EchoStateNetwork {
    input_weights: Array2<f64>,      // W_in
    reservoir_weights: Array2<f64>,   // W (sparse)
    output_weights: Array2<f64>,      // W_out (trained)
    spectral_radius: f64,
    leak_rate: f64,
    reservoir_size: usize,
    sparsity: f64,
}

pub struct LiquidStateMachine {
    neurons: Vec<Box<dyn SpikingNeuron>>,
    connectivity: SparseMatrix,
    input_projections: Vec<InputProjection>,
    readout: LinearReadout,
}

// hebbian.rs
pub trait HebbianRule {
    fn update(&self, pre: &SpikeTrace, post: &SpikeTrace, weight: f64) -> f64;
}

pub struct STDP {
    a_plus: f64,   // LTP amplitude
    a_minus: f64,  // LTD amplitude
    tau_plus: f64, // LTP time constant
    tau_minus: f64, // LTD time constant
}

pub struct BCMRule {
    theta: f64,        // Modification threshold
    tau_theta: f64,    // Threshold adaptation rate
    learning_rate: f64,
}

// pruning.rs
pub enum PruningStrategy {
    MagnitudeBased { threshold: f64 },
    GradientBased { threshold: f64 },
    LotteryTicket { prune_ratio: f64, iterations: usize },
    StructuredChannel { ratio: f64 },
}

pub struct NetworkPruner {
    strategy: PruningStrategy,
    schedule: PruningSchedule,  // OneShot, Iterative, Gradual
}
```

### Acceptance Criteria
- [ ] ESN achieves <0.1 NRMSE on Mackey-Glass prediction
- [ ] STDP produces stable weight distributions
- [ ] Pruning achieves 70% sparsity with <5% accuracy loss

---

## Stream 4: Clinical Pipeline & Inference

**Location:** `dpb-snn/`, `dpb-core/`
**Dependencies:** Stream 1 (HRV), Stream 2 (transforms)
**Effort:** 12-15 days

### Tasks

| Order | Task | File | Description | Effort |
|-------|------|------|-------------|--------|
| 4.1 | **Real-Time Pipeline** | `dpb-core/src/pipeline/` (new) | Streaming inference, buffer management | 4-5 days |
| 4.2 | **Model Calibration** | `dpb-snn/src/calibration/` (new) | Temperature scaling, Platt scaling, isotonic | 3-4 days |
| 4.3 | **Explainability** | `dpb-snn/src/explain/` (new) | Attention viz, spike importance, SHAP-like | 4-5 days |
| 4.4 | **Longitudinal Tracking** | `dpb-core/src/tracking/` (new) | Within-subject change detection, trends | 2-3 days |
| 4.5 | **HIPAA Utilities** | `dpb-core/src/privacy/` (new) | De-identification, k-anonymity | 2 days |

### Implementation Details

```rust
// pipeline/mod.rs
pub struct RealTimePipeline<E: Encoder, M: SpikingNetwork, D: Decoder> {
    encoder: E,
    network: M,
    decoder: D,
    input_buffer: RingBuffer<f64>,
    spike_buffer: RingBuffer<SpikeEvent>,
    window_size: usize,
    hop_size: usize,
    latency_budget_ms: f64,
}

impl<E, M, D> RealTimePipeline<E, M, D> {
    pub fn process_sample(&mut self, sample: f64) -> Option<D::Output>;
    pub fn process_chunk(&mut self, chunk: &[f64]) -> Vec<D::Output>;
    pub fn get_latency_stats(&self) -> LatencyStats;
}

// calibration/mod.rs
pub trait Calibrator {
    fn fit(&mut self, logits: &[f64], labels: &[usize]);
    fn calibrate(&self, logits: &[f64]) -> Vec<f64>;
}

pub struct TemperatureScaling {
    temperature: f64,
}

pub struct IsotonicCalibration {
    isotonic_regression: IsotonicRegression,
}

pub fn expected_calibration_error(probs: &[f64], labels: &[usize], n_bins: usize) -> f64;

// explain/mod.rs
pub struct SpikeImportance {
    pub neuron_id: usize,
    pub spike_time: f64,
    pub importance_score: f64,
    pub contribution_to_output: f64,
}

pub struct AttentionMap {
    pub temporal_attention: Vec<f64>,
    pub channel_attention: Vec<f64>,
    pub cross_attention: Option<Array2<f64>>,
}

pub fn compute_spike_importance(network: &SpikingNetwork, input: &SpikeTrains) -> Vec<SpikeImportance>;
```

### Acceptance Criteria
- [ ] Real-time pipeline latency <50ms for 256 Hz input
- [ ] ECE (Expected Calibration Error) <0.05 after calibration
- [ ] Explainability outputs validated against ground truth synthetic data

---

## Stream 5: Normative Database Extensions

**Location:** `dpb-norms/`, `dpb-core/`
**Dependencies:** Stream 1 (HRV metrics)
**Effort:** 8-10 days

### Tasks

| Order | Task | File | Description | Effort |
|-------|------|------|-------------|--------|
| 5.1 | **Pediatric Norms** | `dpb-norms/src/pediatric.rs` (new) | Ages 2-17, developmental staging | 3-4 days |
| 5.2 | **Geriatric Norms** | `dpb-norms/src/geriatric.rs` (new) | 80+ stratification, frailty adjustment | 2-3 days |
| 5.3 | **Ethnic Stratification** | `dpb-norms/src/ethnic.rs` (new) | Ethnicity-specific reference ranges | 2-3 days |
| 5.4 | **Longitudinal MDC** | `dpb-norms/src/longitudinal.rs` (new) | Minimal detectable change, SEM | 2 days |
| 5.5 | **Practice Effects** | `dpb-norms/src/practice.rs` (new) | Serial testing corrections | 1-2 days |

### Implementation Details

```rust
// pediatric.rs
pub struct PediatricNorms {
    age_months: AgeRange,
    developmental_stage: DevelopmentalStage,
    metrics: HashMap<MetricId, PediatricReference>,
}

pub enum DevelopmentalStage {
    Infant,      // 0-12 months
    Toddler,     // 1-3 years
    Preschool,   // 3-5 years
    SchoolAge,   // 6-12 years
    Adolescent,  // 13-17 years
}

pub struct PediatricReference {
    pub percentiles: Percentiles,
    pub age_regression: Option<AgeRegression>,  // For continuous age adjustment
    pub sex_specific: bool,
}

// longitudinal.rs
pub struct MinimalDetectableChange {
    pub mdc_90: f64,  // 90% confidence
    pub mdc_95: f64,  // 95% confidence
    pub sem: f64,     // Standard error of measurement
    pub icc: f64,     // Intraclass correlation
}

pub fn calculate_mdc(test_retest_data: &[(f64, f64)], confidence: f64) -> MinimalDetectableChange;

pub fn is_real_change(baseline: f64, followup: f64, mdc: &MinimalDetectableChange) -> ChangeStatus;
```

### Acceptance Criteria
- [ ] Pediatric norms cover all existing metric types
- [ ] MDC calculations validated against published reliability studies
- [ ] Practice effect corrections reduce false-positive change detection by >30%

---

## Stream 6: Synthetic Data Enhancements

**Location:** `dpb-synth/`
**Dependencies:** None
**Effort:** 8-10 days

### Tasks

| Order | Task | File | Description | Effort |
|-------|------|------|-------------|--------|
| 6.1 | **Signal Augmentation Suite** | `augmentation/` (new dir) | Noise, scaling, time warp, mixup | 2-3 days |
| 6.2 | **Multi-Patient Cohort** | `cohort.rs` (new) | Population-level variation | 2-3 days |
| 6.3 | **Longitudinal Trajectories** | `longitudinal.rs` (new) | Individual progression curves | 2-3 days |
| 6.4 | **Treatment Response** | `treatment.rs` (new) | Intervention effect modeling | 2 days |
| 6.5 | **Comorbidity Modeling** | `comorbidity.rs` (new) | Multi-condition interactions | 2-3 days |

### Implementation Details

```rust
// augmentation/mod.rs
pub trait SignalAugmentation {
    fn augment(&self, signal: &[f64], rng: &mut impl Rng) -> Vec<f64>;
}

pub struct GaussianNoise { pub snr_db: f64 }
pub struct TimeWarp { pub sigma: f64, pub knots: usize }
pub struct MagnitudeScale { pub range: (f64, f64) }
pub struct TimeShift { pub max_shift_samples: usize }
pub struct Mixup { pub alpha: f64 }
pub struct CutMix { pub ratio: f64 }
pub struct SpecAugment { pub freq_mask: usize, pub time_mask: usize }

pub struct AugmentationPipeline {
    augmentations: Vec<(Box<dyn SignalAugmentation>, f64)>,  // (aug, probability)
}

// cohort.rs
pub struct CohortGenerator {
    n_subjects: usize,
    age_distribution: Distribution,
    sex_ratio: f64,
    disease_prevalence: HashMap<String, f64>,
    inter_subject_variability: f64,
}

pub struct VirtualPatient {
    pub id: PatientId,
    pub demographics: Demographics,
    pub baseline_physiology: PhysiologyProfile,
    pub conditions: Vec<Condition>,
    pub medications: Vec<Medication>,
}

// longitudinal.rs
pub struct LongitudinalTrajectory {
    pub patient: VirtualPatient,
    pub timepoints: Vec<TimePoint>,
    pub progression_model: ProgressionModel,
    pub noise_model: NoiseModel,
}

impl LongitudinalTrajectory {
    pub fn generate_visit(&self, time: Duration) -> VisitData;
    pub fn generate_series(&self, times: &[Duration]) -> Vec<VisitData>;
}
```

### Acceptance Criteria
- [ ] Augmented data improves model generalization by >5%
- [ ] Cohort variation matches real-world population statistics
- [ ] Longitudinal trajectories validated against published disease courses

---

## Stream 7: Integration & Data Formats

**Location:** `dpb-core/`, `dpb-ffi/`
**Dependencies:** None
**Effort:** 10-12 days

### Tasks

| Order | Task | File | Description | Effort |
|-------|------|------|-------------|--------|
| 7.1 | **WFDB/PhysioNet** | `dpb-core/src/io/wfdb.rs` (new) | Read/write MIT-BIH format | 2-3 days |
| 7.2 | **EDF/EDF+** | `dpb-core/src/io/edf.rs` (new) | European Data Format support | 2-3 days |
| 7.3 | **BIDS Format** | `dpb-core/src/io/bids.rs` (new) | Brain Imaging Data Structure | 2-3 days |
| 7.4 | **ONNX Runtime** | `dpb-snn/src/export/onnx.rs` (new) | Export SNN to ONNX | 3-4 days |
| 7.5 | **Neuromorphic Export** | `dpb-snn/src/export/neuromorphic.rs` (new) | Loihi, SpiNNaker formats | 3-4 days |

### Implementation Details

```rust
// wfdb.rs
pub struct WfdbReader {
    header: WfdbHeader,
    signals: Vec<WfdbSignal>,
    annotations: Vec<WfdbAnnotation>,
}

impl WfdbReader {
    pub fn open(record_path: &Path) -> Result<Self>;
    pub fn read_signal(&self, channel: usize, start: usize, length: usize) -> Vec<f64>;
    pub fn get_annotations(&self) -> &[WfdbAnnotation];
}

pub struct WfdbWriter {
    pub fn new(path: &Path, signals: &[SignalSpec]) -> Result<Self>;
    pub fn write_samples(&mut self, samples: &[Vec<f64>]) -> Result<()>;
    pub fn add_annotation(&mut self, ann: WfdbAnnotation) -> Result<()>;
}

// edf.rs
pub struct EdfReader {
    header: EdfHeader,
    signals: Vec<EdfSignal>,
    annotations: Option<Vec<EdfAnnotation>>,  // EDF+ only
}

pub struct EdfHeader {
    pub version: String,
    pub patient_id: String,
    pub recording_id: String,
    pub start_date: NaiveDate,
    pub start_time: NaiveTime,
    pub n_records: usize,
    pub record_duration: f64,
}

// onnx.rs
pub struct OnnxExporter {
    opset_version: i64,
    optimize: bool,
    quantize: Option<QuantizationConfig>,
}

impl OnnxExporter {
    pub fn export_snn<N: SpikingNetwork>(&self, network: &N, path: &Path) -> Result<()>;
    pub fn export_encoder<E: Encoder>(&self, encoder: &E, path: &Path) -> Result<()>;
}

// neuromorphic.rs
pub enum NeuromorphicTarget {
    IntelLoihi { chip_version: u8 },
    SpiNNaker { board_version: u8 },
    BrainScaleS { wafer: u8 },
}

pub struct NeuromorphicExporter {
    target: NeuromorphicTarget,
    weight_precision: u8,
    time_resolution_us: f64,
}
```

### Acceptance Criteria
- [ ] WFDB reader successfully loads all MIT-BIH records
- [ ] EDF reader handles both EDF and EDF+ formats
- [ ] ONNX export produces valid, runnable models

---

## Stream 8: Testing & Documentation

**Location:** All crates
**Dependencies:** All other streams (runs in parallel, tests as features complete)
**Effort:** 10-12 days

### Tasks

| Order | Task | Location | Description | Effort |
|-------|------|----------|-------------|--------|
| 8.1 | **Integration Tests** | `tests/integration/` | End-to-end pipeline tests | 3-4 days |
| 8.2 | **Clinical Validation** | `tests/clinical/` | Tests against published datasets | 4-5 days |
| 8.3 | **API Documentation** | All crates | Comprehensive rustdoc | 2-3 days |
| 8.4 | **Tutorial Notebooks** | `examples/notebooks/` | Jupyter examples | 3-4 days |
| 8.5 | **Architecture Guide** | `docs/ARCHITECTURE.md` | System design documentation | 1-2 days |

### Implementation Details

```rust
// tests/integration/pipeline_test.rs
#[test]
fn test_ecg_to_hrv_pipeline() {
    let ecg_data = load_test_ecg("mit-bih/100");
    let pipeline = Pipeline::new()
        .add_stage(PanTompkinsDetector::default())
        .add_stage(HrvAnalyzer::default())
        .add_stage(HrvEncoder::default())
        .add_stage(SpikingClassifier::load("models/hrv_classifier.bin"));

    let result = pipeline.process(&ecg_data);
    assert!(result.confidence > 0.8);
}

// tests/clinical/validation_test.rs
#[test]
fn validate_seizure_detection_chb_mit() {
    let dataset = ChbMitDataset::load("data/chb-mit");
    let detector = SeizureDetector::default();

    let metrics = evaluate_detector(&detector, &dataset);
    assert!(metrics.sensitivity > 0.90);
    assert!(metrics.false_positive_rate < 0.5);  // per hour
}
```

### Documentation Structure

```
docs/
├── ARCHITECTURE.md          # System design, data flow
├── GETTING_STARTED.md       # Quick start guide
├── API_GUIDE.md             # API usage patterns
├── CLINICAL_GUIDE.md        # Clinical interpretation
├── DEPLOYMENT_GUIDE.md      # Production deployment
└── tutorials/
    ├── 01_basic_pipeline.ipynb
    ├── 02_custom_encoder.ipynb
    ├── 03_training_snn.ipynb
    ├── 04_clinical_assessment.ipynb
    └── 05_real_time_inference.ipynb
```

### Acceptance Criteria
- [ ] Integration test coverage >80%
- [ ] All public APIs have rustdoc with examples
- [ ] Tutorials run without errors on fresh install

---

## Dependency Graph

```
                                    ┌─────────────────┐
                                    │   Stream 8      │
                                    │ Testing & Docs  │
                                    └────────┬────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
    ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
    │   Stream 4      │           │   Stream 5      │           │   Stream 7      │
    │ Clinical Pipeline│◄─────────│ Normative DB    │           │ Integration     │
    └────────┬────────┘           └────────┬────────┘           └─────────────────┘
             │                             │                              ▲
             │                             │                              │
    ┌────────┴────────┐           ┌────────┴────────┐                     │
    │                 │           │                 │                     │
    ▼                 ▼           ▼                 │                     │
┌─────────┐    ┌─────────┐    ┌─────────┐          │                     │
│Stream 1 │    │Stream 2 │    │Stream 3 │          │                     │
│ECG/HRV  │    │Transforms│    │Neural   │          │                     │
└─────────┘    └─────────┘    └─────────┘          │                     │
                                                   │                     │
                                          ┌────────┴────────┐            │
                                          │   Stream 6      │────────────┘
                                          │ Synthetic Data  │
                                          └─────────────────┘

Legend:
  ───► Hard dependency (must complete first)
  - - ► Soft dependency (can start in parallel, merge later)
```

---

## Timeline (Parallel Execution)

```
Week 1:
├── Stream 1: ECG R-Peak, HRV Time-Domain ████████████
├── Stream 2: Wavelet, Hilbert ████████████
├── Stream 3: Reservoir Computing ████████████
├── Stream 4: Real-Time Pipeline (basics) ████████
├── Stream 5: Pediatric Norms ████████████
├── Stream 6: Augmentation Suite ████████████
├── Stream 7: WFDB, EDF ████████████
└── Stream 8: Integration test framework ████████

Week 2:
├── Stream 1: HRV Freq-Domain, QRS Morphology ████████████
├── Stream 2: ICA, EMD ████████████
├── Stream 3: Hebbian Learning, Pruning ████████████
├── Stream 4: Calibration, Explainability ████████████
├── Stream 5: Geriatric, Ethnic, MDC ████████████
├── Stream 6: Cohort, Longitudinal ████████████
├── Stream 7: BIDS, ONNX ████████████
└── Stream 8: Clinical validation tests ████████████

Week 3:
├── Stream 1: Arrhythmia Detection ████████
├── Stream 3: Distillation, Dendritic ████████████
├── Stream 4: Longitudinal, HIPAA ████████
├── Stream 6: Treatment, Comorbidity ████████
├── Stream 7: Neuromorphic Export ████████████
└── Stream 8: Documentation, Tutorials ████████████
```

---

## Resource Allocation

| Stream | Recommended Skills | Can Be Single Dev | Parallelizable Internally |
|--------|-------------------|-------------------|---------------------------|
| 1 | DSP, Cardiology | Yes | Limited (sequential) |
| 2 | DSP, Linear Algebra | Yes | Yes (3 independent tasks) |
| 3 | ML/DL, Neuroscience | No (2 devs ideal) | Yes (mostly independent) |
| 4 | Systems, ML Ops | No (2 devs ideal) | Partial |
| 5 | Statistics, Clinical | Yes | Yes (4 independent tasks) |
| 6 | Statistics, Simulation | Yes | Yes (5 independent tasks) |
| 7 | I/O, Formats | Yes | Yes (5 independent tasks) |
| 8 | Testing, Tech Writing | Yes | Yes (continuous) |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Stream 4 blocked by Stream 1/2 | Start pipeline scaffolding with mock components |
| ONNX export complexity | Start with simplified SNN subset, expand incrementally |
| Clinical validation data access | Use PhysioNet public datasets; document any gaps |
| Integration conflicts | Feature branches per stream, daily rebases |
| Scope creep | Strict task definitions, defer enhancements to v2 |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| All HIGH priority gaps resolved | 100% |
| All MEDIUM priority gaps resolved | >80% |
| Test coverage increase | +15% overall |
| Documentation coverage | 100% public APIs |
| Build time regression | <10% increase |
| No new critical bugs | 0 P0 bugs introduced |

---

## Next Steps

1. **Review this plan** - Identify any missing dependencies or incorrect estimates
2. **Assign streams** - Map developers/agents to streams
3. **Create feature branches** - `feature/stream-{N}-{name}` for each stream
4. **Set up CI gates** - Ensure streams can merge independently
5. **Begin parallel execution** - Start all 8 streams simultaneously
