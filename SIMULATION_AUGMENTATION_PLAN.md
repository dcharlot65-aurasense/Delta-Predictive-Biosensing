# Simulation Augmentation Plan

## Filling the Gaps in Delta-Predictive Biosensing Synthetic Data Generation

This document outlines the implementation plan for augmenting the `dpb-synth` crate to provide full simulation coverage for all algorithms in the DPB framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Phase A: Biomechanical Signals](#phase-a-biomechanical-signals)
3. [Phase B: Neural Enhancement](#phase-b-neural-enhancement)
4. [Phase C: Clinical Protocols](#phase-c-clinical-protocols)
5. [Implementation Standards](#implementation-standards)
6. [Testing Strategy](#testing-strategy)

---

## Overview

### Current Coverage Summary

| Domain | Algorithms | Generators | Gap |
|--------|-----------|------------|-----|
| ECG/Cardiac | 25+ | 25+ | 0% |
| EMG/Tremor | 15+ | 15+ | 0% |
| EDA/Autonomic | 10+ | 10+ | 0% |
| Gait/Pose | 20+ | 18+ | 5% |
| Voice/Speech | 25+ | 25+ | 0% |
| Eye Tracking | 15+ | 13+ | 10% |
| Hand/Fine Motor | 15+ | 15+ | 0% |
| **EEG/ERP** | 20+ | 10 | **50%** |
| **Force Dynamics** | 15+ | 0 | **100%** |
| **Balance/COP** | 12+ | 0 | **100%** |
| **Vestibular** | 10+ | 0 | **100%** |
| **Pain/Sensory** | 15+ | 0 | **100%** |
| **Sleep Features** | 12+ | 5 | **60%** |
| **Cardiopulmonary** | 10+ | 0 | **100%** |
| **Cognitive Tasks** | 15+ | 0 | **100%** |

### Priority Matrix

```
                    High Clinical Value
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    │   Force Dynamics     │   Balance/COP        │
    │   (Sarcopenia)       │   (Fall Risk)        │
    │                      │                      │
Low ├──────────────────────┼──────────────────────┤ High
Effort│                    │                      │ Effort
    │   Sleep Features     │   Vestibular         │
    │   Cognitive Tasks    │   Pain/Sensory       │
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           │
                    Low Clinical Value
```

---

## Phase A: Biomechanical Signals

**Focus:** Force dynamics, balance, and vestibular function generators

### A.1 Force Dynamics Generators

**Location:** `crates/dpb-synth/src/force/`

#### A.1.1 Ground Reaction Force Generator

```rust
// File: crates/dpb-synth/src/force/grf.rs

/// Ground reaction force generator based on biomechanical models
pub struct GrfGenerator {
    sample_rate: f64,
    body_mass: f64,
    gait_speed: f64,
}

impl GrfGenerator {
    /// Generate vertical GRF for walking (double-hump pattern)
    pub fn generate_walking(&self, duration: f64) -> GrfOutput;

    /// Generate GRF for running (single peak)
    pub fn generate_running(&self, duration: f64) -> GrfOutput;

    /// Generate jump landing GRF
    pub fn generate_jump_landing(&self, jump_height: f64) -> GrfOutput;

    /// Generate quiet standing COP
    pub fn generate_standing(&self, duration: f64) -> GrfOutput;
}

pub struct GrfOutput {
    pub vertical: Vec<f64>,      // Fz (body weights)
    pub anterior_posterior: Vec<f64>,  // Fy
    pub medial_lateral: Vec<f64>,      // Fx
    pub cop_x: Vec<f64>,         // Center of pressure X
    pub cop_y: Vec<f64>,         // Center of pressure Y
    pub ground_truth: GrfGroundTruth,
}
```

**Parameters:**
| Parameter | Range | Clinical Relevance |
|-----------|-------|-------------------|
| `body_mass` | 40-150 kg | Normalization |
| `gait_speed` | 0.5-2.0 m/s | Peak force timing |
| `step_width` | 0.05-0.20 m | ML forces |
| `asymmetry` | 0.0-0.5 | Limb differences |
| `noise_level` | 0-5% | Sensor noise |

#### A.1.2 Grip Strength Generator

```rust
// File: crates/dpb-synth/src/force/grip.rs

/// Grip strength force curve generator
pub struct GripStrengthGenerator {
    max_force: f64,        // Peak MVC (kg or N)
    rfd_coefficient: f64,  // Rate of force development
    fatigue_rate: f64,     // Force decay rate
}

impl GripStrengthGenerator {
    /// Generate maximal voluntary contraction trial
    pub fn generate_mvc(&self, hold_duration: f64) -> GripOutput;

    /// Generate sustained submaximal grip (50% MVC)
    pub fn generate_submaximal(&self, target_percent: f64, duration: f64) -> GripOutput;

    /// Generate rapid force pulses
    pub fn generate_pulses(&self, n_pulses: usize) -> GripOutput;

    /// Generate with fatigue (force decline over time)
    pub fn generate_with_fatigue(&self, duration: f64) -> GripOutput;
}
```

#### A.1.3 Rate of Force Development Generator

```rust
// File: crates/dpb-synth/src/force/rfd.rs

/// RFD profile generator for explosive strength assessment
pub struct RfdGenerator {
    peak_force: f64,
    time_to_peak: f64,  // ms
    rfd_type: RfdType,
}

pub enum RfdType {
    Ballistic,      // Maximum speed intention
    Controlled,     // Controlled ramp
    Submaximal,     // Percentage of max
}
```

### A.2 Balance/COP Generators

**Location:** `crates/dpb-synth/src/balance/`

#### A.2.1 Center of Pressure Generator

```rust
// File: crates/dpb-synth/src/balance/cop.rs

/// Center of pressure trajectory generator
pub struct CopGenerator {
    sample_rate: f64,
    sway_amplitude_ap: f64,  // Anterior-posterior (mm)
    sway_amplitude_ml: f64,  // Medial-lateral (mm)
    sway_frequency: f64,     // Dominant frequency (Hz)
}

impl CopGenerator {
    /// Generate quiet standing COP
    pub fn generate_quiet_standing(&self, duration: f64, eyes_open: bool) -> CopOutput;

    /// Generate tandem stance (heel-to-toe)
    pub fn generate_tandem(&self, duration: f64) -> CopOutput;

    /// Generate single-leg stance
    pub fn generate_single_leg(&self, duration: f64, side: Side) -> CopOutput;

    /// Generate with perturbation response
    pub fn generate_perturbation_response(&self, perturbation: Perturbation) -> CopOutput;
}

pub struct CopOutput {
    pub cop_x: Vec<f64>,     // ML position (mm)
    pub cop_y: Vec<f64>,     // AP position (mm)
    pub velocity_x: Vec<f64>,
    pub velocity_y: Vec<f64>,
    pub ground_truth: CopGroundTruth,
}

pub struct CopGroundTruth {
    pub sway_area: f64,
    pub path_length: f64,
    pub mean_velocity: f64,
    pub rms_ap: f64,
    pub rms_ml: f64,
}
```

#### A.2.2 Posturography Generator

```rust
// File: crates/dpb-synth/src/balance/posturography.rs

/// Computerized dynamic posturography (CDP) generator
pub struct PosturographyGenerator {
    sample_rate: f64,
    subject_height: f64,
}

impl PosturographyGenerator {
    /// Generate Sensory Organization Test conditions
    pub fn generate_sot(&self, condition: SotCondition, duration: f64) -> SotOutput;

    /// Generate Motor Control Test response
    pub fn generate_mct(&self, translation: PlatformTranslation) -> MctOutput;

    /// Generate Limits of Stability test
    pub fn generate_los(&self, target_direction: Direction) -> LosOutput;
}

pub enum SotCondition {
    Condition1,  // Eyes open, fixed surface, fixed surround
    Condition2,  // Eyes closed, fixed surface
    Condition3,  // Eyes open, fixed surface, sway-referenced surround
    Condition4,  // Eyes open, sway-referenced surface
    Condition5,  // Eyes closed, sway-referenced surface
    Condition6,  // Eyes open, sway-referenced surface and surround
}
```

#### A.2.3 Pathological Balance Generator

```rust
// File: crates/dpb-synth/src/balance/pathological.rs

/// Pathological balance pattern generator
pub struct PathologicalBalanceGenerator {
    pathology: BalancePathology,
    severity: f64,  // 0-1
}

pub enum BalancePathology {
    VestibularHypofunction,   // Increased sway, direction-specific
    ProprioceptiveDeficit,    // Eyes-closed worse than eyes-open
    CerebellarAtaxia,         // Irregular, high-frequency sway
    ParkinsonsDisease,        // Reduced limits of stability
    Neuropathy,               // Increased sway velocity
    FallRisk,                 // High variability, large excursions
}
```

### A.3 Vestibular Signal Generators

**Location:** `crates/dpb-synth/src/vestibular/`

#### A.3.1 VOR Generator

```rust
// File: crates/dpb-synth/src/vestibular/vor.rs

/// Vestibulo-ocular reflex (VOR) generator
pub struct VorGenerator {
    sample_rate: f64,
    vor_gain: f64,         // Normal: 0.9-1.1
    vor_asymmetry: f64,    // Left-right difference
}

impl VorGenerator {
    /// Generate head impulse test data
    pub fn generate_head_impulse(&self, direction: Direction, velocity: f64) -> VhitOutput;

    /// Generate sinusoidal head rotation response
    pub fn generate_sinusoidal(&self, frequency: f64, amplitude: f64) -> VorOutput;

    /// Generate with corrective saccades (pathological)
    pub fn generate_with_saccades(&self, catch_up: bool, covert: bool) -> VhitOutput;
}

pub struct VhitOutput {
    pub head_velocity: Vec<f64>,
    pub eye_velocity: Vec<f64>,
    pub eye_position: Vec<f64>,
    pub saccades: Vec<CorrectionSaccade>,
    pub ground_truth: VhitGroundTruth,
}
```

#### A.3.2 Nystagmus Generator

```rust
// File: crates/dpb-synth/src/vestibular/nystagmus.rs

/// Nystagmus waveform generator
pub struct NystagmusGenerator {
    nystagmus_type: NystagmusType,
    slow_phase_velocity: f64,  // deg/s
    frequency: f64,            // beats/s
}

pub enum NystagmusType {
    Jerk,           // Slow drift + fast reset
    Pendular,       // Sinusoidal oscillation
    Gaze,           // Direction-dependent
    Positional,     // Position-dependent
    Spontaneous,    // Without stimulation
}
```

---

## Phase B: Neural Enhancement

**Focus:** EEG/ERP components and sleep microstructure

### B.1 ERP Component Generators

**Location:** `crates/dpb-synth/src/neural/erp.rs`

```rust
/// Event-related potential generator
pub struct ErpGenerator {
    sample_rate: f64,
    noise_level: f64,
    n_trials: usize,
}

impl ErpGenerator {
    /// Generate P300 (oddball response)
    pub fn generate_p300(&self, latency: f64, amplitude: f64) -> ErpOutput;

    /// Generate N400 (semantic processing)
    pub fn generate_n400(&self, congruency: f64) -> ErpOutput;

    /// Generate MMN (mismatch negativity)
    pub fn generate_mmn(&self, deviant_probability: f64) -> ErpOutput;

    /// Generate N170 (face processing)
    pub fn generate_n170(&self, stimulus_type: FaceStimulus) -> ErpOutput;

    /// Generate oddball paradigm with targets/standards
    pub fn generate_oddball_paradigm(&self, n_trials: usize, target_prob: f64) -> OddballOutput;
}
```

### B.2 EEG Artifact Generators

```rust
// File: crates/dpb-synth/src/neural/artifacts.rs

/// EEG artifact generator for algorithm testing
pub struct EegArtifactGenerator {
    sample_rate: f64,
}

impl EegArtifactGenerator {
    /// Generate EOG (eye movement) artifacts
    pub fn generate_eog_artifact(&self, blink_rate: f64) -> Vec<f64>;

    /// Generate EMG (muscle) contamination
    pub fn generate_emg_artifact(&self, intensity: f64) -> Vec<f64>;

    /// Generate electrode pop/drift
    pub fn generate_electrode_artifact(&self) -> Vec<f64>;

    /// Inject artifacts into clean EEG
    pub fn inject_artifacts(&self, clean_eeg: &[f64], artifact_rate: f64) -> ArtifactOutput;
}
```

### B.3 Sleep Microstructure Generators

```rust
// File: crates/dpb-synth/src/neural/sleep.rs

/// Sleep microstructure generator
pub struct SleepMicrostructureGenerator {
    sample_rate: f64,
}

impl SleepMicrostructureGenerator {
    /// Generate sleep spindle (11-16 Hz burst)
    pub fn generate_spindle(&self, duration: f64, frequency: f64) -> SpindleOutput;

    /// Generate K-complex
    pub fn generate_k_complex(&self) -> KComplexOutput;

    /// Generate slow wave oscillation
    pub fn generate_slow_wave(&self, amplitude: f64) -> SlowWaveOutput;

    /// Generate complete N2 epoch with spindles
    pub fn generate_n2_epoch(&self, spindle_density: f64) -> EpochOutput;

    /// Generate actigraphy counts
    pub fn generate_actigraphy(&self, sleep_efficiency: f64) -> ActigraphyOutput;
}
```

---

## Phase C: Clinical Protocols

**Focus:** Pain assessment, cardiopulmonary, and cognitive task responses

### C.1 Pain/Sensory Generators

**Location:** `crates/dpb-synth/src/pain/`

```rust
// File: crates/dpb-synth/src/pain/qst.rs

/// Quantitative Sensory Testing (QST) generator
pub struct QstGenerator {
    sample_rate: f64,
}

impl QstGenerator {
    /// Generate pressure pain threshold ramp
    pub fn generate_ppt_ramp(&self, threshold_kpa: f64, rate: f64) -> PptOutput;

    /// Generate temporal summation protocol
    pub fn generate_temporal_summation(&self, n_stimuli: usize) -> TsOutput;

    /// Generate CPM protocol (conditioning + test)
    pub fn generate_cpm(&self, baseline: f64, conditioned: f64) -> CpmOutput;

    /// Generate pain-evoked autonomic response
    pub fn generate_pain_autonomic(&self, pain_intensity: f64) -> PainAutonomicOutput;
}

/// Proprioception testing generator
pub struct ProprioceptionGenerator {
    joint: Joint,
    error_magnitude: f64,  // degrees
}

impl ProprioceptionGenerator {
    /// Generate joint position sense test
    pub fn generate_repositioning(&self, target_angle: f64) -> RepositioningOutput;

    /// Generate threshold to detection of passive motion
    pub fn generate_ttdpm(&self, velocity: f64) -> TtdpmOutput;
}
```

### C.2 Cardiopulmonary Generators

**Location:** `crates/dpb-synth/src/cardiopulmonary/`

```rust
// File: crates/dpb-synth/src/cardiopulmonary/exercise.rs

/// Cardiopulmonary exercise test generator
pub struct CpetGenerator {
    sample_rate: f64,
    vo2max: f64,          // mL/kg/min
    anaerobic_threshold: f64,  // % of VO2max
}

impl CpetGenerator {
    /// Generate ramp protocol VO2/VCO2/VE
    pub fn generate_ramp_protocol(&self, duration: f64) -> CpetOutput;

    /// Generate breath-by-breath data
    pub fn generate_breath_by_breath(&self, duration: f64) -> BreathOutput;

    /// Generate ventilatory threshold crossings
    pub fn generate_with_thresholds(&self) -> ThresholdOutput;

    /// Generate HR recovery post-exercise
    pub fn generate_hr_recovery(&self, exercise_hr: f64) -> HrRecoveryOutput;
}

pub struct CpetOutput {
    pub time: Vec<f64>,
    pub vo2: Vec<f64>,        // mL/min
    pub vco2: Vec<f64>,       // mL/min
    pub ve: Vec<f64>,         // L/min
    pub hr: Vec<f64>,         // bpm
    pub rq: Vec<f64>,         // VCO2/VO2
    pub ground_truth: CpetGroundTruth,
}
```

### C.3 Cognitive Task Response Generators

**Location:** `crates/dpb-synth/src/cognitive/`

```rust
// File: crates/dpb-synth/src/cognitive/reaction_time.rs

/// Reaction time response generator
pub struct ReactionTimeGenerator {
    mean_rt: f64,
    std_rt: f64,
    lapse_rate: f64,
    anticipation_rate: f64,
}

impl ReactionTimeGenerator {
    /// Generate simple RT trial sequence
    pub fn generate_simple_rt(&self, n_trials: usize) -> RtOutput;

    /// Generate choice RT with compatibility effects
    pub fn generate_choice_rt(&self, n_choices: usize, n_trials: usize) -> RtOutput;

    /// Generate with vigilance decrement (fatigue over time)
    pub fn generate_with_fatigue(&self, n_trials: usize, fatigue_rate: f64) -> RtOutput;

    /// Generate ex-Gaussian distribution (realistic RT)
    pub fn generate_ex_gaussian(&self, mu: f64, sigma: f64, tau: f64, n: usize) -> Vec<f64>;
}

/// N-back working memory generator
pub struct NBackGenerator {
    n_level: usize,
    target_percentage: f64,
}

impl NBackGenerator {
    /// Generate N-back trial sequence with responses
    pub fn generate_sequence(&self, n_trials: usize) -> NBackOutput;

    /// Generate with load-dependent performance
    pub fn generate_with_load_effects(&self, capacity: f64) -> NBackOutput;
}

/// Continuous performance test generator
pub struct CptGenerator {
    attention_capacity: f64,
    impulsivity: f64,
}

impl CptGenerator {
    /// Generate CPT trial stream with responses
    pub fn generate_cpt(&self, duration_minutes: f64) -> CptOutput;

    /// Generate with ADHD-like patterns
    pub fn generate_adhd_pattern(&self, severity: f64) -> CptOutput;
}
```

---

## Implementation Standards

### Common Traits

All generators must implement:

```rust
/// Core synthetic generator trait
pub trait SyntheticGenerator {
    type Output;
    type Config;

    /// Create with configuration
    fn new(config: Self::Config) -> Self;

    /// Generate synthetic data
    fn generate(&self, duration: f64) -> Self::Output;

    /// Get ground truth annotations
    fn ground_truth(&self) -> &GroundTruth;

    /// Set random seed for reproducibility
    fn set_seed(&mut self, seed: u64);
}

/// Streaming generator for real-time simulation
pub trait StreamingGenerator: SyntheticGenerator {
    /// Generate next sample(s)
    fn next_sample(&mut self) -> Self::Output;

    /// Reset internal state
    fn reset(&mut self);
}
```

### Ground Truth Requirements

Every generator must provide:
1. **Event markers** - timestamps of discrete events
2. **Segment labels** - continuous state annotations
3. **Parameter values** - true underlying parameters
4. **Quality metrics** - expected algorithm outputs

### Parameterization Guidelines

1. **Severity scales** - Use 0-1 normalized scales
2. **Age effects** - Support age-adjusted parameters
3. **Pathology models** - Named disease patterns
4. **Noise models** - Sensor-realistic artifacts
5. **Reproducibility** - Seeded RNG throughout

---

## Testing Strategy

### Unit Tests

Each generator requires:
- Parameter range validation
- Output shape verification
- Ground truth consistency checks
- Reproducibility (same seed → same output)

### Integration Tests

- Algorithm accuracy with synthetic ground truth
- Multi-modal synchronization
- Streaming performance benchmarks
- Pathological pattern detection rates

### Validation Tests

```rust
#[test]
fn test_grf_walking_peaks() {
    let gen = GrfGenerator::new(80.0, 1.2);  // 80kg, 1.2 m/s
    let output = gen.generate_walking(5.0);

    // Vertical GRF should show double-hump pattern
    let peaks = find_peaks(&output.vertical);
    assert!(peaks.len() >= 8);  // ~2 peaks per step, ~4 steps

    // Peak force should be 1.0-1.3 BW for walking
    let max_bw = output.vertical.iter().cloned().fold(0.0, f64::max);
    assert!(max_bw >= 1.0 && max_bw <= 1.5);
}
```

---

## Summary

### New Generator Count by Phase

| Phase | New Generators | Cumulative |
|-------|----------------|------------|
| Current | 163 | 163 |
| Phase A | +35 | 198 |
| Phase B | +20 | 218 |
| Phase C | +25 | 243 |

### Files to Create

```
crates/dpb-synth/src/
├── force/
│   ├── mod.rs
│   ├── grf.rs          # Ground reaction force
│   ├── grip.rs         # Grip dynamometry
│   └── rfd.rs          # Rate of force development
├── balance/
│   ├── mod.rs
│   ├── cop.rs          # Center of pressure
│   ├── posturography.rs # CDP/SOT
│   └── pathological.rs  # Balance disorders
├── vestibular/
│   ├── mod.rs
│   ├── vor.rs          # VOR/vHIT
│   └── nystagmus.rs    # Nystagmus patterns
├── neural/
│   ├── erp.rs          # NEW: ERP components
│   ├── artifacts.rs    # NEW: EEG artifacts
│   └── sleep.rs        # NEW: Sleep microstructure
├── pain/
│   ├── mod.rs
│   ├── qst.rs          # Quantitative sensory testing
│   └── proprioception.rs
├── cardiopulmonary/
│   ├── mod.rs
│   └── exercise.rs     # CPET data
└── cognitive/
    ├── mod.rs
    ├── reaction_time.rs
    ├── working_memory.rs
    └── attention.rs
```

---

*Simulation Augmentation Plan generated: 2025-12-18*
*Reference: DPB Algorithm Implementation Plan v1.0*
