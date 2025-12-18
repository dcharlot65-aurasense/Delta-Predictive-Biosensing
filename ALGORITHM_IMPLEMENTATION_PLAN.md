# Algorithm Implementation Plan

## Filling the Gaps in Delta Predictive Biosensing

This document outlines the implementation plan for addressing the algorithm gaps identified in the gap analysis. The plan is organized into four phases with detailed specifications for each new algorithm module.

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Foundation Expansion](#phase-1-foundation-expansion)
3. [Phase 2: Clinical Enhancement](#phase-2-clinical-enhancement)
4. [Phase 3: Specialized Domains](#phase-3-specialized-domains)
5. [Phase 4: Integration & Validation](#phase-4-integration--validation)
6. [Implementation Details](#implementation-details)
7. [Dependencies & Prerequisites](#dependencies--prerequisites)
8. [Testing Strategy](#testing-strategy)

---

## Overview

### Current Coverage Summary

| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Peripheral Biosignals | 75% | 95% | 20% |
| Central/Cognitive | 10% | 80% | 70% |
| Force/Strength | 5% | 85% | 80% |
| Vestibular/Balance | 25% | 90% | 65% |

### Priority Matrix

```
                    High Impact
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    │   EEG Processing  │  Cognitive Tasks  │
    │   Force Dynamics  │  Vestibular       │
    │                   │                   │
Low ├───────────────────┼───────────────────┤ High
Effort│                 │                   │ Effort
    │   Sleep Staging   │  Pain Assessment  │
    │   Proprioception  │  Nerve Conduction │
    │                   │                   │
    └───────────────────┼───────────────────┘
                        │
                    Low Impact
```

---

## Phase 1: Foundation Expansion

**Duration:** First development cycle
**Focus:** Core capabilities that enable multiple downstream assessments

### 1.1 EEG Signal Processing Suite

**Location:** `crates/dpb-core/src/signal/eeg/`

#### 1.1.1 EEG Band Power Extraction

```rust
// File: crates/dpb-core/src/signal/eeg/bands.rs

/// EEG frequency bands following clinical conventions
pub struct EegBands {
    pub delta: (f64, f64),    // 0.5-4 Hz - Deep sleep
    pub theta: (f64, f64),    // 4-8 Hz - Drowsiness, meditation
    pub alpha: (f64, f64),    // 8-13 Hz - Relaxed wakefulness
    pub beta: (f64, f64),     // 13-30 Hz - Active thinking
    pub gamma: (f64, f64),    // 30-100 Hz - Cognitive processing
}

/// Extract band power from EEG signal
pub fn extract_band_power(signal: &[f64], sample_rate: f64, band: (f64, f64)) -> f64;

/// Compute all standard band powers
pub fn compute_band_powers(signal: &[f64], sample_rate: f64) -> BandPowers;

/// Theta/Beta ratio (ADHD biomarker)
pub fn theta_beta_ratio(signal: &[f64], sample_rate: f64) -> f64;

/// Alpha asymmetry (depression/anxiety marker)
pub fn alpha_asymmetry(left: &[f64], right: &[f64], sample_rate: f64) -> f64;
```

**Algorithms to implement:**
| Algorithm | Description | Clinical Use |
|-----------|-------------|--------------|
| `band_power_welch` | Welch's method for PSD estimation | Standard band power |
| `band_power_multitaper` | Multitaper spectral estimation | High-resolution spectra |
| `relative_band_power` | Normalized to total power | Cross-subject comparison |
| `band_power_topography` | Multi-channel mapping | Spatial analysis |

#### 1.1.2 EEG Artifact Removal

```rust
// File: crates/dpb-core/src/signal/eeg/artifacts.rs

/// Remove EOG (eye movement) artifacts using regression
pub fn remove_eog_artifacts(eeg: &[f64], eog: &[f64]) -> Vec<f64>;

/// Remove EMG artifacts using high-frequency filtering
pub fn remove_emg_artifacts(eeg: &[f64], sample_rate: f64) -> Vec<f64>;

/// Independent Component Analysis for artifact separation
pub fn ica_artifact_removal(channels: &[Vec<f64>], n_components: usize) -> Vec<Vec<f64>>;

/// Automatic artifact detection and marking
pub fn detect_artifacts(eeg: &[f64], sample_rate: f64) -> Vec<ArtifactSegment>;
```

#### 1.1.3 EEG Synthetic Generator

```rust
// File: crates/dpb-synth/src/neural/eeg.rs

pub struct EegGenerator {
    sample_rate: f64,
    channels: usize,
    band_powers: BandPowers,
    noise_level: f64,
}

impl EegGenerator {
    /// Generate resting-state EEG with specified band characteristics
    pub fn generate_resting_state(&self, duration_sec: f64) -> Vec<Vec<f64>>;

    /// Generate sleep EEG for specific stage
    pub fn generate_sleep_stage(&self, stage: SleepStage, duration_sec: f64) -> Vec<Vec<f64>>;

    /// Generate EEG with embedded artifacts
    pub fn generate_with_artifacts(&self, artifact_rate: f64) -> Vec<Vec<f64>>;

    /// Generate seizure-like activity
    pub fn generate_ictal_pattern(&self, seizure_type: SeizureType) -> Vec<Vec<f64>>;
}

pub enum SleepStage { Wake, N1, N2, N3, REM }
pub enum SeizureType { Absence, TonicClonic, Focal }
```

---

### 1.2 Cognitive Assessment Paradigms

**Location:** `crates/dpb-cognitive/src/`

#### 1.2.1 Reaction Time System

```rust
// File: crates/dpb-cognitive/src/reaction_time.rs

/// Simple reaction time paradigm
pub struct SimpleReactionTime {
    stimulus_intervals: Vec<f64>,  // Variable foreperiods
    response_window: f64,          // Max response time
}

/// Choice reaction time with multiple stimuli
pub struct ChoiceReactionTime {
    n_choices: usize,
    stimulus_probability: Vec<f64>,
    compatibility: ResponseCompatibility,
}

/// Metrics from reaction time testing
pub struct ReactionTimeMetrics {
    pub mean_rt: f64,
    pub median_rt: f64,
    pub std_rt: f64,
    pub coefficient_of_variation: f64,
    pub anticipations: usize,      // Too fast responses
    pub lapses: usize,             // Missed responses
    pub inverse_efficiency: f64,   // RT / accuracy
}

impl SimpleReactionTime {
    pub fn generate_trial_sequence(&self, n_trials: usize) -> Vec<Trial>;
    pub fn score_responses(&self, responses: &[Response]) -> ReactionTimeMetrics;
}
```

#### 1.2.2 Working Memory (N-Back)

```rust
// File: crates/dpb-cognitive/src/working_memory.rs

/// N-Back working memory task
pub struct NBackTask {
    n_level: usize,           // 1-back, 2-back, 3-back
    stimulus_duration: f64,
    inter_stimulus_interval: f64,
    target_percentage: f64,   // Proportion of targets
}

/// N-Back performance metrics
pub struct NBackMetrics {
    pub hits: usize,          // Correct target detections
    pub misses: usize,        // Missed targets
    pub false_alarms: usize,  // Incorrect target responses
    pub correct_rejections: usize,
    pub d_prime: f64,         // Signal detection sensitivity
    pub response_bias: f64,   // Response criterion
    pub mean_rt_hits: f64,
}

impl NBackTask {
    pub fn generate_sequence(&self, n_trials: usize) -> Vec<Stimulus>;
    pub fn score_responses(&self, sequence: &[Stimulus], responses: &[Response]) -> NBackMetrics;
}
```

#### 1.2.3 Attention Tasks

```rust
// File: crates/dpb-cognitive/src/attention.rs

/// Continuous Performance Test (CPT)
pub struct ContinuousPerformanceTest {
    pub variant: CptVariant,  // CPT-II, CPT-3, TOVA style
    pub duration_minutes: f64,
    pub isi_ms: f64,
    pub target_char: char,
}

pub enum CptVariant {
    CptII,      // Press for target
    CptX,       // Press for X after A
    Tova,       // Press for target, inhibit non-target
}

/// Stroop task
pub struct StroopTask {
    pub conditions: Vec<StroopCondition>,
    pub trials_per_condition: usize,
}

pub enum StroopCondition {
    Congruent,      // RED in red ink
    Incongruent,    // RED in blue ink
    Neutral,        // XXX in red ink
}

/// Trail Making Test (digital)
pub struct TrailMakingTest {
    pub part: TrailPart,
    pub n_targets: usize,
}

pub enum TrailPart {
    PartA,  // Numbers only (1-2-3...)
    PartB,  // Alternating (1-A-2-B...)
}
```

---

### 1.3 Force & Dynamometry

**Location:** `crates/dpb-core/src/biomechanics/force/`

#### 1.3.1 Grip Strength Dynamics

```rust
// File: crates/dpb-core/src/biomechanics/force/grip.rs

/// Grip strength measurement and analysis
pub struct GripStrengthAnalyzer {
    sample_rate: f64,
}

impl GripStrengthAnalyzer {
    /// Extract peak force from grip trial
    pub fn peak_force(&self, force_curve: &[f64]) -> f64;

    /// Rate of force development (N/s)
    pub fn rate_of_force_development(&self, force_curve: &[f64]) -> f64;

    /// Time to peak force
    pub fn time_to_peak(&self, force_curve: &[f64]) -> f64;

    /// Force steadiness (CV during sustained grip)
    pub fn force_steadiness(&self, force_curve: &[f64], target_force: f64) -> f64;

    /// Fatigue index (force decline over time)
    pub fn fatigue_index(&self, force_curve: &[f64], window_sec: f64) -> f64;
}

/// Grip strength norms by age and sex
pub struct GripStrengthNorms;

impl GripStrengthNorms {
    pub fn percentile(&self, force_kg: f64, age: u8, sex: Sex) -> f64;
    pub fn sarcopenia_cutoff(&self, sex: Sex) -> f64;  // <27kg M, <16kg F
}
```

#### 1.3.2 Ground Reaction Forces

```rust
// File: crates/dpb-core/src/biomechanics/force/grf.rs

/// Ground reaction force analysis
pub struct GroundReactionForce {
    pub vertical: Vec<f64>,
    pub anterior_posterior: Vec<f64>,
    pub medial_lateral: Vec<f64>,
    pub sample_rate: f64,
}

impl GroundReactionForce {
    /// Peak vertical force (body weights)
    pub fn peak_vertical_force(&self, body_mass: f64) -> f64;

    /// Loading rate (BW/s)
    pub fn loading_rate(&self, body_mass: f64) -> f64;

    /// Impulse calculation
    pub fn impulse(&self, phase: GaitPhase) -> Vector3;

    /// Center of pressure trajectory
    pub fn center_of_pressure(&self) -> Vec<(f64, f64)>;

    /// Symmetry index between limbs
    pub fn symmetry_index(left: &Self, right: &Self) -> f64;
}

/// Force plate data generator
pub struct ForcePlateGenerator {
    sample_rate: f64,
    noise_level: f64,
}

impl ForcePlateGenerator {
    pub fn generate_walking_grf(&self, gait_speed: f64, body_mass: f64) -> GroundReactionForce;
    pub fn generate_jump_landing(&self, jump_height: f64, body_mass: f64) -> GroundReactionForce;
    pub fn generate_balance_standing(&self, duration: f64, sway_amplitude: f64) -> GroundReactionForce;
}
```

#### 1.3.3 Rate of Force Development

```rust
// File: crates/dpb-core/src/biomechanics/force/rfd.rs

/// Rate of force development analysis
pub struct RfdAnalyzer {
    sample_rate: f64,
}

impl RfdAnalyzer {
    /// Peak RFD (maximum slope)
    pub fn peak_rfd(&self, force: &[f64]) -> f64;

    /// RFD at specific time windows (0-50ms, 0-100ms, 0-200ms)
    pub fn rfd_at_intervals(&self, force: &[f64]) -> RfdIntervals;

    /// Time to specific force levels (25%, 50%, 75%, 90% peak)
    pub fn time_to_force_levels(&self, force: &[f64]) -> ForceTiming;

    /// Contractile impulse
    pub fn contractile_impulse(&self, force: &[f64], time_window: f64) -> f64;
}

pub struct RfdIntervals {
    pub rfd_0_50: f64,
    pub rfd_0_100: f64,
    pub rfd_0_200: f64,
    pub rfd_100_200: f64,
}
```

---

### 1.4 Center of Pressure Analysis

**Location:** `crates/dpb-core/src/biomechanics/balance/`

```rust
// File: crates/dpb-core/src/biomechanics/balance/cop.rs

/// Center of pressure analysis for balance assessment
pub struct CopAnalyzer {
    sample_rate: f64,
}

impl CopAnalyzer {
    /// Sway area (95% confidence ellipse)
    pub fn sway_area(&self, cop_x: &[f64], cop_y: &[f64]) -> f64;

    /// Sway path length (total distance traveled)
    pub fn path_length(&self, cop_x: &[f64], cop_y: &[f64]) -> f64;

    /// Mean sway velocity
    pub fn mean_velocity(&self, cop_x: &[f64], cop_y: &[f64]) -> f64;

    /// RMS displacement in AP and ML directions
    pub fn rms_displacement(&self, cop_x: &[f64], cop_y: &[f64]) -> (f64, f64);

    /// Frequency analysis of sway
    pub fn sway_frequency(&self, cop: &[f64]) -> SwayFrequency;

    /// Stabilogram diffusion analysis
    pub fn diffusion_analysis(&self, cop_x: &[f64], cop_y: &[f64]) -> DiffusionPlot;

    /// Romberg quotient (eyes closed / eyes open)
    pub fn romberg_quotient(eyes_open: &CopMetrics, eyes_closed: &CopMetrics) -> f64;
}

pub struct CopMetrics {
    pub sway_area: f64,
    pub path_length: f64,
    pub mean_velocity: f64,
    pub rms_ap: f64,
    pub rms_ml: f64,
    pub mean_frequency: f64,
}
```

---

## Phase 2: Clinical Enhancement

**Duration:** Second development cycle
**Focus:** Advanced clinical algorithms building on Phase 1

### 2.1 Event-Related Potentials (ERPs)

**Location:** `crates/dpb-core/src/signal/eeg/erp.rs`

```rust
/// Event-related potential extraction and analysis
pub struct ErpAnalyzer {
    sample_rate: f64,
    baseline_window: (f64, f64),  // Pre-stimulus baseline
}

impl ErpAnalyzer {
    /// Extract ERP by averaging time-locked epochs
    pub fn extract_erp(&self,
        eeg: &[f64],
        event_times: &[f64],
        epoch_window: (f64, f64)
    ) -> Erp;

    /// Detect P300 component
    pub fn detect_p300(&self, erp: &Erp) -> Option<ErpComponent>;

    /// Detect N400 component (semantic processing)
    pub fn detect_n400(&self, erp: &Erp) -> Option<ErpComponent>;

    /// Detect MMN (mismatch negativity)
    pub fn detect_mmn(&self, standard_erp: &Erp, deviant_erp: &Erp) -> Option<ErpComponent>;

    /// Detect N170 (face processing)
    pub fn detect_n170(&self, erp: &Erp) -> Option<ErpComponent>;
}

pub struct ErpComponent {
    pub latency_ms: f64,
    pub amplitude_uv: f64,
    pub peak_channel: String,
}

/// ERP Generator for synthetic data
pub struct ErpGenerator {
    sample_rate: f64,
}

impl ErpGenerator {
    pub fn generate_p300(&self, amplitude: f64, latency: f64, jitter: f64) -> Vec<f64>;
    pub fn generate_oddball_paradigm(&self, n_trials: usize, target_prob: f64) -> ErpDataset;
}
```

### 2.2 Vestibular Metrics

**Location:** `crates/dpb-core/src/vestibular/`

```rust
// File: crates/dpb-core/src/vestibular/vor.rs

/// Vestibular-Ocular Reflex analysis
pub struct VorAnalyzer {
    sample_rate: f64,
}

impl VorAnalyzer {
    /// Calculate VOR gain (eye velocity / head velocity)
    pub fn vor_gain(&self,
        eye_velocity: &[f64],
        head_velocity: &[f64]
    ) -> f64;

    /// VOR gain asymmetry between directions
    pub fn vor_asymmetry(&self,
        leftward_gain: f64,
        rightward_gain: f64
    ) -> f64;

    /// Detect corrective saccades (overt/covert)
    pub fn detect_corrective_saccades(&self,
        eye_position: &[f64],
        head_impulse_times: &[f64]
    ) -> Vec<CorrectionSaccade>;

    /// Video Head Impulse Test analysis
    pub fn analyze_vhit(&self, trial: &VhitTrial) -> VhitResult;
}

pub struct VhitResult {
    pub gain: f64,
    pub gain_asymmetry: f64,
    pub covert_saccades: Vec<CorrectionSaccade>,
    pub overt_saccades: Vec<CorrectionSaccade>,
    pub pr_score: f64,  // Presence of refixation saccades
}
```

```rust
// File: crates/dpb-core/src/vestibular/posturography.rs

/// Computerized Dynamic Posturography
pub struct DynamicPosturography {
    sample_rate: f64,
}

impl DynamicPosturography {
    /// Sensory Organization Test (SOT) scoring
    pub fn sensory_organization_test(&self,
        conditions: &[SotCondition]
    ) -> SotResults;

    /// Motor Control Test (MCT)
    pub fn motor_control_test(&self,
        translations: &[PlatformTranslation]
    ) -> MctResults;

    /// Adaptation Test
    pub fn adaptation_test(&self,
        rotations: &[PlatformRotation]
    ) -> AdaptationResults;

    /// Limits of Stability (LOS)
    pub fn limits_of_stability(&self,
        cop_trajectories: &[CopTrajectory]
    ) -> LosResults;
}

pub enum SotCondition {
    Condition1,  // Eyes open, fixed surface, fixed surround
    Condition2,  // Eyes closed, fixed surface
    Condition3,  // Eyes open, fixed surface, sway-referenced surround
    Condition4,  // Eyes open, sway-referenced surface
    Condition5,  // Eyes closed, sway-referenced surface
    Condition6,  // Eyes open, sway-referenced surface and surround
}

pub struct SotResults {
    pub equilibrium_scores: [f64; 6],
    pub composite_score: f64,
    pub sensory_analysis: SensoryAnalysis,
}

pub struct SensoryAnalysis {
    pub somatosensory_ratio: f64,  // Condition 2 / Condition 1
    pub visual_ratio: f64,          // Condition 4 / Condition 1
    pub vestibular_ratio: f64,      // Condition 5 / Condition 1
    pub preference_ratio: f64,      // (Cond3 + Cond6) / (Cond2 + Cond5)
}
```

### 2.3 Sleep Staging (Multi-Modal)

**Location:** `crates/dpb-core/src/sleep/`

```rust
// File: crates/dpb-core/src/sleep/staging.rs

/// Sleep stage classification
pub struct SleepStager {
    epoch_duration: f64,  // Typically 30 seconds
}

impl SleepStager {
    /// Stage sleep from EEG (gold standard)
    pub fn stage_from_eeg(&self, eeg: &[f64], sample_rate: f64) -> Vec<SleepStage>;

    /// Stage sleep from HRV (wearable-compatible)
    pub fn stage_from_hrv(&self, rr_intervals: &[f64]) -> Vec<SleepStage>;

    /// Stage sleep from actigraphy
    pub fn stage_from_actigraphy(&self, activity_counts: &[f64]) -> Vec<SleepStage>;

    /// Multi-modal sleep staging (fusion)
    pub fn stage_multimodal(&self,
        hrv: Option<&[f64]>,
        actigraphy: Option<&[f64]>,
        respiratory: Option<&[f64]>
    ) -> Vec<SleepStage>;
}

/// Sleep architecture metrics
pub struct SleepArchitecture {
    pub total_sleep_time: f64,
    pub sleep_efficiency: f64,
    pub sleep_onset_latency: f64,
    pub wake_after_sleep_onset: f64,
    pub n1_percent: f64,
    pub n2_percent: f64,
    pub n3_percent: f64,
    pub rem_percent: f64,
    pub rem_latency: f64,
    pub awakenings: usize,
}

/// Sleep feature extraction for staging
pub struct SleepFeatureExtractor;

impl SleepFeatureExtractor {
    /// Detect sleep spindles (N2 marker)
    pub fn detect_spindles(&self, eeg: &[f64], sample_rate: f64) -> Vec<Spindle>;

    /// Detect K-complexes (N2 marker)
    pub fn detect_k_complexes(&self, eeg: &[f64], sample_rate: f64) -> Vec<KComplex>;

    /// Detect slow wave activity (N3 marker)
    pub fn detect_slow_waves(&self, eeg: &[f64], sample_rate: f64) -> Vec<SlowWave>;

    /// Detect REM from EOG
    pub fn detect_rem_from_eog(&self, eog: &[f64], sample_rate: f64) -> Vec<RemEpisode>;
}
```

### 2.4 Advanced Cognitive Tasks

**Location:** `crates/dpb-cognitive/src/`

```rust
// File: crates/dpb-cognitive/src/executive.rs

/// Go/No-Go task for response inhibition
pub struct GoNoGoTask {
    pub go_probability: f64,
    pub stimulus_duration: f64,
    pub response_window: f64,
}

pub struct GoNoGoMetrics {
    pub commission_errors: usize,  // False alarms (pressing on No-Go)
    pub omission_errors: usize,    // Misses (not pressing on Go)
    pub mean_rt_go: f64,
    pub d_prime: f64,
}

/// Flanker task for selective attention
pub struct FlankerTask {
    pub conditions: Vec<FlankerCondition>,
    pub stimulus_duration: f64,
}

pub enum FlankerCondition {
    Congruent,    // >>>>> or <<<<<
    Incongruent,  // >><>> or <<><<
    Neutral,      // --<-- or -->--
}

pub struct FlankerMetrics {
    pub congruent_rt: f64,
    pub incongruent_rt: f64,
    pub flanker_effect: f64,  // Incongruent - Congruent RT
    pub conflict_adaptation: f64,
}

/// Wisconsin Card Sorting Test
pub struct WisconsinCardSort {
    pub max_categories: usize,
    pub cards_per_category: usize,
}

pub struct WcstMetrics {
    pub categories_completed: usize,
    pub perseverative_errors: usize,
    pub non_perseverative_errors: usize,
    pub trials_to_first_category: usize,
    pub conceptual_level_responses: usize,
}
```

---

## Phase 3: Specialized Domains

**Duration:** Third development cycle
**Focus:** Domain-specific algorithms for specialized assessments

### 3.1 Proprioception Assessment

**Location:** `crates/dpb-core/src/somatosensory/`

```rust
// File: crates/dpb-core/src/somatosensory/proprioception.rs

/// Joint position sense testing
pub struct JointPositionSense {
    joint: Joint,
    sample_rate: f64,
}

impl JointPositionSense {
    /// Passive joint repositioning error
    pub fn repositioning_error(&self,
        target_angle: f64,
        reproduced_angle: f64
    ) -> f64;

    /// Threshold to detection of passive motion (TTDPM)
    pub fn detection_threshold(&self,
        movement_velocity: f64,
        detection_angle: f64
    ) -> f64;

    /// Active vs passive repositioning comparison
    pub fn active_passive_comparison(&self,
        active_errors: &[f64],
        passive_errors: &[f64]
    ) -> ProprioceptiveProfile;
}

/// Vibration sense testing
pub struct VibrationSense {
    frequency: f64,  // Typically 128 Hz tuning fork
}

impl VibrationSense {
    /// Vibration perception threshold
    pub fn perception_threshold(&self,
        intensity_curve: &[f64]
    ) -> f64;

    /// Compare to age-matched norms
    pub fn percentile(&self, threshold: f64, age: u8) -> f64;

    /// Peripheral neuropathy screening
    pub fn neuropathy_screening(&self,
        threshold: f64,
        location: BodyLocation
    ) -> NeuropathyRisk;
}
```

### 3.2 Pain Assessment

**Location:** `crates/dpb-core/src/pain/`

```rust
// File: crates/dpb-core/src/pain/assessment.rs

/// Pressure pain threshold measurement
pub struct PressurePainThreshold {
    sample_rate: f64,
}

impl PressurePainThreshold {
    /// Calculate pain threshold from pressure ramp
    pub fn calculate_threshold(&self,
        pressure: &[f64],
        pain_onset_time: f64
    ) -> f64;

    /// Pain tolerance level
    pub fn calculate_tolerance(&self,
        pressure: &[f64],
        tolerance_time: f64
    ) -> f64;

    /// Temporal summation (wind-up)
    pub fn temporal_summation(&self,
        pain_ratings: &[f64],
        stimulus_times: &[f64]
    ) -> TemporalSummationProfile;
}

/// Conditioned Pain Modulation
pub struct ConditionedPainModulation {
    conditioning_stimulus: StimulusType,
    test_stimulus: StimulusType,
}

impl ConditionedPainModulation {
    /// Calculate CPM effect (inhibition magnitude)
    pub fn cpm_effect(&self,
        baseline_threshold: f64,
        conditioned_threshold: f64
    ) -> f64;

    /// CPM efficiency ratio
    pub fn cpm_efficiency(&self,
        cpm_effect: f64,
        conditioning_pain: f64
    ) -> f64;
}

/// Pain-related autonomic responses
pub struct PainAutonomicAnalyzer;

impl PainAutonomicAnalyzer {
    /// Skin conductance response to pain
    pub fn scr_to_pain(&self, eda: &[f64], pain_onset: f64) -> ScrMetrics;

    /// Heart rate response to pain
    pub fn hr_response(&self, rr_intervals: &[f64], pain_onset: f64) -> HrResponse;

    /// Pupil dilation response
    pub fn pupil_response(&self, pupil_diameter: &[f64], pain_onset: f64) -> PupilResponse;
}
```

### 3.3 ADHD-Specific Paradigms

**Location:** `crates/dpb-cognitive/src/adhd/`

```rust
// File: crates/dpb-cognitive/src/adhd/paradigms.rs

/// QbTest-style continuous performance test
pub struct QbTest {
    duration_minutes: f64,
    target_probability: f64,
}

impl QbTest {
    /// Generate test stimuli
    pub fn generate_stimuli(&self) -> Vec<Stimulus>;

    /// Score attention metrics
    pub fn score_attention(&self, responses: &[Response]) -> AttentionMetrics;

    /// Score impulsivity metrics
    pub fn score_impulsivity(&self, responses: &[Response]) -> ImpulsivityMetrics;

    /// Score activity (from motion tracking)
    pub fn score_activity(&self, head_motion: &[Vector3]) -> ActivityMetrics;
}

pub struct AttentionMetrics {
    pub omission_errors: usize,
    pub reaction_time: f64,
    pub rt_variability: f64,
    pub normative_percentile: f64,
}

pub struct ImpulsivityMetrics {
    pub commission_errors: usize,
    pub anticipatory_responses: usize,
    pub multiresponses: usize,
    pub normative_percentile: f64,
}

pub struct ActivityMetrics {
    pub distance: f64,
    pub area: f64,
    pub microevents: usize,
    pub normative_percentile: f64,
}

/// ADHD-specific eye tracking patterns
pub struct AdhdEyeTracking;

impl AdhdEyeTracking {
    /// Fixation stability during sustained attention
    pub fn fixation_stability(&self, gaze: &[GazePoint]) -> FixationStability;

    /// Saccade metrics during reading
    pub fn reading_saccades(&self, gaze: &[GazePoint]) -> ReadingSaccadeMetrics;

    /// Anticipatory saccade rate
    pub fn anticipatory_saccades(&self,
        gaze: &[GazePoint],
        target_times: &[f64]
    ) -> f64;
}
```

### 3.4 Cardiopulmonary Exercise Estimation

**Location:** `crates/dpb-core/src/cardiopulmonary/`

```rust
// File: crates/dpb-core/src/cardiopulmonary/vo2.rs

/// VO2 estimation from heart rate and other signals
pub struct Vo2Estimator {
    method: Vo2Method,
}

pub enum Vo2Method {
    HrReserve,      // Based on HR reserve
    HrPulseOxygen,  // HR + SpO2
    Multivariate,   // Multiple signals
}

impl Vo2Estimator {
    /// Estimate VO2 from heart rate
    pub fn estimate_from_hr(&self,
        heart_rate: f64,
        hr_max: f64,
        hr_rest: f64,
        age: u8,
        sex: Sex
    ) -> f64;

    /// Estimate VO2max from submaximal test
    pub fn estimate_vo2max_submaximal(&self,
        hr_workload_pairs: &[(f64, f64)],
        hr_max: f64
    ) -> f64;

    /// Rockport walk test estimation
    pub fn rockport_estimation(&self,
        time_minutes: f64,
        hr_end: f64,
        weight_kg: f64,
        age: u8,
        sex: Sex
    ) -> f64;
}

/// Ventilatory threshold detection
pub struct VentilatoryThreshold;

impl VentilatoryThreshold {
    /// Detect VT1 from respiratory data
    pub fn detect_vt1(&self,
        ve: &[f64],      // Ventilation
        vo2: &[f64],     // Oxygen uptake
        vco2: &[f64]     // CO2 output
    ) -> Option<f64>;

    /// Detect VT2 / respiratory compensation point
    pub fn detect_vt2(&self,
        ve: &[f64],
        vco2: &[f64]
    ) -> Option<f64>;

    /// Heart rate at ventilatory thresholds
    pub fn hr_at_thresholds(&self,
        hr: &[f64],
        vt1: f64,
        vt2: f64
    ) -> (f64, f64);
}
```

---

## Phase 4: Integration & Validation

**Duration:** Fourth development cycle
**Focus:** Cross-modal integration, normative databases, validation

### 4.1 Cross-Modal Cognitive-Motor Fusion

**Location:** `crates/dpb-snn/src/fusion/cognitive_motor.rs`

```rust
/// Cognitive-motor integration for comprehensive assessment
pub struct CognitiveMotorFusion {
    cognitive_weight: f64,
    motor_weight: f64,
}

impl CognitiveMotorFusion {
    /// Fuse cognitive and motor assessments
    pub fn fuse(&self,
        cognitive_metrics: &CognitiveProfile,
        motor_metrics: &MotorProfile
    ) -> IntegratedAssessment;

    /// Detect cognitive-motor dissociation
    pub fn detect_dissociation(&self,
        cognitive: &CognitiveProfile,
        motor: &MotorProfile
    ) -> Option<DissociationPattern>;

    /// Longitudinal change detection
    pub fn detect_change(&self,
        baseline: &IntegratedAssessment,
        followup: &IntegratedAssessment
    ) -> ChangeMetrics;
}
```

### 4.2 Normative Database

**Location:** `crates/dpb-norms/src/`

```rust
/// Comprehensive normative database
pub struct NormativeDatabase {
    age_range: (u8, u8),
    sample_size: usize,
}

impl NormativeDatabase {
    /// Get percentile for a metric
    pub fn percentile(&self,
        metric: MetricType,
        value: f64,
        demographics: &Demographics
    ) -> f64;

    /// Get z-score
    pub fn z_score(&self,
        metric: MetricType,
        value: f64,
        demographics: &Demographics
    ) -> f64;

    /// Get reference range (5th-95th percentile)
    pub fn reference_range(&self,
        metric: MetricType,
        demographics: &Demographics
    ) -> (f64, f64);

    /// Minimal detectable change
    pub fn minimal_detectable_change(&self,
        metric: MetricType
    ) -> f64;
}

pub struct Demographics {
    pub age: u8,
    pub sex: Sex,
    pub education_years: Option<u8>,
    pub ethnicity: Option<Ethnicity>,
    pub handedness: Option<Handedness>,
}
```

### 4.3 Clinical Validation Metrics

**Location:** `crates/dpb-core/src/validation/`

```rust
/// Clinical validation suite
pub struct ClinicalValidation;

impl ClinicalValidation {
    /// Test-retest reliability
    pub fn test_retest_reliability(&self,
        test1: &[f64],
        test2: &[f64]
    ) -> ReliabilityMetrics;

    /// Inter-rater reliability (for multi-assessor)
    pub fn inter_rater_reliability(&self,
        rater_scores: &[Vec<f64>]
    ) -> f64;  // ICC

    /// Criterion validity against gold standard
    pub fn criterion_validity(&self,
        algorithm_scores: &[f64],
        gold_standard: &[f64]
    ) -> ValidityMetrics;

    /// Sensitivity to change
    pub fn sensitivity_to_change(&self,
        pre_treatment: &[f64],
        post_treatment: &[f64]
    ) -> ChangeMetrics;

    /// Receiver Operating Characteristic
    pub fn roc_analysis(&self,
        scores: &[f64],
        labels: &[bool]
    ) -> RocCurve;
}

pub struct ReliabilityMetrics {
    pub icc: f64,
    pub sem: f64,  // Standard error of measurement
    pub mdc: f64,  // Minimal detectable change
    pub coefficient_of_variation: f64,
}

pub struct ValidityMetrics {
    pub pearson_r: f64,
    pub spearman_rho: f64,
    pub bland_altman: BlandAltmanResult,
    pub regression_equation: (f64, f64),  // slope, intercept
}
```

---

## Implementation Details

### New Crate Structure

```
crates/
├── dpb-cognitive/           # NEW: Cognitive assessment paradigms
│   └── src/
│       ├── lib.rs
│       ├── reaction_time.rs
│       ├── working_memory.rs
│       ├── attention.rs
│       ├── executive.rs
│       └── adhd/
│           └── paradigms.rs
│
├── dpb-norms/              # NEW: Normative databases
│   └── src/
│       ├── lib.rs
│       ├── database.rs
│       └── demographics.rs
│
├── dpb-core/
│   └── src/
│       ├── signal/
│       │   └── eeg/         # NEW: EEG processing
│       │       ├── mod.rs
│       │       ├── bands.rs
│       │       ├── artifacts.rs
│       │       └── erp.rs
│       │
│       ├── biomechanics/
│       │   ├── force/       # NEW: Force dynamics
│       │   │   ├── mod.rs
│       │   │   ├── grip.rs
│       │   │   ├── grf.rs
│       │   │   └── rfd.rs
│       │   │
│       │   └── balance/     # NEW: Balance assessment
│       │       ├── mod.rs
│       │       └── cop.rs
│       │
│       ├── vestibular/      # NEW: Vestibular function
│       │   ├── mod.rs
│       │   ├── vor.rs
│       │   └── posturography.rs
│       │
│       ├── somatosensory/   # NEW: Proprioception & sensation
│       │   ├── mod.rs
│       │   ├── proprioception.rs
│       │   └── vibration.rs
│       │
│       ├── pain/            # NEW: Pain assessment
│       │   ├── mod.rs
│       │   └── assessment.rs
│       │
│       ├── cardiopulmonary/ # NEW: Exercise physiology
│       │   ├── mod.rs
│       │   └── vo2.rs
│       │
│       ├── sleep/           # NEW: Sleep assessment
│       │   ├── mod.rs
│       │   └── staging.rs
│       │
│       └── validation/      # NEW: Clinical validation
│           ├── mod.rs
│           └── metrics.rs
│
└── dpb-synth/
    └── src/
        └── neural/          # NEW: Neural signal synthesis
            ├── mod.rs
            └── eeg.rs
```

### Cargo.toml Additions

```toml
# New crate: dpb-cognitive
[package]
name = "dpb-cognitive"
version = "0.1.0"
edition = "2021"

[dependencies]
dpb-core = { path = "../dpb-core" }
rand = "0.8"
statrs = "0.16"

# New crate: dpb-norms
[package]
name = "dpb-norms"
version = "0.1.0"
edition = "2021"

[dependencies]
dpb-core = { path = "../dpb-core" }
serde = { version = "1.0", features = ["derive"] }
```

---

## Dependencies & Prerequisites

### External Dependencies

| Dependency | Purpose | Phase |
|------------|---------|-------|
| `rustfft` | FFT for EEG band extraction | Phase 1 |
| `nalgebra` | Linear algebra for ICA | Phase 1 |
| `statrs` | Statistical distributions | Phase 1 |
| `rand` | Random number generation | Phase 1 |
| `ndarray` | Multi-dimensional arrays | Phase 2 |
| `serde` | Serialization for norms | Phase 4 |

### Internal Dependencies

```
Phase 1:
  dpb-core/signal/eeg → dpb-core/signal/fft (existing)
  dpb-cognitive → dpb-core (new dependency)

Phase 2:
  dpb-core/sleep → dpb-core/signal/eeg
  dpb-core/vestibular → dpb-core/signal/filter

Phase 3:
  dpb-cognitive/adhd → dpb-cognitive
  dpb-core/pain → dpb-core/signal (EDA)

Phase 4:
  dpb-norms → all metric modules
  dpb-snn/fusion/cognitive_motor → dpb-cognitive + dpb-core
```

---

## Testing Strategy

### Unit Tests

Each new module requires:
- Basic functionality tests
- Edge case handling
- Known-value tests against published algorithms

### Integration Tests

```rust
// tests/integration/cognitive_motor_integration.rs

#[test]
fn test_comprehensive_assessment() {
    // Generate synthetic data
    let eeg = EegGenerator::default().generate_resting_state(60.0);
    let gait = GaitCycleGenerator::default().generate(10.0);
    let reaction_times = SimpleReactionTime::default().generate_trial_sequence(50);

    // Run assessments
    let eeg_metrics = EegBands::default().compute_band_powers(&eeg[0], 256.0);
    let gait_metrics = GaitAnalyzer::default().analyze(&gait);
    let rt_metrics = SimpleReactionTime::default().score_responses(&reaction_times);

    // Fuse results
    let integrated = CognitiveMotorFusion::default().fuse(
        &CognitiveProfile::from_rt(&rt_metrics),
        &MotorProfile::from_gait(&gait_metrics)
    );

    assert!(integrated.composite_score > 0.0);
}
```

### Clinical Validation Tests

```rust
// tests/validation/test_retest.rs

#[test]
fn test_reaction_time_reliability() {
    // Simulate test-retest data with known ICC
    let (test1, test2) = generate_test_retest_data(icc: 0.85, n: 100);

    let reliability = ClinicalValidation::test_retest_reliability(&test1, &test2);

    assert!((reliability.icc - 0.85).abs() < 0.05);
}
```

---

## Summary

### Algorithm Count by Phase

| Phase | New Algorithms | Cumulative Total |
|-------|----------------|------------------|
| Current | 170 | 170 |
| Phase 1 | +45 | 215 |
| Phase 2 | +35 | 250 |
| Phase 3 | +30 | 280 |
| Phase 4 | +20 | 300 |

### Coverage Improvement

| Domain | Current | After Phase 4 |
|--------|---------|---------------|
| Peripheral Biosignals | 75% | 95% |
| Central/Cognitive | 10% | 85% |
| Force/Strength | 5% | 90% |
| Vestibular/Balance | 25% | 90% |
| Sleep | 20% | 80% |
| Pain/Somatosensory | 0% | 75% |

### New Assessments Enabled

- **Phase 1:** qEEG, Simple/Choice RT, Grip strength, Basic posturography
- **Phase 2:** ERP-based cognitive testing, Full vestibular workup, Sleep staging
- **Phase 3:** ADHD batteries, Pain profiling, Proprioceptive testing
- **Phase 4:** Longitudinal tracking, Cross-modal fusion, Normative comparison

---

*Implementation Plan generated: 2025-12-18*
*Reference: Delta Predictive Biosensing Gap Analysis v1.0*
