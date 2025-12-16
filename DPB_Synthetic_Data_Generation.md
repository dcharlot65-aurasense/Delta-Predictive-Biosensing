# DPB Framework: Synthetic Data Generation

## Controlled Signal Synthesis for Algorithm Validation

---

## 1. Design Principles

### 1.1 Core Philosophy

Synthetic data generators are the **inverse of DPB encoders**:
- DPB Encoder: Signal → Template + Deviation Events
- Generator: Template + Deviation Parameters → Signal + Ground Truth

Every generator produces:
1. **Raw signal** (time series, video frames, audio samples)
2. **Ground truth labels** (exact parameters used)
3. **Event ground truth** (exact timestamps of events that should be detected)

### 1.2 Validation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Synthetic Data Pipeline                       │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Parameter   │───▶│  Generator   │───▶│   Synthetic  │       │
│  │    Space     │    │   (Inverse   │    │    Signal    │       │
│  │              │    │   Template)  │    │      +       │       │
│  └──────────────┘    └──────────────┘    │ Ground Truth │       │
│         │                                 └──────┬───────┘       │
│         │                                        │               │
│         ▼                                        ▼               │
│  ┌──────────────┐                        ┌──────────────┐       │
│  │   Sweep:     │                        │     DPB      │       │
│  │  • Noise     │                        │   Encoder    │       │
│  │  • Severity  │                        │              │       │
│  │  • Rate      │                        └──────┬───────┘       │
│  └──────────────┘                               │               │
│                                                  ▼               │
│                                          ┌──────────────┐       │
│                                          │   Compare    │       │
│                                          │  Extracted   │       │
│                                          │     vs       │       │
│                                          │ Ground Truth │       │
│                                          └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Generator Inventory by Modality

### Total: 156 Generators

| Category | Count |
|----------|-------|
| Contact Biosignals | 42 |
| Pose/Gait | 28 |
| Hand/Fine Motor | 24 |
| Eye Movements | 26 |
| Voice/Audio | 28 |
| Multi-Modal Synchronization | 8 |

---

## 3. Contact Biosignal Generators (42)

### 3.1 Cardiac Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 1 | **ECG morphology generator** | P/QRS/T amplitudes, durations, ST elevation | Beat annotations, interval durations |
| 2 | **Heart rate generator** | Mean HR, HRV (SDNN, RMSSD, pNN50) | R-peak times, instantaneous HR |
| 3 | **HRV spectral generator** | LF/HF power, LF/HF ratio | Power spectral density |
| 4 | **Arrhythmia generator** | PAC/PVC rate, AF burden | Arrhythmia event times + types |
| 5 | **Respiratory sinus arrhythmia** | Breathing rate, RSA amplitude | Breath times, HR modulation |
| 6 | **PPG waveform generator** | Pulse amplitude, dicrotic notch, PTT | Systolic/diastolic peaks, PPG features |
| 7 | **PPG artifact generator** | Motion type, intensity, duration | Artifact intervals |
| 8 | **Heart rate recovery generator** | Peak HR, recovery tau, fitness level | HR trajectory post-exercise |

### 3.2 Electrodermal Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 9 | **EDA tonic generator** | Baseline level, drift rate, temperature effect | Skin conductance level |
| 10 | **SCR event generator** | Event rate, amplitude distribution, rise/decay tau | SCR onset times, amplitudes, durations |
| 11 | **Stimulus-locked SCR** | Stimulus times, response probability, latency | Event-related SCR annotations |
| 12 | **Arousal state generator** | Arousal level (1-10), transition dynamics | Continuous arousal ground truth |
| 13 | **EDA artifact generator** | Movement artifacts, electrode artifacts | Artifact intervals + types |

### 3.3 Tremor/Movement Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 14 | **Physiological tremor generator** | Frequency (8-12 Hz), amplitude | Tremor envelope, instantaneous frequency |
| 15 | **Parkinsonian rest tremor** | Frequency (4-6 Hz), amplitude, intermittency | Tremor on/off, amplitude envelope |
| 16 | **Essential tremor generator** | Frequency (4-12 Hz), postural dependence | Tremor characteristics by posture |
| 17 | **Cerebellar tremor generator** | Frequency (3-5 Hz), intention component | Tremor modulated by movement |
| 18 | **Tremor amplitude modulation** | Waxing/waning pattern, period | Amplitude envelope |
| 19 | **Multi-axis tremor** | X/Y/Z correlation, phase relationships | 3D tremor trajectory |

### 3.4 EMG Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 20 | **Surface EMG generator** | Motor unit firing rates, recruitment | MU activation times |
| 21 | **Voluntary contraction EMG** | Force level (% MVC), fatigue | Force trajectory, fatigue index |
| 22 | **Pathological EMG** | Fasciculations, fibrillations | Abnormal event times |
| 23 | **Co-contraction generator** | Agonist/antagonist ratio | Muscle activation patterns |

### 3.5 Respiratory Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 24 | **Breathing pattern generator** | Rate, depth, I:E ratio | Breath onsets, volumes |
| 25 | **Irregular breathing** | Apnea events, Cheyne-Stokes | Apnea intervals, pattern labels |
| 26 | **Respiratory effort generator** | Thoracic/abdominal motion | Effort signals |

### 3.6 Thermal Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 27 | **Skin temperature generator** | Baseline, vasomotor oscillations | Temperature trajectory |
| 28 | **Thermal response generator** | Stimulus-evoked changes | Response amplitudes, latencies |

### 3.7 Noise & Artifact Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 29 | **White noise generator** | SNR level | Noise-free reference |
| 30 | **Pink (1/f) noise generator** | Spectral slope, amplitude | Noise-free reference |
| 31 | **Powerline interference** | 50/60 Hz, harmonics, amplitude | Clean reference |
| 32 | **Motion artifact generator** | Artifact type, intensity, duration | Artifact intervals |
| 33 | **Baseline wander generator** | Frequency (<0.5 Hz), amplitude | Detrended reference |
| 34 | **Electrode noise generator** | Contact impedance variations | Clean reference |
| 35 | **Quantization noise** | Bit depth | Full-resolution reference |
| 36 | **Sampling jitter generator** | Timing uncertainty | True sample times |
| 37 | **Clipping generator** | Saturation threshold | Clipped intervals |
| 38 | **Missing data generator** | Dropout rate, burst length | Missing data mask |

### 3.8 Multi-Signal Synchronization

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 39 | **ECG-PPG synchronized** | PTT, PAT | Timing relationships |
| 40 | **Cardiorespiratory coupling** | Phase coupling strength | Coupling indices |
| 41 | **Autonomic state generator** | Sympathetic/parasympathetic balance | ANS state labels |
| 42 | **Stress response generator** | Stressor intensity, duration | Multi-signal stress markers |

---

## 4. Pose/Gait Generators (28)

### 4.1 Normal Gait Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 43 | **Gait cycle generator** | Cadence, stride length, velocity | Phase labels, spatiotemporal params |
| 44 | **Joint angle trajectory** | ROM per joint, coordination | Angle time series per joint |
| 45 | **Keypoint trajectory generator** | 17-33 keypoints, 3D positions | Keypoint positions + velocities |
| 46 | **Arm swing generator** | Amplitude, symmetry, phase | Arm angle trajectories |
| 47 | **Trunk motion generator** | Sway amplitude, rotation | Trunk angles |
| 48 | **Age-adjusted gait** | Age (20-90), expected decline | Age-appropriate parameters |
| 49 | **Speed-adjusted gait** | Walking speed (0.5-2.0 m/s) | Speed-appropriate kinematics |

### 4.2 Pathological Gait Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 50 | **Parkinsonian gait** | Severity (H&Y 1-5), medication state | PD feature intensities |
| 51 | **Shuffling gait** | Step height reduction, stride shortening | Step heights, stride lengths |
| 52 | **Festination generator** | Acceleration rate, trigger | Festination onset, step times |
| 53 | **Freezing of gait** | FOG probability, duration, triggers | FOG intervals, triggers |
| 54 | **Asymmetric gait** | L/R asymmetry ratio, affected side | Per-side parameters |
| 55 | **Ataxic gait** | Variability increase, base widening | CV of parameters, step width |
| 56 | **Spastic gait** | Stiffness, circumduction | Joint stiffness parameters |
| 57 | **Antalgic gait** | Pain side, severity | Stance time asymmetry |
| 58 | **Hemiplegic gait** | Affected side, severity | Side-specific impairments |

### 4.3 Gait Variability Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 59 | **Stride time variability** | CV target, fractal index | Per-stride times |
| 60 | **Stride length variability** | CV target | Per-stride lengths |
| 61 | **Dual-task variability** | Cognitive load effect | Task-specific variability |
| 62 | **Fatigue progression** | Fatigue rate, pattern | Time-varying parameters |

### 4.4 Pose Noise Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 63 | **Keypoint jitter** | Noise std per joint | Clean keypoints |
| 64 | **Occlusion generator** | Occluded joints, duration | Occlusion mask |
| 65 | **Tracking dropout** | Dropout rate | Valid frame mask |
| 66 | **ID switch generator** | Switch probability | Correct ID assignments |
| 67 | **Depth ambiguity** | 2D projection ambiguity | True 3D positions |
| 68 | **Camera motion** | Pan, tilt, zoom | Stabilized reference |
| 69 | **Lighting variation** | Illumination changes | Lighting-normalized |
| 70 | **Frame rate variation** | 15/30/60 fps | Full-rate reference |

---

## 5. Hand/Fine Motor Generators (24)

### 5.1 Finger Tapping Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 71 | **Normal finger tapping** | Frequency, amplitude, regularity | Tap times, apertures |
| 72 | **Bradykinetic tapping** | Reduced frequency, amplitude | PD severity |
| 73 | **Amplitude decrement** | Decrement rate, pattern | Per-tap amplitudes |
| 74 | **Frequency decrement** | Slowing rate | Per-tap intervals |
| 75 | **Hesitation/arrest** | Arrest probability, duration | Arrest intervals |
| 76 | **Tapping fatigue** | Fatigue time constant | Performance trajectory |

### 5.2 Hand Tremor Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 77 | **Rest tremor (hand)** | Frequency, amplitude, regularity | Tremor envelope |
| 78 | **Postural tremor** | Frequency, amplitude, position-dependence | Posture-specific tremor |
| 79 | **Kinetic tremor** | Frequency, amplitude, movement-dependence | Movement-phase tremor |
| 80 | **Intention tremor** | Amplitude scaling with precision demand | Target-approach tremor |
| 81 | **Tremor intermittency** | On/off pattern, duty cycle | Tremor presence mask |
| 82 | **Multi-finger tremor** | Per-finger correlation | Finger-specific tremor |

### 5.3 Hand Movement Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 83 | **Pronation-supination** | ROM, speed, regularity | Rotation angles |
| 84 | **Hand open-close** | Aperture range, speed | Aperture trajectory |
| 85 | **Precision grip** | Force control, tremor | Grip force, stability |
| 86 | **Reaching movement** | Speed, accuracy, smoothness | Trajectory, jerk |
| 87 | **Drawing (spiral/line)** | Tremor, accuracy | Drawing trace |

### 5.4 Hand Tracking Noise

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 88 | **Landmark jitter** | Per-landmark noise std | Clean landmarks |
| 89 | **Self-occlusion** | Finger occlusion pattern | Occlusion mask |
| 90 | **Tracking loss** | Loss probability | Valid frame mask |
| 91 | **Depth estimation error** | Depth noise | True 3D positions |
| 92 | **Motion blur (hand)** | Blur kernel, threshold speed | Sharp reference |
| 93 | **Background clutter** | Distractor objects | Segmentation mask |
| 94 | **Skin tone variation** | Lighting, skin tone | Normalized reference |

---

## 6. Eye Movement Generators (26)

### 6.1 Saccade Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 95 | **Main sequence saccade** | Amplitude, obeys main sequence | Velocity, duration |
| 96 | **Hypometric saccade** | Gain <1.0, undershooting | Gain, error |
| 97 | **Hypermetric saccade** | Gain >1.0, overshooting | Gain, error |
| 98 | **Saccade latency** | Latency distribution (µ, σ) | Per-saccade latency |
| 99 | **Express saccade** | Short latency (<100ms) | Latency, probability |
| 100 | **Delayed saccade** | Long latency (>300ms) | Latency, probability |
| 101 | **Corrective saccade** | Post-saccadic correction | Primary + corrective |
| 102 | **Saccade sequence** | Scanpath pattern | Saccade times, targets |
| 103 | **Antisaccade generator** | Error rate, latency | Correct/error labels |
| 104 | **Memory-guided saccade** | Delay period, accuracy | Memory trace |

### 6.2 Fixation Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 105 | **Stable fixation** | BCEA, microsaccade rate | Fixation position |
| 106 | **Unstable fixation** | Increased drift, nystagmus | Instability metrics |
| 107 | **Fixation duration** | Duration distribution | Per-fixation duration |
| 108 | **Microsaccade generator** | Rate, amplitude | Microsaccade times |
| 109 | **Square wave jerks** | Frequency, amplitude | SWJ intervals |
| 110 | **Ocular flutter** | Oscillation parameters | Flutter episodes |

### 6.3 Smooth Pursuit Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 111 | **Normal pursuit** | Gain ~1.0, phase lag | Pursuit metrics |
| 112 | **Impaired pursuit** | Reduced gain, catch-up saccades | Saccade intrusions |
| 113 | **Predictive pursuit** | Phase lead, anticipation | Prediction metrics |

### 6.4 Pupil Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 114 | **Pupil light reflex** | Constriction amplitude, latency, velocity | PLR parameters |
| 115 | **Pupil dilation (cognitive)** | Task-evoked amplitude, latency | Dilation events |
| 116 | **Hippus generator** | Oscillation frequency, amplitude | Spontaneous fluctuations |
| 117 | **Pupil fatigue response** | Fatigue-related changes | Fatigue trajectory |
| 118 | **Afferent pupil defect** | Asymmetric response | RAPD simulation |

### 6.5 Eye Tracking Noise

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 119 | **Gaze estimation noise** | Angular error std | True gaze direction |
| 120 | **Blink artifacts** | Blink rate, duration | Blink intervals |
| 121 | **Pupil detection failure** | Dropout rate | Valid sample mask |
| 122 | **Calibration drift** | Drift rate, pattern | Calibrated reference |
| 123 | **Head movement artifact** | Head pose changes | Compensated gaze |
| 124 | **Glasses/contacts artifacts** | Reflection, distortion | Clean reference |

---

## 7. Voice/Audio Generators (28)

### 7.1 Phonation Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 125 | **Sustained vowel /a/** | F0, jitter, shimmer, HNR | Acoustic parameters |
| 126 | **Sustained vowel /i/** | F0, formants, quality | Formant values |
| 127 | **Sustained vowel /u/** | F0, formants, quality | Formant values |
| 128 | **Pitch contour generator** | F0 trajectory, range | Per-frame F0 |
| 129 | **Jitter generator** | Jitter %, type (RAP, PPQ) | Per-cycle perturbation |
| 130 | **Shimmer generator** | Shimmer %, type | Per-cycle amplitude |
| 131 | **Voice onset time** | VOT distribution | VOT per consonant |
| 132 | **Vocal fry generator** | Creaky voice parameters | Fry intervals |
| 133 | **Breathiness generator** | Aspiration noise level | HNR, breathiness |

### 7.2 Articulation Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 134 | **Vowel space generator** | F1-F2 positions, area | Vowel triangle |
| 135 | **Vowel centralization** | Reduction degree | Centralized formants |
| 136 | **Formant transition** | Transition rate, smoothness | Formant trajectories |
| 137 | **Diadochokinesis** | /pa/-/ta/-/ka/ rate, regularity | DDK rate, variability |
| 138 | **Consonant precision** | Articulation accuracy | Error rate |

### 7.3 Prosody Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 139 | **Speech rate generator** | Syllables/sec, variability | Per-utterance rate |
| 140 | **Pause generator** | Pause rate, duration distribution | Pause intervals |
| 141 | **Filled pause generator** | "um", "uh" rate | Filled pause times |
| 142 | **Pitch range generator** | F0 range, declination | Pitch span |
| 143 | **Intensity contour** | Loudness trajectory | Per-frame intensity |
| 144 | **Rhythm generator** | Stress pattern, timing | Syllable times |

### 7.4 Pathological Voice Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 145 | **Hypokinetic dysarthria** | Reduced loudness, monotone, imprecise | PD severity |
| 146 | **Spastic dysarthria** | Strained, slow, pitch breaks | Dysarthria type |
| 147 | **Ataxic dysarthria** | Irregular, scanning speech | Cerebellar markers |
| 148 | **Voice tremor generator** | Tremor frequency, amplitude | Vocal tremor |
| 149 | **Hypophonia generator** | Reduced loudness progression | Loudness trajectory |

### 7.5 Audio Noise Generators

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 150 | **Background noise** | SNR, noise type (babble, white, pink) | Clean reference |
| 151 | **Room acoustics** | RT60, early reflections | Anechoic reference |
| 152 | **Microphone response** | Frequency response, distortion | Flat reference |
| 153 | **Codec artifacts** | Compression level, codec type | Uncompressed reference |
| 154 | **Clipping (audio)** | Saturation level | Unclipped reference |
| 155 | **Telehealth degradation** | Packet loss, jitter | Full-quality reference |

---

## 8. Multi-Modal Synchronization (8)

| # | Generator | Parameters | Ground Truth Output |
|---|-----------|------------|---------------------|
| 156 | **Hand tremor + Voice tremor** | Correlation coefficient | Synchronized tremor |
| 157 | **Gait + Speech rate** | Coupling strength | Motor-speech correlation |
| 158 | **Saccade + Reaction time** | Cognitive-motor coupling | Latency correlations |
| 159 | **Pupil + Voice affect** | Arousal coherence | Autonomic-speech |
| 160 | **Bradykinesia + Hypomimia** | Motor severity coupling | Multi-system PD |
| 161 | **Gait + Postural tremor** | Movement-tremor interaction | Dynamic tremor |
| 162 | **Full PD simulation** | H&Y stage, medication | All modality params |
| 163 | **Healthy aging simulation** | Age, fitness level | Age-appropriate decline |

---

## 9. Parameter Space Definitions

### 9.1 Pathology Severity Scales

```
Severity: 0.0 ──────────────────────────────── 1.0
          │                                    │
       Healthy                            Severe
       (Control)                         Pathology

Example mapping (Parkinsonian tremor):
  0.0: No tremor (healthy)
  0.2: Barely perceptible (H&Y 1)
  0.4: Mild, intermittent (H&Y 2)
  0.6: Moderate, persistent (H&Y 3)
  0.8: Severe, constant (H&Y 4)
  1.0: Very severe (H&Y 5)
```

### 9.2 Noise Parameter Ranges

| Noise Type | Parameter | Range | Units |
|------------|-----------|-------|-------|
| SNR | Signal-to-noise | -10 to +40 | dB |
| Jitter (temporal) | Sample timing | 0 to 10 | % of sample period |
| Dropout | Missing data rate | 0 to 30 | % |
| Quantization | Bit depth | 4 to 24 | bits |
| Motion artifact | Intensity | 0 to 100 | % of signal range |

### 9.3 Sampling Rate Tests

| Modality | Test Rates | Reference Rate |
|----------|------------|----------------|
| ECG | 50, 100, 250, 500, 1000 Hz | 1000 Hz |
| EDA | 4, 8, 16, 32, 64 Hz | 64 Hz |
| Pose | 15, 30, 60, 120 fps | 120 fps |
| Eye | 30, 60, 120, 240, 1000 Hz | 1000 Hz |
| Voice | 8000, 16000, 44100 Hz | 44100 Hz |

---

## 10. Ground Truth Schema

### 10.1 Universal Ground Truth Format

```rust
struct SyntheticSample {
    // Identification
    id: Uuid,
    modality: Modality,
    generator_version: String,
    
    // Generation parameters (full reproducibility)
    seed: u64,
    parameters: HashMap<String, ParameterValue>,
    
    // Output signals
    signals: HashMap<String, TimeSeries>,
    
    // Ground truth annotations
    events: Vec<GroundTruthEvent>,
    continuous_labels: HashMap<String, TimeSeries>,
    segment_labels: Vec<SegmentLabel>,
    
    // Metadata
    duration_sec: f64,
    sample_rate: f64,
    generated_at: DateTime<Utc>,
}

struct GroundTruthEvent {
    timestamp: f64,         // Exact time in seconds
    event_type: String,     // e.g., "r_peak", "saccade_onset", "tap"
    channel: Option<u32>,   // For multi-channel signals
    parameters: HashMap<String, f64>,  // Event-specific params
}

struct SegmentLabel {
    start: f64,
    end: f64,
    label: String,          // e.g., "tremor_present", "fog_episode"
    confidence: f64,        // Always 1.0 for synthetic (ground truth)
}
```

### 10.2 Per-Modality Ground Truth

**Contact Biosignals:**
```rust
// ECG ground truth
events: [
    { timestamp: 0.123, event_type: "r_peak", parameters: { rr_interval: 0.85 } },
    { timestamp: 0.145, event_type: "t_peak", parameters: { qt_interval: 0.38 } },
    ...
]
continuous_labels: {
    "instantaneous_hr": [71.2, 71.5, 70.8, ...],  // Per-beat HR
    "hrv_sdnn_10s": [45.2, 46.1, ...],             // Rolling HRV
}
```

**Pose/Gait:**
```rust
// Gait ground truth
events: [
    { timestamp: 0.0, event_type: "heel_strike", parameters: { side: "left" } },
    { timestamp: 0.32, event_type: "toe_off", parameters: { side: "left" } },
    { timestamp: 0.45, event_type: "heel_strike", parameters: { side: "right" } },
    ...
]
continuous_labels: {
    "gait_phase": [0, 1, 2, 3, 4, 5, 6, 7, 0, ...],  // 8-phase cycle
    "left_knee_angle": [15.2, 18.5, 22.1, ...],
    "stride_length": [0.68, 0.71, 0.69, ...],  // Per-stride
}
segment_labels: [
    { start: 12.5, end: 14.2, label: "freezing_of_gait" },
]
```

**Eye Movements:**
```rust
// Eye tracking ground truth
events: [
    { timestamp: 0.180, event_type: "saccade_onset", parameters: { 
        amplitude: 12.5, direction: 45.0, latency: 0.180 
    }},
    { timestamp: 0.225, event_type: "saccade_offset", parameters: { 
        duration: 0.045, peak_velocity: 420.0 
    }},
    { timestamp: 0.225, event_type: "fixation_onset", parameters: { 
        x: 512, y: 384 
    }},
    ...
]
```

---

## 11. Experimental Designs Using Synthetic Data

### 11.1 Convergence Experiments

**Protocol:**
1. Generate identical signal N times with same seed
2. Apply DPB encoder, measure time/samples to reach 95% accuracy
3. Apply conventional encoder, measure same
4. Compare convergence curves

**Parameter Sweep:**
```
Signal duration: [1s, 2s, 5s, 10s, 30s, 60s]
Pathology severity: [0.0, 0.25, 0.5, 0.75, 1.0]
SNR: [40dB, 30dB, 20dB, 10dB, 0dB]
```

### 11.2 Robustness Experiments

**Protocol:**
1. Generate clean signal, measure baseline accuracy
2. Progressively add noise, measure accuracy degradation
3. Compare DPB vs conventional degradation curves

**Noise Types:**
- Additive white Gaussian noise
- Pink noise (1/f)
- Motion artifacts
- Dropout/missing data
- Quantization noise
- Sampling jitter

### 11.3 Edge Case Testing

**Protocol:**
1. Generate rare/extreme parameter combinations
2. Verify detection still works
3. Identify failure modes

**Edge Cases:**
- Maximum pathology severity
- Very slow/fast rates
- High variability
- Mixed pathologies
- Transition states (ON→OFF medication)

### 11.4 Cross-Validation Design

```
Synthetic Data Split:
├── Training (60%)
│   ├── Healthy: 1000 samples × 5 durations × 5 SNRs
│   └── Pathological: 1000 samples × 5 severities × 5 SNRs
├── Validation (20%)
│   └── Same distribution as training
└── Test (20%)
    ├── In-distribution: Same parameters
    └── Out-of-distribution: Novel parameter combinations
```

---

## 12. Implementation Architecture

### 12.1 Generator Trait (Rust)

```rust
pub trait SyntheticGenerator {
    type Parameters;
    type Output;
    
    /// Generate synthetic data with given parameters and seed
    fn generate(&self, params: &Self::Parameters, seed: u64) -> GeneratorResult<Self::Output>;
    
    /// Get parameter ranges for sweeps
    fn parameter_ranges(&self) -> ParameterRanges;
    
    /// Generate ground truth labels
    fn ground_truth(&self, params: &Self::Parameters, seed: u64) -> GroundTruth;
    
    /// Validate parameters are within bounds
    fn validate_params(&self, params: &Self::Parameters) -> Result<(), ValidationError>;
}

pub trait NoiseSuperimposer {
    /// Add noise to clean signal
    fn apply(&self, signal: &[f32], noise_params: &NoiseParams, seed: u64) -> Vec<f32>;
    
    /// Return clean reference
    fn clean_reference(&self) -> &[f32];
}
```

### 12.2 Parameter Sweep Framework

```rust
pub struct ParameterSweep<G: SyntheticGenerator> {
    generator: G,
    sweep_config: SweepConfig,
}

pub struct SweepConfig {
    /// Parameters to sweep
    sweep_params: Vec<SweepParameter>,
    /// Number of seeds per configuration
    seeds_per_config: usize,
    /// Output format
    output_format: OutputFormat,
}

pub struct SweepParameter {
    name: String,
    values: Vec<ParameterValue>,
}

impl<G: SyntheticGenerator> ParameterSweep<G> {
    pub fn run(&self) -> SweepResults {
        let configs = self.enumerate_configurations();
        
        configs.par_iter()  // Parallel execution
            .flat_map(|config| {
                (0..self.sweep_config.seeds_per_config)
                    .map(|seed| {
                        let output = self.generator.generate(config, seed as u64);
                        let ground_truth = self.generator.ground_truth(config, seed as u64);
                        (config.clone(), seed, output, ground_truth)
                    })
            })
            .collect()
    }
}
```

### 12.3 GPU-Accelerated Generation

For high-throughput synthetic data generation:

```wgsl
// WGSL kernel for parallel ECG generation
@compute @workgroup_size(256)
fn generate_ecg_batch(
    @builtin(global_invocation_id) id: vec3<u32>,
    @group(0) @binding(0) params: array<EcgParams>,
    @group(0) @binding(1) output: array<f32>,
    @group(0) @binding(2) seeds: array<u32>,
) {
    let sample_idx = id.x;
    let batch_idx = id.y;
    
    let p = params[batch_idx];
    let seed = seeds[batch_idx];
    
    // Generate ECG sample using parameters
    let t = f32(sample_idx) / p.sample_rate;
    let ecg = generate_ecg_sample(t, p, seed);
    
    output[batch_idx * p.num_samples + sample_idx] = ecg;
}
```

---

## 13. Dataset Generation Targets

### 13.1 Per-Modality Dataset Sizes

| Modality | Subjects | Samples/Subject | Total Samples | Duration/Sample |
|----------|----------|-----------------|---------------|-----------------|
| Contact biosignals | 1000 | 100 | 100,000 | 60s |
| Pose/Gait | 1000 | 50 | 50,000 | 30s (walk) |
| Hand | 1000 | 100 | 100,000 | 10s (task) |
| Eye | 1000 | 50 | 50,000 | 60s |
| Voice | 1000 | 50 | 50,000 | 10s |

### 13.2 Parameter Coverage

| Condition | Healthy | Mild | Moderate | Severe |
|-----------|---------|------|----------|--------|
| Subjects | 250 | 250 | 250 | 250 |
| Per severity SNR variants | 5 | 5 | 5 | 5 |
| Total unique conditions | 1250 | 1250 | 1250 | 1250 |

### 13.3 Storage Estimates

| Modality | Per-Sample Size | Total Uncompressed | Compressed (Parquet) |
|----------|-----------------|--------------------|-----------------------|
| Contact | ~5 MB | ~500 GB | ~100 GB |
| Pose | ~2 MB | ~100 GB | ~20 GB |
| Hand | ~1 MB | ~100 GB | ~20 GB |
| Eye | ~3 MB | ~150 GB | ~30 GB |
| Voice | ~2 MB | ~100 GB | ~20 GB |
| **Total** | | **~950 GB** | **~190 GB** |

---

## 14. Algorithm Count Summary

| Category | Generators |
|----------|------------|
| Contact Biosignals | 42 |
| Pose/Gait | 28 |
| Hand/Fine Motor | 24 |
| Eye Movements | 26 |
| Voice/Audio | 28 |
| Multi-Modal Sync | 8 |
| **Total Generators** | **156** |

**Combined with Detection Algorithms:**

| Component | Contact | Non-Contact | Generators | Total |
|-----------|---------|-------------|------------|-------|
| Algorithms | 277 | 417 | 156 | **850** |

---

## 15. Success Criteria for Synthetic Data

| Criterion | Metric | Target |
|-----------|--------|--------|
| Realism | Expert blind test | >80% can't distinguish real vs synthetic |
| Coverage | Parameter space | >95% of clinical range covered |
| Ground truth accuracy | Self-validation | 100% (by construction) |
| Reproducibility | Same seed → same output | Bit-identical |
| Generation speed | Throughput | >1000 samples/sec on GPU |
| Detection validation | Accuracy on synthetic | Within 2% of real data accuracy |

---

*Document: DPB Synthetic Data Generation Framework v1.0*
*AuraSense Tech Corporation - December 2025*
