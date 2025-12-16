# Delta-Predictive Biosensing (DPB) Framework: Non-Contact Modalities

## Extension for Human Pose, Hand Tracking, Eye Tracking, and Voice Analysis

**Author:** AuraSense Tech Corporation  
**Date:** December 2025  
**Version:** 1.0

---

## 1. Executive Summary

This document extends the Delta-Predictive Biosensing (DPB) framework from contact-based biosignal measurements (ECG, EDA, PPG, tremor) to non-contact modalities captured via video and audio. The core DPB principle remains: **encode deviations from population templates** rather than raw signals, enabling sparse representation, faster convergence, and neuromorphic efficiency.

### Non-Contact Modalities Covered

| Modality | Input Source | Primary Clinical Targets |
|----------|--------------|-------------------------|
| **Human Pose** | RGB/RGBD video → 2D/3D keypoints | Gait disorders, Parkinson's, ataxia, stroke |
| **Hand Tracking** | RGB/depth video → hand landmarks | Tremor, bradykinesia, fine motor control |
| **Eye Tracking** | IR/RGB camera → gaze/pupil | Cognitive decline, fatigue, attention |
| **Voice Analysis** | Microphone → acoustic features | Dysarthria, speech motor control, PD |

### Key Innovation: Template-Based Deviation Encoding

Just as the heart occupies a tiny subspace of possible ECG waveforms, human movement and speech occupy constrained manifolds defined by biomechanics and neuromotor control. By encoding deviations from these templates:

- **100-1000× data reduction** through sparse event encoding
- **5-10× faster convergence** to clinical assessments
- **Neuromorphic compatibility** for edge deployment

---

## 2. Human Pose Estimation DPB

### 2.1 Signal Characteristics

| Parameter | Typical Range | Resolution | Information Content |
|-----------|---------------|------------|---------------------|
| Keypoint positions | 17-33 joints | ~5-10mm | ~6 bits per joint per axis |
| Joint angles | 0-180° | ~1° | ~8 bits per angle |
| Gait cycle duration | 0.8-1.4s | ~10ms | ~7 bits |
| Cadence | 80-120 steps/min | ~1 step/min | ~6 bits |
| Step length | 40-80cm | ~1cm | ~6 bits |
| Walking speed | 0.8-1.4 m/s | ~0.05 m/s | ~4 bits |

### 2.2 Population Templates

#### 2.2.1 Gait Cycle Phase Template

```
Gait Cycle Phases (% of cycle):
├── Stance Phase (60%)
│   ├── Initial Contact (0-2%)
│   ├── Loading Response (2-12%)
│   ├── Mid-Stance (12-31%)
│   ├── Terminal Stance (31-50%)
│   └── Pre-Swing (50-60%)
└── Swing Phase (40%)
    ├── Initial Swing (60-73%)
    ├── Mid-Swing (73-87%)
    └── Terminal Swing (87-100%)
```

#### 2.2.2 Joint Angle Templates (Sagittal Plane)

| Joint | Gait Phase | Normal ROM | PD Deviation | Ataxia Deviation |
|-------|------------|------------|--------------|------------------|
| Hip | Mid-stance | 10° extension | Reduced ROM | Increased variability |
| Hip | Terminal swing | 30° flexion | Reduced flexion | Irregular timing |
| Knee | Loading | 20° flexion | Reduced flexion | Excessive flexion |
| Knee | Mid-swing | 60° flexion | Reduced | Irregular |
| Ankle | Push-off | 20° plantarflexion | Reduced | Variable |
| Ankle | Swing | 0° (neutral) | Foot drop | Excessive dorsiflexion |

#### 2.2.3 Literature Priors

```python
GAIT_PRIORS = {
    # Spatiotemporal parameters (mean, std, units)
    'cadence': (110, 10, 'steps/min'),
    'stride_length': (1.4, 0.15, 'm'),
    'walking_speed': (1.2, 0.2, 'm/s'),
    'step_width': (0.08, 0.02, 'm'),
    'double_support_time': (0.12, 0.03, 's'),
    'swing_time': (0.40, 0.04, 's'),
    'stance_time': (0.60, 0.06, 's'),
    
    # Kinematic parameters (degrees)
    'hip_rom': (40, 5, 'deg'),
    'knee_rom': (60, 8, 'deg'),
    'ankle_rom': (30, 5, 'deg'),
    
    # Symmetry indices (ratio, 1.0 = perfect symmetry)
    'step_length_symmetry': (1.0, 0.05, 'ratio'),
    'swing_time_symmetry': (1.0, 0.05, 'ratio'),
    
    # Variability (CV%)
    'stride_time_cv': (3, 1.5, '%'),
    'step_length_cv': (4, 2, '%'),
}

# Age-adjusted priors
AGE_ADJUSTMENTS = {
    '20-40': {'walking_speed': (1.4, 0.15), 'cadence': (115, 8)},
    '40-60': {'walking_speed': (1.3, 0.18), 'cadence': (112, 10)},
    '60-80': {'walking_speed': (1.1, 0.22), 'cadence': (105, 12)},
    '80+':   {'walking_speed': (0.9, 0.25), 'cadence': (95, 15)},
}
```

### 2.3 Event Encoding Strategies

#### 2.3.1 Keypoint Deviation Events

```python
class KeypointDeviationEncoder:
    """
    Generate events when keypoint positions deviate from expected trajectory.
    
    Template: Expected keypoint trajectory for current gait phase
    Event: Fires when |observed - expected| > threshold
    """
    
    def __init__(self, skeleton_template, threshold_mm=15.0):
        self.template = skeleton_template  # [n_phases, n_joints, 3]
        self.threshold = threshold_mm
        self.phase_estimator = GaitPhaseEstimator()
        
    def encode(self, keypoints, timestamps):
        """
        Args:
            keypoints: [T, n_joints, 3] observed positions
            timestamps: [T] time values
        Returns:
            events: List[SpikeEvent] sparse deviation events
        """
        events = []
        
        for t, kp in zip(timestamps, keypoints):
            phase = self.phase_estimator.estimate(kp)
            expected = self.template[phase]
            
            for joint_idx, (obs, exp) in enumerate(zip(kp, expected)):
                deviation = np.linalg.norm(obs - exp)
                
                if deviation > self.threshold:
                    # Encode deviation direction and magnitude
                    direction = (obs - exp) / (deviation + 1e-6)
                    events.append(SpikeEvent(
                        timestamp=t,
                        channel=joint_idx * 3,  # x, y, z channels
                        polarity=np.sign(direction),
                        magnitude=deviation / self.threshold
                    ))
        
        return events
```

#### 2.3.2 Joint Angle Rate-of-Change Events

```python
class JointVelocityEncoder:
    """
    Event-based encoding of joint angular velocity.
    Fires when angular velocity crosses threshold levels.
    """
    
    JOINT_VELOCITY_THRESHOLDS = {
        'hip': 50,    # deg/s
        'knee': 100,  # deg/s
        'ankle': 80,  # deg/s
    }
    
    def encode(self, joint_angles, timestamps):
        events = []
        dt = np.diff(timestamps)
        angular_velocity = np.diff(joint_angles, axis=0) / dt[:, None]
        
        for t_idx, (t, av) in enumerate(zip(timestamps[1:], angular_velocity)):
            for joint_idx, (joint_name, thresh) in enumerate(
                self.JOINT_VELOCITY_THRESHOLDS.items()
            ):
                if abs(av[joint_idx]) > thresh:
                    events.append(SpikeEvent(
                        timestamp=t,
                        channel=joint_idx,
                        polarity=1 if av[joint_idx] > 0 else -1,
                        magnitude=abs(av[joint_idx]) / thresh
                    ))
        
        return events
```

#### 2.3.3 Gait Event Detection (Heel Strike/Toe Off)

```python
class GaitEventEncoder:
    """
    Encode discrete gait events as spikes.
    These are naturally sparse - only 2 events per leg per cycle.
    """
    
    EVENT_TYPES = {
        'heel_strike_left': 0,
        'toe_off_left': 1,
        'heel_strike_right': 2,
        'toe_off_right': 3,
    }
    
    def detect_events(self, foot_keypoints, timestamps):
        """
        Detect gait events from foot vertical velocity zero-crossings.
        """
        events = []
        
        # Vertical velocity of heel/toe markers
        heel_vel = np.gradient(foot_keypoints[:, :, 1], timestamps, axis=0)
        
        # Zero-crossings with direction indicate heel strike (neg→pos) 
        # and toe off (pos→neg)
        for side in ['left', 'right']:
            idx = 0 if side == 'left' else 1
            
            # Heel strikes: vertical velocity crosses zero going up
            hs_indices = self._find_zero_crossings(heel_vel[:, idx], 'positive')
            for hs_idx in hs_indices:
                events.append(SpikeEvent(
                    timestamp=timestamps[hs_idx],
                    channel=self.EVENT_TYPES[f'heel_strike_{side}'],
                    polarity=1,
                    magnitude=1.0
                ))
            
            # Toe offs: similar logic for toe marker
            # ...
        
        return events
```

### 2.4 SNN Architecture for Pose Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    POSE DPB-SNN ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Layer (Sparse Events)                                    │
│  ├── Keypoint deviation events (33 joints × 3 axes = 99 ch)    │
│  ├── Joint velocity events (6 major joints = 6 ch)             │
│  └── Gait events (4 discrete events = 4 ch)                    │
│      Total: ~109 input channels                                 │
│                                                                 │
│  Temporal Encoding Layer                                        │
│  ├── Multi-timescale LIF neurons (τ = 50ms, 200ms, 500ms)      │
│  └── Gait phase embedding (8 phase neurons)                    │
│                                                                 │
│  Spatial Graph Layer (Skeleton Topology)                        │
│  ├── Graph attention over skeletal connections                 │
│  ├── Hierarchical: limbs → body segments → full body           │
│  └── 128 spiking neurons with recurrent connections            │
│                                                                 │
│  Temporal Sequence Layer                                        │
│  ├── Spiking LSTM/GRU for gait cycle memory                    │
│  ├── Hidden state: 64 neurons                                  │
│  └── Captures inter-cycle variability                          │
│                                                                 │
│  Output Decoders                                                │
│  ├── Gait parameters (cadence, speed, stride length)           │
│  ├── Symmetry indices (step length, swing time)                │
│  ├── Variability metrics (CV of stride time)                   │
│  └── Clinical scores (UPDRS gait subscore)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 Clinical Targets & Biomarkers

| Condition | Key Gait Biomarkers | DPB Encoding Strategy |
|-----------|--------------------|-----------------------|
| **Parkinson's Disease** | Reduced stride length, shuffling, festination, arm swing asymmetry | Template deviation for arm swing, step length events |
| **Essential Tremor** | Tandem gait instability, postural sway | Balance deviation from midline template |
| **Ataxia** | Increased variability, wide base, irregular timing | CV of gait cycle parameters, phase timing events |
| **Stroke** | Asymmetry, circumduction, reduced knee flexion | Bilateral comparison deviation encoding |
| **Freezing of Gait** | Sudden cessation, festination, trembling | Velocity threshold events, gait cycle discontinuity |

---

## 3. Hand Tracking DPB

### 3.1 Signal Characteristics

| Parameter | Typical Range | Resolution | Information Content |
|-----------|---------------|------------|---------------------|
| Finger positions | 21 landmarks | ~2mm | ~5 bits per landmark |
| Finger tapping frequency | 2-6 Hz | ~0.1 Hz | ~5 bits |
| Tapping amplitude | 2-8 cm | ~2mm | ~5 bits |
| Opening-closing amplitude | 5-15 cm | ~3mm | ~4 bits |
| Tremor frequency | 3-12 Hz | ~0.5 Hz | ~4 bits |
| Tremor amplitude | 0.1-5 cm | ~0.5mm | ~7 bits |

### 3.2 Population Templates

#### 3.2.1 Finger Tapping Template

```python
FINGER_TAPPING_PRIORS = {
    # Healthy population norms
    'tapping_frequency': (4.5, 0.8, 'Hz'),
    'tapping_amplitude': (5.0, 1.2, 'cm'),
    'amplitude_decrement': (0.02, 0.01, 'cm/tap'),  # Normal fatigue
    'frequency_decrement': (0.01, 0.005, 'Hz/tap'),  # Normal slowing
    'inter_tap_interval_cv': (5, 2, '%'),
    'amplitude_cv': (8, 3, '%'),
    
    # Movement phases (% of tap cycle)
    'opening_phase': (45, 5, '%'),
    'closing_phase': (45, 5, '%'),
    'contact_phase': (10, 3, '%'),
}

# Pathology-specific deviations
PD_BRADYKINESIA_DEVIATIONS = {
    'tapping_frequency': -1.5,  # Hz reduction
    'tapping_amplitude': -2.0,  # cm reduction
    'amplitude_decrement': +0.05,  # Increased fatigue
    'frequency_decrement': +0.03,  # Increased slowing (sequence effect)
}
```

#### 3.2.2 Hand Open-Close Template

```python
HAND_OPEN_CLOSE_PRIORS = {
    'opening_frequency': (3.0, 0.5, 'Hz'),
    'max_aperture': (12.0, 2.0, 'cm'),
    'min_aperture': (1.0, 0.5, 'cm'),
    'opening_velocity': (40, 10, 'cm/s'),
    'closing_velocity': (35, 10, 'cm/s'),
    'movement_smoothness': (0.9, 0.05, 'normalized'),  # Spectral arc length
}
```

#### 3.2.3 Tremor Templates

```python
TREMOR_PRIORS = {
    # Physiological tremor (everyone has this)
    'physiological': {
        'frequency': (10, 2, 'Hz'),
        'amplitude': (0.05, 0.02, 'cm'),
    },
    
    # Pathological tremor types
    'parkinsonian_rest': {
        'frequency': (4.5, 1, 'Hz'),
        'amplitude': (1.5, 1, 'cm'),
    },
    'essential_tremor_postural': {
        'frequency': (6, 1.5, 'Hz'),
        'amplitude': (2, 1.5, 'cm'),
    },
    'cerebellar_intention': {
        'frequency': (3, 1, 'Hz'),
        'amplitude': (3, 2, 'cm'),  # Increases with target approach
    },
}
```

### 3.3 Event Encoding Strategies

#### 3.3.1 Finger Tapping Event Encoder

```python
class FingerTappingEncoder:
    """
    Encode finger tapping as sparse events capturing:
    1. Tap onset/offset
    2. Amplitude deviations from template
    3. Timing deviations from expected rhythm
    """
    
    def __init__(self, expected_frequency=4.5, expected_amplitude=5.0):
        self.expected_period = 1.0 / expected_frequency
        self.expected_amplitude = expected_amplitude
        self.tap_count = 0
        
    def encode(self, thumb_tip, index_tip, timestamps):
        """
        Args:
            thumb_tip: [T, 3] thumb tip positions
            index_tip: [T, 3] index finger tip positions
            timestamps: [T] time values
        """
        events = []
        
        # Calculate thumb-index distance
        distance = np.linalg.norm(thumb_tip - index_tip, axis=1)
        
        # Detect tap events (minima in distance)
        tap_indices = self._detect_taps(distance)
        
        for i, tap_idx in enumerate(tap_indices):
            t = timestamps[tap_idx]
            
            # Tap onset event (always fires)
            events.append(SpikeEvent(
                timestamp=t,
                channel=0,  # Tap onset channel
                polarity=1,
                magnitude=1.0
            ))
            
            # Amplitude deviation event
            amplitude = self._get_tap_amplitude(distance, tap_idx)
            amp_deviation = (amplitude - self.expected_amplitude) / self.expected_amplitude
            if abs(amp_deviation) > 0.1:  # 10% threshold
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=1,  # Amplitude deviation channel
                    polarity=1 if amp_deviation > 0 else -1,
                    magnitude=abs(amp_deviation)
                ))
            
            # Timing deviation event
            if i > 0:
                actual_interval = t - timestamps[tap_indices[i-1]]
                timing_deviation = (actual_interval - self.expected_period) / self.expected_period
                if abs(timing_deviation) > 0.1:  # 10% threshold
                    events.append(SpikeEvent(
                        timestamp=t,
                        channel=2,  # Timing deviation channel
                        polarity=1 if timing_deviation > 0 else -1,
                        magnitude=abs(timing_deviation)
                    ))
            
            # Update template (adaptive)
            self._update_template(amplitude, t)
            self.tap_count += 1
        
        return events
```

#### 3.3.2 Tremor Event Encoder

```python
class TremorEventEncoder:
    """
    Event-based tremor encoding using bandpass-filtered position.
    Fires events at tremor oscillation peaks exceeding threshold.
    """
    
    def __init__(self, tremor_band=(3, 12), amplitude_threshold=0.1):
        self.tremor_band = tremor_band  # Hz
        self.amplitude_threshold = amplitude_threshold  # cm
        
    def encode(self, hand_position, timestamps, fs=30):
        """
        Extract tremor component and encode as events.
        """
        events = []
        
        # Bandpass filter to isolate tremor
        sos = signal.butter(4, self.tremor_band, btype='band', fs=fs, output='sos')
        tremor_component = signal.sosfilt(sos, hand_position, axis=0)
        
        # Calculate instantaneous amplitude (envelope)
        analytic = signal.hilbert(tremor_component, axis=0)
        amplitude = np.abs(analytic)
        
        # Find peaks in tremor amplitude
        for axis in range(3):  # x, y, z
            peaks, properties = signal.find_peaks(
                tremor_component[:, axis],
                height=self.amplitude_threshold,
                distance=int(fs / self.tremor_band[1])  # Min distance based on max freq
            )
            
            for peak_idx in peaks:
                events.append(SpikeEvent(
                    timestamp=timestamps[peak_idx],
                    channel=axis,
                    polarity=1 if tremor_component[peak_idx, axis] > 0 else -1,
                    magnitude=amplitude[peak_idx, axis] / self.amplitude_threshold
                ))
        
        # Encode dominant frequency changes
        freq_events = self._encode_frequency_changes(tremor_component, timestamps, fs)
        events.extend(freq_events)
        
        return events
```

### 3.4 SNN Architecture for Hand Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    HAND DPB-SNN ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Layer (Sparse Events)                                    │
│  ├── Finger tapping events (onset, amplitude, timing = 3 ch)   │
│  ├── Tremor oscillation events (3 axes = 3 ch)                 │
│  ├── Hand aperture events (open/close = 2 ch)                  │
│  └── Individual finger deviation events (5 fingers = 5 ch)     │
│      Total: ~13 input channels                                  │
│                                                                 │
│  Feature Extraction Layer                                       │
│  ├── Tremor frequency estimator (spectral neurons)             │
│  ├── Movement velocity encoder                                  │
│  └── 32 spiking neurons                                        │
│                                                                 │
│  Temporal Integration Layer                                     │
│  ├── Multi-timescale LIF (τ = 100ms for rhythm, 1s for decay)  │
│  ├── Sequence effect detector (cumulative amplitude/freq decay)│
│  └── 64 spiking neurons                                        │
│                                                                 │
│  Output Decoders                                                │
│  ├── Tremor amplitude (continuous)                             │
│  ├── Tremor frequency (continuous)                             │
│  ├── Bradykinesia score (MDS-UPDRS item 3.4-3.6)              │
│  └── Binary: PD vs Healthy classification                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Clinical Targets & Biomarkers

| Clinical Task | Key Biomarkers | DPB Encoding Strategy |
|---------------|---------------|-----------------------|
| **Finger Tapping (UPDRS 3.4)** | Frequency, amplitude, decrement, hesitations | Tap onset + amplitude/timing deviation events |
| **Hand Movements (UPDRS 3.5)** | Opening amplitude, speed, fatiguing | Aperture threshold crossing events |
| **Pronation/Supination (UPDRS 3.6)** | Rotation amplitude, speed, rhythm | Angular velocity events |
| **Rest Tremor (UPDRS 3.17)** | Amplitude, constancy, frequency | Tremor peak events |
| **Postural Tremor (UPDRS 3.15)** | Amplitude in extended position | Position-dependent tremor events |

---

## 4. Eye Tracking DPB

### 4.1 Signal Characteristics

| Parameter | Typical Range | Resolution | Information Content |
|-----------|---------------|------------|---------------------|
| Gaze position | ~60° visual field | ~0.5° | ~7 bits |
| Saccade amplitude | 0.5-30° | ~0.5° | ~6 bits |
| Saccade peak velocity | 100-600°/s | ~20°/s | ~5 bits |
| Saccade latency | 150-350ms | ~10ms | ~5 bits |
| Fixation duration | 100-800ms | ~20ms | ~5 bits |
| Pupil diameter | 2-8mm | ~0.1mm | ~6 bits |
| Microsaccade rate | 0.5-3 Hz | ~0.2 Hz | ~4 bits |

### 4.2 Population Templates

#### 4.2.1 Saccade Main Sequence Template

The main sequence is a fundamental relationship between saccade amplitude and peak velocity:

```python
SACCADE_MAIN_SEQUENCE = {
    # Peak velocity = Vmax * (1 - exp(-amplitude / constant))
    'vmax': (600, 50, 'deg/s'),      # Maximum velocity asymptote
    'constant': (8, 1, 'deg'),        # Rate constant
    
    # Linear approximation for small saccades (< 20°)
    'slope': (35, 5, 'deg/s per deg'),  # Velocity/amplitude ratio
    
    # Duration relationship: Duration ≈ 2.2 * amplitude^0.4
    'duration_coefficient': (2.2, 0.2, 'ms'),
    'duration_exponent': (0.4, 0.05, ''),
}

def expected_saccade_velocity(amplitude):
    """Return expected peak velocity for given amplitude."""
    vmax = 600  # deg/s
    k = 8  # deg
    return vmax * (1 - np.exp(-amplitude / k))

def expected_saccade_duration(amplitude):
    """Return expected saccade duration for given amplitude."""
    return 2.2 * (amplitude ** 0.4) + 21  # ms
```

#### 4.2.2 Pupil Response Templates

```python
PUPIL_TEMPLATES = {
    # Pupillary Light Reflex (PLR)
    'plr_constriction_latency': (200, 30, 'ms'),
    'plr_constriction_amplitude': (30, 10, '%'),  # % reduction from baseline
    'plr_constriction_velocity': (3, 1, 'mm/s'),
    'plr_redilation_t50': (1500, 300, 'ms'),  # Time to 50% recovery
    
    # Baseline pupil diameter (moderate lighting)
    'baseline_diameter': (4.0, 0.8, 'mm'),
    
    # Cognitive load response
    'cognitive_dilation': (0.3, 0.15, 'mm'),  # Per unit cognitive load
    'cognitive_latency': (500, 100, 'ms'),
    
    # Fatigue indicators
    'pupil_unrest_index': (0.5, 0.2, 'mm'),  # Low-frequency oscillations
}
```

#### 4.2.3 Fixation Stability Templates

```python
FIXATION_PRIORS = {
    # Fixation duration distribution (gamma-like)
    'mean_duration': (250, 50, 'ms'),
    'duration_cv': (40, 10, '%'),
    
    # Fixation stability (gaze dispersion during fixation)
    'bcea_95': (1.0, 0.3, 'deg²'),  # Bivariate contour ellipse area
    
    # Microsaccade characteristics
    'microsaccade_rate': (1.5, 0.5, 'Hz'),
    'microsaccade_amplitude': (0.3, 0.1, 'deg'),
}
```

### 4.3 Event Encoding Strategies

#### 4.3.1 Saccade Event Encoder

```python
class SaccadeEventEncoder:
    """
    Encode saccades as sparse events with main sequence deviations.
    """
    
    def __init__(self, main_sequence_template):
        self.template = main_sequence_template
        
    def encode(self, gaze_position, timestamps, fs=120):
        """
        Detect saccades and encode deviations from main sequence.
        """
        events = []
        
        # Calculate gaze velocity
        velocity = np.gradient(gaze_position, 1/fs, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        
        # Detect saccades (velocity threshold)
        saccade_threshold = 30  # deg/s
        saccade_mask = speed > saccade_threshold
        
        # Find saccade onset/offset
        saccade_segments = self._find_segments(saccade_mask)
        
        for start, end in saccade_segments:
            # Calculate saccade metrics
            amplitude = np.linalg.norm(
                gaze_position[end] - gaze_position[start]
            )
            peak_velocity = np.max(speed[start:end])
            duration = (end - start) / fs * 1000  # ms
            latency = timestamps[start]  # From last fixation
            
            # Saccade onset event (always fires)
            events.append(SpikeEvent(
                timestamp=timestamps[start],
                channel=0,  # Saccade onset
                polarity=1,
                magnitude=amplitude / 10  # Normalized by 10°
            ))
            
            # Main sequence deviation: velocity
            expected_velocity = self.template.expected_velocity(amplitude)
            velocity_deviation = (peak_velocity - expected_velocity) / expected_velocity
            if abs(velocity_deviation) > 0.15:  # 15% threshold
                events.append(SpikeEvent(
                    timestamp=timestamps[start],
                    channel=1,  # Velocity deviation
                    polarity=1 if velocity_deviation > 0 else -1,
                    magnitude=abs(velocity_deviation)
                ))
            
            # Main sequence deviation: duration
            expected_duration = self.template.expected_duration(amplitude)
            duration_deviation = (duration - expected_duration) / expected_duration
            if abs(duration_deviation) > 0.2:  # 20% threshold
                events.append(SpikeEvent(
                    timestamp=timestamps[start],
                    channel=2,  # Duration deviation
                    polarity=1 if duration_deviation > 0 else -1,
                    magnitude=abs(duration_deviation)
                ))
            
            # Direction (quadrant encoding)
            direction = np.arctan2(
                gaze_position[end, 1] - gaze_position[start, 1],
                gaze_position[end, 0] - gaze_position[start, 0]
            )
            quadrant = int((direction + np.pi) / (np.pi/2)) % 4
            events.append(SpikeEvent(
                timestamp=timestamps[start],
                channel=3 + quadrant,  # Direction channels 3-6
                polarity=1,
                magnitude=1.0
            ))
        
        return events
```

#### 4.3.2 Pupil Response Event Encoder

```python
class PupilEventEncoder:
    """
    Encode pupil dynamics as events capturing:
    1. Dilation/constriction rate threshold crossings
    2. Deviations from expected PLR template
    3. Cognitive load indicators
    """
    
    def __init__(self, plr_template, rate_threshold=0.5):
        self.plr_template = plr_template
        self.rate_threshold = rate_threshold  # mm/s
        
    def encode(self, pupil_diameter, timestamps, light_events=None):
        """
        Args:
            pupil_diameter: [T] pupil diameter in mm
            timestamps: [T] time values
            light_events: Optional list of light stimulus times
        """
        events = []
        
        # Calculate pupil rate of change
        pupil_rate = np.gradient(pupil_diameter, timestamps)
        
        # Rate threshold crossing events
        for i, (t, rate) in enumerate(zip(timestamps, pupil_rate)):
            if abs(rate) > self.rate_threshold:
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=0 if rate > 0 else 1,  # Dilation/constriction
                    polarity=1 if rate > 0 else -1,
                    magnitude=abs(rate) / self.rate_threshold
                ))
        
        # PLR template deviation (if light events provided)
        if light_events:
            for light_time in light_events:
                plr_events = self._encode_plr_deviation(
                    pupil_diameter, timestamps, light_time
                )
                events.extend(plr_events)
        
        # Low-frequency oscillation (fatigue indicator)
        lfo_events = self._encode_pupil_unrest(pupil_diameter, timestamps)
        events.extend(lfo_events)
        
        return events
```

#### 4.3.3 Fixation/Microsaccade Encoder

```python
class FixationEventEncoder:
    """
    Encode fixation events and microsaccades.
    """
    
    def __init__(self, fixation_template):
        self.template = fixation_template
        
    def encode(self, gaze_position, timestamps, fs=120):
        events = []
        
        # Detect fixations (low velocity periods)
        velocity = np.gradient(gaze_position, 1/fs, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        
        fixation_mask = speed < 20  # deg/s threshold
        fixation_segments = self._find_segments(fixation_mask)
        
        for start, end in fixation_segments:
            duration = (end - start) / fs * 1000  # ms
            
            # Fixation onset
            events.append(SpikeEvent(
                timestamp=timestamps[start],
                channel=0,  # Fixation onset
                polarity=1,
                magnitude=1.0
            ))
            
            # Duration deviation
            expected_duration = self.template['mean_duration'][0]
            duration_deviation = (duration - expected_duration) / expected_duration
            if abs(duration_deviation) > 0.3:  # 30% threshold
                events.append(SpikeEvent(
                    timestamp=timestamps[end],
                    channel=1,  # Duration deviation
                    polarity=1 if duration_deviation > 0 else -1,
                    magnitude=abs(duration_deviation)
                ))
            
            # Detect microsaccades within fixation
            microsaccades = self._detect_microsaccades(
                gaze_position[start:end], 
                timestamps[start:end],
                fs
            )
            for ms_time, ms_amp in microsaccades:
                events.append(SpikeEvent(
                    timestamp=ms_time,
                    channel=2,  # Microsaccade
                    polarity=1,
                    magnitude=ms_amp / 0.5  # Normalized by 0.5°
                ))
        
        return events
```

### 4.4 SNN Architecture for Eye Tracking

```
┌─────────────────────────────────────────────────────────────────┐
│                  EYE TRACKING DPB-SNN ARCHITECTURE              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Layer (Sparse Events)                                    │
│  ├── Saccade events (onset, velocity dev, duration dev = 3 ch) │
│  ├── Saccade direction (4 quadrants = 4 ch)                    │
│  ├── Pupil events (dilation, constriction = 2 ch)              │
│  ├── Fixation events (onset, duration dev = 2 ch)              │
│  └── Microsaccade events (1 ch)                                │
│      Total: ~12 input channels                                  │
│                                                                 │
│  Temporal Integration Layer                                     │
│  ├── Short-term: τ = 100ms (for individual saccades)           │
│  ├── Medium-term: τ = 1s (for saccade sequences)               │
│  ├── Long-term: τ = 10s (for fatigue/vigilance)                │
│  └── 48 spiking neurons                                        │
│                                                                 │
│  Pattern Detection Layer                                        │
│  ├── Anti-saccade error detector                               │
│  ├── Smooth pursuit quality estimator                          │
│  ├── Visual search efficiency                                  │
│  └── 32 spiking neurons                                        │
│                                                                 │
│  Output Decoders                                                │
│  ├── Cognitive load estimate                                   │
│  ├── Fatigue/vigilance state                                   │
│  ├── Attention metrics                                         │
│  └── Clinical: MCI risk, AD biomarker                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 Clinical Targets & Biomarkers

| Condition | Key Eye Tracking Biomarkers | DPB Encoding Strategy |
|-----------|----------------------------|-----------------------|
| **Mild Cognitive Impairment** | Increased saccade latency, reduced smooth pursuit gain | Latency deviation events, pursuit error events |
| **Alzheimer's Disease** | Anti-saccade errors, prolonged fixations, reduced novelty preference | Error rate encoding, duration deviation events |
| **Parkinson's Disease** | Hypometric saccades, square wave jerks | Amplitude deviation events, intrusive saccade detection |
| **Multiple Sclerosis** | Internuclear ophthalmoplegia, nystagmus | Velocity asymmetry events, oscillation detection |
| **Fatigue/Drowsiness** | Slow saccades, prolonged blinks, pupil unrest | Velocity deviation, blink duration, LFO events |
| **ADHD/Attention Deficits** | Increased fixation variability, distractibility | Fixation duration CV, off-target saccade events |

---

## 5. Voice/Audio Analysis DPB

### 5.1 Signal Characteristics

| Parameter | Typical Range | Resolution | Information Content |
|-----------|---------------|------------|---------------------|
| Fundamental frequency (F0) | 80-300 Hz | ~1 Hz | ~8 bits |
| Jitter | 0.2-1.5% | ~0.1% | ~4 bits |
| Shimmer | 1-5% | ~0.2% | ~4 bits |
| Harmonic-to-Noise Ratio | 15-30 dB | ~1 dB | ~4 bits |
| Formants (F1, F2) | 200-2500 Hz | ~20 Hz | ~7 bits each |
| Speech rate | 100-180 words/min | ~5 wpm | ~4 bits |
| Vowel space area | Variable | ~10% | ~4 bits |

### 5.2 Population Templates

#### 5.2.1 Sustained Vowel /a/ Template

```python
SUSTAINED_VOWEL_PRIORS = {
    # Fundamental frequency by gender
    'f0_male': (120, 20, 'Hz'),
    'f0_female': (210, 30, 'Hz'),
    
    # Perturbation measures
    'jitter_local': (0.5, 0.25, '%'),
    'jitter_rap': (0.3, 0.15, '%'),
    'jitter_ppq5': (0.4, 0.2, '%'),
    
    'shimmer_local': (3.0, 1.5, '%'),
    'shimmer_apq3': (2.0, 1.0, '%'),
    'shimmer_apq5': (2.5, 1.2, '%'),
    
    # Noise measures
    'hnr': (22, 4, 'dB'),
    'nhr': (0.02, 0.01, 'ratio'),
    
    # Nonlinear dynamics
    'rpde': (0.5, 0.1, ''),  # Recurrence period density entropy
    'dfa': (0.7, 0.05, ''),  # Detrended fluctuation analysis
    'ppe': (0.15, 0.05, ''), # Pitch period entropy
}
```

#### 5.2.2 Formant Templates (Vowel Space)

```python
FORMANT_PRIORS = {
    # Formant frequencies for cardinal vowels (male/female)
    'vowel_a': {
        'F1': (700, 100, 'Hz'),
        'F2': (1200, 150, 'Hz'),
    },
    'vowel_i': {
        'F1': (300, 50, 'Hz'),
        'F2': (2300, 200, 'Hz'),
    },
    'vowel_u': {
        'F1': (350, 50, 'Hz'),
        'F2': (800, 100, 'Hz'),
    },
    
    # Vowel space metrics
    'vowel_space_area': (300000, 80000, 'Hz²'),  # Triangle /a/-/i/-/u/
    'formant_centralization_ratio': (1.0, 0.15, 'ratio'),
    'vowel_articulation_index': (1.0, 0.1, ''),
}
```

#### 5.2.3 Prosody Templates

```python
PROSODY_PRIORS = {
    # Pitch dynamics
    'f0_range': (1.5, 0.3, 'octaves'),  # Semitone range
    'f0_cv': (15, 5, '%'),  # Coefficient of variation
    'f0_slope': (0, 1, 'Hz/s'),  # Declination
    
    # Timing
    'speech_rate': (150, 20, 'syllables/min'),
    'articulation_rate': (200, 25, 'syllables/min'),
    'pause_rate': (3, 1, 'pauses/100syllables'),
    'pause_duration': (500, 200, 'ms'),
    
    # Rhythm
    'pvi_v': (50, 10, ''),  # Pairwise variability index (vowels)
    'pvi_c': (60, 12, ''),  # Pairwise variability index (consonants)
}
```

#### 5.2.4 Parkinson's Disease Speech Deviations

```python
PD_SPEECH_DEVIATIONS = {
    # Hypokinetic dysarthria characteristics
    'f0_range_reduction': 0.4,  # 40% reduction in pitch range
    'intensity_reduction': 0.3,  # Hypophonia
    'speech_rate_change': -0.2,  # 20% slower (or faster in festinating)
    
    # Articulatory undershoot
    'vowel_space_reduction': 0.3,  # 30% centralization
    'consonant_precision_reduction': 0.25,
    
    # Voice quality
    'hnr_reduction': 5,  # dB reduction
    'jitter_increase': 1.5,  # Factor
    'shimmer_increase': 1.5,  # Factor
    
    # Timing abnormalities
    'pause_increase': 2.0,  # Doubled pause frequency
    'speech_rate_variability_increase': 1.5,
}
```

### 5.3 Event Encoding Strategies

#### 5.3.1 Pitch Event Encoder

```python
class PitchEventEncoder:
    """
    Event-based encoding of fundamental frequency dynamics.
    """
    
    def __init__(self, f0_template, deviation_threshold=0.1):
        self.template = f0_template
        self.deviation_threshold = deviation_threshold
        
    def encode(self, f0_contour, timestamps, voiced_mask):
        """
        Args:
            f0_contour: [T] fundamental frequency (Hz), 0 for unvoiced
            timestamps: [T] time values
            voiced_mask: [T] boolean, True for voiced frames
        """
        events = []
        
        expected_f0 = self.template['f0'][0]  # Gender-appropriate mean
        expected_std = self.template['f0'][1]
        
        # Filter to voiced frames only
        voiced_f0 = f0_contour[voiced_mask]
        voiced_times = timestamps[voiced_mask]
        
        if len(voiced_f0) == 0:
            return events
        
        # F0 deviation events (z-score based)
        z_scores = (voiced_f0 - expected_f0) / expected_std
        
        for t, z in zip(voiced_times, z_scores):
            if abs(z) > 2:  # Beyond 2 standard deviations
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=0,  # F0 deviation
                    polarity=1 if z > 0 else -1,
                    magnitude=abs(z) / 2
                ))
        
        # F0 rate-of-change events (pitch dynamics)
        f0_rate = np.gradient(voiced_f0, voiced_times)
        for t, rate in zip(voiced_times, f0_rate):
            if abs(rate) > 50:  # Hz/s threshold
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=1,  # F0 rate
                    polarity=1 if rate > 0 else -1,
                    magnitude=abs(rate) / 50
                ))
        
        return events
```

#### 5.3.2 Voice Quality Event Encoder

```python
class VoiceQualityEventEncoder:
    """
    Encode perturbation measures (jitter, shimmer, HNR) as events.
    """
    
    def __init__(self, quality_template, window_ms=50):
        self.template = quality_template
        self.window_samples = int(window_ms * 16)  # Assuming 16kHz
        
    def encode(self, audio, timestamps, fs=16000):
        """
        Compute frame-wise quality metrics and encode deviations.
        """
        events = []
        hop = self.window_samples // 2
        
        for i in range(0, len(audio) - self.window_samples, hop):
            frame = audio[i:i + self.window_samples]
            t = timestamps[i + self.window_samples // 2]
            
            # Compute local jitter
            jitter = self._compute_jitter(frame, fs)
            jitter_z = (jitter - self.template['jitter_local'][0]) / self.template['jitter_local'][1]
            
            if abs(jitter_z) > 2:
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=0,  # Jitter deviation
                    polarity=1 if jitter_z > 0 else -1,
                    magnitude=abs(jitter_z) / 2
                ))
            
            # Compute local shimmer
            shimmer = self._compute_shimmer(frame, fs)
            shimmer_z = (shimmer - self.template['shimmer_local'][0]) / self.template['shimmer_local'][1]
            
            if abs(shimmer_z) > 2:
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=1,  # Shimmer deviation
                    polarity=1 if shimmer_z > 0 else -1,
                    magnitude=abs(shimmer_z) / 2
                ))
            
            # Compute HNR
            hnr = self._compute_hnr(frame, fs)
            hnr_z = (hnr - self.template['hnr'][0]) / self.template['hnr'][1]
            
            if abs(hnr_z) > 2:
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=2,  # HNR deviation
                    polarity=1 if hnr_z > 0 else -1,
                    magnitude=abs(hnr_z) / 2
                ))
        
        return events
```

#### 5.3.3 Articulation Event Encoder

```python
class ArticulationEventEncoder:
    """
    Encode articulatory events from formant trajectories.
    """
    
    def __init__(self, formant_template):
        self.template = formant_template
        
    def encode(self, formants, timestamps):
        """
        Args:
            formants: [T, n_formants] formant frequencies (F1, F2, F3...)
            timestamps: [T] time values
        """
        events = []
        
        F1, F2 = formants[:, 0], formants[:, 1]
        
        # Vowel space position events
        # Map to vowel triangle and detect boundary crossings
        for i, (t, f1, f2) in enumerate(zip(timestamps, F1, F2)):
            vowel_class = self._classify_vowel(f1, f2)
            
            # Detect vowel transitions
            if i > 0 and vowel_class != self._classify_vowel(F1[i-1], F2[i-1]):
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=vowel_class,  # Vowel class channel
                    polarity=1,
                    magnitude=1.0
                ))
            
            # Formant centralization events (reduced vowel space)
            expected_f1, expected_f2 = self.template[vowel_class]['F1'][0], self.template[vowel_class]['F2'][0]
            centroid_f1, centroid_f2 = 500, 1500  # Neutral schwa position
            
            # Distance from expected vs distance toward centroid
            expected_dist = np.sqrt((f1 - expected_f1)**2 + (f2 - expected_f2)**2)
            if expected_dist > 100:  # Significant deviation toward center
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=10,  # Centralization channel
                    polarity=-1,  # Negative = underarticulation
                    magnitude=expected_dist / 100
                ))
        
        # Formant transition rate events
        f1_rate = np.gradient(F1, timestamps)
        f2_rate = np.gradient(F2, timestamps)
        formant_velocity = np.sqrt(f1_rate**2 + f2_rate**2)
        
        for t, vel in zip(timestamps, formant_velocity):
            if vel > 1000:  # Hz/s threshold for rapid transitions
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=11,  # Rapid articulation
                    polarity=1,
                    magnitude=vel / 1000
                ))
        
        return events
```

#### 5.3.4 Prosody/Timing Event Encoder

```python
class ProsodyEventEncoder:
    """
    Encode prosodic events: pauses, rate changes, rhythm.
    """
    
    def __init__(self, prosody_template):
        self.template = prosody_template
        
    def encode(self, audio, timestamps, fs=16000):
        """
        Detect and encode prosodic events.
        """
        events = []
        
        # Voice Activity Detection
        vad = self._compute_vad(audio, fs)
        
        # Detect pauses (gaps in speech)
        pause_segments = self._find_pauses(vad, timestamps, min_duration=0.2)
        
        expected_pause_dur = self.template['pause_duration'][0] / 1000  # Convert to seconds
        
        for start, end in pause_segments:
            duration = end - start
            
            # Pause onset event
            events.append(SpikeEvent(
                timestamp=start,
                channel=0,  # Pause onset
                polarity=1,
                magnitude=1.0
            ))
            
            # Pause duration deviation
            dur_deviation = (duration - expected_pause_dur) / expected_pause_dur
            if abs(dur_deviation) > 0.5:  # 50% threshold
                events.append(SpikeEvent(
                    timestamp=end,
                    channel=1,  # Pause duration deviation
                    polarity=1 if dur_deviation > 0 else -1,
                    magnitude=abs(dur_deviation)
                ))
        
        # Speech rate events (syllable rate estimation)
        syllable_times = self._detect_syllables(audio, fs)
        local_rates = self._compute_local_speech_rate(syllable_times, window=2.0)
        
        expected_rate = self.template['speech_rate'][0]
        for t, rate in local_rates:
            rate_deviation = (rate - expected_rate) / expected_rate
            if abs(rate_deviation) > 0.2:  # 20% threshold
                events.append(SpikeEvent(
                    timestamp=t,
                    channel=2,  # Speech rate deviation
                    polarity=1 if rate_deviation > 0 else -1,
                    magnitude=abs(rate_deviation)
                ))
        
        return events
```

### 5.4 SNN Architecture for Voice Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                   VOICE DPB-SNN ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Layer (Sparse Events)                                    │
│  ├── Pitch events (F0 deviation, F0 rate = 2 ch)               │
│  ├── Voice quality events (jitter, shimmer, HNR = 3 ch)        │
│  ├── Articulation events (vowel class, centralization = 12 ch) │
│  ├── Prosody events (pause, rate = 3 ch)                       │
│  └── Spectral events (MFCC delta threshold crossings = 13 ch)  │
│      Total: ~33 input channels                                  │
│                                                                 │
│  Short-Term Feature Layer                                       │
│  ├── Phoneme-level integration (τ = 50-100ms)                  │
│  ├── Pitch pattern detector                                    │
│  └── 64 spiking neurons                                        │
│                                                                 │
│  Utterance-Level Integration                                    │
│  ├── Sentence-level prosody (τ = 1-3s)                         │
│  ├── Speech rate tracker                                       │
│  ├── Voice quality accumulator                                 │
│  └── 128 spiking neurons                                       │
│                                                                 │
│  Task-Specific Layer                                            │
│  ├── Sustained vowel analyzer                                  │
│  ├── Diadochokinetic rate detector (/pa-ta-ka/)               │
│  ├── Connected speech analyzer                                 │
│  └── 64 spiking neurons                                        │
│                                                                 │
│  Output Decoders                                                │
│  ├── Voice quality score                                       │
│  ├── Articulation precision                                    │
│  ├── Prosody naturalness                                       │
│  └── Clinical: PD probability, dysarthria severity             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 Clinical Targets & Biomarkers

| Condition | Key Voice Biomarkers | DPB Encoding Strategy |
|-----------|---------------------|-----------------------|
| **Parkinson's Disease** | Reduced pitch range, increased jitter/shimmer, hypophonia, vowel centralization | Pitch deviation events, quality deviation events, centralization events |
| **Essential Tremor** | Voice tremor (4-12 Hz modulation in pitch/amplitude) | Periodic pitch/amplitude deviation events |
| **Ataxia** | Scanning speech, irregular rhythm, explosive loudness | Prosody timing events, intensity deviation events |
| **ALS** | Hypernasal resonance, strained voice, reduced rate | Formant deviation events, rate events |
| **Cognitive Decline** | Word-finding pauses, reduced fluency, simplified syntax | Pause events, rate events, (linguistic features via ASR) |
| **Depression** | Flat affect, monotone, reduced speech rate | Pitch range events, prosody events |

---

## 6. Multi-Modal Fusion Architecture

### 6.1 Fusion Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-MODAL DPB FUSION ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │
│  │   POSE    │ │   HAND    │ │    EYE    │ │   VOICE   │       │
│  │  DPB-SNN  │ │  DPB-SNN  │ │  DPB-SNN  │ │  DPB-SNN  │       │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘       │
│        │             │             │             │              │
│        ▼             ▼             ▼             ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TEMPORAL ALIGNMENT LAYER                    │   │
│  │  • Synchronize events across modalities                 │   │
│  │  • Handle different sampling rates (30Hz video, 16kHz)  │   │
│  │  • Create common time reference                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CROSS-MODAL ATTENTION LAYER                 │   │
│  │  • Spiking attention mechanism                          │   │
│  │  • Learn correlations: tremor-voice, gait-posture       │   │
│  │  • 128 neurons with cross-modal connections             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              HIERARCHICAL INTEGRATION                    │   │
│  │  Level 1: Within-system (motor, sensory, cognitive)     │   │
│  │  Level 2: Cross-system correlations                     │   │
│  │  Level 3: Global neurological state estimation          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CLINICAL OUTPUT DECODERS                    │   │
│  │  • MDS-UPDRS Total Motor Score                          │   │
│  │  • Diagnostic classification (PD, ET, MSA, PSP)         │   │
│  │  • Cognitive assessment (MoCA proxy)                    │   │
│  │  • Fatigue/medication state                             │   │
│  │  • Longitudinal progression tracking                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Cross-Modal Correlations to Learn

| Modality Pair | Expected Correlation | Clinical Significance |
|---------------|---------------------|----------------------|
| Hand tremor ↔ Voice tremor | r > 0.6 | Common motor system involvement |
| Gait speed ↔ Speech rate | r > 0.4 | Global bradykinesia |
| Saccade latency ↔ Reaction time | r > 0.5 | Cognitive processing speed |
| Pupil response ↔ Voice affect | r > 0.3 | Autonomic/emotional state |
| Hand bradykinesia ↔ Hypomimia | r > 0.7 | Dopaminergic deficit severity |
| Gait variability ↔ Attention metrics | r > 0.4 | Executive function |

---

## 7. Algorithm Inventory Summary

### 7.1 Complete Algorithm Count by Layer

| Layer | Pose | Hand | Eye | Voice | Total |
|-------|------|------|-----|-------|-------|
| **Population Templates** | 12 | 8 | 10 | 14 | 44 |
| **Event Encoders** | 6 | 5 | 5 | 6 | 22 |
| **Feature Extractors** | 8 | 6 | 7 | 12 | 33 |
| **SNN Layers** | 4 | 3 | 3 | 4 | 14 |
| **Output Decoders** | 10 | 8 | 9 | 11 | 38 |
| **Fusion Components** | - | - | - | - | 8 |
| **Total** | 40 | 30 | 34 | 47 | **159** |

### 7.2 Combined with Contact Modalities

| Category | Contact | Non-Contact | Combined |
|----------|---------|-------------|----------|
| Templates/Priors | 17 | 44 | 61 |
| Event Encoders | 25 | 22 | 47 |
| Neuron Models | 19 | 19 (shared) | 19 |
| SNN Architectures | 23 | 14 | 37 |
| Training Algorithms | 25 | 25 (shared) | 25 |
| Output Decoders | 10 | 38 | 48 |
| Fusion | 12 | 8 | 20 |
| Evaluation Metrics | 28 | 28 (shared) | 28 |
| **Grand Total** | 277 | 159 | **285** |

---

## 8. Experimental Questions (Non-Contact Specific)

### 8.1 Pose Estimation Questions

1. **Keypoint noise tolerance**: How does pose estimation noise (5-10mm) affect DPB convergence?
2. **Frame rate requirements**: Minimum FPS for gait event detection (15, 30, 60 Hz)?
3. **2D vs 3D**: Does 3D pose provide significant benefit for neurological assessment?
4. **Occlusion handling**: How to maintain assessment during partial occlusions?

### 8.2 Hand Tracking Questions

5. **Fine motor resolution**: Minimum spatial resolution for tremor quantification?
6. **Depth camera benefit**: RGB vs RGB-D for finger tracking accuracy?
7. **Task standardization**: Effect of instruction variations on biomarker stability?

### 8.3 Eye Tracking Questions

8. **Sampling rate requirements**: 60Hz vs 120Hz vs 240Hz for saccade detection?
9. **Webcam feasibility**: Consumer webcam vs dedicated eye tracker accuracy?
10. **Calibration sensitivity**: Impact of calibration drift on clinical metrics?

### 8.4 Voice Analysis Questions

11. **Recording conditions**: Studio vs smartphone vs telehealth quality requirements?
12. **Language independence**: Do acoustic biomarkers generalize across languages?
13. **Task selection**: Sustained vowel vs connected speech vs diadochokinesis?
14. **Background noise**: SNR requirements for reliable feature extraction?

### 8.5 Multi-Modal Questions

15. **Fusion benefit**: Additive vs synergistic improvement from multi-modal fusion?
16. **Missing modality**: Graceful degradation when one modality unavailable?
17. **Temporal alignment**: Optimal methods for synchronizing different frame rates?
18. **Computational budget**: Can multi-modal run on neuromorphic edge hardware?

---

## 9. Datasets for Validation

### 9.1 Pose/Gait Datasets

| Dataset | Size | Modalities | Clinical Population |
|---------|------|------------|---------------------|
| WearGait-PD | 45 PD + controls | IMU + video | Parkinson's |
| PhysioNet Gait-PD | 93 PD + 73 HC | Force plates | Parkinson's |
| Human3.6M | 3.6M frames | MoCap + video | Healthy (reference) |
| REMAP | 400 participants | Full MoCap | Movement disorders |

### 9.2 Hand Movement Datasets

| Dataset | Size | Tasks | Clinical Population |
|---------|------|-------|---------------------|
| PD Hand Video (Seoul) | 112 samples | Finger tapping | PD + controls |
| mPower | 5800+ | Multiple | PD (smartphone) |
| OpenNeuro tremor | Various | Rest/postural | Tremor disorders |

### 9.3 Eye Tracking Datasets

| Dataset | Size | Paradigms | Clinical Population |
|---------|------|-----------|---------------------|
| ETH-XGaze | 1.1M images | Gaze estimation | Healthy (reference) |
| AD Eye Tracking | 210 participants | VPC task | Alzheimer's/MCI |
| MS Eye Tracking | Various | Anti-saccade | Multiple sclerosis |

### 9.4 Voice/Speech Datasets

| Dataset | Size | Tasks | Clinical Population |
|---------|------|-------|---------------------|
| mPower Voice | 65,000+ recordings | Sustained /a/ | Parkinson's |
| MDVR-KCL | 37 participants | Spontaneous + read | Parkinson's |
| Bridge2AI Voice | Growing | Multiple | Various conditions |
| PC-GITA | 100 participants | Multiple tasks | Parkinson's (Spanish) |

---

## 10. Implementation Roadmap

### Phase 1: Individual Modality Development (Weeks 1-8)

- [ ] Pose: Implement keypoint deviation encoder with MediaPipe
- [ ] Hand: Implement finger tapping encoder with MediaPipe Hands
- [ ] Eye: Implement saccade encoder with basic webcam tracking
- [ ] Voice: Implement pitch/quality encoders with Praat/Parselmouth

### Phase 2: SNN Integration (Weeks 9-12)

- [ ] Port encoders to snnTorch framework
- [ ] Train individual modality SNNs on available datasets
- [ ] Benchmark against ANN baselines

### Phase 3: Multi-Modal Fusion (Weeks 13-16)

- [ ] Implement temporal alignment layer
- [ ] Develop cross-modal attention mechanism
- [ ] Train fusion model on synchronized recordings

### Phase 4: Neuromorphic Deployment (Weeks 17-20)

- [ ] Map to Xylo/Pulsar hardware constraints
- [ ] Optimize for power/latency targets
- [ ] Validate on neuromorphic emulator

---

## 11. References

### Pose Estimation
1. Sabo et al. (2022) - Markerless PD gait analysis
2. Filtjens et al. (2021) - MS-GCN for freezing of gait
3. Cao et al. (2017) - OpenPose

### Hand Tracking
4. Ferraris et al. (2024) - Deep learning hand tracking PD review
5. Guarín et al. (2025) - visionMD hand tremor quantification
6. Guo et al. (2022) - 3D hand pose for finger tapping

### Eye Tracking
7. Marandi & Gazerani (2019) - Eye tracking cognitive decline
8. Wilcockson et al. (2019) - Saccade biomarkers for MCI
9. PMC 2022 - Eye tracking AD diagnosis

### Voice Analysis
10. Rusz et al. (2021) - Speech biomarkers prodromal PD
11. Tsanas et al. (2011) - Nonlinear speech analysis PD
12. Moro-Velazquez et al. (2021) - Articulatory/phonatory review

### Neuromorphic/SNN
13. Yik et al. (2023) - snnTorch framework
14. SynSense - Xylo documentation
15. Innatera - Pulsar documentation

---

*Document generated for AuraSense Tech Corporation NeuroPlay Platform*
*Delta-Predictive Biosensing Framework v2.0*
