# DPB Synthetic Data: Tiered Generation Framework

## Three Levels of Synthetic Data Fidelity

---

## 1. Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REAL-WORLD DATA FLOW                                 │
│                                                                              │
│   CONTACT                           NON-CONTACT                              │
│   ───────                           ───────────                              │
│   Physical      Analog    Digital   Camera/Mic   Video/Audio   Features     │
│   Phenomenon → Sensor → ADC →──┐    Sensor    →   Frames   → Extraction    │
│                                │                              (MediaPipe,   │
│                                │                               Praat, etc.) │
│                                │                                    │       │
│                                ▼                                    ▼       │
│                         ┌──────────────────────────────────────────────┐    │
│                         │           DPB ALGORITHM INPUT                │    │
│                         │     (Digital time series / Features)         │    │
│                         └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      SYNTHETIC DATA INJECTION POINTS                         │
│                                                                              │
│   LEVEL 1                LEVEL 2                      LEVEL 3               │
│   ───────                ───────                      ───────               │
│   Digital Signal         Featurized Data              Raw Media             │
│   Synthesis              Synthesis                    Synthesis             │
│                                                                              │
│   • ECG waveforms        • Keypoint trajectories      • Video of gait       │
│   • EDA traces           • Gaze coordinates           • Video of hands      │
│   • Tremor signals       • F0/formant contours        • Audio of speech     │
│   • PPG waveforms        • Landmark positions         • Eye tracking video  │
│                                                                              │
│   Complexity: LOW        Complexity: MEDIUM           Complexity: HIGH      │
│   Realism: N/A           Realism: MEDIUM              Realism: HIGH         │
│   (Direct input)         (Bypasses extraction)        (Full pipeline)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Level 1: Digital Signal Synthesis

### 2.1 What It Is

Direct generation of digitized time-series that would come from ADC output. This is the input format for contact biosignal algorithms.

### 2.2 Applicable Modalities

| Signal | Sample Rate | Bit Depth | Channels | Units |
|--------|-------------|-----------|----------|-------|
| ECG | 250-1000 Hz | 12-24 bit | 1-12 | mV |
| PPG | 25-256 Hz | 16-24 bit | 1-2 | arbitrary |
| EDA | 4-64 Hz | 16-24 bit | 1 | µS |
| EMG | 1000-4000 Hz | 16-24 bit | 1-8 | µV |
| Accelerometer | 50-200 Hz | 12-16 bit | 3 (XYZ) | g or m/s² |
| Gyroscope | 50-200 Hz | 12-16 bit | 3 (XYZ) | °/s |
| Temperature | 1-10 Hz | 12-16 bit | 1 | °C |
| Force/Pressure | 50-500 Hz | 12-16 bit | 1-N | N or Pa |

### 2.3 Implementation Approach

**Mathematical Models + Noise:**

```rust
/// Level 1: Digital signal generator
pub trait Level1Generator {
    /// Generate raw digital samples as would come from ADC
    fn generate_samples(
        &self,
        params: &SignalParams,
        duration_sec: f64,
        sample_rate: f64,
        seed: u64,
    ) -> DigitalSignal;
}

pub struct DigitalSignal {
    pub samples: Vec<f32>,          // Or Vec<i16> for true ADC output
    pub sample_rate: f64,
    pub bit_depth: u8,
    pub ground_truth: SignalGroundTruth,
}
```

**ECG Example:**

```rust
pub struct EcgGenerator {
    // McSharry et al. dynamical model parameters
    // "A Dynamical Model for Generating Synthetic Electrocardiogram Signals"
    // IEEE Trans Biomed Eng, 2003
}

impl Level1Generator for EcgGenerator {
    fn generate_samples(&self, params: &EcgParams, duration: f64, fs: f64, seed: u64) -> DigitalSignal {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_samples = (duration * fs) as usize;
        let mut samples = Vec::with_capacity(n_samples);
        
        // State variables for ODE integration
        let mut x = 0.0_f64;
        let mut y = 0.0_f64;
        let mut z = 0.0_f64;
        
        let dt = 1.0 / fs;
        
        for i in 0..n_samples {
            // Generate RR interval with HRV
            let rr = self.generate_rr_interval(params, &mut rng);
            
            // ODE integration (McSharry model)
            let (dx, dy, dz) = self.derivatives(x, y, z, params);
            x += dx * dt;
            y += dy * dt;
            z += dz * dt;
            
            // z is the ECG signal
            let clean_sample = z as f32;
            
            // Add realistic noise
            let noisy_sample = self.add_noise(clean_sample, params.snr_db, &mut rng);
            
            // Quantize to ADC resolution
            let quantized = self.quantize(noisy_sample, params.bit_depth);
            
            samples.push(quantized);
        }
        
        DigitalSignal {
            samples,
            sample_rate: fs,
            bit_depth: params.bit_depth,
            ground_truth: self.compute_ground_truth(&samples, fs),
        }
    }
}
```

**Tremor Example (Sum of Sinusoids + Stochastic Modulation):**

```rust
pub struct TremorGenerator;

impl Level1Generator for TremorGenerator {
    fn generate_samples(&self, params: &TremorParams, duration: f64, fs: f64, seed: u64) -> DigitalSignal {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = (duration * fs) as usize;
        let dt = 1.0 / fs;
        
        let mut samples = vec![0.0f32; n];
        
        for i in 0..n {
            let t = i as f64 * dt;
            
            // Fundamental frequency with slight variation
            let f0 = params.frequency + rng.gen::<f64>() * params.freq_jitter;
            
            // Amplitude envelope (can include intermittency)
            let envelope = self.amplitude_envelope(t, params, &mut rng);
            
            // Sum of harmonics
            let mut sample = 0.0;
            for h in 1..=params.n_harmonics {
                let harmonic_amp = params.amplitude / (h as f64).powi(2);
                let phase = params.phases[h-1] + rng.gen::<f64>() * params.phase_jitter;
                sample += harmonic_amp * (2.0 * PI * f0 * h as f64 * t + phase).sin();
            }
            
            samples[i] = (sample * envelope) as f32;
        }
        
        // Add sensor noise
        self.add_accelerometer_noise(&mut samples, params.snr_db, &mut rng);
        
        DigitalSignal { samples, sample_rate: fs, bit_depth: 16, ground_truth: ... }
    }
}
```

### 2.4 Level 1 Generator Inventory

| # | Generator | Model Basis | Key Parameters |
|---|-----------|-------------|----------------|
| 1 | ECG | McSharry ODE model | HR, HRV, morphology |
| 2 | PPG | Pulse decomposition | HR, pulse shape, PAT |
| 3 | EDA (tonic) | Drift + 1/f noise | SCL range, drift rate |
| 4 | EDA (phasic) | Bateman function SCRs | Event rate, amplitude dist |
| 5 | Tremor | Sum of sinusoids | Frequency, amplitude, harmonics |
| 6 | EMG | Motor unit trains | Firing rate, recruitment |
| 7 | Respiration | Sinusoid + harmonics | Rate, depth, I:E ratio |
| 8 | Accelerometer (gait) | Kinematic model | Step frequency, impact |
| 9 | Gyroscope (gait) | Angular velocity model | Rotation patterns |
| 10 | Temperature | Slow drift + vasomotor | Baseline, oscillations |

### 2.5 Advantages & Limitations

| Advantages | Limitations |
|------------|-------------|
| Fast generation | Doesn't test feature extraction |
| Perfect ground truth | May miss sensor-specific artifacts |
| Full parameter control | Realism depends on model fidelity |
| No external dependencies | Contact modalities only |

---

## 3. Level 2: Featurized Data Synthesis

### 3.1 What It Is

Direct generation of the feature/landmark data that would be output by video/audio processing pipelines (MediaPipe, OpenPose, Praat, etc.). Bypasses actual media generation.

### 3.2 Applicable Modalities

| Feature Type | Source Pipeline | Dimensions | Rate |
|--------------|-----------------|------------|------|
| Body pose keypoints | MediaPipe/OpenPose | 33×3 (x,y,z) | 30 fps |
| Hand landmarks | MediaPipe Hands | 21×3 per hand | 30 fps |
| Face landmarks | MediaPipe Face | 468×3 | 30 fps |
| Gaze coordinates | Eye tracker | 2 (x,y) | 60-1000 Hz |
| Pupil diameter | Eye tracker | 1 | 60-1000 Hz |
| F0 (pitch) | RAPT/CREPE/YIN | 1 | 100 Hz (typical) |
| Formants | Praat/LPC | 4 (F1-F4) | 100 Hz |
| MFCCs | Librosa/torchaudio | 13-40 | 100 Hz |
| Voice quality | Praat | 3 (jitter, shimmer, HNR) | Per-frame |

### 3.3 Implementation Approach

**Keypoint Trajectory Generation:**

```rust
/// Level 2: Featurized data generator
pub trait Level2Generator {
    /// Generate feature trajectories as would come from extraction pipeline
    fn generate_features(
        &self,
        params: &FeatureParams,
        duration_sec: f64,
        frame_rate: f64,
        seed: u64,
    ) -> FeatureSequence;
}

pub struct FeatureSequence {
    pub timestamps: Vec<f64>,
    pub features: Vec<FeatureFrame>,  // Per-frame features
    pub ground_truth: FeatureGroundTruth,
}
```

**Pose/Gait Keypoint Generator:**

```rust
pub struct GaitKeypointGenerator {
    // Biomechanical model of gait
    // Joint angle trajectories from Winter's "Biomechanics of Human Movement"
    joint_angle_templates: GaitCycleTemplates,
    body_segment_lengths: BodyModel,
}

impl Level2Generator for GaitKeypointGenerator {
    fn generate_features(&self, params: &GaitParams, duration: f64, fps: f64, seed: u64) -> FeatureSequence {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_frames = (duration * fps) as usize;
        let dt = 1.0 / fps;
        
        let mut keypoints = Vec::with_capacity(n_frames);
        let mut events = Vec::new();
        
        // Gait cycle state
        let mut phase = 0.0_f64;  // 0-1 within cycle
        let cycle_duration = 60.0 / params.cadence;  // seconds per cycle
        
        for frame in 0..n_frames {
            let t = frame as f64 * dt;
            
            // Advance phase
            phase += dt / cycle_duration;
            if phase >= 1.0 {
                phase -= 1.0;
                // Record gait events
                events.push(GaitEvent::HeelStrike { time: t, side: Side::Left });
            }
            if (phase - 0.5).abs() < dt / cycle_duration {
                events.push(GaitEvent::HeelStrike { time: t, side: Side::Right });
            }
            
            // Get joint angles from template + pathology modifications
            let angles = self.get_joint_angles(phase, params);
            
            // Apply pathology effects
            let modified_angles = self.apply_pathology(angles, params.pathology, &mut rng);
            
            // Forward kinematics: angles → keypoint positions
            let mut frame_keypoints = self.forward_kinematics(&modified_angles);
            
            // Add measurement noise (simulating MediaPipe uncertainty)
            self.add_keypoint_noise(&mut frame_keypoints, params.noise_std, &mut rng);
            
            // Simulate occlusions
            self.simulate_occlusions(&mut frame_keypoints, params.occlusion_prob, &mut rng);
            
            keypoints.push(frame_keypoints);
        }
        
        FeatureSequence {
            timestamps: (0..n_frames).map(|i| i as f64 * dt).collect(),
            features: keypoints,
            ground_truth: FeatureGroundTruth::Gait { events, ... },
        }
    }
}

impl GaitKeypointGenerator {
    fn get_joint_angles(&self, phase: f64, params: &GaitParams) -> JointAngles {
        // Interpolate from template
        let template = &self.joint_angle_templates;
        
        JointAngles {
            // Hip flexion/extension (from Winter's data)
            left_hip: template.hip.interpolate(phase) * params.rom_scale,
            right_hip: template.hip.interpolate(phase + 0.5) * params.rom_scale,
            
            // Knee flexion (0 = extended)
            left_knee: template.knee.interpolate(phase) * params.rom_scale,
            right_knee: template.knee.interpolate(phase + 0.5) * params.rom_scale,
            
            // Ankle dorsiflexion
            left_ankle: template.ankle.interpolate(phase) * params.rom_scale,
            right_ankle: template.ankle.interpolate(phase + 0.5) * params.rom_scale,
            
            // Add inter-cycle variability
            // ...
        }
    }
    
    fn apply_pathology(&self, angles: JointAngles, pathology: &Pathology, rng: &mut impl Rng) -> JointAngles {
        match pathology {
            Pathology::Parkinsonian { severity } => {
                // Reduced ROM, asymmetry, shuffling
                JointAngles {
                    left_hip: angles.left_hip * (1.0 - 0.3 * severity),
                    right_hip: angles.right_hip * (1.0 - 0.4 * severity),  // Asymmetry
                    // Reduced knee flexion
                    left_knee: angles.left_knee * (1.0 - 0.2 * severity),
                    // ...
                }
            }
            Pathology::Ataxic { severity } => {
                // Increased variability
                let noise_scale = 1.0 + 2.0 * severity;
                JointAngles {
                    left_hip: angles.left_hip + rng.gen::<f64>() * noise_scale * 5.0,
                    // ...
                }
            }
            // ...
        }
    }
    
    fn forward_kinematics(&self, angles: &JointAngles) -> Vec<Keypoint> {
        // Standard kinematic chain
        // Pelvis → Hip → Knee → Ankle → Foot
        // Returns 33 keypoints in MediaPipe format
        let mut keypoints = vec![Keypoint::default(); 33];
        
        // Pelvis at origin (will translate for walking)
        let pelvis = Vec3::ZERO;
        
        // Left leg
        let left_hip_pos = pelvis + self.body_model.pelvis_to_hip_left;
        let left_knee_pos = left_hip_pos + self.rotate_segment(
            self.body_model.thigh_length,
            angles.left_hip,
        );
        let left_ankle_pos = left_knee_pos + self.rotate_segment(
            self.body_model.shank_length,
            angles.left_hip + angles.left_knee,
        );
        
        keypoints[23] = Keypoint { x: left_hip_pos.x, y: left_hip_pos.y, z: left_hip_pos.z, visibility: 1.0 };
        keypoints[25] = Keypoint { x: left_knee_pos.x, y: left_knee_pos.y, z: left_knee_pos.z, visibility: 1.0 };
        keypoints[27] = Keypoint { x: left_ankle_pos.x, y: left_ankle_pos.y, z: left_ankle_pos.z, visibility: 1.0 };
        
        // Right leg, arms, etc...
        
        keypoints
    }
}
```

**Finger Tapping Landmark Generator:**

```rust
pub struct FingerTapGenerator;

impl Level2Generator for FingerTapGenerator {
    fn generate_features(&self, params: &TapParams, duration: f64, fps: f64, seed: u64) -> FeatureSequence {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_frames = (duration * fps) as usize;
        let dt = 1.0 / fps;
        
        let mut landmarks = Vec::with_capacity(n_frames);
        let mut tap_events = Vec::new();
        
        // Tapping state machine
        let mut tap_phase = 0.0;  // 0-1 within tap cycle
        let mut tap_count = 0;
        let base_tap_period = 1.0 / params.frequency;
        
        for frame in 0..n_frames {
            let t = frame as f64 * dt;
            
            // Compute current tap period (with decrement/fatigue)
            let current_period = base_tap_period * (1.0 + params.frequency_decrement * tap_count as f64);
            
            // Compute current amplitude (with decrement)
            let current_amplitude = params.amplitude * (1.0 - params.amplitude_decrement * tap_count as f64).max(0.2);
            
            // Advance phase
            tap_phase += dt / current_period;
            if tap_phase >= 1.0 {
                tap_phase -= 1.0;
                tap_count += 1;
                tap_events.push(TapEvent {
                    time: t,
                    amplitude: current_amplitude,
                    interval: current_period,
                });
            }
            
            // Generate thumb-index aperture from phase
            // Using smooth trajectory (raised cosine)
            let aperture = current_amplitude * 0.5 * (1.0 - (2.0 * PI * tap_phase).cos());
            
            // Generate 21 hand landmarks
            let hand = self.generate_hand_landmarks(aperture, params, &mut rng);
            
            // Add tracking noise
            self.add_landmark_noise(&mut hand, params.noise_std, &mut rng);
            
            landmarks.push(hand);
        }
        
        FeatureSequence { ... }
    }
}
```

**Voice Feature Generator:**

```rust
pub struct VoiceFeatureGenerator;

impl Level2Generator for VoiceFeatureGenerator {
    fn generate_features(&self, params: &VoiceParams, duration: f64, feature_rate: f64, seed: u64) -> FeatureSequence {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_frames = (duration * feature_rate) as usize;
        let dt = 1.0 / feature_rate;
        
        let mut features = Vec::with_capacity(n_frames);
        
        for frame in 0..n_frames {
            let t = frame as f64 * dt;
            
            // Generate F0 trajectory
            let f0 = self.generate_f0(t, params, &mut rng);
            
            // Generate formants (F1-F4)
            let formants = self.generate_formants(t, params, &mut rng);
            
            // Generate voice quality measures
            let jitter = params.jitter_base + rng.gen::<f64>() * params.jitter_std;
            let shimmer = params.shimmer_base + rng.gen::<f64>() * params.shimmer_std;
            let hnr = params.hnr_base + rng.gen::<f64>() * params.hnr_std;
            
            // Generate MFCCs (or compute from spectral model)
            let mfccs = self.generate_mfccs(f0, &formants, params, &mut rng);
            
            features.push(VoiceFrame {
                f0,
                formants,
                jitter,
                shimmer,
                hnr,
                mfccs,
                voiced: f0 > 0.0,
            });
        }
        
        FeatureSequence { ... }
    }
    
    fn generate_f0(&self, t: f64, params: &VoiceParams, rng: &mut impl Rng) -> f64 {
        if !self.is_voiced(t, params) {
            return 0.0;  // Unvoiced/silence
        }
        
        // Base F0 with declination
        let base_f0 = params.f0_mean - params.declination_rate * t;
        
        // Add micro-prosody (phrase-level contour)
        let contour = params.f0_range * 0.5 * (2.0 * PI * t / params.phrase_duration).sin();
        
        // Add jitter (cycle-to-cycle variation)
        let jitter = rng.gen::<f64>() * params.f0_jitter * base_f0;
        
        // Apply pathology
        let pathology_effect = match &params.pathology {
            Some(Pathology::Parkinsonian { severity }) => {
                // Reduced range, monotone
                -0.5 * severity * contour
            }
            Some(Pathology::Tremor { frequency, amplitude }) => {
                // Vocal tremor
                amplitude * (2.0 * PI * frequency * t).sin()
            }
            None => 0.0,
        };
        
        (base_f0 + contour + jitter + pathology_effect).max(50.0)
    }
}
```

**Eye Tracking Feature Generator:**

```rust
pub struct EyeTrackingFeatureGenerator {
    main_sequence: MainSequenceModel,  // Velocity = f(amplitude)
}

impl Level2Generator for EyeTrackingFeatureGenerator {
    fn generate_features(&self, params: &EyeParams, duration: f64, sample_rate: f64, seed: u64) -> FeatureSequence {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_samples = (duration * sample_rate) as usize;
        let dt = 1.0 / sample_rate;
        
        let mut gaze_x = Vec::with_capacity(n_samples);
        let mut gaze_y = Vec::with_capacity(n_samples);
        let mut pupil = Vec::with_capacity(n_samples);
        let mut events = Vec::new();
        
        // State
        let mut current_pos = (0.0_f64, 0.0_f64);  // degrees
        let mut current_pupil = params.pupil_baseline;
        let mut in_saccade = false;
        let mut fixation_start = 0.0;
        
        for sample in 0..n_samples {
            let t = sample as f64 * dt;
            
            // Decide if saccade should occur
            if !in_saccade && self.should_saccade(t, &params, &mut rng) {
                // Generate saccade
                let target = self.generate_saccade_target(params, &mut rng);
                let amplitude = ((target.0 - current_pos.0).powi(2) + 
                                (target.1 - current_pos.1).powi(2)).sqrt();
                
                // Main sequence: duration and peak velocity from amplitude
                let (duration, peak_vel) = self.main_sequence.predict(amplitude, params);
                
                // Apply pathology
                let (duration, peak_vel) = self.apply_saccade_pathology(
                    duration, peak_vel, amplitude, params
                );
                
                events.push(EyeEvent::SaccadeOnset {
                    time: t,
                    amplitude,
                    direction: (target.1 - current_pos.1).atan2(target.0 - current_pos.0),
                    latency: t - fixation_start,
                });
                
                in_saccade = true;
                // ... execute saccade trajectory over next samples
            }
            
            // Fixation drift + microsaccades
            if !in_saccade {
                current_pos.0 += rng.gen::<f64>() * params.fixation_noise;
                current_pos.1 += rng.gen::<f64>() * params.fixation_noise;
                
                // Occasional microsaccade
                if rng.gen::<f64>() < params.microsaccade_rate * dt {
                    events.push(EyeEvent::Microsaccade { time: t });
                    // Small position jump
                    current_pos.0 += (rng.gen::<f64>() - 0.5) * 0.5;
                    current_pos.1 += (rng.gen::<f64>() - 0.5) * 0.5;
                }
            }
            
            // Pupil dynamics
            current_pupil = self.update_pupil(current_pupil, t, params, &mut rng);
            
            gaze_x.push(current_pos.0 as f32);
            gaze_y.push(current_pos.1 as f32);
            pupil.push(current_pupil as f32);
        }
        
        FeatureSequence { ... }
    }
}
```

### 3.4 Level 2 Generator Inventory

| # | Generator | Output Features | Frame Rate |
|---|-----------|-----------------|------------|
| 1 | Gait keypoints | 33 keypoints × 3D | 30 fps |
| 2 | Hand landmarks | 21 landmarks × 3D | 30 fps |
| 3 | Face landmarks | 468 landmarks × 3D | 30 fps |
| 4 | Finger tapping | Thumb-index aperture + landmarks | 30 fps |
| 5 | Tremor trajectory | 3D hand/finger position | 30-60 fps |
| 6 | Gaze position | x, y coordinates | 60-1000 Hz |
| 7 | Pupil diameter | Diameter in mm | 60-1000 Hz |
| 8 | Saccade/fixation labels | Event sequence | Event-based |
| 9 | F0 trajectory | Pitch in Hz | 100 Hz |
| 10 | Formant trajectories | F1-F4 in Hz | 100 Hz |
| 11 | Voice quality | Jitter, shimmer, HNR | Per-frame |
| 12 | MFCCs | 13-40 coefficients | 100 Hz |
| 13 | Spectral features | Energy, spectral slope | 100 Hz |
| 14 | Prosodic features | Rate, pauses, emphasis | Per-utterance |

### 3.5 Advantages & Limitations

| Advantages | Limitations |
|------------|-------------|
| Tests DPB algorithms directly | Doesn't test feature extraction |
| No need for media synthesis | Feature correlations may be simplified |
| Fast generation | Misses extraction artifacts |
| Ground truth is exact | May not capture all MediaPipe quirks |

---

## 4. Level 3: Raw Media Synthesis

### 4.1 What It Is

Generation of actual video frames and audio waveforms that, when processed by MediaPipe/Praat/etc., produce the desired features. Tests the full pipeline including feature extraction.

### 4.2 Applicable Modalities

| Media Type | Resolution | Rate | Challenge |
|------------|------------|------|-----------|
| Gait video | 1920×1080 | 30 fps | Full body, clothing, lighting |
| Hand video | 640×480 | 30 fps | Skin texture, self-occlusion |
| Face video | 640×480 | 30 fps | Expression, lighting, identity |
| Eye close-up | 640×480 | 60-120 fps | Iris texture, reflections |
| Speech audio | 16-48 kHz | - | Natural voice, speaker identity |

### 4.3 Implementation Approaches

#### 4.3.1 Physics-Based Rendering (Video)

**Approach:** Use 3D modeling + rendering (Blender, Unity, Unreal) with biomechanical animation.

```
┌─────────────────────────────────────────────────────────────────┐
│                 PHYSICS-BASED VIDEO SYNTHESIS                    │
│                                                                  │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│   │ Biomechanical│──▶│   3D Model   │──▶│   Renderer   │        │
│   │   Parameters │   │  Animation   │   │  (Blender)   │        │
│   └──────────────┘   └──────────────┘   └──────────────┘        │
│          │                  │                  │                 │
│          ▼                  ▼                  ▼                 │
│   • Joint angles      • Rigged skeleton   • Lighting            │
│   • Cadence           • Mesh deformation  • Camera              │
│   • Pathology         • Clothing sim      • Background          │
│                                            • Render to frames    │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation (Blender Python API):**

```python
# blender_gait_generator.py
import bpy
import mathutils
import numpy as np
from pathlib import Path

class GaitVideoGenerator:
    def __init__(self, model_path: str):
        # Load rigged human model (e.g., SMPL, MakeHuman)
        bpy.ops.import_scene.fbx(filepath=model_path)
        self.armature = bpy.data.objects['Armature']
        self.setup_scene()
    
    def setup_scene(self):
        # Camera setup
        cam = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam)
        bpy.context.scene.collection.objects.link(cam_obj)
        cam_obj.location = (0, -5, 1.5)  # Side view
        cam_obj.rotation_euler = (np.pi/2, 0, 0)
        bpy.context.scene.camera = cam_obj
        
        # Lighting
        light = bpy.data.lights.new("Light", 'SUN')
        light_obj = bpy.data.objects.new("Light", light)
        bpy.context.scene.collection.objects.link(light_obj)
        
        # Render settings
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080
        bpy.context.scene.render.fps = 30
    
    def generate_gait_animation(self, params: dict, duration_sec: float):
        """Generate gait animation from biomechanical parameters"""
        fps = bpy.context.scene.render.fps
        n_frames = int(duration_sec * fps)
        
        # Clear existing animation
        self.armature.animation_data_clear()
        
        # Create action
        action = bpy.data.actions.new("GaitAction")
        self.armature.animation_data_create()
        self.armature.animation_data.action = action
        
        # Gait cycle parameters
        cycle_duration = 60.0 / params['cadence']  # seconds
        
        for frame in range(n_frames):
            t = frame / fps
            phase = (t / cycle_duration) % 1.0
            
            # Get joint angles from gait model
            angles = self.compute_gait_angles(phase, params)
            
            # Apply to bones
            self.set_bone_rotation('hip_L', angles['left_hip'], frame)
            self.set_bone_rotation('hip_R', angles['right_hip'], frame)
            self.set_bone_rotation('knee_L', angles['left_knee'], frame)
            self.set_bone_rotation('knee_R', angles['right_knee'], frame)
            # ... etc
            
            # Root motion (forward walking)
            stride_length = params['stride_length']
            self.set_root_position(t * stride_length / cycle_duration, frame)
    
    def render_video(self, output_path: str):
        """Render animation to video file"""
        bpy.context.scene.render.filepath = output_path
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        bpy.context.scene.render.ffmpeg.format = 'MPEG4'
        bpy.context.scene.render.ffmpeg.codec = 'H264'
        
        bpy.ops.render.render(animation=True)
```

**Alternative: Unity/Unreal for Real-Time:**

```csharp
// Unity C# - Real-time gait animation
public class GaitAnimationController : MonoBehaviour
{
    public Animator animator;
    public GaitParameters parameters;
    
    private float phase = 0f;
    
    void Update()
    {
        float cycleDuration = 60f / parameters.cadence;
        phase += Time.deltaTime / cycleDuration;
        if (phase >= 1f) phase -= 1f;
        
        // Set animator parameters (drives IK/FK)
        animator.SetFloat("GaitPhase", phase);
        animator.SetFloat("StrideLength", parameters.strideLength);
        animator.SetFloat("Asymmetry", parameters.asymmetry);
        animator.SetFloat("TremorAmplitude", parameters.tremorAmplitude);
    }
}
```

#### 4.3.2 Generative AI (Video)

**Approach:** Fine-tune video diffusion models or use motion-conditioned generation.

```
┌─────────────────────────────────────────────────────────────────┐
│              GENERATIVE AI VIDEO SYNTHESIS                       │
│                                                                  │
│   Option A: Motion-Conditioned Diffusion                         │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│   │   Keypoint   │──▶│ ControlNet/  │──▶│   Output     │        │
│   │  Trajectory  │   │ Motion-Cond  │   │   Video      │        │
│   │ (Level 2)    │   │  Diffusion   │   │              │        │
│   └──────────────┘   └──────────────┘   └──────────────┘        │
│                                                                  │
│   Option B: Text/Parameter-Conditioned                           │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│   │   "Person    │──▶│  Video Gen   │──▶│   Output     │        │
│   │   walking    │   │   Model      │   │   Video      │        │
│   │   with PD"   │   │  (fine-tuned)│   │              │        │
│   └──────────────┘   └──────────────┘   └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

**Models to Consider:**

| Model | Type | Conditioning | Notes |
|-------|------|--------------|-------|
| ControlNet + SD | Image→Video | Pose skeleton | Per-frame control |
| AnimateDiff | Text→Video | Text prompt | Limited control |
| MagicAnimate | Pose→Video | DensePose | Good for body |
| DisCo | Dance→Video | Keypoints | Motion transfer |
| DreamPose | Pose→Video | Skeleton | Fashion/body |
| Custom fine-tune | Any | Clinical data | Requires real data |

**Implementation (ControlNet + Pose):**

```python
# Using ControlNet with OpenPose conditioning
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import cv2
import numpy as np

class PoseConditionedVideoGenerator:
    def __init__(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose"
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        ).to("cuda")
    
    def generate_video(
        self,
        keypoint_sequence: np.ndarray,  # Level 2 output: (n_frames, 33, 3)
        prompt: str = "person walking, side view, clinical setting",
        output_path: str = "output.mp4",
    ):
        frames = []
        
        for frame_idx, keypoints in enumerate(keypoint_sequence):
            # Render keypoints to skeleton image
            skeleton_image = self.render_skeleton(keypoints)
            
            # Generate frame
            image = self.pipe(
                prompt=prompt,
                image=skeleton_image,
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]
            
            frames.append(np.array(image))
        
        # Write video
        self.write_video(frames, output_path, fps=30)
    
    def render_skeleton(self, keypoints: np.ndarray) -> np.ndarray:
        """Render keypoints to OpenPose-style skeleton image"""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Draw bones
        bones = [(0, 1), (1, 2), (2, 3), ...]  # OpenPose connectivity
        for (i, j) in bones:
            pt1 = (int(keypoints[i, 0] * 512), int(keypoints[i, 1] * 512))
            pt2 = (int(keypoints[j, 0] * 512), int(keypoints[j, 1] * 512))
            cv2.line(img, pt1, pt2, (255, 255, 255), 2)
        
        # Draw joints
        for kp in keypoints:
            cv2.circle(img, (int(kp[0] * 512), int(kp[1] * 512)), 4, (0, 255, 0), -1)
        
        return img
```

#### 4.3.3 Speech Synthesis (Audio)

**Approach:** Use neural TTS with control over voice quality parameters.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPEECH AUDIO SYNTHESIS                        │
│                                                                  │
│   Option A: Parametric Vocoder (Full Control)                    │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│   │ F0, formants │──▶│   WORLD /    │──▶│   Audio      │        │
│   │ jitter, etc. │   │   STRAIGHT   │   │   Waveform   │        │
│   │ (Level 2)    │   │   Vocoder    │   │              │        │
│   └──────────────┘   └──────────────┘   └──────────────┘        │
│                                                                  │
│   Option B: Neural TTS + Voice Conversion                        │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│   │    Text +    │──▶│  TTS Model   │──▶│   Voice      │──▶Audio│
│   │   Phonemes   │   │ (VITS, etc.) │   │  Conversion  │        │
│   └──────────────┘   └──────────────┘   └──────────────┘        │
│                             ▲                                    │
│                             │                                    │
│                    Pathology conditioning                        │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation (WORLD Vocoder - Full Parameter Control):**

```python
# Using pyworld for parametric speech synthesis
import pyworld as pw
import numpy as np
from scipy.io import wavfile

class ParametricSpeechGenerator:
    """Generate speech from acoustic parameters with full control"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.frame_period = 5.0  # ms
    
    def generate_sustained_vowel(
        self,
        duration_sec: float,
        params: dict,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate sustained vowel /a/ with specified characteristics"""
        rng = np.random.RandomState(seed)
        n_frames = int(duration_sec * 1000 / self.frame_period)
        
        # Generate F0 contour
        f0 = self.generate_f0_contour(n_frames, params, rng)
        
        # Generate spectral envelope (formants)
        sp = self.generate_spectral_envelope(n_frames, params, rng)
        
        # Generate aperiodicity (breathiness/noise)
        ap = self.generate_aperiodicity(n_frames, params, rng)
        
        # Synthesize using WORLD
        audio = pw.synthesize(f0, sp, ap, self.sr, self.frame_period)
        
        return audio
    
    def generate_f0_contour(self, n_frames: int, params: dict, rng) -> np.ndarray:
        """Generate F0 with jitter and pathological variations"""
        base_f0 = params['f0_mean']
        
        # Base contour with slow variation
        f0 = np.ones(n_frames) * base_f0
        
        # Add declination
        f0 -= np.linspace(0, params.get('declination', 5), n_frames)
        
        # Add jitter (cycle-to-cycle variation)
        jitter_percent = params.get('jitter', 0.5) / 100.0
        for i in range(1, n_frames):
            f0[i] += rng.randn() * base_f0 * jitter_percent
        
        # Add vocal tremor if present
        if 'tremor_frequency' in params:
            t = np.arange(n_frames) * self.frame_period / 1000
            tremor = params['tremor_amplitude'] * np.sin(2 * np.pi * params['tremor_frequency'] * t)
            f0 += tremor
        
        return f0.astype(np.float64)
    
    def generate_spectral_envelope(self, n_frames: int, params: dict, rng) -> np.ndarray:
        """Generate spectral envelope with formants"""
        fft_size = 1024
        sp = np.zeros((n_frames, fft_size // 2 + 1))
        
        # Formant frequencies for vowel /a/
        formants = params.get('formants', [700, 1200, 2600, 3500])
        bandwidths = params.get('bandwidths', [80, 90, 120, 150])
        
        freqs = np.linspace(0, self.sr / 2, fft_size // 2 + 1)
        
        for i in range(n_frames):
            # Sum of resonances
            envelope = np.zeros_like(freqs)
            for f, bw in zip(formants, bandwidths):
                # Add shimmer variation
                shimmer = 1.0 + rng.randn() * params.get('shimmer', 3) / 100
                resonance = shimmer / (1 + ((freqs - f) / bw) ** 2)
                envelope += resonance
            
            sp[i] = envelope
        
        return sp.astype(np.float64)
    
    def generate_aperiodicity(self, n_frames: int, params: dict, rng) -> np.ndarray:
        """Generate aperiodicity (noise component)"""
        fft_size = 1024
        
        # Base aperiodicity from HNR
        hnr_db = params.get('hnr', 22)
        base_ap = 10 ** (-hnr_db / 20)  # Convert HNR to aperiodicity ratio
        
        ap = np.ones((n_frames, fft_size // 2 + 1)) * base_ap
        
        # More breathiness at high frequencies
        freqs = np.linspace(0, self.sr / 2, fft_size // 2 + 1)
        ap *= (1 + freqs / (self.sr / 2))
        
        return ap.astype(np.float64)


class NeuralTTSWithPathology:
    """Use neural TTS and modify output for pathological characteristics"""
    
    def __init__(self):
        # Load VITS or similar model
        from TTS.api import TTS
        self.tts = TTS(model_name="tts_models/en/ljspeech/vits")
    
    def generate_with_pathology(
        self,
        text: str,
        pathology_params: dict,
        output_path: str,
    ):
        # Generate clean speech
        clean_audio = self.tts.tts(text)
        
        # Extract acoustic parameters using WORLD
        f0, sp, ap = pw.wav2world(clean_audio, self.tts.synthesizer.output_sample_rate)
        
        # Modify parameters for pathology
        f0_modified = self.apply_f0_pathology(f0, pathology_params)
        sp_modified = self.apply_spectral_pathology(sp, pathology_params)
        ap_modified = self.apply_aperiodicity_pathology(ap, pathology_params)
        
        # Resynthesize
        audio = pw.synthesize(f0_modified, sp_modified, ap_modified, 
                             self.tts.synthesizer.output_sample_rate, 5.0)
        
        wavfile.write(output_path, self.tts.synthesizer.output_sample_rate, 
                     (audio * 32767).astype(np.int16))
    
    def apply_f0_pathology(self, f0: np.ndarray, params: dict) -> np.ndarray:
        """Apply PD-like F0 modifications"""
        f0_mod = f0.copy()
        
        # Reduce pitch range (monotone)
        if params.get('monotone_severity', 0) > 0:
            mean_f0 = np.mean(f0[f0 > 0])
            f0_mod[f0 > 0] = mean_f0 + (f0[f0 > 0] - mean_f0) * (1 - params['monotone_severity'])
        
        # Add tremor
        if params.get('tremor_frequency', 0) > 0:
            t = np.arange(len(f0)) / 200  # Assuming 200 Hz frame rate
            tremor = params['tremor_amplitude'] * np.sin(2 * np.pi * params['tremor_frequency'] * t)
            f0_mod[f0 > 0] += tremor[f0 > 0]
        
        # Add increased jitter
        jitter_scale = params.get('jitter_scale', 1.0)
        noise = np.random.randn(len(f0)) * np.mean(f0[f0 > 0]) * 0.01 * jitter_scale
        f0_mod[f0 > 0] += noise[f0 > 0]
        
        return f0_mod
```

### 4.4 Level 3 Implementation Summary

| Media Type | Recommended Approach | Tools | Realism | Control |
|------------|---------------------|-------|---------|---------|
| Gait video | Physics-based (Blender) | Blender + SMPL | High | High |
| Hand video | Physics-based + ControlNet | Blender/Unity + SD | Medium-High | High |
| Face video | Generative AI | DreamFace, SadTalker | High | Medium |
| Eye close-up | Physics-based | Custom renderer | Medium | High |
| Speech audio | Parametric vocoder | WORLD/STRAIGHT | Medium-High | Very High |
| Connected speech | Neural TTS + modification | VITS + WORLD | High | Medium |

### 4.5 Level 3 Generator Inventory

| # | Generator | Output | Approach |
|---|-----------|--------|----------|
| 1 | Gait video | MP4 1080p 30fps | Blender + biomech model |
| 2 | Hand/finger video | MP4 720p 30fps | Blender + hand rig |
| 3 | Finger tapping video | MP4 720p 30fps | Blender + task animation |
| 4 | Face video | MP4 720p 30fps | Generative AI (SadTalker) |
| 5 | Eye close-up video | MP4 720p 60fps | Custom renderer |
| 6 | Sustained vowel audio | WAV 16kHz | WORLD vocoder |
| 7 | Connected speech | WAV 16kHz | Neural TTS + WORLD |
| 8 | Diadochokinesis | WAV 16kHz | Template concatenation |
| 9 | Reading passage | WAV 16kHz | Neural TTS + pathology |

### 4.6 Advantages & Limitations

| Advantages | Limitations |
|------------|-------------|
| Tests full pipeline | Computationally expensive |
| Catches extraction bugs | Harder to guarantee ground truth |
| Most realistic validation | Requires specialized tools (Blender, GPUs) |
| Can distribute without privacy issues | Video generation still imperfect |

---

## 5. Summary: When to Use Each Level

| Level | Use Case | Development Phase |
|-------|----------|-------------------|
| **Level 1** | Algorithm development, unit tests, convergence experiments | Early R&D |
| **Level 2** | Integration tests, ablation studies, benchmark comparisons | Mid R&D |
| **Level 3** | Full pipeline validation, publication figures, demo videos | Late R&D / Validation |

### Generator Count by Level

| Level | Contact | Pose/Gait | Hand | Eye | Voice | Total |
|-------|---------|-----------|------|-----|-------|-------|
| Level 1 | 42 | - | - | - | - | 42 |
| Level 2 | - | 14 | 12 | 10 | 14 | 50 |
| Level 3 | - | 2 | 2 | 1 | 4 | 9 |
| **Total** | 42 | 16 | 14 | 11 | 18 | **101** |

### Recommended Workflow

```
1. Start with Level 1 + Level 2
   └── Fast iteration on algorithms
   └── Full parameter control
   └── Exact ground truth
   
2. Validate with Level 3 before publication
   └── Ensure extraction pipeline works
   └── Realistic demo videos
   └── Confidence in real-world performance
   
3. Final validation on real data
   └── Clinical datasets
   └── IRB-approved studies
```

---

*Document: DPB Tiered Synthetic Data Generation v1.0*
*AuraSense Tech Corporation - December 2025*
