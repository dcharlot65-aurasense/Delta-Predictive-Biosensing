# Delta Predictive Biosensing - Algorithm Reference

This document provides a comprehensive inventory of all algorithms implemented in the Delta Predictive Biosensing framework.

---

## Table of Contents

1. [Signal Processing Algorithms](#signal-processing-algorithms)
2. [Neuron Models](#neuron-models)
3. [SNN Training Algorithms](#snn-training-algorithms)
4. [SNN Layers](#snn-layers)
5. [SNN Architectures](#snn-architectures)
6. [SNN Decoders](#snn-decoders)
7. [Fusion Networks](#fusion-networks)
8. [Baseline Models (ANN)](#baseline-models-ann)
9. [Metrics](#metrics)
10. [Power Estimation Models](#power-estimation-models)
11. [Training Analysis Algorithms](#training-analysis-algorithms)
12. [Synthetic Data Generators](#synthetic-data-generators)
13. [Encoder Algorithms & Population Templates](#encoder-algorithms--population-templates)
14. [External Media Interfaces](#external-media-interfaces)
15. [Summary](#summary)

---

## Signal Processing Algorithms

**Location:** `crates/dpb-core/src/signal/`

### FFT & Spectral Analysis

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| Fast Fourier Transform (FFT) | `fft.rs:24` | Computes forward FFT of signals |
| Inverse FFT (IFFT) | `fft.rs:40` | Computes inverse FFT for signal reconstruction |
| Power Spectral Density (PSD) | `fft.rs:60` | Computes magnitude-squared spectrum |
| Magnitude Spectrum | `fft.rs:67` | Extracts magnitude from frequency components |
| Phase Spectrum | `fft.rs:74` | Extracts phase from frequency components |
| Short-Time Fourier Transform (STFT) | `fft.rs:115` | Time-frequency decomposition with windowing |
| Magnitude Spectrogram | `fft.rs:152` | Magnitude-based time-frequency representation |
| Power Spectrogram | `fft.rs:159` | Power-based time-frequency representation |
| FFT Frequency Bins | `fft.rs:244` | Computes frequency bins for FFT output |
| STFT Time Bins | `fft.rs:250` | Computes time bins for STFT frames |

### Window Functions

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| Hann Window | `fft.rs:186` | Hann windowing function (cosine-based) |
| Hamming Window | `fft.rs:191` | Hamming windowing function |
| Blackman Window | `fft.rs:196` | Blackman windowing function (3-term) |
| Kaiser Window | `fft.rs:204-206` | Kaiser windowing with Bessel function |
| Modified Bessel I0 Function | `fft.rs:225` | Bessel I0 approximation for Kaiser window |

### Digital Filtering

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| IIR Butterworth Lowpass Filter | `filter.rs:48` | Butterworth IIR lowpass design |
| IIR Butterworth Highpass Filter | `filter.rs:60` | Butterworth IIR highpass design |
| IIR Butterworth Bandpass Filter | `filter.rs:72` | Butterworth IIR bandpass design |
| IIR Filter Sample Processing | `filter.rs:90` | Single-sample IIR filtering (difference equation) |
| FIR Moving Average Filter | `filter.rs:160` | Moving average FIR filter |
| FIR Filter Sample Processing | `filter.rs:173` | Single-sample FIR filtering |
| Median Filter | `filter.rs:231` | Non-linear median filtering |
| Butterworth Design Algorithm | `filter.rs:196` | Simplified Butterworth coefficient generation |

### Resampling

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| Linear Interpolation Resampling | `resample.rs:7` | Linear interpolation-based resampling |
| Downsampling | `resample.rs:45` | Decimation by integer factor |
| Upsampling | `resample.rs:61` | Interpolation to increase sample rate |
| Fourier Method Resampling | `resample.rs:100` | FFT-based sinc interpolation |
| Fixed-Length Resampling | `resample.rs:117` | Resample to specific target length |
| Anti-aliasing Resampling | `resample.rs:143` | Resampling with anti-aliasing filter |
| Moving Average Helper | `resample.rs:166` | Helper for anti-aliasing |

---

## Neuron Models

**Location:** `crates/dpb-neurons/src/`

### Conductance-Based Models

| Model | File:Line | Description |
|-------|-----------|-------------|
| Hodgkin-Huxley Model | `hodgkin_huxley.rs:94` | Full conductance-based model with Na⁺, K⁺, leak channels. Includes gating variables: m (Na activation), h (Na inactivation), n (K activation) |

### Simplified Spiking Models

| Model | File:Line | Description |
|-------|-----------|-------------|
| Izhikevich Model | `izhikevich.rs:15` | Simple yet biologically realistic spiking neuron. Equations: `dv/dt = 0.04v² + 5v + 140 - u + I`; `du/dt = a(bv - u)`. Multiple presets: Regular Spiking, Fast Spiking, Bursting, etc. |
| Leaky Integrate-and-Fire (LIF) | `lif.rs:20` | Fundamental spiking neuron model. Equation: `dv/dt = (v_rest - v)/tau + R*I` |
| Integrate-and-Fire (IF) | `lif.rs:14` | Simplest spiking model (no leak). Equation: `dv/dt = I` |
| Adaptive Exponential IF (AdEx) | `adex.rs:94` | Combines exponential spike with adaptation. Equations: `C_m*dv/dt = -g_L(v-E_L) + g_L*Δ_T*exp((v-V_T)/Δ_T) - w + I`; Adaptation: `τ_w*dw/dt = a(v-E_L) - w` |

### Spike Timing Models

| Model | File:Line | Description |
|-------|-----------|-------------|
| Spike Response Model (SRM) | `srm.rs:20` | Kernel-based neuron using spike history convolution. Uses synaptic and refractory kernels |

### Calcium Dynamics

| Model | File:Line | Description |
|-------|-----------|-------------|
| Calcium Neuron Model | `calcium.rs:14` | Model with Ca²⁺ dynamics for adaptation. Includes intracellular calcium and AHP (afterhyperpolarization) currents. Provides spike-frequency adaptation |

### Recurrent & Intrinsic Dynamics

| Model | File:Line | Description |
|-------|-----------|-------------|
| Recurrent Neuron | `recurrent.rs:14` | Neuron with self-connections (autapses). Intrinsic dynamics through recurrent current. Presets: Excitatory, Inhibitory, Oscillatory |

### Stochastic Models

| Model | File:Line | Description |
|-------|-----------|-------------|
| Stochastic LIF Neuron | `stochastic.rs:25` | LIF with noise injection. Noise types: Gaussian, Ornstein-Uhlenbeck (colored), Multiplicative |

### Surrogate Gradient Functions

| Function | File:Line | Formula |
|----------|-----------|---------|
| Fast Sigmoid Surrogate | `surrogate.rs:34` | `g(x) = 1/(slope*|x|+1)²` |
| Arctan Surrogate | `surrogate.rs:68` | `g(x) = 1/(π*slope*(1+(slope*x)²))` |
| Triangular Surrogate | `surrogate.rs:96` | Piecewise linear surrogate gradient |

---

## SNN Training Algorithms

**Location:** `crates/dpb-snn/src/training/`

### Loss Functions

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| Spiking Cross-Entropy Loss | `loss.rs:18` | Cross-entropy based on spike rates. Normalizes spike rates to probabilities via softmax |
| Spike Count Loss | `loss.rs:92` | MSE between actual and target spike counts |

### Optimizers

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| Stochastic Gradient Descent (SGD) with Momentum | `optimizer.rs:25` | Update: `v = momentum*v - lr*grad`; `param = param + v` |
| Adam Optimizer | `optimizer.rs:91` | Adaptive learning rates with exponential moving averages |

### Surrogate Gradient Methods

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| Box Surrogate Gradient | `surrogate.rs:33` | Rectangular window surrogate |
| Fast Sigmoid Surrogate Gradient | `surrogate.rs:70` | Smooth sigmoid-based surrogate |
| Exponential Surrogate | `surrogate.rs` | Exponential decay surrogate function |
| SuperSpike Surrogate | `surrogate.rs` | Advanced surrogate for better gradients |

---

## SNN Layers

**Location:** `crates/dpb-snn/src/layers/`

| Layer | File | Description |
|-------|------|-------------|
| Spiking Convolutional Layer (Conv2d) | `conv.rs` | 2D convolution with spiking neurons |
| Spiking Linear Layer | `linear.rs` | Fully connected layer with spiking neurons |
| Spiking Attention Layer | `attention.rs:12` | Multi-head self-attention with spikes. Scaled dot-product attention: `softmax(Q*K^T/√d)*V` |
| Spiking Sum Pooling (Pool2d) | `pool.rs:10` | 2D sum pooling for spike aggregation |
| Spiking Max Pooling (MaxPool2d) | `pool.rs:94` | 2D max pooling over spike values |
| Spiking Recurrent Layer | `recurrent.rs` | Recurrent connections for temporal processing |

---

## SNN Architectures

**Location:** `crates/dpb-snn/src/architectures/`

| Architecture | File | Description |
|--------------|------|-------------|
| Convolutional SNN | `convolutional.rs:13` | Conv layers with pooling and FC layers |
| Feedforward SNN | `feedforward.rs` | Multi-layer fully connected SNN |
| Recurrent SNN | `recurrent.rs` | SNN with recurrent connections |
| Graph SNN | `graph.rs` | Graph neural network with spiking neurons |
| Transformer-based SNN | `transformer.rs` | Attention-based SNN architecture |

---

## SNN Decoders

**Location:** `crates/dpb-snn/src/decoders/`

| Decoder | File:Line | Description |
|---------|-----------|-------------|
| Spike Rate Decoder | `rate.rs:10` | Decodes based on average firing rate with optional softmax |
| Temporal Pattern Decoder | `temporal.rs:10` | Pattern matching using normalized cross-correlation |
| Classification Decoder | `classification.rs` | Softmax classification from spike patterns |
| Regression Decoder | `regression.rs` | Continuous value prediction from spike trains |
| Clinical Decoder (UPDRS) | `clinical.rs:10` | Unified Parkinson's Disease Rating Scale decoder. Decodes motor assessment scores from spike activity |

---

## Fusion Networks

**Location:** `crates/dpb-snn/src/fusion/`

| Network | File | Description |
|---------|------|-------------|
| Early Fusion | `early.rs` | Concatenate modalities before processing |
| Late Fusion | `late.rs` | Combine outputs from separate modality streams |
| Cross-Modal Attention | `attention.rs:13` | Each modality attends to others |
| Gated Fusion | `gated.rs:13` | Adaptive modality weighting via gates |
| Hierarchical Fusion | `hierarchical.rs` | Multi-level fusion structure |
| Temporal Fusion | `temporal.rs` | Sequential fusion across time |
| NeuroPlay Fusion | `neuroplay.rs` | Specialized fusion for neuromorphic applications |

---

## Baseline Models (ANN)

**Location:** `crates/dpb-snn/src/baselines/`

| Model | File:Line | Description |
|-------|-----------|-------------|
| MLP 2-Layer | `mlp.rs:6` | input → hidden → output with ReLU |
| MLP 3-Layer | `mlp.rs:54` | input → hidden1 → hidden2 → output |
| Convolutional Neural Network (CNN) | `cnn.rs` | Standard CNN baseline |
| Recurrent Neural Network (RNN) | `rnn.rs` | RNN baseline for sequential data |
| Transformer | `transformer.rs` | Transformer baseline |
| Specialized Models | `specialized.rs` | Domain-specific baseline architectures |
| ANN to SNN Conversion | `conversion.rs` | Converts trained ANN to SNN |

---

## Metrics

**Location:** `crates/dpb-core/src/metrics/`

### Classification Metrics (`classification.rs`)

| Metric | Description |
|--------|-------------|
| Accuracy | Proportion of correct predictions |
| Precision | TP/(TP+FP) - correct positive predictions |
| Recall | TP/(TP+FN) - coverage of actual positives |
| F1 Score | Harmonic mean of precision and recall |
| Confusion Matrix | TP, TN, FP, FN counts |

### Regression Metrics (`regression.rs`)

| Metric | Description |
|--------|-------------|
| Mean Absolute Error (MAE) | Average absolute difference |
| Mean Squared Error (MSE) | Average squared difference |
| Root Mean Squared Error (RMSE) | sqrt(MSE) |
| R² Score | Coefficient of determination |

### Signal Quality Metrics (`signal.rs`)

| Metric | Description |
|--------|-------------|
| Signal-to-Noise Ratio (SNR) | dB scale: `10*log10(P_signal/P_noise)` |
| Peak Signal-to-Noise Ratio (PSNR) | dB scale using max signal |
| Total Harmonic Distortion (THD) | Ratio of harmonics to fundamental |

### Clinical Validation Metrics (`clinical.rs`)

| Metric | Description |
|--------|-------------|
| Intraclass Correlation Coefficient (ICC) | Between-subject vs within-subject variance |
| Bland-Altman Agreement | Mean difference and limits of agreement |
| Sensitivity/Specificity | Clinical classification performance |
| Positive/Negative Predictive Value | Clinical relevance metrics |

### Efficiency Metrics (`efficiency.rs`)

| Metric | Description |
|--------|-------------|
| Spike Rate | Spikes per second |
| Sparsity | Proportion of zero activations |
| Energy per Inference | Power consumption metrics |
| Parameter Efficiency | Ratio of performance to parameter count |

---

## Power Estimation Models

**Location:** `crates/dpb-core/src/power/`

| Model | File:Line | Description |
|-------|-----------|-------------|
| Xylo Power Estimator | `neuromorphic.rs:14` | SynSense Xylo (~1 mW active, 5 µW standby) |
| Loihi Power Estimator | `neuromorphic.rs` | Intel Loihi neuromorphic chip |
| Digital Power Model | `digital.rs` | Standard digital circuit power estimation |
| Synaptic Power Model | `synaptic.rs` | Per-synapse power consumption |
| Memory Power Model | `memory.rs` | Memory access power consumption |

---

## Training Analysis Algorithms

**Location:** `crates/dpb-snn/src/analysis/`

### Learning & Convergence

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| Learning Curve Smoothing | `learning_curves.rs:6` | Exponential moving average smoothing |
| Generalization Gap Analysis | `learning_curves.rs:98` | Training vs validation gap tracking |
| Loss Plateau Detector | `convergence.rs:7` | Detects when loss stops improving |
| Convergence Criterion | `convergence.rs` | Checks if network has converged |

### Spike Statistics

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| Spike Rate Tracker | `spike_statistics.rs:7` | Tracks firing rates across network |
| Sparsity Tracker | `spike_statistics.rs:87` | Monitors activation sparsity |
| Spike Count Distribution | `spike_statistics.rs` | Histogram of spike counts |
| Temporal Synchrony | `spike_statistics.rs` | Cross-neuronal spike timing correlation |

### Weight Analysis

| Algorithm | File:Line | Description |
|-----------|-----------|-------------|
| Weight Distribution Tracker | `weight_analysis.rs:7` | Monitors weight statistics (mean, variance, std dev, dead weights) |

### Gradient Analysis

| Algorithm | File | Description |
|-----------|------|-------------|
| Gradient Flow Analysis | `gradient_analysis.rs` | Analyzes gradient propagation |
| Vanishing/Exploding Gradient Detection | `gradient_analysis.rs` | Detects training issues |

### Comparative Analysis

| Algorithm | File | Description |
|-----------|------|-------------|
| ANN vs SNN Comparison | `comparison.rs` | Compares performance metrics |

---

## Synthetic Data Generators

**Location:** `crates/dpb-synth/src/`

### ECG (Electrocardiogram)

| Generator | File:Line | Description |
|-----------|-----------|-------------|
| McSharry ECG Model | `contact/ecg.rs:17` | Coupled ODE model for realistic ECG with P, QRS, T wave morphology, heart rate variability (HRV), arrhythmias (PAC, PVC, AF), and respiratory sinus arrhythmia (RSA) |

### Electrodermal Activity (EDA)

| Generator | File:Line | Description |
|-----------|-----------|-------------|
| Tonic EDA Generator | `contact/eda.rs:11` | Skin conductance level (SCL) with drift |
| Phasic EDA/SCR Generator | `contact/eda.rs:86` | Skin conductance response using Bateman function. Rise time (tau1): 1-3s, Recovery time (tau2): 3-10s |

### Electromyography (EMG)

| Generator | File:Line | Description |
|-----------|-----------|-------------|
| Surface EMG Generator | `contact/emg.rs:11` | Realistic EMG with baseline noise, Motor Unit Action Potentials (MUAP) biphasic spikes, and contraction-level modulation |

### Photoplethysmography (PPG)

| Generator | File | Description |
|-----------|------|-------------|
| PPG Pulse Generator | `contact/ppg.rs` | Pulse oximetry signal with pulse rate modulation, arterial vs venous components, and motion artifacts |

### Respiration

| Generator | File | Description |
|-----------|------|-------------|
| Respiratory Signal Generator | `contact/respiratory.rs` | Breathing waveforms with variable respiratory rate, breathing depth modulation, and CO2 effects |

### Voice/Speech

| Generator | File:Line | Description |
|-----------|-----------|-------------|
| Sustained Vowel Generator | `voice/phonation.rs:11` | Source-filter model generating glottal pulse train (fundamental + harmonics), formant filtering (F1, F2, F3 resonances), typical formants for vowels (/a/, /i/, /u/) |
| Articulation Generator | `voice/articulation.rs` | Speech articulation patterns |
| Prosody Generator | `voice/prosody.rs` | Intonation and pitch dynamics |

### Eye Tracking

| Generator | File:Line | Description |
|-----------|-----------|-------------|
| Saccade Generator (Main Sequence) | `eye/saccade.rs:8` | Realistic saccades with main sequence relationship: `velocity = 20*amplitude`, duration: `2.2*amplitude + 21ms`, sigmoid velocity profile |
| Pupil Response Generator | `eye/pupil.rs` | Pupil dilation/constriction |
| Eye Pursuit Generator | `eye/pursuit.rs` | Smooth pursuit eye movements |
| Fixation Generator | `eye/fixation.rs` | Steady gaze points |

### Motor Control/Gait

| Generator | File:Line | Description |
|-----------|-----------|-------------|
| Gait Cycle Generator | `pose/gait.rs:10` | Full kinematic gait model based on Winter's data with hip, knee, ankle joint angles, stance vs swing phase detection, 33 MediaPipe keypoints, cadence-based control |
| Hand Tremor Generator | `hand/tremor.rs` | Pathological and physiological tremor |
| Finger Tapping Generator | `hand/tapping.rs` | Repetitive tapping patterns |
| Hand Movement Generator | `hand/movement.rs` | General hand kinematics |

### Pose/Keypoint

| Generator | File | Description |
|-----------|------|-------------|
| Keypoint Trajectory Generator | `pose/keypoint.rs` | Body keypoint sequences |
| Gait Phase Classification | `pose/gait.rs` | Stride analysis and phase detection |
| Pathological Gait Generator | `pose/pathological.rs` | Disease-specific gait patterns |
| Gait Variability Generator | `pose/variability.rs` | Subject-to-subject variations |

### Noise Models

| Generator | File | Description |
|-----------|------|-------------|
| Contact Noise Generator | `contact/noise.rs` | Artifact and noise patterns |
| Eye Noise Generator | `eye/noise.rs` | Eye tracking noise |
| Hand Noise Generator | `hand/noise.rs` | Motion capture noise |
| Pose Noise Generator | `pose/noise.rs` | Skeletal tracking noise |
| Voice Noise Generator | `voice/noise.rs` | Speech background noise |

### Level 3 Generators (Multimodal)

| Generator | File | Description |
|-----------|------|-------------|
| Audio World Generator | `level3/audio_world.rs` | Spatial audio synthesis |
| SMPL Skeleton Generator | `level3/smpl.rs` | SMPL parametric body model |
| Video Generator | `level3/video.rs` | Photorealistic video rendering |
| Style Transfer Generator | `level3/style_transfer.rs` | Neural style transfer |
| Streaming Generators | `streaming.rs` | Real-time data streaming |

---

## Encoder Algorithms & Population Templates

**Location:** `crates/dpb-encoders/src/`

### Heart-Based Templates

| Template | File:Line | Description |
|----------|-----------|-------------|
| Heart Rate Template | `contact/ecg.rs:12` | Age-based heart rate expectations (60-140 bpm) |
| HRV Template | `contact/ecg.rs:44` | Heart rate variability norms (SDNN in ms) |
| QRS Duration Template | `contact/ecg.rs:68` | Normal QRS complex duration (80-100 ms) |
| Pulse Rate Template | `contact/ppg.rs:13` | Pulse rate based on age |
| PPG Amplitude Template | `contact/ppg.rs:39` | Perfusion index norms (1-5%) |
| Pulse Transit Time Template | `contact/ppg.rs:57` | PTT norms (150-250 ms) |

### Voice/Speech Templates

| Template | File:Line | Description |
|----------|-----------|-------------|
| Fundamental Frequency (F0) Template | `voice/phonation.rs:10` | Sex-specific F0 norms |
| Jitter Template | `voice/phonation.rs:36` | Pitch perturbation norms (<1%) |
| Shimmer Template | `voice/phonation.rs:53` | Amplitude perturbation norms (<3%) |
| HNR Template | `voice/phonation.rs:70` | Harmonics-to-Noise Ratio (>15 dB) |

### Motor Control Templates

| Template | File:Line | Description |
|----------|-----------|-------------|
| Tapping Frequency Template | `hand/tapping.rs:10` | Age-based tapping rate (5-6.5 Hz) |
| Tapping Amplitude Template | `hand/tapping.rs:34` | Finger aperture distance (~3 cm) |
| Saccade Velocity Template | `eye/saccade.rs:10` | Peak velocity (400 deg/s) |
| Saccade Latency Template | `eye/saccade.rs:29` | Reaction time (200-280 ms, age-dependent) |

### Additional Templates

- Baseline SCL Template - EDA baseline expectations
- SCR Amplitude Template - Response magnitude norms
- Muscle Activation Template - EMG amplitude norms by muscle group
- Motor Unit Recruitment Template - MUAP frequency norms

---

## External Media Interfaces

**Location:** `crates/dpb-synth/src/media/`

| Interface | File | Description |
|-----------|------|-------------|
| MediaPipe | `mediapipe.rs` | Body pose and hand tracking |
| OpenPose | `openpose.rs` | Skeleton keypoint detection |
| OpenSim | `opensim.rs` | Biomechanical simulation |
| SMPL/SMPLx | `smplx.rs` | Parametric body models |
| Blender | `blender.rs` | 3D graphics rendering |
| Bark TTS | `bark.rs` | Text-to-speech synthesis |
| F5-TTS | `f5tts.rs` | Advanced speech synthesis |
| CogVideo | `cogvideo.rs` | Video generation |
| LTX-Video | `ltx_video.rs` | Long-form video generation |
| MuJoCo | `mujoco.rs` | Physics simulation |
| PyBullet | `pybullet.rs` | Physics engine |
| Chatterbox | `chatterbox.rs` | Dialogue generation |

---

## Summary

| Category | Count | Key Algorithms |
|----------|-------|----------------|
| Signal Processing | 17 | FFT, STFT, IIR, FIR, Filters, Resampling |
| Neuron Models | 9 | Hodgkin-Huxley, Izhikevich, LIF, AdEx, SRM, Calcium, Recurrent, Stochastic |
| SNN Training | 7 | Cross-Entropy, Spike Loss, SGD, Adam, Surrogates |
| SNN Layers | 6 | Conv2d, Linear, Attention, Pooling, Recurrent |
| SNN Architectures | 5 | Convolutional, Feedforward, Recurrent, Graph, Transformer |
| SNN Decoders | 5 | Rate, Temporal, Classification, Regression, Clinical |
| Fusion Networks | 7 | Early, Late, Cross-Modal, Gated, Hierarchical, Temporal, NeuroPlay |
| Baseline Models | 7 | MLP, CNN, RNN, Transformer, Specialized, Conversion |
| Metrics | 20+ | Classification, Regression, Signal, Clinical, Efficiency |
| Power Models | 5 | Xylo, Loihi, Digital, Synaptic, Memory |
| Training Analysis | 15+ | Learning curves, Convergence, Weight, Gradient, Spike Stats |
| Data Generators | 25+ | ECG, EMG, EDA, PPG, Voice, Eye, Gait, Tremor, etc. |
| Encoders/Templates | 15+ | Population templates, Signal encoders |
| Media Interfaces | 12 | MediaPipe, OpenPose, Blender, TTS, Video, Physics |
| **Total** | **~170+** | Comprehensive biosignal & neuromorphic algorithms |

---

## License

This documentation is part of the Delta Predictive Biosensing framework.

---

*Generated: 2025-12-18*
