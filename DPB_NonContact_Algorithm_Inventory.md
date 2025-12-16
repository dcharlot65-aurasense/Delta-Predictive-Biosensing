# DPB Framework: Non-Contact Modalities Algorithm Inventory

## Study Design: Proving Delta-Predictive Biosensing for Video/Audio Analysis

---

## 1. Overview

This document catalogs all algorithms required to validate the DPB hypothesis for non-contact neurological assessment: **encoding deviations from population templates converges faster and more efficiently than raw signal processing.**

### Modalities Covered
1. **Human Pose Estimation** (full body, gait)
2. **Hand Tracking** (fine motor, tremor)
3. **Eye Tracking** (saccades, pupil, fixation)
4. **Voice/Audio Analysis** (acoustic, prosodic)

### Core Experimental Question
> Does template-based deviation encoding achieve 5-10× faster convergence to clinical-grade measurements compared to conventional frame-by-frame or sample-by-sample processing?

---

## 2. Algorithm Inventory by Layer

### Layer 1: Signal Acquisition & Preprocessing (28 algorithms)

#### 1.1 Pose Estimation Frontend
| # | Algorithm | Purpose | Variants |
|---|-----------|---------|----------|
| 1 | MediaPipe Pose | 2D/3D keypoint extraction | BlazePose, full-body |
| 2 | OpenPose | Multi-person pose | BODY_25, COCO |
| 3 | AlphaPose | Top-down pose estimation | FastPose, HRNet |
| 4 | MMPose | Modular pose estimation | Various backbones |
| 5 | ViTPose | Vision transformer pose | Base, Large, Huge |
| 6 | Temporal smoothing filter | Reduce keypoint jitter | Savitzky-Golay, OneEuro |
| 7 | Occlusion interpolation | Handle missing keypoints | Linear, spline, learned |
| 8 | Coordinate normalization | Body-relative coordinates | Hip-centered, height-normalized |

#### 1.2 Hand Tracking Frontend
| # | Algorithm | Purpose | Variants |
|---|-----------|---------|----------|
| 9 | MediaPipe Hands | 21 landmark extraction | Single/dual hand |
| 10 | Hand pose refinement | Depth-enhanced tracking | RGB-D fusion |
| 11 | Finger segmentation | Individual finger isolation | Thumb-index focus |
| 12 | Hand region cropping | ROI extraction | Detection + tracking |
| 13 | Motion blur compensation | Rapid movement handling | Deblurring, interpolation |

#### 1.3 Eye Tracking Frontend
| # | Algorithm | Purpose | Variants |
|---|-----------|---------|----------|
| 14 | Pupil detection | Center localization | Blob, ellipse fitting |
| 15 | Iris segmentation | Pupil-iris boundary | U-Net, traditional |
| 16 | Gaze estimation | Point-of-regard | Appearance, model-based |
| 17 | Calibration | Screen mapping | 5-point, 9-point |
| 18 | Head pose compensation | Gaze correction | 3D model |
| 19 | Blink detection | Artifact removal | Velocity, aspect ratio |

#### 1.4 Audio Frontend
| # | Algorithm | Purpose | Variants |
|---|-----------|---------|----------|
| 20 | Voice activity detection | Speech segmentation | Energy, WebRTC, Silero |
| 21 | Pitch extraction (F0) | Fundamental frequency | RAPT, SWIPE, CREPE, YIN |
| 22 | Formant tracking | F1-F4 estimation | LPC, Praat, DeepFormants |
| 23 | MFCC extraction | Spectral features | 13, 26, 40 coefficients |
| 24 | Spectrogram computation | Time-frequency | Mel, linear, CQT |
| 25 | Noise reduction | SNR improvement | Spectral subtraction, RNNoise |
| 26 | Resampling | Standardize sample rate | 16kHz target |
| 27 | Pre-emphasis filter | High-frequency boost | α = 0.97 |
| 28 | Windowing | Frame extraction | Hamming, Hann |

---

### Layer 2: Population Template Generation (44 algorithms)

#### 2.1 Gait Templates
| # | Algorithm | Purpose | Source |
|---|-----------|---------|--------|
| 29 | Gait cycle phase template | Expected joint angles vs phase | Literature + learned |
| 30 | Cadence prior | Steps/min distribution | Age-stratified |
| 31 | Stride length prior | Step size distribution | Height-normalized |
| 32 | Walking speed prior | Velocity distribution | Age/gender stratified |
| 33 | Step width prior | Base of support | Population norms |
| 34 | Swing/stance ratio prior | Temporal symmetry | Healthy reference |
| 35 | Joint ROM prior | Hip/knee/ankle range | Biomechanics literature |
| 36 | Arm swing amplitude prior | Contralateral swing | PD-sensitive |
| 37 | Trunk sway prior | Postural stability | Balance norms |
| 38 | Gait variability prior | CV of stride time | Fall risk indicator |
| 39 | Phase coordination prior | Inter-limb timing | Healthy coupling |

#### 2.2 Hand Movement Templates
| # | Algorithm | Purpose | Source |
|---|-----------|---------|--------|
| 40 | Finger tapping frequency prior | Taps/second distribution | UPDRS norms |
| 41 | Finger tapping amplitude prior | Aperture distribution | Task-specific |
| 42 | Amplitude decrement prior | Fatigue pattern | Bradykinesia marker |
| 43 | Frequency decrement prior | Slowing pattern | Sequence effect |
| 44 | Hand aperture prior | Open-close range | Healthy reference |
| 45 | Movement velocity prior | Speed distribution | Age-adjusted |
| 46 | Tremor frequency prior | Oscillation band | By tremor type |
| 47 | Tremor amplitude prior | Displacement range | Severity grading |
| 48 | Movement smoothness prior | Spectral arc length | Coordination metric |
| 49 | Pronation-supination prior | Rotation range | UPDRS item |

#### 2.3 Eye Movement Templates
| # | Algorithm | Purpose | Source |
|---|-----------|---------|--------|
| 50 | Saccade main sequence | Velocity vs amplitude | Bahill et al. |
| 51 | Saccade duration model | Duration vs amplitude | Power law |
| 52 | Prosaccade latency prior | Reaction time | Age-stratified |
| 53 | Antisaccade latency prior | Inhibition time | Cognitive marker |
| 54 | Antisaccade error rate prior | Inhibition failures | Executive function |
| 55 | Fixation duration prior | Dwell time distribution | Gamma model |
| 56 | Fixation stability prior | BCEA distribution | Visual attention |
| 57 | Microsaccade rate prior | Intrusions/second | Fixation quality |
| 58 | Smooth pursuit gain prior | Eye/target velocity ratio | Tracking quality |
| 59 | Pupil light reflex prior | PLR dynamics | Autonomic function |
| 60 | Pupil baseline prior | Resting diameter | Lighting-adjusted |
| 61 | Cognitive pupil response prior | Task-evoked dilation | Workload indicator |

#### 2.4 Voice/Speech Templates
| # | Algorithm | Purpose | Source |
|---|-----------|---------|--------|
| 62 | F0 prior (male) | Pitch distribution | Gender-specific |
| 63 | F0 prior (female) | Pitch distribution | Gender-specific |
| 64 | Jitter prior | Pitch perturbation | Healthy range |
| 65 | Shimmer prior | Amplitude perturbation | Healthy range |
| 66 | HNR prior | Harmonic-to-noise | Voice quality |
| 67 | Formant prior (vowel /a/) | F1, F2 targets | Vowel space |
| 68 | Formant prior (vowel /i/) | F1, F2 targets | Vowel space |
| 69 | Formant prior (vowel /u/) | F1, F2 targets | Vowel space |
| 70 | Vowel space area prior | Triangle area | Articulation |
| 71 | Speech rate prior | Syllables/min | Fluency |
| 72 | Pause duration prior | Silence length | Hesitation |

---

### Layer 3: Event-Based Encoding - DPB Core (52 algorithms)

#### 3.1 Pose/Gait Event Encoders
| # | Algorithm | Purpose | Event Type |
|---|-----------|---------|------------|
| 73 | Heel strike detector | Gait event | Discrete spike |
| 74 | Toe off detector | Gait event | Discrete spike |
| 75 | Gait phase classifier | Cycle segmentation | Phase label |
| 76 | Keypoint deviation encoder | Position vs template | Threshold crossing |
| 77 | Joint angle deviation encoder | Kinematics vs template | Threshold crossing |
| 78 | Joint velocity encoder | Angular velocity | Rate threshold |
| 79 | Symmetry deviation encoder | L/R comparison | Asymmetry event |
| 80 | Stride time deviation encoder | Timing vs expected | Variability event |
| 81 | Arm swing deviation encoder | Amplitude vs template | PD marker |
| 82 | Postural sway encoder | COM deviation | Balance event |
| 83 | Freezing detector | Gait arrest | Binary event |
| 84 | Festination detector | Accelerating steps | Pattern event |

#### 3.2 Hand Movement Event Encoders
| # | Algorithm | Purpose | Event Type |
|---|-----------|---------|------------|
| 85 | Tap onset detector | Finger contact | Discrete spike |
| 86 | Tap offset detector | Finger separation | Discrete spike |
| 87 | Tapping amplitude encoder | Aperture vs template | Deviation event |
| 88 | Tapping timing encoder | ITI vs expected | Rhythm deviation |
| 89 | Amplitude decrement encoder | Fatigue detection | Trend event |
| 90 | Frequency decrement encoder | Slowing detection | Trend event |
| 91 | Hesitation detector | Movement arrest | Pause event |
| 92 | Tremor oscillation encoder | Peak detection | Periodic event |
| 93 | Tremor frequency encoder | Dominant frequency | Spectral event |
| 94 | Tremor amplitude encoder | Envelope threshold | Magnitude event |
| 95 | Hand aperture encoder | Open/close threshold | Level crossing |
| 96 | Movement velocity encoder | Speed threshold | Rate event |

#### 3.3 Eye Movement Event Encoders
| # | Algorithm | Purpose | Event Type |
|---|-----------|---------|------------|
| 97 | Saccade onset detector | Movement initiation | Discrete spike |
| 98 | Saccade offset detector | Movement termination | Discrete spike |
| 99 | Saccade amplitude encoder | Size vs template | Deviation event |
| 100 | Saccade velocity encoder | Main sequence deviation | Deviation event |
| 101 | Saccade duration encoder | Duration vs expected | Deviation event |
| 102 | Saccade direction encoder | Quadrant/angle | Directional event |
| 103 | Saccade latency encoder | Reaction time | Timing event |
| 104 | Fixation onset detector | Stable gaze start | Discrete spike |
| 105 | Fixation duration encoder | Dwell vs expected | Deviation event |
| 106 | Microsaccade detector | Small saccade | Discrete spike |
| 107 | Smooth pursuit error encoder | Tracking deviation | Continuous event |
| 108 | Pupil dilation encoder | Rate threshold | Autonomic event |
| 109 | Pupil constriction encoder | Rate threshold | Autonomic event |
| 110 | PLR deviation encoder | Reflex vs template | Deviation event |
| 111 | Blink encoder | Eye closure | Discrete spike |

#### 3.4 Voice/Audio Event Encoders
| # | Algorithm | Purpose | Event Type |
|---|-----------|---------|------------|
| 112 | Pitch deviation encoder | F0 vs template | Deviation event |
| 113 | Pitch rate encoder | F0 dynamics | Rate threshold |
| 114 | Jitter deviation encoder | Perturbation excess | Quality event |
| 115 | Shimmer deviation encoder | Perturbation excess | Quality event |
| 116 | HNR deviation encoder | Noise excess | Quality event |
| 117 | Formant deviation encoder | Articulation error | Vowel event |
| 118 | Vowel centralization encoder | Reduced space | Undershoot event |
| 119 | Formant transition encoder | Coarticulation | Trajectory event |
| 120 | Speech onset detector | Utterance start | Discrete spike |
| 121 | Pause onset detector | Silence start | Discrete spike |
| 122 | Pause duration encoder | Length vs expected | Deviation event |
| 123 | Speech rate encoder | Local rate deviation | Rhythm event |
| 124 | Intensity deviation encoder | Loudness change | Amplitude event |

---

### Layer 4: Conventional Encoding Baselines (20 algorithms)

| # | Algorithm | Purpose | Comparison |
|---|-----------|---------|------------|
| 125 | Raw keypoint streaming | Uncompressed pose | Baseline bandwidth |
| 126 | Fixed-rate pose sampling | Uniform temporal | vs event-driven |
| 127 | Delta encoding (pose) | Frame differences | Simple compression |
| 128 | Raw audio streaming | Uncompressed audio | Baseline bandwidth |
| 129 | Fixed-frame MFCC | Uniform feature extraction | vs event-driven |
| 130 | Mel spectrogram (dense) | Full time-frequency | vs sparse |
| 131 | Raw gaze coordinates | Uncompressed eye data | Baseline |
| 132 | Fixed-rate pupil sampling | Uniform sampling | vs event-driven |
| 133 | Uniform quantization (all) | Fixed bit-depth | vs adaptive |
| 134 | Moving average filter | Smoothing baseline | vs template |
| 135 | PCA compression | Linear dimensionality | vs sparse |
| 136 | Autoencoder compression | Learned compression | vs DPB |
| 137 | Wavelet compression | Multi-scale | vs event |
| 138 | JPEG-style blocking | Spatial compression | Video baseline |
| 139 | H.264/H.265 encoding | Video codec | Bandwidth baseline |
| 140 | Opus audio encoding | Audio codec | Bandwidth baseline |
| 141 | Linear prediction (audio) | LP coefficients | vs template |
| 142 | Kalman filter baseline | State estimation | vs DPB |
| 143 | Particle filter baseline | Nonlinear estimation | vs DPB |
| 144 | HMM baseline | Sequential model | vs SNN |

---

### Layer 5: Neuron Models (19 algorithms - shared with contact)

| # | Model | Complexity | Use Case |
|---|-------|------------|----------|
| 145 | Integrate-and-Fire (IF) | Minimal | Fast baseline |
| 146 | Leaky IF (LIF) | Low | Standard SNN |
| 147 | Current-based LIF (CLIF) | Low | Input modeling |
| 148 | Adaptive LIF (ALIF) | Medium | Temporal adaptation |
| 149 | Exponential LIF (ELIF) | Medium | Sharp threshold |
| 150 | Quadratic LIF (QLIF) | Medium | Resonance |
| 151 | Generalized LIF (GLIF) | High | Flexible dynamics |
| 152 | Izhikevich | Medium | Rich dynamics |
| 153 | Adaptive Exponential (AdEx) | Medium | Bursting |
| 154 | Hodgkin-Huxley | High | Biophysical |
| 155 | FitzHugh-Nagumo | Medium | Excitability |
| 156 | Morris-Lecar | Medium | Calcium dynamics |
| 157 | Spike Response Model (SRM) | Medium | Analytical |
| 158 | Calcium-based LIF | Medium | Learning |
| 159 | Recurrent LIF | Medium | Temporal memory |
| 160 | Stochastic LIF | Medium | Noise modeling |
| 161 | Xylo LIF | Low | Hardware-specific |
| 162 | Pulsar LIF | Low | Hardware-specific |
| 163 | Quantized LIF | Low | Edge deployment |

---

### Layer 6: SNN Architectures (32 algorithms)

#### 6.1 Pose/Gait SNNs
| # | Architecture | Purpose |
|---|--------------|---------|
| 164 | Spiking GCN (skeleton graph) | Spatial relationships |
| 165 | Temporal convolutional SNN | Sequence modeling |
| 166 | ST-GCN spiking variant | Spatio-temporal graphs |
| 167 | Spiking transformer (pose) | Attention over joints |
| 168 | Hierarchical pose SNN | Limb → body hierarchy |
| 169 | Gait phase recurrent SNN | Cycle memory |
| 170 | Bilateral comparison SNN | Symmetry assessment |

#### 6.2 Hand Movement SNNs
| # | Architecture | Purpose |
|---|--------------|---------|
| 171 | Finger tracking SNN | Fine motor encoding |
| 172 | Tremor spectral SNN | Frequency analysis |
| 173 | Tapping rhythm SNN | Temporal pattern |
| 174 | Sequence effect SNN | Fatigue/decrement |
| 175 | Hand aperture SNN | Grip analysis |

#### 6.3 Eye Movement SNNs
| # | Architecture | Purpose |
|---|--------------|---------|
| 176 | Saccade classifier SNN | Movement type |
| 177 | Main sequence SNN | Velocity-amplitude |
| 178 | Fixation analyzer SNN | Stability metrics |
| 179 | Pupil dynamics SNN | Autonomic encoding |
| 180 | Scanpath SNN | Visual attention |
| 181 | Anti-saccade SNN | Inhibition analysis |

#### 6.4 Voice/Audio SNNs
| # | Architecture | Purpose |
|---|--------------|---------|
| 182 | Pitch tracking SNN | F0 trajectory |
| 183 | Voice quality SNN | Perturbation analysis |
| 184 | Formant tracking SNN | Articulation |
| 185 | Prosody SNN | Rhythm/timing |
| 186 | Phoneme classification SNN | Speech units |
| 187 | Speaker embedding SNN | Individual features |

#### 6.5 Multi-Modal Fusion SNNs
| # | Architecture | Purpose |
|---|--------------|---------|
| 188 | Cross-modal attention SNN | Feature alignment |
| 189 | Temporal alignment SNN | Synchronization |
| 190 | Hierarchical fusion SNN | Multi-level integration |
| 191 | Late fusion SNN | Decision combination |
| 192 | Early fusion SNN | Feature combination |
| 193 | Confidence-weighted SNN | Reliability fusion |
| 194 | Missing modality SNN | Graceful degradation |
| 195 | Multi-task SNN | Shared representations |

---

### Layer 7: ANN Baseline Architectures (24 algorithms)

| # | Architecture | Modality | Purpose |
|---|--------------|----------|---------|
| 196 | CNN (pose) | Pose | Spatial features |
| 197 | LSTM (pose sequence) | Pose | Temporal modeling |
| 198 | Transformer (pose) | Pose | Attention-based |
| 199 | ST-GCN (original) | Pose | Skeleton graph |
| 200 | MS-G3D | Pose | Multi-scale graph |
| 201 | CNN (hand) | Hand | Spatial features |
| 202 | RNN (hand trajectory) | Hand | Movement tracking |
| 203 | 3D-CNN (hand video) | Hand | Spatio-temporal |
| 204 | CNN (eye images) | Eye | Gaze estimation |
| 205 | LSTM (gaze sequence) | Eye | Scanpath modeling |
| 206 | Transformer (fixations) | Eye | Attention patterns |
| 207 | wav2vec 2.0 | Voice | Self-supervised audio |
| 208 | HuBERT | Voice | Hidden unit BERT |
| 209 | WavLM | Voice | Multi-task audio |
| 210 | Conformer | Voice | Conv + attention |
| 211 | TDNN | Voice | Time-delay network |
| 212 | ResNet (spectrogram) | Voice | Image-like audio |
| 213 | BiLSTM (prosody) | Voice | Bidirectional sequence |
| 214 | Multi-modal transformer | All | Cross-attention |
| 215 | Multi-modal CNN-LSTM | All | Hybrid architecture |
| 216 | Ensemble (vote) | All | Decision fusion |
| 217 | Ensemble (stack) | All | Stacked generalization |
| 218 | Knowledge distillation | All | ANN→SNN transfer |
| 219 | Hybrid ANN-SNN | All | Mixed precision |

---

### Layer 8: Training Algorithms (25 algorithms - shared)

| # | Algorithm | Type | Application |
|---|-----------|------|-------------|
| 220 | Backprop through time (BPTT) | Surrogate gradient | SNN training |
| 221 | Online training through time (OTTT) | Online surrogate | Streaming |
| 222 | Spatial-layer-wise training (SLTT) | Layer-wise | Memory efficient |
| 223 | Fast sigmoid surrogate | Surrogate function | Standard |
| 224 | Arctan surrogate | Surrogate function | Smooth gradient |
| 225 | Triangular surrogate | Surrogate function | Sharp gradient |
| 226 | Gaussian surrogate | Surrogate function | Probabilistic |
| 227 | Weight normalization | ANN→SNN | Conversion |
| 228 | Threshold balancing | ANN→SNN | Conversion |
| 229 | Rate matching (RMP) | ANN→SNN | Conversion |
| 230 | Quantization-aware training | ANN→SNN | Low-bit |
| 231 | SlipReLU 1-timestep | ANN→SNN | Fast inference |
| 232 | STDP (unsupervised) | Local learning | Feature extraction |
| 233 | R-STDP (reward) | Reinforcement | Task learning |
| 234 | E-prop | Eligibility trace | Online learning |
| 235 | DECOLLE | Deep continuous | Layer-wise |
| 236 | SuperSpike | Exact gradient | Research |
| 237 | Adam (SNN variant) | Optimizer | Stable training |
| 238 | SGD with momentum | Optimizer | Standard |
| 239 | Learning rate scheduling | Optimizer | Convergence |
| 240 | Gradient clipping | Stability | Spike gradients |
| 241 | Dropout (SNN variant) | Regularization | Prevent overfit |
| 242 | Synaptic pruning | Compression | Sparsification |
| 243 | Knowledge distillation | Transfer | ANN→SNN |
| 244 | Contrastive learning (SNN) | Self-supervised | Representation |

---

### Layer 9: Output Decoding (38 algorithms)

#### 9.1 Pose/Gait Outputs
| # | Decoder | Output Type |
|---|---------|-------------|
| 245 | Cadence estimator | Continuous (steps/min) |
| 246 | Stride length estimator | Continuous (meters) |
| 247 | Walking speed estimator | Continuous (m/s) |
| 248 | Gait phase classifier | Categorical (8 phases) |
| 249 | Symmetry index decoder | Continuous (ratio) |
| 250 | Variability decoder (CV) | Continuous (%) |
| 251 | UPDRS gait score predictor | Ordinal (0-4) |
| 252 | Fall risk classifier | Binary/ordinal |
| 253 | Freezing of gait detector | Binary |
| 254 | Ataxia severity decoder | Ordinal |

#### 9.2 Hand Movement Outputs
| # | Decoder | Output Type |
|---|---------|-------------|
| 255 | Tremor amplitude decoder | Continuous (cm) |
| 256 | Tremor frequency decoder | Continuous (Hz) |
| 257 | Tremor type classifier | Categorical (rest/postural/kinetic) |
| 258 | Bradykinesia score decoder | Ordinal (0-4) |
| 259 | Finger tapping score decoder | UPDRS item 3.4 |
| 260 | Hand movements score decoder | UPDRS item 3.5 |
| 261 | Pronation-supination decoder | UPDRS item 3.6 |
| 262 | Fine motor composite | Continuous |

#### 9.3 Eye Movement Outputs
| # | Decoder | Output Type |
|---|---------|-------------|
| 263 | Saccade latency decoder | Continuous (ms) |
| 264 | Saccade accuracy decoder | Continuous (gain) |
| 265 | Fixation stability decoder | Continuous (BCEA) |
| 266 | Cognitive load decoder | Continuous |
| 267 | Fatigue/vigilance decoder | Continuous |
| 268 | MCI risk classifier | Binary/probabilistic |
| 269 | Attention score decoder | Continuous |
| 270 | Executive function proxy | Continuous |

#### 9.4 Voice Outputs
| # | Decoder | Output Type |
|---|---------|-------------|
| 271 | Voice quality score | Continuous |
| 272 | Articulation precision | Continuous |
| 273 | Prosody naturalness | Continuous |
| 274 | Dysarthria severity | Ordinal |
| 275 | Speech intelligibility | Continuous (%) |
| 276 | PD voice probability | Probabilistic |
| 277 | Depression voice marker | Continuous |

#### 9.5 Integrated Outputs
| # | Decoder | Output Type |
|---|---------|-------------|
| 278 | Total motor score (MDS-UPDRS III) | Continuous |
| 279 | Disease stage classifier | Ordinal (H&Y) |
| 280 | Medication state classifier | Binary (ON/OFF) |
| 281 | Diagnostic classifier | Categorical (PD/ET/MSA/HC) |
| 282 | Progression tracker | Longitudinal |

---

### Layer 10: Power & Energy Estimation (19 algorithms - shared)

| # | Algorithm | Purpose |
|---|-----------|---------|
| 283 | MAC counter (ANN) | Multiply-accumulate ops |
| 284 | AC counter (SNN) | Accumulate-only ops |
| 285 | Spike counter | Total spikes |
| 286 | Synaptic operation counter | SynOps |
| 287 | Memory access counter | DRAM/SRAM |
| 288 | 45nm energy model | Legacy comparison |
| 289 | 28nm energy model | Xylo target |
| 290 | 7nm energy model | Modern GPU |
| 291 | Xylo power model | Hardware-specific |
| 292 | Pulsar power model | Hardware-specific |
| 293 | FPGA power model | Emulation |
| 294 | GPU power profiler | ANN baseline |
| 295 | CPU power profiler | Reference |
| 296 | Camera power model | Sensor input |
| 297 | Microphone power model | Audio input |
| 298 | Memory bandwidth model | Data movement |
| 299 | Latency profiler | End-to-end timing |
| 300 | Energy-delay product | Efficiency metric |
| 301 | Operations per Joule | Efficiency metric |

---

### Layer 11: Convergence Analysis (18 algorithms)

| # | Algorithm | Purpose |
|---|-----------|---------|
| 302 | Time-to-accuracy (pose) | Convergence speed |
| 303 | Time-to-accuracy (hand) | Convergence speed |
| 304 | Time-to-accuracy (eye) | Convergence speed |
| 305 | Time-to-accuracy (voice) | Convergence speed |
| 306 | Time-to-accuracy (fused) | Multi-modal convergence |
| 307 | Timesteps-to-accuracy | SNN-specific |
| 308 | Events-to-accuracy | DPB-specific |
| 309 | Bytes-to-accuracy | Bandwidth efficiency |
| 310 | Energy-to-accuracy | Power efficiency |
| 311 | Convergence rate estimator | Slope analysis |
| 312 | Steady-state error estimator | Final accuracy |
| 313 | Confidence interval tracker | Uncertainty over time |
| 314 | Prior effectiveness metric | Template benefit |
| 315 | Sparsity-accuracy tradeoff | Compression analysis |
| 316 | Latency-accuracy tradeoff | Speed analysis |
| 317 | Cross-modal convergence | Fusion benefit |
| 318 | Online vs batch comparison | Streaming analysis |
| 319 | Cold-start analysis | Initial conditions |

---

### Layer 12: Evaluation Metrics (40 algorithms)

#### 12.1 Classification Metrics
| # | Metric | Application |
|---|--------|-------------|
| 320 | Accuracy | Overall correct |
| 321 | Balanced accuracy | Class-imbalanced |
| 322 | Sensitivity/Recall | True positive rate |
| 323 | Specificity | True negative rate |
| 324 | Precision | Positive predictive |
| 325 | F1 score | Harmonic mean |
| 326 | AUC-ROC | Discrimination |
| 327 | AUC-PR | Precision-recall |
| 328 | Cohen's Kappa | Agreement |
| 329 | Matthews correlation | Balanced binary |

#### 12.2 Regression Metrics
| # | Metric | Application |
|---|--------|-------------|
| 330 | Mean absolute error | Central tendency |
| 331 | Root mean squared error | Sensitivity to outliers |
| 332 | Mean absolute percentage error | Relative error |
| 333 | R² (coefficient of determination) | Variance explained |
| 334 | Pearson correlation | Linear association |
| 335 | Spearman correlation | Rank association |
| 336 | Concordance correlation | Agreement |
| 337 | Intraclass correlation | Reliability |

#### 12.3 Signal Quality Metrics
| # | Metric | Application |
|---|--------|-------------|
| 338 | Signal-to-noise ratio | Quality |
| 339 | Signal-to-distortion ratio | Reconstruction |
| 340 | Percent root-mean-square difference | Waveform fidelity |
| 341 | Structural similarity (SSIM) | Perceptual quality |
| 342 | Compression ratio | Data reduction |
| 343 | Bits per event | Encoding efficiency |

#### 12.4 Clinical Validity Metrics
| # | Metric | Application |
|---|--------|-------------|
| 344 | Inter-rater agreement (vs clinician) | Clinical validity |
| 345 | Test-retest reliability | Stability |
| 346 | Minimal detectable change | Sensitivity |
| 347 | Clinically important difference | Meaningfulness |
| 348 | Diagnostic odds ratio | Clinical utility |
| 349 | Number needed to diagnose | Efficiency |

#### 12.5 Efficiency Metrics
| # | Metric | Application |
|---|--------|-------------|
| 350 | Events per second | Sparsity |
| 351 | Spikes per inference | SNN activity |
| 352 | Operations per inference | Compute cost |
| 353 | Energy per inference | Power cost |
| 354 | Latency (ms) | Speed |
| 355 | Throughput (samples/s) | Capacity |
| 356 | Memory footprint | Resource usage |
| 357 | Model parameters | Complexity |
| 358 | FLOPs | Compute baseline |
| 359 | AC operations | SNN compute |

---

### Layer 13: Experiment Infrastructure (28 algorithms)

| # | Component | Purpose |
|---|-----------|---------|
| 360 | Dataset loader (pose) | Data pipeline |
| 361 | Dataset loader (hand) | Data pipeline |
| 362 | Dataset loader (eye) | Data pipeline |
| 363 | Dataset loader (voice) | Data pipeline |
| 364 | Multi-modal synchronizer | Temporal alignment |
| 365 | Data augmentation (pose) | Robustness |
| 366 | Data augmentation (hand) | Robustness |
| 367 | Data augmentation (eye) | Robustness |
| 368 | Data augmentation (voice) | Robustness |
| 369 | Train/val/test splitter | Stratified sampling |
| 370 | Cross-validation (k-fold) | Robust evaluation |
| 371 | Leave-one-subject-out | Generalization |
| 372 | Hyperparameter search | Grid/random/Bayesian |
| 373 | Early stopping | Overfitting prevention |
| 374 | Checkpointing | Model saving |
| 375 | Logging (TensorBoard/W&B) | Experiment tracking |
| 376 | Seed management | Reproducibility |
| 377 | Ablation study framework | Component analysis |
| 378 | Statistical testing | Significance |
| 379 | Confidence intervals | Uncertainty |
| 380 | Effect size calculation | Magnitude |
| 381 | Power analysis | Sample size |
| 382 | Baseline comparison | Fair evaluation |
| 383 | Multi-seed averaging | Stability |
| 384 | Hardware profiling | Resource monitoring |
| 385 | Neuromorphic emulation | Xylo/Pulsar simulation |
| 386 | NIR export | Cross-platform |
| 387 | Model versioning | Reproducibility |

---

### Layer 14: Visualization (16 algorithms)

| # | Visualization | Purpose |
|---|---------------|---------|
| 388 | Raster plot | Spike trains |
| 389 | Event histogram | Activity distribution |
| 390 | Convergence curve | Time-to-accuracy |
| 391 | Energy vs accuracy scatter | Efficiency tradeoff |
| 392 | Sparsity vs accuracy curve | Compression tradeoff |
| 393 | Confusion matrix | Classification errors |
| 394 | ROC curve | Discrimination |
| 395 | Precision-recall curve | Imbalanced data |
| 396 | Bland-Altman plot | Agreement |
| 397 | Template deviation heatmap | Spatial patterns |
| 398 | Event stream visualization | Temporal patterns |
| 399 | Multi-modal timeline | Synchronized view |
| 400 | Attention map | Feature importance |
| 401 | t-SNE/UMAP embedding | Representation space |
| 402 | Parameter sensitivity | Hyperparameter effect |
| 403 | Clinical score scatter | Prediction vs actual |

---

### Layer 15: Hardware Deployment (14 algorithms)

| # | Component | Target |
|---|-----------|--------|
| 404 | Xylo model mapper | SynSense hardware |
| 405 | Xylo quantizer | 8-bit weights |
| 406 | Xylo compiler | Rockpool SDK |
| 407 | Pulsar model mapper | Innatera hardware |
| 408 | Pulsar compiler | Talamo SDK |
| 409 | FPGA synthesis | Xilinx/Intel |
| 410 | HLS4ML export | FPGA workflow |
| 411 | TensorFlow Lite export | Mobile deployment |
| 412 | ONNX export | Interoperability |
| 413 | CoreML export | Apple devices |
| 414 | Edge TPU export | Google Coral |
| 415 | Jetson optimization | NVIDIA edge |
| 416 | WebGPU export | Browser deployment |
| 417 | NIR standardization | Cross-platform SNN |

---

## 3. Algorithm Count Summary

| Layer | Count |
|-------|-------|
| 1. Signal Acquisition & Preprocessing | 28 |
| 2. Population Template Generation | 44 |
| 3. Event-Based Encoding (DPB Core) | 52 |
| 4. Conventional Encoding Baselines | 20 |
| 5. Neuron Models | 19 |
| 6. SNN Architectures | 32 |
| 7. ANN Baseline Architectures | 24 |
| 8. Training Algorithms | 25 |
| 9. Output Decoding | 38 |
| 10. Power & Energy Estimation | 19 |
| 11. Convergence Analysis | 18 |
| 12. Evaluation Metrics | 40 |
| 13. Experiment Infrastructure | 28 |
| 14. Visualization | 16 |
| 15. Hardware Deployment | 14 |
| **TOTAL** | **417** |

---

## 4. Combined Inventory (Contact + Non-Contact)

| Category | Contact Only | Non-Contact Only | Shared | Combined Total |
|----------|--------------|------------------|--------|----------------|
| Signal Acquisition | 15 | 28 | 0 | 43 |
| Population Templates | 17 | 44 | 0 | 61 |
| DPB Event Encoders | 25 | 52 | 0 | 77 |
| Conventional Baselines | 10 | 20 | 0 | 30 |
| Neuron Models | 0 | 0 | 19 | 19 |
| SNN Architectures | 23 | 32 | 0 | 55 |
| ANN Baselines | 20 | 24 | 0 | 44 |
| Training | 0 | 0 | 25 | 25 |
| Output Decoders | 10 | 38 | 0 | 48 |
| Power/Energy | 0 | 0 | 19 | 19 |
| Convergence Analysis | 10 | 18 | 0 | 28 |
| Evaluation Metrics | 0 | 0 | 40 | 40 |
| Infrastructure | 0 | 0 | 28 | 28 |
| Visualization | 0 | 0 | 16 | 16 |
| Hardware Deployment | 0 | 0 | 14 | 14 |
| **TOTAL** | **130** | **256** | **161** | **547** |

---

## 5. Key Experimental Comparisons

### 5.1 Primary Hypothesis Tests

| Comparison | Metric | Expected Outcome |
|------------|--------|------------------|
| DPB vs Raw streaming | Time-to-accuracy | DPB 5-10× faster |
| DPB vs Fixed-rate sampling | Events-to-accuracy | DPB 10-100× fewer events |
| SNN vs ANN | Energy per inference | SNN 10-100× lower |
| Template-based vs Template-free | Convergence rate | Template 2-5× faster |
| Multi-modal vs Unimodal | Clinical accuracy | Fusion +5-15% accuracy |

### 5.2 Ablation Studies Required

| Ablation | Variables | Purpose |
|----------|-----------|---------|
| Template source | Learned vs literature vs hybrid | Prior quality |
| Threshold adaptation | Fixed vs adaptive | Robustness |
| Neuron model | LIF vs ALIF vs Izhikevich | Complexity-performance |
| Time resolution | 1ms vs 10ms vs 33ms (30fps) | Temporal precision |
| Sparsity level | 1% vs 5% vs 10% vs 20% | Compression-accuracy |
| Fusion timing | Early vs late vs hierarchical | Integration strategy |

### 5.3 Modality-Specific Questions

| Modality | Key Question | Experiment |
|----------|--------------|------------|
| Pose | Does skeleton topology help? | GCN vs MLP |
| Pose | 2D sufficient or need 3D? | Depth ablation |
| Hand | Minimum spatial resolution? | Downsampling study |
| Hand | RGB vs RGB-D? | Depth value |
| Eye | Webcam vs dedicated tracker? | Hardware comparison |
| Eye | Sampling rate requirement? | 30/60/120/240 Hz |
| Voice | Sustained vs connected speech? | Task comparison |
| Voice | Language independence? | Cross-lingual test |

---

## 6. Dataset Requirements

### 6.1 Minimum Dataset Requirements per Modality

| Modality | Participants | Conditions | Recordings | Duration |
|----------|--------------|------------|------------|----------|
| Pose/Gait | 100+ | PD, ET, HC | 5+ per person | 2+ min walk |
| Hand | 100+ | PD, ET, HC | 10+ per task | 10s per task |
| Eye | 100+ | MCI, AD, HC | 3+ paradigms | 5+ min total |
| Voice | 100+ | PD, HC | 3+ tasks | 30s per task |

### 6.2 Available Datasets to Use

| Dataset | Modality | Size | Access |
|---------|----------|------|--------|
| PhysioNet Gait-PD | Gait | 166 | Open |
| mPower | Hand, Voice | 5800+ | Open (controlled) |
| Human3.6M | Pose (reference) | 3.6M frames | Open |
| MDVR-KCL | Voice | 37 | Open |
| ETH-XGaze | Eye (reference) | 1.1M | Open |

---

## 7. Success Criteria

### 7.1 DPB Framework Validation

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Convergence speedup | ≥5× | Time-to-95%-accuracy |
| Data reduction | ≥10× | Events vs samples |
| Energy reduction | ≥10× | Energy per inference |
| Accuracy preservation | ≤2% loss | vs dense baseline |
| Clinical validity | ICC ≥0.8 | vs clinician rating |

### 7.2 Hardware Deployment Targets

| Target | Xylo | Pulsar |
|--------|------|--------|
| Latency | <200ms | <50ms |
| Power | <1mW | <1mW |
| Accuracy | ≥90% baseline | ≥90% baseline |

---

*Document: DPB Non-Contact Algorithm Inventory v1.0*
*AuraSense Tech Corporation - December 2025*
