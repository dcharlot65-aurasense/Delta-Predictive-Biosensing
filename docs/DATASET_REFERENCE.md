# Unified Dataset Reference

> **Delta-Predictive Biosensing (DPB) Framework**
>
> Comprehensive reference for public datasets, download instructions, normalization procedures, and integration guidelines for biosignal research and NeuroPlay development.

---

## Table of Contents

1. [Overview](#1-overview)
2. [ECG & Cardiac Datasets](#2-ecg--cardiac-datasets)
3. [EEG & Neural Datasets](#3-eeg--neural-datasets)
4. [Sleep Datasets](#4-sleep-datasets)
5. [Stress & Emotion Datasets](#5-stress--emotion-datasets)
6. [Parkinson's Disease & Movement Disorders](#6-parkinsons-disease--movement-disorders)
7. [Stroke Rehabilitation](#7-stroke-rehabilitation)
8. [Healthy Motion Baselines & Pretraining](#8-healthy-motion-baselines--pretraining)
9. [Speech Dynamics & Cognitive Health](#9-speech-dynamics--cognitive-health)
10. [Eye Tracking & Visual Attention](#10-eye-tracking--visual-attention)
11. [Download Instructions](#11-download-instructions)
12. [Normalization Procedures](#12-normalization-procedures)
13. [DPB Integration Guidelines](#13-dpb-integration-guidelines)
14. [NeuroPlay Integration Strategy](#14-neuroplay-integration-strategy)
15. [Key Repositories & Toolkits](#15-key-repositories--toolkits)

---

## 1. Overview

This document consolidates all dataset references for the DPB framework, covering:

- **Traditional Biosignals**: ECG, EEG, PPG, EDA for clinical detection
- **Motion & Gait**: IMU, MoCap, video for movement disorder assessment
- **Speech & Cognitive**: Audio, transcripts for dementia/depression screening
- **Eye Tracking**: Gaze patterns for ASD and attention analysis

### Access Timeline Summary

| Category | Immediate (Open) | 1-2 Weeks (Registration) | 2-4 Weeks (DUA) |
|----------|------------------|--------------------------|-----------------|
| **Cardiac** | MIT-BIH, PTB-XL | - | - |
| **Neural** | CHB-MIT, BONN | TUH, BCI Competition | - |
| **Sleep** | Sleep-EDF | CAP-Sleep | SHHS, NSRR |
| **Motion** | PhysioNet PD Gait, AddBiomechanics | Motion-X, NTU RGB+D | WearGait-PD, CARE-PD |
| **Speech** | - | DementiaBank, SimTK | Bridge2AI-Voice |
| **Stress** | WESAD | SWELL | - |

---

## 2. ECG & Cardiac Datasets

### 2.1 Primary Databases

| Dataset | Description | Samples | Format | Access |
|---------|-------------|---------|--------|--------|
| **MIT-BIH Arrhythmia** | Gold standard arrhythmia benchmark | 48 recordings, 30 min each | WFDB | [PhysioNet](https://physionet.org/content/mitdb/) |
| **PTB-XL** | Large 12-lead ECG dataset | 21,837 ECGs | WFDB | [PhysioNet](https://physionet.org/content/ptb-xl/) |
| **CPSC 2018** | China Physiological Signal Challenge | 6,877 recordings | MAT | [PhysioNet](https://physionet.org/content/cpsc2018/) |
| **PTB Diagnostic** | Diagnostic ECG database | 549 records, 15 leads | WFDB | [PhysioNet](https://physionet.org/content/ptbdb/) |

### 2.2 Download Instructions

```bash
# Install PhysioNet tools
pip install wfdb

# Download MIT-BIH Arrhythmia Database
wfdb.dl_database('mitdb', './data/mitdb')

# Download PTB-XL
wfdb.dl_database('ptb-xl', './data/ptbxl')

# Alternative: wget for full dataset
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
```

### 2.3 DPB Loading Example

```rust
use dpb_core::io::wfdb::WfdbReader;

let reader = WfdbReader::open("./data/mitdb/100")?;
let signal = reader.read_signal(0)?;  // Read first channel
let annotations = reader.read_annotations()?;
```

---

## 3. EEG & Neural Datasets

### 3.1 Seizure Detection

| Dataset | Description | Samples | Format | Access |
|---------|-------------|---------|--------|--------|
| **CHB-MIT** | Pediatric seizure recordings | 23 subjects, 844 hours | EDF | [PhysioNet](https://physionet.org/content/chbmit/) |
| **TUH Seizure (TUSZ)** | Temple University seizure corpus | 5,612 sessions | EDF | [TUH](https://isip.piconepress.com/projects/tuh_eeg/) |
| **BONN** | Epilepsy benchmark (5 classes) | 500 segments | TXT | University of Bonn |

### 3.2 Motor Imagery / BCI

| Dataset | Description | Samples | Format | Access |
|---------|-------------|---------|--------|--------|
| **BCI Competition IV-2a** | 4-class motor imagery | 9 subjects | GDF | [BNCI Horizon](http://bnci-horizon-2020.eu/) |
| **EEG Motor Imagery (Stroke)** | 64-channel stroke rehab BCI | 50 acute stroke patients | BIDS | [figshare](https://doi.org/10.6084/m9.figshare.21679035.v5) |
| **PhysioNet Motor Imagery** | Left/right hand, feet, tongue | 109 subjects | EDF | [PhysioNet](https://physionet.org/content/eegmmidb/) |

### 3.3 Other EEG

| Dataset | Description | Use Case | Access |
|---------|-------------|----------|--------|
| **DEAP** | 32 subjects, music videos | Emotion recognition | Queen Mary |
| **SEED** | 15 subjects, film clips | Emotion recognition | BCMI Lab |

### 3.4 Download Instructions

```bash
# CHB-MIT
wfdb.dl_database('chbmit', './data/chbmit')

# PhysioNet Motor Imagery
wfdb.dl_database('eegmmidb', './data/eegmmidb')

# TUH requires registration at:
# https://isip.piconepress.com/projects/tuh_eeg/html/request_access.php
```

### 3.5 DPB Loading Example

```rust
use dpb_core::io::edf::EdfReader;

let reader = EdfReader::open("./data/chbmit/chb01/chb01_01.edf")?;
let channels = reader.channel_names();
let eeg_data = reader.read_all_signals()?;
```

---

## 4. Sleep Datasets

### 4.1 Primary Databases

| Dataset | Description | Samples | Modalities | Access |
|---------|-------------|---------|------------|--------|
| **Sleep-EDF (SEDF18)** | Most widely used benchmark | 197 recordings | PSG (EEG, EOG, EMG) | [PhysioNet](https://physionet.org/content/sleep-edfx/) |
| **CAP Sleep** | Cyclic alternating pattern | 108 recordings | PSG | [PhysioNet](https://physionet.org/content/capslpdb/) |
| **SHHS** | Sleep Heart Health Study | 5,804 subjects | PSG | [NSRR](https://sleepdata.org/datasets/shhs) |
| **NSRR** | National Sleep Research Resource | Multiple cohorts | PSG | [NSRR](https://sleepdata.org/) |

### 4.2 Download Instructions

```bash
# Sleep-EDF Expanded
wfdb.dl_database('sleep-edfx', './data/sleep-edfx')

# SHHS/NSRR requires:
# 1. Create account at sleepdata.org
# 2. Submit data access request
# 3. Use NSRR gem or direct download after approval
gem install nsrr
nsrr download shhs/polysomnography
```

---

## 5. Stress & Emotion Datasets

### 5.1 Primary Databases

| Dataset | Description | Signals | Samples | Access |
|---------|-------------|---------|---------|--------|
| **WESAD** | Wearable stress/affect detection | PPG, EDA, ACC, ECG, EMG, RESP, TEMP | 15 subjects | [UCI](https://archive.ics.uci.edu/ml/datasets/WESAD) |
| **SWELL** | Stress and workload | ECG, EDA, facial | 25 subjects | Radboud University |
| **DREAMER** | Emotion from EEG/ECG | EEG (14 ch), ECG | 23 subjects | Queen Mary |

### 5.2 Download Instructions

```bash
# WESAD
wget https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download -O wesad.zip
unzip wesad.zip -d ./data/wesad

# DREAMER requires contacting authors
```

---

## 6. Parkinson's Disease & Movement Disorders

*Datasets with clinical annotations (UPDRS, MDS-UPDRS) for tremor, gait, and motor function analysis.*

### 6.1 Primary Databases

| Dataset | Modalities | Health Correlation | Samples | Access |
|---------|------------|-------------------|---------|--------|
| **WearGait-PD (FDA)** | IMU (13 sensors), insoles, video, walkway | PD gait, MDS-UPDRS, DBS status | PD + age-matched controls | [FDA RST Portal](https://cdrh-rst.fda.gov/weargait-pd-wearables-dataset-gait-parkinsons-disease-and-age-matched-controls) |
| **CARE-PD** | RGB video, optical MoCap | UPDRS-gait severity | Multi-site, 5+ centers | NeurIPS 2025 (hal-05280110) |
| **PhysioNet PD Gait** | Vertical GRF, force sensors | Gait dynamics, stride variability | 93 PD + 73 controls | [PhysioNet](https://physionet.org/content/?topic=parkinsons) |
| **PADS** | Smartwatch accelerometry | Interactive neuro assessments | PD + differential diagnoses | [PhysioNet](https://physionet.org/content/pads/) |
| **Multimodal FoG** | Video + IMU | Freezing of gait events | PD during turning | [Mendeley Data](https://data.mendeley.com/) |
| **PD@Home** | Wrist gyroscope, video | Real-life tremor monitoring | 24 PD + 24 controls | Open-source |
| **Daphnet FoG** | Wearable accelerometers (legs, hip) | FoG during ADL | Lab + daily living | [Mobilize Center](https://mobilize.stanford.edu/data/available-datasets/) |

### 6.2 Download Instructions

```bash
# PhysioNet PD Gait datasets
wfdb.dl_database('gaitpdb', './data/gaitpdb')  # Gait in PD
wfdb.dl_database('gaitndd', './data/gaitndd')  # Gait in Neurodegenerative Disease

# Daphnet FoG
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00245/dataset.zip
unzip dataset.zip -d ./data/daphnet

# WearGait-PD requires FDA RST Portal registration:
# 1. Visit https://cdrh-rst.fda.gov/
# 2. Create account and request access
# 3. Sign Data Use Agreement
```

### 6.3 Controller Input Correlation Targets

For Joy-Con/DualSense IMU + haptic data correlation:

| Target | Dataset | Metric |
|--------|---------|--------|
| **Tremor metrics** | PD@Home | Wrist gyroscope frequency/power |
| **Motor control** | StrokeRehab | Action primitives (reach, transport, reposition) |
| **Reaction time** | StrokeVision-Bench | Box & Block timing |
| **Coordination** | KIMORE | Bilateral coordination scores |

---

## 7. Stroke Rehabilitation

*Datasets for motor recovery tracking, compensation detection, and functional assessment.*

### 7.1 Primary Databases

| Dataset | Modalities | Health Correlation | Samples | Access |
|---------|------------|-------------------|---------|--------|
| **StrokeRehab** | IMU + video features | Sub-second action primitives | 51 stroke + 20 healthy | [SimTK](https://simtk.org/projects/primseq) |
| **StrokeVision-Bench** | RGB video + 2D skeletal keypoints | Box & Block Test performance | 1,000 annotated videos | arXiv 2509.07994 |
| **Toronto Rehab Stroke** | Kinect RGB-D | Upper limb compensation | 10 healthy + 9 stroke | ACM Pervasive Health |
| **KIMORE** | RGB-D (Kinect) | Clinical scoring, motor dysfunction | 44 healthy + 34 motor dysfunction | On request |

### 7.2 Download Instructions

```bash
# StrokeRehab via SimTK
# 1. Create account at simtk.org
# 2. Visit https://simtk.org/projects/primseq
# 3. Download after registration

# KIMORE
# Contact authors via paper for access
```

---

## 8. Healthy Motion Baselines & Pretraining

*Normative datasets for model pretraining and establishing healthy baselines.*

### 8.1 Primary Databases

| Dataset | Modalities | Scale | Use Case | Access |
|---------|------------|-------|----------|--------|
| **Motion-X** | 3D whole-body (SMPL-X), video | 15.6M poses, 81.1K sequences | Expressive motion pretraining | CC BY-NC-SA |
| **Human3.6M** | RGB + 3D MoCap | 3.6M poses, 11 subjects | Pose estimation benchmark | [vision.imar.ro](http://vision.imar.ro/human3.6m) |
| **H3WB (WholeBody)** | 133 keypoints (body+face+hands) | 100K images | Whole-body pose estimation | [GitHub](https://github.com/wholebody3d/wholebody3d) |
| **NTU RGB+D** | RGB + depth + skeleton | 56K videos, 60 actions, 40 subjects | Action recognition | Application required |
| **MPII Human Pose** | RGB video frames | 25K images, 40K people, 410 activities | 2D pose benchmark | [mpi-inf.mpg.de](http://human-pose.mpi-inf.mpg.de/) |
| **KIT Whole-Body** | MoCap + object interaction | 234 subjects, 2925 experiments | Robotics/rehab motion | KIT Database |
| **Sit-to-Walk MoCap** | MoCap, force plates, EMG, IMU | 65 adults, 19-73 years | Healthy aging benchmark | Open access |
| **3D Gait & Running** | MoCap, GRF | 30 healthy young adults | Walking/running kinematics | Open access |
| **AddBiomechanics** | Scaled skeletons, joint dynamics, GRF | Largest validated motion dataset | Physically-validated motion | [CC BY 4.0](https://addbiomechanics.org/) |

### 8.2 Download Instructions

```bash
# Motion-X
git clone https://github.com/IDEA-Research/Motion-X
# Follow instructions in repository

# Human3.6M (requires registration)
# 1. Visit http://vision.imar.ro/human3.6m
# 2. Register and accept license
# 3. Download via provided scripts

# AddBiomechanics
pip install addbiomechanics
# Or visit https://addbiomechanics.org/

# MPII Human Pose
wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
```

---

## 9. Speech Dynamics & Cognitive Health

### 9.1 Alzheimer's & Dementia

| Dataset | Modalities | Health Correlation | Samples | Access |
|---------|------------|-------------------|---------|--------|
| **DementiaBank Pitt** | Audio + transcripts | AD/MCI/HC, Cookie Theft task | Multi-year longitudinal | [TalkBank](https://dementia.talkbank.org/) |
| **ADReSS / ADReSSo** | Enhanced audio + transcripts | AD classification, MMSE prediction | Balanced age/gender | DementiaBank membership |
| **MultiConAD** | Multilingual audio + transcripts | AD/MCI/HC (3-class) | 16 datasets, 4 languages | arXiv 2502.19208 |
| **EWA-DB (Slovak)** | Audio (vowels, DDK, naming) | AD/MCI/PD detection | 1,649 speakers | [Nature Sci Data](https://www.nature.com/articles/s41597-024-04171-6) |
| **Bridge2AI-Voice** | Audio + clinical metadata | Neuro, mood, respiratory, voice | 833 participants | Controlled access |

### 9.2 Depression & Mood Disorders

| Dataset | Modalities | Health Correlation | Samples | Access |
|---------|------------|-------------------|---------|--------|
| **PDCH** | Audio + transcripts | Depression severity (HAMD-17) | 100 consultations, ~30 min each | Research access |
| **Dem@Care** | Audio + video + physiological | Dementia, multi-modal | Lab + home settings | Controlled access |

### 9.3 Download Instructions

```bash
# DementiaBank
# 1. Visit https://dementia.talkbank.org/
# 2. Request TalkBank membership
# 3. Sign Data Use Agreement
# 4. Access via CLAN tools or direct download

# Install CLAN for transcript processing
# macOS: brew install --cask clan
# Linux: Download from https://dali.talkbank.org/clan/

# Bridge2AI-Voice
# 1. Visit https://bridge2ai.org/voice/
# 2. Apply for controlled access
# 3. Complete IRB requirements
```

---

## 10. Eye Tracking & Visual Attention

### 10.1 Primary Databases

| Dataset | Modalities | Health Correlation | Samples | Access |
|---------|------------|-------------------|---------|--------|
| **Saliency4ASD** | Eye tracking + images | ASD vs TD saliency patterns | Children with ASD + controls | IEEE ICME'19 Challenge |
| **ASD Eye Movement** | Fixation maps + scanpaths | ASD visual attention | 14 ASD + 14 TD, 300 images | [ACM MMSys](https://dl.acm.org/doi/10.1145/3304109.3325818) |
| **EyeT4Empathy** | Eye tracking + empathy scores | Attention, ADHD-relevant metrics | Gaze typing + questionnaire | [Nature Sci Data](https://www.nature.com/articles/s41597-022-01862-w) |

### 10.2 Download Instructions

```bash
# EyeT4Empathy
# Download from Nature Scientific Data supplementary materials
# https://www.nature.com/articles/s41597-022-01862-w

# Saliency4ASD
# Contact IEEE ICME'19 challenge organizers
```

---

## 11. Download Instructions

### 11.1 PhysioNet Datasets (Recommended Approach)

```bash
# Install WFDB Python package
pip install wfdb

# Generic download function
import wfdb

def download_physionet(database_name: str, target_dir: str):
    """Download a PhysioNet database."""
    wfdb.dl_database(database_name, target_dir)
    print(f"Downloaded {database_name} to {target_dir}")

# Examples
download_physionet('mitdb', './data/mitdb')
download_physionet('ptb-xl', './data/ptbxl')
download_physionet('sleep-edfx', './data/sleep-edfx')
download_physionet('gaitpdb', './data/gaitpdb')
```

### 11.2 Bulk Download Script

```bash
#!/bin/bash
# download_datasets.sh - Download all open-access datasets

DATA_DIR="./data"
mkdir -p $DATA_DIR

# PhysioNet datasets
declare -A PHYSIONET_DBS=(
    ["mitdb"]="MIT-BIH Arrhythmia"
    ["ptb-xl"]="PTB-XL ECG"
    ["sleep-edfx"]="Sleep-EDF Expanded"
    ["chbmit"]="CHB-MIT Seizure"
    ["gaitpdb"]="Gait in PD"
    ["eegmmidb"]="EEG Motor Imagery"
)

for db in "${!PHYSIONET_DBS[@]}"; do
    echo "Downloading ${PHYSIONET_DBS[$db]}..."
    python -c "import wfdb; wfdb.dl_database('$db', '$DATA_DIR/$db')"
done

echo "Download complete!"
```

### 11.3 Registration-Required Datasets

| Dataset | Registration URL | Typical Wait |
|---------|------------------|--------------|
| TUH EEG Corpus | https://isip.piconepress.com/projects/tuh_eeg/ | 1-3 days |
| NTU RGB+D | https://rose1.ntu.edu.sg/dataset/actionRecognition/ | 1-2 weeks |
| DementiaBank | https://dementia.talkbank.org/ | 1-2 weeks |
| Human3.6M | http://vision.imar.ro/human3.6m | 1 week |
| WearGait-PD | https://cdrh-rst.fda.gov/ | 2-4 weeks |
| Bridge2AI-Voice | https://bridge2ai.org/voice/ | 2-4 weeks |
| SHHS/NSRR | https://sleepdata.org/ | 1-2 weeks |

---

## 12. Normalization Procedures

### 12.1 Signal Amplitude Normalization

```rust
use dpb_core::signal::Signal;

/// Z-score normalization (zero mean, unit variance)
pub fn zscore_normalize(signal: &mut Signal) {
    let mean = signal.mean();
    let std = signal.std();
    for sample in signal.samples_mut() {
        *sample = (*sample - mean) / std;
    }
}

/// Min-max normalization to [0, 1]
pub fn minmax_normalize(signal: &mut Signal) {
    let min = signal.min();
    let max = signal.max();
    let range = max - min;
    for sample in signal.samples_mut() {
        *sample = (*sample - min) / range;
    }
}

/// Robust normalization using median/IQR (outlier-resistant)
pub fn robust_normalize(signal: &mut Signal) {
    let median = signal.median();
    let iqr = signal.percentile(75.0) - signal.percentile(25.0);
    for sample in signal.samples_mut() {
        *sample = (*sample - median) / iqr;
    }
}
```

### 12.2 Resampling Standards

| Signal Type | Target Sample Rate | Rationale |
|-------------|-------------------|-----------|
| ECG | 250 Hz or 500 Hz | Sufficient for QRS detection |
| EEG | 256 Hz | Standard for clinical analysis |
| PPG | 64 Hz or 128 Hz | Adequate for pulse detection |
| EDA | 4 Hz | Slow physiological changes |
| IMU/Accelerometer | 50-100 Hz | Motion capture fidelity |
| Audio | 16000 Hz | Speech processing standard |

```rust
use dpb_core::signal::resample;

// Resample ECG to 250 Hz
let ecg_250hz = resample(&ecg_signal, 250.0)?;

// Resample EEG to 256 Hz
let eeg_256hz = resample(&eeg_signal, 256.0)?;
```

### 12.3 Filtering Recommendations

| Signal | Highpass | Lowpass | Notch | Purpose |
|--------|----------|---------|-------|---------|
| ECG | 0.5 Hz | 40 Hz | 50/60 Hz | Remove baseline wander, muscle noise |
| EEG | 0.1 Hz | 45 Hz | 50/60 Hz | Preserve slow waves, remove line noise |
| PPG | 0.5 Hz | 8 Hz | - | Isolate pulse waveform |
| EDA | 0.05 Hz | 1 Hz | - | Preserve SCR events |
| EMG | 20 Hz | 500 Hz | 50/60 Hz | Muscle activity band |

```rust
use dpb_core::filters::{Butterworth, NotchFilter};

// ECG preprocessing pipeline
let mut ecg = signal.clone();
ecg.apply_filter(Butterworth::highpass(0.5, fs, 2))?;
ecg.apply_filter(Butterworth::lowpass(40.0, fs, 4))?;
ecg.apply_filter(NotchFilter::new(60.0, fs, 30.0))?;  // US powerline
```

### 12.4 Dataset-Specific Normalization

#### MIT-BIH Arrhythmia
```rust
// MIT-BIH: 11-bit ADC, 360 Hz, ±10 mV range
// Convert to millivolts
let mv_per_unit = 10.0 / 2048.0;  // (range / 2^(bits-1))
for sample in signal.samples_mut() {
    *sample *= mv_per_unit;
}
```

#### PTB-XL
```rust
// PTB-XL: Already in μV, 500 Hz
// Recommended: Convert to mV for consistency
for sample in signal.samples_mut() {
    *sample /= 1000.0;  // μV to mV
}
```

#### Sleep-EDF
```rust
// Sleep-EDF: EDF format with physical dimensions in μV
// Use EdfReader which handles unit conversion
let reader = EdfReader::open(path)?;
let signal = reader.read_signal_physical(channel)?;  // Returns μV
```

### 12.5 Motion Data Normalization

```python
# For IMU/accelerometer data (e.g., WearGait-PD, Daphnet)
import numpy as np

def normalize_imu(accel_data: np.ndarray) -> np.ndarray:
    """
    Normalize 3-axis accelerometer data.

    Args:
        accel_data: Shape (N, 3) for x, y, z axes

    Returns:
        Normalized data with gravity removed and scaled
    """
    # Remove gravity (assume stationary periods exist)
    gravity = np.median(accel_data, axis=0)
    accel_data = accel_data - gravity

    # Scale to g-units if in raw ADC values
    # (Adjust scale factor based on sensor specs)
    scale = 9.81 / 4096  # Example for 12-bit ADC, ±2g range
    accel_data *= scale

    return accel_data

def compute_magnitude(accel_data: np.ndarray) -> np.ndarray:
    """Compute acceleration magnitude (sensor-orientation invariant)."""
    return np.sqrt(np.sum(accel_data ** 2, axis=1))
```

### 12.6 Audio/Speech Normalization

```python
import librosa
import numpy as np

def normalize_speech(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Normalize speech audio for cognitive assessment.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (16kHz standard for speech)

    Returns:
        Normalized audio array
    """
    # Load and resample
    audio, sr = librosa.load(audio_path, sr=target_sr)

    # Peak normalization
    audio = audio / np.max(np.abs(audio))

    # Optional: RMS normalization for consistent loudness
    target_rms = 0.1
    current_rms = np.sqrt(np.mean(audio ** 2))
    audio = audio * (target_rms / current_rms)

    return audio

def extract_speech_features(audio: np.ndarray, sr: int = 16000) -> dict:
    """Extract standard speech features for cognitive assessment."""
    return {
        'mfcc': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
        'f0': librosa.yin(audio, fmin=50, fmax=500),
        'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sr),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio),
    }
```

---

## 13. DPB Integration Guidelines

### 13.1 Supported Data Formats

| Format | Module | Read | Write | Notes |
|--------|--------|------|-------|-------|
| WFDB | `dpb_core::io::wfdb` | ✅ | ✅ | PhysioNet standard |
| EDF/EDF+ | `dpb_core::io::edf` | ✅ | ✅ | Sleep, EEG |
| BDF | `dpb_core::io::bdf` | ✅ | ❌ | BioSemi 24-bit |
| GDF | `dpb_core::io::gdf` | ✅ | ❌ | General Data Format |
| XDF | `dpb_core::io::xdf` | ✅ | ❌ | Lab Streaming Layer |
| BIDS | `dpb_core::io::bids` | ✅ | ✅ | Neuroimaging standard |
| FHIR | `dpb_core::io::fhir` | ✅ | ✅ | Healthcare interop |

### 13.2 Unified Reader

```rust
use dpb_core::io::UnifiedReader;

// Auto-detect format and read
let reader = UnifiedReader::open("./data/recording.edf")?;
let signals = reader.read_all()?;
let metadata = reader.metadata();
```

### 13.3 BIDS Dataset Creation

```rust
use dpb_core::io::bids::{BidsDataset, DatasetDescription};

// Create a new BIDS-compliant dataset
let desc = DatasetDescription::builder()
    .name("PD Gait Study")
    .bids_version("1.8.0")
    .authors(vec!["Research Team".into()])
    .license("CC-BY-4.0".into())
    .build()?;

let dataset = BidsDataset::create("./output/bids_dataset", desc)?;

// Add a participant
dataset.add_participant("sub-001", json!({
    "age": 65,
    "sex": "M",
    "group": "patient"
}))?;

// Add a recording session
dataset.add_session("sub-001", "ses-01")?;
```

### 13.4 Synthetic Data Generation

```rust
use dpb_bench::datasets::{SyntheticECG, SyntheticGait, SyntheticTremor};

// Generate synthetic ECG with known R-peaks
let ecg = SyntheticECG::builder()
    .sample_rate(250.0)
    .heart_rate(72.0)
    .noise_level(0.05)
    .duration_seconds(60.0)
    .seed(42)
    .build()?;

let (signal, ground_truth) = ecg.generate();

// Generate synthetic gait data
let gait = SyntheticGait::builder()
    .stride_frequency(1.8)  // Hz
    .stance_ratio(0.6)
    .noise_level(0.1)
    .build()?;

// Generate Parkinsonian tremor
let tremor = SyntheticTremor::builder()
    .frequency(5.0)  // 4-6 Hz typical for PD
    .amplitude(0.3)
    .with_harmonics(true)
    .build()?;
```

### 13.5 Data Augmentation Pipeline

```rust
use dpb_synth::augmentation::*;

let pipeline = AugmentationPipeline::builder()
    .add(GaussianNoise::new(0.01))
    .add(BaselineWander::new(0.5, 0.1))
    .add(PowerlineNoise::new(60.0, 0.05))
    .add(TimeWarp::new(0.1))
    .build();

let augmented = pipeline.apply(&signal)?;
```

---

## 14. NeuroPlay Integration Strategy

### 14.1 Recommended Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    NeuroPlay Data Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. PRETRAINING (Motion Foundation)                            │
│     ├── Motion-X (15.6M poses) ──► Pose estimation backbone    │
│     ├── Human3.6M ──► 3D pose regression                       │
│     └── KIT Whole-Body ──► Object interaction patterns         │
│                                                                 │
│  2. HEALTHY BASELINES                                          │
│     ├── Sit-to-Walk MoCap ──► Age-stratified normative ranges  │
│     ├── 3D Gait & Running ──► Walking/running kinematics       │
│     └── AddBiomechanics ──► Physically-validated motion        │
│                                                                 │
│  3. CLINICAL FINE-TUNING                                       │
│     ├── WearGait-PD ──► MDS-UPDRS correlation                  │
│     ├── StrokeRehab ──► Action primitive detection             │
│     └── KIMORE ──► Motor dysfunction scoring                   │
│                                                                 │
│  4. SPEECH BIOMARKERS                                          │
│     ├── Bridge2AI-Voice ──► MFCCs, spectrograms                │
│     ├── DementiaBank ──► Linguistic features                   │
│     └── ADReSS ──► AD classification                           │
│                                                                 │
│  5. REAL-WORLD VALIDATION                                      │
│     └── PD@Home ──► Naturalistic behavior (closest to          │
│                     telehealth scenarios)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 14.2 Controller Input Mapping

| Controller Signal | Dataset Correlation | Clinical Target |
|-------------------|---------------------|-----------------|
| Joy-Con gyroscope | PD@Home wrist gyro | Tremor frequency (4-6 Hz) |
| Joy-Con accelerometer | Daphnet FoG | Freezing of gait events |
| DualSense haptic feedback | StrokeRehab | Motor control primitives |
| Button press timing | StrokeVision-Bench | Reaction time, coordination |
| Motion sensor jitter | PADS smartwatch | Bradykinesia assessment |

### 14.3 Feature Extraction Pipeline

```rust
// NeuroPlay controller feature extraction
use neuroplay::features::{TremorAnalyzer, GaitAnalyzer, ReactionTimeAnalyzer};

// Tremor detection from controller IMU
let tremor = TremorAnalyzer::new()
    .frequency_range(3.0..8.0)  // PD tremor band
    .window_size_ms(2000)
    .overlap(0.5)
    .analyze(&controller_imu)?;

// Gait pattern from motion during walking games
let gait = GaitAnalyzer::new()
    .detect_heel_strikes(&accelerometer)
    .compute_stride_variability()
    .compute_asymmetry();

// Reaction time from button presses
let reaction = ReactionTimeAnalyzer::new()
    .stimulus_times(&game_events)
    .response_times(&button_presses)
    .compute_metrics();
```

---

## 15. Key Repositories & Toolkits

### 15.1 Data Repositories

| Repository | Focus | URL |
|------------|-------|-----|
| PhysioNet | Physiological signals, gait, EEG | [physionet.org](https://physionet.org) |
| DementiaBank / TalkBank | Speech/language pathology | [talkbank.org](https://talkbank.org) |
| Mobilize Center (Stanford) | Movement/gait datasets | [mobilize.stanford.edu](https://mobilize.stanford.edu/data/available-datasets/) |
| SimTK | Biomechanics, rehabilitation | [simtk.org](https://simtk.org) |
| NSRR | Sleep research | [sleepdata.org](https://sleepdata.org) |
| OpenNeuro | Neuroimaging (BIDS) | [openneuro.org](https://openneuro.org) |

### 15.2 Software Toolkits

| Toolkit | Focus | URL |
|---------|-------|-----|
| MMPose (OpenMMLab) | Unified pose estimation | [GitHub](https://github.com/open-mmlab/mmpose) |
| MNE-Python | EEG/MEG analysis | [mne.tools](https://mne.tools) |
| WFDB | PhysioNet I/O | [GitHub](https://github.com/MIT-LCP/wfdb-python) |
| NeuroKit2 | Biosignal processing | [GitHub](https://github.com/neuropsychology/NeuroKit) |
| Librosa | Audio/speech analysis | [librosa.org](https://librosa.org) |
| VisionMD | MDS-UPDRS video analysis | Nature npj PD 2025 |

### 15.3 Community Resources

| Resource | Description | URL |
|----------|-------------|-----|
| voice_datasets | Community-curated speech datasets | [GitHub](https://github.com/jim-schwoebel/voice_datasets) |
| awesome-biological-signals | Curated biosignal resources | GitHub |
| Open Neuromorphic | Neuromorphic computing resources | [open-neuromorphic.org](https://open-neuromorphic.org) |

---

## Citation

When using datasets through the DPB framework, please cite:

1. The original dataset paper(s)
2. The DPB Framework

```bibtex
@software{dpb_framework,
  title = {Delta-Predictive Biosensing Framework},
  author = {AuraSense},
  year = {2025},
  url = {https://github.com/dcharlot65-aurasense/Delta-Predictive-Biosensing}
}
```

---

## Updates

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | January 2026 | Initial consolidated reference |

---

*AuraSense Tech Corporation / Delta-Predictive Biosensing Framework*
*Last updated: January 2026*
