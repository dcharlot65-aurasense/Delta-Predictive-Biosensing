# Scientific Sources and References

> **Delta-Predictive Biosensing (DPB) Framework**
>
> This document provides scientific justification for the DPB approach and comprehensive references for all algorithms and methods implemented in the framework.

---

## Table of Contents

1. [Core Approach: Spiking Neural Networks](#1-core-approach-spiking-neural-networks)
2. [Neuromorphic Computing](#2-neuromorphic-computing)
3. [Neuron Models](#3-neuron-models)
4. [Learning Rules & Plasticity](#4-learning-rules--plasticity)
5. [Reservoir Computing](#5-reservoir-computing)
6. [Signal Processing Algorithms](#6-signal-processing-algorithms)
7. [Biosignal Analysis](#7-biosignal-analysis)
8. [Transformers in Biosignal Analysis](#8-transformers-in-biosignal-analysis) *(NEW 2023-2025)*
9. [TinyML & Edge AI for Wearables](#9-tinyml--edge-ai-for-wearables) *(NEW 2023-2025)*
10. [Deep Learning for Clinical Detection](#10-deep-learning-for-clinical-detection) *(NEW 2023-2025)*
11. [Sleep Stage Classification](#11-sleep-stage-classification) *(NEW 2023-2025)*
12. [Stress Detection & Mental Health](#12-stress-detection--mental-health) *(NEW 2023-2025)*
13. [Multimodal Biosignal Fusion](#13-multimodal-biosignal-fusion) *(NEW 2023-2025)*
14. [Federated Learning in Healthcare](#14-federated-learning-in-healthcare) *(NEW 2023-2025)*
15. [Model Calibration & Uncertainty](#15-model-calibration--uncertainty)
16. [Explainability & Interpretability](#16-explainability--interpretability)
17. [Knowledge Distillation](#17-knowledge-distillation)
18. [Data Standards](#18-data-standards)
19. [Neuromorphic Hardware Platforms](#19-neuromorphic-hardware-platforms)
20. [Datasets & Benchmarks](#20-datasets--benchmarks)
21. [Foundational Textbooks](#21-foundational-textbooks)

---

## 1. Core Approach: Spiking Neural Networks

### 1.1 Why SNNs for Biosignals?

Spiking Neural Networks (SNNs) are the third generation of neural networks that use discrete spikes to transmit information, mimicking biological neural communication. They are particularly well-suited for biosignal processing due to their:

- **Temporal precision**: SNNs encode information in spike timing, matching the inherently temporal nature of biosignals
- **Energy efficiency**: Event-driven computation reduces power consumption by 100x compared to ANNs
- **Biological plausibility**: SNNs can interface naturally with neural recording and stimulation systems

### 1.2 Key Review Papers

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| Spiking Neural Networks for Biomedical Signal Analysis | Various | 2024 | Biomedical Engineering Letters | [PMC11362400](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/) |
| Exploring the Potential of SNNs in Biomedical Applications | Various | 2024 | Biomedical Engineering Letters | [Springer](https://link.springer.com/article/10.1007/s13534-024-00403-1) |
| Direct Learning-Based Deep Spiking Neural Networks: A Review | Various | 2023 | Frontiers in Neuroscience | [PMC10313197](https://pmc.ncbi.nlm.nih.gov/articles/PMC10313197/) |
| Direct Training High-Performance Deep SNNs: A Review | Various | 2024 | Frontiers in Neuroscience | [PMC11322636](https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/) |
| A Comprehensive Review of SNNs: Interpretation, Optimization, Efficiency | Various | 2023 | arXiv | [2303.10780](https://arxiv.org/abs/2303.10780) |
| SNNs for EEG Signal Analysis: From Theory to Practice | Various | 2025 | Neural Networks | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S089360802501007X) |

### 1.3 Surrogate Gradient Methods

Training SNNs requires addressing the non-differentiability of spike generation. Surrogate gradients provide smooth approximations.

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| Surrogate Gradient Learning in Spiking Neural Networks | Neftci, Mostafa, Zenke | 2019 | IEEE Signal Processing Magazine | [arXiv:1901.09948](https://arxiv.org/pdf/1901.09948) |
| The Remarkable Robustness of Surrogate Gradient Learning | Various | 2021 | Neural Computation | [MIT Press](https://direct.mit.edu/neco/article/33/4/899/97482/The-Remarkable-Robustness-of-Surrogate-Gradient) |
| Learnable Surrogate Gradient for Direct Training SNNs | Various | 2023 | IJCAI | [PDF](https://www.ijcai.org/proceedings/2023/0335.pdf) |

**Software**: [snnTorch](https://snntorch.readthedocs.io/) - Python library for SNN training with surrogate gradients

---

## 2. Neuromorphic Computing

### 2.1 Overview

Neuromorphic computing uses brain-inspired architectures for energy-efficient, real-time processing. Key advantages for healthcare:

- Low power consumption (critical for wearables and implants)
- Real-time processing with low latency
- Fault tolerance and graceful degradation
- Natural interface with biological neural systems

### 2.2 Key Papers

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| Neuromorphic Applications in Medicine | Johns Hopkins | 2023 | Journal of Neural Engineering | [IOPscience](https://iopscience.iop.org/article/10.1088/1741-2552/aceca3) |
| NeuroCARE: A Generic Neuromorphic Edge Computing Framework | Various | 2023 | Frontiers in Neuroscience | [Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1093865/full) |
| Neuromorphic Algorithms for Brain Implants: A Review | Various | 2024 | PMC | [PMC12021827](https://pmc.ncbi.nlm.nih.gov/articles/PMC12021827/) |
| SNN Based Neuromorphic Computing Towards Healthcare | Various | 2023 | Springer | [SpringerLink](https://link.springer.com/chapter/10.1007/978-3-031-45878-1_18) |

---

## 3. Neuron Models

### 3.1 Leaky Integrate-and-Fire (LIF)

The foundational spiking neuron model, proposed by Louis Lapicque in 1907.

| Resource | Type | Link |
|----------|------|------|
| Tutorial: The LIF Neuron Model | Tutorial | [Neuromatch Academy](https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html) |
| Linear LIF Neuron Model Based SNNs | Paper | [PMC9448910](https://pmc.ncbi.nlm.nih.gov/articles/PMC9448910/) |
| Integrate-and-Fire Models | Textbook Chapter | [Neuronal Dynamics](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) |
| Generalized LIF Models Classify Multiple Neuron Types | Paper | [Nature Communications](https://www.nature.com/articles/s41467-017-02717-4) |

### 3.2 Izhikevich Model

Combines biological plausibility with computational efficiency, reproducing 20+ firing patterns.

| Resource | Authors | Year | Link |
|----------|---------|------|------|
| Simple Model of Spiking Neurons | Izhikevich | 2003 | [IEEE TNNLS](https://www.izhikevich.org/publications/spikes.pdf) |
| Which Model to Use for Cortical Spiking Neurons? | Izhikevich | 2004 | [PDF](https://www.izhikevich.org/publications/whichmod.pdf) |
| ModelDB Implementation | - | - | [ModelDB](https://modeldb.science/showmodel?model=39948) |

### 3.3 Hodgkin-Huxley Model

The Nobel Prize-winning biophysical model of action potentials (1963).

| Resource | Authors | Year | Link |
|----------|---------|------|------|
| Original Paper | Hodgkin & Huxley | 1952 | [Wikipedia](https://en.wikipedia.org/wiki/Hodgkin–Huxley_model) |
| A Brief Historical Perspective | Various | 2012 | [PMC3424716](https://pmc.ncbi.nlm.nih.gov/articles/PMC3424716/) |
| The Most Famous Paper in Neuroscience | Mentalab | 2024 | [Mentalab](https://mentalab.com/hodgkin-and-huxley/) |

### 3.4 Adaptive Exponential (AdEx) Model

| Resource | Link |
|----------|------|
| Scholarpedia Article | [Scholarpedia](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model) |

---

## 4. Learning Rules & Plasticity

### 4.1 Spike-Timing Dependent Plasticity (STDP)

The biological learning rule where synaptic strength changes based on the relative timing of pre- and post-synaptic spikes.

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| STDP Overview | Scholarpedia | - | [Scholarpedia](http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity) |
| The Spike Timing Dependence of Plasticity | Various | 2012 | [PMC3431193](https://pmc.ncbi.nlm.nih.gov/articles/PMC3431193/) |
| STDP: A Consequence of More Fundamental Learning Rules | Various | 2010 | [Frontiers](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2010.00019/full) |
| Reward-Modulated STDP with Application to Biofeedback | Various | 2008 | [PLOS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000180) |

### 4.2 Other Learning Rules

| Rule | Description | Key Reference |
|------|-------------|---------------|
| **BCM Rule** | Bienenstock-Cooper-Munro threshold-based plasticity | Bienenstock et al., 1982 |
| **Oja's Rule** | Principal component extraction | Oja, 1982 |
| **Hebbian Learning** | "Cells that fire together, wire together" | Hebb, 1949 |

---

## 5. Reservoir Computing

### 5.1 Echo State Networks (ESN)

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| Echo State Network Overview | Scholarpedia | - | [Scholarpedia](http://www.scholarpedia.org/article/Echo_state_network) |
| Reservoir Computing Approaches to RNN Training | Lukoševičius & Jaeger | 2009 | Computer Science Review |
| A Practical Guide to Applying ESNs | Lukoševičius | 2012 | Springer |

### 5.2 Liquid State Machines (LSM)

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| Real-Time Computing Without Stable States | Maass et al. | 2002 | Neural Computation |
| Deep Reservoir Computing using ESN and LSM | Various | 2022 | [IEEE](https://ieeexplore.ieee.org/document/9858322/) |

**Key Concept**: Both ESN and LSM use a fixed, high-dimensional "reservoir" to transform inputs into a space where they can be linearly separated, requiring only readout layer training.

---

## 6. Signal Processing Algorithms

### 6.1 Empirical Mode Decomposition (EMD)

Adaptive decomposition method for nonlinear, non-stationary signals.

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| Hilbert-Huang Transform Original Paper | Huang et al. | 1998 | NASA |
| EMD: An Introduction | ResearchGate | - | [ResearchGate](https://www.researchgate.net/publication/221534245_Empirical_Mode_Decomposition_-_an_introduction) |
| EMD Python Package | Quinn et al. | 2021 | [PMC7610596](https://pmc.ncbi.nlm.nih.gov/articles/PMC7610596/) |

**Variants Implemented**:
- EMD (Empirical Mode Decomposition)
- EEMD (Ensemble EMD) - addresses mode mixing
- CEEMDAN (Complete Ensemble EMD with Adaptive Noise)
- VMD (Variational Mode Decomposition)

**Software**: [emd](https://pypi.org/project/emd/) - Python package for EMD and Hilbert-Huang analysis

### 6.2 Wavelet Transform

| Paper | Topic | Link |
|-------|-------|------|
| Fast CWT for Real-Time Analysis | High-quality time-frequency analysis | [Nature Comp Sci](https://www.nature.com/articles/s43588-021-00183-z) |
| CWT Processing of Biosignals | Heart rate from video | [PMC4916481](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4916481/) |
| Morlet Wavelet | Neuroscience applications | [ScienceDirect](https://www.sciencedirect.com/topics/neuroscience/morlet-wavelet) |

**Software**: [PyWavelets](https://pywavelets.readthedocs.io/) - Python wavelet transform library

### 6.3 Independent Component Analysis (ICA)

Blind source separation technique, essential for EEG artifact removal.

| Paper | Topic | Link |
|-------|-------|------|
| EEGLAB ICA Tutorial | Artifact removal | [EEGLAB](https://eeglab.org/tutorials/06_RejectArtifacts/RunICA.html) |
| Comparison of ICA Algorithms | FastICA vs Infomax | [PubMed](https://pubmed.ncbi.nlm.nih.gov/28268452/) |
| MNE-Python ICA | Artifact correction | [MNE](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) |
| ICLabel | Automatic component classification | EEGLAB plugin |

**Key Finding**: FastICA provides the best discrimination between muscle-free and muscle-contaminated recordings in the shortest time.

---

## 7. Biosignal Analysis

### 7.1 ECG Analysis

#### Pan-Tompkins Algorithm (QRS Detection)

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| A Real-Time QRS Detection Algorithm | Pan & Tompkins | 1985 | IEEE Trans Biomed Eng | [IEEE](https://ieeexplore.ieee.org/document/4122029/) |
| Pan-Tompkins++: Robust R-Peak Detection | Various | 2022 | arXiv | [arXiv](https://arxiv.org/abs/2211.03171) |

**Performance**: 99.3% QRS detection accuracy on MIT-BIH Arrhythmia Database

### 7.2 Heart Rate Variability (HRV)

| Paper | Topic | Link |
|-------|-------|------|
| HRV Standards (1996) | Task Force Guidelines | [ESC Guidelines](https://www.escardio.org/static-file/Escardio/Guidelines/Scientific-Statements/guidelines-Heart-Rate-Variability-FT-1996.pdf) |
| HRV Metrics and Norms | Overview | [PMC5624990](https://pmc.ncbi.nlm.nih.gov/articles/PMC5624990/) |
| HRV Measurement and Clinical Utility | Review | [PMC6932537](https://pmc.ncbi.nlm.nih.gov/articles/PMC6932537/) |
| HRV in Congestive Heart Failure | Clinical applications | [PMC12135661](https://pmc.ncbi.nlm.nih.gov/articles/PMC12135661/) |

**Key Metrics**:
- **SDNN**: Standard deviation of NN intervals (total variability)
- **RMSSD**: Root mean square of successive differences (vagal tone)
- **pNN50**: Percentage of successive intervals differing >50ms
- **LF/HF Ratio**: Sympathovagal balance

### 7.3 EEG Analysis

#### Frequency Bands

| Band | Frequency Range | Association |
|------|-----------------|-------------|
| Delta | 0.5-4 Hz | Deep sleep, healing |
| Theta | 4-8 Hz | Drowsiness, creativity |
| Alpha | 8-13 Hz | Relaxed wakefulness |
| Beta | 13-30 Hz | Active thinking, focus |
| Gamma | 30-100 Hz | Higher cognition, binding |

| Paper | Topic | Link |
|-------|-------|------|
| EEG Frequency Bands in Psychiatric Disorders | Review | [Frontiers](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2018.00521/full) |
| EEG Band Power in Infancy | Developmental | [PMC3347767](https://pmc.ncbi.nlm.nih.gov/articles/PMC3347767/) |
| Inconsistency of Band Definitions | Methodology | [Sapien Labs](https://sapienlabs.org/the-remarkable-inconsistency-of-frequency-band-definitions/) |

### 7.4 PPG (Photoplethysmography)

| Paper | Topic | Link |
|-------|-------|------|
| PPG and SpO2 for Health Monitoring | Progress review | [PMC6431353](https://pmc.ncbi.nlm.nih.gov/articles/PMC6431353/) |
| Photoplethysmogram Analysis and Applications | Integrative review | [PMC8920970](https://pmc.ncbi.nlm.nih.gov/articles/PMC8920970/) |
| Wavelet PPG Peak Detection | Near real-time | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405896318333688) |

### 7.5 EDA (Electrodermal Activity)

| Paper | Topic | Link |
|-------|-------|------|
| Wikipedia Overview | General | [Wikipedia](https://en.wikipedia.org/wiki/Electrodermal_activity) |
| Continuous Phasic EDA Measure | Methodology | [PMC2892750](https://pmc.ncbi.nlm.nih.gov/articles/PMC2892750/) |
| Guide for Analyzing EDA | Practical guide | [Birmingham](https://www.birmingham.ac.uk/documents/college-les/psych/saal/guide-electrodermal-activity.pdf) |

**Components**:
- **Tonic (SCL)**: Slow-changing baseline, reflects general arousal
- **Phasic (SCR)**: Rapid responses to stimuli, peaks within 1-5 seconds

### 7.6 PPG-Based Blood Pressure Estimation *(NEW 2023-2025)*

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| Cuff-less BP Monitoring via PPG Using Hybrid CNN-BiLSTM with Attention | Various | 2025 | Scientific Reports | [Nature](https://www.nature.com/articles/s41598-025-07087-2) |
| Deep Learning Approaches for Continuous BP from PPG | Various | 2025 | ScienceDirect | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2665917425000601) |
| Cuffless BP Monitoring: AI and Edge Computing Solutions Review | Various | 2025 | Archives of Computational Methods | [Springer](https://link.springer.com/article/10.1007/s11831-025-10415-4) |
| A Benchmark for ML-Based Non-Invasive BP Estimation Using PPG | Various | 2023 | Scientific Data | [Nature](https://www.nature.com/articles/s41597-023-02020-6) |

**Key Methods**:
- Autoencoder-LSTM achieving MAE of 1.05 (SBP) and 0.92 (DBP)
- Temporal Convolutional Networks (TCN) with attention mechanisms
- Hybrid CNN-BiLSTM architectures for spatial-temporal feature extraction
- Transfer learning for personalized calibration

---

## 8. Transformers in Biosignal Analysis

*(Literature from 2023-2025)*

### 8.1 Overview

Transformer architectures have emerged as powerful tools for biosignal analysis, leveraging self-attention mechanisms to capture long-range dependencies in temporal data.

### 8.2 Key Review Papers

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| Transformers in Biosignal Analysis: A Review | Various | 2024 | Information Fusion | [ACM](https://dl.acm.org/doi/10.1016/j.inffus.2024.102697) |
| A Review of Hybrid EEG-Based Multimodal HCI Using Deep Learning | Various | 2025 | Biomedical Engineering Letters | [Springer](https://link.springer.com/article/10.1007/s13534-025-00469-5) |
| A Comprehensive Review of Biosignal Foundation Models | Lee et al. | 2024 | TechRxiv | [TechRxiv](https://www.techrxiv.org/) |

### 8.3 Foundation Models for Biosignals

| Model | Year | Modality | Description |
|-------|------|----------|-------------|
| **BIOT** | 2023 | Multi-biosignal | Biosignal foundation model |
| **BrainBERT** | 2023 | EEG | BERT-style pre-training for brainwaves |
| **Brant** | 2023 | EEG | Foundation model for brain signals |

### 8.4 Transformer Applications

| Paper | Application | Year | Link |
|-------|-------------|------|------|
| DeepECG-Net: Hybrid Transformer for Real-Time ECG Anomaly Detection | ECG | 2025 | [Nature](https://www.nature.com/articles/s41598-025-07781-1) |
| Hybrid CNN-Transformer for Arrhythmia Detection Using Stockwell Transform | ECG | 2025 | [Nature](https://www.nature.com/articles/s41598-025-92582-9) |
| CNN-Informer for Seizure Detection on Long-Term EEG | EEG | 2024 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608024007792) |
| Multidimensional Transformer + RNN Fusion for Seizure Prediction | EEG | 2024 | [Springer](https://link.springer.com/article/10.1186/s12967-024-05678-7) |
| Arrhythmia Classification from 12-Lead ECG Using Vision Transformers | ECG | 2025 | [arXiv](https://arxiv.org/abs/2502.17887) |

**Key Advantages**:
- Self-attention captures long-range temporal dependencies
- Parallel processing improves training efficiency
- Multi-head attention enables learning diverse feature representations
- Hybrid CNN-Transformer models combine local feature extraction with global context

---

## 9. TinyML & Edge AI for Wearables

*(Literature from 2023-2025)*

### 9.1 Overview

TinyML enables deployment of deep learning models on ultra-low-power microcontrollers (MCUs), critical for wearable health monitoring devices.

### 9.2 Key Statistics (2024-2025)

- Global TinyML market valued at **$1.13 billion** (2024), projected to reach **$4.6 billion** by 2033
- Over **1.4 billion** wearable healthcare devices sold globally in 2023, with 28% using TinyML
- Compiler technologies (TVM, CMSIS-NN) reduced model footprints by **45%** on average
- Deployment possible on devices with as little as **64 KB RAM**

### 9.3 Key Papers

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| TinyML: Enabling Inference Deep Learning Models on Ultra-Low-Power IoT Edge Devices | Various | 2022 | Micromachines | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9227753/) |
| From Tiny Machine Learning to Tiny Deep Learning: A Survey | Various | 2025 | arXiv | [arXiv](https://arxiv.org/html/2506.18927v1) |
| Deploying TinyML for Energy-Efficient Object Detection | Various | 2025 | Scientific Reports | [Nature](https://www.nature.com/articles/s41598-025-27818-9) |
| Reliable ECG Anomaly Detection on Edge Devices for IoMT | Various | 2025 | Sensors | [MDPI](https://www.mdpi.com/1424-8220/25/8/2496) |
| AI-Powered Wearable Sensors for Health Monitoring | Various | 2025 | Preprints | [Preprints.org](https://www.preprints.org/manuscript/202507.2601/v1) |

### 9.4 Hardware Platforms

| Platform | Description | Power |
|----------|-------------|-------|
| **Raspberry Pi** | Edge AI with TinyML frameworks | ~1-5W |
| **Arduino** | Ultra-low-power MCU-based inference | ~mW |
| **BioGAP-Ultra** | Multimodal biosensing with embedded AI | mW-range |
| **Empatica E4** | Medical-grade wrist-worn wearable | Low power |

### 9.5 Applications

- Real-time ECG anomaly detection
- Continuous glucose monitoring
- Stress detection from HRV/EDA
- Sleep quality assessment
- Fall detection and activity recognition

**Key Resource**: [Edge AI Foundation / tinyML](https://www.tinyml.org/)

---

## 10. Deep Learning for Clinical Detection

*(Literature from 2023-2025)*

### 10.1 Arrhythmia Detection

#### CNN and LSTM Hybrid Models

| Paper | Authors | Year | Performance | Link |
|-------|---------|------|-------------|------|
| Deep Learning for ECG Arrhythmia Detection: Progress 2017-2023 | Various | 2023 | Review | [Frontiers](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1246746/full) |
| LDCNN: Linear Deep CNN for Arrhythmia | Various | 2024 | 99.24% (PTB), 99.38% (MIT-BIH) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11366442/) |
| CNN-LSTM-SE Arrhythmia Classification | Various | 2024 | Channel attention mechanism | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11478372/) |
| Transformer-Based DNN for Arrhythmia Detection | Hu & Chen | 2022 | Continuous ECG segments | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35227968/) |

**Key Finding**: Novel CNN-Transformer models can classify heartbeats without explicit segmentation at inference.

### 10.2 Seizure Detection & Prediction

| Paper | Authors | Year | Performance | Link |
|-------|---------|------|-------------|------|
| Deep Learning in Intracranial EEG for Seizure Detection | Various | 2025 | Review of advances | [Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1677898/full) |
| Review of Epilepsy Detection Methods Based on EEG and Deep Learning | Various | 2024 | Comprehensive review | [Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1468967/full) |
| Residual and Bidirectional LSTM for Seizure Detection | Various | 2024 | ResBiLSTM | [Frontiers](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1415967/full) |
| 1D CNN-LSTM with DWT for Seizure Detection | Various | 2025 | TUSZ: 94.32%, BONN: 97.24%, CHB-MIT: 96.94% | [Nature](https://www.nature.com/articles/s41598-025-18479-9) |
| CBAM-3D CNN-LSTM for Seizure Prediction | Various | 2023 | 97.95% accuracy, 0.017 h⁻¹ FAR | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10328218/) |
| Enhanced Hybrid CNN with Attention for Seizure Detection | Various | 2025 | Integrated attention | [AIMS](https://www.aimspress.com/article/doi/10.3934/mbe.2025004) |

**Emerging Approaches**:
- Neuromorphic systems for real-time, low-power detection
- Hyperdimensional computing for implantable devices
- Hybrid Transformer-RNN fusion architectures

---

## 11. Sleep Stage Classification

*(Literature from 2023-2025)*

### 11.1 Overview

Automatic sleep staging traditionally required polysomnography (PSG) in clinical settings. Recent deep learning advances enable classification from single-channel signals (EEG or PPG).

### 11.2 Key Papers

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| Automatic Sleep Stage Classification Using CNNs | Various | 2024 | PLOS One | [PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0297582) |
| SleepPPG-Net2: Deep Learning Generalization for Sleep Staging from PPG | Various | 2024 | arXiv | [arXiv](https://arxiv.org/html/2404.06869v1) |
| Automatic Sleep Stage Classification: A Review | Various | 2024 | Artificial Intelligence Review | [Springer](https://link.springer.com/article/10.1007/s10462-024-10926-9) |
| ML-Empowered Sleep Staging Using Multi-Modality Signals | Various | 2024 | BMC Medical Informatics | [BMC](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-024-02522-2) |
| Optimising Sleep Stage Detection Using Minimal Non-EEG Signal Set | Alarcón et al. | 2025 | Journal of Sleep Research | [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/jsr.70266) |
| Deep Transfer Learning for Wearable Sleep Stage Classification | Various | 2021 | npj Digital Medicine | [Nature](https://www.nature.com/articles/s41746-021-00510-8) |

### 11.3 Signal Types

| Category | Signals | Typical Accuracy |
|----------|---------|------------------|
| **PSG** | EEG, EOG, EMG | Gold standard |
| **Cardiorespiratory** | ECG, PPG, respiration | 69-95% |
| **Contactless** | Radar, Wi-Fi, audio | Emerging |

### 11.4 Key Datasets

| Dataset | Description |
|---------|-------------|
| Sleep-EDF (SEDF13, SEDF18) | Most widely used, PhysioNet |
| CAP-Sleep | PhysioNet |
| Sleep Heart Health Study (SHHS) | Large PSG dataset |
| NSRR | National Sleep Research Resource |

**Performance Highlights**:
- CNN with time-frequency analysis: **99.39%** accuracy (EEG C4-A1 channel)
- PPG-based classification: **94.63%** accuracy with transfer learning
- Four-class PPG staging (wake, light, deep, REM): Generalization across 6 datasets

---

## 12. Stress Detection & Mental Health

*(Literature from 2023-2025)*

### 12.1 Overview

Stress is a prevalent mental health concern with over **$300 billion** spent annually on treatments in the US alone. Physiological monitoring enables objective, continuous stress assessment.

### 12.2 Primary Physiological Signals

| Signal | Measurement | Stress Indicator |
|--------|-------------|------------------|
| **HRV** | Heart rate variability from ECG/PPG | Reduced HRV indicates stress |
| **EDA/GSR** | Electrodermal activity | Increased SCL/SCR with stress |
| **HR** | Heart rate | Elevated with acute stress |
| **Respiration** | Breathing rate/pattern | Irregular with anxiety |

### 12.3 Key Papers

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| State-of-the-Art of Stress Prediction from HRV Using AI | Various | 2023 | Cognitive Computation | [Springer](https://link.springer.com/article/10.1007/s12559-023-10200-0) |
| Adaptive System for Wearable Stress Detection Using Physiological Signals | Various | 2024 | arXiv | [arXiv](https://arxiv.org/html/2407.15252v1) |
| Real-Time Stress Prediction Using Wearable Devices | Various | 2024 | PMC | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11230864/) |
| PPG-Based HRV Analysis and ML for Real-Time Stress Quantification | Various | 2025 | APL Bioengineering | [AIP](https://pubs.aip.org/aip/apb/article/9/2/026103/3342428) |
| Cross Dataset Analysis for HRV-Based Stress Detection Generalizability | Various | 2023 | Sensors | [MDPI](https://www.mdpi.com/1424-8220/23/4/1807) |
| ML-Based Human Stress Detection Using Physiological Sensors | Various | 2024 | Arabian J. Science & Engineering | [Springer](https://link.springer.com/article/10.1007/s13369-024-09927-1) |

### 12.4 Machine Learning Performance

| Method | Typical Accuracy | Notes |
|--------|------------------|-------|
| **SVM** | 85-92% | Most commonly used |
| **Random Forest** | 88-95% | Best for tree-based |
| **Deep Learning** | 90-95% | With EDA+HR combination |
| **KNN** | 80-88% | Simple baseline |

**Key Finding**: EDA + HR combination achieves best performance (>95%) in laboratory environments.

### 12.5 Key Datasets

| Dataset | Device | Signals |
|---------|--------|---------|
| **WESAD** | Empatica E4 + Chest band | PPG, EDA, HR, respiration |
| **SWELL** | Wearables | Multiple physiological |

---

## 13. Multimodal Biosignal Fusion

*(Literature from 2023-2025)*

### 13.1 Overview

Combining complementary biosignals (EEG, ECG, PPG, EMG, EDA) improves estimation accuracy and robustness against noise and motion artifacts.

### 13.2 Key Papers

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| EEG-Based Multimodal Learning for Emotion Recognition: A Review | Various | 2025 | Artificial Intelligence Review | [Springer](https://link.springer.com/article/10.1007/s10462-025-11126-9) |
| BioGAP-Ultra: Modular Edge-AI Platform for Multimodal Biosignal Acquisition | Various | 2025 | arXiv | [arXiv](https://arxiv.org/html/2508.13728v1) |
| ECG Signal Reconstruction from PPG Using Hybrid Attention Network | Various | 2024 | EURASIP Journal | [Springer](https://link.springer.com/article/10.1186/s13634-024-01158-8) |
| Wearable Skin Biosignal Sensors: Developments and Future Directions | Kim et al. | 2024 | Advanced Sensor Research | [Wiley](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adsr.202300118) |
| Finetuning EEG Foundation Models on ECG/PPG for BP Estimation | Various | 2025 | arXiv | [arXiv](https://arxiv.org/html/2502.17460v1) |
| Self-Supervised Learning for Biomedical Signal Processing | Various | 2024 | medRxiv | [medRxiv](https://www.medrxiv.org/content/10.1101/2024.09.30.24314588v1) |

### 13.3 Fusion Strategies

| Strategy | Description | Example |
|----------|-------------|---------|
| **Early Fusion** | Concatenate raw signals | Multi-channel input to CNN |
| **Late Fusion** | Combine model outputs | Ensemble of modality-specific models |
| **Intermediate Fusion** | Fuse learned features | Cross-attention between modalities |
| **Graph-Based** | Model signal relationships | Graph Neural Networks on biosignal graphs |

### 13.4 Hardware for Multimodal Acquisition

| Platform | Signals | Features |
|----------|---------|----------|
| **BioGAP-Ultra** | EEG, EMG, ECG, PPG | Edge AI processing |
| **Empatica E4** | PPG, EDA, accelerometer, temperature | Medical-grade wearable |
| **OpenBCI** | EEG, EMG, ECG | Open-source, research-grade |

---

## 14. Federated Learning in Healthcare

*(Literature from 2023-2025)*

### 14.1 Overview

Federated Learning (FL) enables collaborative model training across healthcare institutions while preserving patient privacy—critical for HIPAA compliance and GDPR regulations.

### 14.2 Key Papers

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| Federated Learning for Privacy Preservation in Smart Healthcare: A Survey | Various | 2022 | PubMed | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35696470/) |
| Enhancing Healthcare Data Privacy and Interoperability with FL | Various | 2025 | PMC | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12192955/) |
| FL in Smart Healthcare: Privacy, Security, and Predictive Analytics Review | Various | 2024 | Healthcare (MDPI) | [MDPI](https://www.mdpi.com/2227-9032/12/24/2587) |
| FL in Healthcare: Model Misconducts, Security, Challenges | Various | 2024 | arXiv | [arXiv](https://arxiv.org/html/2405.13832v1) |
| Integration of Wearable Technology and AI in Digital Health | Various | 2025 | Journal of Cloud Computing | [Springer](https://link.springer.com/article/10.1186/s13677-025-00759-4) |

### 14.3 Key Benefits

- **Privacy**: Raw patient data never leaves local devices
- **Regulatory Compliance**: Supports HIPAA, GDPR requirements
- **Collaboration**: Multi-institution model training
- **Edge AI**: Local inference on wearables

### 14.4 Privacy-Preserving Techniques

| Technique | Description |
|-----------|-------------|
| **Differential Privacy** | Adds noise to gradients before sharing |
| **Secure Multi-Party Computation** | Cryptographic protocols for secure aggregation |
| **Homomorphic Encryption** | Computation on encrypted data |
| **Blockchain Integration** | Decentralized, tamper-proof model updates |

### 14.5 Applications in Biosignal Analysis

- ECG anomaly detection across hospitals
- Wearable health monitoring (continuous glucose, HRV)
- Disease prediction without centralizing patient data
- Collaborative model improvement for rare conditions

**Security Challenges**: Adversarial attacks, data poisoning, model inversion attacks

---

## 15. Model Calibration & Uncertainty

### 15.1 Temperature Scaling

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| On Calibration of Modern Neural Networks | Guo et al. | 2017 | ICML |
| Temperature Scaling Implementation | Pleiss | - | [GitHub](https://github.com/gpleiss/temperature_scaling) |
| Adaptive Temperature Scaling | Various | 2024 | [Springer](https://link.springer.com/article/10.1007/s00521-024-09505-4) |

**Key Concept**: Divides logits by a learned temperature parameter T > 0 to calibrate confidence estimates.

### 15.2 Platt Scaling

Binary classification calibration using logistic regression on outputs.

### 15.3 Isotonic Calibration

Non-parametric calibration using isotonic regression.

### 15.4 Uncertainty Quantification

| Method | Description |
|--------|-------------|
| **MC Dropout** | Monte Carlo sampling with dropout at inference |
| **Deep Ensembles** | Multiple models for epistemic uncertainty |
| **Expected Calibration Error (ECE)** | Primary calibration metric |
| **Brier Score** | Measures accuracy of probabilistic predictions |

---

## 16. Explainability & Interpretability

### 16.1 SHAP (SHapley Additive exPlanations)

| Resource | Type | Link |
|----------|------|------|
| SHAP Original Paper | Lundberg & Lee, 2017 | NeurIPS |
| SHAP Documentation | Official docs | [SHAP](https://shap.readthedocs.io/) |
| SHAP GitHub | Implementation | [GitHub](https://github.com/shap/shap) |
| Interpretable ML Book - SHAP Chapter | Tutorial | [Christoph Molnar](https://christophm.github.io/interpretable-ml-book/shap.html) |
| SHAP in Drug Development | Practical guide | [PMC11513550](https://pmc.ncbi.nlm.nih.gov/articles/PMC11513550/) |

### 16.2 Integrated Gradients

Attribution method that satisfies axioms of sensitivity and implementation invariance.

| Paper | Authors | Year | Publication |
|-------|---------|------|-------------|
| Axiomatic Attribution for Deep Networks | Sundararajan et al. | 2017 | ICML |

### 16.3 Attention Mechanisms

Temporal and spatial attention for interpretable predictions.

### 16.4 XAI in Healthcare *(NEW 2023-2025)*

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| LIME and SHAP in Alzheimer's Disease Detection: A Systematic Review | Various | 2024 | Brain Informatics | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10997568/) |
| XAI for Healthcare: A Systematic Review (2011-2022) | Various | 2022 | ScienceDirect | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169260722005429) |
| A Perspective on SHAP and LIME Methods | Various | 2023 | arXiv | [arXiv](https://arxiv.org/abs/2305.02012) |
| XAI in Disease Prediction: Systematic Review | Various | 2025 | BMC Medical Informatics | [BMC](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-025-02944-6) |
| Survey of Explainable AI Techniques in Healthcare | Various | 2023 | PMC | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9862413/) |
| LIME for Medical Imaging Analysis: Systematic Review | Various | 2024 | Computers in Biology and Medicine | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010482524016548) |

**Key Findings**:
- SHAP provides both global and local explanations; LIME is limited to local only
- SHAP can detect nonlinear associations; LIME fits local linear models
- Both methods are sensitive to feature collinearity—use with caution
- XAI crucial for clinical trust and regulatory approval

---

## 17. Knowledge Distillation

### 17.1 Foundational Papers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| Distilling Knowledge in a Neural Network | Hinton et al. | 2015 | NeurIPS |
| Model Compression | Bucila et al. | 2006 | KDD |

### 17.2 Overview Resources

| Resource | Link |
|----------|------|
| Knowledge Distillation Explained | [Neptune.ai](https://neptune.ai/blog/knowledge-distillation) |
| Intel Distiller Documentation | [Intel](https://intellabs.github.io/distiller/knowledge_distillation.html) |
| IBM Think - Knowledge Distillation | [IBM](https://www.ibm.com/think/topics/knowledge-distillation) |

### 17.3 Types of Knowledge Transfer

- **Response-based**: Transfer from final output layer (soft targets)
- **Feature-based**: Transfer intermediate layer activations
- **Relation-based**: Transfer relationships between samples

---

## 18. Data Standards

### 18.1 BIDS (Brain Imaging Data Structure)

| Resource | Type | Link |
|----------|------|------|
| BIDS Official Website | Specification | [bids.neuroimaging.io](https://bids.neuroimaging.io/index.html) |
| BIDS Specification | Documentation | [ReadTheDocs](https://bids-specification.readthedocs.io/) |
| EEG-BIDS Extension Paper | Publication | [Nature Scientific Data](https://www.nature.com/articles/s41597-019-0104-8) |
| iEEG-BIDS Extension | Publication | [Nature Scientific Data](https://www.nature.com/articles/s41597-019-0105-7) |
| BIDS GitHub | Code | [GitHub](https://github.com/bids-standard) |

### 18.2 HL7 FHIR

| Resource | Type | Link |
|----------|------|------|
| FHIR Official Site | Specification | [hl7.org/fhir](https://www.hl7.org/fhir/overview.html) |
| FHIR Foundation | Community | [fhir.org](https://www.fhir.org/) |
| HealthIT.gov FHIR Guide | Government | [HealthIT.gov](https://www.healthit.gov/topic/standards-technology/standards/fhir) |
| FHIR for Biomedical Signal Acquisition | Paper | [MDPI](https://www.mdpi.com/2076-3417/15/23/12803) |
| FHIR in Health Research: Systematic Review | Review | [PMC9346559](https://pmc.ncbi.nlm.nih.gov/articles/PMC9346559/) |

### 18.3 Other Formats

| Format | Description | Standard |
|--------|-------------|----------|
| **WFDB** | PhysioNet format | PhysioNet |
| **EDF/EDF+** | European Data Format | Kemp et al., 1992 |
| **GDF** | General Data Format | Schlögl, 2006 |
| **BDF** | BioSemi 24-bit format | BioSemi |
| **XDF** | Lab Streaming Layer | Kothe et al. |

---

## 19. Neuromorphic Hardware Platforms

### 19.1 Intel Loihi

| Resource | Type | Link |
|----------|------|------|
| Intel Neuromorphic Computing | Official | [Intel](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html) |
| Loihi Overview | Open Neuromorphic | [Open Neuromorphic](https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-intel/) |
| Loihi 2 Overview | Open Neuromorphic | [Open Neuromorphic](https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/) |
| Brian2Loihi Emulator | Paper | [Frontiers](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.1015624/full) |

**Specs (Loihi 2)**:
- 1 million neurons per chip
- Intel 4 process node
- Lava open-source software framework

**Hala Point System**:
- 1,152 Loihi 2 processors
- 1.15 billion neurons
- 128 billion synapses
- 100x more energy-efficient than CPU/GPU

### 19.2 SpiNNaker

| Resource | Type | Link |
|----------|------|------|
| SpiNNaker Project | Official | [Manchester](https://www.scieng.manchester.ac.uk/tomorrowlabs/spinnaker/) |
| Million-Core Announcement | News | [Manchester](https://www.manchester.ac.uk/about/news/human-brain-supercomputer-with-1million-processors-switched-on-for-first-time/) |
| SpiNNaker 2 Overview | Open Neuromorphic | [Open Neuromorphic](https://open-neuromorphic.org/neuromorphic-computing/hardware/spinnaker-2-university-of-dresden/) |

**Specs**:
- 1,036,800 ARM9 cores
- 7+ TB RAM
- 200 million million actions/second
- Part of Human Brain Project

### 19.3 BrainScaleS

| Resource | Type | Link |
|----------|------|------|
| BrainScaleS Project | Official | Heidelberg University |
| Human Brain Project | Collaboration | EU Flagship |

**Key Feature**: Analog neuromorphic computing with accelerated time scales (1000x faster than biological real-time)

### 19.4 Recent SNN Research *(NEW 2023-2025)*

| Paper | Authors | Year | Publication | Link |
|-------|---------|------|-------------|------|
| Exploring Neuromorphic Computing Based on SNNs: Algorithms to Hardware | Various | 2023 | ACM Computing Surveys | [ACM](https://dl.acm.org/doi/full/10.1145/3571155) |
| SNNs for Multimodal Neuroimaging: A Comprehensive Review | Various | 2025 | PMC | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12189790/) |
| Exploring SNNs in Biomedical Applications: Advantages, Limitations, Future | Various | 2024 | PMC | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362408/) |
| An Accurate and Fast Learning Approach in Biologically Inspired SNNs | Various | 2025 | Scientific Reports | [Nature](https://www.nature.com/articles/s41598-025-90113-0) |
| Low Cost Neuromorphic Learning Engine Based on High Performance Supervised SNN | Various | 2023 | Scientific Reports | [Nature](https://www.nature.com/articles/s41598-023-32120-7) |
| Optimal Mapping of SNNs to Neuromorphic Hardware for Edge-AI | Various | 2022 | PMC | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9572825/) |

**Biosignal-Specific SNN Applications**:
- 28.2 µW neuromorphic sensing for insertable cardiac monitoring
- NIMBLE: Neuromorphic memristor-based computing for EMG hand gesture recognition
- Adaptive graph convolution and LSTM SNNs for EEG-based BCIs (IEEE Trans NSRE 2023)
- NeuCube for multimodal neuroimaging analysis

---

## 20. Datasets & Benchmarks

### 20.1 ECG Databases

| Database | Description | Link |
|----------|-------------|------|
| MIT-BIH Arrhythmia | 48 half-hour recordings, arrhythmia annotations | [PhysioNet](https://physionet.org/content/mitdb/) |
| PTB-XL | 21,837 12-lead ECGs | [PhysioNet](https://physionet.org/content/ptb-xl/) |
| CPSC 2018 | China Physiological Signal Challenge | PhysioNet |

### 20.2 EEG Databases

| Database | Description | Link |
|----------|-------------|------|
| CHB-MIT | Pediatric seizure recordings | [PhysioNet](https://physionet.org/content/chbmit/) |
| Temple University Hospital (TUH) | Large clinical EEG corpus | TUH |
| TUSZ | Temple University Seizure Detection Corpus | TUH |
| BONN | Epilepsy benchmark dataset | University of Bonn |
| DEAP | Emotion analysis | Queen Mary |
| BCI Competition | Motor imagery datasets | Various |

### 20.3 Sleep Databases

| Database | Description | Link |
|----------|-------------|------|
| Sleep-EDF (SEDF13, SEDF18) | Most widely used sleep staging | PhysioNet |
| CAP-Sleep | Cyclic alternating pattern | PhysioNet |
| SHHS | Sleep Heart Health Study | NSRR |
| NSRR | National Sleep Research Resource | [NSRR](https://sleepdata.org/) |

### 20.4 Multimodal & Stress

| Database | Description | Link |
|----------|-------------|------|
| WESAD | Wearable stress and affect (Empatica E4) | UCI |
| SWELL | Stress and workload | Various |
| DREAMER | Emotion recognition | Various |

---

## 21. Foundational Textbooks

### Computational Neuroscience

| Book | Authors | Publisher |
|------|---------|-----------|
| Theoretical Neuroscience | Dayan & Abbott | MIT Press |
| Neuronal Dynamics | Gerstner et al. | [Online Book](https://neuronaldynamics.epfl.ch/) |
| Principles of Computational Modelling in Neuroscience | Sterratt et al. | Cambridge |

### Spiking Neural Networks

| Book | Authors | Publisher |
|------|---------|-----------|
| Spiking Neuron Models | Gerstner & Kistler | Cambridge |
| Dynamical Systems in Neuroscience | Izhikevich | MIT Press |

### Signal Processing

| Book | Authors | Publisher |
|------|---------|-----------|
| Bioelectrical Signal Processing in Cardiac and Neurological Applications | Sörnmo & Laguna | Academic Press |
| A Wavelet Tour of Signal Processing | Mallat | Academic Press |

### Machine Learning

| Book | Authors | Publisher |
|------|---------|-----------|
| Deep Learning | Goodfellow, Bengio, Courville | MIT Press |
| Pattern Recognition and Machine Learning | Bishop | Springer |
| Interpretable Machine Learning | Molnar | [Online Book](https://christophm.github.io/interpretable-ml-book/) |

---

## Citation Guidelines

When citing DPB Framework implementations, please reference both:

1. The original algorithm papers listed above
2. The DPB Framework documentation

### BibTeX Template

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

This document is maintained as part of the DPB Framework. For corrections or additions, please submit a pull request.

**Last Updated**: December 2025
**Version**: 2.0.0

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | Dec 2025 | Added 7 new sections (8-14) covering 2022-2025 research: Transformers, TinyML, Clinical Detection, Sleep, Stress, Multimodal Fusion, Federated Learning |
| 1.0.0 | Dec 2025 | Initial comprehensive reference document |

---

*AuraSense - Delta-Predictive Biosensing Framework*
