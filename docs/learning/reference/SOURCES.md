# Scientific Sources and References

> **Delta-Predictive Biosensing (DPB) Framework**
>
> This document provides scientific justification for the DPB approach and comprehensive references for all algorithms and methods implemented in the framework.
>
> **All citations verified via:** PubMed/NCBI, arXiv, Semantic Scholar, CrossRef, OpenAlex

---

## Table of Contents

1. [Core Approach: Spiking Neural Networks](#1-core-approach-spiking-neural-networks)
2. [Neuromorphic Computing](#2-neuromorphic-computing)
3. [Neuron Models](#3-neuron-models)
4. [Learning Rules & Plasticity](#4-learning-rules--plasticity)
5. [Reservoir Computing](#5-reservoir-computing)
6. [Signal Processing Algorithms](#6-signal-processing-algorithms)
7. [Biosignal Analysis](#7-biosignal-analysis)
8. [Transformers in Biosignal Analysis](#8-transformers-in-biosignal-analysis)
9. [TinyML & Edge AI for Wearables](#9-tinyml--edge-ai-for-wearables)
10. [Deep Learning for Clinical Detection](#10-deep-learning-for-clinical-detection)
11. [Sleep Stage Classification](#11-sleep-stage-classification)
12. [Stress Detection & Mental Health](#12-stress-detection--mental-health)
13. [Multimodal Biosignal Fusion](#13-multimodal-biosignal-fusion)
14. [Federated Learning in Healthcare](#14-federated-learning-in-healthcare)
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

Spiking Neural Networks (SNNs) are the third generation of neural networks that use discrete spikes to transmit information, mimicking biological neural communication.

### 1.2 Key Review Papers

| Paper | Year | PMID/arXiv | Link |
|-------|------|------------|------|
| Spiking neural networks for EEG signal analysis: From theory to practice | 2025 | PMID: 41004906 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/41004906/) |
| Exploring the potential of SNNs in biomedical applications | 2024 | PMID: 39220036 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39220036/) |
| Efficient and generalizable cross-patient epileptic seizure detection through SNN | 2024 | PMID: 38268711 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38268711/) |
| Real-time sub-milliwatt epilepsy detection on SNN edge processor | 2024 | PMID: 39413626 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39413626/) |
| A neuromorphic SNN detects epileptic HFOs in scalp EEG | 2022 | PMID: 35110665 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35110665/) |
| Fractal Spiking Neural Network for EEG-Based Emotion Recognition | 2023 | PMID: 38088998 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38088998/) |

### 1.3 Surrogate Gradient Methods

| Paper | Authors | Year | arXiv | Link |
|-------|---------|------|-------|------|
| Surrogate Gradient Learning in Spiking Neural Networks | Neftci, Mostafa, Zenke | 2019 | 1901.09948 | [arXiv](https://arxiv.org/abs/1901.09948) |
| Elucidating the theoretical underpinnings of surrogate gradient learning | Various | 2024 | 2404.14964 | [arXiv](https://arxiv.org/abs/2404.14964) |
| Directly Training Temporal SNN with Sparse Surrogate Gradient | Various | 2024 | 2406.19645 | [arXiv](https://arxiv.org/abs/2406.19645) |
| Meta-learning SNNs with Surrogate Gradient Descent | Various | 2022 | 2201.10777 | [arXiv](https://arxiv.org/abs/2201.10777) |
| Stabilizing Direct Training of SNNs: MP-Init and TrSG | Various | 2025 | 2511.08708 | [arXiv](https://arxiv.org/abs/2511.08708) |

**Software**: [snnTorch](https://snntorch.readthedocs.io/) - Python library for SNN training with surrogate gradients

---

## 2. Neuromorphic Computing

### 2.1 Key Papers

| Paper | Year | PMID | Link |
|-------|------|------|------|
| Neuromorphic applications in medicine | 2023 | 37531951 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37531951/) |
| NeuroCARE: A generic neuromorphic edge computing framework for healthcare | 2023 | 36755733 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36755733/) |
| Neuromorphic algorithms for brain implants: a review | 2025 | 40292025 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40292025/) |
| Neuromorphic neuromodulation: Towards next-gen closed-loop neurostimulation | 2024 | 39554511 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39554511/) |
| Digital neuromorphic technology: current and future prospects | 2024 | 38577676 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38577676/) |
| Neural interface systems with on-device computing | 2021 | 34735990 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34735990/) |
| A survey on neuromorphic continual learning systems | 2023 | 37214407 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37214407/) |

---

## 3. Neuron Models

### 3.1 Leaky Integrate-and-Fire (LIF)

| Paper | Year | Source | Link |
|-------|------|--------|------|
| Generalized Leaky Integrate-and-Fire Neuron Model | 2014 | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/A-biological-plausible-Generalized-Leaky-neuron-Wang-Guo/42f4ac3c704a650f30ea65850b6859c7a5bda55c) |
| Leaky Integrate-and-Fire Laser Neuron for Ultrafast Cognitive Computing | 2013 | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/A-Leaky-Integrate-and-Fire-Laser-Neuron-for-Nahmias-Shastri/7df54ebe18e044d4761f4a2adda81ae3766240c6) |
| Tutorial: The LIF Neuron Model | - | Neuromatch | [Tutorial](https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html) |
| Integrate-and-Fire Models | - | EPFL | [Neuronal Dynamics](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) |

### 3.2 Izhikevich Model

| Paper | Authors | Year | Source | Link |
|-------|---------|------|--------|------|
| Simple Model of Spiking Neurons | Izhikevich | 2003 | IEEE TNNLS | [Semantic Scholar](https://www.semanticscholar.org/paper/Simple-model-of-spiking-neurons-Izhikevich/1474853a6ba097f5880501bbdd94fb782ef3e4d3) |
| Which Model to Use for Cortical Spiking Neurons? | Izhikevich | 2004 | Neural Computation | [PDF](https://www.izhikevich.org/publications/whichmod.pdf) |

### 3.3 Hodgkin-Huxley Model

| Resource | Link |
|----------|------|
| Wikipedia Overview | [Hodgkin-Huxley Model](https://en.wikipedia.org/wiki/Hodgkin–Huxley_model) |
| Nobel Prize Information | [Nobel 1963](https://www.nobelprize.org/prizes/medicine/1963/summary/) |

---

## 4. Learning Rules & Plasticity

### 4.1 Spike-Timing Dependent Plasticity (STDP)

| Paper | Year | PMID | Link |
|-------|------|------|------|
| STDP: A consequence of more fundamental learning rules | 2010 | 20725599 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/20725599/) |
| Temporal modulation of spike-timing-dependent plasticity | 2011 | 21423505 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/21423505/) |
| Unsupervised learning of visual features through STDP | 2007 | 17305422 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/17305422/) |
| Reinforcement learning based on STDP | 2008 | 18941775 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/18941775/) |
| STDP finds repeating patterns in continuous spike trains | 2008 | 18167538 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/18167538/) |
| Characterization of Generalizability of STDP Trained SNNs | 2021 | 34776837 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34776837/) |
| Unsupervised Learning by STDP in Phase Change Memory Synapses | 2016 | 27013934 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/27013934/) |

---

## 5. Reservoir Computing

### 5.1 Echo State Networks & Liquid State Machines

| Paper | Year | PMID | Link |
|-------|------|------|------|
| Neuromemristive Reservoir Computing for Biosignal Processing | 2016 | 26869876 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/26869876/) |
| Reservoir computing with organic electrochemical networks for biosignal classification | 2021 | 34407948 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34407948/) |
| Experimental unification of reservoir computing methods | 2007 | 17517492 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/17517492/) |
| Recent advances in physical reservoir computing: A review | 2019 | 30981085 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/30981085/) |
| Detection of generalized synchronization using echo state networks | 2018 | 29604650 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/29604650/) |

**Key Applications**:
- 90% accuracy for epileptic seizure detection
- 84% accuracy for EMG prosthetic finger control
- 88% accuracy for arrhythmic heartbeat classification

---

## 6. Signal Processing Algorithms

### 6.1 Empirical Mode Decomposition (EMD)

| Paper | Year | PMID | Link |
|-------|------|------|------|
| EMD for automated detection of epilepsy using EEG | 2012 | 23186276 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/23186276/) |
| Assignment of EMD Components in Biomedical Signals | 2015 | 26419400 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/26419400/) |
| EMD for feature extraction in mental task classification | 2009 | 19965216 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/19965216/) |
| Noise-assisted EMD in biomedical signals | 2010 | 21075730 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/21075730/) |
| Arrhythmia ECG noise reduction by EEMD | 2012 | 22219702 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/22219702/) |
| EEMD for high frequency ECG noise reduction | 2010 | 20569049 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/20569049/) |
| EMD-based ECG enhancement and QRS detection | 2011 | 22119222 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/22119222/) |
| EMDLAB: Toolbox for EEG analysis using EMD | 2015 | 26162614 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/26162614/) |

**Variants Implemented**: EMD, EEMD (Ensemble EMD), CEEMDAN, VMD

**Software**: [emd](https://pypi.org/project/emd/) - Python package for EMD and Hilbert-Huang analysis

---

## 7. Biosignal Analysis

### 7.1 ECG Analysis - Pan-Tompkins Algorithm

| Paper | Authors | Year | PMID | Link |
|-------|---------|------|------|------|
| A real-time QRS detection algorithm | Pan & Tompkins | 1985 | 3997178 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/3997178/) |
| Mobile Platform-Based Real-Time QRS Detection (AMPT) | Various | 2023 | 36772665 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36772665/) |
| Combining algorithms in automatic QRS detection | Various | 2006 | 16871713 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/16871713/) |

**Performance**: 99.3% QRS detection accuracy on MIT-BIH Arrhythmia Database

### 7.2 Heart Rate Variability (HRV)

| Paper | Year | PMID | Link |
|-------|------|------|------|
| HRV Standards: Task Force ESC/NASPE | 1996 | 8598068 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/8598068/) |
| Publication guidelines for HR and HRV (Updated) | 2024 | 38873876 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38873876/) |
| HRV in occupational medicine guidelines | 2024 | 38741189 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38741189/) |
| GRAPH guidelines for psychiatry and HRV | 2016 | 27163204 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/27163204/) |
| HRV Metrics and Norms Overview | 2017 | 29034226 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/29034226/) |
| Reference values for HRV measures | 2016 | 26883166 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/26883166/) |
| HRVB Systematic Review and Guidelines | 2023 | 36917418 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36917418/) |

**Key Metrics**: SDNN, RMSSD, pNN50, LF/HF Ratio

---

## 8. Transformers in Biosignal Analysis

### 8.1 Foundation Models for Biosignals

| Model/Paper | Year | arXiv | Link |
|-------------|------|-------|------|
| BIOT: Cross-data Biosignal Learning in the Wild | 2023 | 2305.10351 | [arXiv](https://arxiv.org/abs/2305.10351) |
| Large Cognition Model: EEG Foundation Model | 2025 | 2502.17464 | [arXiv](https://arxiv.org/html/2502.17464v1) |
| Finetuning EEG Models on ECG/PPG for BP Estimation | 2025 | 2502.17460 | [arXiv](https://arxiv.org/html/2502.17460v1) |
| PhysioWave: Multi-Scale Wavelet-Transformer | 2025 | 2506.10351 | [arXiv](https://arxiv.org/html/2506.10351) |
| Transformer-based EEG Decoding: A Survey | 2025 | 2507.02320 | [arXiv](https://arxiv.org/html/2507.02320v1) |
| Vision Transformers for ECG Emotion Classification | 2025 | 2510.05826 | [arXiv](https://arxiv.org/html/2510.05826) |
| BrainPro: Large-scale Brain State-aware EEG Representation | 2025 | 2509.22050 | [arXiv](https://arxiv.org/html/2509.22050v1) |

---

## 9. TinyML & Edge AI for Wearables

### 9.1 Key Papers

| Paper | Year | PMID | Link |
|-------|------|------|------|
| TinyML: Enabling Deep Learning on Ultra-Low-Power IoT | 2022 | 35744466 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35744466/) |
| TinyML-Based Lightweight AI Healthcare Chatbot | 2024 | 39539515 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39539515/) |
| Efficient HAR on Edge Devices Using DeepConv LSTM | 2025 | 40263516 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40263516/) |
| Optimising TinyML with Quantization and Distillation | 2025 | 40128553 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40128553/) |
| TinyML and On-Device Inference: A Survey | 2025 | 40431982 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40431982/) |
| LPWAN and Embedded ML for Wearables | 2021 | 34372455 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34372455/) |

**Key Statistics**:
- TinyML market: $1.13 billion (2024), projected $4.6 billion (2033)
- Deployment possible on devices with as little as 64 KB RAM
- DeepConv LSTM: 98.24% accuracy on Arduino Nano 33 BLE

**Key Resource**: [Edge AI Foundation / tinyML](https://www.tinyml.org/)

---

## 10. Deep Learning for Clinical Detection

### 10.1 Arrhythmia Detection

| Paper | Year | PMID | Link |
|-------|------|------|------|
| Ensemble learning and fusion for arrhythmia detection | 2024 | 38553158 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38553158/) |
| CLINet: CNN + LSTM + Involution for ECG | 2024 | 38306814 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38306814/) |
| Explainable deep learning for arrhythmia detection | 2025 | 41219304 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/41219304/) |
| Automated diagnosis using CNN and LSTM | 2018 | 29903630 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/29903630/) |
| Deep CNN for arrhythmia detection (17 classes) | 2018 | 30245122 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/30245122/) |
| Hybrid 2D-CNN-LSTM for ECG classification | 2022 | 35447712 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35447712/) |
| A transformer-based DNN for arrhythmia detection | 2022 | 35227968 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35227968/) |

**Performance**: Up to 99.74% accuracy on MIT-BIH, PTB-XL, NST databases

### 10.2 Seizure Detection & Prediction

| Paper | Year | PMID | Link |
|-------|------|------|------|
| Supervised and Unsupervised DL for EEG Seizure Prediction | 2024 | 38681760 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38681760/) |
| Calibrating DL Classifiers for Patient-Independent Seizure Forecasting | 2024 | 38732969 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38732969/) |
| CNN-BiLSTM for Ultra-Long-Term Subcutaneous EEG | 2024 | 41056137 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/41056137/) |
| Biomimetic DL Networks for Epileptic Spasms | 2024 | 37851549 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37851549/) |
| Deep-learning-based seizure detection and prediction | 2022 | 35077027 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35077027/) |
| Hybrid DL Approach for Epileptic Seizure Detection | 2023 | 37037252 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37037252/) |
| LSTM and BiLSTM for Epileptic Seizure Detection | 2025 | 40255197 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40255197/) |

**Performance**: CNN-BiLSTM achieved AUROC 0.98, 94% sensitivity with 1.11 FP/day

---

## 11. Sleep Stage Classification

### 11.1 Key Papers

| Paper | Year | PMID | Link |
|-------|------|------|------|
| PPG-Based Sleep Staging Using SleepPPGNet | 2024 | 40038927 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40038927/) |
| Attention-Based DL for Single-Channel EEG Sleep Staging | 2021 | 33909566 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/33909566/) |
| Deep Transfer Learning for Single-Channel EEG | 2022 | 36433422 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36433422/) |
| Deep learning for PPG sleep staging in OSA patients | 2020 | 32436942 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32436942/) |
| Deep Learning with EEG Spectrogram | 2022 | 35627856 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35627856/) |
| Multi-Class Automatic Sleep Staging from PPG | 2021 | 33477468 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/33477468/) |
| Flexible DL for Sleep Stage Classification (ACC + PPG) | 2022 | 35786544 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35786544/) |

**Key Datasets**: Sleep-EDF, CAP-Sleep, SHHS, NSRR

---

## 12. Stress Detection & Mental Health

### 12.1 Key Papers

| Paper | Year | PMID | Link |
|-------|------|------|------|
| HRV Features for Stress Detection Using Wearables | 2021 | 33921884 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/33921884/) |
| Reliable Stress Recognition Using HRV Features | 2024 | 38794064 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38794064/) |
| Stress Detection Through Wrist-Based EDA | 2023 | 37022004 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37022004/) |
| EDA and HRV for Physiological Stress Response | 2024 | 38762532 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38762532/) |
| Comparison of ECG and PPG for Stress Detection | 2022 | 36086412 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36086412/) |
| Generalizable ML for Stress Monitoring: Systematic Review | 2023 | 36893657 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36893657/) |
| Cross Dataset Analysis for HRV-Based Stress Detection | 2023 | 36850407 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36850407/) |
| Deep SVMs for Stress Identification from EDA | 2020 | 32507059 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/32507059/) |

**Key Datasets**: WESAD (Empatica E4), SWELL

---

## 13. Multimodal Biosignal Fusion

### 13.1 Key Papers

| Paper | Year | arXiv/PMID | Link |
|-------|------|------------|------|
| BIOT: Cross-data Biosignal Learning | 2023 | 2305.10351 | [arXiv](https://arxiv.org/abs/2305.10351) |
| Finetuning EEG Foundation Models on ECG/PPG | 2025 | 2502.17460 | [arXiv](https://arxiv.org/html/2502.17460v1) |
| PhysioWave: Multi-Scale Wavelet-Transformer | 2025 | 2506.10351 | [arXiv](https://arxiv.org/html/2506.10351) |
| Diffusion-Augmented Contrastive Learning for Biosignals | 2025 | 2509.20048 | [arXiv](https://arxiv.org/html/2509.20048) |

### 13.2 Fusion Strategies

| Strategy | Description |
|----------|-------------|
| **Early Fusion** | Concatenate raw signals |
| **Late Fusion** | Combine model outputs |
| **Intermediate Fusion** | Fuse learned features (cross-attention) |
| **Graph-Based** | Model signal relationships with GNNs |

---

## 14. Federated Learning in Healthcare

### 14.1 Key Papers

| Paper | Year | PMID | Link |
|-------|------|------|------|
| Privacy preservation for federated learning in health care | 2024 | 39081567 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39081567/) |
| FL in Smart Healthcare: Privacy, Security, and Predictive Analytics | 2024 | 39766014 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39766014/) |
| Preserving privacy in big data research: FL in spine surgery | 2024 | 38403832 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38403832/) |
| Privacy-preserving FL for collaborative medical data mining | 2025 | 40217112 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40217112/) |
| Federated Security for Privacy Preservation in Edge-Cloud | 2025 | 40871971 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40871971/) |
| Federated Learning in Healthcare: A Privacy Preserving Approach | 2022 | 35612055 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35612055/) |
| Robust and Privacy-Preserving Decentralized Deep FL | 2023 | 37028039 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37028039/) |

### 14.2 Privacy-Preserving Techniques

| Technique | Description |
|-----------|-------------|
| **Differential Privacy** | Adds noise to gradients before sharing |
| **Secure Multi-Party Computation** | Cryptographic protocols for secure aggregation |
| **Homomorphic Encryption** | Computation on encrypted data |

---

## 15. Model Calibration & Uncertainty

### 15.1 Temperature Scaling

| Paper | Authors | Year | Publication |
|-------|---------|------|-------------|
| On Calibration of Modern Neural Networks | Guo et al. | 2017 | ICML |

**Implementation**: [GitHub - temperature_scaling](https://github.com/gpleiss/temperature_scaling)

### 15.2 Uncertainty Quantification Methods

| Method | Description |
|--------|-------------|
| **MC Dropout** | Monte Carlo sampling with dropout at inference |
| **Deep Ensembles** | Multiple models for epistemic uncertainty |
| **Expected Calibration Error (ECE)** | Primary calibration metric |
| **Brier Score** | Measures accuracy of probabilistic predictions |

---

## 16. Explainability & Interpretability

### 16.1 SHAP & LIME

| Paper | Year | PMID | Link |
|-------|------|------|------|
| LIME and SHAP in Alzheimer's Disease Detection | 2024 | 38578524 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/38578524/) |
| Explainable AI for Retinoblastoma Diagnosis | 2023 | 37296784 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37296784/) |
| Randomized Explainable ML for Medical Diagnosis | 2024 | 40030196 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40030196/) |
| The enlightening role of XAI in medical domains: SLR | 2023 | 37806061 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37806061/) |
| XAI for COVID-19 Gene Biomarkers | 2023 | 36738712 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/36738712/) |
| SHAP and LIME in Nasopharyngeal Cancer Survival | 2023 | 37268685 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37268685/) |
| Applications of XAI in Diagnosis and Surgery | 2022 | 35204328 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35204328/) |

**Software**: [SHAP](https://shap.readthedocs.io/) | [LIME](https://github.com/marcotcr/lime)

---

## 17. Knowledge Distillation

### 17.1 Foundational Papers

| Paper | Authors | Year | arXiv | Link |
|-------|---------|------|-------|------|
| Distilling the Knowledge in a Neural Network | Hinton et al. | 2015 | 1503.02531 | [arXiv](https://arxiv.org/abs/1503.02531) |
| Knowledge Distillation: A Survey | Gou et al. | 2021 | 2006.05525 | [arXiv](https://arxiv.org/abs/2006.05525) |
| KD with Integrated Gradients for NN Compression | Various | 2025 | 2503.13008 | [arXiv](https://arxiv.org/abs/2503.13008) |
| Few Sample Knowledge Distillation | Various | 2020 | 1812.01839 | [arXiv](https://arxiv.org/abs/1812.01839) |
| Data-Free Knowledge Distillation | Various | 2017 | 1710.07535 | [arXiv](https://arxiv.org/abs/1710.07535) |
| Knowledge Distillation Beyond Model Compression | Various | 2020 | 2007.01922 | [arXiv](https://arxiv.org/abs/2007.01922) |

### 17.2 Types of Knowledge Transfer

- **Response-based**: Transfer from final output layer (soft targets)
- **Feature-based**: Transfer intermediate layer activations
- **Relation-based**: Transfer relationships between samples

---

## 18. Data Standards

### 18.1 BIDS (Brain Imaging Data Structure)

| Paper | Year | PMID | Link |
|-------|------|------|------|
| EEG-BIDS: Extension to BIDS for EEG | 2019 | 31239435 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/31239435/) |
| iEEG-BIDS: Extension for intracranial EEG | 2019 | 31239438 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/31239438/) |
| The past, present, and future of BIDS | 2024 | 39308505 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/39308505/) |
| MNE-BIDS: Organizing electrophysiological data | 2022 | 35990374 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/35990374/) |
| MEG-BIDS: Extension for MEG | 2018 | 29917016 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/29917016/) |

**Official Site**: [bids.neuroimaging.io](https://bids.neuroimaging.io/)

### 18.2 HL7 FHIR

| Resource | Link |
|----------|------|
| FHIR Official Site | [hl7.org/fhir](https://www.hl7.org/fhir/overview.html) |
| FHIR Foundation | [fhir.org](https://www.fhir.org/) |

### 18.3 Other Formats

| Format | Description |
|--------|-------------|
| **WFDB** | PhysioNet format |
| **EDF/EDF+** | European Data Format |
| **GDF** | General Data Format |
| **BDF** | BioSemi 24-bit format |
| **XDF** | Lab Streaming Layer |

---

## 19. Neuromorphic Hardware Platforms

### 19.1 Intel Loihi

| Paper | Source | Link |
|-------|--------|------|
| Loihi: A Neuromorphic Manycore Processor with On-Chip Learning | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/Loihi:-A-Neuromorphic-Manycore-Processor-with-Davies-Srinivasa/ccc560f932e36b8561a847e02fcf20522bc71c5f) |
| Advancing Neuromorphic Computing With Loihi: A Survey | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/Advancing-Neuromorphic-Computing-With-Loihi:-A-of-Davies-Wild/5e68382b320413fc7672ed019eb0a807baf76f1e) |
| Programming SNNs on Intel's Loihi | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/Programming-Spiking-Neural-Networks-on-Intel%E2%80%99s-Lin-Wild/ca9cf0ca19dc3da09a5bd383cbbd7ad61b08b7ef) |
| NxTF: API and Compiler for Deep SNNs on Loihi | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/NxTF:-An-API-and-Compiler-for-Deep-Spiking-Neural-Rueckauer-Bybee/65053244f5e743552b1c00a5f6d106fea154cf90) |
| Brian2Loihi: Emulator for Loihi | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/Brian2Loihi:-An-emulator-for-the-neuromorphic-chip-Michaelis-Lehr/9d5286ebec2f341f0f7b192e9c238d6074dd5472) |

**Specs (Loihi 2)**: 1 million neurons per chip, Intel 4 process node, Lava software framework

### 19.2 SpiNNaker

| Paper | Source | Link |
|-------|--------|------|
| SpiNNaker 2: 10 Million Core Processor System | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/SpiNNaker-2:-A-10-Million-Core-Processor-System-for-Mayr-Hoeppner/6570969b03f686bc1efb45632d0ef3e17fe98214) |
| SpiNNaker: 1-W 18-Core SoC for Neural Network Simulation | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/SpiNNaker:-A-1-W-18-Core-System-on-Chip-for-Neural-Painkras-Plana/ad2db2e0885b57ab22e01fca4b8b2e688399d8f8) |
| SpiNNaker 2 Processing Element Architecture | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/The-SpiNNaker-2-Processing-Element-Architecture-for-H%C3%B6ppner-Yan/19a634633cfaf15b4fd1498d1b7ece08b1f36a0e) |
| The SpiNNaker Project | Semantic Scholar | [Paper](https://www.semanticscholar.org/paper/The-SpiNNaker-Project-Furber-Galluppi/28ba0ed8526e4322d81602a1da21b2b43fbe9b10) |

**Specs**: 1,036,800 ARM9 cores, 7+ TB RAM, Part of Human Brain Project

### 19.3 BrainScaleS

| Resource | Description |
|----------|-------------|
| Heidelberg University | Official project |
| Human Brain Project | EU Flagship collaboration |

**Key Feature**: Analog neuromorphic computing with 1000x accelerated time scales

---

## 20. Datasets & Benchmarks

### 20.1 ECG Databases

| Database | Description | Link |
|----------|-------------|------|
| MIT-BIH Arrhythmia | 48 half-hour recordings | [PhysioNet](https://physionet.org/content/mitdb/) |
| PTB-XL | 21,837 12-lead ECGs | [PhysioNet](https://physionet.org/content/ptb-xl/) |
| CPSC 2018 | China Physiological Signal Challenge | PhysioNet |

### 20.2 EEG Databases

| Database | Description | Link |
|----------|-------------|------|
| CHB-MIT | Pediatric seizure recordings | [PhysioNet](https://physionet.org/content/chbmit/) |
| Temple University Hospital (TUH) | Large clinical EEG corpus | TUH |
| TUSZ | Temple University Seizure Detection | TUH |
| BONN | Epilepsy benchmark | University of Bonn |
| DEAP | Emotion analysis | Queen Mary |

### 20.3 Sleep Databases

| Database | Description | Link |
|----------|-------------|------|
| Sleep-EDF | Most widely used | [PhysioNet](https://physionet.org/) |
| SHHS | Sleep Heart Health Study | [NSRR](https://sleepdata.org/) |

### 20.4 Multimodal & Stress

| Database | Description |
|----------|-------------|
| WESAD | Wearable stress (Empatica E4) |
| SWELL | Stress and workload |

---

## 21. Foundational Textbooks

### Computational Neuroscience

| Book | Authors | Publisher |
|------|---------|-----------|
| Theoretical Neuroscience | Dayan & Abbott | MIT Press |
| Neuronal Dynamics | Gerstner et al. | [Online](https://neuronaldynamics.epfl.ch/) |

### Spiking Neural Networks

| Book | Authors | Publisher |
|------|---------|-----------|
| Spiking Neuron Models | Gerstner & Kistler | Cambridge |
| Dynamical Systems in Neuroscience | Izhikevich | MIT Press |

### Signal Processing

| Book | Authors | Publisher |
|------|---------|-----------|
| Bioelectrical Signal Processing | Sörnmo & Laguna | Academic Press |
| A Wavelet Tour of Signal Processing | Mallat | Academic Press |

### Machine Learning

| Book | Authors | Publisher |
|------|---------|-----------|
| Deep Learning | Goodfellow, Bengio, Courville | MIT Press |
| Pattern Recognition and Machine Learning | Bishop | Springer |
| Interpretable Machine Learning | Molnar | [Online](https://christophm.github.io/interpretable-ml-book/) |

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

## Verification Sources

All citations in this document were verified using:

| Source | URL | Coverage |
|--------|-----|----------|
| **PubMed/NCBI** | [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/) | Biomedical literature |
| **arXiv** | [arxiv.org](https://arxiv.org/) | Preprints (CS, ML, Physics) |
| **Semantic Scholar** | [semanticscholar.org](https://www.semanticscholar.org/) | AI-powered research tool |
| **OpenAlex** | [openalex.org](https://openalex.org/) | Open scholarly metadata |
| **CrossRef** | [crossref.org](https://www.crossref.org/) | DOI registration |

---

## Updates

**Last Updated**: December 2025
**Version**: 3.0.0

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.0.0 | Dec 2025 | Complete verification of all citations via PubMed, arXiv, Semantic Scholar |
| 2.0.0 | Dec 2025 | Added sections 8-14 covering 2022-2025 research |
| 1.0.0 | Dec 2025 | Initial comprehensive reference document |

---

*AuraSense - Delta-Predictive Biosensing Framework*
