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
8. [Model Calibration & Uncertainty](#8-model-calibration--uncertainty)
9. [Explainability & Interpretability](#9-explainability--interpretability)
10. [Knowledge Distillation](#10-knowledge-distillation)
11. [Data Standards](#11-data-standards)
12. [Neuromorphic Hardware Platforms](#12-neuromorphic-hardware-platforms)
13. [Datasets & Benchmarks](#13-datasets--benchmarks)
14. [Foundational Textbooks](#14-foundational-textbooks)

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

---

## 8. Model Calibration & Uncertainty

### 8.1 Temperature Scaling

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| On Calibration of Modern Neural Networks | Guo et al. | 2017 | ICML |
| Temperature Scaling Implementation | Pleiss | - | [GitHub](https://github.com/gpleiss/temperature_scaling) |
| Adaptive Temperature Scaling | Various | 2024 | [Springer](https://link.springer.com/article/10.1007/s00521-024-09505-4) |

**Key Concept**: Divides logits by a learned temperature parameter T > 0 to calibrate confidence estimates.

### 8.2 Platt Scaling

Binary classification calibration using logistic regression on outputs.

### 8.3 Isotonic Calibration

Non-parametric calibration using isotonic regression.

### 8.4 Uncertainty Quantification

| Method | Description |
|--------|-------------|
| **MC Dropout** | Monte Carlo sampling with dropout at inference |
| **Deep Ensembles** | Multiple models for epistemic uncertainty |
| **Expected Calibration Error (ECE)** | Primary calibration metric |
| **Brier Score** | Measures accuracy of probabilistic predictions |

---

## 9. Explainability & Interpretability

### 9.1 SHAP (SHapley Additive exPlanations)

| Resource | Type | Link |
|----------|------|------|
| SHAP Original Paper | Lundberg & Lee, 2017 | NeurIPS |
| SHAP Documentation | Official docs | [SHAP](https://shap.readthedocs.io/) |
| SHAP GitHub | Implementation | [GitHub](https://github.com/shap/shap) |
| Interpretable ML Book - SHAP Chapter | Tutorial | [Christoph Molnar](https://christophm.github.io/interpretable-ml-book/shap.html) |
| SHAP in Drug Development | Practical guide | [PMC11513550](https://pmc.ncbi.nlm.nih.gov/articles/PMC11513550/) |

### 9.2 Integrated Gradients

Attribution method that satisfies axioms of sensitivity and implementation invariance.

| Paper | Authors | Year | Publication |
|-------|---------|------|-------------|
| Axiomatic Attribution for Deep Networks | Sundararajan et al. | 2017 | ICML |

### 9.3 Attention Mechanisms

Temporal and spatial attention for interpretable predictions.

---

## 10. Knowledge Distillation

### 10.1 Foundational Papers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| Distilling Knowledge in a Neural Network | Hinton et al. | 2015 | NeurIPS |
| Model Compression | Bucila et al. | 2006 | KDD |

### 10.2 Overview Resources

| Resource | Link |
|----------|------|
| Knowledge Distillation Explained | [Neptune.ai](https://neptune.ai/blog/knowledge-distillation) |
| Intel Distiller Documentation | [Intel](https://intellabs.github.io/distiller/knowledge_distillation.html) |
| IBM Think - Knowledge Distillation | [IBM](https://www.ibm.com/think/topics/knowledge-distillation) |

### 10.3 Types of Knowledge Transfer

- **Response-based**: Transfer from final output layer (soft targets)
- **Feature-based**: Transfer intermediate layer activations
- **Relation-based**: Transfer relationships between samples

---

## 11. Data Standards

### 11.1 BIDS (Brain Imaging Data Structure)

| Resource | Type | Link |
|----------|------|------|
| BIDS Official Website | Specification | [bids.neuroimaging.io](https://bids.neuroimaging.io/index.html) |
| BIDS Specification | Documentation | [ReadTheDocs](https://bids-specification.readthedocs.io/) |
| EEG-BIDS Extension Paper | Publication | [Nature Scientific Data](https://www.nature.com/articles/s41597-019-0104-8) |
| iEEG-BIDS Extension | Publication | [Nature Scientific Data](https://www.nature.com/articles/s41597-019-0105-7) |
| BIDS GitHub | Code | [GitHub](https://github.com/bids-standard) |

### 11.2 HL7 FHIR

| Resource | Type | Link |
|----------|------|------|
| FHIR Official Site | Specification | [hl7.org/fhir](https://www.hl7.org/fhir/overview.html) |
| FHIR Foundation | Community | [fhir.org](https://www.fhir.org/) |
| HealthIT.gov FHIR Guide | Government | [HealthIT.gov](https://www.healthit.gov/topic/standards-technology/standards/fhir) |
| FHIR for Biomedical Signal Acquisition | Paper | [MDPI](https://www.mdpi.com/2076-3417/15/23/12803) |
| FHIR in Health Research: Systematic Review | Review | [PMC9346559](https://pmc.ncbi.nlm.nih.gov/articles/PMC9346559/) |

### 11.3 Other Formats

| Format | Description | Standard |
|--------|-------------|----------|
| **WFDB** | PhysioNet format | PhysioNet |
| **EDF/EDF+** | European Data Format | Kemp et al., 1992 |
| **GDF** | General Data Format | Schlögl, 2006 |
| **BDF** | BioSemi 24-bit format | BioSemi |
| **XDF** | Lab Streaming Layer | Kothe et al. |

---

## 12. Neuromorphic Hardware Platforms

### 12.1 Intel Loihi

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

### 12.2 SpiNNaker

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

### 12.3 BrainScaleS

| Resource | Type | Link |
|----------|------|------|
| BrainScaleS Project | Official | Heidelberg University |
| Human Brain Project | Collaboration | EU Flagship |

**Key Feature**: Analog neuromorphic computing with accelerated time scales (1000x faster than biological real-time)

---

## 13. Datasets & Benchmarks

### 13.1 ECG Databases

| Database | Description | Link |
|----------|-------------|------|
| MIT-BIH Arrhythmia | 48 half-hour recordings, arrhythmia annotations | [PhysioNet](https://physionet.org/content/mitdb/) |
| PTB-XL | 21,837 12-lead ECGs | [PhysioNet](https://physionet.org/content/ptb-xl/) |
| CPSC 2018 | China Physiological Signal Challenge | PhysioNet |

### 13.2 EEG Databases

| Database | Description | Link |
|----------|-------------|------|
| CHB-MIT | Pediatric seizure recordings | [PhysioNet](https://physionet.org/content/chbmit/) |
| Temple University Hospital | Large clinical EEG corpus | TUH |
| DEAP | Emotion analysis | Queen Mary |
| BCI Competition | Motor imagery datasets | Various |

### 13.3 Multimodal

| Database | Description | Link |
|----------|-------------|------|
| WESAD | Wearable stress and affect | UCI |
| DREAMER | Emotion recognition | Various |

---

## 14. Foundational Textbooks

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
  year = {2024},
  url = {https://github.com/dcharlot65-aurasense/Delta-Predictive-Biosensing}
}
```

---

## Updates

This document is maintained as part of the DPB Framework. For corrections or additions, please submit a pull request.

**Last Updated**: December 2024
**Version**: 1.0.0

---

*AuraSense - Delta-Predictive Biosensing Framework*
