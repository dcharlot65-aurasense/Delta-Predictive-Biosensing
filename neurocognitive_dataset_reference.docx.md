# **Public Datasets for Human Motion & Speech Dynamics**

*A curated reference for neurocognitive assessment, spatial awareness, and health-correlated performance research*

# **1\. Parkinson's Disease & Movement Disorders**

Datasets with clinical annotations (UPDRS, MDS-UPDRS) for tremor, gait, and motor function analysis.

| Dataset | Modalities | Health Correlation | Sample | Access |
| ----- | ----- | ----- | ----- | ----- |
| WearGait-PD (FDA) | IMU (13 sensors), insoles, video, walkway | PD gait, MDS-UPDRS, DBS status | PD patients \+ age-matched controls | [FDA RST Portal](https://cdrh-rst.fda.gov/weargait-pd-wearables-dataset-gait-parkinsons-disease-and-age-matched-controls) |
| CARE-PD | RGB video, optical MoCap | UPDRS-gait severity | Multi-site, 5+ clinical centers | NeurIPS 2025 (hal-05280110) |
| PhysioNet PD Gait | Vertical GRF, force sensors | Gait dynamics, stride variability | 93 PD \+ 73 controls | [physionet.org](https://physionet.org/content/?topic=parkinsons) |
| PADS | Smartwatch accelerometry | Interactive neuro assessments | PD \+ differential diagnoses \+ controls | PhysioNet |
| Multimodal FoG | Video \+ IMU | Freezing of gait events | PD patients during turning | Mendeley Data |
| PD@Home | Wrist gyroscope, video | Real-life tremor monitoring | 24 PD \+ 24 controls (extensively labeled) | Open-source |
| Daphnet FoG | Wearable accelerometers (legs, hip) | FoG during ADL | Lab \+ daily living tasks | [Mobilize Center](https://mobilize.stanford.edu/data/available-datasets/) |

# **2\. Stroke Rehabilitation**

Datasets for motor recovery tracking, compensation detection, and functional assessment.

| Dataset | Modalities | Health Correlation | Sample | Access |
| ----- | ----- | ----- | ----- | ----- |
| StrokeRehab | IMU \+ video features | Sub-second action primitives | 51 stroke \+ 20 healthy | [SimTK](https://simtk.org/projects/primseq) |
| StrokeVision-Bench | RGB video \+ 2D skeletal keypoints | Box & Block Test performance | 1,000 annotated videos | arXiv 2509.07994 |
| Toronto Rehab Stroke | Kinect RGB-D | Upper limb compensation | 10 healthy \+ 9 stroke | ACM Pervasive Health |
| KIMORE | RGB-D (Kinect) | Clinical scoring, motor dysfunction | 44 healthy \+ 34 motor dysfunction | On request |
| EEG Motor Imagery (Stroke) | 64-channel EEG | Left/right hand MI, BCI rehab | 50 acute stroke patients | [figshare](https://doi.org/10.6084/m9.figshare.21679035.v5) |

# **3\. Healthy Motion Baselines & Large-Scale Pretraining**

Normative datasets for model pretraining and establishing healthy baselines.

| Dataset | Modalities | Use Case | Scale | Access |
| ----- | ----- | ----- | ----- | ----- |
| Motion-X | 3D whole-body (SMPL-X), video | Expressive motion pretraining | 15.6M poses, 81.1K sequences | CC BY-NC-SA |
| Human3.6M | RGB \+ 3D MoCap | Pose estimation benchmark | 3.6M poses, 11 subjects | [vision.imar.ro](http://vision.imar.ro/human3.6m) |
| H3WB (WholeBody) | 133 keypoints (body+face+hands) | Whole-body pose estimation | 100K images | [GitHub](https://github.com/wholebody3d/wholebody3d) |
| NTU RGB+D | RGB \+ depth \+ skeleton | Action recognition | 56K videos, 60 actions, 40 subjects | Application required |
| MPII Human Pose | RGB video frames | 2D pose benchmark | 25K images, 40K people, 410 activities | [mpi-inf.mpg.de](http://human-pose.mpi-inf.mpg.de/) |
| KIT Whole-Body | MoCap \+ object interaction | Robotics/rehab motion | 234 subjects, 2925 experiments | KIT Database |
| Sit-to-Walk MoCap | MoCap, force plates, EMG, IMU | Healthy aging benchmark | 65 adults, 19-73 years | Open access |
| 3D Gait & Running | MoCap, GRF | Walking/running kinematics | 30 healthy young adults | Open access |
| AddBiomechanics | Scaled skeletons, joint dynamics, GRF | Physically-validated motion | Largest validated motion dataset | CC BY 4.0 |

# **4\. Speech Dynamics & Cognitive Health**

## **4.1 Alzheimer's & Dementia**

| Dataset | Modalities | Health Correlation | Sample | Access |
| ----- | ----- | ----- | ----- | ----- |
| DementiaBank Pitt | Audio \+ transcripts | AD/MCI/HC, Cookie Theft task | Multi-year longitudinal | [TalkBank](https://dementia.talkbank.org/) |
| ADReSS / ADReSSo | Enhanced audio \+ transcripts | AD classification, MMSE prediction | Balanced age/gender | DementiaBank membership |
| MultiConAD | Multilingual audio \+ transcripts | AD/MCI/HC (3-class) | 16 datasets, 4 languages | arXiv 2502.19208 |
| EWA-DB (Slovak) | Audio (vowels, DDK, naming) | AD/MCI/PD detection | 1,649 speakers | [Nature Sci Data](https://www.nature.com/articles/s41597-024-04171-6) |
| Bridge2AI-Voice | Audio \+ clinical metadata | Neuro, mood, respiratory, voice | 833 participants, diverse cohorts | Controlled access |

## **4.2 Depression & Mood Disorders**

| Dataset | Modalities | Health Correlation | Sample | Access |
| ----- | ----- | ----- | ----- | ----- |
| PDCH (Depression Consult) | Audio \+ transcripts | Depression severity (HAMD-17) | 100 clinical consultations, \~30 min each | Research access |
| Dem@Care | Audio \+ video \+ physiological | Dementia, multi-modal | Lab \+ home settings (Greece) | Controlled access |

# **5\. Eye Tracking & Visual Attention**

| Dataset | Modalities | Health Correlation | Sample | Access |
| ----- | ----- | ----- | ----- | ----- |
| Saliency4ASD | Eye tracking \+ images | ASD vs TD saliency patterns | Children with ASD \+ controls | IEEE ICME'19 Challenge |
| ASD Eye Movement | Fixation maps \+ scanpaths | ASD visual attention | 14 ASD \+ 14 TD, 300 images | [ACM MMSys](https://dl.acm.org/doi/10.1145/3304109.3325818) |
| EyeT4Empathy | Eye tracking \+ empathy scores | Attention, ADHD-relevant metrics | Gaze typing \+ questionnaire | [Nature Sci Data](https://www.nature.com/articles/s41597-022-01862-w) |

# **6\. Integration Strategy for NeuroPlay**

## **6.1 Recommended Pipeline**

1. **Pretraining:** Use Motion-X (15.6M poses) or Human3.6M for pose estimation backbone, KIT for object interaction patterns  
2. **Healthy Baselines:** Sit-to-Walk MoCap and 3D Gait datasets establish normative ranges across age groups  
3. **Clinical Fine-tuning:** WearGait-PD, StrokeRehab provide gold-standard clinical scores for correlation  
4. **Speech Biomarkers:** Bridge2AI-Voice has derived features (MFCCs, spectrograms) ready for WebRTC audio pipeline  
5. **Real-World Validation:** PD@Home multimodal captures naturalistic behavior closest to telehealth scenarios

## **6.2 Controller Input Correlation Targets**

For Joy-Con/DualSense IMU \+ haptic data correlation:

* **Tremor metrics:** PD@Home wrist gyroscope data, tremor frequency/power  
* **Motor control:** StrokeRehab action primitives (reach, transport, reposition)  
* **Reaction time:** StrokeVision-Bench Box & Block timing  
* **Coordination:** KIMORE bilateral coordination scores

## **6.3 Access Timeline Planning**

* **Immediate (open):** PhysioNet, Motion-X, MPII, H3WB, AddBiomechanics  
* **1-2 weeks (registration):** DementiaBank, SimTK, NTU RGB+D  
* **2-4 weeks (DUA):** WearGait-PD, Bridge2AI-Voice, CARE-PD

# **7\. Key Repositories & Toolkits**

| Repository | Focus | URL |
| ----- | ----- | ----- |
| PhysioNet | Physiological signals, gait, EEG | [physionet.org](https://physionet.org) |
| DementiaBank / TalkBank | Speech/language pathology | [talkbank.org](https://talkbank.org) |
| Mobilize Center (Stanford) | Movement/gait datasets | [mobilize.stanford.edu](https://mobilize.stanford.edu/data/available-datasets/) |
| SimTK | Biomechanics, rehabilitation | [simtk.org](https://simtk.org) |
| MMPose (OpenMMLab) | Unified pose estimation \+ datasets | [GitHub](https://github.com/open-mmlab/mmpose) |
| voice\_datasets (GitHub) | Community-curated speech datasets | [GitHub](https://github.com/jim-schwoebel/voice_datasets) |
| VisionMD | Open-source MDS-UPDRS video analysis | Nature npj PD 2025 |

*Document compiled for AuraSense Tech Corporation / NeuroPlay development.*  
*Last updated: January 2026\. Verify access requirements before use — licensing terms may change.*