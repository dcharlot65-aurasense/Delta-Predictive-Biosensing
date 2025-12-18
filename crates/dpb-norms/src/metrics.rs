//! Metric type definitions for normative database
//!
//! Defines all supported metrics across cognitive, motor, and physiological domains.

use serde::{Deserialize, Serialize};

/// Direction of metric interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricDirection {
    /// Higher values indicate better performance (e.g., accuracy)
    HigherIsBetter,
    /// Lower values indicate better performance (e.g., reaction time)
    LowerIsBetter,
}

/// Domain of assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricDomain {
    /// Cognitive assessments
    Cognitive,
    /// Motor and movement assessments
    Motor,
    /// Physiological measures
    Physiological,
    /// Balance and vestibular
    Balance,
    /// Pain and sensory
    Sensory,
    /// Sleep quality
    Sleep,
    /// Composite/integrated measures
    Composite,
    /// PPG/cardiovascular waveform
    Ppg,
    /// Electrodermal activity
    Eda,
    /// EEG brain activity
    Eeg,
    /// Eye tracking/oculomotor
    Eye,
    /// Voice/speech
    Voice,
    /// Vestibular function
    Vestibular,
}

/// All supported metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    // === Cognitive Metrics ===
    /// Simple reaction time (ms)
    SimpleReactionTime,
    /// Choice reaction time (ms)
    ChoiceReactionTime,
    /// Reaction time variability (CV)
    ReactionTimeVariability,
    /// N-back accuracy (%)
    NBackAccuracy,
    /// N-back d-prime
    NBackDPrime,
    /// Continuous performance test - omissions
    CptOmissions,
    /// Continuous performance test - commissions
    CptCommissions,
    /// Stroop interference (ms)
    StroopInterference,
    /// Trail Making A time (s)
    TrailMakingA,
    /// Trail Making B time (s)
    TrailMakingB,
    /// Trail Making B-A difference (s)
    TrailMakingBMinusA,
    /// Digit span forward
    DigitSpanForward,
    /// Digit span backward
    DigitSpanBackward,
    /// Verbal fluency (words/min)
    VerbalFluency,
    /// MoCA total score
    MocaTotal,

    // === Motor Metrics ===
    /// Gait velocity (m/s)
    GaitVelocity,
    /// Stride length (m)
    StrideLength,
    /// Stride time variability (CV)
    StrideTimeVariability,
    /// Double support time (%)
    DoubleSupportTime,
    /// Cadence (steps/min)
    Cadence,
    /// Timed Up and Go (s)
    TimedUpAndGo,
    /// Grip strength (kg)
    GripStrength,
    /// Rate of force development (N/s)
    RateOfForceDevelopment,
    /// Finger tapping frequency (Hz)
    TappingFrequency,
    /// Tapping variability (CV)
    TappingVariability,
    /// Tremor amplitude (mm)
    TremorAmplitude,
    /// Tremor frequency (Hz)
    TremorFrequency,
    /// UPDRS Motor Score
    UpdrsMotor,

    // === Balance Metrics ===
    /// Sway area (cm²)
    SwayArea,
    /// Sway path length (cm)
    SwayPathLength,
    /// Sway velocity (cm/s)
    SwayVelocity,
    /// Romberg quotient
    RombergQuotient,
    /// Berg Balance Scale
    BergBalanceScale,
    /// Limits of stability - reaction time (ms)
    LosReactionTime,
    /// Limits of stability - max excursion (%)
    LosMaxExcursion,
    /// Limits of stability - directional control (%)
    LosDirectionalControl,

    // === Physiological Metrics ===
    /// Heart rate (bpm)
    HeartRate,
    /// Heart rate variability - SDNN (ms)
    HrvSdnn,
    /// Heart rate variability - RMSSD (ms)
    HrvRmssd,
    /// Heart rate variability - pNN50 (%)
    HrvPnn50,
    /// Heart rate variability - LF/HF ratio
    HrvLfHf,
    /// Respiratory rate (breaths/min)
    RespiratoryRate,
    /// Oxygen saturation (%)
    SpO2,
    /// Blood pressure systolic (mmHg)
    BpSystolic,
    /// Blood pressure diastolic (mmHg)
    BpDiastolic,

    // === Pain/Sensory Metrics ===
    /// Pressure pain threshold (kPa)
    PressurePainThreshold,
    /// Pain tolerance (kPa)
    PainTolerance,
    /// CPM effect (%)
    CpmEffect,
    /// Vibration perception threshold (μm)
    VibrationThreshold,
    /// Joint position error (degrees)
    JointPositionError,

    // === Sleep Metrics ===
    /// Total sleep time (min)
    TotalSleepTime,
    /// Sleep efficiency (%)
    SleepEfficiency,
    /// Sleep onset latency (min)
    SleepOnsetLatency,
    /// Wake after sleep onset (min)
    WakeAfterSleepOnset,
    /// REM percentage (%)
    RemPercent,
    /// Deep sleep percentage (%)
    DeepSleepPercent,

    // === Composite Metrics ===
    /// Cognitive composite score
    CognitiveComposite,
    /// Motor composite score
    MotorComposite,
    /// Global function composite
    GlobalComposite,
    /// Frailty index
    FrailtyIndex,

    // === PPG Metrics ===
    /// Pulse Transit Time (ms)
    PulseTransitTime,
    /// Pulse Wave Velocity (m/s)
    PulseWaveVelocity,
    /// Augmentation Index (%)
    AugmentationIndex,
    /// Stiffness Index (m/s)
    StiffnessIndex,
    /// Perfusion Index (%)
    PerfusionIndex,
    /// Pulse Rate Variability SDNN (ms)
    PrvSdnn,

    // === EDA Metrics ===
    /// Skin Conductance Level (μS)
    SkinConductanceLevel,
    /// SCR Frequency (events/min)
    ScrFrequency,
    /// SCR Amplitude (μS)
    ScrAmplitude,
    /// Non-specific SCR count (events/5min)
    NsScrCount,
    /// EDA Recovery Time (s)
    EdaRecoveryTime,

    // === EEG Band Power Metrics ===
    /// Delta Power (0.5-4 Hz, μV²)
    EegDeltaPower,
    /// Theta Power (4-8 Hz, μV²)
    EegThetaPower,
    /// Alpha Power (8-13 Hz, μV²)
    EegAlphaPower,
    /// Beta Power (13-30 Hz, μV²)
    EegBetaPower,
    /// Gamma Power (30-100 Hz, μV²)
    EegGammaPower,
    /// Alpha/Theta Ratio
    EegAlphaThetaRatio,
    /// Alpha Asymmetry (frontal)
    EegAlphaAsymmetry,

    // === Eye Tracking Metrics ===
    /// Saccade Peak Velocity (°/s)
    SaccadePeakVelocity,
    /// Saccade Amplitude (°)
    SaccadeAmplitude,
    /// Saccade Latency (ms)
    SaccadeLatency,
    /// Fixation Duration (ms)
    FixationDuration,
    /// Fixation Count (per minute)
    FixationCount,
    /// Pupil Diameter (mm)
    PupilDiameter,
    /// Pupil Response Latency (ms)
    PupilResponseLatency,
    /// Smooth Pursuit Gain
    SmoothPursuitGain,
    /// Blink Rate (per minute)
    BlinkRate,

    // === Voice Metrics ===
    /// Fundamental Frequency F0 (Hz)
    VoiceF0,
    /// F0 Variability (semitones)
    VoiceF0Variability,
    /// Jitter (%)
    VoiceJitter,
    /// Shimmer (%)
    VoiceShimmer,
    /// Harmonics to Noise Ratio (dB)
    VoiceHnr,
    /// Speech Rate (syllables/s)
    SpeechRate,
    /// Voice Onset Time (ms)
    VoiceOnsetTime,
    /// Maximum Phonation Time (s)
    MaxPhonationTime,

    // === Vestibular Metrics ===
    /// VOR Gain
    VorGain,
    /// Canal Paresis (%)
    CanalParesis,
    /// DVA Score Loss (logMAR)
    DvaScoreLoss,
    /// Subjective Visual Vertical Error (°)
    SvvError,
    /// Head Impulse Gain
    HeadImpulseGain,
}

impl MetricType {
    /// Get the domain for this metric
    pub fn domain(&self) -> MetricDomain {
        match self {
            // Cognitive
            MetricType::SimpleReactionTime
            | MetricType::ChoiceReactionTime
            | MetricType::ReactionTimeVariability
            | MetricType::NBackAccuracy
            | MetricType::NBackDPrime
            | MetricType::CptOmissions
            | MetricType::CptCommissions
            | MetricType::StroopInterference
            | MetricType::TrailMakingA
            | MetricType::TrailMakingB
            | MetricType::TrailMakingBMinusA
            | MetricType::DigitSpanForward
            | MetricType::DigitSpanBackward
            | MetricType::VerbalFluency
            | MetricType::MocaTotal => MetricDomain::Cognitive,

            // Motor
            MetricType::GaitVelocity
            | MetricType::StrideLength
            | MetricType::StrideTimeVariability
            | MetricType::DoubleSupportTime
            | MetricType::Cadence
            | MetricType::TimedUpAndGo
            | MetricType::GripStrength
            | MetricType::RateOfForceDevelopment
            | MetricType::TappingFrequency
            | MetricType::TappingVariability
            | MetricType::TremorAmplitude
            | MetricType::TremorFrequency
            | MetricType::UpdrsMotor => MetricDomain::Motor,

            // Balance
            MetricType::SwayArea
            | MetricType::SwayPathLength
            | MetricType::SwayVelocity
            | MetricType::RombergQuotient
            | MetricType::BergBalanceScale
            | MetricType::LosReactionTime
            | MetricType::LosMaxExcursion
            | MetricType::LosDirectionalControl => MetricDomain::Balance,

            // Physiological
            MetricType::HeartRate
            | MetricType::HrvSdnn
            | MetricType::HrvRmssd
            | MetricType::HrvPnn50
            | MetricType::HrvLfHf
            | MetricType::RespiratoryRate
            | MetricType::SpO2
            | MetricType::BpSystolic
            | MetricType::BpDiastolic => MetricDomain::Physiological,

            // Sensory
            MetricType::PressurePainThreshold
            | MetricType::PainTolerance
            | MetricType::CpmEffect
            | MetricType::VibrationThreshold
            | MetricType::JointPositionError => MetricDomain::Sensory,

            // Sleep
            MetricType::TotalSleepTime
            | MetricType::SleepEfficiency
            | MetricType::SleepOnsetLatency
            | MetricType::WakeAfterSleepOnset
            | MetricType::RemPercent
            | MetricType::DeepSleepPercent => MetricDomain::Sleep,

            // Composite
            MetricType::CognitiveComposite
            | MetricType::MotorComposite
            | MetricType::GlobalComposite
            | MetricType::FrailtyIndex => MetricDomain::Composite,

            // PPG
            MetricType::PulseTransitTime
            | MetricType::PulseWaveVelocity
            | MetricType::AugmentationIndex
            | MetricType::StiffnessIndex
            | MetricType::PerfusionIndex
            | MetricType::PrvSdnn => MetricDomain::Ppg,

            // EDA
            MetricType::SkinConductanceLevel
            | MetricType::ScrFrequency
            | MetricType::ScrAmplitude
            | MetricType::NsScrCount
            | MetricType::EdaRecoveryTime => MetricDomain::Eda,

            // EEG
            MetricType::EegDeltaPower
            | MetricType::EegThetaPower
            | MetricType::EegAlphaPower
            | MetricType::EegBetaPower
            | MetricType::EegGammaPower
            | MetricType::EegAlphaThetaRatio
            | MetricType::EegAlphaAsymmetry => MetricDomain::Eeg,

            // Eye
            MetricType::SaccadePeakVelocity
            | MetricType::SaccadeAmplitude
            | MetricType::SaccadeLatency
            | MetricType::FixationDuration
            | MetricType::FixationCount
            | MetricType::PupilDiameter
            | MetricType::PupilResponseLatency
            | MetricType::SmoothPursuitGain
            | MetricType::BlinkRate => MetricDomain::Eye,

            // Voice
            MetricType::VoiceF0
            | MetricType::VoiceF0Variability
            | MetricType::VoiceJitter
            | MetricType::VoiceShimmer
            | MetricType::VoiceHnr
            | MetricType::SpeechRate
            | MetricType::VoiceOnsetTime
            | MetricType::MaxPhonationTime => MetricDomain::Voice,

            // Vestibular
            MetricType::VorGain
            | MetricType::CanalParesis
            | MetricType::DvaScoreLoss
            | MetricType::SvvError
            | MetricType::HeadImpulseGain => MetricDomain::Vestibular,
        }
    }

    /// Get the interpretation direction for this metric
    pub fn direction(&self) -> MetricDirection {
        match self {
            // Lower is better (times, errors, variability)
            MetricType::SimpleReactionTime
            | MetricType::ChoiceReactionTime
            | MetricType::ReactionTimeVariability
            | MetricType::CptOmissions
            | MetricType::CptCommissions
            | MetricType::StroopInterference
            | MetricType::TrailMakingA
            | MetricType::TrailMakingB
            | MetricType::TrailMakingBMinusA
            | MetricType::StrideTimeVariability
            | MetricType::DoubleSupportTime
            | MetricType::TimedUpAndGo
            | MetricType::TappingVariability
            | MetricType::TremorAmplitude
            | MetricType::TremorFrequency
            | MetricType::UpdrsMotor
            | MetricType::SwayArea
            | MetricType::SwayPathLength
            | MetricType::SwayVelocity
            | MetricType::RombergQuotient
            | MetricType::LosReactionTime
            | MetricType::HrvLfHf
            | MetricType::VibrationThreshold
            | MetricType::JointPositionError
            | MetricType::SleepOnsetLatency
            | MetricType::WakeAfterSleepOnset
            | MetricType::FrailtyIndex
            // PPG - lower is better
            | MetricType::PulseWaveVelocity
            | MetricType::AugmentationIndex
            | MetricType::StiffnessIndex
            // EDA - lower arousal (generally)
            | MetricType::ScrFrequency
            | MetricType::NsScrCount
            // Eye - lower latency/variability
            | MetricType::SaccadeLatency
            | MetricType::PupilResponseLatency
            // Voice - lower jitter/shimmer
            | MetricType::VoiceJitter
            | MetricType::VoiceShimmer
            | MetricType::VoiceOnsetTime
            // Vestibular - lower error/paresis
            | MetricType::CanalParesis
            | MetricType::DvaScoreLoss
            | MetricType::SvvError => MetricDirection::LowerIsBetter,

            // Higher is better (performance scores, physiological capacity)
            MetricType::NBackAccuracy
            | MetricType::NBackDPrime
            | MetricType::DigitSpanForward
            | MetricType::DigitSpanBackward
            | MetricType::VerbalFluency
            | MetricType::MocaTotal
            | MetricType::GaitVelocity
            | MetricType::StrideLength
            | MetricType::Cadence
            | MetricType::GripStrength
            | MetricType::RateOfForceDevelopment
            | MetricType::TappingFrequency
            | MetricType::BergBalanceScale
            | MetricType::LosMaxExcursion
            | MetricType::LosDirectionalControl
            | MetricType::HrvSdnn
            | MetricType::HrvRmssd
            | MetricType::HrvPnn50
            | MetricType::SpO2
            | MetricType::PressurePainThreshold
            | MetricType::PainTolerance
            | MetricType::CpmEffect
            | MetricType::TotalSleepTime
            | MetricType::SleepEfficiency
            | MetricType::RemPercent
            | MetricType::DeepSleepPercent
            | MetricType::CognitiveComposite
            | MetricType::MotorComposite
            | MetricType::GlobalComposite
            // PPG - higher is better
            | MetricType::PerfusionIndex
            | MetricType::PrvSdnn
            // EDA - context dependent but recovery is good
            | MetricType::EdaRecoveryTime
            // EEG - higher alpha is generally better
            | MetricType::EegAlphaPower
            | MetricType::EegAlphaThetaRatio
            // Eye - higher gain/velocity
            | MetricType::SaccadePeakVelocity
            | MetricType::SmoothPursuitGain
            // Voice - higher HNR, phonation
            | MetricType::VoiceHnr
            | MetricType::SpeechRate
            | MetricType::MaxPhonationTime
            // Vestibular - higher gain
            | MetricType::VorGain
            | MetricType::HeadImpulseGain => MetricDirection::HigherIsBetter,

            // Neutral (depends on context)
            MetricType::HeartRate
            | MetricType::RespiratoryRate
            | MetricType::BpSystolic
            | MetricType::BpDiastolic
            // PPG - pulse transit time context-dependent
            | MetricType::PulseTransitTime
            // EDA - SCL is arousal, context-dependent
            | MetricType::SkinConductanceLevel
            | MetricType::ScrAmplitude
            // EEG - band powers context-dependent
            | MetricType::EegDeltaPower
            | MetricType::EegThetaPower
            | MetricType::EegBetaPower
            | MetricType::EegGammaPower
            | MetricType::EegAlphaAsymmetry
            // Eye - context dependent
            | MetricType::SaccadeAmplitude
            | MetricType::FixationDuration
            | MetricType::FixationCount
            | MetricType::PupilDiameter
            | MetricType::BlinkRate
            // Voice - F0 is person-specific
            | MetricType::VoiceF0
            | MetricType::VoiceF0Variability => MetricDirection::LowerIsBetter, // Default to lower
        }
    }

    /// Get unit of measurement
    pub fn unit(&self) -> &'static str {
        match self {
            MetricType::SimpleReactionTime
            | MetricType::ChoiceReactionTime
            | MetricType::HrvSdnn
            | MetricType::HrvRmssd
            | MetricType::LosReactionTime => "ms",

            MetricType::TrailMakingA
            | MetricType::TrailMakingB
            | MetricType::TrailMakingBMinusA
            | MetricType::TimedUpAndGo => "s",

            MetricType::GaitVelocity => "m/s",
            MetricType::StrideLength => "m",
            MetricType::Cadence => "steps/min",
            MetricType::GripStrength => "kg",
            MetricType::RateOfForceDevelopment => "N/s",
            MetricType::TappingFrequency | MetricType::TremorFrequency => "Hz",
            MetricType::TremorAmplitude => "mm",

            MetricType::SwayArea => "cm²",
            MetricType::SwayPathLength => "cm",
            MetricType::SwayVelocity => "cm/s",

            MetricType::HeartRate => "bpm",
            MetricType::RespiratoryRate => "breaths/min",
            MetricType::BpSystolic | MetricType::BpDiastolic => "mmHg",

            MetricType::PressurePainThreshold | MetricType::PainTolerance => "kPa",
            MetricType::VibrationThreshold => "μm",
            MetricType::JointPositionError => "°",

            MetricType::TotalSleepTime
            | MetricType::SleepOnsetLatency
            | MetricType::WakeAfterSleepOnset => "min",

            MetricType::NBackAccuracy
            | MetricType::SpO2
            | MetricType::DoubleSupportTime
            | MetricType::CpmEffect
            | MetricType::SleepEfficiency
            | MetricType::RemPercent
            | MetricType::DeepSleepPercent
            | MetricType::HrvPnn50
            | MetricType::LosMaxExcursion
            | MetricType::LosDirectionalControl => "%",

            MetricType::ReactionTimeVariability
            | MetricType::StrideTimeVariability
            | MetricType::TappingVariability
            | MetricType::RombergQuotient
            | MetricType::HrvLfHf
            | MetricType::FrailtyIndex => "ratio",

            MetricType::NBackDPrime
            | MetricType::StroopInterference
            | MetricType::CptOmissions
            | MetricType::CptCommissions
            | MetricType::DigitSpanForward
            | MetricType::DigitSpanBackward
            | MetricType::VerbalFluency
            | MetricType::MocaTotal
            | MetricType::BergBalanceScale
            | MetricType::UpdrsMotor
            | MetricType::CognitiveComposite
            | MetricType::MotorComposite
            | MetricType::GlobalComposite => "score",

            // PPG
            MetricType::PulseTransitTime
            | MetricType::PrvSdnn => "ms",
            MetricType::PulseWaveVelocity
            | MetricType::StiffnessIndex => "m/s",
            MetricType::AugmentationIndex
            | MetricType::PerfusionIndex => "%",

            // EDA
            MetricType::SkinConductanceLevel
            | MetricType::ScrAmplitude => "μS",
            MetricType::ScrFrequency => "events/min",
            MetricType::NsScrCount => "events",
            MetricType::EdaRecoveryTime => "s",

            // EEG
            MetricType::EegDeltaPower
            | MetricType::EegThetaPower
            | MetricType::EegAlphaPower
            | MetricType::EegBetaPower
            | MetricType::EegGammaPower => "μV²",
            MetricType::EegAlphaThetaRatio => "ratio",
            MetricType::EegAlphaAsymmetry => "score",

            // Eye
            MetricType::SaccadePeakVelocity => "°/s",
            MetricType::SaccadeAmplitude
            | MetricType::SvvError => "°",
            MetricType::SaccadeLatency
            | MetricType::FixationDuration
            | MetricType::PupilResponseLatency => "ms",
            MetricType::FixationCount
            | MetricType::BlinkRate => "/min",
            MetricType::PupilDiameter => "mm",
            MetricType::SmoothPursuitGain
            | MetricType::VorGain
            | MetricType::HeadImpulseGain => "gain",

            // Voice
            MetricType::VoiceF0 => "Hz",
            MetricType::VoiceF0Variability => "semitones",
            MetricType::VoiceJitter
            | MetricType::VoiceShimmer
            | MetricType::CanalParesis => "%",
            MetricType::VoiceHnr => "dB",
            MetricType::SpeechRate => "syllables/s",
            MetricType::VoiceOnsetTime => "ms",
            MetricType::MaxPhonationTime => "s",

            // Vestibular
            MetricType::DvaScoreLoss => "logMAR",
        }
    }

    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            MetricType::SimpleReactionTime => "Simple Reaction Time",
            MetricType::ChoiceReactionTime => "Choice Reaction Time",
            MetricType::ReactionTimeVariability => "Reaction Time Variability",
            MetricType::NBackAccuracy => "N-Back Accuracy",
            MetricType::NBackDPrime => "N-Back d'",
            MetricType::CptOmissions => "CPT Omissions",
            MetricType::CptCommissions => "CPT Commissions",
            MetricType::StroopInterference => "Stroop Interference",
            MetricType::TrailMakingA => "Trail Making A",
            MetricType::TrailMakingB => "Trail Making B",
            MetricType::TrailMakingBMinusA => "Trail Making B-A",
            MetricType::DigitSpanForward => "Digit Span Forward",
            MetricType::DigitSpanBackward => "Digit Span Backward",
            MetricType::VerbalFluency => "Verbal Fluency",
            MetricType::MocaTotal => "MoCA Total",
            MetricType::GaitVelocity => "Gait Velocity",
            MetricType::StrideLength => "Stride Length",
            MetricType::StrideTimeVariability => "Stride Time Variability",
            MetricType::DoubleSupportTime => "Double Support Time",
            MetricType::Cadence => "Cadence",
            MetricType::TimedUpAndGo => "Timed Up and Go",
            MetricType::GripStrength => "Grip Strength",
            MetricType::RateOfForceDevelopment => "Rate of Force Development",
            MetricType::TappingFrequency => "Tapping Frequency",
            MetricType::TappingVariability => "Tapping Variability",
            MetricType::TremorAmplitude => "Tremor Amplitude",
            MetricType::TremorFrequency => "Tremor Frequency",
            MetricType::UpdrsMotor => "UPDRS Motor Score",
            MetricType::SwayArea => "Sway Area",
            MetricType::SwayPathLength => "Sway Path Length",
            MetricType::SwayVelocity => "Sway Velocity",
            MetricType::RombergQuotient => "Romberg Quotient",
            MetricType::BergBalanceScale => "Berg Balance Scale",
            MetricType::LosReactionTime => "LOS Reaction Time",
            MetricType::LosMaxExcursion => "LOS Max Excursion",
            MetricType::LosDirectionalControl => "LOS Directional Control",
            MetricType::HeartRate => "Heart Rate",
            MetricType::HrvSdnn => "HRV SDNN",
            MetricType::HrvRmssd => "HRV RMSSD",
            MetricType::HrvPnn50 => "HRV pNN50",
            MetricType::HrvLfHf => "HRV LF/HF Ratio",
            MetricType::RespiratoryRate => "Respiratory Rate",
            MetricType::SpO2 => "SpO2",
            MetricType::BpSystolic => "Systolic BP",
            MetricType::BpDiastolic => "Diastolic BP",
            MetricType::PressurePainThreshold => "Pressure Pain Threshold",
            MetricType::PainTolerance => "Pain Tolerance",
            MetricType::CpmEffect => "CPM Effect",
            MetricType::VibrationThreshold => "Vibration Threshold",
            MetricType::JointPositionError => "Joint Position Error",
            MetricType::TotalSleepTime => "Total Sleep Time",
            MetricType::SleepEfficiency => "Sleep Efficiency",
            MetricType::SleepOnsetLatency => "Sleep Onset Latency",
            MetricType::WakeAfterSleepOnset => "WASO",
            MetricType::RemPercent => "REM %",
            MetricType::DeepSleepPercent => "Deep Sleep %",
            MetricType::CognitiveComposite => "Cognitive Composite",
            MetricType::MotorComposite => "Motor Composite",
            MetricType::GlobalComposite => "Global Composite",
            MetricType::FrailtyIndex => "Frailty Index",
            // PPG
            MetricType::PulseTransitTime => "Pulse Transit Time",
            MetricType::PulseWaveVelocity => "Pulse Wave Velocity",
            MetricType::AugmentationIndex => "Augmentation Index",
            MetricType::StiffnessIndex => "Stiffness Index",
            MetricType::PerfusionIndex => "Perfusion Index",
            MetricType::PrvSdnn => "PRV SDNN",
            // EDA
            MetricType::SkinConductanceLevel => "Skin Conductance Level",
            MetricType::ScrFrequency => "SCR Frequency",
            MetricType::ScrAmplitude => "SCR Amplitude",
            MetricType::NsScrCount => "NS-SCR Count",
            MetricType::EdaRecoveryTime => "EDA Recovery Time",
            // EEG
            MetricType::EegDeltaPower => "EEG Delta Power",
            MetricType::EegThetaPower => "EEG Theta Power",
            MetricType::EegAlphaPower => "EEG Alpha Power",
            MetricType::EegBetaPower => "EEG Beta Power",
            MetricType::EegGammaPower => "EEG Gamma Power",
            MetricType::EegAlphaThetaRatio => "EEG Alpha/Theta Ratio",
            MetricType::EegAlphaAsymmetry => "EEG Alpha Asymmetry",
            // Eye
            MetricType::SaccadePeakVelocity => "Saccade Peak Velocity",
            MetricType::SaccadeAmplitude => "Saccade Amplitude",
            MetricType::SaccadeLatency => "Saccade Latency",
            MetricType::FixationDuration => "Fixation Duration",
            MetricType::FixationCount => "Fixation Count",
            MetricType::PupilDiameter => "Pupil Diameter",
            MetricType::PupilResponseLatency => "Pupil Response Latency",
            MetricType::SmoothPursuitGain => "Smooth Pursuit Gain",
            MetricType::BlinkRate => "Blink Rate",
            // Voice
            MetricType::VoiceF0 => "Voice F0",
            MetricType::VoiceF0Variability => "Voice F0 Variability",
            MetricType::VoiceJitter => "Voice Jitter",
            MetricType::VoiceShimmer => "Voice Shimmer",
            MetricType::VoiceHnr => "Voice HNR",
            MetricType::SpeechRate => "Speech Rate",
            MetricType::VoiceOnsetTime => "Voice Onset Time",
            MetricType::MaxPhonationTime => "Max Phonation Time",
            // Vestibular
            MetricType::VorGain => "VOR Gain",
            MetricType::CanalParesis => "Canal Paresis",
            MetricType::DvaScoreLoss => "DVA Score Loss",
            MetricType::SvvError => "SVV Error",
            MetricType::HeadImpulseGain => "Head Impulse Gain",
        }
    }

    /// Get all metrics in a domain
    pub fn by_domain(domain: MetricDomain) -> Vec<MetricType> {
        ALL_METRICS.iter()
            .filter(|m| m.domain() == domain)
            .copied()
            .collect()
    }

    /// Get all metrics
    pub fn all() -> &'static [MetricType] {
        &ALL_METRICS
    }
}

/// All metric types for iteration
const ALL_METRICS: [MetricType; 100] = [
    // Cognitive (15)
    MetricType::SimpleReactionTime,
    MetricType::ChoiceReactionTime,
    MetricType::ReactionTimeVariability,
    MetricType::NBackAccuracy,
    MetricType::NBackDPrime,
    MetricType::CptOmissions,
    MetricType::CptCommissions,
    MetricType::StroopInterference,
    MetricType::TrailMakingA,
    MetricType::TrailMakingB,
    MetricType::TrailMakingBMinusA,
    MetricType::DigitSpanForward,
    MetricType::DigitSpanBackward,
    MetricType::VerbalFluency,
    MetricType::MocaTotal,
    // Motor (13)
    MetricType::GaitVelocity,
    MetricType::StrideLength,
    MetricType::StrideTimeVariability,
    MetricType::DoubleSupportTime,
    MetricType::Cadence,
    MetricType::TimedUpAndGo,
    MetricType::GripStrength,
    MetricType::RateOfForceDevelopment,
    MetricType::TappingFrequency,
    MetricType::TappingVariability,
    MetricType::TremorAmplitude,
    MetricType::TremorFrequency,
    MetricType::UpdrsMotor,
    // Balance (8)
    MetricType::SwayArea,
    MetricType::SwayPathLength,
    MetricType::SwayVelocity,
    MetricType::RombergQuotient,
    MetricType::BergBalanceScale,
    MetricType::LosReactionTime,
    MetricType::LosMaxExcursion,
    MetricType::LosDirectionalControl,
    // Physiological (9)
    MetricType::HeartRate,
    MetricType::HrvSdnn,
    MetricType::HrvRmssd,
    MetricType::HrvPnn50,
    MetricType::HrvLfHf,
    MetricType::RespiratoryRate,
    MetricType::SpO2,
    MetricType::BpSystolic,
    MetricType::BpDiastolic,
    // Sensory (5)
    MetricType::PressurePainThreshold,
    MetricType::PainTolerance,
    MetricType::CpmEffect,
    MetricType::VibrationThreshold,
    MetricType::JointPositionError,
    // Sleep (6)
    MetricType::TotalSleepTime,
    MetricType::SleepEfficiency,
    MetricType::SleepOnsetLatency,
    MetricType::WakeAfterSleepOnset,
    MetricType::RemPercent,
    MetricType::DeepSleepPercent,
    // Composite (4)
    MetricType::CognitiveComposite,
    MetricType::MotorComposite,
    MetricType::GlobalComposite,
    MetricType::FrailtyIndex,
    // PPG (6)
    MetricType::PulseTransitTime,
    MetricType::PulseWaveVelocity,
    MetricType::AugmentationIndex,
    MetricType::StiffnessIndex,
    MetricType::PerfusionIndex,
    MetricType::PrvSdnn,
    // EDA (5)
    MetricType::SkinConductanceLevel,
    MetricType::ScrFrequency,
    MetricType::ScrAmplitude,
    MetricType::NsScrCount,
    MetricType::EdaRecoveryTime,
    // EEG (7)
    MetricType::EegDeltaPower,
    MetricType::EegThetaPower,
    MetricType::EegAlphaPower,
    MetricType::EegBetaPower,
    MetricType::EegGammaPower,
    MetricType::EegAlphaThetaRatio,
    MetricType::EegAlphaAsymmetry,
    // Eye (9)
    MetricType::SaccadePeakVelocity,
    MetricType::SaccadeAmplitude,
    MetricType::SaccadeLatency,
    MetricType::FixationDuration,
    MetricType::FixationCount,
    MetricType::PupilDiameter,
    MetricType::PupilResponseLatency,
    MetricType::SmoothPursuitGain,
    MetricType::BlinkRate,
    // Voice (8)
    MetricType::VoiceF0,
    MetricType::VoiceF0Variability,
    MetricType::VoiceJitter,
    MetricType::VoiceShimmer,
    MetricType::VoiceHnr,
    MetricType::SpeechRate,
    MetricType::VoiceOnsetTime,
    MetricType::MaxPhonationTime,
    // Vestibular (5)
    MetricType::VorGain,
    MetricType::CanalParesis,
    MetricType::DvaScoreLoss,
    MetricType::SvvError,
    MetricType::HeadImpulseGain,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_domains() {
        assert_eq!(MetricType::SimpleReactionTime.domain(), MetricDomain::Cognitive);
        assert_eq!(MetricType::GaitVelocity.domain(), MetricDomain::Motor);
        assert_eq!(MetricType::SwayArea.domain(), MetricDomain::Balance);
        assert_eq!(MetricType::HeartRate.domain(), MetricDomain::Physiological);
    }

    #[test]
    fn test_metric_directions() {
        // Lower is better
        assert_eq!(MetricType::SimpleReactionTime.direction(), MetricDirection::LowerIsBetter);
        assert_eq!(MetricType::TremorAmplitude.direction(), MetricDirection::LowerIsBetter);

        // Higher is better
        assert_eq!(MetricType::GaitVelocity.direction(), MetricDirection::HigherIsBetter);
        assert_eq!(MetricType::GripStrength.direction(), MetricDirection::HigherIsBetter);
    }

    #[test]
    fn test_metric_units() {
        assert_eq!(MetricType::SimpleReactionTime.unit(), "ms");
        assert_eq!(MetricType::GaitVelocity.unit(), "m/s");
        assert_eq!(MetricType::GripStrength.unit(), "kg");
    }

    #[test]
    fn test_metrics_by_domain() {
        let cognitive = MetricType::by_domain(MetricDomain::Cognitive);
        assert!(cognitive.contains(&MetricType::SimpleReactionTime));
        assert!(!cognitive.contains(&MetricType::GaitVelocity));
    }

    #[test]
    fn test_all_metrics() {
        let all = MetricType::all();
        assert_eq!(all.len(), 100);
    }

    #[test]
    fn test_new_domain_metrics() {
        // PPG domain
        assert_eq!(MetricType::PulseWaveVelocity.domain(), MetricDomain::Ppg);
        // EDA domain
        assert_eq!(MetricType::SkinConductanceLevel.domain(), MetricDomain::Eda);
        // EEG domain
        assert_eq!(MetricType::EegAlphaPower.domain(), MetricDomain::Eeg);
        // Eye domain
        assert_eq!(MetricType::SaccadePeakVelocity.domain(), MetricDomain::Eye);
        // Voice domain
        assert_eq!(MetricType::VoiceF0.domain(), MetricDomain::Voice);
        // Vestibular domain
        assert_eq!(MetricType::VorGain.domain(), MetricDomain::Vestibular);
    }
}
