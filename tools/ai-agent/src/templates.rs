//! Code Generation Templates
//!
//! Provides structured templates showing usage patterns that AI agents can follow
//! when generating code using the DPB framework.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Code template for a specific use case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTemplate {
    /// Template identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Category (e.g., "encoding", "training", "inference")
    pub category: String,
    /// Description of what this template does
    pub description: String,
    /// Required crates/dependencies
    pub dependencies: Vec<String>,
    /// Required imports
    pub imports: Vec<String>,
    /// Template code with placeholders
    pub code: String,
    /// Placeholder definitions
    pub placeholders: Vec<Placeholder>,
    /// Example with placeholders filled in
    pub example: String,
    /// Tags for searchability
    pub tags: Vec<String>,
}

/// Placeholder in a code template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Placeholder {
    /// Placeholder name (e.g., "ENCODER_TYPE")
    pub name: String,
    /// Description
    pub description: String,
    /// Type hint
    pub type_hint: String,
    /// Example value
    pub example: String,
    /// Possible values (if enumerated)
    pub options: Option<Vec<String>>,
}

/// Registry of all available templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateRegistry {
    /// Schema version
    pub version: String,
    /// All templates organized by category
    pub templates: HashMap<String, Vec<CodeTemplate>>,
}

impl TemplateRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            version: "1.0.0".to_string(),
            templates: HashMap::new(),
        }
    }

    /// Create the default registry with all built-in templates
    pub fn default_templates() -> Self {
        let mut registry = Self::new();

        // Add encoding templates
        registry.add_templates("encoding", encoding_templates());

        // Add training templates
        registry.add_templates("training", training_templates());

        // Add inference templates
        registry.add_templates("inference", inference_templates());

        // Add signal processing templates
        registry.add_templates("signal_processing", signal_processing_templates());

        // Add visualization templates
        registry.add_templates("visualization", visualization_templates());

        // Add pipeline templates
        registry.add_templates("pipeline", pipeline_templates());

        registry
    }

    /// Add templates to a category
    pub fn add_templates(&mut self, category: &str, templates: Vec<CodeTemplate>) {
        self.templates.insert(category.to_string(), templates);
    }

    /// Get all templates
    pub fn all_templates(&self) -> Vec<&CodeTemplate> {
        self.templates.values().flatten().collect()
    }

    /// Search templates by tag
    pub fn search_by_tag(&self, tag: &str) -> Vec<&CodeTemplate> {
        let tag_lower = tag.to_lowercase();
        self.all_templates()
            .into_iter()
            .filter(|t| t.tags.iter().any(|t| t.to_lowercase().contains(&tag_lower)))
            .collect()
    }

    /// Search templates by name or description
    pub fn search(&self, query: &str) -> Vec<&CodeTemplate> {
        let query_lower = query.to_lowercase();
        self.all_templates()
            .into_iter()
            .filter(|t| {
                t.name.to_lowercase().contains(&query_lower) ||
                t.description.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Get template by ID
    pub fn get(&self, id: &str) -> Option<&CodeTemplate> {
        self.all_templates().into_iter().find(|t| t.id == id)
    }

    /// Save to JSON file
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load from JSON file
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::default_templates()
    }
}

// ============================================================================
// Built-in Templates
// ============================================================================

fn encoding_templates() -> Vec<CodeTemplate> {
    vec![
        CodeTemplate {
            id: "basic_level_crossing".to_string(),
            name: "Basic Level Crossing Encoder".to_string(),
            category: "encoding".to_string(),
            description: "Encode a time series signal using level crossing detection".to_string(),
            dependencies: vec!["dpb-core".to_string(), "dpb-encoders".to_string()],
            imports: vec![
                "use dpb_core::{TimeSeries, SpikeTrain};".to_string(),
                "use dpb_encoders::base::LevelCrossingEncoder;".to_string(),
                "use dpb_core::traits::EventEncoder;".to_string(),
            ],
            code: r#"// Create the encoder with threshold
let encoder = LevelCrossingEncoder::new({{THRESHOLD}});

// Encode the signal
let spikes: SpikeTrain = encoder.encode(&{{SIGNAL}});

// Access spike times
for event in spikes.events() {
    println!("Spike at t={:.3}s, value={:.2}", event.time, event.value);
}"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "THRESHOLD".to_string(),
                    description: "Level crossing threshold value".to_string(),
                    type_hint: "f64".to_string(),
                    example: "0.1".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "SIGNAL".to_string(),
                    description: "Input time series signal".to_string(),
                    type_hint: "TimeSeries".to_string(),
                    example: "time_series".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_core::{TimeSeries, SpikeTrain};
use dpb_encoders::base::LevelCrossingEncoder;
use dpb_core::traits::EventEncoder;

// Create sample signal
let signal = TimeSeries::from_samples(&[0.0, 0.2, 0.5, 0.3, 0.1], 100.0);

// Create the encoder with threshold
let encoder = LevelCrossingEncoder::new(0.1);

// Encode the signal
let spikes: SpikeTrain = encoder.encode(&signal);

// Access spike times
for event in spikes.events() {
    println!("Spike at t={:.3}s, value={:.2}", event.time, event.value);
}"#.to_string(),
            tags: vec!["encoder".to_string(), "level-crossing".to_string(), "basic".to_string()],
        },
        CodeTemplate {
            id: "ecg_rpeak_encoding".to_string(),
            name: "ECG R-Peak Encoder".to_string(),
            category: "encoding".to_string(),
            description: "Detect and encode R-peaks from ECG signals".to_string(),
            dependencies: vec!["dpb-core".to_string(), "dpb-encoders".to_string()],
            imports: vec![
                "use dpb_core::{TimeSeries, SpikeTrain};".to_string(),
                "use dpb_encoders::contact::ecg::EcgRPeakEncoder;".to_string(),
                "use dpb_core::traits::EventEncoder;".to_string(),
            ],
            code: r#"// Create ECG R-peak encoder with sample rate
let encoder = EcgRPeakEncoder::new({{SAMPLE_RATE}});

// Encode ECG signal to R-peak spikes
let rpeak_spikes = encoder.encode(&{{ECG_SIGNAL}});

// Calculate heart rate from R-R intervals
let rr_intervals: Vec<f64> = rpeak_spikes.events()
    .windows(2)
    .map(|w| w[1].time - w[0].time)
    .collect();

let mean_rr = rr_intervals.iter().sum::<f64>() / rr_intervals.len() as f64;
let heart_rate_bpm = 60.0 / mean_rr;
println!("Heart rate: {:.1} BPM", heart_rate_bpm);"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "SAMPLE_RATE".to_string(),
                    description: "ECG signal sample rate in Hz".to_string(),
                    type_hint: "f64".to_string(),
                    example: "250.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "ECG_SIGNAL".to_string(),
                    description: "Input ECG time series".to_string(),
                    type_hint: "TimeSeries".to_string(),
                    example: "ecg_data".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_core::{TimeSeries, SpikeTrain};
use dpb_encoders::contact::ecg::EcgRPeakEncoder;
use dpb_core::traits::EventEncoder;
use dpb_synth::contact::EcgGenerator;

// Generate synthetic ECG
let generator = EcgGenerator::default();
let ecg_data = generator.generate(10.0, 250.0); // 10 seconds at 250 Hz

// Create ECG R-peak encoder
let encoder = EcgRPeakEncoder::new(250.0);

// Encode to spikes
let rpeak_spikes = encoder.encode(&ecg_data);

// Calculate heart rate
let rr_intervals: Vec<f64> = rpeak_spikes.events()
    .windows(2)
    .map(|w| w[1].time - w[0].time)
    .collect();

let mean_rr = rr_intervals.iter().sum::<f64>() / rr_intervals.len() as f64;
let heart_rate_bpm = 60.0 / mean_rr;
println!("Heart rate: {:.1} BPM", heart_rate_bpm);"#.to_string(),
            tags: vec!["encoder".to_string(), "ecg".to_string(), "cardiac".to_string(), "r-peak".to_string()],
        },
        CodeTemplate {
            id: "multimodal_encoding".to_string(),
            name: "Multi-Modal Signal Encoding".to_string(),
            category: "encoding".to_string(),
            description: "Encode multiple signal modalities (ECG, PPG, EMG) simultaneously".to_string(),
            dependencies: vec!["dpb-core".to_string(), "dpb-encoders".to_string()],
            imports: vec![
                "use dpb_core::{TimeSeries, SpikeTrain, Modality};".to_string(),
                "use dpb_encoders::contact::{ecg::EcgRPeakEncoder, ppg::PpgPulseEncoder, emg::EmgBurstEncoder};".to_string(),
                "use dpb_core::traits::EventEncoder;".to_string(),
                "use std::collections::HashMap;".to_string(),
            ],
            code: r#"// Create encoders for each modality
let ecg_encoder = EcgRPeakEncoder::new({{SAMPLE_RATE}});
let ppg_encoder = PpgPulseEncoder::new({{SAMPLE_RATE}});
let emg_encoder = EmgBurstEncoder::new({{SAMPLE_RATE}});

// Encode each signal
let mut spike_trains: HashMap<Modality, SpikeTrain> = HashMap::new();
spike_trains.insert(Modality::Ecg, ecg_encoder.encode(&{{ECG_SIGNAL}}));
spike_trains.insert(Modality::Ppg, ppg_encoder.encode(&{{PPG_SIGNAL}}));
spike_trains.insert(Modality::Emg, emg_encoder.encode(&{{EMG_SIGNAL}}));

// Process multimodal spikes
for (modality, spikes) in &spike_trains {
    println!("{:?}: {} spikes", modality, spikes.len());
}"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "SAMPLE_RATE".to_string(),
                    description: "Common sample rate for all signals".to_string(),
                    type_hint: "f64".to_string(),
                    example: "250.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "ECG_SIGNAL".to_string(),
                    description: "ECG time series".to_string(),
                    type_hint: "TimeSeries".to_string(),
                    example: "ecg".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "PPG_SIGNAL".to_string(),
                    description: "PPG time series".to_string(),
                    type_hint: "TimeSeries".to_string(),
                    example: "ppg".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "EMG_SIGNAL".to_string(),
                    description: "EMG time series".to_string(),
                    type_hint: "TimeSeries".to_string(),
                    example: "emg".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_core::{TimeSeries, SpikeTrain, Modality};
use dpb_encoders::contact::{ecg::EcgRPeakEncoder, ppg::PpgPulseEncoder, emg::EmgBurstEncoder};
use dpb_core::traits::EventEncoder;
use std::collections::HashMap;

// Create encoders
let ecg_encoder = EcgRPeakEncoder::new(250.0);
let ppg_encoder = PpgPulseEncoder::new(250.0);
let emg_encoder = EmgBurstEncoder::new(250.0);

// Encode signals (assuming signals are loaded)
let mut spike_trains: HashMap<Modality, SpikeTrain> = HashMap::new();
spike_trains.insert(Modality::Ecg, ecg_encoder.encode(&ecg));
spike_trains.insert(Modality::Ppg, ppg_encoder.encode(&ppg));
spike_trains.insert(Modality::Emg, emg_encoder.encode(&emg));

for (modality, spikes) in &spike_trains {
    println!("{:?}: {} spikes", modality, spikes.len());
}"#.to_string(),
            tags: vec!["encoder".to_string(), "multimodal".to_string(), "ecg".to_string(), "ppg".to_string(), "emg".to_string()],
        },
    ]
}

fn training_templates() -> Vec<CodeTemplate> {
    vec![
        CodeTemplate {
            id: "basic_snn_training".to_string(),
            name: "Basic SNN Training Loop".to_string(),
            category: "training".to_string(),
            description: "Train a feedforward SNN with BPTT and surrogate gradients".to_string(),
            dependencies: vec!["dpb-core".to_string(), "dpb-snn".to_string(), "dpb-neurons".to_string()],
            imports: vec![
                "use dpb_snn::architectures::FeedforwardSNN;".to_string(),
                "use dpb_snn::training::{BPTT, TrainingConfig};".to_string(),
                "use dpb_snn::loss::SpikingCrossEntropy;".to_string(),
                "use dpb_neurons::surrogate::FastSigmoid;".to_string(),
            ],
            code: r#"// Create SNN architecture
let mut snn = FeedforwardSNN::new(vec![
    {{INPUT_SIZE}},  // Input layer
    {{HIDDEN_SIZE}}, // Hidden layer
    {{OUTPUT_SIZE}}, // Output layer
]);

// Configure training
let config = TrainingConfig {
    learning_rate: {{LEARNING_RATE}},
    epochs: {{EPOCHS}},
    batch_size: {{BATCH_SIZE}},
    surrogate: FastSigmoid::new({{BETA}}),
    ..Default::default()
};

// Create trainer
let mut trainer = BPTT::new(config);
let loss_fn = SpikingCrossEntropy::new();

// Training loop
for epoch in 0..{{EPOCHS}} {
    let mut total_loss = 0.0;
    for (inputs, targets) in {{DATALOADER}}.batches() {
        let loss = trainer.train_step(&mut snn, &inputs, &targets, &loss_fn);
        total_loss += loss;
    }
    println!("Epoch {}: Loss = {:.4}", epoch, total_loss);
}"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "INPUT_SIZE".to_string(),
                    description: "Number of input neurons".to_string(),
                    type_hint: "usize".to_string(),
                    example: "784".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "HIDDEN_SIZE".to_string(),
                    description: "Number of hidden neurons".to_string(),
                    type_hint: "usize".to_string(),
                    example: "256".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "OUTPUT_SIZE".to_string(),
                    description: "Number of output neurons".to_string(),
                    type_hint: "usize".to_string(),
                    example: "10".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "LEARNING_RATE".to_string(),
                    description: "Learning rate".to_string(),
                    type_hint: "f64".to_string(),
                    example: "0.001".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "EPOCHS".to_string(),
                    description: "Number of training epochs".to_string(),
                    type_hint: "usize".to_string(),
                    example: "100".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "BATCH_SIZE".to_string(),
                    description: "Training batch size".to_string(),
                    type_hint: "usize".to_string(),
                    example: "32".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "BETA".to_string(),
                    description: "Surrogate gradient steepness".to_string(),
                    type_hint: "f64".to_string(),
                    example: "5.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "DATALOADER".to_string(),
                    description: "Data loader for training".to_string(),
                    type_hint: "DataLoader".to_string(),
                    example: "train_loader".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_snn::architectures::FeedforwardSNN;
use dpb_snn::training::{BPTT, TrainingConfig};
use dpb_snn::loss::SpikingCrossEntropy;
use dpb_neurons::surrogate::FastSigmoid;

// Create SNN
let mut snn = FeedforwardSNN::new(vec![784, 256, 10]);

// Configure training
let config = TrainingConfig {
    learning_rate: 0.001,
    epochs: 100,
    batch_size: 32,
    surrogate: FastSigmoid::new(5.0),
    ..Default::default()
};

let mut trainer = BPTT::new(config);
let loss_fn = SpikingCrossEntropy::new();

for epoch in 0..100 {
    let loss = trainer.train_epoch(&mut snn, &train_loader, &loss_fn);
    println!("Epoch {}: Loss = {:.4}", epoch, loss);
}"#.to_string(),
            tags: vec!["training".to_string(), "snn".to_string(), "bptt".to_string(), "feedforward".to_string()],
        },
        CodeTemplate {
            id: "knowledge_distillation".to_string(),
            name: "ANN to SNN Knowledge Distillation".to_string(),
            category: "training".to_string(),
            description: "Distill knowledge from a trained ANN to an SNN".to_string(),
            dependencies: vec!["dpb-snn".to_string()],
            imports: vec![
                "use dpb_snn::distillation::{TeacherStudentFramework, DistillationConfig, DistillationMode};".to_string(),
                "use dpb_snn::baselines::MLP;".to_string(),
                "use dpb_snn::architectures::FeedforwardSNN;".to_string(),
            ],
            code: r#"// Load pre-trained ANN teacher
let teacher = MLP::load({{TEACHER_PATH}})?;

// Create SNN student
let student = FeedforwardSNN::new(vec![{{LAYER_SIZES}}]);

// Configure distillation
let config = DistillationConfig {
    mode: DistillationMode::SpikePattern,
    temperature: {{TEMPERATURE}},
    alpha: {{ALPHA}}, // Weight for distillation loss
    epochs: {{EPOCHS}},
    ..Default::default()
};

// Create framework and distill
let mut framework = TeacherStudentFramework::new(teacher, student, config);
let distilled_snn = framework.distill(&{{DATALOADER}})?;

println!("Distillation complete. Student accuracy: {:.2}%",
    distilled_snn.evaluate(&{{TEST_LOADER}}) * 100.0);"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "TEACHER_PATH".to_string(),
                    description: "Path to pre-trained ANN model".to_string(),
                    type_hint: "&str".to_string(),
                    example: "\"models/teacher.bin\"".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "LAYER_SIZES".to_string(),
                    description: "SNN layer sizes".to_string(),
                    type_hint: "Vec<usize>".to_string(),
                    example: "784, 256, 10".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "TEMPERATURE".to_string(),
                    description: "Softmax temperature for distillation".to_string(),
                    type_hint: "f64".to_string(),
                    example: "3.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "ALPHA".to_string(),
                    description: "Weight for distillation loss vs hard labels".to_string(),
                    type_hint: "f64".to_string(),
                    example: "0.7".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "EPOCHS".to_string(),
                    description: "Distillation epochs".to_string(),
                    type_hint: "usize".to_string(),
                    example: "50".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "DATALOADER".to_string(),
                    description: "Training data loader".to_string(),
                    type_hint: "DataLoader".to_string(),
                    example: "train_loader".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "TEST_LOADER".to_string(),
                    description: "Test data loader".to_string(),
                    type_hint: "DataLoader".to_string(),
                    example: "test_loader".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_snn::distillation::{TeacherStudentFramework, DistillationConfig, DistillationMode};
use dpb_snn::baselines::MLP;
use dpb_snn::architectures::FeedforwardSNN;

let teacher = MLP::load("models/teacher.bin")?;
let student = FeedforwardSNN::new(vec![784, 256, 10]);

let config = DistillationConfig {
    mode: DistillationMode::SpikePattern,
    temperature: 3.0,
    alpha: 0.7,
    epochs: 50,
    ..Default::default()
};

let mut framework = TeacherStudentFramework::new(teacher, student, config);
let distilled_snn = framework.distill(&train_loader)?;"#.to_string(),
            tags: vec!["training".to_string(), "distillation".to_string(), "ann".to_string(), "snn".to_string()],
        },
    ]
}

fn inference_templates() -> Vec<CodeTemplate> {
    vec![
        CodeTemplate {
            id: "snn_inference".to_string(),
            name: "SNN Inference".to_string(),
            category: "inference".to_string(),
            description: "Run inference on a trained SNN model".to_string(),
            dependencies: vec!["dpb-snn".to_string()],
            imports: vec![
                "use dpb_snn::architectures::FeedforwardSNN;".to_string(),
                "use dpb_snn::decoders::SpikeRateDecoder;".to_string(),
            ],
            code: r#"// Load trained model
let snn = FeedforwardSNN::load({{MODEL_PATH}})?;

// Create decoder for output interpretation
let decoder = SpikeRateDecoder::new({{NUM_CLASSES}});

// Run inference
let output_spikes = snn.forward(&{{INPUT_SPIKES}}, {{TIME_STEPS}});

// Decode output
let prediction = decoder.decode(&output_spikes);
println!("Predicted class: {}", prediction);"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "MODEL_PATH".to_string(),
                    description: "Path to trained SNN model".to_string(),
                    type_hint: "&str".to_string(),
                    example: "\"models/snn.bin\"".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "NUM_CLASSES".to_string(),
                    description: "Number of output classes".to_string(),
                    type_hint: "usize".to_string(),
                    example: "10".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "INPUT_SPIKES".to_string(),
                    description: "Input spike train".to_string(),
                    type_hint: "SpikeTrain".to_string(),
                    example: "input".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "TIME_STEPS".to_string(),
                    description: "Number of time steps for simulation".to_string(),
                    type_hint: "usize".to_string(),
                    example: "100".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_snn::architectures::FeedforwardSNN;
use dpb_snn::decoders::SpikeRateDecoder;

let snn = FeedforwardSNN::load("models/snn.bin")?;
let decoder = SpikeRateDecoder::new(10);

let output_spikes = snn.forward(&input, 100);
let prediction = decoder.decode(&output_spikes);
println!("Predicted class: {}", prediction);"#.to_string(),
            tags: vec!["inference".to_string(), "snn".to_string(), "prediction".to_string()],
        },
        CodeTemplate {
            id: "clinical_score_prediction".to_string(),
            name: "Clinical Score Prediction".to_string(),
            category: "inference".to_string(),
            description: "Predict clinical scores (UPDRS, MoCA, etc.) from biosignals".to_string(),
            dependencies: vec!["dpb-snn".to_string(), "dpb-encoders".to_string()],
            imports: vec![
                "use dpb_snn::decoders::{UPDRSDecoder, MoCADecoder};".to_string(),
                "use dpb_snn::fusion::MultiModalFusionSNN;".to_string(),
            ],
            code: r#"// Load multimodal fusion model
let model = MultiModalFusionSNN::load({{MODEL_PATH}})?;

// Create clinical decoders
let updrs_decoder = UPDRSDecoder::new();
let moca_decoder = MoCADecoder::new();

// Encode input signals to spikes
let spike_inputs = vec![
    (Modality::Gait, {{GAIT_SPIKES}}),
    (Modality::Tremor, {{TREMOR_SPIKES}}),
    (Modality::Voice, {{VOICE_SPIKES}}),
];

// Run fusion and get output
let output = model.forward(&spike_inputs, {{TIME_STEPS}});

// Decode clinical scores
let updrs_score = updrs_decoder.decode(&output);
let moca_score = moca_decoder.decode(&output);

println!("Predicted UPDRS: {:.1}", updrs_score);
println!("Predicted MoCA: {:.1}", moca_score);"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "MODEL_PATH".to_string(),
                    description: "Path to fusion model".to_string(),
                    type_hint: "&str".to_string(),
                    example: "\"models/clinical_fusion.bin\"".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "GAIT_SPIKES".to_string(),
                    description: "Encoded gait spikes".to_string(),
                    type_hint: "SpikeTrain".to_string(),
                    example: "gait_spikes".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "TREMOR_SPIKES".to_string(),
                    description: "Encoded tremor spikes".to_string(),
                    type_hint: "SpikeTrain".to_string(),
                    example: "tremor_spikes".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "VOICE_SPIKES".to_string(),
                    description: "Encoded voice spikes".to_string(),
                    type_hint: "SpikeTrain".to_string(),
                    example: "voice_spikes".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "TIME_STEPS".to_string(),
                    description: "Simulation time steps".to_string(),
                    type_hint: "usize".to_string(),
                    example: "200".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_snn::decoders::{UPDRSDecoder, MoCADecoder};
use dpb_snn::fusion::MultiModalFusionSNN;

let model = MultiModalFusionSNN::load("models/clinical_fusion.bin")?;
let updrs_decoder = UPDRSDecoder::new();

let spike_inputs = vec![
    (Modality::Gait, gait_spikes),
    (Modality::Tremor, tremor_spikes),
    (Modality::Voice, voice_spikes),
];

let output = model.forward(&spike_inputs, 200);
let updrs_score = updrs_decoder.decode(&output);
println!("Predicted UPDRS: {:.1}", updrs_score);"#.to_string(),
            tags: vec!["inference".to_string(), "clinical".to_string(), "updrs".to_string(), "moca".to_string(), "multimodal".to_string()],
        },
    ]
}

fn signal_processing_templates() -> Vec<CodeTemplate> {
    vec![
        CodeTemplate {
            id: "ecg_preprocessing".to_string(),
            name: "ECG Signal Preprocessing".to_string(),
            category: "signal_processing".to_string(),
            description: "Filter and preprocess ECG signals for analysis".to_string(),
            dependencies: vec!["dpb-core".to_string()],
            imports: vec![
                "use dpb_core::signal::filter::{BandpassFilter, NotchFilter};".to_string(),
                "use dpb_core::signal::ecg::{BaselineRemoval, QrsDetector};".to_string(),
                "use dpb_core::TimeSeries;".to_string(),
            ],
            code: r#"// Create filters
let bandpass = BandpassFilter::new({{LOW_CUTOFF}}, {{HIGH_CUTOFF}}, {{SAMPLE_RATE}});
let notch = NotchFilter::new({{NOTCH_FREQ}}, {{SAMPLE_RATE}});
let baseline_removal = BaselineRemoval::new({{WINDOW_SIZE}});

// Apply preprocessing pipeline
let filtered = bandpass.apply(&{{RAW_SIGNAL}});
let denoised = notch.apply(&filtered);
let clean_ecg = baseline_removal.apply(&denoised);

// Detect QRS complexes
let qrs_detector = QrsDetector::new({{SAMPLE_RATE}});
let qrs_peaks = qrs_detector.detect(&clean_ecg);

println!("Detected {} QRS complexes", qrs_peaks.len());"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "LOW_CUTOFF".to_string(),
                    description: "Low cutoff frequency (Hz)".to_string(),
                    type_hint: "f64".to_string(),
                    example: "0.5".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "HIGH_CUTOFF".to_string(),
                    description: "High cutoff frequency (Hz)".to_string(),
                    type_hint: "f64".to_string(),
                    example: "40.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "SAMPLE_RATE".to_string(),
                    description: "Signal sample rate (Hz)".to_string(),
                    type_hint: "f64".to_string(),
                    example: "250.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "NOTCH_FREQ".to_string(),
                    description: "Power line frequency (Hz)".to_string(),
                    type_hint: "f64".to_string(),
                    example: "60.0".to_string(),
                    options: Some(vec!["50.0".to_string(), "60.0".to_string()]),
                },
                Placeholder {
                    name: "WINDOW_SIZE".to_string(),
                    description: "Baseline removal window (samples)".to_string(),
                    type_hint: "usize".to_string(),
                    example: "200".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "RAW_SIGNAL".to_string(),
                    description: "Raw ECG time series".to_string(),
                    type_hint: "TimeSeries".to_string(),
                    example: "raw_ecg".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_core::signal::filter::{BandpassFilter, NotchFilter};
use dpb_core::signal::ecg::{BaselineRemoval, QrsDetector};

let bandpass = BandpassFilter::new(0.5, 40.0, 250.0);
let notch = NotchFilter::new(60.0, 250.0);
let baseline_removal = BaselineRemoval::new(200);

let filtered = bandpass.apply(&raw_ecg);
let denoised = notch.apply(&filtered);
let clean_ecg = baseline_removal.apply(&denoised);

let qrs_detector = QrsDetector::new(250.0);
let qrs_peaks = qrs_detector.detect(&clean_ecg);"#.to_string(),
            tags: vec!["signal".to_string(), "ecg".to_string(), "filter".to_string(), "preprocessing".to_string()],
        },
    ]
}

fn visualization_templates() -> Vec<CodeTemplate> {
    vec![
        CodeTemplate {
            id: "spike_raster_plot".to_string(),
            name: "Spike Raster Plot".to_string(),
            category: "visualization".to_string(),
            description: "Create a raster plot of spike trains".to_string(),
            dependencies: vec!["dpb-viz".to_string()],
            imports: vec![
                "use dpb_viz::{RasterPlot, RasterConfig};".to_string(),
                "use dpb_viz::export::SvgExporter;".to_string(),
            ],
            code: r#"// Configure raster plot
let config = RasterConfig {
    width: {{WIDTH}},
    height: {{HEIGHT}},
    time_range: ({{START_TIME}}, {{END_TIME}}),
    marker_size: {{MARKER_SIZE}},
    color_by_neuron: {{COLOR_BY_NEURON}},
    ..Default::default()
};

// Create raster plot
let plot = RasterPlot::new(config);
plot.add_spike_trains(&{{SPIKE_TRAINS}});

// Export to SVG
let exporter = SvgExporter::new();
exporter.export(&plot, {{OUTPUT_PATH}})?;"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "WIDTH".to_string(),
                    description: "Plot width in pixels".to_string(),
                    type_hint: "u32".to_string(),
                    example: "800".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "HEIGHT".to_string(),
                    description: "Plot height in pixels".to_string(),
                    type_hint: "u32".to_string(),
                    example: "400".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "START_TIME".to_string(),
                    description: "Start time (seconds)".to_string(),
                    type_hint: "f64".to_string(),
                    example: "0.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "END_TIME".to_string(),
                    description: "End time (seconds)".to_string(),
                    type_hint: "f64".to_string(),
                    example: "10.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "MARKER_SIZE".to_string(),
                    description: "Spike marker size".to_string(),
                    type_hint: "f64".to_string(),
                    example: "2.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "COLOR_BY_NEURON".to_string(),
                    description: "Color spikes by neuron index".to_string(),
                    type_hint: "bool".to_string(),
                    example: "true".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "SPIKE_TRAINS".to_string(),
                    description: "Vector of spike trains".to_string(),
                    type_hint: "Vec<SpikeTrain>".to_string(),
                    example: "spike_trains".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "OUTPUT_PATH".to_string(),
                    description: "Output file path".to_string(),
                    type_hint: "&str".to_string(),
                    example: "\"output/raster.svg\"".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_viz::{RasterPlot, RasterConfig};
use dpb_viz::export::SvgExporter;

let config = RasterConfig {
    width: 800,
    height: 400,
    time_range: (0.0, 10.0),
    marker_size: 2.0,
    color_by_neuron: true,
    ..Default::default()
};

let plot = RasterPlot::new(config);
plot.add_spike_trains(&spike_trains);

let exporter = SvgExporter::new();
exporter.export(&plot, "output/raster.svg")?;"#.to_string(),
            tags: vec!["visualization".to_string(), "raster".to_string(), "spikes".to_string(), "svg".to_string()],
        },
    ]
}

fn pipeline_templates() -> Vec<CodeTemplate> {
    vec![
        CodeTemplate {
            id: "realtime_pipeline".to_string(),
            name: "Real-time Processing Pipeline".to_string(),
            category: "pipeline".to_string(),
            description: "Create a streaming pipeline for real-time biosignal processing".to_string(),
            dependencies: vec!["dpb-core".to_string(), "dpb-encoders".to_string(), "dpb-snn".to_string()],
            imports: vec![
                "use dpb_core::pipeline::{Pipeline, PipelineConfig, StreamBuffer};".to_string(),
                "use dpb_encoders::contact::ecg::EcgRPeakEncoder;".to_string(),
                "use dpb_snn::architectures::FeedforwardSNN;".to_string(),
                "use tokio::sync::mpsc;".to_string(),
            ],
            code: r#"// Configure pipeline
let config = PipelineConfig {
    buffer_size: {{BUFFER_SIZE}},
    sample_rate: {{SAMPLE_RATE}},
    window_size: {{WINDOW_SIZE}},
    overlap: {{OVERLAP}},
    ..Default::default()
};

// Create pipeline components
let encoder = EcgRPeakEncoder::new({{SAMPLE_RATE}});
let model = FeedforwardSNN::load({{MODEL_PATH}})?;

// Create async channel for results
let (tx, mut rx) = mpsc::channel(100);

// Build pipeline
let mut pipeline = Pipeline::new(config)
    .add_stage("encode", move |signal| encoder.encode(&signal))
    .add_stage("classify", move |spikes| model.forward(&spikes, 100))
    .with_output(tx);

// Start processing
pipeline.start().await;

// Handle results
while let Some(result) = rx.recv().await {
    println!("Classification: {:?}", result);
}"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "BUFFER_SIZE".to_string(),
                    description: "Internal buffer size (samples)".to_string(),
                    type_hint: "usize".to_string(),
                    example: "1024".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "SAMPLE_RATE".to_string(),
                    description: "Input sample rate (Hz)".to_string(),
                    type_hint: "f64".to_string(),
                    example: "250.0".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "WINDOW_SIZE".to_string(),
                    description: "Processing window size (samples)".to_string(),
                    type_hint: "usize".to_string(),
                    example: "500".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "OVERLAP".to_string(),
                    description: "Window overlap (samples)".to_string(),
                    type_hint: "usize".to_string(),
                    example: "250".to_string(),
                    options: None,
                },
                Placeholder {
                    name: "MODEL_PATH".to_string(),
                    description: "Path to trained model".to_string(),
                    type_hint: "&str".to_string(),
                    example: "\"models/classifier.bin\"".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_core::pipeline::{Pipeline, PipelineConfig};
use dpb_encoders::contact::ecg::EcgRPeakEncoder;
use dpb_snn::architectures::FeedforwardSNN;

let config = PipelineConfig {
    buffer_size: 1024,
    sample_rate: 250.0,
    window_size: 500,
    overlap: 250,
    ..Default::default()
};

let encoder = EcgRPeakEncoder::new(250.0);
let model = FeedforwardSNN::load("models/classifier.bin")?;

let mut pipeline = Pipeline::new(config)
    .add_stage("encode", move |signal| encoder.encode(&signal))
    .add_stage("classify", move |spikes| model.forward(&spikes, 100));

pipeline.start().await;"#.to_string(),
            tags: vec!["pipeline".to_string(), "realtime".to_string(), "streaming".to_string(), "async".to_string()],
        },
        CodeTemplate {
            id: "lsl_integration".to_string(),
            name: "Lab Streaming Layer Integration".to_string(),
            category: "pipeline".to_string(),
            description: "Connect to LSL streams for real-time biosignal acquisition".to_string(),
            dependencies: vec!["dpb-lsl".to_string(), "dpb-core".to_string()],
            imports: vec![
                "use dpb_lsl::{LslInlet, StreamResolver, ChannelFormat};".to_string(),
                "use dpb_core::TimeSeries;".to_string(),
            ],
            code: r#"// Resolve LSL streams
let resolver = StreamResolver::new();
let streams = resolver.resolve_by_type({{STREAM_TYPE}}, {{TIMEOUT}}).await?;

if streams.is_empty() {
    return Err(anyhow::anyhow!("No {} streams found", {{STREAM_TYPE}}));
}

// Create inlet for first matching stream
let inlet = LslInlet::new(&streams[0]).await?;

// Read samples
let mut buffer = vec![0.0f64; inlet.channel_count()];
loop {
    let timestamp = inlet.pull_sample(&mut buffer, {{TIMEOUT}}).await?;

    // Process sample
    println!("t={:.3}: {:?}", timestamp, buffer);

    // Convert to TimeSeries for further processing
    let sample = TimeSeries::from_sample(&buffer, inlet.sample_rate());
    // ... process sample
}"#.to_string(),
            placeholders: vec![
                Placeholder {
                    name: "STREAM_TYPE".to_string(),
                    description: "LSL stream type to search for".to_string(),
                    type_hint: "&str".to_string(),
                    example: "\"EEG\"".to_string(),
                    options: Some(vec!["\"EEG\"".to_string(), "\"ECG\"".to_string(), "\"EMG\"".to_string(), "\"Markers\"".to_string()]),
                },
                Placeholder {
                    name: "TIMEOUT".to_string(),
                    description: "Timeout in seconds".to_string(),
                    type_hint: "f64".to_string(),
                    example: "5.0".to_string(),
                    options: None,
                },
            ],
            example: r#"use dpb_lsl::{LslInlet, StreamResolver};

let resolver = StreamResolver::new();
let streams = resolver.resolve_by_type("EEG", 5.0).await?;

let inlet = LslInlet::new(&streams[0]).await?;
let mut buffer = vec![0.0f64; inlet.channel_count()];

loop {
    let timestamp = inlet.pull_sample(&mut buffer, 1.0).await?;
    println!("t={:.3}: {:?}", timestamp, buffer);
}"#.to_string(),
            tags: vec!["lsl".to_string(), "streaming".to_string(), "realtime".to_string(), "eeg".to_string()],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_registry_creation() {
        let registry = TemplateRegistry::default_templates();
        assert!(registry.templates.len() > 0);
    }

    #[test]
    fn test_template_search() {
        let registry = TemplateRegistry::default_templates();
        let results = registry.search_by_tag("ecg");
        assert!(results.len() > 0);
    }
}
