//! Integration tests for the DPB Framework
//!
//! Run with: `cargo test --test integration_tests`
//!
//! This test suite validates end-to-end pipelines across all DPB crates:
//! - dpb-core: Core types and traits
//! - dpb-synth: Synthetic data generation
//! - dpb-encoders: Event-based encoding
//! - dpb-snn: Spiking neural networks
//!
//! ## Test Categories
//!
//! ### Pipeline Tests
//! - `ecg_pipeline`: ECG signal → encoding → SNN → heart rate
//! - `gait_pipeline`: Gait keypoints → encoding → SNN → UPDRS
//! - `tremor_pipeline`: Tremor signal → encoding → SNN → classification
//! - `voice_pipeline`: Voice features → encoding → SNN → assessment
//! - `multimodal_pipeline`: Multi-modal fusion pipelines
//!
//! ### Validation Tests
//! - `synthetic_validation`: Validate synthetic data ground truth
//! - `encoder_accuracy`: Encoder precision and recall
//!
//! ### Training & Performance
//! - `training_loop`: BPTT convergence tests
//! - `inference_latency`: Performance benchmarks
//!
//! ### Compatibility
//! - `cross_crate`: Cross-crate API compatibility

mod integration;
