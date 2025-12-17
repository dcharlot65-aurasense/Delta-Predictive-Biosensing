//! Level 3 Synthetic Data Generators
//!
//! Level 3 generators produce actual multimedia files (video, audio) using
//! external rendering tools. These are intended for full end-to-end pipeline
//! testing with MediaPipe, audio processing, and complete validation workflows.
//!
//! # Architecture
//!
//! Level 3 generators work differently from Level 1 and 2:
//!
//! - **Level 1**: Pure mathematical synthesis (no external dependencies)
//! - **Level 2**: Signal-level synthesis with noise and artifacts
//! - **Level 3**: Full multimedia rendering via external tools (Blender, espeak-ng, etc.)
//!
//! # Requirements
//!
//! Level 3 generators require external tools to be installed:
//!
//! ## Video Generation
//! - **Blender 3.x or 4.x**: For 3D rendering and animation
//!   - Install: https://www.blender.org/download/
//!   - Must be available in PATH or specify path explicitly
//!
//! ## Audio Generation
//! - **espeak-ng**: For text-to-speech synthesis
//!   - Install: `sudo apt-get install espeak-ng` (Linux)
//!   - Install: `brew install espeak` (macOS)
//!
//! # Usage
//!
//! ```rust,no_run
//! use dpb_synth::level3::video::{Level3VideoGenerator, GaitVideoParams};
//!
//! // Create video generator
//! let generator = Level3VideoGenerator::new(None, None);
//!
//! // Check if Blender is available
//! if generator.check_blender().unwrap_or(false) {
//!     // Generate gait video
//!     let params = GaitVideoParams::default();
//!     let output = generator.generate_gait_video(&params).unwrap();
//!     
//!     println!("Video: {:?}", output.video_path);
//!     println!("Ground truth: {:?}", output.ground_truth_path);
//! }
//! ```
//!
//! # Fallback Mode
//!
//! If external tools are not available, Level 3 generators can still produce
//! ground truth data without rendering actual media files. Set `render: false`
//! in parameters to generate only ground truth annotations.

pub mod video;
pub mod audio;
pub mod audio_world;
pub mod skeleton;
pub mod smpl;
pub mod style_transfer;

pub use video::{
    Level3VideoGenerator,
    GaitVideoParams,
    HandVideoParams,
    TappingVideoParams,
    VideoOutput,
    PoseGroundTruth,
    HandGroundTruth,
    TappingGroundTruth,
    VideoGeneratorError,
};

pub use audio::{
    Level3AudioGenerator,
    VoiceAudioParams,
    AudioOutput,
    VoiceGroundTruth,
    AudioBackend,
    AudioGeneratorError,
};

pub use audio_world::{
    SustainedVowelGenerator,
    ConnectedSpeechGenerator,
    DiadochokinesisGenerator,
    ReadingPassageGenerator,
    SustainedVowelParams,
    ConnectedSpeechParams,
    DiadochokinesisParams,
    ReadingPassageParams,
    AudioGroundTruth,
    AudioEvent,
    WorldAudioGenerator,
};

pub use skeleton::{
    SkeletonRenderer,
    SkeletonParams,
    SkeletonError,
    RenderStyle,
    CameraView,
    PoseLandmarks,
    HandLandmarks,
    render_clinical_pose_frame,
    render_clinical_hand_frame,
};

pub use smpl::{
    SmplRenderer,
    SmplParams,
    SmplError,
    SmplBodyModel,
    SmplPose,
};

pub use style_transfer::{
    StyleTransferRenderer,
    StyleTransferParams,
    StyleTransferError,
    StyleType,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_generator_exists() {
        let _gen = Level3VideoGenerator::new(None, None);
    }

    #[test]
    fn test_audio_generator_exists() {
        let _gen = Level3AudioGenerator::new();
    }
}
