//! MediaPipe Comprehensive Video and Audio Processing Integration
//!
//! This module provides full integration with Google MediaPipe for extracting
//! and analyzing biosignals from video and audio - closing the loop from
//! synthetic generation back to biosignal extraction.
//!
//! # Vision Solutions
//!
//! - **Pose Landmarker**: 33-point BlazePose body landmarks with world coordinates
//! - **Hand Landmarker**: 21-point hand landmarks per hand with gesture recognition
//! - **Face Mesh**: 468-point face mesh with blendshapes for expression analysis
//! - **Face Detection**: Fast face detection with 6 keypoints
//! - **Holistic**: Combined face, hands, and pose in unified model
//! - **Gesture Recognition**: ASL alphabet and common gestures
//! - **Image Segmentation**: Person/selfie segmentation masks
//! - **Object Detection**: General object detection
//! - **Iris Tracking**: Pupil and iris landmark tracking
//!
//! # Audio Solutions
//!
//! - **Audio Classification**: Speech, music, environment sound classification
//! - **Audio Embedding**: Feature extraction for downstream tasks
//!
//! # Biosignal Applications
//!
//! - Gait analysis from pose sequences
//! - Tremor detection from hand/pose landmarks
//! - Facial expression analysis (hypomimia detection)
//! - Eye tracking and pupil response
//! - Voice activity detection
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install mediapipe opencv-python numpy`
//!
//! # References
//!
//! - MediaPipe Solutions: https://developers.google.com/mediapipe/solutions
//! - MediaPipe Tasks: https://developers.google.com/mediapipe/solutions/guide

use super::{MediaError, Result, MediaGroundTruth, JointPositions3D, GaitGroundTruth};
use std::path::{Path, PathBuf};
use std::process::Command;

/// MediaPipe pose/hand/face extractor
#[derive(Debug)]
pub struct MediaPipeExtractor {
    config: MediaPipeConfig,
    python_path: PathBuf,
}

/// Configuration for MediaPipe extraction
#[derive(Debug, Clone)]
pub struct MediaPipeConfig {
    /// Enable pose detection
    pub enable_pose: bool,
    /// Enable hand detection
    pub enable_hands: bool,
    /// Enable face mesh
    pub enable_face: bool,
    /// Model complexity (0=lite, 1=full, 2=heavy)
    pub model_complexity: u8,
    /// Minimum detection confidence
    pub min_detection_confidence: f32,
    /// Minimum tracking confidence
    pub min_tracking_confidence: f32,
    /// Enable segmentation mask
    pub enable_segmentation: bool,
    /// Output directory
    pub output_dir: PathBuf,
}

impl Default for MediaPipeConfig {
    fn default() -> Self {
        Self {
            enable_pose: true,
            enable_hands: true,
            enable_face: false,
            model_complexity: 1,
            min_detection_confidence: 0.5,
            min_tracking_confidence: 0.5,
            enable_segmentation: false,
            output_dir: PathBuf::from("output/poses"),
        }
    }
}

/// Pose estimation result for a single frame
#[derive(Debug, Clone)]
pub struct PoseEstimate {
    /// Frame number
    pub frame: u32,
    /// Timestamp in seconds
    pub timestamp: f64,
    /// 33 body landmarks (BlazePose)
    pub landmarks: Vec<Landmark3D>,
    /// World landmarks (metric scale)
    pub world_landmarks: Option<Vec<Landmark3D>>,
    /// Visibility scores for each landmark
    pub visibility: Vec<f32>,
    /// Segmentation mask (if enabled)
    pub segmentation_mask: Option<Vec<Vec<f32>>>,
}

/// 3D landmark point
#[derive(Debug, Clone, Copy)]
pub struct Landmark3D {
    /// X coordinate (normalized 0-1 or meters for world)
    pub x: f32,
    /// Y coordinate
    pub y: f32,
    /// Z coordinate
    pub z: f32,
    /// Visibility score (0-1)
    pub visibility: f32,
}

/// Hand landmarks result
#[derive(Debug, Clone)]
pub struct HandLandmarks {
    /// Frame number
    pub frame: u32,
    /// Timestamp
    pub timestamp: f64,
    /// Left hand landmarks (21 points)
    pub left_hand: Option<Vec<Landmark3D>>,
    /// Right hand landmarks (21 points)
    pub right_hand: Option<Vec<Landmark3D>>,
    /// Handedness classification scores
    pub handedness: Option<(f32, f32)>, // (left_score, right_score)
}

/// Face mesh result
#[derive(Debug, Clone)]
pub struct FaceMesh {
    /// Frame number
    pub frame: u32,
    /// Timestamp
    pub timestamp: f64,
    /// 468 face landmarks
    pub landmarks: Vec<Landmark3D>,
    /// Face detection confidence
    pub confidence: f32,
}

/// Complete extraction result from a video
#[derive(Debug)]
pub struct ExtractionResult {
    /// Video file processed
    pub video_path: PathBuf,
    /// Total frames processed
    pub total_frames: u32,
    /// Frames per second
    pub fps: f64,
    /// Duration in seconds
    pub duration: f64,
    /// Pose estimates per frame
    pub poses: Vec<PoseEstimate>,
    /// Hand landmarks per frame
    pub hands: Vec<HandLandmarks>,
    /// Face mesh per frame (if enabled)
    pub faces: Vec<FaceMesh>,
    /// Processing time in seconds
    pub processing_time: f64,
}

/// BlazePose landmark indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PoseLandmark {
    Nose = 0,
    LeftEyeInner = 1,
    LeftEye = 2,
    LeftEyeOuter = 3,
    RightEyeInner = 4,
    RightEye = 5,
    RightEyeOuter = 6,
    LeftEar = 7,
    RightEar = 8,
    MouthLeft = 9,
    MouthRight = 10,
    LeftShoulder = 11,
    RightShoulder = 12,
    LeftElbow = 13,
    RightElbow = 14,
    LeftWrist = 15,
    RightWrist = 16,
    LeftPinky = 17,
    RightPinky = 18,
    LeftIndex = 19,
    RightIndex = 20,
    LeftThumb = 21,
    RightThumb = 22,
    LeftHip = 23,
    RightHip = 24,
    LeftKnee = 25,
    RightKnee = 26,
    LeftAnkle = 27,
    RightAnkle = 28,
    LeftHeel = 29,
    RightHeel = 30,
    LeftFootIndex = 31,
    RightFootIndex = 32,
}

impl MediaPipeExtractor {
    /// Create a new MediaPipe extractor
    pub fn new(config: MediaPipeConfig) -> Result<Self> {
        // Check if MediaPipe is available
        let check = Command::new("python3")
            .args(["-c", "import mediapipe; import cv2; print('ok')"])
            .output();

        match check {
            Ok(output) if output.status.success() => {}
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "mediapipe".to_string(),
                    install_url: "pip install mediapipe opencv-python".to_string(),
                });
            }
        }

        let python_path = which::which("python3")
            .map_err(|_| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: "https://www.python.org/downloads/".to_string(),
            })?;

        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self {
            config,
            python_path,
        })
    }

    /// Check if MediaPipe is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import mediapipe"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Extract poses from video file
    pub fn extract_from_video(&self, video_path: &Path) -> Result<ExtractionResult> {
        let script = self.generate_extraction_script(video_path)?;
        self.run_extraction_script(&script, video_path)
    }

    /// Extract poses from image
    pub fn extract_from_image(&self, image_path: &Path) -> Result<PoseEstimate> {
        let script = format!(r#"
import mediapipe as mp
import cv2
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity={complexity},
    min_detection_confidence={det_conf}
)

image = cv2.imread('{image_path}')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

if results.pose_landmarks:
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.append({{
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility
        }})

    world_landmarks = []
    if results.pose_world_landmarks:
        for lm in results.pose_world_landmarks.landmark:
            world_landmarks.append({{
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            }})

    result = {{
        'frame': 0,
        'timestamp': 0.0,
        'landmarks': landmarks,
        'world_landmarks': world_landmarks,
        'visibility': [lm['visibility'] for lm in landmarks]
    }}
    print('RESULT_JSON:' + json.dumps(result))
else:
    print('RESULT_JSON:{{"error": "No pose detected"}}')

pose.close()
"#,
            complexity = self.config.model_complexity,
            det_conf = self.config.min_detection_confidence,
            image_path = image_path.display(),
        );

        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mediapipe_image.py");
        std::fs::write(&script_path, &script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        if data.get("error").is_some() {
            return Err(MediaError::ExecutionFailed("No pose detected".to_string()));
        }

        Ok(self.parse_pose_estimate(&data))
    }

    /// Generate extraction script
    fn generate_extraction_script(&self, video_path: &Path) -> Result<String> {
        Ok(format!(r#"
import mediapipe as mp
import cv2
import json
import time

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

pose = mp_pose.Pose(
    model_complexity={complexity},
    min_detection_confidence={det_conf},
    min_tracking_confidence={track_conf},
    enable_segmentation={segmentation}
) if {enable_pose} else None

hands = mp_hands.Hands(
    model_complexity={complexity},
    min_detection_confidence={det_conf},
    min_tracking_confidence={track_conf}
) if {enable_hands} else None

face = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence={det_conf},
    min_tracking_confidence={track_conf}
) if {enable_face} else None

# Open video
cap = cv2.VideoCapture('{video_path}')
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

poses = []
hands_data = []
faces_data = []

start_time = time.time()
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp = frame_num / fps

    # Pose extraction
    if pose:
        pose_results = pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            landmarks = []
            for lm in pose_results.pose_landmarks.landmark:
                landmarks.append({{
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z),
                    'visibility': float(lm.visibility)
                }})

            world_landmarks = []
            if pose_results.pose_world_landmarks:
                for lm in pose_results.pose_world_landmarks.landmark:
                    world_landmarks.append({{
                        'x': float(lm.x),
                        'y': float(lm.y),
                        'z': float(lm.z),
                        'visibility': float(lm.visibility)
                    }})

            poses.append({{
                'frame': frame_num,
                'timestamp': timestamp,
                'landmarks': landmarks,
                'world_landmarks': world_landmarks,
                'visibility': [lm['visibility'] for lm in landmarks]
            }})

    # Hand extraction
    if hands:
        hand_results = hands.process(frame_rgb)
        hand_entry = {{
            'frame': frame_num,
            'timestamp': timestamp,
            'left_hand': None,
            'right_hand': None
        }}

        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_lms = []
                for lm in hand_landmarks.landmark:
                    hand_lms.append({{
                        'x': float(lm.x),
                        'y': float(lm.y),
                        'z': float(lm.z),
                        'visibility': 1.0
                    }})

                # Determine handedness
                if hand_results.multi_handedness:
                    label = hand_results.multi_handedness[idx].classification[0].label
                    if label == 'Left':
                        hand_entry['left_hand'] = hand_lms
                    else:
                        hand_entry['right_hand'] = hand_lms

        hands_data.append(hand_entry)

    # Face extraction
    if face:
        face_results = face.process(frame_rgb)
        if face_results.multi_face_landmarks:
            face_lms = []
            for lm in face_results.multi_face_landmarks[0].landmark:
                face_lms.append({{
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z),
                    'visibility': 1.0
                }})

            faces_data.append({{
                'frame': frame_num,
                'timestamp': timestamp,
                'landmarks': face_lms,
                'confidence': 1.0
            }})

    frame_num += 1

    # Progress update every 100 frames
    if frame_num % 100 == 0:
        print(f"Processed {{frame_num}}/{{total_frames}} frames", flush=True)

processing_time = time.time() - start_time

cap.release()
if pose:
    pose.close()
if hands:
    hands.close()
if face:
    face.close()

# Output results
result = {{
    'total_frames': total_frames,
    'fps': fps,
    'duration': duration,
    'poses': poses,
    'hands': hands_data,
    'faces': faces_data,
    'processing_time': processing_time
}}

# Write to file (too large for stdout)
output_path = '{output_path}'
with open(output_path, 'w') as f:
    json.dump(result, f)

print(f'RESULT_FILE:{{output_path}}')
"#,
            complexity = self.config.model_complexity,
            det_conf = self.config.min_detection_confidence,
            track_conf = self.config.min_tracking_confidence,
            segmentation = self.config.enable_segmentation,
            enable_pose = self.config.enable_pose,
            enable_hands = self.config.enable_hands,
            enable_face = self.config.enable_face,
            video_path = video_path.display(),
            output_path = self.config.output_dir.join("extraction_result.json").display(),
        ))
    }

    /// Run extraction script and parse results
    fn run_extraction_script(&self, script: &str, video_path: &Path) -> Result<ExtractionResult> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mediapipe_extract.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        // Get result file path from stdout
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_FILE:"))
            .ok_or_else(|| MediaError::SerializationError("No result file".to_string()))?;

        let result_path = result_line.trim_start_matches("RESULT_FILE:");
        let result_content = std::fs::read_to_string(result_path)?;
        let data: serde_json::Value = serde_json::from_str(&result_content)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        // Parse results
        let poses: Vec<PoseEstimate> = data["poses"].as_array()
            .map(|arr| arr.iter().map(|p| self.parse_pose_estimate(p)).collect())
            .unwrap_or_default();

        let hands: Vec<HandLandmarks> = data["hands"].as_array()
            .map(|arr| arr.iter().map(|h| self.parse_hand_landmarks(h)).collect())
            .unwrap_or_default();

        let faces: Vec<FaceMesh> = data["faces"].as_array()
            .map(|arr| arr.iter().map(|f| self.parse_face_mesh(f)).collect())
            .unwrap_or_default();

        Ok(ExtractionResult {
            video_path: video_path.to_path_buf(),
            total_frames: data["total_frames"].as_u64().unwrap_or(0) as u32,
            fps: data["fps"].as_f64().unwrap_or(30.0),
            duration: data["duration"].as_f64().unwrap_or(0.0),
            poses,
            hands,
            faces,
            processing_time: data["processing_time"].as_f64().unwrap_or(0.0),
        })
    }

    /// Parse pose estimate from JSON
    fn parse_pose_estimate(&self, data: &serde_json::Value) -> PoseEstimate {
        let landmarks: Vec<Landmark3D> = data["landmarks"].as_array()
            .map(|arr| arr.iter().map(|l| Landmark3D {
                x: l["x"].as_f64().unwrap_or(0.0) as f32,
                y: l["y"].as_f64().unwrap_or(0.0) as f32,
                z: l["z"].as_f64().unwrap_or(0.0) as f32,
                visibility: l["visibility"].as_f64().unwrap_or(0.0) as f32,
            }).collect())
            .unwrap_or_default();

        let world_landmarks: Option<Vec<Landmark3D>> = data["world_landmarks"].as_array()
            .map(|arr| arr.iter().map(|l| Landmark3D {
                x: l["x"].as_f64().unwrap_or(0.0) as f32,
                y: l["y"].as_f64().unwrap_or(0.0) as f32,
                z: l["z"].as_f64().unwrap_or(0.0) as f32,
                visibility: l["visibility"].as_f64().unwrap_or(0.0) as f32,
            }).collect());

        PoseEstimate {
            frame: data["frame"].as_u64().unwrap_or(0) as u32,
            timestamp: data["timestamp"].as_f64().unwrap_or(0.0),
            visibility: landmarks.iter().map(|l| l.visibility).collect(),
            landmarks,
            world_landmarks,
            segmentation_mask: None,
        }
    }

    /// Parse hand landmarks from JSON
    fn parse_hand_landmarks(&self, data: &serde_json::Value) -> HandLandmarks {
        let parse_hand = |hand_data: &serde_json::Value| -> Option<Vec<Landmark3D>> {
            hand_data.as_array().map(|arr| {
                arr.iter().map(|l| Landmark3D {
                    x: l["x"].as_f64().unwrap_or(0.0) as f32,
                    y: l["y"].as_f64().unwrap_or(0.0) as f32,
                    z: l["z"].as_f64().unwrap_or(0.0) as f32,
                    visibility: l["visibility"].as_f64().unwrap_or(1.0) as f32,
                }).collect()
            })
        };

        HandLandmarks {
            frame: data["frame"].as_u64().unwrap_or(0) as u32,
            timestamp: data["timestamp"].as_f64().unwrap_or(0.0),
            left_hand: parse_hand(&data["left_hand"]),
            right_hand: parse_hand(&data["right_hand"]),
            handedness: None,
        }
    }

    /// Parse face mesh from JSON
    fn parse_face_mesh(&self, data: &serde_json::Value) -> FaceMesh {
        let landmarks: Vec<Landmark3D> = data["landmarks"].as_array()
            .map(|arr| arr.iter().map(|l| Landmark3D {
                x: l["x"].as_f64().unwrap_or(0.0) as f32,
                y: l["y"].as_f64().unwrap_or(0.0) as f32,
                z: l["z"].as_f64().unwrap_or(0.0) as f32,
                visibility: l["visibility"].as_f64().unwrap_or(1.0) as f32,
            }).collect())
            .unwrap_or_default();

        FaceMesh {
            frame: data["frame"].as_u64().unwrap_or(0) as u32,
            timestamp: data["timestamp"].as_f64().unwrap_or(0.0),
            landmarks,
            confidence: data["confidence"].as_f64().unwrap_or(1.0) as f32,
        }
    }

    /// Compute gait parameters from pose sequence
    pub fn compute_gait_parameters(&self, poses: &[PoseEstimate]) -> Result<GaitGroundTruth> {
        if poses.is_empty() {
            return Err(MediaError::InvalidConfig("No poses to analyze".to_string()));
        }

        // Get world landmarks for metric measurements
        let world_poses: Vec<_> = poses.iter()
            .filter_map(|p| p.world_landmarks.as_ref())
            .collect();

        if world_poses.is_empty() {
            return Err(MediaError::InvalidConfig("No world landmarks available".to_string()));
        }

        // Analyze ankle positions for gait
        let left_ankle_idx = PoseLandmark::LeftAnkle as usize;
        let right_ankle_idx = PoseLandmark::RightAnkle as usize;
        let left_hip_idx = PoseLandmark::LeftHip as usize;
        let right_hip_idx = PoseLandmark::RightHip as usize;

        // Compute stride length from ankle positions
        let mut stride_lengths = Vec::new();
        let mut step_times = Vec::new();
        let mut last_left_forward = None;
        let mut last_step_time = 0.0;

        for (i, wl) in world_poses.iter().enumerate() {
            let left_ankle = &wl[left_ankle_idx];
            let right_ankle = &wl[right_ankle_idx];

            // Detect step (left foot forward)
            let left_forward = left_ankle.y > right_ankle.y;

            if let Some(prev_left_forward) = last_left_forward
                && left_forward != prev_left_forward {
                    // Step detected
                    let stride = ((left_ankle.x - right_ankle.x).powi(2) +
                                  (left_ankle.y - right_ankle.y).powi(2)).sqrt();
                    stride_lengths.push(stride);

                    let timestamp = poses.get(i).map(|p| p.timestamp).unwrap_or(0.0);
                    if last_step_time > 0.0 {
                        step_times.push(timestamp - last_step_time);
                    }
                    last_step_time = timestamp;
                }
            last_left_forward = Some(left_forward);
        }

        // Compute statistics
        let avg_stride = if stride_lengths.is_empty() {
            1.0
        } else {
            stride_lengths.iter().sum::<f32>() / stride_lengths.len() as f32
        };

        let avg_step_time = if step_times.is_empty() {
            0.5
        } else {
            step_times.iter().sum::<f64>() / step_times.len() as f64
        };

        let cadence = 60.0 / avg_step_time as f32; // Steps per minute
        let speed = avg_stride * cadence / 60.0; // m/s

        // Compute arm swing asymmetry
        let left_shoulder_idx = PoseLandmark::LeftShoulder as usize;
        let right_shoulder_idx = PoseLandmark::RightShoulder as usize;
        let left_wrist_idx = PoseLandmark::LeftWrist as usize;
        let right_wrist_idx = PoseLandmark::RightWrist as usize;

        let mut left_arm_range = 0.0f32;
        let mut right_arm_range = 0.0f32;

        for wl in &world_poses {
            let left_swing = (wl[left_wrist_idx].y - wl[left_shoulder_idx].y).abs();
            let right_swing = (wl[right_wrist_idx].y - wl[right_shoulder_idx].y).abs();
            left_arm_range = left_arm_range.max(left_swing);
            right_arm_range = right_arm_range.max(right_swing);
        }

        let arm_swing_asymmetry = if left_arm_range + right_arm_range > 0.0 {
            (left_arm_range - right_arm_range) / (left_arm_range + right_arm_range)
        } else {
            0.0
        };

        Ok(GaitGroundTruth {
            stride_length: avg_stride,
            cadence,
            speed,
            double_support_fraction: 0.3, // Default estimate
            step_width: 0.1, // Would need frontal view
            arm_swing_asymmetry,
            festination: cadence > 140.0 && avg_stride < 0.4,
            freezing_episodes: Vec::new(), // Would need velocity analysis
        })
    }
}

// =============================================================================
// HOLISTIC MODE - Combined Face, Hands, and Pose
// =============================================================================

/// MediaPipe Holistic processor for unified extraction
#[derive(Debug)]
pub struct HolisticExtractor {
    config: HolisticConfig,
    python_path: PathBuf,
}

/// Configuration for Holistic mode
#[derive(Debug, Clone)]
pub struct HolisticConfig {
    /// Model complexity (0=lite, 1=full, 2=heavy)
    pub model_complexity: u8,
    /// Minimum detection confidence
    pub min_detection_confidence: f32,
    /// Minimum tracking confidence
    pub min_tracking_confidence: f32,
    /// Enable segmentation mask
    pub enable_segmentation: bool,
    /// Refine face landmarks (more accurate but slower)
    pub refine_face_landmarks: bool,
    /// Output directory
    pub output_dir: PathBuf,
}

impl Default for HolisticConfig {
    fn default() -> Self {
        Self {
            model_complexity: 1,
            min_detection_confidence: 0.5,
            min_tracking_confidence: 0.5,
            enable_segmentation: false,
            refine_face_landmarks: true,
            output_dir: PathBuf::from("output/holistic"),
        }
    }
}

/// Complete holistic result for a single frame
#[derive(Debug, Clone)]
pub struct HolisticResult {
    /// Frame number
    pub frame: u32,
    /// Timestamp in seconds
    pub timestamp: f64,
    /// Pose landmarks (33 points)
    pub pose: Option<Vec<Landmark3D>>,
    /// Pose world landmarks (metric scale)
    pub pose_world: Option<Vec<Landmark3D>>,
    /// Left hand landmarks (21 points)
    pub left_hand: Option<Vec<Landmark3D>>,
    /// Right hand landmarks (21 points)
    pub right_hand: Option<Vec<Landmark3D>>,
    /// Face landmarks (468 points)
    pub face: Option<Vec<Landmark3D>>,
    /// Segmentation mask
    pub segmentation: Option<Vec<Vec<f32>>>,
}

impl HolisticExtractor {
    /// Create a new Holistic extractor
    pub fn new(config: HolisticConfig) -> Result<Self> {
        let check = Command::new("python3")
            .args(["-c", "import mediapipe as mp; mp.solutions.holistic; print('ok')"])
            .output();

        match check {
            Ok(output) if output.status.success() => {}
            _ => {
                return Err(MediaError::ToolNotFound {
                    tool: "mediapipe".to_string(),
                    install_url: "pip install mediapipe".to_string(),
                });
            }
        }

        let python_path = which::which("python3")
            .map_err(|_| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: "https://www.python.org/downloads/".to_string(),
            })?;

        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self { config, python_path })
    }

    /// Process video with holistic model
    pub fn process_video(&self, video_path: &Path) -> Result<Vec<HolisticResult>> {
        let script = self.generate_script(video_path)?;
        self.run_script(&script)
    }

    fn generate_script(&self, video_path: &Path) -> Result<String> {
        Ok(format!(r#"
import mediapipe as mp
import cv2
import json
import numpy as np

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    model_complexity={complexity},
    min_detection_confidence={det_conf},
    min_tracking_confidence={track_conf},
    enable_segmentation={segmentation},
    refine_face_landmarks={refine_face}
)

cap = cv2.VideoCapture('{video_path}')
fps = cap.get(cv2.CAP_PROP_FPS)
results_list = []
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    frame_data = {{
        'frame': frame_num,
        'timestamp': frame_num / fps,
        'pose': None,
        'pose_world': None,
        'left_hand': None,
        'right_hand': None,
        'face': None
    }}

    if results.pose_landmarks:
        frame_data['pose'] = [
            {{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}}
            for lm in results.pose_landmarks.landmark
        ]

    if results.pose_world_landmarks:
        frame_data['pose_world'] = [
            {{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}}
            for lm in results.pose_world_landmarks.landmark
        ]

    if results.left_hand_landmarks:
        frame_data['left_hand'] = [
            {{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': 1.0}}
            for lm in results.left_hand_landmarks.landmark
        ]

    if results.right_hand_landmarks:
        frame_data['right_hand'] = [
            {{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': 1.0}}
            for lm in results.right_hand_landmarks.landmark
        ]

    if results.face_landmarks:
        frame_data['face'] = [
            {{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': 1.0}}
            for lm in results.face_landmarks.landmark
        ]

    results_list.append(frame_data)
    frame_num += 1

cap.release()
holistic.close()

print('RESULT_JSON:' + json.dumps(results_list))
"#,
            complexity = self.config.model_complexity,
            det_conf = self.config.min_detection_confidence,
            track_conf = self.config.min_tracking_confidence,
            segmentation = self.config.enable_segmentation,
            refine_face = self.config.refine_face_landmarks,
            video_path = video_path.display(),
        ))
    }

    fn run_script(&self, script: &str) -> Result<Vec<HolisticResult>> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mediapipe_holistic.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: Vec<serde_json::Value> = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(data.iter().map(|f| self.parse_frame(f)).collect())
    }

    fn parse_frame(&self, data: &serde_json::Value) -> HolisticResult {
        let parse_landmarks = |arr: &serde_json::Value| -> Option<Vec<Landmark3D>> {
            arr.as_array().map(|a| {
                a.iter().map(|l| Landmark3D {
                    x: l["x"].as_f64().unwrap_or(0.0) as f32,
                    y: l["y"].as_f64().unwrap_or(0.0) as f32,
                    z: l["z"].as_f64().unwrap_or(0.0) as f32,
                    visibility: l["visibility"].as_f64().unwrap_or(1.0) as f32,
                }).collect()
            })
        };

        HolisticResult {
            frame: data["frame"].as_u64().unwrap_or(0) as u32,
            timestamp: data["timestamp"].as_f64().unwrap_or(0.0),
            pose: parse_landmarks(&data["pose"]),
            pose_world: parse_landmarks(&data["pose_world"]),
            left_hand: parse_landmarks(&data["left_hand"]),
            right_hand: parse_landmarks(&data["right_hand"]),
            face: parse_landmarks(&data["face"]),
            segmentation: None,
        }
    }
}

// =============================================================================
// GESTURE RECOGNITION
// =============================================================================

/// MediaPipe Gesture Recognizer
#[derive(Debug)]
pub struct GestureRecognizer {
    config: GestureConfig,
    python_path: PathBuf,
}

/// Configuration for Gesture Recognition
#[derive(Debug, Clone)]
pub struct GestureConfig {
    /// Number of hands to detect
    pub num_hands: u8,
    /// Minimum detection confidence
    pub min_detection_confidence: f32,
    /// Minimum tracking confidence
    pub min_tracking_confidence: f32,
    /// Output directory
    pub output_dir: PathBuf,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            num_hands: 2,
            min_detection_confidence: 0.5,
            min_tracking_confidence: 0.5,
            output_dir: PathBuf::from("output/gestures"),
        }
    }
}

/// Recognized gesture
#[derive(Debug, Clone)]
pub struct GestureResult {
    /// Frame number
    pub frame: u32,
    /// Timestamp
    pub timestamp: f64,
    /// Detected gestures per hand
    pub gestures: Vec<DetectedGesture>,
    /// Hand landmarks
    pub hand_landmarks: Vec<Vec<Landmark3D>>,
}

/// Single detected gesture
#[derive(Debug, Clone)]
pub struct DetectedGesture {
    /// Gesture category name
    pub category: String,
    /// Confidence score
    pub score: f32,
    /// Hand index (0 or 1)
    pub hand_index: u8,
    /// Handedness (Left/Right)
    pub handedness: String,
}

/// Standard gesture categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GestureCategory {
    /// No gesture detected
    None,
    /// Closed fist
    ClosedFist,
    /// Open palm
    OpenPalm,
    /// Pointing up
    PointingUp,
    /// Thumbs down
    ThumbDown,
    /// Thumbs up
    ThumbUp,
    /// Victory/Peace sign
    Victory,
    /// I Love You sign
    ILoveYou,
}

impl GestureCategory {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "closed_fist" => GestureCategory::ClosedFist,
            "open_palm" => GestureCategory::OpenPalm,
            "pointing_up" => GestureCategory::PointingUp,
            "thumb_down" => GestureCategory::ThumbDown,
            "thumb_up" => GestureCategory::ThumbUp,
            "victory" => GestureCategory::Victory,
            "iloveyou" => GestureCategory::ILoveYou,
            _ => GestureCategory::None,
        }
    }
}

impl GestureRecognizer {
    /// Create a new Gesture Recognizer
    pub fn new(config: GestureConfig) -> Result<Self> {
        let python_path = which::which("python3")
            .map_err(|_| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: "https://www.python.org/downloads/".to_string(),
            })?;

        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self { config, python_path })
    }

    /// Recognize gestures from video
    pub fn process_video(&self, video_path: &Path) -> Result<Vec<GestureResult>> {
        let script = self.generate_script(video_path)?;
        self.run_script(&script)
    }

    fn generate_script(&self, video_path: &Path) -> Result<String> {
        Ok(format!(r#"
import mediapipe as mp
import cv2
import json

# Use the Tasks API for gesture recognition
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download model if needed
import urllib.request
import os

model_path = '/tmp/gesture_recognizer.task'
if not os.path.exists(model_path):
    print("Downloading gesture recognition model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands={num_hands},
    min_hand_detection_confidence={det_conf},
    min_tracking_confidence={track_conf}
)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture('{video_path}')
fps = cap.get(cv2.CAP_PROP_FPS)
results_list = []
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    results = recognizer.recognize(mp_image)

    frame_data = {{
        'frame': frame_num,
        'timestamp': frame_num / fps,
        'gestures': [],
        'hand_landmarks': []
    }}

    if results.gestures:
        for idx, gesture_list in enumerate(results.gestures):
            if gesture_list:
                top_gesture = gesture_list[0]
                handedness = results.handedness[idx][0].category_name if results.handedness else 'Unknown'
                frame_data['gestures'].append({{
                    'category': top_gesture.category_name,
                    'score': float(top_gesture.score),
                    'hand_index': idx,
                    'handedness': handedness
                }})

    if results.hand_landmarks:
        for hand_lms in results.hand_landmarks:
            frame_data['hand_landmarks'].append([
                {{'x': lm.x, 'y': lm.y, 'z': lm.z}}
                for lm in hand_lms
            ])

    results_list.append(frame_data)
    frame_num += 1

cap.release()

print('RESULT_JSON:' + json.dumps(results_list))
"#,
            num_hands = self.config.num_hands,
            det_conf = self.config.min_detection_confidence,
            track_conf = self.config.min_tracking_confidence,
            video_path = video_path.display(),
        ))
    }

    fn run_script(&self, script: &str) -> Result<Vec<GestureResult>> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mediapipe_gesture.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: Vec<serde_json::Value> = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(data.iter().map(|f| self.parse_frame(f)).collect())
    }

    fn parse_frame(&self, data: &serde_json::Value) -> GestureResult {
        let gestures = data["gestures"].as_array()
            .map(|arr| arr.iter().map(|g| DetectedGesture {
                category: g["category"].as_str().unwrap_or("None").to_string(),
                score: g["score"].as_f64().unwrap_or(0.0) as f32,
                hand_index: g["hand_index"].as_u64().unwrap_or(0) as u8,
                handedness: g["handedness"].as_str().unwrap_or("Unknown").to_string(),
            }).collect())
            .unwrap_or_default();

        let hand_landmarks = data["hand_landmarks"].as_array()
            .map(|hands| hands.iter().map(|h| {
                h.as_array().map(|lms| lms.iter().map(|l| Landmark3D {
                    x: l["x"].as_f64().unwrap_or(0.0) as f32,
                    y: l["y"].as_f64().unwrap_or(0.0) as f32,
                    z: l["z"].as_f64().unwrap_or(0.0) as f32,
                    visibility: 1.0,
                }).collect()).unwrap_or_default()
            }).collect())
            .unwrap_or_default();

        GestureResult {
            frame: data["frame"].as_u64().unwrap_or(0) as u32,
            timestamp: data["timestamp"].as_f64().unwrap_or(0.0),
            gestures,
            hand_landmarks,
        }
    }
}

// =============================================================================
// FACE BLENDSHAPES - Expression Analysis
// =============================================================================

/// MediaPipe Face Landmarker with Blendshapes
#[derive(Debug)]
pub struct FaceLandmarker {
    config: FaceLandmarkerConfig,
    python_path: PathBuf,
}

/// Configuration for Face Landmarker
#[derive(Debug, Clone)]
pub struct FaceLandmarkerConfig {
    /// Number of faces to detect
    pub num_faces: u8,
    /// Minimum detection confidence
    pub min_detection_confidence: f32,
    /// Minimum tracking confidence
    pub min_tracking_confidence: f32,
    /// Output blendshapes (52 expression coefficients)
    pub output_blendshapes: bool,
    /// Output face transformation matrix
    pub output_face_transform: bool,
    /// Output directory
    pub output_dir: PathBuf,
}

impl Default for FaceLandmarkerConfig {
    fn default() -> Self {
        Self {
            num_faces: 1,
            min_detection_confidence: 0.5,
            min_tracking_confidence: 0.5,
            output_blendshapes: true,
            output_face_transform: true,
            output_dir: PathBuf::from("output/face"),
        }
    }
}

/// Face landmarker result with blendshapes
#[derive(Debug, Clone)]
pub struct FaceLandmarkerResult {
    /// Frame number
    pub frame: u32,
    /// Timestamp
    pub timestamp: f64,
    /// Face landmarks (478 points with iris)
    pub landmarks: Vec<Landmark3D>,
    /// Blendshape coefficients (52 values)
    pub blendshapes: Option<FaceBlendshapes>,
    /// Face transformation matrix
    pub face_transform: Option<[[f32; 4]; 4]>,
}

/// 52 ARKit-compatible face blendshapes
#[derive(Debug, Clone, Default)]
pub struct FaceBlendshapes {
    // Eye blendshapes
    pub eye_blink_left: f32,
    pub eye_blink_right: f32,
    pub eye_look_down_left: f32,
    pub eye_look_down_right: f32,
    pub eye_look_in_left: f32,
    pub eye_look_in_right: f32,
    pub eye_look_out_left: f32,
    pub eye_look_out_right: f32,
    pub eye_look_up_left: f32,
    pub eye_look_up_right: f32,
    pub eye_squint_left: f32,
    pub eye_squint_right: f32,
    pub eye_wide_left: f32,
    pub eye_wide_right: f32,

    // Brow blendshapes
    pub brow_down_left: f32,
    pub brow_down_right: f32,
    pub brow_inner_up: f32,
    pub brow_outer_up_left: f32,
    pub brow_outer_up_right: f32,

    // Nose blendshapes
    pub nose_sneer_left: f32,
    pub nose_sneer_right: f32,

    // Cheek blendshapes
    pub cheek_puff: f32,
    pub cheek_squint_left: f32,
    pub cheek_squint_right: f32,

    // Jaw blendshapes
    pub jaw_forward: f32,
    pub jaw_left: f32,
    pub jaw_right: f32,
    pub jaw_open: f32,

    // Mouth blendshapes
    pub mouth_close: f32,
    pub mouth_funnel: f32,
    pub mouth_pucker: f32,
    pub mouth_left: f32,
    pub mouth_right: f32,
    pub mouth_smile_left: f32,
    pub mouth_smile_right: f32,
    pub mouth_frown_left: f32,
    pub mouth_frown_right: f32,
    pub mouth_dimple_left: f32,
    pub mouth_dimple_right: f32,
    pub mouth_stretch_left: f32,
    pub mouth_stretch_right: f32,
    pub mouth_roll_lower: f32,
    pub mouth_roll_upper: f32,
    pub mouth_shrug_lower: f32,
    pub mouth_shrug_upper: f32,
    pub mouth_press_left: f32,
    pub mouth_press_right: f32,
    pub mouth_lower_down_left: f32,
    pub mouth_lower_down_right: f32,
    pub mouth_upper_up_left: f32,
    pub mouth_upper_up_right: f32,
}

impl FaceBlendshapes {
    /// Calculate overall expressiveness score (0-1)
    /// Low scores may indicate hypomimia (reduced facial expression)
    pub fn expressiveness_score(&self) -> f32 {
        let values = [
            self.eye_blink_left, self.eye_blink_right,
            self.brow_inner_up, self.brow_down_left, self.brow_down_right,
            self.mouth_smile_left, self.mouth_smile_right,
            self.mouth_frown_left, self.mouth_frown_right,
            self.cheek_puff, self.jaw_open,
        ];

        let sum: f32 = values.iter().map(|v| v.abs()).sum();
        (sum / values.len() as f32).min(1.0)
    }

    /// Detect potential hypomimia (masked face in Parkinson's)
    pub fn hypomimia_score(&self) -> f32 {
        // Low expressiveness = high hypomimia score
        1.0 - self.expressiveness_score()
    }

    /// Calculate smile asymmetry (0 = symmetric, 1 = fully asymmetric)
    pub fn smile_asymmetry(&self) -> f32 {
        let total = self.mouth_smile_left + self.mouth_smile_right;
        if total > 0.01 {
            (self.mouth_smile_left - self.mouth_smile_right).abs() / total
        } else {
            0.0
        }
    }

    /// Calculate blink rate from sequence of blendshapes
    pub fn is_blinking(&self, threshold: f32) -> bool {
        self.eye_blink_left > threshold || self.eye_blink_right > threshold
    }
}

impl FaceLandmarker {
    /// Create a new Face Landmarker
    pub fn new(config: FaceLandmarkerConfig) -> Result<Self> {
        let python_path = which::which("python3")
            .map_err(|_| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: "https://www.python.org/downloads/".to_string(),
            })?;

        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self { config, python_path })
    }

    /// Process video for face landmarks and blendshapes
    pub fn process_video(&self, video_path: &Path) -> Result<Vec<FaceLandmarkerResult>> {
        let script = self.generate_script(video_path)?;
        self.run_script(&script)
    }

    fn generate_script(&self, video_path: &Path) -> Result<String> {
        Ok(format!(r#"
import mediapipe as mp
import cv2
import json
import urllib.request
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '/tmp/face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces={num_faces},
    min_face_detection_confidence={det_conf},
    min_tracking_confidence={track_conf},
    output_face_blendshapes={blendshapes},
    output_facial_transformation_matrixes={transform}
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture('{video_path}')
fps = cap.get(cv2.CAP_PROP_FPS)
results_list = []
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    results = landmarker.detect(mp_image)

    if results.face_landmarks:
        landmarks = [
            {{'x': lm.x, 'y': lm.y, 'z': lm.z}}
            for lm in results.face_landmarks[0]
        ]

        blendshapes = None
        if results.face_blendshapes:
            blendshapes = {{
                bs.category_name: float(bs.score)
                for bs in results.face_blendshapes[0]
            }}

        transform = None
        if results.facial_transformation_matrixes:
            transform = results.facial_transformation_matrixes[0].tolist()

        results_list.append({{
            'frame': frame_num,
            'timestamp': frame_num / fps,
            'landmarks': landmarks,
            'blendshapes': blendshapes,
            'face_transform': transform
        }})

    frame_num += 1

cap.release()

print('RESULT_JSON:' + json.dumps(results_list))
"#,
            num_faces = self.config.num_faces,
            det_conf = self.config.min_detection_confidence,
            track_conf = self.config.min_tracking_confidence,
            blendshapes = self.config.output_blendshapes,
            transform = self.config.output_face_transform,
            video_path = video_path.display(),
        ))
    }

    fn run_script(&self, script: &str) -> Result<Vec<FaceLandmarkerResult>> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mediapipe_face_landmarker.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: Vec<serde_json::Value> = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(data.iter().map(|f| self.parse_frame(f)).collect())
    }

    fn parse_frame(&self, data: &serde_json::Value) -> FaceLandmarkerResult {
        let landmarks = data["landmarks"].as_array()
            .map(|arr| arr.iter().map(|l| Landmark3D {
                x: l["x"].as_f64().unwrap_or(0.0) as f32,
                y: l["y"].as_f64().unwrap_or(0.0) as f32,
                z: l["z"].as_f64().unwrap_or(0.0) as f32,
                visibility: 1.0,
            }).collect())
            .unwrap_or_default();

        let blendshapes = data["blendshapes"].as_object().map(|bs| {
            let get = |key: &str| -> f32 {
                bs.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32
            };

            FaceBlendshapes {
                eye_blink_left: get("eyeBlinkLeft"),
                eye_blink_right: get("eyeBlinkRight"),
                eye_look_down_left: get("eyeLookDownLeft"),
                eye_look_down_right: get("eyeLookDownRight"),
                eye_look_in_left: get("eyeLookInLeft"),
                eye_look_in_right: get("eyeLookInRight"),
                eye_look_out_left: get("eyeLookOutLeft"),
                eye_look_out_right: get("eyeLookOutRight"),
                eye_look_up_left: get("eyeLookUpLeft"),
                eye_look_up_right: get("eyeLookUpRight"),
                eye_squint_left: get("eyeSquintLeft"),
                eye_squint_right: get("eyeSquintRight"),
                eye_wide_left: get("eyeWideLeft"),
                eye_wide_right: get("eyeWideRight"),
                brow_down_left: get("browDownLeft"),
                brow_down_right: get("browDownRight"),
                brow_inner_up: get("browInnerUp"),
                brow_outer_up_left: get("browOuterUpLeft"),
                brow_outer_up_right: get("browOuterUpRight"),
                nose_sneer_left: get("noseSneerLeft"),
                nose_sneer_right: get("noseSneerRight"),
                cheek_puff: get("cheekPuff"),
                cheek_squint_left: get("cheekSquintLeft"),
                cheek_squint_right: get("cheekSquintRight"),
                jaw_forward: get("jawForward"),
                jaw_left: get("jawLeft"),
                jaw_right: get("jawRight"),
                jaw_open: get("jawOpen"),
                mouth_close: get("mouthClose"),
                mouth_funnel: get("mouthFunnel"),
                mouth_pucker: get("mouthPucker"),
                mouth_left: get("mouthLeft"),
                mouth_right: get("mouthRight"),
                mouth_smile_left: get("mouthSmileLeft"),
                mouth_smile_right: get("mouthSmileRight"),
                mouth_frown_left: get("mouthFrownLeft"),
                mouth_frown_right: get("mouthFrownRight"),
                mouth_dimple_left: get("mouthDimpleLeft"),
                mouth_dimple_right: get("mouthDimpleRight"),
                mouth_stretch_left: get("mouthStretchLeft"),
                mouth_stretch_right: get("mouthStretchRight"),
                mouth_roll_lower: get("mouthRollLower"),
                mouth_roll_upper: get("mouthRollUpper"),
                mouth_shrug_lower: get("mouthShrugLower"),
                mouth_shrug_upper: get("mouthShrugUpper"),
                mouth_press_left: get("mouthPressLeft"),
                mouth_press_right: get("mouthPressRight"),
                mouth_lower_down_left: get("mouthLowerDownLeft"),
                mouth_lower_down_right: get("mouthLowerDownRight"),
                mouth_upper_up_left: get("mouthUpperUpLeft"),
                mouth_upper_up_right: get("mouthUpperUpRight"),
            }
        });

        FaceLandmarkerResult {
            frame: data["frame"].as_u64().unwrap_or(0) as u32,
            timestamp: data["timestamp"].as_f64().unwrap_or(0.0),
            landmarks,
            blendshapes,
            face_transform: None,
        }
    }
}

// =============================================================================
// IMAGE SEGMENTATION
// =============================================================================

/// MediaPipe Image Segmenter
#[derive(Debug)]
pub struct ImageSegmenter {
    config: SegmenterConfig,
    python_path: PathBuf,
}

/// Configuration for Image Segmentation
#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    /// Segmentation model type
    pub model_type: SegmentationModel,
    /// Output category mask
    pub output_category_mask: bool,
    /// Output confidence masks
    pub output_confidence_mask: bool,
    /// Output directory
    pub output_dir: PathBuf,
}

/// Segmentation model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentationModel {
    /// Selfie segmentation (person vs background)
    SelfieSegmenter,
    /// Multi-class segmentation
    DeepLabV3,
    /// Hair segmentation
    HairSegmenter,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            model_type: SegmentationModel::SelfieSegmenter,
            output_category_mask: true,
            output_confidence_mask: true,
            output_dir: PathBuf::from("output/segmentation"),
        }
    }
}

/// Segmentation result
#[derive(Debug, Clone)]
pub struct SegmentationResult {
    /// Frame number
    pub frame: u32,
    /// Timestamp
    pub timestamp: f64,
    /// Category mask (class ID per pixel)
    pub category_mask: Option<Vec<Vec<u8>>>,
    /// Confidence mask (probability per pixel)
    pub confidence_mask: Option<Vec<Vec<f32>>>,
    /// Image dimensions (width, height)
    pub dimensions: (u32, u32),
}

impl ImageSegmenter {
    /// Create a new Image Segmenter
    pub fn new(config: SegmenterConfig) -> Result<Self> {
        let python_path = which::which("python3")
            .map_err(|_| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: "https://www.python.org/downloads/".to_string(),
            })?;

        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self { config, python_path })
    }

    /// Segment video frames
    pub fn process_video(&self, video_path: &Path) -> Result<Vec<SegmentationResult>> {
        let script = self.generate_script(video_path)?;
        self.run_script(&script)
    }

    fn generate_script(&self, video_path: &Path) -> Result<String> {
        let model_url = match self.config.model_type {
            SegmentationModel::SelfieSegmenter =>
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            SegmentationModel::DeepLabV3 =>
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite",
            SegmentationModel::HairSegmenter =>
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
        };

        Ok(format!(r#"
import mediapipe as mp
import cv2
import json
import numpy as np
import urllib.request
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '/tmp/segmenter.tflite'
if not os.path.exists(model_path):
    print("Downloading segmentation model...")
    urllib.request.urlretrieve('{model_url}', model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ImageSegmenterOptions(
    base_options=base_options,
    output_category_mask={category_mask},
    output_confidence_masks={confidence_mask}
)
segmenter = vision.ImageSegmenter.create_from_options(options)

cap = cv2.VideoCapture('{video_path}')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
results_list = []
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    results = segmenter.segment(mp_image)

    frame_data = {{
        'frame': frame_num,
        'timestamp': frame_num / fps,
        'dimensions': [width, height],
        'category_mask': None,
        'confidence_mask': None
    }}

    # Subsample masks to reduce data size
    if results.category_mask:
        mask = results.category_mask.numpy_view()
        # Downsample by factor of 4
        downsampled = mask[::4, ::4]
        frame_data['category_mask'] = downsampled.tolist()

    if results.confidence_masks:
        mask = results.confidence_masks[0].numpy_view()
        downsampled = mask[::4, ::4]
        frame_data['confidence_mask'] = downsampled.tolist()

    results_list.append(frame_data)
    frame_num += 1

    # Only process every 5th frame for efficiency
    if frame_num % 5 != 0:
        cap.read()  # Skip frames
        frame_num += 1

cap.release()

print('RESULT_JSON:' + json.dumps(results_list[:100]))  # Limit output size
"#,
            model_url = model_url,
            category_mask = self.config.output_category_mask,
            confidence_mask = self.config.output_confidence_mask,
            video_path = video_path.display(),
        ))
    }

    fn run_script(&self, script: &str) -> Result<Vec<SegmentationResult>> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mediapipe_segment.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: Vec<serde_json::Value> = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(data.iter().map(|f| SegmentationResult {
            frame: f["frame"].as_u64().unwrap_or(0) as u32,
            timestamp: f["timestamp"].as_f64().unwrap_or(0.0),
            category_mask: None, // Would need to parse arrays
            confidence_mask: None,
            dimensions: (
                f["dimensions"][0].as_u64().unwrap_or(640) as u32,
                f["dimensions"][1].as_u64().unwrap_or(480) as u32,
            ),
        }).collect())
    }
}

// =============================================================================
// AUDIO CLASSIFICATION
// =============================================================================

/// MediaPipe Audio Classifier
#[derive(Debug)]
pub struct AudioClassifier {
    config: AudioClassifierConfig,
    python_path: PathBuf,
}

/// Configuration for Audio Classification
#[derive(Debug, Clone)]
pub struct AudioClassifierConfig {
    /// Maximum number of results to return
    pub max_results: u32,
    /// Minimum score threshold
    pub score_threshold: f32,
    /// Output directory
    pub output_dir: PathBuf,
}

impl Default for AudioClassifierConfig {
    fn default() -> Self {
        Self {
            max_results: 5,
            score_threshold: 0.2,
            output_dir: PathBuf::from("output/audio"),
        }
    }
}

/// Audio classification result
#[derive(Debug, Clone)]
pub struct AudioClassificationResult {
    /// Timestamp in seconds
    pub timestamp: f64,
    /// Duration of analyzed segment
    pub duration: f64,
    /// Classified categories with scores
    pub classifications: Vec<AudioCategory>,
}

/// Classified audio category
#[derive(Debug, Clone)]
pub struct AudioCategory {
    /// Category name (e.g., "Speech", "Music", "Cough")
    pub category: String,
    /// Confidence score
    pub score: f32,
    /// Category index
    pub index: u32,
}

/// Common audio categories for biosignal analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiosignalAudioCategory {
    Speech,
    Silence,
    Cough,
    Sneeze,
    Breathing,
    Laughter,
    Crying,
    Music,
    Noise,
}

impl BiosignalAudioCategory {
    fn from_yamnet_label(label: &str) -> Option<Self> {
        let lower = label.to_lowercase();
        if lower.contains("speech") || lower.contains("talking") {
            Some(BiosignalAudioCategory::Speech)
        } else if lower.contains("silence") {
            Some(BiosignalAudioCategory::Silence)
        } else if lower.contains("cough") {
            Some(BiosignalAudioCategory::Cough)
        } else if lower.contains("sneeze") {
            Some(BiosignalAudioCategory::Sneeze)
        } else if lower.contains("breath") {
            Some(BiosignalAudioCategory::Breathing)
        } else if lower.contains("laugh") {
            Some(BiosignalAudioCategory::Laughter)
        } else if lower.contains("cry") || lower.contains("sob") {
            Some(BiosignalAudioCategory::Crying)
        } else if lower.contains("music") || lower.contains("sing") {
            Some(BiosignalAudioCategory::Music)
        } else {
            None
        }
    }
}

impl AudioClassifier {
    /// Create a new Audio Classifier
    pub fn new(config: AudioClassifierConfig) -> Result<Self> {
        let python_path = which::which("python3")
            .map_err(|_| MediaError::ToolNotFound {
                tool: "python3".to_string(),
                install_url: "https://www.python.org/downloads/".to_string(),
            })?;

        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self { config, python_path })
    }

    /// Classify audio from file
    pub fn classify_audio(&self, audio_path: &Path) -> Result<Vec<AudioClassificationResult>> {
        let script = self.generate_script(audio_path)?;
        self.run_script(&script)
    }

    /// Extract audio embeddings for downstream tasks
    pub fn extract_embeddings(&self, audio_path: &Path) -> Result<Vec<Vec<f32>>> {
        let script = self.generate_embedding_script(audio_path)?;
        self.run_embedding_script(&script)
    }

    fn generate_script(&self, audio_path: &Path) -> Result<String> {
        Ok(format!(r#"
import json
import urllib.request
import os
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import audio
from mediapipe.tasks.python.components import containers

model_path = '/tmp/audio_classifier.tflite'
if not os.path.exists(model_path):
    print("Downloading audio classifier model (YAMNet)...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite',
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = audio.AudioClassifierOptions(
    base_options=base_options,
    max_results={max_results},
    score_threshold={score_threshold}
)
classifier = audio.AudioClassifier.create_from_options(options)

# Load audio file
import scipy.io.wavfile as wav
try:
    sample_rate, audio_data = wav.read('{audio_path}')
except:
    # Try with librosa for other formats
    import librosa
    audio_data, sample_rate = librosa.load('{audio_path}', sr=16000)
    audio_data = (audio_data * 32767).astype(np.int16)

# Ensure mono
if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1).astype(np.int16)

# Create AudioData
audio_clip = containers.AudioData.create_from_array(
    audio_data.astype(np.float32) / 32768.0,
    sample_rate
)

# Classify in segments
segment_duration = 0.975  # YAMNet default
results_list = []

result = classifier.classify(audio_clip)

for idx, classification_result in enumerate(result.classifications):
    timestamp = idx * segment_duration
    classifications = []

    for category in classification_result.categories:
        classifications.append({{
            'category': category.category_name,
            'score': float(category.score),
            'index': category.index
        }})

    results_list.append({{
        'timestamp': timestamp,
        'duration': segment_duration,
        'classifications': classifications
    }})

print('RESULT_JSON:' + json.dumps(results_list))
"#,
            max_results = self.config.max_results,
            score_threshold = self.config.score_threshold,
            audio_path = audio_path.display(),
        ))
    }

    fn generate_embedding_script(&self, audio_path: &Path) -> Result<String> {
        Ok(format!(r#"
import json
import urllib.request
import os
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import audio
from mediapipe.tasks.python.components import containers

model_path = '/tmp/audio_embedder.tflite'
if not os.path.exists(model_path):
    print("Downloading audio embedder model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/audio_embedder/yamnet/float32/1/yamnet.tflite',
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = audio.AudioEmbedderOptions(base_options=base_options)
embedder = audio.AudioEmbedder.create_from_options(options)

# Load audio
import scipy.io.wavfile as wav
try:
    sample_rate, audio_data = wav.read('{audio_path}')
except:
    import librosa
    audio_data, sample_rate = librosa.load('{audio_path}', sr=16000)
    audio_data = (audio_data * 32767).astype(np.int16)

if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1).astype(np.int16)

audio_clip = containers.AudioData.create_from_array(
    audio_data.astype(np.float32) / 32768.0,
    sample_rate
)

result = embedder.embed(audio_clip)

embeddings = []
for embedding_result in result.embeddings:
    embeddings.append(embedding_result.embedding.tolist())

print('RESULT_JSON:' + json.dumps(embeddings))
"#,
            audio_path = audio_path.display(),
        ))
    }

    fn run_script(&self, script: &str) -> Result<Vec<AudioClassificationResult>> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mediapipe_audio_classify.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: Vec<serde_json::Value> = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(data.iter().map(|r| {
            let classifications = r["classifications"].as_array()
                .map(|arr| arr.iter().map(|c| AudioCategory {
                    category: c["category"].as_str().unwrap_or("Unknown").to_string(),
                    score: c["score"].as_f64().unwrap_or(0.0) as f32,
                    index: c["index"].as_u64().unwrap_or(0) as u32,
                }).collect())
                .unwrap_or_default();

            AudioClassificationResult {
                timestamp: r["timestamp"].as_f64().unwrap_or(0.0),
                duration: r["duration"].as_f64().unwrap_or(0.975),
                classifications,
            }
        }).collect())
    }

    fn run_embedding_script(&self, script: &str) -> Result<Vec<Vec<f32>>> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mediapipe_audio_embed.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout.lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let data: Vec<Vec<f64>> = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        Ok(data.iter()
            .map(|emb| emb.iter().map(|v| *v as f32).collect())
            .collect())
    }
}

// =============================================================================
// BIOSIGNAL ANALYSIS UTILITIES
// =============================================================================

/// Biosignal analysis from MediaPipe outputs
pub mod biosignal_analysis {
    use super::*;

    /// Tremor analysis result
    #[derive(Debug, Clone)]
    pub struct TremorAnalysis {
        /// Detected tremor frequency (Hz)
        pub frequency: Option<f32>,
        /// Tremor amplitude (normalized units)
        pub amplitude: f32,
        /// Regularity score (0-1, higher = more regular)
        pub regularity: f32,
        /// Affected body part
        pub location: TremorLocation,
        /// Confidence in detection
        pub confidence: f32,
    }

    /// Location of detected tremor
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TremorLocation {
        LeftHand,
        RightHand,
        Head,
        Jaw,
        LeftLeg,
        RightLeg,
        Trunk,
    }

    /// Detect tremor from hand landmark sequence
    pub fn detect_hand_tremor(
        landmarks: &[Vec<Landmark3D>],
        sample_rate: f32,
    ) -> TremorAnalysis {
        if landmarks.is_empty() || landmarks[0].is_empty() {
            return TremorAnalysis {
                frequency: None,
                amplitude: 0.0,
                regularity: 0.0,
                location: TremorLocation::RightHand,
                confidence: 0.0,
            };
        }

        // Extract index fingertip trajectory (landmark 8)
        let fingertip_idx = 8;
        let trajectory: Vec<(f32, f32, f32)> = landmarks.iter()
            .filter_map(|frame| {
                frame.get(fingertip_idx).map(|l| (l.x, l.y, l.z))
            })
            .collect();

        if trajectory.len() < 10 {
            return TremorAnalysis {
                frequency: None,
                amplitude: 0.0,
                regularity: 0.0,
                location: TremorLocation::RightHand,
                confidence: 0.0,
            };
        }

        // Compute velocity magnitude
        let velocities: Vec<f32> = trajectory.windows(2)
            .map(|w| {
                let dx = w[1].0 - w[0].0;
                let dy = w[1].1 - w[0].1;
                let dz = w[1].2 - w[0].2;
                (dx * dx + dy * dy + dz * dz).sqrt() * sample_rate
            })
            .collect();

        // Estimate amplitude from velocity variance
        let mean_vel: f32 = velocities.iter().sum::<f32>() / velocities.len() as f32;
        let variance: f32 = velocities.iter()
            .map(|v| (v - mean_vel).powi(2))
            .sum::<f32>() / velocities.len() as f32;
        let amplitude = variance.sqrt();

        // Simple frequency estimation via zero crossings
        let detrended: Vec<f32> = velocities.iter()
            .map(|v| v - mean_vel)
            .collect();

        let zero_crossings = detrended.windows(2)
            .filter(|w| w[0] * w[1] < 0.0)
            .count();

        let duration = trajectory.len() as f32 / sample_rate;
        let frequency = if duration > 0.5 {
            Some(zero_crossings as f32 / (2.0 * duration))
        } else {
            None
        };

        // Confidence based on signal strength
        let confidence = (amplitude * 10.0).min(1.0);

        TremorAnalysis {
            frequency,
            amplitude,
            regularity: 0.5, // Would need FFT for proper regularity
            location: TremorLocation::RightHand,
            confidence,
        }
    }

    /// Blink analysis result
    #[derive(Debug, Clone)]
    pub struct BlinkAnalysis {
        /// Blink rate (blinks per minute)
        pub blink_rate: f32,
        /// Average blink duration (seconds)
        pub avg_duration: f32,
        /// Blink regularity (coefficient of variation)
        pub regularity_cv: f32,
        /// Number of blinks detected
        pub blink_count: u32,
        /// Timestamps of detected blinks
        pub blink_times: Vec<f64>,
    }

    /// Analyze blink patterns from face blendshapes
    pub fn analyze_blinks(
        blendshapes: &[FaceBlendshapes],
        timestamps: &[f64],
        threshold: f32,
    ) -> BlinkAnalysis {
        if blendshapes.is_empty() || timestamps.is_empty() {
            return BlinkAnalysis {
                blink_rate: 0.0,
                avg_duration: 0.0,
                regularity_cv: 0.0,
                blink_count: 0,
                blink_times: Vec::new(),
            };
        }

        let mut blink_times = Vec::new();
        let mut blink_durations = Vec::new();
        let mut in_blink = false;
        let mut blink_start = 0.0;

        for (bs, &t) in blendshapes.iter().zip(timestamps) {
            let is_blink = bs.is_blinking(threshold);

            if is_blink && !in_blink {
                // Blink started
                in_blink = true;
                blink_start = t;
                blink_times.push(t);
            } else if !is_blink && in_blink {
                // Blink ended
                in_blink = false;
                blink_durations.push((t - blink_start) as f32);
            }
        }

        let blink_count = blink_times.len() as u32;
        let duration = timestamps.last().unwrap_or(&0.0) - timestamps.first().unwrap_or(&0.0);

        let blink_rate = if duration > 0.0 {
            (blink_count as f64 / duration * 60.0) as f32
        } else {
            0.0
        };

        let avg_duration = if !blink_durations.is_empty() {
            blink_durations.iter().sum::<f32>() / blink_durations.len() as f32
        } else {
            0.0
        };

        // Calculate coefficient of variation for inter-blink intervals
        let intervals: Vec<f64> = blink_times.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let regularity_cv = if intervals.len() > 1 {
            let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let variance = intervals.iter()
                .map(|i| (i - mean).powi(2))
                .sum::<f64>() / intervals.len() as f64;
            (variance.sqrt() / mean) as f32
        } else {
            0.0
        };

        BlinkAnalysis {
            blink_rate,
            avg_duration,
            regularity_cv,
            blink_count,
            blink_times,
        }
    }

    /// Hypomimia (masked face) analysis
    #[derive(Debug, Clone)]
    pub struct HypomimiaAnalysis {
        /// Overall hypomimia severity (0-1)
        pub severity: f32,
        /// Reduced facial expressiveness score
        pub expressiveness: f32,
        /// Smile asymmetry
        pub smile_asymmetry: f32,
        /// Reduced blink rate indicator
        pub reduced_blinking: bool,
        /// Spontaneous expression count
        pub expression_count: u32,
    }

    /// Analyze facial expression for hypomimia
    pub fn analyze_hypomimia(
        blendshapes: &[FaceBlendshapes],
        duration_seconds: f64,
    ) -> HypomimiaAnalysis {
        if blendshapes.is_empty() {
            return HypomimiaAnalysis {
                severity: 0.0,
                expressiveness: 1.0,
                smile_asymmetry: 0.0,
                reduced_blinking: false,
                expression_count: 0,
            };
        }

        // Calculate mean expressiveness
        let expressiveness: f32 = blendshapes.iter()
            .map(|bs| bs.expressiveness_score())
            .sum::<f32>() / blendshapes.len() as f32;

        // Calculate mean smile asymmetry
        let smile_asymmetry: f32 = blendshapes.iter()
            .map(|bs| bs.smile_asymmetry())
            .sum::<f32>() / blendshapes.len() as f32;

        // Count significant expressions (threshold > 0.3)
        let expression_count = blendshapes.iter()
            .filter(|bs| bs.expressiveness_score() > 0.3)
            .count() as u32;

        // Check blink rate
        let blinks = blendshapes.iter()
            .filter(|bs| bs.is_blinking(0.5))
            .count();
        let blink_rate = if duration_seconds > 0.0 {
            (blinks as f64 / duration_seconds * 60.0) as f32
        } else {
            0.0
        };

        // Normal blink rate is 15-20 per minute
        let reduced_blinking = blink_rate < 10.0;

        // Calculate severity
        let severity = (1.0 - expressiveness) * 0.5
            + smile_asymmetry * 0.2
            + if reduced_blinking { 0.3 } else { 0.0 };

        HypomimiaAnalysis {
            severity: severity.min(1.0),
            expressiveness,
            smile_asymmetry,
            reduced_blinking,
            expression_count,
        }
    }

    /// Voice activity detection result
    #[derive(Debug, Clone)]
    pub struct VoiceActivityResult {
        /// Speech segments (start_time, end_time)
        pub speech_segments: Vec<(f64, f64)>,
        /// Total speech duration (seconds)
        pub total_speech_duration: f64,
        /// Speech ratio (speech time / total time)
        pub speech_ratio: f32,
        /// Number of speech segments
        pub segment_count: u32,
        /// Average segment duration
        pub avg_segment_duration: f32,
    }

    /// Detect voice activity from audio classifications
    pub fn detect_voice_activity(
        classifications: &[AudioClassificationResult],
    ) -> VoiceActivityResult {
        let mut speech_segments = Vec::new();
        let mut in_speech = false;
        let mut speech_start = 0.0;

        for result in classifications {
            let is_speech = result.classifications.iter()
                .any(|c| {
                    let lower = c.category.to_lowercase();
                    (lower.contains("speech") || lower.contains("talk")) && c.score > 0.3
                });

            if is_speech && !in_speech {
                in_speech = true;
                speech_start = result.timestamp;
            } else if !is_speech && in_speech {
                in_speech = false;
                speech_segments.push((speech_start, result.timestamp));
            }
        }

        // Close final segment if still in speech
        if in_speech
            && let Some(last) = classifications.last() {
                speech_segments.push((speech_start, last.timestamp + last.duration));
            }

        let total_speech_duration: f64 = speech_segments.iter()
            .map(|(s, e)| e - s)
            .sum();

        let total_duration = classifications.last()
            .map(|c| c.timestamp + c.duration)
            .unwrap_or(0.0);

        let speech_ratio = if total_duration > 0.0 {
            (total_speech_duration / total_duration) as f32
        } else {
            0.0
        };

        let segment_count = speech_segments.len() as u32;

        let avg_segment_duration = if segment_count > 0 {
            total_speech_duration as f32 / segment_count as f32
        } else {
            0.0
        };

        VoiceActivityResult {
            speech_segments,
            total_speech_duration,
            speech_ratio,
            segment_count,
            avg_segment_duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = MediaPipeConfig::default();
        assert!(config.enable_pose);
        assert!(config.enable_hands);
        assert!(!config.enable_face);
    }

    #[test]
    fn test_landmark_indices() {
        assert_eq!(PoseLandmark::Nose as u8, 0);
        assert_eq!(PoseLandmark::LeftAnkle as u8, 27);
        assert_eq!(PoseLandmark::RightFootIndex as u8, 32);
    }
}
