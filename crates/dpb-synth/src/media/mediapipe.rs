//! MediaPipe Pose Extraction Integration
//!
//! This module provides integration with Google MediaPipe for extracting
//! pose, hand, and face landmarks from video - closing the loop from
//! synthetic generation back to biosignal extraction.
//!
//! # Features
//!
//! - 33-point BlazePose body landmarks
//! - 21-point hand landmarks per hand
//! - 468-point face mesh
//! - Real-time processing (30+ FPS)
//! - 3D landmark estimation
//!
//! # Requirements
//!
//! - Python 3.8+
//! - `pip install mediapipe opencv-python`
//!
//! # References
//!
//! - MediaPipe: https://google.github.io/mediapipe/

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

            if let Some(prev_left_forward) = last_left_forward {
                if left_forward != prev_left_forward {
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
