//! OpenPose Multi-Person Pose Estimation Integration
//!
//! OpenPose is the first real-time multi-person system to jointly detect
//! human body, hand, facial, and foot keypoints (135 keypoints total).
//!
//! # Features
//!
//! - 135 keypoints: body (25), hands (2x21), face (70)
//! - Multi-person detection
//! - Real-time performance with GPU
//! - Higher accuracy than MediaPipe for detailed analysis
//!
//! # Requirements
//!
//! - OpenPose installation (build from source)
//! - Python bindings (pyopenpose)
//! - GPU with CUDA support recommended

use super::{MediaError, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// OpenPose pose extractor
#[derive(Debug)]
pub struct OpenPoseExtractor {
    config: OpenPoseConfig,
    python_path: PathBuf,
}

/// Configuration for OpenPose
#[derive(Debug, Clone)]
pub struct OpenPoseConfig {
    /// OpenPose installation directory
    pub openpose_dir: PathBuf,
    /// Output directory
    pub output_dir: PathBuf,
    /// Enable body detection
    pub enable_body: bool,
    /// Enable hand detection
    pub enable_hand: bool,
    /// Enable face detection
    pub enable_face: bool,
    /// Enable foot detection
    pub enable_foot: bool,
    /// GPU device ID (-1 for CPU)
    pub device: i32,
    /// Network resolution (higher = more accurate but slower)
    pub net_resolution: String,
    /// Scale number for multi-scale inference
    pub scale_number: u32,
    /// Render output visualization
    pub render_output: bool,
}

impl Default for OpenPoseConfig {
    fn default() -> Self {
        Self {
            openpose_dir: PathBuf::from("/usr/local/openpose"),
            output_dir: PathBuf::from("output/openpose"),
            enable_body: true,
            enable_hand: true,
            enable_face: true,
            enable_foot: true,
            device: 0,
            net_resolution: "-1x368".to_string(),
            scale_number: 1,
            render_output: true,
        }
    }
}

/// Body keypoint indices (COCO 25-point model)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BodyKeypoint {
    Nose = 0,
    Neck = 1,
    RShoulder = 2,
    RElbow = 3,
    RWrist = 4,
    LShoulder = 5,
    LElbow = 6,
    LWrist = 7,
    MidHip = 8,
    RHip = 9,
    RKnee = 10,
    RAnkle = 11,
    LHip = 12,
    LKnee = 13,
    LAnkle = 14,
    REye = 15,
    LEye = 16,
    REar = 17,
    LEar = 18,
    LBigToe = 19,
    LSmallToe = 20,
    LHeel = 21,
    RBigToe = 22,
    RSmallToe = 23,
    RHeel = 24,
}

/// Hand keypoint indices (21 per hand)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HandKeypoint {
    Wrist = 0,
    Thumb1 = 1,
    Thumb2 = 2,
    Thumb3 = 3,
    ThumbTip = 4,
    Index1 = 5,
    Index2 = 6,
    Index3 = 7,
    IndexTip = 8,
    Middle1 = 9,
    Middle2 = 10,
    Middle3 = 11,
    MiddleTip = 12,
    Ring1 = 13,
    Ring2 = 14,
    Ring3 = 15,
    RingTip = 16,
    Pinky1 = 17,
    Pinky2 = 18,
    Pinky3 = 19,
    PinkyTip = 20,
}

/// Complete pose estimate from OpenPose
#[derive(Debug, Clone)]
pub struct OpenPosePose {
    /// Person ID for multi-person tracking
    pub person_id: u32,
    /// Body keypoints (25 points)
    pub body: Option<Vec<Keypoint>>,
    /// Left hand keypoints (21 points)
    pub left_hand: Option<Vec<Keypoint>>,
    /// Right hand keypoints (21 points)
    pub right_hand: Option<Vec<Keypoint>>,
    /// Face keypoints (70 points)
    pub face: Option<Vec<Keypoint>>,
    /// Overall confidence score
    pub confidence: f32,
}

/// Single keypoint with position and confidence
#[derive(Debug, Clone)]
pub struct Keypoint {
    /// X coordinate (normalized 0-1 or pixels)
    pub x: f32,
    /// Y coordinate (normalized 0-1 or pixels)
    pub y: f32,
    /// Confidence score (0-1)
    pub confidence: f32,
}

impl Keypoint {
    /// Check if keypoint is detected (confidence above threshold)
    pub fn is_detected(&self, threshold: f32) -> bool {
        self.confidence > threshold
    }
}

/// Frame of pose estimates
#[derive(Debug, Clone)]
pub struct OpenPoseFrame {
    /// Frame number
    pub frame_idx: u32,
    /// Timestamp in seconds
    pub timestamp: f64,
    /// Poses for all detected persons
    pub poses: Vec<OpenPosePose>,
    /// Image resolution (width, height)
    pub resolution: (u32, u32),
}

/// Output from OpenPose processing
#[derive(Debug)]
pub struct OpenPoseOutput {
    /// Extracted frames
    pub frames: Vec<OpenPoseFrame>,
    /// Processing time per frame (average)
    pub avg_frame_time: f64,
    /// Total processing time
    pub total_time: f64,
    /// Rendered output video path (if enabled)
    pub render_path: Option<PathBuf>,
    /// JSON output path
    pub json_path: PathBuf,
}

impl OpenPoseExtractor {
    /// Create a new OpenPose extractor
    pub fn new(config: OpenPoseConfig) -> Result<Self> {
        // Check for OpenPose Python bindings
        let check = Command::new("python3")
            .args(["-c", "import pyopenpose as op; print('ok')"])
            .output();

        // Also check for alternative import
        let check_alt = Command::new("python3")
            .args(["-c", "import openpose; print('ok')"])
            .output();

        let available = check.map(|o| o.status.success()).unwrap_or(false)
            || check_alt.map(|o| o.status.success()).unwrap_or(false);

        if !available {
            return Err(MediaError::ToolNotFound {
                tool: "pyopenpose".to_string(),
                install_url: "https://github.com/CMU-Perceptual-Computing-Lab/openpose".to_string(),
            });
        }

        let python_path = which::which("python3").map_err(|_| MediaError::ToolNotFound {
            tool: "python3".to_string(),
            install_url: "https://www.python.org/downloads/".to_string(),
        })?;

        std::fs::create_dir_all(&config.output_dir)?;

        Ok(Self {
            config,
            python_path,
        })
    }

    /// Check if OpenPose is available
    pub fn is_available() -> bool {
        Command::new("python3")
            .args(["-c", "import pyopenpose as op"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
            || Command::new("python3")
                .args(["-c", "import openpose"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
    }

    /// Process a video file
    pub fn process_video(&self, video_path: &Path) -> Result<OpenPoseOutput> {
        let output_json = self.config.output_dir.join(format!(
            "openpose_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));

        let script = self.generate_video_script(video_path, &output_json)?;
        self.run_script(&script, &output_json)
    }

    /// Process a single image
    pub fn process_image(&self, image_path: &Path) -> Result<OpenPoseFrame> {
        let output = self.process_video(image_path)?;
        output
            .frames
            .into_iter()
            .next()
            .ok_or_else(|| MediaError::ExecutionFailed("No poses detected in image".to_string()))
    }

    fn generate_video_script(&self, video_path: &Path, output_json: &Path) -> Result<String> {
        let render_path = if self.config.render_output {
            format!(
                "'{}'",
                self.config.output_dir.join("openpose_render.mp4").display()
            )
        } else {
            "None".to_string()
        };

        Ok(format!(
            r#"
import sys
import cv2
import json
import time
import numpy as np

# Try different OpenPose import methods
try:
    import pyopenpose as op
except ImportError:
    try:
        sys.path.append('{openpose_dir}/python')
        from openpose import pyopenpose as op
    except ImportError:
        print("Error: OpenPose not found")
        sys.exit(1)

# Configure OpenPose
params = dict()
params["model_folder"] = "{openpose_dir}/models/"
params["net_resolution"] = "{net_resolution}"
params["number_people_max"] = 5
params["scale_number"] = {scale_number}

if {enable_hand}:
    params["hand"] = True
if {enable_face}:
    params["face"] = True

# Set GPU
if {device} >= 0:
    params["num_gpu"] = 1
    params["num_gpu_start"] = {device}
else:
    params["num_gpu"] = 0

# Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Open video
cap = cv2.VideoCapture('{video_path}')
if not cap.isOpened():
    print("Error: Cannot open video")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup output video if rendering
render_path = {render_path}
if render_path:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(render_path, fourcc, fps, (width, height))

# Process frames
frames_data = []
frame_times = []
frame_idx = 0

print(f"Processing {{total_frames}} frames...")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.time()

    # Process with OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Extract keypoints
    frame_poses = []

    if datum.poseKeypoints is not None:
        for person_idx in range(datum.poseKeypoints.shape[0]):
            pose = {{
                'person_id': person_idx,
                'body': [],
                'left_hand': None,
                'right_hand': None,
                'face': None,
                'confidence': 0.0
            }}

            # Body keypoints
            body_kps = datum.poseKeypoints[person_idx]
            total_conf = 0
            for kp in body_kps:
                pose['body'].append({{
                    'x': float(kp[0]),
                    'y': float(kp[1]),
                    'confidence': float(kp[2])
                }})
                total_conf += kp[2]
            pose['confidence'] = total_conf / len(body_kps) if body_kps.shape[0] > 0 else 0

            # Hand keypoints
            if {enable_hand} and datum.handKeypoints is not None:
                if datum.handKeypoints[0] is not None and person_idx < datum.handKeypoints[0].shape[0]:
                    pose['left_hand'] = [
                        {{'x': float(kp[0]), 'y': float(kp[1]), 'confidence': float(kp[2])}}
                        for kp in datum.handKeypoints[0][person_idx]
                    ]
                if datum.handKeypoints[1] is not None and person_idx < datum.handKeypoints[1].shape[0]:
                    pose['right_hand'] = [
                        {{'x': float(kp[0]), 'y': float(kp[1]), 'confidence': float(kp[2])}}
                        for kp in datum.handKeypoints[1][person_idx]
                    ]

            # Face keypoints
            if {enable_face} and datum.faceKeypoints is not None:
                if person_idx < datum.faceKeypoints.shape[0]:
                    pose['face'] = [
                        {{'x': float(kp[0]), 'y': float(kp[1]), 'confidence': float(kp[2])}}
                        for kp in datum.faceKeypoints[person_idx]
                    ]

            frame_poses.append(pose)

    timestamp = frame_idx / fps if fps > 0 else 0
    frames_data.append({{
        'frame_idx': frame_idx,
        'timestamp': timestamp,
        'poses': frame_poses,
        'resolution': [width, height]
    }})

    frame_times.append(time.time() - frame_start)

    # Write rendered frame
    if render_path and datum.cvOutputData is not None:
        out.write(datum.cvOutputData)

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"Processed {{frame_idx}}/{{total_frames}} frames")

total_time = time.time() - start_time
avg_frame_time = np.mean(frame_times) if frame_times else 0

# Cleanup
cap.release()
if render_path:
    out.release()

# Save results
result = {{
    'frames': frames_data,
    'avg_frame_time': avg_frame_time,
    'total_time': total_time,
    'render_path': render_path if render_path else None,
    'config': {{
        'enable_body': True,
        'enable_hand': {enable_hand},
        'enable_face': {enable_face}
    }}
}}

with open('{output_json}', 'w') as f:
    json.dump(result, f)

print('RESULT_JSON:' + json.dumps({{
    'num_frames': len(frames_data),
    'avg_frame_time': avg_frame_time,
    'total_time': total_time,
    'render_path': render_path
}}))
"#,
            openpose_dir = self.config.openpose_dir.display(),
            net_resolution = self.config.net_resolution,
            scale_number = self.config.scale_number,
            enable_hand = self.config.enable_hand,
            enable_face = self.config.enable_face,
            device = self.config.device,
            video_path = video_path.display(),
            render_path = render_path,
            output_json = output_json.display(),
        ))
    }

    fn run_script(&self, script: &str, output_json: &Path) -> Result<OpenPoseOutput> {
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("openpose_process.py");
        std::fs::write(&script_path, script)?;

        let output = Command::new(&self.python_path)
            .arg(&script_path)
            .output()
            .map_err(|e| MediaError::ExecutionFailed(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(MediaError::PythonError(stderr.to_string()));
        }

        // Parse result
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result_line = stdout
            .lines()
            .find(|l| l.starts_with("RESULT_JSON:"))
            .ok_or_else(|| MediaError::SerializationError("No result found".to_string()))?;

        let json_str = result_line.trim_start_matches("RESULT_JSON:");
        let summary: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        // Load full results from file
        let full_json = std::fs::read_to_string(output_json)?;
        let data: serde_json::Value = serde_json::from_str(&full_json)
            .map_err(|e| MediaError::SerializationError(e.to_string()))?;

        // Parse frames
        let frames = data["frames"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|f| {
                        Some(OpenPoseFrame {
                            frame_idx: f["frame_idx"].as_u64()? as u32,
                            timestamp: f["timestamp"].as_f64()?,
                            poses: f["poses"]
                                .as_array()?
                                .iter()
                                .filter_map(|p| {
                                    Some(OpenPosePose {
                                        person_id: p["person_id"].as_u64()? as u32,
                                        body: p["body"].as_array().map(|kps| {
                                            kps.iter()
                                                .filter_map(|k| {
                                                    Some(Keypoint {
                                                        x: k["x"].as_f64()? as f32,
                                                        y: k["y"].as_f64()? as f32,
                                                        confidence: k["confidence"].as_f64()?
                                                            as f32,
                                                    })
                                                })
                                                .collect()
                                        }),
                                        left_hand: p["left_hand"].as_array().map(|kps| {
                                            kps.iter()
                                                .filter_map(|k| {
                                                    Some(Keypoint {
                                                        x: k["x"].as_f64()? as f32,
                                                        y: k["y"].as_f64()? as f32,
                                                        confidence: k["confidence"].as_f64()?
                                                            as f32,
                                                    })
                                                })
                                                .collect()
                                        }),
                                        right_hand: p["right_hand"].as_array().map(|kps| {
                                            kps.iter()
                                                .filter_map(|k| {
                                                    Some(Keypoint {
                                                        x: k["x"].as_f64()? as f32,
                                                        y: k["y"].as_f64()? as f32,
                                                        confidence: k["confidence"].as_f64()?
                                                            as f32,
                                                    })
                                                })
                                                .collect()
                                        }),
                                        face: p["face"].as_array().map(|kps| {
                                            kps.iter()
                                                .filter_map(|k| {
                                                    Some(Keypoint {
                                                        x: k["x"].as_f64()? as f32,
                                                        y: k["y"].as_f64()? as f32,
                                                        confidence: k["confidence"].as_f64()?
                                                            as f32,
                                                    })
                                                })
                                                .collect()
                                        }),
                                        confidence: p["confidence"].as_f64().unwrap_or(0.0) as f32,
                                    })
                                })
                                .collect(),
                            resolution: (
                                f["resolution"][0].as_u64().unwrap_or(1920) as u32,
                                f["resolution"][1].as_u64().unwrap_or(1080) as u32,
                            ),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let render_path = summary["render_path"].as_str().map(PathBuf::from);

        Ok(OpenPoseOutput {
            frames,
            avg_frame_time: summary["avg_frame_time"].as_f64().unwrap_or(0.0),
            total_time: summary["total_time"].as_f64().unwrap_or(0.0),
            render_path,
            json_path: output_json.to_path_buf(),
        })
    }

    /// Calculate total keypoint count based on config
    pub fn keypoint_count(&self) -> u32 {
        let mut count = 0;
        if self.config.enable_body {
            count += 25; // Includes 6 foot keypoints when enable_foot is true
        }
        if self.config.enable_hand {
            count += 42; // 21 per hand
        }
        if self.config.enable_face {
            count += 70;
        }
        // Note: enable_foot doesn't add extra keypoints; they're already in body
        count
    }
}

/// Utility functions for biomechanical analysis
pub mod analysis {
    use super::*;

    /// Calculate joint angle from three keypoints
    pub fn calculate_angle(a: &Keypoint, b: &Keypoint, c: &Keypoint) -> f32 {
        let ba = ((a.x - b.x), (a.y - b.y));
        let bc = ((c.x - b.x), (c.y - b.y));

        let dot = ba.0 * bc.0 + ba.1 * bc.1;
        let mag_ba = (ba.0 * ba.0 + ba.1 * ba.1).sqrt();
        let mag_bc = (bc.0 * bc.0 + bc.1 * bc.1).sqrt();

        if mag_ba * mag_bc > 0.0 {
            (dot / (mag_ba * mag_bc)).clamp(-1.0, 1.0).acos()
        } else {
            0.0
        }
    }

    /// Calculate elbow angle from body keypoints
    pub fn elbow_angle(body: &[Keypoint], is_right: bool) -> Option<f32> {
        let (shoulder, elbow, wrist) = if is_right { (2, 3, 4) } else { (5, 6, 7) };

        if body.len() > wrist {
            Some(calculate_angle(&body[shoulder], &body[elbow], &body[wrist]))
        } else {
            None
        }
    }

    /// Calculate knee angle from body keypoints
    pub fn knee_angle(body: &[Keypoint], is_right: bool) -> Option<f32> {
        let (hip, knee, ankle) = if is_right { (9, 10, 11) } else { (12, 13, 14) };

        if body.len() > ankle {
            Some(calculate_angle(&body[hip], &body[knee], &body[ankle]))
        } else {
            None
        }
    }

    /// Calculate fingertip tremor amplitude from hand keypoints
    pub fn fingertip_displacement(frames: &[OpenPoseFrame], hand: bool) -> Vec<f32> {
        let mut displacements = Vec::new();

        for i in 1..frames.len() {
            if let (Some(prev_pose), Some(curr_pose)) =
                (frames[i - 1].poses.first(), frames[i].poses.first())
            {
                let (prev_hand, curr_hand) = if hand {
                    (&prev_pose.right_hand, &curr_pose.right_hand)
                } else {
                    (&prev_pose.left_hand, &curr_pose.left_hand)
                };

                if let (Some(prev), Some(curr)) = (prev_hand, curr_hand) {
                    // Index fingertip is keypoint 8
                    if prev.len() > 8 && curr.len() > 8 {
                        let dx = curr[8].x - prev[8].x;
                        let dy = curr[8].y - prev[8].y;
                        displacements.push((dx * dx + dy * dy).sqrt());
                    }
                }
            }
        }

        displacements
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OpenPoseConfig::default();
        assert!(config.enable_body);
        assert!(config.enable_hand);
        assert!(config.enable_face);
    }

    #[test]
    fn test_keypoint_count() {
        let config = OpenPoseConfig {
            enable_body: true,
            enable_hand: true,
            enable_face: true,
            enable_foot: true,
            ..Default::default()
        };
        let extractor = OpenPoseExtractor {
            config,
            python_path: PathBuf::from("python3"),
        };
        // 25 body + 42 hands + 70 face = 137
        assert_eq!(extractor.keypoint_count(), 137);
    }

    #[test]
    fn test_angle_calculation() {
        let a = Keypoint {
            x: 0.0,
            y: 0.0,
            confidence: 1.0,
        };
        let b = Keypoint {
            x: 1.0,
            y: 0.0,
            confidence: 1.0,
        };
        let c = Keypoint {
            x: 1.0,
            y: 1.0,
            confidence: 1.0,
        };

        let angle = analysis::calculate_angle(&a, &b, &c);
        // Should be approximately 90 degrees (π/2)
        assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 0.01);
    }
}
