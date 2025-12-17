# Level 3 Video Generation for DPB Framework

This directory contains Blender-based video generators for creating synthetic gait, hand movement, and finger tapping videos with MediaPipe-compatible ground truth.

## Overview

Level 3 generators produce actual video files (MP4) with:
- **Rendered 3D animations** of human movement
- **Frame-by-frame ground truth** in MediaPipe format
- **Kinematic data** (joint angles, velocities, events)
- **Pathological patterns** for clinical validation

These are intended for **full pipeline testing** with MediaPipe pose/hand estimation and video processing systems.

## Requirements

### Blender Installation

You need **Blender 3.x or 4.x** installed and accessible from the command line.

#### Linux
```bash
# Ubuntu/Debian
sudo snap install blender --classic

# Or download from blender.org
wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
tar -xf blender-4.0.2-linux-x64.tar.xz
sudo mv blender-4.0.2-linux-x64 /opt/blender
sudo ln -s /opt/blender/blender /usr/local/bin/blender
```

#### macOS
```bash
# Using Homebrew
brew install --cask blender

# Or download from blender.org
# After installing, add to PATH:
export PATH="/Applications/Blender.app/Contents/MacOS:$PATH"
```

#### Windows
Download from https://www.blender.org/download/ and add to PATH:
```
C:\Program Files\Blender Foundation\Blender 4.0\
```

### Verify Installation
```bash
blender --version
# Should output: Blender 3.x.x or 4.x.x
```

## Video Generators

### 1. Gait Video Generator (`gait_generator.py`)

Generates walking videos with configurable pathological patterns.

#### Features
- Normal gait with Winter's biomechanical profiles
- Parkinsonian gait (shuffling, reduced arm swing, freezing)
- Asymmetric gait patterns
- Trunk sway and postural abnormalities
- 33 MediaPipe pose landmarks per frame

#### Usage

**Command Line:**
```bash
blender --background --python gait_generator.py -- '{
  "duration_sec": 10.0,
  "cadence": 100.0,
  "stride_length": 0.7,
  "shuffling_severity": 0.5,
  "asymmetry": 0.3,
  "fps": 30,
  "output_path": "/tmp/gait_output.mp4",
  "ground_truth_path": "/tmp/gait_ground_truth.json"
}'
```

**From Rust:**
```rust
use dpb_synth::level3::video::{Level3VideoGenerator, GaitVideoParams};

let generator = Level3VideoGenerator::new(None, None);
let params = GaitVideoParams {
    duration_sec: 10.0,
    cadence: 100.0,
    stride_length: 0.7,
    shuffling_severity: 0.5,
    asymmetry: 0.3,
    ..Default::default()
};

let output = generator.generate_gait_video(&params)?;
println!("Video: {:?}", output.video_path);
println!("Ground truth: {:?}", output.ground_truth_path);
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration_sec` | float | 10.0 | Video duration in seconds |
| `cadence` | float | 100.0 | Steps per minute |
| `stride_length` | float | 0.7 | Meters per stride |
| `velocity` | float | 1.2 | Walking speed (m/s) |
| `asymmetry` | float | 0.0 | Left-right asymmetry (0-1) |
| `shuffling_severity` | float | 0.0 | Parkinsonian shuffling (0-1) |
| `festination` | bool | false | Progressive acceleration |
| `freezing_probability` | float | 0.0 | Probability of freezing episodes (0-1) |
| `arm_swing_amplitude` | float | 0.3 | Arm swing in radians |
| `trunk_sway` | float | 0.02 | Lateral trunk motion (meters) |
| `fps` | int | 30 | Frames per second |
| `resolution` | tuple | (1920, 1080) | Video resolution |
| `camera_view` | string | "side" | Camera angle: side, front, oblique, top |
| `height` | float | 1.75 | Subject height (meters) |
| `render` | bool | true | If false, only generate ground truth |

### 2. Hand Video Generator (`hand_generator.py`)

Generates hand movement videos for tremor and motor assessment.

#### Features
- Rest tremor (4-6 Hz, typical of PD)
- Spiral drawing task
- Simple finger tapping
- 21 MediaPipe hand landmarks per frame

#### Usage

```bash
blender --background --python hand_generator.py -- '{
  "duration_sec": 10.0,
  "task": "rest",
  "tremor_frequency": 5.0,
  "tremor_amplitude": 0.01,
  "fps": 30,
  "output_path": "/tmp/hand_output.mp4"
}'
```

#### Tasks
- `"rest"` - Resting hand with optional tremor
- `"spiral"` - Spiral drawing (Archimedes spiral)
- `"tapping"` - Simple finger tapping (for quick tests)

### 3. Finger Tapping Video Generator (`tapping_generator.py`)

Specialized generator for detailed finger tapping assessment (UPDRS motor exam).

#### Features
- Progressive amplitude reduction (decrementing)
- Progressive frequency reduction (bradykinesia)
- Irregular timing (rhythm disruption)
- Hesitation/freezing episodes
- Fatigue effects
- Detailed tap event annotations

#### Usage

```bash
blender --background --python tapping_generator.py -- '{
  "duration_sec": 10.0,
  "target_frequency": 2.0,
  "amplitude_reduction": 0.3,
  "frequency_reduction": 0.2,
  "irregularity": 0.4,
  "hesitations": 2,
  "fps": 60,
  "output_path": "/tmp/tapping_output.mp4"
}'
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_frequency` | float | 2.0 | Target taps per second |
| `amplitude_reduction` | float | 0.0 | Progressive amplitude decay (0-1) |
| `frequency_reduction` | float | 0.0 | Progressive frequency decay (0-1) |
| `hesitations` | int | 0 | Number of freezing episodes |
| `irregularity` | float | 0.0 | Timing variability (0-1) |
| `fatigue_factor` | float | 0.0 | Progressive slowing (0-1) |
| `fps` | int | 60 | Frames per second (higher for detail) |

## Ground Truth Format

### Gait/Pose Ground Truth
```json
{
  "keypoints": [
    [  // Frame 0
      [x0, y0, z0],  // Landmark 0 (NOSE)
      [x1, y1, z1],  // Landmark 1 (LEFT_EYE_INNER)
      ...
      [x32, y32, z32]  // Landmark 32 (RIGHT_FOOT_INDEX)
    ],
    [  // Frame 1
      ...
    ]
  ],
  "format": "mediapipe_pose_33",
  "num_frames": 300,
  "num_landmarks": 33,
  "metadata": {
    "cadence": 100.0,
    "stride_length": 0.7,
    "shuffling_severity": 0.5
  }
}
```

### Hand Ground Truth
```json
{
  "keypoints": [...],  // 21 landmarks per frame
  "format": "mediapipe_hand_21",
  "num_frames": 300,
  "num_landmarks": 21,
  "metadata": {
    "task": "rest",
    "tremor_frequency": 5.0
  }
}
```

### Tapping Ground Truth
```json
{
  "tap_events": [
    {"time": 0.5, "amplitude": 1.0, "index": 0},
    {"time": 1.0, "amplitude": 0.95, "index": 1},
    ...
  ],
  "num_taps": 20,
  "mean_frequency": 1.95,
  "interval_coefficient_of_variation": 0.15,
  "amplitude_trend": -0.05,
  "metadata": {
    "target_frequency": 2.0,
    "amplitude_reduction": 0.3
  }
}
```

## Running Without Blender (Fallback Mode)

If Blender is not available, you can still generate ground truth without rendering:

```python
# Set render=False in parameters
params = {
    "duration_sec": 10.0,
    "render": False,  # Only generate ground truth
    "output_path": "/tmp/output.mp4",
    "ground_truth_path": "/tmp/ground_truth.json"
}
```

This will create the ground truth JSON file but skip video rendering.

## Performance Tips

### Fast Rendering (Testing)
- Use `render_engine='EEVEE'` (default, faster)
- Lower resolution: `resolution=(640, 480)`
- Lower FPS: `fps=15`
- Shorter duration: `duration_sec=5.0`

### High Quality (Production)
- Use `render_engine='CYCLES'`
- Higher resolution: `resolution=(1920, 1080)`
- Higher FPS: `fps=60`
- Increase Cycles samples in `utils.py`

### Batch Generation
Create a Python script to generate multiple videos:

```python
import subprocess
import json

# Define parameter sweep
params_list = [
    {"cadence": 80, "shuffling_severity": 0.3},
    {"cadence": 90, "shuffling_severity": 0.5},
    {"cadence": 100, "shuffling_severity": 0.7},
]

for i, params in enumerate(params_list):
    params["output_path"] = f"/tmp/gait_{i:03d}.mp4"
    params["ground_truth_path"] = f"/tmp/gait_{i:03d}.json"

    params_json = json.dumps(params)

    subprocess.run([
        "blender", "--background",
        "--python", "gait_generator.py",
        "--", params_json
    ])
```

## Troubleshooting

### Blender Not Found
```
Error: Blender not found at path: blender
```
**Solution:** Install Blender and add to PATH, or specify full path:
```rust
let generator = Level3VideoGenerator::new(
    Some(PathBuf::from("/opt/blender/blender")),
    None
);
```

### Python Import Errors
```
ImportError: No module named 'utils'
```
**Solution:** Run from the `tools/level3_video` directory or add to PYTHONPATH:
```bash
export PYTHONPATH=/path/to/tools/level3_video:$PYTHONPATH
blender --background --python gait_generator.py
```

### Rendering Very Slow
**Solution:**
- Use EEVEE instead of Cycles
- Reduce resolution and samples
- Use headless mode (already using `--background`)

### Out of Memory
**Solution:**
- Reduce video duration
- Lower resolution
- Close other applications

## Examples

### Example 1: Normal Gait
```bash
blender --background --python gait_generator.py -- '{
  "duration_sec": 10.0,
  "cadence": 110.0,
  "stride_length": 1.4,
  "camera_view": "side"
}'
```

### Example 2: Parkinsonian Gait
```bash
blender --background --python gait_generator.py -- '{
  "duration_sec": 15.0,
  "cadence": 90.0,
  "stride_length": 0.6,
  "shuffling_severity": 0.7,
  "arm_swing_amplitude": 0.1,
  "festination": true,
  "freezing_probability": 0.2
}'
```

### Example 3: Rest Tremor
```bash
blender --background --python hand_generator.py -- '{
  "duration_sec": 10.0,
  "task": "rest",
  "tremor_frequency": 5.0,
  "tremor_amplitude": 0.015
}'
```

### Example 4: Bradykinetic Tapping
```bash
blender --background --python tapping_generator.py -- '{
  "duration_sec": 20.0,
  "target_frequency": 2.0,
  "amplitude_reduction": 0.5,
  "frequency_reduction": 0.3,
  "irregularity": 0.4,
  "fatigue_factor": 0.2
}'
```

## Integration with MediaPipe

Process generated videos with MediaPipe:

```python
import mediapipe as mp
import cv2
import json

# Load ground truth
with open('/tmp/gait_ground_truth.json') as f:
    ground_truth = json.load(f)

# Run MediaPipe on generated video
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture('/tmp/gait_output.mp4')
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process with MediaPipe
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Compare with ground truth
    if results.pose_landmarks:
        gt_landmarks = ground_truth['keypoints'][frame_idx]
        # ... compare detected vs ground truth ...

    frame_idx += 1

cap.release()
```

## Citation

If you use these generators in research, please cite:

```
Delta-Predictive Biosensing Framework
Level 3 Synthetic Video Generation
https://github.com/dcharlot65-aurasense/Delta-Predictive-Biosensing
```

## License

Part of the DPB Framework. See main repository LICENSE file.

## Contributing

To add new video generators:

1. Create new Python script in this directory
2. Extend `StickFigureRig` or create custom rig in `utils.py`
3. Add corresponding Rust interface in `crates/dpb-synth/src/level3/video.rs`
4. Update this README with usage examples

## Contact

For issues or questions, please open an issue in the main repository.
