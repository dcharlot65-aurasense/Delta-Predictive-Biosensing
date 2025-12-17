#!/usr/bin/env python3
"""
Blender-based hand movement video generator for the DPB Framework.
Generates videos of hand movements (rest tremor, spiral drawing, etc.) with MediaPipe Hand ground truth.

Run with:
    blender --background --python hand_generator.py -- '{"duration_sec": 10, ...}'
"""

import sys
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

# Import utilities
import utils
from utils import (
    parse_args_from_blender,
    render_animation,
    IN_BLENDER
)

if IN_BLENDER:
    import bpy
    import mathutils


@dataclass
class HandVideoParams:
    """Parameters for hand video generation."""
    duration_sec: float = 10.0
    task: str = "rest"  # rest, tapping, pronation_supination, spiral, reaching
    tremor_frequency: float = 0.0  # Hz (4-6 Hz for PD rest tremor)
    tremor_amplitude: float = 0.0  # meters
    tapping_frequency: float = 2.0  # Hz
    tapping_amplitude: float = 0.05  # meters
    bradykinesia_severity: float = 0.0  # 0-1, movement slowness
    dyskinesia_severity: float = 0.0  # 0-1, involuntary movements
    fps: int = 30
    resolution: tuple = (1280, 720)
    camera_view: str = "top"  # top, side, front
    output_path: str = "/tmp/hand_output.mp4"
    ground_truth_path: str = "/tmp/hand_ground_truth.json"
    seed: int = 42
    render: bool = True


@dataclass
class HandKeypoint:
    """Single hand landmark (MediaPipe Hand format - 21 landmarks)."""
    x: float
    y: float
    z: float
    visibility: float = 1.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


class HandRig:
    """
    Creates a hand rig with 21 MediaPipe Hand landmarks.

    MediaPipe Hand landmarks:
    0: WRIST
    1-4: THUMB (CMC, MCP, IP, TIP)
    5-8: INDEX (MCP, PIP, DIP, TIP)
    9-12: MIDDLE (MCP, PIP, DIP, TIP)
    13-16: RING (MCP, PIP, DIP, TIP)
    17-20: PINKY (MCP, PIP, DIP, TIP)
    """

    # Finger bone lengths (relative to hand size)
    BONE_LENGTHS = {
        'thumb_cmc': 0.03,
        'thumb_mcp': 0.025,
        'thumb_ip': 0.025,
        'thumb_tip': 0.02,
        'index_mcp': 0.04,
        'index_pip': 0.03,
        'index_dip': 0.025,
        'index_tip': 0.02,
        'middle_mcp': 0.045,
        'middle_pip': 0.035,
        'middle_dip': 0.03,
        'middle_tip': 0.02,
        'ring_mcp': 0.04,
        'ring_pip': 0.03,
        'ring_dip': 0.025,
        'ring_tip': 0.02,
        'pinky_mcp': 0.035,
        'pinky_pip': 0.025,
        'pinky_dip': 0.02,
        'pinky_tip': 0.015,
    }

    def __init__(self, hand_size: float = 0.18):
        """
        Initialize hand rig.

        Args:
            hand_size: Total hand length in meters (wrist to middle fingertip)
        """
        self.hand_size = hand_size
        self.armature = None
        self.bones = {}

    def create(self):
        """Create the hand armature."""
        if not IN_BLENDER:
            return None

        # Create armature
        bpy.ops.object.armature_add(location=(0, 0, 0.5))
        self.armature = bpy.context.active_object
        self.armature.name = "HandRig"

        # Enter edit mode
        bpy.ops.object.mode_set(mode='EDIT')

        # Create wrist (root)
        edit_bones = self.armature.data.edit_bones
        wrist = edit_bones.new('wrist')
        wrist.head = (0, 0, 0.5)
        wrist.tail = (0, 0, 0.52)
        self.bones['wrist'] = wrist

        # Create fingers
        self._create_thumb(wrist)
        self._create_finger('index', wrist, offset_x=-0.03)
        self._create_finger('middle', wrist, offset_x=-0.01)
        self._create_finger('ring', wrist, offset_x=0.01)
        self._create_finger('pinky', wrist, offset_x=0.03)

        bpy.ops.object.mode_set(mode='OBJECT')

        return self.armature

    def _create_thumb(self, parent_bone):
        """Create thumb bones."""
        if not IN_BLENDER:
            return

        edit_bones = self.armature.data.edit_bones

        # Thumb has different orientation (perpendicular to palm)
        x_offset = -0.04
        y_offset = 0.0
        z_base = parent_bone.tail[2]

        bones_chain = ['thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip']
        prev_bone = parent_bone

        for i, bone_name in enumerate(bones_chain):
            bone = edit_bones.new(bone_name)
            length = self.BONE_LENGTHS[bone_name]

            if i == 0:
                bone.head = (x_offset, y_offset, z_base)
                bone.tail = (x_offset - length * 0.7, y_offset + length * 0.7, z_base)
            else:
                bone.head = prev_bone.tail
                bone.tail = (bone.head[0] - length * 0.5, bone.head[1] + length * 0.866, bone.head[2])

            bone.parent = prev_bone
            self.bones[bone_name] = bone
            prev_bone = bone

    def _create_finger(self, finger_name: str, parent_bone, offset_x: float = 0.0):
        """Create finger bones."""
        if not IN_BLENDER:
            return

        edit_bones = self.armature.data.edit_bones

        z_base = parent_bone.tail[2]
        bones_chain = [
            f'{finger_name}_mcp',
            f'{finger_name}_pip',
            f'{finger_name}_dip',
            f'{finger_name}_tip'
        ]

        prev_bone = parent_bone

        for i, bone_name in enumerate(bones_chain):
            bone = edit_bones.new(bone_name)
            length = self.BONE_LENGTHS[bone_name]

            if i == 0:
                bone.head = (offset_x, 0, z_base)
                bone.tail = (offset_x, length, z_base)
            else:
                bone.head = prev_bone.tail
                bone.tail = (bone.head[0], bone.head[1] + length, bone.head[2])

            bone.parent = prev_bone
            self.bones[bone_name] = bone
            prev_bone = bone

    def extract_hand_keypoints(self, frame: int) -> List[HandKeypoint]:
        """
        Extract MediaPipe Hand keypoints from current pose.

        Returns:
            List of 21 hand keypoints
        """
        if not IN_BLENDER:
            return [HandKeypoint(0, 0, 0) for _ in range(21)]

        bpy.context.scene.frame_set(frame)

        keypoints = []

        # Map bones to MediaPipe indices
        bone_names = [
            'wrist',  # 0
            'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',  # 1-4
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',  # 5-8
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',  # 9-12
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',  # 13-16
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',  # 17-20
        ]

        for bone_name in bone_names:
            bone = self.armature.pose.bones.get(bone_name)
            if bone:
                # Use tail position for joint landmarks
                pos = self.armature.matrix_world @ bone.tail
                keypoints.append(HandKeypoint(pos.x, pos.y, pos.z))
            else:
                keypoints.append(HandKeypoint(0, 0, 0, visibility=0))

        return keypoints


class HandVideoGenerator:
    """Generate hand movement videos."""

    def __init__(self):
        self.rig: Optional[HandRig] = None
        self.params: Optional[HandVideoParams] = None
        self.total_frames: int = 0

    def setup_scene(self, params: HandVideoParams):
        """Initialize Blender scene."""
        if not IN_BLENDER:
            return

        self.params = params
        self.total_frames = int(params.duration_sec * params.fps)

        # Clear scene
        utils.BlenderSceneSetup.clear_scene()

        # Setup camera for hand view
        if params.camera_view == "top":
            location = (0, 0, 1.5)
            rotation = (0, 0, 0)
        elif params.camera_view == "side":
            location = (1.0, 0, 0.5)
            rotation = (math.pi/2, 0, math.pi/2)
        else:  # front
            location = (0, -1.0, 0.5)
            rotation = (math.pi/2, 0, 0)

        bpy.ops.object.camera_add(location=location, rotation=rotation)
        camera = bpy.context.active_object
        bpy.context.scene.camera = camera

        # Setup lighting
        utils.BlenderSceneSetup.setup_lighting(energy=300.0)

        # Setup render settings
        utils.BlenderSceneSetup.setup_render_settings(
            resolution=params.resolution,
            fps=params.fps,
            output_path=params.output_path,
            render_engine='EEVEE'
        )

        # Create hand rig
        self.rig = HandRig(hand_size=0.18)
        self.rig.create()

        # Add table surface for reference
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0.4))
        table = bpy.context.active_object
        table.name = "Table"

    def animate_rest_tremor(self):
        """Animate resting tremor (typical of PD)."""
        if not IN_BLENDER or not self.rig or not self.params:
            return

        random.seed(self.params.seed)

        for frame in range(1, self.total_frames + 1):
            bpy.context.scene.frame_set(frame)
            t = frame / self.params.fps

            # Apply tremor to fingers (especially thumb and index - "pill-rolling")
            if self.params.tremor_frequency > 0:
                tremor_offset = self.params.tremor_amplitude * math.sin(
                    2 * math.pi * self.params.tremor_frequency * t
                )

                # Thumb tremor
                thumb_bones = ['thumb_cmc', 'thumb_mcp', 'thumb_ip']
                for bone_name in thumb_bones:
                    bone = self.rig.armature.pose.bones.get(bone_name)
                    if bone:
                        bone.rotation_mode = 'XYZ'
                        bone.rotation_euler.z = tremor_offset * 0.5
                        bone.keyframe_insert(data_path='rotation_euler', index=2, frame=frame)

                # Index finger tremor
                index_bones = ['index_mcp', 'index_pip']
                for bone_name in index_bones:
                    bone = self.rig.armature.pose.bones.get(bone_name)
                    if bone:
                        bone.rotation_mode = 'XYZ'
                        bone.rotation_euler.x = tremor_offset * 0.3
                        bone.keyframe_insert(data_path='rotation_euler', index=0, frame=frame)

    def animate_spiral_drawing(self):
        """Animate spiral drawing task (tests tremor and bradykinesia)."""
        if not IN_BLENDER or not self.rig or not self.params:
            return

        # Move index finger in spiral pattern
        for frame in range(1, self.total_frames + 1):
            bpy.context.scene.frame_set(frame)
            t = frame / self.total_frames  # 0 to 1

            # Spiral parameters
            radius = 0.1 * t  # Expanding spiral
            angle = t * 4 * math.pi  # Two complete rotations

            # Slow down if bradykinesia
            speed_factor = 1.0 - self.params.bradykinesia_severity * 0.7
            angle *= speed_factor

            # Hand position traces spiral
            wrist_bone = self.rig.armature.pose.bones.get('wrist')
            if wrist_bone:
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                wrist_bone.location = (x, y, 0)
                wrist_bone.keyframe_insert(data_path='location', frame=frame)

            # Add tremor if present
            if self.params.tremor_frequency > 0:
                tremor = self.params.tremor_amplitude * math.sin(
                    2 * math.pi * self.params.tremor_frequency * t * self.params.duration_sec
                )
                wrist_bone.location.z = tremor
                wrist_bone.keyframe_insert(data_path='location', frame=frame)

    def animate_finger_tapping(self):
        """Animate finger tapping (moved to tapping_generator.py for detail)."""
        if not IN_BLENDER or not self.rig or not self.params:
            return

        for frame in range(1, self.total_frames + 1):
            bpy.context.scene.frame_set(frame)
            t = frame / self.params.fps

            # Simple tapping motion
            tap_phase = (t * self.params.tapping_frequency) % 1.0
            tap_angle = -math.pi / 4 * (1.0 - math.cos(2 * math.pi * tap_phase))

            # Apply to index finger
            index_mcp = self.rig.armature.pose.bones.get('index_mcp')
            if index_mcp:
                index_mcp.rotation_mode = 'XYZ'
                index_mcp.rotation_euler.x = tap_angle
                index_mcp.keyframe_insert(data_path='rotation_euler', index=0, frame=frame)

    def animate_task(self):
        """Animate based on selected task."""
        if self.params.task == "rest":
            self.animate_rest_tremor()
        elif self.params.task == "spiral":
            self.animate_spiral_drawing()
        elif self.params.task == "tapping":
            self.animate_finger_tapping()
        else:
            print(f"Warning: Unknown task '{self.params.task}'")

    def extract_ground_truth(self) -> List[List[HandKeypoint]]:
        """Extract hand keypoints for all frames."""
        if not IN_BLENDER or not self.rig:
            return []

        keypoints_per_frame = []

        for frame in range(1, self.total_frames + 1):
            keypoints = self.rig.extract_hand_keypoints(frame)
            keypoints_per_frame.append(keypoints)

            if frame % 30 == 0:
                print(f"Extracted hand keypoints for frame {frame}/{self.total_frames}")

        return keypoints_per_frame

    def generate(self, params: HandVideoParams) -> dict:
        """Generate hand video with ground truth."""
        print(f"Generating hand video with parameters:")
        print(f"  Duration: {params.duration_sec}s @ {params.fps} fps")
        print(f"  Task: {params.task}")
        print(f"  Tremor: {params.tremor_frequency} Hz, amplitude {params.tremor_amplitude}m")

        self.setup_scene(params)

        if IN_BLENDER:
            print("Animating hand movement...")
            self.animate_task()

            print("Extracting ground truth keypoints...")
            keypoints_per_frame = self.extract_ground_truth()

            # Save ground truth
            data = {
                'keypoints': [
                    [kp.to_list() for kp in frame]
                    for frame in keypoints_per_frame
                ],
                'format': 'mediapipe_hand_21',
                'num_frames': len(keypoints_per_frame),
                'num_landmarks': 21,
                'metadata': {
                    'task': params.task,
                    'tremor_frequency': params.tremor_frequency,
                    'tremor_amplitude': params.tremor_amplitude,
                    'tapping_frequency': params.tapping_frequency,
                }
            }

            with open(params.ground_truth_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Ground truth saved to: {params.ground_truth_path}")

            # Render video
            if params.render:
                print("Rendering video...")
                render_animation(1, self.total_frames, params.output_path)
                print(f"Video saved to: {params.output_path}")

            return {
                'video_path': params.output_path if params.render else None,
                'ground_truth_path': params.ground_truth_path,
                'num_frames': self.total_frames,
            }
        else:
            return {
                'video_path': None,
                'ground_truth_path': params.ground_truth_path,
                'num_frames': int(params.duration_sec * params.fps),
            }


def main():
    """Main entry point."""
    print("=" * 60)
    print("DPB Framework - Level 3 Hand Video Generator")
    print("=" * 60)

    args = parse_args_from_blender()
    params = HandVideoParams(**args) if args else HandVideoParams()

    generator = HandVideoGenerator()
    result = generator.generate(params)

    print("\nGeneration complete!")
    print(f"Results: {json.dumps(result, indent=2)}")

    return result


if __name__ == "__main__":
    main()
