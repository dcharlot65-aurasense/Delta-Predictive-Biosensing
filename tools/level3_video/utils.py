"""
Shared utilities for Level 3 video generation with Blender.
Contains common functions for scene setup, rendering, and MediaPipe keypoint generation.
"""

import sys
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

# Check if running in Blender
try:
    import bpy
    import mathutils
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    # Define minimal stubs for non-Blender environments
    class mathutils:
        class Vector:
            def __init__(self, coords):
                self.coords = coords

    class bpy:
        class ops:
            class object:
                @staticmethod
                def select_all(action='SELECT'):
                    pass
                @staticmethod
                def delete():
                    pass
        class data:
            objects = []


@dataclass
class MediaPipeKeypoint:
    """Represents a single MediaPipe pose landmark."""
    x: float
    y: float
    z: float
    visibility: float = 1.0

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


class MediaPipeMapper:
    """Maps skeleton bones to MediaPipe 33-landmark format."""

    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    @staticmethod
    def create_empty_frame() -> List[MediaPipeKeypoint]:
        """Create a frame with 33 zero-initialized keypoints."""
        return [MediaPipeKeypoint(0.0, 0.0, 0.0) for _ in range(33)]


class BlenderSceneSetup:
    """Handles common Blender scene setup operations."""

    @staticmethod
    def clear_scene():
        """Remove all objects from the scene."""
        if not IN_BLENDER:
            return

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Clear orphaned data
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

    @staticmethod
    def setup_camera(view: str = "side", distance: float = 5.0, height: float = 1.0):
        """
        Setup camera based on view angle.

        Args:
            view: Camera view - 'side', 'front', 'oblique', or 'top'
            distance: Distance from subject
            height: Camera height
        """
        if not IN_BLENDER:
            return None

        bpy.ops.object.camera_add()
        camera = bpy.context.active_object

        if view == "side":
            camera.location = (distance, 0, height)
            camera.rotation_euler = (math.pi/2, 0, math.pi/2)
        elif view == "front":
            camera.location = (0, -distance, height)
            camera.rotation_euler = (math.pi/2, 0, 0)
        elif view == "oblique":
            camera.location = (distance * 0.7, -distance * 0.7, height)
            camera.rotation_euler = (math.pi/2, 0, math.pi/4)
        elif view == "top":
            camera.location = (0, 0, distance + height)
            camera.rotation_euler = (0, 0, 0)

        bpy.context.scene.camera = camera
        return camera

    @staticmethod
    def setup_lighting(energy: float = 1000.0):
        """Setup basic 3-point lighting."""
        if not IN_BLENDER:
            return []

        lights = []

        # Key light
        bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
        key_light = bpy.context.active_object
        key_light.data.energy = energy
        lights.append(key_light)

        # Fill light
        bpy.ops.object.light_add(type='SUN', location=(-5, -3, 5))
        fill_light = bpy.context.active_object
        fill_light.data.energy = energy * 0.5
        lights.append(fill_light)

        # Back light
        bpy.ops.object.light_add(type='SUN', location=(0, 5, 8))
        back_light = bpy.context.active_object
        back_light.data.energy = energy * 0.3
        lights.append(back_light)

        return lights

    @staticmethod
    def setup_render_settings(resolution: Tuple[int, int] = (1920, 1080),
                             fps: int = 30,
                             output_path: str = "/tmp/output.mp4",
                             render_engine: str = "CYCLES"):
        """
        Configure render settings.

        Args:
            resolution: (width, height) in pixels
            fps: Frames per second
            output_path: Output video file path
            render_engine: 'CYCLES' or 'EEVEE'
        """
        if not IN_BLENDER:
            return

        scene = bpy.context.scene

        # Resolution
        scene.render.resolution_x = resolution[0]
        scene.render.resolution_y = resolution[1]
        scene.render.resolution_percentage = 100

        # Frame rate
        scene.render.fps = fps

        # Output
        scene.render.filepath = output_path
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        scene.render.ffmpeg.constant_rate_factor = 'HIGH'

        # Render engine
        scene.render.engine = render_engine

        if render_engine == 'CYCLES':
            scene.cycles.samples = 32  # Lower for faster rendering
            scene.cycles.use_denoising = True
        elif render_engine == 'EEVEE':
            scene.eevee.taa_render_samples = 16


class StickFigureRig:
    """Creates a simple stick figure for gait/movement animation."""

    def __init__(self, height: float = 1.75):
        """
        Initialize stick figure with anthropometric proportions.

        Args:
            height: Total body height in meters
        """
        self.height = height
        self.armature = None
        self.bones = {}

        # Anthropometric ratios (based on Winter's biomechanics)
        self.ratios = {
            'head': 0.13,
            'trunk': 0.30,
            'upper_arm': 0.186,
            'forearm': 0.146,
            'hand': 0.108,
            'thigh': 0.245,
            'shank': 0.246,
            'foot': 0.152,
        }

    def create(self):
        """Create the stick figure armature."""
        if not IN_BLENDER:
            return None

        # Create armature
        bpy.ops.object.armature_add(location=(0, 0, 0))
        self.armature = bpy.context.active_object
        self.armature.name = "StickFigure"

        # Enter edit mode to create bones
        bpy.ops.object.mode_set(mode='EDIT')

        # Build skeleton hierarchy
        self._create_spine()
        self._create_head()
        self._create_arms()
        self._create_legs()

        bpy.ops.object.mode_set(mode='OBJECT')

        return self.armature

    def _create_spine(self):
        """Create spine bones."""
        if not IN_BLENDER:
            return

        edit_bones = self.armature.data.edit_bones

        # Pelvis (root)
        pelvis_height = self.height * 0.55
        pelvis = edit_bones.new('pelvis')
        pelvis.head = (0, 0, pelvis_height - 0.05)
        pelvis.tail = (0, 0, pelvis_height + 0.05)
        self.bones['pelvis'] = pelvis

        # Spine
        spine = edit_bones.new('spine')
        spine.head = pelvis.tail
        spine.tail = (0, 0, self.height * 0.82)
        spine.parent = pelvis
        self.bones['spine'] = spine

    def _create_head(self):
        """Create head bone."""
        if not IN_BLENDER:
            return

        edit_bones = self.armature.data.edit_bones
        spine = self.bones['spine']

        head = edit_bones.new('head')
        head.head = spine.tail
        head.tail = (0, 0, self.height)
        head.parent = spine
        self.bones['head'] = head

    def _create_arms(self):
        """Create arm bones."""
        if not IN_BLENDER:
            return

        edit_bones = self.armature.data.edit_bones
        spine = self.bones['spine']

        shoulder_height = spine.tail[2]
        shoulder_width = self.height * 0.23

        for side, x_sign in [('left', -1), ('right', 1)]:
            # Shoulder
            shoulder = edit_bones.new(f'{side}_shoulder')
            shoulder.head = (x_sign * shoulder_width / 2, 0, shoulder_height)
            shoulder.tail = (x_sign * (shoulder_width / 2 + 0.05), 0, shoulder_height)
            shoulder.parent = spine
            self.bones[f'{side}_shoulder'] = shoulder

            # Upper arm
            upper_arm_length = self.height * self.ratios['upper_arm']
            upper_arm = edit_bones.new(f'{side}_upper_arm')
            upper_arm.head = shoulder.tail
            upper_arm.tail = (x_sign * (shoulder_width / 2 + 0.05), 0, shoulder_height - upper_arm_length)
            upper_arm.parent = shoulder
            self.bones[f'{side}_upper_arm'] = upper_arm

            # Forearm
            forearm_length = self.height * self.ratios['forearm']
            forearm = edit_bones.new(f'{side}_forearm')
            forearm.head = upper_arm.tail
            forearm.tail = (x_sign * (shoulder_width / 2 + 0.05), 0,
                           shoulder_height - upper_arm_length - forearm_length)
            forearm.parent = upper_arm
            self.bones[f'{side}_forearm'] = forearm

            # Hand
            hand_length = self.height * self.ratios['hand']
            hand = edit_bones.new(f'{side}_hand')
            hand.head = forearm.tail
            hand.tail = (x_sign * (shoulder_width / 2 + 0.05), 0,
                        forearm.tail[2] - hand_length)
            hand.parent = forearm
            self.bones[f'{side}_hand'] = hand

    def _create_legs(self):
        """Create leg bones."""
        if not IN_BLENDER:
            return

        edit_bones = self.armature.data.edit_bones
        pelvis = self.bones['pelvis']

        hip_width = self.height * 0.15
        pelvis_height = pelvis.head[2]

        for side, x_sign in [('left', -1), ('right', 1)]:
            # Hip
            hip = edit_bones.new(f'{side}_hip')
            hip.head = (x_sign * hip_width / 2, 0, pelvis_height)
            hip.tail = (x_sign * (hip_width / 2 + 0.05), 0, pelvis_height)
            hip.parent = pelvis
            self.bones[f'{side}_hip'] = hip

            # Thigh
            thigh_length = self.height * self.ratios['thigh']
            thigh = edit_bones.new(f'{side}_thigh')
            thigh.head = hip.tail
            thigh.tail = (x_sign * (hip_width / 2 + 0.05), 0, pelvis_height - thigh_length)
            thigh.parent = hip
            self.bones[f'{side}_thigh'] = thigh

            # Shank
            shank_length = self.height * self.ratios['shank']
            shank = edit_bones.new(f'{side}_shank')
            shank.head = thigh.tail
            shank.tail = (x_sign * (hip_width / 2 + 0.05), 0,
                         pelvis_height - thigh_length - shank_length)
            shank.parent = thigh
            self.bones[f'{side}_shank'] = shank

            # Foot
            foot_length = self.height * self.ratios['foot']
            foot = edit_bones.new(f'{side}_foot')
            foot.head = shank.tail
            foot.tail = (x_sign * (hip_width / 2 + 0.05), foot_length * 0.7, shank.tail[2])
            foot.parent = shank
            self.bones[f'{side}_foot'] = foot

    def get_bone_world_position(self, bone_name: str, head: bool = True) -> Tuple[float, float, float]:
        """
        Get world position of a bone.

        Args:
            bone_name: Name of the bone
            head: If True, return head position, else tail position

        Returns:
            (x, y, z) coordinates
        """
        if not IN_BLENDER or not self.armature:
            return (0, 0, 0)

        bone = self.armature.pose.bones.get(bone_name)
        if not bone:
            return (0, 0, 0)

        if head:
            pos = self.armature.matrix_world @ bone.head
        else:
            pos = self.armature.matrix_world @ bone.tail

        return (pos.x, pos.y, pos.z)

    def extract_mediapipe_keypoints(self, frame: int) -> List[MediaPipeKeypoint]:
        """
        Extract MediaPipe-compatible keypoints from current pose.

        Args:
            frame: Current frame number

        Returns:
            List of 33 MediaPipe keypoints
        """
        if not IN_BLENDER:
            return MediaPipeMapper.create_empty_frame()

        # Set frame
        bpy.context.scene.frame_set(frame)

        keypoints = MediaPipeMapper.create_empty_frame()

        # Map bones to MediaPipe indices
        bone_map = {
            MediaPipeMapper.NOSE: ('head', True),  # head bone, tail position
            MediaPipeMapper.LEFT_SHOULDER: ('left_shoulder', False),
            MediaPipeMapper.RIGHT_SHOULDER: ('right_shoulder', False),
            MediaPipeMapper.LEFT_ELBOW: ('left_upper_arm', False),
            MediaPipeMapper.RIGHT_ELBOW: ('right_upper_arm', False),
            MediaPipeMapper.LEFT_WRIST: ('left_forearm', False),
            MediaPipeMapper.RIGHT_WRIST: ('right_forearm', False),
            MediaPipeMapper.LEFT_HIP: ('left_hip', False),
            MediaPipeMapper.RIGHT_HIP: ('right_hip', False),
            MediaPipeMapper.LEFT_KNEE: ('left_thigh', False),
            MediaPipeMapper.RIGHT_KNEE: ('right_thigh', False),
            MediaPipeMapper.LEFT_ANKLE: ('left_shank', False),
            MediaPipeMapper.RIGHT_ANKLE: ('right_shank', False),
            MediaPipeMapper.LEFT_HEEL: ('left_foot', True),
            MediaPipeMapper.RIGHT_HEEL: ('right_foot', True),
            MediaPipeMapper.LEFT_FOOT_INDEX: ('left_foot', False),
            MediaPipeMapper.RIGHT_FOOT_INDEX: ('right_foot', False),
        }

        for mp_idx, (bone_name, use_head) in bone_map.items():
            pos = self.get_bone_world_position(bone_name, head=use_head)
            keypoints[mp_idx] = MediaPipeKeypoint(pos[0], pos[1], pos[2])

        # Fill in missing face landmarks with interpolated values
        if self.armature:
            nose_pos = keypoints[MediaPipeMapper.NOSE]
            for i in range(1, 11):  # Eyes, ears, mouth
                keypoints[i] = MediaPipeKeypoint(nose_pos.x, nose_pos.y, nose_pos.z)

        return keypoints


def save_ground_truth(keypoints_per_frame: List[List[MediaPipeKeypoint]],
                     output_path: str,
                     metadata: Optional[Dict] = None):
    """
    Save ground truth keypoints to JSON file.

    Args:
        keypoints_per_frame: List of frames, each containing 33 keypoints
        output_path: Output JSON file path
        metadata: Additional metadata to include
    """
    data = {
        'keypoints': [
            [kp.to_list() for kp in frame]
            for frame in keypoints_per_frame
        ],
        'format': 'mediapipe_pose_33',
        'num_frames': len(keypoints_per_frame),
        'num_landmarks': 33,
    }

    if metadata:
        data['metadata'] = metadata

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def parse_args_from_blender():
    """
    Parse command-line arguments when running from Blender.
    Blender uses '--' to separate its args from script args.

    Returns:
        Dictionary of parsed arguments
    """
    try:
        # Find the '--' separator
        separator_idx = sys.argv.index('--')
        script_args = sys.argv[separator_idx + 1:]
    except ValueError:
        script_args = []

    if len(script_args) > 0:
        # First arg is expected to be JSON parameters
        try:
            return json.loads(script_args[0])
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON args: {script_args[0]}")
            return {}

    return {}


def render_animation(start_frame: int, end_frame: int, output_path: str):
    """
    Render animation to video file.

    Args:
        start_frame: First frame to render
        end_frame: Last frame to render (inclusive)
        output_path: Output video file path
    """
    if not IN_BLENDER:
        print("Not running in Blender - skipping render")
        return

    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    scene.render.filepath = output_path

    # Render animation
    bpy.ops.render.render(animation=True)
    print(f"Rendered {end_frame - start_frame + 1} frames to {output_path}")


if __name__ == "__main__":
    print("Level 3 Video Generation Utilities")
    print(f"Running in Blender: {IN_BLENDER}")

    if IN_BLENDER:
        print(f"Blender version: {bpy.app.version_string}")
