#!/usr/bin/env python3
"""
Blender-based gait video generator for the DPB Framework.
Generates realistic gait videos with MediaPipe-compatible ground truth.

Run with:
    blender --background --python gait_generator.py -- '{"duration_sec": 10, ...}'
"""

import sys
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

# Import utilities
import utils
from utils import (
    MediaPipeKeypoint,
    MediaPipeMapper,
    BlenderSceneSetup,
    StickFigureRig,
    save_ground_truth,
    parse_args_from_blender,
    render_animation,
    IN_BLENDER
)

if IN_BLENDER:
    import bpy
    import mathutils


@dataclass
class GaitVideoParams:
    """Parameters for gait video generation."""
    duration_sec: float = 10.0
    cadence: float = 100.0  # steps per minute
    stride_length: float = 0.7  # meters
    velocity: float = 1.2  # m/s
    asymmetry: float = 0.0  # 0-1, left-right stride difference
    shuffling_severity: float = 0.0  # 0-1, PD shuffling gait
    festination: bool = False  # PD festinating gait
    freezing_probability: float = 0.0  # 0-1, probability of freezing episodes
    arm_swing_amplitude: float = 0.3  # radians
    trunk_sway: float = 0.02  # meters
    fps: int = 30
    resolution: tuple = (1920, 1080)
    camera_view: str = "side"  # side, front, oblique, top
    output_path: str = "/tmp/gait_output.mp4"
    ground_truth_path: str = "/tmp/gait_ground_truth.json"
    seed: int = 42
    height: float = 1.75  # meters, subject height
    render: bool = True  # If False, only generate ground truth


class GaitVideoGenerator:
    """Generate gait videos using Blender animation."""

    # Joint angle profiles from Winter's biomechanics (normalized 0-1 gait cycle)
    # Format: [(phase, angle), ...]

    HIP_FLEXION_PROFILE = [
        (0.0, 30.0),   # Initial contact: flexion
        (0.12, 15.0),  # Loading response
        (0.5, -10.0),  # Terminal stance: extension
        (0.62, 20.0),  # Initial swing: flexion
        (0.73, 30.0),  # Mid swing: peak flexion
        (1.0, 30.0),   # Terminal swing
    ]

    KNEE_FLEXION_PROFILE = [
        (0.0, 5.0),    # Initial contact: slight flexion
        (0.12, 20.0),  # Loading response: flexion
        (0.35, 5.0),   # Mid stance: extension
        (0.5, 0.0),    # Terminal stance: full extension
        (0.62, 40.0),  # Pre-swing: flexion
        (0.73, 60.0),  # Initial/mid swing: peak flexion
        (0.87, 30.0),  # Terminal swing: partial extension
        (1.0, 5.0),    # Next contact
    ]

    ANKLE_DORSIFLEXION_PROFILE = [
        (0.0, 0.0),    # Initial contact: neutral
        (0.12, -5.0),  # Loading response: plantarflexion
        (0.35, 5.0),   # Mid stance: dorsiflexion
        (0.5, 10.0),   # Terminal stance: peak dorsiflexion
        (0.62, -15.0), # Pre-swing: plantarflexion
        (0.73, 0.0),   # Mid swing: neutral
        (1.0, 0.0),    # Terminal swing
    ]

    def __init__(self):
        self.rig: Optional[StickFigureRig] = None
        self.params: Optional[GaitVideoParams] = None
        self.total_frames: int = 0
        self.keyframes_per_cycle: int = 60  # Keyframes for one gait cycle

    def setup_scene(self, params: GaitVideoParams):
        """Initialize Blender scene with camera, lights, and rig."""
        if not IN_BLENDER:
            return

        self.params = params
        self.total_frames = int(params.duration_sec * params.fps)

        # Clear scene
        BlenderSceneSetup.clear_scene()

        # Setup camera
        BlenderSceneSetup.setup_camera(
            view=params.camera_view,
            distance=5.0,
            height=params.height * 0.6
        )

        # Setup lighting
        BlenderSceneSetup.setup_lighting(energy=500.0)

        # Setup render settings
        BlenderSceneSetup.setup_render_settings(
            resolution=params.resolution,
            fps=params.fps,
            output_path=params.output_path,
            render_engine='EEVEE'  # Faster rendering
        )

        # Create stick figure rig
        self.rig = StickFigureRig(height=params.height)
        self.rig.create()

        # Add ground plane for reference
        bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
        ground = bpy.context.active_object
        ground.name = "Ground"

    def interpolate_profile(self, profile: List[tuple], phase: float) -> float:
        """
        Interpolate angle from a profile given the gait phase.

        Args:
            profile: List of (phase, angle) tuples
            phase: Current gait phase (0-1)

        Returns:
            Interpolated angle in degrees
        """
        # Find surrounding points
        for i in range(len(profile) - 1):
            phase1, angle1 = profile[i]
            phase2, angle2 = profile[i + 1]

            if phase1 <= phase <= phase2:
                # Linear interpolation
                t = (phase - phase1) / (phase2 - phase1)
                return angle1 + t * (angle2 - angle1)

        # Handle edge case (should not happen with proper profiles)
        return profile[-1][1]

    def apply_pathology_modifiers(self, joint_angle: float, joint_name: str) -> float:
        """
        Apply pathological gait modifiers to joint angles.

        Args:
            joint_angle: Base joint angle
            joint_name: Name of the joint

        Returns:
            Modified angle
        """
        if not self.params:
            return joint_angle

        # Shuffling gait (reduced range of motion)
        if self.params.shuffling_severity > 0:
            joint_angle *= (1.0 - self.params.shuffling_severity * 0.7)

        return joint_angle

    def animate_gait_cycle(self):
        """Create keyframe animation for gait cycle."""
        if not IN_BLENDER or not self.rig or not self.params:
            return

        cycle_duration = 60.0 / self.params.cadence  # seconds per stride
        frames_per_cycle = int(cycle_duration * self.params.fps)

        # Calculate number of complete cycles
        num_cycles = int(self.total_frames / frames_per_cycle)

        # Animate each cycle
        for cycle in range(num_cycles + 1):
            start_frame = cycle * frames_per_cycle + 1

            for phase_step in range(self.keyframes_per_cycle):
                phase = phase_step / self.keyframes_per_cycle
                frame = start_frame + int(phase * frames_per_cycle)

                if frame > self.total_frames:
                    break

                bpy.context.scene.frame_set(frame)

                # Animate both legs (left leg is opposite phase)
                for side, phase_offset in [('right', 0.0), ('left', 0.5)]:
                    leg_phase = (phase + phase_offset) % 1.0

                    # Get joint angles from profiles
                    hip_angle = self.interpolate_profile(self.HIP_FLEXION_PROFILE, leg_phase)
                    knee_angle = self.interpolate_profile(self.KNEE_FLEXION_PROFILE, leg_phase)
                    ankle_angle = self.interpolate_profile(self.ANKLE_DORSIFLEXION_PROFILE, leg_phase)

                    # Apply pathology modifiers
                    hip_angle = self.apply_pathology_modifiers(hip_angle, f'{side}_hip')
                    knee_angle = self.apply_pathology_modifiers(knee_angle, f'{side}_knee')
                    ankle_angle = self.apply_pathology_modifiers(ankle_angle, f'{side}_ankle')

                    # Apply asymmetry
                    if side == 'right':
                        hip_angle *= (1.0 - self.params.asymmetry * 0.3)
                        knee_angle *= (1.0 - self.params.asymmetry * 0.3)

                    # Set bone rotations
                    self._set_bone_rotation(f'{side}_thigh', 'X', math.radians(hip_angle), frame)
                    self._set_bone_rotation(f'{side}_shank', 'X', math.radians(-knee_angle), frame)
                    self._set_bone_rotation(f'{side}_foot', 'X', math.radians(ankle_angle), frame)

                # Arm swing (opposite to leg swing)
                for side, phase_offset in [('right', 0.5), ('left', 0.0)]:
                    arm_phase = (phase + phase_offset) % 1.0
                    arm_angle = self.params.arm_swing_amplitude * math.sin(2 * math.pi * arm_phase)
                    self._set_bone_rotation(f'{side}_upper_arm', 'X', arm_angle, frame)

                # Trunk sway
                trunk_sway = self.params.trunk_sway * math.sin(2 * math.pi * phase)
                pelvis_bone = self.rig.armature.pose.bones.get('pelvis')
                if pelvis_bone:
                    pelvis_bone.location.x = trunk_sway
                    pelvis_bone.keyframe_insert(data_path='location', frame=frame)

                # Forward progression
                forward_distance = cycle * self.params.stride_length + phase * self.params.stride_length
                if pelvis_bone:
                    pelvis_bone.location.y = forward_distance
                    pelvis_bone.keyframe_insert(data_path='location', frame=frame)

    def _set_bone_rotation(self, bone_name: str, axis: str, angle: float, frame: int):
        """
        Set bone rotation and insert keyframe.

        Args:
            bone_name: Name of the bone
            axis: Rotation axis ('X', 'Y', or 'Z')
            angle: Rotation angle in radians
            frame: Frame number
        """
        if not IN_BLENDER or not self.rig:
            return

        bone = self.rig.armature.pose.bones.get(bone_name)
        if not bone:
            return

        # Set rotation mode to XYZ Euler
        bone.rotation_mode = 'XYZ'

        # Set rotation on specified axis
        axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        bone.rotation_euler[axis_idx] = angle

        # Insert keyframe
        bone.keyframe_insert(data_path='rotation_euler', index=axis_idx, frame=frame)

    def extract_ground_truth(self) -> List[List[MediaPipeKeypoint]]:
        """
        Extract MediaPipe keypoints for all frames.

        Returns:
            List of frames, each containing 33 keypoints
        """
        if not IN_BLENDER or not self.rig:
            return []

        keypoints_per_frame = []

        for frame in range(1, self.total_frames + 1):
            keypoints = self.rig.extract_mediapipe_keypoints(frame)
            keypoints_per_frame.append(keypoints)

            if frame % 30 == 0:
                print(f"Extracted keypoints for frame {frame}/{self.total_frames}")

        return keypoints_per_frame

    def generate(self, params: GaitVideoParams) -> dict:
        """
        Generate gait video with ground truth.

        Args:
            params: Generation parameters

        Returns:
            Dictionary with output paths and metadata
        """
        print(f"Generating gait video with parameters:")
        print(f"  Duration: {params.duration_sec}s @ {params.fps} fps")
        print(f"  Cadence: {params.cadence} steps/min")
        print(f"  Stride length: {params.stride_length}m")
        print(f"  Shuffling severity: {params.shuffling_severity}")

        self.setup_scene(params)

        if IN_BLENDER:
            print("Animating gait cycle...")
            self.animate_gait_cycle()

            print("Extracting ground truth keypoints...")
            keypoints_per_frame = self.extract_ground_truth()

            # Save ground truth
            metadata = {
                'duration_sec': params.duration_sec,
                'fps': params.fps,
                'cadence': params.cadence,
                'stride_length': params.stride_length,
                'asymmetry': params.asymmetry,
                'shuffling_severity': params.shuffling_severity,
                'height': params.height,
            }
            save_ground_truth(keypoints_per_frame, params.ground_truth_path, metadata)
            print(f"Ground truth saved to: {params.ground_truth_path}")

            # Render video
            if params.render:
                print("Rendering video...")
                render_animation(1, self.total_frames, params.output_path)
                print(f"Video saved to: {params.output_path}")
            else:
                print("Skipping render (render=False)")

            return {
                'video_path': params.output_path if params.render else None,
                'ground_truth_path': params.ground_truth_path,
                'num_frames': self.total_frames,
                'duration_sec': params.duration_sec,
            }
        else:
            print("Not running in Blender - generating mock output")
            return {
                'video_path': None,
                'ground_truth_path': params.ground_truth_path,
                'num_frames': int(params.duration_sec * params.fps),
                'duration_sec': params.duration_sec,
            }


def main():
    """Main entry point when run from Blender."""
    print("=" * 60)
    print("DPB Framework - Level 3 Gait Video Generator")
    print("=" * 60)

    # Parse arguments
    args = parse_args_from_blender()

    # Create parameters from args
    params = GaitVideoParams(**args) if args else GaitVideoParams()

    # Generate video
    generator = GaitVideoGenerator()
    result = generator.generate(params)

    # Print results
    print("\nGeneration complete!")
    print(f"Results: {json.dumps(result, indent=2)}")

    return result


if __name__ == "__main__":
    main()
