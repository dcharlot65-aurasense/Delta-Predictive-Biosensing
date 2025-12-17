#!/usr/bin/env python3
"""
Blender-based finger tapping video generator for the DPB Framework.
Specialized generator for finger tapping tasks (critical PD motor assessment).

Run with:
    blender --background --python tapping_generator.py -- '{"duration_sec": 10, ...}'
"""

import sys
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import utils
from utils import parse_args_from_blender, render_animation, IN_BLENDER

if IN_BLENDER:
    import bpy
    import mathutils


@dataclass
class TappingVideoParams:
    """Parameters for finger tapping video generation."""
    duration_sec: float = 10.0
    target_frequency: float = 2.0  # Hz (taps per second)
    amplitude_reduction: float = 0.0  # 0-1, bradykinesia amplitude decay
    frequency_reduction: float = 0.0  # 0-1, bradykinesia frequency decay
    hesitations: int = 0  # Number of hesitation/freezing episodes
    irregularity: float = 0.0  # 0-1, timing variability (higher in PD)
    fatigue_factor: float = 0.0  # 0-1, progressive slowing
    fps: int = 60  # Higher fps for detailed movement capture
    resolution: tuple = (1280, 720)
    output_path: str = "/tmp/tapping_output.mp4"
    ground_truth_path: str = "/tmp/tapping_ground_truth.json"
    seed: int = 42
    render: bool = True


class TappingVideoGenerator:
    """
    Generate finger tapping videos with pathological patterns.

    Finger tapping is a key UPDRS motor assessment task.
    Pathological patterns include:
    - Progressive amplitude reduction (decrementing amplitude)
    - Progressive frequency reduction (bradykinesia)
    - Irregular timing (disrupted rhythm)
    - Hesitations (freezing)
    - Fatigue effects
    """

    def __init__(self):
        self.params: Optional[TappingVideoParams] = None
        self.total_frames: int = 0
        self.hand_armature = None

    def setup_scene(self, params: TappingVideoParams):
        """Initialize Blender scene."""
        if not IN_BLENDER:
            return

        self.params = params
        self.total_frames = int(params.duration_sec * params.fps)

        # Clear scene
        utils.BlenderSceneSetup.clear_scene()

        # Close-up camera on hand
        bpy.ops.object.camera_add(location=(0.3, -0.5, 0.6), rotation=(1.2, 0, 0.5))
        camera = bpy.context.active_object
        bpy.context.scene.camera = camera

        # Focused lighting
        utils.BlenderSceneSetup.setup_lighting(energy=400.0)

        # Render settings
        utils.BlenderSceneSetup.setup_render_settings(
            resolution=params.resolution,
            fps=params.fps,
            output_path=params.output_path,
            render_engine='EEVEE'
        )

        # Create simplified hand (thumb and index finger only for tapping)
        self._create_tapping_hand()

    def _create_tapping_hand(self):
        """Create simplified hand rig for tapping."""
        if not IN_BLENDER:
            return

        # Create armature
        bpy.ops.object.armature_add(location=(0, 0, 0.5))
        self.hand_armature = bpy.context.active_object
        self.hand_armature.name = "TappingHand"

        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = self.hand_armature.data.edit_bones

        # Wrist
        wrist = edit_bones.new('wrist')
        wrist.head = (0, 0, 0.5)
        wrist.tail = (0, 0, 0.52)

        # Thumb (stationary)
        thumb_mcp = edit_bones.new('thumb_mcp')
        thumb_mcp.head = (-0.03, 0, 0.52)
        thumb_mcp.tail = (-0.05, 0.03, 0.52)
        thumb_mcp.parent = wrist

        thumb_tip = edit_bones.new('thumb_tip')
        thumb_tip.head = thumb_mcp.tail
        thumb_tip.tail = (-0.06, 0.05, 0.52)
        thumb_tip.parent = thumb_mcp

        # Index finger (tapping)
        index_mcp = edit_bones.new('index_mcp')
        index_mcp.head = (-0.01, 0, 0.52)
        index_mcp.tail = (-0.01, 0.04, 0.52)
        index_mcp.parent = wrist

        index_pip = edit_bones.new('index_pip')
        index_pip.head = index_mcp.tail
        index_pip.tail = (-0.01, 0.07, 0.52)
        index_pip.parent = index_mcp

        index_tip = edit_bones.new('index_tip')
        index_tip.head = index_pip.tail
        index_tip.tail = (-0.01, 0.09, 0.52)
        index_tip.parent = index_pip

        bpy.ops.object.mode_set(mode='OBJECT')

    def calculate_tap_timing(self) -> List[dict]:
        """
        Calculate timing of each tap with pathological modifications.

        Returns:
            List of tap events with timing and amplitude
        """
        random.seed(self.params.seed)

        taps = []
        expected_taps = int(self.params.duration_sec * self.params.target_frequency)

        # Base inter-tap interval
        base_interval = 1.0 / self.params.target_frequency

        current_time = 0.5  # Start after half second
        current_amplitude = 1.0
        current_frequency = self.params.target_frequency

        for tap_idx in range(expected_taps):
            # Calculate progress (for progressive effects)
            progress = tap_idx / max(expected_taps - 1, 1)

            # Progressive amplitude reduction (bradykinesia)
            amplitude_decay = 1.0 - self.params.amplitude_reduction * progress
            amplitude = current_amplitude * amplitude_decay

            # Progressive frequency reduction
            frequency_decay = 1.0 - self.params.frequency_reduction * progress
            interval = base_interval / frequency_decay

            # Add timing irregularity
            if self.params.irregularity > 0:
                noise = random.gauss(0, self.params.irregularity * base_interval * 0.3)
                interval += noise

            # Add fatigue effect (progressive slowing)
            interval *= (1.0 + self.params.fatigue_factor * progress * 0.5)

            # Check for hesitation episodes
            if tap_idx < self.params.hesitations:
                # Add pause (freezing)
                hesitation_duration = random.uniform(0.3, 0.8)
                current_time += hesitation_duration

            taps.append({
                'time': current_time,
                'amplitude': max(amplitude, 0.1),  # Minimum amplitude
                'index': tap_idx,
            })

            current_time += interval

            if current_time > self.params.duration_sec:
                break

        return taps

    def animate_tapping(self, taps: List[dict]):
        """
        Animate finger tapping based on calculated tap events.

        Args:
            taps: List of tap event dictionaries
        """
        if not IN_BLENDER or not self.hand_armature:
            return

        # Get bones
        index_mcp = self.hand_armature.pose.bones.get('index_mcp')
        index_pip = self.hand_armature.pose.bones.get('index_pip')

        if not index_mcp or not index_pip:
            return

        index_mcp.rotation_mode = 'XYZ'
        index_pip.rotation_mode = 'XYZ'

        # Animate each tap
        for tap in taps:
            tap_time = tap['time']
            amplitude = tap['amplitude']

            # Start frame for this tap
            start_frame = int(tap_time * self.params.fps)

            # Tap cycle: contact (25% of interval), release (75% of interval)
            next_tap_time = tap_time + (1.0 / self.params.target_frequency)
            tap_duration_frames = int((next_tap_time - tap_time) * self.params.fps)
            contact_frames = max(int(tap_duration_frames * 0.25), 3)
            release_frames = tap_duration_frames - contact_frames

            # Max flexion angle (proportional to amplitude)
            max_mcp_angle = -math.pi / 3 * amplitude  # -60 degrees scaled
            max_pip_angle = -math.pi / 4 * amplitude  # -45 degrees scaled

            # Keyframe: Start position (extended)
            bpy.context.scene.frame_set(start_frame)
            index_mcp.rotation_euler.x = 0
            index_pip.rotation_euler.x = 0
            index_mcp.keyframe_insert(data_path='rotation_euler', index=0, frame=start_frame)
            index_pip.keyframe_insert(data_path='rotation_euler', index=0, frame=start_frame)

            # Keyframe: Contact position (flexed)
            contact_frame = start_frame + contact_frames
            bpy.context.scene.frame_set(contact_frame)
            index_mcp.rotation_euler.x = max_mcp_angle
            index_pip.rotation_euler.x = max_pip_angle
            index_mcp.keyframe_insert(data_path='rotation_euler', index=0, frame=contact_frame)
            index_pip.keyframe_insert(data_path='rotation_euler', index=0, frame=contact_frame)

            # Keyframe: Release position (extended)
            release_frame = start_frame + tap_duration_frames
            if release_frame <= self.total_frames:
                bpy.context.scene.frame_set(release_frame)
                index_mcp.rotation_euler.x = 0
                index_pip.rotation_euler.x = 0
                index_mcp.keyframe_insert(data_path='rotation_euler', index=0, frame=release_frame)
                index_pip.keyframe_insert(data_path='rotation_euler', index=0, frame=release_frame)

    def extract_ground_truth(self, taps: List[dict]) -> dict:
        """
        Extract ground truth data including tap events and kinematics.

        Args:
            taps: List of tap events

        Returns:
            Ground truth dictionary
        """
        # Extract tap metrics
        tap_times = [tap['time'] for tap in taps]
        tap_amplitudes = [tap['amplitude'] for tap in taps]

        # Calculate inter-tap intervals
        intervals = []
        for i in range(len(tap_times) - 1):
            intervals.append(tap_times[i + 1] - tap_times[i])

        # Calculate tap statistics
        if intervals:
            mean_interval = sum(intervals) / len(intervals)
            mean_frequency = 1.0 / mean_interval if mean_interval > 0 else 0
            interval_cv = (sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5 / mean_interval if mean_interval > 0 else 0
        else:
            mean_frequency = 0
            interval_cv = 0

        # Calculate amplitude statistics
        if tap_amplitudes:
            amplitude_trend = (tap_amplitudes[-1] - tap_amplitudes[0]) / len(tap_amplitudes)
        else:
            amplitude_trend = 0

        ground_truth = {
            'tap_events': taps,
            'num_taps': len(taps),
            'mean_frequency': mean_frequency,
            'interval_coefficient_of_variation': interval_cv,
            'amplitude_trend': amplitude_trend,
            'metadata': {
                'target_frequency': self.params.target_frequency,
                'amplitude_reduction': self.params.amplitude_reduction,
                'frequency_reduction': self.params.frequency_reduction,
                'irregularity': self.params.irregularity,
                'hesitations': self.params.hesitations,
                'fatigue_factor': self.params.fatigue_factor,
            }
        }

        return ground_truth

    def generate(self, params: TappingVideoParams) -> dict:
        """Generate finger tapping video with ground truth."""
        print(f"Generating finger tapping video:")
        print(f"  Duration: {params.duration_sec}s @ {params.fps} fps")
        print(f"  Target frequency: {params.target_frequency} Hz")
        print(f"  Amplitude reduction: {params.amplitude_reduction}")
        print(f"  Frequency reduction: {params.frequency_reduction}")
        print(f"  Irregularity: {params.irregularity}")

        self.setup_scene(params)

        # Calculate tap timing
        print("Calculating tap timing...")
        taps = self.calculate_tap_timing()
        print(f"Generated {len(taps)} tap events")

        if IN_BLENDER:
            # Animate tapping
            print("Animating finger tapping...")
            self.animate_tapping(taps)

            # Extract ground truth
            print("Extracting ground truth...")
            ground_truth = self.extract_ground_truth(taps)

            # Save ground truth
            with open(params.ground_truth_path, 'w') as f:
                json.dump(ground_truth, f, indent=2)
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
                'num_taps': len(taps),
            }
        else:
            # Generate ground truth without rendering
            ground_truth = self.extract_ground_truth(taps)
            with open(params.ground_truth_path, 'w') as f:
                json.dump(ground_truth, f, indent=2)

            return {
                'video_path': None,
                'ground_truth_path': params.ground_truth_path,
                'num_frames': int(params.duration_sec * params.fps),
                'num_taps': len(taps),
            }


def main():
    """Main entry point."""
    print("=" * 60)
    print("DPB Framework - Level 3 Finger Tapping Video Generator")
    print("=" * 60)

    args = parse_args_from_blender()
    params = TappingVideoParams(**args) if args else TappingVideoParams()

    generator = TappingVideoGenerator()
    result = generator.generate(params)

    print("\nGeneration complete!")
    print(f"Results: {json.dumps(result, indent=2)}")

    return result


if __name__ == "__main__":
    main()
