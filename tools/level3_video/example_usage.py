#!/usr/bin/env python3
"""
Example usage script for Level 3 video generators.
Demonstrates how to generate different types of videos programmatically.
"""

import subprocess
import json
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("/tmp/dpb_level3_examples")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_blender_script(script_name, params):
    """Run a Blender script with given parameters."""
    params_json = json.dumps(params)

    cmd = [
        "blender",
        "--background",
        "--python",
        str(Path(__file__).parent / script_name),
        "--",
        params_json
    ]

    print(f"\nRunning: {script_name}")
    print(f"Parameters: {json.dumps(params, indent=2)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Success: {script_name}")
    else:
        print(f"✗ Failed: {script_name}")
        print(f"Error: {result.stderr}")

    return result.returncode == 0

def example_normal_gait():
    """Generate normal gait video."""
    params = {
        "duration_sec": 5.0,
        "cadence": 110.0,
        "stride_length": 1.4,
        "asymmetry": 0.0,
        "shuffling_severity": 0.0,
        "fps": 30,
        "camera_view": "side",
        "output_path": str(OUTPUT_DIR / "normal_gait.mp4"),
        "ground_truth_path": str(OUTPUT_DIR / "normal_gait_gt.json"),
        "render": True,
    }
    return run_blender_script("gait_generator.py", params)

def example_parkinsonian_gait():
    """Generate Parkinsonian gait video."""
    params = {
        "duration_sec": 10.0,
        "cadence": 90.0,
        "stride_length": 0.6,
        "asymmetry": 0.0,
        "shuffling_severity": 0.7,
        "arm_swing_amplitude": 0.1,
        "festination": True,
        "freezing_probability": 0.2,
        "fps": 30,
        "camera_view": "side",
        "output_path": str(OUTPUT_DIR / "parkinsonian_gait.mp4"),
        "ground_truth_path": str(OUTPUT_DIR / "parkinsonian_gait_gt.json"),
        "render": True,
    }
    return run_blender_script("gait_generator.py", params)

def example_rest_tremor():
    """Generate rest tremor video."""
    params = {
        "duration_sec": 10.0,
        "task": "rest",
        "tremor_frequency": 5.0,
        "tremor_amplitude": 0.015,
        "fps": 60,
        "camera_view": "top",
        "output_path": str(OUTPUT_DIR / "rest_tremor.mp4"),
        "ground_truth_path": str(OUTPUT_DIR / "rest_tremor_gt.json"),
        "render": True,
    }
    return run_blender_script("hand_generator.py", params)

def example_spiral_drawing():
    """Generate spiral drawing video."""
    params = {
        "duration_sec": 10.0,
        "task": "spiral",
        "tremor_frequency": 4.5,
        "tremor_amplitude": 0.005,
        "bradykinesia_severity": 0.4,
        "fps": 30,
        "camera_view": "top",
        "output_path": str(OUTPUT_DIR / "spiral_drawing.mp4"),
        "ground_truth_path": str(OUTPUT_DIR / "spiral_drawing_gt.json"),
        "render": True,
    }
    return run_blender_script("hand_generator.py", params)

def example_normal_tapping():
    """Generate normal finger tapping video."""
    params = {
        "duration_sec": 10.0,
        "target_frequency": 2.0,
        "amplitude_reduction": 0.0,
        "frequency_reduction": 0.0,
        "irregularity": 0.0,
        "hesitations": 0,
        "fps": 60,
        "output_path": str(OUTPUT_DIR / "normal_tapping.mp4"),
        "ground_truth_path": str(OUTPUT_DIR / "normal_tapping_gt.json"),
        "render": True,
    }
    return run_blender_script("tapping_generator.py", params)

def example_bradykinetic_tapping():
    """Generate bradykinetic finger tapping video."""
    params = {
        "duration_sec": 20.0,
        "target_frequency": 2.0,
        "amplitude_reduction": 0.5,
        "frequency_reduction": 0.3,
        "irregularity": 0.4,
        "hesitations": 2,
        "fatigue_factor": 0.3,
        "fps": 60,
        "output_path": str(OUTPUT_DIR / "bradykinetic_tapping.mp4"),
        "ground_truth_path": str(OUTPUT_DIR / "bradykinetic_tapping_gt.json"),
        "render": True,
    }
    return run_blender_script("tapping_generator.py", params)

def example_ground_truth_only():
    """Generate ground truth without rendering (fast)."""
    params = {
        "duration_sec": 30.0,
        "cadence": 100.0,
        "stride_length": 1.0,
        "shuffling_severity": 0.3,
        "fps": 30,
        "output_path": str(OUTPUT_DIR / "gt_only.mp4"),
        "ground_truth_path": str(OUTPUT_DIR / "gt_only.json"),
        "render": False,  # Skip rendering, only generate ground truth
    }
    return run_blender_script("gait_generator.py", params)

def main():
    """Run all examples."""
    print("=" * 70)
    print("Level 3 Video Generation Examples")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Check if Blender is available
    result = subprocess.run(["blender", "--version"], capture_output=True)
    if result.returncode != 0:
        print("\n✗ ERROR: Blender not found!")
        print("Please install Blender and add it to your PATH.")
        print("See README.md for installation instructions.")
        return

    print(f"\n✓ Blender found: {result.stdout.decode().split()[0]}")

    # Run examples
    examples = [
        ("Normal Gait", example_normal_gait),
        ("Parkinsonian Gait", example_parkinsonian_gait),
        ("Rest Tremor", example_rest_tremor),
        ("Spiral Drawing", example_spiral_drawing),
        ("Normal Finger Tapping", example_normal_tapping),
        ("Bradykinetic Tapping", example_bradykinetic_tapping),
        ("Ground Truth Only (No Render)", example_ground_truth_only),
    ]

    print("\n" + "=" * 70)
    print("Running Examples")
    print("=" * 70)

    results = []
    for name, example_func in examples:
        print(f"\n[{name}]")
        success = example_func()
        results.append((name, success))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")

    successful = sum(1 for _, s in results if s)
    print(f"\n{successful}/{len(results)} examples completed successfully")

    if successful > 0:
        print(f"\nGenerated files in: {OUTPUT_DIR}")
        print("\nTo view videos:")
        print(f"  ls {OUTPUT_DIR}/*.mp4")
        print("\nTo view ground truth:")
        print(f"  cat {OUTPUT_DIR}/*_gt.json | jq .")

if __name__ == "__main__":
    main()
