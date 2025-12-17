#!/usr/bin/env python3
"""
Test script for Level 3 audio generators.

Verifies that all generators work correctly and can produce audio with ground truth.
"""

import sys
import json
import numpy as np

try:
    from generator import (
        SustainedVowelGenerator,
        ConnectedSpeechGenerator,
        DiadochokinesisGenerator,
        ReadingPassageGenerator,
        VowelParams,
        ConnectedSpeechParams,
        DiadochokinesisParams,
        ReadingPassageParams,
        WORLD_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing generator module: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available - WAV export disabled")


def test_sustained_vowel():
    """Test sustained vowel generation"""
    print("Testing SustainedVowelGenerator...")

    if not WORLD_AVAILABLE:
        print("SKIP: pyworld not available")
        return False

    try:
        gen = SustainedVowelGenerator()

        # Test normal vowel
        params = VowelParams(
            duration_sec=1.0,
            f0_mean=120.0,
            vowel="a",
            sample_rate=16000,
            seed=42
        )

        audio, ground_truth = gen.generate(params)

        # Verify output
        assert len(audio) > 0, "No audio generated"
        assert len(ground_truth.f0_contour) > 0, "No F0 contour"
        assert len(ground_truth.f0_times) == len(ground_truth.f0_contour), "F0 time mismatch"
        assert ground_truth.formants is not None, "No formants"

        # Verify F0 statistics
        f0_array = np.array(ground_truth.f0_contour)
        f0_voiced = f0_array[f0_array > 0]
        mean_f0 = np.mean(f0_voiced)
        assert 110 < mean_f0 < 130, f"F0 mean out of range: {mean_f0}"

        print(f"  Generated {len(audio)} samples ({len(audio)/params.sample_rate:.2f} sec)")
        print(f"  F0 mean: {mean_f0:.1f} Hz")
        print(f"  F0 frames: {len(ground_truth.f0_contour)}")

        # Test pathological vowel (tremor)
        params_tremor = VowelParams(
            duration_sec=1.0,
            f0_mean=120.0,
            tremor_frequency=5.0,
            tremor_amplitude=10.0,
            vowel="i",
            seed=42
        )

        audio_tremor, gt_tremor = gen.generate(params_tremor)
        assert len(audio_tremor) > 0, "No tremor audio generated"

        print(f"  Tremor variant: {len(audio_tremor)} samples")

        # Save example if soundfile available
        if SOUNDFILE_AVAILABLE:
            sf.write("/tmp/test_vowel_a.wav", audio, params.sample_rate)
            sf.write("/tmp/test_vowel_tremor.wav", audio_tremor, params_tremor.sample_rate)
            print("  Saved test files to /tmp/test_vowel_*.wav")

        print("  PASS")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_connected_speech():
    """Test connected speech generation"""
    print("Testing ConnectedSpeechGenerator...")

    if not WORLD_AVAILABLE:
        print("SKIP: pyworld not available")
        return False

    try:
        gen = ConnectedSpeechGenerator()

        params = ConnectedSpeechParams(
            duration_sec=2.0,
            base_f0=120.0,
            f0_range=50.0,
            speech_rate=1.0,
            sample_rate=16000,
            seed=42
        )

        audio, ground_truth = gen.generate(params)

        assert len(audio) > 0, "No audio generated"
        assert len(ground_truth.events) > 0, "No events"

        # Count syllables and pauses
        syllables = [e for e in ground_truth.events if e["type"] == "syllable"]
        pauses = [e for e in ground_truth.events if e["type"] == "pause"]

        print(f"  Generated {len(audio)} samples ({len(audio)/params.sample_rate:.2f} sec)")
        print(f"  Syllables: {len(syllables)}, Pauses: {len(pauses)}")

        # Test pathological speech (monotone + hypophonia)
        params_pd = ConnectedSpeechParams(
            duration_sec=2.0,
            base_f0=120.0,
            monotone_factor=0.8,
            hypophonia_db=6.0,
            speech_rate=0.8,
            seed=42
        )

        audio_pd, gt_pd = gen.generate(params_pd)
        assert len(audio_pd) > 0, "No PD audio generated"

        print(f"  PD variant: {len(audio_pd)} samples")

        if SOUNDFILE_AVAILABLE:
            sf.write("/tmp/test_speech_normal.wav", audio, params.sample_rate)
            sf.write("/tmp/test_speech_pd.wav", audio_pd, params_pd.sample_rate)
            print("  Saved test files to /tmp/test_speech_*.wav")

        print("  PASS")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diadochokinesis():
    """Test diadochokinesis generation"""
    print("Testing DiadochokinesisGenerator...")

    if not WORLD_AVAILABLE:
        print("SKIP: pyworld not available")
        return False

    try:
        gen = DiadochokinesisGenerator()

        # Test AMR (alternating motion rate) - single syllable
        params_amr = DiadochokinesisParams(
            syllables="pa",
            repetitions=10,
            target_rate=6.0,
            sample_rate=16000,
            seed=42
        )

        audio_amr, gt_amr = gen.generate(params_amr)

        assert len(audio_amr) > 0, "No AMR audio generated"
        assert len(gt_amr.events) == 10, f"Expected 10 events, got {len(gt_amr.events)}"

        # Verify syllable timings
        event_times = [e["time"] for e in gt_amr.events]
        intervals = np.diff(event_times)
        mean_interval = np.mean(intervals)
        expected_interval = 1.0 / params_amr.target_rate

        print(f"  AMR: {len(audio_amr)} samples")
        print(f"  Mean interval: {mean_interval:.3f} sec (expected ~{expected_interval:.3f})")

        # Test SMR (sequential motion rate) - syllable sequence
        params_smr = DiadochokinesisParams(
            syllables="pa-ta-ka",
            repetitions=5,
            target_rate=6.0,
            sample_rate=16000,
            seed=42
        )

        audio_smr, gt_smr = gen.generate(params_smr)

        assert len(audio_smr) > 0, "No SMR audio generated"
        assert len(gt_smr.events) == 15, f"Expected 15 events, got {len(gt_smr.events)}"  # 5 reps * 3 syllables

        print(f"  SMR: {len(audio_smr)} samples, {len(gt_smr.events)} syllables")

        if SOUNDFILE_AVAILABLE:
            sf.write("/tmp/test_ddk_amr.wav", audio_amr, params_amr.sample_rate)
            sf.write("/tmp/test_ddk_smr.wav", audio_smr, params_smr.sample_rate)
            print("  Saved test files to /tmp/test_ddk_*.wav")

        print("  PASS")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reading_passage():
    """Test reading passage generation"""
    print("Testing ReadingPassageGenerator...")

    if not WORLD_AVAILABLE:
        print("SKIP: pyworld not available")
        return False

    try:
        gen = ReadingPassageGenerator()

        params = ReadingPassageParams(
            passage="standard",
            base_f0=120.0,
            speech_rate=1.0,
            sample_rate=16000,
            seed=42
        )

        audio, ground_truth = gen.generate(params)

        assert len(audio) > 0, "No audio generated"

        print(f"  Generated {len(audio)} samples ({len(audio)/params.sample_rate:.2f} sec)")
        print(f"  F0 frames: {len(ground_truth.f0_contour)}")

        if SOUNDFILE_AVAILABLE:
            sf.write("/tmp/test_reading.wav", audio, params.sample_rate)
            print("  Saved test file to /tmp/test_reading.wav")

        print("  PASS")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reproducibility():
    """Test that generators are reproducible with the same seed"""
    print("Testing reproducibility...")

    if not WORLD_AVAILABLE:
        print("SKIP: pyworld not available")
        return False

    try:
        gen = SustainedVowelGenerator()
        params = VowelParams(duration_sec=0.5, seed=42)

        audio1, _ = gen.generate(params)
        audio2, _ = gen.generate(params)

        diff = np.abs(audio1 - audio2).max()
        assert diff < 1e-6, f"Not reproducible: max diff = {diff}"

        print(f"  Max difference: {diff:.2e}")
        print("  PASS")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Level 3 Audio Generator Tests")
    print("=" * 60)
    print()

    if not WORLD_AVAILABLE:
        print("ERROR: pyworld is not installed!")
        print("Install with: pip install pyworld")
        print()
        return 1

    tests = [
        test_sustained_vowel,
        test_connected_speech,
        test_diadochokinesis,
        test_reading_passage,
        test_reproducibility,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
