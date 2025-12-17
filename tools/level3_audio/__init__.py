"""
Level 3 Audio Generators for DPB Framework

This package provides full audio waveform synthesis using WORLD vocoder
for complete pipeline testing with pathological voice characteristics.
"""

from .generator import (
    SustainedVowelGenerator,
    ConnectedSpeechGenerator,
    DiadochokinesisGenerator,
    ReadingPassageGenerator,
    VowelParams,
    ConnectedSpeechParams,
    DiadochokinesisParams,
    ReadingPassageParams,
    AudioGroundTruth,
    WORLD_AVAILABLE,
)

__all__ = [
    "SustainedVowelGenerator",
    "ConnectedSpeechGenerator",
    "DiadochokinesisGenerator",
    "ReadingPassageGenerator",
    "VowelParams",
    "ConnectedSpeechParams",
    "DiadochokinesisParams",
    "ReadingPassageParams",
    "AudioGroundTruth",
    "WORLD_AVAILABLE",
]

__version__ = "0.1.0"
