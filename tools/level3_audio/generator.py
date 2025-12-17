#!/usr/bin/env python3
"""
Level 3 Audio Generators for DPB Framework

Generates actual audio waveforms using the WORLD vocoder for full pipeline testing.
Supports pathological voice characteristics for Parkinson's Disease and other conditions.
"""

import numpy as np
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import warnings

# Conditional imports
try:
    import pyworld as pw
    WORLD_AVAILABLE = True
except ImportError:
    WORLD_AVAILABLE = False
    warnings.warn("pyworld not available - WORLD vocoder features disabled")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    warnings.warn("soundfile not available - WAV export disabled")

from scipy import signal
from scipy.interpolate import interp1d


@dataclass
class VowelParams:
    """Parameters for sustained vowel generation"""
    duration_sec: float = 3.0
    f0_mean: float = 120.0  # Hz
    f0_std: float = 2.0
    jitter_percent: float = 0.5
    shimmer_percent: float = 3.0
    hnr_db: float = 22.0
    tremor_frequency: float = 0.0  # Hz, 0 = no tremor
    tremor_amplitude: float = 0.0  # Hz
    vowel: str = "a"  # a, i, u, e, o
    sample_rate: int = 16000
    seed: int = 42


@dataclass
class ConnectedSpeechParams:
    """Parameters for connected speech generation"""
    duration_sec: float = 5.0
    base_f0: float = 120.0
    f0_range: float = 50.0  # Hz variation
    speech_rate: float = 1.0  # 1.0 = normal, <1 = slower
    hypophonia_db: float = 0.0  # Reduction in amplitude
    monotone_factor: float = 0.0  # 0-1, reduces F0 variation
    tremor_frequency: float = 0.0
    tremor_amplitude: float = 0.0
    pause_probability: float = 0.1
    sample_rate: int = 16000
    seed: int = 42


@dataclass
class DiadochokinesisParams:
    """Parameters for rapid syllable repetition (AMR/SMR)"""
    syllables: str = "pa-ta-ka"  # pa, ta, ka, or combinations
    repetitions: int = 10
    target_rate: float = 6.0  # syllables per second
    rate_variability: float = 0.1  # CV of syllable durations
    amplitude_variability: float = 0.05  # CV of amplitudes
    f0_mean: float = 120.0
    sample_rate: int = 16000
    seed: int = 42


@dataclass
class ReadingPassageParams:
    """Parameters for reading passage generation"""
    passage: str = "standard"  # standard, rainbow, grandfather
    base_f0: float = 120.0
    speech_rate: float = 1.0
    articulation_precision: float = 1.0  # 0-1, affects formant clarity
    breath_pause_regularity: float = 1.0  # 0-1, affects pause timing
    sample_rate: int = 16000
    seed: int = 42


@dataclass
class AudioGroundTruth:
    """Ground truth for generated audio"""
    f0_contour: List[float]  # F0 values at each time frame
    f0_times: List[float]  # Time points for F0 values
    formants: Optional[Dict[str, List[float]]] = None  # F1, F2, F3, F4
    events: Optional[List[Dict[str, float]]] = None  # Event markers
    parameters: Optional[Dict[str, float]] = None  # Generator parameters
    spectral_features: Optional[Dict[str, List[float]]] = None


class SustainedVowelGenerator:
    """
    Generate sustained vowel utterances with pathological characteristics.

    Uses WORLD vocoder to synthesize vowels with controlled:
    - F0 (pitch) with jitter and tremor
    - Formants (F1-F4)
    - Shimmer and HNR (breathiness)
    """

    # Formant frequencies for vowels (F1, F2, F3, F4) in Hz
    # Based on Peterson & Barney (1952) for adult males
    VOWEL_FORMANTS = {
        "a": [730, 1090, 2440, 3400],   # /ɑ/ as in "father"
        "i": [270, 2290, 3010, 3500],   # /i/ as in "feet"
        "u": [300, 870, 2240, 3400],    # /u/ as in "boot"
        "e": [530, 1840, 2480, 3500],   # /ɛ/ as in "bed"
        "o": [570, 840, 2410, 3400],    # /ɔ/ as in "bought"
    }

    def __init__(self):
        if not WORLD_AVAILABLE:
            raise ImportError("pyworld is required for audio generation")

    def generate(self, params: VowelParams) -> Tuple[np.ndarray, AudioGroundTruth]:
        """Generate sustained vowel with ground truth"""
        rng = np.random.RandomState(params.seed)

        # Calculate number of samples and frames
        n_samples = int(params.duration_sec * params.sample_rate)
        frame_period = 5.0  # ms
        n_frames = int(params.duration_sec * 1000 / frame_period)

        # Generate F0 contour
        f0 = self._generate_f0(n_frames, params, rng)
        f0_times = np.arange(n_frames) * frame_period / 1000.0

        # Generate spectral envelope (formants)
        sp = self._generate_spectral_envelope(n_frames, params, rng)

        # Generate aperiodicity (relates to HNR)
        ap = self._generate_aperiodicity(n_frames, params, rng)

        # Synthesize using WORLD
        audio = pw.synthesize(f0, sp, ap, params.sample_rate, frame_period)

        # Apply shimmer (amplitude variation)
        if params.shimmer_percent > 0:
            audio = self._apply_shimmer(audio, params, rng)

        # Normalize to [-1, 1]
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Create ground truth
        ground_truth = AudioGroundTruth(
            f0_contour=f0.tolist(),
            f0_times=f0_times.tolist(),
            formants={
                f"F{i+1}": [self.VOWEL_FORMANTS[params.vowel][i]] * n_frames
                for i in range(4)
            },
            parameters={
                "f0_mean": float(np.mean(f0[f0 > 0])),
                "f0_std": float(np.std(f0[f0 > 0])),
                "jitter_percent": params.jitter_percent,
                "shimmer_percent": params.shimmer_percent,
                "hnr_db": params.hnr_db,
                "tremor_frequency": params.tremor_frequency,
                "tremor_amplitude": params.tremor_amplitude,
            }
        )

        return audio.astype(np.float32), ground_truth

    def _generate_f0(self, n_frames: int, params: VowelParams, rng) -> np.ndarray:
        """Generate F0 contour with jitter and tremor"""
        time = np.arange(n_frames) * 5.0 / 1000.0  # Time in seconds

        # Base F0
        f0 = np.ones(n_frames) * params.f0_mean

        # Add slow variation (normal physiological variation)
        slow_variation = params.f0_std * rng.randn(n_frames)
        slow_variation = signal.savgol_filter(slow_variation, 11, 3, mode='nearest')
        f0 += slow_variation

        # Add tremor (pathological rhythmic variation)
        if params.tremor_frequency > 0 and params.tremor_amplitude > 0:
            tremor = params.tremor_amplitude * np.sin(2 * np.pi * params.tremor_frequency * time)
            f0 += tremor

        # Add jitter (cycle-to-cycle variation)
        if params.jitter_percent > 0:
            jitter_std = params.f0_mean * params.jitter_percent / 100.0
            jitter = rng.randn(n_frames) * jitter_std
            f0 += jitter

        # Ensure F0 is positive
        f0 = np.maximum(f0, 50.0)

        return f0

    def _generate_spectral_envelope(self, n_frames: int, params: VowelParams, rng) -> np.ndarray:
        """Generate spectral envelope with formants"""
        # Get formant frequencies for this vowel
        if params.vowel not in self.VOWEL_FORMANTS:
            raise ValueError(f"Unknown vowel: {params.vowel}")

        formants = self.VOWEL_FORMANTS[params.vowel]

        # Create spectral envelope
        # WORLD expects sp shape: (n_frames, fft_size//2 + 1)
        fft_size = pw.get_cheaptrick_fft_size(params.sample_rate)
        sp = np.zeros((n_frames, fft_size // 2 + 1))

        freqs = np.fft.rfftfreq(fft_size, 1.0 / params.sample_rate)

        for frame in range(n_frames):
            # Start with flat spectrum
            spectrum = np.ones_like(freqs) * -60  # dB

            # Add formant peaks
            for i, f_freq in enumerate(formants):
                # Formant bandwidth (increases with frequency)
                bandwidth = 50 + f_freq * 0.05

                # Gaussian formant shape
                formant_peak = 40 * np.exp(-((freqs - f_freq) / bandwidth) ** 2)
                spectrum += formant_peak

            # Add slight spectral tilt (more energy in low frequencies)
            tilt = -6 * np.log10(freqs / 100.0 + 1)
            spectrum += tilt

            # Convert dB to linear magnitude
            sp[frame, :] = 10 ** (spectrum / 20.0)

        return sp

    def _generate_aperiodicity(self, n_frames: int, params: VowelParams, rng) -> np.ndarray:
        """Generate aperiodicity based on HNR"""
        # WORLD aperiodicity: 0 = periodic, 1 = aperiodic
        # HNR relates to harmonics-to-noise ratio

        # Convert HNR (dB) to aperiodicity ratio
        # HNR = 10 * log10(harmonic_power / noise_power)
        # Higher HNR = more periodic = lower aperiodicity
        noise_ratio = 10 ** (-params.hnr_db / 10.0)
        aperiodicty_value = noise_ratio / (1 + noise_ratio)

        fft_size = pw.get_cheaptrick_fft_size(params.sample_rate)
        ap = np.ones((n_frames, fft_size // 2 + 1)) * aperiodicty_value

        # Add slight random variation
        ap += rng.randn(n_frames, fft_size // 2 + 1) * 0.01
        ap = np.clip(ap, 0, 1)

        return ap

    def _apply_shimmer(self, audio: np.ndarray, params: VowelParams, rng) -> np.ndarray:
        """Apply shimmer (amplitude variation)"""
        # Shimmer is cycle-to-cycle amplitude variation
        # We approximate by applying slow amplitude modulation

        n_samples = len(audio)

        # Create amplitude modulation envelope
        # Modulation at pitch frequency to simulate cycle-to-cycle variation
        time = np.arange(n_samples) / params.sample_rate

        # Use low-frequency noise for shimmer
        shimmer_freq = params.f0_mean / 10.0  # Slow modulation
        n_noise_samples = int(n_samples / (params.sample_rate / shimmer_freq))
        noise = rng.randn(n_noise_samples) * params.shimmer_percent / 100.0

        # Interpolate to match audio length
        noise_time = np.linspace(0, time[-1], n_noise_samples)
        interp_func = interp1d(noise_time, noise, kind='cubic', fill_value='extrapolate')
        amplitude_mod = 1.0 + interp_func(time)

        return audio * amplitude_mod


class ConnectedSpeechGenerator:
    """
    Generate connected speech with pathological characteristics.

    Uses simplified speech model with WORLD vocoder modifications.
    """

    def __init__(self):
        if not WORLD_AVAILABLE:
            raise ImportError("pyworld is required for audio generation")

    def generate(self, params: ConnectedSpeechParams) -> Tuple[np.ndarray, AudioGroundTruth]:
        """Generate connected speech with ground truth"""
        rng = np.random.RandomState(params.seed)

        n_samples = int(params.duration_sec * params.sample_rate)
        frame_period = 5.0  # ms
        n_frames = int(params.duration_sec * 1000 / frame_period)

        # Generate speech-like F0 contour
        f0, f0_times, events = self._generate_speech_f0(n_frames, params, rng)

        # Generate spectral envelope (simplified speech formants)
        sp = self._generate_speech_spectrum(n_frames, params, rng)

        # Generate aperiodicity
        ap = self._generate_speech_aperiodicity(n_frames, params, rng)

        # Synthesize using WORLD
        audio = pw.synthesize(f0, sp, ap, params.sample_rate, frame_period)

        # Apply hypophonia (reduced amplitude)
        if params.hypophonia_db > 0:
            reduction_factor = 10 ** (-params.hypophonia_db / 20.0)
            audio = audio * reduction_factor

        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        ground_truth = AudioGroundTruth(
            f0_contour=f0.tolist(),
            f0_times=f0_times.tolist(),
            events=events,
            parameters={
                "base_f0": params.base_f0,
                "f0_range": params.f0_range,
                "speech_rate": params.speech_rate,
                "hypophonia_db": params.hypophonia_db,
                "monotone_factor": params.monotone_factor,
            }
        )

        return audio.astype(np.float32), ground_truth

    def _generate_speech_f0(self, n_frames: int, params: ConnectedSpeechParams,
                           rng) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Generate speech-like F0 contour"""
        time = np.arange(n_frames) * 5.0 / 1000.0
        f0 = np.zeros(n_frames)
        events = []

        # Generate syllable-like patterns
        syllable_duration = 0.2 / params.speech_rate  # seconds
        frames_per_syllable = int(syllable_duration * 1000 / 5.0)

        current_frame = 0
        syllable_count = 0

        while current_frame < n_frames:
            # Decide if this is a pause
            if rng.rand() < params.pause_probability:
                # Pause (unvoiced)
                pause_frames = int(rng.uniform(0.1, 0.3) * 1000 / 5.0)
                pause_frames = min(pause_frames, n_frames - current_frame)
                f0[current_frame:current_frame + pause_frames] = 0

                events.append({
                    "time": time[current_frame],
                    "type": "pause",
                    "duration": pause_frames * 5.0 / 1000.0
                })

                current_frame += pause_frames
            else:
                # Voiced syllable
                syl_frames = min(frames_per_syllable, n_frames - current_frame)

                # F0 contour for syllable (rise-fall pattern)
                syl_time = np.linspace(0, 1, syl_frames)

                # Base F0 with pitch accent
                f0_range_actual = params.f0_range * (1 - params.monotone_factor)
                accent_height = rng.uniform(-0.5, 1.0) * f0_range_actual

                syl_f0 = params.base_f0 + accent_height * np.sin(np.pi * syl_time)

                # Add tremor if specified
                if params.tremor_frequency > 0 and params.tremor_amplitude > 0:
                    tremor = params.tremor_amplitude * np.sin(
                        2 * np.pi * params.tremor_frequency * time[current_frame:current_frame + syl_frames]
                    )
                    syl_f0 += tremor

                f0[current_frame:current_frame + syl_frames] = syl_f0

                events.append({
                    "time": time[current_frame],
                    "type": "syllable",
                    "f0_mean": float(np.mean(syl_f0))
                })

                current_frame += syl_frames
                syllable_count += 1

        return f0, time, events

    def _generate_speech_spectrum(self, n_frames: int, params: ConnectedSpeechParams, rng) -> np.ndarray:
        """Generate speech-like spectral envelope"""
        fft_size = pw.get_cheaptrick_fft_size(params.sample_rate)
        sp = np.zeros((n_frames, fft_size // 2 + 1))
        freqs = np.fft.rfftfreq(fft_size, 1.0 / params.sample_rate)

        # Alternate between vowel-like and consonant-like spectra
        vowel_formants = [
            [700, 1200, 2600, 3500],   # /a/-like
            [400, 2000, 2800, 3500],   # schwa-like
            [500, 1500, 2500, 3500],   # neutral
        ]

        for frame in range(n_frames):
            # Choose formant set (simulate phoneme changes)
            formant_idx = (frame // 20) % len(vowel_formants)
            formants = vowel_formants[formant_idx]

            spectrum = np.ones_like(freqs) * -60

            for f_freq in formants:
                bandwidth = 50 + f_freq * 0.05
                formant_peak = 40 * np.exp(-((freqs - f_freq) / bandwidth) ** 2)
                spectrum += formant_peak

            tilt = -6 * np.log10(freqs / 100.0 + 1)
            spectrum += tilt

            sp[frame, :] = 10 ** (spectrum / 20.0)

        return sp

    def _generate_speech_aperiodicity(self, n_frames: int, params: ConnectedSpeechParams, rng) -> np.ndarray:
        """Generate speech-like aperiodicity (voiced/unvoiced patterns)"""
        fft_size = pw.get_cheaptrick_fft_size(params.sample_rate)
        ap = np.zeros((n_frames, fft_size // 2 + 1))

        for frame in range(n_frames):
            # More aperiodicity in high frequencies (like fricatives)
            ap[frame, :] = np.linspace(0.1, 0.5, fft_size // 2 + 1)

        return ap


class DiadochokinesisGenerator:
    """
    Generate diadochokinesis (DDK) tasks - rapid alternating/sequential motion rates.

    Tests: /pa/-/pa/-/pa/ (AMR), /pa/-/ta/-/ka/ (SMR)
    """

    # Formant patterns for consonants (very simplified)
    SYLLABLE_FORMANTS = {
        "pa": [700, 1200, 2600, 3500],   # Bilabial + /a/
        "ta": [700, 1700, 2600, 3500],   # Alveolar + /a/
        "ka": [700, 2500, 2900, 3500],   # Velar + /a/
    }

    def __init__(self):
        if not WORLD_AVAILABLE:
            raise ImportError("pyworld is required for audio generation")

    def generate(self, params: DiadochokinesisParams) -> Tuple[np.ndarray, AudioGroundTruth]:
        """Generate DDK sequence with ground truth"""
        rng = np.random.RandomState(params.seed)

        # Parse syllable sequence
        syllables = params.syllables.split('-')

        # Calculate timing
        mean_syllable_duration = 1.0 / params.target_rate
        total_duration = mean_syllable_duration * params.repetitions * len(syllables)

        frame_period = 5.0  # ms
        n_frames = int(total_duration * 1000 / frame_period)

        # Generate sequence
        f0, sp, ap, events = self._generate_ddk_sequence(
            syllables, params, n_frames, rng
        )

        f0_times = np.arange(n_frames) * frame_period / 1000.0

        # Synthesize
        audio = pw.synthesize(f0, sp, ap, params.sample_rate, frame_period)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        ground_truth = AudioGroundTruth(
            f0_contour=f0.tolist(),
            f0_times=f0_times.tolist(),
            events=events,
            parameters={
                "target_rate": params.target_rate,
                "actual_rate": len(events) / total_duration,
                "rate_variability": params.rate_variability,
                "amplitude_variability": params.amplitude_variability,
            }
        )

        return audio.astype(np.float32), ground_truth

    def _generate_ddk_sequence(self, syllables: List[str], params: DiadochokinesisParams,
                               n_frames: int, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """Generate DDK syllable sequence"""
        fft_size = pw.get_cheaptrick_fft_size(params.sample_rate)
        f0 = np.zeros(n_frames)
        sp = np.zeros((n_frames, fft_size // 2 + 1))
        ap = np.ones((n_frames, fft_size // 2 + 1)) * 0.2

        events = []
        frame_period = 5.0  # ms

        mean_duration = 1.0 / params.target_rate
        current_time = 0.0

        for rep in range(params.repetitions):
            for syl in syllables:
                # Vary duration
                duration = mean_duration * (1 + rng.randn() * params.rate_variability)
                duration = max(0.05, duration)  # Minimum 50ms

                # Vary amplitude
                amplitude = 1.0 + rng.randn() * params.amplitude_variability
                amplitude = max(0.5, amplitude)

                start_frame = int(current_time * 1000 / frame_period)
                end_frame = int((current_time + duration) * 1000 / frame_period)
                end_frame = min(end_frame, n_frames)

                if start_frame >= n_frames:
                    break

                n_syl_frames = end_frame - start_frame

                # Generate F0 for this syllable
                f0[start_frame:end_frame] = params.f0_mean

                # Generate spectrum
                if syl in self.SYLLABLE_FORMANTS:
                    formants = self.SYLLABLE_FORMANTS[syl]
                else:
                    formants = self.SYLLABLE_FORMANTS["pa"]

                freqs = np.fft.rfftfreq(fft_size, 1.0 / params.sample_rate)

                for frame in range(start_frame, end_frame):
                    spectrum = np.ones_like(freqs) * -60

                    for f_freq in formants:
                        bandwidth = 80
                        formant_peak = 40 * amplitude * np.exp(-((freqs - f_freq) / bandwidth) ** 2)
                        spectrum += formant_peak

                    sp[frame, :] = 10 ** (spectrum / 20.0)

                events.append({
                    "time": current_time,
                    "type": "syllable",
                    "syllable": syl,
                    "duration": duration,
                    "amplitude": amplitude
                })

                current_time += duration

        return f0, sp, ap, events


class ReadingPassageGenerator:
    """
    Generate reading passage with pathological characteristics.

    Simplified version - generates prosodic patterns similar to reading.
    """

    PASSAGES = {
        "standard": "The standard reading passage",
        "rainbow": "The Rainbow Passage",
        "grandfather": "The Grandfather Passage"
    }

    def __init__(self):
        if not WORLD_AVAILABLE:
            raise ImportError("pyworld is required for audio generation")

    def generate(self, params: ReadingPassageParams) -> Tuple[np.ndarray, AudioGroundTruth]:
        """Generate reading passage with ground truth"""
        # Use connected speech generator with reading-specific patterns
        speech_params = ConnectedSpeechParams(
            duration_sec=10.0,  # Typical passage length
            base_f0=params.base_f0,
            f0_range=30.0,
            speech_rate=params.speech_rate,
            pause_probability=0.15,  # More pauses in reading
            sample_rate=params.sample_rate,
            seed=params.seed
        )

        speech_gen = ConnectedSpeechGenerator()
        audio, ground_truth = speech_gen.generate(speech_params)

        # Add reading-specific metadata
        ground_truth.parameters["passage"] = params.passage
        ground_truth.parameters["articulation_precision"] = params.articulation_precision

        return audio, ground_truth


def main():
    """CLI interface for generator"""
    if len(sys.argv) < 3:
        print("Usage: generator.py <generator_type> <params_json>", file=sys.stderr)
        sys.exit(1)

    generator_type = sys.argv[1]
    params_json = sys.argv[2]

    try:
        params_dict = json.loads(params_json)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if generator_type == "sustained_vowel":
            params = VowelParams(**params_dict)
            gen = SustainedVowelGenerator()
            audio, ground_truth = gen.generate(params)

        elif generator_type == "connected_speech":
            params = ConnectedSpeechParams(**params_dict)
            gen = ConnectedSpeechGenerator()
            audio, ground_truth = gen.generate(params)

        elif generator_type == "diadochokinesis":
            params = DiadochokinesisParams(**params_dict)
            gen = DiadochokinesisGenerator()
            audio, ground_truth = gen.generate(params)

        elif generator_type == "reading_passage":
            params = ReadingPassageParams(**params_dict)
            gen = ReadingPassageGenerator()
            audio, ground_truth = gen.generate(params)

        else:
            print(f"Unknown generator type: {generator_type}", file=sys.stderr)
            sys.exit(1)

        # Output results as JSON
        result = {
            "audio": audio.tolist(),
            "sample_rate": params.sample_rate,
            "ground_truth": {
                "f0_contour": ground_truth.f0_contour,
                "f0_times": ground_truth.f0_times,
                "formants": ground_truth.formants,
                "events": ground_truth.events,
                "parameters": ground_truth.parameters,
            }
        }

        print(json.dumps(result))

    except Exception as e:
        print(f"Error generating audio: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
