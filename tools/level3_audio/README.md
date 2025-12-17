# Level 3 Audio Generators

Full audio waveform synthesis using WORLD vocoder for complete pipeline testing.

## Overview

Level 3 generators produce actual audio signals with pathological characteristics for:
- Sustained vowel phonation (/a/, /i/, /u/)
- Connected speech with prosody
- Diadochokinesis (rapid syllable repetition)
- Reading passage tasks

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Note**: `pyworld` requires a C compiler and may need additional setup on some platforms.

### Platform-specific notes

**Linux/macOS:**
```bash
pip install pyworld
```

**Windows:**
May require Visual C++ build tools. See: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder

## Usage

### Python Direct Usage

```python
from generator import SustainedVowelGenerator, VowelParams

# Create generator
gen = SustainedVowelGenerator()

# Configure parameters
params = VowelParams(
    duration_sec=3.0,
    f0_mean=120.0,
    jitter_percent=1.0,
    tremor_frequency=5.0,  # 5 Hz tremor (PD-like)
    tremor_amplitude=10.0,  # 10 Hz amplitude
    vowel="a",
    seed=42
)

# Generate audio
audio, ground_truth = gen.generate(params)

# Save to file
import soundfile as sf
sf.write("vowel.wav", audio, params.sample_rate)
```

### Rust Interface

```rust
use dpb_synth::level3::*;
use dpb_synth::SyntheticGenerator;

// Create generator
let gen = SustainedVowelGenerator::new();

// Use default parameters or customize
let mut params = SustainedVowelParams::default();
params.tremor_frequency = 5.0;
params.tremor_amplitude = 10.0;

// Generate with seed
let result = gen.generate(&params, 42)?;

// Access audio samples
let audio = result.signal;  // Array1<f32>
let f0_contour = result.ground_truth.f0_contour;
```

### Command Line

```bash
python generator.py sustained_vowel '{"duration_sec": 3.0, "f0_mean": 120.0, "vowel": "a", "seed": 42}'
```

## Testing

Run the test suite:

```bash
python test_generator.py
```

This will generate test audio files in `/tmp/` (if soundfile is installed) and verify all generators work correctly.

## Generators

### 1. SustainedVowelGenerator

Generates sustained vowels with:
- **F0 control**: Mean, std, jitter, tremor
- **Formants**: Vowel-specific F1-F4
- **Quality**: Shimmer, HNR (breathiness)

**Parameters:**
- `duration_sec`: Duration in seconds (default: 3.0)
- `f0_mean`: Mean F0 in Hz (default: 120.0)
- `f0_std`: F0 standard deviation (default: 2.0)
- `jitter_percent`: Cycle-to-cycle F0 variation (default: 0.5)
- `shimmer_percent`: Cycle-to-cycle amplitude variation (default: 3.0)
- `hnr_db`: Harmonics-to-noise ratio (default: 22.0)
- `tremor_frequency`: Tremor frequency in Hz (default: 0.0)
- `tremor_amplitude`: Tremor amplitude in Hz (default: 0.0)
- `vowel`: Vowel to generate: "a", "i", "u", "e", "o" (default: "a")

**Pathological Variations:**
- PD monotone: Low f0_std, high jitter
- Vocal tremor: tremor_frequency=4-6 Hz
- Breathiness: Low hnr_db

### 2. ConnectedSpeechGenerator

Generates speech-like signals with:
- Syllable patterns
- Prosodic variation
- Pauses

**Parameters:**
- `duration_sec`: Duration in seconds (default: 5.0)
- `base_f0`: Base F0 in Hz (default: 120.0)
- `f0_range`: F0 variation range (default: 50.0)
- `speech_rate`: Speed multiplier (default: 1.0)
- `hypophonia_db`: Amplitude reduction (default: 0.0)
- `monotone_factor`: 0-1, reduces F0 variation (default: 0.0)
- `tremor_frequency`: Tremor frequency (default: 0.0)
- `pause_probability`: Probability of pause (default: 0.1)

**Pathological Variations:**
- Hypophonia: High hypophonia_db
- Monotone: High monotone_factor
- Slow speech: Low speech_rate

### 3. DiadochokinesisGenerator

Generates rapid syllable repetition:

**Parameters:**
- `syllables`: Syllable sequence (default: "pa-ta-ka")
  - AMR (alternating): "pa", "ta", or "ka"
  - SMR (sequential): "pa-ta-ka"
- `repetitions`: Number of repetitions (default: 10)
- `target_rate`: Syllables per second (default: 6.0)
- `rate_variability`: CV of durations (default: 0.1)
- `amplitude_variability`: CV of amplitudes (default: 0.05)

**Clinical Standards:**
- Normal AMR: 5-7 syllables/second
- Normal SMR: 1.5-2.5 sequences/second
- PD patients: Reduced rate, increased variability

### 4. ReadingPassageGenerator

Generates reading task audio:

**Parameters:**
- `passage`: Passage type (default: "standard")
- `base_f0`: Base F0 (default: 120.0)
- `speech_rate`: Reading speed (default: 1.0)
- `articulation_precision`: 0-1 (default: 1.0)
- `breath_pause_regularity`: 0-1 (default: 1.0)

## Ground Truth

All generators provide:

```python
@dataclass
class AudioGroundTruth:
    f0_contour: List[float]          # F0 at each frame
    f0_times: List[float]            # Time points
    formants: Dict[str, List[float]] # F1-F4 frequencies
    events: List[Dict]               # Syllables, pauses, etc.
    parameters: Dict[str, float]     # Generator parameters
```

## Implementation Details

### WORLD Vocoder

Uses the WORLD vocoder (Morise et al., 2016) for high-quality speech synthesis:
- **F0 estimation**: Fundamental frequency contour
- **Spectral envelope**: Formant structure
- **Aperiodicity**: Voice quality (HNR, breathiness)

### Reproducibility

All generators are fully reproducible with seed parameter:

```python
# Same seed = same output
audio1, _ = gen.generate(VowelParams(seed=42))
audio2, _ = gen.generate(VowelParams(seed=42))
assert np.allclose(audio1, audio2)
```

## Clinical Validity

Pathological parameters based on:
- Parkinson's Disease: Tremor (4-6 Hz), monotone, hypophonia
- Vocal fold pathology: High jitter/shimmer, low HNR
- Dysarthria: Slow rate, reduced articulation precision

## References

1. Morise, M., Yokomori, F., & Ozawa, K. (2016). WORLD: a vocoder-based high-quality speech synthesis system for real-time applications. IEICE Transactions on Information and Systems.

2. Skodda, S., Visser, W., & Schlegel, U. (2011). Vowel articulation in Parkinson's disease. Journal of Voice, 25(4), 467-472.

3. Rusz, J., Cmejla, R., Ruzickova, H., & Ruzicka, E. (2011). Quantitative acoustic measurements for characterization of speech and voice disorders in early untreated Parkinson's disease. The Journal of the Acoustical Society of America, 129(1), 350-367.
