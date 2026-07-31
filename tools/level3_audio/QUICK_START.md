# Level 3 Audio Generators - Quick Start

## Installation

1. Install Python dependencies:
```bash
cd tools/level3_audio
pip install -r requirements.txt
```

2. Verify installation:
```bash
python test_generator.py
```

## Quick Examples

### Python Usage

```python
from tools.level3_audio.generator import SustainedVowelGenerator, VowelParams

# Generate sustained vowel with vocal tremor (PD-like)
gen = SustainedVowelGenerator()
params = VowelParams(
    duration_sec=3.0,
    f0_mean=120.0,
    tremor_frequency=5.0,   # 5 Hz tremor
    tremor_amplitude=10.0,  # 10 Hz amplitude
    vowel="a",
    seed=42
)

audio, ground_truth = gen.generate(params)

# Save to WAV
import soundfile as sf
sf.write("tremor_vowel.wav", audio, params.sample_rate)

# Access ground truth
print(f"Mean F0: {ground_truth.parameters['f0_mean']:.1f} Hz")
print(f"F0 contour length: {len(ground_truth.f0_contour)}")
```

### Rust Usage

```rust
use dpb_synth::level3::*;
use dpb_synth::SyntheticGenerator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create generator
    let gen = SustainedVowelGenerator::new();

    // Configure parameters
    let mut params = SustainedVowelParams::default();
    params.tremor_frequency = 5.0;
    params.tremor_amplitude = 10.0;

    // Generate with seed for reproducibility
    let result = gen.generate(&params, 42)?;

    println!("Generated {} samples", result.signal.len());
    println!("Mean F0: {:.1} Hz",
        result.ground_truth.f0_contour.iter()
            .filter(|&&f| f > 0.0)
            .sum::<f64>() / result.ground_truth.f0_contour.len() as f64
    );

    Ok(())
}
```

### Run Rust Demo

```bash
cd 
cargo run --example level3_audio_demo
```

## Generator Types

### 1. Sustained Vowel
- **Task**: Sustained /a/, /i/, /u/, /e/, /o/
- **Parameters**: F0, jitter, shimmer, HNR, tremor
- **Use case**: Voice quality assessment, PD screening

### 2. Connected Speech
- **Task**: Spontaneous speech simulation
- **Parameters**: Prosody, rate, hypophonia, monotone
- **Use case**: Speech naturalness, PD hypophonia

### 3. Diadochokinesis (DDK)
- **Task**: Rapid /pa/-/ta/-/ka/ repetition
- **Parameters**: Rate, variability, syllable pattern
- **Use case**: Motor speech assessment, coordination

### 4. Reading Passage
- **Task**: Reading task simulation
- **Parameters**: Rate, articulation, breath pauses
- **Use case**: Speech intelligibility, prosody

## Pathological Presets

### Parkinson's Disease

```python
# Vocal tremor
vowel_params = VowelParams(
    tremor_frequency=5.0,
    tremor_amplitude=10.0,
    jitter_percent=1.5,
)

# Monotone speech
speech_params = ConnectedSpeechParams(
    monotone_factor=0.8,
    hypophonia_db=6.0,
    speech_rate=0.8,
)

# Slow DDK
ddk_params = DiadochokinesisParams(
    target_rate=4.0,  # Reduced from normal 6.0
    rate_variability=0.3,  # Increased variability
)
```

### Vocal Fold Pathology

```python
# Breathy voice
vowel_params = VowelParams(
    hnr_db=10.0,  # Low HNR = breathy
    shimmer_percent=8.0,  # High shimmer
    jitter_percent=2.0,
)
```

## Testing

Run comprehensive tests:
```bash
python test_generator.py
```

Expected output:
```
=== Level 3 Audio Generator Tests ===

Testing SustainedVowelGenerator...
  Generated 16000 samples (1.00 sec)
  F0 mean: 120.0 Hz
  PASS

Testing ConnectedSpeechGenerator...
  Generated 32000 samples (2.00 sec)
  Syllables: 8, Pauses: 2
  PASS

Testing DiadochokinesisGenerator...
  AMR: 26667 samples
  Mean interval: 0.167 sec (expected ~0.167)
  PASS

Testing ReadingPassageGenerator...
  Generated 160000 samples (10.00 sec)
  PASS

Testing reproducibility...
  Max difference: 0.00e+00
  PASS

Results: 5/5 tests passed
```

## Output Files

Test audio files are saved to `/tmp/`:
- `test_vowel_a.wav` - Normal /a/ vowel
- `test_vowel_tremor.wav` - /a/ with tremor
- `test_speech_normal.wav` - Normal speech
- `test_speech_pd.wav` - PD-like speech
- `test_ddk_amr.wav` - AMR /pa/ repetition
- `test_ddk_smr.wav` - SMR /pa-ta-ka/ sequence
- `test_reading.wav` - Reading passage

## Ground Truth Structure

All generators provide comprehensive ground truth:

```python
@dataclass
class AudioGroundTruth:
    f0_contour: List[float]              # F0 at each 5ms frame
    f0_times: List[float]                # Time points (seconds)
    formants: Dict[str, List[float]]     # F1-F4 frequencies
    events: List[Dict]                   # Syllables, pauses, etc.
    parameters: Dict[str, float]         # Actual parameter values
```

Access ground truth:
```python
audio, gt = gen.generate(params)

# F0 contour
for t, f0 in zip(gt.f0_times, gt.f0_contour):
    print(f"Time {t:.3f}s: F0 = {f0:.1f} Hz")

# Formants
f1_values = gt.formants["F1"]
f2_values = gt.formants["F2"]

# Events
for event in gt.events:
    print(f"Event at {event['time']:.3f}s: {event['type']}")
```

## Troubleshooting

### pyworld installation fails

**Linux/macOS:**
```bash
# Install Cython first
pip install Cython
pip install pyworld
```

**Windows:**
```bash
# Install Visual C++ Build Tools first
pip install pyworld
```

### Python script not found (Rust)

The Rust interface expects the Python script at:
```
tools/level3_audio/generator.py
```

To use a custom path:
```rust
let gen = SustainedVowelGenerator::with_custom_python(
    PathBuf::from("/custom/path/generator.py"),
    "python3".to_string()
);
```

### No audio output

Check that:
1. pyworld is installed: `python -c "import pyworld"`
2. Python script is executable: `chmod +x tools/level3_audio/generator.py`
3. Python is accessible: `which python3`

## Performance

Typical generation times on modern CPU:
- Sustained vowel (3s): ~100ms
- Connected speech (5s): ~150ms
- DDK sequence (3s): ~80ms
- Reading passage (10s): ~300ms

All generators are deterministic with seed for reproducibility.

## References

See `README.md` for detailed documentation and clinical references.
