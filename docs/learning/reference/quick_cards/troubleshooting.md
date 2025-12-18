# Quick Reference Card 5: Common Problems & Fixes 🔧

> **FAQ-style troubleshooting** - When things go wrong, start here!

---

## 1. Signal Quality Problems

### 🔴 Problem: "My signal looks flat"

**What you see:**
```
Expected:    /\  /\  /\
           _/  \/  \/  \

Actual:    _____________
```

**Possible Causes & Fixes:**

✓ **Check electrode connection**
- Are electrodes properly attached?
- Is there gel/paste on electrodes?
- Are wires plugged in correctly?

✓ **Check input scaling**
```python
# Try scaling the signal
signal = signal * 1000  # Convert V to mV
print(f"Range: {signal.min():.3f} to {signal.max():.3f}")
```

✓ **Check correct channel**
```python
# Make sure you're reading the right channel
signal = reader.read_signal(0)  # Try channel 0, 1, 2...
```

✓ **Verify device is on and recording**
- Is the device powered?
- Is it in recording mode (not paused)?

---

### 🔴 Problem: "Too much noise - can't see the signal"

**What you see:**
```
Expected:     /\  /\  /\
           __/  \/  \/  \

Actual:    /\/\/\/\/\/\/\
           Noisy mess!
```

**Fixes by Noise Type:**

#### High-Frequency Noise (Fuzzy/Hairy)
```python
# Apply lowpass or bandpass filter
filt = dpb.signal.IirFilter.bandpass(
    low_freq=0.5,
    high_freq=40.0,      # ← Adjust this
    sample_rate=250.0,
    order=4
)
clean = filt.apply(signal)
```

#### 50/60 Hz Power Line Noise (Regular Waves)
```python
# Add notch filter for power line
notch = dpb.signal.IirFilter.notch(
    freq=60.0,           # Or 50 Hz in Europe
    sample_rate=250.0,
    quality=30.0
)
clean = notch.apply(signal)
```

#### Baseline Wander (Slow Drift)
```python
# 1. Remove DC offset
clean = dpb.signal.remove_dc_offset(signal)

# 2. Use highpass filter
highpass = dpb.signal.IirFilter.highpass(
    freq=0.5,            # Remove drift < 0.5 Hz
    sample_rate=250.0,
    order=4
)
clean = highpass.apply(signal)
```

#### Motion Artifacts (Random Spikes)
```
Signal:  __|¯\    /\     spike!
            \_/\/\       ↑
```
- **Fix:** Stabilize device/subject
- **Fix:** Use artifact detection and removal
```python
# Detect and interpolate artifacts
artifacts = dpb.signal.eeg.detect_artifacts(signal, sample_rate)
clean = interpolate_artifacts(signal, artifacts)
```

---

### 🔴 Problem: "Signal amplitude is way too high/low"

**What you see:**
```
Expected amplitude: 0.5-1.0 mV
Your signal: 500-1000 mV (or 0.0005 mV)
```

**Fixes:**

✓ **Check units and scale**
```python
# Print signal stats
print(f"Min: {signal.min()}")
print(f"Max: {signal.max()}")
print(f"Mean: {signal.mean()}")
print(f"Std: {signal.std()}")

# Common unit conversions
signal_mv = signal * 1000      # V → mV
signal_uv = signal * 1_000_000 # V → µV
```

✓ **Normalize if needed**
```python
# Z-score normalization (mean=0, std=1)
normalized = (signal - signal.mean()) / signal.std()

# Min-Max scaling (0 to 1)
normalized = (signal - signal.min()) / (signal.max() - signal.min())
```

---

## 2. Peak Detection Problems

### 🔴 Problem: "Can't find peaks" or "Finding wrong peaks"

**What you see:**
```
Signal:     R   R   R
          /\  /\  /\
         /  \/  \/  \
Detected: ●     ●            ← Missing peaks!

OR

Detected: ● ● ● ● ● ● ●      ← Too many!
```

**Fixes:**

✓ **Make sure signal is filtered first!**
```python
# ALWAYS filter before detecting
filt = dpb.signal.IirFilter.bandpass(0.5, 40.0, 250.0, 4)
clean = filt.apply(signal)

# Then detect on filtered signal
peaks = detector.detect(clean)  # ← Use clean, not raw!
```

✓ **Adjust detection threshold**
```python
# For generic peak detection
peaks = dpb.signal.find_peaks(
    signal,
    height=0.5,          # ← Try different values: 0.3, 0.7, 1.0
    distance=150,        # Minimum samples between peaks
    prominence=0.2       # How much peak stands out
)

# Too few peaks? Lower height/prominence
# Too many peaks? Raise height/prominence
```

✓ **Visual debugging**
```python
import matplotlib.pyplot as plt

# Plot to see what's happening
plt.figure(figsize=(12, 4))
plt.plot(signal, label='Signal')
plt.plot(peaks, signal[peaks], 'ro', label='Detected', markersize=8)
plt.legend()
plt.title(f"Found {len(peaks)} peaks")
plt.show()

# Do the red dots look right?
```

✓ **Check sample rate is correct**
```python
# Wrong sample rate = wrong peak spacing
detector = dpb.signal.ecg.PanTompkinsDetector(
    sample_rate=250.0    # ← Make sure this is correct!
)
```

---

### 🔴 Problem: "Heart rate doesn't match expected"

**What you see:**
```
Expected: 70 bpm
Your result: 140 bpm (or 35 bpm)
```

**Fixes:**

✓ **Double-check sample rate**
```python
# Wrong sample rate = wrong timing
sample_rate = 250.0  # Verify this is correct!

# Calculate heart rate
rr_intervals = np.diff(peaks) / sample_rate  # seconds
heart_rate = 60.0 / np.mean(rr_intervals)    # bpm
```

✓ **Check for duplicate detections**
```python
# Are you detecting both R and S peaks?
print(f"Average peak distance: {np.mean(np.diff(peaks)):.1f} samples")
expected = sample_rate * 0.8  # ~0.8 sec between beats at 75 bpm
print(f"Expected distance: {expected:.1f} samples")

# If actual << expected, you're finding too many peaks
```

✓ **Verify enough data**
```python
# Need at least 10 seconds for reliable HR
duration = len(signal) / sample_rate
if duration < 10:
    print(f"Warning: Only {duration:.1f} seconds of data!")
    print("Need at least 10 seconds for accurate heart rate")
```

---

## 3. Filtering Problems

### 🔴 Problem: "Filter makes signal worse"

**What you see:**
```
Before filter: Clear peaks
After filter: Peaks gone or distorted!
```

**Fixes:**

✓ **Check filter range is appropriate**
```python
# Don't filter out your signal!

# ECG example - R-peaks are ~10-20 Hz
good_filter = dpb.signal.IirFilter.bandpass(
    0.5, 40.0,  # ✓ Includes 10-20 Hz
    250.0, 4
)

bad_filter = dpb.signal.IirFilter.bandpass(
    30.0, 100.0,  # ✗ Removes R-peaks!
    250.0, 4
)
```

**Common Ranges:**
| Signal | Keep This Range |
|--------|-----------------|
| ECG | 0.5 - 40 Hz |
| EEG | 0.5 - 100 Hz |
| EMG | 20 - 500 Hz |
| PPG | 0.5 - 10 Hz |

✓ **Lower filter order if too aggressive**
```python
# High order = steep cutoff (may ring)
steep = IirFilter.bandpass(0.5, 40.0, 250.0, order=8)  # Very steep

# Lower order = gentler (try this first)
gentle = IirFilter.bandpass(0.5, 40.0, 250.0, order=4)  # ✓ Start here
```

---

### 🔴 Problem: "Filter introduces weird oscillations/ringing"

**What you see:**
```
         Ringing
           ~~~
         /     \
        /       \
_______/         \_______
```

**Fixes:**

✓ **Use forward-backward filtering**
```python
# Many filters can introduce phase shift
# Some implementations handle this automatically

# Or apply filter in both directions
import scipy.signal
filtered = scipy.signal.filtfilt(b, a, signal)  # Zero-phase
```

✓ **Reduce filter order**
```python
# Lower order = less ringing
filter = IirFilter.bandpass(0.5, 40.0, 250.0, order=2)  # Gentler
```

---

## 4. Code Errors

### 🔴 Problem: "Code won't run - Import errors"

**Error message:**
```
ModuleNotFoundError: No module named 'dpb'
```

**Fixes:**

✓ **Check installation**
```bash
# Install DPB
pip install dpb

# Or if using conda
conda install dpb

# Verify installation
python -c "import dpb; print(dpb.__version__)"
```

✓ **Check Python environment**
```bash
# Are you in the right environment?
which python
python --version

# Activate environment if needed
conda activate myenv
# or
source venv/bin/activate
```

---

### 🔴 Problem: "Shape mismatch errors"

**Error message:**
```
ValueError: shapes (1000,) and (500,) not aligned
```

**Fixes:**

✓ **Check signal dimensions**
```python
# Print shapes to debug
print(f"Signal shape: {signal.shape}")
print(f"Expected: 1D array, got {signal.ndim}D")

# Fix 2D to 1D
if signal.ndim == 2:
    signal = signal[:, 0]  # Take first channel
    # or
    signal = signal.flatten()  # Flatten to 1D
```

✓ **Check lengths match**
```python
# When working with multiple signals
print(f"Signal 1: {len(signal1)} samples")
print(f"Signal 2: {len(signal2)} samples")

# Trim to same length
min_len = min(len(signal1), len(signal2))
signal1 = signal1[:min_len]
signal2 = signal2[:min_len]
```

---

### 🔴 Problem: "Index out of bounds"

**Error message:**
```
IndexError: index 1500 is out of bounds for axis 0 with size 1000
```

**Fixes:**

✓ **Check peak indices are valid**
```python
# Make sure peaks are within signal
peaks = peaks[peaks < len(signal)]
print(f"Valid peaks: {len(peaks)}")
```

✓ **Check array slicing**
```python
# Print indices before accessing
print(f"Signal length: {len(signal)}")
print(f"Trying to access index: {index}")

# Safe indexing
if index < len(signal):
    value = signal[index]
else:
    print(f"Index {index} out of range!")
```

---

## 5. Results Don't Make Sense

### 🔴 Problem: "Results don't match expected values"

**What you see:**
```
Expected SDNN: ~50 ms
Your result: SDNN: 500,000 ms
```

**Fixes:**

✓ **Check units**
```python
# Are your R-peaks in samples or seconds?
# HRV functions usually expect samples

# If you have times in seconds, convert:
peaks_samples = peaks_seconds * sample_rate

# Check units of result
print(f"SDNN: {sdnn:.1f} ms")  # Should be milliseconds
print(f"Is this reasonable? (expect 20-100 ms)")
```

✓ **Verify input data quality**
```python
# Check R-peaks look reasonable
rr_intervals = np.diff(peaks) / sample_rate * 1000  # ms
print(f"RR interval range: {rr_intervals.min():.0f} - {rr_intervals.max():.0f} ms")
print(f"Expected: 600-1000 ms (60-100 bpm)")

# Remove outliers if needed
valid = (rr_intervals > 300) & (rr_intervals < 2000)
rr_clean = rr_intervals[valid]
```

✓ **Need more data**
```python
# Many metrics need minimum duration
duration = len(signal) / sample_rate

# Requirements:
# - Heart rate: 10+ seconds
# - HRV time-domain: 60+ seconds
# - HRV frequency-domain: 180+ seconds (3 min)

if duration < 60:
    print(f"Warning: Only {duration:.0f}s of data")
    print("HRV results may be unreliable")
```

---

## 6. Performance Problems

### 🔴 Problem: "Processing is too slow"

**What you see:**
```
Processing 1 hour of data... (waiting forever)
```

**Fixes:**

✓ **Process in chunks**
```python
# Don't load entire file at once
chunk_size = int(30 * sample_rate)  # 30 second chunks

results = []
for i in range(0, len(signal), chunk_size):
    chunk = signal[i:i+chunk_size]
    result = process(chunk)
    results.append(result)
```

✓ **Use more efficient filters**
```python
# IIR filters are faster than FIR for same performance
iir_filter = dpb.signal.IirFilter.bandpass(0.5, 40.0, 250.0, 4)  # Fast

# Reduce filter order if possible
faster_filter = dpb.signal.IirFilter.bandpass(0.5, 40.0, 250.0, 2)  # Faster
```

✓ **Enable GPU acceleration (if available)**
```python
# Check if GPU available
if dpb.gpu.is_available():
    ctx = dpb.gpu.GpuContext(device_id=0)
    ctx.initialize()
    # Processing will use GPU
```

---

## 🎯 Quick Diagnostic Flowchart

```
Problem?
   │
   ├─→ Signal flat/weird
   │   ├─→ Check connections
   │   ├─→ Check scaling/units
   │   └─→ Plot raw signal
   │
   ├─→ Too noisy
   │   ├─→ Apply bandpass filter
   │   ├─→ Add notch filter (50/60 Hz)
   │   └─→ Check motion artifacts
   │
   ├─→ Can't find peaks
   │   ├─→ Filter signal first!
   │   ├─→ Adjust threshold
   │   └─→ Visual debugging (plot)
   │
   ├─→ Wrong results
   │   ├─→ Verify sample rate
   │   ├─→ Check units
   │   └─→ Need more data?
   │
   └─→ Code error
       ├─→ Check installation
       ├─→ Check dimensions
       └─→ Print debug info
```

---

## 🔍 Debugging Template

**Use this when stuck:**

```python
import dpb
import numpy as np
import matplotlib.pyplot as plt

# 1. Check signal basics
print("=== Signal Info ===")
print(f"Length: {len(signal)} samples")
print(f"Duration: {len(signal)/sample_rate:.1f} seconds")
print(f"Range: {signal.min():.4f} to {signal.max():.4f}")
print(f"Mean: {signal.mean():.4f}")
print(f"Std: {signal.std():.4f}")
print(f"Sample rate: {sample_rate} Hz")

# 2. Plot raw signal
plt.figure(figsize=(12, 4))
time = np.arange(len(signal)) / sample_rate
plt.plot(time, signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Raw Signal')
plt.grid(True)
plt.show()

# 3. Plot frequency spectrum
from scipy.fft import fft, fftfreq
fft_vals = np.abs(fft(signal))
freqs = fftfreq(len(signal), 1/sample_rate)
plt.figure(figsize=(10, 4))
plt.plot(freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.xlim(0, 50)
plt.grid(True)
plt.show()

# 4. Try filtering
filt = dpb.signal.IirFilter.bandpass(0.5, 40.0, sample_rate, 4)
filtered = filt.apply(signal)

# 5. Plot filtered
plt.figure(figsize=(12, 4))
plt.plot(time, filtered)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Filtered Signal')
plt.grid(True)
plt.show()

# Did this help? What changed?
```

---

## 💡 Top 10 Debugging Tips

1. **Always plot your signal first** - Don't process blind!
2. **Filter before detecting** - Raw signals are too noisy
3. **Check sample rate** - Most common cause of wrong results
4. **Start simple** - Load → Filter → Plot before complex analysis
5. **Print everything** - Signal shape, range, length, units
6. **Use realistic test data** - Try synthetic data first
7. **Visual debugging** - Plot intermediate steps
8. **Check edge cases** - Short signals, missing data, outliers
9. **Verify units** - Seconds vs samples, V vs mV vs µV
10. **Read error messages carefully** - They usually tell you the problem!

---

**Remember:** Most problems are either:
1. Signal not filtered
2. Wrong sample rate
3. Wrong units/scaling
4. Not enough data

Check these four things first!
