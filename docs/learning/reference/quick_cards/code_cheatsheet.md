# Quick Reference Card 4: Essential Code 💻

> **Most-used functions** - Copy-paste ready code snippets that actually work

---

## 1. Loading Data 📁

### Load ECG from PhysioNet (WFDB)
```python
import dpb

# Auto-detect format
reader = dpb.io.UnifiedReader("path/to/record")
signal = reader.read_signal(0)  # First channel
sample_rate = reader.sample_rate()
print(f"Loaded {len(signal)} samples at {sample_rate} Hz")
```

### Generate Synthetic Data (For Testing)
```python
import dpb

# Create ECG generator
gen = dpb.synth.EcgGenerator(
    heart_rate=70,      # bpm
    hrv_sdnn=50         # HRV variability
)

# Generate 30 seconds at 250 Hz
signal, ground_truth = gen.generate(
    duration=30.0,
    sample_rate=250.0,
    seed=42
)
print(f"Generated signal: {signal.shape}")
print(f"True R-peaks at: {ground_truth['r_peaks']}")
```

### Load from CSV
```python
import numpy as np

# Simple CSV with one column
signal = np.loadtxt("data.csv")
sample_rate = 250.0  # You need to know this!
```

---

## 2. Filtering 🎚️

### Bandpass Filter (Remove Noise Above & Below)
```python
import dpb

# Create filter (0.5-40 Hz for ECG)
filter = dpb.signal.IirFilter.bandpass(
    low_freq=0.5,      # Hz
    high_freq=40.0,    # Hz
    sample_rate=250.0, # Hz
    order=4            # Filter strength
)

# Apply to signal
filtered = filter.apply(signal)
print(f"Filtered signal: {filtered.shape}")
```

### Notch Filter (Remove 50/60 Hz Power Line Noise)
```python
# Remove 60 Hz interference
notch = dpb.signal.IirFilter.notch(
    freq=60.0,         # Frequency to remove
    sample_rate=250.0,
    quality=30.0       # Narrowness of notch
)
clean_signal = notch.apply(signal)
```

### Quick Filter Presets
```python
# ECG
ecg_filter = dpb.signal.IirFilter.bandpass(0.5, 40.0, 250.0, 4)

# EEG
eeg_filter = dpb.signal.IirFilter.bandpass(0.5, 100.0, 500.0, 4)

# EMG
emg_filter = dpb.signal.IirFilter.bandpass(20.0, 500.0, 1000.0, 4)

# PPG
ppg_filter = dpb.signal.IirFilter.bandpass(0.5, 10.0, 100.0, 4)
```

---

## 3. Finding Peaks 🎯

### Find R-Peaks in ECG (Pan-Tompkins Algorithm)
```python
# Create detector
detector = dpb.signal.ecg.PanTompkinsDetector(
    sample_rate=250.0
)

# Detect R-peaks
peaks = detector.detect(filtered_signal)
print(f"Found {len(peaks)} heartbeats")
print(f"Peak locations (samples): {peaks}")

# Calculate heart rate
rr_intervals = np.diff(peaks) / sample_rate  # seconds
heart_rate = 60.0 / np.mean(rr_intervals)
print(f"Heart Rate: {heart_rate:.1f} bpm")
```

### Generic Peak Detection (Any Signal)
```python
# Find peaks with minimum height and distance
peaks = dpb.signal.find_peaks(
    signal,
    height=0.5,          # Minimum peak height
    distance=200,        # Minimum samples between peaks
    prominence=0.3       # How much peak stands out
)
print(f"Found {len(peaks)} peaks")
```

### Find Valleys (Opposite of Peaks)
```python
valleys = dpb.signal.find_valleys(
    signal,
    height=-0.5,         # Maximum valley depth
    distance=200
)
```

---

## 4. Computing Metrics 📐

### Heart Rate Variability (HRV)
```python
# Create analyzer
hrv = dpb.signal.hrv.HrvAnalyzer()

# Time-domain metrics
time_metrics = hrv.compute_time_domain(
    r_peaks,            # Peak locations (samples)
    sample_rate=250.0
)
print(f"Mean HR: {time_metrics.mean_hr:.1f} bpm")
print(f"SDNN: {time_metrics.sdnn:.1f} ms")
print(f"RMSSD: {time_metrics.rmssd:.1f} ms")
print(f"pNN50: {time_metrics.pnn50:.1f}%")

# Frequency-domain metrics
freq_metrics = hrv.compute_frequency_domain(
    r_peaks,
    sample_rate=250.0
)
print(f"LF Power: {freq_metrics.lf_power:.1f}")
print(f"HF Power: {freq_metrics.hf_power:.1f}")
print(f"LF/HF Ratio: {freq_metrics.lf_hf_ratio:.2f}")
```

### EEG Band Powers (Alpha, Beta, etc.)
```python
# Compute power in each frequency band
band_powers = dpb.signal.eeg.compute_band_powers(
    eeg_signal,
    sample_rate=500.0
)
print(f"Delta (0.5-4 Hz): {band_powers.delta:.2f}")
print(f"Theta (4-8 Hz): {band_powers.theta:.2f}")
print(f"Alpha (8-13 Hz): {band_powers.alpha:.2f}")
print(f"Beta (13-30 Hz): {band_powers.beta:.2f}")
print(f"Gamma (30+ Hz): {band_powers.gamma:.2f}")

# Alpha/Beta ratio (relaxation indicator)
ratio = band_powers.alpha / band_powers.beta
print(f"Alpha/Beta Ratio: {ratio:.2f}")
```

### PPG Analysis (Pulse, SpO2)
```python
# Analyze PPG signal
ppg_analyzer = dpb.signal.ppg.PpgAnalyzer(
    sample_rate=100.0
)
features = ppg_analyzer.analyze(ppg_signal)

print(f"Heart Rate: {features.heart_rate:.1f} bpm")
print(f"Pulse amplitude: {features.pulse_amplitude:.3f}")
print(f"SpO2: {features.spo2:.1f}%")
```

### EMG Fatigue Detection
```python
# Analyze muscle fatigue
emg_analyzer = dpb.signal.emg.EmgAnalyzer(
    sample_rate=1000.0
)
fatigue = emg_analyzer.compute_fatigue(emg_signal)

print(f"Median Frequency: {fatigue.median_freq:.1f} Hz")
print(f"Frequency slope: {fatigue.freq_slope:.2f}")  # Negative = fatigue
print(f"RMS amplitude: {fatigue.rms:.3f}")
print(f"Fatigue index: {fatigue.fatigue_index:.2f}")  # 0-1 scale
```

### Basic Signal Statistics
```python
import numpy as np

# Common measurements
mean_val = np.mean(signal)
std_val = np.std(signal)
max_val = np.max(signal)
min_val = np.min(signal)
rms = np.sqrt(np.mean(signal**2))

print(f"Mean: {mean_val:.3f}")
print(f"Std Dev: {std_val:.3f}")
print(f"Range: {min_val:.3f} to {max_val:.3f}")
print(f"RMS: {rms:.3f}")
```

---

## 5. Visualization 📊

### Plot Raw Signal
```python
import matplotlib.pyplot as plt
import numpy as np

time = np.arange(len(signal)) / sample_rate

plt.figure(figsize=(12, 4))
plt.plot(time, signal)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Raw Signal')
plt.grid(True)
plt.show()
```

### Plot with Detected Peaks
```python
plt.figure(figsize=(12, 4))
plt.plot(time, signal, label='Signal')
plt.plot(peaks / sample_rate, signal[peaks], 'ro',
         label='R-peaks', markersize=8)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('ECG with Detected R-peaks')
plt.legend()
plt.grid(True)
plt.show()
```

### Plot Frequency Spectrum
```python
from scipy.fft import fft, fftfreq

# Compute FFT
fft_vals = np.abs(fft(signal))
freqs = fftfreq(len(signal), 1/sample_rate)

# Plot positive frequencies only
plt.figure(figsize=(10, 4))
plt.plot(freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.xlim(0, 50)  # Focus on 0-50 Hz
plt.grid(True)
plt.show()
```

### Plot Before/After Filtering
```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# Before
ax1.plot(time, raw_signal)
ax1.set_title('Before Filtering')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

# After
ax2.plot(time, filtered_signal)
ax2.set_title('After Filtering')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Amplitude')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

## 6. Exporting Results 💾

### Save Processed Signal
```python
# Save as numpy array
np.save('filtered_ecg.npy', filtered_signal)

# Save as CSV
np.savetxt('filtered_ecg.csv', filtered_signal)

# Load back
loaded = np.load('filtered_ecg.npy')
```

### Save Metrics to CSV
```python
import pandas as pd

# Create results dictionary
results = {
    'heart_rate': [heart_rate],
    'sdnn': [time_metrics.sdnn],
    'rmssd': [time_metrics.rmssd],
    'lf_hf_ratio': [freq_metrics.lf_hf_ratio]
}

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('hrv_results.csv', index=False)
print("Results saved!")
```

### Save Peak Locations
```python
# Save R-peak indices
np.savetxt('r_peaks.csv', peaks, fmt='%d')

# Or with timestamps
peak_times = peaks / sample_rate
np.savetxt('r_peak_times.csv', peak_times, fmt='%.3f')
```

---

## 🎯 Complete Example Workflows

### 1. Quick ECG Analysis
```python
import dpb
import numpy as np

# Load data
gen = dpb.synth.EcgGenerator(heart_rate=70, hrv_sdnn=50)
signal, gt = gen.generate(duration=30.0, sample_rate=250.0)

# Filter
filt = dpb.signal.IirFilter.bandpass(0.5, 40.0, 250.0, 4)
clean = filt.apply(signal)

# Detect R-peaks
detector = dpb.signal.ecg.PanTompkinsDetector(250.0)
peaks = detector.detect(clean)

# Calculate HRV
hrv = dpb.signal.hrv.HrvAnalyzer()
metrics = hrv.compute_time_domain(peaks, 250.0)

print(f"Heart Rate: {metrics.mean_hr:.1f} bpm")
print(f"SDNN: {metrics.sdnn:.1f} ms")
```

### 2. EEG Band Power Analysis
```python
import dpb

# Generate EEG data
gen = dpb.synth.EegGenerator(dominant_freq=10.0)  # 10 Hz alpha
signal, gt = gen.generate(duration=60.0, sample_rate=500.0)

# Filter
filt = dpb.signal.IirFilter.bandpass(0.5, 100.0, 500.0, 4)
clean = filt.apply(signal)

# Compute band powers
bands = dpb.signal.eeg.compute_band_powers(clean, 500.0)

print(f"Alpha power: {bands.alpha:.2f}")
print(f"Beta power: {bands.beta:.2f}")
print(f"Relaxation ratio: {bands.alpha/bands.beta:.2f}")
```

### 3. Real-Time Processing Template
```python
import dpb

# Setup pipeline
config = dpb.pipeline.PipelineConfig(
    window_size=256,
    hop_size=128,
    sample_rate=250.0
)
executor = dpb.pipeline.PipelineExecutor(config)

# Processing loop
while True:
    sample = get_next_sample()  # Your data source

    result = executor.process_sample(sample,
        lambda window: analyze_window(window))

    if result:
        print(f"Result: {result}")
```

---

## 💡 Pro Tips

### Filtering:
```python
# Always filter before detecting!
signal = load_data()
signal = filter.apply(signal)  # ← Don't skip this
peaks = detect_peaks(signal)
```

### Error Handling:
```python
try:
    peaks = detector.detect(signal)
except Exception as e:
    print(f"Detection failed: {e}")
    # Try adjusting threshold or check signal quality
```

### Check Signal Quality:
```python
# Before processing, check:
print(f"Length: {len(signal)} samples")
print(f"Duration: {len(signal)/sample_rate:.1f} seconds")
print(f"Range: {signal.min():.3f} to {signal.max():.3f}")
print(f"Mean: {signal.mean():.3f}")

# Should be reasonable values!
```

### Memory Management:
```python
# For long recordings, process in chunks
chunk_size = int(30 * sample_rate)  # 30 seconds
for i in range(0, len(signal), chunk_size):
    chunk = signal[i:i+chunk_size]
    process_chunk(chunk)
```

---

## ⚡ Common Parameter Values

| Signal | Sample Rate | Bandpass Filter | Peak Distance |
|--------|-------------|-----------------|---------------|
| ECG | 250-500 Hz | 0.5-40 Hz | ~150 samples (250Hz) |
| EEG | 250-500 Hz | 0.5-100 Hz | Varies |
| EMG | 1000-2000 Hz | 20-500 Hz | ~100 samples |
| PPG | 50-100 Hz | 0.5-10 Hz | ~80 samples (100Hz) |
| EDA | 10-50 Hz | 0-5 Hz | ~50 samples |

---

**Remember:** Start simple, then add complexity!
1. Load → Filter → Detect → Plot (verify it works!)
2. Then add metrics and analysis
