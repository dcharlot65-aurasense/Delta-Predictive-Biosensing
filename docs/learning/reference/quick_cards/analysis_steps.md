# Quick Reference Card 3: From Raw to Results 🔄

> **Signal processing flowchart** - Step-by-step from messy data to clean insights

---

## The Big Picture Pipeline

```
┌──────────┐   ┌───────┐   ┌────────┐   ┌────────┐   ┌─────────┐   ┌─────────┐   ┌────────┐
│  Raw     │──→│ Load  │──→│ Clean  │──→│ Filter │──→│ Detect  │──→│ Measure │──→│ Report │
│ Signal   │   │ Data  │   │ Signal │   │ Noise  │   │ Events  │   │ Metrics │   │ Results│
└──────────┘   └───────┘   └────────┘   └────────┘   └─────────┘   └─────────┘   └────────┘
    📊            📁          🧹           🎚️           🎯            📐            📋
 Noisy data    Import      Remove DC    Bandpass    Find peaks    Calculate     Export
                            offset       filter      R-peaks       HRV, etc.
```

---

## Step 1: Load Data 📁

**What:** Import signal from file or device

**Common Formats:**
- WFDB (PhysioNet)
- EDF/EDF+ (European Data Format)
- CSV/text files
- Streaming from device

**Tools in DPB:**
```
UnifiedReader       - Auto-detect format
WfdbReader          - PhysioNet data
EdfReader           - EDF files
```

**When to use:** Always the first step!

**Example Output:**
```
Signal: [0.2, 0.3, 0.5, 0.8, ...]
Sample Rate: 250 Hz
Duration: 60 seconds
```

---

## Step 2: Clean Signal 🧹

**What:** Remove obvious problems

### 2a. Check for Issues
- Missing data (gaps)
- Flat sections
- Extreme outliers
- Incorrect units

### 2b. Remove DC Offset
```
Before:           After:
    /\  /\            /\  /\
___/  \/  \       ___/  \/  \___
Baseline drifting → Centered at 0
```

**Tools in DPB:**
```
remove_dc_offset()  - Center signal at zero
```

**When to use:** If signal has a "drift" or isn't centered

---

## Step 3: Filter Noise 🎚️

**What:** Remove unwanted frequencies

### Filter Types:

#### Bandpass Filter (Most Common)
```
Keep this range ↓
    ┌────────┐
────┘        └────
0.5Hz      40Hz
(ECG example)
```
Keeps useful frequencies, removes noise above and below

#### Notch Filter
```
Remove 50/60 Hz
────┐  ┌────
    └──┘
Power line noise
```

#### Lowpass Filter
```
Keep low frequencies
────┐
    └──────
    ↑ Cutoff
```
Removes high-frequency noise

**Tools in DPB:**
```
IirFilter::bandpass()   - Keep frequency range
IirFilter::notch()      - Remove specific frequency
FirFilter               - Alternative filter type
```

**When to use:**
- **Always** for ECG, EEG, EMG
- After loading, before detecting features
- Match filter to your signal type!

**Common Settings:**

| Signal | Bandpass Range |
|--------|----------------|
| ECG | 0.5 - 40 Hz |
| EEG | 0.5 - 100 Hz |
| EMG | 20 - 500 Hz |
| PPG | 0.5 - 10 Hz |
| EDA | 0 - 5 Hz |

---

## Step 4: Detect Events 🎯

**What:** Find important features in your signal

### Common Detections:

#### ECG - Find R-Peaks
```
      R   R   R
     /\  /\  /\
    /  \/  \/  \
   ▲   ▲   ▲
   Found!
```

#### EEG - Find Artifacts
```
Eye Blink ↓
     ___
  __/   \__
  ▲
  Found!
```

#### EMG - Find Bursts
```
        ____
_______|    |______
       ↑    ↑
     Start End
```

**Tools in DPB:**
```
PanTompkinsDetector     - ECG R-peaks
find_peaks()            - Generic peak detection
EmgAnalyzer             - EMG bursts
detect_artifacts()      - EEG artifacts
```

**When to use:**
- After filtering
- When you need timing of events
- Before calculating metrics

---

## Step 5: Measure Metrics 📐

**What:** Calculate numbers that describe your signal

### Time-Domain Metrics
```
R   R   R   R
│←t→│←t→│←t→│
850ms 920ms 880ms

SDNN = Standard deviation of intervals
RMSSD = Root mean square of differences
```

### Frequency-Domain Metrics
```
Power Spectrum
    │  VLF   LF    HF
    │  ___  ___   __
────┴──────────────────
    0  0.04 0.15 0.4 Hz

LF/HF Ratio = Stress indicator
```

### Amplitude Metrics
```
    Peak
     /\
    /  \  ← Amplitude
___/    \___

Mean, Max, RMS
```

**Tools in DPB:**
```
HrvAnalyzer             - Heart rate variability
compute_band_powers()   - EEG frequency bands
FatigueMetrics          - Muscle fatigue
PpgAnalyzer            - Pulse wave features
```

**When to use:**
- After detecting events
- For quantitative analysis
- To compare conditions

---

## Step 6: Compare Results 📊

**What:** Put metrics in context

### Compare To:
1. **Normative Values** (What's typical?)
2. **Baseline** (Subject's own normal)
3. **Conditions** (Before vs. after task)

**Tools in DPB:**
```
NormativeDatabase       - Age/sex norms
Statistical comparisons
```

**When to use:** Final step for interpretation

---

## Step 7: Report/Export 📋

**What:** Save and share results

**Export Options:**
- Summary statistics (CSV)
- Plots/visualizations
- Processed signals
- Event markers

---

## 🎯 Decision Tree: Which Steps Do I Need?

```
Start
  │
  ├─→ Just need heart rate?
  │   └─→ Load → Filter → Detect R-peaks → Calculate HR ✓
  │
  ├─→ Analyzing stress/HRV?
  │   └─→ Load → Filter → Detect R-peaks → HRV Metrics → Compare ✓
  │
  ├─→ Sleep staging (EEG)?
  │   └─→ Load → Filter → Band Power → Classify Stages ✓
  │
  └─→ Muscle fatigue (EMG)?
      └─→ Load → Filter → Burst Detection → Fatigue Metrics ✓
```

---

## 📋 Pipeline Checklist

### Before You Start:
- [ ] What signal type? (ECG, EEG, EMG, PPG, EDA)
- [ ] What question am I answering?
- [ ] What metrics do I need?

### During Processing:
- [ ] Signal loaded correctly?
- [ ] Sample rate known?
- [ ] Filter settings match signal type?
- [ ] Features detected correctly?

### After Processing:
- [ ] Results make sense?
- [ ] Compare to expected values?
- [ ] Save processed data?
- [ ] Document parameters used?

---

## 💡 Pro Tips by Step

### Loading:
- **Always check sample rate!** Same signal, wrong rate = wrong results
- Note units (mV, µV, etc.)

### Cleaning:
- Plot raw signal first to see problems
- Don't over-clean (might remove real features)

### Filtering:
- **Too much filtering = loss of features**
- **Too little filtering = noise remains**
- When in doubt, use standard ranges (see table above)

### Detection:
- Adjust thresholds if too many/few detections
- Visually verify first few detections

### Measurement:
- Garbage in = garbage out (filter first!)
- Always report which metrics and how calculated

### Comparison:
- Need a baseline (don't compare to nothing)
- Account for age, sex, fitness level

### Reporting:
- Include processing steps used
- Report filter settings
- Show example waveforms

---

## ⚡ Common Pipeline Examples

### Fast Heart Rate Check:
```
Load → Bandpass(0.5-40Hz) → Detect R-peaks → Count beats
Time: ~1-2 seconds
```

### Full HRV Analysis:
```
Load → Bandpass → Detect R-peaks → Time Metrics →
  Freq Metrics → Compare to Norms → Report
Time: ~10-30 seconds
```

### Real-Time Monitoring:
```
Stream → Circular Buffer → Filter → Detect →
  Update Display (repeat)
Latency: <10ms per sample
```

---

## 🔧 Tools Quick Reference

| Step | Function | What It Does |
|------|----------|--------------|
| Load | `UnifiedReader` | Auto-detect & load any format |
| Clean | `remove_dc_offset()` | Center signal at zero |
| Filter | `IirFilter::bandpass()` | Remove noise outside range |
| Detect | `PanTompkinsDetector` | Find ECG R-peaks |
| Detect | `find_peaks()` | Find any peaks |
| Measure | `HrvAnalyzer` | Calculate HRV metrics |
| Measure | `compute_band_powers()` | EEG frequency analysis |
| Compare | `NormativeDatabase` | Check against norms |

---

**Remember:** You can skip steps, but don't skip filtering!
Almost all biosignals need filtering before analysis.
