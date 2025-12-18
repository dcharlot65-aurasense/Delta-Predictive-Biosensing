# Chapter 3: Finding the Heartbeats

## The Peak Detection Challenge

Look at an ECG on a monitor. You can instantly spot the heartbeats - those sharp spikes marching across the screen. Count them for six seconds, multiply by ten, and you've got heart rate. Easy, right?

Now try to explain to a computer how to do the same thing. "Find the sharp spikes" - but how sharp? "Look for the peaks" - but there are lots of little peaks, how do you know which ones are heartbeats? "They repeat regularly" - but what if someone has an irregular heartbeat?

What seems obvious to your brain is actually a complex pattern recognition task. This chapter teaches you how algorithms automatically detect peaks in biosignals, handle tricky cases, and avoid false alarms. We'll focus on ECG R-peak detection (finding heartbeats), but the principles apply to any peak-finding task.

---

## What is a Peak?

In the simplest terms, a **peak** is a point higher than the points around it. The value goes up, reaches a maximum, then goes down.

**Formal definition**: A peak at time t occurs when:
- Signal(t) > Signal(t-1) [higher than previous point]
- Signal(t) > Signal(t+1) [higher than next point]

That's the basic idea, but real peak detection needs to handle complications:
- Noise creates tiny false peaks everywhere
- Peaks have different heights
- Peaks have different widths
- Real peaks sometimes have small bumps on them
- Sometimes peaks are missing
- Sometimes noise looks like peaks

---

## The ECG R-Peak

The ECG waveform has several peaks and valleys:
- **P wave**: Atrial depolarization (small bump)
- **Q wave**: Start of ventricular depolarization (small dip)
- **R wave**: Main ventricular depolarization (BIG spike)
- **S wave**: End of ventricular depolarization (small dip)
- **T wave**: Ventricular repolarization (medium bump)

The **R-peak** is usually the tallest, sharpest feature - the most obvious marker of a heartbeat. Finding R-peaks = finding heartbeats.

**Why R-peaks matter**:
- Each R-peak = one heartbeat
- Count R-peaks = heart rate
- Time between R-peaks = beat-to-beat intervals
- Regular/irregular R-peaks = rhythm analysis
- R-peak shape = cardiac health indicators

---

## Simple Threshold Detection

The simplest approach: set a threshold. Any point above the threshold is a peak.

**Algorithm**:
1. Choose a threshold value (e.g., 0.5 mV for ECG)
2. Check every point
3. If signal > threshold, it's a peak
4. Count the peaks

**Pros**:
- Simple to understand
- Easy to implement
- Fast

**Cons**:
- How do you choose the threshold?
- What if signal amplitude varies?
- Noise above threshold creates false peaks
- Real peaks below threshold get missed

**Example**:
- Threshold = 0.5 mV
- Strong R-peaks at 1.0 mV: ✓ Detected
- Weak R-peaks at 0.4 mV: ✗ Missed
- Noise spike at 0.6 mV: ✗ False detection

Simple thresholding only works in perfect conditions - clean signal, consistent amplitude, no noise. Real-world signals need smarter approaches.

---

## Adaptive Thresholding

The problem with fixed thresholds: signals change. Someone moves, electrode contact changes, breathing varies - suddenly yesterday's perfect threshold doesn't work anymore.

**Solution**: Make the threshold adapt to the signal.

**Basic Adaptive Algorithm**:
1. Estimate recent signal level (average or median of last N seconds)
2. Set threshold as percentage above signal level (e.g., 60% above)
3. Detect peaks above adaptive threshold
4. Update signal level estimate continuously

**Example**:
- Average signal level: 0.2 mV
- Adaptive threshold: 0.2 + 60% = 0.32 mV
- Signal level increases to 0.4 mV
- Adaptive threshold updates to: 0.4 + 60% = 0.64 mV

The threshold follows the signal level, maintaining constant sensitivity.

**Dual Thresholds**:
Better algorithms use two thresholds:
- **High threshold**: Definitely a peak (very confident)
- **Low threshold**: Maybe a peak (less confident)

Confirmed peaks teach the algorithm what peaks look like, improving detection of unclear cases.

---

## The Pan-Tompkins Algorithm

Developed in 1985 by Jiapu Pan and Willis Tompkins, this algorithm has become the gold standard for ECG R-peak detection. It's used in countless medical devices and research studies.

Let's break it down into simple steps:

### Step 1: Band-Pass Filter (5-15 Hz)
**Purpose**: Remove baseline wander and high-frequency noise.
**Why these frequencies**: R-peaks have most energy in 5-15 Hz range.
**Result**: Cleaner signal, emphasis on R-peak frequencies.

### Step 2: Differentiation
**Purpose**: Emphasize rapid changes (like the R-peak's sharp rise).
**How it works**: Calculate slope (how fast signal is changing).
**Result**: Sharp R-peak slopes become large positive and negative values.

**Analogy**: Imagine you're in a car. Differentiation is like measuring acceleration, not speed. Sudden changes (like an R-peak) create high acceleration.

### Step 3: Squaring
**Purpose**:
- Make all values positive (no more negative values from differentiation)
- Amplify large values more than small values
- Emphasize R-peaks even more

**Math**: Signal² = Signal × Signal
**Result**: Large peaks become HUGE, small noise stays small.

### Step 4: Moving Window Integration
**Purpose**: Smooth out the squared signal, find regions of high energy.
**How it works**: Average over short window (typically 150 ms).
**Result**: Each R-peak becomes a smooth bump of high energy.

**Analogy**: Like looking at a cityscape through fog. Sharp building edges blur into general shapes, but you can still see where the tall buildings are.

### Step 5: Adaptive Threshold Detection
**Purpose**: Find the integrated signal peaks that represent R-peaks.

**Two thresholds**:
- **Threshold1** (high): For clear R-peaks
- **Threshold2** (low): For uncertain cases

**Two running estimates**:
- **SPKI** (signal peak level): Average of recent R-peaks
- **NPKI** (noise peak level): Average of recent noise

**Threshold formulas**:
- Threshold1 = NPKI + 0.25 × (SPKI - NPKI)
- Threshold2 = 0.5 × Threshold1

If a peak exceeds Threshold1, it's definitely an R-peak. If it's between Threshold1 and Threshold2, apply additional checks.

### Step 6: Decision Rules
**Additional checks**:
1. **Timing**: Is it at least 200 ms since the last R-peak? (Prevents detecting same beat twice)
2. **Searchback**: If too much time has passed (like 1.5× average RR interval), search back for a missed beat
3. **T-wave rejection**: Is the peak too soon after the last R-peak? Might be a T-wave, not an R-peak

**Analogy**: Like a bouncer at a club checking multiple IDs. Height check (threshold), timing check (not too soon after last entry), plausibility check (is this really a person or something else?).

---

## Handling Tricky Cases

### Case 1: Irregular Heartbeats
**Problem**: Beat timing is unpredictable.
**Solution**: Wider searchback window, lower minimum interval threshold.
**Trade-off**: Higher risk of false detections.

### Case 2: Low Amplitude Beats
**Problem**: Some R-peaks are much shorter than others.
**Solution**: Adaptive threshold tracks both high and low peaks.
**Example**: Premature ventricular contractions (PVCs) are often lower amplitude.

### Case 3: Baseline Wander
**Problem**: Entire signal drifts up and down.
**Solution**: High-pass filter (already in Pan-Tompkins) removes slow drift.

### Case 4: Muscle Noise
**Problem**: High-frequency noise creates false peaks.
**Solution**:
- Low-pass filter removes high frequencies
- Moving window integration smooths out individual noise spikes
- Threshold adaptation learns noise level

### Case 5: T-wave Confusion
**Problem**: T-waves can be tall and look like R-peaks.
**Solution**: Timing check - R-peaks can't occur sooner than 200 ms after previous R-peak.
**Advanced**: Check waveform shape, not just height.

### Case 6: Missing Beats
**Problem**: Electrode contact loss or very weak beat.
**Solution**: Searchback mechanism - if expected beat doesn't appear, search back through recent signal with lower threshold.

---

## Peak Detection Beyond ECG

The principles extend to other signals:

### PPG (Photoplethysmography) Peaks
**Signal**: Blood volume pulses in fingertip or wrist
**Peaks**: Each heartbeat creates a pulse wave
**Challenges**:
- Motion artifacts are huge
- Signal inverts sometimes (peak becomes valley)
- Slower rise time than ECG

**Adaptations**:
- More aggressive filtering (0.5-4 Hz band-pass)
- Check for both peaks and valleys
- Stricter motion artifact rejection

### EEG Spike Detection
**Signal**: Brain electrical activity
**Peaks**: Epileptic spikes or sleep spindles
**Challenges**:
- Much smaller amplitude
- More variable shapes
- More noise

**Adaptations**:
- Shape-based detection (not just height)
- Frequency content analysis
- Statistical outlier detection

### Respiratory Peak Detection
**Signal**: Breathing waveform
**Peaks**: Each breath creates peak (inhale) and valley (exhale)
**Challenges**:
- Very slow (0.2-0.5 Hz)
- Highly variable amplitude
- Often irregular

**Adaptations**:
- Very low frequency filtering
- Detect both peaks and valleys
- Longer adaptation windows

---

## Evaluating Peak Detection Performance

How do you know if your peak detector is working well?

### Metrics

**Sensitivity** (Recall):
- What percentage of real peaks did you find?
- Sensitivity = True Positives / (True Positives + False Negatives)
- High sensitivity means few missed beats

**Positive Predictive Value** (Precision):
- What percentage of detected peaks were real?
- PPV = True Positives / (True Positives + False Positives)
- High PPV means few false alarms

**F1 Score**:
- Combined metric balancing sensitivity and PPV
- F1 = 2 × (Sensitivity × PPV) / (Sensitivity + PPV)
- Higher is better (maximum = 1.0)

**Example**:
- 100 real R-peaks in signal
- Algorithm detects 98 real peaks (2 missed)
- Algorithm detects 3 false peaks

Metrics:
- Sensitivity = 98/100 = 98%
- PPV = 98/(98+3) = 97%
- F1 = 2 × (0.98 × 0.97)/(0.98 + 0.97) = 97.5%

For medical applications, you typically want >99% sensitivity and >99% PPV.

---

## Practical Implementation Tips

### Tip 1: Start with Good Filtering
Poor filtering = poor peak detection. Clean your signal first (Chapters 1 and 2).

### Tip 2: Visualize Results
Always plot detected peaks on the original signal. Visual inspection catches problems algorithms miss.

### Tip 3: Tune Parameters on Real Data
Default parameters often don't work perfectly. Test on your specific signals and adjust.

### Tip 4: Handle Edge Cases
Beginning and end of recordings need special handling - not enough data for moving averages.

### Tip 5: Document False Detections
When the algorithm fails, figure out why. Failure modes teach you how to improve.

### Tip 6: Consider Context
Heart rate normally between 40-200 bpm. If you're detecting 300 bpm, something's wrong.

### Tip 7: Use Redundancy
Multiple methods agreeing is more reliable than one method alone. Combine different approaches.

---

## Common Mistakes

### Mistake 1: No Validation
**Problem**: Assuming detection worked without checking.
**Solution**: Always validate on annotated data with known peaks.

### Mistake 2: Over-Tuning
**Problem**: Adjusting parameters to work perfectly on one recording, fails on others.
**Solution**: Test on diverse data, not just one example.

### Mistake 3: Ignoring Noise
**Problem**: Running peak detection on unfiltered signals.
**Solution**: Filter first, then detect peaks.

### Mistake 4: Fixed Parameters
**Problem**: Same parameters for all recordings regardless of signal quality.
**Solution**: Adaptive parameters based on signal characteristics.

### Mistake 5: No Refractory Period
**Problem**: Detecting same peak multiple times.
**Solution**: Enforce minimum time between peaks (refractory period).

---

## Advanced Topics

### Machine Learning Approaches
Modern methods use neural networks trained on thousands of ECGs. They learn complex patterns humans can't easily describe.

**Pros**:
- Handle unusual cases better
- Adapt to different signal types
- Often more accurate

**Cons**:
- Need training data
- Black box (hard to understand why it works)
- Computationally expensive

### Multi-Lead Analysis
Using multiple ECG leads simultaneously increases accuracy - if one lead is noisy, others might be clear.

### Confidence Scoring
Not all detected peaks are equally certain. Assigning confidence scores helps downstream analysis weight reliable detections more.

---

## Try It Out

Want to practice peak detection?

1. **Manual Detection**: Download an ECG from PhysioNet and manually mark R-peaks. Compare to automated algorithm. Where does it fail?

2. **Implement Simple Detector**: If you code, implement basic threshold detection. See how it fails, then add improvements one by one.

3. **Parameter Exploration**: Using existing software (Python, MATLAB), vary threshold, filter cutoffs, and window sizes. Observe effects.

4. **Challenge Cases**: Find recordings with artifacts, arrhythmias, or noise. These test algorithm robustness.

---

## Going Deeper

**Key Takeaways**:
- Peak detection finds signal maxima representing events (like heartbeats)
- Simple thresholding works only in perfect conditions
- Adaptive thresholding handles varying signal levels
- Pan-Tompkins algorithm is the gold standard for ECG R-peaks
- Good filtering is essential for good peak detection
- Validation against ground truth is mandatory

**Next Steps**:
Chapter 4 explores what you do with detected peaks - calculating heart rate, measuring intervals, and analyzing rhythm patterns. Once you can reliably find heartbeats, you can measure how the heart is behaving over time.

**Further Reading**:
- Pan & Tompkins (1985): "A Real-Time QRS Detection Algorithm"
- PhysioNet challenges on QRS detection
- Open-source implementations: PyBioSignal, BioSPPy, HeartPy

---

Finding peaks transforms continuous signals into discrete events you can count, time, and analyze. With reliable peak detection, you're ready to compute the metrics that matter - heart rate, rhythm, and variability.

**Next**: [Chapter 4: Measuring the Beat](ch04_rhythm.md)
