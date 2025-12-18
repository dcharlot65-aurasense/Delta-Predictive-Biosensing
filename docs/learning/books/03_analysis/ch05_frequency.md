# Chapter 5: The Hidden Frequencies

## The Prism of Signal Analysis

In 1666, Isaac Newton darkened his room, made a small hole in the curtain, and placed a glass prism in the beam of sunlight. What came out the other side amazed the world: the white light split into a rainbow of colors - red, orange, yellow, green, blue, violet.

White light isn't really white - it's all colors mixed together. The prism separates them so we can see each one.

Biosignals work the same way. What looks like a complex, messy waveform is actually many simple frequencies mixed together. The **Fourier transform** is like Newton's prism - it separates the signal into its frequency components so we can see what's really there.

This chapter teaches you to think in the frequency domain. You'll learn what frequencies mean, how the Fourier transform works (without heavy math), and why frequency analysis reveals patterns invisible in time-domain plots. By the end, you'll understand why this 200-year-old mathematical tool remains essential in modern biosignal analysis.

---

## What is Frequency?

**Frequency** measures how fast something repeats - how many cycles occur per second.

**Units**: Hertz (Hz) = cycles per second
- 1 Hz = one complete cycle every second
- 10 Hz = ten cycles every second
- 0.5 Hz = one cycle every two seconds

### Examples in Daily Life

**Low Frequency** (slow):
- Breathing: 0.2-0.3 Hz (12-18 breaths/minute)
- Ocean waves: 0.1 Hz (one wave every 10 seconds)

**Medium Frequency**:
- Heartbeat: 1 Hz (60 bpm)
- Ceiling fan: 5-10 Hz

**High Frequency** (fast):
- Fluorescent light flicker: 100-120 Hz
- Middle C musical note: 261.6 Hz
- High-pitched mosquito: 600 Hz

### Frequency in Biosignals

Every biosignal has characteristic frequencies:

**ECG**:
- Heart rate (R-R variation): 0.5-2 Hz
- QRS complex features: 10-30 Hz
- High-frequency noise: 50-60 Hz (powerline)

**EEG**:
- Delta waves (deep sleep): 0.5-4 Hz
- Theta waves (drowsy): 4-8 Hz
- Alpha waves (relaxed): 8-13 Hz
- Beta waves (alert): 13-30 Hz
- Gamma waves (focused): 30-100 Hz

**EMG**:
- Muscle activation: 20-200 Hz
- Peak power: 50-100 Hz

Understanding these frequency ranges helps you design filters (Chapter 2) and interpret analysis results.

---

## Period vs Frequency

**Period**: How long one complete cycle takes (in seconds)
**Frequency**: How many cycles happen per second (in Hz)

They're inverses of each other:
- Frequency = 1 / Period
- Period = 1 / Frequency

**Examples**:
- Heart rate 60 bpm = 1 Hz = 1 second period
- Heart rate 75 bpm = 1.25 Hz = 0.8 second period
- EEG alpha wave 10 Hz = 0.1 second period

**Analogy**: Imagine a Ferris wheel. The period is how long one complete rotation takes. The frequency is how many rotations happen per minute.

---

## The Fourier Transform: Breaking Signals Apart

Named after Jean-Baptiste Joseph Fourier (1822), this mathematical operation decomposes any signal into a sum of simple sine waves at different frequencies.

### The Core Idea

**Any signal can be built from sine waves.**

No matter how complex or irregular a signal looks, it can be perfectly reconstructed by adding together the right sine waves at the right frequencies, amplitudes, and phases.

**Simple Example**:
- Start with a 5 Hz sine wave
- Add a 10 Hz sine wave (half the amplitude)
- Add a 15 Hz sine wave (quarter the amplitude)
- Result: A complex waveform that looks nothing like a sine wave

The Fourier transform does the reverse: it starts with a complex waveform and figures out which sine waves you'd need to build it.

### Visual Description

**Time Domain** (what you normally see):
- Horizontal axis: Time (seconds)
- Vertical axis: Amplitude (voltage, pressure, etc.)
- Shows how signal changes over time

**Frequency Domain** (after Fourier transform):
- Horizontal axis: Frequency (Hz)
- Vertical axis: Power or amplitude at each frequency
- Shows which frequencies are present and how strong they are

**Analogy**: A musical chord played on a piano.
- **Time domain**: Recording of the sound over time (waveform)
- **Frequency domain**: List of which notes were played and how loud each was

### The Power Spectrum

The **power spectrum** (or **power spectral density**) shows how much power (energy) exists at each frequency.

**Reading a Power Spectrum**:
- Tall peak at 10 Hz: Strong 10 Hz component in signal
- Flat low values at 50 Hz: Weak or no 50 Hz component
- Elevated band from 8-13 Hz: Broad activity in that range

**Example**: ECG power spectrum might show:
- Large peak at 1.2 Hz: Heart rate (72 bpm)
- Smaller peak at 2.4 Hz: Second harmonic
- Spike at 60 Hz: Powerline interference
- Elevated region 10-40 Hz: QRS complex features

---

## The Prism Analogy in Detail

Let's extend Newton's prism analogy:

**White Light = Complex Signal**
- Looks simple (just "white")
- Actually contains all visible frequencies (colors)

**Prism = Fourier Transform**
- Separates the mixed frequencies
- Shows contribution of each frequency

**Rainbow = Power Spectrum**
- Each color's brightness = power at that frequency
- Red (low frequency) to violet (high frequency)
- Lets you see composition at a glance

**Recombining = Inverse Fourier Transform**
- Pass the rainbow through another prism
- Get white light back
- Original signal reconstructed perfectly

Just as Newton's prism revealed the true nature of light, the Fourier transform reveals the true frequency composition of signals.

---

## Frequency Bands in EEG

EEG analysis relies heavily on frequency decomposition. Different brain states show characteristic frequency patterns.

### Delta (0.5-4 Hz)
**When prominent**: Deep sleep, unconsciousness
**Meaning**: Low brain arousal
**Typical power**: Dominant during slow-wave sleep

### Theta (4-8 Hz)
**When prominent**: Drowsiness, light sleep, meditation
**Meaning**: Reduced alertness, creativity, memory consolidation
**Typical power**: Increases as you fall asleep

### Alpha (8-13 Hz)
**When prominent**: Relaxed wakefulness, eyes closed
**Meaning**: Calm, not actively processing information
**Typical power**: Strong in back of head, disappears when eyes open

**Fun fact**: Alpha waves were the first brain waves Hans Berger discovered in 1924.

### Beta (13-30 Hz)
**When prominent**: Active thinking, focus, anxiety
**Meaning**: Alert, engaged, sometimes stressed
**Typical power**: Front of brain during mental tasks

### Gamma (30-100 Hz)
**When prominent**: Intense focus, binding sensory information
**Meaning**: Consciousness, perception, attention
**Typical power**: Bursts during cognitive tasks

### Clinical Example

**Normal awake EEG**:
- Alpha: Moderate (relaxed but alert)
- Beta: Moderate (normal thinking)
- Theta: Low
- Delta: Very low

**Falling asleep**:
- Alpha: Decreasing
- Theta: Increasing
- Beta: Decreasing

**Deep sleep**:
- Delta: Dominant
- All others: Very low

**Alert and focused**:
- Beta: High
- Gamma: Bursts
- Alpha: Low

Frequency analysis reveals brain state objectively, without asking the person how they feel.

---

## HRV Frequency Analysis

Remember HRV from Chapter 4? We measured variation in time domain (SDNN, RMSSD). Frequency analysis reveals different information.

### Frequency Bands in HRV

**Very Low Frequency (VLF): 0.003-0.04 Hz**
- Very slow changes (periods of 25-300 seconds)
- Related to thermoregulation, hormones, long-term regulation
- Requires long recordings (24+ hours)

**Low Frequency (LF): 0.04-0.15 Hz**
- Periods of 7-25 seconds
- Reflects both sympathetic and parasympathetic activity
- Baroreceptor reflex (blood pressure regulation)

**High Frequency (HF): 0.15-0.4 Hz**
- Periods of 2.5-7 seconds
- Reflects parasympathetic (vagal) activity
- Respiratory sinus arrhythmia (breathing-related changes)

### LF/HF Ratio

The ratio of low frequency to high frequency power:

**LF/HF Ratio**:
- **High ratio (>2)**: Sympathetic dominance (stress, exercise)
- **Low ratio (<1)**: Parasympathetic dominance (rest, recovery)
- **Balanced (1-2)**: Healthy autonomic balance

**Example Interpretation**:
- Resting: LF/HF = 1.5 (balanced)
- During stress test: LF/HF = 4.0 (sympathetic activation)
- During meditation: LF/HF = 0.6 (parasympathetic activation)

**Controversy**: Some researchers debate what LF actually represents. It's not purely sympathetic. Use cautiously and consider context.

---

## Practical Frequency Analysis

### FFT: The Fast Fourier Transform

The **Fast Fourier Transform (FFT)** is an efficient algorithm for computing the Fourier transform. Developed in 1965, it made frequency analysis practical for computers.

**Before FFT**: Computing Fourier transform of 1000 points took ~1 million operations (very slow)
**With FFT**: Same computation takes ~10,000 operations (100× faster)

**Modern impact**: FFT is one of the most important algorithms ever developed. It's used in:
- Audio processing (MP3, noise cancellation)
- Image compression (JPEG)
- Telecommunications
- Medical imaging (MRI)
- Biosignal analysis

### Window Length

How much data do you analyze at once?

**Short Window** (1-2 seconds):
- **Pros**: See how frequencies change quickly over time
- **Cons**: Poor frequency resolution (can't distinguish close frequencies)

**Long Window** (10-60 seconds):
- **Pros**: Excellent frequency resolution
- **Cons**: Assumes signal is stationary (not changing) during window

**Trade-off**: Time resolution vs. frequency resolution. You can't maximize both simultaneously (Heisenberg uncertainty principle applies).

### Windowing Functions

**Problem**: Abrupt start and stop of analysis window creates artificial high frequencies.

**Solution**: Window functions that gradually fade in/out at edges.

**Common windows**:
- **Rectangular**: No tapering (sharp edges)
- **Hanning**: Smooth bell-shaped taper
- **Hamming**: Similar to Hanning, slightly different shape
- **Blackman**: More aggressive tapering

**Effect**: Reduces spectral leakage (frequencies bleeding into adjacent bins).

**Analogy**: Like fading music in and out instead of starting/stopping abruptly. Sounds smoother, less artificial.

---

## Spectrogram: Frequency Over Time

A **spectrogram** shows how frequencies change over time.

**Axes**:
- Horizontal: Time
- Vertical: Frequency
- Color/intensity: Power at each time-frequency point

**Visual Description**: Like a heat map. Bright areas show strong frequencies at that time. Dark areas show weak or absent frequencies.

### EEG Spectrogram Example

**Going to sleep**:
- Time 0-5 min: Bright band at 10 Hz (alpha) - awake with eyes closed
- Time 5-10 min: Alpha fading, theta (5-7 Hz) brightening - drowsy
- Time 10-30 min: Theta dominant - light sleep
- Time 30+ min: Low frequencies (delta, 1-3 Hz) brighten - deep sleep

You can literally see someone fall asleep by watching the spectrogram.

### Heart Rate Spectrogram

Shows how HRV frequency content changes during different activities:
- Resting: Strong peak at breathing frequency (HF band)
- Exercise: LF band increases, breathing frequency shifts higher
- Recovery: Gradual return to resting pattern

---

## Applications of Frequency Analysis

### Detecting Atrial Fibrillation

Normal sinus rhythm has:
- Clear peak at heart rate frequency
- Harmonics at multiples of heart rate

Atrial fibrillation has:
- No clear peak (irregular rhythm)
- Broad, diffuse power spectrum
- High entropy (disorder)

Automatic AF detection uses frequency features.

### Sleep Staging

Different sleep stages have characteristic EEG frequencies:
- **Wake**: Alpha and beta
- **Stage 1**: Theta increases
- **Stage 2**: Sleep spindles (12-14 Hz bursts)
- **Stage 3**: Delta waves dominant
- **REM**: Mixed frequencies, like wake but dreaming

Algorithms use frequency content to automatically score sleep stages.

### Muscle Fatigue Detection

Fresh muscle:
- EMG peak frequency: 80-100 Hz
- Sharp spectrum

Fatigued muscle:
- EMG peak frequency: 50-60 Hz (shifts lower)
- Broader spectrum

Frequency shift indicates fatigue objectively.

### Artifact Detection

Movement artifacts:
- Low-frequency power surge
- Broadband increase

Powerline noise:
- Sharp spike at 50 or 60 Hz

Muscle noise:
- High-frequency (>30 Hz) increase

Frequency analysis helps identify and remove artifacts.

---

## Common Mistakes

### Mistake 1: Too Short Signal
**Problem**: Need sufficient data for frequency resolution
**Rule**: At least 10 cycles of lowest frequency of interest
**Example**: To detect 0.1 Hz variation, need 100+ seconds of data

### Mistake 2: Ignoring Stationarity
**Problem**: Fourier transform assumes signal properties don't change
**Reality**: Biosignals often non-stationary
**Solution**: Use short windows or time-frequency methods (spectrogram)

### Mistake 3: Over-Interpreting Small Peaks
**Problem**: Noise creates random small peaks
**Solution**: Statistical significance testing, compare to noise floor

### Mistake 4: Wrong Frequency Range
**Problem**: Analyzing frequencies outside the signal's range
**Solution**: Know your signal's expected frequency content

### Mistake 5: Aliasing
**Problem**: Sampling rate too low for frequencies present
**Effect**: High frequencies appear as low frequencies
**Solution**: Sample at >2× highest frequency (Nyquist theorem)

---

## Beyond Basic Fourier Transform

### Wavelet Transform
Like Fourier transform but uses wavelets instead of sine waves. Better for non-stationary signals and localized events.

### Hilbert Transform
Extracts instantaneous frequency and amplitude. Good for analyzing oscillations that change frequency over time.

### Coherence Analysis
Measures frequency-domain correlation between two signals. Used to study brain connectivity, cardiorespiratory coupling.

### Bispectrum
Analyzes frequency interactions (how frequencies couple). Advanced technique for nonlinear systems.

---

## Try It Out

Want to explore frequency analysis?

1. **Audio Spectrum**: Use audio software with spectrum analyzer. Play different notes, see peaks at corresponding frequencies. Play chords, see multiple peaks.

2. **EEG Frequency Bands**: Some neurofeedback apps show real-time frequency bands. Close your eyes (alpha increases), do mental math (beta increases).

3. **Heart Rate Frequency**: Apps that measure HRV often show frequency analysis. Try slow breathing (increases HF power).

4. **Code It**: Python (SciPy), MATLAB, or even spreadsheet software can compute FFT. Generate sine waves, compute FFT, see the peaks.

---

## Going Deeper

**Key Takeaways**:
- Frequency measures how fast something repeats (cycles per second)
- Fourier transform decomposes signals into frequency components
- Power spectrum shows which frequencies are present and how strong
- Different frequencies correspond to different physiological processes
- EEG bands (delta, theta, alpha, beta, gamma) indicate brain states
- HRV frequency bands (VLF, LF, HF) reflect autonomic function
- Spectrogram shows frequency content changing over time

**Next Steps**:
Chapter 6 explores normative comparisons - how to determine if measurements are "normal" or "abnormal." You can measure heart rate, HRV, frequency content, but what do those numbers mean? Are they healthy? Concerning? That's what we'll cover next.

**Further Reading**:
- "The Scientist and Engineer's Guide to Digital Signal Processing" by Steven W. Smith (free online)
- "Understanding Digital Signal Processing" by Richard Lyons
- Interactive Fourier transform visualizations online

---

Frequency analysis reveals hidden patterns in biosignals - patterns invisible when you just look at the waveform. Like Newton's prism revealed the hidden rainbow in white light, the Fourier transform reveals the hidden frequencies in biosignals.

**Next**: [Chapter 6: Is This Normal?](ch06_comparing.md)
