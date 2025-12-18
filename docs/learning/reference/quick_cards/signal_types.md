# Quick Reference Card 1: Know Your Signals 🫀🧠💪

> **Visual guide to biosignal types** - What they measure, what they look like, where they come from

---

## 1. ECG (Electrocardiogram) - Heart Electrical Activity

**What it measures:** Electrical signals from your heartbeat

**Waveform:**
```
        R
       /\
      /  \
     /    \___S
  P /    Q
   /
__/____________T___
              / \
             /   \
```

**Key Features:**
- **P wave** = Atria contracting (top chambers squeeze)
- **QRS complex** = Ventricles contracting (main pump)
- **T wave** = Heart relaxing and resetting

**Electrode Placement:**
```
     Right Arm (-) ●        ● Left Arm (+)
                    \      /
                     \    /
                      \  /
                    🫀 Heart
                       |
                Right Leg (⏚) Ground
```

**Normal Values:**
- Heart Rate: 60-100 beats/min
- R-R Interval: Regular spacing

---

## 2. EEG (Electroencephalogram) - Brain Waves

**What it measures:** Electrical activity in your brain

**Different Wave Types:**
```
Delta (0.5-4 Hz) - Deep Sleep
~~~~~~~~~~~~~~~~~~~~~~~

Theta (4-8 Hz) - Drowsy/Creative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alpha (8-13 Hz) - Relaxed/Awake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beta (13-30 Hz) - Alert/Thinking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gamma (30+ Hz) - Focus/Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

**Electrode Cap:**
```
    Top View of Head

    Fp1   Fpz   Fp2    (Front)
     •     •     •
    F3    Fz    F4
     •     •     •
    C3    Cz    C4     (Center)
     •     •     •
    P3    Pz    P4
     •     •     •
    O1    Oz    O2     (Back)
```

**What You See:**
- Eyes closed = Strong alpha waves (back of head)
- Focused thinking = Beta waves (front)
- Sleeping = Delta waves (all over)

---

## 3. EMG (Electromyography) - Muscle Activity

**What it measures:** Electrical signals when muscles contract

**Waveform - Muscle at Rest vs. Active:**
```
RESTING (Quiet):
_________________

LIGHT CONTRACTION:
__|¯|__|¯|___|¯|__

STRONG CONTRACTION:
|¯|¯|¯|¯|¯|¯|¯|¯|¯
```

**Electrode Placement Example (Bicep):**
```
      Shoulder
         |
    +---------+
    |         |  ← Muscle
    | ●     ● |  ← Electrodes
    +---------+
         |
       Elbow
```

**Normal Pattern:**
- Rest = Almost no signal
- Squeeze = Burst of spikes
- Release = Back to quiet

---

## 4. PPG (Photoplethysmography) - Blood Flow

**What it measures:** Blood volume changes (using light)

**How It Works:**
```
   LED Light ●
      ↓↓↓
   [Finger/Skin]
   Blood vessels
      ↑↑↑
   Sensor ●
```

**Pulse Wave:**
```
     Systolic Peak
         /\
        /  \_____ Dicrotic Notch
       /         \
      /           \
_____/             \____

  ←─ One heartbeat ─→
```

**What You Can Measure:**
- Heart rate (pulse)
- Blood oxygen (SpO2)
- Pulse wave timing
- Arterial stiffness

**Normal Values:**
- Pulse Rate: 60-100 bpm
- SpO2: 95-100%

---

## 5. EDA (Electrodermal Activity) - Skin Conductance

**What it measures:** Sweat/stress response on skin

**Typical Response:**
```
STRESS EVENT ↓

Tonic ________/¯¯¯¯¯¯¯¯¯¯\_____
(Baseline)

Phasic         /\  /\
(Reactions)   /  \/  \
_____________/        \________

← Calm  |  Stressed  | Calm →
```

**Electrode Placement:**
```
   Palm of Hand

   ● Finger 1 (index)

   ● Finger 2 (middle)

   (Electrodes measure
    sweat gland activity)
```

**Normal Pattern:**
- Relaxed = Low, steady
- Stress/emotion = Quick spike (SCR)
- Stays elevated during stress

---

## 🎯 Pro Tips

| Signal | Best For | Watch Out For |
|--------|----------|---------------|
| **ECG** | Heart rhythm, HRV, fitness | Motion artifacts, poor contact |
| **EEG** | Sleep, focus, meditation | Eye blinks, jaw tension |
| **EMG** | Movement, fatigue, rehabilitation | Electrical noise, nearby muscles |
| **PPG** | Heart rate, stress, sleep | Motion, cold fingers |
| **EDA** | Stress, emotion, arousal | Slow response (2-5 sec delay) |

---

## Quick Signal Comparison

```
FREQUENCY RANGES:

ECG   ██░░░░░░░░░░░░  (0.5-40 Hz)
EEG   ████████░░░░░░  (0.5-100 Hz)
EMG   ░░░░░░░░██████  (20-500 Hz)
PPG   ██░░░░░░░░░░░░  (0.5-10 Hz)
EDA   █░░░░░░░░░░░░░  (0-5 Hz)

      └──────────────┘
      0    50   100  500 Hz
```

---

**Remember:** Different signals need different processing!
- **Slow signals** (EDA, PPG) = Lower sampling rates (10-100 Hz)
- **Fast signals** (EMG, EEG) = Higher sampling rates (250-1000 Hz)
