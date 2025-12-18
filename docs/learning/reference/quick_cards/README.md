# Quick Reference Cards - Delta-Predictive-Biosensing 🎯

> **One-page guides** for fast learning and troubleshooting

---

## What Are These?

Quick reference cards are **1-page scannable summaries** designed to give you instant answers without searching through documentation. Print them, keep them on a second monitor, or pull them up on your phone!

**Target Level:** 8th grade reading level
**Format:** Visual, with ASCII diagrams, tables, and real code examples
**Best For:** Learning basics, quick reminders, troubleshooting

---

## 📚 The Cards

### [Card 1: Know Your Signals](./signal_types.md)
**What's inside:** Visual guide to all biosignal types
- ECG (heart) - waveform shapes, electrode placement
- EEG (brain) - different wave types (alpha, beta, delta, etc.)
- EMG (muscle) - burst patterns
- PPG (blood flow) - pulse wave features
- EDA (skin conductance) - stress responses
- ASCII waveform drawings for each signal
- Normal value ranges

**Use this when:** "What signal should I use?" or "What does ECG look like?"

---

### [Card 2: What's Normal?](./normal_vs_abnormal.md)
**What's inside:** Side-by-side comparisons of healthy vs. concerning patterns
- Normal heart rhythm vs. arrhythmia (AFib, tachycardia)
- Awake EEG vs. sleep EEG vs. seizure patterns
- Relaxed EMG vs. active vs. fatigued muscle
- Calm EDA vs. stress response
- Healthy HRV vs. low HRV
- Visual indicators (✓ normal, ⚠️ check this, 🚨 urgent)

**Use this when:** "Does this signal look okay?" or "What am I looking at?"

---

### [Card 3: From Raw to Results](./analysis_steps.md)
**What's inside:** Complete signal processing workflow
- Flowchart: Load → Clean → Filter → Detect → Measure → Compare → Report
- What each step does
- When to use each step
- Common tools/functions
- Decision tree for different analysis goals
- Pipeline checklist

**Use this when:** "How do I analyze this signal?" or "What steps do I need?"

---

### [Card 4: Essential Code](./code_cheatsheet.md)
**What's inside:** Copy-paste ready code snippets
- Loading data (WFDB, CSV, synthetic)
- Filtering (bandpass, notch, presets for each signal type)
- Finding peaks (R-peaks, generic peak detection)
- Computing metrics (HRV, EEG bands, PPG, EMG fatigue)
- Visualization (plotting signals, peaks, spectra)
- Export (saving results to CSV, numpy)
- 3 complete workflow examples

**Use this when:** "How do I code this?" or "What's the syntax?"

---

### [Card 5: Common Problems & Fixes](./troubleshooting.md)
**What's inside:** FAQ-style problem solving
- "My signal looks flat" → Check connections, scaling
- "Too much noise" → Try filtering (with visual examples)
- "Can't find peaks" → Adjust threshold, filter first
- "Results don't match expected" → Check units, sample rate
- "Code won't run" → Installation, shape mismatches, imports
- Debugging template and flowchart

**Use this when:** Something's wrong and you need a quick fix!

---

## 🎯 How to Use These Cards

### For Learning:
1. **Start with Card 1** (Signal Types) - understand what you're measuring
2. **Then Card 2** (Normal vs Abnormal) - learn to recognize patterns
3. **Then Card 3** (Analysis Steps) - understand the workflow
4. **Use Card 4** (Code) - when you're ready to code
5. **Keep Card 5** (Troubleshooting) - handy for when things go wrong

### For Quick Reference:
- **Need code syntax?** → Card 4
- **Debugging a problem?** → Card 5
- **Forgot filter settings?** → Card 3 or 4
- **Is my signal normal?** → Card 2
- **What signal type is this?** → Card 1

### For Teaching:
These cards work great for:
- Workshop handouts (print one per page)
- New team member onboarding
- Student reference materials
- Conference/tutorial quick guides

---

## 📋 Quick Navigation Table

| I Want To... | Use This Card |
|--------------|---------------|
| Learn about different biosignals | [Signal Types](./signal_types.md) |
| Check if my signal is normal | [Normal vs Abnormal](./normal_vs_abnormal.md) |
| Understand the analysis workflow | [Analysis Steps](./analysis_steps.md) |
| Find code examples | [Code Cheatsheet](./code_cheatsheet.md) |
| Fix a problem | [Troubleshooting](./troubleshooting.md) |
| Get filter settings for ECG | [Code Cheatsheet](./code_cheatsheet.md#2-filtering-) |
| Detect R-peaks in ECG | [Code Cheatsheet](./code_cheatsheet.md#3-finding-peaks-) |
| Calculate HRV | [Code Cheatsheet](./code_cheatsheet.md#4-computing-metrics-) |
| Plot my signal | [Code Cheatsheet](./code_cheatsheet.md#5-visualization-) |
| Signal is too noisy | [Troubleshooting](./troubleshooting.md#-problem-too-much-noise---cant-see-the-signal) |
| Can't detect peaks | [Troubleshooting](./troubleshooting.md#-problem-cant-find-peaks-or-finding-wrong-peaks) |

---

## 🖨️ Printing Tips

Each card is designed to fit on one page when printed:

### For Physical Cards:
```
Paper: Letter/A4
Margins: 0.5 inch (12mm)
Font: Default markdown rendering
```

### For Digital Use:
- Keep them open in separate browser tabs
- Use markdown preview in your editor
- Render to PDF for annotation

---

## 💡 Pro Tips

1. **Start with the visuals** - Look at the ASCII diagrams first
2. **Try the code examples** - They're designed to actually work
3. **Use the checklists** - Don't skip steps
4. **Cross-reference** - Cards reference each other
5. **Print Card 5** - Keep troubleshooting handy for debugging sessions

---

## 🔗 Related Documentation

For deeper learning, check out:
- **Full API Guide:** `/docs/API_GUIDE.md` - Complete function reference
- **System Catalog:** `/docs/SYSTEM_CATALOG.md` - All capabilities
- **Learning Plan:** `/docs/learning/DOCUMENTATION_PLAN.md` - Complete curriculum
- **Books:** `/docs/learning/books/` - Comprehensive guides

---

## 📊 Card Statistics

| Card | Topic | Length | Code Examples | Diagrams |
|------|-------|--------|---------------|----------|
| 1 | Signal Types | 1 page | 0 | 15+ |
| 2 | Normal vs Abnormal | 1 page | 0 | 20+ |
| 3 | Analysis Steps | 1 page | 5 | 10+ |
| 4 | Code Cheatsheet | 1 page | 30+ | 5 |
| 5 | Troubleshooting | 1 page | 20+ | 10+ |

**Total:** 5 cards, 55+ code examples, 60+ visual diagrams

---

## 🤝 Feedback

Found an error? Have a suggestion?
- These cards are living documents
- Update them as you learn what works
- Add your own examples and tips

---

## Version History

- **v1.0** (Dec 2024) - Initial 5-card set created
  - Card 1: Signal Types
  - Card 2: Normal vs Abnormal
  - Card 3: Analysis Steps
  - Card 4: Code Cheatsheet
  - Card 5: Troubleshooting

---

**Happy Learning!** 🎓

Remember: These cards are meant to be **quick and practical**. For deeper understanding, dive into the full documentation. But for a fast answer or reminder, start here!
