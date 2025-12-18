//! Sleep feature extraction for staging algorithms

/// Sleep spindle detection result
#[derive(Debug, Clone)]
pub struct SleepSpindle {
    /// Start sample index
    pub start_sample: usize,
    /// End sample index
    pub end_sample: usize,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Dominant frequency in Hz
    pub frequency_hz: f64,
    /// Peak amplitude
    pub amplitude: f64,
}

/// K-complex detection result
#[derive(Debug, Clone)]
pub struct KComplex {
    /// Sample at negative peak
    pub peak_sample: usize,
    /// Peak-to-peak amplitude
    pub amplitude: f64,
    /// Duration in milliseconds
    pub duration_ms: f64,
}

/// Slow wave detection result
#[derive(Debug, Clone)]
pub struct SlowWave {
    /// Start sample index
    pub start_sample: usize,
    /// End sample index
    pub end_sample: usize,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Peak-to-peak amplitude
    pub peak_to_peak_amplitude: f64,
    /// Frequency in Hz
    pub frequency_hz: f64,
}

/// Sleep feature extractor
#[derive(Debug, Clone)]
pub struct SleepFeatureExtractor {
    /// Sample rate in Hz
    pub sample_rate: f64,
}

impl Default for SleepFeatureExtractor {
    fn default() -> Self {
        Self { sample_rate: 256.0 }
    }
}

impl SleepFeatureExtractor {
    /// Create new extractor with specified sample rate
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }

    /// Detect sleep spindles (12-14 Hz bursts, 0.5-2 seconds)
    pub fn detect_spindles(&self, eeg: &[f64]) -> Vec<SleepSpindle> {
        let mut spindles = Vec::new();

        let window_size = (0.5 * self.sample_rate) as usize;

        for i in (0..eeg.len()).step_by(window_size / 2) {
            let end = (i + window_size).min(eeg.len());
            if end - i < window_size {
                break;
            }

            let window = &eeg[i..end];
            let zcr = count_zero_crossings_rate(window, self.sample_rate);

            // Spindle frequency range: 12-14 Hz
            if zcr > 20.0 && zcr < 32.0 {
                let amplitude = window.iter().map(|x| x.abs()).fold(0.0f64, f64::max);

                if amplitude > 20.0 {
                    spindles.push(SleepSpindle {
                        start_sample: i,
                        end_sample: end,
                        duration_ms: (end - i) as f64 / self.sample_rate * 1000.0,
                        frequency_hz: zcr / 2.0,
                        amplitude,
                    });
                }
            }
        }

        spindles
    }

    /// Detect K-complexes (sharp negative wave followed by positive wave)
    pub fn detect_k_complexes(&self, eeg: &[f64]) -> Vec<KComplex> {
        let mut k_complexes = Vec::new();

        let threshold = 75.0;

        for i in 1..eeg.len().saturating_sub(1) {
            if eeg[i] < -threshold && eeg[i] < eeg[i - 1] && eeg[i] < eeg[i + 1] {
                let search_end = (i + (self.sample_rate * 0.5) as usize).min(eeg.len());

                let max_pos = eeg[i..search_end]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);

                if max_pos > threshold / 2.0 {
                    k_complexes.push(KComplex {
                        peak_sample: i,
                        amplitude: eeg[i].abs() + max_pos,
                        duration_ms: (search_end - i) as f64 / self.sample_rate * 1000.0,
                    });
                }
            }
        }

        k_complexes
    }

    /// Detect slow waves (N3 markers, 0.5-2 Hz, >75µV)
    pub fn detect_slow_waves(&self, eeg: &[f64]) -> Vec<SlowWave> {
        let mut slow_waves = Vec::new();

        let min_duration = (self.sample_rate / 2.0) as usize;
        let max_duration = (self.sample_rate * 2.0) as usize;

        let mut i = 0;
        while i < eeg.len() {
            if eeg[i] < -37.5 {
                let start = i;

                while i < eeg.len() && eeg[i] < 0.0 {
                    i += 1;
                }

                if i < eeg.len() {
                    let mut max_pos = 0.0f64;
                    while i < eeg.len() && eeg[i] >= 0.0 {
                        max_pos = max_pos.max(eeg[i]);
                        i += 1;
                    }

                    let duration = i - start;
                    let min_neg = eeg[start..i.min(eeg.len())]
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, f64::min);
                    let peak_to_peak = max_pos - min_neg;

                    if duration >= min_duration && duration <= max_duration && peak_to_peak > 75.0 {
                        slow_waves.push(SlowWave {
                            start_sample: start,
                            end_sample: i,
                            duration_ms: duration as f64 / self.sample_rate * 1000.0,
                            peak_to_peak_amplitude: peak_to_peak,
                            frequency_hz: self.sample_rate / duration as f64,
                        });
                    }
                }
            } else {
                i += 1;
            }
        }

        slow_waves
    }

    /// Calculate spindle density (spindles per minute)
    pub fn spindle_density(&self, spindles: &[SleepSpindle], duration_sec: f64) -> f64 {
        if duration_sec <= 0.0 {
            return 0.0;
        }
        spindles.len() as f64 / (duration_sec / 60.0)
    }

    /// Calculate slow wave activity (SWA) - power in 0.5-4 Hz band
    pub fn slow_wave_activity(&self, eeg: &[f64]) -> f64 {
        if eeg.is_empty() {
            return 0.0;
        }

        // Simplified SWA calculation using variance of low-pass filtered signal
        // In production would use proper bandpass filter
        let mean: f64 = eeg.iter().sum::<f64>() / eeg.len() as f64;
        let variance: f64 = eeg.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / eeg.len() as f64;

        variance.sqrt()
    }
}

fn count_zero_crossings_rate(signal: &[f64], sample_rate: f64) -> f64 {
    let crossings = signal
        .windows(2)
        .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
        .count();

    crossings as f64 / signal.len() as f64 * sample_rate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = SleepFeatureExtractor::default();
        assert_eq!(extractor.sample_rate, 256.0);
    }

    #[test]
    fn test_spindle_density() {
        let extractor = SleepFeatureExtractor::default();
        let spindles = vec![
            SleepSpindle {
                start_sample: 0,
                end_sample: 100,
                duration_ms: 500.0,
                frequency_hz: 13.0,
                amplitude: 30.0,
            },
            SleepSpindle {
                start_sample: 200,
                end_sample: 300,
                duration_ms: 500.0,
                frequency_hz: 13.0,
                amplitude: 25.0,
            },
        ];

        let density = extractor.spindle_density(&spindles, 60.0);
        assert!((density - 2.0).abs() < 0.01);
    }
}
