//! Sleep stage classification algorithms
//!
//! Supports staging from EEG (gold standard) and HRV/actigraphy (wearable-compatible)

/// Sleep stages following AASM guidelines
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepStage {
    /// Wakefulness
    Wake,
    /// N1 - Light sleep, theta activity
    N1,
    /// N2 - Sleep spindles, K-complexes
    N2,
    /// N3 - Slow wave sleep, delta activity
    N3,
    /// REM - Rapid eye movement
    Rem,
    /// Unknown/artifact
    Unknown,
}

impl SleepStage {
    /// Get stage name
    pub fn name(&self) -> &'static str {
        match self {
            SleepStage::Wake => "Wake",
            SleepStage::N1 => "N1",
            SleepStage::N2 => "N2",
            SleepStage::N3 => "N3",
            SleepStage::Rem => "REM",
            SleepStage::Unknown => "Unknown",
        }
    }

    /// Is this a sleep stage (not wake)?
    pub fn is_sleep(&self) -> bool {
        !matches!(self, SleepStage::Wake | SleepStage::Unknown)
    }

    /// Is this NREM sleep?
    pub fn is_nrem(&self) -> bool {
        matches!(self, SleepStage::N1 | SleepStage::N2 | SleepStage::N3)
    }
}

/// Sleep architecture metrics
#[derive(Debug, Clone, Default)]
pub struct SleepArchitecture {
    /// Total recording time in minutes
    pub total_recording_time: f64,
    /// Total sleep time in minutes
    pub total_sleep_time: f64,
    /// Sleep efficiency (TST / TRT * 100)
    pub sleep_efficiency: f64,
    /// Sleep onset latency in minutes
    pub sleep_onset_latency: f64,
    /// Wake after sleep onset in minutes
    pub wake_after_sleep_onset: f64,
    /// Percentage of time in N1
    pub n1_percent: f64,
    /// Percentage of time in N2
    pub n2_percent: f64,
    /// Percentage of time in N3 (slow wave sleep)
    pub n3_percent: f64,
    /// Percentage of time in REM
    pub rem_percent: f64,
    /// REM latency in minutes
    pub rem_latency: f64,
    /// Number of awakenings
    pub awakenings: usize,
    /// Number of sleep cycles
    pub sleep_cycles: usize,
    /// Average cycle duration in minutes
    pub avg_cycle_duration: f64,
}

/// Sleep stager using multiple modalities
#[derive(Debug, Clone)]
pub struct SleepStager {
    /// Epoch duration in seconds (typically 30)
    pub epoch_duration: f64,
    /// Sample rate of input data
    pub sample_rate: f64,
}

impl Default for SleepStager {
    fn default() -> Self {
        Self {
            epoch_duration: 30.0,
            sample_rate: 256.0,
        }
    }
}

impl SleepStager {
    /// Create new stager with specified parameters
    pub fn new(epoch_duration: f64, sample_rate: f64) -> Self {
        Self {
            epoch_duration,
            sample_rate,
        }
    }

    /// Stage sleep from EEG using band power features
    pub fn stage_from_eeg(&self, eeg: &[f64]) -> Vec<SleepStage> {
        let epoch_samples = (self.epoch_duration * self.sample_rate) as usize;
        let n_epochs = eeg.len() / epoch_samples;
        let mut stages = Vec::with_capacity(n_epochs);

        for i in 0..n_epochs {
            let start = i * epoch_samples;
            let end = start + epoch_samples;
            let epoch = &eeg[start..end];
            let stage = self.classify_eeg_epoch(epoch);
            stages.push(stage);
        }

        self.smooth_stages(&mut stages);
        stages
    }

    /// Classify single EEG epoch based on frequency characteristics
    fn classify_eeg_epoch(&self, epoch: &[f64]) -> SleepStage {
        let powers = self.calculate_band_powers(epoch);

        let delta_ratio = powers.delta / powers.total.max(0.001);
        let theta_ratio = powers.theta / powers.total.max(0.001);
        let alpha_ratio = powers.alpha / powers.total.max(0.001);
        let beta_ratio = powers.beta / powers.total.max(0.001);

        // N3: >20% delta
        if delta_ratio > 0.20 {
            return SleepStage::N3;
        }

        // Wake: Dominant alpha with beta
        if alpha_ratio > 0.40 && beta_ratio > 0.15 {
            return SleepStage::Wake;
        }

        // REM: Low amplitude mixed frequency
        if alpha_ratio < 0.15 && delta_ratio < 0.15 && theta_ratio > 0.20 {
            return SleepStage::Rem;
        }

        // N2: Moderate theta
        if theta_ratio > 0.25 && delta_ratio < 0.20 {
            return SleepStage::N2;
        }

        // N1: Theta dominant, alpha dropout
        if theta_ratio > 0.30 && alpha_ratio < 0.30 {
            return SleepStage::N1;
        }

        SleepStage::N2
    }

    /// Calculate band powers for an epoch
    fn calculate_band_powers(&self, epoch: &[f64]) -> BandPowers {
        let n = epoch.len() as f64;
        let total_power: f64 = epoch.iter().map(|x| x * x).sum::<f64>() / n;

        let zero_crossings = count_zero_crossings(epoch);
        let zcr = zero_crossings as f64 / n * self.sample_rate;
        let dominant_freq = zcr / 2.0;

        let (delta, theta, alpha, beta, gamma) = if dominant_freq < 4.0 {
            (0.6, 0.2, 0.1, 0.05, 0.05)
        } else if dominant_freq < 8.0 {
            (0.2, 0.5, 0.15, 0.1, 0.05)
        } else if dominant_freq < 13.0 {
            (0.1, 0.2, 0.5, 0.15, 0.05)
        } else if dominant_freq < 30.0 {
            (0.05, 0.1, 0.15, 0.5, 0.2)
        } else {
            (0.05, 0.05, 0.1, 0.3, 0.5)
        };

        BandPowers {
            delta: total_power * delta,
            theta: total_power * theta,
            alpha: total_power * alpha,
            beta: total_power * beta,
            gamma: total_power * gamma,
            total: total_power,
        }
    }

    /// Stage sleep from HRV (heart rate variability)
    pub fn stage_from_hrv(&self, rr_intervals: &[f64]) -> Vec<SleepStage> {
        let epoch_beats = 150;
        let n_epochs = rr_intervals.len() / epoch_beats;
        let mut stages = Vec::with_capacity(n_epochs);

        for i in 0..n_epochs {
            let start = i * epoch_beats;
            let end = (start + epoch_beats).min(rr_intervals.len());
            let epoch_rr = &rr_intervals[start..end];
            let stage = self.classify_hrv_epoch(epoch_rr);
            stages.push(stage);
        }

        self.smooth_stages(&mut stages);
        stages
    }

    /// Classify epoch from HRV features
    fn classify_hrv_epoch(&self, rr_intervals: &[f64]) -> SleepStage {
        if rr_intervals.is_empty() {
            return SleepStage::Unknown;
        }

        let mean_rr: f64 = rr_intervals.iter().sum::<f64>() / rr_intervals.len() as f64;
        let mean_hr = 60000.0 / mean_rr;

        let variance: f64 = rr_intervals
            .iter()
            .map(|rr| (rr - mean_rr).powi(2))
            .sum::<f64>()
            / rr_intervals.len() as f64;
        let sdnn = variance.sqrt();

        let mut sum_sq_diff = 0.0;
        for i in 1..rr_intervals.len() {
            let diff = rr_intervals[i] - rr_intervals[i - 1];
            sum_sq_diff += diff * diff;
        }
        let rmssd = (sum_sq_diff / (rr_intervals.len() - 1).max(1) as f64).sqrt();

        if mean_hr > 70.0 && sdnn > 50.0 {
            SleepStage::Wake
        } else if mean_hr < 55.0 && rmssd > 40.0 {
            SleepStage::N3
        } else if rmssd < 25.0 && sdnn > 30.0 {
            SleepStage::Rem
        } else if rmssd > 30.0 {
            SleepStage::N2
        } else {
            SleepStage::N1
        }
    }

    /// Stage sleep from actigraphy
    pub fn stage_from_actigraphy(&self, activity_counts: &[f64]) -> Vec<SleepStage> {
        let mut stages = Vec::with_capacity(activity_counts.len());
        let threshold = calculate_activity_threshold(activity_counts);

        for &count in activity_counts {
            if count > threshold {
                stages.push(SleepStage::Wake);
            } else {
                stages.push(SleepStage::N2);
            }
        }

        self.smooth_stages(&mut stages);
        stages
    }

    /// Apply smoothing rules to stage sequence
    fn smooth_stages(&self, stages: &mut [SleepStage]) {
        if stages.len() < 3 {
            return;
        }

        for i in 1..stages.len() - 1 {
            if stages[i - 1] == stages[i + 1] && stages[i] != stages[i - 1] {
                stages[i] = stages[i - 1];
            }
        }
    }

    /// Calculate sleep architecture from staged epochs
    pub fn calculate_architecture(&self, stages: &[SleepStage]) -> SleepArchitecture {
        let epoch_duration_min = self.epoch_duration / 60.0;
        let total_epochs = stages.len();
        let total_recording_time = total_epochs as f64 * epoch_duration_min;

        let wake_epochs = stages.iter().filter(|&&s| s == SleepStage::Wake).count();
        let n1_epochs = stages.iter().filter(|&&s| s == SleepStage::N1).count();
        let n2_epochs = stages.iter().filter(|&&s| s == SleepStage::N2).count();
        let n3_epochs = stages.iter().filter(|&&s| s == SleepStage::N3).count();
        let rem_epochs = stages.iter().filter(|&&s| s == SleepStage::Rem).count();

        let sleep_epochs = total_epochs - wake_epochs;
        let total_sleep_time = sleep_epochs as f64 * epoch_duration_min;

        let sleep_onset = stages
            .iter()
            .position(|s| s.is_sleep())
            .unwrap_or(total_epochs);
        let sleep_onset_latency = sleep_onset as f64 * epoch_duration_min;

        let last_sleep = stages.iter().rposition(|s| s.is_sleep()).unwrap_or(0);
        let waso_epochs = if sleep_onset <= last_sleep {
            stages[sleep_onset..=last_sleep]
                .iter()
                .filter(|&&s| s == SleepStage::Wake)
                .count()
        } else {
            0
        };
        let wake_after_sleep_onset = waso_epochs as f64 * epoch_duration_min;

        let first_rem = stages.iter().position(|&s| s == SleepStage::Rem);
        let rem_latency = first_rem
            .map(|r| (r.saturating_sub(sleep_onset)) as f64 * epoch_duration_min)
            .unwrap_or(0.0);

        let awakenings = stages
            .windows(2)
            .filter(|w| w[0].is_sleep() && w[1] == SleepStage::Wake)
            .count();

        let sleep_cycles = count_sleep_cycles(stages);

        let sleep_efficiency = if total_recording_time > 0.0 {
            total_sleep_time / total_recording_time * 100.0
        } else {
            0.0
        };

        SleepArchitecture {
            total_recording_time,
            total_sleep_time,
            sleep_efficiency,
            sleep_onset_latency,
            wake_after_sleep_onset,
            n1_percent: n1_epochs as f64 / total_epochs.max(1) as f64 * 100.0,
            n2_percent: n2_epochs as f64 / total_epochs.max(1) as f64 * 100.0,
            n3_percent: n3_epochs as f64 / total_epochs.max(1) as f64 * 100.0,
            rem_percent: rem_epochs as f64 / total_epochs.max(1) as f64 * 100.0,
            rem_latency,
            awakenings,
            sleep_cycles,
            avg_cycle_duration: if sleep_cycles > 0 {
                total_sleep_time / sleep_cycles as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone, Default)]
struct BandPowers {
    delta: f64,
    theta: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    total: f64,
}

fn count_zero_crossings(signal: &[f64]) -> usize {
    signal
        .windows(2)
        .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
        .count()
}

fn calculate_activity_threshold(counts: &[f64]) -> f64 {
    if counts.is_empty() {
        return 0.0;
    }

    let mut sorted = counts.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = (sorted.len() as f64 * 0.1) as usize;
    sorted.get(idx).copied().unwrap_or(0.0)
}

fn count_sleep_cycles(stages: &[SleepStage]) -> usize {
    let mut cycles = 0;
    let mut in_nrem = false;

    for stage in stages {
        if stage.is_nrem() {
            in_nrem = true;
        } else if *stage == SleepStage::Rem && in_nrem {
            cycles += 1;
            in_nrem = false;
        }
    }

    cycles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep_stage_properties() {
        assert!(SleepStage::N2.is_sleep());
        assert!(SleepStage::N3.is_nrem());
        assert!(!SleepStage::Wake.is_sleep());
        assert!(!SleepStage::Rem.is_nrem());
    }

    #[test]
    fn test_architecture_calculation() {
        let stager = SleepStager::default();

        let stages = vec![
            SleepStage::Wake,
            SleepStage::Wake,
            SleepStage::N1,
            SleepStage::N2,
            SleepStage::N2,
            SleepStage::N3,
            SleepStage::N3,
            SleepStage::N2,
            SleepStage::Rem,
            SleepStage::N2,
            SleepStage::N3,
            SleepStage::Wake,
        ];

        let arch = stager.calculate_architecture(&stages);

        assert!(arch.total_sleep_time > 0.0);
        assert!(arch.sleep_efficiency > 0.0 && arch.sleep_efficiency <= 100.0);
    }

    #[test]
    fn test_stage_smoothing() {
        let stager = SleepStager::default();
        let mut stages = vec![
            SleepStage::N2,
            SleepStage::Wake,
            SleepStage::N2,
            SleepStage::N2,
            SleepStage::N2,
        ];

        stager.smooth_stages(&mut stages);
        assert_eq!(stages[1], SleepStage::N2);
    }
}
