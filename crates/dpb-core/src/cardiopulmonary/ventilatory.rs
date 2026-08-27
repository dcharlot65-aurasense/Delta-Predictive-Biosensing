//! Ventilatory threshold detection
//!
//! Implements algorithms for detecting:
//! - First ventilatory threshold (VT1/aerobic threshold)
//! - Second ventilatory threshold (VT2/anaerobic threshold)
//! - Respiratory compensation point

/// Method for detecting ventilatory threshold
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VtMethod {
    /// V-slope method (VCO2 vs VO2)
    VSlope,
    /// Ventilatory equivalents (VE/VO2, VE/VCO2)
    VentilatoryEquivalents,
    /// Excess CO2 method
    ExcessCo2,
    /// End-tidal gas method (PETO2, PETCO2)
    EndTidal,
    /// Heart rate deflection point
    HrDeflection,
    /// Talk test threshold
    TalkTest,
}

/// Ventilatory threshold result
#[derive(Debug, Clone)]
pub struct VtResult {
    /// VO2 at threshold (L/min)
    pub vo2_at_vt: f64,
    /// Percent of VO2max
    pub percent_vo2max: f64,
    /// Heart rate at threshold
    pub hr_at_vt: Option<f64>,
    /// Workload at threshold (watts or speed)
    pub workload_at_vt: Option<f64>,
    /// Detection method used
    pub method: VtMethod,
    /// Confidence in detection (0-1)
    pub confidence: f64,
}

/// Ventilatory threshold analyzer
#[derive(Debug, Clone)]
pub struct VentilatoryThreshold {
    /// Sampling interval (seconds)
    pub sample_interval: f64,
    /// Moving average window size
    pub smoothing_window: usize,
}

impl Default for VentilatoryThreshold {
    fn default() -> Self {
        Self {
            sample_interval: 15.0, // 15-second averages typical for CPET
            smoothing_window: 5,
        }
    }
}

impl VentilatoryThreshold {
    /// Create new VT analyzer
    pub fn new(sample_interval: f64, smoothing_window: usize) -> Self {
        Self {
            sample_interval,
            smoothing_window,
        }
    }

    /// Detect VT1 using V-slope method
    /// vo2: VO2 values (L/min), vco2: VCO2 values (L/min)
    pub fn v_slope(&self, vo2: &[f64], vco2: &[f64]) -> Option<VtResult> {
        if vo2.len() != vco2.len() || vo2.len() < 10 {
            return None;
        }

        // Smooth data
        let vo2_smooth = self.moving_average(vo2);
        let vco2_smooth = self.moving_average(vco2);

        // Find breakpoint where VCO2/VO2 slope increases
        let mut best_breakpoint = vo2.len() / 2;
        let mut best_score = f64::MAX;

        for bp in (vo2.len() / 4)..(3 * vo2.len() / 4) {
            // Fit two lines: before and after breakpoint
            let (slope1, _, residuals1) = self.linear_fit(&vo2_smooth[..bp], &vco2_smooth[..bp]);
            let (slope2, _, residuals2) = self.linear_fit(&vo2_smooth[bp..], &vco2_smooth[bp..]);

            // VT1: slope increases from ~1.0 to >1.0
            if slope2 > slope1 && slope1 > 0.5 && slope2 < 2.0 {
                let score = residuals1 + residuals2;
                if score < best_score {
                    best_score = score;
                    best_breakpoint = bp;
                }
            }
        }

        let vo2_at_vt = vo2_smooth[best_breakpoint];
        let vo2max = vo2.iter().cloned().fold(f64::NAN, f64::max);

        Some(VtResult {
            vo2_at_vt,
            percent_vo2max: (vo2_at_vt / vo2max) * 100.0,
            hr_at_vt: None,
            workload_at_vt: None,
            method: VtMethod::VSlope,
            confidence: 1.0 - (best_score / vo2.len() as f64).min(0.5),
        })
    }

    /// Detect VT using ventilatory equivalents
    /// ve: minute ventilation, vo2: oxygen uptake, vco2: CO2 production
    pub fn ventilatory_equivalents(
        &self,
        ve: &[f64],
        vo2: &[f64],
        vco2: &[f64],
    ) -> Option<VtResult> {
        if ve.len() != vo2.len() || vo2.len() != vco2.len() || vo2.len() < 10 {
            return None;
        }

        // Calculate ventilatory equivalents
        let ve_vo2: Vec<f64> = ve.iter().zip(vo2.iter()).map(|(v, o)| v / o).collect();
        let ve_vco2: Vec<f64> = ve.iter().zip(vco2.iter()).map(|(v, c)| v / c).collect();

        // VT1: VE/VO2 increases while VE/VCO2 stable or decreasing
        // VT2: Both VE/VO2 and VE/VCO2 increase

        // Find nadir of VE/VO2
        let mut vt1_idx = 0;
        let mut min_ve_vo2 = f64::MAX;

        for (i, &v) in ve_vo2.iter().enumerate().skip(5) {
            if v < min_ve_vo2 && i < ve_vo2.len() - 5 {
                min_ve_vo2 = v;
                vt1_idx = i;
            }
        }

        // Verify VT1: VE/VO2 rises after its nadir *while VE/VCO2 stays flat
        // or falls*. The second half is what separates VT1 from VT2 -- at VT2
        // both equivalents climb -- so checking only VE/VO2 will report VT2 as
        // VT1 in a test that reaches it.
        if vt1_idx > 0 && vt1_idx < ve_vo2.len() - 3 {
            let mean_after: f64 = ve_vo2[vt1_idx..vt1_idx + 3].iter().sum::<f64>() / 3.0;
            let vco2_at_nadir = ve_vco2[vt1_idx];
            let vco2_after: f64 = ve_vco2[vt1_idx..vt1_idx + 3].iter().sum::<f64>() / 3.0;
            let vco2_stable = vco2_after <= vco2_at_nadir * 1.05;
            if mean_after > min_ve_vo2 * 1.05 && vco2_stable {
                let vo2_at_vt = vo2[vt1_idx];
                let vo2max = vo2.iter().cloned().fold(f64::NAN, f64::max);

                return Some(VtResult {
                    vo2_at_vt,
                    percent_vo2max: (vo2_at_vt / vo2max) * 100.0,
                    hr_at_vt: None,
                    workload_at_vt: None,
                    method: VtMethod::VentilatoryEquivalents,
                    confidence: 0.85,
                });
            }
        }

        None
    }

    /// Detect VT using excess CO2 method
    pub fn excess_co2(&self, vo2: &[f64], vco2: &[f64]) -> Option<VtResult> {
        if vo2.len() != vco2.len() || vo2.len() < 10 {
            return None;
        }

        // Excess CO2 = VCO2^2 / VO2 - VCO2
        let excess: Vec<f64> = vo2
            .iter()
            .zip(vco2.iter())
            .map(|(&o, &c)| (c * c) / o - c)
            .collect();

        let excess_smooth = self.moving_average(&excess);

        // VT at point where excess CO2 starts increasing non-linearly
        let mut vt_idx = excess_smooth.len() / 2;
        let mut max_second_derivative = 0.0;

        for i in 2..excess_smooth.len() - 2 {
            let second_deriv = excess_smooth[i + 1] - 2.0 * excess_smooth[i] + excess_smooth[i - 1];
            if second_deriv > max_second_derivative {
                max_second_derivative = second_deriv;
                vt_idx = i;
            }
        }

        let vo2_at_vt = vo2[vt_idx];
        let vo2max = vo2.iter().cloned().fold(f64::NAN, f64::max);

        Some(VtResult {
            vo2_at_vt,
            percent_vo2max: (vo2_at_vt / vo2max) * 100.0,
            hr_at_vt: None,
            workload_at_vt: None,
            method: VtMethod::ExcessCo2,
            confidence: 0.80,
        })
    }

    /// Detect VT using end-tidal gas pressures
    pub fn end_tidal(&self, peto2: &[f64], petco2: &[f64], vo2: &[f64]) -> Option<VtResult> {
        if peto2.len() != petco2.len() || vo2.len() != peto2.len() || vo2.len() < 10 {
            return None;
        }

        // VT1: PETO2 starts increasing while PETCO2 stable
        // VT2: PETCO2 starts decreasing

        // Find nadir of PETO2
        let peto2_smooth = self.moving_average(peto2);
        let mut vt_idx = 0;
        let mut min_peto2 = f64::MAX;

        for (i, &p) in peto2_smooth.iter().enumerate().skip(3) {
            if p < min_peto2 && i < peto2_smooth.len() - 3 {
                min_peto2 = p;
                vt_idx = i;
            }
        }

        if vt_idx > 0 {
            let vo2_at_vt = vo2[vt_idx];
            let vo2max = vo2.iter().cloned().fold(f64::NAN, f64::max);

            return Some(VtResult {
                vo2_at_vt,
                percent_vo2max: (vo2_at_vt / vo2max) * 100.0,
                hr_at_vt: None,
                workload_at_vt: None,
                method: VtMethod::EndTidal,
                confidence: 0.85,
            });
        }

        None
    }

    /// Detect VT using heart rate deflection point (Conconi)
    pub fn hr_deflection(&self, hr: &[f64], workload: &[f64]) -> Option<VtResult> {
        if hr.len() != workload.len() || hr.len() < 8 {
            return None;
        }

        let hr_smooth = self.moving_average(hr);

        // Find deflection point: where HR-workload relationship becomes non-linear
        let mut best_breakpoint = hr.len() / 2;
        let mut best_score = f64::MAX;

        for bp in (hr.len() / 3)..(2 * hr.len() / 3) {
            let (slope1, _, res1) = self.linear_fit(&workload[..bp], &hr_smooth[..bp]);
            let (slope2, _, res2) = self.linear_fit(&workload[bp..], &hr_smooth[bp..]);

            // Deflection: slope decreases
            if slope2 < slope1 {
                let score = res1 + res2 + (slope1 - slope2).abs() * 10.0;
                if score < best_score {
                    best_score = score;
                    best_breakpoint = bp;
                }
            }
        }

        Some(VtResult {
            vo2_at_vt: 0.0, // Not available from HR method
            percent_vo2max: 0.0,
            hr_at_vt: Some(hr_smooth[best_breakpoint]),
            workload_at_vt: Some(workload[best_breakpoint]),
            method: VtMethod::HrDeflection,
            confidence: 0.70, // HR deflection less reliable
        })
    }

    /// Estimate VT from talk test (percent HR max where talking becomes difficult)
    pub fn talk_test_estimate(&self, hr_max: f64, can_talk_comfortably: bool) -> VtResult {
        // Talk test typically corresponds to VT1
        let hr_at_vt = if can_talk_comfortably {
            hr_max * 0.65 // Below VT1
        } else {
            hr_max * 0.77 // Near or above VT1
        };

        VtResult {
            vo2_at_vt: 0.0,
            percent_vo2max: if can_talk_comfortably { 55.0 } else { 70.0 },
            hr_at_vt: Some(hr_at_vt),
            workload_at_vt: None,
            method: VtMethod::TalkTest,
            confidence: 0.60,
        }
    }

    /// Detect second ventilatory threshold (respiratory compensation point)
    pub fn detect_vt2(&self, ve: &[f64], vco2: &[f64], vo2: &[f64]) -> Option<VtResult> {
        if ve.len() != vco2.len() || vo2.len() != vco2.len() || vo2.len() < 10 {
            return None;
        }

        // VE/VCO2 method: VT2 when VE/VCO2 starts increasing
        let ve_vco2: Vec<f64> = ve.iter().zip(vco2.iter()).map(|(v, c)| v / c).collect();

        let ve_vco2_smooth = self.moving_average(&ve_vco2);

        // Find nadir of VE/VCO2 (typically VT2)
        let mut vt2_idx = ve_vco2_smooth.len() * 2 / 3;
        let mut min_ve_vco2 = f64::MAX;

        for (i, &v) in ve_vco2_smooth
            .iter()
            .enumerate()
            .skip(ve_vco2_smooth.len() / 3)
        {
            if v < min_ve_vco2 {
                min_ve_vco2 = v;
                vt2_idx = i;
            }
        }

        // Verify: VE/VCO2 should increase after VT2
        if vt2_idx < ve_vco2_smooth.len() - 2 {
            let after_mean = (ve_vco2_smooth[vt2_idx + 1] + ve_vco2_smooth[vt2_idx + 2]) / 2.0;
            if after_mean > min_ve_vco2 * 1.03 {
                let vo2_at_vt = vo2[vt2_idx];
                let vo2max = vo2.iter().cloned().fold(f64::NAN, f64::max);

                return Some(VtResult {
                    vo2_at_vt,
                    percent_vo2max: (vo2_at_vt / vo2max) * 100.0,
                    hr_at_vt: None,
                    workload_at_vt: None,
                    method: VtMethod::VentilatoryEquivalents,
                    confidence: 0.80,
                });
            }
        }

        None
    }

    /// Calculate training zones based on VT1 and VT2
    pub fn training_zones(&self, vt1: &VtResult, vt2: Option<&VtResult>) -> TrainingZones {
        let vt1_hr = vt1.hr_at_vt.unwrap_or(0.0);
        let vt2_hr = vt2.and_then(|v| v.hr_at_vt).unwrap_or(vt1_hr * 1.12);

        TrainingZones {
            zone1_max_hr: vt1_hr * 0.85,           // Recovery
            zone2_max_hr: vt1_hr,                  // Aerobic base
            zone3_max_hr: (vt1_hr + vt2_hr) / 2.0, // Tempo
            zone4_max_hr: vt2_hr,                  // Threshold
            zone5_max_hr: vt2_hr * 1.05,           // VO2max
        }
    }

    /// Moving average smoothing
    fn moving_average(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.smoothing_window {
            return data.to_vec();
        }

        let mut result = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let start = i.saturating_sub(self.smoothing_window / 2);
            let end = (i + self.smoothing_window / 2 + 1).min(data.len());
            let avg = data[start..end].iter().sum::<f64>() / (end - start) as f64;
            result.push(avg);
        }

        result
    }

    /// Simple linear regression
    fn linear_fit(&self, x: &[f64], y: &[f64]) -> (f64, f64, f64) {
        if x.len() != y.len() || x.is_empty() {
            return (0.0, 0.0, f64::MAX);
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return (0.0, sum_y / n, f64::MAX);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate residuals
        let residuals: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
            .sum();

        (slope, intercept, residuals)
    }
}

/// Training zones based on ventilatory thresholds
#[derive(Debug, Clone)]
pub struct TrainingZones {
    /// Zone 1 ceiling (recovery)
    pub zone1_max_hr: f64,
    /// Zone 2 ceiling (aerobic base)
    pub zone2_max_hr: f64,
    /// Zone 3 ceiling (tempo)
    pub zone3_max_hr: f64,
    /// Zone 4 ceiling (threshold)
    pub zone4_max_hr: f64,
    /// Zone 5 ceiling (VO2max)
    pub zone5_max_hr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v_slope() {
        let vt = VentilatoryThreshold::default();

        // Simulate exercise test data
        let vo2: Vec<f64> = (1..=20).map(|i| i as f64 * 0.2).collect();
        let vco2: Vec<f64> = vo2
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i < 10 {
                    v * 0.95 // Below VT: RER ~0.95
                } else {
                    v * 0.95 + (i as f64 - 10.0) * 0.05 // Above VT: RER increases
                }
            })
            .collect();

        let result = vt.v_slope(&vo2, &vco2);
        assert!(result.is_some());

        let vt_result = result.unwrap();
        assert!(vt_result.vo2_at_vt > 1.5 && vt_result.vo2_at_vt < 3.0);
    }

    #[test]
    fn test_training_zones() {
        let vt = VentilatoryThreshold::default();

        let vt1 = VtResult {
            vo2_at_vt: 2.0,
            percent_vo2max: 60.0,
            hr_at_vt: Some(140.0),
            workload_at_vt: Some(150.0),
            method: VtMethod::VSlope,
            confidence: 0.9,
        };

        let vt2 = VtResult {
            vo2_at_vt: 2.8,
            percent_vo2max: 85.0,
            hr_at_vt: Some(165.0),
            workload_at_vt: Some(220.0),
            method: VtMethod::VSlope,
            confidence: 0.85,
        };

        let zones = vt.training_zones(&vt1, Some(&vt2));

        assert!(zones.zone2_max_hr < zones.zone3_max_hr);
        assert!(zones.zone3_max_hr < zones.zone4_max_hr);
    }
}
