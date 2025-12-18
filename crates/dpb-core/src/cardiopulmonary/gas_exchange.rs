//! Respiratory gas exchange analysis
//!
//! Implements algorithms for analyzing respiratory gases during exercise:
//! - Respiratory exchange ratio (RER/RQ)
//! - Oxygen pulse
//! - Ventilatory efficiency
//! - Substrate utilization

/// Respiratory quotient/exchange ratio classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RespiratoryQuotient {
    /// RER < 0.7 (ketosis/fasting)
    VeryLowFat,
    /// RER 0.7-0.75 (predominant fat oxidation)
    FatOxidation,
    /// RER 0.75-0.85 (mixed substrates)
    Mixed,
    /// RER 0.85-1.0 (predominant carbohydrate)
    CarbOxidation,
    /// RER > 1.0 (anaerobic metabolism)
    Anaerobic,
}

impl RespiratoryQuotient {
    /// Classify RER value
    pub fn from_rer(rer: f64) -> Self {
        match rer {
            r if r < 0.70 => RespiratoryQuotient::VeryLowFat,
            r if r < 0.75 => RespiratoryQuotient::FatOxidation,
            r if r < 0.85 => RespiratoryQuotient::Mixed,
            r if r < 1.00 => RespiratoryQuotient::CarbOxidation,
            _ => RespiratoryQuotient::Anaerobic,
        }
    }

    /// Estimate percent carbohydrate contribution
    pub fn carb_percentage(rer: f64) -> f64 {
        // Non-protein RQ table approximation
        let rer_clamped = rer.clamp(0.7, 1.0);
        ((rer_clamped - 0.7) / 0.3) * 100.0
    }

    /// Estimate percent fat contribution
    pub fn fat_percentage(rer: f64) -> f64 {
        100.0 - Self::carb_percentage(rer)
    }
}

/// Gas exchange analyzer
#[derive(Debug, Clone)]
pub struct GasExchange {
    /// STPD correction factor (if needed)
    pub stpd_factor: f64,
    /// BTPS correction factor (if needed)
    pub btps_factor: f64,
}

impl Default for GasExchange {
    fn default() -> Self {
        Self {
            stpd_factor: 1.0,
            btps_factor: 1.0,
        }
    }
}

impl GasExchange {
    /// Create new gas exchange analyzer with correction factors
    pub fn with_corrections(stpd: f64, btps: f64) -> Self {
        Self {
            stpd_factor: stpd,
            btps_factor: btps,
        }
    }

    /// Calculate respiratory exchange ratio
    pub fn rer(&self, vo2_l_min: f64, vco2_l_min: f64) -> f64 {
        if vo2_l_min < 0.001 {
            return 1.0;
        }
        vco2_l_min / vo2_l_min
    }

    /// Calculate oxygen pulse (VO2/HR)
    pub fn oxygen_pulse(&self, vo2_ml_min: f64, hr: f64) -> f64 {
        if hr < 1.0 {
            return 0.0;
        }
        vo2_ml_min / hr
    }

    /// Calculate ventilatory efficiency (VE/VCO2 slope)
    pub fn ventilatory_efficiency(&self, ve: &[f64], vco2: &[f64]) -> f64 {
        if ve.len() != vco2.len() || ve.len() < 3 {
            return 0.0;
        }

        // Linear regression: VE = slope * VCO2 + intercept
        let n = ve.len() as f64;
        let sum_x: f64 = vco2.iter().sum();
        let sum_y: f64 = ve.iter().sum();
        let sum_xy: f64 = vco2.iter().zip(ve.iter()).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = vco2.iter().map(|x| x * x).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }

    /// Calculate breathing reserve
    pub fn breathing_reserve(&self, ve_max: f64, mvv: f64) -> f64 {
        if mvv < 1.0 {
            return 0.0;
        }
        ((mvv - ve_max) / mvv) * 100.0
    }

    /// Calculate dead space to tidal volume ratio (VD/VT)
    /// Using Bohr equation approximation
    pub fn dead_space_ratio(&self, paco2: f64, peco2: f64) -> f64 {
        if paco2 < 1.0 {
            return 0.0;
        }
        (paco2 - peco2) / paco2
    }

    /// Calculate oxygen extraction ratio
    pub fn o2_extraction(&self, cao2: f64, cvo2: f64, cao2_content: f64) -> f64 {
        if cao2_content < 0.01 {
            return 0.0;
        }
        (cao2 - cvo2) / cao2_content
    }

    /// Estimate energy expenditure from gas exchange (Weir equation)
    pub fn energy_expenditure(&self, vo2_l_min: f64, vco2_l_min: f64) -> f64 {
        // Weir equation: EE (kcal/min) = 3.941*VO2 + 1.106*VCO2
        3.941 * vo2_l_min + 1.106 * vco2_l_min
    }

    /// Estimate fat oxidation rate (g/min)
    pub fn fat_oxidation(&self, vo2_l_min: f64, vco2_l_min: f64) -> f64 {
        // Frayn equation
        1.67 * vo2_l_min - 1.67 * vco2_l_min
    }

    /// Estimate carbohydrate oxidation rate (g/min)
    pub fn carb_oxidation(&self, vo2_l_min: f64, vco2_l_min: f64) -> f64 {
        // Frayn equation
        4.55 * vco2_l_min - 3.21 * vo2_l_min
    }

    /// Calculate OUES (Oxygen Uptake Efficiency Slope)
    pub fn oues(&self, vo2: &[f64], ve: &[f64]) -> f64 {
        if vo2.len() != ve.len() || vo2.len() < 5 {
            return 0.0;
        }

        // OUES = a from VO2 = a * log10(VE) + b
        let log_ve: Vec<f64> = ve.iter().map(|v| v.log10()).collect();

        let n = vo2.len() as f64;
        let sum_x: f64 = log_ve.iter().sum();
        let sum_y: f64 = vo2.iter().sum();
        let sum_xy: f64 = log_ve.iter().zip(vo2.iter()).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = log_ve.iter().map(|x| x * x).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }

    /// Detect maximal effort from gas exchange criteria
    pub fn is_maximal_effort(
        &self,
        rer_max: f64,
        hr_max: f64,
        predicted_hr_max: f64,
        rpe: Option<u8>,
    ) -> MaximalEffortCriteria {
        let mut criteria_met = 0;

        let rer_criterion = rer_max >= 1.10;
        if rer_criterion {
            criteria_met += 1;
        }

        let hr_criterion = hr_max >= predicted_hr_max * 0.85;
        if hr_criterion {
            criteria_met += 1;
        }

        let rpe_criterion = rpe.map(|r| r >= 17).unwrap_or(false);
        if rpe_criterion {
            criteria_met += 1;
        }

        MaximalEffortCriteria {
            rer_criterion,
            hr_criterion,
            rpe_criterion,
            total_met: criteria_met,
            is_maximal: criteria_met >= 2,
        }
    }

    /// Generate comprehensive gas exchange metrics
    pub fn analyze(&self, data: &GasExchangeData) -> GasExchangeMetrics {
        let rer = self.rer(data.vo2_l_min, data.vco2_l_min);
        let rq_class = RespiratoryQuotient::from_rer(rer);
        let o2_pulse = self.oxygen_pulse(data.vo2_l_min * 1000.0, data.hr);

        let ve_vco2_slope = if !data.ve_series.is_empty() && !data.vco2_series.is_empty() {
            self.ventilatory_efficiency(&data.ve_series, &data.vco2_series)
        } else {
            data.ve / data.vco2_l_min
        };

        let breathing_reserve = self.breathing_reserve(data.ve, data.mvv);
        let energy_exp = self.energy_expenditure(data.vo2_l_min, data.vco2_l_min);
        let fat_ox = self.fat_oxidation(data.vo2_l_min, data.vco2_l_min);
        let carb_ox = self.carb_oxidation(data.vo2_l_min, data.vco2_l_min);

        GasExchangeMetrics {
            rer,
            rq_classification: rq_class,
            o2_pulse_ml_beat: o2_pulse,
            ve_vco2_slope,
            breathing_reserve_percent: breathing_reserve,
            energy_expenditure_kcal_min: energy_exp,
            fat_oxidation_g_min: fat_ox.max(0.0),
            carb_oxidation_g_min: carb_ox.max(0.0),
        }
    }
}

/// Input data for gas exchange analysis
#[derive(Debug, Clone)]
pub struct GasExchangeData {
    /// Oxygen uptake (L/min)
    pub vo2_l_min: f64,
    /// CO2 production (L/min)
    pub vco2_l_min: f64,
    /// Minute ventilation (L/min)
    pub ve: f64,
    /// Heart rate (bpm)
    pub hr: f64,
    /// Maximal voluntary ventilation (L/min)
    pub mvv: f64,
    /// VE time series (optional)
    pub ve_series: Vec<f64>,
    /// VCO2 time series (optional)
    pub vco2_series: Vec<f64>,
}

/// Comprehensive gas exchange metrics
#[derive(Debug, Clone)]
pub struct GasExchangeMetrics {
    /// Respiratory exchange ratio
    pub rer: f64,
    /// RQ classification
    pub rq_classification: RespiratoryQuotient,
    /// Oxygen pulse (ml/beat)
    pub o2_pulse_ml_beat: f64,
    /// VE/VCO2 slope
    pub ve_vco2_slope: f64,
    /// Breathing reserve (%)
    pub breathing_reserve_percent: f64,
    /// Energy expenditure (kcal/min)
    pub energy_expenditure_kcal_min: f64,
    /// Fat oxidation rate (g/min)
    pub fat_oxidation_g_min: f64,
    /// Carbohydrate oxidation rate (g/min)
    pub carb_oxidation_g_min: f64,
}

/// Maximal effort verification criteria
#[derive(Debug, Clone)]
pub struct MaximalEffortCriteria {
    /// RER ≥ 1.10
    pub rer_criterion: bool,
    /// HR ≥ 85% predicted max
    pub hr_criterion: bool,
    /// RPE ≥ 17
    pub rpe_criterion: bool,
    /// Total criteria met
    pub total_met: usize,
    /// Test considered maximal (≥2 criteria)
    pub is_maximal: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rer_calculation() {
        let ge = GasExchange::default();

        let rer = ge.rer(2.5, 2.25);
        assert!((rer - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_rq_classification() {
        assert_eq!(
            RespiratoryQuotient::from_rer(0.72),
            RespiratoryQuotient::FatOxidation
        );
        assert_eq!(
            RespiratoryQuotient::from_rer(0.95),
            RespiratoryQuotient::CarbOxidation
        );
        assert_eq!(
            RespiratoryQuotient::from_rer(1.15),
            RespiratoryQuotient::Anaerobic
        );
    }

    #[test]
    fn test_oxygen_pulse() {
        let ge = GasExchange::default();

        let o2_pulse = ge.oxygen_pulse(3000.0, 150.0);
        assert!((o2_pulse - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_energy_expenditure() {
        let ge = GasExchange::default();

        // At RER 0.85, typical substrate mix
        let ee = ge.energy_expenditure(2.0, 1.7);
        assert!(ee > 9.0 && ee < 10.0);
    }

    #[test]
    fn test_maximal_effort() {
        let ge = GasExchange::default();

        let criteria = ge.is_maximal_effort(1.15, 185.0, 190.0, Some(19));

        assert!(criteria.rer_criterion);
        assert!(criteria.hr_criterion);
        assert!(criteria.rpe_criterion);
        assert!(criteria.is_maximal);
    }

    #[test]
    fn test_substrate_percentages() {
        // At RER 0.85 (mixed)
        let carb = RespiratoryQuotient::carb_percentage(0.85);
        let fat = RespiratoryQuotient::fat_percentage(0.85);

        assert!((carb - 50.0).abs() < 1.0);
        assert!((fat - 50.0).abs() < 1.0);
    }
}
