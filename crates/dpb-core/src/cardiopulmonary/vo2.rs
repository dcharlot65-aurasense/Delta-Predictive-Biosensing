//! VO2 estimation algorithms
//!
//! Implements submaximal and maximal VO2 prediction from:
//! - Heart rate response to exercise
//! - Power output/workload
//! - Walk/run tests

/// Exercise protocol type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExerciseProtocol {
    /// Astrand-Rhyming submaximal cycle test
    AstrandCycle,
    /// YMCA submaximal cycle test
    YmcaCycle,
    /// Bruce treadmill protocol
    BruceTreadmill,
    /// Modified Bruce protocol
    ModifiedBruce,
    /// 6-minute walk test
    SixMinuteWalk,
    /// Cooper 12-minute run
    Cooper12Min,
    /// Rockport walk test
    RockportWalk,
    /// Queens College step test
    QueensCollegeStep,
}

/// VO2 estimator for various protocols
#[derive(Debug, Clone)]
pub struct Vo2Estimator {
    /// Subject age (years)
    pub age: u8,
    /// Subject sex (true = male)
    pub is_male: bool,
    /// Body weight (kg)
    pub weight_kg: f64,
    /// Height (cm)
    pub height_cm: f64,
}

impl Vo2Estimator {
    /// Create new VO2 estimator
    pub fn new(age: u8, is_male: bool, weight_kg: f64, height_cm: f64) -> Self {
        Self {
            age,
            is_male,
            weight_kg,
            height_cm,
        }
    }

    /// Estimate VO2max from Astrand-Rhyming cycle test
    /// power_watts: workload, hr: steady-state heart rate
    pub fn astrand_cycle(&self, power_watts: f64, hr: f64) -> Vo2Prediction {
        // Calculate VO2 from power output (ACSM metabolic equation)
        // VO2 (ml/min) = 1.8 * work rate (kpm/min) / body mass (kg) + 3.5 + 3.5
        // 1 watt = 6.12 kpm/min
        let vo2_submaximal = (1.8 * power_watts * 6.12 / self.weight_kg) + 7.0;

        // Convert to L/min
        let vo2_l_min = vo2_submaximal * self.weight_kg / 1000.0;

        // Estimate VO2max using HR
        let hr_max = self.predicted_hr_max();
        let correction = self.astrand_age_correction();

        // Astrand nomogram approximation
        let vo2max_uncorrected = if self.is_male {
            vo2_l_min * (hr_max / hr)
        } else {
            vo2_l_min * (hr_max / hr) * 0.9 // Sex correction
        };

        let vo2max = vo2max_uncorrected * correction;

        Vo2Prediction {
            vo2max_l_min: vo2max,
            vo2max_ml_kg_min: vo2max * 1000.0 / self.weight_kg,
            protocol: ExerciseProtocol::AstrandCycle,
            confidence: self.estimate_confidence(hr),
        }
    }

    /// Estimate VO2max from YMCA cycle test
    /// Uses two submaximal stages with HR 110-150
    pub fn ymca_cycle(&self, stages: &[(f64, f64)]) -> Vo2Prediction {
        if stages.len() < 2 {
            return Vo2Prediction::default();
        }

        // Find two consecutive stages with HR 110-150
        let valid_stages: Vec<(f64, f64)> = stages
            .iter()
            .filter(|(_, hr)| *hr >= 110.0 && *hr <= 150.0)
            .copied()
            .collect();

        if valid_stages.len() < 2 {
            return Vo2Prediction::default();
        }

        let (watts1, hr1) = valid_stages[0];
        let (watts2, hr2) = valid_stages[1];

        // Calculate VO2 for each stage
        let vo2_1 = self.watts_to_vo2(watts1);
        let vo2_2 = self.watts_to_vo2(watts2);

        // Linear extrapolation to HRmax
        let hr_max = self.predicted_hr_max();
        let slope = (vo2_2 - vo2_1) / (hr2 - hr1);
        let vo2max = vo2_1 + slope * (hr_max - hr1);

        Vo2Prediction {
            vo2max_l_min: vo2max,
            vo2max_ml_kg_min: vo2max * 1000.0 / self.weight_kg,
            protocol: ExerciseProtocol::YmcaCycle,
            confidence: 0.85,
        }
    }

    /// Estimate VO2max from Bruce treadmill test time
    pub fn bruce_treadmill(&self, time_minutes: f64) -> Vo2Prediction {
        // Foster et al. equations
        let vo2max = if self.is_male {
            14.8 - (1.379 * time_minutes)
                + (0.451 * time_minutes.powi(2))
                - (0.012 * time_minutes.powi(3))
        } else {
            4.38 * time_minutes - 3.9
        };

        Vo2Prediction {
            vo2max_l_min: vo2max * self.weight_kg / 1000.0,
            vo2max_ml_kg_min: vo2max,
            protocol: ExerciseProtocol::BruceTreadmill,
            confidence: 0.90,
        }
    }

    /// Estimate VO2max from 6-minute walk test
    pub fn six_minute_walk(&self, distance_meters: f64) -> Vo2Prediction {
        // Burr et al. equation
        let vo2max = (0.03 * distance_meters) + (0.023 * self.age as f64) - 0.33;

        // Alternative: Cahalin equation
        let vo2_alt = (0.02 * distance_meters) - (0.191 * self.age as f64)
            + (0.07 * self.weight_kg)
            + (0.09 * self.height_cm)
            + if self.is_male { 0.26 } else { 0.0 }
            + 5.3;

        let vo2_avg = (vo2max + vo2_alt) / 2.0;

        Vo2Prediction {
            vo2max_l_min: vo2_avg * self.weight_kg / 1000.0,
            vo2max_ml_kg_min: vo2_avg,
            protocol: ExerciseProtocol::SixMinuteWalk,
            confidence: 0.75,
        }
    }

    /// Estimate VO2max from Cooper 12-minute run
    pub fn cooper_12min(&self, distance_meters: f64) -> Vo2Prediction {
        // Cooper equation
        let vo2max = (distance_meters - 504.9) / 44.73;

        Vo2Prediction {
            vo2max_l_min: vo2max * self.weight_kg / 1000.0,
            vo2max_ml_kg_min: vo2max,
            protocol: ExerciseProtocol::Cooper12Min,
            confidence: 0.85,
        }
    }

    /// Estimate VO2max from Rockport 1-mile walk test
    pub fn rockport_walk(&self, time_minutes: f64, hr_finish: f64) -> Vo2Prediction {
        let sex = if self.is_male { 1.0 } else { 0.0 };

        // Rockport equation
        let vo2max = 132.853 - (0.0769 * self.weight_kg * 2.205) // Convert to lbs
            - (0.3877 * self.age as f64)
            + (6.315 * sex)
            - (3.2649 * time_minutes)
            - (0.1565 * hr_finish);

        Vo2Prediction {
            vo2max_l_min: vo2max * self.weight_kg / 1000.0,
            vo2max_ml_kg_min: vo2max,
            protocol: ExerciseProtocol::RockportWalk,
            confidence: 0.80,
        }
    }

    /// Estimate VO2max from Queens College step test
    pub fn queens_college_step(&self, hr_recovery: f64) -> Vo2Prediction {
        // McArdle equation
        let vo2max = if self.is_male {
            111.33 - (0.42 * hr_recovery)
        } else {
            65.81 - (0.1847 * hr_recovery)
        };

        Vo2Prediction {
            vo2max_l_min: vo2max * self.weight_kg / 1000.0,
            vo2max_ml_kg_min: vo2max,
            protocol: ExerciseProtocol::QueensCollegeStep,
            confidence: 0.80,
        }
    }

    /// Calculate predicted HRmax using Tanaka formula
    pub fn predicted_hr_max(&self) -> f64 {
        208.0 - (0.7 * self.age as f64)
    }

    /// Get Astrand age correction factor
    fn astrand_age_correction(&self) -> f64 {
        match self.age {
            15..=24 => 1.10,
            25..=34 => 1.00,
            35..=44 => 0.87,
            45..=54 => 0.78,
            55..=64 => 0.71,
            _ => 0.65,
        }
    }

    /// Convert watts to VO2 (L/min) using ACSM equation
    fn watts_to_vo2(&self, watts: f64) -> f64 {
        // ACSM leg ergometer equation
        let kpm_min = watts * 6.12;
        let vo2_ml_min = (1.8 * kpm_min / self.weight_kg) + 7.0;
        vo2_ml_min * self.weight_kg / 1000.0
    }

    /// Estimate confidence based on HR response
    fn estimate_confidence(&self, hr: f64) -> f64 {
        // Higher confidence when HR is in target zone (120-170)
        if hr >= 120.0 && hr <= 170.0 {
            0.90
        } else if hr >= 110.0 && hr <= 180.0 {
            0.80
        } else {
            0.70
        }
    }

    /// Calculate fitness percentile based on age and sex
    pub fn fitness_percentile(&self, vo2max: f64) -> f64 {
        // Normative data (approximate)
        let (mean, sd) = if self.is_male {
            match self.age {
                20..=29 => (44.0, 8.0),
                30..=39 => (42.0, 7.0),
                40..=49 => (40.0, 7.0),
                50..=59 => (36.0, 7.0),
                60..=69 => (32.0, 6.0),
                _ => (28.0, 6.0),
            }
        } else {
            match self.age {
                20..=29 => (36.0, 7.0),
                30..=39 => (34.0, 6.0),
                40..=49 => (32.0, 6.0),
                50..=59 => (28.0, 6.0),
                60..=69 => (24.0, 5.0),
                _ => (22.0, 5.0),
            }
        };

        let z = (vo2max - mean) / sd;
        normal_cdf(z) * 100.0
    }

    /// Classify cardiorespiratory fitness level
    pub fn fitness_classification(&self, vo2max: f64) -> FitnessLevel {
        let percentile = self.fitness_percentile(vo2max);

        match percentile as u32 {
            0..=19 => FitnessLevel::VeryPoor,
            20..=39 => FitnessLevel::Poor,
            40..=59 => FitnessLevel::Fair,
            60..=79 => FitnessLevel::Good,
            80..=94 => FitnessLevel::Excellent,
            _ => FitnessLevel::Superior,
        }
    }

    /// Analyze complete VO2 test results
    pub fn analyze(&self, prediction: &Vo2Prediction) -> Vo2Metrics {
        let percentile = self.fitness_percentile(prediction.vo2max_ml_kg_min);
        let classification = self.fitness_classification(prediction.vo2max_ml_kg_min);

        // Calculate MET capacity
        let met_capacity = prediction.vo2max_ml_kg_min / 3.5;

        Vo2Metrics {
            vo2max_l_min: prediction.vo2max_l_min,
            vo2max_ml_kg_min: prediction.vo2max_ml_kg_min,
            met_capacity,
            percentile,
            classification,
        }
    }
}

/// VO2max prediction result
#[derive(Debug, Clone)]
pub struct Vo2Prediction {
    /// VO2max in L/min
    pub vo2max_l_min: f64,
    /// VO2max in ml/kg/min
    pub vo2max_ml_kg_min: f64,
    /// Protocol used
    pub protocol: ExerciseProtocol,
    /// Confidence level (0-1)
    pub confidence: f64,
}

impl Default for Vo2Prediction {
    fn default() -> Self {
        Self {
            vo2max_l_min: 0.0,
            vo2max_ml_kg_min: 0.0,
            protocol: ExerciseProtocol::AstrandCycle,
            confidence: 0.0,
        }
    }
}

/// Comprehensive VO2 metrics
#[derive(Debug, Clone)]
pub struct Vo2Metrics {
    /// VO2max in L/min
    pub vo2max_l_min: f64,
    /// VO2max in ml/kg/min
    pub vo2max_ml_kg_min: f64,
    /// MET capacity
    pub met_capacity: f64,
    /// Age/sex percentile
    pub percentile: f64,
    /// Fitness classification
    pub classification: FitnessLevel,
}

/// Cardiorespiratory fitness level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitnessLevel {
    VeryPoor,
    Poor,
    Fair,
    Good,
    Excellent,
    Superior,
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_astrand_cycle() {
        let estimator = Vo2Estimator::new(30, true, 75.0, 175.0);

        let prediction = estimator.astrand_cycle(100.0, 140.0);

        // Should give reasonable VO2max for healthy male
        assert!(prediction.vo2max_ml_kg_min > 25.0);
        assert!(prediction.vo2max_ml_kg_min < 60.0);
    }

    #[test]
    fn test_cooper_12min() {
        let estimator = Vo2Estimator::new(25, true, 70.0, 180.0);

        // Good runner covering 2800m
        let prediction = estimator.cooper_12min(2800.0);

        // Should indicate good fitness
        assert!(prediction.vo2max_ml_kg_min > 50.0);
    }

    #[test]
    fn test_fitness_classification() {
        let estimator = Vo2Estimator::new(30, true, 75.0, 175.0);

        assert_eq!(estimator.fitness_classification(50.0), FitnessLevel::Good);
        assert_eq!(estimator.fitness_classification(30.0), FitnessLevel::Poor);
    }

    #[test]
    fn test_predicted_hr_max() {
        let estimator = Vo2Estimator::new(40, true, 80.0, 180.0);

        let hr_max = estimator.predicted_hr_max();
        assert!((hr_max - 180.0).abs() < 1.0);
    }
}
