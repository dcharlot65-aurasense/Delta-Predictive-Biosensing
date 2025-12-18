//! Ground reaction force (GRF) analysis algorithms
//!
//! This module provides tools for analyzing ground reaction forces from force plates
//! during walking, running, and jumping activities.

/// Ground reaction force data from a force plate
///
/// Contains the three components of GRF and temporal information.
///
/// # Example
///
/// ```
/// use dpb_core::biomechanics::force::GroundReactionForce;
///
/// let vertical = vec![0.0, 500.0, 1000.0, 500.0, 0.0];
/// let ap = vec![0.0, 50.0, 0.0, -50.0, 0.0];
/// let ml = vec![0.0, 10.0, 5.0, -5.0, 0.0];
/// let grf = GroundReactionForce::new(vertical, ap, ml, 1000.0);
/// let metrics = grf.analyze(70.0); // 70 kg body mass
/// ```
#[derive(Debug, Clone)]
pub struct GroundReactionForce {
    /// Vertical force component (N)
    pub vertical: Vec<f64>,
    /// Anterior-posterior force component (N)
    pub anterior_posterior: Vec<f64>,
    /// Medial-lateral force component (N)
    pub medial_lateral: Vec<f64>,
    /// Sample rate in Hz
    pub sample_rate: f64,
}

/// Metrics derived from ground reaction force analysis
#[derive(Debug, Clone, Default)]
pub struct GrfMetrics {
    /// Peak vertical force in body weights
    pub peak_vertical: f64,
    /// Loading rate in body weights per second
    pub loading_rate: f64,
    /// Impulse in Newton-seconds
    pub impulse: f64,
    /// Contact time in seconds
    pub contact_time: f64,
    /// Time to reach peak force in seconds
    pub time_to_peak: f64,
}

impl GroundReactionForce {
    /// Create a new ground reaction force data structure
    ///
    /// # Arguments
    ///
    /// * `vertical` - Vertical force component in Newtons
    /// * `ap` - Anterior-posterior force component in Newtons
    /// * `ml` - Medial-lateral force component in Newtons
    /// * `sample_rate` - Sampling frequency in Hz
    pub fn new(vertical: Vec<f64>, ap: Vec<f64>, ml: Vec<f64>, sample_rate: f64) -> Self {
        Self {
            vertical,
            anterior_posterior: ap,
            medial_lateral: ml,
            sample_rate,
        }
    }

    /// Analyze ground reaction force data to extract key metrics
    ///
    /// # Arguments
    ///
    /// * `body_mass` - Body mass in kilograms for normalization
    ///
    /// # Returns
    ///
    /// Comprehensive GRF metrics
    pub fn analyze(&self, body_mass: f64) -> GrfMetrics {
        let peak_vertical = self.peak_vertical_force(body_mass);
        let loading_rate = self.loading_rate(body_mass);
        let impulse = self.impulse();
        let contact_time = self.contact_time();
        let time_to_peak = self.time_to_peak();

        GrfMetrics {
            peak_vertical,
            loading_rate,
            impulse,
            contact_time,
            time_to_peak,
        }
    }

    /// Calculate peak vertical force normalized to body weight
    ///
    /// # Arguments
    ///
    /// * `body_mass` - Body mass in kilograms
    ///
    /// # Returns
    ///
    /// Peak vertical force in body weights (BW)
    pub fn peak_vertical_force(&self, body_mass: f64) -> f64 {
        if self.vertical.is_empty() || body_mass == 0.0 {
            return 0.0;
        }

        let body_weight = body_mass * 9.81; // Convert kg to Newtons
        let peak_force = self.vertical
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        peak_force / body_weight
    }

    /// Calculate vertical loading rate
    ///
    /// Loading rate is the maximum rate of force application during initial contact.
    ///
    /// # Arguments
    ///
    /// * `body_mass` - Body mass in kilograms
    ///
    /// # Returns
    ///
    /// Maximum loading rate in body weights per second
    pub fn loading_rate(&self, body_mass: f64) -> f64 {
        if self.vertical.len() < 2 || body_mass == 0.0 {
            return 0.0;
        }

        let body_weight = body_mass * 9.81;
        let dt = 1.0 / self.sample_rate;
        let mut max_rate = 0.0;

        for window in self.vertical.windows(2) {
            let rate = (window[1] - window[0]) / dt / body_weight;
            max_rate = max_rate.max(rate);
        }

        max_rate
    }

    /// Calculate impulse (integral of force over time)
    ///
    /// # Returns
    ///
    /// Impulse in Newton-seconds
    pub fn impulse(&self) -> f64 {
        if self.vertical.is_empty() {
            return 0.0;
        }

        let dt = 1.0 / self.sample_rate;
        self.vertical.iter().sum::<f64>() * dt
    }

    /// Calculate contact time (time above threshold)
    ///
    /// Contact time is estimated as the duration where vertical force
    /// exceeds 10% of peak force.
    ///
    /// # Returns
    ///
    /// Contact time in seconds
    fn contact_time(&self) -> f64 {
        if self.vertical.is_empty() {
            return 0.0;
        }

        let peak = self.vertical
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let threshold = peak * 0.1;

        let contact_samples = self.vertical
            .iter()
            .filter(|&&f| f > threshold)
            .count();

        contact_samples as f64 / self.sample_rate
    }

    /// Calculate time to reach peak force
    ///
    /// # Returns
    ///
    /// Time to peak in seconds
    fn time_to_peak(&self) -> f64 {
        if self.vertical.is_empty() {
            return 0.0;
        }

        let peak = self.vertical
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let peak_idx = self.vertical
            .iter()
            .position(|&f| f == peak)
            .unwrap_or(0);

        peak_idx as f64 / self.sample_rate
    }

    /// Calculate symmetry index between left and right limbs
    ///
    /// Symmetry index = 100 * |left - right| / (0.5 * (left + right))
    ///
    /// # Arguments
    ///
    /// * `left` - GRF data from left limb
    /// * `right` - GRF data from right limb
    ///
    /// # Returns
    ///
    /// Symmetry index percentage (0 = perfect symmetry)
    pub fn symmetry_index(left: &Self, right: &Self) -> f64 {
        if left.vertical.is_empty() || right.vertical.is_empty() {
            return 0.0;
        }

        let left_peak = left.vertical
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let right_peak = right.vertical
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let mean = (left_peak + right_peak) / 2.0;
        if mean == 0.0 {
            return 0.0;
        }

        100.0 * (left_peak - right_peak).abs() / mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_vertical_force() {
        let vertical = vec![0.0, 500.0, 1000.0, 500.0, 0.0];
        let grf = GroundReactionForce::new(
            vertical,
            vec![0.0; 5],
            vec![0.0; 5],
            1000.0,
        );

        let body_mass = 70.0; // kg
        let peak_bw = grf.peak_vertical_force(body_mass);

        // Peak force = 1000 N, body weight = 70 * 9.81 = 686.7 N
        // Peak in BW ≈ 1.46
        assert!((peak_bw - 1.456).abs() < 0.01);
    }

    #[test]
    fn test_impulse() {
        let vertical = vec![100.0, 200.0, 300.0, 200.0, 100.0];
        let grf = GroundReactionForce::new(
            vertical,
            vec![0.0; 5],
            vec![0.0; 5],
            1000.0,
        );

        let impulse = grf.impulse();
        // Sum = 900, dt = 0.001, impulse = 0.9
        assert!((impulse - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_symmetry_index() {
        let left = GroundReactionForce::new(
            vec![0.0, 100.0, 0.0],
            vec![0.0; 3],
            vec![0.0; 3],
            1000.0,
        );

        let right = GroundReactionForce::new(
            vec![0.0, 80.0, 0.0],
            vec![0.0; 3],
            vec![0.0; 3],
            1000.0,
        );

        let symmetry = GroundReactionForce::symmetry_index(&left, &right);
        // |100-80| / 90 * 100 = 22.22
        assert!((symmetry - 22.22).abs() < 0.1);
    }

    #[test]
    fn test_contact_time() {
        let vertical = vec![0.0, 100.0, 100.0, 100.0, 0.0];
        let grf = GroundReactionForce::new(
            vertical,
            vec![0.0; 5],
            vec![0.0; 5],
            1000.0, // 1000 Hz
        );

        let contact_time = grf.contact_time();
        // 3 samples above threshold at 1000 Hz = 0.003 s
        assert!((contact_time - 0.003).abs() < 0.0001);
    }
}
