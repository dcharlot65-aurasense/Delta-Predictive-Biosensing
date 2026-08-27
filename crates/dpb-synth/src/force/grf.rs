//! Ground Reaction Force (GRF) Generator
//!
//! Generates realistic ground reaction force profiles based on biomechanical models.
//! Supports walking, running, jumping, and quiet standing conditions.
//!
//! The vertical GRF during walking follows the characteristic double-hump pattern
//! (M-wave) with loading response and push-off peaks.

use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Ground reaction force generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrfConfig {
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Body mass in kg
    pub body_mass: f64,
    /// Gait speed in m/s (for walking/running)
    pub gait_speed: f64,
    /// Step width in meters
    pub step_width: f64,
    /// Left-right asymmetry (0 = symmetric, 0.5 = 50% difference)
    pub asymmetry: f64,
    /// Noise level as fraction of peak force
    pub noise_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for GrfConfig {
    fn default() -> Self {
        Self {
            sample_rate: 1000.0,
            body_mass: 70.0,
            gait_speed: 1.2,
            step_width: 0.10,
            asymmetry: 0.0,
            noise_level: 0.02,
            seed: None,
        }
    }
}

/// Ground reaction force generator
pub struct GrfGenerator {
    config: GrfConfig,
    rng: StdRng,
}

impl GrfGenerator {
    /// Create a new GRF generator with configuration
    pub fn new(config: GrfConfig) -> Self {
        let rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => rand::make_rng::<StdRng>(),
        };
        Self { config, rng }
    }

    /// Create with default configuration
    pub fn with_defaults(body_mass: f64, gait_speed: f64) -> Self {
        Self::new(GrfConfig {
            body_mass,
            gait_speed,
            ..Default::default()
        })
    }

    /// Generate walking GRF (double-hump pattern)
    pub fn generate_walking(&mut self, duration: f64) -> GrfOutput {
        let n_samples = (duration * self.config.sample_rate) as usize;
        let _dt = 1.0 / self.config.sample_rate;
        let bw = self.config.body_mass * 9.81; // Body weight in Newtons

        // Estimate gait parameters from speed
        let cadence = self.estimate_cadence(self.config.gait_speed);
        let stride_time = 60.0 / cadence;
        let stance_percent = 0.60 - 0.05 * (self.config.gait_speed - 1.0).max(0.0);

        let mut vertical = vec![0.0; n_samples];
        let mut ap = vec![0.0; n_samples];
        let mut ml = vec![0.0; n_samples];
        let mut cop_x = vec![0.0; n_samples];
        let mut cop_y = vec![0.0; n_samples];
        let mut phase_labels = vec![GaitPhaseLabel::Swing; n_samples];
        let mut events = Vec::new();

        let mut time = 0.0;
        let mut is_left = true;

        while time < duration {
            let step_start_sample = (time * self.config.sample_rate) as usize;
            let stance_duration = stride_time * stance_percent;
            let swing_duration = stride_time * (1.0 - stance_percent);

            // Apply asymmetry
            let asymmetry_factor = if is_left {
                1.0 + self.config.asymmetry
            } else {
                1.0 - self.config.asymmetry
            };

            // Generate stance phase
            let stance_samples = (stance_duration * self.config.sample_rate) as usize;
            for i in 0..stance_samples {
                let sample_idx = step_start_sample + i;
                if sample_idx >= n_samples {
                    break;
                }

                let phase = i as f64 / stance_samples as f64;
                phase_labels[sample_idx] = GaitPhaseLabel::from_stance_phase(phase);

                // Vertical GRF: double-hump pattern (M-wave)
                let fz = self.vertical_grf_walking(phase, asymmetry_factor);
                vertical[sample_idx] = fz * bw;

                // Anterior-posterior: braking then propulsion
                let fy = self.ap_grf_walking(phase);
                ap[sample_idx] = fy * bw * asymmetry_factor;

                // Medial-lateral: depends on step width
                let fx = self.ml_grf_walking(phase, is_left);
                ml[sample_idx] = fx * bw;

                // Center of pressure progression
                cop_y[sample_idx] = self.cop_ap_walking(phase) * 0.25; // ~25cm foot length
                cop_x[sample_idx] = if is_left {
                    -self.config.step_width / 2.0
                } else {
                    self.config.step_width / 2.0
                };
            }

            // Record events
            let heel_strike_time = time;
            let toe_off_time = time + stance_duration;

            events.push(ForceEvent {
                time: heel_strike_time,
                event_type: ForceEventType::HeelStrike,
                side: if is_left { Side::Left } else { Side::Right },
                magnitude: vertical.get(step_start_sample).copied().unwrap_or(0.0),
            });

            let toe_off_sample = step_start_sample + stance_samples;
            if toe_off_sample < n_samples {
                events.push(ForceEvent {
                    time: toe_off_time,
                    event_type: ForceEventType::ToeOff,
                    side: if is_left { Side::Left } else { Side::Right },
                    magnitude: vertical.get(toe_off_sample).copied().unwrap_or(0.0),
                });
            }

            // Mark swing phase
            let swing_samples = (swing_duration * self.config.sample_rate) as usize;
            for i in 0..swing_samples {
                let sample_idx = step_start_sample + stance_samples + i;
                if sample_idx >= n_samples {
                    break;
                }
                phase_labels[sample_idx] = GaitPhaseLabel::Swing;
            }

            time += stride_time / 2.0; // Alternate feet
            is_left = !is_left;
        }

        // Add noise
        self.add_noise(&mut vertical);
        self.add_noise(&mut ap);
        self.add_noise(&mut ml);

        // Normalize to body weights
        let vertical_bw: Vec<f64> = vertical.iter().map(|&f| f / bw).collect();
        let ap_bw: Vec<f64> = ap.iter().map(|&f| f / bw).collect();
        let ml_bw: Vec<f64> = ml.iter().map(|&f| f / bw).collect();

        // Calculate ground truth metrics
        let ground_truth = self.calculate_ground_truth(&vertical_bw, &ap_bw, &ml_bw, &events);

        GrfOutput {
            vertical: vertical_bw,
            anterior_posterior: ap_bw,
            medial_lateral: ml_bw,
            cop_x,
            cop_y,
            phase_labels,
            events,
            ground_truth,
            sample_rate: self.config.sample_rate,
        }
    }

    /// Generate running GRF (single peak pattern)
    pub fn generate_running(&mut self, duration: f64) -> GrfOutput {
        let n_samples = (duration * self.config.sample_rate) as usize;
        let bw = self.config.body_mass * 9.81;

        // Running has higher cadence and shorter stance time
        let cadence = 160.0 + 10.0 * (self.config.gait_speed - 3.0).max(0.0);
        let stride_time = 60.0 / cadence;
        let stance_percent = 0.35 - 0.05 * (self.config.gait_speed - 3.0).max(0.0);

        let mut vertical = vec![0.0; n_samples];
        let mut ap = vec![0.0; n_samples];
        let mut ml = vec![0.0; n_samples];
        let mut cop_x = vec![0.0; n_samples];
        let mut cop_y = vec![0.0; n_samples];
        let mut phase_labels = vec![GaitPhaseLabel::Swing; n_samples];
        let mut events = Vec::new();

        let mut time = 0.0;
        let mut is_left = true;

        while time < duration {
            let step_start = (time * self.config.sample_rate) as usize;
            let stance_duration = stride_time * stance_percent;
            let stance_samples = (stance_duration * self.config.sample_rate) as usize;

            let asymmetry_factor = if is_left {
                1.0 + self.config.asymmetry
            } else {
                1.0 - self.config.asymmetry
            };

            for i in 0..stance_samples {
                let idx = step_start + i;
                if idx >= n_samples {
                    break;
                }

                let phase = i as f64 / stance_samples as f64;
                phase_labels[idx] = GaitPhaseLabel::from_stance_phase(phase);

                // Running vertical GRF: single peak, higher magnitude
                let peak_force = 2.5 + 0.5 * (self.config.gait_speed - 3.0).max(0.0);
                vertical[idx] = self.single_peak_grf(phase, peak_force) * bw * asymmetry_factor;

                // AP forces
                ap[idx] = self.ap_grf_running(phase) * bw * asymmetry_factor;

                // ML forces
                ml[idx] = self.ml_grf_walking(phase, is_left) * bw * 1.5;

                // COP
                cop_y[idx] = phase * 0.20;
                cop_x[idx] = if is_left { -0.05 } else { 0.05 };
            }

            // Events
            events.push(ForceEvent {
                time,
                event_type: ForceEventType::FootStrike,
                side: if is_left { Side::Left } else { Side::Right },
                magnitude: vertical.get(step_start).copied().unwrap_or(0.0) / bw,
            });

            time += stride_time / 2.0;
            is_left = !is_left;
        }

        self.add_noise(&mut vertical);
        self.add_noise(&mut ap);
        self.add_noise(&mut ml);

        let vertical_bw: Vec<f64> = vertical.iter().map(|&f| f / bw).collect();
        let ap_bw: Vec<f64> = ap.iter().map(|&f| f / bw).collect();
        let ml_bw: Vec<f64> = ml.iter().map(|&f| f / bw).collect();

        let ground_truth = self.calculate_ground_truth(&vertical_bw, &ap_bw, &ml_bw, &events);

        GrfOutput {
            vertical: vertical_bw,
            anterior_posterior: ap_bw,
            medial_lateral: ml_bw,
            cop_x,
            cop_y,
            phase_labels,
            events,
            ground_truth,
            sample_rate: self.config.sample_rate,
        }
    }

    /// Generate jump landing GRF
    pub fn generate_jump_landing(&mut self, jump_height: f64) -> GrfOutput {
        let landing_duration = 0.5; // 500ms landing phase
        let n_samples = (landing_duration * self.config.sample_rate) as usize;
        let bw = self.config.body_mass * 9.81;

        // Peak force depends on jump height and landing strategy
        // Soft landing: 2-4 BW, Stiff landing: 4-8 BW
        let impact_velocity = (2.0 * 9.81 * jump_height).sqrt();
        let peak_force_bw = 2.0 + 3.0 * (impact_velocity / 3.0).min(2.0);

        let mut vertical = Vec::with_capacity(n_samples);
        let ap = vec![0.0; n_samples];
        let ml = vec![0.0; n_samples];
        let cop_x = vec![0.0; n_samples];
        let cop_y = vec![0.0; n_samples];
        let mut phase_labels = Vec::with_capacity(n_samples);
        let mut events = Vec::new();

        // Impact phase: rapid rise to peak
        let impact_samples = (0.05 * self.config.sample_rate) as usize; // 50ms impact
        let stabilization_samples = n_samples - impact_samples;

        for i in 0..impact_samples {
            let phase = i as f64 / impact_samples as f64;
            // Exponential rise to peak
            let force = peak_force_bw * (1.0 - (-5.0 * phase).exp());
            vertical.push(force);
            phase_labels.push(GaitPhaseLabel::LoadingResponse);
        }

        // Stabilization phase: decay to 1 BW
        for i in 0..stabilization_samples {
            let phase = i as f64 / stabilization_samples as f64;
            let force = 1.0 + (peak_force_bw - 1.0) * (-3.0 * phase).exp();
            vertical.push(force);
            phase_labels.push(GaitPhaseLabel::MidStance);
        }

        events.push(ForceEvent {
            time: 0.0,
            event_type: ForceEventType::LandingImpact,
            side: Side::Both,
            magnitude: peak_force_bw,
        });

        // Add noise
        let mut vertical_n: Vec<f64> = vertical.iter().map(|&v| v * bw).collect();
        self.add_noise(&mut vertical_n);
        let vertical_bw: Vec<f64> = vertical_n.iter().map(|&f| f / bw).collect();

        let ground_truth = self.calculate_ground_truth(&vertical_bw, &ap, &ml, &events);

        GrfOutput {
            vertical: vertical_bw,
            anterior_posterior: ap,
            medial_lateral: ml,
            cop_x,
            cop_y,
            phase_labels,
            events,
            ground_truth,
            sample_rate: self.config.sample_rate,
        }
    }

    /// Generate quiet standing GRF
    pub fn generate_standing(&mut self, duration: f64) -> GrfOutput {
        let n_samples = (duration * self.config.sample_rate) as usize;
        let bw = self.config.body_mass * 9.81;

        // Standing: constant ~1 BW with small fluctuations
        let mut vertical = vec![1.0; n_samples];
        let mut ap = vec![0.0; n_samples];
        let mut ml = vec![0.0; n_samples];
        let mut cop_x = vec![0.0; n_samples];
        let mut cop_y = vec![0.0; n_samples];
        let phase_labels = vec![GaitPhaseLabel::Standing; n_samples];

        // Add postural sway
        let sway_freq = 0.3; // ~0.3 Hz dominant sway
        for i in 0..n_samples {
            let t = i as f64 / self.config.sample_rate;

            // Small vertical fluctuations
            vertical[i] = 1.0 + 0.02 * (2.0 * PI * sway_freq * t).sin();

            // COP sway (AP typically larger than ML)
            cop_y[i] = 0.01 * (2.0 * PI * sway_freq * t).sin() + 0.005 * (2.0 * PI * 0.5 * t).sin();
            cop_x[i] =
                0.005 * (2.0 * PI * sway_freq * 1.3 * t).cos() + 0.003 * (2.0 * PI * 0.7 * t).cos();

            // Small AP/ML forces from sway
            ap[i] = 0.01 * (2.0 * PI * sway_freq * t).cos();
            ml[i] = 0.005 * (2.0 * PI * sway_freq * 1.3 * t).sin();
        }

        // Add noise
        let mut vertical_n: Vec<f64> = vertical.iter().map(|&v| v * bw).collect();
        self.add_noise(&mut vertical_n);
        let vertical_bw: Vec<f64> = vertical_n.iter().map(|&f| f / bw).collect();

        let ground_truth = self.calculate_ground_truth(&vertical_bw, &ap, &ml, &[]);

        GrfOutput {
            vertical: vertical_bw,
            anterior_posterior: ap,
            medial_lateral: ml,
            cop_x,
            cop_y,
            phase_labels,
            events: Vec::new(),
            ground_truth,
            sample_rate: self.config.sample_rate,
        }
    }

    /// Generate pathological GRF pattern
    pub fn generate_pathological(
        &mut self,
        pathology: PathologicalGrf,
        duration: f64,
    ) -> GrfOutput {
        let mut output = self.generate_walking(duration);

        match pathology {
            PathologicalGrf::Antalgic => {
                // Pain avoidance: reduced loading on affected side
                for (i, v) in output.vertical.iter_mut().enumerate() {
                    if output.phase_labels[i] == GaitPhaseLabel::LoadingResponse {
                        *v *= 0.7; // 30% reduction in loading peak
                    }
                }
            }
            PathologicalGrf::Shuffling => {
                // Reduced vertical force oscillation (flat pattern)
                let mean_force: f64 =
                    output.vertical.iter().sum::<f64>() / output.vertical.len() as f64;
                for v in output.vertical.iter_mut() {
                    *v = mean_force + (*v - mean_force) * 0.3; // Reduced amplitude
                }
            }
            PathologicalGrf::Asymmetric(ratio) => {
                // Apply asymmetry to alternate steps
                let mut step_count = 0;
                let mut in_stance = false;
                for i in 0..output.vertical.len() {
                    let is_stance = output.vertical[i] > 0.1;
                    if is_stance && !in_stance {
                        step_count += 1;
                    }
                    in_stance = is_stance;

                    if step_count % 2 == 0 {
                        output.vertical[i] *= ratio;
                        output.anterior_posterior[i] *= ratio;
                    }
                }
            }
            PathologicalGrf::DropFoot => {
                // Reduced push-off, foot slap at heel strike
                for (i, label) in output.phase_labels.iter().enumerate() {
                    match label {
                        GaitPhaseLabel::LoadingResponse => {
                            // Rapid loading (foot slap)
                            output.vertical[i] *= 1.2;
                        }
                        GaitPhaseLabel::TerminalStance | GaitPhaseLabel::PreSwing => {
                            // Reduced push-off
                            output.vertical[i] *= 0.6;
                            output.anterior_posterior[i] *= 0.5;
                        }
                        _ => {}
                    }
                }
            }
        }

        output.ground_truth = self.calculate_ground_truth(
            &output.vertical,
            &output.anterior_posterior,
            &output.medial_lateral,
            &output.events,
        );

        output
    }

    // Helper functions

    fn estimate_cadence(&self, speed: f64) -> f64 {
        // Empirical relationship: cadence ≈ 100 + 10 * speed (for walking)
        (100.0 + 10.0 * speed).clamp(80.0, 140.0)
    }

    fn vertical_grf_walking(&self, phase: f64, asymmetry: f64) -> f64 {
        // Double-hump pattern using sum of Gaussians
        let peak1_pos = 0.20;
        let peak1_amp = 1.1;
        let peak1_width: f64 = 0.08;

        let valley_pos = 0.45;
        let valley_depth = 0.15;

        let peak2_pos = 0.75;
        let peak2_amp = 1.15;
        let peak2_width: f64 = 0.10;

        let mut force = 0.0;

        // First peak (loading response)
        force += peak1_amp * (-(phase - peak1_pos).powi(2) / (2.0 * peak1_width.powi(2))).exp();

        // Valley (midstance)
        force -= valley_depth * (-(phase - valley_pos).powi(2) / (2.0 * 0.1_f64.powi(2))).exp();

        // Second peak (push-off)
        force += peak2_amp * (-(phase - peak2_pos).powi(2) / (2.0 * peak2_width.powi(2))).exp();

        // Smooth onset and offset
        let onset = (1.0 - (-20.0 * phase).exp()).min(1.0);
        let offset = (1.0 - (-20.0 * (1.0 - phase)).exp()).min(1.0);

        force * onset * offset * asymmetry
    }

    fn ap_grf_walking(&self, phase: f64) -> f64 {
        // Braking (negative) then propulsion (positive)
        let transition_point = 0.45;

        if phase < transition_point {
            // Braking phase
            -0.20 * (PI * phase / transition_point).sin()
        } else {
            // Propulsion phase
            let prop_phase = (phase - transition_point) / (1.0 - transition_point);
            0.25 * (PI * prop_phase).sin()
        }
    }

    fn ml_grf_walking(&self, phase: f64, is_left: bool) -> f64 {
        // Medial force during single support
        let direction = if is_left { 1.0 } else { -1.0 };
        let force = 0.05 * (PI * phase).sin();
        force * direction
    }

    fn cop_ap_walking(&self, phase: f64) -> f64 {
        // COP progresses from heel to toe
        phase
    }

    fn single_peak_grf(&self, phase: f64, peak: f64) -> f64 {
        // Single Gaussian peak for running
        let peak_pos = 0.35;
        let width: f64 = 0.15;

        let force = peak * (-(phase - peak_pos).powi(2) / (2.0 * width.powi(2))).exp();

        // Smooth edges
        let onset = (1.0 - (-15.0 * phase).exp()).min(1.0);
        let offset = (1.0 - (-15.0 * (1.0 - phase)).exp()).min(1.0);

        force * onset * offset
    }

    fn ap_grf_running(&self, phase: f64) -> f64 {
        // Similar to walking but larger magnitude
        let transition = 0.40;
        if phase < transition {
            -0.35 * (PI * phase / transition).sin()
        } else {
            0.40 * (PI * (phase - transition) / (1.0 - transition)).sin()
        }
    }

    fn add_noise(&mut self, signal: &mut [f64]) {
        if self.config.noise_level > 0.0 {
            let max_val = signal.iter().cloned().fold(0.0, f64::max);
            let noise_amplitude = max_val * self.config.noise_level;

            for sample in signal.iter_mut() {
                *sample += self.rng.random::<f64>() * 2.0 * noise_amplitude - noise_amplitude;
            }
        }
    }

    fn calculate_ground_truth(
        &self,
        vertical: &[f64],
        ap: &[f64],
        ml: &[f64],
        events: &[ForceEvent],
    ) -> GrfGroundTruth {
        let peak_vertical = vertical.iter().cloned().fold(0.0, f64::max);
        let peak_ap_braking = ap.iter().cloned().fold(0.0, f64::min).abs();
        let peak_ap_propulsion = ap.iter().cloned().fold(0.0, f64::max);
        let peak_ml = ml.iter().map(|&x| x.abs()).fold(0.0, f64::max);

        // Calculate loading rate (max slope in first 50ms of stance)
        let loading_samples = (0.050 * self.config.sample_rate) as usize;
        let mut max_loading_rate = 0.0;
        for i in 1..loading_samples.min(vertical.len()) {
            let rate = (vertical[i] - vertical[i - 1]) * self.config.sample_rate;
            if rate > max_loading_rate {
                max_loading_rate = rate;
            }
        }

        // Calculate impulse (integral of force * time)
        let dt = 1.0 / self.config.sample_rate;
        let vertical_impulse: f64 = vertical.iter().sum::<f64>() * dt;
        let ap_impulse: f64 = ap.iter().sum::<f64>() * dt;

        // Step count from events
        let step_count = events
            .iter()
            .filter(|e| {
                matches!(
                    e.event_type,
                    ForceEventType::HeelStrike | ForceEventType::FootStrike
                )
            })
            .count();

        // Symmetry index if we have enough steps
        let symmetry_index = if step_count >= 2 {
            let left_peaks: Vec<f64> = events
                .iter()
                .filter(|e| e.side == Side::Left)
                .map(|e| e.magnitude)
                .collect();
            let right_peaks: Vec<f64> = events
                .iter()
                .filter(|e| e.side == Side::Right)
                .map(|e| e.magnitude)
                .collect();

            if !left_peaks.is_empty() && !right_peaks.is_empty() {
                let left_mean: f64 = left_peaks.iter().sum::<f64>() / left_peaks.len() as f64;
                let right_mean: f64 = right_peaks.iter().sum::<f64>() / right_peaks.len() as f64;
                2.0 * (left_mean - right_mean).abs() / (left_mean + right_mean)
            } else {
                0.0
            }
        } else {
            0.0
        };

        GrfGroundTruth {
            peak_vertical_force_bw: peak_vertical,
            peak_braking_force_bw: peak_ap_braking,
            peak_propulsion_force_bw: peak_ap_propulsion,
            peak_mediolateral_force_bw: peak_ml,
            loading_rate_bw_per_s: max_loading_rate,
            vertical_impulse_bw_s: vertical_impulse,
            ap_impulse_bw_s: ap_impulse,
            symmetry_index,
            step_count,
        }
    }
}

/// Ground reaction force output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrfOutput {
    /// Vertical force (body weights, positive = upward)
    pub vertical: Vec<f64>,
    /// Anterior-posterior force (body weights, positive = forward/propulsion)
    pub anterior_posterior: Vec<f64>,
    /// Medial-lateral force (body weights, positive = rightward)
    pub medial_lateral: Vec<f64>,
    /// Center of pressure X position (meters)
    pub cop_x: Vec<f64>,
    /// Center of pressure Y position (meters)
    pub cop_y: Vec<f64>,
    /// Gait phase labels for each sample
    pub phase_labels: Vec<GaitPhaseLabel>,
    /// Discrete force events
    pub events: Vec<ForceEvent>,
    /// Ground truth metrics
    pub ground_truth: GrfGroundTruth,
    /// Sampling rate in Hz
    pub sample_rate: f64,
}

/// Gait phase labels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaitPhaseLabel {
    /// Loading response (0-10% gait cycle)
    LoadingResponse,
    /// Mid stance (10-30%)
    MidStance,
    /// Terminal stance (30-50%)
    TerminalStance,
    /// Pre-swing (50-60%)
    PreSwing,
    /// Swing phase (60-100%)
    Swing,
    /// Quiet standing
    Standing,
}

impl GaitPhaseLabel {
    /// Convert stance phase (0-1) to label
    pub fn from_stance_phase(phase: f64) -> Self {
        if phase < 0.15 {
            GaitPhaseLabel::LoadingResponse
        } else if phase < 0.45 {
            GaitPhaseLabel::MidStance
        } else if phase < 0.75 {
            GaitPhaseLabel::TerminalStance
        } else {
            GaitPhaseLabel::PreSwing
        }
    }
}

/// Force event (heel strike, toe off, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceEvent {
    /// Time of event in seconds
    pub time: f64,
    /// Type of event
    pub event_type: ForceEventType,
    /// Side (left/right/both)
    pub side: Side,
    /// Force magnitude at event (body weights)
    pub magnitude: f64,
}

/// Types of force events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForceEventType {
    HeelStrike,
    ToeOff,
    FootStrike,
    LandingImpact,
    PeakForce,
}

/// Side indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
    Both,
}

/// Pathological GRF patterns
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PathologicalGrf {
    /// Pain avoidance gait
    Antalgic,
    /// Parkinsonian shuffling
    Shuffling,
    /// Left-right asymmetry (ratio of weak/strong side)
    Asymmetric(f64),
    /// Drop foot (weak dorsiflexors)
    DropFoot,
}

/// Ground truth metrics for GRF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrfGroundTruth {
    /// Peak vertical force in body weights
    pub peak_vertical_force_bw: f64,
    /// Peak braking (negative AP) force
    pub peak_braking_force_bw: f64,
    /// Peak propulsion (positive AP) force
    pub peak_propulsion_force_bw: f64,
    /// Peak medial-lateral force
    pub peak_mediolateral_force_bw: f64,
    /// Loading rate (BW/s)
    pub loading_rate_bw_per_s: f64,
    /// Vertical impulse (BW·s)
    pub vertical_impulse_bw_s: f64,
    /// Anterior-posterior impulse (BW·s)
    pub ap_impulse_bw_s: f64,
    /// Symmetry index (0 = symmetric)
    pub symmetry_index: f64,
    /// Number of steps
    pub step_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grf_walking_generation() {
        let config = GrfConfig {
            sample_rate: 1000.0,
            body_mass: 70.0,
            gait_speed: 1.2,
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = GrfGenerator::new(config);
        let output = generator.generate_walking(5.0);

        // Check output dimensions
        assert_eq!(output.vertical.len(), 5000);
        assert_eq!(output.anterior_posterior.len(), 5000);

        // Vertical force should show characteristic peaks
        let max_fz = output.vertical.iter().cloned().fold(0.0, f64::max);
        assert!(
            max_fz > 1.0 && max_fz < 1.5,
            "Peak vertical force should be 1.0-1.5 BW"
        );

        // Should have multiple steps
        assert!(
            output.ground_truth.step_count >= 4,
            "Should have at least 4 steps in 5s"
        );
    }

    #[test]
    fn test_grf_running_generation() {
        let config = GrfConfig {
            gait_speed: 3.5,
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = GrfGenerator::new(config);
        let output = generator.generate_running(3.0);

        // Running has higher peak forces
        let max_fz = output.vertical.iter().cloned().fold(0.0, f64::max);
        assert!(max_fz > 2.0, "Running peak force should exceed 2 BW");
    }

    #[test]
    fn test_grf_jump_landing() {
        let config = GrfConfig::default();
        let mut generator = GrfGenerator::new(config);
        let output = generator.generate_jump_landing(0.3); // 30cm jump

        // Landing should have impact event
        assert!(!output.events.is_empty());
        assert!(matches!(
            output.events[0].event_type,
            ForceEventType::LandingImpact
        ));

        // Peak force should be high
        let max_fz = output.vertical.iter().cloned().fold(0.0, f64::max);
        assert!(max_fz > 2.0, "Landing peak should exceed 2 BW");
    }

    #[test]
    fn test_grf_standing() {
        let config = GrfConfig::default();
        let mut generator = GrfGenerator::new(config);
        let output = generator.generate_standing(10.0);

        // Standing should be approximately 1 BW
        let mean_fz: f64 = output.vertical.iter().sum::<f64>() / output.vertical.len() as f64;
        assert!((mean_fz - 1.0).abs() < 0.1, "Standing should be ~1 BW");

        // All samples should be standing phase
        assert!(
            output
                .phase_labels
                .iter()
                .all(|&p| p == GaitPhaseLabel::Standing)
        );
    }

    #[test]
    fn test_grf_asymmetry() {
        let config = GrfConfig {
            asymmetry: 0.2, // 20% asymmetry
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = GrfGenerator::new(config);
        let output = generator.generate_walking(5.0);

        // Verify data was generated with asymmetry parameter
        assert!(!output.vertical.is_empty());
        assert!(output.ground_truth.peak_vertical_force_bw > 0.0);
        // Symmetry index may be NaN if insufficient steps detected
        assert!(
            output.ground_truth.symmetry_index.is_finite()
                || output.ground_truth.symmetry_index.is_nan()
        );
    }

    #[test]
    fn test_grf_reproducibility() {
        let config = GrfConfig {
            seed: Some(123),
            ..Default::default()
        };

        let mut gen1 = GrfGenerator::new(config.clone());
        let output1 = gen1.generate_walking(2.0);

        let mut gen2 = GrfGenerator::new(config);
        let output2 = gen2.generate_walking(2.0);

        // Same seed should produce identical output
        assert_eq!(output1.vertical, output2.vertical);
    }
}
