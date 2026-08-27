//! Clinical pain assessment scales
//!
//! Implements standardized pain questionnaires and scales:
//! - Visual Analog Scale (VAS)
//! - Numeric Rating Scale (NRS)
//! - McGill Pain Questionnaire
//! - Brief Pain Inventory
//! - Neuropathic Pain Scale

/// Pain scale trait for all assessment instruments
pub trait PainScale {
    /// Calculate total score
    fn total_score(&self) -> f64;

    /// Get severity classification
    fn severity(&self) -> PainSeverity;

    /// Check if score indicates clinically significant pain
    fn is_clinically_significant(&self) -> bool;
}

/// Pain severity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PainSeverity {
    /// No pain
    None,
    /// Mild pain
    Mild,
    /// Moderate pain
    Moderate,
    /// Severe pain
    Severe,
}

/// Visual Analog Scale score (0-100mm)
#[derive(Debug, Clone)]
pub struct VasScore {
    /// Score in millimeters (0-100)
    pub score_mm: f64,
    /// Pain dimension being measured
    pub dimension: PainDimension,
}

/// Dimension of pain being assessed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PainDimension {
    /// Current pain intensity
    CurrentIntensity,
    /// Average pain over time period
    AverageIntensity,
    /// Worst pain experienced
    WorstIntensity,
    /// Pain unpleasantness (affective)
    Unpleasantness,
}

impl VasScore {
    /// Create new VAS score
    pub fn new(score_mm: f64, dimension: PainDimension) -> Self {
        Self {
            score_mm: score_mm.clamp(0.0, 100.0),
            dimension,
        }
    }

    /// Convert to 0-10 numeric scale
    pub fn to_nrs(&self) -> f64 {
        self.score_mm / 10.0
    }
}

impl PainScale for VasScore {
    fn total_score(&self) -> f64 {
        self.score_mm
    }

    fn severity(&self) -> PainSeverity {
        match self.score_mm as u32 {
            0 => PainSeverity::None,
            1..=30 => PainSeverity::Mild,
            31..=69 => PainSeverity::Moderate,
            _ => PainSeverity::Severe,
        }
    }

    fn is_clinically_significant(&self) -> bool {
        self.score_mm >= 30.0
    }
}

/// Brief Pain Inventory (BPI) assessment
#[derive(Debug, Clone)]
pub struct BriefPainInventory {
    /// Worst pain in past 24 hours (0-10)
    pub worst_pain: u8,
    /// Least pain in past 24 hours (0-10)
    pub least_pain: u8,
    /// Average pain (0-10)
    pub average_pain: u8,
    /// Current pain (0-10)
    pub current_pain: u8,
    /// Interference items (0-10 each)
    pub interference: BpiInterference,
}

/// BPI interference subscale
#[derive(Debug, Clone)]
pub struct BpiInterference {
    /// General activity
    pub general_activity: u8,
    /// Mood
    pub mood: u8,
    /// Walking ability
    pub walking: u8,
    /// Normal work
    pub work: u8,
    /// Relations with others
    pub relations: u8,
    /// Sleep
    pub sleep: u8,
    /// Enjoyment of life
    pub enjoyment: u8,
}

impl BriefPainInventory {
    /// Calculate pain severity score (mean of 4 severity items)
    pub fn severity_score(&self) -> f64 {
        (self.worst_pain as f64
            + self.least_pain as f64
            + self.average_pain as f64
            + self.current_pain as f64)
            / 4.0
    }

    /// Calculate interference score (mean of 7 interference items)
    pub fn interference_score(&self) -> f64 {
        let sum = self.interference.general_activity as f64
            + self.interference.mood as f64
            + self.interference.walking as f64
            + self.interference.work as f64
            + self.interference.relations as f64
            + self.interference.sleep as f64
            + self.interference.enjoyment as f64;
        sum / 7.0
    }
}

impl PainScale for BriefPainInventory {
    fn total_score(&self) -> f64 {
        (self.severity_score() + self.interference_score()) / 2.0
    }

    fn severity(&self) -> PainSeverity {
        match self.severity_score() as u32 {
            0 => PainSeverity::None,
            1..=3 => PainSeverity::Mild,
            4..=6 => PainSeverity::Moderate,
            _ => PainSeverity::Severe,
        }
    }

    fn is_clinically_significant(&self) -> bool {
        self.severity_score() >= 4.0 || self.interference_score() >= 4.0
    }
}

/// McGill Pain Questionnaire (Short Form)
#[derive(Debug, Clone)]
pub struct McGillPainQuestionnaire {
    /// Sensory descriptors (0-3 each)
    pub sensory: McGillSensory,
    /// Affective descriptors (0-3 each)
    pub affective: McGillAffective,
    /// Present Pain Index (0-5)
    pub present_pain_index: u8,
    /// VAS score (0-100)
    pub vas: f64,
}

/// McGill sensory subscale descriptors.
///
/// The eleven sensory items of the Short-Form McGill Pain Questionnaire.
/// Each is rated 0 = none, 1 = mild, 2 = moderate, 3 = severe, so
/// [`McGillSensory::score`] ranges 0-33.
#[derive(Debug, Clone)]
pub struct McGillSensory {
    /// Pulsing or beating quality, rated 0-3.
    pub throbbing: u8,
    /// Pain that travels along a path, rated 0-3.
    pub shooting: u8,
    /// Piercing, penetrating quality, rated 0-3.
    pub stabbing: u8,
    /// Keen, cutting quality, rated 0-3.
    pub sharp: u8,
    /// Constricting, clenching quality, rated 0-3.
    pub cramping: u8,
    /// Persistent boring or grinding quality, rated 0-3.
    pub gnawing: u8,
    /// Burning or scalding quality, rated 0-3.
    pub hot_burning: u8,
    /// Dull, continuous quality, rated 0-3.
    pub aching: u8,
    /// Sensation of weight or pressure, rated 0-3.
    pub heavy: u8,
    /// Pain on touch or palpation, rated 0-3.
    pub tender: u8,
    /// Bursting or rending quality, rated 0-3.
    pub splitting: u8,
}

impl McGillSensory {
    /// Calculate sensory subscale score
    pub fn score(&self) -> u8 {
        self.throbbing
            + self.shooting
            + self.stabbing
            + self.sharp
            + self.cramping
            + self.gnawing
            + self.hot_burning
            + self.aching
            + self.heavy
            + self.tender
            + self.splitting
    }
}

/// McGill affective subscale descriptors.
///
/// The four affective items of the Short-Form McGill Pain Questionnaire,
/// capturing the emotional rather than sensory dimension. Each is rated
/// 0 = none, 1 = mild, 2 = moderate, 3 = severe, so
/// [`McGillAffective::score`] ranges 0-12.
#[derive(Debug, Clone)]
pub struct McGillAffective {
    /// Draining, wearing quality, rated 0-3.
    pub tiring_exhausting: u8,
    /// Nausea-inducing quality, rated 0-3.
    pub sickening: u8,
    /// Frightening quality, rated 0-3.
    pub fearful: u8,
    /// Quality experienced as punitive or cruel, rated 0-3.
    pub punishing_cruel: u8,
}

impl McGillAffective {
    /// Calculate affective subscale score
    pub fn score(&self) -> u8 {
        self.tiring_exhausting + self.sickening + self.fearful + self.punishing_cruel
    }
}

impl McGillPainQuestionnaire {
    /// Calculate total Pain Rating Index
    pub fn pain_rating_index(&self) -> u8 {
        self.sensory.score() + self.affective.score()
    }

    /// Get sensory ratio (sensory/total)
    pub fn sensory_ratio(&self) -> f64 {
        let total = self.pain_rating_index();
        if total == 0 {
            return 0.0;
        }
        self.sensory.score() as f64 / total as f64
    }

    /// Get affective ratio (affective/total)
    pub fn affective_ratio(&self) -> f64 {
        let total = self.pain_rating_index();
        if total == 0 {
            return 0.0;
        }
        self.affective.score() as f64 / total as f64
    }
}

impl PainScale for McGillPainQuestionnaire {
    fn total_score(&self) -> f64 {
        self.pain_rating_index() as f64
    }

    fn severity(&self) -> PainSeverity {
        match self.present_pain_index {
            0 => PainSeverity::None,
            1..=2 => PainSeverity::Mild,
            3..=4 => PainSeverity::Moderate,
            _ => PainSeverity::Severe,
        }
    }

    fn is_clinically_significant(&self) -> bool {
        self.pain_rating_index() >= 10 || self.present_pain_index >= 3
    }
}

/// Neuropathic Pain Scale (NPS)
#[derive(Debug, Clone)]
pub struct NeuropathicPainScale {
    /// Intense (0-10)
    pub intense: u8,
    /// Sharp (0-10)
    pub sharp: u8,
    /// Hot (0-10)
    pub hot: u8,
    /// Dull (0-10)
    pub dull: u8,
    /// Cold (0-10)
    pub cold: u8,
    /// Sensitive (0-10)
    pub sensitive: u8,
    /// Itchy (0-10)
    pub itchy: u8,
    /// Surface vs deep (-10 to +10, converted to 0-10)
    pub surface_deep: i8,
    /// Unpleasant (0-10)
    pub unpleasant: u8,
    /// Overall intensity (0-10)
    pub overall: u8,
}

impl NeuropathicPainScale {
    /// Calculate NPS composite score
    pub fn composite_score(&self) -> f64 {
        let sum = self.intense as f64
            + self.sharp as f64
            + self.hot as f64
            + self.dull as f64
            + self.cold as f64
            + self.sensitive as f64
            + self.itchy as f64
            + (self.surface_deep.abs() as f64)
            + self.unpleasant as f64
            + self.overall as f64;
        sum / 10.0
    }

    /// Check for neuropathic pain characteristics
    pub fn is_neuropathic_pattern(&self) -> bool {
        // High sharp, hot, or sensitive scores suggest neuropathic pain
        self.sharp >= 5 || self.hot >= 5 || self.sensitive >= 5 || self.itchy >= 4
    }

    /// Calculate pain quality score
    pub fn pain_quality_score(&self) -> f64 {
        // Mean of quality descriptors (excluding intensity)
        (self.sharp as f64
            + self.hot as f64
            + self.dull as f64
            + self.cold as f64
            + self.sensitive as f64
            + self.itchy as f64)
            / 6.0
    }
}

impl PainScale for NeuropathicPainScale {
    fn total_score(&self) -> f64 {
        self.composite_score()
    }

    fn severity(&self) -> PainSeverity {
        match self.overall {
            0 => PainSeverity::None,
            1..=3 => PainSeverity::Mild,
            4..=6 => PainSeverity::Moderate,
            _ => PainSeverity::Severe,
        }
    }

    fn is_clinically_significant(&self) -> bool {
        self.composite_score() >= 4.0
    }
}

/// Pain Catastrophizing Scale (PCS)
#[derive(Debug, Clone)]
pub struct PainCatastrophizing {
    /// Rumination subscale items (0-4 each)
    pub rumination: [u8; 4],
    /// Magnification subscale items (0-4 each)
    pub magnification: [u8; 3],
    /// Helplessness subscale items (0-4 each)
    pub helplessness: [u8; 6],
}

impl PainCatastrophizing {
    /// Calculate rumination subscale score
    pub fn rumination_score(&self) -> u8 {
        self.rumination.iter().sum()
    }

    /// Calculate magnification subscale score
    pub fn magnification_score(&self) -> u8 {
        self.magnification.iter().sum()
    }

    /// Calculate helplessness subscale score
    pub fn helplessness_score(&self) -> u8 {
        self.helplessness.iter().sum()
    }

    /// Check for clinically elevated catastrophizing
    pub fn is_elevated(&self) -> bool {
        self.total_score() >= 30.0
    }

    /// Get catastrophizing level
    pub fn level(&self) -> CatastrophizingLevel {
        let score = self.total_score() as u32;
        match score {
            0..=9 => CatastrophizingLevel::Low,
            10..=19 => CatastrophizingLevel::Subclinical,
            20..=29 => CatastrophizingLevel::Moderate,
            _ => CatastrophizingLevel::Clinical,
        }
    }
}

impl PainScale for PainCatastrophizing {
    fn total_score(&self) -> f64 {
        (self.rumination_score() + self.magnification_score() + self.helplessness_score()) as f64
    }

    fn severity(&self) -> PainSeverity {
        match self.level() {
            CatastrophizingLevel::Low => PainSeverity::None,
            CatastrophizingLevel::Subclinical => PainSeverity::Mild,
            CatastrophizingLevel::Moderate => PainSeverity::Moderate,
            CatastrophizingLevel::Clinical => PainSeverity::Severe,
        }
    }

    fn is_clinically_significant(&self) -> bool {
        self.is_elevated()
    }
}

/// Catastrophizing level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatastrophizingLevel {
    /// Low catastrophizing
    Low,
    /// Subclinical
    Subclinical,
    /// Moderate
    Moderate,
    /// Clinically significant
    Clinical,
}

/// Calculate Minimal Clinically Important Difference (MCID)
pub fn mcid_reached(baseline: f64, current: f64, scale: McidScale) -> bool {
    let change = baseline - current;
    let threshold = match scale {
        McidScale::Vas100 => 20.0,
        McidScale::Nrs11 => 2.0,
        McidScale::BpiSeverity => 1.0,
        McidScale::BpiInterference => 1.0,
    };
    change >= threshold
}

/// Scales with defined MCID values
#[derive(Debug, Clone, Copy)]
pub enum McidScale {
    /// VAS 0-100mm
    Vas100,
    /// NRS 0-10
    Nrs11,
    /// BPI severity subscale
    BpiSeverity,
    /// BPI interference subscale
    BpiInterference,
}

/// Calculate percent change in pain
pub fn percent_change(baseline: f64, current: f64) -> f64 {
    if baseline < 0.1 {
        return 0.0;
    }
    (baseline - current) / baseline * 100.0
}

/// Classify treatment response
pub fn classify_response(percent_reduction: f64) -> TreatmentResponse {
    match percent_reduction as i32 {
        i32::MIN..=-1 => TreatmentResponse::Worsened,
        0..=29 => TreatmentResponse::NoResponse,
        30..=49 => TreatmentResponse::Moderate,
        50..=99 => TreatmentResponse::Substantial,
        _ => TreatmentResponse::Complete,
    }
}

/// Treatment response classification (IMMPACT guidelines)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreatmentResponse {
    /// Pain worsened
    Worsened,
    /// No meaningful response (<30% reduction)
    NoResponse,
    /// Moderate response (30-49% reduction)
    Moderate,
    /// Substantial response (≥50% reduction)
    Substantial,
    /// Complete response (100% reduction)
    Complete,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vas_score() {
        let vas = VasScore::new(45.0, PainDimension::CurrentIntensity);

        assert_eq!(vas.severity(), PainSeverity::Moderate);
        assert!((vas.to_nrs() - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_bpi() {
        let bpi = BriefPainInventory {
            worst_pain: 8,
            least_pain: 2,
            average_pain: 5,
            current_pain: 5,
            interference: BpiInterference {
                general_activity: 6,
                mood: 5,
                walking: 4,
                work: 6,
                relations: 3,
                sleep: 7,
                enjoyment: 5,
            },
        };

        let severity = bpi.severity_score();
        assert!((severity - 5.0).abs() < 0.01);

        assert!(bpi.is_clinically_significant());
    }

    #[test]
    fn test_mcid() {
        assert!(mcid_reached(50.0, 25.0, McidScale::Vas100));
        assert!(!mcid_reached(50.0, 45.0, McidScale::Vas100));
    }

    #[test]
    fn test_treatment_response() {
        assert_eq!(classify_response(55.0), TreatmentResponse::Substantial);
        assert_eq!(classify_response(35.0), TreatmentResponse::Moderate);
        assert_eq!(classify_response(15.0), TreatmentResponse::NoResponse);
    }
}
