//! Per-operation energy figures, with their sources.
//!
//! The headline number this crate produces is a ratio of energies, so it is
//! only meaningful when both sides are measured the same way. That was not the
//! case: the ANN path charged a multiply-accumulate 4.6 pJ, a figure for an
//! isolated arithmetic circuit, while the SNN path charged a synaptic
//! operation 50 pJ, a figure measured on whole neuromorphic silicon with its
//! memory and routing included. Comparing the two made spiking hardware look
//! about eleven times *worse* per operation, and a comment claiming the
//! opposite sat beside it.
//!
//! The constants are gathered here so the basis of a comparison is visible at
//! the point it is chosen, and so the same number cannot drift between
//! modules. Two of these were already scattered through the crate; one of them
//! had been written in the wrong unit (4600 where the field was picojoules).
//!
//! # Choosing a basis
//!
//! Use [`cmos_45nm`] for both sides of an architectural comparison. The
//! question it answers is what the arithmetic costs, and it is the basis the
//! spiking-network literature uses when it reports efficiency against an ANN.
//!
//! Use [`neuromorphic_silicon`] when comparing against a measurement from a
//! real chip, and then compare it against a measurement of the ANN on real
//! hardware too -- not against [`cmos_45nm`].

/// Energy per operation for a 45 nm process, after Horowitz, "Computing's
/// Energy Problem (and what we can do about it)", ISSCC 2014, which is the
/// table SNN efficiency results are conventionally quoted against.
pub mod cmos_45nm {
    /// 32-bit floating-point multiply.
    pub const FLOAT_MULTIPLY_PJ: f64 = 3.7;

    /// 32-bit floating-point add.
    pub const FLOAT_ADD_PJ: f64 = 0.9;

    /// 32-bit integer add.
    pub const INT_ADD_PJ: f64 = 0.1;

    /// Multiply-accumulate: one multiply and one add. This is the unit of work
    /// in a dense layer or a convolution.
    pub const FLOAT_MAC_PJ: f64 = FLOAT_MULTIPLY_PJ + FLOAT_ADD_PJ;

    /// Accumulate alone, which is the unit of work in a spiking network: a
    /// spike is binary, so its contribution is its weight added, with no
    /// multiply. Being an add rather than a multiply-accumulate is the whole
    /// arithmetic argument for spiking hardware, and it is worth about 5.1x.
    pub const FLOAT_ACCUMULATE_PJ: f64 = FLOAT_ADD_PJ;
}

/// Energy per synaptic operation measured on shipped neuromorphic parts.
///
/// These are whole-chip figures: they include reading the weight and routing
/// the spike, which is why they are an order of magnitude above the bare
/// arithmetic in [`cmos_45nm`]. They are not comparable to it.
pub mod neuromorphic_silicon {
    /// Intel Loihi, per synaptic operation (Davies et al., IEEE Micro 2018).
    pub const LOIHI_PJ: f64 = 23.6;

    /// IBM TrueNorth, per synaptic event (Merolla et al., Science 2014).
    pub const TRUENORTH_PJ: f64 = 26.0;

    /// A deliberately conservative figure for unspecified spiking hardware,
    /// above both parts above.
    pub const CONSERVATIVE_PJ: f64 = 50.0;
}

/// Picojoules to joules.
pub const PJ_TO_J: f64 = 1e-12;

/// Picojoules to millijoules.
pub const PJ_TO_MJ: f64 = 1e-9;

// These relationships are properties of the constants above, so they are
// checked when the crate compiles rather than when a test happens to run.
// Bounds rather than equalities: 3.7 and 0.9 are not exact in binary floating
// point, so their sum need not equal the literal 4.6 bit for bit.

/// A multiply-accumulate is a multiply plus an add.
const _: () = assert!(cmos_45nm::FLOAT_MAC_PJ > 4.59 && cmos_45nm::FLOAT_MAC_PJ < 4.61);

/// An add cannot cost more than a multiply and an add. The efficiency claim
/// for spiking hardware rests on this, and it used to be inverted: the SNN
/// side was charged 50 pJ against the ANN's 4.6 pJ, with a test asserting that
/// ordering, so the mistake was pinned in place rather than caught.
const _: () = assert!(cmos_45nm::FLOAT_ACCUMULATE_PJ < cmos_45nm::FLOAT_MAC_PJ);

/// The 45 nm accumulate-to-MAC advantage is about 5.1x.
const _: () = assert!(
    cmos_45nm::FLOAT_MAC_PJ > cmos_45nm::FLOAT_ACCUMULATE_PJ * 5.0
        && cmos_45nm::FLOAT_MAC_PJ < cmos_45nm::FLOAT_ACCUMULATE_PJ * 5.2
);

/// Whole-chip figures sit well above the bare arithmetic, which is exactly why
/// the two bases must not be mixed in one ratio.
const _: () = assert!(neuromorphic_silicon::LOIHI_PJ > cmos_45nm::FLOAT_MAC_PJ * 4.0);
const _: () = assert!(neuromorphic_silicon::CONSERVATIVE_PJ > neuromorphic_silicon::TRUENORTH_PJ);
