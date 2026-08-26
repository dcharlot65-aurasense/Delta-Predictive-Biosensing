//! Random number generation helpers for dyn Rng compatibility
//!
//! These helpers allow generating random values in ranges using only Rng trait,
//! which is dyn-compatible (object-safe).

use rand::Rng;

/// Generate a random f64 in [0, 1)
#[inline]
pub fn random_f64(rng: &mut dyn Rng) -> f64 {
    (rng.next_u64() as f64) / (u64::MAX as f64)
}

/// Generate a random f64 in [min, max]
#[inline]
pub fn random_f64_range(rng: &mut dyn Rng, min: f64, max: f64) -> f64 {
    min + random_f64(rng) * (max - min)
}

/// Generate a random usize in [min, max] (inclusive)
#[inline]
pub fn random_usize_range(rng: &mut dyn Rng, min: usize, max: usize) -> usize {
    if min >= max {
        return min;
    }
    let range = (max - min + 1) as u64;
    let random = rng.next_u64() % range;
    min + random as usize
}

/// Generate a random i32 in [min, max] (inclusive)
#[inline]
pub fn random_i32_range(rng: &mut dyn Rng, min: i32, max: i32) -> i32 {
    if min >= max {
        return min;
    }
    let range = (max - min + 1) as u64;
    let random = rng.next_u64() % range;
    min + random as i32
}
