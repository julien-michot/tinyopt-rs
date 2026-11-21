// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Small numerical helpers, mirroring tinyopt's `math/utils.h`.

use crate::real::Real;

/// Default finite-difference step for numerical differentiation.
///
/// Matches tinyopt: `1e-4` in single precision, `1e-7` in double precision.
pub fn float_epsilon<S: Real>() -> S {
    if core::mem::size_of::<S>() == 4 {
        S::from_f64(1e-4).expect("1e-4 fits")
    } else {
        S::from_f64(1e-7).expect("1e-7 fits")
    }
}

/// Maximum absolute value of a slice.
pub fn max_abs<S: Real>(values: &[S]) -> S {
    values.iter().fold(S::zero(), |acc, v| {
        let a = num_traits::Float::abs(*v);
        if a > acc { a } else { acc }
    })
}

/// Whether every value is finite.
pub fn all_finite<S: Real>(values: &[S]) -> bool {
    values.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilons() {
        assert_eq!(float_epsilon::<f64>(), 1e-7);
        assert_eq!(float_epsilon::<f32>(), 1e-4);
    }

    #[test]
    fn finite_checks() {
        assert!(all_finite(&[1.0, -2.0]));
        assert!(!all_finite(&[1.0, f64::NAN]));
        assert_eq!(max_abs(&[1.0, -5.0, 2.0]), 5.0);
    }
}
