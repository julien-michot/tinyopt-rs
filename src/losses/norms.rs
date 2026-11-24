// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Classical norms, mirroring `tinyopt::losses::{SquaredL2, L2, L1, Linf}`.

use crate::math::sqrt;
use crate::real::Real;

/// Squared L2 norm (sum of squares).
pub fn squared_l2<S: Real>(x: &[S]) -> S {
    x.iter().fold(S::zero(), |acc, v| acc + *v * *v)
}

/// Squared L2 norm and its Jacobian row (`2x`).
pub fn squared_l2_with_jac<S: Real>(x: &[S]) -> (S, Vec<S>) {
    let two = S::from_f64(2.0).expect("2 fits");
    (squared_l2(x), x.iter().map(|v| two * *v).collect())
}

/// L2 norm.
pub fn l2<S: Real>(x: &[S]) -> S {
    sqrt(squared_l2(x))
}

/// L2 norm and its Jacobian row (`x / ||x||`, or `x` near zero).
pub fn l2_with_jac<S: Real>(x: &[S]) -> (S, Vec<S>) {
    let norm = l2(x);
    if norm > S::epsilon() {
        let inv = S::one() / norm;
        (norm, x.iter().map(|v| *v * inv).collect())
    } else {
        (norm, x.to_vec())
    }
}

/// L1 norm.
pub fn l1<S: Real>(x: &[S]) -> S {
    x.iter()
        .fold(S::zero(), |acc, v| acc + num_traits::Float::abs(*v))
}

/// L1 norm and its Jacobian row (`sign(x)`).
pub fn l1_with_jac<S: Real>(x: &[S]) -> (S, Vec<S>) {
    let jac = x
        .iter()
        .map(|v| {
            if *v > S::zero() {
                S::one()
            } else if *v < S::zero() {
                -S::one()
            } else {
                S::zero()
            }
        })
        .collect();
    (l1(x), jac)
}

/// L-infinity norm.
pub fn linf<S: Real>(x: &[S]) -> S {
    x.iter().fold(S::zero(), |acc, v| {
        let a = num_traits::Float::abs(*v);
        if a > acc { a } else { acc }
    })
}

/// L-infinity norm and its Jacobian row (one-hot on the largest magnitude).
pub fn linf_with_jac<S: Real>(x: &[S]) -> (S, Vec<S>) {
    let mut max_idx = 0;
    let mut max_abs = S::zero();
    for (i, v) in x.iter().enumerate() {
        let a = num_traits::Float::abs(*v);
        if a > max_abs {
            max_abs = a;
            max_idx = i;
        }
    }
    let mut jac = vec![S::zero(); x.len()];
    if !x.is_empty() {
        jac[max_idx] = if x[max_idx] >= S::zero() {
            S::one()
        } else {
            -S::one()
        };
    }
    (max_abs, jac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn values() {
        let x = [3.0_f64, -4.0];
        assert_relative_eq!(squared_l2(&x), 25.0);
        assert_relative_eq!(l2(&x), 5.0);
        assert_relative_eq!(l1(&x), 7.0);
        assert_relative_eq!(linf(&x), 4.0);
    }

    #[test]
    fn jacobians() {
        let x = [3.0_f64, -4.0];
        let (_, j) = squared_l2_with_jac(&x);
        assert_eq!(j, vec![6.0, -8.0]);

        let (_, j) = l2_with_jac(&x);
        assert_relative_eq!(j[0], 0.6);
        assert_relative_eq!(j[1], -0.8);

        let (_, j) = l1_with_jac(&x);
        assert_eq!(j, vec![1.0, -1.0]);

        let (_, j) = linf_with_jac(&x);
        assert_eq!(j, vec![0.0, -1.0]);
    }

    #[test]
    fn zero_norm_jacobian_is_safe() {
        let (_l, j) = l2_with_jac(&[0.0_f64, 0.0]);
        assert!(j.iter().all(|v| v.is_finite()));
    }
}
