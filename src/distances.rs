// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Distances between two vectors, mirroring `tinyopt::distances`.
//!
//! The `*_with_jac` variants return `(distance, ∂d/∂a, ∂d/∂b)`.

use crate::losses::mahalanobis::{maha_norm_cov_with_jac, maha_norm_vars};
use crate::losses::norms::{l1, l2, linf};
use crate::real::Real;

fn difference<S: Real>(a: &[S], b: &[S]) -> Vec<S> {
    assert_eq!(a.len(), b.len(), "distances require matching sizes");
    a.iter().zip(b).map(|(x, y)| *x - *y).collect()
}

fn negate<S: Real>(v: &[S]) -> Vec<S> {
    v.iter().map(|x| -*x).collect()
}

/// Euclidean (L2) distance.
pub fn euclidean<S: Real>(a: &[S], b: &[S]) -> S {
    l2(&difference(a, b))
}

/// Euclidean distance with gradients w.r.t. both arguments.
pub fn euclidean_with_jac<S: Real>(a: &[S], b: &[S]) -> (S, Vec<S>, Vec<S>) {
    let d = difference(a, b);
    let (dist, jac) = crate::losses::norms::l2_with_jac(&d);
    (dist, jac.clone(), negate(&jac))
}

/// Alias of [`euclidean`].
pub fn l2_distance<S: Real>(a: &[S], b: &[S]) -> S {
    euclidean(a, b)
}

/// Alias of [`euclidean_with_jac`].
pub fn l2_distance_with_jac<S: Real>(a: &[S], b: &[S]) -> (S, Vec<S>, Vec<S>) {
    euclidean_with_jac(a, b)
}

/// Manhattan (L1) distance.
pub fn manhattan<S: Real>(a: &[S], b: &[S]) -> S {
    l1(&difference(a, b))
}

/// Manhattan distance with gradients w.r.t. both arguments.
pub fn manhattan_with_jac<S: Real>(a: &[S], b: &[S]) -> (S, Vec<S>, Vec<S>) {
    let d = difference(a, b);
    let (dist, jac) = crate::losses::norms::l1_with_jac(&d);
    (dist, jac.clone(), negate(&jac))
}

/// Alias of [`manhattan`].
pub fn l1_distance<S: Real>(a: &[S], b: &[S]) -> S {
    manhattan(a, b)
}

/// Alias of [`manhattan_with_jac`].
pub fn l1_distance_with_jac<S: Real>(a: &[S], b: &[S]) -> (S, Vec<S>, Vec<S>) {
    manhattan_with_jac(a, b)
}

/// L-infinity distance.
pub fn linf_distance<S: Real>(a: &[S], b: &[S]) -> S {
    linf(&difference(a, b))
}

/// L-infinity distance with gradients w.r.t. both arguments.
pub fn linf_distance_with_jac<S: Real>(a: &[S], b: &[S]) -> (S, Vec<S>, Vec<S>) {
    let d = difference(a, b);
    let (dist, jac) = crate::losses::norms::linf_with_jac(&d);
    (dist, jac.clone(), negate(&jac))
}

/// Cosine similarity `a·b / (||a|| ||b||)`.
pub fn cosine<S: Real>(a: &[S], b: &[S]) -> S {
    cosine_with_jac(a, b).0
}

/// Cosine similarity with gradients w.r.t. both arguments.
pub fn cosine_with_jac<S: Real>(a: &[S], b: &[S]) -> (S, Vec<S>, Vec<S>) {
    let n = a.len();
    let a_norm = l2(a);
    let b_norm = l2(b);
    let eps2 = S::from_f64(if core::mem::size_of::<S>() == 4 {
        1e-8
    } else {
        1e-14
    })
    .expect("epsilon fits");

    if a_norm * b_norm < eps2 {
        return (S::zero(), vec![S::zero(); n], vec![S::zero(); n]);
    }

    let mut ab = S::zero();
    for (x, y) in a.iter().zip(b) {
        ab += *x * *y;
    }
    let denom = a_norm * b_norm;
    let d = ab / denom;

    let mut ja = vec![S::zero(); n];
    let mut jb = vec![S::zero(); n];
    let a3b = a_norm * a_norm * a_norm * b_norm;
    let ab3 = a_norm * b_norm * b_norm * b_norm;
    for i in 0..n {
        ja[i] = b[i] / denom - ab * a[i] / a3b;
        jb[i] = a[i] / denom - ab * b[i] / ab3;
    }
    (d, ja, jb)
}

/// Mahalanobis distance with per-coefficient variances.
pub fn maha_norm_vars_distance<S: Real>(a: &[S], b: &[S], vars: &[S]) -> S {
    maha_norm_vars(&difference(a, b), vars)
}

/// Mahalanobis distance with a full covariance (column-major).
pub fn maha_norm_cov_distance<S: Real>(a: &[S], b: &[S], cov: &[S], n: usize) -> Option<S> {
    let d = difference(a, b);
    crate::losses::mahalanobis::maha_norm_cov(&d, cov, n)
}

/// Mahalanobis distance with its gradients w.r.t. both arguments.
pub fn maha_norm_cov_distance_with_jac<S: Real>(
    a: &[S],
    b: &[S],
    cov: &[S],
    n: usize,
) -> Option<(S, Vec<S>, Vec<S>)> {
    let d = difference(a, b);
    let (dist, grad) = maha_norm_cov_with_jac(&d, cov, n)?;
    Some((dist, grad.clone(), negate(&grad)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn l2_and_gradients() {
        let a = [1.0_f64, 2.0];
        let b = [4.0_f64, 6.0];
        let (d, ja, jb) = euclidean_with_jac(&a, &b);
        assert_relative_eq!(d, 5.0);
        assert_relative_eq!(ja[0], -0.6);
        assert_relative_eq!(ja[1], -0.8);
        assert_relative_eq!(jb[0], 0.6);
        assert_relative_eq!(jb[1], 0.8);
    }

    #[test]
    fn l1_and_linf() {
        let a = [1.0_f64, 2.0];
        let b = [4.0_f64, 0.0];
        assert_relative_eq!(manhattan(&a, &b), 5.0);
        assert_relative_eq!(linf_distance(&a, &b), 3.0);
    }

    #[test]
    fn cosine_similarity() {
        let a = [1.0_f64, 0.0];
        let b = [1.0_f64, 1.0];
        let (c, ja, jb) = cosine_with_jac(&a, &b);
        assert_relative_eq!(c, 1.0 / 2.0_f64.sqrt());
        assert!(ja.iter().chain(jb.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn mahalanobis_distance() {
        let a = [2.0_f64, 4.0];
        let b = [0.0_f64, 0.0];
        let vars = [2.0_f64, 8.0];
        assert_relative_eq!(
            maha_norm_vars_distance(&a, &b, &vars),
            maha_norm_vars(&a, &vars)
        );
    }
}
