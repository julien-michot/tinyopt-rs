// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Small numerical helpers, mirroring tinyopt's `math/utils.h`.

use crate::backend::{DenseMatrix, DenseVector, Dim, Matrix, Vector, zeros_matrix, zeros_vector};
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

/// Scalar square root (disambiguated from nalgebra's `ComplexField`).
pub fn sqrt<S: Real>(x: S) -> S {
    num_traits::Float::sqrt(x)
}

/// Scalar exponential.
pub fn exp<S: Real>(x: S) -> S {
    num_traits::Float::exp(x)
}

/// Scalar natural logarithm.
pub fn ln<S: Real>(x: S) -> S {
    num_traits::Float::ln(x)
}

/// Scalar arctangent.
pub fn atan<S: Real>(x: S) -> S {
    num_traits::Float::atan(x)
}

/// Scalar hyperbolic tangent.
pub fn tanh<S: Real>(x: S) -> S {
    num_traits::Float::tanh(x)
}

/// Dense matrix inverse, mirroring C++ `tinyopt::Inverse`.
pub fn inverse<S: Real, D: Dim>(m: &Matrix<S, D>) -> Option<Matrix<S, D>> {
    crate::backend::inverse::<S, D>(m)
}

/// Dense matrix product `a * b`.
///
/// Implemented with explicit loops so it works on every backend; the cost is
/// `O(n³)` which is acceptable for the dense quasi-Newton updates.
pub fn mat_mul<S: Real, D: Dim>(a: &Matrix<S, D>, b: &Matrix<S, D>) -> Matrix<S, D> {
    let n = a.nrows();
    let mut out = zeros_matrix::<S, D>(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = S::zero();
            for k in 0..n {
                acc += a.elem(i, k) * b.elem(k, j);
            }
            out.set_elem(i, j, acc);
        }
    }
    out
}

/// Dense matrix transpose.
pub fn mat_transpose<S: Real, D: Dim>(a: &Matrix<S, D>) -> Matrix<S, D> {
    let n = a.nrows();
    let mut out = zeros_matrix::<S, D>(n, n);
    for i in 0..n {
        for j in 0..n {
            out.set_elem(i, j, a.elem(j, i));
        }
    }
    out
}

/// Dense identity matrix.
pub fn mat_identity<S: Real, D: Dim>(n: usize) -> Matrix<S, D> {
    let mut m = zeros_matrix::<S, D>(n, n);
    for i in 0..n {
        m.set_elem(i, i, S::one());
    }
    m
}

/// Dense matrix-vector product.
pub fn mat_vec<S: Real, D: Dim>(a: &Matrix<S, D>, x: &Vector<S, D>) -> Vector<S, D> {
    let n = x.len();
    let mut out = zeros_vector::<S, D>(n);
    for i in 0..n {
        let mut acc = S::zero();
        for j in 0..n {
            acc += a.elem(i, j) * x.at(j);
        }
        out.set_at(i, acc);
    }
    out
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
