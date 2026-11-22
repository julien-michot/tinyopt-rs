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

/// Squared version of [`float_epsilon`], matching the C++ `FloatEpsilon2`.
pub fn float_epsilon2<S: Real>() -> S {
    if core::mem::size_of::<S>() == 4 {
        S::from_f64(1e-8).expect("1e-8 fits")
    } else {
        S::from_f64(1e-14).expect("1e-14 fits")
    }
}

/// Integer floor square root (`SQRT(N)` in C++).
#[allow(clippy::manual_div_ceil)]
pub const fn sqrt_int(n: usize) -> usize {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Closed-form solve of a small symmetric system (up to 4 unknowns), mirroring
/// the C++ `SolveSmallDenseSystem`.
pub fn solve_small_dense_system<S: Real, D: Dim>(
    a: &Matrix<S, D>,
    b: &Vector<S, D>,
) -> Option<Vector<S, D>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n || n == 0 || n > 4 {
        return None;
    }
    let eps = S::epsilon();
    let m = |i: usize, j: usize| a.elem(i, j);
    let mut out = zeros_vector::<S, D>(n);
    let finite = |v: S| v.is_finite();
    match n {
        1 => {
            let value = m(0, 0);
            if value <= eps || !finite(value) {
                return None;
            }
            out.set_at(0, b.at(0) / value);
        }
        2 => {
            let (a00, a01, a11) = (m(0, 0), m(0, 1), m(1, 1));
            let trace = a00 + a11;
            let scale = num_traits::Float::max(
                num_traits::Float::abs(a00),
                num_traits::Float::max(num_traits::Float::abs(a01), num_traits::Float::abs(a11)),
            );
            if trace <= eps * scale || !finite(trace) || !finite(scale) {
                return None;
            }
            let n00 = a00 / trace;
            let n01 = a01 / trace;
            let n11 = a11 / trace;
            let determinant = n00 * n11 - n01 * n01;
            if num_traits::Float::abs(determinant) <= eps || !finite(determinant) {
                return None;
            }
            let denominator = trace * determinant;
            if denominator == S::zero() || !finite(denominator) {
                return None;
            }
            out.set_at(0, (n11 * b.at(0) - n01 * b.at(1)) / denominator);
            out.set_at(1, (n00 * b.at(1) - n01 * b.at(0)) / denominator);
        }
        3 => {
            let (a00, a01, a02) = (m(0, 0), m(0, 1), m(0, 2));
            let (a11, a12) = (m(1, 1), m(1, 2));
            let a22 = m(2, 2);
            let trace = a00 + a11 + a22;
            let scale = [
                num_traits::Float::abs(a00),
                num_traits::Float::abs(a01),
                num_traits::Float::abs(a02),
                num_traits::Float::abs(a11),
                num_traits::Float::abs(a12),
                num_traits::Float::abs(a22),
            ]
            .into_iter()
            .fold(S::zero(), num_traits::Float::max);
            if num_traits::Float::abs(trace) <= eps * scale || !finite(trace) || !finite(scale) {
                return None;
            }
            let n00 = a00 / trace;
            let n01 = a01 / trace;
            let n02 = a02 / trace;
            let n11 = a11 / trace;
            let n12 = a12 / trace;
            let n22 = a22 / trace;
            let determinant = n00 * (n11 * n22 - n12 * n12) - n01 * (n01 * n22 - n12 * n02)
                + n02 * (n01 * n12 - n11 * n02);
            if num_traits::Float::abs(determinant) <= eps || !finite(determinant) {
                return None;
            }
            let denominator = trace * determinant;
            if denominator == S::zero() || !finite(denominator) {
                return None;
            }
            let (b0, b1, b2) = (b.at(0), b.at(1), b.at(2));
            out.set_at(
                0,
                ((n11 * n22 - n12 * n12) * b0
                    + (n02 * n12 - n01 * n22) * b1
                    + (n01 * n12 - n02 * n11) * b2)
                    / denominator,
            );
            out.set_at(
                1,
                ((n02 * n12 - n01 * n22) * b0
                    + (n00 * n22 - n02 * n02) * b1
                    + (n01 * n02 - n00 * n12) * b2)
                    / denominator,
            );
            out.set_at(
                2,
                ((n01 * n12 - n02 * n11) * b0
                    + (n01 * n02 - n00 * n12) * b1
                    + (n00 * n11 - n01 * n01) * b2)
                    / denominator,
            );
        }
        _ => {
            let (a00, a01, a02, a03) = (m(0, 0), m(0, 1), m(0, 2), m(0, 3));
            let (a11, a12, a13) = (m(1, 1), m(1, 2), m(1, 3));
            let (a22, a23) = (m(2, 2), m(2, 3));
            let a33 = m(3, 3);
            let trace = a00 + a11 + a22 + a33;
            let scale = [
                num_traits::Float::abs(a00),
                num_traits::Float::abs(a01),
                num_traits::Float::abs(a02),
                num_traits::Float::abs(a03),
                num_traits::Float::abs(a11),
                num_traits::Float::abs(a12),
                num_traits::Float::abs(a13),
                num_traits::Float::abs(a22),
                num_traits::Float::abs(a23),
                num_traits::Float::abs(a33),
            ]
            .into_iter()
            .fold(S::zero(), num_traits::Float::max);
            if num_traits::Float::abs(trace) <= eps * scale || !finite(trace) || !finite(scale) {
                return None;
            }
            let n = |v: S| v / trace;
            let (n00, n01, n02, n03) = (n(a00), n(a01), n(a02), n(a03));
            let (n11, n12, n13) = (n(a11), n(a12), n(a13));
            let (n22, n23) = (n(a22), n(a23));
            let n33 = n(a33);
            let det3 = |m00: S, m01: S, m02: S, m10: S, m11: S, m12: S, m20: S, m21: S, m22: S| {
                m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
                    + m02 * (m10 * m21 - m11 * m20)
            };
            let c00 = det3(n11, n12, n13, n12, n22, n23, n13, n23, n33);
            let c01 = -det3(n01, n12, n13, n02, n22, n23, n03, n23, n33);
            let c02 = det3(n01, n11, n13, n02, n12, n23, n03, n13, n33);
            let c03 = -det3(n01, n11, n12, n02, n12, n22, n03, n13, n23);
            let determinant = n00 * c00 + n01 * c01 + n02 * c02 + n03 * c03;
            if num_traits::Float::abs(determinant) <= eps * eps || !finite(determinant) {
                return None;
            }
            let denominator = trace * determinant;
            if denominator == S::zero() || !finite(denominator) {
                return None;
            }
            let c11 = det3(n00, n02, n03, n02, n22, n23, n03, n23, n33);
            let c12 = -det3(n00, n01, n03, n02, n12, n23, n03, n13, n33);
            let c13 = det3(n00, n01, n02, n02, n12, n22, n03, n13, n23);
            let c22 = det3(n00, n01, n03, n01, n11, n13, n03, n13, n33);
            let c23 = -det3(n00, n01, n02, n01, n11, n12, n03, n13, n23);
            let c33 = det3(n00, n01, n02, n01, n11, n12, n02, n12, n22);
            let (b0, b1, b2, b3) = (b.at(0), b.at(1), b.at(2), b.at(3));
            out.set_at(0, (c00 * b0 + c01 * b1 + c02 * b2 + c03 * b3) / denominator);
            out.set_at(1, (c01 * b0 + c11 * b1 + c12 * b2 + c13 * b3) / denominator);
            out.set_at(2, (c02 * b0 + c12 * b1 + c22 * b2 + c23 * b3) / denominator);
            out.set_at(3, (c03 * b0 + c13 * b1 + c23 * b2 + c33 * b3) / denominator);
        }
    }
    (0..n).all(|i| out.at(i).is_finite()).then_some(out)
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

/// Dense inverse with an SVD pseudo-inverse fallback for singular systems,
/// matching the robustness of the C++ `Inverse` helper.
pub fn robust_inverse<S: Real, D: Dim>(m: &Matrix<S, D>) -> Option<Matrix<S, D>> {
    if m.nrows() == 0 || m.nrows() != m.ncols() {
        return None;
    }
    if let Some(inv) = crate::backend::inverse::<S, D>(m) {
        let n = inv.nrows();
        let finite = (0..n).all(|j| (0..n).all(|i| inv.elem(i, j).is_finite()));
        if finite {
            return Some(inv);
        }
    }
    // SVD-based pseudo-inverse: solve `A x = e_j` for every column.
    let n = m.nrows();
    let mut out = zeros_matrix::<S, D>(n, n);
    for j in 0..n {
        let mut e = crate::backend::zeros_vector::<S, D>(n);
        e.set_at(j, S::one());
        let x = crate::backend::solve::<S, D>(crate::backend::LinearSolverMethod::Svd, m, &e)?;
        for i in 0..n {
            out.set_elem(i, j, x.at(i));
        }
    }
    Some(out)
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
