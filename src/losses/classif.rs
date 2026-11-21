// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Classification losses (softmax), mirroring `tinyopt::losses::classif`.

use crate::math::exp;
use crate::real::Real;

/// Softmax probabilities.
pub fn softmax<S: Real>(x: &[S]) -> Vec<S> {
    let mut sum = S::zero();
    for v in x {
        sum += exp(*v);
    }
    x.iter().map(|v| exp(*v) / sum).collect()
}

/// Softmax probabilities and the row-major Jacobian.
pub fn softmax_with_jac<S: Real>(x: &[S]) -> (Vec<S>, Vec<S>) {
    let out = softmax(x);
    let n = out.len();
    let mut jac = vec![S::zero(); n * n];
    for r in 0..n {
        for c in 0..n {
            jac[r * n + c] = if r == c {
                out[c] * (S::one() - out[c])
            } else {
                -out[r] * out[c]
            };
        }
    }
    (out, jac)
}

/// Numerically safe softmax (subtracts the maximum).
pub fn safe_softmax<S: Real>(x: &[S]) -> Vec<S> {
    if x.is_empty() {
        return Vec::new();
    }
    let max = x
        .iter()
        .fold(x[0], |acc, v| if *v > acc { *v } else { acc });
    let mut sum = S::zero();
    for v in x {
        sum += exp(*v - max);
    }
    x.iter().map(|v| exp(*v - max) / sum).collect()
}

/// Numerically safe softmax and the row-major Jacobian.
pub fn safe_softmax_with_jac<S: Real>(x: &[S]) -> (Vec<S>, Vec<S>) {
    let out = safe_softmax(x);
    let n = out.len();
    let mut jac = vec![S::zero(); n * n];
    for r in 0..n {
        for c in 0..n {
            jac[r * n + c] = if r == c {
                out[c] * (S::one() - out[c])
            } else {
                -out[r] * out[c]
            };
        }
    }
    (out, jac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn softmax_sums_to_one() {
        let x = [1.0_f64, 2.0, 3.0];
        let p = softmax(&x);
        assert_relative_eq!(p.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
        let ps = safe_softmax(&x);
        for (a, b) in p.iter().zip(&ps) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn softmax_jacobian_rows_sum_to_zero() {
        let x = [0.5_f64, -1.0, 2.0];
        let (p, jac) = softmax_with_jac(&x);
        let n = p.len();
        for r in 0..n {
            let row_sum: f64 = (0..n).map(|c| jac[r * n + c]).sum();
            assert_relative_eq!(row_sum, 0.0, epsilon = 1e-12);
        }
        // Diagonal entries are p_i (1 - p_i).
        for i in 0..n {
            assert_relative_eq!(jac[i * n + i], p[i] * (1.0 - p[i]));
        }
    }

    #[test]
    fn scalar_softmax_is_one() {
        assert_eq!(softmax(&[3.0_f64]), vec![1.0]);
    }
}
