// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mahalanobis losses, mirroring `tinyopt::losses` Mahalanobis functions.
//!
//! Covariances are passed as flat **column-major** slices together with their
//! dimension, so these helpers stay backend-agnostic.

use crate::backend::{
    DenseMatrix, DenseVector, Dim, Dynamic, LinearSolverMethod, Matrix, Vector, inverse, solve,
    zeros_matrix, zeros_vector,
};
use crate::math::sqrt;
use crate::real::Real;

fn dyn_matrix<S: Real>(data: &[S], n: usize) -> Matrix<S, Dynamic> {
    let mut m = zeros_matrix::<S, Dynamic>(n, n);
    for j in 0..n {
        for i in 0..n {
            m.set_elem(i, j, data[j * n + i]);
        }
    }
    m
}

fn dyn_vector<S: Real>(data: &[S]) -> Vector<S, Dynamic> {
    let mut v = zeros_vector::<S, Dynamic>(data.len());
    for (i, x) in data.iter().enumerate() {
        v.set_at(i, *x);
    }
    v
}

/// Squared Mahalanobis norm of a scalar with variance `var`.
pub fn maha_squared_norm_scalar<S: Real>(x: S, var: S) -> S {
    let s = if var < S::epsilon() {
        S::one()
    } else {
        S::one() / var
    };
    x * x * s
}

/// Mahalanobis norm of a scalar with variance `var`.
pub fn maha_norm_scalar<S: Real>(x: S, var: S) -> S {
    sqrt(maha_squared_norm_scalar(x, var))
}

/// Squared Mahalanobis norm with per-coefficient variances.
pub fn maha_squared_norm_vars<S: Real>(x: &[S], vars: &[S]) -> S {
    assert_eq!(x.len(), vars.len(), "x and variances must match");
    let mut n2 = S::zero();
    for (xi, v) in x.iter().zip(vars) {
        n2 += *xi * *xi / *v;
    }
    n2
}

/// Mahalanobis norm with per-coefficient variances.
pub fn maha_norm_vars<S: Real>(x: &[S], vars: &[S]) -> S {
    sqrt(maha_squared_norm_vars(x, vars))
}

/// Squared Mahalanobis norm with a full covariance (column-major, `n × n`).
pub fn maha_squared_norm_cov<S: Real>(x: &[S], cov: &[S], n: usize) -> Option<S> {
    let c = dyn_matrix(cov, n);
    let inv = inverse::<S, Dynamic>(&c)?;
    let mut n2 = S::zero();
    for i in 0..n {
        for j in 0..n {
            n2 += x[i] * inv.elem(i, j) * x[j];
        }
    }
    Some(n2)
}

/// Mahalanobis norm with a full covariance.
pub fn maha_norm_cov<S: Real>(x: &[S], cov: &[S], n: usize) -> Option<S> {
    maha_squared_norm_cov(x, cov, n).map(sqrt)
}

/// Gradient of the Mahalanobis norm w.r.t. `x` (full covariance).
pub fn maha_norm_cov_with_jac<S: Real>(x: &[S], cov: &[S], n: usize) -> Option<(S, Vec<S>)> {
    let c = dyn_matrix(cov, n);
    let inv = inverse::<S, Dynamic>(&c)?;
    let mut grad = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = S::zero();
        for (j, xj) in x.iter().enumerate() {
            acc += inv.elem(i, j) * *xj;
        }
        grad.push(acc);
    }
    let mut n2 = S::zero();
    for i in 0..n {
        n2 += x[i] * grad[i];
    }
    let norm = sqrt(n2);
    if norm > S::epsilon() {
        for g in &mut grad {
            *g /= norm;
        }
    }
    Some((norm, grad))
}

/// Whitened residual for an isotropic standard deviation.
pub fn maha_whitened_scalar<S: Real>(res: &[S], stdev: S) -> Vec<S> {
    res.iter().map(|r| *r / stdev).collect()
}

/// Whitened residuals for per-coefficient standard deviations.
pub fn maha_whitened_vars<S: Real>(res: &[S], stdevs: &[S]) -> Vec<S> {
    assert_eq!(res.len(), stdevs.len(), "res and stdevs must match");
    res.iter().zip(stdevs).map(|(r, s)| *r / *s).collect()
}

/// Whitened residuals for a full covariance (`L⁻¹ r`, `cov = L Lᵀ`).
pub fn maha_whitened_cov<S: Real>(res: &[S], cov: &[S], n: usize) -> Option<Vec<S>> {
    let c = dyn_matrix(cov, n);
    let l = Dynamic::cholesky_lower(&c)?;
    let r = dyn_vector(res);
    let z = solve::<S, Dynamic>(LinearSolverMethod::Lu, &l, &r)?;
    Some(z.as_slice().to_vec())
}

/// Whitened residuals given the upper information factor `U` (`z = U r`).
pub fn maha_whitened_info_u<S: Real>(res: &[S], u: &[S], n: usize) -> Vec<S> {
    let up = dyn_matrix(u, n);
    let mut z = zeros_vector::<S, Dynamic>(n);
    for i in 0..n {
        let mut acc = S::zero();
        for (j, rj) in res.iter().enumerate().take(n) {
            acc += up.elem(i, j) * *rj;
        }
        z.set_at(i, acc);
    }
    z.as_slice().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn scalar_variance() {
        assert_relative_eq!(maha_squared_norm_scalar(2.0_f64, 4.0), 1.0);
        assert_relative_eq!(maha_norm_scalar(2.0_f64, 4.0), 1.0);
    }

    #[test]
    fn variances() {
        let x = [1.0_f64, 2.0];
        let vars = [1.0_f64, 4.0];
        assert_relative_eq!(maha_squared_norm_vars(&x, &vars), 1.0 + 1.0);
        assert_relative_eq!(maha_norm_vars(&x, &vars), 2.0_f64.sqrt());
    }

    #[test]
    fn full_covariance() {
        // cov = [[2, 0], [0, 8]] (column-major), x = [2, 4].
        let cov = [2.0_f64, 0.0, 0.0, 8.0];
        let x = [2.0_f64, 4.0];
        let n2 = maha_squared_norm_cov(&x, &cov, 2).unwrap();
        assert_relative_eq!(n2, 4.0 / 2.0 + 16.0 / 8.0);

        let (norm, grad) = maha_norm_cov_with_jac(&x, &cov, 2).unwrap();
        assert_relative_eq!(norm, (2.0_f64 + 2.0).sqrt());
        // Σ⁻¹ x = [1, 0.5], gradient = Σ⁻¹ x / ||x||_Σ.
        assert_relative_eq!(grad[0], 1.0 / norm);
        assert_relative_eq!(grad[1], 0.5 / norm);
    }

    #[test]
    fn whitening() {
        let res = [2.0_f64, 4.0];
        let stdevs = [2.0_f64, 4.0];
        assert_eq!(maha_whitened_vars(&res, &stdevs), vec![1.0, 1.0]);

        let cov = [4.0_f64, 0.0, 0.0, 16.0];
        let z = maha_whitened_cov(&res, &cov, 2).unwrap();
        assert_relative_eq!(z[0], 1.0);
        assert_relative_eq!(z[1], 1.0);

        let u = [0.5_f64, 0.0, 0.0, 0.25];
        let z = maha_whitened_info_u(&res, &u, 2);
        assert_relative_eq!(z[0], 1.0);
        assert_relative_eq!(z[1], 1.0);
    }
}
