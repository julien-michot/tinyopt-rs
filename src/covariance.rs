// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Dense covariance export, mirroring tinyopt's `CovarianceOptions`.

use crate::backend::{DenseMatrix, Dim, Matrix, inverse};
use crate::cost::Cost;
use crate::real::Real;

/// Options controlling the final covariance computation.
#[derive(Clone, Copy, Debug)]
pub struct CovarianceOptions {
    /// Zero the rows and columns of fixed parameters (otherwise leave identity).
    pub zero_fixed_variances: bool,
    /// Scale the covariance by `cost / (num_residuals - num_params)`.
    pub scale_by_residual_variance: bool,
    /// Extra diagonal damping added before inversion.
    pub damping: f64,
}

impl Default for CovarianceOptions {
    fn default() -> Self {
        Self {
            zero_fixed_variances: true,
            scale_by_residual_variance: false,
            damping: 0.0,
        }
    }
}

/// Compute the dense covariance from an (un-damped) Hessian approximation.
pub fn compute_covariance<S: Real, D: Dim>(
    hessian: &Matrix<S, D>,
    options: &CovarianceOptions,
    cost: Cost,
    fixed: &[usize],
    dims: usize,
) -> Option<Matrix<S, D>> {
    let mut h = hessian.clone();
    if options.damping > 0.0 {
        let d = S::from_f64(options.damping).expect("damping fits");
        for i in 0..dims {
            h.set_elem(i, i, h.elem(i, i) + d);
        }
    }

    let mut cov = inverse::<S, D>(&h)?;

    if options.zero_fixed_variances {
        for &idx in fixed {
            if idx < dims {
                cov.zero_row_col(idx);
            }
        }
    }

    if options.scale_by_residual_variance {
        let n = cost.num_residuals as f64;
        let p = dims as f64;
        if n > p && cost.cost.is_finite() {
            let s2 = S::from_f64(cost.cost / (n - p)).expect("variance fits");
            for i in 0..dims {
                for j in 0..dims {
                    cov.set_elem(i, j, cov.elem(i, j) * s2);
                }
            }
        }
    }

    Some(cov)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DenseMatrix, Dynamic, Matrix, zeros_matrix};
    use approx::assert_relative_eq;

    #[test]
    fn inverse_of_identity() {
        let h: Matrix<f64, Dynamic> = zeros_matrix::<f64, Dynamic>(2, 2);
        let mut h = h;
        h.set_elem(0, 0, 2.0);
        h.set_elem(1, 1, 4.0);
        let cov = compute_covariance::<f64, Dynamic>(
            &h,
            &CovarianceOptions::default(),
            Cost::default(),
            &[],
            2,
        )
        .expect("invertible");
        assert_relative_eq!(cov.elem(0, 0), 0.5);
        assert_relative_eq!(cov.elem(1, 1), 0.25);
    }

    #[test]
    fn fixed_and_scaling() {
        let mut h: Matrix<f64, Dynamic> = zeros_matrix::<f64, Dynamic>(2, 2);
        h.set_elem(0, 0, 2.0);
        h.set_elem(1, 1, 2.0);
        let cost = Cost::with_num_residuals(8.0, 4);
        let options = CovarianceOptions {
            scale_by_residual_variance: true,
            ..Default::default()
        };
        let cov =
            compute_covariance::<f64, Dynamic>(&h, &options, cost, &[1], 2).expect("invertible");
        // s² = 8 / (4 - 2) = 4; var0 = 0.5 * 4 = 2.
        assert_relative_eq!(cov.elem(0, 0), 2.0);
        assert_relative_eq!(cov.elem(1, 1), 0.0);
    }
}
