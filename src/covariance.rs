// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Dense covariance export, mirroring tinyopt's `CovarianceOptions`.

use crate::backend::{DenseMatrix, Dim, Dynamic, Matrix, inverse};
use crate::cost::Cost;
use crate::real::Real;
use crate::sparse::SparseMatrix;

/// Structure of the exported covariance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CovariancePattern {
    /// Natural structure for the Hessian storage (dense or sparse).
    #[default]
    Auto,
    /// Keep only block-diagonal (or diagonal) entries.
    BlockDiagonal,
    /// Keep the sparsity pattern of the Hessian.
    PreserveSparsity,
    /// Dense reduced block + block-diagonal eliminated blocks (Schur).
    SchurPartitioned,
    /// Always export a dense covariance.
    FullDense,
}

/// Exported covariance, whose variant follows the Hessian storage.
#[derive(Clone, Debug)]
pub enum FinalCovariance {
    /// Dense covariance.
    Dense(Matrix<f32, Dynamic>),
    /// Sparse covariance.
    Sparse(SparseMatrix<f32>),
    /// Block-tridiagonal covariance.
    BlockTridiag(crate::block_tridiag::BlockTridiagMat<0, f32>),
}

impl FinalCovariance {
    /// Dense view, when available.
    pub fn as_dense(&self) -> Option<&Matrix<f32, Dynamic>> {
        match self {
            FinalCovariance::Dense(m) => Some(m),
            FinalCovariance::Sparse(_) | FinalCovariance::BlockTridiag(_) => None,
        }
    }

    /// Sparse view, when available.
    pub fn as_sparse(&self) -> Option<&SparseMatrix<f32>> {
        match self {
            FinalCovariance::Sparse(m) => Some(m),
            FinalCovariance::Dense(_) | FinalCovariance::BlockTridiag(_) => None,
        }
    }

    /// Element accessor `(i, j)`.
    pub fn elem(&self, i: usize, j: usize) -> f32 {
        self.at(i, j)
    }

    /// Element accessor `(i, j)`.
    pub fn at(&self, i: usize, j: usize) -> f32 {
        match self {
            FinalCovariance::Dense(m) => m.elem(i, j),
            FinalCovariance::Sparse(m) => m.coeff(i, j),
            FinalCovariance::BlockTridiag(m) => m.get(i, j),
        }
    }
}

/// Options controlling the final covariance computation.
#[derive(Clone, Copy, Debug)]
pub struct CovarianceOptions {
    /// Zero the rows and columns of fixed parameters (otherwise leave identity).
    pub zero_fixed_variances: bool,
    /// Scale the covariance by `cost / (num_residuals - num_params)`.
    pub scale_by_residual_variance: bool,
    /// Extra diagonal damping added before inversion.
    pub damping: f64,
    /// Structure of the exported covariance.
    pub pattern: CovariancePattern,
}

impl Default for CovarianceOptions {
    fn default() -> Self {
        Self {
            zero_fixed_variances: true,
            scale_by_residual_variance: false,
            damping: 0.0,
            pattern: CovariancePattern::Auto,
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

/// Compute a sparse covariance from an un-damped sparse Hessian.
pub fn compute_sparse_covariance<S: Real>(
    hessian: &SparseMatrix<S>,
    options: &CovarianceOptions,
    cost: Cost,
    fixed: &[usize],
    dims: usize,
    num_reduced: usize,
) -> Option<FinalCovariance> {
    let mut h = hessian.clone();
    if options.damping > 0.0 {
        let d = S::from_f64(options.damping).expect("damping fits");
        for i in 0..dims {
            *h.coeff_ref(i, i) += d;
        }
    }

    let linear = crate::options::LinearSolverOptions {
        method: crate::backend::LinearSolverMethod::Ldlt,
        allow_schur: true,
        ..Default::default()
    };
    let mut solver = crate::sparse::SparseLinearSolvers::<S>::new(linear);
    if !solver.analyze(&h, num_reduced) || !solver.compute(&h) {
        return None;
    }
    let mut cov = solver.inverse(&h)?;

    if options.zero_fixed_variances {
        for &idx in fixed {
            if idx < dims {
                for j in 0..dims {
                    cov.set_coeff(idx, j, S::zero());
                    cov.set_coeff(j, idx, S::zero());
                }
            }
        }
    }

    if options.scale_by_residual_variance {
        let n = cost.num_residuals as f64;
        let p = dims as f64;
        if n > p && cost.cost.is_finite() {
            let s2 = S::from_f64(cost.cost / (n - p)).expect("variance fits");
            for j in 0..dims {
                for i in 0..dims {
                    let v = cov.coeff(i, j);
                    if v != S::zero() {
                        cov.set_coeff(i, j, v * s2);
                    }
                }
            }
        }
    }

    let result = match options.pattern {
        CovariancePattern::FullDense => FinalCovariance::Dense(cov.cast::<f32>().to_dense()),
        CovariancePattern::PreserveSparsity => {
            FinalCovariance::Sparse(cov.mask_by_pattern(&h).cast::<f32>())
        }
        CovariancePattern::BlockDiagonal => {
            let mut diag = crate::sparse::SparseMatrix::<S>::new(dims, dims);
            for i in 0..dims {
                let v = cov.coeff(i, i);
                if v != S::zero() {
                    diag.set_coeff(i, i, v);
                }
            }
            FinalCovariance::Sparse(diag.cast::<f32>())
        }
        CovariancePattern::Auto | CovariancePattern::SchurPartitioned => {
            FinalCovariance::Sparse(cov.cast::<f32>())
        }
    };
    Some(result)
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
