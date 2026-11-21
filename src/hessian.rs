// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Hessian storage abstraction: dense matrices and sparse matrices share the
//! same optimizer loop, mirroring tinyopt's `Hessian_t` dispatch.

use crate::backend::{
    DenseMatrix, DenseVector, Dynamic, Matrix, solve, zeros_matrix, zeros_vector,
};
use crate::cost::Cost;
use crate::covariance::{
    CovarianceOptions, CovariancePattern, FinalCovariance, compute_covariance,
    compute_sparse_covariance,
};
use crate::options::LinearSolverOptions;
use crate::real::Real;
use crate::sparse::{SparseLinearSolvers, SparseMatrix};

/// Operations the optimizer needs from a Hessian representation.
pub trait HessianStorage<S: Real>: Clone + core::fmt::Debug + Send + Sync {
    /// Zeroed storage of the given scalar dimension.
    fn zeros(dims: usize) -> Self;

    /// Zeroed storage shaped like `prototype` (for dynamic layouts).
    fn zeros_like(prototype: &Self, dims: usize) -> Self {
        let _ = prototype;
        Self::zeros(dims)
    }
    /// Scalar dimension.
    fn dims(&self) -> usize;
    /// Zero all stored values, keeping the allocation.
    fn set_zero(&mut self);
    /// Write the diagonal coefficients into `out`.
    fn diagonal_into(&self, out: &mut [S]);
    /// Multiply the diagonal by `factor` (LM damping).
    fn scale_diagonal(&mut self, factor: S);
    /// Scale by `diag(scale) * diag(scale)` (Jacobi), or divide when `inverse`.
    fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool);
    /// Zero fixed rows/columns and set their diagonal to one.
    fn apply_fixed_indices(&mut self, indices: &[usize]);
    /// Whether every stored value is finite.
    fn is_all_finite(&self) -> bool;
    /// Compute `H x` into `out`.
    fn mul_vec_into(&self, x: &[S], out: &mut [S]);
    /// Solve the (possibly Schur-partitioned) system into `out`.
    fn solve_into(
        &self,
        options: &LinearSolverOptions,
        rhs: &[S],
        num_reduced: usize,
        out: &mut [S],
    ) -> bool;

    /// Matrix-vector product (convenience allocation).
    fn mul_vec(&self, x: &[S]) -> Vec<S> {
        let mut out = vec![S::zero(); self.dims()];
        self.mul_vec_into(x, &mut out);
        out
    }

    /// Solve (convenience allocation).
    fn solve(
        &self,
        options: &LinearSolverOptions,
        rhs: &[S],
        num_reduced: usize,
    ) -> Option<Vec<S>> {
        let mut out = vec![S::zero(); rhs.len()];
        if self.solve_into(options, rhs, num_reduced, &mut out) {
            Some(out)
        } else {
            None
        }
    }
    /// Dense conversion.
    fn to_dense(&self) -> Matrix<S, Dynamic>;
    /// Covariance export following the requested pattern.
    fn to_final_covariance(
        &self,
        options: &CovarianceOptions,
        cost: Cost,
        fixed: &[usize],
        num_reduced: usize,
    ) -> Option<FinalCovariance>;
}

fn dense_diagonal_into<S: Real, M: DenseMatrix<S>>(m: &M, out: &mut [S]) {
    for (i, o) in out.iter_mut().enumerate().take(m.nrows().min(m.ncols())) {
        *o = m.elem(i, i);
    }
}

fn dense_scale_diagonal<S: Real, M: DenseMatrix<S>>(m: &mut M, factor: S) {
    for i in 0..m.nrows().min(m.ncols()) {
        m.set_elem(i, i, m.elem(i, i) * factor);
    }
}

fn dense_scale_by_diagonal<S: Real, M: DenseMatrix<S>>(m: &mut M, scale: &[S], inverse: bool) {
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            let factor = scale[i] * scale[j];
            let value = if inverse {
                m.elem(i, j) / factor
            } else {
                m.elem(i, j) * factor
            };
            m.set_elem(i, j, value);
        }
    }
}

fn dense_apply_fixed<S: Real, M: DenseMatrix<S>>(m: &mut M, indices: &[usize]) {
    for &idx in indices {
        if idx < m.nrows().min(m.ncols()) {
            m.zero_row_col(idx);
            m.set_elem(idx, idx, S::one());
        }
    }
}

fn dense_mul_vec_into<S: Real, M: DenseMatrix<S>>(m: &M, x: &[S], out: &mut [S]) {
    for (i, o) in out.iter_mut().enumerate().take(m.nrows()) {
        *o = (0..m.ncols()).fold(S::zero(), |acc, j| acc + m.elem(i, j) * x[j]);
    }
}

fn dense_all_finite<S: Real, M: DenseMatrix<S>>(m: &M) -> bool {
    (0..m.nrows()).all(|i| (0..m.ncols()).all(|j| m.elem(i, j).is_finite()))
}

fn dense_to_dynamic<S: Real, M: DenseMatrix<S>>(m: &M) -> Matrix<S, Dynamic> {
    let mut out = zeros_matrix::<S, Dynamic>(m.nrows(), m.ncols());
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            out.set_elem(i, j, m.elem(i, j));
        }
    }
    out
}

fn dense_solve_into<S: Real, D: crate::backend::Dim>(
    m: &Matrix<S, D>,
    options: &LinearSolverOptions,
    rhs: &[S],
    out: &mut [S],
) -> bool {
    let mut v = zeros_vector::<S, D>(rhs.len());
    for (i, x) in rhs.iter().enumerate() {
        v.set_at(i, *x);
    }
    match solve::<S, D>(options.method, m, &v) {
        Some(x) => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = x.at(i);
            }
            true
        }
        None => false,
    }
}

fn dense_covariance<S: Real, M: DenseMatrix<S>>(
    m: &M,
    options: &CovarianceOptions,
    cost: Cost,
    fixed: &[usize],
) -> Option<FinalCovariance> {
    let n = m.nrows();
    let dense = dense_to_dynamic(m);
    let mut cov = compute_covariance::<S, Dynamic>(&dense, options, cost, fixed, n)?;
    if options.pattern == CovariancePattern::BlockDiagonal {
        for j in 0..n {
            for i in 0..n {
                if i != j {
                    cov.set_elem(i, j, S::zero());
                }
            }
        }
    }
    let mut out = zeros_matrix::<f32, Dynamic>(n, n);
    for j in 0..n {
        for i in 0..n {
            out.set_elem(i, j, cov.elem(i, j).to_f32().expect("fits in f32"));
        }
    }
    Some(FinalCovariance::Dense(out))
}

#[cfg(feature = "nalgebra")]
mod impls {
    use super::*;
    use nalgebra::{DMatrix, SMatrix};

    macro_rules! impl_dense_hessian {
        ($ty:ty) => {
            impl<S: Real> HessianStorage<S> for $ty {
                fn zeros(dims: usize) -> Self {
                    <Self as DenseMatrix<S>>::zeros(dims, dims)
                }

                fn dims(&self) -> usize {
                    self.nrows()
                }

                fn set_zero(&mut self) {
                    DenseMatrix::set_zero(self);
                }

                fn diagonal_into(&self, out: &mut [S]) {
                    dense_diagonal_into(self, out);
                }

                fn scale_diagonal(&mut self, factor: S) {
                    dense_scale_diagonal(self, factor);
                }

                fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
                    dense_scale_by_diagonal(self, scale, inverse);
                }

                fn apply_fixed_indices(&mut self, indices: &[usize]) {
                    dense_apply_fixed(self, indices);
                }

                fn is_all_finite(&self) -> bool {
                    dense_all_finite(self)
                }

                fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
                    dense_mul_vec_into(self, x, out);
                }

                fn solve_into(
                    &self,
                    options: &LinearSolverOptions,
                    rhs: &[S],
                    _num_reduced: usize,
                    out: &mut [S],
                ) -> bool {
                    dense_solve_into::<S, Dynamic>(self, options, rhs, out)
                }

                fn to_dense(&self) -> Matrix<S, Dynamic> {
                    dense_to_dynamic(self)
                }

                fn to_final_covariance(
                    &self,
                    options: &CovarianceOptions,
                    cost: Cost,
                    fixed: &[usize],
                    _num_reduced: usize,
                ) -> Option<FinalCovariance> {
                    dense_covariance(self, options, cost, fixed)
                }
            }
        };
    }

    impl_dense_hessian!(DMatrix<S>);

    impl<S: Real, const N: usize> HessianStorage<S> for SMatrix<S, N, N> {
        fn zeros(dims: usize) -> Self {
            assert_eq!(dims, N, "static Hessian dimension mismatch");
            SMatrix::zeros()
        }

        fn dims(&self) -> usize {
            N
        }

        fn set_zero(&mut self) {
            DenseMatrix::set_zero(self);
        }

        fn diagonal_into(&self, out: &mut [S]) {
            dense_diagonal_into(self, out);
        }

        fn scale_diagonal(&mut self, factor: S) {
            dense_scale_diagonal(self, factor);
        }

        fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
            dense_scale_by_diagonal(self, scale, inverse);
        }

        fn apply_fixed_indices(&mut self, indices: &[usize]) {
            dense_apply_fixed(self, indices);
        }

        fn is_all_finite(&self) -> bool {
            dense_all_finite(self)
        }

        fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
            dense_mul_vec_into(self, x, out);
        }

        fn solve_into(
            &self,
            options: &LinearSolverOptions,
            rhs: &[S],
            _num_reduced: usize,
            out: &mut [S],
        ) -> bool {
            dense_solve_into::<S, crate::backend::Static<N>>(self, options, rhs, out)
        }

        fn to_dense(&self) -> Matrix<S, Dynamic> {
            dense_to_dynamic(self)
        }

        fn to_final_covariance(
            &self,
            options: &CovarianceOptions,
            cost: Cost,
            fixed: &[usize],
            _num_reduced: usize,
        ) -> Option<FinalCovariance> {
            dense_covariance(self, options, cost, fixed)
        }
    }
}

#[cfg(feature = "faer")]
mod impls {
    use super::*;
    use faer::Mat;

    impl<S: Real> HessianStorage<S> for Mat<S> {
        fn zeros(dims: usize) -> Self {
            Mat::from_fn(dims, dims, |_, _| S::zero())
        }

        fn dims(&self) -> usize {
            self.nrows()
        }

        fn set_zero(&mut self) {
            DenseMatrix::set_zero(self);
        }

        fn diagonal_into(&self, out: &mut [S]) {
            dense_diagonal_into(self, out);
        }

        fn scale_diagonal(&mut self, factor: S) {
            dense_scale_diagonal(self, factor);
        }

        fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
            dense_scale_by_diagonal(self, scale, inverse);
        }

        fn apply_fixed_indices(&mut self, indices: &[usize]) {
            dense_apply_fixed(self, indices);
        }

        fn is_all_finite(&self) -> bool {
            dense_all_finite(self)
        }

        fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
            dense_mul_vec_into(self, x, out);
        }

        fn solve_into(
            &self,
            options: &LinearSolverOptions,
            rhs: &[S],
            _num_reduced: usize,
            out: &mut [S],
        ) -> bool {
            dense_solve_into::<S, Dynamic>(self, options, rhs, out)
        }

        fn to_dense(&self) -> Matrix<S, Dynamic> {
            dense_to_dynamic(self)
        }

        fn to_final_covariance(
            &self,
            options: &CovarianceOptions,
            cost: Cost,
            fixed: &[usize],
            _num_reduced: usize,
        ) -> Option<FinalCovariance> {
            dense_covariance(self, options, cost, fixed)
        }
    }
}

impl<S: Real> HessianStorage<S> for SparseMatrix<S> {
    fn zeros(dims: usize) -> Self {
        SparseMatrix::new(dims, dims)
    }

    fn dims(&self) -> usize {
        self.nrows()
    }

    fn set_zero(&mut self) {
        SparseMatrix::set_zero(self);
    }

    fn diagonal_into(&self, out: &mut [S]) {
        let diag = SparseMatrix::diagonal(self);
        for (i, o) in out.iter_mut().enumerate().take(diag.len()) {
            *o = diag[i];
        }
    }

    fn scale_diagonal(&mut self, factor: S) {
        SparseMatrix::scale_diagonal(self, factor);
    }

    fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
        SparseMatrix::scale_by_diagonal(self, scale, inverse);
    }

    fn apply_fixed_indices(&mut self, indices: &[usize]) {
        SparseMatrix::apply_fixed_indices(self, indices);
    }

    fn is_all_finite(&self) -> bool {
        SparseMatrix::is_all_finite(self)
    }

    fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
        let values = SparseMatrix::mul_vec(self, x);
        for (o, v) in out.iter_mut().zip(values) {
            *o = v;
        }
    }

    fn solve_into(
        &self,
        options: &LinearSolverOptions,
        rhs: &[S],
        num_reduced: usize,
        out: &mut [S],
    ) -> bool {
        let mut solver = SparseLinearSolvers::<S>::new(options.clone());
        if !solver.analyze(self, num_reduced) || !solver.compute(self) {
            return false;
        }
        match solver.solve(self, rhs) {
            Some(x) => {
                for (o, v) in out.iter_mut().zip(x) {
                    *o = v;
                }
                true
            }
            None => false,
        }
    }

    fn to_dense(&self) -> Matrix<S, Dynamic> {
        SparseMatrix::to_dense(self)
    }

    fn to_final_covariance(
        &self,
        options: &CovarianceOptions,
        cost: Cost,
        fixed: &[usize],
        num_reduced: usize,
    ) -> Option<FinalCovariance> {
        compute_sparse_covariance(self, options, cost, fixed, self.nrows(), num_reduced)
    }
}
