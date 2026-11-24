// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Dense linear-algebra backend abstraction.
//!
//! Exactly one backend is compiled per build: [`nalgebra`](https://docs.rs/nalgebra)
//! (static and dynamic sizes) or [`faer`](https://docs.rs/faer) (dynamic sizes).
//! The rest of the crate is written against [`Dim`], [`DenseVector`] and
//! [`DenseMatrix`], which map onto the active backend's types.
//!
//! All matrices are assumed to be **column-major** and contiguous, which holds
//! for both backends. Accumulation helpers rely on this layout.

#[cfg(feature = "faer")]
mod faer_impl;
#[cfg(feature = "nalgebra")]
mod nalgebra_impl;

use crate::real::Real;
use num_traits::Float;

/// Compile-time (`Static<N>`) or runtime (`Dynamic`) dimension.
///
/// The associated types map a dimension to the active backend's vector and
/// matrix types, mirroring tinyopt's `traits::params_trait` dimension handling.
pub trait Dim: 'static + Copy + Default + core::fmt::Debug {
    /// `Some(n)` for static dimensions, `None` for dynamic ones.
    const VALUE: Option<usize>;

    /// Dense vector type of this dimension.
    type Vec<S: Real>: DenseVector<S>;
    /// Dense square matrix type of this dimension.
    type Mat<S: Real>: DenseMatrix<S>;

    /// Solve `A x = b` with a symmetric-indefinite (LDLT-family) factorization.
    fn solve_ldlt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>>;
    /// Solve `A x = b` with a Cholesky (LLT) factorization.
    fn solve_llt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>>;
    /// Solve `A x = b` with a pivoted LU factorization.
    fn solve_lu<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>>;
    /// Solve `A x = b` with a QR factorization.
    fn solve_qr<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>>;
    /// Solve `A x = b` with a singular value decomposition.
    fn solve_svd<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>>;
    /// Dense matrix inverse.
    fn inverse<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>>;
    /// Lower Cholesky factor `L` such that `A = L Lᵀ`.
    fn cholesky_lower<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>>;
}

/// Runtime-known dimension (C++ `Eigen::Dynamic`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dynamic;

/// Compile-time dimension known from the parameter type (C++ `Eigen::N`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Static<const N: usize>;

/// Dense vector of real scalars.
pub trait DenseVector<S: Real>: Clone + core::fmt::Debug + Send + Sync {
    /// Number of coefficients.
    fn len(&self) -> usize;

    /// All-zeros vector of length `len` (static types assert `len == N`).
    fn zeros(len: usize) -> Self;

    /// Coefficient accessor.
    fn at(&self, i: usize) -> S;
    /// Coefficient mutator.
    fn set_at(&mut self, i: usize, v: S);

    /// Contiguous read access.
    fn as_slice(&self) -> &[S];
    /// Contiguous write access.
    fn as_mut_slice(&mut self) -> &mut [S];

    /// Whether the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dot product.
    fn dot(&self, other: &Self) -> S {
        let (a, b) = (self.as_slice(), other.as_slice());
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b).fold(S::zero(), |acc, (x, y)| acc + *x * *y)
    }

    /// Squared Euclidean norm.
    fn norm2(&self) -> S {
        self.dot(self)
    }

    /// Euclidean norm.
    fn norm(&self) -> S {
        Float::sqrt(self.norm2())
    }

    /// `self += a * x`.
    fn axpy(&mut self, a: S, x: &Self) {
        let (lhs, rhs) = (self.as_mut_slice(), x.as_slice());
        debug_assert_eq!(lhs.len(), rhs.len());
        for (l, r) in lhs.iter_mut().zip(rhs) {
            *l += a * *r;
        }
    }

    /// `self *= a`.
    fn scale(&mut self, a: S) {
        for v in self.as_mut_slice() {
            *v *= a;
        }
    }

    /// Set all coefficients to zero.
    fn set_zero(&mut self) {
        for v in self.as_mut_slice() {
            *v = S::zero();
        }
    }

    /// Copy coefficients from `other`.
    fn copy_from(&mut self, other: &Self) {
        let (lhs, rhs) = (self.as_mut_slice(), other.as_slice());
        debug_assert_eq!(lhs.len(), rhs.len());
        lhs.copy_from_slice(rhs);
    }
}

/// Dense square matrix of real scalars.
///
/// Backends may pad the leading dimension (faer does), so matrices are always
/// accessed through [`DenseMatrix::elem`] / [`DenseMatrix::set_elem`] rather than a
/// contiguous slice.
pub trait DenseMatrix<S: Real>: Clone + core::fmt::Debug + Send + Sync {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;

    /// All-zeros matrix (static types assert the requested shape).
    fn zeros(nrows: usize, ncols: usize) -> Self;

    /// Coefficient accessor.
    fn elem(&self, i: usize, j: usize) -> S;
    /// Coefficient mutator.
    fn set_elem(&mut self, i: usize, j: usize, v: S);

    /// Set all coefficients to zero.
    fn set_zero(&mut self) {
        let (r, c) = (self.nrows(), self.ncols());
        for j in 0..c {
            for i in 0..r {
                self.set_elem(i, j, S::zero());
            }
        }
    }

    /// `self += a * x * yᵀ`, with `x` and `y` given as slices.
    ///
    /// This is the workhorse of the accumulation pattern (`H += Jᵀ J`,
    /// `grad += Jᵀ r`).
    fn add_outer_product(&mut self, a: S, x: &[S], y: &[S]) {
        let (r, c) = (self.nrows(), self.ncols());
        debug_assert_eq!(r, x.len());
        debug_assert_eq!(c, y.len());
        let _ = (r, c);
        for (j, yj) in y.iter().enumerate() {
            let ay = a * *yj;
            if ay == S::zero() {
                continue;
            }
            for (i, xi) in x.iter().enumerate() {
                let v = self.elem(i, j) + ay * *xi;
                self.set_elem(i, j, v);
            }
        }
    }

    /// `self += a * x * xᵀ`.
    fn add_outer_product_self(&mut self, a: S, x: &[S]) {
        self.add_outer_product(a, x, x);
    }

    /// Zero the given row and column (used to mask fixed parameters).
    fn zero_row_col(&mut self, index: usize) {
        let n = self.nrows().min(self.ncols());
        for k in 0..n {
            self.set_elem(index, k, S::zero());
            self.set_elem(k, index, S::zero());
        }
    }
}

/// Dense linear solver method (dense subset of tinyopt's `LinearSolverMethod`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LinearSolverMethod {
    /// Symmetric-indefinite LDLT-family factorization (C++ default).
    #[default]
    Ldlt,
    /// Cholesky (requires a positive-definite system).
    Llt,
    /// Pivoted LU.
    Lu,
    /// QR.
    Qr,
    /// Singular value decomposition (pseudo-inverse step).
    Svd,
    /// CHOLMOD sparse solver (mapped to the built-in sparse LDLT here).
    SuiteSparse,
    /// Incomplete Cholesky preconditioner (mapped to the built-in sparse LDLT).
    IncompleteCholesky,
}

/// Backend vector type for scalar `S` and dimension `D`.
pub type Vector<S, D> = <D as Dim>::Vec<S>;
/// Backend matrix type for scalar `S` and dimension `D`.
pub type Matrix<S, D> = <D as Dim>::Mat<S>;

/// All-zeros vector of the given length.
///
/// Helper that avoids the inherent/trait `zeros` ambiguity of the backends.
pub fn zeros_vector<S: Real, D: Dim>(len: usize) -> Vector<S, D> {
    <Vector<S, D> as DenseVector<S>>::zeros(len)
}

/// All-zeros matrix of the given shape.
pub fn zeros_matrix<S: Real, D: Dim>(nrows: usize, ncols: usize) -> Matrix<S, D> {
    <Matrix<S, D> as DenseMatrix<S>>::zeros(nrows, ncols)
}

/// Solve `A x = b` using `method`.
pub fn solve<S: Real, D: Dim>(
    method: LinearSolverMethod,
    a: &Matrix<S, D>,
    b: &Vector<S, D>,
) -> Option<Vector<S, D>> {
    // Closed-form small systems (mirrors the C++ DenseLinearSolvers fast path).
    if matches!(method, LinearSolverMethod::Ldlt | LinearSolverMethod::Llt) {
        let n = a.nrows();
        if n == a.ncols()
            && (1..=4).contains(&n)
            && let Some(x) = crate::math::solve_small_dense_system::<S, D>(a, b)
        {
            return Some(x);
        }
    }
    match method {
        LinearSolverMethod::Ldlt => D::solve_ldlt(a, b),
        LinearSolverMethod::Llt => D::solve_llt(a, b),
        LinearSolverMethod::Lu => D::solve_lu(a, b),
        LinearSolverMethod::Qr => D::solve_qr(a, b),
        LinearSolverMethod::Svd => D::solve_svd(a, b),
        // No dedicated sparse backend here: fall back to the dense LDLT.
        LinearSolverMethod::SuiteSparse | LinearSolverMethod::IncompleteCholesky => {
            D::solve_ldlt(a, b)
        }
    }
}

/// Dense matrix inverse.
pub fn inverse<S: Real, D: Dim>(a: &Matrix<S, D>) -> Option<Matrix<S, D>> {
    D::inverse(a)
}
