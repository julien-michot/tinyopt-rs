// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! nalgebra backend: static (`SVector`/`SMatrix`) and dynamic (`DVector`/`DMatrix`) sizes.

use nalgebra::{
    DMatrix, DVector, SMatrix, SVector,
    linalg::{Cholesky, FullPivLU, LBLT, QR, SVD},
};

use super::{DenseMatrix, DenseVector, Dim, Dynamic, Static};
use crate::real::Real;
use num_traits::Float;

impl<S: Real, const N: usize> DenseVector<S> for SVector<S, N> {
    fn len(&self) -> usize {
        N
    }

    fn zeros(len: usize) -> Self {
        assert_eq!(len, N, "static vector dimension mismatch");
        SVector::zeros()
    }

    fn at(&self, i: usize) -> S {
        self[i]
    }

    fn set_at(&mut self, i: usize, v: S) {
        self[i] = v;
    }

    fn as_slice(&self) -> &[S] {
        nalgebra::Matrix::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [S] {
        nalgebra::Matrix::as_mut_slice(self)
    }
}

impl<S: Real> DenseVector<S> for DVector<S> {
    fn len(&self) -> usize {
        nalgebra::Matrix::nrows(self)
    }

    fn zeros(len: usize) -> Self {
        DVector::zeros(len)
    }

    fn at(&self, i: usize) -> S {
        self[i]
    }

    fn set_at(&mut self, i: usize, v: S) {
        self[i] = v;
    }

    fn as_slice(&self) -> &[S] {
        nalgebra::Matrix::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [S] {
        nalgebra::Matrix::as_mut_slice(self)
    }
}

impl<S: Real, const N: usize> DenseMatrix<S> for SMatrix<S, N, N> {
    fn nrows(&self) -> usize {
        N
    }

    fn ncols(&self) -> usize {
        N
    }

    fn zeros(nrows: usize, ncols: usize) -> Self {
        assert_eq!((nrows, ncols), (N, N), "static matrix dimension mismatch");
        SMatrix::zeros()
    }

    fn elem(&self, i: usize, j: usize) -> S {
        self[(i, j)]
    }

    fn set_elem(&mut self, i: usize, j: usize, v: S) {
        self[(i, j)] = v;
    }
}

impl<S: Real> DenseMatrix<S> for DMatrix<S> {
    fn nrows(&self) -> usize {
        nalgebra::Matrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        nalgebra::Matrix::ncols(self)
    }

    fn zeros(nrows: usize, ncols: usize) -> Self {
        DMatrix::zeros(nrows, ncols)
    }

    fn elem(&self, i: usize, j: usize) -> S {
        self[(i, j)]
    }

    fn set_elem(&mut self, i: usize, j: usize, v: S) {
        self[(i, j)] = v;
    }
}

/// Relative threshold used by the SVD pseudo-inverse step.
fn svd_eps<S: Real, M: DenseMatrix<S>>(a: &M) -> S {
    // The largest singular value is bounded by the Frobenius norm; using the
    // Frobenius norm here keeps this helper backend-agnostic and allocation-free.
    let mut max_sv = S::zero();
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let s = Float::abs(a.elem(i, j));
            if s > max_sv {
                max_sv = s;
            }
        }
    }
    max_sv * S::from_f64(1e-12).expect("1e-12 fits in scalar")
}

impl Dim for Dynamic {
    const VALUE: Option<usize> = None;

    type Vec<S: Real> = DVector<S>;
    type Mat<S: Real> = DMatrix<S>;

    fn solve_ldlt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        LBLT::new(a.clone()).solve(b)
    }

    fn solve_llt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        Some(Cholesky::new(a.clone())?.solve(b))
    }

    fn solve_lu<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        FullPivLU::new(a.clone()).solve(b)
    }

    fn solve_qr<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        QR::new(a.clone()).solve(b)
    }

    fn solve_svd<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        SVD::new(a.clone(), true, true).solve(b, svd_eps(a)).ok()
    }

    fn inverse<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>> {
        a.clone().try_inverse()
    }

    fn cholesky_lower<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>> {
        let chol = Cholesky::new(a.clone())?;
        Some(chol.l())
    }
}

impl<const N: usize> Dim for Static<N> {
    const VALUE: Option<usize> = Some(N);

    type Vec<S: Real> = SVector<S, N>;
    type Mat<S: Real> = SMatrix<S, N, N>;

    fn solve_ldlt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        LBLT::new(*a).solve(b)
    }

    fn solve_llt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        Some(Cholesky::new(*a)?.solve(b))
    }

    fn solve_lu<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        // nalgebra 0.35's QR/SVD/LU `solve` requires typenum dimensions, which
        // generic `Const<N>` does not implement; use a dynamic view instead.
        let ad = DMatrix::from_column_slice(N, N, nalgebra::Matrix::as_slice(a));
        let bd = DVector::from_column_slice(nalgebra::Matrix::as_slice(b));
        FullPivLU::new(ad)
            .solve(&bd)
            .map(|x| SVector::from_column_slice(x.as_slice()))
    }

    fn solve_qr<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        let ad = DMatrix::from_column_slice(N, N, nalgebra::Matrix::as_slice(a));
        let bd = DVector::from_column_slice(nalgebra::Matrix::as_slice(b));
        QR::new(ad)
            .solve(&bd)
            .map(|x| SVector::from_column_slice(x.as_slice()))
    }

    fn solve_svd<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        let ad = DMatrix::from_column_slice(N, N, nalgebra::Matrix::as_slice(a));
        let bd = DVector::from_column_slice(nalgebra::Matrix::as_slice(b));
        SVD::new(ad.clone(), true, true)
            .solve(&bd, svd_eps(&ad))
            .ok()
            .map(|x| SVector::from_column_slice(x.as_slice()))
    }

    fn inverse<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>> {
        a.try_inverse()
    }

    fn cholesky_lower<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>> {
        let chol = Cholesky::new(*a)?;
        Some(chol.l())
    }
}
