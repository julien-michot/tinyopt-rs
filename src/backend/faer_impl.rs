// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! faer backend.
//!
//! faer's matrices are heap-allocated and runtime-sized (`Col`/`Mat`), so
//! static dimensions are represented by the same dynamic types, with their
//! length checked when a static dimension is requested.

use faer::linalg::solvers::{DenseSolveCore, Ldlt, Llt, PartialPivLu, Qr, Svd};
use faer::mat::AsMatRef;
use faer::prelude::*;
use faer::{Col, Mat, Side};

use super::{DenseMatrix, DenseVector, Dim, Dynamic, Static};
use crate::real::Real;

impl<S: Real> DenseVector<S> for Col<S> {
    fn len(&self) -> usize {
        self.nrows()
    }

    fn zeros(len: usize) -> Self {
        Col::from_fn(len, |_| S::zero())
    }

    fn at(&self, i: usize) -> S {
        self.as_slice()[i]
    }

    fn set_at(&mut self, i: usize, v: S) {
        self.as_mut_slice()[i] = v;
    }

    fn as_slice(&self) -> &[S] {
        // Owned `Col` is contiguous with a row stride of 1 (column-major),
        // as documented by faer.
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.nrows()) }
    }

    fn as_mut_slice(&mut self) -> &mut [S] {
        let n = self.nrows();
        // SAFETY: same layout guarantee as `as_slice`, borrowed mutably.
        unsafe { core::slice::from_raw_parts_mut(self.as_ptr_mut(), n) }
    }
}

impl<S: Real> DenseMatrix<S> for Mat<S> {
    fn nrows(&self) -> usize {
        self.shape().0
    }

    fn ncols(&self) -> usize {
        self.shape().1
    }

    fn zeros(nrows: usize, ncols: usize) -> Self {
        Mat::from_fn(nrows, ncols, |_, _| S::zero())
    }

    fn elem(&self, i: usize, j: usize) -> S {
        // faer pads the leading dimension (`col_stride` may exceed `nrows`),
        // so branches must go through `get`/`get_mut` rather than a slice.
        *self.get(i, j)
    }

    fn set_elem(&mut self, i: usize, j: usize, v: S) {
        *self.get_mut(i, j) = v;
    }
}

impl Dim for Dynamic {
    const VALUE: Option<usize> = None;

    type Vec<S: Real> = Col<S>;
    type Mat<S: Real> = Mat<S>;

    fn solve_ldlt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        let f = Ldlt::new(a.as_mat_ref(), Side::Lower).ok()?;
        let mut out = b.clone();
        f.solve_in_place(out.as_mat_mut());
        Some(out)
    }

    fn solve_llt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        let f = Llt::new(a.as_mat_ref(), Side::Lower).ok()?;
        let mut out = b.clone();
        f.solve_in_place(out.as_mat_mut());
        Some(out)
    }

    fn solve_lu<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        let f = PartialPivLu::new(a.as_mat_ref());
        let mut out = b.clone();
        f.solve_in_place(out.as_mat_mut());
        Some(out)
    }

    fn solve_qr<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        let f = Qr::new(a.as_mat_ref());
        let mut out = b.clone();
        f.solve_in_place(out.as_mat_mut());
        Some(out)
    }

    fn solve_svd<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        let f = Svd::new(a.as_mat_ref()).ok()?;
        let mut out = b.clone();
        f.solve_in_place(out.as_mat_mut());
        Some(out)
    }

    fn inverse<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>> {
        let f = Ldlt::new(a.as_mat_ref(), Side::Lower).ok()?;
        Some(f.inverse())
    }
}

impl<const N: usize> Dim for Static<N> {
    const VALUE: Option<usize> = Some(N);

    type Vec<S: Real> = Col<S>;
    type Mat<S: Real> = Mat<S>;

    fn solve_ldlt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        debug_assert_eq!(a.nrows(), N);
        Dynamic::solve_ldlt(a, b)
    }

    fn solve_llt<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        debug_assert_eq!(a.nrows(), N);
        Dynamic::solve_llt(a, b)
    }

    fn solve_lu<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        debug_assert_eq!(a.nrows(), N);
        Dynamic::solve_lu(a, b)
    }

    fn solve_qr<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        debug_assert_eq!(a.nrows(), N);
        Dynamic::solve_qr(a, b)
    }

    fn solve_svd<S: Real>(a: &Self::Mat<S>, b: &Self::Vec<S>) -> Option<Self::Vec<S>> {
        debug_assert_eq!(a.nrows(), N);
        Dynamic::solve_svd(a, b)
    }

    fn inverse<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>> {
        debug_assert_eq!(a.nrows(), N);
        Dynamic::inverse(a)
    }
}
