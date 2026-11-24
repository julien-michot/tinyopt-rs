// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! faer backend.
//!
//! faer's matrices are heap-allocated and runtime-sized (`Col`/`Mat`), so
//! static dimensions are represented by the same dynamic types, with their
//! length checked when a static dimension is requested.

use faer::linalg::solvers::{DenseSolveCore, Ldlt, Llt, PartialPivLu, Qr};
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

/// Scratch requirement for `pseudoinverse_from_svd_with_tolerance`.
pub(crate) fn pseudo_inverse_scratch<S: Real>(n: usize, par: Par) -> faer::dyn_stack::StackReq {
    faer::linalg::svd::pseudoinverse_from_svd_scratch::<S>(n, n, par)
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
        // Pseudo-inverse solve with a relative tolerance: faer's `solve_in_place`
        // divides by exactly zero singular values, producing NaNs on singular
        // systems.
        use faer::linalg::svd::{
            ComputeSvdVectors, pseudoinverse_from_svd_with_tolerance, svd, svd_scratch,
        };

        let n = a.nrows();
        if a.ncols() != n || b.nrows() != n {
            return None;
        }
        let par = Par::Seq;
        let mut s = faer::diag::Diag::<S>::zeros(n);
        let mut u = Mat::<S>::zeros(n, n);
        let mut v = Mat::<S>::zeros(n, n);
        let scratch = svd_scratch::<S>(
            n,
            n,
            ComputeSvdVectors::Full,
            ComputeSvdVectors::Full,
            par,
            Default::default(),
        );
        let pinv_scratch = crate::backend::faer_impl::pseudo_inverse_scratch::<S>(n, par);
        let mut mem = faer::dyn_stack::MemBuffer::new(scratch);
        {
            let stack = faer::dyn_stack::MemStack::new(&mut mem);
            let _ = svd(
                a.as_mat_ref(),
                s.as_mut(),
                Some(u.as_mut()),
                Some(v.as_mut()),
                par,
                stack,
                Default::default(),
            );
        }
        let mut pinv = Mat::<S>::zeros(n, n);
        let mut mem = faer::dyn_stack::MemBuffer::new(pinv_scratch);
        {
            let stack = faer::dyn_stack::MemStack::new(&mut mem);
            pseudoinverse_from_svd_with_tolerance(
                pinv.as_mut(),
                s.as_ref(),
                u.as_ref(),
                v.as_ref(),
                S::zero(),
                S::epsilon() * S::from_f64(n as f64).unwrap_or_else(S::one),
                par,
                stack,
            );
        }
        let mut out = Col::<S>::zeros(n);
        for i in 0..n {
            let mut acc = S::zero();
            for j in 0..n {
                acc += pinv[(i, j)] * b.as_slice()[j];
            }
            if !acc.is_finite() {
                return None;
            }
            out.as_mut_slice()[i] = acc;
        }
        Some(out)
    }

    fn inverse<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>> {
        let f = Ldlt::new(a.as_mat_ref(), Side::Lower).ok()?;
        Some(f.inverse())
    }

    fn cholesky_lower<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>> {
        let f = Llt::new(a.as_mat_ref(), Side::Lower).ok()?;
        let l = f.L();
        let mut out: Mat<S> = Mat::from_fn(l.nrows(), l.ncols(), |i, j| *l.get(i, j));
        // Keep only the lower triangle.
        for j in 0..out.ncols() {
            for i in 0..j {
                *out.get_mut(i, j) = S::zero();
            }
        }
        Some(out)
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

    fn cholesky_lower<S: Real>(a: &Self::Mat<S>) -> Option<Self::Mat<S>> {
        debug_assert_eq!(a.nrows(), N);
        Dynamic::cholesky_lower(a)
    }
}
