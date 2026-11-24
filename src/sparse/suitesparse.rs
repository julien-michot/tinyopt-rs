// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Optional CHOLMOD backend (`suitesparse` feature), mirroring the C++
//! `math/suitesparse.h` wrapper.

use std::ffi::c_void;
use std::marker::PhantomData;

use crate::real::Real;
use crate::sparse::SparseMatrix;

unsafe extern "C" {
    fn tinyopt_cholmod_create(single: i32) -> *mut c_void;
    fn tinyopt_cholmod_destroy(ctx: *mut c_void);
    fn tinyopt_cholmod_analyze(
        ctx: *mut c_void,
        n: i32,
        col_ptr: *const i64,
        row_idx: *const i64,
        nnz: usize,
    ) -> i32;
    fn tinyopt_cholmod_factorize(ctx: *mut c_void, values: *const c_void) -> i32;
    fn tinyopt_cholmod_solve(ctx: *mut c_void, rhs: *const c_void, out: *mut c_void) -> i32;
}

/// CHOLMOD factorization of a symmetric positive-definite sparse matrix.
pub struct CholmodSolver<S> {
    ctx: *mut c_void,
    n: usize,
    single: bool,
    _marker: PhantomData<S>,
}

impl<S> core::fmt::Debug for CholmodSolver<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CholmodSolver")
            .field("n", &self.n)
            .field("single", &self.single)
            .finish()
    }
}

impl<S> Clone for CholmodSolver<S> {
    fn clone(&self) -> Self {
        // The symbolic/numeric factors are rebuilt by the next `analyze`.
        let ctx = unsafe { tinyopt_cholmod_create(self.single as i32) };
        Self {
            ctx,
            n: 0,
            single: self.single,
            _marker: PhantomData,
        }
    }
}

impl<S> Drop for CholmodSolver<S> {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { tinyopt_cholmod_destroy(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

impl<S> CholmodSolver<S> {
    /// Create a CHOLMOD context (`None` if the library could not start).
    pub fn new() -> Option<Self> {
        let single = core::mem::size_of::<S>() == 4;
        let ctx = unsafe { tinyopt_cholmod_create(single as i32) };
        if ctx.is_null() {
            None
        } else {
            Some(Self {
                ctx,
                n: 0,
                single,
                _marker: PhantomData,
            })
        }
    }
}

impl<S: Real> CholmodSolver<S> {
    /// Symbolic analysis of the lower triangle of `h`.
    pub fn analyze(&mut self, h: &SparseMatrix<S>) -> bool {
        if self.ctx.is_null() || h.nrows() != h.ncols() || h.nrows() == 0 {
            return false;
        }
        let n = h.nrows();
        // CHOLMOD expects the lower triangle of a symmetric matrix.
        let lower = h.to_lower();
        if n > i32::MAX as usize || lower.non_zeros() > i32::MAX as usize {
            return false;
        }
        // `usize` and `i64` share the layout on 64-bit targets; CHOLMOD copies
        // the indices before we return.
        let ok = unsafe {
            tinyopt_cholmod_analyze(
                self.ctx,
                n as i32,
                lower.col_ptrs().as_ptr() as *const i64,
                lower.row_indices().as_ptr() as *const i64,
                lower.non_zeros(),
            )
        };
        self.n = n;
        ok != 0
    }

    /// Numeric factorization with the analyzed pattern.
    pub fn factorize(&self, h: &SparseMatrix<S>) -> bool {
        let lower = h.to_lower();
        if self.ctx.is_null() || h.nrows() != self.n || lower.values().len() != lower.non_zeros() {
            return false;
        }
        unsafe {
            tinyopt_cholmod_factorize(self.ctx, lower.values().as_ptr() as *const c_void) != 0
        }
    }

    /// Solve `A x = rhs` into `out`.
    pub fn solve(&self, rhs: &[S], out: &mut [S]) -> bool {
        if self.ctx.is_null() || self.n == 0 || rhs.len() != self.n || out.len() != self.n {
            return false;
        }
        unsafe {
            tinyopt_cholmod_solve(
                self.ctx,
                rhs.as_ptr() as *const c_void,
                out.as_mut_ptr() as *mut c_void,
            ) != 0
        }
    }
}
