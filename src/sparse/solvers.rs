// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Scalar sparse linear solvers, including the column-streamed Schur path.

use crate::backend::{DenseMatrix, DenseVector, Dynamic, LinearSolverMethod, solve, zeros_vector};
use crate::options::LinearSolverOptions;
use crate::real::Real;
use crate::sparse::ldlt::SimplicialLdl;
use crate::sparse::matrix::SparseMatrix;

/// Sparse linear solver with optional Schur-complement partitioning.
///
/// Mirrors tinyopt's `SparseLinearSolvers`: the reduced block is the **head**
/// of the parameter vector (`0..num_reduced`) and the eliminated block is the
/// sparse tail.
#[derive(Clone, Debug)]
pub struct SparseLinearSolvers<S> {
    options: LinearSolverOptions,
    ldlt: SimplicialLdl<S>,
    analyzed: bool,
    computed: bool,
    num_reduced: usize,
}

impl<S: Real> Default for SparseLinearSolvers<S> {
    fn default() -> Self {
        Self::new(LinearSolverOptions::default())
    }
}

/// Symbolic-analysis cache for sparse solves.
///
/// The optimizer calls `solve_into` on every iteration with the same sparsity
/// pattern: keep the analyzed factorization and only refactor the values.
#[derive(Clone, Debug)]
pub struct SparseSolverCache<S> {
    solver: SparseLinearSolvers<S>,
    pattern: Option<(
        Vec<usize>,
        Vec<usize>,
        usize,
        crate::backend::LinearSolverMethod,
    )>,
}

impl<S: Real> Default for SparseSolverCache<S> {
    fn default() -> Self {
        Self {
            solver: SparseLinearSolvers::default(),
            pattern: None,
        }
    }
}

impl<S: Real> SparseSolverCache<S> {
    /// Solve with symbolic reuse when the pattern and options are unchanged.
    pub fn solve_into(
        &mut self,
        h: &SparseMatrix<S>,
        options: &LinearSolverOptions,
        rhs: &[S],
        num_reduced: usize,
        out: &mut [S],
    ) -> bool {
        let reusable = options.reuse_symbolic
            && match &self.pattern {
                Some((col_ptrs, row_indices, reduced, method)) => {
                    *reduced == num_reduced
                        && *method == options.method
                        && col_ptrs.as_slice() == h.col_ptrs()
                        && row_indices.as_slice() == h.row_indices()
                }
                None => false,
            };
        if !reusable {
            self.solver = SparseLinearSolvers::new(options.clone());
            if !self.solver.analyze(h, num_reduced) {
                self.pattern = None;
                return false;
            }
            self.pattern = Some((
                h.col_ptrs().to_vec(),
                h.row_indices().to_vec(),
                num_reduced,
                options.method,
            ));
        }
        if !self.solver.compute(h) {
            return false;
        }
        match self.solver.solve(h, rhs) {
            Some(x) => {
                out.copy_from_slice(&x);
                true
            }
            None => false,
        }
    }
}

impl<S: Real> SparseLinearSolvers<S> {
    /// Create a solver with the given options.
    pub fn new(options: LinearSolverOptions) -> Self {
        Self {
            options,
            ldlt: SimplicialLdl::new(),
            analyzed: false,
            computed: false,
            num_reduced: 0,
        }
    }

    /// Whether the configured method factorizes the dense conversion instead
    /// of the sparse LDLᵀ (LU, QR and SVD are dense-only, as in Eigen).
    fn uses_dense(&self) -> bool {
        matches!(
            self.options.method,
            LinearSolverMethod::Lu | LinearSolverMethod::Qr | LinearSolverMethod::Svd
        )
    }

    /// Analyze the sparsity pattern of `h` with `num_reduced` head parameters.
    pub fn analyze(&mut self, h: &SparseMatrix<S>, num_reduced: usize) -> bool {
        if h.nrows() != h.ncols() || h.nrows() == 0 || num_reduced >= h.nrows() {
            self.num_reduced = usize::MAX;
            self.analyzed = false;
            return false;
        }
        if num_reduced > 0 && !self.options.allow_schur {
            self.num_reduced = usize::MAX;
            self.analyzed = false;
            return false;
        }
        self.num_reduced = num_reduced;
        if self.uses_dense() {
            self.analyzed = true;
            self.computed = false;
            return true;
        }
        let eliminated = self.eliminated_block(h, num_reduced);
        self.analyzed = self
            .ldlt
            .analyze_pattern_with(&eliminated.to_lower(), self.options.ordering);
        self.computed = false;
        self.analyzed
    }

    /// Numeric factorization of the analyzed pattern.
    pub fn compute(&mut self, h: &SparseMatrix<S>) -> bool {
        if !self.analyzed {
            return false;
        }
        if self.uses_dense() {
            self.computed = true;
            return true;
        }
        let eliminated = self.eliminated_block(h, self.num_reduced);
        self.computed = self.ldlt.factorize(&eliminated.to_lower());
        self.computed
    }

    /// Solve `H x = rhs` using the cached factorization.
    pub fn solve(&self, h: &SparseMatrix<S>, rhs: &[S]) -> Option<Vec<S>> {
        if !self.computed || self.num_reduced == usize::MAX || rhs.len() != h.nrows() {
            return None;
        }
        let n = h.nrows();
        if self.uses_dense() {
            let dense = h.to_dense();
            let mut b = crate::backend::zeros_vector::<S, Dynamic>(n);
            for (i, v) in rhs.iter().enumerate() {
                b.set_at(i, *v);
            }
            let x = crate::backend::solve::<S, Dynamic>(self.options.method, &dense, &b)?;
            return Some((0..n).map(|i| x.at(i)).collect());
        }
        let nr = self.num_reduced;
        if nr == 0 {
            return self.ldlt.solve(rhs);
        }

        let ne = n - nr;
        // Eliminated-block solve of the tail.
        let y = self.ldlt.solve(&rhs[nr..])?;

        // C = H[0..nr, nr..n], C^T built once for column access.
        let c = h.submatrix(0, nr, nr, n);
        let ct = c.transpose();

        // Dense reduced block H_rr.
        let h_rr = h.submatrix(0, nr, 0, nr);
        let mut reduced = h_rr.to_dense();

        // Schur complement S = H_rr - C A_e^{-1} C^T.
        let mut rhs_e = vec![S::zero(); ne];
        for col in 0..nr {
            rhs_e.iter_mut().for_each(|v| *v = S::zero());
            for (e, v) in ct.column(col) {
                rhs_e[e] = v;
            }
            let w = self.ldlt.solve(&rhs_e)?;
            let cw = c.mul_vec(&w);
            for (r, cw_r) in cw.iter().enumerate().take(nr) {
                let value = reduced.elem(r, col) - *cw_r;
                reduced.set_elem(r, col, value);
            }
        }

        // Reduced right-hand side and solve.
        let cy = c.mul_vec(&y);
        let mut reduced_rhs = zeros_vector::<S, Dynamic>(nr);
        for r in 0..nr {
            reduced_rhs.set_at(r, rhs[r] - cy[r]);
        }
        let xr = solve::<S, Dynamic>(LinearSolverMethod::Ldlt, &reduced, &reduced_rhs)?;

        // Back-substitution of the eliminated block.
        let mut ct_xr_rhs = zeros_vector::<S, Dynamic>(ne);
        for e in 0..ne {
            let mut acc = S::zero();
            for (r, v) in c.column(e) {
                acc += v * xr.at(r);
            }
            ct_xr_rhs.set_at(e, acc);
        }
        let correction = self.ldlt.solve(ct_xr_rhs.as_slice())?;

        let mut x = vec![S::zero(); n];
        for (r, xr_r) in (0..nr).map(|r| (r, xr.at(r))) {
            x[r] = xr_r;
        }
        for (e, correction_e) in correction.iter().enumerate().take(ne) {
            x[nr + e] = y[e] - *correction_e;
        }

        if x.iter().all(|v| v.is_finite()) {
            Some(x)
        } else {
            None
        }
    }

    /// Dense inverse of `h` (as a sparse matrix with a dense pattern).
    pub fn inverse(&self, h: &SparseMatrix<S>) -> Option<SparseMatrix<S>> {
        let n = h.nrows();
        let mut triplets = Vec::with_capacity(n * n);
        let mut e = vec![S::zero(); n];
        for j in 0..n {
            e.iter_mut().for_each(|v| *v = S::zero());
            e[j] = S::one();
            let column = self.solve(h, &e)?;
            for (i, v) in column.into_iter().enumerate() {
                triplets.push((i, j, v));
            }
        }
        Some(SparseMatrix::from_triplets(n, n, &triplets))
    }

    /// Number of reduced (head) parameters.
    pub fn num_reduced(&self) -> usize {
        self.num_reduced
    }

    /// Whether the pattern was analyzed.
    pub fn is_analyzed(&self) -> bool {
        self.analyzed
    }

    /// Whether the numeric factorization is available.
    pub fn is_computed(&self) -> bool {
        self.computed
    }

    /// Current linear solver options.
    pub fn options(&self) -> &LinearSolverOptions {
        &self.options
    }

    fn eliminated_block(&self, h: &SparseMatrix<S>, num_reduced: usize) -> SparseMatrix<S> {
        let n = h.nrows();
        h.submatrix(num_reduced, n, num_reduced, n)
    }
}
