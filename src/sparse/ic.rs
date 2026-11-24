// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Incomplete Cholesky IC(0) factorization and preconditioned CG.

use crate::real::Real;
use crate::sparse::matrix::SparseMatrix;

/// Incomplete Cholesky factorization with the sparsity pattern of the lower
/// triangle of the input (IC(0)).
///
/// The factor `L` satisfies `L Lᵀ ≈ A` and is used as a preconditioner for the
/// conjugate-gradient solve selected by `LinearSolverMethod::IncompleteCholesky`.
#[derive(Clone, Debug)]
pub struct IncompleteCholesky<S> {
    n: usize,
    /// Lower-triangular factor in CSR (row-major), including the diagonal.
    row_ptr: Vec<usize>,
    row_col: Vec<usize>,
    row_val: Vec<S>,
    /// Lower-triangular factor in CSC (column-major), including the diagonal.
    col_ptr: Vec<usize>,
    col_row: Vec<usize>,
    col_val: Vec<S>,
    computed: bool,
}

impl<S: Real> Default for IncompleteCholesky<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Real> IncompleteCholesky<S> {
    /// Create an empty factorization.
    pub fn new() -> Self {
        Self {
            n: 0,
            row_ptr: vec![0],
            row_col: Vec::new(),
            row_val: Vec::new(),
            col_ptr: vec![0],
            col_row: Vec::new(),
            col_val: Vec::new(),
            computed: false,
        }
    }

    /// Whether a factorization is available.
    pub fn is_computed(&self) -> bool {
        self.computed
    }

    /// Matrix dimension.
    pub fn dims(&self) -> usize {
        self.n
    }

    /// Build the IC(0) factor of the lower triangle of `a`.
    ///
    /// Returns false when a pivot is not positive (the system is not SPD, or
    /// the incomplete factor broke down).
    pub fn compute(&mut self, a: &SparseMatrix<S>) -> bool {
        self.computed = false;
        let n = a.nrows();
        if n == 0 || a.ncols() != n {
            return false;
        }
        self.n = n;

        // Lower triangle in CSC (column j stores rows i >= j, already sorted).
        if self.col_ptr.len() != n + 1 {
            self.col_ptr.resize(n + 1, 0);
        }
        self.col_ptr.fill(0);
        for j in 0..n {
            for p in a.col_ptrs()[j]..a.col_ptrs()[j + 1] {
                if a.row_indices()[p] >= j {
                    self.col_ptr[j + 1] += 1;
                }
            }
        }
        for j in 0..n {
            self.col_ptr[j + 1] += self.col_ptr[j];
        }
        let nnz = self.col_ptr[n];
        self.col_row.clear();
        self.col_val.clear();
        for j in 0..n {
            for p in a.col_ptrs()[j]..a.col_ptrs()[j + 1] {
                let i = a.row_indices()[p];
                if i >= j {
                    self.col_row.push(i);
                    self.col_val.push(a.values()[p]);
                }
            }
        }

        // Row-major view (CSR) for the row-oriented IC recurrence.
        if self.row_ptr.len() != n + 1 {
            self.row_ptr.resize(n + 1, 0);
        }
        self.row_ptr.fill(0);
        for &i in &self.col_row {
            self.row_ptr[i + 1] += 1;
        }
        for i in 0..n {
            self.row_ptr[i + 1] += self.row_ptr[i];
        }
        self.row_col.clear();
        self.row_col.resize(nnz, 0);
        self.row_val.clear();
        self.row_val.resize(nnz, S::zero());
        let mut pos = self.row_ptr.clone();
        for j in 0..n {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.col_row[p];
                let q = pos[i];
                self.row_col[q] = j;
                self.row_val[q] = self.col_val[p];
                pos[i] += 1;
            }
        }

        // IC(0): process each row in column order.
        let mut diag = vec![S::zero(); n];
        for i in 0..n {
            let (lo, hi) = (self.row_ptr[i], self.row_ptr[i + 1]);
            let mut diagonal_seen = false;
            for p in lo..hi {
                let k = self.row_col[p];
                if k == i {
                    let mut sum = S::zero();
                    for &value in &self.row_val[lo..p] {
                        sum += value * value;
                    }
                    let pivot = self.row_val[p] - sum;
                    if pivot <= S::zero() || !pivot.is_finite() {
                        return false;
                    }
                    let value = num_traits::Float::sqrt(pivot);
                    self.row_val[p] = value;
                    diag[i] = value;
                    diagonal_seen = true;
                } else {
                    // (a_ik - Σ_{j<k} l_ij l_kj) / l_kk.
                    let mut sum = S::zero();
                    let (klo, khi) = (self.row_ptr[k], self.row_ptr[k + 1]);
                    let mut q = lo;
                    let mut r = klo;
                    while q < p && r < khi {
                        let (jq, jr) = (self.row_col[q], self.row_col[r]);
                        if jq == jr {
                            if jr < k {
                                sum += self.row_val[q] * self.row_val[r];
                            }
                            q += 1;
                            r += 1;
                        } else if jq < jr {
                            q += 1;
                        } else {
                            r += 1;
                        }
                    }
                    if diag[k] == S::zero() {
                        return false;
                    }
                    self.row_val[p] = (self.row_val[p] - sum) / diag[k];
                }
            }
            if !diagonal_seen {
                return false;
            }
        }

        // Rewrite the CSC values from the row values.
        let mut pos = self.col_ptr.clone();
        for i in 0..n {
            for p in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.row_col[p];
                self.col_val[pos[j]] = self.row_val[p];
                pos[j] += 1;
            }
        }

        self.computed = true;
        true
    }

    /// Apply the preconditioner: `out = (L Lᵀ)⁻¹ rhs`.
    pub fn apply(&self, rhs: &[S], out: &mut [S]) {
        debug_assert_eq!(rhs.len(), self.n);
        debug_assert_eq!(out.len(), self.n);
        // Forward substitution L y = rhs.
        for i in 0..self.n {
            let mut value = rhs[i];
            let (lo, hi) = (self.row_ptr[i], self.row_ptr[i + 1]);
            for p in lo..hi {
                let j = self.row_col[p];
                if j < i {
                    value -= self.row_val[p] * out[j];
                }
            }
            debug_assert!(hi > lo && self.row_col[hi - 1] == i);
            out[i] = value / self.row_val[hi - 1];
        }
        // Backward substitution Lᵀ x = y.
        for i in (0..self.n).rev() {
            let mut value = out[i];
            for p in self.col_ptr[i]..self.col_ptr[i + 1] {
                let j = self.col_row[p];
                if j > i {
                    value -= self.col_val[p] * out[j];
                }
            }
            let diagonal = self.col_val[self.col_ptr[i]];
            out[i] = value / diagonal;
        }
    }
}

/// Symmetric matrix-vector product from its lower-triangular CSC storage.
pub(crate) fn symmetric_mul_vec_into<S: Real>(lower: &SparseMatrix<S>, x: &[S], out: &mut [S]) {
    out.iter_mut().for_each(|v| *v = S::zero());
    let n = lower.ncols();
    for j in 0..n {
        for p in lower.col_ptrs()[j]..lower.col_ptrs()[j + 1] {
            let i = lower.row_indices()[p];
            let value = lower.values()[p];
            out[i] += value * x[j];
            if i != j {
                out[j] += value * x[i];
            }
        }
    }
}
