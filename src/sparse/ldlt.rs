// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Simplicial LDLᵀ factorization for symmetric sparse matrices.
//!
//! The input is the **lower triangle** (row ≥ column) of a symmetric matrix in
//! CSC form. The implementation is a left-looking LDLᵀ with a symbolic
//! elimination-tree analysis. A fill-reducing minimum-degree ordering (AMD
//! family) can be applied; natural ordering is also available.
//!
//! Ordering `P` is applied symmetrically (`P A Pᵀ`); solving permutes the
//! right-hand side into the elimination order and unpermutes the result.

use crate::options::Ordering;
use crate::real::Real;
use crate::sparse::SparseMatrix;

/// Symbolic + numeric LDLᵀ factorization of a symmetric sparse matrix.
#[derive(Clone, Debug)]
pub struct SimplicialLdl<S> {
    n: usize,
    /// `perm[k]` is the original index eliminated at step `k`.
    perm: Vec<usize>,
    /// `inv_perm[old]` is the elimination position of an original index.
    inv_perm: Vec<usize>,
    permuted: bool,
    /// Permuted lower-triangle pattern (CSC) of `P A Pᵀ`.
    work_ptr: Vec<usize>,
    work_idx: Vec<usize>,
    /// `dest[p]` is the position of original value `p` in the permuted values.
    dest: Vec<usize>,
    parent: Vec<isize>,
    lp: Vec<usize>,
    li: Vec<usize>,
    lx: Vec<S>,
    d: Vec<S>,
    perm_values: Vec<S>,
    row_entries: Vec<Vec<(usize, usize)>>,
    analyzed: bool,
    computed: bool,
}

impl<S: Real> Default for SimplicialLdl<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Real> SimplicialLdl<S> {
    /// Create an empty factorization.
    pub fn new() -> Self {
        Self {
            n: 0,
            perm: Vec::new(),
            inv_perm: Vec::new(),
            permuted: false,
            work_ptr: vec![0],
            work_idx: Vec::new(),
            dest: Vec::new(),
            parent: Vec::new(),
            lp: vec![0],
            li: Vec::new(),
            lx: Vec::new(),
            d: Vec::new(),
            perm_values: Vec::new(),
            row_entries: Vec::new(),
            analyzed: false,
            computed: false,
        }
    }

    /// Analyze the pattern with the natural ordering.
    pub fn analyze_pattern(&mut self, lower: &SparseMatrix<S>) -> bool {
        self.analyze_pattern_with(lower, Ordering::Natural)
    }

    /// Analyze the pattern with the given fill-reducing ordering.
    ///
    /// `lower` must be square and contain the lower triangle.
    pub fn analyze_pattern_with(&mut self, lower: &SparseMatrix<S>, ordering: Ordering) -> bool {
        if lower.nrows() != lower.ncols() {
            return false;
        }
        let n = lower.nrows();
        self.n = n;
        self.analyzed = false;
        self.computed = false;

        // Fill-reducing permutation.
        match ordering {
            Ordering::Natural => self.perm = (0..n).collect(),
            Ordering::Amd => {
                self.perm = crate::sparse::minimum_degree(n, lower.col_ptrs(), lower.row_indices());
            }
        }
        self.inv_perm = vec![0usize; n];
        for (k, &v) in self.perm.iter().enumerate() {
            self.inv_perm[v] = k;
        }
        self.permuted = (0..n).any(|i| self.perm[i] != i);

        // Permuted lower pattern and the value map used by `factorize`.
        let nnz = lower.non_zeros();
        if self.permuted {
            let mut targets: Vec<(usize, usize, usize)> = Vec::with_capacity(nnz);
            for j in 0..n {
                for p in lower.col_ptrs()[j]..lower.col_ptrs()[j + 1] {
                    let i = lower.row_indices()[p];
                    let (pi, pj) = (self.inv_perm[i], self.inv_perm[j]);
                    let (r, c) = if pi >= pj { (pi, pj) } else { (pj, pi) };
                    targets.push((c, r, p));
                }
            }
            targets.sort_unstable();
            self.dest = vec![0; nnz];
            self.work_ptr = vec![0; n + 1];
            self.work_idx = Vec::with_capacity(nnz);
            for (q, &(c, r, p)) in targets.iter().enumerate() {
                self.dest[p] = q;
                self.work_ptr[c + 1] += 1;
                self.work_idx.push(r);
            }
            for j in 0..n {
                self.work_ptr[j + 1] += self.work_ptr[j];
            }
        } else {
            self.dest.clear();
            self.work_ptr = lower.col_ptrs().to_vec();
            self.work_idx = lower.row_indices().to_vec();
        }

        // Upper-triangle CSC of the permuted pattern, i.e. the transpose of the
        // stored lower triangle, used by the elimination-tree analysis.
        let mut upper_ptr = vec![0usize; n + 1];
        for &r in &self.work_idx {
            upper_ptr[r + 1] += 1;
        }
        for k in 0..n {
            upper_ptr[k + 1] += upper_ptr[k];
        }
        let mut upper_idx = vec![0usize; upper_ptr[n]];
        let mut pos = upper_ptr.clone();
        for c in 0..n {
            for &r in &self.work_idx[self.work_ptr[c]..self.work_ptr[c + 1]] {
                upper_idx[pos[r]] = c;
                pos[r] += 1;
            }
        }

        // Elimination tree (cs_etree).
        let mut parent = vec![-1isize; n];
        let mut flag: Vec<usize> = (0..n).collect();
        for k in 0..n {
            for &i in &upper_idx[upper_ptr[k]..upper_ptr[k + 1]] {
                if i < k {
                    let mut node = i;
                    while flag[node] != k {
                        if parent[node] < 0 {
                            parent[node] = k as isize;
                        }
                        flag[node] = k;
                        node = parent[node] as usize;
                    }
                }
            }
        }
        self.parent = parent;

        // Symbolic pattern of L. The etree reach gives the pattern of each
        // *row* (Davis, cs_ereach); by symmetry of the filled structure, column
        // j of L holds all rows k whose row pattern contains j. This captures
        // fill-in induced by earlier eliminations, unlike a lower-triangle walk.
        let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut mark = vec![usize::MAX; n];
        for k in 0..n {
            mark[k] = k;
            for &i in &upper_idx[upper_ptr[k]..upper_ptr[k + 1]] {
                if i >= k {
                    continue; // only the strictly upper triangle contributes
                }
                let mut node = i;
                while mark[node] != k {
                    debug_assert!(node < k, "etree reach must stay below the pivot");
                    mark[node] = k;
                    col_rows[node].push(k);
                    match self.parent[node] {
                        p if p >= 0 => node = p as usize,
                        _ => break,
                    }
                }
            }
        }
        let mut lp = vec![0usize; n + 1];
        let mut li: Vec<usize> = Vec::new();
        for (j, rows) in col_rows.iter_mut().enumerate() {
            rows.sort_unstable();
            li.extend_from_slice(rows);
            lp[j + 1] = li.len();
        }
        self.lp = lp;
        self.li = li;
        self.lx = vec![S::zero(); self.li.len()];
        self.d = vec![S::zero(); n];
        self.row_entries = vec![Vec::new(); n];
        for k in 0..n {
            for p in self.lp[k]..self.lp[k + 1] {
                let r = self.li[p];
                self.row_entries[r].push((k, p));
            }
        }
        self.analyzed = true;
        true
    }

    /// Numeric factorization, reusing the symbolic analysis.
    pub fn factorize(&mut self, lower: &SparseMatrix<S>) -> bool {
        if !self.analyzed || lower.nrows() != self.n || lower.ncols() != self.n {
            return false;
        }
        self.computed = false;
        let n = self.n;

        // Values in elimination order; the pattern is reused from the analysis.
        let (ptr, idx, values): (&[usize], &[usize], &[S]) = if self.permuted {
            if lower.non_zeros() != self.dest.len() {
                return false;
            }
            self.perm_values.clear();
            self.perm_values.resize(self.work_idx.len(), S::zero());
            for (p, &q) in self.dest.iter().enumerate() {
                self.perm_values[q] = lower.values()[p];
            }
            (&self.work_ptr, &self.work_idx, &self.perm_values)
        } else {
            (lower.col_ptrs(), lower.row_indices(), lower.values())
        };

        let mut y = vec![S::zero(); n];
        let mut stamp = vec![0u32; n];
        let mut touched: Vec<usize> = Vec::with_capacity(n);
        let mut current = 0u32;

        for j in 0..n {
            current = current.wrapping_add(1);
            if current == 0 {
                stamp.fill(0);
                current = 1;
            }
            touched.clear();

            // Scatter A(:, j) (rows >= j) into the workspace.
            for q in ptr[j]..ptr[j + 1] {
                let r = idx[q];
                if stamp[r] != current {
                    stamp[r] = current;
                    touched.push(r);
                }
                y[r] += values[q];
            }

            // Remove previously computed columns: y -= L(j,k) * D[k] * L(:,k).
            for &(k, p_k) in &self.row_entries[j] {
                let l_jk = self.lx[p_k];
                if l_jk == S::zero() {
                    continue;
                }
                let coeff = l_jk * self.d[k];
                for q in self.lp[k]..self.lp[k + 1] {
                    let r = self.li[q];
                    if stamp[r] != current {
                        stamp[r] = current;
                        touched.push(r);
                    }
                    y[r] -= coeff * self.lx[q];
                }
            }

            let dj = y[j];
            y[j] = S::zero();
            if !dj.is_finite() || dj <= S::zero() {
                for &r in &touched {
                    y[r] = S::zero();
                }
                return false;
            }
            self.d[j] = dj;

            // L(:, j) = y / D[j] on the symbolic pattern.
            let inv_d = S::one() / dj;
            for q in self.lp[j]..self.lp[j + 1] {
                let r = self.li[q];
                let value = if stamp[r] == current { y[r] } else { S::zero() };
                let l = value * inv_d;
                if !l.is_finite() {
                    for &rr in &touched {
                        y[rr] = S::zero();
                    }
                    return false;
                }
                self.lx[q] = l;
            }

            for &r in &touched {
                y[r] = S::zero();
            }
        }

        self.computed = true;
        true
    }

    /// Analyze and factorize with the natural ordering.
    pub fn analyze_and_factorize(&mut self, lower: &SparseMatrix<S>) -> bool {
        self.analyze_pattern(lower) && self.factorize(lower)
    }

    /// Analyze and factorize with the given ordering.
    pub fn analyze_and_factorize_with(
        &mut self,
        lower: &SparseMatrix<S>,
        ordering: Ordering,
    ) -> bool {
        self.analyze_pattern_with(lower, ordering) && self.factorize(lower)
    }

    /// Solve `A x = rhs` using the cached factorization.
    pub fn solve(&self, rhs: &[S]) -> Option<Vec<S>> {
        let mut out = vec![S::zero(); self.n];
        if self.solve_into(rhs, &mut out) {
            Some(out)
        } else {
            None
        }
    }

    /// Solve into a caller-provided buffer (no allocation).
    pub fn solve_into(&self, rhs: &[S], out: &mut [S]) -> bool {
        if !self.computed || rhs.len() != self.n || out.len() != self.n {
            return false;
        }
        let n = self.n;
        // Permute the right-hand side into the elimination order.
        let mut x = vec![S::zero(); n];
        for (i, v) in rhs.iter().enumerate() {
            x[self.inv_perm[i]] = *v;
        }

        // Forward substitution: L y = b (unit lower triangular).
        for j in 0..n {
            let xj = x[j];
            if xj == S::zero() {
                continue;
            }
            for p in self.lp[j]..self.lp[j + 1] {
                x[self.li[p]] -= self.lx[p] * xj;
            }
        }
        // Diagonal scaling.
        for (xi, di) in x.iter_mut().zip(&self.d) {
            *xi /= *di;
        }
        // Backward substitution: Lᵀ x = z.
        for j in (0..n).rev() {
            let mut acc = x[j];
            for p in self.lp[j]..self.lp[j + 1] {
                acc -= self.lx[p] * x[self.li[p]];
            }
            x[j] = acc;
        }

        // Unpermute the solution back to the original ordering.
        for (i, v) in x.iter().enumerate() {
            out[self.perm[i]] = *v;
        }

        out.iter().all(|v| v.is_finite())
    }

    /// Dimension of the factorization.
    pub fn dims(&self) -> usize {
        self.n
    }

    /// Whether the symbolic analysis was performed.
    pub fn is_analyzed(&self) -> bool {
        self.analyzed
    }

    /// Whether the numeric factorization is available.
    pub fn is_computed(&self) -> bool {
        self.computed
    }

    /// Diagonal factor `D`.
    pub fn diagonal(&self) -> &[S] {
        &self.d
    }

    /// Number of entries in the `L` factor (fill-in measure).
    pub fn fill_in(&self) -> usize {
        self.li.len()
    }

    /// Permutation applied by the factorization (`perm[k]` is the original
    /// index of the `k`-th pivot).
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DenseVector, Dynamic, LinearSolverMethod};
    use approx::assert_relative_eq;

    fn spd(n: usize) -> SparseMatrix<f64> {
        // Tridiagonal SPD: 4 on the diagonal, -1 off-diagonal.
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 4.0));
            if i + 1 < n {
                triplets.push((i + 1, i, -1.0));
                triplets.push((i, i + 1, -1.0));
            }
        }
        SparseMatrix::from_triplets(n, n, &triplets)
    }

    fn dense_solve(a: &SparseMatrix<f64>, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let m = a.to_dense();
        let mut rhs = crate::backend::zeros_vector::<f64, Dynamic>(n);
        for (i, v) in b.iter().enumerate() {
            rhs.set_at(i, *v);
        }
        let x = crate::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &m, &rhs)
            .expect("dense solve");
        (0..n).map(|i| x.at(i)).collect()
    }

    #[test]
    fn solves_spd_system() {
        let a = spd(6);
        let lower = a.to_lower();
        let mut ldlt = SimplicialLdl::new();
        assert!(ldlt.analyze_and_factorize(&lower));

        let b = [1.0, 2.0, -1.0, 0.5, 0.0, 3.0];
        let x = ldlt.solve(&b).expect("solve");
        let expected = dense_solve(&a, &b);
        for (a, e) in x.iter().zip(&expected) {
            assert_relative_eq!(a, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn solves_with_amd_ordering() {
        let a = spd(8);
        let lower = a.to_lower();
        let mut ldlt = SimplicialLdl::new();
        assert!(ldlt.analyze_and_factorize_with(&lower, Ordering::Amd));
        let b: Vec<f64> = (0..8).map(|i| i as f64 - 3.0).collect();
        let x = ldlt.solve(&b).expect("solve");
        let expected = dense_solve(&a, &b);
        for (a, e) in x.iter().zip(&expected) {
            assert_relative_eq!(a, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn diagonal_and_identity() {
        let a = spd(4);
        let lower = a.to_lower();
        let mut ldlt = SimplicialLdl::new();
        assert!(ldlt.analyze_and_factorize(&lower));
        assert_eq!(ldlt.dims(), 4);
        let mut rhs = vec![0.0; 4];
        rhs[0] = 1.0;
        let x = ldlt.solve(&rhs).expect("solve");
        let expected = dense_solve(&a, &rhs);
        for (a, e) in x.iter().zip(&expected) {
            assert_relative_eq!(a, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn rejects_non_positive_definite() {
        let m = SparseMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, -2.0)]);
        let mut ldlt = SimplicialLdl::new();
        assert!(!ldlt.analyze_and_factorize(&m.to_lower()));
    }
}
