// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Simplicial LDLᵀ factorization for symmetric sparse matrices.
//!
//! The input is the **lower triangle** (row ≥ column) of a symmetric matrix in
//! CSC form. The implementation is a left-looking LDLᵀ with a symbolic
//! elimination-tree analysis and natural ordering.

use crate::real::Real;
use crate::sparse::SparseMatrix;

/// Symbolic + numeric LDLᵀ factorization of a symmetric sparse matrix.
#[derive(Clone, Debug)]
pub struct SimplicialLdl<S> {
    n: usize,
    parent: Vec<isize>,
    lp: Vec<usize>,
    li: Vec<usize>,
    lx: Vec<S>,
    d: Vec<S>,
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
            parent: Vec::new(),
            lp: vec![0],
            li: Vec::new(),
            lx: Vec::new(),
            d: Vec::new(),
            row_entries: Vec::new(),
            analyzed: false,
            computed: false,
        }
    }

    /// Compute the elimination tree and the symbolic pattern of `L`.
    ///
    /// `lower` must be square and contain the lower triangle.
    pub fn analyze_pattern(&mut self, lower: &SparseMatrix<S>) -> bool {
        if lower.nrows() != lower.ncols() {
            return false;
        }
        let n = lower.nrows();
        self.n = n;
        self.analyzed = false;
        self.computed = false;

        // Elimination tree (cs_etree) on the upper triangle.
        let upper = lower.transpose();
        let mut parent = vec![-1isize; n];
        let mut flag: Vec<usize> = (0..n).collect();
        for k in 0..n {
            for (i, _) in upper.column(k) {
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

        // Symbolic pattern of L by column (rows > column), via etree closure.
        let mut lp = vec![0usize; n + 1];
        let mut li: Vec<usize> = Vec::new();
        let mut mark = vec![usize::MAX; n];
        let mut pattern: Vec<usize> = Vec::new();
        for j in 0..n {
            pattern.clear();
            for (i, _) in lower.column(j) {
                if i > j {
                    let mut node = i;
                    while node != usize::MAX && mark[node] != j {
                        mark[node] = j;
                        pattern.push(node);
                        node = if self.parent[node] >= 0 {
                            self.parent[node] as usize
                        } else {
                            usize::MAX
                        };
                    }
                }
            }
            pattern.sort_unstable();
            li.extend_from_slice(&pattern);
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
            for (r, v) in lower.column(j) {
                if r >= j {
                    if stamp[r] != current {
                        stamp[r] = current;
                        touched.push(r);
                    }
                    y[r] += v;
                }
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

    /// Analyze and factorize in one call.
    pub fn analyze_and_factorize(&mut self, lower: &SparseMatrix<S>) -> bool {
        self.analyze_pattern(lower) && self.factorize(lower)
    }

    /// Solve `A x = rhs` using the cached factorization.
    pub fn solve(&self, rhs: &[S]) -> Option<Vec<S>> {
        if !self.computed || rhs.len() != self.n {
            return None;
        }
        let n = self.n;
        let mut x = rhs.to_vec();

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

        if x.iter().all(|v| v.is_finite()) {
            Some(x)
        } else {
            None
        }
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
        let mut rhs = crate::backend::zeros_vector::<f64, crate::backend::Dynamic>(n);
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
    fn diagonal_and_identity() {
        let a = spd(4);
        let lower = a.to_lower();
        let mut ldlt = SimplicialLdl::new();
        assert!(ldlt.analyze_and_factorize(&lower));
        assert_eq!(ldlt.dims(), 4);
        // Solve A x = e0.
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
        let mut triplets = vec![(0, 0, 1.0), (1, 1, -2.0)];
        let m = SparseMatrix::from_triplets(2, 2, &triplets);
        triplets.push((0, 1, 0.5));
        let mut ldlt = SimplicialLdl::new();
        assert!(!ldlt.analyze_and_factorize(&m.to_lower()));
    }
}
