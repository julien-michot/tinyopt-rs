// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Compressed sparse column matrix, mirroring Eigen's `SparseMatrix` usage in
//! tinyopt (full symmetric storage, triplets, `coeffRef` insertion).

use core::fmt;
use std::collections::BTreeMap;

use crate::backend::{DenseMatrix, Dynamic, Matrix, zeros_matrix};
use crate::real::Real;

/// Compressed sparse column matrix.
///
/// Entries are stored sorted by column then row. The optimizer treats the
/// matrix as symmetric and only reads one triangle when solving.
#[derive(Clone, PartialEq)]
pub struct SparseMatrix<S> {
    nrows: usize,
    ncols: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<S>,
}

impl<S> fmt::Debug for SparseMatrix<S>
where
    S: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseMatrix")
            .field("rows", &self.nrows)
            .field("cols", &self.ncols)
            .field("nnz", &self.values.len())
            .finish()
    }
}

impl<S: Real> Default for SparseMatrix<S> {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl<S: Real> SparseMatrix<S> {
    /// An empty `rows × cols` matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            nrows: rows,
            ncols: cols,
            col_ptrs: vec![0; cols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Build from `(row, col, value)` triplets; duplicate entries are summed.
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, S)]) -> Self {
        // Key by (column, row) so iteration yields the CSC order.
        let mut map: BTreeMap<(usize, usize), S> = BTreeMap::new();
        for &(i, j, v) in triplets {
            assert!(i < rows && j < cols, "triplet index out of bounds");
            *map.entry((j, i)).or_insert_with(S::zero) += v;
        }
        let mut matrix = Self::new(rows, cols);
        matrix.col_ptrs = vec![0; cols + 1];
        matrix.row_indices = Vec::with_capacity(map.len());
        matrix.values = Vec::with_capacity(map.len());
        for (&(j, i), &v) in &map {
            matrix.col_ptrs[j + 1] += 1;
            matrix.row_indices.push(i);
            matrix.values.push(v);
        }
        // BTreeMap iterates in (col,row) order, matching CSC.
        for j in 0..cols {
            matrix.col_ptrs[j + 1] += matrix.col_ptrs[j];
        }
        matrix
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored entries.
    pub fn non_zeros(&self) -> usize {
        self.values.len()
    }

    /// Whether no entry is stored.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Column pointers (length `ncols + 1`).
    pub fn col_ptrs(&self) -> &[usize] {
        &self.col_ptrs
    }

    /// Row indices of the stored entries.
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Stored values.
    pub fn values(&self) -> &[S] {
        &self.values
    }

    /// Iterate over `(row, col, value)`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, S)> + '_ {
        (0..self.ncols).flat_map(move |j| {
            (self.col_ptrs[j]..self.col_ptrs[j + 1])
                .map(move |p| (self.row_indices[p], j, self.values[p]))
        })
    }

    /// Entries of column `j` as `(row, value)` pairs.
    pub fn column(&self, j: usize) -> impl Iterator<Item = (usize, S)> + '_ {
        (self.col_ptrs[j]..self.col_ptrs[j + 1]).map(move |p| (self.row_indices[p], self.values[p]))
    }

    /// Coefficient value (zero when not stored).
    pub fn coeff(&self, i: usize, j: usize) -> S {
        let (lo, hi) = (self.col_ptrs[j], self.col_ptrs[j + 1]);
        match self.row_indices[lo..hi].binary_search(&i) {
            Ok(offset) => self.values[lo + offset],
            Err(_) => S::zero(),
        }
    }

    /// Mutable coefficient, inserting a new entry when needed.
    ///
    /// This rebuilds the affected column, which is `O(nnz(column))`; use
    /// triplets for bulk assembly.
    pub fn coeff_ref(&mut self, i: usize, j: usize) -> &mut S {
        let (lo, hi) = (self.col_ptrs[j], self.col_ptrs[j + 1]);
        match self.row_indices[lo..hi].binary_search(&i) {
            Ok(offset) => &mut self.values[lo + offset],
            Err(offset) => {
                self.row_indices.insert(lo + offset, i);
                self.values.insert(lo + offset, S::zero());
                for ptr in self.col_ptrs.iter_mut().skip(j + 1) {
                    *ptr += 1;
                }
                &mut self.values[lo + offset]
            }
        }
    }

    /// Set a coefficient (inserting or overwriting).
    pub fn set_coeff(&mut self, i: usize, j: usize, value: S) {
        *self.coeff_ref(i, j) = value;
    }

    /// Zero all stored values, keeping the sparsity pattern.
    pub fn set_zero(&mut self) {
        for v in &mut self.values {
            *v = S::zero();
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.col_ptrs = vec![0; self.ncols + 1];
        self.row_indices.clear();
        self.values.clear();
    }

    /// Diagonal coefficients.
    pub fn diagonal(&self) -> Vec<S> {
        let n = self.nrows.min(self.ncols);
        (0..n).map(|i| self.coeff(i, i)).collect()
    }

    /// Multiply the diagonal by `factor`.
    pub fn scale_diagonal(&mut self, factor: S) {
        let n = self.nrows.min(self.ncols);
        for i in 0..n {
            *self.coeff_ref(i, i) *= factor;
        }
    }

    /// Scale by `diag(s) * diag(s)`, or divide when `inverse` is true.
    pub fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
        for j in 0..self.ncols {
            let sj = scale[j];
            for p in self.col_ptrs[j]..self.col_ptrs[j + 1] {
                let si = scale[self.row_indices[p]];
                let factor = si * sj;
                if inverse {
                    self.values[p] /= factor;
                } else {
                    self.values[p] *= factor;
                }
            }
        }
    }

    /// Zero rows and columns of fixed parameters and set their diagonal to one.
    pub fn apply_fixed_indices(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }
        let is_fixed = |i: usize| indices.binary_search(&i).is_ok();
        for j in 0..self.ncols {
            for p in self.col_ptrs[j]..self.col_ptrs[j + 1] {
                let i = self.row_indices[p];
                if i == j {
                    if is_fixed(i) {
                        self.values[p] = S::one();
                    }
                } else if is_fixed(i) || is_fixed(j) {
                    self.values[p] = S::zero();
                }
            }
        }
        for &i in indices {
            if i < self.nrows && i < self.ncols {
                *self.coeff_ref(i, i) = S::one();
            }
        }
    }

    /// Whether every stored value is finite.
    pub fn is_all_finite(&self) -> bool {
        self.values.iter().all(|v| v.is_finite())
    }

    /// Dense conversion.
    pub fn to_dense(&self) -> Matrix<S, Dynamic> {
        let mut dense = zeros_matrix::<S, Dynamic>(self.nrows, self.ncols);
        for j in 0..self.ncols {
            for (i, v) in self.column(j) {
                dense.set_elem(i, j, v);
            }
        }
        dense
    }

    /// Lower-triangular part, mirrored from the upper triangle when needed.
    ///
    /// Accepts full-symmetric, upper-only, or lower-only input.
    pub fn to_lower(&self) -> Self {
        let mut map: BTreeMap<(usize, usize), S> = BTreeMap::new();
        for (i, j, v) in self.iter() {
            if i >= j {
                *map.entry((i, j)).or_insert_with(S::zero) += v;
            }
        }
        for (i, j, v) in self.iter() {
            if i < j && !map.contains_key(&(j, i)) {
                map.insert((j, i), v);
            }
        }
        let triplets: Vec<(usize, usize, S)> =
            map.into_iter().map(|((i, j), v)| (i, j, v)).collect();
        Self::from_triplets(self.nrows, self.ncols, &triplets)
    }

    /// Matrix inverse (dense result pattern), mirroring C++ `Inverse` for
    /// sparse input.
    pub fn inverse(&self) -> Option<Self> {
        if self.nrows() == 0 || self.nrows() != self.ncols() {
            return None;
        }
        let mut solver = crate::sparse::SparseLinearSolvers::<S>::new(
            crate::options::LinearSolverOptions::default(),
        );
        if !solver.analyze(self, 0) || !solver.compute(self) {
            return None;
        }
        solver.inverse(self)
    }

    /// Upper-triangular part, mirrored from the lower triangle when needed.
    pub fn to_upper(&self) -> Self {
        let mut map: BTreeMap<(usize, usize), S> = BTreeMap::new();
        for (i, j, v) in self.iter() {
            if i <= j {
                *map.entry((i, j)).or_insert_with(S::zero) += v;
            }
        }
        for (i, j, v) in self.iter() {
            if i > j && !map.contains_key(&(j, i)) {
                map.insert((j, i), v);
            }
        }
        let triplets: Vec<(usize, usize, S)> =
            map.into_iter().map(|((i, j), v)| (i, j, v)).collect();
        Self::from_triplets(self.nrows, self.ncols, &triplets)
    }

    /// Transpose (swapping rows and columns).
    pub fn transpose(&self) -> Self {
        let triplets: Vec<(usize, usize, S)> = self.iter().map(|(i, j, v)| (j, i, v)).collect();
        Self::from_triplets(self.ncols, self.nrows, &triplets)
    }

    /// Sub-matrix `rows × cols` (ranges), preserving the CSC layout.
    pub fn submatrix(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Self {
        let mut triplets = Vec::new();
        for j in col_start..col_end {
            for (i, v) in self.column(j) {
                if i >= row_start && i < row_end {
                    triplets.push((i - row_start, j - col_start, v));
                }
            }
        }
        Self::from_triplets(row_end - row_start, col_end - col_start, &triplets)
    }

    /// Matrix-vector product with a dense vector.
    pub fn mul_vec(&self, x: &[S]) -> Vec<S> {
        let mut out = vec![S::zero(); self.nrows];
        for (j, xj) in x.iter().enumerate().take(self.ncols) {
            for (i, v) in self.column(j) {
                out[i] += v * *xj;
            }
        }
        out
    }

    /// Squared Frobenius norm of the stored values.
    pub fn squared_norm(&self) -> S {
        self.values.iter().fold(S::zero(), |acc, v| acc + *v * *v)
    }

    /// Frobenius norm.
    pub fn norm(&self) -> S {
        num_traits::Float::sqrt(self.squared_norm())
    }

    /// Cast to another scalar type.
    pub fn cast<T: Real>(&self) -> SparseMatrix<T> {
        let triplets: Vec<(usize, usize, T)> = self
            .iter()
            .map(|(i, j, v)| {
                (
                    i,
                    j,
                    T::from_f64(v.to_f64().unwrap_or(f64::NAN)).expect("cast"),
                )
            })
            .collect();
        SparseMatrix::from_triplets(self.nrows, self.ncols, &triplets)
    }

    /// Keep only the entries present in `pattern`.
    pub fn mask_by_pattern(&self, pattern: &SparseMatrix<S>) -> SparseMatrix<S> {
        let triplets: Vec<(usize, usize, S)> = self
            .iter()
            .filter(|(i, j, _)| pattern.coeff(*i, *j) != S::zero() || (i == j))
            .collect();
        SparseMatrix::from_triplets(self.nrows, self.ncols, &triplets)
    }

    /// Whether this matrix is approximately equal to another.
    pub fn is_approx(&self, other: &Self, epsilon: S) -> bool {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            return false;
        }
        for j in 0..self.ncols {
            let mut a = self.column(j);
            let mut b = other.column(j);
            loop {
                match (a.next(), b.next()) {
                    (None, None) => break,
                    (Some((ia, va)), Some((ib, vb))) => {
                        if ia != ib || num_traits::Float::abs(va - vb) > epsilon {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn triplets_sum_duplicates() {
        let m = SparseMatrix::from_triplets(
            2,
            2,
            &[(0, 0, 1.0), (1, 1, 2.0), (0, 1, 3.0), (0, 1, 4.0)],
        );
        assert_eq!(m.non_zeros(), 3);
        assert_relative_eq!(m.coeff(0, 1), 7.0);
        assert_relative_eq!(m.coeff(1, 0), 0.0);
    }

    #[test]
    fn coeff_ref_inserts_and_overwrites() {
        let mut m = SparseMatrix::<f64>::new(3, 3);
        *m.coeff_ref(1, 1) = 2.0;
        *m.coeff_ref(0, 2) = 5.0;
        assert_relative_eq!(m.coeff(1, 1), 2.0);
        assert_relative_eq!(m.coeff(0, 2), 5.0);
        assert_eq!(m.non_zeros(), 2);
        m.scale_diagonal(3.0);
        assert_relative_eq!(m.coeff(1, 1), 6.0);
    }

    #[test]
    fn lower_and_upper_conversions() {
        let mut m = SparseMatrix::<f64>::new(2, 2);
        *m.coeff_ref(0, 0) = 2.0;
        *m.coeff_ref(0, 1) = 1.0; // upper only

        let lower = m.to_lower();
        assert_relative_eq!(lower.coeff(1, 0), 1.0);
        assert_relative_eq!(lower.coeff(0, 1), 0.0);

        let upper = lower.to_upper();
        assert_relative_eq!(upper.coeff(0, 1), 1.0);
        assert_relative_eq!(upper.coeff(1, 0), 0.0);
    }

    #[test]
    fn dense_conversion_and_product() {
        let m = SparseMatrix::from_triplets(
            2,
            2,
            &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)],
        );
        let d = m.to_dense();
        assert_relative_eq!(d.elem(1, 1), 3.0);
        let prod = m.mul_vec(&[1.0, 2.0]);
        assert_relative_eq!(prod[0], 4.0);
        assert_relative_eq!(prod[1], 7.0);
    }

    #[test]
    fn fixed_indices_are_zeroed() {
        let m = SparseMatrix::from_triplets(2, 2, &[(0, 0, 2.0), (1, 1, 3.0), (0, 1, 1.0)]);
        let mut m2 = m.clone();
        m2.apply_fixed_indices(&[1]);
        assert_relative_eq!(m2.coeff(1, 1), 1.0);
        assert_relative_eq!(m2.coeff(0, 1), 0.0);
        assert_relative_eq!(m2.coeff(0, 0), 2.0);
    }

    #[test]
    fn submatrix_and_scaling() {
        let m = SparseMatrix::from_triplets(
            3,
            3,
            &[(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0), (1, 2, 1.0)],
        );
        let sub = m.submatrix(1, 3, 1, 3);
        assert_eq!(sub.nrows(), 2);
        assert_relative_eq!(sub.coeff(1, 1), 3.0);

        let mut scaled = m.clone();
        scaled.scale_by_diagonal(&[2.0, 3.0, 4.0], false);
        assert_relative_eq!(scaled.coeff(1, 2), 1.0 * 3.0 * 4.0);
    }
}
