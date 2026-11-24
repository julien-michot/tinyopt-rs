// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Compressed sparse column matrix, mirroring Eigen's `SparseMatrix` usage in
//! tinyopt (full symmetric storage, triplets, `coeffRef` insertion).

use core::fmt;

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
        // Counting sort by column (O(nnz + n)), then sort the (short) row lists
        // inside each column and merge duplicates. This avoids the
        // O(nnz log nnz) comparison sort over the whole triplet list.
        let mut counts = vec![0usize; cols];
        for &(i, j, _) in triplets {
            assert!(i < rows && j < cols, "triplet index out of bounds");
            counts[j] += 1;
        }
        let mut col_ptrs = vec![0usize; cols + 1];
        for j in 0..cols {
            col_ptrs[j + 1] = col_ptrs[j] + counts[j];
        }
        let mut pos = col_ptrs.clone();
        // Every slot is written exactly once, so skip zero-initialization.
        let mut buckets: Vec<core::mem::MaybeUninit<(usize, S)>> =
            Vec::with_capacity(triplets.len());
        // SAFETY: the capacity is `triplets.len()` and each slot is written below.
        unsafe { buckets.set_len(triplets.len()) };
        for &(i, j, v) in triplets {
            let p = pos[j];
            buckets[p].write((i, v));
            pos[j] += 1;
        }
        // SAFETY: all slots were initialized by the loop above and
        // `MaybeUninit<T>` has the same layout as `T`.
        let (ptr, len, cap) = (buckets.as_mut_ptr(), buckets.len(), buckets.capacity());
        core::mem::forget(buckets);
        let mut buckets: Vec<(usize, S)> =
            unsafe { Vec::from_raw_parts(ptr as *mut (usize, S), len, cap) };

        let mut row_indices = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());
        let mut merged_ptrs = vec![0usize; cols + 1];
        for j in 0..cols {
            let slice = &mut buckets[col_ptrs[j]..col_ptrs[j + 1]];
            slice.sort_unstable_by_key(|&(row, _)| row);
            let mut k = 0;
            while k < slice.len() {
                let (row, mut value) = slice[k];
                k += 1;
                while k < slice.len() && slice[k].0 == row {
                    value += slice[k].1;
                    k += 1;
                }
                row_indices.push(row);
                values.push(value);
            }
            merged_ptrs[j + 1] = row_indices.len();
        }
        Self::from_csc_parts(rows, cols, merged_ptrs, row_indices, values)
    }

    /// Build a CSC matrix from prepared arrays (no validation, no sorting).
    pub(crate) fn from_csc_parts(
        rows: usize,
        cols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<S>,
    ) -> Self {
        Self {
            nrows: rows,
            ncols: cols,
            col_ptrs,
            row_indices,
            values,
        }
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

    /// Refresh the values of a stored lower-triangle block from `h`, without
    /// recomputing the pattern.
    ///
    /// `self` must hold the lower triangle of `h[start.., start..]` (built by
    /// `submatrix(...).to_lower()`); returns false when the pattern has
    /// changed or when `h` has upper-only entries that need mirroring.
    pub fn refresh_lower_block_from(&mut self, h: &SparseMatrix<S>, start: usize) -> bool {
        let n = h.nrows();
        if start >= n || self.nrows != n - start || self.ncols != n - start {
            return false;
        }
        let mut p = 0usize;
        for j in start..n {
            let col = j - start;
            let (lo, hi) = (h.col_ptrs[j], h.col_ptrs[j + 1]);
            for q in lo..hi {
                let i = h.row_indices[q];
                if i < j {
                    continue;
                }
                if p >= self.row_indices.len() || self.row_indices[p] != i - start {
                    return false;
                }
                self.values[p] = h.values[q];
                p += 1;
            }
            if self.col_ptrs[col + 1] != p {
                return false;
            }
        }
        p == self.row_indices.len()
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
        let mut is_fixed = vec![false; self.ncols.max(self.nrows)];
        for &i in indices {
            if i < is_fixed.len() {
                is_fixed[i] = true;
            }
        }
        for j in 0..self.ncols {
            for p in self.col_ptrs[j]..self.col_ptrs[j + 1] {
                let i = self.row_indices[p];
                if i == j {
                    if is_fixed[i] {
                        self.values[p] = S::one();
                    }
                } else if is_fixed[i] || is_fixed[j] {
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
        let lower = self.lower_csc();
        // Fast path: every strict-upper entry already has a lower counterpart.
        let mut complete = true;
        for j in 0..self.ncols {
            for &i in &self.row_indices[self.col_ptrs[j]..self.col_ptrs[j + 1]] {
                if i < j && !lower.has_entry(j, i) {
                    complete = false;
                    break;
                }
            }
            if !complete {
                break;
            }
        }
        if complete {
            return lower;
        }
        // Upper-only or asymmetric input: mirror the missing entries.
        let mut triplets: Vec<(usize, usize, S)> =
            self.iter().filter(|&(i, j, _)| i >= j).collect();
        let mut keys: Vec<(usize, usize)> = triplets.iter().map(|&(i, j, _)| (i, j)).collect();
        keys.sort_unstable();
        for (i, j, v) in self.iter() {
            if i < j && keys.binary_search(&(j, i)).is_err() {
                triplets.push((j, i, v));
            }
        }
        Self::from_triplets(self.nrows, self.ncols, &triplets)
    }

    /// Lower triangle as CSC (direct copy, rows stay sorted).
    fn lower_csc(&self) -> Self {
        let mut col_ptrs = Vec::with_capacity(self.ncols + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        col_ptrs.push(0);
        for j in 0..self.ncols {
            let (lo, hi) = (self.col_ptrs[j], self.col_ptrs[j + 1]);
            let start = lo + self.row_indices[lo..hi].partition_point(|&r| r < j);
            row_indices.extend_from_slice(&self.row_indices[start..hi]);
            values.extend_from_slice(&self.values[start..hi]);
            col_ptrs.push(row_indices.len());
        }
        Self::from_csc_parts(self.nrows, self.ncols, col_ptrs, row_indices, values)
    }

    /// Whether `(i, j)` is explicitly stored.
    fn has_entry(&self, i: usize, j: usize) -> bool {
        let (lo, hi) = (self.col_ptrs[j], self.col_ptrs[j + 1]);
        self.row_indices[lo..hi].binary_search(&i).is_ok()
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
        let upper = self.upper_csc();
        let mut complete = true;
        for j in 0..self.ncols {
            for &i in &self.row_indices[self.col_ptrs[j]..self.col_ptrs[j + 1]] {
                if i > j && !upper.has_entry(j, i) {
                    complete = false;
                    break;
                }
            }
            if !complete {
                break;
            }
        }
        if complete {
            return upper;
        }
        let mut triplets: Vec<(usize, usize, S)> =
            self.iter().filter(|&(i, j, _)| i <= j).collect();
        let mut keys: Vec<(usize, usize)> = triplets.iter().map(|&(i, j, _)| (i, j)).collect();
        keys.sort_unstable();
        for (i, j, v) in self.iter() {
            if i > j && keys.binary_search(&(j, i)).is_err() {
                triplets.push((j, i, v));
            }
        }
        Self::from_triplets(self.nrows, self.ncols, &triplets)
    }

    /// Upper triangle as CSC (direct copy, rows stay sorted).
    fn upper_csc(&self) -> Self {
        let mut col_ptrs = Vec::with_capacity(self.ncols + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        col_ptrs.push(0);
        for j in 0..self.ncols {
            let (lo, hi) = (self.col_ptrs[j], self.col_ptrs[j + 1]);
            let end = lo + self.row_indices[lo..hi].partition_point(|&r| r <= j);
            row_indices.extend_from_slice(&self.row_indices[lo..end]);
            values.extend_from_slice(&self.values[lo..end]);
            col_ptrs.push(row_indices.len());
        }
        Self::from_csc_parts(self.nrows, self.ncols, col_ptrs, row_indices, values)
    }

    /// Transpose (swapping rows and columns).
    pub fn transpose(&self) -> Self {
        // Counting sort by the new column (= old row); within a column the
        // entries stay ordered by the old column, so rows remain sorted.
        let mut counts = vec![0usize; self.nrows];
        for &i in &self.row_indices {
            counts[i] += 1;
        }
        let mut col_ptrs = vec![0usize; self.nrows + 1];
        for j in 0..self.nrows {
            col_ptrs[j + 1] = col_ptrs[j] + counts[j];
        }
        let mut pos = col_ptrs.clone();
        let mut row_indices = vec![0usize; self.row_indices.len()];
        let mut values = vec![S::zero(); self.values.len()];
        for j in 0..self.ncols {
            for p in self.col_ptrs[j]..self.col_ptrs[j + 1] {
                let r = self.row_indices[p];
                let q = pos[r];
                pos[r] += 1;
                row_indices[q] = j;
                values[q] = self.values[p];
            }
        }
        Self::from_csc_parts(self.ncols, self.nrows, col_ptrs, row_indices, values)
    }

    /// Sub-matrix `rows × cols` (ranges), preserving the CSC layout.
    pub fn submatrix(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> Self {
        // Rows are sorted inside each column: copy the matching range directly.
        let mut col_ptrs = Vec::with_capacity(col_end - col_start + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        col_ptrs.push(0);
        for j in col_start..col_end {
            let (lo, hi) = (self.col_ptrs[j], self.col_ptrs[j + 1]);
            let rows = &self.row_indices[lo..hi];
            let start = lo + rows.partition_point(|&r| r < row_start);
            let end = lo + rows.partition_point(|&r| r < row_end);
            row_indices.extend(rows[start - lo..end - lo].iter().map(|r| r - row_start));
            values.extend_from_slice(&self.values[start..end]);
            col_ptrs.push(row_indices.len());
        }
        Self::from_csc_parts(
            row_end - row_start,
            col_end - col_start,
            col_ptrs,
            row_indices,
            values,
        )
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
