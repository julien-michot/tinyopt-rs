// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Compile-time block-sparse symmetric matrices, mirroring tinyopt's
//! `BlockSparseMatrix<Scalar, BlockSizes...>` (upper type pairs only).
//!
//! Block sizes are declared with `block_sparse_types!`, which also generates
//! the compile-time `HasPair<R, C>` checks used by [`BlockSparseMat::block_mut`].

use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::backend::{DenseMatrix, Dynamic, Matrix, zeros_matrix};
use crate::block_tridiag::BlockMut;
use crate::cost::Cost;
use crate::covariance::{CovarianceOptions, FinalCovariance};
use crate::hessian::HessianStorage;
use crate::options::LinearSolverOptions;
use crate::real::Real;
use crate::sparse::{SparseLinearSolvers, SparseMatrix};

/// Set of compile-time block sizes.
pub trait BlockTypes: Clone + Copy + Default + core::fmt::Debug + Send + Sync + 'static {
    /// Block sizes, in declaration order.
    const SIZES: &'static [usize];
}

/// Compile-time marker: the block type pair `(R, C)` exists.
pub trait HasPair<const R: usize, const C: usize> {}

/// Declare a block-size list and all its allowed type pairs.
///
/// ```ignore
/// block_sparse_types!(BaTypes, [6, 3]);
/// // BlockSparseMat<BaTypes, f64>: block_mut::<6, 3>(camera, point)
/// ```
#[macro_export]
macro_rules! block_sparse_types {
    ($name:ident, [$($sizes:literal),+ $(,)?]) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl $crate::block_sparse::BlockTypes for $name {
            const SIZES: &'static [usize] = &[$($sizes),+];
        }

        $crate::block_sparse_types!(@rows $name, [$($sizes),+] = [$($sizes),+]);
    };
    (@rows $name:ident, [$($all:literal),+] = []) => {};
    (@rows $name:ident, [$($all:literal),+] = [$r:literal $(, $rest:literal)*]) => {
        $( impl $crate::block_sparse::HasPair<$r, $all> for $name {} )+
        $crate::block_sparse_types!(@rows $name, [$($all),+] = [$($rest),*]);
    };
}

/// Symmetric block-sparse matrix with compile-time block sizes.
pub struct BlockSparseMat<T: BlockTypes, S: Real = f64> {
    sizes: &'static [usize],
    counts: Vec<usize>,
    blocks: Vec<BTreeMap<(usize, usize), Vec<S>>>,
    _marker: PhantomData<T>,
}

impl<T: BlockTypes, S: Real> Clone for BlockSparseMat<T, S> {
    fn clone(&self) -> Self {
        Self {
            sizes: self.sizes,
            counts: self.counts.clone(),
            blocks: self.blocks.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: BlockTypes, S: Real> core::fmt::Debug for BlockSparseMat<T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BlockSparseMat")
            .field("sizes", &self.sizes)
            .field("counts", &self.counts)
            .finish()
    }
}

impl<T: BlockTypes, S: Real> BlockSparseMat<T, S> {
    fn num_types() -> usize {
        T::SIZES.len()
    }

    fn num_pairs() -> usize {
        let nt = Self::num_types();
        nt * (nt + 1) / 2
    }

    fn pair_index(a: usize, b: usize) -> usize {
        let nt = Self::num_types();
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        lo * nt - lo * lo.saturating_sub(1) / 2 + (hi - lo)
    }

    fn type_index(size: usize) -> usize {
        T::SIZES
            .iter()
            .position(|s| *s == size)
            .unwrap_or_else(|| panic!("block size {size} is not declared"))
    }

    /// An empty matrix (all block counts zero).
    pub fn new() -> Self {
        Self::with_counts(&vec![0; Self::num_types()])
    }

    /// A matrix with explicit per-type block counts.
    pub fn with_counts(counts: &[usize]) -> Self {
        assert_eq!(counts.len(), Self::num_types());
        Self {
            sizes: T::SIZES,
            counts: counts.to_vec(),
            blocks: (0..Self::num_pairs()).map(|_| BTreeMap::new()).collect(),
            _marker: PhantomData,
        }
    }

    /// Per-type block counts.
    pub fn block_counts(&self) -> &[usize] {
        &self.counts
    }

    /// Block sizes.
    pub fn block_sizes(&self) -> &[usize] {
        self.sizes
    }

    /// Number of block types.
    pub fn num_block_types(&self) -> usize {
        Self::num_types()
    }

    /// Number of blocks of a type.
    pub fn num_blocks_of(&self, type_index: usize) -> usize {
        self.counts[type_index]
    }

    /// Scalar offset of a type's first block.
    pub fn base_offset(&self, type_index: usize) -> usize {
        (0..type_index)
            .map(|i| self.counts[i] * self.sizes[i])
            .sum()
    }

    /// Total scalar dimension.
    pub fn dims(&self) -> usize {
        (0..Self::num_types())
            .map(|i| self.counts[i] * self.sizes[i])
            .sum()
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.dims()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.dims()
    }

    /// Mutable upper-triangle block `(r, c)` with compile-time sizes.
    pub fn block_mut<const R: usize, const C: usize>(
        &mut self,
        r: usize,
        c: usize,
    ) -> BlockMut<'_, S>
    where
        T: HasPair<R, C>,
    {
        let ta = Self::type_index(R);
        let tb = Self::type_index(C);
        assert!(
            ta <= tb,
            "write the upper-triangle type pair (declare larger/head types first)"
        );
        self.counts[ta] = self.counts[ta].max(r + 1);
        self.counts[tb] = self.counts[tb].max(c + 1);
        let key = (r, c);
        let pair = Self::pair_index(ta, tb);
        let data = self.blocks[pair]
            .entry(key)
            .or_insert_with(|| vec![S::zero(); R * C]);
        BlockMut::from_data(data, R, C)
    }

    /// Mutable diagonal block `(i, i)` of a given block size.
    pub fn diagonal_mut<const N: usize>(&mut self, i: usize) -> BlockMut<'_, S>
    where
        T: HasPair<N, N>,
    {
        self.block_mut::<N, N>(i, i)
    }

    /// Zero all blocks, keeping the structure.
    pub fn set_zero(&mut self) {
        for map in &mut self.blocks {
            for block in map.values_mut() {
                block.iter_mut().for_each(|v| *v = S::zero());
            }
        }
    }

    /// Set the diagonal blocks of every type to identity.
    pub fn set_identity(&mut self) {
        self.set_zero();
        for t in 0..Self::num_types() {
            let size = self.sizes[t];
            let pair = Self::pair_index(t, t);
            for i in 0..self.counts[t] {
                let block = self.blocks[pair]
                    .entry((i, i))
                    .or_insert_with(|| vec![S::zero(); size * size]);
                for d in 0..size {
                    block[d * size + d] = S::one();
                }
            }
        }
    }

    /// Zero fixed rows/columns and set their diagonal to one.
    pub fn apply_fixed_indices(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }
        let is_fixed = |i: usize| indices.binary_search(&i).is_ok();
        let mut offsets = Vec::with_capacity(Self::num_types());
        let mut off = 0;
        for t in 0..Self::num_types() {
            offsets.push(off);
            off += self.counts[t] * self.sizes[t];
        }

        for a in 0..Self::num_types() {
            for b in a..Self::num_types() {
                let pair = Self::pair_index(a, b);
                let (ra, ca) = (self.sizes[a], self.sizes[b]);
                let (oa, ob) = (offsets[a], offsets[b]);
                let map = &mut self.blocks[pair];
                for (&(r, c), block) in map.iter_mut() {
                    let row_base = oa + r * ra;
                    let col_base = ob + c * ca;
                    for j in 0..ca {
                        for i in 0..ra {
                            let gi = row_base + i;
                            let gj = col_base + j;
                            if a == b && r == c && i == j {
                                if is_fixed(gi) {
                                    block[j * ra + i] = S::one();
                                }
                            } else if is_fixed(gi) || is_fixed(gj) {
                                block[j * ra + i] = S::zero();
                            }
                        }
                    }
                }
            }
        }
        // Ensure fixed diagonal entries exist.
        for &idx in indices {
            if idx < self.dims() {
                self.set_entry(idx, idx, S::one());
            }
        }
    }

    /// Whether every coefficient is finite.
    pub fn is_all_finite(&self) -> bool {
        self.blocks
            .iter()
            .all(|map| map.values().all(|b| b.iter().all(|v| v.is_finite())))
    }

    /// Scalar coefficient `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> S {
        let mut offset = 0;
        let mut row_type = 0;
        for (t, &size) in self.sizes.iter().enumerate() {
            let span = self.counts[t] * size;
            if i < offset + span {
                row_type = t;
                break;
            }
            offset += span;
        }
        let row_off = offset;
        let mut offset = 0;
        let mut col_type = 0;
        for (t, &size) in self.sizes.iter().enumerate() {
            let span = self.counts[t] * size;
            if j < offset + span {
                col_type = t;
                break;
            }
            offset += span;
        }
        let col_off = offset;
        let (ta, tb) = (row_type, col_type);
        let (r, li) = (
            (i - row_off) / self.sizes[ta],
            (i - row_off) % self.sizes[ta],
        );
        let (c, lj) = (
            (j - col_off) / self.sizes[tb],
            (j - col_off) % self.sizes[tb],
        );
        let (pair, key, transpose) = if ta <= tb {
            (Self::pair_index(ta, tb), (r, c), false)
        } else {
            (Self::pair_index(tb, ta), (c, r), true)
        };
        match self.blocks[pair].get(&key) {
            Some(block) => {
                let rows = self.sizes[if ta <= tb { ta } else { tb }];
                if transpose {
                    block[li * rows + lj]
                } else {
                    block[lj * rows + li]
                }
            }
            None => S::zero(),
        }
    }

    /// Set a scalar coefficient (creating the owning block when needed).
    pub fn set_entry(&mut self, i: usize, j: usize, value: S) {
        let (row_type, row_off) = self.type_and_offset(i);
        let (col_type, col_off) = self.type_and_offset(j);
        let (ta, tb) = (row_type, col_type);
        let ra = self.sizes[ta];
        let ca = self.sizes[tb];
        let (r, li) = ((i - row_off) / ra, (i - row_off) % ra);
        let (c, lj) = ((j - col_off) / ca, (j - col_off) % ca);
        let (pair, key, transpose) = if ta <= tb {
            (Self::pair_index(ta, tb), (r, c), false)
        } else {
            (Self::pair_index(tb, ta), (c, r), true)
        };
        let rows = if ta <= tb { ra } else { ca };
        let cols = if ta <= tb { ca } else { ra };
        let block = self.blocks[pair]
            .entry(key)
            .or_insert_with(|| vec![S::zero(); rows * cols]);
        let index = if transpose {
            li * rows + lj
        } else {
            lj * rows + li
        };
        block[index] = value;
    }

    fn type_and_offset(&self, index: usize) -> (usize, usize) {
        let mut offset = 0;
        for (t, &size) in self.sizes.iter().enumerate() {
            let span = self.counts[t] * size;
            if index < offset + span {
                return (t, offset);
            }
            offset += span;
        }
        panic!("index out of bounds");
    }

    /// Diagonal coefficients.
    pub fn diagonal_into(&self, out: &mut [S]) {
        for (i, o) in out.iter_mut().enumerate().take(self.dims()) {
            *o = self.get(i, i);
        }
    }

    /// Multiply the diagonal blocks by `factor`.
    pub fn scale_diagonal(&mut self, factor: S) {
        for t in 0..Self::num_types() {
            let pair = Self::pair_index(t, t);
            for (&(r, c), block) in self.blocks[pair].iter_mut() {
                if r == c {
                    block.iter_mut().for_each(|v| *v *= factor);
                }
            }
        }
    }

    /// Scale by `diag(s) * diag(s)` (or divide when `inverse`).
    pub fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
        let mut offsets = Vec::with_capacity(Self::num_types());
        let mut off = 0;
        for t in 0..Self::num_types() {
            offsets.push(off);
            off += self.counts[t] * self.sizes[t];
        }
        for a in 0..Self::num_types() {
            for b in a..Self::num_types() {
                let pair = Self::pair_index(a, b);
                let (ra, ca) = (self.sizes[a], self.sizes[b]);
                let (oa, ob) = (offsets[a], offsets[b]);
                for (&(r, c), block) in self.blocks[pair].iter_mut() {
                    let row_base = oa + r * ra;
                    let col_base = ob + c * ca;
                    for j in 0..ca {
                        for i in 0..ra {
                            let factor = scale[row_base + i] * scale[col_base + j];
                            if inverse {
                                block[j * ra + i] /= factor;
                            } else {
                                block[j * ra + i] *= factor;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Matrix-vector product into `out`.
    pub fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
        out.iter_mut().for_each(|v| *v = S::zero());
        let mut offsets = Vec::with_capacity(Self::num_types());
        let mut off = 0;
        for t in 0..Self::num_types() {
            offsets.push(off);
            off += self.counts[t] * self.sizes[t];
        }
        for a in 0..Self::num_types() {
            for b in a..Self::num_types() {
                let pair = Self::pair_index(a, b);
                let (ra, ca) = (self.sizes[a], self.sizes[b]);
                let (oa, ob) = (offsets[a], offsets[b]);
                for (&(r, c), block) in &self.blocks[pair] {
                    let row_base = oa + r * ra;
                    let col_base = ob + c * ca;
                    // A(ab) * x_b contributes to rows a.
                    for i in 0..ra {
                        let mut acc = S::zero();
                        for j in 0..ca {
                            acc += block[j * ra + i] * x[col_base + j];
                        }
                        out[row_base + i] += acc;
                    }
                    if a != b {
                        // Symmetric contribution to rows b.
                        for i in 0..ca {
                            let mut acc = S::zero();
                            for j in 0..ra {
                                acc += block[i * ra + j] * x[row_base + j];
                            }
                            out[col_base + i] += acc;
                        }
                    }
                }
            }
        }
    }

    /// Dense conversion.
    pub fn to_dense(&self) -> Matrix<S, Dynamic> {
        let n = self.dims();
        let mut dense = zeros_matrix::<S, Dynamic>(n, n);
        for j in 0..n {
            for i in 0..n {
                let v = self.get(i, j);
                if v != S::zero() {
                    dense.set_elem(i, j, v);
                }
            }
        }
        dense
    }

    /// Sparse conversion (full symmetric pattern).
    pub fn to_sparse(&self) -> SparseMatrix<S> {
        let n = self.dims();
        let mut triplets = Vec::new();
        let mut offsets = Vec::with_capacity(Self::num_types());
        let mut off = 0;
        for t in 0..Self::num_types() {
            offsets.push(off);
            off += self.counts[t] * self.sizes[t];
        }
        for a in 0..Self::num_types() {
            for b in a..Self::num_types() {
                let pair = Self::pair_index(a, b);
                let (ra, ca) = (self.sizes[a], self.sizes[b]);
                let (oa, ob) = (offsets[a], offsets[b]);
                for (&(r, c), block) in &self.blocks[pair] {
                    let row_base = oa + r * ra;
                    let col_base = ob + c * ca;
                    for j in 0..ca {
                        for i in 0..ra {
                            let v = block[j * ra + i];
                            if v != S::zero() {
                                triplets.push((row_base + i, col_base + j, v));
                                if a != b {
                                    triplets.push((col_base + j, row_base + i, v));
                                }
                            }
                        }
                    }
                }
            }
        }
        triplets.sort_unstable_by_key(|&(row, col, _)| (col, row));
        SparseMatrix::from_sorted_triplets(n, n, &triplets)
    }

    /// Number of allocated blocks.
    pub fn num_allocated_blocks(&self) -> usize {
        self.blocks.iter().map(|m| m.len()).sum()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.iter().all(|m| m.is_empty())
    }

    /// Layout information `(sizes, counts)`.
    pub fn layout(&self) -> (Vec<usize>, Vec<usize>) {
        (self.sizes.to_vec(), self.counts.clone())
    }
}

impl<T: BlockTypes, S: Real> Default for BlockSparseMat<T, S> {
    fn default() -> Self {
        Self::new()
    }
}

// --- Runtime-layout block-sparse matrix (covariance carrier, C++ MatX) -----

/// Runtime-layout block-sparse matrix (C++ `BlockSparseMatX`).
#[derive(Clone, Debug)]
pub struct BlockSparseMatX<S: Real = f64> {
    sparse: SparseMatrix<S>,
    block_sizes: Vec<usize>,
    block_counts: Vec<usize>,
}

impl<S: Real> BlockSparseMatX<S> {
    /// Build an empty matrix with the given block layout.
    pub fn new(block_sizes: &[usize], block_counts: &[usize]) -> Self {
        assert_eq!(block_sizes.len(), block_counts.len());
        let dims: usize = block_sizes
            .iter()
            .zip(block_counts)
            .map(|(s, c)| s * c)
            .sum();
        Self {
            sparse: SparseMatrix::new(dims, dims),
            block_sizes: block_sizes.to_vec(),
            block_counts: block_counts.to_vec(),
        }
    }

    /// Scalar offset of a type's first block.
    fn base_offset(&self, type_index: usize) -> usize {
        (0..type_index)
            .map(|i| self.block_sizes[i] * self.block_counts[i])
            .sum()
    }

    /// Resolve a global block index to `(type, local block)`.
    fn block_type(&self, index: usize) -> (usize, usize) {
        let mut offset = 0;
        for (t, &count) in self.block_counts.iter().enumerate() {
            if index < offset + count {
                return (t, index - offset);
            }
            offset += count;
        }
        panic!("block index out of bounds");
    }

    /// Write a column-major block at block coordinates `(row, col)`.
    ///
    /// The lower triangle is mirrored into the stored upper triangle.
    pub fn set_block(&mut self, row: usize, col: usize, data: &[S]) {
        let (tr, ri) = self.block_type(row);
        let (tc, ci) = self.block_type(col);
        let (sr, sc) = (self.block_sizes[tr], self.block_sizes[tc]);
        assert_eq!(data.len(), sr * sc, "block data size mismatch");
        let row_off = self.base_offset(tr) + ri * sr;
        let col_off = self.base_offset(tc) + ci * sc;
        for j in 0..sc {
            for i in 0..sr {
                let v = data[j * sr + i];
                self.sparse.set_coeff(row_off + i, col_off + j, v);
                if tr != tc {
                    self.sparse.set_coeff(col_off + j, row_off + i, v);
                }
            }
        }
    }

    /// Matrix-vector product into `out`.
    pub fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
        let values = self.sparse.mul_vec(x);
        out.copy_from_slice(&values);
    }

    /// Build from a scalar sparse matrix and a block layout.
    pub fn from_parts(
        sparse: SparseMatrix<S>,
        block_sizes: Vec<usize>,
        block_counts: Vec<usize>,
    ) -> Self {
        Self {
            sparse,
            block_sizes,
            block_counts,
        }
    }

    /// Block sizes.
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }

    /// Per-type block counts.
    pub fn block_counts(&self) -> &[usize] {
        &self.block_counts
    }

    /// Scalar dimension.
    pub fn dims(&self) -> usize {
        self.sparse.nrows()
    }

    /// Coefficient accessor.
    pub fn get(&self, i: usize, j: usize) -> S {
        self.sparse.coeff(i, j)
    }

    /// Underlying sparse matrix.
    pub fn to_sparse(&self) -> &SparseMatrix<S> {
        &self.sparse
    }

    /// Dense conversion.
    pub fn to_dense(&self) -> Matrix<S, Dynamic> {
        self.sparse.to_dense()
    }

    /// Cast to another precision.
    pub fn cast<T: Real>(&self) -> BlockSparseMatX<T> {
        BlockSparseMatX {
            sparse: self.sparse.cast::<T>(),
            block_sizes: self.block_sizes.clone(),
            block_counts: self.block_counts.clone(),
        }
    }
}

// --- LDLT with Schur complement --------------------------------------------

/// Block-sparse LDLᵀ with a sparse Schur complement on the reduced head block.
///
/// The eliminated block is factorized sparsely (AMD ordering by default) and the
/// reduced system stays dense (`num_reduced²`), mirroring the scalar Schur path
/// of the C++ solver. Memory is `O(nnz)` instead of the former dense partitions.
#[derive(Clone, Debug)]
pub struct BlockSparseLdlt<T: BlockTypes, S: Real = f64> {
    num_reduced: usize,
    analyzed: bool,
    computed: bool,
    solver: SparseLinearSolvers<S>,
    sparse: SparseMatrix<S>,
    _marker: PhantomData<T>,
}

impl<T: BlockTypes, S: Real> Default for BlockSparseLdlt<T, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: BlockTypes, S: Real> BlockSparseLdlt<T, S> {
    /// Create an empty factorization.
    pub fn new() -> Self {
        Self {
            num_reduced: 0,
            analyzed: false,
            computed: false,
            solver: SparseLinearSolvers::new(LinearSolverOptions::default()),
            sparse: SparseMatrix::new(0, 0),
            _marker: PhantomData,
        }
    }

    /// Analyze the block structure; the split must fall on a type boundary.
    pub fn analyze(&mut self, h: &BlockSparseMat<T, S>, num_reduced: usize) -> bool {
        if h.nrows() != h.ncols() || h.dims() == 0 || num_reduced >= h.dims() {
            self.analyzed = false;
            return false;
        }
        let mut boundary_ok = num_reduced == 0;
        for t in 0..h.num_block_types() {
            if h.base_offset(t) == num_reduced {
                boundary_ok = true;
            }
        }
        if !boundary_ok {
            self.analyzed = false;
            return false;
        }
        self.num_reduced = num_reduced;
        self.sparse = h.to_sparse();
        self.analyzed = self.solver.analyze(&self.sparse, num_reduced);
        self.computed = false;
        self.analyzed
    }

    /// Numeric factorization, reusing the analyzed pattern.
    pub fn compute(&mut self, h: &BlockSparseMat<T, S>) -> bool {
        if !self.analyzed {
            return false;
        }
        // Refresh the values; the sparsity pattern is unchanged by the optimizer.
        self.sparse = h.to_sparse();
        self.computed = self.solver.compute(&self.sparse);
        self.computed
    }

    /// Solve `H x = rhs` into `out`.
    pub fn solve_into(&self, _h: &BlockSparseMat<T, S>, rhs: &[S], out: &mut [S]) -> bool {
        if !self.computed {
            return false;
        }
        match self.solver.solve(&self.sparse, rhs) {
            Some(x) => {
                out.copy_from_slice(&x);
                true
            }
            None => false,
        }
    }

    /// Inverse of the reduced Schur block (top-left block of `H⁻¹`).
    pub fn inverse_schur_reduced(&self) -> Option<Matrix<S, Dynamic>> {
        self.inverse_block(0, self.num_reduced)
    }

    /// Marginal covariance of one eliminated block (top-left block of `H⁻¹`).
    pub fn marginal_eliminated_block(
        &self,
        elim_start: usize,
        elim_size: usize,
    ) -> Option<Matrix<S, Dynamic>> {
        self.inverse_block(elim_start, elim_size)
    }

    /// Inverse of the diagonal block `H[start..start+size, start..start+size]`.
    fn inverse_block(&self, start: usize, size: usize) -> Option<Matrix<S, Dynamic>> {
        if !self.computed || size == 0 || start + size > self.sparse.nrows() {
            return None;
        }
        let n = self.sparse.nrows();
        let mut out = zeros_matrix::<S, Dynamic>(size, size);
        let mut e = vec![S::zero(); n];
        for j in 0..size {
            e[start + j] = S::one();
            let x = self.solver.solve(&self.sparse, &e)?;
            e[start + j] = S::zero();
            for i in 0..size {
                out.set_elem(i, j, x[start + i]);
            }
        }
        Some(out)
    }

    /// Whether the factorization is available.
    pub fn is_computed(&self) -> bool {
        self.computed
    }

    /// Number of reduced parameters.
    pub fn num_reduced(&self) -> usize {
        self.num_reduced
    }
}

// --- HessianStorage implementation -----------------------------------------

impl<T: BlockTypes, S: Real> HessianStorage<S> for BlockSparseMat<T, S> {
    fn zeros(_dims: usize) -> Self {
        Self::new()
    }

    fn zeros_like(prototype: &Self, _dims: usize) -> Self {
        Self::with_counts(prototype.block_counts())
    }

    fn dims(&self) -> usize {
        BlockSparseMat::<T, S>::dims(self)
    }

    fn set_zero(&mut self) {
        BlockSparseMat::set_zero(self);
    }

    fn diagonal_into(&self, out: &mut [S]) {
        BlockSparseMat::diagonal_into(self, out);
    }

    fn scale_diagonal(&mut self, factor: S) {
        BlockSparseMat::scale_diagonal(self, factor);
    }

    fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
        BlockSparseMat::scale_by_diagonal(self, scale, inverse);
    }

    fn apply_fixed_indices(&mut self, indices: &[usize]) {
        BlockSparseMat::apply_fixed_indices(self, indices);
    }

    fn is_all_finite(&self) -> bool {
        BlockSparseMat::is_all_finite(self)
    }

    fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
        BlockSparseMat::mul_vec_into(self, x, out);
    }

    fn solve_into(
        &self,
        options: &LinearSolverOptions,
        rhs: &[S],
        num_reduced: usize,
        out: &mut [S],
    ) -> bool {
        // Convert once per solve and use the scalar sparse Schur solver.
        let mut solver = SparseLinearSolvers::<S>::new(options.clone());
        let sparse = self.to_sparse();
        if !solver.analyze(&sparse, num_reduced) || !solver.compute(&sparse) {
            return false;
        }
        match solver.solve(&sparse, rhs) {
            Some(x) => {
                out.copy_from_slice(&x);
                true
            }
            None => false,
        }
    }

    fn to_dense(&self) -> Matrix<S, Dynamic> {
        BlockSparseMat::to_dense(self)
    }

    fn to_final_covariance(
        &self,
        options: &CovarianceOptions,
        cost: Cost,
        fixed: &[usize],
        num_reduced: usize,
    ) -> Option<FinalCovariance> {
        let sparse = self.to_sparse();
        let cov = crate::covariance::compute_sparse_covariance(
            &sparse,
            options,
            cost,
            fixed,
            sparse.nrows(),
            num_reduced,
        )?;
        let (sizes, counts) = self.layout();
        match cov {
            FinalCovariance::Dense(dense) => Some(FinalCovariance::Dense(dense)),
            FinalCovariance::Sparse(cov_sparse) => Some(FinalCovariance::BlockSparse(
                BlockSparseMatX::from_parts(cov_sparse, sizes, counts),
            )),
            other => Some(other),
        }
    }
}

impl<S: Real> HessianStorage<S> for BlockSparseMatX<S> {
    fn zeros(dims: usize) -> Self {
        Self::new(&[1], &[dims])
    }

    fn zeros_like(prototype: &Self, _dims: usize) -> Self {
        Self::new(prototype.block_sizes(), prototype.block_counts())
    }

    fn dims(&self) -> usize {
        BlockSparseMatX::dims(self)
    }

    fn set_zero(&mut self) {
        self.sparse.set_zero();
    }

    fn diagonal_into(&self, out: &mut [S]) {
        let diag = self.sparse.diagonal();
        for (o, v) in out.iter_mut().zip(diag) {
            *o = v;
        }
    }

    fn scale_diagonal(&mut self, factor: S) {
        self.sparse.scale_diagonal(factor);
    }

    fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
        self.sparse.scale_by_diagonal(scale, inverse);
    }

    fn apply_fixed_indices(&mut self, indices: &[usize]) {
        self.sparse.apply_fixed_indices(indices);
    }

    fn is_all_finite(&self) -> bool {
        self.sparse.is_all_finite()
    }

    fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
        BlockSparseMatX::mul_vec_into(self, x, out);
    }

    fn solve_into(
        &self,
        options: &LinearSolverOptions,
        rhs: &[S],
        num_reduced: usize,
        out: &mut [S],
    ) -> bool {
        let mut solver = SparseLinearSolvers::<S>::new(options.clone());
        if !solver.analyze(&self.sparse, num_reduced) || !solver.compute(&self.sparse) {
            return false;
        }
        match solver.solve(&self.sparse, rhs) {
            Some(x) => {
                out.copy_from_slice(&x);
                true
            }
            None => false,
        }
    }

    fn to_dense(&self) -> Matrix<S, Dynamic> {
        self.sparse.to_dense()
    }

    fn to_final_covariance(
        &self,
        options: &CovarianceOptions,
        cost: Cost,
        fixed: &[usize],
        num_reduced: usize,
    ) -> Option<FinalCovariance> {
        let cov = crate::covariance::compute_sparse_covariance(
            &self.sparse,
            options,
            cost,
            fixed,
            self.sparse.nrows(),
            num_reduced,
        )?;
        match cov {
            FinalCovariance::Dense(dense) => Some(FinalCovariance::Dense(dense)),
            FinalCovariance::Sparse(cov_sparse) => {
                Some(FinalCovariance::BlockSparse(BlockSparseMatX::from_parts(
                    cov_sparse,
                    self.block_sizes.clone(),
                    self.block_counts.clone(),
                )))
            }
            other => Some(other),
        }
    }
}
