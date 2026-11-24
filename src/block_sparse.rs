// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Compile-time block-sparse symmetric matrices, mirroring tinyopt's
//! `BlockSparseMatrix<Scalar, BlockSizes...>` (upper type pairs only).
//!
//! Block sizes are declared with `block_sparse_types!`, which also generates
//! the compile-time `HasPair<R, C>` checks used by [`BlockSparseMat::block_mut`].

use std::collections::HashMap;
use std::marker::PhantomData;

use crate::backend::{DenseMatrix, DenseVector, Dynamic, Matrix, zeros_matrix, zeros_vector};
use crate::block_tridiag::BlockMutN;
use crate::cost::Cost;
use crate::covariance::{CovarianceOptions, FinalCovariance};
use crate::hessian::HessianStorage;
use crate::options::LinearSolverOptions;
use crate::real::Real;
use crate::sparse::{SparseLinearSolvers, SparseMatrix};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

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

/// Maximum size of a dense block-index grid (same order of magnitude as the
/// C++ `kMaxDenseGridElements`); larger pairs fall back to a hash map.
#[cfg(not(test))]
const MAX_DENSE_GRID: usize = 16_000_000;
/// Unit tests exercise the hash fallback with a tiny grid budget.
#[cfg(test)]
const MAX_DENSE_GRID: usize = 8;

/// Blocks of one upper-triangle type pair.
///
/// Blocks are stored contiguously in a single `Vec` (no per-block allocation,
/// like the C++ `std::vector<Matrix<R, C>>`), and resolved in O(1) through a
/// diagonal grid and a dense row-major index grid, with a hash fallback for
/// very large pairs.
#[derive(Clone, Debug)]
struct PairStorage<S> {
    rows: usize,
    cols: usize,
    block_size: usize,
    diagonal: bool,
    coords: Vec<(u32, u32)>,
    data: Vec<S>,
    diag_grid: Vec<i32>,
    grid: Vec<i32>,
    sparse: HashMap<u64, u32>,
}

impl<S: Real> PairStorage<S> {
    fn new(rows: usize, cols: usize, block_size: usize, diagonal: bool) -> Self {
        Self {
            rows,
            cols,
            block_size,
            diagonal,
            coords: Vec::new(),
            data: Vec::new(),
            diag_grid: Vec::new(),
            grid: Vec::new(),
            sparse: HashMap::new(),
        }
    }

    #[inline]
    fn key(r: usize, c: usize) -> u64 {
        ((r as u64) << 32) | c as u64
    }

    fn len(&self) -> usize {
        self.coords.len()
    }

    fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Grow the block grid without touching existing blocks.
    fn resize(&mut self, rows: usize, cols: usize) {
        if rows <= self.rows && cols <= self.cols {
            return;
        }
        self.rows = self.rows.max(rows);
        self.cols = self.cols.max(cols);
        if !self.grid.is_empty() || !self.sparse.is_empty() {
            self.rebuild_grids();
        }
    }

    fn rebuild_grids(&mut self) {
        self.diag_grid = if self.diagonal {
            vec![-1; self.rows]
        } else {
            Vec::new()
        };
        self.grid.clear();
        self.sparse.clear();
        let has_off_diagonal = self.coords.iter().any(|&(r, c)| !(self.diagonal && r == c));
        let cells = self.rows.saturating_mul(self.cols);
        if has_off_diagonal && cells > 0 && cells <= MAX_DENSE_GRID {
            self.grid = vec![-1; cells];
        }
        for (index, &(r, c)) in self.coords.iter().enumerate() {
            let (r, c) = (r as usize, c as usize);
            if self.diagonal && r == c {
                if r < self.diag_grid.len() {
                    self.diag_grid[r] = index as i32;
                }
            } else if !self.grid.is_empty() {
                self.grid[r * self.cols + c] = index as i32;
            } else {
                self.sparse.insert(Self::key(r, c), index as u32);
            }
        }
    }

    /// Block index of `(r, c)`, if it exists.
    fn index(&self, r: usize, c: usize) -> Option<usize> {
        if r >= self.rows || c >= self.cols {
            return None;
        }
        if self.diagonal && r == c {
            let value = self.diag_grid.get(r).copied().unwrap_or(-1);
            return (value >= 0).then_some(value as usize);
        }
        if !self.grid.is_empty() {
            let value = self.grid[r * self.cols + c];
            return (value >= 0).then_some(value as usize);
        }
        self.sparse.get(&Self::key(r, c)).map(|&v| v as usize)
    }

    /// Block data of `(r, c)`, if it exists.
    fn get(&self, r: usize, c: usize) -> Option<&[S]> {
        let index = self.index(r, c)?;
        let start = index * self.block_size;
        Some(&self.data[start..start + self.block_size])
    }

    /// Block data of `(r, c)`, creating a zero block when needed.
    fn get_or_insert(&mut self, r: usize, c: usize) -> &mut [S] {
        if let Some(index) = self.index(r, c) {
            let start = index * self.block_size;
            return &mut self.data[start..start + self.block_size];
        }
        if r >= self.rows || c >= self.cols {
            self.resize(r + 1, c + 1);
        }
        let index = self.coords.len();
        self.coords.push((r as u32, c as u32));
        let start = index * self.block_size;
        self.data.resize(start + self.block_size, S::zero());
        if self.diagonal && r == c {
            if r >= self.diag_grid.len() {
                self.diag_grid.resize(r + 1, -1);
            }
            self.diag_grid[r] = index as i32;
        } else {
            // Lazily allocate the dense index grid on the first off-diagonal
            // block; oversized pairs keep using the hash map.
            if self.grid.is_empty() && self.sparse.is_empty() {
                let cells = self.rows.saturating_mul(self.cols);
                if cells > 0 && cells <= MAX_DENSE_GRID {
                    self.grid = vec![-1; cells];
                }
            }
            if !self.grid.is_empty() {
                self.grid[r * self.cols + c] = index as i32;
            } else {
                self.sparse.insert(Self::key(r, c), index as u32);
            }
        }
        &mut self.data[start..start + self.block_size]
    }

    fn zero(&mut self) {
        self.data.iter_mut().for_each(|v| *v = S::zero());
    }

    fn iter(&self) -> impl Iterator<Item = ((usize, usize), &[S])> {
        self.coords
            .iter()
            .zip(self.data.chunks_exact(self.block_size))
            .map(|(&(r, c), block)| ((r as usize, c as usize), block))
    }

    /// Block indices sorted by `(row, col)`, which yields sorted rows inside
    /// each CSC column when converting to a scalar sparse matrix.
    fn sorted_order(&self) -> Vec<u32> {
        let mut order: Vec<u32> = (0..self.coords.len() as u32).collect();
        if !self.coords.windows(2).all(|w| w[0] <= w[1]) {
            order.sort_unstable_by_key(|&i| self.coords[i as usize]);
        }
        order
    }

    fn block_at(&self, index: usize) -> &[S] {
        let start = index * self.block_size;
        &self.data[start..start + self.block_size]
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut [S])> {
        let block_size = self.block_size;
        self.coords
            .iter()
            .zip(self.data.chunks_exact_mut(block_size))
            .map(|(&(r, c), block)| ((r as usize, c as usize), block))
    }
}

/// Symmetric block-sparse matrix with compile-time block sizes.
pub struct BlockSparseMat<T: BlockTypes, S: Real = f64> {
    sizes: &'static [usize],
    counts: Vec<usize>,
    blocks: Vec<PairStorage<S>>,
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

    fn pair_types(pair: usize) -> (usize, usize) {
        let nt = Self::num_types();
        for lo in 0..nt {
            let span = nt - lo;
            if pair < lo * nt - lo * lo.saturating_sub(1) / 2 + span {
                let hi = lo + (pair - (lo * nt - lo * lo.saturating_sub(1) / 2));
                return (lo, hi);
            }
        }
        (nt - 1, nt - 1)
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
            blocks: (0..Self::num_pairs())
                .map(|pair| {
                    let (a, b) = Self::pair_types(pair);
                    PairStorage::new(counts[a], counts[b], T::SIZES[a] * T::SIZES[b], a == b)
                })
                .collect(),
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
    ) -> BlockMutN<'_, S, R, C>
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
        let pair = Self::pair_index(ta, tb);
        let (rows, cols) = (self.counts[ta], self.counts[tb]);
        self.blocks[pair].resize(rows, cols);
        let data = self.blocks[pair].get_or_insert(r, c);
        BlockMutN::from_data(data)
    }

    /// Mutable diagonal block `(i, i)` of a given block size.
    pub fn diagonal_mut<const N: usize>(&mut self, i: usize) -> BlockMutN<'_, S, N, N>
    where
        T: HasPair<N, N>,
    {
        self.block_mut::<N, N>(i, i)
    }

    /// Zero all blocks, keeping the structure.
    pub fn set_zero(&mut self) {
        for storage in &mut self.blocks {
            storage.zero();
        }
    }

    /// Add another block-sparse matrix, growing this matrix's block counts and
    /// sparsity pattern as needed.
    pub fn add_assign(&mut self, other: &Self) {
        assert_eq!(self.sizes, other.sizes, "block sizes must match");
        for (target_count, source_count) in self.counts.iter_mut().zip(&other.counts) {
            *target_count = (*target_count).max(*source_count);
        }
        for pair in 0..self.blocks.len() {
            let (row_type, col_type) = Self::pair_types(pair);
            self.blocks[pair].resize(self.counts[row_type], self.counts[col_type]);
            for ((row, col), source) in other.blocks[pair].iter() {
                let target = self.blocks[pair].get_or_insert(row, col);
                for (target, source) in target.iter_mut().zip(source) {
                    *target += *source;
                }
            }
        }
    }

    /// Set the diagonal blocks of every type to identity.
    pub fn set_identity(&mut self) {
        self.set_zero();
        for t in 0..Self::num_types() {
            let size = self.sizes[t];
            let pair = Self::pair_index(t, t);
            let count = self.counts[t];
            let storage = &mut self.blocks[pair];
            storage.resize(count, count);
            for i in 0..count {
                let block = storage.get_or_insert(i, i);
                block.iter_mut().for_each(|v| *v = S::zero());
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
        let dims = self.dims();
        let mut is_fixed = vec![false; dims];
        for &i in indices {
            if i < dims {
                is_fixed[i] = true;
            }
        }
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
                for ((r, c), block) in self.blocks[pair].iter_mut() {
                    let row_base = oa + r * ra;
                    let col_base = ob + c * ca;
                    for j in 0..ca {
                        for i in 0..ra {
                            let gi = row_base + i;
                            let gj = col_base + j;
                            if a == b && r == c && i == j {
                                if is_fixed[gi] {
                                    block[j * ra + i] = S::one();
                                }
                            } else if is_fixed[gi] || is_fixed[gj] {
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
            .all(|storage| storage.data.iter().all(|v| v.is_finite()))
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
        let (pair, key, transpose) = if ta < tb {
            (Self::pair_index(ta, tb), (r, c), false)
        } else if ta > tb {
            (Self::pair_index(tb, ta), (c, r), true)
        } else if r <= c {
            (Self::pair_index(ta, tb), (r, c), false)
        } else {
            (Self::pair_index(ta, tb), (c, r), true)
        };
        match self.blocks[pair].get(key.0, key.1) {
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
        let (pair, key, transpose) = if ta < tb {
            (Self::pair_index(ta, tb), (r, c), false)
        } else if ta > tb {
            (Self::pair_index(tb, ta), (c, r), true)
        } else if r <= c {
            (Self::pair_index(ta, tb), (r, c), false)
        } else {
            (Self::pair_index(ta, tb), (c, r), true)
        };
        let rows = if ta < tb { ra } else { ca };
        let cols = if ta < tb { ca } else { ra };
        let block = self.blocks[pair].get_or_insert(key.0, key.1);
        debug_assert_eq!(block.len(), rows * cols);
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
            for ((r, c), block) in self.blocks[pair].iter_mut() {
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
                for ((r, c), block) in self.blocks[pair].iter_mut() {
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
                for ((r, c), block) in self.blocks[pair].iter() {
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
    ///
    /// Two passes (count, then fill) build the CSC arrays directly: the block
    /// iteration order already yields sorted rows inside each column.
    pub fn to_sparse(&self) -> SparseMatrix<S> {
        let n = self.dims();
        let mut offsets = Vec::with_capacity(Self::num_types());
        let mut off = 0;
        for t in 0..Self::num_types() {
            offsets.push(off);
            off += self.counts[t] * self.sizes[t];
        }

        let mut counts = vec![0usize; n];
        for a in 0..Self::num_types() {
            for b in a..Self::num_types() {
                let pair = Self::pair_index(a, b);
                let (ra, ca) = (self.sizes[a], self.sizes[b]);
                let (oa, ob) = (offsets[a], offsets[b]);
                let storage = &self.blocks[pair];
                for index in storage.sorted_order() {
                    let (r, c) = storage.coords[index as usize];
                    let (r, c) = (r as usize, c as usize);
                    let block = storage.block_at(index as usize);
                    let row_base = oa + r * ra;
                    let col_base = ob + c * ca;
                    for j in 0..ca {
                        for i in 0..ra {
                            if block[j * ra + i] != S::zero() {
                                counts[col_base + j] += 1;
                                if a != b || r != c {
                                    counts[row_base + i] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut col_ptrs = vec![0usize; n + 1];
        for j in 0..n {
            col_ptrs[j + 1] = col_ptrs[j] + counts[j];
        }
        let total = col_ptrs[n];
        let mut pos = col_ptrs.clone();
        let mut row_indices = vec![0usize; total];
        let mut values = vec![S::zero(); total];

        for a in 0..Self::num_types() {
            for b in a..Self::num_types() {
                let pair = Self::pair_index(a, b);
                let (ra, ca) = (self.sizes[a], self.sizes[b]);
                let (oa, ob) = (offsets[a], offsets[b]);
                let storage = &self.blocks[pair];
                for index in storage.sorted_order() {
                    let (r, c) = storage.coords[index as usize];
                    let (r, c) = (r as usize, c as usize);
                    let block = storage.block_at(index as usize);
                    let row_base = oa + r * ra;
                    let col_base = ob + c * ca;
                    for j in 0..ca {
                        for i in 0..ra {
                            let v = block[j * ra + i];
                            if v == S::zero() {
                                continue;
                            }
                            let col = col_base + j;
                            let p = pos[col];
                            pos[col] += 1;
                            row_indices[p] = row_base + i;
                            values[p] = v;
                            // Mirror off-diagonal blocks (different types or different
                            // blocks of the same type).
                            if a != b || r != c {
                                let col = row_base + i;
                                let p = pos[col];
                                pos[col] += 1;
                                row_indices[p] = col_base + j;
                                values[p] = v;
                            }
                        }
                    }
                }
            }
        }
        SparseMatrix::from_csc_parts(n, n, col_ptrs, row_indices, values)
    }

    /// Number of allocated blocks.
    pub fn num_allocated_blocks(&self) -> usize {
        self.blocks.iter().map(PairStorage::len).sum()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.iter().all(PairStorage::is_empty)
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

// --- Block LDLT with Schur complement ---------------------------------------

/// Dense inverse of a small column-major block (allocates nothing).
fn inverse_block<S: Real>(block: &[S], size: usize, out: &mut [S]) -> bool {
    const MAX: usize = 8;
    if size > MAX || out.len() != size * size {
        return false;
    }
    let mut a = [[S::zero(); MAX]; MAX];
    let mut inv = [[S::zero(); MAX]; MAX];
    for j in 0..size {
        for i in 0..size {
            a[i][j] = block[j * size + i];
            inv[i][j] = if i == j { S::one() } else { S::zero() };
        }
    }
    for col in 0..size {
        let mut pivot = col;
        let mut best = num_traits::Float::abs(a[col][col]);
        for (r, row) in a.iter().enumerate().take(size).skip(col + 1) {
            let v = num_traits::Float::abs(row[col]);
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best <= S::epsilon() || !best.is_finite() {
            return false;
        }
        if pivot != col {
            a.swap(col, pivot);
            inv.swap(col, pivot);
        }
        let d = a[col][col];
        for j in 0..size {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..size {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == S::zero() {
                continue;
            }
            for j in 0..size {
                a[r][j] -= f * a[col][j];
                inv[r][j] -= f * inv[col][j];
            }
        }
    }
    for j in 0..size {
        for i in 0..size {
            out[j * size + i] = inv[i][j];
        }
    }
    out.iter().all(|v| v.is_finite())
}

/// Dense matrix-vector product with a column-major square block.
fn block_mat_vec_into<S: Real>(m: &[S], v: &[S], size: usize, out: &mut [S]) {
    for i in 0..size {
        let mut acc = S::zero();
        for j in 0..size {
            acc += m[j * size + i] * v[j];
        }
        out[i] = acc;
    }
}

#[derive(Clone, Debug)]
struct EliminatedBlock {
    start: usize,
    size: usize,
    /// Offset of the `size × size` inverse in `BlockSparseLdlt::inverses`.
    inv_start: usize,
    /// Range of couplings in `BlockSparseLdlt::couplings`.
    coupling_start: usize,
    coupling_end: usize,
    /// Type pair storing the diagonal block.
    pair: usize,
    /// Diagonal block index.
    block: usize,
}

#[derive(Clone, Debug)]
struct Coupling {
    reduced_start: usize,
    rows: usize,
    data_start: usize,
    len: usize,
    /// Type pair storing the coupling block, and its block coordinates.
    pair: usize,
    block: (usize, usize),
}

#[derive(Clone, Debug)]
struct SchurScratch<S> {
    product: Vec<S>,
    product2: Vec<S>,
}

impl<S> Default for SchurScratch<S> {
    fn default() -> Self {
        Self {
            product: Vec::new(),
            product2: Vec::new(),
        }
    }
}

fn accumulate_eliminated_schur<S: Real>(
    elims: &[EliminatedBlock],
    inverses: &[S],
    couplings: &[Coupling],
    coupling_data: &[S],
    scratch: &mut SchurScratch<S>,
    schur: &mut [S],
    nr: usize,
) {
    for e in elims {
        let k = e.size;
        let inv = &inverses[e.inv_start..e.inv_start + k * k];
        for i in e.coupling_start..e.coupling_end {
            let ci = &couplings[i];
            let c0 = &coupling_data[ci.data_start..ci.data_start + ci.len];
            scratch.product.clear();
            scratch.product.resize(ci.rows * k, S::zero());
            block_mat_mul_inv(c0, inv, ci.rows, k, &mut scratch.product);
            let t0 = &scratch.product[..];
            for cj in &couplings[i..e.coupling_end] {
                let c1 = &coupling_data[cj.data_start..cj.data_start + cj.len];
                if accumulate_pair_tile(
                    t0,
                    c1,
                    schur,
                    ci.rows,
                    cj.rows,
                    k,
                    ci.reduced_start,
                    cj.reduced_start,
                    nr,
                ) {
                    continue;
                }
                // P = (C_i E^{-1})^T C_j = t0^T C_j (ci.rows x cj.rows),
                // accumulated contiguously before updating S.
                let p_len = ci.rows * cj.rows;
                scratch.product2.clear();
                scratch.product2.resize(p_len, S::zero());
                for a in 0..ci.rows {
                    let row = &mut scratch.product2[a * cj.rows..a * cj.rows + cj.rows];
                    for q in 0..k {
                        let t = t0[q * ci.rows + a];
                        if t == S::zero() {
                            continue;
                        }
                        let c1q = &c1[q * cj.rows..q * cj.rows + cj.rows];
                        for (value, c) in row.iter_mut().zip(c1q) {
                            *value += t * *c;
                        }
                    }
                }
                for a in 0..ci.rows {
                    let row_a = (ci.reduced_start + a) * nr + cj.reduced_start;
                    for b in 0..cj.rows {
                        schur[row_a + b] -= scratch.product2[a * cj.rows + b];
                    }
                }
            }
        }
    }
}

/// Block-sparse LDLᵀ with a block Schur complement on the reduced head block.
///
/// When the eliminated blocks are independent (no eliminated-eliminated
/// coupling, the bundle-adjustment structure), the Schur complement is built
/// with dense per-block operations: `S = H_rr − Σ C_e E_e⁻¹ C_eᵀ`. Otherwise
/// the solve falls back to the scalar sparse Schur path.
#[derive(Clone, Debug)]
pub struct BlockSparseLdlt<T: BlockTypes, S: Real = f64> {
    num_reduced: usize,
    analyzed: bool,
    computed: bool,
    reduced_types: usize,
    schur: Option<Vec<S>>,
    elims: Vec<EliminatedBlock>,
    inverses: Vec<S>,
    couplings: Vec<Coupling>,
    coupling_data: Vec<S>,
    fallback: Option<SparseLinearSolvers<S>>,
    fallback_sparse: SparseMatrix<S>,
    /// Per-pair block counts of the analyzed structure (re-analysis is skipped
    /// when it is unchanged).
    pattern: Option<Vec<usize>>,
    scratch: SchurScratch<S>,
    ys: Vec<S>,
    correction: Vec<S>,
    solution: Vec<S>,
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
            reduced_types: 0,
            schur: None,
            elims: Vec::new(),
            inverses: Vec::new(),
            couplings: Vec::new(),
            coupling_data: Vec::new(),
            fallback: None,
            fallback_sparse: SparseMatrix::new(0, 0),
            pattern: None,
            scratch: SchurScratch::default(),
            ys: Vec::new(),
            correction: Vec::new(),
            solution: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Analyze the block structure; the split must fall on a type boundary.
    ///
    /// The analysis is skipped when the block pattern and split are unchanged,
    /// so the optimizer only pays for it on the first evaluation.
    pub fn analyze(&mut self, h: &BlockSparseMat<T, S>, num_reduced: usize) -> bool {
        if h.nrows() != h.ncols() || h.dims() == 0 || num_reduced >= h.dims() {
            self.analyzed = false;
            self.pattern = None;
            return false;
        }
        let mut reduced_types = None;
        for t in 0..h.num_block_types() {
            if h.base_offset(t) == num_reduced {
                reduced_types = Some(t);
            }
        }
        let Some(reduced_types) = reduced_types else {
            self.analyzed = false;
            self.pattern = None;
            return false;
        };

        let fingerprint: Vec<usize> = h.blocks.iter().map(PairStorage::len).collect();
        if self.analyzed
            && self.num_reduced == num_reduced
            && self.reduced_types == reduced_types
            && self.pattern.as_ref() == Some(&fingerprint)
        {
            return true;
        }

        self.num_reduced = num_reduced;
        self.reduced_types = reduced_types;
        // Keep the Schur buffer for reuse; `compute` overwrites it entirely.
        self.computed = false;
        self.elims.clear();
        self.couplings.clear();
        self.inverses.clear();
        self.coupling_data.clear();
        self.fallback = None;
        self.pattern = Some(fingerprint);

        // Block Schur requires independent eliminated blocks.
        let independent = (reduced_types..h.num_block_types()).all(|a| {
            (a..h.num_block_types()).all(|b| {
                let storage = &h.blocks[Self::block_pair(a, b)];
                if storage.is_empty() {
                    return true;
                }
                a != b && storage.is_empty()
                    || (a == b && storage.coords.iter().all(|&(r, c)| r == c))
            })
        });
        self.analyzed = true;
        if num_reduced == 0 || !independent {
            // Fallback: scalar sparse Schur (handles coupling and nr == 0).
            self.fallback_sparse = h.to_sparse();
            let mut solver = SparseLinearSolvers::new(LinearSolverOptions::default());
            let ok = solver.analyze(&self.fallback_sparse, num_reduced);
            self.fallback = if ok { Some(solver) } else { None };
            return ok;
        }

        // Structural skeleton: eliminated blocks and their couplings. The
        // values are refreshed by `compute` without touching this layout.
        let nt = T::SIZES.len();
        for te in reduced_types..nt {
            let k = T::SIZES[te];
            let base = h.base_offset(te);
            let pair = Self::block_pair(te, te);
            for be in 0..h.counts[te] {
                let start = base + be * k;
                let inv_start = self.inverses.len();
                self.inverses.resize(inv_start + k * k, S::zero());
                let coupling_start = self.couplings.len();
                for tr in 0..reduced_types {
                    let rs = T::SIZES[tr];
                    let cp = Self::block_pair(tr, te);
                    for br in 0..h.counts[tr] {
                        let Some(c) = h.blocks[cp].get(br, be) else {
                            continue;
                        };
                        let data_start = self.coupling_data.len();
                        self.coupling_data.resize(data_start + c.len(), S::zero());
                        self.couplings.push(Coupling {
                            reduced_start: h.base_offset(tr) + br * rs,
                            rows: rs,
                            data_start,
                            len: c.len(),
                            pair: cp,
                            block: (br, be),
                        });
                    }
                }
                self.elims.push(EliminatedBlock {
                    start,
                    size: k,
                    inv_start,
                    coupling_start,
                    coupling_end: self.couplings.len(),
                    pair,
                    block: be,
                });
            }
        }
        true
    }

    fn block_pair(a: usize, b: usize) -> usize {
        let nt = T::SIZES.len();
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        lo * nt - lo * lo.saturating_sub(1) / 2 + (hi - lo)
    }

    /// Numeric factorization.
    pub fn compute(&mut self, h: &BlockSparseMat<T, S>) -> bool {
        if !self.analyzed {
            return false;
        }
        self.computed = false;
        if let Some(solver) = self.fallback.as_mut() {
            self.fallback_sparse = h.to_sparse();
            self.computed = solver.compute(&self.fallback_sparse);
            return self.computed;
        }

        let nr = self.num_reduced;
        let rt = self.reduced_types;
        let Self {
            schur,
            elims,
            inverses,
            couplings,
            coupling_data,
            scratch,
            ..
        } = self;
        let schur = schur.get_or_insert_with(|| vec![S::zero(); nr * nr]);
        if schur.len() != nr * nr {
            schur.resize(nr * nr, S::zero());
        }
        schur.fill(S::zero());

        // Reduced-reduced blocks (both triangles).
        for a in 0..rt {
            for b in a..rt {
                let (ra, ca) = (T::SIZES[a], T::SIZES[b]);
                let (oa, ob) = (h.base_offset(a), h.base_offset(b));
                let storage = &h.blocks[Self::block_pair(a, b)];
                for ((r, c), _) in storage.iter() {
                    let (key, transpose) = if a == b && r > c {
                        ((c, r), true)
                    } else {
                        ((r, c), false)
                    };
                    if let Some(data) = storage.get(key.0, key.1) {
                        let rows = if transpose { ca } else { ra };
                        let row0 = oa + r * ra;
                        let col0 = ob + c * ca;
                        for j in 0..ca {
                            for i in 0..ra {
                                let v = if transpose {
                                    data[i * rows + j]
                                } else {
                                    data[j * rows + i]
                                };
                                schur[(col0 + j) * nr + row0 + i] = v;
                                schur[(row0 + i) * nr + col0 + j] = v;
                            }
                        }
                    }
                }
            }
        }

        // Refresh the eliminated inverses and the coupling values.
        for e in elims.iter() {
            let Some(block) = h.blocks[e.pair].get(e.block, e.block) else {
                return false;
            };
            let k = e.size;
            if !inverse_block(block, k, &mut inverses[e.inv_start..e.inv_start + k * k]) {
                return false;
            }
        }
        for c in couplings.iter() {
            let Some(block) = h.blocks[c.pair].get(c.block.0, c.block.1) else {
                return false;
            };
            coupling_data[c.data_start..c.data_start + c.len].copy_from_slice(block);
        }

        // Each eliminated block contributes independently to the reduced
        // system. Use one dense Schur scratch per Rayon chunk to avoid locks.
        #[cfg(feature = "rayon")]
        if rayon::current_num_threads() > 1 && elims.len() >= 32 {
            let chunk_size = elims.len().div_ceil(rayon::current_num_threads());
            let partials: Vec<Vec<S>> = elims
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut partial = vec![S::zero(); nr * nr];
                    let mut scratch = SchurScratch::default();
                    accumulate_eliminated_schur(
                        chunk,
                        inverses,
                        couplings,
                        coupling_data,
                        &mut scratch,
                        &mut partial,
                        nr,
                    );
                    partial
                })
                .collect();
            for partial in partials {
                for (value, contribution) in schur.iter_mut().zip(partial) {
                    *value += contribution;
                }
            }
        } else {
            accumulate_eliminated_schur(
                elims,
                inverses,
                couplings,
                coupling_data,
                scratch,
                schur,
                nr,
            );
        }
        #[cfg(not(feature = "rayon"))]
        accumulate_eliminated_schur(
            elims,
            inverses,
            couplings,
            coupling_data,
            scratch,
            schur,
            nr,
        );

        if nr > 0 {
            for col in 1..nr {
                for row in 0..col {
                    schur[col * nr + row] = schur[row * nr + col];
                }
            }
        }
        self.computed = true;
        true
    }

    /// Solve `H x = rhs` into `out`.
    pub fn solve_into(&mut self, _h: &BlockSparseMat<T, S>, rhs: &[S], out: &mut [S]) -> bool {
        if !self.computed {
            return false;
        }
        let Self {
            fallback,
            fallback_sparse,
            ..
        } = self;
        if let Some(solver) = fallback {
            return match solver.solve(fallback_sparse, rhs) {
                Some(x) => {
                    out.copy_from_slice(&x);
                    true
                }
                None => false,
            };
        }
        match self.solve_rhs(rhs) {
            Some(x) => {
                out.copy_from_slice(x);
                true
            }
            None => false,
        }
    }

    fn solve_rhs(&mut self, rhs: &[S]) -> Option<&[S]> {
        let nr = self.num_reduced;
        let ne = rhs.len() - nr;
        self.ys.clear();
        self.ys.resize(ne, S::zero());
        self.correction.clear();
        self.correction.resize(ne, S::zero());
        self.solution.clear();
        self.solution.resize(rhs.len(), S::zero());

        let Self {
            schur,
            elims,
            inverses,
            couplings,
            coupling_data,
            ys,
            correction,
            solution,
            scratch,
            ..
        } = self;
        let schur = schur.as_ref()?;

        let mut reduced_rhs = zeros_vector::<S, Dynamic>(nr);
        for (i, value) in rhs.iter().enumerate().take(nr) {
            reduced_rhs.set_at(i, *value);
        }

        // Eliminated blocks: y = E^{-1} b and the reduced right-hand side.
        for e in elims.iter() {
            let inv = &inverses[e.inv_start..e.inv_start + e.size * e.size];
            let y = &mut ys[e.start - nr..e.start - nr + e.size];
            block_mat_vec_into(inv, &rhs[e.start..e.start + e.size], e.size, y);
            for coupling in &couplings[e.coupling_start..e.coupling_end] {
                let c = &coupling_data[coupling.data_start..coupling.data_start + coupling.len];
                for i in 0..coupling.rows {
                    let mut acc = S::zero();
                    for j in 0..e.size {
                        acc += c[j * coupling.rows + i] * y[j];
                    }
                    let index = coupling.reduced_start + i;
                    reduced_rhs.set_at(index, reduced_rhs.at(index) - acc);
                }
            }
        }

        // Dense reduced system.
        let mut dense = zeros_matrix::<S, Dynamic>(nr, nr);
        for j in 0..nr {
            for i in 0..nr {
                dense.set_elem(i, j, schur[j * nr + i]);
            }
        }
        let xr = crate::backend::solve::<S, Dynamic>(
            crate::backend::LinearSolverMethod::Ldlt,
            &dense,
            &reduced_rhs,
        )?;
        for (i, value) in solution.iter_mut().enumerate().take(nr) {
            *value = xr.at(i);
        }

        // Back-substitution of the eliminated blocks.
        for e in elims.iter() {
            let y = &ys[e.start - nr..e.start - nr + e.size];
            let t = &mut correction[e.start - nr..e.start - nr + e.size];
            t.fill(S::zero());
            for coupling in &couplings[e.coupling_start..e.coupling_end] {
                let c = &coupling_data[coupling.data_start..coupling.data_start + coupling.len];
                for j in 0..e.size {
                    let mut acc = S::zero();
                    for i in 0..coupling.rows {
                        acc += c[j * coupling.rows + i] * xr.at(coupling.reduced_start + i);
                    }
                    t[j] += acc;
                }
            }
            let inv = &inverses[e.inv_start..e.inv_start + e.size * e.size];
            scratch.product.resize(e.size, S::zero());
            block_mat_vec_into(inv, t, e.size, &mut scratch.product);
            for j in 0..e.size {
                solution[e.start + j] = y[j] - scratch.product[j];
            }
        }
        solution
            .iter()
            .all(|v| v.is_finite())
            .then_some(solution.as_slice())
    }

    /// Inverse of the reduced Schur block (top-left block of `H⁻¹`).
    pub fn inverse_schur_reduced(&self) -> Option<Matrix<S, Dynamic>> {
        let schur = self.schur.as_ref()?;
        let nr = self.num_reduced;
        let mut dense = zeros_matrix::<S, Dynamic>(nr, nr);
        for j in 0..nr {
            for i in 0..nr {
                dense.set_elem(i, j, schur[j * nr + i]);
            }
        }
        crate::backend::inverse::<S, Dynamic>(&dense)
    }

    /// Marginal covariance of one eliminated block (the matching block of `H⁻¹`).
    pub fn marginal_eliminated_block(
        &mut self,
        elim_start: usize,
        elim_size: usize,
    ) -> Option<Matrix<S, Dynamic>> {
        if !self.computed {
            return None;
        }
        let n = self.num_reduced + self.elims.iter().map(|e| e.size).sum::<usize>();
        let mut out = zeros_matrix::<S, Dynamic>(elim_size, elim_size);
        let mut rhs = vec![S::zero(); n];
        for j in 0..elim_size {
            rhs[elim_start + j] = S::one();
            let x = self.solve_rhs(&rhs)?;
            rhs[elim_start + j] = S::zero();
            for i in 0..elim_size {
                out.set_elem(i, j, x[elim_start + i]);
            }
        }
        Some(out)
    }

    /// Whether the block Schur path is in use (as opposed to the sparse
    /// fallback for coupled eliminated blocks).
    pub fn uses_block_schur(&self) -> bool {
        self.analyzed && self.fallback.is_none() && self.num_reduced > 0
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

/// `C * E^{-1}` for a `rows × k` coupling block `C` (column-major), into `out`.
fn block_mat_mul_inv<S: Real>(c: &[S], inv: &[S], rows: usize, k: usize, out: &mut [S]) {
    debug_assert_eq!(out.len(), rows * k);
    for q in 0..k {
        for i in 0..rows {
            let mut acc = S::zero();
            for p in 0..k {
                acc += c[p * rows + i] * inv[q * k + p];
            }
            out[q * rows + i] = acc;
        }
    }
}

/// Fixed-size Schur tile update `S -= (C1 E^-1)^T C2`.
///
/// The compile-time block sizes let LLVM unroll and vectorize the small
/// matrix product, matching the C++ fixed-size Eigen expression.
#[inline(always)]
fn accumulate_pair_fixed<const R: usize, const C: usize, const K: usize, S: Real>(
    t0: &[S],
    c1: &[S],
    schur: &mut [S],
    row_start: usize,
    col_start: usize,
    nr: usize,
) {
    let mut p = [[S::zero(); C]; R];
    for (a, row) in p.iter_mut().enumerate() {
        for q in 0..K {
            let t = t0[q * R + a];
            if t == S::zero() {
                continue;
            }
            let c1q = &c1[q * C..q * C + C];
            for (value, c) in row.iter_mut().zip(c1q) {
                *value += t * *c;
            }
        }
    }
    for (a, row) in p.iter().enumerate() {
        let base = (row_start + a) * nr + col_start;
        for (value, entry) in schur[base..base + C].iter_mut().zip(row) {
            *value -= *entry;
        }
    }
}

/// Dispatch the fixed-size tile update for the usual block shapes.
#[inline]
#[allow(clippy::too_many_arguments)]
fn accumulate_pair_tile<S: Real>(
    t0: &[S],
    c1: &[S],
    schur: &mut [S],
    rows: usize,
    cols: usize,
    k: usize,
    row_start: usize,
    col_start: usize,
    nr: usize,
) -> bool {
    match (rows, cols, k) {
        (6, 6, 3) => accumulate_pair_fixed::<6, 6, 3, S>(t0, c1, schur, row_start, col_start, nr),
        (6, 6, 1) => accumulate_pair_fixed::<6, 6, 1, S>(t0, c1, schur, row_start, col_start, nr),
        (6, 6, 2) => accumulate_pair_fixed::<6, 6, 2, S>(t0, c1, schur, row_start, col_start, nr),
        (6, 6, 6) => accumulate_pair_fixed::<6, 6, 6, S>(t0, c1, schur, row_start, col_start, nr),
        (3, 3, 3) => accumulate_pair_fixed::<3, 3, 3, S>(t0, c1, schur, row_start, col_start, nr),
        (3, 3, 2) => accumulate_pair_fixed::<3, 3, 2, S>(t0, c1, schur, row_start, col_start, nr),
        (2, 2, 2) => accumulate_pair_fixed::<2, 2, 2, S>(t0, c1, schur, row_start, col_start, nr),
        (2, 2, 3) => accumulate_pair_fixed::<2, 2, 3, S>(t0, c1, schur, row_start, col_start, nr),
        (3, 3, 6) => accumulate_pair_fixed::<3, 3, 6, S>(t0, c1, schur, row_start, col_start, nr),
        (6, 3, 3) => accumulate_pair_fixed::<6, 3, 3, S>(t0, c1, schur, row_start, col_start, nr),
        (3, 6, 3) => accumulate_pair_fixed::<3, 6, 3, S>(t0, c1, schur, row_start, col_start, nr),
        _ => return false,
    }
    true
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

    type SolverCache = BlockSparseLdlt<T, S>;

    fn solve_into(
        &self,
        _options: &LinearSolverOptions,
        rhs: &[S],
        num_reduced: usize,
        out: &mut [S],
        cache: &mut BlockSparseLdlt<T, S>,
    ) -> bool {
        cache.analyze(self, num_reduced) && cache.compute(self) && cache.solve_into(self, rhs, out)
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

    type SolverCache = crate::sparse::SparseSolverCache<S>;

    fn solve_into(
        &self,
        options: &LinearSolverOptions,
        rhs: &[S],
        num_reduced: usize,
        out: &mut [S],
        cache: &mut crate::sparse::SparseSolverCache<S>,
    ) -> bool {
        cache.solve_into(&self.sparse, options, rhs, num_reduced, out)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_storage_uses_a_dense_grid_for_small_pairs() {
        let mut storage = PairStorage::<f64>::new(2, 2, 4, false);
        storage.get_or_insert(1, 1)[0] = 3.0;
        assert!(!storage.grid.is_empty());
        assert!(storage.sparse.is_empty());
        assert_eq!(storage.get(1, 1).unwrap()[0], 3.0);
        assert!(storage.get(0, 0).is_none());
        // Existing blocks keep their identity after a grow.
        storage.resize(3, 3);
        assert_eq!(storage.get(1, 1).unwrap()[0], 3.0);
        storage.get_or_insert(2, 2)[0] = 4.0;
        assert_eq!(storage.get(2, 2).unwrap()[0], 4.0);
    }

    #[test]
    fn pair_storage_falls_back_to_the_hash_map_for_large_pairs() {
        // 4 x 4 = 16 cells, above the test grid budget.
        let mut storage = PairStorage::<f64>::new(4, 4, 4, false);
        storage.get_or_insert(3, 2)[0] = 5.0;
        assert!(storage.grid.is_empty());
        assert_eq!(storage.get(3, 2).unwrap()[0], 5.0);
        assert_eq!(storage.get(0, 1), None);
    }

    #[test]
    fn pair_storage_keeps_diagonal_blocks_in_the_diagonal_grid() {
        let mut storage = PairStorage::<f64>::new(8, 8, 4, true);
        storage.get_or_insert(5, 5)[1] = 2.0;
        storage.get_or_insert(1, 6)[0] = 7.0;
        assert_eq!(storage.diag_grid[5], 0);
        assert_eq!(storage.get(5, 5).unwrap()[1], 2.0);
        assert_eq!(storage.get(1, 6).unwrap()[0], 7.0);
        assert_eq!(storage.get(6, 1), None);
    }
}
