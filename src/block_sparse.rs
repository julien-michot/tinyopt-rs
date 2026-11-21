// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Compile-time block-sparse symmetric matrices, mirroring tinyopt's
//! `BlockSparseMatrix<Scalar, BlockSizes...>` (upper type pairs only).
//!
//! Block sizes are declared with `block_sparse_types!`, which also generates
//! the compile-time `HasPair<R, C>` checks used by [`BlockSparseMat::block_mut`].

use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::backend::{DenseMatrix, DenseVector, Dynamic, Matrix, zeros_matrix, zeros_vector};
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
        let (pair, key, transpose) = if ta < tb {
            (Self::pair_index(ta, tb), (r, c), false)
        } else if ta > tb {
            (Self::pair_index(tb, ta), (c, r), true)
        } else if r <= c {
            (Self::pair_index(ta, tb), (r, c), false)
        } else {
            (Self::pair_index(ta, tb), (c, r), true)
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
                for (&(r, c), block) in &self.blocks[pair] {
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
                for (&(r, c), block) in &self.blocks[pair] {
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

// --- Block LDLT with Schur complement ---------------------------------------

/// Dense inverse of a small column-major block.
fn inverse_block<S: Real>(block: &[S], size: usize) -> Option<Vec<S>> {
    let mut m = zeros_matrix::<S, Dynamic>(size, size);
    for j in 0..size {
        for i in 0..size {
            m.set_elem(i, j, block[j * size + i]);
        }
    }
    let inv = crate::backend::inverse::<S, Dynamic>(&m)?;
    let mut out = vec![S::zero(); size * size];
    for j in 0..size {
        for i in 0..size {
            let v = inv.elem(i, j);
            if !v.is_finite() {
                return None;
            }
            out[j * size + i] = v;
        }
    }
    Some(out)
}

/// Dense matrix-vector product with a column-major square block.
fn block_mat_vec<S: Real>(m: &[S], v: &[S], size: usize) -> Vec<S> {
    let mut out = vec![S::zero(); size];
    for i in 0..size {
        let mut acc = S::zero();
        for j in 0..size {
            acc += m[j * size + i] * v[j];
        }
        out[i] = acc;
    }
    out
}

#[derive(Clone, Debug)]
struct EliminatedBlock<S> {
    start: usize,
    size: usize,
    /// Inverse of the diagonal block (`size × size`, column-major).
    inv: Vec<S>,
    /// Couplings to reduced blocks: `(reduced scalar start, C)` where `C` is
    /// `reduced_size × size`, column-major.
    couplings: Vec<(usize, Vec<S>)>,
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
    elims: Vec<EliminatedBlock<S>>,
    fallback: Option<SparseLinearSolvers<S>>,
    fallback_sparse: SparseMatrix<S>,
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
            fallback: None,
            fallback_sparse: SparseMatrix::new(0, 0),
            _marker: PhantomData,
        }
    }

    /// Analyze the block structure; the split must fall on a type boundary.
    pub fn analyze(&mut self, h: &BlockSparseMat<T, S>, num_reduced: usize) -> bool {
        if h.nrows() != h.ncols() || h.dims() == 0 || num_reduced >= h.dims() {
            self.analyzed = false;
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
            return false;
        };
        self.num_reduced = num_reduced;
        self.reduced_types = reduced_types;
        self.schur = None;
        self.elims.clear();
        self.fallback = None;

        // Block Schur requires independent eliminated blocks.
        let independent = (reduced_types..h.num_block_types()).all(|a| {
            (a..h.num_block_types()).all(|b| {
                let map = &h.blocks[Self::block_pair(a, b)];
                if map.is_empty() {
                    return true;
                }
                a != b && map.is_empty() || (a == b && map.keys().all(|&(r, c)| r == c))
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
        let nt = T::SIZES.len();
        let rt = self.reduced_types;
        let mut schur = vec![S::zero(); nr * nr];

        // Reduced-reduced blocks (both triangles).
        for a in 0..rt {
            for b in a..rt {
                let (ra, ca) = (T::SIZES[a], T::SIZES[b]);
                let (oa, ob) = (h.base_offset(a), h.base_offset(b));
                for (&(r, c), block) in &h.blocks[Self::block_pair(a, b)] {
                    let (key, transpose) = if a == b && r > c {
                        ((c, r), true)
                    } else {
                        ((r, c), false)
                    };
                    if let Some(data) = h.blocks[Self::block_pair(a, b)].get(&key) {
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
                    let _ = block;
                }
            }
        }

        // Eliminated blocks: invert and accumulate C E^{-1} C^T.
        self.elims.clear();
        for te in rt..nt {
            let k = T::SIZES[te];
            let base = h.base_offset(te);
            for be in 0..h.counts[te] {
                let start = base + be * k;
                let pair = Self::block_pair(te, te);
                let e = match h.blocks[pair].get(&(be, be)) {
                    Some(block) => block.clone(),
                    None => return false,
                };
                let Some(inv) = inverse_block(&e, k) else {
                    return false;
                };
                let mut couplings = Vec::new();
                for tr in 0..rt {
                    let rs = T::SIZES[tr];
                    let pair = Self::block_pair(tr, te);
                    for br in 0..h.counts[tr] {
                        if let Some(c) = h.blocks[pair].get(&(br, be)) {
                            if c.iter().any(|v| *v != S::zero()) {
                                couplings.push((h.base_offset(tr) + br * rs, c.clone()));
                            }
                        }
                    }
                }
                // S(r1, r2) -= C1 E^{-1} C2^T.
                for (i, coupling_i) in couplings.iter().enumerate() {
                    let (r0, c0) = (&coupling_i.0, &coupling_i.1);
                    let rs0 = c0.len() / k;
                    let t0 = block_mat_mul_inv(c0, &inv, rs0, k);
                    for (j, coupling_j) in couplings.iter().enumerate().skip(i) {
                        let (r1, c1) = (&coupling_j.0, &coupling_j.1);
                        let rs1 = c1.len() / k;
                        for a in 0..rs0 {
                            for b in 0..rs1 {
                                let mut acc = S::zero();
                                for q in 0..k {
                                    acc += t0[q * rs0 + a] * c1[q * rs1 + b];
                                }
                                schur[(r0 + a) * nr + r1 + b] -= acc;
                                // Mirror across coupling blocks only: for the same
                                // block the (a, b) loops already cover both triangles.
                                if i != j {
                                    schur[(*r1 + b) * nr + (r0 + a)] -= acc;
                                }
                            }
                        }
                    }
                }
                self.elims.push(EliminatedBlock {
                    start,
                    size: k,
                    inv,
                    couplings,
                });
            }
        }

        self.schur = Some(schur);
        self.computed = true;
        true
    }

    /// Solve `H x = rhs` into `out`.
    pub fn solve_into(&self, _h: &BlockSparseMat<T, S>, rhs: &[S], out: &mut [S]) -> bool {
        if !self.computed {
            return false;
        }
        if let Some(solver) = &self.fallback {
            return match solver.solve(&self.fallback_sparse, rhs) {
                Some(x) => {
                    out.copy_from_slice(&x);
                    true
                }
                None => false,
            };
        }
        match self.solve_rhs(rhs) {
            Some(x) => {
                out.copy_from_slice(&x);
                true
            }
            None => false,
        }
    }

    fn solve_rhs(&self, rhs: &[S]) -> Option<Vec<S>> {
        let nr = self.num_reduced;
        let schur = self.schur.as_ref()?;
        let mut reduced_rhs = zeros_vector::<S, Dynamic>(nr);
        for (i, value) in rhs.iter().enumerate().take(nr) {
            reduced_rhs.set_at(i, *value);
        }
        let mut ys = Vec::with_capacity(self.elims.len());
        for e in &self.elims {
            let y = block_mat_vec(&e.inv, &rhs[e.start..e.start + e.size], e.size);
            for (r0, c) in &e.couplings {
                let rs = c.len() / e.size;
                for i in 0..rs {
                    let mut acc = S::zero();
                    for j in 0..e.size {
                        acc += c[j * rs + i] * y[j];
                    }
                    reduced_rhs.set_at(r0 + i, reduced_rhs.at(r0 + i) - acc);
                }
            }
            ys.push(y);
        }
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

        let mut out = vec![S::zero(); rhs.len()];
        for (i, value) in out.iter_mut().enumerate().take(nr) {
            *value = xr.at(i);
        }
        for (e, y) in self.elims.iter().zip(ys) {
            let mut t = vec![S::zero(); e.size];
            for (r0, c) in &e.couplings {
                let rs = c.len() / e.size;
                for j in 0..e.size {
                    let mut acc = S::zero();
                    for i in 0..rs {
                        acc += c[j * rs + i] * xr.at(r0 + i);
                    }
                    t[j] += acc;
                }
            }
            let corr = block_mat_vec(&e.inv, &t, e.size);
            for j in 0..e.size {
                out[e.start + j] = y[j] - corr[j];
            }
        }
        out.iter().all(|v| v.is_finite()).then_some(out)
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
        &self,
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

/// `C * E^{-1}` for a `rows × k` coupling block `C` (column-major).
fn block_mat_mul_inv<S: Real>(c: &[S], inv: &[S], rows: usize, k: usize) -> Vec<S> {
    let mut out = vec![S::zero(); rows * k];
    for q in 0..k {
        for i in 0..rows {
            let mut acc = S::zero();
            for p in 0..k {
                acc += c[p * rows + i] * inv[q * k + p];
            }
            out[q * rows + i] = acc;
        }
    }
    out
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
