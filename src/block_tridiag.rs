// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Symmetric block-tridiagonal matrices with fixed or dynamic block sizes,
//! mirroring tinyopt's `BlockTridiagMatrix` / `BlockTridiagMat<B>`.
//!
//! Blocks are stored flat in column-major order, so accumulation functions can
//! use the same slice-based helpers as the rest of the crate.

use crate::backend::{DenseMatrix, Dynamic, Matrix, inverse, zeros_matrix};
use crate::cost::Cost;
use crate::covariance::{CovarianceOptions, CovariancePattern, FinalCovariance};
use crate::hessian::HessianStorage;
use crate::options::LinearSolverOptions;
use crate::real::Real;

/// Mutable view over one block.
#[derive(Debug)]
pub struct BlockMut<'a, S> {
    data: &'a mut [S],
    rows: usize,
    cols: usize,
}

impl<'a, S> BlockMut<'a, S> {
    /// Wrap raw column-major block data.
    pub(crate) fn from_data(data: &'a mut [S], rows: usize, cols: usize) -> Self {
        Self { data, rows, cols }
    }
}

impl<'a, S: Real> BlockMut<'a, S> {
    /// Block shape.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Coefficient accessor.
    pub fn at(&self, i: usize, j: usize) -> S {
        self.data[j * self.rows + i]
    }

    /// Coefficient mutator.
    pub fn set_at(&mut self, i: usize, j: usize, value: S) {
        self.data[j * self.rows + i] = value;
    }

    /// `self += a * x * xᵀ`.
    pub fn add_outer_product_self(&mut self, a: S, x: &[S]) {
        for (j, xj) in x.iter().enumerate().take(self.cols) {
            let ay = a * *xj;
            if ay == S::zero() {
                continue;
            }
            for (i, xi) in x.iter().enumerate().take(self.rows) {
                self.data[j * self.rows + i] += ay * *xi;
            }
        }
    }

    /// `self += a * x * yᵀ`.
    pub fn add_outer_product(&mut self, a: S, x: &[S], y: &[S]) {
        for (j, yj) in y.iter().enumerate().take(self.cols) {
            let ay = a * *yj;
            if ay == S::zero() {
                continue;
            }
            for (i, xi) in x.iter().enumerate().take(self.rows) {
                self.data[j * self.rows + i] += ay * *xi;
            }
        }
    }

    /// `self += a * Jᵀ J` where `J` is `m × cols` row-major.
    pub fn add_jtj(&mut self, a: S, j: &[S], m: usize) {
        debug_assert_eq!(j.len(), m * self.cols);
        for row in 0..m {
            let slice = &j[row * self.cols..(row + 1) * self.cols];
            self.add_outer_product_self(a, slice);
        }
    }

    /// Immutable block data (column-major).
    pub fn as_slice(&self) -> &[S] {
        self.data
    }
}

/// Symmetric block-tridiagonal matrix.
///
/// `B == 0` selects a dynamic block size (created with [`Self::new_dynamic`]).
/// `BlockTridiagMat<B>` uses `f64` by default, like the C++ alias.
pub struct BlockTridiagMat<const B: usize, S: Real = f64> {
    n_blocks: usize,
    block_size: usize,
    diag: Vec<S>,
    off: Vec<S>,
}

impl<const B: usize, S: Real> Clone for BlockTridiagMat<B, S> {
    fn clone(&self) -> Self {
        Self {
            n_blocks: self.n_blocks,
            block_size: self.block_size,
            diag: self.diag.clone(),
            off: self.off.clone(),
        }
    }
}

impl<const B: usize, S: Real> core::fmt::Debug for BlockTridiagMat<B, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BlockTridiagMat")
            .field("n_blocks", &self.n_blocks)
            .field("block_size", &self.block_size)
            .finish()
    }
}

impl<const B: usize, S: Real> BlockTridiagMat<B, S> {
    /// Create a matrix with `n_blocks` diagonal and `n_blocks - 1` off-diagonal
    /// blocks (requires a compile-time block size, `B > 0`).
    pub fn new(n_blocks: usize) -> Self {
        assert!(B > 0, "use new_dynamic() when B == 0");
        Self::with_block_size(n_blocks, B)
    }

    /// Create a matrix with a runtime block size (requires `B == 0`).
    pub fn new_dynamic(n_blocks: usize, block_size: usize) -> Self {
        assert_eq!(B, 0, "new_dynamic() requires BlockTridiagMat<0>");
        Self::with_block_size(n_blocks, block_size)
    }

    /// Create a matrix with a runtime block size without a compile-time check.
    pub fn with_block_size(n_blocks: usize, block_size: usize) -> Self {
        let bs2 = block_size * block_size;
        Self {
            n_blocks,
            block_size,
            diag: vec![S::zero(); n_blocks * bs2],
            off: vec![S::zero(); n_blocks.saturating_sub(1) * bs2],
        }
    }

    /// Number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Scalar dimension.
    pub fn dims(&self) -> usize {
        self.n_blocks * self.block_size
    }

    /// Diagonal block `i`.
    pub fn diagonal(&mut self, i: usize) -> BlockMut<'_, S> {
        let bs = self.block_size;
        let start = i * bs * bs;
        BlockMut {
            data: &mut self.diag[start..start + bs * bs],
            rows: bs,
            cols: bs,
        }
    }

    /// Super-diagonal block `i` (between blocks `i` and `i + 1`).
    pub fn off_diagonal(&mut self, i: usize) -> BlockMut<'_, S> {
        let bs = self.block_size;
        let start = i * bs * bs;
        BlockMut {
            data: &mut self.off[start..start + bs * bs],
            rows: bs,
            cols: bs,
        }
    }

    /// Immutable diagonal block.
    pub fn diagonal_ref(&self, i: usize) -> &[S] {
        let bs = self.block_size;
        &self.diag[i * bs * bs..(i + 1) * bs * bs]
    }

    /// Immutable super-diagonal block.
    pub fn off_diagonal_ref(&self, i: usize) -> &[S] {
        let bs = self.block_size;
        &self.off[i * bs * bs..(i + 1) * bs * bs]
    }

    /// Zero all blocks.
    pub fn set_zero(&mut self) {
        self.diag.iter_mut().for_each(|v| *v = S::zero());
        self.off.iter_mut().for_each(|v| *v = S::zero());
    }

    /// Set the diagonal blocks to identity and the off-diagonal blocks to zero.
    pub fn set_identity(&mut self) {
        self.set_zero();
        let bs = self.block_size;
        for i in 0..self.n_blocks {
            for d in 0..bs {
                self.diag[i * bs * bs + d * bs + d] = S::one();
            }
        }
    }

    /// Multiply the diagonal blocks by `factor`.
    pub fn scale_diagonal(&mut self, factor: S) {
        self.diag.iter_mut().for_each(|v| *v *= factor);
    }

    /// Scale by `diag(s) * diag(s)` (or divide when `inverse`).
    pub fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
        let bs = self.block_size;
        for b in 0..self.n_blocks {
            let base = b * bs;
            let block = &mut self.diag[b * bs * bs..(b + 1) * bs * bs];
            for j in 0..bs {
                for i in 0..bs {
                    let factor = scale[base + i] * scale[base + j];
                    if inverse {
                        block[j * bs + i] /= factor;
                    } else {
                        block[j * bs + i] *= factor;
                    }
                }
            }
        }
        for b in 0..self.n_blocks.saturating_sub(1) {
            let base = b * bs;
            let block = &mut self.off[b * bs * bs..(b + 1) * bs * bs];
            for j in 0..bs {
                for i in 0..bs {
                    let factor = scale[base + i] * scale[base + bs + j];
                    if inverse {
                        block[j * bs + i] /= factor;
                    } else {
                        block[j * bs + i] *= factor;
                    }
                }
            }
        }
    }

    /// Zero fixed rows/columns and set their diagonal to one.
    pub fn apply_fixed_indices(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }
        let bs = self.block_size;
        let is_fixed = |i: usize| indices.binary_search(&i).is_ok();
        for b in 0..self.n_blocks {
            let base = b * bs;
            let block = &mut self.diag[b * bs * bs..(b + 1) * bs * bs];
            for j in 0..bs {
                for i in 0..bs {
                    if i == j {
                        if is_fixed(base + i) {
                            block[j * bs + i] = S::one();
                        }
                    } else if is_fixed(base + i) || is_fixed(base + j) {
                        block[j * bs + i] = S::zero();
                    }
                }
            }
        }
        for b in 0..self.n_blocks.saturating_sub(1) {
            let base = b * bs;
            let block = &mut self.off[b * bs * bs..(b + 1) * bs * bs];
            for j in 0..bs {
                for i in 0..bs {
                    if is_fixed(base + i) || is_fixed(base + bs + j) {
                        block[j * bs + i] = S::zero();
                    }
                }
            }
        }
    }

    /// Whether every coefficient is finite.
    pub fn is_all_finite(&self) -> bool {
        self.diag.iter().all(|v| v.is_finite()) && self.off.iter().all(|v| v.is_finite())
    }

    /// Coefficient `(i, j)` (dense indexing).
    pub fn get(&self, i: usize, j: usize) -> S {
        let bs = self.block_size;
        let (bi, li) = (i / bs, i % bs);
        let (bj, lj) = (j / bs, j % bs);
        if bi == bj {
            self.diag[bi * bs * bs + lj * bs + li]
        } else if bi + 1 == bj {
            self.off[bi * bs * bs + lj * bs + li]
        } else if bj + 1 == bi {
            self.off[bj * bs * bs + li * bs + lj]
        } else {
            S::zero()
        }
    }

    /// Set coefficient `(i, j)` (kept symmetric).
    pub fn set(&mut self, i: usize, j: usize, value: S) {
        let bs = self.block_size;
        let (bi, li) = (i / bs, i % bs);
        let (bj, lj) = (j / bs, j % bs);
        if bi == bj {
            self.diag[bi * bs * bs + lj * bs + li] = value;
        } else if bi + 1 == bj {
            self.off[bi * bs * bs + lj * bs + li] = value;
        } else if bj + 1 == bi {
            self.off[bj * bs * bs + li * bs + lj] = value;
        } else {
            panic!("coefficient outside the block-tridiagonal pattern");
        }
    }

    /// Sparse conversion (full symmetric tridiagonal pattern).
    pub fn to_sparse(&self) -> crate::sparse::SparseMatrix<S> {
        let bs = self.block_size;
        let n = self.dims();
        let mut triplets = Vec::with_capacity(self.n_blocks * bs * bs * 3);
        for b in 0..self.n_blocks {
            for j in 0..bs {
                for i in 0..bs {
                    let v = self.diag[b * bs * bs + j * bs + i];
                    if v != S::zero() {
                        triplets.push((b * bs + i, b * bs + j, v));
                    }
                }
            }
            if b + 1 < self.n_blocks {
                for j in 0..bs {
                    for i in 0..bs {
                        let v = self.off[b * bs * bs + j * bs + i];
                        if v != S::zero() {
                            triplets.push((b * bs + i, (b + 1) * bs + j, v));
                            triplets.push(((b + 1) * bs + j, b * bs + i, v));
                        }
                    }
                }
            }
        }
        crate::sparse::SparseMatrix::from_triplets(n, n, &triplets)
    }

    /// Block-sparse conversion (one block type of size `B`).
    pub fn to_block_sparse(&self) -> crate::block_sparse::BlockSparseMatX<S> {
        crate::block_sparse::BlockSparseMatX::from_parts(
            self.to_sparse(),
            vec![self.block_size],
            vec![self.n_blocks],
        )
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

    /// Matrix-vector product.
    pub fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
        let bs = self.block_size;
        for b in 0..self.n_blocks {
            let base = b * bs;
            for i in 0..bs {
                let mut acc = S::zero();
                for j in 0..bs {
                    acc += self.diag[b * bs * bs + j * bs + i] * x[base + j];
                    if b + 1 < self.n_blocks {
                        acc += self.off[b * bs * bs + j * bs + i] * x[base + bs + j];
                    }
                    if b > 0 {
                        acc += self.off[(b - 1) * bs * bs + i * bs + j] * x[base - bs + j];
                    }
                }
                out[base + i] = acc;
            }
        }
    }
}

impl<const B: usize, S: Real> BlockTridiagMat<B, S> {
    /// Cast to another precision (used for covariance export).
    pub fn cast<T: Real>(&self) -> BlockTridiagMat<B, T> {
        let mut out = BlockTridiagMat::<B, T>::with_block_size(self.n_blocks, self.block_size);
        for (o, v) in out.diag.iter_mut().zip(&self.diag) {
            *o = T::from_f64(v.to_f64().expect("cast")).expect("cast");
        }
        for (o, v) in out.off.iter_mut().zip(&self.off) {
            *o = T::from_f64(v.to_f64().expect("cast")).expect("cast");
        }
        out
    }
}

// --- Block operations used by the LDLT -------------------------------------

fn block_inverse<S: Real>(data: &[S], bs: usize) -> Option<Vec<S>> {
    let mut m = zeros_matrix::<S, Dynamic>(bs, bs);
    for j in 0..bs {
        for i in 0..bs {
            m.set_elem(i, j, data[j * bs + i]);
        }
    }
    let inv = inverse::<S, Dynamic>(&m)?;
    let mut out = vec![S::zero(); bs * bs];
    for j in 0..bs {
        for i in 0..bs {
            out[j * bs + i] = inv.elem(i, j);
        }
    }
    Some(out)
}

fn block_mul<S: Real>(a: &[S], b: &[S], bs: usize) -> Vec<S> {
    let mut out = vec![S::zero(); bs * bs];
    for j in 0..bs {
        for i in 0..bs {
            let mut acc = S::zero();
            for k in 0..bs {
                acc += a[k * bs + i] * b[j * bs + k];
            }
            out[j * bs + i] = acc;
        }
    }
    out
}

fn block_mul_transpose_left<S: Real>(a: &[S], b: &[S], bs: usize) -> Vec<S> {
    // aᵀ * b
    let mut out = vec![S::zero(); bs * bs];
    for j in 0..bs {
        for i in 0..bs {
            let mut acc = S::zero();
            for k in 0..bs {
                acc += a[i * bs + k] * b[j * bs + k];
            }
            out[j * bs + i] = acc;
        }
    }
    out
}

/// Block-tridiagonal LDLᵀ factorization (block Thomas algorithm).
#[derive(Clone, Debug)]
pub struct BlockTridiagLdlt<S> {
    n_blocks: usize,
    block_size: usize,
    inv_delta: Vec<S>,
    c: Vec<S>,
    computed: bool,
}

impl<S: Real> Default for BlockTridiagLdlt<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Real> BlockTridiagLdlt<S> {
    /// Create an empty factorization.
    pub fn new() -> Self {
        Self {
            n_blocks: 0,
            block_size: 0,
            inv_delta: Vec::new(),
            c: Vec::new(),
            computed: false,
        }
    }

    /// Compute the factorization of `matrix`.
    pub fn compute<const B: usize>(&mut self, matrix: &BlockTridiagMat<B, S>) -> bool {
        let n = matrix.num_blocks();
        let bs = matrix.block_size();
        let bs2 = bs * bs;
        self.n_blocks = n;
        self.block_size = bs;
        self.inv_delta = vec![S::zero(); n * bs2];
        self.c = vec![S::zero(); n.saturating_sub(1) * bs2];

        // Δ₀ = D₀.
        let mut delta = matrix.diagonal_ref(0).to_vec();
        match block_inverse(&delta, bs) {
            Some(inv) => self.inv_delta[..bs2].copy_from_slice(&inv),
            None => return false,
        }
        for i in 0..n.saturating_sub(1) {
            // C_i = Δ_i^{-1} U_i.
            let inv_i = &self.inv_delta[i * bs2..(i + 1) * bs2];
            let u = matrix.off_diagonal_ref(i);
            let c_i = block_mul(inv_i, u, bs);
            if !c_i.iter().all(|v| v.is_finite()) {
                return false;
            }
            self.c[i * bs2..(i + 1) * bs2].copy_from_slice(&c_i);

            // Δ_{i+1} = D_{i+1} - U_iᵀ C_i.
            let utc = block_mul_transpose_left(u, &c_i, bs);
            let next = matrix.diagonal_ref(i + 1);
            delta = (0..bs2).map(|k| next[k] - utc[k]).collect::<Vec<_>>();
            match block_inverse(&delta, bs) {
                Some(inv) => self.inv_delta[(i + 1) * bs2..(i + 2) * bs2].copy_from_slice(&inv),
                None => return false,
            }
        }
        self.computed = true;
        true
    }

    /// Solve `A x = rhs` into `out`.
    pub fn solve_into(&self, rhs: &[S], out: &mut [S]) -> bool {
        if !self.computed {
            return false;
        }
        let (n, bs) = (self.n_blocks, self.block_size);
        let mut z = rhs.to_vec();
        // Forward: z_i = b_i - C_{i-1}ᵀ z_{i-1}.
        for i in 1..n {
            let c_prev = &self.c[(i - 1) * bs * bs..i * bs * bs];
            for j in 0..bs {
                let mut acc = S::zero();
                for k in 0..bs {
                    acc += c_prev[j * bs + k] * z[(i - 1) * bs + k];
                }
                z[i * bs + j] -= acc;
            }
        }
        // Diagonal: y_i = Δ_i^{-1} z_i.
        let mut y = vec![S::zero(); n * bs];
        for i in 0..n {
            let inv = &self.inv_delta[i * bs * bs..(i + 1) * bs * bs];
            for j in 0..bs {
                let mut acc = S::zero();
                for k in 0..bs {
                    acc += inv[j * bs + k] * z[i * bs + k];
                }
                y[i * bs + j] = acc;
            }
        }
        // Backward: x_i = y_i - C_i x_{i+1}.
        let mut x = y;
        for i in (0..n.saturating_sub(1)).rev() {
            let c_i = &self.c[i * bs * bs..(i + 1) * bs * bs];
            for j in 0..bs {
                let mut acc = S::zero();
                for k in 0..bs {
                    acc += c_i[k * bs + j] * x[(i + 1) * bs + k];
                }
                x[i * bs + j] -= acc;
            }
        }
        out.copy_from_slice(&x);
        x.iter().all(|v| v.is_finite())
    }

    /// Block-tridiagonal inverse (Takahashi recurrence).
    pub fn inverse_block_tridiag<const B: usize>(&self) -> Option<BlockTridiagMat<B, S>> {
        if !self.computed {
            return None;
        }
        let (n, bs) = (self.n_blocks, self.block_size);
        let bs2 = bs * bs;
        let mut diag = vec![S::zero(); n * bs2];
        let mut off = vec![S::zero(); n.saturating_sub(1) * bs2];

        // Last diagonal: Δ_{n-1}^{-1}.
        diag[(n - 1) * bs2..n * bs2].copy_from_slice(&self.inv_delta[(n - 1) * bs2..n * bs2]);
        for i in (0..n.saturating_sub(1)).rev() {
            let c_i = &self.c[i * bs2..(i + 1) * bs2];
            // Σ_{i,i+1} = -C_i Σ_{i+1,i+1}.
            let next = &diag[(i + 1) * bs2..(i + 2) * bs2];
            let s_next = block_mul(c_i, next, bs);
            let mut off_i = vec![S::zero(); bs2];
            for k in 0..bs2 {
                off_i[k] = -s_next[k];
            }
            off[i * bs2..(i + 1) * bs2].copy_from_slice(&off_i);
            // Σ_{i,i} = Δ_i^{-1} - Σ_{i,i+1} C_iᵀ.
            let sc = block_mul(&off_i, &transpose(c_i, bs), bs);
            let inv_i = &self.inv_delta[i * bs2..(i + 1) * bs2];
            for k in 0..bs2 {
                diag[i * bs2 + k] = inv_i[k] - sc[k];
            }
        }

        let mut matrix = BlockTridiagMat::<B, S>::with_block_size(n, bs);
        matrix.diag.copy_from_slice(&diag);
        matrix.off.copy_from_slice(&off);
        Some(matrix)
    }

    /// Whether the factorization is available.
    pub fn is_computed(&self) -> bool {
        self.computed
    }
}

fn transpose<S: Real>(a: &[S], bs: usize) -> Vec<S> {
    let mut out = vec![S::zero(); bs * bs];
    for j in 0..bs {
        for i in 0..bs {
            out[j * bs + i] = a[i * bs + j];
        }
    }
    out
}

// --- HessianStorage implementation -----------------------------------------

impl<const B: usize, S: Real> HessianStorage<S> for BlockTridiagMat<B, S> {
    fn zeros(dims: usize) -> Self {
        // Only meaningful for fixed block sizes; dynamic users provide a
        // prototype through `Optimizer::init_with`.
        if B > 0 {
            assert_eq!(
                dims % B,
                0,
                "dimension must be a multiple of the block size"
            );
            Self::with_block_size(dims / B, B)
        } else {
            Self::with_block_size(dims, 1)
        }
    }

    fn zeros_like(prototype: &Self, dims: usize) -> Self {
        let bs = prototype.block_size;
        assert_eq!(
            dims % bs,
            0,
            "dimension must be a multiple of the block size"
        );
        Self::with_block_size(dims / bs, bs)
    }

    fn dims(&self) -> usize {
        self.n_blocks * self.block_size
    }

    fn set_zero(&mut self) {
        self.set_zero();
    }

    fn diagonal_into(&self, out: &mut [S]) {
        for (i, o) in out.iter_mut().enumerate().take(self.dims()) {
            *o = self.get(i, i);
        }
    }

    fn scale_diagonal(&mut self, factor: S) {
        self.scale_diagonal(factor);
    }

    fn scale_by_diagonal(&mut self, scale: &[S], inverse: bool) {
        self.scale_by_diagonal(scale, inverse);
    }

    fn apply_fixed_indices(&mut self, indices: &[usize]) {
        self.apply_fixed_indices(indices);
    }

    fn is_all_finite(&self) -> bool {
        self.is_all_finite()
    }

    fn mul_vec_into(&self, x: &[S], out: &mut [S]) {
        self.mul_vec_into(x, out);
    }

    type SolverCache = ();

    fn solve_into(
        &self,
        _options: &LinearSolverOptions,
        rhs: &[S],
        num_reduced: usize,
        out: &mut [S],
        _cache: &mut (),
    ) -> bool {
        if num_reduced > 0 {
            return false;
        }
        let mut ldlt = BlockTridiagLdlt::new();
        if !ldlt.compute(self) {
            return false;
        }
        ldlt.solve_into(rhs, out)
    }

    fn to_dense(&self) -> Matrix<S, Dynamic> {
        BlockTridiagMat::to_dense(self)
    }

    fn to_final_covariance(
        &self,
        options: &CovarianceOptions,
        _cost: Cost,
        _fixed: &[usize],
        _num_reduced: usize,
    ) -> Option<FinalCovariance> {
        let mut ldlt = BlockTridiagLdlt::new();
        if !ldlt.compute(self) {
            return None;
        }
        match options.pattern {
            CovariancePattern::FullDense => {
                let inv = ldlt.inverse_block_tridiag::<0>()?;
                let dense = inv.to_dense();
                let mut out = zeros_matrix::<f32, Dynamic>(dense.nrows(), dense.ncols());
                for j in 0..dense.ncols() {
                    for i in 0..dense.nrows() {
                        out.set_elem(i, j, dense.elem(i, j).to_f32()?);
                    }
                }
                Some(FinalCovariance::Dense(out))
            }
            CovariancePattern::BlockDiagonal => {
                let mut inv = ldlt.inverse_block_tridiag::<0>()?;
                inv.off.iter_mut().for_each(|v| *v = S::zero());
                Some(FinalCovariance::BlockTridiag(inv.cast::<f32>()))
            }
            _ => {
                let inv = ldlt.inverse_block_tridiag::<0>()?;
                Some(FinalCovariance::BlockTridiag(inv.cast::<f32>()))
            }
        }
    }
}

/// Dynamic block-size block-tridiagonal matrix (C++ `BlockTridiagMatX`).
pub type BlockTridiagMatX<S> = BlockTridiagMat<0, S>;
