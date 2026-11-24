// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Sparse matrices and solvers (dense-block/sparse parity with the C++ library).

mod amd;
mod ic;
mod ldlt;
mod matrix;
mod solvers;
#[cfg(feature = "suitesparse")]
mod suitesparse;

pub use amd::minimum_degree;
pub use ic::IncompleteCholesky;
pub use ldlt::SimplicialLdl;
pub use matrix::SparseMatrix;
pub use solvers::{SparseLinearSolvers, SparseSolverCache};
#[cfg(feature = "suitesparse")]
pub use suitesparse::CholmodSolver;

/// Double-precision sparse matrix (C++ `SparseMat`).
pub type SparseMat = SparseMatrix<f64>;
/// Single-precision sparse matrix (C++ `SparseMatf`).
pub type SparseMatf = SparseMatrix<f32>;
