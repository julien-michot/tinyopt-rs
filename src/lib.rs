// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

// The README examples use the autodiff API, so the full documentation is only
// included when that feature is available (docs.rs builds with defaults).
#![cfg_attr(feature = "autodiff", doc = include_str!("../README.md"))]
#![cfg_attr(
    not(feature = "autodiff"),
    doc = "Rust native optimization and non-linear least squares.\n\nThis build has the `autodiff` feature disabled; see the README for examples."
)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(feature = "nalgebra", feature = "faer"))]
compile_error!(
    "enable exactly one backend: `nalgebra` (default) or `faer` (with --no-default-features)"
);

#[cfg(not(any(feature = "nalgebra", feature = "faer")))]
compile_error!("enable a backend: `nalgebra` (default) or `faer`");

pub mod backend;
pub mod block_sparse;
pub mod block_tridiag;
pub mod cost;
pub mod covariance;
pub mod diff;
pub mod distances;
pub mod hessian;
pub mod losses;
pub mod math;
pub mod optimizers;
pub mod options;
pub mod params;
pub mod params_pack;
pub mod real;
pub mod sparse;
pub mod stop_reasons;
pub mod summary;
pub mod time;

mod optimize;

pub use backend::{Dim, Dynamic, Static};
pub use block_sparse::{BlockSparseLdlt, BlockSparseMat, BlockSparseMatX, BlockTypes, HasPair};
pub use block_tridiag::{BlockMut, BlockMutN, BlockTridiagLdlt, BlockTridiagMat, BlockTridiagMatX};
pub use cost::Cost;
pub use covariance::{CovarianceOptions, CovariancePattern, FinalCovariance};
pub use math::inverse;
#[cfg(feature = "autodiff")]
pub use optimize::optimize_robust_residuals;
pub use optimize::{
    DynGradient, DynHessian, optimize, optimize_block_sparse, optimize_default, optimize_sparse,
    optimize_tridiag, optimize2, optimize2_default, optimize3, optimize3_default,
};
#[cfg(feature = "autodiff")]
pub use optimize::{optimize_cost, optimize_residuals, optimize_residuals_default};
pub use optimizers::AccumulationFn;
pub use options::{LineSearchMethod, Options, Ordering, Solver};
pub use params::{Bounds, Params, ParamsWrapper};
pub use params_pack::{PackValues, ParamsPack};
#[cfg(feature = "rayon")]
pub use rayon;
pub use real::Real;
pub use sparse::{SimplicialLdl, SparseLinearSolvers, SparseMat, SparseMatf, SparseMatrix};
pub use stop_reasons::StopReason;
pub use summary::{History, Summary};

/// Library version, kept in sync with the C++ tinyopt release.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
