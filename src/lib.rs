// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! # tinyopt-rs
//!
//! A lightweight, Rust native optimization library for unconstrained
//! optimization and non-linear least squares (NLLS), focused on dense systems.
//!
//! This crate is the Rust counterpart of the C++ [tinyopt](https://github.com/julienmichot/tinyopt)
//! library (version parity: **0.8.0**). It follows the same *accumulation pattern*:
//! users accumulate the gradient `Jᵀr` and Hessian approximation `JᵀJ` directly into
//! the linear system, avoiding large residual/Jacobian buffers.
//!
//! ## Backends
//!
//! Exactly one linear-algebra backend must be enabled:
//!
//! - `nalgebra` (default): static (`SVector`/`SMatrix`) and dynamic (`DVector`/`DMatrix`) sizes.
//! - `faer`: dynamic sizes (`Col`/`Mat`); static-sized problems use the dynamic types.
//!
//! ## Differentiation
//!
//! - `autodiff` (default): forward-mode automatic differentiation using
//!   [`num-dual`](https://docs.rs/num-dual) `DualVec`, the Rust equivalent of
//!   tinyopt's `Jet` (Ceres-style dual numbers).
//! - `numdiff`: central finite differences.
//!
//! ## Example
//!
//! ```no_run
//! use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
//! use tinyopt_rs::{Cost, Options, Static, optimize};
//!
//! type Vec2 = Vector<f64, Static<2>>;
//! type Mat2 = Matrix<f64, Static<2>>;
//!
//! // Minimize (x0² - 2)² + (x0*x1 - 3)² with a manual accumulation function.
//! let acc = |x: &[f64], grad: Option<&mut Vec2>, h: Option<&mut Mat2>| -> Cost {
//!     let r0 = x[0] * x[0] - 2.0;
//!     let r1 = x[0] * x[1] - 3.0;
//!     if let Some(g) = grad {
//!         g.set_at(0, 2.0 * x[0] * r0 + x[1] * r1);
//!         g.set_at(1, x[0] * r1);
//!     }
//!     if let Some(h) = h {
//!         let h00 = 4.0 * x[0] * x[0] + x[1] * x[1];
//!         let h01 = 2.0 * x[0] * x[1] + x[0] * x[1];
//!         h.set_elem(0, 0, h00);
//!         h.set_elem(0, 1, h01);
//!         h.set_elem(1, 0, h01);
//!         h.set_elem(1, 1, x[0] * x[0]);
//!     }
//!     Cost::with_num_residuals(r0 * r0 + r1 * r1, 2)
//! };
//!
//! let mut x = <Vec2 as tinyopt_rs::backend::DenseVector<f64>>::zeros(2);
//! x.set_at(0, 1.0);
//! x.set_at(1, 1.0);
//! let summary = optimize(&mut x, acc, &Options::default());
//! assert!(summary.succeeded());
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(feature = "nalgebra", feature = "faer"))]
compile_error!(
    "enable exactly one backend: `nalgebra` (default) or `faer` (with --no-default-features)"
);

#[cfg(not(any(feature = "nalgebra", feature = "faer")))]
compile_error!("enable a backend: `nalgebra` (default) or `faer`");

pub mod backend;
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

mod optimize;

pub use backend::{Dim, Dynamic, Static};
pub use cost::Cost;
pub use covariance::{CovarianceOptions, CovariancePattern, FinalCovariance};
pub use math::inverse;
#[cfg(feature = "autodiff")]
pub use optimize::optimize_robust_residuals;
pub use optimize::{
    DynGradient, DynHessian, optimize, optimize_default, optimize_sparse, optimize2,
    optimize2_default, optimize3, optimize3_default,
};
#[cfg(feature = "autodiff")]
pub use optimize::{optimize_cost, optimize_residuals, optimize_residuals_default};
pub use optimizers::AccumulationFn;
pub use options::{LineSearchMethod, Options, Solver};
pub use params::{Params, ParamsWrapper};
pub use params_pack::{PackValues, ParamsPack};
pub use real::Real;
pub use sparse::{SimplicialLdl, SparseLinearSolvers, SparseMat, SparseMatf, SparseMatrix};
pub use stop_reasons::StopReason;
pub use summary::{History, Summary};

/// Library version, kept in sync with the C++ tinyopt release.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
