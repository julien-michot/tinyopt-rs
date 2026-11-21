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
//! ## Status
//!
//! M0 scaffolding: backend abstraction, parameter traits, differentiation
//! entry points.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_debug_implementations)]

#[cfg(all(feature = "nalgebra", feature = "faer"))]
compile_error!(
    "enable exactly one backend: `nalgebra` (default) or `faer` (with --no-default-features)"
);

#[cfg(not(any(feature = "nalgebra", feature = "faer")))]
compile_error!("enable a backend: `nalgebra` (default) or `faer`");

pub mod backend;
pub mod diff;
pub mod params;
pub mod real;

pub use backend::{Dim, Dynamic, Static};
pub use params::Params;
pub use real::Real;

/// Library version, kept in sync with the C++ tinyopt release.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
