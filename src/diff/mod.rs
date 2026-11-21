// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Differentiation utilities.
//!
//! The C++ library ships its own `Jet` dual number. In Rust we use
//! [`num_dual`](https://docs.rs/num-dual), whose `DualVec<T, D>` is the direct
//! equivalent of tinyopt's `Jet<T, N>`: a value plus an `N`-dimensional
//! derivative vector, backed by nalgebra storage and usable as a scalar inside
//! nalgebra vectors and matrices.

#[cfg(feature = "autodiff")]
pub mod autodiff;

#[cfg(feature = "autodiff")]
pub use autodiff::{
    CostAndGradient, CostFn, ResidualsAndJacobian, ResidualsFn, cost_and_gradient,
    residuals_and_jacobian,
};

#[cfg(feature = "autodiff")]
pub use num_dual;
#[cfg(feature = "autodiff")]
pub use num_dual::{Dual, DualNum, DualVec};
