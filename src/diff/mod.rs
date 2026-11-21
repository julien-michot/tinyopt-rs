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
pub mod gradient_check;
pub mod num_diff;

#[cfg(feature = "autodiff")]
pub use autodiff::{
    AdAccumulator, CostAndGradient, CostFn, ResidualsAndJacobian, ResidualsFn, ad_accumulation,
    cost_and_gradient, residuals_and_jacobian,
};
pub use gradient_check::{check_gradient, check_residuals_gradient, default_eps};
pub use num_diff::{
    Method, NumDiffAccumulator, NumDiffCostAccumulator, create_num_diff_cost_func,
    create_num_diff_func, estimate_num_jac, num_eval,
};

#[cfg(feature = "autodiff")]
pub use num_dual;
#[cfg(feature = "autodiff")]
pub use num_dual::{Dual, DualNum, DualVec};
