// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Iterative optimizers.

pub mod bfgs;
pub mod cg;
pub mod dl;
pub mod gd;
pub mod gn;
pub mod lbfgs;
pub mod lm;
#[allow(clippy::module_inception)]
pub mod optimizer;

pub use optimizer::AccumulationFn;
