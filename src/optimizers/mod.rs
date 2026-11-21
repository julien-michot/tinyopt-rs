// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Iterative optimizers.

pub mod gn;
pub mod lm;
#[allow(clippy::module_inception)]
pub mod optimizer;

pub use optimizer::AccumulationFn;
