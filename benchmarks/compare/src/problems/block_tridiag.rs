// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Fixed-size block-tridiagonal pose chain (6x6 blocks).
//!
//! Each pose has a unary prior plus a relative constraint to the next pose, so
//! the Hessian is block-tridiagonal. The problem is linear and noise-free, which
//! makes it a fair linear-solver comparison across backends.

/// Scalar size of one pose block.
pub const BLOCK: usize = 6;

/// Target value of pose `index`, coordinate `k`.
pub fn target(index: usize, k: usize) -> f64 {
    0.05 * index as f64 + 0.01 * k as f64
}

/// Coupling target between poses `index` and `index + 1`.
pub fn delta(index: usize, k: usize) -> f64 {
    target(index + 1, k) - target(index, k)
}
