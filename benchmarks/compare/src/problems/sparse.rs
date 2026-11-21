// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Sparse chain problem, ported from `benchmarks/common/sparse_problem.h`.

/// Target of parameter `index`.
pub fn target(index: usize) -> f64 {
    0.1 * (index % 17) as f64 - 0.8
}

/// Initial value of parameter `index`.
pub fn initial(index: usize) -> f64 {
    1.5 - 0.03 * (index % 19) as f64
}

/// Target of the pairwise difference `x[index] - x[index - 1]`.
pub fn difference_target(index: usize) -> f64 {
    target(index) - target(index - 1)
}

/// Initial vector.
pub fn initial_vector(dims: usize) -> Vec<f64> {
    (0..dims).map(initial).collect()
}
