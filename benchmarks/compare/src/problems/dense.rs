// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Dense math and prior problems, ported from `benchmarks/common/dense_problems.h`.

/// Initial parameters of the dense math problem.
pub fn math_initial(dims: usize) -> Vec<f64> {
    if dims == 1 {
        vec![1.0]
    } else {
        vec![1.5; dims]
    }
}

/// Residual vector of the dense math problem (`dims` residuals).
pub fn math_residuals(x: &[f64]) -> Vec<f64> {
    match x.len() {
        1 => vec![x[0] * x[0] - 2.0],
        2 => vec![x[0] * x[0] + x[1] - 2.0, x[0] + x[1] * x[1] - 2.0],
        _ => vec![
            x[0] * x[0] + x[1] + x[2] - 3.0,
            x[0] + x[1] * x[1] + x[2] - 3.0,
            x[0] + x[1] + x[2] * x[2] - 3.0,
        ],
    }
}

/// Row-major Jacobian of [`math_residuals`].
pub fn math_jacobian(x: &[f64]) -> Vec<f64> {
    match x.len() {
        1 => vec![2.0 * x[0]],
        2 => vec![2.0 * x[0], 1.0, 1.0, 2.0 * x[1]],
        _ => vec![
            2.0 * x[0],
            1.0,
            1.0, //
            1.0,
            2.0 * x[1],
            1.0, //
            1.0,
            1.0,
            2.0 * x[2],
        ],
    }
}

/// Prior initial vector.
pub fn prior_initial(dims: usize) -> Vec<f64> {
    (0..dims).map(|i| 2.0 + 0.1 * i as f64).collect()
}

/// Prior target vector.
pub fn prior_target(dims: usize) -> Vec<f64> {
    (0..dims).map(|i| 0.25 * (i + 1) as f64).collect()
}
