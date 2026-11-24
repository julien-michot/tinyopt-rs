// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Utility parity tests (M8): timing helpers, robust inverse, sparse inverse.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, Dynamic, zeros_matrix};
use tinyopt::sparse::SparseMatrix;

#[test]
fn timing_helpers_measure_elapsed_time() {
    let t0 = tinyopt::time::tic();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let elapsed = tinyopt::time::toc_ms(t0);
    assert!(elapsed >= 1.0, "elapsed = {elapsed} ms");
    let t1 = tinyopt::time::tic();
    let dt = tinyopt::time::dt_ms(t0, t1);
    assert!(dt >= 0.0);
}

#[test]
fn robust_inverse_matches_inverse_for_spd() {
    let mut m = zeros_matrix::<f64, Dynamic>(3, 3);
    m.set_elem(0, 0, 4.0);
    m.set_elem(1, 1, 5.0);
    m.set_elem(2, 2, 6.0);
    m.set_elem(0, 1, 1.0);
    m.set_elem(1, 0, 1.0);
    let inv = tinyopt::math::inverse::<f64, Dynamic>(&m).expect("inverse");
    let robust = tinyopt::math::robust_inverse::<f64, Dynamic>(&m).expect("robust inverse");
    for j in 0..3 {
        for i in 0..3 {
            assert_relative_eq!(robust.elem(i, j), inv.elem(i, j), epsilon = 1e-10);
        }
    }
}

#[test]
fn robust_inverse_handles_singular_matrices() {
    // Rank-one 2x2: [[1, 1], [1, 1]] has a pseudo-inverse [[0.25, 0.25], [0.25, 0.25]].
    let mut m = zeros_matrix::<f64, Dynamic>(2, 2);
    m.set_elem(0, 0, 1.0);
    m.set_elem(0, 1, 1.0);
    m.set_elem(1, 0, 1.0);
    m.set_elem(1, 1, 1.0);
    let inv = tinyopt::math::robust_inverse::<f64, Dynamic>(&m).expect("pseudo-inverse");
    for j in 0..2 {
        for i in 0..2 {
            assert_relative_eq!(inv.elem(i, j), 0.25, epsilon = 1e-8);
        }
    }
}

#[test]
fn sparse_inverse_matches_dense() {
    let triplets = [
        (0, 0, 4.0),
        (1, 1, 5.0),
        (2, 2, 6.0),
        (0, 1, 1.0),
        (1, 0, 1.0),
    ];
    let a = SparseMatrix::<f64>::from_triplets(3, 3, &triplets);
    let inv = a.inverse().expect("sparse inverse");
    let dense = a.to_dense();
    let dense_inv = tinyopt::math::inverse::<f64, Dynamic>(&dense).expect("dense inverse");
    for j in 0..3 {
        for i in 0..3 {
            assert_relative_eq!(inv.coeff(i, j), dense_inv.elem(i, j), epsilon = 1e-10);
        }
    }
}
