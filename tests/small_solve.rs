// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Closed-form small-system solve and math utilities (C++ `SolveSmallDenseSystem`).

use approx::assert_relative_eq;
use tinyopt::backend::{
    DenseMatrix, DenseVector, Dynamic, LinearSolverMethod, Matrix, zeros_matrix, zeros_vector,
};
use tinyopt::math::{float_epsilon2, solve_small_dense_system, sqrt_int};

fn spd(n: usize) -> Matrix<f64, Dynamic> {
    let mut a = zeros_matrix::<f64, Dynamic>(n, n);
    for i in 0..n {
        a.set_elem(i, i, 4.0 + i as f64);
        if i + 1 < n {
            a.set_elem(i, i + 1, 1.0);
            a.set_elem(i + 1, i, 1.0);
        }
    }
    a
}

#[test]
fn closed_form_matches_the_factorization() {
    for n in 1..=4 {
        let a = spd(n);
        let mut b = zeros_vector::<f64, Dynamic>(n);
        for i in 0..n {
            b.set_at(i, 1.0 + i as f64);
        }
        let x = solve_small_dense_system::<f64, Dynamic>(&a, &b).expect("closed form");
        let reference = tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &a, &b)
            .expect("factored");
        for i in 0..n {
            assert_relative_eq!(x.at(i), reference.at(i), epsilon = 1e-10);
        }
    }
}

#[test]
fn small_system_fast_path_matches_dense_expectations() {
    // `backend::solve` uses the closed form for n <= 4; the result must be the
    // exact solution of the linear system.
    let a = spd(3);
    let expected = [1.0, -2.0, 0.5];
    let mut b = zeros_vector::<f64, Dynamic>(3);
    for i in 0..3 {
        let row: f64 = (0..3).map(|j| a.elem(i, j) * expected[j]).sum();
        b.set_at(i, row);
    }
    let x = tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &a, &b).unwrap();
    for (i, e) in expected.iter().enumerate() {
        assert_relative_eq!(x.at(i), *e, epsilon = 1e-10);
    }
}

#[test]
fn singular_small_system_falls_back() {
    let mut a = zeros_matrix::<f64, Dynamic>(2, 2);
    a.set_elem(0, 0, 1.0);
    a.set_elem(0, 1, 1.0);
    a.set_elem(1, 0, 1.0);
    a.set_elem(1, 1, 1.0);
    let mut b = zeros_vector::<f64, Dynamic>(2);
    b.set_at(0, 1.0);
    b.set_at(1, 1.0);
    // The closed form rejects it (singular).
    assert!(solve_small_dense_system::<f64, Dynamic>(&a, &b).is_none());
    // The SVD path still returns the minimum-norm solution.
    let x = tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Svd, &a, &b).unwrap();
    assert_relative_eq!(x.at(0), 0.5, epsilon = 1e-8);
    assert_relative_eq!(x.at(1), 0.5, epsilon = 1e-8);
}

#[test]
fn math_utilities_match_cpp() {
    assert_eq!(sqrt_int(0), 0);
    assert_eq!(sqrt_int(1), 1);
    assert_eq!(sqrt_int(2), 1);
    assert_eq!(sqrt_int(15), 3);
    assert_eq!(sqrt_int(16), 4);
    assert_eq!(sqrt_int(17), 4);
    assert_eq!(sqrt_int(1_000_000), 1000);
    assert_relative_eq!(float_epsilon2::<f64>(), 1e-14, epsilon = 0.0);
    assert_relative_eq!(float_epsilon2::<f32>(), 1e-8, epsilon = 0.0);
}
