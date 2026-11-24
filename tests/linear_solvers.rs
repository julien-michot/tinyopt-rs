// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/linear_solvers.cpp and linear_solver_structs.cpp.

use approx::assert_relative_eq;
use tinyopt::backend::{
    DenseMatrix, DenseVector, Dynamic, LinearSolverMethod, Matrix, Static, Vector, zeros_matrix,
    zeros_vector,
};

type V3 = Vector<f64, Static<3>>;
type M3 = Matrix<f64, Static<3>>;

fn fixed_spd() -> M3 {
    let mut m = zeros_matrix::<f64, Static<3>>(3, 3);
    m.set_elem(0, 0, 4.0);
    m.set_elem(1, 1, 6.0);
    m.set_elem(2, 2, 5.0);
    m.set_elem(0, 1, 1.0);
    m.set_elem(1, 0, 1.0);
    m.set_elem(1, 2, 2.0);
    m.set_elem(2, 1, 2.0);
    m
}

fn dynamic_spd(n: usize) -> Matrix<f64, Dynamic> {
    let mut m = zeros_matrix::<f64, Dynamic>(n, n);
    for i in 0..n {
        m.set_elem(i, i, 4.0);
        if i + 1 < n {
            m.set_elem(i, i + 1, 1.0);
            m.set_elem(i + 1, i, 1.0);
        }
    }
    m
}

#[test]
fn dense_methods_solve_fixed_size_system() {
    let a = fixed_spd();
    let expected = [1.0, -2.0, 3.0];
    let mut b = <V3 as DenseVector<f64>>::zeros(3);
    for i in 0..3 {
        let row: f64 = (0..3).map(|j| a.elem(i, j) * expected[j]).sum();
        b.set_at(i, row);
    }
    for method in [
        LinearSolverMethod::Ldlt,
        LinearSolverMethod::Llt,
        LinearSolverMethod::Lu,
        LinearSolverMethod::Qr,
        LinearSolverMethod::Svd,
    ] {
        let x = tinyopt::backend::solve::<f64, Static<3>>(method, &a, &b)
            .unwrap_or_else(|| panic!("solve failed for {method:?}"));
        for i in 0..3 {
            assert_relative_eq!(x.at(i), expected[i], epsilon = 1e-9);
        }
    }
}

#[test]
fn dense_methods_solve_dynamic_system() {
    let a = dynamic_spd(6);
    let expected: Vec<f64> = (0..6).map(|i| (i as f64) * 0.5 - 1.0).collect();
    let mut b = zeros_vector::<f64, Dynamic>(6);
    for i in 0..6 {
        let row: f64 = (0..6).map(|j| a.elem(i, j) * expected[j]).sum();
        b.set_at(i, row);
    }
    for method in [
        LinearSolverMethod::Ldlt,
        LinearSolverMethod::Llt,
        LinearSolverMethod::Lu,
        LinearSolverMethod::Qr,
        LinearSolverMethod::Svd,
    ] {
        let x = tinyopt::backend::solve::<f64, Dynamic>(method, &a, &b)
            .unwrap_or_else(|| panic!("solve failed for {method:?}"));
        for i in 0..6 {
            assert_relative_eq!(x.at(i), expected[i], epsilon = 1e-8);
        }
    }
}

#[test]
fn svd_solves_rank_deficient_system() {
    // Rank-one 2x2: SVD returns the minimum-norm least-squares solution.
    let mut a = zeros_matrix::<f64, Dynamic>(2, 2);
    a.set_elem(0, 0, 1.0);
    a.set_elem(1, 0, 1.0);
    let mut b = zeros_vector::<f64, Dynamic>(2);
    b.set_at(0, 1.0);
    b.set_at(1, 1.0);
    let x = tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Svd, &a, &b)
        .expect("svd solve");
    // a x = [x0, x0] = [1, 1] -> x0 = 1, x1 = 0 (minimum norm).
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(1), 0.0, epsilon = 1e-8);
}

#[test]
fn linear_solver_options_defaults_match_cpp() {
    let options = tinyopt::options::LinearSolverOptions::default();
    assert_eq!(options.method, LinearSolverMethod::Ldlt);
    assert!(options.allow_schur);
    assert_eq!(options.ordering, tinyopt::Ordering::Amd);
}
