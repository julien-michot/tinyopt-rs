// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Sparse matrix and solver tests (M6a).

use approx::assert_relative_eq;
use tinyopt_rs::Options;
use tinyopt_rs::backend::{
    DenseMatrix, DenseVector, Dynamic, LinearSolverMethod, Matrix, Vector, zeros_vector,
};
use tinyopt_rs::sparse::{SimplicialLdl, SparseLinearSolvers, SparseMatrix};

fn dense_solve(a: &SparseMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let m: Matrix<f64, Dynamic> = a.to_dense();
    let mut rhs = zeros_vector::<f64, Dynamic>(n);
    for (i, v) in b.iter().enumerate() {
        rhs.set_at(i, *v);
    }
    let x = tinyopt_rs::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &m, &rhs)
        .expect("dense solve");
    (0..n).map(|i| x.at(i)).collect()
}

fn chain_spd(n: usize) -> SparseMatrix<f64> {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 4.0));
        if i + 1 < n {
            triplets.push((i, i + 1, -1.0));
            triplets.push((i + 1, i, -1.0));
        }
    }
    SparseMatrix::from_triplets(n, n, &triplets)
}

#[test]
fn simplicial_ldlt_matches_dense() {
    let a = chain_spd(8);
    let mut ldlt = SimplicialLdl::new();
    assert!(ldlt.analyze_and_factorize(&a.to_lower()));

    let b: Vec<f64> = (0..8).map(|i| (i as f64) - 2.5).collect();
    let x = ldlt.solve(&b).expect("sparse solve");
    let expected = dense_solve(&a, &b);
    for (a, e) in x.iter().zip(&expected) {
        assert_relative_eq!(a, e, epsilon = 1e-10);
    }
}

#[test]
fn sparse_solver_without_schur() {
    let a = chain_spd(6);
    let options = Options::default().linear;
    let mut solver = SparseLinearSolvers::<f64>::new(options);
    assert!(solver.analyze(&a, 0));
    assert!(solver.compute(&a));

    let b = vec![1.0, -1.0, 2.0, 0.0, 3.0, -2.0];
    let x = solver.solve(&a, &b).expect("solve");
    let expected = dense_solve(&a, &b);
    for (a, e) in x.iter().zip(&expected) {
        assert_relative_eq!(a, e, epsilon = 1e-10);
    }
}

fn block_system() -> SparseMatrix<f64> {
    // 2 reduced + 3 eliminated variables.
    let mut triplets = Vec::new();
    // H_rr (2x2 SPD).
    triplets.extend([(0, 0, 5.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 4.0)]);
    // H_ee (3x3 SPD).
    triplets.extend([
        (2, 2, 6.0),
        (2, 3, 1.0),
        (3, 2, 1.0),
        (3, 3, 5.0),
        (3, 4, 1.0),
        (4, 3, 1.0),
        (4, 4, 4.0),
    ]);
    // Coupling C (2x3) and its transpose.
    triplets.extend([
        (0, 2, 1.0),
        (2, 0, 1.0),
        (1, 3, 2.0),
        (3, 1, 2.0),
        (0, 4, -1.0),
        (4, 0, -1.0),
    ]);
    SparseMatrix::from_triplets(5, 5, &triplets)
}

#[test]
fn sparse_solver_schur_matches_dense() {
    let a = block_system();
    let options = Options::default().linear;
    let mut solver = SparseLinearSolvers::<f64>::new(options);
    assert!(solver.analyze(&a, 2), "analyze schur");
    assert!(solver.compute(&a), "compute schur");

    let b = vec![1.0, 2.0, -1.0, 0.5, 3.0];
    let x = solver.solve(&a, &b).expect("schur solve");
    let expected = dense_solve(&a, &b);
    for (a, e) in x.iter().zip(&expected) {
        assert_relative_eq!(a, e, epsilon = 1e-10);
    }
}

#[test]
fn schur_disabled_is_rejected() {
    let a = block_system();
    let mut options = Options::default().linear;
    options.allow_schur = false;
    let mut solver = SparseLinearSolvers::<f64>::new(options);
    assert!(!solver.analyze(&a, 2));
}

#[test]
fn sparse_inverse_matches_dense() {
    let a = chain_spd(5);
    let mut solver = SparseLinearSolvers::<f64>::new(Options::default().linear);
    assert!(solver.analyze(&a, 0));
    assert!(solver.compute(&a));

    let inv = solver.inverse(&a).expect("inverse");
    let dense = a.to_dense();
    let inv_dense = inv.to_dense();
    // A * A^{-1} = I.
    for i in 0..5 {
        for j in 0..5 {
            let mut acc = 0.0;
            for k in 0..5 {
                acc += dense.elem(i, k) * inv_dense.elem(k, j);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(acc, expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn singular_system_fails_cleanly() {
    // Zero matrix is not positive definite.
    let a = SparseMatrix::<f64>::new(3, 3);
    let mut solver = SparseLinearSolvers::<f64>::new(Options::default().linear);
    assert!(solver.analyze(&a, 0));
    assert!(!solver.compute(&a));
    assert!(solver.solve(&a, &[1.0, 0.0, 0.0]).is_none());
}

#[test]
fn vector_helpers_are_consistent() {
    let _ = <Vector<f64, Dynamic> as DenseVector<f64>>::zeros(1);
}
