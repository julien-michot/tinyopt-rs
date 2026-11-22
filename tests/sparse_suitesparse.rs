// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Optional CHOLMOD backend (`suitesparse` feature).

#![cfg(feature = "suitesparse")]

use approx::assert_relative_eq;
use tinyopt::Real;
use tinyopt::backend::{DenseVector, Dynamic, LinearSolverMethod, Matrix, zeros_vector};
use tinyopt::options::LinearSolverOptions;
use tinyopt::sparse::{SparseLinearSolvers, SparseMatrix, SparseSolverCache};

fn chain_spd<S: Real>(n: usize) -> SparseMatrix<S> {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i, i, S::from_f64(4.0).expect("fits")));
        if i + 1 < n {
            let off = S::from_f64(-1.0).expect("fits");
            triplets.push((i, i + 1, off));
            triplets.push((i + 1, i, off));
        }
    }
    SparseMatrix::from_triplets(n, n, &triplets)
}

fn dense_solve<S: Real>(a: &SparseMatrix<S>, b: &[S]) -> Vec<S> {
    let n = b.len();
    let m: Matrix<S, Dynamic> = a.to_dense();
    let mut rhs = zeros_vector::<S, Dynamic>(n);
    for (i, v) in b.iter().enumerate() {
        rhs.set_at(i, *v);
    }
    let x = tinyopt::backend::solve::<S, Dynamic>(LinearSolverMethod::Ldlt, &m, &rhs)
        .expect("dense solve");
    (0..n).map(|i| x.at(i)).collect()
}

fn suitesparse_options() -> LinearSolverOptions {
    LinearSolverOptions {
        method: LinearSolverMethod::SuiteSparse,
        ..Default::default()
    }
}

fn check_suitesparse<S: Real>() {
    let n = 12;
    let a = chain_spd::<S>(n);
    let b: Vec<S> = (0..n)
        .map(|i| S::from_f64((i as f64) - 2.5).expect("fits"))
        .collect();
    let expected = dense_solve(&a, &b);

    let mut solver = SparseLinearSolvers::<S>::new(suitesparse_options());
    assert!(solver.analyze(&a, 0));
    assert!(solver.compute(&a));
    let x = solver.solve(&a, &b).expect("CHOLMOD solve");
    let tol = if core::mem::size_of::<S>() == 4 {
        1e-4
    } else {
        1e-10
    };
    for (got, want) in x.iter().zip(&expected) {
        assert_relative_eq!(
            got.to_f64().expect("fits"),
            want.to_f64().expect("fits"),
            epsilon = tol
        );
    }

    // A new right-hand side reuses the symbolic and numeric factors.
    let b2: Vec<S> = (0..n)
        .map(|i| S::from_f64(1.0 + i as f64).expect("fits"))
        .collect();
    let expected2 = dense_solve(&a, &b2);
    let x2 = solver.solve(&a, &b2).expect("second solve");
    for (got, want) in x2.iter().zip(&expected2) {
        assert_relative_eq!(
            got.to_f64().expect("fits"),
            want.to_f64().expect("fits"),
            epsilon = tol
        );
    }
}

#[test]
fn cholmod_matches_dense_f64() {
    check_suitesparse::<f64>();
}

#[test]
fn cholmod_matches_dense_f32() {
    check_suitesparse::<f32>();
}

#[test]
fn cholmod_factors_the_schur_eliminated_block() {
    // With a Schur partition CHOLMOD factorizes the eliminated bottom-right
    // block (mirroring the C++ dispatch); the Schur solution must stay exact.
    let n = 8;
    let a = chain_spd::<f64>(n);
    let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();
    let expected = dense_solve(&a, &b);

    let mut solver = SparseLinearSolvers::<f64>::new(suitesparse_options());
    assert!(solver.analyze(&a, 2));
    assert!(solver.compute(&a));
    let x = solver.solve(&a, &b).expect("Schur solve");
    for (got, want) in x.iter().zip(&expected) {
        assert_relative_eq!(*got, *want, epsilon = 1e-10);
    }
}

#[test]
fn cache_reuses_cholmod_factorization() {
    let n = 10;
    let a = chain_spd::<f64>(n);
    let b: Vec<f64> = (0..n).map(|i| 1.0 - i as f64).collect();
    let expected = dense_solve(&a, &b);

    let mut cache = SparseSolverCache::<f64>::default();
    let options = suitesparse_options();
    let mut out = vec![0.0; n];
    assert!(cache.solve_into(&a, &options, &b, 0, &mut out));
    for (got, want) in out.iter().zip(&expected) {
        assert_relative_eq!(*got, *want, epsilon = 1e-10);
    }
    // Second call reuses the analyzed pattern and refactorizes the values.
    out.iter_mut().for_each(|v| *v = 0.0);
    assert!(cache.solve_into(&a, &options, &b, 0, &mut out));
    for (got, want) in out.iter().zip(&expected) {
        assert_relative_eq!(*got, *want, epsilon = 1e-10);
    }
}
