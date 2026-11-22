// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! IC(0) factorization and preconditioned CG (`IncompleteCholesky` method).

use approx::assert_relative_eq;
use tinyopt::backend::{DenseVector, Dynamic, LinearSolverMethod, Matrix, zeros_vector};
use tinyopt::options::LinearSolverOptions;
use tinyopt::sparse::{IncompleteCholesky, SparseLinearSolvers, SparseMatrix};

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

fn grid_spd(n: usize) -> SparseMatrix<f64> {
    let mut triplets = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            triplets.push((k, k, 4.0));
            if j + 1 < n {
                triplets.push((k, k + 1, -1.0));
                triplets.push((k + 1, k, -1.0));
            }
            if i + 1 < n {
                triplets.push((k, k + n, -1.0));
                triplets.push((k + n, k, -1.0));
            }
        }
    }
    SparseMatrix::from_triplets(n * n, n * n, &triplets)
}

fn dense_solve(a: &SparseMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let dense: Matrix<f64, Dynamic> = a.to_dense();
    let mut rhs = zeros_vector::<f64, Dynamic>(n);
    for (i, v) in b.iter().enumerate() {
        rhs.set_at(i, *v);
    }
    let x = tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &dense, &rhs)
        .expect("dense solve");
    (0..n).map(|i| x.at(i)).collect()
}

#[test]
fn ic_is_exact_for_a_tridiagonal_matrix() {
    // A tridiagonal pattern has no fill, so IC(0) reproduces the full Cholesky.
    let n = 12;
    let a = chain_spd(n);
    let mut ic = IncompleteCholesky::<f64>::new();
    assert!(ic.compute(&a.to_lower()));
    assert!(ic.is_computed());

    let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.3 - 1.5).collect();
    let mut out = vec![0.0; n];
    ic.apply(&b, &mut out);
    let expected = dense_solve(&a, &b);
    for (got, want) in out.iter().zip(&expected) {
        assert_relative_eq!(*got, *want, epsilon = 1e-10);
    }
}

#[test]
fn ic_rejects_non_positive_pivots() {
    // Indefinite matrix (negative diagonal entry).
    let a = SparseMatrix::<f64>::from_triplets(2, 2, &[(0, 0, -1.0), (1, 1, 2.0)]);
    let mut ic = IncompleteCholesky::<f64>::new();
    assert!(!ic.compute(&a.to_lower()));
    assert!(!ic.is_computed());
}

#[test]
fn preconditioned_cg_matches_ldlt_on_a_2d_laplacian() {
    let n = 20;
    let a = grid_spd(n);
    let dims = n * n;
    let b: Vec<f64> = (0..dims).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
    let expected = dense_solve(&a, &b);

    let options = LinearSolverOptions {
        method: LinearSolverMethod::IncompleteCholesky,
        ..Default::default()
    };
    let mut solver = SparseLinearSolvers::<f64>::new(options);
    assert!(solver.analyze(&a, 0));
    assert!(solver.compute(&a));
    let x = solver.solve(&a, &b).expect("IC-PCG solve");
    for (got, want) in x.iter().zip(&expected) {
        assert_relative_eq!(*got, *want, epsilon = 1e-7);
    }
}

#[test]
fn preconditioned_cg_handles_a_zero_right_hand_side() {
    let a = grid_spd(6);
    let options = LinearSolverOptions {
        method: LinearSolverMethod::IncompleteCholesky,
        ..Default::default()
    };
    let mut solver = SparseLinearSolvers::<f64>::new(options);
    assert!(solver.analyze(&a, 0));
    assert!(solver.compute(&a));
    let dims = a.nrows();
    let x = solver.solve(&a, &vec![0.0; dims]).expect("zero solve");
    assert!(x.iter().all(|v| *v == 0.0));
}

#[test]
fn preconditioned_cg_with_a_schur_partition() {
    let a = grid_spd(10);
    let dims = a.nrows();
    let b: Vec<f64> = (0..dims).map(|i| (i as f64 * 0.25).sin()).collect();
    let expected = dense_solve(&a, &b);

    let options = LinearSolverOptions {
        method: LinearSolverMethod::IncompleteCholesky,
        ..Default::default()
    };
    let mut solver = SparseLinearSolvers::<f64>::new(options);
    assert!(solver.analyze(&a, 8));
    assert!(solver.compute(&a));
    let x = solver.solve(&a, &b).expect("Schur IC-PCG solve");
    for (got, want) in x.iter().zip(&expected) {
        assert_relative_eq!(*got, *want, epsilon = 1e-6);
    }
}
