// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! AMD ordering tests (M7): exactness and fill-in reduction.

use approx::assert_relative_eq;
use tinyopt::Ordering;
use tinyopt::backend::{DenseVector, Dynamic, LinearSolverMethod, zeros_vector};
use tinyopt::sparse::{SimplicialLdl, SparseMatrix};

/// Star graph: node 0 connected to every leaf, plus a dominant diagonal.
fn star_spd(n: usize) -> SparseMatrix<f64> {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i, i, n as f64));
        if i > 0 {
            triplets.push((i, 0, 1.0));
            triplets.push((0, i, 1.0));
        }
    }
    SparseMatrix::from_triplets(n, n, &triplets)
}

fn dense_solve(a: &SparseMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let m = a.to_dense();
    let mut rhs = zeros_vector::<f64, Dynamic>(n);
    for (i, v) in b.iter().enumerate() {
        rhs.set_at(i, *v);
    }
    let x = tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &m, &rhs)
        .expect("dense solve");
    (0..n).map(|i| x.at(i)).collect()
}

#[test]
fn amd_solution_matches_dense() {
    let a = star_spd(20);
    let b: Vec<f64> = (0..20).map(|i| (i as f64) * 0.3 - 2.0).collect();
    let expected = dense_solve(&a, &b);
    for ordering in [Ordering::Natural, Ordering::Amd] {
        let mut ldlt = SimplicialLdl::default();
        assert!(ldlt.analyze_and_factorize_with(&a.to_lower(), ordering));
        let x = ldlt.solve(&b).expect("solve");
        for (a, e) in x.iter().zip(&expected) {
            assert_relative_eq!(a, e, epsilon = 1e-10);
        }
    }
}

#[test]
fn amd_reduces_fill_in_on_a_star() {
    let n = 40;
    let a = star_spd(n);
    let lower = a.to_lower();

    let mut natural = SimplicialLdl::default();
    assert!(natural.analyze_pattern_with(&lower, Ordering::Natural));
    let mut amd = SimplicialLdl::default();
    assert!(amd.analyze_pattern_with(&lower, Ordering::Amd));

    // Natural order eliminates the centre first, creating a large clique.
    assert!(
        amd.fill_in() * 4 < natural.fill_in(),
        "expected AMD fill {} to be much smaller than natural fill {}",
        amd.fill_in(),
        natural.fill_in()
    );
    assert_eq!(amd.permutation().len(), n);
}

#[test]
fn amd_ordering_is_a_permutation() {
    let a = star_spd(12);
    let lower = a.to_lower();
    let mut ldlt = SimplicialLdl::default();
    assert!(ldlt.analyze_pattern_with(&lower, Ordering::Amd));
    let mut sorted = ldlt.permutation().to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..12).collect::<Vec<_>>());
}

#[test]
fn sparse_solver_ordering_option() {
    let a = star_spd(16);
    let b: Vec<f64> = (0..16).map(|i| 1.0 + i as f64).collect();
    let expected = dense_solve(&a, &b);
    for ordering in [Ordering::Natural, Ordering::Amd] {
        let options = tinyopt::options::LinearSolverOptions {
            ordering,
            ..Default::default()
        };
        let mut solver = tinyopt::sparse::SparseLinearSolvers::<f64>::new(options);
        assert!(solver.analyze(&a, 0));
        assert!(solver.compute(&a));
        let x = solver.solve(&a, &b).expect("solve");
        for (a, e) in x.iter().zip(&expected) {
            assert_relative_eq!(a, e, epsilon = 1e-9);
        }
    }
}
