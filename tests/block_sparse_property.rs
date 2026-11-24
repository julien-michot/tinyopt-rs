// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Property-style checks: random block matrices and triplets against dense.

use approx::assert_relative_eq;
use tinyopt::backend::{
    DenseMatrix, DenseVector, Dynamic, LinearSolverMethod, zeros_matrix, zeros_vector,
};
use tinyopt::block_sparse::{BlockSparseLdlt, BlockSparseMat};
use tinyopt::sparse::SparseMatrix;

tinyopt::block_sparse_types!(Mixed, [3, 2]);

/// Deterministic SplitMix64 in `[0, 1)`.
fn next(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn random_block_matrices_match_dense() {
    let mut state = 0x1234_5678_9abc_def0;
    for _case in 0..25 {
        let cams = 1 + (next(&mut state) * 3.0) as usize;
        let points = 1 + (next(&mut state) * 4.0) as usize;
        let dims = 3 * cams + 2 * points;
        let mut h = BlockSparseMat::<Mixed, f64>::with_counts(&[cams, points]);
        let mut dense = zeros_matrix::<f64, Dynamic>(dims, dims);

        for i in 0..cams {
            for r in 0..3 {
                for c in 0..3 {
                    let value = if r == c { 4.0 } else { 0.05 * (r + c) as f64 };
                    h.block_mut::<3, 3>(i, i).set_at(r, c, value);
                    dense.set_elem(i * 3 + r, i * 3 + c, value);
                }
            }
        }
        for j in 0..points {
            for r in 0..2 {
                for c in 0..2 {
                    let value = if r == c { 3.0 } else { 0.1 };
                    h.block_mut::<2, 2>(j, j).set_at(r, c, value);
                    dense.set_elem(3 * cams + 2 * j + r, 3 * cams + 2 * j + c, value);
                }
            }
        }
        for i in 0..cams {
            for j in 0..points {
                if next(&mut state) < 0.3 {
                    continue;
                }
                for r in 0..3 {
                    for c in 0..2 {
                        let value = next(&mut state) - 0.5;
                        h.block_mut::<3, 2>(i, j).set_at(r, c, value);
                        let row = i * 3 + r;
                        let col = 3 * cams + 2 * j + c;
                        dense.set_elem(row, col, value);
                        dense.set_elem(col, row, value);
                    }
                }
            }
        }

        // Scalar access and both conversions agree with the dense matrix.
        let sparse = h.to_sparse();
        for j in 0..dims {
            for i in 0..dims {
                assert_relative_eq!(h.get(i, j), dense.elem(i, j), epsilon = 1e-12);
                assert_relative_eq!(sparse.coeff(i, j), dense.elem(i, j), epsilon = 1e-12);
            }
        }
        let dense_from_block = h.to_dense();
        for j in 0..dims {
            for i in 0..dims {
                assert_relative_eq!(
                    dense_from_block.elem(i, j),
                    dense.elem(i, j),
                    epsilon = 1e-12
                );
            }
        }

        // Block Schur solve against the dense factorization.
        let rhs: Vec<f64> = (0..dims).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut ldlt = BlockSparseLdlt::<Mixed, f64>::new();
        assert!(ldlt.analyze(&h, 3 * cams));
        assert!(ldlt.compute(&h));
        assert!(ldlt.uses_block_schur());
        let mut x = vec![0.0; dims];
        assert!(ldlt.solve_into(&h, &rhs, &mut x));

        let mut b = zeros_vector::<f64, Dynamic>(dims);
        for (i, v) in rhs.iter().enumerate() {
            b.set_at(i, *v);
        }
        let reference =
            tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &dense, &b).unwrap();
        for (value, expected) in x.iter().zip((0..dims).map(|i| reference.at(i))) {
            assert_relative_eq!(*value, expected, epsilon = 1e-9);
        }
    }
}

#[test]
fn random_triplets_with_duplicates_match_dense() {
    let mut state = 0xdead_beef_cafe_babe;
    for _case in 0..25 {
        let n = 2 + (next(&mut state) * 8.0) as usize;
        let mut triplets = Vec::new();
        let mut dense = zeros_matrix::<f64, Dynamic>(n, n);
        let entries = 3 * n;
        for _ in 0..entries {
            let i = (next(&mut state) * n as f64) as usize % n;
            let j = (next(&mut state) * n as f64) as usize % n;
            let value = next(&mut state) - 0.5;
            triplets.push((i, j, value));
            let current = dense.elem(i, j);
            dense.set_elem(i, j, current + value);
        }
        let sparse = SparseMatrix::<f64>::from_triplets(n, n, &triplets);
        for j in 0..n {
            for i in 0..n {
                assert_relative_eq!(sparse.coeff(i, j), dense.elem(i, j), epsilon = 1e-12);
            }
        }
    }
}
