// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/cov_block.cpp and block_sparse_cov.cpp: block covariance
//! exports match the dense inverse of the final Hessian.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, Dynamic};
use tinyopt::block_sparse::BlockSparseMat;
use tinyopt::block_sparse_types;
use tinyopt::block_tridiag::BlockTridiagMat;
use tinyopt::hessian::HessianStorage;
use tinyopt::{Cost, CovarianceOptions, CovariancePattern, FinalCovariance};

block_sparse_types!(TwoThree, [2, 3]);

fn tridiag_chain<const B: usize, const N: usize>() -> BlockTridiagMat<B, f64> {
    let mut m = BlockTridiagMat::<B, f64>::new(N);
    for i in 0..N {
        let mut d = m.diagonal(i);
        for k in 0..B {
            d.set_at(k, k, 4.0);
        }
    }
    for i in 0..N.saturating_sub(1) {
        let mut o = m.off_diagonal(i);
        for k in 0..B {
            o.set_at(k, k, -1.0);
        }
    }
    m
}

#[test]
fn block_tridiag_auto_covariance_matches_dense() {
    const N: usize = 4;
    const B: usize = 2;
    let m = tridiag_chain::<B, N>();
    let dense = m.to_dense();
    let inv = tinyopt::math::inverse::<f64, Dynamic>(&dense).expect("dense inverse");

    let cov = m
        .to_final_covariance(&CovarianceOptions::default(), Cost::default(), &[], 0)
        .expect("covariance");
    let FinalCovariance::BlockTridiag(cov) = cov else {
        panic!("expected a block-tridiagonal covariance");
    };
    assert_eq!(cov.block_size(), B);
    for b in 0..N {
        for j in 0..B {
            for i in 0..B {
                let expected = inv.elem(b * B + i, b * B + j);
                assert_relative_eq!(
                    cov.diagonal_ref(b)[j * B + i] as f64,
                    expected,
                    epsilon = 1e-5
                );
            }
        }
        if b + 1 < N {
            for j in 0..B {
                for i in 0..B {
                    let expected = inv.elem(b * B + i, (b + 1) * B + j);
                    assert_relative_eq!(
                        cov.off_diagonal_ref(b)[j * B + i] as f64,
                        expected,
                        epsilon = 1e-5
                    );
                }
            }
        }
    }
}

#[test]
fn block_tridiag_block_diagonal_covariance_zeroes_couplings() {
    let m = tridiag_chain::<2, 3>();
    let options = CovarianceOptions {
        pattern: CovariancePattern::BlockDiagonal,
        ..Default::default()
    };
    let cov = m
        .to_final_covariance(&options, Cost::default(), &[], 0)
        .expect("covariance");
    let FinalCovariance::BlockTridiag(cov) = cov else {
        panic!("expected a block-tridiagonal covariance");
    };
    for i in 0..cov.num_blocks() - 1 {
        assert!(cov.off_diagonal_ref(i).iter().all(|v| *v == 0.0));
    }
    for i in 0..cov.num_blocks() {
        assert!(cov.diagonal_ref(i).iter().any(|v| *v != 0.0));
    }
}

#[test]
fn block_tridiag_full_dense_covariance_matches_dense() {
    let m = tridiag_chain::<2, 3>();
    let dense = m.to_dense();
    let inv = tinyopt::math::inverse::<f64, Dynamic>(&dense).expect("dense inverse");
    let options = CovarianceOptions {
        pattern: CovariancePattern::FullDense,
        ..Default::default()
    };
    let cov = m
        .to_final_covariance(&options, Cost::default(), &[], 0)
        .expect("covariance");
    let FinalCovariance::Dense(cov) = cov else {
        panic!("expected a dense covariance");
    };
    // The block-tridiagonal factorization only provides the banded covariance
    // (diagonal and first super-diagonal blocks), like the C++ implementation.
    const B: usize = 2;
    let n = dense.nrows();
    for j in 0..n {
        for i in 0..n {
            let banded = (i / B).abs_diff(j / B) <= 1;
            let expected = if banded { inv.elem(i, j) } else { 0.0 };
            assert_relative_eq!(cov.elem(i, j) as f64, expected, epsilon = 1e-5);
        }
    }
}

fn block_spd() -> BlockSparseMat<TwoThree, f64> {
    let mut h = BlockSparseMat::<TwoThree, f64>::with_counts(&[1, 1]);
    for i in 0..5 {
        h.set_entry(i, i, 4.0);
        if i + 1 < 5 {
            h.set_entry(i, i + 1, 0.5);
            h.set_entry(i + 1, i, 0.5);
        }
    }
    h
}

#[test]
fn block_sparse_covariance_matches_dense() {
    let h = block_spd();
    let dense = h.to_dense();
    let inv = tinyopt::math::inverse::<f64, Dynamic>(&dense).expect("dense inverse");

    for pattern in [CovariancePattern::Auto, CovariancePattern::SchurPartitioned] {
        let options = CovarianceOptions {
            pattern,
            ..Default::default()
        };
        let cov = h
            .to_final_covariance(&options, Cost::default(), &[], 0)
            .expect("covariance");
        let FinalCovariance::BlockSparse(cov) = cov else {
            panic!("expected a block-sparse covariance for {pattern:?}");
        };
        for j in 0..5 {
            for i in 0..5 {
                assert_relative_eq!(cov.get(i, j) as f64, inv.elem(i, j), epsilon = 1e-5);
            }
        }
    }
}

#[test]
fn block_sparse_block_diagonal_covariance_only_keeps_blocks() {
    let h = block_spd();
    let options = CovarianceOptions {
        pattern: CovariancePattern::BlockDiagonal,
        ..Default::default()
    };
    let cov = h
        .to_final_covariance(&options, Cost::default(), &[], 0)
        .expect("covariance");
    let FinalCovariance::BlockSparse(cov) = cov else {
        panic!("expected a block-sparse covariance");
    };
    // Entries across the two block types (rows 0-1 vs columns 2-4) are zero.
    for j in 2..5 {
        for i in 0..1 {
            assert_relative_eq!(cov.get(i, j) as f64, 0.0, epsilon = 1e-12);
        }
    }
    assert!(cov.get(0, 0) > 0.0);
}
