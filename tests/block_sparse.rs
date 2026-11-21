// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Block-sparse matrix, Schur solver and optimizer tests (M6d).
//! Mirrors tests/cpp/block_sparse_matrix.cpp, block_sparse_ldlt.cpp and
//! block_sparse_optimizer.cpp.

use approx::assert_relative_eq;
use tinyopt::backend::{
    DenseMatrix, DenseVector, Dynamic, LinearSolverMethod, Matrix, Vector, zeros_vector,
};
use tinyopt::block_sparse::{BlockSparseLdlt, BlockSparseMat};
use tinyopt::block_sparse_types;
use tinyopt::optimizers::lm;
use tinyopt::{Cost, Options, Params, Summary};

block_sparse_types!(TwoThree, [2, 3]);
block_sparse_types!(Threes, [3]);

type H = BlockSparseMat<TwoThree, f64>;
type Vx = Vector<f64, Dynamic>;

const TARGET: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

fn dense_solve(a: &Matrix<f64, Dynamic>, b: &[f64]) -> Vec<f64> {
    let mut rhs = zeros_vector::<f64, Dynamic>(b.len());
    for (i, v) in b.iter().enumerate() {
        rhs.set_at(i, *v);
    }
    let x = tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, a, &rhs)
        .expect("dense solve");
    (0..b.len()).map(|i| x.at(i)).collect()
}

#[test]
fn matrix_blocks_and_conversion() {
    let mut h = H::new();
    assert_eq!(h.dims(), 0);
    h.block_mut::<2, 2>(0, 0).set_at(0, 0, 4.0);
    h.block_mut::<3, 3>(0, 0).set_at(1, 1, 5.0);
    h.block_mut::<2, 3>(0, 0).set_at(0, 0, 1.0);
    assert_eq!(h.block_counts(), &[1, 1]);
    assert_eq!(h.dims(), 5);
    assert_eq!(h.num_allocated_blocks(), 3);

    assert_relative_eq!(h.get(0, 0), 4.0);
    assert_relative_eq!(h.get(3, 3), 5.0); // block (1,1) local (1,1)
    assert_relative_eq!(h.get(0, 2), 1.0); // coupling local (0,0)
    assert_relative_eq!(h.get(2, 0), 1.0); // symmetric

    let dense = h.to_dense();
    assert_relative_eq!(dense.elem(0, 2), 1.0);
    let sparse = h.to_sparse();
    assert_relative_eq!(sparse.coeff(2, 0), 1.0);

    h.set_identity();
    assert_relative_eq!(h.get(0, 0), 1.0);
    assert_relative_eq!(h.get(0, 2), 0.0);
    h.scale_diagonal(3.0);
    assert_relative_eq!(h.get(1, 1), 3.0);
    h.apply_fixed_indices(&[1]);
    assert_relative_eq!(h.get(1, 1), 1.0);

    let mut out: Vec<f64> = vec![0.0; h.dims()];
    h.mul_vec_into(&vec![1.0; h.dims()], &mut out);
    assert!(out.iter().all(|v| v.is_finite()));
}

fn spd_block_matrix() -> H {
    let mut h = H::with_counts(&[1, 1]);
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
fn schur_solver_matches_dense() {
    let h = spd_block_matrix();
    let dense = h.to_dense();
    let b = [1.0, -2.0, 3.0, 0.5, 2.0];
    let expected = dense_solve(&dense, &b);

    // Direct factorization (num_reduced == 0).
    let mut ldlt = BlockSparseLdlt::<TwoThree, f64>::new();
    assert!(ldlt.analyze(&h, 0));
    assert!(ldlt.compute(&h));
    let mut x = vec![0.0; 5];
    assert!(ldlt.solve_into(&h, &b, &mut x));
    for (a, e) in x.iter().zip(&expected) {
        assert_relative_eq!(a, e, epsilon = 1e-9);
    }

    // Schur partition after the first block type (2 scalars).
    assert!(ldlt.analyze(&h, 2));
    assert!(ldlt.compute(&h));
    assert_eq!(ldlt.num_reduced(), 2);
    let mut x = vec![0.0; 5];
    assert!(ldlt.solve_into(&h, &b, &mut x));
    for (a, e) in x.iter().zip(&expected) {
        assert_relative_eq!(a, e, epsilon = 1e-9);
    }
    // Covariance helpers return the corresponding blocks of H⁻¹.
    let inv = tinyopt::inverse::<f64, Dynamic>(&dense).expect("dense inverse");
    let s_inv = ldlt.inverse_schur_reduced().expect("reduced inverse");
    for j in 0..2 {
        for i in 0..2 {
            assert_relative_eq!(s_inv.elem(i, j), inv.elem(i, j), epsilon = 1e-9);
        }
    }
    let m = ldlt
        .marginal_eliminated_block(2, 3)
        .expect("eliminated marginal");
    for j in 0..3 {
        for i in 0..3 {
            assert_relative_eq!(m.elem(i, j), inv.elem(2 + i, 2 + j), epsilon = 1e-9);
        }
    }
}

#[test]
fn invalid_schur_boundary_is_rejected() {
    let h = spd_block_matrix();
    let mut ldlt = BlockSparseLdlt::<TwoThree, f64>::new();
    // 3 is not a type boundary (types are 2 and 3).
    assert!(!ldlt.analyze(&h, 3));
}

#[derive(Clone, Debug)]
struct Split([f64; 5]);

impl Params for Split {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dims(&self) -> usize {
        5
    }

    fn as_slice(&self) -> &[f64] {
        &self.0
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        for (v, d) in self.0.iter_mut().zip(delta) {
            *v += *d;
        }
    }

    fn norm2(&self) -> f64 {
        self.0.iter().map(|v| v * v).sum()
    }

    fn reduced(&self) -> Option<(usize, usize)> {
        Some((0, 2))
    }
}

/// Prior residuals on all 5 parameters plus two coupling residuals.
fn block_acc(x: &[f64], grad: Option<&mut Vx>, mut h: Option<&mut H>) -> Cost {
    let mut cost = 0.0;
    let mut g = grad;
    for i in 0..5 {
        let r = x[i] - TARGET[i];
        cost += r * r;
        if let Some(g) = g.as_mut() {
            g.set_at(i, g.at(i) + r);
        }
    }
    let r5 = x[0] + x[2] - 4.0;
    let r6 = x[1] - x[3] + 2.0;
    cost += r5 * r5 + r6 * r6;
    if let Some(g) = g.as_mut() {
        g.set_at(0, g.at(0) + r5);
        g.set_at(2, g.at(2) + r5);
        g.set_at(1, g.at(1) + r6);
        g.set_at(3, g.at(3) - r6);
    }

    if let Some(h) = h.as_mut() {
        // Prior identity blocks.
        let e0 = [1.0, 0.0];
        let e1 = [0.0, 1.0];
        let f0 = [1.0, 0.0, 0.0];
        let f1 = [0.0, 1.0, 0.0];
        let f2 = [0.0, 0.0, 1.0];
        h.block_mut::<2, 2>(0, 0).add_outer_product_self(1.0, &e0);
        h.block_mut::<2, 2>(0, 0).add_outer_product_self(1.0, &e1);
        h.block_mut::<3, 3>(0, 0).add_outer_product_self(1.0, &f0);
        h.block_mut::<3, 3>(0, 0).add_outer_product_self(1.0, &f1);
        h.block_mut::<3, 3>(0, 0).add_outer_product_self(1.0, &f2);
        // Coupling rows: [1,0,1,0,0] and [0,1,0,-1,0].
        h.block_mut::<2, 2>(0, 0).add_outer_product_self(1.0, &e0);
        h.block_mut::<3, 3>(0, 0).add_outer_product_self(1.0, &f0);
        h.block_mut::<2, 3>(0, 0).add_outer_product(1.0, &e0, &f0);
        h.block_mut::<2, 2>(0, 0).add_outer_product_self(1.0, &e1);
        h.block_mut::<3, 3>(0, 0).add_outer_product_self(1.0, &f1);
        h.block_mut::<2, 3>(0, 0).add_outer_product(-1.0, &e1, &f1);
    }
    Cost::with_num_residuals(cost, 7)
}

#[test]
fn optimizer_with_block_sparse_schur() {
    let mut x = Split([0.0; 5]);
    let mut opt = lm::Optimizer::<Split, H>::new(Options::default());
    opt.init_with(H::with_counts(&[1, 1]));
    let summary = opt.optimize(&mut x, block_acc);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(opt.has_reduced());
    assert_eq!(opt.num_reduced(), 2);
    for (v, t) in x.0.iter().zip(&TARGET) {
        assert_relative_eq!(v, t, epsilon = 1e-6);
    }
}

#[test]
fn block_sparse_covariance_patterns() {
    let mut options = Options::default();
    options.exp.final_cov = true;
    let mut x = Split([0.0; 5]);
    let mut opt = lm::Optimizer::<Split, H>::new(options);
    opt.init_with(H::with_counts(&[1, 1]));
    let summary = opt.optimize(&mut x, block_acc);
    assert!(summary.succeeded());
    let cov = summary.final_cov.as_ref().expect("covariance");
    match cov {
        tinyopt::FinalCovariance::BlockSparse(m) => {
            assert_eq!(m.dims(), 5);
            assert!(m.get(0, 0) > 0.0);
        }
        other => panic!("expected a block-sparse covariance, got {other:?}"),
    }

    let mut options = Options::default();
    options.exp.final_cov = true;
    options.exp.cov_options.pattern = tinyopt::CovariancePattern::FullDense;
    let mut x = Split([0.0; 5]);
    let mut opt = lm::Optimizer::<Split, H>::new(options);
    opt.init_with(H::with_counts(&[1, 1]));
    let summary = opt.optimize(&mut x, block_acc);
    assert!(matches!(
        summary.final_cov,
        Some(tinyopt::FinalCovariance::Dense(_))
    ));
}

#[test]
fn free_function_entry_point() {
    let mut x = Split([0.0; 5]);
    let acc = |xv: &[f64], g: Option<&mut Vx>, h: Option<&mut H>| block_acc(xv, g, h);
    let summary: Summary =
        tinyopt::optimize_block_sparse::<TwoThree, _, _>(&mut x, acc, &Options::default());
    assert!(summary.succeeded());
    let _ = <Matrix<f64, Dynamic> as tinyopt::hessian::HessianStorage<f64>>::zeros(5);
}

#[test]
fn tridiagonal_block_layout_matches_block_tridiagonal_solver() {
    // Mirrors tests/cpp/block_tridiag_block_sparse.cpp: a fully tridiagonal
    // block-sparse system solves like the dedicated block-tridiagonal solver.
    use tinyopt::block_tridiag::{BlockTridiagLdlt, BlockTridiagMat};
    const N: usize = 4;
    const B: usize = 3;

    let mut block = BlockSparseMat::<Threes, f64>::with_counts(&[N]);
    let mut tridiag = BlockTridiagMat::<B, f64>::new(N);
    for i in 0..N {
        for j in 0..B {
            block.set_entry(i * B + j, i * B + j, 4.0);
            tridiag.diagonal(i).set_at(j, j, 4.0);
        }
    }
    for i in 0..N - 1 {
        for j in 0..B {
            block.set_entry(i * B + j, (i + 1) * B + j, -1.0);
            block.set_entry((i + 1) * B + j, i * B + j, -1.0);
            tridiag.off_diagonal(i).set_at(j, j, -1.0);
        }
    }

    let n = N * B;
    let b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25 - 1.0).collect();

    let mut ldlt = BlockSparseLdlt::<Threes, f64>::new();
    assert!(ldlt.analyze(&block, 0));
    assert!(ldlt.compute(&block));
    let mut x = vec![0.0; n];
    assert!(ldlt.solve_into(&block, &b, &mut x));

    let mut tridiag_ldlt = BlockTridiagLdlt::new();
    assert!(tridiag_ldlt.compute(&tridiag));
    let mut x_tridiag = vec![0.0; n];
    assert!(tridiag_ldlt.solve_into(&b, &mut x_tridiag));

    for (a, e) in x.iter().zip(&x_tridiag) {
        assert_relative_eq!(a, e, epsilon = 1e-9);
    }
}

block_sparse_types!(TwoThreeSorted, [2, 3]);

#[test]
fn sparse_conversion_keeps_sorted_columns_and_matches_dense() {
    for (cams, points) in [(2usize, 20usize), (3, 7)] {
        let mut h = BlockSparseMat::<TwoThreeSorted, f64>::with_counts(&[cams, points]);
        // Dense pseudo-random blocks (deterministic, diagonally dominant).
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed % 1000) as f64) / 1000.0 - 0.5
        };
        for c in 0..cams {
            let mut block = h.block_mut::<2, 2>(c, c);
            for j in 0..2 {
                for i in j..2 {
                    let value = next() + if i == j { 5.0 } else { 0.0 };
                    block.set_at(i, j, value);
                    block.set_at(j, i, value);
                }
            }
        }
        for p in 0..points {
            let mut block = h.block_mut::<3, 3>(p, p);
            for j in 0..3 {
                for i in j..3 {
                    let value = next() + if i == j { 5.0 } else { 0.0 };
                    block.set_at(i, j, value);
                    block.set_at(j, i, value);
                }
            }
        }
        for c in 0..cams {
            for p in 0..points {
                if (c + p) % 2 == 0 {
                    let mut block = h.block_mut::<2, 3>(c, p);
                    for j in 0..3 {
                        for i in 0..2 {
                            block.set_at(i, j, next());
                        }
                    }
                }
            }
        }

        let sparse = h.to_sparse();
        let dense = h.to_dense();
        assert_eq!(sparse.nrows(), h.dims());
        for j in 0..h.dims() {
            let rows: Vec<usize> = sparse.column(j).map(|(i, _)| i).collect();
            assert!(
                rows.windows(2).all(|w| w[0] < w[1]),
                "column {j} rows are not strictly increasing: {rows:?}"
            );
            for i in 0..h.dims() {
                assert_relative_eq!(sparse.coeff(i, j), dense.elem(i, j), epsilon = 1e-12);
            }
        }
        // Triangle extraction, and mirroring from an upper-only matrix.
        let lower = sparse.to_lower();
        let upper = sparse.to_upper();
        for j in 0..h.dims() {
            for i in 0..h.dims() {
                let expected_lower = if i >= j { dense.elem(i, j) } else { 0.0 };
                let expected_upper = if i <= j { dense.elem(i, j) } else { 0.0 };
                assert_relative_eq!(lower.coeff(i, j), expected_lower, epsilon = 1e-12);
                assert_relative_eq!(upper.coeff(i, j), expected_upper, epsilon = 1e-12);
            }
        }
        let mirrored = upper.to_lower();
        for j in 0..h.dims() {
            for i in 0..h.dims() {
                assert_relative_eq!(
                    mirrored.coeff(i, j),
                    if i >= j { dense.elem(i, j) } else { 0.0 },
                    epsilon = 1e-12
                );
            }
        }
    }
}
