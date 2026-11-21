// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Block-tridiagonal matrix, factorization and optimizer tests (M6c).
//! Mirrors tests/cpp/block_tridiag_ldlt.cpp and block_tridiag_optimizer.cpp.

use approx::assert_relative_eq;
use tinyopt::backend::{
    DenseMatrix, DenseVector, Dynamic, LinearSolverMethod, Vector, zeros_vector,
};
use tinyopt::block_tridiag::{BlockTridiagLdlt, BlockTridiagMat};
use tinyopt::hessian::HessianStorage;
use tinyopt::optimizers::lm;
use tinyopt::{BlockMut, Cost, CovariancePattern, Options, optimize_tridiag};

type Vx = Vector<f64, Dynamic>;

fn spd_chain<const B: usize, const N: usize>() -> BlockTridiagMat<B, f64> {
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

fn dense_solve_matrix(m: &BlockTridiagMat<0, f64>, b: &[f64]) -> Vec<f64> {
    let dense = m.to_dense();
    let mut rhs = zeros_vector::<f64, Dynamic>(b.len());
    for (i, v) in b.iter().enumerate() {
        rhs.set_at(i, *v);
    }
    let x = tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &dense, &rhs)
        .expect("dense solve");
    (0..b.len()).map(|i| x.at(i)).collect()
}

fn dense_solve_fixed<const B: usize, const N: usize>(
    m: &BlockTridiagMat<B, f64>,
    b: &[f64],
) -> Vec<f64> {
    let dense = m.to_dense();
    let mut rhs = zeros_vector::<f64, Dynamic>(b.len());
    for (i, v) in b.iter().enumerate() {
        rhs.set_at(i, *v);
    }
    let x = tinyopt::backend::solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &dense, &rhs)
        .expect("dense solve");
    (0..b.len()).map(|i| x.at(i)).collect()
}

#[test]
fn sparse_and_block_sparse_conversions() {
    let m = spd_chain::<2, 4>();
    let dense = m.to_dense();

    let sparse = m.to_sparse();
    assert_eq!(sparse.nrows(), m.dims());
    for j in 0..m.dims() {
        for i in 0..m.dims() {
            assert_relative_eq!(sparse.coeff(i, j), dense.elem(i, j), epsilon = 1e-12);
        }
    }

    let block = m.to_block_sparse();
    assert_eq!(block.dims(), m.dims());
    let block_dense = block.to_dense();
    for j in 0..m.dims() {
        for i in 0..m.dims() {
            assert_relative_eq!(block_dense.elem(i, j), dense.elem(i, j), epsilon = 1e-12);
        }
    }
}

#[test]
fn matrix_access_and_dense_conversion() {
    let mut m = BlockTridiagMat::<2, f64>::new(3);
    assert_eq!(m.num_blocks(), 3);
    assert_eq!(m.block_size(), 2);
    assert_eq!(m.dims(), 6);

    m.diagonal(0).set_at(0, 0, 2.0);
    m.off_diagonal(0).set_at(1, 0, 3.0);
    assert_relative_eq!(m.get(0, 0), 2.0);
    // Off block 0: local (1, 0) -> global (1, 2).
    assert_relative_eq!(m.get(1, 2), 3.0);
    assert_relative_eq!(m.get(2, 1), 3.0);

    let dense = m.to_dense();
    assert_relative_eq!(dense.elem(0, 0), 2.0);
    assert_relative_eq!(dense.elem(1, 2), 3.0);
    assert_relative_eq!(dense.elem(1, 1), 0.0);
}

#[test]
fn ldl_solve_fixed_block_sizes() {
    macro_rules! check {
        ($b:literal) => {{
            let m = spd_chain::<$b, 5>();
            let b: Vec<f64> = (0..m.dims()).map(|i| (i as f64) - 2.0).collect();
            let mut ldlt = BlockTridiagLdlt::new();
            assert!(ldlt.compute(&m));
            let mut x = vec![0.0; b.len()];
            assert!(ldlt.solve_into(&b, &mut x));
            let expected = dense_solve_fixed::<$b, 5>(&m, &b);
            for (a, e) in x.iter().zip(&expected) {
                assert_relative_eq!(a, e, epsilon = 1e-10);
            }
        }};
    }
    check!(1);
    check!(2);
    check!(3);
    check!(4);
    check!(6);
}

#[test]
fn ldl_solve_dynamic_block_size() {
    for bs in [1usize, 2, 5, 7] {
        let mut m = BlockTridiagMat::<0, f64>::new_dynamic(4, bs);
        for i in 0..4 {
            let mut d = m.diagonal(i);
            for k in 0..bs {
                d.set_at(k, k, 4.0);
            }
        }
        for i in 0..3 {
            let mut o = m.off_diagonal(i);
            for k in 0..bs {
                o.set_at(k, k, -1.0);
            }
        }
        let b: Vec<f64> = (0..m.dims()).map(|i| (i as f64) * 0.5 - 1.0).collect();
        let mut ldlt = BlockTridiagLdlt::new();
        assert!(ldlt.compute(&m));
        let mut x = vec![0.0; b.len()];
        assert!(ldlt.solve_into(&b, &mut x));
        let expected = dense_solve_matrix(&m, &b);
        for (a, e) in x.iter().zip(&expected) {
            assert_relative_eq!(a, e, epsilon = 1e-10);
        }
    }
}

#[test]
fn tridiagonal_inverse_matches_dense_blocks() {
    // The block-tridiagonal inverse is the banded part of the full inverse
    // (diagonal and first super-diagonal blocks), as in the C++ API.
    const N: usize = 4;
    const B: usize = 2;
    let m = spd_chain::<B, N>();
    let mut ldlt = BlockTridiagLdlt::new();
    assert!(ldlt.compute(&m));
    let inv = ldlt.inverse_block_tridiag::<B>().expect("inverse");

    let dense = m.to_dense();
    let inv_dense = tinyopt::backend::inverse::<f64, Dynamic>(&dense).expect("dense inverse");
    for i in 0..N {
        for r in 0..B {
            for c in 0..B {
                let expected = inv_dense.elem(i * B + r, i * B + c);
                assert_relative_eq!(inv.diagonal_ref(i)[c * B + r], expected, epsilon = 1e-10);
            }
        }
    }
    for i in 0..N - 1 {
        for r in 0..B {
            for c in 0..B {
                let expected = inv_dense.elem(i * B + r, (i + 1) * B + c);
                assert_relative_eq!(
                    inv.off_diagonal_ref(i)[c * B + r],
                    expected,
                    epsilon = 1e-10
                );
            }
        }
    }
}

const TARGET: [f64; 2] = [1.0, -2.0];

/// Access to diagonal/off-diagonal blocks, abstracting over the block size.
trait ChainHessian {
    fn diag(&mut self, i: usize) -> BlockMut<'_, f64>;
    fn off(&mut self, i: usize) -> BlockMut<'_, f64>;
}

impl<const B: usize> ChainHessian for BlockTridiagMat<B, f64> {
    fn diag(&mut self, i: usize) -> BlockMut<'_, f64> {
        self.diagonal(i)
    }

    fn off(&mut self, i: usize) -> BlockMut<'_, f64> {
        self.off_diagonal(i)
    }
}

fn chain_cost<H: ChainHessian>(
    x: &[f64],
    grad: Option<&mut Vx>,
    h: Option<&mut H>,
    n: usize,
    bs: usize,
) -> Cost {
    let mut cost = 0.0;
    let mut g = grad;
    let mut h = h;
    for i in 0..n {
        for k in 0..bs {
            let idx = i * bs + k;
            let r = x[idx] - TARGET[k % 2];
            cost += r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(idx, g.at(idx) + r);
            }
            if let Some(h) = h.as_mut() {
                let mut e = vec![0.0; bs];
                e[k] = 1.0;
                h.diag(i).add_outer_product_self(1.0, &e);
            }
        }
    }
    for i in 0..n.saturating_sub(1) {
        for k in 0..bs {
            let r = x[(i + 1) * bs + k] - x[i * bs + k];
            cost += r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(i * bs + k, g.at(i * bs + k) - r);
                g.set_at((i + 1) * bs + k, g.at((i + 1) * bs + k) + r);
            }
            if let Some(h) = h.as_mut() {
                let mut e = vec![0.0; bs];
                e[k] = 1.0;
                h.diag(i).add_outer_product_self(1.0, &e);
                h.diag(i + 1).add_outer_product_self(1.0, &e);
                h.off(i).add_outer_product(-1.0, &e, &e);
            }
        }
    }
    Cost::with_num_residuals(cost, (n * bs + n.saturating_sub(1) * bs) as i32)
}

#[test]
fn optimizer_with_block_tridiagonal_hessian() {
    const N: usize = 5;
    let mut x = vec![0.0; 2 * N];
    let acc = |xv: &[f64], g: Option<&mut Vx>, h: Option<&mut BlockTridiagMat<2, f64>>| {
        chain_cost::<_>(xv, g, h, N, 2)
    };

    let mut options = Options::default();
    options.exp.final_cov = true;
    let summary = optimize_tridiag::<2, _, _>(&mut x, acc, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    for i in 0..N {
        assert_relative_eq!(x[i * 2], TARGET[0], epsilon = 1e-6);
        assert_relative_eq!(x[i * 2 + 1], TARGET[1], epsilon = 1e-6);
    }

    let cov = summary.final_cov.as_ref().expect("covariance");
    match cov {
        tinyopt::FinalCovariance::BlockTridiag(m) => {
            assert_eq!(m.num_blocks(), N);
            assert!(m.get(0, 0) > 0.0);
        }
        other => panic!("expected a block-tridiagonal covariance, got {other:?}"),
    }

    let mut options = Options::default();
    options.exp.final_cov = true;
    options.exp.cov_options.pattern = CovariancePattern::FullDense;
    let mut x = vec![0.0; 2 * N];
    let summary = optimize_tridiag::<2, _, _>(&mut x, acc, &options);
    assert!(matches!(
        summary.final_cov,
        Some(tinyopt::FinalCovariance::Dense(_))
    ));
}

#[test]
fn dynamic_block_size_with_prototype() {
    const N: usize = 4;
    let mut x = vec![0.0; 2 * N];
    let acc = |xv: &[f64], g: Option<&mut Vx>, h: Option<&mut BlockTridiagMat<0, f64>>| {
        chain_cost::<_>(xv, g, h, N, 2)
    };

    let mut opt = lm::Optimizer::<Vec<f64>, BlockTridiagMat<0, f64>>::new(Options::default());
    opt.init_with(BlockTridiagMat::<0, f64>::new_dynamic(0, 2));
    let summary = opt.optimize(&mut x, acc);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    for i in 0..N {
        assert_relative_eq!(x[i * 2], TARGET[0], epsilon = 1e-6);
        assert_relative_eq!(x[i * 2 + 1], TARGET[1], epsilon = 1e-6);
    }
}

#[test]
fn fixed_indices_and_scaling() {
    let mut m = BlockTridiagMat::<2, f64>::new(3);
    m.set_identity();
    m.scale_diagonal(2.0);
    assert_relative_eq!(m.get(0, 0), 2.0);
    m.apply_fixed_indices(&[1]);
    assert_relative_eq!(m.get(1, 1), 1.0);
    assert_relative_eq!(m.get(0, 1), 0.0);
    let mut out = vec![0.0; m.dims()];
    m.mul_vec_into(&vec![1.0; m.dims()], &mut out);
    assert!(out.iter().all(|v| v.is_finite()));
    let _ = <BlockTridiagMat<2, f64> as HessianStorage<f64>>::zeros(6);
}
