// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Sparse optimizer integration (M6b): sparse Hessians, reduced/Schur blocks and
//! covariance patterns. Mirrors the dense parts of tests/cpp/{sparse,schur,
//! cov_options}.cpp.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Dynamic, Matrix, Vector, zeros_vector};
use tinyopt_rs::hessian::HessianStorage;
use tinyopt_rs::optimizers::{gn, lm};
use tinyopt_rs::sparse::SparseMat;
use tinyopt_rs::{Cost, CovariancePattern, Options, Params, Summary, optimize_sparse};

type VX = Vector<f64, Dynamic>;

/// Five parameters with a two-parameter reduced head block.
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

/// Residual rows (unit rows plus two coupling rows).
const JACOBIAN: [[f64; 5]; 7] = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -1.0, 0.0],
];
const TARGET: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

fn sparse_acc(x: &[f64], grad: Option<&mut VX>, h: Option<&mut SparseMat>) -> Cost {
    let mut cost = 0.0;
    let mut g = grad;
    let mut h = h;
    for (row_index, row) in JACOBIAN.iter().enumerate() {
        let rhs = match row_index {
            0..=4 => TARGET[row_index],
            5 => 4.0,
            _ => -2.0,
        };
        let value: f64 = row.iter().zip(x).map(|(j, xi)| j * xi).sum();
        let r = value - rhs;
        cost += r * r;

        if let Some(g) = g.as_mut() {
            for (i, j) in row.iter().enumerate() {
                if *j != 0.0 {
                    g.set_at(i, g.at(i) + j * r);
                }
            }
        }
        if let Some(h) = h.as_mut() {
            for (i, ji) in row.iter().enumerate() {
                if *ji == 0.0 {
                    continue;
                }
                for (k, jk) in row.iter().enumerate() {
                    if *jk == 0.0 {
                        continue;
                    }
                    *h.coeff_ref(i, k) += ji * jk;
                }
            }
        }
    }
    Cost::with_num_residuals(cost, JACOBIAN.len() as i32)
}

fn initial() -> Split {
    Split([0.0; 5])
}

#[test]
fn sparse_levenberg_marquardt_converges() {
    let mut x = initial();
    let summary = optimize_sparse(&mut x, sparse_acc, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.0[0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(x.0[1], 2.0, epsilon = 1e-6);
    assert_relative_eq!(x.0[2], 3.0, epsilon = 1e-6);
    assert_relative_eq!(x.0[3], 4.0, epsilon = 1e-6);
    assert_relative_eq!(x.0[4], 5.0, epsilon = 1e-6);
}

#[test]
fn sparse_gauss_newton_uses_schur() {
    let mut opt = gn::Optimizer::<Split, SparseMat>::new(Options::default());
    let mut x = initial();
    let summary = opt.optimize(&mut x, sparse_acc);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(opt.has_reduced());
    assert_eq!(opt.num_reduced(), 2);
    assert_relative_eq!(x.0[0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(x.0[4], 5.0, epsilon = 1e-6);
}

#[test]
fn schur_can_be_disabled() {
    let mut options = Options::default();
    options.linear.allow_schur = false;
    let mut x = initial();
    let summary = optimize_sparse(&mut x, sparse_acc, &options);
    assert_eq!(summary.stop_reason, tinyopt_rs::StopReason::SolverFailed);
}

#[test]
fn sparse_covariance_auto_and_dense_match() {
    // Auto: sparse covariance; FullDense: dense covariance with the same values.
    let mut options = Options::default();
    options.exp.final_cov = true;
    let mut x = initial();
    let summary = optimize_sparse(&mut x, sparse_acc, &options);
    assert!(summary.succeeded());
    let cov = summary.final_cov.as_ref().expect("sparse covariance");
    assert!(cov.as_sparse().is_some());

    // H = JᵀJ; inverse via the dense backend for comparison.
    let mut h = SparseMat::new(5, 5);
    for row in JACOBIAN {
        for (i, ji) in row.iter().enumerate() {
            for (k, jk) in row.iter().enumerate() {
                *h.coeff_ref(i, k) += ji * jk;
            }
        }
    }
    let dense = h.to_dense();
    let inv = tinyopt_rs::backend::inverse::<f64, Dynamic>(&dense).expect("invertible");
    for i in 0..5 {
        assert_relative_eq!(cov.elem(i, i), inv.elem(i, i) as f32, epsilon = 1e-5);
    }

    let mut options = Options::default();
    options.exp.final_cov = true;
    options.exp.cov_options.pattern = CovariancePattern::FullDense;
    let mut x = initial();
    let summary = optimize_sparse(&mut x, sparse_acc, &options);
    let cov = summary.final_cov.as_ref().expect("dense covariance");
    assert!(cov.as_dense().is_some());
    for i in 0..5 {
        assert_relative_eq!(cov.elem(i, i), inv.elem(i, i) as f32, epsilon = 1e-5);
    }
}

#[test]
fn sparse_covariance_block_diagonal() {
    let mut options = Options::default();
    options.exp.final_cov = true;
    options.exp.cov_options.pattern = CovariancePattern::BlockDiagonal;
    let mut x = initial();
    let summary = optimize_sparse(&mut x, sparse_acc, &options);
    let cov = summary.final_cov.as_ref().expect("covariance");
    let sparse = cov.as_sparse().expect("sparse diagonal");
    // Off-diagonal entries are dropped.
    assert_relative_eq!(sparse.coeff(0, 1), 0.0);
    assert!(sparse.coeff(0, 0) != 0.0);
}

#[test]
fn sparse_optimizer_covariance_accessor() {
    let mut opt = lm::Optimizer::<Split, SparseMat>::new(Options::default());
    let mut x = initial();
    let summary = opt.optimize(&mut x, sparse_acc);
    assert!(summary.succeeded());
    let cov = opt.covariance().expect("covariance");
    assert!(cov.as_sparse().is_some());
}

#[test]
fn dense_defaults_still_type_check() {
    // The dense storage remains the default.
    fn _assert_dense_default(_: Summary) {}
    let mut x = <Vector<f64, Dynamic> as DenseVector<f64>>::zeros(2);
    let _ = &mut x;
    let _ = zeros_vector::<f64, Dynamic>(2);
    let _ = <Matrix<f64, Dynamic> as HessianStorage<f64>>::zeros(2);
    _assert_dense_default(Summary::default());
}
