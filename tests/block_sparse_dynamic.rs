// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-layout block-sparse optimizer tests (M6e).
//! Mirrors tests/cpp/block_sparse_dynamic.cpp.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Dynamic, Vector};
use tinyopt::block_sparse::BlockSparseMatX;
use tinyopt::optimizers::lm;
use tinyopt::{Cost, Options, Params};

type Vx = Vector<f64, Dynamic>;
const TARGET: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

/// Residual rows: five prior rows and two coupling rows.
const ROWS: [[f64; 5]; 7] = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, -1.0, 0.0],
];
const RHS: [f64; 7] = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, -2.0];

fn dynamic_acc(x: &[f64], grad: Option<&mut Vx>, mut h: Option<&mut BlockSparseMatX<f64>>) -> Cost {
    let mut cost = 0.0;
    let mut g = grad;
    for (row, rhs) in ROWS.iter().zip(RHS) {
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
    }
    if let Some(h) = h.as_mut() {
        // Assemble H = J^T J block by block (column-major).
        let mut d00 = [0.0f64; 4];
        let mut d11 = [0.0f64; 9];
        let mut c01 = [0.0f64; 6];
        for row in ROWS {
            for i in 0..2 {
                for j in 0..2 {
                    d00[j * 2 + i] += row[i] * row[j];
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    d11[j * 3 + i] += row[2 + i] * row[2 + j];
                }
            }
            for i in 0..2 {
                for j in 0..3 {
                    c01[j * 2 + i] += row[i] * row[2 + j];
                }
            }
        }
        h.set_block(0, 0, &d00);
        h.set_block(1, 1, &d11);
        h.set_block(0, 1, &c01);
    }
    Cost::with_num_residuals(cost, ROWS.len() as i32)
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

#[test]
fn dynamic_block_sparse_optimizer() {
    let mut x = Split([0.0; 5]);
    let mut opt = lm::Optimizer::<Split, BlockSparseMatX<f64>>::new(Options::default());
    opt.init_with(BlockSparseMatX::new(&[2, 3], &[1, 1]));
    let summary = opt.optimize(&mut x, dynamic_acc);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(opt.has_reduced());
    assert_eq!(opt.num_reduced(), 2);
    for (v, t) in x.0.iter().zip(&TARGET) {
        assert_relative_eq!(v, t, epsilon = 1e-6);
    }
}

#[test]
fn dynamic_block_sparse_covariance() {
    let mut options = Options::default();
    options.exp.final_cov = true;
    let mut x = Split([0.0; 5]);
    let mut opt = lm::Optimizer::<Split, BlockSparseMatX<f64>>::new(options);
    opt.init_with(BlockSparseMatX::new(&[2, 3], &[1, 1]));
    let summary = opt.optimize(&mut x, dynamic_acc);
    assert!(summary.succeeded());
    match summary.final_cov.as_ref().expect("covariance") {
        tinyopt::FinalCovariance::BlockSparse(m) => {
            assert_eq!(m.dims(), 5);
            assert!(m.get(0, 0) > 0.0);
        }
        other => panic!("expected a block-sparse covariance, got {other:?}"),
    }
}

#[test]
fn set_block_mirrors_lower_triangle() {
    let mut h = BlockSparseMatX::<f64>::new(&[2, 3], &[1, 1]);
    h.set_block(0, 1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_relative_eq!(h.get(0, 2), 1.0);
    assert_relative_eq!(h.get(2, 0), 1.0);
    assert_relative_eq!(h.get(1, 4), 6.0);
    assert_relative_eq!(h.get(4, 1), 6.0);
    let dense = h.to_dense();
    assert_relative_eq!(dense.elem(4, 1), 6.0);
}
