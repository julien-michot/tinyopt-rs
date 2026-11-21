// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/tutorial_fixed_schur_cov.cpp: fixed parameters, Schur
//! reduction and covariance export combined.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseVector, Dynamic, Vector};
use tinyopt::sparse::SparseMat;
use tinyopt::{Cost, Options, Params, optimize_sparse};

type Vx = Vector<f64, Dynamic>;

const TARGET: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

#[derive(Clone, Debug)]
struct SchurParams {
    values: [f64; 6],
    fixed: Option<usize>,
}

impl Params for SchurParams {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dims(&self) -> usize {
        6
    }

    fn as_slice(&self) -> &[f64] {
        &self.values
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        for (v, d) in self.values.iter_mut().zip(delta) {
            *v += d;
        }
    }

    fn norm2(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum()
    }

    fn reduced(&self) -> Option<(usize, usize)> {
        // The first two parameters are the reduced Schur head block.
        Some((0, 2))
    }

    fn fixed(&self) -> Option<Vec<usize>> {
        self.fixed.map(|i| vec![i])
    }
}

fn accumulate(x: &[f64], grad: Option<&mut Vx>, h: Option<&mut SparseMat>) -> Cost {
    let mut cost = 0.0;
    let mut g = grad;
    let mut h = h;
    for i in 0..6 {
        let r = x[i] - TARGET[i];
        cost += r * r;
        if let Some(g) = g.as_mut() {
            g.set_at(i, g.at(i) + r);
        }
        if let Some(h) = h.as_mut() {
            *h.coeff_ref(i, i) += 1.0;
        }
    }
    Cost::with_num_residuals(cost, 6)
}

#[test]
fn fixed_schur_covariance_end_to_end() {
    let mut options = Options::default();
    options.stop.max_iters = 100;
    options.stop.min_rerr_dec = 0.0;
    options.stop.min_error = 0.0;
    options.exp.final_cov = true;
    options.log.enable = false;

    let mut x = SchurParams {
        values: [0.0; 6],
        fixed: Some(3),
    };
    let summary = optimize_sparse(&mut x, accumulate, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_eq!(x.values[3], 0.0, "fixed parameter must not move");
    for i in [0usize, 1, 2, 4, 5] {
        assert_relative_eq!(x.values[i], TARGET[i], epsilon = 1e-5);
    }

    let cov = summary.final_cov.as_ref().expect("final covariance");
    let (nrows, ncols) = (cov.elem(0, 0), cov.elem(5, 5));
    assert!(nrows.is_finite() && ncols.is_finite());
    // Fixed parameters keep a zero variance column/row.
    assert_eq!(cov.elem(3, 3), 0.0);
    assert!(cov.elem(0, 0) > 0.0);
}

#[test]
fn schur_reduction_reports_the_reduced_block() {
    let params = SchurParams {
        values: [0.0; 6],
        fixed: None,
    };
    assert_eq!(params.reduced(), Some((0, 2)));
    assert_eq!(params.dims(), 6);

    let mut options = Options::default();
    options.stop.min_rerr_dec = 0.0;
    options.stop.min_error = 0.0;
    options.log.enable = false;
    let mut x = params;
    let summary = optimize_sparse(&mut x, accumulate, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    for (value, target) in x.values.iter().zip(TARGET) {
        assert_relative_eq!(*value, target, epsilon = 1e-5);
    }
}
