// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/cov_options.cpp (dense parts).

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::optimizers::lm;
use tinyopt_rs::{Cost, CovarianceOptions, Options, Static};

type V2 = Vector<f64, Static<2>>;
type M2 = Matrix<f64, Static<2>>;

fn vec2(x: f64, y: f64) -> V2 {
    let mut v = <V2 as DenseVector<f64>>::zeros(2);
    v.set_at(0, x);
    v.set_at(1, y);
    v
}

/// r = [2 (x0 - 1), x1 - 2] -> H = diag(4, 1).
fn anisotropic() -> impl Fn(&[f64], Option<&mut V2>, Option<&mut M2>) -> Cost {
    |x: &[f64], grad: Option<&mut V2>, h: Option<&mut M2>| -> Cost {
        let r0 = 2.0 * (x[0] - 1.0);
        let r1 = x[1] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, 4.0 * r0);
            g.set_at(1, 2.0 * r1);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 4.0);
            h.set_elem(1, 1, 1.0);
        }
        Cost::with_num_residuals(r0 * r0 + r1 * r1, 2)
    }
}

#[test]
fn fixed_variances_are_zeroed() {
    let mut opt = lm::Optimizer::<V2>::new(Options::default());
    opt.fix(&[1]);
    let mut x = vec2(0.0, 0.0);
    let summary = opt.optimize(&mut x, anisotropic());
    assert!(summary.succeeded());

    let cov = opt
        .covariance_with(&CovarianceOptions::default())
        .expect("covariance");
    assert_relative_eq!(cov.elem(0, 0), 0.25, epsilon = 1e-6);
    assert_relative_eq!(cov.elem(1, 1), 0.0);
}

#[test]
fn fixed_variances_can_be_kept() {
    let mut opt = lm::Optimizer::<V2>::new(Options::default());
    opt.fix(&[1]);
    let mut x = vec2(0.0, 0.0);
    let _ = opt.optimize(&mut x, anisotropic());

    let options = CovarianceOptions {
        zero_fixed_variances: false,
        ..Default::default()
    };
    let cov = opt.covariance_with(&options).expect("covariance");
    assert_relative_eq!(cov.elem(0, 0), 0.25, epsilon = 1e-6);
    assert_relative_eq!(cov.elem(1, 1), 1.0, epsilon = 1e-6);
}

#[test]
fn scale_by_residual_variance() {
    type V2 = Vector<f64, Static<2>>;
    type M2 = Matrix<f64, Static<2>>;
    // Three residuals, two parameters: r = [x0 - 1, x1 - 2, 0.5].
    let acc = |x: &[f64], grad: Option<&mut V2>, h: Option<&mut M2>| -> Cost {
        let r0 = x[0] - 1.0;
        let r1 = x[1] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r0);
            g.set_at(1, 2.0 * r1);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
            h.set_elem(1, 1, 2.0);
        }
        Cost::with_num_residuals(r0 * r0 + r1 * r1 + 0.25, 3)
    };

    let mut opt = lm::Optimizer::<V2>::new(Options::default());
    let mut x = vec2(0.0, 0.0);
    let summary = opt.optimize(&mut x, acc);
    assert!(summary.succeeded());
    // At the minimum the cost is 0.25, s² = 0.25 / (3 - 2) = 0.25.
    let options = CovarianceOptions {
        scale_by_residual_variance: true,
        ..Default::default()
    };
    let cov = opt.covariance_with(&options).expect("covariance");
    assert_relative_eq!(cov.elem(0, 0), 0.5 * 0.25, epsilon = 1e-6);
    assert_relative_eq!(cov.elem(1, 1), 0.5 * 0.25, epsilon = 1e-6);
}
