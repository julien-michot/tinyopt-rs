// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/cov_dense.cpp (dense parts): covariance export.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::optimizers::gn;
use tinyopt::{Cost, Options, Static, optimize};

type V2 = Vector<f64, Static<2>>;
type M2 = Matrix<f64, Static<2>>;

fn vec2(x: f64, y: f64) -> V2 {
    let mut v = <V2 as DenseVector<f64>>::zeros(2);
    v.set_at(0, x);
    v.set_at(1, y);
    v
}

/// r = [2 (x0 - 1), (x1 - 2)] -> J = diag(2, 1), H = diag(4, 1).
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
fn summary_exports_final_covariance() {
    let mut options = Options::default();
    options.exp.final_cov = true;

    let mut x = vec2(0.0, 0.0);
    let summary = optimize(&mut x, anisotropic(), &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(summary.has_final_cov());

    let cov = summary.final_cov.as_ref().expect("covariance");
    assert_relative_eq!(cov.elem(0, 0), 0.25, epsilon = 1e-6);
    assert_relative_eq!(cov.elem(1, 1), 1.0, epsilon = 1e-6);
    assert_relative_eq!(cov.elem(0, 1), 0.0, epsilon = 1e-6);

    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(x.at(1), 2.0, epsilon = 1e-6);
}

#[test]
fn covariance_is_disabled_by_default() {
    let mut x = vec2(0.0, 0.0);
    let summary = optimize(&mut x, anisotropic(), &Options::default());
    assert!(!summary.has_final_cov());
}

#[test]
fn optimizer_covariance_accessor() {
    let mut opt = gn::Optimizer::<V2>::new(Options::default());
    let mut x = vec2(0.0, 0.0);
    let summary = opt.optimize(&mut x, anisotropic());
    assert!(summary.succeeded());

    let cov = opt.covariance().expect("covariance from the final system");
    assert_relative_eq!(cov.elem(0, 0), 0.25, epsilon = 1e-6);
    assert_relative_eq!(cov.elem(1, 1), 1.0, epsilon = 1e-6);
}

#[test]
fn covariance_after_levenberg_marquardt_is_undamped() {
    // The exported covariance must use the un-damped Hessian, not (H + λI).
    let mut options = Options::default();
    options.exp.final_cov = true;
    options.lm.damping_init = 1.0; // large damping during the solve only

    let mut x = vec2(0.0, 0.0);
    let summary = optimize(&mut x, anisotropic(), &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    let cov = summary.final_cov.as_ref().expect("covariance");
    assert_relative_eq!(cov.elem(0, 0), 0.25, epsilon = 1e-6);
    assert_relative_eq!(cov.elem(1, 1), 1.0, epsilon = 1e-6);
}
