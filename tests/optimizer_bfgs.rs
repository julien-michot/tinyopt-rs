// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/optimizer_bfgs.cpp and solver_bfgs.cpp.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::optimizers::{bfgs, lbfgs};
use tinyopt_rs::{Cost, Options, Solver, Static, optimize};

type V2 = Vector<f64, Static<2>>;
type M2 = Matrix<f64, Static<2>>;

fn vec2(x: f64, y: f64) -> V2 {
    let mut v = <V2 as DenseVector<f64>>::zeros(2);
    v.set_at(0, x);
    v.set_at(1, y);
    v
}

fn quadratic() -> impl Fn(&[f64], Option<&mut V2>, Option<&mut M2>) -> Cost {
    |x: &[f64], grad: Option<&mut V2>, _h: Option<&mut M2>| -> Cost {
        let (dx, dy) = (x[0] - 1.0, x[1] - 2.0);
        if let Some(g) = grad {
            g.set_at(0, dx);
            g.set_at(1, dy);
        }
        Cost::new(0.5 * (dx * dx + dy * dy))
    }
}

fn rosenbrock() -> impl Fn(&[f64], Option<&mut V2>, Option<&mut M2>) -> Cost {
    |x: &[f64], grad: Option<&mut V2>, _h: Option<&mut M2>| -> Cost {
        let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
        if let Some(g) = grad {
            g.set_at(0, -2.0 * a - 400.0 * x[0] * b);
            g.set_at(1, 200.0 * b);
        }
        Cost::new(a * a + 100.0 * b * b)
    }
}

#[test]
fn bfgs_first_step_uses_identity_inverse_hessian() {
    let mut x = vec2(0.0, 0.0);
    let mut options = Options::new(Solver::Bfgs);
    options.bfgs.step_size = 1.0;
    options.stop.max_iters = 1;
    options.stop.min_error = 0.0;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 0.0;
    options.stop.min_step_norm2 = 0.0;
    options.stop.min_step_rel_norm2 = 0.0;
    options.stop.min_rerr_dec = 0.0;

    let summary = optimize(&mut x, quadratic(), &options);
    // dx = -grad = (1, 2)
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-12);
    assert_relative_eq!(x.at(1), 2.0, epsilon = 1e-12);
    assert_eq!(summary.num_iters, 1);
}

#[test]
fn lbfgs_first_step_uses_identity_inverse_hessian() {
    let mut x = vec2(0.0, 0.0);
    let mut options = Options::new(Solver::Lbfgs);
    options.lbfgs.step_size = 1.0;
    options.stop.max_iters = 1;
    options.stop.min_error = 0.0;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 0.0;
    options.stop.min_step_norm2 = 0.0;
    options.stop.min_step_rel_norm2 = 0.0;
    options.stop.min_rerr_dec = 0.0;

    let _ = optimize(&mut x, quadratic(), &options);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-12);
    assert_relative_eq!(x.at(1), 2.0, epsilon = 1e-12);
}

#[test]
fn bfgs_converges_on_quadratic() {
    let mut x = vec2(0.0, 0.0);
    let summary = optimize(&mut x, quadratic(), &Options::new(Solver::Bfgs));
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(x.at(1), 2.0, epsilon = 1e-6);
}

#[test]
fn lbfgs_converges_on_quadratic() {
    let mut x = vec2(0.0, 0.0);
    let summary = optimize(&mut x, quadratic(), &Options::new(Solver::Lbfgs));
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(x.at(1), 2.0, epsilon = 1e-6);
}

#[test]
fn bfgs_and_lbfgs_on_rosenbrock() {
    for solver in [Solver::Bfgs, Solver::Lbfgs] {
        let mut options = Options::new(solver);
        options.stop.max_iters = 5000;
        options.stop.max_consec_failures = 20;
        options.stop.min_rerr_dec = 0.0;
        options.bfgs.step_size = 0.005;
        options.lbfgs.step_size = 0.005;
        let mut x = vec2(-1.2, 1.0);
        let summary = optimize(&mut x, rosenbrock(), &options);
        assert!(summary.succeeded(), "{solver:?}: {:?}", summary.stop_reason);
        assert_relative_eq!(x.at(0), 1.0, epsilon = 5e-2);
        assert_relative_eq!(x.at(1), 1.0, epsilon = 5e-2);
    }
}

#[test]
fn bfgs_exposes_inverse_hessian_as_covariance() {
    let mut opt = bfgs::Optimizer::<V2>::new(Options::new(Solver::Bfgs));
    let mut x = vec2(0.0, 0.0);
    let summary = opt.optimize(&mut x, quadratic());
    assert!(summary.succeeded());

    let covariance = opt.covariance().expect("BFGS inverse Hessian");
    // For an isotropic quadratic the inverse Hessian is the identity.
    assert_relative_eq!(covariance.elem(0, 0), 1.0, epsilon = 1e-2);
    assert_relative_eq!(covariance.elem(1, 1), 1.0, epsilon = 1e-2);
    assert_relative_eq!(covariance.elem(0, 1), 0.0, epsilon = 1e-2);
}

#[test]
fn lbfgs_history_size_is_bounded() {
    let mut options = Options::new(Solver::Lbfgs);
    options.lbfgs.history_size = 3;
    let mut opt = lbfgs::Optimizer::<V2>::new(options);
    let mut x = vec2(-1.2, 1.0);
    let summary = opt.optimize(&mut x, rosenbrock());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
}
