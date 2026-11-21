// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/optimizer_cg.cpp.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseVector, Matrix, Vector};
use tinyopt_rs::{Cost, Options, Solver, Static, optimize};

type V1 = Vector<f64, Static<1>>;
type V2 = Vector<f64, Static<2>>;
type M2 = Matrix<f64, Static<2>>;

fn vec2(x: f64, y: f64) -> V2 {
    let mut v = <V2 as DenseVector<f64>>::zeros(2);
    v.set_at(0, x);
    v.set_at(1, y);
    v
}

#[test]
fn first_conjugate_gradient_step_is_steepest_descent() {
    let acc = |x: &[f64], grad: Option<&mut V1>, _h: Option<&mut Matrix<f64, Static<1>>>| -> Cost {
        let r = x[0] - 1.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r);
        }
        Cost::new(r * r)
    };
    let mut x = <V1 as DenseVector<f64>>::zeros(1);
    x.set_at(0, 0.0);

    let mut options = Options::new(Solver::ConjugateGradient);
    options.cg.step_size = 0.1;
    options.stop.max_iters = 1;
    options.stop.min_error = 0.0;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 0.0;
    options.stop.min_step_norm2 = 0.0;
    options.stop.min_step_rel_norm2 = 0.0;
    options.stop.min_rerr_dec = 0.0;

    let summary = optimize(&mut x, acc, &options);
    // direction = -grad = 2; step = 0.1 * 2 = 0.2 (options are f32).
    assert_relative_eq!(x.at(0), 0.2, epsilon = 1e-6);
    assert_eq!(summary.num_iters, 1);
}

#[test]
fn conjugate_gradient_solves_quadratic() {
    let acc = |x: &[f64], grad: Option<&mut V1>, _h: Option<&mut Matrix<f64, Static<1>>>| -> Cost {
        let r = x[0] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r);
        }
        Cost::new(r * r)
    };
    let mut x = <V1 as DenseVector<f64>>::zeros(1);
    x.set_at(0, 8.0);

    let mut options = Options::new(Solver::ConjugateGradient);
    options.cg.step_size = 0.25;
    options.stop.max_iters = 100;

    let summary = optimize(&mut x, acc, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(summary.converged(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-4);
}

#[test]
fn conjugate_gradient_dispatch_two_dimensional() {
    let target = (2.0, -3.0);
    let acc = |x: &[f64], grad: Option<&mut V2>, _h: Option<&mut M2>| -> Cost {
        let (dx, dy) = (x[0] - target.0, x[1] - target.1);
        if let Some(g) = grad {
            g.set_at(0, 2.0 * dx);
            g.set_at(1, 2.0 * dy);
        }
        Cost::new(dx * dx + dy * dy)
    };
    let mut x = vec2(8.0, -8.0);

    let mut options = Options::new(Solver::ConjugateGradient);
    options.cg.step_size = 0.25;
    options.stop.max_iters = 100;

    let summary = optimize(&mut x, acc, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), target.0, epsilon = 1e-4);
    assert_relative_eq!(x.at(1), target.1, epsilon = 1e-4);
}

#[test]
fn conjugate_gradient_dynamic_regression() {
    use tinyopt_rs::Dynamic;
    type Vx = Vector<f64, Dynamic>;
    type Mx = Matrix<f64, Dynamic>;
    let target = [2.0, -3.0];
    let acc = |x: &[f64], mut grad: Option<&mut Vx>, _h: Option<&mut Mx>| -> Cost {
        let mut cost = 0.0;
        for i in 0..x.len() {
            let d = x[i] - target[i];
            cost += d * d;
            if let Some(g) = grad.as_mut() {
                g.set_at(i, 2.0 * d);
            }
        }
        Cost::new(cost)
    };
    let mut x = <Vx as DenseVector<f64>>::zeros(2);
    x.set_at(0, 8.0);
    x.set_at(1, -8.0);

    let mut options = Options::new(Solver::ConjugateGradient);
    options.cg.step_size = 0.25;
    options.stop.max_iters = 100;
    let summary = optimize(&mut x, acc, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), target[0], epsilon = 1e-3);
    assert_relative_eq!(x.at(1), target[1], epsilon = 1e-3);
}
