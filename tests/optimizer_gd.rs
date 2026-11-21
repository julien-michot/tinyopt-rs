// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/optimizer_gd.cpp, solver_gd.cpp and unconstrained.cpp:
//! gradient-only accumulation, first step, and convergence.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseVector, Matrix, Vector};
use tinyopt::{Cost, Options, Solver, Static, optimize};

type V1 = Vector<f64, Static<1>>;

fn scalar(data: f64) -> V1 {
    let mut v = <V1 as DenseVector<f64>>::zeros(1);
    v.set_at(0, data);
    v
}

type GradAcc = Box<dyn Fn(&[f64], Option<&mut V1>, Option<&mut Matrix<f64, Static<1>>>) -> Cost>;

/// cost = 0.5 * (x - 2)², grad = x - 2 (Hessian intentionally ignored).
fn grad_only() -> GradAcc {
    Box::new(
        |x: &[f64], grad: Option<&mut V1>, _h: Option<&mut Matrix<f64, Static<1>>>| -> Cost {
            let d = x[0] - 2.0;
            if let Some(g) = grad {
                g.set_at(0, d);
            }
            Cost::new(0.5 * d * d)
        },
    )
}

fn one_iter_options(lr: f32) -> Options {
    let mut options = Options::new(Solver::GradientDescent);
    options.gd.lr = lr;
    options.stop.max_iters = 1;
    options.stop.min_error = 0.0;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 0.0;
    options.stop.min_step_norm2 = 0.0;
    options.stop.min_step_rel_norm2 = 0.0;
    options.stop.min_rerr_dec = 0.0;
    options
}

#[test]
fn first_gradient_descent_step() {
    let mut x = scalar(0.0);
    let summary = optimize(&mut x, grad_only(), &one_iter_options(0.1));
    // x1 = x0 - lr * grad = 0 - 0.1 * (-2) = 0.2
    assert_relative_eq!(x.at(0), 0.2, epsilon = 1e-6);
    assert_eq!(summary.num_iters, 1);
}

#[test]
fn gradient_descent_converges() {
    let mut options = Options::new(Solver::GradientDescent);
    options.gd.lr = 0.5;
    let mut x = scalar(0.0);
    let summary = optimize(&mut x, grad_only(), &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-6);
}

#[test]
fn gradient_descent_dynamic_problem() {
    // 2D quadratic with dynamic-size parameters.
    use tinyopt::Dynamic;
    type Vx = Vector<f64, Dynamic>;
    type Mx = Matrix<f64, Dynamic>;
    let acc = |x: &[f64], grad: Option<&mut Vx>, _h: Option<&mut Mx>| -> Cost {
        let (dx, dy) = (x[0] - 1.0, x[1] + 3.0);
        if let Some(g) = grad {
            g.set_at(0, 2.0 * dx);
            g.set_at(1, 2.0 * dy);
        }
        Cost::new(dx * dx + dy * dy)
    };
    let mut x = <Vx as DenseVector<f64>>::zeros(2);
    x.set_at(0, 0.0);
    x.set_at(1, 0.0);
    let mut options = Options::new(Solver::GradientDescent);
    options.gd.lr = 0.1;
    let summary = optimize(&mut x, acc, &options);
    assert!(summary.succeeded());
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-3);
    assert_relative_eq!(x.at(1), -3.0, epsilon = 1e-3);
}

#[test]
fn scalar_unconstrained_quartic() {
    // Mirrors tests/cpp/unconstrained.cpp: cost = 3y² + y⁴ - 2, y = x - 42.
    let acc = |x: &[f64], grad: Option<&mut V1>, _h: Option<&mut Matrix<f64, Static<1>>>| -> Cost {
        let y = x[0] - 42.0;
        if let Some(g) = grad {
            g.set_at(0, 6.0 * y + 4.0 * y * y * y);
        }
        Cost::new(3.0 * y * y + y * y * y * y - 2.0)
    };
    let mut x = 40.1_f64;

    let mut options = Options::new(Solver::GradientDescent);
    options.gd.lr = 0.01;
    options.stop.max_iters = 1000;
    options.stop.min_error = 0.0;
    options.stop.min_rerr_dec = 0.0;

    let summary = optimize(&mut x, acc, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(summary.converged(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x, 42.0, epsilon = 1e-5);
}
