// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/optimizer_dl.cpp and solver_dl.cpp: trust-region step and
//! Rosenbrock convergence.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::{Cost, Options, Solver, Static, optimize};

type V1 = Vector<f64, Static<1>>;
type M1 = Matrix<f64, Static<1>>;
type V2 = Vector<f64, Static<2>>;
type M2 = Matrix<f64, Static<2>>;

fn scalar(data: f64) -> V1 {
    let mut v = <V1 as DenseVector<f64>>::zeros(1);
    v.set_at(0, data);
    v
}

fn vec2(x: f64, y: f64) -> V2 {
    let mut v = <V2 as DenseVector<f64>>::zeros(2);
    v.set_at(0, x);
    v.set_at(1, y);
    v
}

/// NLLS cost = r² with r = x - 2, J = 1 -> grad = r, H = 1.
fn shifted_quadratic() -> impl Fn(&[f64], Option<&mut V1>, Option<&mut M1>) -> Cost {
    |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
        let r = x[0] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, r);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 1.0);
        }
        Cost::new(r * r)
    }
}

/// Rosenbrock as NLLS: r0 = 1 - x, r1 = 10 (y - x²), grad = Jᵀr, H = JᵀJ.
fn rosenbrock() -> impl Fn(&[f64], Option<&mut V2>, Option<&mut M2>) -> Cost {
    |x: &[f64], grad: Option<&mut V2>, h: Option<&mut M2>| -> Cost {
        let (r0, r1) = (1.0 - x[0], 10.0 * (x[1] - x[0] * x[0]));
        let j = [[-1.0, 0.0], [-20.0 * x[0], 10.0]];
        if let Some(g) = grad {
            g.set_at(0, j[0][0] * r0 + j[1][0] * r1);
            g.set_at(1, j[0][1] * r0 + j[1][1] * r1);
        }
        if let Some(h) = h {
            for i in 0..2 {
                for k in 0..2 {
                    h.set_elem(i, k, j[0][i] * j[0][k] + j[1][i] * j[1][k]);
                }
            }
        }
        Cost::with_num_residuals(r0 * r0 + r1 * r1, 2)
    }
}

#[test]
fn dogleg_step_hits_trust_region_boundary() {
    let mut x = scalar(0.0);
    let mut options = Options::new(Solver::DogLeg);
    options.dl.radius_init = 1.0;
    options.stop.max_iters = 1;
    options.stop.min_error = 0.0;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 0.0;
    options.stop.min_step_norm2 = 0.0;
    options.stop.min_step_rel_norm2 = 0.0;
    options.stop.min_rerr_dec = 0.0;

    let _ = optimize(&mut x, shifted_quadratic(), &options);
    // The unconstrained GN step (2.0) overshoots the radius (1.0).
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-10);
}

#[test]
fn dogleg_direction_is_a_descent() {
    let mut x = scalar(0.0);
    let mut options = Options::new(Solver::DogLeg);
    options.stop.max_iters = 1;
    options.stop.min_error = 0.0;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 0.0;
    options.stop.min_step_norm2 = 0.0;
    options.stop.min_step_rel_norm2 = 0.0;
    options.stop.min_rerr_dec = 0.0;

    let _ = optimize(&mut x, shifted_quadratic(), &options);
    // Moving toward x = 2 decreases the cost.
    assert!(x.at(0) > 0.0);
}

#[test]
fn dogleg_rosenbrock() {
    let mut x = vec2(-1.2, 1.0);
    let mut options = Options::new(Solver::DogLeg);
    options.stop.max_iters = 500;
    let summary = optimize(&mut x, rosenbrock(), &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 5e-2);
    assert_relative_eq!(x.at(1), 1.0, epsilon = 5e-2);
}

#[test]
fn dogleg_radius_controls_via_optimizer_object() {
    let mut opt = tinyopt_rs::optimizers::dl::Optimizer::<V1>::new(Options::new(Solver::DogLeg));
    let mut x = scalar(0.0);
    let summary = opt.optimize(&mut x, shifted_quadratic());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(opt.radius() > 0.0);
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-8);
}
