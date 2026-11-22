// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Steihaug–Toint trust-region Newton-CG optimizer.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Dynamic, Matrix, Vector};
use tinyopt::{Cost, Options, Solver, StopReason, optimize, optimize_sparse};

type V = Vector<f64, Dynamic>;
type M = Matrix<f64, Dynamic>;

fn trcg_options() -> Options {
    let mut options = Options::new(Solver::SteihaugToint);
    options.log.enable = false;
    options.stop.max_iters = 200;
    options.stop.min_error = 1e-14;
    options.stop.min_grad_max = 1e-9;
    options.stop.min_rerr_dec = 1e-12;
    options
}

fn quadratic(target: Vec<f64>) -> impl Fn(&[f64], Option<&mut V>, Option<&mut M>) -> Cost {
    move |x: &[f64], grad: Option<&mut V>, h: Option<&mut M>| {
        let mut cost = 0.0;
        if let Some(grad) = grad {
            for i in 0..x.len() {
                let r = x[i] - target[i];
                cost += 0.5 * r * r;
                grad.set_at(i, r);
            }
        } else {
            for i in 0..x.len() {
                cost += 0.5 * (x[i] - target[i]).powi(2);
            }
        }
        if let Some(h) = h {
            for i in 0..x.len() {
                h.set_elem(i, i, 1.0);
            }
        }
        Cost::with_num_residuals(cost, x.len() as i32)
    }
}

#[test]
fn quadratic_converges() {
    let target = vec![1.0, -2.0, 3.5];
    let mut x = <V as DenseVector<f64>>::zeros(3);
    let summary = optimize(&mut x, quadratic(target.clone()), &trcg_options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    for (got, want) in (0..3).map(|i| (x.at(i), target[i])) {
        assert_relative_eq!(got, want, epsilon = 1e-8);
    }
}

#[test]
fn rosenbrock_matches_levenberg_marquardt() {
    // Rosenbrock as two residuals.
    let acc = |x: &[f64], grad: Option<&mut V>, h: Option<&mut M>| {
        let (a, b) = (x[0], x[1]);
        let r0 = 10.0 * (b - a * a);
        let r1 = 1.0 - a;
        let mut cost = 0.5 * (r0 * r0 + r1 * r1);
        if let Some(g) = grad {
            g.set_at(0, 10.0 * (b - a * a) * -20.0 * a - (1.0 - a));
            g.set_at(1, 10.0 * (b - a * a) * 10.0);
        }
        if let Some(h) = h {
            // Gauss-Newton approximation.
            let j00 = -20.0 * a;
            let j01 = 10.0;
            let j10 = -1.0;
            h.set_elem(0, 0, j00 * j00 + j10 * j10);
            h.set_elem(0, 1, j00 * j01);
            h.set_elem(1, 0, j00 * j01);
            h.set_elem(1, 1, j01 * j01);
            cost += 0.0;
        }
        Cost::with_num_residuals(cost, 2)
    };

    let mut x = <V as DenseVector<f64>>::zeros(2);
    x.set_at(0, -1.2);
    x.set_at(1, 1.0);
    let summary = optimize(&mut x, acc, &trcg_options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-4);
    assert_relative_eq!(x.at(1), 1.0, epsilon = 1e-4);
}

#[test]
fn sparse_problem_converges() {
    // Sparse chain with a coupling term (sparse Hessian).
    let n = 200;
    let acc =
        |x: &[f64], mut grad: Option<&mut V>, mut h: Option<&mut tinyopt::sparse::SparseMat>| {
            let mut cost = 0.0;
            for (i, value) in x.iter().enumerate() {
                let r = value - 1.0;
                cost += 0.5 * r * r;
                if let Some(g) = grad.as_mut() {
                    g.set_at(i, g.at(i) + r);
                }
                if let Some(h) = h.as_mut() {
                    *h.coeff_ref(i, i) += 1.0;
                }
            }
            for i in 0..n - 1 {
                let r = x[i + 1] - x[i];
                cost += 0.5 * r * r;
                if let Some(g) = grad.as_mut() {
                    g.set_at(i, g.at(i) - r);
                    g.set_at(i + 1, g.at(i + 1) + r);
                }
                if let Some(h) = h.as_mut() {
                    *h.coeff_ref(i, i) += 1.0;
                    *h.coeff_ref(i + 1, i + 1) += 1.0;
                    *h.coeff_ref(i, i + 1) -= 1.0;
                    *h.coeff_ref(i + 1, i) -= 1.0;
                }
            }
            Cost::with_num_residuals(cost, (2 * n) as i32)
        };
    let mut x = vec![0.0; n];
    let summary = optimize_sparse(&mut x, acc, &trcg_options());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-6));
}

#[test]
fn indefinite_curvature_reaches_a_finite_step() {
    // Negative curvature along x1: the CG must stop at the trust boundary
    // instead of producing an unbounded step.
    let acc = |x: &[f64], grad: Option<&mut V>, h: Option<&mut M>| {
        let mut cost = 0.5 * x[0] * x[0] - 0.5 * x[1] * x[1];
        if let Some(g) = grad {
            g.set_at(0, x[0]);
            g.set_at(1, -x[1]);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 1.0);
            h.set_elem(1, 1, -1.0);
        }
        cost += 0.0;
        Cost::with_num_residuals(cost, 2)
    };
    let mut x = <V as DenseVector<f64>>::zeros(2);
    x.set_at(1, 1.0);
    let mut options = trcg_options();
    options.stop.max_iters = 5;
    // The cost is unbounded below; disable the (non-negative) error criterion.
    options.stop.min_error = 0.0;
    let summary = optimize(&mut x, acc, &options);
    assert!(x.iter().all(|v| v.is_finite()));
    assert!(x.at(1).abs() > 1.0, "the boundary step should expand x1");
    assert_ne!(summary.stop_reason, StopReason::SystemHasNanOrInf);
}

#[test]
fn fixed_parameters_stay_fixed() {
    let target = vec![1.0, 2.0];
    let mut x = <V as DenseVector<f64>>::zeros(2);
    let mut optimizer = tinyopt::optimizers::trcg::Optimizer::<V, M>::new(trcg_options());
    optimizer.fix(&[0]);
    let summary = optimizer.optimize(&mut x, quadratic(target));
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 0.0, epsilon = 1e-12);
    assert_relative_eq!(x.at(1), 2.0, epsilon = 1e-8);
}
