// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors the dense parts of tests/cpp/diff.cpp: numerical differentiation
//! accumulators and gradient checks.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::diff::{
    Method, create_num_diff_cost_func, create_num_diff_func, estimate_num_jac, num_eval,
};
use tinyopt_rs::{AccumulationFn, Cost, Options, Static, optimize};

type V1 = Vector<f64, Static<1>>;
type M1 = Matrix<f64, Static<1>>;

fn residuals(x: &[f64]) -> Vec<f64> {
    vec![x[0] * x[0] - 2.0]
}

fn cost(x: &[f64]) -> f64 {
    (x[0] * x[0] - 2.0).powi(2)
}

#[test]
fn numerical_jacobian() {
    let x = [1.3];
    let jac = estimate_num_jac(&x, &residuals, Method::Central, 1e-7);
    assert_relative_eq!(jac[0], 2.0 * x[0], epsilon = 1e-6);

    let out = num_eval(&x, &residuals, Method::FastCentral, 1e-7);
    assert_eq!(out.num_residuals, 1);
    assert_eq!(out.num_params, 1);
    assert_relative_eq!(out.residuals[0], x[0] * x[0] - 2.0);
}

#[test]
fn numdiff_accumulator_optimizes() {
    let acc = create_num_diff_func(&residuals, Method::Central, 1e-7);
    let mut x = <V1 as DenseVector<f64>>::zeros(1);
    x.set_at(0, 1.0);

    let summary = optimize(&mut x, acc, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(summary.num_diff_used);
    assert_relative_eq!(x.at(0), 2.0_f64.sqrt(), epsilon = 1e-5);
}

#[test]
fn numdiff_cost_accumulator_gradient() {
    let acc = create_num_diff_cost_func(&cost, Method::Central, 1e-6);
    let mut g = <V1 as DenseVector<f64>>::zeros(1);
    let c = AccumulationFn::<f64, Static<1>>::eval(&acc, &[1.3], Some(&mut g), None);
    assert_relative_eq!(c.cost, cost(&[1.3]));
    // d/dx (x²-2)² = 4x(x²-2)
    assert_relative_eq!(g.at(0), 4.0 * 1.3 * (1.3f64 * 1.3 - 2.0), epsilon = 1e-4);
}

#[test]
fn numdiff_accumulator_builds_hessian() {
    let acc = create_num_diff_func(&residuals, Method::Central, 1e-7);
    let mut g = <V1 as DenseVector<f64>>::zeros(1);
    let mut h = <M1 as DenseMatrix<f64>>::zeros(1, 1);
    let _ = AccumulationFn::<f64, Static<1>>::eval(&acc, &[1.5], Some(&mut g), Some(&mut h));
    assert_relative_eq!(h.elem(0, 0), (2.0 * 1.5f64).powi(2), epsilon = 1e-5);
    assert_relative_eq!(g.at(0), 2.0 * 1.5 * (1.5f64 * 1.5 - 2.0), epsilon = 1e-5);
    let _ = Cost::new(0.0);
}
