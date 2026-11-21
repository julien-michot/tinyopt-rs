// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "autodiff")]

//! Mirrors tests/cpp/diff.cpp (dense parts): autodiff evaluation helpers and
//! agreement with numerical differentiation.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::diff::{
    DualNum, Method, ResidualsFn, calculate_jac, estimate_num_jac, eval, residuals_and_jacobian,
};
use tinyopt_rs::{AccumulationFn, Dynamic, Static};

/// r(x) = [x0*x1 - 3, x0² + x1]
struct Mixed;

impl ResidualsFn for Mixed {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        let (x0, x1) = (x[0].clone(), x[1].clone());
        vec![x0.clone() * x1.clone() - 3.0, x0.clone() * x0 + x1.clone()]
    }
}

fn plain(x: &[f64]) -> Vec<f64> {
    vec![x[0] * x[1] - 3.0, x[0] * x[0] + x[1]]
}

#[test]
fn autodiff_matches_numerical_jacobian() {
    let x = [1.2, -0.7];
    let ad = residuals_and_jacobian(&Mixed, &x);
    let num = estimate_num_jac(&x, &plain, Method::Central, 1e-7);
    assert_eq!(ad.jacobian.len(), num.len());
    for (a, b) in ad.jacobian.iter().zip(&num) {
        assert_relative_eq!(a, b, epsilon = 1e-6);
    }
    for (a, b) in ad.residuals.iter().zip(plain(&x).iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-12);
    }
}

#[test]
fn eval_and_calculate_jac_helpers() {
    let x = [0.5, 2.0];
    let out = eval(&Mixed, &x);
    assert_eq!(out.num_residuals, 2);
    assert_eq!(out.num_params, 2);
    let jac = calculate_jac(&Mixed, &x);
    assert_eq!(jac, out.jacobian);
}

#[test]
fn accumulation_static_and_dynamic_agree() {
    type V2 = Vector<f64, Static<2>>;
    type M2 = Matrix<f64, Static<2>>;
    type Vx = Vector<f64, Dynamic>;
    type Mx = Matrix<f64, Dynamic>;

    let x = [1.2, -0.7];
    let acc = tinyopt_rs::diff::ad_accumulation(&Mixed);

    let mut gs = <V2 as DenseVector<f64>>::zeros(2);
    let mut hs = <M2 as tinyopt_rs::backend::DenseMatrix<f64>>::zeros(2, 2);
    let cs = AccumulationFn::<f64, Static<2>>::eval(&acc, &x, Some(&mut gs), Some(&mut hs));

    let mut gd = <Vx as DenseVector<f64>>::zeros(2);
    let mut hd = <Mx as tinyopt_rs::backend::DenseMatrix<f64>>::zeros(2, 2);
    let cd = AccumulationFn::<f64, Dynamic>::eval(&acc, &x, Some(&mut gd), Some(&mut hd));

    assert_eq!(cs.cost, cd.cost);
    for i in 0..2 {
        assert_eq!(gs.at(i), gd.at(i));
        for j in 0..2 {
            assert_eq!(hs.elem(i, j), hd.elem(i, j));
        }
    }
}

#[test]
fn accumulation_builds_jtj() {
    let x = [1.2, -0.7];
    let out = residuals_and_jacobian(&Mixed, &x);
    let acc = tinyopt_rs::diff::ad_accumulation(&Mixed);
    type V2 = Vector<f64, Static<2>>;
    type M2 = Matrix<f64, Static<2>>;
    let mut g = <V2 as DenseVector<f64>>::zeros(2);
    let mut h = <M2 as tinyopt_rs::backend::DenseMatrix<f64>>::zeros(2, 2);
    let _ = AccumulationFn::<f64, Static<2>>::eval(&acc, &x, Some(&mut g), Some(&mut h));

    for i in 0..2 {
        let expected: f64 = (0..2)
            .map(|k| out.jacobian[k * 2 + i] * out.residuals[k])
            .sum();
        assert_relative_eq!(g.at(i), expected, epsilon = 1e-12);
        for j in 0..2 {
            let expected_h: f64 = (0..2)
                .map(|k| out.jacobian[k * 2 + i] * out.jacobian[k * 2 + j])
                .sum();
            assert_relative_eq!(h.elem(i, j), expected_h, epsilon = 1e-12);
        }
    }
}
