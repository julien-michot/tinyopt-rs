// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/norms.cpp: values and Jacobians of classical norms.

use approx::assert_relative_eq;
use tinyopt_rs::diff::{Method, estimate_num_jac};
use tinyopt_rs::losses::{
    l1, l1_with_jac, l2, l2_with_jac, linf, linf_with_jac, squared_l2, squared_l2_with_jac,
};

fn num_grad(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    estimate_num_jac(x, &|xx: &[f64]| vec![f(xx)], Method::Central, 1e-6)
}

fn sample() -> [f64; 4] {
    [0.1, -0.2, -0.3, 0.4]
}

#[test]
fn values() {
    let x = sample();
    assert_relative_eq!(squared_l2(&x), x.iter().map(|v| v * v).sum::<f64>());
    assert_relative_eq!(l2(&x), (x.iter().map(|v| v * v).sum::<f64>()).sqrt());
    assert_relative_eq!(l1(&x), x.iter().map(|v| v.abs()).sum::<f64>());
    assert_relative_eq!(linf(&x), 0.4);
}

#[test]
fn squared_l2_jacobian_matches_numeric() {
    let x = sample();
    let (_, jac) = squared_l2_with_jac(&x);
    let num = num_grad(squared_l2::<f64>, &x);
    for (a, b) in jac.iter().zip(&num) {
        assert_relative_eq!(a, b, epsilon = 1e-5);
    }
}

#[test]
fn l2_jacobian_matches_numeric() {
    let x = sample();
    let (_, jac) = l2_with_jac(&x);
    let num = num_grad(l2::<f64>, &x);
    for (a, b) in jac.iter().zip(&num) {
        assert_relative_eq!(a, b, epsilon = 1e-5);
    }
}

#[test]
fn l1_jacobian_matches_numeric() {
    let x = sample();
    let (_, jac) = l1_with_jac(&x);
    let num = num_grad(l1::<f64>, &x);
    for (a, b) in jac.iter().zip(&num) {
        assert_relative_eq!(a, b, epsilon = 1e-5);
    }
}

#[test]
fn linf_jacobian_matches_numeric() {
    let x = sample();
    let (_, jac) = linf_with_jac(&x);
    let num = num_grad(linf::<f64>, &x);
    for (a, b) in jac.iter().zip(&num) {
        assert_relative_eq!(a, b, epsilon = 1e-5);
    }
}

#[test]
fn scalar_norms() {
    let x = [0.8];
    assert_relative_eq!(squared_l2(&x), 0.64);
    let (v, j) = squared_l2_with_jac(&x);
    assert_relative_eq!(v, 0.64);
    assert_relative_eq!(j[0], 1.6);
    let (_, j) = l2_with_jac(&x);
    assert_relative_eq!(j[0], 1.0);
    let (_, j) = l1_with_jac(&x);
    assert_relative_eq!(j[0], 1.0);
}
