// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/classif.cpp: softmax values and Jacobians.

use approx::assert_relative_eq;
use tinyopt_rs::diff::{Method, estimate_num_jac};
use tinyopt_rs::losses::{safe_softmax, safe_softmax_with_jac, softmax, softmax_with_jac};

fn num_grad(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    estimate_num_jac(x, &|xx: &[f64]| vec![f(xx)], Method::Central, 1e-6)
}

#[test]
fn softmax_values_sum_to_one() {
    let x = [1.0_f64, 2.0, 3.0, -1.0];
    let p = softmax(&x);
    assert_relative_eq!(p.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    let ps = safe_softmax(&x);
    for (a, b) in p.iter().zip(&ps) {
        assert_relative_eq!(a, b, epsilon = 1e-12);
    }
}

#[test]
fn softmax_jacobian_matches_numeric() {
    let x = [1.0_f64, 2.0, 3.0];
    let (_, jac) = softmax_with_jac(&x);
    let n = x.len();
    for r in 0..n {
        let num = num_grad(|zz| softmax(zz)[r], &x);
        for c in 0..n {
            assert_relative_eq!(jac[r * n + c], num[c], epsilon = 1e-5);
        }
    }
}

#[test]
fn safe_softmax_jacobian_matches_numeric() {
    let x = [0.5_f64, -1.0, 2.0];
    let (_, jac) = safe_softmax_with_jac(&x);
    let n = x.len();
    for r in 0..n {
        let num = num_grad(|zz| safe_softmax(zz)[r], &x);
        for c in 0..n {
            assert_relative_eq!(jac[r * n + c], num[c], epsilon = 1e-5);
        }
    }
}

#[test]
fn scalar_softmax() {
    assert_eq!(softmax(&[3.0_f64]), vec![1.0]);
    assert_eq!(safe_softmax(&[3.0_f64]), vec![1.0]);
}
