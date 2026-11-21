// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/activations.cpp.

use approx::assert_relative_eq;
use tinyopt_rs::diff::{Method, estimate_num_jac};
use tinyopt_rs::losses::{
    leaky_relu, leaky_relu_vec, leaky_relu_vec_with_jac, leaky_relu_with_jac, relu, relu_vec,
    relu_vec_with_jac, relu_with_jac, sigmoid, sigmoid_vec_with_jac, sigmoid_with_jac, tanh,
    tanh_vec_with_jac, tanh_with_jac,
};

fn num_grad(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    estimate_num_jac(x, &|xx: &[f64]| vec![f(xx)], Method::Central, 1e-6)
}

#[test]
fn scalar_derivatives() {
    let x = 0.7_f64;
    let (s, d) = sigmoid_with_jac(x);
    assert_relative_eq!(s, sigmoid(x));
    assert_relative_eq!(d, num_grad(|v| sigmoid(v[0]), &[x])[0], epsilon = 1e-5);

    let (t, d) = tanh_with_jac(x);
    assert_relative_eq!(t, tanh(x));
    assert_relative_eq!(d, num_grad(|v| tanh(v[0]), &[x])[0], epsilon = 1e-5);

    let (r, d) = relu_with_jac(x);
    assert_relative_eq!(r, relu(x));
    assert_relative_eq!(d, 1.0);

    let (l, d) = leaky_relu_with_jac(-0.7, 0.1);
    assert_relative_eq!(l, leaky_relu(-0.7, 0.1));
    assert_relative_eq!(d, 0.1);
}

#[test]
fn vector_derivatives_match_numeric() {
    let x = [0.7_f64, -0.3, 1.2];

    let (values, jac) = sigmoid_vec_with_jac(&x);
    let num = num_grad(|v| sigmoid(v[0]), &x); // per-coordinate check below
    for (i, v) in values.iter().enumerate() {
        assert_relative_eq!(*v, sigmoid(x[i]));
    }
    for i in 0..x.len() {
        assert_relative_eq!(
            jac[i],
            num_grad(|v| sigmoid(v[0]), &[x[i]])[0],
            epsilon = 1e-5
        );
    }
    let _ = num;

    let (values, jac) = tanh_vec_with_jac(&x);
    for i in 0..x.len() {
        assert_relative_eq!(values[i], tanh(x[i]));
        assert_relative_eq!(jac[i], num_grad(|v| tanh(v[0]), &[x[i]])[0], epsilon = 1e-5);
    }

    let (values, _) = relu_vec_with_jac(&x);
    assert_eq!(values, relu_vec(&x));

    let (values, jac) = leaky_relu_vec_with_jac(&x, 0.05);
    assert_eq!(values, leaky_relu_vec(&x, 0.05));
    for (i, d) in jac.iter().enumerate() {
        assert_eq!(*d, if x[i] > 0.0 { 1.0 } else { 0.05 });
    }
}

#[test]
fn relu_at_zero() {
    assert_eq!(relu(0.0_f64), 0.0);
    assert_eq!(relu_with_jac(0.0_f64), (0.0, 0.0));
    assert_eq!(leaky_relu(0.0_f64, 0.1), 0.0);
}
