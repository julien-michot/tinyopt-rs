// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/mahalanobis.cpp (dense parts).

use approx::assert_relative_eq;
use tinyopt_rs::diff::{Method, estimate_num_jac};
use tinyopt_rs::losses::{
    maha_norm_cov, maha_norm_cov_with_jac, maha_norm_scalar, maha_norm_vars, maha_squared_norm_cov,
    maha_squared_norm_scalar, maha_squared_norm_vars, maha_whitened_cov, maha_whitened_info_u,
    maha_whitened_scalar, maha_whitened_vars,
};

fn num_grad(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    estimate_num_jac(x, &|xx: &[f64]| vec![f(xx)], Method::Central, 1e-6)
}

#[test]
fn scalar_variance() {
    let x = 2.0_f64;
    let var = 4.0;
    assert_relative_eq!(maha_squared_norm_scalar(x, var), 1.0);
    assert_relative_eq!(maha_norm_scalar(x, var), 1.0);
    assert_relative_eq!(maha_norm_scalar(3.0, 9.0), 1.0);
}

#[test]
fn variances() {
    let x = [1.0_f64, 2.0, -3.0];
    let vars = [1.0_f64, 4.0, 9.0];
    assert_relative_eq!(maha_squared_norm_vars(&x, &vars), 1.0 + 1.0 + 1.0);
    assert_relative_eq!(maha_norm_vars(&x, &vars), 3.0_f64.sqrt());
}

#[test]
fn covariance_and_gradient() {
    // Symmetric positive definite covariance (column-major).
    let cov = [2.0_f64, 0.5, 0.5, 3.0];
    let x = [1.0_f64, -2.0];
    let n2 = maha_squared_norm_cov(&x, &cov, 2).unwrap();
    let norm = maha_norm_cov(&x, &cov, 2).unwrap();
    assert_relative_eq!(norm, n2.sqrt());

    let (norm2, grad) = maha_norm_cov_with_jac(&x, &cov, 2).unwrap();
    assert_relative_eq!(norm2, norm);
    let num = num_grad(|xx| maha_norm_cov(xx, &cov, 2).unwrap(), &x);
    for i in 0..x.len() {
        assert_relative_eq!(grad[i], num[i], epsilon = 1e-5);
    }
}

#[test]
fn whitening() {
    let res = [2.0_f64, 4.0, -6.0];
    let stdev = 2.0;
    assert_eq!(maha_whitened_scalar(&res, stdev), vec![1.0, 2.0, -3.0]);

    let stdevs = [2.0_f64, 4.0, 3.0];
    assert_eq!(maha_whitened_vars(&res, &stdevs), vec![1.0, 1.0, -2.0]);

    // Diagonal covariance, L = diag(2, 4).
    let cov = [4.0_f64, 0.0, 0.0, 16.0];
    let z = maha_whitened_cov(&res[..2], &cov, 2).unwrap();
    assert_relative_eq!(z[0], 1.0);
    assert_relative_eq!(z[1], 1.0);

    // Information factor U, z = U r.
    let u = [0.5_f64, 0.0, 0.0, 0.25];
    let z = maha_whitened_info_u(&res[..2], &u, 2);
    assert_relative_eq!(z[0], 1.0);
    assert_relative_eq!(z[1], 1.0);
}
