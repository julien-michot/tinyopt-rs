// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/distances.cpp (scalar and Vec4 parts).

use approx::assert_relative_eq;
use tinyopt::diff::{Method, estimate_num_jac};
use tinyopt::distances::{
    cosine, cosine_with_jac, euclidean, euclidean_with_jac, l1_distance_with_jac, linf_distance,
    linf_distance_with_jac, maha_norm_cov_distance, maha_norm_cov_distance_with_jac,
    maha_norm_vars_distance, manhattan,
};

fn num_grad(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    estimate_num_jac(x, &|xx: &[f64]| vec![f(xx)], Method::Central, 1e-6)
}

#[test]
fn l2_distance_and_gradients() {
    let a = [0.1_f64, -0.2, -0.3, 0.4];
    let b = [0.5_f64, 0.1, 0.0, -0.7];

    let (d, ja, jb) = euclidean_with_jac(&a, &b);
    assert_relative_eq!(d, euclidean(&a, &b));

    let na = num_grad(|aa| euclidean(aa, &b), &a);
    let nb = num_grad(|bb| euclidean(&a, bb), &b);
    for i in 0..a.len() {
        assert_relative_eq!(ja[i], na[i], epsilon = 1e-5);
        assert_relative_eq!(jb[i], nb[i], epsilon = 1e-5);
    }
}

#[test]
fn l1_distance_and_gradients() {
    let a = [0.1_f64, -0.2, -0.3, 0.4];
    let b = [0.5_f64, 0.1, 0.0, -0.7];
    let (d, ja, jb) = l1_distance_with_jac(&a, &b);
    assert_relative_eq!(d, manhattan(&a, &b));
    let na = num_grad(|aa| manhattan(aa, &b), &a);
    let nb = num_grad(|bb| manhattan(&a, bb), &b);
    for i in 0..a.len() {
        assert_relative_eq!(ja[i], na[i], epsilon = 1e-5);
        assert_relative_eq!(jb[i], nb[i], epsilon = 1e-5);
    }
}

#[test]
fn linf_distance_and_gradients() {
    let a = [0.1_f64, -0.2, -0.3, 0.4];
    let b = [0.5_f64, 0.1, 0.0, -0.7];
    let (d, ja, jb) = linf_distance_with_jac(&a, &b);
    assert_relative_eq!(d, linf_distance(&a, &b));
    let na = num_grad(|aa| linf_distance(aa, &b), &a);
    let nb = num_grad(|bb| linf_distance(&a, bb), &b);
    for i in 0..a.len() {
        assert_relative_eq!(ja[i], na[i], epsilon = 1e-5);
        assert_relative_eq!(jb[i], nb[i], epsilon = 1e-5);
    }
}

#[test]
fn cosine_similarity_and_gradients() {
    let a = [0.1_f64, -0.2, -0.3, 0.4];
    let b = [0.5_f64, 0.1, 0.0, -0.7];
    let (d, ja, jb) = cosine_with_jac(&a, &b);
    assert_relative_eq!(d, cosine(&a, &b));
    let na = num_grad(|aa| cosine(aa, &b), &a);
    let nb = num_grad(|bb| cosine(&a, bb), &b);
    for i in 0..a.len() {
        assert_relative_eq!(ja[i], na[i], epsilon = 1e-5);
        assert_relative_eq!(jb[i], nb[i], epsilon = 1e-5);
    }
}

#[test]
fn scalar_distances() {
    let a = [0.8_f64];
    let b = [0.3];
    let (d, _, jb) = l1_distance_with_jac(&a, &b);
    assert_relative_eq!(d, 0.5);
    // d|a-b|/db = -1.
    assert_relative_eq!(jb[0], -1.0);
    let (d, _, _) = euclidean_with_jac(&a, &b);
    assert_relative_eq!(d, 0.5);
}

#[test]
fn mahalanobis_distances() {
    let a = [2.0_f64, 4.0];
    let b = [0.0, 0.0];
    let vars = [2.0_f64, 8.0];
    let d = maha_norm_vars_distance(&a, &b, &vars);
    assert_relative_eq!(d, (4.0_f64 / 2.0 + 16.0 / 8.0).sqrt());

    // Full covariance (column-major, diagonal).
    let cov = [2.0_f64, 0.0, 0.0, 8.0];
    let dc = maha_norm_cov_distance(&a, &b, &cov, 2).unwrap();
    assert_relative_eq!(dc, d);

    let (dc2, ja, jb) = maha_norm_cov_distance_with_jac(&a, &b, &cov, 2).unwrap();
    assert_relative_eq!(dc2, dc);
    let na = num_grad(|aa| maha_norm_cov_distance(aa, &b, &cov, 2).unwrap(), &a);
    for i in 0..a.len() {
        assert_relative_eq!(ja[i], na[i], epsilon = 1e-5);
        assert_relative_eq!(jb[i], -na[i], epsilon = 1e-5);
    }
}
