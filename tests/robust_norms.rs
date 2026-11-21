// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/robust_norms.cpp (dense parts): values and Jacobian scales
//! of every M-estimator, verified against numerical differentiation.

use approx::assert_relative_eq;
use tinyopt_rs::diff::{Method, estimate_num_jac};
use tinyopt_rs::losses::{
    arctan, arctan_with_scale, blake_zisserman, blake_zisserman_with_scale, cauchy,
    cauchy_with_scale, geman_mcclure, geman_mcclure_with_scale, huber, huber_with_scale,
    squared_l2, truncated, truncated_with_scale, tukey, tukey_with_scale,
};

fn num_grad(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    estimate_num_jac(x, &|xx: &[f64]| vec![f(xx)], Method::Central, 1e-6)
}

type ScalarLoss = fn(f64, f64) -> f64;
type ScalarLossWithScale = fn(f64, f64) -> (f64, f64);

const LOSSES: &[(&str, ScalarLoss, ScalarLossWithScale)] = &[
    ("truncated", truncated, truncated_with_scale),
    ("huber", huber, huber_with_scale),
    ("tukey", tukey, tukey_with_scale),
    ("arctan", arctan, arctan_with_scale),
    ("cauchy", cauchy, cauchy_with_scale),
    ("geman_mcclure", geman_mcclure, geman_mcclure_with_scale),
    (
        "blake_zisserman",
        blake_zisserman,
        blake_zisserman_with_scale,
    ),
];

fn check_scalar(name: &str, loss: ScalarLoss, with_scale: ScalarLossWithScale, th2: f64, n2: f64) {
    let (l, scale) = with_scale(n2, th2);
    assert_relative_eq!(l, loss(n2, th2), epsilon = 1e-12);
    // The returned scale is d loss / d n2.
    let num = num_grad(|x| loss(x[0], th2), &[n2]);
    assert_relative_eq!(scale, num[0], epsilon = 1e-5);
    let _ = name;
}

#[test]
fn scalar_inlier_and_outlier() {
    let th2 = 1.3_f64 * 1.3;
    for (name, loss, with_scale) in LOSSES {
        check_scalar(name, *loss, *with_scale, th2, 0.3);
        check_scalar(name, *loss, *with_scale, th2, 2.3 * 2.3);
        check_scalar(name, *loss, *with_scale, th2, 0.5);
    }
}

#[test]
fn vector_scales_match_numeric_jacobian() {
    let x = [0.1_f64, -0.2, -0.3, 0.4];
    let n2 = squared_l2(&x);
    for (name, loss, with_scale) in LOSSES {
        for th2 in [1.3_f64 * 1.3, 0.03 * 0.03] {
            let (l, scale) = with_scale(n2, th2);
            assert_relative_eq!(l, loss(n2, th2), epsilon = 1e-12);
            // Vector-loss Jacobian: d/dx loss(||x||²) = scale * 2x.
            let num = num_grad(|xx| loss(squared_l2(xx), th2), &x);
            for (i, n) in num.iter().enumerate() {
                assert_relative_eq!(scale * 2.0 * x[i], *n, epsilon = 1e-5, epsilon = 1e-5);
            }
            let _ = name;
        }
    }
}

#[test]
fn inlier_and_outlier_branches() {
    // th2 = 4, th = 2.
    assert_relative_eq!(huber(1.0_f64, 4.0), 1.0);
    let (l, s) = huber_with_scale(1.0, 4.0);
    assert_relative_eq!(l, 1.0);
    assert_relative_eq!(s, 1.0);
    let (l, s) = huber_with_scale(25.0, 4.0);
    assert_relative_eq!(l, 2.0 * 2.0 * 5.0 - 4.0);
    assert_relative_eq!(s, 0.4);

    assert_eq!(truncated(25.0_f64, 4.0), 4.0);
    assert_eq!(truncated_with_scale(25.0, 4.0), (4.0, 0.0));

    assert_eq!(tukey(25.0_f64, 4.0), 4.0);
    assert_eq!(tukey_with_scale(25.0, 4.0), (4.0, 0.0));
}
