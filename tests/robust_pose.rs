// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/robust_pose.cpp: robust estimation with outliers.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Static, Vector};
use tinyopt::losses::huber_weight;
use tinyopt::{Cost, Options, optimize};

type V3 = Vector<f64, Static<3>>;
type M3 = Matrix<f64, Static<3>>;

/// Ground-truth planar transform: rotation + translation.
const TRUTH: [f64; 3] = [0.35, 1.2, -0.8];
const DELTA: f64 = 0.15;

fn truth_transform(px: f64, py: f64) -> (f64, f64) {
    let (c, s) = (TRUTH[0].cos(), TRUTH[0].sin());
    (c * px - s * py + TRUTH[1], s * px + c * py + TRUTH[2])
}

/// Correspondences: 70% inliers, 30% outliers with a fixed offset.
fn correspondences() -> Vec<((f64, f64), (f64, f64))> {
    let worlds = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (2.0, 0.5),
        (-0.5, 2.0),
    ];
    worlds
        .iter()
        .enumerate()
        .map(|(i, &(px, py))| {
            let (qx, qy) = truth_transform(px, py);
            let outlier = i % 10 >= 7;
            let offset = if outlier { (2.5, -1.8) } else { (0.0, 0.0) };
            ((px, py), (qx + offset.0, qy + offset.1))
        })
        .collect()
}

fn accumulator(robust: bool) -> impl Fn(&[f64], Option<&mut V3>, Option<&mut M3>) -> Cost {
    move |x: &[f64], grad: Option<&mut V3>, h: Option<&mut M3>| -> Cost {
        let (theta, tx, ty) = (x[0], x[1], x[2]);
        let (c, s) = (theta.cos(), theta.sin());
        let mut cost = 0.0;
        let mut g = grad;
        let mut h = h;
        for ((px, py), (qx, qy)) in correspondences() {
            let (cw, sw) = (c, s);
            let rx = cw * px - sw * py + tx - qx;
            let ry = sw * px + cw * py + ty - qy;
            let r2 = rx * rx + ry * ry;
            let w = if robust { huber_weight(r2, DELTA) } else { 1.0 };
            cost += w * r2;
            // Jacobian rows: d(theta) = (-s*px - c*py, c*px - s*py), d(tx) = 1, d(ty) = 1.
            let jx = -sw * px - cw * py;
            let jy = cw * px - sw * py;
            if let Some(g) = g.as_mut() {
                g.set_at(0, g.at(0) + w * (rx * jx + ry * jy));
                g.set_at(1, g.at(1) + w * rx);
                g.set_at(2, g.at(2) + w * ry);
            }
            if let Some(h) = h.as_mut() {
                h.set_elem(0, 0, h.elem(0, 0) + w * (jx * jx + jy * jy));
                h.set_elem(0, 1, h.elem(0, 1) + w * jx);
                h.set_elem(1, 0, h.elem(1, 0) + w * jx);
                h.set_elem(0, 2, h.elem(0, 2) + w * jy);
                h.set_elem(2, 0, h.elem(2, 0) + w * jy);
                h.set_elem(1, 1, h.elem(1, 1) + w);
                h.set_elem(2, 2, h.elem(2, 2) + w);
            }
        }
        Cost::with_num_residuals(cost, 2 * correspondences().len() as i32)
    }
}

fn estimate(robust: bool) -> [f64; 3] {
    let mut options = Options::default();
    options.stop.max_iters = 100;
    options.stop.min_grad_norm2 = 1e-16;
    options.stop.min_grad_max = 0.0;
    options.stop.min_rerr_dec = 0.0;
    let mut x = <V3 as DenseVector<f64>>::zeros(3);
    let summary = optimize(&mut x, accumulator(robust), &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    [x.at(0), x.at(1), x.at(2)]
}

#[test]
fn huber_rejects_outliers() {
    let naive = estimate(false);
    let robust = estimate(true);
    let err = |e: [f64; 3]| {
        (e[0] - TRUTH[0]).powi(2) + (e[1] - TRUTH[1]).powi(2) + (e[2] - TRUTH[2]).powi(2)
    };
    assert!(
        err(robust) < 0.5 * err(naive),
        "robust {robust:?} (err² {}) should beat L2 {naive:?} (err² {})",
        err(robust),
        err(naive)
    );
    for i in 0..3 {
        assert_relative_eq!(robust[i], TRUTH[i], epsilon = 0.05);
    }
}

#[test]
fn huber_weights_cap_outliers() {
    assert_relative_eq!(huber_weight(0.01, DELTA), 1.0, epsilon = 1e-12);
    let w = huber_weight(1.0, DELTA);
    assert!(w < 0.2, "outlier weight = {w}");
}
