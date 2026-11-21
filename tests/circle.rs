// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "autodiff")]

//! Mirrors tests/cpp/circle.cpp: fit a 2D circle (center + radius) to noisy
//! points using autodiff residuals and Levenberg-Marquardt.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseVector, Vector};
use tinyopt::diff::{DualNum, ResidualsFn, residuals_and_jacobian};
use tinyopt::{Options, Static, optimize_residuals};

type V3 = Vector<f64, Static<3>>;

struct CircleData {
    points: Vec<(f64, f64)>,
}

impl ResidualsFn for CircleData {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        let (cx, cy, r) = (x[0].clone(), x[1].clone(), x[2].clone());
        self.points
            .iter()
            .map(|&(px, py)| {
                let dx = cx.clone() - px;
                let dy = cy.clone() - py;
                (dx.clone() * dx + dy.clone() * dy).sqrt() - r.clone()
            })
            .collect()
    }
}

fn make_circle(center: (f64, f64), radius: f64, n: usize) -> CircleData {
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let noise = 0.01 * ((i as f64 * 1.7).sin());
        points.push((
            center.0 + (radius + noise) * angle.cos(),
            center.1 + (radius + noise) * angle.sin(),
        ));
    }
    CircleData { points }
}

#[test]
fn fit_circle_from_noisy_points() {
    let true_center = (0.3, -0.2);
    let true_radius = 2.0;
    let data = make_circle(true_center, true_radius, 20);

    let mut x = <V3 as DenseVector<f64>>::zeros(3);
    x.set_at(0, 0.0);
    x.set_at(1, 0.0);
    x.set_at(2, 1.0);

    let mut options = Options::default();
    options.lm.damping_init = 1e-2;

    let summary = optimize_residuals(&mut x, &data, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(summary.converged(), "{:?}", summary.stop_reason);
    assert_eq!(summary.num_residuals, 20);

    assert_relative_eq!(x.at(0), true_center.0, epsilon = 2e-2);
    assert_relative_eq!(x.at(1), true_center.1, epsilon = 2e-2);
    assert_relative_eq!(x.at(2), true_radius, epsilon = 2e-2);
}

#[test]
fn circle_jacobian_is_exact() {
    let data = make_circle((0.0, 0.0), 1.0, 3);
    let x = [0.1, -0.05, 1.2];
    let out = residuals_and_jacobian(&data, &x);
    assert_eq!(out.num_residuals, 3);
    assert_eq!(out.num_params, 3);

    let (cx, cy, r) = (x[0], x[1], x[2]);
    for (i, &(px, py)) in data.points.iter().enumerate() {
        let (dx, dy) = (cx - px, cy - py);
        let d = (dx * dx + dy * dy).sqrt();
        let row = &out.jacobian[i * 3..(i + 1) * 3];
        assert_relative_eq!(row[0], dx / d, epsilon = 1e-6);
        assert_relative_eq!(row[1], dy / d, epsilon = 1e-6);
        assert_relative_eq!(row[2], -1.0, epsilon = 1e-12);
        assert_relative_eq!(out.residuals[i], d - r, epsilon = 1e-10);
    }
}
