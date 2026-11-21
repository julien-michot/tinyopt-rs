// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/params_wrapper.cpp: wrapping a manifold-like parameter.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::{Cost, Options, Params, ParamsWrapper, Static};

type V3 = Vector<f64, Static<3>>;
type M3 = Matrix<f64, Static<3>>;

/// A 3-DoF pose-like parameter with a trivial tangent retraction.
#[derive(Clone, Debug)]
struct Pose2 {
    data: [f64; 3], // x, y, theta
}

impl Pose2 {
    fn new(x: f64, y: f64, theta: f64) -> Self {
        Self {
            data: [x, y, theta],
        }
    }
}

impl Params for Pose2 {
    type Scalar = f64;
    type Dim = Static<3>;

    fn dims(&self) -> usize {
        3
    }

    fn as_slice(&self) -> &[f64] {
        &self.data
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        // Tangent-space retraction: translation add, angle wrapped.
        self.data[0] += delta[0];
        self.data[1] += delta[1];
        self.data[2] = wrap_angle(self.data[2] + delta[2]);
    }

    fn norm2(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum()
    }
}

fn wrap_angle(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut w = a % two_pi;
    if w > std::f64::consts::PI {
        w -= two_pi;
    } else if w < -std::f64::consts::PI {
        w += two_pi;
    }
    w
}

fn residuals() -> impl Fn(&[f64], Option<&mut V3>, Option<&mut M3>) -> Cost {
    |x: &[f64], mut grad: Option<&mut V3>, mut h: Option<&mut M3>| -> Cost {
        let target = [1.0, -2.0, 0.5];
        let mut cost = 0.0;
        for i in 0..3 {
            let r = x[i] - target[i];
            cost += r * r;
            if let Some(g) = grad.as_mut() {
                g.set_at(i, 2.0 * r);
            }
            if let Some(h) = h.as_mut() {
                h.set_elem(i, i, 2.0);
            }
        }
        Cost::with_num_residuals(cost, 3)
    }
}

#[test]
fn wrapper_delegates_params_trait() {
    let mut wrapped = ParamsWrapper::new(Pose2::new(0.0, 0.0, 0.0));
    assert_eq!(Params::dims(&wrapped), 3);
    Params::plus_eq(&mut wrapped, &[1.0, 2.0, 3.0]);
    assert_eq!(wrapped.value.data, [1.0, 2.0, 3.0]);
    // Deref works.
    assert_eq!(wrapped.data[0], 1.0);
}

#[test]
fn optimize_wrapped_manifold() {
    let mut wrapped = ParamsWrapper::new(Pose2::new(0.0, 0.0, 0.0));
    let summary = tinyopt_rs::optimize(&mut wrapped, residuals(), &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(wrapped.value.data[0], 1.0, epsilon = 1e-8);
    assert_relative_eq!(wrapped.value.data[1], -2.0, epsilon = 1e-8);
    assert_relative_eq!(wrapped.value.data[2], 0.5, epsilon = 1e-8);
}

#[test]
fn angle_wrapping_in_retraction() {
    assert_relative_eq!(
        wrap_angle(std::f64::consts::PI + 0.1),
        -std::f64::consts::PI + 0.1
    );
    assert_relative_eq!(wrap_angle(3.0), 3.0);
    let _ = <V3 as DenseVector<f64>>::zeros(3);
}
