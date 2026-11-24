// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "autodiff")]

//! Mirrors tests/cpp/types_vector.cpp: vector parameters with residual
//! vectors, scalar residuals, and dynamic sizes.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseVector, Vector};
use tinyopt::diff::{DualNum, ResidualsFn};
use tinyopt::{Dynamic, Options, Params, Static, optimize_residuals};

type V2 = Vector<f64, Static<2>>;

fn vec2(x: f64, y: f64) -> V2 {
    let mut v = <V2 as DenseVector<f64>>::zeros(2);
    v.set_at(0, x);
    v.set_at(1, y);
    v
}

/// Two linear residuals: [x + y - 3, x - y - 1] -> solution (2, 1).
struct Linear;

impl ResidualsFn for Linear {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        let (x0, x1) = (x[0].clone(), x[1].clone());
        vec![x0.clone() + x1.clone() - 3.0, x0 - x1 - 1.0]
    }
}

/// One scalar residual: x0 + 2 x1 - 5 -> the minimum set is a line.
struct ScalarResidual;

impl ResidualsFn for ScalarResidual {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        vec![x[0].clone() + x[1].clone() * 2.0 - 5.0]
    }
}

#[test]
fn vector_parameter_with_residual_vector() {
    let mut x = vec2(0.0, 0.0);
    let summary = optimize_residuals(&mut x, &Linear, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(1), 1.0, epsilon = 1e-8);
    assert_eq!(summary.num_residuals, 2);
}

#[test]
fn vector_parameter_with_scalar_residual() {
    let mut x = vec2(0.0, 0.0);
    let summary = optimize_residuals(&mut x, &ScalarResidual, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(summary.final_cost.cost < 1e-12);
    assert_relative_eq!(x.at(0) + 2.0 * x.at(1), 5.0, epsilon = 1e-6);
    assert_eq!(summary.num_residuals, 1);
}

#[test]
fn dynamic_parameter_with_array_residual() {
    type Vx = Vector<f64, Dynamic>;

    struct Shift {
        target: Vec<f64>,
    }

    impl ResidualsFn for Shift {
        type Scalar = f64;

        fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
            x.iter()
                .zip(&self.target)
                .map(|(xi, t)| xi.clone() - *t)
                .collect()
        }
    }

    let mut x = <Vx as DenseVector<f64>>::zeros(3);
    x.set_at(0, 0.0);
    x.set_at(1, 0.0);
    x.set_at(2, 0.0);
    let f = Shift {
        target: vec![1.0, -2.0, 3.0],
    };
    let summary = optimize_residuals(&mut x, &f, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(1), -2.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(2), 3.0, epsilon = 1e-8);
    assert_eq!(Params::dims(&x), 3);
}
