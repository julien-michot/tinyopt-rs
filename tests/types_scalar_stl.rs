// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "autodiff")]

//! Mirrors tests/cpp/types_scalar_stl.cpp: scalars, arrays, and `Vec` parameters.

use approx::assert_relative_eq;
use tinyopt_rs::diff::{DualNum, ResidualsFn};
use tinyopt_rs::{Options, Params, optimize_residuals};

struct Sqrt2;
impl ResidualsFn for Sqrt2 {
    type Scalar = f64;
    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        let x0 = x[0].clone();
        vec![x0.clone() * x0 - 2.0]
    }
}

#[test]
fn scalar_f64_parameter() {
    let mut x = 1.0_f64;
    assert_eq!(Params::dims(&x), 1);
    let summary = optimize_residuals(&mut x, &Sqrt2, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x, 2.0_f64.sqrt(), epsilon = 1e-8);
}

#[test]
fn fixed_size_array_parameter() {
    struct Shift;
    impl ResidualsFn for Shift {
        type Scalar = f64;
        fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
            vec![x[0].clone() - 1.0, x[1].clone() + 2.0, x[2].clone() - 3.0]
        }
    }

    let mut x = [0.0_f64, 0.0, 0.0];
    assert_eq!(Params::dims(&x), 3);
    let summary = optimize_residuals(&mut x, &Shift, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x[0], 1.0, epsilon = 1e-8);
    assert_relative_eq!(x[1], -2.0, epsilon = 1e-8);
    assert_relative_eq!(x[2], 3.0, epsilon = 1e-8);
}

#[test]
fn vec_parameter() {
    struct Shift {
        target: [f64; 2],
    }
    impl ResidualsFn for Shift {
        type Scalar = f64;
        fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
            vec![x[0].clone() - self.target[0], x[1].clone() - self.target[1]]
        }
    }

    let mut x = vec![0.0_f64, 0.0];
    assert_eq!(Params::dims(&x), 2);
    let f = Shift {
        target: [2.0, -4.0],
    };
    let summary = optimize_residuals(&mut x, &f, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x[0], 2.0, epsilon = 1e-6);
    assert_relative_eq!(x[1], -4.0, epsilon = 1e-6);
}
