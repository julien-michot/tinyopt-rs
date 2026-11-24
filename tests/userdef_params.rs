// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/userdef_params.cpp: custom parameter types.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::{Cost, Options, Params, Static, optimize};

type V2 = Vector<f64, Static<2>>;
type M2 = Matrix<f64, Static<2>>;

/// A rectangle described by its width and height.
#[derive(Clone, Debug)]
struct Rectangle {
    size: [f64; 2],
}

impl Params for Rectangle {
    type Scalar = f64;
    type Dim = Static<2>;

    fn dims(&self) -> usize {
        2
    }

    fn as_slice(&self) -> &[f64] {
        &self.size
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        self.size[0] += delta[0];
        self.size[1] += delta[1];
    }

    fn norm2(&self) -> f64 {
        self.size.iter().map(|v| v * v).sum()
    }
}

/// cost = (w - 3)² + (h - 4)².
fn rectangle_cost() -> impl Fn(&[f64], Option<&mut V2>, Option<&mut M2>) -> Cost {
    |x: &[f64], mut grad: Option<&mut V2>, mut h: Option<&mut M2>| -> Cost {
        let (rw, rh) = (x[0] - 3.0, x[1] - 4.0);
        if let Some(g) = grad.as_mut() {
            g.set_at(0, 2.0 * rw);
            g.set_at(1, 2.0 * rh);
        }
        if let Some(h) = h.as_mut() {
            h.set_elem(0, 0, 2.0);
            h.set_elem(1, 1, 2.0);
        }
        Cost::with_num_residuals(rw * rw + rh * rh, 2)
    }
}

#[test]
fn custom_parameters_are_optimized() {
    let mut rect = Rectangle { size: [1.0, 1.0] };
    assert_eq!(Params::dims(&rect), 2);
    assert_relative_eq!(Params::norm2(&rect), 2.0);

    let summary = optimize(&mut rect, rectangle_cost(), &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(rect.size[0], 3.0, epsilon = 1e-8);
    assert_relative_eq!(rect.size[1], 4.0, epsilon = 1e-8);
}

/// Runtime-sized custom parameter type.
#[derive(Clone, Debug)]
struct DynamicBox {
    values: Vec<f64>,
}

impl Params for DynamicBox {
    type Scalar = f64;
    type Dim = tinyopt::Dynamic;

    fn dims(&self) -> usize {
        self.values.len()
    }

    fn as_slice(&self) -> &[f64] {
        &self.values
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        for (v, d) in self.values.iter_mut().zip(delta) {
            *v += *d;
        }
    }

    fn norm2(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum()
    }
}

#[test]
fn dynamic_custom_parameters() {
    use tinyopt::Dynamic;
    type Vx = Vector<f64, Dynamic>;
    type Mx = Matrix<f64, Dynamic>;

    let acc = |x: &[f64], mut grad: Option<&mut Vx>, mut h: Option<&mut Mx>| -> Cost {
        let mut cost = 0.0;
        for (i, xi) in x.iter().enumerate() {
            let r = xi - i as f64;
            cost += r * r;
            if let Some(g) = grad.as_mut() {
                g.set_at(i, 2.0 * r);
            }
            if let Some(hm) = h.as_mut() {
                hm.set_elem(i, i, 2.0);
            }
        }
        Cost::with_num_residuals(cost, x.len() as i32)
    };

    let mut param = DynamicBox {
        values: vec![0.0, 0.0, 0.0],
    };
    let summary = optimize(&mut param, acc, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    for (i, v) in param.values.iter().enumerate() {
        assert_relative_eq!(*v, i as f64, epsilon = 1e-8);
    }
}
