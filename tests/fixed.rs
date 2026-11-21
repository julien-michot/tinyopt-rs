// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/fixed.cpp (dense parts): fixed parameter dimensions.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::optimizers::lm;
use tinyopt_rs::{Cost, Options, Params, Static, optimize};

type V3 = Vector<f64, Static<3>>;
type M3 = Matrix<f64, Static<3>>;

/// cost = Σ (x_i - target_i)², J = I -> H = 2 I (cost convention rᵀr).
fn shifted_quadratic() -> impl Fn(&[f64], Option<&mut V3>, Option<&mut M3>) -> Cost {
    let target = [1.0, 2.0, 3.0];
    move |x: &[f64], mut grad: Option<&mut V3>, mut h: Option<&mut M3>| -> Cost {
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

fn init() -> V3 {
    let mut x = <V3 as DenseVector<f64>>::zeros(3);
    x.set_at(0, 0.0);
    x.set_at(1, 0.0);
    x.set_at(2, 0.0);
    x
}

#[test]
fn explicit_fixed_index_is_not_updated() {
    let mut opt = lm::Optimizer::<V3>::new(Options::default());
    opt.fix(&[1]);
    assert!(opt.has_fixed());
    assert_eq!(opt.num_fixed(), 1);
    assert!(opt.is_fixed(1));
    assert!(!opt.is_fixed(0));

    let mut x = init();
    let summary = opt.optimize(&mut x, shifted_quadratic());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    // x1 stays exactly at its initial value.
    assert_eq!(x.at(1), 0.0);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(2), 3.0, epsilon = 1e-8);
}

#[test]
fn clear_fixed_releases_parameters() {
    let mut opt = lm::Optimizer::<V3>::new(Options::default());
    opt.fix(&[1]);
    opt.clear_fixed();
    assert!(!opt.has_fixed());

    let mut x = init();
    let summary = opt.optimize(&mut x, shifted_quadratic());
    assert!(summary.succeeded());
    assert_relative_eq!(x.at(1), 2.0, epsilon = 1e-6);
}

#[derive(Clone)]
struct FixedMid([f64; 3]);

impl Params for FixedMid {
    type Scalar = f64;
    type Dim = Static<3>;

    fn dims(&self) -> usize {
        3
    }

    fn as_slice(&self) -> &[f64] {
        &self.0
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        for (x, d) in self.0.iter_mut().zip(delta) {
            *x += *d;
        }
    }

    fn norm2(&self) -> f64 {
        self.0.iter().map(|v| v * v).sum()
    }

    fn fixed(&self) -> Option<Vec<usize>> {
        Some(vec![1])
    }
}

#[test]
fn trait_provided_fixed_indices() {
    let mut x = FixedMid([0.0; 3]);
    let summary = optimize(&mut x, shifted_quadratic(), &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_eq!(x.0[1], 0.0);
    assert_relative_eq!(x.0[0], 1.0, epsilon = 1e-8);
    assert_relative_eq!(x.0[2], 3.0, epsilon = 1e-8);
}
