// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/optimizer_lm.cpp and optimizer_gn.cpp: direct optimizer
//! objects, reuse, and options.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::optimizers::{gn, lm};
use tinyopt_rs::{Cost, Options, Static};

type V1 = Vector<f64, Static<1>>;
type M1 = Matrix<f64, Static<1>>;

fn scalar(data: f64) -> V1 {
    let mut v = <V1 as DenseVector<f64>>::zeros(1);
    v.set_at(0, data);
    v
}

fn quadratic_acc() -> impl Fn(&[f64], Option<&mut V1>, Option<&mut M1>) -> Cost {
    |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
        let r = x[0] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
        }
        Cost::new(r * r)
    }
}

#[test]
fn lm_optimizer_object() {
    let mut opt = lm::Optimizer::<V1>::new(Options::default());
    let mut x = scalar(0.0);
    let summary = opt.optimize(&mut x, quadratic_acc());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-8);
    assert!(opt.lambda() > 0.0);
}

#[test]
fn gn_optimizer_object() {
    let mut opt = gn::Optimizer::<V1>::new(Options::new(tinyopt_rs::Solver::GaussNewton));
    let mut x = scalar(0.0);
    let summary = opt.optimize(&mut x, quadratic_acc());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-8);
}

#[test]
fn optimizer_reuse_across_calls() {
    let mut opt = gn::Optimizer::<V1>::new(Options::default());
    let mut x = scalar(0.0);
    let s1 = opt.optimize(&mut x, quadratic_acc());
    assert!(s1.succeeded());
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-8);

    // Perturb and optimize again with the same optimizer object.
    x.set_at(0, -5.0);
    let s2 = opt.optimize(&mut x, quadratic_acc());
    assert!(s2.succeeded());
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-8);
}
