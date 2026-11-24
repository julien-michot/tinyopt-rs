// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/variadic.cpp: multi-parameter optimization.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector};
use tinyopt::{Cost, DynGradient, DynHessian, Options, Params, optimize2, optimize3};

#[test]
fn two_scalar_parameters() {
    let mut x = 0.0_f64;
    let mut y = 0.0_f64;
    assert_eq!(Params::dims(&x), 1);

    let acc = |xv: &[f64],
               yv: &[f64],
               grad: Option<&mut DynGradient>,
               h: Option<&mut DynHessian>|
     -> Cost {
        let (dx, dy) = (xv[0] - 1.0, yv[0] + 2.0);
        if let Some(g) = grad {
            g.set_at(0, 2.0 * dx);
            g.set_at(1, 2.0 * dy);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
            h.set_elem(1, 1, 2.0);
        }
        Cost::new(dx * dx + dy * dy)
    };

    let summary = optimize2(&mut x, &mut y, acc, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x, 1.0, epsilon = 1e-8);
    assert_relative_eq!(y, -2.0, epsilon = 1e-8);
}

#[test]
fn three_scalar_parameters() {
    let mut x = 0.0_f64;
    let mut y = 0.0_f64;
    let mut z = 0.0_f64;

    let acc = |xv: &[f64],
               yv: &[f64],
               zv: &[f64],
               grad: Option<&mut DynGradient>,
               h: Option<&mut DynHessian>|
     -> Cost {
        let (dx, dy, dz) = (xv[0] - 1.0, yv[0] - 2.0, zv[0] - 3.0);
        if let Some(g) = grad {
            g.set_at(0, 2.0 * dx);
            g.set_at(1, 2.0 * dy);
            g.set_at(2, 2.0 * dz);
        }
        if let Some(h) = h {
            for i in 0..3 {
                h.set_elem(i, i, 2.0);
            }
        }
        Cost::new(dx * dx + dy * dy + dz * dz)
    };

    let summary = optimize3(&mut x, &mut y, &mut z, acc, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x, 1.0, epsilon = 1e-8);
    assert_relative_eq!(y, 2.0, epsilon = 1e-8);
    assert_relative_eq!(z, 3.0, epsilon = 1e-8);
}

#[test]
fn vector_and_scalar_parameters() {
    use tinyopt::Static;
    use tinyopt::backend::{DenseVector, Vector};
    type V2 = Vector<f64, Static<2>>;
    let mut center = <V2 as DenseVector<f64>>::zeros(2);
    let mut scale = 0.0_f64;

    let acc = |c: &[f64],
               s: &[f64],
               grad: Option<&mut DynGradient>,
               h: Option<&mut DynHessian>|
     -> Cost {
        // Independent residuals: [c0 - 1, c1 + 1, s - 2].
        let (r0, r1, r2) = (c[0] - 1.0, c[1] + 1.0, s[0] - 2.0);
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r0);
            g.set_at(1, 2.0 * r1);
            g.set_at(2, 2.0 * r2);
        }
        if let Some(hm) = h {
            for i in 0..3 {
                hm.set_elem(i, i, 2.0);
            }
        }
        Cost::with_num_residuals(r0 * r0 + r1 * r1 + r2 * r2, 3)
    };

    let summary = optimize2(&mut center, &mut scale, acc, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(center.at(0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(center.at(1), -1.0, epsilon = 1e-6);
    assert_relative_eq!(scale, 2.0, epsilon = 1e-6);
}
