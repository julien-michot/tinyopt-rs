// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/parameter_pack.cpp: packed parameters.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::{Cost, DynGradient, DynHessian, Options, Params, ParamsPack, Static, optimize};

type V2 = Vector<f64, Static<2>>;

fn vec2(x: f64, y: f64) -> V2 {
    let mut v = <V2 as DenseVector<f64>>::zeros(2);
    v.set_at(0, x);
    v.set_at(1, y);
    v
}

#[test]
fn pack_keeps_values_in_sync() {
    let mut pack = ParamsPack::new((vec2(1.0, 2.0), 3.0_f64));
    assert_eq!(Params::dims(&pack), 3);
    assert_eq!(pack.as_slice(), &[1.0, 2.0, 3.0]);

    Params::plus_eq(&mut pack, &[0.5, -1.0, 2.0]);
    assert_eq!(pack.values.0.at(0), 1.5);
    assert_eq!(pack.values.0.at(1), 1.0);
    assert_eq!(pack.values.1, 5.0);
    assert_eq!(pack.as_slice(), &[1.5, 1.0, 5.0]);
    assert_eq!(Params::norm2(&pack), 1.5f64.powi(2) + 1.0 + 25.0);
}

#[test]
fn optimize_a_pack_directly() {
    let mut pack = ParamsPack::new((vec2(0.0, 0.0), 0.0_f64));

    let acc = |flat: &[f64],
               mut grad: Option<&mut DynGradient>,
               mut h: Option<&mut DynHessian>|
     -> Cost {
        // flat = [x0, x1, y]; targets (1, -2, 3).
        let targets = [1.0, -2.0, 3.0];
        let mut cost = 0.0;
        for (i, t) in targets.iter().enumerate() {
            let r = flat[i] - t;
            cost += r * r;
            if let Some(g) = grad.as_mut() {
                g.set_at(i, 2.0 * r);
            }
            if let Some(hm) = h.as_mut() {
                hm.set_elem(i, i, 2.0);
            }
        }
        Cost::with_num_residuals(cost, 3)
    };

    let summary = optimize(&mut pack, acc, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(pack.values.0.at(0), 1.0, epsilon = 1e-8);
    assert_relative_eq!(pack.values.0.at(1), -2.0, epsilon = 1e-8);
    assert_relative_eq!(pack.values.1, 3.0, epsilon = 1e-8);
}

#[test]
fn pack_of_three_elements() {
    let mut pack = ParamsPack::new((vec2(0.0, 0.0), vec2(0.0, 0.0), 0.0_f64));
    assert_eq!(Params::dims(&pack), 5);

    Params::plus_eq(&mut pack, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(pack.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let _ = <Matrix<f64, Static<2>> as DenseMatrix<f64>>::zeros(2, 2);
}
