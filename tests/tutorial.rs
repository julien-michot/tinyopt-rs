// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "autodiff")]

//! Mirrors the dense sections of docs/tutorials/cpp.md.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::diff::{DualNum, ResidualsFn};
use tinyopt::losses::{huber, huber_with_scale, squared_l2};
use tinyopt::optimizers::lm;
use tinyopt::{Cost, Dynamic, Options, Solver, Static, optimize, optimize_residuals};

type V2 = Vector<f64, Static<2>>;
type M2 = Matrix<f64, Static<2>>;

fn vec2(x: f64, y: f64) -> V2 {
    let mut v = <V2 as DenseVector<f64>>::zeros(2);
    v.set_at(0, x);
    v.set_at(1, y);
    v
}

#[test]
fn hello_world_scalar() {
    // 1) Scalar cost with manual gradient: (x - 2)².
    type V1 = Vector<f64, Static<1>>;
    type M1 = Matrix<f64, Static<1>>;
    let acc = |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
        let r = x[0] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
        }
        Cost::new(r * r)
    };
    let mut x = <V1 as DenseVector<f64>>::zeros(1);
    let summary = optimize(&mut x, acc, &Options::default());
    assert!(summary.converged());
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-8);
}

#[test]
fn residual_vector_autodiff() {
    struct Target;
    impl ResidualsFn for Target {
        type Scalar = f64;
        fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
            vec![x[0].clone() - 1.0, x[1].clone() - 2.0]
        }
    }
    let mut x = vec2(0.0, 0.0);
    let summary = optimize_residuals(&mut x, &Target, &Options::default());
    assert!(summary.succeeded());
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(1), 2.0, epsilon = 1e-8);
}

#[test]
fn explicit_optimizer_with_manual_accumulation() {
    // 3) + 4) `lm::Optimizer` object and manual Jᵀr / JᵀJ accumulation.
    let acc = |x: &[f64], grad: Option<&mut V2>, h: Option<&mut M2>| -> Cost {
        // r = [x0 - 1, x1 + 3], J = I.
        let r0 = x[0] - 1.0;
        let r1 = x[1] + 3.0;
        if let Some(g) = grad {
            g.set_at(0, r0);
            g.set_at(1, r1);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 1.0);
            h.set_elem(1, 1, 1.0);
        }
        Cost::with_num_residuals(r0 * r0 + r1 * r1, 2)
    };
    let mut opt = lm::Optimizer::<V2>::new(Options::default());
    let mut x = vec2(0.0, 0.0);
    let summary = opt.optimize(&mut x, acc);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(1), -3.0, epsilon = 1e-8);
}

#[test]
fn solver_choice_gradient_descent() {
    let acc = |x: &[f64], grad: Option<&mut V2>, _h: Option<&mut M2>| -> Cost {
        let (dx, dy) = (x[0] - 1.0, x[1] + 2.0);
        if let Some(g) = grad {
            g.set_at(0, 2.0 * dx);
            g.set_at(1, 2.0 * dy);
        }
        Cost::new(dx * dx + dy * dy)
    };
    let mut options = Options::new(Solver::GradientDescent);
    options.gd.lr = 0.1;
    let mut x = vec2(0.0, 0.0);
    let summary = optimize(&mut x, acc, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-3);
    assert_relative_eq!(x.at(1), -2.0, epsilon = 1e-3);
}

#[test]
fn robust_scalar_objective() {
    // 9) Huber applied to the squared norm of a residual, with the scale
    // returned by the loss applied to the gradient.
    let th2 = 0.7_f64 * 0.7;
    let acc = |x: &[f64], grad: Option<&mut V2>, h: Option<&mut M2>| -> Cost {
        let residual = [x[0] - 1.0, x[1] - 2.0];
        let n2 = squared_l2(&residual);
        if let Some(g) = grad {
            let (_, scale) = huber_with_scale(n2, th2);
            g.set_at(0, 2.0 * scale * residual[0]);
            g.set_at(1, 2.0 * scale * residual[1]);
        }
        if let Some(h) = h {
            let (_, scale) = huber_with_scale(n2, th2);
            h.set_elem(0, 0, 2.0 * scale);
            h.set_elem(1, 1, 2.0 * scale);
        }
        Cost::with_num_residuals(huber(n2, th2), 2)
    };
    let mut x = vec2(0.0, 0.0);
    let summary = optimize(&mut x, acc, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-6);
    assert_relative_eq!(x.at(1), 2.0, epsilon = 1e-6);
}

#[test]
fn dynamic_parameters_autodiff() {
    type Vx = Vector<f64, Dynamic>;
    struct Shift;
    impl ResidualsFn for Shift {
        type Scalar = f64;
        fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
            vec![x[0].clone() - 3.0, x[1].clone() + 1.0]
        }
    }
    let mut x = <Vx as DenseVector<f64>>::zeros(2);
    let summary = optimize_residuals(&mut x, &Shift, &Options::default());
    assert!(summary.succeeded());
    assert_relative_eq!(x.at(0), 3.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(1), -1.0, epsilon = 1e-8);
}
