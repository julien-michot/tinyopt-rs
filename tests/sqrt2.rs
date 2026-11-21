// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/sqrt2.cpp: minimizing x² - 2 with LM/GN, autodiff and
//! manual accumulation, plus gradient verification.

use approx::assert_relative_eq;
#[cfg(feature = "autodiff")]
use tinyopt::AccumulationFn;
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
#[cfg(feature = "autodiff")]
use tinyopt::diff::{DualNum, ResidualsFn};
use tinyopt::diff::{Method, check_gradient, check_residuals_gradient};
#[cfg(feature = "autodiff")]
use tinyopt::optimize_residuals;
use tinyopt::{Cost, Options, Params, Solver, Static, optimize};

type V1 = Vector<f64, Static<1>>;
type M1 = Matrix<f64, Static<1>>;

fn scalar(data: f64) -> V1 {
    let mut v = <V1 as DenseVector<f64>>::zeros(1);
    v.set_at(0, data);
    v
}

/// r(x) = x² - 2
#[cfg(feature = "autodiff")]
struct Sqrt2;

#[cfg(feature = "autodiff")]
impl ResidualsFn for Sqrt2 {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
        let x0 = x[0].clone();
        vec![x0.clone() * x0 - 2.0]
    }
}

/// Manual accumulation: cost = (x² - 2)², grad = 2x r, H = 4x².
fn manual_acc() -> impl Fn(&[f64], Option<&mut V1>, Option<&mut M1>) -> Cost {
    |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
        let r = x[0] * x[0] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * x[0] * r);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 4.0 * x[0] * x[0]);
        }
        Cost::new(r * r)
    }
}

#[cfg(feature = "autodiff")]
#[test]
fn autodiff_levenberg_marquardt() {
    let mut x = scalar(1.0);
    let summary = optimize_residuals(&mut x, &Sqrt2, &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert!(summary.converged(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0_f64.sqrt(), epsilon = 1e-8);
    assert!(summary.num_iters > 0);
    assert!(!summary.num_diff_used);
}

#[cfg(feature = "autodiff")]
#[test]
fn autodiff_gauss_newton() {
    let mut x = scalar(2.0);
    let options = Options::new(Solver::GaussNewton);
    let summary = optimize_residuals(&mut x, &Sqrt2, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0_f64.sqrt(), epsilon = 1e-8);
}

#[test]
fn manual_accumulation_levenberg_marquardt() {
    let mut x = scalar(1.0);
    let summary = optimize(&mut x, manual_acc(), &Options::default());
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0_f64.sqrt(), epsilon = 1e-8);
    assert!(summary.final_cost.cost < 1e-12);
}

#[test]
fn manual_accumulation_gauss_newton() {
    let mut x = scalar(1.0);
    let summary = optimize(&mut x, manual_acc(), &Options::new(Solver::GaussNewton));
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0_f64.sqrt(), epsilon = 1e-8);
}

#[test]
fn scalar_parameter() {
    // The parameter itself is an f64 (no vector).
    let acc = |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
        let r = x[0] * x[0] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * x[0] * r);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 4.0 * x[0] * x[0]);
        }
        Cost::new(r * r)
    };
    let mut x = 1.0_f64;
    assert_eq!(Params::dims(&x), 1);
    let summary = optimize(&mut x, acc, &Options::default());
    assert!(summary.succeeded());
    assert_relative_eq!(x, 2.0_f64.sqrt(), epsilon = 1e-8);
}

#[test]
fn gradient_and_hessian_match_numerical() {
    let x = scalar(1.3);
    // NLLS convention: cost = rᵀr with grad = Jᵀr and H = JᵀJ, so the
    // residuals-based check is the meaningful one (as in C++).
    let acc = manual_acc();
    let res = |x: &[f64]| vec![x[0] * x[0] - 2.0];
    assert!(check_residuals_gradient::<f64, Static<1>, _, _>(
        DenseVector::as_slice(&x),
        &res,
        &acc,
        1e-5,
        Method::Central,
        false,
        true
    ));

    #[cfg(feature = "autodiff")]
    {
        let ad = tinyopt::diff::ad_accumulation(&Sqrt2);
        assert!(check_residuals_gradient::<f64, Static<1>, _, _>(
            DenseVector::as_slice(&x),
            &res,
            &ad,
            1e-5,
            Method::Central,
            false,
            true
        ));
    }
}

#[test]
fn scalar_cost_gradient_matches_numerical() {
    // A genuine scalar cost accumulator: cost = (x - 2)², grad = 2(x - 2).
    let acc = |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
        let d = x[0] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * d);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
        }
        Cost::new(d * d)
    };
    assert!(check_gradient::<f64, Static<1>, _>(
        &[0.7],
        &acc,
        1e-5,
        Method::Central,
        false
    ));
}

#[cfg(feature = "autodiff")]
#[test]
fn accumulator_is_invocable_with_partial_outputs() {
    let x = [1.5];
    let acc = tinyopt::diff::ad_accumulation(&Sqrt2);
    let cost = AccumulationFn::<f64, Static<1>>::eval(&acc, &x, None, None);
    assert!(cost.is_valid());
    assert_relative_eq!(cost.cost, (1.5f64 * 1.5 - 2.0).powi(2));
}
