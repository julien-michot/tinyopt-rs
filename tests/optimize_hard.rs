// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/optimize_hard.cpp: challenging objectives and line searches.

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
#[cfg(feature = "autodiff")]
use tinyopt::optimize_residuals;
use tinyopt::{Cost, LineSearchMethod, Options, Solver, Static, optimize};

type V2 = Vector<f64, Static<2>>;
type M2 = Matrix<f64, Static<2>>;

fn rosenbrock() -> impl Fn(&[f64], Option<&mut V2>, Option<&mut M2>) -> Cost {
    |x: &[f64], grad: Option<&mut V2>, h: Option<&mut M2>| -> Cost {
        let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
        if let Some(g) = grad {
            g.set_at(0, -2.0 * a - 400.0 * x[0] * b);
            g.set_at(1, 200.0 * b);
        }
        if let Some(h) = h {
            // Dense Gauss-Newton approximation of the Rosenbrock Hessian.
            h.set_elem(0, 0, 2.0 + 400.0 * (3.0 * x[0] * x[0] - x[1]));
            h.set_elem(0, 1, -400.0 * x[0]);
            h.set_elem(1, 0, -400.0 * x[0]);
            h.set_elem(1, 1, 200.0);
        }
        Cost::new(a * a + 100.0 * b * b)
    }
}

#[test]
fn levenberg_marquardt_solves_rosenbrock() {
    let mut options = Options::new(Solver::LevenbergMarquardt);
    options.stop.max_iters = 300;
    options.stop.min_grad_norm2 = 1e-18;
    options.stop.max_consec_failures = 0;
    options.lm.damping_init = 1e-2;
    let mut x = <V2 as DenseVector<f64>>::zeros(2);
    x.set_at(0, -1.2);
    x.set_at(1, 1.0);
    let summary = optimize(&mut x, rosenbrock(), &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-3);
    assert_relative_eq!(x.at(1), 1.0, epsilon = 1e-3);
}

#[test]
fn ill_conditioned_quadratic_converges() {
    // Condition number 1e4: an exact Newton step still solves it in one go.
    let acc = |x: &[f64], grad: Option<&mut V2>, h: Option<&mut M2>| -> Cost {
        let (d0, d1) = (x[0] - 2.0, x[1] + 1.0);
        if let Some(g) = grad {
            g.set_at(0, 2.0 * d0);
            g.set_at(1, 2.0e4 * d1);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
            h.set_elem(1, 1, 2.0e4);
        }
        Cost::with_num_residuals(d0 * d0 + 1.0e4 * d1 * d1, 2)
    };
    let mut options = Options::new(Solver::GaussNewton);
    options.stop.max_iters = 20;
    let mut x = <V2 as DenseVector<f64>>::zeros(2);
    let summary = optimize(&mut x, acc, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-8);
    assert_relative_eq!(x.at(1), -1.0, epsilon = 1e-8);
}

#[test]
fn line_search_methods_reach_the_solution() {
    for method in [
        LineSearchMethod::None,
        LineSearchMethod::Backtracking,
        LineSearchMethod::Wolfe,
        LineSearchMethod::GoldenSection,
    ] {
        let mut options = Options::new(Solver::GradientDescent);
        options.stop.max_iters = 500;
        options.stop.min_grad_norm2 = 1e-14;
        options.line_search.method = method;
        options.line_search.initial_step = 1.0;
        options.gd.lr = 0.5;
        let mut x = <V2 as DenseVector<f64>>::zeros(2);
        let summary = optimize(&mut x, quadratic(), &options);
        assert!(summary.succeeded(), "{method:?}: {:?}", summary.stop_reason);
        assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-4);
        assert_relative_eq!(x.at(1), -2.0, epsilon = 1e-4);
    }
}

fn quadratic() -> impl Fn(&[f64], Option<&mut V2>, Option<&mut M2>) -> Cost {
    |x: &[f64], grad: Option<&mut V2>, h: Option<&mut M2>| -> Cost {
        let (d0, d1) = (x[0] - 1.0, x[1] + 2.0);
        if let Some(g) = grad {
            g.set_at(0, 2.0 * d0);
            g.set_at(1, 2.0 * d1);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
            h.set_elem(1, 1, 2.0);
        }
        Cost::new(d0 * d0 + d1 * d1)
    }
}

#[cfg(feature = "autodiff")]
#[test]
fn autodiff_residuals_solve_a_hard_least_squares_problem() {
    // Circle fit: (x - cx)² + (y - cy)² = r², heavily over-determined.
    let points = [
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (0.7, 0.7),
        (-0.7, 0.7),
        (-0.7, -0.7),
        (0.7, -0.7),
    ];
    struct CircleResiduals {
        points: [(f64, f64); 8],
    }
    impl tinyopt::diff::ResidualsFn for CircleResiduals {
        type Scalar = f64;
        fn eval<T: tinyopt::diff::DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
            self.points
                .iter()
                .map(|(px, py)| {
                    let dx = T::from_f64(*px).unwrap() - x[0].clone();
                    let dy = T::from_f64(*py).unwrap() - x[1].clone();
                    dx.clone() * dx + dy.clone() * dy - x[2].clone() * x[2].clone()
                })
                .collect()
        }
    }
    let mut options = Options::default();
    options.stop.max_iters = 100;
    options.stop.min_step_norm2 = 0.0;
    options.stop.min_rerr_dec = 0.0;
    let mut x = <Vector<f64, Static<3>> as DenseVector<f64>>::zeros(3);
    x.set_at(0, 0.3);
    x.set_at(2, 1.0);
    let summary = optimize_residuals(&mut x, &CircleResiduals { points }, &options);
    assert!(summary.succeeded(), "{:?}", summary.stop_reason);
    assert_relative_eq!(x.at(0), 0.0, epsilon = 1e-6);
    assert_relative_eq!(x.at(1), 0.0, epsilon = 1e-6);
    // Least-squares radius: half the points are at radius 1, half at sqrt(0.98).
    assert_relative_eq!(x.at(2), 0.99_f64.sqrt(), epsilon = 1e-6);
}
