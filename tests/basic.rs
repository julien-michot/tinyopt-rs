// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/basic.cpp: summary contract, stop reasons, and callbacks.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use approx::assert_relative_eq;
use tinyopt_rs::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt_rs::{Cost, Options, Static, StopReason, optimize};

type V1 = Vector<f64, Static<1>>;
type M1 = Matrix<f64, Static<1>>;

fn scalar(data: f64) -> V1 {
    let mut v = <V1 as DenseVector<f64>>::zeros(1);
    v.set_at(0, data);
    v
}

/// Quadratic cost (x - 2)² with gradient and Hessian.
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
fn skipped_when_no_residuals() {
    let mut x = scalar(0.0);
    let acc = |_x: &[f64], _g: Option<&mut V1>, _h: Option<&mut M1>| Cost::default();
    let summary = optimize(&mut x, acc, &Options::default());
    assert_eq!(summary.stop_reason, StopReason::Skipped);
    assert!(!summary.succeeded());
}

#[test]
fn nan_cost_stops() {
    let mut x = scalar(0.0);
    let acc = |_x: &[f64], _g: Option<&mut V1>, _h: Option<&mut M1>| Cost::new(f64::NAN);
    let summary = optimize(&mut x, acc, &Options::default());
    assert_eq!(summary.stop_reason, StopReason::SystemHasNanOrInf);
    assert!(!summary.succeeded());
}

#[test]
fn max_iters_is_reported() {
    let mut options = Options::default();
    options.stop.max_iters = 1;
    options.stop.min_error = 0.0;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 0.0;
    options.stop.min_step_norm2 = 0.0;
    options.stop.min_step_rel_norm2 = 0.0;
    options.stop.min_rerr_dec = 0.0;

    let mut x = scalar(0.0);
    let summary = optimize(&mut x, quadratic_acc(), &options);
    assert_eq!(summary.stop_reason, StopReason::MaxIters);
    assert!(summary.succeeded());
    assert!(!summary.converged());
    assert_eq!(summary.num_iters, 1);
}

#[test]
fn step_callback_can_stop() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let mut options = Options::default();
    options.stop.step_callback = Some(Arc::new(move |_dx: &[f32], _rollback: bool| {
        c.fetch_add(1, Ordering::Relaxed) >= 1
    }));

    let mut x = scalar(0.0);
    let summary = optimize(&mut x, quadratic_acc(), &options);
    assert_eq!(summary.stop_reason, StopReason::UserStopped);
    assert!(calls.load(Ordering::Relaxed) >= 2);
    assert!(summary.succeeded());
}

#[test]
fn stop_callback_can_stop() {
    let mut options = Options::default();
    options.stop.stop_callback = Some(Arc::new(|_err: f64, _dx: f64, _g: f64| true));
    let mut x = scalar(0.0);
    let summary = optimize(&mut x, quadratic_acc(), &options);
    assert_eq!(summary.stop_reason, StopReason::UserStopped);
}

#[test]
fn timeout_is_reported() {
    let mut options = Options::default();
    options.stop.max_duration_ms = 1e-9;
    options.stop.min_rerr_dec = 0.0;
    options.stop.min_step_norm2 = 0.0;
    options.stop.min_step_rel_norm2 = 0.0;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 0.0;
    options.stop.min_error = 0.0;

    let mut x = scalar(0.0);
    let summary = optimize(&mut x, quadratic_acc(), &options);
    assert_eq!(summary.stop_reason, StopReason::TimedOut);
}

#[test]
fn summary_statistics_and_history() {
    let mut x = scalar(0.0);
    let summary = optimize(&mut x, quadratic_acc(), &Options::default());
    assert!(summary.succeeded());
    assert!(summary.converged());
    assert_eq!(summary.num_residuals, 1);
    assert!(summary.num_iters > 0);
    assert_eq!(summary.num_failures, 0);
    assert!(!summary.has_final_cov());
    assert!(!summary.hist.is_empty());
    assert_eq!(summary.hist.len(), summary.num_iters as usize);
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-5);
    assert!(summary.final_cost.cost < 1e-12);
}

#[test]
fn history_can_be_disabled() {
    let mut options = Options::default();
    options.exp.save_history = false;
    let mut x = scalar(0.0);
    let summary = optimize(&mut x, quadratic_acc(), &options);
    assert!(summary.hist.is_empty());
}

#[test]
fn stop_reason_descriptions() {
    let o = Options::default();
    assert!(
        StopReason::MinError
            .description(Some(&o), 1e-13)
            .contains("minimum error")
    );
    assert!(
        StopReason::SolverFailed
            .description(None, 0.0)
            .contains("failure")
    );
}
