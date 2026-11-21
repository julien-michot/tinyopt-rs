// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Mirrors tests/cpp/logging.cpp: optimization logs go through the `log` crate.

use std::sync::{Mutex, Once};

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::{Cost, Options, Static, optimize};

static LOGS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static INIT: Once = Once::new();

static CAPTURE: Capture = Capture;

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _metadata: &log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &log::Record<'_>) {
        if let Ok(mut logs) = LOGS.lock() {
            logs.push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

fn init_logger() {
    INIT.call_once(|| {
        let _ = log::set_logger(&CAPTURE);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

fn clear_logs() {
    if let Ok(mut logs) = LOGS.lock() {
        logs.clear();
    }
}

fn captured() -> Vec<String> {
    LOGS.lock().map(|l| l.clone()).unwrap_or_default()
}

type V1 = Vector<f64, Static<1>>;
type M1 = Matrix<f64, Static<1>>;

fn acc() -> impl Fn(&[f64], Option<&mut V1>, Option<&mut M1>) -> Cost {
    |x: &[f64], grad: Option<&mut V1>, mut h: Option<&mut M1>| -> Cost {
        let r = x[0] - 2.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r);
        }
        if let Some(h) = h.as_mut() {
            h.set_elem(0, 0, 2.0);
        }
        Cost::new(r * r)
    }
}

#[test]
fn logging_behaviour() {
    init_logger();
    clear_logs();

    let mut options = Options::default();
    options.stop.max_iters = 3;
    options.stop.min_error = 0.0;
    options.stop.min_rerr_dec = 0.0;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 0.0;

    let mut x = <V1 as DenseVector<f64>>::zeros(1);
    let summary = optimize(&mut x, acc(), &options);
    assert!(summary.succeeded());

    let logs = captured();
    assert!(!logs.is_empty(), "expected per-iteration log messages");
    assert!(logs.iter().any(|m| m.contains("cost")));

    // Now disable logging and check that nothing new is emitted.
    clear_logs();
    let mut options = Options::default();
    options.log.enable = false;

    let mut x = <V1 as DenseVector<f64>>::zeros(1);
    let summary = optimize(&mut x, acc(), &options);
    assert!(summary.succeeded());
    assert_relative_eq!(x.at(0), 2.0, epsilon = 1e-6);

    assert!(captured().is_empty(), "log.enable = false must stay silent");
}
