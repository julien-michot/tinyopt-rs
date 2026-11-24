// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Behaviour of the C++ option knobs: cost scaling, gradient clipping,
//! Hessian diagonal checks, fixed-parameter permutation and failure logs.

use std::sync::{Mutex, Once};

use approx::assert_relative_eq;
use tinyopt::backend::{DenseMatrix, DenseVector, Dynamic, Matrix, Vector};
use tinyopt::optimizers::lm;
use tinyopt::{Cost, Options, StopReason, optimize};

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

type V1 = Vector<f64, tinyopt::Static<1>>;
type M1 = Matrix<f64, tinyopt::Static<1>>;

/// cost = r^2 with r = x - 1 and 3 residuals reported.
fn simple_target() -> impl Fn(&[f64], Option<&mut V1>, Option<&mut M1>) -> Cost {
    |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
        let r = x[0] - 1.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
        }
        Cost::with_num_residuals(r * r, 3)
    }
}

#[test]
fn cost_scaling_is_applied() {
    let run = |options: &Options| -> f64 {
        let mut xx = <V1 as DenseVector<f64>>::zeros(1);
        xx.set_at(0, 4.0); // r = 3, cost = 9
        let summary = optimize(&mut xx, simple_target(), options);
        summary.final_cost.cost
    };

    let mut options = Options::default();
    options.stop.max_iters = 0;
    assert_relative_eq!(run(&options), 9.0, epsilon = 1e-12);

    options.cost.use_squared_norm = false;
    assert_relative_eq!(run(&options), 3.0, epsilon = 1e-12);

    let mut options = Options::default();
    options.stop.max_iters = 0;
    options.cost.downscale_by_2 = true;
    assert_relative_eq!(run(&options), 4.5, epsilon = 1e-12);

    let mut options = Options::default();
    options.stop.max_iters = 0;
    options.cost.normalize = true;
    assert_relative_eq!(run(&options), 3.0, epsilon = 1e-12);
}

#[test]
fn gradient_clipping_bounds_the_step() {
    let acc = |x: &[f64], grad: Option<&mut V1>, h: Option<&mut M1>| -> Cost {
        let r = x[0] - 100.0;
        if let Some(g) = grad {
            g.set_at(0, 2.0 * r);
        }
        if let Some(h) = h {
            h.set_elem(0, 0, 2.0);
        }
        Cost::new(r * r)
    };
    let mut options = Options::default();
    options.stop.max_iters = 1;
    options.opt.grad_clipping = 1.0;
    let mut x = <V1 as DenseVector<f64>>::zeros(1);
    let _ = optimize(&mut x, acc, &options);
    assert!(x.at(0) < 1.0, "step was not clipped: x = {}", x.at(0));
}

#[test]
fn zero_hessian_diagonal_is_rejected() {
    // The accumulator fills the gradient but leaves the Hessian at zero.
    let acc = |x: &[f64], grad: Option<&mut V1>, _h: Option<&mut M1>| -> Cost {
        if let Some(g) = grad {
            g.set_at(0, 2.0 * (x[0] - 1.0));
        }
        Cost::new((x[0] - 1.0) * (x[0] - 1.0))
    };
    let mut options = Options::default();
    options.hessian.check_min_h_diag = 1e-6;
    let mut x = <V1 as DenseVector<f64>>::zeros(1);
    let summary = optimize(&mut x, acc, &options);
    assert_eq!(summary.stop_reason, StopReason::SolverFailed);
}

#[test]
fn fixed_parameter_permutation_matches_masking() {
    // 4 dynamic parameters, 3 fixed (ratio 0.75 >= 0.5 triggers the reduced solve).
    let acc = |x: &[f64],
               mut grad: Option<&mut Vector<f64, Dynamic>>,
               mut h: Option<&mut Matrix<f64, Dynamic>>|
     -> Cost {
        let target = [1.0, 2.0, 3.0, 4.0];
        let mut cost = 0.0;
        for i in 0..4 {
            let r = x[i] - target[i];
            cost += r * r;
            if let Some(g) = grad.as_mut() {
                g.set_at(i, 2.0 * r);
            }
            if let Some(h) = h.as_mut() {
                h.set_elem(i, i, 2.0);
            }
        }
        Cost::with_num_residuals(cost, 4)
    };

    let run = |ratio: f32| -> Vec<f64> {
        let mut options = Options::default();
        options.opt.min_fixed_ratio = ratio;
        let mut x = vec![0.0; 4];
        let mut opt = lm::Optimizer::<Vec<f64>, Matrix<f64, Dynamic>>::new(options);
        opt.fix(&[0, 1, 2]);
        let summary = opt.optimize(&mut x, acc);
        assert!(summary.succeeded(), "{:?}", summary.stop_reason);
        x
    };

    let masked = run(0.0);
    let compressed = run(0.5);
    for i in 0..4 {
        assert_relative_eq!(compressed[i], masked[i], epsilon = 1e-10);
    }
    // The three fixed parameters did not move.
    for value in compressed.iter().take(3) {
        assert_eq!(*value, 0.0);
    }
    assert_relative_eq!(compressed[3], 4.0, epsilon = 1e-6);
}

#[test]
fn print_failure_logs_the_system() {
    init_logger();
    clear_logs();
    // Zero Hessian with damping disabled: the solve produces a non-finite step
    // (or fails), and the diagnostics are logged when print_failure is set.
    let acc = |x: &[f64], grad: Option<&mut V1>, _h: Option<&mut M1>| -> Cost {
        if let Some(g) = grad {
            g.set_at(0, 2.0 * (x[0] - 1.0));
        }
        Cost::new((x[0] - 1.0) * (x[0] - 1.0))
    };
    let mut options = Options::default();
    options.log.enable = true;
    options.log.print_failure = true;
    options.lm.damping_init = 0.0;
    options.lm.jacobi_scaling = false;
    let mut x = <V1 as DenseVector<f64>>::zeros(1);
    let _ = optimize(&mut x, acc, &options);
    let logs = captured();
    assert!(
        logs.iter()
            .any(|m| m.contains("non-finite step") || m.contains("failed to solve")),
        "expected a failure log, got {logs:?}"
    );
}
