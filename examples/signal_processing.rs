// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Estimate a sampled sinusoid's amplitude and DC offset when its frequency is
//! known, using conjugate gradients.
//!
//! Mirrors `examples/cpp/signal_processing.cpp`.

mod common;

use tinyopt::backend::{DenseVector, Vector};
use tinyopt::diff::{CostFn, DualNum, cost_accumulation};
use tinyopt::optimizers::cg;
use tinyopt::{Options, Static};

type V2 = Vector<f64, Static<2>>;

const ANGULAR_FREQUENCY: f64 = 2.0 * std::f64::consts::PI;
const TRUE_AMPLITUDE: f64 = 1.8;
const TRUE_OFFSET: f64 = 0.35;
const TIMES: [f64; 12] = [
    0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88,
];

struct Sinusoid;

impl CostFn for Sinusoid {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, estimate: &[T]) -> T {
        let mut error = estimate[0].clone() * 0.0;
        for time in TIMES {
            let basis = (ANGULAR_FREQUENCY * time).sin();
            let target = TRUE_AMPLITUDE * basis + TRUE_OFFSET;
            let residual = estimate[0].clone() * basis + estimate[1].clone() - target;
            error += residual.clone() * residual;
        }
        error
    }
}

fn main() {
    // [amplitude, DC offset], starting at zero.
    let mut parameters = <V2 as DenseVector<f64>>::zeros(2);

    let mut options = Options::default();
    options.log.enable = false;
    options.stop.max_iters = 100;
    options.stop.min_rerr_dec = 0.0;
    options.cg.step_size = 0.25;

    let summary =
        cg::Optimizer::<V2>::new(options).optimize(&mut parameters, cost_accumulation(&Sinusoid));

    println!(
        "Estimated [amplitude, offset]: [{}]",
        common::fmt_slice(DenseVector::as_slice(&parameters))
    );
    println!("Optimization succeeded: {}", summary.succeeded());

    std::process::exit(if summary.succeeded() { 0 } else { 1 });
}
