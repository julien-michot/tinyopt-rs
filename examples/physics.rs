// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Fit a damped harmonic oscillator's amplitude, damping, frequency and offset
//! from sampled measurements, using a dog-leg trust region.
//!
//! Mirrors `examples/cpp/physics.cpp`.

mod common;

use tinyopt::backend::{DenseVector, Vector};
use tinyopt::diff::{DualNum, ResidualsFn, ad_accumulation};
use tinyopt::optimizers::dl;
use tinyopt::{Options, Static};

type V4 = Vector<f64, Static<4>>;

const TIMES: [f64; 8] = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1];
const TRUTH: [f64; 4] = [2.5, 0.15, 1.7, 0.2];

fn predict_f64(p: &[f64], t: f64) -> f64 {
    p[0] * (-p[1] * t).exp() * (p[2] * t).cos() + p[3]
}

fn predict<T: DualNum<Primitive = f64>>(p: &[T], t: f64) -> T {
    p[0].clone() * (-(p[1].clone() * t)).exp() * (p[2].clone() * t).cos() + p[3].clone()
}

/// One residual per sample: model minus the reference signal.
struct Oscillator;

impl ResidualsFn for Oscillator {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, p: &[T]) -> Vec<T> {
        (0..TIMES.len())
            .map(|i| predict(p, TIMES[i]) - predict_f64(&TRUTH, TIMES[i]))
            .collect()
    }
}

fn main() {
    // [amplitude, damping, frequency, offset], starting away from the truth.
    let mut parameters = <V4 as DenseVector<f64>>::zeros(4);
    parameters.set_at(0, 2.2);
    parameters.set_at(1, 0.10);
    parameters.set_at(2, 1.55);
    parameters.set_at(3, 0.0);

    let mut options = Options::default();
    options.log.enable = false;
    options.stop.max_iters = 300;
    options.stop.max_consec_failures = 20;

    let summary =
        dl::Optimizer::<V4>::new(options).optimize(&mut parameters, ad_accumulation(&Oscillator));

    println!(
        "[amplitude, damping, frequency, offset]: [{}]",
        common::fmt_slice(DenseVector::as_slice(&parameters))
    );
    println!("Optimization succeeded: {}", summary.succeeded());

    std::process::exit(if summary.succeeded() { 0 } else { 1 });
}
