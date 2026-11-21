// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Fit a logistic population-growth curve; the optimized variables are
//! log(K) and log(r) so both stay positive.
//!
//! Mirrors `examples/cpp/biology.cpp`.

use tinyopt_rs::backend::{DenseVector, Vector};
use tinyopt_rs::diff::{CostFn, DualNum, cost_accumulation};
use tinyopt_rs::optimizers::bfgs;
use tinyopt_rs::{Options, Static};

type V2 = Vector<f64, Static<2>>;

const INITIAL_POPULATION: f64 = 80.0;
const TIMES: [f64; 6] = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
const POPULATION: [f64; 6] = [80.0, 179.3, 362.1, 618.2, 868.2, 1038.5];

struct Logistic;

impl CostFn for Logistic {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, p: &[T]) -> T {
        let carrying_capacity = p[0].clone().exp();
        let growth_rate = p[1].clone().exp();
        let one = T::from(1.0);
        let mut error = p[0].clone() * 0.0;
        for i in 0..TIMES.len() {
            let prediction = carrying_capacity.clone()
                / (one.clone()
                    + (carrying_capacity.clone() / INITIAL_POPULATION - one.clone())
                        * (-(growth_rate.clone() * TIMES[i])).exp());
            let residual = (prediction - POPULATION[i]) / 1000.0;
            error += residual.clone() * residual;
        }
        error
    }
}

fn main() {
    // Plausible biological starting point (K = 900, r = 0.35).
    let mut x = <V2 as DenseVector<f64>>::zeros(2);
    x.set_at(0, 900.0_f64.ln());
    x.set_at(1, 0.35_f64.ln());

    let mut options = Options::default();
    options.log.enable = false;
    options.stop.max_iters = 1500;
    options.stop.min_rerr_dec = 0.0;
    options.bfgs.step_size = 0.05;
    options.bfgs.max_step_size = 0.05;

    let summary =
        bfgs::Optimizer::<V2>::new(options).optimize(&mut x, cost_accumulation(&Logistic));

    println!("Estimated carrying capacity: {}", x.at(0).exp());
    println!("Estimated growth rate: {} per time unit", x.at(1).exp());
    println!("Optimization succeeded: {}", summary.succeeded());

    std::process::exit(if summary.succeeded() { 0 } else { 1 });
}
