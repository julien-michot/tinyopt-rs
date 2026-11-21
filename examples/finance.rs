// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Choose a long-only portfolio with a target expected return. A softmax
//! parameterization keeps the weights nonnegative and summing to one, while
//! L-BFGS minimizes the variance plus the return-target penalty.
//!
//! Mirrors `examples/cpp/finance.cpp`.

mod common;

use tinyopt_rs::backend::{DenseVector, Vector};
use tinyopt_rs::diff::{CostFn, DualNum, cost_accumulation};
use tinyopt_rs::optimizers::lbfgs;
use tinyopt_rs::{Options, Static};

type V2 = Vector<f64, Static<2>>;

const COVARIANCE: [[f64; 3]; 3] = [
    [0.040, 0.006, 0.004],
    [0.006, 0.090, 0.010],
    [0.004, 0.010, 0.160],
];
const EXPECTED_RETURNS: [f64; 3] = [0.05, 0.09, 0.14];
const TARGET_RETURN: f64 = 0.09;

struct Portfolio;

impl CostFn for Portfolio {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, logits: &[T]) -> T {
        let one = T::from(1.0);
        let mut scores = [logits[0].clone().exp(), logits[1].clone().exp(), one];
        let sum = scores[0].clone() + scores[1].clone() + scores[2].clone();
        for s in &mut scores {
            *s = s.clone() / sum.clone();
        }
        let allocation = scores;

        let mut variance = T::from(0.0);
        for i in 0..3 {
            for j in 0..3 {
                variance += allocation[i].clone() * COVARIANCE[i][j] * allocation[j].clone();
            }
        }
        let budget_error =
            allocation[0].clone() + allocation[1].clone() + allocation[2].clone() - 1.0;
        let mut expected = T::from(0.0);
        for i in 0..3 {
            expected += allocation[i].clone() * EXPECTED_RETURNS[i];
        }
        let return_error = expected - TARGET_RETURN;

        T::from(0.5) * variance
            + T::from(100.0) * budget_error.clone() * budget_error
            + T::from(100.0) * return_error.clone() * return_error
    }
}

fn main() {
    // Two free logits determine three weights (the third score is fixed at 1).
    let mut logits = <V2 as DenseVector<f64>>::zeros(2);

    let mut options = Options::default();
    options.log.enable = false;
    options.stop.max_iters = 2000;
    options.stop.min_rerr_dec = 0.0;
    options.lbfgs.step_size = 0.1;
    options.lbfgs.max_step_size = 0.5;

    let summary =
        lbfgs::Optimizer::<V2>::new(options).optimize(&mut logits, cost_accumulation(&Portfolio));

    let scores = [logits.at(0).exp(), logits.at(1).exp(), 1.0];
    let sum: f64 = scores.iter().sum();
    let weights = [scores[0] / sum, scores[1] / sum, scores[2] / sum];
    let expected_return: f64 = (0..3).map(|i| EXPECTED_RETURNS[i] * weights[i]).sum();

    println!(
        "Long-only portfolio weights: [{}]",
        common::fmt_slice(&weights)
    );
    println!("Expected return: {expected_return:.6}");
    println!("Optimization succeeded: {}", summary.succeeded());

    std::process::exit(if summary.succeeded() { 0 } else { 1 });
}
