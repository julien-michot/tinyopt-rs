// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Recover one-dimensional astrometry parameters from simulated observations:
//! reference position, proper motion, and annual parallax.
//!
//! Mirrors `examples/cpp/astronomy.cpp`.

mod common;

use tinyopt::backend::{DenseVector, Vector};
use tinyopt::diff::{DualNum, ResidualsFn, ad_accumulation};
use tinyopt::optimizers::lm;
use tinyopt::{Options, Static};

type V3 = Vector<f64, Static<3>>;

const EPOCHS: [f64; 4] = [-1.5, -0.5, 0.5, 1.5];
const PARALLAX_FACTORS: [f64; 4] = [1.0, 0.0, -1.0, 0.0];
const MEASURED_POSITIONS: [f64; 4] = [0.075, 0.085, 0.095, 0.145];

/// residual_i = position + motion * t_i + parallax * factor_i - measured_i
struct Astrometry;

impl ResidualsFn for Astrometry {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, p: &[T]) -> Vec<T> {
        (0..EPOCHS.len())
            .map(|i| {
                p[0].clone() + p[1].clone() * EPOCHS[i] + p[2].clone() * PARALLAX_FACTORS[i]
                    - MEASURED_POSITIONS[i]
            })
            .collect()
    }
}

fn main() {
    // Reference position, proper motion, parallax.
    let mut astrometry = <V3 as DenseVector<f64>>::zeros(3);

    let mut options = Options::default();
    options.log.enable = false;

    let summary =
        lm::Optimizer::<V3>::new(options).optimize(&mut astrometry, ad_accumulation(&Astrometry));

    println!(
        "[position, proper motion, parallax]: [{}]",
        common::fmt_slice(DenseVector::as_slice(&astrometry))
    );
    println!("Optimization succeeded: {}", summary.succeeded());

    std::process::exit(if summary.succeeded() { 0 } else { 1 });
}
