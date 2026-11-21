// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Estimate a sensor's 2D translation by aligning known local landmarks with
//! map points, with the yaw assumed known. Uses gradient descent.
//!
//! Mirrors `examples/cpp/robotics.cpp`.

mod common;

use tinyopt::backend::{DenseVector, Vector};
use tinyopt::diff::{CostFn, DualNum, cost_accumulation};
use tinyopt::optimizers::gd;
use tinyopt::{Options, Static};

type V2 = Vector<f64, Static<2>>;

const LANDMARKS: [[f64; 2]; 4] = [[-1.0, -0.5], [0.8, -0.3], [-0.4, 0.9], [1.2, 0.7]];
const SENSOR_YAW: f64 = 0.35;
const TRUE_TRANSLATION: [f64; 2] = [1.2, -0.7];

struct Alignment {
    cosine: f64,
    sine: f64,
    map_points: [[f64; 2]; 4],
}

impl CostFn for Alignment {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, estimate: &[T]) -> T {
        let mut error = estimate[0].clone() * 0.0;
        for (landmark, map_point) in LANDMARKS.iter().zip(&self.map_points) {
            let rotated_x = self.cosine * landmark[0] - self.sine * landmark[1];
            let rotated_y = self.sine * landmark[0] + self.cosine * landmark[1];
            let rx = estimate[0].clone() + rotated_x - map_point[0];
            let ry = estimate[1].clone() + rotated_y - map_point[1];
            error += rx.clone() * rx + ry.clone() * ry;
        }
        error
    }
}

fn main() {
    let cosine = SENSOR_YAW.cos();
    let sine = SENSOR_YAW.sin();
    let mut map_points = [[0.0; 2]; 4];
    for i in 0..LANDMARKS.len() {
        let rotated = [
            cosine * LANDMARKS[i][0] - sine * LANDMARKS[i][1],
            sine * LANDMARKS[i][0] + cosine * LANDMARKS[i][1],
        ];
        map_points[i] = [
            TRUE_TRANSLATION[0] + rotated[0],
            TRUE_TRANSLATION[1] + rotated[1],
        ];
    }

    let mut translation = <V2 as DenseVector<f64>>::zeros(2);

    let mut options = Options::default();
    options.log.enable = false;
    options.stop.max_iters = 300;
    options.stop.min_rerr_dec = 0.0;
    options.gd.lr = 0.05;

    let objective = Alignment {
        cosine,
        sine,
        map_points,
    };

    let summary =
        gd::Optimizer::<V2>::new(options).optimize(&mut translation, cost_accumulation(&objective));

    println!(
        "Estimated sensor translation: [{}]",
        common::fmt_slice(DenseVector::as_slice(&translation))
    );
    println!("Optimization succeeded: {}", summary.succeeded());

    std::process::exit(if summary.succeeded() { 0 } else { 1 });
}
