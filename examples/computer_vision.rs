// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Estimate a pinhole camera's focal length and principal point from known 3D/2D
//! correspondences (no lens distortion).
//!
//! Mirrors `examples/cpp/computer_vision.cpp`.

mod common;

use tinyopt::backend::{DenseVector, Vector};
use tinyopt::diff::{DualNum, ResidualsFn, ad_accumulation};
use tinyopt::optimizers::gn;
use tinyopt::{Options, Static};

type V3 = Vector<f64, Static<3>>;

const POINTS: [[f64; 3]; 4] = [
    [-1.0, -0.5, 3.0],
    [0.8, -0.3, 2.5],
    [-0.4, 0.9, 4.0],
    [1.2, 0.7, 3.5],
];
const IMAGE_POINTS: [[f64; 2]; 4] = [
    [120.0, 140.0],
    [512.0, 168.0],
    [260.0, 375.0],
    [525.714286, 360.0],
];

/// Eight reprojection residuals: two image-coordinate errors per correspondence.
struct Camera;

impl ResidualsFn for Camera {
    type Scalar = f64;

    fn eval<T: DualNum<Primitive = f64>>(&self, p: &[T]) -> Vec<T> {
        let mut result = Vec::with_capacity(8);
        for i in 0..POINTS.len() {
            let (x, y, z) = (POINTS[i][0], POINTS[i][1], POINTS[i][2]);
            result.push(p[0].clone() * x / z + p[1].clone() - IMAGE_POINTS[i][0]);
            result.push(p[0].clone() * y / z + p[2].clone() - IMAGE_POINTS[i][1]);
        }
        result
    }
}

fn main() {
    // [focal length, principal x, principal y], starting away from the truth.
    let mut camera = <V3 as DenseVector<f64>>::zeros(3);
    camera.set_at(0, 650.0);
    camera.set_at(1, 280.0);
    camera.set_at(2, 200.0);

    let mut options = Options::default();
    options.log.enable = false;

    let summary = gn::Optimizer::<V3>::new(options).optimize(&mut camera, ad_accumulation(&Camera));

    println!(
        "Estimated [focal length, principal x, principal y]: [{}]",
        common::fmt_slice(DenseVector::as_slice(&camera))
    );
    println!("Optimization succeeded: {}", summary.succeeded());

    std::process::exit(if summary.succeeded() { 0 } else { 1 });
}
