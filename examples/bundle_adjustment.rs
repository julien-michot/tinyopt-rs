// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Toy bundle adjustment with a scalar sparse Hessian and a Schur reduced block.
//!
//! Mirrors `examples/cpp/bundle_adjustment.cpp`: two cameras (6 parameters each)
//! are the reduced head block and three landmarks (3 parameters each) are the
//! eliminated block.

use tinyopt_rs::backend::{DenseVector, Dynamic, Vector};
use tinyopt_rs::sparse::SparseMat;
use tinyopt_rs::{Cost, Options, Params, Solver, optimize_sparse};

type Vx = Vector<f64, Dynamic>;

const NCAM: usize = 2;
const NPOINT: usize = 3;
const CAM: usize = 6;
const POINT: usize = 3;
const DIMS: usize = NCAM * CAM + NPOINT * POINT;

/// Camera and landmark parameters, with the cameras as the reduced head block.
struct Scene {
    values: Vec<f64>,
}

impl Scene {
    fn new() -> Self {
        Self {
            values: vec![0.0; DIMS],
        }
    }

    fn camera(&self, i: usize) -> &[f64] {
        &self.values[i * CAM..(i + 1) * CAM]
    }

    fn point(&self, j: usize) -> &[f64] {
        &self.values[NCAM * CAM + j * POINT..NCAM * CAM + (j + 1) * POINT]
    }
}

impl Params for Scene {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dims(&self) -> usize {
        DIMS
    }

    fn as_slice(&self) -> &[f64] {
        &self.values
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        for (v, d) in self.values.iter_mut().zip(delta) {
            *v += *d;
        }
    }

    fn norm2(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum()
    }

    fn reduced(&self) -> Option<(usize, usize)> {
        Some((0, NCAM * CAM))
    }
}

/// Ground-truth cameras and points used to generate the observations.
fn truth() -> Scene {
    let mut s = Scene::new();
    for i in 0..NCAM {
        s.values[i * CAM] = 0.2 + 0.1 * i as f64; // translation x
        s.values[i * CAM + 1] = -0.15 * i as f64; // translation y
        s.values[i * CAM + 2] = 1.0;
    }
    for j in 0..NPOINT {
        s.values[NCAM * CAM + j * POINT] = 0.5 * (j as f64) - 0.4;
        s.values[NCAM * CAM + j * POINT + 1] = 0.3 * (j as f64) - 0.2;
        s.values[NCAM * CAM + j * POINT + 2] = 0.9;
    }
    s
}

fn accumulate(x: &[f64], grad: Option<&mut Vx>, h: Option<&mut SparseMat>) -> Cost {
    let mut cost = 0.0;
    let mut g = grad;
    let mut h = h;

    let mut add_row = |row: &[(usize, f64)], rhs: f64, cost: &mut f64| {
        let mut value = 0.0;
        for &(i, j) in row {
            value += j * x[i];
        }
        let r = value - rhs;
        *cost += r * r;
        if let Some(g) = g.as_mut() {
            for &(i, j) in row {
                g.set_at(i, g.at(i) + j * r);
            }
        }
        if let Some(h) = h.as_mut() {
            for &(i, ji) in row {
                for &(k, jk) in row {
                    *h.coeff_ref(i, k) += ji * jk;
                }
            }
        }
    };

    let truth = truth();
    // Priors keep the gauge fixed.
    for i in 0..NCAM * CAM {
        add_row(&[(i, 1.0)], truth.values[i], &mut cost);
    }
    for j in 0..NPOINT * POINT {
        let i = NCAM * CAM + j;
        add_row(&[(i, 1.0)], truth.values[i], &mut cost);
    }
    // Reprojection-like observations: camera translation + point position.
    for i in 0..NCAM {
        let cam = i * CAM;
        for j in 0..NPOINT {
            let pt = NCAM * CAM + j * POINT;
            let cx = truth.camera(i)[0] + truth.point(j)[0];
            let cy = truth.camera(i)[1] + truth.point(j)[1];
            add_row(&[(cam, 1.0), (pt, 1.0)], cx, &mut cost);
            add_row(&[(cam + 1, 1.0), (pt + 1, 1.0)], cy, &mut cost);
        }
    }

    Cost::with_num_residuals(
        cost,
        (NCAM * CAM + NPOINT * POINT + NCAM * NPOINT * 2) as i32,
    )
}

fn main() {
    let mut scene = Scene::new();
    let mut options = Options::new(Solver::GaussNewton);
    options.log.enable = false;

    let summary = optimize_sparse(&mut scene, accumulate, &options);
    let truth = truth();
    let max_error = scene
        .values
        .iter()
        .zip(&truth.values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    println!(
        "Bundle adjustment: {} cameras ({} reduced), {} points, {} parameters",
        NCAM,
        NCAM * CAM,
        NPOINT,
        DIMS
    );
    println!("Maximum parameter error: {max_error:.3e}");
    println!("Optimization succeeded: {}", summary.succeeded());

    std::process::exit(if summary.succeeded() { 0 } else { 1 });
}
