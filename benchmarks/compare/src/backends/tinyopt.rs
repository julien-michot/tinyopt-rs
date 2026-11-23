// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! tinyopt-rs backend.

use tinyopt::backend::{DenseMatrix, DenseVector, Matrix, Vector};
use tinyopt::block_sparse::BlockSparseMat;
use tinyopt::block_sparse_types;
use tinyopt::block_tridiag::BlockTridiagMat;
use tinyopt::optimizers::lm;
use tinyopt::sparse::{SparseMat, SparseMatrix};
use tinyopt::{Cost, Dynamic, Params, Real, Static};

use num_traits::FromPrimitive;

use crate::harness::Outcome;
use crate::options::tinyopt_options;
use crate::problems::{
    Case, Kind, block_tridiag, bundle_adjustment as ba, dense, robust_pose as pose, sparse,
};

block_sparse_types!(BaTypes, [6, 3]);

#[cfg(feature = "rayon")]
mod parallel_ba;
#[cfg(feature = "rayon")]
pub(crate) use parallel_ba::{measure_ba_block_schur, run_ba_block_parallel};
#[cfg(feature = "wgpu")]
mod wgpu_ba;
#[cfg(feature = "wgpu")]
pub(crate) use wgpu_ba::{run_ba_block_wgpu, run_ba_block_wgpu_f64, wgpu_f64_supported};

pub fn run(case: &Case) -> Outcome {
    match case.kind {
        Kind::DenseMathStatic { dims, single } => run_dense(case, dims, single, false),
        Kind::DenseMathDynamic { dims, single } => run_dense(case, dims, single, true),
        Kind::DensePrior { dims } => run_prior(case, dims),
        Kind::SparseChain { dims } => run_sparse_chain(case, dims),
        Kind::BundleAdjustmentBlock { cams, points } => run_ba_block(case, cams, points),
        Kind::BundleAdjustmentSparse { cams, points } => run_ba_sparse(case, cams, points),
        Kind::BlockTridiag { blocks } => run_block_tridiag(case, blocks),
        Kind::RobustPose { obs, robust } => run_pose(case, obs, robust),
    }
}

fn outcome(summary: &tinyopt::Summary) -> Outcome {
    Outcome {
        converged: summary.converged(),
        iterations: summary.num_iters as u32,
        final_cost: summary.final_cost.cost,
    }
}

// --- Dense ----------------------------------------------------------------

fn dense_vec_acc<S: Real, D: tinyopt::backend::Dim>(
    x: &[S],
    grad: Option<&mut Vector<S, D>>,
    h: Option<&mut Matrix<S, D>>,
) -> Cost {
    let xf: Vec<f64> = x.iter().map(|v| v.to_f64().expect("scalar")).collect();
    let r = dense::math_residuals(&xf);
    let n = r.len();
    let j = dense::math_jacobian(&xf);
    let mut cost = 0.0;
    for value in &r {
        cost += 0.5 * value * value;
    }
    if let Some(g) = grad {
        for i in 0..n {
            let acc: f64 = (0..n).map(|k| j[k * n + i] * r[k]).sum();
            g.set_at(i, S::from_f64(acc).expect("fits"));
        }
    }
    if let Some(h) = h {
        for i in 0..n {
            for l in 0..n {
                let acc: f64 = (0..n).map(|k| j[k * n + i] * j[k * n + l]).sum();
                h.set_elem(i, l, S::from_f64(acc).expect("fits"));
            }
        }
    }
    Cost::with_num_residuals(cost, n as i32)
}

fn run_dense(case: &Case, dims: usize, single: bool, dynamic: bool) -> Outcome {
    let options = tinyopt_options(&case.kind);
    macro_rules! run_vec {
        ($s:ty, $d:ty, $initial:expr) => {{
            type V = Vector<$s, $d>;
            type M = Matrix<$s, $d>;
            let mut opt = lm::Optimizer::<V, M>::new(options);
            let mut x = <V as DenseVector<$s>>::zeros(dims);
            for (i, value) in $initial.iter().enumerate() {
                x.set_at(i, <$s>::from_f64(*value).expect("fits"));
            }
            let acc =
                |xv: &[$s], g: Option<&mut V>, h: Option<&mut M>| dense_vec_acc::<$s, $d>(xv, g, h);
            let summary = opt.optimize(&mut x, acc);
            outcome(&summary)
        }};
    }

    let initial = dense::math_initial(dims);
    match (single, dynamic) {
        (false, false) => match dims {
            1 => run_vec!(f64, Static<1>, initial),
            2 => run_vec!(f64, Static<2>, initial),
            3 => run_vec!(f64, Static<3>, initial),
            _ => unreachable!(),
        },
        (true, false) => match dims {
            1 => run_vec!(f32, Static<1>, initial),
            2 => run_vec!(f32, Static<2>, initial),
            3 => run_vec!(f32, Static<3>, initial),
            _ => unreachable!(),
        },
        (false, true) => run_vec!(f64, Dynamic, initial),
        (true, true) => run_vec!(f32, Dynamic, initial),
    }
}

fn run_prior(case: &Case, dims: usize) -> Outcome {
    type V = Vector<f64, Dynamic>;
    type M = Matrix<f64, Dynamic>;
    let target = dense::prior_target(dims);
    let mut opt = lm::Optimizer::<V, M>::new(tinyopt_options(&case.kind));
    let mut x = <V as DenseVector<f64>>::zeros(dims);
    for (i, value) in dense::prior_initial(dims).iter().enumerate() {
        x.set_at(i, *value);
    }
    let acc = |xv: &[f64], grad: Option<&mut V>, h: Option<&mut M>| {
        let mut cost = 0.0;
        let mut g = grad;
        let mut h = h;
        for i in 0..dims {
            let r = xv[i] - target[i];
            cost += 0.5 * r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(i, r);
            }
            if let Some(h) = h.as_mut() {
                h.set_elem(i, i, 1.0);
            }
        }
        Cost::with_num_residuals(cost, dims as i32)
    };
    let summary = opt.optimize(&mut x, acc);
    outcome(&summary)
}

// --- Sparse ---------------------------------------------------------------

fn run_sparse_chain(case: &Case, dims: usize) -> Outcome {
    type V = Vector<f64, Dynamic>;
    let target: Vec<f64> = (0..dims).map(sparse::target).collect();
    let difference: Vec<f64> = (0..dims).map(sparse::difference_target).collect();
    let mut opt = lm::Optimizer::<V, SparseMat>::new(tinyopt_options(&case.kind));
    let mut x = <V as DenseVector<f64>>::zeros(dims);
    for (i, value) in sparse::initial_vector(dims).iter().enumerate() {
        x.set_at(i, *value);
    }
    let acc = |xv: &[f64], grad: Option<&mut V>, h: Option<&mut SparseMat>| {
        let mut cost = 0.0;
        let mut g = grad;
        let mut h = h;
        for i in 0..dims {
            let r = xv[i] - target[i];
            cost += r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(i, g.at(i) + r);
            }
            if let Some(h) = h.as_mut() {
                *h.coeff_ref(i, i) += 1.0;
            }
        }
        for i in 0..dims.saturating_sub(1) {
            let r = 0.1 * ((xv[i + 1] - xv[i]) - difference[i + 1]);
            cost += r * r;
            if let Some(g) = g.as_mut() {
                g.set_at(i, g.at(i) - 0.1 * r);
                g.set_at(i + 1, g.at(i + 1) + 0.1 * r);
            }
            if let Some(h) = h.as_mut() {
                *h.coeff_ref(i, i) += 0.01;
                *h.coeff_ref(i, i + 1) -= 0.01;
                *h.coeff_ref(i + 1, i) -= 0.01;
                *h.coeff_ref(i + 1, i + 1) += 0.01;
            }
        }
        Cost::with_num_residuals(cost, (2 * dims - 1) as i32)
    };
    let summary = opt.optimize(&mut x, acc);
    outcome(&summary)
}

// --- Bundle adjustment ------------------------------------------------------

#[derive(Clone, Debug)]
struct BaParams {
    /// Ambient SE(3) camera state (7 doubles each), followed by Euclidean points.
    values: Vec<f64>,
    camera_count: usize,
}

impl Params for BaParams {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dims(&self) -> usize {
        6 * self.camera_count + (self.values.len() - 7 * self.camera_count)
    }

    fn as_slice(&self) -> &[f64] {
        &self.values
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        for camera in 0..self.camera_count {
            let mut xi = [0.0; 6];
            xi.copy_from_slice(&delta[camera * 6..camera * 6 + 6]);
            let start = camera * 7;
            let camera_value = [
                self.values[start],
                self.values[start + 1],
                self.values[start + 2],
                self.values[start + 3],
                self.values[start + 4],
                self.values[start + 5],
                self.values[start + 6],
            ];
            let updated = ba::se3_plus(&camera_value, &xi);
            self.values[start..start + 7].copy_from_slice(&updated);
        }
        let camera_offset = 7 * self.camera_count;
        let point_offset = 6 * self.camera_count;
        let point_count = (self.values.len() - camera_offset) / 3;
        for point in 0..point_count {
            let value_start = camera_offset + point * 3;
            let delta_start = point_offset + point * 3;
            for axis in 0..3 {
                self.values[value_start + axis] += delta[delta_start + axis];
            }
        }
    }

    fn norm2(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum()
    }

    fn reduced(&self) -> Option<(usize, usize)> {
        Some((0, 6 * self.camera_count))
    }

    fn fixed(&self) -> Option<Vec<usize>> {
        let camera_offset = 6 * self.camera_count;
        let mut fixed: Vec<usize> = (0..6).collect();
        fixed.extend(camera_offset..camera_offset + 3);
        Some(fixed)
    }
}

fn ba_params(problem: &ba::Problem) -> BaParams {
    let mut values = Vec::with_capacity(7 * problem.camera_count() + 3 * problem.point_count());
    for camera in &problem.initial_cameras {
        values.extend_from_slice(camera);
    }
    for point in &problem.initial_points {
        values.extend_from_slice(point);
    }
    BaParams {
        values,
        camera_count: problem.camera_count(),
    }
}

fn run_ba_block(case: &Case, cams: usize, points: usize) -> Outcome {
    type V = Vector<f64, Dynamic>;
    let problem = ba::make_problem(cams, points);
    let ambient_camera_offset = 7 * problem.camera_count();
    let tangent_camera_offset = 6 * problem.camera_count();

    let mut opt =
        lm::Optimizer::<BaParams, BlockSparseMat<BaTypes, f64>>::new(tinyopt_options(&case.kind));
    opt.init_with(BlockSparseMat::with_counts(&[
        problem.camera_count(),
        problem.point_count(),
    ]));
    let mut x = ba_params(&problem);

    let acc = |xv: &[f64], grad: Option<&mut V>, h: Option<&mut BlockSparseMat<BaTypes, f64>>| {
        let mut cost = 0.0;
        let mut g = grad;
        let mut h = h;
        for observation in &problem.observations {
            let camera_start = observation.camera * 7;
            let camera = [
                xv[camera_start],
                xv[camera_start + 1],
                xv[camera_start + 2],
                xv[camera_start + 3],
                xv[camera_start + 4],
                xv[camera_start + 5],
                xv[camera_start + 6],
            ];
            let point_start = ambient_camera_offset + observation.point * 3;
            let point = [xv[point_start], xv[point_start + 1], xv[point_start + 2]];
            let (jc, jp) = ba::projection_jacobians(&camera, &point);
            let projected = ba::project(&camera, &point);
            let r = [
                projected[0] - observation.measurement[0],
                projected[1] - observation.measurement[1],
            ];
            cost += r[0] * r[0] + r[1] * r[1];
            if let Some(g) = g.as_mut() {
                for i in 0..6 {
                    let acc: f64 = (0..2).map(|k| jc[k * 6 + i] * r[k]).sum();
                    g.set_at(
                        observation.camera * 6 + i,
                        g.at(observation.camera * 6 + i) + acc,
                    );
                }
                for i in 0..3 {
                    let acc: f64 = (0..2).map(|k| jp[k * 3 + i] * r[k]).sum();
                    g.set_at(
                        tangent_camera_offset + observation.point * 3 + i,
                        g.at(tangent_camera_offset + observation.point * 3 + i) + acc,
                    );
                }
            }
            if let Some(h) = h.as_mut() {
                h.block_mut::<6, 6>(observation.camera, observation.camera)
                    .add_jtj(1.0, &jc, 2);
                h.block_mut::<3, 3>(observation.point, observation.point)
                    .add_jtj(1.0, &jp, 2);
                let mut block = h.block_mut::<6, 3>(observation.camera, observation.point);
                for k in 0..2 {
                    block.add_outer_product(1.0, &jc[k * 6..k * 6 + 6], &jp[k * 3..k * 3 + 3]);
                }
            }
        }
        Cost::with_num_residuals(0.5 * cost, (2 * problem.observations.len()) as i32)
    };

    let summary = opt.optimize(&mut x, acc);
    outcome(&summary)
}

fn run_ba_sparse(case: &Case, cams: usize, points: usize) -> Outcome {
    type V = Vector<f64, Dynamic>;
    let problem = ba::make_problem(cams, points);
    let ambient_camera_offset = 7 * problem.camera_count();
    let tangent_camera_offset = 6 * problem.camera_count();
    let mut opt = lm::Optimizer::<BaParams, SparseMat>::new(tinyopt_options(&case.kind));
    let mut x = ba_params(&problem);
    // Reuse the triplet buffer across evaluations (the accumulation API is `Fn`).
    let buffer = std::cell::RefCell::new(Vec::with_capacity(
        problem.observations.len() * (36 + 9 + 18),
    ));

    let acc = |xv: &[f64], grad: Option<&mut V>, h: Option<&mut SparseMat>| {
        let mut cost = 0.0;
        let mut g = grad;
        let mut triplets = buffer.borrow_mut();
        triplets.clear();
        for observation in &problem.observations {
            let camera_index = observation.camera;
            let point_index = observation.point;
            let camera_start = camera_index * 7;
            let camera = [
                xv[camera_start],
                xv[camera_start + 1],
                xv[camera_start + 2],
                xv[camera_start + 3],
                xv[camera_start + 4],
                xv[camera_start + 5],
                xv[camera_start + 6],
            ];
            let point_start = ambient_camera_offset + point_index * 3;
            let point = [xv[point_start], xv[point_start + 1], xv[point_start + 2]];
            let (jc, jp) = ba::projection_jacobians(&camera, &point);
            let projected = ba::project(&camera, &point);
            let r = [
                projected[0] - observation.measurement[0],
                projected[1] - observation.measurement[1],
            ];
            cost += r[0] * r[0] + r[1] * r[1];
            if let Some(g) = g.as_mut() {
                for i in 0..6 {
                    let acc: f64 = (0..2).map(|k| jc[k * 6 + i] * r[k]).sum();
                    g.set_at(camera_index * 6 + i, g.at(camera_index * 6 + i) + acc);
                }
                for i in 0..3 {
                    let acc: f64 = (0..2).map(|k| jp[k * 3 + i] * r[k]).sum();
                    g.set_at(
                        tangent_camera_offset + point_index * 3 + i,
                        g.at(tangent_camera_offset + point_index * 3 + i) + acc,
                    );
                }
            }
            // Accumulate the full J^T J blocks once (duplicates across
            // observations are summed by `from_triplets`).
            if h.is_some() {
                for i in 0..6 {
                    for j in 0..6 {
                        let acc: f64 = (0..2).map(|row| jc[row * 6 + i] * jc[row * 6 + j]).sum();
                        triplets.push((camera_index * 6 + i, camera_index * 6 + j, acc));
                    }
                }
                for i in 0..3 {
                    for j in 0..3 {
                        let acc: f64 = (0..2).map(|row| jp[row * 3 + i] * jp[row * 3 + j]).sum();
                        let point_tangent = tangent_camera_offset + point_index * 3;
                        triplets.push((point_tangent + i, point_tangent + j, acc));
                    }
                }
                for i in 0..6 {
                    for j in 0..3 {
                        let acc: f64 = (0..2).map(|row| jc[row * 6 + i] * jp[row * 3 + j]).sum();
                        let point_tangent = tangent_camera_offset + point_index * 3;
                        triplets.push((camera_index * 6 + i, point_tangent + j, acc));
                        triplets.push((point_tangent + j, camera_index * 6 + i, acc));
                    }
                }
            }
        }
        if let Some(h) = h {
            *h = SparseMatrix::from_triplets(
                tangent_camera_offset + 3 * problem.point_count(),
                tangent_camera_offset + 3 * problem.point_count(),
                &triplets,
            );
        }
        Cost::with_num_residuals(0.5 * cost, (2 * problem.observations.len()) as i32)
    };

    let summary = opt.optimize(&mut x, acc);
    outcome(&summary)
}

// --- Block-tridiagonal pose chain --------------------------------------------

fn run_block_tridiag(case: &Case, blocks: usize) -> Outcome {
    const B: usize = block_tridiag::BLOCK;
    type V = Vector<f64, Dynamic>;
    let mut x = vec![0.0; blocks * B];
    let acc = |xv: &[f64], grad: Option<&mut V>, h: Option<&mut BlockTridiagMat<B, f64>>| {
        let mut cost = 0.0;
        let mut g = grad;
        let mut h = h;
        for i in 0..blocks {
            for k in 0..B {
                let r = xv[i * B + k] - block_tridiag::target(i, k);
                cost += r * r;
                if let Some(g) = g.as_mut() {
                    g.set_at(i * B + k, g.at(i * B + k) + r);
                }
                if let Some(h) = h.as_mut() {
                    let mut e = [0.0; B];
                    e[k] = 1.0;
                    h.diagonal(i).add_outer_product_self(1.0, &e);
                }
            }
        }
        for i in 0..blocks.saturating_sub(1) {
            for k in 0..B {
                let r = (xv[(i + 1) * B + k] - xv[i * B + k]) - block_tridiag::delta(i, k);
                cost += r * r;
                if let Some(g) = g.as_mut() {
                    g.set_at(i * B + k, g.at(i * B + k) - r);
                    g.set_at((i + 1) * B + k, g.at((i + 1) * B + k) + r);
                }
                if let Some(h) = h.as_mut() {
                    let mut e = [0.0; B];
                    e[k] = 1.0;
                    h.diagonal(i).add_outer_product_self(1.0, &e);
                    h.diagonal(i + 1).add_outer_product_self(1.0, &e);
                    h.off_diagonal(i).add_outer_product(-1.0, &e, &e);
                }
            }
        }
        Cost::with_num_residuals(0.5 * cost, (2 * blocks * B) as i32)
    };
    let summary = tinyopt::optimize_tridiag::<B, _, _>(&mut x, acc, &tinyopt_options(&case.kind));
    outcome(&summary)
}

// --- Robust pose -------------------------------------------------------------

#[derive(Clone, Debug)]
struct Se3 {
    pose: pose::Pose,
    buffer: [f64; 7],
}

impl Se3 {
    fn new(pose: pose::Pose) -> Self {
        Self {
            buffer: pose.params(),
            pose,
        }
    }
}

impl Params for Se3 {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dims(&self) -> usize {
        6
    }

    fn as_slice(&self) -> &[f64] {
        &self.buffer
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        let mut xi = [0.0; 6];
        xi.copy_from_slice(&delta[..6]);
        self.pose = self.pose.plus(&xi);
        self.buffer = self.pose.params();
    }

    fn norm2(&self) -> f64 {
        self.buffer.iter().map(|v| v * v).sum()
    }
}

fn run_pose(case: &Case, observations: usize, robust: bool) -> Outcome {
    const DELTA: f64 = 0.3;
    let ground_truth = pose::ground_truth();
    let outliers = observations / 4;
    let data =
        pose::make_observations(&ground_truth, observations - outliers, outliers, 0.01, 50.0);
    let mut opt = lm::Optimizer::<Se3, Matrix<f64, Dynamic>>::new(tinyopt_options(&case.kind));
    let mut x = Se3::new(pose::initial_pose(&ground_truth));

    let acc = |xv: &[f64],
               grad: Option<&mut Vector<f64, Dynamic>>,
               h: Option<&mut Matrix<f64, Dynamic>>| {
        let current = pose::Pose {
            q: [xv[0], xv[1], xv[2], xv[3]],
            t: [xv[4], xv[5], xv[6]],
        };
        let mut cost = 0.0;
        let mut g = grad;
        let mut h = h;
        if let Some(h) = h.as_mut() {
            h.set_zero();
        }
        for observation in &data {
            let transformed = current.transform(&observation.world_point);
            let r = [
                transformed[0] - observation.measurement[0],
                transformed[1] - observation.measurement[1],
                transformed[2] - observation.measurement[2],
            ];
            let squared_norm = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
            let (scale, scale_derivative) = if robust && squared_norm > DELTA * DELTA {
                let inverse_norm = 1.0 / squared_norm.sqrt();
                let scale = (2.0 * DELTA * inverse_norm
                    - DELTA * DELTA * inverse_norm * inverse_norm)
                    .sqrt();
                let derivative =
                    -DELTA * inverse_norm.powi(3) + DELTA * DELTA * inverse_norm.powi(4);
                cost += 0.5 * scale * scale * squared_norm;
                (scale, derivative)
            } else {
                cost += 0.5 * squared_norm;
                (1.0, 0.0)
            };

            let jp = pose::pose_jacobian(&current, &observation.world_point);
            // J = J_r * J_pose, with J_r = scale I + (dscale/scale) r rᵀ.
            let mut j = [0.0; 18];
            for row in 0..3 {
                for col in 0..6 {
                    let mut value = scale * jp[row * 6 + col];
                    if scale_derivative != 0.0 {
                        let factor = (scale_derivative / scale) * r[row];
                        for k in 0..3 {
                            value += factor * r[k] * jp[k * 6 + col];
                        }
                    }
                    j[row * 6 + col] = value;
                }
            }
            if let Some(g) = g.as_mut() {
                for col in 0..6 {
                    let acc: f64 = (0..3).map(|row| j[row * 6 + col] * scale * r[row]).sum();
                    g.set_at(col, g.at(col) + acc);
                }
            }
            if let Some(h) = h.as_mut() {
                for row in 0..6 {
                    for col in 0..6 {
                        let acc: f64 = (0..3).map(|k| j[k * 6 + row] * j[k * 6 + col]).sum();
                        h.set_elem(row, col, h.elem(row, col) + acc);
                    }
                }
            }
        }
        Cost::with_num_residuals(cost, (3 * data.len()) as i32)
    };

    let summary = opt.optimize(&mut x, acc);
    outcome(&summary)
}
