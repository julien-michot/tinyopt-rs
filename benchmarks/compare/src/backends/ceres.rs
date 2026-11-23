// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Ceres Solver backend (Rust binding).
//!
//! The safe binding has no manifold API, so camera poses use normalized
//! quaternion + translation blocks. The residual Jacobian accounts for
//! quaternion normalization, leaving its radial direction null.

use ceres_solver::solver::{LinearSolverType, LoggingType, SolverOptions, TrustRegionStrategyType};
use ceres_solver::{CostFunctionType, LossFunction, NllsProblem, ParameterBlockOrIndex};

use crate::harness::Outcome;
use crate::problems::{
    Case, Kind, block_tridiag, bundle_adjustment as ba, dense, robust_pose as pose, sparse,
};

fn options(linear_solver: LinearSolverType, max_invalid_steps: i32) -> SolverOptions {
    SolverOptions::builder()
        .linear_solver_type(linear_solver)
        .trust_region_strategy_type(TrustRegionStrategyType::LEVENBERG_MARQUARDT)
        .initial_trust_region_radius(1e4)
        .max_num_iterations(100)
        .max_num_consecutive_invalid_steps(max_invalid_steps)
        .num_threads(1)
        .function_tolerance(1e-6)
        .gradient_tolerance(1e-9)
        .parameter_tolerance(1e-8)
        .min_relative_decrease(1e-12)
        .logging_type(LoggingType::SILENT)
        .build()
        .expect("valid Ceres options")
}

fn outcome(solution: &ceres_solver::nlls_problem::NllsProblemSolution) -> Outcome {
    let summary = &solution.summary;
    let iterations =
        (summary.num_successful_steps() + summary.num_unsuccessful_steps()).max(0) as u32;
    Outcome {
        converged: summary.is_solution_usable() && summary.brief_report().contains("CONVERGENCE"),
        iterations,
        final_cost: summary.final_cost(),
    }
}

pub fn run(case: &Case) -> Outcome {
    match case.kind {
        Kind::DenseMathStatic { dims, .. } | Kind::DenseMathDynamic { dims, .. } => run_dense(dims),
        Kind::DensePrior { dims } => run_prior(dims),
        Kind::SparseChain { dims } => run_sparse_chain(dims),
        Kind::BundleAdjustmentBlock { cams, points }
        | Kind::BundleAdjustmentSparse { cams, points } => run_ba(cams, points),
        Kind::BlockTridiag { blocks } => run_block_tridiag(blocks),
        Kind::RobustPose { obs, robust } => run_pose(obs, robust),
    }
}

fn run_dense(dims: usize) -> Outcome {
    let initial = dense::math_initial(dims);
    let cost: CostFunctionType = Box::new(move |parameters, residuals, jacobians| {
        let x = parameters[0];
        let r = dense::math_residuals(x);
        residuals.copy_from_slice(&r);
        if let Some(jacobians) = jacobians
            && let Some(block) = &mut jacobians[0]
        {
            let j = dense::math_jacobian(x);
            for row in 0..dims {
                for col in 0..dims {
                    block[row][col] = j[row * dims + col];
                }
            }
        }
        true
    });
    let problem = NllsProblem::new()
        .residual_block_builder()
        .set_cost(cost, dims)
        .set_parameters(vec![initial])
        .build_into_problem()
        .expect("Ceres problem")
        .0;
    let solution = problem
        .solve(&options(LinearSolverType::DENSE_NORMAL_CHOLESKY, 3))
        .expect("Ceres solve");
    outcome(&solution)
}

fn run_prior(dims: usize) -> Outcome {
    let initial = dense::prior_initial(dims);
    let target = dense::prior_target(dims);
    let cost: CostFunctionType = Box::new(move |parameters, residuals, jacobians| {
        let x = parameters[0];
        for i in 0..dims {
            residuals[i] = x[i] - target[i];
        }
        if let Some(jacobians) = jacobians
            && let Some(block) = &mut jacobians[0]
        {
            for row in 0..dims {
                for col in 0..dims {
                    block[row][col] = if row == col { 1.0 } else { 0.0 };
                }
            }
        }
        true
    });
    let problem = NllsProblem::new()
        .residual_block_builder()
        .set_cost(cost, dims)
        .set_parameters(vec![initial])
        .build_into_problem()
        .expect("Ceres problem")
        .0;
    let solution = problem
        .solve(&options(LinearSolverType::DENSE_NORMAL_CHOLESKY, 3))
        .expect("Ceres solve");
    outcome(&solution)
}

fn run_sparse_chain(dims: usize) -> Outcome {
    let target: Vec<f64> = (0..dims).map(sparse::target).collect();
    let difference: Vec<f64> = (0..dims).map(sparse::difference_target).collect();
    let mut problem = NllsProblem::new();

    // One parameter block per scalar, unary residual per parameter: this is the
    // structure of the C++ Ceres benchmark and exposes the true sparsity.
    for (i, &target_i) in target.iter().enumerate() {
        let value = sparse::initial(i);
        let cost: CostFunctionType = Box::new(move |parameters, residuals, jacobians| {
            residuals[0] = parameters[0][0] - target_i;
            if let Some(jacobians) = jacobians
                && let Some(block) = &mut jacobians[0]
            {
                block[0][0] = 1.0;
            }
            true
        });
        problem = problem
            .residual_block_builder()
            .set_cost(cost, 1)
            .set_parameters([ParameterBlockOrIndex::Block(vec![value].into())])
            .build_into_problem()
            .expect("Ceres residual block")
            .0;
    }
    for (i, &difference_i) in difference.iter().enumerate().skip(1) {
        let cost: CostFunctionType = Box::new(move |parameters, residuals, jacobians| {
            residuals[0] = 0.1 * ((parameters[1][0] - parameters[0][0]) - difference_i);
            if let Some(jacobians) = jacobians {
                if let Some(block) = &mut jacobians[0] {
                    block[0][0] = -0.1;
                }
                if let Some(block) = &mut jacobians[1] {
                    block[0][0] = 0.1;
                }
            }
            true
        });
        problem = problem
            .residual_block_builder()
            .set_cost(cost, 1)
            .set_parameters([
                ParameterBlockOrIndex::Index(i - 1),
                ParameterBlockOrIndex::Index(i),
            ])
            .build_into_problem()
            .expect("Ceres residual block")
            .0;
    }

    let solution = problem
        .solve(&options(LinearSolverType::SPARSE_NORMAL_CHOLESKY, 3))
        .expect("Ceres solve");
    outcome(&solution)
}

fn run_ba(cams: usize, points: usize) -> Outcome {
    let problem_data = ba::make_problem(cams, points);
    let cameras: Vec<Vec<f64>> = problem_data
        .initial_cameras
        .iter()
        .map(|camera| camera.to_vec())
        .collect();
    let points_data: Vec<Vec<f64>> = problem_data
        .initial_points
        .iter()
        .map(|point| point.to_vec())
        .collect();
    let observations = &problem_data.observations;

    let mut camera_index: Vec<Option<usize>> = vec![None; cameras.len()];
    let mut point_index: Vec<Option<usize>> = vec![None; points_data.len()];
    let mut next_index = 0usize;
    let mut problem = NllsProblem::new();

    for observation in observations {
        let measurement = observation.measurement;
        let cost: CostFunctionType = Box::new(move |parameters, residuals, jacobians| {
            let camera: [f64; 7] = parameters[0].try_into().expect("7 SE(3) camera params");
            let point: [f64; 3] = parameters[1].try_into().expect("3 point params");
            let projected = ba::project(&camera, &point);
            residuals[0] = projected[0] - measurement[0];
            residuals[1] = projected[1] - measurement[1];
            if let Some(jacobians) = jacobians {
                let (jc, jp) = ba::projection_jacobians_ambient(&camera, &point);
                if let Some(block) = &mut jacobians[0] {
                    for row in 0..2 {
                        for col in 0..7 {
                            block[row][col] = jc[row * 7 + col];
                        }
                    }
                }
                if let Some(block) = &mut jacobians[1] {
                    for row in 0..2 {
                        for col in 0..3 {
                            block[row][col] = jp[row * 3 + col];
                        }
                    }
                }
            }
            true
        });

        let camera_param = match camera_index[observation.camera] {
            Some(index) => ParameterBlockOrIndex::Index(index),
            None => {
                let index = next_index;
                next_index += 1;
                camera_index[observation.camera] = Some(index);
                ParameterBlockOrIndex::Block(cameras[observation.camera].clone().into())
            }
        };
        let point_param = match point_index[observation.point] {
            Some(index) => ParameterBlockOrIndex::Index(index),
            None => {
                let index = next_index;
                next_index += 1;
                point_index[observation.point] = Some(index);
                ParameterBlockOrIndex::Block(points_data[observation.point].clone().into())
            }
        };

        let (next_problem, _) = problem
            .residual_block_builder()
            .set_cost(cost, 2)
            .set_parameters([camera_param, point_param])
            .build_into_problem()
            .expect("Ceres residual block");
        problem = next_problem;
    }

    // Gauge fixing: camera 0 and point 0 stay constant (as in the C++ benchmark).
    if let Some(index) = camera_index[0] {
        let _ = problem.set_parameter_block_constant(index);
    }
    if let Some(index) = point_index[0] {
        let _ = problem.set_parameter_block_constant(index);
    }

    let solution = problem
        .solve(&options(LinearSolverType::SPARSE_SCHUR, 3))
        .expect("Ceres solve");
    outcome(&solution)
}

fn run_block_tridiag(blocks: usize) -> Outcome {
    const B: usize = block_tridiag::BLOCK;
    let mut problem = NllsProblem::new();

    // Unary priors: r_k = x_i[k] - target(i, k). Blocks are indexed by
    // insertion order, so pose i is `Index(i)`.
    for i in 0..blocks {
        let cost: CostFunctionType = Box::new(move |values, residuals, jacobians| {
            for k in 0..B {
                residuals[k] = values[0][k] - block_tridiag::target(i, k);
            }
            if let Some(jacobians) = jacobians
                && let Some(block) = &mut jacobians[0]
            {
                for row in 0..B {
                    for col in 0..B {
                        block[row][col] = if row == col { 1.0 } else { 0.0 };
                    }
                }
            }
            true
        });
        problem = problem
            .residual_block_builder()
            .set_cost(cost, B)
            .set_parameters([ParameterBlockOrIndex::Block(vec![0.0; B].into())])
            .build_into_problem()
            .expect("Ceres residual block")
            .0;
    }

    // Relative constraints: r_k = (x_{i+1}[k] - x_i[k]) - delta(i, k).
    for i in 0..blocks.saturating_sub(1) {
        let cost: CostFunctionType = Box::new(move |values, residuals, jacobians| {
            for k in 0..B {
                residuals[k] = (values[1][k] - values[0][k]) - block_tridiag::delta(i, k);
            }
            if let Some(jacobians) = jacobians {
                if let Some(block) = &mut jacobians[0] {
                    for row in 0..B {
                        for col in 0..B {
                            block[row][col] = if row == col { -1.0 } else { 0.0 };
                        }
                    }
                }
                if let Some(block) = &mut jacobians[1] {
                    for row in 0..B {
                        for col in 0..B {
                            block[row][col] = if row == col { 1.0 } else { 0.0 };
                        }
                    }
                }
            }
            true
        });
        problem = problem
            .residual_block_builder()
            .set_cost(cost, B)
            .set_parameters([
                ParameterBlockOrIndex::Index(i),
                ParameterBlockOrIndex::Index(i + 1),
            ])
            .build_into_problem()
            .expect("Ceres residual block")
            .0;
    }

    let solution = problem
        .solve(&options(LinearSolverType::SPARSE_NORMAL_CHOLESKY, 10))
        .expect("Ceres solve");
    outcome(&solution)
}

fn run_pose(count: usize, robust: bool) -> Outcome {
    const DELTA: f64 = 0.3;
    let ground_truth = pose::ground_truth();
    let outliers = count / 4;
    let data = pose::make_observations(&ground_truth, count - outliers, outliers, 0.01, 50.0);
    let initial = pose::initial_pose(&ground_truth);
    let parameters = initial.params().to_vec();

    let mut problem = NllsProblem::new();
    for (k, observation) in data.iter().enumerate() {
        let world = observation.world_point;
        let measurement = observation.measurement;
        let cost: CostFunctionType = Box::new(move |parameters, residuals, jacobians| {
            let raw = parameters[0];
            let norm =
                (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]).sqrt();
            let u = [raw[0] / norm, raw[1] / norm, raw[2] / norm, raw[3] / norm];
            let rotation = pose::Pose { q: u, t: [0.0; 3] };
            let rotated = rotation.rotate(&world);
            residuals[0] = rotated[0] + raw[4] - measurement[0];
            residuals[1] = rotated[1] + raw[5] - measurement[1];
            residuals[2] = rotated[2] + raw[6] - measurement[2];
            if let Some(jacobians) = jacobians
                && let Some(block) = &mut jacobians[0]
            {
                // J_q = J_u * (I - u u^T) / ||q||.
                let ju = ba::quaternion_rotation_jacobian(&u, &world);
                for row in 0..3 {
                    for col in 0..4 {
                        let mut acc = 0.0;
                        for k in 0..4 {
                            let jk = ju[row * 4 + k];
                            let projector = (if k == col { 1.0 } else { 0.0 }) - u[k] * u[col];
                            acc += jk * projector / norm;
                        }
                        block[row][col] = acc;
                    }
                    for col in 0..3 {
                        block[row][4 + col] = if row == col { 1.0 } else { 0.0 };
                    }
                }
            }
            true
        });
        let parameter = if k == 0 {
            ParameterBlockOrIndex::Block(parameters.clone().into())
        } else {
            ParameterBlockOrIndex::Index(0)
        };
        let mut builder = problem
            .residual_block_builder()
            .set_cost(cost, 3)
            .set_parameters([parameter]);
        if robust {
            builder = builder.set_loss(LossFunction::huber(DELTA));
        }
        problem = builder
            .build_into_problem()
            .expect("Ceres residual block")
            .0;
    }

    let solution = problem
        .solve(&options(LinearSolverType::DENSE_NORMAL_CHOLESKY, 20))
        .expect("Ceres solve");
    outcome(&solution)
}
