use tinyopt::backend::{DenseVector, Vector};
use tinyopt::block_sparse::{BlockSparseLdlt, BlockSparseMat};
use tinyopt::optimizers::lm;
use tinyopt::{Cost, Dynamic};

use super::{BaTypes, ba_params, outcome};
use crate::harness::{self, Outcome};
use crate::options::tinyopt_options;
#[cfg(test)]
use crate::problems::Kind;
use crate::problems::{Case, bundle_adjustment as ba};
use std::sync::Mutex;

#[derive(Clone, Copy, Default)]
struct ObservationEvaluation {
    camera_jacobian: [f64; 12],
    point_jacobian: [f64; 6],
    residual: [f64; 2],
}

#[derive(Default)]
struct PointAccumulation {
    gradient: [f64; 3],
    hessian: [f64; 9],
}

struct CameraAccumulation {
    gradient: [f64; 6],
    hessian: [f64; 36],
}

impl Default for CameraAccumulation {
    fn default() -> Self {
        Self {
            gradient: [0.0; 6],
            hessian: [0.0; 36],
        }
    }
}

#[derive(Default)]
struct BaScratch {
    evaluations: Vec<ObservationEvaluation>,
    points: Vec<PointAccumulation>,
    cameras: Vec<CameraAccumulation>,
    couplings: Vec<[f64; 18]>,
}

fn observation_groups(
    count: usize,
    observations: &[ba::Observation],
    index: impl Fn(&ba::Observation) -> usize,
) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); count];
    for (observation_index, observation) in observations.iter().enumerate() {
        groups[index(observation)].push(observation_index);
    }
    groups
}

fn accumulate_gradient<const N: usize>(
    gradient: &mut [f64; N],
    jacobian: &[f64],
    residual: [f64; 2],
) {
    for i in 0..N {
        gradient[i] += jacobian[i] * residual[0] + jacobian[N + i] * residual[1];
    }
}

fn accumulate_jtj(hessian: &mut [f64], jacobian: &[f64], size: usize) {
    for column in 0..size {
        for row in 0..size {
            hessian[column * size + row] +=
                jacobian[row] * jacobian[column] + jacobian[size + row] * jacobian[size + column];
        }
    }
}

fn accumulate_cross(hessian: &mut [f64; 18], camera_jacobian: &[f64], point_jacobian: &[f64]) {
    for column in 0..3 {
        for row in 0..6 {
            hessian[column * 6 + row] += camera_jacobian[row] * point_jacobian[column]
                + camera_jacobian[6 + row] * point_jacobian[3 + column];
        }
    }
}

pub(crate) fn run_ba_block_parallel(
    case: &Case,
    cams: usize,
    points: usize,
    pool: &tinyopt::rayon::ThreadPool,
) -> Outcome {
    type V = Vector<f64, Dynamic>;
    let problem = ba::make_problem(cams, points);
    let ambient_camera_offset = 7 * problem.camera_count();
    let tangent_camera_offset = 6 * problem.camera_count();
    let point_observations = observation_groups(
        problem.point_count(),
        &problem.observations,
        |observation| observation.point,
    );
    let camera_observations = observation_groups(
        problem.camera_count(),
        &problem.observations,
        |observation| observation.camera,
    );
    let scratch = Mutex::new(BaScratch::default());
    let mut x = ba_params(&problem);

    let accumulator = |values: &[f64],
                       gradient: Option<&mut V>,
                       hessian: Option<&mut BlockSparseMat<BaTypes, f64>>| {
        if gradient.is_none() && hessian.is_none() {
            let cost = if tinyopt::rayon::current_num_threads() > 1 {
                use tinyopt::rayon::prelude::*;
                problem
                    .observations
                    .par_iter()
                    .map(|observation| {
                        let camera_start = observation.camera * 7;
                        let camera: [f64; 7] = values[camera_start..camera_start + 7]
                            .try_into()
                            .expect("camera parameter block");
                        let point_start = ambient_camera_offset + observation.point * 3;
                        let point: [f64; 3] = values[point_start..point_start + 3]
                            .try_into()
                            .expect("point parameter block");
                        let projected = ba::project(&camera, &point);
                        let dx = projected[0] - observation.measurement[0];
                        let dy = projected[1] - observation.measurement[1];
                        0.5 * (dx * dx + dy * dy)
                    })
                    .sum()
            } else {
                problem
                    .observations
                    .iter()
                    .map(|observation| {
                        let camera_start = observation.camera * 7;
                        let camera: [f64; 7] = values[camera_start..camera_start + 7]
                            .try_into()
                            .expect("camera parameter block");
                        let point_start = ambient_camera_offset + observation.point * 3;
                        let point: [f64; 3] = values[point_start..point_start + 3]
                            .try_into()
                            .expect("point parameter block");
                        let projected = ba::project(&camera, &point);
                        let dx = projected[0] - observation.measurement[0];
                        let dy = projected[1] - observation.measurement[1];
                        0.5 * (dx * dx + dy * dy)
                    })
                    .sum()
            };
            return Cost::with_num_residuals(cost, (2 * problem.observations.len()) as i32);
        }

        use tinyopt::rayon::prelude::*;
        let mut scratch = scratch.lock().expect("BA scratch lock poisoned");
        scratch
            .evaluations
            .resize(problem.observations.len(), ObservationEvaluation::default());
        scratch
            .points
            .resize_with(problem.point_count(), PointAccumulation::default);
        scratch
            .cameras
            .resize_with(problem.camera_count(), CameraAccumulation::default);
        scratch
            .couplings
            .resize(problem.observations.len(), [0.0; 18]);
        scratch
            .couplings
            .iter_mut()
            .for_each(|coupling| coupling.fill(0.0));
        let needs_gradient = gradient.is_some();
        let needs_hessian = hessian.is_some();
        let BaScratch {
            evaluations,
            points: point_accumulations,
            cameras: camera_accumulations,
            couplings,
        } = &mut *scratch;

        evaluations
            .par_iter_mut()
            .enumerate()
            .for_each(|(observation_index, evaluation)| {
                let observation = &problem.observations[observation_index];
                let camera_start = observation.camera * 7;
                let camera: [f64; 7] = values[camera_start..camera_start + 7]
                    .try_into()
                    .expect("camera parameter block");
                let point_start = ambient_camera_offset + observation.point * 3;
                let point: [f64; 3] = values[point_start..point_start + 3]
                    .try_into()
                    .expect("point parameter block");
                let (camera_jacobian, point_jacobian) = ba::projection_jacobians(&camera, &point);
                let projected = ba::project(&camera, &point);
                *evaluation = ObservationEvaluation {
                    camera_jacobian,
                    point_jacobian,
                    residual: [
                        projected[0] - observation.measurement[0],
                        projected[1] - observation.measurement[1],
                    ],
                };
            });

        point_accumulations.par_iter_mut().enumerate().for_each(
            |(point_index, point_accumulation)| {
                point_accumulation.gradient.fill(0.0);
                point_accumulation.hessian.fill(0.0);
                for &observation_index in &point_observations[point_index] {
                    let evaluation = &evaluations[observation_index];
                    if needs_gradient {
                        accumulate_gradient(
                            &mut point_accumulation.gradient,
                            &evaluation.point_jacobian,
                            evaluation.residual,
                        );
                    }
                    if needs_hessian {
                        accumulate_jtj(
                            &mut point_accumulation.hessian,
                            &evaluation.point_jacobian,
                            3,
                        );
                    }
                }
            },
        );

        camera_accumulations.par_iter_mut().enumerate().for_each(
            |(camera_index, camera_accumulation)| {
                camera_accumulation.gradient.fill(0.0);
                camera_accumulation.hessian.fill(0.0);
                for &observation_index in &camera_observations[camera_index] {
                    let evaluation = &evaluations[observation_index];
                    if needs_gradient {
                        accumulate_gradient(
                            &mut camera_accumulation.gradient,
                            &evaluation.camera_jacobian,
                            evaluation.residual,
                        );
                    }
                    if needs_hessian {
                        accumulate_jtj(
                            &mut camera_accumulation.hessian,
                            &evaluation.camera_jacobian,
                            6,
                        );
                    }
                }
            },
        );

        if needs_hessian {
            couplings
                .par_iter_mut()
                .enumerate()
                .for_each(|(observation_index, coupling)| {
                    let evaluation = &evaluations[observation_index];
                    accumulate_cross(
                        coupling,
                        &evaluation.camera_jacobian,
                        &evaluation.point_jacobian,
                    );
                });
        }

        if let Some(gradient) = gradient {
            for (point_index, accumulation) in point_accumulations.iter().enumerate() {
                for i in 0..3 {
                    let index = tangent_camera_offset + point_index * 3 + i;
                    gradient.set_at(index, gradient.at(index) + accumulation.gradient[i]);
                }
            }
            for (camera_index, accumulation) in camera_accumulations.iter().enumerate() {
                for i in 0..6 {
                    let index = camera_index * 6 + i;
                    gradient.set_at(index, gradient.at(index) + accumulation.gradient[i]);
                }
            }
        }

        if let Some(hessian) = hessian {
            for (point_index, accumulation) in point_accumulations.iter().enumerate() {
                let mut block = hessian.block_mut::<3, 3>(point_index, point_index);
                for column in 0..3 {
                    for row in 0..3 {
                        block.set_at(
                            row,
                            column,
                            block.at(row, column) + accumulation.hessian[column * 3 + row],
                        );
                    }
                }
            }
            for (camera_index, accumulation) in camera_accumulations.iter().enumerate() {
                let mut block = hessian.block_mut::<6, 6>(camera_index, camera_index);
                for column in 0..6 {
                    for row in 0..6 {
                        block.set_at(
                            row,
                            column,
                            block.at(row, column) + accumulation.hessian[column * 6 + row],
                        );
                    }
                }
            }
            for (observation, coupling) in problem.observations.iter().zip(couplings.iter()) {
                let mut block = hessian.block_mut::<6, 3>(observation.camera, observation.point);
                for column in 0..3 {
                    for row in 0..6 {
                        block.set_at(
                            row,
                            column,
                            block.at(row, column) + coupling[column * 6 + row],
                        );
                    }
                }
            }
        }

        let cost = if tinyopt::rayon::current_num_threads() > 1 {
            evaluations
                .par_iter()
                .map(|evaluation| {
                    0.5 * (evaluation.residual[0] * evaluation.residual[0]
                        + evaluation.residual[1] * evaluation.residual[1])
                })
                .sum()
        } else {
            evaluations
                .iter()
                .map(|evaluation| {
                    0.5 * (evaluation.residual[0] * evaluation.residual[0]
                        + evaluation.residual[1] * evaluation.residual[1])
                })
                .sum()
        };
        Cost::with_num_residuals(cost, (2 * problem.observations.len()) as i32)
    };

    let mut optimizer =
        lm::Optimizer::<_, BlockSparseMat<BaTypes, f64>>::new(tinyopt_options(&case.kind));
    optimizer.init_with(BlockSparseMat::with_counts(&[
        problem.camera_count(),
        problem.point_count(),
    ]));
    let summary = pool.install(|| optimizer.optimize(&mut x, accumulator));
    outcome(&summary)
}

pub(crate) fn measure_ba_block_schur(
    cams: usize,
    points: usize,
    threads: usize,
    warmup: std::time::Duration,
    samples: usize,
) -> harness::Stats {
    let mut hessian = BlockSparseMat::<BaTypes, f64>::with_counts(&[cams, points]);
    for camera in 0..cams {
        let mut block = hessian.block_mut::<6, 6>(camera, camera);
        for i in 0..6 {
            block.set_at(i, i, 10.0);
        }
    }
    for point in 0..points {
        let mut block = hessian.block_mut::<3, 3>(point, point);
        for i in 0..3 {
            block.set_at(i, i, 4.0);
        }
        let first_camera = point % (cams - cams.min(ba::POINT_CAMERA_WINDOW) + 1);
        for camera in first_camera..first_camera + cams.min(ba::POINT_CAMERA_WINDOW) {
            let mut coupling = hessian.block_mut::<6, 3>(camera, point);
            coupling.set_at(0, 0, 0.01);
            coupling.set_at(1, 1, -0.01);
        }
    }
    let mut solver = BlockSparseLdlt::<BaTypes, f64>::new();
    assert!(solver.analyze(&hessian, 6 * cams));
    let rhs = vec![1.0; hessian.dims()];
    let mut solution = vec![0.0; hessian.dims()];
    let pool = tinyopt::rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build Rayon thread pool");
    harness::measure(warmup, samples, || {
        pool.install(|| {
            assert!(solver.compute(&hessian));
            assert!(solver.solve_into(&hessian, &rhs, &mut solution));
        });
        Outcome {
            converged: true,
            iterations: 1,
            final_cost: 0.0,
        }
    })
    .0
}

#[cfg(test)]
mod tests {
    use super::super::run_ba_block;
    use super::*;

    #[test]
    fn parallel_ba_matches_serial_accumulation() {
        let cams = 10;
        let points = 3000;
        let case = Case {
            kind: Kind::BundleAdjustmentBlock { cams, points },
            name: format!("{cams}c {points}p"),
            short: format!("{cams}c {points}p"),
        };
        let serial = run_ba_block(&case, cams, points);
        let pool = tinyopt::rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("Rayon pool");
        let parallel = run_ba_block_parallel(&case, cams, points, &pool);

        assert!(serial.converged && parallel.converged);
        assert!((serial.final_cost - parallel.final_cost).abs() < 1e-8);
    }
}
