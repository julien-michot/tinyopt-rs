use tinyopt::backend::{DenseVector, Vector};
use tinyopt::block_sparse::{BlockSparseLdlt, BlockSparseMat};
use tinyopt::optimizers::lm;
use tinyopt::{Cost, Dynamic};

#[cfg(test)]
use super::run_ba_block;
use super::{BaParams, BaTypes, ba_params, outcome};

use crate::harness::{self, Outcome};
use crate::options::tinyopt_options;
#[cfg(test)]
use crate::problems::Kind;
use crate::problems::{Case, bundle_adjustment as ba};

use tinyopt::rayon::prelude::*;

#[derive(Clone, Copy)]
struct CameraContribution {
    camera: usize,
    gradient: [f64; 6],
    hessian: [f64; 36],
    cross: [f64; 18],
}

impl Default for CameraContribution {
    fn default() -> Self {
        Self {
            camera: 0,
            gradient: [0.0; 6],
            hessian: [0.0; 36],
            cross: [0.0; 18],
        }
    }
}

#[derive(Clone, Copy)]
struct PointContribution {
    point: usize,
    cost: f64,
    gradient: [f64; 3],
    hessian: [f64; 9],
    cameras: [CameraContribution; 5],
    camera_count: usize,
}

impl Default for PointContribution {
    fn default() -> Self {
        Self {
            point: 0,
            cost: 0.0,
            gradient: [0.0; 3],
            hessian: [0.0; 9],
            cameras: [CameraContribution::default(); 5],
            camera_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_ba_matches_serial_accumulation() {
        let cams = 4;
        let points = 600;
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

pub(crate) fn run_ba_block_parallel(
    case: &Case,
    cams: usize,
    points: usize,
    pool: &tinyopt::rayon::ThreadPool,
) -> Outcome {
    type V = Vector<f64, Dynamic>;
    let problem = ba::make_problem(cams, points);
    let camera_offset = 6 * problem.camera_count();
    let window = cams.min(ba::POINT_CAMERA_WINDOW);

    let mut opt =
        lm::Optimizer::<BaParams, BlockSparseMat<BaTypes, f64>>::new(tinyopt_options(&case.kind));
    opt.init_with(BlockSparseMat::with_counts(&[
        problem.camera_count(),
        problem.point_count(),
    ]));
    let mut x = ba_params(&problem);

    let acc = |xv: &[f64], grad: Option<&mut V>, h: Option<&mut BlockSparseMat<BaTypes, f64>>| {
        if grad.is_none() && h.is_none() {
            let squared_error = problem
                .observations
                .par_chunks(window)
                .map(|observations| {
                    observations
                        .iter()
                        .map(|observation| {
                            let camera = [
                                xv[observation.camera * 6],
                                xv[observation.camera * 6 + 1],
                                xv[observation.camera * 6 + 2],
                                xv[observation.camera * 6 + 3],
                                xv[observation.camera * 6 + 4],
                                xv[observation.camera * 6 + 5],
                            ];
                            let point_start = camera_offset + observation.point * 3;
                            let point = [xv[point_start], xv[point_start + 1], xv[point_start + 2]];
                            let projected = ba::project(&camera, &point);
                            let dx = projected[0] - observation.measurement[0];
                            let dy = projected[1] - observation.measurement[1];
                            dx * dx + dy * dy
                        })
                        .sum::<f64>()
                })
                .sum::<f64>();
            return Cost::with_num_residuals(
                0.5 * squared_error,
                (2 * problem.observations.len()) as i32,
            );
        }

        let points: Vec<_> = problem
            .observations
            .par_chunks(window)
            .map(|observations| {
                let mut contribution = PointContribution {
                    point: observations[0].point,
                    camera_count: observations.len(),
                    ..PointContribution::default()
                };
                for (index, observation) in observations.iter().enumerate() {
                    let camera = [
                        xv[observation.camera * 6],
                        xv[observation.camera * 6 + 1],
                        xv[observation.camera * 6 + 2],
                        xv[observation.camera * 6 + 3],
                        xv[observation.camera * 6 + 4],
                        xv[observation.camera * 6 + 5],
                    ];
                    let point_start = camera_offset + observation.point * 3;
                    let point = [xv[point_start], xv[point_start + 1], xv[point_start + 2]];
                    let (jc, jp) = ba::projection_jacobians(&camera, &point);
                    let projected = ba::project(&camera, &point);
                    let residual = [
                        projected[0] - observation.measurement[0],
                        projected[1] - observation.measurement[1],
                    ];
                    contribution.cost += residual[0] * residual[0] + residual[1] * residual[1];

                    let camera_contribution = &mut contribution.cameras[index];
                    camera_contribution.camera = observation.camera;
                    for i in 0..6 {
                        for k in 0..2 {
                            camera_contribution.gradient[i] += jc[k * 6 + i] * residual[k];
                        }
                        for j in 0..6 {
                            for k in 0..2 {
                                camera_contribution.hessian[i * 6 + j] +=
                                    jc[k * 6 + i] * jc[k * 6 + j];
                            }
                        }
                        for j in 0..3 {
                            for k in 0..2 {
                                camera_contribution.cross[i * 3 + j] +=
                                    jc[k * 6 + i] * jp[k * 3 + j];
                            }
                        }
                    }
                    for i in 0..3 {
                        for k in 0..2 {
                            contribution.gradient[i] += jp[k * 3 + i] * residual[k];
                        }
                        for j in 0..3 {
                            for k in 0..2 {
                                contribution.hessian[i * 3 + j] += jp[k * 3 + i] * jp[k * 3 + j];
                            }
                        }
                    }
                }
                contribution
            })
            .collect();

        let mut cost = 0.0;
        let mut g = grad;
        let mut h = h;
        for point in points {
            cost += point.cost;
            if let Some(g) = g.as_mut() {
                let point_start = camera_offset + point.point * 3;
                for i in 0..3 {
                    g.set_at(point_start + i, g.at(point_start + i) + point.gradient[i]);
                }
                for camera in &point.cameras[..point.camera_count] {
                    for i in 0..6 {
                        let index = camera.camera * 6 + i;
                        g.set_at(index, g.at(index) + camera.gradient[i]);
                    }
                }
            }
            if let Some(h) = h.as_mut() {
                let mut point_block = h.block_mut::<3, 3>(point.point, point.point);
                for i in 0..3 {
                    for j in 0..3 {
                        point_block.set_at(i, j, point_block.at(i, j) + point.hessian[i * 3 + j]);
                    }
                }
                for camera in &point.cameras[..point.camera_count] {
                    let mut camera_block = h.block_mut::<6, 6>(camera.camera, camera.camera);
                    for i in 0..6 {
                        for j in 0..6 {
                            camera_block.set_at(
                                i,
                                j,
                                camera_block.at(i, j) + camera.hessian[i * 6 + j],
                            );
                        }
                    }
                    let mut cross_block = h.block_mut::<6, 3>(camera.camera, point.point);
                    for i in 0..6 {
                        for j in 0..3 {
                            cross_block.set_at(i, j, camera.cross[i * 3 + j]);
                        }
                    }
                }
            }
        }
        Cost::with_num_residuals(cost * 0.5, (2 * problem.observations.len()) as i32)
    };

    let summary = pool.install(|| opt.optimize(&mut x, acc));
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
