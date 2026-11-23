use tinyopt::backend::{DenseVector, Vector};
use tinyopt::block_sparse::{BlockSparseLdlt, BlockSparseMat};
use tinyopt::{Cost, Dynamic, optimize_parallel_block_sparse};

use super::{BaTypes, ba_params, outcome};
use crate::harness::{self, Outcome};
use crate::options::tinyopt_options;
#[cfg(test)]
use crate::problems::Kind;
use crate::problems::{Case, bundle_adjustment as ba};

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
    let window = cams.min(ba::POINT_CAMERA_WINDOW);
    let mut x = ba_params(&problem);

    let factor = |point_index: usize,
                  values: &[f64],
                  gradient: Option<&mut V>,
                  hessian: Option<&mut BlockSparseMat<BaTypes, f64>>| {
        let start = point_index * window;
        let end = (start + window).min(problem.observations.len());
        let observations = &problem.observations[start..end];
        let mut cost = 0.0;
        let mut gradient = gradient;
        let mut hessian = hessian;

        for observation in observations {
            let camera_start = observation.camera * 7;
            let camera = [
                values[camera_start],
                values[camera_start + 1],
                values[camera_start + 2],
                values[camera_start + 3],
                values[camera_start + 4],
                values[camera_start + 5],
                values[camera_start + 6],
            ];
            let point_start = ambient_camera_offset + observation.point * 3;
            let point = [
                values[point_start],
                values[point_start + 1],
                values[point_start + 2],
            ];
            let projected = ba::project(&camera, &point);
            let residual = [
                projected[0] - observation.measurement[0],
                projected[1] - observation.measurement[1],
            ];
            cost += 0.5 * (residual[0] * residual[0] + residual[1] * residual[1]);

            if gradient.is_some() || hessian.is_some() {
                let (camera_jacobian, point_jacobian) = ba::projection_jacobians(&camera, &point);
                if let Some(gradient) = gradient.as_deref_mut() {
                    for i in 0..6 {
                        let contribution: f64 = (0..2)
                            .map(|k| camera_jacobian[k * 6 + i] * residual[k])
                            .sum();
                        let index = observation.camera * 6 + i;
                        gradient.set_at(index, gradient.at(index) + contribution);
                    }
                    for i in 0..3 {
                        let contribution: f64 = (0..2)
                            .map(|k| point_jacobian[k * 3 + i] * residual[k])
                            .sum();
                        let index = tangent_camera_offset + observation.point * 3 + i;
                        gradient.set_at(index, gradient.at(index) + contribution);
                    }
                }
                if let Some(hessian) = hessian.as_deref_mut() {
                    hessian
                        .block_mut::<6, 6>(observation.camera, observation.camera)
                        .add_jtj(1.0, &camera_jacobian, 2);
                    hessian
                        .block_mut::<3, 3>(observation.point, observation.point)
                        .add_jtj(1.0, &point_jacobian, 2);
                    let mut cross =
                        hessian.block_mut::<6, 3>(observation.camera, observation.point);
                    for k in 0..2 {
                        cross.add_outer_product(
                            1.0,
                            &camera_jacobian[k * 6..k * 6 + 6],
                            &point_jacobian[k * 3..k * 3 + 3],
                        );
                    }
                }
            }
        }

        Cost::with_num_residuals(cost, (2 * observations.len()) as i32)
    };

    let options = tinyopt_options(&case.kind);
    let summary = pool.install(|| {
        optimize_parallel_block_sparse::<BaTypes, _, _>(
            &mut x,
            problem.point_count(),
            factor,
            &options,
        )
    });
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
