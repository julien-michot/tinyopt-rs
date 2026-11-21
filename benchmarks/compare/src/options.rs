// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Solver options, mirroring the C++ `benchmarks/common/options.h` and the
//! Ceres mapping documented in `docs/development/benchmarking-and-profiling.md`.

use tinyopt::Options;

use crate::problems::Kind;

/// Options for the tinyopt-rs backend.
pub fn tinyopt_options(kind: &Kind) -> Options {
    let mut options = Options::default();
    options.log.enable = false;
    options.exp.save_history = false;
    options.exp.measure_time = false;
    options.exp.final_cov = false;
    options.stop.max_iters = 100;
    options.stop.min_error = 1e-12;
    options.stop.min_rerr_dec = 1e-6;
    options.stop.min_step_norm2 = 1e-16;
    options.stop.min_grad_norm2 = 0.0;
    options.stop.min_grad_max = 1e-9;
    options.stop.max_consec_failures = 10;
    options.lm.jacobi_scaling = true;

    match kind {
        Kind::DenseMathStatic { single, .. } | Kind::DenseMathDynamic { single, .. } => {
            if *single {
                options.stop.min_error = 1e-6;
                options.stop.min_step_norm2 = 1e-10;
            } else {
                options.stop.min_error = 1e-14;
            }
        }
        Kind::DensePrior { .. } => {
            options.stop.max_iters = 20;
            options.stop.min_error = 1e-14;
            options.stop.min_step_rel_norm2 = 0.0;
        }
        Kind::RobustPose { .. } => {
            options.stop.max_consec_failures = 20;
        }
        Kind::SparseChain { .. }
        | Kind::BundleAdjustmentBlock { .. }
        | Kind::BundleAdjustmentSparse { .. } => {}
    }
    options
}
