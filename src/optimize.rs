// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! High-level optimization entry points, mirroring `tinyopt::Optimize`.

use crate::optimizers::AccumulationFn;
use crate::optimizers::{gn, lm};
use crate::options::{Options, Solver};
use crate::params::Params;
use crate::stop_reasons::StopReason;
use crate::summary::Summary;

/// Optimize `x` in place with the solver selected by `options`.
///
/// The accumulation function may compute the cost only, the cost and gradient,
/// or the cost, gradient and Hessian; the optimizer requests what it needs.
pub fn optimize<P, F>(x: &mut P, f: F, options: &Options) -> Summary
where
    P: Params,
    F: AccumulationFn<P::Scalar, P::Dim>,
{
    match options.solver_type {
        Solver::LevenbergMarquardt => lm::Optimizer::<P>::new(options.clone()).optimize(x, f),
        Solver::GaussNewton => gn::Optimizer::<P>::new(options.clone()).optimize(x, f),
        other => {
            log::warn!("solver {other:?} is not implemented yet in tinyopt-rs");
            Summary {
                stop_reason: StopReason::Skipped,
                ..Default::default()
            }
        }
    }
}

/// Optimize `x` in place with default options (Levenberg-Marquardt).
pub fn optimize_default<P, F>(x: &mut P, f: F) -> Summary
where
    P: Params,
    F: AccumulationFn<P::Scalar, P::Dim>,
{
    optimize(x, f, &Options::default())
}

/// Optimize a residual function using forward-mode autodiff.
///
/// This is the Rust equivalent of passing a residuals lambda to C++ `Optimize`.
#[cfg(feature = "autodiff")]
pub fn optimize_residuals<P, F>(x: &mut P, f: &F, options: &Options) -> Summary
where
    P: Params<Scalar = f64>,
    F: crate::diff::ResidualsFn<Scalar = f64> + ?Sized,
{
    optimize(x, crate::diff::ad_accumulation(f), options)
}

/// Optimize a residual function with default options.
#[cfg(feature = "autodiff")]
pub fn optimize_residuals_default<P, F>(x: &mut P, f: &F) -> Summary
where
    P: Params<Scalar = f64>,
    F: crate::diff::ResidualsFn<Scalar = f64> + ?Sized,
{
    optimize_residuals(x, f, &Options::default())
}
