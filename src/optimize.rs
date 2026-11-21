// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! High-level optimization entry points, mirroring `tinyopt::Optimize`.

use crate::backend::{Dynamic, Matrix, Vector};
use crate::cost::Cost;
use crate::optimizers::AccumulationFn;
use crate::optimizers::{bfgs, cg, dl, gd, gn, lbfgs, lm};
use crate::options::{Options, Solver};
use crate::params::Params;
use crate::params_pack::ParamsPack;
use crate::summary::Summary;

/// Dynamic gradient type used by multi-parameter optimization.
pub type DynGradient = Vector<f64, Dynamic>;
/// Dynamic Hessian type used by multi-parameter optimization.
pub type DynHessian = Matrix<f64, Dynamic>;

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
        Solver::GradientDescent => gd::Optimizer::<P>::new(options.clone()).optimize(x, f),
        Solver::ConjugateGradient => cg::Optimizer::<P>::new(options.clone()).optimize(x, f),
        Solver::DogLeg => dl::Optimizer::<P>::new(options.clone()).optimize(x, f),
        Solver::Bfgs => bfgs::Optimizer::<P>::new(options.clone()).optimize(x, f),
        Solver::Lbfgs => lbfgs::Optimizer::<P>::new(options.clone()).optimize(x, f),
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
    P::Dim: crate::diff::AdDim,
    F: crate::diff::ResidualsFn<Scalar = f64> + ?Sized,
{
    optimize(x, crate::diff::ad_accumulation(f), options)
}

/// Optimize a residual function with default options.
#[cfg(feature = "autodiff")]
pub fn optimize_residuals_default<P, F>(x: &mut P, f: &F) -> Summary
where
    P: Params<Scalar = f64>,
    P::Dim: crate::diff::AdDim,
    F: crate::diff::ResidualsFn<Scalar = f64> + ?Sized,
{
    optimize_residuals(x, f, &Options::default())
}

/// Optimize robust residual blocks using forward-mode autodiff.
#[cfg(feature = "autodiff")]
pub fn optimize_robust_residuals<P, F>(x: &mut P, f: &F, options: &Options) -> Summary
where
    P: Params<Scalar = f64>,
    P::Dim: crate::diff::AdDim,
    F: crate::diff::RobustResidualsFn<Scalar = f64> + ?Sized,
{
    optimize(x, crate::diff::robust_accumulation(f), options)
}

/// Optimize two parameters together.
pub fn optimize2<A, B, F>(a: &mut A, b: &mut B, f: F, options: &Options) -> Summary
where
    A: Params<Scalar = f64> + Clone,
    B: Params<Scalar = f64> + Clone,
    F: Fn(&[f64], &[f64], Option<&mut DynGradient>, Option<&mut DynHessian>) -> Cost,
{
    let da = a.dims();
    let mut pack = ParamsPack::new((a.clone(), b.clone()));
    let summary = {
        let acc = |flat: &[f64], grad: Option<&mut DynGradient>, h: Option<&mut DynHessian>| {
            f(&flat[..da], &flat[da..], grad, h)
        };
        optimize(&mut pack, acc, options)
    };
    *a = pack.values.0.clone();
    *b = pack.values.1.clone();
    summary
}

/// Optimize two parameters with default options.
pub fn optimize2_default<A, B, F>(a: &mut A, b: &mut B, f: F) -> Summary
where
    A: Params<Scalar = f64> + Clone,
    B: Params<Scalar = f64> + Clone,
    F: Fn(&[f64], &[f64], Option<&mut DynGradient>, Option<&mut DynHessian>) -> Cost,
{
    optimize2(a, b, f, &Options::default())
}

/// Optimize three parameters together.
pub fn optimize3<A, B, C, F>(a: &mut A, b: &mut B, c: &mut C, f: F, options: &Options) -> Summary
where
    A: Params<Scalar = f64> + Clone,
    B: Params<Scalar = f64> + Clone,
    C: Params<Scalar = f64> + Clone,
    F: Fn(&[f64], &[f64], &[f64], Option<&mut DynGradient>, Option<&mut DynHessian>) -> Cost,
{
    let (da, db) = (a.dims(), b.dims());
    let mut pack = ParamsPack::new((a.clone(), b.clone(), c.clone()));
    let summary = {
        let acc = |flat: &[f64], grad: Option<&mut DynGradient>, h: Option<&mut DynHessian>| {
            f(&flat[..da], &flat[da..da + db], &flat[da + db..], grad, h)
        };
        optimize(&mut pack, acc, options)
    };
    *a = pack.values.0.clone();
    *b = pack.values.1.clone();
    *c = pack.values.2.clone();
    summary
}

/// Optimize three parameters with default options.
pub fn optimize3_default<A, B, C, F>(a: &mut A, b: &mut B, c: &mut C, f: F) -> Summary
where
    A: Params<Scalar = f64> + Clone,
    B: Params<Scalar = f64> + Clone,
    C: Params<Scalar = f64> + Clone,
    F: Fn(&[f64], &[f64], &[f64], Option<&mut DynGradient>, Option<&mut DynHessian>) -> Cost,
{
    optimize3(a, b, c, f, &Options::default())
}
