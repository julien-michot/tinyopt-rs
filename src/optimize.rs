// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! High-level optimization entry points, mirroring `tinyopt::Optimize`.

use crate::backend::{DenseMatrix, Dim, Dynamic, Matrix, Vector};
use crate::block_sparse::{BlockSparseMat, BlockTypes};
use crate::block_tridiag::BlockTridiagMat;
use crate::cost::Cost;
use crate::hessian::HessianStorage;
use crate::optimizers::AccumulationFn;
use crate::optimizers::{bfgs, cg, dl, gd, gn, lbfgs, lm, trcg};
use crate::options::{Options, Solver};
use crate::params::Params;
use crate::params_pack::ParamsPack;
use crate::real::Real;
use crate::sparse::SparseMatrix;
use crate::summary::Summary;

#[cfg(feature = "rayon")]
use crate::backend::DenseVector;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Dynamic gradient type used by multi-parameter optimization.
pub type DynGradient = Vector<f64, Dynamic>;
/// Dynamic Hessian type used by multi-parameter optimization.
pub type DynHessian = Matrix<f64, Dynamic>;

#[cfg(feature = "rayon")]
const MIN_PARALLEL_FACTORS: usize = 16;
#[cfg(feature = "rayon")]
const MAX_PARALLEL_PARTIALS: usize = 4;
#[cfg(feature = "rayon")]
const MAX_PARALLEL_FACTORS: usize = 10_000;

#[cfg(feature = "rayon")]
struct FactorPartial<S: Real, D: Dim, H> {
    cost: Cost,
    gradient: Option<Vector<S, D>>,
    hessian: Option<H>,
}

struct FactorAccumulatorFn<S: Real, D: Dim, H, F, Z, A> {
    #[cfg(feature = "rayon")]
    dims: usize,
    num_factors: usize,
    factor: F,
    #[cfg(feature = "rayon")]
    zero_hessian: Z,
    #[cfg(feature = "rayon")]
    add_hessian: A,
    _marker: core::marker::PhantomData<fn(S, D, H)>,
    #[cfg(not(feature = "rayon"))]
    _zero_hessian: core::marker::PhantomData<Z>,
    #[cfg(not(feature = "rayon"))]
    _add_hessian: core::marker::PhantomData<A>,
}

impl<S, D, H, F, Z, A> FactorAccumulatorFn<S, D, H, F, Z, A>
where
    S: Real,
    D: Dim,
    H: HessianStorage<S>,
    F: Fn(usize, &[S], Option<&mut Vector<S, D>>, Option<&mut H>) -> Cost + Sync,
    Z: Fn(usize) -> H + Sync,
    A: Fn(&mut H, &H) + Sync,
{
    #[cfg(feature = "rayon")]
    fn add_factor(&self, partial: &mut FactorPartial<S, D, H>, index: usize, values: &[S]) {
        partial.cost += (self.factor)(
            index,
            values,
            partial.gradient.as_mut(),
            partial.hessian.as_mut(),
        );
    }

    #[cfg(feature = "rayon")]
    fn finish(
        &self,
        partial: FactorPartial<S, D, H>,
        gradient: Option<&mut Vector<S, D>>,
        hessian: Option<&mut H>,
    ) -> Cost {
        if let (Some(output), Some(contribution)) = (gradient, partial.gradient) {
            for (output, contribution) in DenseVector::as_mut_slice(output)
                .iter_mut()
                .zip(DenseVector::as_slice(&contribution))
            {
                *output += *contribution;
            }
        }
        if let (Some(output), Some(contribution)) = (hessian, partial.hessian) {
            (self.add_hessian)(output, &contribution);
        }
        partial.cost
    }

    fn eval_serial(
        &self,
        values: &[S],
        gradient: Option<&mut Vector<S, D>>,
        hessian: Option<&mut H>,
    ) -> Cost {
        let mut cost = Cost::default();
        let mut gradient = gradient;
        let mut hessian = hessian;
        for index in 0..self.num_factors {
            cost += (self.factor)(
                index,
                values,
                gradient.as_deref_mut(),
                hessian.as_deref_mut(),
            );
        }
        cost
    }

    #[cfg(feature = "rayon")]
    fn eval_parallel(
        &self,
        values: &[S],
        gradient: Option<&mut Vector<S, D>>,
        hessian: Option<&mut H>,
    ) -> Cost {
        let needs_gradient = gradient.is_some();
        let needs_hessian = hessian.is_some();
        let worker_count = rayon::current_num_threads()
            .min(MAX_PARALLEL_PARTIALS)
            .min(self.num_factors);
        let chunk_size = self.num_factors.div_ceil(worker_count);
        let partial = (0..worker_count)
            .into_par_iter()
            .map(|worker| {
                let start = worker * chunk_size;
                let end = (start + chunk_size).min(self.num_factors);
                let mut partial = FactorPartial {
                    cost: Cost::default(),
                    gradient: needs_gradient
                        .then(|| crate::backend::zeros_vector::<S, D>(self.dims)),
                    hessian: needs_hessian.then(|| (self.zero_hessian)(self.dims)),
                };
                for index in start..end {
                    self.add_factor(&mut partial, index, values);
                }
                partial
            })
            .reduce(
                || FactorPartial {
                    cost: Cost::default(),
                    gradient: needs_gradient
                        .then(|| crate::backend::zeros_vector::<S, D>(self.dims)),
                    hessian: needs_hessian.then(|| (self.zero_hessian)(self.dims)),
                },
                |mut left, right| {
                    left.cost += right.cost;
                    if let (Some(left), Some(right)) = (&mut left.gradient, right.gradient) {
                        for (left, right) in DenseVector::as_mut_slice(left)
                            .iter_mut()
                            .zip(DenseVector::as_slice(&right))
                        {
                            *left += *right;
                        }
                    }
                    if let (Some(left), Some(right)) = (&mut left.hessian, right.hessian) {
                        (self.add_hessian)(left, &right);
                    }
                    left
                },
            );
        self.finish(partial, gradient, hessian)
    }
}

impl<S, D, H, F, Z, A> AccumulationFn<S, D, H> for FactorAccumulatorFn<S, D, H, F, Z, A>
where
    S: Real,
    D: Dim,
    H: HessianStorage<S>,
    F: Fn(usize, &[S], Option<&mut Vector<S, D>>, Option<&mut H>) -> Cost + Sync,
    Z: Fn(usize) -> H + Sync,
    A: Fn(&mut H, &H) + Sync,
{
    fn eval(
        &self,
        values: &[S],
        gradient: Option<&mut Vector<S, D>>,
        hessian: Option<&mut H>,
    ) -> Cost {
        #[cfg(feature = "rayon")]
        if self.num_factors >= MIN_PARALLEL_FACTORS
            && self.num_factors <= MAX_PARALLEL_FACTORS
            && rayon::current_num_threads() > 1
        {
            return self.eval_parallel(values, gradient, hessian);
        }
        self.eval_serial(values, gradient, hessian)
    }
}

fn make_factor_accumulator<S, D, H, F, Z, A>(
    dims: usize,
    num_factors: usize,
    factor: F,
    zero_hessian: Z,
    add_hessian: A,
) -> FactorAccumulatorFn<S, D, H, F, Z, A>
where
    S: Real,
    D: Dim,
    H: HessianStorage<S>,
    F: Fn(usize, &[S], Option<&mut Vector<S, D>>, Option<&mut H>) -> Cost + Sync,
    Z: Fn(usize) -> H + Sync,
    A: Fn(&mut H, &H) + Sync,
{
    #[cfg(not(feature = "rayon"))]
    let _ = (&dims, &zero_hessian, &add_hessian);
    FactorAccumulatorFn {
        #[cfg(feature = "rayon")]
        dims,
        num_factors,
        factor,
        #[cfg(feature = "rayon")]
        zero_hessian,
        #[cfg(feature = "rayon")]
        add_hessian,
        _marker: core::marker::PhantomData,
        #[cfg(not(feature = "rayon"))]
        _zero_hessian: core::marker::PhantomData,
        #[cfg(not(feature = "rayon"))]
        _add_hessian: core::marker::PhantomData,
    }
}

fn optimize_with_hessian<P, H, F>(x: &mut P, f: F, options: &Options) -> Summary
where
    P: Params,
    H: HessianStorage<P::Scalar>,
    F: AccumulationFn<P::Scalar, P::Dim, H>,
{
    match options.solver_type {
        Solver::LevenbergMarquardt => lm::Optimizer::<P, H>::new(options.clone()).optimize(x, f),
        Solver::GaussNewton => gn::Optimizer::<P, H>::new(options.clone()).optimize(x, f),
        Solver::GradientDescent => gd::Optimizer::<P, H>::new(options.clone()).optimize(x, f),
        Solver::ConjugateGradient => cg::Optimizer::<P, H>::new(options.clone()).optimize(x, f),
        Solver::DogLeg => dl::Optimizer::<P, H>::new(options.clone()).optimize(x, f),
        Solver::SteihaugToint => trcg::Optimizer::<P, H>::new(options.clone()).optimize(x, f),
        Solver::Bfgs => bfgs::Optimizer::<P, H>::new(options.clone()).optimize(x, f),
        Solver::Lbfgs => lbfgs::Optimizer::<P, H>::new(options.clone()).optimize(x, f),
    }
}

/// Optimize `x` in place with the solver selected by `options`.
///
/// The accumulation function may compute the cost only, the cost and gradient,
/// or the cost, gradient and Hessian; the optimizer requests what it needs.
pub fn optimize<P, F>(x: &mut P, f: F, options: &Options) -> Summary
where
    P: Params,
    Matrix<P::Scalar, P::Dim>: HessianStorage<P::Scalar>,
    F: AccumulationFn<P::Scalar, P::Dim>,
{
    optimize_with_hessian::<P, Matrix<P::Scalar, P::Dim>, F>(x, f, options)
}

/// Optimize independent factors using dense normal-equation accumulators.
///
/// Each factor receives its index, the current parameter slice, and optional
/// gradient/Hessian accumulators. Add that factor's contribution to any
/// provided accumulator and return its cost. With the `rayon` feature enabled,
/// 16 or more factors use parallel evaluation when the active Rayon pool has
/// multiple threads; otherwise, evaluation is serial. Parallel callbacks must
/// be safe to call concurrently, and reduction order may affect floating-point
/// rounding.
/// Use [`optimize_parallel_sparse`] or [`optimize_parallel_block_sparse`] for
/// sparse Hessian storage.
///
/// # Panics
///
/// Panics if `num_factors` is zero.
pub fn optimize_parallel<P, F>(
    x: &mut P,
    num_factors: usize,
    factor: F,
    options: &Options,
) -> Summary
where
    P: Params,
    Matrix<P::Scalar, P::Dim>: HessianStorage<P::Scalar>,
    F: Fn(
            usize,
            &[P::Scalar],
            Option<&mut Vector<P::Scalar, P::Dim>>,
            Option<&mut Matrix<P::Scalar, P::Dim>>,
        ) -> Cost
        + Sync,
{
    assert!(num_factors > 0, "num_factors must be positive");
    let dims = x.dims();
    let accumulator = make_factor_accumulator(
        dims,
        num_factors,
        factor,
        |dims| crate::backend::zeros_matrix::<P::Scalar, P::Dim>(dims, dims),
        |target: &mut Matrix<P::Scalar, P::Dim>, source| {
            for row in 0..target.nrows() {
                for col in 0..target.ncols() {
                    target.set_elem(row, col, target.elem(row, col) + source.elem(row, col));
                }
            }
        },
    );
    optimize(x, accumulator, options)
}

/// Optimize independent factors with a scalar sparse Hessian.
///
/// The callback follows [`optimize_parallel`] but receives a [`SparseMatrix`]
/// Hessian accumulator. With `rayon`, sparse contributions are accumulated
/// privately per worker and merged after factor evaluation.
///
/// # Panics
///
/// Panics if `num_factors` is zero.
pub fn optimize_parallel_sparse<P, F>(
    x: &mut P,
    num_factors: usize,
    factor: F,
    options: &Options,
) -> Summary
where
    P: Params,
    F: Fn(
            usize,
            &[P::Scalar],
            Option<&mut Vector<P::Scalar, P::Dim>>,
            Option<&mut SparseMatrix<P::Scalar>>,
        ) -> Cost
        + Sync,
{
    assert!(num_factors > 0, "num_factors must be positive");
    let accumulator = make_factor_accumulator(
        x.dims(),
        num_factors,
        factor,
        |dims| SparseMatrix::new(dims, dims),
        |target: &mut SparseMatrix<P::Scalar>, source| {
            for (row, col, value) in source.iter() {
                *target.coeff_ref(row, col) += value;
            }
        },
    );
    optimize_with_hessian::<P, SparseMatrix<P::Scalar>, _>(x, accumulator, options)
}

/// Optimize independent factors with a compile-time block-sparse Hessian.
///
/// Factor contributions use [`BlockSparseMat`] block accessors; the serial
/// path and Rayon partials both allow the block pattern to grow as factors are
/// accumulated.
///
/// # Panics
///
/// Panics if `num_factors` is zero.
pub fn optimize_parallel_block_sparse<T, P, F>(
    x: &mut P,
    num_factors: usize,
    factor: F,
    options: &Options,
) -> Summary
where
    T: BlockTypes,
    P: Params,
    F: Fn(
            usize,
            &[P::Scalar],
            Option<&mut Vector<P::Scalar, P::Dim>>,
            Option<&mut BlockSparseMat<T, P::Scalar>>,
        ) -> Cost
        + Sync,
{
    assert!(num_factors > 0, "num_factors must be positive");
    let accumulator = make_factor_accumulator(
        x.dims(),
        num_factors,
        factor,
        |_| BlockSparseMat::<T, P::Scalar>::new(),
        |target: &mut BlockSparseMat<T, P::Scalar>, source| target.add_assign(source),
    );
    optimize_with_hessian::<P, BlockSparseMat<T, P::Scalar>, _>(x, accumulator, options)
}

/// Optimize `x` in place with default options (Levenberg-Marquardt).
pub fn optimize_default<P, F>(x: &mut P, f: F) -> Summary
where
    P: Params,
    Matrix<P::Scalar, P::Dim>: HessianStorage<P::Scalar>,
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
    Matrix<f64, P::Dim>: HessianStorage<f64>,
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
    Matrix<f64, P::Dim>: HessianStorage<f64>,
    F: crate::diff::ResidualsFn<Scalar = f64> + ?Sized,
{
    optimize_residuals(x, f, &Options::default())
}

/// Optimize `x` in place with a compile-time block-sparse Hessian.
///
/// Provide `Optimizer::init_with(prototype)` for an explicit block layout; the
/// layout also grows automatically as accumulation blocks are created.
pub fn optimize_block_sparse<T, P, F>(x: &mut P, f: F, options: &Options) -> Summary
where
    T: BlockTypes,
    P: Params,
    F: AccumulationFn<P::Scalar, P::Dim, BlockSparseMat<T, P::Scalar>>,
{
    match options.solver_type {
        Solver::LevenbergMarquardt => {
            lm::Optimizer::<P, BlockSparseMat<T, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::GaussNewton => {
            gn::Optimizer::<P, BlockSparseMat<T, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::DogLeg => {
            dl::Optimizer::<P, BlockSparseMat<T, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::SteihaugToint => {
            trcg::Optimizer::<P, BlockSparseMat<T, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::GradientDescent => {
            gd::Optimizer::<P, BlockSparseMat<T, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::ConjugateGradient => {
            cg::Optimizer::<P, BlockSparseMat<T, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::Bfgs => {
            bfgs::Optimizer::<P, BlockSparseMat<T, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::Lbfgs => {
            lbfgs::Optimizer::<P, BlockSparseMat<T, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
    }
}

/// Optimize `x` in place with a block-tridiagonal Hessian of block size `B`.
pub fn optimize_tridiag<const B: usize, P, F>(x: &mut P, f: F, options: &Options) -> Summary
where
    P: Params,
    F: AccumulationFn<P::Scalar, P::Dim, BlockTridiagMat<B, P::Scalar>>,
{
    match options.solver_type {
        Solver::LevenbergMarquardt => {
            lm::Optimizer::<P, BlockTridiagMat<B, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::GaussNewton => {
            gn::Optimizer::<P, BlockTridiagMat<B, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::DogLeg => {
            dl::Optimizer::<P, BlockTridiagMat<B, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::SteihaugToint => {
            trcg::Optimizer::<P, BlockTridiagMat<B, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::GradientDescent => {
            gd::Optimizer::<P, BlockTridiagMat<B, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::ConjugateGradient => {
            cg::Optimizer::<P, BlockTridiagMat<B, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::Bfgs => {
            bfgs::Optimizer::<P, BlockTridiagMat<B, P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::Lbfgs => lbfgs::Optimizer::<P, BlockTridiagMat<B, P::Scalar>>::new(options.clone())
            .optimize(x, f),
    }
}

/// Optimize `x` in place with a sparse Hessian.
///
/// The accumulation function receives `Option<&mut SparseMatrix<...>>`; use
/// `Params::reduced()` to enable the Schur-complement path.
pub fn optimize_sparse<P, F>(x: &mut P, f: F, options: &Options) -> Summary
where
    P: Params,
    F: AccumulationFn<P::Scalar, P::Dim, SparseMatrix<P::Scalar>>,
{
    match options.solver_type {
        Solver::LevenbergMarquardt => {
            lm::Optimizer::<P, SparseMatrix<P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::GaussNewton => {
            gn::Optimizer::<P, SparseMatrix<P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::DogLeg => {
            dl::Optimizer::<P, SparseMatrix<P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::SteihaugToint => {
            trcg::Optimizer::<P, SparseMatrix<P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::GradientDescent => {
            gd::Optimizer::<P, SparseMatrix<P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::ConjugateGradient => {
            cg::Optimizer::<P, SparseMatrix<P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::Bfgs => {
            bfgs::Optimizer::<P, SparseMatrix<P::Scalar>>::new(options.clone()).optimize(x, f)
        }
        Solver::Lbfgs => {
            lbfgs::Optimizer::<P, SparseMatrix<P::Scalar>>::new(options.clone()).optimize(x, f)
        }
    }
}

/// Optimize a scalar cost using forward-mode autodiff.
#[cfg(feature = "autodiff")]
pub fn optimize_cost<P, F>(x: &mut P, f: &F, options: &Options) -> Summary
where
    P: Params<Scalar = f64>,
    P::Dim: crate::diff::AdDim,
    Matrix<f64, P::Dim>: HessianStorage<f64>,
    F: crate::diff::CostFn<Scalar = f64> + ?Sized,
{
    optimize(x, crate::diff::cost_accumulation(f), options)
}

/// Optimize robust residual blocks using forward-mode autodiff.
#[cfg(feature = "autodiff")]
pub fn optimize_robust_residuals<P, F>(x: &mut P, f: &F, options: &Options) -> Summary
where
    P: Params<Scalar = f64>,
    P::Dim: crate::diff::AdDim,
    Matrix<f64, P::Dim>: HessianStorage<f64>,
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
