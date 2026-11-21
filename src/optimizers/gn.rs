// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Gauss-Newton optimizer.

use crate::backend::{Matrix, Vector};
use crate::cost::Cost;
use crate::covariance::{CovarianceOptions, FinalCovariance};
use crate::hessian::HessianStorage;
use crate::optimizers::fixed::FixedSet;
use crate::optimizers::impl_fixed_api;
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, neg_solve, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;

/// Gauss-Newton optimizer, mirroring `tinyopt::gn::Optimizer`.
#[derive(Clone, Debug)]
pub struct Optimizer<
    P: Params,
    H: HessianStorage<<P as Params>::Scalar> = Matrix<<P as Params>::Scalar, <P as Params>::Dim>,
> {
    options: Options,
    fixed: FixedSet,
    num_reduced: usize,
    final_h: Option<H>,
    final_cost: Cost,
    prototype: Option<H>,
    _marker: core::marker::PhantomData<P>,
}

impl<P: Params, H: HessianStorage<P::Scalar>> Optimizer<P, H> {
    impl_fixed_api!();

    /// Provide a Hessian prototype (required for dynamic block layouts).
    pub fn init_with(&mut self, prototype: H) -> &mut Self {
        self.prototype = Some(prototype);
        self
    }

    /// Create an optimizer with the given options.
    pub fn new(options: Options) -> Self {
        Self {
            options,
            fixed: FixedSet::default(),
            num_reduced: 0,
            final_h: None,
            final_cost: Cost::default(),
            prototype: None,
            _marker: core::marker::PhantomData,
        }
    }

    /// Options in use.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Whether a Schur reduced block was detected.
    pub fn has_reduced(&self) -> bool {
        self.num_reduced > 0
    }

    /// Number of reduced (head) parameters.
    pub fn num_reduced(&self) -> usize {
        self.num_reduced
    }

    /// Posterior covariance estimate (default options).
    pub fn covariance(&self) -> Option<FinalCovariance> {
        <Self as Algorithm<P, H>>::covariance(self, &CovarianceOptions::default())
    }

    /// Posterior covariance estimate with explicit options.
    pub fn covariance_with(&self, options: &CovarianceOptions) -> Option<FinalCovariance> {
        <Self as Algorithm<P, H>>::covariance(self, options)
    }

    /// Optimize `x` in place.
    pub fn optimize<F>(&mut self, x: &mut P, f: F) -> Summary
    where
        F: AccumulationFn<P::Scalar, P::Dim, H>,
    {
        let options = self.options.clone();
        run(x, f, &options, self)
    }
}

impl<P: Params, H: HessianStorage<P::Scalar>> Algorithm<P, H> for Optimizer<P, H> {
    fn reset(&mut self, _dims: usize) {}

    fn set_num_reduced(&mut self, num_reduced: usize) {
        self.num_reduced = num_reduced;
    }

    fn hessian_prototype(&self) -> Option<&H> {
        self.prototype.as_ref()
    }

    fn fixed(&self) -> &FixedSet {
        &self.fixed
    }

    fn set_fixed_indices(&mut self, indices: &[usize]) {
        if self.fixed.indices().is_empty() {
            self.fixed.fix(indices);
        }
    }

    fn precondition(&mut self, _grad: &mut Vector<P::Scalar, P::Dim>, _h: &mut H) {}

    fn solve(
        &mut self,
        grad: &Vector<P::Scalar, P::Dim>,
        h: &H,
    ) -> Option<Step<P::Scalar, P::Dim>> {
        neg_solve::<P::Scalar, P::Dim, H>(&self.options, grad, h, self.num_reduced)
            .map(|dx| (dx, 0.0))
    }

    fn on_good_step(&mut self, _quality: f64) {}

    fn on_bad_step(&mut self) {}

    fn save_final_system(&mut self, h: &H, cost: Cost) {
        self.final_h = Some(h.clone());
        self.final_cost = cost;
    }

    fn covariance(&self, options: &CovarianceOptions) -> Option<FinalCovariance> {
        let h = self.final_h.as_ref()?;
        h.to_final_covariance(
            options,
            self.final_cost,
            self.fixed.indices(),
            self.num_reduced,
        )
    }
}
