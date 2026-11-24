// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Gradient descent optimizer.

use num_traits::FromPrimitive;

use crate::backend::{DenseVector, Matrix, Vector};
use crate::hessian::HessianStorage;
use crate::optimizers::fixed::FixedSet;
use crate::optimizers::impl_fixed_api;
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;

/// Gradient descent optimizer, mirroring `tinyopt::gd::Optimizer`.
#[derive(Clone, Debug)]
pub struct Optimizer<
    P: Params,
    H: HessianStorage<<P as Params>::Scalar> = Matrix<<P as Params>::Scalar, <P as Params>::Dim>,
> {
    options: Options,
    fixed: FixedSet,
    _marker: core::marker::PhantomData<(P, H)>,
}

impl<P: Params, H: HessianStorage<P::Scalar>> Optimizer<P, H> {
    impl_fixed_api!();

    /// Create an optimizer with the given options.
    pub fn new(options: Options) -> Self {
        Self {
            options,
            fixed: FixedSet::default(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Options in use.
    pub fn options(&self) -> &Options {
        &self.options
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

    fn needs_hessian(&self) -> bool {
        false
    }

    fn fixed(&self) -> &FixedSet {
        &self.fixed
    }

    fn precondition(&mut self, _grad: &mut Vector<P::Scalar, P::Dim>, _h: &mut H) {}

    fn solve(
        &mut self,
        grad: &Vector<P::Scalar, P::Dim>,
        _h: &H,
    ) -> Option<Step<P::Scalar, P::Dim>> {
        let lr = P::Scalar::from_f32(self.options.gd.lr).expect("learning rate fits");
        let mut dx = grad.clone();
        DenseVector::scale(&mut dx, -lr);
        Some((dx, 0.0))
    }

    fn on_good_step(&mut self, _quality: f64) {}

    fn on_bad_step(&mut self) {}
}
