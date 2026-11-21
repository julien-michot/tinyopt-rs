// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Gradient descent optimizer.

use num_traits::FromPrimitive;

use crate::backend::{DenseVector, Matrix, Vector};
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;

/// Gradient descent optimizer, mirroring `tinyopt::gd::Optimizer`.
#[derive(Clone, Debug)]
pub struct Optimizer<P: Params> {
    options: Options,
    _marker: core::marker::PhantomData<P>,
}

impl<P: Params> Optimizer<P> {
    /// Create an optimizer with the given options.
    pub fn new(options: Options) -> Self {
        Self {
            options,
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
        F: AccumulationFn<P::Scalar, P::Dim>,
    {
        let options = self.options.clone();
        run(x, f, &options, self)
    }
}

impl<P: Params> Algorithm<P> for Optimizer<P> {
    fn reset(&mut self, _dims: usize) {}

    fn needs_hessian(&self) -> bool {
        false
    }

    fn precondition(
        &mut self,
        _grad: &mut Vector<P::Scalar, P::Dim>,
        _h: &mut Matrix<P::Scalar, P::Dim>,
    ) {
    }

    fn solve(
        &mut self,
        grad: &Vector<P::Scalar, P::Dim>,
        _h: &Matrix<P::Scalar, P::Dim>,
    ) -> Option<Step<P::Scalar, P::Dim>> {
        let lr = P::Scalar::from_f32(self.options.gd.lr).expect("learning rate fits");
        let mut dx = grad.clone();
        DenseVector::scale(&mut dx, -lr);
        Some((dx, 0.0))
    }

    fn on_good_step(&mut self, _quality: f64) {}

    fn on_bad_step(&mut self) {}
}
