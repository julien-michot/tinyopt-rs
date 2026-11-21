// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Gauss-Newton optimizer.

use crate::backend::{Matrix, Vector};
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, neg_solve, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;

/// Gauss-Newton optimizer, mirroring `tinyopt::gn::Optimizer`.
///
/// Solves the normal equations `JᵀJ dx = -Jᵀr` directly, without damping.
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

    fn precondition(
        &mut self,
        _grad: &mut Vector<P::Scalar, P::Dim>,
        _h: &mut Matrix<P::Scalar, P::Dim>,
    ) {
    }

    fn solve(
        &mut self,
        grad: &Vector<P::Scalar, P::Dim>,
        h: &Matrix<P::Scalar, P::Dim>,
    ) -> Option<Step<P::Scalar, P::Dim>> {
        neg_solve::<P::Scalar, P::Dim>(&self.options, grad, h).map(|dx| (dx, 0.0))
    }

    fn on_good_step(&mut self, _quality: f64) {}

    fn on_bad_step(&mut self) {}
}
