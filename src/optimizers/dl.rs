// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Dog-leg trust-region optimizer.

use num_traits::ToPrimitive;

use crate::backend::{DenseVector, Matrix, Vector, zeros_vector};
use crate::math::mat_vec;
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, neg_solve, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;

/// Dog-leg optimizer, mirroring `tinyopt::dl::Optimizer`.
#[derive(Clone, Debug)]
pub struct Optimizer<P: Params> {
    options: Options,
    state: Option<State>,
    _marker: core::marker::PhantomData<P>,
}

#[derive(Clone, Debug)]
struct State {
    radius: f64,
    pending_prediction: f64,
    pending_boundary: bool,
}

impl<P: Params> Optimizer<P> {
    /// Create an optimizer with the given options.
    pub fn new(options: Options) -> Self {
        Self {
            options,
            state: None,
            _marker: core::marker::PhantomData,
        }
    }

    /// Options in use.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Current trust-region radius.
    pub fn radius(&self) -> f64 {
        self.state.as_ref().map_or(0.0, |s| s.radius)
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

fn scaled<S: crate::real::Real, D: crate::backend::Dim>(
    factor: f64,
    v: &Vector<S, D>,
) -> Vector<S, D> {
    let mut out = v.clone();
    DenseVector::scale(&mut out, S::from_f64(factor).expect("factor fits"));
    out
}

impl<P: Params> Algorithm<P> for Optimizer<P> {
    fn reset(&mut self, _dims: usize) {
        self.state = Some(State {
            radius: (self.options.dl.radius_init as f64).max(f64::EPSILON),
            pending_prediction: 0.0,
            pending_boundary: false,
        });
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
        h: &Matrix<P::Scalar, P::Dim>,
    ) -> Option<Step<P::Scalar, P::Dim>> {
        let state = self.state.as_mut().expect("reset() must be called first");
        let n = grad.len();
        let gradient_norm2 = grad.norm2().to_f64().unwrap_or(f64::NAN);
        if !gradient_norm2.is_finite() {
            return None;
        }
        if gradient_norm2 <= f64::EPSILON {
            return Some((zeros_vector::<P::Scalar, P::Dim>(n), 0.0));
        }
        let gradient_norm = gradient_norm2.sqrt();

        // Gauss-Newton step and Cauchy point.
        let gauss_newton = neg_solve::<P::Scalar, P::Dim>(&self.options, grad, h);
        let hg = mat_vec::<P::Scalar, P::Dim>(h, grad);
        let curvature = DenseVector::dot(grad, &hg).to_f64().unwrap_or(0.0);
        let cauchy = if curvature > f64::EPSILON {
            scaled::<P::Scalar, P::Dim>(-(gradient_norm2 / curvature), grad)
        } else {
            scaled::<P::Scalar, P::Dim>(-(state.radius / gradient_norm), grad)
        };

        let cauchy_norm = cauchy.norm().to_f64().unwrap_or(0.0);
        let mut on_boundary = false;
        let step = if let Some(gn) = &gauss_newton {
            if gn.norm().to_f64().unwrap_or(f64::INFINITY) <= state.radius {
                gn.clone()
            } else if cauchy_norm >= state.radius {
                scaled::<P::Scalar, P::Dim>(state.radius / cauchy_norm, &cauchy)
            } else {
                on_boundary = true;
                dogleg_interpolation::<P::Scalar, P::Dim>(&cauchy, gn, state.radius)
            }
        } else if cauchy_norm >= state.radius {
            scaled::<P::Scalar, P::Dim>(state.radius / cauchy_norm, &cauchy)
        } else {
            on_boundary = true;
            cauchy
        };

        if cauchy_norm >= state.radius || gauss_newton.is_none() {
            on_boundary = true;
        }

        // Predicted reduction of the model.
        let h_step = mat_vec::<P::Scalar, P::Dim>(h, &step);
        let predicted = -DenseVector::dot(grad, &step).to_f64().unwrap_or(0.0)
            - 0.5 * DenseVector::dot(&step, &h_step).to_f64().unwrap_or(0.0);

        state.pending_prediction = predicted;
        state.pending_boundary = on_boundary;
        Some((step, predicted))
    }

    fn on_good_step(&mut self, quality: f64) {
        if let Some(state) = self.state.as_mut() {
            if quality < 0.25 {
                state.radius *= self.options.dl.shrink_factor as f64;
            } else if quality > 0.75 && state.pending_boundary {
                state.radius = (state.radius * self.options.dl.expand_factor as f64)
                    .min(self.options.dl.radius_max as f64);
            }
            state.pending_boundary = false;
        }
    }

    fn on_bad_step(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.radius *= self.options.dl.shrink_factor as f64;
            state.pending_boundary = false;
        }
    }
}

/// Dog-leg interpolation between the Cauchy point and the Gauss-Newton step.
fn dogleg_interpolation<S: crate::real::Real, D: crate::backend::Dim>(
    cauchy: &Vector<S, D>,
    gauss_newton: &Vector<S, D>,
    radius: f64,
) -> Vector<S, D> {
    let mut difference = gauss_newton.clone();
    let mut neg_cauchy = cauchy.clone();
    DenseVector::scale(&mut neg_cauchy, -S::one());
    DenseVector::axpy(&mut difference, S::one(), &neg_cauchy);

    let a = difference.norm2().to_f64().unwrap_or(0.0);
    let b = 2.0
        * DenseVector::dot(cauchy, &difference)
            .to_f64()
            .unwrap_or(0.0);
    let c = cauchy.norm2().to_f64().unwrap_or(0.0) - radius * radius;
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let tau = ((-b + discriminant.sqrt()) / (2.0 * a)).clamp(0.0, 1.0);

    let mut step = cauchy.clone();
    let mut scaled_diff = difference;
    DenseVector::scale(&mut scaled_diff, S::from_f64(tau).expect("tau fits"));
    DenseVector::axpy(&mut step, S::one(), &scaled_diff);
    step
}
