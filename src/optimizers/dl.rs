// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Dog-leg trust-region optimizer.

use num_traits::ToPrimitive;

use crate::backend::{DenseVector, Matrix, Vector, zeros_vector};
use crate::cost::Cost;
use crate::covariance::{CovarianceOptions, FinalCovariance};
use crate::hessian::HessianStorage;
use crate::optimizers::fixed::FixedSet;
use crate::optimizers::impl_fixed_api;
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, neg_solve, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;

/// Dog-leg optimizer, mirroring `tinyopt::dl::Optimizer`.
#[derive(Clone, Debug)]
pub struct Optimizer<
    P: Params,
    H: HessianStorage<<P as Params>::Scalar> = Matrix<<P as Params>::Scalar, <P as Params>::Dim>,
> {
    options: Options,
    state: Option<State>,
    fixed: FixedSet,
    num_reduced: usize,
    final_h: Option<H>,
    final_cost: Cost,
    prototype: Option<H>,
    cache: H::SolverCache,
    _marker: core::marker::PhantomData<P>,
}

#[derive(Clone, Debug)]
struct State {
    radius: f64,
    pending_prediction: f64,
    pending_boundary: bool,
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
            state: None,
            fixed: FixedSet::default(),
            num_reduced: 0,
            final_h: None,
            final_cost: Cost::default(),
            prototype: None,
            cache: H::SolverCache::default(),
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

fn scaled<S: crate::real::Real, D: crate::backend::Dim>(
    factor: f64,
    v: &Vector<S, D>,
) -> Vector<S, D> {
    let mut out = v.clone();
    DenseVector::scale(&mut out, S::from_f64(factor).expect("factor fits"));
    out
}

impl<P: Params, H: HessianStorage<P::Scalar>> Algorithm<P, H> for Optimizer<P, H> {
    fn reset(&mut self, _dims: usize) {
        self.state = Some(State {
            radius: (self.options.dl.radius_init as f64).max(f64::EPSILON),
            pending_prediction: 0.0,
            pending_boundary: false,
        });
    }

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
        let grad_slice = DenseVector::as_slice(grad);

        // Gauss-Newton step and Cauchy point.
        let gauss_newton = neg_solve::<P::Scalar, P::Dim, H>(
            &self.options,
            grad,
            h,
            self.num_reduced,
            &mut self.cache,
            &self.fixed,
        );
        let hg = h.mul_vec(grad_slice);
        let curvature = (0..n).fold(0.0, |acc, i| {
            acc + grad.at(i).to_f64().unwrap_or(0.0) * hg[i].to_f64().unwrap_or(0.0)
        });
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
        let h_step = h.mul_vec(DenseVector::as_slice(&step));
        let predicted = -(0..n).fold(0.0, |acc, i| {
            acc + grad.at(i).to_f64().unwrap_or(0.0) * step.at(i).to_f64().unwrap_or(0.0)
        }) - 0.5
            * (0..n).fold(0.0, |acc, i| {
                acc + step.at(i).to_f64().unwrap_or(0.0) * h_step[i].to_f64().unwrap_or(0.0)
            });

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
            state.pending_prediction = 0.0;
        }
    }

    fn on_bad_step(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.radius *= self.options.dl.shrink_factor as f64;
            state.pending_boundary = false;
            state.pending_prediction = 0.0;
        }
    }

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
