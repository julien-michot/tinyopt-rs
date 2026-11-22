// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Levenberg-Marquardt optimizer.

use num_traits::{FromPrimitive, One, ToPrimitive};

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

/// Levenberg-Marquardt optimizer, mirroring `tinyopt::lm::Optimizer`.
///
/// Solves `(JᵀJ + λ diag(JᵀJ)) dx = -Jᵀr` with adaptive damping, optionally
/// preceded by Jacobi scaling of the normal equations.
#[derive(Clone, Debug)]
pub struct Optimizer<
    P: Params,
    H: HessianStorage<<P as Params>::Scalar> = Matrix<<P as Params>::Scalar, <P as Params>::Dim>,
> {
    options: Options,
    state: Option<State<P>>,
    fixed: FixedSet,
    num_reduced: usize,
    final_h: Option<H>,
    final_cost: Cost,
    prototype: Option<H>,
    cache: H::SolverCache,
    _marker: core::marker::PhantomData<P>,
}

#[derive(Clone, Debug)]
struct State<P: Params> {
    lambda: f64,
    bad_factor: f64,
    scaling: Vector<P::Scalar, P::Dim>,
    diag: Vector<P::Scalar, P::Dim>,
}

impl<P: Params, H: HessianStorage<P::Scalar>> Optimizer<P, H> {
    impl_fixed_api!();

    /// Provide a Hessian prototype (required for dynamic block layouts).
    pub fn init_with(&mut self, prototype: H) -> &mut Self {
        self.prototype = Some(prototype);
        self
    }

    /// Posterior covariance estimate (default options).
    pub fn covariance(&self) -> Option<FinalCovariance> {
        <Self as Algorithm<P, H>>::covariance(self, &CovarianceOptions::default())
    }

    /// Posterior covariance estimate with explicit options.
    pub fn covariance_with(&self, options: &CovarianceOptions) -> Option<FinalCovariance> {
        <Self as Algorithm<P, H>>::covariance(self, options)
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

    /// Current damping factor.
    pub fn lambda(&self) -> f64 {
        self.state.as_ref().map_or(0.0, |s| s.lambda)
    }

    /// Whether a Schur reduced block was detected.
    pub fn has_reduced(&self) -> bool {
        self.num_reduced > 0
    }

    /// Number of reduced (head) parameters.
    pub fn num_reduced(&self) -> usize {
        self.num_reduced
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
    fn reset(&mut self, dims: usize) {
        let lambda = self.options.lm.damping_init as f64;
        self.state = Some(State {
            lambda,
            bad_factor: self.options.lm.bad_factor as f64,
            scaling: zeros_vector::<P::Scalar, P::Dim>(dims),
            diag: zeros_vector::<P::Scalar, P::Dim>(dims),
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

    fn state_string(&self) -> String {
        match self.state.as_ref() {
            Some(state) if state.lambda > 0.0 => format!("○:{:.2e} ", 1.0 / state.lambda),
            _ => String::new(),
        }
    }

    fn precondition(&mut self, grad: &mut Vector<P::Scalar, P::Dim>, h: &mut H) {
        let state = self.state.as_mut().expect("reset() must be called first");
        let n = h.dims();

        // Jacobi scaling: D = 1/sqrt(clamp(diag(H), 1e-6, 1e32)).
        if self.options.lm.jacobi_scaling {
            h.diagonal_into(DenseVector::as_mut_slice(&mut state.diag));
            for i in 0..n {
                let d = state.diag.at(i).to_f64().unwrap_or(1.0).clamp(1e-6, 1e32);
                let s = P::Scalar::from_f64(1.0 / d.sqrt()).expect("fits");
                state.scaling.set_at(i, s);
            }
            for i in 0..n {
                let s = state.scaling.at(i);
                grad.set_at(i, grad.at(i) * s);
            }
            h.scale_by_diagonal(DenseVector::as_slice(&state.scaling), false);
        } else {
            for i in 0..n {
                state.scaling.set_at(i, P::Scalar::one());
            }
        }

        // Damping: H += λ diag(H).
        if state.lambda > 0.0 {
            let factor = P::Scalar::from_f64(1.0 + state.lambda).expect("fits");
            h.scale_diagonal(factor);
        }
        h.diagonal_into(DenseVector::as_mut_slice(&mut state.diag));
    }

    fn solve(
        &mut self,
        grad: &Vector<P::Scalar, P::Dim>,
        h: &H,
    ) -> Option<Step<P::Scalar, P::Dim>> {
        let mut step = neg_solve::<P::Scalar, P::Dim, H>(
            &self.options,
            grad,
            h,
            self.num_reduced,
            &mut self.cache,
            &self.fixed,
        )?;
        let state = self.state.as_ref().expect("reset() must be called first");
        let n = step.len();

        // Predicted reduction, computed in the same (scaled or raw) frame as
        // the solved system, exactly like the C++ SolveLM().
        let mut g_dot_dx = 0.0;
        if self.options.lm.jacobi_scaling {
            // Scaled frame: H_s = D H D, g_s = D g, step_s = D^-1 step.
            for i in 0..n {
                g_dot_dx += grad.at(i).to_f64().unwrap_or(0.0) * step.at(i).to_f64().unwrap_or(0.0);
            }
            let step_norm2 = step.norm2().to_f64().unwrap_or(0.0);
            let predicted = 0.5 * (-g_dot_dx + state.lambda * step_norm2);

            // Unscale the step back to the original coordinates.
            for i in 0..n {
                step.set_at(i, step.at(i) * state.scaling.at(i));
            }
            Some((step, predicted.max(0.0)))
        } else {
            let mut dx_d_dx = 0.0;
            for i in 0..n {
                let gi = grad.at(i).to_f64().unwrap_or(0.0);
                let di = step.at(i).to_f64().unwrap_or(0.0);
                let d = state.diag.at(i).to_f64().unwrap_or(0.0);
                g_dot_dx += gi * di;
                dx_d_dx += d * di * di;
            }
            let predicted = 0.5 * (-g_dot_dx + state.lambda * dx_d_dx);
            Some((step, predicted.max(0.0)))
        }
    }

    fn on_good_step(&mut self, quality: f64) {
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };
        if state.lambda <= 0.0 {
            return;
        }
        let q = if quality.is_finite() { quality } else { 0.0 };
        let factor = if q != 0.0 {
            (1.0 - (2.0 * q - 1.0).powi(3)).max(self.options.lm.good_factor as f64)
        } else {
            self.options.lm.good_factor as f64
        };
        state.lambda = clamp(
            state.lambda * factor,
            self.options.lm.damping_range[0] as f64,
            self.options.lm.damping_range[1] as f64,
        );
        state.bad_factor = self.options.lm.bad_factor as f64;
    }

    fn on_bad_step(&mut self) {
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };
        if state.lambda <= 0.0 {
            return;
        }
        state.lambda = clamp(
            state.lambda * state.bad_factor,
            self.options.lm.damping_range[0] as f64,
            self.options.lm.damping_range[1] as f64,
        );
        state.bad_factor *= self.options.lm.bad_factor as f64;
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

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}
