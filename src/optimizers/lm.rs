// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Levenberg-Marquardt optimizer.

use crate::backend::{DenseMatrix, DenseVector, Matrix, Vector, zeros_vector};
use crate::cost::Cost;
use crate::covariance::{CovarianceOptions, compute_covariance};
use crate::optimizers::fixed::FixedSet;
use crate::optimizers::impl_fixed_api;
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, neg_solve, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;
use num_traits::{FromPrimitive, One, ToPrimitive};

/// Levenberg-Marquardt optimizer, mirroring `tinyopt::lm::Optimizer`.
///
/// Solves `(JᵀJ + λ diag(JᵀJ)) dx = -Jᵀr` with adaptive damping, optionally
/// preceded by Jacobi scaling of the normal equations.
#[derive(Clone, Debug)]
pub struct Optimizer<P: Params> {
    options: Options,
    state: Option<State<P>>,
    final_h: Option<Matrix<P::Scalar, P::Dim>>,
    final_cost: Cost,
    fixed: FixedSet,
}

#[derive(Clone, Debug)]
struct State<P: Params> {
    lambda: f64,
    bad_factor: f64,
    scaling: Vector<P::Scalar, P::Dim>,
}

impl<P: Params> Optimizer<P> {
    impl_fixed_api!();

    /// Posterior covariance estimate (with default options).
    pub fn covariance(&self) -> Option<Matrix<P::Scalar, P::Dim>> {
        <Self as Algorithm<P>>::covariance(self, &CovarianceOptions::default())
    }

    /// Posterior covariance estimate with explicit options.
    pub fn covariance_with(
        &self,
        options: &CovarianceOptions,
    ) -> Option<Matrix<P::Scalar, P::Dim>> {
        <Self as Algorithm<P>>::covariance(self, options)
    }

    /// Create an optimizer with the given options.
    pub fn new(options: Options) -> Self {
        Self {
            options,
            state: None,
            final_h: None,
            final_cost: Cost::default(),
            fixed: FixedSet::default(),
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
    fn fixed(&self) -> &FixedSet {
        &self.fixed
    }

    fn save_final_system(&mut self, h: &Matrix<P::Scalar, P::Dim>, cost: Cost) {
        self.final_h = Some(h.clone());
        self.final_cost = cost;
    }

    fn covariance(&self, options: &CovarianceOptions) -> Option<Matrix<P::Scalar, P::Dim>> {
        let h = self.final_h.as_ref()?;
        compute_covariance::<P::Scalar, P::Dim>(
            h,
            options,
            self.final_cost,
            self.fixed.indices(),
            h.nrows(),
        )
    }
    fn reset(&mut self, dims: usize) {
        let lambda = self.options.lm.damping_init as f64;
        self.state = Some(State {
            lambda,
            bad_factor: self.options.lm.bad_factor as f64,
            scaling: zeros_vector::<P::Scalar, P::Dim>(dims),
        });
    }

    fn precondition(
        &mut self,
        grad: &mut Vector<P::Scalar, P::Dim>,
        h: &mut Matrix<P::Scalar, P::Dim>,
    ) {
        let state = self.state.as_mut().expect("reset() must be called first");
        let n = h.nrows();
        let scalar = P::Scalar::one();

        // Jacobi scaling: D = 1/sqrt(clamp(diag(H), 1e-6, 1e32)).
        if self.options.lm.jacobi_scaling {
            for i in 0..n {
                let d = h.elem(i, i).to_f64().unwrap_or(1.0).clamp(1e-6, 1e32);
                let s = P::Scalar::from_f64(1.0 / d.sqrt()).expect("fits");
                state.scaling.set_at(i, s);
            }
            for i in 0..n {
                let s = state.scaling.at(i);
                grad.set_at(i, grad.at(i) * s);
            }
            for j in 0..n {
                for i in 0..n {
                    let v = h.elem(i, j) * state.scaling.at(i) * state.scaling.at(j);
                    h.set_elem(i, j, v);
                }
            }
        } else {
            for i in 0..n {
                state.scaling.set_at(i, scalar);
            }
        }

        // Damping: H += λ diag(H).
        if state.lambda > 0.0 {
            let factor = P::Scalar::from_f64(1.0 + state.lambda).expect("fits");
            for i in 0..n {
                h.set_elem(i, i, h.elem(i, i) * factor);
            }
        }
    }

    fn solve(
        &mut self,
        grad: &Vector<P::Scalar, P::Dim>,
        h: &Matrix<P::Scalar, P::Dim>,
    ) -> Option<Step<P::Scalar, P::Dim>> {
        let mut step = neg_solve::<P::Scalar, P::Dim>(&self.options, grad, h)?;
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
                let d = h.elem(i, i).to_f64().unwrap_or(0.0);
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
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}
