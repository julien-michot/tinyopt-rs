// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! L-BFGS optimizer with a fixed-capacity correction history.

use num_traits::{FromPrimitive, One, ToPrimitive, Zero};

use crate::backend::{DenseVector, Matrix, Vector, zeros_vector};
use crate::math::all_finite;
use crate::optimizers::fixed::FixedSet;
use crate::optimizers::impl_fixed_api;
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;

const HISTORY_CAPACITY: usize = 8;

/// L-BFGS optimizer, mirroring `tinyopt::lbfgs::Optimizer`.
#[derive(Clone, Debug)]
pub struct Optimizer<P: Params> {
    options: Options,
    state: Option<State<P>>,
    fixed: FixedSet,
}

#[derive(Clone, Debug)]
struct State<P: Params> {
    steps: [Vector<P::Scalar, P::Dim>; HISTORY_CAPACITY],
    gradients: [Vector<P::Scalar, P::Dim>; HISTORY_CAPACITY],
    alpha: [f64; HISTORY_CAPACITY],
    rho: [f64; HISTORY_CAPACITY],
    previous_gradient: Vector<P::Scalar, P::Dim>,
    update_step: Vector<P::Scalar, P::Dim>,
    gradient_delta: Vector<P::Scalar, P::Dim>,
    last_step: Vector<P::Scalar, P::Dim>,
    step_size: f64,
    history_limit: usize,
    history_count: usize,
    history_start: usize,
    has_gradient: bool,
    has_step: bool,
    update_available: bool,
}

impl<P: Params> Optimizer<P> {
    impl_fixed_api!();

    /// Create an optimizer with the given options.
    pub fn new(options: Options) -> Self {
        Self {
            options,
            state: None,
            fixed: FixedSet::default(),
        }
    }

    /// Options in use.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Current step size.
    pub fn step_size(&self) -> f64 {
        self.state.as_ref().map_or(0.0, |s| s.step_size)
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

fn make_history<P: Params>(dims: usize) -> [Vector<P::Scalar, P::Dim>; HISTORY_CAPACITY] {
    core::array::from_fn(|_| zeros_vector::<P::Scalar, P::Dim>(dims))
}

impl<P: Params> Algorithm<P> for Optimizer<P> {
    fn fixed(&self) -> &FixedSet {
        &self.fixed
    }
    fn needs_hessian(&self) -> bool {
        false
    }

    fn reset(&mut self, dims: usize) {
        let history_limit = (self.options.lbfgs.history_size as usize).clamp(1, HISTORY_CAPACITY);
        self.state = Some(State {
            steps: make_history::<P>(dims),
            gradients: make_history::<P>(dims),
            alpha: [0.0; HISTORY_CAPACITY],
            rho: [0.0; HISTORY_CAPACITY],
            previous_gradient: zeros_vector::<P::Scalar, P::Dim>(dims),
            update_step: zeros_vector::<P::Scalar, P::Dim>(dims),
            gradient_delta: zeros_vector::<P::Scalar, P::Dim>(dims),
            last_step: zeros_vector::<P::Scalar, P::Dim>(dims),
            step_size: self.options.lbfgs.step_size as f64,
            history_limit,
            history_count: 0,
            history_start: 0,
            has_gradient: false,
            has_step: false,
            update_available: false,
        });
    }

    fn on_build(&mut self, grad: &Vector<P::Scalar, P::Dim>) {
        let state = self.state.as_mut().expect("reset() must be called first");
        if state.has_gradient && state.has_step {
            state.update_step.copy_from(&state.last_step);
            for i in 0..grad.len() {
                state
                    .gradient_delta
                    .set_at(i, grad.at(i) - state.previous_gradient.at(i));
            }
            state.update_available = true;
        } else {
            state.update_available = false;
        }
        state.previous_gradient.copy_from(grad);
        state.has_gradient = true;
        state.has_step = false;
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
        let state = self.state.as_mut().expect("reset() must be called first");
        let capacity = HISTORY_CAPACITY;
        let count = state.history_count;

        // First loop.
        let mut q = grad.clone();
        for offset in 0..count {
            let index = (state.history_start + count - offset - 1) % capacity;
            state.alpha[index] = state.rho[index]
                * DenseVector::dot(&state.steps[index], &q)
                    .to_f64()
                    .unwrap_or(0.0);
            let a = P::Scalar::from_f64(state.alpha[index]).expect("alpha fits");
            let g = state.gradients[index].clone();
            DenseVector::axpy(&mut q, -a, &g);
        }

        // Scaling.
        let mut scale = 1.0;
        if count > 0 {
            let newest = (state.history_start + count - 1) % capacity;
            let yy = state.gradients[newest].norm2().to_f64().unwrap_or(0.0);
            if yy > f64::EPSILON {
                scale = DenseVector::dot(&state.steps[newest], &state.gradients[newest])
                    .to_f64()
                    .unwrap_or(1.0)
                    / yy;
            }
        }
        let mut r = q;
        DenseVector::scale(&mut r, P::Scalar::from_f64(scale).expect("scale fits"));

        // Second loop.
        for offset in 0..count {
            let index = (state.history_start + offset) % capacity;
            let beta = state.rho[index]
                * DenseVector::dot(&state.gradients[index], &r)
                    .to_f64()
                    .unwrap_or(0.0);
            let a = P::Scalar::from_f64(state.alpha[index] - beta).expect("coefficient fits");
            let s = state.steps[index].clone();
            DenseVector::axpy(&mut r, a, &s);
        }

        let mut direction = r;
        DenseVector::scale(&mut direction, -P::Scalar::one());
        if !all_finite(DenseVector::as_slice(&direction))
            || DenseVector::dot(grad, &direction) >= P::Scalar::zero()
        {
            state.history_count = 0;
            state.history_start = 0;
            direction = grad.clone();
            DenseVector::scale(&mut direction, -P::Scalar::one());
        }

        let step_scale = P::Scalar::from_f64(state.step_size).expect("step size fits");
        DenseVector::scale(&mut direction, step_scale);
        state.last_step.copy_from(&direction);
        state.has_step = true;
        Some((direction, 0.0))
    }

    fn update_step(&mut self, step: &Vector<P::Scalar, P::Dim>) {
        if let Some(state) = self.state.as_mut() {
            state.last_step.copy_from(step);
        }
    }

    fn on_good_step(&mut self, _quality: f64) {
        let options = self.options.clone();
        let state = self.state.as_mut().expect("reset() must be called first");
        update_lbfgs(state, &options);
        state.update_available = false;
        state.step_size = (state.step_size * options.lbfgs.step_growth as f64)
            .min(options.lbfgs.max_step_size as f64);
    }

    fn on_bad_step(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.step_size *= self.options.lbfgs.step_reduction as f64;
            state.has_gradient = false;
            state.has_step = false;
            state.update_available = false;
        }
    }
}

fn update_lbfgs<P: Params>(state: &mut State<P>, options: &Options) {
    if !state.update_available {
        return;
    }
    let curvature = DenseVector::dot(&state.update_step, &state.gradient_delta)
        .to_f64()
        .unwrap_or(f64::NAN);
    let threshold = options.lbfgs.curvature_threshold as f64
        * state.update_step.norm().to_f64().unwrap_or(0.0)
        * state.gradient_delta.norm().to_f64().unwrap_or(0.0);
    if !curvature.is_finite() || curvature <= threshold {
        return;
    }

    let index = if state.history_count < state.history_limit {
        let idx = (state.history_start + state.history_count) % HISTORY_CAPACITY;
        state.history_count += 1;
        idx
    } else {
        let idx = state.history_start;
        state.history_start = (state.history_start + 1) % HISTORY_CAPACITY;
        idx
    };
    state.steps[index].copy_from(&state.update_step);
    state.gradients[index].copy_from(&state.gradient_delta);
    state.rho[index] = 1.0 / curvature;
}
