// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Non-linear conjugate gradient optimizer (Polak-Ribière+).

use num_traits::{FromPrimitive, ToPrimitive, Zero};

use crate::backend::{DenseVector, Matrix, Vector, zeros_vector};
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;

/// Conjugate gradient optimizer, mirroring `tinyopt::cg::Optimizer`.
#[derive(Clone, Debug)]
pub struct Optimizer<P: Params> {
    options: Options,
    state: Option<State<P>>,
}

#[derive(Clone, Debug)]
struct State<P: Params> {
    previous_gradient: Vector<P::Scalar, P::Dim>,
    previous_direction: Vector<P::Scalar, P::Dim>,
    direction: Vector<P::Scalar, P::Dim>,
    step_size: f64,
    has_previous: bool,
}

impl<P: Params> Optimizer<P> {
    /// Create an optimizer with the given options.
    pub fn new(options: Options) -> Self {
        Self {
            options,
            state: None,
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

impl<P: Params> Algorithm<P> for Optimizer<P> {
    fn needs_hessian(&self) -> bool {
        false
    }

    fn reset(&mut self, dims: usize) {
        self.state = Some(State {
            previous_gradient: zeros_vector::<P::Scalar, P::Dim>(dims),
            previous_direction: zeros_vector::<P::Scalar, P::Dim>(dims),
            direction: zeros_vector::<P::Scalar, P::Dim>(dims),
            step_size: self.options.cg.step_size as f64,
            has_previous: false,
        });
    }

    fn on_build(&mut self, grad: &Vector<P::Scalar, P::Dim>) {
        let state = self.state.as_mut().expect("reset() must be called first");
        let n = grad.len();

        // Polak-Ribiere+ update.
        let mut beta = 0.0;
        if state.has_previous {
            let denominator = state.previous_gradient.norm2().to_f64().unwrap_or(0.0);
            if denominator > f64::EPSILON {
                let mut numerator = 0.0;
                for i in 0..n {
                    numerator += grad.at(i).to_f64().unwrap_or(0.0)
                        * (grad.at(i) - state.previous_gradient.at(i))
                            .to_f64()
                            .unwrap_or(0.0);
                }
                beta = (numerator / denominator).max(0.0);
            }
        }

        // direction = -grad + beta * previous_direction
        let beta = P::Scalar::from_f64(beta).expect("beta fits");
        for i in 0..n {
            let d = -grad.at(i) + beta * state.previous_direction.at(i);
            state.direction.set_at(i, d);
        }

        // Restart when the direction is not a descent direction.
        if DenseVector::dot(grad, &state.direction) >= P::Scalar::zero() {
            for i in 0..n {
                state.direction.set_at(i, -grad.at(i));
            }
        }

        state.previous_gradient.copy_from(grad);
        state.previous_direction.copy_from(&state.direction);
        state.has_previous = true;
    }

    fn precondition(
        &mut self,
        _grad: &mut Vector<P::Scalar, P::Dim>,
        _h: &mut Matrix<P::Scalar, P::Dim>,
    ) {
    }

    fn solve(
        &mut self,
        _grad: &Vector<P::Scalar, P::Dim>,
        _h: &Matrix<P::Scalar, P::Dim>,
    ) -> Option<Step<P::Scalar, P::Dim>> {
        let state = self.state.as_ref().expect("reset() must be called first");
        let scale = P::Scalar::from_f64(state.step_size).expect("step size fits");
        let mut dx = state.direction.clone();
        DenseVector::scale(&mut dx, scale);
        Some((dx, 0.0))
    }

    fn on_good_step(&mut self, _quality: f64) {}

    fn on_bad_step(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.step_size *= self.options.cg.step_reduction as f64;
            state.has_previous = false;
        }
    }
}
