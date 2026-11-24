// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! BFGS optimizer with a dense inverse-Hessian approximation.

use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};

use crate::backend::{
    DenseMatrix, DenseVector, Dynamic, Matrix, Vector, zeros_matrix, zeros_vector,
};
use crate::covariance::{CovarianceOptions, FinalCovariance};
use crate::hessian::HessianStorage;
use crate::math::{all_finite, mat_identity, mat_mul, mat_transpose, mat_vec};
use crate::optimizers::fixed::FixedSet;
use crate::optimizers::impl_fixed_api;
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, run};
use crate::options::Options;
use crate::params::Params;
use crate::summary::Summary;

/// BFGS optimizer, mirroring `tinyopt::bfgs::Optimizer`.
///
/// Maintains a dense inverse-Hessian approximation, also exposed as the
/// posterior covariance estimate.
#[derive(Clone, Debug)]
pub struct Optimizer<
    P: Params,
    H: HessianStorage<<P as Params>::Scalar> = Matrix<<P as Params>::Scalar, <P as Params>::Dim>,
> {
    options: Options,
    state: Option<State<P>>,
    fixed: FixedSet,
    _marker: core::marker::PhantomData<H>,
}

#[derive(Clone, Debug)]
struct State<P: Params> {
    inverse_hessian: Matrix<P::Scalar, P::Dim>,
    previous_gradient: Vector<P::Scalar, P::Dim>,
    update_step: Vector<P::Scalar, P::Dim>,
    gradient_delta: Vector<P::Scalar, P::Dim>,
    last_step: Vector<P::Scalar, P::Dim>,
    step_size: f64,
    matrix_initialized: bool,
    has_gradient: bool,
    has_step: bool,
    update_available: bool,
}

impl<P: Params, H: HessianStorage<P::Scalar>> Optimizer<P, H> {
    impl_fixed_api!();

    /// Create an optimizer with the given options.
    pub fn new(options: Options) -> Self {
        Self {
            options,
            state: None,
            fixed: FixedSet::default(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Options in use.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Current inverse-Hessian approximation (the covariance estimate).
    pub fn covariance(&self) -> Option<Matrix<P::Scalar, P::Dim>> {
        self.state
            .as_ref()
            .filter(|s| s.matrix_initialized)
            .map(|s| s.inverse_hessian.clone())
    }

    /// Current step size.
    pub fn step_size(&self) -> f64 {
        self.state.as_ref().map_or(0.0, |s| s.step_size)
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
    fn needs_hessian(&self) -> bool {
        false
    }

    fn fixed(&self) -> &FixedSet {
        &self.fixed
    }

    fn reset(&mut self, dims: usize) {
        self.state = Some(State {
            inverse_hessian: mat_identity::<P::Scalar, P::Dim>(dims),
            previous_gradient: zeros_vector::<P::Scalar, P::Dim>(dims),
            update_step: zeros_vector::<P::Scalar, P::Dim>(dims),
            gradient_delta: zeros_vector::<P::Scalar, P::Dim>(dims),
            last_step: zeros_vector::<P::Scalar, P::Dim>(dims),
            step_size: self.options.bfgs.step_size as f64,
            matrix_initialized: true,
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

    fn precondition(&mut self, _grad: &mut Vector<P::Scalar, P::Dim>, _h: &mut H) {}

    fn solve(
        &mut self,
        grad: &Vector<P::Scalar, P::Dim>,
        _h: &H,
    ) -> Option<Step<P::Scalar, P::Dim>> {
        let state = self.state.as_mut().expect("reset() must be called first");
        let n = grad.len();

        let mut direction = mat_vec::<P::Scalar, P::Dim>(&state.inverse_hessian, grad);
        DenseVector::scale(&mut direction, -P::Scalar::one());
        if !all_finite(DenseVector::as_slice(&direction))
            || DenseVector::dot(grad, &direction) >= P::Scalar::zero()
        {
            state.inverse_hessian = mat_identity::<P::Scalar, P::Dim>(n);
            direction = grad.clone();
            DenseVector::scale(&mut direction, -P::Scalar::one());
        }

        let scale = P::Scalar::from_f64(state.step_size).expect("step size fits");
        DenseVector::scale(&mut direction, scale);
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
        update_bfgs(state, &options);
        state.update_available = false;
        state.step_size = (state.step_size * options.bfgs.step_growth as f64)
            .min(options.bfgs.max_step_size as f64);
    }

    fn on_bad_step(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.step_size *= self.options.bfgs.step_reduction as f64;
            state.has_gradient = false;
            state.has_step = false;
            state.update_available = false;
        }
    }

    fn covariance(&self, _options: &CovarianceOptions) -> Option<FinalCovariance> {
        let state = self.state.as_ref().filter(|s| s.matrix_initialized)?;
        let n = state.inverse_hessian.nrows();
        let mut out = zeros_matrix::<f32, Dynamic>(n, n);
        for i in 0..n {
            for j in 0..n {
                out.set_elem(
                    i,
                    j,
                    state
                        .inverse_hessian
                        .elem(i, j)
                        .to_f32()
                        .expect("fits in f32"),
                );
            }
        }
        Some(FinalCovariance::Dense(out))
    }
}

fn update_bfgs<P: Params>(state: &mut State<P>, options: &Options) {
    if !state.update_available {
        return;
    }
    let n = state.update_step.len();
    let curvature = DenseVector::dot(&state.update_step, &state.gradient_delta);
    let threshold = P::Scalar::from_f64(options.bfgs.curvature_threshold as f64)
        .expect("threshold fits")
        * state.update_step.norm()
        * state.gradient_delta.norm();
    if !curvature.is_finite() || curvature <= threshold {
        return;
    }

    let inv_c = P::Scalar::one() / curvature;

    let mut transform = mat_identity::<P::Scalar, P::Dim>(n);
    for i in 0..n {
        for j in 0..n {
            let v =
                transform.elem(i, j) - inv_c * state.update_step.at(i) * state.gradient_delta.at(j);
            transform.set_elem(i, j, v);
        }
    }

    let tmp = mat_mul::<P::Scalar, P::Dim>(&transform, &state.inverse_hessian);
    let mut updated =
        mat_mul::<P::Scalar, P::Dim>(&tmp, &mat_transpose::<P::Scalar, P::Dim>(&transform));
    for i in 0..n {
        for j in 0..n {
            let v = updated.elem(i, j) + inv_c * state.update_step.at(i) * state.update_step.at(j);
            updated.set_elem(i, j, v);
        }
    }
    state.inverse_hessian = updated;
}
