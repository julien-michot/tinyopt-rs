// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Steihaug–Toint trust-region Newton-CG optimizer.
//!
//! The trust-region subproblem `min ½ dᵀH d + gᵀd` is solved approximately with
//! conjugate gradients, stopping at the trust radius or on negative curvature
//! (Steihaug–Toint). It only needs Hessian-vector products, so it works with
//! dense, sparse and block-structured Hessians alike.

use num_traits::{FromPrimitive, ToPrimitive};

use crate::backend::{DenseVector, Matrix, Vector, zeros_vector};
use crate::cost::Cost;
use crate::covariance::{CovarianceOptions, FinalCovariance};
use crate::hessian::HessianStorage;
use crate::optimizers::fixed::FixedSet;
use crate::optimizers::impl_fixed_api;
use crate::optimizers::optimizer::{AccumulationFn, Algorithm, Step, run};
use crate::options::Options;
use crate::params::Params;
use crate::real::Real;
use crate::summary::Summary;

/// Steihaug–Toint trust-region Newton-CG optimizer.
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
    _marker: core::marker::PhantomData<P>,
}

#[derive(Clone, Debug)]
struct State<P: Params> {
    radius: f64,
    boundary: bool,
    direction: Vector<P::Scalar, P::Dim>,
    residual: Vector<P::Scalar, P::Dim>,
    conjugate: Vector<P::Scalar, P::Dim>,
    hessian_product: Vector<P::Scalar, P::Dim>,
    hessian_direction: Vector<P::Scalar, P::Dim>,
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
        F: AccumulationFn<P::Scalar, P::Dim, H>,
    {
        let options = self.options.clone();
        run(x, f, &options, self)
    }
}

fn dot<S: Real>(a: &[S], b: &[S]) -> f64 {
    a.iter().zip(b).fold(0.0, |acc, (x, y)| {
        acc + x.to_f64().expect("scalar") * y.to_f64().expect("scalar")
    })
}

/// Positive `tau` with `‖d + tau p‖ = radius`.
fn boundary_step<S: Real>(direction: &[S], conjugate: &[S], radius: f64) -> f64 {
    let a = dot(conjugate, conjugate);
    if a <= 0.0 {
        return 0.0;
    }
    let b = 2.0 * dot(direction, conjugate);
    let c = dot(direction, direction) - radius * radius;
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    ((-b + discriminant.sqrt()) / (2.0 * a)).clamp(0.0, f64::MAX)
}

impl<P: Params, H: HessianStorage<P::Scalar>> Algorithm<P, H> for Optimizer<P, H> {
    fn reset(&mut self, dims: usize) {
        self.state = Some(State {
            radius: (self.options.tr.radius_init as f64).max(f64::EPSILON),
            boundary: false,
            direction: zeros_vector::<P::Scalar, P::Dim>(dims),
            residual: zeros_vector::<P::Scalar, P::Dim>(dims),
            conjugate: zeros_vector::<P::Scalar, P::Dim>(dims),
            hessian_product: zeros_vector::<P::Scalar, P::Dim>(dims),
            hessian_direction: zeros_vector::<P::Scalar, P::Dim>(dims),
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
        let gradient_norm2 = dot(DenseVector::as_slice(grad), DenseVector::as_slice(grad));
        if !gradient_norm2.is_finite() {
            return None;
        }
        if gradient_norm2 <= f64::EPSILON {
            return Some((zeros_vector::<P::Scalar, P::Dim>(n), 0.0));
        }
        let gradient_norm = gradient_norm2.sqrt();
        let tolerance = (self.options.tr.cg_tolerance as f64) * gradient_norm;
        let radius = state.radius;
        let max_iters = self.options.tr.max_cg_iters as usize;

        // d = 0, r = -g, p = r.
        DenseVector::set_zero(&mut state.direction);
        for (r, g) in DenseVector::as_mut_slice(&mut state.residual)
            .iter_mut()
            .zip(DenseVector::as_slice(grad))
        {
            *r = -*g;
        }
        state.conjugate.copy_from(&state.residual);
        let mut rr = dot(
            DenseVector::as_slice(&state.residual),
            DenseVector::as_slice(&state.residual),
        );
        let mut boundary = false;

        for _ in 0..max_iters {
            h.mul_vec_into(
                DenseVector::as_slice(&state.conjugate),
                DenseVector::as_mut_slice(&mut state.hessian_product),
            );
            let php = dot(
                DenseVector::as_slice(&state.conjugate),
                DenseVector::as_slice(&state.hessian_product),
            );
            if php <= 0.0 {
                // Negative curvature: move to the trust-region boundary.
                let tau = boundary_step(
                    DenseVector::as_slice(&state.direction),
                    DenseVector::as_slice(&state.conjugate),
                    radius,
                );
                DenseVector::axpy(
                    &mut state.direction,
                    P::Scalar::from_f64(tau).expect("tau fits"),
                    &state.conjugate,
                );
                boundary = true;
                break;
            }
            let alpha = rr / php;

            // Stop at the boundary when the CG iterate leaves the region.
            let mut exceeds = false;
            let mut norm2 = 0.0;
            for (d, p) in DenseVector::as_slice(&state.direction)
                .iter()
                .zip(DenseVector::as_slice(&state.conjugate))
            {
                let value = d.to_f64().expect("scalar") + alpha * p.to_f64().expect("scalar");
                norm2 += value * value;
                if norm2 >= radius * radius {
                    exceeds = true;
                    break;
                }
            }
            if exceeds {
                let tau = boundary_step(
                    DenseVector::as_slice(&state.direction),
                    DenseVector::as_slice(&state.conjugate),
                    radius,
                );
                DenseVector::axpy(
                    &mut state.direction,
                    P::Scalar::from_f64(tau).expect("tau fits"),
                    &state.conjugate,
                );
                boundary = true;
                break;
            }

            // Accept the CG iterate: d += alpha p, r -= alpha H p.
            let alpha = P::Scalar::from_f64(alpha).expect("alpha fits");
            DenseVector::axpy(&mut state.direction, alpha, &state.conjugate);
            let neg_alpha = -alpha;
            DenseVector::axpy(&mut state.residual, neg_alpha, &state.hessian_product);

            let rr_new = dot(
                DenseVector::as_slice(&state.residual),
                DenseVector::as_slice(&state.residual),
            );
            if rr_new.sqrt() <= tolerance {
                break;
            }
            // p = r + beta p.
            let beta = P::Scalar::from_f64(rr_new / rr).expect("beta fits");
            let residual = DenseVector::as_slice(&state.residual);
            let conjugate = DenseVector::as_mut_slice(&mut state.conjugate);
            for (p, r) in conjugate.iter_mut().zip(residual) {
                *p = *r + beta * *p;
            }
            rr = rr_new;
        }

        // Predicted reduction of the quadratic model.
        h.mul_vec_into(
            DenseVector::as_slice(&state.direction),
            DenseVector::as_mut_slice(&mut state.hessian_direction),
        );
        let predicted = -dot(
            DenseVector::as_slice(grad),
            DenseVector::as_slice(&state.direction),
        ) - 0.5
            * dot(
                DenseVector::as_slice(&state.direction),
                DenseVector::as_slice(&state.hessian_direction),
            );

        state.boundary = boundary;
        Some((state.direction.clone(), predicted))
    }

    fn on_good_step(&mut self, quality: f64) {
        if let Some(state) = self.state.as_mut() {
            if quality < 0.25 {
                state.radius = (state.radius * self.options.tr.shrink_factor as f64)
                    .max(self.options.tr.radius_min as f64);
            } else if quality > 0.75 && state.boundary {
                state.radius = (state.radius * self.options.tr.expand_factor as f64)
                    .min(self.options.tr.radius_max as f64);
            }
            state.boundary = false;
        }
    }

    fn on_bad_step(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.radius = (state.radius * self.options.tr.shrink_factor as f64)
                .max(self.options.tr.radius_min as f64);
            state.boundary = false;
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
