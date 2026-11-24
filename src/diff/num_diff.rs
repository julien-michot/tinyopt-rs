// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Numerical differentiation (finite differences).
//!
//! Mirrors tinyopt's `diff::NumEval`, `EstimateNumJac` and
//! `CreateNumDiffFunc1/2`. All functions operate on flat parameter and residual
//! slices so that they work with any parameter type.

use crate::backend::{DenseMatrix, DenseVector, Dim, Matrix, Vector, zeros_matrix};
use crate::cost::Cost;
use crate::hessian::HessianStorage;
use crate::optimizers::AccumulationFn;
use crate::real::Real;

/// Numerical differentiation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Method {
    /// Forward difference `(f(x+h) - f(x)) / h` (first order).
    Forward,
    /// Central difference `(f(x+h) - f(x-h)) / (2h)` (second order, default).
    #[default]
    Central,
    /// Central difference reusing the `x+h` evaluation (fast, manifold-friendly).
    FastCentral,
}

/// Residuals and row-major Jacobian estimated by finite differences.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NumEval<S> {
    /// Residual values at `x`, length `num_residuals`.
    pub residuals: Vec<S>,
    /// Row-major Jacobian, `num_residuals × num_params`.
    pub jacobian: Vec<S>,
    /// Number of residuals.
    pub num_residuals: usize,
    /// Number of parameters.
    pub num_params: usize,
}

/// Evaluate residuals and estimate the Jacobian at `x`.
pub fn num_eval<S, F>(x: &[S], f: &F, method: Method, h: S) -> NumEval<S>
where
    S: Real,
    F: Fn(&[S]) -> Vec<S> + ?Sized,
{
    let n = x.len();
    let residuals = f(x);
    let m = residuals.len();

    let mut jacobian = vec![S::zero(); m * n];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();

    for r in 0..n {
        xp[r] = x[r] + h;
        let res_plus = f(&xp);

        let (res_minus, scale) = match method {
            Method::Forward => (None, S::one() / h),
            Method::Central | Method::FastCentral => {
                xm[r] = x[r] - h;
                (Some(f(&xm)), S::one() / (S::from_f64(2.0).expect("2") * h))
            }
        };

        for k in 0..m {
            let numerator = match &res_minus {
                Some(rm) => res_plus[k] - rm[k],
                None => res_plus[k] - residuals[k],
            };
            jacobian[k * n + r] = numerator * scale;
        }

        xp[r] = x[r];
        xm[r] = x[r];
    }

    NumEval {
        residuals,
        jacobian,
        num_residuals: m,
        num_params: n,
    }
}

/// Estimate the Jacobian at `x` (row-major).
pub fn estimate_num_jac<S, F>(x: &[S], f: &F, method: Method, h: S) -> Vec<S>
where
    S: Real,
    F: Fn(&[S]) -> Vec<S> + ?Sized,
{
    num_eval(x, f, method, h).jacobian
}

/// Accumulation function backed by numerical differentiation of residuals.
///
/// Equivalent to C++ `CreateNumDiffFunc1` / `CreateNumDiffFunc2`: the gradient
/// is `Jᵀr` and the Hessian approximation is `JᵀJ`.
#[derive(Clone, Debug)]
pub struct NumDiffAccumulator<'a, F: ?Sized, S> {
    f: &'a F,
    method: Method,
    h: S,
}

impl<'a, F: ?Sized, S: Real> NumDiffAccumulator<'a, F, S> {
    /// Create an accumulator from a residuals function.
    pub fn new(f: &'a F, method: Method, h: S) -> Self {
        Self { f, method, h }
    }
}

impl<S, F, D> AccumulationFn<S, D> for NumDiffAccumulator<'_, F, S>
where
    S: Real,
    D: Dim,
    Matrix<S, D>: HessianStorage<S>,
    F: Fn(&[S]) -> Vec<S> + ?Sized,
{
    fn eval(&self, x: &[S], grad: Option<&mut Vector<S, D>>, h: Option<&mut Matrix<S, D>>) -> Cost {
        let out = num_eval(x, self.f, self.method, self.h);
        let (m, n) = (out.num_residuals, out.num_params);

        let mut cost = S::zero();
        for r in &out.residuals {
            cost += *r * *r;
        }
        let cost = Cost::with_num_residuals(cost.to_f64().expect("scalar fits in f64"), m as i32);

        if let Some(g) = grad {
            DenseVector::set_zero(g);
            for i in 0..n {
                let mut acc = S::zero();
                for k in 0..m {
                    acc += out.jacobian[k * n + i] * out.residuals[k];
                }
                g.set_at(i, acc);
            }
        }
        if let Some(hess) = h {
            DenseMatrix::set_zero(hess);
            for k in 0..m {
                DenseMatrix::add_outer_product_self(
                    hess,
                    S::one(),
                    &out.jacobian[k * n..(k + 1) * n],
                );
            }
        }
        cost
    }

    fn uses_numerical_diff(&self) -> bool {
        true
    }
}

/// Accumulation function backed by numerical differentiation of a scalar cost.
#[derive(Clone, Debug)]
pub struct NumDiffCostAccumulator<'a, F: ?Sized, S> {
    f: &'a F,
    method: Method,
    h: S,
}

impl<'a, F: ?Sized, S: Real> NumDiffCostAccumulator<'a, F, S> {
    /// Create an accumulator from a scalar cost function.
    pub fn new(f: &'a F, method: Method, h: S) -> Self {
        Self { f, method, h }
    }
}

impl<S, F, D> AccumulationFn<S, D> for NumDiffCostAccumulator<'_, F, S>
where
    S: Real,
    D: Dim,
    Matrix<S, D>: HessianStorage<S>,
    F: Fn(&[S]) -> S + ?Sized,
{
    fn eval(&self, x: &[S], grad: Option<&mut Vector<S, D>>, h: Option<&mut Matrix<S, D>>) -> Cost {
        let f_res = |x: &[S]| vec![(self.f)(x)];
        let out = num_eval(x, &f_res, self.method, self.h);
        let cost = Cost::new(out.residuals[0].to_f64().expect("scalar fits in f64"));
        if let Some(g) = grad {
            DenseVector::set_zero(g);
            for (i, &derivative) in out.jacobian.iter().take(out.num_params).enumerate() {
                g.set_at(i, derivative);
            }
        }
        if let Some(hess) = h {
            DenseMatrix::set_zero(hess);
        }
        cost
    }

    fn uses_numerical_diff(&self) -> bool {
        true
    }
}

/// Create a numerical-differentiation accumulator from a residual function.
pub fn create_num_diff_func<S, F>(f: &F, method: Method, h: S) -> NumDiffAccumulator<'_, F, S>
where
    S: Real,
    F: Fn(&[S]) -> Vec<S> + ?Sized,
{
    NumDiffAccumulator::new(f, method, h)
}

/// Create a numerical-differentiation accumulator from a scalar cost function.
pub fn create_num_diff_cost_func<S, F>(
    f: &F,
    method: Method,
    h: S,
) -> NumDiffCostAccumulator<'_, F, S>
where
    S: Real,
    F: Fn(&[S]) -> S + ?Sized,
{
    NumDiffCostAccumulator::new(f, method, h)
}

/// Convenience: a zeros matrix helper used by tests.
#[allow(dead_code)]
pub(crate) fn zeros<S: Real, D: Dim>(n: usize) -> Matrix<S, D> {
    zeros_matrix::<S, D>(n, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Dynamic, Static};

    fn residuals(x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] - 2.0, x[0] * x[1] - 3.0]
    }

    #[test]
    fn central_matches_analytic() {
        let x = [1.3, 2.1];
        let out = num_eval(&x, &residuals, Method::Central, 1e-7);
        // J = [[2*x0, 0], [x1, x0]]
        assert!((out.jacobian[0] - 2.0 * x[0]).abs() < 1e-6);
        assert!(out.jacobian[1].abs() < 1e-6);
        assert!((out.jacobian[2] - x[1]).abs() < 1e-6);
        assert!((out.jacobian[3] - x[0]).abs() < 1e-6);
    }

    #[test]
    fn forward_is_first_order() {
        let x = [1.3, 2.1];
        let out = num_eval(&x, &residuals, Method::Forward, 1e-7);
        assert!((out.jacobian[0] - 2.0 * x[0]).abs() < 1e-4);
    }

    #[test]
    fn accumulator_builds_grad_and_hessian() {
        let x = [1.3, 2.1];
        let acc = create_num_diff_func(&residuals, Method::Central, 1e-7);
        let mut g = <Vector<f64, Static<2>> as DenseVector<f64>>::zeros(2);
        let mut h = <Matrix<f64, Static<2>> as DenseMatrix<f64>>::zeros(2, 2);
        let cost = AccumulationFn::<f64, Static<2>>::eval(&acc, &x, Some(&mut g), Some(&mut h));

        let r0 = x[0] * x[0] - 2.0;
        let r1 = x[0] * x[1] - 3.0;
        let (j00, j10, j11) = (2.0 * x[0], x[1], x[0]);
        assert!((cost.cost - (r0 * r0 + r1 * r1)).abs() < 1e-12);
        assert!((g.at(0) - (j00 * r0 + j10 * r1)).abs() < 1e-5);
        assert!((g.at(1) - j11 * r1).abs() < 1e-5);
        assert!((h.elem(0, 0) - (j00 * j00 + j10 * j10)).abs() < 1e-5);
        assert!((h.elem(1, 1) - j11 * j11).abs() < 1e-5);
        assert!((h.elem(0, 1) - j10 * j11).abs() < 1e-5);
    }

    #[test]
    fn dynamic_dims_work() {
        let x = [1.3, 2.1];
        let acc = create_num_diff_func(&residuals, Method::FastCentral, 1e-7);
        let mut g = <Vector<f64, Dynamic> as DenseVector<f64>>::zeros(2);
        let cost = AccumulationFn::<f64, Dynamic>::eval(&acc, &x, Some(&mut g), None);
        assert!(cost.is_valid());
        assert!(g.at(0).is_finite());
    }
}
