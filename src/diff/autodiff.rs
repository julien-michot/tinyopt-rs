// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Forward-mode automatic differentiation with num-dual.
//!
//! User functions are written once, generically over the scalar, and invoked
//! with either the plain scalar (evaluating values only) or `DualVec` (values
//! and derivatives in a single pass):
//!
//! ```
//! use tinyopt_rs::diff::{DualNum, ResidualsFn, residuals_and_jacobian};
//!
//! struct Sqrt2;
//!
//! impl ResidualsFn for Sqrt2 {
//!     type Scalar = f64;
//!
//!     fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
//!         let x0 = x[0].clone();
//!         vec![x0.clone() * x0 - 2.0]
//!     }
//! }
//!
//! let (r, jac, m, n) = {
//!     let out = residuals_and_jacobian(&Sqrt2, &[2.0]);
//!     (out.residuals, out.jacobian, out.num_residuals, out.num_params)
//! };
//! assert_eq!((m, n), (1, 1));
//! assert!((r[0] - 2.0).abs() < 1e-12);
//! assert!((jac[0] - 4.0).abs() < 1e-12);
//! ```

use nalgebra::{Dyn, U1};
use num_dual::{Derivative, DualNum, DualVec};

use crate::real::Real;

/// Scalar cost function, generic over the arithmetic scalar.
///
/// Implemented by user structs; the same `eval` is used with `f64` (plain
/// evaluation) and `DualVec` (automatic differentiation).
pub trait CostFn {
    /// Base scalar type.
    type Scalar: Real;

    /// Evaluate the scalar cost at `x`.
    fn eval<T: DualNum<Primitive = Self::Scalar>>(&self, x: &[T]) -> T;
}

/// Residual-vector function, generic over the arithmetic scalar.
pub trait ResidualsFn {
    /// Base scalar type.
    type Scalar: Real;

    /// Evaluate the residual vector at `x`.
    fn eval<T: DualNum<Primitive = Self::Scalar>>(&self, x: &[T]) -> Vec<T>;
}

/// Residuals and row-major Jacobian produced by autodiff evaluation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResidualsAndJacobian {
    /// Residual values, length `num_residuals`.
    pub residuals: Vec<f64>,
    /// Row-major Jacobian, `num_residuals × num_params`.
    pub jacobian: Vec<f64>,
    /// Number of residuals.
    pub num_residuals: usize,
    /// Number of parameters.
    pub num_params: usize,
}

/// Cost value and gradient produced by autodiff evaluation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostAndGradient {
    /// Cost value.
    pub cost: f64,
    /// Gradient, length `num_params`.
    pub gradient: Vec<f64>,
    /// Number of parameters.
    pub num_params: usize,
}

/// Seed one dual number per parameter, each carrying the corresponding unit
/// derivative so that a single function evaluation yields all partials.
fn seeded_jets(x: &[f64]) -> Vec<DualVec<f64, Dyn>> {
    let n = x.len();
    (0..n)
        .map(|i| DualVec::new(x[i], Derivative::derivative_generic(Dyn(n), U1, i)))
        .collect()
}

/// Evaluate residuals and the full Jacobian at `x` using forward-mode autodiff.
pub fn residuals_and_jacobian<F>(f: &F, x: &[f64]) -> ResidualsAndJacobian
where
    F: ResidualsFn<Scalar = f64>,
{
    let jets = seeded_jets(x);
    let out = f.eval(&jets);
    let (m, n) = (out.len(), x.len());

    let mut residuals = Vec::with_capacity(m);
    let mut jacobian = vec![0.0; m * n];
    for (row, r) in out.into_iter().enumerate() {
        residuals.push(r.re);
        let eps = r
            .eps
            .0
            .as_ref()
            .expect("dual number always carries its derivative");
        jacobian[row * n..(row + 1) * n].copy_from_slice(eps.as_slice());
    }

    ResidualsAndJacobian {
        residuals,
        jacobian,
        num_residuals: m,
        num_params: n,
    }
}

/// Evaluate a scalar cost and its gradient at `x` using forward-mode autodiff.
pub fn cost_and_gradient<F>(f: &F, x: &[f64]) -> CostAndGradient
where
    F: CostFn<Scalar = f64>,
{
    let jets = seeded_jets(x);
    let out = f.eval(&jets);
    let n = x.len();

    let mut gradient = Vec::with_capacity(n);
    let eps = out
        .eps
        .0
        .as_ref()
        .expect("dual number always carries its derivative");
    gradient.extend_from_slice(eps.as_slice());

    CostAndGradient {
        cost: out.re,
        gradient,
        num_params: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sqrt2;
    impl ResidualsFn for Sqrt2 {
        type Scalar = f64;
        fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
            let x0 = x[0].clone();
            vec![x0.clone() * x0 - 2.0]
        }
    }

    struct Quadratic;
    impl CostFn for Quadratic {
        type Scalar = f64;
        fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> T {
            let dx = x[0].clone() - 3.0;
            let dy = x[1].clone() + 1.0;
            dx.clone() * dx + dy.clone() * dy
        }
    }

    #[test]
    fn residuals_and_full_jacobian() {
        struct Cross;
        impl ResidualsFn for Cross {
            type Scalar = f64;
            fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
                let (x0, x1) = (x[0].clone(), x[1].clone());
                vec![x0.clone() * x1, x0.clone() * x0 + 1.0]
            }
        }

        let out = residuals_and_jacobian(&Cross, &[2.0, 5.0]);
        assert_eq!(out.num_residuals, 2);
        assert_eq!(out.num_params, 2);
        assert_eq!(out.residuals, vec![10.0, 5.0]);
        // r0 = x0*x1 -> [x1, x0]; r1 = x0^2 + 1 -> [2*x0, 0]
        assert_eq!(out.jacobian, vec![5.0, 2.0, 4.0, 0.0]);
    }

    #[test]
    fn cost_gradient() {
        let out = cost_and_gradient(&Quadratic, &[3.0, -1.0]);
        assert_eq!(out.cost, 0.0);
        assert_eq!(out.gradient, vec![0.0, 0.0]);

        let out = cost_and_gradient(&Quadratic, &[4.0, 1.0]);
        assert_eq!(out.cost, 1.0 + 4.0);
        assert_eq!(out.gradient, vec![2.0, 4.0]);
    }

    #[test]
    fn sqrt2_residual() {
        let out = residuals_and_jacobian(&Sqrt2, &[2.0]);
        assert!((out.residuals[0] - 2.0).abs() < 1e-15);
        assert!((out.jacobian[0] - 4.0).abs() < 1e-15);
    }
}
