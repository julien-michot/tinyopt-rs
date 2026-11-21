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
//! let out = residuals_and_jacobian(&Sqrt2, &[2.0]);
//! assert!((out.residuals[0] - 2.0).abs() < 1e-12);
//! assert!((out.jacobian[0] - 4.0).abs() < 1e-12);
//! ```

use nalgebra::{Const, Dyn, U1};
use num_dual::{Derivative, DualNum, DualVec};

use crate::backend::{DenseMatrix, DenseVector, Dim, Dynamic, Matrix, Static, Vector};
use crate::cost::Cost;
use crate::optimizers::AccumulationFn;
use crate::real::Real;

/// Scalar cost function, generic over the arithmetic scalar.
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
    F: ResidualsFn<Scalar = f64> + ?Sized,
{
    let jets = seeded_jets(x);
    let out = f.eval(&jets);
    let (m, n) = (out.len(), x.len());

    let mut residuals = Vec::with_capacity(m);
    let mut jacobian = vec![0.0; m * n];
    for (row, r) in out.into_iter().enumerate() {
        residuals.push(r.re);
        if let Some(eps) = r.eps.0.as_ref() {
            jacobian[row * n..(row + 1) * n].copy_from_slice(eps.as_slice());
        }
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
    F: CostFn<Scalar = f64> + ?Sized,
{
    let jets = seeded_jets(x);
    let out = f.eval(&jets);
    let n = x.len();

    let mut gradient = vec![0.0; n];
    if let Some(eps) = out.eps.0.as_ref() {
        gradient.copy_from_slice(eps.as_slice());
    }

    CostAndGradient {
        cost: out.re,
        gradient,
        num_params: n,
    }
}

/// Mirrors C++ `diff::Eval`: residuals and Jacobian of a residual function.
pub fn eval<F>(f: &F, x: &[f64]) -> ResidualsAndJacobian
where
    F: ResidualsFn<Scalar = f64> + ?Sized,
{
    residuals_and_jacobian(f, x)
}

/// Mirrors C++ `diff::CalculateJac`: estimated Jacobian of a residual function.
pub fn calculate_jac<F>(f: &F, x: &[f64]) -> Vec<f64>
where
    F: ResidualsFn<Scalar = f64> + ?Sized,
{
    residuals_and_jacobian(f, x).jacobian
}

/// Maps a [`Dim`] to the corresponding num-dual dimension and storage.
///
/// Static dimensions use stack-allocated `DualVec<T, Const<N>>` values, so the
/// fixed-size autodiff path performs no heap allocation.
pub trait AdDim: Dim {
    /// Dual-number type used to evaluate the residuals.
    type Dual: DualNum<Primitive = f64> + Clone;
    /// Container holding one dual number per parameter.
    type Jets: AsRef<[Self::Dual]>;

    /// Seed the dual numbers at `x`.
    fn seed(x: &[f64]) -> Self::Jets;
    /// Value part of a dual number.
    fn value(d: &Self::Dual) -> f64;
    /// Derivative part of a dual number (`None` for constants).
    fn derivative(d: &Self::Dual) -> Option<&[f64]>;
}

impl AdDim for Dynamic {
    type Dual = DualVec<f64, Dyn>;
    type Jets = Vec<Self::Dual>;

    fn seed(x: &[f64]) -> Vec<DualVec<f64, Dyn>> {
        let n = x.len();
        (0..n)
            .map(|i| DualVec::new(x[i], Derivative::derivative_generic(Dyn(n), U1, i)))
            .collect()
    }

    fn value(d: &Self::Dual) -> f64 {
        d.re
    }

    fn derivative(d: &Self::Dual) -> Option<&[f64]> {
        d.eps.0.as_ref().map(|eps| eps.as_slice())
    }
}

impl<const N: usize> AdDim for Static<N> {
    type Dual = DualVec<f64, Const<N>>;
    type Jets = [Self::Dual; N];

    fn seed(x: &[f64]) -> [DualVec<f64, Const<N>>; N] {
        assert_eq!(x.len(), N, "static autodiff dimension mismatch");
        core::array::from_fn(|i| {
            DualVec::new(x[i], Derivative::derivative_generic(Const::<N>, U1, i))
        })
    }

    fn value(d: &Self::Dual) -> f64 {
        d.re
    }

    fn derivative(d: &Self::Dual) -> Option<&[f64]> {
        d.eps.0.as_ref().map(|eps| eps.as_slice())
    }
}

/// Accumulation function backed by forward-mode autodiff of residuals.
///
/// This is the Rust equivalent of passing a residuals lambda to the C++
/// `Optimize` entry point: the Jacobian is computed with dual numbers and
/// accumulated directly into `Jᵀr` and `JᵀJ`, without materializing it.
#[derive(Clone, Debug)]
pub struct AdAccumulator<'a, F: ?Sized> {
    f: &'a F,
}

/// Wrap a residuals function into an accumulation function.
pub fn ad_accumulation<F>(f: &F) -> AdAccumulator<'_, F>
where
    F: ResidualsFn<Scalar = f64> + ?Sized,
{
    AdAccumulator { f }
}

impl<F, D> AccumulationFn<f64, D> for AdAccumulator<'_, F>
where
    F: ResidualsFn<Scalar = f64> + ?Sized,
    D: AdDim,
{
    fn eval(
        &self,
        x: &[f64],
        grad: Option<&mut Vector<f64, D>>,
        h: Option<&mut Matrix<f64, D>>,
    ) -> Cost {
        let jets = D::seed(x);
        let out = self.f.eval(jets.as_ref());
        let m = out.len();

        let mut grad = grad;
        let mut h = h;
        if let Some(g) = grad.as_mut() {
            DenseVector::set_zero(*g);
        }
        if let Some(hm) = h.as_mut() {
            DenseMatrix::set_zero(*hm);
        }

        let mut cost = 0.0;
        for r in &out {
            let value = D::value(r);
            cost += value * value;
            if let Some(jac) = D::derivative(r) {
                if let Some(g) = grad.as_mut() {
                    for (i, d) in jac.iter().enumerate() {
                        g.set_at(i, g.at(i) + d * value);
                    }
                }
                if let Some(hm) = h.as_mut() {
                    DenseMatrix::add_outer_product_self(*hm, 1.0, jac);
                }
            }
        }

        Cost::with_num_residuals(cost, m as i32)
    }
}

/// A residual block with an optional robust loss.
#[derive(Clone, Debug)]
pub struct ResidualBlock<T> {
    /// Residual values of the block.
    pub residual: Vec<T>,
    /// Robust re-weighting applied to the block.
    pub robust: RobustLoss,
}

/// Robust loss tag attached to a residual block.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum RobustLoss {
    /// No robust weighting.
    #[default]
    None,
    /// Huber loss with threshold `delta`.
    Huber(f64),
    /// Cauchy loss with threshold `delta`.
    Cauchy(f64),
    /// Tukey loss with threshold `delta`.
    Tukey(f64),
}

impl<T> ResidualBlock<T> {
    /// A plain (non-robust) residual block.
    pub fn plain(residual: Vec<T>) -> Self {
        Self {
            residual,
            robust: RobustLoss::None,
        }
    }

    /// A Huber-weighted residual block.
    pub fn huber(residual: Vec<T>, delta: f64) -> Self {
        Self {
            residual,
            robust: RobustLoss::Huber(delta),
        }
    }

    /// A Cauchy-weighted residual block.
    pub fn cauchy(residual: Vec<T>, delta: f64) -> Self {
        Self {
            residual,
            robust: RobustLoss::Cauchy(delta),
        }
    }

    /// A Tukey-weighted residual block.
    pub fn tukey(residual: Vec<T>, delta: f64) -> Self {
        Self {
            residual,
            robust: RobustLoss::Tukey(delta),
        }
    }
}

/// Residual function returning robust-tagged blocks, mirroring the C++
/// `Residuals(losses::Huber(res, delta), prior)` tuple API.
pub trait RobustResidualsFn {
    /// Base scalar type.
    type Scalar: Real;

    /// Evaluate the residual blocks at `x`.
    fn eval<T: DualNum<Primitive = Self::Scalar>>(&self, x: &[T]) -> Vec<ResidualBlock<T>>;
}

/// Accumulation function backed by autodiff of robust residual blocks.
#[derive(Clone, Debug)]
pub struct RobustAdAccumulator<'a, F: ?Sized> {
    f: &'a F,
}

/// Wrap a block residuals function into an accumulation function.
pub fn robust_accumulation<F>(f: &F) -> RobustAdAccumulator<'_, F>
where
    F: RobustResidualsFn<Scalar = f64> + ?Sized,
{
    RobustAdAccumulator { f }
}

impl<F, D> AccumulationFn<f64, D> for RobustAdAccumulator<'_, F>
where
    F: RobustResidualsFn<Scalar = f64> + ?Sized,
    D: AdDim,
{
    fn eval(
        &self,
        x: &[f64],
        grad: Option<&mut Vector<f64, D>>,
        h: Option<&mut Matrix<f64, D>>,
    ) -> Cost {
        let jets = D::seed(x);
        let blocks = self.f.eval(jets.as_ref());

        let mut grad = grad;
        let mut h = h;
        if let Some(g) = grad.as_mut() {
            DenseVector::set_zero(*g);
        }
        if let Some(hm) = h.as_mut() {
            DenseMatrix::set_zero(*hm);
        }

        let mut cost = 0.0;
        let mut num_residuals = 0;
        let mut num_inliers = 0;

        for block in &blocks {
            let m = block.residual.len();
            num_residuals += m;

            // Moment statistics from the value part (like C++: `raw[i].a`).
            let mut sq_norm = 0.0;
            for r in &block.residual {
                let v = D::value(r);
                sq_norm += v * v;
            }

            let (weight, block_cost) = match block.robust {
                RobustLoss::None => (1.0, sq_norm),
                RobustLoss::Huber(delta) | RobustLoss::Cauchy(delta) | RobustLoss::Tukey(delta) => {
                    // Mirrors the C++ block adapter, which applies the Huber
                    // weight and value to every robust block.
                    let delta2 = delta * delta;
                    if sq_norm <= delta2 {
                        num_inliers += m;
                        (1.0, sq_norm)
                    } else {
                        let n = sq_norm.max(f64::MIN_POSITIVE).sqrt();
                        (delta / n, 2.0 * delta * n - delta2)
                    }
                }
            };
            cost += block_cost;

            if weight == 0.0 {
                continue;
            }
            let w = weight;

            if let Some(g) = grad.as_mut() {
                for r in &block.residual {
                    let value = D::value(r);
                    if let Some(jac) = D::derivative(r) {
                        for (i, d) in jac.iter().enumerate() {
                            g.set_at(i, g.at(i) + w * d * value);
                        }
                    }
                }
            }
            if let Some(hm) = h.as_mut() {
                for r in &block.residual {
                    if let Some(jac) = D::derivative(r) {
                        DenseMatrix::add_outer_product_self(*hm, w, jac);
                    }
                }
            }
        }

        let inlier_ratio = if num_residuals > 0 {
            num_inliers as f32 / num_residuals as f32
        } else {
            1.0
        };
        Cost::with_inliers(cost, num_residuals as i32, inlier_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{zeros_matrix, zeros_vector};

    fn test_vector<D: Dim>(n: usize) -> Vector<f64, D> {
        zeros_vector::<f64, D>(n)
    }

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
        let jac = calculate_jac(&Sqrt2, &[2.0]);
        assert!((jac[0] - 4.0).abs() < 1e-15);
    }

    #[test]
    fn static_and_dynamic_accumulation_agree() {
        let x = [1.3];

        let acc_static = ad_accumulation(&Sqrt2);
        let mut g_static = test_vector::<Static<1>>(1);
        let mut h_static = zeros_matrix::<f64, Static<1>>(1, 1);
        let c_static = AccumulationFn::<f64, Static<1>>::eval(
            &acc_static,
            &x,
            Some(&mut g_static),
            Some(&mut h_static),
        );

        let acc_dyn = ad_accumulation(&Sqrt2);
        let mut g_dyn = test_vector::<Dynamic>(1);
        let mut h_dyn = zeros_matrix::<f64, Dynamic>(1, 1);
        let c_dyn =
            AccumulationFn::<f64, Dynamic>::eval(&acc_dyn, &x, Some(&mut g_dyn), Some(&mut h_dyn));

        assert_eq!(c_static.cost, c_dyn.cost);
        assert_eq!(g_static.at(0), g_dyn.at(0));
        assert_eq!(h_static.elem(0, 0), h_dyn.elem(0, 0));
        // r = 1.3² - 2 = -0.31, J = 2.6
        assert!((g_static.at(0) - 2.6 * -0.31).abs() < 1e-12);
        assert!((h_static.elem(0, 0) - 2.6 * 2.6).abs() < 1e-12);
    }

    #[test]
    fn robust_block_weights_outliers() {
        // Residual 10 with delta 1: Huber weight = 1/10 and cost = 2*1*10 - 1.
        struct Outlier;
        impl RobustResidualsFn for Outlier {
            type Scalar = f64;
            fn eval<T: DualNum<Primitive = f64>>(&self, _x: &[T]) -> Vec<ResidualBlock<T>> {
                vec![ResidualBlock::huber(vec![T::from(10.0)], 1.0)]
            }
        }
        let acc = robust_accumulation(&Outlier);
        let mut g = test_vector::<Static<1>>(1);
        let c = AccumulationFn::<f64, Static<1>>::eval(&acc, &[0.0], Some(&mut g), None);
        assert!((c.cost - (2.0 * 10.0 - 1.0)).abs() < 1e-12);
        // The residual is constant, so its derivative is zero.
        assert_eq!(g.at(0), 0.0);
        assert_eq!(c.num_inliers(), 0);
    }
}
