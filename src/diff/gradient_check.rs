// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Gradient and Hessian verification, mirroring tinyopt's
//! `diff::CheckGradient` and `diff::CheckResidualsGradient`.

use crate::backend::{DenseMatrix, DenseVector, Dim, Matrix, zeros_matrix, zeros_vector};
use crate::diff::num_diff::{Method, estimate_num_jac, num_eval};
use crate::hessian::HessianStorage;
use crate::optimizers::AccumulationFn;
use crate::real::Real;

/// Default comparison tolerance: `1e-2` in single precision, `1e-5` in double.
pub fn default_eps<S: Real>() -> f64 {
    if core::mem::size_of::<S>() == 4 {
        1e-2
    } else {
        1e-5
    }
}

/// Compare the analytical gradient of an accumulation function against a
/// numerical estimate of the cost gradient.
///
/// Returns `true` when the maximum absolute difference is below `eps`.
pub fn check_gradient<S, D, F>(x: &[S], f: &F, eps: f64, method: Method, verbose: bool) -> bool
where
    S: Real,
    D: Dim,
    Matrix<S, D>: HessianStorage<S>,
    F: AccumulationFn<S, D> + ?Sized,
{
    let n = x.len();
    let mut grad = zeros_vector::<S, D>(n);
    f.eval(x, Some(&mut grad), None);

    let cost_fn = |xx: &[S]| {
        let cost = f.eval(xx, None, None).cost;
        vec![S::from_f64(cost).expect("cost fits in scalar")]
    };
    let step = S::from_f64(eps / 10.0).expect("step fits in scalar");
    let jac = estimate_num_jac(x, &cost_fn, method, step);

    let mut max_dist = 0.0_f64;
    for (i, j) in jac.iter().take(n).enumerate() {
        let d = (j.to_f64().unwrap_or(0.0) - grad.at(i).to_f64().unwrap_or(0.0)).abs();
        max_dist = max_dist.max(d);
    }

    let ok = max_dist < eps;
    if !ok && verbose {
        log::warn!("wrong gradient: max distance {max_dist:.3e} >= eps {eps:.3e}");
    }
    ok
}

/// Compare an accumulation function's gradient and Hessian against numerical
/// derivatives of the residuals (`Jᵀr` and `JᵀJ`).
#[allow(clippy::too_many_arguments)]
pub fn check_residuals_gradient<S, D, R, A>(
    x: &[S],
    residuals: &R,
    acc: &A,
    eps: f64,
    method: Method,
    verbose: bool,
    check_h: bool,
) -> bool
where
    S: Real,
    D: Dim,
    Matrix<S, D>: HessianStorage<S>,
    R: Fn(&[S]) -> Vec<S> + ?Sized,
    A: AccumulationFn<S, D> + ?Sized,
{
    let n = x.len();
    let step = S::from_f64(eps / 10.0).expect("step fits in scalar");
    let out = num_eval(x, residuals, method, step);
    let m = out.num_residuals;

    // Numerical gradient and Gauss-Newton Hessian.
    let mut grad_num = zeros_vector::<S, D>(n);
    for (i, g) in grad_num.as_mut_slice().iter_mut().enumerate() {
        let mut acc_i = S::zero();
        for k in 0..m {
            acc_i += out.jacobian[k * n + i] * out.residuals[k];
        }
        *g = acc_i;
    }
    let mut h_num = zeros_matrix::<S, D>(n, n);
    for k in 0..m {
        DenseMatrix::add_outer_product_self(
            &mut h_num,
            S::one(),
            &out.jacobian[k * n..(k + 1) * n],
        );
    }

    // Analytical gradient and Hessian.
    let mut grad = zeros_vector::<S, D>(n);
    let mut h = check_h.then(|| zeros_matrix::<S, D>(n, n));
    acc.eval(x, Some(&mut grad), h.as_mut());

    let mut grad_dist = 0.0_f64;
    for i in 0..n {
        let d = (grad_num.at(i).to_f64().unwrap_or(0.0) - grad.at(i).to_f64().unwrap_or(0.0)).abs();
        grad_dist = grad_dist.max(d);
    }
    let grad_ok = grad_dist < eps;
    if !grad_ok && verbose {
        log::warn!("wrong residuals gradient: max distance {grad_dist:.3e} >= eps {eps:.3e}");
    }

    let Some(h) = h.as_ref() else {
        return grad_ok;
    };
    let mut h_dist = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            let d = (h_num.elem(i, j).to_f64().unwrap_or(0.0)
                - h.elem(i, j).to_f64().unwrap_or(0.0))
            .abs();
            h_dist = h_dist.max(d);
        }
    }
    let h_ok = h_dist < eps;
    if !h_ok && verbose {
        log::warn!("wrong Hessian approximation: max distance {h_dist:.3e} >= eps {eps:.3e}");
    }
    grad_ok && h_ok
}
