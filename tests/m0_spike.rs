// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! M0 design-validation spike: backend abstraction, static and dynamic
//! dimensions, accumulation pattern, dense solvers, parameters, and
//! (optionally) num-dual autodiff.

use approx::assert_relative_eq;
use tinyopt_rs::backend::{
    DenseMatrix, DenseVector, Dim, LinearSolverMethod, Matrix, Vector, inverse, solve,
    zeros_matrix, zeros_vector,
};
use tinyopt_rs::{Dynamic, Params, Static};

type V3 = Vector<f64, Static<3>>;
type M3 = Matrix<f64, Static<3>>;
type VX = Vector<f64, Dynamic>;
type MX = Matrix<f64, Dynamic>;

fn mat<M: DenseMatrix<f64>>(rows: &[&[f64]]) -> M {
    let (r, c) = (rows.len(), rows[0].len());
    let mut m = <M as DenseMatrix<f64>>::zeros(r, c);
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            m.set_elem(i, j, *v);
        }
    }
    m
}

fn vec<V: DenseVector<f64>>(data: &[f64]) -> V {
    let mut v = <V as DenseVector<f64>>::zeros(data.len());
    for (i, x) in data.iter().enumerate() {
        v.set_at(i, *x);
    }
    v
}

fn mat_vec<M: DenseMatrix<f64>, V: DenseVector<f64>>(a: &M, x: &V) -> V {
    let n = x.len();
    let mut out = <V as DenseVector<f64>>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += a.elem(i, j) * x.at(j);
        }
        out.set_at(i, acc);
    }
    out
}

#[test]
fn dim_constants() {
    assert_eq!(Static::<4>::VALUE, Some(4));
    assert_eq!(Dynamic::VALUE, None);
}

#[test]
fn static_vector_ops() {
    let mut a = vec::<V3>(&[1.0, 2.0, 3.0]);
    let b = vec::<V3>(&[4.0, 5.0, 6.0]);
    assert_eq!(DenseVector::len(&a), 3);
    assert_relative_eq!(DenseVector::dot(&a, &b), 32.0);
    assert_relative_eq!(DenseVector::norm2(&a), 14.0);
    DenseVector::axpy(&mut a, 2.0, &b);
    assert_relative_eq!(a.at(0), 9.0);
    DenseVector::scale(&mut a, 0.5);
    assert_relative_eq!(a.at(2), 7.5);
    DenseVector::set_zero(&mut a);
    assert_eq!(DenseVector::as_slice(&a), &[0.0, 0.0, 0.0]);
}

#[test]
fn dynamic_vector_ops() {
    let a = vec::<VX>(&[1.0, 2.0, 3.0, 4.0]);
    let b = vec::<VX>(&[1.0, 0.0, -1.0, 2.0]);
    assert_eq!(DenseVector::len(&a), 4);
    assert_relative_eq!(DenseVector::dot(&a, &b), 1.0 + 0.0 - 3.0 + 8.0);
    assert_relative_eq!(DenseVector::norm(&a), (30.0f64).sqrt());
    let mut c = a.clone();
    DenseVector::copy_from(&mut c, &b);
    assert_eq!(DenseVector::as_slice(&c), DenseVector::as_slice(&b));
}

#[test]
fn outer_product_accumulation() {
    // H += J^T J for a single residual row J = [1, 2, 0].
    let mut h = zeros_matrix::<f64, Static<3>>(3, 3);
    let j = [1.0, 2.0, 0.0];
    DenseMatrix::add_outer_product_self(&mut h, 1.0, &j);
    assert_relative_eq!(h.elem(0, 0), 1.0);
    assert_relative_eq!(h.elem(0, 1), 2.0);
    assert_relative_eq!(h.elem(1, 1), 4.0);
    assert_relative_eq!(h.elem(2, 2), 0.0);

    // H += a * x * y^T
    let x = [1.0, 0.0, 0.0];
    let y = [0.0, 3.0, 0.0];
    DenseMatrix::add_outer_product(&mut h, 0.5, &x, &y);
    assert_relative_eq!(h.elem(0, 1), 2.0 + 1.5);
}

#[test]
fn matrix_access_roundtrip_is_asymmetric_safe() {
    // A 2x2 matrix with distinct entries catches padded-stride bugs (faer
    // pads its leading dimension, so slices cannot be assumed contiguous).
    type M2 = Matrix<f64, Static<2>>;
    let mut m: M2 = zeros_matrix::<f64, Static<2>>(2, 2);
    m.set_elem(0, 0, 1.0);
    m.set_elem(0, 1, 2.0);
    m.set_elem(1, 0, 3.0);
    m.set_elem(1, 1, 4.0);
    assert_relative_eq!(m.elem(0, 0), 1.0);
    assert_relative_eq!(m.elem(0, 1), 2.0);
    assert_relative_eq!(m.elem(1, 0), 3.0);
    assert_relative_eq!(m.elem(1, 1), 4.0);

    // A non-symmetric solve with LU: [[2, 1], [0, 3]] x = [5, 9] -> x = [1, 3].
    let a = mat::<M2>(&[&[2.0, 1.0], &[0.0, 3.0]]);
    let b: Vector<f64, Static<2>> = vec(&[5.0, 9.0]);
    let x = solve::<f64, Static<2>>(LinearSolverMethod::Lu, &a, &b).expect("solve");
    assert_relative_eq!(x.at(0), 1.0, epsilon = 1e-12);
    assert_relative_eq!(x.at(1), 3.0, epsilon = 1e-12);
}

#[test]
fn ldlt_solve_static() {
    type V2 = Vector<f64, Static<2>>;
    type M2 = Matrix<f64, Static<2>>;
    let a: M2 = mat(&[&[4.0, 1.0], &[1.0, 3.0]]);
    let b: V2 = vec(&[1.0, 2.0]);
    let x = solve::<f64, Static<2>>(LinearSolverMethod::Ldlt, &a, &b).expect("SPD solve");
    // x = [1/11, 7/11]
    assert_relative_eq!(x.at(0), 1.0 / 11.0, epsilon = 1e-12);
    assert_relative_eq!(x.at(1), 7.0 / 11.0, epsilon = 1e-12);
    assert_eq!(zeros_vector::<f64, Static<3>>(3).len(), 3);
}

#[test]
fn ldlt_solve_dynamic() {
    let a: MX = mat(&[&[4.0, 1.0, 0.0], &[1.0, 3.0, 1.0], &[0.0, 1.0, 2.0]]);
    let b: VX = vec(&[1.0, 2.0, 3.0]);
    let x = solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &a, &b).expect("SPD solve");
    let ax = mat_vec(&a, &x);
    for i in 0..3 {
        assert_relative_eq!(ax.at(i), b.at(i), epsilon = 1e-12);
    }
}

#[test]
fn all_dense_solver_methods() {
    let a: M3 = mat(&[&[4.0, 1.0, 0.0], &[1.0, 3.0, 1.0], &[0.0, 1.0, 2.0]]);
    let b: V3 = vec(&[1.0, 2.0, 3.0]);
    for method in [
        LinearSolverMethod::Ldlt,
        LinearSolverMethod::Llt,
        LinearSolverMethod::Lu,
        LinearSolverMethod::Qr,
        LinearSolverMethod::Svd,
    ] {
        let x =
            solve::<f64, Static<3>>(method, &a, &b).unwrap_or_else(|| panic!("{method:?} failed"));
        let ax = mat_vec(&a, &x);
        for i in 0..3 {
            assert_relative_eq!(ax.at(i), b.at(i), epsilon = 1e-9);
        }
    }
}

#[test]
fn inverse_roundtrip() {
    let a: MX = mat(&[&[4.0, 1.0], &[1.0, 3.0]]);
    let inv = inverse::<f64, Dynamic>(&a).expect("invertible");
    // A * A^-1 = I.
    for i in 0..2 {
        for j in 0..2 {
            let mut acc = 0.0;
            for k in 0..2 {
                acc += a.elem(i, k) * inv.elem(k, j);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(acc, expected, epsilon = 1e-12);
        }
    }
}

#[test]
fn params_static_and_dynamic() {
    let mut x = vec::<V3>(&[1.0, -2.0, 3.0]);
    assert_eq!(Params::dims(&x), 3);
    assert_relative_eq!(Params::norm2(&x), 14.0);
    Params::plus_eq(&mut x, &[0.5, 0.5, -0.5]);
    assert_eq!(DenseVector::as_slice(&x), &[1.5, -1.5, 2.5]);
    assert!(Params::fixed(&x).is_none());

    let mut y = vec::<VX>(&[1.0, 2.0]);
    assert_eq!(Params::dims(&y), 2);
    Params::plus_eq(&mut y, &[1.0, 1.0]);
    assert_eq!(DenseVector::as_slice(&y), &[2.0, 3.0]);

    let mut s = 2.0_f64;
    assert_eq!(Params::dims(&s), 1);
    Params::plus_eq(&mut s, &[0.5]);
    assert_relative_eq!(s, 2.5);
}

#[cfg(feature = "autodiff")]
mod autodiff_spike {
    use super::*;
    use tinyopt_rs::diff::{DualNum, ResidualsAndJacobian, ResidualsFn, residuals_and_jacobian};

    /// r(x) = [x0^2 + x1^2 - 2, x0 - x1]
    struct CircleLine;

    impl ResidualsFn for CircleLine {
        type Scalar = f64;

        fn eval<T: DualNum<Primitive = f64>>(&self, x: &[T]) -> Vec<T> {
            let (x0, x1) = (x[0].clone(), x[1].clone());
            vec![
                x0.clone() * x0.clone() + x1.clone() * x1.clone() - 2.0,
                x0 - x1,
            ]
        }
    }

    #[test]
    fn jacobian_is_exact() {
        let ResidualsAndJacobian {
            residuals,
            jacobian,
            num_residuals,
            num_params,
        } = residuals_and_jacobian(&CircleLine, &[1.0, 1.0]);
        assert_eq!((num_residuals, num_params), (2, 2));
        assert_relative_eq!(residuals[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(residuals[1], 0.0, epsilon = 1e-15);
        // d r0/dx = [2, 2], d r1/dx = [1, -1]
        assert_relative_eq!(jacobian[0], 2.0);
        assert_relative_eq!(jacobian[1], 2.0);
        assert_relative_eq!(jacobian[2], 1.0);
        assert_relative_eq!(jacobian[3], -1.0);
    }

    #[test]
    fn autodiff_plus_backend_gauss_newton_step() {
        // Assemble grad = J^T r and H = J^T J from autodiff, then take a
        // Gauss-Newton step and check that the parameters converge.
        let mut x = [1.4, 0.6];

        for _ in 0..10 {
            let out = residuals_and_jacobian(&CircleLine, &x);
            let n = out.num_params;
            let m = out.num_residuals;

            let mut grad = zeros_vector::<f64, Dynamic>(n);
            let mut h = zeros_matrix::<f64, Dynamic>(n, n);
            for row in 0..m {
                let j = &out.jacobian[row * n..(row + 1) * n];
                let r = out.residuals[row];
                for (i, ji) in j.iter().enumerate() {
                    grad.set_at(i, grad.at(i) + ji * r);
                }
                DenseMatrix::add_outer_product_self(&mut h, 1.0, j);
            }

            let mut neg_grad = zeros_vector::<f64, Dynamic>(n);
            for i in 0..n {
                neg_grad.set_at(i, -grad.at(i));
            }
            let dx =
                solve::<f64, Dynamic>(LinearSolverMethod::Ldlt, &h, &neg_grad).expect("GN system");
            for (i, xi) in x.iter_mut().enumerate() {
                *xi += dx.at(i);
            }
        }

        let out = residuals_and_jacobian(&CircleLine, &x);
        let norm: f64 = out.residuals.iter().map(|r| r * r).sum();
        assert!(norm < 1e-18, "final residual norm^2 = {norm}");
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-8);
    }
}
