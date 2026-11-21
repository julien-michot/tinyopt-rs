// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Activation functions, mirroring `tinyopt::losses` activations.

use crate::math::exp;
use crate::real::Real;

/// Sigmoid `1 / (1 + e⁻ˣ)`.
pub fn sigmoid<S: Real>(x: S) -> S {
    S::one() / (S::one() + exp(-x))
}

/// Sigmoid and its derivative.
pub fn sigmoid_with_jac<S: Real>(x: S) -> (S, S) {
    let s = sigmoid(x);
    (s, s * (S::one() - s))
}

/// Hyperbolic tangent.
pub fn tanh<S: Real>(x: S) -> S {
    crate::math::tanh(x)
}

/// Hyperbolic tangent and its derivative.
pub fn tanh_with_jac<S: Real>(x: S) -> (S, S) {
    let t = tanh(x);
    (t, S::one() - t * t)
}

/// Rectified linear unit.
pub fn relu<S: Real>(x: S) -> S {
    if x > S::zero() { x } else { S::zero() }
}

/// Rectified linear unit and its derivative.
pub fn relu_with_jac<S: Real>(x: S) -> (S, S) {
    if x > S::zero() {
        (x, S::one())
    } else {
        (S::zero(), S::zero())
    }
}

/// Leaky rectified linear unit.
pub fn leaky_relu<S: Real>(x: S, a: S) -> S {
    if x > S::zero() { x } else { a * x }
}

/// Leaky rectified linear unit and its derivative.
pub fn leaky_relu_with_jac<S: Real>(x: S, a: S) -> (S, S) {
    if x > S::zero() {
        (x, S::one())
    } else {
        (a * x, a)
    }
}

macro_rules! vector_activation {
    ($name:ident, $scalar:ident) => {
        /// Element-wise activation over a vector.
        pub fn $name<S: Real>(x: &[S]) -> Vec<S> {
            x.iter().map(|v| $scalar(*v)).collect()
        }
    };
    ($name:ident, $scalar:ident, jac) => {
        /// Element-wise activation and its diagonal Jacobian.
        pub fn $name<S: Real>(x: &[S]) -> (Vec<S>, Vec<S>) {
            let mut values = Vec::with_capacity(x.len());
            let mut jac = Vec::with_capacity(x.len());
            for v in x {
                let (f, d) = $scalar(*v);
                values.push(f);
                jac.push(d);
            }
            (values, jac)
        }
    };
    ($name:ident, $scalar:ident, jac, $extra:ident) => {
        /// Element-wise activation with an extra parameter, and its diagonal Jacobian.
        pub fn $name<S: Real>(x: &[S], a: S) -> (Vec<S>, Vec<S>) {
            let mut values = Vec::with_capacity(x.len());
            let mut jac = Vec::with_capacity(x.len());
            for v in x {
                let (f, d) = $scalar(*v, a);
                values.push(f);
                jac.push(d);
            }
            (values, jac)
        }
    };
}

vector_activation!(sigmoid_vec, sigmoid);
vector_activation!(sigmoid_vec_with_jac, sigmoid_with_jac, jac);
vector_activation!(tanh_vec, tanh);
vector_activation!(tanh_vec_with_jac, tanh_with_jac, jac);
vector_activation!(relu_vec, relu);
vector_activation!(relu_vec_with_jac, relu_with_jac, jac);
/// Element-wise leaky ReLU with slope `a`.
pub fn leaky_relu_vec<S: Real>(x: &[S], a: S) -> Vec<S> {
    x.iter().map(|v| leaky_relu(*v, a)).collect()
}

vector_activation!(leaky_relu_vec_with_jac, leaky_relu_with_jac, jac, a);

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn scalar_values_and_derivatives() {
        assert_relative_eq!(sigmoid(0.0_f64), 0.5);
        let (s, d) = sigmoid_with_jac(0.0_f64);
        assert_relative_eq!(s, 0.5);
        assert_relative_eq!(d, 0.25);

        assert_relative_eq!(tanh(0.0_f64), 0.0);
        assert_relative_eq!(tanh_with_jac(0.0_f64).1, 1.0);

        assert_eq!(relu(2.0_f64), 2.0);
        assert_eq!(relu(-2.0_f64), 0.0);
        assert_eq!(relu_with_jac(-2.0_f64), (0.0, 0.0));

        assert_eq!(leaky_relu(-2.0_f64, 0.1), -0.2);
        assert_eq!(leaky_relu_with_jac(-2.0, 0.1), (-0.2, 0.1));
    }

    #[test]
    fn vector_versions() {
        let x = [0.0_f64, 1.0];
        let (v, j) = sigmoid_vec_with_jac(&x);
        assert_relative_eq!(v[0], 0.5);
        assert_relative_eq!(j[0], 0.25);
        assert_eq!(relu_vec(&[-1.0_f64, 2.0]), vec![0.0, 2.0]);
    }
}
