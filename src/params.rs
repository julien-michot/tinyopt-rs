// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Parameter types and the [`Params`] trait.
//!
//! Mirrors tinyopt's `traits::params_trait`: a parameter type exposes its
//! scalar, its (static or dynamic) dimension, a manifold-aware `plus_eq`
//! update, and an optional set of fixed indices.

use crate::backend::{Dim, Dynamic, Static};
use crate::real::Real;

/// A type that can be optimized by tinyopt-rs.
///
/// Implement this trait to optimize custom parameter types (for instance
/// Lie-group elements with a manifold retraction in [`Params::plus_eq`]).
pub trait Params {
    /// Scalar type (`f32` or `f64`).
    type Scalar: Real;
    /// Compile-time dimension when known, [`Dynamic`] otherwise.
    type Dim: Dim;

    /// Runtime number of parameters (scalar degrees of freedom).
    fn dims(&self) -> usize;

    /// Flat view of the scalar parameters.
    ///
    /// Vector-like parameters return their contiguous storage; scalar
    /// parameters return a one-element slice.
    fn as_slice(&self) -> &[Self::Scalar];

    /// Manifold-aware in-place update `self += delta`.
    ///
    /// For Euclidean parameters this is a plain coefficient-wise addition.
    /// For manifold parameters (rotations, poses, ...) this applies the
    /// retraction from the tangent space.
    fn plus_eq(&mut self, delta: &[Self::Scalar]);

    /// Squared norm used by the relative step-size stopping criterion.
    fn norm2(&self) -> Self::Scalar;

    /// Indices of parameters held fixed during optimization, if any.
    fn fixed(&self) -> Option<Vec<usize>> {
        None
    }
}

/// Thin by-reference adapter around a parameter type, mirroring C++
/// `ParamsWrapper`.
///
/// Useful to keep external storage (for instance a manifold element) at the
/// call site while exposing the `Params` contract to the optimizer.
#[derive(Clone, Debug, Default)]
pub struct ParamsWrapper<T> {
    /// Wrapped parameter value.
    pub value: T,
}

impl<T> ParamsWrapper<T> {
    /// Wrap a value.
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> core::ops::Deref for ParamsWrapper<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> core::ops::DerefMut for ParamsWrapper<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T> Params for ParamsWrapper<T>
where
    T: Params,
    T::Scalar: Real,
{
    type Scalar = T::Scalar;
    type Dim = T::Dim;

    fn dims(&self) -> usize {
        self.value.dims()
    }

    fn as_slice(&self) -> &[T::Scalar] {
        self.value.as_slice()
    }

    fn plus_eq(&mut self, delta: &[T::Scalar]) {
        self.value.plus_eq(delta);
    }

    fn norm2(&self) -> T::Scalar {
        self.value.norm2()
    }

    fn fixed(&self) -> Option<Vec<usize>> {
        self.value.fixed()
    }
}

#[cfg(feature = "nalgebra")]
mod nalgebra_params {
    use super::*;
    use crate::backend::DenseVector;
    use nalgebra::{DVector, SVector};

    impl<S: Real, const N: usize> Params for SVector<S, N> {
        type Scalar = S;
        type Dim = Static<N>;

        fn dims(&self) -> usize {
            N
        }

        fn as_slice(&self) -> &[S] {
            DenseVector::as_slice(self)
        }

        fn plus_eq(&mut self, delta: &[S]) {
            assert_eq!(delta.len(), N, "parameter dimension mismatch");
            for (p, d) in self.iter_mut().zip(delta) {
                *p += *d;
            }
        }

        fn norm2(&self) -> S {
            self.iter().fold(S::zero(), |acc, v| acc + *v * *v)
        }
    }

    impl<S: Real> Params for DVector<S> {
        type Scalar = S;
        type Dim = Dynamic;

        fn dims(&self) -> usize {
            DenseVector::len(self)
        }

        fn as_slice(&self) -> &[S] {
            DenseVector::as_slice(self)
        }

        fn plus_eq(&mut self, delta: &[S]) {
            assert_eq!(delta.len(), self.len(), "parameter dimension mismatch");
            for (p, d) in self.iter_mut().zip(delta) {
                *p += *d;
            }
        }

        fn norm2(&self) -> S {
            self.iter().fold(S::zero(), |acc, v| acc + *v * *v)
        }
    }
}

#[cfg(feature = "faer")]
mod faer_params {
    use super::*;
    use crate::backend::DenseVector;
    use faer::Col;

    impl<S: Real> Params for Col<S> {
        type Scalar = S;
        type Dim = Dynamic;

        fn dims(&self) -> usize {
            DenseVector::len(self)
        }

        fn as_slice(&self) -> &[S] {
            DenseVector::as_slice(self)
        }

        fn plus_eq(&mut self, delta: &[S]) {
            assert_eq!(delta.len(), self.len(), "parameter dimension mismatch");
            for (p, d) in self.as_mut_slice().iter_mut().zip(delta) {
                *p += *d;
            }
        }

        fn norm2(&self) -> S {
            DenseVector::as_slice(self)
                .iter()
                .fold(S::zero(), |acc, v| acc + *v * *v)
        }
    }
}

impl<S: Real> Params for Vec<S> {
    type Scalar = S;
    type Dim = Dynamic;

    fn dims(&self) -> usize {
        self.len()
    }

    fn as_slice(&self) -> &[S] {
        self
    }

    fn plus_eq(&mut self, delta: &[S]) {
        assert_eq!(delta.len(), self.len(), "parameter dimension mismatch");
        for (p, d) in self.iter_mut().zip(delta) {
            *p += *d;
        }
    }

    fn norm2(&self) -> S {
        self.iter().fold(S::zero(), |acc, v| acc + *v * *v)
    }
}

impl<S: Real, const N: usize> Params for [S; N] {
    type Scalar = S;
    type Dim = Static<N>;

    fn dims(&self) -> usize {
        N
    }

    fn as_slice(&self) -> &[S] {
        self
    }

    fn plus_eq(&mut self, delta: &[S]) {
        assert_eq!(delta.len(), N, "parameter dimension mismatch");
        for (p, d) in self.iter_mut().zip(delta) {
            *p += *d;
        }
    }

    fn norm2(&self) -> S {
        self.iter().fold(S::zero(), |acc, v| acc + *v * *v)
    }
}

impl Params for f64 {
    type Scalar = f64;
    type Dim = Static<1>;

    fn dims(&self) -> usize {
        1
    }

    fn as_slice(&self) -> &[f64] {
        std::slice::from_ref(self)
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        *self += delta[0];
    }

    fn norm2(&self) -> f64 {
        *self * *self
    }
}

impl Params for f32 {
    type Scalar = f32;
    type Dim = Static<1>;

    fn dims(&self) -> usize {
        1
    }

    fn as_slice(&self) -> &[f32] {
        std::slice::from_ref(self)
    }

    fn plus_eq(&mut self, delta: &[f32]) {
        *self += delta[0];
    }

    fn norm2(&self) -> f32 {
        *self * *self
    }
}
