// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Scalar abstraction shared by the dense backends.

use num_traits::Float;

/// Real scalar types supported by the dense backends.
///
/// Implemented for `f32` and `f64`. Kept deliberately close to
/// [`num_traits::Float`] so that user code can be written once and
/// instantiated for both precisions.
///
/// The active backend's scalar trait is included as a supertrait so that
/// generic code can call backend factorizations without extra bounds.
#[cfg(feature = "nalgebra")]
pub trait Real:
    Float + num_traits::FromPrimitive + core::fmt::Display + Send + Sync + 'static + nalgebra::RealField
{
}

#[cfg(all(feature = "faer", not(feature = "nalgebra")))]
pub trait Real:
    Float
    + num_traits::FromPrimitive
    + core::fmt::Display
    + Send
    + Sync
    + 'static
    + faer::traits::ComplexField
{
}

impl Real for f32 {}
impl Real for f64 {}
