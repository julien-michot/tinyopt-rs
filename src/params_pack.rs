// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Packing several parameters into one optimization problem, mirroring
//! tinyopt's `ParamsPack` and the variadic `Optimize(x, y, ...)` overloads.
//!
//! The pack owns copies of the individual parameters and keeps a flat buffer
//! in sync, because the accumulation API is slice-based. `optimize2`/`3`/`4`
//! copy the result back into the caller's parameters once at the end.

use crate::backend::{Dim, Dynamic};
use crate::params::Params;
use crate::real::Real;

/// A group of parameters optimized together.
#[derive(Clone, Debug)]
pub struct ParamsPack<T> {
    /// The individual parameters.
    pub values: T,
    buffer: Vec<f64>,
}

impl<T: PackValues> ParamsPack<T> {
    /// Create a pack and initialize its flat buffer.
    pub fn new(values: T) -> Self {
        let mut pack = Self {
            values,
            buffer: Vec::new(),
        };
        pack.sync();
        pack
    }

    /// Flat parameter buffer.
    pub fn buffer(&self) -> &[f64] {
        &self.buffer
    }

    fn sync(&mut self) {
        self.buffer.clear();
        self.values.write_into(&mut self.buffer);
    }
}

impl<T: PackValues> Params for ParamsPack<T> {
    type Scalar = f64;
    type Dim = Dynamic;

    fn dims(&self) -> usize {
        self.buffer.len()
    }

    fn as_slice(&self) -> &[f64] {
        &self.buffer
    }

    fn plus_eq(&mut self, delta: &[f64]) {
        self.values.plus_eq(delta);
        self.sync();
    }

    fn norm2(&self) -> f64 {
        self.values.norm2()
    }

    fn fixed(&self) -> Option<Vec<usize>> {
        self.values.fixed()
    }
}

/// Trait implemented by tuples of parameters.
pub trait PackValues {
    /// Total number of scalar parameters.
    fn dims(&self) -> usize;
    /// Append all parameters to `out`.
    fn write_into(&self, out: &mut Vec<f64>);
    /// Apply a flat update to every element.
    fn plus_eq(&mut self, delta: &[f64]);
    /// Sum of the elements' squared norms.
    fn norm2(&self) -> f64;
    /// Fixed indices, offset by element position.
    fn fixed(&self) -> Option<Vec<usize>>;
}

macro_rules! impl_pack {
    ($(($T:ident, $v:ident)),+) => {
        impl<$($T: Params<Scalar = f64>),+> PackValues for ($($T,)+) {
            fn dims(&self) -> usize {
                let ($($v,)+) = self;
                0 $(+ $v.dims())+
            }

            fn write_into(&self, out: &mut Vec<f64>) {
                let ($($v,)+) = self;
                $(out.extend_from_slice($v.as_slice());)+
            }

            fn plus_eq(&mut self, delta: &[f64]) {
                let ($($v,)+) = self;
                let mut offset = 0;
                $(
                    let dims = $v.dims();
                    $v.plus_eq(&delta[offset..offset + dims]);
                    offset += dims;
                )+
                let _ = offset;
            }

            fn norm2(&self) -> f64 {
                let ($($v,)+) = self;
                0.0 $(+ $v.norm2())+
            }

            fn fixed(&self) -> Option<Vec<usize>> {
                let ($($v,)+) = self;
                let mut out = Vec::new();
                let mut offset = 0;
                $(
                    if let Some(idx) = $v.fixed() {
                        out.extend(idx.into_iter().map(|i| i + offset));
                    }
                    offset += $v.dims();
                )+
                if out.is_empty() { None } else { Some(out) }
            }
        }
    };
}

impl_pack!((A, a), (B, b));
impl_pack!((A, a), (B, b), (C, c));
impl_pack!((A, a), (B, b), (C, c), (D, d));

/// Compile-time dimension helper used by tests.
pub fn pack_dim<D: Dim>() -> Option<usize> {
    D::VALUE
}

/// Scalar conversion helper for tests.
pub fn same_scalar<S: Real>(v: S) -> S {
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Static;
    use crate::backend::{DenseVector, Vector};

    fn vec2(x: f64, y: f64) -> Vector<f64, Static<2>> {
        let mut v = <Vector<f64, Static<2>> as DenseVector<f64>>::zeros(2);
        v.set_at(0, x);
        v.set_at(1, y);
        v
    }

    #[test]
    fn pack_dimensions_and_updates() {
        let mut pack = ParamsPack::new((vec2(1.0, 2.0), 3.0_f64));
        assert_eq!(Params::dims(&pack), 3);
        assert_eq!(pack.as_slice(), &[1.0, 2.0, 3.0]);

        Params::plus_eq(&mut pack, &[0.5, -0.5, 1.0]);
        assert_eq!(pack.as_slice(), &[1.5, 1.5, 4.0]);
        assert_eq!(pack.values.0.at(0), 1.5);
        assert_eq!(pack.values.1, 4.0);
    }

    #[test]
    fn pack_norm_and_scalar() {
        let pack = ParamsPack::new((vec2(1.0, 2.0), 3.0_f64));
        assert_eq!(Params::norm2(&pack), 1.0 + 4.0 + 9.0);
        assert_eq!(pack_dim::<Static<3>>(), Some(3));
        assert_eq!(same_scalar(2.5_f64), 2.5);
    }
}
