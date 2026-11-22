// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Iterative optimizers.

pub mod bfgs;
pub mod cg;
pub mod dl;
pub mod fixed;
pub mod gd;
pub mod gn;
pub mod lbfgs;
pub mod lm;
#[allow(clippy::module_inception)]
pub mod optimizer;
pub mod trcg;

pub use fixed::FixedSet;
pub use optimizer::AccumulationFn;

/// Generate the common fixed-parameter API for an optimizer with a `fixed` field.
macro_rules! impl_fixed_api {
    () => {
        /// Hold the given parameter indices fixed.
        pub fn fix(&mut self, indices: &[usize]) -> &mut Self {
            self.fixed.fix(indices);
            self
        }

        /// Release all fixed parameters.
        pub fn clear_fixed(&mut self) {
            self.fixed.clear();
        }

        /// Whether any parameter is fixed.
        pub fn has_fixed(&self) -> bool {
            self.fixed.has_fixed()
        }

        /// Number of fixed parameters.
        pub fn num_fixed(&self) -> usize {
            self.fixed.len()
        }

        /// Whether `index` is fixed.
        pub fn is_fixed(&self, index: usize) -> bool {
            self.fixed.contains(index)
        }

        /// Fixed parameter indices.
        pub fn fixed_indices(&self) -> &[usize] {
            self.fixed.indices()
        }
    };
}
pub(crate) use impl_fixed_api;
