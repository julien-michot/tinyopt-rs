// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

//! Loss functions and norms, mirroring `tinyopt::losses`.

pub mod activations;
pub mod classif;
pub mod mahalanobis;
pub mod norms;
pub mod robust_norms;

pub use activations::*;
pub use classif::*;
pub use mahalanobis::*;
pub use norms::*;
pub use robust_norms::*;
